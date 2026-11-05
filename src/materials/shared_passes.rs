//! Full-screen passes written once for both backends: the
//! [`passes!`](crate::passes) macro.
//!
//! [`shader!`](crate::shader) is a material's way in. This is the other kind
//! of shader a renderer is full of — a family of full-screen passes over a
//! few textures and one uniform block, the way a fluid solver, a blur chain
//! or a post effect is written — in the same syntax (WGSL in Rust tokens,
//! see [`shared_shader`](super::shared_shader)), emitted as a WGSL module for
//! wgpu and as MSL for the native Metal backend.
//!
//! ```
//! let blur = threers::passes! {
//!     struct BlurParams {
//!         texel_size: vec2f,
//!         weight: f32,
//!         pad: f32,
//!     }
//!     fn soften(i: Frag) -> vec4f {
//!         let side = (tex_a(i.vl) + tex_a(i.vr)) * 0.5;
//!         return mix(tex_a(i.uv), side, u.weight);
//!     }
//! };
//! let wgsl = blur.wgsl("BlurParams").unwrap();
//! assert!(wgsl.contains("fn fs_soften("));
//! let msl = blur.msl("BlurParams", "blur_").unwrap();
//! assert!(msl.contains("fragment float4 blur_fs_soften("));
//! ```
//!
//! # The interface
//!
//! - The uniform block is one of the source's structs, named to
//!   [`wgsl`](PassShader::wgsl) and [`msl`](PassShader::msl), and read as
//!   `u`. A `texel_size: vec2f` in it places the vertex stages' neighbour
//!   taps; without one they sit on the texel itself. Keep `vec3f` out of it:
//!   the two languages lay a three-vector out differently.
//! - `tex_a(uv) -> vec4f`, `tex_b`, `tex_c`: the pass's three textures,
//!   sampled linearly and clamped to the edge; `load_a(texel: vec2i)`, …,
//!   the texel itself, unfiltered; `size_a() -> vec2u`, …, their sizes.
//! - `batch(k: u32) -> vec4f`: a uniform array of up to 64 `vec4f`s.
//! - A pass is any `fn name(i: Frag) -> vec4f`. `Frag` carries `pos` (the
//!   fragment's position in pixels), `uv` (origin at the top left) and the
//!   four neighbours `vl`, `vr`, `vt`, `vb`.
//!
//! Every pass becomes an entry point, `fs_name` in WGSL and
//! `{prefix}fs_name` in MSL, with two vertex stages: `vs`, whose neighbours
//! are one `texel_size` away, and `vs_blur`, whose are 1.333 — the offset at
//! which two bilinear fetches and the centre make a five-tap kernel.
//!
//! # Bindings
//!
//! | | wgpu (group 0) | Metal |
//! |---|---|---|
//! | `u` | binding 0 | vertex and fragment buffer 0 |
//! | `tex_a`, `tex_b`, `tex_c` | bindings 1–3 | fragment textures 0–2 |
//! | the sampler | binding 4 | in the shader |
//! | `batch` | binding 5 | fragment buffer 1, in the passes that use it |
//!
//! Structs and constants are global in MSL, beside whatever else the library
//! holds: give them names of the module's own.

use super::shared_shader::{parse_items, FnItem, Item};

/// Functions written once for both backends — what
/// [`shader_functions!`](crate::shader_functions) returns: the source's
/// structs, constants and functions as WGSL, and as MSL with the functions
/// `static inline`, for a library of helpers both backends' shaders call.
/// The same syntax as [`passes!`](crate::passes), without the pass
/// interface: no `u`, no textures.
#[derive(Clone, Debug)]
pub struct ShaderFunctions {
    source: String,
}

/// Write shader functions once for wgpu and the native Metal backend. See
/// [`ShaderFunctions`].
#[macro_export]
macro_rules! shader_functions {
    ($($body:tt)*) => {
        $crate::materials::shared_passes::ShaderFunctions::new(stringify!($($body)*))
    };
}

impl ShaderFunctions {
    /// Wrap shared source. Prefer the [`shader_functions!`](crate::shader_functions) macro.
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
        }
    }

    fn items(&self) -> Result<Vec<Item>, String> {
        let items =
            parse_items(&self.source).map_err(|e| e.replace("shader!", "shader_functions!"))?;
        check_names(&items, "shader_functions!")?;
        Ok(items)
    }

    /// The items as WGSL, in source order.
    pub fn wgsl(&self) -> Result<String, String> {
        Ok(self.items()?.iter().map(|it| it.to_wgsl() + "\n").collect())
    }

    /// The items as MSL, in source order: structs, `constant`s, and
    /// `static inline` functions.
    pub fn msl(&self) -> Result<String, String> {
        Ok(self
            .items()?
            .iter()
            .map(|it| match it {
                Item::Fn(_) => format!("static inline {}\n", it.to_msl().trim_start()),
                _ => it.to_msl() + "\n",
            })
            .collect())
    }
}

/// Full-screen passes written once for both backends — what
/// [`passes!`](crate::passes) returns. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct PassShader {
    source: String,
}

/// Write full-screen passes once for wgpu and the native Metal backend. See
/// [`shared_passes`](crate::materials::shared_passes).
#[macro_export]
macro_rules! passes {
    ($($body:tt)*) => {
        $crate::materials::shared_passes::PassShader::new(stringify!($($body)*))
    };
}

impl PassShader {
    /// Wrap shared source. Prefer the [`passes!`](crate::passes) macro.
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
        }
    }

    /// The passes' names, in source order.
    pub fn passes(&self) -> Result<Vec<String>, String> {
        Ok(self
            .parsed(None)?
            .1
            .iter()
            .map(|f| f.name.clone())
            .collect())
    }

    /// The WGSL module: the varyings, the bindings and accessors, the
    /// source's items, the two vertex stages and an `@fragment fn
    /// fs_{name}` for every pass. `uniforms` names the struct bound as `u`.
    pub fn wgsl(&self, uniforms: &str) -> Result<String, String> {
        let (items, passes) = self.parsed(Some(uniforms))?;
        let offset = if has_texel_size(&items, uniforms) {
            "u.texel_size"
        } else {
            "vec2<f32>(0.0)"
        };
        let mut out = String::from(
            "struct VsOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) uv : vec2<f32>,
    @location(1) vl : vec2<f32>,
    @location(2) vr : vec2<f32>,
    @location(3) vt : vec2<f32>,
    @location(4) vb : vec2<f32>,
};
alias Frag = VsOut;
",
        );
        // The structs first: the bindings name the uniform block.
        for item in items.iter().filter(|it| matches!(it, Item::Struct(_))) {
            out.push_str(&item.to_wgsl());
            out.push('\n');
        }
        out.push_str(&format!(
            "@group(0) @binding(0) var<uniform> u : {uniforms};
@group(0) @binding(1) var pass_tex_a : texture_2d<f32>;
@group(0) @binding(2) var pass_tex_b : texture_2d<f32>;
@group(0) @binding(3) var pass_tex_c : texture_2d<f32>;
@group(0) @binding(4) var pass_samp : sampler;
@group(0) @binding(5) var<uniform> pass_batch : array<vec4<f32>, 64>;
fn tex_a(p : vec2<f32>) -> vec4<f32> {{ return textureSample(pass_tex_a, pass_samp, p); }}
fn tex_b(p : vec2<f32>) -> vec4<f32> {{ return textureSample(pass_tex_b, pass_samp, p); }}
fn tex_c(p : vec2<f32>) -> vec4<f32> {{ return textureSample(pass_tex_c, pass_samp, p); }}
fn load_a(c : vec2<i32>) -> vec4<f32> {{ return textureLoad(pass_tex_a, c, 0); }}
fn load_b(c : vec2<i32>) -> vec4<f32> {{ return textureLoad(pass_tex_b, c, 0); }}
fn load_c(c : vec2<i32>) -> vec4<f32> {{ return textureLoad(pass_tex_c, c, 0); }}
fn size_a() -> vec2<u32> {{ return textureDimensions(pass_tex_a); }}
fn size_b() -> vec2<u32> {{ return textureDimensions(pass_tex_b); }}
fn size_c() -> vec2<u32> {{ return textureDimensions(pass_tex_c); }}
fn batch(k : u32) -> vec4<f32> {{ return pass_batch[min(k, 63u)]; }}
fn pass_vertex(vid : u32, offset : vec2<f32>, blur : bool) -> VsOut {{
    // One triangle twice the viewport; uv is the raw corner, (0, 0) at the
    // top left — the texture's own origin. The blur's taps share one axis,
    // which `offset` carries.
    let c = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    var o : VsOut;
    o.pos = vec4<f32>(c.x * 2.0 - 1.0, 1.0 - c.y * 2.0, 0.0, 1.0);
    o.uv = c;
    o.vl = select(c - vec2<f32>(offset.x, 0.0), c - offset, blur);
    o.vr = select(c + vec2<f32>(offset.x, 0.0), c + offset, blur);
    o.vt = select(c + vec2<f32>(0.0, offset.y), c, blur);
    o.vb = select(c - vec2<f32>(0.0, offset.y), c, blur);
    return o;
}}
@vertex
fn vs(@builtin(vertex_index) vid : u32) -> VsOut {{ return pass_vertex(vid, {offset}, false); }}
@vertex
fn vs_blur(@builtin(vertex_index) vid : u32) -> VsOut {{ return pass_vertex(vid, {offset} * 1.33333333, true); }}
"
        ));
        for item in items.iter().filter(|it| !matches!(it, Item::Struct(_))) {
            out.push_str(&item.to_wgsl());
            out.push('\n');
        }
        for f in &passes {
            out.push_str(&format!(
                "@fragment\nfn fs_{0}(i : VsOut) -> @location(0) vec4<f32> {{ return {0}(i); }}\n",
                f.name
            ));
        }
        Ok(out)
    }

    /// The MSL: the varyings, the source's structs and constants, a context
    /// struct holding the uniforms and textures with the source's functions
    /// as its members, the vertex stages `{prefix}vs` and `{prefix}vs_blur`,
    /// and a `fragment float4 {prefix}fs_{name}` for every pass.
    pub fn msl(&self, uniforms: &str, prefix: &str) -> Result<String, String> {
        let (items, passes) = self.parsed(Some(uniforms))?;
        let (offset, vertex_u) = if has_texel_size(&items, uniforms) {
            (
                "u.texel_size",
                format!(", constant {uniforms} &u [[buffer(0)]]"),
            )
        } else {
            ("float2(0.0)", String::new())
        };
        let functions: Vec<&FnItem> = items
            .iter()
            .filter_map(|it| if let Item::Fn(f) = it { Some(f) } else { None })
            .collect();
        let batched = calls_batch(&functions);
        let vertex = format!("{prefix}PassVertex");
        let ctx = format!("{prefix}PassContext");
        let sampler = format!("{prefix}pass_sampler");
        let mut out = format!(
            "struct {vertex} {{
    float4 pos [[position]];
    float2 uv;
    float2 vl;
    float2 vr;
    float2 vt;
    float2 vb;
}};
constexpr sampler {sampler}(coord::normalized, address::clamp_to_edge, filter::linear);
"
        );
        for item in items.iter().filter(|it| !matches!(it, Item::Fn(_))) {
            out.push_str(&item.to_msl());
            out.push('\n');
        }
        out.push_str(&format!(
            "struct {ctx} {{
    typedef {vertex} Frag;
    constant {uniforms} &u;
    texture2d<float> pass_tex_a;
    texture2d<float> pass_tex_b;
    texture2d<float> pass_tex_c;
    constant float4 *pass_batch;
    float4 tex_a(float2 p) const {{ return pass_tex_a.sample({sampler}, p); }}
    float4 tex_b(float2 p) const {{ return pass_tex_b.sample({sampler}, p); }}
    float4 tex_c(float2 p) const {{ return pass_tex_c.sample({sampler}, p); }}
    float4 load_a(int2 c) const {{ return pass_tex_a.read(uint2(c)); }}
    float4 load_b(int2 c) const {{ return pass_tex_b.read(uint2(c)); }}
    float4 load_c(int2 c) const {{ return pass_tex_c.read(uint2(c)); }}
    uint2 size_a() const {{ return uint2(pass_tex_a.get_width(), pass_tex_a.get_height()); }}
    uint2 size_b() const {{ return uint2(pass_tex_b.get_width(), pass_tex_b.get_height()); }}
    uint2 size_c() const {{ return uint2(pass_tex_c.get_width(), pass_tex_c.get_height()); }}
    float4 batch(uint k) const {{ return pass_batch[min(k, 63u)]; }}
"
        ));
        for f in &functions {
            out.push_str(&Item::Fn((*f).clone()).to_msl());
        }
        out.push_str("};\n");
        out.push_str(&format!(
            "static inline {vertex} {prefix}pass_vertex(uint vid, float2 offset, bool blur) {{
    const float2 c = float2(float((vid << 1u) & 2u), float(vid & 2u));
    {vertex} o;
    o.pos = float4(c.x * 2.0 - 1.0, 1.0 - c.y * 2.0, 0.0, 1.0);
    o.uv = c;
    o.vl = blur ? c - offset : c - float2(offset.x, 0.0);
    o.vr = blur ? c + offset : c + float2(offset.x, 0.0);
    o.vt = blur ? c : c + float2(0.0, offset.y);
    o.vb = blur ? c : c - float2(0.0, offset.y);
    return o;
}}
vertex {vertex} {prefix}vs(uint vid [[vertex_id]]{vertex_u}) {{
    return {prefix}pass_vertex(vid, {offset}, false);
}}
vertex {vertex} {prefix}vs_blur(uint vid [[vertex_id]]{vertex_u}) {{
    return {prefix}pass_vertex(vid, {offset} * 1.33333333, true);
}}
"
        ));
        for f in &passes {
            let (param, arg) = if batched.contains(&f.name) {
                (", constant float4 *batch [[buffer(1)]]", "batch")
            } else {
                ("", "nullptr")
            };
            out.push_str(&format!(
                "fragment float4 {prefix}fs_{name}({vertex} i [[stage_in]], constant {uniforms} &u [[buffer(0)]]{param},
        texture2d<float> a [[texture(0)]], texture2d<float> b [[texture(1)]], texture2d<float> c [[texture(2)]]) {{
    {ctx} ctx = {{ u, a, b, c, {arg} }};
    return ctx.{name}(i);
}}
",
                name = f.name
            ));
        }
        Ok(out)
    }

    /// The items, and the passes among them; `uniforms`, when given, must be
    /// one of the structs.
    fn parsed(&self, uniforms: Option<&str>) -> Result<(Vec<Item>, Vec<FnItem>), String> {
        let items = parse_items(&self.source).map_err(|e| e.replace("shader!", "passes!"))?;
        check_names(&items, "passes!")?;
        if let Some(name) = uniforms {
            if !items
                .iter()
                .any(|it| matches!(it, Item::Struct(st) if st.name == name))
            {
                return Err(format!("passes!: no `struct {name}` for the uniform block"));
            }
        }
        let passes: Vec<FnItem> = items
            .iter()
            .filter_map(|it| match it {
                Item::Fn(f) if is_pass(f) => Some(f.clone()),
                _ => None,
            })
            .collect();
        if passes.is_empty() {
            return Err("passes!: no `fn name(i: Frag) -> vec4f` pass".into());
        }
        Ok((items, passes))
    }
}

/// Names Rust takes but WGSL or MSL reserves, which would otherwise fail
/// only when the other backend compiles the shader: `restrict` is a WGSL
/// keyword, `kernel` an MSL one.
const RESERVED: &[&str] = &[
    "auto",
    "case",
    "centroid",
    "char",
    "class",
    "constant",
    "default",
    "delete",
    "device",
    "do",
    "double",
    "enum",
    "explicit",
    "extern",
    "flat",
    "fragment",
    "friend",
    "goto",
    "half",
    "inline",
    "kernel",
    "long",
    "namespace",
    "new",
    "operator",
    "packed",
    "private",
    "protected",
    "public",
    "register",
    "restrict",
    "sample",
    "sampler",
    "short",
    "signed",
    "sizeof",
    "smooth",
    "static",
    "switch",
    "template",
    "texture",
    "this",
    "thread",
    "threadgroup",
    "typedef",
    "typename",
    "uniform",
    "union",
    "unsigned",
    "vertex",
    "virtual",
    "volatile",
];

/// Every function, parameter and local name, against [`RESERVED`].
fn check_names(items: &[Item], macro_name: &str) -> Result<(), String> {
    let bad = |name: &str, what: &str| {
        Err(format!(
            "{macro_name}: `{name}` ({what}) is reserved in WGSL or MSL; rename it"
        ))
    };
    for it in items {
        let Item::Fn(f) = it else { continue };
        if RESERVED.contains(&f.name.as_str()) {
            return bad(&f.name, "a function");
        }
        for (p, _) in &f.params {
            if RESERVED.contains(&p.as_str()) {
                return bad(p, "a parameter");
            }
        }
        for w in f.body.windows(2) {
            if w[0].is("let") || w[0].is("var") {
                if let Some(name) = RESERVED.iter().find(|r| w[1].is(r)) {
                    return bad(name, "a local");
                }
            }
        }
    }
    Ok(())
}

/// Whether the uniform block has a `texel_size`, for the vertex stages.
fn has_texel_size(items: &[Item], uniforms: &str) -> bool {
    items.iter().any(|it| {
        matches!(it, Item::Struct(st) if st.name == uniforms && st.fields.iter().any(|(n, _)| n == "texel_size"))
    })
}

/// `fn name(i: Frag) -> vec4f`.
fn is_pass(f: &FnItem) -> bool {
    f.params.len() == 1
        && f.params[0].1.len() == 1
        && f.params[0].1[0].is("Frag")
        && f.ret
            .as_ref()
            .is_some_and(|r| r.len() == 1 && r[0].is("vec4f"))
}

/// The functions that read `batch`, directly or through another.
fn calls_batch(functions: &[&FnItem]) -> Vec<String> {
    let calls = |f: &FnItem, name: &str| f.body.windows(2).any(|w| w[0].is(name) && w[1].is("("));
    let mut found: Vec<String> = Vec::new();
    loop {
        let before = found.len();
        for f in functions {
            if !found.contains(&f.name) && (calls(f, "batch") || found.iter().any(|g| calls(f, g)))
            {
                found.push(f.name.clone());
            }
        }
        if found.len() == before {
            return found;
        }
    }
}

/// For tests: `source` validates, and each of `entries` translates to GLSL
/// ES 3.00 as WebGL2 needs it. Every `PassShader` with a WebGL2 path holds
/// itself to this.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn assert_webgl2(source: &str, entries: &[(naga::ShaderStage, String)]) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(
        entries.len(),
        module.entry_points.len(),
        "an entry point not checked, or one that is not there"
    );
    for (stage, entry) in entries {
        let options = naga::back::glsl::Options {
            version: naga::back::glsl::Version::Embedded {
                version: 300,
                is_webgl: true,
            },
            ..Default::default()
        };
        let pipeline = naga::back::glsl::PipelineOptions {
            shader_stage: *stage,
            entry_point: entry.clone(),
            multiview: None,
        };
        let mut out = String::new();
        naga::back::glsl::Writer::new(
            &mut out,
            &module,
            &info,
            &options,
            &pipeline,
            Default::default(),
        )
        .and_then(|mut w| w.write().map(|_| ()))
        .unwrap_or_else(|e| panic!("{entry}: {e}"));
    }
}

/// Every entry point of `shader`'s WGSL: the two vertex stages and each pass.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn entry_points(shader: &PassShader) -> Vec<(naga::ShaderStage, String)> {
    let mut out = vec![
        (naga::ShaderStage::Vertex, "vs".to_string()),
        (naga::ShaderStage::Vertex, "vs_blur".into()),
    ];
    out.extend(
        shader
            .passes()
            .unwrap()
            .into_iter()
            .map(|p| (naga::ShaderStage::Fragment, format!("fs_{p}"))),
    );
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_pass_reaches_both_languages() {
        let p = crate::passes! {
            struct TestParams {
                texel_size: vec2f,
                gain: f32,
                pad: f32,
            }
            const SCALE: f32 = 0.5;
            fn halved(c: vec4f) -> vec4f { return c * SCALE; }
            fn darker(i: Frag) -> vec4f { return halved(tex_a(i.uv)) * u.gain; }
            fn summed(i: Frag) -> vec4f {
                var acc = vec4f(0.0);
                for (var k = 0u32; k < 2u32; k += 1u32) { acc += batch(k); }
                return acc + tex_b(i.vl);
            }
        };
        assert_eq!(p.passes().unwrap(), ["darker", "summed"]);
        let w = p.wgsl("TestParams").unwrap();
        assert!(w.contains("var<uniform> u : TestParams;"), "{w}");
        assert!(
            w.contains("fn fs_darker(i : VsOut) -> @location(0) vec4<f32> { return darker(i); }"),
            "{w}"
        );
        assert!(w.contains("fn vs_blur("), "{w}");
        let m = p.msl("TestParams", "t_").unwrap();
        assert!(m.contains("fragment float4 t_fs_darker("), "{m}");
        assert!(m.contains("vertex t_PassVertex t_vs("), "{m}");
        // Only the pass that reads the batch takes its buffer.
        let summed = &m[m.find("t_fs_summed(").unwrap()..];
        assert!(
            summed[..summed.find('{').unwrap()].contains("[[buffer(1)]]"),
            "{m}"
        );
        let darker = &m[m.find("t_fs_darker(").unwrap()..];
        assert!(
            !darker[..darker.find('{').unwrap()].contains("[[buffer(1)]]"),
            "{m}"
        );
        assert!(m.contains("constant float SCALE = 0.5;"), "{m}");
    }

    #[test]
    fn the_uniform_block_must_exist() {
        let p = crate::passes! { fn f(i: Frag) -> vec4f { return tex_a(i.uv); } };
        assert!(p.wgsl("Missing").unwrap_err().contains("struct Missing"));
    }

    #[test]
    fn functions_reach_both_languages() {
        let f = crate::shader_functions! {
            const GAIN: f32 = 2.0;
            fn brighter(c: vec3f) -> vec3f { return c * GAIN; }
        };
        let w = f.wgsl().unwrap();
        assert!(
            w.contains("const GAIN: f32 = 2.0;") && w.contains("fn brighter(c: vec3f) -> vec3f {"),
            "{w}"
        );
        let m = f.msl().unwrap();
        assert!(
            m.contains("constant float GAIN = 2.0;")
                && m.contains("static inline float3 brighter(float3 c) {"),
            "{m}"
        );
    }

    #[test]
    fn a_reserved_name_is_an_error() {
        let f = crate::shader_functions! {
            fn widen(x: f32) -> f32 { let kernel = x * 2.0; return kernel; }
        };
        assert!(f.msl().unwrap_err().contains("`kernel`"));
        let p = crate::passes! {
            struct P { gain: f32, a: f32, b: f32, c: f32, }
            fn restrict(i: Frag) -> vec4f { return tex_a(i.uv); }
        };
        assert!(p.wgsl("P").unwrap_err().contains("`restrict`"));
    }

    /// Without a `texel_size` the vertex stages do not read the uniforms, so
    /// a pass that fetches texels needs no vertex binding at all.
    #[test]
    fn texel_fetches_and_no_texel_size() {
        let p = crate::passes! {
            struct FetchParams { gain: f32, pad0: f32, pad1: f32, pad2: f32, }
            fn fetch(i: Frag) -> vec4f {
                let dim = vec2i(size_a());
                return load_a(clamp(vec2i(i.pos.xy), vec2i(0), dim - vec2i(1))) * u.gain;
            }
        };
        let w = p.wgsl("FetchParams").unwrap();
        assert!(
            w.contains("return pass_vertex(vid, vec2<f32>(0.0), false); }"),
            "{w}"
        );
        let module = naga::front::wgsl::parse_str(&w)
            .unwrap_or_else(|e| panic!("{}\n{w}", e.emit_to_string(&w)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{e:?}\n{w}"));
        let m = p.msl("FetchParams", "f_").unwrap();
        assert!(
            m.contains("vertex f_PassVertex f_vs(uint vid [[vertex_id]]) {"),
            "{m}"
        );
        assert!(m.contains("pass_tex_a.read(uint2(c))"), "{m}");
    }

    /// The WGSL is a module naga accepts.
    #[test]
    fn the_wgsl_validates() {
        let p = crate::passes! {
            struct TestParams { texel_size: vec2f, gain: f32, pad: f32, }
            fn darker(i: Frag) -> vec4f { return tex_a(i.uv) * u.gain + batch(0u32) + tex_c(i.pos.xy); }
        };
        let w = p.wgsl("TestParams").unwrap();
        let module = naga::front::wgsl::parse_str(&w)
            .unwrap_or_else(|e| panic!("{}\n{w}", e.emit_to_string(&w)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{e:?}\n{w}"));
    }
}
