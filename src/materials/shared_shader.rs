//! One shader source for both backends: the [`shader!`](crate::shader) macro.
//!
//! A [`ShaderMaterial`](super::ShaderMaterial) takes WGSL against the
//! renderer's whole preamble, which the native Metal backend translates when
//! it first draws it. `shader!` is the smaller, checked way in: a syntax in
//! Rust tokens, a `shade` entry point and a fixed set of accessors. It
//! produces WGSL, and MSL by putting that WGSL through the same translation,
//! so the two backends run one shader. An error shows when the material is
//! built, not as a fallback colour at the first draw.
//!
//! ```
//! use threers::{shader, ShaderMaterial};
//!
//! let rim = shader! {
//!     // Linear colour out; the renderer encodes it for the target.
//!     fn shade(s: Shading) -> vec4f {
//!         let base = data(0u32).xyz;
//!         let facing = max(dot(s.normal, s.view_dir), 0.0);
//!         let rim = pow(1.0 - facing, 3.0);
//!         var lit = ambient_light() * base;
//!         for (var i = 0u32; i < light_count(); i += 1u32) {
//!             lit += base * light_color(i) * max(dot(s.normal, light_direction(i)), 0.0);
//!         }
//!         return vec4f(lit + rim * data(1u32).xyz, opacity());
//!     }
//! };
//! let material = ShaderMaterial::from_shared(&rim)
//!     .with_data(vec![[0.8, 0.3, 0.2, 0.0], [0.3, 0.6, 1.0, 0.0]]);
//! assert!(material.msl.is_some());
//! ```
//!
//! # The shared syntax
//!
//! WGSL's, written in Rust tokens so the macro can take it:
//!
//! - `fn name(a: T, b: U) -> R { ... }`; the entry point is
//!   `fn shade(s: Shading) -> vec4f`, returning linear colour and alpha.
//! - `let` (immutable) and `var` (mutable) locals, with or without a type.
//! - `const NAME: T = value;` at the top level.
//! - Scalars `f32`, `u32`, `i32`, `bool`; vectors `vec2f`/`vec3f`/`vec4f`,
//!   `vec2u`…, `vec2i`…; `mat3x3f`, `mat4x4f`. Constructors are the type's
//!   name: `vec3f(1.0)`, `f32(i)`.
//! - Literals carry Rust's suffixes where WGSL would need one: `1u32` for
//!   `1u`, `2i32`, `0.5f32`. Plain `1.0` is an `f32`.
//! - `if`/`else`, `for`, `while`, `loop`/`break`/`continue`, `return`,
//!   `discard`, compound assignment (`+=`; not `++`).
//! - The built-ins both languages share: `abs`, `min`, `max`, `clamp`,
//!   `saturate`, `mix`, `step`, `smoothstep`, `sign`, `floor`, `ceil`,
//!   `round`, `fract`, `sqrt`, `inverseSqrt`, `pow`, `exp`, `exp2`, `log`,
//!   `log2`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `dot`,
//!   `cross`, `length`, `distance`, `normalize`, `reflect`, `refract`,
//!   `faceforward`, `fma`, `select`, `all`, `any`, `dpdx`, `dpdy`, `fwidth`.
//!   `pow` with a negative base is undefined on Metal; square with `x * x`.
//!
//! What the shader reads comes through accessor functions, implemented on
//! both sides:
//!
//! | Accessor | |
//! |---|---|
//! | `s.world_pos`, `s.normal`, `s.view_dir`, `s.uv`, `s.color`, `s.view_depth`, `s.front_facing` | the fragment ([`Shading`](#the-shading-struct)) |
//! | `data(i: u32) -> vec4f` | the material's [`data`](super::ShaderMaterial::data) slots |
//! | `s0(i: u32) -> f32`, `s1`, `s2`; `s0_len() -> u32`, … | its storage arrays |
//! | `camera_position() -> vec3f` | the eye |
//! | `ambient_light() -> vec3f` | ambient light, intensity applied |
//! | `light_count() -> u32`, `light_direction(i) -> vec3f`, `light_color(i) -> vec3f` | directional lights (up to four): direction *towards* the light, colour × intensity |
//! | `directional_shadow(i, s) -> f32` | directional light `i`'s shadow at the fragment: 1 lit, 0 shadowed, filtered as the scene's other shadows are; 1 when the object does not `receive_shadow` |
//! | `point_light_count() -> u32`, `point_light_direction(i, s) -> vec3f`, `point_light_color(i, s) -> vec3f`, `point_shadow(i, s) -> f32` | point lights (up to eight), the colour attenuated by distance as the built-in materials attenuate it |
//! | `hemisphere_light(n: vec3f) -> vec3f` | the hemisphere lights' irradiance on a normal (not included in `ambient_light`) |
//! | `receives_shadow() -> bool` | whether the object receives shadows |
//! | `tex0(uv) -> vec4f` … `tex3`; `tex0_level(uv, lod)`; `tex0_size() -> vec2u` | the material's [`maps`](super::ShaderMaterial::maps), mipmapped, repeating; white where there is no map. `tex0(uv)` takes derivatives, so call it outside non-uniform branches (or use `_level`) |
//! | `opacity() -> f32` | the material's opacity |
//! | `fog(color: vec3f, s: Shading) -> vec3f` | the scene's fog |
//! | `tone_map(color: vec3f) -> vec3f` | the renderer's tone mapping |
//!
//! # The `Shading` struct
//!
//! `world_pos`, `normal` (unit, turned to face the eye on a back face),
//! `view_dir` (unit, surface to eye), `uv`, `color` (vertex colour, white
//! when there is none), `view_depth` (distance to the eye) and
//! `front_facing`.
//!
//! For anything outside this — a custom vertex stage, the preamble's other
//! uniforms — write WGSL in
//! [`ShaderMaterial::fragment`](super::ShaderMaterial::fragment) directly.
//!
//! Full-screen passes — a solver, a blur chain, a post effect — are the
//! other half of what a renderer writes: [`passes!`](crate::passes), in the
//! same syntax, emits a family of them for both backends (see
//! [`shared_passes`](super::shared_passes)).

/// A shader written once for both backends — what [`shader!`](crate::shader)
/// returns. See the [module docs](self).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedShader {
    /// The source as the macro captured it.
    pub source: String,
}

/// Write a shader once for wgpu and the native Metal backend. See
/// [`shared_shader`](crate::materials::shared_shader).
///
/// The body is captured as tokens — so it is checked for balanced brackets
/// and valid literals as it compiles — and translated to WGSL, and that to
/// MSL, when the material is built.
#[macro_export]
macro_rules! shader {
    ($($body:tt)*) => {
        $crate::materials::SharedShader::new(stringify!($($body)*))
    };
}

impl SharedShader {
    /// Wrap shared source. Prefer the [`shader!`](crate::shader) macro.
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
        }
    }

    /// The fragment as a [`ShaderMaterial`](super::ShaderMaterial) takes it:
    /// the accessors, the user's functions, and an `fs_main` that encodes
    /// `shade`'s linear colour for the target.
    pub fn wgsl(&self) -> Result<String, String> {
        let program = parse(&self.source)?;
        let mut out = String::from(WGSL_INTERFACE);
        // Structs and constants are WGSL already; functions are re-emitted.
        for item in &program {
            out.push_str(&item.to_wgsl());
            out.push('\n');
        }
        out.push_str(WGSL_ENTRY);
        Ok(out)
    }

    /// The fragment for the Metal backend: [`wgsl`](Self::wgsl) translated
    /// as any WGSL `ShaderMaterial` is, so the two cannot drift apart.
    pub fn msl(&self) -> Result<String, String> {
        wgsl_material_to_msl(&self.wgsl()?, None)
    }
}

// ---------------------------------------------------------------- interfaces

/// The accessors in WGSL, over the custom-shader preamble's `frame`, `mesh`,
/// `u_data` and `s0..s2`.
const WGSL_INTERFACE: &str = "
struct Shading {
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    uv: vec2<f32>,
    color: vec4<f32>,
    view_depth: f32,
    front_facing: bool,
    view_z: f32,
};
fn data(i: u32) -> vec4<f32> { return u_data.data[min(i, 15u)]; }
fn camera_position() -> vec3<f32> { return frame.camera_position.xyz; }
fn ambient_light() -> vec3<f32> { return frame.ambient.rgb; }
fn light_count() -> u32 { return min(frame.light_counts.x, 4u); }
fn light_direction(i: u32) -> vec3<f32> { return normalize(-frame.dir_lights[min(i, 3u)].direction.xyz); }
fn light_color(i: u32) -> vec3<f32> { return frame.dir_lights[min(i, 3u)].color.rgb; }
fn receives_shadow() -> bool { return (mesh.flags.z & FLAG_RECEIVE_SHADOW) != 0u; }
fn threers_shadow(color_w: f32, s: Shading) -> f32 {
    if (!receives_shadow()) { return 1.0; }
    return shadow_at(light_shadow_record(color_w), s.world_pos, s.normal);
}
fn directional_shadow(i: u32, s: Shading) -> f32 {
    if (i >= light_count()) { return 1.0; }
    return threers_shadow(frame.dir_lights[i].color.w, s);
}
fn point_light_count() -> u32 { return min(frame.light_counts.y, 8u); }
fn point_light_direction(i: u32, s: Shading) -> vec3<f32> {
    return normalize(frame.point_lights[min(i, 7u)].position.xyz - s.world_pos);
}
fn point_light_color(i: u32, s: Shading) -> vec3<f32> {
    let l = frame.point_lights[min(i, 7u)];
    return l.color.rgb * punctual_attenuation(distance(l.position.xyz, s.world_pos), l.params.x, l.params.y);
}
fn point_shadow(i: u32, s: Shading) -> f32 {
    if (i >= point_light_count()) { return 1.0; }
    return threers_shadow(frame.point_lights[i].color.w, s);
}
fn hemisphere_light(n: vec3<f32>) -> vec3<f32> {
    var c = vec3<f32>(0.0);
    for (var k = 0u; k < min(frame.light_counts.w, 2u); k = k + 1u) {
        let h = frame.hemi_lights[k];
        c = c + mix(h.ground_color.rgb, h.sky_color.rgb, dot(n, normalize(h.direction.xyz)) * 0.5 + 0.5);
    }
    return c;
}
fn tex0(uv: vec2<f32>) -> vec4<f32> { return textureSample(u_tex0, u_samp, uv); }
fn tex1(uv: vec2<f32>) -> vec4<f32> { return textureSample(u_tex1, u_samp, uv); }
fn tex2(uv: vec2<f32>) -> vec4<f32> { return textureSample(u_tex2, u_samp, uv); }
fn tex3(uv: vec2<f32>) -> vec4<f32> { return textureSample(u_tex3, u_samp, uv); }
fn tex0_level(uv: vec2<f32>, lod: f32) -> vec4<f32> { return textureSampleLevel(u_tex0, u_samp, uv, lod); }
fn tex1_level(uv: vec2<f32>, lod: f32) -> vec4<f32> { return textureSampleLevel(u_tex1, u_samp, uv, lod); }
fn tex2_level(uv: vec2<f32>, lod: f32) -> vec4<f32> { return textureSampleLevel(u_tex2, u_samp, uv, lod); }
fn tex3_level(uv: vec2<f32>, lod: f32) -> vec4<f32> { return textureSampleLevel(u_tex3, u_samp, uv, lod); }
fn tex0_size() -> vec2<u32> { return textureDimensions(u_tex0); }
fn tex1_size() -> vec2<u32> { return textureDimensions(u_tex1); }
fn tex2_size() -> vec2<u32> { return textureDimensions(u_tex2); }
fn tex3_size() -> vec2<u32> { return textureDimensions(u_tex3); }
fn opacity() -> f32 { return mesh.params.y; }
fn fog(color: vec3<f32>, s: Shading) -> vec3<f32> { return apply_fog(color, s.view_z); }
fn tone_map(color: vec3<f32>) -> vec3<f32> { return apply_tone_mapping(color); }
fn s0_len() -> u32 { return s_len(0u); }
fn s1_len() -> u32 { return s_len(1u); }
fn s2_len() -> u32 { return s_len(2u); }
";

const WGSL_ENTRY: &str = "
@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    var s: Shading;
    s.world_pos = in.world_pos;
    s.normal = normalize(in.world_normal) * select(-1.0, 1.0, front_facing);
    s.view_dir = normalize(frame.camera_position.xyz - in.world_pos);
    s.uv = in.uv;
    s.color = in.vertex_color;
    s.view_depth = length(frame.camera_position.xyz - in.world_pos);
    s.front_facing = front_facing;
    s.view_z = in.view_z;
    let c = shade(s);
    return vec4<f32>(framebuffer_encode(c.rgb), c.a);
}
";

// ------------------------------------------------------------------- parsing

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Tok {
    Ident(String),
    /// A number with Rust's suffix split off.
    Num {
        text: String,
        suffix: String,
    },
    Punct(String),
}

impl Tok {
    pub(super) fn is(&self, s: &str) -> bool {
        matches!(self, Tok::Punct(p) | Tok::Ident(p) if p == s)
    }
}

fn tokenize(src: &str) -> Result<Vec<Tok>, String> {
    // `>>` is a shift, so a nested template type has to close as `> >`;
    // the shorthand types (`vec4f`) seldom need one.
    const MULTI: [&str; 18] = [
        "->", "==", "!=", "<=", ">=", "&&", "||", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
        "::", "<<", ">>",
    ];
    let chars: Vec<char> = src.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c == '/' && chars.get(i + 1) == Some(&'/') {
            // WGSL text has comments; the macro's tokens never do.
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if c == '/' && chars.get(i + 1) == Some(&'*') {
            // Block comments nest in WGSL.
            let mut depth = 0;
            while i < chars.len() {
                if chars[i] == '/' && chars.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                } else if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            toks.push(Tok::Ident(chars[start..i].iter().collect()));
        } else if c.is_ascii_digit() {
            let start = i;
            while i < chars.len()
                && (chars[i].is_ascii_alphanumeric()
                    || chars[i] == '_'
                    || (chars[i] == '.' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit()))
                    || ((chars[i] == '-' || chars[i] == '+')
                        && matches!(chars[i - 1], 'e' | 'E')
                        && !chars[start..i]
                            .iter()
                            .any(|c| c.is_ascii_alphabetic() && *c != 'e' && *c != 'E')))
            {
                i += 1;
            }
            let raw: String = chars[start..i].iter().filter(|c| **c != '_').collect();
            let split = ["f32", "u32", "i32", "f64", "u64", "i64", "usize", "isize"]
                .iter()
                .find(|s| raw.ends_with(*s) && raw.len() > s.len());
            let (text, suffix) = match split {
                Some(s) => (raw[..raw.len() - s.len()].to_string(), s.to_string()),
                None => (raw, String::new()),
            };
            toks.push(Tok::Num { text, suffix });
        } else {
            let two: String = chars[i..(i + 2).min(chars.len())].iter().collect();
            if MULTI.contains(&two.as_str()) {
                toks.push(Tok::Punct(two));
                i += 2;
            } else if "{}()[]<>;:,.=+-*/%!&|^?@#".contains(c) {
                toks.push(Tok::Punct(c.to_string()));
                i += 1;
            } else {
                return Err(format!("shader!: unexpected character `{c}`"));
            }
        }
    }
    Ok(toks)
}

#[derive(Clone, Debug)]
pub(super) enum Item {
    Const(ConstItem),
    Fn(FnItem),
    Struct(StructItem),
}

#[derive(Clone, Debug)]
pub(super) struct StructItem {
    pub(super) name: String,
    pub(super) fields: Vec<(String, Vec<Tok>)>,
}

#[derive(Clone, Debug)]
pub(super) struct ConstItem {
    name: String,
    /// Empty when WGSL infers it.
    ty: Vec<Tok>,
    value: Vec<Tok>,
}

#[derive(Clone, Debug)]
pub(super) struct FnItem {
    pub(super) name: String,
    pub(super) params: Vec<(String, Vec<Tok>)>,
    /// Each parameter's `@builtin(...)`, if it has one. Read by the Metal
    /// translation only.
    #[allow(dead_code)]
    builtins: Vec<Option<String>>,
    pub(super) ret: Option<Vec<Tok>>,
    pub(super) body: Vec<Tok>,
}

fn parse(src: &str) -> Result<Vec<Item>, String> {
    let items = parse_items(src)?;
    if !items
        .iter()
        .any(|it| matches!(it, Item::Fn(f) if f.name == "shade"))
    {
        return Err("shader!: no `fn shade(s: Shading) -> vec4f` entry point".into());
    }
    Ok(items)
}

/// `@name` or `@name(...)` at `i`: its name and the index after it.
fn attribute(toks: &[Tok], i: usize) -> Option<(String, Option<String>, usize)> {
    if !toks.get(i)?.is("@") {
        return None;
    }
    let Tok::Ident(name) = toks.get(i + 1)? else {
        return None;
    };
    if toks.get(i + 2).is_some_and(|t| t.is("(")) {
        let close = matching(toks, i + 2).ok()?;
        let arg = toks[i + 3..close]
            .iter()
            .map(tok_text)
            .collect::<Vec<_>>()
            .join("");
        Some((name.clone(), Some(arg), close + 1))
    } else {
        Some((name.clone(), None, i + 2))
    }
}

fn tok_text(t: &Tok) -> String {
    match t {
        Tok::Ident(s) | Tok::Punct(s) => s.clone(),
        Tok::Num { text, suffix } => format!("{text}{suffix}"),
    }
}

/// Drop every `@attribute` from a token run.
fn strip_attributes(toks: &[Tok]) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if let Some((_, _, next)) = attribute(toks, i) {
            i = next;
        } else {
            out.push(toks[i].clone());
            i += 1;
        }
    }
    out
}

pub(super) fn parse_items(src: &str) -> Result<Vec<Item>, String> {
    let toks = tokenize(src)?;
    let mut items = Vec::new();
    let mut i = 0;
    let expect = |i: usize, s: &str| -> Result<(), String> {
        match toks.get(i) {
            Some(t) if t.is(s) => Ok(()),
            other => Err(format!("shader!: expected `{s}`, found {other:?}")),
        }
    };
    let ident = |i: usize| -> Result<String, String> {
        match toks.get(i) {
            Some(Tok::Ident(s)) => Ok(s.clone()),
            other => Err(format!("shader!: expected a name, found {other:?}")),
        }
    };
    while i < toks.len() {
        // `@fragment`, `@vertex`: the entry points are found by name.
        if let Some((_, _, next)) = attribute(&toks, i) {
            i = next;
            continue;
        }
        if toks[i].is(";") {
            i += 1;
        } else if toks[i].is("const") || toks[i].is("override") {
            let name = ident(i + 1)?;
            let eq = (i + 2..toks.len())
                .find(|&j| toks[j].is("="))
                .ok_or("shader!: `const` without a value")?;
            let semi = (eq..toks.len())
                .find(|&j| toks[j].is(";"))
                .ok_or("shader!: `const` without `;`")?;
            let ty = if toks[i + 2].is(":") {
                toks[i + 3..eq].to_vec()
            } else {
                Vec::new()
            };
            items.push(Item::Const(ConstItem {
                name,
                ty,
                value: toks[eq + 1..semi].to_vec(),
            }));
            i = semi + 1;
        } else if toks[i].is("struct") {
            let name = ident(i + 1)?;
            expect(i + 2, "{")?;
            let close = matching(&toks, i + 2)?;
            let mut fields = Vec::new();
            for part in split_top(&strip_attributes(&toks[i + 3..close]), ",") {
                if part.is_empty() {
                    continue;
                }
                let Some(Tok::Ident(fname)) = part.first() else {
                    return Err(format!("shader!: bad field in `{name}`"));
                };
                fields.push((fname.clone(), part[2..].to_vec()));
            }
            items.push(Item::Struct(StructItem { name, fields }));
            i = close + 1;
        } else if toks[i].is("fn") {
            let name = ident(i + 1)?;
            expect(i + 2, "(")?;
            let close = matching(&toks, i + 2)?;
            let mut params = Vec::new();
            let mut builtins = Vec::new();
            for part in split_top(&toks[i + 3..close], ",") {
                if part.is_empty() {
                    continue;
                }
                let mut builtin = None;
                let mut k = 0;
                while let Some((attr, arg, next)) = attribute(part, k) {
                    if attr == "builtin" {
                        builtin = arg;
                    }
                    k = next;
                }
                let part = &part[k..];
                let Some(Tok::Ident(pname)) = part.first() else {
                    return Err(format!("shader!: bad parameter in `{name}`"));
                };
                if !part.get(1).is_some_and(|t| t.is(":")) {
                    return Err(format!(
                        "shader!: parameter `{pname}` of `{name}` needs a type"
                    ));
                }
                params.push((pname.clone(), part[2..].to_vec()));
                builtins.push(builtin);
            }
            let mut j = close + 1;
            let ret = if toks.get(j).is_some_and(|t| t.is("->")) {
                let open = (j..toks.len())
                    .find(|&k| toks[k].is("{"))
                    .ok_or_else(|| format!("shader!: `{name}` has no body"))?;
                let r = strip_attributes(&toks[j + 1..open]);
                j = open;
                Some(r)
            } else {
                None
            };
            expect(j, "{")?;
            let end = matching(&toks, j)?;
            items.push(Item::Fn(FnItem {
                name,
                params,
                builtins,
                ret,
                body: toks[j + 1..end].to_vec(),
            }));
            i = end + 1;
        } else {
            return Err(format!(
                "shader!: expected `fn`, `const` or `struct` at the top level, found {:?}",
                toks[i]
            ));
        }
    }
    Ok(items)
}

/// Index of the bracket closing the one at `open`.
fn matching(toks: &[Tok], open: usize) -> Result<usize, String> {
    let mut depth = 0i32;
    for (k, t) in toks.iter().enumerate().skip(open) {
        if t.is("(") || t.is("{") || t.is("[") {
            depth += 1;
        } else if t.is(")") || t.is("}") || t.is("]") {
            depth -= 1;
            if depth == 0 {
                return Ok(k);
            }
        }
    }
    Err("shader!: unbalanced brackets".into())
}

/// Split at `sep` outside any brackets (`<`…`>` included, for `vec3<f32>`).
fn split_top<'a>(toks: &'a [Tok], sep: &str) -> Vec<&'a [Tok]> {
    let mut parts = Vec::new();
    let (mut depth, mut start) = (0i32, 0);
    for (k, t) in toks.iter().enumerate() {
        if t.is("(") || t.is("[") || t.is("{") || t.is("<") {
            depth += 1;
        } else if t.is(")") || t.is("]") || t.is("}") || t.is(">") {
            depth -= 1;
        } else if depth == 0 && t.is(sep) {
            parts.push(&toks[start..k]);
            start = k + 1;
        }
    }
    parts.push(&toks[start..]);
    parts
}

// ---------------------------------------------------------------- emitting

#[derive(Clone, Copy, PartialEq)]
enum Lang {
    Wgsl,
    Msl,
}

/// A type or a name in either language.
fn map_ident(name: &str, lang: Lang) -> String {
    if lang == Lang::Wgsl {
        return name.to_string();
    }
    match name {
        "f32" => "float",
        "u32" => "uint",
        "i32" => "int",
        "vec2f" => "float2",
        "vec3f" => "float3",
        "vec4f" => "float4",
        "vec2u" => "uint2",
        "vec3u" => "uint3",
        "vec4u" => "uint4",
        "vec2i" => "int2",
        "vec3i" => "int3",
        "vec4i" => "int4",
        "mat3x3f" => "float3x3",
        "mat4x4f" => "float4x4",
        "inverseSqrt" => "rsqrt",
        "dpdx" | "dpdxFine" | "dpdxCoarse" => "dfdx",
        "dpdy" | "dpdyFine" | "dpdyCoarse" => "dfdy",
        "countOneBits" => "popcount",
        "reverseBits" => "reverse_bits",
        "extractBits" => "extract_bits",
        "insertBits" => "insert_bits",
        "pack4x8unorm" => "pack_float_to_unorm4x8",
        "unpack4x8unorm" => "unpack_unorm4x8_to_float",
        other => other,
    }
    .to_string()
}

fn num(text: &str, suffix: &str, lang: Lang) -> String {
    let float = text.contains('.') || text.contains('e') || text.contains('E');
    // WGSL's own suffixes, on literals written as WGSL text: `1u`, `2i`,
    // `1f`, `0.5h`. MSL has `u`, and `f` only after a decimal point.
    if suffix.is_empty() && lang == Lang::Msl && !text.starts_with("0x") {
        if let Some(stem) = text.strip_suffix('i') {
            return stem.to_string();
        }
        if let Some(stem) = text.strip_suffix('f').or_else(|| text.strip_suffix('h')) {
            return if stem.contains('.') || stem.contains('e') {
                format!("{stem}f")
            } else {
                format!("{stem}.0f")
            };
        }
    }
    match (suffix, lang) {
        ("u32" | "u64" | "usize", _) => format!("{text}u"),
        ("f32" | "f64", _) if float => text.to_string(),
        ("f32" | "f64", _) => format!("{text}.0"),
        ("i32" | "i64" | "isize", Lang::Wgsl) => format!("{text}i"),
        _ => text.to_string(),
    }
}

/// WGSL's templated names — `vec3<f32>`, `array<f32, 4>`, `bitcast<u32>` —
/// as one MSL token each.
fn collapse_generics(toks: &[Tok]) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if let Tok::Ident(name) = &toks[i] {
            let templated = [
                "vec2", "vec3", "vec4", "mat2x2", "mat3x3", "mat4x4", "mat2x3", "mat2x4", "mat3x2",
                "mat3x4", "mat4x2", "mat4x3", "array", "ptr", "bitcast",
            ]
            .contains(&name.as_str());
            if templated && toks.get(i + 1).is_some_and(|t| t.is("<")) {
                // The `>` closing this list, counting nested `<`.
                let mut depth = 0;
                let mut close = None;
                for (k, t) in toks.iter().enumerate().skip(i + 1) {
                    if t.is("<") {
                        depth += 1;
                    } else if t.is(">") {
                        depth -= 1;
                        if depth == 0 {
                            close = Some(k);
                            break;
                        }
                    }
                }
                if let Some(close) = close {
                    let inner = collapse_generics(&toks[i + 2..close]);
                    let args: Vec<String> = split_top(&inner, ",")
                        .iter()
                        .map(|a| join(a, Lang::Msl))
                        .collect();
                    let elem = |a: &str| -> String {
                        match a {
                            "f32" | "float" => "float".into(),
                            "u32" | "uint" => "uint".into(),
                            "i32" | "int" => "int".into(),
                            "f16" | "half" => "half".into(),
                            "bool" => "bool".into(),
                            other => other.into(),
                        }
                    };
                    let mapped = match name.as_str() {
                        n if n.starts_with("vec") => format!("{}{}", elem(&args[0]), &n[3..]),
                        n if n.starts_with("mat") => format!("{}{}", elem(&args[0]), &n[3..]),
                        "array" => match args.get(1) {
                            Some(len) => format!("array<{}, {}>", args[0], len),
                            None => format!("array<{}, 1>", args[0]),
                        },
                        // `ptr<function, T>` is a reference parameter.
                        "ptr" => format!("thread {}&", args.get(1).cloned().unwrap_or_default()),
                        "bitcast" => format!("as_type<{}>", args[0]),
                        _ => unreachable!(),
                    };
                    out.push(Tok::Ident(mapped));
                    i = close + 1;
                    continue;
                }
            }
        }
        out.push(toks[i].clone());
        i += 1;
    }
    out
}

/// Tokens back to text, spaced so neither compiler mis-joins them.
fn join(toks: &[Tok], lang: Lang) -> String {
    let collapsed;
    let toks = if lang == Lang::Msl {
        let c = without_pointer_ops(&collapse_generics(toks));
        // WGSL allows a trailing comma in an argument list; MSL does not.
        collapsed = c
            .iter()
            .enumerate()
            .filter(|(k, t)| {
                !(t.is(",")
                    && c.get(k + 1)
                        .is_some_and(|n| n.is(")") || n.is("]") || n.is("}")))
            })
            .map(|(_, t)| t.clone())
            .collect::<Vec<_>>();
        &collapsed[..]
    } else {
        toks
    };
    let mut out = String::new();
    for (k, t) in toks.iter().enumerate() {
        let s = match t {
            Tok::Ident(s) => map_ident(s, lang),
            Tok::Num { text, suffix } => num(text, suffix, lang),
            Tok::Punct(p) => p.clone(),
        };
        let glue_left = matches!(t, Tok::Punct(p) if [".", ",", ";", ")", "]"].contains(&p.as_str()))
            || (matches!(t, Tok::Punct(p) if p == "(" || p == "[")
                && k > 0
                && matches!(toks[k - 1], Tok::Ident(_) | Tok::Punct(_) if !toks[k - 1].is(",") && !toks[k - 1].is("(") && !is_operator(&toks[k - 1])));
        let after_dot = k > 0 && toks[k - 1].is(".");
        if !out.is_empty() && !glue_left && !after_dot && !out.ends_with('(') && !out.ends_with('[')
        {
            out.push(' ');
        }
        out.push_str(&s);
    }
    out
}

/// WGSL's `&x` (passing a pointer) and `*p` (reading through one) are plain
/// names in MSL, whose pointer parameters are references.
fn without_pointer_ops(toks: &[Tok]) -> Vec<Tok> {
    let mut out: Vec<Tok> = Vec::new();
    for (k, t) in toks.iter().enumerate() {
        let unary = k == 0
            || toks[k - 1].is("(")
            || toks[k - 1].is(",")
            || toks[k - 1].is("=")
            || toks[k - 1].is("return")
            || (is_operator(&toks[k - 1]) && !toks[k - 1].is(")"));
        let next_is_name = matches!(toks.get(k + 1), Some(Tok::Ident(_)) | Some(Tok::Punct(_)) if toks[k + 1].is("(") || matches!(toks[k + 1], Tok::Ident(_)));
        if (t.is("&") || t.is("*")) && unary && next_is_name {
            continue;
        }
        out.push(t.clone());
    }
    out
}

fn is_operator(t: &Tok) -> bool {
    matches!(t, Tok::Punct(p) if ["=", "+", "-", "*", "/", "%", "<", ">", "!", "&&", "||", "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "?", ":", "return"].contains(&p.as_str()))
        || t.is("return")
}

/// A function body or statement list, statement by statement.
fn body(toks: &[Tok], lang: Lang, indent: usize) -> String {
    let pad = "    ".repeat(indent);
    let mut out = String::new();
    let mut i = 0;
    while i < toks.len() {
        let t = &toks[i];
        // Control flow with a block: `if (..) {`, `for (..) {`, `else {`, `loop {`.
        if t.is("if") || t.is("for") || t.is("while") || t.is("else") || t.is("loop") {
            let open = (i..toks.len()).find(|&k| toks[k].is("{"));
            let Some(open) = open else {
                out.push_str(&format!("{pad}{}\n", join(&toks[i..], lang)));
                break;
            };
            let head = &toks[i..open];
            let head = if t.is("for") {
                for_head(head, lang)
            } else if t.is("loop") && lang == Lang::Msl {
                "while (true)".to_string()
            } else {
                join(head, lang)
            };
            let close = matching(toks, open).unwrap_or(toks.len() - 1);
            out.push_str(&format!("{pad}{head} {{\n"));
            out.push_str(&body(&toks[open + 1..close], lang, indent + 1));
            out.push_str(&format!("{pad}}}\n"));
            i = close + 1;
            continue;
        }
        let end = (i..toks.len())
            .find(|&k| toks[k].is(";"))
            .unwrap_or(toks.len());
        out.push_str(&format!("{pad}{};\n", statement(&toks[i..end], lang)));
        i = end + 1;
    }
    out
}

fn statement(st: &[Tok], lang: Lang) -> String {
    if st.is_empty() {
        return String::new();
    }
    if st[0].is("discard") && lang == Lang::Msl {
        return "discard_fragment()".into();
    }
    if (st[0].is("let") || st[0].is("var") || st[0].is("const")) && lang == Lang::Msl {
        let mutable = st[0].is("var");
        // `var<function>` and friends.
        let st = &if st.get(1).is_some_and(|t| t.is("<")) {
            let close = st.iter().position(|t| t.is(">")).unwrap_or(1);
            [&st[..1], &st[close + 1..]].concat()
        } else {
            st.to_vec()
        }[..];
        let Some(Tok::Ident(name)) = st.get(1) else {
            return join(st, lang);
        };
        let eq = st.iter().position(|t| t.is("="));
        let ty = if st.get(2).is_some_and(|t| t.is(":")) {
            Some(join(&st[3..eq.unwrap_or(st.len())], lang))
        } else {
            None
        };
        let value = eq.map(|e| join(&st[e + 1..], lang));
        // One of the frame's light or shadow records, which live in
        // constant memory: a reference, not a copy. A copy indexed at run
        // time (`rec.vp[cascade]`) goes to stack memory, and cost a smoke
        // column's shadow lookup four times wgpu's.
        if !mutable && ty.is_none() && eq.is_some_and(|e| is_frame_record(&st[e + 1..])) {
            return format!("constant auto &{name} = {}", value.unwrap_or_default());
        }
        let decl = match (&ty, mutable) {
            (Some(ty), true) => ty.clone(),
            (Some(ty), false) => format!("const {ty}"),
            (None, true) => "auto".into(),
            (None, false) => "const auto".into(),
        };
        return match value {
            Some(v) => format!("{decl} {name} = {v}"),
            None => format!("{decl} {name}"),
        };
    }
    join(st, lang)
}

/// `frame.shadows[r]` and the like: an element of an array `WgslFrame`
/// points at in constant memory, and nothing more.
fn is_frame_record(value: &[Tok]) -> bool {
    const ARRAYS: [&str; 5] = [
        "shadows",
        "dir_lights",
        "point_lights",
        "spot_lights",
        "hemi_lights",
    ];
    value.len() > 4
        && value[0].is("frame")
        && value[1].is(".")
        && ARRAYS.iter().any(|a| value[2].is(a))
        && value[3].is("[")
        && matching(value, 3).is_ok_and(|close| close == value.len() - 1)
}

/// `for (var i = 0u32; i < n; i += 1u32)`: the initialiser is a declaration.
fn for_head(head: &[Tok], lang: Lang) -> String {
    let Some(open) = head.iter().position(|t| t.is("(")) else {
        return join(head, lang);
    };
    let close = matching(head, open).unwrap_or(head.len() - 1);
    let parts = split_top(&head[open + 1..close], ";");
    let rendered: Vec<String> = parts
        .iter()
        .enumerate()
        .map(|(k, p)| {
            if k == 0 {
                statement(p, lang)
            } else {
                join(p, lang)
            }
        })
        .collect();
    format!("for ({})", rendered.join("; "))
}

impl Item {
    pub(super) fn to_wgsl(&self) -> String {
        match self {
            Item::Struct(st) => {
                let fields: Vec<String> = st
                    .fields
                    .iter()
                    .map(|(n, t)| format!("    {n}: {},", join(t, Lang::Wgsl)))
                    .collect();
                format!("struct {} {{\n{}\n}};", st.name, fields.join("\n"))
            }
            Item::Const(c) if c.ty.is_empty() => {
                format!("const {} = {};", c.name, join(&c.value, Lang::Wgsl))
            }
            Item::Const(c) => format!(
                "const {}: {} = {};",
                c.name,
                join(&c.ty, Lang::Wgsl),
                join(&c.value, Lang::Wgsl)
            ),
            Item::Fn(f) => {
                let params: Vec<String> = f
                    .params
                    .iter()
                    .map(|(n, t)| format!("{n}: {}", join(t, Lang::Wgsl)))
                    .collect();
                let ret = f
                    .ret
                    .as_ref()
                    .map(|r| format!(" -> {}", join(r, Lang::Wgsl)))
                    .unwrap_or_default();
                format!(
                    "fn {}({}){ret} {{\n{}}}\n",
                    f.name,
                    params.join(", "),
                    body(&f.body, Lang::Wgsl, 1)
                )
            }
        }
    }

    pub(super) fn to_msl(&self) -> String {
        match self {
            Item::Struct(st) => {
                let fields: Vec<String> = st
                    .fields
                    .iter()
                    .map(|(n, t)| format!("    {} {n};", join(t, Lang::Msl)))
                    .collect();
                format!("struct {} {{\n{}\n}};", st.name, fields.join("\n"))
            }
            Item::Const(c) => format!(
                "constant {} {} = {};",
                if c.ty.is_empty() {
                    "auto".to_string()
                } else {
                    join(&c.ty, Lang::Msl)
                },
                c.name,
                join(&c.value, Lang::Msl)
            ),
            Item::Fn(f) => {
                let params: Vec<String> = f
                    .params
                    .iter()
                    .map(|(n, t)| format!("{} {n}", join(t, Lang::Msl)))
                    .collect();
                let ret = f
                    .ret
                    .as_ref()
                    .map(|r| join(r, Lang::Msl))
                    .unwrap_or_else(|| "void".into());
                // Not `const`: a WGSL function may write module state
                // (`var<private>`), which lives in the struct.
                format!(
                    "    {ret} {}({}) {{\n{}    }}\n",
                    f.name,
                    params.join(", "),
                    body(&f.body, Lang::Msl, 2)
                )
            }
        }
    }
}

// ------------------------------------------------- WGSL materials on Metal

/// A [`ShaderMaterial`](super::ShaderMaterial)'s own WGSL, as MSL
/// for the native Metal backend: no shared source, no hand-written port.
///
/// The fragment's functions become members of a struct that holds what the
/// custom-shader preamble gives a WGSL fragment — `frame`, `mesh` and
/// `u_data` under wgpu's field names, filled from Metal's own uniforms;
/// `s0`…`s2` and `s_len`; `framebuffer_encode` (the identity unless
/// standing in for a `Unorm` target, as on wgpu), `apply_fog` and `apply_tone_mapping` — and
/// `fs_main` is called with a `VsOut` built from Metal's interpolants.
///
/// A `screen_space` material's `ss_color_tex`, `ss_color_samp` and
/// `ss_depth_tex` are the Metal capture of the opaque scene: the
/// `textureSampleLevel`, `textureSample`, `textureLoad` and
/// `textureDimensions` calls on them are overloads in the environment.
///
/// `u_tex0`..`u_tex3` and `u_samp` are the material's
/// [`maps`](super::ShaderMaterial::maps), bound at textures 16–19 and
/// sampler 15 in both stages.
///
/// What it cannot give is a reason, not a wrong picture: a shader that reads
/// a volume (a wgpu handle, which Metal cannot bind) or returns
/// more than a colour is an error, and the renderer draws the material's
/// fallback colour.
///
/// A custom vertex stage (`vs_custom`, taking a `VsIn` or an
/// `InstancedCustomIn`) comes along: one module, as wgpu assembles it, with a
/// vertex entry that builds that input from Metal's buffers — the instance
/// matrix, tint and two custom columns included — and hands every `VsOut`
/// field to the fragment.
// Only the native Metal backend calls this; other targets still build it.
#[allow(dead_code)]
pub(crate) fn wgsl_material_to_msl(fragment: &str, vertex: Option<&str>) -> Result<String, String> {
    let module = match vertex {
        Some(v) => format!("{v}\n{fragment}"),
        None => fragment.to_string(),
    };
    let fragment = module.as_str();
    if fragment.contains("ThreersFragOut") {
        return Err("its WGSL writes a sample mask, which the Metal backend cannot bind".into());
    }
    let items = parse_items(fragment)?;
    let entry = items
        .iter()
        .find_map(|it| match it {
            Item::Fn(f) if f.name == "fs_main" => Some(f),
            _ => None,
        })
        .ok_or("no `fs_main`")?;
    let returns_colour = entry.ret.as_ref().is_some_and(|r| {
        let t = join(r, Lang::Msl);
        t == "float4" || t == "vec4f"
    });
    if !returns_colour {
        return Err("its `fs_main` returns something other than a colour".into());
    }
    let mut args = Vec::new();
    for ((name, ty), builtin) in entry.params.iter().zip(&entry.builtins) {
        args.push(match builtin.as_deref() {
            Some("front_facing") => "front_facing".to_string(),
            Some("position") => "vin.clip_pos".to_string(),
            Some(other) => return Err(format!("`fs_main` takes @builtin({other})")),
            None if join(ty, Lang::Msl) == "VsOut" => "vin".to_string(),
            None => return Err(format!("`fs_main`'s parameter `{name}` is not a VsOut")),
        });
    }
    let vertex_entry = match vertex {
        None => None,
        Some(_) => {
            let vs = items
                .iter()
                .find_map(|it| match it {
                    Item::Fn(f) if f.name == "vs_custom" => Some(f),
                    _ => None,
                })
                .ok_or("its vertex stage has no `vs_custom`")?;
            let mut vargs = Vec::new();
            let mut input = None;
            for ((name, ty), builtin) in vs.params.iter().zip(&vs.builtins) {
                vargs.push(match builtin.as_deref() {
                    Some("vertex_index") => "vid".to_string(),
                    Some("instance_index") => "iid".to_string(),
                    Some(other) => return Err(format!("`vs_custom` takes @builtin({other})")),
                    None => {
                        let t = join(ty, Lang::Msl);
                        if t != "VsIn" && t != "InstancedCustomIn" {
                            return Err(format!("`vs_custom`'s `{name}` is a {t}, not a VsIn"));
                        }
                        input = Some(t);
                        "vin".to_string()
                    }
                });
            }
            let input = input.unwrap_or_else(|| "VsIn".into());
            let instanced = if input == "InstancedCustomIn" {
                "vin.imat0 = im[0]; vin.imat1 = im[1]; vin.imat2 = im[2]; vin.imat3 = im[3];
    uint at = draw.flags.w >> 1;
    vin.itint = (instanced && at != 0u) ? float4(((device const float4 *)instances)[at + iid].rgb, 1.0) : float4(1.0);
    vin.icustom0 = custom[iid * 2u];
    vin.icustom1 = custom[iid * 2u + 1u];"
            } else {
                ""
            };
            Some(
                WGSL_COMPAT_VERTEX_ENTRY
                    .replace("$INPUT", &input)
                    .replace("$ARGS", &vargs.join(", "))
                    .replace("THREERS_INSTANCED_FIELDS", instanced),
            )
        }
    };
    let mut globals = String::new();
    let mut members = String::new();
    // The preamble's own functions this shader calls, and theirs in turn.
    for f in preamble_functions_used(&items) {
        members.push_str(&f.to_msl());
        members.push('\n');
    }
    for item in &items {
        match item {
            Item::Fn(_) => {
                members.push_str(&item.to_msl());
                members.push('\n');
            }
            _ => {
                globals.push_str(&item.to_msl());
                globals.push('\n');
            }
        }
    }
    let fragment_entry = match vertex_entry {
        None => WGSL_COMPAT_ENTRY,
        Some(_) => WGSL_COMPAT_VARYINGS_ENTRY,
    }
    .replace("$ARGS", &args.join(", "));
    let fragment_entry = with_encoded_variants(&fragment_entry);
    // The preamble's constants and the shader's own globals live in a
    // namespace with the context struct, so a name Metal's library also uses
    // (`MAP_NORMAL`, say) resolves to the WGSL one inside the shader.
    Ok(format!(
        "{WGSL_COMPAT_PRELUDE}{OUTPUT_HELPERS}namespace threers_wgsl {{\n{WGSL_COMPAT_ALIASES}{}{globals}{WGSL_COMPAT_OPEN}{members}}};\n}}\nusing threers_wgsl::ThreersWgsl;\n{}{fragment_entry}",
        preamble_constants(),
        vertex_entry.unwrap_or_default()
    ))
}

/// wgpu's names for what a WGSL fragment reads, over Metal's structs.
const WGSL_COMPAT_PRELUDE: &str = "
// ---- threers: a WGSL ShaderMaterial on Metal ----
struct WgslFrame {
    float4x4 view;
    float4x4 projection;
    float4x4 view_proj;
    float4 camera_position;
    float4 ambient;
    uint4 light_counts;
    float4 tone_mapping_exposure;
    float4 fog_color;
    float4 fog_params;
    float4 viewport_size;
    // Pointers into Metal's frame rather than copies: indexed the same way,
    // and a fragment does not carry every light and shadow record with it.
    // (An array member indexed at runtime would put this struct, and the
    // shader's whole state around it, in stack memory: the snow cover's
    // hemisphere loop doubled its cost under MSAA.)
    constant DirLight *dir_lights;
    constant PointLight *point_lights;
    constant SpotLight *spot_lights;
    constant HemiLight *hemi_lights;
    uint4 shadow_info;
    constant ShadowRecord *shadows;
};
struct WgslMesh {
    float4x4 model;
    float4x4 normal_matrix;
    float4 color;
    float4 emissive;
    float4 specular;
    float4 params;
    float4 params2;
    uint4 flags;
};
struct VsOut {
    float4 clip_pos;
    float3 world_pos;
    float3 world_normal;
    float2 uv;
    float view_z;
    float4 vertex_color;
    float2 clip_zw;
    float4 proj_uv;
    float4 world_tangent;
    float3 local_pos;
    float4 uv12;
    float2 uv3;
};
struct ThreersUserData { float4 data[16]; };
/// The opaque scene behind a screen-space material: the capture's colour
/// (one slice per view, mipped) and its depth.
struct ThreersScreen {
    texture2d_array<float> color;
    sampler samp;
    texture2d<float> depth;
    uint slice;
};
/// `ss_color_tex`, `ss_color_samp` and `ss_depth_tex` as values: the WGSL
/// passes them to `textureSampleLevel` and friends, which the compat struct
/// overloads on these types.
struct ThreersSsColor {};
struct ThreersSsSamp {};
struct ThreersSsDepth {};
/// A material's own textures, `u_tex0`..`u_tex3`, and `u_samp`.
struct ThreersUser {
    array<texture2d<float>, 4> tex;
    sampler samp;
};
/// `u_tex0`..`u_tex3` and `u_samp` as values, like the screen-space tags.
template <int N> struct ThreersUTex {};
struct ThreersUSamp {};
/// The material's volumes, `u_vol0` and `u_vol1`, and `u_vol_samp`.
struct ThreersVolumes {
    array<texture3d<float>, 2> vol;
    sampler samp;
};
template <int N> struct ThreersUVol {};
struct ThreersUVolSamp {};
/// The scene's environment as Metal binds it for every mesh draw.
struct ThreersEnv {
    texturecube<float> cube;
    sampler cube_samp;
    texture2d<float> atlas;
    sampler atlas_samp;
};
struct VsIn {
    float3 position;
    float3 normal;
    float2 uv;
    float4 color;
    float4 tangent;
};
struct InstancedCustomIn {
    float3 position;
    float3 normal;
    float2 uv;
    float4 color;
    float4 imat0;
    float4 imat1;
    float4 imat2;
    float4 imat3;
    float4 tangent;
    float4 itint;
    float4 icustom0;
    float4 icustom1;
};
/// `VsOut` between the stages: every field interpolated, as wgpu passes it.
struct WgslVaryings {
    float4 clip_pos [[position]];
    float3 world_pos;
    float3 world_normal;
    float2 uv;
    float  view_z;
    float4 vertex_color;
    float2 clip_zw;
    float4 proj_uv;
    float4 world_tangent;
    float3 local_pos;
    float4 uv12;
    float2 uv3;
};

// WGSL built-ins MSL spells differently or lacks.
template <typename T> static inline T degrees(T x) { return x * 57.29577951308232f; }
template <typename T> static inline T radians(T x) { return x * 0.017453292519943295f; }

/// The inverse of an affine transform: rotation, scale and shear, then
/// translation. MSL has no general inverse.
static inline float4x4 threers_affine_inverse(float4x4 m) {
    float3 a = m[0].xyz, b = m[1].xyz, c = m[2].xyz;
    float3 r0 = cross(b, c), r1 = cross(c, a), r2 = cross(a, b);
    float det = dot(a, r0);
    float3x3 inv3 = transpose(float3x3(r0, r1, r2)) * (1.0 / det);
    float3 t = -(inv3 * m[3].xyz);
    return float4x4(float4(inv3[0], 0.0), float4(inv3[1], 0.0), float4(inv3[2], 0.0), float4(t, 1.0));
}

static inline WgslFrame threers_wgsl_frame(constant Frame &f, uint v) {
    WgslFrame w;
    w.view = f.view[v];
    w.view_proj = f.view_proj[v];
    w.projection = f.view_proj[v] * threers_affine_inverse(f.view[v]);
    w.camera_position = f.camera_pos[v];
    w.ambient = f.ambient;
    w.light_counts = f.counts;
    w.tone_mapping_exposure = float4(0.0);
    w.fog_color = float4(f.fog_color.rgb, 0.0);
    w.fog_params = float4(f.fog_params.xyz, f.fog_color.w);
    // wgpu's `viewport_size.zw` is the camera's near and far, which a
    // shader linearising the captured depth needs. Read back out of the
    // projection: depth is `p32 / d - p22` at distance `d` in perspective
    // and `p32 - p22 d` orthographic, 0 at the near plane and 1 at the far —
    // the other way round with reversed depth (`camera_position.w`).
    const float p22 = w.projection[2][2], p32 = w.projection[3][2];
    const bool persp = abs(w.projection[2][3]) > 0.5;
    const float at0 = p32 / p22;
    const float at1 = persp ? p32 / (1.0 + p22) : (p32 - 1.0) / p22;
    const bool reversed = f.camera_pos[v].w > 0.5;
    w.viewport_size = float4(f.viewport.xy, reversed ? at1 : at0, reversed ? at0 : at1);
    w.dir_lights = f.dir;
    w.point_lights = f.pts;
    w.spot_lights = f.spots;
    w.shadow_info = f.shadow_info;
    w.shadows = f.shadows;
    w.hemi_lights = f.hemis;
    return w;
}

static inline WgslMesh threers_wgsl_mesh(constant Draw &d) {
    WgslMesh m;
    m.model = d.model;
    m.normal_matrix = d.normal_mat;
    m.color = d.base_color;
    m.emissive = d.emissive;
    m.specular = d.specular;
    // wgpu: shininess, opacity, roughness, metalness.
    m.params = float4(d.specular.w, d.base_color.a, d.pbr.x, d.pbr.y);
    m.params2 = float4(d.map_params.y, d.map_params.x, d.map_params.x, 0.0);
    // wgpu's packing: kind; map bits; shader flags (receive shadow 512, no
    // tone map 16384); the alpha test, bit-cast.
    m.flags = uint4(d.flags.x,
                    d.flags.y != 0u ? 1u : 0u,
                    (d.object.x == 1u ? 512u : 0u) | (d.object.w == 0u ? 16384u : 0u),
                    as_type<uint>(d.pbr.z));
    return m;
}

static inline VsOut threers_vsout(FIn fin, constant Frame &f, constant Draw &d) {
    VsOut o;
    float4 clip = f.view_proj[fin.view_index] * float4(fin.world_pos, 1.0);
    o.clip_pos = fin.position;
    o.world_pos = fin.world_pos;
    o.world_normal = fin.normal;
    o.uv = fin.uv;
    o.view_z = -(f.view[fin.view_index] * float4(fin.world_pos, 1.0)).z;
    o.vertex_color = fin.color;
    o.clip_zw = clip.zw;
    o.proj_uv = float4(0.0, 0.0, 0.0, 1.0);
    o.world_tangent = fin.tangent;
    o.local_pos = (threers_affine_inverse(d.model) * float4(fin.world_pos, 1.0)).xyz;
    o.uv12 = fin.uv12;
    o.uv3 = fin.uv3;
    return o;
}
";

const WGSL_COMPAT_OPEN: &str = "
struct ThreersWgsl {
    constant Frame *frame_m_;
    constant Draw *draw_m_;
    WgslFrame frame;
    WgslMesh mesh;
    constant ThreersUserData &u_data;
    // The storage arrays under the preamble's names, read directly
    // (`u_s0[i]`) or through `s0(i)`.
    const device float *u_s0;
    const device float *u_s1;
    const device float *u_s2;
    uint4 lens_;
    /// The fragment's shadow maps; null in the vertex stage, which has none
    /// bound and takes no shadow lookups.
    thread const ShadowMaps *maps_;
    /// The environment; null in the vertex stage, like the shadow maps.
    thread const ThreersEnv *env_;
    /// The screen-space capture; null in the vertex stage.
    thread const ThreersScreen *ss_;
    /// The material's textures, in both stages.
    thread const ThreersUser *ut_;
    /// Its volumes, in both stages.
    thread const ThreersVolumes *vt_;
    /// The preamble's `var<private>`: each record's shadow at this fragment.
    /// An array of the entry's own, not a member: indexed by a runtime
    /// record number, it would put this whole struct in stack memory, and
    /// the snow cover drew ten times slower under MSAA (75 ms, against 8).
    /// Null in the vertex stage, which takes no shadow lookups.
    thread float *shadow_cache;
    ThreersSsColor ss_color_tex;
    ThreersSsSamp ss_color_samp;
    ThreersSsDepth ss_depth_tex;

    float4 textureSampleLevel(ThreersSsColor, ThreersSsSamp, float2 uv, float lod) const {
        return ss_->color.sample(ss_->samp, uv, ss_->slice, level(lod));
    }
    float4 textureSample(ThreersSsColor, ThreersSsSamp, float2 uv) const {
        return ss_->color.sample(ss_->samp, uv, ss_->slice);
    }
    float4 textureLoad(ThreersSsColor, int2 c, int lod) const {
        return ss_->color.read(uint2(c), ss_->slice, uint(lod));
    }
    float4 textureLoad(ThreersSsDepth, int2 c, int lod) const {
        return ss_->depth.read(uint2(c), uint(lod));
    }
    uint2 textureDimensions(ThreersSsColor, uint lod = 0) const {
        return uint2(ss_->color.get_width(lod), ss_->color.get_height(lod));
    }
    uint2 textureDimensions(ThreersSsDepth, uint lod = 0) const {
        return uint2(ss_->depth.get_width(lod), ss_->depth.get_height(lod));
    }
    uint textureNumLevels(ThreersSsColor) const { return ss_->color.get_num_mip_levels(); }
    ThreersUTex<0> u_tex0;
    ThreersUTex<1> u_tex1;
    ThreersUTex<2> u_tex2;
    ThreersUTex<3> u_tex3;
    ThreersUSamp u_samp;

    template <int N> float4 textureSample(ThreersUTex<N>, ThreersUSamp, float2 uv) const {
        return ut_->tex[N].sample(ut_->samp, uv);
    }
    template <int N> float4 textureSampleLevel(ThreersUTex<N>, ThreersUSamp, float2 uv, float lod) const {
        return ut_->tex[N].sample(ut_->samp, uv, level(lod));
    }
    template <int N> float4 textureSampleBias(ThreersUTex<N>, ThreersUSamp, float2 uv, float b) const {
        return ut_->tex[N].sample(ut_->samp, uv, bias(b));
    }
    template <int N> float4 textureSampleGrad(ThreersUTex<N>, ThreersUSamp, float2 uv, float2 dx, float2 dy) const {
        return ut_->tex[N].sample(ut_->samp, uv, gradient2d(dx, dy));
    }
    template <int N> float4 textureLoad(ThreersUTex<N>, int2 c, int lod) const {
        return ut_->tex[N].read(uint2(c), uint(lod));
    }
    template <int N> uint2 textureDimensions(ThreersUTex<N>, uint lod = 0) const {
        return uint2(ut_->tex[N].get_width(lod), ut_->tex[N].get_height(lod));
    }
    template <int N> uint textureNumLevels(ThreersUTex<N>) const { return ut_->tex[N].get_num_mip_levels(); }
    ThreersUVol<0> u_vol0;
    ThreersUVol<1> u_vol1;
    ThreersUVolSamp u_vol_samp;

    template <int N> float4 textureSample(ThreersUVol<N>, ThreersUVolSamp, float3 p) const {
        return vt_->vol[N].sample(vt_->samp, p);
    }
    template <int N> float4 textureSampleLevel(ThreersUVol<N>, ThreersUVolSamp, float3 p, float lod) const {
        return vt_->vol[N].sample(vt_->samp, p, level(lod));
    }
    template <int N> float4 textureLoad(ThreersUVol<N>, int3 c, int lod) const {
        return vt_->vol[N].read(uint3(c), uint(lod));
    }
    template <int N> uint3 textureDimensions(ThreersUVol<N>, uint lod = 0) const {
        return uint3(vt_->vol[N].get_width(lod), vt_->vol[N].get_height(lod), vt_->vol[N].get_depth(lod));
    }

    float shadow_compare(uint kind, float2 uv, int layer, float depth) const {
        return kind == 0u ? maps_->dir.sample_compare(maps_->samp, uv, uint(layer), depth)
             : (kind == 1u ? maps_->spot.sample_compare(maps_->samp, uv, uint(layer), depth)
                           : maps_->point.sample_compare(maps_->samp, uv, uint(layer), depth));
    }
    float shadow_tap(uint kind, int layer, float3 centre, float3 t, float3 b, float2 o, float depth) const {
        return ::shadow_tap(*maps_, kind, layer, centre, t, b, o, depth);
    }
    /// The built-in lookup rather than the preamble's, translated: that one
    /// copies the record (`let rec = frame.shadows[r]`) and indexes the
    /// copy by cascade, which puts it in stack memory.
    float shadow_at(int r, float3 world_pos, float3 world_normal) const {
        return ::shadow_at(*frame_m_, r, *maps_, world_pos, world_normal);
    }
    float3 sample_env_cube(float3 dir, float roughness) const {
        return ::env_radiance(dir, roughness, env_->cube, env_->cube_samp, *draw_m_, env_->atlas, env_->atlas_samp);
    }
    float3 sample_env_cube_unrotated(float3 dir, float roughness) const {
        return draw_m_->env_params.w > 0.5
            ? texture_cube_uv(env_->atlas, env_->atlas_samp, dir, roughness, draw_m_->env_params)
            : sample_env(env_->cube, env_->cube_samp, dir, roughness);
    }
    int light_shadow_record(float color_w) const { return int(color_w + 0.5) - 1; }
    float light_shadow(float color_w) const {
        int r = light_shadow_record(color_w);
        return (r < 0 || r >= 16) ? 1.0 : shadow_cache[r];
    }
    float shadow_factor_dir(uint i) const { return light_shadow(frame.dir_lights[i].color.w); }
    float shadow_factor_spot(uint i) const { return light_shadow(frame.spot_lights[i].color.w); }
    float shadow_factor_point(uint i) const { return light_shadow(frame.point_lights[i].color.w); }

    float s0(uint i) const { return i < lens_.x ? u_s0[i] : 0.0; }
    float s1(uint i) const { return i < lens_.y ? u_s1[i] : 0.0; }
    float s2(uint i) const { return i < lens_.z ? u_s2[i] : 0.0; }
    uint s_len(uint slot) const { return slot == 0u ? lens_.x : (slot == 1u ? lens_.y : lens_.z); }
    // The attachment encodes; standing in for a `Unorm` target, the shader
    // does, as on wgpu (`threers_raw_out` undoes the attachment's).
    float3 framebuffer_encode(float3 c) const {
        return (frame_m_->view_info.y & 2u) != 0u ? srgb_encode(c) : c;
    }
    float3 apply_tone_mapping(float3 c) const { return ::tone_map(c, *frame_m_, *draw_m_); }
    float3 apply_fog(float3 c, float view_z) const { return ::apply_fog(c, view_z, *frame_m_); }
    /// The built-in transform, for a custom vertex stage to start from.
    VsOut standard_vertex(VsIn in) const {
        VsOut out;
        float4 world_p = mesh.model * float4(in.position, 1.0);
        out.world_pos = world_p.xyz;
        out.clip_pos = frame.view_proj * world_p;
        out.clip_zw = out.clip_pos.zw;
        out.world_normal = (mesh.normal_matrix * float4(in.normal, 0.0)).xyz;
        out.uv = in.uv;
        out.view_z = -(frame.view * world_p).z;
        out.vertex_color = in.color;
        out.proj_uv = float4(0.0, 0.0, 0.0, 1.0);
        out.world_tangent = float4((mesh.model * float4(in.tangent.xyz, 0.0)).xyz, in.tangent.w);
        out.local_pos = in.position;
        out.uv12 = float4(in.uv, in.uv);
        out.uv3 = in.uv;
        return out;
    }
    VsOut standard_vertex_morphed(VsIn in, uint vid) const { return standard_vertex(in); }
    float3 morph_position(float3 p, uint vid) const { return p; }
    float3 morph_normal(float3 n, uint vid) const { return n; }
    float4 morph_color(float4 c, uint vid) const { return c; }
";

const WGSL_COMPAT_ENTRY: &str = "
fragment float4 fs_custom(FIn fin [[stage_in]],
                          bool front_facing [[front_facing]],
                          constant Frame &frame [[buffer(0)]],
                          constant Draw &draw [[buffer(1)]],
                          constant ThreersUserData &user [[buffer(2)]],
                          const device float *s0 [[buffer(3)]],
                          const device float *s1 [[buffer(4)]],
                          const device float *s2 [[buffer(5)]],
                          constant uint4 &lens [[buffer(6)]],
                          depth2d_array<float> shadow_map [[texture(6)]],
                          depth2d_array<float> spot_map [[texture(7)]],
                          depth2d_array<float> point_map [[texture(8)]],
                          sampler shadow_samp [[sampler(11)]],
                          texturecube<float> env_map [[texture(5)]],
                          sampler env_samp [[sampler(10)]],
                          texture2d<float> env_atlas [[texture(15)]],
                          sampler env_atlas_samp [[sampler(13)]],
                          texture2d_array<float> ss_color [[texture(29)]],
                          sampler ss_samp [[sampler(14)]],
                          texture2d<float> ss_depth [[texture(30)]],
                          array<texture2d<float>, 4> u_tex [[texture(16)]],
                          sampler u_samp [[sampler(15)]],
                          array<texture3d<float>, 2> u_vol [[texture(20)]],
                          sampler u_vol_samp [[sampler(9)]]) {
    ShadowMaps maps = { shadow_map, spot_map, point_map, shadow_samp };
    ThreersEnv env = { env_map, env_samp, env_atlas, env_atlas_samp };
    ThreersUser ut = { u_tex, u_samp };
    ThreersVolumes vt = { u_vol, u_vol_samp };
    ThreersScreen ss = { ss_color, ss_samp, ss_depth,
                         min(fin.view_index + frame.rect_counts.w, ss_color.get_array_size() - 1u) };
    ThreersWgsl sh = { &frame, &draw, threers_wgsl_frame(frame, fin.view_index),
                       threers_wgsl_mesh(draw), user, s0, s1, s2, lens, &maps, &env, &ss, &ut, &vt };
    float shadow_cache[16];
    sh.shadow_cache = shadow_cache;
    for (uint r = 0; r < 16; r++) { shadow_cache[r] = 1.0; }
    VsOut vin = threers_vsout(fin, frame, draw);
    return threers_raw_out(sh.fs_main($ARGS), frame);
}
";

const WGSL_COMPAT_VERTEX_ENTRY: &str = "
vertex WgslVaryings vs_custom_entry(uint vid [[vertex_id]],
                                    uint iid [[instance_id]],
                                    device const Vertex *verts [[buffer(0)]],
                                    constant Frame &frame [[buffer(1)]],
                                    constant Draw &draw [[buffer(2)]],
                                    device const float4x4 *instances [[buffer(3)]],
                                    device const float4 *custom [[buffer(10)]],
                                    constant ThreersUserData &user [[buffer(11)]],
                                    const device float *s0 [[buffer(12)]],
                                    const device float *s1 [[buffer(13)]],
                                    const device float *s2 [[buffer(14)]],
                                    constant uint4 &lens [[buffer(15)]],
                                    array<texture2d<float>, 4> u_tex [[texture(16)]],
                                    sampler u_samp [[sampler(15)]],
                                    array<texture3d<float>, 2> u_vol [[texture(20)]],
                                    sampler u_vol_samp [[sampler(9)]]) {
    // The vertex stage has no shadow maps bound; it never reads them.
    ThreersUser ut = { u_tex, u_samp };
    ThreersVolumes vt = { u_vol, u_vol_samp };
    ThreersWgsl sh = { &frame, &draw, threers_wgsl_frame(frame, 0u),
                       threers_wgsl_mesh(draw), user, s0, s1, s2, lens, nullptr, nullptr, nullptr, &ut, &vt };
    Vertex v = verts[vid];
    bool instanced = (draw.flags.w & 1u) == 1u;
    float4x4 im = instanced ? instances[iid] : float4x4(1.0);
    $INPUT vin;
    vin.position = float3(v.position);
    vin.normal = float3(v.normal);
    vin.uv = v.uv;
    vin.color = float4(float3(v.color), v.alpha);
    vin.tangent = float4(0.0);
    THREERS_INSTANCED_FIELDS
    VsOut o = sh.vs_custom($ARGS);
    WgslVaryings w;
    w.clip_pos = o.clip_pos;
    w.world_pos = o.world_pos;
    w.world_normal = o.world_normal;
    w.uv = o.uv;
    w.view_z = o.view_z;
    w.vertex_color = o.vertex_color;
    w.clip_zw = o.clip_zw;
    w.proj_uv = o.proj_uv;
    w.world_tangent = o.world_tangent;
    w.local_pos = o.local_pos;
    w.uv12 = o.uv12;
    w.uv3 = o.uv3;
    return w;
}
";

const WGSL_COMPAT_VARYINGS_ENTRY: &str = "
fragment float4 fs_custom(WgslVaryings w [[stage_in]],
                          bool front_facing [[front_facing]],
                          constant Frame &frame [[buffer(0)]],
                          constant Draw &draw [[buffer(1)]],
                          constant ThreersUserData &user [[buffer(2)]],
                          const device float *s0 [[buffer(3)]],
                          const device float *s1 [[buffer(4)]],
                          const device float *s2 [[buffer(5)]],
                          constant uint4 &lens [[buffer(6)]],
                          depth2d_array<float> shadow_map [[texture(6)]],
                          depth2d_array<float> spot_map [[texture(7)]],
                          depth2d_array<float> point_map [[texture(8)]],
                          sampler shadow_samp [[sampler(11)]],
                          texturecube<float> env_map [[texture(5)]],
                          sampler env_samp [[sampler(10)]],
                          texture2d<float> env_atlas [[texture(15)]],
                          sampler env_atlas_samp [[sampler(13)]],
                          texture2d_array<float> ss_color [[texture(29)]],
                          sampler ss_samp [[sampler(14)]],
                          texture2d<float> ss_depth [[texture(30)]],
                          array<texture2d<float>, 4> u_tex [[texture(16)]],
                          sampler u_samp [[sampler(15)]],
                          array<texture3d<float>, 2> u_vol [[texture(20)]],
                          sampler u_vol_samp [[sampler(9)]]) {
    ShadowMaps maps = { shadow_map, spot_map, point_map, shadow_samp };
    ThreersEnv env = { env_map, env_samp, env_atlas, env_atlas_samp };
    ThreersUser ut = { u_tex, u_samp };
    ThreersVolumes vt = { u_vol, u_vol_samp };
    ThreersScreen ss = { ss_color, ss_samp, ss_depth,
                         min(0u + frame.rect_counts.w, ss_color.get_array_size() - 1u) };
    ThreersWgsl sh = { &frame, &draw, threers_wgsl_frame(frame, 0u),
                       threers_wgsl_mesh(draw), user, s0, s1, s2, lens, &maps, &env, &ss, &ut, &vt };
    float shadow_cache[16];
    sh.shadow_cache = shadow_cache;
    for (uint r = 0; r < 16; r++) { shadow_cache[r] = 1.0; }
    VsOut vin;
    vin.clip_pos = w.clip_pos;
    vin.world_pos = w.world_pos;
    vin.world_normal = w.world_normal;
    vin.uv = w.uv;
    vin.view_z = w.view_z;
    vin.vertex_color = w.vertex_color;
    vin.clip_zw = w.clip_zw;
    vin.proj_uv = w.proj_uv;
    vin.world_tangent = w.world_tangent;
    vin.local_pos = w.local_pos;
    vin.uv12 = w.uv12;
    vin.uv3 = w.uv3;
    return threers_raw_out(sh.fs_main($ARGS), frame);
}
";

/// Every top-level `const` of wgpu's shader preamble, as MSL — the `MAT_*`
/// kinds and `FLAG_*` bits a WGSL material tests — translated once.
fn preamble_constants() -> &'static str {
    static CONSTS: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CONSTS.get_or_init(|| {
        let source = crate::renderer::shader::legacy_shader_source();
        let mut out = String::new();
        let mut rest = source;
        while let Some(at) = rest.find("\nconst ") {
            let tail = &rest[at + 1..];
            let Some(end) = tail.find(';') else { break };
            if let Ok(items) = parse_items(&tail[..=end]) {
                for item in &items {
                    out.push_str(&item.to_msl());
                    out.push('\n');
                }
            }
            rest = &tail[end..];
        }
        out
    })
}

/// Names the Metal environment provides itself, which the preamble's WGSL
/// versions (texture reads, bindings) must not replace.
const PROVIDED: [&str; 22] = [
    "sample_env_cube",
    "sample_env_cube_unrotated",
    "s0",
    "s1",
    "s2",
    "s_len",
    "framebuffer_encode",
    "apply_tone_mapping",
    "apply_fog",
    "standard_vertex",
    "standard_vertex_morphed",
    "morph_position",
    "morph_normal",
    "morph_color",
    "shadow_compare",
    "shadow_tap",
    "shadow_at",
    "light_shadow_record",
    "light_shadow",
    "shadow_factor_dir",
    "shadow_factor_spot",
    "shadow_factor_point",
];

/// The preamble's functions, by name, parsed once.
fn preamble_functions() -> &'static std::collections::HashMap<String, Item> {
    static FNS: std::sync::OnceLock<std::collections::HashMap<String, Item>> =
        std::sync::OnceLock::new();
    FNS.get_or_init(|| {
        let source = crate::renderer::shader::legacy_shader_source();
        let preamble = source.split("@fragment").next().unwrap_or(source);
        let mut map = std::collections::HashMap::new();
        let mut rest = preamble;
        while let Some(at) = rest.find("\nfn ") {
            let tail = &rest[at + 1..];
            // The body: from the first `{` to its match, by characters (the
            // preamble's comments hold no braces that matter here).
            let Some(open) = tail.find('{') else { break };
            let mut depth = 0;
            let mut end = None;
            for (k, c) in tail[open..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(open + k + 1);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let Some(end) = end else { break };
            if let Ok(mut items) = parse_items(&tail[..end]) {
                if let Some(Item::Fn(f)) = items.pop() {
                    if !PROVIDED.contains(&f.name.as_str()) {
                        map.insert(f.name.clone(), Item::Fn(f));
                    }
                }
            }
            rest = &tail[end..];
        }
        map
    })
}

/// The preamble functions `items` reaches, callees before callers.
fn preamble_functions_used(items: &[Item]) -> Vec<&'static Item> {
    let preamble = preamble_functions();
    let defined: std::collections::HashSet<&str> = items
        .iter()
        .filter_map(|i| match i {
            Item::Fn(f) => Some(f.name.as_str()),
            _ => None,
        })
        .collect();
    let calls = |body: &[Tok]| -> Vec<String> {
        body.windows(2)
            .filter_map(|w| match (&w[0], &w[1]) {
                (Tok::Ident(n), t) if t.is("(") => Some(n.clone()),
                _ => None,
            })
            .collect()
    };
    let mut order: Vec<&'static Item> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    fn visit(
        name: &str,
        preamble: &'static std::collections::HashMap<String, Item>,
        seen: &mut std::collections::HashSet<String>,
        order: &mut Vec<&'static Item>,
        calls: &dyn Fn(&[Tok]) -> Vec<String>,
    ) {
        if !seen.insert(name.to_string()) {
            return;
        }
        if let Some(item @ Item::Fn(f)) = preamble.get(name) {
            for callee in calls(&f.body) {
                visit(&callee, preamble, seen, order, calls);
            }
            order.push(item);
        }
    }
    for item in items {
        if let Item::Fn(f) = item {
            for callee in calls(&f.body) {
                if !defined.contains(callee.as_str()) {
                    visit(&callee, preamble, &mut seen, &mut order, &calls);
                }
            }
        }
    }
    order
}

/// wgpu's struct names where Metal's differ, inside the shader's namespace.
const WGSL_COMPAT_ALIASES: &str = "
typedef WgslFrame FrameUniforms;
typedef WgslMesh MeshUniforms;
";

/// The output helpers the fragment entries share.
const OUTPUT_HELPERS: &str = "
/// What a fragment returns, as the attachment should store it: standing in
/// for a `Unorm` target (`view_info.y` bit 1), the value is the stored byte,
/// so it is decoded for the sRGB attachment to encode back.
static inline float4 threers_raw_out(float4 c, constant Frame &frame) {
    return (frame.view_info.y & 2u) != 0u ? float4(srgb_decode(c.rgb), c.a) : c;
}

/// three.js's canvas blend: on the sRGB-*encoded* values, as wgpu blends an
/// 8-bit sRGB target — normal (`SrcAlpha, OneMinusSrcAlpha`) or additive
/// (`SrcAlpha, One`). The attachment encodes on write, so decode the result.
static inline float4 threers_encoded_blend(float4 src, float4 dst, bool additive) {
    float a = saturate(src.a);
    float3 s = srgb_encode(src.rgb);
    float3 d = srgb_encode(dst.rgb);
    float3 e = additive ? d + s * a : s * a + d * (1.0 - a);
    return float4(srgb_decode(e), additive ? saturate(dst.a + a) : a + dst.a * (1.0 - a));
}
";

/// `fs_custom` plus `fs_custom_eb` and `fs_custom_eb_add`: the same entry
/// reading the destination and blending in encoded values itself, for a
/// blended draw into an sRGB target (the pipeline turns hardware blending off).
fn with_encoded_variants(entry: &str) -> String {
    let variant = |name: &str, additive: bool| -> String {
        let mut e = entry.replacen(
            "fragment float4 fs_custom(",
            &format!("fragment float4 {name}("),
            1,
        );
        // The destination, as one more parameter.
        if let Some(at) = e.find(") {") {
            e.insert_str(
                at,
                ",\n                          float4 threers_dst [[color(0)]]",
            );
        }
        // The value returned, blended.
        if let Some(at) = e.rfind("return ") {
            let end = e[at..].find(';').map(|k| at + k).unwrap_or(e.len());
            let value = e[at + 7..end].to_string();
            e.replace_range(
                at..end,
                &format!("return threers_encoded_blend({value}, threers_dst, {additive})"),
            );
        }
        e
    };
    format!(
        "{entry}{}{}",
        variant("fs_custom_eb", false),
        variant("fs_custom_eb_add", true)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rim() -> SharedShader {
        crate::shader! {
            const GAIN: f32 = 2.0;
            fn rim(n: vec3f, v: vec3f) -> f32 {
                let f = 1.0 - max(dot(n, v), 0.0);
                return f * f * GAIN;
            }
            fn shade(s: Shading) -> vec4f {
                var c = data(0u32).xyz * ambient_light();
                for (var i = 0u32; i < light_count(); i += 1u32) {
                    c += light_color(i) * max(dot(s.normal, light_direction(i)), 0.0);
                }
                if (s.uv.x < 0.01) { discard; }
                let r: f32 = rim(s.normal, s.view_dir);
                return vec4f(c + vec3f(r), opacity());
            }
        }
    }

    #[test]
    fn wgsl_keeps_the_shared_syntax() {
        let w = rim().wgsl().unwrap();
        assert!(w.contains("fn shade(s: Shading) -> vec4f {"), "{w}");
        assert!(w.contains("var c = data(0u)"), "{w}");
        assert!(
            w.contains("for (var i = 0u; i < light_count(); i += 1u)"),
            "{w}"
        );
        assert!(w.contains("const GAIN: f32 = 2.0;"), "{w}");
        assert!(w.contains("discard;"), "{w}");
        assert!(w.contains("fn fs_main(in: VsOut"), "{w}");
    }

    #[test]
    fn msl_declares_types_and_members() {
        let m = rim().msl().unwrap();
        assert!(m.contains("float4 shade(Shading s) {"), "{m}");
        assert!(m.contains("float rim(float3 n, float3 v) {"), "{m}");
        assert!(m.contains("auto c = data(0u)"), "{m}");
        assert!(
            m.contains("const float r = rim(s.normal, s.view_dir)"),
            "{m}"
        );
        assert!(
            m.contains("for (auto i = 0u; i < light_count(); i += 1u)"),
            "{m}"
        );
        assert!(m.contains("constant float GAIN = 2.0;"), "{m}");
        assert!(m.contains("discard_fragment();"), "{m}");
        assert!(m.contains("float4(c + float3(r), opacity())"), "{m}");
    }

    #[test]
    fn lights_shadows_and_maps_reach_both_languages() {
        let lit = crate::shader! {
            fn shade(s: Shading) -> vec4f {
                var c = tex0(s.uv).xyz * hemisphere_light(s.normal);
                for (var i = 0u32; i < point_light_count(); i += 1u32) {
                    c += point_light_color(i, s) * point_shadow(i, s);
                }
                return vec4f(c * directional_shadow(0u32, s), 1.0);
            }
        };
        let (w, m) = (lit.wgsl().unwrap(), lit.msl().unwrap());
        for name in [
            "tex0(",
            "hemisphere_light(",
            "point_light_color(",
            "point_shadow(",
            "directional_shadow(",
        ] {
            assert!(w.contains(&format!("fn {name}")), "WGSL lacks {name}");
            assert!(m.contains(name), "MSL lacks {name}");
        }
    }

    #[test]
    fn a_frame_record_is_bound_by_reference() {
        let m = wgsl_material_to_msl(
            "@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
                 let rec = frame.shadows[0];\n\
                 let w = frame.shadows[0].params.x;\n\
                 var c = 0.0;\n\
                 for (var k = 0u; k < frame.light_counts.w; k = k + 1u) {\n\
                     let h = frame.hemi_lights[k];\n\
                     c = c + h.sky_color.r;\n\
                 }\n\
                 return vec4<f32>(rec.vp[1][0].x + w + c);\n\
             }",
            None,
        )
        .unwrap();
        assert!(m.contains("constant auto &rec = frame.shadows[0]"), "{m}");
        // The hemisphere lights too, through a pointer into Metal's frame: an
        // array member of the frame copy, indexed by `k`, would be in stack
        // memory.
        assert!(m.contains("constant auto &h = frame.hemi_lights[k]"), "{m}");
        assert!(m.contains("constant HemiLight *hemi_lights;"), "{m}");
        // A field of one is a value as before.
        assert!(
            m.contains("const auto w = frame.shadows[0].params.x"),
            "{m}"
        );
    }

    #[test]
    fn a_shader_without_an_entry_point_is_an_error() {
        let e = crate::shader! { fn f(x: f32) -> f32 { return x; } }
            .wgsl()
            .unwrap_err();
        assert!(e.contains("shade"), "{e}");
    }
}
