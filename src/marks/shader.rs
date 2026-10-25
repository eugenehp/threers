//! WGSL for every kind of mark, and the assembly of a module from it.
//!
//! A module is the shared preamble, then the hooks (the caller's or the
//! identity ones), then the kind's own stages. Custom columns are declared
//! with the width they actually have — a one-float column is a `f32` input,
//! not a `vec4` — and widened to `vec4` in the shader, which keeps the vertex
//! formats and the shader's input types in exact agreement.

use super::hooks::ShaderHooks;

/// The kinds of mark, one pipeline family each.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarkKind {
    /// Circles a fixed number of pixels or world units across.
    Point,
    /// Wide lines and arcs between two points.
    Segment,
    /// Instanced geometry — extruded columns, icons in 3D.
    Shape,
    /// Text glyphs on camera-facing quads.
    Glyph,
    /// Gaussian kernels summed into a density field.
    Splat,
    /// Triangulated polygons, extruded per feature.
    Polygon,
    /// Atlas images on camera-facing quads.
    Sprite,
    /// Atlas images on quads in the world.
    Image,
    /// A kind defined outside threers by a [`CustomShader`]; the number is
    /// its source's hash, so equal sources share pipelines.
    Custom(u64),
}

/// The WGSL of a mark kind defined outside threers.
///
/// `source` is the kind's own part of the module: `vs_main`, `fs_main` and
/// `fs_pick` (pick output `vec2<u32>(layer.ids.x, row)`), written against
/// [`PREAMBLE`] — the frame and layer uniforms, `MarkInstance`,
/// `mark_relative`, `mark_size_px`, `mark_offset_px`, `mark_lift_depth`,
/// `mark_hidden`, `mark_decode`, `mark_out` — and the hooks, which it should
/// call (`hook_instance`, `hook_color`) so users' hooks apply to it as to
/// every other mark.
///
/// Custom columns are spliced in where the source writes `/*CUSTOM_IN*/`
/// (inside its vertex input struct), `/*CUSTOM0*/` and `/*CUSTOM1*/` (the
/// `vec4<f32>` values, zero-padded), starting at shader location
/// `custom_location`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CustomShader {
    /// Named in compile errors.
    pub name: String,
    pub source: String,
    pub custom_location: u32,
}

impl CustomShader {
    pub fn new(name: impl Into<String>, source: impl Into<String>, custom_location: u32) -> Self {
        Self {
            name: name.into(),
            source: source.into(),
            custom_location,
        }
    }

    /// The kind marks drawn with this shader report from
    /// [`Mark::kind`](super::Mark::kind).
    pub fn kind(&self) -> MarkKind {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut h);
        MarkKind::Custom(h.finish())
    }

    /// The whole module, preamble and hooks included, with custom columns
    /// `custom` floats wide (0 for none) — what the GPU compiles.
    pub fn module_source(&self, hooks: &ShaderHooks, custom: [u32; 2]) -> String {
        assemble(&self.source, self.custom_location, hooks, custom, false)
    }

    /// [`module_source`](Self::module_source) for a layer whose positions
    /// are local: the `*_lo` inputs dropped and read as zero.
    pub fn module_source_local(&self, hooks: &ShaderHooks, custom: [u32; 2]) -> String {
        assemble(&self.source, self.custom_location, hooks, custom, true)
    }

    /// Compile-check against `hooks` with both custom columns four wide,
    /// without a GPU. Always `Ok` on `wasm32`, where naga is not built in.
    pub fn validate(&self, hooks: &ShaderHooks) -> Result<(), String> {
        let src = self.module_source(hooks, [4, 4]);
        let offset = src
            .lines()
            .position(|l| l.starts_with("// ---- mark ----"))
            .map_or(1, |i| i + 2);
        crate::renderer::validate_custom_shader(&src, 0).map_err(|e| {
            format!(
                "{} mark: {e}\n(its source starts at line {offset})",
                self.name
            )
        })
    }
}

/// The kinds whose module includes the hooks, for validation.
pub(crate) const HOOKED_KINDS: &[MarkKind] = &[
    MarkKind::Point,
    MarkKind::Segment,
    MarkKind::Shape,
    MarkKind::Glyph,
    MarkKind::Splat,
    MarkKind::Polygon,
    MarkKind::Sprite,
    MarkKind::Image,
];

/// Uniforms and helpers every module starts with.
/// The WGSL every mark module starts with: uniforms, `MarkInstance` and the
/// helpers a [`CustomShader`] builds on.
pub const PREAMBLE: &str = r#"
struct MarkFrame {
    view_proj : mat4x4<f32>,
    origin_hi : vec4<f32>,
    origin_lo : vec4<f32>,
    // width, height, 1/width, 1/height — in pixels.
    viewport  : vec4<f32>,
    // x: pixels per world unit at clip w = 1; y: 1 when the target wants
    // sRGB-encoded values written to it.
    misc      : vec4<f32>,
    // xyz: direction towards the light, world space.
    light     : vec4<f32>,
    // The camera: a position with w = 1, or the direction towards it with
    // w = 0 (orthographic).
    eye       : vec4<f32>,
    // x: how far past its edge a mark's quad reaches, in pixels — half a
    // pixel, where coverage ends, or a whole one when multisampled, whose
    // samples sit up to half a pixel from the centre that shades them.
    aa        : vec4<f32>,
    // x: the clip w a line facing the camera is cut at, behind it.
    clip      : vec4<f32>,
    // x: 1 when drawn upside down (`MarkView::flip_y`).
    raster    : vec4<f32>,
    // The scene's lights (`MarkView::scene_lights`): counts (points,
    // directionals, 1 when set), ambient colour, then each light.
    scene_counts : vec4<f32>,
    scene_ambient : vec4<f32>,
    point_pos : array<vec4<f32>, 4>,
    point_color : array<vec4<f32>, 4>,
    point_att : array<vec4<f32>, 4>,
    dir_dir : array<vec4<f32>, 4>,
    dir_color : array<vec4<f32>, 4>,
    // The scene's shadow map: its matrix (origin-relative to the light's
    // clip space, GL depth), colour (sRGB) and strength, x 1 when on.
    shadow_matrix : mat4x4<f32>,
    shadow_color : vec4<f32>,
    shadow_on : vec4<f32>,
};

struct MarkLayer {
    params : array<vec4<f32>, 4>,
    style  : vec4<f32>,
    style2 : vec4<f32>,
    style3 : vec4<f32>,
    stroke : vec4<f32>,
    ids    : vec4<u32>,
    // x: how far towards the camera to depth-test, in drawn sizes.
    depth  : vec4<f32>,
    // x: 1 when the layer takes the scene's shadows.
    shadow : vec4<f32>,
    // What `mark_relative` subtracts, split into high and low parts: the
    // view's origin for absolute positions, and minus the offset from the
    // view's origin to the layer's own for local ones.
    origin_hi : vec4<f32>,
    origin_lo : vec4<f32>,
    // The layer's clip rectangle on the ground, relative: (x0, z0, x1, z1).
    clip : vec4<f32>,
    // A fourth vec4 of the kind's own.
    extra : vec4<f32>,
    // The highlighted row's colour, linear; alpha is how much of it.
    highlight : vec4<f32>,
    // A planet's centre, relative, and radius: what is behind it is culled.
    // Radius 0 for none.
    horizon : vec4<f32>,
    // x: specular reflectance, y: shininess (lit kinds' `Lighting`).
    material : vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame : MarkFrame;
@group(0) @binding(1) var shadow_map : texture_2d<f32>;
@group(0) @binding(2) var shadow_sampler : sampler;
@group(1) @binding(0) var<uniform> layer : MarkLayer;

/// deck.gl's shadow depth: the light's clip depth of relative point `p`,
/// 0 to 1, and where it falls on the shadow map.
fn mark_shadow_coord(p : vec3<f32>) -> vec3<f32> {
    let c = frame.shadow_matrix * vec4<f32>(p, 1.0);
    return (c.xyz / c.w + 1.0) * 0.5;
}

/// deck.gl's `shadow_filterShadowColor`: colour `c` (linear) darkened where
/// the shadow map has something nearer the light than relative point `p`,
/// mixed on sRGB values, as a translucent shadow colour over it.
fn mark_shadow(c : vec4<f32>, p : vec3<f32>) -> vec4<f32> {
    if (frame.shadow_on.x < 0.5 || layer.shadow.x < 0.5) {
        return c;
    }
    let s = mark_shadow_coord(p);
    // The map's rows run down from its top: GL's y up.
    let rgba = textureSampleLevel(shadow_map, shadow_sampler, vec2<f32>(s.x, 1.0 - s.y), 0.0);
    let z = dot(rgba, vec4<f32>(1.0, 1.0 / 255.0, 1.0 / 65025.0, 1.0 / 16581375.0));
    let k = smoothstep(0.001, 0.01, s.z - z) * frame.shadow_color.a;
    let a = k + c.a * (1.0 - k);
    let rgb = mix(mark_linear_to_srgb(c.rgb), frame.shadow_color.rgb, k / max(a, 1e-6));
    return vec4<f32>(mark_srgb_to_linear(rgb), a);
}

/// deck.gl's shadow pass colour: the light's depth of `p`, packed into
/// RGBA bytes.
fn mark_shadow_depth(p : vec3<f32>) -> vec4<f32> {
    let z = mark_shadow_coord(p).z;
    var rgba = fract(z * vec4<f32>(1.0, 255.0, 65025.0, 16581375.0));
    rgba -= rgba.yzww * vec4<f32>(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 0.0);
    return rgba;
}

struct MarkInstance {
    position  : vec3<f32>,
    position2 : vec3<f32>,
    size      : f32,
    color     : vec4<f32>,
    color2    : vec4<f32>,
    custom0   : vec4<f32>,
    custom1   : vec4<f32>,
    visible   : f32,
    index     : u32,
};

/// An `f64` position, split into high and low parts, relative to the origin.
///
/// One expression for absolute and local layers alike, with the difference
/// in the uniforms: a branch here let Metal's compiler reassociate the
/// subtraction and lose the precision it exists for.
fn mark_relative(hi : vec3<f32>, lo : vec3<f32>) -> vec3<f32> {
    let d_hi = hi - layer.origin_hi.xyz;
    let d_lo = lo - layer.origin_lo.xyz;
    // A barrier: `ids.w` is always 0, but the compiler cannot know that, so
    // it has to produce `d_hi` as a value before adding `d_lo` — it can no
    // longer fold the two into `(hi + lo) - (origin_hi + origin_lo)`, which
    // rounds away what the split exists to keep. Without it, unrelated code
    // elsewhere in a vertex stage was enough to change the folding on Metal.
    return select(d_hi, vec3<f32>(0.0), layer.ids.w != 0u) + d_lo;
}

fn mark_srgb_to_linear(c : vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

fn mark_linear_to_srgb(c : vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    return select(1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3<f32>(0.0031308));
}

/// Colour columns hold sRGB bytes, as colours in data usually are.
fn mark_decode(c : vec4<f32>) -> vec4<f32> {
    return vec4<f32>(mark_srgb_to_linear(c.rgb), c.a);
}

/// Linear in, whatever the target stores out.
///
/// A layer with `tone_mapped` set (`depth.w`) first takes the scene's tone
/// mapping (`frame.misc.z` the mode, `.w` the exposure) — the renderer's own
/// curves, spliced in after this preamble.
fn mark_out(c : vec4<f32>) -> vec4<f32> {
    var rgb = c.rgb;
    if (layer.depth.w > 0.5) {
        rgb = mark_tone(rgb);
    }
    if (frame.misc.y > 0.5) {
        return vec4<f32>(mark_linear_to_srgb(rgb), c.a);
    }
    return vec4<f32>(rgb, c.a);
}

/// A size in the layer's units, in pixels at clip depth `w`, clamped to the
/// layer's pixel range. `style`: (scale, min px, max px, 1 = world units).
fn mark_size_px(size : f32, w : f32) -> f32 {
    // With focal pixels (`ids.z` bit 2) and a focal plane (`frame.aa.y`),
    // the size and its clamps are taken at that plane, then scaled by
    // perspective — deck.gl's `project_pixel_size_to_clipspace`.
    let focal = frame.aa.y;
    let at_focal = (layer.ids.z & 4u) != 0u && focal > 0.0;
    let measured_w = select(w, focal, at_focal);
    var s = size * layer.style.x;
    if (layer.style.w > 0.5) {
        s = s * frame.misc.x / max(measured_w, 1e-6);
    }
    s = clamp(s, layer.style.y, layer.style.z);
    return select(s, s * focal / max(w, 1e-6), at_focal);
}

/// Vertex `vid` of a two-triangle quad, as a corner index 0..4 (CCW from the
/// bottom left). Arithmetic rather than an indexed constant array, which not
/// every WGSL compiler allows with a runtime index.
fn mark_quad_corner(vid : u32) -> u32 {
    var k = vid % 6u;
    if (k == 3u) {
        k = 0u;
    } else if (k >= 4u) {
        k = k - 2u;
    }
    return k;
}

/// Corner `k` in [-1, 1]².
fn mark_corner_signed(k : u32) -> vec2<f32> {
    return vec2<f32>(select(-1.0, 1.0, k == 1u || k == 2u), select(-1.0, 1.0, k >= 2u));
}

/// A clip position outside the view volume: the triangle is dropped.
fn mark_hidden() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 2.0, 1.0);
}

/// `clip` with the depth of `p` moved towards the camera by `px` screen
/// pixels' worth of world, times the layer's lift, so a sprite resting on a
/// surface is not cut off where the surface comes nearer than its anchor.
fn mark_lift_depth(clip : vec4<f32>, p : vec3<f32>, px : f32) -> vec4<f32> {
    let r = px * layer.depth.x * clip.w / max(frame.misc.x, 1e-6);
    let to_eye = frame.eye.xyz - p * frame.eye.w;
    let d = length(to_eye);
    if (r <= 0.0 || d < 1e-6) {
        return clip;
    }
    // Never to or past a perspective eye.
    let s = select(r, min(r, d * 0.9), frame.eye.w > 0.5);
    let c = frame.view_proj * vec4<f32>(p + to_eye * (s / d), 1.0);
    if (c.w <= 1e-6) {
        return clip;
    }
    return vec4<f32>(clip.xy, clamp(c.z / c.w, 0.0, 1.0) * clip.w, clip.w);
}

/// The layer's draw order (`depth.y`, already scaled) taken off the depth,
/// so coplanar layers — water over land use under roads — stack in order
/// instead of fighting.
fn mark_order(clip : vec4<f32>) -> vec4<f32> {
    // Towards the camera: smaller depth, or larger with reversed depth
    // (`frame.light.w`).
    let toward = select(1.0, -1.0, frame.light.w > 0.5);
    let xy = mark_snap(clip);
    // Flipped last, after every screen-space step and the snap.
    let y = select(xy.y, -xy.y, frame.raster.x > 0.5);
    return vec4<f32>(xy.x, y, clip.z - toward * layer.depth.y * clip.w, clip.w);
}

/// `clip.xy` snapped to the view's sub-pixel grid (`frame.clip.y` steps a
/// pixel, in window coordinates), the bias `frame.clip.zw` kept off it.
fn mark_snap(clip : vec4<f32>) -> vec2<f32> {
    let steps = frame.clip.y;
    if (steps <= 0.0 || clip.w <= 0.0) {
        return clip.xy;
    }
    let bias = frame.clip.zw;
    let start = frame.aa.zw;
    let px = (clip.xy / clip.w * 0.5 + 0.5) * frame.viewport.xy + start - bias;
    let snapped = round(px * steps) / steps + bias - start;
    return ((snapped / frame.viewport.xy - 0.5) * 2.0) * clip.w;
}

/// Whether relative point `p` is outside the layer's clip rectangle on the
/// ground (`depth.z` set) — a tile's edge.
fn mark_clipped(p : vec3<f32>) -> bool {
    return layer.depth.z > 0.5
        && (p.x < layer.clip.x || p.z < layer.clip.y || p.x > layer.clip.z || p.z > layer.clip.w);
}

/// Whether relative point `p` is on the far side of the layer's horizon
/// sphere from the eye — hidden by the planet, for marks (labels, icons)
/// that do not depth-test against it.
fn mark_beyond_horizon(p : vec3<f32>) -> bool {
    let c = layer.horizon.xyz;
    let r = layer.horizon.w;
    let pc = p - c;
    // A perspective eye sees past the horizon plane, dot(p - c, eye - c) = r²;
    // an orthographic one sees the hemisphere facing it.
    let seen = select(
        dot(pc, frame.eye.xyz) >= 0.0,
        dot(pc, frame.eye.xyz - c) >= r * r * 0.999,
        frame.eye.w > 0.5,
    );
    return r > 0.0 && !seen;
}

/// Inside the globe a layer has (`horizon`), well under its surface: a
/// mesh anchored at the earth's centre, which no horizon hides.
fn mark_inside_globe(p : vec3<f32>) -> bool {
    let pc = p - layer.horizon.xyz;
    let r = layer.horizon.w;
    return r > 0.0 && dot(pc, pc) < r * r * 0.98;
}

/// Outside the clip rectangle or beyond the horizon.
fn mark_culled(p : vec3<f32>) -> bool {
    return mark_clipped(p) || mark_beyond_horizon(p);
}

/// Edge coverage as drawn: `c` itself, or all-or-nothing for a layer with
/// antialiasing off (`ids.z` bit 0).
fn mark_cov(c : f32) -> f32 {
    return select(c, step(0.5, c), mark_aliased());
}

/// Antialiasing is off for this layer (`ids.z` bit 0). Its quads then cover
/// exactly the shape, unpadded, so the rasterizer's own fill rule decides
/// the edge pixels — as it does for deck.gl's quads.
fn mark_aliased() -> bool {
    return (layer.ids.z & 1u) != 0u;
}

/// `rgb` lit with [`Lighting`](super::Lighting) for unit normal `n`
/// (`style2.y` > 0: diffuse; `style2.z`: on sRGB values; `stroke` and
/// `extra`: the lights), or `None`-style ambient lighting, `fallback`.
/// One light's share under luma.gl's physically based model
/// (`calculateFinalColor`): Schlick Fresnel, Smith occlusion, GGX
/// distribution, Lambert diffuse.
fn mark_pbr_light(ndotl : f32, ndotv : f32, ndoth : f32, vdoth : f32, base : vec3<f32>, metallic : f32, alpha : f32, light : f32) -> vec3<f32> {
    let f0 = vec3<f32>(0.04);
    let diffuse_color = base * (vec3<f32>(1.0) - f0) * (1.0 - metallic);
    let specular_color = mix(f0, base, metallic);
    let r90 = vec3<f32>(clamp(max(max(specular_color.r, specular_color.g), specular_color.b) * 25.0, 0.0, 1.0));
    let f = specular_color + (r90 - specular_color) * pow(clamp(1.0 - vdoth, 0.0, 1.0), 5.0);
    let r2 = alpha * alpha;
    let att_l = 2.0 * ndotl / (ndotl + sqrt(r2 + (1.0 - r2) * ndotl * ndotl));
    let att_v = 2.0 * ndotv / (ndotv + sqrt(r2 + (1.0 - r2) * ndotv * ndotv));
    let g = att_l * att_v;
    let rs = r2;
    let fd = (ndoth * rs - ndoth) * ndoth + 1.0;
    let d = rs / (3.14159265 * fd * fd);
    let diffuse = (vec3<f32>(1.0) - f) * diffuse_color / 3.14159265;
    let spec = f * g * d / (4.0 * ndotl * ndotv);
    return ndotl * light * (diffuse + spec);
}

/// luma.gl's `pbrMaterial` with an ambient light (`style2.x`) and the two
/// directional ones: linear in, linear out.
fn mark_pbr(rgb : vec3<f32>, n : vec3<f32>, p : vec3<f32>) -> vec3<f32> {
    let v = normalize(frame.eye.xyz - p * frame.eye.w);
    let metallic = layer.material.x;
    let alpha = layer.material.y * layer.material.y;
    let ndotv = clamp(abs(dot(n, v)), 0.001, 1.0);
    // The ambient light, as luma.gl takes it: along the normal.
    var c = mark_pbr_light(1.0, ndotv, 0.0, 1.0, rgb, metallic, alpha, layer.style2.x);
    for (var k = 0; k < 2; k++) {
        let l4 = select(layer.extra, layer.stroke, k == 0);
        let l = l4.xyz;
        let h = normalize(l + v);
        c += mark_pbr_light(
            clamp(dot(n, l), 0.001, 1.0),
            ndotv,
            clamp(dot(n, h), 0.0, 1.0),
            clamp(dot(v, h), 0.0, 1.0),
            rgb,
            metallic,
            alpha,
            l4.w,
        );
    }
    return c;
}

fn mark_lit(rgb : vec3<f32>, n : vec3<f32>, p : vec3<f32>, fallback : f32) -> vec3<f32> {
    if (layer.material.w > 0.5) {
        return mark_pbr(rgb, n, p);
    }
    if (layer.style2.y <= 0.0) {
        return rgb * fallback;
    }
    return mark_phong(rgb, n, p, layer.style2.x, layer.style2.y, layer.style2.z > 0.5);
}

/// Blinn-Phong, as luma.gl's `lighting_getLightColor`, by the layer's two
/// lights (`stroke`, `extra`) and material: highlights only where the
/// light reaches; products on sRGB values when `srgb`.
fn mark_phong(rgb : vec3<f32>, n : vec3<f32>, p : vec3<f32>, ambient : f32, diffuse : f32, srgb : bool) -> vec3<f32> {
    let view = normalize(frame.eye.xyz - p * frame.eye.w);
    var d = 0.0;
    var spec = 0.0;
    for (var k = 0; k < 2; k++) {
        let l = select(layer.extra, layer.stroke, k == 0);
        let lambert = dot(n, l.xyz);
        if (lambert > 0.0 && layer.material.x > 0.0) {
            spec += pow(max(dot(n, normalize(l.xyz + view)), 0.0), layer.material.y) * l.w;
        }
        d += max(lambert, 0.0) * l.w;
    }
    let lit = ambient + diffuse * d;
    let shine = vec3<f32>(layer.material.x * spec);
    if (srgb) {
        return mark_srgb_to_linear(mark_linear_to_srgb(rgb) * lit + shine);
    }
    return rgb * lit + shine;
}

/// `a` to `b` by `t`, on linear values, or on sRGB ones for a layer with
/// `mix_srgb` (`ids.z` bit 1).
fn mark_mix(a : vec4<f32>, b : vec4<f32>, t : f32) -> vec4<f32> {
    if ((layer.ids.z & 2u) != 0u) {
        let rgb = mix(mark_linear_to_srgb(a.rgb), mark_linear_to_srgb(b.rgb), t);
        return vec4<f32>(mark_srgb_to_linear(rgb), mix(a.a, b.a, t));
    }
    return mix(a, b, t);
}

/// `c`, mixed towards the highlight colour when `index` is the highlighted
/// row (`ids.y` is that row plus one, 0 for none).
fn mark_highlight(c : vec4<f32>, index : u32) -> vec4<f32> {
    // ids.y: the first highlighted row plus one; ids.z >> 8: how many.
    if (layer.ids.y != 0u && index + 1u >= layer.ids.y && index + 1u < layer.ids.y + max(layer.ids.z >> 8u, 1u)) {
        // deck.gl's `picking_filterHighlightColor`: the highlight laid over
        // the colour as one translucent colour over another, on sRGB
        // values.
        let h = layer.highlight;
        let a = h.a + c.a * (1.0 - h.a);
        let k = h.a / max(a, 1e-6);
        let rgb = mix(mark_linear_to_srgb(c.rgb), mark_linear_to_srgb(h.rgb), k);
        return vec4<f32>(mark_srgb_to_linear(rgb), a);
    }
    return c;
}

/// Offset a clip position by `px` screen pixels (y up).
fn mark_offset_px(clip : vec4<f32>, px : vec2<f32>) -> vec4<f32> {
    return clip + vec4<f32>(px * 2.0 * frame.viewport.zw * clip.w, 0.0, 0.0);
}

// Fill patterns: deck.gl's FillStyleExtension (procedural distance fields
// and the background behind a pattern).
fn fill_over_background(c : vec4<f32>, bg : vec4<f32>) -> vec4<f32> {
    if (bg.a <= 0.0) {
        return c;
    }
    let a = c.a + bg.a * (1.0 - c.a);
    return vec4<f32>(mark_mix(bg, c, c.a / max(a, 1e-6)).rgb, a);
}

// Procedural fill patterns (`PolygonMarks::set_procedural`,
// `PointMarks::set_procedural`; `ids.z` bit 3): deck.gl's FillStyleExtension, its distance fields in pattern units.
// custom0: (shape, size, gap0, gap1); custom1: (angle0, angle1, count, world
// units per pattern unit, negative for pattern y along -z). Screen-space
// ramps come from the position's derivatives, taken once in uniform control
// flow (`grad`: d/dx and d/dy of the ground point, in pattern axes), and
// each field's gradient.

// A coordinate along `axis`, modulo `period`, of the absolute ground point:
// the view origin's part reduced first, for precision.
fn fill_pattern_coord(local : vec2<f32>, origin : vec2<f32>, axis : vec2<f32>, unit : f32, period : f32, offset : f32) -> f32 {
    let high = (dot(origin, axis) / unit) % period;
    let low = dot(local, axis) / unit;
    let v = high + low + offset * period;
    return v - period * floor(v / period);
}

// Coverage of a signed distance `d` (pattern units) whose gradient is the
// unit vector `g` (pattern axes): a pixel's ramp, as deck.gl's
// `fill_getScreenSpaceAlpha`.
fn fill_screen_alpha(d : f32, g : vec2<f32>, grad : vec4<f32>, unit : f32) -> f32 {
    let w = (abs(dot(g, grad.xy)) + abs(dot(g, grad.zw))) / unit;
    if (w == 0.0) {
        return select(0.0, 1.0, d <= 0.0);
    }
    return 1.0 - smoothstep(-0.5, 0.5, d / w);
}

fn fill_line_alpha(position : f32, center : f32, period : f32, stroke : f32, g : vec2<f32>, grad : vec4<f32>, unit : f32) -> f32 {
    let x = position - center + period * 0.5;
    let to_line = abs(x - period * floor(x / period) - period * 0.5);
    return fill_screen_alpha(to_line - stroke * 0.5, g, grad, unit);
}

fn fill_hatch(local : vec2<f32>, origin : vec2<f32>, grad : vec4<f32>, unit : f32, angle : f32, stroke : f32, gaps : vec2<f32>, alternating : bool) -> f32 {
    let dir = vec2<f32>(cos(angle), sin(angle));
    let normal = vec2<f32>(-dir.y, dir.x);
    let period = select(stroke + gaps.x, stroke * 2.0 + gaps.x + gaps.y, alternating);
    let position = fill_pattern_coord(local, origin, normal, unit, period, 0.0);
    var a = fill_line_alpha(position, 0.0, period, stroke, normal, grad, unit);
    if (alternating) {
        a = max(a, fill_line_alpha(position, stroke + gaps.x, period, stroke, normal, grad, unit));
    }
    return a;
}

fn fill_dots(local : vec2<f32>, origin : vec2<f32>, grad : vec4<f32>, unit : f32, angle : f32, skew : f32, radius : f32, gap : f32) -> f32 {
    let axis0 = vec2<f32>(cos(angle), sin(angle));
    let a1 = angle + 1.5707963267948966 - skew;
    let axis1 = vec2<f32>(cos(a1), sin(a1));
    let det = axis0.x * axis1.y - axis0.y * axis1.x;
    let r0 = vec2<f32>(axis1.y, -axis1.x) / det;
    let r1 = vec2<f32>(-axis0.y, axis0.x) / det;
    let period = radius * 2.0 + gap;
    let lattice = vec2<f32>(
        fill_pattern_coord(local, origin, r0, unit, period, 0.0),
        fill_pattern_coord(local, origin, r1, unit, period, 0.0),
    );
    let c = lattice + vec2<f32>(period * 0.5);
    let cell = c - period * floor(c / period) - vec2<f32>(period * 0.5);
    var best = 1.0e20;
    var nearest = vec2<f32>(1.0, 0.0);
    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            let n = cell + vec2<f32>(f32(x), f32(y)) * period;
            let o = axis0 * n.x + axis1 * n.y;
            let d2 = dot(o, o);
            if (d2 < best) {
                best = d2;
                nearest = o;
            }
        }
    }
    let dist = sqrt(best);
    let g = select(vec2<f32>(1.0, 0.0), nearest / max(dist, 1e-20), dist > 0.0);
    return fill_screen_alpha(dist - radius, g, grad, unit);
}


/// A procedural pattern's coverage at `local` (the ground point relative
/// to the view's origin, in pattern axes): `p0` (shape, size, gap0, gap1),
/// `p1` (angle0, angle1, count, world units per pattern unit, negative for
/// pattern y along -z); `origin` the view's origin in pattern axes.
fn fill_procedural_alpha(p0 : vec4<f32>, p1 : vec4<f32>, local : vec2<f32>, origin : vec2<f32>, grad : vec4<f32>) -> f32 {
    let unit = abs(p1.w);
    if (p0.x < 0.5 || unit <= 0.0) {
        return 1.0;
    }
    if (p0.x < 1.5) {
        return fill_hatch(local, origin, grad, unit, p1.x, p0.y, p0.zw, p1.z > 1.5);
    }
    if (p0.x < 2.5) {
        return max(
            fill_hatch(local, origin, grad, unit, p1.x, p0.y, p0.zw, false),
            fill_hatch(local, origin, grad, unit, p1.y, p0.y, p0.zw, false),
        );
    }
    return fill_dots(local, origin, grad, unit, p1.x, p1.y, p0.y, p0.z);
}

/// The view's origin on the ground in pattern axes, plus `offset`.
fn fill_pattern_origin(p1 : vec4<f32>, offset : vec2<f32>) -> vec2<f32> {
    let flip = select(-1.0, 1.0, p1.w > 0.0);
    return vec2<f32>(frame.origin_hi.x + frame.origin_lo.x, (frame.origin_hi.z + frame.origin_lo.z) * flip) + offset;
}
"#;

const POINT: &str = r#"
struct PointIn {
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32,
    @location(0) pos_hi : vec3<f32>,
    @location(1) pos_lo : vec3<f32>,
    // The radius, the stroke width as a multiple of the layer's, and the
    // stroke's dash and gap (in stroke widths).
    @location(2) radius : vec4<f32>,
    @location(3) color  : vec4<f32>,
    @location(4) stroke_color : vec4<f32>,
    // Towards which the point faces, for lighting.
    @location(5) normal : vec3<f32>,
    // The colour behind a fill pattern (`ids.z` bit 3).
    @location(6) background : vec4<f32>,
    /*CUSTOM_IN*/
};

struct PointOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) color : vec4<f32>,
    // Pixels from the centre, and the radius in pixels.
    @location(1) local : vec2<f32>,
    @location(2) radius : f32,
    @location(3) custom0 : vec4<f32>,
    @location(4) custom1 : vec4<f32>,
    @location(5) @interpolate(flat) index : u32,
    // The row's stroke: colour, and width in pixels.
    @location(6) @interpolate(flat) stroke : vec4<f32>,
    @location(7) @interpolate(flat) stroke_width : f32,
    @location(8) @interpolate(flat) dash : vec2<f32>,
    // The fragment's ground point relative to the view's origin (world x,
    // z), for fill patterns.
    @location(9) ground : vec2<f32>,
    @location(10) @interpolate(flat) background : vec4<f32>,
};

@vertex
fn vs_main(in : PointIn) -> PointOut {
    var i : MarkInstance;
    i.position = mark_relative(in.pos_hi, in.pos_lo);
    i.position2 = i.position;
    i.size = in.radius.x;
    i.color = mark_decode(in.color);
    // style3.y: lit (the diffuse reflectance), per point as deck.gl's
    // PointCloudLayer lights them; style3.x ambient, style3.z sRGB.
    if (layer.style3.y > 0.0) {
        i.color = vec4<f32>(mark_phong(i.color.rgb, normalize(in.normal), i.position, layer.style3.x, layer.style3.y, layer.style3.z > 0.5), i.color.a);
    }
    i.color2 = i.color;
    let stroke_width = layer.style2.x * in.radius.y;
    i.custom0 = /*CUSTOM0*/;
    i.custom1 = /*CUSTOM1*/;
    i.visible = 1.0;
    i.index = in.iid;
    i = hook_instance(i);

    var out : PointOut;
    let clip = frame.view_proj * vec4<f32>(i.position, 1.0);
    if (i.visible <= 0.0 || clip.w <= 0.0 || mark_culled(i.position)) {
        out.clip = mark_hidden();
        return out;
    }
    let r = mark_size_px(i.size, clip.w);
    // style2.x is the stroke width. Coverage falls to zero half a pixel
    // past the edge, so single-sampled a quad that far out holds every
    // fragment centre with any — no wider: for the small points of a big
    // layer the extra half pixel was a third of the fragments.
    var extent = r + stroke_width * 0.5 + select(frame.aa.x, 0.0, mark_aliased());
    var corner = mark_corner_signed(mark_quad_corner(in.vid));
    // style3.w: one triangle whose edges touch the disc (vertices at twice
    // the radius, at 0, 120 and 240 degrees) instead of a square — how
    // deck.gl's PointCloudLayer draws a point, hard-edged; the instance's
    // other three vertices are dropped.
    if (layer.style3.w > 0.5 && mark_aliased()) {
        if (in.vid % 6u >= 3u) {
            out.clip = mark_hidden();
            return out;
        }
        let a = f32(in.vid % 6u) * 2.0943951;
        corner = vec2<f32>(cos(a), sin(a)) * 2.0;
        extent = r + stroke_width * 0.5;
    }
    // style2.z: lie flat on the ground (y up, north toward -z) rather than
    // face the camera. The disc keeps the size it has in pixels at its
    // centre, laid out in world units there, so coverage below is the same
    // sum in the same units.
    let world_extent = extent * clip.w / max(frame.misc.x, 1e-6);
    let on_ground = frame.view_proj
        * vec4<f32>(i.position + vec3<f32>(corner.x, 0.0, -corner.y) * world_extent, 1.0);
    let facing = mark_lift_depth(mark_offset_px(clip, corner * extent), i.position, r);
    out.clip = mark_order(select(facing, on_ground, layer.style2.z > 0.5));
    out.local = corner * extent;
    out.radius = r;
    out.stroke = mark_decode(in.stroke_color);
    out.stroke_width = stroke_width;
    out.dash = in.radius.zw;
    out.ground = i.position.xz + vec2<f32>(corner.x, -corner.y) * world_extent;
    out.background = mark_decode(in.background);
    out.color = i.color;
    out.custom0 = i.custom0;
    out.custom1 = i.custom1;
    out.index = i.index;
    return out;
}

/// (fill coverage, outer coverage) at this fragment.
fn point_coverage(in : PointOut) -> vec2<f32> {
    let d = length(in.local);
    let half_stroke = in.stroke_width * 0.5;
    let outer = mark_cov(clamp(in.radius + half_stroke - d + 0.5, 0.0, 1.0));
    var fill = mark_cov(clamp(in.radius - half_stroke - d + 0.5, 0.0, 1.0));
    if (half_stroke <= 0.0) {
        fill = outer;
    }
    return vec2<f32>(fill, outer);
}

@fragment
fn fs_main(in : PointOut) -> @location(0) vec4<f32> {
    // A procedural fill pattern's derivatives, in uniform control flow.
    let pl = vec2<f32>(in.ground.x, in.ground.y * select(-1.0, 1.0, in.custom1.w > 0.0));
    let grad = vec4<f32>(dpdx(pl), dpdy(pl));
    let cov = point_coverage(in);
    if (cov.y <= 0.0) {
        discard;
    }
    let fill = hook_color(in.color, in.custom0, in.custom1);
    // style2.y: 1 = no fill, stroke only.
    let fill_a = select(fill.a, 0.0, layer.style2.y > 0.5);
    var stroke = in.stroke;
    // A dashed stroke: in its gaps, the fill (nothing, without one).
    let period = in.dash.x + in.dash.y;
    if (period > 0.0 && in.stroke_width > 0.0) {
        let angle = atan2(in.local.y, in.local.x) + 3.14159265;
        let along = angle * in.radius / in.stroke_width;
        if (along - period * floor(along / period) > in.dash.x) {
            stroke = vec4<f32>(fill.rgb, fill_a);
        }
    }
    let fa = fill_a * cov.x;
    let sa = stroke.a * (cov.y - cov.x);
    let a = fa + sa;
    if (a <= 0.0) {
        discard;
    }
    var c = vec4<f32>((fill.rgb * fa + stroke.rgb * sa) / a, a);
    // ids.z bit 3: a procedural pattern over the whole disc, stroke too, as
    // deck.gl's FillStyleExtension filters a ScatterplotLayer's colour.
    if (((layer.ids.z >> 3u) & 1u) != 0u) {
        let origin = fill_pattern_origin(in.custom1, vec2<f32>(0.0));
        c.a *= fill_procedural_alpha(in.custom0, in.custom1, pl, origin, grad);
        c = fill_over_background(c, in.background);
    }
    return mark_out(mark_highlight(vec4<f32>(c.rgb, c.a * layer.style2.w), in.index));
}

@fragment
fn fs_pick(in : PointOut) -> @location(0) vec2<u32> {
    if (point_coverage(in).y < 0.5) {
        discard;
    }
    return vec2<u32>(layer.ids.x, in.index);
}
"#;

const SEGMENT: &str = r#"
struct SegIn {
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32,
    @location(0) a_hi : vec3<f32>,
    @location(1) a_lo : vec3<f32>,
    @location(2) b_hi : vec3<f32>,
    @location(3) b_lo : vec3<f32>,
    // The width, the arc's height as a multiple of the layer's, and the
    // row's dash and gap in line widths (zero: the layer's).
    @location(4) width : vec4<f32>,
    @location(5) color : vec4<f32>,
    @location(6) color2 : vec4<f32>,
    // The neighbouring points of a path, relative to the source and the
    // target; zero at a path's ends.
    // xyz: the neighbour minus this end; prev.w the distance along the path
    // to the source, next.w the segment's length.
    @location(7) prev : vec4<f32>,
    @location(8) next : vec4<f32>,
    /*CUSTOM_IN*/
};

struct SegOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) color : vec4<f32>,
    @location(1) color2 : vec4<f32>,
    // Pixels along the piece from its start, and across from its centre line.
    @location(2) uv : vec2<f32>,
    // Length in pixels, half the width in pixels, and what each end is:
    // capped (1), butted against the next piece of the same arc (0), or
    // mitred against the next segment of a path (2).
    @location(3) dims : vec4<f32>,
    // The arc parameter at the piece's two ends.
    @location(4) t_span : vec2<f32>,
    @location(5) custom0 : vec4<f32>,
    @location(6) custom1 : vec4<f32>,
    @location(7) @interpolate(flat) index : u32,
    // Relative world position, for the clip rectangle.
    @location(8) world : vec3<f32>,
    // Distance along the path to this segment's source, the segment's
    // length and the path's, in world units — for dashes (zero on arcs);
    // w: the line's sideways offset in widths.
    @location(9) @interpolate(flat) along : vec4<f32>,
    // Each end's outward mitre direction, in (along, across) terms: where
    // a mitre limit cuts it.
    @location(10) @interpolate(flat) miter : vec4<f32>,
    // The row's dash and gap, line widths; zero for the layer's.
    @location(11) @interpolate(flat) dash : vec2<f32>,
};

/// extra.w: bits 0-1 the cap, bit 2 round joints, bits 3 and up the mitre
/// limit in eighths plus one (0: none).
fn seg_miter_limit() -> f32 {
    let q = u32(layer.extra.w) >> 3u;
    return select(-1.0, (f32(q) - 1.0) / 8.0, q > 0u);
}

/// How sharp a turn may be and still mitre: the cosine of half of it.
var<private> seg_min_cos : f32 = 0.5;

/// A point on the arc from `a` to `b`, lifted by `style2.x` times the chord
/// at its peak (height 0 is the straight segment). On the flat
/// ground the lift is along `style3.xyz`; on a globe (`style3.w` set,
/// `style3.xyz` its centre) the arc follows the great circle and lifts away
/// from the centre. Chosen with `select`, not a branch.
/// The row's multiple of the layer's arc height, set before `seg_arc`.
var<private> seg_height : f32 = 1.0;

fn seg_arc(a : vec3<f32>, b : vec3<f32>, t : f32) -> vec3<f32> {
    // style2.z bit 0 selects the profile: a parabola, or deck.gl's half
    // ellipse, `2 sqrt(t (1 - t))`; both peak at `style2.x` of the chord.
    let s = t * (1.0 - t);
    let profile = select(4.0 * s, 2.0 * sqrt(max(s, 0.0)), (u32(layer.style2.z) & 1u) != 0u);
    let ellipse = (u32(layer.style2.z) & 1u) != 0u;
    let h = layer.style2.x * seg_height * profile * distance(a, b);
    let parabola = mix(a, b, t) + layer.style3.xyz * h;
    // deck.gl's `paraboloid`: over the ground distance, through both ends'
    // heights (the lower end at r = 0), peaking at the same height when the
    // ends are level.
    let up = layer.style3.xyz;
    let za = dot(a, up);
    let zb = dot(b, up);
    let ga = a - up * za;
    let gb = b - up * zb;
    let dh = 2.0 * layer.style2.x * seg_height * distance(ga, gb);
    let dz = zb - za;
    let unit = dz / max(dh, 1e-12);
    let down = dz <= 0.0;
    let r = select(t, 1.0 - t, down);
    let z = select(sqrt(max(r * (unit * unit + 1.0 - r), 0.0)) * dh + select(za, zb, down), za + dz * t, dh <= 0.0);
    // material.w: the arc tilted this many radians about its chord, its
    // height leaning left of the way from source to target (deck.gl's
    // `getTilt`).
    let tilt = layer.material.w;
    let chord = gb - ga;
    let east_north = normalize(vec2<f32>(chord.x, -chord.z) + vec2<f32>(1e-20, 0.0));
    let left = vec3<f32>(-east_north.y, 0.0, -east_north.x);
    let lean = up * cos(tilt) + left * sin(tilt);
    let flat = select(parabola, mix(ga, gb, t) + lean * z, ellipse);
    let c = layer.style3.xyz;
    let ca = a - c;
    let cb = b - c;
    let ra = length(ca);
    let rb = length(cb);
    let da = ca / max(ra, 1e-6);
    let db = cb / max(rb, 1e-6);
    let w = acos(clamp(dot(da, db), -1.0, 1.0));
    let sw = sin(w);
    let dir = select(
        normalize(mix(da, db, t)),
        (sin((1.0 - t) * w) * da + sin(t * w) * db) / max(sw, 1e-6),
        sw > 1e-5,
    );
    let globe = c + dir * (mix(ra, rb, t) + h);
    return select(flat, globe, layer.style3.w > 0.5);
}

fn seg_px(c : vec4<f32>) -> vec2<f32> {
    return c.xy / c.w * frame.viewport.xy * 0.5;
}

/// Lines flat on the ground (`style2.z` bit 2): joints, mitres and caps are
/// worked out on the ground plane in world units, not on the screen.
var<private> seg_flat : bool = false;

/// `p` (clip `c`) in the plane the line is drawn in: screen pixels, or the
/// ground's (x, north) in world units.
fn seg_pt(p : vec3<f32>, c : vec4<f32>) -> vec2<f32> {
    return select(seg_px(c), vec2<f32>(p.x, -p.z), seg_flat);
}

/// A neighbour's clip position `c` behind the camera (facing lines), cut
/// toward the joint's `r` where it crosses `frame.clip.x`: deck.gl's
/// `clipLine`, which its joints take their direction from.
fn seg_clip_toward(c : vec4<f32>, r : vec4<f32>) -> vec4<f32> {
    let eps = frame.clip.x;
    if (seg_flat || c.w >= eps || r.w < eps) {
        return c;
    }
    return r + (c - r) * ((eps - r.w) / (c.w - r.w));
}

/// The mitre at a joint whose neighbour is `offset` from `p`, for a segment
/// with screen normal `normal`: the bisector of the two normals, scaled so
/// the quad keeps its width along it, in xy; z is 1 when the joint is
/// mitred and 0 when the end should be capped — no neighbour, or a turn too
/// sharp to mitre. `away` is 1 when the neighbour lies past the segment's far
/// end (the next point) and -1 when it lies before the near one.
fn seg_miter(p : vec3<f32>, s : vec2<f32>, offset : vec3<f32>, normal : vec2<f32>, away : f32) -> vec3<f32> {
    if (dot(offset, offset) <= 0.0) {
        return vec3<f32>(normal, 0.0);
    }
    let c = seg_clip_toward(frame.view_proj * vec4<f32>(p + offset, 1.0), frame.view_proj * vec4<f32>(p, 1.0));
    if (c.w <= 0.0 && !seg_flat) {
        return vec3<f32>(normal, 0.0);
    }
    let d = (seg_pt(p + offset, c) - s) * away;
    let l = length(d);
    if (l < 1e-6) {
        return vec3<f32>(normal, 0.0);
    }
    let n2 = vec2<f32>(-d.y, d.x) / l;
    let m = normal + n2;
    let ml = length(m);
    // cos of half the turn: 0.5 is a 120° turn, a mitre twice the width.
    let cosine = ml * 0.5;
    if (cosine < seg_min_cos) {
        return vec3<f32>(normal, 0.0);
    }
    return vec3<f32>(m / ml / cosine, 1.0);
}

@vertex
fn vs_main(in : SegIn) -> SegOut {
    var i : MarkInstance;
    i.position = mark_relative(in.a_hi, in.a_lo);
    i.position2 = mark_relative(in.b_hi, in.b_lo);
    i.size = in.width.x;
    // style2.z bit 4: width.y is each row's sideways offset in widths
    // (right of the direction of travel), not an arc height.
    let offsets = (u32(layer.style2.z) & 16u) != 0u;
    seg_height = select(in.width.y, 1.0, offsets);
    let offset = select(0.0, in.width.y, offsets);
    i.color = mark_decode(in.color);
    i.color2 = mark_decode(in.color2);
    i.custom0 = /*CUSTOM0*/;
    i.custom1 = /*CUSTOM1*/;
    i.visible = 1.0;
    i.index = in.iid;
    i = hook_instance(i);

    var out : SegOut;
    seg_flat = (u32(layer.style2.z) & 4u) != 0u;
    let n = max(u32(layer.style2.y), 1u);
    let piece = in.vid / 6u;
    // style2.z bit 1: pieces eased toward the ends (smoothstep), where a
    // steep profile needs them — deck.gl's `getSegmentRatio`.
    let ease = (u32(layer.style2.z) & 2u) != 0u;
    let u0 = f32(piece) / f32(n);
    let u1 = f32(piece + 1u) / f32(n);
    let t0 = select(u0, smoothstep(0.0, 1.0, u0), ease);
    let t1 = select(u1, smoothstep(0.0, 1.0, u1), ease);
    let q0 = seg_arc(i.position, i.position2, t0);
    let q1 = seg_arc(i.position, i.position2, t1);
    let w0 = (frame.view_proj * vec4<f32>(q0, 1.0)).w;
    let w1 = (frame.view_proj * vec4<f32>(q1, 1.0)).w;
    // A straight line facing the camera that passes behind it is cut where
    // it does (w = 0.001, deck.gl's `clipLine`), not dropped; `clip` is
    // the visible part of the segment, for its dashes and colours.
    let eps = frame.clip.x;
    let cuttable = !seg_flat && n == 1u && (w0 >= eps || w1 >= eps);
    let behind0 = cuttable && w0 < eps;
    let behind1 = cuttable && w1 < eps;
    let r0 = select(0.0, clamp((eps - w0) / (w1 - w0), 0.0, 1.0), behind0);
    let r1 = select(1.0, clamp((eps - w0) / (w1 - w0), 0.0, 1.0), behind1);
    let p0 = mix(q0, q1, r0);
    let p1 = mix(q0, q1, r1);
    let c0 = frame.view_proj * vec4<f32>(p0, 1.0);
    let c1 = frame.view_proj * vec4<f32>(p1, 1.0);
    if (i.visible <= 0.0 || piece >= n || c0.w <= 0.0 || c1.w <= 0.0) {
        out.clip = mark_hidden();
        return out;
    }
    let s0 = seg_pt(p0, c0);
    let s1 = seg_pt(p1, c1);
    var dir = s1 - s0;
    let len = length(dir);
    if (len > 1e-6) {
        dir = dir / len;
    } else {
        dir = vec2<f32>(1.0, 0.0);
    }
    let normal = vec2<f32>(-dir.y, dir.x);
    let w_mid = 0.5 * (c0.w + c1.w);
    let half_px = mark_size_px(i.size, w_mid) * 0.5;
    // Flat: the same size in world units at that depth (a pixel there is
    // `w / misc.x` of them).
    let world_per_px = w_mid / max(frame.misc.x, 1e-6);
    let half = select(half_px, half_px * world_per_px, seg_flat);
    // Caps only where the line really ends — extra.w: 0 butt, 1 round,
    // 3 square. The joints inside an arc are mitred (below), so its pieces
    // meet on one edge: no gap outside the curve, no overlap inside it to
    // darken a translucent arc.
    // Bit 2 of extra.w: round joints.
    let cap_code = u32(layer.extra.w);
    let end_cap = f32(cap_code & 3u);
    let round_join = (cap_code & 4u) != 0u;
    // A mitre limit: mitre nearly any turn, the fragments cut it.
    seg_min_cos = select(0.5, 0.02, seg_miter_limit() >= 0.0);
    var cap0 = select(0.0, end_cap, piece == 0u && !behind0);
    var cap1 = select(0.0, end_cap, piece + 1u == n && !behind1);
    // Straight segments of a path mitre against their neighbours; both
    // segments at a joint compute the same bisector, so they meet on it.
    // On a curved path (an arc, a great circle) the neighbour's direction at
    // the joint is that of its nearest piece: the point a piece along it, which
    // the segments either side compute identically, so they still meet on one
    // bisector. Straight (n = 1) this is the neighbouring point itself.
    // No new branches here: control flow near mark_relative lets Metal
    // reassociate its subtraction (see mark_relative), so the joints are
    // chosen with `select`.
    let step = 1.0 / f32(n);
    let has0 = dot(in.prev.xyz, in.prev.xyz) > 0.0 && !behind0;
    let has1 = dot(in.next.xyz, in.next.xyz) > 0.0 && !behind1;
    // Straight, the stored offset itself: exact, where the round trip
    // through p0 would depend on p0's last bits.
    let off0 = select(seg_arc(p0 + in.prev.xyz, p0, 1.0 - step) - p0, in.prev.xyz, n == 1u);
    let off1 = select(seg_arc(p1, p1 + in.next.xyz, step) - p1, in.next.xyz, n == 1u);
    let mm0 = seg_miter(p0, s0, select(vec3<f32>(0.0), off0, has0), normal, -1.0);
    let mm1 = seg_miter(p1, s1, select(vec3<f32>(0.0), off1, has1), normal, 1.0);
    let first = piece == 0u;
    let last = piece + 1u == n;
    // Inside an arc: the neighbouring pieces' far points, so both pieces at
    // a joint mitre on the same bisector.
    let up = max(f32(piece) - 1.0, 0.0) / f32(n);
    let un = min(f32(piece + 2u), f32(n)) / f32(n);
    let pp = seg_arc(i.position, i.position2, select(up, smoothstep(0.0, 1.0, up), ease));
    let pn = seg_arc(i.position, i.position2, select(un, smoothstep(0.0, 1.0, un), ease));
    let mi0 = seg_miter(p0, s0, pp - p0, normal, -1.0);
    let mi1 = seg_miter(p1, s1, pn - p1, normal, 1.0);
    // Round joints: each segment ends in a disc there instead.
    let plain = vec3<f32>(normal, 0.0);
    let m0 = select(mi0, select(mm0, plain, round_join && has0), first);
    let m1 = select(mi1, select(mm1, plain, round_join && has1), last);
    cap0 = select(cap0, 2.0, !first && mi0.z > 0.5);
    cap1 = select(cap1, 2.0, !last && mi1.z > 0.5);
    cap0 = select(cap0, 2.0, first && mm0.z > 0.5);
    cap1 = select(cap1, 2.0, last && mm1.z > 0.5);
    // A joint too sharp to mitre is rounded, whatever the path's own end
    // caps are: a butt there would leave a notch on the outside of the turn.
    // These are discs (4) cut on the line across the joint, each segment
    // keeping its own side, so a translucent path is not doubled there.
    cap0 = select(cap0, 4.0, first && has0 && (mm0.z <= 0.5 || round_join));
    cap1 = select(cap1, 4.0, last && has1 && (mm1.z <= 0.5 || round_join));
    // An offset line: the quad widened to hold it, as deck.gl's
    // `PathStyleExtension` widens it; the fragments keep the band.
    let pad = half * (2.0 * abs(offset) + 1.0) + select(1.0, 0.0, mark_aliased()) * select(1.0, world_per_px, seg_flat);
    let k = mark_quad_corner(in.vid);
    let far = k == 1u || k == 2u;
    let side = select(-1.0, 1.0, k >= 2u);
    let m = select(m0, m1, far);
    var off : vec2<f32>;
    if (m.z > 0.5) {
        off = m.xy * side * pad;
    } else {
        // Round and square ends reach half a width past the point; a butt
        // end stops at it.
        let cap = min(select(cap0, cap1, far), 1.0);
        off = dir * select(-pad * cap, pad * cap, far) + normal * side * pad;
    }
    let base = select(c0, c1, far);
    let corner = select(p0, p1, far) + vec3<f32>(off.x, 0.0, -off.y);
    // style2.z bit 5: flat on a globe's surface — the screen offset laid in
    // the sphere's tangent plane at the segment, its along part along the
    // segment and its across part across it, a pixel `world_per_px` there
    // (deck.gl's paths extruded in common space): foreshortened at the
    // limb as the surface is.
    let sphere_flat = (u32(layer.style2.z) & 32u) != 0u;
    let nrm = normalize(0.5 * (p0 + p1) - layer.style3.xyz);
    let t3 = normalize(p1 - p0 + vec3<f32>(1e-20, 0.0, 0.0));
    var across3 = normalize(cross(nrm, t3) + vec3<f32>(0.0, 1e-20, 0.0));
    let probe = seg_px(frame.view_proj * vec4<f32>(p0 + across3 * world_per_px, 1.0)) - s0;
    across3 = select(across3, -across3, dot(probe, normal) < 0.0);
    let on_sphere = select(p0, p1, far) + (t3 * dot(off, dir) + across3 * dot(off, normal)) * world_per_px;
    var placed = select(mark_offset_px(base, off), frame.view_proj * vec4<f32>(corner, 1.0), seg_flat);
    placed = select(placed, frame.view_proj * vec4<f32>(on_sphere, 1.0), sphere_flat);
    // Its length as laid on the sphere, in the same pixels.
    let len_laid = select(len, distance(p0, p1) / max(world_per_px, 1e-12), sphere_flat);
    out.clip = mark_order(mark_lift_depth(placed, select(p0, p1, far), half_px));
    out.world = select(p0, p1, far);
    out.uv = vec2<f32>(select(0.0, len_laid, far) + dot(off, dir), dot(off, normal));
    out.dims = vec4<f32>(len_laid, half, cap0, cap1);
    // A half turn (the path goes back the way it came, A-B-A): deck.gl's
    // joint has no bisector there — its tangent falls back to the normal,
    // the turn counts as clockwise — and pulls the corner on the inner side
    // (+normal) back by the whole segment, to its other end. Each segment
    // becomes half its rectangle, cut on the diagonal; the two overlap
    // along it.
    let back_out = seg_pt(p1 + off1, seg_clip_toward(frame.view_proj * vec4<f32>(p1 + off1, 1.0), c1)) - s1;
    let back_in = s0 - seg_pt(p0 + off0, seg_clip_toward(frame.view_proj * vec4<f32>(p0 + off0, 1.0), c0));
    let half_turn1 = n == 1u && has1 && dot(back_out, back_out) > 0.0 && dot(normalize(back_out), dir) < -0.99999;
    let half_turn0 = n == 1u && has0 && dot(back_in, back_in) > 0.0 && dot(normalize(back_in), dir) < -0.99999;
    if ((far && half_turn1 && side > 0.0) || (!far && half_turn0 && side > 0.0)) {
        out.clip = mark_order(select(c1, c0, far));
        out.uv = vec2<f32>(select(len_laid, 0.0, far), 0.0);
    }
    if (half_turn0) {
        out.dims.z = 2.0;
    }
    if (half_turn1) {
        out.dims.w = 2.0;
    }
    // style2.z bit 6: an arc as deck.gl's ArcLayer draws it, a triangle
    // strip without joints — each vertex pushed across the direction to its
    // neighbour in clip space, before the perspective divide (the previous
    // vertex's, but the first's next), by half the width at the focal plane,
    // added in clip space unscaled by w. Hard-edged: the rasterizer decides.
    let strip = (u32(layer.style2.z) & 64u) != 0u && n > 1u && mark_aliased();
    if (strip) {
        let up_prev = max(f32(piece) - 1.0, 0.0) / f32(n);
        let cprev = frame.view_proj * vec4<f32>(seg_arc(i.position, i.position2, select(up_prev, smoothstep(0.0, 1.0, up_prev), ease)), 1.0);
        let along_clip = select(select(c0.xy - cprev.xy, c1.xy - c0.xy, piece == 0u), c1.xy - c0.xy, far);
        let d = normalize(along_clip * frame.viewport.xy + vec2<f32>(1e-20, 0.0));
        let focal = select(c0.w, frame.aa.y, frame.aa.y > 0.0);
        let half_focal = mark_size_px(i.size, focal) * 0.5;
        let px = vec2<f32>(-d.y, d.x) * side * half_focal;
        let base_clip = select(c0, c1, far);
        out.clip = mark_order(base_clip + vec4<f32>(px * 2.0 * frame.viewport.zw * focal, 0.0, 0.0));
        out.uv = vec2<f32>(select(0.0, len_laid, far), 0.0);
        out.dims = vec4<f32>(len_laid, half, 2.0, 2.0);
    }
    // Straight, the visible part of the segment (all of it unless cut).
    out.t_span = select(vec2<f32>(t0, t1), vec2<f32>(r0, r1), n == 1u);
    out.along = vec4<f32>(select(vec3<f32>(0.0), vec3<f32>(in.prev.w, distance(i.position, i.position2), in.next.w), n == 1u), offset);
    out.dash = in.width.zw;
    // The outward mitre at each end: the side of the bisector behind the
    // start, and ahead of the end.
    var o0 = normalize(m0.xy) * select(-1.0, 1.0, dot(m0.xy, dir) <= 0.0);
    var o1 = normalize(m1.xy) * select(-1.0, 1.0, dot(m1.xy, dir) >= 0.0);
    // At a disc joint, instead: the cut's normal, the mean of the two
    // directions (across the path on a half turn).
    let cp = seg_clip_toward(frame.view_proj * vec4<f32>(p0 + off0, 1.0), c0);
    let cn = seg_clip_toward(frame.view_proj * vec4<f32>(p1 + off1, 1.0), c1);
    let d_in = s0 - seg_pt(p0 + off0, cp);
    let d_out = seg_pt(p1 + off1, cn) - s1;
    let k0 = select(dir, normalize(d_in), dot(d_in, d_in) > 1e-12) + dir;
    let k1 = dir + select(dir, normalize(d_out), dot(d_out, d_out) > 1e-12);
    let cut0 = select(normal, normalize(k0), dot(k0, k0) > 1e-8);
    let cut1 = select(normal, normalize(k1), dot(k1, k1) > 1e-8);
    o0 = select(o0, cut0, cap0 > 3.5);
    o1 = select(o1, cut1, cap1 > 3.5);
    out.miter = vec4<f32>(dot(o0, dir), dot(o0, normal), dot(o1, dir), dot(o1, normal));
    out.color = i.color;
    out.color2 = i.color2;
    out.custom0 = i.custom0;
    out.custom1 = i.custom1;
    out.index = i.index;
    return out;
}

/// Distance in pixels from the line's centre, ends included: a butt end
/// (0) cuts, a round one (1) is a half disc, a mitred one (2) is cut by the
/// quad's own edge on the bisector, a square one (3) a half square, a
/// disc joint (4) round, cut across the joint in `seg_coverage`.
fn seg_end_dist(u : f32, v : f32, cap : f32) -> f32 {
    if (cap > 3.5) {
        // A disc joint: round; its cut is applied with the coverage.
        return length(vec2<f32>(u, v));
    }
    if (cap > 2.5) {
        return max(abs(v), u);
    }
    return select(1e9, length(vec2<f32>(u, v)), cap > 0.5);
}

fn seg_dist(in : SegOut) -> f32 {
    let u = in.uv.x;
    let v = in.uv.y;
    let len = in.dims.x;
    // Aliased, the quad's sides are the line's: the rasterizer decides
    // them (sample by sample under MSAA, where a fragment's centre may lie
    // just outside), as for deck.gl's paths.
    var dist = select(abs(v), 0.0, mark_aliased() && in.along.w == 0.0);
    if (u < 0.0 && (in.dims.z < 1.5 || in.dims.z > 2.5)) {
        dist = seg_offset_round(seg_end_dist(-u, v, in.dims.z), -u, v, in.dims.z, in);
    } else if (u > len && (in.dims.w < 1.5 || in.dims.w > 2.5)) {
        dist = seg_offset_round(seg_end_dist(u - len, v, in.dims.w), u - len, v, in.dims.w, in);
    }
    return dist;
}

/// A round end or disc joint of an offset line (`along.w`), as deck.gl's
/// `PathStyleExtension` leaves it: the path layer's disc, of the widened
/// quad's half width (`2·|offset| + 1` half widths) about the path's own
/// point, not the offset band's, cut to the band. `d` is the distance as
/// for an unshifted line, `p` how far past the point, `v` across from the
/// band's centre.
fn seg_offset_round(d : f32, p : f32, v : f32, cap : f32, in : SegOut) -> f32 {
    let off = in.along.w;
    let round = (cap > 0.5 && cap < 1.5) || cap > 3.5;
    if (!round || off == 0.0) {
        return d;
    }
    let half = in.dims.y;
    let from_axis = length(vec2<f32>(p, v - off * 2.0 * half));
    return max(from_axis - 2.0 * abs(off) * half, abs(v));
}

/// Screen derivatives of `uv` (pixels), set by `seg_scaled`: x along, y
/// across, each (d/dx, d/dy).
var<private> seg_grad_along : vec2<f32> = vec2<f32>(1.0, 0.0);
var<private> seg_grad_across : vec2<f32> = vec2<f32>(0.0, 1.0);
/// The segment's length as drawn, before `seg_scaled` turns it to pixels.
var<private> seg_raw_len : f32 = 1.0;

/// Solid stroke before `position` in a square wave of period `unit` that
/// starts with `solid`: its integral.
fn seg_dash_integral(position : f32, solid : f32, unit : f32) -> f32 {
    return floor(position / unit) * solid + min(position - unit * floor(position / unit), solid);
}

/// Coverage of the dash pattern at this fragment, deck.gl's
/// `PathStyleExtension`: the square wave integrated over the fragment's
/// footprint along the line, so dash ends are smooth and periods under a
/// pixel fade to the duty cycle; round caps round the dashes' ends too.
/// `extra.xy` the layer's dash and gap; material.x pixels per dash unit (0:
/// line widths); material.y bit 0 measures along the whole path, bit 1
/// justifies. All derivatives come from `seg_scaled`: no branches here
/// before they are used.
fn seg_dash(in : SegOut) -> f32 {
    let half = max(in.dims.y, 1e-6);
    // The row's dash, or the layer's.
    let pattern = select(layer.extra.xy, in.dash, in.dash.x + in.dash.y > 0.0);
    let mode = u32(layer.material.y);
    let whole_path = (mode & 1u) != 0u && in.along.y > 0.0;
    let justified = (mode & 2u) != 0u;
    // Pixels per world unit along this segment, for the distances along
    // the path (world units): as drawn, or (material.z, for lines facing
    // the camera, or flat on a globe) the view's nominal scale, as deck.gl
    // measures such a path's dashes — its world length at that scale,
    // spread over the segment on screen.
    let flat = (u32(layer.style2.z) & 4u) != 0u;
    let sphere_flat = (u32(layer.style2.z) & 32u) != 0u;
    let nominal = layer.material.z > 0.0 && !flat && in.along.y > 0.0;
    // Cut by the camera: what of the segment is left, and from where.
    let part = max(in.t_span.y - in.t_span.x, 1e-6);
    let seg_world = in.along.y * part;
    // Flat, the ground's pixels per world unit, and the segment's true
    // (3D) length spread over its ground length, as deck.gl scales a
    // climbing path's coordinate by its arc length ratio.
    let ground = max(seg_raw_len, 1e-12);
    var k = select(in.dims.x / max(seg_world, 1e-12), layer.material.z, nominal);
    k = select(k, in.dims.x / ground, flat);
    var stretch = select(1.0, seg_world * k / max(in.dims.x, 1e-6), nominal);
    stretch = select(stretch, max(in.along.y / ground, 1.0), flat && in.along.y > 0.0);
    // Flat on a globe, the width is in the world too: at the nominal scale
    // as the distances along (a billboard's is in pixels either way).
    let width_px = half * 2.0 * select(1.0, stretch, nominal && sphere_flat);
    let unit_px = select(width_px, layer.material.x, layer.material.x > 0.0);
    var solid = pattern.x * unit_px;
    let gap = pattern.y * unit_px;
    var period = solid + gap;
    // The part cut off before what is drawn, and the whole segment.
    let cut_before = in.along.y * in.t_span.x * k;
    let before = in.along.x * k + cut_before;
    let run = select(in.dims.x * stretch / part, in.along.z * k, whole_path);
    let fitted = run / max(round(run / max(period, 1e-4)), 1.0);
    period = select(period, fitted, justified);
    solid = min(solid, period);
    var offset = select(0.0, solid * 0.5, justified);
    offset += select(cut_before, before - period * floor(before / max(period, 1e-6)), whole_path);
    let along = in.uv.x * stretch + offset;
    let unit = max(period, 1e-6);
    let phase = along - unit * floor(along / unit);
    let footprint = max((abs(seg_grad_along.x) + abs(seg_grad_along.y)) * stretch, 1e-4);
    // Square ends: the integral over the footprint.
    let h = footprint * 0.5;
    let p = phase;
    let square = (seg_dash_integral(p + h, solid, unit) - seg_dash_integral(p - h, solid, unit)) / footprint;
    // Round ends (round caps): the distance, in half widths, to the nearer
    // dash end's disc, its ramp a pixel wide.
    let into_gap = min(phase - solid, unit - phase);
    let x = in.uv.y / half;
    let g = max(into_gap, 0.0) / half;
    let to_end = length(vec2<f32>(g, x));
    let sign_along = select(-1.0, 1.0, phase - solid < unit - phase);
    let grad = (select(0.0, g, into_gap > 0.0) * sign_along * stretch * seg_grad_along + x * seg_grad_across) / (half * max(to_end, 1e-6));
    let edge = (1.0 - to_end) / max(abs(grad.x) + abs(grad.y), 1e-6);
    var rounded = select(1.0, smoothstep(-0.5, 0.5, edge), into_gap > 0.0);
    // Periods inside a pixel: each scanline's share of the capsules.
    let blend = smoothstep(unit, 2.0 * unit, footprint);
    let cap_span = 2.0 * half * sqrt(max(1.0 - x * x, 0.0));
    let duty = clamp((solid + min(max(unit - solid, 0.0), cap_span)) / unit, 0.0, 1.0);
    rounded = mix(rounded, duty, blend);
    let round_caps = (u32(layer.extra.w) & 3u) == 1u;
    let cov = clamp(select(square, rounded, round_caps), 0.0, 1.0);
    // None on arcs (no distance along).
    return select(cov, 1.0, pattern.x <= 0.0 || pattern.y <= 0.0 || period <= 0.0 || in.along.y <= 0.0);
}

/// `in` with its lengths in pixels: flat lines carry world units, turned
/// to pixels here by how fast `uv.y` changes across the screen. Call first
/// thing, where derivatives are defined.
fn seg_scaled(in : SegOut) -> SegOut {
    let g = vec2<f32>(dpdx(in.uv.y), dpdy(in.uv.y));
    let k = 1.0 / max(length(g), 1e-9);
    let ga = vec2<f32>(dpdx(in.uv.x), dpdy(in.uv.x));
    var o = in;
    seg_raw_len = in.dims.x;
    // Flat on the ground or on a globe: lengths to scale to the screen.
    let flat = (u32(layer.style2.z) & 36u) != 0u;
    let scale = select(1.0, k, flat);
    seg_grad_along = ga * scale;
    seg_grad_across = g * scale;
    if (flat) {
        o.uv = in.uv * k;
        o.dims = vec4<f32>(in.dims.xy * k, in.dims.zw);
    }
    // Across from the offset band's centre line, right of travel (-y).
    o.uv.y += in.along.w * 2.0 * o.dims.y;
    // How wide a pixel is across the line, and towards its axis (round
    // ends): `1` square to a row or column, `√2` on the diagonal. Taken
    // from the gradients rather than `fwidth` of the distance, which a 2×2
    // block astride the centre line or an end's point folds to nothing.
    seg_fw_across = max(abs(seg_grad_across.x) + abs(seg_grad_across.y), 1e-6);
    let before = o.uv.x < 0.0;
    let past = select(max(o.uv.x - o.dims.x, 0.0), -o.uv.x, before);
    let r = length(vec2<f32>(past, o.uv.y));
    let toward = (past * select(1.0, -1.0, before) * seg_grad_along + o.uv.y * seg_grad_across) / max(r, 1e-6);
    seg_fw_dist = select(seg_fw_across, max(abs(toward.x) + abs(toward.y), 1e-6), r > 1e-4);
    // deck.gl's: `fwidth` of the distances themselves (its ramp, style2.z
    // bit 7).
    let past_any = max(max(-o.uv.x, o.uv.x - o.dims.x), 0.0);
    seg_fw_across_deck = fwidth(abs(o.uv.y));
    seg_fw_dist_deck = fwidth(length(vec2<f32>(past_any, o.uv.y)));
    return o;
}

/// The width of a pixel across the line and towards its axis, in pixels
/// of the line's own lengths, set by `seg_scaled`.
var<private> seg_fw_across : f32 = 1.0;
var<private> seg_fw_dist : f32 = 1.0;
var<private> seg_fw_across_deck : f32 = 1.0;
var<private> seg_fw_dist_deck : f32 = 1.0;

/// Coverage of a pixel `w` wide by a stroke `half` either side of its centre
/// line, `u` from it: the share of the pixel it covers, a box filter
/// (`sdf_coverage_stroke`). A line thinner than a pixel covers its share
/// rather than a pixel or two at nearly full ink, and keeps one weight as it
/// moves between pixels; a wide one ramps over its edge pixel as before.
fn seg_box(u : f32, half : f32, w : f32) -> f32 {
    let a = abs(u);
    let h = max(half, 0.0);
    let lo = max(a - 0.5 * w, -h);
    let hi = min(a + 0.5 * w, h);
    return clamp((hi - lo) / w, 0.0, 1.0);
}

fn seg_coverage(in : SegOut) -> f32 {
    // Antialiased: the share of the pixel the stroke covers, across the
    // line and, for round ends and disc joints, towards its axis — where
    // deck.gl ramps a `smoothstep` over a pixel of the distance to the edge,
    // which draws a line thinner than a pixel too dark and too wide. Butt
    // and square ends are cut hard, as deck.gl leaves them.
    let edge = in.dims.y - seg_dist(in);
    let u = in.uv.x;
    let end_cap = select(select(2.0, in.dims.w, u > in.dims.x), in.dims.z, u < 0.0);
    let round_end = (end_cap > 0.5 && end_cap < 1.5) || end_cap > 3.5;
    // style2.z bit 7: deck.gl's ramp instead.
    let deck_ramp = (u32(layer.style2.z) & 128u) != 0u;
    let across = select(
        seg_box(in.uv.y, in.dims.y, seg_fw_across),
        smoothstep(-0.5, 0.5, (in.dims.y - abs(in.uv.y)) / max(seg_fw_across_deck, 1e-6)),
        deck_ramp,
    );
    let radial = select(
        seg_box(seg_dist(in), in.dims.y, seg_fw_dist),
        smoothstep(-0.5, 0.5, edge / max(seg_fw_dist_deck, 1e-6)),
        deck_ramp,
    );
    // How far past the segment's end (0 within it): a butt end stops at
    // it, a square one half a width on.
    let past = max(max(-u, u - in.dims.x), 0.0);
    let reach = select(0.0, in.dims.y, end_cap > 2.5 && end_cap < 3.5);
    let mitred = end_cap > 1.5 && end_cap < 2.5;
    let ramped = select(across * select(step(past, reach), 1.0, mitred), radial, round_end);
    var cov = select(ramped, clamp(edge + 0.5, 0.0, 1.0), mark_aliased());
    // Disc joints: each segment keeps its side of the cut across the
    // joint. A hard cut: inside the path both sides are one colour, and two
    // translucent ramps would not add back up to one layer.
    // The cut sits an odd fraction of a pixel past the joint, off the
    // lines of pixel centres a turn's cut tends to follow (through the
    // joint, across or diagonal), where each side's rounding could split a
    // centre between both segments or neither.
    let cut_at = 0.1234;
    if (in.dims.z > 3.5 && dot(in.uv, in.miter.xy) < cut_at) {
        cov = 0.0;
    }
    if (in.dims.w > 3.5 && dot(vec2<f32>(in.uv.x - in.dims.x, in.uv.y), in.miter.zw) >= cut_at) {
        cov = 0.0;
    }
    // A mitre limit cuts mitred ends flat, (limit + 1) half widths out
    // from the joint along the bisector.
    let limit = seg_miter_limit();
    if (limit >= 0.0) {
        let cut = (limit + 1.0) * in.dims.y;
        let t0 = dot(in.uv, in.miter.xy);
        let t1 = dot(vec2<f32>(in.uv.x - in.dims.x, in.uv.y), in.miter.zw);
        cov = min(cov, select(1.0, clamp(cut - t0 + 0.5, 0.0, 1.0), abs(in.dims.z - 2.0) < 0.5));
        cov = min(cov, select(1.0, clamp(cut - t1 + 0.5, 0.0, 1.0), abs(in.dims.w - 2.0) < 0.5));
    }
    return mark_cov(cov) * seg_dash(in);
}

@fragment
fn fs_main(raw : SegOut) -> @location(0) vec4<f32> {
    let in = seg_scaled(raw);
    if (mark_culled(in.world)) {
        discard;
    }
    let cov = seg_coverage(in);
    if (cov <= 0.0) {
        discard;
    }
    let t = mix(in.t_span.x, in.t_span.y, clamp(in.uv.x / max(in.dims.x, 1e-6), 0.0, 1.0));
    var c = hook_color(mark_mix(in.color, in.color2, t), in.custom0, in.custom1);
    // The casing: `extra.z` pixels of `stroke` colour inside the edge.
    if (layer.extra.z > 0.0) {
        let inner = clamp(in.dims.y - layer.extra.z - seg_dist(in) + 0.5, 0.0, 1.0);
        c = mix(layer.stroke, c, inner);
    }
    return mark_out(mark_highlight(vec4<f32>(c.rgb, c.a * cov * layer.style2.w), in.index));
}

@fragment
fn fs_pick(raw : SegOut) -> @location(0) vec2<u32> {
    let in = seg_scaled(raw);
    if (mark_culled(in.world)) {
        discard;
    }
    if (seg_coverage(in) < 0.5) {
        discard;
    }
    return vec2<u32>(layer.ids.x, in.index);
}
"#;

const SHAPE: &str = r#"
struct ShapeIn {
    @builtin(instance_index) iid : u32,
    @location(0) vertex : vec3<f32>,
    @location(1) normal : vec3<f32>,
    // Where in the texture's atlas, and the vertex's colour (linear).
    @location(2) uv : vec2<f32>,
    @location(3) vertex_color : vec4<f32>,
    @location(4) pos_hi : vec3<f32>,
    @location(5) pos_lo : vec3<f32>,
    @location(6) scale : vec3<f32>,
    @location(7) color : vec4<f32>,
    // A unit quaternion turning the scaled mesh.
    @location(8) rotation : vec4<f32>,
    /*CUSTOM_IN*/
};

struct ShapeOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) color : vec4<f32>,
    @location(1) custom0 : vec4<f32>,
    @location(2) custom1 : vec4<f32>,
    @location(3) @interpolate(flat) index : u32,
    @location(4) uv : vec2<f32>,
    // For the scene's lights, shaded per pixel: where, and the normal
    // (zero for a mesh without normals, shaded flat).
    @location(5) world : vec3<f32>,
    @location(6) normal : vec3<f32>,
};

/// One light by luma.gl's Phong (`lighting_getLightColor`), on sRGB values.
fn shape_phong_light(s : vec3<f32>, l : vec3<f32>, v : vec3<f32>, n : vec3<f32>, color : vec3<f32>) -> vec3<f32> {
    let h = normalize(l + v);
    let lambert = dot(l, n);
    var spec = 0.0;
    if (lambert > 0.0) {
        spec = pow(max(dot(n, h), 0.0), layer.material.y);
    }
    return (max(lambert, 0.0) * layer.style2.y * s + spec * layer.stroke.rgb) * color;
}

/// `rgb` (linear) lit by the view's scene lights at `p` with normal `n`.
fn shape_scene_lit(rgb : vec3<f32>, n : vec3<f32>, p : vec3<f32>) -> vec3<f32> {
    let s = mark_linear_to_srgb(rgb);
    let v = normalize(frame.eye.xyz - p * frame.eye.w);
    var c = layer.style2.x * s * frame.scene_ambient.rgb;
    for (var i = 0; i < 4; i++) {
        if (f32(i) < frame.scene_counts.x) {
            let d = frame.point_pos[i].xyz - p;
            let dist = length(d);
            let a = frame.point_att[i].xyz;
            let att = max(a.x + a.y * dist + a.z * dist * dist, 1e-6);
            c += shape_phong_light(s, d / max(dist, 1e-9), v, n, frame.point_color[i].rgb / att);
        }
        if (f32(i) < frame.scene_counts.y) {
            c += shape_phong_light(s, -frame.dir_dir[i].xyz, v, n, frame.dir_color[i].rgb);
        }
    }
    return mark_srgb_to_linear(c);
}

/// The shape's colour at this fragment: lit per pixel by the scene's
/// lights when its layer asks (`material.w` 2), as the vertex lit it
/// otherwise. `flat` is the face's normal, for meshes without normals.
fn shape_shade(in : ShapeOut, flat : vec3<f32>) -> vec3<f32> {
    if (layer.material.w > 1.5 && layer.material.w < 2.5) {
        var n = in.normal;
        if (dot(n, n) < 1e-12) {
            n = flat;
        }
        return shape_scene_lit(in.color.rgb, normalize(n), in.world);
    }
    return in.color.rgb;
}

/// The face's normal from the screen derivatives, towards the camera.
fn shape_flat_normal(world : vec3<f32>) -> vec3<f32> {
    let n = normalize(cross(dpdx(world), dpdy(world)));
    let v = frame.eye.xyz - world * frame.eye.w;
    return select(-n, n, dot(n, v) >= 0.0);
}

// A textured shape's atlas (`ShapeMarks::set_texture`); only `fs_pattern`
// reads it.
@group(2) @binding(0) var shape_texture : texture_2d<f32>;
@group(2) @binding(1) var shape_sampler : sampler;

/// `v` turned by the unit quaternion `q`.
fn shape_rotate(q : vec4<f32>, v : vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

@vertex
fn vs_main(in : ShapeIn) -> ShapeOut {
    var i : MarkInstance;
    i.position = mark_relative(in.pos_hi, in.pos_lo);
    i.position2 = i.position;
    i.size = 1.0;
    i.color = mark_decode(in.color) * in.vertex_color;
    i.color2 = i.color;
    i.custom0 = /*CUSTOM0*/;
    i.custom1 = /*CUSTOM1*/;
    i.visible = 1.0;
    i.index = in.iid;
    i = hook_instance(i);

    var out : ShapeOut;
    out.uv = in.uv;
    // An anchor inside the globe is no reason to hide a mesh: one round
    // the earth's centre (an earth) shows what faces the eye.
    let culled = mark_culled(i.position) && !(mark_inside_globe(i.position) && !mark_clipped(i.position));
    if (i.visible <= 0.0 || culled) {
        out.clip = mark_hidden();
        return out;
    }
    let scale = in.scale * i.size * layer.style.x;
    let q = in.rotation;
    let world = i.position + shape_rotate(q, in.vertex * scale);
    out.clip = mark_order(frame.view_proj * vec4<f32>(world, 1.0));
    // A normal transforms by the inverse scale; zero scale leaves it as is.
    let safe = select(scale, vec3<f32>(1.0), abs(scale) < vec3<f32>(1e-12));
    let raw = shape_rotate(q, in.normal / safe);
    out.world = world;
    out.normal = select(vec3<f32>(0.0), normalize(raw), dot(raw, raw) > 1e-12);
    // Geometry without normals is drawn unlit; lit by the scene, it is lit
    // per pixel instead.
    var rgb = i.color.rgb;
    let scene_lit = layer.material.w > 1.5 && layer.material.w < 2.5;
    if (dot(raw, raw) > 1e-12 && !scene_lit) {
        let n = normalize(raw);
        let ambient = layer.style2.x;
        rgb = mark_lit(rgb, n, world, ambient + (1.0 - ambient) * max(dot(n, frame.light.xyz), 0.0));
    }
    out.color = vec4<f32>(rgb, i.color.a);
    out.custom0 = i.custom0;
    out.custom1 = i.custom1;
    out.index = i.index;
    return out;
}

@fragment
fn fs_main(in : ShapeOut) -> @location(0) vec4<f32> {
    let flat = shape_flat_normal(in.world);
    let c = hook_color(vec4<f32>(shape_shade(in, flat), in.color.a), in.custom0, in.custom1);
    let shaded = mark_shadow(c, in.world);
    return mark_out(mark_highlight(vec4<f32>(shaded.rgb, shaded.a * layer.style2.w), in.index));
}

/// The shadow pass: the light's depth, packed.
@fragment
fn fs_shadow(in : ShapeOut) -> @location(0) vec4<f32> {
    return mark_shadow_depth(in.world);
}

/// A textured shape: the lit colour times the texture (an sRGB atlas,
/// sampled linear).
@fragment
fn fs_pattern(in : ShapeOut) -> @location(0) vec4<f32> {
    // style3: the image's place in the atlas; repeated past 0 and 1 (or
    // clamped, for a negative width), sampled with the unwrapped uv's
    // derivatives so the wrap leaves no seam.
    let rect = vec4<f32>(layer.style3.xy, abs(layer.style3.z), layer.style3.w);
    let repeat = layer.style3.z >= 0.0;
    let inside = select(clamp(in.uv, vec2<f32>(0.0), vec2<f32>(1.0)), fract(in.uv), repeat);
    // Half a texel in, so filtering stays within the image.
    let size = vec2<f32>(textureDimensions(shape_texture));
    let half = 0.5 / size;
    let at = rect.xy + clamp(inside * rect.zw, half, rect.zw - half);
    var t = textureSampleGrad(shape_texture, shape_sampler, at, dpdx(in.uv) * rect.zw, dpdy(in.uv) * rect.zw);
    // ids.z bit 3: a frame the marks drew — premultiplied sRGB bytes.
    if (((layer.ids.z >> 3u) & 1u) != 0u) {
        // Nothing drawn there: no surface (nor depth) at all.
        if (t.a < 0.5 / 255.0) {
            discard;
        }
        t = vec4<f32>(mark_srgb_to_linear(t.rgb / max(t.a, 1e-6)), t.a);
    }
    let flat = shape_flat_normal(in.world);
    // Lit by the scene, the texture is part of the surface's colour.
    var surface = in;
    surface.color = vec4<f32>(in.color.rgb * select(vec3<f32>(1.0), t.rgb, layer.material.w > 1.5 && layer.material.w < 2.5), in.color.a);
    let c = hook_color(vec4<f32>(shape_shade(surface, flat), in.color.a), in.custom0, in.custom1);
    let rgb = c.rgb * select(t.rgb, vec3<f32>(1.0), layer.material.w > 1.5 && layer.material.w < 2.5);
    let shaded = mark_shadow(vec4<f32>(rgb, c.a * t.a), in.world);
    return mark_out(mark_highlight(vec4<f32>(shaded.rgb, shaded.a * layer.style2.w), in.index));
}

@fragment
fn fs_pick(in : ShapeOut) -> @location(0) vec2<u32> {
    return vec2<u32>(layer.ids.x, in.index);
}
"#;

const POLYGON: &str = r#"
struct PolyIn {
    @location(0) pos_hi : vec3<f32>,
    @location(1) pos_lo : vec3<f32>,
    // xyz: the normal; w: 1 where the vertex rises with the elevation.
    @location(2) normal_top : vec4<f32>,
    @location(3) feature : f32,
    @location(4) elevation : f32,
    @location(5) color : vec4<f32>,
    // The colour behind a fill pattern.
    @location(6) background : vec4<f32>,
    /*CUSTOM_IN*/
};

struct PolyOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) color : vec4<f32>,
    @location(1) custom0 : vec4<f32>,
    @location(2) custom1 : vec4<f32>,
    @location(3) @interpolate(flat) index : u32,
    @location(4) world : vec3<f32>,
    // The vertex in the layer's own frame (local positions): what a fill
    // pattern tiles over.
    @location(5) local : vec3<f32>,
    @location(6) @interpolate(flat) background : vec4<f32>,
};

// A fill pattern's atlas (`PolygonMarks::set_pattern`); only `fs_pattern`
// uses it, so other pipelines bind nothing here.
@group(2) @binding(0) var pattern_texture : texture_2d<f32>;
@group(2) @binding(1) var pattern_sampler : sampler;

@vertex
fn vs_main(in : PolyIn) -> PolyOut {
    var i : MarkInstance;
    i.position = mark_relative(in.pos_hi, in.pos_lo);
    i.position2 = i.position;
    i.size = in.elevation;
    i.color = mark_decode(in.color);
    i.color2 = i.color;
    i.custom0 = /*CUSTOM0*/;
    i.custom1 = /*CUSTOM1*/;
    i.visible = 1.0;
    i.index = u32(in.feature + 0.5);
    i = hook_instance(i);

    var out : PolyOut;
    if (i.visible <= 0.0) {
        out.clip = mark_hidden();
        return out;
    }
    let rise = i.size * layer.style.x * in.normal_top.w;
    // Up is +Y on the flat ground, away from the centre (style3.xyz) on a
    // globe — chosen with `select`, not a branch, beside mark_relative.
    let radial = i.position - layer.style3.xyz;
    let up = select(vec3<f32>(0.0, 1.0, 0.0), radial * inverseSqrt(max(dot(radial, radial), 1e-12)), layer.style3.w > 0.5);
    out.clip = mark_order(frame.view_proj * vec4<f32>(i.position + up * rise, 1.0));
    out.world = i.position;
    out.local = in.pos_hi + in.pos_lo;
    // Lit relative to a roof, so a roof shows its colour exactly and a
    // choropleth's legend reads true; walls darken as they turn away.
    let l = frame.light.xyz;
    let facing = clamp(dot(in.normal_top.xyz, l) / max(l.y, 0.05), 0.0, 1.0);
    let ambient = layer.style2.x;
    // Passed on sRGB-encoded, so colours that differ per vertex blend
    // across a triangle as deck.gl's do, on sRGB values.
    let lit = mark_lit(i.color.rgb, in.normal_top.xyz, i.position + up * rise, ambient + (1.0 - ambient) * facing);
    out.color = vec4<f32>(mark_linear_to_srgb(lit), i.color.a);
    out.custom0 = i.custom0;
    out.custom1 = i.custom1;
    out.index = i.index;
    out.background = mark_decode(in.background);
    return out;
}

// deck.gl's `fill_blendOverBackground`: the pattern over its background,
// before the layer's opacity, on sRGB values with `mix_srgb`.
/// The ground point in pattern axes (x east, y down the pattern).
fn poly_pattern_local(in : PolyOut) -> vec2<f32> {
    let flip = select(-1.0, 1.0, in.custom1.w > 0.0);
    return vec2<f32>(in.world.x, in.world.z * flip);
}

fn poly_procedural_alpha(in : PolyOut, grad : vec4<f32>) -> f32 {
    // style3.xy: where pattern space starts (flat ground only).
    let origin = fill_pattern_origin(in.custom1, layer.style3.xy);
    return fill_procedural_alpha(in.custom0, in.custom1, poly_pattern_local(in), origin, grad);
}

@fragment
fn fs_main(in : PolyOut) -> @location(0) vec4<f32> {
    // Derivatives first, in uniform control flow.
    let pl = poly_pattern_local(in);
    let grad = vec4<f32>(dpdx(pl), dpdy(pl));
    if (mark_culled(in.world)) {
        discard;
    }
    var c = hook_color(mark_decode(in.color), in.custom0, in.custom1);
    if (((layer.ids.z >> 3u) & 1u) != 0u) {
        c.a *= poly_procedural_alpha(in, grad);
        c = fill_over_background(c, in.background);
    }
    c = mark_shadow(c, in.world);
    return mark_out(mark_highlight(vec4<f32>(c.rgb, c.a * layer.style2.w), in.index));
}

/// The shadow pass: the light's depth, packed.
@fragment
fn fs_shadow(in : PolyOut) -> @location(0) vec4<f32> {
    return mark_shadow_depth(in.world);
}

// The fill tiled with its feature's pattern: `custom0` is the frame in the
// atlas (uv origin and size, 0 width for none), `custom1` the tile's size
// in the layer's units and the layer's origin reduced to within a tile
// (offset included), x and z. `material.z`: the pattern's alpha masks the
// fill colour rather than replacing it.
@fragment
fn fs_pattern(in : PolyOut) -> @location(0) vec4<f32> {
    let frame = in.custom0;
    let tile = in.custom1;
    // Derivatives first, in uniform control flow: of the unwrapped
    // coordinate, as the wrap would pick the smallest mip at every tile edge.
    let ty = select(min(tile.y, -1e-12), max(tile.y, 1e-12), tile.y >= 0.0);
    let uv = vec2<f32>((in.local.x + tile.z) / max(tile.x, 1e-12), (in.local.z + tile.w) / ty);
    let ddx = dpdx(uv) * frame.zw;
    let ddy = dpdy(uv) * frame.zw;
    let p = textureSampleGrad(pattern_texture, pattern_sampler, frame.xy + frame.zw * fract(uv), ddx, ddy);
    if (mark_culled(in.world)) {
        discard;
    }
    var c = mark_decode(in.color);
    if (frame.z > 0.0 && tile.x > 0.0 && tile.y != 0.0) {
        c = select(vec4<f32>(p.rgb, c.a * p.a), vec4<f32>(c.rgb, c.a * p.a), layer.material.z > 0.5);
        c = fill_over_background(c, in.background);
    }
    return mark_out(mark_highlight(vec4<f32>(c.rgb, c.a * layer.style2.w), in.index));
}

@fragment
fn fs_pick(in : PolyOut) -> @location(0) vec2<u32> {
    if (mark_culled(in.world)) {
        discard;
    }
    return vec2<u32>(layer.ids.x, in.index);
}
"#;

const SPRITE: &str = r#"
@group(2) @binding(0) var atlas_tex : texture_2d<f32>;
@group(2) @binding(1) var atlas_samp : sampler;

struct SpriteIn {
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32,
    @location(0) pos_hi : vec3<f32>,
    @location(1) pos_lo : vec3<f32>,
    // The size, and the image's centre relative to the anchored point in
    // the same units (y down).
    @location(2) size : vec3<f32>,
    // x, y, width, height in atlas pixels.
    @location(3) rect : vec4<f32>,
    // The angle, and a pixel offset after turning (y down).
    @location(4) angle : vec3<f32>,
    @location(5) color : vec4<f32>,
    // x: 0 when collision placement hid the icon. depth: each corner's
    // depth offset, in the quad's corner order.
    @location(6) shown : vec4<f32>,
    @location(7) depth : vec4<f32>,
    /*CUSTOM_IN*/
};

struct SpriteOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) uv : vec2<f32>,
    @location(1) color : vec4<f32>,
    @location(2) custom0 : vec4<f32>,
    @location(3) custom1 : vec4<f32>,
    @location(4) @interpolate(flat) index : u32,
    // Pixels from the image's centre, unturned; and its half size.
    @location(5) local : vec2<f32>,
    @location(6) @interpolate(flat) half : vec2<f32>,
    // A box's border dash and gap (its rect's x and y).
    @location(7) @interpolate(flat) dash : vec2<f32>,
};

@vertex
fn vs_main(in : SpriteIn) -> SpriteOut {
    var i : MarkInstance;
    i.position = mark_relative(in.pos_hi, in.pos_lo);
    i.position2 = i.position;
    i.size = in.size.x;
    i.color = mark_decode(in.color);
    i.color2 = i.color;
    i.custom0 = /*CUSTOM0*/;
    i.custom1 = /*CUSTOM1*/;
    i.visible = in.shown.x;
    i.index = in.iid;
    i = hook_instance(i);

    var out : SpriteOut;
    let clip = frame.view_proj * vec4<f32>(i.position, 1.0);
    if (i.visible <= 0.0 || clip.w <= 0.0 || in.rect.w <= 0.0 || mark_culled(i.position)) {
        out.clip = mark_hidden();
        return out;
    }
    let h = mark_size_px(i.size, clip.w);
    // The row's offset, scaled as the size was (y up from here on).
    let unit = select(0.0, h / i.size, abs(i.size) > 0.0);
    let offset = vec2<f32>(in.size.y, -in.size.z) * unit;
    // A box's quad reaches half a pixel past it, for its smooth edge.
    let pad = select(0.0, 1.0, layer.style2.x > 2.5);
    let half = vec2<f32>(h * in.rect.z / in.rect.w, h) * 0.5;
    let c = mark_corner_signed(mark_quad_corner(in.vid));
    let grow = (half + vec2<f32>(pad)) / max(half, vec2<f32>(1e-6));
    // style3.zw: the anchor, the image point that sits on the position.
    let local = (c * grow - layer.style3.zw) * half + offset;
    let s = sin(in.angle.x);
    let k = cos(in.angle.x);
    let turned = vec2<f32>(k * local.x - s * local.y, s * local.x + k * local.y)
        + vec2<f32>(in.angle.y, -in.angle.z);
    // style2.z: flat on the ground, the pixel offsets laid out in the world
    // at the size they have at the focal plane (y up, north toward -z).
    let world_per_px = clip.w / max(frame.misc.x, 1e-6);
    let on_ground = frame.view_proj
        * vec4<f32>(i.position + vec3<f32>(turned.x, 0.0, -turned.y) * world_per_px, 1.0);
    let facing = mark_lift_depth(mark_offset_px(clip, turned), i.position, h * 0.5);
    out.clip = mark_order(select(facing, on_ground, layer.style2.z > 0.5));
    // The corner's depth offset (towards the far plane; reversed depth
    // runs the other way, `frame.light.w`).
    let toward_far = select(1.0, -1.0, frame.light.w > 0.5);
    out.clip.z += toward_far * in.depth[mark_quad_corner(in.vid)] * out.clip.w;
    out.local = c * grow * half;
    out.half = half;
    out.dash = in.rect.xy;
    // Image y runs down; the quad's runs up. style3.xy: the atlas size.
    let t = vec2<f32>(c.x * grow.x * 0.5 + 0.5, 0.5 - c.y * grow.y * 0.5);
    out.uv = (in.rect.xy + t * in.rect.zw) / max(layer.style3.xy, vec2<f32>(1.0));
    out.color = i.color;
    out.custom0 = i.custom0;
    out.custom1 = i.custom1;
    out.index = i.index;
    return out;
}

/// How far along a box's edge `local` (pixels from its centre, y up) is,
/// in pixels, as deck.gl measures it: from the top-left corner, down the
/// left side, along the bottom, up the right and back along the top, round
/// corners of radius `r` as arcs.
fn sprite_box_perimeter(local : vec2<f32>, half : vec2<f32>, r : f32) -> f32 {
    let size = half * 2.0;
    // deck.gl's frame: x right, y down from the top-left.
    let p = vec2<f32>(local.x + half.x, half.y - local.y);
    let pi = 3.14159265;
    let left = size.y - 2.0 * r;
    let top = size.x - 2.0 * r;
    let arc = pi * 0.5 * r;
    let d = min(min(p.x, size.x - p.x), min(p.y, size.y - p.y));
    if (r > 0.0) {
        if (p.x < r && p.y < r) {
            let q = p - vec2<f32>(r, r);
            return atan2(-q.x, -q.y) / (pi * 0.5) * arc;
        }
        if (p.x < r && p.y > size.y - r) {
            let q = p - vec2<f32>(r, size.y - r);
            return arc + left + atan2(q.y, -q.x) / (pi * 0.5) * arc;
        }
        if (p.x > size.x - r && p.y > size.y - r) {
            let q = p - vec2<f32>(size.x - r, size.y - r);
            return 2.0 * arc + left + top + atan2(q.x, q.y) / (pi * 0.5) * arc;
        }
        if (p.x > size.x - r && p.y < r) {
            let q = p - vec2<f32>(size.x - r, r);
            return 3.0 * arc + 2.0 * left + top + atan2(-q.y, q.x) / (pi * 0.5) * arc;
        }
    }
    if (d == p.x) {
        return arc + clamp(p.y - r, 0.0, left);
    }
    if (d == size.y - p.y) {
        return 2.0 * arc + left + clamp(p.x - r, 0.0, top);
    }
    if (d == size.x - p.x) {
        return 3.0 * arc + left + top + clamp(size.y - r - p.y, 0.0, left);
    }
    return 4.0 * arc + 2.0 * left + top + clamp(size.x - r - p.x, 0.0, top);
}

/// The sprite's colour before the hook: style2.x picks colour, mask or
/// distance field.
fn sprite_color(in : SpriteOut) -> vec4<f32> {
    // The mip level as GL's specification picks it — the longer of the
    // texel footprint's two screen derivatives — which WebGL's software
    // rasterizer follows to the letter and hardware samplers only
    // approximate (most visibly on a pitched view).
    let texels = vec2<f32>(textureDimensions(atlas_tex, 0));
    let dx = dpdx(in.uv) * texels;
    let dy = dpdy(in.uv) * texels;
    let lod = 0.5 * log2(max(max(dot(dx, dx), dot(dy, dy)), 1e-12));
    let t = textureSampleLevel(atlas_tex, atlas_samp, in.uv, lod);
    // The ramp's half width, before any branch (derivatives must be taken
    // in uniform control flow).
    let auto_w = max(fwidth(t.a), 1e-4) * 0.7;
    if (layer.style2.x < 0.5) {
        // extra.w: sRGB bytes in a linear texture, filtered as they are.
        if (layer.extra.w > 0.5) {
            return vec4<f32>(mark_srgb_to_linear(t.rgb), t.a) * in.color;
        }
        return t * in.color;
    }
    if (layer.style2.x < 1.5) {
        return vec4<f32>(in.color.rgb, in.color.a * t.a);
    }
    if (layer.style2.x > 2.5) {
        // A rounded box: distance to its edge in pixels (extra.x the
        // corner radius, extra.y the border; material the border colour).
        let r = min(layer.extra.x, min(in.half.x, in.half.y));
        let q = abs(in.local) - in.half + vec2<f32>(r);
        let d = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
        let inside = clamp(0.5 - d, 0.0, 1.0);
        var c = in.color;
        if (layer.extra.y > 0.0) {
            var border = clamp(d + layer.extra.y + 0.5, 0.0, 1.0);
            // A dashed border: in its gaps, the fill — deck.gl's
            // PathStyleExtension on a text background.
            let period = in.dash.x + in.dash.y;
            if (period > 0.0 && -d <= layer.extra.y && -d >= 0.0) {
                let along = sprite_box_perimeter(in.local, in.half, r) / layer.extra.y;
                if (along - period * floor(along / period) > in.dash.x) {
                    border = 0.0;
                    if (c.a <= 0.0) {
                        return vec4<f32>(0.0);
                    }
                }
            }
            c = mix(c, layer.material, border);
        }
        return vec4<f32>(c.rgb, c.a * inside);
    }
    // extra: the edge, the ramp's half width (0: a pixel) and the outline's
    // edge (-1: none); material the outline's colour.
    let w = select(auto_w, layer.extra.y, layer.extra.y > 0.0);
    let fill = smoothstep(layer.extra.x - w, layer.extra.x + w, t.a);
    if (layer.extra.z >= 0.0) {
        let border = smoothstep(layer.extra.z - w, layer.extra.z + w, t.a);
        // On sRGB values with `mix_srgb`, as deck.gl's shader mixes them.
        let c = mark_mix(layer.material, in.color, fill);
        return vec4<f32>(c.rgb, c.a * border);
    }
    return vec4<f32>(in.color.rgb, in.color.a * fill);
}

@fragment
fn fs_main(in : SpriteOut) -> @location(0) vec4<f32> {
    let c = hook_color(sprite_color(in), in.custom0, in.custom1);
    // style2.y: the alpha cutoff, after opacity (style2.w).
    if (c.a <= 0.0 || c.a * layer.style2.w < layer.style2.y) {
        discard;
    }
    return mark_out(mark_highlight(vec4<f32>(c.rgb, c.a * layer.style2.w), in.index));
}

@fragment
fn fs_pick(in : SpriteOut) -> @location(0) vec2<u32> {
    if (sprite_color(in).a < 0.5) {
        discard;
    }
    return vec2<u32>(layer.ids.x, in.index);
}
"#;

const IMAGE: &str = r#"
@group(2) @binding(0) var atlas_tex : texture_2d<f32>;
@group(2) @binding(1) var atlas_samp : sampler;

struct ImageIn {
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32,
    @location(0) pos_hi : vec3<f32>,
    @location(1) pos_lo : vec3<f32>,
    @location(2) right : vec3<f32>,
    @location(3) down : vec3<f32>,
    @location(4) rect : vec4<f32>,
    @location(5) color : vec4<f32>,
    /*CUSTOM_IN*/
};

struct ImageOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) uv : vec2<f32>,
    @location(1) color : vec4<f32>,
    @location(2) custom0 : vec4<f32>,
    @location(3) custom1 : vec4<f32>,
    @location(4) @interpolate(flat) index : u32,
    @location(5) world : vec3<f32>,
};

@vertex
fn vs_main(in : ImageIn) -> ImageOut {
    var i : MarkInstance;
    i.position = mark_relative(in.pos_hi, in.pos_lo);
    i.position2 = i.position + in.right + in.down;
    i.size = 1.0;
    i.color = mark_decode(in.color);
    i.color2 = i.color;
    i.custom0 = /*CUSTOM0*/;
    i.custom1 = /*CUSTOM1*/;
    i.visible = 1.0;
    i.index = in.iid;
    i = hook_instance(i);

    var out : ImageOut;
    let n = max(u32(layer.style2.x), 1u);
    let cell = in.vid / 6u;
    if (i.visible <= 0.0 || cell >= n * n) {
        out.clip = mark_hidden();
        return out;
    }
    let c = mark_corner_signed(mark_quad_corner(in.vid)) * 0.5 + 0.5;
    // s across, t down, both 0..1 over the whole quad.
    let s = (f32(cell % n) + c.x) / f32(n);
    let t = (f32(cell / n) + 1.0 - c.y) / f32(n);
    let flat = i.position + in.right * s + in.down * t;
    // On a sphere (stroke.w its radius, stroke.xyz its centre) every point of
    // the grid is pushed out onto it, so a tile bends with the globe.
    let centre = layer.stroke.xyz;
    let p = select(flat, centre + normalize(flat - centre) * layer.stroke.w, layer.stroke.w > 0.0);
    out.clip = mark_order(frame.view_proj * vec4<f32>(p, 1.0));
    out.world = p;
    out.uv = (in.rect.xy + vec2<f32>(s, t) * in.rect.zw) / max(layer.style3.xy, vec2<f32>(1.0));
    out.color = i.color;
    out.custom0 = i.custom0;
    out.custom1 = i.custom1;
    out.index = i.index;
    return out;
}

@fragment
fn fs_main(in : ImageOut) -> @location(0) vec4<f32> {
    if (mark_culled(in.world)) {
        discard;
    }
    let c = hook_color(textureSample(atlas_tex, atlas_samp, in.uv) * in.color, in.custom0, in.custom1);
    if (c.a <= 0.002) {
        discard;
    }
    return mark_out(mark_highlight(vec4<f32>(c.rgb, c.a * layer.style2.w), in.index));
}

@fragment
fn fs_pick(in : ImageOut) -> @location(0) vec2<u32> {
    if (mark_culled(in.world)) {
        discard;
    }
    if ((textureSample(atlas_tex, atlas_samp, in.uv) * in.color).a < 0.5) {
        discard;
    }
    return vec2<u32>(layer.ids.x, in.index);
}
"#;

const GLYPH: &str = r#"
@group(2) @binding(0) var glyph_atlas : texture_2d<f32>;
@group(2) @binding(1) var glyph_sampler : sampler;

struct GlyphIn {
    @builtin(vertex_index) vid : u32,
    @location(0) pos_hi : vec3<f32>,
    @location(1) pos_lo : vec3<f32>,
    // Pixel offset of the glyph's top-left from the anchor, and its size.
    @location(2) rect : vec4<f32>,
    // Atlas uv of the top-left and bottom-right.
    @location(3) uv : vec4<f32>,
    @location(4) color : vec4<f32>,
    // (visible, label index, angle in radians on screen with y down, the
    // field's texel size relative to a letter's — for the halo's width)
    @location(5) state : vec4<f32>,
};

struct GlyphOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) color : vec4<f32>,
    @location(1) uv : vec2<f32>,
    @location(2) @interpolate(flat) index : u32,
    @location(3) @interpolate(flat) halo_scale : f32,
};

@vertex
fn vs_main(in : GlyphIn) -> GlyphOut {
    var i : MarkInstance;
    i.position = mark_relative(in.pos_hi, in.pos_lo);
    i.position2 = i.position;
    i.size = 1.0;
    i.color = mark_decode(in.color);
    i.color2 = i.color;
    i.custom0 = vec4<f32>(0.0);
    i.custom1 = vec4<f32>(0.0);
    i.visible = in.state.x;
    i.index = u32(in.state.y);
    i = hook_instance(i);

    var out : GlyphOut;
    let clip = frame.view_proj * vec4<f32>(i.position, 1.0);
    if (i.visible <= 0.0 || clip.w <= 0.0 || mark_culled(i.position)) {
        out.clip = mark_hidden();
        return out;
    }
    let k = mark_quad_corner(in.vid);
    let c = vec2<f32>(select(0.0, 1.0, k == 1u || k == 2u), select(1.0, 0.0, k >= 2u));
    // Layout is in screen pixels with y down; clip space has y up.
    // A glyph of a label set along a line turns about its anchor.
    let unturned = (in.rect.xy + c * in.rect.zw) * i.size;
    let sn = sin(in.state.z);
    let cs = cos(in.state.z);
    let px = vec2<f32>(unturned.x * cs - unturned.y * sn, unturned.x * sn + unturned.y * cs);
    // Snap the anchor to a pixel so text does not shimmer as it moves.
    let ndc = clip.xy / clip.w;
    let sp = (ndc * 0.5 + 0.5) * frame.viewport.xy;
    let snapped = round(sp) * frame.viewport.zw * 2.0 - 1.0;
    let base = vec4<f32>(snapped * clip.w, clip.z, clip.w);
    // Lift by half an em (`style2.y`): the text's height, not its length.
    out.clip = mark_order(mark_lift_depth(
        mark_offset_px(base, vec2<f32>(px.x, -px.y)),
        i.position,
        0.5 * layer.style2.y,
    ));
    out.uv = mix(in.uv.xy, in.uv.zw, c);
    out.color = i.color;
    out.index = i.index;
    // Its sign tells the halo copy (negative) from the fill.
    out.halo_scale = select(max(in.state.w, 1e-3), min(in.state.w, -1e-3), in.state.w < 0.0);
    return out;
}

/// (fill, outer) coverage: the fill from `fill_d` — the multi-channel
/// median, or the distance — and the halo from the true distance `true_d`.
/// `halo` is the halo's width in distance-field units, `px` the screen
/// pixels one field unit spans: each edge is a linear ramp one pixel wide,
/// centred on the outline, which is what a box filter over the true shape
/// gives a straight edge.
fn glyph_coverage(fill_d : f32, true_d : f32, halo : f32, px : f32) -> vec2<f32> {
    let fill = clamp((fill_d - 0.5) * px + 0.5, 0.0, 1.0);
    let edge = 0.5 - halo;
    let outer = max(clamp((true_d - edge) * px + 0.5, 0.0, 1.0), fill);
    return vec2<f32>(fill, outer);
}

/// Coverage `c` of text in linear colour `rgb`, bent so a blend in linear
/// light lands where a blend in the display's encoding would against the
/// background contrasting most with the text — as CoreText and Skia draw
/// text, and as the CPU caption painter does. Unbent, dark text reads thin
/// and light text fat. Mid-grey has no contrasting side and is left alone.
fn glyph_text_coverage(c : f32, rgb : vec3<f32>) -> f32 {
    let src_l = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let src = mark_linear_to_srgb(vec3<f32>(src_l)).x;
    let dst = 1.0 - src;
    if (abs(src - dst) < 1.0 / 64.0) {
        return c;
    }
    let dst_l = mark_srgb_to_linear(vec3<f32>(dst)).x;
    let mixed = mark_srgb_to_linear(vec3<f32>(src * c + dst * (1.0 - c))).x;
    return clamp((mixed - dst_l) / (src_l - dst_l), 0.0, 1.0);
}

@fragment
fn fs_main(in : GlyphOut) -> @location(0) vec4<f32> {
    let t = textureSample(glyph_atlas, glyph_sampler, in.uv);
    // style2.z: 1 when rgb is a multi-channel field; its median is the
    // distance, with corners kept. `a` is always the plain distance.
    let m = max(min(t.r, t.g), min(max(t.r, t.g), t.b));
    let d = select(t.a, m, layer.style2.z > 0.5);
    // Screen pixels per field unit, from the field's range in atlas texels
    // (style3.x) over the texels a screen pixel covers — measured from the
    // texture coordinates, not from `fwidth` of the field, which jumps
    // where the median switches channel and blurred every corner. The
    // geometric mean of the two axes keeps a turned label as sharp as a
    // level one; at least one pixel over the whole range, so text drawn
    // smaller than its field softens instead of aliasing.
    let texels = vec2<f32>(textureDimensions(glyph_atlas));
    let per_px = sqrt(max(length(dpdx(in.uv) * texels) * length(dpdy(in.uv) * texels), 1e-8));
    let px = max(layer.style3.x / per_px, 1.0);
    let cov = glyph_coverage(d, t.a, min(layer.style2.x / abs(in.halo_scale), 0.5), px);
    let text = hook_color(in.color, vec4<f32>(0.0), vec4<f32>(0.0));
    // A halo is drawn as its own copy of the glyphs, all under the fills.
    let is_halo = in.halo_scale < 0.0;
    // Bent only where the target blends in linear: an encoding pass
    // (misc.y) blends the encoded values already.
    var fill_cov = cov.x;
    var halo_cov = cov.y;
    if (frame.misc.y < 0.5) {
        fill_cov = glyph_text_coverage(fill_cov, text.rgb);
        halo_cov = glyph_text_coverage(halo_cov, layer.stroke.rgb);
    }
    let a = select(text.a * fill_cov, layer.stroke.a * halo_cov, is_halo);
    if (a <= 0.002) {
        discard;
    }
    let rgb = select(text.rgb, layer.stroke.rgb, is_halo);
    return mark_out(mark_highlight(vec4<f32>(rgb, a * layer.style2.w), in.index));
}

@fragment
fn fs_pick(in : GlyphOut) -> @location(0) vec2<u32> {
    // The whole glyph box picks, not only the ink: aiming at a thin letter
    // should not miss the label.
    return vec2<u32>(layer.ids.x, in.index);
}
"#;

const SPLAT: &str = r#"
struct SplatIn {
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32,
    @location(0) pos_hi : vec3<f32>,
    @location(1) pos_lo : vec3<f32>,
    @location(2) weight : f32,
    /*CUSTOM_IN*/
};

struct SplatOut {
    @builtin(position) clip : vec4<f32>,
    // Position across the kernel in radii, and the weight.
    @location(0) local : vec2<f32>,
    @location(1) weight : f32,
};

@vertex
fn vs_main(in : SplatIn) -> SplatOut {
    var i : MarkInstance;
    i.position = mark_relative(in.pos_hi, in.pos_lo);
    i.position2 = i.position;
    // The radius comes from the layer; `size` scales it.
    i.size = 1.0;
    // Splats carry their weight in alpha.
    i.color = vec4<f32>(1.0, 1.0, 1.0, in.weight);
    i.color2 = i.color;
    i.custom0 = /*CUSTOM0*/;
    i.custom1 = /*CUSTOM1*/;
    i.visible = 1.0;
    i.index = in.iid;
    i = hook_instance(i);

    var out : SplatOut;
    let clip = frame.view_proj * vec4<f32>(i.position, 1.0);
    if (i.visible <= 0.0 || clip.w <= 0.0 || mark_culled(i.position)) {
        out.clip = mark_hidden();
        return out;
    }
    let r = mark_size_px(i.size, clip.w);
    let corner = mark_corner_signed(mark_quad_corner(in.vid));
    out.clip = mark_order(mark_offset_px(clip, corner * r));
    out.local = corner;
    out.weight = i.color.a;
    return out;
}

@fragment
fn fs_main(in : SplatOut) -> @location(0) vec4<f32> {
    let d2 = dot(in.local, in.local);
    if (d2 > 1.0) {
        discard;
    }
    // A Gaussian with sigma a third of the radius, cut off at the radius.
    let g = exp(-4.5 * d2);
    // Alpha: the kernels' coverage, for a mean (style2.y: `mean`).
    return vec4<f32>(in.weight * g, 0.0, 0.0, g * layer.style2.y);
}
"#;

/// Maps the summed density through a colour ramp onto the target.
pub(crate) const DENSITY_COMPOSITE: &str = r#"
@group(2) @binding(0) var density_field : texture_2d<f32>;
@group(2) @binding(1) var density_ramp : texture_2d<f32>;
@group(2) @binding(2) var density_sampler : sampler;
@group(2) @binding(3) var density_max : texture_2d<f32>;

@fragment
fn fs_main(in : FullscreenVertex) -> @location(0) vec4<f32> {
    // The field may be coarser than the target; it is smooth, so filtering
    // it up loses nothing.
    // aa.zw: where the target's rectangle begins.
    let uv = (in.clip.xy - frame.aa.zw) * frame.viewport.zw;
    let v = density_value(textureSampleLevel(density_field, density_sampler, uv, 0.0));
    // style3: (weight that maps to the top of the ramp, fade-in threshold
    // as a fraction of the ramp, 1 to scale to the field's own maximum, the
    // fraction of it that does). `style` is the kernel's size.
    let measured = textureSampleLevel(density_max, density_sampler, vec2<f32>(0.5), 0.0).r;
    let top = select(layer.style3.x, measured * layer.style3.w, layer.style3.z > 0.5);
    let t = clamp(v / max(top, 1e-12), 0.0, 1.0);
    let c = textureSampleLevel(density_ramp, density_sampler, vec2<f32>(t, 0.5), 0.0);
    // style2.x: fade in linearly rather than along a smoothstep.
    let edge = max(layer.style3.y, 1e-6);
    let a = select(smoothstep(0.0, edge, t), min(t / edge, 1.0), layer.style2.x > 0.5) * c.a;
    if (a <= 0.0) {
        discard;
    }
    return mark_out(vec4<f32>(c.rgb, a * layer.style2.w));
}
"#;

fn kind_source(kind: MarkKind) -> (&'static str, u32) {
    // (source, first custom location)
    match kind {
        MarkKind::Custom(_) => unreachable!("a custom kind brings its own source"),
        MarkKind::Point => (POINT, 7),
        MarkKind::Segment => (SEGMENT, 9),
        MarkKind::Shape => (SHAPE, 9),
        MarkKind::Glyph => (GLYPH, 6),
        MarkKind::Splat => (SPLAT, 3),
        MarkKind::Polygon => (POLYGON, 7),
        MarkKind::Sprite => (SPRITE, 8),
        MarkKind::Image => (IMAGE, 6),
    }
}

/// The first shader location custom columns take for `kind`.
pub(crate) fn custom_location(kind: MarkKind) -> u32 {
    kind_source(kind).1
}

/// The whole module for `kind`, with `custom` giving each custom column's
/// width (`0` for none).
pub(crate) fn module_source(kind: MarkKind, hooks: &ShaderHooks, custom: [u32; 2]) -> String {
    module_source_with(kind, hooks, custom, false)
}

/// [`module_source`] with `local` choosing the variant for local positions.
pub(crate) fn module_source_with(
    kind: MarkKind,
    hooks: &ShaderHooks,
    custom: [u32; 2],
    local: bool,
) -> String {
    let (body, first) = kind_source(kind);
    assemble(body, first, hooks, custom, local)
}

/// Drop every `@location(n) name_lo : vec3<f32>,` input and read
/// `in.name_lo` as zero: the vertex layout of a layer whose position columns
/// are [`ColumnFormat::PositionLocal`](super::ColumnFormat::PositionLocal)
/// has no low parts.
fn localize(body: &str) -> String {
    let mut names = Vec::new();
    let mut out = String::with_capacity(body.len());
    for line in body.lines() {
        let t = line.trim();
        if t.starts_with("@location(") && t.ends_with("_lo : vec3<f32>,") {
            if let Some(name) = t.split(')').nth(1).and_then(|r| r.split(':').next()) {
                names.push(name.trim().to_string());
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    for n in names {
        out = out.replace(&format!("in.{n}"), "vec3<f32>(0.0)");
    }
    out
}

/// The renderer's tone-mapping curves, taken from its own shader so marks can
/// never drift from the scene they are drawn into, and `mark_tone` over them.
pub(crate) fn tone_wgsl() -> &'static str {
    static WGSL: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    WGSL.get_or_init(|| {
        let src = crate::renderer::shader::legacy_shader_source();
        let start = src
            .find("fn aces_tonemap(")
            .expect("the renderer's shader has an aces_tonemap");
        let end = src[start..]
            .find("// Exposure, then whichever curve")
            .map(|e| start + e)
            .expect("the renderer's shader ends its curves where marks expect");
        format!(
            "{}
fn mark_tone(c : vec3<f32>) -> vec3<f32> {{
    let tm = u32(frame.misc.z);
    let e = c * frame.misc.w;
    if (tm == 0u) {{ return c; }}
    if (tm == 1u) {{ return e; }}
    if (tm == 2u) {{ return aces_tonemap(e); }}
    if (tm == 3u) {{ return neutral_tonemap(e); }}
    return agx_tonemap(e);
}}
",
            &src[start..end]
        )
    })
}

fn assemble(body: &str, first: u32, hooks: &ShaderHooks, custom: [u32; 2], local: bool) -> String {
    let localized;
    let body = if local {
        localized = localize(body);
        localized.as_str()
    } else {
        body
    };
    let mut decl = String::new();
    let mut exprs = ["vec4<f32>(0.0)".to_string(), "vec4<f32>(0.0)".to_string()];
    let mut loc = first;
    for (slot, &n) in custom.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let ty = match n {
            1 => "f32",
            2 => "vec2<f32>",
            3 => "vec3<f32>",
            _ => "vec4<f32>",
        };
        decl.push_str(&format!("@location({loc}) custom{slot} : {ty},\n    "));
        exprs[slot] = match n {
            1 => format!("vec4<f32>(in.custom{slot}, 0.0, 0.0, 0.0)"),
            2 => format!("vec4<f32>(in.custom{slot}, 0.0, 0.0)"),
            3 => format!("vec4<f32>(in.custom{slot}, 0.0)"),
            _ => format!("in.custom{slot}"),
        };
        loc += 1;
    }
    let body = body
        .replace("/*CUSTOM_IN*/", &decl)
        .replace("/*CUSTOM0*/", &exprs[0])
        .replace("/*CUSTOM1*/", &exprs[1]);
    format!(
        "{PREAMBLE}\n{}\n// ---- hooks ----\n{}\n// ---- mark ----\n{body}",
        tone_wgsl(),
        hooks.source()
    )
}

/// One step of a density field's max reduction: each output texel the
/// largest of a 4×4 block of the level before. Render passes only, so it runs
/// where there is no compute stage.
/// A density field texel's value: the summed weight, or for a field with
/// coverage in alpha (a mean), the weight over the coverage — coverage
/// below `DENSITY_MEAN_FLOOR` counting as that much, as deck.gl's
/// `HeatmapLayer` floors its normalized kernels' sum at 1. Reduced levels
/// carry alpha 1, so their values pass through.
pub(crate) const DENSITY_VALUE: &str = r#"
const DENSITY_MEAN_FLOOR : f32 = 0.29540897;
fn density_value(t : vec4<f32>) -> f32 {
    return select(t.r, t.r / max(DENSITY_MEAN_FLOOR, t.a), t.a > 0.0);
}
"#;

pub(crate) const DENSITY_REDUCE: &str = r#"
@group(0) @binding(0) var reduce_src : texture_2d<f32>;
@group(0) @binding(1) var reduce_samp : sampler;

@fragment
fn fs_main(in : FullscreenVertex) -> @location(0) vec4<f32> {
    let dim = vec2<i32>(textureDimensions(reduce_src, 0));
    let base = vec2<i32>(in.clip.xy) * 4;
    var m = 0.0;
    for (var y = 0; y < 4; y = y + 1) {
        for (var x = 0; x < 4; x = x + 1) {
            let c = min(base + vec2<i32>(x, y), dim - vec2<i32>(1));
            let uv = (vec2<f32>(c) + vec2<f32>(0.5)) / vec2<f32>(dim);
            m = max(m, density_value(textureSampleLevel(reduce_src, reduce_samp, uv, 0.0)));
        }
    }
    return vec4<f32>(m, 0.0, 0.0, 1.0);
}
"#;

/// The density reduction module.
pub(crate) fn reduce_source() -> String {
    format!("{}\n{DENSITY_VALUE}\n{DENSITY_REDUCE}", crate::renderer::fullscreen::VS)
}

/// The density composite module.
pub(crate) fn composite_source() -> String {
    format!(
        "{PREAMBLE}\n{}\n{}\n{DENSITY_VALUE}\n{DENSITY_COMPOSITE}",
        tone_wgsl(),
        crate::renderer::fullscreen::VS
    )
}

/// 1-based line where the hooks begin in an assembled module.
pub(crate) fn hook_line(source: &str) -> usize {
    source
        .lines()
        .position(|l| l.starts_with("// ---- hooks ----"))
        .map(|i| i + 2)
        .unwrap_or(1)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn every_kind_compiles_with_each_custom_width() {
        for kind in HOOKED_KINDS {
            for w in 0..=4 {
                let src = module_source(*kind, &ShaderHooks::none(), [w, 4 - w]);
                crate::renderer::validate_custom_shader(&src, 0)
                    .unwrap_or_else(|e| panic!("{kind:?} [{w}, {}]: {e}", 4 - w));
            }
        }
    }

    #[test]
    fn every_kind_compiles_with_local_positions() {
        for kind in HOOKED_KINDS {
            let src = module_source_with(*kind, &ShaderHooks::none(), [4, 2], true);
            assert!(
                !src.contains("_lo : vec3<f32>,"),
                "{kind:?} kept a low part"
            );
            crate::renderer::validate_custom_shader(&src, 0)
                .unwrap_or_else(|e| panic!("{kind:?} local: {e}"));
        }
    }

    #[test]
    fn reduce_compiles() {
        crate::renderer::validate_custom_shader(&reduce_source(), 0).unwrap();
    }

    #[test]
    fn composite_compiles() {
        crate::renderer::validate_custom_shader(&composite_source(), 0).unwrap();
    }
}
