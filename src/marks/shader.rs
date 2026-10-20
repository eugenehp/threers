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
};

@group(0) @binding(0) var<uniform> frame : MarkFrame;
@group(1) @binding(0) var<uniform> layer : MarkLayer;

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
    var s = size * layer.style.x;
    if (layer.style.w > 0.5) {
        s = s * frame.misc.x / max(w, 1e-6);
    }
    return clamp(s, layer.style.y, layer.style.z);
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
    return vec4<f32>(clip.xy, clip.z - toward * layer.depth.y * clip.w, clip.w);
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

/// Outside the clip rectangle or beyond the horizon.
fn mark_culled(p : vec3<f32>) -> bool {
    return mark_clipped(p) || mark_beyond_horizon(p);
}

/// `c`, mixed towards the highlight colour when `index` is the highlighted
/// row (`ids.y` is that row plus one, 0 for none).
fn mark_highlight(c : vec4<f32>, index : u32) -> vec4<f32> {
    if (layer.ids.y != 0u && index + 1u == layer.ids.y) {
        return vec4<f32>(mix(c.rgb, layer.highlight.rgb, layer.highlight.a), c.a);
    }
    return c;
}

/// Offset a clip position by `px` screen pixels (y up).
fn mark_offset_px(clip : vec4<f32>, px : vec2<f32>) -> vec4<f32> {
    return clip + vec4<f32>(px * 2.0 * frame.viewport.zw * clip.w, 0.0, 0.0);
}
"#;

const POINT: &str = r#"
struct PointIn {
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32,
    @location(0) pos_hi : vec3<f32>,
    @location(1) pos_lo : vec3<f32>,
    @location(2) radius : f32,
    @location(3) color  : vec4<f32>,
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
};

@vertex
fn vs_main(in : PointIn) -> PointOut {
    var i : MarkInstance;
    i.position = mark_relative(in.pos_hi, in.pos_lo);
    i.position2 = i.position;
    i.size = in.radius;
    i.color = mark_decode(in.color);
    i.color2 = i.color;
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
    let extent = r + layer.style2.x * 0.5 + frame.aa.x;
    let corner = mark_corner_signed(mark_quad_corner(in.vid));
    out.clip = mark_order(mark_lift_depth(mark_offset_px(clip, corner * extent), i.position, r));
    out.local = corner * extent;
    out.radius = r;
    out.color = i.color;
    out.custom0 = i.custom0;
    out.custom1 = i.custom1;
    out.index = i.index;
    return out;
}

/// (fill coverage, outer coverage) at this fragment.
fn point_coverage(in : PointOut) -> vec2<f32> {
    let d = length(in.local);
    let half_stroke = layer.style2.x * 0.5;
    let outer = clamp(in.radius + half_stroke - d + 0.5, 0.0, 1.0);
    var fill = clamp(in.radius - half_stroke - d + 0.5, 0.0, 1.0);
    if (half_stroke <= 0.0) {
        fill = outer;
    }
    return vec2<f32>(fill, outer);
}

@fragment
fn fs_main(in : PointOut) -> @location(0) vec4<f32> {
    let cov = point_coverage(in);
    if (cov.y <= 0.0) {
        discard;
    }
    let fill = hook_color(in.color, in.custom0, in.custom1);
    // style2.y: 1 = no fill, stroke only.
    let fill_a = select(fill.a, 0.0, layer.style2.y > 0.5);
    let stroke = layer.stroke;
    let fa = fill_a * cov.x;
    let sa = stroke.a * (cov.y - cov.x);
    let a = fa + sa;
    if (a <= 0.0) {
        discard;
    }
    let rgb = (fill.rgb * fa + stroke.rgb * sa) / a;
    return mark_out(mark_highlight(vec4<f32>(rgb, a * layer.style2.w), in.index));
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
    @location(4) width : f32,
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
    // Distance along the path to this segment's source, and the segment's
    // length, in world units — for dashes. Zero on arcs.
    @location(9) @interpolate(flat) along : vec2<f32>,
};

/// A point on the arc from `a` to `b`, lifted by `style2.x` times the chord
/// at its peak (a parabola; height 0 is the straight segment). On the flat
/// ground the lift is along `style3.xyz`; on a globe (`style3.w` set,
/// `style3.xyz` its centre) the arc follows the great circle and lifts away
/// from the centre. Chosen with `select`, not a branch.
fn seg_arc(a : vec3<f32>, b : vec3<f32>, t : f32) -> vec3<f32> {
    let h = layer.style2.x * 4.0 * t * (1.0 - t) * distance(a, b);
    let flat = mix(a, b, t) + layer.style3.xyz * h;
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
    let c = frame.view_proj * vec4<f32>(p + offset, 1.0);
    if (c.w <= 0.0) {
        return vec3<f32>(normal, 0.0);
    }
    let d = (seg_px(c) - s) * away;
    let l = length(d);
    if (l < 1e-6) {
        return vec3<f32>(normal, 0.0);
    }
    let n2 = vec2<f32>(-d.y, d.x) / l;
    let m = normal + n2;
    let ml = length(m);
    // cos of half the turn: 0.5 is a 120° turn, a mitre twice the width.
    let cosine = ml * 0.5;
    if (cosine < 0.5) {
        return vec3<f32>(normal, 0.0);
    }
    return vec3<f32>(m / ml / cosine, 1.0);
}

@vertex
fn vs_main(in : SegIn) -> SegOut {
    var i : MarkInstance;
    i.position = mark_relative(in.a_hi, in.a_lo);
    i.position2 = mark_relative(in.b_hi, in.b_lo);
    i.size = in.width;
    i.color = mark_decode(in.color);
    i.color2 = mark_decode(in.color2);
    i.custom0 = /*CUSTOM0*/;
    i.custom1 = /*CUSTOM1*/;
    i.visible = 1.0;
    i.index = in.iid;
    i = hook_instance(i);

    var out : SegOut;
    let n = max(u32(layer.style2.y), 1u);
    let piece = in.vid / 6u;
    let t0 = f32(piece) / f32(n);
    let t1 = f32(piece + 1u) / f32(n);
    let p0 = seg_arc(i.position, i.position2, t0);
    let p1 = seg_arc(i.position, i.position2, t1);
    let c0 = frame.view_proj * vec4<f32>(p0, 1.0);
    let c1 = frame.view_proj * vec4<f32>(p1, 1.0);
    if (i.visible <= 0.0 || piece >= n || c0.w <= 0.0 || c1.w <= 0.0) {
        out.clip = mark_hidden();
        return out;
    }
    let s0 = seg_px(c0);
    let s1 = seg_px(c1);
    var dir = s1 - s0;
    let len = length(dir);
    if (len > 1e-6) {
        dir = dir / len;
    } else {
        dir = vec2<f32>(1.0, 0.0);
    }
    let normal = vec2<f32>(-dir.y, dir.x);
    let half = mark_size_px(i.size, 0.5 * (c0.w + c1.w)) * 0.5;
    // Caps only where the line really ends — extra.w: 0 butt, 1 round,
    // 3 square; the joints inside an arc butt up against each other so a
    // translucent arc is not darker there.
    let end_cap = layer.extra.w;
    var cap0 = select(0.0, end_cap, piece == 0u);
    var cap1 = select(0.0, end_cap, piece + 1u == n);
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
    let has0 = dot(in.prev.xyz, in.prev.xyz) > 0.0;
    let has1 = dot(in.next.xyz, in.next.xyz) > 0.0;
    // Straight, the stored offset itself: exact, where the round trip
    // through p0 would depend on p0's last bits.
    let off0 = select(seg_arc(p0 + in.prev.xyz, p0, 1.0 - step) - p0, in.prev.xyz, n == 1u);
    let off1 = select(seg_arc(p1, p1 + in.next.xyz, step) - p1, in.next.xyz, n == 1u);
    let mm0 = seg_miter(p0, s0, select(vec3<f32>(0.0), off0, has0), normal, -1.0);
    let mm1 = seg_miter(p1, s1, select(vec3<f32>(0.0), off1, has1), normal, 1.0);
    let first = piece == 0u;
    let last = piece + 1u == n;
    let m0 = select(vec3<f32>(normal, 0.0), mm0, first);
    let m1 = select(vec3<f32>(normal, 0.0), mm1, last);
    cap0 = select(cap0, 2.0, first && mm0.z > 0.5);
    cap1 = select(cap1, 2.0, last && mm1.z > 0.5);
    let pad = half + 1.0;
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
    out.clip = mark_order(mark_lift_depth(mark_offset_px(base, off), select(p0, p1, far), half));
    out.world = select(p0, p1, far);
    out.uv = vec2<f32>(select(0.0, len, far) + dot(off, dir), dot(off, normal));
    out.dims = vec4<f32>(len, half, cap0, cap1);
    out.t_span = vec2<f32>(t0, t1);
    out.along = select(vec2<f32>(0.0), vec2<f32>(in.prev.w, in.next.w), n == 1u);
    out.color = i.color;
    out.color2 = i.color2;
    out.custom0 = i.custom0;
    out.custom1 = i.custom1;
    out.index = i.index;
    return out;
}

/// Distance in pixels from the line's centre, ends included: a butt end
/// (0) cuts, a round one (1) is a half disc, a mitred one (2) is cut by the
/// quad's own edge on the bisector, a square one (3) a half square.
fn seg_end_dist(u : f32, v : f32, cap : f32) -> f32 {
    if (cap > 2.5) {
        return max(abs(v), u);
    }
    return select(1e9, length(vec2<f32>(u, v)), cap > 0.5);
}

fn seg_dist(in : SegOut) -> f32 {
    let u = in.uv.x;
    let v = in.uv.y;
    let len = in.dims.x;
    var dist = abs(v);
    if (u < 0.0 && (in.dims.z < 1.5 || in.dims.z > 2.5)) {
        dist = seg_end_dist(-u, v, in.dims.z);
    } else if (u > len && (in.dims.w < 1.5 || in.dims.w > 2.5)) {
        dist = seg_end_dist(u - len, v, in.dims.w);
    }
    return dist;
}

/// Coverage of the dash pattern at this fragment: `extra.xy` the dash and
/// gap in line widths, measured along the path.
fn seg_dash(in : SegOut) -> f32 {
    let width = in.dims.y * 2.0;
    let dash = layer.extra.x * width;
    let period = dash + layer.extra.y * width;
    if (layer.extra.x <= 0.0 || period <= 0.0 || in.along.y <= 0.0) {
        return 1.0;
    }
    // Pixels along the whole path: the segment's own pixels, plus the world
    // distance before it at this segment's pixels per world unit.
    let along = in.along.x * in.dims.x / in.along.y + in.uv.x;
    let phase = along - floor(along / period) * period;
    var d = min(phase, dash - phase);
    if (phase > dash) {
        d = -min(phase - dash, period - phase);
    }
    return clamp(d + 0.5, 0.0, 1.0);
}

fn seg_coverage(in : SegOut) -> f32 {
    return clamp(in.dims.y - seg_dist(in) + 0.5, 0.0, 1.0) * seg_dash(in);
}

@fragment
fn fs_main(in : SegOut) -> @location(0) vec4<f32> {
    if (mark_culled(in.world)) {
        discard;
    }
    let cov = seg_coverage(in);
    if (cov <= 0.0) {
        discard;
    }
    let t = mix(in.t_span.x, in.t_span.y, clamp(in.uv.x / max(in.dims.x, 1e-6), 0.0, 1.0));
    var c = hook_color(mix(in.color, in.color2, t), in.custom0, in.custom1);
    // The casing: `extra.z` pixels of `stroke` colour inside the edge.
    if (layer.extra.z > 0.0) {
        let inner = clamp(in.dims.y - layer.extra.z - seg_dist(in) + 0.5, 0.0, 1.0);
        c = mix(layer.stroke, c, inner);
    }
    return mark_out(mark_highlight(vec4<f32>(c.rgb, c.a * cov * layer.style2.w), in.index));
}

@fragment
fn fs_pick(in : SegOut) -> @location(0) vec2<u32> {
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
    @location(2) pos_hi : vec3<f32>,
    @location(3) pos_lo : vec3<f32>,
    @location(4) scale : vec3<f32>,
    @location(5) color : vec4<f32>,
    /*CUSTOM_IN*/
};

struct ShapeOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) color : vec4<f32>,
    @location(1) custom0 : vec4<f32>,
    @location(2) custom1 : vec4<f32>,
    @location(3) @interpolate(flat) index : u32,
};

@vertex
fn vs_main(in : ShapeIn) -> ShapeOut {
    var i : MarkInstance;
    i.position = mark_relative(in.pos_hi, in.pos_lo);
    i.position2 = i.position;
    i.size = 1.0;
    i.color = mark_decode(in.color);
    i.color2 = i.color;
    i.custom0 = /*CUSTOM0*/;
    i.custom1 = /*CUSTOM1*/;
    i.visible = 1.0;
    i.index = in.iid;
    i = hook_instance(i);

    var out : ShapeOut;
    if (i.visible <= 0.0 || mark_culled(i.position)) {
        out.clip = mark_hidden();
        return out;
    }
    let scale = in.scale * i.size * layer.style.x;
    let world = i.position + in.vertex * scale;
    out.clip = mark_order(frame.view_proj * vec4<f32>(world, 1.0));
    // A normal transforms by the inverse scale; zero scale leaves it as is.
    let safe = select(scale, vec3<f32>(1.0), abs(scale) < vec3<f32>(1e-12));
    let raw = in.normal / safe;
    // Geometry without normals is drawn unlit.
    var lit = 1.0;
    if (dot(raw, raw) > 1e-12) {
        let ambient = layer.style2.x;
        lit = ambient + (1.0 - ambient) * max(dot(normalize(raw), frame.light.xyz), 0.0);
    }
    out.color = vec4<f32>(i.color.rgb * lit, i.color.a);
    out.custom0 = i.custom0;
    out.custom1 = i.custom1;
    out.index = i.index;
    return out;
}

@fragment
fn fs_main(in : ShapeOut) -> @location(0) vec4<f32> {
    let c = hook_color(in.color, in.custom0, in.custom1);
    return mark_out(mark_highlight(vec4<f32>(c.rgb, c.a * layer.style2.w), in.index));
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
    /*CUSTOM_IN*/
};

struct PolyOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) color : vec4<f32>,
    @location(1) custom0 : vec4<f32>,
    @location(2) custom1 : vec4<f32>,
    @location(3) @interpolate(flat) index : u32,
    @location(4) world : vec3<f32>,
};

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
    // Lit relative to a roof, so a roof shows its colour exactly and a
    // choropleth's legend reads true; walls darken as they turn away.
    let l = frame.light.xyz;
    let facing = clamp(dot(in.normal_top.xyz, l) / max(l.y, 0.05), 0.0, 1.0);
    let ambient = layer.style2.x;
    out.color = vec4<f32>(i.color.rgb * (ambient + (1.0 - ambient) * facing), i.color.a);
    out.custom0 = i.custom0;
    out.custom1 = i.custom1;
    out.index = i.index;
    return out;
}

@fragment
fn fs_main(in : PolyOut) -> @location(0) vec4<f32> {
    if (mark_culled(in.world)) {
        discard;
    }
    let c = hook_color(in.color, in.custom0, in.custom1);
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
    @location(2) size : f32,
    // x, y, width, height in atlas pixels.
    @location(3) rect : vec4<f32>,
    @location(4) angle : f32,
    @location(5) color : vec4<f32>,
    // 0 when collision placement hid the icon.
    @location(6) shown : f32,
    /*CUSTOM_IN*/
};

struct SpriteOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) uv : vec2<f32>,
    @location(1) color : vec4<f32>,
    @location(2) custom0 : vec4<f32>,
    @location(3) custom1 : vec4<f32>,
    @location(4) @interpolate(flat) index : u32,
};

@vertex
fn vs_main(in : SpriteIn) -> SpriteOut {
    var i : MarkInstance;
    i.position = mark_relative(in.pos_hi, in.pos_lo);
    i.position2 = i.position;
    i.size = in.size;
    i.color = mark_decode(in.color);
    i.color2 = i.color;
    i.custom0 = /*CUSTOM0*/;
    i.custom1 = /*CUSTOM1*/;
    i.visible = in.shown;
    i.index = in.iid;
    i = hook_instance(i);

    var out : SpriteOut;
    let clip = frame.view_proj * vec4<f32>(i.position, 1.0);
    if (i.visible <= 0.0 || clip.w <= 0.0 || in.rect.w <= 0.0 || mark_culled(i.position)) {
        out.clip = mark_hidden();
        return out;
    }
    let h = mark_size_px(i.size, clip.w);
    let half = vec2<f32>(h * in.rect.z / in.rect.w, h) * 0.5;
    let c = mark_corner_signed(mark_quad_corner(in.vid));
    // style3.zw: the anchor, the image point that sits on the position.
    let local = (c - layer.style3.zw) * half;
    let s = sin(in.angle);
    let k = cos(in.angle);
    let turned = vec2<f32>(k * local.x - s * local.y, s * local.x + k * local.y);
    out.clip = mark_order(mark_lift_depth(mark_offset_px(clip, turned), i.position, h * 0.5));
    // Image y runs down; the quad's runs up. style3.xy: the atlas size.
    let t = vec2<f32>(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
    out.uv = (in.rect.xy + t * in.rect.zw) / max(layer.style3.xy, vec2<f32>(1.0));
    out.color = i.color;
    out.custom0 = i.custom0;
    out.custom1 = i.custom1;
    out.index = i.index;
    return out;
}

/// The sprite's colour before the hook: style2.x picks colour, mask or
/// distance field.
fn sprite_color(in : SpriteOut) -> vec4<f32> {
    let t = textureSample(atlas_tex, atlas_samp, in.uv);
    if (layer.style2.x < 0.5) {
        return t * in.color;
    }
    if (layer.style2.x < 1.5) {
        return vec4<f32>(in.color.rgb, in.color.a * t.a);
    }
    let w = max(fwidth(t.a), 1e-4) * 0.7;
    return vec4<f32>(in.color.rgb, in.color.a * smoothstep(0.5 - w, 0.5 + w, t.a));
}

@fragment
fn fs_main(in : SpriteOut) -> @location(0) vec4<f32> {
    let c = hook_color(sprite_color(in), in.custom0, in.custom1);
    if (c.a <= 0.002) {
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
    out.halo_scale = max(in.state.w, 1e-3);
    return out;
}

/// (fill, outer) coverage: the fill from `fill_d` — the multi-channel
/// median, or the distance — and the halo from the true distance `true_d`.
/// `halo` is the halo's width in distance-field units.
fn glyph_coverage(fill_d : f32, true_d : f32, halo : f32, aa : f32) -> vec2<f32> {
    let fill = smoothstep(0.5 - aa, 0.5 + aa, fill_d);
    let edge = 0.5 - halo;
    let outer = max(smoothstep(edge - aa, edge + aa, true_d), fill);
    return vec2<f32>(fill, outer);
}

@fragment
fn fs_main(in : GlyphOut) -> @location(0) vec4<f32> {
    let t = textureSample(glyph_atlas, glyph_sampler, in.uv);
    // style2.z: 1 when rgb is a multi-channel field; its median is the
    // distance, with corners kept. `a` is always the plain distance.
    let m = max(min(t.r, t.g), min(max(t.r, t.g), t.b));
    let d = select(t.a, m, layer.style2.z > 0.5);
    let aa = max(fwidth(d) * 0.75, 1e-3);
    let cov = glyph_coverage(d, t.a, min(layer.style2.x / in.halo_scale, 0.5), aa);
    let text = hook_color(in.color, vec4<f32>(0.0), vec4<f32>(0.0));
    let halo = layer.stroke;
    let ta = text.a * cov.x;
    let ha = halo.a * (cov.y - cov.x);
    let a = ta + ha;
    if (a <= 0.002) {
        discard;
    }
    let rgb = (text.rgb * ta + halo.rgb * ha) / a;
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
    return vec4<f32>(in.weight * g, 0.0, 0.0, 0.0);
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
    let uv = in.clip.xy * frame.viewport.zw;
    let v = textureSampleLevel(density_field, density_sampler, uv, 0.0).r;
    // style3: (weight that maps to the top of the ramp, fade-in threshold
    // as a fraction of the ramp, 1 to scale to the field's own maximum, the
    // fraction of it that does). `style` is the kernel's size.
    let measured = textureSampleLevel(density_max, density_sampler, vec2<f32>(0.5), 0.0).r;
    let top = select(layer.style3.x, measured * layer.style3.w, layer.style3.z > 0.5);
    let t = clamp(v / max(top, 1e-12), 0.0, 1.0);
    let c = textureSampleLevel(density_ramp, density_sampler, vec2<f32>(t, 0.5), 0.0);
    let a = smoothstep(0.0, max(layer.style3.y, 1e-6), t) * c.a;
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
        MarkKind::Point => (POINT, 4),
        MarkKind::Segment => (SEGMENT, 9),
        MarkKind::Shape => (SHAPE, 6),
        MarkKind::Glyph => (GLYPH, 6),
        MarkKind::Splat => (SPLAT, 3),
        MarkKind::Polygon => (POLYGON, 6),
        MarkKind::Sprite => (SPRITE, 7),
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
            m = max(m, textureSampleLevel(reduce_src, reduce_samp, uv, 0.0).r);
        }
    }
    return vec4<f32>(m, 0.0, 0.0, 1.0);
}
"#;

/// The density reduction module.
pub(crate) fn reduce_source() -> String {
    format!("{}\n{DENSITY_REDUCE}", crate::renderer::fullscreen::VS)
}

/// The density composite module.
pub(crate) fn composite_source() -> String {
    format!(
        "{PREAMBLE}\n{}\n{}\n{DENSITY_COMPOSITE}",
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
