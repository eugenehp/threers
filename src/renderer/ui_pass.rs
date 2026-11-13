//! The retained UI drawn on the GPU: a [`Scene`] in one instanced draw.
//!
//! [`OverlayPainter`] rasterises a UI on the CPU into an overlay that is then
//! uploaded, which is most of a frame's cost at a high resolution. This draws
//! the same scene on the GPU: every rectangle, rounded
//! rectangle, border and gradient band is one instance of one quad, shaded by
//! a signed distance field with the overlay's own coverage rule — `0.5 − d`
//! over the pixel's footprint across the edge — so edges come out as the CPU
//! draws them. Text and paths are rasterised once each, by the overlay's own
//! code, into a coverage atlas kept from frame to frame, and drawn as
//! textured quads tinted by their colour. Everything is in one draw call, in
//! the scene's order, so what overlaps paints as it would on the CPU.
//!
//! Blending happens in linear light, as the overlay's does, when the target
//! is an sRGB format. On a non-sRGB target the shader encodes the colour and
//! the blend mixes encoded values, as a browser's canvas does.
//!
//! A box with a custom shader on it (see [`crate::ui::shader`]) is drawn
//! into a layer of its own, in its own space, then put in its place through
//! the shader: a render pass per layer, before the pass it shows in.
//!
//! Two differences from [`OverlayPainter`], both towards what a browser does:
//! clipping is per pixel (the overlay drops whole characters at a clip's
//! edge, and draws a cut rounded rectangle square), and a rounded clip keeps
//! its corners.
//!
//! ```no_run
//! # fn demo(device: &wgpu::Device, queue: &wgpu::Queue, view: &wgpu::TextureView) {
//! use threers::renderer::UiPass;
//! use threers::ui::Ui;
//!
//! let mut pass = UiPass::new(device);
//! let mut ui = Ui::new();
//! // ... build the tree ...
//! // Lay out with the pass's own measurer, so the text fits what it draws.
//! ui.layout(1280.0, 720.0, &mut pass.text_measure(2.0));
//! pass.draw(device, queue, &ui.scene(), 2.0, view, (2560, 1440), wgpu::TextureFormat::Rgba8UnormSrgb);
//! # }
//! ```

use std::borrow::Cow;
use std::collections::HashMap;

use crate::overlay::Overlay;
use crate::ui::scene::Scene;
use crate::ui::shader::{Effect, ParamKind, Shader};
use crate::ui::{OverlayPainter, Painter, Rect, Rgba, Style, TextMeasure, TextStyle};

const SHADER: &str = crate::wgsl!(
    r#"
struct Globals {
    viewport: vec2<f32>,
    // 1 when the target stores encoded values and blends them as they are.
    encode: f32,
    unused: f32,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var images: texture_2d<f32>;
@group(0) @binding(3) var images_sampler: sampler;
// Path edges, one segment (ax, ay, bx, by) a texel, row-major.
@group(0) @binding(4) var edges: texture_2d<f32>;
// A surface: drawn by `fs_surface`, one bind group per texture.
@group(1) @binding(0) var surface_tex: texture_2d<f32>;
@group(1) @binding(1) var surface_sampler: sampler;

struct VsIn {
    @location(0) rect: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) clip: vec4<f32>,
    // radius, clip radius, kind (0 shape, 1 sprite, 2 image, 3 shadow,
    // 4 wavy line, 5 arc, 6 path fill, 7 path stroke, 8 gradient stretch,
    // 10 gradient of another shape), clipped
    @location(3) params: vec4<f32>,
    // a sprite's texel origin in the atlas; an image's, and its size
    @location(4) sprite: vec4<f32>,
    // A projective map of the shape, by rows (a box turned in 3D): the
    // identity for most.
    @location(5) xform0: vec4<f32>,
    @location(6) xform1: vec4<f32>,
    @location(7) xform2: vec4<f32>,
    // A second clip's box (no width: none); its radius is xform0.w.
    @location(8) clip2: vec4<f32>,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) rect: vec4<f32>,
    @location(1) @interpolate(flat) color: vec4<f32>,
    @location(2) @interpolate(flat) clip: vec4<f32>,
    @location(3) @interpolate(flat) params: vec4<f32>,
    @location(4) @interpolate(flat) sprite: vec4<f32>,
    // Where the fragment is on the shape, before its projective map:
    // shapes are shaded here; clips are tested at `position`.
    @location(5) local: vec2<f32>,
    @location(6) @interpolate(flat) clip2: vec4<f32>,
    @location(7) @interpolate(flat) clip2_radius: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, inst: VsIn) -> VsOut {
    // Two triangles: corners 0 1 2, 2 1 3 of the quad, by bits rather than
    // a table (a table indexed at run time goes to stack memory on Metal).
    let corner = vec2<f32>(
        select(0.0, 1.0, vertex == 1u || vertex == 4u || vertex == 5u),
        select(0.0, 1.0, vertex == 2u || vertex == 3u || vertex == 5u),
    );
    // A shape's edge pixels are partly covered a pixel outside its box; a
    // shadow reaches three deviations (`sprite.x`) past it.
    var grow = select(1.0, 0.0, inst.params.z > 0.5 && inst.params.z < 1.5);
    if (inst.params.z > 2.5 && inst.params.z < 3.5) {
        grow = 3.0 * inst.sprite.x + 1.0;
    }
    let p = inst.rect.xy - vec2<f32>(grow) + corner * (inst.rect.zw + vec2<f32>(2.0 * grow));
    // Projected, with its weight as w: the GPU interpolates `local`
    // perspective-correctly across the turned shape.
    let h = vec3<f32>(p, 1.0);
    let q = vec3<f32>(dot(inst.xform0.xyz, h), dot(inst.xform1.xyz, h), dot(inst.xform2.xyz, h));
    let w = max(q.z, 1e-4);
    var out: VsOut;
    out.position = vec4<f32>(
        q.x / globals.viewport.x * 2.0 - w,
        w - q.y / globals.viewport.y * 2.0,
        0.0,
        w,
    );
    out.local = p;
    out.rect = inst.rect;
    out.color = inst.color;
    out.clip = inst.clip;
    out.params = inst.params;
    out.sprite = inst.sprite;
    out.clip2 = inst.clip2;
    out.clip2_radius = inst.xform0.w;
    return out;
}

// What of a fragment at `screen` its clips leave: the clip, and the one
// round it where that one's corners cut too.
fn clip_cover(screen: vec2<f32>, in: VsOut) -> f32 {
    var c = coverage(screen, in.clip, in.params.y);
    if (in.clip2.z > 0.0) {
        c = c * coverage(screen, in.clip2, in.clip2_radius);
    }
    return c;
}

fn round_box(p: vec2<f32>, half: vec2<f32>, r: f32) -> f32 {
    let rr = min(max(r, 0.0), max(min(half.x, half.y), 0.0));
    let q = abs(p) - (half - vec2<f32>(rr));
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - rr;
}

fn field(p: vec2<f32>, rect: vec4<f32>, r: f32) -> f32 {
    let half = abs(rect.zw * 0.5);
    return round_box(p - (rect.xy + rect.zw * 0.5), half, r);
}

// The overlay's coverage: `0.5 - d` across a ramp as wide as the pixel is
// across the edge, from forward differences of the field.
fn coverage(p: vec2<f32>, rect: vec4<f32>, r: f32) -> f32 {
    let band = 0.70710677;
    let d = field(p, rect, r);
    if (d >= band) {
        return 0.0;
    }
    if (d <= -band) {
        return 1.0;
    }
    let gx = (field(p + vec2<f32>(0.5, 0.0), rect, r) - d) / 0.5;
    let gy = (field(p + vec2<f32>(0.0, 0.5), rect, r) - d) / 0.5;
    let l = length(vec2<f32>(gx, gy));
    let w = select(1.0, (abs(gx) + abs(gy)) / l, l > 1e-12);
    return clamp(0.5 - d / max(w, 1e-6), 0.0, 1.0);
}

// Wallace's error function, as the overlay's `erf_approx`.
fn erf_approx(x: f32) -> f32 {
    let s = sign(x);
    let a = abs(x);
    var t = 1.0 + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
    t = t * t;
    return s - s / (t * t);
}

// A blurred rounded box, as the overlay's `shadow_coverage`.
fn shadow_coverage(rect: vec4<f32>, point: vec2<f32>, sigma: f32, radius: f32) -> f32 {
    let half = rect.zw * 0.5;
    let p = point - (rect.xy + half);
    let corner = clamp(radius, 0.0, min(half.x, half.y));
    let low = p.y - half.y;
    let high = p.y + half.y;
    let start = clamp(-3.0 * sigma, low, high);
    let end = clamp(3.0 * sigma, low, high);
    let step = (end - start) / 4.0;
    var v = start + step * 0.5;
    var sum = 0.0;
    let k = 0.70710678 / sigma;
    for (var i = 0; i < 4; i = i + 1) {
        let y = p.y - v;
        let delta = min(half.y - corner - abs(y), 0.0);
        let curved = half.x - corner + sqrt(max(corner * corner - delta * delta, 0.0));
        let row = (0.5 + 0.5 * erf_approx((p.x + curved) * k)) - (0.5 + 0.5 * erf_approx((p.x - curved) * k));
        let gaussian = exp(-(v * v) / (2.0 * sigma * sigma)) / (2.5066283 * sigma);
        sum = sum + row * gaussian * step;
        v = v + step;
    }
    return clamp(sum, 0.0, 1.0);
}

// Distance to the wave `y + a sin(k (x - x0))` between `x0` and `x1`, as
// the overlay's `wavy_line` measures it.
fn wave_centre(p: vec2<f32>, x0: f32, x1: f32, y: f32, a: f32, k: f32) -> f32 {
    let x = clamp(p.x, x0, x1);
    let t = k * (x - x0);
    let dy = p.y - (y + a * sin(t));
    let slope = a * k * cos(t);
    return abs(dy) / sqrt(1.0 + slope * slope) + abs(p.x - x);
}

// The overlay's stroke coverage: the box-filtered share of the pixel the
// stroke covers, its footprint from the larger of forward and backward
// differences (a stroke's field is folded at its centre line).
fn wave_coverage(p: vec2<f32>, x0: f32, x1: f32, y: f32, a: f32, k: f32, half: f32) -> f32 {
    let band = 0.70710677;
    let d = wave_centre(p, x0, x1, y, a, k) - half;
    if (d >= band) {
        return 0.0;
    }
    if (d <= -band) {
        return 1.0;
    }
    let e = 0.5;
    let fx = (wave_centre(p + vec2<f32>(e, 0.0), x0, x1, y, a, k) - half - d) / e;
    let bx = (d - (wave_centre(p - vec2<f32>(e, 0.0), x0, x1, y, a, k) - half)) / e;
    let fy = (wave_centre(p + vec2<f32>(0.0, e), x0, x1, y, a, k) - half - d) / e;
    let by = (d - (wave_centre(p - vec2<f32>(0.0, e), x0, x1, y, a, k) - half)) / e;
    let gx = max(abs(fx), abs(bx));
    let gy = max(abs(fy), abs(by));
    let l = length(vec2<f32>(gx, gy));
    let w = max(select(1.0, (gx + gy) / l, l > 1e-12), 1e-6);
    let u = abs(d + half);
    let lo = max(u - 0.5 * w, -half);
    let hi = min(u + 0.5 * w, half);
    return clamp((hi - lo) / w, 0.0, 1.0);
}

// A surface fills its rect with the whole texture, edges anti-aliased as
// a rectangle's, under the clip.
@fragment
fn fs_surface(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.local;
    let screen = in.position.xy;
    let uv = clamp((p - in.rect.xy) / in.rect.zw, vec2<f32>(0.0), vec2<f32>(1.0));
    let texel = textureSampleLevel(surface_tex, surface_sampler, uv, 0.0);
    var cover = coverage(p, in.rect, 0.0);
    if (in.params.w > 0.5) {
        cover = cover * clip_cover(screen, in);
    }
    var rgb = texel.rgb;
    if (globals.encode > 0.5) {
        rgb = encode_srgb(rgb);
    }
    return vec4<f32>(rgb, in.color.a * texel.a * cover);
}

// A material's backdrop: what was under the box, blurred (bound as the
// surface texture, `sprite` the region it was copied from in this target's
// pixels, `color.g` how much smaller it was blurred at), its colour scaled
// by `color.r`, in the box's shape. Its texels are already in the target's
// own encoding.
@fragment
fn fs_backdrop(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.local;
    let screen = in.position.xy;
    let dims = vec2<f32>(textureDimensions(surface_tex));
    let uv = (screen - in.sprite.xy) / max(in.color.g, 1.0) / dims;
    let texel = textureSampleLevel(surface_tex, surface_sampler, uv, 0.0);
    let grey = vec3<f32>(dot(texel.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)));
    let rgb = clamp(mix(grey, texel.rgb, in.color.r), vec3<f32>(0.0), vec3<f32>(1.0));
    var cover = coverage(p, in.rect, in.params.x);
    if (in.params.w > 0.5) {
        cover = cover * clip_cover(screen, in);
    }
    return vec4<f32>(rgb, in.color.a * cover);
}

fn edge(i: u32) -> vec4<f32> {
    let w = textureDimensions(edges).x;
    return textureLoad(edges, vec2<i32>(i32(i % w), i32(i / w)), 0);
}

fn segment_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let v = b - a;
    let q = p - a;
    let len2 = dot(v, v);
    var t = 0.0;
    if (len2 > 0.0) {
        t = clamp(dot(q, v) / len2, 0.0, 1.0);
    }
    return length(q - v * t);
}

// A filled polygon's signed distance, as the overlay's `d2::polygon`:
// the nearest edge, negative inside by the crossing count.
fn fill_field(p: vec2<f32>, start: u32, count: u32) -> f32 {
    var best = 3.0e38;
    var inside = false;
    for (var i = 0u; i < count; i = i + 1u) {
        let e = edge(start + i);
        let a = e.xy;
        let b = e.zw;
        best = min(best, segment_distance(p, a, b));
        if ((p.y >= a.y) != (p.y >= b.y)) {
            let t = (p.y - a.y) / (b.y - a.y);
            if (a.x + t * (b.x - a.x) > p.x) {
                inside = !inside;
            }
        }
    }
    return select(best, -best, inside);
}

fn fill_coverage(p: vec2<f32>, start: u32, count: u32) -> f32 {
    let band = 0.70710677;
    let d = fill_field(p, start, count);
    if (d >= band) {
        return 0.0;
    }
    if (d <= -band) {
        return 1.0;
    }
    let gx = (fill_field(p + vec2<f32>(0.5, 0.0), start, count) - d) / 0.5;
    let gy = (fill_field(p + vec2<f32>(0.0, 0.5), start, count) - d) / 0.5;
    let l = length(vec2<f32>(gx, gy));
    let w = select(1.0, (abs(gx) + abs(gy)) / l, l > 1e-12);
    return clamp(0.5 - d / max(w, 1e-6), 0.0, 1.0);
}

// A circular arc's distance, stroked `half` each side with round ends:
// centre `c`, radius `r`, from `start` round `sweep` (radians, clockwise
// from straight up; a whole turn or more is the ring). As
// `ArcShape::distance`.
fn arc_field(p: vec2<f32>, c: vec2<f32>, r: f32, half: f32, start: f32, sweep: f32) -> f32 {
    let q = p - c;
    if (abs(sweep) >= 6.2830853) {
        return abs(length(q) - r) - half;
    }
    let s0 = select(start, start + sweep, sweep < 0.0);
    let s = abs(sweep);
    let mid = s0 + s * 0.5;
    let m = vec2<f32>(sin(mid), cos(mid));
    let along = q.x * m.x - q.y * m.y;
    let across = abs(q.x * m.y + q.y * m.x);
    let h = vec2<f32>(sin(s * 0.5), cos(s * 0.5));
    if (h.y * across > h.x * along) {
        return length(vec2<f32>(across, along) - h * r) - half;
    }
    return abs(length(vec2<f32>(across, along)) - r) - half;
}

// A polyline's distance, as the overlay's `polyline`, less half its width.
fn stroke_field(p: vec2<f32>, start: u32, count: u32, half: f32) -> f32 {
    var best = 3.0e38;
    for (var i = 0u; i < count; i = i + 1u) {
        let e = edge(start + i);
        best = min(best, segment_distance(p, e.xy, e.zw));
    }
    return best - half;
}

fn stroke_coverage(p: vec2<f32>, start: u32, count: u32, half: f32) -> f32 {
    let band = 0.70710677;
    let d = stroke_field(p, start, count, half);
    if (d >= band) {
        return 0.0;
    }
    if (d <= -band) {
        return 1.0;
    }
    let e = 0.5;
    let fx = (stroke_field(p + vec2<f32>(e, 0.0), start, count, half) - d) / e;
    let bx = (d - stroke_field(p - vec2<f32>(e, 0.0), start, count, half)) / e;
    let fy = (stroke_field(p + vec2<f32>(0.0, e), start, count, half) - d) / e;
    let by = (d - stroke_field(p - vec2<f32>(0.0, e), start, count, half)) / e;
    let gx = max(abs(fx), abs(bx));
    let gy = max(abs(fy), abs(by));
    let l = length(vec2<f32>(gx, gy));
    let w = max(select(1.0, (gx + gy) / l, l > 1e-12), 1e-6);
    let u = abs(d + half);
    let lo = max(u - 0.5 * w, -half);
    let hi = min(u + 0.5 * w, half);
    return clamp((hi - lo) / w, 0.0, 1.0);
}

fn encode_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn decode_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((max(c, vec3<f32>(0.0)) + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.local;
    let screen = in.position.xy;
    var cover = 0.0;
    var rgb = in.color.rgb;
    var alpha = in.color.a;
    if (in.params.z < 0.5) {
        cover = coverage(p, in.rect, in.params.x);
    } else if (in.params.z < 1.5) {
        let texel = vec2<i32>(in.sprite.xy + floor(p - in.rect.xy));
        cover = textureLoad(atlas, texel, 0).r;
    } else if (in.params.z > 9.5) {
        // A gradient of another shape: `t` from where the fragment is
        // (`color.x` the shape: 0 along `color.yz` over `color.w` px, 1 out
        // from the centre to `color.y` px, 2 round it from `color.y`
        // radians), the colour from its stops' ramp in the image atlas at
        // `sprite.xy`; `sprite.zw` the centre.
        let c = in.sprite.zw;
        let d = p - c;
        var t = 0.0;
        if (in.color.x < 0.5) {
            t = dot(d, in.color.yz) / max(in.color.w, 1e-6) + 0.5;
        } else if (in.color.x < 1.5) {
            t = length(d) / max(in.color.y, 1e-6);
        } else {
            t = fract((atan2(d.x, -d.y) - in.color.y) / 6.2831853);
        }
        let texel = textureLoad(images, vec2<i32>(i32(in.sprite.x + clamp(t, 0.0, 1.0) * 255.0 + 0.5), i32(in.sprite.y)), 0);
        rgb = texel.rgb;
        alpha = texel.a;
        cover = coverage(p, in.rect, 0.0);
    } else if (in.params.z > 7.5) {
        // A stretch of a linear gradient, from `color` at its left to
        // `sprite` at its right, mixed as encoded (as CSS mixes). A pixel
        // is in the stretch its centre falls in, so stretches meet without
        // a seam; only the top and bottom are antialiased.
        let x = p.x;
        let inside = x >= in.rect.x && x < in.rect.x + in.rect.z;
        let d = max(in.rect.y - p.y, p.y - (in.rect.y + in.rect.w));
        cover = select(0.0, clamp(0.5 - d, 0.0, 1.0), inside);
        let t = clamp((x - in.rect.x) / max(in.rect.z, 1e-6), 0.0, 1.0);
        rgb = decode_srgb(mix(encode_srgb(in.color.rgb), encode_srgb(in.sprite.rgb), t));
        alpha = mix(in.color.a, in.sprite.a, t);
    } else if (in.params.z > 6.5) {
        // A path's stroke: sprite is (first edge, edges, half width).
        cover = stroke_coverage(p, u32(in.sprite.x), u32(in.sprite.y), in.sprite.z);
    } else if (in.params.z > 5.5) {
        cover = fill_coverage(p, u32(in.sprite.x), u32(in.sprite.y));
    } else if (in.params.z > 4.5) {
        // An arc: sprite is (radius, half its width, start, sweep), its
        // centre the quad's.
        let c = in.rect.xy + in.rect.zw * 0.5;
        cover = clamp(0.5 - arc_field(p, c, in.sprite.x, in.sprite.y, in.sprite.z, in.sprite.w), 0.0, 1.0);
    } else if (in.params.z > 3.5) {
        // A wavy line: sprite is (x0, centre y, amplitude, k), params.x half
        // its thickness; the quad reaches amplitude + thickness past it.
        let reach = in.sprite.z + 2.0 * in.params.x;
        let x1 = in.rect.x + in.rect.z - reach;
        cover = wave_coverage(p, in.sprite.x, x1, in.sprite.y, in.sprite.z, in.sprite.w, in.params.x);
    } else if (in.params.z > 2.5) {
        cover = shadow_coverage(in.rect, p, in.sprite.x, in.params.x);
    } else {
        // An image: bilinear within its own texels (half a texel in from
        // its edge, so neighbours in the atlas never bleed in).
        let t = (p - in.rect.xy) / in.rect.zw * in.sprite.zw;
        let inside = clamp(t, vec2<f32>(0.5), in.sprite.zw - vec2<f32>(0.5));
        let size = vec2<f32>(textureDimensions(images));
        let texel = textureSampleLevel(images, images_sampler, (in.sprite.xy + inside) / size, 0.0);
        // Straight alpha, filtered as stored; the colour is the texel's.
        rgb = texel.rgb;
        alpha = alpha * texel.a;
        cover = coverage(p, in.rect, 0.0);
    }
    if (in.params.w > 0.5) {
        cover = cover * clip_cover(screen, in);
    }
    if (globals.encode > 0.5) {
        rgb = encode_srgb(rgb);
    }
    return vec4<f32>(rgb, alpha * cover);
}
"#
);

/// What a custom shader's WGSL is put after: a vertex stage of its own for
/// the layer's quad, the clip's coverage, and `layer`. Its own names start
/// `threers_`, out of the way of the shader's.
const EFFECT_PRELUDE: &str = r#"
struct ThreersGlobals {
    viewport: vec2<f32>,
    encode: f32,
    unused: f32,
};

struct ThreersEffect {
    // The layer's top left, in the device pixels of the box's space.
    origin: vec2<f32>,
    // The box's top left and size, in UI pixels.
    box_origin: vec2<f32>,
    size: vec2<f32>,
    // How much of the layer texture holds the layer, in device pixels.
    extent: vec2<f32>,
    scale: f32,
    time: f32,
    unused: vec2<f32>,
    args: array<vec4<f32>, 8>,
};

@group(0) @binding(0) var<uniform> threers_globals: ThreersGlobals;
@group(1) @binding(0) var<uniform> threers_effect: ThreersEffect;
@group(1) @binding(1) var threers_layer: texture_2d<f32>;
@group(1) @binding(2) var threers_sampler: sampler;

struct ThreersIn {
    @location(0) rect: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) clip: vec4<f32>,
    @location(3) params: vec4<f32>,
    @location(4) sprite: vec4<f32>,
    @location(5) xform0: vec4<f32>,
    @location(6) xform1: vec4<f32>,
    @location(7) xform2: vec4<f32>,
    @location(8) clip2: vec4<f32>,
};

struct ThreersOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) clip: vec4<f32>,
    @location(1) @interpolate(flat) params: vec4<f32>,
    @location(2) local: vec2<f32>,
    @location(3) @interpolate(flat) clip2: vec4<f32>,
    @location(4) @interpolate(flat) clip2_radius: f32,
};

@vertex
fn threers_vs(@builtin(vertex_index) vertex: u32, inst: ThreersIn) -> ThreersOut {
    let corner = vec2<f32>(
        select(0.0, 1.0, vertex == 1u || vertex == 4u || vertex == 5u),
        select(0.0, 1.0, vertex == 2u || vertex == 3u || vertex == 5u),
    );
    let p = inst.rect.xy + corner * inst.rect.zw;
    let h = vec3<f32>(p, 1.0);
    let q = vec3<f32>(dot(inst.xform0.xyz, h), dot(inst.xform1.xyz, h), dot(inst.xform2.xyz, h));
    let w = max(q.z, 1e-4);
    var out: ThreersOut;
    out.position = vec4<f32>(
        q.x / threers_globals.viewport.x * 2.0 - w,
        w - q.y / threers_globals.viewport.y * 2.0,
        0.0,
        w,
    );
    out.local = p;
    out.clip = inst.clip;
    out.params = inst.params;
    out.clip2 = inst.clip2;
    out.clip2_radius = inst.xform0.w;
    return out;
}

fn threers_field(p: vec2<f32>, rect: vec4<f32>, r: f32) -> f32 {
    let half = abs(rect.zw * 0.5);
    let rr = min(max(r, 0.0), max(min(half.x, half.y), 0.0));
    let q = abs(p - (rect.xy + rect.zw * 0.5)) - (half - vec2<f32>(rr));
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - rr;
}

fn threers_coverage(p: vec2<f32>, rect: vec4<f32>, r: f32) -> f32 {
    let d = threers_field(p, rect, r);
    let gx = (threers_field(p + vec2<f32>(0.5, 0.0), rect, r) - d) / 0.5;
    let gy = (threers_field(p + vec2<f32>(0.0, 0.5), rect, r) - d) / 0.5;
    let l = length(vec2<f32>(gx, gy));
    let w = select(1.0, (abs(gx) + abs(gy)) / l, l > 1e-12);
    return clamp(0.5 - d / max(w, 1e-6), 0.0, 1.0);
}

// What the box drew at `q` (UI pixels from its top left), premultiplied;
// nothing outside the layer.
fn layer(q: vec2<f32>) -> vec4<f32> {
    let d = (q + threers_effect.box_origin) * threers_effect.scale - threers_effect.origin;
    let texel = textureSampleLevel(threers_layer, threers_sampler, d / vec2<f32>(textureDimensions(threers_layer)), 0.0);
    let inside = all(d >= vec2<f32>(0.0)) && all(d <= threers_effect.extent);
    return select(vec4<f32>(0.0), texel, inside);
}
"#;

/// A shader's module: the prelude, its WGSL, and a fragment stage calling
/// its function with what each parameter asks for.
fn effect_module(shader: &Shader) -> String {
    let args: Vec<String> = shader
        .params()
        .iter()
        .map(|p| match p.kind {
            ParamKind::Position => "p".to_string(),
            ParamKind::Color => "layer(p)".to_string(),
            ParamKind::Time => "threers_effect.time".to_string(),
            ParamKind::Size => "threers_effect.size".to_string(),
            ParamKind::Arg { components, slot } => {
                let swizzle = ["", ".x", ".xy", ".xyz", ""][components as usize];
                format!("threers_effect.args[{slot}]{swizzle}")
            }
        })
        .collect();
    format!(
        "{EFFECT_PRELUDE}\n{}\n
@fragment
fn threers_fs(in: ThreersOut) -> @location(0) vec4<f32> {{
    let p = in.local / threers_effect.scale - threers_effect.box_origin;
    let c = clamp({}({}), vec4<f32>(0.0), vec4<f32>(1.0));
    var cover = 1.0;
    if (in.params.w > 0.5) {{
        cover = threers_coverage(in.position.xy, in.clip, in.params.y);
        if (in.clip2.z > 0.0) {{
            cover = cover * threers_coverage(in.position.xy, in.clip2, in.clip2_radius);
        }}
    }}
    return c * cover;
}}
",
        shader.source(),
        shader.function(),
        args.join(", ")
    )
}

/// The layer as it is: what a box with a broken shader is drawn through.
fn passthrough() -> Shader {
    Shader::new("fn threers_passthrough(p: vec2<f32>, color: vec4<f32>) -> vec4<f32> { return color; }")
}

/// [`ThreersEffect`] in the prelude.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EffectUniform {
    origin: [f32; 2],
    box_origin: [f32; 2],
    size: [f32; 2],
    extent: [f32; 2],
    scale: f32,
    time: f32,
    unused: [f32; 2],
    args: [[f32; 4]; 8],
}

/// Uniform blocks bound at a dynamic offset are this far apart.
const UNIFORM_STRIDE: u64 = 256;

/// Layers a frame can have; boxes past it are drawn without their effect.
const MAX_LAYERS: usize = 63;

/// The largest layer, in device pixels a side.
const MAX_LAYER_SIDE: u32 = 4096;

/// A custom shader's pipeline for one format, and whether it compiled.
struct EffectPipeline {
    pipeline: wgpu::RenderPipeline,
    /// The compiler's verdict, until it is in.
    #[cfg(not(target_arch = "wasm32"))]
    check: Option<std::pin::Pin<Box<dyn std::future::Future<Output = Option<wgpu::Error>> + Send>>>,
    #[cfg(target_arch = "wasm32")]
    check: Option<std::pin::Pin<Box<dyn std::future::Future<Output = Option<wgpu::Error>>>>>,
    error: Option<String>,
}

/// What a pass draws, in order.
#[derive(Debug, Clone, Copy)]
enum Op {
    /// Instances in one draw.
    Draw(u32, u32),
    /// The instance at this index, a surface with this id.
    Surface(u32, u64),
    /// The instance at this index, a layer put back through an effect (by
    /// its index in the frame's effects).
    Effect(u32, usize),
    /// The instance at this index, a material's backdrop: what is under
    /// it so far, blurred by this many device pixels.
    Backdrop(u32, f32),
}

/// A layer's drawing: its size in device pixels and what goes in it.
#[derive(Default)]
struct Target {
    size: (u32, u32),
    ops: Vec<Op>,
}

/// An effect drawn this frame: its shader, its uniform block, and the
/// layer (target) it puts back.
struct EffectDraw {
    shader: Shader,
    blend: crate::ui::BlendMode,
    uniform: EffectUniform,
    target: usize,
}

/// A layer being drawn: what to put back when it closes.
struct OpenLayer {
    target: usize,
    clips: Vec<Clip>,
    xforms: Vec<[f32; 9]>,
    rect: [f32; 4],
    effect: EffectDraw,
}

/// A subpath cut into straight pieces: its points (device pixels), and
/// whether it closes.
type Subpath = (Vec<(f32, f32)>, bool);

/// A clip: its box and corner radius (device pixels), and the one round it
/// that also cuts, if any.
type Clip = (Rect, f32, Option<(Rect, f32)>);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    rect: [f32; 4],
    color: [f32; 4],
    clip: [f32; 4],
    params: [f32; 4],
    sprite: [f32; 4],
    /// The projective map of the shape, by rows (device pixels); the first
    /// row's last the second clip's radius.
    xform: [[f32; 4]; 3],
    /// A second clip's box (no width: none).
    clip2: [f32; 4],
}

/// The identity, as an instance's projective map.
const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// `a` after `b`, 3 × 3 by rows.
fn compose(a: [f32; 9], b: [f32; 9]) -> [f32; 9] {
    std::array::from_fn(|i| {
        let (r, c) = (i / 3, i % 3);
        (0..3).map(|k| a[r * 3 + k] * b[k * 3 + c]).sum()
    })
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    viewport: [f32; 2],
    encode: f32,
    unused: f32,
}

/// Where a sprite is in the atlas.
#[derive(Clone, Copy)]
struct Slot {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// Coverage sprites, packed in shelves; emptied and repacked when full.
struct Atlas {
    texture: wgpu::Texture,
    size: u32,
    slots: HashMap<String, Slot>,
    shelf_x: u32,
    shelf_y: u32,
    shelf_h: u32,
}

impl Atlas {
    fn new(texture: wgpu::Texture, size: u32) -> Self {
        Atlas {
            texture,
            size,
            slots: HashMap::new(),
            shelf_x: 0,
            shelf_y: 0,
            shelf_h: 0,
        }
    }

    fn place(&mut self, w: u32, h: u32) -> Option<(u32, u32)> {
        if w > self.size || h > self.size {
            return None;
        }
        if self.shelf_x + w > self.size {
            self.shelf_y += self.shelf_h;
            self.shelf_x = 0;
            self.shelf_h = 0;
        }
        if self.shelf_y + h > self.size {
            return None;
        }
        let at = (self.shelf_x, self.shelf_y);
        self.shelf_x += w + 1;
        self.shelf_h = self.shelf_h.max(h + 1);
        Some(at)
    }

    fn reset(&mut self) {
        self.slots.clear();
        self.shelf_x = 0;
        self.shelf_y = 0;
        self.shelf_h = 0;
    }
}

/// A path of at most this many straight pieces is drawn from its distance
/// field on the GPU, whatever its size, rather than into a sprite.
const SDF_PATH_PIECES: usize = 96;

/// What the last [`UiPass::draw`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UiPassStats {
    /// Quads drawn, in the one draw call.
    pub instances: usize,
    /// Text runs and paths rasterised because the atlas lacked them.
    pub rasterized: usize,
    /// Text runs and paths left out because the atlas could not hold them.
    pub dropped: usize,
    /// Draw calls: one, and two more for each surface drawn.
    pub draws: usize,
    /// Subpaths drawn from their edges by the shader.
    pub gpu_paths: usize,
    /// Boxes drawn into layers and put back through custom shaders.
    pub effects: usize,
    /// Of those, ones whose shader failed and were put back as they are
    /// (see [`UiPass::effect_error`]).
    pub failed_effects: usize,
}

/// The GPU pass for a [`Scene`]. See the [module docs](self).
pub struct UiPass {
    /// Where the target's top left is in the UI, in UI pixels: a patch of
    /// it drawn on its own (see [`draw_patch`](Self::draw_patch)).
    origin: (f32, f32),
    shader: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
    /// By output format: the pass's pipeline, and the one for surfaces.
    pipelines: Vec<(wgpu::TextureFormat, wgpu::RenderPipeline, wgpu::RenderPipeline, wgpu::RenderPipeline)>,
    surface_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// Textures shown by surface nodes, by id, with their bind groups.
    surfaces: HashMap<u64, (wgpu::TextureView, Option<wgpu::BindGroup>)>,
    /// Path edges for the shader, and this frame's.
    edges: wgpu::Texture,
    edge_data: Vec<[f32; 4]>,
    /// Paths whose box covers more device pixels than this are drawn from
    /// their edges on the GPU; smaller ones (icons) are rasterised once and
    /// kept in the atlas. 0 draws every path on the GPU.
    pub gpu_path_area: f32,
    globals: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    instances: Option<(wgpu::Buffer, u64)>,
    atlas: Atlas,
    /// Images, in colour, by identity.
    images: Atlas,
    /// Measures text the way the sprites are drawn. Wide, because the
    /// overlay wraps text at the room it has left: a narrow one would
    /// measure every run broken onto lines.
    measure: Overlay,
    stats: UiPassStats,
    /// Custom shaders: their bind group layout, their pipelines by shader
    /// and format, their uniform blocks, and the layers they read.
    effect_layout: wgpu::BindGroupLayout,
    effect_pipelines: HashMap<(u64, wgpu::TextureFormat, crate::ui::BlendMode), EffectPipeline>,
    effect_uniforms: Option<(wgpu::Buffer, u64)>,
    layers: Vec<(wgpu::Texture, wgpu::TextureView, (u32, u32), wgpu::TextureFormat)>,
    passthrough: Shader,
    /// [`draw_retained`](Self::draw_retained)'s last frame: its picture,
    /// the scene and scale it was painted from.
    retained: Option<Retained>,
    /// Copies a retained picture to a target, by format.
    blits: HashMap<wgpu::TextureFormat, (wgpu::BindGroupLayout, wgpu::RenderPipeline)>,
    backdrops: Backdrops,
}

/// A separable Gaussian: one direction, reading `picture` at `ratio`
/// source pixels a target pixel (a downsample when above 1).
const BLUR: &str = r#"
struct Blur {
    dir: vec2<f32>,
    ratio: f32,
    sigma: f32,
};

@group(0) @binding(0) var picture: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var<uniform> blur: Blur;

@vertex
fn vs_blur(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
}

@fragment
fn fs_blur(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(picture));
    let at = p.xy * blur.ratio;
    let step = blur.dir * blur.ratio / dims;
    let r = i32(ceil(blur.sigma * 2.5));
    var sum = vec4<f32>(0.0);
    var weight = 0.0;
    for (var i = -r; i <= r; i = i + 1) {
        let x = f32(i);
        let w = exp(-(x * x) / (2.0 * blur.sigma * blur.sigma));
        sum = sum + textureSampleLevel(picture, linear_sampler, at / dims + step * x, 0.0) * w;
        weight = weight + w;
    }
    return sum / weight;
}
"#;

/// What a material's blur needs, made once: its pipeline (by format), the
/// scratch pictures (a copy of the region, then two smaller) and their
/// uniform blocks.
#[derive(Default)]
struct Backdrops {
    layout: Option<wgpu::BindGroupLayout>,
    pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    scratch: Vec<(wgpu::Texture, wgpu::TextureView, (u32, u32), wgpu::TextureFormat)>,
    uniforms: Option<(wgpu::Buffer, u64)>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurUniform {
    dir: [f32; 2],
    ratio: f32,
    sigma: f32,
}

/// How much smaller a backdrop blurred by `sigma` device pixels is blurred.
fn backdrop_step(sigma: f32) -> f32 {
    if sigma < 4.0 {
        1.0
    } else if sigma < 12.0 {
        2.0
    } else {
        4.0
    }
}

/// A frame kept to paint the next one over.
struct Retained {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    scene: Scene,
    scale: f32,
    clear: Rgba,
}

/// Copies a texture to the target, texel for pixel.
const BLIT: &str = r#"
@group(0) @binding(0) var picture: texture_2d<f32>;

@vertex
fn vs_blit(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
}

@fragment
fn fs_blit(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(picture, vec2<i32>(p.xy), 0);
}
"#;

impl UiPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers ui"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers ui bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    // One block a pass: the window's, each layer's.
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Globals>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let size = device.limits().max_texture_dimension_2d.min(4096);
        let atlas_texture = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let texture = atlas_texture("threers ui atlas", wgpu::TextureFormat::R8Unorm);
        // sRGB, so filtering and the blend see light, as the overlay mixes.
        let image_texture = atlas_texture("threers ui images", wgpu::TextureFormat::Rgba8UnormSrgb);
        let image_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers ui images sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers ui globals"),
            size: UNIFORM_STRIDE * (MAX_LAYERS as u64 + 1),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let surface_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers ui surface bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let effect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers ui effect bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<EffectUniform>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let edges = Self::edge_texture(device, 1);
        let bind_group = Self::make_bind_group(device, &layout, &globals, &texture, &image_texture, &image_sampler, &edges);
        UiPass {
            edges,
            edge_data: vec![],
            gpu_path_area: 256.0 * 256.0,
            shader,
            layout,
            pipelines: vec![],
            globals,
            bind_group,
            surface_layout,
            sampler: image_sampler,
            surfaces: HashMap::new(),
            instances: None,
            atlas: Atlas::new(texture, size),
            images: Atlas::new(image_texture, size),
            measure: Overlay::new(MEASURE_WIDTH, 1),
            stats: UiPassStats::default(),
            effect_layout,
            effect_pipelines: HashMap::new(),
            effect_uniforms: None,
            origin: (0.0, 0.0),
            layers: vec![],
            passthrough: passthrough(),
            retained: None,
            blits: HashMap::new(),
            backdrops: Backdrops::default(),
        }
    }

    fn edge_texture(device: &wgpu::Device, rows: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers ui path edges"),
            size: wgpu::Extent3d {
                width: EDGE_ROW,
                height: rows.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn make_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        globals: &wgpu::Buffer,
        atlas: &wgpu::Texture,
        images: &wgpu::Texture,
        sampler: &wgpu::Sampler,
        edges: &wgpu::Texture,
    ) -> wgpu::BindGroup {
        let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let (a, i, e) = (view(atlas), view(images), view(edges));
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers ui bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: globals,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<Globals>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&a),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&i),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&e),
                },
            ],
        })
    }

    /// Uploads this frame's path edges, growing the texture (and so the bind
    /// group) when they outgrow it.
    fn upload_edges(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.edge_data.is_empty() {
            return;
        }
        let rows = (self.edge_data.len() as u32).div_ceil(EDGE_ROW);
        if self.edges.height() < rows {
            self.edges = Self::edge_texture(device, rows.next_power_of_two());
            self.bind_group = Self::make_bind_group(
                device,
                &self.layout,
                &self.globals,
                &self.atlas.texture,
                &self.images.texture,
                &self.sampler,
                &self.edges,
            );
        }
        let mut texels = self.edge_data.clone();
        texels.resize((rows * EDGE_ROW) as usize, [0.0; 4]);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.edges,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(EDGE_ROW * 16),
                rows_per_image: Some(rows),
            },
            wgpu::Extent3d {
                width: EDGE_ROW,
                height: rows,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Measures text as this pass draws it at `scale` (device pixels per UI
    /// pixel): pass it to [`Ui::layout`](crate::ui::Ui::layout).
    pub fn text_measure(&mut self, scale: f32) -> OverlayPainter<'_> {
        OverlayPainter::with_scale(&mut self.measure, scale)
    }

    pub fn stats(&self) -> UiPassStats {
        self.stats
    }

    /// Draws `scene` into a new `size` image over `clear` (straight sRGB
    /// RGBA, 0..1) and reads it back: RGBA8, sRGB-encoded, rows from the
    /// top. For screenshots and golden-image tests of a UI. Native only: it waits for the GPU.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_to_rgba(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        scale: f32,
        size: (u32, u32),
        clear: Rgba,
    ) -> Vec<u8> {
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers ui image"),
            size: wgpu::Extent3d {
                width: size.0.max(1),
                height: size.1.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let c = crate::ui::linear(clear);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("threers ui image clear"),
        });
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("threers ui image clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // The target is sRGB, so the clear is given in light.
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: c.r as f64,
                        g: c.g as f64,
                        b: c.b as f64,
                        a: clear[3] as f64,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        queue.submit(Some(encoder.finish()));
        // Its own texture: materials can read what is under them.
        self.draw_with(device, queue, scene, scale, &view, Some(&target), size, format, None, None);
        super::readback::texture_bytes(device, queue, &target)
    }

    /// Draws only the parts `regions` (UI pixels) of `scene` into `target`,
    /// keeping everything else there: the patches that changed, a few
    /// widgets over a 3D view, one card of many. As [`draw`](Self::draw)
    /// otherwise.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_regions(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        scale: f32,
        target: &wgpu::TextureView,
        size: (u32, u32),
        format: wgpu::TextureFormat,
        regions: &[Rect],
    ) {
        let scissors: Vec<[u32; 4]> = regions.iter().filter_map(|r| device_rect(*r, scale, size)).collect();
        if scissors.is_empty() {
            return;
        }
        self.draw_with(device, queue, scene, scale, target, None, size, format, None, Some(&scissors));
    }

    /// Draws the patch `region` (UI pixels) of `scene` alone, filling
    /// `target` (`size` device pixels): its top left at the target's, at
    /// `size.0 / region.width` device pixels a UI pixel. A thumbnail of one
    /// card, a widget onto a texture for a 3D scene, a big UI in tiles.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_patch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        region: Rect,
        target: &wgpu::TextureView,
        size: (u32, u32),
        format: wgpu::TextureFormat,
    ) {
        let scale = size.0 as f32 / region.width.max(1e-3);
        self.origin = (region.x, region.y);
        self.draw_with(device, queue, scene, scale, target, None, size, format, None, None);
        self.origin = (0.0, 0.0);
    }

    /// [`render_to_rgba`](Self::render_to_rgba) of the patch `region` (UI
    /// pixels) alone, `scale` device pixels a UI pixel: RGBA8,
    /// sRGB-encoded, `(region.width * scale).round()` by
    /// `(region.height * scale).round()`, rows from the top.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_patch_to_rgba(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, region: Rect, scale: f32, clear: Rgba) -> ((u32, u32), Vec<u8>) {
        let size = (((region.width * scale).round() as u32).max(1), ((region.height * scale).round() as u32).max(1));
        self.origin = (region.x, region.y);
        let image = self.render_to_rgba(device, queue, scene, scale, size, clear);
        self.origin = (0.0, 0.0);
        (size, image)
    }

    /// The pass's WGSL, for validating it as a browser would.
    pub fn wgsl() -> &'static str {
        SHADER
    }

    /// The copy's and the materials' blur's WGSL (entry points `vs_blit`,
    /// `fs_blit`; `vs_blur`, `fs_blur`), for the same.
    #[doc(hidden)]
    pub fn helper_wgsl() -> [&'static str; 2] {
        [BLIT, BLUR]
    }

    /// The WGSL `shader` is compiled as (entry points `threers_vs` and
    /// `threers_fs`), for validating it as a browser would.
    pub fn effect_wgsl(shader: &Shader) -> String {
        effect_module(shader)
    }

    fn pipeline_index(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) -> usize {
        if let Some(i) = self.pipelines.iter().position(|(f, ..)| *f == format) {
            return i;
        }
        let main = self.make_pipeline(device, format, "fs_main", &[Some(&self.layout)]);
        let surface = self.make_pipeline(
            device,
            format,
            "fs_surface",
            &[Some(&self.layout), Some(&self.surface_layout)],
        );
        let backdrop = self.make_pipeline(device, format, "fs_backdrop", &[Some(&self.layout), Some(&self.surface_layout)]);
        self.pipelines.push((format, main, surface, backdrop));
        self.pipelines.len() - 1
    }

    fn make_pipeline(
        &self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        fragment: &str,
        groups: &[Option<&wgpu::BindGroupLayout>],
    ) -> wgpu::RenderPipeline {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers ui layout"),
            bind_group_layouts: groups,
            immediate_size: 0,
        });
        let attributes = wgpu::vertex_attr_array![
            0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32x4,
            5 => Float32x4, 6 => Float32x4, 7 => Float32x4, 8 => Float32x4
        ];
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("threers ui pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Instance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &attributes,
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: Some(fragment),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // Straight-alpha source-over, as the overlay composites.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    /// Shows `view` wherever a surface node has `id` (see
    /// [`NodeKind::Surface`](crate::ui::NodeKind::Surface)): a video frame,
    /// a camera, another renderer's output. Give an sRGB view for sRGB
    /// content, so it is filtered and blended as light. Set it again when
    /// the texture is replaced; a surface whose id has no texture is not
    /// drawn.
    pub fn set_surface(&mut self, id: u64, view: wgpu::TextureView) {
        self.surfaces.insert(id, (view, None));
    }

    pub fn remove_surface(&mut self, id: u64) {
        self.surfaces.remove(&id);
    }

    /// Draws `scene`, laid out in UI pixels, over `target` (`size` device
    /// pixels, `scale` of them per UI pixel), keeping what is there.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        scale: f32,
        target: &wgpu::TextureView,
        size: (u32, u32),
        format: wgpu::TextureFormat,
    ) {
        self.draw_with(device, queue, scene, scale, target, None, size, format, None, None);
    }

    /// Draws `scene` as [`draw`](Self::draw) does, into a picture kept from
    /// the last call, painting again only where it differs from the last
    /// scene (see [`Scene::damage`](crate::ui::scene::Scene::damage)), over
    /// `clear` (straight sRGB); then copies the picture to `target`. A
    /// window where one thing animates costs that thing, not the window.
    /// Returns what was painted again.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_retained(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        scale: f32,
        target: &wgpu::TextureView,
        size: (u32, u32),
        format: wgpu::TextureFormat,
        clear: Rgba,
    ) -> crate::ui::scene::SceneDamage {
        use crate::ui::scene::SceneDamage;
        if size.0 == 0 || size.1 == 0 {
            return SceneDamage::None;
        }
        let fresh = self.retained.as_ref().is_none_or(|r| r.size != size || r.format != format || r.scale != scale || r.clear != clear);
        if fresh {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers ui retained"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            self.retained = Some(Retained { texture, view, size, format, scene: Scene::default(), scale, clear });
        }
        let r = self.retained.take().expect("just made");
        let damage = if fresh { SceneDamage::Full } else { scene.damage(&r.scene, size.0 as f32 / scale, size.1 as f32 / scale) };
        let scissors: Option<Vec<[u32; 4]>> = match &damage {
            SceneDamage::None => Some(vec![]),
            SceneDamage::Full => None,
            SceneDamage::Regions(rects) => Some(
                rects
                    .iter()
                    .filter_map(|q| {
                        let x0 = ((q.x * scale).floor().max(0.0) as u32).min(size.0);
                        let y0 = ((q.y * scale).floor().max(0.0) as u32).min(size.1);
                        let x1 = ((q.x + q.width) * scale).ceil().clamp(0.0, size.0 as f32) as u32;
                        let y1 = ((q.y + q.height) * scale).ceil().clamp(0.0, size.1 as f32) as u32;
                        (x1 > x0 && y1 > y0).then_some([x0, y0, x1 - x0, y1 - y0])
                    })
                    .collect(),
            ),
        };
        if scissors.as_ref().is_none_or(|s| !s.is_empty()) {
            self.draw_with(device, queue, scene, scale, &r.view, Some(&r.texture), size, format, Some(clear), scissors.as_deref());
        }
        self.blit(device, queue, &r.view, target, format);
        self.retained = Some(Retained { scene: scene.clone(), ..r });
        damage
    }

    /// Forgets the retained picture: the next
    /// [`draw_retained`](Self::draw_retained) paints everything.
    pub fn invalidate(&mut self) {
        self.retained = None;
    }

    fn blit(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, picture: &wgpu::TextureView, target: &wgpu::TextureView, format: wgpu::TextureFormat) {
        let (layout, pipeline) = self.blits.entry(format).or_insert_with(|| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("threers ui blit"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BLIT)) });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("threers ui blit"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                }],
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("threers ui blit"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("threers ui blit"),
                layout: Some(&pl),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_blit"), compilation_options: Default::default(), buffers: &[] },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_blit"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
            (layout, pipeline)
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers ui blit"),
            layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(picture) }],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers ui blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
    }

    /// [`draw`](Self::draw), over `background` when given (a box over the
    /// whole target, first), and with the target's own pass cut to
    /// `scissors` (device pixels `[x, y, w, h]`) when given.
    #[allow(clippy::too_many_arguments)]
    fn draw_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        scale: f32,
        target: &wgpu::TextureView,
        target_texture: Option<&wgpu::Texture>,
        size: (u32, u32),
        format: wgpu::TextureFormat,
        background: Option<Rgba>,
        scissors: Option<&[[u32; 4]]>,
    ) {
        if size.0 == 0 || size.1 == 0 {
            return;
        }
        // Past the edges: a box ending on them shades its corners as edges.
        let whole = Rect { x: -2.0, y: -2.0, width: size.0 as f32 / scale + 4.0, height: size.1 as f32 / scale + 4.0 };
        let mut uploads = Uploads::default();
        self.edge_data.clear();
        let mut builder = Builder::new(&mut self.edge_data, self.gpu_path_area, &mut self.atlas, &mut self.images, &mut self.measure, &mut uploads, scale, false);
        if let Some(bg) = background {
            builder.fill_rect(whole, bg);
        }
        let (ox, oy) = self.origin;
        let shifted = ox != 0.0 || oy != 0.0;
        if shifted {
            builder.push_transform([1.0, 0.0, 0.0, 1.0, -ox, -oy]);
        }
        scene.replay(&mut builder);
        if shifted {
            builder.pop_transform();
        }
        let mut frame = Frame::take(&mut builder);
        let (sprites_full, images_full) = (builder.overflowed, builder.images_overflowed);
        if sprites_full || images_full {
            // Full: start the full atlas again with only this frame's
            // contents, and leave out what still does not fit.
            if sprites_full {
                self.atlas.reset();
            }
            if images_full {
                self.images.reset();
            }
            uploads = Uploads::default();
            self.edge_data.clear();
            let mut builder = Builder::new(&mut self.edge_data, self.gpu_path_area, &mut self.atlas, &mut self.images, &mut self.measure, &mut uploads, scale, true);
            if let Some(bg) = background {
                builder.fill_rect(whole, bg);
            }
            scene.replay(&mut builder);
            frame = Frame::take(&mut builder);
        }
        let Frame { mut stats, mut instances, surfaces, targets, effects, done } = frame;
        // Materials: each backdrop's region within the target it is read
        // from, and its place among this frame's blurs.
        let mut backdrop_slot: HashMap<u32, usize> = HashMap::new();
        let mut backdrop_list: Vec<([u32; 4], f32, f32)> = vec![];
        for (t, tg) in targets.iter().enumerate() {
            let (tw, th) = if t == 0 { size } else { tg.size };
            for &op in &tg.ops {
                if let Op::Backdrop(at, sigma) = op {
                    let inst = &mut instances[at as usize];
                    let [x, y, w, h] = inst.sprite;
                    let (x0, y0) = (x.clamp(0.0, tw as f32), y.clamp(0.0, th as f32));
                    let (x1, y1) = ((x + w).clamp(0.0, tw as f32), (y + h).clamp(0.0, th as f32));
                    inst.sprite = [x0, y0, x1 - x0, y1 - y0];
                    backdrop_slot.insert(at, backdrop_list.len());
                    backdrop_list.push(([x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32], sigma, inst.color[1]));
                }
            }
        }
        stats.instances = instances.len();
        stats.effects = effects.len();
        for (x, y, w, h, pixels) in uploads.sprites {
            upload(queue, &self.atlas.texture, x, y, w, h, 1, &pixels);
        }
        for (x, y, w, h, pixels) in uploads.images {
            upload(queue, &self.images.texture, x, y, w, h, 4, &pixels);
        }
        if instances.is_empty() {
            self.stats = stats;
            return;
        }
        // Each pass's viewport: the target's, then each layer texture's.
        let mut layer_views = vec![None; targets.len()];
        for (t, target) in targets.iter().enumerate().skip(1) {
            layer_views[t] = Some(self.layer(device, t - 1, target.size, format));
        }
        let mut blocks = vec![0u8; UNIFORM_STRIDE as usize * targets.len()];
        for (t, view) in layer_views.iter().enumerate() {
            let viewport = match view {
                Some(i) => {
                    let (w, h) = self.layers[*i].2;
                    [w as f32, h as f32]
                }
                None => [size.0 as f32, size.1 as f32],
            };
            let globals = Globals {
                viewport,
                encode: if format.is_srgb() { 0.0 } else { 1.0 },
                unused: 0.0,
            };
            let at = t * UNIFORM_STRIDE as usize;
            blocks[at..at + std::mem::size_of::<Globals>()].copy_from_slice(bytemuck::bytes_of(&globals));
        }
        queue.write_buffer(&self.globals, 0, &blocks);
        let bytes: &[u8] = bytemuck::cast_slice(&instances);
        let need = bytes.len() as u64;
        if self.instances.as_ref().is_none_or(|(_, cap)| *cap < need) {
            let cap = need.next_power_of_two().max(4096);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("threers ui instances"),
                size: cap,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instances = Some((buffer, cap));
        }
        self.upload_edges(device, queue);
        let index = self.pipeline_index(device, format);
        // Bind groups for the surfaces drawn, made when first needed.
        for &(_, id) in &surfaces {
            if let Some((view, group @ None)) = self.surfaces.get_mut(&id) {
                *group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers ui surface"),
                    layout: &self.surface_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                }));
            }
        }
        // The effects: their pipelines (a broken one's box is put back as
        // it is), uniform blocks and bind groups.
        let mut effect_groups = Vec::with_capacity(effects.len());
        if !effects.is_empty() {
            let need = UNIFORM_STRIDE * effects.len() as u64;
            if self.effect_uniforms.as_ref().is_none_or(|(_, cap)| *cap < need) {
                let cap = need.next_power_of_two().max(UNIFORM_STRIDE * 8);
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("threers ui effects"),
                    size: cap,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.effect_uniforms = Some((buffer, cap));
            }
            let mut blocks = vec![0u8; need as usize];
            for (i, e) in effects.iter().enumerate() {
                let at = i * UNIFORM_STRIDE as usize;
                blocks[at..at + std::mem::size_of::<EffectUniform>()].copy_from_slice(bytemuck::bytes_of(&e.uniform));
            }
            let (buffer, _) = self.effect_uniforms.as_ref().expect("just made");
            queue.write_buffer(buffer, 0, &blocks);
            for e in &effects {
                let layer = layer_views[e.target].expect("a layer");
                effect_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers ui effect"),
                    layout: &self.effect_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer,
                                offset: 0,
                                size: wgpu::BufferSize::new(std::mem::size_of::<EffectUniform>() as u64),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.layers[layer].1),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                }));
            }
            // Each effect's, and the stand-in for one that fails, in its blend.
            let shaders: Vec<(Shader, crate::ui::BlendMode)> = effects.iter().flat_map(|e| [(e.shader.clone(), e.blend), (self.passthrough.clone(), e.blend)]).collect();
            for (shader, blend) in shaders {
                self.effect_pipeline(device, &shader, format, blend);
            }
        }
        let backdrop_groups = if backdrop_list.is_empty() { None } else { Some(self.prepare_backdrops(device, queue, format, &backdrop_list)) };
        let effect_pipelines: Vec<&wgpu::RenderPipeline> = effects
            .iter()
            .map(|e| {
                let p = &self.effect_pipelines[&(e.shader.id(), format, e.blend)];
                match p.error {
                    None => &p.pipeline,
                    Some(_) => {
                        stats.failed_effects += 1;
                        &self.effect_pipelines[&(self.passthrough.id(), format, e.blend)].pipeline
                    }
                }
            })
            .collect();
        let (buffer, _) = self.instances.as_ref().expect("just made");
        queue.write_buffer(buffer, 0, bytes);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("threers ui encoder"),
        });
        let (_, main, surface, backdrop_pipeline) = &self.pipelines[index];
        let mut draws = 0;
        // Layers whose box lies outside every region painted again are left
        // as they are (nothing of theirs would show), and those inside them.
        let mut skip = vec![false; targets.len()];
        if let Some(regions) = scissors {
            let mut parent = vec![0usize; targets.len()];
            let mut put_back = vec![None; targets.len()];
            for (p, target) in targets.iter().enumerate() {
                for &op in &target.ops {
                    if let Op::Effect(at, e) = op {
                        parent[effects[e].target] = p;
                        put_back[effects[e].target] = Some(at);
                    }
                }
            }
            for (t, skipped) in skip.iter_mut().enumerate().skip(1) {
                // The layer's outermost one, put back into the target.
                let mut top = t;
                while parent[top] != 0 {
                    top = parent[top];
                }
                let Some(at) = put_back[top] else { continue };
                let inst = &instances[at as usize];
                let h = inst.xform;
                let [x, y, w, hgt] = inst.rect;
                let mut lo = [f32::MAX; 2];
                let mut hi = [f32::MIN; 2];
                let mut behind = false;
                for (cx, cy) in [(x - 2.0, y - 2.0), (x + w + 2.0, y - 2.0), (x - 2.0, y + hgt + 2.0), (x + w + 2.0, y + hgt + 2.0)] {
                    let q = [0, 1, 2].map(|r| h[r][0] * cx + h[r][1] * cy + h[r][2]);
                    if q[2] <= 1e-6 {
                        behind = true;
                        break;
                    }
                    for k in 0..2 {
                        lo[k] = lo[k].min(q[k] / q[2]);
                        hi[k] = hi[k].max(q[k] / q[2]);
                    }
                }
                let touches = behind
                    || regions.iter().any(|&[rx, ry, rw, rh]| lo[0] < (rx + rw) as f32 && hi[0] > rx as f32 && lo[1] < (ry + rh) as f32 && hi[1] > ry as f32);
                *skipped = !touches;
            }
        }
        // Each layer before the passes that put it back, the target last.
        for t in done.iter().copied().filter(|&t| !skip[t]).chain([0]) {
            let (view, load) = match layer_views[t] {
                Some(i) => (&self.layers[i].1, wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)),
                None => (target, wgpu::LoadOp::Load),
            };
            // A pass on this target: begun again, keeping what it holds,
            // after a material has read from it.
            macro_rules! begin {
                ($load:expr) => {{
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("threers ui pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations { load: $load, store: wgpu::StoreOp::Store },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_bind_group(0, &self.bind_group, &[(t as u64 * UNIFORM_STRIDE) as u32]);
                    pass.set_vertex_buffer(0, buffer.slice(..need));
                    pass
                }};
            }
            let mut pass = begin!(load);
            // What a material reads: this target's texture.
            let source = if t == 0 { target_texture } else { layer_views[t].map(|i| &self.layers[i].0) };
            // The target's pass, once for each region to paint again (all
            // of it when there are none given); a layer's, once.
            let regions: Vec<Option<[u32; 4]>> = match scissors {
                Some(s) if t == 0 => s.iter().copied().map(Some).collect(),
                _ => vec![None],
            };
            for region in regions {
                if let Some([x, y, w, h]) = region {
                    pass.set_scissor_rect(x, y, w, h);
                }
                for &op in &targets[t].ops {
                    match op {
                        Op::Draw(a, b) => {
                            pass.set_pipeline(main);
                            pass.draw(0..6, a..b);
                        }
                        Op::Surface(at, id) => {
                            let Some((_, Some(group))) = self.surfaces.get(&id) else { continue };
                            pass.set_pipeline(surface);
                            pass.set_bind_group(1, group, &[]);
                            pass.draw(0..6, at..at + 1);
                        }
                        Op::Effect(at, e) => {
                            pass.set_pipeline(effect_pipelines[e]);
                            pass.set_bind_group(1, &effect_groups[e], &[(e as u64 * UNIFORM_STRIDE) as u32]);
                            pass.draw(0..6, at..at + 1);
                        }
                        Op::Backdrop(at, _) => {
                            let (Some(src), Some(groups), Some(&slot)) = (source, backdrop_groups.as_ref(), backdrop_slot.get(&at)) else { continue };
                            let ([x, y, w, h], _, step) = backdrop_list[slot];
                            if w == 0 || h == 0 {
                                continue;
                            }
                            // What is under it so far, out of this pass: copied,
                            // blurred across (smaller), then down.
                            drop(pass);
                            let b = &self.backdrops;
                            encoder.copy_texture_to_texture(
                                wgpu::TexelCopyTextureInfo { texture: src, mip_level: 0, origin: wgpu::Origin3d { x, y, z: 0 }, aspect: wgpu::TextureAspect::All },
                                wgpu::TexelCopyTextureInfo { texture: &b.scratch[0].0, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                            );
                            let (sw, sh) = ((w as f32 / step).ceil() as u32, (h as f32 / step).ceil() as u32);
                            for (k, (group, into)) in [(&groups.0, 1usize), (&groups.1, 2usize)].into_iter().enumerate() {
                                let mut blur = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("threers ui backdrop blur"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &b.scratch[into].1,
                                        depth_slice: None,
                                        resolve_target: None,
                                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                                    })],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                    multiview_mask: None,
                                });
                                blur.set_pipeline(&b.pipelines[&format]);
                                blur.set_bind_group(0, group, &[((slot * 2 + k) as u64 * UNIFORM_STRIDE) as u32]);
                                blur.set_scissor_rect(0, 0, sw.max(1), sh.max(1));
                                blur.draw(0..3, 0..1);
                            }
                            pass = begin!(wgpu::LoadOp::Load);
                            if let Some([x, y, w, h]) = region {
                                pass.set_scissor_rect(x, y, w, h);
                            }
                            pass.set_pipeline(backdrop_pipeline);
                            pass.set_bind_group(1, &groups.2, &[]);
                            pass.draw(0..6, at..at + 1);
                        }
                    }
                    draws += 1;
                }
            }
        }
        stats.draws = draws;
        queue.submit(Some(encoder.finish()));
        self.stats = stats;
    }

    /// The blur's pipeline, scratch pictures (large enough for every
    /// backdrop this frame) and uniform blocks (two a backdrop: across,
    /// then down); and the bind groups that read them: the copy (to blur
    /// across), the first blur (to blur down), and the result (to draw).
    fn prepare_backdrops(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        list: &[([u32; 4], f32, f32)],
    ) -> (wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup) {
        let b = &mut self.backdrops;
        let layout = b.layout.get_or_insert_with(|| {
            let texture = |binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                count: None,
            };
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("threers ui backdrop blur"),
                entries: &[
                    texture(0),
                    wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<BlurUniform>() as u64) },
                        count: None,
                    },
                ],
            })
        });
        if let std::collections::hash_map::Entry::Vacant(slot) = b.pipelines.entry(format) {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("threers ui backdrop blur"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BLUR)) });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("threers ui backdrop blur"), bind_group_layouts: &[Some(&*layout)], immediate_size: 0 });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("threers ui backdrop blur"),
                layout: Some(&pl),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_blur"), compilation_options: Default::default(), buffers: &[] },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_blur"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
            slot.insert(pipeline);
        }
        // Scratch: the copy at full size, the blurs smaller.
        let full = list.iter().fold((1, 1), |a, (r, ..)| (a.0.max(r[2]), a.1.max(r[3])));
        let small = list.iter().fold((1, 1), |a, (r, _, step)| (a.0.max((r[2] as f32 / step).ceil() as u32), a.1.max((r[3] as f32 / step).ceil() as u32)));
        for (i, want) in [full, small, small].into_iter().enumerate() {
            let fits = b.scratch.get(i).is_some_and(|s| s.2.0 >= want.0 && s.2.1 >= want.1 && s.3 == format);
            if fits {
                continue;
            }
            let step = |v: u32| v.div_ceil(128) * 128;
            let have = b.scratch.get(i).filter(|s| s.3 == format).map_or((0, 0), |s| s.2);
            let grown = (step(want.0).max(have.0).min(MAX_LAYER_SIDE), step(want.1).max(have.1).min(MAX_LAYER_SIDE));
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers ui backdrop"),
                size: wgpu::Extent3d { width: grown.0, height: grown.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let entry = (texture, view, grown, format);
            if i < b.scratch.len() {
                b.scratch[i] = entry;
            } else {
                b.scratch.push(entry);
            }
        }
        // Across from the copy (and smaller by its step), then down.
        let need = UNIFORM_STRIDE * 2 * list.len() as u64;
        if b.uniforms.as_ref().is_none_or(|(_, cap)| *cap < need) {
            let cap = need.next_power_of_two().max(UNIFORM_STRIDE * 8);
            b.uniforms = Some((device.create_buffer(&wgpu::BufferDescriptor { label: Some("threers ui backdrop blurs"), size: cap, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }), cap));
        }
        let mut blocks = vec![0u8; need as usize];
        for (i, &(_, sigma, step)) in list.iter().enumerate() {
            let s = (sigma / step).max(0.5);
            for (k, u) in [BlurUniform { dir: [1.0, 0.0], ratio: step, sigma: s }, BlurUniform { dir: [0.0, 1.0], ratio: 1.0, sigma: s }].iter().enumerate() {
                let at = (i * 2 + k) * UNIFORM_STRIDE as usize;
                blocks[at..at + std::mem::size_of::<BlurUniform>()].copy_from_slice(bytemuck::bytes_of(u));
            }
        }
        let (buffer, _) = b.uniforms.as_ref().expect("just made");
        queue.write_buffer(buffer, 0, &blocks);
        let blur_group = |view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("threers ui backdrop blur"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<BlurUniform>() as u64) }),
                    },
                ],
            })
        };
        let across = blur_group(&b.scratch[0].1);
        let down = blur_group(&b.scratch[1].1);
        let result = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers ui backdrop"),
            layout: &self.surface_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&b.scratch[2].1) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        (across, down, result)
    }

    /// The layer texture `i`, at least `size` and in `format`: kept from
    /// frame to frame, grown in steps so a box that grows a little does not
    /// make a new one each frame.
    fn layer(&mut self, device: &wgpu::Device, i: usize, size: (u32, u32), format: wgpu::TextureFormat) -> usize {
        let fits = |l: &(wgpu::Texture, wgpu::TextureView, (u32, u32), wgpu::TextureFormat)| l.2.0 >= size.0 && l.2.1 >= size.1 && l.3 == format;
        if self.layers.get(i).is_some_and(fits) {
            return i;
        }
        let step = |v: u32| v.div_ceil(128) * 128;
        let have = self.layers.get(i).filter(|l| l.3 == format).map_or((0, 0), |l| l.2);
        let grown = (step(size.0).max(have.0).min(MAX_LAYER_SIDE), step(size.1).max(have.1).min(MAX_LAYER_SIDE));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers ui layer"),
            size: wgpu::Extent3d {
                width: grown.0,
                height: grown.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let layer = (texture, view, grown, format);
        if i < self.layers.len() {
            self.layers[i] = layer;
        } else {
            self.layers.push(layer);
        }
        i
    }

    /// `shader`'s pipeline for `format`, made the first time and checked by
    /// the compiler as it comes in.
    fn effect_pipeline(&mut self, device: &wgpu::Device, shader: &Shader, format: wgpu::TextureFormat, blend: crate::ui::BlendMode) {
        use std::future::Future;
        let key = (shader.id(), format, blend);
        if let Some(p) = self.effect_pipelines.get_mut(&key) {
            if let Some(check) = &mut p.check {
                let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
                if let std::task::Poll::Ready(result) = check.as_mut().poll(&mut cx) {
                    p.error = result.map(|e| e.to_string());
                    p.check = None;
                }
            }
            return;
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers ui effect"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(effect_module(shader))),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers ui effect layout"),
            bind_group_layouts: &[Some(&self.layout), Some(&self.effect_layout)],
            immediate_size: 0,
        });
        let attributes = wgpu::vertex_attr_array![
            0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32x4,
            5 => Float32x4, 6 => Float32x4, 7 => Float32x4, 8 => Float32x4
        ];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("threers ui effect pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("threers_vs"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Instance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &attributes,
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("threers_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // The layer is premultiplied, and so is what the shader
                    // makes of it.
                    blend: Some(blend_state(blend)),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let mut check = Box::pin(scope.pop());
        // Native compilers answer at once; a browser's later.
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        let (check, error) = match check.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(result) => (None, result.map(|e| e.to_string())),
            std::task::Poll::Pending => (Some(check as _), None),
        };
        self.effect_pipelines.insert(key, EffectPipeline { pipeline, check, error });
    }

    /// What the GPU's compiler said about `shader`, if it failed: boxes
    /// with it are drawn as they are. `None` too before it has been drawn,
    /// or while a browser is still compiling it.
    pub fn effect_error(&self, shader: &Shader) -> Option<&str> {
        self.effect_pipelines.iter().find(|((id, ..), _)| *id == shader.id()).and_then(|(_, p)| p.error.as_deref())
    }
}

/// How a premultiplied layer is put over what is behind it in each mode.
fn blend_state(mode: crate::ui::BlendMode) -> wgpu::BlendState {
    use crate::ui::BlendMode;
    use wgpu::{BlendComponent as C, BlendFactor as F, BlendOperation as O};
    let over = C { src_factor: F::One, dst_factor: F::OneMinusSrcAlpha, operation: O::Add };
    let color = match mode {
        BlendMode::Normal => over,
        // src·dst + dst·(1 − αs): the colours multiplied where it covers.
        BlendMode::Multiply => C { src_factor: F::Dst, dst_factor: F::OneMinusSrcAlpha, operation: O::Add },
        // src + dst − src·dst.
        BlendMode::Screen => C { src_factor: F::One, dst_factor: F::OneMinusSrc, operation: O::Add },
        BlendMode::Plus => C { src_factor: F::One, dst_factor: F::One, operation: O::Add },
        BlendMode::Darken => C { src_factor: F::One, dst_factor: F::One, operation: O::Min },
        BlendMode::Lighten => C { src_factor: F::One, dst_factor: F::One, operation: O::Max },
    };
    wgpu::BlendState { color, alpha: over }
}

/// What the builder made of a scene.
struct Frame {
    stats: UiPassStats,
    instances: Vec<Instance>,
    surfaces: Vec<(usize, u64)>,
    targets: Vec<Target>,
    effects: Vec<EffectDraw>,
    done: Vec<usize>,
}

impl Frame {
    fn take(b: &mut Builder<'_>) -> Frame {
        // Effects left open: closed, so their layers are put back.
        while !b.open.is_empty() {
            b.pop_effect();
        }
        Frame {
            stats: b.stats,
            instances: std::mem::take(&mut b.instances),
            surfaces: std::mem::take(&mut b.surfaces),
            targets: std::mem::take(&mut b.targets),
            effects: std::mem::take(&mut b.effects),
            done: std::mem::take(&mut b.done),
        }
    }
}

/// Copies a sprite into the atlas, its rows padded to what copies want.
/// What a frame adds to the atlases: `(x, y, w, h, pixels)`.
#[derive(Default)]
struct Uploads {
    sprites: Vec<(u32, u32, u32, u32, Vec<u8>)>,
    images: Vec<(u32, u32, u32, u32, Vec<u8>)>,
}

#[allow(clippy::too_many_arguments)]
fn upload(queue: &wgpu::Queue, texture: &wgpu::Texture, x: u32, y: u32, w: u32, h: u32, texel: u32, pixels: &[u8]) {
    let row = (w * texel) as usize;
    let padded = (w * texel).div_ceil(256) * 256;
    let mut rows = vec![0u8; (padded * h) as usize];
    for r in 0..h as usize {
        rows[r * padded as usize..][..row].copy_from_slice(&pixels[r * row..][..row]);
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x, y, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        &rows,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(padded),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
}

/// Turns painter calls into instances, rasterising sprites as it meets them.
struct Builder<'a> {
    edges: &'a mut Vec<[f32; 4]>,
    gpu_path_area: f32,
    atlas: &'a mut Atlas,
    images: &'a mut Atlas,
    measure: &'a mut Overlay,
    uploads: &'a mut Uploads,
    scale: f32,
    /// Device-pixel clip rectangles, each with its corner radius, and the
    /// clip round it where that one's rounded corners can still cut what
    /// is inside (a rounded box in a rounded box, overlapping): each shape
    /// is cut by both.
    clips: Vec<Clip>,
    instances: Vec<Instance>,
    stats: UiPassStats,
    overflowed: bool,
    images_overflowed: bool,
    /// Instances that are surfaces, by index, with their texture's id.
    surfaces: Vec<(usize, u64)>,
    /// The projective maps in force (device pixels), outermost first.
    xforms: Vec<[f32; 9]>,
    /// On the second try: leave out what does not fit instead of starting over.
    drop_overflow: bool,
    /// What each pass draws: the target's first, then each layer's.
    targets: Vec<Target>,
    /// The pass drawn into now.
    current: usize,
    /// Effects opened and not yet closed; `None` for one left out.
    open: Vec<Option<OpenLayer>>,
    effects: Vec<EffectDraw>,
    /// Layers in the order they closed: each is drawn before any it is in.
    done: Vec<usize>,
}

impl<'a> Builder<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        edges: &'a mut Vec<[f32; 4]>,
        gpu_path_area: f32,
        atlas: &'a mut Atlas,
        images: &'a mut Atlas,
        measure: &'a mut Overlay,
        uploads: &'a mut Uploads,
        scale: f32,
        drop_overflow: bool,
    ) -> Self {
        Builder {
            edges,
            gpu_path_area,
            atlas,
            images,
            measure,
            uploads,
            scale,
            clips: vec![],
            xforms: vec![],
            instances: vec![],
            stats: UiPassStats::default(),
            overflowed: false,
            images_overflowed: false,
            surfaces: vec![],
            drop_overflow,
            targets: vec![Target::default()],
            current: 0,
            open: vec![],
            effects: vec![],
            done: vec![],
        }
    }

    /// Notes that the instance at `i` is drawn in the current pass.
    fn record(&mut self, i: usize) {
        let i = i as u32;
        let ops = &mut self.targets[self.current].ops;
        match ops.last_mut() {
            Some(Op::Draw(_, end)) if *end == i => *end = i + 1,
            _ => ops.push(Op::Draw(i, i + 1)),
        }
    }

    /// Draws the instance just pushed as `op` instead, in its own draw.
    fn retag(&mut self, op: Op) {
        let ops = &mut self.targets[self.current].ops;
        if let Some(Op::Draw(start, end)) = ops.last_mut() {
            *end -= 1;
            if *start == *end {
                ops.pop();
            }
        }
        ops.push(op);
    }

    fn px(&self, r: Rect) -> Rect {
        let k = self.scale;
        Rect {
            x: r.x * k,
            y: r.y * k,
            width: r.width * k,
            height: r.height * k,
        }
    }

    fn push(&mut self, rect: Rect, color: Rgba, radius: f32, sprite: Option<(u32, u32)>) {
        let c = crate::ui::linear(color);
        let (clip, clip_radius, clipped, second) = match self.clips.last() {
            Some(&(r, radius, second)) => (r, radius, 1.0, second),
            None => (Rect::default(), 0.0, 0.0, None),
        };
        if clipped > 0.0 && (clip.width <= 0.0 || clip.height <= 0.0) {
            return;
        }
        let (sx, sy) = sprite.unwrap_or((0, 0));
        let h = self.xforms.last().copied().unwrap_or(IDENTITY);
        let (c2, r2) = second.map_or(([0.0; 4], 0.0), |(r, radius)| ([r.x, r.y, r.width, r.height], radius));
        self.instances.push(Instance {
            rect: [rect.x, rect.y, rect.width, rect.height],
            color: [c.r, c.g, c.b, color[3]],
            clip: [clip.x, clip.y, clip.width, clip.height],
            params: [radius, clip_radius, sprite.is_some() as u32 as f32, clipped],
            sprite: [sx as f32, sy as f32, 0.0, 0.0],
            xform: [[h[0], h[1], h[2], r2], [h[3], h[4], h[5], 0.0], [h[6], h[7], h[8], 0.0]],
            clip2: c2,
        });
        self.record(self.instances.len() - 1);
    }

    /// A sprite for `key`: from the atlas, or drawn by `draw` into a fresh
    /// overlay of the size `size` gives and added. `None` when it cannot be
    /// held.
    fn sprite(
        &mut self,
        key: String,
        size: impl FnOnce() -> (u32, u32),
        draw: impl FnOnce(&mut OverlayPainter<'_>),
    ) -> Option<Slot> {
        if let Some(slot) = self.atlas.slots.get(&key) {
            return Some(*slot);
        }
        let (w, h) = size();
        let Some((x, y)) = self.atlas.place(w, h) else {
            if self.drop_overflow || w > self.atlas.size || h > self.atlas.size {
                self.stats.dropped += 1;
            } else {
                self.overflowed = true;
            }
            return None;
        };
        let mut overlay = Overlay::new(w, h);
        draw(&mut OverlayPainter::with_scale(&mut overlay, self.scale));
        let coverage: Vec<u8> = overlay.pixels().as_chunks::<4>().0.iter().map(|p| p[3]).collect();
        self.uploads.sprites.push((x, y, w, h, coverage));
        self.stats.rasterized += 1;
        let slot = Slot { x, y, w, h };
        self.atlas.slots.insert(key, slot);
        Some(slot)
    }

    /// A path drawn from its edges by the shader, subpath by subpath as the
    /// overlay draws it: the fill, then the stroke.
    /// `d` in `view_box`, mapped onto `dev` (device pixels) and cut into
    /// straight pieces: each subpath's points and whether it closes, and
    /// the scale from view-box units to device pixels.
    fn flatten(dev: Rect, d: &[String], view_box: [f32; 4]) -> (Vec<Subpath>, f32) {
        let [vx, vy, vw, vh] = view_box;
        let (sx, sy) = (dev.width / vw.max(1e-6), dev.height / vh.max(1e-6));
        let mut out = vec![];
        for data in d {
            let Ok(subpaths) = crate::curves::parse_svg_subpaths(data) else {
                continue;
            };
            for sub in subpaths {
                let closed = sub.curve_path.auto_close;
                let points: Vec<(f32, f32)> = sub
                    .get_points(crate::ui::curve_divisions(dev))
                    .into_iter()
                    .map(|p| (dev.x + (p.x - vx) * sx, dev.y + (p.y - vy) * sy))
                    .collect();
                if points.len() >= 2 {
                    out.push((points, closed));
                }
            }
        }
        (out, sx.min(sy))
    }

    /// A path drawn from its distance field: each pixel's distance to its
    /// pieces (a stroke) and its winding (a fill), worked out on the GPU,
    /// exact at any size.
    #[allow(clippy::too_many_arguments)]
    fn gpu_path(&mut self, flat: &[Subpath], scale: f32, stroke: Option<f32>, fill: bool, color: Rgba) {
        for (points, closed) in flat {
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for &(x, y) in points {
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
            }
            if fill && points.len() >= 3 {
                let start = self.edges.len();
                let n = points.len();
                for i in 0..n {
                    let (a, b) = (points[i], points[(i + n - 1) % n]);
                    self.edges.push([a.0, a.1, b.0, b.1]);
                }
                let quad = Rect { x: x0 - 1.0, y: y0 - 1.0, width: x1 - x0 + 2.0, height: y1 - y0 + 2.0 };
                self.push_edges(quad, color, 6.0, start, n, 0.0);
            }
            if let Some(w) = stroke {
                let half = w * scale * 0.5;
                let mut pts = points.clone();
                if *closed {
                    pts.push(pts[0]);
                }
                let start = self.edges.len();
                for pair in pts.windows(2) {
                    self.edges.push([pair[0].0, pair[0].1, pair[1].0, pair[1].1]);
                }
                let pad = half + 1.0;
                let quad = Rect { x: x0 - pad, y: y0 - pad, width: x1 - x0 + 2.0 * pad, height: y1 - y0 + 2.0 * pad };
                self.push_edges(quad, color, 7.0, start, pts.len() - 1, half);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push_edges(&mut self, quad: Rect, color: Rgba, kind: f32, start: usize, count: usize, half: f32) {
        let before = self.instances.len();
        self.push(quad, color, 0.0, None);
        if self.instances.len() > before {
            let last = self.instances.last_mut().expect("just pushed");
            last.params[2] = kind;
            last.sprite = [start as f32, count as f32, half, 0.0];
            self.stats.gpu_paths += 1;
        }
    }

    /// Places a sprite whose pixels start at device pixel `(x0, y0)`.
    fn push_sprite(&mut self, slot: Slot, x0: i32, y0: i32, color: Rgba) {
        let rect = Rect {
            x: x0 as f32,
            y: y0 as f32,
            width: slot.w as f32,
            height: slot.h as f32,
        };
        self.push(rect, color, 0.0, Some((slot.x, slot.y)));
    }
}

const WHITE: Rgba = [1.0, 1.0, 1.0, 1.0];

/// Edge texels per row of the edge texture.
const EDGE_ROW: u32 = 1024;

/// Room enough that no run a UI lays out on one line wraps when measured.
const MEASURE_WIDTH: u32 = 16384;

impl TextMeasure for Builder<'_> {
    fn measure(&mut self, _: &str, _: &Style) -> (f32, f32) {
        unreachable!("painting does not measure text")
    }
}

impl Painter for Builder<'_> {
    /// One instance a stretch between stops, its colour mixed at each
    /// pixel (the end colours carried out to the box's edges, as CSS
    /// does): smooth, where bands of flat colour step, and seamless. Cut
    /// to a square clip as a rectangle is.
    fn fill_linear_gradient(&mut self, rect: Rect, stops: &[(f32, Rgba)], _blur: f32) {
        if stops.is_empty() {
            return;
        }
        let r = self.px(rect);
        if r.width <= 0.0 || r.height <= 0.0 {
            return;
        }
        // The colour at `t` across the box, mixed as encoded.
        let at = |t: f32| -> Rgba {
            let (first, last) = (stops[0], stops[stops.len() - 1]);
            if t <= first.0 {
                return first.1;
            }
            if t >= last.0 {
                return last.1;
            }
            let i = stops.windows(2).position(|w| t >= w[0].0 && t < w[1].0).unwrap_or(0);
            let ((t0, c0), (t1, c1)) = (stops[i], stops[i + 1]);
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            std::array::from_fn(|c| c0[c] + (c1[c] - c0[c]) * f)
        };
        let (mut x0, mut y0, mut x1, mut y1) = (r.x, r.y, r.x + r.width, r.y + r.height);
        let square = matches!(self.clips.last(), Some((_, radius, None)) if *radius <= 0.0);
        if let Some((clip, _, _)) = self.clips.last().copied().filter(|_| square) {
            x0 = x0.max(clip.x);
            y0 = y0.max(clip.y);
            x1 = x1.min(clip.x + clip.width);
            y1 = y1.min(clip.y + clip.height);
        }
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        // Stretches between the stops, within what shows.
        let mut cuts = vec![x0];
        cuts.extend(stops.iter().map(|&(t, _)| r.x + r.width * t).filter(|&x| x > x0 && x < x1));
        cuts.push(x1);
        let clips = if square { std::mem::take(&mut self.clips) } else { vec![] };
        for w in cuts.windows(2) {
            let (a, b) = (w[0], w[1]);
            if b <= a {
                continue;
            }
            let (c0, c1) = (at((a - r.x) / r.width), at((b - r.x) / r.width));
            let before = self.instances.len();
            self.push(Rect { x: a, y: y0, width: b - a, height: y1 - y0 }, c0, 0.0, None);
            if self.instances.len() > before {
                let right = crate::ui::linear(c1);
                let last = self.instances.last_mut().expect("just pushed");
                last.params[2] = 8.0;
                last.sprite = [right.r, right.g, right.b, c1[3]];
            }
        }
        if square {
            self.clips = clips;
        }
    }

    /// A radial, conic or turned linear gradient: one instance, `t` worked
    /// out at each fragment, its colour from a 256-texel ramp of the stops
    /// kept in the image atlas.
    fn fill_gradient(&mut self, rect: Rect, shape: crate::ui::GradientShape, stops: &[(f32, Rgba)], blur: f32) {
        use crate::ui::GradientShape;
        if shape.is_across() || stops.is_empty() {
            return self.fill_linear_gradient(rect, stops, blur);
        }
        let key = format!("G{stops:?}");
        let slot = match self.images.slots.get(&key) {
            Some(slot) => *slot,
            None => {
                let Some((x, y)) = self.images.place(256, 1) else {
                    self.images_overflowed |= !self.drop_overflow;
                    return;
                };
                let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                let ramp: Vec<u8> = (0..256).flat_map(|i| crate::ui::gradient_at(stops, i as f32 / 255.0).map(byte)).collect();
                let slot = Slot { x, y, w: 256, h: 1 };
                self.images.slots.insert(key, slot);
                self.uploads.images.push((x, y, 256, 1, ramp));
                slot
            }
        };
        let r = self.px(rect);
        let (cx, cy, params) = match shape {
            GradientShape::Linear(deg) => {
                let a = deg.to_radians();
                let (dx, dy) = (a.sin(), -a.cos());
                let len = (r.width * dx).abs() + (r.height * dy).abs();
                (r.x + r.width / 2.0, r.y + r.height / 2.0, [0.0, dx, dy, len])
            }
            GradientShape::Radial { cx, cy } => {
                let (px, py) = (r.x + r.width * cx, r.y + r.height * cy);
                let far = [(r.x, r.y), (r.x + r.width, r.y), (r.x, r.y + r.height), (r.x + r.width, r.y + r.height)]
                    .iter()
                    .map(|&(qx, qy)| ((qx - px).powi(2) + (qy - py).powi(2)).sqrt())
                    .fold(0.0f32, f32::max);
                (px, py, [1.0, far, 0.0, 0.0])
            }
            GradientShape::Conic { cx, cy, from } => (r.x + r.width * cx, r.y + r.height * cy, [2.0, from.to_radians(), 0.0, 0.0]),
        };
        let before = self.instances.len();
        self.push(r, WHITE, 0.0, None);
        if self.instances.len() > before {
            let last = self.instances.last_mut().expect("just pushed");
            last.params[2] = 10.0;
            last.color = params;
            last.sprite = [slot.x as f32, slot.y as f32, cx, cy];
        }
    }

    fn fill_rect(&mut self, rect: Rect, color: Rgba) {
        let r = self.px(rect);
        // Under a square clip, cut the rectangle itself, as the overlay does:
        // the cut edge is then shaded by the one field, where clip coverage
        // times the rectangle's would come out lighter at a corner.
        match self.clips.last().copied() {
            Some((clip, radius, None)) if radius <= 0.0 => {
                let x0 = r.x.max(clip.x);
                let y0 = r.y.max(clip.y);
                let x1 = (r.x + r.width).min(clip.x + clip.width);
                let y1 = (r.y + r.height).min(clip.y + clip.height);
                if x1 > x0 && y1 > y0 {
                    let cut = Rect {
                        x: x0,
                        y: y0,
                        width: x1 - x0,
                        height: y1 - y0,
                    };
                    let clips = std::mem::take(&mut self.clips);
                    self.push(cut, color, 0.0, None);
                    self.clips = clips;
                }
            }
            _ => self.push(r, color, 0.0, None),
        }
    }

    fn fill_rounded_rect(&mut self, rect: Rect, rx: f32, ry: f32, color: Rgba) {
        let r = self.px(rect);
        self.push(r, color, rx.min(ry) * self.scale, None);
    }

    fn text(&mut self, line: Rect, text: &str, style: &TextStyle) {
        if text.is_empty() || style.color[3] <= 0.0 {
            return;
        }
        let k = self.scale;
        let mut dev = self.px(line);
        // On a whole device pixel down, as browsers put a baseline: a
        // horizontal stem between two rows is drawn as two half-grey rows.
        // Across, a quarter pixel is close enough to be seen as exact, and
        // lets one sprite serve more places.
        // Where it lands: under a plain move, snapped where it shows; under
        // a turn or a scale, no pixel grid lines up with it anyway.
        let h = self.xforms.last().copied().unwrap_or(IDENTITY);
        if h[0] == 1.0 && h[1] == 0.0 && h[3] == 0.0 && h[4] == 1.0 && h[6] == 0.0 && h[7] == 0.0 && h[8] == 1.0 {
            dev.y = (dev.y + h[5]).round() - h[5];
            dev.x = ((dev.x + h[2]) * 4.0).round() / 4.0 - h[2];
        }
        // Room round the line box for what glyphs reach past it: ascenders,
        // descenders, an italic's lean, synthesised bold.
        let pad = (style.font_size * k * 0.5).ceil() as i32 + 2;
        let (x0, y0) = (dev.x.floor() as i32 - pad, dev.y.floor() as i32 - pad);
        // The line where it falls within its whole pixel, so a sprite keeps
        // the overlay's sub-pixel placement.
        let (fx, fy) = (dev.x - dev.x.floor(), dev.y - dev.y.floor());
        // In its own colour, at full strength: the caption painter bends
        // glyph coverage by the colour's luminance (as Skia does), so a white
        // mask tinted grey would not be the grey text the overlay draws. The
        // pass applies the alpha.
        let [r, g, b, _] = style.color;
        let opaque = TextStyle {
            color: [r, g, b, 1.0],
            ..style.clone()
        };
        let key = format!("T{k:?}|{fx:?}|{fy:?}|{:?}|{text}|{opaque:?}", dev.height);
        let size = if self.atlas.slots.contains_key(&key) {
            (0, 0)
        } else {
            let o = &mut *self.measure;
            o.text_size(style.font_size * k);
            o.text_weight(style.weight());
            o.text_italic(style.italic);
            let (tw, th) = o.measure_text(text);
            // The overlay measures a run against 80% of its width and wraps
            // what is wider, so the sprite is wide enough that the run stays
            // one line, as it is on a full-size overlay.
            let w = ((tw + k) * 1.25 + fx).ceil() as u32 + 2 * pad as u32;
            // Twice the room below: the caption painter slides a block that
            // reaches the canvas's bottom edge up off it.
            let h = (dev.height.max(th) + fy).ceil() as u32 + 4 * pad as u32;
            (w, h)
        };
        let size = || size;
        let local = Rect {
            x: (fx + pad as f32) / k,
            y: (fy + pad as f32) / k,
            width: line.width,
            height: line.height,
        };
        if let Some(slot) = self.sprite(key, size, |p| p.text(local, text, &opaque)) {
            self.push_sprite(slot, x0, y0, style.color);
        }
    }

    fn path(
        &mut self,
        rect: Rect,
        d: &[String],
        view_box: [f32; 4],
        stroke: Option<f32>,
        fill: bool,
        color: Rgba,
    ) {
        let k = self.scale;
        let dev = self.px(rect);
        // From its distance field on the GPU: a large path (a sprite of it
        // would be large), and any of few pieces (an icon's lines, a tick, a
        // needle, a chart's line), where working the distance out costs
        // little and the result is exact at any size, turned or scaled.
        // Others are drawn once into a sprite and reused.
        let (flat, scale) = Self::flatten(dev, d, view_box);
        let pieces: usize = flat.iter().map(|(p, _)| p.len()).sum();
        if dev.width * dev.height > self.gpu_path_area || pieces <= SDF_PATH_PIECES {
            self.gpu_path(&flat, scale, stroke, fill, color);
            return;
        }
        let reach = stroke.unwrap_or(0.0) * k * (dev.width / view_box[2].max(1e-6)).max(1.0);
        let pad = reach.ceil() as i32 + 2;
        let (x0, y0) = (dev.x.floor() as i32 - pad, dev.y.floor() as i32 - pad);
        let (fx, fy) = (dev.x - dev.x.floor(), dev.y - dev.y.floor());
        let w = (dev.width + fx).ceil() as u32 + 2 * pad as u32;
        let h = (dev.height + fy).ceil() as u32 + 2 * pad as u32;
        let size = || (w, h);
        let key = format!("P{k:?}|{fx:?}|{fy:?}|{:?}|{:?}|{view_box:?}|{stroke:?}|{fill}|{d:?}", dev.width, dev.height);
        let local = Rect {
            x: (fx + pad as f32) / k,
            y: (fy + pad as f32) / k,
            width: rect.width,
            height: rect.height,
        };
        if let Some(slot) = self.sprite(key, size, |p| p.path(local, d, view_box, stroke, fill, WHITE)) {
            self.push_sprite(slot, x0, y0, color);
        }
    }

    /// From its distance field (kind 5): smooth at any size.
    fn stroke_arc(&mut self, arc: crate::ui::ArcShape, width: f32, color: Rgba) {
        let k = self.scale;
        let half = width * k / 2.0;
        let (r, cx, cy) = (arc.radius * k, arc.cx * k, arc.cy * k);
        let reach = r + half + 1.0;
        let quad = Rect { x: cx - reach, y: cy - reach, width: 2.0 * reach, height: 2.0 * reach };
        let before = self.instances.len();
        self.push(quad, color, 0.0, None);
        if self.instances.len() > before {
            let last = self.instances.last_mut().expect("just pushed");
            last.params[2] = 5.0;
            last.sprite = [r, half, arc.start, arc.sweep];
        }
    }

    fn shadow(&mut self, rect: Rect, radius: f32, blur: f32, color: Rgba) {
        let (r, k) = (self.px(rect), self.scale);
        let sigma = blur * k / 2.0;
        if sigma < 0.5 {
            // Sharp: the rounded box itself, as the overlay draws it.
            self.push(r, color, radius * k, None);
            return;
        }
        let before = self.instances.len();
        self.push(r, color, radius * k, None);
        if self.instances.len() > before {
            let last = self.instances.last_mut().expect("just pushed");
            last.params[2] = 3.0;
            last.sprite = [sigma, 0.0, 0.0, 0.0];
        }
    }

    fn underline(&mut self, x: f32, y: f32, width: f32, thickness: f32, color: Rgba, wavy: bool) {
        let k = self.scale;
        let (x, y, width, t) = (x * k, y * k, width * k, thickness * k);
        if !wavy {
            self.fill_rect(
                Rect {
                    x: x / k,
                    y: (y - t / 2.0) / k,
                    width: width / k,
                    height: t / k,
                },
                color,
            );
            return;
        }
        let a = t * crate::ui::WAVE_AMPLITUDE;
        let period = t * crate::ui::WAVE_PERIOD;
        let reach = a + t;
        let before = self.instances.len();
        self.push(
            Rect {
                x: x - reach,
                y: y - reach,
                width: width + 2.0 * reach,
                height: 2.0 * reach,
            },
            color,
            0.0,
            None,
        );
        if self.instances.len() > before {
            let last = self.instances.last_mut().expect("just pushed");
            // The quad covers the wave; the wave itself runs x .. x + width.
            last.rect = [x - reach, y - reach, width + 2.0 * reach, 2.0 * reach];
            last.params = [t / 2.0, last.params[1], 4.0, last.params[3]];
            last.sprite = [x, y, a, std::f32::consts::TAU / period.max(1e-3)];
        }
    }

    fn backdrop(&mut self, rect: Rect, radius: f32, blur: f32, saturate: f32) {
        let (r, k) = (self.px(rect), self.scale);
        let sigma = blur * k;
        // The region read: the box, and as far round it as the blur reaches,
        // where it lands in this target.
        let h = self.xforms.last().copied().unwrap_or(IDENTITY);
        let reach = (sigma * 3.0).ceil();
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for (x, y) in [(r.x - reach, r.y - reach), (r.x + r.width + reach, r.y - reach), (r.x - reach, r.y + r.height + reach), (r.x + r.width + reach, r.y + r.height + reach)] {
            let w = (h[6] * x + h[7] * y + h[8]).max(1e-4);
            let (px, py) = ((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w);
            lo = [lo[0].min(px), lo[1].min(py)];
            hi = [hi[0].max(px), hi[1].max(py)];
        }
        let region = [lo[0].floor(), lo[1].floor(), hi[0].ceil() - lo[0].floor(), hi[1].ceil() - lo[1].floor()];
        let before = self.instances.len();
        self.push(r, [saturate, backdrop_step(sigma), 0.0, 1.0], radius * k, None);
        if self.instances.len() > before {
            let last = self.instances.last_mut().expect("just pushed");
            // The colour carries numbers, not a colour: as given.
            last.color = [saturate, backdrop_step(sigma), 0.0, 1.0];
            last.sprite = region;
            self.retag(Op::Backdrop(before as u32, sigma));
        }
    }

    fn surface(&mut self, rect: Rect, id: u64, opacity: f32) {
        let r = self.px(rect);
        let before = self.instances.len();
        self.push(r, [1.0, 1.0, 1.0, opacity], 0.0, None);
        if self.instances.len() > before {
            self.instances.last_mut().expect("just pushed").params[2] = 5.0;
            self.surfaces.push((before, id));
            self.retag(Op::Surface(before as u32, id));
        }
    }

    fn image(&mut self, rect: Rect, image: &crate::ui::image::ImageRef, opacity: f32) {
        let key = format!("I{}", image.id());
        let slot = match self.images.slots.get(&key) {
            Some(slot) => *slot,
            None => {
                let Some((x, y)) = self.images.place(image.width, image.height) else {
                    if self.drop_overflow || image.width > self.images.size || image.height > self.images.size {
                        self.stats.dropped += 1;
                    } else {
                        self.images_overflowed = true;
                    }
                    return;
                };
                let slot = Slot {
                    x,
                    y,
                    w: image.width,
                    h: image.height,
                };
                self.images.slots.insert(key, slot);
                self.uploads.images.push((x, y, image.width, image.height, image.rgba.clone()));
                self.stats.rasterized += 1;
                slot
            }
        };
        let r = self.px(rect);
        // White at the opacity: the shader takes the colour from the image.
        let before = self.instances.len();
        self.push(r, [1.0, 1.0, 1.0, opacity], 0.0, Some((slot.x, slot.y)));
        if self.instances.len() > before {
            let last = self.instances.last_mut().expect("just pushed");
            last.params[2] = 2.0;
            last.sprite = [slot.x as f32, slot.y as f32, slot.w as f32, slot.h as f32];
        }
    }

    fn push_clip(&mut self, rect: Rect) {
        self.push_clip_rounded(rect, 0.0, 0.0);
    }

    fn push_clip_rounded(&mut self, rect: Rect, rx: f32, ry: f32) {
        let mut r = self.px(rect);
        // Clips are tested on screen: one inside a projective map clips to
        // the box its corners are taken to.
        if let Some(&h) = self.xforms.last() {
            let at = |x: f32, y: f32| {
                let w = h[6] * x + h[7] * y + h[8];
                ((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w)
            };
            let c = [at(r.x, r.y), at(r.x + r.width, r.y), at(r.x, r.y + r.height), at(r.x + r.width, r.y + r.height)];
            let (x0, x1) = c.iter().fold((f32::MAX, f32::MIN), |(a, b), p| (a.min(p.0), b.max(p.0)));
            let (y0, y1) = c.iter().fold((f32::MAX, f32::MIN), |(a, b), p| (a.min(p.1), b.max(p.1)));
            r = Rect { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
        }
        let radius = rx.min(ry) * self.scale;
        let clip = match self.clips.last() {
            None => (r, radius, None),
            Some(&(top, top_radius, _)) => {
                let x0 = r.x.max(top.x);
                let y0 = r.y.max(top.y);
                let x1 = (r.x + r.width).min(top.x + top.width);
                let y1 = (r.y + r.height).min(top.y + top.height);
                let cut = Rect {
                    x: x0,
                    y: y0,
                    width: (x1 - x0).max(0.0),
                    height: (y1 - y0).max(0.0),
                };
                // Whether `inner` keeps clear of `outer`'s rounded corners
                // (inside it by the corners' reach): then `outer` cuts it
                // as its box alone would.
                let clear = |inner: Rect, outer: Rect, r: f32| {
                    let k = r * (1.0 - std::f32::consts::FRAC_1_SQRT_2) + 0.5;
                    r <= 0.0 || (inner.x >= outer.x + k && inner.y >= outer.y + k && inner.x + inner.width <= outer.x + outer.width - k && inner.y + inner.height <= outer.y + outer.height - k)
                };
                if cut == r && clear(r, top, top_radius) {
                    // Inside it, clear of its corners: its own shape.
                    (r, radius, None)
                } else if cut == top && clear(top, r, radius) {
                    // Round the outer one: that one's shape.
                    (top, top_radius, None)
                } else if radius <= 0.0 {
                    // A square box crossing a rounded one: the two boxes'
                    // overlap, cut by the rounded one too.
                    (cut, 0.0, (top_radius > 0.0).then_some((top, top_radius)))
                } else {
                    // Rounded and crossing: each cuts with its own corners.
                    (r, radius, Some((top, top_radius)))
                }
            }
        };
        self.clips.push(clip);
    }

    fn pop_clip(&mut self) {
        self.clips.pop();
    }

    fn push_transform(&mut self, m: [f32; 6]) {
        self.push_projective([m[0], m[2], m[4], m[1], m[3], m[5], 0.0, 0.0, 1.0], m);
    }

    /// True perspective: the map is applied to each shape's corners on the
    /// GPU, and the shape shaded where it is on itself.
    fn push_projective(&mut self, h: [f32; 9], _affine: [f32; 6]) {
        // In device pixels: scaled in, mapped, scaled out.
        let s = self.scale;
        let dev = [h[0], h[1], h[2] * s, h[3], h[4], h[5] * s, h[6] / s, h[7] / s, h[8]];
        let top = self.xforms.last().copied().unwrap_or(IDENTITY);
        self.xforms.push(compose(top, dev));
    }

    fn pop_transform(&mut self) {
        self.xforms.pop();
    }

    /// A layer of its own, in the box's space (so it turns with the box):
    /// what follows is drawn into it, with no clip yet, and moved so the
    /// layer's corner is its origin.
    fn push_effect(&mut self, rect: Rect, effect: &Effect, time: f32) {
        let k = self.scale;
        let pad = effect.overflow;
        let (x0, y0) = (((rect.x - pad) * k).floor(), ((rect.y - pad) * k).floor());
        let (x1, y1) = (((rect.x + rect.width + pad) * k).ceil(), ((rect.y + rect.height + pad) * k).ceil());
        let (w, h) = ((x1 - x0) as u32, (y1 - y0) as u32);
        let room = self.targets.len() <= MAX_LAYERS;
        if w == 0 || h == 0 || w > MAX_LAYER_SIDE || h > MAX_LAYER_SIDE || !room {
            self.open.push(None);
            return;
        }
        let target = self.targets.len();
        self.targets.push(Target { size: (w, h), ops: vec![] });
        let uniform = EffectUniform {
            origin: [x0, y0],
            box_origin: [rect.x, rect.y],
            size: [rect.width, rect.height],
            extent: [w as f32, h as f32],
            scale: k,
            time,
            unused: [0.0; 2],
            args: *effect.values(),
        };
        let clips = std::mem::take(&mut self.clips);
        let xforms = std::mem::replace(&mut self.xforms, vec![[1.0, 0.0, -x0, 0.0, 1.0, -y0, 0.0, 0.0, 1.0]]);
        self.open.push(Some(OpenLayer {
            target,
            clips,
            xforms,
            rect: [x0, y0, w as f32, h as f32],
            effect: EffectDraw { shader: effect.shader.clone(), blend: effect.blend, uniform, target },
        }));
        // Each pass knows the one it is drawn into by the stack of open ones.
        self.current = target;
    }

    /// Closes the layer and puts it back, through its effect, where it was.
    fn pop_effect(&mut self) {
        let Some(Some(open)) = self.open.pop() else { return };
        self.clips = open.clips;
        self.xforms = open.xforms;
        self.current = self.open.iter().rev().find_map(|o| o.as_ref().map(|o| o.target)).unwrap_or(0);
        self.done.push(open.target);
        let [x, y, w, h] = open.rect;
        let before = self.instances.len();
        self.push(Rect { x, y, width: w, height: h }, WHITE, 0.0, None);
        if self.instances.len() > before {
            self.instances.last_mut().expect("just pushed").params[2] = 8.0;
            self.effects.push(open.effect);
            self.retag(Op::Effect(before as u32, self.effects.len() - 1));
        }
    }
}

/// `r` (UI pixels) as a scissor in a `size` target at `scale`, rounded out
/// and cut to the target; `None` when nothing of it is in it.
fn device_rect(r: Rect, scale: f32, size: (u32, u32)) -> Option<[u32; 4]> {
    let x0 = (r.x * scale).floor().max(0.0) as u32;
    let y0 = (r.y * scale).floor().max(0.0) as u32;
    let x1 = ((r.x + r.width) * scale).ceil().max(0.0).min(size.0 as f32) as u32;
    let y1 = ((r.y + r.height) * scale).ceil().max(0.0).min(size.1 as f32) as u32;
    (x1 > x0 && y1 > y0).then_some([x0, y0, x1 - x0, y1 - y0])
}
