//! WGSL sources for the main mesh pass and the post-processing fullscreen pass.
//!
//! - [`SHADER_SOURCE`]: lit/unlit materials, shadows, fog, PBR, skinning
//! - [`POSTFX_SHADER`]: effect kinds 0–34 here (copy, FXAA, glitch, bloom,
//!   SSAO, …) and 40 up in `postfx_kinds.wgsl`, the three.js passes behind
//!   [`crate::postprocessing`]
//!
//! Post-fx kinds used from the composer in [`crate::postprocessing`]:
//! | Kind | Effect |
//! |------|--------|
//! | 5, 31 | Digital glitch (with / without snow) |
//! | 1 | FXAA |
//! | 3–4 | Dot screen, halftone |
//! | 11, 13, 14 | SSAO |
//! | 40– | see `postfx_kinds.wgsl` |

/// Post-fx shader. A single WGSL module with a fullscreen-triangle vertex
/// shader and a switch-driven fragment shader keyed by an `effect_kind`
/// uniform. Each effect path is implemented in this same module so the
/// pipeline can be reused across post-fx classes.
pub const POSTFX_SHADER: &str = concat!(r#"
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
struct PostFxUniforms {
    // x: effect kind (0=copy, 1=fxaa, 2=film, 3=dotscreen, 4=halftone, 5=glitch,
    // 6=bloom-threshold, 7=blur-h, 8=blur-v, 9=bloom-composite, 10=sobel,
    // 11=ssao, 12=ssr, 13=ssao-blur, 14=ssao-composite, 16=outline-edge,
    // 17=outline-overlay, 25=adaptive tone map, 26=bokeh, 31/32=glitch
    // variants, 33=luminosity, 34=adapt luminance; the rest at `fs_main`).
    // Every kind has one branch: a number two branches test runs the first.
    // y: time (seconds), z/w: reserved
    params: vec4<f32>,
    // Effect-specific parameters: x/y/z/w meaning varies per effect.
    params2: vec4<f32>,
    params3: vec4<f32>,
    /// Resolution (x: width, y: height) in pixels; z=1 writes linear (f16
    /// target); w: sRGB-view bits, see `postfx_flag`.
    resolution: vec4<f32>,
    /// SSAO: x=near, y=far, z=kernelRadius, w=kernelSize
    ssao0: vec4<f32>,
    inv_proj: mat4x4<f32>,
    proj: mat4x4<f32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var<uniform> u: PostFxUniforms;
@group(0) @binding(3) var depth_tex: texture_2d<f32>;
@group(0) @binding(4) var depth_sampler: sampler;
@group(0) @binding(5) var normal_tex: texture_2d<f32>;
// A float texture, not `texture_depth_2d`: `sample_postfx_depth` reads it
// through a plain (non-comparison) sampler, which GLSL cannot do with a depth
// sampler type — it once used `textureLoad`, which has no GLSL form for one
// either. Naga compiles
// both arms of that branch whether or not either is taken, so declaring this as
// depth made the whole module unusable on WebGL2. Depth arrives in `.r`.
@group(0) @binding(6) var hw_depth_tex: texture_2d<f32>;
// A uniform, not a storage buffer. It is thirty-two fixed vec4s — 512 bytes —
// so storage bought nothing, and WebGL2 has no storage buffers at all: one
// declaration of one is enough to stop the whole module compiling on a GL
// backend, used or not.
struct SsaoKernel { samples : array<vec4<f32>, 32> }
@group(0) @binding(7) var<uniform> ssao_kernel_buf : SsaoKernel;
@group(0) @binding(8) var noise_tex: texture_2d<f32>;
// Nearest, repeating. Also the sampler `sample_postfx_depth` reads
// `hw_depth_tex` through, which is what makes that read work on WebGL2.
@group(0) @binding(9) var noise_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> Out {
    // Fullscreen triangle: three vertices covering [-1, 3] so the triangle
    // fully covers the NDC clip box without needing index/vertex buffers.
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    // uv in texture space — (0, 0) at the top-left, the same origin as every
    // wgpu texture and as `renderer::fullscreen::VS` — so sampling the input
    // at `uv` writes it back the right way up, into a target or onto the
    // canvas alike. Effects that are ports of three.js shaders and do maths in
    // GL's bottom-left `vUv` (glitch bands, halftone grid, SSAO's NDC
    // reconstruction) take `gl_uv(uv)` for that and sample with `uv`.
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: Out;
    out.pos = vec4<f32>(p[vid], 0.0, 1.0);
    out.uv = uv[vid];
    return out;
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

// Texture-space uv (v = 0 at the top row) <-> GL `vUv` (v = 0 at the bottom).
// The map is its own inverse, so the one function goes either way.
fn gl_uv(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(uv.x, 1.0 - uv.y);
}

// Encode linear → sRGB. The main shader does this for non-sRGB framebuffer
// writes; we mirror it on post-fx outputs since the surface is non-sRGB on
// this system.
fn postfx_encode(c: vec3<f32>) -> vec3<f32> {
    return pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}

fn postfx_linear_out() -> bool {
    return u.resolution.z > 0.5;
}

fn postfx_write_rgb(c: vec3<f32>) -> vec3<f32> {
    if (postfx_linear_out()) {
        return c;
    }
    return postfx_encode(c);
}

fn postfx_passthrough_rgb(c: vec3<f32>) -> vec3<f32> {
    if (postfx_linear_out()) {
        return c;
    }
    // three.js DotScreen/FilmShader write clamped linear to the canvas (no shader gamma).
    return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn postfx_passthrough(c: vec3<f32>, a: f32) -> vec4<f32> {
    return vec4<f32>(postfx_passthrough_rgb(c), a);
}

fn postfx_write(c: vec3<f32>, a: f32) -> vec4<f32> {
    return vec4<f32>(postfx_write_rgb(c), a);
}

// `resolution.w` bits: how the views bound to this pass treat stored bytes.
// Every branch is written for a plain `Unorm` 8-bit target (the WebGPU canvas
// and its render targets): what it returns is the byte stored, and a "raw" read
// returns the stored byte. On an sRGB-format target (every 8-bit target and
// the canvas on WebGL2, which has neither a linear canvas format nor view
// formats) the hardware encodes on write and its "raw" views decode on read,
// so these bits undo exactly that and both backends store the same bytes.
//   1: the output is sRGB — decode the result, the hardware encodes it back;
//   2: the input's raw view decodes — re-encode what it returns;
//   4: likewise `depth_tex`; 8: likewise `normal_tex`.
fn postfx_flag(bit: u32) -> bool {
    return (u32(u.resolution.w) & bit) != 0u;
}

// The exact sRGB curves the hardware applies (not `postfx_encode`'s 2.2).
fn postfx_srgb_encode(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    return select(1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3<f32>(0.0031308));
}

fn postfx_srgb_decode(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    return select(pow((x + 0.055) / 1.055, vec3<f32>(2.4)), x / 12.92, x <= vec3<f32>(0.04045));
}

fn postfx_raw(c: vec4<f32>, bit: u32) -> vec4<f32> {
    if (postfx_flag(bit)) {
        return vec4<f32>(postfx_srgb_encode(c.rgb), c.a);
    }
    return c;
}

// A read of `input_tex` through the raw view kinds 0, 13, 15, 16, 18 and 19
// bind (bit 2 is only ever set for those).
fn input_raw(uv: vec2<f32>) -> vec4<f32> {
    return postfx_raw(textureSampleLevel(input_tex, input_sampler, uv, 0.0), 2u);
}

// three.js `luminance()` (common.glsl): Rec. 709 weights.
fn postfx_luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126729, 0.7151522, 0.0721750));
}

// three.js BokehShader's taps after the centre one: (x, y) in GL `vUv` units
// before the circle of confusion, z the ring's scale (1, 0.9, 0.7, 0.4).
const BOKEH_TAPS = array<vec3<f32>, 40>(
    vec3<f32>( 0.0,   0.4,  1.0), vec3<f32>( 0.15,  0.37, 1.0), vec3<f32>( 0.29,  0.29, 1.0), vec3<f32>(-0.37,  0.15, 1.0),
    vec3<f32>( 0.4,   0.0,  1.0), vec3<f32>( 0.37, -0.15, 1.0), vec3<f32>( 0.29, -0.29, 1.0), vec3<f32>(-0.15, -0.37, 1.0),
    vec3<f32>( 0.0,  -0.4,  1.0), vec3<f32>(-0.15,  0.37, 1.0), vec3<f32>(-0.29,  0.29, 1.0), vec3<f32>( 0.37,  0.15, 1.0),
    vec3<f32>(-0.4,   0.0,  1.0), vec3<f32>(-0.37, -0.15, 1.0), vec3<f32>(-0.29, -0.29, 1.0), vec3<f32>( 0.15, -0.37, 1.0),
    vec3<f32>( 0.15,  0.37, 0.9), vec3<f32>(-0.37,  0.15, 0.9), vec3<f32>( 0.37, -0.15, 0.9), vec3<f32>(-0.15, -0.37, 0.9),
    vec3<f32>(-0.15,  0.37, 0.9), vec3<f32>( 0.37,  0.15, 0.9), vec3<f32>(-0.37, -0.15, 0.9), vec3<f32>( 0.15, -0.37, 0.9),
    vec3<f32>( 0.29,  0.29, 0.7), vec3<f32>( 0.4,   0.0,  0.7), vec3<f32>( 0.29, -0.29, 0.7), vec3<f32>( 0.0,  -0.4,  0.7),
    vec3<f32>(-0.29,  0.29, 0.7), vec3<f32>(-0.4,   0.0,  0.7), vec3<f32>(-0.29, -0.29, 0.7), vec3<f32>( 0.0,   0.4,  0.7),
    vec3<f32>( 0.29,  0.29, 0.4), vec3<f32>( 0.4,   0.0,  0.4), vec3<f32>( 0.29, -0.29, 0.4), vec3<f32>( 0.0,  -0.4,  0.4),
    vec3<f32>(-0.29,  0.29, 0.4), vec3<f32>(-0.4,   0.0,  0.4), vec3<f32>(-0.29, -0.29, 0.4), vec3<f32>( 0.0,   0.4,  0.4),
);

fn ssao_kernel_sample(i: u32) -> vec3<f32> {
    return ssao_kernel_buf.samples[i].xyz;
}

fn sample_ssao_noise(uv: vec2<f32>, res: vec2<f32>) -> f32 {
    let noise_scale = res / vec2<f32>(4.0, 4.0);
    let scaled = fract(uv * noise_scale);
    let dim = textureDimensions(noise_tex, 0);
    let fc = scaled * vec2<f32>(dim);
    let coord = clamp(vec2<i32>(fc), vec2<i32>(0), vec2<i32>(i32(dim.x) - 1, i32(dim.y) - 1));
    return textureLoad(noise_tex, coord, 0).r;
}

// three.js perspectiveDepthToViewZ / viewZToOrthographicDepth (OpenGL view space).
fn perspective_depth_to_view_z(depth: f32, near: f32, far: f32) -> f32 {
    return (near * far) / ((far - near) * depth - far);
}

fn view_z_to_ortho_depth(view_z: f32, near: f32, far: f32) -> f32 {
    return (view_z + near) / (near - far);
}

fn linear_depth_from_depth_tex(depth: f32, near: f32, far: f32) -> f32 {
    let view_z = perspective_depth_to_view_z(depth, near, far);
    return view_z_to_ortho_depth(view_z, near, far);
}

// `uv` is GL `vUv` (bottom-left origin): NDC = uv * 2 - 1.
fn view_pos_from_depth(uv: vec2<f32>, depth: f32, view_z: f32, proj: mat4x4<f32>, inv_proj: mat4x4<f32>) -> vec3<f32> {
    let clip_w = proj[2].w * view_z + proj[3].w;
    var clip_pos = vec4<f32>((vec3<f32>(uv, depth) - 0.5) * 2.0, 1.0);
    clip_pos = clip_pos * clip_w;
    return (inv_proj * clip_pos).xyz;
}

// NDC -> GL `vUv` (bottom-left origin); `gl_uv` turns it into a texture fetch.
fn ndc_to_uv(ndc_xy: vec2<f32>) -> vec2<f32> {
    return ndc_xy * 0.5 + 0.5;
}

fn view_z_to_depth_tex(view_z_pos: f32, near: f32, far: f32) -> f32 {
    // Encode view-space Z (negative in front of camera) for perspectiveDepthToViewZ.
    let view_z = -view_z_pos;
    return (far + near * far / view_z) / (far - near);
}

fn unpack_view_normal(enc: vec3<f32>) -> vec3<f32> {
    return normalize(enc * 2.0 - 1.0);
}

fn sample_postfx_depth(uv: vec2<f32>, use_hw: bool) -> f32 {
    if (use_hw) {
        let dim = textureDimensions(hw_depth_tex, 0);
        let coord = clamp(vec2<i32>(uv * vec2<f32>(dim)), vec2<i32>(0), vec2<i32>(i32(dim.x) - 1, i32(dim.y) - 1));
        // Bit 16: the renderer drew with reversed depth; every effect below
        // is written for standard depth.
        let flip = postfx_flag(16u);
        // A nearest sample at the texel's centre — the same texel a
        // `textureLoad` would fetch — rather than the fetch itself. On WebGL2
        // a texture read without a sampler takes its filtering from the
        // texture object, whose default minification filter is
        // NEAREST_MIPMAP_LINEAR; GLES 3.0 calls a depth texture with a
        // non-NEAREST filter incomplete, and an incomplete texture fetches 0.
        // wgpu-hal resets the filter only for formats that are unfilterable
        // as colour, not for depth, so `texelFetch` read 0 everywhere and
        // SSAO occluded whatever it pleased. Pairing the texture with the
        // nearest `noise_sampler` makes wgpu bind that sampler object to the
        // unit, and its NEAREST filters make the texture complete.
        let centre = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dim);
        let d = textureSampleLevel(hw_depth_tex, noise_sampler, centre, 0.0).r;
        return select(d, 1.0 - d, flip);
    }
    return postfx_raw(textureSampleLevel(depth_tex, depth_sampler, uv, 0.0), 4u).r;
}

fn outline_gaussian_pdf(x: f32, sigma: f32) -> f32 {
    return 0.39894 * exp(-0.5 * x * x / (sigma * sigma)) / sigma;
}

// Halftone + Glitch helpers (three.js HalftoneShader / DigitalGlitch ports).
const HT_SQRT2_MINUS_ONE: f32 = 0.41421356;
const HT_SQRT2_HALF_MINUS_ONE: f32 = 0.20710678;
const HT_PI2: f32 = 6.28318531;
const HT_SAMPLES: i32 = 8;

struct HtCell {
    normal: vec2<f32>,
    p1: vec2<f32>,
    p2: vec2<f32>,
    p3: vec2<f32>,
    p4: vec2<f32>,
}

fn ht_mod(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

fn ht_blend(a: f32, b: f32, t: f32) -> f32 {
    return a * (1.0 - t) + b * t;
}

fn ht_hypot(x: f32, y: f32) -> f32 {
    return sqrt(x * x + y * y);
}

fn ht_rand(seed: vec2<f32>) -> f32 {
    return fract(sin(dot(seed, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn ht_distance_to_dot_radius(
    channel: f32, coord: vec2<f32>, normal: vec2<f32>, p: vec2<f32>,
    angle: f32, rad_max: f32, shape: i32,
) -> f32 {
    var dist = ht_hypot(coord.x - p.x, coord.y - p.y);
    var rad = channel;
    if (shape == 1) {
        rad = pow(abs(rad), 1.125) * rad_max;
    } else if (shape == 2) {
        rad = pow(abs(rad), 1.125) * rad_max;
        if (dist != 0.0) {
            let dot_p = abs((p.x - coord.x) / dist * normal.x + (p.y - coord.y) / dist * normal.y);
            dist = dist * (1.0 - HT_SQRT2_HALF_MINUS_ONE) + dot_p * dist * HT_SQRT2_MINUS_ONE;
        }
    } else if (shape == 3) {
        rad = pow(abs(rad), 1.5) * rad_max;
        let dot_p = (p.x - coord.x) * normal.x + (p.y - coord.y) * normal.y;
        dist = ht_hypot(normal.x * dot_p, normal.y * dot_p);
    } else if (shape == 4) {
        let theta = atan2(p.y - coord.y, p.x - coord.x) - angle;
        let sin_t = abs(sin(theta));
        let cos_t = abs(cos(theta));
        rad = pow(abs(rad), 1.4);
        rad = rad_max * (rad + select(rad - cos_t * rad, rad - sin_t * rad, sin_t > cos_t));
    }
    return rad - dist;
}

// `point` is in GL pixels (bottom-left origin), as in three.js; the input is
// sampled at the texture-space position of the same spot.
fn ht_get_sample_channel(point: vec2<f32>, res: vec2<f32>, channel: i32) -> f32 {
    let uv_s = gl_uv(point / res);
    var tex = textureSampleLevel(input_tex, input_sampler, uv_s, 0.0);
    let base = ht_rand(vec2<f32>(floor(point.x), floor(point.y))) * HT_PI2;
    let step_a = HT_PI2 / f32(HT_SAMPLES);
    let dist = u.params2.x * 0.66;
    for (var i: i32 = 0; i < HT_SAMPLES; i = i + 1) {
        let r = base + step_a * f32(i);
        let coord = point + vec2<f32>(cos(r) * dist, sin(r) * dist);
        tex = tex + textureSampleLevel(input_tex, input_sampler, gl_uv(coord / res), 0.0);
    }
    tex = tex / f32(HT_SAMPLES + 1);
    if (channel == 0) { return tex.r; }
    if (channel == 1) { return tex.g; }
    return tex.b;
}

fn ht_get_dot_colour(
    cell: HtCell, p: vec2<f32>, channel: i32, angle: f32, aa: f32,
    radius: f32, shape: i32, res: vec2<f32>,
) -> f32 {
    let samp1 = ht_get_sample_channel(cell.p1, res, channel);
    let samp2 = ht_get_sample_channel(cell.p2, res, channel);
    let samp3 = ht_get_sample_channel(cell.p3, res, channel);
    let samp4 = ht_get_sample_channel(cell.p4, res, channel);
    let dist_c_1 = ht_distance_to_dot_radius(samp1, cell.p1, cell.normal, p, angle, radius, shape);
    let dist_c_2 = ht_distance_to_dot_radius(samp2, cell.p2, cell.normal, p, angle, radius, shape);
    let dist_c_3 = ht_distance_to_dot_radius(samp3, cell.p3, cell.normal, p, angle, radius, shape);
    let dist_c_4 = ht_distance_to_dot_radius(samp4, cell.p4, cell.normal, p, angle, radius, shape);
    var res_c = 0.0;
    res_c = res_c + select(0.0, clamp(dist_c_1 / aa, 0.0, 1.0), dist_c_1 > 0.0);
    res_c = res_c + select(0.0, clamp(dist_c_2 / aa, 0.0, 1.0), dist_c_2 > 0.0);
    res_c = res_c + select(0.0, clamp(dist_c_3 / aa, 0.0, 1.0), dist_c_3 > 0.0);
    res_c = res_c + select(0.0, clamp(dist_c_4 / aa, 0.0, 1.0), dist_c_4 > 0.0);
    return clamp(res_c, 0.0, 1.0);
}

fn ht_get_reference_cell(p: vec2<f32>, origin: vec2<f32>, grid_angle: f32, step: f32, scatter: f32) -> HtCell {
    var cell: HtCell;
    let n = vec2<f32>(cos(grid_angle), sin(grid_angle));
    let threshold = step * 0.5;
    let dot_normal = n.x * (p.x - origin.x) + n.y * (p.y - origin.y);
    let dot_line = -n.y * (p.x - origin.x) + n.x * (p.y - origin.y);
    let offset = vec2<f32>(n.x * dot_normal, n.y * dot_normal);
    let offset_normal = ht_mod(ht_hypot(offset.x, offset.y), step);
    let normal_dir = select(-1.0, 1.0, dot_normal < 0.0);
    let normal_scale = select(step - offset_normal, -offset_normal, offset_normal < threshold) * normal_dir;
    let offset_line = ht_mod(
        ht_hypot((p.x - offset.x) - origin.x, (p.y - offset.y) - origin.y),
        step,
    );
    let line_dir = select(-1.0, 1.0, dot_line < 0.0);
    let line_scale = select(step - offset_line, -offset_line, offset_line < threshold) * line_dir;
    cell.normal = n;
    cell.p1 = vec2<f32>(
        p.x - n.x * normal_scale + n.y * line_scale,
        p.y - n.y * normal_scale - n.x * line_scale,
    );
    if (scatter != 0.0) {
        let off_mag = scatter * threshold * 0.5;
        let off_angle = ht_rand(vec2<f32>(floor(cell.p1.x), floor(cell.p1.y))) * HT_PI2;
        cell.p1 = cell.p1 + vec2<f32>(cos(off_angle) * off_mag, sin(off_angle) * off_mag);
    }
    let normal_step = normal_dir * select(-step, step, offset_normal < threshold);
    let line_step = line_dir * select(-step, step, offset_line < threshold);
    cell.p2 = cell.p1 - n * normal_step;
    cell.p3 = cell.p1 + vec2<f32>(n.y * line_step, -n.x * line_step);
    cell.p4 = cell.p1 - n * normal_step + vec2<f32>(n.y * line_step, -n.x * line_step);
    return cell;
}

fn ht_blend_colour(a: f32, b: f32, t: f32, mode: i32) -> f32 {
    if (mode == 2) { return ht_blend(a, max(0.0, a * b), t); }
    if (mode == 3) { return ht_blend(a, min(1.0, a + b), t); }
    if (mode == 4) { return ht_blend(a, max(a, b), t); }
    if (mode == 5) { return ht_blend(a, min(a, b), t); }
    return ht_blend(a, b, 1.0 - t);
}

// `uv` is GL `vUv` (bottom-left origin): the dot grid is laid out from there,
// as three.js's HalftoneShader does.
fn postfx_halftone(uv: vec2<f32>, res: vec2<f32>, base: vec4<f32>) -> vec4<f32> {
    if (u.params.w > 0.5) { return base; }
    let radius = u.params2.x;
    let scatter = u.params2.y;
    let shape = i32(u.params2.z + 0.5);
    let blending = u.params2.w;
    let rotate_r = u.params3.x;
    let rotate_g = u.params3.y;
    let rotate_b = u.params3.z;
    let blending_mode = i32(u.params3.w + 0.5);
    let greyscale = u.ssao0.x > 0.5;
    let p = uv * res;
    let origin = vec2<f32>(0.0, 0.0);
    let aa = select(1.25, radius * 0.5, radius < 2.5);
    let cell_r = ht_get_reference_cell(p, origin, rotate_r, radius, scatter);
    let cell_g = ht_get_reference_cell(p, origin, rotate_g, radius, scatter);
    let cell_b = ht_get_reference_cell(p, origin, rotate_b, radius, scatter);
    var r = ht_get_dot_colour(cell_r, p, 0, rotate_r, aa, radius, shape, res);
    var g = ht_get_dot_colour(cell_g, p, 1, rotate_g, aa, radius, shape, res);
    var b = ht_get_dot_colour(cell_b, p, 2, rotate_b, aa, radius, shape, res);
    r = ht_blend_colour(r, base.r, blending, blending_mode);
    g = ht_blend_colour(g, base.g, blending, blending_mode);
    b = ht_blend_colour(b, base.b, blending, blending_mode);
    if (greyscale) { r = (r + g + b) / 3.0; g = r; b = r; }
    return vec4<f32>(postfx_passthrough_rgb(vec3<f32>(r, g, b)), base.a);
}

fn glitch_rand(co: vec2<f32>) -> f32 {
    return fract(sin(dot(co, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn sample_input_linear(uv: vec2<f32>) -> vec4<f32> {
    let dim = textureDimensions(input_tex, 0);
    let st = uv * vec2<f32>(dim) - vec2<f32>(0.5);
    let i0 = floor(st);
    let f = st - i0;
    let max_c = vec2<i32>(i32(dim.x) - 1, i32(dim.y) - 1);
    let c00 = textureLoad(input_tex, clamp(vec2<i32>(i0), vec2<i32>(0), max_c), 0);
    let c10 = textureLoad(input_tex, clamp(vec2<i32>(i0 + vec2<f32>(1.0, 0.0)), vec2<i32>(0), max_c), 0);
    let c01 = textureLoad(input_tex, clamp(vec2<i32>(i0 + vec2<f32>(0.0, 1.0)), vec2<i32>(0), max_c), 0);
    let c11 = textureLoad(input_tex, clamp(vec2<i32>(i0 + vec2<f32>(1.0, 1.0)), vec2<i32>(0), max_c), 0);
    let c0 = mix(c00, c10, f.x);
    let c1 = mix(c01, c11, f.x);
    return mix(c0, c1, f.y);
}

fn sample_input_linear_f16(uv: vec2<f32>) -> vec4<f32> {
    let dim = textureDimensions(input_tex, 0);
    let st = uv * vec2<f32>(dim) - vec2<f32>(0.5);
    let i0 = floor(st);
    let f = st - i0;
    let max_c = vec2<i32>(i32(dim.x) - 1, i32(dim.y) - 1);
    let c00 = f16_round_rgba(textureLoad(input_tex, clamp(vec2<i32>(i0), vec2<i32>(0), max_c), 0));
    let c10 = f16_round_rgba(textureLoad(input_tex, clamp(vec2<i32>(i0 + vec2<f32>(1.0, 0.0)), vec2<i32>(0), max_c), 0));
    let c01 = f16_round_rgba(textureLoad(input_tex, clamp(vec2<i32>(i0 + vec2<f32>(0.0, 1.0)), vec2<i32>(0), max_c), 0));
    let c11 = f16_round_rgba(textureLoad(input_tex, clamp(vec2<i32>(i0 + vec2<f32>(1.0, 1.0)), vec2<i32>(0), max_c), 0));
    let c0 = mix(c00, c10, f.x);
    let c1 = mix(c01, c11, f.x);
    return f16_round_rgba(mix(c0, c1, f.y));
}

// three.js's `DataTexture` defaults to nearest filtering, and GlitchPass's
// heightmap keeps it: the texel under `uv`, clamped to the edge.
fn sample_glitch_disp(uv: vec2<f32>) -> f32 {
    let dim = textureDimensions(noise_tex, 0);
    let max_c = vec2<i32>(i32(dim.x) - 1, i32(dim.y) - 1);
    let c = clamp(vec2<i32>(floor(uv * vec2<f32>(dim))), vec2<i32>(0), max_c);
    return textureLoad(noise_tex, c, 0).r;
}

fn postfx_encode_unclamped(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

fn glitch_write(c: vec3<f32>) -> vec3<f32> {
    if (postfx_linear_out()) {
        return max(c, vec3<f32>(0.0));
    }
    return postfx_encode_unclamped(c);
}

// `uv` is GL `vUv` (bottom-left origin), so the bands, the displacement map
// and the snow sit where three.js's DigitalGlitch puts them; the input is
// sampled at the texture-space position of each displaced point.
fn postfx_glitch(uv: vec2<f32>, res: vec2<f32>, base: vec4<f32>, fc: vec2<f32>) -> vec4<f32> {
    if (u.params.z >= 1.0) {
        return postfx_glitch_out(base);
    }
    var p = uv;
    let seed = u.params.y;
    let amount = u.params2.x;
    let angle = u.params2.y;
    let distortion_x = u.params2.z;
    let distortion_y = u.params2.w;
    let seed_x = u.params3.x;
    let seed_y = u.params3.y;
    let col_s = u.params3.z;
    let disp = sample_glitch_disp(p * seed * seed);
    if (p.y < distortion_x + col_s && p.y > distortion_x - col_s * seed) {
        if (seed_x > 0.0) { p.y = 1.0 - (p.y + distortion_y); }
        else { p.y = distortion_y; }
    }
    if (p.x < distortion_y + col_s && p.x > distortion_y - col_s * seed) {
        if (seed_y > 0.0) { p.x = distortion_x; }
        else { p.x = 1.0 - (p.x + distortion_x); }
    }
    p.x = p.x + disp * seed_x * (seed / 5.0);
    p.y = p.y + disp * seed_y * (seed / 5.0);
    let offset = amount * vec2<f32>(cos(angle), sin(angle));
    let cr = sample_input_linear_f16(gl_uv(p + offset)).r;
    let cga = sample_input_linear_f16(gl_uv(p));
    let cb = sample_input_linear_f16(gl_uv(p - offset)).b;
    var col = vec4<f32>(cr, cga.g, cb, cga.a);
    if (col_s >= 0.0 && u32(u.params.x) == 5u) {
        // three.js hashes `gl_FragCoord`: pixel centres, GL's bottom-left
        // origin, so xs = 2i + 1 and ys = 2j + 1 with j counted up from the
        // bottom row. `fc` is the WGSL frag coord — the same centres, counted
        // down from the top — so y is flipped against the target's height.
        // Taken from `fc` rather than from the interpolated uv because the
        // floor of a centre computed as `uv * res` can land on 2i instead of
        // 2i + 1. It was `(1 - v) * (res - 1) + 0.5`: mirrored, and off the
        // centres by up to half a pixel.
        let xs = floor(fc.x / 0.5);
        let ys = floor((res.y - fc.y) / 0.5);
        let snow = 200.0 * amount * glitch_rand(vec2<f32>(xs * seed, ys * seed * 50.0)) * 0.2;
        col = col + vec4<f32>(snow, snow, snow, snow);
    }
    return postfx_glitch_out(col);
}

fn postfx_glitch_out(col: vec4<f32>) -> vec4<f32> {
    return postfx_passthrough(col.rgb, col.a);
}

// Round through IEEE f16 — matches WebGL HalfFloat RT read precision.
fn f16_round(x: f32) -> f32 {
    return unpack2x16float(pack2x16float(vec2<f32>(x, 0.0))).x;
}

fn f16_round_rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(f16_round(c.r), f16_round(c.g), f16_round(c.b));
}

fn f16_round_rgba(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(f16_round_rgb(c.rgb), f16_round(c.a));
}

fn fxaa_contrast(a: vec4<f32>, b: vec4<f32>) -> f32 {
    let diff = abs(a - b);
    return max(max(max(diff.r, diff.g), diff.b), diff.a);
}

// three.js FXAAShader's `FxaaTexTop`: a sample at `p`, which is GL `vUv`
// (+y up) as the whole port is — its neighbour offsets, the edge walk and the
// tie-breaks all assume that orientation, so the image is not mirrored
// relative to three.js's.
// The stored bytes (bit 2: the input view decodes, so re-encode).
fn fxaa_top(p: vec2<f32>) -> vec4<f32> {
    return postfx_raw(textureSampleLevel(input_tex, input_sampler, gl_uv(p), 0.0), 2u);
}

// FXAA 3.11 as three.js r165's FXAAShader has it, line for line. `pos_m` is
// GL `vUv`; `rcp` is the `resolution` uniform (texel size).
fn fxaa_pixel_shader(pos_m: vec2<f32>, rcp: vec2<f32>) -> vec4<f32> {
    let edge_threshold: f32 = 0.2;
    let inv_edge_threshold: f32 = 1.0 / edge_threshold;
    let rgba_m = fxaa_top(pos_m);
    let rgba_s = fxaa_top(pos_m + vec2<f32>(0.0, 1.0) * rcp);
    let rgba_e = fxaa_top(pos_m + vec2<f32>(1.0, 0.0) * rcp);
    let rgba_n0 = fxaa_top(pos_m + vec2<f32>(0.0, -1.0) * rcp);
    let rgba_w = fxaa_top(pos_m + vec2<f32>(-1.0, 0.0) * rcp);

    let contrast_n = fxaa_contrast(rgba_m, rgba_n0);
    let contrast_s = fxaa_contrast(rgba_m, rgba_s);
    let contrast_e = fxaa_contrast(rgba_m, rgba_e);
    let contrast_w = fxaa_contrast(rgba_m, rgba_w);
    if (max(max(max(contrast_n, contrast_s), contrast_e), contrast_w) < edge_threshold) {
        return rgba_m;
    }

    var relative_v = ((contrast_n + contrast_s) - (contrast_e + contrast_w)) * inv_edge_threshold;
    var horz_span = relative_v > 0.0;

    // 45 degree edges and corners: V and H contrast too alike to choose.
    if (abs(relative_v) < 0.3) {
        let dir_x = select(-1.0, 1.0, contrast_e > contrast_w);
        let dir_y = select(-1.0, 1.0, contrast_s > contrast_n);
        let rgba_along_h = fxaa_top(pos_m + vec2<f32>(dir_x, -dir_y) * rcp);
        let match_along_h = fxaa_contrast(rgba_m, rgba_along_h);
        let rgba_along_v = fxaa_top(pos_m + vec2<f32>(-dir_x, dir_y) * rcp);
        let match_along_v = fxaa_contrast(rgba_m, rgba_along_v);
        relative_v = (match_along_v - match_along_h) * inv_edge_threshold;
        if (abs(relative_v) < 0.3) {
            return mix(rgba_m, (rgba_n0 + rgba_s + rgba_e + rgba_w) * 0.25, 0.4);
        }
        horz_span = relative_v > 0.0;
    }

    var rgba_n = rgba_n0;
    var rgba_s2 = rgba_s;
    if (!horz_span) {
        rgba_n = rgba_w;
        rgba_s2 = rgba_e;
    }
    if (!(fxaa_contrast(rgba_m, rgba_n) > fxaa_contrast(rgba_m, rgba_s2))) {
        rgba_n = rgba_s2;
    }
    let off_np = vec2<f32>(select(0.0, rcp.x, horz_span), select(rcp.y, 0.0, horz_span));

    var done_n = false;
    var done_p = false;
    var n_dist = 0.0;
    var p_dist = 0.0;
    var iterations_n = 0;
    var iterations_p = 0;
    for (var i: i32 = 0; i < 5; i = i + 1) {
        let increment = f32(i + 1);
        if (!done_n) {
            n_dist = n_dist + increment;
            let end_n = fxaa_top(pos_m + off_np * n_dist);
            done_n = fxaa_contrast(end_n, rgba_m) > fxaa_contrast(end_n, rgba_n);
            iterations_n = i;
        }
        if (!done_p) {
            p_dist = p_dist + increment;
            let end_p = fxaa_top(pos_m - off_np * p_dist);
            done_p = fxaa_contrast(end_p, rgba_m) > fxaa_contrast(end_p, rgba_n);
            iterations_p = i;
        }
        if (done_n || done_p) {
            break;
        }
    }
    if (!done_p && !done_n) {
        return rgba_m;
    }
    var dist = min(
        select(1.0, f32(iterations_n) / 4.0, done_n),
        select(1.0, f32(iterations_p) / 4.0, done_p),
    );
    dist = 1.0 - pow(dist, 0.5);
    return mix(rgba_m, rgba_n, dist * 0.5);
}

// The entry point: every branch below, then the output decode `postfx_flag`
// describes, so an sRGB target stores the bytes a `Unorm` one would.
@fragment
fn fs_main(in: Out) -> @location(0) vec4<f32> {
    let c = postfx_fragment(in);
    if (postfx_flag(1u)) {
        return vec4<f32>(postfx_srgb_decode(c.rgb), c.a);
    }
    return c;
}

fn postfx_fragment(in: Out) -> vec4<f32> {
    let kind = u32(u.params.x);
    let res = u.resolution.xy;
    let texel = 1.0 / max(res, vec2<f32>(1.0));
    let uv = in.uv;
    let base = postfx_raw(textureSampleLevel(input_tex, input_sampler, uv, 0.0), 2u);

    // 0: copy / pass-through — point sample so RT pixels map 1:1 to canvas
    // (linear filtering bleeds foreground into background at silhouettes).
    if (kind == 0u) {
        let dim = textureDimensions(input_tex, 0);
        let fc = clamp(
            vec2<i32>(uv * vec2<f32>(dim)),
            vec2<i32>(0),
            vec2<i32>(i32(dim.x) - 1, i32(dim.y) - 1),
        );
        let px = postfx_raw(textureLoad(input_tex, fc, 0), 2u);
        if (postfx_linear_out()) {
            return vec4<f32>(px.rgb, px.a);
        }
        // params2.y: legacy sRGB-decode + raw passthrough (OutlinePass / SSAOPass scene copy).
        if (u.params2.y > 0.5) {
            return vec4<f32>(px.rgb, px.a);
        }
        // params2.z: raw 8-bit texel copy (RT→RT / RT→canvas without gamma round-trip).
        if (u.params2.z > 0.5) {
            return vec4<f32>(px.rgb, px.a);
        }
        return postfx_write(px.rgb, px.a);
    }
    // 20: bilinear resample (outline mask half-res downsample).
    if (kind == 20u) {
        let px = textureSampleLevel(input_tex, input_sampler, uv, 0.0);
        return postfx_write(px.rgb, px.a);
    }
    // 1: FXAA (three.js FXAAShader / Sturk port).
    if (kind == 1u) {
        var rcp = vec2<f32>(u.params2.x, u.params2.y);
        if (rcp.x <= 0.0 || rcp.y <= 0.0) {
            rcp = vec2<f32>(1.0 / 1024.0, 1.0 / 512.0);
        }
        let col = fxaa_pixel_shader(gl_uv(uv), rcp);
        return postfx_passthrough(col.rgb, col.a);
    }
    if (kind == 2u) {
        let t = u.params.y;
        let intensity = u.params2.x;
        let grayscale = u.params2.y > 0.5;
        // three.js FilmShader hashes `vUv`: GL's bottom-left origin.
        let noise = fract(sin(dot(fract(gl_uv(uv) + vec2<f32>(t)), vec2<f32>(12.9898, 78.233))) * 43758.5453);
        var col = base.rgb + base.rgb * clamp(0.1 + noise, 0.0, 1.0);
        col = mix(base.rgb, col, intensity);
        if (grayscale) {
            let lum = dot(col, vec3<f32>(0.2126729, 0.7151522, 0.0721750));
            col = vec3<f32>(lum);
        }
        return postfx_passthrough(col, base.a);
    }
    // 3: Dot screen (three.js DotScreenShader). The pattern is laid out in GL
    // `vUv`; the input is read at the pixel's own texture-space position.
    // params2: angle, scale, center; params3.xy: `tSize`, the input's size
    // (the output's when zero).
    if (kind == 3u) {
        let col = sample_input_linear_f16(uv);
        let guv = gl_uv(uv);
        let angle = u.params2.x;
        let scale = u.params2.y;
        let center = vec2<f32>(u.params2.z, u.params2.w);
        let t_size = select(res, u.params3.xy, u.params3.x > 0.0);
        let s = sin(angle);
        let c = cos(angle);
        let tex = guv * t_size - center;
        let point = vec2<f32>(c * tex.x - s * tex.y, s * tex.x + c * tex.y) * scale;
        let pattern = sin(point.x) * sin(point.y) * 4.0;
        let average = (col.r + col.g + col.b) / 3.0;
        let dot_val = average * 10.0 - 5.0 + pattern;
        return postfx_passthrough(vec3<f32>(dot_val), col.a);
    }
    // 4: RGB Halftone (three.js HalftoneShader). Its `width` / `height`
    // uniforms ride in params.yz when set; the output's size otherwise.
    if (kind == 4u) {
        let ht_res = select(res, vec2<f32>(u.params.y, u.params.z), u.params.y > 0.0);
        return postfx_halftone(gl_uv(uv), ht_res, base);
    }
    // 5: DigitalGlitch with snow; 31: the same without it. (31 was 25, which
    // was already Reinhard tone mapping: this branch came first, so that kind
    // ran glitch.)
    if (kind == 5u || kind == 31u) {
        return postfx_glitch(gl_uv(uv), res, base, in.pos.xy);
    }
    // 6: Bloom brightness threshold. Pass-only pixels brighter than params2.x.
    // params2.y: smooth-knee width (0 = hard cutoff).
    if (kind == 6u) {
        let threshold = u.params2.x;
        let knee = u.params2.y;
        let l = luma(base.rgb);
        let factor = smoothstep(threshold, threshold + knee + 0.0001, l);
        // Note: NO sRGB encode — bloom intermediates stay linear.
        return vec4<f32>(base.rgb * factor, base.a);
    }
    // 7: Gaussian blur horizontal. 9-tap separable kernel. params2.x = radius (px).
    if (kind == 7u) {
        let radius = u.params2.x;
        // `var` (not `let`): WGSL/naga only allows dynamic indexing (`weights[i]`
        // with a non-const `i`) on an addressable variable, not a value array.
        var weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
        var acc = base.rgb * weights[0];
        for (var i: i32 = 1; i < 5; i = i + 1) {
            let off = vec2<f32>(texel.x * radius * f32(i), 0.0);
            acc = acc + textureSampleLevel(input_tex, input_sampler, uv + off, 0.0).rgb * weights[i];
            acc = acc + textureSampleLevel(input_tex, input_sampler, uv - off, 0.0).rgb * weights[i];
        }
        return vec4<f32>(acc, base.a);
    }
    // 8: Gaussian blur vertical. Companion to kind 7.
    if (kind == 8u) {
        let radius = u.params2.x;
        // `var` (not `let`): WGSL/naga only allows dynamic indexing (`weights[i]`
        // with a non-const `i`) on an addressable variable, not a value array.
        var weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
        var acc = base.rgb * weights[0];
        for (var i: i32 = 1; i < 5; i = i + 1) {
            let off = vec2<f32>(0.0, texel.y * radius * f32(i));
            acc = acc + textureSampleLevel(input_tex, input_sampler, uv + off, 0.0).rgb * weights[i];
            acc = acc + textureSampleLevel(input_tex, input_sampler, uv - off, 0.0).rgb * weights[i];
        }
        return vec4<f32>(acc, base.a);
    }
    // 9: Bloom composite. params2.x = strength. Adds bloom (in this pass's
    // base) to the original scene (sampled from a SECOND input bound at... we
    // don't have a second input slot, so the JS layer composites via an
    // additive blend or by passing the original alongside. For now, treat the
    // single bound input as already containing the bloom, and `params2.y`
    // selects "additive over output" behavior — see composite_with_scene.
    if (kind == 9u) {
        let strength = u.params2.x;
        return vec4<f32>(postfx_encode(base.rgb * strength), base.a);
    }
    // 10: Sobel edge detect on the input (luminance-based, so it picks up
    // edges regardless of the underlying scene color). params2.x = edge color
    // R, .y = G, .z = B, .w = strength. Output stays linear so the JS layer
    // can additively composite on the rendered scene.
    if (kind == 10u) {
        let edge_color = vec3<f32>(u.params2.x, u.params2.y, u.params2.z);
        let strength = u.params2.w;
        let lw = vec3<f32>(0.299, 0.587, 0.114);
        let tl = dot(textureSampleLevel(input_tex, input_sampler, uv + vec2<f32>(-texel.x, -texel.y), 0.0).rgb, lw);
        let tm = dot(textureSampleLevel(input_tex, input_sampler, uv + vec2<f32>( 0.0,     -texel.y), 0.0).rgb, lw);
        let tr = dot(textureSampleLevel(input_tex, input_sampler, uv + vec2<f32>( texel.x, -texel.y), 0.0).rgb, lw);
        let ml = dot(textureSampleLevel(input_tex, input_sampler, uv + vec2<f32>(-texel.x,  0.0    ), 0.0).rgb, lw);
        let mr = dot(textureSampleLevel(input_tex, input_sampler, uv + vec2<f32>( texel.x,  0.0    ), 0.0).rgb, lw);
        let bl = dot(textureSampleLevel(input_tex, input_sampler, uv + vec2<f32>(-texel.x,  texel.y), 0.0).rgb, lw);
        let bm = dot(textureSampleLevel(input_tex, input_sampler, uv + vec2<f32>( 0.0,      texel.y), 0.0).rgb, lw);
        let br = dot(textureSampleLevel(input_tex, input_sampler, uv + vec2<f32>( texel.x,  texel.y), 0.0).rgb, lw);
        let gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
        let gy = -tl - 2.0*tm - tr + bl + 2.0*bm + br;
        let mag = clamp(sqrt(gx*gx + gy*gy) * strength, 0.0, 1.0);
        return vec4<f32>(edge_color * mag, mag);
    }
    // 16: Outline edge detect — four-tap cross diff on mask .r.
    if (kind == 16u) {
        let edge_color = vec3<f32>(u.params2.x, u.params2.y, u.params2.z);
        let hx = vec2<f32>(texel.x, 0.0);
        let hy = vec2<f32>(0.0, texel.y);
        let diff1 = (
            input_raw(uv + hx).r
            - input_raw(uv - hx).r
        ) * 0.5;
        let diff2 = (
            input_raw(uv + hy).r
            - input_raw(uv - hy).r
        ) * 0.5;
        let d = length(vec2<f32>(diff1, diff2));
        return postfx_write(edge_color * d, d);
    }
    // 17: Outline overlay — edgeStrength * mask.r * edge (three.js AdditiveBlending).
    if (kind == 17u) {
        let edge = textureSampleLevel(input_tex, input_sampler, uv, 0.0);
        let mask_r = textureSampleLevel(normal_tex, depth_sampler, uv, 0.0).r;
        let strength = u.params2.x;
        return edge * strength * mask_r;
    }
    // 18/19: Outline separable blur — matches three.js OutlinePass gaussian blur.
    // params2.x = kernelRadius (edgeThickness). 18 = horizontal, 19 = vertical.
    if (kind == 18u || kind == 19u) {
        let kernel_radius = max(u.params2.x, 0.001);
        let sigma = kernel_radius * 0.5;
        let inv_size = texel;
        let direction = select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), kind == 18u);
        var weight_sum = outline_gaussian_pdf(0.0, sigma);
        var acc = base * weight_sum;
        let delta = direction * inv_size * kernel_radius / 4.0;
        var uv_offset = delta;
        for (var i: i32 = 1; i <= 4; i = i + 1) {
            let x = kernel_radius * f32(i) / 4.0;
            let w = outline_gaussian_pdf(x, sigma);
            acc = acc + (input_raw(uv + uv_offset)
                + input_raw(uv - uv_offset)) * w;
            weight_sum = weight_sum + 2.0 * w;
            uv_offset = uv_offset + delta;
        }
        let out = acc / weight_sum;
        return postfx_write(out.rgb, out.a);
    }
    // 11: SSAO — view-space hemisphere oriented by normal prepass +
    // depth prepass. params2.x=minDistance, .y=maxDistance. Outputs AO factor
    // in rgb (1 = no occlusion), matching three.js's SSAO pass output.
    if (kind == 11u) {
        let near = u.ssao0.x;
        let far = u.ssao0.y;
        let kernel_radius = u.ssao0.z;
        let kernel_size = u32(u.ssao0.w);
        let min_dist = u.params2.x;
        let max_dist = u.params2.y;
        let use_hw = u.params.w > 0.5;
        let depth_r = sample_postfx_depth(uv, use_hw);
        // three.js r165 SSAOShader: `depth == 1.0`, the cleared background.
        if (depth_r >= 1.0) {
            return vec4<f32>(1.0, 1.0, 1.0, 1.0);
        }
        let view_z = perspective_depth_to_view_z(depth_r, near, far);
        // Reconstruction and projection work in GL `vUv` (NDC y up), as
        // three.js's SSAOShader does; every fetch is at the texture-space
        // position of the same point.
        let view_pos = view_pos_from_depth(gl_uv(uv), depth_r, view_z, u.proj, u.inv_proj);
        let view_normal = unpack_view_normal(postfx_raw(textureSampleLevel(normal_tex, depth_sampler, uv, 0.0), 8u).rgb);
        let noise_r = sample_ssao_noise(gl_uv(uv), res);
        // Scalar broadcast, matching three.js SSAOShader's
        // `vec3 random = vec3( getRandom( vUv * noiseScale ) )`. It is a weak
        // rotation — the same axis up to sign rather than a real turn — and it
        // degenerates to normalize(vec3(0)) for a normal along (1,1,1). Both
        // faults are three.js's; this pass is a port and the parity scenes in
        // tests/parity compare against it, so they stay.
        let random = vec3<f32>(noise_r, noise_r, noise_r);
        let tangent = normalize(random - view_normal * dot(random, view_normal));
        let bitangent = cross(view_normal, tangent);
        let kernel_matrix = mat3x3<f32>(tangent, bitangent, view_normal);
        var occlusion = 0.0;
        let ks = min(max(kernel_size, 1u), 32u);
        for (var i: u32 = 0u; i < 32u; i = i + 1u) {
            if (i >= ks) { break; }
            let sample_vec = kernel_matrix * ssao_kernel_sample(i);
            let sample_point = view_pos + sample_vec * kernel_radius;
            let sample_ndc = u.proj * vec4<f32>(sample_point, 1.0);
            let sample_uv = ndc_to_uv(sample_ndc.xy / max(sample_ndc.w, 0.0001));
            // Off-screen samples are not skipped: three.js reads its depth
            // texture there, clamped to the edge, and counts them like any
            // other. Clamped here too, the same texel either way.
            let sample_depth_r = sample_postfx_depth(gl_uv(sample_uv), use_hw);
            let real_depth = linear_depth_from_depth_tex(sample_depth_r, near, far);
            let sample_depth = view_z_to_ortho_depth(sample_point.z, near, far);
            let delta = sample_depth - real_depth;
            if (delta > min_dist && delta < max_dist) {
                occlusion = occlusion + 1.0;
            }
        }
        occlusion = clamp(occlusion / f32(ks), 0.0, 1.0);
        let ao = vec3<f32>(1.0 - occlusion);
        return vec4<f32>(ao, 1.0);
    }
    // 12: SSR composite (screen-space ray march on depth). params2.x = step size (px),
    // .y = thickness, .z = max opacity, .w = stride scale.
    if (kind == 12u) {
        let color = base.rgb;
        let center = postfx_raw(textureSampleLevel(depth_tex, depth_sampler, uv, 0.0), 4u).r;
        var reflect_color = color;
        var strength = 0.0;
        let step_px = max(u.params2.x, 0.5);
        let thickness = u.params2.y;
        let max_opacity = u.params2.z;
        let stride = max(u.params2.w, 1.0);
        // Reflect UV around the horizontal midline — good enough for floor reflections.
        for (var i: i32 = 1; i <= 24; i = i + 1) {
            let t = f32(i) / 24.0;
            let sample_uv = uv + vec2<f32>(0.0, t * step_px * stride * texel.y * 24.0);
            if (sample_uv.y < 0.0 || sample_uv.y > 1.0) { break; }
            let mirror_uv = vec2<f32>(sample_uv.x, 1.0 - sample_uv.y);
            let sample_d = postfx_raw(textureSampleLevel(depth_tex, depth_sampler, mirror_uv, 0.0), 4u).r;
            if (sample_d > center + thickness) {
                reflect_color = textureSampleLevel(input_tex, input_sampler, mirror_uv, 0.0).rgb;
                strength = (1.0 - t) * max_opacity;
                break;
            }
        }
        let out = mix(color, reflect_color, strength);
        return vec4<f32>(postfx_encode(out), base.a);
    }
    // 13: SSAO blur — 5×5 box filter on the AO factor (stored in input .r).
    if (kind == 13u) {
        var result = 0.0;
        for (var i: i32 = -2; i <= 2; i = i + 1) {
            for (var j: i32 = -2; j <= 2; j = j + 1) {
                let off = vec2<f32>(f32(i), f32(j)) * texel;
                result = result + input_raw(uv + off).r;
            }
        }
        let blurred = result / 25.0;
        return vec4<f32>(blurred, blurred, blurred, 1.0);
    }
    // 14: SSAO composite (single-pass scene * AO, the AO bound as depth).
    // params2.y: three.js SSAOPass's OUTPUT.Default — its beauty copy (kind
    // 0's legacy decode-and-pass-through) times the blurred AO, which three.js
    // draws as a second copy with DstColor × SrcColor blending. One pass here
    // because a second draw onto a WebGL2 canvas does not see the first: the
    // surface texture comes back uninitialised and its load is a clear.
    if (kind == 14u) {
        let ao = postfx_raw(textureSampleLevel(depth_tex, depth_sampler, uv, 0.0), 4u).r;
        if (u.params2.y > 0.5) {
            return vec4<f32>(base.rgb * ao, base.a);
        }
        return postfx_write(base.rgb * ao, base.a);
    }
    // 15: SSAO multiply onto loaded canvas (three.js CustomBlending DstColor*SrcColor).
    if (kind == 15u) {
        let ao = base.r;
        return vec4<f32>(ao, ao, ao, 1.0);
    }
    // 21: Afterimage — mix current input with previous frame bound on normal_tex.
    if (kind == 21u) {
        let damp = u.params2.x;
        let prev = textureSampleLevel(normal_tex, depth_sampler, uv, 0.0);
        let mixed = mix(prev.rgb, base.rgb, damp);
        return postfx_write(mixed, base.a);
    }
    // 22: Transition — mix input (A) with normal_tex (B). params2.x = mix factor.
    if (kind == 22u) {
        let t = clamp(u.params2.x, 0.0, 1.0);
        let b = textureSampleLevel(normal_tex, depth_sampler, uv, 0.0);
        return postfx_write(mix(base.rgb, b.rgb, t), mix(base.a, b.a, t));
    }
    // 23: 3D LUT color grade sampled from normal_tex atlas. params2.x = lut edge length.
    if (kind == 23u) {
        let lut_size = max(u.params2.x, 2.0);
        let c = clamp(base.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        let slice = c.b * (lut_size - 1.0);
        let slice0 = floor(slice);
        let slice1 = min(slice0 + 1.0, lut_size - 1.0);
        let t = slice - slice0;
        let uv0 = (vec2<f32>(c.r, c.g) * (lut_size - 1.0) + vec2<f32>(slice0, 0.0)) / vec2<f32>(lut_size, lut_size);
        let uv1 = (vec2<f32>(c.r, c.g) * (lut_size - 1.0) + vec2<f32>(slice1, 0.0)) / vec2<f32>(lut_size, lut_size);
        let c0 = textureSampleLevel(normal_tex, depth_sampler, uv0, 0.0).rgb;
        let c1 = textureSampleLevel(normal_tex, depth_sampler, uv1, 0.0).rgb;
        return postfx_write(mix(c0, c1, t), base.a);
    }
    // 24: Pixelated — block nearest-neighbor. params2.x = block size in pixels.
    if (kind == 24u) {
        let block = max(u.params2.x, 1.0);
        let dim = textureDimensions(input_tex, 0);
        let px = floor(uv * vec2<f32>(dim) / block) * block + vec2<f32>(block * 0.5);
        let fc = clamp(vec2<i32>(px), vec2<i32>(0), vec2<i32>(i32(dim.x) - 1, i32(dim.y) - 1));
        let samp = textureLoad(input_tex, fc, 0);
        return vec4<f32>(samp.rgb, samp.a);
    }
    // 25: Adaptive tone mapping — ToneMapShader, the operator
    // AdaptiveToneMappingPass ends on (Reinhard's modified operator, Eq. 4).
    // Both were removed from three.js in r153; this follows r152's source.
    // params2: x = middleGrey, y = minLuminance, z = maxLuminance,
    // w = averageLuminance (used when not adaptive). params3.x > 0.5: adaptive,
    // the average luminance read at (0.5, 0.5) of the target bound as
    // `normal_tex` (kinds 33, 20 and 34 build it). As in three.js it scales the colour by
    // the compressed luminance rather than by its ratio to the pixel's own.
    if (kind == 25u) {
        let middle_grey = u.params2.x;
        let min_lum = u.params2.y;
        let max_lum = u.params2.z;
        var lum_avg = u.params2.w;
        if (u.params3.x > 0.5) {
            lum_avg = textureSampleLevel(normal_tex, depth_sampler, vec2<f32>(0.5, 0.5), 0.0).r;
        }
        let lum_pixel = postfx_luminance(base.rgb);
        let lum_scaled = (lum_pixel * middle_grey) / max(min_lum, lum_avg);
        let lum_compressed = (lum_scaled * (1.0 + (lum_scaled / (max_lum * max_lum)))) / (1.0 + lum_scaled);
        return postfx_passthrough(lum_compressed * base.rgb, base.a);
    }
    // 26: Bokeh — three.js r165 BokehShader, BokehPass's: 41 taps on rings
    // scaled by the circle of confusion `(focus + viewZ) * aperture`, clamped
    // to ±maxblur (uv units). params2: x = focus, y = aperture, z = maxblur,
    // w = camera aspect; near/far in `ssao0.xy`. Depth is the hardware depth of
    // the target bound as both depth and normal (as SSAO's): the same
    // `gl_FragCoord.z` three.js packs with MeshDepthMaterial. Offsets are in GL
    // `vUv` (y up), so they are mirrored in v for the texture-space fetch.
    if (kind == 26u) {
        let focus = u.params2.x;
        let aperture = u.params2.y;
        let max_blur = u.params2.z;
        let aspect_correct = vec2<f32>(1.0, u.params2.w);
        let depth = sample_postfx_depth(uv, u.params.w > 0.5);
        let view_z = perspective_depth_to_view_z(depth, u.ssao0.x, u.ssao0.y);
        let factor = focus + view_z;
        let dofblur = vec2<f32>(clamp(factor * aperture, -max_blur, max_blur));
        var taps = BOKEH_TAPS;
        var col = textureSampleLevel(input_tex, input_sampler, uv, 0.0);
        for (var i: i32 = 0; i < 40; i = i + 1) {
            let t = taps[i];
            let off = t.xy * aspect_correct * dofblur * t.z;
            col = col + textureSampleLevel(input_tex, input_sampler, uv + vec2<f32>(off.x, -off.y), 0.0);
        }
        return postfx_passthrough(col.rgb / 41.0, 1.0);
    }
    // 27: SAO — depth-difference ambient occlusion. params2.x = intensity.
    if (kind == 27u) {
        let intensity = u.params2.x;
        let d = sample_postfx_depth(uv, false);
        var occ = 0.0;
        for (var i: i32 = -2; i <= 2; i = i + 1) {
            for (var j: i32 = -2; j <= 2; j = j + 1) {
                let off = vec2<f32>(f32(i), f32(j)) * texel * 2.0;
                let sd = sample_postfx_depth(uv + off, false);
                occ = occ + max(0.0, sd - d);
            }
        }
        occ = clamp(occ * intensity, 0.0, 1.0);
        return postfx_write(base.rgb * (1.0 - occ), base.a);
    }
    // 28: SMAA-style edge blend (subpixel AA via luma neighborhood).
    if (kind == 28u) {
        let nw = textureSampleLevel(input_tex, input_sampler, uv + vec2<f32>(-texel.x, -texel.y), 0.0);
        let ne = textureSampleLevel(input_tex, input_sampler, uv + vec2<f32>( texel.x, -texel.y), 0.0);
        let sw = textureSampleLevel(input_tex, input_sampler, uv + vec2<f32>(-texel.x,  texel.y), 0.0);
        let se = textureSampleLevel(input_tex, input_sampler, uv + vec2<f32>( texel.x,  texel.y), 0.0);
        let l_m = luma(base.rgb);
        let l_avg = (luma(nw.rgb) + luma(ne.rgb) + luma(sw.rgb) + luma(se.rgb)) * 0.25;
        let edge = abs(l_m - l_avg);
        let blend = smoothstep(0.05, 0.2, edge);
        let avg = (base.rgb + nw.rgb + ne.rgb + sw.rgb + se.rgb) / 5.0;
        return postfx_write(mix(base.rgb, avg, blend * 0.65), base.a);
    }
    // 29: Solid clear — params2 = rgba (ignores input).
    if (kind == 29u) {
        return vec4<f32>(u.params2.x, u.params2.y, u.params2.z, u.params2.w);
    }
    // 33: Luminosity — three.js LuminosityShader: the luminance in rgb.
    // AdaptiveToneMappingPass's first step, into a half-float target.
    if (kind == 33u) {
        return postfx_passthrough(vec3<f32>(postfx_luminance(base.rgb)), base.a);
    }
    // 34: Adapt luminance — AdaptiveToneMappingPass's adaptive luminance
    // shader (Pattanaik; three.js r152, the last release with the pass).
    // Input: this frame's average luminance (1×1, what three.js reads from the
    // mip chain's 1×1 level); `normal_tex`: last frame's adapted luminance,
    // read per pixel at `vUv` as three.js does (its targets are resolution²).
    // params2: x = minLuminance, y = delta (seconds), z = tau (the adaption
    // rate). The current luminance is squared, as three.js does.
    if (kind == 34u) {
        let min_lum = u.params2.x;
        let last = max(min_lum, textureSampleLevel(normal_tex, depth_sampler, uv, 0.0).r);
        var current = max(min_lum, base.r);
        current = current * current;
        let adapted = last + (current - last) * (1.0 - exp(-u.params2.y * u.params2.z));
        return vec4<f32>(adapted, adapted, adapted, 1.0);
    }
    // 30: Output / display encode. params2.x > 0.5 applies sRGB encode —
    // three.js r165 OutputShader's `sRGBTransferOETF`, the exact curve, not
    // `postfx_encode`'s 2.2 power.
    if (kind == 30u) {
        if (u.params2.x > 0.5) {
            return vec4<f32>(postfx_srgb_encode(base.rgb), base.a);
        }
        return vec4<f32>(base.rgb, base.a);
    }
    if (kind >= 40u) {
        return postfx_fragment_ext(in, kind, uv, res, texel, base);
    }
    return vec4<f32>(postfx_encode(base.rgb), base.a);
}
"#, include_str!("postfx_kinds.wgsl"));

pub const MAX_DIR_LIGHTS: usize = 8;
pub const MAX_POINT_LIGHTS: usize = 8;
pub const MAX_SPOT_LIGHTS: usize = 8;
pub const MAX_HEMI_LIGHTS: usize = 8;
pub const MAX_RECT_LIGHTS: usize = 8;

pub const SHADER_SOURCE: &str = r#"
const MAT_BASIC    : u32 = 0u;
const MAT_LAMBERT  : u32 = 1u;
const MAT_PHONG    : u32 = 2u;
const MAT_STANDARD : u32 = 3u;
const MAT_PHYSICAL : u32 = 4u;
const MAT_NORMAL   : u32 = 5u;
const MAT_DEPTH    : u32 = 6u;
const MAT_TOON     : u32 = 7u;
const MAT_MATCAP   : u32 = 8u;
const MAT_LINE     : u32 = 9u;
const MAT_POINTS   : u32 = 10u;
const MAT_SPRITE   : u32 = 11u;
const MAT_DISTANCE : u32 = 12u;
const MAT_SKY      : u32 = 13u;
const MAT_MIRROR   : u32 = 14u;
const MAT_ATMOSPHERE : u32 = 16u;

// three.js `alphaTest`, carried bit-cast in `mesh.flags.w`; 0 (the default)
// never discards, since no alpha is below it.
fn alpha_test_of() -> f32 {
    return bitcast<f32>(mesh.flags.w);
}

// three.js `alphamap_fragment`: the alpha map's green channel scales the
// opacity, after the colour map and before `alphaTest`.
fn alpha_map_of(uv : vec2<f32>) -> f32 {
    if (HAS_ALPHA_MAP) {
        return sample_alpha_map(uv).g;
    }
    return 1.0;
}

// three.js `lights_fragment_maps`: the light map's texel times
// `lightMapIntensity`, added to the indirect irradiance.
fn light_map_irradiance(uv : vec2<f32>) -> vec3<f32> {
    if (HAS_LIGHT_MAP) {
        return sample_light_map(uv).rgb * map_params().x;
    }
    return vec3<f32>(0.0);
}

// three.js `specularmap_fragment`: the specular map's red channel.
fn specular_strength_of(uv : vec2<f32>) -> f32 {
    if (HAS_SPECULAR_MAP) {
        return sample_specular_map(uv).r;
    }
    return 1.0;
}

// three.js `envmap_fragment` (Basic, Lambert, Phong): the reflected — or, for
// a refraction-mapped cube, refracted — view ray looks the `envMap` up, and
// the result meets `outgoing` per `combine`, weighted by `specularStrength ×
// reflectivity`. The ray runs from the camera to the fragment, or along the
// camera's forward axis for an orthographic one (`isOrthographic`).
fn apply_env_map(outgoing : vec3<f32>, n : vec3<f32>, world_pos : vec3<f32>, specular_strength : f32) -> vec3<f32> {
    if (!HAS_ENV_MAP) {
        return outgoing;
    }
    var to_frag = normalize(world_pos - frame.camera_position.xyz);
    if (frame.projection[2][3] == 0.0) {
        to_frag = normalize(-vec3<f32>(frame.view[0][2], frame.view[1][2], frame.view[2][2]));
    }
    let p = env_params();
    var r = reflect(to_frag, n);
    if (p.w > 0.5) {
        r = refract(to_frag, n, p.z);
    }
    let env = sample_env_map(r).rgb;
    let k = specular_strength * p.y;
    let combine = u32(p.x);
    if (combine == 1u) {
        return mix(outgoing, env, k);
    }
    if (combine == 2u) {
        return outgoing + env * k;
    }
    return mix(outgoing, outgoing * env, k);
}

// three.js `bumpmap_pars_fragment`: `dHdxy_fwd` and `perturbNormalArb`
// (Mikkelsen, "Bump Mapping Unparametrized Surfaces on the GPU"). The height is
// the map's red channel times `bumpScale`, differenced forward along the
// pixel's uv derivatives; `face` is three.js's `faceDirection`. Taken in world
// space rather than view space, which changes nothing: the construction uses
// only crosses and dots, and the view transform is a rotation. Neither is the
// framebuffer's y direction: flipping it negates `dy` terms in pairs.
fn bump_normal(n : vec3<f32>, pos : vec3<f32>, uv : vec2<f32>, face : f32) -> vec3<f32> {
    let scale = map_params().y;
    let dstdx = dpdx(uv);
    let dstdy = dpdy(uv);
    let hll = scale * sample_bump_map(uv).x;
    let dbx = scale * sample_bump_map(uv + dstdx).x - hll;
    let dby = scale * sample_bump_map(uv + dstdy).x - hll;
    let sigma_x = normalize(dpdx(pos));
    let sigma_y = normalize(dpdy(pos));
    let r1 = cross(sigma_y, n);
    let r2 = cross(n, sigma_x);
    let det = dot(sigma_x, r1) * face;
    let grad = sign(det) * (dbx * r1 + dby * r2);
    return normalize(abs(det) * n - grad);
}

const FLAG_MAP           : u32 = 1u;
const FLAG_NORMAL_MAP    : u32 = 2u;
const FLAG_ROUGHNESS_MAP : u32 = 4u;
const FLAG_METALNESS_MAP : u32 = 8u;
const FLAG_AO_MAP        : u32 = 16u;
const FLAG_EMISSIVE_MAP  : u32 = 32u;
const FLAG_MATCAP_MAP    : u32 = 64u;
const FLAG_CLOUD_SHADOW     : u32 = 4096u;
const FLAG_DASHED        : u32 = 128u;
const FLAG_SHADOW_MAT    : u32 = 256u;
const FLAG_RECEIVE_SHADOW: u32 = 512u;
// How many lights of each kind a frame carries (renderer.rs `MAX_*_LIGHTS`).
const MAX_LIGHTS_PER_KIND: u32 = 8u;
// The draw is in the encoded-blend pass: a `Unorm` view of an sRGB target, so
// the output is encoded here and the hardware blends encoded values, as
// three.js's canvas does (renderer.rs `encoded_pass`).
const FLAG_ENCODED_OUT: u32 = 8192u;
// three.js `toneMapped: false`: `apply_tone_mapping` passes the colour through.
const FLAG_NO_TONE_MAP: u32 = 16384u;
const FLAG_OPAQUE_OUT: u32 = 32768u;
// A transmissive physical material in a blended glass mode (`Glass`, `Oit`):
// what is behind arrives through blending, not from the transmission capture
// (renderer.rs `FLAG_BLENDED_GLASS`).
const FLAG_BLENDED_GLASS: u32 = 65536u;
// A transmissive physical caster (renderer.rs `FLAG_SHADOW_TRANSMITS`).
const FLAG_SHADOW_TRANSMITS: u32 = 131072u;
// Which half of a `Glass`-mode blend this pipeline draws (renderer.rs
// `make_glass_pipeline`). The glass is `behind · T + L`, and one alpha cannot
// say that: it tints by its mean, and a highlight divided by a small alpha to
// survive the blend clamps first. So it is two draws — 1: the tint, `T` per
// channel, multiplied over what is behind; 2: the light, `L`, added. 0 is
// every other pipeline.
override GLASS_OUTPUT : u32 = 0u;

const PI : f32 = 3.14159265358979;

struct DirLight {
    direction : vec4<f32>,
    color     : vec4<f32>,
};

struct PointLight {
    position : vec4<f32>,
    color    : vec4<f32>,
    params   : vec4<f32>,
};

struct SpotLight {
    position  : vec4<f32>,
    direction : vec4<f32>,
    color     : vec4<f32>,
    params    : vec4<f32>,
};

/// Rectangular area light. `right`/`up` are half-extent vectors, so the corners
/// are `position ± right ± up`.
struct RectLight {
    position : vec4<f32>,
    color    : vec4<f32>,
    right    : vec4<f32>,
    up       : vec4<f32>,
};

struct HemiLight {
    sky_color    : vec4<f32>,
    ground_color : vec4<f32>,
    direction    : vec4<f32>,
};

struct FrameUniforms {
    view            : mat4x4<f32>,
    projection      : mat4x4<f32>,
    view_proj       : mat4x4<f32>,
    camera_position : vec4<f32>,
    ambient         : vec4<f32>,
    light_counts    : vec4<u32>,
    /// x: tone-mapping mode (0 none, 1 linear, 2 ACES), y: exposure, z: IBL on,
    /// w: linear framebuffer (1 = HalfFloat RT)
    tone_mapping_exposure : vec4<f32>,
    /// Fog color rgb (a unused — packed for layout alignment with vec4).
    fog_color       : vec4<f32>,
    /// x: linear-fog near, y: linear-fog far, z: exp2 density, w: mode (0/1/2)
    fog_params      : vec4<f32>,
    /// x: viewport width (pixels), y: viewport height (pixels), z: camera near, w: camera far.
    viewport_size   : vec4<f32>,
    /// CubeUV PMREM: x/y = texel size, z = max mip, w = 1 when CubeUV env is active.
    env_map_params  : vec4<f32>,
    dir_lights      : array<DirLight,   8>,
    point_lights    : array<PointLight, 8>,
    spot_lights     : array<SpotLight,  8>,
    hemi_lights     : array<HemiLight,  8>,
    rect_lights     : array<RectLight,  8>,
    /// x: rect-area light count, y: 1 for geometric specular antialiasing
    /// (`Renderer::set_specular_antialiasing`), z: 1 for reflective
    /// transparency (`Renderer::set_reflective_transparency`), w: the
    /// transmission capture's mip count (0 = none).
    light_counts2   : vec4<u32>,
    /// x: shadow records in use.
    shadow_info     : vec4<u32>,
    /// One record per shadow-casting light, which names it in its own
    /// `color.w` (record + 1; 0 for a light that casts none).
    shadows         : array<ShadowRecord, 16>,
};

/// A shadow-casting light's shadow (renderer.rs `ShadowRecordGpu`).
struct ShadowRecord {
    /// x: kind (0 directional, 1 spot, 2 point); y: its first layer in the
    /// kind's map array; z: cascades (directional; 1 otherwise); w: filter
    /// (`ShadowType`: 0 basic, 1 PCF, 2 soft PCF, 3 PCSS).
    info   : vec4<u32>,
    /// x: depth bias; y: normal bias, world units; z: the filter disc's
    /// radius in map uv (or direction units for a point light); w: one
    /// texel in the same units.
    params : vec4<f32>,
    /// Directional: each cascade's far edge as a view depth. Point: xyz the
    /// light's position, w its radius.
    extra  : vec4<f32>,
    /// PCSS (`shadow_pcss`): x the blocker search's radius in the filter's
    /// units; y, z a spot's shadow camera near and far; w the light's size
    /// as the penumbra's scale.
    pcss   : vec4<f32>,
    /// World → light clip, a cascade each (directional), or the light's.
    vp     : array<mat4x4<f32>, 4>,
};

/// What a shadow pass draws with (renderer.rs `ShadowPassGpu`), at a
/// dynamic offset per pass: its light's matrix — one cascade, the spot's
/// frustum, or one face of a point light's cube — and for a point light its
/// position and radius.
struct ShadowPass {
    vp    : mat4x4<f32>,
    light : vec4<f32>,
};

struct MeshUniforms {
    model         : mat4x4<f32>,
    normal_matrix : mat4x4<f32>,
    color         : vec4<f32>,
    emissive      : vec4<f32>,
    specular      : vec4<f32>,
    /// x: shininess (phong) / depth_near, y: opacity / depth_far,
    /// z: roughness, w: metalness
    params        : vec4<f32>,
    /// x: ao_intensity, y: normal_scale.x, z: normal_scale.y, w: unused
    params2       : vec4<f32>,
    /// Physical material: x: clearcoat, y: clearcoat_roughness, z: ior, w: transmission
    params3       : vec4<f32>,
    /// Physical volume: x: thickness (glass refraction/march depth),
    /// y: dispersion, z: vertex_emissive, w: anisotropy_rotation (radians).
    params4       : vec4<f32>,
    /// Physical layers 2: x: iridescence, y: iridescence_ior,
    /// z: iridescence_thickness (nm), w: sheen strength.
    params5       : vec4<f32>,
    /// x/y/z: sheen_color, w: sheen_roughness.
    params6       : vec4<f32>,
    /// x/y/z: attenuation_color, w: attenuation_distance (0 = disabled).
    params7       : vec4<f32>,
    /// x: displacement_scale, y: displacement_bias.
    params8       : vec4<f32>,
    /// UV transform: xy offset, zw repeat (three.js `Texture.offset/repeat`).
    params9       : vec4<f32>,
    /// x: cloud shell height, y: cloud shadow strength, z: cloud longitude
    /// offset (turns), w: twilight wrap width.
    params10      : vec4<f32>,
    /// xyz: twilight colour, w: the sun's angular radius in radians.
    params11      : vec4<f32>,
    /// Eclipse occluder: world-space xyz centre, w radius (0 = none).
    params12      : vec4<f32>,
    /// x: cloud forward-scatter strength, y: scattering anisotropy g,
    /// z: ozone absorption strength, w: aurora strength.
    params13      : vec4<f32>,
    /// xyz: aurora colour, w: colatitude of the auroral oval, in degrees.
    params14      : vec4<f32>,
    /// x: material kind, y: texture-slot flags
    flags         : vec4<u32>,
    /// three.js morph targets: x target count (0 = none), y
    /// `morphTargetBaseInfluence`, z texels per vertex in `morph_tex`, w flags
    /// (1 normals blend, 2 colours blend, 4 colour has alpha, 8 positions
    /// blend).
    morph         : vec4<f32>,
    /// `morphTargetInfluences`, four to a vector (`MAX_MORPH_TARGETS` = 64).
    morph_weights : array<vec4<f32>, 16>,
    /// `Object3D::clip_plane`: xyz normal, w constant; all zero = no plane.
    clip_plane    : vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame : FrameUniforms;
// The shadow passes' own group 0 (renderer.rs `shadow_pass_bgl`).
@group(0) @binding(3) var<uniform> shadow_pass : ShadowPass;

/// three.js's `clipping_planes_fragment`, for one world-space plane: drop the
/// fragment on the plane's negative side. Every colour entry point calls this
/// first, so lines, points and transparent passes clip exactly as meshes do.
fn clip_fragment(world_pos : vec3<f32>) {
    let pl = mesh.clip_plane;
    if (any(pl.xyz != vec3<f32>(0.0)) && dot(pl.xyz, world_pos) + pl.w < 0.0) {
        discard;
    }
}
// three.js's transmission render target (`transmissionSamplerMap`): the
// opaque scene drawn linear and untone-mapped, mipmapped — a 1×1 stand-in
// while the capture itself is drawn. Sampled trilinearly, clamped.
@group(0) @binding(1) var transmission_tex : texture_2d<f32>;
@group(0) @binding(2) var transmission_sampler : sampler;
@group(1) @binding(0) var<uniform> mesh  : MeshUniforms;

// The material's maps: declarations and `sample_*` accessors, generated per
// material feature set — see `shader_source_with_maps` below.
//@@MATERIAL_MAPS@@
// three.js's morph-target texture (`WebGLMorphtargets`): one layer per target,
// `morph.z` texels per vertex — position, then normal, then colour — laid out
// row-major over the layer's width. Read with `textureLoad` only, so it is an
// unfilterable float texture and lowers to `texelFetch` on WebGL2, which has
// no storage buffers to hold it any other way.
@group(2) @binding(10) var morph_tex : texture_2d_array<f32>;

// The geometry's extra uv sets — three.js's `uv1`, `uv2`, `uv3` — two texels
// a vertex, `(uv1, uv2)` and `(uv3, 0, 0)`, read by vertex index: a texture,
// so the interleaved vertex layout stays as `Renderer::vertex_buffer`
// publishes it. `mesh.morph.w` bit 32 says the geometry has them; without,
// they read zero, as an attribute a geometry lacks does in three.js.
@group(2) @binding(8) var uv_sets_tex : texture_2d<f32>;

struct UvSets {
    uv12 : vec4<f32>,
    uv3  : vec2<f32>,
};

fn vertex_uv_sets(vid : u32) -> UvSets {
    var out : UvSets;
    if ((u32(mesh.morph.w) & 32u) == 0u) {
        return out;
    }
    let width = textureDimensions(uv_sets_tex).x;
    let t0 = vid * 2u;
    let t1 = t0 + 1u;
    out.uv12 = textureLoad(uv_sets_tex, vec2<i32>(i32(t0 % width), i32(t0 / width)), 0);
    out.uv3 = textureLoad(uv_sets_tex, vec2<i32>(i32(t1 % width), i32(t1 / width)), 0).xy;
    return out;
}

// The uv sets at this vertex or fragment, for a map whose `channel` names
// one (`map_xform` in a material variant's prelude).
var<private> map_uv12 : vec4<f32>;
var<private> map_uv3 : vec2<f32>;

/// three.js `Texture.channel`: `uv` (0), `uv1`, `uv2` or `uv3`.
fn map_uv_set(channel : f32, uv : vec2<f32>) -> vec2<f32> {
    if (channel < 0.5) {
        return uv;
    }
    if (channel < 1.5) {
        return map_uv12.xy;
    }
    if (channel < 2.5) {
        return map_uv12.zw;
    }
    return map_uv3;
}

/// Read vertex `vid`'s uv sets for the maps of this vertex stage and hand
/// them on to the fragment stage.
fn pass_uv_sets(vid : u32, out : ptr<function, VsOut>) {
    let sets = vertex_uv_sets(vid);
    map_uv12 = sets.uv12;
    map_uv3 = sets.uv3;
    (*out).uv12 = sets.uv12;
    (*out).uv3 = sets.uv3;
}

/// three.js `getMorph(vertexIndex, morphTargetIndex, offset)`.
fn morph_fetch(vid : u32, t : u32, slot : u32) -> vec4<f32> {
    let texel = vid * u32(mesh.morph.z) + slot;
    let width = textureDimensions(morph_tex).x;
    return textureLoad(morph_tex, vec2<i32>(i32(texel % width), i32(texel / width)), i32(t), 0);
}

/// Whether this draw's morph influences are per instance (three.js's
/// `USE_INSTANCING_MORPH`, an `InstancedMesh` with a `morphTexture`): then
/// the layer after the last target holds one row per instance —
/// `morphTargetBaseInfluence`, then the influences — one float a texel.
fn morph_per_instance() -> bool {
    return (u32(mesh.morph.w) & 16u) != 0u;
}

/// One float of the per-instance rows (`morphinstance_vertex.glsl`'s
/// `texelFetch(morphTexture, ivec2(k, gl_InstanceID))`).
fn morph_instance_texel(iid : u32, k : u32) -> f32 {
    let count = u32(mesh.morph.x);
    let texel = iid * (count + 1u) + k;
    let width = textureDimensions(morph_tex).x;
    return textureLoad(morph_tex, vec2<i32>(i32(texel % width), i32(texel / width)), i32(count), 0).x;
}

/// `morphTargetBaseInfluence` for instance `iid`.
fn morph_base(iid : u32) -> f32 {
    if (morph_per_instance()) {
        return morph_instance_texel(iid, 0u);
    }
    return mesh.morph.y;
}

/// `morphTargetInfluences[i]` for instance `iid`.
fn morph_weight(i : u32, iid : u32) -> f32 {
    if (morph_per_instance()) {
        return morph_instance_texel(iid, i + 1u);
    }
    return mesh.morph_weights[i / 4u][i % 4u];
}

/// `base · v + Σ wᵢ · targetᵢ[slot]` — `morphtarget_vertex.glsl` (slot 0) and
/// `morphnormal_vertex.glsl` (slot 1) — with instance `iid`'s influences.
fn morph_blend3_instance(v : vec3<f32>, vid : u32, iid : u32, slot : u32) -> vec3<f32> {
    let count = u32(mesh.morph.x);
    if (count == 0u) {
        return v;
    }
    var out = v * morph_base(iid);
    for (var i = 0u; i < count; i = i + 1u) {
        let w = morph_weight(i, iid);
        if (w != 0.0) {
            out = out + morph_fetch(vid, i, slot).xyz * w;
        }
    }
    return out;
}

fn morph_blend3(v : vec3<f32>, vid : u32, slot : u32) -> vec3<f32> {
    return morph_blend3_instance(v, vid, 0u, slot);
}

/// three.js `#include <morphtarget_vertex>` for vertex `vid` (the
/// geometry's own index, `gl_VertexID`) of instance `iid`. A custom vertex
/// stage calls this (or [`morph_position`] outside instancing, or
/// `standard_vertex_morphed`) to morph a `ShaderMaterial`'s geometry.
fn morph_position_instance(p : vec3<f32>, vid : u32, iid : u32) -> vec3<f32> {
    if ((u32(mesh.morph.w) & 8u) == 0u) {
        return p;
    }
    return morph_blend3_instance(p, vid, iid, 0u);
}

/// three.js `#include <morphnormal_vertex>`, per instance.
fn morph_normal_instance(n : vec3<f32>, vid : u32, iid : u32) -> vec3<f32> {
    if ((u32(mesh.morph.w) & 1u) == 0u) {
        return n;
    }
    return morph_blend3_instance(n, vid, iid, 1u);
}

/// `morphcolor_vertex.glsl`, per instance: rgb for a 3-component colour, all
/// four channels for a 4-component one.
fn morph_color_instance(c : vec4<f32>, vid : u32, iid : u32) -> vec4<f32> {
    let count = u32(mesh.morph.x);
    let flags = u32(mesh.morph.w);
    if (count == 0u || (flags & 2u) == 0u) {
        return c;
    }
    var out = c * morph_base(iid);
    for (var i = 0u; i < count; i = i + 1u) {
        let w = morph_weight(i, iid);
        if (w != 0.0) {
            out = out + morph_fetch(vid, i, 2u) * w;
        }
    }
    if ((flags & 4u) == 0u) {
        return vec4<f32>(out.rgb, c.a);
    }
    return out;
}

/// three.js `#include <morphtarget_vertex>`: vertex `vid`'s position, morphed.
fn morph_position(p : vec3<f32>, vid : u32) -> vec3<f32> {
    return morph_position_instance(p, vid, 0u);
}

/// three.js `#include <morphnormal_vertex>`.
fn morph_normal(n : vec3<f32>, vid : u32) -> vec3<f32> {
    return morph_normal_instance(n, vid, 0u);
}

/// three.js `#include <morphcolor_vertex>`.
fn morph_color(c : vec4<f32>, vid : u32) -> vec4<f32> {
    return morph_color_instance(c, vid, 0u);
}

/// How much of the sun a sphere leaves visible from this point.
///
/// The sun and the occluder are both circles on the sky, of angular radii `a`
/// and `b`, their centres `d` apart. What is lost is the area the two discs
/// share, over the sun's area — the standard circle-circle lens. Doing it this
/// way rather than with a shadow map gives the penumbra for nothing, because
/// the sun is half a degree wide and not a point; it gives the annular ring
/// when the occluder is smaller and centred; and it works over a lunar orbit,
/// where a shadow map fitted to the planet would never reach the moon.
/// Lambert's cosine, softened by the sun being a disc rather than a point.
///
/// `max(dot(n, l), 0)` is the cosine for a *point* source, and it puts a hard
/// edge on the terminator. The sun is about half a degree across, so near the
/// terminator part of its disc is below the horizon and part is above, and the
/// light falls off over that angular width instead of switching. From orbit
/// that is a band roughly thirty kilometres wide on Earth — narrow, but a hard
/// edge at this scale reads as a rendering artefact rather than as a sunrise.
///
/// The returned value is the first moment of the visible part of the disc: it
/// meets `dot(n, l)` exactly where the disc clears the horizon either way, so
/// nothing outside the band changes.
fn soft_lambert(ndl : f32, sun_radius : f32) -> f32 {
    if (sun_radius <= 0.0) {
        return max(ndl, 0.0);
    }
    let x = ndl / sun_radius;
    if (x >= 1.0) { return ndl; }
    if (x <= -1.0) { return 0.0; }
    return sun_radius * (x * acos(-x) + sqrt(max(1.0 - x * x, 0.0))) / PI;
}

/// Henyey-Greenstein phase function: how much light scattering off a droplet
/// carries on in a given direction.
///
/// `g` is the asymmetry — 0 scatters evenly, positive keeps light going the way
/// it was already headed. Cloud droplets are far larger than visible
/// wavelengths and sit around 0.8, which is why a deck with the sun behind it
/// glows and the same deck lit from over your shoulder does not.
fn henyey_greenstein(cos_theta : f32, g : f32) -> f32 {
    let g2 = g * g;
    let d = max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4);
    return (1.0 - g2) / (4.0 * PI * d * sqrt(d));
}

fn eclipse_factor(world_p : vec3<f32>, to_sun : vec3<f32>) -> f32 {
    let radius = mesh.params12.w;
    let a = mesh.params11.w;
    if (radius <= 0.0 || a <= 0.0) {
        return 1.0;
    }
    let v = mesh.params12.xyz - world_p;
    let dist = length(v);
    if (dist < 1e-5) {
        return 1.0;
    }
    let dir = v / dist;
    let cos_sep = dot(dir, to_sun);
    // Behind us relative to the sun: it cannot be in the way.
    if (cos_sep <= 0.0) {
        return 1.0;
    }
    let b = asin(clamp(radius / max(dist, radius), 0.0, 1.0));
    let d = acos(clamp(cos_sep, -1.0, 1.0));
    if (d >= a + b) {
        return 1.0;                       // discs apart
    }
    if (d <= b - a) {
        return 0.0;                       // total: the sun is behind it
    }
    if (d <= a - b) {
        return 1.0 - (b * b) / (a * a);   // annular: occluder wholly inside
    }
    // Partial: the lens where the two discs overlap.
    let a2 = a * a;
    let b2 = b * b;
    let d2 = d * d;
    let x = (d2 + a2 - b2) / (2.0 * d);
    let y = (d2 + b2 - a2) / (2.0 * d);
    let lens = a2 * acos(clamp(x / a, -1.0, 1.0)) - x * sqrt(max(a2 - x * x, 0.0))
             + b2 * acos(clamp(y / b, -1.0, 1.0)) - y * sqrt(max(b2 - y * y, 0.0));
    return clamp(1.0 - lens / (PI * a2), 0.0, 1.0);
}

/// How much sunlight a cloud deck above this point lets through.
///
/// Intersects the ray from the shaded point toward the sun with a shell
/// `params10.x` above the body and samples the cloud map's alpha where it
/// crosses. That offset is the whole point: directly under the sun a cloud
/// shadows the ground beneath it, and as the sun drops the shadow slides away
/// across the surface, which is what stops the deck looking painted on.
///
/// Works in the body's own space, where it is a unit sphere. The inverse of the
/// model's linear part is the transpose of the normal matrix — `normal_matrix`
/// is `(M⁻¹)ᵀ`, so its transpose is `M⁻¹` — which also makes this correct under
/// the flattening a spinning planet has.
fn cloud_shadow(local_p : vec3<f32>, sun_world : vec3<f32>) -> f32 {
    if ((mesh.flags.y & FLAG_CLOUD_SHADOW) == 0u) {
        return 1.0;
    }
    let inv = transpose(mat3x3<f32>(
        mesh.normal_matrix[0].xyz, mesh.normal_matrix[1].xyz, mesh.normal_matrix[2].xyz));
    let s = normalize(inv * sun_world);
    let n = normalize(local_p);
    let rc = 1.0 + max(mesh.params10.x, 1e-4);
    let b = dot(n, s);
    // The far root: the ray leaves through the top of the shell.
    let t = -b + sqrt(max(b * b + rc * rc - 1.0, 0.0));
    let q = normalize(n + s * t);
    // Equirectangular UV, matching SphereGeometry's mapping:
    //   position = (-cos(phi)·sin(theta), cos(theta), sin(phi)·sin(theta))
    let u = atan2(q.z, -q.x) / (2.0 * PI) + mesh.params10.z;
    let v = 1.0 - acos(clamp(q.y, -1.0, 1.0)) / PI;
    let cover = sample_cloud(vec2<f32>(u, v)).a;
    return 1.0 - clamp(cover, 0.0, 1.0) * clamp(mesh.params10.y, 0.0, 1.0);
}

/// Offset an object-space position along its normal by the height map.
///
/// Sampled with `textureSampleLevel`: the vertex stage has no derivatives, so
/// an implicit-LOD `textureSample` is not available there. Level 0 is also what
/// you want — a mip-filtered height map would flatten the very relief this is
/// displacing by.
fn transform_uv(uv : vec2<f32>) -> vec2<f32> {
    return uv * mesh.params9.zw + mesh.params9.xy;
}

fn apply_displacement(position : vec3<f32>, normal : vec3<f32>, uv : vec2<f32>) -> vec3<f32> {
    if (!HAS_DISPLACEMENT_MAP) {
        return position;
    }
    let h = sample_displacement_map(uv).r;
    return position + normalize(normal) * (h * mesh.params8.x + mesh.params8.y);
}

// The environment a draw's image-based light reads — the scene's, or a
// Standard/Physical material's own `envMap` — and how: `params` the PMREM
// atlas's texel width and height, its top mip (or the plain cube's mip count)
// and whether it is an atlas; `info.x` whether there is one at all,
// `info.y` three.js's `envMapIntensity` (the material's with its own map,
// `scene.environmentIntensity` with the scene's); `rot` the rows of
// `envMapRotation` (or `scene.environmentRotation`).
struct EnvUniforms {
    params : vec4<f32>,
    info   : vec4<f32>,
    rot    : array<vec4<f32>, 3>,
};
@group(3) @binding(13) var<uniform> env_u : EnvUniforms;
@group(3) @binding(0) var env_tex              : texture_cube<f32>;
@group(3) @binding(1) var env_sampler          : sampler;
@group(3) @binding(8) var env_cube_uv_tex      : texture_2d<f32>;
// One layer per shadow-casting light of the kind; six per point light, its
// cube faces, looked up by `point_shadow_face` (WebGL2 has no cube arrays).
@group(3) @binding(2) var shadow_tex           : texture_depth_2d_array;
@group(3) @binding(3) var shadow_sampler       : sampler_comparison;
@group(3) @binding(4) var spot_shadow_tex      : texture_depth_2d_array;
@group(3) @binding(5) var spot_shadow_sampler  : sampler_comparison;
@group(3) @binding(6) var point_shadow_tex     : texture_depth_2d_array;
@group(3) @binding(7) var point_shadow_sampler : sampler_comparison;
// Screen-space glass: mipmapped capture of the opaque scene + its trilinear
// sampler. Only read by fs_ss_glass; a 1×1 placeholder is bound otherwise.
@group(3) @binding(9)  var ss_color_tex   : texture_2d<f32>;
@group(3) @binding(10) var ss_color_samp  : sampler;
// Declared as plain float textures rather than `texture_depth_2d`, and bound
// with an unfilterable-float sample type, which a depth format permits. The
// reason is `textureLoad`: WGSL allows it on a depth texture but GLSL has no
// `texelFetch` for a depth sampler, so naga refuses the module for WebGL2. As
// ordinary float textures the same reads lower to `texelFetch` and the pass
// compiles everywhere. Depth arrives in `.r`.
@group(3) @binding(11) var ss_depth_tex   : texture_2d<f32>;
@group(3) @binding(12) var ss_back_depth_tex : texture_2d<f32>;
// Each light's shadow on the refracting surface at this pixel, drawn by
// `fs_glass_shadow` just before the refraction pass (see there).
@group(3) @binding(15) var glass_shadow_tex : texture_2d<u32>;

struct VsIn {
    @location(0) position : vec3<f32>,
    @location(1) normal   : vec3<f32>,
    @location(2) uv       : vec2<f32>,
    @location(3) color    : vec4<f32>,
    /// xyz = object-space tangent, w = bitangent handedness (glTF convention).
    /// All-zero means the geometry carries no tangents.
    ///
    /// Location 8, not 4: the skinned and instanced pipelines already consume
    /// locations 4-7 from their extra vertex buffers, and this lives in the
    /// shared base buffer that all three read.
    @location(8) tangent  : vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0) world_pos      : vec3<f32>,
    @location(1) world_normal   : vec3<f32>,
    @location(2) uv             : vec2<f32>,
    @location(3) view_z         : f32,
    @location(4) vertex_color   : vec4<f32>,
    @location(5) clip_zw        : vec2<f32>,
    /// Homogeneous projective UV for mirror materials (three.js `vUv`).
    @location(6) proj_uv        : vec4<f32>,
    /// World-space tangent + handedness, zero when the mesh has none.
    @location(7) world_tangent  : vec4<f32>,
    /// Object-space position, for shading that needs the body's own frame —
    /// the cloud-shadow shell intersection reads it.
    @location(8) local_pos      : vec3<f32>,
    /// three.js `uv1` (xy) and `uv2` (zw), and `uv3`.
    @location(9) uv12           : vec4<f32>,
    @location(10) uv3           : vec2<f32>,
};

fn mirror_proj_uv(local_pos: vec3<f32>) -> vec4<f32> {
    if (mesh.flags.x == MAT_MIRROR) {
        return mesh.normal_matrix * vec4<f32>(local_pos, 1.0);
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

/// Object-space tangent → world space.
///
/// Tangents transform by the model matrix (they are directions *along* the
/// surface), not by the inverse-transpose that normals use — under non-uniform
/// scale the two genuinely differ. An all-zero input is passed through as zero,
/// which the fragment shader reads as "no tangent supplied".
fn world_tangent(t : vec4<f32>) -> vec4<f32> {
    if (dot(t.xyz, t.xyz) < 1e-12) {
        return vec4<f32>(0.0);
    }
    let w = (mesh.model * vec4<f32>(t.xyz, 0.0)).xyz;
    return vec4<f32>(w, t.w);
}

/// The standard vertex transform, as a plain function.
///
/// `vs_main` is a thin wrapper over this, and the split is for custom vertex
/// stages: a WGSL entry point cannot be called, so a `ShaderMaterial` that
/// wants to *start* from the standard output and move it — a billboard, a
/// ribbon turned to face the camera, a strand expanded from control points —
/// had no way to reach it and had to re-derive the whole transform, including
/// the displacement and sky-depth cases it probably did not know about.
///
///     @vertex
///     fn vs_custom(in: VsIn, @builtin(vertex_index) vi: u32) -> VsOut {
///         var out = standard_vertex(in);
///         out.world_pos = out.world_pos + offset;
///         out.clip_pos = frame.view_proj * vec4<f32>(out.world_pos, 1.0);
///         return out;
///     }
fn standard_vertex(in : VsIn) -> VsOut {
    // A shader material binds no morph texture (its morph count is 0), so
    // the vertex index is never read.
    return standard_vertex_morphed(in, 0u);
}

/// `standard_vertex` with three.js's morph targets applied to vertex `vid`
/// (the geometry's own index — `gl_VertexID` — which is what the morph
/// texture is laid out by). Kept out of `VsIn` so a custom vertex stage can
/// still declare `@builtin(vertex_index)` itself.
fn standard_vertex_morphed(in : VsIn, vid : u32) -> VsOut {
    var out : VsOut;
    pass_uv_sets(vid, &out);
    let uv = transform_uv(in.uv);
    // three.js order: `begin_vertex`, `morphtarget_vertex` (and the normal's
    // `morphnormal_vertex`), then displacement along the morphed normal.
    let morphed_p = morph_position(in.position, vid);
    let morphed_n = morph_normal(in.normal, vid);
    let local_p = apply_displacement(morphed_p, morphed_n, uv);
    let world_p = mesh.model * vec4<f32>(local_p, 1.0);
    out.world_pos = world_p.xyz;
    out.clip_pos = frame.view_proj * world_p;
    // Sky: pin near far plane (three.js sets gl_Position.z = gl_Position.w). WebGPU
    // depth is [0, 1] with Less compare vs a 1.0 clear — z must stay strictly below w.
    if (mesh.flags.x == MAT_SKY) {
        // Reversed depth (camera_position.w) puts the far plane at 0.
        out.clip_pos.z = out.clip_pos.w * select(0.999, 0.001, frame.camera_position.w > 0.5);
    }
    out.clip_zw = out.clip_pos.zw;
    let n4 = mesh.normal_matrix * vec4<f32>(morphed_n, 0.0);
    out.world_normal = n4.xyz;
    out.uv = uv;
    let view_p = frame.view * world_p;
    out.view_z = -view_p.z;
    out.vertex_color = morph_color(in.color, vid);
    out.proj_uv = mirror_proj_uv(morphed_p);
    out.world_tangent = world_tangent(in.tangent);
    out.local_pos = local_p;
    return out;
}

@vertex
fn vs_main(in : VsIn, @builtin(vertex_index) vid : u32) -> VsOut {
    return standard_vertex_morphed(in, vid);
}

// three.js `points_vert`: a `PointsMaterial` point, one instance per vertex,
// expanded here to a quad — WebGPU has no point size. `mesh.params.x` is the
// size in device pixels; with `params.w` set and a perspective camera it
// attenuates as three.js's does, `size × (height / 2) / −z`. The quad's uv is
// three.js's `(gl_PointCoord.x, 1 − gl_PointCoord.y)`, through the map's
// transform (`uvTransform`).
@vertex
fn vs_points(in : VsIn, @builtin(vertex_index) corner : u32, @builtin(instance_index) point : u32) -> VsOut {
    var out = standard_vertex_morphed(in, point);
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let c = corners[corner];
    var size = mesh.params.x;
    let perspective = frame.projection[2][3] != 0.0;
    if (mesh.params.w > 0.5 && perspective) {
        size = size * (frame.viewport_size.y * 0.5) / max(out.view_z, 1e-4);
    }
    // Half the size in pixels, as NDC (2 / viewport per pixel), times w.
    out.clip_pos = vec4<f32>(
        out.clip_pos.xy + c * size / frame.viewport_size.xy * out.clip_pos.w,
        out.clip_pos.zw,
    );
    out.clip_zw = out.clip_pos.zw;
    out.uv = transform_uv(c * 0.5 + vec2<f32>(0.5));
    return out;
}

// A shadow caster's clip position and map uv. The depth-only pipelines ignore
// the uv; the cut-out ones (`fs_shadow_cut`) test the map's alpha with it.
struct ShadowOut {
    @builtin(position) position : vec4<f32>,
    @location(0)       uv       : vec2<f32>,
};

fn shadow_out(position : vec4<f32>, uv : vec2<f32>) -> ShadowOut {
    var o : ShadowOut;
    o.position = position;
    o.uv = transform_uv(uv);
    return o;
}

@vertex
fn vs_shadow(in : VsIn, @builtin(vertex_index) vid : u32) -> ShadowOut {
    // Displaced geometry has to cast the shadow its displaced silhouette
    // implies, or the terrain and its shadow disagree — and a morphed one
    // the shadow of its current shape.
    let p = morph_position(in.position, vid);
    let n = morph_normal(in.normal, vid);
    return shadow_out(
        shadow_pass.vp * mesh.model * vec4<f32>(apply_displacement(p, n, transform_uv(in.uv)), 1.0),
        in.uv,
    );
}

@vertex
fn vs_shadow_spot(in : VsIn, @builtin(vertex_index) vid : u32) -> ShadowOut {
    return shadow_out(
        shadow_pass.vp * mesh.model * vec4<f32>(morph_position(in.position, vid), 1.0),
        in.uv,
    );
}

struct ShadowPointOut {
    @builtin(position) position : vec4<f32>,
    @location(0)       world    : vec3<f32>,
    @location(1)       uv       : vec2<f32>,
};

@vertex
fn vs_shadow_point(in : VsIn, @builtin(vertex_index) vid : u32) -> ShadowPointOut {
    let world = mesh.model * vec4<f32>(morph_position(in.position, vid), 1.0);
    var o : ShadowPointOut;
    o.position = shadow_pass.vp * world;
    o.world = world.xyz;
    o.uv = transform_uv(in.uv);
    return o;
}


struct SkinnedVsIn {
    @location(0) position : vec3<f32>,
    @location(1) normal   : vec3<f32>,
    @location(2) uv       : vec2<f32>,
    @location(3) color    : vec4<f32>,
    @location(4) joints   : vec4<f32>,
    @location(5) weights  : vec4<f32>,
    @location(8) tangent  : vec4<f32>,
    @builtin(vertex_index) vertex_index : u32,
};

// A fixed uniform array, not a runtime-sized storage buffer.
//
// WebGL2 has no storage buffers, and one declaration is enough to stop the
// module compiling on a GL backend whether or not anything skinned is drawn.
// 128 matrices is 8 KB — inside WebGL2's 16 KB uniform-block floor — and more
// bones than any rig this renderer has been pointed at.
const MAX_BONES : u32 = 128u;
struct Bones { m : array<mat4x4<f32>, 128> }
@group(1) @binding(1) var<uniform> bone_matrices : Bones;

/// `Σ wᵢ · boneMatrixᵢ` for one vertex.
fn skin_matrix(joints : vec4<f32>, weights : vec4<f32>) -> mat4x4<f32> {
    let j = vec4<u32>(u32(joints.x), u32(joints.y), u32(joints.z), u32(joints.w));
    // Clamped, because a fixed array has no bounds check to fall back on and a
    // rig with more bones than the limit would read past the end.
    let last = MAX_BONES - 1u;
    return bone_matrices.m[min(j.x, last)] * weights.x
         + bone_matrices.m[min(j.y, last)] * weights.y
         + bone_matrices.m[min(j.z, last)] * weights.z
         + bone_matrices.m[min(j.w, last)] * weights.w;
}

/// A skinned vertex in world space; see `vs_skinned`.
fn skinned_world(in : SkinnedVsIn) -> vec4<f32> {
    let p = morph_position(in.position, in.vertex_index);
    let n = morph_normal(in.normal, in.vertex_index);
    let skinned = skin_matrix(in.joints, in.weights)
        * vec4<f32>(apply_displacement(p, n, transform_uv(in.uv)), 1.0);
    return vec4<f32>(skinned.xyz + (1.0 - skinned.w) * mesh.model[3].xyz, 1.0);
}

// The shadow passes for a skinned caster: the posed (and morphed) mesh, as
// the colour pass draws it — not the bind pose.
@vertex
fn vs_shadow_skinned(in : SkinnedVsIn) -> ShadowOut {
    return shadow_out(shadow_pass.vp * skinned_world(in), in.uv);
}

@vertex
fn vs_shadow_spot_skinned(in : SkinnedVsIn) -> ShadowOut {
    return shadow_out(shadow_pass.vp * skinned_world(in), in.uv);
}

@vertex
fn vs_shadow_point_skinned(in : SkinnedVsIn) -> ShadowPointOut {
    let world = skinned_world(in);
    var o : ShadowPointOut;
    o.position = shadow_pass.vp * world;
    o.world = world.xyz;
    o.uv = transform_uv(in.uv);
    return o;
}

@vertex
fn vs_skinned(in : SkinnedVsIn) -> VsOut {
    var out : VsOut;
    pass_uv_sets(in.vertex_index, &out);
    let skin = skin_matrix(in.joints, in.weights);
    let skin_uv = transform_uv(in.uv);
    // Morph targets first, as three.js runs `morphtarget_vertex` before
    // `skinning_vertex`.
    let morphed_p = morph_position(in.position, in.vertex_index);
    let morphed_n = morph_normal(in.normal, in.vertex_index);
    // three.js `skinning_vertex.glsl` with an identity `bindMatrix` and
    // `bindMatrixInverse` = inverse(model) (attached bind mode):
    //   skinned     = skin * vec4(position, 1)        (w = the weight sum)
    //   transformed = (inverse(model) * skinned).xyz
    //   world       = model * vec4(transformed, 1)
    // which is `skinned.xyz + (1 - skinned.w) * translation(model)` — the
    // model cancels when the weights sum to one, and the leftover is what
    // three.js draws when they do not. See `SkinnedMesh`; the raycaster, the
    // Metal backend and the path tracer compute the same.
    let skinned = skin * vec4<f32>(apply_displacement(morphed_p, morphed_n, skin_uv), 1.0);
    let world_p = vec4<f32>(skinned.xyz + (1.0 - skinned.w) * mesh.model[3].xyz, 1.0);
    out.world_pos = world_p.xyz;
    out.clip_pos = frame.view_proj * world_p;
    out.clip_zw = out.clip_pos.zw;
    // `skinnormal_vertex.glsl`: objectNormal = inverse(model) * skin * n,
    // then the normal matrix. inverse(model)'s linear part is the transpose
    // of the normal matrix (the inverse transpose), so for a rigid model the
    // two cancel and this is `skin * n`.
    let skin_n = (skin * vec4<f32>(morphed_n, 0.0)).xyz;
    let object_n = (transpose(mesh.normal_matrix) * vec4<f32>(skin_n, 0.0)).xyz;
    out.world_normal = (mesh.normal_matrix * vec4<f32>(object_n, 0.0)).xyz;
    out.uv = skin_uv;
    let view_p = frame.view * world_p;
    out.view_z = -view_p.z;
    out.vertex_color = morph_color(in.color, in.vertex_index);
    out.proj_uv = mirror_proj_uv(morphed_p);
    out.world_tangent = world_tangent(in.tangent);
    out.local_pos = morphed_p;
    return out;
}

struct InstancedVsIn {
    @location(0) position : vec3<f32>,
    @location(1) normal   : vec3<f32>,
    @location(2) uv       : vec2<f32>,
    @location(3) color    : vec4<f32>,
    @location(4) imat0    : vec4<f32>,
    @location(5) imat1    : vec4<f32>,
    @location(6) imat2    : vec4<f32>,
    @location(7) imat3    : vec4<f32>,
    @location(8) tangent  : vec4<f32>,
    @location(9) itint    : vec4<f32>,
};

// The shadow passes for an `InstancedMesh` caster: each instance, morphed
// by its own row of the morph texture where it has one, as `vs_instanced`
// draws it — three.js renders an instanced mesh into the shadow map with the
// instanced depth material.
fn instanced_shadow_world(in : InstancedVsIn, vid : u32, iid : u32, displace : bool) -> vec4<f32> {
    let model = mesh.model * mat4x4<f32>(in.imat0, in.imat1, in.imat2, in.imat3);
    var p = morph_position_instance(in.position, vid, iid);
    if (displace) {
        let n = morph_normal_instance(in.normal, vid, iid);
        p = apply_displacement(p, n, transform_uv(in.uv));
    }
    return model * vec4<f32>(p, 1.0);
}

@vertex
fn vs_shadow_instanced(
    in : InstancedVsIn,
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32,
) -> ShadowOut {
    return shadow_out(shadow_pass.vp * instanced_shadow_world(in, vid, iid, true), in.uv);
}

@vertex
fn vs_shadow_spot_instanced(
    in : InstancedVsIn,
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32,
) -> ShadowOut {
    return shadow_out(shadow_pass.vp * instanced_shadow_world(in, vid, iid, false), in.uv);
}

@vertex
fn vs_shadow_point_instanced(
    in : InstancedVsIn,
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32,
) -> ShadowPointOut {
    let world = instanced_shadow_world(in, vid, iid, false);
    var o : ShadowPointOut;
    o.position = shadow_pass.vp * world;
    o.world = world.xyz;
    o.uv = transform_uv(in.uv);
    return o;
}

/// `vs_instanced` as a plain function, as `standard_vertex` is `vs_main`'s:
/// a custom vertex stage on an `InstancedMesh` starts from it.
fn standard_vertex_instanced(in : InstancedVsIn, vid : u32, iid : u32) -> VsOut {
    var out : VsOut;
    pass_uv_sets(vid, &out);
    let instance_mat = mat4x4<f32>(in.imat0, in.imat1, in.imat2, in.imat3);
    let model = mesh.model * instance_mat;
    let uv = transform_uv(in.uv);
    // three.js order: morph targets (shared influences, or this instance's
    // row of the morph texture), then displacement, then the instance matrix.
    let morphed_p = morph_position_instance(in.position, vid, iid);
    let morphed_n = morph_normal_instance(in.normal, vid, iid);
    let world_p = model * vec4<f32>(apply_displacement(morphed_p, morphed_n, uv), 1.0);
    out.world_pos = world_p.xyz;
    out.clip_pos = frame.view_proj * world_p;
    out.clip_zw = out.clip_pos.zw;
    // `defaultnormal_vertex.glsl` under USE_INSTANCING: the instance's linear
    // part, divided by each column's squared length (so a scaled instance
    // keeps its normals perpendicular), then the mesh's normal matrix.
    let im = mat3x3<f32>(in.imat0.xyz, in.imat1.xyz, in.imat2.xyz);
    let inst_n = im * (morphed_n / vec3<f32>(dot(im[0], im[0]), dot(im[1], im[1]), dot(im[2], im[2])));
    let n4 = mesh.normal_matrix * vec4<f32>(inst_n, 0.0);
    out.world_normal = n4.xyz;
    out.uv = uv;
    let view_p = frame.view * world_p;
    out.view_z = -view_p.z;
    // Per-instance tint multiplies the mesh's own vertex colour, so one
    // instanced mesh can carry a whole palette.
    out.vertex_color = morph_color_instance(in.color, vid, iid) * in.itint;
    out.proj_uv = mirror_proj_uv(morphed_p);
    out.world_tangent = world_tangent(in.tangent);
    out.local_pos = morphed_p;
    return out;
}

@vertex
fn vs_instanced(
    in : InstancedVsIn,
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32,
) -> VsOut {
    return standard_vertex_instanced(in, vid, iid);
}

/// What a `ShaderMaterial` on an `InstancedMesh` reads: the instanced
/// inputs, plus the mesh's two per-instance custom columns
/// (`InstancedMesh::set_custom`), each zero-padded to a `vec4`. A custom
/// vertex stage there is
///
///     @vertex
///     fn vs_custom(in: InstancedCustomIn, @builtin(vertex_index) vid: u32,
///                  @builtin(instance_index) iid: u32) -> VsOut {
///         var out = standard_vertex_instanced(instanced_in(in), vid, iid);
///         out.vertex_color = vec4<f32>(in.icustom0.rgb, 1.0);
///         return out;
///     }
struct InstancedCustomIn {
    @location(0) position : vec3<f32>,
    @location(1) normal   : vec3<f32>,
    @location(2) uv       : vec2<f32>,
    @location(3) color    : vec4<f32>,
    @location(4) imat0    : vec4<f32>,
    @location(5) imat1    : vec4<f32>,
    @location(6) imat2    : vec4<f32>,
    @location(7) imat3    : vec4<f32>,
    @location(8) tangent  : vec4<f32>,
    @location(9) itint    : vec4<f32>,
    @location(10) icustom0 : vec4<f32>,
    @location(11) icustom1 : vec4<f32>,
};

/// The standard instanced inputs out of `InstancedCustomIn`.
fn instanced_in(in : InstancedCustomIn) -> InstancedVsIn {
    var o : InstancedVsIn;
    o.position = in.position;
    o.normal = in.normal;
    o.uv = in.uv;
    o.color = in.color;
    o.imat0 = in.imat0;
    o.imat1 = in.imat1;
    o.imat2 = in.imat2;
    o.imat3 = in.imat3;
    o.tangent = in.tangent;
    o.itint = in.itint;
    return o;
}

/// The vertex entry of a `ShaderMaterial` on an `InstancedMesh` that brings
/// no vertex stage of its own.
@vertex
fn vs_instanced_custom(
    in : InstancedCustomIn,
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32,
) -> VsOut {
    return standard_vertex_instanced(instanced_in(in), vid, iid);
}

@vertex
fn vs_sprite(in : VsIn) -> VsOut {
    var out : VsOut;
    let center_world = vec4<f32>(mesh.model[3].xyz, 1.0);
    let center_view = frame.view * center_world;
    // three.js `sprite_vert`. `params8` holds (center - 0.5, _, rotation)
    // and `params13.x` is 1 with size attenuation off. The raycaster builds
    // the same quad (`Raycaster::raycast_sprite`).
    var scale = vec2<f32>(length(mesh.model[0].xyz), length(mesh.model[1].xyz));
    // A perspective projection's last column is (0, 0, -1, 0); the
    // raycaster tells the two apart the same way.
    if (mesh.params13.x > 0.5 && frame.projection[3][3] == 0.0) {
        scale = scale * -center_view.z;
    }
    let aligned = (in.position.xy - mesh.params8.xy) * scale;
    let rotation = mesh.params8.w;
    let c = cos(rotation);
    let s = sin(rotation);
    let rotated = vec2<f32>(c * aligned.x - s * aligned.y, s * aligned.x + c * aligned.y);
    let view_pos = vec4<f32>(center_view.xy + rotated, center_view.z, 1.0);
    out.clip_pos = frame.projection * view_pos;
    out.clip_zw = out.clip_pos.zw;
    out.world_pos = center_world.xyz;
    out.world_normal = vec3<f32>(0.0, 0.0, 1.0);
    out.uv = in.position.xy + vec2<f32>(0.5, 0.5);
    out.view_z = -view_pos.z;
    out.vertex_color = in.color;
    out.proj_uv = mirror_proj_uv(in.position);
    out.world_tangent = world_tangent(in.tangent);
    out.local_pos = in.position;
    return out;
}

// ---- helpers ----

fn punctual_attenuation(d : f32, max_d : f32, decay : f32) -> f32 {
    // Match three.js `getDistanceAttenuation` (lights_pars_begin.glsl.js):
    // 1.0 / max(pow(lightDistance, decayExponent), 0.01)
    var att = 1.0 / max(pow(d, decay), 0.01);
    if (max_d > 0.0) {
        let t = clamp(1.0 - pow(d / max_d, 4.0), 0.0, 1.0);
        att = att * t * t;
    }
    return att;
}

// three.js r165 `ACESFilmicToneMapping` (Stephen Hill's RRT + ODT fit, with
// three.js's 1/0.6 pre-scale) on an already exposed colour. It used to be
// Narkowicz's single-curve fit, which reads 0.267 for mid grey where three.js
// reads 0.213 — a visibly brighter, more saturated picture than the reference.
fn aces_tonemap(x : vec3<f32>) -> vec3<f32> {
    let aces_input = mat3x3<f32>(
        vec3<f32>(0.59719, 0.07600, 0.02840),
        vec3<f32>(0.35458, 0.90834, 0.13383),
        vec3<f32>(0.04823, 0.01566, 0.83777),
    );
    let aces_output = mat3x3<f32>(
        vec3<f32>(1.60475, -0.10208, -0.00327),
        vec3<f32>(-0.53108, 1.10813, -0.07276),
        vec3<f32>(-0.07367, -0.00605, 1.07602),
    );
    let v = aces_input * (x / 0.6);
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return clamp(aces_output * (a / b), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn gamma_to_linear(c : vec3<f32>) -> vec3<f32> { return pow(c, vec3<f32>(2.2)); }
fn linear_to_gamma(c : vec3<f32>) -> vec3<f32> { return pow(c, vec3<f32>(1.0 / 2.2)); }

// Khronos PBR Neutral. Almost the identity below the shoulder — a small black
// offset and nothing else — so an authored colour reaches the screen as itself
// rather than restyled. Only the top is compressed, desaturating as it goes so
// an over-exposed highlight resolves towards white.
fn neutral_tonemap(c_in : vec3<f32>) -> vec3<f32> {
    let start_compression = 0.8 - 0.04;
    let desaturation = 0.15;
    var c = c_in;
    let x = min(c.r, min(c.g, c.b));
    var offset = 0.04;
    if (x < 0.08) { offset = x - 6.25 * x * x; }
    c = c - vec3<f32>(offset);
    let peak = max(c.r, max(c.g, c.b));
    if (peak < start_compression) { return c; }
    let d = 1.0 - start_compression;
    let new_peak = 1.0 - d * d / (peak + d - start_compression);
    c = c * (new_peak / max(peak, 1e-6));
    let g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return mix(c, vec3<f32>(new_peak), g);
}

const LINEAR_SRGB_TO_REC2020 = mat3x3<f32>(
    vec3<f32>(0.6274, 0.0691, 0.0164),
    vec3<f32>(0.3293, 0.9195, 0.0880),
    vec3<f32>(0.0433, 0.0113, 0.8956),
);
const REC2020_TO_LINEAR_SRGB = mat3x3<f32>(
    vec3<f32>( 1.6605, -0.1246, -0.0182),
    vec3<f32>(-0.5876,  1.1329, -0.1006),
    vec3<f32>(-0.0728, -0.0083,  1.1187),
);

fn agx_contrast(x : vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2
        - 40.14 * x4 * x
        + 31.96 * x4
        - 6.868 * x2 * x
        + 0.4298 * x2
        + 0.1191 * x
        - 0.00232;
}

// AgX, as three.js ships it. A long shoulder — 3.0 lands at 0.82 where ACES is
// already at 0.95 — so a very bright environment keeps detail in its highlights.
// It desaturates the shadows noticeably in exchange; that is the look.
fn agx_tonemap(c_in : vec3<f32>) -> vec3<f32> {
    let inset = mat3x3<f32>(
        vec3<f32>(0.856627153315983, 0.137318972929847, 0.11189821299995),
        vec3<f32>(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
        vec3<f32>(0.0482516061458583, 0.101439036467562, 0.811302368396859),
    );
    let outset = mat3x3<f32>(
        vec3<f32>( 1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
        vec3<f32>(-0.11060664309660323,  1.157823702216272, -0.11060664309660294),
        vec3<f32>(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405),
    );
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var c = inset * (LINEAR_SRGB_TO_REC2020 * c_in);
    c = max(c, vec3<f32>(1e-10));
    c = (log2(c) - vec3<f32>(min_ev)) / (max_ev - min_ev);
    c = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    c = agx_contrast(c);
    c = outset * c;
    c = pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2));
    return clamp(REC2020_TO_LINEAR_SRGB * c, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Exposure, then whichever curve the renderer is set to.
//
// `tone_mapping_exposure.x`: 0 none, 1 linear, 2 ACES filmic, 3 Khronos PBR
// neutral, 4 AgX. A `ShaderMaterial` should end with
// `framebuffer_encode(apply_tone_mapping(hdr))`, which is exactly what the
// built-in fragment does — otherwise a custom surface is the one thing in the
// frame that does not respond to `Renderer::set_tone_mapping`.
//
// A material with `toneMapped: false` (FLAG_NO_TONE_MAP) is left alone, as
// three.js compiles its program without TONE_MAPPING.
fn apply_tone_mapping(c : vec3<f32>) -> vec3<f32> {
    if ((mesh.flags.z & FLAG_NO_TONE_MAP) != 0u) { return c; }
    let tm = u32(frame.tone_mapping_exposure.x);
    let exposure = frame.tone_mapping_exposure.y;
    if (tm == 0u) { return c; }
    let e = c * exposure;
    if (tm == 1u) { return e; }
    if (tm == 2u) { return aces_tonemap(e); }
    if (tm == 3u) { return neutral_tonemap(e); }
    return agx_tonemap(e);
}

// IEC 61966-2-1 sRGB OETF — matches three.js `sRGBTransferOETF`.
fn linear_to_srgb(c : vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let lo = c * 12.92;
    let hi = pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) * 1.055 - vec3<f32>(0.055);
    return mix(hi, lo, step(c, cutoff));
}

// tone_mapping_exposure.w > 0.5: write linear (HalfFloat / postfx RT), else
// sRGB-encode for the canvas; 2 rather than 1 when the target is an sRGB
// format, which encodes on write (see `framebuffer_data`).
//
// The float path deliberately does *not* clamp. It used to, which quietly made
// a half-float target behave like an 8-bit one: anything above 1.0 was gone
// before the post chain saw it, so bloom had nothing to threshold and a custom
// shader had no way to hand HDR onwards. Clamping is right for the 8-bit canvas
// — the encode is only defined on [0, 1] — and wrong for a format chosen
// because the values exceed it.
fn framebuffer_encode(c: vec3<f32>) -> vec3<f32> {
    if ((mesh.flags.z & FLAG_ENCODED_OUT) != 0u) {
        return linear_to_srgb(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
    }
    if (frame.tone_mapping_exposure.w > 0.5) {
        return max(c, vec3<f32>(0.0));
    }
    return linear_to_srgb(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
}

// `framebuffer_encode` backwards: a texel of a target drawn like this one
// (the screen-space glass capture) as the tone-mapped linear value it holds.
// A hardware sRGB target decodes on sample, so it comes back as it is.
fn framebuffer_decode(c: vec3<f32>) -> vec3<f32> {
    if ((mesh.flags.z & FLAG_ENCODED_OUT) == 0u && frame.tone_mapping_exposure.w > 0.5) {
        return c;
    }
    let lo = c / 12.92;
    let hi = pow((max(c, vec3<f32>(0.0)) + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return mix(hi, lo, step(c, vec3<f32>(0.04045)));
}

// Data rather than colour — MeshDepthMaterial's depth, MeshNormalMaterial's
// packed normal — which three.js writes unconverted, so the target stores the
// value itself. `tone_mapping_exposure.w > 1.5` says the target is an sRGB
// format (every 8-bit target on WebGL2), which would encode it on write: it is
// decoded first so the stored byte is the value, as on a `Unorm` target.
fn framebuffer_data(v: vec3<f32>) -> vec3<f32> {
    if ((mesh.flags.z & FLAG_ENCODED_OUT) != 0u) {
        return v;
    }
    if (frame.tone_mapping_exposure.w > 1.5) {
        let x = clamp(v, vec3<f32>(0.0), vec3<f32>(1.0));
        return select(pow((x + 0.055) / 1.055, vec3<f32>(2.4)), x / 12.92, x <= vec3<f32>(0.04045));
    }
    return v;
}

fn view_z_to_depth_tex(view_z_pos: f32, near: f32, far: f32) -> f32 {
    let view_z = -view_z_pos;
    return (far + near * far / view_z) / (far - near);
}

// Apply fog (linear or exp2) to a linear-space color based on view-space depth.
// Mirrors three.js's fog: linear factor = smoothstep(near, far, dist),
// exp2 factor = 1 - exp(-density^2 * dist^2).
// How much fog covers a point `view_z` away: 0 for none, 1 for all fog.
fn fog_factor(view_z : f32) -> f32 {
    let mode = frame.fog_params.w;
    if (mode < 0.5) { return 0.0; }
    let dist = max(view_z, 0.0);
    if (mode < 1.5) {
        let near = frame.fog_params.x;
        let far  = frame.fog_params.y;
        return clamp((dist - near) / (far - near), 0.0, 1.0);
    }
    let d = frame.fog_params.z;
    return 1.0 - exp(-d * d * dist * dist);
}

fn apply_fog(color : vec3<f32>, view_z : f32) -> vec3<f32> {
    return mix(color, frame.fog_color.rgb, fog_factor(view_z));
}

// ---- Shadows --------------------------------------------------------------
//
// Every shadow-casting light owns a `ShadowRecord` and names it in its own
// `color.w`, so a light loop asks `shadow_at` for any kind of light the same
// way. Each kind's maps are one depth array — a layer per light, a cascade
// each for a directional light with several, six (its cube faces) for a
// point light — read through comparison samplers only: WebGL2 cannot read a
// depth texture any other way, so even the PCSS blocker search compares.

// The record a light names in `color.w` (−1: it casts none).
fn light_shadow_record(color_w : f32) -> i32 {
    return i32(color_w + 0.5) - 1;
}

// Every record's shadow at this fragment, worked out once by
// `compute_shadows` and then read by each light that names it. One call site
// for the whole filter machinery rather than one per light loop: a shading
// language inlines every call, and on WebGL2's software rasteriser eight
// inlined copies of the PCSS search made the lit shaders take tens of
// seconds to compile.
var<private> shadow_cache : array<f32, 16>;

fn compute_shadows(world_pos : vec3<f32>, world_normal : vec3<f32>) {
    for (var r = 0u; r < 16u; r = r + 1u) {
        shadow_cache[r] = 1.0;
    }
    let count = min(frame.shadow_info.x, 16u);
    for (var r = 0u; r < count; r = r + 1u) {
        shadow_cache[r] = shadow_at(i32(r), world_pos, world_normal);
    }
}

// The shadow of the light whose `color.w` is `color_w`, from the cache.
fn light_shadow(color_w : f32) -> f32 {
    let r = light_shadow_record(color_w);
    if (r < 0 || r >= 16) {
        return 1.0;
    }
    return shadow_cache[r];
}

// Light `i` of each kind: 1 lit, 0 shadowed (after `compute_shadows`).
fn shadow_factor_dir(i : u32) -> f32 {
    return light_shadow(frame.dir_lights[i].color.w);
}
fn shadow_factor_spot(i : u32) -> f32 {
    return light_shadow(frame.spot_lights[i].color.w);
}
fn shadow_factor_point(i : u32) -> f32 {
    return light_shadow(frame.point_lights[i].color.w);
}

// One compare in layer `layer` of kind `kind`'s map array.
fn shadow_compare(kind : u32, uv : vec2<f32>, layer : i32, depth : f32) -> f32 {
    if (kind == 0u) {
        return textureSampleCompareLevel(shadow_tex, shadow_sampler, uv, layer, depth);
    }
    if (kind == 1u) {
        return textureSampleCompareLevel(spot_shadow_tex, spot_shadow_sampler, uv, layer, depth);
    }
    return textureSampleCompareLevel(point_shadow_tex, point_shadow_sampler, uv, layer, depth);
}

// One filter tap, offset `o` from the lookup's centre. A 2D map moves the uv
// (`centre.xy`); a point light's cube turns the direction (`centre`) across
// `t` and `b` and picks the face the tap lands on, so a filter crosses cube
// edges without a seam.
fn shadow_tap(kind : u32, layer : i32, centre : vec3<f32>, t : vec3<f32>, b : vec3<f32>, o : vec2<f32>, depth : f32) -> f32 {
    if (kind == 2u) {
        let f = point_shadow_face(centre + t * o.x + b * o.y);
        return shadow_compare(2u, f.xy, layer + i32(f.z), depth);
    }
    return shadow_compare(kind, centre.xy + o, layer, depth);
}

// Percentage-closer filtering over a disc `r` wide: `taps` compares on a
// Vogel spiral, turned per point in the world so a wide penumbra shows no
// banding and does not crawl as the camera moves. A single compare gives a
// binary answer per pixel — the map's own texel grid, magnified — and the
// average turns it into a gradient: anti-aliasing at a texel's radius, a
// penumbra wider.
fn shadow_pcf(kind : u32, layer : i32, centre : vec3<f32>, t : vec3<f32>, b : vec3<f32>, depth : f32, r : f32, taps : u32, world_pos : vec3<f32>) -> f32 {
    if (r <= 0.0 || taps <= 1u) {
        return shadow_tap(kind, layer, centre, t, b, vec2<f32>(0.0), depth);
    }
    let phi = shadow_disc_rotation(world_pos);
    var sum = 0.0;
    for (var k = 0u; k < taps; k = k + 1u) {
        let o = shadow_disc_tap(k, taps, phi) * r * 1.5;
        sum = sum + shadow_tap(kind, layer, centre, t, b, o, depth);
    }
    return sum / f32(taps);
}

// Percentage-closer soft shadows (Fernando 2005): the penumbra widens with
// the gap between the receiver and what blocks it, so a shadow is sharp
// where an object touches the ground and soft further off.
//
// The blocker search usually reads depths. Here it compares, because a
// depth texture is only readable through a comparison sampler on WebGL2:
// compares against four depths between the light and the receiver give the
// fraction of the search disc blocked nearer than each, a cumulative
// distribution of blocker depth, and its mean comes out of integrating it.
// Then a PCF as wide as that mean blocker says.
fn shadow_pcss(rec : ShadowRecord, layer : i32, centre : vec3<f32>, t : vec3<f32>, b : vec3<f32>, depth : f32, world_pos : vec3<f32>) -> f32 {
    let kind = rec.info.x;
    let texel = rec.params.w;
    let search = max(rec.pcss.x, 2.0 * texel);
    let phi = shadow_disc_rotation(world_pos);
    var blocked = vec4<f32>(0.0);
    for (var k = 0u; k < 12u; k = k + 1u) {
        let o = shadow_disc_tap(k, 12u, phi) * search;
        for (var l = 0u; l < 4u; l = l + 1u) {
            let z = depth * f32(l + 1u) * 0.25;
            blocked[l] = blocked[l] + 1.0 - shadow_tap(kind, layer, centre, t, b, o, z);
        }
    }
    blocked = blocked / 12.0;
    if (blocked.w <= 0.0) {
        return 1.0;
    }
    // E[d | d < depth] = depth − ∫₀^depth F(z)/F(depth) dz, trapezoids.
    let integral = depth * 0.25 * (blocked.x + blocked.y + blocked.z + 0.5 * blocked.w) / blocked.w;
    let blocker = max(depth - integral, 1e-4);
    // To linear distance from the light, for the similar triangles: an
    // orthographic map is linear already, a spot's is perspective depth, a
    // point light's is distance over radius.
    var gap = 0.0;
    if (kind == 0u) {
        gap = (depth - blocker) * rec.pcss.w;
    } else if (kind == 1u) {
        let n = rec.pcss.y;
        let f = rec.pcss.z;
        let dr = n * f / max(f - depth * (f - n), 1e-6);
        let db = n * f / max(f - blocker * (f - n), 1e-6);
        gap = (dr - db) / max(db * dr, 1e-6) * rec.pcss.w;
    } else {
        let radius = rec.extra.w;
        let dr = depth * radius;
        let db = blocker * radius;
        gap = (dr - db) / max(db * dr, 1e-6) * rec.pcss.w;
    }
    let r = clamp(gap, texel, 64.0 * texel);
    return shadow_pcf(kind, layer, centre, t, b, depth, r, 16u, world_pos);
}

// The record's filter at one lookup.
fn shadow_filter(rec : ShadowRecord, layer : i32, centre : vec3<f32>, t : vec3<f32>, b : vec3<f32>, depth : f32, world_pos : vec3<f32>) -> f32 {
    let kind = rec.info.x;
    let r = rec.params.z;
    switch (rec.info.w) {
        case 0u: {
            return shadow_tap(kind, layer, centre, t, b, vec2<f32>(0.0), depth);
        }
        case 2u: {
            return shadow_pcf(kind, layer, centre, t, b, depth, max(r * 2.0, 2.0 * rec.params.w), 24u, world_pos);
        }
        case 3u: {
            return shadow_pcss(rec, layer, centre, t, b, depth, world_pos);
        }
        default: {
            return shadow_pcf(kind, layer, centre, t, b, depth, r, SHADOW_TAPS, world_pos);
        }
    }
}

// Record `r`'s shadow at a point: 1 lit, 0 shadowed, 1 for no record.
//
// NORMAL BIAS: the lookup steps off the surface along the normal first. A
// constant depth bias has to cover the worst case — a face nearly edge-on
// to the light, where one shadow texel spans a long slope — and sized for
// that it is far too much everywhere else, so shadows visibly detach from
// their casters. Moving along the normal scales with the geometry instead.
fn shadow_at(r : i32, world_pos : vec3<f32>, world_normal : vec3<f32>) -> f32 {
    if (r < 0 || u32(r) >= frame.shadow_info.x) {
        return 1.0;
    }
    let rec = frame.shadows[r];
    let kind = rec.info.x;
    let biased = world_pos + world_normal * rec.params.y;
    let layer = i32(rec.info.y);
    if (kind == 2u) {
        let radius = rec.extra.w;
        let to_frag = biased - rec.extra.xyz;
        let dist = length(to_frag);
        if (dist > radius) { return 1.0; }
        // Half a percent of the radius, at most a quarter of a unit: a light
        // with no `distance` normalises by its shadow camera's far plane,
        // 500 by default.
        let depth = dist / radius - min(0.005, 0.25 / radius) - rec.params.x;
        let dir = to_frag / max(dist, 1e-6);
        let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(dir.y) > 0.9);
        let t = normalize(cross(dir, helper));
        return shadow_filter(rec, layer, dir, t, cross(dir, t), depth, world_pos);
    }
    // A directional light's cascade: the first whose far edge lies beyond
    // this point's view depth. Past the last, unshadowed.
    var cascade = 0u;
    if (kind == 0u && rec.info.z > 1u) {
        let view_depth = -(frame.view * vec4<f32>(world_pos, 1.0)).z;
        cascade = rec.info.z;
        for (var c = 0u; c < rec.info.z; c = c + 1u) {
            if (view_depth <= rec.extra[c]) {
                cascade = c;
                break;
            }
        }
        if (cascade >= rec.info.z) { return 1.0; }
    }
    let p = rec.vp[cascade] * vec4<f32>(biased, 1.0);
    let ndc = p.xyz / max(p.w, 0.0001);
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { return 1.0; }
    if (ndc.z < 0.0 || ndc.z > 1.0) { return 1.0; }
    let zero = vec3<f32>(0.0);
    return shadow_filter(rec, layer + i32(cascade), vec3<f32>(uv, 0.0), zero, zero, ndc.z - rec.params.x, world_pos);
}

// How many compares a PCF lookup takes: three more than the 3×3 grid it
// replaced, spread over a disc instead of a square.
const SHADOW_TAPS : u32 = 12u;

// Tap `k` of `n` on a Vogel disc — a golden-angle spiral, evenly spread at
// any tap count — turned by `phi`, inside the unit circle.
fn shadow_disc_tap(k : u32, n : u32, phi : f32) -> vec2<f32> {
    let r = sqrt((f32(k) + 0.5) / f32(n));
    let t = f32(k) * 2.39996323 + phi;
    return vec2<f32>(cos(t), sin(t)) * r;
}

// The disc's turn at a point: varied so the taps' pattern does not show as
// banding in a wide penumbra, and fixed to the world so it does not crawl as
// the camera moves.
fn shadow_disc_rotation(p : vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453) * 6.28318530718;
}

// The cube face a direction lands on and its uv there: the selection every
// API's cube sampler makes (major axis, then the face's `sc`/`tc` table),
// done by hand because the faces live in a 2D array — WebGL2 has no cube
// arrays. Faces are +X −X +Y −Y +Z −Z, as the point passes draw them.
fn point_shadow_face(d : vec3<f32>) -> vec3<f32> {
    let a = abs(d);
    var face = 0.0;
    var sc = 0.0;
    var tc = 0.0;
    var ma = 1.0;
    if (a.x >= a.y && a.x >= a.z) {
        ma = a.x;
        if (d.x > 0.0) { face = 0.0; sc = -d.z; } else { face = 1.0; sc = d.z; }
        tc = -d.y;
    } else if (a.y >= a.z) {
        ma = a.y;
        sc = d.x;
        if (d.y > 0.0) { face = 2.0; tc = d.z; } else { face = 3.0; tc = -d.z; }
    } else {
        ma = a.z;
        if (d.z > 0.0) { face = 4.0; sc = d.x; } else { face = 5.0; sc = -d.x; }
        tc = -d.y;
    }
    return vec3<f32>(0.5 * (sc / ma + 1.0), 0.5 * (tc / ma + 1.0), face);
}

// three.js `getShadowMask()`: every shadow-casting light's shadow, multiplied.
fn shadow_mask(world_pos : vec3<f32>, world_normal : vec3<f32>) -> f32 {
    compute_shadows(world_pos, world_normal);
    var m = 1.0;
    for (var r = 0u; r < 16u; r = r + 1u) {
        m = m * shadow_cache[r];
    }
    return m;
}

// ---- Lambert / Phong / Toon: three.js r165's non-physical lit models ----
//
// All three share three.js's light loop (`lights_fragment_begin`): each direct
// light gives an `IncidentLight` — a direction and a colour already
// attenuated by distance, cone and shadow — and the model turns it into
// diffuse (`irradiance · BRDF_Lambert`, i.e. ÷π) and, for Phong, specular.
// Ambient and hemisphere irradiance go through the same `BRDF_Lambert`
// (`RE_IndirectDiffuse_*`), so the indirect terms are ÷π as well.

/// One direct light as three.js's `IncidentLight`: the direction to it and
/// its colour after attenuation (distance, spot cone — the caller applies
/// shadows).
struct Incident {
    dir   : vec3<f32>,
    color : vec3<f32>,
};

fn incident_point(world_pos : vec3<f32>, l : PointLight) -> Incident {
    let to_light_vec = l.position.xyz - world_pos;
    let d = length(to_light_vec);
    return Incident(to_light_vec / max(d, 0.0001), l.color.rgb * punctual_attenuation(d, l.params.x, l.params.y));
}

fn incident_spot(world_pos : vec3<f32>, l : SpotLight) -> Incident {
    let to_light_vec = l.position.xyz - world_pos;
    let d = length(to_light_vec);
    let to_light = to_light_vec / max(d, 0.0001);
    let cos_angle = dot(-to_light, normalize(l.direction.xyz));
    var cone = 0.0;
    if (cos_angle > l.params.z) {
        cone = smoothstep(l.params.z, l.params.w, cos_angle);
    }
    return Incident(to_light, l.color.rgb * punctual_attenuation(d, l.params.x, l.params.y) * cone);
}

/// three.js `BRDF_BlinnPhong`: `F_Schlick(specular, 1, v·h) · ¼ ·
/// D_BlinnPhong`, with `D_BlinnPhong = (s/2 + 1)/π · (n·h)^s`.
fn brdf_blinn_phong(l : vec3<f32>, v : vec3<f32>, n : vec3<f32>, specular : vec3<f32>, shininess : f32) -> vec3<f32> {
    let h = normalize(l + v);
    let n_dot_h = clamp(dot(n, h), 0.0, 1.0);
    let v_dot_h = clamp(dot(v, h), 0.0, 1.0);
    let d = (1.0 / PI) * (shininess * 0.5 + 1.0) * pow(n_dot_h, shininess);
    return f_schlick(v_dot_h, specular) * (0.25 * d);
}

/// three.js `getGradientIrradiance`, at `coord = n·l/2 + ½`. With a
/// `gradientMap`, its rgb there (`dx`, `dy`: `coord`'s screen derivatives,
/// passed in because the call sits in the light loop). Without one, 0.7 below
/// `coord = 0.7` and 1.0 above, the step antialiased over `fw`
/// (`fwidth(coord)/2`). Not clamped to the lit side: a toon surface facing
/// away from a light still gets the ramp's lowest tone of it, as in three.js.
fn toon_gradient(n_dot_l : f32, fw : f32, dx : f32, dy : f32) -> vec3<f32> {
    let coord = n_dot_l * 0.5 + 0.5;
    if (HAS_GRADIENT_MAP) {
        return sample_gradient_map_grad(coord, dx, dy).rgb;
    }
    return vec3<f32>(mix(0.7, 1.0, smoothstep(0.7 - fw, 0.7 + fw, coord)));
}

fn shade_hemi(n : vec3<f32>, l : HemiLight) -> vec3<f32> {
    let up = normalize(l.direction.xyz);
    let t = dot(n, up) * 0.5 + 0.5;
    return mix(l.ground_color.rgb, l.sky_color.rgb, t);
}

// ---- PBR (Cook-Torrance GGX) ----

fn d_ggx(n_dot_h : f32, a : f32) -> f32 {
    let a2 = a * a;
    let f = (n_dot_h * a2 - n_dot_h) * n_dot_h + 1.0;
    return a2 / (PI * f * f);
}

// three.js `V_GGX_SmithCorrelated`: height-correlated Smith visibility, which
// already carries the 1/(4·NoV·NoL) of the microfacet denominator — so a
// specular lobe is `D · V · F` with no further divide.
fn v_ggx_smith_correlated(a : f32, n_dot_l : f32, n_dot_v : f32) -> f32 {
    let a2 = a * a;
    let gv = n_dot_l * sqrt(a2 + (1.0 - a2) * n_dot_v * n_dot_v);
    let gl = n_dot_v * sqrt(a2 + (1.0 - a2) * n_dot_l * n_dot_l);
    return 0.5 / max(gv + gl, 1e-6);
}

// three.js `F_Schlick(f0, 1.0, dotVH)`, in its spherical-Gaussian form
// (`exp2((-5.55473·x − 6.98316)·x)` for `(1 − x)^5`), which is what r165 runs.
fn f_schlick(v_dot_h : f32, f0 : vec3<f32>) -> vec3<f32> {
    let x = clamp(v_dot_h, 0.0, 1.0);
    let fresnel = exp2((-5.55473 * x - 6.98316) * x);
    return f0 * (1.0 - fresnel) + vec3<f32>(fresnel);
}

// PMREM CubeUV sampling — ports three.js cube_uv_reflection_fragment.
const CUBEUV_MIN_MIP_LEVEL : f32 = 4.0;
const CUBEUV_MIN_TILE_SIZE : f32 = 16.0;

fn cube_uv_get_face(direction : vec3<f32>) -> f32 {
    let abs_dir = abs(direction);
    var face = -1.0;
    if (abs_dir.x > abs_dir.z) {
        if (abs_dir.x > abs_dir.y) {
            face = select(3.0, 0.0, direction.x > 0.0);
        } else {
            face = select(4.0, 1.0, direction.y > 0.0);
        }
    } else {
        if (abs_dir.z > abs_dir.y) {
            face = select(5.0, 2.0, direction.z > 0.0);
        } else {
            face = select(4.0, 1.0, direction.y > 0.0);
        }
    }
    return face;
}

fn cube_uv_get_uv(direction : vec3<f32>, face : f32) -> vec2<f32> {
    var uv = vec2<f32>(0.0);
    if (face == 0.0) {
        uv = vec2<f32>(direction.z, direction.y) / abs(direction.x);
    } else if (face == 1.0) {
        uv = vec2<f32>(-direction.x, -direction.z) / abs(direction.y);
    } else if (face == 2.0) {
        uv = vec2<f32>(-direction.x, direction.y) / abs(direction.z);
    } else if (face == 3.0) {
        uv = vec2<f32>(-direction.z, direction.y) / abs(direction.x);
    } else if (face == 4.0) {
        uv = vec2<f32>(-direction.x, direction.z) / abs(direction.y);
    } else {
        uv = vec2<f32>(direction.x, direction.y) / abs(direction.z);
    }
    return 0.5 * (uv + 1.0);
}

fn bilinear_cube_uv(direction : vec3<f32>, mip_int : f32) -> vec3<f32> {
    let cube_uv_max_mip = env_u.params.z;
    let texel_w = env_u.params.x;
    let texel_h = env_u.params.y;
    var face = cube_uv_get_face(direction);
    let filter_int = max(CUBEUV_MIN_MIP_LEVEL - mip_int, 0.0);
    var mip_i = max(mip_int, CUBEUV_MIN_MIP_LEVEL);
    let face_size = exp2(mip_i);
    var uv = cube_uv_get_uv(direction, face) * (face_size - 2.0) + 1.0;
    if (face > 2.0) {
        uv.y += face_size;
        face -= 3.0;
    }
    uv.x += face * face_size;
    uv.x += filter_int * 3.0 * CUBEUV_MIN_TILE_SIZE;
    uv.y += 4.0 * (exp2(cube_uv_max_mip) - face_size);
    uv.x *= texel_w;
    uv.y *= texel_h;
    // Shader UV math is bottom-left (three.js); atlas rows are top-first (WebGPU).
    uv.y = 1.0 - uv.y;
    return textureSampleLevel(env_cube_uv_tex, env_sampler, uv, 0.0).rgb;
}

fn roughness_to_mip(roughness : f32) -> f32 {
    var mip = 0.0;
    if (roughness >= 0.8) {
        mip = (1.0 - roughness) * (-1.0 - (-2.0)) / (1.0 - 0.8) + (-2.0);
    } else if (roughness >= 0.4) {
        mip = (0.8 - roughness) * (2.0 - (-1.0)) / (0.8 - 0.4) + (-1.0);
    } else if (roughness >= 0.305) {
        mip = (0.4 - roughness) * (3.0 - 2.0) / (0.4 - 0.305) + 2.0;
    } else if (roughness >= 0.21) {
        mip = (0.305 - roughness) * (4.0 - 3.0) / (0.305 - 0.21) + 3.0;
    } else {
        mip = -2.0 * log2(1.16 * max(roughness, 0.001));
    }
    return mip;
}

fn texture_cube_uv(sample_dir : vec3<f32>, roughness : f32) -> vec3<f32> {
    let cube_uv_m0 = -2.0;
    let cube_uv_max_mip = env_u.params.z;
    let mip = clamp(roughness_to_mip(roughness), cube_uv_m0, cube_uv_max_mip);
    let mip_f = fract(mip);
    let mip_i = floor(mip);
    let color0 = bilinear_cube_uv(sample_dir, mip_i);
    if (mip_f <= 0.0) {
        return color0;
    }
    let color1 = bilinear_cube_uv(sample_dir, mip_i + 1.0);
    return mix(color0, color1, mip_f);
}

// three.js `getIBLRadiance` / `getIBLIrradiance`'s lookup: the direction
// through `envMapRotation`, the result times `envMapIntensity`.
fn sample_env_cube(dir : vec3<f32>, roughness : f32) -> vec3<f32> {
    let d = vec3<f32>(dot(env_u.rot[0].xyz, dir), dot(env_u.rot[1].xyz, dir), dot(env_u.rot[2].xyz, dir));
    return sample_env_cube_unrotated(d, roughness) * env_u.info.y;
}

fn sample_env_cube_unrotated(dir : vec3<f32>, roughness : f32) -> vec3<f32> {
    if (env_u.params.w > 0.5) {
        return texture_cube_uv(dir, roughness);
    }
    // A plain cube, with no PMREM atlas to index by roughness — so walk its own
    // mip chain instead.
    //
    // This used to be `(roughness + 2) / 10 * 7`, which maps the whole of
    // 0..1 onto mips 1.40..2.10: seven tenths of a single level. A mirror and
    // a fully rough surface came back the same blur, and neither was the right
    // one. Squaring is the usual perceptual fit — roughness is authored
    // perceptually and the filter width goes as alpha — and the chain's real
    // length is asked for rather than assumed to be eight.
    // The chain length comes from the CPU rather than from `textureNumLevels`,
    // which GLSL ES 3.0 has no equivalent for and which therefore cannot be
    // lowered for WebGL2. It is not recoverable from `textureDimensions`
    // either: a PMREM chain is shorter than a full one.
    let levels = env_u.params.z;
    let mip = clamp(roughness * roughness * levels, 0.0, levels);
    let mip_i = floor(mip);
    let mip_f = mip - mip_i;
    let c0 = textureSampleLevel(env_tex, env_sampler, dir, mip_i).rgb;
    if (mip_f <= 0.0) {
        return c0;
    }
    let c1 = textureSampleLevel(env_tex, env_sampler, dir, mip_i + 1.0).rgb;
    return mix(c0, c1, mip_f);
}

// three.js `DFGApprox()` — split-sum approximation used by MeshStandardMaterial IBL.
fn dfg_approx(normal : vec3<f32>, view_dir : vec3<f32>, roughness : f32) -> vec2<f32> {
    let dot_nv = clamp(dot(normal, view_dir), 0.0, 1.0);
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * dot_nv)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

fn pbr_brdf(n : vec3<f32>, v : vec3<f32>, l : vec3<f32>, albedo : vec3<f32>, roughness : f32, metalness : f32, f0 : vec3<f32>) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    // three.js's GGX uses alpha = roughness² as the perceived-roughness
    // remapping (so user-facing roughness stays linear-ish). Without this,
    // low-roughness highlights are far too broad (e.g. r=0.15 was ~44× wider).
    let a = max(roughness * roughness, 0.0016);
    let f = f_schlick(v_dot_h, f0);
    let d = d_ggx(n_dot_h, a);
    // three.js r165 `RE_Direct_Physical`: `irradiance · (BRDF_GGX +
    // BRDF_Lambert(diffuseColor))` with `diffuseColor = color · (1 − metalness)`
    // — the diffuse lobe is not weighted by (1 − F).
    let spec = d * v_ggx_smith_correlated(a, n_dot_l, n_dot_v) * f;
    let diff = albedo * (1.0 - metalness) / PI;
    // Softened by the sun's angular size — see `soft_lambert`. `params11.w` is
    // zero for anything that has not set a sun, which gives back `max(ndl, 0)`.
    return (diff + spec) * soft_lambert(dot(n, l), mesh.params11.w);
}

// ---------------------------------------------------------------------------
// Sheen — Estevez & Kulla "Charlie" distribution with Neubelt visibility, the
// same pair three.js uses. Unlike GGX this lobe *peaks at grazing angles*, which
// is what gives cloth its bright rim and what makes a near-black thermal
// blanket still show its silhouette against a dark background.
// ---------------------------------------------------------------------------
fn d_charlie(n_dot_h : f32, a : f32) -> f32 {
    let inv_a = 1.0 / max(a, 0.0016);
    let cos2h = n_dot_h * n_dot_h;
    let sin2h = max(1.0 - cos2h, 0.0078125); // 2^-7, avoids a fp16 blowup
    return (2.0 + inv_a) * pow(sin2h, inv_a * 0.5) / (2.0 * PI);
}

// three.js `V_Neubelt`, saturated as three.js saturates it.
fn v_neubelt(n_dot_v : f32, n_dot_l : f32) -> f32 {
    return clamp(1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v)), 0.0, 1.0);
}

fn brdf_sheen(
    n : vec3<f32>, v : vec3<f32>, l : vec3<f32>,
    sheen_color : vec3<f32>, sheen_roughness : f32,
) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let a = max(sheen_roughness * sheen_roughness, 0.0016);
    return sheen_color * d_charlie(n_dot_h, a) * v_neubelt(n_dot_v, n_dot_l) * n_dot_l;
}

struct TangentFrame {
    t : vec3<f32>,
    b : vec3<f32>,
};

/// UV-aligned tangent basis solved from screen-space derivatives (Mikkelsen's
/// "derivative maps" trick), so meshes need no baked tangent attribute.
///
/// Falls back progressively: UV Jacobian → position-only tangent → an arbitrary
/// in-plane axis. The last two only trigger on meshes with no UVs or on
/// zero-area fragments, where any frame is as good as another.
/// Pick the best available tangent frame for this fragment.
///
/// Prefers a real per-vertex tangent (glTF `TANGENT`, or `compute_tangents`),
/// Gram-Schmidt-orthogonalised against the shading normal, because it is stable
/// under camera motion and follows the authored UV/tangent flow — which is what
/// lets anisotropy trace a brushed *ring* on a turned part rather than a
/// straight streak. Falls back to the screen-space derivative frame when the
/// geometry supplies none.
fn surface_frame(n : vec3<f32>, p : vec3<f32>, uv : vec2<f32>, vt : vec4<f32>) -> TangentFrame {
    // The derivative fallback is computed unconditionally, *before* any branch
    // on `vt`. `vt` is interpolated per-fragment, so branching on it first would
    // put the dpdx/dpdy calls inside non-uniform control flow — which WGSL
    // forbids. (naga accepts it; Chrome's Tint correctly rejects it, so this
    // only showed up in the browser.)
    let deriv = cotangent_frame(n, p, uv);

    if (dot(vt.xyz, vt.xyz) > 1e-12) {
        let t_raw = vt.xyz - n * dot(n, vt.xyz);
        if (dot(t_raw, t_raw) > 1e-12) {
            let t = normalize(t_raw);
            // w carries handedness, so mirrored UV shells flip the bitangent.
            let sign = select(-1.0, 1.0, vt.w >= 0.0);
            return TangentFrame(t, normalize(cross(n, t)) * sign);
        }
    }
    return deriv;
}

fn cotangent_frame(n : vec3<f32>, p : vec3<f32>, uv : vec2<f32>) -> TangentFrame {
    let dpx = dpdx(p);
    let dpy = dpdy(p);
    let duvx = dpdx(uv);
    let duvy = dpdy(uv);
    let det = duvx.x * duvy.y - duvy.x * duvx.y;

    var t = vec3<f32>(0.0);
    if (abs(det) > 1e-12) {
        t = (dpx * duvy.y - dpy * duvx.y) / det;
        t = t - n * dot(n, t);
    }
    if (dot(t, t) < 1e-12) {
        t = dpx - n * dot(n, dpx);
        if (dot(t, t) < 1e-12) { t = dpy - n * dot(n, dpy); }
    }
    if (dot(t, t) < 1e-12) {
        t = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
        t = t - n * dot(n, t);
    }
    let tn = normalize(t);
    return TangentFrame(tn, normalize(cross(n, tn)));
}

/// Beer-Lambert transmittance through `thickness` world units of the material's
/// interior.
///
/// Two regimes, selected by `attenuation_distance` (params7.w):
///   - 0 → the material never opted in. Falls back to the historical threers
///     behaviour of absorbing by `1 - albedo`, so existing glass keeps tinting
///     from its base color and nothing re-renders differently.
///   - > 0 → the physical formulation: transmitted light decays toward
///     `attenuation_color` over `attenuation_distance`, i.e.
///     σ = -ln(color) / distance and T = exp(-σ·d).
fn volume_transmittance(albedo : vec3<f32>, thickness : f32) -> vec3<f32> {
    let dist = mesh.params7.w;
    let d = max(thickness, 0.0);
    if (dist <= 0.0) {
        return exp(-(vec3<f32>(1.0) - albedo) * d);
    }
    // Clamp away from 0 so a fully-black attenuation channel gives a very large
    // (but finite) coefficient rather than -log(0) = inf.
    let att_color = clamp(mesh.params7.rgb, vec3<f32>(1e-4), vec3<f32>(1.0));
    let sigma = -log(att_color) / dist;
    return exp(-sigma * d);
}

// ---------------------------------------------------------------------------
// Iridescence — thin-film interference after Belcour & Barla 2017, "A Practical
// Extension to Microfacet Theory for the Modeling of Varying Iridescence"
// (matching three.js's iridescence_fragment). A film of thickness d over the
// base sets up optical path differences that reinforce some wavelengths and
// cancel others; `eval_sensitivity` integrates that against Gaussian fits of
// the CIE XYZ response to get an RGB reflectance directly.
// ---------------------------------------------------------------------------
fn ior_to_fresnel0(transmitted : vec3<f32>, incident : f32) -> vec3<f32> {
    let t = (transmitted - vec3<f32>(incident)) / (transmitted + vec3<f32>(incident));
    return t * t;
}

fn ior_to_fresnel0_f(transmitted : f32, incident : f32) -> f32 {
    let t = (transmitted - incident) / (transmitted + incident);
    return t * t;
}

/// Invert Schlick's F0 back to a relative IOR. Only valid for F0 < 1.
fn fresnel0_to_ior(f0 : vec3<f32>) -> vec3<f32> {
    let s = sqrt(clamp(f0, vec3<f32>(0.0), vec3<f32>(0.9999)));
    return (vec3<f32>(1.0) + s) / (vec3<f32>(1.0) - s);
}

/// Schlick with a custom F90, used for the film's own interface.
// three.js `F_Schlick(f0, f90, dotVH)` (`bsdfs`), in its spherical-Gaussian
// form — the one `BRDF_GGX`, the clearcoat and `evalIridescence` all use.
fn f_schlick_f90(f0 : vec3<f32>, f90 : vec3<f32>, v_dot_h : f32) -> vec3<f32> {
    let fresnel = exp2((-5.55473 * v_dot_h - 6.98316) * v_dot_h);
    return f0 * (1.0 - fresnel) + f90 * fresnel;
}

fn f_schlick_f90_f(f0 : f32, f90 : f32, v_dot_h : f32) -> f32 {
    let fresnel = exp2((-5.55473 * v_dot_h - 6.98316) * v_dot_h);
    return f0 * (1.0 - fresnel) + f90 * fresnel;
}

/// Gaussian fit of the CIE XYZ colour-matching functions evaluated at optical
/// path difference `opd`, converted to linear sRGB.
fn eval_sensitivity(opd : f32, shift : vec3<f32>) -> vec3<f32> {
    let phase = 2.0 * PI * opd * 1.0e-9;
    let val = vec3<f32>(5.4856e-13, 4.4201e-13, 5.2481e-13);
    let pos = vec3<f32>(1.6810e+06, 1.7953e+06, 2.2084e+06);
    let vr  = vec3<f32>(4.3278e+09, 9.3046e+09, 6.6121e+09);

    var xyz = val * sqrt(2.0 * PI * vr) * cos(pos * phase + shift) * exp(-(phase * phase) * vr);
    // The extra x-bar lobe the Gaussian fit needs to stay accurate in the red.
    let x_extra = 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09)
        * cos(2.2399e+06 * phase + shift.x) * exp(-4.5282e+09 * phase * phase);
    xyz.x = xyz.x + x_extra;
    xyz = xyz / 1.0685e-7;

    let xyz_to_rgb = mat3x3<f32>(
        vec3<f32>( 3.2404542, -0.9692660,  0.0556434),
        vec3<f32>(-1.5371385,  1.8760108, -0.2040259),
        vec3<f32>(-0.4985314,  0.0415560,  1.0572252),
    );
    return xyz_to_rgb * xyz;
}

/// Reflectance of a `thickness_nm` film of IOR `film_ior` over a base of
/// reflectance `base_f0`, seen at `cos_theta1`.
fn iridescence_fresnel(
    outside_ior : f32, film_ior_in : f32, base_f0 : vec3<f32>,
    thickness_nm : f32, cos_theta1 : f32,
) -> vec3<f32> {
    // Force the film IOR toward the outside IOR as the film vanishes, so
    // thickness→0 degrades gracefully to the plain base reflectance.
    let film_ior = mix(outside_ior, film_ior_in, smoothstep(0.0, 0.03, thickness_nm));
    let sin_theta2_sq = pow(outside_ior / film_ior, 2.0) * (1.0 - cos_theta1 * cos_theta1);
    let cos_theta2_sq = 1.0 - sin_theta2_sq;

    // Total internal reflection inside the film — no transmitted lobe at all.
    if (cos_theta2_sq < 0.0) {
        return vec3<f32>(1.0);
    }
    let cos_theta2 = sqrt(cos_theta2_sq);

    // First interface (outside → film).
    let r0 = ior_to_fresnel0_f(film_ior, outside_ior);
    let r12 = f_schlick_f90_f(r0, 1.0, cos_theta1);
    let t121 = 1.0 - r12;

    // Second interface (film → base). Phase shift is π when the film is the
    // rarer medium, which flips which wavelengths reinforce.
    var phi12 = 0.0;
    if (film_ior < outside_ior) { phi12 = PI; }
    let phi21 = PI - phi12;

    let base_ior = fresnel0_to_ior(clamp(base_f0, vec3<f32>(0.0), vec3<f32>(0.9999)));
    let r1 = ior_to_fresnel0(base_ior, film_ior);
    let r23 = f_schlick_f90(r1, vec3<f32>(1.0), cos_theta2);

    var phi23 = vec3<f32>(0.0);
    if (base_ior.r < film_ior) { phi23.r = PI; }
    if (base_ior.g < film_ior) { phi23.g = PI; }
    if (base_ior.b < film_ior) { phi23.b = PI; }

    let opd = 2.0 * film_ior * thickness_nm * cos_theta2;
    let phi = vec3<f32>(phi21) + phi23;

    // Compound the infinite series of internal bounces analytically: `rs` is
    // the summed contribution of every round trip inside the film, `c0` the
    // non-interfering DC term, and the loop adds the first two interference
    // orders (higher orders are below perceptual threshold).
    let r123 = clamp(r12 * r23, vec3<f32>(1e-5), vec3<f32>(0.9999));
    let r123_sqrt = sqrt(r123);
    let rs = (t121 * t121) * r23 / max(vec3<f32>(1.0) - r123, vec3<f32>(1e-5));

    let c0 = vec3<f32>(r12) + rs;
    var i_out = c0;

    var cm = rs - vec3<f32>(t121);
    for (var m : i32 = 1; m <= 2; m = m + 1) {
        cm = cm * r123_sqrt;
        let sm = 2.0 * eval_sensitivity(f32(m) * opd, f32(m) * phi);
        i_out = i_out + cm * sm;
    }

    return max(i_out, vec3<f32>(0.0));
}

// Specular-AA tuning. SIGMA is the screen-space filter width, KAPPA the clamp
// that stops a silhouette (where the normal derivative explodes) from being
// blurred into a matte band.
const SPEC_AA_SIGMA : f32 = 0.25;
const SPEC_AA_KAPPA : f32 = 0.18;

/// Horizon-based specular occlusion (Lagarde & de Rousiers, "Moving Frostbite
/// to PBR", §4.10.2).
///
/// Ambient occlusion answers "how much of the hemisphere can this point see",
/// which is the right question for a diffuse lobe and the wrong one for a
/// specular lobe — that only cares about the cone around the reflection
/// vector. Applying AO to diffuse alone, as three.js does, leaves a cavity in a
/// *metal* reflecting the whole environment as though nothing were in the way:
/// a speaker grille's dimples come out as bright crescents rather than as
/// holes, and a row of them beats against the pixel grid.
///
/// Returns 1.0 for an unoccluded point, so a scene with no AO is unchanged.
fn specular_occlusion(n_dot_v : f32, ao : f32, roughness : f32) -> f32 {
    return clamp(
        pow(n_dot_v + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao,
        0.0,
        1.0,
    );
}

/// Widen the specular lobe to cover the sub-pixel normal distribution.
///
/// Works in alpha (= roughness²) space, where variance actually adds; adding in
/// perceptual-roughness space — as the previous implementation did — is not the
/// same operation and over-blurs.
fn specular_aa_roughness(n : vec3<f32>, perceptual_roughness : f32) -> f32 {
    let du = dpdx(n);
    let dv = dpdy(n);
    let variance = SPEC_AA_SIGMA * (dot(du, du) + dot(dv, dv));
    let alpha = perceptual_roughness * perceptual_roughness;
    let kernel = min(2.0 * variance, SPEC_AA_KAPPA);
    return clamp(sqrt(alpha + kernel), 0.0, 1.0);
}


/// One light's contribution to the clearcoat lobe: plain isotropic GGX at the
/// coat's own roughness, over a fixed dielectric F0.
fn clearcoat_spec(
    n : vec3<f32>, v : vec3<f32>, l : vec3<f32>, cc_a : f32, fc0 : vec3<f32>,
) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);
    let d = d_ggx(n_dot_h, cc_a);
    let f = f_schlick(v_dot_h, fc0);
    // three.js `BRDF_GGX_Clearcoat`: D · V_GGX_SmithCorrelated · F.
    let spec = d * v_ggx_smith_correlated(cc_a, n_dot_l, n_dot_v) * f;
    return spec * n_dot_l;
}

// Direct-light PBR dispatch: isotropic GGX unless the material carries an
// anisotropy strength, in which case the anisotropic lobe is used. `f0` is
// passed in rather than derived so the iridescence layer can override it.
// three.js `transmission_pars_fragment` (from the glTF Sample Viewer): N8's
// mipped bicubic filter, the refracted ray through the volume, Beer–Lambert
// attenuation and `getIBLVolumeRefraction`.
fn bicubic_w0(a : f32) -> f32 { return (1.0 / 6.0) * (a * (a * (-a + 3.0) - 3.0) + 1.0); }
fn bicubic_w1(a : f32) -> f32 { return (1.0 / 6.0) * (a * a * (3.0 * a - 6.0) + 4.0); }
fn bicubic_w2(a : f32) -> f32 { return (1.0 / 6.0) * (a * (a * (-3.0 * a + 3.0) + 3.0) + 1.0); }
fn bicubic_w3(a : f32) -> f32 { return (1.0 / 6.0) * (a * a * a); }
fn bicubic_g0(a : f32) -> f32 { return bicubic_w0(a) + bicubic_w1(a); }
fn bicubic_g1(a : f32) -> f32 { return bicubic_w2(a) + bicubic_w3(a); }
fn bicubic_h0(a : f32) -> f32 { return -1.0 + bicubic_w1(a) / (bicubic_w0(a) + bicubic_w1(a)); }
fn bicubic_h1(a : f32) -> f32 { return 1.0 + bicubic_w3(a) / (bicubic_w2(a) + bicubic_w3(a)); }

fn transmission_bicubic(uv_in : vec2<f32>, texel : vec4<f32>, lod : f32) -> vec4<f32> {
    let uv = uv_in * texel.zw + 0.5;
    let iuv = floor(uv);
    let fuv = fract(uv);
    let g0x = bicubic_g0(fuv.x);
    let g1x = bicubic_g1(fuv.x);
    let h0x = bicubic_h0(fuv.x);
    let h1x = bicubic_h1(fuv.x);
    let h0y = bicubic_h0(fuv.y);
    let h1y = bicubic_h1(fuv.y);
    let p0 = (vec2<f32>(iuv.x + h0x, iuv.y + h0y) - 0.5) * texel.xy;
    let p1 = (vec2<f32>(iuv.x + h1x, iuv.y + h0y) - 0.5) * texel.xy;
    let p2 = (vec2<f32>(iuv.x + h0x, iuv.y + h1y) - 0.5) * texel.xy;
    let p3 = (vec2<f32>(iuv.x + h1x, iuv.y + h1y) - 0.5) * texel.xy;
    return bicubic_g0(fuv.y) * (g0x * textureSampleLevel(transmission_tex, transmission_sampler, p0, lod)
            + g1x * textureSampleLevel(transmission_tex, transmission_sampler, p1, lod))
        + bicubic_g1(fuv.y) * (g0x * textureSampleLevel(transmission_tex, transmission_sampler, p2, lod)
            + g1x * textureSampleLevel(transmission_tex, transmission_sampler, p3, lod));
}

// three.js `textureBicubic`: the two levels around `lod`, mixed. Levels
// past the chain's end read its last, as GL clamps them.
fn transmission_texture_bicubic(uv : vec2<f32>, lod : f32) -> vec4<f32> {
    let last = max(i32(frame.light_counts2.w) - 1, 0);
    let fl = min(i32(floor(lod)), last);
    let cl = min(i32(ceil(lod)), last);
    let f_size = vec2<f32>(textureDimensions(transmission_tex, fl));
    let c_size = vec2<f32>(textureDimensions(transmission_tex, cl));
    let f_sample = transmission_bicubic(uv, vec4<f32>(1.0 / f_size, f_size), f32(fl));
    let c_sample = transmission_bicubic(uv, vec4<f32>(1.0 / c_size, c_size), f32(cl));
    return mix(f_sample, c_sample, fract(lod));
}

// three.js `getTransmissionSample` at the projected exit point `p` (world).
// `applyIorToRoughness`: an ior of 1 blurs nothing, 1.5 and up fully.
fn transmission_sample(p : vec3<f32>, roughness : f32, ior : f32) -> vec4<f32> {
    let ndc = frame.view_proj * vec4<f32>(p, 1.0);
    let xy = ndc.xy / ndc.w;
    // The capture's rows run top first, where GL's run bottom first.
    let uv = vec2<f32>(xy.x * 0.5 + 0.5, 0.5 - xy.y * 0.5);
    let width = f32(textureDimensions(transmission_tex, 0).x);
    let lod = log2(width) * roughness * clamp(ior * 2.0 - 2.0, 0.0, 1.0);
    return transmission_texture_bicubic(uv, lod);
}

// three.js `getVolumeTransmissionRay`: the refracted view ray, `thickness`
// long in the model's own scale.
fn volume_transmission_ray(n : vec3<f32>, v : vec3<f32>, thickness : f32, ior : f32) -> vec3<f32> {
    let refraction = refract(-v, normalize(n), 1.0 / ior);
    let scale = vec3<f32>(length(mesh.model[0].xyz), length(mesh.model[1].xyz), length(mesh.model[2].xyz));
    return normalize(refraction) * thickness * scale;
}

// three.js `volumeAttenuation` (Beer's law); an infinite distance — here 0,
// as the material uploads it — attenuates nothing.
fn volume_attenuation(distance : f32, color : vec3<f32>, attenuation_distance : f32) -> vec3<f32> {
    if (attenuation_distance <= 0.0) {
        return vec3<f32>(1.0);
    }
    return exp(log(color) / attenuation_distance * distance);
}

fn ibl_volume_refraction(
    n : vec3<f32>, v : vec3<f32>, roughness : f32, diffuse_color : vec3<f32>,
    specular_color : vec3<f32>, specular_f90 : f32, position : vec3<f32>,
    dispersion : f32, ior : f32, thickness : f32,
    attenuation_color : vec3<f32>, attenuation_distance : f32,
) -> vec4<f32> {
    var transmitted = vec4<f32>(0.0);
    var transmittance = vec3<f32>(0.0);
    if (dispersion > 0.0) {
        // `USE_DISPERSION`: each channel along its own ior.
        let half_spread = (ior - 1.0) * 0.025 * dispersion;
        let iors = vec3<f32>(ior - half_spread, ior, ior + half_spread);
        for (var i = 0u; i < 3u; i = i + 1u) {
            let ray = volume_transmission_ray(n, v, thickness, iors[i]);
            let t = transmission_sample(position + ray, roughness, iors[i]);
            transmitted[i] = t[i];
            transmitted.a = transmitted.a + t.a;
            transmittance[i] = diffuse_color[i] * volume_attenuation(length(ray), attenuation_color, attenuation_distance)[i];
        }
        transmitted.a = transmitted.a / 3.0;
    } else {
        let ray = volume_transmission_ray(n, v, thickness, ior);
        transmitted = transmission_sample(position + ray, roughness, ior);
        transmittance = diffuse_color * volume_attenuation(length(ray), attenuation_color, attenuation_distance);
    }
    let attenuated = transmittance * transmitted.rgb;
    // `EnvironmentBRDF`: what is reflected is not transmitted.
    let fab = dfg_approx(n, v, roughness);
    let f = specular_color * fab.x + specular_f90 * fab.y;
    let factor = (transmittance.r + transmittance.g + transmittance.b) / 3.0;
    return vec4<f32>((vec3<f32>(1.0) - f) * attenuated, 1.0 - (1.0 - transmitted.a) * factor);
}

// three.js `RectAreaLightUniformsLib`'s tables, `ltc_1` (layer 0) and
// `ltc_2` (layer 1), 64×64, sampled linearly.
@group(3) @binding(14) var ltc_tex : texture_2d_array<f32>;

// three.js `LTC_Uv`: the table coordinate for roughness and the view angle.
fn ltc_uv(n : vec3<f32>, v : vec3<f32>, roughness : f32) -> vec2<f32> {
    let lut_size = 64.0;
    let lut_scale = (lut_size - 1.0) / lut_size;
    let lut_bias = 0.5 / lut_size;
    let n_dot_v = clamp(dot(n, v), 0.0, 1.0);
    return vec2<f32>(roughness, sqrt(1.0 - n_dot_v)) * lut_scale + lut_bias;
}

// three.js `LTC_ClippedSphereFormFactor`.
fn ltc_clipped_sphere_form_factor(f : vec3<f32>) -> f32 {
    let l = length(f);
    return max((l * l + f.z) / (l + 1.0), 0.0);
}

// three.js `LTC_EdgeVectorFormFactor`.
fn ltc_edge_vector_form_factor(v1 : vec3<f32>, v2 : vec3<f32>) -> vec3<f32> {
    let x = dot(v1, v2);
    let y = abs(x);
    let a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    let b = 3.4175940 + (4.1616724 + y) * y;
    let v = a / b;
    let theta_sintheta = select(0.5 * inverseSqrt(max(1.0 - x * x, 1e-7)) - v, v, x > 0.0);
    return cross(v1, v2) * theta_sintheta;
}

// three.js `LTC_Evaluate`: the polygon's form factor under the cosine
// `m_inv` maps to, horizon-clipped; zero behind the light's emitting face.
fn ltc_evaluate(n : vec3<f32>, v : vec3<f32>, p : vec3<f32>, m_inv : mat3x3<f32>, r0 : vec3<f32>, r1 : vec3<f32>, r2 : vec3<f32>, r3 : vec3<f32>) -> f32 {
    let e1 = r1 - r0;
    let e2 = r3 - r0;
    let light_normal = cross(e1, e2);
    if (dot(light_normal, p - r0) < 0.0) {
        return 0.0;
    }
    let t1 = normalize(v - n * dot(v, n));
    let t2 = -cross(n, t1);
    let m = m_inv * transpose(mat3x3<f32>(t1, t2, n));
    let c0 = normalize(m * (r0 - p));
    let c1 = normalize(m * (r1 - p));
    let c2 = normalize(m * (r2 - p));
    let c3 = normalize(m * (r3 - p));
    var f = ltc_edge_vector_form_factor(c0, c1);
    f = f + ltc_edge_vector_form_factor(c1, c2);
    f = f + ltc_edge_vector_form_factor(c2, c3);
    f = f + ltc_edge_vector_form_factor(c3, c0);
    return ltc_clipped_sphere_form_factor(f);
}

// three.js `RE_Direct_RectArea_Physical` for one light: `right` and `up`
// are its half width and half height (three.js's `halfWidth`,
// `halfHeight`), `color` its colour × intensity.
fn rect_area_light(rl : RectLight, p : vec3<f32>, n : vec3<f32>, v : vec3<f32>, roughness : f32, specular_color : vec3<f32>, diffuse_color : vec3<f32>) -> LightTerms {
    let c = rl.position.xyz;
    let hw = rl.right.xyz;
    let hh = rl.up.xyz;
    let r0 = c + hw - hh;
    let r1 = c - hw - hh;
    let r2 = c - hw + hh;
    let r3 = c + hw + hh;
    let uv = ltc_uv(n, v, roughness);
    let t1 = textureSampleLevel(ltc_tex, env_sampler, uv, 0, 0.0);
    let t2 = textureSampleLevel(ltc_tex, env_sampler, uv, 1, 0.0);
    let m_inv = mat3x3<f32>(
        vec3<f32>(t1.x, 0.0, t1.y),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(t1.z, 0.0, t1.w),
    );
    let fresnel = specular_color * t2.x + (vec3<f32>(1.0) - specular_color) * t2.y;
    let identity = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    let spec = rl.color.rgb * fresnel * ltc_evaluate(n, v, p, m_inv, r0, r1, r2, r3);
    let diff = rl.color.rgb * diffuse_color * ltc_evaluate(n, v, p, identity, r0, r1, r2, r3);
    return LightTerms(diff, spec);
}

fn pbr_direct(n : vec3<f32>, v : vec3<f32>, l : vec3<f32>, diffuse_color : vec3<f32>, lobe : SpecLobe) -> vec3<f32> {
    let t = pbr_direct_split(n, v, l, diffuse_color, lobe);
    return t.diffuse + t.specular;
}

// three.js keeps `reflectedLight.directDiffuse` and `directSpecular` apart:
// transmission replaces the diffuse alone.
struct LightTerms {
    diffuse : vec3<f32>,
    specular : vec3<f32>,
};

fn pbr_direct_split(n : vec3<f32>, v : vec3<f32>, l : vec3<f32>, diffuse_color : vec3<f32>, lobe : SpecLobe) -> LightTerms {
    // Softened by the sun's angular size — see `soft_lambert`. `params11.w` is
    // zero for anything that has not set a sun, which gives back `max(ndl, 0)`.
    let irradiance = soft_lambert(dot(n, l), mesh.params11.w);
    return LightTerms(diffuse_color / PI * irradiance, brdf_ggx(l, v, n, lobe) * irradiance);
}

// three.js r165 `PhysicalMaterial`'s specular lobe, as `lights_physical_fragment`
// fills it: reflectance at normal incidence (`specularColor`) and at grazing
// (`specularF90`), GGX alpha, the iridescence weight and thin-film Fresnel, and
// for anisotropy its strength, the tangent alpha and the anisotropy direction
// and its perpendicular — not unit vectors, as three.js's `tbn` is not.
struct SpecLobe {
    f0 : vec3<f32>,
    f90 : f32,
    alpha : f32,
    irid : f32,
    irid_fresnel : vec3<f32>,
    aniso : f32,
    alpha_t : f32,
    at : vec3<f32>,
    ab : vec3<f32>,
};

// three.js `BRDF_GGX`: `F · V · D`, the Fresnel mixed toward the thin film's,
// and the anisotropic V and D (Filament's) when there is anisotropy.
fn brdf_ggx(l : vec3<f32>, v : vec3<f32>, n : vec3<f32>, s : SpecLobe) -> vec3<f32> {
    let h = normalize(l + v);
    let n_dot_l = clamp(dot(n, l), 0.0, 1.0);
    let n_dot_v = clamp(dot(n, v), 0.0, 1.0);
    let n_dot_h = clamp(dot(n, h), 0.0, 1.0);
    let v_dot_h = clamp(dot(v, h), 0.0, 1.0);
    var f = f_schlick_f90(s.f0, vec3<f32>(s.f90), v_dot_h);
    f = mix(f, s.irid_fresnel, s.irid);
    if (s.aniso >= 0.0) {
        let t_l = dot(s.at, l);
        let t_v = dot(s.at, v);
        let t_h = dot(s.at, h);
        let b_l = dot(s.ab, l);
        let b_v = dot(s.ab, v);
        let b_h = dot(s.ab, h);
        // `V_GGX_SmithCorrelated_Anisotropic`. (Not `gv` / `gl`: GLSL
        // reserves `gl_`, and naga would name a second `gl` that.)
        let vis_v = n_dot_l * length(vec3<f32>(s.alpha_t * t_v, s.alpha * b_v, n_dot_v));
        let vis_l = n_dot_v * length(vec3<f32>(s.alpha_t * t_l, s.alpha * b_l, n_dot_l));
        let vis = clamp(0.5 / (vis_v + vis_l), 0.0, 1.0);
        // `D_GGX_Anisotropic`
        let a2 = s.alpha_t * s.alpha;
        let dv = vec3<f32>(s.alpha * t_h, s.alpha_t * b_h, a2 * n_dot_h);
        let w2 = a2 / dot(dv, dv);
        let d = a2 * w2 * w2 / PI;
        return f * (vis * d);
    }
    return f * (v_ggx_smith_correlated(s.alpha, n_dot_l, n_dot_v) * d_ggx(n_dot_h, s.alpha));
}

// three.js `getTangentFrame(eyePos, surfNorm, uv)`: the tangent and bitangent
// from screen-space derivatives of position and uv, scaled together (so not
// each of unit length) — the frame three.js's anisotropy and normal maps use
// on geometry without a `tangent` attribute. In world space here; the frame
// rotates with the space it is built in.
fn three_tangent_frame(p : vec3<f32>, n : vec3<f32>, uv : vec2<f32>) -> TangentFrame {
    let q0 = dpdx(p);
    let q1 = dpdy(p);
    let st0 = dpdx(uv);
    let st1 = dpdy(uv);
    let q1perp = cross(q1, n);
    let q0perp = cross(n, q0);
    let t = q1perp * st0.x + q0perp * st1.x;
    let b = q1perp * st0.y + q0perp * st1.y;
    let det = max(dot(t, t), dot(b, b));
    let scale = select(inverseSqrt(det), 0.0, det == 0.0);
    return TangentFrame(t * scale, b * scale);
}

// three.js `IBLSheenBRDF`: Estevez and Kulla's Charlie sheen integrated over
// the hemisphere, as a curve fit.
fn ibl_sheen_brdf(n : vec3<f32>, v : vec3<f32>, roughness : f32) -> f32 {
    let n_dot_v = clamp(dot(n, v), 0.0, 1.0);
    let r2 = roughness * roughness;
    let a = select(-8.48 * r2 + 14.3 * roughness - 9.95, -339.2 * r2 + 161.4 * roughness - 25.9, roughness < 0.25);
    let b = select(1.97 * r2 - 3.27 * roughness + 0.72, 44.0 * r2 - 23.7 * roughness + 3.26, roughness < 0.25);
    let dg = exp(a * n_dot_v + b) + select(0.1 * (roughness - 0.25), 0.0, roughness < 0.25);
    return clamp(dg / PI, 0.0, 1.0);
}

@fragment
fn fs_main(in : VsOut, @builtin(front_facing) front_facing : bool) -> @location(0) vec4<f32> {
    return shade_main(in, front_facing);
}

// What `shade_main` found passes through a blended surface, per channel, as
// the factor the blend multiplies what is behind by (encoded, as the blend
// works); negative when it returned before working it out, its colour then
// not premultiplied. `fs_oit` reads it.
var<private> glass_t_out : vec3<f32>;

// The lit materials' shading, for `fs_main` and the OIT accumulation.
fn shade_main(in : VsOut, front_facing : bool) -> vec4<f32> {
    glass_t_out = vec3<f32>(-1.0);
    clip_fragment(in.world_pos);
    map_uv12 = in.uv12;
    map_uv3 = in.uv3;
    let kind = mesh.flags.x;
    let tex_flags = mesh.flags.y;
    var base_color = mesh.color.rgb * in.vertex_color.rgb;
    // `var`, because a bound `map` contributes its alpha to the lit paths below
    // the same way three.js's `diffuseColor *= sampledDiffuseColor` does.
    var opacity = mesh.params.y * in.vertex_color.a;

    // Basic: just return the color × optional map, no lighting. The JS shim
    // sRGB-decoded the input color to linear; we encode it back to sRGB on
    // output so the byte round-trips (matches three.js's MeshBasicMaterial).
    if (kind == MAT_BASIC) {
        var col = base_color;
        var a = opacity;
        if ((tex_flags & FLAG_MAP) != 0u) {
            let s = sample_map(in.uv);
            col = col * s.rgb;
            a = a * s.a;
        }
        a = a * alpha_map_of(in.uv);
        if (a < alpha_test_of()) { discard; }
        // three.js `meshbasic_frag`: with a light map the indirect diffuse is
        // its texel × `lightMapIntensity` / π instead of 1.
        if (HAS_LIGHT_MAP) {
            col = col * light_map_irradiance(in.uv) * 0.3183098861837907;
        }
        // three.js Basic reflects per vertex (`vReflect`); here per fragment
        // from the interpolated normal, the same on any flat face.
        if (HAS_ENV_MAP) {
            let nb = normalize(in.world_normal) * select(-1.0, 1.0, front_facing);
            col = apply_env_map(col, nb, in.world_pos, specular_strength_of(in.uv));
        }
        if ((mesh.flags.z & FLAG_SHADOW_MAT) != 0u) {
            // three.js ShadowMaterial: `vec4(color, opacity · (1 −
            // getShadowMask()))`, the mask being the product of every
            // shadow-casting light's shadow — and only on a mesh that
            // receives shadows (`shadowmask_pars_fragment`).
            var mask = 1.0;
            if ((mesh.flags.z & FLAG_RECEIVE_SHADOW) != 0u) {
                mask = shadow_mask(in.world_pos, normalize(in.world_normal));
            }
            let shadow_col = apply_fog(apply_tone_mapping(mesh.color.rgb), in.view_z);
            return vec4<f32>(framebuffer_encode(shadow_col), opacity * (1.0 - mask));
        }
        // three.js `meshbasic_frag` ends with `tonemapping_fragment` too.
        col = apply_fog(apply_tone_mapping(col), in.view_z);
        if ((mesh.flags.z & FLAG_OPAQUE_OUT) != 0u) { a = 1.0; }
        return vec4<f32>(framebuffer_encode(col), a);
    }
    // Reflector / Refractor / Water — projectively samples the RT texture
    // bound on the albedo slot. `mesh.normal_matrix` carries the per-frame
    // texture matrix (scaleBias * P * V * matrixWorld); the vertex shader
    // passes perspective-correct homogeneous UV (three.js `texture2DProj`).
    if (kind == MAT_MIRROR) {
        let denom = max(in.proj_uv.w, 0.0001);
        var uv = in.proj_uv.xy / denom;
        // WebGPU RT rows are top-first; three.js Reflector bias assumes WebGL flipY.
        uv.y = 1.0 - uv.y;
        var col = vec3<f32>(0.0);
        let base_rgb = mesh.color.rgb;
        if ((tex_flags & FLAG_MAP) != 0u) {
            let s = sample_mirror(uv).rgb;
            // three.js's Reflector blendOverlay(sample, color): tint that
            // leaves mid-gray (0.5,0.5,0.5) unchanged.
            let one = vec3<f32>(1.0);
            let two = vec3<f32>(2.0);
            let low  = two * s * base_rgb;
            let high = one - two * (one - s) * (one - base_rgb);
            let lt = step(s, vec3<f32>(0.5));
            col = mix(high, low, lt);
        } else {
            col = base_rgb;
        }
        // three.js's Reflector shader ends in `tonemapping_fragment`.
        return vec4<f32>(framebuffer_encode(apply_tone_mapping(col)), 1.0);
    }
    // Line / Points: unlit color, just the material's color × per-vertex color.
    // Lines have no normals so we can't fall through to PBR — without this branch
    // the geometry rasterizes but the fragment shader divides by zero on the
    // normal-dependent paths and renders nearly black.
    if (kind == MAT_LINE || kind == MAT_POINTS) {
        var line_col = base_color;
        if (kind == MAT_LINE) {
            let total = mesh.params3.y + mesh.params3.z;
            if (total > 0.0) {
                line_col = mesh.color.rgb;
                let dist = in.uv.x * mesh.params3.x;
                if ((dist % total) > mesh.params3.y) {
                    discard;
                }
            }
        }
        // three.js `map_particle_fragment`: the point's map and alpha map at its
        // point coordinate.
        if (kind == MAT_POINTS) {
            if ((tex_flags & FLAG_MAP) != 0u) {
                let s = sample_map(in.uv);
                line_col = line_col * s.rgb;
                opacity = opacity * s.a;
            }
            opacity = opacity * alpha_map_of(in.uv);
        }
        if (opacity < alpha_test_of()) { discard; }
        // LineBasicMaterial draws with three.js's `basic` program, and
        // PointsMaterial's `points` one tone-maps the same way.
        let col = apply_fog(apply_tone_mapping(line_col), in.view_z);
        return vec4<f32>(framebuffer_encode(col), opacity);
    }
    if (kind == MAT_SPRITE) {
        var col = base_color;
        if ((tex_flags & FLAG_MAP) != 0u) {
            // three.js `sprite_frag`: `diffuseColor *= sampledDiffuseColor`,
            // alpha included — a sprite's map is usually its cut-out.
            let s = sample_map(in.uv);
            col = col * s.rgb;
            opacity = opacity * s.a;
        }
        opacity = opacity * alpha_map_of(in.uv);
        if (opacity < alpha_test_of()) { discard; }
        // three.js `sprite_frag` includes `tonemapping_fragment`.
        col = apply_fog(apply_tone_mapping(col), in.view_z);
        if ((mesh.flags.z & FLAG_OPAQUE_OUT) != 0u) { opacity = 1.0; }
        return vec4<f32>(framebuffer_encode(col), opacity);
    }
    // Window-space depth [0,1] from @builtin(position) (matches GPU depth buffer).
    if (kind == MAT_DEPTH) {
        let depth = 0.5 * in.clip_zw.x / in.clip_zw.y + 0.5;
        let vis = 1.0 - depth;
        return vec4<f32>(framebuffer_data(vec3<f32>(vis)), 1.0);
    }
    if (kind == MAT_NORMAL) {
        // View-space normals (matches three.js MeshNormalMaterial for SSAO prepass).
        let view_n = normalize((frame.view * vec4<f32>(normalize(in.world_normal), 0.0)).xyz);
        return vec4<f32>(framebuffer_data(view_n * 0.5 + 0.5), 1.0);
    }
    // Sky shader. Preetham atmospheric scattering model — same algorithm
    // three.js's Sky uses. Reads sun position from mesh.specular.xyz and
    // atmospheric tuning (turbidity, rayleigh, mieCoefficient, mieDirectionalG)
    // from mesh.params/params2. The mesh is a large sphere with BackSide so the
    // user is looking at the inside of the dome.
    if (kind == MAT_SKY) {
        // Direct port of three.js examples/jsm/objects/Sky.js (Preetham model).
        // Vertex-side varyings (vSunfade, vBetaR, vBetaM, vSunE, vSunDirection)
        // are recomputed here per-fragment.
        let cameraPos = frame.camera_position.xyz;
        let worldPos = in.world_pos;
        let sunPosition = mesh.specular.xyz;
        let turbidity = mesh.params.x;
        let rayleigh = mesh.params.y;
        let mieCoefficient = mesh.params.z;
        let mieDirG = mesh.params.w;

        let up_v = vec3<f32>(0.0, 1.0, 0.0);
        let PI = 3.141592653589793;
        let E_CONST = 2.718281828459045;
        let CUTOFF_ANGLE = 1.6110731556870734;
        let STEEPNESS = 1.5;
        let EE = 1000.0;
        let sunDir = normalize(sunPosition);

        // vSunfade — earth-shadow soft attenuation as sun approaches horizon.
        let vSunfade = 1.0 - clamp(1.0 - exp(sunPosition.y / 450000.0), 0.0, 1.0);

        // rayleighCoefficient = rayleigh - 1.0 * (1.0 - vSunfade)
        let rayleighCoefficient = rayleigh - (1.0 - vSunfade);
        let totalRayleigh = vec3<f32>(5.804542996261093e-6, 1.3562911419845635e-5, 3.0265902468824876e-5);
        let vBetaR = totalRayleigh * rayleighCoefficient;

        // totalMie(T) = 0.434 * (0.2*T*1e-17) * MieConst — pre-baked constants from three.js.
        let MieConst = vec3<f32>(1.8399918514433978e14, 2.7798023919660528e14, 4.0790479543861094e14);
        let c_mie = 0.2 * turbidity * 1.0e-17;
        let vBetaM = (0.434 * c_mie * MieConst) * mieCoefficient;

        // vSunE — sun intensity falloff from horizon (sunIntensity in three.js).
        let zenithAngleCos_sun = clamp(dot(sunDir, up_v), -1.0, 1.0);
        let vSunE = EE * max(0.0, 1.0 - pow(E_CONST, -((CUTOFF_ANGLE - acos(zenithAngleCos_sun)) / STEEPNESS)));

        // View direction in world space.
        let direction = normalize(worldPos - cameraPos);

        // Optical length along view; cutoff at 90° to avoid singularity.
        let zenithAngle = acos(max(0.0, dot(up_v, direction)));
        let inv = 1.0 / (cos(zenithAngle) + 0.15 * pow(max(0.001, 93.885 - (zenithAngle * 180.0 / PI)), -1.253));
        let sR = 8.4e3 * inv;
        let sM = 1.25e3 * inv;

        // Combined extinction factor.
        let Fex = exp(-(vBetaR * sR + vBetaM * sM));

        let cosTheta = dot(direction, sunDir);
        let rPhase = 0.05968310365946075 * (1.0 + pow(cosTheta * 0.5 + 0.5, 2.0));
        let betaRTheta = vBetaR * rPhase;
        let g2 = mieDirG * mieDirG;
        let inv_mie = 1.0 / pow(max(0.0001, 1.0 - 2.0 * mieDirG * cosTheta + g2), 1.5);
        let mPhase = 0.07957747154594767 * ((1.0 - g2) * inv_mie);
        let betaMTheta = vBetaM * mPhase;

        let denom = max(vBetaR + vBetaM, vec3<f32>(1.0e-12));
        let scatter_ratio = (betaRTheta + betaMTheta) / denom;
        var Lin = pow(max(vSunE * scatter_ratio * (vec3<f32>(1.0) - Fex), vec3<f32>(0.0)), vec3<f32>(1.5));
        let sun_horizon_t = clamp(pow(1.0 - dot(up_v, sunDir), 5.0), 0.0, 1.0);
        let Lin_mix = pow(max(vSunE * scatter_ratio * Fex, vec3<f32>(0.0)), vec3<f32>(0.5));
        Lin = Lin * mix(vec3<f32>(1.0), Lin_mix, sun_horizon_t);

        var L0 = vec3<f32>(0.1) * Fex;
        let sundisk = smoothstep(0.999956676946448, 0.999976676946448, cosTheta);
        L0 = L0 + (vSunE * 19000.0 * Fex) * sundisk;

        let texColor = (Lin + L0) * 0.04 + vec3<f32>(0.0, 0.0003, 0.00075);
        let retColor = pow(max(texColor, vec3<f32>(0.0)), vec3<f32>(1.0 / (1.2 + 1.2 * vSunfade)));

        // three.js's Sky ends in `tonemapping_fragment` then
        // `colorspace_fragment`: the renderer's tone mapping (its examples set
        // ACES with a low exposure, which this sky is tuned for), then the
        // encode. Clamped to [0,1] as GL's framebuffer write is.
        return vec4<f32>(framebuffer_encode(apply_tone_mapping(retColor)), 1.0);
    }
    // MeshDistanceMaterial: linear distance from `mesh.specular.xyz` (the
    // reference position, typically a point light) packed into RGBA. Mirrors
    // three.js's `packRGBAToFloat`. Output is the depth used by the point-light
    // shadow pass.
    if (kind == MAT_DISTANCE) {
        let ref_pos = mesh.specular.xyz;
        let near = mesh.params.x;
        let far  = mesh.params.y;
        let dist = length(in.world_pos - ref_pos);
        let v = clamp((dist - near) / max(far - near, 0.0001), 0.0, 1.0);
        // packRGBAToFloat: spread the float into 4 8-bit channels for precision.
        let scaled = fract(v * vec4<f32>(1.0, 256.0, 65536.0, 16777216.0));
        let packed = vec4<f32>(
            scaled.x - scaled.y / 256.0,
            scaled.y - scaled.z / 256.0,
            scaled.z - scaled.w / 256.0,
            scaled.w
        );
        return packed;
    }

    var n_geom = normalize(in.world_normal);
    // DoubleSide / no-cull: back faces keep the authored winding but must
    // shade with a normal that faces the camera (matches three.js).
    if (!front_facing) {
        n_geom = -n_geom;
    }
    // three.js `nonPerturbedNormal`: before any normal or bump map — the
    // clearcoat's normal, and the one the anisotropy frame is built about.
    let n_flat = n_geom;
    let v_dir = normalize(frame.camera_position.xyz - in.world_pos);
    // three.js `getTangentFrame` about it, for anisotropy and the clearcoat
    // normal map: taken here, before any branch, so every material and
    // shader variant reads its screen-space derivatives at the same point.
    let deriv_frame = three_tangent_frame(in.world_pos, n_flat, in.uv);

    // Normal mapping. The slot and FLAG_NORMAL_MAP were already plumbed through
    // from the material; this is where the sampled tangent-space normal finally
    // perturbs the shading normal. `normal_scale` (params2.yz) scales the
    // tangent/bitangent components, so 0 flattens the map and >1 exaggerates it.
    if ((tex_flags & FLAG_NORMAL_MAP) != 0u) {
        let ts = sample_normal_map(in.uv).xyz * 2.0 - vec3<f32>(1.0);
        let tf = surface_frame(n_geom, in.world_pos, in.uv, in.world_tangent);
        let scaled = vec3<f32>(ts.x * mesh.params2.y, ts.y * mesh.params2.z, ts.z);
        let perturbed = tf.t * scaled.x + tf.b * scaled.y + n_geom * scaled.z;
        if (dot(perturbed, perturbed) > 1e-12) {
            n_geom = normalize(perturbed);
        }
    } else if (HAS_BUMP_MAP) {
        n_geom = bump_normal(n_geom, in.world_pos, in.uv, select(-1.0, 1.0, front_facing));
    }

    // Matcap: sample by view-space normal (xy * 0.5 + 0.5).
    if (kind == MAT_MATCAP) {
        // three.js `meshmatcap.glsl`: the view-space normal projected onto a
        // frame built around the view direction (so a sphere off the axis
        // still shows the whole matcap), scaled by 0.495 to stay inside the
        // disc; without a matcap texture, `mix(0.2, 0.8, uv.y)`.
        let view_n = normalize((frame.view * vec4<f32>(n_geom, 0.0)).xyz);
        let view_dir = normalize(-(frame.view * vec4<f32>(in.world_pos, 1.0)).xyz);
        let mx = normalize(vec3<f32>(view_dir.z, 0.0, -view_dir.x));
        let my = cross(view_dir, mx);
        let uv = vec2<f32>(dot(mx, view_n), dot(my, view_n)) * 0.495 + 0.5;
        let sampled = sample_matcap(uv).rgb;
        var mc = vec3<f32>(mix(0.2, 0.8, uv.y));
        if ((tex_flags & FLAG_MATCAP_MAP) != 0u) {
            mc = sampled;
        }
        // three.js `meshmatcap.glsl`: `map_fragment` before the matcap.
        if ((tex_flags & FLAG_MAP) != 0u) {
            let s = sample_map(in.uv);
            base_color = base_color * s.rgb;
            opacity = opacity * s.a;
        }
        opacity = opacity * alpha_map_of(in.uv);
        if (opacity < alpha_test_of()) { discard; }
        let mcol = apply_fog(apply_tone_mapping(base_color * mc), in.view_z);
        return vec4<f32>(framebuffer_encode(mcol), opacity);
    }

    // Planetary atmosphere: shade from how much air the view ray crosses.
    //
    // The shell is not treated as a surface. The ray is intersected with the
    // atmosphere sphere and with the planet, and what is shaded is the length
    // of the segment between them — so the glow thickens toward the limb where
    // the path is long, and stops dead where the planet blocks it. That is one
    // analytic pass on the front faces, and it needs neither a back-face trick
    // nor a stack of nested shells to produce a gradient.
    if (kind == MAT_ATMOSPHERE) {
        let centre = (mesh.model * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
        let r_planet = mesh.params3.x;
        let r_air = max(mesh.params3.y, r_planet + 1e-5);
        let strength = mesh.params3.z;
        let falloff = max(mesh.params3.w, 0.1);

        let eye = frame.camera_position.xyz;
        let dir = normalize(in.world_pos - eye);
        let oc = eye - centre;

        // Ray against the outer shell.
        let b = dot(oc, dir);
        let c_air = dot(oc, oc) - r_air * r_air;
        let disc_air = b * b - c_air;
        if (disc_air <= 0.0) { discard; }
        let root = sqrt(disc_air);
        var t_enter = max(-b - root, 0.0);
        var t_exit = -b + root;
        if (t_exit <= t_enter) { discard; }

        // …and against the planet, which ends the ray early when it is hit.
        let c_planet = dot(oc, oc) - r_planet * r_planet;
        let disc_planet = b * b - c_planet;
        if (disc_planet > 0.0) {
            let t_hit = -b - sqrt(disc_planet);
            if (t_hit > 0.0) { t_exit = min(t_exit, t_hit); }
        }
        if (t_exit <= t_enter) { discard; }

        // March the segment coarsely, weighting by an exponential density that
        // falls off with altitude. A closed form would need a Chapman function;
        // eight samples is cheaper and, on a shell this thin, indistinguishable.
        let steps = 8;
        let dt = (t_exit - t_enter) / f32(steps);
        let depth_scale = max(r_air - r_planet, 1e-5);
        var density = 0.0;
        var lit = 0.0;
        var sun_path = 0.0;
        // Aurora rides along in this march rather than running a second one of
        // its own. Both walk the same segment and want the same positions, and
        // at Retina resolution the duplicate was 0.9 ms a frame — a fifth of
        // the whole thing — for a ring that covers a sliver of the shell.
        let aurora = mesh.params13.w;
        let ring = radians(max(mesh.params14.w, 1.0));
        // Geomagnetic axis, tilted off the model's own up, so the oval sits
        // off-centre the way it does on Earth.
        let mag_tilt = 0.19;
        let mag_axis = normalize(vec3<f32>(sin(mag_tilt) * 0.9, 1.0, sin(mag_tilt) * 0.4));
        var glow_g = 0.0;
        var glow_r = 0.0;
        // Direction *toward* the sun; `direction` is the way the light travels.
        var to_sun = vec3<f32>(0.0, 1.0, 0.0);
        if (frame.light_counts.x > 0u) {
            to_sun = normalize(-frame.dir_lights[0].direction.xyz);
        }
        for (var i = 0; i < steps; i = i + 1) {
            let p = eye + dir * (t_enter + dt * (f32(i) + 0.5));
            let up = p - centre;
            let altitude = (length(up) - r_planet) / depth_scale;
            let d = exp(-max(altitude, 0.0) * falloff) * dt;
            density = density + d;
            // How much sunlight reaches this point: the day side is lit, the
            // night side is not, with a soft terminator.
            let sun_dot = dot(normalize(up), to_sun);
            lit = lit + d * smoothstep(-0.25, 0.35, sun_dot);
            // Reddening is what the *sunbeam* picks up on its way in, so it
            // needs both a grazing sun and thick air to graze through. Gating
            // on the angle alone turns the whole limb of a full-phase planet
            // orange, because from head-on the limb is the terminator; the
            // altitude term keeps the red where the air is dense and leaves the
            // upper shell blue, which is how a real limb reads.
            let low = exp(-max(altitude, 0.0) * falloff * 2.0);
            sun_path = sun_path + d * low * (1.0 - smoothstep(0.0, 0.55, abs(sun_dot)));

            if (aurora > 0.0) {
                // `abs`, so both poles get an oval — the southern lights are
                // the same particles down the other end of the same field
                // lines, and the two are near mirror images.
                let colat = acos(clamp(abs(dot(normalize(up), mag_axis)), -1.0, 1.0));
                let band = exp(-pow((colat - ring) / (ring * 0.34), 2.0));
                // Curtains: the ring is drapes, not a uniform stripe.
                let ph = atan2(dot(normalize(up), vec3<f32>(0.0, 0.0, 1.0)),
                               dot(normalize(up), vec3<f32>(1.0, 0.0, 0.0)));
                let curtain = 0.55 + 0.45 * sin(ph * 9.0) * sin(ph * 23.0 + 1.7);
                let wgt = band * curtain * dt;
                // Green low, red high: both are atomic oxygen, but the 630 nm
                // red transition takes two minutes to radiate and is quenched
                // by a collision long before that unless the air is thin, so it
                // only survives as a crown above the green.
                glow_g = glow_g + wgt * smoothstep(0.04, 0.16, altitude)
                                      * (1.0 - smoothstep(0.22, 0.5, altitude));
                glow_r = glow_r + wgt * smoothstep(0.3, 0.55, altitude)
                                      * (1.0 - smoothstep(0.75, 1.0, altitude)) * 0.35;
            }
        }
        if (density <= 1e-6) { discard; }
        let day = lit / density;
        let grazing = sun_path / density;

        // Rayleigh phase: forward and backward scattering both favoured.
        let mu = dot(dir, to_sun);
        let phase = 0.75 * (1.0 + mu * mu);

        // Optical depth → coverage.
        //
        // Normalised against the longest path any ray can take — the grazing
        // chord at the limb — not against the shell's thickness. Dividing by
        // the thickness makes a ray straight through the middle as opaque as
        // one along the limb, which buries the planet under flat blue.
        let max_chord = 2.0 * sqrt(max(r_air * r_air - r_planet * r_planet, 1e-6));
        let tau = density / max_chord * strength;
        let alpha = (1.0 - exp(-tau)) * day * mesh.params.y;
        if (alpha <= 0.002) { discard; }

        let sunset = mesh.params4.xyz;
        let tint = mix(mesh.color.rgb, sunset, clamp(grazing * 1.15, 0.0, 1.0));
        var col = tint * phase * (0.35 + 0.65 * day);
        var out_a = alpha;

        // Ozone. Scattering only ever *adds* light, and the blue saturates
        // first, so a limb built from Rayleigh alone whitens as it thickens and
        // ends up reading as haze. What keeps a real twilight blue is
        // absorption: ozone's Chappuis band eats the middle of the spectrum —
        // orange and green — and leaves blue alone, and the sunbeam's path near
        // the terminator is long enough through the layer for that to win.
        //
        // Coefficients are the band's shape, normalised to the blue: it takes
        // out roughly three times as much red as blue and twice as much green.
        let ozone = mesh.params13.z;
        if (ozone > 0.0) {
            let sigma = vec3<f32>(3.1, 2.0, 0.35);
            // The grazing term already measures how far the beam ran through
            // dense air on its way in, which is the same path that matters here.
            col = col * exp(-sigma * ozone * grazing);
        }

        // Aurora. Solar wind funnelled down the field lines, hitting the
        // upper atmosphere in a ring around each geomagnetic pole. Accumulated
        // in the density march above; this is only the shading.
        if (aurora > 0.0) {
            let night = 1.0 - day;
            let gg = glow_g / max_chord * aurora;
            let gr = glow_r / max_chord * aurora;
            if (gg + gr > 0.0) {
                let red = vec3<f32>(1.0, 0.22, 0.30);
                col = col + (mesh.params14.xyz * gg + red * gr) * night * 6.0;
                out_a = max(out_a, clamp((gg + gr) * night * 8.0, 0.0, 1.0));
            }
        }
        return vec4<f32>(framebuffer_encode(apply_fog(col, in.view_z)), clamp(out_a, 0.0, 1.0));
    }

    // Sample albedo if a map is bound. The map's alpha multiplies opacity, as
    // three.js does — that is what lets a cloud layer or a cut-out decal be a
    // *lit* material instead of an unlit one. Opaque materials never read the
    // result, so this only affects draws already on the alpha pipeline.
    if ((tex_flags & FLAG_MAP) != 0u) {
        let sampled = sample_map(in.uv);
        base_color = base_color * sampled.rgb;
        opacity = opacity * sampled.a;
    }
    opacity = opacity * alpha_map_of(in.uv);
    // three.js `alphatest_fragment`: after the maps, before any lighting.
    if (opacity < alpha_test_of()) { discard; }
    var emissive = mesh.emissive.rgb;
    if ((tex_flags & FLAG_EMISSIVE_MAP) != 0u) {
        emissive = emissive * sample_emissive_map(in.uv).rgb;
    }
    // A night map is not a lamp. `emissive` is added to the shaded colour
    // unconditionally, which is right for something that glows on its own and
    // wrong for city lights: the map says WHERE the cities are, not WHETHER it
    // is night there, so without this every city burns through full daylight.
    // params8.z (emissive_night_side) gates the term on how far the point is
    // past the terminator, using the brightest directional light as the sun.
    // Zero by default, so nothing that does not ask for it changes.
    if (mesh.params8.z > 0.0 && frame.light_counts.x > 0u) {
        var sun_dir = vec3<f32>(0.0, 0.0, 0.0);
        var best = -1.0;
        for (var i : u32 = 0u; i < frame.light_counts.x; i = i + 1u) {
            if (i >= MAX_LIGHTS_PER_KIND) { break; }
            let l = frame.dir_lights[i];
            let p = dot(l.color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            if (p > best) { best = p; sun_dir = -normalize(l.direction.xyz); }
        }
        if (best > 0.0) {
            // Softened across the terminator rather than switched: a hard edge
            // would put a line of lit windows against unlit ones.
            let cosine = dot(normalize(n_geom), sun_dir);
            let night = 1.0 - smoothstep(-0.10, 0.18, cosine);
            emissive = emissive * mix(1.0, night, clamp(mesh.params8.z, 0.0, 1.0));
        }
    }

    // Lambert, Phong and Toon: three.js r165's `RE_Direct_Lambert`,
    // `RE_Direct_BlinnPhong` and `RE_Direct_Toon` over every light, each
    // receiving the shadows three.js gives it (`directLight.color *=
    // getShadow(...)` when the mesh receives shadows), then the same tone
    // mapping and output encode as every other lit material.
    if (kind == MAT_LAMBERT || kind == MAT_PHONG || kind == MAT_TOON) {
        let RECIP_PI_LP : f32 = 0.3183098861837907;
        let specular = mesh.specular.rgb;
        let shininess = mesh.params.x;
        // `fwidth(coord)/2` for the toon ramp, from the normal's derivatives
        // (taken here, in uniform control flow).
        let dndx = dpdx(n_geom);
        let dndy = dpdy(n_geom);
        let receive = (mesh.flags.z & FLAG_RECEIVE_SHADOW) != 0u;
        if (receive) {
            compute_shadows(in.world_pos, normalize(in.world_normal));
        }
        // Indirect (ambient + hemisphere) and direct irradiance apart: three.js's
        // `aomap_fragment` occludes `reflectedLight.indirectDiffuse` only.
        var indirect = frame.ambient.rgb;
        var irr = vec3<f32>(0.0);
        var spec = vec3<f32>(0.0);
        for (var i : u32 = 0u; i < frame.light_counts.w; i = i + 1u) {
            if (i >= MAX_LIGHTS_PER_KIND) { break; }
            indirect = indirect + shade_hemi(n_geom, frame.hemi_lights[i]);
        }
        indirect = indirect + light_map_irradiance(in.uv);
        if ((tex_flags & FLAG_AO_MAP) != 0u) {
            let ao_texel = sample_ao_map(in.uv).r;
            indirect = indirect * ((ao_texel - 1.0) * mesh.params2.x + 1.0);
        }
        let n_lights = min(frame.light_counts.x, MAX_LIGHTS_PER_KIND) + min(frame.light_counts.y, MAX_LIGHTS_PER_KIND) + min(frame.light_counts.z, MAX_LIGHTS_PER_KIND);
        for (var j : u32 = 0u; j < n_lights; j = j + 1u) {
            var inc : Incident;
            var shadow = 1.0;
            let nd = min(frame.light_counts.x, MAX_LIGHTS_PER_KIND);
            let np = min(frame.light_counts.y, MAX_LIGHTS_PER_KIND);
            if (j < nd) {
                let l = frame.dir_lights[j];
                inc = Incident(-normalize(l.direction.xyz), l.color.rgb);
                if (receive) { shadow = shadow_factor_dir(j); }
            } else if (j < nd + np) {
                inc = incident_point(in.world_pos, frame.point_lights[j - nd]);
                if (receive) { shadow = shadow_factor_point(j - nd); }
            } else {
                inc = incident_spot(in.world_pos, frame.spot_lights[j - nd - np]);
                if (receive) { shadow = shadow_factor_spot(j - nd - np); }
            }
            let color = inc.color * shadow;
            let n_dot_l = dot(n_geom, inc.dir);
            if (kind == MAT_TOON) {
                let dx = 0.5 * dot(dndx, inc.dir);
                let dy = 0.5 * dot(dndy, inc.dir);
                let fw = 0.5 * (abs(dx) + abs(dy));
                irr = irr + color * toon_gradient(n_dot_l, fw, dx, dy);
            } else {
                let e = color * clamp(n_dot_l, 0.0, 1.0);
                irr = irr + e;
                if (kind == MAT_PHONG) {
                    spec = spec + e * brdf_blinn_phong(inc.dir, v_dir, n_geom, specular, shininess);
                }
            }
        }
        // three.js `RE_Direct_BlinnPhong`: `material.specularStrength`.
        spec = spec * specular_strength_of(in.uv);
        var lit = base_color * (indirect + irr) * RECIP_PI_LP + spec + emissive;
        lit = apply_env_map(lit, n_geom, in.world_pos, specular_strength_of(in.uv));
        lit = apply_tone_mapping(lit);
        lit = apply_fog(lit, in.view_z);
        return vec4<f32>(framebuffer_encode(lit), opacity);
    }

    // PBR (Standard / Physical) — common path.
    // Inputs from per-mesh params + optional textures.
    var roughness = mesh.params.z;
    var metalness = mesh.params.w;
    if ((tex_flags & FLAG_ROUGHNESS_MAP) != 0u) {
        roughness = roughness * sample_roughness_map(in.uv).g;
    }
    if ((tex_flags & FLAG_METALNESS_MAP) != 0u) {
        metalness = metalness * sample_metalness_map(in.uv).b;
    }
    // Material roughness before screen-space geometry term — env uses this on WebGPU
    // because SwiftShader dpdx(viewNormal) is much noisier than three.js WebGL dFdx,
    // inflating geometryRoughness and over-blurring CubeUV samples vs the reference.
    let roughness_env = max(roughness, 0.0525);
    // three.js `lights_physical_fragment`: `max(roughness, 0.0525)` plus
    // `geometryRoughness` — the largest screen-space change of any component
    // of the view-space, unperturbed normal — capped at 1.
    //
    // `Renderer::set_specular_antialiasing` swaps that geometry term for
    // geometric specular antialiasing (Kaplanyan et al. 2016 / Filament):
    // the variance of the *shading* normal across the pixel folded into
    // alpha. It absorbs normal-map variance as well as curvature, which is
    // what stops crinkled foil or a fine normal map from sparkling and
    // crawling under motion, at the cost of no longer matching three.js.
    let n_view_flat = (frame.view * vec4<f32>(n_flat, 0.0)).xyz;
    let n_dxy = max(abs(dpdx(n_view_flat)), abs(dpdy(n_view_flat)));
    let geometry_roughness = max(max(n_dxy.x, n_dxy.y), n_dxy.z);
    let specular_aa = frame.light_counts2.y == 1u;
    if (specular_aa) {
        roughness = specular_aa_roughness(n_geom, roughness_env);
    } else {
        roughness = min(roughness_env + geometry_roughness, 1.0);
    }
    var ao = 1.0;
    if ((tex_flags & FLAG_AO_MAP) != 0u) {
        let ao_sample = sample_ao_map(in.uv).r;
        let ao_int = mesh.params2.x;
        ao = 1.0 + (ao_sample - 1.0) * ao_int;
    }

    // Gamma-decode albedo if sampled from an sRGB texture? We treat material
    // color and texture sample as already linear here — three.js does sRGB
    // decode at upload time by texture format; we mirror that via Rgba8UnormSrgb.
    let albedo_lin = base_color;

    // `lit` gathers the diffuse light and `spec_lit` the specular, as three.js's
    // `reflectedLight` keeps them, until transmission has replaced the diffuse.
    var lit = vec3<f32>(0.0);
    var spec_lit = vec3<f32>(0.0);
    // AmbientLight is treated as indirect-diffuse irradiance and passed through
    // BRDF_Lambert (= diffuseColor * RECIPROCAL_PI), where in three.js's PBR
    // diffuseColor = (1 - metalness) * baseColor. Without the metalness factor
    // metals show an unphysical ambient diffuse glow.
    let RECIP_PI : f32 = 0.3183098861837907;
    let diffuse_color = albedo_lin * (1.0 - metalness);
    lit = lit + diffuse_color * (frame.ambient.rgb + light_map_irradiance(in.uv)) * ao * RECIP_PI;

    // three.js r165 `lights_physical_fragment`: the specular lobe's inputs.
    // Standard has no `ior`: 0.04 at normal incidence and 1 at grazing.
    // Physical derives both from `ior`, `specularIntensity` (`specular.w`) and
    // `specularColor` (`specular.rgb`), and their maps.
    var lobe : SpecLobe;
    lobe.alpha = max(roughness * roughness, 0.0016);
    lobe.f0 = mix(vec3<f32>(0.04), albedo_lin, metalness);
    lobe.f90 = 1.0;
    lobe.irid = 0.0;
    lobe.irid_fresnel = vec3<f32>(0.0);
    // Negative: no anisotropy. three.js keeps the anisotropic lobe once
    // `anisotropy` is set, even where its map brings the strength to 0.
    lobe.aniso = -1.0;
    lobe.alpha_t = lobe.alpha;
    lobe.at = vec3<f32>(0.0);
    lobe.ab = vec3<f32>(0.0);
    if (kind == MAT_PHYSICAL) {
        var spec_intensity = mesh.specular.w;
        var spec_tint = mesh.specular.rgb;
        if (HAS_SPECULAR_COLOR_MAP) {
            spec_tint = spec_tint * sample_specular_color_map(in.uv).rgb;
        }
        if (HAS_SPECULAR_INTENSITY_MAP) {
            spec_intensity = spec_intensity * sample_specular_intensity_map(in.uv).a;
        }
        let ior = mesh.params3.z;
        let r0 = (ior - 1.0) / (ior + 1.0);
        lobe.f0 = mix(min(vec3<f32>(r0 * r0) * spec_tint, vec3<f32>(1.0)) * spec_intensity, albedo_lin, metalness);
        lobe.f90 = mix(spec_intensity, 1.0, metalness);
    }

    // Iridescence (`lights_fragment_begin`): the thin film's Fresnel at the
    // view angle, over the base's `specularColor`. A film of 0 nm is none.
    if (kind == MAT_PHYSICAL && mesh.params5.x > 0.0) {
        var irid = mesh.params5.x;
        if (HAS_IRIDESCENCE_MAP) {
            irid = irid * sample_iridescence_map(in.uv).r;
        }
        // `iridescenceThicknessRange`: the maximum (`params5.z`), or between the
        // minimum (`map_params().z`) and it by the thickness map's green.
        var film_nm = mesh.params5.z;
        if (HAS_IRIDESCENCE_THICKNESS_MAP) {
            let lo = map_params().z;
            film_nm = (film_nm - lo) * sample_iridescence_thickness_map(in.uv).g + lo;
        }
        let n_dot_v_i = clamp(dot(n_geom, v_dir), 0.0, 1.0);
        irid = select(clamp(irid, 0.0, 1.0), 0.0, film_nm == 0.0);
        if (irid > 0.0) {
            lobe.irid = irid;
            lobe.irid_fresnel = iridescence_fresnel(1.0, mesh.params5.y, lobe.f0, film_nm, n_dot_v_i);
        }
    }

    // Anisotropy: three.js's `anisotropyVector`, `anisotropy · (cos, sin)` of
    // `anisotropyRotation` (`emissive.w`, `params4.w`), turned by the map's
    // direction and scaled by its strength, in the tangent frame of
    // `normal_fragment_begin` — the `tangent` attribute's, else
    // `getTangentFrame`'s, flipped with a back face.
    let aniso_strength = select(0.0, mesh.emissive.w, kind == MAT_PHYSICAL);
    if (aniso_strength > 0.0) {
        var t = deriv_frame.t;
        var b = deriv_frame.b;
        if (dot(in.world_tangent.xyz, in.world_tangent.xyz) > 1e-12) {
            t = normalize(in.world_tangent.xyz);
            b = normalize(cross(normalize(in.world_normal), t) * select(-1.0, 1.0, in.world_tangent.w >= 0.0));
        }
        if (!front_facing) {
            t = -t;
            b = -b;
        }
        let rot = mesh.params4.w;
        var av = aniso_strength * vec2<f32>(cos(rot), sin(rot));
        if (HAS_ANISOTROPY_MAP) {
            let polar = sample_anisotropy_map(in.uv).rgb;
            av = mat2x2<f32>(av.x, av.y, -av.y, av.x) * normalize(2.0 * polar.rg - vec2<f32>(1.0)) * polar.b;
        }
        var a = length(av);
        if (a == 0.0) {
            av = vec2<f32>(1.0, 0.0);
        } else {
            av = av / a;
            a = clamp(a, 0.0, 1.0);
        }
        lobe.aniso = a;
        lobe.alpha_t = mix(lobe.alpha, 1.0, a * a);
        lobe.at = t * av.x + b * av.y;
        lobe.ab = b * av.x - t * av.y;
    }

    // Sheen (`lights_physical_fragment`): `sheenColor` (× `sheen`, as three.js
    // uploads it) and its map, and `sheenRoughness` clamped to [0.07, 1]
    // before its map scales it.
    let sheen_strength = select(0.0, mesh.params5.w, kind == MAT_PHYSICAL);
    var sheen_tint = mesh.params6.rgb * sheen_strength;
    var sheen_rough = clamp(mesh.params6.w, 0.07, 1.0);
    if (sheen_strength > 0.0) {
        if (HAS_SHEEN_COLOR_MAP) {
            sheen_tint = sheen_tint * sample_sheen_color_map(in.uv).rgb;
        }
        if (HAS_SHEEN_ROUGHNESS_MAP) {
            sheen_rough = sheen_rough * sample_sheen_roughness_map(in.uv).a;
        }
    }
    var sheen_lit = vec3<f32>(0.0);

    let receive_shadow = (mesh.flags.z & FLAG_RECEIVE_SHADOW) != 0u;
    // Each light's shadow once, for the direct loops and the clearcoat ones.
    var sf_dir : array<f32, 8>;
    var sf_point : array<f32, 8>;
    var sf_spot : array<f32, 8>;
    if (receive_shadow) {
        compute_shadows(in.world_pos, normalize(in.world_normal));
    }
    for (var i = 0u; i < MAX_LIGHTS_PER_KIND; i = i + 1u) {
        sf_dir[i] = select(1.0, shadow_factor_dir(i), receive_shadow && i < frame.light_counts.x);
        sf_point[i] = select(1.0, shadow_factor_point(i), receive_shadow && i < frame.light_counts.y);
        sf_spot[i] = select(1.0, shadow_factor_spot(i), receive_shadow && i < frame.light_counts.z);
    }
    let twilight_width = mesh.params10.w;
    // Computed outside the loop: `textureSample` needs uniform control flow,
    // and a call from inside a conditional branch has undefined derivatives.
    var cloud_att = 1.0;
    var eclipse_att = 1.0;
    if (frame.light_counts.x > 0u) {
        let sun_dir = -normalize(frame.dir_lights[0].direction.xyz);
        cloud_att = cloud_shadow(in.local_pos, sun_dir);
        eclipse_att = eclipse_factor(in.world_pos, sun_dir);
    }
    for (var i : u32 = 0u; i < frame.light_counts.x; i = i + 1u) {
        if (i >= MAX_LIGHTS_PER_KIND) { break; }
        let l = frame.dir_lights[i];
        let to_light = -normalize(l.direction.xyz);
        // The first directional light is the key, and the only one a cloud deck
        // is modelled as shadowing.
        let attenuation = sf_dir[i] * select(1.0, cloud_att * eclipse_att, i == 0u);
        let dt = pbr_direct_split(n_geom, v_dir, to_light, diffuse_color, lobe);
        lit = lit + l.color.rgb * dt.diffuse * attenuation;
        spec_lit = spec_lit + l.color.rgb * dt.specular * attenuation;

        // Twilight. A hard Lambert terminator is what an airless body has; on
        // one with an atmosphere, light scatters round the limb and carries a
        // good way onto the night side, reddened by the length of the path it
        // took. Modelled as wrapped diffuse, tinted, and counting only the part
        // beyond what Lambert already gave — so the day side is untouched and
        // only the terminator softens.
        if (twilight_width > 0.0 && i == 0u) {
            // A narrow band at the terminator, and a faint one.
            //
            // The broad orange you see round a planet's night edge from space
            // is the *atmosphere* glowing along the limb — that is the shell's
            // job, and it does it. What reaches the *ground* falls away far
            // faster than the sky does: civil twilight, six degrees past the
            // terminator, is already about a four-hundredth of daylight. An
            // earlier version of this was three orders of magnitude too bright
            // and washed a third of the night side in red.
            let ndl = dot(n_geom, to_light);
            let x = ndl / twilight_width;
            // Tapered off the day side too: there the direct sun lights the
            // ground and the scattered part is a tint on it, not a source.
            let band = exp(-x * x * 3.0) * (1.0 - smoothstep(0.0, 1.0, max(x, 0.0)));
            lit = lit + l.color.rgb * mesh.params11.xyz * albedo_lin * band * 0.15 * attenuation * RECIP_PI;
        }
        // Cloud deck: light through the volume, rather than off a surface.
        //
        // A Lambert shell is brightest facing the sun and black past the
        // terminator, which is the opposite of how a cloud behaves at both
        // ends. Droplets scatter hard forward, so a deck with the sun *behind*
        // it — near the limb, and just past the terminator — is the brightest
        // thing in frame, and its edge lights before its middle does. The
        // phase function carries that, and it does not need `n · l`: the light
        // is coming through the deck, not bouncing off it.
        //
        // Weighted by how much deck the ray crosses, which grows as the view
        // grazes — the silver lining is geometry, not a separate effect.
        let cloud_scatter = mesh.params13.x;
        if (cloud_scatter > 0.0 && i == 0u) {
            // The scattering angle is between the way the light was already
            // travelling (`-to_light`) and the way it has to leave to reach the
            // eye (`v_dir`, surface to camera). Forward scattering is those two
            // agreeing, so a positive `g` peaks when the sun is beyond the deck.
            let cos_theta = dot(-to_light, v_dir);
            let phase = henyey_greenstein(cos_theta, clamp(mesh.params13.y, -0.95, 0.95));
            // Slant path through a thin shell, capped so the limb does not
            // divide by zero.
            let slant = min(1.0 / max(abs(dot(n_geom, v_dir)), 0.08), 8.0);
            // Still gated on the sun being up *somewhere* along the path, which
            // the softened cosine already describes — just far more gently than
            // the surface term, since the deck is lit from the side too.
            let reach = clamp(dot(n_geom, to_light) * 2.0 + 0.55, 0.0, 1.0);
            lit = lit + l.color.rgb * albedo_lin * cloud_scatter * phase * slant * reach * attenuation;
        }
        if (sheen_strength > 0.0) {
            sheen_lit = sheen_lit + l.color.rgb * brdf_sheen(n_geom, v_dir, to_light, sheen_tint, sheen_rough) * attenuation;
        }
    }
    for (var i : u32 = 0u; i < frame.light_counts.y; i = i + 1u) {
        if (i >= MAX_LIGHTS_PER_KIND) { break; }
        let l = frame.point_lights[i];
        let to_light_vec = l.position.xyz - in.world_pos;
        let d = length(to_light_vec);
        let to_light = to_light_vec / max(d, 0.0001);
        let att = punctual_attenuation(d, l.params.x, l.params.y);
        let sf_pt = sf_point[i];
        let pt = pbr_direct_split(n_geom, v_dir, to_light, diffuse_color, lobe);
        lit = lit + l.color.rgb * att * sf_pt * pt.diffuse;
        spec_lit = spec_lit + l.color.rgb * att * sf_pt * pt.specular;
        if (sheen_strength > 0.0) {
            sheen_lit = sheen_lit + l.color.rgb * att * sf_pt * brdf_sheen(n_geom, v_dir, to_light, sheen_tint, sheen_rough);
        }
    }
    for (var i : u32 = 0u; i < frame.light_counts.z; i = i + 1u) {
        if (i >= MAX_LIGHTS_PER_KIND) { break; }
        let l = frame.spot_lights[i];
        let to_light_vec = l.position.xyz - in.world_pos;
        let d = length(to_light_vec);
        let to_light = to_light_vec / max(d, 0.0001);
        let dir = normalize(l.direction.xyz);
        let cos_angle = dot(-to_light, dir);
        var cone = 0.0;
        if (cos_angle > l.params.z) {
            cone = smoothstep(l.params.z, l.params.w, cos_angle);
        }
        let att = punctual_attenuation(d, l.params.x, l.params.y) * cone;
        let sf_sp = sf_spot[i];
        let st = pbr_direct_split(n_geom, v_dir, to_light, diffuse_color, lobe);
        lit = lit + l.color.rgb * att * sf_sp * st.diffuse;
        spec_lit = spec_lit + l.color.rgb * att * sf_sp * st.specular;
        if (sheen_strength > 0.0) {
            sheen_lit = sheen_lit + l.color.rgb * att * sf_sp * brdf_sheen(n_geom, v_dir, to_light, sheen_tint, sheen_rough);
        }
    }
    // HemisphereLight is also indirect-diffuse and metals don't diffuse it.
    for (var i : u32 = 0u; i < frame.light_counts.w; i = i + 1u) {
        if (i >= MAX_LIGHTS_PER_KIND) { break; }
        lit = lit + diffuse_color * shade_hemi(n_geom, frame.hemi_lights[i]) * ao * RECIP_PI;
    }

    // Physical: gather the clearcoat layer's own specular from every light type.
    //
    // The composite is deliberately NOT done here — it happens after the IBL
    // block below, so the clearcoat Fresnel attenuates the *whole* base
    // response (direct + environment) the way three.js does. Compositing here
    // would leave the base's environment reflection un-attenuated, i.e. lacquer
    // that dims the diffuse but not the reflections underneath it.
    var cc = 0.0;
    var cc_roughness = 0.0;
    var cc_fresnel = vec3<f32>(0.0);
    var cc_lit = vec3<f32>(0.0);
    // three.js `geometryClearcoatNormal`: the unperturbed normal — the base's
    // normal and bump maps are under the coat — or the coat's own normal map.
    var cc_n = n_flat;
    if (kind == MAT_PHYSICAL && mesh.params3.x > 0.0) {
        cc = mesh.params3.x;
        cc_roughness = mesh.params3.y;
        if (HAS_CLEARCOAT_MAP) {
            cc = cc * sample_clearcoat_map(in.uv).x;
        }
        if (HAS_CLEARCOAT_ROUGHNESS_MAP) {
            cc_roughness = cc_roughness * sample_clearcoat_roughness_map(in.uv).y;
        }
        if (HAS_CLEARCOAT_NORMAL_MAP) {
            // `clearcoat_normal_fragment_maps`, scaled by `clearcoatNormalScale`
            // (`map_params2().xy`), in `tbn2` — the `tangent` attribute's
            // frame, else `getTangentFrame`'s, flipped with a back face.
            var t = deriv_frame.t;
            var b = deriv_frame.b;
            if (dot(in.world_tangent.xyz, in.world_tangent.xyz) > 1e-12) {
                t = normalize(in.world_tangent.xyz);
                b = normalize(cross(normalize(in.world_normal), t) * select(-1.0, 1.0, in.world_tangent.w >= 0.0));
            }
            if (!front_facing) {
                t = -t;
                b = -b;
            }
            var m = sample_clearcoat_normal_map(in.uv).xyz * 2.0 - vec3<f32>(1.0);
            m = vec3<f32>(m.xy * map_params2().xy, m.z);
            cc_n = normalize(t * m.x + b * m.y + n_flat * m.z);
        }
        cc = clamp(cc, 0.0, 1.0);
        // `max(clearcoatRoughness, 0.0525)` plus the geometry term, as the
        // base's roughness has it.
        if (specular_aa) {
            cc_roughness = specular_aa_roughness(cc_n, max(cc_roughness, 0.0525));
        } else {
            cc_roughness = min(max(cc_roughness, 0.0525) + geometry_roughness, 1.0);
        }
        let cc_a = max(cc_roughness * cc_roughness, 0.0016);
        let fc0 = vec3<f32>(0.04);
        cc_fresnel = f_schlick(clamp(dot(cc_n, v_dir), 0.0, 1.0), fc0);

        for (var i : u32 = 0u; i < frame.light_counts.x; i = i + 1u) {
            if (i >= MAX_LIGHTS_PER_KIND) { break; }
            let l = frame.dir_lights[i];
            let to_light = -normalize(l.direction.xyz);
            let att = sf_dir[i];
            cc_lit = cc_lit + l.color.rgb * att
                * clearcoat_spec(cc_n, v_dir, to_light, cc_a, fc0);
        }
        for (var i : u32 = 0u; i < frame.light_counts.y; i = i + 1u) {
            if (i >= MAX_LIGHTS_PER_KIND) { break; }
            let l = frame.point_lights[i];
            let to_light_vec = l.position.xyz - in.world_pos;
            let d = length(to_light_vec);
            let to_light = to_light_vec / max(d, 0.0001);
            let att = punctual_attenuation(d, l.params.x, l.params.y);
            let sf_pt = sf_point[i];
            cc_lit = cc_lit + l.color.rgb * att * sf_pt
                * clearcoat_spec(cc_n, v_dir, to_light, cc_a, fc0);
        }
        for (var i : u32 = 0u; i < frame.light_counts.z; i = i + 1u) {
            if (i >= MAX_LIGHTS_PER_KIND) { break; }
            let l = frame.spot_lights[i];
            let to_light_vec = l.position.xyz - in.world_pos;
            let d = length(to_light_vec);
            let to_light = to_light_vec / max(d, 0.0001);
            let dir = normalize(l.direction.xyz);
            let cos_angle = dot(-to_light, dir);
            var cone = 0.0;
            if (cos_angle > l.params.z) {
                cone = smoothstep(l.params.z, l.params.w, cos_angle);
            }
            let att = punctual_attenuation(d, l.params.x, l.params.y) * cone;
            let sf_sp = sf_spot[i];
            cc_lit = cc_lit + l.color.rgb * att * sf_sp
                * clearcoat_spec(cc_n, v_dir, to_light, cc_a, fc0);
        }
    }

    // three.js r165 `RE_Direct_RectArea_Physical`: the specular through the
    // linearly transformed cosine fitted to this view and roughness (the
    // `ltc_1` matrix, Hill's `ltc_2` Fresnel), the diffuse through the
    // unit one. No sheen, clearcoat or ambient occlusion: three.js gives a
    // rect light none of them.
    for (var i : u32 = 0u; i < frame.light_counts2.x; i = i + 1u) {
        if (i >= MAX_LIGHTS_PER_KIND) { break; }
        let rl = frame.rect_lights[i];
        let rt = rect_area_light(rl, in.world_pos, n_geom, v_dir, roughness, lobe.f0, diffuse_color);
        lit = lit + rt.diffuse;
        spec_lit = spec_lit + rt.specular;
    }

    // How much of the light arriving here leaves as a specular reflection.
    //
    // Needed below to stop a transparent surface from blending its own
    // reflection away: alpha says how much of what is BEHIND the surface shows
    // through, and a reflection happens in front of it. Filled in by whichever
    // of the two branches below runs.
    var spec_coverage = 0.0;

    // IBL — mirrors three.js MeshStandardMaterial envmap_physical_pars_fragment +
    // RE_IndirectSpecular_Physical (getIBLRadiance + computeMultiscattering).
    // `env_u.info.x`: whether this draw has an environment (1 = on).
    if (env_u.info.x > 0.5) {
        // three.js `computeMultiscatteringIridescence`: the thin film's
        // Fresnel mixed into the reflectance, so an iridescent film tints the
        // environment's reflection too.
        let specular_color = mix(lobe.f0, lobe.irid_fresnel, lobe.irid);
        let specular_f90 = lobe.f90;

        // `getIBLAnisotropyRadiance`: anisotropy bends the normal the
        // environment is reflected about toward the anisotropy bitangent's
        // plane, so a brushed surface smears it along the streak; and
        // `getIBLRadiance` pulls the reflection toward that normal by
        // roughness².
        var refl_normal = n_geom;
        if (lobe.aniso >= 0.0) {
            let bent = normalize(cross(cross(lobe.ab, v_dir), lobe.ab));
            var bend = 1.0 - lobe.aniso * (1.0 - roughness_env);
            bend = bend * bend;
            refl_normal = normalize(mix(bent, n_geom, bend * bend));
        }
        var reflect_vec = reflect(-v_dir, refl_normal);
        reflect_vec = normalize(mix(reflect_vec, refl_normal, roughness_env * roughness_env));
        // NOTE: prefiltering this by the reflection's screen footprint
        // (`dpdx(reflect_vec)`) was tried and measures as a no-op here — on a
        // phone's side rail the reflection direction changes little enough per
        // pixel that the footprint never opens up, and forcing it with a large
        // gain only blurs. The banding down such a rail is plain undersampling
        // of a strongly foreshortened specular surface: it falls off as the
        // supersample factor (0.99 rms at 1x, 0.62 at 3x, 0.30 at 6x) and
        // ignores the normal map, the environment's resolution, and
        // `SPEC_AA_SIGMA` alike.
        let radiance = sample_env_cube(reflect_vec, roughness_env);

        let irradiance = PI * sample_env_cube(n_geom, 1.0);
        let fab = dfg_approx(n_geom, v_dir, roughness_env);
        let fss_ess = specular_color * fab.x + specular_f90 * fab.y;
        let ess = fab.x + fab.y;
        let ems = 1.0 - ess;
        let favg = specular_color + (vec3<f32>(1.0) - specular_color) * 0.047619;
        let fms = fss_ess * favg / (vec3<f32>(1.0) - ems * favg);
        let single_scatter = fss_ess;
        let multi_scatter = fms * ems;
        let total_scatter = single_scatter + multi_scatter;
        let scatter_max = max(max(total_scatter.r, total_scatter.g), total_scatter.b);
        let cosine_weighted_irr = irradiance * RECIP_PI;

        // Occlude the environment before it is added. Both terms are indirect
        // light arriving from the whole room, so both are attenuated by how
        // much of the room this point can actually see.
        let ibl_n_dot_v = clamp(dot(n_geom, v_dir), 0.0, 1.0);
        let spec_occ = specular_occlusion(ibl_n_dot_v, ao, roughness_env);
        spec_lit = spec_lit + (radiance * single_scatter + multi_scatter * cosine_weighted_irr) * spec_occ;
        // The environment's diffuse term, added exactly ONCE.
        //
        // In three.js this line lives inside RE_IndirectSpecular_Physical, and
        // the separate RE_IndirectDiffuse call is fed `irradiance` from ambient
        // lights and light probes — *not* from the env map. threers already adds
        // the ambient term further up, so mirroring both three.js call sites
        // here counted the environment twice: a white Lambertian in a uniform
        // environment measured exactly 2.0× its surroundings, which is
        // impossible (it cannot reflect more light than reaches it).
        lit = lit + diffuse_color * (1.0 - scatter_max) * cosine_weighted_irr * ao;
        // `total_scatter` IS the specular reflectance — single plus multiple
        // scattering, Fresnel and roughness already in it. Exactly the fraction
        // wanted, and free.
        spec_coverage = clamp(scatter_max, 0.0, 1.0);
    } else {
        // No environment to reflect, but the surface still has a reflectance.
        // Same split-sum term the IBL path uses, evaluated for its Fresnel
        // alone, so a transparent metal is opaque whether or not a cube is
        // bound instead of flipping appearance when one is.
        let fab_a = dfg_approx(n_geom, v_dir, roughness_env);
        let refl = lobe.f0 * fab_a.x + vec3<f32>(lobe.f90 * fab_a.y);
        spec_coverage = clamp(max(max(refl.r, refl.g), refl.b), 0.0, 1.0);
    }

    var out_opacity = opacity;

    // threers's reflective transparency (`Renderer::set_reflective_transparency`,
    // off by default): a transparent surface's alpha raised by its specular
    // reflectance — `opacity + (1 − opacity) · reflectance` — so a reflection,
    // which happens in front of the surface and is not transmitted, does not
    // fade with the alpha that lets the background through. three.js blends
    // the whole colour by `opacity`; this keeps polished glass faked with
    // `alphaMode: BLEND` reading as glass rather than grey film.
    if (opacity < 1.0 && frame.light_counts2.z == 1u) {
        out_opacity = clamp(opacity + (1.0 - opacity) * spec_coverage, 0.0, 1.0);
    }

    // three.js r165 `transmission_fragment`: the opaque scene behind, read
    // from the transmission capture along the ray refracted through
    // `thickness` (`getIBLVolumeRefraction`), replaces the diffuse light by
    // `transmission`; and `transmissionAlpha` scales the alpha a transparent
    // material blends with. `transmissionMap`'s red and `thicknessMap`'s green
    // scale the two — sampled here, in uniform control flow.
    var transmission_tex = 1.0;
    var thickness_tex = 1.0;
    if (HAS_TRANSMISSION_MAP) {
        transmission_tex = sample_transmission_map(in.uv).r;
    }
    if (HAS_THICKNESS_MAP) {
        thickness_tex = sample_thickness_map(in.uv).g;
    }
    let blended_glass = (mesh.flags.z & FLAG_BLENDED_GLASS) != 0u;
    // What of the background passes, per channel (blended glass only).
    var glass_through = vec3<f32>(0.0);
    if (kind == MAT_PHYSICAL && mesh.params3.w > 0.0 && blended_glass) {
        // The blended glass modes: the background comes through the blend,
        // so transmission is a share of it — what Fresnel does not reflect,
        // absorbed over `thickness` — and the alpha is what does not pass.
        // The two-draw `Glass` pipelines pass it by channel; one alpha (the
        // OIT accumulation) can only pass the mean.
        let transmission = mesh.params3.w * transmission_tex;
        let thickness = mesh.params4.x * thickness_tex;
        let fab_t = dfg_approx(n_geom, v_dir, roughness);
        let f_t = lobe.f0 * fab_t.x + vec3<f32>(lobe.f90 * fab_t.y);
        let through = (vec3<f32>(1.0) - f_t) * diffuse_color
            * volume_attenuation(thickness, mesh.params7.rgb, mesh.params7.w);
        glass_through = vec3<f32>(1.0) - out_opacity * (vec3<f32>(1.0) - transmission * through);
        out_opacity = out_opacity * (1.0 - transmission * (through.r + through.g + through.b) / 3.0);
        lit = lit * (1.0 - transmission);
    } else if (kind == MAT_PHYSICAL && mesh.params3.w > 0.0) {
        let transmission = mesh.params3.w * transmission_tex;
        let thickness = mesh.params4.x * thickness_tex;
        let transmitted = ibl_volume_refraction(
            n_geom, v_dir, roughness, diffuse_color, lobe.f0, lobe.f90, in.world_pos,
            mesh.params4.y, mesh.params3.z, thickness, mesh.params7.rgb, mesh.params7.w,
        );
        out_opacity = out_opacity * mix(1.0, transmitted.a, transmission);
        lit = mix(lit, transmitted.rgb, transmission);
    }
    lit = lit + spec_lit;

    // three.js `meshphysical` composes `outgoingLight = diffuse + specular +
    // emissive`, then the sheen and then the clearcoat over all of it.
    lit = lit + emissive;
    // General per-vertex emission (Standard/Physical): app bakes color into
    // vertexColor.rgb + a gate into vertexColor.a, scaled by vertex_emissive
    // (params4.z). Zero by default, so no effect unless the material opts in.
    lit = lit + in.vertex_color.rgb * in.vertex_color.a * mesh.params4.z;

    // Sheen: the base darkened by three.js's energy compensation,
    // `1 − 0.157 · max3(sheenColor)`, plus the sheen lobe — direct, and from
    // the environment `iblIrradiance · sheenColor · IBLSheenBRDF`
    // (`RE_IndirectSpecular_Physical`).
    if (sheen_strength > 0.0) {
        if (env_u.info.x > 0.5) {
            let ibl_irradiance = PI * sample_env_cube(n_geom, 1.0);
            sheen_lit = sheen_lit + ibl_irradiance * sheen_tint * ibl_sheen_brdf(n_geom, v_dir, sheen_rough);
        }
        let sheen_max = max(max(sheen_tint.r, sheen_tint.g), sheen_tint.b);
        lit = lit * (1.0 - 0.157 * sheen_max) + sheen_lit;
    }

    // Clearcoat: `outgoingLight · (1 − clearcoat · Fcc) + clearcoatSpecular ·
    // clearcoat`. The coat reflects the environment in its own right
    // (`getIBLRadiance` about its normal, weighted by `EnvironmentBRDF`).
    if (kind == MAT_PHYSICAL && cc > 0.0) {
        if (env_u.info.x > 0.5) {
            var cc_refl = reflect(-v_dir, cc_n);
            cc_refl = normalize(mix(cc_refl, cc_n, cc_roughness * cc_roughness));
            let cc_radiance = sample_env_cube(cc_refl, cc_roughness);
            let cc_fab = dfg_approx(cc_n, v_dir, cc_roughness);
            cc_lit = cc_lit + cc_radiance * (vec3<f32>(0.04) * cc_fab.x + cc_fab.y);
        }
        lit = lit * (vec3<f32>(1.0) - cc_fresnel * cc) + cc_lit * cc;
    }
    lit = apply_tone_mapping(lit);
    lit = apply_fog(lit, in.view_z);
    // The two `Glass` draws (see `GLASS_OUTPUT`): what is behind times `T`,
    // then plus `L`.
    if (GLASS_OUTPUT != 0u) {
        if (blended_glass) {
            // Glass passes the background by channel, and fog in front of it
            // veils that too. `T` is a linear factor; the blend works on
            // encoded values, where it is the encoded factor (the encoding is
            // a power law to within sRGB's toe).
            let t = glass_through * (1.0 - fog_factor(in.view_z));
            glass_t_out = framebuffer_encode(t);
            if (GLASS_OUTPUT == 1u) {
                return vec4<f32>(glass_t_out, 1.0);
            }
            return vec4<f32>(framebuffer_encode(lit), clamp(1.0 - (t.r + t.g + t.b) / 3.0, 0.0, 1.0));
        }
        // Anything else in the mode is an ordinary alpha blend, in halves.
        glass_t_out = vec3<f32>(1.0 - out_opacity);
        if (GLASS_OUTPUT == 1u) {
            return vec4<f32>(glass_t_out, 1.0);
        }
        return vec4<f32>(framebuffer_encode(lit) * out_opacity, out_opacity);
    }
    // The canvas surface is bgra8unorm (not sRGB), so we MUST do the
    // linear→sRGB encoding ourselves; otherwise our linearly computed PBR
    // output writes raw linear bytes and the browser misinterprets them as
    // sRGB, making low-luminance scenes (e.g. spot lights) ~13× too dark.
    let lit_srgb = framebuffer_encode(lit);
    return vec4<f32>(lit_srgb, out_opacity);
}

// Weighted-blended order-independent transparency output (McGuire and
// Bavoil, with McGuire and Mara's coloured transmission). Instead of a single
// blended colour, each fragment contributes to two accumulation buffers:
//   accum   (Rgba16F, additive)        = sum(premultiplied light * weight),
//                                        sum(alpha * weight)
//   reveal  (Rgba16F, multiplicative)  = product(transmittance) per channel,
//                                        product(1 - alpha)
// The resolve pass turns these into an order-independent composite. Shaded
// as `fs_main` shades — the pipeline asks `shade_main` for the light half of
// a blend (`GLASS_OUTPUT` 2) and reads the transmittance it found.
// The point cube stores *radial* distance, not projected depth.
//
// A cube's six faces each have their own frustum, so projected depth is
// per-face and discontinuous across the seams — but the lookup in
// `shadow_factor_point` is by direction and has one number to compare against,
// `length(world - light) / radius`. Storing anything else means the comparison
// never matches, which is what this pass used to do: it ran with no fragment
// stage at all, wrote projected depth, and its shadows silently never appeared.
//
// It lives down here, below `fs_main`, and it has to: a `ShaderMaterial` is
// compiled against everything in this file *above the first* `@fragment`, so an
// entry point placed any earlier silently cuts the preamble off at itself and
// takes `framebuffer_encode`, `sample_env_cube` and every other documented
// helper with it. `the_custom_preamble_carries_the_documented_helpers` in
// `renderer.rs` is what noticed, and what will notice again.
// Whether a shadow caster's fragment at `uv` is cut away. three.js's shadow
// depth material takes the caster's `map` and `alphaTest`
// (`WebGLShadowMap.getDepthMaterial`), not its opacity, so a cut-out casts the
// shadow of what is left of it.
// Its `alphaMap` too.
// Interleaved gradient noise (Jimenez): a per-texel threshold that tiles
// without visible structure under a shadow filter.
fn interleaved_gradient_noise(p : vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// The share of light thin glass lets through: what Fresnel does not reflect
// at either face at normal incidence, filtered by its colour. Zero for every
// other caster, solid glass included.
fn shadow_pass_through() -> f32 {
    if ((mesh.flags.z & FLAG_SHADOW_TRANSMITS) == 0u) { return 0.0; }
    let f0 = ior_to_fresnel0_f(max(mesh.params3.z, 1.0), 1.0);
    let through = mesh.color.rgb * (1.0 - f0) * (1.0 - f0);
    return clamp(mesh.params3.w * (through.r + through.g + through.b) / 3.0, 0.0, 1.0);
}

// Whether a caster's shadow-map texel is dropped: cut out by `alphaTest`, or
// one of the share a transmissive caster lets through — dithered, so the
// shadow filter averages it into a partial shadow (Unity's dithered shadows).
fn shadow_cut(uv : vec2<f32>, frag : vec2<f32>) -> bool {
    if (shadow_pass_through() > interleaved_gradient_noise(floor(frag))) {
        return true;
    }
    let test = alpha_test_of();
    let has_map = (mesh.flags.y & FLAG_MAP) != 0u;
    if (test <= 0.0 || !(has_map || HAS_ALPHA_MAP)) {
        return false;
    }
    var a = 1.0;
    if (has_map) {
        a = sample_map_level(uv).a;
    }
    if (HAS_ALPHA_MAP) {
        a = a * sample_alpha_map_level(uv).g;
    }
    return a < test;
}

@fragment
fn fs_shadow_cut(in : ShadowOut) {
    if (shadow_cut(in.uv, in.position.xy)) { discard; }
}

@fragment
fn fs_shadow_point(in : ShadowPointOut) -> @builtin(frag_depth) f32 {
    if (shadow_cut(in.uv, in.position.xy)) { discard; }
    let radius = max(shadow_pass.light.w, 1e-4);
    return clamp(length(in.world - shadow_pass.light.xyz) / radius, 0.0, 1.0);
}

struct OitOut {
    @location(0) accum  : vec4<f32>,
    @location(1) reveal : vec4<f32>,
};

@fragment
fn fs_oit(in : VsOut, @builtin(front_facing) front_facing : bool) -> OitOut {
    let c = shade_main(in, front_facing);
    var light = c.rgb;
    var through = glass_t_out;
    if (through.x < 0.0) {
        // Returned before the blend halves: straight colour and alpha.
        light = c.rgb * c.a;
        through = vec3<f32>(1.0 - c.a);
    }
    let alpha = clamp(c.a, 0.0, 1.0);

    // McGuire depth weight: nearer fragments weigh more. `view_z` is positive.
    // Floored above zero so clear glass, which blocks almost nothing, still
    // carries its highlight.
    let a = max(alpha, 1e-3);
    let z = max(in.view_z, 1e-3);
    let w = a * clamp(10.0 / (1e-5 + pow(z / 60.0, 3.0) + pow(z / 300.0, 6.0)), 1e-2, 3e3);

    var o : OitOut;
    o.accum  = vec4<f32>(light * w, a * w);
    o.reveal = vec4<f32>(vec3<f32>(1.0) - through, alpha);
    return o;
}

// --- Screen-space refraction glass (TransparencyMode::Refract) --------------
// Project a world point to screen UV (framebuffer top-left origin).
/// A depth read from (or computed for) the view's depth buffer, as standard
/// depth: with reversed depth on (`camera_position.w`) that is `1 - d`. The
/// screen-space marches compare in standard depth.
fn std_depth(d : f32) -> f32 {
    return select(d, 1.0 - d, frame.camera_position.w > 0.5);
}

fn ss_project(p : vec3<f32>) -> vec2<f32> {
    let clip = frame.view_proj * vec4<f32>(p, 1.0);
    let ndc = clip.xy / max(abs(clip.w), 1e-5);
    return ndc * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
}

// March a world-space ray against the captured opaque depth (EEVEE-style
// screen-space trace). Returns (uv.x, uv.y, hit) — hit=1 where the ray crosses
// behind on-screen geometry; else uv is the last on-screen sample point.
fn ss_march(origin : vec3<f32>, dir : vec3<f32>, dist : f32, steps : i32) -> vec3<f32> {
    let dims = frame.viewport_size.xy;
    var prev_uv = ss_project(origin);
    var i = 1;
    loop {
        if (i > steps) { break; }
        let p = origin + dir * (dist * f32(i) / f32(steps));
        let clip = frame.view_proj * vec4<f32>(p, 1.0);
        if (clip.w <= 0.0) { break; }
        let uv = clip.xy / clip.w * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { break; }
        let ray_d = std_depth(clip.z / clip.w);
        let coord = vec2<i32>(uv * dims);
        let scene_d = std_depth(textureLoad(ss_depth_tex, coord, 0).r);
        if (scene_d < 0.9999 && ray_d > scene_d + 2e-4) {
            // Crossed behind captured geometry — refine once for the hit UV.
            return vec3<f32>(mix(prev_uv, uv, 0.5), 1.0);
        }
        prev_uv = uv;
        i = i + 1;
    }
    return vec3<f32>(prev_uv, 0.0);
}

// Accurate screen-space reflection march: linear search for the depth crossing,
// then a binary-search refinement for a precise hit point (returns (uv, hit)).
// This is the "accurate reflections" path — combined with the roughness mip blur
// and (optionally) TAA it resolves reflected geometry at its true position
// rather than a clamped approximation.
fn ss_march_refl(origin : vec3<f32>, dir : vec3<f32>, dist : f32, steps : i32) -> vec3<f32> {
    let dims = frame.viewport_size.xy;
    var prev_t = 0.0;
    var i = 1;
    loop {
        if (i > steps) { break; }
        let t = f32(i) / f32(steps);
        let p = origin + dir * (dist * t);
        let clip = frame.view_proj * vec4<f32>(p, 1.0);
        if (clip.w <= 0.0) { break; }
        let uv = clip.xy / clip.w * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { break; }
        let ray_d = std_depth(clip.z / clip.w);
        let scene_d = std_depth(textureLoad(ss_depth_tex, vec2<i32>(uv * dims), 0).r);
        // Require a *thin* crossing (ray just behind the surface) to reject rays
        // that plunge far behind geometry — the classic SSR thickness test.
        if (scene_d < 0.9999 && ray_d > scene_d + 2e-5 && ray_d < scene_d + 0.02) {
            var lo = prev_t;
            var hi = t;
            for (var k = 0; k < 6; k = k + 1) {
                let mt = (lo + hi) * 0.5;
                let mc = frame.view_proj * vec4<f32>(origin + dir * (dist * mt), 1.0);
                let muv = mc.xy / mc.w * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
                let msd = std_depth(textureLoad(ss_depth_tex, vec2<i32>(muv * dims), 0).r);
                if (std_depth(mc.z / mc.w) > msd + 2e-5) { hi = mt; } else { lo = mt; }
            }
            let hc = frame.view_proj * vec4<f32>(origin + dir * (dist * (lo + hi) * 0.5), 1.0);
            return vec3<f32>(hc.xy / hc.w * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5), 1.0);
        }
        prev_t = t;
        i = i + 1;
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Each light's shadow on a refracting surface, for `fs_ss_glass`.
//
// The refraction pass's group 3 holds the scene capture where the lit pass
// holds the shadow maps — WebGL2's sixteen texture units do not stretch to
// both — so the shadows are looked up here, by a pass over the same surfaces
// with the lit layout, and handed over in one integer texture: five bits of
// shadow per shadow record (0 lit, 31 fully shadowed), six to a word, all
// sixteen. The pass
// writes depth and the refraction pass tests `LessEqual`, so both shade the
// same front-most surface at each pixel.
@fragment
fn fs_glass_shadow(in : VsOut) -> @location(0) vec4<u32> {
    clip_fragment(in.world_pos);
    var w = array<u32, 4>(0u, 0u, 0u, 0u);
    if ((mesh.flags.z & FLAG_RECEIVE_SHADOW) != 0u) {
        compute_shadows(in.world_pos, normalize(in.world_normal));
        for (var r = 0u; r < 16u; r = r + 1u) {
            let q = u32(round((1.0 - clamp(shadow_cache[r], 0.0, 1.0)) * 31.0));
            w[r / 6u] = w[r / 6u] | (q << ((r % 6u) * 5u));
        }
    }
    return vec4<u32>(w[0], w[1], w[2], w[3]);
}

// The shadow of the light whose `color.w` is `color_w`, in what
// `fs_glass_shadow` packed: 1 lit, 0 shadowed.
fn glass_shadow(packed : vec4<u32>, color_w : f32) -> f32 {
    let r = light_shadow_record(color_w);
    if (r < 0 || r >= 16) {
        return 1.0;
    }
    let k = u32(r);
    let q = (packed[k / 6u] >> ((k % 6u) * 5u)) & 31u;
    return 1.0 - f32(q) / 31.0;
}

// Dielectric glass, closer to a Cycles/Principled Glass BSDF: at the surface the
// view ray either refracts or reflects, weighted by Fresnel; both terms trace
// the captured scene in screen space (roughness → mip blur), with the
// environment probe as the off-screen fallback. There is no diffuse/opaque body
// — the glass "color" only tints transmission (absorption). `thickness`
// (params4.x) is the world-space march distance.
@fragment
fn fs_ss_glass(in : VsOut) -> @location(0) vec4<f32> {
    clip_fragment(in.world_pos);
    let dims = frame.viewport_size.xy;
    let screen_uv = in.clip_pos.xy / dims;

    var n = normalize(in.world_normal);
    let v = normalize(frame.camera_position.xyz - in.world_pos);
    // Analytic silhouette coverage: `n·v` falls to zero at a curved
    // silhouette, so `n·v` over its screen-space derivative is how much of
    // the pixel the surface covers. This pass replaces what is behind rather
    // than blending, and the frame cannot multisample under it (its depth
    // march needs a single-sampled copy), so this is its antialiasing.
    let facing = abs(dot(n, v));
    let coverage = clamp(facing / max(fwidth(facing), 1e-4), 0.0, 1.0);
    if (dot(n, v) < 0.0) { n = -n; }              // orient toward the camera
    let ndv = clamp(dot(n, v), 1e-3, 1.0);

    let tint       = mesh.color.rgb * in.vertex_color.rgb;   // what the glass filters by
    let roughness  = clamp(mesh.params.z, 0.02, 1.0);
    let ior        = max(mesh.params3.z, 1.0);
    let dispersion = mesh.params4.y;
    let width      = f32(textureDimensions(ss_color_tex, 0).x);
    // three.js `applyIorToRoughness`: an ior of 1 blurs nothing, 1.5 and up
    // fully; the capture's mips stand in for the transmission target's.
    let lod = log2(width) * roughness * clamp(ior * 2.0 - 2.0, 0.0, 1.0);

    // --- How far the light travels inside: the measured distance to the
    // back face behind this pixel, else the material's `thickness` in the
    // model's scale (three.js's only option) ---
    let near = frame.viewport_size.z;
    let far  = frame.viewport_size.w;
    let back_d = std_depth(textureLoad(ss_back_depth_tex, vec2<i32>(screen_uv * dims), 0).r);
    let scale = vec3<f32>(length(mesh.model[0].xyz), length(mesh.model[1].xyz), length(mesh.model[2].xyz));
    var path = mesh.params4.x * (scale.x + scale.y + scale.z) / 3.0;
    // `thickness` 0 is three.js's thin-walled glass: no volume to cross, so
    // the light leaves where it entered and nothing bends.
    if (back_d < 0.9999 && mesh.params4.x > 0.0) {
        let back_dist = near * far / max(far - back_d * (far - near), 1e-4);
        path = max(back_dist - in.view_z, 0.0) / ndv;
    }

    // Composited in display space. The capture holds the opaque scene as it
    // was drawn — tone-mapped, and encoded on a plain Unorm target — so what
    // is read from it is decoded and not tone-mapped again; light computed
    // here (environment, highlights, emission) is tone-mapped once and added.

    // --- Transmission: three.js r165's volume model. Each channel follows
    // the refracted ray (its own ior with dispersion) for the path length and
    // reads the opaque scene where it leaves; off screen, the environment ---
    let half_spread = (ior - 1.0) * 0.025 * dispersion;
    let iors = vec3<f32>(ior - half_spread, ior, ior + half_spread);
    var transmitted = vec3<f32>(0.0);
    var transmittance = vec3<f32>(0.0);
    let refl_dir = reflect(-v, n);
    let tir = dot(refract(-v, n, 1.0 / ior), refract(-v, n, 1.0 / ior)) < 1e-6;
    for (var c = 0u; c < 3u; c = c + 1u) {
        let r = refract(-v, n, 1.0 / iors[c]);
        let dir = select(normalize(r), refl_dir, dot(r, r) < 1e-6);
        let exit = in.world_pos + dir * path;
        let clip = frame.view_proj * vec4<f32>(exit, 1.0);
        let ndc = clip.xy / max(clip.w, 1e-6);
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        let on_screen = clip.w > 0.0 && all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
        let scene_c = framebuffer_decode(textureSampleLevel(ss_color_tex, ss_color_samp, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), lod).rgb);
        let env_c = apply_tone_mapping(sample_env_cube(dir, roughness));
        transmitted[c] = select(env_c, scene_c, on_screen)[c];
        transmittance[c] = tint[c] * volume_attenuation(path, mesh.params7.rgb, mesh.params7.w)[c];
    }

    // --- Reflection: the scene where a screen-space march finds it, the
    // environment where it does not ---
    var reflection = apply_tone_mapping(sample_env_cube(refl_dir, roughness));
    let rm = ss_march_refl(in.world_pos, refl_dir, max(path, 1.0) * 4.0, 40);
    if (rm.z > 0.5) {
        reflection = framebuffer_decode(textureSampleLevel(ss_color_tex, ss_color_samp, rm.xy, lod).rgb);
    }

    // --- Dielectric Fresnel, as three.js's `EnvironmentBRDF`: what is
    // reflected is not transmitted. Total internal reflection reflects all ---
    let f0 = ior_to_fresnel0_f(ior, 1.0);
    let fab = dfg_approx(n, v, roughness);
    var f = f0 * fab.x + fab.y;
    if (tir) { f = 1.0; }
    let display = (1.0 - f) * transmittance * transmitted + f * reflection;
    var lit = vec3<f32>(0.0);

    // --- The scene's lights: a GGX highlight from each, shadowed. Glass has
    // no diffuse term to light. The shadows come packed from
    // `fs_glass_shadow`: this pass's group 3 has no room for the maps ---
    let a = roughness * roughness;
    let shadows = textureLoad(glass_shadow_tex, vec2<i32>(in.clip_pos.xy), 0);
    let nd = min(frame.light_counts.x, MAX_LIGHTS_PER_KIND);
    let np = min(frame.light_counts.y, MAX_LIGHTS_PER_KIND);
    let n_lights = nd + np + min(frame.light_counts.z, MAX_LIGHTS_PER_KIND);
    for (var j : u32 = 0u; j < n_lights; j = j + 1u) {
        var inc : Incident;
        var shadow = 1.0;
        if (j < nd) {
            let l = frame.dir_lights[j];
            inc = Incident(-normalize(l.direction.xyz), l.color.rgb);
            shadow = glass_shadow(shadows, l.color.w);
        } else if (j < nd + np) {
            inc = incident_point(in.world_pos, frame.point_lights[j - nd]);
            shadow = glass_shadow(shadows, frame.point_lights[j - nd].color.w);
        } else {
            inc = incident_spot(in.world_pos, frame.spot_lights[j - nd - np]);
            shadow = glass_shadow(shadows, frame.spot_lights[j - nd - np].color.w);
        }
        let n_dot_l = clamp(dot(n, inc.dir), 0.0, 1.0);
        if (n_dot_l <= 0.0 || shadow <= 0.0) { continue; }
        let h = normalize(inc.dir + v);
        let n_dot_h = clamp(dot(n, h), 0.0, 1.0);
        let v_dot_h = clamp(dot(v, h), 0.0, 1.0);
        let fl = f0 + (1.0 - f0) * pow(1.0 - v_dot_h, 5.0);
        let spec = d_ggx(n_dot_h, a) * v_ggx_smith_correlated(a, n_dot_l, ndv) * fl;
        lit = lit + inc.color * shadow * n_dot_l * spec;
    }

    // --- Emission ---------------------------------------------------------
    // Material emissive plus the general per-vertex emission term: the app bakes
    // a color into vertexColor.rgb and a gate into vertexColor.a, scaled by the
    // material's vertex_emissive (params4.z). Zero by default (no effect).
    lit = lit + mesh.emissive.rgb + in.vertex_color.rgb * in.vertex_color.a * mesh.params4.z;

    var color = display + apply_tone_mapping(lit);
    color = mix(color, apply_tone_mapping(frame.fog_color.rgb), fog_factor(in.view_z));
    // Partly covered silhouette pixels show the scene behind in proportion.
    let behind = framebuffer_decode(textureSampleLevel(ss_color_tex, ss_color_samp, screen_uv, 0.0).rgb);
    color = mix(behind, color, coverage);
    return vec4<f32>(framebuffer_encode(color), 1.0);
}
"#;

/// Where [`SHADER_SOURCE`] takes its material-map declarations and accessors.
pub const MATERIAL_MAPS_MARKER: &str = "//@@MATERIAL_MAPS@@";

/// The legacy material-map layout: every map at a fixed binding in group 2,
/// one sampler shared by all of them, the uv already transformed by the vertex
/// stage. Custom shader materials compile against it — their documented
/// preamble names `albedo_tex`, `tex_sampler` and the rest.
pub const LEGACY_MATERIAL_MAPS: &str = r#"@group(2) @binding(0) var albedo_tex     : texture_2d<f32>;
@group(2) @binding(1) var normal_tex     : texture_2d<f32>;
@group(2) @binding(2) var roughness_tex  : texture_2d<f32>;
@group(2) @binding(3) var metalness_tex  : texture_2d<f32>;
@group(2) @binding(4) var ao_tex         : texture_2d<f32>;
@group(2) @binding(5) var emissive_tex   : texture_2d<f32>;
@group(2) @binding(6) var matcap_tex     : texture_2d<f32>;
@group(2) @binding(7) var tex_sampler    : sampler;

fn sample_map(uv : vec2<f32>) -> vec4<f32> { return textureSample(albedo_tex, tex_sampler, uv); }
fn sample_map_level(uv : vec2<f32>) -> vec4<f32> { return textureSampleLevel(albedo_tex, tex_sampler, uv, 0.0); }
// A Reflector's mirror view rides the map slot, sampled at a projected uv.
fn sample_mirror(uv : vec2<f32>) -> vec4<f32> { return textureSampleLevel(albedo_tex, tex_sampler, uv, 0.0); }
fn sample_normal_map(uv : vec2<f32>) -> vec4<f32> { return textureSample(normal_tex, tex_sampler, uv); }
fn sample_roughness_map(uv : vec2<f32>) -> vec4<f32> { return textureSample(roughness_tex, tex_sampler, uv); }
fn sample_metalness_map(uv : vec2<f32>) -> vec4<f32> { return textureSample(metalness_tex, tex_sampler, uv); }
fn sample_ao_map(uv : vec2<f32>) -> vec4<f32> { return textureSample(ao_tex, tex_sampler, uv); }
fn sample_emissive_map(uv : vec2<f32>) -> vec4<f32> { return textureSample(emissive_tex, tex_sampler, uv); }
fn sample_matcap(uv : vec2<f32>) -> vec4<f32> { return textureSample(matcap_tex, tex_sampler, uv); }
// Cloud cover shares the matcap binding: the two never meet on one material.
fn sample_cloud(uv : vec2<f32>) -> vec4<f32> { return textureSample(matcap_tex, tex_sampler, uv); }
"#;

/// [`SHADER_SOURCE`] with a material-map prelude in place.
pub fn shader_source_with_maps(prelude: &str) -> String {
    SHADER_SOURCE.replacen(MATERIAL_MAPS_MARKER, prelude, 1)
}

/// The main shader with the legacy material-map layout — what every built-in
/// pipeline and every custom shader material compiles against.
pub fn legacy_shader_source() -> &'static str {
    static SOURCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SOURCE.get_or_init(|| {
        let prelude = format!("{LEGACY_MATERIAL_MAPS}{}", super::material_maps::legacy_extras());
        shader_source_with_maps(&prelude)
    })
}


/// Fullscreen "sample one texture → write" — used for the screen-space glass
/// mip downsample chain and the blit of the opaque capture to the frame target.
/// A single bilinear tap box-filters a 2× downsample and is an identity copy at
/// matched resolution.
pub const SS_BLIT_SHADER: &str = r#"
@group(0) @binding(0) var src_tex  : texture_2d<f32>;
@group(0) @binding(1) var src_samp : sampler;
struct Vo { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32> };
@vertex
fn vs_main(@builtin(vertex_index) vid : u32) -> Vo {
    var o : Vo;
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    o.uv = vec2<f32>(x, y);
    o.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return o;
}
@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    return textureSampleLevel(src_tex, src_samp, uv, 0.0);
}
"#;

/// Temporal anti-aliasing resolve. Reads the current (jittered) frame, the depth,
/// and the previous accumulated history; reconstructs each pixel's world position
/// from depth, reprojects it through the previous camera to fetch history,
/// neighbourhood-clamps that history to the current 3×3 colour box (kills
/// ghosting), and blends. Exact for camera motion over static geometry (no motion
/// vectors); moving geometry falls back to the clamp.
///
/// # Geometry pinned to the camera
///
/// A skybox or starfield is drawn at a fixed offset from the viewer, so its
/// world position changes every frame and this reprojection — which assumes the
/// world holds still — reads history from the wrong pixel. The clamp does not
/// rescue it: a star is one bright pixel on black, so its 3x3 box runs from
/// black to star and a mis-fetched star sits inside it. It shows up as stars
/// arriving as short dashes and doubled dots during fast camera translation.
///
/// The obvious repair — treat everything past some distance as pinned and undo
/// the camera's translation for it — DOES NOT WORK, and it is worth saying why
/// so it is not tried again. Distance cannot separate the two cases. On the
/// approach in this project's film the Moon sits at 7.2e6 units and the
/// starfield just beyond it; the Moon is static and really does shift about 9
/// pixels a frame, while the starfield is pinned and shifts none. A threshold
/// that catches the stars also catches the Moon and smears it.
///
/// Doing this properly needs the renderer to be TOLD which objects are pinned —
/// a per-object flag rasterised into a mask this pass can sample. Until then a
/// caller whose camera translates fast against a pinned sky should turn TAA off
/// for that stretch and let MSAA carry it.
pub const TAA_SHADER: &str = r#"
struct TaaU {
    inv_view_proj  : mat4x4<f32>,  // current, un-jittered
    prev_view_proj : mat4x4<f32>,  // previous, un-jittered
    params         : vec4<f32>,    // x:width y:height z:history-weight w:first-frame
    /// xyz: camera movement since the last frame. Carried for the benefit of a
    /// caller that knows which of its objects are pinned to the viewer; this
    /// pass cannot work that out for itself. See the note on reprojection below.
    /// w: converge mode — 0 for the ordinary feedback blend, 1 when the pose is
    /// held still and the history is a running mean over sRGB-encoded bytes,
    /// 2 when it is a running mean over linear light.
    cam_delta      : vec4<f32>,
    // x: 1 when the depth is reversed; reconstruction wants standard depth.
    flags          : vec4<f32>,
};
@group(0) @binding(0) var cur_tex   : texture_2d<f32>;
@group(0) @binding(1) var hist_tex  : texture_2d<f32>;
// A float texture, not `texture_depth_2d`, so the `textureLoad` below lowers to
// `texelFetch` — GLSL has no depth-sampler equivalent. Depth arrives in `.r`.
@group(0) @binding(2) var taa_depth : texture_2d<f32>;
@group(0) @binding(3) var taa_samp  : sampler;
@group(0) @binding(4) var<uniform> taa : TaaU;

struct Vo { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32> };
@vertex
fn vs_main(@builtin(vertex_index) vid : u32) -> Vo {
    var o : Vo;
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    o.uv = vec2<f32>(x, y);
    o.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return o;
}
@fragment
fn fs_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
    let dims = taa.params.xy;
    let coord = vec2<i32>(uv * dims);
    let cur = textureLoad(cur_tex, coord, 0);
    let converge = taa.cam_delta.w > 0.5;

    // CONVERGE MODE: the pose is not moving, so neither is anything on screen.
    //
    // Everything below this point exists to make a *moving* image hold
    // together: reproject the history through the previous camera, then clamp
    // it to the current pixel's neighbourhood so anything that moved out from
    // under it cannot smear. Both steps are wrong when the camera is parked
    // and the only thing changing is the sub-pixel jitter. Reprojection is a
    // no-op at best and a resample at worst, and the clamp is actively fatal:
    // it pins the running mean inside the 3x3 range of a single sample every
    // frame, so the average can never settle anywhere that one sample would
    // not already have reached. That is the whole point of accumulating.
    //
    // So in this mode: same pixel, no clamp, and a `n/(n+1)` weight, which
    // makes the history the exact mean of every sample so far rather than a
    // 0.9 feedback with a floor under it. That is a true progressive
    // supersample, and unlike rendering at `size x N` it costs one buffer at
    // output size however many samples are asked for — which is what makes it
    // usable at 4K and 8K, where the supersampled target would not fit.
    if (converge) {
        // The mean has to be taken in linear light. Averaging sRGB-encoded
        // bytes is averaging over a convex curve, which is not a rounding
        // difference: on a black-to-white edge Jensen's inequality reads
        // 21/255 where the light says 0. So decode both sides, mix, and
        // re-encode — the history stays in the framebuffer's own space, which
        // is what lets the blit stay a plain copy, and the round trip is exact
        // because the converging history is float.
        //
        // `cam_delta.w` is 2 when the framebuffer is already linear, in which
        // case there is nothing to convert.
        let encoded = taa.cam_delta.w < 1.5;
        var c = cur;
        if (encoded) { c = vec4<f32>(taa_srgb_to_linear(c.rgb), c.a); }
        var mean = c;
        if (taa.params.w <= 0.5) {
            var h = textureLoad(hist_tex, coord, 0);
            if (encoded) { h = vec4<f32>(taa_srgb_to_linear(h.rgb), h.a); }
            mean = mix(c, h, taa.params.z);
        }
        if (encoded) {
            // Negative light is reconstruction ringing, and the encode is
            // undefined below zero.
            mean = vec4<f32>(taa_linear_to_srgb(max(mean.rgb, vec3<f32>(0.0))), mean.a);
        }
        return mean;
    }
    if (taa.params.w > 0.5) { return cur; }          // first frame: seed history

    let raw_d = textureLoad(taa_depth, coord, 0).r;
    let d = select(raw_d, 1.0 - raw_d, taa.flags.x > 0.5);
    if (d >= 0.99999) { return cur; }                // background: nothing to reproject

    // Reconstruct world position from depth, reproject through the prev camera.
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
    let wh = taa.inv_view_proj * ndc;
    let world = wh.xyz / wh.w;
    let pc = taa.prev_view_proj * vec4<f32>(world, 1.0);
    if (pc.w <= 0.0) { return cur; }
    let prev_uv = (pc.xy / pc.w) * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0) { return cur; }

    var hist = textureSampleLevel(hist_tex, taa_samp, prev_uv, 0.0);
    // Neighbourhood colour clamp — the standard TAA anti-ghosting step.
    var mn = cur.rgb;
    var mx = cur.rgb;
    for (var y : i32 = -1; y <= 1; y = y + 1) {
        for (var x : i32 = -1; x <= 1; x = x + 1) {
            let s = textureLoad(cur_tex, coord + vec2<i32>(x, y), 0).rgb;
            mn = min(mn, s);
            mx = max(mx, s);
        }
    }
    hist = vec4<f32>(clamp(hist.rgb, mn, mx), hist.a);
    return mix(cur, hist, taa.params.z);
}

fn taa_srgb_to_linear(c : vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)),
                  c / 12.92,
                  c <= vec3<f32>(0.04045));
}

fn taa_linear_to_srgb(c : vec3<f32>) -> vec3<f32> {
    return select(1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055,
                  c * 12.92,
                  c <= vec3<f32>(0.0031308));
}
"#;

/// A multisampled depth buffer to a single-sampled one, sample 0 of each
/// pixel: what the screen-space refraction pass marches and tests against
/// when the opaque scene under it was multisampled.
pub const DEPTH_RESOLVE_SHADER: &str = r#"
@group(0) @binding(0) var src : texture_depth_multisampled_2d;

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos : vec4<f32>) -> @builtin(frag_depth) f32 {
    return textureLoad(src, vec2<i32>(pos.xy), 0);
}
"#;

/// Resolve pass for weighted-blended OIT: reads the accum + revealage buffers and
/// produces the order-independent composite, alpha-blended over the opaque image.
pub const OIT_RESOLVE_SHADER: &str = r#"
@group(0) @binding(0) var accum_tex  : texture_2d<f32>;
@group(0) @binding(1) var reveal_tex : texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}

// Which half of the composite this pipeline draws: 0 multiplies what is
// behind by the product of the layers' transmittance, per channel; 1 adds
// their weighted mean light, scaled by how much of the pixel they cover.
override RESOLVE_PASS : u32 = 0u;

@fragment
fn fs_main(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(i32(pos.x), i32(pos.y));
    let reveal = textureLoad(reveal_tex, p, 0);
    let accum = textureLoad(accum_tex, p, 0);
    if (accum.a <= 0.0) { discard; }
    if (RESOLVE_PASS == 0u) {
        return vec4<f32>(reveal.rgb, 1.0);
    }
    let cover = 1.0 - reveal.a;
    let avg = accum.rgb / max(accum.a, 1e-5);
    return vec4<f32>(avg * cover, cover);
}
"#;
