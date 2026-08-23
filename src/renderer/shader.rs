//! WGSL sources for the main mesh pass and the post-processing fullscreen pass.
//!
//! - [`SHADER_SOURCE`]: lit/unlit materials, shadows, fog, PBR, skinning
//! - [`POSTFX_SHADER`]: effect kinds 0–30 (copy, FXAA, glitch, bloom, SSAO, …)
//!
//! Post-fx kinds used from JS `EffectComposer` / Rust [`crate::postprocessing`]:
//! | Kind | Effect |
//! |------|--------|
//! | 5, 25, 26 | Digital glitch (snow / no-snow / GLSL-baked disp) |
//! | 1 | FXAA |
//! | 3–4 | Dot screen, halftone |
//! | 6–9 | Bloom chain |
//! | 11–15 | SSAO |

/// Post-fx shader. A single WGSL module with a fullscreen-triangle vertex
/// shader and a switch-driven fragment shader keyed by an `effect_kind`
/// uniform. Each effect path is implemented in this same module so the
/// pipeline can be reused across post-fx classes.
pub const POSTFX_SHADER: &str = r#"
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
struct PostFxUniforms {
    // x: effect kind (0=copy, 1=fxaa, 2=film, 3=dotscreen, 4=halftone, 5=glitch,
    // 6=bloom-threshold, 7=blur-h, 8=blur-v, 9=bloom-composite, 10=sobel,
    // 11=ssao, 12=ssr, 13=ssao-blur, 14=ssao-composite, 16=outline-edge,
    // 17=outline-overlay)
    // y: time (seconds), z/w: reserved
    params: vec4<f32>,
    // Effect-specific parameters: x/y/z/w meaning varies per effect.
    params2: vec4<f32>,
    params3: vec4<f32>,
    /// Resolution (x: width, y: height) in pixels; w=1 flips V when
    /// sampling RTs for swapchain presentation (WebGPU top-first storage).
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
@group(0) @binding(6) var hw_depth_tex: texture_depth_2d;
@group(0) @binding(7) var<storage, read> ssao_kernel_buf: array<vec4<f32>, 32>;
@group(0) @binding(8) var noise_tex: texture_2d<f32>;
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
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 2.0),
    );
    var out: Out;
    out.pos = vec4<f32>(p[vid], 0.0, 1.0);
    out.uv = uv[vid];
    return out;
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
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

fn ssao_kernel_sample(i: u32) -> vec3<f32> {
    return ssao_kernel_buf[i].xyz;
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

fn view_pos_from_depth(uv: vec2<f32>, depth: f32, view_z: f32, proj: mat4x4<f32>, inv_proj: mat4x4<f32>) -> vec3<f32> {
    let clip_w = proj[2].w * view_z + proj[3].w;
    var clip_pos = vec4<f32>((vec3<f32>(uv, depth) - 0.5) * 2.0, 1.0);
    clip_pos = clip_pos * clip_w;
    return (inv_proj * clip_pos).xyz;
}

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
        return textureLoad(hw_depth_tex, coord, 0);
    }
    return textureSampleLevel(depth_tex, depth_sampler, uv, 0.0).r;
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

fn ht_get_sample_channel(point: vec2<f32>, res: vec2<f32>, channel: i32) -> f32 {
    let uv_s = point / res;
    var tex = textureSampleLevel(input_tex, input_sampler, uv_s, 0.0);
    let base = ht_rand(vec2<f32>(floor(point.x), floor(point.y))) * HT_PI2;
    let step_a = HT_PI2 / f32(HT_SAMPLES);
    let dist = u.params2.x * 0.66;
    for (var i: i32 = 0; i < HT_SAMPLES; i = i + 1) {
        let r = base + step_a * f32(i);
        let coord = point + vec2<f32>(cos(r) * dist, sin(r) * dist);
        tex = tex + textureSampleLevel(input_tex, input_sampler, coord / res, 0.0);
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

fn sample_glitch_disp(uv: vec2<f32>) -> f32 {
    let dim = textureDimensions(noise_tex, 0);
    let st = uv * vec2<f32>(dim) - vec2<f32>(0.5);
    let i0 = floor(st);
    let f = st - i0;
    let max_c = vec2<i32>(i32(dim.x) - 1, i32(dim.y) - 1);
    let c00 = textureLoad(noise_tex, clamp(vec2<i32>(i0), vec2<i32>(0), max_c), 0).r;
    let c10 = textureLoad(noise_tex, clamp(vec2<i32>(i0 + vec2<f32>(1.0, 0.0)), vec2<i32>(0), max_c), 0).r;
    let c01 = textureLoad(noise_tex, clamp(vec2<i32>(i0 + vec2<f32>(0.0, 1.0)), vec2<i32>(0), max_c), 0).r;
    let c11 = textureLoad(noise_tex, clamp(vec2<i32>(i0 + vec2<f32>(1.0, 1.0)), vec2<i32>(0), max_c), 0).r;
    let c0 = mix(c00, c10, f.x);
    let c1 = mix(c01, c11, f.x);
    return mix(c0, c1, f.y);
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
    var disp = sample_glitch_disp(p * seed * seed);
    if (u32(u.params.x) == 26u) {
        let dim = textureDimensions(noise_tex, 0);
        let fc_disp = clamp(
            vec2<i32>(uv * vec2<f32>(dim)),
            vec2<i32>(0),
            vec2<i32>(i32(dim.x) - 1, i32(dim.y) - 1),
        );
        disp = textureLoad(noise_tex, fc_disp, 0).r;
    }
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
    let cr = sample_input_linear_f16(p + offset).r;
    let cga = sample_input_linear_f16(p);
    let cb = sample_input_linear_f16(p - offset).b;
    var col = vec4<f32>(cr, cga.g, cb, cga.a);
    if (col_s >= 0.0 && u32(u.params.x) == 5u) {
        let gl_x = uv.x * (res.x - 1.0) + 0.5;
        let gl_y = (1.0 - uv.y) * (res.y - 1.0) + 0.5;
        let xs = floor(gl_x / 0.5);
        let ys = floor(gl_y / 0.5);
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

fn fxaa_tex_center(uv: vec2<f32>) -> vec4<f32> {
    let dim = textureDimensions(input_tex, 0);
    let fc = clamp(
        vec2<i32>(uv * vec2<f32>(dim)),
        vec2<i32>(0),
        vec2<i32>(i32(dim.x) - 1, i32(dim.y) - 1),
    );
    return textureLoad(input_tex, fc, 0);
}

fn fxaa_tex(uv: vec2<f32>) -> vec4<f32> {
    return fxaa_tex_center(uv);
}

fn fxaa_tex_off(uv: vec2<f32>, off: vec2<f32>, rcp: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(input_tex, input_sampler, uv + off * rcp, 0.0);
}

fn fxaa_pixel_shader(uv: vec2<f32>, rcp_frame: vec2<f32>) -> vec4<f32> {
    let edge_threshold: f32 = 0.2;
    let inv_edge_threshold: f32 = 1.0 / edge_threshold;
    let rgbaM = fxaa_tex(uv);
    let rgbaS = fxaa_tex_off(uv, vec2<f32>(0.0, 1.0), rcp_frame);
    let rgbaE = fxaa_tex_off(uv, vec2<f32>(1.0, 0.0), rcp_frame);
    let rgbaN = fxaa_tex_off(uv, vec2<f32>(0.0, -1.0), rcp_frame);
    let rgbaW = fxaa_tex_off(uv, vec2<f32>(-1.0, 0.0), rcp_frame);

    let cN = fxaa_contrast(rgbaM, rgbaN);
    let cS = fxaa_contrast(rgbaM, rgbaS);
    let cE = fxaa_contrast(rgbaM, rgbaE);
    let cW = fxaa_contrast(rgbaM, rgbaW);
    if (max(max(cN, cS), max(cE, cW)) < edge_threshold) {
        return rgbaM;
    }

    var rgbaNy = rgbaN;
    var rgbaSy = rgbaS;
    var contrastN = cN;
    var contrastS = cS;
    var contrastE = cE;
    var contrastW = cW;

    var relativeVContrast = (contrastN + contrastS) - (contrastE + contrastW);
    relativeVContrast = relativeVContrast * inv_edge_threshold;
    var horzSpan = relativeVContrast > 0.0;

    if (abs(relativeVContrast) < 0.3) {
        let dirToEdge = vec2<f32>(
            select(-1.0, 1.0, contrastE > contrastW),
            select(-1.0, 1.0, contrastS > contrastN),
        );
        let rgbaAlongH = fxaa_tex_off(uv, dirToEdge, rcp_frame);
        let matchAlongH = fxaa_contrast(rgbaM, rgbaAlongH);
        let rgbaAlongV = fxaa_tex_off(uv, vec2<f32>(-dirToEdge.x, dirToEdge.y), rcp_frame);
        let matchAlongV = fxaa_contrast(rgbaM, rgbaAlongV);
        relativeVContrast = (matchAlongV - matchAlongH) * inv_edge_threshold;
        if (abs(relativeVContrast) < 0.3) {
            return mix(rgbaM, (rgbaN + rgbaS + rgbaE + rgbaW) * 0.25, 0.4);
        }
        horzSpan = relativeVContrast > 0.0;
    }

    var rgbaRefN = rgbaNy;
    var rgbaRefS = rgbaSy;
    if (!horzSpan) {
        rgbaRefN = rgbaW;
        rgbaRefS = rgbaE;
    }
    var rgbaEdge = rgbaRefN;
    if (fxaa_contrast(rgbaM, rgbaRefN) <= fxaa_contrast(rgbaM, rgbaRefS)) {
        rgbaEdge = rgbaRefS;
    }

    var doneN = false;
    var doneP = false;
    var nDist: f32 = 0.0;
    var pDist: f32 = 0.0;
    var iterationsUsedN: i32 = 0;
    var iterationsUsedP: i32 = 0;

    for (var i: i32 = 0; i < 5; i = i + 1) {
        let increment = f32(i + 1);
        if (!doneN) {
            nDist = nDist + increment;
            let offN = select(vec2<f32>(nDist, 0.0), vec2<f32>(0.0, nDist), horzSpan);
            let sampleN = fxaa_tex_off(uv, offN, rcp_frame);
            doneN = fxaa_contrast(sampleN, rgbaM) > fxaa_contrast(sampleN, rgbaEdge);
            iterationsUsedN = i;
        }
        if (!doneP) {
            pDist = pDist + increment;
            let offP = select(vec2<f32>(-pDist, 0.0), vec2<f32>(0.0, -pDist), horzSpan);
            let sampleP = fxaa_tex_off(uv, offP, rcp_frame);
            doneP = fxaa_contrast(sampleP, rgbaM) > fxaa_contrast(sampleP, rgbaEdge);
            iterationsUsedP = i;
        }
        if (doneN || doneP) {
            break;
        }
    }

    if (!doneP && !doneN) {
        return rgbaM;
    }

    var dist = min(
        select(1.0, f32(iterationsUsedN) / 4.0, doneN),
        select(1.0, f32(iterationsUsedP) / 4.0, doneP),
    );
    dist = pow(dist, 0.5);
    dist = 1.0 - dist;
    return mix(rgbaM, rgbaEdge, dist * 0.5);
}

@fragment
fn fs_main(in: Out) -> @location(0) vec4<f32> {
    let kind = u32(u.params.x);
    let res = u.resolution.xy;
    let texel = 1.0 / max(res, vec2<f32>(1.0));
    var uv = in.uv;
    if (u.resolution.w > 0.5) {
        uv.y = 1.0 - uv.y;
    }
    let base = textureSampleLevel(input_tex, input_sampler, uv, 0.0);

    // 0: copy / pass-through — point sample so RT pixels map 1:1 to canvas
    // (linear filtering bleeds foreground into background at silhouettes).
    if (kind == 0u) {
        let dim = textureDimensions(input_tex, 0);
        let fc = clamp(
            vec2<i32>(uv * vec2<f32>(dim)),
            vec2<i32>(0),
            vec2<i32>(i32(dim.x) - 1, i32(dim.y) - 1),
        );
        let px = textureLoad(input_tex, fc, 0);
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
        let col = fxaa_pixel_shader(uv, rcp);
        return postfx_passthrough(col.rgb, col.a);
    }
    if (kind == 2u) {
        let t = u.params.y;
        let intensity = u.params2.x;
        let grayscale = u.params2.y > 0.5;
        let noise = fract(sin(dot(fract(uv + vec2<f32>(t)), vec2<f32>(12.9898, 78.233))) * 43758.5453);
        var col = base.rgb + base.rgb * clamp(0.1 + noise, 0.0, 1.0);
        col = mix(base.rgb, col, intensity);
        if (grayscale) {
            let lum = dot(col, vec3<f32>(0.2126729, 0.7151522, 0.0721750));
            col = vec3<f32>(lum);
        }
        return postfx_passthrough(col, base.a);
    }
    // 3: Dot screen (three.js DotScreenShader).
    if (kind == 3u) {
        let col = sample_input_linear_f16(uv);
        let angle = u.params2.x;
        let scale = max(u.params2.y, 0.0001);
        let center = vec2<f32>(u.params2.z, u.params2.w);
        var pattern = 0.0;
        if (u.params3.w > 0.5) {
            let pdim = textureDimensions(noise_tex, 0);
            let pfc = clamp(
                vec2<i32>(uv * vec2<f32>(pdim)),
                vec2<i32>(0),
                vec2<i32>(i32(pdim.x) - 1, i32(pdim.y) - 1),
            );
            pattern = textureLoad(noise_tex, pfc, 0).r;
        } else {
            let s = sin(angle); let c = cos(angle);
            let tex = uv * res - center;
            let point = vec2<f32>(c * tex.x - s * tex.y, s * tex.x + c * tex.y) * scale;
            pattern = sin(point.x) * sin(point.y) * 4.0;
        }
        let average = (col.r + col.g + col.b) / 3.0;
        let dot_val = average * 10.0 - 5.0 + pattern;
        return postfx_passthrough(vec3<f32>(dot_val), col.a);
    }
    // 4: RGB Halftone (three.js HalftoneShader).
    if (kind == 4u) {
        return postfx_halftone(uv, res, base);
    }
    // 5: DigitalGlitch with snow; 25: same without snow; 26: snow-free + GLSL-baked disp.
    if (kind == 5u || kind == 25u || kind == 26u) {
        return postfx_glitch(uv, res, base, in.pos.xy);
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
            textureSampleLevel(input_tex, input_sampler, uv + hx, 0.0).r
            - textureSampleLevel(input_tex, input_sampler, uv - hx, 0.0).r
        ) * 0.5;
        let diff2 = (
            textureSampleLevel(input_tex, input_sampler, uv + hy, 0.0).r
            - textureSampleLevel(input_tex, input_sampler, uv - hy, 0.0).r
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
            acc = acc + (textureSampleLevel(input_tex, input_sampler, uv + uv_offset, 0.0)
                + textureSampleLevel(input_tex, input_sampler, uv - uv_offset, 0.0)) * w;
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
        if (depth_r >= 0.9999) {
            return vec4<f32>(1.0, 1.0, 1.0, 1.0);
        }
        let view_z = perspective_depth_to_view_z(depth_r, near, far);
        let view_pos = view_pos_from_depth(uv, depth_r, view_z, u.proj, u.inv_proj);
        let view_normal = unpack_view_normal(textureSampleLevel(normal_tex, depth_sampler, uv, 0.0).rgb);
        let noise_r = sample_ssao_noise(uv, res);
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
            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                continue;
            }
            let sample_depth_r = sample_postfx_depth(sample_uv, use_hw);
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
        let center = textureSampleLevel(depth_tex, depth_sampler, uv, 0.0).r;
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
            let sample_d = textureSampleLevel(depth_tex, depth_sampler, mirror_uv, 0.0).r;
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
                result = result + textureSampleLevel(input_tex, input_sampler, uv + off, 0.0).r;
            }
        }
        let blurred = result / 25.0;
        return vec4<f32>(blurred, blurred, blurred, 1.0);
    }
    // 14: SSAO composite (single-pass scene * AO).
    if (kind == 14u) {
        let ao = textureSampleLevel(depth_tex, depth_sampler, uv, 0.0).r;
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
    // 25: Reinhard tone mapping. params2.x = exposure multiplier.
    if (kind == 25u) {
        let exposure = max(u.params2.x, 0.0001);
        let mapped = vec3<f32>(1.0) - exp(-base.rgb * exposure);
        return postfx_write(mapped, base.a);
    }
    // 26: Bokeh disc blur. params2.y = max blur radius in pixels.
    if (kind == 26u) {
        let max_blur = u.params2.y;
        let coc = clamp((1.0 - base.a) * max_blur, 0.0, max_blur);
        var acc = base.rgb;
        var wsum = 1.0;
        for (var i: i32 = 0; i < 8; i = i + 1) {
            let angle = f32(i) * 0.785398;
            let r = coc * texel * f32(i + 1);
            let off = vec2<f32>(cos(angle), sin(angle)) * r;
            let s = textureSampleLevel(input_tex, input_sampler, uv + off, 0.0).rgb;
            acc = acc + s;
            wsum = wsum + 1.0;
        }
        return postfx_write(acc / wsum, base.a);
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
    // 30: Output / display encode. params2.x > 0.5 applies sRGB encode.
    if (kind == 30u) {
        if (u.params2.x > 0.5) {
            return vec4<f32>(postfx_encode(base.rgb), base.a);
        }
        return vec4<f32>(base.rgb, base.a);
    }
    return vec4<f32>(postfx_encode(base.rgb), base.a);
}
"#;

pub const MAX_DIR_LIGHTS: usize = 4;
pub const MAX_POINT_LIGHTS: usize = 4;
pub const MAX_SPOT_LIGHTS: usize = 4;
pub const MAX_HEMI_LIGHTS: usize = 4;
pub const MAX_RECT_LIGHTS: usize = 2;

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

const FLAG_MAP           : u32 = 1u;
const FLAG_NORMAL_MAP    : u32 = 2u;
const FLAG_ROUGHNESS_MAP : u32 = 4u;
const FLAG_METALNESS_MAP : u32 = 8u;
const FLAG_AO_MAP        : u32 = 16u;
const FLAG_EMISSIVE_MAP  : u32 = 32u;
const FLAG_MATCAP_MAP    : u32 = 64u;
const FLAG_DISPLACEMENT_MAP : u32 = 2048u;
const FLAG_CLOUD_SHADOW     : u32 = 4096u;
const FLAG_DASHED        : u32 = 128u;
const FLAG_SHADOW_MAT    : u32 = 256u;
const FLAG_RECEIVE_SHADOW: u32 = 512u;
const FLAG_IRIDESCENCE_MAP: u32 = 1024u;

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
    shadow_vp       : mat4x4<f32>,
    spot_shadow_vp  : mat4x4<f32>,
    /// View-projection matrix for the cube face currently being rendered to.
    cube_face_vp    : mat4x4<f32>,
    camera_position : vec4<f32>,
    ambient         : vec4<f32>,
    light_counts    : vec4<u32>,
    /// x: dir-shadow on, y: dir bias, z: spot-shadow on, w: spot bias
    shadow_params   : vec4<f32>,
    point_shadow_pos: vec4<f32>,
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
    dir_lights      : array<DirLight,   4>,
    point_lights    : array<PointLight, 4>,
    spot_lights     : array<SpotLight,  4>,
    hemi_lights     : array<HemiLight,  4>,
    rect_lights     : array<RectLight,  2>,
    /// x: rect-area light count (yzw reserved).
    light_counts2   : vec4<u32>,
    /// One texel of each shadow map: x directional, y spot.
    shadow_texel    : vec4<f32>,
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
    /// x: ao_intensity, y: normal_scale.x, z: normal_scale.y, w: toon_steps
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
};

@group(0) @binding(0) var<uniform> frame : FrameUniforms;
@group(1) @binding(0) var<uniform> mesh  : MeshUniforms;

@group(2) @binding(0) var albedo_tex     : texture_2d<f32>;
@group(2) @binding(1) var normal_tex     : texture_2d<f32>;
@group(2) @binding(2) var roughness_tex  : texture_2d<f32>;
@group(2) @binding(3) var metalness_tex  : texture_2d<f32>;
@group(2) @binding(4) var ao_tex         : texture_2d<f32>;
@group(2) @binding(5) var emissive_tex   : texture_2d<f32>;
@group(2) @binding(6) var matcap_tex     : texture_2d<f32>;
@group(2) @binding(7) var tex_sampler    : sampler;
@group(2) @binding(8) var iridescence_thickness_tex : texture_2d<f32>;
@group(2) @binding(9) var displacement_tex : texture_2d<f32>;

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
    let cover = textureSample(matcap_tex, tex_sampler, vec2<f32>(u, v)).a;
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
    if ((mesh.flags.y & FLAG_DISPLACEMENT_MAP) == 0u) {
        return position;
    }
    let h = textureSampleLevel(displacement_tex, tex_sampler, uv, 0.0).r;
    return position + normalize(normal) * (h * mesh.params8.x + mesh.params8.y);
}

@group(3) @binding(0) var env_tex              : texture_cube<f32>;
@group(3) @binding(1) var env_sampler          : sampler;
@group(3) @binding(8) var env_cube_uv_tex      : texture_2d<f32>;
@group(3) @binding(2) var shadow_tex           : texture_depth_2d;
@group(3) @binding(3) var shadow_sampler       : sampler_comparison;
@group(3) @binding(4) var spot_shadow_tex      : texture_depth_2d;
@group(3) @binding(5) var spot_shadow_sampler  : sampler_comparison;
@group(3) @binding(6) var point_shadow_tex     : texture_depth_cube;
@group(3) @binding(7) var point_shadow_sampler : sampler_comparison;
// Screen-space glass: mipmapped capture of the opaque scene + its trilinear
// sampler. Only read by fs_ss_glass; a 1×1 placeholder is bound otherwise.
@group(3) @binding(9)  var ss_color_tex   : texture_2d<f32>;
@group(3) @binding(10) var ss_color_samp  : sampler;
@group(3) @binding(11) var ss_depth_tex   : texture_depth_2d;
@group(3) @binding(12) var ss_back_depth_tex : texture_depth_2d;

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

@vertex
fn vs_main(in : VsIn) -> VsOut {
    var out : VsOut;
    let uv = transform_uv(in.uv);
    let local_p = apply_displacement(in.position, in.normal, uv);
    let world_p = mesh.model * vec4<f32>(local_p, 1.0);
    out.world_pos = world_p.xyz;
    out.clip_pos = frame.view_proj * world_p;
    // Sky: pin near far plane (three.js sets gl_Position.z = gl_Position.w). WebGPU
    // depth is [0, 1] with Less compare vs a 1.0 clear — z must stay strictly below w.
    if (mesh.flags.x == MAT_SKY) {
        out.clip_pos.z = out.clip_pos.w * 0.999;
    }
    out.clip_zw = out.clip_pos.zw;
    let n4 = mesh.normal_matrix * vec4<f32>(in.normal, 0.0);
    out.world_normal = n4.xyz;
    out.uv = uv;
    let view_p = frame.view * world_p;
    out.view_z = -view_p.z;
    out.vertex_color = in.color;
    out.proj_uv = mirror_proj_uv(in.position);
    out.world_tangent = world_tangent(in.tangent);
    out.local_pos = local_p;
    return out;
}

@vertex
fn vs_shadow(in : VsIn) -> @builtin(position) vec4<f32> {
    // Displaced geometry has to cast the shadow its displaced silhouette
    // implies, or the terrain and its shadow disagree.
    return frame.shadow_vp * mesh.model
        * vec4<f32>(apply_displacement(in.position, in.normal, transform_uv(in.uv)), 1.0);
}

@vertex
fn vs_shadow_spot(in : VsIn) -> @builtin(position) vec4<f32> {
    return frame.spot_shadow_vp * mesh.model * vec4<f32>(in.position, 1.0);
}

@vertex
fn vs_shadow_point(in : VsIn) -> @builtin(position) vec4<f32> {
    return frame.cube_face_vp * mesh.model * vec4<f32>(in.position, 1.0);
}

struct SkinnedVsIn {
    @location(0) position : vec3<f32>,
    @location(1) normal   : vec3<f32>,
    @location(2) uv       : vec2<f32>,
    @location(3) color    : vec4<f32>,
    @location(4) joints   : vec4<f32>,
    @location(5) weights  : vec4<f32>,
    @location(8) tangent  : vec4<f32>,
};

@group(1) @binding(1) var<storage, read> bone_matrices : array<mat4x4<f32>>;

@vertex
fn vs_skinned(in : SkinnedVsIn) -> VsOut {
    var out : VsOut;
    let j = vec4<u32>(u32(in.joints.x), u32(in.joints.y), u32(in.joints.z), u32(in.joints.w));
    let skin = bone_matrices[j.x] * in.weights.x
             + bone_matrices[j.y] * in.weights.y
             + bone_matrices[j.z] * in.weights.z
             + bone_matrices[j.w] * in.weights.w;
    let skin_uv = transform_uv(in.uv);
    let world_p = skin * vec4<f32>(apply_displacement(in.position, in.normal, skin_uv), 1.0);
    out.world_pos = world_p.xyz;
    out.clip_pos = frame.view_proj * world_p;
    out.clip_zw = out.clip_pos.zw;
    let n4 = skin * vec4<f32>(in.normal, 0.0);
    out.world_normal = n4.xyz;
    out.uv = skin_uv;
    let view_p = frame.view * world_p;
    out.view_z = -view_p.z;
    out.vertex_color = in.color;
    out.proj_uv = mirror_proj_uv(in.position);
    out.world_tangent = world_tangent(in.tangent);
    out.local_pos = in.position;
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
};

@vertex
fn vs_instanced(in : InstancedVsIn) -> VsOut {
    var out : VsOut;
    let instance_mat = mat4x4<f32>(in.imat0, in.imat1, in.imat2, in.imat3);
    let model = mesh.model * instance_mat;
    let uv = transform_uv(in.uv);
    let world_p = model * vec4<f32>(apply_displacement(in.position, in.normal, uv), 1.0);
    out.world_pos = world_p.xyz;
    out.clip_pos = frame.view_proj * world_p;
    out.clip_zw = out.clip_pos.zw;
    let n4 = mesh.normal_matrix * vec4<f32>(in.normal, 0.0);
    out.world_normal = n4.xyz;
    out.uv = uv;
    let view_p = frame.view * world_p;
    out.view_z = -view_p.z;
    out.vertex_color = in.color;
    out.proj_uv = mirror_proj_uv(in.position);
    out.world_tangent = world_tangent(in.tangent);
    out.local_pos = in.position;
    return out;
}

@vertex
fn vs_sprite(in : VsIn) -> VsOut {
    var out : VsOut;
    let center_world = vec4<f32>(mesh.model[3].xyz, 1.0);
    let center_view = frame.view * center_world;
    let sx = length(mesh.model[0].xyz);
    let sy = length(mesh.model[1].xyz);
    let view_pos = vec4<f32>(
        center_view.x + in.position.x * sx,
        center_view.y + in.position.y * sy,
        center_view.z,
        1.0,
    );
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

fn aces_tonemap(x : vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn gamma_to_linear(c : vec3<f32>) -> vec3<f32> { return pow(c, vec3<f32>(2.2)); }
fn linear_to_gamma(c : vec3<f32>) -> vec3<f32> { return pow(c, vec3<f32>(1.0 / 2.2)); }

// IEC 61966-2-1 sRGB OETF — matches three.js `sRGBTransferOETF`.
fn linear_to_srgb(c : vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let lo = c * 12.92;
    let hi = pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) * 1.055 - vec3<f32>(0.055);
    return mix(hi, lo, step(c, cutoff));
}

// tone_mapping_exposure.w > 0.5: write linear (HalfFloat / postfx RT), else sRGB-encode for canvas.
fn framebuffer_encode(c: vec3<f32>) -> vec3<f32> {
    let clamped = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    if (frame.tone_mapping_exposure.w > 0.5) {
        return clamped;
    }
    return linear_to_srgb(clamped);
}

fn view_z_to_depth_tex(view_z_pos: f32, near: f32, far: f32) -> f32 {
    let view_z = -view_z_pos;
    return (far + near * far / view_z) / (far - near);
}

// Apply fog (linear or exp2) to a linear-space color based on view-space depth.
// Mirrors three.js's fog: linear factor = smoothstep(near, far, dist),
// exp2 factor = 1 - exp(-density^2 * dist^2).
fn apply_fog(color : vec3<f32>, view_z : f32) -> vec3<f32> {
    let mode = frame.fog_params.w;
    if (mode < 0.5) { return color; }
    let dist = max(view_z, 0.0);
    var factor : f32 = 0.0;
    if (mode < 1.5) {
        let near = frame.fog_params.x;
        let far  = frame.fog_params.y;
        factor = clamp((dist - near) / (far - near), 0.0, 1.0);
    } else {
        let d = frame.fog_params.z;
        factor = 1.0 - exp(-d * d * dist * dist);
    }
    return mix(color, frame.fog_color.rgb, factor);
}

fn shadow_factor(world_pos : vec3<f32>, world_normal : vec3<f32>) -> f32 {
    if (frame.shadow_params.x < 0.5) { return 1.0; }
    // NORMAL BIAS: step off the surface before looking the depth up.
    //
    // A constant depth bias has to cover the worst case — a face nearly edge-on
    // to the light, where one shadow texel spans a long slope and the stored
    // depth is far from this fragment's. Sized for that, it is far too much
    // everywhere else and shadows visibly detach from their casters. Moving the
    // lookup along the surface normal instead scales with the geometry rather
    // than the angle, so the grazing case is covered without paying for it on
    // faces that point at the light.
    let biased = world_pos + world_normal * frame.shadow_texel.y;
    let p = frame.shadow_vp * vec4<f32>(biased, 1.0);
    let ndc = p.xyz / max(p.w, 0.0001);
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { return 1.0; }
    if (ndc.z < 0.0 || ndc.z > 1.0) { return 1.0; }
    let bias = frame.shadow_params.y;

    // PERCENTAGE-CLOSER FILTERING, 3x3.
    //
    // A single compare sample gives a binary answer per pixel, so the shadow
    // edge is the shadow map's own pixel grid, drawn at whatever size the
    // camera happens to magnify it to — a staircase that crawls as the light
    // or the object moves. Comparing nine neighbours and averaging the results
    // turns that into a one-texel gradient, which reads as an edge rather than
    // as a grid. The comparison sampler does the depth test per tap, so this is
    // nine compares, not nine fetches plus arithmetic.
    //
    // Nine and not twenty-five: at 2 mm a texel the penumbra this fakes is
    // already finer than the geometry it falls on, and in vacuum a real shadow
    // edge IS nearly hard — the sun is half a degree wide and there is no air
    // to scatter into it. This is anti-aliasing, not softness.
    var sum = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let o = vec2<f32>(f32(dx), f32(dy)) * frame.shadow_texel.x;
            sum = sum + textureSampleCompareLevel(
                shadow_tex, shadow_sampler, uv + o, ndc.z - bias);
        }
    }
    return sum / 9.0;
}

fn shadow_factor_spot(world_pos : vec3<f32>) -> f32 {
    if (frame.shadow_params.z < 0.5) { return 1.0; }
    let p = frame.spot_shadow_vp * vec4<f32>(world_pos, 1.0);
    let ndc = p.xyz / max(p.w, 0.0001);
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { return 1.0; }
    if (ndc.z < 0.0 || ndc.z > 1.0) { return 1.0; }
    let bias = frame.shadow_params.w;
    return textureSampleCompareLevel(spot_shadow_tex, spot_shadow_sampler, uv, ndc.z - bias);
}

fn shadow_factor_point(world_pos : vec3<f32>) -> f32 {
    let radius = frame.point_shadow_pos.w;
    if (radius <= 0.0) { return 1.0; }
    let to_frag = world_pos - frame.point_shadow_pos.xyz;
    let dist = length(to_frag);
    if (dist > radius) { return 1.0; }
    let depth = dist / radius;
    return textureSampleCompareLevel(point_shadow_tex, point_shadow_sampler, normalize(to_frag), depth - 0.005);
}

// ---- Lambert/Phong (kept for backward compatibility) ----

fn shade_dir_lp(n : vec3<f32>, view_dir : vec3<f32>, kind : u32, specular : vec3<f32>, shininess : f32, l : DirLight) -> vec3<f32> {
    let to_light = -normalize(l.direction.xyz);
    let lambert = max(dot(n, to_light), 0.0);
    var col = l.color.rgb * lambert;
    if (kind == MAT_PHONG && lambert > 0.0) {
        let half_v = normalize(to_light + view_dir);
        let spec = pow(max(dot(n, half_v), 0.0), max(shininess, 1.0));
        col = col + l.color.rgb * specular * spec;
    }
    return col;
}

fn shade_point_lp(world_pos : vec3<f32>, n : vec3<f32>, view_dir : vec3<f32>, kind : u32, specular : vec3<f32>, shininess : f32, l : PointLight) -> vec3<f32> {
    let to_light_vec = l.position.xyz - world_pos;
    let d = length(to_light_vec);
    let to_light = to_light_vec / max(d, 0.0001);
    let lambert = max(dot(n, to_light), 0.0);
    let att = punctual_attenuation(d, l.params.x, l.params.y);
    var col = l.color.rgb * lambert * att;
    if (kind == MAT_PHONG && lambert > 0.0) {
        let half_v = normalize(to_light + view_dir);
        let spec = pow(max(dot(n, half_v), 0.0), max(shininess, 1.0));
        col = col + l.color.rgb * specular * spec * att;
    }
    return col;
}

fn shade_spot_lp(world_pos : vec3<f32>, n : vec3<f32>, view_dir : vec3<f32>, kind : u32, specular : vec3<f32>, shininess : f32, l : SpotLight) -> vec3<f32> {
    let to_light_vec = l.position.xyz - world_pos;
    let d = length(to_light_vec);
    let to_light = to_light_vec / max(d, 0.0001);
    let dir = normalize(l.direction.xyz);
    let cos_angle = dot(-to_light, dir);
    let cos_outer = l.params.z;
    let cos_inner = l.params.w;
    var cone = 0.0;
    if (cos_angle > cos_outer) {
        cone = smoothstep(cos_outer, cos_inner, cos_angle);
    }
    let lambert = max(dot(n, to_light), 0.0);
    let att = punctual_attenuation(d, l.params.x, l.params.y) * cone;
    var col = l.color.rgb * lambert * att;
    if (kind == MAT_PHONG && lambert > 0.0) {
        let half_v = normalize(to_light + view_dir);
        let spec = pow(max(dot(n, half_v), 0.0), max(shininess, 1.0));
        col = col + l.color.rgb * specular * spec * att;
    }
    return col;
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

fn g_smith(n_dot_v : f32, n_dot_l : f32, a : f32) -> f32 {
    let k = (a + 1.0) * (a + 1.0) / 8.0;
    let gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

fn f_schlick(v_dot_h : f32, f0 : vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
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
    let cube_uv_max_mip = frame.env_map_params.z;
    let texel_w = frame.env_map_params.x;
    let texel_h = frame.env_map_params.y;
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
    let cube_uv_max_mip = frame.env_map_params.z;
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

fn sample_env_cube(dir : vec3<f32>, roughness : f32) -> vec3<f32> {
    if (frame.env_map_params.w > 0.5) {
        return texture_cube_uv(dir, roughness);
    }
    let mip = clamp((roughness + 2.0) / 10.0 * 7.0, 0.0, 7.0);
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
    let g = g_smith(n_dot_v, n_dot_l, a);

    let spec = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metalness);
    let diff = kd * albedo / PI;
    // Softened by the sun's angular size — see `soft_lambert`. `params11.w` is
    // zero for anything that has not set a sun, which gives back `max(ndl, 0)`.
    return (diff + spec) * soft_lambert(dot(n, l), mesh.params11.w);
}

// Anisotropic GGX specular (Filament / Burley). `an` in [-1, 1] biases the GGX
// alpha along the tangent (`at`) vs. bitangent (`ab`), stretching the highlight
// into a streak. T and B span the surface tangent plane (n = T × B). The Smith
// visibility term is height-correlated and already carries the 1/(4·NoV·NoL)
// denominator, so specular = D · V · F with no extra divide.
fn pbr_brdf_aniso(
    n : vec3<f32>, v : vec3<f32>, l : vec3<f32>, t : vec3<f32>, b : vec3<f32>,
    albedo : vec3<f32>, roughness : f32, metalness : f32, an : f32, f0 : vec3<f32>,
) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    let a = max(roughness * roughness, 0.0016);
    let at = max(a * (1.0 + an), 0.0016);
    let ab = max(a * (1.0 - an), 0.0016);

    let t_dot_h = dot(t, h);
    let b_dot_h = dot(b, h);
    let t_dot_v = dot(t, v);
    let b_dot_v = dot(b, v);
    let t_dot_l = dot(t, l);
    let b_dot_l = dot(b, l);

    // D — anisotropic GGX normal distribution.
    let a2 = at * ab;
    let dv = vec3<f32>(ab * t_dot_h, at * b_dot_h, a2 * n_dot_h);
    let d2 = dot(dv, dv);
    let w2 = a2 / max(d2, 1e-8);
    let d = a2 * w2 * w2 * (1.0 / PI);

    // V — height-correlated Smith visibility (anisotropic).
    let lambda_v = n_dot_l * length(vec3<f32>(at * t_dot_v, ab * b_dot_v, n_dot_v));
    let lambda_l = n_dot_v * length(vec3<f32>(at * t_dot_l, ab * b_dot_l, n_dot_l));
    let vis = 0.5 / max(lambda_v + lambda_l, 1e-5);

    let f = f_schlick(v_dot_h, f0);

    let spec = d * vis * f;
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metalness);
    let diff = kd * albedo / PI;
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

fn v_neubelt(n_dot_v : f32, n_dot_l : f32) -> f32 {
    return 1.0 / max(4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v), 1e-5);
}

/// Directional albedo of the Charlie sheen lobe, E(NoV, roughness).
///
/// three.js ships this as a 32×32 lookup texture; this is the analytic fit from
/// Estevez & Kulla's course notes, which stays within a few percent of the LUT
/// and costs no extra binding. Peaks at grazing angles, which is what gives
/// sheen its signature bright rim under environment lighting.
fn sheen_env_albedo(n_dot_v : f32, roughness : f32) -> f32 {
    let r = clamp(roughness, 0.07, 1.0);
    // Fitted coefficients interpolating between the smooth and rough regimes.
    let a = mix(21.5473, 25.3245, r);
    let b = mix(3.82987, 3.32435, r);
    let c = mix(0.19823, 0.16801, r);
    let d = mix(-1.97760, -1.27393, r);
    let e = mix(-4.32054, -4.85315, r);
    let x = clamp(n_dot_v, 0.0, 1.0);
    let v = a / (1.0 + b * pow(x, c)) + d * x + e;
    return clamp(exp2(v), 0.0, 1.0);
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
fn f_schlick_f90(f0 : vec3<f32>, f90 : vec3<f32>, v_dot_h : f32) -> vec3<f32> {
    let w = pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
    return f0 + (f90 - f0) * w;
}

fn f_schlick_f90_f(f0 : f32, f90 : f32, v_dot_h : f32) -> f32 {
    let w = pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
    return f0 + (f90 - f0) * w;
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

// ---------------------------------------------------------------------------
// Rectangular area lights.
//
// three.js uses linearly-transformed cosines (LTC), which needs two 64×64
// lookup textures. This is the LUT-free alternative: an exact analytic form
// factor for the diffuse lobe, and Karis' representative-point approximation
// for the specular one. The diffuse term is correct; the specular term is an
// approximation that gets the size, shape and softness of the highlight right,
// which is the reason to reach for an area light in the first place.
// ---------------------------------------------------------------------------

/// Exact Lambertian form factor of a quad, integrated edge by edge.
///
/// Sums the signed solid angle of each edge as seen from the shading point.
/// Unlike a point-light approximation this stays correct when the light is
/// large and close — the case where a rectangle stops looking like a point.
fn rect_diffuse_factor(
    n : vec3<f32>, p : vec3<f32>,
    c0 : vec3<f32>, c1 : vec3<f32>, c2 : vec3<f32>, c3 : vec3<f32>,
) -> f32 {
    let v0 = normalize(c0 - p);
    let v1 = normalize(c1 - p);
    let v2 = normalize(c2 - p);
    let v3 = normalize(c3 - p);

    var sum = 0.0;
    // Each edge contributes acos(v_i · v_j) weighted by how much its swept
    // plane faces the shading normal.
    let e0 = cross(v0, v1);
    let e1 = cross(v1, v2);
    let e2 = cross(v2, v3);
    let e3 = cross(v3, v0);
    if (dot(e0, e0) > 1e-12) {
        sum = sum + acos(clamp(dot(v0, v1), -1.0, 1.0)) * dot(normalize(e0), n);
    }
    if (dot(e1, e1) > 1e-12) {
        sum = sum + acos(clamp(dot(v1, v2), -1.0, 1.0)) * dot(normalize(e1), n);
    }
    if (dot(e2, e2) > 1e-12) {
        sum = sum + acos(clamp(dot(v2, v3), -1.0, 1.0)) * dot(normalize(e2), n);
    }
    if (dot(e3, e3) > 1e-12) {
        sum = sum + acos(clamp(dot(v3, v0), -1.0, 1.0)) * dot(normalize(e3), n);
    }
    return max(sum * 0.5 / PI, 0.0);
}

/// Point on the rectangle that best represents its specular contribution:
/// where the mirror ray hits the light's plane, clamped to the rectangle.
fn rect_representative_point(
    p : vec3<f32>, refl : vec3<f32>,
    center : vec3<f32>, right : vec3<f32>, up : vec3<f32>,
) -> vec3<f32> {
    let plane_n = cross(right, up);
    let denom = dot(refl, plane_n);
    var hit = center;
    if (abs(denom) > 1e-6) {
        let t = dot(center - p, plane_n) / denom;
        // Behind the surface: fall back to the centre rather than mirroring.
        if (t > 0.0) {
            hit = p + refl * t;
        }
    }
    let d = hit - center;
    let hw = length(right);
    let hh = length(up);
    let rx = select(vec3<f32>(0.0), right / hw, hw > 1e-6);
    let uy = select(vec3<f32>(0.0), up / hh, hh > 1e-6);
    let x = clamp(dot(d, rx), -hw, hw);
    let y = clamp(dot(d, uy), -hh, hh);
    return center + rx * x + uy * y;
}

/// Widen the GGX lobe to account for the light's angular size, and renormalise
/// so total energy is preserved (Karis, "Real Shading in Unreal Engine 4").
fn rect_spec_normalisation(alpha : f32, light_radius : f32, dist : f32) -> f32 {
    let a_prime = clamp(alpha + light_radius / max(2.0 * dist, 1e-4), 0.0, 1.0);
    let ratio = alpha / max(a_prime, 1e-4);
    return ratio * ratio;
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
    let g = g_smith(n_dot_v, n_dot_l, cc_a);
    let f = f_schlick(v_dot_h, fc0);
    let spec = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 0.0001);
    return spec * n_dot_l;
}

// Direct-light PBR dispatch: isotropic GGX unless the material carries an
// anisotropy strength, in which case the anisotropic lobe is used. `f0` is
// passed in rather than derived so the iridescence layer can override it.
fn pbr_direct(
    n : vec3<f32>, v : vec3<f32>, l : vec3<f32>, t : vec3<f32>, b : vec3<f32>,
    albedo : vec3<f32>, roughness : f32, metalness : f32, an : f32, f0 : vec3<f32>,
) -> vec3<f32> {
    if (abs(an) > 0.001) {
        return pbr_brdf_aniso(n, v, l, t, b, albedo, roughness, metalness, an, f0);
    }
    return pbr_brdf(n, v, l, albedo, roughness, metalness, f0);
}

@fragment
fn fs_main(in : VsOut, @builtin(front_facing) front_facing : bool) -> @location(0) vec4<f32> {
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
            let s = textureSample(albedo_tex, tex_sampler, in.uv);
            col = col * s.rgb;
            a = a * s.a;
        }
        if (mesh.flags.w != 0u) {
            let threshold = bitcast<f32>(mesh.flags.w);
            if (a < threshold) { discard; }
        }
        if ((mesh.flags.z & FLAG_SHADOW_MAT) != 0u) {
            let sf = shadow_factor(in.world_pos, normalize(in.world_normal));
            let darkness = 1.0 - sf;
            return vec4<f32>(0.0, 0.0, 0.0, opacity * darkness);
        }
        col = apply_fog(col, in.view_z);
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
            let s = textureSampleLevel(albedo_tex, tex_sampler, uv, 0.0).rgb;
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
        return vec4<f32>(framebuffer_encode(col), 1.0);
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
        let col = apply_fog(line_col, in.view_z);
        return vec4<f32>(framebuffer_encode(col), opacity);
    }
    if (kind == MAT_SPRITE) {
        var col = base_color;
        if ((tex_flags & FLAG_MAP) != 0u) {
            let s = textureSample(albedo_tex, tex_sampler, in.uv);
            col = col * s.rgb;
        }
        col = apply_fog(col, in.view_z);
        return vec4<f32>(framebuffer_encode(col), opacity);
    }
    // Window-space depth [0,1] from @builtin(position) (matches GPU depth buffer).
    if (kind == MAT_DEPTH) {
        let depth = 0.5 * in.clip_zw.x / in.clip_zw.y + 0.5;
        let vis = 1.0 - depth;
        return vec4<f32>(vis, vis, vis, 1.0);
    }
    if (kind == MAT_NORMAL) {
        // View-space normals (matches three.js MeshNormalMaterial for SSAO prepass).
        let view_n = normalize((frame.view * vec4<f32>(normalize(in.world_normal), 0.0)).xyz);
        return vec4<f32>(view_n * 0.5 + 0.5, 1.0);
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

        // three.js Sky uses default NoToneMapping + sRGB output color space, so
        // the equivalent of #include <colorspace_fragment> is the linear→gamma
        // encode here. Clamp the input to [0,1] to match GL's framebuffer write.
        return vec4<f32>(framebuffer_encode(retColor), 1.0);
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
    let v_dir = normalize(frame.camera_position.xyz - in.world_pos);

    // Normal mapping. The slot and FLAG_NORMAL_MAP were already plumbed through
    // from the material; this is where the sampled tangent-space normal finally
    // perturbs the shading normal. `normal_scale` (params2.yz) scales the
    // tangent/bitangent components, so 0 flattens the map and >1 exaggerates it.
    if ((tex_flags & FLAG_NORMAL_MAP) != 0u) {
        let ts = textureSample(normal_tex, tex_sampler, in.uv).xyz * 2.0 - vec3<f32>(1.0);
        let tf = surface_frame(n_geom, in.world_pos, in.uv, in.world_tangent);
        let scaled = vec3<f32>(ts.x * mesh.params2.y, ts.y * mesh.params2.z, ts.z);
        let perturbed = tf.t * scaled.x + tf.b * scaled.y + n_geom * scaled.z;
        if (dot(perturbed, perturbed) > 1e-12) {
            n_geom = normalize(perturbed);
        }
    }

    // Matcap: sample by view-space normal (xy * 0.5 + 0.5).
    if (kind == MAT_MATCAP) {
        // View-space normal.
        let view_n = normalize((frame.view * vec4<f32>(n_geom, 0.0)).xyz);
        let uv = view_n.xy * 0.5 + 0.5;
        var mc = vec3<f32>(1.0);
        if ((tex_flags & FLAG_MATCAP_MAP) != 0u) {
            mc = textureSample(matcap_tex, tex_sampler, uv).rgb;
        }
        let mcol = apply_fog(base_color * mc, in.view_z);
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
        let sampled = textureSample(albedo_tex, tex_sampler, in.uv);
        base_color = base_color * sampled.rgb;
        opacity = opacity * sampled.a;
    }
    var emissive = mesh.emissive.rgb;
    if ((tex_flags & FLAG_EMISSIVE_MAP) != 0u) {
        emissive = emissive * textureSample(emissive_tex, tex_sampler, in.uv).rgb;
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
            if (i >= 4u) { break; }
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

    // Toon: stepped Lambert from directional + ambient. Mirrors three.js's
    // MeshToonMaterial where the diffuse term goes through BRDF_Lambert
    // (= albedo / π) — without /π the lit side over-saturates.
    if (kind == MAT_TOON) {
        let RECIP_PI : f32 = 0.3183098861837907;
        let steps = max(f32(bitcast<u32>(mesh.params2.w)), 1.0);
        var sum = frame.ambient.rgb;
        for (var i : u32 = 0u; i < frame.light_counts.x; i = i + 1u) {
            if (i >= 4u) { break; }
            let l = frame.dir_lights[i];
            let to_light = -normalize(l.direction.xyz);
            let lambert = max(dot(n_geom, to_light), 0.0);
            let stepped = floor(lambert * steps) / max(steps - 1.0, 1.0);
            sum = sum + l.color.rgb * stepped;
        }
        let lit = base_color * sum * RECIP_PI + emissive;
        let fogged = apply_fog(lit, in.view_z);
        return vec4<f32>(framebuffer_encode(fogged), opacity);
    }

    // Lambert / Phong path. three.js's MeshLambertMaterial applies BRDF_Lambert
    // (= diffuseColor / PI) to the direct-light irradiance (dotNL * lightColor),
    // so direct contributions need a 1/PI factor. Ambient and hemi are sent in
    // pre-multiplied by PI on the JS side (because they're indirect irradiance),
    // so they pass through without the divide and the result is just
    // ambient * diffuseColor. We pre-multiply our ambient/hemi inputs by PI
    // on the JS side too so the math here mirrors three.js exactly.
    if (kind == MAT_LAMBERT || kind == MAT_PHONG) {
        let LAMBERT_RECIP_PI : f32 = 0.3183098861837907;
        var direct_sum = vec3<f32>(0.0);
        var indirect_sum = frame.ambient.rgb;
        let specular = mesh.specular.rgb;
        let shininess = mesh.params.x;
        for (var i : u32 = 0u; i < frame.light_counts.x; i = i + 1u) {
            if (i >= 4u) { break; }
            direct_sum = direct_sum + shade_dir_lp(n_geom, v_dir, kind, specular, shininess, frame.dir_lights[i]);
        }
        for (var i : u32 = 0u; i < frame.light_counts.y; i = i + 1u) {
            if (i >= 4u) { break; }
            direct_sum = direct_sum + shade_point_lp(in.world_pos, n_geom, v_dir, kind, specular, shininess, frame.point_lights[i]);
        }
        for (var i : u32 = 0u; i < frame.light_counts.z; i = i + 1u) {
            if (i >= 4u) { break; }
            direct_sum = direct_sum + shade_spot_lp(in.world_pos, n_geom, v_dir, kind, specular, shininess, frame.spot_lights[i]);
        }
        for (var i : u32 = 0u; i < frame.light_counts.w; i = i + 1u) {
            if (i >= 4u) { break; }
            indirect_sum = indirect_sum + shade_hemi(n_geom, frame.hemi_lights[i]);
        }
        let light_sum = indirect_sum + direct_sum * LAMBERT_RECIP_PI;
        let lit = base_color * light_sum + emissive;
        return vec4<f32>(apply_fog(lit, in.view_z), opacity);
    }

    // PBR (Standard / Physical) — common path.
    // Inputs from per-mesh params + optional textures.
    var roughness = mesh.params.z;
    var metalness = mesh.params.w;
    if ((tex_flags & FLAG_ROUGHNESS_MAP) != 0u) {
        roughness = roughness * textureSample(roughness_tex, tex_sampler, in.uv).g;
    }
    if ((tex_flags & FLAG_METALNESS_MAP) != 0u) {
        metalness = metalness * textureSample(metalness_tex, tex_sampler, in.uv).b;
    }
    // Material roughness before screen-space geometry term — env uses this on WebGPU
    // because SwiftShader dpdx(viewNormal) is much noisier than three.js WebGL dFdx,
    // inflating geometryRoughness and over-blurring CubeUV samples vs the reference.
    let roughness_env = max(roughness, 0.0525);
    // Geometric specular antialiasing (Kaplanyan et al. 2016 / Filament).
    //
    // Replaces three.js's `geometryRoughness`, which adds a max-of-derivatives
    // term directly in *perceptual* roughness space. That both over-blurs
    // silhouettes and under-corrects the actual problem, which is variance: a
    // pixel covering many differently-oriented microfacets is physically a
    // wider specular lobe. Estimating that variance and folding it into alpha
    // is what stops crinkled foil from sparkling and crawling under motion.
    //
    // Note this reads the *shading* normal, so it also absorbs normal-map
    // variance, not just geometric curvature — which is exactly what a
    // high-frequency crinkle map needs.
    roughness = specular_aa_roughness(n_geom, roughness_env);
    var ao = 1.0;
    if ((tex_flags & FLAG_AO_MAP) != 0u) {
        let ao_sample = textureSample(ao_tex, tex_sampler, in.uv).r;
        let ao_int = mesh.params2.x;
        ao = 1.0 + (ao_sample - 1.0) * ao_int;
    }

    // Gamma-decode albedo if sampled from an sRGB texture? We treat material
    // color and texture sample as already linear here — three.js does sRGB
    // decode at upload time by texture format; we mirror that via Rgba8UnormSrgb.
    let albedo_lin = base_color;

    var lit = vec3<f32>(0.0);
    // AmbientLight is treated as indirect-diffuse irradiance and passed through
    // BRDF_Lambert (= diffuseColor * RECIPROCAL_PI), where in three.js's PBR
    // diffuseColor = (1 - metalness) * baseColor. Without the metalness factor
    // metals show an unphysical ambient diffuse glow.
    let RECIP_PI : f32 = 0.3183098861837907;
    let diffuse_color = albedo_lin * (1.0 - metalness);
    lit = lit + diffuse_color * frame.ambient.rgb * ao * RECIP_PI;

    // Anisotropy: strength is packed in emissive.w for physical materials (0 =
    // isotropic), the in-plane rotation in params4.w. The direct-light BRDF
    // stretches the specular lobe along this frame into a brushed streak.
    //
    // The frame is a UV-aligned cotangent basis (Mikkelsen's derivative trick),
    // solved from screen-space derivatives of world position against UV. That
    // makes the streak direction a property of the *surface* rather than of the
    // viewer, which is what lets anisotropy_rotation mean anything. A mesh with
    // no UVs has a degenerate UV Jacobian, so fall back to the older
    // position-only frame — still a plausible streak, just view-dependent.
    let aniso = select(0.0, mesh.emissive.w, kind == MAT_PHYSICAL);
    var tangent = vec3<f32>(1.0, 0.0, 0.0);
    var bitangent = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(aniso) > 0.001) {
        let tf = surface_frame(n_geom, in.world_pos, in.uv, in.world_tangent);
        tangent = tf.t;
        bitangent = tf.b;

        // Rotate the frame within the surface plane (Rodrigues about n_geom,
        // which reduces to a plain 2D rotation for in-plane vectors).
        let rot = mesh.params4.w;
        if (abs(rot) > 1e-6) {
            let cr = cos(rot);
            let sr = sin(rot);
            let t_rot = tangent * cr + bitangent * sr;
            bitangent = bitangent * cr - tangent * sr;
            tangent = t_rot;
        }
    }

    // Specular F0. Dielectrics sit at 0.04; for metals the base color *is* the
    // reflectance. The iridescence layer, when enabled, replaces this with the
    // thin-film reflectance so both the direct and IBL lobes pick it up.
    let base_f0 = mix(vec3<f32>(0.04), albedo_lin, metalness);
    var spec_f0 = base_f0;
    if (kind == MAT_PHYSICAL && mesh.params5.x > 0.0) {
        let irid = clamp(mesh.params5.x, 0.0, 1.0);
        let film_ior = max(mesh.params5.y, 1.0);
        var film_nm = max(mesh.params5.z, 0.0);
        // Thickness map (three.js `iridescenceThicknessMap`): the green channel
        // scales the material thickness, so a noise or gradient turns one flat
        // hue into an oil-slick sweep across the surface.
        if ((tex_flags & FLAG_IRIDESCENCE_MAP) != 0u) {
            film_nm = film_nm * textureSample(iridescence_thickness_tex, tex_sampler, in.uv).g;
        }
        let n_dot_v_i = clamp(dot(n_geom, v_dir), 0.0, 1.0);
        let irid_f = iridescence_fresnel(1.0, film_ior, base_f0, film_nm, n_dot_v_i);
        spec_f0 = mix(base_f0, clamp(irid_f, vec3<f32>(0.0), vec3<f32>(1.0)), irid);
    }

    // Sheen — a grazing-angle lobe layered over the base, accumulated per light
    // alongside the main BRDF below.
    let sheen_strength = select(0.0, clamp(mesh.params5.w, 0.0, 1.0), kind == MAT_PHYSICAL);
    let sheen_tint = mesh.params6.rgb * sheen_strength;
    let sheen_rough = clamp(mesh.params6.w, 0.07, 1.0);
    var sheen_lit = vec3<f32>(0.0);

    let receive_shadow = (mesh.flags.z & FLAG_RECEIVE_SHADOW) != 0u;
    let sf = select(1.0, shadow_factor(in.world_pos, normalize(in.world_normal)), receive_shadow);
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
        if (i >= 4u) { break; }
        let l = frame.dir_lights[i];
        let to_light = -normalize(l.direction.xyz);
        // The first directional light is the key, and the only one a cloud deck
        // is modelled as shadowing.
        let attenuation = select(1.0, sf * cloud_att * eclipse_att, i == 0u);
        lit = lit + l.color.rgb * pbr_direct(n_geom, v_dir, to_light, tangent, bitangent, albedo_lin, roughness, metalness, aniso, spec_f0) * attenuation;

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
        if (i >= 4u) { break; }
        let l = frame.point_lights[i];
        let to_light_vec = l.position.xyz - in.world_pos;
        let d = length(to_light_vec);
        let to_light = to_light_vec / max(d, 0.0001);
        let att = punctual_attenuation(d, l.params.x, l.params.y);
        let sf_pt = select(1.0, select(1.0, shadow_factor_point(in.world_pos), receive_shadow), i == 0u);
        lit = lit + l.color.rgb * att * sf_pt * pbr_direct(n_geom, v_dir, to_light, tangent, bitangent, albedo_lin, roughness, metalness, aniso, spec_f0);
        if (sheen_strength > 0.0) {
            sheen_lit = sheen_lit + l.color.rgb * att * sf_pt * brdf_sheen(n_geom, v_dir, to_light, sheen_tint, sheen_rough);
        }
    }
    for (var i : u32 = 0u; i < frame.light_counts.z; i = i + 1u) {
        if (i >= 4u) { break; }
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
        let sf_spot = select(1.0, select(1.0, shadow_factor_spot(in.world_pos), receive_shadow), i == 0u);
        lit = lit + l.color.rgb * att * sf_spot * pbr_direct(n_geom, v_dir, to_light, tangent, bitangent, albedo_lin, roughness, metalness, aniso, spec_f0);
        if (sheen_strength > 0.0) {
            sheen_lit = sheen_lit + l.color.rgb * att * sf_spot * brdf_sheen(n_geom, v_dir, to_light, sheen_tint, sheen_rough);
        }
    }
    // HemisphereLight is also indirect-diffuse and metals don't diffuse it.
    for (var i : u32 = 0u; i < frame.light_counts.w; i = i + 1u) {
        if (i >= 4u) { break; }
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
    if (kind == MAT_PHYSICAL) {
        cc = clamp(mesh.params3.x, 0.0, 1.0);
        cc_roughness = clamp(mesh.params3.y, 0.0, 1.0);
        // three.js clearcoat: α = roughness² (same remap as the base GGX).
        let cc_a = max(cc_roughness * cc_roughness, 0.0016);
        let n_dot_v = max(dot(n_geom, v_dir), 0.0);
        let fc0 = vec3<f32>(0.04);
        cc_fresnel = f_schlick(n_dot_v, fc0);

        for (var i : u32 = 0u; i < frame.light_counts.x; i = i + 1u) {
            if (i >= 4u) { break; }
            let l = frame.dir_lights[i];
            let to_light = -normalize(l.direction.xyz);
            let att = select(1.0, sf, i == 0u);
            cc_lit = cc_lit + l.color.rgb * att
                * clearcoat_spec(n_geom, v_dir, to_light, cc_a, fc0);
        }
        // Point and spot lights were previously ignored by the clearcoat layer,
        // so a lacquered surface lit only by a lamp had no coat highlight.
        for (var i : u32 = 0u; i < frame.light_counts.y; i = i + 1u) {
            if (i >= 4u) { break; }
            let l = frame.point_lights[i];
            let to_light_vec = l.position.xyz - in.world_pos;
            let d = length(to_light_vec);
            let to_light = to_light_vec / max(d, 0.0001);
            let att = punctual_attenuation(d, l.params.x, l.params.y);
            let sf_pt = select(1.0, select(1.0, shadow_factor_point(in.world_pos), receive_shadow), i == 0u);
            cc_lit = cc_lit + l.color.rgb * att * sf_pt
                * clearcoat_spec(n_geom, v_dir, to_light, cc_a, fc0);
        }
        for (var i : u32 = 0u; i < frame.light_counts.z; i = i + 1u) {
            if (i >= 4u) { break; }
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
            let sf_spot = select(1.0, select(1.0, shadow_factor_spot(in.world_pos), receive_shadow), i == 0u);
            cc_lit = cc_lit + l.color.rgb * att * sf_spot
                * clearcoat_spec(n_geom, v_dir, to_light, cc_a, fc0);
        }
    }

    // Rectangular area lights. Previously a no-op stub, so a scene lit only by
    // a RectAreaLight rendered black.
    for (var i : u32 = 0u; i < frame.light_counts2.x; i = i + 1u) {
        if (i >= 2u) { break; }
        let rl = frame.rect_lights[i];
        let center = rl.position.xyz;
        let right = rl.right.xyz;
        let up = rl.up.xyz;

        // Single-sided, like three.js: a RectAreaLight emits along its local
        // -Z, and cross(right, up) is local +Z — so the emitting direction is
        // the *negated* cross product.
        let light_n = -normalize(cross(right, up));
        let to_surface = in.world_pos - center;
        if (dot(to_surface, light_n) <= 0.0) { continue; }

        // Winding matters: the edge cross-products below must accumulate a
        // vector pointing *toward* the lit side, or every contribution comes
        // out negative and clamps to zero. This order is the one that does.
        let c0 = center - right - up;
        let c1 = center + right - up;
        let c2 = center + right + up;
        let c3 = center - right + up;

        // Diffuse: exact form factor over the quad.
        let ff = rect_diffuse_factor(n_geom, in.world_pos, c0, c1, c2, c3);
        lit = lit + diffuse_color * rl.color.rgb * ff * ao;

        // Specular: representative point, with the lobe widened by the light's
        // angular size so a big panel gives a big soft highlight.
        let mrp = rect_representative_point(in.world_pos, reflect(-v_dir, n_geom), center, right, up);
        let to_mrp = mrp - in.world_pos;
        let dist = length(to_mrp);
        if (dist > 1e-4) {
            let l_dir = to_mrp / dist;
            let alpha = max(roughness * roughness, 0.0016);
            let radius = max(length(right), length(up));
            let norm = rect_spec_normalisation(alpha, radius, dist);
            // Reuse the main BRDF so anisotropy and iridescence apply here too;
            // the form factor already carries the light's solid angle, so weight
            // the specular by it rather than by a 1/d² falloff.
            let brdf = pbr_direct(
                n_geom, v_dir, l_dir, tangent, bitangent,
                albedo_lin, roughness, metalness, aniso, spec_f0,
            );
            let n_dot_l = max(dot(n_geom, l_dir), 0.0);
            // Strip the diffuse half — it is already accounted for by `ff`.
            let spec_only = max(brdf - diffuse_color * RECIP_PI * n_dot_l, vec3<f32>(0.0));
            lit = lit + rl.color.rgb * spec_only * ff * norm * PI;

            if (sheen_strength > 0.0) {
                sheen_lit = sheen_lit + rl.color.rgb
                    * brdf_sheen(n_geom, v_dir, l_dir, sheen_tint, sheen_rough) * ff * PI;
            }
            if (kind == MAT_PHYSICAL && cc > 0.0) {
                let cc_a = max(cc_roughness * cc_roughness, 0.0016);
                cc_lit = cc_lit + rl.color.rgb
                    * clearcoat_spec(n_geom, v_dir, l_dir, cc_a, vec3<f32>(0.04))
                    * ff * rect_spec_normalisation(cc_a, radius, dist) * PI;
            }
        }
    }

    // IBL — mirrors three.js MeshStandardMaterial envmap_physical_pars_fragment +
    // RE_IndirectSpecular_Physical (getIBLRadiance + computeMultiscattering).
    // tone_mapping_exposure.z is the env-enabled flag (1 = on, 0 = off).
    if (frame.tone_mapping_exposure.z > 0.5) {
        // spec_f0 rather than a locally-derived F0, so an iridescent film tints
        // the environment reflection too — otherwise thin-film only showed up
        // under direct lights and vanished on a purely IBL-lit object.
        let specular_color = spec_f0;
        let specular_f90 = 1.0;

        // Anisotropy bends the reflection toward the low-roughness axis, so a
        // brushed surface smears the environment along the streak instead of
        // reflecting it like a mirror (three.js getIBLAnisotropyRadiance).
        var refl_normal = n_geom;
        if (abs(aniso) > 0.001) {
            let aniso_t = cross(bitangent, v_dir);
            let aniso_n = cross(aniso_t, bitangent);
            var bend = 1.0 - abs(aniso) * (1.0 - roughness_env);
            bend = bend * bend;
            bend = bend * bend;
            if (dot(aniso_n, aniso_n) > 1e-8) {
                refl_normal = normalize(mix(normalize(aniso_n), n_geom, bend));
            }
        }
        var reflect_vec = reflect(-v_dir, refl_normal);
        reflect_vec = normalize(mix(reflect_vec, n_geom, roughness_env * roughness_env));
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

        lit = lit + radiance * single_scatter + multi_scatter * cosine_weighted_irr;
        // The environment's diffuse term, added exactly ONCE.
        //
        // In three.js this line lives inside RE_IndirectSpecular_Physical, and
        // the separate RE_IndirectDiffuse call is fed `irradiance` from ambient
        // lights and light probes — *not* from the env map. threers already adds
        // the ambient term further up, so mirroring both three.js call sites
        // here counted the environment twice: a white Lambertian in a uniform
        // environment measured exactly 2.0× its surroundings, which is
        // impossible (it cannot reflect more light than reaches it).
        lit = lit + diffuse_color * (1.0 - scatter_max) * cosine_weighted_irr;
    }

    // Clearcoat composite, deferred to here so the coat's Fresnel attenuates
    // the base's environment reflection too — matching three.js's
    // `outgoingLight * (1 - clearcoat*Fcc) + clearcoatSpecular * clearcoat`.
    if (kind == MAT_PHYSICAL && cc > 0.0) {
        // The coat is a smooth dielectric shell, so it reflects the environment
        // in its own right. Without this a cover glass or car lacquer only
        // showed a highlight where a light happened to point at it.
        if (frame.tone_mapping_exposure.z > 0.5) {
            let cc_refl = reflect(-v_dir, n_geom);
            let cc_radiance = sample_env_cube(cc_refl, cc_roughness);
            let cc_fab = dfg_approx(n_geom, v_dir, cc_roughness);
            cc_lit = cc_lit + cc_radiance * (vec3<f32>(0.04) * cc_fab.x + cc_fab.y);
        }
        lit = lit * (vec3<f32>(1.0) - cc_fresnel * cc) + cc_lit * cc;
    }

    // Sheen composite. three.js darkens the base by an energy-compensation
    // factor before adding the sheen lobe, so a strong sheen doesn't push the
    // surface over 1.0. Applied after IBL (rather than before clearcoat, where
    // three.js puts it) so the compensation covers the *whole* accumulated
    // result — threers runs IBL after clearcoat, unlike three.js.
    if (sheen_strength > 0.0) {
        // Environment term. The Charlie lobe's directional albedo is not the
        // GGX one, so `dfg_approx` is wrong here; use Estevez & Kulla's fitted
        // E(NoV, roughness) instead. Without this a sheen surface lit only by
        // an environment stayed flat — the retroreflective rim, which is the
        // entire point of sheen, only appeared under punctual lights.
        if (frame.tone_mapping_exposure.z > 0.5) {
            let sheen_irr = sample_env_cube(n_geom, sheen_rough);
            let n_dot_v = max(dot(n_geom, v_dir), 0.0);
            sheen_lit = sheen_lit + sheen_irr * sheen_tint * sheen_env_albedo(n_dot_v, sheen_rough);
        }
        let sheen_max = max(max(sheen_tint.r, sheen_tint.g), sheen_tint.b);
        let sheen_energy_comp = 1.0 - 0.157 * sheen_max;
        lit = lit * sheen_energy_comp + sheen_lit;
    }

    // Transmission (glass): MeshPhysicalMaterial.transmission makes the surface
    // see-through, Fresnel-weighted by the material IOR — opaque at the grazing
    // rim, transparent face-on.
    var out_opacity = opacity;
    if (kind == MAT_PHYSICAL && mesh.params3.w > 0.0) {
        let transmission = clamp(mesh.params3.w, 0.0, 1.0);
        let t_ior = max(mesh.params3.z, 1.0);
        let thickness = max(mesh.params4.x, 0.0);
        let f0s = pow((t_ior - 1.0) / (t_ior + 1.0), 2.0);
        let ndv = max(dot(n_geom, v_dir), 0.0);
        let fresnel = f0s + (1.0 - f0s) * pow(1.0 - ndv, 5.0);

        if (frame.tone_mapping_exposure.z > 0.5) {
            // Volumetric refraction (three.js getIBLVolumeRefraction fallback):
            // with an environment but no transmission framebuffer, bend the view
            // ray through the surface by the IOR and read the environment along
            // it, so the glass shows a refracted, thickness-attenuated image of
            // the surroundings rather than just fading out. This is real IOR
            // refraction of the IBL environment, composited opaquely in-shader
            // (no back-to-front sorting → flicker-free).
            let refr = refract(-v_dir, n_geom, 1.0 / t_ior);
            // refract() returns 0 on total internal reflection; guard it.
            var refr_dir = n_geom;
            if (dot(refr, refr) > 1e-6) { refr_dir = normalize(refr); }
            var transmitted = sample_env_cube(refr_dir, roughness_env);
            transmitted = transmitted * volume_transmittance(albedo_lin, thickness);
            // Mix the surface shading toward the refracted image by the
            // transmission factor; the Fresnel rim keeps its reflective
            // highlight (already added by the IBL block above).
            lit = mix(lit, transmitted, transmission * (1.0 - fresnel));
            // Glassy edge: the signature "curved shell" read comes from the
            // grazing rim lighting up with reflected environment. Brighten the
            // silhouette with the mirror-reflected env plus a thin white sheen,
            // scaled by a steep Fresnel so only the very edge glows.
            let rim = pow(1.0 - ndv, 4.0);
            let rim_env = sample_env_cube(reflect(-v_dir, n_geom), roughness_env);
            lit = lit + rim * (rim_env * 0.7 + vec3<f32>(0.06)) * transmission;
            // Keep a faint body so the silhouette and heatmap stay legible even
            // face-on; only the grazing Fresnel rim is fully opaque.
            out_opacity = opacity * clamp(fresnel + rim * 0.6 + (1.0 - transmission) + 0.15, 0.0, 1.0);
        } else {
            // No environment: screen-composite approximation (Fresnel opacity),
            // drawn over the already-rendered scene via alpha blending.
            out_opacity = opacity * (1.0 - transmission * (1.0 - fresnel));
        }
    }

    lit = lit + emissive;
    // General per-vertex emission (Standard/Physical): app bakes color into
    // vertexColor.rgb + a gate into vertexColor.a, scaled by vertex_emissive
    // (params4.z). Zero by default, so no effect unless the material opts in.
    lit = lit + in.vertex_color.rgb * in.vertex_color.a * mesh.params4.z;
    let tm = u32(frame.tone_mapping_exposure.x);
    let exposure = frame.tone_mapping_exposure.y;
    if (tm == 1u) { lit = lit * exposure; }
    else if (tm == 2u) { lit = aces_tonemap(lit * exposure); }
    lit = apply_fog(lit, in.view_z);
    // The canvas surface is bgra8unorm (not sRGB), so we MUST do the
    // linear→sRGB encoding ourselves; otherwise our linearly computed PBR
    // output writes raw linear bytes and the browser misinterprets them as
    // sRGB, making low-luminance scenes (e.g. spot lights) ~13× too dark.
    let lit_srgb = framebuffer_encode(lit);
    return vec4<f32>(lit_srgb, out_opacity);
}

// Weighted-blended order-independent transparency output. Instead of a single
// blended color, each fragment contributes to two accumulation buffers:
//   accum   (Rgba16F, additive)      = sum(premultiplied_color * weight)
//   reveal  (R16F, multiplicative)   = product(1 - alpha)
// The resolve pass turns these into an order-independent composite. Uses simple
// ambient+hemisphere+directional Lambert lighting on the vertex color (enough
// for a translucent heatmap surface) so it stays self-contained.
struct OitOut {
    @location(0) accum  : vec4<f32>,
    @location(1) reveal : f32,
}
@fragment
fn fs_oit(in : VsOut) -> OitOut {
    let base = mesh.color.rgb * in.vertex_color.rgb;
    let alpha = clamp(mesh.params.y * in.vertex_color.a, 0.0, 1.0);
    let n = normalize(in.world_normal);
    var light = frame.ambient.rgb;
    for (var i : u32 = 0u; i < frame.light_counts.x; i = i + 1u) {
        if (i >= 4u) { break; }
        let l = frame.dir_lights[i];
        light = light + l.color.rgb * max(dot(n, -normalize(l.direction.xyz)), 0.0);
    }
    for (var i : u32 = 0u; i < frame.light_counts.w; i = i + 1u) {
        if (i >= 4u) { break; }
        light = light + shade_hemi(n, frame.hemi_lights[i]);
    }
    let col = framebuffer_encode(base * light);

    // McGuire depth weight: nearer fragments weigh more. `view_z` is positive.
    let z = max(in.view_z, 1e-3);
    let w = alpha * clamp(10.0 / (1e-5 + pow(z / 60.0, 3.0) + pow(z / 300.0, 6.0)), 1e-2, 3e3);

    var o : OitOut;
    o.accum  = vec4<f32>(col * alpha * w, alpha * w);
    o.reveal = alpha;
    return o;
}

// --- Screen-space refraction glass (TransparencyMode::Refract) --------------
// Project a world point to screen UV (framebuffer top-left origin).
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
        let ray_d = clip.z / clip.w;
        let coord = vec2<i32>(uv * dims);
        let scene_d = textureLoad(ss_depth_tex, coord, 0);
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
        let ray_d = clip.z / clip.w;
        let scene_d = textureLoad(ss_depth_tex, vec2<i32>(uv * dims), 0);
        // Require a *thin* crossing (ray just behind the surface) to reject rays
        // that plunge far behind geometry — the classic SSR thickness test.
        if (scene_d < 0.9999 && ray_d > scene_d + 2e-5 && ray_d < scene_d + 0.02) {
            var lo = prev_t;
            var hi = t;
            for (var k = 0; k < 6; k = k + 1) {
                let mt = (lo + hi) * 0.5;
                let mc = frame.view_proj * vec4<f32>(origin + dir * (dist * mt), 1.0);
                let muv = mc.xy / mc.w * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
                let msd = textureLoad(ss_depth_tex, vec2<i32>(muv * dims), 0);
                if (mc.z / mc.w > msd + 2e-5) { hi = mt; } else { lo = mt; }
            }
            let hc = frame.view_proj * vec4<f32>(origin + dir * (dist * (lo + hi) * 0.5), 1.0);
            return vec3<f32>(hc.xy / hc.w * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5), 1.0);
        }
        prev_t = t;
        i = i + 1;
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Dielectric glass, closer to a Cycles/Principled Glass BSDF: at the surface the
// view ray either refracts or reflects, weighted by Fresnel; both terms trace
// the captured scene in screen space (roughness → mip blur), with the
// environment probe as the off-screen fallback. There is no diffuse/opaque body
// — the glass "color" only tints transmission (absorption). `thickness`
// (params4.x) is the world-space march distance.
@fragment
fn fs_ss_glass(in : VsOut) -> @location(0) vec4<f32> {
    let dims = frame.viewport_size.xy;
    let screen_uv = in.clip_pos.xy / dims;

    var n = normalize(in.world_normal);
    let v = normalize(frame.camera_position.xyz - in.world_pos);
    if (dot(n, v) < 0.0) { n = -n; }              // orient toward the camera
    let ndv = clamp(dot(n, v), 1e-3, 1.0);

    let tint       = mesh.color.rgb * in.vertex_color.rgb;   // glass color
    let roughness  = clamp(mesh.params.z, 0.02, 1.0);
    let ior        = max(mesh.params3.z, 1.0);
    let march_dist = max(mesh.params4.x, 1.0);               // thickness → march
    let dispersion = mesh.params4.y;
    let lod        = roughness * 6.0;                        // roughness → mip blur

    // --- Two-surface thickness: eye-space distance from this front glass surface
    // to its back face (from the captured back-face depth), i.e. the path length
    // through the glass volume. Drives path-length absorption + a volumetric
    // density tint so thick regions read denser than thin edges. ---
    let near = frame.viewport_size.z;
    let far  = frame.viewport_size.w;
    let bcoord = vec2<i32>(screen_uv * dims);
    let back_d = textureLoad(ss_back_depth_tex, bcoord, 0);
    var thick = 0.0;
    if (back_d < 0.9999) {
        let back_dist = near * far / max(far - back_d * (far - near), 1e-4);
        thick = max(back_dist - in.view_z, 0.0);
    }

    // --- Refraction: march the depth buffer for the real exit point --------
    // The cortex normal is extremely high-frequency, so a sharp screen-space
    // refraction speckles (neighbouring fragments hit scattered pixels). Blur
    // the refracted tap (mip floor) so it averages into smooth frosted glass.
    let refr_lod = max(lod, 2.0);
    let refr_dir = refract(-v, n, 1.0 / ior);
    let tir = dot(refr_dir, refr_dir) < 1e-6;   // total internal reflection
    // Sample the captured scene along the refracted ray. On a "miss" the march
    // endpoint projects the ray to the background, so this reads the actual
    // background (black) through the glass — transparent, not an env-filled body.
    // The cortex normal is very high-frequency, so clamp the screen displacement
    // (EEVEE-style bounded trace) to keep the refraction coherent, not speckled.
    let m = ss_march(in.world_pos, normalize(select(refr_dir, reflect(-v, n), tir)), march_dist, 48);
    let max_off = 0.025;
    var roff = m.xy - screen_uv;
    let rlen = length(roff);
    if (rlen > max_off) { roff = roff * (max_off / rlen); }
    let ruv = screen_uv + roff;
    let disp = roff * dispersion;                 // dispersion: fan RGB taps
    var refracted = vec3<f32>(
        textureSampleLevel(ss_color_tex, ss_color_samp, ruv + disp, refr_lod).r,
        textureSampleLevel(ss_color_tex, ss_color_samp, ruv,        refr_lod).g,
        textureSampleLevel(ss_color_tex, ss_color_samp, ruv - disp, refr_lod).b,
    );
    // Glass color as transmission absorption over the real path length
    // (Beer–Lambert): thicker glass tints/darkens the transmitted image more.
    // With attenuation_distance set, `thick` is a genuine world-space path
    // length so the physical coefficient applies directly; without it, keep the
    // empirically-scaled fallback that existing scenes were authored against.
    if (mesh.params7.w > 0.0) {
        refracted = refracted * volume_transmittance(tint, thick);
    } else {
        refracted = refracted * exp(-(vec3<f32>(1.0) - tint) * (0.15 + thick * 0.006));
    }

    // --- Reflection: accurate SSR of the scene, black where it misses --------
    // The visible background is black, so a transparent glass reflects black on
    // its faces (not the studio env — that would gray the whole bumpy surface).
    // A binary-refined march resolves the reflected geometry at its true screen
    // position (roughness selects the mip blur); env is only a grazing-rim sheen.
    let refl_dir = reflect(-v, n);
    var reflection = vec3<f32>(0.0);
    let rm = ss_march_refl(in.world_pos, refl_dir, march_dist, 40);
    if (rm.z > 0.5) {
        reflection = textureSampleLevel(ss_color_tex, ss_color_samp, rm.xy, max(lod, 1.0)).rgb;
    }

    // --- Pure dielectric Fresnel reflect/refract mix -----------------------
    let f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
    var fresnel = f0 + (1.0 - f0) * pow(1.0 - ndv, 5.0);
    if (tir) { fresnel = 1.0; }
    var color = mix(refracted, reflection, fresnel);
    // Two-surface volumetric density: thicker glass reads as a denser body,
    // picking up a faint tint of the glass color toward the middle (a real
    // depth cue on a black background where absorption alone is invisible).
    let density = 1.0 - exp(-thick * 0.010);
    color = color + tint * density * 0.30;
    // Environment sheen only at the grazing silhouette — keeps the body black
    // (transparent) while the rim still catches a glassy studio highlight.
    let env_refl = sample_env_cube(refl_dir, roughness);
    let rim = pow(1.0 - ndv, 5.0);
    color = color + rim * env_refl * 0.9;

    // --- Caustics (approximation) -----------------------------------------
    let caustic = pow(max(dot(refracted, vec3<f32>(0.333)) - 0.55, 0.0), 2.0) * (1.0 - fresnel);
    color = color + caustic * vec3<f32>(1.0, 0.97, 0.9) * 1.2;

    // --- Emission ---------------------------------------------------------
    // Material emissive plus the general per-vertex emission term: the app bakes
    // a color into vertexColor.rgb and a gate into vertexColor.a, scaled by the
    // material's vertex_emissive (params4.z). Zero by default (no effect).
    color = color + mesh.emissive.rgb + in.vertex_color.rgb * in.vertex_color.a * mesh.params4.z;

    color = apply_fog(color, in.view_z);
    let tm = u32(frame.tone_mapping_exposure.x);
    let exposure = frame.tone_mapping_exposure.y;
    if (tm == 1u) { color = color * exposure; }
    else if (tm == 2u) { color = aces_tonemap(color * exposure); }
    return vec4<f32>(framebuffer_encode(color), 1.0);
}
"#;

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
    cam_delta      : vec4<f32>,
};
@group(0) @binding(0) var cur_tex   : texture_2d<f32>;
@group(0) @binding(1) var hist_tex  : texture_2d<f32>;
@group(0) @binding(2) var taa_depth : texture_depth_2d;
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
    if (taa.params.w > 0.5) { return cur; }          // first frame: seed history

    let d = textureLoad(taa_depth, coord, 0);
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

@fragment
fn fs_main(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(i32(pos.x), i32(pos.y));
    let reveal = textureLoad(reveal_tex, p, 0).r;
    if (reveal > 0.9999) { discard; }
    let accum = textureLoad(accum_tex, p, 0);
    let avg = accum.rgb / max(accum.a, 1e-5);
    return vec4<f32>(avg, 1.0 - reveal);
}
"#;
