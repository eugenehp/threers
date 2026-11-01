// RGB Halftone helpers — port of three.js HalftoneShader (r165).

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
            dist = dist * (1.0 - 0.20710678) + dot_p * dist * 0.41421356;
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
    let base = ht_rand(vec2<f32>(floor(point.x), floor(point.y))) * 6.28318531;
    let step_a = 6.28318531 / f32(8);
    let dist = u.params2.x * 0.66;
    for (var i: i32 = 0; i < 8; i = i + 1) {
        let r = base + step_a * f32(i);
        let coord = point + vec2<f32>(cos(r) * dist, sin(r) * dist);
        tex = tex + textureSampleLevel(input_tex, input_sampler, coord / res, 0.0);
    }
    tex = tex / f32(8 + 1);
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
        let off_angle = ht_rand(vec2<f32>(floor(cell.p1.x), floor(cell.p1.y))) * 6.28318531;
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
