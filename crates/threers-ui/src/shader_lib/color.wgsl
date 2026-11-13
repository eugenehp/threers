// Colour for effects (threers_ui::shader_lib::COLOR).

// Through three colours as `t` goes 0 to 1 (the middle one at a half).
fn ui_ramp3(c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, t: f32) -> vec3<f32> {
    let u = clamp(t, 0.0, 1.0);
    let a = mix(c0.rgb, c1.rgb, clamp(u * 2.0, 0.0, 1.0));
    let b = mix(c1.rgb, c2.rgb, clamp(u * 2.0 - 1.0, 0.0, 1.0));
    return select(b, a, u < 0.5);
}

// sRGB-encoded to linear light, and back.
fn ui_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

fn ui_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let k = max(c, vec3<f32>(0.0));
    return select(1.055 * pow(k, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055), k * 12.92, k <= vec3<f32>(0.0031308));
}

// A colour from its hue (degrees), saturation and lightness (0 to 1).
fn ui_hsl(h: f32, s: f32, l: f32) -> vec3<f32> {
    let k = (vec3<f32>(0.0, 8.0, 4.0) + vec3<f32>(h / 30.0)) % vec3<f32>(12.0);
    let a = s * min(l, 1.0 - l);
    return vec3<f32>(l) - a * max(min(min(k - vec3<f32>(3.0), vec3<f32>(9.0) - k), vec3<f32>(1.0)), vec3<f32>(-1.0));
}

// Black body at `kelvin`, roughly (1000 to 12000): embers, flames, stars.
fn ui_blackbody(kelvin: f32) -> vec3<f32> {
    let t = clamp(kelvin, 1000.0, 12000.0) / 100.0;
    let r = select(1.293 * pow(t - 60.0, -0.1332), 1.0, t <= 66.0);
    let g = select(1.13 * pow(t - 60.0, -0.0755), 0.3901 * log(t) - 0.6318, t <= 66.0);
    let b = select(select(0.5432 * log(t - 10.0) - 1.1962, 0.0, t <= 19.0), 1.0, t >= 66.0);
    return ui_to_linear(clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0)));
}
