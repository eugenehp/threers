// The studio solid things reflect, in the view (x right, y up, z toward
// the eye): a dark floor, a sky growing lighter upward, a softbox
// overhead, a tall one to the left and a cool strip light to the right.
// Shared by the solid boxes and the solid indicators.
fn studio_env(r: vec3<f32>) -> vec3<f32> {
    let sky = mix(vec3<f32>(0.04, 0.05, 0.07), vec3<f32>(0.2, 0.23, 0.3), clamp(r.y, 0.0, 1.0));
    var c = select(vec3<f32>(0.012, 0.013, 0.016), sky, r.y > -0.05);
    c = c + vec3<f32>(2.4) * smoothstep(0.8, 0.9, r.y) * smoothstep(0.6, 0.35, abs(r.x));
    let left = (r.x + 0.72) / 0.11;
    c = c + vec3<f32>(1.6) * exp(-left * left) * smoothstep(-0.25, 0.25, r.y);
    let right = (r.x - 0.82) / 0.07;
    c = c + vec3<f32>(0.45, 0.62, 1.0) * 0.9 * exp(-right * right) * smoothstep(-0.4, 0.2, r.y);
    return c;
}
