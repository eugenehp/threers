// Noise for effects (threers_ui::shader_lib::NOISE).

// A pseudo-random number 0 to 1 for each point (each cell, given whole
// numbers).
fn ui_hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

// Smooth value noise, 0 to 1, one feature a unit.
fn ui_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (vec2<f32>(3.0) - 2.0 * f);
    let a = mix(ui_hash(i), ui_hash(i + vec2<f32>(1.0, 0.0)), u.x);
    let b = mix(ui_hash(i + vec2<f32>(0.0, 1.0)), ui_hash(i + vec2<f32>(1.0, 1.0)), u.x);
    return mix(a, b, u.y);
}

// Three octaves of value noise, 0 to 1: clouds, flames, turbulence.
fn ui_fbm(p: vec2<f32>) -> f32 {
    return (ui_noise(p) * 0.5 + ui_noise(p * 2.03 + vec2<f32>(3.1, 7.7)) * 0.3 + ui_noise(p * 4.1 + vec2<f32>(9.2, 1.4)) * 0.2);
}

// Cells round random points a unit apart (Worley noise): the distances to
// the nearest point and the next nearest, and the nearest cell's corner.
// `d2 - d1` is small along the borders: cracks, scales, foam.
fn ui_voronoi(p: vec2<f32>) -> vec4<f32> {
    let cell = floor(p);
    var d1 = 10.0;
    var d2 = 10.0;
    var nearest = cell;
    for (var j = -1; j <= 1; j = j + 1) {
        for (var i = -1; i <= 1; i = i + 1) {
            let c = cell + vec2<f32>(f32(i), f32(j));
            let at = c + vec2<f32>(ui_hash(c + 0.5), ui_hash(c + 4.1));
            let d = length(p - at);
            if (d < d1) {
                d2 = d1;
                d1 = d;
                nearest = c;
            } else if (d < d2) {
                d2 = d;
            }
        }
    }
    return vec4<f32>(d1, d2, nearest);
}
