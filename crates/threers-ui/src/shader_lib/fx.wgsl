// Fill effects (threers_ui::shader_lib::FX; needs NOISE and COLOR), as the
// indicators draw them. Each takes a point `along` pixels along the fill
// and `up` pixels off its middle (up a bar, out from a ring), in a stroke
// `half` thick; `env` how far within the fill's run (0 past its ends),
// `cover` how much of the pixel the fill covers, `u` where along the whole
// track (0 to 1, for colour), the time, and three colours. Premultiplied:
// the light added, and how opaque.

// Fire: heat falling off with height, torn by rising noise, embers.
fn ui_fire(along: f32, up: f32, half: f32, env: f32, cover: f32, u: f32, now: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>) -> vec4<f32> {
    let hh = max(half, 0.5);
    // Above the stroke only where the fill runs.
    let h = (up + hh) / (hh * 5.0);
    let n = ui_fbm(vec2<f32>(along / (hh * 1.4), up / (hh * 1.1) - now * 3.4));
    let n2 = ui_fbm(vec2<f32>(along / (hh * 0.6) + 7.0, up / (hh * 0.5) - now * 5.5));
    // Nothing below the stroke: flames only rise (out from a ring).
    let above = smoothstep(-hh * 1.15, -hh * 0.7, up);
    var heat = ((1.0 - h) * 1.2 + (n - 0.5) * 1.5 + (n2 - 0.5) * 0.5) * env * above;
    heat = max(heat, cover * (0.7 + 0.3 * n));
    let t = clamp(heat, 0.0, 1.0);
    let a = smoothstep(0.18, 0.5, t);
    var col = ui_ramp3(c0, c1, c2, t);
    col = mix(col, vec3<f32>(1.0, 0.96, 0.85), smoothstep(0.86, 1.0, t));
    var light = col * (0.55 + 1.3 * t) * a;
    // Embers, drifting up and out of the flames.
    let g = vec2<f32>(along / (hh * 1.1), (up - now * hh * 2.6) / (hh * 1.1));
    let cell = floor(g);
    let r = ui_hash(cell);
    let at = vec2<f32>(ui_hash(cell + 3.1), ui_hash(cell + 7.3));
    let d = length(fract(g) - at);
    let ember = select(0.0, exp(-d * d * 220.0) * (1.0 - clamp(h, 0.0, 1.0)) * env * above, r > 0.86);
    light = light + c2.rgb * ember * 1.6;
    return vec4<f32>(light, max(a, ember * 0.8));
}

// Water: deeper toward the bottom, caustics drifting, bubbles rising.
fn ui_water(along: f32, up: f32, half: f32, env: f32, cover: f32, u: f32, now: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>) -> vec4<f32> {
    let hh = max(half, 0.5);

    // Water: deeper toward the bottom, caustics drifting, bubbles.
    let depth = clamp(0.5 - up / (hh * 2.0), 0.0, 1.0);
    let base = ui_ramp3(c0, c1, c2, u) * (1.0 - depth * 0.45);
    let c = ui_fbm(vec2<f32>(along / (hh * 2.6) + now * 0.4, up / (hh * 1.8) - now * 0.25));
    let caustic = pow(1.0 - abs(sin(c * 7.0 + now * 1.1)), 14.0) * 0.4;
    let g = vec2<f32>(along / (hh * 1.3), (up - now * hh * 1.8) / (hh * 1.3));
    let cell = floor(g);
    let at = vec2<f32>(0.3 + 0.4 * ui_hash(cell + 1.7), 0.3 + 0.4 * ui_hash(cell + 5.9));
    let rad = 0.1 + 0.12 * ui_hash(cell + 2.2);
    let d = length(fract(g) - at);
    let bubble = select(0.0, smoothstep(0.03, 0.0, abs(d - rad)) * 0.8 + smoothstep(rad, 0.0, d) * 0.12, ui_hash(cell) > 0.82);
    let light = base + vec3<f32>(caustic + bubble) + c2.rgb * caustic * 0.4;
    return vec4<f32>(light * cover, cover);
}

// Plasma: bands of the colours swirling.
fn ui_plasma(along: f32, up: f32, half: f32, env: f32, cover: f32, u: f32, now: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>) -> vec4<f32> {
    let hh = max(half, 0.5);

    // Plasma: bands of the palette swirling.
    let w = ui_fbm(vec2<f32>(along / (hh * 2.4) + sin(now * 0.7) * 0.6, up / (hh * 1.2) + now * 0.5));
    let w2 = ui_fbm(vec2<f32>(along / (hh * 1.1) - now * 0.8, up / hh));
    let t = fract(w * 1.8 + w2 * 0.6 + u * 0.4 - now * 0.12);
    let col = ui_ramp3(c0, c1, c2, abs(t * 2.0 - 1.0));
    let pulse = 0.85 + 0.35 * sin(w * 24.0 + now * 3.0);
    let core = pow(1.0 - clamp(abs(up) / hh, 0.0, 1.0), 2.0) * 0.35;
    return vec4<f32>((col * pulse + vec3<f32>(core)) * cover, cover);
}

// Frost: ice in cells, cracked where they meet, sparkling.
fn ui_frost(along: f32, up: f32, half: f32, env: f32, cover: f32, u: f32, now: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>) -> vec4<f32> {
    let hh = max(half, 0.5);
    let v = ui_voronoi(vec2<f32>(along, up) / (hh * 0.85));
    let d1 = v.x;
    let d2 = v.y;
    let nearest = v.zw;
    let crack = 1.0 - smoothstep(0.02, 0.09, d2 - d1);
    let shade = 0.75 + 0.35 * ui_hash(nearest);
    let ice = mix(ui_ramp3(c0, c1, c2, u), vec3<f32>(0.92, 0.97, 1.0), 0.35) * shade;
    let twinkle = pow(max(sin(now * 3.0 + ui_hash(nearest) * 40.0), 0.0), 24.0) * smoothstep(0.25, 0.0, d1);
    let light = ice + vec3<f32>(crack * 0.55 + twinkle);
    return vec4<f32>(light * cover, cover);
}
