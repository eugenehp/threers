fn pfx_hash1(x: f32) -> f32 {
    return fract(sin(x * 127.1 + 311.7) * 43758.5453);
}

fn pfx_noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (vec2<f32>(3.0) - 2.0 * f);
    let n = i.x + i.y * 57.0;
    let a = mix(pfx_hash1(n), pfx_hash1(n + 1.0), u.x);
    let b = mix(pfx_hash1(n + 57.0), pfx_hash1(n + 58.0), u.x);
    return mix(a, b, u.y) * 2.0 - 1.0;
}

fn pfx_fbm(p: vec2<f32>) -> f32 {
    return pfx_noise2(p) * 0.6 + pfx_noise2(p * 2.3 + vec2<f32>(5.2, 1.3)) * 0.28 + pfx_noise2(p * 5.1 + vec2<f32>(1.7, 9.2)) * 0.12;
}

// The bolts' light at `along` pixels along and `across` pixels off the
// middle of a stroke `half` pixels thick.
fn pfx_bolts(along: f32, across: f32, half: f32, now: f32, col: vec3<f32>) -> vec3<f32> {
    var light = vec3<f32>(0.0);
    for (var i = 0; i < 3; i = i + 1) {
        let fi = f32(i);
        let seed = fi * 17.31;
        let strike = floor(now * 13.0 + fi * 0.37);
        let jitter = pfx_hash1(strike + seed);
        let off = pfx_fbm(vec2<f32>(along / max(half * 2.6, 1.0) + seed + strike * 3.7, now * 1.5 + seed)) * half * (0.9 + 0.7 * jitter);
        let d = abs(across - off);
        let w = max(half * 0.1, 0.55);
        let core = exp(-d * d / (w * w));
        let glow = exp(-d / (w * 6.0));
        let flicker = 0.45 + 0.55 * pfx_hash1(strike * 1.7 + seed + 3.0);
        let hot = select(vec3<f32>(core * 0.9), col * core * 1.6, pfx_on_light);
        light = light + (hot + col * (core * 0.8 + glow * 0.55)) * flicker;
    }
    return light;
}

// Sparks round the head, `d` pixels from it, at angle `ang`.
fn pfx_sparks(d: f32, ang: f32, half: f32, now: f32, col: vec3<f32>) -> vec3<f32> {
    let rays = pow(max(sin(ang * 7.0 + pfx_hash1(floor(now * 20.0)) * 6.28), 0.0), 6.0);
    let fall = exp(-d / max(half * 1.2, 1.0));
    return (vec3<f32>(0.7) + col) * fall * (0.4 + 0.8 * rays);
}

// Light added over a body (premultiplied), toned to fit.
// On a light background, ink of the light's colour instead.
fn pfx_add(body: vec4<f32>, light: vec3<f32>) -> vec4<f32> {
    let l = vec3<f32>(1.0) - exp(-max(light, vec3<f32>(0.0)));
    let la = clamp(max(l.r, max(l.g, l.b)), 0.0, 1.0);
    if (pfx_on_light) {
        let ink = l / max(la, 0.0001) * 0.75;
        return vec4<f32>(body.rgb * (1.0 - la) + ink * la, clamp(body.a + la * (1.0 - body.a), 0.0, 1.0));
    }
    return vec4<f32>(body.rgb * (1.0 - la * 0.5) + l, clamp(body.a + la * 0.7 * (1.0 - body.a), 0.0, 1.0));
}

fn pfx_bar_electric(p: vec2<f32>, size: vec2<f32>, now: f32, value: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, look: vec4<f32>, reach: f32) -> vec4<f32> {
    let aa = max(fwidth(p.x), 0.001);
    let r = size.y * 0.5;
    let a = vec2<f32>(r, r);
    let b = vec2<f32>(max(size.x - r, r), r);
    let len = max(b.x - a.x, 0.001);
    let span = pfx_span(now, value, look.z);
    let has = span.y > span.x + 0.0005;
    let fa = a + (b - a) * span.x;
    let fb = a + (b - a) * span.y;
    let u = clamp((p.x - a.x) / len, 0.0, 1.0);
    let col = ui_ramp3(c0, c1, c2, u);
    let d_track = ui_segment(p, a, b) - r;
    let tc = ui_cover(d_track, aa);
    // A dark tube, its rim faintly lit.
    let rim = exp(-abs(d_track + 0.5) / 1.2) * 0.35;
    let track = select(vec4<f32>(col * (0.05 + rim) + vec3<f32>(0.02), 0.55) * tc, pfx_track(col, tc), pfx_on_light);
    // Along the fill, faded in at its tail and out past its head.
    let env = smoothstep(fa.x - r, fa.x + r, p.x) * (1.0 - smoothstep(fb.x - r * 0.3, fb.x + r * 0.6, p.x));
    let across = p.y - r;
    var light = pfx_bolts(p.x, across, r * 1.3, now, col) * env * look.x * 1.4;
    // Plasma in the tube.
    light = light + col * 0.45 * exp(-abs(across) / (r * 0.7)) * env * tc;
    let dh = length(p - fb);
    let head = select(vec3<f32>(0.0), pfx_sparks(dh, atan2(p.y - fb.y, p.x - fb.x + 0.00001), r, now, col) * look.x, has && span.y < 0.999);
    light = select(vec3<f32>(0.0), light + head, has);
    // The light fades out by the layer's edge.
    let edge = pfx_falloff(max(abs(across) - r, 0.0), reach * 1.5);
    return pfx_add(track, light * select(edge, 1.0, abs(across) <= r));
}

fn pfx_ring_electric(p: vec2<f32>, size: vec2<f32>, now: f32, value: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, look: vec4<f32>, reach: f32) -> vec4<f32> {
    let aa = max(fwidth(p.x), 0.001);
    let c = size * 0.5;
    let hw = look.w * 0.5;
    let rr = min(size.x, size.y) * 0.5 - hw - 1.0;
    let q = p - c;
    let dist = length(q);
    let ang = fract(atan2(q.x, -q.y + 0.00001) / PFX_TAU);
    let arc = pfx_arc(now, value, look.z);
    let has = arc.y > 0.0005;
    let full = arc.y > 0.9995;
    let rel = fract(ang - arc.x);
    let d_band = abs(dist - rr) - hw;
    let tc = ui_cover(d_band, aa);
    let col = select(pfx_ring_ramp(c0, c1, c2, ang), ui_ramp3(c0, c1, c2, rel / max(arc.y, 0.0001)), look.z > 0.5);
    let rim = exp(-abs(d_band + 0.5) / 1.2) * 0.35;
    let track = select(vec4<f32>(col * (0.05 + rim) + vec3<f32>(0.02), 0.55) * tc, pfx_track(col, tc), pfx_on_light);
    // Along the arc in pixels, faded in at its start and out past its end.
    let along_turns = select(rel, ang, full);
    let span_px = arc.y * PFX_TAU * rr;
    let s_px = rel * PFX_TAU * rr;
    let fade_in = smoothstep(0.0, hw * 2.0, s_px) * smoothstep(PFX_TAU * rr, PFX_TAU * rr - hw * 2.0, s_px);
    let env = select(fade_in * (1.0 - smoothstep(span_px - hw * 0.3, span_px + hw * 0.6, s_px)), 1.0, full);
    let across = dist - rr;
    var light = pfx_bolts(along_turns * PFX_TAU * rr, across, hw * 1.3, now, col) * env * look.x * 1.4;
    light = light + col * 0.45 * exp(-abs(across) / (hw * 0.7)) * env * tc;
    let a1 = (arc.x + arc.y) * PFX_TAU;
    let head_at = c + rr * vec2<f32>(sin(a1), -cos(a1));
    let dh = length(p - head_at);
    let head = select(vec3<f32>(0.0), pfx_sparks(dh, atan2(p.y - head_at.y, p.x - head_at.x + 0.00001), hw, now, col) * look.x, has && !full);
    light = select(vec3<f32>(0.0), light + head, has);
    let edge = pfx_falloff(max(abs(across) - hw, 0.0), reach * 1.5);
    return pfx_add(track, light * select(edge, 1.0, abs(across) <= hw));
}
