const PFX_TAU: f32 = 6.2831853;

// Round a ring, closed: the last tenth turns back into the first colour,
// so a full ring (and its glow and glass) has no seam at the top.
fn pfx_ring_ramp(c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, t: f32) -> vec3<f32> {
    let u = fract(t);
    return mix(ui_ramp3(c0, c1, c2, u / 0.9), c0.rgb, smoothstep(0.9, 1.0, u));
}

// Where along the bar the fill runs, 0 to 1: from the start to the value,
// or a comet running along it while busy.
fn pfx_span(now: f32, value: f32, busy: f32) -> vec2<f32> {
    if (busy < 0.5) {
        return vec2<f32>(0.0, clamp(value, 0.0, 1.0));
    }
    let c = fract(now * 0.6);
    let e = c * c * (3.0 - 2.0 * c);
    let head = e * 1.45 - 0.05;
    return vec2<f32>(clamp(head - 0.4, 0.0, 1.0), clamp(head, 0.0, 1.0));
}

// Where round the ring the arc starts and how far it goes, in turns: from
// the top to the value, or a chasing arc that grows and shrinks while busy.
fn pfx_arc(now: f32, value: f32, busy: f32) -> vec2<f32> {
    if (busy < 0.5) {
        return vec2<f32>(0.0, clamp(value, 0.0, 1.0));
    }
    let breathe = 0.5 + 0.5 * sin(now * 3.1);
    return vec2<f32>(fract(now * 0.9 + 0.08 * sin(now * 3.1)), 0.1 + 0.55 * breathe);
}

// How much of the glow reaches `d` pixels out: none at `reach`, the edge
// of the layer.
fn pfx_falloff(d: f32, reach: f32) -> f32 {
    let x = clamp(max(d, 0.0) / max(reach, 0.001), 0.0, 1.0);
    let w = 1.0 - x;
    return exp(-4.0 * x) * w * w;
}

// A band of light running along (`u`, 0 to 1).
fn pfx_shimmer(u: f32, now: f32) -> f32 {
    let x = fract(u * 0.8 - now * 0.55) - 0.5;
    return exp(-x * x * 90.0);
}

// The body (premultiplied linear) with the halo's light (linear) added
// round it, as the effect returns it.
// On a light background added light does not show: there the halo is a
// haze of its colour, laid over (deeper, and more opaque).
fn pfx_out(body: vec4<f32>, halo: vec3<f32>) -> vec4<f32> {
    let a = clamp(body.a, 0.0, 1.0);
    let h = (vec3<f32>(1.0) - exp(-max(halo, vec3<f32>(0.0)))) * (1.0 - a);
    let hm = clamp(max(h.r, max(h.g, h.b)), 0.0, 1.0);
    if (pfx_on_light) {
        let tint = h / max(hm, 0.0001);
        let ha = hm * 0.55;
        return vec4<f32>(min(body.rgb, vec3<f32>(a)) + tint * 0.8 * ha, a + ha);
    }
    return vec4<f32>(min(body.rgb, vec3<f32>(a)) + h, a + hm * 0.5);
}

// An empty track of colour `col`, `cover` of the pixel: dark and clear on
// a dark background, pale and tinted on a light one.
fn pfx_track(col: vec3<f32>, cover: f32) -> vec4<f32> {
    if (pfx_on_light) {
        return vec4<f32>(mix(vec3<f32>(0.78, 0.8, 0.84), col, 0.18) * 0.6, 0.6) * cover;
    }
    return vec4<f32>(col * 0.08 + vec3<f32>(0.03), 0.4) * cover;
}
