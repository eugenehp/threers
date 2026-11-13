fn pfx_bar_flat(p: vec2<f32>, size: vec2<f32>, now: f32, value: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, look: vec4<f32>, reach: f32) -> vec4<f32> {
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
    let d_fill = select(100000.0, ui_segment(p, fa, fb) - r, has);
    // Along the fill, 0 at its tail: a comet fades toward it.
    let k = clamp((p.x - fa.x) / max(fb.x - fa.x, 0.001), 0.0, 1.0);
    let fade = select(1.0, smoothstep(0.0, 0.8, k), look.z > 0.5);
    // A white core along the middle, a spark at the head.
    let mid = clamp((d_fill + r) / max(r, 0.001), 0.0, 1.0);
    let core = pow(1.0 - mid, 3.0);
    let dh = length(p - fb) / max(r, 0.5);
    let spark = select(0.0, exp(-dh * dh * 0.5), has && (look.z > 0.5 || span.y < 0.999));
    let sh = look.y * pfx_shimmer(u, now);
    let light = col * (1.0 + sh * 0.9) * fade + vec3<f32>(core * 0.55 + spark * 0.5 + sh * 0.25) * fade;
    let fc = ui_cover(d_fill, aa) * max(fade, 0.15);
    let tc = ui_cover(d_track, aa);
    let track = pfx_track(col, tc);
    let body = vec4<f32>(light * fc, fc) + track * (1.0 - fc);
    let halo = select(vec3<f32>(0.0), col * look.x * 1.5 * fade * pfx_falloff(d_fill, reach) + vec3<f32>(spark * look.x * 0.4), has);
    return pfx_out(body, halo);
}
