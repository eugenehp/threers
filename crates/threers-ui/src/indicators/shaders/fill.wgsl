// A fill (`PFX_FILL`: a built-in's dispatcher or a custom fill's function)
// drawn along a bar and round a ring.
fn pfx_bar_fx(p: vec2<f32>, size: vec2<f32>, now: f32, value: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, look: vec4<f32>, reach: f32) -> vec4<f32> {
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
    let busy = look.z > 0.5;
    let k = clamp((p.x - fa.x) / max(fb.x - fa.x, 0.001), 0.0, 1.0);
    let fade = select(1.0, smoothstep(0.0, 0.8, k), busy);
    let track = pfx_track(col, ui_cover(d_track, aa));
    let cover = ui_cover(d_fill, aa) * max(fade, 0.15);
    let env = select(0.0, smoothstep(fa.x - r * 0.5, fa.x + r, p.x) * (1.0 - smoothstep(fb.x - r * 0.6, fb.x + r * 0.3, p.x)) * fade, has);
    let fx = PFX_FILL(p.x, r - p.y, r, env, cover, u, now, c0, c1, c2);
    let body = fx + track * (1.0 - fx.a);
    let halo = select(vec3<f32>(0.0), col * look.x * 1.2 * fade * pfx_falloff(d_fill, reach), has);
    return pfx_out(body, halo);
}

fn pfx_ring_fx(p: vec2<f32>, size: vec2<f32>, now: f32, value: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, look: vec4<f32>, reach: f32) -> vec4<f32> {
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
    let a0 = arc.x * PFX_TAU;
    let a1 = (arc.x + arc.y) * PFX_TAU;
    let e0 = c + rr * vec2<f32>(sin(a0), -cos(a0));
    let e1 = c + rr * vec2<f32>(sin(a1), -cos(a1));
    let d_band = abs(dist - rr) - hw;
    let inside = full || rel <= arc.y;
    let d_fill = select(100000.0, select(min(length(p - e0), length(p - e1)) - hw, d_band, inside), has);
    let busy = look.z > 0.5;
    let k = clamp(rel / max(arc.y, 0.0001), 0.0, 1.0);
    let fade = select(1.0, smoothstep(0.0, 0.85, k), busy);
    let col = select(pfx_ring_ramp(c0, c1, c2, ang), ui_ramp3(c0, c1, c2, k), busy);
    let track = pfx_track(pfx_ring_ramp(c0, c1, c2, ang), ui_cover(d_band, aa));
    let cover = ui_cover(d_fill, aa) * max(fade, 0.15);
    // Along the arc in pixels, faded in at its start and out past its end.
    let s_px = rel * PFX_TAU * rr;
    let span_px = arc.y * PFX_TAU * rr;
    let run = smoothstep(0.0, hw * 2.0, s_px) * (1.0 - smoothstep(span_px - hw * 0.6, span_px + hw * 0.3, s_px)) * fade;
    let env = select(0.0, select(run, 1.0, full), has);
    let fx = PFX_FILL(select(rel, ang, full) * PFX_TAU * rr, dist - rr, hw, env, cover, select(ang, k, busy), now, c0, c1, c2);
    let body = fx + track * (1.0 - fx.a);
    let halo = select(vec3<f32>(0.0), col * look.x * 1.2 * fade * pfx_falloff(d_fill, reach), has);
    return pfx_out(body, halo);
}
