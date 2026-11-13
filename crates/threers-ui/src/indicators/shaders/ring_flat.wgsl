fn pfx_ring_flat(p: vec2<f32>, size: vec2<f32>, now: f32, value: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, look: vec4<f32>, reach: f32) -> vec4<f32> {
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
    let d0 = length(p - e0);
    let d1 = length(p - e1);
    let inside = full || rel <= arc.y;
    var d_fill = select(min(d0, d1) - hw, d_band, inside);
    d_fill = select(100000.0, d_fill, has);
    // Along the arc, 0 at its start (outside it, the nearer end).
    let k = select(select(0.0, 1.0, d1 < d0), rel / max(arc.y, 0.0001), inside);
    let busy = look.z > 0.5;
    // Determinate, by where round the ring the nearest point of the arc is.
    let col = select(pfx_ring_ramp(c0, c1, c2, select(k * arc.y, ang, full)), ui_ramp3(c0, c1, c2, k), busy);
    let fade = select(1.0, smoothstep(0.0, 0.85, k), busy);
    let mid = clamp((d_fill + hw) / max(hw, 0.001), 0.0, 1.0);
    let core = pow(1.0 - mid, 3.0);
    let dh = d1 / max(hw, 0.5);
    let spark = select(0.0, exp(-dh * dh * 0.5), has && !full);
    let sh = look.y * pfx_shimmer(select(ang, k, busy), now);
    let light = col * (1.0 + sh * 0.9) * fade + vec3<f32>(core * 0.55 + spark * 0.5 + sh * 0.25) * fade;
    let fc = ui_cover(d_fill, aa) * max(fade, 0.15);
    let tc = ui_cover(d_band, aa);
    let tcol = pfx_ring_ramp(c0, c1, c2, ang);
    let track = pfx_track(tcol, tc);
    let body = vec4<f32>(light * fc, fc) + track * (1.0 - fc);
    let halo = select(vec3<f32>(0.0), col * look.x * 1.5 * fade * pfx_falloff(d_fill, reach) + vec3<f32>(spark * look.x * 0.4), has);
    return pfx_out(body, halo);
}
