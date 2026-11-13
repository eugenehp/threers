fn pfx_battery(p: vec2<f32>, size: vec2<f32>, now: f32, value: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, look: vec4<f32>, reach: f32, solid: bool) -> vec4<f32> {
    let aa = max(fwidth(p.x), 0.001);
    let h = size.y;
    let nub = h * 0.14;
    let bw = size.x - nub;
    let t = max(h * 0.08, 1.0);
    let r = h * 0.24;
    let bc = vec2<f32>(bw * 0.5, h * 0.5);
    let bh = vec2<f32>(bw * 0.5, h * 0.5);
    let d_body = ui_rbox(p, bc, bh, r);
    let d_shell = abs(ui_rbox(p, bc, bh - vec2<f32>(t * 0.5), r - t * 0.5)) - t * 0.5;
    let d_nub = ui_rbox(p, vec2<f32>(bw + nub * 0.5 - t * 0.5, h * 0.5), vec2<f32>(nub * 0.5 + t * 0.5, h * 0.2), nub * 0.45);
    // The charge, inside with a gap.
    let inset = t + max(h * 0.06, 1.0);
    let ih = h * 0.5 - inset;
    let iw = bw - 2.0 * inset;
    let span = pfx_span(now, value, look.z);
    let has = span.y > span.x + 0.0005;
    let busy = look.z > 0.5;
    let x0 = inset + iw * span.x;
    let x1 = inset + iw * span.y;
    let d_fill = select(100000.0, ui_rbox(p, vec2<f32>((x0 + x1) * 0.5, h * 0.5), vec2<f32>(max((x1 - x0) * 0.5, 0.0), ih), max(r - inset, h * 0.05)), has);
    let k = clamp((p.x - x0) / max(x1 - x0, 0.001), 0.0, 1.0);
    let col = select(ui_ramp3(c0, c1, c2, clamp(value, 0.0, 1.0)), ui_ramp3(c0, c1, c2, k), busy);
    let fade = select(1.0, smoothstep(0.0, 0.8, k), busy);
    let u = clamp((p.x - inset) / max(iw, 0.001), 0.0, 1.0);
    let sh = look.y * pfx_shimmer(u, now);
    // Lighter at the top, as a lit cylinder of charge.
    let top = 1.0 - clamp((p.y - (h * 0.5 - ih)) / max(2.0 * ih, 0.001), 0.0, 1.0);
    let charge = (col * (0.75 + 0.45 * top) * (1.0 + sh * 0.9) + vec3<f32>(sh * 0.25 + pow(top, 6.0) * 0.25)) * fade;
    let fc = ui_cover(d_fill, aa) * max(fade, 0.15);
    let filled = vec4<f32>(charge * fc, fc);
    var body: vec4<f32>;
    if (solid) {
        // Glass: the body's rim rounded over, reflecting the studio.
        let e = vec2<f32>(0.5, 0.0);
        let g = vec2<f32>(ui_rbox(p + e.xy, bc, bh, r) - ui_rbox(p - e.xy, bc, bh, r), ui_rbox(p + e.yx, bc, bh, r) - ui_rbox(p - e.yx, bc, bh, r));
        let bevel = max(t * 2.5, 2.0);
        let rim = 1.0 - clamp(-d_body / bevel, 0.0, 1.0);
        let slope = rim / sqrt(max(1.0 - rim * rim, 0.0004));
        let gd = normalize(g + vec2<f32>(0.00001, 0.0));
        let n = normalize(vec3<f32>(gd.x * slope, -gd.y * slope, 1.0));
        let inside = vec4<f32>(vec3<f32>(0.02, 0.025, 0.035) * 0.35, 0.35);
        let behind = filled + inside * (1.0 - fc);
        let glass = pfx_glass(n, vec3<f32>(0.0, 0.0, -1.0), behind, vec3<f32>(0.85, 0.93, 1.0));
        let cover = ui_cover(d_body, aa);
        let terminal = vec4<f32>(vec3<f32>(0.55, 0.6, 0.68), 1.0) * ui_cover(d_nub, aa) * (1.0 - cover);
        body = glass * cover + terminal;
    } else {
        let outline = ui_cover(min(d_shell, d_nub), aa);
        let shell = select(vec4<f32>(vec3<f32>(0.82, 0.85, 0.9) * 0.85, 0.85), vec4<f32>(vec3<f32>(0.3, 0.32, 0.36) * 0.9, 0.9), pfx_on_light) * outline;
        body = shell + filled * (1.0 - outline);
    }
    let halo = select(vec3<f32>(0.0), col * look.x * 1.2 * fade * pfx_falloff(d_fill, reach), has);
    return pfx_out(body, halo);
}

fn pfx_battery_flat(p: vec2<f32>, size: vec2<f32>, now: f32, value: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, look: vec4<f32>, reach: f32) -> vec4<f32> {
    return pfx_battery(p, size, now, value, c0, c1, c2, look, reach, false);
}

fn pfx_battery_solid(p: vec2<f32>, size: vec2<f32>, now: f32, value: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, look: vec4<f32>, reach: f32) -> vec4<f32> {
    return pfx_battery(p, size, now, value, c0, c1, c2, look, reach, true);
}
