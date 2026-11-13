fn pfx_bar_solid(p: vec2<f32>, size: vec2<f32>, now: f32, value: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, look: vec4<f32>, reach: f32) -> vec4<f32> {
    let aa = max(fwidth(p.x), 0.001);
    let c = size * 0.5;
    let r = size.y * 0.5;
    let half = max(size.x * 0.5 - r, 0.0);
    let m = pfx_orient(0.42);
    let mt = transpose(m);
    let eye = max(size.x, size.y) * 3.0;
    let rdv = normalize(vec3<f32>(p.x - c.x, c.y - p.y, -eye));
    let ro = mt * vec3<f32>(0.0, 0.0, eye);
    let rd = mt * rdv;
    let pa = vec3<f32>(-half, 0.0, 0.0);
    let pb = vec3<f32>(half, 0.0, 0.0);
    let len = max(2.0 * half, 0.001);
    let span = pfx_span(now, value, look.z);
    let has = span.y > span.x + 0.0005;
    let busy = look.z > 0.5;
    let ri = r * 0.62;
    let la = pa + (pb - pa) * span.x;
    let lb = pa + (pb - pa) * span.y;
    // The glass: where the ray meets it, or (just outside it, its edge
    // smoothed) where it passes nearest.
    let tg = pfx_capsule_t(ro, rd, pa, pb, r);
    let near_g = pfx_ray_segment(ro, rd, pa, pb);
    let gcov = ui_cover(near_g.x - r, aa);
    let hg = ro + rd * select(near_g.y, tg, tg > 0.0);
    let gn = normalize(hg - pfx_on_segment(hg, pa, pb) + vec3<f32>(0.0, 0.0, 0.00001));
    // The liquid, through the glass: the ray bent going in.
    let rr = refract(rd, gn, 1.0 / 1.5);
    let tl = select(-1.0, pfx_capsule_t(hg, rr, la, lb, ri), has);
    let near_l = pfx_ray_segment(hg, rr, la, lb);
    let hl = hg + rr * select(max(near_l.y, 0.0), tl, tl > 0.0);
    let ln = normalize(hl - pfx_on_segment(hl, la, lb) + vec3<f32>(0.0, 0.0, 0.00001));
    let u = clamp((hl.x + half) / len, 0.0, 1.0);
    let col = ui_ramp3(c0, c1, c2, u);
    let k = clamp((hl.x - la.x) / max(lb.x - la.x, 0.001), 0.0, 1.0);
    let fade = select(1.0, smoothstep(0.0, 0.8, k), busy);
    let facing = max(dot(ln, -rr), 0.0);
    let sh = look.y * pfx_shimmer(u, now);
    let hd = length(hl - lb) / max(ri, 0.5);
    let spark = select(0.0, exp(-hd * hd * 0.6), has && (busy || span.y < 0.999));
    let liquid = (col * (0.45 + 0.8 * facing) * (1.0 + sh * 0.8) + vec3<f32>(pow(facing, 8.0) * 0.45 + spark * 0.4 + sh * 0.2)) * fade;
    let lcov = select(0.0, ui_cover(near_l.x - ri, aa), has) * max(fade, 0.15);
    // The liquid's light scattered in the glass round it.
    let dl = length(hg - pfx_on_segment(hg, la, lb)) - ri;
    let scatter = select(vec3<f32>(0.0), col * 0.3 * exp(-max(dl, 0.0) / (r * 1.5)) * fade, has);
    let behind = vec4<f32>(liquid * lcov + scatter * (1.0 - lcov), lcov);
    let tint = vec3<f32>(0.85, 0.93, 1.0);
    let body = pfx_glass(m * gn, rdv, behind, tint) * gcov;
    // The glow, round the liquid as the eye sees it.
    let near_h = pfx_ray_segment(ro, rd, la, lb);
    let hcol = ui_ramp3(c0, c1, c2, (la.x + (lb.x - la.x) * near_h.z + half) / len);
    let hfade = select(1.0, smoothstep(0.0, 0.8, near_h.z), busy);
    let halo = select(vec3<f32>(0.0), hcol * look.x * 1.4 * hfade * pfx_falloff(near_h.x - ri, reach), has);
    return pfx_out(body, halo);
}
