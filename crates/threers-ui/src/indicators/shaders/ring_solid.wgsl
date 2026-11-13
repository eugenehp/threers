fn pfx_torus(q: vec3<f32>, rr: f32, rt: f32) -> f32 {
    return length(vec2<f32>(length(q.xy) - rr, q.z)) - rt;
}

// The nearest point of the ring's core circle to `q`.
fn pfx_core(q: vec3<f32>, rr: f32) -> vec3<f32> {
    let l = length(q.xy);
    return vec3<f32>(select(vec2<f32>(0.0, rr), q.xy / l * rr, l > 0.000001), 0.0);
}

// The liquid: the torus from `arc.x` round `arc.y` turns, its ends round.
// (Its nearest core point: in `pfx_liquid_core`.)
fn pfx_liquid_core(q: vec3<f32>, rr: f32, arc: vec2<f32>) -> vec3<f32> {
    if (arc.y >= 0.9995) {
        return pfx_core(q, rr);
    }
    let ang = fract(atan2(q.x, q.y + 0.00001) / PFX_TAU);
    if (fract(ang - arc.x) <= arc.y) {
        return pfx_core(q, rr);
    }
    let a0 = arc.x * PFX_TAU;
    let a1 = (arc.x + arc.y) * PFX_TAU;
    let e0 = vec3<f32>(sin(a0), cos(a0), 0.0) * rr;
    let e1 = vec3<f32>(sin(a1), cos(a1), 0.0) * rr;
    return select(e1, e0, length(q - e0) < length(q - e1));
}

fn pfx_liquid(q: vec3<f32>, rr: f32, ri: f32, arc: vec2<f32>) -> f32 {
    if (arc.y <= 0.0005) {
        return 100000.0;
    }
    return length(q - pfx_liquid_core(q, rr, arc)) - ri;
}

fn pfx_ring_solid(p: vec2<f32>, size: vec2<f32>, now: f32, value: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, look: vec4<f32>, reach: f32) -> vec4<f32> {
    let aa = max(fwidth(p.x), 0.001);
    let c = size * 0.5;
    let rt = look.w * 0.5;
    let rr = min(size.x, size.y) * 0.5 - rt - 1.0;
    let ri = rt * 0.6;
    let m = pfx_orient(0.5);
    let mt = transpose(m);
    let eye = min(size.x, size.y) * 3.5;
    let rdv = normalize(vec3<f32>(p.x - c.x, c.y - p.y, -eye));
    let ro = mt * vec3<f32>(0.0, 0.0, eye);
    let rd = mt * rdv;
    let arc = pfx_arc(now, value, look.z);
    let has = arc.y > 0.0005;
    let busy = look.z > 0.5;
    let bound = rr + rt + 1.0;
    let t0 = max(eye - bound - 1.0, 0.0);
    let t1 = eye + bound;
    let eps = 0.02;
    // Fewer steps for a small ring: there is less to miss.
    let steps = select(48, 28, min(size.x, size.y) < 72.0);
    // The liquid along the ray as it is: how near it comes (the glow).
    var t = t0;
    var near = 100000.0;
    var near_at = ro + rd * t0;
    for (var i = 0; i < steps; i = i + 1) {
        let q = ro + rd * t;
        let d = pfx_liquid(q, rr, ri, arc);
        if (d < near) {
            near = d;
            near_at = q;
        }
        if (d < eps) {
            break;
        }
        t = t + d;
        if (t > t1) {
            break;
        }
    }
    // The glass.
    var tg = t0;
    var near_g = 100000.0;
    var near_gt = t0;
    var hit_g = false;
    for (var i = 0; i < steps; i = i + 1) {
        let d = pfx_torus(ro + rd * tg, rr, rt);
        if (d < near_g) {
            near_g = d;
            near_gt = tg;
        }
        if (d < eps) {
            hit_g = true;
            break;
        }
        tg = tg + d;
        if (tg > t1) {
            break;
        }
    }
    let qg = ro + rd * select(near_gt, tg, hit_g);
    let gn = normalize(qg - pfx_core(qg, rr) + vec3<f32>(0.0, 0.0, 0.00001));
    let gcov = select(ui_cover(near_g, aa), 1.0, hit_g);
    // The liquid through the glass, along the bent ray.
    let rr_dir = refract(rd, gn, 1.0 / 1.5);
    var tl = 0.0;
    var near_l = 100000.0;
    var at_l = qg;
    for (var i = 0; i < 28; i = i + 1) {
        let q = qg + rr_dir * tl;
        let d = pfx_liquid(q, rr, ri, arc);
        if (d < near_l) {
            near_l = d;
            at_l = q;
        }
        if (d < eps) {
            break;
        }
        tl = tl + d;
        if (tl > rt * 4.0) {
            break;
        }
    }
    let ln = normalize(at_l - pfx_liquid_core(at_l, rr, arc) + vec3<f32>(0.0, 0.0, 0.00001));
    let ang = fract(atan2(at_l.x, at_l.y + 0.00001) / PFX_TAU);
    let rel = fract(ang - arc.x);
    let kk = clamp(rel / max(arc.y, 0.0001), 0.0, 1.0);
    // Outside the arc, the nearer end.
    let k = select(kk, select(1.0, 0.0, rel > arc.y + (1.0 - arc.y) * 0.5), rel > arc.y);
    let col = select(pfx_ring_ramp(c0, c1, c2, select(k * arc.y, ang, arc.y >= 0.9995)), ui_ramp3(c0, c1, c2, k), busy);
    let fade = select(1.0, smoothstep(0.0, 0.85, k), busy);
    let facing = max(dot(ln, -rr_dir), 0.0);
    let sh = look.y * pfx_shimmer(select(ang, k, busy), now);
    let liquid = (col * (0.45 + 0.8 * facing) * (1.0 + sh * 0.8) + vec3<f32>(pow(facing, 8.0) * 0.45 + sh * 0.2)) * fade;
    let lcov = select(0.0, ui_cover(near_l, aa), has) * max(fade, 0.15);
    // The liquid's light scattered in the glass, fading with the distance
    // from it (so no seam where the colour flips, in the gap's middle).
    let dl = pfx_liquid(qg, rr, ri, arc);
    let scatter = select(vec3<f32>(0.0), col * 0.35 * exp(-max(dl, 0.0) / (rt * 1.2)) * fade, has);
    let behind = vec4<f32>(liquid * lcov + scatter * (1.0 - lcov), lcov);
    let tint = vec3<f32>(0.85, 0.93, 1.0);
    let body = pfx_glass(m * gn, rdv, behind, tint) * gcov;
    // The glow, by where the ray passes nearest the liquid.
    let ha = fract(atan2(near_at.x, near_at.y + 0.00001) / PFX_TAU);
    let hrel = fract(ha - arc.x);
    let hkk = clamp(hrel / max(arc.y, 0.0001), 0.0, 1.0);
    let hk = select(hkk, select(1.0, 0.0, hrel > arc.y + (1.0 - arc.y) * 0.5), hrel > arc.y);
    let hcol = select(pfx_ring_ramp(c0, c1, c2, select(hk * arc.y, ha, arc.y >= 0.9995)), ui_ramp3(c0, c1, c2, hk), busy);
    let hfade = select(1.0, smoothstep(0.0, 0.85, hk), busy);
    let halo = select(vec3<f32>(0.0), hcol * look.x * 1.4 * hfade * pfx_falloff(near, reach), has);
    return pfx_out(body, halo);
}
