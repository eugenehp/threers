// A beam of light running round a box's border, drawn over what the box
// draws: a crisp line on the edge, a tight halo, and a wide soft glow
// reaching in. Everything is measured from the edge of a rounded box, so
// nothing spills outside it.

const BB_PI: f32 = 3.14159265;

// Signed distance to a rounded box centred at the origin (negative inside).
fn bb_rounded_box(p: vec2<f32>, hs: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - (hs - r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

// The border's length: four straight sides and four quarter arcs.
fn bb_perimeter(hs: vec2<f32>, r: f32) -> f32 {
    let ih = max(hs - r, vec2<f32>(0.0));
    return 4.0 * ih.x + 4.0 * ih.y + 2.0 * BB_PI * r;
}

// How far round the border `p` is, 0 to 1, clockwise on screen from the
// middle of the right side, measured along the edge (not by angle from the
// middle, which would fan the beam's ends out towards the centre).
fn bb_along(p: vec2<f32>, hs: vec2<f32>, r: f32) -> f32 {
    let ih = max(hs - r, vec2<f32>(0.0));
    let q = BB_PI * r * 0.5;
    let l = bb_perimeter(hs, r);
    let base_bl = ih.y + q + 2.0 * ih.x;
    let base_l = base_bl + q;
    let base_tl = base_l + 2.0 * ih.y;
    let base_tr = base_tl + q + 2.0 * ih.x;
    let ap = abs(p);
    var s = 0.0;
    if (ap.x > ih.x && ap.y > ih.y) {
        let v = ap - ih;
        let ang = atan2(v.y, v.x);
        if (p.x >= 0.0 && p.y >= 0.0) {
            s = ih.y + r * ang;
        } else if (p.x < 0.0 && p.y >= 0.0) {
            s = base_bl + r * (BB_PI * 0.5 - ang);
        } else if (p.x < 0.0) {
            s = base_tl + r * ang;
        } else {
            s = base_tr + r * (BB_PI * 0.5 - ang);
        }
    } else if ((hs.x - ap.x) < (hs.y - ap.y)) {
        if (p.x >= 0.0) {
            s = select(l + p.y, p.y, p.y >= 0.0);
        } else {
            s = base_l + (ih.y - p.y);
        }
    } else {
        if (p.y >= 0.0) {
            s = ih.y + q + (ih.x - p.x);
        } else {
            s = base_tl + q + (p.x + ih.x);
        }
    }
    return fract(s / max(l, 0.001));
}

// The point `s` pixels round the border from where `bb_along` starts.
fn bb_border_point(s_in: f32, hs: vec2<f32>, r: f32) -> vec2<f32> {
    let ih = max(hs - r, vec2<f32>(0.0));
    let q = BB_PI * r * 0.5;
    let l = bb_perimeter(hs, r);
    var s = s_in - l * floor(s_in / max(l, 0.001));
    if (s < ih.y) { return vec2<f32>(hs.x, s); }
    s -= ih.y;
    if (s < q) { let a = s / r; return vec2<f32>(ih.x, ih.y) + r * vec2<f32>(cos(a), sin(a)); }
    s -= q;
    if (s < 2.0 * ih.x) { return vec2<f32>(ih.x - s, hs.y); }
    s -= 2.0 * ih.x;
    if (s < q) { let a = BB_PI * 0.5 + s / r; return vec2<f32>(-ih.x, ih.y) + r * vec2<f32>(cos(a), sin(a)); }
    s -= q;
    if (s < 2.0 * ih.y) { return vec2<f32>(-hs.x, ih.y - s); }
    s -= 2.0 * ih.y;
    if (s < q) { let a = BB_PI + s / r; return vec2<f32>(-ih.x, -ih.y) + r * vec2<f32>(cos(a), sin(a)); }
    s -= q;
    if (s < 2.0 * ih.x) { return vec2<f32>(-ih.x + s, -hs.y); }
    s -= 2.0 * ih.x;
    if (s < q) { let a = BB_PI * 1.5 + s / r; return vec2<f32>(ih.x, -ih.y) + r * vec2<f32>(cos(a), sin(a)); }
    s -= q;
    return vec2<f32>(hs.x, -ih.y + s);
}

// Five colour stops, tail (0) to head (1), blended linearly. Chained mixes
// rather than an indexed array, which Metal would put in stack memory.
fn bb_ramp(c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, c3: vec4<f32>, c4: vec4<f32>, t: f32) -> vec4<f32> {
    let x = clamp(t, 0.0, 1.0) * 4.0;
    let a = mix(c0, c1, clamp(x, 0.0, 1.0));
    let b = mix(a, c2, clamp(x - 1.0, 0.0, 1.0));
    let c = mix(b, c3, clamp(x - 2.0, 0.0, 1.0));
    return mix(c, c4, clamp(x - 3.0, 0.0, 1.0));
}

// Lighter towards both ends of the beam.
fn bb_bell(t: f32, k: f32) -> f32 {
    return pow(max(sin(BB_PI * clamp(t, 0.0, 1.0)), 0.0), k);
}

// The wide glow: a chain of round Gaussian blobs laid along each comet's
// path on the border, like a blurred stroke. It never asks which edge is
// nearest, so corners come out round with no seams. Premultiplied colour
// and weight, peaking near 1 on the border. `head` is where round the
// border (0 to 1) the first comet's head is; `n` comets share the border,
// each `arc` of its own share long.
fn bb_glow(p: vec2<f32>, dist: f32, hs: vec2<f32>, r: f32, head: f32, dir: f32, n: i32, arc: f32, sigma_base: f32,
           c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, c3: vec4<f32>, c4: vec4<f32>) -> vec4<f32> {
    let l = bb_perimeter(hs, r);
    let k = 24;
    let delta = arc * l / (f32(n) * f32(k));
    // Overlapping, so the chain does not bead.
    let sigma = max(sigma_base, 0.8 * delta);
    if (dist > 3.0 * sigma) {
        return vec4<f32>(0.0);
    }
    let inv2s2 = 0.5 / (sigma * sigma);
    let norm = delta / (sigma * 2.50663);
    var acc = vec4<f32>(0.0);
    for (var j = 0; j < n; j = j + 1) {
        for (var i = 0; i < k; i = i + 1) {
            let t = (f32(i) + 0.5) / f32(k);
            let sp = fract(head + (f32(j) + 1.0 - arc + t * arc) / f32(n));
            let d = p - bb_border_point(sp * l, hs, r);
            let g = exp(-dot(d, d) * inv2s2);
            if (g < 0.003) {
                continue;
            }
            // The head leads whichever way it goes.
            let c = bb_ramp(c0, c1, c2, c3, c4, select(t, 1.0 - t, dir < 0.0));
            let v = g * norm * bb_bell(t, 1.6) * clamp(c.a, 0.0, 1.0);
            acc += vec4<f32>(c.rgb * v, v);
        }
    }
    return acc;
}

// Where round the border (0 to 1, as `bb_along`) the point is that lies
// `turns` clockwise from straight up, seen from the middle.
fn bb_at_angle(turns: f32, hs: vec2<f32>, r: f32) -> f32 {
    let a = turns * 2.0 * BB_PI;
    let d = vec2<f32>(sin(a), -cos(a));
    let reach = min(hs.x / max(abs(d.x), 0.0001), hs.y / max(abs(d.y), 0.0001));
    return bb_along(d * reach, hs, r);
}

// A line `width` pixels wide on the edge, `dist` pixels in from it.
fn bb_line(dist: f32, width: f32) -> f32 {
    return 1.0 - smoothstep(max(width - 1.0, 0.0), width + 0.75, dist);
}

// The beam alone at `position` in a box `size`, premultiplied.
// `geom`: the corner radius (-1, the circle inside the box; -2 - f, a
// share f of the shorter side), how many comets with the sign of their
// direction (+ clockwise), the edge a line beam runs along (1 top, 2
// bottom, 3 left, 4 right; 0 round the border), the line's width.
// `look`: each comet's length as a share of the border (of its stretch of
// it, with several), the glow's reach in pixels, intensity, and where the
// head starts in turns clockwise from straight up. `run`: laps a second,
// where round it is at time 0, when the fade started, and the fade's
// seconds plus one (above 0 fading in, below fading out).
fn bb_beam(position: vec2<f32>, size: vec2<f32>, now: f32,
           c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, c3: vec4<f32>, c4: vec4<f32>,
           geom: vec4<f32>, look: vec4<f32>, run: vec4<f32>) -> vec4<f32> {
    let fade = abs(run.w) - 1.0;
    let faded = clamp((now - run.z) / max(fade, 0.0001), 0.0, 1.0);
    let eased = faded * faded * (3.0 - 2.0 * faded);
    let level = select(1.0 - eased, eased, run.w > 0.0);
    if (level <= 0.0) {
        return vec4<f32>(0.0);
    }
    var hs = size * 0.5;
    var r = min(max(geom.x, 0.0), min(hs.x, hs.y));
    if (geom.x < -1.5) {
        r = min((-geom.x - 2.0) * min(size.x, size.y), min(hs.x, hs.y));
    } else if (geom.x < 0.0) {
        hs = vec2<f32>(min(hs.x, hs.y));
        r = hs.x;
    }
    let p = position - size * 0.5;
    let dist = -bb_rounded_box(p, hs, r);
    if (dist < 0.0) {
        return vec4<f32>(0.0);
    }
    let phase = fract(now * run.x + run.y);
    let dir = sign(geom.y);
    let n = clamp(i32(abs(geom.y) + 0.5), 1, 8);
    let width = geom.w;
    let glow = look.y;
    var acc = vec4<f32>(0.0);
    if (geom.z < 0.5) {
        let arc = clamp(look.x * f32(n), 0.02, 0.98);
        // The end at the larger `bb_along`: the head going clockwise, the
        // tail the other way (whose head then starts at the angle).
        let head = fract(dir * phase + bb_at_angle(look.w, hs, r) + select(0.0, arc / f32(n), dir < 0.0));
        // A crisp line and a tight halo on the edge (where the nearest
        // edge is a fine guide to how far round).
        let a = bb_along(p, hs, r);
        let tt = (fract((a - head) * f32(n)) - (1.0 - arc)) / arc;
        if (tt >= 0.0) {
            let halo = 0.6 * exp(-dist / max(glow * 0.1, 0.6));
            let c = bb_ramp(c0, c1, c2, c3, c4, select(tt, 1.0 - tt, dir < 0.0));
            let w = (bb_line(dist, width) + halo) * bb_bell(tt, 1.2) * clamp(c.a, 0.0, 1.0);
            acc += vec4<f32>(c.rgb * w, w);
        }
        acc += 0.55 * bb_glow(p, dist, hs, r, head, dir, n, arc, max(glow, 0.001) * 0.7, c0, c1, c2, c3, c4);
    } else {
        // To and fro along one edge, easing at the ends.
        let side = i32(geom.z + 0.5);
        var u = position.x / size.x;
        var edge = size.y - position.y;
        if (side == 1) {
            edge = position.y;
        } else if (side == 3) {
            u = position.y / size.y;
            edge = position.x;
        } else if (side == 4) {
            u = position.y / size.y;
            edge = size.x - position.x;
        }
        // Against the direction: from the far end.
        u = select(u, 1.0 - u, dir < 0.0);
        let len = clamp(look.x, 0.05, 0.95);
        let tri = select((1.0 - phase) * 2.0, phase * 2.0, phase < 0.5);
        let centre = len * 0.5 + (1.0 - len) * tri * tri * (3.0 - 2.0 * tri);
        let k = (u - (centre - len * 0.5)) / len;
        if (k >= 0.0 && k <= 1.0) {
            let tp = 1.0 - abs(2.0 * k - 1.0);
            let profile = tp * tp * (3.0 - 2.0 * tp);
            let c = bb_ramp(c0, c1, c2, c3, c4, k);
            let halo = 0.8 * exp(-edge / max(glow * 0.2, 0.5));
            let wide = 0.6 * exp(-pow(max(edge, 0.0) / max(glow, 0.001), 1.4));
            let w = (bb_line(edge, width) + halo + wide) * profile * clamp(c.a, 0.0, 1.0);
            acc += vec4<f32>(c.rgb * w, w);
        }
    }
    if (acc.a <= 0.0001) {
        return vec4<f32>(0.0);
    }
    // The colour of everything lighting the pixel, averaged; as opaque as
    // it all adds up to.
    let alpha = clamp(acc.a * look.z, 0.0, 1.0) * level;
    return vec4<f32>(acc.rgb / acc.a * alpha, alpha);
}
