// threers::fx::Surface — a liquid surface drawn from one distance field.
// Prepended with threers::sdf::WGSL and the renderer's ShaderMaterial preamble.
//
// u_data[0]  = viewport w, h (css px), dpr, strands in use
// u_data[1]  = shape: centre xy, half xy (round rect) | centre xy, radius (circle)
// u_data[2]  = mound from xy, to xy
// u_data[3]  = mound radius at base, at tip, lean, -
// u_data[4]  = the box worth shading: x0, y0, x1, y1
// u_data[5]  = shape kind (0 round rect, 1 circle, 2 polygon, 3 backdrop), corner radius, thickness, polygon points
//              a backdrop is a vertical gradient over the viewport: top, middle (at 55%), bottom in u_data[6..8].rgb
// u_data[6]  = look base rgb, f0
// u_data[7]  = look tint rgb, glints
// u_data[8]  = softbox, window, fill, floor
// u_data[9]  = shadow opacity, front, -, -
// s0         = strand points: x, y, radius, unused  (MS * NP)
// s1         = spikes (base xy, tip xy) * MSPK, spike (radius, alive) * MSPK, drops (x, y, r, alive) * MDROP
// s2         = polygon points: x, y, ...
const MS: i32 = 7;
const NP: i32 = 12;
const MSPK: i32 = 8;
const MDROP: i32 = 6;

fn v4(which: i32, i: i32) -> vec4<f32> {
    let o = u32(i) * 4u;
    if (which == 0) { return vec4<f32>(s0(o), s0(o + 1u), s0(o + 2u), s0(o + 3u)); }
    return vec4<f32>(s1(o), s1(o + 1u), s1(o + 2u), s1(o + 3u));
}

// Smooth union of (distance, shading thickness, radius) triples: the
// thickness and the geometric radius blend through the same fillet the
// surface does. The radius is how wide the nearest thing really is — a
// strand thinning to a thread — which is what decides whether a pixel at
// its edge needs a stroke's coverage rather than an edge's.
fn melt(a: vec3<f32>, b: vec3<f32>, k: f32) -> vec3<f32> {
    let m = sdf_smooth_union_blend(a.x, b.x, k);
    return vec3<f32>(m.x, mix(b.y, a.y, m.y), mix(b.z, a.z, m.y));
}

fn shape_distance(p: vec2<f32>) -> f32 {
    let h = u_data.data[1];
    let kind = u_data.data[5];
    if (kind.x < 0.5) {
        return sdf2_round_rect(p - h.xy, h.zw, kind.y);
    }
    if (kind.x < 1.5) {
        return sdf2_circle(p - h.xy, h.z);
    }
    // polygon: nearest edge, sign by crossings, grown by the corner radius
    let n = u32(kind.w);
    var best = 1e9;
    var inside = false;
    var j = n - 1u;
    for (var i = 0u; i < n; i = i + 1u) {
        let a = vec2<f32>(s2(i * 2u), s2(i * 2u + 1u));
        let b = vec2<f32>(s2(j * 2u), s2(j * 2u + 1u));
        best = min(best, sdf2_segment(p, a, b));
        if ((p.y >= a.y) != (p.y >= b.y)) {
            let t = (p.y - a.y) / (b.y - a.y);
            if (a.x + t * (b.x - a.x) > p.x) { inside = !inside; }
        }
        j = i;
    }
    return select(best, -best, inside) - kind.y;
}

fn field(p: vec2<f32>) -> vec3<f32> {
    let thick = u_data.data[5].z;
    // the body is as wide as it is: never thin
    var body = vec3<f32>(shape_distance(p), thick, 1e4);
    let mr = u_data.data[3].xy;
    if (mr.x > 0.01) {
        let m = u_data.data[2];
        body = melt(body, vec3<f32>(sdf2_tapered_capsule(p, m.xy, m.zw, mr.x, mr.y).x, thick, 1e4), 14.0);
    }
    // strands: one strand is a hard union of its pieces (no beads at the
    // joints), and neighbouring strands melt together where they touch
    var st = vec3<f32>(1e5, 2.0, 1e4);
    let count = i32(u_data.data[0].w);
    for (var s = 0; s < MS; s = s + 1) {
        if (s >= count) { break; }
        var one = vec3<f32>(1e5, 2.0, 1e4);
        for (var i = 0; i < NP - 1; i = i + 1) {
            let A = v4(0, s * NP + i);
            let B = v4(0, s * NP + i + 1);
            let d = sdf2_tapered_capsule(p, A.xy, B.xy, A.z, B.z);
            if (d.x < one.x) { one = vec3<f32>(d.x, d.y + 0.3, d.y); }
        }
        st = melt(st, one, 3.5);
    }
    for (var k = 0; k < MSPK; k = k + 1) {
        let r = v4(1, MSPK + k);
        if (r.y < 0.5) { continue; }
        let sp = v4(1, k);
        let d = sdf2_tapered_capsule(p, sp.xy, sp.zw, r.x, 0.35);
        st = melt(st, vec3<f32>(d.x, d.y + 1.0, d.y), 3.0);
    }
    for (var k = 0; k < MDROP; k = k + 1) {
        let d = v4(1, 2 * MSPK + k);
        if (d.w < 0.5) { continue; }
        st = melt(st, vec3<f32>(sdf2_circle(p - d.xy, d.z), d.z, d.z), 5.0);
    }
    var res = melt(body, st, 11.0);
    // deep inside the body, keep the body's round profile even where a strand root overlaps it
    res.y = max(res.y, body.y * clamp(-body.x / 6.0, 0.0, 1.0));
    return res;
}

// The studio the surface reflects: a softbox above, a window on the left, a
// fill on the right and the pale floor below.
fn env(r_in: vec3<f32>) -> f32 {
    var R = r_in;
    R.x = R.x + u_data.data[3].z * 0.18;
    let up = -R.y;
    let lv = u_data.data[8];
    var e = 0.015;
    e = e + smoothstep(0.3, 0.5, up) * smoothstep(0.75, 0.55, up) * smoothstep(0.8, 0.5, abs(R.x)) * 2.6 * lv.x;
    e = e + smoothstep(0.8, 0.95, up) * 0.6 * lv.x;
    e = e + smoothstep(-0.5, -0.7, R.x) * smoothstep(0.5, 0.25, abs(R.y)) * 1.1 * lv.y;
    e = e + smoothstep(0.6, 0.85, R.x) * smoothstep(0.6, 0.2, abs(R.y)) * 0.5 * lv.z;
    e = e + smoothstep(0.25, 0.7, R.y) * 1.25 * lv.w;
    // the room behind the viewer, a little brighter overhead
    e = e + smoothstep(0.5, 1.0, R.z) * (0.9 - 0.3 * R.y) * u_data.data[9].y;
    return e;
}

fn decode(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dpr = u_data.data[0].z;
    let p = in.clip_pos.xy / dpr;
    if (u_data.data[5].x > 2.5) {
        // the backdrop, as CSS draws linear-gradient(180deg, ...): mixed in sRGB
        let t = clamp(p.y / u_data.data[0].y, 0.0, 1.0);
        var bg = mix(u_data.data[7].rgb, u_data.data[8].rgb, clamp((t - 0.55) / 0.45, 0.0, 1.0));
        if (t < 0.55) { bg = mix(u_data.data[6].rgb, u_data.data[7].rgb, t / 0.55); }
        return vec4<f32>(framebuffer_encode(decode(bg)), 1.0);
    }
    let f = field(p);
    let d = f.x;
    let dd = d * dpr;
    // A pixel is partly covered up to half its widest footprint (45 degrees)
    // from the edge.
    let band = 0.70711;
    var aa = 0.0;
    var g = vec2<f32>(0.0);
    if (dd < band) {
        // The field's gradient: the surface normal's direction, and how wide
        // the pixel is across the edge here. Where the nearest thing is
        // thinner than about a pixel the forward step can cross its middle,
        // where the distance has a crease; per axis the larger of the forward
        // and backward difference is safe from that.
        let e = 0.5;
        let fx = field(p + vec2<f32>(e, 0.0)).x;
        let fy = field(p + vec2<f32>(0.0, e)).x;
        g = vec2<f32>(fx - d, fy - d);
        var across = abs(g);
        let radius = f.z * dpr;
        if (radius < 1.0) {
            across = max(across, abs(vec2<f32>(d - field(p - vec2<f32>(e, 0.0)).x, d - field(p - vec2<f32>(0.0, e)).x)));
        }
        let w = sdf2_footprint(across);
        if (radius < 4.0) {
            // thin: the share of the pixel the strand really covers
            aa = sdf_coverage_stroke(dd + radius, radius, w);
        } else {
            aa = sdf_coverage_edge(dd, w);
        }
        if (length(g) > 1e-6) { g = normalize(g); } else { g = vec2<f32>(0.0); }
    }

    // soft contact shadow, cast down and a little right
    var sh = 0.0;
    if (aa < 1.0) {
        let ds = field(p - vec2<f32>(2.0, 9.0)).x;
        sh = (1.0 - smoothstep(-4.0, 16.0, ds)) * u_data.data[9].x;
    }
    let alpha = aa + (1.0 - aa) * sh;
    if (alpha <= 0.0) { discard; }

    var col = vec3<f32>(0.0);
    if (aa > 0.0) {
        // round profile: a circle of radius = local thickness. The slope comes
        // from the profile analytically, the direction from the field.
        let B = max(f.y, 0.8);
        let s = clamp(-d, 0.0, B);
        let z = sqrt(max(1e-3, s * (2.0 * B - s)));
        let slope = min((B - s) / z, 6.0);
        let n = normalize(vec3<f32>(g * slope, 1.0));
        let R = reflect(vec3<f32>(0.0, 0.0, -1.0), n);
        let base = u_data.data[6];
        let tint = u_data.data[7];
        let fres = base.w + (1.0 - base.w) * pow(1.0 - max(n.z, 0.0), 3.0);
        col = base.rgb + min(env(R) * fres, 1.0) * tint.rgb;
        col = col + pow(max(dot(R, normalize(vec3<f32>(-0.35, -0.75, 0.56))), 0.0), 70.0) * 1.2 * tint.w;
        col = col + pow(max(dot(R, normalize(vec3<f32>(0.55, -0.35, 0.76))), 0.0), 140.0) * 0.55 * tint.w;
    }
    // premultiplied goo over a black shadow, as straight alpha
    let straight = clamp(col * aa / alpha, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(framebuffer_encode(decode(straight)), alpha);
}
