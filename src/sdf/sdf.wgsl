// threers::sdf — signed distance primitives and operators. Negative inside.
// The same functions as the Rust `threers::sdf` module, tested against it.

// ---- operators (2D or 3D) ----

fn sdf_union(a: f32, b: f32) -> f32 { return min(a, b); }
fn sdf_intersection(a: f32, b: f32) -> f32 { return max(a, b); }
fn sdf_subtraction(a: f32, b: f32) -> f32 { return max(a, -b); }

// Smooth union, and how much of the result is `a` (1) versus `b` (0) in .y.
// k <= 0 is a hard union.
fn sdf_smooth_union_blend(a: f32, b: f32, k: f32) -> vec2<f32> {
    if (k <= 0.0) {
        return select(vec2<f32>(b, 0.0), vec2<f32>(a, 1.0), a <= b);
    }
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return vec2<f32>(mix(b, a, h) - k * h * (1.0 - h), h);
}
fn sdf_smooth_union(a: f32, b: f32, k: f32) -> f32 { return sdf_smooth_union_blend(a, b, k).x; }
fn sdf_smooth_intersection(a: f32, b: f32, k: f32) -> f32 { return -sdf_smooth_union(-a, -b, k); }
fn sdf_smooth_subtraction(a: f32, b: f32, k: f32) -> f32 { return -sdf_smooth_union(-a, b, k); }
fn sdf_round(d: f32, r: f32) -> f32 { return d - r; }
fn sdf_onion(d: f32, t: f32) -> f32 { return abs(d) - t; }
fn sdf_coverage(d: f32) -> f32 { return clamp(0.5 - d, 0.0, 1.0); }
// How wide a pixel is across an edge with field gradient g: 1 along a row or
// column, sqrt(2) at 45 degrees.
fn sdf2_footprint(g: vec2<f32>) -> f32 {
    let l = length(g);
    if (l <= 1e-12) { return 1.0; }
    return (abs(g.x) + abs(g.y)) / l;
}
fn sdf_coverage_edge(d: f32, footprint: f32) -> f32 {
    return clamp(0.5 - d / max(footprint, 1e-6), 0.0, 1.0);
}
// Box-filter coverage of a stroke half_width either side of a centre line,
// sampled u from it: right for strokes thinner than a pixel too.
fn sdf_coverage_stroke(u_in: f32, half_width: f32, footprint: f32) -> f32 {
    let u = abs(u_in);
    let h = max(half_width, 0.0);
    let w = max(footprint, 1e-6);
    let lo = max(u - 0.5 * w, -h);
    let hi = min(u + 0.5 * w, h);
    return clamp((hi - lo) / w, 0.0, 1.0);
}

// ---- 2D ----

fn sdf2_circle(p: vec2<f32>, r: f32) -> f32 { return length(p) - r; }

fn sdf2_rect(p: vec2<f32>, half: vec2<f32>) -> f32 {
    let q = abs(p) - half;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
}

fn sdf2_round_rect(p: vec2<f32>, half: vec2<f32>, r: f32) -> f32 {
    let rr = min(max(r, 0.0), max(min(half.x, half.y), 0.0));
    return sdf2_rect(p, half - vec2<f32>(rr)) - rr;
}

fn sdf2_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let v = b - a;
    let w = p - a;
    let len2 = dot(v, v);
    let t = select(0.0, clamp(dot(w, v) / len2, 0.0, 1.0), len2 > 0.0);
    return length(w - v * t);
}

fn sdf2_capsule(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    return sdf2_segment(p, a, b) - r;
}

// Tapered capsule (the hull of two discs): distance in .x, and the radius at
// the nearest point along the axis in .y.
fn sdf2_tapered_capsule(p_in: vec2<f32>, a: vec2<f32>, b: vec2<f32>, ra: f32, rb: f32) -> vec2<f32> {
    let pb = b - a;
    let p = p_in - a;
    let h = dot(pb, pb);
    let t = clamp(dot(p, pb) / (h + 1e-12), 0.0, 1.0);
    let r = mix(ra, rb, t);
    let bb = ra - rb;
    if (h <= bb * bb) {
        if (ra > rb) { return vec2<f32>(length(p) - ra, ra); }
        return vec2<f32>(length(p - pb) - rb, rb);
    }
    let qx = abs(p.x * pb.y - p.y * pb.x) / h;
    let qy = dot(p, pb) / h;
    let cx = sqrt(h - bb * bb);
    let k = cx * qy - bb * qx;
    let n = qx * qx + qy * qy;
    var d: f32;
    if (k < 0.0) {
        d = sqrt(h * n) - ra;
    } else if (k > cx) {
        d = sqrt(h * (n + 1.0 - 2.0 * qy)) - rb;
    } else {
        d = cx * qx + bb * qy - ra;
    }
    return vec2<f32>(d, r);
}

// ---- 3D ----

fn sdf3_sphere(p: vec3<f32>, r: f32) -> f32 { return length(p) - r; }

fn sdf3_cuboid(p: vec3<f32>, half: vec3<f32>) -> f32 {
    let q = abs(p) - half;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdf3_round_cuboid(p: vec3<f32>, half: vec3<f32>, r: f32) -> f32 {
    let rr = min(max(r, 0.0), max(min(half.x, min(half.y, half.z)), 0.0));
    return sdf3_cuboid(p, half - vec3<f32>(rr)) - rr;
}

fn sdf3_capsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let ba = b - a;
    let pa = p - a;
    let len2 = dot(ba, ba);
    let t = select(0.0, clamp(dot(pa, ba) / len2, 0.0, 1.0), len2 > 0.0);
    return length(pa - ba * t) - r;
}

fn sdf3_tapered_capsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, ra: f32, rb: f32) -> f32 {
    let ba = b - a;
    let l2 = dot(ba, ba);
    let rr = ra - rb;
    let a2 = l2 - rr * rr;
    if (a2 <= 0.0) {
        if (ra > rb) { return length(p - a) - ra; }
        return length(p - b) - rb;
    }
    let il2 = 1.0 / l2;
    let pa = p - a;
    let y = dot(pa, ba);
    let z = y - l2;
    let xv = pa * l2 - ba * y;
    let x2 = dot(xv, xv);
    let y2 = y * y * l2;
    let z2 = z * z * l2;
    let k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k) { return sqrt(x2 + z2) * il2 - rb; }
    if (sign(y) * a2 * y2 < k) { return sqrt(x2 + y2) * il2 - ra; }
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - ra;
}

fn sdf3_torus(p: vec3<f32>, major: f32, minor: f32) -> f32 {
    let q = vec2<f32>(length(p.xz) - major, p.y);
    return length(q) - minor;
}

fn sdf3_cylinder(p: vec3<f32>, half_height: f32, r: f32) -> f32 {
    let d = vec2<f32>(length(p.xz) - r, abs(p.y) - half_height);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn sdf3_plane(p: vec3<f32>, normal: vec3<f32>, offset: f32) -> f32 {
    return dot(p, normal) + offset;
}
