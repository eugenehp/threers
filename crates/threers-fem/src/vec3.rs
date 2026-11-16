//! `[f64; 3]` arithmetic. The solvers work in double precision throughout;
//! `threers::Vector3` is `f32`, the renderer's precision, and only appears
//! at the API edges.

pub type V3 = [f64; 3];

#[inline]
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn norm2(a: V3) -> f64 {
    dot(a, a)
}

#[inline]
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
pub fn dist(a: V3, b: V3) -> f64 {
    norm(sub(a, b))
}

#[inline]
pub fn normalize(a: V3) -> V3 {
    let l = norm(a);
    if l > 0.0 { scale(a, 1.0 / l) } else { a }
}

#[inline]
pub fn lerp(a: V3, b: V3, t: f64) -> V3 {
    add(a, scale(sub(b, a), t))
}

#[inline]
pub fn min3(a: V3, b: V3) -> V3 {
    [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])]
}

#[inline]
pub fn max3(a: V3, b: V3) -> V3 {
    [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])]
}

pub fn bounds(points: &[V3]) -> (V3, V3) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        lo = min3(lo, *p);
        hi = max3(hi, *p);
    }
    (lo, hi)
}

/// Six times the signed volume of tetrahedron `abcd`: positive when `d` is on
/// the side of `abc` that its counter-clockwise winding faces away from.
#[inline]
pub fn tet_volume6(a: V3, b: V3, c: V3, d: V3) -> f64 {
    dot(sub(b, a), cross(sub(c, a), sub(d, a)))
}
