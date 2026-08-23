//! Exact geometric predicates — **M2 foundation**.
//!
//! `orient2d` returns a value with the *exactly correct sign* of the 2D
//! orientation determinant, even when naive f64 rounds to the wrong sign near
//! degeneracy. From-scratch adaptive-precision arithmetic (Shewchuk 1997):
//! a fast floating-point path with an error filter, falling back to exact
//! expansion arithmetic only when the sign is in doubt. Pure Rust, no deps.
//!
//! This is the precision layer the degenerate cases (coplanar/collinear) need;
//! the coplanar-face *logic* that uses it is the next M2 step.

const EPSILON: f64 = 1.110_223_024_625_156_5e-16; // 2^-53
const CCWERRBOUND_A: f64 = (3.0 + 16.0 * EPSILON) * EPSILON;
const CCWERRBOUND_B: f64 = (2.0 + 12.0 * EPSILON) * EPSILON;
const CCWERRBOUND_C: f64 = (9.0 + 64.0 * EPSILON) * EPSILON * EPSILON;
const RESULTERRBOUND: f64 = (3.0 + 8.0 * EPSILON) * EPSILON;
const O3DERRBOUND_A: f64 = (7.0 + 56.0 * EPSILON) * EPSILON;

// --- Error-free transformations ---

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let bv = x - a;
    let y = (a - (x - bv)) + (b - bv);
    (x, y)
}

#[inline]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    // Requires |a| >= |b|.
    let x = a + b;
    let y = b - (x - a);
    (x, y)
}

#[inline]
fn two_diff_tail(a: f64, b: f64, x: f64) -> f64 {
    let bv = a - x;
    let av = x + bv;
    let br = bv - b;
    let ar = a - av;
    ar + br
}

#[inline]
fn two_diff(a: f64, b: f64) -> (f64, f64) {
    let x = a - b;
    (x, two_diff_tail(a, b, x))
}

#[inline]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    let y = a.mul_add(b, -x); // exact product tail via FMA
    (x, y)
}

/// `(a1,a0) - b` as a 3-component expansion `(x2, x1, x0)`.
#[inline]
fn two_one_diff(a1: f64, a0: f64, b: f64) -> (f64, f64, f64) {
    let (i, x0) = two_diff(a0, b);
    let (x2, x1) = two_sum(a1, i);
    (x2, x1, x0)
}

/// `(a1,a0) - (b1,b0)` as a 4-component expansion `[x0, x1, x2, x3]`.
#[inline]
fn two_two_diff(a1: f64, a0: f64, b1: f64, b0: f64) -> [f64; 4] {
    let (j, z, x0) = two_one_diff(a1, a0, b0);
    let (x3, x2, x1) = two_one_diff(j, z, b1);
    [x0, x1, x2, x3]
}

/// Merge two non-overlapping increasing expansions into one, dropping zeros.
fn fast_expansion_sum_zeroelim(e: &[f64], f: &[f64]) -> Vec<f64> {
    let (elen, flen) = (e.len(), f.len());
    let mut h = Vec::with_capacity(elen + flen);
    let (mut ei, mut fi) = (0usize, 0usize);
    let mut enow = e[0];
    let mut fnow = f[0];
    let mut q;
    if (fnow > enow) == (fnow > -enow) {
        q = enow;
        ei += 1;
        enow = if ei < elen { e[ei] } else { 0.0 };
    } else {
        q = fnow;
        fi += 1;
        fnow = if fi < flen { f[fi] } else { 0.0 };
    }
    if ei < elen && fi < flen {
        let (qn, hh) = if (fnow > enow) == (fnow > -enow) {
            let r = fast_two_sum(enow, q);
            ei += 1;
            enow = if ei < elen { e[ei] } else { 0.0 };
            r
        } else {
            let r = fast_two_sum(fnow, q);
            fi += 1;
            fnow = if fi < flen { f[fi] } else { 0.0 };
            r
        };
        q = qn;
        if hh != 0.0 {
            h.push(hh);
        }
        while ei < elen && fi < flen {
            let (qn, hh) = if (fnow > enow) == (fnow > -enow) {
                let r = two_sum(q, enow);
                ei += 1;
                enow = if ei < elen { e[ei] } else { 0.0 };
                r
            } else {
                let r = two_sum(q, fnow);
                fi += 1;
                fnow = if fi < flen { f[fi] } else { 0.0 };
                r
            };
            q = qn;
            if hh != 0.0 {
                h.push(hh);
            }
        }
    }
    while ei < elen {
        let (qn, hh) = two_sum(q, enow);
        ei += 1;
        enow = if ei < elen { e[ei] } else { 0.0 };
        q = qn;
        if hh != 0.0 {
            h.push(hh);
        }
    }
    while fi < flen {
        let (qn, hh) = two_sum(q, fnow);
        fi += 1;
        fnow = if fi < flen { f[fi] } else { 0.0 };
        q = qn;
        if hh != 0.0 {
            h.push(hh);
        }
    }
    if q != 0.0 || h.is_empty() {
        h.push(q);
    }
    h
}

/// Multiply an expansion by a scalar, dropping zeros.
fn scale_expansion_zeroelim(e: &[f64], b: f64) -> Vec<f64> {
    let mut h = Vec::with_capacity(e.len() * 2);
    let (mut q, hh) = two_product(e[0], b);
    if hh != 0.0 {
        h.push(hh);
    }
    for &ei in &e[1..] {
        let (product1, product0) = two_product(ei, b);
        let (sum, hh) = two_sum(q, product0);
        if hh != 0.0 {
            h.push(hh);
        }
        let (qn, hh) = fast_two_sum(product1, sum);
        q = qn;
        if hh != 0.0 {
            h.push(hh);
        }
    }
    if q != 0.0 || h.is_empty() {
        h.push(q);
    }
    h
}

fn negate(e: &[f64]) -> Vec<f64> {
    e.iter().map(|x| -x).collect()
}

/// xy-minor `p·q` for two points: `px·qy − qx·py` as a 4-component expansion.
fn xy_minor(p: [f64; 3], q: [f64; 3]) -> [f64; 4] {
    let (h1, h0) = two_product(p[0], q[1]);
    let (g1, g0) = two_product(q[0], p[1]);
    two_two_diff(h1, h0, g1, g0)
}

/// Exact 3D orientation via the 4×4 (homogeneous) determinant, expanded in the
/// xy-minors of point pairs scaled by z. Always exact.
fn orient3d_exact(pa: [f64; 3], pb: [f64; 3], pc: [f64; 3], pd: [f64; 3]) -> f64 {
    let ab = xy_minor(pa, pb);
    let bc = xy_minor(pb, pc);
    let cd = xy_minor(pc, pd);
    let da = xy_minor(pd, pa);
    let ac = xy_minor(pa, pc);
    let bd = xy_minor(pb, pd);

    // Signed xy-areas (×2) of the four opposite faces.
    let sum3 = |x: &[f64], y: &[f64], z: &[f64]| {
        fast_expansion_sum_zeroelim(&fast_expansion_sum_zeroelim(x, y), z)
    };
    let bcd = sum3(&bc, &cd, &negate(&bd)); // area(b,c,d)
    let acd = sum3(&ac, &cd, &da); // area(a,c,d)
    let abd = sum3(&ab, &bd, &da); // area(a,b,d)
    let abc = sum3(&ab, &bc, &negate(&ac)); // area(a,b,c)

    // det = az·[bcd] − bz·[acd] + cz·[abd] − dz·[abc]
    let d1 = scale_expansion_zeroelim(&bcd, pa[2]);
    let d2 = scale_expansion_zeroelim(&acd, -pb[2]);
    let d3 = scale_expansion_zeroelim(&abd, pc[2]);
    let d4 = scale_expansion_zeroelim(&abc, -pd[2]);
    let det = fast_expansion_sum_zeroelim(
        &fast_expansion_sum_zeroelim(&d1, &d2),
        &fast_expansion_sum_zeroelim(&d3, &d4),
    );
    *det.last().unwrap()
}

/// Exactly-signed 3D orientation of `(pa, pb, pc, pd)`: `> 0` if `pd` is below
/// the plane `pa,pb,pc` (CCW from above), `< 0` above, `0` exactly coplanar.
pub fn orient3d(pa: [f64; 3], pb: [f64; 3], pc: [f64; 3], pd: [f64; 3]) -> f64 {
    let (adx, ady, adz) = (pa[0] - pd[0], pa[1] - pd[1], pa[2] - pd[2]);
    let (bdx, bdy, bdz) = (pb[0] - pd[0], pb[1] - pd[1], pb[2] - pd[2]);
    let (cdx, cdy, cdz) = (pc[0] - pd[0], pc[1] - pd[1], pc[2] - pd[2]);

    let bdxcdy = bdx * cdy;
    let cdxbdy = cdx * bdy;
    let cdxady = cdx * ady;
    let adxcdy = adx * cdy;
    let adxbdy = adx * bdy;
    let bdxady = bdx * ady;

    let det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);
    let permanent = (bdxcdy.abs() + cdxbdy.abs()) * adz.abs()
        + (cdxady.abs() + adxcdy.abs()) * bdz.abs()
        + (adxbdy.abs() + bdxady.abs()) * cdz.abs();
    let errbound = O3DERRBOUND_A * permanent;
    if det > errbound || -det > errbound {
        return det;
    }
    orient3d_exact(pa, pb, pc, pd)
}

/// Adaptive exact stage — only reached when the fast filter is inconclusive.
fn orient2d_adapt(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2], detsum: f64) -> f64 {
    let acx = pa[0] - pc[0];
    let bcx = pb[0] - pc[0];
    let acy = pa[1] - pc[1];
    let bcy = pb[1] - pc[1];

    let (detleft, detlefttail) = two_product(acx, bcy);
    let (detright, detrighttail) = two_product(acy, bcx);
    let b = two_two_diff(detleft, detlefttail, detright, detrighttail);
    let mut det: f64 = b.iter().sum();
    let errbound = CCWERRBOUND_B * detsum;
    if det >= errbound || -det >= errbound {
        return det;
    }

    let acxtail = two_diff_tail(pa[0], pc[0], acx);
    let bcxtail = two_diff_tail(pb[0], pc[0], bcx);
    let acytail = two_diff_tail(pa[1], pc[1], acy);
    let bcytail = two_diff_tail(pb[1], pc[1], bcy);
    if acxtail == 0.0 && acytail == 0.0 && bcxtail == 0.0 && bcytail == 0.0 {
        return det;
    }

    let errbound = CCWERRBOUND_C * detsum + RESULTERRBOUND * det.abs();
    det += (acx * bcytail + bcy * acxtail) - (acy * bcxtail + bcx * acytail);
    if det >= errbound || -det >= errbound {
        return det;
    }

    let (s1, s0) = two_product(acxtail, bcy);
    let (t1, t0) = two_product(acytail, bcx);
    let u = two_two_diff(s1, s0, t1, t0);
    let c1 = fast_expansion_sum_zeroelim(&b, &u);

    let (s1, s0) = two_product(acx, bcytail);
    let (t1, t0) = two_product(acy, bcxtail);
    let u = two_two_diff(s1, s0, t1, t0);
    let c2 = fast_expansion_sum_zeroelim(&c1, &u);

    let (s1, s0) = two_product(acxtail, bcytail);
    let (t1, t0) = two_product(acytail, bcxtail);
    let u = two_two_diff(s1, s0, t1, t0);
    let d = fast_expansion_sum_zeroelim(&c2, &u);

    *d.last().unwrap()
}

/// Exactly-signed 2D orientation of `(pa, pb, pc)`: `> 0` counter-clockwise,
/// `< 0` clockwise, `0` exactly collinear.
pub fn orient2d(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2]) -> f64 {
    let detleft = (pa[0] - pc[0]) * (pb[1] - pc[1]);
    let detright = (pa[1] - pc[1]) * (pb[0] - pc[0]);
    let det = detleft - detright;

    let detsum = if detleft > 0.0 {
        if detright <= 0.0 {
            return det;
        }
        detleft + detright
    } else if detleft < 0.0 {
        if detright >= 0.0 {
            return det;
        }
        -detleft - detright
    } else {
        return det;
    };

    let errbound = CCWERRBOUND_A * detsum;
    if det >= errbound || -det >= errbound {
        return det;
    }
    orient2d_adapt(pa, pb, pc, detsum)
}

/// Exactly coplanar? True iff all three vertices of `b` lie on `a`'s plane.
/// (Assumes `a` is a non-degenerate triangle.) The entry point for the coplanar
/// boolean rules — the next M2 step.
pub fn coplanar(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> bool {
    orient3d(a[0], a[1], a[2], b[0]) == 0.0
        && orient3d(a[0], a[1], a[2], b[1]) == 0.0
        && orient3d(a[0], a[1], a[2], b[2]) == 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sgn(x: f64) -> i32 {
        if x > 0.0 {
            1
        } else if x < 0.0 {
            -1
        } else {
            0
        }
    }
    fn exact_i128(a: [i64; 2], b: [i64; 2], c: [i64; 2]) -> i32 {
        let acx = a[0] as i128 - c[0] as i128;
        let acy = a[1] as i128 - c[1] as i128;
        let bcx = b[0] as i128 - c[0] as i128;
        let bcy = b[1] as i128 - c[1] as i128;
        (acx * bcy - acy * bcx).signum() as i32
    }

    fn exact3d_i128(a: [i64; 3], b: [i64; 3], c: [i64; 3], d: [i64; 3]) -> i32 {
        let v = |p: [i64; 3]| {
            [
                p[0] as i128 - d[0] as i128,
                p[1] as i128 - d[1] as i128,
                p[2] as i128 - d[2] as i128,
            ]
        };
        let (ad, bd, cd) = (v(a), v(b), v(c));
        let det = ad[0] * (bd[1] * cd[2] - bd[2] * cd[1]) - ad[1] * (bd[0] * cd[2] - bd[2] * cd[0])
            + ad[2] * (bd[0] * cd[1] - bd[1] * cd[0]);
        det.signum() as i32
    }

    #[test]
    fn orient3d_is_exact() {
        let mut seed = 0x0bad_c0de_1234_5678u64;
        let mut rnd = |m: i64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as i64).rem_euclid(2 * m + 1) - m
        };
        let m = 1i64 << 26;
        let f = |p: [i64; 3]| [p[0] as f64, p[1] as f64, p[2] as f64];

        // Exactly-coplanar d (in the plane of a,b,c) → exact ⇒ 0.
        let mut naive_wrong = 0;
        for _ in 0..3000 {
            let a = [rnd(m), rnd(m), rnd(m)];
            let b = [rnd(m), rnd(m), rnd(m)];
            let c = [rnd(m), rnd(m), rnd(m)];
            let (s, t) = (rnd(3), rnd(3));
            let d = [
                a[0] + s * (b[0] - a[0]) + t * (c[0] - a[0]),
                a[1] + s * (b[1] - a[1]) + t * (c[1] - a[1]),
                a[2] + s * (b[2] - a[2]) + t * (c[2] - a[2]),
            ];
            assert_eq!(
                sgn(orient3d(f(a), f(b), f(c), f(d))),
                0,
                "coplanar must be exactly 0"
            );
            let (af, bf, cf, df) = (f(a), f(b), f(c), f(d));
            let (adx, ady, adz) = (af[0] - df[0], af[1] - df[1], af[2] - df[2]);
            let (bdx, bdy, bdz) = (bf[0] - df[0], bf[1] - df[1], bf[2] - df[2]);
            let (cdx, cdy, cdz) = (cf[0] - df[0], cf[1] - df[1], cf[2] - df[2]);
            let naive = adz * (bdx * cdy - cdx * bdy)
                + bdz * (cdx * ady - adx * cdy)
                + cdz * (adx * bdy - bdx * ady);
            if sgn(naive) != 0 {
                naive_wrong += 1;
            }
        }
        assert!(
            naive_wrong > 50,
            "naive f64 should misjudge coplanar cases; got {naive_wrong}"
        );

        // Random points: exact sign must match the i128 determinant.
        for _ in 0..4000 {
            let a = [rnd(m), rnd(m), rnd(m)];
            let b = [rnd(m), rnd(m), rnd(m)];
            let c = [rnd(m), rnd(m), rnd(m)];
            let d = [rnd(m), rnd(m), rnd(m)];
            assert_eq!(
                sgn(orient3d(f(a), f(b), f(c), f(d))),
                exact3d_i128(a, b, c, d)
            );
        }
    }

    #[test]
    fn basic_orientation() {
        assert!(orient2d([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]) > 0.0); // CCW
        assert!(orient2d([0.0, 0.0], [0.0, 1.0], [1.0, 0.0]) < 0.0); // CW
        assert_eq!(orient2d([0.0, 0.0], [1.0, 1.0], [2.0, 2.0]), 0.0); // collinear
    }

    #[test]
    fn coplanar_detection() {
        // Two triangles in z = 0 (e.g. coincident box faces) → coplanar.
        let a = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let b = [[1.0, 1.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]];
        assert!(coplanar(&a, &b));
        // Tilt one vertex off the plane → not coplanar.
        let c = [[1.0, 1.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.001]];
        assert!(!coplanar(&a, &c));
    }

    #[test]
    fn exact_where_naive_fails() {
        // Large integer coords → determinant products exceed 2^53, so naive f64
        // rounds. The exact predicate must still match the i128 determinant, and
        // must return exactly 0 on truly-collinear points where naive does not.
        let mut seed = 0x1234_9abc_dead_beefu64;
        let mut rnd = |m: i64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as i64).rem_euclid(2 * m + 1) - m
        };
        let m = 1i64 << 30;
        let big = 1i64 << 28;
        let f = |p: [i64; 2]| [p[0] as f64, p[1] as f64];

        // Near-collinear points with a *tiny* true determinant (= −s) but huge
        // (~2^60) products, so naive f64 rounds the sign away. Construction:
        // b = a + (dx, dy); c = a + (k·dx + 1, k·dy + 1) ⇒ det = dx − dy = −s.
        let mut naive_wrong = 0;
        for _ in 0..4000 {
            let a = [rnd(m), rnd(m)];
            let dx = rnd(big).abs() + big; // ~2^28
            let s = rnd(5); // small; can be 0 (exactly collinear)
            let dy = dx + s;
            let k = rnd(3).abs() + 1;
            let b = [a[0] + dx, a[1] + dy];
            let c = [a[0] + k * dx + 1, a[1] + k * dy + 1];
            let want = exact_i128(a, b, c); // = sign(-s)
            let (af, bf, cf) = (f(a), f(b), f(c));
            assert_eq!(
                sgn(orient2d(af, bf, cf)),
                want,
                "exact must match i128 (det={})",
                -s
            );
            let naive = (af[0] - cf[0]) * (bf[1] - cf[1]) - (af[1] - cf[1]) * (bf[0] - cf[0]);
            if sgn(naive) != want {
                naive_wrong += 1;
            }
        }
        assert!(
            naive_wrong > 100,
            "naive f64 should misjudge many; got {naive_wrong}"
        );

        // Random points: exact sign must match the i128 determinant.
        for _ in 0..4000 {
            let a = [rnd(m), rnd(m)];
            let b = [rnd(m), rnd(m)];
            let c = [rnd(m), rnd(m)];
            assert_eq!(sgn(orient2d(f(a), f(b), f(c))), exact_i128(a, b, c));
        }
    }
}
