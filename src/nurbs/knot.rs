//! Knot operations: insertion, refinement, Bézier decomposition, degree
//! elevation, and making two curves compatible.
//!
//! These are the algebra behind everything structural. Splitting a surface at an
//! intersection curve (Stage 3), lofting between curves of different degree, and
//! writing a STEP `B_SPLINE_SURFACE_WITH_KNOTS` all reduce to some combination
//! of them.
//!
//! Every operation here is **shape-preserving**: the curve's point set is
//! unchanged, only its representation moves. The tests assert exactly that.

use super::curve::NurbsCurve;
use super::surface::NurbsSurface;
use super::{NurbsError, V4};

fn lerp4(a: V4, b: V4, t: f64) -> V4 {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

/// Multiplicity of `u` in `knots`, comparing exactly.
///
/// Exact comparison is correct here and a tolerance would be wrong: knot values
/// produced by insertion are *the same f64* as the value inserted, and two knots
/// that differ in the last bit are genuinely distinct knots defining a very
/// short span. Fuzzing them together would silently change the curve.
pub fn multiplicity(knots: &[f64], u: f64) -> usize {
    knots.iter().filter(|&&k| k == u).count()
}

/// The distinct knot values in `knots`, in increasing order, with their
/// multiplicities.
pub fn distinct_knots(knots: &[f64]) -> Vec<(f64, usize)> {
    let mut out: Vec<(f64, usize)> = Vec::new();
    for &k in knots {
        match out.last_mut() {
            Some((v, m)) if *v == k => *m += 1,
            _ => out.push((k, 1)),
        }
    }
    out
}

/// Insert knot `u` into `curve`, `times` times — Boehm's algorithm (A5.1).
///
/// The insertion count is clamped so the resulting multiplicity never exceeds
/// the degree: multiplicity `p` already splits the curve there (the control
/// point interpolates), and multiplicity `p + 1` would disconnect it. Callers
/// asking for more get the maximum legal amount rather than a broken curve.
pub fn insert_knot(curve: &NurbsCurve, u: f64, times: usize) -> NurbsCurve {
    let p = curve.degree();
    let up = curve.knots();
    let pw = curve.homogeneous();
    let np = pw.len() - 1;
    let mp = np + p + 1;

    let (u0, u1) = curve.domain();
    debug_assert!(u >= u0 && u <= u1, "knot {u} outside domain [{u0}, {u1}]");

    let s = multiplicity(up, u);
    let r = times.min(p.saturating_sub(s));
    if r == 0 {
        return curve.clone();
    }
    let k = super::basis::find_span(p, pw.len(), up, u);

    // New knot vector: the old one with `r` copies of u spliced in after k.
    let mut uq = Vec::with_capacity(mp + r + 1);
    uq.extend_from_slice(&up[..=k]);
    uq.extend(std::iter::repeat_n(u, r));
    uq.extend_from_slice(&up[k + 1..=mp]);

    // Control points: the ends are carried over untouched, the middle rebuilt.
    let mut qw = vec![[0.0f64; 4]; np + r + 1];
    qw[..=(k - p)].copy_from_slice(&pw[..=(k - p)]);
    qw[(k - s + r)..=(np + r)].copy_from_slice(&pw[(k - s)..=np]);

    let mut rw: Vec<V4> = (0..=(p - s)).map(|i| pw[k - p + i]).collect();

    for j in 1..=r {
        let l = k - p + j;
        for i in 0..=(p - j - s) {
            let denom = up[i + k + 1] - up[l + i];
            // Zero denominators mean a knot span of zero length, which the
            // multiplicity clamp above already excluded.
            let alpha = if denom.abs() < 1e-300 {
                0.0
            } else {
                (u - up[l + i]) / denom
            };
            rw[i] = lerp4(rw[i], rw[i + 1], alpha);
        }
        qw[l] = rw[0];
        qw[k + r - j - s] = rw[p - j - s];
    }

    // Whatever `rw` still holds between the two rebuilt ends. Empty when the
    // insertion saturated the multiplicity (r == p - s), so guard the range —
    // slicing backwards panics where the equivalent loop simply would not run.
    let l = k - p + r;
    if k > s + l + 1 {
        let end = k - s;
        qw[(l + 1)..end].copy_from_slice(&rw[1..(end - l)]);
    }

    NurbsCurve::from_homogeneous_unchecked(p, uq, qw)
}

/// Insert every knot in `new_knots` (with multiplicity), returning the refined
/// curve.
///
/// Implemented as repeated [`insert_knot`] rather than A5.4's simultaneous
/// refinement. That is `O(n · r)` where A5.4 is `O(n + r)`, which matters for
/// bulk refinement of large surfaces but not for the handful of knots the
/// current callers insert — and repeated insertion is correct by construction
/// given `insert_knot` is, which is the trade this file wants to make until a
/// profile says otherwise.
pub fn refine(curve: &NurbsCurve, new_knots: &[f64]) -> NurbsCurve {
    let mut out = curve.clone();
    for &u in new_knots {
        out = insert_knot(&out, u, 1);
    }
    out
}

/// Split `curve` at `u`, returning the two pieces.
///
/// Works by raising the multiplicity at `u` to the degree, at which point the
/// control polygon interpolates the curve there and the two halves can simply
/// be read off. Returns `None` if `u` is at or outside an end of the domain.
pub fn split(curve: &NurbsCurve, u: f64) -> Option<(NurbsCurve, NurbsCurve)> {
    let (u0, u1) = curve.domain();
    if u <= u0 || u >= u1 {
        return None;
    }
    let p = curve.degree();
    let s = multiplicity(curve.knots(), u);
    let full = insert_knot(curve, u, p.saturating_sub(s));

    let k = super::basis::find_span(p, full.n_control(), full.knots(), u);
    // After raising to multiplicity p, control point `k - p` is the split point.
    let cut = k - p;

    let left_cw = full.homogeneous()[..=cut].to_vec();
    let right_cw = full.homogeneous()[cut..].to_vec();

    let mut left_knots = full.knots()[..=k].to_vec();
    left_knots.push(u);
    let mut right_knots = vec![u; p + 1];
    right_knots.extend_from_slice(&full.knots()[k + 1..]);

    Some((
        NurbsCurve::from_homogeneous_unchecked(p, left_knots, left_cw),
        NurbsCurve::from_homogeneous_unchecked(p, right_knots, right_cw),
    ))
}

/// Decompose into Bézier segments — every interior knot raised to multiplicity
/// `degree`, then the control points read off in overlapping groups of
/// `degree + 1`.
///
/// Returns the raw homogeneous control points per segment; each segment is a
/// Bézier curve on `[0, 1]`, and consecutive segments share an endpoint.
pub fn bezier_segments(curve: &NurbsCurve) -> Vec<Vec<V4>> {
    let p = curve.degree();
    let (u0, u1) = curve.domain();

    let mut full = curve.clone();
    for (value, mult) in distinct_knots(curve.knots()) {
        if value > u0 && value < u1 && mult < p {
            full = insert_knot(&full, value, p - mult);
        }
    }

    let cw = full.homogeneous();
    let n_seg = (cw.len() - 1) / p;
    (0..n_seg).map(|s| cw[s * p..=s * p + p].to_vec()).collect()
}

/// Elevate the degree by `t`.
///
/// Decomposes to Bézier, applies the Bézier elevation formula
/// `Qᵢ = (i/(p+1))·Pᵢ₋₁ + (1 − i/(p+1))·Pᵢ` to each segment, and reassembles.
///
/// The reassembly leaves interior knots at full multiplicity `p + t`; a
/// [`compact`] pass then removes the ones that carry no shape, bringing them
/// back to `original + t` — which is what A5.9 achieves inline. Splitting it in
/// two costs a pass over the curve and makes both halves independently testable,
/// which for index arithmetic this dense is the better trade.
pub fn elevate_degree(curve: &NurbsCurve, t: usize) -> NurbsCurve {
    if t == 0 {
        return curve.clone();
    }
    let p = curve.degree();
    let pn = p + t;

    let mut segments = bezier_segments(curve);
    for _ in 0..t {
        for seg in segments.iter_mut() {
            let d = seg.len() - 1;
            let mut next = Vec::with_capacity(d + 2);
            next.push(seg[0]);
            for i in 1..=d {
                let a = i as f64 / (d + 1) as f64;
                next.push(lerp4(seg[i], seg[i - 1], a));
            }
            next.push(seg[d]);
            *seg = next;
        }
    }

    // Reassemble: first segment whole, then each subsequent one minus its
    // shared first point.
    let mut cw = segments[0].clone();
    for seg in &segments[1..] {
        cw.extend_from_slice(&seg[1..]);
    }

    // Knot vector: clamped ends, interior breakpoints at multiplicity pn,
    // reusing the original breakpoint *values* so the parameterization is
    // preserved (the curve is the same map, not just the same point set).
    let breaks: Vec<f64> = {
        let (u0, u1) = curve.domain();
        let mut b = vec![u0];
        for (value, _) in distinct_knots(curve.knots()) {
            if value > u0 && value < u1 {
                b.push(value);
            }
        }
        b.push(u1);
        b
    };

    let mut knots = Vec::with_capacity(cw.len() + pn + 1);
    knots.extend(std::iter::repeat_n(breaks[0], pn + 1));
    for &b in &breaks[1..breaks.len() - 1] {
        knots.extend(std::iter::repeat_n(b, pn));
    }
    knots.extend(std::iter::repeat_n(*breaks.last().unwrap(), pn + 1));

    debug_assert_eq!(knots.len(), cw.len() + pn + 1);
    let raw = NurbsCurve::from_homogeneous_unchecked(pn, knots, cw);

    // Tolerance scaled to the control net, so a millimetre curve and a metre
    // curve compact alike.
    let scale = raw
        .homogeneous()
        .iter()
        .flat_map(|c| c[..3].iter().map(|x| x.abs()))
        .fold(1.0f64, f64::max);
    compact(&raw, scale * 1e-12)
}

/// Affinely reparameterize onto `[0, 1]`. The curve's shape is untouched; only
/// the knot values move.
pub fn reparameterize_unit(curve: &NurbsCurve) -> NurbsCurve {
    let (u0, u1) = curve.domain();
    let span = u1 - u0;
    let knots = curve
        .knots()
        .iter()
        .map(|&k| ((k - u0) / span).clamp(0.0, 1.0))
        .collect();
    NurbsCurve::from_homogeneous_unchecked(curve.degree(), knots, curve.homogeneous().to_vec())
}

/// Bring two curves to a common degree and knot vector, so they can be lofted,
/// ruled, or stored in one surface's control grid.
///
/// Both are reparameterized to `[0, 1]`, the lower degree is elevated, and each
/// then receives whatever knots the other has that it lacks.
pub fn make_compatible(
    a: &NurbsCurve,
    b: &NurbsCurve,
) -> Result<(NurbsCurve, NurbsCurve), NurbsError> {
    let mut a = reparameterize_unit(a);
    let mut b = reparameterize_unit(b);

    match a.degree().cmp(&b.degree()) {
        std::cmp::Ordering::Less => a = elevate_degree(&a, b.degree() - a.degree()),
        std::cmp::Ordering::Greater => b = elevate_degree(&b, a.degree() - b.degree()),
        std::cmp::Ordering::Equal => {}
    }

    // Union of knot values, each at the higher of the two multiplicities.
    let da = distinct_knots(a.knots());
    let db = distinct_knots(b.knots());
    let mut add_to_a: Vec<f64> = Vec::new();
    let mut add_to_b: Vec<f64> = Vec::new();

    let mut values: Vec<f64> = da.iter().map(|&(v, _)| v).collect();
    values.extend(db.iter().map(|&(v, _)| v));
    values.sort_by(|x, y| x.partial_cmp(y).unwrap());
    values.dedup();

    for v in values {
        let ma = da.iter().find(|&&(k, _)| k == v).map_or(0, |&(_, m)| m);
        let mb = db.iter().find(|&&(k, _)| k == v).map_or(0, |&(_, m)| m);
        let target = ma.max(mb);
        add_to_a.extend(std::iter::repeat_n(v, target - ma));
        add_to_b.extend(std::iter::repeat_n(v, target - mb));
    }

    let a = refine(&a, &add_to_a);
    let b = refine(&b, &add_to_b);

    if a.degree() != b.degree() || a.n_control() != b.n_control() {
        return Err(NurbsError::Incompatible(
            "degree elevation and knot merging did not converge",
        ));
    }
    Ok((a, b))
}

/// Insert a knot in the surface's u-direction, `times` times.
///
/// Each iso-v column of the control grid is a curve in u sharing one knot
/// vector, so this is [`insert_knot`] applied column-wise.
pub fn insert_knot_surface_u(surface: &NurbsSurface, u: f64, times: usize) -> NurbsSurface {
    let (n_u, n_v) = (surface.n_u(), surface.n_v());
    let p = surface.degree_u();

    let mut new_knots: Option<Vec<f64>> = None;
    let mut columns: Vec<Vec<V4>> = Vec::with_capacity(n_v);

    for j in 0..n_v {
        let col: Vec<V4> = (0..n_u)
            .map(|i| surface.homogeneous()[i * n_v + j])
            .collect();
        let c = NurbsCurve::from_homogeneous_unchecked(p, surface.knots_u().to_vec(), col);
        let inserted = insert_knot(&c, u, times);
        if new_knots.is_none() {
            new_knots = Some(inserted.knots().to_vec());
        }
        columns.push(inserted.homogeneous().to_vec());
    }

    let new_n_u = columns[0].len();
    let mut control = vec![[0.0f64; 4]; new_n_u * n_v];
    for (j, col) in columns.iter().enumerate() {
        for (i, &cp) in col.iter().enumerate() {
            control[i * n_v + j] = cp;
        }
    }

    NurbsSurface::from_homogeneous_unchecked(
        p,
        surface.degree_v(),
        new_knots.expect("at least one column"),
        surface.knots_v().to_vec(),
        new_n_u,
        n_v,
        control,
    )
}

/// Insert a knot in the surface's v-direction, `times` times.
pub fn insert_knot_surface_v(surface: &NurbsSurface, v: f64, times: usize) -> NurbsSurface {
    let (n_u, n_v) = (surface.n_u(), surface.n_v());
    let q = surface.degree_v();

    let mut new_knots: Option<Vec<f64>> = None;
    let mut rows: Vec<Vec<V4>> = Vec::with_capacity(n_u);

    for i in 0..n_u {
        let row = surface.homogeneous()[i * n_v..(i + 1) * n_v].to_vec();
        let c = NurbsCurve::from_homogeneous_unchecked(q, surface.knots_v().to_vec(), row);
        let inserted = insert_knot(&c, v, times);
        if new_knots.is_none() {
            new_knots = Some(inserted.knots().to_vec());
        }
        rows.push(inserted.homogeneous().to_vec());
    }

    let new_n_v = rows[0].len();
    let mut control = Vec::with_capacity(n_u * new_n_v);
    for row in &rows {
        control.extend_from_slice(row);
    }

    NurbsSurface::from_homogeneous_unchecked(
        surface.degree_u(),
        q,
        surface.knots_u().to_vec(),
        new_knots.expect("at least one row"),
        n_u,
        new_n_v,
        control,
    )
}


/// Remove the knot at `u` up to `times` times, keeping the curve within
/// `tolerance` — Piegl & Tiller A5.8.
///
/// The inverse of [`insert_knot`], and the operation that makes representations
/// *minimal* rather than merely correct. A knot is removable when the control
/// points it was computed from can be reconstructed by the same recurrence run
/// backwards; A5.8 does that reconstruction and checks the residual before
/// committing, so a knot that carries real shape is simply not removed.
///
/// Returns the curve and how many removals actually happened. Removing zero is a
/// normal outcome, not a failure.
// A5.8's index shuffling is a set of interlocking counters (`first`/`last`,
// `i`/`j`, `ii`/`jj`) that walk toward each other. Rewriting them as iterators
// reads worse against the reference and makes an off-by-one harder to spot,
// which for arithmetic this dense is the wrong trade — the `fout` term alone
// already cost a debugging round.
#[allow(
    clippy::mut_range_bound,
    clippy::needless_range_loop,
    clippy::explicit_counter_loop
)]
pub fn remove_knot(
    curve: &NurbsCurve,
    u: f64,
    times: usize,
    tolerance: f64,
) -> (NurbsCurve, usize) {
    let p = curve.degree();
    let s = multiplicity(curve.knots(), u);
    if s == 0 || times == 0 {
        return (curve.clone(), 0);
    }
    let (u0, u1) = curve.domain();
    if u <= u0 || u >= u1 {
        return (curve.clone(), 0); // end knots define the clamping
    }

    let mut uv = curve.knots().to_vec();
    let mut pw = curve.homogeneous().to_vec();
    let n = pw.len() - 1;
    let r = super::basis::find_span(p, pw.len(), &uv, u);

    // A5.8 works on a scratch copy `temp` of the affected control points, and
    // only writes back once the residual passes.
    let mut removed = 0usize;
    let mut first = r - p;
    let mut last = r - s;

    for t in 0..times.min(s) {
        // `off = first - 1`; once `first` reaches 0 there is no room left to
        // reconstruct into and no further removal is possible.
        if first == 0 {
            break;
        }
        let off = first - 1;
        let mut temp: Vec<V4> = vec![[0.0; 4]; (last + 1 - off) + 2];
        temp[0] = pw[off];
        temp[last + 1 - off] = pw[last + 1];

        let (mut i, mut j) = (first, last);
        let (mut ii, mut jj) = (1usize, last - off);
        let mut remflag = false;

        while j as isize - i as isize > t as isize {
            let alfi = (u - uv[i]) / (uv[i + p + 1 + t] - uv[i]);
            let alfj = (u - uv[j - t]) / (uv[j + p + 1] - uv[j - t]);
            for c in 0..4 {
                temp[ii][c] = (pw[i][c] - (1.0 - alfi) * temp[ii - 1][c]) / alfi;
                temp[jj][c] = (pw[j][c] - alfj * temp[jj + 1][c]) / (1.0 - alfj);
            }
            i += 1;
            ii += 1;
            j -= 1;
            jj -= 1;
        }

        // Is the reconstruction consistent? If so the knot carried no shape.
        if (j as isize - i as isize) < t as isize {
            if dist4(temp[ii - 1], temp[jj + 1]) <= tolerance {
                remflag = true;
            }
        } else {
            let alfi = (u - uv[i]) / (uv[i + p + 1 + t] - uv[i]);
            let mut lhs = [0.0f64; 4];
            for c in 0..4 {
                lhs[c] = alfi * temp[ii + t + 1][c] + (1.0 - alfi) * temp[ii - 1][c];
            }
            if dist4(pw[i], lhs) <= tolerance {
                remflag = true;
            }
        }

        if !remflag {
            break;
        }

        // Commit: shuffle the reconstructed points back in.
        let (mut i2, mut j2) = (first, last);
        while j2 as isize - i2 as isize > t as isize {
            pw[i2] = temp[i2 - off];
            pw[j2] = temp[j2 - off];
            i2 += 1;
            j2 -= 1;
        }
        first = first.saturating_sub(1);
        last += 1;
        removed += 1;
    }

    if removed == 0 {
        return (curve.clone(), 0);
    }

    // Close the gaps the removals left in the knot vector and control points.
    let m = n + p + 1;
    for k in (r + 1)..=m {
        uv[k - removed] = uv[k];
    }
    uv.truncate(m + 1 - removed);

    // A5.8's `fout` — the first control point that goes away. Note it does *not*
    // depend on how many were removed; getting that wrong shifts the whole tail
    // by one and bends the curve by ~1e-4 while still producing a plausible
    // control net.
    let fout = ((2 * r) as isize - s as isize - p as isize) / 2;
    let (mut i3, mut j3) = (fout, fout);
    for k in 1..removed {
        if (k % 2) == 1 {
            i3 += 1;
        } else {
            j3 -= 1;
        }
    }
    let mut j = j3 as usize;
    for k in ((i3 + 1) as usize)..=n {
        pw[j] = pw[k];
        j += 1;
    }
    pw.truncate(n + 1 - removed);

    (NurbsCurve::from_homogeneous_unchecked(p, uv, pw), removed)
}

fn dist4(a: V4, b: V4) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2) + (a[3] - b[3]).powi(2))
        .sqrt()
}

/// Remove every removable knot, leaving the minimal representation of the same
/// curve.
///
/// Interior knots are tried from the highest multiplicity down, which is the
/// order that lets a knot inserted `k` times come out `k` times.
pub fn compact(curve: &NurbsCurve, tolerance: f64) -> NurbsCurve {
    let mut out = curve.clone();
    loop {
        let (u0, u1) = out.domain();
        let interior: Vec<(f64, usize)> = distinct_knots(out.knots())
            .into_iter()
            .filter(|&(v, _)| v > u0 && v < u1)
            .collect();
        let mut changed = false;
        for (value, mult) in interior {
            let (next, removed) = remove_knot(&out, value, mult, tolerance);
            if removed > 0 {
                out = next;
                changed = true;
            }
        }
        if !changed {
            return out;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nurbs::construct;
    use crate::nurbs::v3;

    fn sample_curve() -> NurbsCurve {
        NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0],
            &[
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
                [3.0, -1.0, 1.0],
                [5.0, 2.0, -1.0],
                [7.0, 0.0, 0.0],
                [8.0, 3.0, 2.0],
                [10.0, 1.0, 0.0],
            ],
            Some(&[1.0, 2.0, 0.5, 1.5, 1.0, 3.0, 1.0]),
        )
        .unwrap()
    }

    fn assert_same_curve(a: &NurbsCurve, b: &NurbsCurve, tol: f64, what: &str) {
        for i in 0..=500 {
            let t = i as f64 / 500.0;
            let pa = a.point(a.param_at(t));
            let pb = b.point(b.param_at(t));
            let d = v3::dist(pa, pb);
            assert!(d < tol, "{what}: diverged by {d} at t = {t}");
        }
    }

    #[test]
    fn insertion_preserves_the_curve() {
        let c = sample_curve();
        let inserted = insert_knot(&c, 0.4, 1);
        assert_eq!(inserted.n_control(), c.n_control() + 1);
        assert_eq!(inserted.knots().len(), c.knots().len() + 1);
        assert_same_curve(&c, &inserted, 1e-13, "single insertion");
    }

    #[test]
    fn repeated_insertion_preserves_the_curve() {
        let c = sample_curve();
        let inserted = insert_knot(&c, 0.6, 3);
        assert_eq!(inserted.n_control(), c.n_control() + 3);
        assert_same_curve(&c, &inserted, 1e-13, "triple insertion");
    }

    #[test]
    fn insertion_at_an_existing_knot_preserves_the_curve() {
        let c = sample_curve();
        let inserted = insert_knot(&c, 0.5, 2);
        assert_same_curve(&c, &inserted, 1e-13, "insertion at existing knot");
    }

    #[test]
    fn insertion_is_clamped_to_the_degree() {
        let c = sample_curve();
        // Degree 3, existing multiplicity 1 → at most 2 more.
        let inserted = insert_knot(&c, 0.5, 99);
        assert_eq!(multiplicity(inserted.knots(), 0.5), 3);
        assert_same_curve(&c, &inserted, 1e-13, "clamped insertion");
    }

    #[test]
    fn refinement_preserves_the_curve() {
        let c = sample_curve();
        let refined = refine(&c, &[0.1, 0.2, 0.3, 0.45, 0.62, 0.9]);
        assert_eq!(refined.n_control(), c.n_control() + 6);
        assert_same_curve(&c, &refined, 1e-12, "refinement");
    }

    #[test]
    fn refinement_preserves_a_rational_circle() {
        let c = construct::circle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0);
        let refined = refine(&c, &[0.1, 0.35, 0.6, 0.85]);
        for i in 0..=1000 {
            let u = refined.param_at(i as f64 / 1000.0);
            let p = refined.point(u);
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!((r - 2.0).abs() < 1e-12, "radius {r} after refinement");
        }
    }

    #[test]
    fn split_halves_reproduce_the_original() {
        let c = sample_curve();
        let (left, right) = split(&c, 0.4).expect("interior split");
        for i in 0..=200 {
            let t = i as f64 / 200.0;
            // Left covers [0, 0.4], right covers [0.4, 1] in the original param.
            let u_left = 0.4 * t;
            let u_right = 0.4 + 0.6 * t;
            assert!(
                v3::dist(left.point(u_left), c.point(u_left)) < 1e-12,
                "left at {u_left}"
            );
            assert!(
                v3::dist(right.point(u_right), c.point(u_right)) < 1e-12,
                "right at {u_right}"
            );
        }
    }

    #[test]
    fn split_rejects_the_endpoints() {
        let c = sample_curve();
        assert!(split(&c, 0.0).is_none());
        assert!(split(&c, 1.0).is_none());
        assert!(split(&c, 2.0).is_none());
    }

    #[test]
    fn bezier_decomposition_has_the_right_shape() {
        let c = sample_curve();
        let segs = bezier_segments(&c);
        // Four interior spans → four Bézier segments of degree 3.
        assert_eq!(segs.len(), 4);
        assert!(segs.iter().all(|s| s.len() == 4));
        // Consecutive segments share an endpoint.
        for w in segs.windows(2) {
            for (end, start) in w[0][3].iter().zip(w[1][0].iter()) {
                assert!((end - start).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn degree_elevation_preserves_the_curve() {
        let c = sample_curve();
        for t in 1..=3 {
            let e = elevate_degree(&c, t);
            assert_eq!(e.degree(), c.degree() + t);
            assert_same_curve(&c, &e, 1e-11, &format!("elevation by {t}"));
        }
    }

    #[test]
    fn degree_elevation_preserves_a_rational_circle() {
        let c = construct::circle([1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 3.0);
        let e = elevate_degree(&c, 2);
        assert_eq!(e.degree(), 4);
        for i in 0..=500 {
            let u = e.param_at(i as f64 / 500.0);
            let p = e.point(u);
            let r = ((p[0] - 1.0).powi(2) + (p[1] - 1.0).powi(2)).sqrt();
            assert!((r - 3.0).abs() < 1e-11, "radius {r} after elevation");
        }
    }

    #[test]
    fn make_compatible_agrees_on_degree_and_knots() {
        let a = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            &[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            None,
        )
        .unwrap();
        let b = construct::circle([0.0, 0.0, 5.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 4.0);

        let (ca, cb) = make_compatible(&a, &b).unwrap();
        assert_eq!(ca.degree(), cb.degree());
        assert_eq!(ca.n_control(), cb.n_control());
        assert_eq!(ca.knots(), cb.knots());
        assert_same_curve(&a, &ca, 1e-11, "compatible a");
        assert_same_curve(&b, &cb, 1e-11, "compatible b");
    }

    #[test]
    fn surface_insertion_preserves_the_surface() {
        let s = construct::torus([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 3.0, 1.0);
        let su = insert_knot_surface_u(&s, 0.4, 1);
        let sv = insert_knot_surface_v(&su, 0.7, 1);
        assert_eq!(su.n_u(), s.n_u() + 1);
        assert_eq!(sv.n_v(), s.n_v() + 1);

        for i in 0..=40 {
            for j in 0..=40 {
                let (a, b) = (i as f64 / 40.0, j as f64 / 40.0);
                let (u, v) = s.param_at(a, b);
                let (u2, v2) = sv.param_at(a, b);
                let d = v3::dist(s.point(u, v), sv.point(u2, v2));
                assert!(d < 1e-12, "diverged by {d} at ({a}, {b})");
            }
        }
    }

    #[test]
    fn removing_an_inserted_knot_restores_the_original_representation() {
        // The round trip that defines removability: insert, remove, and the
        // control net comes back — not merely a curve through the same points.
        let c = sample_curve();
        for &u in &[0.3, 0.55, 0.8] {
            let inserted = insert_knot(&c, u, 1);
            assert_eq!(inserted.n_control(), c.n_control() + 1);

            let (back, removed) = remove_knot(&inserted, u, 1, 1e-9);
            assert_eq!(removed, 1, "the knot we just inserted must be removable");
            assert_eq!(back.n_control(), c.n_control());
            assert_same_curve(&c, &back, 1e-12, "insert-then-remove");
        }
    }

    #[test]
    fn a_knot_that_carries_shape_is_not_removed() {
        // Removability is a property of the geometry, not of the knot vector.
        // Move a control point so the knot genuinely bends the curve, and the
        // residual check must refuse.
        let c = sample_curve();
        let inserted = insert_knot(&c, 0.5, 1);
        let mut cw = inserted.homogeneous().to_vec();
        let k = cw.len() / 2;
        cw[k][0] += 3.0;
        let bent = NurbsCurve::from_homogeneous_unchecked(
            inserted.degree(),
            inserted.knots().to_vec(),
            cw,
        );
        let (_, removed) = remove_knot(&bent, 0.5, 1, 1e-9);
        assert_eq!(removed, 0, "a load-bearing knot must survive");
    }

    #[test]
    fn degree_elevation_now_compacts() {
        // The deferred item: elevation used to leave every interior knot at full
        // multiplicity `p + t`. It should leave `original + t`.
        let c = sample_curve(); // degree 3, interior knots at .25/.5/.75, mult 1
        for t in 1..=2 {
            let e = elevate_degree(&c, t);
            assert_eq!(e.degree(), 3 + t);
            for (value, mult) in distinct_knots(e.knots()) {
                if value > 0.0 && value < 1.0 {
                    assert_eq!(
                        mult,
                        1 + t,
                        "interior knot {value} came back at multiplicity {mult}, want {}",
                        1 + t
                    );
                }
            }
            // Minimal, and still the same curve.
            assert_eq!(e.n_control(), c.n_control() + t * 4);
            assert_same_curve(&c, &e, 1e-10, &format!("compact elevation by {t}"));
        }
    }

    #[test]
    fn compaction_does_not_touch_a_curve_that_is_already_minimal() {
        let c = sample_curve();
        let same = compact(&c, 1e-12);
        assert_eq!(same.knots(), c.knots());
        assert_eq!(same.n_control(), c.n_control());
    }

    #[test]
    fn compaction_undoes_refinement() {
        let c = sample_curve();
        let refined = refine(&c, &[0.1, 0.35, 0.6, 0.9]);
        assert_eq!(refined.n_control(), c.n_control() + 4);
        let back = compact(&refined, 1e-9);
        assert_eq!(back.n_control(), c.n_control(), "refinement was not undone");
        assert_same_curve(&c, &back, 1e-11, "refine-then-compact");
    }

    #[test]
    fn a_compacted_circle_is_still_exactly_circular() {
        let c = construct::circle([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 4.0);
        let e = elevate_degree(&c, 2);
        for i in 0..=1000 {
            let u = e.param_at(i as f64 / 1000.0);
            let p = e.point(u);
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!((r - 4.0).abs() < 1e-10, "radius {r}");
        }
    }

    #[test]
    fn end_knots_are_never_removed() {
        // The clamping is what makes the curve interpolate its endpoints.
        let c = sample_curve();
        for &u in &[0.0, 1.0] {
            let (_, removed) = remove_knot(&c, u, 3, 1e-6);
            assert_eq!(removed, 0, "removing an end knot would unclamp the curve");
        }
    }

    #[test]
    fn distinct_knots_counts_multiplicity() {
        let d = distinct_knots(&[0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]);
        assert_eq!(d, vec![(0.0, 3), (0.5, 2), (1.0, 3)]);
    }
}
