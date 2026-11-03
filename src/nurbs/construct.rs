//! Exact constructors — the reason NURBS is worth having as *the* surface type.
//!
//! A circular arc here is a rational quadratic with weight `cos(Δθ/2)` on its
//! middle control point, which reproduces the circle to machine precision at
//! every parameter. Sphere, cylinder, cone and torus are surfaces of revolution
//! built on those arcs, so they are exact too — a sampled sphere's radius is
//! constant to ~1e-15, not to whatever `$fn` was chosen.
//!
//! That exactness is the property Stage 1 (`brep`) carries as provenance and
//! Stage 2 (`brep-csg`) intersects in closed form. Everything downstream depends
//! on the analytic shape surviving construction, which is precisely what the
//! current mesh-first primitives in `src/geometries/` and `src/openscad/` do not
//! do.
//!
//! Algorithm numbers refer to Piegl & Tiller, *The NURBS Book* (2nd ed.):
//! A7.1 for the arc, A8.1 for revolution.

use super::curve::NurbsCurve;
use super::knot::make_compatible;
use super::surface::NurbsSurface;
use super::v3;
use super::{project, NurbsError, V3, V4};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// A unit vector perpendicular to `axis`, chosen from the coordinate direction
/// least aligned with it so the cross product is well conditioned.
pub fn perpendicular(axis: V3) -> V3 {
    let a = [axis[0].abs(), axis[1].abs(), axis[2].abs()];
    let i = if a[0] <= a[1] && a[0] <= a[2] {
        0
    } else if a[1] <= a[2] {
        1
    } else {
        2
    };
    let mut e = [0.0; 3];
    e[i] = 1.0;
    v3::normalize(v3::cross(axis, e)).unwrap_or([1.0, 0.0, 0.0])
}

/// Orthonormalize a pair of in-plane directions (Gram–Schmidt).
///
/// Callers pass axes that are conceptually perpendicular but often only
/// approximately so — a normal recovered from a mesh, say. Orthonormalizing
/// here means a slightly-off `y_axis` skews nothing: the arc stays circular.
fn ortho_frame(x_axis: V3, y_axis: V3) -> (V3, V3) {
    let x = v3::normalize(x_axis).unwrap_or([1.0, 0.0, 0.0]);
    let y_res = v3::sub(y_axis, v3::scale(x, v3::dot(y_axis, x)));
    let y = v3::normalize(y_res).unwrap_or_else(|| perpendicular(x));
    (x, y)
}

/// How many rational-quadratic segments a sweep of `theta` needs. Each segment
/// spans at most a quarter turn, above which the middle control point runs off
/// to infinity.
fn arc_segments(theta: f64) -> usize {
    const EPS: f64 = 1e-12;
    if theta <= FRAC_PI_2 + EPS {
        1
    } else if theta <= PI + EPS {
        2
    } else if theta <= 3.0 * FRAC_PI_2 + EPS {
        3
    } else {
        4
    }
}

/// Clamped knot vector for an `n_seg`-segment quadratic arc: `[0,0,0, 1/n,1/n,
/// …, 1,1,1]`.
fn arc_knots(n_seg: usize) -> Vec<f64> {
    let n = 2 * n_seg + 4;
    let mut u = vec![0.0; n];
    for i in 1..n_seg {
        let v = i as f64 / n_seg as f64;
        u[2 * i + 1] = v;
        u[2 * i + 2] = v;
    }
    for k in u.iter_mut().skip(n - 3) {
        *k = 1.0;
    }
    u
}

/// Intersect two coplanar lines by solving in the plane's 2D basis.
///
/// Doing it in 2D rather than with a 3D least-squares solve keeps the result
/// exactly in the plane, which matters: the arc's middle control points must be
/// coplanar with its endpoints or the "circle" acquires a wobble out of plane.
fn intersect_in_plane(center: V3, x: V3, y: V3, p0: V3, t0: V3, p2: V3, t2: V3) -> V3 {
    let to2 = |p: V3| {
        let d = v3::sub(p, center);
        [v3::dot(d, x), v3::dot(d, y)]
    };
    let dir2 = |t: V3| [v3::dot(t, x), v3::dot(t, y)];

    let a0 = to2(p0);
    let a2 = to2(p2);
    let d0 = dir2(t0);
    let d2 = dir2(t2);

    let det = d2[0] * d0[1] - d0[0] * d2[1];
    if det.abs() < 1e-300 {
        // Parallel tangents. Only reachable for a zero-length sweep, where the
        // two points coincide and p0 is the right answer anyway.
        return p0;
    }
    let rx = a2[0] - a0[0];
    let ry = a2[1] - a0[1];
    let a = (-rx * d2[1] + d2[0] * ry) / det;
    v3::add(p0, v3::scale(t0, a))
}

/// Homogeneous control points for a circular arc — A7.1.
///
/// A zero `radius` is legal and produces `2·n_seg + 1` copies of `center`, which
/// is exactly what a surface of revolution needs for a profile point sitting on
/// the axis (the degenerate row at a sphere's pole).
fn arc_control(center: V3, x: V3, y: V3, radius: f64, start: f64, theta: f64) -> Vec<V4> {
    let n_seg = arc_segments(theta);
    let dtheta = theta / n_seg as f64;
    let w1 = (dtheta / 2.0).cos();

    let at = |a: f64| {
        v3::add(
            center,
            v3::add(
                v3::scale(x, radius * a.cos()),
                v3::scale(y, radius * a.sin()),
            ),
        )
    };
    let tangent_at = |a: f64| v3::add(v3::scale(x, -a.sin()), v3::scale(y, a.cos()));

    let mut cw = vec![[0.0f64; 4]; 2 * n_seg + 1];
    let mut p0 = at(start);
    let mut t0 = tangent_at(start);
    cw[0] = [p0[0], p0[1], p0[2], 1.0];

    let mut index = 0;
    let mut angle = start;
    for _ in 0..n_seg {
        angle += dtheta;
        let p2 = at(angle);
        let t2 = tangent_at(angle);
        cw[index + 2] = [p2[0], p2[1], p2[2], 1.0];

        let p1 = intersect_in_plane(center, x, y, p0, t0, p2, t2);
        cw[index + 1] = [p1[0] * w1, p1[1] * w1, p1[2] * w1, w1];

        index += 2;
        p0 = p2;
        t0 = t2;
    }
    cw
}

/// Normalize a sweep angle into `(0, 2π]`. Zero means a full turn — the reading
/// every caller wants from `arc(…, 0.0, 0.0)` or `revolve(…, 0.0)`.
fn sweep_angle(theta: f64) -> f64 {
    let mut t = theta;
    if t <= 0.0 {
        t += TAU;
    }
    t.clamp(f64::MIN_POSITIVE, TAU)
}

/// A straight line segment as a degree-1 NURBS curve.
pub fn line(a: V3, b: V3) -> NurbsCurve {
    NurbsCurve::from_homogeneous_unchecked(
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![[a[0], a[1], a[2], 1.0], [b[0], b[1], b[2], 1.0]],
    )
}

/// An exact circular arc from `start` to `end` (radians, measured from
/// `x_axis` toward `y_axis`).
///
/// `end == start` means a full circle. The axes are orthonormalized, so they
/// need only span the plane, not be perpendicular to begin with.
pub fn arc(center: V3, x_axis: V3, y_axis: V3, radius: f64, start: f64, end: f64) -> NurbsCurve {
    let (x, y) = ortho_frame(x_axis, y_axis);
    let theta = sweep_angle(end - start);
    let cw = arc_control(center, x, y, radius, start, theta);
    NurbsCurve::from_homogeneous_unchecked(2, arc_knots(arc_segments(theta)), cw)
}

/// An exact full circle — four rational-quadratic segments, nine control points.
pub fn circle(center: V3, x_axis: V3, y_axis: V3, radius: f64) -> NurbsCurve {
    arc(center, x_axis, y_axis, radius, 0.0, TAU)
}

/// Sweep `profile` about an axis — A8.1.
///
/// `u` runs around the axis, `v` along the profile, so the `u = 0` iso-curve is
/// the profile itself. Profile points lying *on* the axis produce a degenerate
/// (collapsed) control row; that is legal and expected — it is how a sphere's
/// poles arise — and [`NurbsSurface::normal`] handles the resulting vanishing
/// cross product.
pub fn revolve(profile: &NurbsCurve, axis_point: V3, axis_dir: V3, angle: f64) -> NurbsSurface {
    let axis = v3::normalize(axis_dir).unwrap_or([0.0, 0.0, 1.0]);
    let theta = sweep_angle(angle);
    let n_seg = arc_segments(theta);
    let n_u = 2 * n_seg + 1;
    let n_v = profile.n_control();

    let mut control = vec![[0.0f64; 4]; n_u * n_v];
    for j in 0..n_v {
        let hj = profile.homogeneous()[j];
        let wj = hj[3];
        let pj = project(hj);

        // Foot of the perpendicular from the profile point to the axis.
        let d = v3::dot(v3::sub(pj, axis_point), axis);
        let o = v3::add(axis_point, v3::scale(axis, d));
        let radial = v3::sub(pj, o);
        let r = v3::norm(radial);

        let x = match v3::normalize(radial) {
            Some(x) => x,
            // On the axis: the radius is zero so the frame is arbitrary.
            None => perpendicular(axis),
        };
        let y = v3::cross(axis, x);

        let col = arc_control(o, x, y, r, 0.0, theta);
        for (i, c) in col.iter().enumerate() {
            control[i * n_v + j] = [c[0] * wj, c[1] * wj, c[2] * wj, c[3] * wj];
        }
    }

    NurbsSurface::from_homogeneous_unchecked(
        2,
        profile.degree(),
        arc_knots(n_seg),
        profile.knots().to_vec(),
        n_u,
        n_v,
        control,
    )
}

/// Sweep `profile` linearly along `dir`. `u` follows the profile, `v` the
/// translation.
pub fn extrude(profile: &NurbsCurve, dir: V3) -> NurbsSurface {
    let n_u = profile.n_control();
    let mut control = Vec::with_capacity(n_u * 2);
    for &h in profile.homogeneous() {
        let w = h[3];
        control.push(h);
        // Translation in homogeneous coordinates is `+ dir · w`.
        control.push([h[0] + dir[0] * w, h[1] + dir[1] * w, h[2] + dir[2] * w, w]);
    }
    NurbsSurface::from_homogeneous_unchecked(
        profile.degree(),
        1,
        profile.knots().to_vec(),
        vec![0.0, 0.0, 1.0, 1.0],
        n_u,
        2,
        control,
    )
}

/// The ruled surface between two curves — straight lines joining points of
/// equal normalized parameter.
///
/// The curves are made compatible first (common degree, common knot vector), so
/// they may differ in both to begin with.
pub fn ruled(a: &NurbsCurve, b: &NurbsCurve) -> Result<NurbsSurface, NurbsError> {
    let (a, b) = make_compatible(a, b)?;
    let n_u = a.n_control();
    let mut control = Vec::with_capacity(n_u * 2);
    for i in 0..n_u {
        control.push(a.homogeneous()[i]);
        control.push(b.homogeneous()[i]);
    }
    Ok(NurbsSurface::from_homogeneous_unchecked(
        a.degree(),
        1,
        a.knots().to_vec(),
        vec![0.0, 0.0, 1.0, 1.0],
        n_u,
        2,
        control,
    ))
}

/// A flat bilinear patch spanning `u_dir` × `v_dir` from `origin`.
pub fn plane(origin: V3, u_dir: V3, v_dir: V3) -> NurbsSurface {
    let p = |a: V3| [a[0], a[1], a[2], 1.0];
    let o_u = v3::add(origin, u_dir);
    NurbsSurface::from_homogeneous_unchecked(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        2,
        2,
        // u-major: (0,0), (0,1), (1,0), (1,1)
        vec![
            p(origin),
            p(v3::add(origin, v_dir)),
            p(o_u),
            p(v3::add(o_u, v_dir)),
        ],
    )
}

/// An exact sphere, as a full revolution of a pole-to-pole semicircle about `+Z`.
///
/// The two pole rows are degenerate by construction.
pub fn sphere(center: V3, radius: f64) -> NurbsSurface {
    let profile = arc(
        center,
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        radius,
        -FRAC_PI_2,
        FRAC_PI_2,
    );
    revolve(&profile, center, [0.0, 0.0, 1.0], TAU)
}

/// An exact cylindrical side surface (no caps) of the given radius and height,
/// starting at `base_center` and running along `axis`.
pub fn cylinder(base_center: V3, axis: V3, radius: f64, height: f64) -> NurbsSurface {
    let axis = v3::normalize(axis).unwrap_or([0.0, 0.0, 1.0]);
    let x = perpendicular(axis);
    let p0 = v3::add(base_center, v3::scale(x, radius));
    let p1 = v3::add(p0, v3::scale(axis, height));
    revolve(&line(p0, p1), base_center, axis, TAU)
}

/// An exact conical side surface (no cap) from `apex`, opening to `base_radius`
/// at distance `height` along `axis`. The apex row is degenerate.
pub fn cone(apex: V3, axis: V3, base_radius: f64, height: f64) -> NurbsSurface {
    let axis = v3::normalize(axis).unwrap_or([0.0, 0.0, 1.0]);
    let x = perpendicular(axis);
    let rim = v3::add(
        v3::add(apex, v3::scale(axis, height)),
        v3::scale(x, base_radius),
    );
    revolve(&line(apex, rim), apex, axis, TAU)
}

/// An exact torus: a circle of radius `minor`, centred `major` from the axis,
/// swept fully about it.
pub fn torus(center: V3, axis: V3, major: f64, minor: f64) -> NurbsSurface {
    let axis = v3::normalize(axis).unwrap_or([0.0, 0.0, 1.0]);
    let x = perpendicular(axis);
    let profile_center = v3::add(center, v3::scale(x, major));
    let profile = circle(profile_center, x, axis, minor);
    revolve(&profile, center, axis, TAU)
}


/// Bring a set of curves to a common degree and knot vector.
///
/// [`make_compatible`] pairwise is not enough for three or more: each pass can
/// introduce knots the earlier ones have not seen. Elevating everything to the
/// maximum degree first and then merging *all* the knot vectors in one pass is,
/// and it is also the only way the result does not depend on the input order.
fn unify(curves: &[NurbsCurve]) -> Result<Vec<NurbsCurve>, NurbsError> {
    use super::knot::{distinct_knots, elevate_degree, refine, reparameterize_unit};

    if curves.is_empty() {
        return Err(NurbsError::Incompatible("no curves"));
    }
    let mut out: Vec<NurbsCurve> = curves.iter().map(reparameterize_unit).collect();

    let max_degree = out.iter().map(|c| c.degree()).max().unwrap();
    for c in out.iter_mut() {
        if c.degree() < max_degree {
            *c = elevate_degree(c, max_degree - c.degree());
        }
    }

    // Target multiplicity per knot value = the maximum any curve asks for.
    let mut wanted: Vec<(f64, usize)> = Vec::new();
    for c in &out {
        for (v, m) in distinct_knots(c.knots()) {
            match wanted.iter_mut().find(|(w, _)| *w == v) {
                Some((_, wm)) => *wm = (*wm).max(m),
                None => wanted.push((v, m)),
            }
        }
    }
    wanted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    for c in out.iter_mut() {
        let have = distinct_knots(c.knots());
        let mut add: Vec<f64> = Vec::new();
        for &(v, m) in &wanted {
            let cur = have.iter().find(|(w, _)| *w == v).map_or(0, |&(_, k)| k);
            add.extend(std::iter::repeat_n(v, m.saturating_sub(cur)));
        }
        *c = refine(c, &add);
    }

    let n = out[0].n_control();
    if out.iter().any(|c| c.n_control() != n) {
        return Err(NurbsError::Incompatible(
            "degree elevation and knot merging did not converge",
        ));
    }
    Ok(out)
}

/// Chord-length parameters for a sequence of points, normalized to `[0, 1]`.
///
/// Chord length rather than uniform spacing: uniform parameters on unevenly
/// spaced data produce the loops and overshoots that make interpolated curves
/// look wrong, and the fix is to let the parameter track arc length.
fn chord_params(points: &[V4]) -> Vec<f64> {
    let n = points.len();
    let mut d = vec![0.0f64; n];
    let mut total = 0.0;
    for k in 1..n {
        let (a, b) = (project(points[k - 1]), project(points[k]));
        total += v3::dist(a, b);
        d[k] = total;
    }
    if total < 1e-300 {
        return (0..n).map(|k| k as f64 / (n - 1).max(1) as f64).collect();
    }
    d.iter().map(|x| x / total).collect()
}

/// Averaged knot vector for interpolation at `params` — P&T eq. 9.8.
///
/// Averaging is what keeps the interpolation system banded *and* non-singular;
/// a uniform knot vector on chord-length parameters can be neither.
fn averaged_knots(params: &[f64], degree: usize) -> Vec<f64> {
    let n = params.len();
    let p = degree;
    let mut u = vec![0.0f64; n + p + 1];
    for k in (n..n + p + 1).take(p + 1) {
        u[k] = 1.0;
    }
    for j in 1..(n - p) {
        u[j + p] = params[j..j + p].iter().sum::<f64>() / p as f64;
    }
    u
}

/// Solve `A · X = B` for `X`, where `B` and `X` are lists of 4-vectors.
/// Gaussian elimination with partial pivoting — `n` here is the number of
/// sections, which is small.
// The elimination indexes two *different* rows of `a` by the same counter, which
// no iterator rewrite expresses more clearly than the subscripts do.
#[allow(clippy::needless_range_loop)]
fn solve4(mut a: Vec<Vec<f64>>, mut b: Vec<V4>) -> Option<Vec<V4>> {
    let n = b.len();
    for col in 0..n {
        let pivot =
            (col..n).max_by(|&x, &y| a[x][col].abs().partial_cmp(&a[y][col].abs()).unwrap())?;
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in (col + 1)..n {
            let f = a[row][col] / a[col][col];
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            let (br, bc) = (b[row], b[col]);
            b[row] = [
                br[0] - f * bc[0],
                br[1] - f * bc[1],
                br[2] - f * bc[2],
                br[3] - f * bc[3],
            ];
        }
    }
    let mut x = vec![[0.0f64; 4]; n];
    for row in (0..n).rev() {
        let mut acc = b[row];
        for k in (row + 1)..n {
            for e in 0..4 {
                acc[e] -= a[row][k] * x[k][e];
            }
        }
        for e in 0..4 {
            x[row][e] = acc[e] / a[row][row];
        }
    }
    Some(x)
}

/// A surface skinned through a sequence of section curves — a *loft*.
///
/// `u` follows the sections, `v` runs across them. The sections are unified
/// first, so they may differ in degree and knot vector; `degree_v` is the
/// interpolation degree across them, clamped to what the section count supports
/// (two sections can only be joined linearly, whatever is asked for).
///
/// # The surface passes through every section
///
/// That is what makes it a loft rather than a fit, and it does not come for
/// free. Stacking the sections *as* control rows — the obvious construction —
/// interpolates only the first and last, because a clamped B-spline is pulled
/// toward its interior control points without reaching them. A three-section
/// loft built that way misses its middle section by 0.625 on a unit-scale model.
///
/// So the control rows are *solved for*: chord-length parameters across the
/// sections, an averaged knot vector, and a small dense linear system per
/// control column (P&T §9.2.1, applied column-wise as in §10.3). Sections are
/// interpolated in homogeneous coordinates, so rational sections — a circle —
/// come through exactly too.
pub fn loft(sections: &[NurbsCurve], degree_v: usize) -> Result<NurbsSurface, NurbsError> {
    loft_with_params(sections, degree_v).map(|(s, _)| s)
}

/// [`loft`], plus the `v` parameter each section sits at.
///
/// Sections land at **chord-length** parameters, not uniform ones — that is what
/// stops unevenly spaced sections from producing the overshoot that uniform
/// spacing gives. A caller that wants to query, trim or split at a section needs
/// those values, and cannot guess them.
pub fn loft_with_params(
    sections: &[NurbsCurve],
    degree_v: usize,
) -> Result<(NurbsSurface, Vec<f64>), NurbsError> {
    if sections.len() < 2 {
        return Err(NurbsError::Incompatible(
            "a loft needs at least two sections",
        ));
    }
    let unified = unify(sections)?;
    let n_u = unified[0].n_control();
    let n_v = unified.len();
    let q = degree_v.clamp(1, n_v - 1);

    // One parameterization for the whole surface, averaged over the columns —
    // per-column parameters would make the rows inconsistent and shear the
    // surface.
    let mut params = vec![0.0f64; n_v];
    for i in 0..n_u {
        let column: Vec<V4> = unified.iter().map(|c| c.homogeneous()[i]).collect();
        for (k, t) in chord_params(&column).into_iter().enumerate() {
            params[k] += t / n_u as f64;
        }
    }
    // Guard against a degenerate average (all sections coincident).
    if params.windows(2).any(|w| w[1] <= w[0]) {
        params = (0..n_v).map(|k| k as f64 / (n_v - 1) as f64).collect();
    }

    let knots_v = averaged_knots(&params, q);

    // The interpolation matrix is the same for every column.
    let mut mat = vec![vec![0.0f64; n_v]; n_v];
    for (k, &t) in params.iter().enumerate() {
        let span = super::basis::find_span(q, n_v, &knots_v, t);
        let n = super::basis::basis_funcs(span, t, q, &knots_v);
        for (j, &b) in n.iter().enumerate() {
            mat[k][span - q + j] = b;
        }
    }

    let mut control = vec![[0.0f64; 4]; n_u * n_v];
    for i in 0..n_u {
        let rhs: Vec<V4> = unified.iter().map(|c| c.homogeneous()[i]).collect();
        let solved = solve4(mat.clone(), rhs)
            .ok_or(NurbsError::Incompatible("the loft system is singular"))?;
        for (j, cp) in solved.into_iter().enumerate() {
            control[i * n_v + j] = cp;
        }
    }

    Ok((
        NurbsSurface::from_homogeneous_unchecked(
            unified[0].degree(),
            q,
            unified[0].knots().to_vec(),
            knots_v,
            n_u,
            n_v,
            control,
        ),
        params,
    ))
}

/// Sweep `profile` along `spine`, carrying it in a rotation-minimising frame.
///
/// `u` follows the profile, `v` the spine.
///
/// # The frame
///
/// The obvious choice — a Frenet frame — is unusable: its normal flips through
/// an inflection point and spins without bound where the spine is locally
/// straight, so a swept tube visibly twists at exactly the places a designer did
/// not ask it to. This uses the *double-reflection* method instead (Wang et al.
/// 2008): each frame is the previous one reflected twice, which propagates the
/// orientation with no torsion of its own and is exact for a planar spine.
///
/// `samples` sets how many frames are placed along the spine; the result
/// interpolates the profile at each, so it is exact at the samples and
/// approximate between them. That is the honest character of a swept surface
/// built this way — an exact sweep of a general profile along a general spine is
/// not a NURBS surface at all.
pub fn sweep(
    profile: &NurbsCurve,
    spine: &NurbsCurve,
    samples: usize,
) -> Result<NurbsSurface, NurbsError> {
    let n = samples.max(2);
    let (s0, s1) = spine.domain();

    // Reference point the profile is carried relative to: the spine's start.
    let origin = spine.point(s0);
    let mut tangent = spine.tangent(s0).ok_or(NurbsError::Incompatible(
        "the spine has no tangent at its start",
    ))?;
    let mut normal = perpendicular(tangent);

    let mut sections: Vec<NurbsCurve> = Vec::with_capacity(n);
    let mut prev_point = origin;

    for k in 0..n {
        let t = s0 + (s1 - s0) * k as f64 / (n - 1) as f64;
        let point = spine.point(t);
        let next_tangent = spine.tangent(t).unwrap_or(tangent);

        if k > 0 {
            // Double reflection: reflect the frame in the plane bisecting the
            // step, then in the plane bisecting the tangent turn.
            let v1 = v3::sub(point, prev_point);
            let c1 = v3::dot(v1, v1);
            if c1 > 1e-300 {
                let r_l = v3::sub(normal, v3::scale(v1, 2.0 / c1 * v3::dot(v1, normal)));
                let t_l = v3::sub(tangent, v3::scale(v1, 2.0 / c1 * v3::dot(v1, tangent)));
                let v2 = v3::sub(next_tangent, t_l);
                let c2 = v3::dot(v2, v2);
                normal = if c2 > 1e-300 {
                    v3::sub(r_l, v3::scale(v2, 2.0 / c2 * v3::dot(v2, r_l)))
                } else {
                    r_l
                };
            }
            tangent = next_tangent;
        }
        // Re-orthonormalize against drift.
        let n_hat = v3::normalize(v3::sub(
            normal,
            v3::scale(tangent, v3::dot(normal, tangent)),
        ))
        .unwrap_or_else(|| perpendicular(tangent));
        normal = n_hat;
        let binormal = v3::cross(tangent, n_hat);

        // Place the profile in this frame. The profile is read in the *start*
        // frame's coordinates so an unrotated spine reproduces it verbatim.
        let cw: Vec<V4> = profile
            .homogeneous()
            .iter()
            .map(|h| {
                let w = h[3];
                let p = project(*h);
                let local = v3::sub(p, origin);
                let placed = v3::add(
                    point,
                    v3::add(
                        v3::add(
                            v3::scale(n_hat, v3::dot(local, [1.0, 0.0, 0.0])),
                            v3::scale(binormal, v3::dot(local, [0.0, 1.0, 0.0])),
                        ),
                        v3::scale(tangent, v3::dot(local, [0.0, 0.0, 1.0])),
                    ),
                );
                [placed[0] * w, placed[1] * w, placed[2] * w, w]
            })
            .collect();

        sections.push(NurbsCurve::from_homogeneous_unchecked(
            profile.degree(),
            profile.knots().to_vec(),
            cw,
        ));
        prev_point = point;
    }

    loft(&sections, 3.min(n - 1))
}

/// A cubic B-spline through every one of `points`, closing periodically when
/// asked.
///
/// This is what a traced intersection wants to be. Kept as a polyline it is
/// frozen: whoever samples it first fixes the boundary, and everything else —
/// the face across it, a later refinement, a second boolean — has to chase that
/// sampling or come apart from it. A curve can be evaluated anywhere, so two
/// faces asking for the same parameter get the same point by construction, and
/// refining is re-evaluation rather than subdividing chords that sag off the
/// surface.
///
/// Interpolating rather than approximating is deliberate: the input points are
/// already vertices two faces share, and a fit that moved them would move a
/// boundary. Every input point is on the curve, at a knot.
///
/// Returns the curve and the parameter of each input point, so a caller can ask
/// for "the third point" without re-deriving where that landed.
///
/// Catmull-Rom tangents, each span written as a cubic Bezier, so there is no
/// linear system to solve and no end conditions to choose — and a closed curve
/// closes *exactly*, because the wrap is in how the tangents are taken rather
/// than in a distance test against the first point.
pub fn interpolate(points: &[V3], closed: bool) -> Result<(NurbsCurve, Vec<f64>), NurbsError> {
    let n = points.len();
    if n < 2 {
        return Err(NurbsError::DegreeTooHigh {
            degree: 3,
            n_ctrl: n,
        });
    }
    // Chord-length parameters, one per span.
    let span_of = |i: usize| -> (V3, V3) { (points[i], points[(i + 1) % n]) };
    let spans = if closed { n } else { n - 1 };
    let mut lengths = Vec::with_capacity(spans);
    for i in 0..spans {
        let (a, b) = span_of(i);
        lengths.push(v3::dist(b, a).max(1e-12));
    }

    // Catmull-Rom tangent at each point: the chord across its neighbours,
    // scaled to the spans either side. At an open end, the one-sided chord.
    let at = |i: isize| -> V3 {
        if closed {
            points[i.rem_euclid(n as isize) as usize]
        } else {
            points[i.clamp(0, n as isize - 1) as usize]
        }
    };
    let tangent =
        |i: usize| -> V3 { v3::scale(v3::sub(at(i as isize + 1), at(i as isize - 1)), 0.5) };

    // One cubic Bezier per span, in B-spline form: interior knots of
    // multiplicity 3 make the segments independent and C1 across the joins.
    let mut ctrl: Vec<V3> = Vec::with_capacity(3 * spans + 1);
    for i in 0..spans {
        let (a, b) = span_of(i);
        let (ta, tb) = (tangent(i), tangent((i + 1) % n));
        ctrl.push(a);
        ctrl.push(v3::add(a, v3::scale(ta, 1.0 / 3.0)));
        ctrl.push(v3::sub(b, v3::scale(tb, 1.0 / 3.0)));
    }
    ctrl.push(if closed { points[0] } else { points[n - 1] });

    // Knots: clamped, with each interior join repeated three times, spaced by
    // chord length so the parameterisation follows the curve.
    let total: f64 = lengths.iter().sum();
    let mut cuts = Vec::with_capacity(spans + 1);
    let mut run = 0.0;
    cuts.push(0.0);
    for l in &lengths {
        run += l / total;
        cuts.push(run);
    }
    let mut knots = vec![0.0, 0.0, 0.0, 0.0];
    for c in cuts.iter().take(spans).skip(1) {
        knots.extend_from_slice(&[*c, *c, *c]);
    }
    knots.extend_from_slice(&[1.0, 1.0, 1.0, 1.0]);

    Ok((NurbsCurve::new(3, knots, &ctrl, None)?, cuts))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn radius_from(p: V3, center: V3, axis: V3) -> f64 {
        let d = v3::sub(p, center);
        let along = v3::dot(d, axis);
        v3::norm(v3::sub(d, v3::scale(axis, along)))
    }

    #[test]
    fn full_circle_has_nine_control_points_and_four_spans() {
        let c = circle([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1.0);
        assert_eq!(c.degree(), 2);
        assert_eq!(c.n_control(), 9);
        assert_eq!(c.knots().len(), 12);
    }

    #[test]
    fn circle_weights_are_the_expected_conic_weights() {
        let c = circle([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1.0);
        let expected = std::f64::consts::FRAC_1_SQRT_2; // cos(45°)
        for i in 0..9 {
            let w = c.weight(i);
            let want = if i % 2 == 0 { 1.0 } else { expected };
            assert!((w - want).abs() < 1e-15, "weight {i} = {w}, want {want}");
        }
    }

    #[test]
    fn arcs_of_every_segment_count_are_exact() {
        // Sweeps that land in each of the 1/2/3/4-segment buckets.
        for &deg in &[30.0, 89.0, 91.0, 179.0, 181.0, 269.0, 271.0, 359.0, 360.0] {
            let end = deg * PI / 180.0;
            let c = arc(
                [1.0, 2.0, 3.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                2.5,
                0.0,
                end,
            );
            for i in 0..=500 {
                let u = c.param_at(i as f64 / 500.0);
                let p = c.point(u);
                let r = radius_from(p, [1.0, 2.0, 3.0], [0.0, 0.0, 1.0]);
                assert!((r - 2.5).abs() < 1e-13, "{deg}°: radius {r} at u = {u}");
                assert!((p[2] - 3.0).abs() < 1e-13, "{deg}°: left the plane");
            }
        }
    }

    #[test]
    fn arc_is_orthonormalized_against_a_skewed_y_axis() {
        // y_axis deliberately not perpendicular to x_axis.
        let c = arc(
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.6, 1.0, 0.0],
            1.0,
            0.0,
            FRAC_PI_2,
        );
        for i in 0..=200 {
            let u = c.param_at(i as f64 / 200.0);
            let p = c.point(u);
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!(
                (r - 1.0).abs() < 1e-13,
                "radius {r} — frame was not orthonormalized"
            );
        }
    }

    #[test]
    fn arc_respects_a_non_zero_start_angle() {
        let c = arc(
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            PI / 6.0,
            PI / 3.0,
        );
        let start = c.point(c.domain().0);
        let end = c.point(c.domain().1);
        assert!(v3::dist(start, [(PI / 6.0).cos(), (PI / 6.0).sin(), 0.0]) < 1e-14);
        assert!(v3::dist(end, [(PI / 3.0).cos(), (PI / 3.0).sin(), 0.0]) < 1e-14);
    }

    #[test]
    fn cylinder_is_exact() {
        let s = cylinder([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], 2.0, 5.0);
        for i in 0..=40 {
            for j in 0..=40 {
                let (u, v) = s.param_at(i as f64 / 40.0, j as f64 / 40.0);
                let p = s.point(u, v);
                let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
                assert!((r - 2.0).abs() < 1e-12, "radius {r}");
                assert!(p[2] >= -1.0 - 1e-12 && p[2] <= 4.0 + 1e-12, "z = {}", p[2]);
            }
        }
    }

    #[test]
    fn cylinder_normal_is_radial() {
        let s = cylinder([0.0; 3], [0.0, 0.0, 1.0], 1.5, 3.0);
        for i in 0..=20 {
            for j in 0..=20 {
                let (u, v) = s.param_at(i as f64 / 20.0, j as f64 / 20.0);
                let p = s.point(u, v);
                let n = s.normal(u, v).unwrap();
                let radial = v3::normalize([p[0], p[1], 0.0]).unwrap();
                assert!(v3::dot(n, radial).abs() > 1.0 - 1e-9);
            }
        }
    }

    #[test]
    fn cone_is_exact_and_has_a_degenerate_apex() {
        let (apex, axis, br, h) = ([0.0, 0.0, 4.0], [0.0, 0.0, -1.0], 2.0, 4.0);
        let s = cone(apex, axis, br, h);
        for i in 0..=30 {
            for j in 0..=30 {
                let (u, v) = s.param_at(i as f64 / 30.0, j as f64 / 30.0);
                let p = s.point(u, v);
                // Radius grows linearly from the apex along the axis.
                let t = (4.0 - p[2]) / h;
                let want = br * t;
                let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
                assert!(
                    (r - want).abs() < 1e-11,
                    "r = {r}, want {want} at z = {}",
                    p[2]
                );
            }
        }
        // Every u collapses to the apex at v = 0.
        for i in 0..=8 {
            let (u, v) = s.param_at(i as f64 / 8.0, 0.0);
            assert!(v3::dist(s.point(u, v), apex) < 1e-13);
        }
    }

    #[test]
    fn torus_is_exact() {
        let (center, axis, major, minor) = ([1.0, 1.0, 1.0], [0.0, 0.0, 1.0], 5.0, 1.5);
        let s = torus(center, axis, major, minor);
        for i in 0..=40 {
            for j in 0..=40 {
                let (u, v) = s.param_at(i as f64 / 40.0, j as f64 / 40.0);
                let p = s.point(u, v);
                // Distance from the tube's centre circle is exactly `minor`.
                let d = v3::sub(p, center);
                let along = v3::dot(d, axis);
                let radial = v3::norm(v3::sub(d, v3::scale(axis, along)));
                let tube = ((radial - major).powi(2) + along * along).sqrt();
                assert!((tube - minor).abs() < 1e-11, "tube radius {tube}");
            }
        }
    }

    #[test]
    fn revolve_partial_sweep_covers_exactly_the_requested_angle() {
        let profile = line([2.0, 0.0, 0.0], [2.0, 0.0, 3.0]);
        let s = revolve(&profile, [0.0; 3], [0.0, 0.0, 1.0], PI / 3.0);
        let start = s.point(s.domain_u().0, s.domain_v().0);
        let end = s.point(s.domain_u().1, s.domain_v().0);
        let a0 = start[1].atan2(start[0]);
        let a1 = end[1].atan2(end[0]);
        assert!(a0.abs() < 1e-13, "start angle {a0}");
        assert!((a1 - PI / 3.0).abs() < 1e-13, "end angle {a1}");
    }

    #[test]
    fn extrude_translates_the_profile() {
        let profile = circle([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0);
        let s = extrude(&profile, [0.0, 0.0, 7.0]);
        for i in 0..=20 {
            let t = i as f64 / 20.0;
            let (u, _) = s.param_at(t, 0.0);
            let base = s.point(u, s.domain_v().0);
            let top = s.point(u, s.domain_v().1);
            assert!(v3::dist(base, profile.point(profile.param_at(t))) < 1e-13);
            assert!(v3::dist(v3::sub(top, base), [0.0, 0.0, 7.0]) < 1e-13);
        }
    }

    #[test]
    fn ruled_surface_joins_curves_of_different_degree() {
        let a = line([-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]);
        let b = circle([0.0, 0.0, 6.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 3.0);
        let s = ruled(&a, &b).expect("compatible");
        for i in 0..=20 {
            let t = i as f64 / 20.0;
            let (u, _) = s.param_at(t, 0.0);
            let p0 = s.point(u, s.domain_v().0);
            let p1 = s.point(u, s.domain_v().1);

            // v = 0 is `a` and v = 1 is `b`.
            assert!(
                v3::dist(p0, a.point(a.param_at(t))) < 1e-11,
                "v=0 is not `a`"
            );
            assert!(
                v3::dist(p1, b.point(b.param_at(t))) < 1e-11,
                "v=1 is not `b`"
            );

            // Every iso-u line is *straight*. It is not, however, uniformly
            // parameterized: the curves carry different weights (`b` is a
            // rational circle), so S(u, ½) is the weighted — not arithmetic —
            // mean of the endpoints. Collinearity is the property that actually
            // defines a ruled surface, and the one invariant to that weighting.
            let axis = v3::normalize(v3::sub(p1, p0)).expect("non-degenerate rule line");
            for k in 1..8 {
                let vt = k as f64 / 8.0;
                let (_, v) = s.param_at(t, vt);
                let p = s.point(u, v);
                let d = v3::sub(p, p0);
                let off = v3::norm(v3::sub(d, v3::scale(axis, v3::dot(d, axis))));
                assert!(off < 1e-11, "rule line bent by {off} at t = {t}, v = {vt}");
            }
        }
    }

    #[test]
    fn plane_patch_spans_its_directions() {
        let s = plane([1.0, 2.0, 3.0], [4.0, 0.0, 0.0], [0.0, 5.0, 0.0]);
        assert!(v3::dist(s.point(0.0, 0.0), [1.0, 2.0, 3.0]) < 1e-15);
        assert!(v3::dist(s.point(1.0, 0.0), [5.0, 2.0, 3.0]) < 1e-15);
        assert!(v3::dist(s.point(0.0, 1.0), [1.0, 7.0, 3.0]) < 1e-15);
        assert!(v3::dist(s.point(1.0, 1.0), [5.0, 7.0, 3.0]) < 1e-15);
    }

    #[test]
    fn a_loft_passes_exactly_through_every_section() {
        // The defining property. A loft that merely approximates its sections is
        // a fit, not a loft.
        let sections = vec![
            circle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1.0),
            circle([0.0, 0.0, 2.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0),
            circle([0.0, 0.0, 4.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.5),
        ];
        let (s, params) = loft_with_params(&sections, 2).expect("sections unify");
        assert_eq!(params.len(), sections.len());
        for (k, section) in sections.iter().enumerate() {
            let v = params[k];
            for i in 0..=40 {
                let t = i as f64 / 40.0;
                let want = section.point(section.param_at(t));
                let (u, _) = s.param_at(t, 0.0);
                let got = s.point(u, v);
                assert!(
                    v3::dist(got, want) < 1e-9,
                    "section {k} at t = {t}: off by {}",
                    v3::dist(got, want)
                );
            }
        }
    }

    #[test]
    fn a_loft_unifies_sections_of_different_degree() {
        // A line and a circle: degree 1 against degree 2, different knots.
        let sections = vec![
            line([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
            circle([0.0, 0.0, 3.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0),
        ];
        let s = loft(&sections, 1).expect("a line and a circle can be unified");
        let (v0, v1) = s.domain_v();
        for i in 0..=20 {
            let t = i as f64 / 20.0;
            let (u, _) = s.param_at(t, 0.0);
            assert!(v3::dist(s.point(u, v0), sections[0].point(sections[0].param_at(t))) < 1e-9);
            assert!(v3::dist(s.point(u, v1), sections[1].point(sections[1].param_at(t))) < 1e-9);
        }
    }

    #[test]
    fn lofting_two_sections_reduces_to_the_ruled_surface() {
        let a = line([-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]);
        let b = circle([0.0, 0.0, 6.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 3.0);
        let l = loft(&[a.clone(), b.clone()], 9).expect("clamped to degree 1");
        let r = ruled(&a, &b).expect("compatible");
        assert_eq!(l.degree_v(), 1, "two sections can only be joined linearly");
        for i in 0..=20 {
            for j in 0..=8 {
                let (su, sv) = (i as f64 / 20.0, j as f64 / 8.0);
                let (lu, lv) = l.param_at(su, sv);
                let (ru, rv) = r.param_at(su, sv);
                assert!(v3::dist(l.point(lu, lv), r.point(ru, rv)) < 1e-9);
            }
        }
    }

    #[test]
    fn a_loft_needs_at_least_two_sections() {
        let one = vec![circle([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1.0)];
        assert!(loft(&one, 1).is_err());
        assert!(loft(&[], 1).is_err());
    }

    #[test]
    fn sweeping_along_a_straight_spine_is_an_extrusion() {
        // The frame must not rotate where the spine does not turn. A Frenet frame
        // is undefined on a straight spine; the double-reflection frame is not.
        let profile = circle([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1.0);
        let spine = line([0.0, 0.0, 0.0], [0.0, 0.0, 5.0]);
        let s = sweep(&profile, &spine, 6).expect("sweep");

        for i in 0..=24 {
            for j in 0..=8 {
                let (su, sv) = (i as f64 / 24.0, j as f64 / 8.0);
                let (u, v) = s.param_at(su, sv);
                let p = s.point(u, v);
                let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
                assert!((r - 1.0).abs() < 1e-9, "radius {r} — the frame twisted");
            }
        }
        // And it spans the spine.
        let bottom = s.point(s.domain_u().0, s.domain_v().0);
        let top = s.point(s.domain_u().0, s.domain_v().1);
        assert!(bottom[2].abs() < 1e-9, "z = {}", bottom[2]);
        assert!((top[2] - 5.0).abs() < 1e-9, "z = {}", top[2]);
    }

    #[test]
    fn a_swept_profile_stays_perpendicular_to_a_curved_spine() {
        // The point of a rotation-minimising frame: the section stays square to
        // the spine all the way round without accumulating twist.
        let profile = circle([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.3);
        let spine = arc([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 4.0, 0.0, PI);
        let s = sweep(&profile, &spine, 24).expect("sweep");

        for j in 1..8 {
            let sv = j as f64 / 8.0;
            // Sample the section at this v and check it lies in a plane
            // perpendicular to the spine's tangent there.
            let (_, v) = s.param_at(0.0, sv);
            let ring: Vec<_> = (0..16)
                .map(|i| {
                    let (u, _) = s.param_at(i as f64 / 16.0, 0.0);
                    s.point(u, v)
                })
                .collect();
            let centre = ring.iter().fold([0.0; 3], |a, p| v3::add(a, *p));
            let centre = v3::scale(centre, 1.0 / ring.len() as f64);
            let spread = ring
                .iter()
                .map(|p| v3::dist(*p, centre))
                .fold(0.0f64, f64::max);
            assert!(
                (spread - 0.3).abs() < 0.02,
                "section at v = {sv} is not a circle of radius 0.3 (spread {spread})"
            );
        }
    }

    #[test]
    fn sphere_off_axis_center_is_still_exact() {
        let s = sphere([-3.0, 7.0, 2.0], 0.75);
        for i in 0..=30 {
            for j in 0..=30 {
                let (u, v) = s.param_at(i as f64 / 30.0, j as f64 / 30.0);
                let r = v3::dist(s.point(u, v), [-3.0, 7.0, 2.0]);
                assert!((r - 0.75).abs() < 1e-13, "radius {r}");
            }
        }
    }
}
