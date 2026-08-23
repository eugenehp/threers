//! Closed-form surface–surface intersection.
//!
//! Stage 2 of [`docs/brep-nurbs-plan.md`](../../docs/brep-nurbs-plan.md).
//!
//! # The contract
//!
//! This module is **strictly an accelerator**. Every pair without a closed form
//! returns [`SsiResult::Unknown`], and the kernel then runs exactly the code it
//! runs today. Nothing here may make an answer worse than the numeric path's,
//! which is why the honest thing — `Unknown` — is the default for everything
//! not explicitly enumerated below. The acceptance suite proves it: with `ssi`
//! forced to `Unknown` everywhere, kernel output is byte-identical.
//!
//! # Why this is worth having
//!
//! `src/exact_csg/mod.rs:20-25` names the kernel's two remaining failure modes:
//! *"two identical primitives translated along an axis"* and *"very large
//! coplanar faces"*. The first is two coaxial cylinders of equal radius — a
//! numeric nightmare, where every triangle of one is a hair from a triangle of
//! the other and `tri_tri_segment` produces slivers. Analytically it is a
//! struct comparison: same axis line, same radius, therefore the same surface.
//!
//! And where surfaces genuinely cross, the *exact* curve is available. The mesh
//! kernel's seam points carry ~1e-6 of error, which is what its snapping and
//! T-junction healing exist to paper over; projecting them onto the closed-form
//! intersection curve removes the error rather than compensating for it.
//!
//! `!(x > 0.0)` is the deliberate NaN-catching form; see `crate::nurbs::curve`.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use crate::nurbs::v3;
use crate::nurbs::V3;

use super::curve3d::{circle, Curve3d};
use super::Surface;

/// Absolute tolerance for deciding two surfaces share a frame element.
///
/// Absolute rather than relative because the quantities compared are already
/// normalized: unit directions, and distances measured against radii that the
/// caller's model scale sets. A model in millimetres and one in metres both want
/// "these axes are the same line" to mean the same thing at their own scale, and
/// the callers here pass scale-appropriate values in.
const EPS: f64 = 1e-9;

/// What two surfaces meet along.
#[derive(Debug, Clone, PartialEq)]
pub enum SsiResult {
    /// Proven not to intersect. The kernel can skip the pair entirely.
    Disjoint,
    /// Proven to be the same surface. `opposite` records whether their canonical
    /// normals point the same way — which decides the boolean's coincident-face
    /// rule, so it is not an incidental detail.
    Coincident { opposite: bool },
    /// The exact intersection curves.
    Curves(Vec<Curve3d>),
    /// No closed form here. Falls through to the numeric path, unchanged.
    Unknown,
}

impl SsiResult {
    /// Did this resolve to anything the kernel can act on?
    pub fn is_known(&self) -> bool {
        !matches!(self, SsiResult::Unknown)
    }
}

/// Intersect two surfaces in closed form, or report [`SsiResult::Unknown`].
///
/// Symmetric: `ssi(a, b)` and `ssi(b, a)` describe the same point set.
pub fn ssi(a: &Surface, b: &Surface) -> SsiResult {
    // Identity first — it is the cheapest test and the one that resolves the
    // kernel's worst degeneracies.
    if let Some(opposite) = coincident(a, b) {
        return SsiResult::Coincident { opposite };
    }

    use Surface::*;
    match (a, b) {
        (Plane { .. }, Plane { .. }) => plane_plane(a, b),
        (Plane { .. }, Sphere { .. }) => plane_sphere(a, b),
        (Sphere { .. }, Plane { .. }) => plane_sphere(b, a),
        (Plane { .. }, Cylinder { .. }) => plane_cylinder(a, b),
        (Cylinder { .. }, Plane { .. }) => plane_cylinder(b, a),
        (Plane { .. }, Cone { .. }) => plane_cone(a, b),
        (Cone { .. }, Plane { .. }) => plane_cone(b, a),
        (Plane { .. }, Torus { .. }) => plane_torus(a, b),
        (Torus { .. }, Plane { .. }) => plane_torus(b, a),
        (Sphere { .. }, Sphere { .. }) => sphere_sphere(a, b),
        (Sphere { .. }, Cylinder { .. }) => sphere_cylinder(a, b),
        (Cylinder { .. }, Sphere { .. }) => sphere_cylinder(b, a),
        (Cylinder { .. }, Cylinder { .. }) => cylinder_cylinder(a, b),
        (Cylinder { .. }, Cone { .. }) => cylinder_cone(a, b),
        (Cone { .. }, Cylinder { .. }) => cylinder_cone(b, a),
        (Cone { .. }, Cone { .. }) => cone_cone(a, b),
        // A torus against another surface of revolution. Exact when they share
        // an axis, `Unknown` otherwise.
        (Torus { .. }, Cylinder { .. } | Sphere { .. } | Cone { .. } | Torus { .. })
        | (Cylinder { .. } | Sphere { .. } | Cone { .. }, Torus { .. }) => coaxial(a, b),
        _ => SsiResult::Unknown,
    }
}

// ---------------------------------------------------------------------------
// identity
// ---------------------------------------------------------------------------

/// Trace an intersection curve that has no closed form.
///
/// Two cylinders crossing at different radii meet in a quartic; so does a torus
/// with anything off its axis. There is no conic to name — and the alternative
/// to tracing one is declining the operation, which for a cross-drilled hole is
/// declining something entirely ordinary.
///
/// The method is the standard one. The intersection of two smooth surfaces runs
/// along `n₁ × n₂`, so step that way and then pull the point back onto both
/// surfaces; repeat until it closes or leaves. What makes it trustworthy rather
/// than plausible is the correction: each point is driven to within
/// `tolerance × 1e-3` of *both* surfaces before it is kept, and a step that will
/// not converge ends the trace instead of being accepted.
/// Trace the curves two surfaces share.
///
/// The step is a fraction of the *model*, and a model is as big as its biggest
/// part: bore a small hole through a long rod and the curve where they meet gets
/// four points for a whole circle. Every one of them is on both surfaces
/// exactly, and the chords between them are nowhere near — 1.1e-3 off a sphere
/// at a tolerance of 1e-3. Everything downstream that asks whether a point is on
/// the surface then answers no for all but a sixth of the curve, and
/// `clip_to_face` hands the subdivision six fragments of one circle.
///
/// Putting points in afterwards, until every chord is within tolerance of what
/// it is a chord of, fixes that and costs two of the six results that could be
/// cut a second time — denser curves change what the subdivision sees
/// everywhere, not only here. Reverted. The step wants to come from the
/// curvature it is walking rather than from the size of the model, which is a
/// change to `trace` and not a pass over its output.
///
/// Tried: halve the step where the chord's midpoint falls further from the curve
/// than the tolerance allows, grow it where it falls much nearer. It costs
/// `a_result_can_be_cut_again` — a bored ball cut again comes back with 58 open
/// edges — with or without settling the shared vertices afterwards.
///
/// Not because anything depends on the step being *uniform*, which was the
/// obvious guess and is wrong. Measured, the result is
///
/// ```text
/// 2 faces, 4 edges, defects [EdgeFaceCount { edge: 0, uses: 1 }]
/// ```
///
/// — the drill's wall is missing, and an edge reads `sphere/sphere` because only
/// one face ever claimed it. That is the same missing-piece failure a coarse
/// trace gives, not two samplings disagreeing along a shared boundary. So an
/// adaptive step loses a piece for the same reason the wrong box did, and what
/// it costs is not paid for by the sampling being even.
pub fn march(a: &Surface, b: &Surface, bounds: (V3, V3), tolerance: f64) -> Vec<Curve3d> {
    // A step of the box, and the box is whatever the caller had lying around: a
    // ball of radius 3 drilled by a tool 60 long hands over a box 60 tall, so
    // the step comes to 0.6 for a ring whose whole circumference is 2.2 and the
    // ring arrives with six points on it. Nothing about that pair chose six; the
    // length of the drill did.
    //
    // What a step should answer to is the curve — `sqrt(8·r·tol)` is the
    // longest that holds the sagitta of a circle of radius `r` within
    // tolerance, and gives that ring the 41 points the trace can always find.
    // Measured, it costs three tests, all of them results lost where the
    // wrapping path cannot take a face whose only ring wraps. That path first,
    // then this.
    let scale = [a, b]
        .into_iter()
        .filter_map(|s| match s {
            Surface::Cylinder { radius, .. } | Surface::Sphere { radius, .. } => Some(*radius),
            Surface::Cone { half_angle, .. } => Some(half_angle.sin().max(1e-6)),
            Surface::Torus { minor, .. } => Some(*minor),
            _ => None,
        })
        .fold(f64::MAX, f64::min);
    let step = if scale < f64::MAX {
        (8.0 * scale * tolerance)
            .sqrt()
            .clamp(tolerance * 4.0, scale * 0.5)
    } else {
        (span(bounds) * 0.01).max(tolerance * 4.0)
    };
    let mut out: Vec<Curve3d> = Vec::new();

    for seed in seeds(a, b, bounds, tolerance, step) {
        // Already on a curve we have?
        if out.iter().any(|c| c.distance(seed) <= step) {
            continue;
        }
        let Some(points) = trace(a, b, seed, step, bounds, tolerance) else {
            continue;
        };
        if points.len() < 3 {
            continue;
        }
        let closed = v3::dist(points[0], points[points.len() - 1]) <= step;
        let mut points = points;
        if closed {
            points.pop();
        }
        out.push(Curve3d::Sampled { points, closed });
    }
    out
}

fn span(bounds: (V3, V3)) -> f64 {
    let (lo, hi) = bounds;
    (hi[0] - lo[0])
        .max(hi[1] - lo[1])
        .max(hi[2] - lo[2])
        .max(1e-9)
}

/// Pull `p` onto both surfaces at once.
///
/// Each surface contributes a correction along its own normal, which is a
/// Newton step on the pair of distance functions. `None` when it will not
/// converge — a tangential meeting, where the two normals are parallel and the
/// correction has no unique answer.
pub(crate) fn settle(a: &Surface, b: &Surface, mut p: V3, tolerance: f64) -> Option<V3> {
    let want = tolerance * 1e-3;
    for _ in 0..64 {
        let (da, db) = (a.signed_distance(p), b.signed_distance(p));
        if da.abs() <= want && db.abs() <= want {
            return Some(p);
        }
        let (Some((ua, va)), Some((ub, vb))) = (a.invert(p), b.invert(p)) else {
            return None;
        };
        let (Some(na), Some(nb)) = (a.normal(ua, va), b.normal(ub, vb)) else {
            return None;
        };
        // Solve the 2x2 system for the move along the two normals.
        let dot = v3::dot(na, nb);
        let det = 1.0 - dot * dot;
        if det.abs() < 1e-12 {
            return None; // tangential: no unique correction
        }
        let ca = (-da + db * dot) / det;
        let cb = (-db + da * dot) / det;
        p = v3::add(p, v3::add(v3::scale(na, ca), v3::scale(nb, cb)));
    }
    None
}

/// Points to start tracing from, one per component of the intersection.
fn seeds(a: &Surface, b: &Surface, bounds: (V3, V3), tolerance: f64, apart: f64) -> Vec<V3> {
    let (lo, hi) = bounds;
    const N: usize = 12;
    let mut out = Vec::new();
    let reach = span(bounds);
    for i in 0..=N {
        for j in 0..=N {
            for k in 0..=N {
                let p = [
                    lo[0] + (hi[0] - lo[0]) * i as f64 / N as f64,
                    lo[1] + (hi[1] - lo[1]) * j as f64 / N as f64,
                    lo[2] + (hi[2] - lo[2]) * k as f64 / N as f64,
                ];
                // Only bother where both surfaces are near.
                if a.distance(p) > reach / N as f64 || b.distance(p) > reach / N as f64 {
                    continue;
                }
                if let Some(q) = settle(a, b, p, tolerance) {
                    // Far enough apart to be *different branches*, which is a
                    // question about the curves and not about the grid. At a
                    // twelfth of the box, two rings 5.2 apart in a 60-tall box
                    // were 5.0 apart by this measure and the second was thrown
                    // away — never traced, never missed by anything downstream.
                    //
                    // A seed that repeats one already traced costs a `distance`
                    // and is dropped by `march` itself, so this errs small.
                    if out.iter().all(|r| v3::dist(*r, q) > apart) {
                        out.push(q);
                    }
                }
            }
        }
    }
    out
}

/// Walk one component from `seed`, forwards then backwards.
fn trace(
    a: &Surface,
    b: &Surface,
    seed: V3,
    step: f64,
    bounds: (V3, V3),
    tolerance: f64,
) -> Option<Vec<V3>> {
    let inside = |p: V3| {
        let (lo, hi) = bounds;
        (0..3).all(|k| p[k] >= lo[k] - step && p[k] <= hi[k] + step)
    };
    let tangent = |p: V3| -> Option<V3> {
        let (ua, va) = a.invert(p)?;
        let (ub, vb) = b.invert(p)?;
        v3::normalize(v3::cross(a.normal(ua, va)?, b.normal(ub, vb)?))
    };

    let cap = (span(bounds) / step * 8.0) as usize + 16;
    let mut forward = vec![seed];
    let mut dir = tangent(seed)?;
    let mut p = seed;
    let mut closed = false;
    for _ in 0..cap {
        let t = tangent(p)?;
        // Keep going the same way round.
        let t = if v3::dot(t, dir) < 0.0 {
            v3::scale(t, -1.0)
        } else {
            t
        };
        dir = t;
        let Some(next) = settle(a, b, v3::add(p, v3::scale(t, step)), tolerance) else {
            break;
        };
        if !inside(next) {
            break;
        }
        p = next;
        forward.push(p);
        if forward.len() > 3 && v3::dist(p, seed) <= step * 0.75 {
            // The point is pushed before it is asked about, so a walk that comes
            // back *past* where it started keeps the points it overshot by — and
            // the ring then laps its own head, crossing itself there. On a face
            // that is a ring with no ear anywhere, which is a fill that stops
            // short of its boundary.
            //
            // The seed lying behind the direction of travel is what "past" means.
            while forward.len() > 3 {
                let last = forward[forward.len() - 1];
                if v3::dot(v3::sub(seed, last), dir) >= 0.0 {
                    break;
                }
                forward.pop();
            }
            forward.push(seed);
            closed = true;
            break;
        }
    }
    if closed {
        return Some(forward);
    }

    // Open: walk the other way from the seed and prepend.
    let mut backward = Vec::new();
    let mut dir = v3::scale(tangent(seed)?, -1.0);
    let mut p = seed;
    for _ in 0..cap {
        let t = tangent(p)?;
        let t = if v3::dot(t, dir) < 0.0 {
            v3::scale(t, -1.0)
        } else {
            t
        };
        dir = t;
        let Some(next) = settle(a, b, v3::add(p, v3::scale(t, step)), tolerance) else {
            break;
        };
        if !inside(next) {
            break;
        }
        p = next;
        backward.push(p);
    }
    backward.reverse();
    backward.extend(forward);
    Some(backward)
}

/// Are these the same surface? `Some(opposite)` if so.
///
/// Frame-independent by construction: two cylinders are the same surface when
/// they share an axis *line* and a radius, whatever their `origin` along that
/// line or their `x_dir` around it. Comparing the structs field-for-field would
/// miss exactly the case this exists to catch — a coaxial pair built by two
/// different generators.
fn coincident(a: &Surface, b: &Surface) -> Option<bool> {
    use Surface::*;
    match (a, b) {
        (
            Plane {
                origin: o1,
                normal: n1,
                ..
            },
            Plane {
                origin: o2,
                normal: n2,
                ..
            },
        ) => {
            let d = v3::dot(*n1, *n2);
            if (d.abs() - 1.0).abs() > EPS {
                return None;
            }
            // Same plane if the second origin lies in the first.
            (v3::dot(v3::sub(*o2, *o1), *n1).abs() <= EPS).then_some(d < 0.0)
        }
        (
            Sphere {
                center: c1,
                radius: r1,
                axis: a1,
                ..
            },
            Sphere {
                center: c2,
                radius: r2,
                axis: a2,
                ..
            },
        ) => {
            if v3::dist(*c1, *c2) > EPS || (r1 - r2).abs() > EPS {
                return None;
            }
            // A sphere's outward normal is radial regardless of its axis, so two
            // coincident spheres are never opposite — the axis only sets the
            // parameterization.
            let _ = (a1, a2);
            Some(false)
        }
        (
            Cylinder {
                origin: o1,
                axis: x1,
                radius: r1,
                ..
            },
            Cylinder {
                origin: o2,
                axis: x2,
                radius: r2,
                ..
            },
        ) => {
            if (r1 - r2).abs() > EPS || !same_axis_line(*o1, *x1, *o2, *x2) {
                return None;
            }
            Some(false)
        }
        (
            Cone {
                apex: p1,
                axis: x1,
                half_angle: h1,
                ..
            },
            Cone {
                apex: p2,
                axis: x2,
                half_angle: h2,
                ..
            },
        ) => {
            if v3::dist(*p1, *p2) > EPS || (h1 - h2).abs() > EPS {
                return None;
            }
            // Same nappe only: cones opening opposite ways share an apex and an
            // angle but are different surfaces.
            (v3::dot(*x1, *x2) > 1.0 - EPS).then_some(false)
        }
        (
            Torus {
                center: c1,
                axis: x1,
                major: m1,
                minor: n1,
                ..
            },
            Torus {
                center: c2,
                axis: x2,
                major: m2,
                minor: n2,
                ..
            },
        ) => {
            if v3::dist(*c1, *c2) > EPS
                || (m1 - m2).abs() > EPS
                || (n1 - n2).abs() > EPS
                || (v3::dot(*x1, *x2).abs() - 1.0).abs() > EPS
            {
                return None;
            }
            Some(false)
        }
        _ => None,
    }
}

/// Do `(o1, d1)` and `(o2, d2)` describe the same infinite line?
fn same_axis_line(o1: V3, d1: V3, o2: V3, d2: V3) -> bool {
    if (v3::dot(d1, d2).abs() - 1.0).abs() > EPS {
        return false;
    }
    // The offset between origins must be along the shared direction.
    let off = v3::sub(o2, o1);
    v3::norm(v3::sub(off, v3::scale(d1, v3::dot(off, d1)))) <= EPS
}

// ---------------------------------------------------------------------------
// plane pairs
// ---------------------------------------------------------------------------

fn plane_plane(a: &Surface, b: &Surface) -> SsiResult {
    let (
        Surface::Plane {
            origin: o1,
            normal: n1,
            ..
        },
        Surface::Plane {
            origin: o2,
            normal: n2,
            ..
        },
    ) = (a, b)
    else {
        return SsiResult::Unknown;
    };

    let dir = v3::cross(*n1, *n2);
    let Some(dir) = v3::normalize(dir) else {
        // Parallel. Coincidence was already ruled out above, so they are apart.
        return SsiResult::Disjoint;
    };

    // A point on both planes: solve the 2×2 system in the plane spanned by the
    // two normals.
    let (d1, d2) = (v3::dot(*n1, *o1), v3::dot(*n2, *o2));
    let n1n2 = v3::dot(*n1, *n2);
    let det = 1.0 - n1n2 * n1n2;
    if det.abs() < 1e-300 {
        return SsiResult::Unknown;
    }
    let c1 = (d1 - d2 * n1n2) / det;
    let c2 = (d2 - d1 * n1n2) / det;
    let origin = v3::add(v3::scale(*n1, c1), v3::scale(*n2, c2));

    SsiResult::Curves(vec![Curve3d::Line { origin, dir }])
}

fn plane_sphere(p: &Surface, s: &Surface) -> SsiResult {
    let (Surface::Plane { origin, normal, .. }, Surface::Sphere { center, radius, .. }) = (p, s)
    else {
        return SsiResult::Unknown;
    };
    let signed = v3::dot(v3::sub(*center, *origin), *normal);
    let d = signed.abs();
    let foot = v3::sub(*center, v3::scale(*normal, signed));

    if d > radius + EPS {
        return SsiResult::Disjoint;
    }
    if (d - radius).abs() <= EPS {
        return SsiResult::Curves(vec![Curve3d::Point(foot)]);
    }
    let r = (radius * radius - d * d).max(0.0).sqrt();
    SsiResult::Curves(vec![circle(foot, *normal, r)])
}

fn plane_cylinder(p: &Surface, c: &Surface) -> SsiResult {
    let (
        Surface::Plane {
            origin: po, normal, ..
        },
        Surface::Cylinder {
            origin: co,
            axis,
            radius,
            ..
        },
    ) = (p, c)
    else {
        return SsiResult::Unknown;
    };

    let cos = v3::dot(*normal, *axis);

    // Perpendicular to the axis → a circle.
    if (cos.abs() - 1.0).abs() <= EPS {
        let t = v3::dot(v3::sub(*po, *co), *normal) / cos;
        let center = v3::add(*co, v3::scale(*axis, t));
        return SsiResult::Curves(vec![circle(center, *axis, *radius)]);
    }

    // Parallel to the axis → zero, one or two rulings.
    if cos.abs() <= EPS {
        // Distance from the axis line to the plane.
        let d = v3::dot(v3::sub(*co, *po), *normal);
        if d.abs() > radius + EPS {
            return SsiResult::Disjoint;
        }
        // Foot of the axis on the plane, then step along the in-plane direction
        // perpendicular to the axis.
        let foot = v3::sub(*co, v3::scale(*normal, d));
        let along = v3::normalize(v3::cross(*normal, *axis)).unwrap_or([1.0, 0.0, 0.0]);
        if (d.abs() - radius).abs() <= EPS {
            return SsiResult::Curves(vec![Curve3d::Line {
                origin: foot,
                dir: *axis,
            }]);
        }
        let h = (radius * radius - d * d).max(0.0).sqrt();
        return SsiResult::Curves(vec![
            Curve3d::Line {
                origin: v3::add(foot, v3::scale(along, h)),
                dir: *axis,
            },
            Curve3d::Line {
                origin: v3::sub(foot, v3::scale(along, h)),
                dir: *axis,
            },
        ]);
    }

    // Oblique → an ellipse. Its minor axis is the cylinder's radius along the
    // in-plane direction perpendicular to the axis; its major stretches by
    // 1/|cos| along the plane's projection of the axis.
    let minor_dir = v3::normalize(v3::cross(*normal, *axis)).unwrap_or([1.0, 0.0, 0.0]);
    let major_dir = v3::normalize(v3::cross(minor_dir, *normal)).unwrap_or([0.0, 1.0, 0.0]);
    // Centre: where the axis pierces the plane.
    let t = v3::dot(v3::sub(*po, *co), *normal) / cos;
    let center = v3::add(*co, v3::scale(*axis, t));

    SsiResult::Curves(vec![Curve3d::Ellipse {
        center,
        x_dir: major_dir,
        y_dir: minor_dir,
        a: radius / cos.abs(),
        b: *radius,
    }])
}

fn plane_cone(p: &Surface, c: &Surface) -> SsiResult {
    let (
        Surface::Plane {
            origin: po, normal, ..
        },
        Surface::Cone {
            apex,
            axis,
            half_angle,
            ..
        },
    ) = (p, c)
    else {
        return SsiResult::Unknown;
    };

    let cos = v3::dot(*normal, *axis);
    let apex_offset = v3::dot(v3::sub(*apex, *po), *normal);

    // Through the apex: a degenerate conic (a point, or a pair of rulings).
    // Which one needs the plane's inclination against the cone's; that case is
    // left to the mesh kernel rather than guessed at.
    if apex_offset.abs() <= EPS {
        return SsiResult::Unknown;
    }

    // Perpendicular to the axis → a circle.
    if (cos.abs() - 1.0).abs() <= EPS {
        let h = apex_offset.abs() / cos.abs();
        let center = v3::add(
            *apex,
            v3::scale(*axis, if apex_offset * cos < 0.0 { h } else { -h }),
        );
        let r = h * half_angle.tan();
        return SsiResult::Curves(vec![circle(center, *axis, r)]);
    }

    // The conic is an ellipse only when the plane cuts every ruling, i.e. its
    // angle to the axis exceeds the half-angle. At equality it is a parabola and
    // below it a hyperbola — both unbounded, neither in `Curve3d`, so both are
    // honestly `Unknown` rather than approximated by an ellipse.
    let plane_tilt = cos.abs().acos(); // angle between the normal and the axis
    let axis_to_plane = std::f64::consts::FRAC_PI_2 - plane_tilt;
    if axis_to_plane <= *half_angle + EPS {
        return SsiResult::Unknown;
    }

    // Ellipse: intersect the two extreme rulings in the plane containing the
    // axis and the plane's steepest-descent direction.
    let minor_dir = match v3::normalize(v3::cross(*normal, *axis)) {
        Some(d) => d,
        None => return SsiResult::Unknown,
    };
    let steepest = match v3::normalize(v3::cross(minor_dir, *normal)) {
        Some(d) => d,
        None => return SsiResult::Unknown,
    };
    // In the (axis, steepest) plane the two extreme rulings leave the apex at
    // ±half_angle from the axis.
    let in_plane_axis =
        match v3::normalize(v3::sub(*axis, v3::scale(*normal, v3::dot(*axis, *normal)))) {
            Some(d) => d,
            None => return SsiResult::Unknown,
        };
    let _ = steepest;
    let (sh, ch) = half_angle.sin_cos();
    let perp = match v3::normalize(v3::cross(minor_dir, *axis)) {
        Some(d) => d,
        None => return SsiResult::Unknown,
    };

    let mut ends: Vec<V3> = Vec::with_capacity(2);
    for sign in [1.0f64, -1.0] {
        let ruling = v3::add(v3::scale(*axis, ch), v3::scale(perp, sign * sh));
        let denom = v3::dot(ruling, *normal);
        if denom.abs() < 1e-12 {
            return SsiResult::Unknown;
        }
        let t = -apex_offset / denom;
        ends.push(v3::add(*apex, v3::scale(ruling, t)));
    }

    let center = v3::scale(v3::add(ends[0], ends[1]), 0.5);
    let a = 0.5 * v3::dist(ends[0], ends[1]);
    if !(a > EPS) {
        return SsiResult::Unknown;
    }
    let x_dir = match v3::normalize(v3::sub(ends[1], center)) {
        Some(d) => d,
        None => return SsiResult::Unknown,
    };
    // The semi-minor axis is the cone's radius at the centre's height, measured
    // perpendicular to the major axis and inside the plane.
    let h = v3::dot(v3::sub(center, *apex), *axis);
    let b2 = (h * half_angle.tan()).powi(2)
        - v3::norm(v3::sub(v3::sub(center, *apex), v3::scale(*axis, h))).powi(2);
    if !(b2 > 0.0) {
        return SsiResult::Unknown;
    }
    let _ = in_plane_axis;

    SsiResult::Curves(vec![Curve3d::Ellipse {
        center,
        x_dir,
        y_dir: minor_dir,
        a,
        b: b2.sqrt(),
    }])
}

fn plane_torus(p: &Surface, t: &Surface) -> SsiResult {
    let (
        Surface::Plane {
            origin: po, normal, ..
        },
        Surface::Torus {
            center,
            axis,
            major,
            minor,
            ..
        },
    ) = (p, t)
    else {
        return SsiResult::Unknown;
    };

    let cos = v3::dot(*normal, *axis);

    // Perpendicular to the axis → up to two concentric circles.
    if (cos.abs() - 1.0).abs() <= EPS {
        let h = v3::dot(v3::sub(*po, *center), *normal) / cos;
        if h.abs() > minor + EPS {
            return SsiResult::Disjoint;
        }
        let plane_center = v3::add(*center, v3::scale(*axis, h));
        let d = (minor * minor - h * h).max(0.0).sqrt();
        if d <= EPS {
            // Tangent at the extreme of the tube: one circle of radius `major`.
            return SsiResult::Curves(vec![circle(plane_center, *axis, *major)]);
        }
        let mut out = vec![circle(plane_center, *axis, major + d)];
        if major - d > EPS {
            out.push(circle(plane_center, *axis, major - d));
        }
        return SsiResult::Curves(out);
    }

    // Containing the axis → two circles of the tube radius, at ±major.
    if cos.abs() <= EPS && v3::dot(v3::sub(*center, *po), *normal).abs() <= EPS {
        let radial = match v3::normalize(v3::cross(*normal, *axis)) {
            Some(d) => d,
            None => return SsiResult::Unknown,
        };
        return SsiResult::Curves(vec![
            circle(v3::add(*center, v3::scale(radial, *major)), *normal, *minor),
            circle(v3::sub(*center, v3::scale(radial, *major)), *normal, *minor),
        ]);
    }

    // Everything else is a quartic (Villarceau circles among them). Not guessed.
    SsiResult::Unknown
}

// ---------------------------------------------------------------------------
// coaxial surfaces of revolution
// ---------------------------------------------------------------------------

/// A surface of revolution, as its profile in the `(ρ, h)` half-plane of its own
/// axis — `ρ` the distance from the axis, `h` the height along it.
///
/// This is what makes the coaxial case exact and general at once. Two surfaces
/// that share an axis meet in *circles*, one per crossing of their profiles, so
/// a three-dimensional quartic problem becomes a two-dimensional conic one that
/// is already solved. The pairs it covers — a torus with a cylinder, a sphere, a
/// cone, another torus — have no closed form in general position, and every one
/// of them is routine when coaxial: an O-ring groove, a rounded rim, a bore
/// through a doughnut.
#[derive(Debug, Clone, Copy)]
enum Profile {
    /// `ρ = radius`, a cylinder.
    Radial { rho: f64 },
    /// `(ρ − rho0)² + (h − h0)² = r²`: a torus tube, or a sphere with `rho0 = 0`.
    Round { rho0: f64, h0: f64, r: f64 },
    /// `ρ = |h − apex| · tan(half_angle)`, a cone.
    Slanted { apex: f64, tan: f64 },
}

/// The profile of `s` about `axis` through `origin`, if it is a surface of
/// revolution about exactly that axis.
fn profile(s: &Surface, origin: V3, axis: V3) -> Option<Profile> {
    let height = |p: V3| v3::dot(v3::sub(p, origin), axis);
    let on_axis = |p: V3| {
        let d = v3::sub(p, origin);
        v3::norm(v3::sub(d, v3::scale(axis, v3::dot(d, axis)))) <= EPS
    };
    let aligned = |a: V3| (v3::dot(a, axis).abs() - 1.0).abs() <= EPS;
    match s {
        Surface::Cylinder {
            origin: o,
            axis: a,
            radius,
            ..
        } if aligned(*a) && on_axis(*o) => Some(Profile::Radial { rho: *radius }),
        Surface::Sphere { center, radius, .. } if on_axis(*center) => Some(Profile::Round {
            rho0: 0.0,
            h0: height(*center),
            r: *radius,
        }),
        Surface::Cone {
            apex,
            axis: a,
            half_angle,
            ..
        } if aligned(*a) && on_axis(*apex) => Some(Profile::Slanted {
            apex: height(*apex),
            tan: half_angle.tan(),
        }),
        Surface::Torus {
            center,
            axis: a,
            major,
            minor,
            ..
        } if aligned(*a) && on_axis(*center) => Some(Profile::Round {
            rho0: *major,
            h0: height(*center),
            r: *minor,
        }),
        _ => None,
    }
}

/// The axis a surface revolves about, as `(a point on it, its direction)`.
fn revolution_axis(s: &Surface) -> Option<(V3, V3)> {
    match s {
        Surface::Cylinder { origin, axis, .. } => Some((*origin, *axis)),
        Surface::Cone { apex, axis, .. } => Some((*apex, *axis)),
        Surface::Torus { center, axis, .. } => Some((*center, *axis)),
        // A sphere revolves about *every* line through its centre, so it takes
        // the other surface's axis rather than offering one.
        _ => None,
    }
}

/// Two surfaces of revolution sharing an axis.
fn coaxial(a: &Surface, b: &Surface) -> SsiResult {
    let Some((origin, axis)) = revolution_axis(a).or_else(|| revolution_axis(b)) else {
        return SsiResult::Unknown;
    };
    let (Some(pa), Some(pb)) = (profile(a, origin, axis), profile(b, origin, axis)) else {
        // Not coaxial — a general-position quartic, which is not guessed at.
        return SsiResult::Unknown;
    };

    let mut hits = profile_meet(pa, pb);
    // A tangency reached from both branches of a cone, or from a symmetric
    // pair, arrives twice; one circle is one circle.
    hits.retain(|&(rho, _)| rho > EPS);
    hits.sort_by(|x, y| {
        x.1.partial_cmp(&y.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal))
    });
    hits.dedup_by(|x, y| (x.0 - y.0).abs() <= EPS && (x.1 - y.1).abs() <= EPS);

    if hits.is_empty() {
        return SsiResult::Disjoint;
    }
    SsiResult::Curves(
        hits.into_iter()
            .map(|(rho, h)| circle(v3::add(origin, v3::scale(axis, h)), axis, rho))
            .collect(),
    )
}

/// Where two profiles cross, as `(ρ, h)` pairs.
fn profile_meet(a: Profile, b: Profile) -> Vec<(f64, f64)> {
    use Profile::*;
    match (a, b) {
        (Radial { rho }, Round { rho0, h0, r }) | (Round { rho0, h0, r }, Radial { rho }) => {
            // A vertical line through a circle.
            let dx = rho - rho0;
            let inside = r * r - dx * dx;
            if inside < -EPS {
                return Vec::new();
            }
            let dh = inside.max(0.0).sqrt();
            vec![(rho, h0 + dh), (rho, h0 - dh)]
        }
        (Radial { rho }, Slanted { apex, tan }) | (Slanted { apex, tan }, Radial { rho }) => {
            if tan.abs() <= EPS {
                return Vec::new();
            }
            let dh = rho / tan;
            vec![(rho, apex + dh), (rho, apex - dh)]
        }
        (Radial { rho: p }, Radial { rho: q }) => {
            // Same cylinder or none; identity is handled before this is called.
            let _ = (p, q);
            Vec::new()
        }
        (
            Round {
                rho0: p0,
                h0: q0,
                r: r0,
            },
            Round {
                rho0: p1,
                h0: q1,
                r: r1,
            },
        ) => {
            // Two circles in the profile plane — the same computation as two
            // circles anywhere.
            let (dx, dy) = (p1 - p0, q1 - q0);
            let d = dx.hypot(dy);
            if d <= EPS || d > r0 + r1 + EPS || d < (r0 - r1).abs() - EPS {
                return Vec::new();
            }
            let x = (d * d + r0 * r0 - r1 * r1) / (2.0 * d);
            let inside = r0 * r0 - x * x;
            if inside < -EPS {
                return Vec::new();
            }
            let y = inside.max(0.0).sqrt();
            let (ux, uy) = (dx / d, dy / d);
            let (bx, by) = (p0 + ux * x, q0 + uy * x);
            vec![(bx - uy * y, by + ux * y), (bx + uy * y, by - ux * y)]
        }
        (Round { rho0, h0, r }, Slanted { apex, tan })
        | (Slanted { apex, tan }, Round { rho0, h0, r }) => {
            // ρ = ±(h − apex)·tan into the circle gives a quadratic in h, once
            // per branch of the cone.
            let mut out = Vec::new();
            for sign in [1.0f64, -1.0] {
                let t = sign * tan;
                // ((h − apex)t − rho0)² + (h − h0)² = r²
                let qa = t * t + 1.0;
                let qb = -2.0 * (t * (apex * t + rho0) + h0);
                let qc = (apex * t + rho0).powi(2) + h0 * h0 - r * r;
                let disc = qb * qb - 4.0 * qa * qc;
                if disc < -EPS || qa.abs() <= EPS {
                    continue;
                }
                let root = disc.max(0.0).sqrt();
                for h in [(-qb + root) / (2.0 * qa), (-qb - root) / (2.0 * qa)] {
                    let rho = (h - apex) * t;
                    if rho > EPS {
                        out.push((rho, h));
                    }
                }
            }
            out
        }
        (Slanted { .. }, Slanted { .. }) => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// quadric pairs
// ---------------------------------------------------------------------------

fn sphere_sphere(a: &Surface, b: &Surface) -> SsiResult {
    let (
        Surface::Sphere {
            center: c1,
            radius: r1,
            ..
        },
        Surface::Sphere {
            center: c2,
            radius: r2,
            ..
        },
    ) = (a, b)
    else {
        return SsiResult::Unknown;
    };
    let d = v3::dist(*c1, *c2);
    if d <= EPS {
        // Concentric; coincidence was ruled out, so different radii.
        return SsiResult::Disjoint;
    }
    if d > r1 + r2 + EPS || d < (r1 - r2).abs() - EPS {
        return SsiResult::Disjoint;
    }
    let axis = v3::normalize(v3::sub(*c2, *c1)).unwrap();
    // Distance from c1 to the radical plane.
    let x = (d * d + r1 * r1 - r2 * r2) / (2.0 * d);
    let h2 = r1 * r1 - x * x;
    let center = v3::add(*c1, v3::scale(axis, x));
    if h2 <= EPS * EPS {
        return SsiResult::Curves(vec![Curve3d::Point(center)]);
    }
    SsiResult::Curves(vec![circle(center, axis, h2.sqrt())])
}

fn sphere_cylinder(s: &Surface, c: &Surface) -> SsiResult {
    let (
        Surface::Sphere {
            center: sc,
            radius: sr,
            ..
        },
        Surface::Cylinder {
            origin: co,
            axis,
            radius: cr,
            ..
        },
    ) = (s, c)
    else {
        return SsiResult::Unknown;
    };

    // Only the coaxial case has a closed form here: the centre must lie on the
    // cylinder's axis, or the intersection is a quartic.
    let off = v3::sub(*sc, *co);
    let radial = v3::sub(off, v3::scale(*axis, v3::dot(off, *axis)));
    if v3::norm(radial) > EPS {
        return SsiResult::Unknown;
    }

    if *sr < cr - EPS {
        return SsiResult::Disjoint;
    }
    if (sr - cr).abs() <= EPS {
        // Tangent along the sphere's equator.
        return SsiResult::Curves(vec![circle(*sc, *axis, *cr)]);
    }
    let h = (sr * sr - cr * cr).max(0.0).sqrt();
    SsiResult::Curves(vec![
        circle(v3::add(*sc, v3::scale(*axis, h)), *axis, *cr),
        circle(v3::sub(*sc, v3::scale(*axis, h)), *axis, *cr),
    ])
}

fn cylinder_cylinder(a: &Surface, b: &Surface) -> SsiResult {
    let (
        Surface::Cylinder {
            origin: o1,
            axis: a1,
            radius: r1,
            ..
        },
        Surface::Cylinder {
            origin: o2,
            axis: a2,
            radius: r2,
            ..
        },
    ) = (a, b)
    else {
        return SsiResult::Unknown;
    };

    let parallel = (v3::dot(*a1, *a2).abs() - 1.0).abs() <= EPS;
    if parallel {
        // Coaxial with different radii — nested, never touching.
        if same_axis_line(*o1, *a1, *o2, *a2) {
            return SsiResult::Disjoint;
        }
        // Parallel but offset: the problem reduces to two circles in the
        // common cross-section, and each solution lifts to a ruling.
        let off = v3::sub(*o2, *o1);
        let perp = v3::sub(off, v3::scale(*a1, v3::dot(off, *a1)));
        let d = v3::norm(perp);
        if d > r1 + r2 + EPS || d < (r1 - r2).abs() - EPS {
            return SsiResult::Disjoint;
        }
        let u = v3::normalize(perp).unwrap();
        let w = v3::cross(*a1, u);
        let x = (d * d + r1 * r1 - r2 * r2) / (2.0 * d);
        let h2 = r1 * r1 - x * x;
        let base = v3::add(*o1, v3::scale(u, x));
        if h2 <= EPS * EPS {
            return SsiResult::Curves(vec![Curve3d::Line {
                origin: base,
                dir: *a1,
            }]);
        }
        let h = h2.sqrt();
        return SsiResult::Curves(vec![
            Curve3d::Line {
                origin: v3::add(base, v3::scale(w, h)),
                dir: *a1,
            },
            Curve3d::Line {
                origin: v3::sub(base, v3::scale(w, h)),
                dir: *a1,
            },
        ]);
    }

    // Crossing axes of *equal radius* that actually meet: the classic
    // two-ellipse case (a Steinmetz solid's seam). Anything else is a quartic
    // space curve with no closed form, and is left alone.
    if (r1 - r2).abs() > EPS {
        return SsiResult::Unknown;
    }
    let Some((meet, ok)) = closest_approach(*o1, *a1, *o2, *a2) else {
        return SsiResult::Unknown;
    };
    if !ok {
        return SsiResult::Unknown; // skew axes — quartic
    }

    // The two ellipses lie in the planes bisecting the axes.
    let mut out = Vec::with_capacity(2);
    for sign in [1.0f64, -1.0] {
        let bis = match v3::normalize(v3::add(*a1, v3::scale(*a2, sign))) {
            Some(d) => d,
            None => continue,
        };
        // Semi-minor is the common radius; semi-major stretches by 1/sin(θ/2)
        // where θ is the angle between the axes on this side.
        let half = 0.5 * v3::dot(*a1, v3::scale(*a2, sign)).clamp(-1.0, 1.0).acos();
        let s = half.sin();
        if s < EPS {
            continue;
        }
        let minor_dir = match v3::normalize(v3::cross(*a1, *a2)) {
            Some(d) => d,
            None => continue,
        };
        out.push(Curve3d::Ellipse {
            center: meet,
            x_dir: bis,
            y_dir: minor_dir,
            a: r1 / s,
            b: *r1,
        });
    }
    if out.is_empty() {
        return SsiResult::Unknown;
    }
    SsiResult::Curves(out)
}

fn cylinder_cone(c: &Surface, k: &Surface) -> SsiResult {
    let (
        Surface::Cylinder {
            origin: co,
            axis: ca,
            radius,
            ..
        },
        Surface::Cone {
            apex,
            axis: ka,
            half_angle,
            ..
        },
    ) = (c, k)
    else {
        return SsiResult::Unknown;
    };
    // Coaxial only.
    if !same_axis_line(*co, *ca, *apex, *ka) {
        return SsiResult::Unknown;
    }
    // The cone reaches the cylinder's radius at one height along its own axis.
    let t = half_angle.tan();
    if !(t > 0.0) {
        return SsiResult::Unknown;
    }
    let h = radius / t;
    SsiResult::Curves(vec![circle(
        v3::add(*apex, v3::scale(*ka, h)),
        *ka,
        *radius,
    )])
}

fn cone_cone(a: &Surface, b: &Surface) -> SsiResult {
    let (
        Surface::Cone {
            apex: p1,
            axis: a1,
            half_angle: h1,
            ..
        },
        Surface::Cone {
            apex: p2,
            axis: a2,
            half_angle: h2,
            ..
        },
    ) = (a, b)
    else {
        return SsiResult::Unknown;
    };
    // Coaxial only. Same nappe direction and different angles → they meet only
    // at the shared apex; opposite directions → also only the apex.
    if !same_axis_line(*p1, *a1, *p2, *a2) {
        return SsiResult::Unknown;
    }
    if v3::dist(*p1, *p2) <= EPS {
        if (h1 - h2).abs() <= EPS && v3::dot(*a1, *a2) > 1.0 - EPS {
            return SsiResult::Coincident { opposite: false };
        }
        return SsiResult::Curves(vec![Curve3d::Point(*p1)]);
    }
    // Coaxial with offset apexes: the radii match at one height when the
    // half-angles differ, giving a circle.
    let (t1, t2) = (h1.tan(), h2.tan());
    let along = v3::dot(v3::sub(*p2, *p1), *a1);
    let s2 = if v3::dot(*a1, *a2) > 0.0 { 1.0 } else { -1.0 };
    // r1(x) = x·t1 measured from p1; r2(x) = (x − along)·s2·t2.
    let denom = t1 - s2 * t2;
    if denom.abs() < EPS {
        return SsiResult::Unknown; // parallel rulings — no isolated circle
    }
    let x = -s2 * t2 * along / denom;
    let r = x * t1;
    if !(r > EPS) {
        return SsiResult::Curves(vec![Curve3d::Point(v3::add(*p1, v3::scale(*a1, x)))]);
    }
    SsiResult::Curves(vec![circle(v3::add(*p1, v3::scale(*a1, x)), *a1, r)])
}

/// Midpoint of the shortest segment between two lines, plus whether they
/// actually meet (as opposed to passing by).
fn closest_approach(o1: V3, d1: V3, o2: V3, d2: V3) -> Option<(V3, bool)> {
    let w = v3::sub(o1, o2);
    let (a, b, c) = (v3::dot(d1, d1), v3::dot(d1, d2), v3::dot(d2, d2));
    let (d, e) = (v3::dot(d1, w), v3::dot(d2, w));
    let denom = a * c - b * b;
    if denom.abs() < 1e-12 {
        return None;
    }
    let s = (b * e - c * d) / denom;
    let t = (a * e - b * d) / denom;
    let p1 = v3::add(o1, v3::scale(d1, s));
    let p2 = v3::add(o2, v3::scale(d2, t));
    let gap = v3::dist(p1, p2);
    Some((v3::scale(v3::add(p1, p2), 0.5), gap <= EPS))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    /// Sample a reported curve and confirm every point really is on *both*
    /// surfaces. This is the test that matters: a closed form that is subtly
    /// wrong produces a plausible curve in the wrong place, and only measuring
    /// against both surfaces catches it.
    fn assert_on_both(a: &Surface, b: &Surface, result: &SsiResult, tol: f64, what: &str) {
        let SsiResult::Curves(curves) = result else {
            panic!("{what}: expected curves, got {result:?}");
        };
        assert!(!curves.is_empty(), "{what}: empty curve list");
        for c in curves {
            let samples: Vec<V3> = match c {
                Curve3d::Point(p) => vec![*p],
                Curve3d::Line { .. } => (-20..=20).map(|i| c.point(i as f64 * 0.5)).collect(),
                _ => (0..64)
                    .map(|i| c.point(PI * 2.0 * i as f64 / 64.0))
                    .collect(),
            };
            for p in samples {
                assert!(
                    a.distance(p) < tol,
                    "{what}: {} point {p:?} is {} off surface A",
                    c.kind(),
                    a.distance(p)
                );
                assert!(
                    b.distance(p) < tol,
                    "{what}: {} point {p:?} is {} off surface B",
                    c.kind(),
                    b.distance(p)
                );
            }
        }
    }

    // -- identity ----------------------------------------------------------

    #[test]
    fn coaxial_equal_radius_cylinders_are_the_same_surface() {
        // The kernel's named degeneracy — "two identical primitives translated
        // along an axis". Numerically it is a sliver factory; analytically it is
        // a struct comparison.
        let a = Surface::cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 2.0);
        let b = Surface::cylinder([0.0, 0.0, 7.5], [0.0, 0.0, 1.0], 2.0);
        assert_eq!(ssi(&a, &b), SsiResult::Coincident { opposite: false });

        // And the *reversed* axis is still the same cylinder.
        let c = Surface::cylinder([0.0, 0.0, -3.0], [0.0, 0.0, -1.0], 2.0);
        assert_eq!(ssi(&a, &c), SsiResult::Coincident { opposite: false });
    }

    #[test]
    fn coaxial_cylinders_of_different_radius_are_disjoint() {
        let a = Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0);
        let b = Surface::cylinder([0.0, 0.0, 5.0], [0.0, 0.0, 1.0], 3.0);
        assert_eq!(ssi(&a, &b), SsiResult::Disjoint);
    }

    #[test]
    fn coincident_planes_record_whether_they_face_the_same_way() {
        let a = Surface::plane([0.0; 3], [0.0, 0.0, 1.0]);
        let same = Surface::plane([5.0, -3.0, 0.0], [0.0, 0.0, 1.0]);
        let flipped = Surface::plane([1.0, 1.0, 0.0], [0.0, 0.0, -1.0]);
        assert_eq!(ssi(&a, &same), SsiResult::Coincident { opposite: false });
        assert_eq!(ssi(&a, &flipped), SsiResult::Coincident { opposite: true });
    }

    #[test]
    fn identical_spheres_are_coincident_whatever_their_frames() {
        let a = Surface::sphere([1.0, 2.0, 3.0], 4.0);
        let b = Surface::sphere([1.0, 2.0, 3.0], 4.0)
            .with_axis([1.0, 1.0, 0.0])
            .with_x_dir([0.0, 0.0, 1.0]);
        assert_eq!(ssi(&a, &b), SsiResult::Coincident { opposite: false });
    }

    #[test]
    fn cones_sharing_an_apex_but_opening_opposite_ways_are_not_coincident() {
        let up = Surface::cone([0.0; 3], [0.0, 0.0, 1.0], FRAC_PI_4);
        let down = Surface::cone([0.0; 3], [0.0, 0.0, -1.0], FRAC_PI_4);
        assert_ne!(ssi(&up, &down), SsiResult::Coincident { opposite: false });
    }

    // -- plane pairs -------------------------------------------------------

    #[test]
    fn two_planes_meet_in_a_line() {
        let a = Surface::plane([0.0; 3], [0.0, 0.0, 1.0]);
        let b = Surface::plane([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let r = ssi(&a, &b);
        assert_on_both(&a, &b, &r, 1e-12, "plane/plane");
        let SsiResult::Curves(c) = &r else {
            unreachable!()
        };
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].kind(), "line");
    }

    #[test]
    fn parallel_planes_are_disjoint() {
        let a = Surface::plane([0.0; 3], [0.0, 0.0, 1.0]);
        let b = Surface::plane([0.0, 0.0, 4.0], [0.0, 0.0, 1.0]);
        assert_eq!(ssi(&a, &b), SsiResult::Disjoint);
    }

    #[test]
    fn plane_and_sphere_meet_in_a_circle_touch_at_a_point_or_miss() {
        let s = Surface::sphere([0.0, 0.0, 0.0], 5.0);
        for &(h, want) in &[(0.0, "circle"), (3.0, "circle"), (5.0, "point")] {
            let p = Surface::plane([0.0, 0.0, h], [0.0, 0.0, 1.0]);
            let r = ssi(&p, &s);
            assert_on_both(&p, &s, &r, 1e-9, &format!("plane/sphere at h={h}"));
            let SsiResult::Curves(c) = &r else {
                unreachable!()
            };
            assert_eq!(c[0].kind(), want, "h = {h}");
        }
        let far = Surface::plane([0.0, 0.0, 6.0], [0.0, 0.0, 1.0]);
        assert_eq!(ssi(&far, &s), SsiResult::Disjoint);
    }

    #[test]
    fn plane_and_sphere_agree_on_the_circle_radius() {
        let s = Surface::sphere([0.0; 3], 5.0);
        let p = Surface::plane([0.0, 0.0, 3.0], [0.0, 0.0, 1.0]);
        let SsiResult::Curves(c) = ssi(&p, &s) else {
            panic!()
        };
        match &c[0] {
            Curve3d::Circle { radius, center, .. } => {
                assert!((radius - 4.0).abs() < 1e-12, "radius {radius}");
                assert!(v3::dist(*center, [0.0, 0.0, 3.0]) < 1e-12);
            }
            other => panic!("expected a circle, got {}", other.kind()),
        }
    }

    #[test]
    fn plane_and_cylinder_give_circle_rulings_or_ellipse() {
        let c = Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0);

        let perp = Surface::plane([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]);
        let r = ssi(&perp, &c);
        assert_on_both(&perp, &c, &r, 1e-9, "perpendicular");
        let SsiResult::Curves(x) = &r else {
            unreachable!()
        };
        assert_eq!(x[0].kind(), "circle");

        let through = Surface::plane([0.0; 3], [1.0, 0.0, 0.0]);
        let r = ssi(&through, &c);
        assert_on_both(&through, &c, &r, 1e-9, "through the axis");
        let SsiResult::Curves(x) = &r else {
            unreachable!()
        };
        assert_eq!(x.len(), 2, "a plane through the axis cuts two rulings");

        let tangent = Surface::plane([2.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let SsiResult::Curves(x) = ssi(&tangent, &c) else {
            panic!()
        };
        assert_eq!(x.len(), 1, "a tangent plane touches along one ruling");

        let missing = Surface::plane([3.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert_eq!(ssi(&missing, &c), SsiResult::Disjoint);

        let oblique = Surface::plane([0.0; 3], v3::normalize([0.0, 1.0, 1.0]).unwrap());
        let r = ssi(&oblique, &c);
        assert_on_both(&oblique, &c, &r, 1e-9, "oblique");
        let SsiResult::Curves(x) = &r else {
            unreachable!()
        };
        assert_eq!(x[0].kind(), "ellipse");
    }

    #[test]
    fn an_oblique_cut_of_a_cylinder_has_the_expected_axes() {
        // At 45° the major axis is r·√2 and the minor is r.
        let c = Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0);
        let p = Surface::plane([0.0; 3], v3::normalize([0.0, 1.0, 1.0]).unwrap());
        let SsiResult::Curves(x) = ssi(&p, &c) else {
            panic!()
        };
        match &x[0] {
            Curve3d::Ellipse { a, b, .. } => {
                assert!((a - 2.0 * 2f64.sqrt()).abs() < 1e-12, "a = {a}");
                assert!((b - 2.0).abs() < 1e-12, "b = {b}");
            }
            other => panic!("expected an ellipse, got {}", other.kind()),
        }
    }

    #[test]
    fn plane_and_cone_give_a_circle_perpendicular_and_an_ellipse_when_it_cuts_every_ruling() {
        let k = Surface::cone([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], FRAC_PI_4);

        let perp = Surface::plane([0.0, 0.0, 3.0], [0.0, 0.0, 1.0]);
        let r = ssi(&perp, &k);
        assert_on_both(&perp, &k, &r, 1e-9, "cone/perpendicular");
        let SsiResult::Curves(x) = &r else {
            unreachable!()
        };
        match &x[0] {
            Curve3d::Circle { radius, .. } => assert!((radius - 3.0).abs() < 1e-12),
            other => panic!("expected a circle, got {}", other.kind()),
        }

        // A gentle tilt still cuts every ruling of a 45° cone.
        let tilt = Surface::plane([0.0, 0.0, 4.0], v3::normalize([0.2, 0.0, 1.0]).unwrap());
        let r = ssi(&tilt, &k);
        assert_on_both(&tilt, &k, &r, 1e-7, "cone/tilted");
    }

    #[test]
    fn a_plane_parallel_to_a_cone_ruling_is_unknown_not_approximated() {
        // Parabola and hyperbola are unbounded and have no `Curve3d`. Reporting
        // an ellipse would be worse than reporting nothing.
        let k = Surface::cone([0.0; 3], [0.0, 0.0, 1.0], FRAC_PI_4);
        let parabolic = Surface::plane([0.0, 0.0, 2.0], v3::normalize([1.0, 0.0, 1.0]).unwrap());
        assert_eq!(ssi(&parabolic, &k), SsiResult::Unknown);
        let hyperbolic = Surface::plane([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert_eq!(ssi(&hyperbolic, &k), SsiResult::Unknown);
    }

    #[test]
    fn plane_and_torus_give_concentric_circles_or_tube_circles() {
        let t = Surface::torus([0.0; 3], [0.0, 0.0, 1.0], 5.0, 1.0);

        let mid = Surface::plane([0.0; 3], [0.0, 0.0, 1.0]);
        let r = ssi(&mid, &t);
        assert_on_both(&mid, &t, &r, 1e-9, "torus/equator");
        let SsiResult::Curves(x) = &r else {
            unreachable!()
        };
        assert_eq!(x.len(), 2, "the equatorial plane cuts two circles");

        let top = Surface::plane([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]);
        let SsiResult::Curves(x) = ssi(&top, &t) else {
            panic!()
        };
        assert_eq!(x.len(), 1, "the tangent plane touches one circle");

        let above = Surface::plane([0.0, 0.0, 2.0], [0.0, 0.0, 1.0]);
        assert_eq!(ssi(&above, &t), SsiResult::Disjoint);

        let axial = Surface::plane([0.0; 3], [0.0, 1.0, 0.0]);
        let r = ssi(&axial, &t);
        assert_on_both(&axial, &t, &r, 1e-9, "torus/axial");
        let SsiResult::Curves(x) = &r else {
            unreachable!()
        };
        assert_eq!(
            x.len(),
            2,
            "a plane through the axis cuts both tube circles"
        );
    }

    // -- quadric pairs -----------------------------------------------------

    #[test]
    fn two_spheres_meet_in_a_circle() {
        let a = Surface::sphere([0.0; 3], 3.0);
        let b = Surface::sphere([4.0, 0.0, 0.0], 3.0);
        let r = ssi(&a, &b);
        assert_on_both(&a, &b, &r, 1e-12, "sphere/sphere");
        let SsiResult::Curves(c) = &r else {
            unreachable!()
        };
        match &c[0] {
            Curve3d::Circle { center, radius, .. } => {
                assert!(v3::dist(*center, [2.0, 0.0, 0.0]) < 1e-12);
                assert!((radius - 5f64.sqrt()).abs() < 1e-12, "radius {radius}");
            }
            other => panic!("expected a circle, got {}", other.kind()),
        }
    }

    #[test]
    fn spheres_that_touch_or_miss_are_reported_as_such() {
        let a = Surface::sphere([0.0; 3], 2.0);
        assert!(matches!(
            ssi(&a, &Surface::sphere([4.0, 0.0, 0.0], 2.0)),
            SsiResult::Curves(_)
        ));
        assert_eq!(
            ssi(&a, &Surface::sphere([9.0, 0.0, 0.0], 2.0)),
            SsiResult::Disjoint
        );
        // Nested.
        assert_eq!(
            ssi(&a, &Surface::sphere([0.1, 0.0, 0.0], 5.0)),
            SsiResult::Disjoint
        );
    }

    #[test]
    fn a_coaxial_sphere_and_cylinder_meet_in_two_circles() {
        let s = Surface::sphere([0.0; 3], 5.0);
        let c = Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 3.0);
        let r = ssi(&s, &c);
        assert_on_both(&s, &c, &r, 1e-9, "sphere/cylinder");
        let SsiResult::Curves(x) = &r else {
            unreachable!()
        };
        assert_eq!(x.len(), 2);
        // At z = ±4, radius 3.
        for c in x {
            match c {
                Curve3d::Circle { center, radius, .. } => {
                    assert!((center[2].abs() - 4.0).abs() < 1e-12, "z = {}", center[2]);
                    assert!((radius - 3.0).abs() < 1e-12);
                }
                other => panic!("expected circles, got {}", other.kind()),
            }
        }
    }

    #[test]
    fn an_off_axis_sphere_and_cylinder_is_unknown() {
        let s = Surface::sphere([1.0, 0.0, 0.0], 5.0);
        let c = Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 3.0);
        assert_eq!(
            ssi(&s, &c),
            SsiResult::Unknown,
            "a quartic is not guessed at"
        );
    }

    #[test]
    fn parallel_offset_cylinders_meet_in_rulings() {
        let a = Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0);
        let b = Surface::cylinder([3.0, 0.0, 0.0], [0.0, 0.0, 1.0], 2.0);
        let r = ssi(&a, &b);
        assert_on_both(&a, &b, &r, 1e-9, "parallel cylinders");
        let SsiResult::Curves(x) = &r else {
            unreachable!()
        };
        assert_eq!(x.len(), 2);
        assert!(x.iter().all(|c| c.kind() == "line"));

        let far = Surface::cylinder([10.0, 0.0, 0.0], [0.0, 0.0, 1.0], 2.0);
        assert_eq!(ssi(&a, &far), SsiResult::Disjoint);
    }

    #[test]
    fn perpendicular_equal_radius_cylinders_meet_in_two_ellipses() {
        // The Steinmetz seam. Analytically two ellipses; numerically the case
        // the mesh kernel handles worst.
        let a = Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0);
        let b = Surface::cylinder([0.0; 3], [1.0, 0.0, 0.0], 2.0);
        let r = ssi(&a, &b);
        assert_on_both(&a, &b, &r, 1e-9, "crossed cylinders");
        let SsiResult::Curves(x) = &r else {
            unreachable!()
        };
        assert_eq!(x.len(), 2);
        for c in x {
            match c {
                Curve3d::Ellipse { a: maj, b: min, .. } => {
                    assert!((maj - 2.0 * 2f64.sqrt()).abs() < 1e-9, "a = {maj}");
                    assert!((min - 2.0).abs() < 1e-12, "b = {min}");
                }
                other => panic!("expected ellipses, got {}", other.kind()),
            }
        }
    }

    #[test]
    fn crossed_cylinders_of_different_radius_are_unknown() {
        let a = Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0);
        let b = Surface::cylinder([0.0; 3], [1.0, 0.0, 0.0], 3.0);
        assert_eq!(ssi(&a, &b), SsiResult::Unknown);
    }

    #[test]
    fn skew_cylinders_are_unknown() {
        // Genuinely skew: offset in y, so the axes pass without meeting. (An
        // x-axis line through (0, 0, 5) would *cross* the z-axis at that point,
        // which is the two-ellipse case, not this one.)
        let a = Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0);
        let b = Surface::cylinder([0.0, 6.0, 5.0], [1.0, 0.0, 0.0], 2.0);
        assert_eq!(ssi(&a, &b), SsiResult::Unknown, "axes that never meet");
    }

    #[test]
    fn crossed_cylinders_centre_their_ellipses_where_the_axes_meet() {
        let a = Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0);
        let b = Surface::cylinder([0.0, 0.0, 5.0], [1.0, 0.0, 0.0], 2.0);
        let r = ssi(&a, &b);
        assert_on_both(&a, &b, &r, 1e-9, "crossed at z = 5");
        let SsiResult::Curves(x) = &r else {
            unreachable!()
        };
        for c in x {
            match c {
                Curve3d::Ellipse { center, .. } => {
                    assert!(v3::dist(*center, [0.0, 0.0, 5.0]) < 1e-12, "{center:?}")
                }
                other => panic!("expected ellipses, got {}", other.kind()),
            }
        }
    }

    #[test]
    fn a_coaxial_cylinder_and_cone_meet_in_a_circle() {
        let c = Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0);
        let k = Surface::cone([0.0; 3], [0.0, 0.0, 1.0], FRAC_PI_4);
        let r = ssi(&c, &k);
        assert_on_both(&c, &k, &r, 1e-9, "cylinder/cone");
        let SsiResult::Curves(x) = &r else {
            unreachable!()
        };
        match &x[0] {
            Curve3d::Circle { center, radius, .. } => {
                assert!((radius - 2.0).abs() < 1e-12);
                assert!((center[2] - 2.0).abs() < 1e-12, "z = {}", center[2]);
            }
            other => panic!("expected a circle, got {}", other.kind()),
        }
    }

    #[test]
    fn coaxial_cones_with_offset_apexes_meet_in_a_circle() {
        let a = Surface::cone([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], FRAC_PI_4);
        let b = Surface::cone([0.0, 0.0, 6.0], [0.0, 0.0, -1.0], FRAC_PI_4);
        let r = ssi(&a, &b);
        assert_on_both(&a, &b, &r, 1e-9, "cone/cone");
        let SsiResult::Curves(x) = &r else {
            unreachable!()
        };
        match &x[0] {
            Curve3d::Circle { center, radius, .. } => {
                assert!((center[2] - 3.0).abs() < 1e-12, "z = {}", center[2]);
                assert!((radius - 3.0).abs() < 1e-12);
            }
            other => panic!("expected a circle, got {}", other.kind()),
        }
    }

    // -- the contract ------------------------------------------------------

    #[test]
    fn every_pair_involving_nurbs_is_unknown() {
        let n = Surface::nurbs(crate::nurbs::construct::sphere([0.0; 3], 1.0));
        let others = [
            Surface::plane([0.0; 3], [0.0, 0.0, 1.0]),
            Surface::sphere([0.0; 3], 1.0),
            Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 1.0),
            Surface::cone([0.0; 3], [0.0, 0.0, 1.0], FRAC_PI_4),
            Surface::torus([0.0; 3], [0.0, 0.0, 1.0], 3.0, 1.0),
        ];
        for o in &others {
            assert_eq!(ssi(&n, o), SsiResult::Unknown, "{} vs nurbs", o.kind());
            assert_eq!(ssi(o, &n), SsiResult::Unknown, "nurbs vs {}", o.kind());
        }
    }

    #[test]
    fn curved_pairs_without_a_closed_form_are_unknown_not_approximated() {
        let t = Surface::torus([0.0; 3], [0.0, 0.0, 1.0], 5.0, 1.0);
        let s = Surface::sphere([1.0, 2.0, 0.0], 3.0);
        let c = Surface::cylinder([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0);
        assert_eq!(ssi(&t, &s), SsiResult::Unknown);
        assert_eq!(ssi(&t, &c), SsiResult::Unknown);
    }

    #[test]
    fn ssi_is_symmetric() {
        let surfaces = [
            Surface::plane([0.0; 3], [0.0, 0.0, 1.0]),
            Surface::plane([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            Surface::sphere([0.0; 3], 3.0),
            Surface::sphere([4.0, 0.0, 0.0], 3.0),
            Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0),
            Surface::cylinder([3.0, 0.0, 0.0], [0.0, 0.0, 1.0], 2.0),
            Surface::cone([0.0; 3], [0.0, 0.0, 1.0], FRAC_PI_4),
            Surface::torus([0.0; 3], [0.0, 0.0, 1.0], 5.0, 1.0),
        ];
        for a in &surfaces {
            for b in &surfaces {
                let (ab, ba) = (ssi(a, b), ssi(b, a));
                assert_eq!(
                    ab.is_known(),
                    ba.is_known(),
                    "{} vs {}: one direction resolved and the other did not",
                    a.kind(),
                    b.kind()
                );
                // Where both resolve, the point sets must agree.
                if let (SsiResult::Curves(x), SsiResult::Curves(y)) = (&ab, &ba) {
                    assert_eq!(x.len(), y.len(), "{} vs {}", a.kind(), b.kind());
                    for cx in x {
                        let p = cx.point(0.7);
                        let best = y.iter().map(|cy| cy.distance(p)).fold(f64::MAX, f64::min);
                        assert!(best < 1e-7, "{} vs {}: curves differ", a.kind(), b.kind());
                    }
                }
            }
        }
    }

    #[test]
    fn disjoint_is_only_reported_when_it_is_provable() {
        // A false `Disjoint` would make the kernel skip a real intersection —
        // the one way this module could produce a *worse* answer than the
        // numeric path. Sample both surfaces densely and confirm they really do
        // stay apart.
        let pairs = [
            (
                Surface::sphere([0.0; 3], 2.0),
                Surface::sphere([9.0, 0.0, 0.0], 2.0),
            ),
            (
                Surface::plane([0.0; 3], [0.0, 0.0, 1.0]),
                Surface::plane([0.0, 0.0, 4.0], [0.0, 0.0, 1.0]),
            ),
            (
                Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0),
                Surface::cylinder([10.0, 0.0, 0.0], [0.0, 0.0, 1.0], 2.0),
            ),
            (
                Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0),
                Surface::cylinder([0.0, 0.0, 3.0], [0.0, 0.0, 1.0], 5.0),
            ),
        ];
        for (a, b) in &pairs {
            assert_eq!(ssi(a, b), SsiResult::Disjoint);
            let mut closest = f64::INFINITY;
            for i in 0..80 {
                for j in 0..80 {
                    let u = PI * 2.0 * i as f64 / 80.0;
                    let v = -6.0 + 12.0 * j as f64 / 80.0;
                    let v = if matches!(a, Surface::Sphere { .. }) {
                        v.clamp(-FRAC_PI_2, FRAC_PI_2)
                    } else {
                        v
                    };
                    closest = closest.min(b.distance(a.point(u, v)));
                }
            }
            assert!(
                closest > 1e-6,
                "{} vs {}: reported disjoint but they come within {closest}",
                a.kind(),
                b.kind()
            );
        }
    }
}
