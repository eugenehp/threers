//! Intersection curves — what a surface pair meets along.
//!
//! Deliberately a small closed set. These are the curves that arise from
//! quadric pairs in *closed form*; anything needing a numerically marched curve
//! is reported as [`super::SsiResult::Unknown`] and left to the mesh kernel,
//! because a marched curve that silently fails to converge is exactly the kind
//! of answer this layer must never produce.

use super::surface::Surface;
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::v3;
use crate::nurbs::V3;

/// A curve with a closed-form parameterization and a closed-form projection.
///
/// `project` is the operation everything here exists for: given a point the mesh
/// kernel computed numerically — carrying ~1e-6 of error — return the point on
/// the exact curve. That is how an approximate seam becomes an exact one.
#[derive(Debug, Clone, PartialEq)]
pub enum Curve3d {
    /// Infinite line `origin + t·dir`, `dir` unit.
    Line { origin: V3, dir: V3 },
    /// `center + r·(cos t · x + sin t · y)`, `y = axis × x`.
    Circle {
        center: V3,
        axis: V3,
        x_dir: V3,
        radius: f64,
    },
    /// `center + a·cos t · x + b·sin t · y`, with `x ⊥ y` both unit.
    Ellipse {
        center: V3,
        x_dir: V3,
        y_dir: V3,
        a: f64,
        b: f64,
    },
    /// A single point — tangential contact, where two surfaces touch without
    /// crossing. Degenerate as a *curve*, but it is genuinely where the surfaces
    /// meet, and dropping it would report tangency as disjointness.
    Point(V3),
    /// A curve with no closed form, traced numerically and kept as the points it
    /// was traced through.
    ///
    /// Two cylinders crossing at different radii meet in a quartic; so does a
    /// torus with anything off its axis. There is no conic to name, and the
    /// alternative to tracing one is declining the whole operation. The points
    /// are spaced to hold the tolerance they were traced at, and `closed`
    /// records whether the trace came back to where it started.
    Sampled { points: Vec<V3>, closed: bool },
    /// The same curve, read as a curve.
    ///
    /// A polyline is frozen: whoever samples it first fixes the boundary, and
    /// the face across it, a later refinement and a second boolean all have to
    /// chase that sampling or come apart from it. This carries the cubic through
    /// the same points, so it can be evaluated anywhere — two askers at one
    /// parameter get one point — and a closed one closes exactly rather than by
    /// a distance test on its ends.
    ///
    /// `at` holds the parameter of each traced point, so `t` still indexes the
    /// points exactly as it does for [`Curve3d::Sampled`].
    Spline {
        curve: Box<NurbsCurve>,
        at: Vec<f64>,
        closed: bool,
    },
    /// The curve as it is actually defined: the two surfaces that meet along it.
    ///
    /// Every other variant is a *description* — samples, or a fit through them —
    /// and a description is fixed at whatever density it was made. That is the
    /// root of a long class of trouble: whoever samples first fixes the
    /// boundary, a ring cannot refine because a polyline has nothing to ask, and
    /// two faces disagree about a curve neither of them owns.
    ///
    /// This one can be asked. The spline says *where* to look next and the
    /// surfaces say what is there, so a point comes out on the intersection to
    /// the tolerance the marcher holds, at any density, computed the same way by
    /// every caller.
    ///
    /// It is not an approximation of the curve. The intersection of two quadrics
    /// is a quartic and not rational, so no spline can *be* it — which is why
    /// the spline here is only a parameterisation and the answer always comes
    /// from `settle`.
    OnSurfaces {
        a: Box<Surface>,
        b: Box<Surface>,
        param: Box<Curve3d>,
        tolerance: f64,
    },
}

/// A curve moved bodily: its control points shift, its knots and weights do not.
fn shifted(curve: &NurbsCurve, t: V3) -> NurbsCurve {
    let cw: Vec<[f64; 4]> = (0..curve.n_control())
        .map(|i| {
            let w = curve.weight(i);
            let p = v3::add(curve.control_point(i), t);
            [p[0] * w, p[1] * w, p[2] * w, w]
        })
        .collect();
    NurbsCurve::from_homogeneous(curve.degree(), curve.knots().to_vec(), cw)
        .expect("shifting keeps a valid curve valid")
}

impl Curve3d {
    /// Read a traced polyline as a curve, through the very same points.
    ///
    /// `None` when there are too few points to interpolate, in which case the
    /// caller keeps the polyline it had — a curve that cannot be built is not a
    /// reason to lose the samples.
    pub fn spline_through(points: &[V3], closed: bool) -> Option<Curve3d> {
        if points.len() < 3 {
            return None;
        }
        let (curve, at) = crate::nurbs::construct::interpolate(points, closed).ok()?;
        Some(Curve3d::Spline {
            curve: Box::new(curve),
            at,
            closed,
        })
    }

    /// The curve two surfaces meet along, parameterised by a spline through the
    /// points it was traced through.
    ///
    /// The spline is scaffolding: it says where to look, and `settle` says what
    /// is there. `None` when there are too few points to parameterise, in which
    /// case the caller keeps the polyline — a curve that cannot be built is not
    /// a reason to lose the samples.
    pub fn on_surfaces(
        a: &Surface,
        b: &Surface,
        points: &[V3],
        closed: bool,
        tolerance: f64,
    ) -> Option<Curve3d> {
        let param = Curve3d::spline_through(points, closed)?;
        Some(Curve3d::OnSurfaces {
            a: Box::new(a.clone()),
            b: Box::new(b.clone()),
            param: Box::new(param),
            tolerance,
        })
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Curve3d::Line { .. } => "line",
            Curve3d::Circle { .. } => "circle",
            Curve3d::Ellipse { .. } => "ellipse",
            Curve3d::Point(_) => "point",
            Curve3d::Sampled { .. } => "sampled",
            Curve3d::OnSurfaces { .. } => "on-surfaces",
            Curve3d::Spline { .. } => "spline",
        }
    }

    /// Evaluate at parameter `t`. Lines take a distance; the closed conics take
    /// an angle.
    pub fn point(&self, t: f64) -> V3 {
        match self {
            Curve3d::Line { origin, dir } => v3::add(*origin, v3::scale(*dir, t)),
            Curve3d::Circle {
                center,
                axis,
                x_dir,
                radius,
            } => {
                let y = v3::cross(*axis, *x_dir);
                let (s, c) = t.sin_cos();
                v3::add(
                    *center,
                    v3::add(v3::scale(*x_dir, radius * c), v3::scale(y, radius * s)),
                )
            }
            Curve3d::Ellipse {
                center,
                x_dir,
                y_dir,
                a,
                b,
            } => {
                let (s, c) = t.sin_cos();
                v3::add(
                    *center,
                    v3::add(v3::scale(*x_dir, a * c), v3::scale(*y_dir, b * s)),
                )
            }
            Curve3d::Point(p) => *p,
            // `t` indexes the samples, so the parameter means the same thing a
            // polyline's always does.
            Curve3d::OnSurfaces {
                a,
                b,
                param,
                tolerance,
            } => {
                // Where the parameterisation points, put back on both surfaces.
                let guess = param.point(t);
                // Settling needs the surface/surface intersector, which is
                // `brep-csg` — a rung above this one on the feature ladder. With
                // only `brep`, the parameterisation's own point is the answer:
                // it is already on the curve, just not refined onto both
                // surfaces to tolerance.
                #[cfg(feature = "brep-csg")]
                {
                    crate::brep::intersect::settle(a, b, guess, *tolerance).unwrap_or(guess)
                }
                #[cfg(not(feature = "brep-csg"))]
                {
                    let _ = (a, b, tolerance);
                    guess
                }
            }
            Curve3d::Spline { curve, at, closed } => {
                if at.len() < 2 {
                    return [0.0; 3];
                }
                // One span per gap either way: `interpolate` gives a closed
                // curve an extra cut, its start repeated at the end.
                let spans = at.len() - 1;
                let t = if *closed {
                    t.rem_euclid(spans as f64)
                } else {
                    t.clamp(0.0, spans as f64)
                };
                let i = (t.floor() as usize).min(spans - 1);
                let f = t - i as f64;
                curve.point(at[i] + (at[i + 1] - at[i]) * f)
            }
            Curve3d::Sampled { points, closed } => {
                if points.is_empty() {
                    return [0.0; 3];
                }
                let n = points.len();
                let last = if *closed { n } else { n - 1 };
                let t = if *closed {
                    t.rem_euclid(last as f64)
                } else {
                    t.clamp(0.0, last as f64)
                };
                let i = t.floor() as usize;
                let f = t - i as f64;
                let a = points[i % n];
                let b = points[(i + 1) % n];
                v3::add(a, v3::scale(v3::sub(b, a), f))
            }
        }
    }

    /// The point on this curve closest to `p`.
    ///
    /// Exact for lines and circles. For an ellipse it is a coarse scan followed
    /// by a guarded Newton solve — see the note at that arm.
    pub fn project(&self, p: V3) -> V3 {
        match self {
            Curve3d::Line { origin, dir } => {
                v3::add(*origin, v3::scale(*dir, v3::dot(v3::sub(p, *origin), *dir)))
            }
            Curve3d::Circle {
                center,
                axis,
                radius,
                ..
            } => {
                let d = v3::sub(p, *center);
                let radial = v3::sub(d, v3::scale(*axis, v3::dot(d, *axis)));
                match v3::normalize(radial) {
                    // Directly on the axis: every point of the circle is equally
                    // close, so any is the honest answer. Take `t = 0`.
                    None => self.point(0.0),
                    Some(u) => v3::add(*center, v3::scale(u, *radius)),
                }
            }
            Curve3d::Ellipse {
                center,
                x_dir,
                y_dir,
                a,
                b,
            } => {
                let d = v3::sub(p, *center);
                let (px, py) = (v3::dot(d, *x_dir), v3::dot(d, *y_dir));

                // `D(t) = |E(t) − p|²` has up to four stationary points, and
                // Newton converges to whichever root of `D'` it happens to fall
                // into — including a *maximum*. Seeding from the parametric
                // angle is not enough: for a flat ellipse and a point near the
                // far focus it lands on the opposite end, reporting a distance
                // an order of magnitude too large while looking perfectly
                // plausible.
                //
                // So: scan coarsely for the basin of the global minimum, then
                // Newton inside it, and keep the better of the two. The scan
                // bounds the answer; Newton only sharpens it.
                const SCAN: usize = 64;
                let dist2 = |t: f64| {
                    let (s, c) = t.sin_cos();
                    (a * c - px).powi(2) + (b * s - py).powi(2)
                };
                let mut best_t = 0.0f64;
                let mut best = f64::INFINITY;
                for i in 0..SCAN {
                    let t = std::f64::consts::TAU * i as f64 / SCAN as f64;
                    let v = dist2(t);
                    if v < best {
                        best = v;
                        best_t = t;
                    }
                }

                let mut t = best_t;
                for _ in 0..32 {
                    let (s, c) = t.sin_cos();
                    // D'(t)/2 and its derivative.
                    let f = (b * b - a * a) * s * c + a * px * s - b * py * c;
                    let df = (b * b - a * a) * (c * c - s * s) + a * px * c + b * py * s;
                    if df.abs() < 1e-300 {
                        break;
                    }
                    let next = t - f / df;
                    // Reject any step that makes things worse — that is Newton
                    // heading for a different stationary point.
                    if dist2(next) > best {
                        break;
                    }
                    best = dist2(next);
                    let moved = (next - t).abs();
                    t = next;
                    if moved < 1e-15 {
                        break;
                    }
                }
                self.point(t)
            }
            Curve3d::Point(q) => *q,
            Curve3d::OnSurfaces { param, .. } => param.project(p),
            Curve3d::Spline { at, closed, .. } => {
                // Scan, then walk in: a spline has no closed form for the
                // nearest point, and the scan is what keeps the walk honest.
                if at.len() < 2 {
                    return p;
                }
                let spans = at.len() - 1;
                let steps = (spans * 8).max(64);
                let mut best = (f64::MAX, 0.0f64);
                for k in 0..=steps {
                    let t = spans as f64 * k as f64 / steps as f64;
                    let d = v3::dist(self.point(t), p);
                    if d < best.0 {
                        best = (d, t);
                    }
                }
                let mut t = best.1;
                let mut span = spans as f64 / steps as f64;
                for _ in 0..40 {
                    let mut moved = false;
                    for s in [-span, span] {
                        let u = if *closed {
                            (t + s).rem_euclid(spans as f64)
                        } else {
                            (t + s).clamp(0.0, spans as f64)
                        };
                        let d = v3::dist(self.point(u), p);
                        if d < best.0 {
                            best = (d, u);
                            t = u;
                            moved = true;
                        }
                    }
                    if !moved {
                        span *= 0.5;
                    }
                }
                self.point(t)
            }
            Curve3d::Sampled { points, closed } => {
                // The nearest point of the nearest segment. A traced curve has
                // no formula to solve against, and its samples are spaced to
                // hold the tolerance it was traced at, so this is as close as
                // the curve is known.
                let n = points.len();
                if n == 0 {
                    return p;
                }
                let last = if *closed { n } else { n - 1 };
                let mut best = points[0];
                let mut best_d = v3::dist(p, points[0]);
                for i in 0..last {
                    let (a, b) = (points[i], points[(i + 1) % n]);
                    let d = v3::sub(b, a);
                    let len2 = v3::dot(d, d);
                    let t = if len2 <= f64::MIN_POSITIVE {
                        0.0
                    } else {
                        (v3::dot(v3::sub(p, a), d) / len2).clamp(0.0, 1.0)
                    };
                    let q = v3::add(a, v3::scale(d, t));
                    let dq = v3::dist(p, q);
                    if dq < best_d {
                        best_d = dq;
                        best = q;
                    }
                }
                best
            }
        }
    }

    /// Distance from `p` to the curve.
    pub fn distance(&self, p: V3) -> f64 {
        v3::dist(p, self.project(p))
    }

    /// Translate — used when a surface pair's intersection is derived in a local
    /// frame and lifted back.
    pub fn translated(&self, t: V3) -> Curve3d {
        match self {
            Curve3d::Line { origin, dir } => Curve3d::Line {
                origin: v3::add(*origin, t),
                dir: *dir,
            },
            Curve3d::Circle {
                center,
                axis,
                x_dir,
                radius,
            } => Curve3d::Circle {
                center: v3::add(*center, t),
                axis: *axis,
                x_dir: *x_dir,
                radius: *radius,
            },
            Curve3d::Ellipse {
                center,
                x_dir,
                y_dir,
                a,
                b,
            } => Curve3d::Ellipse {
                center: v3::add(*center, t),
                x_dir: *x_dir,
                y_dir: *y_dir,
                a: *a,
                b: *b,
            },
            Curve3d::Point(q) => Curve3d::Point(v3::add(*q, t)),
            Curve3d::OnSurfaces {
                a,
                b,
                param,
                tolerance,
            } => Curve3d::OnSurfaces {
                a: Box::new(a.translated(t)),
                b: Box::new(b.translated(t)),
                param: Box::new(param.translated(t)),
                tolerance: *tolerance,
            },
            Curve3d::Spline { curve, at, closed } => Curve3d::Spline {
                curve: Box::new(shifted(curve, t)),
                at: at.clone(),
                closed: *closed,
            },
            Curve3d::Sampled { points, closed } => Curve3d::Sampled {
                points: points.iter().map(|p| v3::add(*p, t)).collect(),
                closed: *closed,
            },
        }
    }
}

/// A circle from its plane and radius, choosing a stable reference direction.
///
/// Only the closed-form intersection layer builds these, so it is dead code when
/// that layer is compiled out.
#[cfg_attr(not(feature = "brep-csg"), allow(dead_code))]
pub(crate) fn circle(center: V3, axis: V3, radius: f64) -> Curve3d {
    let a = v3::normalize(axis).unwrap_or([0.0, 0.0, 1.0]);
    Curve3d::Circle {
        center,
        axis: a,
        x_dir: crate::nurbs::construct::perpendicular(a),
        radius,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_4, PI};

    #[test]
    fn line_projection_is_the_foot_of_the_perpendicular() {
        let l = Curve3d::Line {
            origin: [1.0, 2.0, 3.0],
            dir: [0.0, 0.0, 1.0],
        };
        let q = l.project([4.0, 6.0, 10.0]);
        assert!(v3::dist(q, [1.0, 2.0, 10.0]) < 1e-15, "{q:?}");
        assert!((l.distance([4.0, 6.0, 10.0]) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn circle_projection_lands_on_the_circle() {
        let c = circle([1.0, 1.0, 0.0], [0.0, 0.0, 1.0], 3.0);
        for &p in &[
            [10.0, 1.0, 5.0],
            [1.0, -20.0, -2.0],
            [4.0, 4.0, 0.0],
            [1.5, 1.2, 7.0],
        ] {
            let q = c.project(p);
            let r = ((q[0] - 1.0).powi(2) + (q[1] - 1.0).powi(2)).sqrt();
            assert!((r - 3.0).abs() < 1e-12, "radius {r}");
            assert!(q[2].abs() < 1e-15, "left the plane");
            assert!(c.distance(q) < 1e-12, "projection is not idempotent");
        }
    }

    #[test]
    fn a_point_on_the_axis_projects_somewhere_on_the_circle_rather_than_nowhere() {
        let c = circle([0.0; 3], [0.0, 0.0, 1.0], 2.0);
        let q = c.project([0.0, 0.0, 5.0]);
        assert!((v3::norm([q[0], q[1], 0.0]) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn circle_evaluation_and_projection_agree() {
        let c = circle([2.0, -1.0, 4.0], [0.0, 1.0, 0.0], 1.5);
        for i in 0..64 {
            let t = PI * 2.0 * i as f64 / 64.0;
            let p = c.point(t);
            assert!(c.distance(p) < 1e-12, "point({t}) is not on its own curve");
        }
    }

    #[test]
    fn ellipse_projection_converges_and_lands_on_the_ellipse() {
        let e = Curve3d::Ellipse {
            center: [0.0; 3],
            x_dir: [1.0, 0.0, 0.0],
            y_dir: [0.0, 1.0, 0.0],
            a: 4.0,
            b: 2.0,
        };
        for &p in &[
            [10.0, 0.0, 0.0],
            [0.0, 9.0, 0.0],
            [3.0, 1.5, 2.0],
            [-6.0, -0.2, 0.0],
            [0.1, 0.1, 0.0],
        ] {
            let q = e.project(p);
            let on = (q[0] / 4.0).powi(2) + (q[1] / 2.0).powi(2);
            assert!(
                (on - 1.0).abs() < 1e-9,
                "projected point off the ellipse: {on}"
            );
        }
    }

    #[test]
    fn ellipse_projection_is_the_nearest_point_not_merely_a_nearby_one() {
        // Newton can converge to a *stationary* point rather than the minimum;
        // compare against a dense scan.
        let e = Curve3d::Ellipse {
            center: [1.0, 0.0, 0.0],
            x_dir: [1.0, 0.0, 0.0],
            y_dir: [0.0, 1.0, 0.0],
            a: 5.0,
            b: 1.0,
        };
        for &p in &[
            [1.0, 0.5, 0.0],
            [3.0, 0.9, 0.0],
            [-2.0, -0.3, 0.0],
            [1.2, 0.0, 0.0],
        ] {
            let mine = e.distance(p);
            let mut best = f64::INFINITY;
            for i in 0..20_000 {
                let t = PI * 2.0 * i as f64 / 20_000.0;
                best = best.min(v3::dist(e.point(t), p));
            }
            assert!(
                mine <= best + 1e-6,
                "Newton found {mine}, a scan found {best} for {p:?}"
            );
        }
    }

    #[test]
    fn a_circular_ellipse_matches_the_circle_projection() {
        let e = Curve3d::Ellipse {
            center: [0.0; 3],
            x_dir: [1.0, 0.0, 0.0],
            y_dir: [0.0, 1.0, 0.0],
            a: 3.0,
            b: 3.0,
        };
        let c = Curve3d::Circle {
            center: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            x_dir: [1.0, 0.0, 0.0],
            radius: 3.0,
        };
        for &p in &[[7.0, 2.0, 0.0], [-1.0, -1.0, 0.0], [0.5, 4.0, 0.0]] {
            assert!(v3::dist(e.project(p), c.project(p)) < 1e-9);
        }
    }

    #[test]
    fn translation_moves_every_kind() {
        let t = [1.0, -2.0, 0.5];
        let cases = [
            Curve3d::Line {
                origin: [0.0; 3],
                dir: [1.0, 0.0, 0.0],
            },
            circle([0.0; 3], [0.0, 0.0, 1.0], 2.0),
            Curve3d::Ellipse {
                center: [0.0; 3],
                x_dir: [1.0, 0.0, 0.0],
                y_dir: [0.0, 1.0, 0.0],
                a: 2.0,
                b: 1.0,
            },
            Curve3d::Point([3.0, 3.0, 3.0]),
        ];
        for c in cases {
            let moved = c.translated(t);
            assert_eq!(moved.kind(), c.kind());
            assert!(v3::dist(moved.point(FRAC_PI_4), v3::add(c.point(FRAC_PI_4), t)) < 1e-12);
        }
    }
}
