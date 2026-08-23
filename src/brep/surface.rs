//! Analytic surfaces — what a triangle came from, rather than what it is.
//!
//! `!(x > 0.0)` in the validators is the deliberate NaN-catching form; see the
//! note in `crate::nurbs::curve`.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use crate::math::Matrix4;
use crate::nurbs::v3;
use crate::nurbs::{NurbsSurface, V3};

/// A parametric surface with a closed-form evaluation, or a NURBS patch.
///
/// Every variant carries a full frame (`axis` plus a reference `x_dir`), not
/// just the minimum data that pins the point set down. A sphere is determined by
/// its centre and radius alone — but then `u` would be arbitrary, `invert` would
/// not be a function, and two tessellations of the same sphere could not be
/// compared. The frame makes the parameterization reproducible, which is what
/// later stages actually consume.
///
/// **Surfaces here are unbounded.** A cylinder extends infinitely along its
/// axis; a plane fills space. The triangles tagged with one sample some region
/// of it. Restricting a surface to a region is *trimming*, which needs the
/// `Loop`/`Edge` topology of Stage 3 — until then, "which part" is carried by
/// the triangles themselves.
#[derive(Debug, Clone, PartialEq)]
pub enum Surface {
    /// `S(u, v) = origin + u·x_dir + v·(normal × x_dir)`. Parameters are model
    /// units, not normalized — a plane has no intrinsic scale to normalize by.
    Plane { origin: V3, normal: V3, x_dir: V3 },
    /// `S(u, v) = origin + r·(cos u · x + sin u · y) + v·axis`, `y = axis × x`.
    Cylinder {
        origin: V3,
        axis: V3,
        x_dir: V3,
        radius: f64,
    },
    /// `S(u, v) = center + r·(cos v·(cos u · x + sin u · y) + sin v · axis)`,
    /// with `v ∈ [-π/2, π/2]` — latitude, not colatitude.
    Sphere {
        center: V3,
        axis: V3,
        x_dir: V3,
        radius: f64,
    },
    /// `S(u, v) = apex + v·axis + v·tan(α)·(cos u · x + sin u · y)`.
    ///
    /// `v` is signed distance along the axis, so `v < 0` is the opposite nappe.
    /// `half_angle` is measured from the axis and lies in `(0, π/2)`.
    Cone {
        apex: V3,
        axis: V3,
        x_dir: V3,
        half_angle: f64,
    },
    /// `S(u, v) = center + (R + r·cos v)·(cos u · x + sin u · y) + r·sin v · axis`.
    Torus {
        center: V3,
        axis: V3,
        x_dir: V3,
        major: f64,
        minor: f64,
    },
    /// Anything without a closed form — including a quadric that a non-uniform
    /// scale turned into one (an ellipsoid is an exact rational patch).
    Nurbs(Box<NurbsSurface>),
}

/// Build an orthonormal frame `(x, y)` perpendicular to `axis`, preferring the
/// caller's `x_hint` when it is usable.
fn frame(axis: V3, x_hint: V3) -> (V3, V3) {
    let a = v3::normalize(axis).unwrap_or([0.0, 0.0, 1.0]);
    let projected = v3::sub(x_hint, v3::scale(a, v3::dot(x_hint, a)));
    let x = v3::normalize(projected).unwrap_or_else(|| crate::nurbs::construct::perpendicular(a));
    (x, v3::cross(a, x))
}

impl Surface {
    /// The same surface, moved bodily. Only its position changes; its axes,
    /// radii and angles are what it *is*.
    pub fn translated(&self, t: V3) -> Surface {
        let go = |p: V3| [p[0] + t[0], p[1] + t[1], p[2] + t[2]];
        match self {
            Surface::Plane {
                origin,
                normal,
                x_dir,
            } => Surface::Plane {
                origin: go(*origin),
                normal: *normal,
                x_dir: *x_dir,
            },
            Surface::Cylinder {
                origin,
                axis,
                x_dir,
                radius,
            } => Surface::Cylinder {
                origin: go(*origin),
                axis: *axis,
                x_dir: *x_dir,
                radius: *radius,
            },
            Surface::Sphere {
                center,
                axis,
                x_dir,
                radius,
            } => Surface::Sphere {
                center: go(*center),
                axis: *axis,
                x_dir: *x_dir,
                radius: *radius,
            },
            Surface::Cone {
                apex,
                axis,
                x_dir,
                half_angle,
            } => Surface::Cone {
                apex: go(*apex),
                axis: *axis,
                x_dir: *x_dir,
                half_angle: *half_angle,
            },
            Surface::Torus {
                center,
                axis,
                x_dir,
                major,
                minor,
            } => Surface::Torus {
                center: go(*center),
                axis: *axis,
                x_dir: *x_dir,
                major: *major,
                minor: *minor,
            },
            // A NURBS surface moves by moving its control points.
            Surface::Nurbs(n) => {
                let cw: Vec<[f64; 4]> = (0..n.n_u() * n.n_v())
                    .map(|i| {
                        let (iu, iv) = (i / n.n_v(), i % n.n_v());
                        let w = n.weight(iu, iv);
                        let p = go(n.control_point(iu, iv));
                        [p[0] * w, p[1] * w, p[2] * w, w]
                    })
                    .collect();
                match NurbsSurface::from_homogeneous(
                    n.degree_u(),
                    n.degree_v(),
                    n.knots_u().to_vec(),
                    n.knots_v().to_vec(),
                    n.n_u(),
                    n.n_v(),
                    cw,
                ) {
                    Ok(moved) => Surface::Nurbs(Box::new(moved)),
                    Err(_) => self.clone(),
                }
            }
        }
    }

    // -- constructors ------------------------------------------------------

    pub fn plane(origin: V3, normal: V3) -> Self {
        let n = v3::normalize(normal).unwrap_or([0.0, 0.0, 1.0]);
        let (x, _) = frame(n, crate::nurbs::construct::perpendicular(n));
        Surface::Plane {
            origin,
            normal: n,
            x_dir: x,
        }
    }

    pub fn cylinder(origin: V3, axis: V3, radius: f64) -> Self {
        let a = v3::normalize(axis).unwrap_or([0.0, 0.0, 1.0]);
        let (x, _) = frame(a, crate::nurbs::construct::perpendicular(a));
        Surface::Cylinder {
            origin,
            axis: a,
            x_dir: x,
            radius,
        }
    }

    pub fn sphere(center: V3, radius: f64) -> Self {
        Surface::Sphere {
            center,
            axis: [0.0, 0.0, 1.0],
            x_dir: [1.0, 0.0, 0.0],
            radius,
        }
    }

    /// A cone from its apex and half-angle. `half_angle` must be in `(0, π/2)`.
    pub fn cone(apex: V3, axis: V3, half_angle: f64) -> Self {
        let a = v3::normalize(axis).unwrap_or([0.0, 0.0, 1.0]);
        let (x, _) = frame(a, crate::nurbs::construct::perpendicular(a));
        Surface::Cone {
            apex,
            axis: a,
            x_dir: x,
            half_angle,
        }
    }

    /// A cone from a rim: radius `r` at distance `h` along the axis from `apex`.
    pub fn cone_from_rim(apex: V3, axis: V3, radius: f64, height: f64) -> Option<Self> {
        if !(radius > 0.0) || height.abs() < 1e-12 {
            return None;
        }
        Some(Self::cone(apex, axis, (radius / height.abs()).atan()))
    }

    pub fn torus(center: V3, axis: V3, major: f64, minor: f64) -> Self {
        let a = v3::normalize(axis).unwrap_or([0.0, 0.0, 1.0]);
        let (x, _) = frame(a, crate::nurbs::construct::perpendicular(a));
        Surface::Torus {
            center,
            axis: a,
            x_dir: x,
            major,
            minor,
        }
    }

    pub fn nurbs(s: NurbsSurface) -> Self {
        Surface::Nurbs(Box::new(s))
    }

    /// Replace the reference direction that pins `u = 0`, orthonormalizing it
    /// against the axis.
    ///
    /// The constructors pick an arbitrary perpendicular, which describes the
    /// same point set but a rotated parameterization. A generator that already
    /// knows where its own `θ = 0` sits uses this so `invert` returns the angle
    /// the mesh was built with, rather than one offset by an arbitrary amount.
    pub fn with_x_dir(self, x_hint: V3) -> Self {
        match self {
            Surface::Plane { origin, normal, .. } => {
                let (x, _) = frame(normal, x_hint);
                Surface::Plane {
                    origin,
                    normal,
                    x_dir: x,
                }
            }
            Surface::Cylinder {
                origin,
                axis,
                radius,
                ..
            } => {
                let (x, _) = frame(axis, x_hint);
                Surface::Cylinder {
                    origin,
                    axis,
                    x_dir: x,
                    radius,
                }
            }
            Surface::Sphere {
                center,
                axis,
                radius,
                ..
            } => {
                let (x, _) = frame(axis, x_hint);
                Surface::Sphere {
                    center,
                    axis,
                    x_dir: x,
                    radius,
                }
            }
            Surface::Cone {
                apex,
                axis,
                half_angle,
                ..
            } => {
                let (x, _) = frame(axis, x_hint);
                Surface::Cone {
                    apex,
                    axis,
                    x_dir: x,
                    half_angle,
                }
            }
            Surface::Torus {
                center,
                axis,
                major,
                minor,
                ..
            } => {
                let (x, _) = frame(axis, x_hint);
                Surface::Torus {
                    center,
                    axis,
                    x_dir: x,
                    major,
                    minor,
                }
            }
            Surface::Nurbs(_) => self,
        }
    }

    /// Set the axis, keeping everything else. Paired with [`Self::with_x_dir`]
    /// for generators whose convention differs from the constructors' default.
    pub fn with_axis(self, new_axis: V3) -> Self {
        let a = v3::normalize(new_axis).unwrap_or([0.0, 0.0, 1.0]);
        match self {
            Surface::Sphere {
                center,
                x_dir,
                radius,
                ..
            } => Surface::Sphere {
                center,
                axis: a,
                x_dir,
                radius,
            }
            .with_x_dir(x_dir),
            other => other,
        }
    }

    /// A short name for diagnostics and reports.
    pub fn kind(&self) -> &'static str {
        match self {
            Surface::Plane { .. } => "plane",
            Surface::Cylinder { .. } => "cylinder",
            Surface::Sphere { .. } => "sphere",
            Surface::Cone { .. } => "cone",
            Surface::Torus { .. } => "torus",
            Surface::Nurbs(_) => "nurbs",
        }
    }

    /// Is the parameterization periodic in `u` / `v`? Retessellation needs this
    /// to know whether a `u` range that wraps past 2π is a seam or an error.
    pub fn periodic(&self) -> (bool, bool) {
        match self {
            Surface::Plane { .. } => (false, false),
            Surface::Cylinder { .. } | Surface::Cone { .. } => (true, false),
            Surface::Sphere { .. } => (true, false),
            Surface::Torus { .. } => (true, true),
            Surface::Nurbs(_) => (false, false),
        }
    }

    /// The period of each parameter, where it has one.
    ///
    /// [`Self::periodic`] answers *whether*, and every caller then assumes the
    /// period is a full turn — eighty-one places in this module reach for `TAU`
    /// to fold a parameter. That assumption is true today of every closed-form
    /// surface here and it is the surface's fact to state, not the caller's to
    /// guess: a NURBS patch closed on itself has whatever period its knots give
    /// it, and nothing outside this file can know it.
    pub fn period(&self) -> (Option<f64>, Option<f64>) {
        use std::f64::consts::TAU;
        let (pu, pv) = self.periodic();
        match self {
            // Angular in `u`, and in `v` for a torus's tube.
            Surface::Cylinder { .. }
            | Surface::Cone { .. }
            | Surface::Sphere { .. }
            | Surface::Torus { .. } => (pu.then_some(TAU), pv.then_some(TAU)),
            Surface::Plane { .. } | Surface::Nurbs(_) => (None, None),
        }
    }

    /// Invert, and answer on the branch nearest `anchor`.
    ///
    /// [`Self::invert`] answers in the canonical period, which is the right
    /// answer to a question almost nobody is asking. A face's rings live
    /// wherever the modelling left them — a sphere cut by a sphere keeps a
    /// cavity wall whose `u` runs from `pi` to `3pi` — so a point inverted
    /// canonically misses its own face by a whole turn. That cost two
    /// twelve-per-cent wrong answers before it was found.
    ///
    /// Asking for the branch you are working on makes the right answer the easy
    /// one.
    pub fn invert_near(&self, p: V3, anchor: (f64, f64)) -> Option<(f64, f64)> {
        let (mut u, mut v) = self.invert(p)?;
        let (period_u, period_v) = self.period();
        if let Some(t) = period_u {
            u -= t * ((u - anchor.0) / t).round();
        }
        if let Some(t) = period_v {
            v -= t * ((v - anchor.1) / t).round();
        }
        Some((u, v))
    }

    // -- evaluation --------------------------------------------------------

    pub fn point(&self, u: f64, v: f64) -> V3 {
        match self {
            Surface::Plane {
                origin,
                normal,
                x_dir,
            } => {
                let y = v3::cross(*normal, *x_dir);
                v3::add(*origin, v3::add(v3::scale(*x_dir, u), v3::scale(y, v)))
            }
            Surface::Cylinder {
                origin,
                axis,
                x_dir,
                radius,
            } => {
                let r = radial(*axis, *x_dir, u);
                v3::add(*origin, v3::add(v3::scale(r, *radius), v3::scale(*axis, v)))
            }
            Surface::Sphere {
                center,
                axis,
                x_dir,
                radius,
            } => {
                let r = radial(*axis, *x_dir, u);
                let (sv, cv) = v.sin_cos();
                v3::add(
                    *center,
                    v3::scale(v3::add(v3::scale(r, cv), v3::scale(*axis, sv)), *radius),
                )
            }
            Surface::Cone {
                apex,
                axis,
                x_dir,
                half_angle,
            } => {
                let r = radial(*axis, *x_dir, u);
                v3::add(
                    *apex,
                    v3::add(v3::scale(*axis, v), v3::scale(r, v * half_angle.tan())),
                )
            }
            Surface::Torus {
                center,
                axis,
                x_dir,
                major,
                minor,
            } => {
                let r = radial(*axis, *x_dir, u);
                let (sv, cv) = v.sin_cos();
                v3::add(
                    *center,
                    v3::add(
                        v3::scale(r, major + minor * cv),
                        v3::scale(*axis, minor * sv),
                    ),
                )
            }
            Surface::Nurbs(s) => s.point(u, v),
        }
    }

    /// Outward unit normal at `(u, v)`.
    ///
    /// "Outward" means away from the axis or centre for the closed quadrics, and
    /// `normal` for a plane. A face whose triangles wind the other way records
    /// that in its own orientation; the surface itself has one canonical sense,
    /// otherwise two faces sharing a surface could not be compared.
    pub fn normal(&self, u: f64, v: f64) -> Option<V3> {
        match self {
            Surface::Plane { normal, .. } => Some(*normal),
            Surface::Cylinder { axis, x_dir, .. } => Some(radial(*axis, *x_dir, u)),
            Surface::Sphere { axis, x_dir, .. } => {
                let r = radial(*axis, *x_dir, u);
                let (sv, cv) = v.sin_cos();
                v3::normalize(v3::add(v3::scale(r, cv), v3::scale(*axis, sv)))
            }
            Surface::Cone {
                axis,
                x_dir,
                half_angle,
                ..
            } => {
                let r = radial(*axis, *x_dir, u);
                let (sa, ca) = half_angle.sin_cos();
                // Perpendicular to the ruling, tilted back toward the apex.
                v3::normalize(v3::sub(v3::scale(r, ca), v3::scale(*axis, sa)))
            }
            Surface::Torus { axis, x_dir, .. } => {
                let r = radial(*axis, *x_dir, u);
                let (sv, cv) = v.sin_cos();
                v3::normalize(v3::add(v3::scale(r, cv), v3::scale(*axis, sv)))
            }
            Surface::Nurbs(s) => s.normal(u, v),
        }
    }

    /// Parameters of the closest point on the surface to `p` — the inverse of
    /// [`Self::point`], to within the surface's own degeneracies.
    ///
    /// Where the parameterization degenerates — a cylinder's axis, a cone's
    /// apex, a sphere's pole — only `u` is undetermined; `v` is perfectly well
    /// defined, and so is the point. So this reports `u = 0` there rather than
    /// failing, and [`Self::parameter_degenerate_at`] is how a caller learns
    /// that `u` was arbitrary.
    ///
    /// Returning `None` instead loses the `v` too, which is worse than useless:
    /// a cone's apex vertices simply vanished from its face's parameter
    /// footprint, collapsing it to a single `v` and producing a body whose side
    /// had no extent.
    pub fn invert(&self, p: V3) -> Option<(f64, f64)> {
        match self {
            Surface::Plane {
                origin,
                normal,
                x_dir,
            } => {
                let y = v3::cross(*normal, *x_dir);
                let d = v3::sub(p, *origin);
                Some((v3::dot(d, *x_dir), v3::dot(d, y)))
            }
            Surface::Cylinder {
                origin,
                axis,
                x_dir,
                ..
            } => {
                let d = v3::sub(p, *origin);
                let v = v3::dot(d, *axis);
                let rad = v3::sub(d, v3::scale(*axis, v));
                Some((angle_of(rad, *axis, *x_dir).unwrap_or(0.0), v))
            }
            Surface::Sphere {
                center,
                axis,
                x_dir,
                radius,
            } => {
                let d = v3::sub(p, *center);
                let len = v3::norm(d);
                if len < 1e-300 {
                    return None; // the centre itself projects nowhere
                }
                let r = if *radius > 0.0 { *radius } else { len };
                let v = (v3::dot(d, *axis) / r).clamp(-1.0, 1.0).asin();
                let rad = v3::sub(d, v3::scale(*axis, v3::dot(d, *axis)));
                Some((angle_of(rad, *axis, *x_dir).unwrap_or(0.0), v))
            }
            Surface::Cone {
                apex, axis, x_dir, ..
            } => {
                let d = v3::sub(p, *apex);
                let v = v3::dot(d, *axis);
                let rad = v3::sub(d, v3::scale(*axis, v));
                Some((angle_of(rad, *axis, *x_dir).unwrap_or(0.0), v))
            }
            Surface::Torus {
                center,
                axis,
                x_dir,
                major,
                ..
            } => {
                let d = v3::sub(p, *center);
                let along = v3::dot(d, *axis);
                let rad = v3::sub(d, v3::scale(*axis, along));
                let rho = v3::norm(rad);
                let u = angle_of(rad, *axis, *x_dir)?;
                Some((u, along.atan2(rho - major)))
            }
            Surface::Nurbs(s) => nurbs_invert(s, p),
        }
    }

    /// Is `p` at a place where this parameterization has no unique `u`?
    ///
    /// A sphere's poles and a cylinder's axis. [`Self::invert`] deliberately
    /// returns a usable parameter there — the *point* is well defined and
    /// callers want it — but anything reasoning about `u` as a coordinate has to
    /// know it is arbitrary. UV assignment in particular: seeding a texture's
    /// branch from a pole vertex flags every triangle around that pole as
    /// crossing a seam it does not cross.
    pub fn parameter_degenerate_at(&self, p: V3) -> bool {
        let radial_len = |origin: V3, axis: V3| {
            let d = v3::sub(p, origin);
            v3::norm(v3::sub(d, v3::scale(axis, v3::dot(d, axis))))
        };
        match self {
            Surface::Plane { .. } | Surface::Torus { .. } => false,
            Surface::Cylinder { origin, axis, .. } => radial_len(*origin, *axis) < 1e-9,
            Surface::Cone { apex, axis, .. } => radial_len(*apex, *axis) < 1e-9,
            Surface::Sphere {
                center,
                axis,
                radius,
                ..
            } => radial_len(*center, *axis) < 1e-9 * radius.max(1.0),
            // A NURBS patch can be degenerate anywhere; `normal` already knows.
            Surface::Nurbs(s) => match nurbs_invert(s, p) {
                Some((u, v)) => s.normal(u, v).is_none(),
                None => true,
            },
        }
    }

    /// Distance from `p` to the surface. Zero (to rounding) means `p` lies on it.
    ///
    /// This is the workhorse for validating provenance: a tag claiming a
    /// triangle came from a given surface is checkable by measuring its
    /// vertices against that surface, which is exactly what
    /// [`crate::brep::SurfaceTable::max_deviation`] does.
    pub fn distance(&self, p: V3) -> f64 {
        match self {
            Surface::Plane { origin, normal, .. } => v3::dot(v3::sub(p, *origin), *normal).abs(),
            Surface::Cylinder {
                origin,
                axis,
                radius,
                ..
            } => {
                let d = v3::sub(p, *origin);
                let rad = v3::sub(d, v3::scale(*axis, v3::dot(d, *axis)));
                (v3::norm(rad) - radius).abs()
            }
            Surface::Sphere { center, radius, .. } => (v3::dist(p, *center) - radius).abs(),
            Surface::Cone {
                apex,
                axis,
                half_angle,
                ..
            } => {
                // In the (ρ, h) half-plane the cone is the line ρ = h·tan α,
                // whose unit normal is (cos α, −sin α).
                let d = v3::sub(p, *apex);
                let h = v3::dot(d, *axis);
                let rho = v3::norm(v3::sub(d, v3::scale(*axis, h)));
                let (sa, ca) = half_angle.sin_cos();
                (rho * ca - h * sa).abs()
            }
            Surface::Torus {
                center,
                axis,
                major,
                minor,
                ..
            } => {
                let d = v3::sub(p, *center);
                let along = v3::dot(d, *axis);
                let rho = v3::norm(v3::sub(d, v3::scale(*axis, along)));
                (((rho - major).powi(2) + along * along).sqrt() - minor).abs()
            }
            Surface::Nurbs(s) => match nurbs_invert(s, p) {
                Some((u, v)) => v3::dist(p, s.point(u, v)),
                None => f64::INFINITY,
            },
        }
    }

    /// Signed distance to the surface: negative inside, positive outside.
    ///
    /// "Inside" means the side the outward [`Self::normal`] points away from,
    /// which for the closed quadrics is the enclosed volume and for a plane is
    /// the half-space behind its normal.
    ///
    /// The signed form is what makes a *root find along a mesh edge* possible —
    /// [`Self::distance`] has a minimum at the surface, not a sign change, so it
    /// cannot bracket. That root find is how an approximate seam point becomes
    /// exact without leaving the edge it has to stay on.
    pub fn signed_distance(&self, p: V3) -> f64 {
        match self {
            Surface::Plane { origin, normal, .. } => v3::dot(v3::sub(p, *origin), *normal),
            Surface::Cylinder {
                origin,
                axis,
                radius,
                ..
            } => {
                let d = v3::sub(p, *origin);
                let rad = v3::sub(d, v3::scale(*axis, v3::dot(d, *axis)));
                v3::norm(rad) - radius
            }
            Surface::Sphere { center, radius, .. } => v3::dist(p, *center) - radius,
            Surface::Cone {
                apex,
                axis,
                half_angle,
                ..
            } => {
                let d = v3::sub(p, *apex);
                let h = v3::dot(d, *axis);
                let rho = v3::norm(v3::sub(d, v3::scale(*axis, h)));
                let (sa, ca) = half_angle.sin_cos();
                rho * ca - h * sa
            }
            Surface::Torus {
                center,
                axis,
                major,
                minor,
                ..
            } => {
                let d = v3::sub(p, *center);
                let along = v3::dot(d, *axis);
                let rho = v3::norm(v3::sub(d, v3::scale(*axis, along)));
                ((rho - major).powi(2) + along * along).sqrt() - minor
            }
            Surface::Nurbs(s) => match nurbs_invert(s, p) {
                Some((u, v)) => {
                    let q = s.point(u, v);
                    // Behind the normal is inside. A patch with no normal at
                    // this parameter (a pole) has no side to be on, so take the
                    // unsigned distance rather than invent a sign.
                    let inside = s
                        .normal(u, v)
                        .is_some_and(|n| v3::dot(v3::sub(p, q), n) < 0.0);
                    let sign = if inside { -1.0 } else { 1.0 };
                    sign * v3::dist(p, q)
                }
                None => f64::INFINITY,
            },
        }
    }

    /// The point on the segment `p0 → p1` that lies exactly on this surface,
    /// nearest to `near`.
    ///
    /// Returns `None` unless the segment genuinely crosses the surface. Bisection
    /// on [`Self::signed_distance`] — robust where a Newton step is not, since
    /// the bracket is guaranteed and a cylinder's signed distance is not smooth
    /// through its axis.
    ///
    /// This is how a seam point becomes exact *without moving off its edge*: the
    /// answer is constrained to the segment by construction. Projecting onto the
    /// surfaces' intersection curve instead — the obvious thing — moves the
    /// point off the mesh edge the refinement needs it to lie on.
    pub fn edge_crossing(&self, p0: V3, p1: V3, near: V3) -> Option<V3> {
        let (mut a, mut b) = (0.0f64, 1.0f64);
        let at = |t: f64| v3::add(p0, v3::scale(v3::sub(p1, p0), t));
        let (mut fa, mut fb) = (self.signed_distance(p0), self.signed_distance(p1));
        if !fa.is_finite() || !fb.is_finite() || fa * fb > 0.0 {
            return None;
        }
        for _ in 0..60 {
            let m = 0.5 * (a + b);
            let fm = self.signed_distance(at(m));
            if fa * fm <= 0.0 {
                b = m;
                fb = fm;
            } else {
                a = m;
                fa = fm;
            }
            if (b - a) < 1e-15 {
                break;
            }
        }
        let _ = fb;
        let q = at(0.5 * (a + b));
        let _ = near;
        Some(q)
    }

    // -- transformation ----------------------------------------------------

    /// Map through an affine transform, or `None` if the result is not
    /// representable.
    ///
    /// Quadrics survive **similarity** transforms (rigid motion plus uniform
    /// scale) and nothing else: a non-uniformly scaled sphere is an ellipsoid,
    /// which this enum has no variant for. Rather than silently keep a `Sphere`
    /// tag whose radius is now a lie, that case returns `None` and the caller
    /// drops the provenance — always a safe outcome, since every consumer treats
    /// missing provenance as "fall back to the mesh".
    ///
    /// Planes and NURBS survive any invertible affine map: a plane maps to a
    /// plane (with the inverse-transpose normal), and NURBS control points
    /// transform exactly in homogeneous coordinates.
    pub fn transform(&self, m: &Matrix4) -> Option<Surface> {
        let a = Affine::from_matrix4(m)?;

        if let Surface::Plane {
            origin,
            normal,
            x_dir,
        } = self
        {
            let o = a.point(*origin);
            let n = v3::normalize(a.normal(*normal))?;
            let x = v3::normalize(a.direction(*x_dir))?;
            let (x, _) = frame(n, x);
            return Some(Surface::Plane {
                origin: o,
                normal: n,
                x_dir: x,
            });
        }

        if let Surface::Nurbs(s) = self {
            let control: Vec<[f64; 4]> = s
                .homogeneous()
                .iter()
                .map(|h| {
                    // Homogeneous: transform (x, y, z) as a point scaled by w.
                    let w = h[3];
                    let p = a.point([h[0] / w, h[1] / w, h[2] / w]);
                    [p[0] * w, p[1] * w, p[2] * w, w]
                })
                .collect();
            return NurbsSurface::from_homogeneous(
                s.degree_u(),
                s.degree_v(),
                s.knots_u().to_vec(),
                s.knots_v().to_vec(),
                s.n_u(),
                s.n_v(),
                control,
            )
            .ok()
            .map(Surface::nurbs);
        }

        // Everything below is a quadric and needs a similarity.
        let s = a.uniform_scale()?;
        match self {
            Surface::Cylinder {
                origin,
                axis,
                x_dir,
                radius,
            } => Some(Surface::Cylinder {
                origin: a.point(*origin),
                axis: v3::normalize(a.direction(*axis))?,
                x_dir: v3::normalize(a.direction(*x_dir))?,
                radius: radius * s,
            }),
            Surface::Sphere {
                center,
                axis,
                x_dir,
                radius,
            } => Some(Surface::Sphere {
                center: a.point(*center),
                axis: v3::normalize(a.direction(*axis))?,
                x_dir: v3::normalize(a.direction(*x_dir))?,
                radius: radius * s,
            }),
            Surface::Cone {
                apex,
                axis,
                x_dir,
                half_angle,
            } => Some(Surface::Cone {
                apex: a.point(*apex),
                axis: v3::normalize(a.direction(*axis))?,
                x_dir: v3::normalize(a.direction(*x_dir))?,
                // Uniform scale preserves angles.
                half_angle: *half_angle,
            }),
            Surface::Torus {
                center,
                axis,
                x_dir,
                major,
                minor,
            } => Some(Surface::Torus {
                center: a.point(*center),
                axis: v3::normalize(a.direction(*axis))?,
                x_dir: v3::normalize(a.direction(*x_dir))?,
                major: major * s,
                minor: minor * s,
            }),
            Surface::Plane { .. } | Surface::Nurbs(_) => unreachable!("handled above"),
        }
    }
}

/// `cos u · x + sin u · y`, the unit radial direction at angle `u`.
fn radial(axis: V3, x_dir: V3, u: f64) -> V3 {
    let y = v3::cross(axis, x_dir);
    let (s, c) = u.sin_cos();
    v3::add(v3::scale(x_dir, c), v3::scale(y, s))
}

/// The angle of `rad` in the `(x_dir, axis × x_dir)` frame, or `None` if `rad`
/// is too short to have one.
fn angle_of(rad: V3, axis: V3, x_dir: V3) -> Option<f64> {
    if v3::norm(rad) < 1e-12 {
        return None;
    }
    let y = v3::cross(axis, x_dir);
    Some(v3::dot(rad, y).atan2(v3::dot(rad, x_dir)))
}

/// Point inversion on a NURBS surface — grid seed, then a **guarded** Newton on
/// the two orthogonality conditions `(S − p)·Sᵤ = 0`, `(S − p)·Sᵥ = 0` (A6.1).
///
/// Both halves matter, and leaving either out fails quietly rather than loudly.
///
/// *The seed must be dense enough for the control net.* A fixed 12×12 grid is
/// fine for a quadric patch and far too coarse for a lofted tube, whose 9×17 net
/// bends through several turns — the nearest sample can sit in a different basin
/// entirely.
///
/// *The Newton must be guarded.* Its step is unbounded, so from a mediocre seed
/// it walks out of the basin and converges to a stationary point on the far side
/// of the surface. Measured on a tube of radius 0.4: the true distance was
/// 4.5e-3 and the unguarded solve reported **2.2**, which read as a wildly
/// mis-tagged mesh rather than as a failed root find. Keeping the best sample
/// seen and rejecting any step that makes things worse costs one distance
/// evaluation per iteration and removes the failure mode.
fn nurbs_invert(s: &NurbsSurface, p: V3) -> Option<(f64, f64)> {
    let (u0, u1) = s.domain_u();
    let (v0, v1) = s.domain_v();

    // Two samples per control point per direction, floored so small patches are
    // still seeded properly.
    let seed_u = (2 * s.n_u()).clamp(12, 96);
    let seed_v = (2 * s.n_v()).clamp(12, 96);

    let (mut bu, mut bv, mut best) = (u0, v0, f64::INFINITY);
    for i in 0..=seed_u {
        for j in 0..=seed_v {
            let u = u0 + (u1 - u0) * i as f64 / seed_u as f64;
            let v = v0 + (v1 - v0) * j as f64 / seed_v as f64;
            let d = v3::dist(s.point(u, v), p);
            if d < best {
                best = d;
                bu = u;
                bv = v;
            }
        }
    }

    let (mut u, mut v) = (bu, bv);
    for _ in 0..48 {
        let d = s.derivatives(u, v, 2);
        let r = v3::sub(d[0][0], p);
        let (su, sv) = (d[1][0], d[0][1]);

        let f = v3::dot(r, su);
        let g = v3::dot(r, sv);
        if f.abs() < 1e-14 && g.abs() < 1e-14 {
            break;
        }

        // Jacobian of (f, g) with respect to (u, v).
        let j00 = v3::dot(su, su) + v3::dot(r, d[2][0]);
        let j01 = v3::dot(su, sv) + v3::dot(r, d[1][1]);
        let j11 = v3::dot(sv, sv) + v3::dot(r, d[0][2]);
        let det = j00 * j11 - j01 * j01;
        if det.abs() < 1e-300 {
            break;
        }

        let du = (-f * j11 + g * j01) / det;
        let dv = (-g * j00 + f * j01) / det;

        // Backtracking line search: a full Newton step that increases the
        // distance is heading for a different stationary point.
        let mut step = 1.0f64;
        let mut accepted = false;
        for _ in 0..8 {
            let nu = (u + du * step).clamp(u0, u1);
            let nv = (v + dv * step).clamp(v0, v1);
            let nd = v3::dist(s.point(nu, nv), p);
            if nd <= best {
                let moved = (nu - u).abs() + (nv - v).abs();
                best = nd;
                u = nu;
                v = nv;
                accepted = true;
                if moved < 1e-15 {
                    return Some((u, v));
                }
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break; // already at the best point this basin offers
        }
    }
    Some((u, v))
}

/// An f64 affine transform lifted out of the crate's f32 [`Matrix4`].
///
/// The lift does not recover precision the f32 matrix never had; it exists so
/// the *composition* with f64 surface data does not round twice.
pub(crate) struct Affine {
    /// Column-major 3×3 linear part.
    linear: [[f64; 3]; 3],
    translation: V3,
}

impl Affine {
    pub(crate) fn from_matrix4(m: &Matrix4) -> Option<Self> {
        let e = &m.elements;
        // Reject a projective bottom row: these surfaces are affine objects.
        if e[3].abs() > 1e-9
            || e[7].abs() > 1e-9
            || e[11].abs() > 1e-9
            || (e[15] - 1.0).abs() > 1e-6
        {
            return None;
        }
        let col = |c: usize| [e[c * 4] as f64, e[c * 4 + 1] as f64, e[c * 4 + 2] as f64];
        let linear = [col(0), col(1), col(2)];
        // Singular linear part collapses the surface — nothing to represent.
        let det = v3::dot(linear[0], v3::cross(linear[1], linear[2]));
        if det.abs() < 1e-18 {
            return None;
        }
        Some(Affine {
            linear,
            translation: col(3),
        })
    }

    fn direction(&self, d: V3) -> V3 {
        v3::add(
            v3::add(
                v3::scale(self.linear[0], d[0]),
                v3::scale(self.linear[1], d[1]),
            ),
            v3::scale(self.linear[2], d[2]),
        )
    }

    pub(crate) fn point(&self, p: V3) -> V3 {
        v3::add(self.direction(p), self.translation)
    }

    /// Normals transform by the inverse transpose, not the matrix itself —
    /// using the matrix directly is right for rotations and wrong for every
    /// shear, which is the kind of bug that only shows up on skewed models.
    ///
    /// `M⁻¹` has *rows* `cross(b,c)`, `cross(c,a)`, `cross(a,b)` over the
    /// determinant, so `M⁻ᵀ` has those as *columns* and `M⁻ᵀ·n` is their linear
    /// combination weighted by `n` — not the three dot products, which would
    /// compute `M⁻¹·n` and differ from the answer on exactly the shears this
    /// exists to handle. The `1/det` is dropped; the caller normalizes.
    fn normal(&self, n: V3) -> V3 {
        let (a, b, c) = (self.linear[0], self.linear[1], self.linear[2]);
        let rows = [v3::cross(b, c), v3::cross(c, a), v3::cross(a, b)];
        v3::add(
            v3::add(v3::scale(rows[0], n[0]), v3::scale(rows[1], n[1])),
            v3::scale(rows[2], n[2]),
        )
    }

    /// The uniform scale factor, or `None` if the linear part is not a
    /// similarity (orthogonal columns of equal length).
    ///
    /// The tolerance is `1e-5` relative, not the `1e-9` an f64 matrix would
    /// justify: [`Matrix4`] is f32, so a composed rotation's columns are only
    /// orthonormal to about `1e-7` before this ever sees them. Demanding f64
    /// tightness of f32 data rejects every genuine rigid motion.
    fn uniform_scale(&self) -> Option<f64> {
        const REL: f64 = 1e-5;
        let l: Vec<f64> = self.linear.iter().map(|c| v3::norm(*c)).collect();
        if l.iter().any(|&x| x < 1e-12) {
            return None;
        }
        let s = l[0];
        if l.iter().any(|&x| (x - s).abs() > REL * s) {
            return None;
        }
        for (i, j) in [(0, 1), (1, 2), (0, 2)] {
            if v3::dot(self.linear[i], self.linear[j]).abs() > REL * s * s {
                return None;
            }
        }
        Some(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vector3;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

    fn round_trip(s: &Surface, us: &[f64], vs: &[f64]) {
        for &u in us {
            for &v in vs {
                let p = s.point(u, v);
                assert!(
                    s.distance(p) < 1e-12,
                    "{}: point({u}, {v}) is {} off its own surface",
                    s.kind(),
                    s.distance(p)
                );
                let (iu, iv) = s.invert(p).expect("invertible away from degeneracies");
                let q = s.point(iu, iv);
                assert!(
                    v3::dist(p, q) < 1e-9,
                    "{}: invert round-trip moved the point by {}",
                    s.kind(),
                    v3::dist(p, q)
                );
            }
        }
    }

    #[test]
    fn plane_round_trips() {
        let s = Surface::plane([1.0, 2.0, 3.0], [0.0, 1.0, 1.0]);
        round_trip(&s, &[-3.0, 0.0, 2.5], &[-1.0, 0.0, 4.0]);
        assert!((s.distance([1.0, 2.0, 3.0])).abs() < 1e-15);
    }

    #[test]
    fn cylinder_round_trips_and_measures_distance() {
        let s = Surface::cylinder([0.0, 0.0, -2.0], [0.0, 0.0, 1.0], 3.0);
        round_trip(&s, &[0.0, 1.0, PI, 5.0], &[-4.0, 0.0, 7.0]);
        assert!((s.distance([5.0, 0.0, 0.0]) - 2.0).abs() < 1e-12);
        // On the axis the *angle* is undetermined, but the height is not — so
        // this reports a usable parameter and flags it, rather than discarding
        // the height along with the angle.
        let (_, v) = s
            .invert([0.0, 0.0, 5.0])
            .expect("the axis still has a height");
        assert!((v - 7.0).abs() < 1e-12, "v = {v}");
        assert!(s.parameter_degenerate_at([0.0, 0.0, 5.0]));
    }

    #[test]
    fn sphere_round_trips_and_poles_report_a_parameter() {
        let s = Surface::sphere([1.0, -1.0, 0.5], 2.0);
        round_trip(&s, &[0.0, 1.3, PI, 6.0], &[-1.4, -0.5, 0.0, 0.7, 1.4]);
        // A pole has no meaningful u, but the point is real and must invert.
        let north = s.point(0.0, FRAC_PI_2);
        let (_, v) = s.invert(north).expect("poles still invert");
        assert!((v - FRAC_PI_2).abs() < 1e-9);
    }

    #[test]
    fn cone_round_trips_and_normal_is_perpendicular_to_the_ruling() {
        let s = Surface::cone([0.0, 0.0, 4.0], [0.0, 0.0, -1.0], FRAC_PI_4);
        round_trip(&s, &[0.0, 2.0, 4.0], &[0.5, 1.0, 3.0]);

        for &u in &[0.0, 1.1, 4.0] {
            let (p0, p1) = (s.point(u, 1.0), s.point(u, 2.0));
            let ruling = v3::normalize(v3::sub(p1, p0)).unwrap();
            let n = s.normal(u, 1.5).unwrap();
            assert!(v3::dot(n, ruling).abs() < 1e-12, "normal not ⊥ ruling");
            assert!((v3::norm(n) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn cone_from_rim_matches_the_half_angle_form() {
        let a = Surface::cone_from_rim([0.0, 0.0, 2.0], [0.0, 0.0, -1.0], 2.0, 2.0).unwrap();
        let b = Surface::cone([0.0, 0.0, 2.0], [0.0, 0.0, -1.0], FRAC_PI_4);
        assert_eq!(a, b);
        assert!(Surface::cone_from_rim([0.0; 3], [0.0, 0.0, 1.0], 0.0, 1.0).is_none());
    }

    #[test]
    fn torus_round_trips() {
        let s = Surface::torus([2.0, 0.0, 1.0], [0.0, 1.0, 0.0], 5.0, 1.5);
        round_trip(&s, &[0.0, 1.0, 3.0, 6.0], &[-2.0, 0.0, 2.0]);
        // Dead centre of the hole is `major` from the tube.
        assert!((s.distance([2.0, 0.0, 1.0]) - (5.0 - 1.5)).abs() < 1e-12);
    }

    #[test]
    fn quadric_normals_point_outward() {
        let cases = [
            Surface::sphere([0.0; 3], 2.0),
            Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0),
            Surface::torus([0.0; 3], [0.0, 0.0, 1.0], 5.0, 1.0),
        ];
        for s in &cases {
            for &u in &[0.0, 1.0, 2.5, 4.0] {
                let p = s.point(u, 0.0);
                let n = s.normal(u, 0.0).unwrap();
                // Outward: moving along the normal increases distance from the axis.
                let out = s.point(u, 0.0);
                let moved = v3::add(out, v3::scale(n, 1e-4));
                let r0 = (p[0] * p[0] + p[1] * p[1]).sqrt();
                let r1 = (moved[0] * moved[0] + moved[1] * moved[1]).sqrt();
                assert!(r1 > r0, "{}: normal points inward at u = {u}", s.kind());
            }
        }
    }

    #[test]
    fn nurbs_inversion_finds_the_closest_point() {
        let s = Surface::nurbs(crate::nurbs::construct::torus(
            [0.0; 3],
            [0.0, 0.0, 1.0],
            4.0,
            1.0,
        ));
        for i in 0..12 {
            for j in 0..12 {
                let (tu, tv) = (TAU * i as f64 / 12.0, TAU * j as f64 / 12.0);
                let target = Surface::torus([0.0; 3], [0.0, 0.0, 1.0], 4.0, 1.0).point(tu, tv);
                let (u, v) = s.invert(target).expect("inversion converges");
                let d = v3::dist(s.point(u, v), target);
                assert!(d < 1e-9, "inverted point is {d} away");
            }
        }
    }

    #[test]
    fn nurbs_distance_is_zero_on_the_surface_and_positive_off_it() {
        let s = Surface::nurbs(crate::nurbs::construct::sphere([0.0; 3], 1.0));
        assert!(s.distance([1.0, 0.0, 0.0]) < 1e-9);
        assert!((s.distance([3.0, 0.0, 0.0]) - 2.0).abs() < 1e-6);
    }

    // -- transforms --------------------------------------------------------

    #[test]
    fn rigid_transform_preserves_the_analytic_type() {
        let m =
            Matrix4::translation(Vector3::new(1.0, 2.0, 3.0)).multiply(&Matrix4::from_quaternion(
                crate::math::Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), 0.7),
            ));
        let s = Surface::sphere([0.0; 3], 2.0);
        let t = s.transform(&m).expect("rigid motion is representable");
        assert!(matches!(t, Surface::Sphere { radius, .. } if (radius - 2.0).abs() < 1e-6));
    }

    #[test]
    fn uniform_scale_scales_the_radius() {
        let m = Matrix4::scale(Vector3::new(3.0, 3.0, 3.0));
        let t = Surface::torus([0.0; 3], [0.0, 0.0, 1.0], 5.0, 1.0)
            .transform(&m)
            .expect("similarity is representable");
        match t {
            Surface::Torus { major, minor, .. } => {
                assert!((major - 15.0).abs() < 1e-6);
                assert!((minor - 3.0).abs() < 1e-6);
            }
            other => panic!("expected a torus, got {}", other.kind()),
        }
    }

    #[test]
    fn non_uniform_scale_drops_a_quadric_rather_than_lying_about_it() {
        let m = Matrix4::scale(Vector3::new(1.0, 2.0, 1.0));
        assert!(
            Surface::sphere([0.0; 3], 1.0).transform(&m).is_none(),
            "a squashed sphere is an ellipsoid, not a sphere with some radius"
        );
        assert!(Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 1.0)
            .transform(&m)
            .is_none());
    }

    #[test]
    fn a_plane_survives_shear_with_the_inverse_transpose_normal() {
        // Shear x by y: the plane z = 0 is unchanged, but the plane x = 0 tilts.
        let mut m = Matrix4::identity();
        m.elements[4] = 0.5; // column 1, row 0
        let s = Surface::plane([0.0; 3], [1.0, 0.0, 0.0]);
        let t = s.transform(&m).expect("planes survive any affine map");

        // Sample the original plane, push the points through the transform, and
        // check they land on the transformed plane. Using the matrix directly on
        // the normal would fail this.
        for &(u, v) in &[(1.0, 0.0), (0.0, 1.0), (-2.0, 3.0), (4.0, -1.5)] {
            let p = s.point(u, v);
            let q = [p[0] + 0.5 * p[1], p[1], p[2]];
            assert!(
                t.distance(q) < 1e-9,
                "sheared point is {} off",
                t.distance(q)
            );
        }
    }

    #[test]
    fn nurbs_survives_any_affine_map() {
        let mut m = Matrix4::scale(Vector3::new(1.0, 2.0, 3.0));
        m.elements[4] = 0.25;
        let s = Surface::nurbs(crate::nurbs::construct::sphere([0.0; 3], 1.0));
        let t = s
            .transform(&m)
            .expect("NURBS control points transform exactly");

        for i in 0..8 {
            for j in 0..8 {
                let (u, v) = (i as f64 / 8.0, j as f64 / 8.0);
                let p = s.point(u, v);
                let want = [p[0] * 1.0 + 0.25 * p[1], p[1] * 2.0, p[2] * 3.0];
                assert!(v3::dist(t.point(u, v), want) < 1e-9);
            }
        }
    }

    #[test]
    fn projective_and_singular_matrices_are_refused() {
        let mut proj = Matrix4::identity();
        proj.elements[3] = 0.5;
        assert!(Surface::sphere([0.0; 3], 1.0).transform(&proj).is_none());

        let flat = Matrix4::scale(Vector3::new(1.0, 1.0, 0.0));
        assert!(Surface::plane([0.0; 3], [0.0, 0.0, 1.0])
            .transform(&flat)
            .is_none());
    }
}
