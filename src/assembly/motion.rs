//! Recovering what a body did, from where it was and where it ended up.
//!
//! Every rigid motion is a screw: a turn about an axis and a slide along it.
//! That is Chasles' theorem, and it is why one function covers a hinge, a
//! slider, a lead screw and a part that was simply moved — they are the same
//! thing with two of the four numbers set to zero.
//!
//! # Why recover it at all
//!
//! Because the transform that produced the poses and the hardware drawn to match
//! it are two copies of one fact, and copies drift. A chain that says the elbow
//! is at `[0, 40, 8]` and a bracket whose bore is at `[0, 40, 7.5]` both look
//! right on their own. Recovering the axis from the geometry asks the parts
//! where they actually turn, which is the only account that cannot be out of
//! date.

use super::{centroid_area, Tri};

/// A rigid motion, in the form every rigid motion takes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Screw {
    /// A point the axis passes through. Meaningless when `angle` is zero —
    /// a pure translation has no located axis, only a direction.
    pub point: [f64; 3],
    /// Unit direction of the axis.
    pub direction: [f64; 3],
    /// Turn about the axis, in radians, right-handed about `direction`.
    pub angle: f64,
    /// Travel along the axis. Zero for a hinge, non-zero for anything helical.
    pub slide: f64,
    /// How far the fit missed, in model units — the RMS distance between where
    /// this motion puts each vertex and where it actually ended up.
    ///
    /// Worth reading rather than discarding. A residual that is not small
    /// against the body means the two poses are not related by a rigid motion
    /// at all: the mesh was re-evaluated and the vertices no longer correspond,
    /// or the "body" is really two bodies that moved differently.
    pub residual: f64,
}

impl Screw {
    /// Whether this is a hinge: turning, not sliding.
    pub fn is_revolute(&self, tol: f64) -> bool {
        self.angle.abs() > tol && self.slide.abs() <= tol
    }

    /// Whether this is a slider: sliding, not turning.
    pub fn is_prismatic(&self, tol: f64) -> bool {
        self.angle.abs() <= tol && self.slide.abs() > tol
    }

    /// Travel along the axis per radian turned — a thread's lead. Infinite for
    /// a pure slide, zero for a pure turn.
    pub fn lead(&self) -> f64 {
        if self.angle.abs() < 1e-12 {
            f64::INFINITY
        } else {
            self.slide / self.angle
        }
    }
}

/// Recover the whole motion between two poses of one body, direction included.
///
/// # What it needs
///
/// **The same vertices in the same order.** The two poses must be the same mesh
/// moved, not the same *shape* re-evaluated — this reads vertex *i* of one
/// against vertex *i* of the other, and a boolean kernel run twice is free to
/// hand back its triangles in a different order.
///
/// That requirement is not a hope. When it is violated the fit misses by roughly
/// the size of the body, and [`Screw::residual`] says so; a caller that checks it
/// cannot be quietly told the wrong axis. [`recover_axis`](super::recover_axis)
/// carries the same guarantee.
///
/// Returns `None` when the poses have different triangle counts, or when the
/// body did not move at all.
///
/// ```
/// use threers::assembly::recover_screw;
/// # fn tri(p: [f64; 3]) -> [[f64; 3]; 3] { [p, [p[0] + 1.0, p[1], p[2]], [p[0], p[1] + 1.0, p[2]]] }
/// // A quarter turn about the z axis through the origin.
/// let before = vec![tri([2.0, 0.0, 0.0])];
/// let after = vec![[[0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [-1.0, 2.0, 0.0]]];
/// let s = recover_screw(&before, &after).unwrap();
/// assert!((s.angle - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
/// assert!(s.direction[2].abs() > 0.99);
/// assert!(s.slide.abs() < 1e-9);
/// ```
pub fn recover_screw(before: &[Tri], after: &[Tri]) -> Option<Screw> {
    if before.is_empty() || before.len() != after.len() {
        return None;
    }
    let ca = centroid_area(before);
    let cb = centroid_area(after);

    // The displacement of every point of a rigid body has the SAME component
    // along the screw axis — that component is the slide. So the differences
    // between displacements are all perpendicular to the axis, and the axis is
    // the direction those differences never point in: the smallest eigenvector
    // of their covariance.
    let pts: Vec<([f64; 3], [f64; 3])> = before
        .iter()
        .zip(after.iter())
        .flat_map(|(a, b)| a.iter().copied().zip(b.iter().copied()))
        .collect();

    let mut mean = [0.0; 3];
    for (p, q) in &pts {
        for k in 0..3 {
            mean[k] += (q[k] - p[k]) / pts.len() as f64;
        }
    }
    let mut cov = [[0.0f64; 3]; 3];
    for (p, q) in &pts {
        let d = [
            q[0] - p[0] - mean[0],
            q[1] - p[1] - mean[1],
            q[2] - p[2] - mean[2],
        ];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }

    let spread = cov[0][0] + cov[1][1] + cov[2][2];
    let scale = radius(before, ca).max(1e-12);
    // Every point moved by the same vector: a pure translation, which has a
    // direction and no located axis.
    //
    // The threshold is a millionth of the body rather than a billionth for a
    // reason: a slider's displacements agree only to float precision, and a
    // test tight enough to reject that sends a pure slide down the rotation
    // path, where it recovers an axis out of rounding and reports the slide as
    // a screw.
    if spread <= (scale * 1e-6).powi(2) * pts.len() as f64 {
        let slide = norm(mean);
        if slide < 1e-12 {
            return None; // it did not move
        }
        return Some(Screw {
            point: ca,
            direction: scaled(mean, 1.0 / slide),
            angle: 0.0,
            slide,
            residual: 0.0,
        });
    }

    let axis = smallest_eigenvector(cov)?;
    let planar = recover_in_plane(before, after, ca, cb, axis)?;
    Some(Screw {
        point: planar.point,
        direction: axis,
        angle: planar.angle,
        slide: dot(sub(cb, ca), axis),
        residual: planar.residual,
    })
}

/// The part of the motion that lies in the plane perpendicular to `axis`.
pub(super) struct Planar {
    pub point: [f64; 3],
    pub angle: f64,
    pub residual: f64,
}

/// Angle and centre of rotation in the plane perpendicular to `axis`.
///
/// The angle comes from every vertex at once rather than from the furthest one:
/// the least-squares rotation between two matched planar point sets is
/// `atan2(Σ p × q, Σ p · q)`, which is one pass and is not at the mercy of
/// whichever vertex happens to be the outlier.
pub(super) fn recover_in_plane(
    before: &[Tri],
    after: &[Tri],
    ca: [f64; 3],
    cb: [f64; 3],
    axis: [f64; 3],
) -> Option<Planar> {
    let (u, v) = basis(axis);
    let proj = |p: [f64; 3], o: [f64; 3]| {
        let d = sub(p, o);
        [dot(d, u), dot(d, v)]
    };

    let (mut cross_sum, mut dot_sum) = (0.0, 0.0);
    let mut count = 0usize;
    for (ta, tb) in before.iter().zip(after.iter()) {
        for (pa, pb) in ta.iter().zip(tb.iter()) {
            let a = proj(*pa, ca);
            let b = proj(*pb, cb);
            cross_sum += a[0] * b[1] - a[1] * b[0];
            dot_sum += a[0] * b[0] + a[1] * b[1];
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    let angle = cross_sum.atan2(dot_sum);

    // How far this rotation misses. A caller that skips this reading is a
    // caller that can be handed an axis derived from vertices that were never
    // the same vertices.
    let (co, si) = (angle.cos(), angle.sin());
    let mut residual = 0.0;
    for (ta, tb) in before.iter().zip(after.iter()) {
        for (pa, pb) in ta.iter().zip(tb.iter()) {
            let a = proj(*pa, ca);
            let b = proj(*pb, cb);
            let rx = co * a[0] - si * a[1];
            let ry = si * a[0] + co * a[1];
            residual += (rx - b[0]).powi(2) + (ry - b[1]).powi(2);
        }
    }
    let residual = (residual / count as f64).sqrt();

    // Where the axis crosses the plane: the fixed point of the motion.
    //
    // The determinant below is 2(1 − cos θ), which is of order θ². A body that
    // barely turned therefore locates its axis barely at all, and the answer
    // runs off to infinity rather than being merely imprecise. Sampling poses
    // far enough apart to turn several degrees is not a nicety.
    let cap = proj(ca, ca);
    let cbp = proj(cb, ca);
    let det = 2.0 * (1.0 - co);
    if det < 1e-9 {
        return None;
    }
    let dx = cbp[0] - (co * cap[0] - si * cap[1]);
    let dy = cbp[1] - (si * cap[0] + co * cap[1]);
    let cx = ((1.0 - co) * dx - si * dy) / det;
    let cy = (si * dx + (1.0 - co) * dy) / det;
    Some(Planar {
        point: [
            ca[0] + u[0] * cx + v[0] * cy,
            ca[1] + u[1] * cx + v[1] * cy,
            ca[2] + u[2] * cx + v[2] * cy,
        ],
        angle,
        residual,
    })
}

/// Largest distance from `c` to any vertex — the body's own scale, for judging
/// whether a residual is small.
pub(super) fn radius(tris: &[Tri], c: [f64; 3]) -> f64 {
    let mut r: f64 = 0.0;
    for t in tris {
        for v in t {
            r = r.max(norm(sub(*v, c)));
        }
    }
    r
}

// ---------------------------------------------------------------------------
// Symmetric 3x3 eigen-decomposition
// ---------------------------------------------------------------------------

/// Eigenvector of the smallest eigenvalue of a symmetric matrix.
///
/// Closed form (Smith's method) rather than an iterative solve: a 3×3 symmetric
/// matrix has its eigenvalues in the roots of a cubic, and the cubic has a
/// trigonometric solution. No dependency, no iteration count to tune, and it
/// cannot fail to converge.
fn smallest_eigenvector(a: [[f64; 3]; 3]) -> Option<[f64; 3]> {
    eigenvector(a, symmetric_eigenvalues(a)[2])
}

/// Eigenvalues (descending) and their eigenvectors.
///
/// The eigenvectors are unit and in the same order as the values, but their
/// *signs* are arbitrary — an eigenvector and its negation are the same
/// eigenvector. Any caller that cares which way they point has to fix that
/// itself, from something about the body; see [`super::rigid_key`], which does
/// exactly that to work out a handedness.
pub(super) fn symmetric_eigen(a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let values = symmetric_eigenvalues(a);
    let mut vectors = [[0.0; 3]; 3];
    for (i, &l) in values.iter().enumerate() {
        vectors[i] = eigenvector(a, l).unwrap_or([0.0; 3]);
    }
    (values, vectors)
}

/// A unit vector in the null space of `a - lambda I`.
fn eigenvector(a: [[f64; 3]; 3], lambda: f64) -> Option<[f64; 3]> {
    let shifted = [
        [a[0][0] - lambda, a[0][1], a[0][2]],
        [a[1][0], a[1][1] - lambda, a[1][2]],
        [a[2][0], a[2][1], a[2][2] - lambda],
    ];
    // The null space is what two independent rows cross into. Whichever pair
    // gives the longest product is the best conditioned.
    let mut best: Option<([f64; 3], f64)> = None;
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        let c = cross(shifted[i], shifted[j]);
        let l = norm(c);
        if best.is_none_or(|(_, bl)| l > bl) {
            best = Some((c, l));
        }
    }
    let (v, l) = best?;
    (l > 1e-300).then(|| scaled(v, 1.0 / l))
}

/// Eigenvalues of a symmetric 3×3, largest first.
fn symmetric_eigenvalues(a: [[f64; 3]; 3]) -> [f64; 3] {
    let p1 = a[0][1].powi(2) + a[0][2].powi(2) + a[1][2].powi(2);
    let q = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
    if p1 <= 0.0 {
        // Already diagonal.
        let mut d = [a[0][0], a[1][1], a[2][2]];
        d.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
        return d;
    }
    let p2 = (a[0][0] - q).powi(2) + (a[1][1] - q).powi(2) + (a[2][2] - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    if p <= 0.0 {
        return [q, q, q];
    }
    let b = [
        [(a[0][0] - q) / p, a[0][1] / p, a[0][2] / p],
        [a[1][0] / p, (a[1][1] - q) / p, a[1][2] / p],
        [a[2][0] / p, a[2][1] / p, (a[2][2] - q) / p],
    ];
    let det = b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
        - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
        + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]);
    let r = (det / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;
    let e1 = q + 2.0 * p * phi.cos();
    let e3 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
    [e1, 3.0 * q - e1 - e3, e3]
}

// ---------------------------------------------------------------------------

fn basis(n: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let t = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = unit(cross(t, n));
    (u, unit(cross(n, u)))
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scaled(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let l = norm(a);
    if l > 0.0 {
        scaled(a, 1.0 / l)
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// What a part does when a model PARAMETER changes.
///
/// A parametric model states motion once, in the geometry: a body inside
/// `translate([EJECT, 0, 0])` moves when `EJECT` does. Anything downstream that
/// also needs to know — a renderer posing meshes per frame, a collision pass
/// building a broadphase, a physics step assigning bodies to a rigid group —
/// tends to restate it, usually by matching a part's NAME. That restatement is
/// a second copy of the fact, and copies drift.
///
/// The failure it drifts into has a shape worth naming. When one named part
/// contains bodies belonging to DIFFERENT motion groups, no name-based rule can
/// describe it: whatever the rule decides is wrong for half the part. Pose it
/// and the fixed half moves; leave it and the moving half stays behind. The
/// stranded half then exists only downstream, so nothing that checks the MODEL
/// can see it — a collision pass over the geometry reports the corridor clear
/// while the render shows a panel standing in it.
///
/// [`Mixed`](ParamMotion::Mixed) is that case, and it is why this enum exists:
/// a part that cannot be posed as one body should be split, and the only way to
/// find out is to ask the geometry rather than the naming.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamMotion {
    /// Unchanged. The parameter does not reach this part.
    Fixed,
    /// Moved as ONE rigid body. Safe to pose downstream by this screw.
    Rigid(Screw),
    /// The mesh is the same but no single rigid motion fits it: the part holds
    /// bodies that moved differently, and it cannot be posed as a unit.
    Mixed {
        /// Best-fit motion, kept because its residual is the evidence.
        screw: Screw,
        /// RMS miss of that fit, in model units.
        residual: f64,
    },
    /// The mesh itself changed. The parameter RESHAPES this part rather than
    /// posing it — a telescoping mast, a fold that re-cuts a relief — so there
    /// is no pose to hand downstream and it must be re-evaluated per state.
    Reshaped {
        /// Triangle count before and after.
        before: usize,
        /// Triangle count after.
        after: usize,
    },
}

impl ParamMotion {
    /// Whether a downstream consumer can pose this part with a single
    /// transform. False for both [`Mixed`](ParamMotion::Mixed) and
    /// [`Reshaped`](ParamMotion::Reshaped), for different reasons.
    pub fn is_poseable(&self) -> bool {
        matches!(self, ParamMotion::Fixed | ParamMotion::Rigid(_))
    }
}

/// Classify what a parameter did to a part, from its mesh before and after.
///
/// `tol` is in model units and is compared against the fit residual, so it
/// should be the smallest displacement worth calling motion — not a fraction.
/// A part is [`Fixed`](ParamMotion::Fixed) when the best-fit screw neither
/// turns nor slides beyond `tol`.
///
/// The two meshes must come from evaluating the SAME source at two parameter
/// values, so that triangle order corresponds. A differing triangle count is
/// reported as [`Reshaped`](ParamMotion::Reshaped) rather than guessed at.
pub fn classify(before: &[Tri], after: &[Tri], tol: f64) -> ParamMotion {
    if before.len() != after.len() {
        return ParamMotion::Reshaped {
            before: before.len(),
            after: after.len(),
        };
    }
    // FIXED IS TESTED FIRST, and it has to be: recover_screw fits a motion, and
    // there is no motion to fit when nothing moved. It returns None on an
    // unmoved body, and reading that as "no rigid motion describes this" would
    // report every static class in the model as a defect -- which is exactly
    // what the first version of this did, on thirteen of nineteen classes.
    let moved = before.iter().zip(after).any(|(b, a)| {
        b.iter()
            .zip(a)
            .any(|(p, q)| (0..3).any(|i| (p[i] - q[i]).abs() > tol))
    });
    if !moved {
        return ParamMotion::Fixed;
    }
    let Some(screw) = recover_screw(before, after) else {
        // Same mesh, something moved, and no screw fits it. That is the
        // two-motion-groups case with no best-fit to report.
        return ParamMotion::Mixed {
            screw: Screw {
                point: [0.0; 3],
                direction: [0.0, 0.0, 1.0],
                angle: 0.0,
                slide: 0.0,
                residual: f64::INFINITY,
            },
            residual: f64::INFINITY,
        };
    };
    if screw.residual > tol {
        return ParamMotion::Mixed {
            screw,
            residual: screw.residual,
        };
    }
    ParamMotion::Rigid(screw)
}
