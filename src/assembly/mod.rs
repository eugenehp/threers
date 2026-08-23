//! Generic checks for assemblies that move.
//!
//! Nothing here knows what it is inspecting. These are the predicates a model
//! needs to answer three questions that geometry alone cannot:
//!
//!   * **Does this body stay attached?** A joint coming apart produces no
//!     collision, so an interference test passes just as loudly for a hinge that
//!     has separated as for one that has not. [`crate::assembly::engagement`] watches the gap
//!     instead, across every pose.
//!   * **What does this body actually turn about?** Recovered from two poses,
//!     rather than taken on trust from the transform that produced them. A
//!     hand-derived chain and the hardware drawn to match it are two copies of
//!     one fact, and this is how you find out they have drifted.
//!   * **Where is there room?** Across a whole sweep, not at sampled poses. A
//!     part placed beside a moving joint needs the free volume found FIRST.
//!
//! | Question | Function |
//! |---|---|
//! | What bodies are in this soup? | [`crate::assembly::shells`] |
//! | Which body is which, across poses? | [`crate::assembly::rigid_key`], [`crate::assembly::correspond`], [`crate::assembly::align`] |
//! | What does it turn about? | [`crate::assembly::recover_screw`], [`crate::assembly::recover_axis`] |
//! | How close do two bodies come? | [`crate::assembly::nearest`], [`crate::assembly::interfere`], [`crate::assembly::inside`] |
//! | Does the joint stay together? | [`crate::assembly::engagement`] |
//! | Is anything unattached? | [`crate::assembly::floating`] |
//! | Where is there room to put something? | [`crate::assembly::free_annuli`] |
//!
//! # Correspondence
//!
//! Every cross-pose check needs to know which body in pose B is which body in
//! pose A, and the obvious answer -- position in the list -- is wrong: a rebuilt
//! model hands its shells back in a different order. [`crate::assembly::rigid_key`] identifies a
//! body by properties that survive rigid motion, so it can be found again.
//! Identical parts still tie, which is what [`crate::assembly::correspond`] resolves.
//!
//! # Two assumptions, and what happens when they fail
//!
//! **Retriangulation.** [`crate::assembly::rigid_key`] survives it: every quantity in the key is
//! an exact integral over the surface, so cutting a face into four does not move
//! it. [`crate::assembly::recover_axis`] and [`crate::assembly::recover_screw`] do *not* — they read vertex *i* of
//! one pose against vertex *i* of the other, which is only meaningful when the
//! mesh was moved rather than rebuilt. They do not fail silently: the fit misses
//! by about the size of the body and says so in its residual.
//!
//! **Closure.** [`crate::assembly::inside`] and the volume term of [`crate::assembly::rigid_key`] want a closed,
//! consistently wound surface. On an open one, [`crate::assembly::winding`] returns something
//! near neither 0 nor 1 and the containment question has no answer.

mod mesh;
mod motion;
mod pose;

pub use mesh::{
    inside, interfere, nearest, point_tri_distance, seg_seg_distance, shells, tri_bbs,
    tri_tri_distance, tri_tri_intersect, winding,
};
pub use motion::{classify, recover_screw, ParamMotion, Screw};
pub use pose::{align, engagement, engagement_with, floating, floating_with, Body, Engagement};

/// A triangle, three points.
pub type Tri = [[f64; 3]; 3];

/// Triangles out of a drawn mesh.
///
/// The one place this module looks at anything but numbers, and deliberately the
/// only one: the predicates take `&[Tri]` so they can be pointed at a boolean
/// result, an imported STL, a simulated pose or a hand-written array without
/// caring which. This is the adapter for the common case, not a dependency of
/// the checks.
///
/// Widened to `f64` on the way through. The checks measure gaps between parts
/// that are drawn to a thousandth of a millimetre and positioned in metres, and
/// `f32` runs out of significant figures somewhere in the middle of that.
///
/// ```
/// use threers::assembly::{from_geometry, shells};
/// use threers::geometries::BoxGeometry;
///
/// let tris = from_geometry(&BoxGeometry::new(1.0, 1.0, 1.0));
/// assert_eq!(tris.len(), 12);
/// assert_eq!(shells(&tris, 0.0).len(), 1);
/// ```
pub fn from_geometry(geometry: &crate::core::BufferGeometry) -> Vec<Tri> {
    let Some(pos) = geometry.get_attribute("position") else {
        return Vec::new();
    };
    let point = |i: usize| -> [f64; 3] {
        let b = i * 3;
        [
            pos.array[b] as f64,
            pos.array[b + 1] as f64,
            pos.array[b + 2] as f64,
        ]
    };
    let count = pos.array.len() / 3;
    match &geometry.index {
        Some(index) => index
            .chunks_exact(3)
            .filter(|t| t.iter().all(|&i| (i as usize) < count))
            .map(|t| {
                [
                    point(t[0] as usize),
                    point(t[1] as usize),
                    point(t[2] as usize),
                ]
            })
            .collect(),
        None => (0..count / 3)
            .map(|t| [point(t * 3), point(t * 3 + 1), point(t * 3 + 2)])
            .collect(),
    }
}

/// The same, moved by a transform.
///
/// For reading a body's successive poses out of a simulation: the mesh is
/// evaluated once and only ever moved, so the vertices go on corresponding —
/// which is exactly what [`crate::assembly::recover_screw`] needs and cannot check for itself.
pub fn from_geometry_at(
    geometry: &crate::core::BufferGeometry,
    transform: &crate::math::Matrix4,
) -> Vec<Tri> {
    from_geometry(geometry)
        .into_iter()
        .map(|t| {
            t.map(|p| {
                let v = crate::math::Vector3::new(p[0] as f32, p[1] as f32, p[2] as f32)
                    .apply_matrix4(transform);
                [v.x as f64, v.y as f64, v.z as f64]
            })
        })
        .collect()
}

/// Axis-aligned bounds.
pub fn aabb(tris: &[Tri]) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for t in tris {
        for v in t {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
    }
    (lo, hi)
}

/// Centroid of a body's vertices.
///
/// The cheap one, and not the one to match bodies with: cutting a face into four
/// leaves the shape alone and moves this, because it counts vertices rather than
/// surface. [`centroid_area`] is the one that survives that.
pub fn centroid(tris: &[Tri]) -> [f64; 3] {
    let n = (tris.len() * 3).max(1) as f64;
    let mut c = [0.0; 3];
    for t in tris {
        for v in t {
            for k in 0..3 {
                c[k] += v[k] / n;
            }
        }
    }
    c
}

/// Centroid of a body's *surface*, weighted by area.
///
/// The centre a rebuilt mesh still agrees on. A triangle's own centroid is its
/// vertex mean and its weight is its area, so subdividing a face contributes
/// exactly what the face contributed — which is what makes this survive a
/// boolean kernel running twice and handing back different triangles.
pub fn centroid_area(tris: &[Tri]) -> [f64; 3] {
    let mut total = 0.0;
    let mut acc = [0.0; 3];
    for t in tris {
        let a = tri_area(t);
        total += a;
        for k in 0..3 {
            acc[k] += a * (t[0][k] + t[1][k] + t[2][k]) / 3.0;
        }
    }
    if total <= 0.0 {
        return centroid(tris);
    }
    [acc[0] / total, acc[1] / total, acc[2] / total]
}

/// Total surface area.
pub fn area(tris: &[Tri]) -> f64 {
    tris.iter().map(tri_area).sum()
}

/// Volume enclosed by a closed surface.
///
/// Measured about the body's own centroid, so it is translation-invariant even
/// on a mesh that is not quite closed — where the usual divergence-theorem sum
/// about the origin would drift with position and be useless as an identity.
pub fn volume(tris: &[Tri]) -> f64 {
    let c = centroid_area(tris);
    let mut v = 0.0;
    for t in tris {
        let a = [t[0][0] - c[0], t[0][1] - c[1], t[0][2] - c[2]];
        let b = [t[1][0] - c[0], t[1][1] - c[1], t[1][2] - c[2]];
        let d = [t[2][0] - c[0], t[2][1] - c[1], t[2][2] - c[2]];
        v += (a[0] * (b[1] * d[2] - b[2] * d[1]) - a[1] * (b[0] * d[2] - b[2] * d[0])
            + a[2] * (b[0] * d[1] - b[1] * d[0]))
            / 6.0;
    }
    v
}

fn tri_area(t: &Tri) -> f64 {
    let u = [t[1][0] - t[0][0], t[1][1] - t[0][1], t[1][2] - t[0][2]];
    let v = [t[2][0] - t[0][0], t[2][1] - t[0][1], t[2][2] - t[0][2]];
    let c = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt() / 2.0
}

/// A body's identity, insofar as rigid motion preserves it.
///
/// Five numbers, every one of them an exact integral over the surface and every
/// one of them a **length**, so a single relative tolerance compares them all:
///
/// | Field | From |
/// |---|---|
/// | [`area_scale`](Self::area_scale) | the square root of total area |
/// | [`volume_scale`](Self::volume_scale) | the cube root of enclosed volume |
/// | [`moments`](Self::moments) | the three rotation invariants of the surface's second-moment tensor |
///
/// # Why not the obvious things
///
/// **Triangle count** is invariant under rigid motion and worthless as an
/// identity anyway: a boolean kernel run twice on the same solid is free to
/// triangulate it differently, and then the same part has two identities. Every
/// quantity here is an integral, so subdividing a face changes none of them.
///
/// **Bounding-box diagonal** is invariant too, and is one number where these are
/// five — but the real problem is quantising it. A key rounded to a fixed step
/// is a key whose behaviour depends on what units the model is drawn in, and
/// whose two sides of a bucket boundary are a different body. These are compared
/// with a *relative* tolerance instead, by [`RigidKey::matches`].
///
/// # Mirror images
///
/// Area, volume and every moment invariant are the same for a part and its
/// reflection -- no rotation turns one into the other, but none of those numbers
/// can tell. [`handedness`](Self::handedness) is the one that can, and it is the
/// only field here that is not a length.
///
/// # What still ties
///
/// Bodies of identical shape, deliberately -- they are genuinely
/// indistinguishable by shape, and [`crate::assembly::correspond`] separates them by continuity
/// instead. So do bodies with no handedness to speak of, which is most simple
/// ones: a symmetric part *is* its own mirror image, and there is nothing to
/// tell apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidKey {
    /// Square root of the total surface area.
    pub area_scale: f64,
    /// Cube root of the enclosed volume. Zero for an open surface.
    pub volume_scale: f64,
    /// The trace, second invariant and determinant of the second-moment tensor
    /// about the centroid, each reduced to a length by taking the appropriate
    /// root. Descending.
    pub moments: [f64; 3],
    /// Which way round the body is: `1`, `-1`, or `0` for neither.
    ///
    /// A reflection reverses it and a rotation does not, which is what makes it
    /// the field that separates a left-hand part from a right-hand one.
    ///
    /// Zero means the question has no answer for this body, and two such bodies
    /// match anything. That covers a genuinely symmetric part — which is its own
    /// mirror image, so there is nothing to distinguish — and one whose
    /// principal axes are not determined, such as anything with rotational
    /// symmetry. A cube is zero on both counts.
    pub handedness: i8,
}

impl RigidKey {
    /// How far apart two keys may be and still be the same body, as a fraction.
    ///
    /// A part re-evaluated by a boolean kernel moves its vertices by something
    /// like a part in a million; a millionth of a length is a millionth of every
    /// number here, because they are all lengths. Ten thousandths is four orders
    /// of margin over that and still refuses two parts that differ by a tenth of
    /// a percent in any dimension.
    pub const TOLERANCE: f64 = 1e-4;

    /// Whether two keys describe the same body, within [`Self::TOLERANCE`].
    pub fn matches(&self, other: &Self) -> bool {
        self.matches_within(other, Self::TOLERANCE)
    }

    /// The same, with the tolerance given.
    ///
    /// Deliberately not [`PartialEq`]: a tolerant comparison is not transitive,
    /// and `==` promising something it cannot deliver is worse than a method
    /// that says what it does. `PartialEq` on this type is exact.
    pub fn matches_within(&self, other: &Self, tol: f64) -> bool {
        // Judge everything against the body's overall size, so a moment that is
        // legitimately near zero -- a flat plate has one -- is compared
        // absolutely rather than being asked for a relative agreement it cannot
        // have.
        let scale = self.area_scale.max(other.area_scale);
        let close = |a: f64, b: f64| (a - b).abs() <= tol * scale.max(a.abs()).max(b.abs());
        // A body with no handedness matches either hand: it has no hand to
        // disagree about.
        let same_hand =
            self.handedness == other.handedness || self.handedness == 0 || other.handedness == 0;
        same_hand
            && close(self.area_scale, other.area_scale)
            && close(self.volume_scale, other.volume_scale)
            && (0..3).all(|k| close(self.moments[k], other.moments[k]))
    }
}

/// Identify a body by what rigid motion cannot change about it.
///
/// See [`RigidKey`] for what is in the key and what it cannot tell apart.
pub fn rigid_key(tris: &[Tri]) -> RigidKey {
    let c = centroid_area(tris);
    let mut total_area = 0.0;
    // Second moment of the surface about the origin, shifted to the centroid.
    // Exact per triangle, which is what makes it survive retriangulation:
    // ∫ x xᵀ dA = (A/12)(Σ vᵢvᵢᵀ + s sᵀ), for s = Σ vᵢ.
    let mut m = [[0.0f64; 3]; 3];
    for t in tris {
        let a = tri_area(t);
        if a <= 0.0 {
            continue;
        }
        total_area += a;
        let v: [[f64; 3]; 3] = [
            [t[0][0] - c[0], t[0][1] - c[1], t[0][2] - c[2]],
            [t[1][0] - c[0], t[1][1] - c[1], t[1][2] - c[2]],
            [t[2][0] - c[0], t[2][1] - c[1], t[2][2] - c[2]],
        ];
        let s = [
            v[0][0] + v[1][0] + v[2][0],
            v[0][1] + v[1][1] + v[2][1],
            v[0][2] + v[1][2] + v[2][2],
        ];
        for i in 0..3 {
            for j in 0..3 {
                let sum: f64 = (0..3).map(|k| v[k][i] * v[k][j]).sum();
                m[i][j] += (a / 12.0) * (sum + s[i] * s[j]);
            }
        }
    }

    // The characteristic polynomial's coefficients are rotation invariants, and
    // cost no eigen-solve to obtain.
    let trace = m[0][0] + m[1][1] + m[2][2];
    let second = (m[0][0] * m[1][1] - m[0][1] * m[1][0])
        + (m[0][0] * m[2][2] - m[0][2] * m[2][0])
        + (m[1][1] * m[2][2] - m[1][2] * m[2][1]);
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    // Each has the units of a length to some power; take the root so they are
    // all lengths and one tolerance serves.
    RigidKey {
        area_scale: total_area.max(0.0).sqrt(),
        volume_scale: cbrt(volume(tris).abs()),
        moments: [
            root(trace.abs(), 4.0),
            root(second.abs(), 8.0),
            root(det.abs(), 12.0),
        ],
        handedness: handedness(tris, c, m, total_area),
    }
}

/// Which way round a body is: `1`, `-1`, or `0` when it has no way round.
///
/// The second-moment tensor gives three principal axes, but an eigenvector and
/// its negation are the same eigenvector, so the frame they make has no
/// handedness until something about the body chooses the signs. The **third**
/// moment along each axis is what chooses them: it is odd, so it distinguishes
/// the two ends of an axis, and it is a property of the shape rather than of the
/// numbering.
///
/// Point each axis the way its third moment is positive, and take the
/// determinant of the resulting frame. A rotation carries the shape and its axes
/// together, so the third moments and the determinant are unchanged. A
/// reflection carries them together too — but reflection has determinant −1, so
/// the frame comes out the other way round. That difference is the whole
/// mechanism.
///
/// # When there is no answer
///
/// Two of them, and both return zero rather than a coin toss:
///
/// * **Repeated eigenvalues.** Anything with rotational symmetry has an
///   eigenplane rather than two eigenvectors, and which pair comes out of the
///   solve is arbitrary. A cube has all three equal.
/// * **A vanishing third moment.** A body symmetric about a principal plane has
///   no odd moment across it, so that axis has no preferred end. Which is the
///   same as saying the body equals its own mirror image, and has no handedness
///   to report.
fn handedness(tris: &[Tri], c: [f64; 3], m: [[f64; 3]; 3], total_area: f64) -> i8 {
    if total_area <= 0.0 {
        return 0;
    }
    let (values, mut axes) = motion::symmetric_eigen(m);
    let scale = values[0].abs();
    if scale <= 0.0 {
        return 0;
    }
    // An eigenvalue gap smaller than this is a symmetry, not a shape.
    const DISTINCT: f64 = 1e-6;
    if (values[0] - values[1]).abs() <= DISTINCT * scale
        || (values[1] - values[2]).abs() <= DISTINCT * scale
    {
        return 0;
    }

    // A third moment is an area times a length cubed; judge it against one
    // built from this body, so the test means the same at any scale.
    let length = root(scale, 4.0);
    let floor = total_area * length * length * length * 1e-6;
    for axis in axes.iter_mut() {
        let m3 = third_moment(tris, c, *axis);
        if m3.abs() <= floor {
            return 0; // symmetric across this axis: no preferred end
        }
        if m3 < 0.0 {
            *axis = [-axis[0], -axis[1], -axis[2]];
        }
    }

    let det = axes[0][0] * (axes[1][1] * axes[2][2] - axes[1][2] * axes[2][1])
        - axes[0][1] * (axes[1][0] * axes[2][2] - axes[1][2] * axes[2][0])
        + axes[0][2] * (axes[1][0] * axes[2][1] - axes[1][1] * axes[2][0]);
    if det > 0.0 {
        1
    } else if det < 0.0 {
        -1
    } else {
        0
    }
}

/// `∫ (x·e)³ dA` over the surface, about `c`.
///
/// Exact per triangle, like everything else in the key: for a linear `f` over a
/// triangle of area `A`, `∫ f³ dA` is `A/10` times the complete homogeneous
/// symmetric polynomial of the three vertex values. So subdividing a face
/// changes nothing, and a rebuilt mesh keeps its handedness.
fn third_moment(tris: &[Tri], c: [f64; 3], e: [f64; 3]) -> f64 {
    let mut acc = 0.0;
    for t in tris {
        let a = tri_area(t);
        if a <= 0.0 {
            continue;
        }
        let f: [f64; 3] = std::array::from_fn(|i| {
            (t[i][0] - c[0]) * e[0] + (t[i][1] - c[1]) * e[1] + (t[i][2] - c[2]) * e[2]
        });
        let (x, y, z) = (f[0], f[1], f[2]);
        let h3 = x * x * x
            + y * y * y
            + z * z * z
            + x * x * y
            + x * x * z
            + y * y * x
            + y * y * z
            + z * z * x
            + z * z * y
            + x * y * z;
        acc += a / 10.0 * h3;
    }
    acc
}

fn root(x: f64, n: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        x.powf(1.0 / n)
    }
}

fn cbrt(x: f64) -> f64 {
    root(x, 3.0)
}

/// Match bodies between two poses.
///
/// Same key, then nearest centroid within the tie group. The second step is why
/// the poses should be CLOSE: over a small step a body cannot have travelled
/// further than its identical neighbours are apart, and the match is unambiguous.
/// Over a large one it is a guess -- which is what [`crate::assembly::align`] avoids by walking
/// adjacent poses rather than reaching across the whole sweep.
///
/// Greedy, in the order the first pose lists its bodies. Where several bodies
/// are equally good candidates that order decides, which is another reason to
/// keep the step small enough that there is only ever one real candidate.
///
/// Bodies with no counterpart are simply absent from the result.
pub fn correspond(a: &[Vec<Tri>], b: &[Vec<Tri>]) -> Vec<(usize, usize)> {
    let ka: Vec<_> = a.iter().map(|s| (rigid_key(s), centroid_area(s))).collect();
    let kb: Vec<_> = b.iter().map(|s| (rigid_key(s), centroid_area(s))).collect();
    let mut used = vec![false; b.len()];
    let mut out = Vec::new();
    for (i, (k, c)) in ka.iter().enumerate() {
        let mut best: Option<(f64, usize)> = None;
        for (j, (k2, c2)) in kb.iter().enumerate() {
            if used[j] || !k.matches(k2) {
                continue;
            }
            let d = (0..3).map(|m| (c[m] - c2[m]).powi(2)).sum::<f64>();
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, j));
            }
        }
        if let Some((_, j)) = best {
            used[j] = true;
            out.push((i, j));
        }
    }
    out
}

/// How far a fit may miss, as a fraction of the body's own radius, before it is
/// reported as no fit at all.
///
/// A real mesh carries rounding; a mesh whose vertices are not the same vertices
/// carries the whole body. There are orders of magnitude between those, so the
/// threshold does not need to be delicate — it needs to exist.
const MAX_RESIDUAL: f64 = 1e-2;

/// What a body turns about, recovered from two poses.
///
/// `normal` is the axis direction to solve in -- the rotation is assumed to be
/// about it. Returns a point on the axis and the angle turned, in radians.
/// [`crate::assembly::recover_screw`] is the version that finds the direction too, and reports
/// any slide along it.
///
/// `None` when:
///
/// * the two poses have different triangle counts;
/// * the body did not turn enough to locate an axis. The fixed-point solve has a
///   determinant of order theta squared, so a small angle amplifies rounding into
///   an axis far outside the model. Sample poses far enough apart to turn several
///   degrees;
/// * **the fit does not hold.** The angle comes from every vertex at once, so a
///   rotation that suits them all is a real one; a rotation that suits none is
///   what you get when the mesh was rebuilt between poses and vertex *i* is no
///   longer the same vertex. That case used to return an axis anyway.
pub fn recover_axis(before: &[Tri], after: &[Tri], normal: [f64; 3]) -> Option<([f64; 3], f64)> {
    if before.is_empty() || before.len() != after.len() {
        return None;
    }
    let n = {
        let l = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
        if l <= 0.0 {
            return None;
        }
        [normal[0] / l, normal[1] / l, normal[2] / l]
    };
    let ca = centroid_area(before);
    let cb = centroid_area(after);
    let planar = motion::recover_in_plane(before, after, ca, cb, n)?;
    if planar.angle.abs() < 1e-3 {
        return None; // not turning, or not enough to locate an axis
    }
    let scale = motion::radius(before, ca);
    if scale > 0.0 && planar.residual > scale * MAX_RESIDUAL {
        return None; // these vertices are not related by this rotation
    }
    Some((planar.point, planar.angle))
}

/// Free annuli about an axis, unioned over every pose given.
///
/// Conservative: a triangle marks every bin its bounding box touches, so a band
/// reported free IS free. Returns, per z bin, the widest run of empty radial
/// bins -- which is where a part may be placed beside a moving joint.
///
/// # How pessimistic
///
/// An annulus is free only if it is free at *every* angle, because that is what
/// an annulus is. A pocket that is clear on one side of the axis and blocked on
/// the other reports as blocked. That is the safe direction to be wrong in — a
/// band reported free is free — but on a mechanism that is not roughly
/// rotationally symmetric about the axis, expect it to report less room than
/// there is.
///
/// The bounding-box marking compounds this: a triangle lying at 45° across the
/// grid claims every bin its box spans, which is more than it occupies. Finer
/// bins reduce it, at the usual cost.
///
/// Returns nothing rather than panicking when asked for zero bins, a
/// non-positive step, or a non-positive radius.
#[allow(clippy::too_many_arguments)]
pub fn free_annuli(
    poses: &[Vec<Tri>],
    axis_at: [f64; 3],
    normal: [f64; 3],
    r_max: f64,
    bins: usize,
    z_lo: f64,
    z_bins: usize,
    z_step: f64,
) -> Vec<(f64, f64, f64)> {
    if bins == 0 || z_bins == 0 || z_step <= 0.0 || r_max <= 0.0 || !r_max.is_finite() {
        return Vec::new();
    }
    let n = {
        let l = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
        if l <= 0.0 {
            return Vec::new();
        }
        [normal[0] / l, normal[1] / l, normal[2] / l]
    };
    let (u, v) = plane_basis(n);
    let bw = r_max / bins as f64;
    let mut occ = vec![vec![false; bins]; z_bins];

    for pose in poses {
        for t in pose {
            let (mut r0, mut r1) = (f64::MAX, 0.0f64);
            let (mut a0, mut a1) = (f64::MAX, f64::MIN);
            for p in t {
                let d = [p[0] - axis_at[0], p[1] - axis_at[1], p[2] - axis_at[2]];
                let r = dot3(&d, &u).hypot(dot3(&d, &v));
                let h = dot3(&d, &n);
                r0 = r0.min(r);
                r1 = r1.max(r);
                a0 = a0.min(h);
                a1 = a1.max(h);
            }
            if r0 > r_max {
                continue;
            }
            let zi0 = (((a0 - z_lo) / z_step).floor().max(0.0)) as usize;
            let zi1 = (((a1 - z_lo) / z_step).ceil().max(0.0)) as usize;
            let ri0 = ((r0 / bw).floor().max(0.0)) as usize;
            let ri1 = ((r1 / bw).ceil().max(0.0)) as usize;
            for row in occ.iter_mut().take(zi1.min(z_bins - 1) + 1).skip(zi0) {
                for slot in row.iter_mut().take(ri1.min(bins - 1) + 1).skip(ri0) {
                    *slot = true;
                }
            }
        }
    }

    let mut out = Vec::new();
    for (zi, row) in occ.iter().enumerate() {
        if !row.iter().any(|x| *x) {
            continue;
        }
        let (mut best, mut bs, mut run, mut rs) = (0usize, 0usize, 0usize, 0usize);
        for (ri, o) in row.iter().enumerate() {
            if *o {
                run = 0;
            } else {
                if run == 0 {
                    rs = ri;
                }
                run += 1;
                if run > best {
                    best = run;
                    bs = rs;
                }
            }
        }
        if best > 0 {
            out.push((
                z_lo + zi as f64 * z_step,
                bs as f64 * bw,
                (bs + best) as f64 * bw,
            ));
        }
    }
    out
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn plane_basis(n: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let t = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = unit3(cross3(t, n));
    (u, unit3(cross3(n, u)))
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit3(a: [f64; 3]) -> [f64; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if l > 0.0 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        [1.0, 0.0, 0.0]
    }
}

#[cfg(test)]
mod tests;

// -----------------------------------------------------------------------------
// Swept occupancy
// -----------------------------------------------------------------------------
// "Can this part go here?" answered by drawing it and running an interference
// sweep is trial and error, and it does not converge: six attempts to fit a
// spring leg and a bonding leaf around one hinge failed against five different
// occupants, each discovered only by colliding with it. The question the model
// should be able to answer is the inverse one -- what does this mechanism NEVER
// occupy -- and that is a property of the motion, not of any candidate part.
//
// Voxels rather than CSG on purpose. A union of a hundred posed solids is the
// operation exact kernels are worst at, and the answer wanted here is "is there
// room", which does not need exact boundaries. Marking is CONSERVATIVE: a
// triangle marks every cell its bounding box touches, so a cell reported free IS
// free, and the error is always toward reporting less room than exists.

/// What a mechanism occupies over its whole travel, on a regular grid.
pub struct Occupancy {
    lo: [f64; 3],
    cell: f64,
    n: [usize; 3],
    bits: Vec<bool>,
}

impl Occupancy {
    /// An empty grid covering `lo`..`hi` at `cell` resolution.
    pub fn new(lo: [f64; 3], hi: [f64; 3], cell: f64) -> Self {
        let n = [
            (((hi[0] - lo[0]) / cell).ceil().max(1.0)) as usize,
            (((hi[1] - lo[1]) / cell).ceil().max(1.0)) as usize,
            (((hi[2] - lo[2]) / cell).ceil().max(1.0)) as usize,
        ];
        Occupancy {
            lo,
            cell,
            n,
            bits: vec![false; n[0] * n[1] * n[2]],
        }
    }

    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.n[1] + j) * self.n[0] + i
    }

    /// Mark everything this pose occupies. Call once per sampled step of the
    /// motion; the union across calls is the swept volume.
    pub fn add_pose(&mut self, tris: &[Tri]) {
        for t in tris {
            let mut mn = [f64::INFINITY; 3];
            let mut mx = [f64::NEG_INFINITY; 3];
            for v in t {
                for c in 0..3 {
                    mn[c] = mn[c].min(v[c]);
                    mx[c] = mx[c].max(v[c]);
                }
            }
            let mut a = [0usize; 3];
            let mut b = [0usize; 3];
            let mut skip = false;
            for c in 0..3 {
                let f0 = ((mn[c] - self.lo[c]) / self.cell).floor();
                let f1 = ((mx[c] - self.lo[c]) / self.cell).ceil();
                if f1 < 0.0 || f0 >= self.n[c] as f64 {
                    skip = true;
                    break;
                }
                a[c] = f0.max(0.0) as usize;
                b[c] = (f1 as usize).min(self.n[c] - 1);
            }
            if skip {
                continue;
            }
            for k in a[2]..=b[2] {
                for j in a[1]..=b[1] {
                    for i in a[0]..=b[0] {
                        let x = self.idx(i, j, k);
                        self.bits[x] = true;
                    }
                }
            }
        }
    }

    /// Is this point never occupied, at any sampled step?
    pub fn free_at(&self, p: [f64; 3]) -> bool {
        let mut c = [0usize; 3];
        for a in 0..3 {
            let f = ((p[a] - self.lo[a]) / self.cell).floor();
            if f < 0.0 || f >= self.n[a] as f64 {
                return true; // outside the grid is outside the mechanism
            }
            c[a] = f as usize;
        }
        !self.bits[self.idx(c[0], c[1], c[2])]
    }

    /// Fraction of the grid nothing ever passes through.
    pub fn free_fraction(&self) -> f64 {
        let f = self.bits.iter().filter(|b| !**b).count();
        f as f64 / self.bits.len().max(1) as f64
    }

    /// The largest empty box in the grid, as (lo, hi) in model units.
    ///
    /// This is the question a part wants answered before it is drawn: not "does
    /// my candidate collide" but "what is the biggest thing that fits, and
    /// where". Greedy expansion from each free cell -- not provably maximal, but
    /// it does not report a box that is occupied, which is the property that
    /// matters.
    pub fn largest_free_box(&self) -> Option<([f64; 3], [f64; 3])> {
        let mut best: Option<(usize, [usize; 3], [usize; 3])> = None;
        for k in 0..self.n[2] {
            for j in 0..self.n[1] {
                for i in 0..self.n[0] {
                    if self.bits[self.idx(i, j, k)] {
                        continue;
                    }
                    let (mut ei, mut ej, mut ek) = (i, j, k);
                    loop {
                        let mut grew = false;
                        if ei + 1 < self.n[0] && self.clear(i, ei + 1, j, ej, k, ek) {
                            ei += 1;
                            grew = true;
                        }
                        if ej + 1 < self.n[1] && self.clear(i, ei, j, ej + 1, k, ek) {
                            ej += 1;
                            grew = true;
                        }
                        if ek + 1 < self.n[2] && self.clear(i, ei, j, ej, k, ek + 1) {
                            ek += 1;
                            grew = true;
                        }
                        if !grew {
                            break;
                        }
                    }
                    let vol = (ei - i + 1) * (ej - j + 1) * (ek - k + 1);
                    if best.is_none_or(|(v, _, _)| vol > v) {
                        best = Some((vol, [i, j, k], [ei, ej, ek]));
                    }
                }
            }
        }
        best.map(|(_, a, b)| {
            (
                [
                    self.lo[0] + a[0] as f64 * self.cell,
                    self.lo[1] + a[1] as f64 * self.cell,
                    self.lo[2] + a[2] as f64 * self.cell,
                ],
                [
                    self.lo[0] + (b[0] + 1) as f64 * self.cell,
                    self.lo[1] + (b[1] + 1) as f64 * self.cell,
                    self.lo[2] + (b[2] + 1) as f64 * self.cell,
                ],
            )
        })
    }

    fn clear(&self, i0: usize, i1: usize, j0: usize, j1: usize, k0: usize, k1: usize) -> bool {
        for k in k0..=k1 {
            for j in j0..=j1 {
                for i in i0..=i1 {
                    if self.bits[self.idx(i, j, k)] {
                        return false;
                    }
                }
            }
        }
        true
    }
}
