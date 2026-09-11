//! The region a lattice fills: a signed field, and the box it lives in.
//!
//! A [`Lattice`](super::Lattice) on its own fills a box, because a box is what
//! marching cubes needs to close the mesh against. A `Region` is how it fills
//! anything else. It carries two things:
//!
//! - a field that is **positive inside** and negative out, so intersecting it
//!   with the lattice is a `min` and the mesh closes over the cut ends;
//! - the bounds it occupies, so [`Lattice::fill`](super::Lattice::fill) can
//!   size the sample grid without being told twice.
//!
//! ```
//! use threers::{Lattice, LatticeKind, Region, Tpms, Vector3};
//!
//! // A gyroid poured into a sphere with two bites taken out of it.
//! let shape = Region::sphere(Vector3::ZERO, 10.0)
//!     .difference(Region::capsule(
//!         Vector3::new(-12.0, 0.0, 0.0),
//!         Vector3::new(12.0, 0.0, 0.0),
//!         3.0,
//!     ))
//!     .difference(Region::sphere(Vector3::new(0.0, 9.0, 0.0), 4.0));
//!
//! let geom = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
//!     .fill(shape)
//!     .cell_size(Vector3::new(4.0, 4.0, 4.0))
//!     .thickness(0.8)
//!     .build();
//! assert!(geom.index.is_some());
//! ```
//!
//! # Exact, and approximate
//!
//! The primitives are true signed distances, so `offset` moves a surface by the
//! length you give it and the lattice's own thickness stays honest right up to
//! the cut. The combinators are not: `min` and `max` are exact on the surface
//! that wins but under-estimate near a seam, which is the usual and harmless
//! property of constructive distance fields — the zero set is exactly right,
//! and that is what the mesh is built from.

use crate::curves::Curve3;
use crate::math::{Box3, Matrix4, Quaternion, Vector3};
use std::sync::Arc;

use super::strut::segment_distance;

/// A closed region of space, as a field that is positive inside it.
///
/// See the module docs.
///
/// Cloning one is a reference count, not a copy of the field — which matters
/// because a region is routinely wanted twice. Filling a shell and conforming
/// the cells to the same surface is two uses of one region, and rebuilding it
/// for the second is a second BVH or a second CSG evaluation.
///
/// ```
/// use threers::{Conform, Lattice, LatticeKind, Region, Tpms, Vector3};
///
/// let shell = Region::sphere(Vector3::ZERO, 10.0)
///     .difference(Region::sphere(Vector3::ZERO, 6.0));
///
/// let geom = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
///     .conform(Conform::depth(shell.clone()))
///     .fill(shell)
///     .cell_size(Vector3::new(4.0, 4.0, 2.0))
///     .thickness(0.5)
///     .build();
/// # assert!(geom.index.is_some());
/// ```
#[derive(Clone)]
pub struct Region<'a> {
    inside: Arc<dyn Fn(Vector3) -> f32 + Sync + Send + 'a>,
    bounds: Box3,
}

impl<'a> Region<'a> {
    /// Wrap your own field. Positive inside, negative outside, and zero on the
    /// surface you want.
    ///
    /// The bounds are yours to get right: nothing outside them is sampled, so a
    /// box that clips the field is a region that comes out clipped. Only the
    /// sign has to be correct far from the surface; near it, the field should
    /// behave like a distance, or the cut will land in the wrong place.
    pub fn new(bounds: Box3, inside: impl Fn(Vector3) -> f32 + Sync + Send + 'a) -> Self {
        Self {
            inside: Arc::new(inside),
            bounds,
        }
    }

    /// The box the region fits inside.
    pub fn bounds(&self) -> Box3 {
        self.bounds
    }

    /// Distance into the region — positive inside, negative outside.
    pub fn distance(&self, p: Vector3) -> f32 {
        (self.inside)(p)
    }

    /// Whether a point is inside.
    pub fn contains(&self, p: Vector3) -> bool {
        self.distance(p) > 0.0
    }

    pub(super) fn into_field(self) -> Arc<dyn Fn(Vector3) -> f32 + Sync + Send + 'a> {
        self.inside
    }

    // --- primitives ---

    /// A ball.
    pub fn sphere(center: Vector3, radius: f32) -> Self {
        let radius = radius.max(0.0);
        Self::new(
            Box3::from_center_and_size(center, Vector3::new(1.0, 1.0, 1.0) * (radius * 2.0)),
            move |p| radius - (p - center).length(),
        )
    }

    /// An axis-aligned box.
    pub fn cuboid(bounds: Box3) -> Self {
        let center = bounds.center();
        let half = bounds.size() * 0.5;
        Self::new(bounds, move |p| {
            // Exact either side: outside, the distance to the nearest corner or
            // face; inside, how far the nearest face is.
            let q = Vector3::new(
                (p.x - center.x).abs() - half.x,
                (p.y - center.y).abs() - half.y,
                (p.z - center.z).abs() - half.z,
            );
            let outside = Vector3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).length();
            let inside = q.x.max(q.y).max(q.z).min(0.0);
            -(outside + inside)
        })
    }

    /// A box with its edges rounded off.
    ///
    /// The radius is capped at half the shortest side, so the result never
    /// grows past the box it was given — round a 2 mm cube by 5 mm and you get
    /// a 2 mm ball, not a 10 mm one. Inset-then-offset is only the same shape
    /// as a rounded box while there is something left to inset.
    pub fn rounded_cuboid(bounds: Box3, radius: f32) -> Self {
        let size = bounds.size();
        let radius = radius
            .max(0.0)
            .min(0.5 * size.x.min(size.y).min(size.z).max(0.0));
        let shrunk = Box3::from_center_and_size(
            bounds.center(),
            Vector3::new(
                (bounds.size().x - 2.0 * radius).max(0.0),
                (bounds.size().y - 2.0 * radius).max(0.0),
                (bounds.size().z - 2.0 * radius).max(0.0),
            ),
        );
        Self::cuboid(shrunk).offset(radius)
    }

    /// A capsule: everything within `radius` of the segment `a`–`b`, so the
    /// ends are round.
    pub fn capsule(a: Vector3, b: Vector3, radius: f32) -> Self {
        let radius = radius.max(0.0);
        let mut bounds = Box3::from_points(&[a, b]);
        bounds.expand_by_scalar(radius);
        Self::new(bounds, move |p| radius - segment_distance(p, a, b))
    }

    /// A cylinder from `a` to `b`, with flat ends.
    pub fn cylinder(a: Vector3, b: Vector3, radius: f32) -> Self {
        Self::cone(a, b, radius, radius)
    }

    /// A cone frustum from `a` to `b`, `radius_a` at one end and `radius_b` at
    /// the other. Equal radii give a cylinder; one of them zero gives a point.
    pub fn cone(a: Vector3, b: Vector3, radius_a: f32, radius_b: f32) -> Self {
        let (ra, rb) = (radius_a.max(0.0), radius_b.max(0.0));
        Self::new(frustum_bounds(a, b, ra, rb), move |p| {
            let ba = b - a;
            let baba = ba.length_sq();
            if baba <= f32::EPSILON {
                return ra.max(rb) - (p - a).length();
            }
            // In the (radial, axial) half-plane the frustum is a trapezoid, and
            // the nearest point on it is either on a cap or on the slanted
            // side. Both candidates are measured, and the nearer one wins —
            // taking only the side would over-state the distance past a rim,
            // where the true nearest point is the rim itself.
            let pa = p - a;
            let paba = pa.dot(ba) / baba;
            let radial = (pa.length_sq() - paba * paba * baba).max(0.0).sqrt();

            // To the nearer cap's rim.
            let cap_radial = (radial - if paba < 0.5 { ra } else { rb }).max(0.0);
            let cap_axial = (paba - 0.5).abs() - 0.5;

            // To the slanted side, via the foot of the perpendicular.
            let rba = rb - ra;
            let k = rba * rba + baba;
            let foot = ((rba * (radial - ra) + paba * baba) / k).clamp(0.0, 1.0);
            let side_radial = radial - ra - foot * rba;
            let side_axial = paba - foot;

            let inside = side_radial < 0.0 && cap_axial < 0.0;
            let squared = (cap_radial * cap_radial + cap_axial * cap_axial * baba)
                .min(side_radial * side_radial + side_axial * side_axial * baba);
            if inside {
                squared.sqrt()
            } else {
                -squared.sqrt()
            }
        })
    }

    /// A torus in the plane `z = center.z`, of the given ring and tube radii.
    /// Use [`rotate`](Self::rotate) to stand it up.
    pub fn torus(center: Vector3, ring: f32, tube: f32) -> Self {
        let (ring, tube) = (ring.max(0.0), tube.max(0.0));
        let bounds = Box3::from_center_and_size(
            center,
            Vector3::new(2.0 * (ring + tube), 2.0 * (ring + tube), 2.0 * tube),
        );
        Self::new(bounds, move |p| {
            let d = p - center;
            let radial = (d.x * d.x + d.y * d.y).sqrt() - ring;
            tube - (radial * radial + d.z * d.z).sqrt()
        })
    }

    /// Everything on the `-normal` side of a plane through `point`, clipped to
    /// `bounds` — a half-space is infinite, and a lattice has to be sampled
    /// somewhere.
    pub fn half_space(bounds: Box3, point: Vector3, normal: Vector3) -> Self {
        let n = normal.normalize();
        Self::new(bounds, move |p| -(p - point).dot(n))
    }

    /// A tube of the given radius following a curve.
    ///
    /// The curve is sampled into `segments` straight pieces, so the tube is
    /// exact on each piece and cuts the corner between them by the sagitta —
    /// raise `segments` where the curve turns tightly.
    pub fn tube(curve: &dyn Curve3, radius: f32, segments: usize) -> Self {
        let radius = radius.max(0.0);
        let points = curve.get_points(segments.max(1));
        let mut bounds = Box3::from_points(&points);
        bounds.expand_by_scalar(radius);
        Self::new(bounds, move |p| {
            let mut nearest = f32::MAX;
            for pair in points.windows(2) {
                nearest = nearest.min(segment_distance(p, pair[0], pair[1]));
            }
            radius - nearest
        })
    }

    // --- combinators ---

    /// Everything in either region.
    pub fn union(self, other: Region<'a>) -> Self {
        let bounds = self.bounds.union(&other.bounds);
        let (a, b) = (self.inside, other.inside);
        Self::new(bounds, move |p| a(p).max(b(p)))
    }

    /// Everything in both.
    pub fn intersection(self, other: Region<'a>) -> Self {
        let bounds = self.bounds.intersect(&other.bounds);
        let (a, b) = (self.inside, other.inside);
        Self::new(bounds, move |p| a(p).min(b(p)))
    }

    /// This region with the other cut out of it.
    pub fn difference(self, other: Region<'a>) -> Self {
        let bounds = self.bounds;
        let (a, b) = (self.inside, other.inside);
        Self::new(bounds, move |p| a(p).min(-b(p)))
    }

    /// A union whose seam is filleted rather than creased, over a radius of
    /// about `blend`.
    pub fn smooth_union(self, other: Region<'a>, blend: f32) -> Self {
        let bounds = self.bounds.union(&other.bounds);
        let (a, b) = (self.inside, other.inside);
        let k = blend.max(1e-6);
        Self::new(bounds, move |p| {
            let (x, y) = (a(p), b(p));
            let h = (0.5 + 0.5 * (x - y) / k).clamp(0.0, 1.0);
            // Lerp between the two fields, then push out by the amount the
            // blend rounds the corner.
            y + (x - y) * h + k * h * (1.0 - h)
        })
    }

    /// Grow the region by `distance`, or shrink it with a negative one.
    pub fn offset(self, distance: f32) -> Self {
        let mut bounds = self.bounds;
        bounds.expand_by_scalar(distance);
        let f = self.inside;
        Self::new(bounds, move |p| f(p) + distance)
    }

    /// Replace the region with a shell of the given thickness on its surface,
    /// half in and half out.
    pub fn shell(self, thickness: f32) -> Self {
        let half = thickness.abs() * 0.5;
        let mut bounds = self.bounds;
        bounds.expand_by_scalar(half);
        let f = self.inside;
        Self::new(bounds, move |p| half - f(p).abs())
    }

    /// Swap inside for outside. The bounds are kept as they were — the
    /// complement of a bounded region is not bounded, so intersect it with
    /// something before filling it.
    pub fn invert(self) -> Self {
        let bounds = self.bounds;
        let f = self.inside;
        Self::new(bounds, move |p| -f(p))
    }

    /// Move the region.
    pub fn translate(self, offset: Vector3) -> Self {
        let bounds = self.bounds.translate(offset);
        let f = self.inside;
        Self::new(bounds, move |p| f(p - offset))
    }

    /// Scale about the origin. Distances scale with it, so the field stays a
    /// distance.
    pub fn scale(self, factor: f32) -> Self {
        let s = if factor.abs() < 1e-9 {
            1e-9
        } else {
            factor.abs()
        };
        let bounds = Box3::new(self.bounds.min * s, self.bounds.max * s);
        let f = self.inside;
        Self::new(bounds, move |p| f(p * (1.0 / s)) * s)
    }

    /// Rotate about the origin.
    pub fn rotate(self, rotation: Quaternion) -> Self {
        let inverse = rotation.invert();
        let bounds = self
            .bounds
            .apply_matrix4(&Matrix4::from_quaternion(rotation));
        let f = self.inside;
        Self::new(bounds, move |p| f(p.apply_quaternion(inverse)))
    }

    /// Place the region by a full transform — a scene-graph `matrix_world`, for
    /// instance.
    ///
    /// Exact for rigid motions and uniform scales. A non-uniform scale leaves
    /// the sign right and the surface in the right place, but the field is no
    /// longer a distance: it reads short along the stretched axis, so an
    /// `offset` on top of one will not move the surface by the length asked
    /// for. Compose out of [`translate`](Self::translate),
    /// [`rotate`](Self::rotate) and [`scale`](Self::scale) when that matters.
    pub fn transform(self, matrix: &Matrix4) -> Self {
        let inverse = matrix.invert();
        // How much the transform shrinks a length, at worst — the field is
        // divided by it, so the result never over-states a distance.
        let axis = |c: [f32; 3]| (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        let m = matrix.elements;
        let scale = axis([m[0], m[1], m[2]])
            .max(axis([m[4], m[5], m[6]]))
            .max(axis([m[8], m[9], m[10]]))
            .max(1e-9);
        let bounds = self.bounds.apply_matrix4(matrix);
        let f = self.inside;
        Self::new(bounds, move |p| f(p.apply_matrix4(&inverse)) * scale)
    }

    // --- meshes ---

    /// The inside of a closed triangle mesh.
    ///
    /// Distance comes from the nearest point on the surface; the sign comes
    /// from counting how many triangles a ray from the point crosses, which is
    /// odd inside and even outside. Both are BVH queries, so the cost per
    /// sample is logarithmic in the triangle count rather than linear — but it
    /// is still a great deal more than an analytic primitive, and a lattice
    /// samples millions of points. Build these at a resolution you have the
    /// patience for, and turn on the crate's `parallel` feature.
    ///
    /// The mesh has to be closed for the parity test to mean anything. A mesh
    /// with holes will read as inside on one side of them.
    #[cfg(feature = "mesh-bvh")]
    pub fn mesh(geometry: &crate::core::BufferGeometry) -> Option<Self> {
        use crate::mesh_bvh::{BuildOptions, MeshBvh};
        let bvh = MeshBvh::build(geometry, BuildOptions::default())?;
        Some(Self::from_bvh(bvh))
    }

    /// The inside of a mesh with a BVH already built over it.
    #[cfg(feature = "mesh-bvh")]
    pub fn from_bvh(bvh: crate::mesh_bvh::MeshBvh) -> Self {
        use crate::math::Ray;
        let bounds = bvh.bounding_box();
        let reach = bounds.size().length() + 1.0;
        Self::new(bounds, move |p| {
            let (_, distance, _) = bvh.closest_point_to_point(p);
            // A direction off every axis and off the diagonals, so a ray is
            // unlikely to graze an edge or run along a face — which is where
            // parity counting goes wrong.
            let ray = Ray::new(p, Vector3::new(0.5773503, 0.3216, 0.7503).normalize());
            let crossings = bvh.raycast(&ray, 0.0, reach, false).len();
            if crossings % 2 == 1 {
                distance
            } else {
                -distance
            }
        })
    }

    /// The inside of an OpenSCAD [`Solid`](crate::openscad::Solid), evaluated
    /// through the exact-CSG kernel.
    #[cfg(feature = "openscad")]
    pub fn solid(solid: crate::openscad::Solid) -> Option<Self> {
        Self::mesh(&solid.to_geometry_exact())
    }

    /// The inside of an OpenSCAD program.
    ///
    /// ```no_run
    /// # #[cfg(feature = "openscad")] {
    /// use threers::{Lattice, LatticeKind, Region, Tpms, Vector3};
    ///
    /// let shape = Region::scad("difference(){ cube(40, center=true); sphere(24); }")
    ///     .expect("valid scad");
    /// let geom = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
    ///     .fill(shape)
    ///     .cells([6, 6, 6])
    ///     .thickness(1.0)
    ///     .build();
    /// # let _ = geom;
    /// # }
    /// ```
    #[cfg(feature = "openscad")]
    pub fn scad(source: &str) -> Result<Self, String> {
        let solid = crate::openscad::scad::parse_scad(source)?;
        Self::solid(solid).ok_or_else(|| "the model produced no geometry".to_string())
    }
}

/// The exact box around a cone frustum.
///
/// Not the endpoints grown by the radius: that is the box of a *capsule*, and
/// on a squat cylinder it overshoots by a radius at each end. A disc's extent
/// along an axis is its radius times the sine of the angle between that axis
/// and the disc's normal — zero along the axis it faces down.
fn frustum_bounds(a: Vector3, b: Vector3, radius_a: f32, radius_b: f32) -> Box3 {
    let axis = b - a;
    let length = axis.length();
    let d = if length > 1e-9 {
        axis * (1.0 / length)
    } else {
        Vector3::new(0.0, 0.0, 1.0)
    };
    let spread = |r: f32| {
        Vector3::new(
            r * (1.0 - d.x * d.x).max(0.0).sqrt(),
            r * (1.0 - d.y * d.y).max(0.0).sqrt(),
            r * (1.0 - d.z * d.z).max(0.0).sqrt(),
        )
    };
    let (sa, sb) = (spread(radius_a), spread(radius_b));
    Box3::from_points(&[a - sa, a + sa, b - sb, b + sb])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curves::CatmullRomCurve3;

    /// Sample a region over its own bounds and report what fraction is inside —
    /// the cheap way to check a field is the shape it claims to be.
    fn occupancy(region: &Region<'_>, n: usize) -> f32 {
        let b = region.bounds();
        let size = b.size();
        let mut inside = 0;
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let at = |a: usize, lo: f32, len: f32| lo + (a as f32 + 0.5) / n as f32 * len;
                    let p = Vector3::new(
                        at(i, b.min.x, size.x),
                        at(j, b.min.y, size.y),
                        at(k, b.min.z, size.z),
                    );
                    if region.contains(p) {
                        inside += 1;
                    }
                }
            }
        }
        inside as f32 / (n * n * n) as f32
    }

    #[test]
    fn a_sphere_fills_its_box_by_pi_over_six() {
        let s = Region::sphere(Vector3::new(1.0, -2.0, 0.5), 3.0);
        assert!((occupancy(&s, 48) - std::f32::consts::PI / 6.0).abs() < 0.01);
        assert!(s.contains(Vector3::new(1.0, -2.0, 0.5)));
        assert!(!s.contains(Vector3::new(1.0, -2.0, 4.0)));
        // And the field is a distance: two units in from the surface reads two.
        assert!((s.distance(Vector3::new(2.0, -2.0, 0.5)) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn a_cuboid_is_exactly_its_box() {
        let b = Box3::new(Vector3::new(-1.0, -2.0, -3.0), Vector3::new(1.0, 2.0, 3.0));
        let r = Region::cuboid(b);
        assert!((occupancy(&r, 32) - 1.0).abs() < 1e-6);
        // Distance to the nearest face, inside…
        assert!((r.distance(Vector3::ZERO) - 1.0).abs() < 1e-5);
        // …and to the nearest corner, outside.
        let corner = Vector3::new(4.0, 6.0, 3.0);
        assert!((r.distance(corner) + 5.0).abs() < 1e-4);
    }

    #[test]
    fn a_cylinder_holds_its_radius_and_its_ends() {
        let c = Region::cylinder(
            Vector3::new(0.0, 0.0, -2.0),
            Vector3::new(0.0, 0.0, 2.0),
            1.0,
        );
        assert!(c.contains(Vector3::new(0.9, 0.0, 0.0)));
        assert!(!c.contains(Vector3::new(1.1, 0.0, 0.0)));
        assert!(c.contains(Vector3::new(0.0, 0.0, 1.9)));
        assert!(!c.contains(Vector3::new(0.0, 0.0, 2.1)));
        // Volume is πr²h over a box of 2r × 2r × h.
        assert!((occupancy(&c, 48) - std::f32::consts::PI / 4.0).abs() < 0.01);
        // Distance is a length on the side and on the end cap alike.
        assert!((c.distance(Vector3::new(0.5, 0.0, 0.0)) - 0.5).abs() < 1e-4);
        assert!((c.distance(Vector3::new(0.0, 0.0, 1.5)) - 0.5).abs() < 1e-4);
    }

    #[test]
    fn a_cone_narrows() {
        let c = Region::cone(Vector3::ZERO, Vector3::new(0.0, 0.0, 4.0), 2.0, 0.0);
        assert!(c.contains(Vector3::new(1.9, 0.0, 0.05)));
        assert!(!c.contains(Vector3::new(1.9, 0.0, 2.0)));
        assert!(c.contains(Vector3::new(0.9, 0.0, 2.0)));
        // A cone is a third of its cylinder, which is π/4 of its box.
        assert!((occupancy(&c, 48) - std::f32::consts::PI / 12.0).abs() < 0.02);
    }

    #[test]
    fn a_torus_has_a_hole() {
        let t = Region::torus(Vector3::ZERO, 3.0, 1.0);
        assert!(!t.contains(Vector3::ZERO), "the hole is not filled");
        assert!(t.contains(Vector3::new(3.0, 0.0, 0.0)));
        assert!(!t.contains(Vector3::new(3.0, 0.0, 1.5)));
        assert!(t.contains(Vector3::new(0.0, -3.0, 0.5)));
    }

    #[test]
    fn a_tube_follows_its_curve() {
        let curve = CatmullRomCurve3::new(vec![
            Vector3::new(-4.0, 0.0, 0.0),
            Vector3::new(-1.0, 3.0, 0.0),
            Vector3::new(1.0, -3.0, 0.0),
            Vector3::new(4.0, 0.0, 0.0),
        ]);
        let t = Region::tube(&curve, 0.5, 64);
        for u in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let on = curve.get_point(u);
            assert!(
                t.contains(on),
                "the curve is not inside its own tube at {u}"
            );
            assert!(
                !t.contains(on + Vector3::new(0.0, 0.0, 0.8)),
                "the tube is too fat at {u}"
            );
        }
        assert!(t.bounds().contains_point(curve.get_point(0.5)));
    }

    #[test]
    fn combinators_do_what_they_say() {
        let a = Region::sphere(Vector3::new(-0.5, 0.0, 0.0), 1.0);
        let b = Region::sphere(Vector3::new(0.5, 0.0, 0.0), 1.0);

        let both = Region::sphere(Vector3::new(-0.5, 0.0, 0.0), 1.0)
            .intersection(Region::sphere(Vector3::new(0.5, 0.0, 0.0), 1.0));
        assert!(both.contains(Vector3::ZERO));
        assert!(!both.contains(Vector3::new(-1.2, 0.0, 0.0)));

        let cut = Region::sphere(Vector3::new(-0.5, 0.0, 0.0), 1.0)
            .difference(Region::sphere(Vector3::new(0.5, 0.0, 0.0), 1.0));
        assert!(!cut.contains(Vector3::ZERO));
        assert!(cut.contains(Vector3::new(-1.2, 0.0, 0.0)));

        let joined = a.union(b);
        assert!(joined.contains(Vector3::ZERO));
        assert!(joined.contains(Vector3::new(-1.2, 0.0, 0.0)));
        assert!(joined.contains(Vector3::new(1.2, 0.0, 0.0)));
        assert!(!joined.contains(Vector3::new(1.6, 0.0, 0.0)));
        // Bounds cover both.
        assert!(joined.bounds().contains_point(Vector3::new(1.49, 0.0, 0.0)));
    }

    #[test]
    fn offset_and_shell_move_by_the_length_given() {
        let grown = Region::sphere(Vector3::ZERO, 1.0).offset(0.5);
        assert!(grown.contains(Vector3::new(1.4, 0.0, 0.0)));
        assert!(!grown.contains(Vector3::new(1.6, 0.0, 0.0)));
        assert!(grown.bounds().contains_point(Vector3::new(1.5, 0.0, 0.0)));

        let hollow = Region::sphere(Vector3::ZERO, 2.0).shell(0.4);
        assert!(!hollow.contains(Vector3::ZERO), "a shell is hollow");
        assert!(hollow.contains(Vector3::new(2.0, 0.0, 0.0)));
        assert!(!hollow.contains(Vector3::new(1.6, 0.0, 0.0)));
    }

    #[test]
    fn transforms_move_the_region_and_its_bounds() {
        let moved = Region::sphere(Vector3::ZERO, 1.0).translate(Vector3::new(5.0, 0.0, 0.0));
        assert!(moved.contains(Vector3::new(5.0, 0.0, 0.0)));
        assert!(!moved.contains(Vector3::ZERO));
        assert!(moved.bounds().contains_point(Vector3::new(5.9, 0.0, 0.0)));

        let big = Region::sphere(Vector3::ZERO, 1.0).scale(3.0);
        assert!(big.contains(Vector3::new(2.9, 0.0, 0.0)));
        assert!(!big.contains(Vector3::new(3.1, 0.0, 0.0)));
        // Scaling scales the distance with it.
        assert!((big.distance(Vector3::ZERO) - 3.0).abs() < 1e-4);

        // A torus lies in xy; rotating a quarter turn about x stands it up.
        let upright = Region::torus(Vector3::ZERO, 3.0, 1.0).rotate(Quaternion::from_axis_angle(
            Vector3::new(1.0, 0.0, 0.0),
            std::f32::consts::FRAC_PI_2,
        ));
        assert!(upright.contains(Vector3::new(3.0, 0.0, 0.0)));
        assert!(upright.contains(Vector3::new(0.0, 0.0, 3.0)));
        assert!(!upright.contains(Vector3::new(0.0, 3.0, 0.0)));
    }

    #[test]
    fn a_general_transform_keeps_the_surface_in_place() {
        let m = Matrix4::compose(
            Vector3::new(2.0, -1.0, 0.5),
            Quaternion::from_axis_angle(Vector3::new(0.3, 1.0, 0.2).normalize(), 0.9),
            Vector3::new(2.0, 2.0, 2.0),
        );
        let moved = Region::sphere(Vector3::ZERO, 1.0).transform(&m);
        // The image of a point on the original surface is on the new one.
        for dir in [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ] {
            let on = dir.apply_matrix4(&m);
            assert!(moved.distance(on).abs() < 1e-3, "surface moved: {on:?}");
        }
        assert!(moved.contains(Vector3::new(2.0, -1.0, 0.5)));
    }

    #[test]
    fn a_rounded_cuboid_never_grows_past_its_box() {
        // Inset-then-offset is only a rounded box while there is something left
        // to inset. Asked for a 2 mm cube rounded by 5 mm, an unclamped radius
        // gives a 10 mm ball — bigger than the box it was handed, and with
        // bounds to match.
        let box3 = Box3::new(Vector3::new(-1.0, -1.0, -1.0), Vector3::new(1.0, 1.0, 1.0));
        let over = Region::rounded_cuboid(box3, 5.0);
        assert!((over.distance(Vector3::ZERO) - 1.0).abs() < 1e-5);
        assert!(
            !over.contains(Vector3::new(1.0, 1.0, 1.0)),
            "corner is outside a ball"
        );
        assert!(over.bounds().contains_point(Vector3::new(1.0, 1.0, 1.0)));
        assert!(!over.contains(Vector3::new(1.2, 0.0, 0.0)));

        // Under the cap it is the box with rounded edges: faces where the box's
        // are, corners pulled in.
        let rounded = Region::rounded_cuboid(box3, 0.4);
        assert!((rounded.distance(Vector3::ZERO) - 1.0).abs() < 1e-5);
        assert!(rounded.contains(Vector3::new(0.99, 0.0, 0.0)));
        assert!(!rounded.contains(Vector3::new(0.95, 0.95, 0.95)));
    }

    #[test]
    fn a_smooth_union_fills_the_crease_and_leaves_the_rest() {
        let a = || Region::sphere(Vector3::new(-0.6, 0.0, 0.0), 1.0);
        let b = || Region::sphere(Vector3::new(0.6, 0.0, 0.0), 1.0);
        let hard = a().union(b());
        let soft = a().smooth_union(b(), 0.5);

        // In the crease between the two, the blend adds material.
        let crease = Vector3::new(0.0, 0.92, 0.0);
        assert!(!hard.contains(crease));
        assert!(soft.contains(crease), "the blend did not fill the crease");

        // Away from it, the two agree to within the blend radius.
        for p in [
            Vector3::new(-1.5, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.9),
            Vector3::new(0.0, 3.0, 0.0),
        ] {
            assert!(
                (soft.distance(p) - hard.distance(p)).abs() <= 0.5,
                "the blend reached too far at {p:?}"
            );
        }
    }

    #[test]
    fn a_custom_field_is_taken_as_given() {
        let bounds = Box3::new(Vector3::new(-2.0, -2.0, -2.0), Vector3::new(2.0, 2.0, 2.0));
        let wavy = Region::new(bounds, |p| 1.0 + 0.5 * (p.x * 3.0).sin() - p.y.abs());
        assert!(wavy.contains(Vector3::ZERO));
        assert!(!wavy.contains(Vector3::new(0.0, 1.9, 0.0)));
        assert_eq!(wavy.bounds(), bounds);
    }
}
