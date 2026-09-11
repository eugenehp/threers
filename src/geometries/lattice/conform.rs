//! Conformal lattices — cells that follow the part instead of the world.
//!
//! [`Lattice::fill`](super::Lattice::fill) *trims*: the cells are laid out on a
//! world-space grid and whatever pokes out of the part is cut off. On a flat
//! box that is exactly right. On a curved shell it is not — the cut lands
//! mid-strut, so the surface is a rash of severed stubs that carry no load, and
//! the cells nearest the surface are whatever fraction of a cell happened to
//! fit.
//!
//! A conformal lattice maps the point into cell space first. Give the map
//! angle-around-the-axis for x and the tiling closes on itself around a nozzle;
//! give it depth-below-the-surface for z and every cell through a curved wall
//! is a whole cell, meeting the skin square.
//!
//! ```
//! use threers::{Conform, Lattice, LatticeKind, Region, Strut, Vector3};
//!
//! // Sixteen cells around a 20 mm nozzle, three through its wall.
//! let radius = 20.0;
//! let geom = Lattice::new(LatticeKind::Strut(Strut::Cubic))
//!     .fill(
//!         Region::cylinder(Vector3::new(0.0, 0.0, -15.0), Vector3::new(0.0, 0.0, 15.0), radius)
//!             .difference(Region::cylinder(
//!                 Vector3::new(0.0, 0.0, -16.0),
//!                 Vector3::new(0.0, 0.0, 16.0),
//!                 radius - 6.0,
//!             )),
//!     )
//!     .conform(Conform::cylindrical(Vector3::ZERO, Vector3::new(0.0, 0.0, 1.0), radius))
//!     .cell_size(Vector3::new(Conform::ring_pitch(radius, 16), 5.0, 2.0))
//!     .thickness(0.8)
//!     .build();
//! assert!(geom.index.is_some());
//! ```
//!
//! # What a map costs
//!
//! [`thickness`](super::Lattice::thickness) is a length in *cell* space, and a
//! map that is not an isometry does not preserve lengths. A cylindrical map
//! stretches the tangential direction by `reference / r`, so a wall asked for
//! at 0.8 mm comes out thinner than that inside the reference radius and
//! thicker outside it.
//!
//! Each map therefore reports a [`stretch`](Conform::stretch_at), and the
//! thickness is divided by it before the field is cut — so the wall is exactly
//! right wherever the map is an isometry, and the error elsewhere is the spread
//! between the map's principal stretches rather than their magnitude. Where
//! that is not good enough, ask the map what it did and undo it:
//!
//! ```no_run
//! # use threers::{Conform, Lattice, LatticeKind, Tpms, Vector3};
//! # let radius = 20.0;
//! let axis = Vector3::new(0.0, 0.0, 1.0);
//! let geom = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
//!     .conform(Conform::cylindrical(Vector3::ZERO, axis, radius))
//!     // Thicker walls further out, where the cells are stretched wider.
//!     .grade(move |p| {
//!         let r = (p.x * p.x + p.y * p.y).sqrt().max(1e-3);
//!         r / radius
//!     })
//!     .build();
//! ```
//!
//! [`wall_samples`](super::Lattice::wall_samples) measures the world-space
//! thickness against the world-space sample step, and the stretch correction
//! puts the wall back in world units before it is contoured — so the reading is
//! right to within the *spread* of the map's principal stretches, not off by
//! their magnitude. On a cylindrical map that spread is `|1 - r₀/r| / 3`, which
//! is a few percent over a wall and does not need thinking about; on a map that
//! compresses one direction tenfold, it does.

use super::Region;
use crate::math::Vector3;
use std::f32::consts::TAU;
use std::sync::Arc;

/// A map from world space into the space the cells are tiled in.
///
/// [`Lattice::fill`](super::Lattice::fill) *trims*: the cells are laid out on a
/// world-space grid and whatever pokes out of the part is cut off. On a curved
/// shell that cut lands mid-strut, so the surface is a rash of severed stubs
/// that carry no load. A conformal lattice maps the point into cell space
/// first — angle around an axis for x and the tiling closes around a nozzle;
/// depth below a surface for z and every cell through a curved wall is a whole
/// cell.
///
/// [`thickness`](super::Lattice::thickness) is a length in *cell* space, and a
/// map that is not an isometry does not preserve lengths, so every map reports
/// a [`stretch_at`](Self::stretch_at) and the thickness is divided by it. The
/// wall is then exactly right wherever the map is an isometry, and the error
/// elsewhere is the spread between the map's principal stretches rather than
/// their magnitude.
pub struct Conform<'a> {
    map: Arc<dyn Fn(Vector3) -> Vector3 + Sync + Send + 'a>,
    stretch: Box<dyn Fn(Vector3) -> f32 + Sync + Send + 'a>,
}

impl<'a> Conform<'a> {
    /// An arbitrary map, with the stretch measured by differencing it.
    ///
    /// Four evaluations of the map per sample rather than one — use
    /// [`with_stretch`](Self::with_stretch) where the map's Jacobian is known
    /// and the map itself is not cheap.
    pub fn new(map: impl Fn(Vector3) -> Vector3 + Sync + Send + 'a) -> Self {
        let map: Arc<dyn Fn(Vector3) -> Vector3 + Sync + Send + 'a> = Arc::new(map);
        let differenced = Arc::clone(&map);
        Self {
            map,
            stretch: Box::new(move |p| numeric_stretch(&*differenced, p)),
        }
    }

    /// An arbitrary map and its own account of how much it stretches space.
    ///
    /// The stretch is the mean of the map's principal stretches — the factor a
    /// length in world space is multiplied by on its way into cell space. One
    /// means an isometry, which is what makes `thickness` a true length.
    pub fn with_stretch(
        map: impl Fn(Vector3) -> Vector3 + Sync + Send + 'a,
        stretch: impl Fn(Vector3) -> f32 + Sync + Send + 'a,
    ) -> Self {
        Self {
            map: Arc::new(map),
            stretch: Box::new(stretch),
        }
    }

    /// Wrap the lattice around an axis: **x** is arc length at `reference`,
    /// **y** runs along the axis, **z** is the radius.
    ///
    /// The pattern closes on itself only if a whole number of cells fits the
    /// circumference — see [`ring_pitch`](Self::ring_pitch). Anything else
    /// leaves a seam where the last cell meets the first, which is a real
    /// crack in the mesh and not a rendering artefact.
    pub fn cylindrical(origin: Vector3, axis: Vector3, reference: f32) -> Self {
        let axis = axis.normalize();
        let (u, v) = frame(axis);
        let reference = reference.abs().max(1e-6);
        let map = move |p: Vector3| {
            let w = p - origin;
            let along = w.dot(axis);
            let radial = w - axis * along;
            let angle = radial.dot(v).atan2(radial.dot(u));
            Vector3::new(angle * reference, along, radial.length())
        };
        let stretch = move |p: Vector3| {
            let w = p - origin;
            let r = (w - axis * w.dot(axis)).length().max(1e-6);
            // Tangential r0/r, axial 1, radial 1.
            (reference / r + 2.0) / 3.0
        };
        Self::with_stretch(map, stretch)
    }

    /// Wrap the lattice around a point: **x** and **y** are arc lengths along
    /// the two angles at `reference`, **z** is the radius.
    ///
    /// A helmet liner, a hip cup, anything that is a shell around a centre.
    /// The two poles of the map are singular — cells converge to nothing on the
    /// axis through the centre, the way meridians do on a globe — so aim them
    /// somewhere the part is not, or does not matter.
    pub fn spherical(center: Vector3, reference: f32) -> Self {
        let reference = reference.abs().max(1e-6);
        let map = move |p: Vector3| {
            let w = p - center;
            let r = w.length();
            let azimuth = w.y.atan2(w.x);
            let polar = (w.z / r.max(1e-6)).clamp(-1.0, 1.0).acos();
            Vector3::new(azimuth * reference, polar * reference, r)
        };
        let stretch = move |p: Vector3| {
            let w = p - center;
            let r = w.length().max(1e-6);
            // Away from the poles the two angular stretches are r0/(r sin φ)
            // and r0/r; on them the first diverges, and a lattice cannot be
            // built from cells of no width however it is corrected.
            let sin_polar = ((w.x * w.x + w.y * w.y).sqrt() / r).max(0.05);
            (reference / (r * sin_polar) + reference / r + 1.0) / 3.0
        };
        Self::with_stretch(map, stretch)
    }

    /// Follow a surface through the wall: **x** and **y** stay world x and y,
    /// **z** becomes depth below `region`'s surface.
    ///
    /// Zero on the surface and positive inside, so `cell_size.z` is the layer
    /// spacing through the wall and a whole number of cells spans a wall a
    /// whole number of cells thick, however the surface curves. This is the
    /// cheap half of conforming — the cells still tile in plan, so a wall that
    /// turns over past vertical gets the same treatment as an overhang — and it
    /// is the half that matters for a shell whose thickness is what is being
    /// filled.
    ///
    /// A region whose `distance` is not a true distance (a mesh, an
    /// intersection) makes the layers uneven in proportion to how far off it
    /// is.
    pub fn depth(region: Region<'a>) -> Self {
        // The gradient of a distance field is a unit vector, so the map is an
        // isometry wherever the region is honest about its distances.
        Self::with_stretch(
            move |p| Vector3::new(p.x, p.y, region.distance(p)),
            |_| 1.0,
        )
    }

    /// The cell pitch that fits `count` cells exactly around a circle of
    /// `radius` — the x cell size a [`cylindrical`](Self::cylindrical) map
    /// needs to close without a seam.
    pub fn ring_pitch(radius: f32, count: usize) -> f32 {
        TAU * radius.abs() / count.max(1) as f32
    }

    /// Where `p` lands in cell space.
    pub fn point(&self, p: Vector3) -> Vector3 {
        (self.map)(p)
    }

    /// How much the map stretches a length at `p`.
    ///
    /// The mean of the map's principal stretches: the factor a length in world
    /// space is multiplied by on its way into cell space, and the number
    /// [`thickness`](super::Lattice::thickness) is divided by so that a wall
    /// stays a wall. One means an isometry.
    pub fn stretch_at(&self, p: Vector3) -> f32 {
        let s = (self.stretch)(p);
        if s.is_finite() && s > 1e-6 {
            s
        } else {
            1.0
        }
    }
}

/// A pair of unit vectors perpendicular to `axis` and to each other.
fn frame(axis: Vector3) -> (Vector3, Vector3) {
    let seed = if axis.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let u = axis.cross(seed).normalize();
    (u, axis.cross(u))
}

/// The mean of the three column lengths of the map's Jacobian.
///
/// Not the largest and not the smallest: a single number cannot describe an
/// anisotropic stretch, and the mean is the one that splits the error rather
/// than putting it all on one side.
fn numeric_stretch(map: &(dyn Fn(Vector3) -> Vector3 + Sync + Send), p: Vector3) -> f32 {
    // Relative to the point, so the step survives a part far from the origin
    // without being lost in `f32`, and absolute near the origin where a
    // relative step would be zero.
    let h = 1e-3 * (1.0 + p.length());
    let at = map(p);
    let mut total = 0.0;
    for axis in 0..3 {
        let mut step = Vector3::ZERO;
        match axis {
            0 => step.x = h,
            1 => step.y = h,
            _ => step.z = h,
        }
        total += (map(p + step) - at).length() / h;
    }
    total / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cylindrical_is_an_isometry_at_the_reference_radius() {
        let c = Conform::cylindrical(Vector3::ZERO, Vector3::new(0.0, 0.0, 1.0), 10.0);
        assert!((c.stretch_at(Vector3::new(10.0, 0.0, 3.0)) - 1.0).abs() < 1e-5);
        // Half the radius, twice the tangential stretch: (2 + 2) / 3.
        assert!((c.stretch_at(Vector3::new(5.0, 0.0, 0.0)) - 4.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn cylindrical_maps_arc_length_to_x() {
        let c = Conform::cylindrical(Vector3::ZERO, Vector3::new(0.0, 0.0, 1.0), 4.0);
        let a = c.point(Vector3::new(4.0, 0.0, 1.0));
        let b = c.point(Vector3::new(0.0, 4.0, 1.0));
        // A quarter turn at the reference radius is a quarter of the
        // circumference in cell space.
        assert!((b.x - a.x - TAU * 4.0 / 4.0).abs() < 1e-4, "{a:?} {b:?}");
        // Axial and radial pass straight through.
        assert!((a.y - 1.0).abs() < 1e-5 && (a.z - 4.0).abs() < 1e-5);
    }

    #[test]
    fn ring_pitch_closes_the_seam() {
        let radius = 7.0;
        let count = 12;
        let pitch = Conform::ring_pitch(radius, count);
        assert!((pitch * count as f32 - TAU * radius).abs() < 1e-4);
    }

    #[test]
    fn depth_measures_from_the_surface() {
        let c = Conform::depth(Region::sphere(Vector3::ZERO, 5.0));
        // On the surface, at the centre, and outside.
        assert!(c.point(Vector3::new(5.0, 0.0, 0.0)).z.abs() < 1e-4);
        assert!((c.point(Vector3::ZERO).z - 5.0).abs() < 1e-4);
        assert!(c.point(Vector3::new(8.0, 0.0, 0.0)).z < 0.0);
        // And x, y are untouched.
        let p = Vector3::new(1.0, 2.0, 3.0);
        let q = c.point(p);
        assert_eq!((q.x, q.y), (p.x, p.y));
    }

    #[test]
    fn a_differenced_stretch_finds_a_uniform_scale() {
        let c = Conform::new(|p| p * 3.0);
        assert!((c.stretch_at(Vector3::new(1.0, -2.0, 0.5)) - 3.0).abs() < 1e-2);
    }

    #[test]
    fn a_degenerate_stretch_falls_back_to_one() {
        let c = Conform::with_stretch(|p| p, |_| 0.0);
        assert_eq!(c.stretch_at(Vector3::ZERO), 1.0);
        let nan = Conform::with_stretch(|p| p, |_| f32::NAN);
        assert_eq!(nan.stretch_at(Vector3::ZERO), 1.0);
    }
}
