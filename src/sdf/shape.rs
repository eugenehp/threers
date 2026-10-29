//! [`Shape`]: a field built as data — primitives, rigid moves and the
//! operators as a tree — rather than as a closure.
//!
//! A closure is the quickest way to write a field in Rust, but it cannot be
//! built at run time, crossed into JavaScript, or asked where it ends. A
//! `Shape` can: it is assembled from calls, evaluates in Rust however it was
//! built (which is what makes it worth having in a browser, where a field
//! written in JavaScript costs a call across into wasm for every sample of a
//! mesh), and carries conservative [`bounds`](Shape::bounds), so it can be
//! meshed without guessing a box.
//!
//! ```
//! use threers::math::Vector3;
//! use threers::sdf::{Sdf3, Shape};
//!
//! // A dome with a hollow underneath, its rim rounded.
//! let v = Vector3::new;
//! let dome = Shape::ellipsoid(v(0.6, 0.45, 0.6))
//!     .intersect(Shape::plane(v(0.0, -1.0, 0.0), 0.0), 0.0)
//!     .subtract(Shape::ellipsoid(v(0.45, 0.25, 0.45)), 0.04);
//! assert!(dome.distance(v(0.0, 0.35, 0.0)) < 0.0);  // in the crown
//! assert!(dome.distance(v(0.0, 0.1, 0.0)) > 0.0);   // in the hollow
//! let (lo, hi) = dome.bounds().expect("bounded");
//! let mesh = dome.to_surface_nets(0.02, lo, hi);
//! assert!(mesh.triangle_count() > 0);
//! ```

use super::{d3, smooth_union, Sdf3};
use crate::math::{Quaternion, Vector3};

/// A signed distance field as a tree. Primitives sit at the origin (a capsule
/// at its endpoints); [`translate`](Self::translate),
/// [`rotate`](Self::rotate) and [`scale`](Self::scale) move them, and the
/// operators join them. `k` is a blend width everywhere: `0` is the hard
/// operator, more fillets the seam (see [`smooth_union`](super::smooth_union)).
#[derive(Debug, Clone)]
pub enum Shape {
    Sphere { r: f32 },
    Ellipsoid { radii: Vector3 },
    /// [`d3::round_cuboid`]; `round` 0 is a sharp box.
    Cuboid { half: Vector3, round: f32 },
    /// [`d3::tapered_capsule`]; equal radii are a plain capsule.
    Capsule { a: Vector3, b: Vector3, ra: f32, rb: f32 },
    /// In the xz plane.
    Torus { major: f32, minor: f32 },
    /// Along y.
    Cylinder { half_height: f32, r: f32 },
    /// The half-space below `dot(p, normal) + offset = 0`.
    Plane { normal: Vector3, offset: f32 },
    Union { parts: Vec<Shape>, k: f32 },
    Intersection { parts: Vec<Shape>, k: f32 },
    Subtraction { base: Box<Shape>, cut: Box<Shape>, k: f32 },
    Round { shape: Box<Shape>, r: f32 },
    Onion { shape: Box<Shape>, t: f32 },
    /// Moved by `offset`, turned by `rotation`, then scaled by `scale` about
    /// the origin (uniform, so it stays a distance).
    Transform { shape: Box<Shape>, offset: Vector3, rotation: Quaternion, scale: f32 },
}

impl Shape {
    pub fn sphere(r: f32) -> Self {
        Shape::Sphere { r }
    }
    pub fn ellipsoid(radii: Vector3) -> Self {
        Shape::Ellipsoid { radii }
    }
    pub fn cuboid(half: Vector3) -> Self {
        Shape::Cuboid { half, round: 0.0 }
    }
    pub fn round_cuboid(half: Vector3, round: f32) -> Self {
        Shape::Cuboid { half, round }
    }
    pub fn capsule(a: Vector3, b: Vector3, r: f32) -> Self {
        Shape::Capsule { a, b, ra: r, rb: r }
    }
    pub fn tapered_capsule(a: Vector3, b: Vector3, ra: f32, rb: f32) -> Self {
        Shape::Capsule { a, b, ra, rb }
    }
    pub fn torus(major: f32, minor: f32) -> Self {
        Shape::Torus { major, minor }
    }
    pub fn cylinder(half_height: f32, r: f32) -> Self {
        Shape::Cylinder { half_height, r }
    }
    /// `normal` is normalised here.
    pub fn plane(normal: Vector3, offset: f32) -> Self {
        Shape::Plane { normal: normal.normalize(), offset }
    }

    /// Every part, joined; the seams filleted over `k`.
    pub fn union_all(parts: Vec<Shape>, k: f32) -> Self {
        Shape::Union { parts, k }
    }
    pub fn union(self, other: Shape, k: f32) -> Self {
        match self {
            // Keep a chain of unions flat: one node, evaluated in a loop.
            Shape::Union { mut parts, k: k0 } if k0 == k => {
                parts.push(other);
                Shape::Union { parts, k }
            }
            s => Shape::Union { parts: vec![s, other], k },
        }
    }
    pub fn intersect(self, other: Shape, k: f32) -> Self {
        match self {
            Shape::Intersection { mut parts, k: k0 } if k0 == k => {
                parts.push(other);
                Shape::Intersection { parts, k }
            }
            s => Shape::Intersection { parts: vec![s, other], k },
        }
    }
    pub fn subtract(self, cut: Shape, k: f32) -> Self {
        Shape::Subtraction { base: Box::new(self), cut: Box::new(cut), k }
    }
    pub fn round(self, r: f32) -> Self {
        Shape::Round { shape: Box::new(self), r }
    }
    pub fn onion(self, t: f32) -> Self {
        Shape::Onion { shape: Box::new(self), t }
    }
    pub fn translate(self, by: Vector3) -> Self {
        self.transformed(by, Quaternion::identity(), 1.0)
    }
    /// Turned about the origin.
    pub fn rotate(self, rotation: Quaternion) -> Self {
        self.transformed(Vector3::ZERO, rotation, 1.0)
    }
    /// Scaled about the origin; `s` must be positive.
    pub fn scale(self, s: f32) -> Self {
        self.transformed(Vector3::ZERO, Quaternion::identity(), s)
    }
    fn transformed(self, offset: Vector3, rotation: Quaternion, scale: f32) -> Self {
        match self {
            // Fold successive moves into one: p ↦ s·R·p + o, applied after
            // the inner one, is s·R·(s0·R0·p + o0) + o.
            Shape::Transform { shape, offset: o0, rotation: r0, scale: s0 } => Shape::Transform {
                shape,
                offset: o0.apply_quaternion(rotation) * scale + offset,
                rotation: rotation.multiply(r0),
                scale: s0 * scale,
            },
            s => Shape::Transform { shape: Box::new(s), offset, rotation, scale },
        }
    }

    /// An axis-aligned box the surface lies inside, as `(min, max)`; `None`
    /// for a field with no end (a plane, and anything not cut down from one).
    /// Conservative: blends and rotations only ever grow it.
    pub fn bounds(&self) -> Option<(Vector3, Vector3)> {
        let cube = |h: Vector3| Some((h * -1.0, h));
        match self {
            Shape::Sphere { r } => cube(Vector3::new(*r, *r, *r)),
            Shape::Ellipsoid { radii } => cube(*radii),
            Shape::Cuboid { half, .. } => cube(*half),
            Shape::Capsule { a, b, ra, rb } => {
                let r = ra.max(*rb);
                let g = Vector3::new(r, r, r);
                Some((a.min(*b) - g, a.max(*b) + g))
            }
            Shape::Torus { major, minor } => cube(Vector3::new(major + minor, *minor, major + minor)),
            Shape::Cylinder { half_height, r } => cube(Vector3::new(*r, *half_height, *r)),
            Shape::Plane { .. } => None,
            Shape::Union { parts, k } => {
                let mut acc: Option<(Vector3, Vector3)> = None;
                for p in parts {
                    let (lo, hi) = p.bounds()?;
                    acc = Some(match acc {
                        None => (lo, hi),
                        Some((a, b)) => (a.min(lo), b.max(hi)),
                    });
                }
                // A smooth union can swell past the hard one by k/4.
                acc.map(|(lo, hi)| grow(lo, hi, k.max(0.0) * 0.25))
            }
            Shape::Intersection { parts, .. } => {
                let mut acc: Option<(Vector3, Vector3)> = None;
                for p in parts {
                    if let Some((lo, hi)) = p.bounds() {
                        acc = Some(match acc {
                            None => (lo, hi),
                            Some((a, b)) => (a.max(lo), b.min(hi)),
                        });
                    }
                }
                acc
            }
            Shape::Subtraction { base, .. } => base.bounds(),
            Shape::Round { shape, r } => shape.bounds().map(|(lo, hi)| grow(lo, hi, r.max(0.0))),
            Shape::Onion { shape, t } => shape.bounds().map(|(lo, hi)| grow(lo, hi, t.abs())),
            Shape::Transform { shape, offset, rotation, scale } => {
                let (lo, hi) = shape.bounds()?;
                let (mut a, mut b) = (Vector3::new(f32::MAX, f32::MAX, f32::MAX), Vector3::new(f32::MIN, f32::MIN, f32::MIN));
                for c in 0..8 {
                    let corner = Vector3::new(
                        if c & 1 == 0 { lo.x } else { hi.x },
                        if c & 2 == 0 { lo.y } else { hi.y },
                        if c & 4 == 0 { lo.z } else { hi.z },
                    );
                    let p = corner.apply_quaternion(*rotation) * *scale + *offset;
                    a = a.min(p);
                    b = b.max(p);
                }
                Some((a, b))
            }
        }
    }
}

fn grow(lo: Vector3, hi: Vector3, by: f32) -> (Vector3, Vector3) {
    let g = Vector3::new(by, by, by);
    (lo - g, hi + g)
}

impl Sdf3 for Shape {
    fn distance(&self, p: Vector3) -> f32 {
        match self {
            Shape::Sphere { r } => d3::sphere(p, *r),
            Shape::Ellipsoid { radii } => d3::ellipsoid(p, *radii),
            Shape::Cuboid { half, round } => d3::round_cuboid(p, *half, *round),
            Shape::Capsule { a, b, ra, rb } => {
                if ra == rb {
                    d3::capsule(p, *a, *b, *ra)
                } else {
                    d3::tapered_capsule(p, *a, *b, *ra, *rb)
                }
            }
            Shape::Torus { major, minor } => d3::torus(p, *major, *minor),
            Shape::Cylinder { half_height, r } => d3::cylinder(p, *half_height, *r),
            Shape::Plane { normal, offset } => d3::plane(p, *normal, *offset),
            Shape::Union { parts, k } => {
                let mut it = parts.iter();
                let Some(first) = it.next() else { return f32::MAX };
                it.fold(first.distance(p), |d, s| smooth_union(d, s.distance(p), *k))
            }
            Shape::Intersection { parts, k } => {
                let mut it = parts.iter();
                let Some(first) = it.next() else { return f32::MAX };
                it.fold(first.distance(p), |d, s| -smooth_union(-d, -s.distance(p), *k))
            }
            Shape::Subtraction { base, cut, k } => -smooth_union(-base.distance(p), cut.distance(p), *k),
            Shape::Round { shape, r } => shape.distance(p) - r,
            Shape::Onion { shape, t } => shape.distance(p).abs() - t,
            Shape::Transform { shape, offset, rotation, scale } => {
                // Back into the shape's own frame, and the distance out of it.
                let local = ((p - *offset) * (1.0 / scale)).apply_quaternion(rotation.conjugate());
                shape.distance(local) * scale
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf;

    fn v(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3::new(x, y, z)
    }

    #[test]
    fn a_tree_is_its_closure() {
        // The same field written both ways, compared on a grid.
        let shape = Shape::sphere(0.5)
            .translate(v(0.0, 0.6, 0.0))
            .union(Shape::capsule(v(-1.0, 0.0, 0.0), v(1.0, 0.0, 0.0), 0.2), 0.3)
            .subtract(Shape::cuboid(v(0.2, 0.2, 0.2)).translate(v(0.9, 0.0, 0.0)), 0.05)
            .round(0.01);
        let closure = |p: Vector3| {
            let ball = d3::sphere(p - v(0.0, 0.6, 0.0), 0.5);
            let bar = d3::capsule(p, v(-1.0, 0.0, 0.0), v(1.0, 0.0, 0.0), 0.2);
            let cut = d3::cuboid(p - v(0.9, 0.0, 0.0), v(0.2, 0.2, 0.2));
            sdf::smooth_subtraction(sdf::smooth_union(ball, bar, 0.3), cut, 0.05) - 0.01
        };
        for i in 0..9 {
            for j in 0..9 {
                for k in 0..9 {
                    let p = v(i as f32 * 0.3 - 1.2, j as f32 * 0.25 - 0.8, k as f32 * 0.2 - 0.8);
                    assert!((shape.distance(p) - closure(p)).abs() < 1e-6, "at {p:?}");
                }
            }
        }
    }

    #[test]
    fn moves_compose_and_keep_distances() {
        let q = Quaternion::from_axis_angle(v(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);
        // A long box along x, turned to lie along y, doubled, then moved.
        let s = Shape::cuboid(v(1.0, 0.1, 0.1)).rotate(q).scale(2.0).translate(v(3.0, 0.0, 0.0));
        assert!((s.distance(v(3.0, 2.0, 0.0))).abs() < 1e-5, "the end of the turned, doubled box");
        assert!((s.distance(v(3.0, 0.0, 1.2)) - 1.0).abs() < 1e-5, "a distance, not a scaled one");
        let (lo, hi) = s.bounds().unwrap();
        assert!(lo.y < -1.99 && hi.y > 1.99 && lo.x > 2.7 && hi.x < 3.3, "{lo:?} {hi:?}");
    }

    #[test]
    fn bounds_hold_the_surface() {
        let s = Shape::torus(0.5, 0.1)
            .translate(v(0.2, 0.0, 0.0))
            .union(Shape::ellipsoid(v(0.3, 0.6, 0.2)), 0.2)
            .intersect(Shape::plane(v(0.0, -1.0, 0.0), -0.1), 0.0);
        let (lo, hi) = s.bounds().unwrap();
        assert!(lo.y >= -0.6 - 0.06 && hi.y <= 0.6 + 0.06);
        // Nothing just outside the box is inside the shape.
        for i in 0..20 {
            let t = i as f32 / 19.0;
            for p in [v(lo.x - 0.01, 0.2, t - 0.5), v(hi.x + 0.01, 0.3, t - 0.5), v(t - 0.5, hi.y + 0.01, 0.0)] {
                assert!(s.distance(p) > 0.0, "{p:?} outside the bounds is inside the shape");
            }
        }
        assert!(Shape::plane(v(0.0, 1.0, 0.0), 0.0).bounds().is_none());
    }

    #[test]
    fn surface_nets_are_closed_round_and_wound_outward() {
        let s = Shape::sphere(0.5);
        let (lo, hi) = s.bounds().unwrap();
        let m = s.to_surface_nets(0.05, lo - v(0.05, 0.05, 0.05), hi + v(0.05, 0.05, 0.05));
        assert!(m.triangle_count() > 400);
        // On the surface, not on the grid.
        for p in &m.position {
            assert!((p.length() - 0.5).abs() < 1e-3, "{p:?} is off the sphere");
        }
        // Closed: every edge in exactly two triangles, once each way.
        let mut edges = std::collections::HashMap::new();
        for t in m.index.chunks_exact(3) {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edges.entry((a, b)).or_insert(0) += 1;
            }
        }
        for (&(a, b), &n) in &edges {
            assert_eq!(n, 1, "edge {a}-{b} used {n} times the same way");
            assert_eq!(edges.get(&(b, a)), Some(&1), "edge {a}-{b} has no twin");
        }
        // Outward: the signed volume is the ball's.
        let vol: f32 = m
            .index
            .chunks_exact(3)
            .map(|t| m.position[t[0] as usize].dot(m.position[t[1] as usize].cross(m.position[t[2] as usize])) / 6.0)
            .sum();
        let ball = 4.0 / 3.0 * std::f32::consts::PI * 0.125;
        assert!((vol - ball).abs() < ball * 0.03, "volume {vol} against {ball}");
    }
}
