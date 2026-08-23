//! Convex hulls: quickhull construction plus the volume integrals the mass
//! properties need.
//!
//! Hulls are stored triangulated. That is all the narrow phase asks for — GJK
//! only needs the vertex cloud, and EPA only needs a support function — while
//! the face list gives exact volume, centre of mass and inertia.

use crate::math::Aabb;
use threers::math::Vector3;

/// One triangle of a hull, wound counter-clockwise seen from outside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HullFace {
    /// Indices into [`ConvexHull::vertices`].
    pub indices: [u32; 3],
    /// Outward unit normal.
    pub normal: Vector3,
    /// Plane offset: points on the face satisfy `normal · p == offset`.
    pub offset: f32,
}

/// A convex polyhedron.
///
/// Build one with [`ConvexHull::from_points`], which discards interior and
/// duplicate points, or with [`ConvexHull::from_vertices_unchecked`] when the
/// data is already a hull.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvexHull {
    pub vertices: Vec<Vector3>,
    pub faces: Vec<HullFace>,
    aabb: Aabb,
}

impl ConvexHull {
    /// Build the convex hull of a point cloud.
    ///
    /// Returns `None` if the input is degenerate — fewer than four points, or
    /// all points collinear or coplanar — since such a set has no volume and
    /// cannot act as a solid collider. Use a [`crate::shape::Shape::Cuboid`] or a
    /// thin box for flat geometry instead.
    ///
    /// ```
    /// use threers_physics::prelude::*;
    ///
    /// let cube: Vec<Vector3> = [-1.0f32, 1.0]
    ///     .iter()
    ///     .flat_map(|&x| [-1.0f32, 1.0].iter().flat_map(move |&y| {
    ///         [-1.0f32, 1.0].iter().map(move |&z| Vector3::new(x, y, z))
    ///     }))
    ///     .collect();
    /// let hull = ConvexHull::from_points(&cube).unwrap();
    /// assert_eq!(hull.vertices.len(), 8);
    /// assert!((hull.volume() - 8.0).abs() < 1e-3);
    /// ```
    pub fn from_points(points: &[Vector3]) -> Option<Self> {
        let faces = quickhull(points)?;

        // Compact: keep only vertices the hull actually references.
        let mut remap = vec![u32::MAX; points.len()];
        let mut vertices = Vec::new();
        let mut out_faces = Vec::with_capacity(faces.len());
        for f in &faces {
            let mut idx = [0u32; 3];
            for (slot, &old) in idx.iter_mut().zip(f.indices.iter()) {
                let old = old as usize;
                if remap[old] == u32::MAX {
                    remap[old] = vertices.len() as u32;
                    vertices.push(points[old]);
                }
                *slot = remap[old];
            }
            out_faces.push(HullFace {
                indices: idx,
                normal: f.normal,
                offset: f.offset,
            });
        }

        Some(Self {
            aabb: aabb_of(&vertices),
            vertices,
            faces: out_faces,
        })
    }

    /// Wrap vertices that are already known to form a convex set, recomputing
    /// the faces. Still runs quickhull; the name marks that no validation of
    /// *your* claim is performed beyond that.
    pub fn from_vertices_unchecked(vertices: Vec<Vector3>) -> Option<Self> {
        Self::from_points(&vertices)
    }

    pub fn aabb(&self) -> Aabb {
        self.aabb
    }

    /// Farthest vertex along `dir` — the GJK/EPA support function.
    pub fn support(&self, dir: Vector3) -> Vector3 {
        let mut best = self.vertices[0];
        let mut best_dot = best.dot(dir);
        for &v in &self.vertices[1..] {
            let d = v.dot(dir);
            if d > best_dot {
                best_dot = d;
                best = v;
            }
        }
        best
    }

    pub fn volume(&self) -> f32 {
        let mut v = 0.0;
        for f in &self.faces {
            let a = self.vertices[f.indices[0] as usize];
            let b = self.vertices[f.indices[1] as usize];
            let c = self.vertices[f.indices[2] as usize];
            v += a.dot(b.cross(c));
        }
        (v / 6.0).abs()
    }

    pub fn contains_point(&self, p: Vector3) -> bool {
        self.faces
            .iter()
            .all(|f| f.normal.dot(p) - f.offset <= 1e-5)
    }
}

fn aabb_of(points: &[Vector3]) -> Aabb {
    let mut b = Aabb::empty();
    for &p in points {
        b.expand_by_point(p);
    }
    b
}

/// Face under construction — indices into the *original* point slice.
#[derive(Clone, Copy)]
struct RawFace {
    indices: [u32; 3],
    normal: Vector3,
    offset: f32,
}

fn make_face(points: &[Vector3], a: u32, b: u32, c: u32, interior: Vector3) -> Option<RawFace> {
    let (pa, pb, pc) = (
        points[a as usize],
        points[b as usize],
        points[c as usize],
    );
    let n = (pb - pa).cross(pc - pa);
    let normal = crate::math::try_normalize(n)?;
    // Orient outward: the hull interior must be on the negative side.
    let (indices, normal) = if normal.dot(interior - pa) > 0.0 {
        ([a, c, b], -normal)
    } else {
        ([a, b, c], normal)
    };
    Some(RawFace {
        indices,
        normal,
        offset: normal.dot(pa),
    })
}

/// Quickhull. Returns `None` for degenerate (sub-3D) inputs.
///
/// Visible-set and horizon extraction are done by scanning the face list rather
/// than maintaining half-edge adjacency. That is `O(faces)` per inserted point
/// instead of `O(visible)`, which is irrelevant at collider sizes and removes
/// the class of bugs that adjacency bookkeeping invites.
fn quickhull(points: &[Vector3]) -> Option<Vec<RawFace>> {
    if points.len() < 4 {
        return None;
    }

    let scale = {
        let b = aabb_of(points);
        let s = b.size();
        s.x.max(s.y).max(s.z)
    };
    if !scale.is_finite() || scale < 1e-9 {
        return None;
    }
    let eps = scale * 1e-6;

    let [i0, i1, i2, i3] = initial_simplex(points, eps)?;
    let interior = (points[i0 as usize]
        + points[i1 as usize]
        + points[i2 as usize]
        + points[i3 as usize])
        * 0.25;

    let mut faces = Vec::with_capacity(16);
    for (a, b, c) in [
        (i0, i1, i2),
        (i0, i1, i3),
        (i0, i2, i3),
        (i1, i2, i3),
    ] {
        faces.push(make_face(points, a, b, c, interior)?);
    }

    // Points still outside the current hull.
    let mut remaining: Vec<u32> = (0..points.len() as u32)
        .filter(|&i| i != i0 && i != i1 && i != i2 && i != i3)
        .collect();

    // Each iteration adds at least one vertex, so `points.len()` iterations is
    // a hard upper bound; the guard just makes non-termination impossible.
    for _ in 0..points.len() + 4 {
        // Find the point farthest outside any face.
        let mut best: Option<(usize, f32)> = None;
        for (slot, &pi) in remaining.iter().enumerate() {
            let p = points[pi as usize];
            let dist = faces
                .iter()
                .map(|f| f.normal.dot(p) - f.offset)
                .fold(f32::NEG_INFINITY, f32::max);
            if dist > eps && best.is_none_or(|(_, d)| dist > d) {
                best = Some((slot, dist));
            }
        }
        let Some((slot, _)) = best else { break };
        let apex = remaining.swap_remove(slot);
        let p = points[apex as usize];

        // Split faces into visible (to be removed) and kept.
        let mut visible = Vec::new();
        let mut kept = Vec::with_capacity(faces.len());
        for f in faces.drain(..) {
            if f.normal.dot(p) - f.offset > eps {
                visible.push(f);
            } else {
                kept.push(f);
            }
        }
        if visible.is_empty() {
            faces = kept;
            continue;
        }

        // Horizon = directed edges of visible faces whose reverse is not also
        // visible. Those bound the hole left behind.
        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(visible.len() * 3);
        for f in &visible {
            edges.push((f.indices[0], f.indices[1]));
            edges.push((f.indices[1], f.indices[2]));
            edges.push((f.indices[2], f.indices[0]));
        }
        faces = kept;
        for &(a, b) in &edges {
            if edges.iter().any(|&(c, d)| c == b && d == a) {
                continue; // interior edge — shared by two visible faces
            }
            if let Some(f) = make_face(points, a, b, apex, interior) {
                faces.push(f);
            }
        }
        if faces.is_empty() {
            return None;
        }
    }

    if faces.len() < 4 {
        None
    } else {
        Some(faces)
    }
}

/// Pick four points spanning three dimensions, or `None` if impossible.
fn initial_simplex(points: &[Vector3], eps: f32) -> Option<[u32; 4]> {
    // Extremes along each axis give a well-separated starting pair.
    let mut min_i = [0u32; 3];
    let mut max_i = [0u32; 3];
    for (i, p) in points.iter().enumerate() {
        let c = [p.x, p.y, p.z];
        for axis in 0..3 {
            let lo = points[min_i[axis] as usize];
            let hi = points[max_i[axis] as usize];
            if c[axis] < [lo.x, lo.y, lo.z][axis] {
                min_i[axis] = i as u32;
            }
            if c[axis] > [hi.x, hi.y, hi.z][axis] {
                max_i[axis] = i as u32;
            }
        }
    }
    let (mut i0, mut i1, mut best) = (0u32, 0u32, 0.0f32);
    for axis in 0..3 {
        let d = points[min_i[axis] as usize].distance_to(points[max_i[axis] as usize]);
        if d > best {
            best = d;
            i0 = min_i[axis];
            i1 = max_i[axis];
        }
    }
    if best <= eps {
        return None; // every point coincident
    }

    // Farthest point from the line i0-i1.
    let a = points[i0 as usize];
    let dir = (points[i1 as usize] - a).normalize();
    let mut i2 = u32::MAX;
    let mut best = eps;
    for (i, &p) in points.iter().enumerate() {
        let v = p - a;
        let d = (v - dir * v.dot(dir)).length();
        if d > best {
            best = d;
            i2 = i as u32;
        }
    }
    if i2 == u32::MAX {
        return None; // collinear
    }

    // Farthest point from the plane i0-i1-i2.
    let n = (points[i1 as usize] - a).cross(points[i2 as usize] - a);
    let n = crate::math::try_normalize(n)?;
    let mut i3 = u32::MAX;
    let mut best = eps;
    for (i, &p) in points.iter().enumerate() {
        let d = n.dot(p - a).abs();
        if d > best {
            best = d;
            i3 = i as u32;
        }
    }
    if i3 == u32::MAX {
        return None; // coplanar
    }

    Some([i0, i1, i2, i3])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_points(h: f32) -> Vec<Vector3> {
        let mut v = Vec::new();
        for &x in &[-h, h] {
            for &y in &[-h, h] {
                for &z in &[-h, h] {
                    v.push(Vector3::new(x, y, z));
                }
            }
        }
        v
    }

    #[test]
    fn hull_of_a_cube_has_eight_vertices_and_right_volume() {
        let hull = ConvexHull::from_points(&cube_points(1.0)).unwrap();
        assert_eq!(hull.vertices.len(), 8);
        assert_eq!(hull.faces.len(), 12);
        assert!((hull.volume() - 8.0).abs() < 1e-3);
    }

    #[test]
    fn interior_points_are_discarded() {
        let mut pts = cube_points(1.0);
        pts.push(Vector3::ZERO);
        pts.push(Vector3::new(0.2, -0.3, 0.1));
        let hull = ConvexHull::from_points(&pts).unwrap();
        assert_eq!(hull.vertices.len(), 8);
        assert!((hull.volume() - 8.0).abs() < 1e-3);
    }

    #[test]
    fn all_face_normals_point_outward() {
        let hull = ConvexHull::from_points(&cube_points(1.5)).unwrap();
        for f in &hull.faces {
            // The origin is interior, so it must be behind every face plane.
            assert!(f.normal.dot(Vector3::ZERO) - f.offset < -1e-4);
        }
    }

    #[test]
    fn support_returns_the_extreme_vertex() {
        let hull = ConvexHull::from_points(&cube_points(1.0)).unwrap();
        let s = hull.support(Vector3::new(1.0, 1.0, 1.0));
        assert_eq!(s, Vector3::new(1.0, 1.0, 1.0));
        let s = hull.support(Vector3::new(-1.0, 0.1, -1.0));
        assert_eq!(s.x, -1.0);
        assert_eq!(s.z, -1.0);
    }

    #[test]
    fn degenerate_inputs_return_none() {
        assert!(ConvexHull::from_points(&[]).is_none());
        assert!(ConvexHull::from_points(&[Vector3::ZERO; 8]).is_none());
        // Collinear.
        let line: Vec<_> = (0..10).map(|i| Vector3::new(i as f32, 0.0, 0.0)).collect();
        assert!(ConvexHull::from_points(&line).is_none());
        // Coplanar.
        let plane = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.5, 0.0, 0.5),
        ];
        assert!(ConvexHull::from_points(&plane).is_none());
    }

    #[test]
    fn hull_of_a_sphere_cloud_contains_all_inputs() {
        // Fibonacci sphere — a stress test with no axis-aligned structure.
        let n = 200;
        let pts: Vec<Vector3> = (0..n)
            .map(|i| {
                let t = (i as f32 + 0.5) / n as f32;
                let phi = (1.0 - 2.0 * t).acos();
                let theta = std::f32::consts::PI * (1.0 + 5.0f32.sqrt()) * i as f32;
                Vector3::new(
                    phi.sin() * theta.cos(),
                    phi.sin() * theta.sin(),
                    phi.cos(),
                )
            })
            .collect();
        let hull = ConvexHull::from_points(&pts).unwrap();
        for p in &pts {
            assert!(hull.contains_point(*p), "{p:?} fell outside its own hull");
        }
        // Volume should approach 4/3 π ≈ 4.19 from below.
        assert!(hull.volume() > 4.0 && hull.volume() < 4.19, "{}", hull.volume());
    }
}
