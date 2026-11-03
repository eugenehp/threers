//! Triangle-mesh colliders, accelerated by the `threers` mesh BVH.
//!
//! Triangle meshes are **hollow surfaces, not solids**. They collide correctly
//! but have no well-defined inside, so attaching one to a dynamic body lets fast
//! movers tunnel through and lets the solver push objects out the wrong face.
//! Use them for static level geometry; use [`crate::shape::Shape::ConvexHull`] or a
//! [`crate::shape::Shape::Compound`] of primitives for dynamic bodies.

use crate::math::Aabb;
use std::sync::Arc;
use threers::core::{BufferAttribute, BufferGeometry};
use threers::math::{Ray, Vector3};
use threers::mesh_bvh::{BuildOptions, MeshBvh};

/// A flattened BVH node, decoded once from [`MeshBvh::node_buffer`].
///
/// The buffer layout — `min(3), max(3), meta0, meta1` with `meta0 < 0` marking
/// a leaf whose triangle range is `[-meta0 - 1, -meta0 - 1 + meta1)` — is the
/// documented export format the JS `shapecast` bridge also consumes.
#[derive(Debug, Clone, Copy)]
struct Node {
    bounds: Aabb,
    /// Leaf: first order-index. Internal: left child node index.
    left_or_start: u32,
    /// Leaf: triangle count. Internal: right child node index.
    right_or_count: u32,
    is_leaf: bool,
}

/// An indexed triangle soup with a BVH over it.
#[derive(Debug, Clone)]
pub struct TriMesh {
    vertices: Vec<Vector3>,
    indices: Vec<[u32; 3]>,
    bvh: Arc<MeshBvh>,
    nodes: Vec<Node>,
    aabb: Aabb,
}

impl TriMesh {
    /// Build from vertices and triangle indices.
    ///
    /// Returns `None` if there are no triangles or an index is out of range.
    pub fn new(vertices: Vec<Vector3>, indices: Vec<[u32; 3]>) -> Option<Self> {
        if indices.is_empty() || vertices.is_empty() {
            return None;
        }
        let n = vertices.len() as u32;
        if indices.iter().any(|t| t.iter().any(|&i| i >= n)) {
            return None;
        }

        let mut geometry = BufferGeometry::new();
        let mut flat = Vec::with_capacity(vertices.len() * 3);
        for v in &vertices {
            flat.extend_from_slice(&[v.x, v.y, v.z]);
        }
        geometry.set_attribute("position", BufferAttribute::new(flat, 3));
        geometry.set_index(indices.iter().flat_map(|t| t.iter().copied()).collect());

        let bvh = MeshBvh::build(&geometry, BuildOptions::default())?;
        let nodes = decode_nodes(bvh.node_buffer());
        if nodes.is_empty() {
            return None;
        }
        let aabb = bvh.bounding_box();

        Some(Self {
            vertices,
            indices,
            bvh: Arc::new(bvh),
            nodes,
            aabb,
        })
    }

    /// Build from a `threers` geometry — the usual path, since this is whatever
    /// you were already drawing.
    ///
    /// Non-indexed geometry is treated as a flat triangle list.
    pub fn from_geometry(geometry: &BufferGeometry) -> Option<Self> {
        let pos = geometry.get_attribute("position")?;
        if pos.item_size < 3 {
            return None;
        }
        let vertices: Vec<Vector3> = pos
            .array
            .chunks_exact(pos.item_size)
            .map(|c| Vector3::new(c[0], c[1], c[2]))
            .collect();

        let indices: Vec<[u32; 3]> = match &geometry.index {
            Some(idx) => idx.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
            None => (0..vertices.len() as u32 / 3)
                .map(|i| [i * 3, i * 3 + 1, i * 3 + 2])
                .collect(),
        };
        Self::new(vertices, indices)
    }

    pub fn aabb(&self) -> Aabb {
        self.aabb
    }

    pub fn vertices(&self) -> &[Vector3] {
        &self.vertices
    }

    pub fn indices(&self) -> &[[u32; 3]] {
        &self.indices
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// The underlying BVH, for raycasts and any `mesh-bvh` query you already use.
    pub fn bvh(&self) -> &MeshBvh {
        &self.bvh
    }

    /// Vertices of triangle `i`.
    #[inline]
    pub fn triangle(&self, i: usize) -> [Vector3; 3] {
        let t = self.indices[i];
        [
            self.vertices[t[0] as usize],
            self.vertices[t[1] as usize],
            self.vertices[t[2] as usize],
        ]
    }

    /// Unit face normal of triangle `i`, or `None` for degenerate triangles.
    pub fn triangle_normal(&self, i: usize) -> Option<Vector3> {
        let [a, b, c] = self.triangle(i);
        crate::math::try_normalize((b - a).cross(c - a))
    }

    /// Visit every triangle whose AABB overlaps `query`, in BVH order.
    ///
    /// The callback receives the geometry triangle index and its three vertices.
    pub fn for_each_triangle_in_aabb(
        &self,
        query: &Aabb,
        mut visit: impl FnMut(usize, [Vector3; 3]),
    ) {
        if !self.aabb.intersects_box(query) {
            return;
        }
        // Explicit stack: recursion depth on a deep BVH is a real overflow risk
        // for large meshes, and this is on the per-step hot path.
        let mut stack = vec![0u32];
        while let Some(ni) = stack.pop() {
            let node = self.nodes[ni as usize];
            if !node.bounds.intersects_box(query) {
                continue;
            }
            if node.is_leaf {
                let start = node.left_or_start as usize;
                for order in start..start + node.right_or_count as usize {
                    let Some(tri) = self.bvh.resolve_triangle_index(order) else {
                        continue;
                    };
                    let verts = self.triangle(tri);
                    let mut tb = Aabb::empty();
                    for v in verts {
                        tb.expand_by_point(v);
                    }
                    if tb.intersects_box(query) {
                        visit(tri, verts);
                    }
                }
            } else {
                stack.push(node.left_or_start);
                stack.push(node.right_or_count);
            }
        }
    }

    /// Nearest ray hit as `(toi, triangle index)`.
    pub fn raycast(&self, ray: &Ray, max_toi: f32, backface_culling: bool) -> Option<(f32, usize)> {
        self.bvh
            .raycast_first(ray, 0.0, max_toi, backface_culling)
            .map(|hit| (hit.distance, hit.face_index))
    }

    /// Signed volume via the divergence theorem. Meaningful only for closed,
    /// outward-wound meshes; open surfaces give an arbitrary value.
    pub fn signed_volume(&self) -> f32 {
        let mut v = 0.0;
        for i in 0..self.indices.len() {
            let [a, b, c] = self.triangle(i);
            v += a.dot(b.cross(c));
        }
        v / 6.0
    }
}

fn decode_nodes(buffer: &[f32]) -> Vec<Node> {
    buffer
        .chunks_exact(8)
        .map(|c| {
            let bounds = Aabb::new(
                Vector3::new(c[0], c[1], c[2]),
                Vector3::new(c[3], c[4], c[5]),
            );
            let (meta0, meta1) = (c[6], c[7]);
            if meta0 < 0.0 {
                Node {
                    bounds,
                    left_or_start: (-meta0 - 1.0) as u32,
                    right_or_count: meta1 as u32,
                    is_leaf: true,
                }
            } else {
                Node {
                    bounds,
                    left_or_start: meta0 as u32,
                    right_or_count: meta1 as u32,
                    is_leaf: false,
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Axis-aligned box as a closed, outward-wound triangle mesh.
    fn box_mesh(h: f32) -> TriMesh {
        let v: Vec<Vector3> = [
            (-1., -1., -1.),
            (1., -1., -1.),
            (1., 1., -1.),
            (-1., 1., -1.),
            (-1., -1., 1.),
            (1., -1., 1.),
            (1., 1., 1.),
            (-1., 1., 1.),
        ]
        .iter()
        .map(|&(x, y, z)| Vector3::new(x * h, y * h, z * h))
        .collect();
        let idx = vec![
            [0, 2, 1],
            [0, 3, 2], // -z
            [4, 5, 6],
            [4, 6, 7], // +z
            [0, 1, 5],
            [0, 5, 4], // -y
            [3, 7, 6],
            [3, 6, 2], // +y
            [0, 4, 7],
            [0, 7, 3], // -x
            [1, 2, 6],
            [1, 6, 5], // +x
        ];
        TriMesh::new(v, idx).unwrap()
    }

    #[test]
    fn builds_and_reports_bounds() {
        let m = box_mesh(2.0);
        assert_eq!(m.triangle_count(), 12);
        assert!((m.aabb().min.x + 2.0).abs() < 1e-5);
        assert!((m.aabb().max.z - 2.0).abs() < 1e-5);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(TriMesh::new(vec![], vec![]).is_none());
        assert!(TriMesh::new(vec![Vector3::ZERO; 3], vec![[0, 1, 9]]).is_none());
    }

    #[test]
    fn aabb_query_finds_only_overlapping_triangles() {
        let m = box_mesh(1.0);
        // A box hugging the +x face should find the two triangles there and no
        // triangle from the opposite face.
        let q = Aabb::new(
            Vector3::new(0.9, -0.5, -0.5),
            Vector3::new(1.5, 0.5, 0.5),
        );
        let mut hits = Vec::new();
        m.for_each_triangle_in_aabb(&q, |i, _| hits.push(i));
        assert!(!hits.is_empty());
        for i in hits {
            let [a, b, c] = m.triangle(i);
            assert!(a.x > 0.9 || b.x > 0.9 || c.x > 0.9, "tri {i} is not on +x");
        }
    }

    #[test]
    fn aabb_query_outside_the_mesh_finds_nothing() {
        let m = box_mesh(1.0);
        let q = Aabb::new(Vector3::new(50.0, 50.0, 50.0), Vector3::new(51.0, 51.0, 51.0));
        let mut count = 0;
        m.for_each_triangle_in_aabb(&q, |_, _| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn aabb_query_covering_everything_finds_every_triangle() {
        let m = box_mesh(1.0);
        let q = Aabb::new(Vector3::new(-9.0, -9.0, -9.0), Vector3::new(9.0, 9.0, 9.0));
        let mut hits = std::collections::HashSet::new();
        m.for_each_triangle_in_aabb(&q, |i, _| {
            hits.insert(i);
        });
        assert_eq!(hits.len(), 12);
    }

    #[test]
    fn raycast_hits_the_near_face() {
        let m = box_mesh(1.0);
        let ray = Ray::new(Vector3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        let (toi, _) = m.raycast(&ray, 100.0, false).unwrap();
        assert!((toi - 4.0).abs() < 1e-3, "toi = {toi}");
    }

    #[test]
    fn signed_volume_of_a_closed_box_is_positive_and_exact() {
        let m = box_mesh(1.5);
        assert!((m.signed_volume() - 27.0).abs() < 1e-3, "{}", m.signed_volume());
    }
}
