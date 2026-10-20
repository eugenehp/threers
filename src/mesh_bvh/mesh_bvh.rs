use crate::core::BufferGeometry;
use crate::math::{Box3, Ray, Sphere, Triangle, Vector3};

use super::build::{build_bvh, BuildOptions};
use super::hit::BvhHit;
use super::node::{bounds_from_triangles, BvhNode};
use super::serialize::SerializedMeshBvh;
/// One node of a [`MeshBvh`], from [`MeshBvh::node`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BvhNodeView {
    /// Two children, by node index.
    Internal {
        bounds: Box3,
        left: usize,
        right: usize,
    },
    /// `count` triangles from position `offset` in the BVH's triangle order.
    Leaf {
        bounds: Box3,
        offset: usize,
        count: usize,
    },
}

impl BvhNodeView {
    /// The node's bounding box.
    pub fn bounds(&self) -> Box3 {
        match *self {
            BvhNodeView::Internal { bounds, .. } | BvhNodeView::Leaf { bounds, .. } => bounds,
        }
    }
}

/// Bounding Volume Hierarchy for accelerated raycasting on a single geometry.
#[derive(Debug, Clone)]
pub struct MeshBvh {
    nodes: Vec<BvhNode>,
    triangle_order: Vec<usize>,
    positions: Vec<f32>,
    triangle_indices: Vec<(u32, u32, u32)>,
    root_bounds: Box3,
    node_buffer: Vec<f32>,
    indirect: bool,
}

impl MeshBvh {
    pub fn build(geometry: &BufferGeometry, options: BuildOptions) -> Option<Self> {
        let result = build_bvh(geometry, options)?;
        let mut node_buffer = vec![0.0f32; result.nodes.len() * 8];
        for (i, node) in result.nodes.iter().enumerate() {
            node.to_buffer(&mut node_buffer, i);
        }
        Some(Self {
            nodes: result.nodes,
            triangle_order: result.triangle_order,
            positions: result.positions,
            triangle_indices: result.triangle_indices,
            root_bounds: result.root_bounds,
            node_buffer,
            indirect: options.indirect,
        })
    }

    pub fn serialize(&self) -> SerializedMeshBvh {
        SerializedMeshBvh::from_bvh(self)
    }

    pub fn deserialize(data: SerializedMeshBvh) -> Option<Self> {
        if data.version != super::serialize::SERIALIZE_VERSION {
            return None;
        }
        let tri_count = data.triangle_indices.len() / 3;
        if tri_count == 0 || data.triangle_order.len() != tri_count {
            return None;
        }
        Self::from_parts(
            data.node_buffer,
            data.triangle_order.iter().map(|&i| i as usize).collect(),
            data.positions,
            data.triangle_indices
                .chunks_exact(3)
                .map(|c| (c[0], c[1], c[2]))
                .collect(),
        )
    }

    pub(crate) fn from_parts(
        node_buffer: Vec<f32>,
        triangle_order: Vec<usize>,
        positions: Vec<f32>,
        triangle_indices: Vec<(u32, u32, u32)>,
    ) -> Option<Self> {
        if !node_buffer.len().is_multiple_of(8) || triangle_indices.is_empty() {
            return None;
        }
        let node_count = node_buffer.len() / 8;
        let mut nodes = Vec::with_capacity(node_count);
        for i in 0..node_count {
            let base = i * 8;
            let min = Vector3::new(
                node_buffer[base],
                node_buffer[base + 1],
                node_buffer[base + 2],
            );
            let max = Vector3::new(
                node_buffer[base + 3],
                node_buffer[base + 4],
                node_buffer[base + 5],
            );
            let meta0 = node_buffer[base + 6];
            let meta1 = node_buffer[base + 7];
            let is_leaf = meta0 < 0.0;
            nodes.push(if is_leaf {
                BvhNode::leaf(Box3::new(min, max), (-meta0 - 1.0) as u32, meta1 as u32)
            } else {
                BvhNode::internal(Box3::new(min, max), meta0 as u32, meta1 as u32)
            });
        }
        let root_bounds = nodes.first().map(|n| n.bounds).unwrap_or_else(Box3::empty);
        Some(Self {
            nodes,
            triangle_order,
            positions,
            triangle_indices,
            root_bounds,
            node_buffer,
            indirect: true,
        })
    }

    pub fn is_indirect(&self) -> bool {
        self.indirect
    }

    /// Map a BVH-layout triangle index to the original geometry triangle index.
    pub fn resolve_triangle_index(&self, bvh_triangle_index: usize) -> Option<usize> {
        self.triangle_order.get(bvh_triangle_index).copied()
    }

    /// Recompute node bounds after position buffer changes (tree topology unchanged).
    pub fn refit(&mut self, positions: &[f32]) {
        if positions.len() != self.positions.len() {
            return;
        }
        self.positions.copy_from_slice(positions);
        for node in &mut self.nodes {
            if node.is_leaf {
                let start = node.left_or_offset as usize;
                let count = node.right_or_count as usize;
                node.bounds = bounds_from_triangles(
                    &self.positions,
                    &self.triangle_indices,
                    &self.triangle_order,
                    start,
                    count,
                );
            }
        }
        for i in (0..self.nodes.len()).rev() {
            if !self.nodes[i].is_leaf {
                let left = self.nodes[i].left_or_offset as usize;
                let right = self.nodes[i].right_or_count as usize;
                self.nodes[i].bounds = self.nodes[left].bounds.union(&self.nodes[right].bounds);
            }
            self.nodes[i].to_buffer(&mut self.node_buffer, i);
        }
        if !self.nodes.is_empty() {
            self.root_bounds = self.nodes[0].bounds;
        }
    }

    pub fn intersects_box(&self, box_query: &Box3) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        self.intersects_box_node(0, box_query)
    }

    pub fn intersects_sphere(&self, sphere: &Sphere) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        self.intersects_sphere_node(0, sphere)
    }

    pub fn closest_point_to_point(&self, point: Vector3) -> (Vector3, f32, usize) {
        let mut best_dist = f32::INFINITY;
        let mut best_point = point;
        let mut best_tri = 0usize;
        if !self.nodes.is_empty() {
            self.closest_point_node(0, point, &mut best_point, &mut best_dist, &mut best_tri);
        }
        (best_point, best_dist, best_tri)
    }

    pub fn bounding_box(&self) -> Box3 {
        self.root_bounds
    }

    pub fn node_buffer(&self) -> &[f32] {
        &self.node_buffer
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.triangle_indices.len()
    }

    pub fn positions(&self) -> &[f32] {
        &self.positions
    }

    pub fn triangle_indices(&self) -> &[(u32, u32, u32)] {
        &self.triangle_indices
    }

    pub fn triangle_order(&self) -> &[usize] {
        &self.triangle_order
    }

    pub fn raycast(&self, ray: &Ray, near: f32, far: f32, backface_culling: bool) -> Vec<BvhHit> {
        let mut hits = Vec::new();
        if self.nodes.is_empty() {
            return hits;
        }
        self.raycast_node(0, ray, near, far, backface_culling, &mut hits);
        hits.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits
    }

    pub fn raycast_first(
        &self,
        ray: &Ray,
        near: f32,
        far: f32,
        backface_culling: bool,
    ) -> Option<BvhHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv = Vector3::new(
            1.0 / ray.direction.x,
            1.0 / ray.direction.y,
            1.0 / ray.direction.z,
        );
        let mut best: Option<BvhHit> = None;
        self.raycast_node_first(0, ray, inv, near, far, backface_culling, &mut best);
        best
    }

    /// Raycast side constants (mirror three.js Material.side).
    pub fn raycast_first_with_side(
        &self,
        ray: &Ray,
        near: f32,
        far: f32,
        side: u32,
    ) -> Option<BvhHit> {
        let cull = match side {
            super::FRONT_SIDE => true,
            super::BACK_SIDE => false,
            _ => false,
        };
        let mut hits = Vec::new();
        self.raycast_node(0, ray, near, far, cull, &mut hits);
        if side == super::BACK_SIDE {
            hits.retain(|h| {
                let (ia, ib, ic) = self.triangle_indices[h.face_index];
                let tri = Triangle::new(
                    read_vec(&self.positions, ia),
                    read_vec(&self.positions, ib),
                    read_vec(&self.positions, ic),
                );
                !tri.is_front_facing(ray.direction)
            });
        }
        hits.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.into_iter().next()
    }

    pub fn bvhcast(
        &self,
        other: &MeshBvh,
        matrix_to_local: &crate::math::Matrix4,
    ) -> Vec<(usize, usize)> {
        super::bvhcast::bvhcast(self, other, matrix_to_local)
    }

    pub(crate) fn node_bounds(&self, index: u32) -> Box3 {
        self.nodes[index as usize].bounds
    }

    pub(crate) fn is_leaf(&self, index: u32) -> bool {
        self.nodes[index as usize].is_leaf
    }

    pub(crate) fn child_left(&self, index: u32) -> u32 {
        self.nodes[index as usize].left_or_offset
    }

    pub(crate) fn child_right(&self, index: u32) -> u32 {
        self.nodes[index as usize].right_or_count
    }

    pub(crate) fn leaf_range(&self, index: u32) -> (usize, usize) {
        let n = &self.nodes[index as usize];
        (n.left_or_offset as usize, n.right_or_count as usize)
    }

    /// Node `index` (the root is 0, `node_count()` in all), for walking the
    /// tree yourself — a query three-mesh-bvh's `shapecast` does not cover,
    /// or a visualisation of the hierarchy. A leaf's triangles are
    /// [`triangle_at_order_index`](Self::triangle_at_order_index)`(offset..offset
    /// + count)`. `None` past the end.
    pub fn node(&self, index: usize) -> Option<BvhNodeView> {
        let n = self.nodes.get(index)?;
        Some(if n.is_leaf {
            BvhNodeView::Leaf {
                bounds: n.bounds,
                offset: n.left_or_offset as usize,
                count: n.right_or_count as usize,
            }
        } else {
            BvhNodeView::Internal {
                bounds: n.bounds,
                left: n.left_or_offset as usize,
                right: n.right_or_count as usize,
            }
        })
    }

    pub fn triangle_at_order_index(&self, order_index: usize) -> Triangle {
        let tri_idx = self.triangle_order[order_index];
        self.triangle_at_geometry_index(tri_idx)
    }

    /// Triangle by geometry triangle index (matches `BvhHit.face_index` / JS `faceIndex`).
    /// three-mesh-bvh 0.7.6's `raycastFirst`, in f64 as JavaScript runs it:
    /// the closest hit along `origin + t·direction` with `distance` in
    /// `[near, far]`, on `side` (0 front, 1 back, 2 double). The traversal
    /// tries the child the ray meets first along each node's split axis and
    /// skips the other when the first hit lies outside it, so on a tie it
    /// returns the triangle three-mesh-bvh returns — which is what decides
    /// three-bvh-csg's inside/outside test on a shared edge.
    pub fn raycast_first_js(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        side: u32,
        near: f64,
        far: f64,
    ) -> Option<JsRayHit> {
        if self.nodes.is_empty() {
            return None;
        }
        self.raycast_first_js_node(0, origin, direction, side, near, far)
    }

    fn raycast_first_js_node(
        &self,
        node_index: u32,
        origin: [f64; 3],
        dir: [f64; 3],
        side: u32,
        near: f64,
        far: f64,
    ) -> Option<JsRayHit> {
        let node = &self.nodes[node_index as usize];
        if node.is_leaf {
            // `intersectClosestTri`: the nearest, the first of equals.
            let (offset, count) = (node.left_or_offset as usize, node.right_or_count as usize);
            let mut best: Option<JsRayHit> = None;
            for &tri in &self.triangle_order[offset..offset + count] {
                if let Some(hit) = self.intersect_triangle_js(tri, origin, dir, side, near, far) {
                    if best.is_none_or(|b| hit.distance < b.distance) {
                        best = Some(hit);
                    }
                }
            }
            return best;
        }
        let axis = node.split_axis as usize;
        let left_to_right = dir[axis] >= 0.0;
        let (c1, c2) = if left_to_right {
            (node.left_or_offset, node.right_or_count)
        } else {
            (node.right_or_count, node.left_or_offset)
        };
        let r1 = if self.ray_meets_node_js(c1, origin, dir, near, far) {
            self.raycast_first_js_node(c1, origin, dir, side, near, far)
        } else {
            None
        };
        if let Some(hit) = r1 {
            // A hit short of the other child's bounds cannot be beaten there.
            let b = &self.nodes[c2 as usize].bounds;
            let outside = if left_to_right {
                hit.point[axis] <= axis_of(b.min, axis) as f64
            } else {
                hit.point[axis] >= axis_of(b.max, axis) as f64
            };
            if outside {
                return Some(hit);
            }
        }
        let r2 = if self.ray_meets_node_js(c2, origin, dir, near, far) {
            self.raycast_first_js_node(c2, origin, dir, side, near, far)
        } else {
            None
        };
        match (r1, r2) {
            (Some(a), Some(b)) => Some(if a.distance <= b.distance { a } else { b }),
            (a, b) => a.or(b),
        }
    }

    /// three-mesh-bvh's `intersectRay`: whether the ray's slab interval over
    /// a node's box overlaps `[near, far]`.
    fn ray_meets_node_js(
        &self,
        node_index: u32,
        o: [f64; 3],
        d: [f64; 3],
        near: f64,
        far: f64,
    ) -> bool {
        let b = &self.nodes[node_index as usize].bounds;
        let (min, max) = (
            [b.min.x as f64, b.min.y as f64, b.min.z as f64],
            [b.max.x as f64, b.max.y as f64, b.max.z as f64],
        );
        let slab = |i: usize| {
            let inv = 1.0 / d[i];
            if inv >= 0.0 {
                ((min[i] - o[i]) * inv, (max[i] - o[i]) * inv)
            } else {
                ((max[i] - o[i]) * inv, (min[i] - o[i]) * inv)
            }
        };
        let (mut tmin, mut tmax) = slab(0);
        let (tymin, tymax) = slab(1);
        if tmin > tymax || tymin > tmax {
            return false;
        }
        if tymin > tmin || tmin.is_nan() {
            tmin = tymin;
        }
        if tymax < tmax || tmax.is_nan() {
            tmax = tymax;
        }
        let (tzmin, tzmax) = slab(2);
        if tmin > tzmax || tzmin > tmax {
            return false;
        }
        if tzmin > tmin || tmin.is_nan() {
            tmin = tzmin;
        }
        if tzmax < tmax || tmax.is_nan() {
            tmax = tzmax;
        }
        tmin <= far && tmax >= near
    }

    /// three.js `checkIntersection` on one triangle, in f64.
    fn intersect_triangle_js(
        &self,
        tri: usize,
        origin: [f64; 3],
        dir: [f64; 3],
        side: u32,
        near: f64,
        far: f64,
    ) -> Option<JsRayHit> {
        let (ia, ib, ic) = self.triangle_indices[tri];
        let p = |i: u32| {
            let v = read_vec(&self.positions, i);
            [v.x as f64, v.y as f64, v.z as f64]
        };
        let (a, b, c) = (p(ia), p(ib), p(ic));
        let (t, _, _) = if side == 1 {
            crate::math::intersect_triangle_f64(origin, dir, c, b, a, true)?
        } else {
            crate::math::intersect_triangle_f64(origin, dir, a, b, c, side != 2)?
        };
        // `Ray.at`: origin + direction·t.
        let point = [
            origin[0] + dir[0] * t,
            origin[1] + dir[1] * t,
            origin[2] + dir[2] * t,
        ];
        let (dx, dy, dz) = (
            origin[0] - point[0],
            origin[1] - point[1],
            origin[2] - point[2],
        );
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        if distance < near || distance > far {
            return None;
        }
        // `Triangle.getNormal`: (c − b) × (a − b), normalised.
        let cb = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
        let ab = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let n = [
            cb[1] * ab[2] - cb[2] * ab[1],
            cb[2] * ab[0] - cb[0] * ab[2],
            cb[0] * ab[1] - cb[1] * ab[0],
        ];
        let len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
        let face_normal = if len_sq > 0.0 {
            let inv = 1.0 / len_sq.sqrt();
            [n[0] * inv, n[1] * inv, n[2] * inv]
        } else {
            [0.0; 3]
        };
        Some(JsRayHit {
            distance,
            point,
            face_index: tri,
            face_normal,
        })
    }

    pub fn triangle_at_geometry_index(&self, tri_idx: usize) -> Triangle {
        let (ia, ib, ic) = self.triangle_indices[tri_idx];
        Triangle::new(
            read_vec(&self.positions, ia),
            read_vec(&self.positions, ib),
            read_vec(&self.positions, ic),
        )
    }

    fn intersects_box_node(&self, node_index: u32, box_query: &Box3) -> bool {
        let node = &self.nodes[node_index as usize];
        if !node.bounds.intersects_box(box_query) {
            return false;
        }
        if node.is_leaf {
            let start = node.left_or_offset as usize;
            let count = node.right_or_count as usize;
            for i in start..start + count {
                let tri_idx = self.triangle_order[i];
                let tri = self.triangle_indices[tri_idx];
                let tb = triangle_bounds(&self.positions, tri);
                if tb.intersects_box(box_query) {
                    return true;
                }
            }
            false
        } else {
            self.intersects_box_node(node.left_or_offset, box_query)
                || self.intersects_box_node(node.right_or_count, box_query)
        }
    }

    fn intersects_sphere_node(&self, node_index: u32, sphere: &Sphere) -> bool {
        let node = &self.nodes[node_index as usize];
        if !sphere.intersects_box(&node.bounds) {
            return false;
        }
        if node.is_leaf {
            let start = node.left_or_offset as usize;
            let count = node.right_or_count as usize;
            for i in start..start + count {
                let tri_idx = self.triangle_order[i];
                let tri = self.triangle_indices[tri_idx];
                let (a, b, c) = read_tri(&self.positions, tri);
                if sphere_contains_triangle(sphere, a, b, c) {
                    return true;
                }
            }
            false
        } else {
            self.intersects_sphere_node(node.left_or_offset, sphere)
                || self.intersects_sphere_node(node.right_or_count, sphere)
        }
    }

    fn closest_point_node(
        &self,
        node_index: u32,
        point: Vector3,
        best_point: &mut Vector3,
        best_dist: &mut f32,
        best_tri: &mut usize,
    ) {
        let node = &self.nodes[node_index as usize];
        if node.bounds.distance_to_point(point) >= *best_dist {
            return;
        }
        if node.is_leaf {
            let start = node.left_or_offset as usize;
            let count = node.right_or_count as usize;
            for i in start..start + count {
                let tri_idx = self.triangle_order[i];
                let tri = self.triangle_indices[tri_idx];
                let (a, b, c) = read_tri(&self.positions, tri);
                let cp = closest_point_on_triangle(point, a, b, c);
                let d = cp.distance_to(point);
                if d < *best_dist {
                    *best_dist = d;
                    *best_point = cp;
                    *best_tri = tri_idx;
                }
            }
        } else {
            self.closest_point_node(node.left_or_offset, point, best_point, best_dist, best_tri);
            self.closest_point_node(node.right_or_count, point, best_point, best_dist, best_tri);
        }
    }

    fn raycast_node(
        &self,
        node_index: u32,
        ray: &Ray,
        near: f32,
        far: f32,
        backface_culling: bool,
        hits: &mut Vec<BvhHit>,
    ) {
        let node = &self.nodes[node_index as usize];
        if !ray.intersects_box(&node.bounds) {
            return;
        }

        if node.is_leaf {
            let start = node.left_or_offset as usize;
            let count = node.right_or_count as usize;
            for i in start..start + count {
                let tri_idx = self.triangle_order[i];
                if let Some(hit) = self.intersect_triangle(tri_idx, ray, backface_culling) {
                    if hit.distance >= near && hit.distance <= far {
                        hits.push(hit);
                    }
                }
            }
        } else {
            self.raycast_node(node.left_or_offset, ray, near, far, backface_culling, hits);
            self.raycast_node(node.right_or_count, ray, near, far, backface_culling, hits);
        }
    }

    fn raycast_node_first(
        &self,
        node_index: u32,
        ray: &Ray,
        inv: Vector3,
        near: f32,
        far: f32,
        backface_culling: bool,
        best: &mut Option<BvhHit>,
    ) {
        let node = &self.nodes[node_index as usize];
        if node.is_leaf {
            let start = node.left_or_offset as usize;
            let count = node.right_or_count as usize;
            for i in start..start + count {
                let tri_idx = self.triangle_order[i];
                if let Some(hit) = self.intersect_triangle(tri_idx, ray, backface_culling) {
                    if hit.distance >= near && hit.distance <= far {
                        let replace = match best {
                            None => true,
                            Some(b) => hit.distance < b.distance,
                        };
                        if replace {
                            *best = Some(hit);
                        }
                    }
                }
            }
            return;
        }

        // Each child is slab-tested exactly once, and the nearer one is opened
        // first so the hit it finds tightens the bound the far one is measured
        // against. Testing the node itself on entry as well — the obvious
        // shape — triples the slab work for nothing, since the parent already
        // established that this box is worth opening.
        let (l, r) = (node.left_or_offset, node.right_or_count);
        let dl = box_entry(ray.origin, inv, &self.nodes[l as usize].bounds);
        let dr = box_entry(ray.origin, inv, &self.nodes[r as usize].bounds);
        let (n1, d1, n2, d2) = if dl <= dr {
            (l, dl, r, dr)
        } else {
            (r, dr, l, dl)
        };
        // `box_entry` reports a missed box as infinitely far, so a miss has to
        // be refused outright: with an unbounded `far` and no hit yet, the
        // limit is infinite too and `<=` alone would open every missed box.
        let limit = |best: &Option<BvhHit>| best.as_ref().map(|h| h.distance).unwrap_or(far);
        if d1.is_finite() && d1 <= limit(best) {
            self.raycast_node_first(n1, ray, inv, near, far, backface_culling, best);
        }
        if d2.is_finite() && d2 <= limit(best) {
            self.raycast_node_first(n2, ray, inv, near, far, backface_culling, best);
        }
    }

    fn intersect_triangle(
        &self,
        tri_idx: usize,
        ray: &Ray,
        backface_culling: bool,
    ) -> Option<BvhHit> {
        let (ia, ib, ic) = self.triangle_indices[tri_idx];
        let tri = Triangle::new(
            read_vec(&self.positions, ia),
            read_vec(&self.positions, ib),
            read_vec(&self.positions, ic),
        );
        let (t, u, v) = ray.intersect_triangle_weights(tri.a, tri.b, tri.c, backface_culling)?;
        Some(BvhHit::new(t, ray.at(t), tri_idx, Vector3::new(u, v, 0.0)))
    }
}

/// A first hit from [`MeshBvh::raycast_first_js`], in f64: the distance
/// along the ray, the point, the geometry triangle, and its face normal
/// (three.js's `face.normal`).
#[derive(Debug, Clone, Copy)]
pub struct JsRayHit {
    pub distance: f64,
    pub point: [f64; 3],
    pub face_index: usize,
    pub face_normal: [f64; 3],
}

fn axis_of(v: Vector3, axis: usize) -> f32 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

fn read_vec(positions: &[f32], vi: u32) -> Vector3 {
    let b = vi as usize * 3;
    Vector3::new(positions[b], positions[b + 1], positions[b + 2])
}

fn read_tri(positions: &[f32], tri: (u32, u32, u32)) -> (Vector3, Vector3, Vector3) {
    (
        read_vec(positions, tri.0),
        read_vec(positions, tri.1),
        read_vec(positions, tri.2),
    )
}

fn triangle_bounds(positions: &[f32], tri: (u32, u32, u32)) -> Box3 {
    let mut b = Box3::empty();
    for vi in [tri.0, tri.1, tri.2] {
        b.expand_by_point(read_vec(positions, vi));
    }
    b
}

fn sphere_contains_triangle(sphere: &Sphere, a: Vector3, b: Vector3, c: Vector3) -> bool {
    sphere.contains_point(a)
        || sphere.contains_point(b)
        || sphere.contains_point(c)
        || closest_point_on_triangle(sphere.center, a, b, c).distance_to(sphere.center)
            <= sphere.radius
}

/// Closest point on triangle ABC to point P (Ericson, Real-Time Collision Detection).
fn closest_point_on_triangle(p: Vector3, a: Vector3, b: Vector3, c: Vector3) -> Vector3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

/// Distance at which a ray enters an axis-aligned box, or `INFINITY` if it
/// misses. Zero when the origin is already inside.
///
/// `Ray::intersect_box` cannot be used for this: it returns the *exit*
/// distance when the origin is inside the box, which a traversal would read as
/// a far-away entry and cull — starting with the root, on every ray that
/// begins inside the scene.
///
/// A ray parallel to an axis (a zero direction component, either sign) whose
/// origin lies exactly on one of that axis's faces gives `0 · ∞ = NaN` for
/// that face. three.js's `Ray.intersectBox` drops a NaN bound, so a ray
/// running along a face counts as inside the slab; so does this. Rust's
/// `f32::min`/`max` would instead return the *other* bound, which for such a
/// ray is the infinite one on the wrong side, and cull a box the ray is in.
#[inline(always)]
fn box_entry(origin: Vector3, inv: Vector3, b: &Box3) -> f32 {
    #[inline(always)]
    fn axis(o: f32, min: f32, max: f32, inv: f32) -> (f32, f32) {
        let (near, far) = if inv >= 0.0 {
            ((min - o) * inv, (max - o) * inv)
        } else {
            ((max - o) * inv, (min - o) * inv)
        };
        let lowest = f32::NEG_INFINITY;
        let near = if near.is_nan() { lowest } else { near };
        let far = if far.is_nan() { f32::INFINITY } else { far };
        (near, far)
    }
    let (x0, x1) = axis(origin.x, b.min.x, b.max.x, inv.x);
    let (y0, y1) = axis(origin.y, b.min.y, b.max.y, inv.y);
    let (z0, z1) = axis(origin.z, b.min.z, b.max.z, inv.z);
    let lo = x0.max(y0).max(z0).max(0.0);
    let hi = x1.min(y1).min(z1);
    if hi < lo {
        f32::INFINITY
    } else {
        lo
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BufferAttribute, BufferGeometry, Mesh, Object3D, ObjectArena, Raycaster};
    use crate::materials::{BasicMaterial, Material};
    use crate::math::Color;
    use crate::mesh_bvh::{BuildOptions, BACK_SIDE, DOUBLE_SIDE, FRONT_SIDE};

    fn unit_triangle_geometry() -> BufferGeometry {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3),
        );
        g
    }

    fn indexed_cube_geometry() -> BufferGeometry {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(
                vec![
                    -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0,
                    1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
                ],
                3,
            ),
        );
        g.set_index(vec![
            0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 0, 4, 7, 0, 7, 3, 1, 5, 6, 1, 6, 2, 3, 2, 6, 3, 6,
            7, 0, 1, 5, 0, 5, 4,
        ]);
        g
    }

    fn bvh_hits_match_bruteforce(geometry: BufferGeometry, ray: Ray) {
        let bvh = MeshBvh::build(&geometry, BuildOptions::default()).expect("bvh build");
        let bvh_hits = bvh.raycast(&ray, 0.0, f32::INFINITY, false);

        let mut geom_with_bvh = geometry;
        geom_with_bvh
            .compute_bounds_tree(BuildOptions::default())
            .expect("compute_bounds_tree");

        let mut arena = ObjectArena::new();
        let root = arena.insert(Object3D::group());
        // DoubleSide, to compare against the unculled BVH cast: the raycaster
        // honours the material's side as three.js does.
        let mut basic = BasicMaterial::new(Color::from_hex(0xffffff));
        basic.side = DOUBLE_SIDE;
        let mat = Material::Basic(basic);
        let mesh_id = arena.insert(Object3D::mesh(Mesh::new(geom_with_bvh, mat)));
        arena.add_child(root, mesh_id);
        arena.update_world_matrices(root, crate::math::Matrix4::identity());

        let mut rc = Raycaster::new(ray.origin, ray.direction, 0.0, f32::INFINITY);
        rc.ray = ray;
        let brute = rc.intersect_objects(&arena, root, true);

        assert_eq!(bvh_hits.len(), brute.len());
        for (bh, br) in bvh_hits.iter().zip(brute.iter()) {
            assert!((bh.distance - br.distance).abs() < 1e-3);
            assert_eq!(bh.face_index, br.face_index);
        }
    }

    #[test]
    fn bvh_matches_bruteforce_unit_triangle() {
        let ray = Ray::new(Vector3::new(0.25, 0.25, -1.0), Vector3::new(0.0, 0.0, 1.0));
        bvh_hits_match_bruteforce(unit_triangle_geometry(), ray);
    }

    #[test]
    fn bvh_matches_bruteforce_indexed_cube() {
        let ray = Ray::new(Vector3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        bvh_hits_match_bruteforce(indexed_cube_geometry(), ray);
    }

    /// A ray running exactly along a face of a node's box, with a zero
    /// direction component of either sign: `0 · ∞` is NaN in the slab test,
    /// and both traversals must still open the box, as three.js's
    /// `Ray.intersectBox` does. Every box of a flat plane has zero
    /// thickness, and a pick on the plane's edge lies on its children's faces
    /// too — 16×16 quads, so there are children to open.
    #[test]
    fn grazing_rays_open_their_boxes() {
        let geom = crate::geometries::PlaneGeometry::with_segments(2.0, 2.0, 16, 16);
        let bvh = MeshBvh::build(&geom, BuildOptions::default()).expect("bvh");
        assert!(bvh.nodes.len() > 3, "the test needs interior nodes");
        for (x, y) in [(1.0, 0.3), (-1.0, 0.3), (0.3, 1.0), (0.3, -1.0), (1.0, 1.0)] {
            for dir in [Vector3::new(0.0, 0.0, -1.0), Vector3::new(-0.0, -0.0, -1.0)] {
                let ray = Ray::new(Vector3::new(x, y, 5.0), dir);
                let all = bvh.raycast(&ray, 0.0, f32::INFINITY, true);
                assert!(!all.is_empty(), "raycast ({x}, {y}) dir {dir:?}");
                let first = bvh.raycast_first(&ray, 0.0, f32::INFINITY, true);
                let first = first.unwrap_or_else(|| panic!("raycast_first ({x}, {y}) {dir:?}"));
                assert!((first.distance - 5.0).abs() < 1e-6);
            }
        }
        // Grazing *outside* the box still misses.
        let ray = Ray::new(Vector3::new(1.0001, 0.3, 5.0), Vector3::new(0.0, 0.0, -1.0));
        assert!(bvh.raycast_first(&ray, 0.0, f32::INFINITY, true).is_none());
        assert!(bvh.raycast(&ray, 0.0, f32::INFINITY, true).is_empty());
    }

    #[test]
    fn serialize_roundtrip() {
        let geom = indexed_cube_geometry();
        let bvh = MeshBvh::build(&geom, BuildOptions::default()).expect("bvh");
        let data = bvh.serialize();
        let restored = MeshBvh::deserialize(data).expect("deserialize");
        let ray = Ray::new(Vector3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));
        assert_eq!(
            bvh.raycast(&ray, 0.0, f32::INFINITY, false).len(),
            restored.raycast(&ray, 0.0, f32::INFINITY, false).len()
        );
    }

    #[test]
    fn resolve_triangle_index_roundtrip() {
        let geom = indexed_cube_geometry();
        let bvh = MeshBvh::build(
            &geom,
            BuildOptions {
                indirect: true,
                ..Default::default()
            },
        )
        .expect("bvh");
        for i in 0..bvh.triangle_count() {
            let resolved = bvh.resolve_triangle_index(i).expect("resolve");
            assert_eq!(resolved, bvh.triangle_order()[i]);
        }
    }

    #[test]
    fn bvhcast_finds_overlapping_boxes() {
        let geom_a = indexed_cube_geometry();
        let mut geom_b = indexed_cube_geometry();
        let mut shifted = geom_b.get_attribute("position").unwrap().array.clone();
        for v in shifted.iter_mut() {
            *v += 0.5;
        }
        geom_b.set_attribute("position", BufferAttribute::new(shifted, 3));
        let bvh_a = MeshBvh::build(&geom_a, BuildOptions::default()).expect("bvh a");
        let bvh_b = MeshBvh::build(&geom_b, BuildOptions::default()).expect("bvh b");
        let matrix = crate::math::Matrix4::identity();
        let pairs = bvh_a.bvhcast(&bvh_b, &matrix);
        assert!(!pairs.is_empty(), "expected overlapping triangle pairs");
    }

    /// Two trees of very different sizes, overlapping.
    ///
    /// The dual descent alternates which tree it walks, and the node index it
    /// produced while walking A used to be written into the slot the next frame
    /// reads as a B index. With both trees the same shape that is invisible —
    /// the index is in range in either one, and the pairs come out plausible.
    /// Give one tree a hundred times the nodes of the other and it reads off the
    /// end of the small one, which is what a sphere cast against a cube did.
    #[test]
    fn bvhcast_between_trees_of_different_sizes() {
        let small =
            MeshBvh::build(&unit_triangle_geometry(), BuildOptions::default()).expect("small bvh");

        // A ball of a few thousand triangles whose *shell* cuts through the
        // triangle above — a ball merely containing it would share no bounding
        // boxes at all, and the cast would be empty for the honest reason.
        let mut sphere = crate::geometries::SphereGeometry::new(0.45, 48, 32);
        if let Some(pos) = sphere.get_attribute("position") {
            let moved: Vec<f32> = pos
                .array
                .chunks_exact(3)
                .flat_map(|p| [p[0] + 0.3, p[1] + 0.3, p[2]])
                .collect();
            sphere.set_attribute("position", BufferAttribute::new(moved, 3));
        }
        let big = MeshBvh::build(&sphere, BuildOptions::default()).expect("big bvh");
        assert!(
            big.node_count() > small.node_count() * 4,
            "the point of this test is the size difference: {} vs {}",
            big.node_count(),
            small.node_count()
        );

        let matrix = crate::math::Matrix4::identity();
        // Both ways round: which tree is `self` decides which one the reversed
        // half of the descent walks.
        assert!(!small.bvhcast(&big, &matrix).is_empty());
        assert!(!big.bvhcast(&small, &matrix).is_empty());
    }

    #[test]
    fn raycast_first_double_side() {
        let geom = indexed_cube_geometry();
        let bvh = MeshBvh::build(&geom, BuildOptions::default()).expect("bvh");
        let ray = Ray::new(Vector3::new(0.0, 0.0, 5.0), Vector3::new(0.0, 0.0, -1.0));
        let front = bvh.raycast_first_with_side(&ray, 0.0, f32::INFINITY, FRONT_SIDE);
        let back = bvh.raycast_first_with_side(&ray, 0.0, f32::INFINITY, BACK_SIDE);
        let both = bvh.raycast_first_with_side(&ray, 0.0, f32::INFINITY, DOUBLE_SIDE);
        assert!(front.is_some());
        assert!(both.is_some());
        assert_eq!(front.unwrap().face_index, both.unwrap().face_index);
        assert!(back.is_none() || back.unwrap().distance >= front.as_ref().unwrap().distance);
    }

    #[test]
    fn refit_updates_bounds() {
        let geom = unit_triangle_geometry();
        let mut bvh = MeshBvh::build(&geom, BuildOptions::default()).expect("bvh");
        let old_min = bvh.bounding_box().min;
        let mut shifted: Vec<f32> = geom.get_attribute("position").unwrap().array.clone();
        for v in shifted.iter_mut() {
            *v += 1.0;
        }
        bvh.refit(&shifted);
        assert!(bvh.bounding_box().min.x > old_min.x);
    }
}
