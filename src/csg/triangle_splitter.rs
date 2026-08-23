//! Mirrors `web/csg/core/TriangleSplitter.js` precisely.

use super::js_topology::{
    is_tri_degenerate, JsLine3, JsPlane, JsTriangle, JsVec3, SplitBary, TOPO_EPSILON,
};

const COPLANAR_EPSILON: f64 = TOPO_EPSILON;
const PARALLEL_EPSILON: f64 = TOPO_EPSILON;

/// Mirrors `TrianglePool` in `TriangleSplitter.js`.
#[derive(Debug)]
struct TrianglePool {
    pool: Vec<JsTriangle>,
    index: usize,
}

impl TrianglePool {
    fn new() -> Self {
        Self {
            pool: Vec::new(),
            index: 0,
        }
    }

    fn get_triangle(&mut self) -> usize {
        if self.index >= self.pool.len() {
            self.pool.push(JsTriangle::default());
        }
        let i = self.index;
        self.index += 1;
        i
    }

    fn clear(&mut self) {
        self.index = 0;
    }
}

const DEGENERATE_EPSILON: f64 = 1e-14;

/// Ceiling on how many fragments one source triangle may be cut into. Each
/// `split_by_triangle` can multiply the pool, so a triangle overlapping many
/// near-degenerate neighbours (the pathological coarse-union case that pushes the
/// exact kernel into this float fallback) grows roughly as 2^N and never returns.
/// Past this bound `split_by_triangle` becomes a no-op, so the boolean still
/// terminates with a slightly-approximate result for that one triangle instead of
/// hanging — a deterministic guard that holds identically on wasm (where a live
/// loop can't be wall-clock-timed-out) and native. Legitimate dense intersections
/// stay far below this; it only ever trips on runaway degenerate splits.
const MAX_SPLIT_FRAGMENTS: usize = 50_000;

/// Mirrors `web/csg/core/TriangleSplitter.js`.
#[derive(Debug)]
pub struct TriangleSplitter {
    triangle_pool: TrianglePool,
    /// Indices into `triangle_pool.pool` for the active clipped triangles.
    triangles: Vec<usize>,
    pub normal: JsVec3,
    pub coplanar_triangle_used: bool,
}

impl Default for TriangleSplitter {
    fn default() -> Self {
        Self {
            triangle_pool: TrianglePool::new(),
            triangles: Vec::new(),
            normal: JsVec3::default(),
            coplanar_triangle_used: false,
        }
    }
}

impl TriangleSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.triangles.clear();
        self.triangle_pool.clear();
        self.coplanar_triangle_used = false;
    }

    /// Mirrors `initialize(tri)`.
    pub fn initialize(&mut self, tri: JsTriangle) {
        self.reset();
        self.normal = tri.get_normal();
        let idx = self.triangle_pool.get_triangle();
        self.triangle_pool.pool[idx] = tri;
        self.triangles.push(idx);
    }

    /// Mirrors `splitByTriangle(triangle)`.
    pub fn split_by_triangle(&mut self, triangle: JsTriangle) {
        // Runaway-split guard: stop cutting once the pool blows past the ceiling so
        // a degenerate self-overlap can't spin forever (see MAX_SPLIT_FRAGMENTS).
        if self.triangles.len() > MAX_SPLIT_FRAGMENTS {
            return;
        }
        let clip = triangle;
        let tri_normal = clip.get_normal().normalize();

        if (1.0 - tri_normal.dot(self.normal).abs()).abs() < PARALLEL_EPSILON {
            self.coplanar_triangle_used = true;

            let verts = [clip.a, clip.b, clip.c];
            for i in 0..3 {
                let next = (i + 1) % 3;
                let v0 = verts[i];
                let v1 = verts[next];
                let edge_dir = JsVec3::default().sub_vectors(v1, v0).normalize();
                let plane_normal = JsVec3::default().cross_vectors(tri_normal, edge_dir);
                let plane = JsPlane::default().set_from_normal_and_coplanar_point(plane_normal, v0);
                self.split_by_plane(plane, clip);
            }
        } else {
            let plane = clip.get_plane();
            self.split_by_plane(plane, clip);
        }
    }

    /// Mirrors `splitByPlane(plane, clippingTriangle)`.
    fn split_by_plane(&mut self, plane: JsPlane, clipping_triangle: JsTriangle) {
        let mut edge = JsLine3::default();
        let mut found_edge = JsLine3::default();
        let mut hit_vec = JsVec3::default();

        let mut i = 0usize;
        let mut l = self.triangles.len();
        while i < l {
            let tri_idx = self.triangles[i];
            let tri = self.triangle_pool.pool[tri_idx];

            if !clipping_triangle.intersects_triangle(tri, true) {
                i += 1;
                continue;
            }

            let arr = [tri.a, tri.b, tri.c];
            let mut intersects = 0;
            let mut vertex_split_end = -1i32;
            let mut coplanar_edge = false;
            let mut pos_side_verts = Vec::new();
            let mut neg_side_verts = Vec::new();

            for t in 0..3 {
                let t_next = (t + 1) % 3;
                edge.start = arr[t];
                edge.end = arr[t_next];

                let start_dist = plane.distance_to_point(edge.start);
                let end_dist = plane.distance_to_point(edge.end);
                if start_dist.abs() < COPLANAR_EPSILON && end_dist.abs() < COPLANAR_EPSILON {
                    coplanar_edge = true;
                    break;
                }

                if start_dist > 0.0 {
                    pos_side_verts.push(t);
                } else {
                    neg_side_verts.push(t);
                }

                if start_dist.abs() < COPLANAR_EPSILON {
                    continue;
                }

                let mut did_intersect = plane
                    .intersect_line(edge.start, edge.end, &mut hit_vec)
                    .is_some();
                if !did_intersect && end_dist.abs() < COPLANAR_EPSILON {
                    hit_vec = edge.end;
                    did_intersect = true;
                }

                let from_start = hit_vec.distance_to(edge.start);
                if did_intersect && (from_start >= TOPO_EPSILON || from_start.is_nan()) {
                    if hit_vec.distance_to(edge.end) < TOPO_EPSILON {
                        vertex_split_end = t as i32;
                    }

                    if intersects == 0 {
                        found_edge.start = hit_vec;
                    } else {
                        found_edge.end = hit_vec;
                    }
                    intersects += 1;
                }
            }

            if !coplanar_edge && intersects == 2 && found_edge.distance() > COPLANAR_EPSILON {
                if vertex_split_end != -1 {
                    let vertex_split_end = ((vertex_split_end + 1) % 3) as usize;

                    let mut other_vert1 = 0usize;
                    if other_vert1 == vertex_split_end {
                        other_vert1 = (other_vert1 + 1) % 3;
                    }
                    let mut other_vert2 = other_vert1 + 1;
                    if other_vert2 == vertex_split_end {
                        other_vert2 = (other_vert2 + 1) % 3;
                    }

                    let next_idx = self.triangle_pool.get_triangle();
                    self.triangle_pool.pool[next_idx] = JsTriangle {
                        a: arr[other_vert2],
                        b: found_edge.end,
                        c: found_edge.start,
                    };
                    if !is_tri_degenerate(self.triangle_pool.pool[next_idx], DEGENERATE_EPSILON) {
                        self.triangles.push(next_idx);
                    }

                    self.triangle_pool.pool[tri_idx] = JsTriangle {
                        a: arr[other_vert1],
                        b: found_edge.start,
                        c: found_edge.end,
                    };

                    if is_tri_degenerate(self.triangle_pool.pool[tri_idx], DEGENERATE_EPSILON) {
                        self.triangles.remove(i);
                        l -= 1;
                    } else {
                        i += 1;
                    }
                } else {
                    let single_vert = if pos_side_verts.len() >= 2 {
                        neg_side_verts[0]
                    } else {
                        pos_side_verts[0]
                    };

                    let (mut fs, mut fe) = (found_edge.start, found_edge.end);
                    if single_vert == 0 {
                        std::mem::swap(&mut fs, &mut fe);
                    }

                    let next_vert1 = (single_vert + 1) % 3;
                    let next_vert2 = (single_vert + 2) % 3;

                    let (tri1, tri2) = if arr[next_vert1].distance_to_squared(fs)
                        < arr[next_vert2].distance_to_squared(fe)
                    {
                        (
                            JsTriangle {
                                a: arr[next_vert1],
                                b: fs,
                                c: fe,
                            },
                            JsTriangle {
                                a: arr[next_vert1],
                                b: arr[next_vert2],
                                c: fs,
                            },
                        )
                    } else {
                        (
                            JsTriangle {
                                a: arr[next_vert2],
                                b: fs,
                                c: fe,
                            },
                            JsTriangle {
                                a: arr[next_vert1],
                                b: arr[next_vert2],
                                c: fe,
                            },
                        )
                    };

                    self.triangle_pool.pool[tri_idx] = JsTriangle {
                        a: arr[single_vert],
                        b: fe,
                        c: fs,
                    };

                    let tri1_idx = self.triangle_pool.get_triangle();
                    self.triangle_pool.pool[tri1_idx] = tri1;
                    if !is_tri_degenerate(self.triangle_pool.pool[tri1_idx], DEGENERATE_EPSILON) {
                        self.triangles.push(tri1_idx);
                    }

                    let tri2_idx = self.triangle_pool.get_triangle();
                    self.triangle_pool.pool[tri2_idx] = tri2;
                    if !is_tri_degenerate(self.triangle_pool.pool[tri2_idx], DEGENERATE_EPSILON) {
                        self.triangles.push(tri2_idx);
                    }

                    if is_tri_degenerate(self.triangle_pool.pool[tri_idx], DEGENERATE_EPSILON) {
                        self.triangles.remove(i);
                        l -= 1;
                    } else {
                        i += 1;
                    }
                }
            } else {
                i += 1;
            }
        }
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    pub fn clipped_triangle(&self, i: usize) -> JsTriangle {
        self.triangle_pool.pool[self.triangles[i]]
    }

    /// Active clipped triangles (f64 pool, mirrors JS `splitter.triangles`).
    #[cfg(test)]
    pub fn clipped_js_triangles(&self) -> Vec<JsTriangle> {
        self.triangles
            .iter()
            .map(|&i| self.triangle_pool.pool[i])
            .collect()
    }

    #[cfg(test)]
    pub fn clipped_with_bary(
        &self,
        tri_a: JsTriangle,
    ) -> impl Iterator<Item = (JsTriangle, SplitBary)> + '_ {
        self.triangles.iter().map(move |&i| {
            let clipped = self.triangle_pool.pool[i];
            (clipped, Self::barycoords_in_tri(tri_a, clipped))
        })
    }

    pub(crate) fn barycoords_in_tri(tri_a: JsTriangle, clipped: JsTriangle) -> SplitBary {
        (
            tri_a.get_barycoord(clipped.a),
            tri_a.get_barycoord(clipped.b),
            tri_a.get_barycoord(clipped.c),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csg::js_topology::JsTriangle;
    use crate::math::{Triangle, Vector3};

    #[test]
    fn clip_intersects_unit_triangle() {
        let tri = JsTriangle::from_triangle(Triangle::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(0.0, 2.0, 0.0),
        ));
        let clip = JsTriangle::from_triangle(Triangle::new(
            Vector3::new(0.5, -1.0, 0.0),
            Vector3::new(0.5, 2.0, 0.0),
            Vector3::new(0.5, 0.0, 1.0),
        ));
        assert!(clip.intersects_triangle(tri, true));
    }

    #[test]
    fn split_by_triangle_increases_count() {
        let mut splitter = TriangleSplitter::new();
        let tri = JsTriangle::from_triangle(Triangle::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(0.0, 2.0, 0.0),
        ));
        splitter.initialize(tri);
        let clip = JsTriangle::from_triangle(Triangle::new(
            Vector3::new(0.5, -1.0, 0.0),
            Vector3::new(0.5, 2.0, 0.0),
            Vector3::new(0.5, 0.0, 1.0),
        ));
        splitter.split_by_triangle(clip);
        assert!(
            splitter.triangle_count() > 1,
            "expected split, got {}",
            splitter.triangle_count()
        );
    }
}
