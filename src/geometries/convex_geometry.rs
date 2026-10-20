//! three.js r165 `ConvexHull` (examples/jsm/math/ConvexHull.js, itself a port
//! of quickhull3d) and `ConvexGeometry`, ported step for step.
//!
//! Faithful down to the order the hull's faces come out in, because
//! `ConvexGeometry` writes one triangle per face in that order: a different —
//! equally correct — quickhull would give a different buffer. The arithmetic is
//! `f64`, as JavaScript's is.

use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::Vector3;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn length(a: V3) -> f64 {
    dot(a, a).sqrt()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mark {
    Visible,
    Deleted,
}

/// One hull face: a triangle when created, as every quickhull3d face is.
#[derive(Debug, Clone)]
pub struct HullFace {
    pub normal: V3,
    pub midpoint: V3,
    pub area: f64,
    /// Signed distance from the face's plane to the origin along `normal`.
    pub constant: f64,
    outside: Option<usize>,
    mark: Mark,
    /// One of the face's half-edges.
    edge: usize,
}

impl HullFace {
    pub fn distance_to_point(&self, p: V3) -> f64 {
        dot(self.normal, p) - self.constant
    }
}

#[derive(Debug, Clone)]
struct HalfEdge {
    vertex: usize,
    prev: usize,
    next: usize,
    twin: Option<usize>,
    face: usize,
}

#[derive(Debug, Clone)]
struct VertexNode {
    point: V3,
    prev: Option<usize>,
    next: Option<usize>,
    face: Option<usize>,
}

/// A doubly linked list threaded through [`VertexNode`]s.
#[derive(Debug, Clone, Copy, Default)]
struct VertexList {
    head: Option<usize>,
    tail: Option<usize>,
}

/// three.js `ConvexHull`.
#[derive(Debug, Clone, Default)]
pub struct ConvexHull {
    pub tolerance: f64,
    faces: Vec<usize>,
    new_faces: Vec<usize>,
    assigned: VertexList,
    unassigned: VertexList,
    vertices: Vec<VertexNode>,
    face_pool: Vec<HullFace>,
    edges: Vec<HalfEdge>,
}

impl ConvexHull {
    pub fn new() -> Self {
        Self {
            tolerance: -1.0,
            ..Default::default()
        }
    }

    /// `setFromPoints`: fewer than four points leave the hull as it was, as
    /// three.js does. A degenerate (flat or collinear) set gives no faces
    /// where three.js would throw.
    pub fn set_from_points(&mut self, points: &[V3]) -> &mut Self {
        if points.len() >= 4 {
            self.make_empty();
            self.vertices = points
                .iter()
                .map(|&point| VertexNode {
                    point,
                    prev: None,
                    next: None,
                    face: None,
                })
                .collect();
            if self.compute().is_none() {
                self.faces.clear();
            }
        }
        self
    }

    pub fn make_empty(&mut self) -> &mut Self {
        self.faces.clear();
        self.vertices.clear();
        self.face_pool.clear();
        self.edges.clear();
        self.new_faces.clear();
        self.assigned = VertexList::default();
        self.unassigned = VertexList::default();
        self
    }

    /// The hull's faces, in three.js's order.
    pub fn faces(&self) -> impl Iterator<Item = &HullFace> + '_ {
        self.faces.iter().map(|&f| &self.face_pool[f])
    }

    /// Each face's corners as indices into the input points, walked from
    /// `face.edge` the way `ConvexGeometry` walks them (`edge.head()`, then
    /// `edge.next`).
    pub fn face_vertex_indices(&self) -> Vec<Vec<usize>> {
        self.faces
            .iter()
            .map(|&f| {
                let start = self.face_pool[f].edge;
                let mut out = Vec::with_capacity(3);
                let mut e = start;
                loop {
                    out.push(self.edges[e].vertex);
                    e = self.edges[e].next;
                    if e == start {
                        break;
                    }
                }
                out
            })
            .collect()
    }

    pub fn point(&self, vertex: usize) -> V3 {
        self.vertices[vertex].point
    }

    pub fn contains_point(&self, p: V3) -> bool {
        self.faces().all(|f| f.distance_to_point(p) <= self.tolerance)
    }

    /// `intersectRay`: the distance along `direction` from `origin` to the
    /// hull, or `None`. three.js returns `ray.at(t)`; the caller does that.
    pub fn intersect_ray(&self, origin: V3, direction: V3) -> Option<f64> {
        let mut t_near = f64::NEG_INFINITY;
        let mut t_far = f64::INFINITY;
        for face in self.faces() {
            let vn = face.distance_to_point(origin);
            let vd = dot(face.normal, direction);
            if vn > 0.0 && vd >= 0.0 {
                return None;
            }
            let t = if vd != 0.0 { -vn / vd } else { 0.0 };
            if t <= 0.0 {
                continue;
            }
            if vd > 0.0 {
                t_far = t.min(t_far);
            } else {
                t_near = t.max(t_near);
            }
            if t_near > t_far {
                return None;
            }
        }
        Some(if t_near != f64::NEG_INFINITY { t_near } else { t_far })
    }

    // ---- vertex lists ----

    fn list_append(&mut self, which: bool, v: usize) {
        let list = if which { self.assigned } else { self.unassigned };
        let mut list = list;
        match list.head {
            None => list.head = Some(v),
            Some(_) => self.vertices[list.tail.unwrap()].next = Some(v),
        }
        self.vertices[v].prev = list.tail;
        self.vertices[v].next = None;
        list.tail = Some(v);
        self.set_list(which, list);
    }

    fn set_list(&mut self, which: bool, list: VertexList) {
        if which {
            self.assigned = list;
        } else {
            self.unassigned = list;
        }
    }

    fn list_append_chain(&mut self, which: bool, v: usize) {
        let mut list = if which { self.assigned } else { self.unassigned };
        match list.head {
            None => list.head = Some(v),
            Some(_) => self.vertices[list.tail.unwrap()].next = Some(v),
        }
        self.vertices[v].prev = list.tail;
        let mut v = v;
        while let Some(n) = self.vertices[v].next {
            v = n;
        }
        list.tail = Some(v);
        self.set_list(which, list);
    }

    fn assigned_insert_before(&mut self, target: usize, v: usize) {
        self.vertices[v].prev = self.vertices[target].prev;
        self.vertices[v].next = Some(target);
        match self.vertices[v].prev {
            None => self.assigned.head = Some(v),
            Some(p) => self.vertices[p].next = Some(v),
        }
        self.vertices[target].prev = Some(v);
    }

    fn assigned_remove(&mut self, v: usize) {
        let (prev, next) = (self.vertices[v].prev, self.vertices[v].next);
        match prev {
            None => self.assigned.head = next,
            Some(p) => self.vertices[p].next = next,
        }
        match next {
            None => self.assigned.tail = prev,
            Some(n) => self.vertices[n].prev = prev,
        }
    }

    fn assigned_remove_sub_list(&mut self, a: usize, b: usize) {
        let (a_prev, b_next) = (self.vertices[a].prev, self.vertices[b].next);
        match a_prev {
            None => self.assigned.head = b_next,
            Some(p) => self.vertices[p].next = b_next,
        }
        match b_next {
            None => self.assigned.tail = a_prev,
            Some(n) => self.vertices[n].prev = a_prev,
        }
    }

    // ---- faces ----

    fn create_face(&mut self, a: usize, b: usize, c: usize) -> usize {
        let f = self.face_pool.len();
        let e0 = self.edges.len();
        let (e1, e2) = (e0 + 1, e0 + 2);
        for (v, prev, next) in [(a, e2, e1), (b, e0, e2), (c, e1, e0)] {
            self.edges.push(HalfEdge {
                vertex: v,
                prev,
                next,
                twin: None,
                face: f,
            });
        }
        let (pa, pb, pc) = (
            self.vertices[self.edges[self.edges[e0].prev].vertex].point,
            self.vertices[self.edges[e0].vertex].point,
            self.vertices[self.edges[self.edges[e0].next].vertex].point,
        );
        // Triangle.getNormal / getMidpoint / getArea.
        let n = cross(sub(pc, pb), sub(pa, pb));
        let lsq = dot(n, n);
        let normal = if lsq > 0.0 {
            let s = 1.0 / lsq.sqrt();
            [n[0] * s, n[1] * s, n[2] * s]
        } else {
            [0.0; 3]
        };
        let third = 1.0 / 3.0;
        let midpoint = [
            (pa[0] + pb[0] + pc[0]) * third,
            (pa[1] + pb[1] + pc[1]) * third,
            (pa[2] + pb[2] + pc[2]) * third,
        ];
        let area = length(cross(sub(pc, pb), sub(pa, pb))) * 0.5;
        let constant = dot(normal, midpoint);
        self.face_pool.push(HullFace {
            normal,
            midpoint,
            area,
            constant,
            outside: None,
            mark: Mark::Visible,
            edge: e0,
        });
        f
    }

    fn get_edge(&self, face: usize, mut i: i32) -> usize {
        let mut e = self.face_pool[face].edge;
        while i > 0 {
            e = self.edges[e].next;
            i -= 1;
        }
        while i < 0 {
            e = self.edges[e].prev;
            i += 1;
        }
        e
    }

    fn set_twin(&mut self, a: usize, b: usize) {
        self.edges[a].twin = Some(b);
        self.edges[b].twin = Some(a);
    }

    fn head(&self, e: usize) -> usize {
        self.edges[e].vertex
    }

    fn tail(&self, e: usize) -> usize {
        self.edges[self.edges[e].prev].vertex
    }

    fn dist(&self, face: usize, v: usize) -> f64 {
        self.face_pool[face].distance_to_point(self.vertices[v].point)
    }

    // ---- quickhull ----

    fn add_vertex_to_face(&mut self, v: usize, face: usize) {
        self.vertices[v].face = Some(face);
        match self.face_pool[face].outside {
            None => self.list_append(true, v),
            Some(o) => self.assigned_insert_before(o, v),
        }
        self.face_pool[face].outside = Some(v);
    }

    fn remove_vertex_from_face(&mut self, v: usize, face: usize) {
        if self.face_pool[face].outside == Some(v) {
            let next = self.vertices[v].next;
            self.face_pool[face].outside = match next {
                Some(n) if self.vertices[n].face == Some(face) => Some(n),
                _ => None,
            };
        }
        self.assigned_remove(v);
    }

    fn remove_all_vertices_from_face(&mut self, face: usize) -> Option<usize> {
        let start = self.face_pool[face].outside?;
        let mut end = start;
        while let Some(n) = self.vertices[end].next {
            if self.vertices[n].face != Some(face) {
                break;
            }
            end = n;
        }
        self.assigned_remove_sub_list(start, end);
        self.vertices[start].prev = None;
        self.vertices[end].next = None;
        self.face_pool[face].outside = None;
        Some(start)
    }

    fn delete_face_vertices(&mut self, face: usize, absorbing: Option<usize>) {
        if let Some(first) = self.remove_all_vertices_from_face(face) {
            match absorbing {
                None => self.list_append_chain(false, first),
                Some(absorbing) => {
                    let mut v = Some(first);
                    while let Some(cur) = v {
                        let next = self.vertices[cur].next;
                        if self.dist(absorbing, cur) > self.tolerance {
                            self.add_vertex_to_face(cur, absorbing);
                        } else {
                            self.list_append(false, cur);
                        }
                        v = next;
                    }
                }
            }
        }
    }

    fn resolve_unassigned_points(&mut self) {
        let mut v = self.unassigned.head;
        while let Some(cur) = v {
            let next = self.vertices[cur].next;
            let mut max_distance = self.tolerance;
            let mut max_face = None;
            for i in 0..self.new_faces.len() {
                let face = self.new_faces[i];
                if self.face_pool[face].mark == Mark::Visible {
                    let d = self.dist(face, cur);
                    if d > max_distance {
                        max_distance = d;
                        max_face = Some(face);
                    }
                    if max_distance > 1000.0 * self.tolerance {
                        break;
                    }
                }
            }
            if let Some(f) = max_face {
                self.add_vertex_to_face(cur, f);
            }
            v = next;
        }
    }

    fn compute_extremes(&mut self) -> ([usize; 3], [usize; 3]) {
        let mut min_v = [0usize; 3];
        let mut max_v = [0usize; 3];
        let mut min = self.vertices[0].point;
        let mut max = self.vertices[0].point;
        for (i, v) in self.vertices.iter().enumerate() {
            let p = v.point;
            for j in 0..3 {
                if p[j] < min[j] {
                    min[j] = p[j];
                    min_v[j] = i;
                }
            }
            for j in 0..3 {
                if p[j] > max[j] {
                    max[j] = p[j];
                    max_v[j] = i;
                }
            }
        }
        self.tolerance = 3.0
            * f64::EPSILON
            * (min[0].abs().max(max[0].abs())
                + min[1].abs().max(max[1].abs())
                + min[2].abs().max(max[2].abs()));
        (min_v, max_v)
    }

    fn compute_initial_hull(&mut self) -> Option<()> {
        let (min, max) = self.compute_extremes();
        let mut max_distance = 0.0;
        let mut index = 0;
        for i in 0..3 {
            let d = self.vertices[max[i]].point[i] - self.vertices[min[i]].point[i];
            if d > max_distance {
                max_distance = d;
                index = i;
            }
        }
        let v0 = min[index];
        let v1 = max[index];
        let (p0, p1) = (self.vertices[v0].point, self.vertices[v1].point);

        // Line3.closestPointToPoint(point, clamp = true).
        let mut v2 = None;
        max_distance = 0.0;
        let delta = sub(p1, p0);
        let se2 = dot(delta, delta);
        for (i, v) in self.vertices.iter().enumerate() {
            if i != v0 && i != v1 {
                let t = (dot(delta, sub(v.point, p0)) / se2).clamp(0.0, 1.0);
                let c = [delta[0] * t + p0[0], delta[1] * t + p0[1], delta[2] * t + p0[2]];
                let d = sub(c, v.point);
                let d = dot(d, d);
                if d > max_distance {
                    max_distance = d;
                    v2 = Some(i);
                }
            }
        }
        let v2 = v2?;
        let p2 = self.vertices[v2].point;

        // Plane.setFromCoplanarPoints(p0, p1, p2).
        let n = cross(sub(p2, p1), sub(p0, p1));
        let l = length(n);
        let l = if l == 0.0 { 1.0 } else { l };
        let pn = [n[0] / l, n[1] / l, n[2] / l];
        let pc = -dot(p0, pn);
        let plane_dist = |p: V3| dot(pn, p) + pc;
        let mut v3 = None;
        max_distance = -1.0;
        for (i, v) in self.vertices.iter().enumerate() {
            if i != v0 && i != v1 && i != v2 {
                let d = plane_dist(v.point).abs();
                if d > max_distance {
                    max_distance = d;
                    v3 = Some(i);
                }
            }
        }
        let v3 = v3?;

        let faces;
        if plane_dist(self.vertices[v3].point) < 0.0 {
            faces = [
                self.create_face(v0, v1, v2),
                self.create_face(v3, v1, v0),
                self.create_face(v3, v2, v1),
                self.create_face(v3, v0, v2),
            ];
            for i in 0..3 {
                let j = (i + 1) % 3;
                let a = self.get_edge(faces[i + 1], 2);
                let b = self.get_edge(faces[0], j as i32);
                self.set_twin(a, b);
                let a = self.get_edge(faces[i + 1], 1);
                let b = self.get_edge(faces[j + 1], 0);
                self.set_twin(a, b);
            }
        } else {
            faces = [
                self.create_face(v0, v2, v1),
                self.create_face(v3, v0, v1),
                self.create_face(v3, v1, v2),
                self.create_face(v3, v2, v0),
            ];
            for i in 0..3 {
                let j = (i + 1) % 3;
                let a = self.get_edge(faces[i + 1], 2);
                let b = self.get_edge(faces[0], ((3 - i) % 3) as i32);
                self.set_twin(a, b);
                let a = self.get_edge(faces[i + 1], 0);
                let b = self.get_edge(faces[j + 1], 1);
                self.set_twin(a, b);
            }
        }
        self.faces.extend_from_slice(&faces);

        for i in 0..self.vertices.len() {
            if i != v0 && i != v1 && i != v2 && i != v3 {
                let mut max_distance = self.tolerance;
                let mut max_face = None;
                for j in 0..4 {
                    let d = self.dist(self.faces[j], i);
                    if d > max_distance {
                        max_distance = d;
                        max_face = Some(self.faces[j]);
                    }
                }
                if let Some(f) = max_face {
                    self.add_vertex_to_face(i, f);
                }
            }
        }
        Some(())
    }

    fn reindex_faces(&mut self) {
        let pool = &self.face_pool;
        self.faces.retain(|&f| pool[f].mark == Mark::Visible);
    }

    fn next_vertex_to_add(&self) -> Option<usize> {
        let first = self.assigned.head?;
        let eye_face = self.vertices[first].face?;
        let mut eye_vertex = None;
        let mut max_distance = 0.0;
        let mut v = self.face_pool[eye_face].outside;
        while let Some(cur) = v {
            let d = self.dist(eye_face, cur);
            if d > max_distance {
                max_distance = d;
                eye_vertex = Some(cur);
            }
            v = self.vertices[cur].next;
            if v.is_some_and(|n| self.vertices[n].face != Some(eye_face)) {
                break;
            }
        }
        eye_vertex
    }

    fn compute_horizon(
        &mut self,
        eye_point: V3,
        cross_edge: Option<usize>,
        face: usize,
        horizon: &mut Vec<usize>,
    ) {
        self.delete_face_vertices(face, None);
        self.face_pool[face].mark = Mark::Deleted;
        let (mut edge, cross_edge) = match cross_edge {
            None => {
                let e = self.get_edge(face, 0);
                (e, e)
            }
            Some(c) => (self.edges[c].next, c),
        };
        while let Some(twin) = self.edges[edge].twin {
            let opposite = self.edges[twin].face;
            if self.face_pool[opposite].mark == Mark::Visible {
                if self.face_pool[opposite].distance_to_point(eye_point) > self.tolerance {
                    self.compute_horizon(eye_point, Some(twin), opposite, horizon);
                } else {
                    horizon.push(edge);
                }
            }
            edge = self.edges[edge].next;
            if edge == cross_edge {
                break;
            }
        }
    }

    fn add_adjoining_face(&mut self, eye: usize, horizon_edge: usize) -> usize {
        let (tail, head) = (self.tail(horizon_edge), self.head(horizon_edge));
        let face = self.create_face(eye, tail, head);
        self.faces.push(face);
        let last = self.get_edge(face, -1);
        if let Some(twin) = self.edges[horizon_edge].twin {
            self.set_twin(last, twin);
        }
        self.get_edge(face, 0)
    }

    fn add_new_faces(&mut self, eye: usize, horizon: &[usize]) {
        self.new_faces.clear();
        let mut first_side: Option<usize> = None;
        let mut previous_side: Option<usize> = None;
        for &h in horizon {
            let side = self.add_adjoining_face(eye, h);
            match first_side {
                None => first_side = Some(side),
                Some(_) => {
                    let next = self.edges[side].next;
                    self.set_twin(next, previous_side.unwrap());
                }
            }
            self.new_faces.push(self.edges[side].face);
            previous_side = Some(side);
        }
        if let (Some(first), Some(prev)) = (first_side, previous_side) {
            let next = self.edges[first].next;
            self.set_twin(next, prev);
        }
    }

    fn add_vertex_to_hull(&mut self, eye: usize) {
        let mut horizon = Vec::new();
        self.unassigned = VertexList::default();
        let face = self.vertices[eye].face.unwrap();
        self.remove_vertex_from_face(eye, face);
        self.compute_horizon(self.vertices[eye].point, None, face, &mut horizon);
        self.add_new_faces(eye, &horizon);
        self.resolve_unassigned_points();
    }

    fn compute(&mut self) -> Option<()> {
        self.compute_initial_hull()?;
        while let Some(v) = self.next_vertex_to_add() {
            self.add_vertex_to_hull(v);
        }
        self.reindex_faces();
        self.assigned = VertexList::default();
        self.unassigned = VertexList::default();
        self.new_faces.clear();
        Some(())
    }
}

pub struct ConvexGeometry;

impl ConvexGeometry {
    /// three.js `ConvexGeometry`: the convex hull of `points`, unindexed, one
    /// triangle per hull face with the face normal on each corner.
    pub fn new(points: &[Vector3]) -> BufferGeometry {
        let pts: Vec<V3> = points.iter().map(|p| [p.x as f64, p.y as f64, p.z as f64]).collect();
        Self::from_f64(&pts)
    }

    /// [`new`](Self::new) from `f64` points, as JavaScript hands them over.
    pub fn from_f64(points: &[V3]) -> BufferGeometry {
        let mut hull = ConvexHull::new();
        hull.set_from_points(points);
        Self::from_hull(&hull)
    }

    pub fn from_hull(hull: &ConvexHull) -> BufferGeometry {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        for (face, corners) in hull.faces().zip(hull.face_vertex_indices()) {
            for v in corners {
                let p = hull.point(v);
                positions.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32]);
                normals.extend_from_slice(&[
                    face.normal[0] as f32,
                    face.normal[1] as f32,
                    face.normal[2] as f32,
                ]);
            }
        }
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_attribute("normal", BufferAttribute::new(normals, 3));
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_corners_give_twelve_outward_triangles() {
        let mut pts = Vec::new();
        for x in [-1.0, 1.0] {
            for y in [-1.0, 1.0] {
                for z in [-1.0, 1.0] {
                    pts.push([x, y, z]);
                }
            }
        }
        pts.push([0.1, 0.2, 0.3]); // inside
        let mut hull = ConvexHull::new();
        hull.set_from_points(&pts);
        assert_eq!(hull.faces().count(), 12);
        for f in hull.faces() {
            assert!((f.constant - 1.0).abs() < 1e-12, "outward unit distance");
        }
        assert!(hull.contains_point([0.0, 0.0, 0.0]));
        assert!(!hull.contains_point([2.0, 0.0, 0.0]));
        let t = hull.intersect_ray([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap();
        assert!((t - 4.0).abs() < 1e-12);
    }

    #[test]
    fn degenerate_input_gives_no_faces() {
        let pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let g = ConvexGeometry::from_f64(&pts);
        assert_eq!(g.get_attribute("position").unwrap().array.len() % 9, 0);
    }
}
