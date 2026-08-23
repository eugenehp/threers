//! Place panels in 3D by walking the dual and rotating about hinges.

use super::pattern::{Assignment, CreasePattern, EdgeKind};
use super::vertex::VertexMode;
use super::{add3, cross3, dot3, sub3, V3};
use crate::core::{BufferAttribute, BufferGeometry};

/// Folded image of a crease pattern: one 3D position per net vertex.
#[derive(Debug, Clone)]
pub struct FoldedState {
    pub verts: Vec<V3>,
    pub faces: Vec<Vec<usize>>,
}

impl FoldedState {
    pub fn to_geometry(&self) -> BufferGeometry {
        paper_geometry(&self.verts, &self.faces, |_| true, PAPER_THICK)
    }

    /// Checkerboard split so adjacent panels can take different materials.
    pub fn to_geometry_checker(&self) -> (BufferGeometry, BufferGeometry) {
        (
            paper_geometry(&self.verts, &self.faces, |i| i % 2 == 0, PAPER_THICK),
            paper_geometry(&self.verts, &self.faces, |i| i % 2 == 1, PAPER_THICK),
        )
    }

    /// Every unique undirected edge of the mesh (for crease overlays).
    pub fn all_edges(&self) -> Vec<[usize; 2]> {
        let mut seen = std::collections::BTreeSet::new();
        let mut out = Vec::new();
        for face in &self.faces {
            let ids = clean_ring(&self.verts, face);
            let n = ids.len();
            if n < 2 {
                continue;
            }
            for i in 0..n {
                let a = ids[i];
                let b = ids[(i + 1) % n];
                let key = if a < b { (a, b) } else { (b, a) };
                if seen.insert(key) {
                    out.push([key.0, key.1]);
                }
            }
        }
        out
    }

    /// Line list of selected net edges (coplanar with paper — may soft-z-fight).
    pub fn line_geometry(&self, segs: &[[usize; 2]]) -> BufferGeometry {
        // No normal offset — offsetting along averaged normals pushes creases
        // through overlapping folded layers (X-ray lines).
        let mut positions = Vec::with_capacity(segs.len() * 6);
        for &[a, b] in segs {
            if a >= self.verts.len() || b >= self.verts.len() || a == b {
                continue;
            }
            let pa = self.verts[a];
            let pb = self.verts[b];
            positions.extend_from_slice(&[
                pa[0] as f32,
                pa[1] as f32,
                pa[2] as f32,
                pb[0] as f32,
                pb[1] as f32,
                pb[2] as f32,
            ]);
        }
        let mut geom = BufferGeometry::new();
        if !positions.is_empty() {
            geom.set_attribute("position", BufferAttribute::new(positions, 3));
        }
        geom
    }
}

impl CreasePattern {
    /// Propagate half-angle tangents from one driving hinge.
    ///
    /// Boundary edges stay 0. Unused hinges in a mode (the Figure 1 case) get 0
    /// as well. Returns `None` when two vertices disagree on a crease's speed.
    pub fn propagate(
        &self,
        assign: &Assignment,
        drive_edge: usize,
        drive_t: f64,
    ) -> Option<Vec<f64>> {
        let mut t = vec![0.0; self.edges.len()];
        let mut known = vec![false; self.edges.len()];
        if self.edges[drive_edge].kind != EdgeKind::Hinge {
            return None;
        }
        t[drive_edge] = drive_t;
        known[drive_edge] = true;
        let mut queue = vec![drive_edge];
        while let Some(ei) = queue.pop() {
            for v in [self.edges[ei].a, self.edges[ei].b] {
                let Some((d4, ids)) = self.degree4_at(v) else {
                    continue;
                };
                let mode = *assign.mode.get(v).unwrap_or(&VertexMode::A);
                let mut param = None;
                for (k, &id) in ids.iter().enumerate() {
                    if known[id] {
                        if let Some(p) = d4.drive_from(mode, k, t[id]) {
                            match param {
                                None => param = Some(p),
                                Some(q) if (p - q).abs() <= 1e-8 * (1.0 + q.abs()) => {}
                                Some(_) => return None,
                            }
                        } else if t[id].abs() > 1e-10 {
                            // Crease should be unused in this mode but isn't.
                            return None;
                        }
                    }
                }
                let p = param.unwrap_or(0.0);
                let tangents = d4.tangents(mode, p);
                for (k, &id) in ids.iter().enumerate() {
                    if known[id] {
                        if (t[id] - tangents[k]).abs() > 1e-8 * (1.0 + tangents[k].abs()) {
                            return None;
                        }
                    } else {
                        t[id] = tangents[k];
                        known[id] = true;
                        queue.push(id);
                    }
                }
            }
        }
        Some(t)
    }

    /// Rigid motion from the flat net, driven by `tan(ρ/2)` on `drive_edge`.
    pub fn fold(
        &self,
        assign: &Assignment,
        drive_edge: usize,
        drive_t: f64,
    ) -> Option<FoldedState> {
        let tangents = self.propagate(assign, drive_edge, drive_t)?;
        let rho: Vec<f64> = tangents.iter().map(|&ti| 2.0 * ti.atan()).collect();
        self.fold_with_angles(&rho)
    }

    /// Fold a tessellation from one seed vertex and a mode, without enumerating
    /// every assignment.
    pub fn fold_seed(
        &self,
        seed: usize,
        mode: VertexMode,
        drive_t: f64,
    ) -> Option<(Assignment, Vec<f64>, FoldedState)> {
        let assign = self.assignment_from_seed(seed, mode)?;
        let drive = self.drive_hinge(&assign)?;
        let tangents = self.propagate(&assign, drive, drive_t)?;
        let folded = self.fold(&assign, drive, drive_t)?;
        Some((assign, tangents, folded))
    }

    /// Try uniform modes, then a small exhaustive search, then a seed walk.
    pub fn fold_any(&self, drive_t: f64) -> Option<(Assignment, Vec<f64>, FoldedState)> {
        let n = self.verts.len();
        for mode in [VertexMode::A, VertexMode::B] {
            let assign = Assignment::uniform(n, mode);
            if let Some(drive) = self.drive_hinge(&assign) {
                if let Some(tangents) = self.propagate(&assign, drive, drive_t) {
                    if let Some(folded) = self.fold(&assign, drive, drive_t) {
                        return Some((assign, tangents, folded));
                    }
                }
            }
        }
        for assign in self.find_assignments(1e-6) {
            if let Some(drive) = self.drive_hinge(&assign) {
                if let Some(tangents) = self.propagate(&assign, drive, drive_t) {
                    if let Some(folded) = self.fold(&assign, drive, drive_t) {
                        return Some((assign, tangents, folded));
                    }
                }
            }
        }
        for v in self.interior_deg4() {
            for mode in [VertexMode::A, VertexMode::B] {
                if let Some(out) = self.fold_seed(v, mode, drive_t) {
                    return Some(out);
                }
            }
        }
        None
    }

    pub fn fold_with_angles(&self, rho: &[f64]) -> Option<FoldedState> {
        let n_f = self.faces.len();
        if n_f == 0 {
            return None;
        }
        let adj = self.face_adjacency();
        let mut placed = vec![false; n_f];
        let mut xf = vec![Rigid::identity(); n_f];
        let mut q = vec![0usize];
        placed[0] = true;
        while let Some(f) = q.pop() {
            for &(g, ei, a, b) in &adj[f] {
                if placed[g] {
                    continue;
                }
                let angle = if ei < rho.len() { rho[ei] } else { 0.0 };
                // Axis is the already-placed image of the hinge, oriented so f
                // lies to the left. The neighbour is rotated off the common plane.
                let a3 = xf[f].apply_xy(self.verts[a]);
                let b3 = xf[f].apply_xy(self.verts[b]);
                let rot = Rigid::rotate_about(a3, sub3(b3, a3), -angle);
                xf[g] = rot.compose(xf[f]);
                placed[g] = true;
                q.push(g);
            }
        }
        if placed.iter().any(|&p| !p) {
            return None;
        }
        let mut verts = vec![[0.0; 3]; self.verts.len()];
        let mut done = vec![false; self.verts.len()];
        for (fi, face) in self.faces.iter().enumerate() {
            for &v in face {
                let p = xf[fi].apply_xy(self.verts[v]);
                if done[v] {
                    let d = sub3(p, verts[v]);
                    if dot3(d, d).sqrt() > 1e-6 {
                        return None;
                    }
                } else {
                    verts[v] = p;
                    done[v] = true;
                }
            }
        }
        Some(FoldedState {
            verts,
            faces: self.faces.clone(),
        })
    }

    /// Dual: for each face, (neighbour, hinge edge, hinge endpoints in the
    /// current face's CCW order).
    fn face_adjacency(&self) -> Vec<Vec<(usize, usize, usize, usize)>> {
        let mut adj = vec![vec![]; self.faces.len()];
        for (f, edges) in self.face_edges.iter().enumerate() {
            for (j, &ei) in edges.iter().enumerate() {
                if self.edges[ei].kind != EdgeKind::Hinge {
                    continue;
                }
                let a = self.faces[f][j];
                let b = self.faces[f][(j + 1) % self.faces[f].len()];
                for (g, ge) in self.face_edges.iter().enumerate() {
                    if g == f {
                        continue;
                    }
                    if ge.contains(&ei) {
                        adj[f].push((g, ei, a, b));
                        break;
                    }
                }
            }
        }
        adj
    }
}

#[derive(Clone, Copy)]
struct Rigid {
    r: [[f64; 3]; 3],
    t: V3,
}

impl Rigid {
    fn identity() -> Self {
        Self {
            r: super::vertex::mat_ident(),
            t: [0.0, 0.0, 0.0],
        }
    }

    fn apply(&self, p: V3) -> V3 {
        [
            self.r[0][0] * p[0] + self.r[0][1] * p[1] + self.r[0][2] * p[2] + self.t[0],
            self.r[1][0] * p[0] + self.r[1][1] * p[1] + self.r[1][2] * p[2] + self.t[1],
            self.r[2][0] * p[0] + self.r[2][1] * p[1] + self.r[2][2] * p[2] + self.t[2],
        ]
    }

    fn apply_xy(&self, p: super::V2) -> V3 {
        self.apply([p[0], p[1], 0.0])
    }

    fn compose(self, other: Self) -> Self {
        // self ∘ other
        let r = super::vertex::matmul(self.r, other.r);
        let t = self.apply(other.t);
        Self { r, t }
    }

    fn rotate_about(origin: V3, axis: V3, angle: f64) -> Self {
        let n2 = dot3(axis, axis);
        if n2 < 1e-30 || angle.abs() < 1e-18 {
            return Self::identity();
        }
        let k = [
            axis[0] / n2.sqrt(),
            axis[1] / n2.sqrt(),
            axis[2] / n2.sqrt(),
        ];
        let (s, c) = angle.sin_cos();
        let ic = 1.0 - c;
        // Rodrigues, rows.
        let r = [
            [
                c + k[0] * k[0] * ic,
                k[0] * k[1] * ic - k[2] * s,
                k[0] * k[2] * ic + k[1] * s,
            ],
            [
                k[1] * k[0] * ic + k[2] * s,
                c + k[1] * k[1] * ic,
                k[1] * k[2] * ic - k[0] * s,
            ],
            [
                k[2] * k[0] * ic - k[1] * s,
                k[2] * k[1] * ic + k[0] * s,
                c + k[2] * k[2] * ic,
            ],
        ];
        let t = sub3(
            origin,
            [
                r[0][0] * origin[0] + r[0][1] * origin[1] + r[0][2] * origin[2],
                r[1][0] * origin[0] + r[1][1] * origin[1] + r[1][2] * origin[2],
                r[2][0] * origin[0] + r[2][1] * origin[1] + r[2][2] * origin[2],
            ],
        );
        Self { r, t }
    }
}

const PAPER_THICK: f64 = 0.014;

fn scale3(v: V3, s: f64) -> V3 {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn paper_geometry(
    verts: &[V3],
    faces: &[Vec<usize>],
    take: impl Fn(usize) -> bool,
    thick: f64,
) -> BufferGeometry {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    let _ = thick;

    for (fi, face) in faces.iter().enumerate() {
        if !take(fi) {
            continue;
        }
        let ring = clean_ring(verts, face);
        if ring.len() < 3 {
            continue;
        }
        let n = face_normal(verts, &ring);
        let tris = triangulate(verts, &ring, n);
        if tris.is_empty() {
            continue;
        }
        push_shell(
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut indices,
            verts,
            &tris,
            n,
            0.0,
            false,
        );
    }

    let mut geom = BufferGeometry::new();
    if positions.is_empty() {
        return geom;
    }
    geom.set_attribute("position", BufferAttribute::new(positions, 3));
    geom.set_attribute("normal", BufferAttribute::new(normals, 3));
    geom.set_attribute("uv", BufferAttribute::new(uvs, 2));
    geom.set_index(indices);
    geom
}

#[allow(clippy::too_many_arguments)]
fn push_shell(
    positions: &mut Vec<f32>,
    normals: &mut Vec<f32>,
    uvs: &mut Vec<f32>,
    indices: &mut Vec<u32>,
    verts: &[V3],
    tris: &[[usize; 3]],
    n: V3,
    half: f64,
    flip: bool,
) {
    let off = scale3(n, half);
    let sn = if flip { scale3(n, -1.0) } else { n };
    for t in tris {
        let base = (positions.len() / 3) as u32;
        let order = if flip { [t[0], t[2], t[1]] } else { *t };
        for (k, &vi) in order.iter().enumerate() {
            let p = add3(verts[vi], off);
            positions.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32]);
            normals.extend_from_slice(&[sn[0] as f32, sn[1] as f32, sn[2] as f32]);
            let u = if k == 0 { 0.0 } else { 1.0 };
            let v = if k == 2 { 1.0 } else { 0.0 };
            uvs.extend_from_slice(&[u, v]);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
}

fn triangulate(verts: &[V3], ring: &[usize], n: V3) -> Vec<[usize; 3]> {
    if ring.len() < 3 {
        return Vec::new();
    }
    if ring.len() == 3 {
        return if tri_ok(verts, ring[0], ring[1], ring[2], n) {
            vec![[ring[0], ring[1], ring[2]]]
        } else if tri_ok(verts, ring[0], ring[2], ring[1], n) {
            vec![[ring[0], ring[2], ring[1]]]
        } else {
            Vec::new()
        };
    }

    let (u, v) = plane_basis(n);
    let origin = verts[ring[0]];
    let pts: Vec<[f64; 2]> = ring
        .iter()
        .map(|&i| {
            let d = sub3(verts[i], origin);
            [dot3(d, u), dot3(d, v)]
        })
        .collect();

    let mut idx: Vec<usize> = (0..ring.len()).collect();
    // Ensure CCW in the (u,v) frame so ears match outward normal n.
    if signed_area2(&pts, &idx) < 0.0 {
        idx.reverse();
    }

    let mut tris = Vec::new();
    let mut guard = idx.len() * idx.len() + 2;
    while idx.len() > 3 && guard > 0 {
        guard -= 1;
        let m = idx.len();
        let mut found = false;
        for i in 0..m {
            let i0 = idx[(i + m - 1) % m];
            let i1 = idx[i];
            let i2 = idx[(i + 1) % m];
            if !is_convex2(pts[i0], pts[i1], pts[i2]) {
                continue;
            }
            let mut inside = false;
            for &j in &idx {
                if j == i0 || j == i1 || j == i2 {
                    continue;
                }
                if point_in_tri2(pts[j], pts[i0], pts[i1], pts[i2]) {
                    inside = true;
                    break;
                }
            }
            if inside {
                continue;
            }
            let (a, b, c) = (ring[i0], ring[i1], ring[i2]);
            // Non-planar panels can make a 3D triangle disagree with the
            // Newell normal — still emit it, just wind to match n.
            if tri_ok(verts, a, b, c, n) {
                tris.push([a, b, c]);
            } else {
                tris.push([a, c, b]);
            }
            idx.remove(i);
            found = true;
            break;
        }
        if !found {
            break;
        }
    }
    if idx.len() == 3 {
        let (a, b, c) = (ring[idx[0]], ring[idx[1]], ring[idx[2]]);
        if tri_ok(verts, a, b, c, n) {
            tris.push([a, b, c]);
        } else {
            tris.push([a, c, b]);
        }
    }
    tris
}

fn plane_basis(n: V3) -> (V3, V3) {
    let a = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let mut u = cross3(n, a);
    let ul = dot3(u, u).sqrt();
    u = if ul > 1e-12 {
        scale3(u, 1.0 / ul)
    } else {
        [1.0, 0.0, 0.0]
    };
    let v = cross3(n, u);
    (u, v)
}

fn signed_area2(pts: &[[f64; 2]], idx: &[usize]) -> f64 {
    let mut a = 0.0;
    for i in 0..idx.len() {
        let p = pts[idx[i]];
        let q = pts[idx[(i + 1) % idx.len()]];
        a += p[0] * q[1] - q[0] * p[1];
    }
    a * 0.5
}

fn is_convex2(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 1e-14
}

fn point_in_tri2(p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    let s1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    let s2 = (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0]);
    let s3 = (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0]);
    (s1 >= -1e-12 && s2 >= -1e-12 && s3 >= -1e-12)
        || (s1 <= 1e-12 && s2 <= 1e-12 && s3 <= 1e-12)
}

fn tri_ok(verts: &[V3], a: usize, b: usize, c: usize, n: V3) -> bool {
    let e1 = sub3(verts[b], verts[a]);
    let e2 = sub3(verts[c], verts[a]);
    let tn = cross3(e1, e2);
    let area2 = dot3(tn, tn);
    if area2 < 1e-16 {
        return false;
    }
    dot3(tn, n) > 0.0
}

fn clean_ring(verts: &[V3], face: &[usize]) -> Vec<usize> {
    let mut out = Vec::new();
    for &i in face {
        if i >= verts.len() {
            continue;
        }
        if let Some(&prev) = out.last() {
            if prev == i || nearly_same(verts[prev], verts[i]) {
                continue;
            }
        }
        out.push(i);
    }
    if out.len() >= 2
        && (out[0] == *out.last().unwrap() || nearly_same(verts[out[0]], verts[*out.last().unwrap()])) {
            out.pop();
        }
    out
}

fn nearly_same(a: V3, b: V3) -> bool {
    let d = sub3(a, b);
    dot3(d, d) < 1e-16
}

fn face_normal(verts: &[V3], face: &[usize]) -> V3 {
    // Newell — stable for non-planar / non-convex rings.
    let mut n = [0.0; 3];
    let m = face.len();
    for i in 0..m {
        let cur = verts[face[i]];
        let nxt = verts[face[(i + 1) % m]];
        n[0] += (cur[1] - nxt[1]) * (cur[2] + nxt[2]);
        n[1] += (cur[2] - nxt[2]) * (cur[0] + nxt[0]);
        n[2] += (cur[0] - nxt[0]) * (cur[1] + nxt[1]);
    }
    let len = dot3(n, n).sqrt();
    if len > 1e-15 {
        scale3(n, 1.0 / len)
    } else {
        [0.0, 0.0, 1.0]
    }
}
