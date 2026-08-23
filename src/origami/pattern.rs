//! Planar crease pattern: hinges, boundary, faces, and speed-coefficient closure.

use super::vertex::{degree4_from_dirs, Degree4, VertexMode};
use super::{cross2, sub2, V2};
use std::collections::HashSet;

/// One edge of the net. Hinges rotate; boundary edges only close the sheet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    Hinge,
    Boundary,
}

#[derive(Debug, Clone, Copy)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
    pub kind: EdgeKind,
}

/// Straight-line drawing of a crease pattern on a polygonal sheet.
#[derive(Debug, Clone)]
pub struct CreasePattern {
    pub verts: Vec<V2>,
    pub edges: Vec<Edge>,
    /// Bounded faces, each a CCW loop of vertex indices.
    pub faces: Vec<Vec<usize>>,
    /// Edge index of each step of `faces[i][j] → faces[i][j+1]`.
    pub face_edges: Vec<Vec<usize>>,
}

impl CreasePattern {
    pub fn new(verts: Vec<V2>, edges: Vec<Edge>) -> Self {
        let (faces, face_edges) = extract_faces(&verts, &edges);
        Self {
            verts,
            edges,
            faces,
            face_edges,
        }
    }

    /// Single degree-4 vertex with sector angles `(α, β, π−α, π−β)` and a
    /// square-ish clipping of the four sectors — Figure 1 when both are π/2.
    pub fn cross(alpha: f64, beta: f64) -> Self {
        let o = [0.0, 0.0];
        let angles = [
            0.0,
            alpha,
            alpha + beta,
            alpha + beta + std::f64::consts::PI - alpha,
        ];
        let mut verts = vec![o];
        for &th in &angles {
            verts.push([th.cos(), th.sin()]);
        }
        // 0 = origin, 1..4 = arm tips CCW.
        let mut edges = Vec::new();
        for i in 1..=4 {
            edges.push(Edge {
                a: 0,
                b: i,
                kind: EdgeKind::Hinge,
            });
        }
        for i in 0..4 {
            edges.push(Edge {
                a: 1 + i,
                b: 1 + ((i + 1) % 4),
                kind: EdgeKind::Boundary,
            });
        }
        Self::new(verts, edges)
    }

    /// Rigid-foldable square twist: inner square plus four outer pairs.
    ///
    /// `alpha` is the twist (smallest sector at each inner vertex). `β = π/2`
    /// by construction, so Lemma 14 applies. The sheet is a rectangle that
    /// every spoke hits, so every crease ends on the boundary.
    pub fn square_twist(alpha: f64) -> Self {
        let inner = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let bbox = [-2.0, 3.0, -2.0, 3.0]; // xmin, xmax, ymin, ymax
        let d0 = [
            [(-alpha).cos(), (-alpha).sin()],
            [
                (1.5 * std::f64::consts::PI - alpha).cos(),
                (1.5 * std::f64::consts::PI - alpha).sin(),
            ],
        ];
        let mut verts: Vec<V2> = inner.to_vec();
        let mut outer_ids = Vec::new();
        for (k, &corner) in inner.iter().enumerate() {
            let th = k as f64 * std::f64::consts::FRAC_PI_2;
            let (s, ch) = th.sin_cos();
            let rot = |d: V2| -> V2 { [ch * d[0] - s * d[1], s * d[0] + ch * d[1]] };
            for d in d0 {
                let dir = rot(d);
                let hit = ray_hit_box(corner, dir, bbox);
                outer_ids.push(verts.len());
                verts.push(hit);
            }
        }
        let mut edges = Vec::new();
        for i in 0..4 {
            edges.push(Edge {
                a: i,
                b: (i + 1) % 4,
                kind: EdgeKind::Hinge,
            });
        }
        for k in 0..4 {
            edges.push(Edge {
                a: k,
                b: 4 + 2 * k,
                kind: EdgeKind::Hinge,
            });
            edges.push(Edge {
                a: k,
                b: 5 + 2 * k,
                kind: EdgeKind::Hinge,
            });
        }
        let mut ring = outer_ids;
        ring.sort_by(|&i, &j| {
            box_perimeter(verts[i], bbox)
                .partial_cmp(&box_perimeter(verts[j], bbox))
                .unwrap()
        });
        for i in 0..ring.len() {
            edges.push(Edge {
                a: ring[i],
                b: ring[(i + 1) % ring.len()],
                kind: EdgeKind::Boundary,
            });
        }
        Self::new(verts, edges)
    }

    /// Regular *n*-gonal twist: the square twist’s siblings (pentagon, hex, …).
    ///
    /// Each corner of a regular *n*-gon is a flat-foldable degree-4 vertex. `n = 4`
    /// is [`Self::square_twist`].
    pub fn polygon_twist(n: usize, alpha: f64) -> Self {
        let n = n.max(3);
        if n == 4 {
            return Self::square_twist(alpha);
        }
        let tau = std::f64::consts::TAU;
        let inner: Vec<V2> = (0..n)
            .map(|k| {
                let th = tau * k as f64 / n as f64;
                [th.cos(), th.sin()]
            })
            .collect();
        let bbox = [-3.2, 3.2, -3.2, 3.2];
        let mut verts = inner.clone();
        let mut outer_ids = Vec::new();
        for k in 0..n {
            let nxt = inner[(k + 1) % n];
            let prv = inner[(k + n - 1) % n];
            let u = sub2(nxt, inner[k]);
            let w = sub2(prv, inner[k]);
            let ang_u = u[1].atan2(u[0]);
            let ang_w = w[1].atan2(w[0]);
            let d0 = [ang_u - alpha, ang_w + std::f64::consts::PI - alpha];
            for ang in d0 {
                let dir = [ang.cos(), ang.sin()];
                let hit = ray_hit_box(inner[k], dir, bbox);
                outer_ids.push(verts.len());
                verts.push(hit);
            }
        }
        let mut edges = Vec::new();
        for i in 0..n {
            edges.push(Edge {
                a: i,
                b: (i + 1) % n,
                kind: EdgeKind::Hinge,
            });
        }
        for k in 0..n {
            edges.push(Edge {
                a: k,
                b: n + 2 * k,
                kind: EdgeKind::Hinge,
            });
            edges.push(Edge {
                a: k,
                b: n + 2 * k + 1,
                kind: EdgeKind::Hinge,
            });
        }
        let mut ring = outer_ids;
        ring.sort_by(|&i, &j| {
            box_perimeter(verts[i], bbox)
                .partial_cmp(&box_perimeter(verts[j], bbox))
                .unwrap()
        });
        for i in 0..ring.len() {
            edges.push(Edge {
                a: ring[i],
                b: ring[(i + 1) % ring.len()],
                kind: EdgeKind::Boundary,
            });
        }
        Self::new(verts, edges)
    }

    /// Unfolded Miura-ori: a zigzag grid whose interior vertices are
    /// flat-foldable degree-4.
    ///
    /// `cols` × `rows` cells. Odd rows are shifted so opposite sector angles at
    /// each interior vertex sum to π (Kawasaki).
    pub fn miura(cols: usize, rows: usize) -> Self {
        Self::miura_cell(cols, rows, 1.0, 0.65, 0.38)
    }

    pub fn miura_cell(cols: usize, rows: usize, lx: f64, ly: f64, dx: f64) -> Self {
        let cols = cols.max(1);
        let rows = rows.max(1);
        let nx = cols + 1;
        let ny = rows + 1;
        let mut verts = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            let off = if j % 2 == 1 { dx } else { 0.0 };
            for i in 0..nx {
                verts.push([i as f64 * lx + off, j as f64 * ly]);
            }
        }
        let at = |i: usize, j: usize| j * nx + i;
        let mut edges = Vec::new();
        for j in 0..ny {
            for i in 0..cols {
                let on_rim = j == 0 || j == rows;
                edges.push(Edge {
                    a: at(i, j),
                    b: at(i + 1, j),
                    kind: if on_rim {
                        EdgeKind::Boundary
                    } else {
                        EdgeKind::Hinge
                    },
                });
            }
        }
        for j in 0..rows {
            for i in 0..nx {
                let on_rim = i == 0 || i == cols;
                edges.push(Edge {
                    a: at(i, j),
                    b: at(i, j + 1),
                    kind: if on_rim {
                        EdgeKind::Boundary
                    } else {
                        EdgeKind::Hinge
                    },
                });
            }
        }
        Self::new(verts, edges)
    }

    pub fn hinge_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.edges
            .iter()
            .enumerate()
            .filter(|(_, e)| e.kind == EdgeKind::Hinge)
            .map(|(i, _)| i)
    }

    /// Interior degree-4 vertices (not on the paper boundary).
    pub fn interior_deg4(&self) -> Vec<usize> {
        let on_boundary = self.boundary_vertices();
        (0..self.verts.len())
            .filter(|&v| !on_boundary[v] && self.hinges_at(v).len() == 4)
            .collect()
    }

    fn boundary_vertices(&self) -> Vec<bool> {
        let mut b = vec![false; self.verts.len()];
        for e in &self.edges {
            if e.kind == EdgeKind::Boundary {
                b[e.a] = true;
                b[e.b] = true;
            }
        }
        b
    }

    pub fn hinges_at(&self, v: usize) -> Vec<usize> {
        self.edges
            .iter()
            .enumerate()
            .filter(|(_, e)| e.kind == EdgeKind::Hinge && (e.a == v || e.b == v))
            .map(|(i, _)| i)
            .collect()
    }

    /// Crease directions from `v`, sorted CCW, with the matching hinge index.
    pub fn ccw_hinges(&self, v: usize) -> Option<Vec<(usize, V2)>> {
        let mut d: Vec<(usize, V2, f64)> = self
            .hinges_at(v)
            .into_iter()
            .map(|ei| {
                let e = self.edges[ei];
                let w = if e.a == v { e.b } else { e.a };
                let dir = sub2(self.verts[w], self.verts[v]);
                let ang = dir[1].atan2(dir[0]);
                (ei, dir, ang)
            })
            .collect();
        if d.len() < 2 {
            return None;
        }
        d.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        Some(d.into_iter().map(|(ei, dir, _)| (ei, dir)).collect())
    }

    pub fn degree4_at(&self, v: usize) -> Option<(Degree4, [usize; 4])> {
        let h = self.ccw_hinges(v)?;
        if h.len() != 4 {
            return None;
        }
        let dirs = [h[0].1, h[1].1, h[2].1, h[3].1];
        let ids = [h[0].0, h[1].0, h[2].0, h[3].0];
        Some((degree4_from_dirs(dirs)?, ids))
    }

    /// Every interior degree-4 vertex satisfies Kawasaki.
    pub fn kawasaki_all(&self) -> bool {
        self.interior_deg4()
            .into_iter()
            .all(|v| self.degree4_at(v).is_some())
    }

    /// Faces whose entire boundary is hinges — the loops Corollary 11 constrains.
    pub fn interior_face_ids(&self) -> Vec<usize> {
        self.face_edges
            .iter()
            .enumerate()
            .filter(|(_, es)| es.iter().all(|&e| self.edges[e].kind == EdgeKind::Hinge))
            .map(|(i, _)| i)
            .collect()
    }
}

/// Mode at each pattern vertex. Entries for non-degree-4 vertices are ignored.
#[derive(Debug, Clone)]
pub struct Assignment {
    pub mode: Vec<VertexMode>,
}

impl Assignment {
    pub fn uniform(n_verts: usize, mode: VertexMode) -> Self {
        Self {
            mode: vec![mode; n_verts],
        }
    }
}

/// How far a candidate assignment misses loop closure (Corollary 11).
#[derive(Debug, Clone)]
pub struct ClosureReport {
    /// `max |∏ p_i − 1|` over interior faces. Zero is exact rigid foldability.
    pub max_err: f64,
    pub face_products: Vec<(usize, f64)>,
}

impl CreasePattern {
    /// Finite-precision rigid-foldability using all creases: each interior face
    /// must have speed-coefficient product in `[1−ε, 1+ε]`.
    pub fn check_rigid(&self, assign: &Assignment, _eps: f64) -> ClosureReport {
        let mut face_products = Vec::new();
        let mut max_err: f64 = 0.0;
        for fi in self.interior_face_ids() {
            match self.face_speed_product(fi, assign) {
                Some(p) => {
                    max_err = max_err.max((p - 1.0).abs());
                    face_products.push((fi, p));
                }
                None => {
                    max_err = f64::INFINITY;
                    face_products.push((fi, f64::NAN));
                }
            }
        }
        ClosureReport {
            max_err,
            face_products,
        }
    }

    pub fn is_rigid(&self, assign: &Assignment, eps: f64) -> bool {
        let r = self.check_rigid(assign, eps);
        r.max_err.is_finite() && r.max_err <= eps
    }

    /// Enumerate mode assignments on interior degree-4 vertices.
    ///
    /// This is exponential in the number of those vertices. The paper's point is
    /// that you cannot do better in general; we refuse past 12 vertices rather
    /// than pretend this is a solver for large nets.
    pub fn find_assignments(&self, eps: f64) -> Vec<Assignment> {
        let verts = self.interior_deg4();
        if verts.is_empty() || verts.len() > 12 {
            return Vec::new();
        }
        let k = verts.len();
        let n = 1usize << k;
        let mut out = Vec::new();
        for bits in 0..n {
            let mut mode = vec![VertexMode::A; self.verts.len()];
            for (i, &v) in verts.iter().enumerate() {
                mode[v] = if (bits >> i) & 1 == 1 {
                    VertexMode::B
                } else {
                    VertexMode::A
                };
            }
            let assign = Assignment { mode };
            if self.is_rigid(&assign, eps) {
                if let Some(drive) = self.drive_hinge(&assign) {
                    if self.propagate(&assign, drive, 1.0).is_some() {
                        out.push(assign);
                    }
                }
            }
        }
        out
    }

    /// A hinge that actually folds under this assignment — not a Figure 1
    /// unused crease, which cannot be a drive.
    pub fn drive_hinge(&self, assign: &Assignment) -> Option<usize> {
        for v in self.interior_deg4() {
            let (d4, ids) = self.degree4_at(v)?;
            let mode = *assign.mode.get(v).unwrap_or(&VertexMode::A);
            let t = d4.tangents(mode, 1.0);
            for (k, &id) in ids.iter().enumerate() {
                if t[k].abs() > 1e-12 {
                    return Some(id);
                }
            }
        }
        self.hinge_indices().next()
    }

    /// Walk out from one interior vertex, choosing the unique mode at each
    /// neighbour that matches already-known crease speeds.
    ///
    /// This is how a tessellation (Miura, a row of twists) is folded without
    /// enumerating `2^V` assignments — the NP-hard search the paper rules out.
    pub fn assignment_from_seed(&self, seed: usize, seed_mode: VertexMode) -> Option<Assignment> {
        let interiors = self.interior_deg4();
        if !interiors.contains(&seed) {
            return None;
        }
        let mut mode = vec![VertexMode::A; self.verts.len()];
        let mut have = vec![false; self.verts.len()];
        let mut t = vec![0.0; self.edges.len()];
        let mut have_t = vec![false; self.edges.len()];
        if !fill_vertex(
            self,
            seed,
            seed_mode,
            &mut mode,
            &mut have,
            &mut t,
            &mut have_t,
        ) {
            return None;
        }
        let mut guard = 0;
        loop {
            guard += 1;
            if guard > interiors.len() * 4 {
                break;
            }
            let mut progress = false;
            for &v in &interiors {
                if have[v] {
                    continue;
                }
                let Some((_, ids)) = self.degree4_at(v) else {
                    continue;
                };
                if !ids.iter().any(|&id| have_t[id]) {
                    continue;
                }
                let mut fits: Vec<VertexMode> = Vec::new();
                for cand in [VertexMode::A, VertexMode::B] {
                    let mut m = mode.clone();
                    let mut h = have.clone();
                    let mut tt = t.clone();
                    let mut ht = have_t.clone();
                    if fill_vertex(self, v, cand, &mut m, &mut h, &mut tt, &mut ht) {
                        fits.push(cand);
                    }
                }
                match fits.as_slice() {
                    [only] => {
                        fill_vertex(self, v, *only, &mut mode, &mut have, &mut t, &mut have_t);
                        progress = true;
                    }
                    [] => return None,
                    _ => {}
                }
            }
            if !progress {
                break;
            }
        }
        if interiors.iter().any(|&v| !have[v]) {
            return None;
        }
        Some(Assignment { mode })
    }

    /// Crease pattern as an SVG net. Pass fold tangents to colour valley
    /// (solid red) vs mountain (dashed blue).
    pub fn to_svg(&self, tangents: Option<&[f64]>) -> String {
        let mut min = [f64::MAX, f64::MAX];
        let mut max = [f64::MIN, f64::MIN];
        for v in &self.verts {
            min[0] = min[0].min(v[0]);
            min[1] = min[1].min(v[1]);
            max[0] = max[0].max(v[0]);
            max[1] = max[1].max(v[1]);
        }
        let pad = 0.15 * (max[0] - min[0]).max(max[1] - min[1]).max(1.0);
        let w = (max[0] - min[0] + 2.0 * pad).max(1e-6);
        let h = (max[1] - min[1] + 2.0 * pad).max(1e-6);
        let sw = 0.012 * w.min(h);
        let mut out = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{:.4} {:.4} {:.4} {:.4}\" \
             width=\"800\" height=\"600\" fill=\"none\">\n  \
             <rect x=\"{:.4}\" y=\"{:.4}\" width=\"{:.4}\" height=\"{:.4}\" fill=\"#f4efe6\"/>\n",
            min[0] - pad,
            min[1] - pad,
            w,
            h,
            min[0] - pad,
            min[1] - pad,
            w,
            h
        );
        for (ei, e) in self.edges.iter().enumerate() {
            let a = self.verts[e.a];
            let b = self.verts[e.b];
            let (stroke, width, dash): (&str, f64, String) = match e.kind {
                EdgeKind::Boundary => ("#6b6358", sw * 1.2, "none".into()),
                EdgeKind::Hinge => {
                    let ti = tangents.and_then(|t| t.get(ei)).copied().unwrap_or(0.0);
                    if ti > 1e-8 {
                        ("#c23b22", sw, "none".into())
                    } else if ti < -1e-8 {
                        ("#2a5ea8", sw, format!("{:.3} {:.3}", sw * 4.0, sw * 2.5))
                    } else {
                        (
                            "#8a8478",
                            sw * 0.7,
                            format!("{:.3} {:.3}", sw * 1.5, sw * 2.0),
                        )
                    }
                }
            };
            out.push_str(&format!(
                "  <line x1=\"{:.4}\" y1=\"{:.4}\" x2=\"{:.4}\" y2=\"{:.4}\" \
                 stroke=\"{stroke}\" stroke-width=\"{width:.4}\" stroke-dasharray=\"{dash}\" \
                 stroke-linecap=\"round\"/>\n",
                a[0], a[1], b[0], b[1]
            ));
        }
        out.push_str("</svg>\n");
        out
    }

    /// `∏ p(c_{i+1}, c_i)` around an interior face.
    fn face_speed_product(&self, fi: usize, assign: &Assignment) -> Option<f64> {
        let vs = &self.faces[fi];
        let es = &self.face_edges[fi];
        let n = vs.len();
        if n < 3 {
            return None;
        }
        let mut prod = 1.0;
        for i in 0..n {
            let v = vs[(i + 1) % n];
            let e_in = es[i];
            let e_out = es[(i + 1) % n];
            let (d4, ids) = self.degree4_at(v)?;
            let mode = *assign.mode.get(v).unwrap_or(&VertexMode::A);
            let tangents = d4.tangents(mode, 1.0);
            let t_in = tangent_on(&ids, &tangents, e_in)?;
            let t_out = tangent_on(&ids, &tangents, e_out)?;
            if t_in.abs() < 1e-15 {
                return None;
            }
            prod *= t_out / t_in;
        }
        Some(prod)
    }
}

fn tangent_on(ids: &[usize; 4], t: &[f64; 4], edge: usize) -> Option<f64> {
    ids.iter().position(|&e| e == edge).map(|i| t[i])
}

fn fill_vertex(
    cp: &CreasePattern,
    v: usize,
    m: VertexMode,
    mode: &mut [VertexMode],
    have: &mut [bool],
    t: &mut [f64],
    have_t: &mut [bool],
) -> bool {
    let Some((d4, ids)) = cp.degree4_at(v) else {
        return false;
    };
    let mut param: Option<f64> = None;
    for (k, &id) in ids.iter().enumerate() {
        if have_t[id] {
            if let Some(p) = d4.drive_from(m, k, t[id]) {
                if let Some(q) = param {
                    if (p - q).abs() > 1e-8 * (1.0 + q.abs()) {
                        return false;
                    }
                } else {
                    param = Some(p);
                }
            } else if t[id].abs() > 1e-10 {
                return false;
            }
        }
    }
    let p = param.unwrap_or(1.0);
    let tangents = d4.tangents(m, p);
    for (k, &id) in ids.iter().enumerate() {
        if have_t[id] {
            if (t[id] - tangents[k]).abs() > 1e-8 * (1.0 + tangents[k].abs()) {
                return false;
            }
        } else {
            t[id] = tangents[k];
            have_t[id] = true;
        }
    }
    mode[v] = m;
    have[v] = true;
    true
}

fn extract_faces(verts: &[V2], edges: &[Edge]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n = verts.len();
    // For each vertex, incident edges sorted CCW by the far endpoint.
    let mut adj: Vec<Vec<(usize, usize)>> = vec![vec![]; n];
    for (ei, e) in edges.iter().enumerate() {
        adj[e.a].push((e.b, ei));
        adj[e.b].push((e.a, ei));
    }
    for v in 0..n {
        adj[v].sort_by(|x, y| {
            let ax = sub2(verts[x.0], verts[v]);
            let ay = sub2(verts[y.0], verts[v]);
            ax[1].atan2(ax[0]).partial_cmp(&ay[1].atan2(ay[0])).unwrap()
        });
    }

    // next(u → v) = neighbor of v immediately CCW from u.
    let next_half = |u: usize, v: usize| -> Option<(usize, usize)> {
        let nbrs = &adj[v];
        let idx = nbrs.iter().position(|&(w, _)| w == u)?;
        let (w, ei) = nbrs[(idx + nbrs.len() - 1) % nbrs.len()];
        Some((w, ei))
    };

    let mut used: HashSet<(usize, usize)> = HashSet::new();
    let mut faces = Vec::new();
    let mut face_edges = Vec::new();

    for (ei, e) in edges.iter().enumerate() {
        for (a, b) in [(e.a, e.b), (e.b, e.a)] {
            if !used.insert((a, b)) {
                continue;
            }
            let mut loop_v = vec![a, b];
            let mut loop_e = vec![ei];
            let mut prev = a;
            let mut cur = b;
            let start_a = a;
            let mut guard = 0;
            loop {
                guard += 1;
                if guard > edges.len() * 4 {
                    break;
                }
                let Some((nxt, nei)) = next_half(prev, cur) else {
                    break;
                };
                used.insert((cur, nxt));
                loop_e.push(nei);
                if nxt == start_a {
                    break;
                }
                loop_v.push(nxt);
                prev = cur;
                cur = nxt;
            }
            if loop_v.len() < 3 {
                continue;
            }
            let mut uniq = loop_v.clone();
            uniq.sort_unstable();
            uniq.dedup();
            if uniq.len() != loop_v.len() {
                continue;
            }
            let area = signed_area(verts, &loop_v);
            if area > 1e-12 {
                faces.push(loop_v);
                face_edges.push(loop_e);
            }
        }
    }
    (faces, face_edges)
}

fn signed_area(verts: &[V2], loop_v: &[usize]) -> f64 {
    let mut a = 0.0;
    let n = loop_v.len();
    for i in 0..n {
        let p = verts[loop_v[i]];
        let q = verts[loop_v[(i + 1) % n]];
        a += cross2(p, q);
    }
    0.5 * a
}

fn ray_hit_box(p: V2, d: V2, bbox: [f64; 4]) -> V2 {
    let [xmin, xmax, ymin, ymax] = bbox;
    let mut t = f64::INFINITY;
    if d[0] > 1e-14 {
        t = t.min((xmax - p[0]) / d[0]);
    } else if d[0] < -1e-14 {
        t = t.min((xmin - p[0]) / d[0]);
    }
    if d[1] > 1e-14 {
        t = t.min((ymax - p[1]) / d[1]);
    } else if d[1] < -1e-14 {
        t = t.min((ymin - p[1]) / d[1]);
    }
    [p[0] + t * d[0], p[1] + t * d[1]]
}

/// Arc length along the box boundary, starting at the bottom-left corner, CCW.
fn box_perimeter(p: V2, bbox: [f64; 4]) -> f64 {
    let [xmin, xmax, ymin, ymax] = bbox;
    let w = xmax - xmin;
    let h = ymax - ymin;
    let e = 1e-6;
    if (p[1] - ymin).abs() <= e {
        p[0] - xmin
    } else if (p[0] - xmax).abs() <= e {
        w + (p[1] - ymin)
    } else if (p[1] - ymax).abs() <= e {
        w + h + (xmax - p[0])
    } else {
        w + h + w + (ymax - p[1])
    }
}
