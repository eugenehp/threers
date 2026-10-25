//! Mesh simplification by quadric edge collapse, that cannot tear a closed
//! mesh.
//!
//! The method is Garland and Heckbert's (1997): every vertex carries the sum
//! of the squared distances to the planes of its faces, as a quadric, and the
//! edge whose two ends can merge at the least added distance goes first. On
//! flat regions that error is zero, so a contoured box comes back down to its
//! dozen triangles; on curved ones it stops at the tolerance asked for.
//!
//! What makes a simplifier unsafe for printing is not the error — it is
//! topology. Collapsing an edge whose ends share a third neighbour pinches the
//! surface into a non-manifold fin, and a later pass that tries to repair the
//! fin is what opens the holes. So a collapse is only made when the *link
//! condition* holds (the two ends share exactly the two neighbours that sit
//! across the edge), which keeps a closed 2-manifold a closed 2-manifold, and
//! never when it would turn a surviving triangle over. Nothing has to be
//! checked or redone afterwards.

use crate::core::{BufferAttribute, BufferGeometry};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// How far to simplify.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimplifyOptions {
    /// Stop once this many triangles are left. 0 lets `max_error` alone decide.
    pub target_triangles: usize,
    /// How far, in world units, a merged vertex may sit from the planes of the
    /// original faces it now stands for — as their root-mean-square distance,
    /// weighted by area. An RMS and not a maximum: one small face among many
    /// can end up further off than this, which is what lets a curved surface
    /// simplify at all. On flats it is zero either way.
    pub max_error: f32,
}

impl Default for SimplifyOptions {
    /// Down to whatever stays within a ten-thousandth of a unit — flat regions
    /// collapse completely, curved ones barely at all.
    fn default() -> Self {
        Self {
            target_triangles: 0,
            max_error: 1e-4,
        }
    }
}

/// What a simplification did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SimplifyReport {
    pub triangles_before: usize,
    pub triangles_after: usize,
    /// Edges merged.
    pub collapses: usize,
    /// Collapses refused because they would have pinched the surface or turned
    /// a triangle over — each one a tear or a fold a less careful simplifier
    /// would have made.
    pub refused: usize,
}

/// A symmetric 4×4 matrix: `[a², ab, ac, ad, b², bc, bd, c², cd, d²]`.
#[derive(Clone, Copy, Default)]
struct Quadric([f64; 10]);

impl Quadric {
    fn plane(n: [f64; 3], d: f64, weight: f64) -> Quadric {
        let [a, b, c] = n;
        Quadric(
            [
                a * a,
                a * b,
                a * c,
                a * d,
                b * b,
                b * c,
                b * d,
                c * c,
                c * d,
                d * d,
            ]
            .map(|v| v * weight),
        )
    }

    fn add(&mut self, o: &Quadric) {
        for (a, b) in self.0.iter_mut().zip(o.0) {
            *a += b;
        }
    }

    fn sum(a: &Quadric, b: &Quadric) -> Quadric {
        let mut q = *a;
        q.add(b);
        q
    }

    /// `vᵀQv` with `v = (x, y, z, 1)`.
    fn error(&self, p: [f64; 3]) -> f64 {
        let q = &self.0;
        let [x, y, z] = p;
        q[0] * x * x
            + 2.0 * q[1] * x * y
            + 2.0 * q[2] * x * z
            + 2.0 * q[3] * x
            + q[4] * y * y
            + 2.0 * q[5] * y * z
            + 2.0 * q[6] * y
            + q[7] * z * z
            + 2.0 * q[8] * z
            + q[9]
    }

    /// The point of least error, when the quadric pins one down — not on a
    /// flat or a straight crease, where a whole plane or line ties.
    fn optimum(&self) -> Option<[f64; 3]> {
        let q = &self.0;
        let m = [[q[0], q[1], q[2]], [q[1], q[4], q[5]], [q[2], q[5], q[7]]];
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let scale = q[0].abs() + q[4].abs() + q[7].abs();
        if det.abs() <= 1e-9 * scale.powi(3).max(1e-300) {
            return None;
        }
        let r = [-q[3], -q[6], -q[8]];
        let solve = |col: usize| {
            let mut k = m;
            for row in 0..3 {
                k[row][col] = r[row];
            }
            (k[0][0] * (k[1][1] * k[2][2] - k[1][2] * k[2][1])
                - k[0][1] * (k[1][0] * k[2][2] - k[1][2] * k[2][0])
                + k[0][2] * (k[1][0] * k[2][1] - k[1][1] * k[2][0]))
                / det
        };
        Some([solve(0), solve(1), solve(2)])
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// A candidate collapse in the queue, cheapest first.
struct Candidate {
    cost: f64,
    a: u32,
    b: u32,
    /// The ends' versions when this was costed; a stale entry is skipped.
    stamp: (u32, u32),
    to: [f64; 3],
}

impl PartialEq for Candidate {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Candidate {
    fn cmp(&self, o: &Self) -> Ordering {
        // Reversed for a min-heap; ties broken by the edge so the order — and
        // so the result — never depends on anything but the input.
        o.cost
            .total_cmp(&self.cost)
            .then_with(|| (o.a, o.b).cmp(&(self.a, self.b)))
    }
}

/// Simplify an indexed triangle mesh.
///
/// Returns positions, recomputed normals when the input had normals, and the
/// index; other attributes do not survive a collapse and are dropped. Vertices
/// on an open boundary never move, so an open mesh keeps its outline.
pub fn simplify(
    geom: &BufferGeometry,
    options: SimplifyOptions,
) -> (BufferGeometry, SimplifyReport) {
    let (Some(index), Some(position)) = (geom.index.as_ref(), geom.get_attribute("position"))
    else {
        return (geom.clone(), SimplifyReport::default());
    };
    let mut pos: Vec<[f64; 3]> = position
        .array
        .chunks_exact(3)
        .map(|c| [c[0] as f64, c[1] as f64, c[2] as f64])
        .collect();
    let n = pos.len();
    let mut tris: Vec<[u32; 3]> = index.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
    let mut alive = vec![true; tris.len()];
    let mut live = tris.len();
    let mut report = SimplifyReport {
        triangles_before: live,
        ..Default::default()
    };

    // Who touches what.
    let mut around: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (t, tri) in tris.iter().enumerate() {
        for &v in tri {
            around[v as usize].push(t as u32);
        }
    }
    // Boundary vertices: an edge used once.
    let mut boundary = vec![false; n];
    {
        let mut count: std::collections::HashMap<(u32, u32), u32> =
            std::collections::HashMap::new();
        for tri in &tris {
            for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                *count.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        for ((a, b), c) in count {
            if c != 2 {
                boundary[a as usize] = true;
                boundary[b as usize] = true;
            }
        }
    }
    // Area-weighted face planes.
    let mut quadric = vec![Quadric::default(); n];
    for tri in &tris {
        let [a, b, c] = tri.map(|v| pos[v as usize]);
        let nrm = cross(sub(b, a), sub(c, a));
        let len = dot(nrm, nrm).sqrt();
        if len <= 0.0 {
            continue;
        }
        let unit = nrm.map(|x| x / len);
        let q = Quadric::plane(unit, -dot(unit, a), 0.5 * len);
        for &v in tri {
            quadric[v as usize].add(&q);
        }
    }

    let mut version = vec![0u32; n];
    let mut heap = BinaryHeap::new();
    let max_error = (options.max_error as f64).powi(2);
    let cost = |a: u32,
                b: u32,
                pos: &[[f64; 3]],
                quadric: &[Quadric],
                boundary: &[bool]|
     -> Option<(f64, [f64; 3])> {
        if boundary[a as usize] || boundary[b as usize] {
            return None;
        }
        let q = Quadric::sum(&quadric[a as usize], &quadric[b as usize]);
        let (pa, pb) = (pos[a as usize], pos[b as usize]);
        let mid = [
            (pa[0] + pb[0]) * 0.5,
            (pa[1] + pb[1]) * 0.5,
            (pa[2] + pb[2]) * 0.5,
        ];
        let mut best = (q.error(pa), pa);
        for p in [pb, mid].into_iter().chain(q.optimum()) {
            let e = q.error(p);
            if e < best.0 {
                best = (e, p);
            }
        }
        // Normalised by the faces' total weight, so the tolerance is a
        // distance and not a distance times an area.
        let weight = (q.0[0] + q.0[4] + q.0[7]).max(1e-300);
        Some((best.0.max(0.0) / weight, best.1))
    };
    let push = |heap: &mut BinaryHeap<Candidate>,
                a: u32,
                b: u32,
                pos: &[[f64; 3]],
                quadric: &[Quadric],
                version: &[u32]| {
        let (a, b) = (a.min(b), a.max(b));
        if let Some((c, to)) = cost(a, b, pos, quadric, &boundary) {
            if c <= max_error {
                heap.push(Candidate {
                    cost: c,
                    a,
                    b,
                    stamp: (version[a as usize], version[b as usize]),
                    to,
                });
            }
        }
    };
    {
        let mut seen = HashSet::new();
        for tri in &tris {
            for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                if seen.insert((a.min(b), a.max(b))) {
                    push(&mut heap, a, b, &pos, &quadric, &version);
                }
            }
        }
    }

    let neighbours = |v: u32, tris: &[[u32; 3]], around: &[Vec<u32>]| -> Vec<u32> {
        let mut out: Vec<u32> = around[v as usize]
            .iter()
            .flat_map(|&t| tris[t as usize])
            .filter(|&w| w != v)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    };

    while let Some(c) = heap.pop() {
        if options.target_triangles > 0 && live <= options.target_triangles {
            break;
        }
        let (a, b) = (c.a, c.b);
        if c.stamp != (version[a as usize], version[b as usize]) {
            continue;
        }
        // The two triangles across the edge, and the link condition.
        let shared: Vec<u32> = around[a as usize]
            .iter()
            .copied()
            .filter(|t| tris[*t as usize].contains(&b))
            .collect();
        if shared.len() != 2 || live <= 4 {
            report.refused += 1;
            continue;
        }
        let opposite: Vec<u32> = shared
            .iter()
            .map(|&t| {
                *tris[t as usize]
                    .iter()
                    .find(|&&w| w != a && w != b)
                    .unwrap()
            })
            .collect();
        let (na, nb) = (neighbours(a, &tris, &around), neighbours(b, &tris, &around));
        let common = na.iter().filter(|w| nb.binary_search(w).is_ok()).count();
        if common != 2 || opposite[0] == opposite[1] {
            report.refused += 1;
            continue;
        }
        // No surviving triangle may turn over or collapse to a line.
        let to = c.to;
        let mut flips = false;
        for &v in &[a, b] {
            for &t in &around[v as usize] {
                let tri = tris[t as usize];
                if tri.contains(&a) && tri.contains(&b) {
                    continue;
                }
                let before = tri.map(|w| pos[w as usize]);
                let after = tri.map(|w| {
                    if w == a || w == b {
                        to
                    } else {
                        pos[w as usize]
                    }
                });
                let n0 = cross(sub(before[1], before[0]), sub(before[2], before[0]));
                let n1 = cross(sub(after[1], after[0]), sub(after[2], after[0]));
                let (l0, l1) = (dot(n0, n0).sqrt(), dot(n1, n1).sqrt());
                if l1 <= 1e-12 * l0.max(1e-300) || dot(n0, n1) <= 0.2 * l0 * l1 {
                    flips = true;
                    break;
                }
            }
            if flips {
                break;
            }
        }
        if flips {
            report.refused += 1;
            continue;
        }

        // Merge b into a.
        for &t in &shared {
            alive[t as usize] = false;
            live -= 1;
            for &w in &tris[t as usize] {
                around[w as usize].retain(|&x| x != t);
            }
        }
        let moved: Vec<u32> = std::mem::take(&mut around[b as usize]);
        for &t in &moved {
            for w in tris[t as usize].iter_mut() {
                if *w == b {
                    *w = a;
                }
            }
            around[a as usize].push(t);
        }
        pos[a as usize] = to;
        let qb = quadric[b as usize];
        quadric[a as usize].add(&qb);
        version[a as usize] += 1;
        version[b as usize] += 1;
        report.collapses += 1;
        for w in neighbours(a, &tris, &around) {
            push(&mut heap, a, w, &pos, &quadric, &version);
        }
    }

    // Compact what is left.
    let mut remap = vec![u32::MAX; n];
    let mut out_pos = Vec::new();
    let mut out_index = Vec::with_capacity(live * 3);
    for (t, tri) in tris.iter().enumerate() {
        if !alive[t] {
            continue;
        }
        for &v in tri {
            if remap[v as usize] == u32::MAX {
                remap[v as usize] = (out_pos.len() / 3) as u32;
                let p = pos[v as usize];
                out_pos.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32]);
            }
            out_index.push(remap[v as usize]);
        }
    }
    report.triangles_after = live;
    let mut out = BufferGeometry::new();
    out.set_attribute("position", BufferAttribute::new(out_pos, 3));
    out.set_index(out_index);
    if geom.get_attribute("normal").is_some() {
        crate::utils::compute_vertex_normals(&mut out);
    }
    (out, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometries::lattice::{Lattice, LatticeKind, Stochastic, Tpms};
    use crate::math::Vector3;
    use crate::utils::mesh_check::{edge_report, split_pieces, triangle_quality};

    fn volume(g: &BufferGeometry) -> f32 {
        split_pieces(g).iter().map(|p| p.volume).sum()
    }

    /// A closed unit box with every face cut into an `n × n` grid, welded.
    fn subdivided_box(n: usize) -> BufferGeometry {
        let mut index_of = std::collections::HashMap::new();
        let mut p: Vec<f32> = Vec::new();
        let mut idx = Vec::new();
        let mut vertex = |v: [i32; 3]| -> u32 {
            *index_of.entry(v).or_insert_with(|| {
                p.extend(v.map(|c| c as f32 / n as f32));
                (p.len() / 3 - 1) as u32
            })
        };
        let n = n as i32;
        // Each face: the fixed axis, its side, and the two in-plane axes in
        // the order that winds outwards.
        for (axis, side, u, w) in [
            (0, n, 1, 2),
            (0, 0, 2, 1),
            (1, n, 2, 0),
            (1, 0, 0, 2),
            (2, n, 0, 1),
            (2, 0, 1, 0),
        ] {
            for j in 0..n {
                for i in 0..n {
                    let at = |a: i32, b: i32| {
                        let mut v = [0; 3];
                        v[axis] = side;
                        v[u] = a;
                        v[w] = b;
                        v
                    };
                    let q =
                        [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)].map(&mut vertex);
                    idx.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
                }
            }
        }
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(p, 3));
        g.set_index(idx);
        g
    }

    #[test]
    fn a_subdivided_box_comes_back_to_its_dozen_triangles() {
        // Every triangle lies in one of six planes, so the quadric error of
        // merging along a face is nil, and across an edge it is not.
        let g = subdivided_box(12);
        assert!(edge_report(&g).is_closed());
        assert!((volume(&g) - 1.0).abs() < 1e-5);
        let (s, report) = simplify(&g, SimplifyOptions::default());
        assert!(edge_report(&s).is_closed(), "{:?}", edge_report(&s));
        assert_eq!(report.triangles_before, 6 * 12 * 12 * 2);
        assert_eq!(report.triangles_after, 12, "{report:?}");
        assert!((volume(&s) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_contoured_box_loses_its_flat_triangles() {
        // Marching cubes chamfers a box's edges, so it cannot get back to
        // twelve — but the flat faces still go.
        let g = crate::extras::MarchingCubes::new(24, Vector3::new(2.0, 2.0, 2.0))
            .extract(0.0, |x, y, z| 0.55 - x.abs().max(y.abs()).max(z.abs()));
        let before = g.index.as_ref().unwrap().len() / 3;
        let (s, report) = simplify(&g, SimplifyOptions::default());
        assert!(edge_report(&s).is_closed(), "{:?}", edge_report(&s));
        assert!(report.triangles_after * 20 < before, "{report:?}");
        assert!((volume(&s) - volume(&g)).abs() < 1e-3 * volume(&g));
    }

    #[test]
    fn a_sphere_keeps_its_shape_and_its_closure() {
        let g = crate::extras::MarchingCubes::new(40, Vector3::new(2.0, 2.0, 2.0))
            .extract(0.0, |x, y, z| 0.8 - (x * x + y * y + z * z).sqrt());
        let before = g.index.as_ref().unwrap().len() / 3;
        let (s, report) = simplify(
            &g,
            SimplifyOptions {
                target_triangles: before / 4,
                max_error: 0.02,
            },
        );
        assert!(edge_report(&s).is_closed());
        assert!(report.triangles_after <= before / 4 + 2, "{report:?}");
        let exact = 4.0 / 3.0 * std::f32::consts::PI * 0.8f32.powi(3);
        assert!(
            (volume(&s) - exact).abs() < 0.02 * exact,
            "{} vs {exact}",
            volume(&s)
        );
        // Every vertex still on the sphere, to the tolerance.
        let p = &s.get_attribute("position").unwrap().array;
        for v in p.chunks_exact(3) {
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((r - 0.8).abs() < 0.03, "vertex at radius {r}");
        }
    }

    #[test]
    fn lattices_simplify_without_tearing() {
        // Thin walls and many small pieces are where simplifiers tear; the
        // link condition is what keeps them closed.
        for kind in [
            LatticeKind::Tpms(Tpms::Gyroid),
            LatticeKind::Stochastic(Stochastic::Voronoi),
            LatticeKind::Stochastic(Stochastic::VoronoiWall),
        ] {
            let g = Lattice::new(kind)
                .size(Vector3::new(4.0, 4.0, 4.0))
                .cells([2, 2, 2])
                .thickness(0.4)
                .resolution(16)
                .build();
            let before = g.index.as_ref().unwrap().len() / 3;
            // A tolerance well under the features: a twentieth of the wall.
            let (s, report) = simplify(
                &g,
                SimplifyOptions {
                    target_triangles: 0,
                    max_error: 0.02,
                },
            );
            assert!(
                edge_report(&s).is_closed(),
                "{kind:?}: {:?}",
                edge_report(&s)
            );
            assert!(
                report.triangles_after < before,
                "{kind:?}: nothing collapsed"
            );
            assert_eq!(
                split_pieces(&s).len(),
                split_pieces(&g).len(),
                "{kind:?}: pieces changed"
            );
            let (v0, v1) = (volume(&g), volume(&s));
            // Quadric collapses round slightly into curved walls.
            assert!(
                (v1 - v0).abs() < 0.02 * v0,
                "{kind:?}: volume {v0} -> {v1}, {report:?}"
            );
            let q = triangle_quality(&s);
            assert!(q.needle_fraction < 0.2, "{kind:?}: {q:?}");
        }
    }

    #[test]
    fn an_open_mesh_keeps_its_outline() {
        // A flat grid: every interior vertex can go, the border cannot.
        let n = 10;
        let mut p = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                p.extend_from_slice(&[i as f32, j as f32, 0.0]);
            }
        }
        let mut idx = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let v = (j * (n + 1) + i) as u32;
                let w = v + n as u32 + 1;
                idx.extend_from_slice(&[v, v + 1, w + 1, v, w + 1, w]);
            }
        }
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(p, 3));
        g.set_index(idx);
        let (s, _) = simplify(&g, SimplifyOptions::default());
        let report = edge_report(&s);
        // Same outline: 40 border edges, all still open, none torn inside.
        assert_eq!(report.open, 4 * n);
        assert_eq!(report.non_manifold, 0);
    }
}
