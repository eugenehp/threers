//! Recovering quads from a triangle-only mesh — Blender's "Tris to Quads".
//!
//! glTF and STL carry triangles and nothing else, so a mesh that was authored
//! as quads arrives with every quad split along a diagonal and no record of
//! which two triangles were one face. [`BufferGeometry::infer_quads`] pairs
//! them back up: two triangles that share an edge become a quad when the quad
//! they make is flat enough, convex, and close enough to rectangular, best
//! pairs first.
//!
//! # The measure
//!
//! A candidate is two triangles sharing an edge with opposite windings along
//! it (so the quad has a consistent side). It is refused when
//!
//! - the angle between the triangles' normals exceeds `max_face_angle`;
//! - the quad is not strictly convex about its mean normal;
//! - any corner angle is further than `max_shape_angle` from 90°.
//!
//! Survivors are ranked by an error that adds the normal angle (over π), the
//! summed corner-angle deviation from 90° (over 2π), and `1 − min/max` of the
//! areas of the quad's two diagonal splits — the term that is zero only for a
//! planar convex quad. This is the error Blender's `join_triangles` operator
//! uses, and its defaults are 40° for both thresholds. Pairs are taken
//! greedily in ascending error, ties broken by triangle number, so the result
//! is deterministic.
//!
//! # What counts as a shared edge
//!
//! Corners are matched on their position *and* every other per-vertex
//! attribute except `normal`. Position alone would pair across a UV seam,
//! making a quad whose corners carry UVs from two sides of it; including the
//! normal would refuse every pair on a flat-shaded import (STL, a faceted
//! glTF), where each triangle has its own normals, which is exactly where
//! recovery is wanted. A vertex split only for its normal still pairs.
// `!(x <= limit)` is deliberate throughout: it also rejects NaN, which a
// degenerate triangle produces, where `x > limit` would let it through.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use super::BufferGeometry;
use std::collections::HashMap;

/// Blender's default for both thresholds of "Tris to Quads": 40°.
pub const DEFAULT_INFER_QUADS_ANGLE: f32 = 40.0 * std::f32::consts::PI / 180.0;

type P3 = [f32; 3];

fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: P3, b: P3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn length(a: P3) -> f32 {
    dot(a, a).sqrt()
}
fn tri_area(a: P3, b: P3, c: P3) -> f32 {
    0.5 * length(cross(sub(b, a), sub(c, a)))
}
fn angle_between(a: P3, b: P3) -> f32 {
    let (la, lb) = (length(a), length(b));
    if la == 0.0 || lb == 0.0 {
        return std::f32::consts::PI;
    }
    (dot(a, b) / (la * lb)).clamp(-1.0, 1.0).acos()
}

/// How bad a quad `q` (in loop order) made of two triangles with unit normals
/// `n1`, `n2` would be, or `None` when it breaks a threshold.
fn quad_error(q: [P3; 4], n1: P3, n2: P3, max_face: f32, max_shape: f32) -> Option<f32> {
    use std::f32::consts::{FRAC_PI_2, PI};
    let face = angle_between(n1, n2);
    if !(face <= max_face) {
        return None;
    }
    let mean = [n1[0] + n2[0], n1[1] + n2[1], n1[2] + n2[2]];
    let scale = (0..4)
        .map(|i| length(sub(q[(i + 1) % 4], q[i])))
        .fold(0.0f32, f32::max);
    if scale == 0.0 || length(mean) == 0.0 {
        return None;
    }
    let mut shape_sum = 0.0;
    for i in 0..4 {
        let (prev, here, next) = (q[(i + 3) % 4], q[i], q[(i + 1) % 4]);
        // Strictly convex: every corner turns the same way about the mean
        // normal, by more than rounding.
        let turn = dot(cross(sub(here, prev), sub(next, here)), mean);
        if !(turn > 1e-6 * scale * scale * length(mean)) {
            return None;
        }
        let deviation = (angle_between(sub(prev, here), sub(next, here)) - FRAC_PI_2).abs();
        if !(deviation <= max_shape) {
            return None;
        }
        shape_sum += deviation;
    }
    let split_02 = tri_area(q[0], q[1], q[2]) + tri_area(q[0], q[2], q[3]);
    let split_13 = tri_area(q[1], q[2], q[3]) + tri_area(q[1], q[3], q[0]);
    let (lo, hi) = (split_02.min(split_13), split_02.max(split_13));
    let concavity = if hi > 0.0 { 1.0 - lo / hi } else { 1.0 };
    Some(face / PI + shape_sum / (2.0 * PI) + concavity)
}

impl BufferGeometry {
    /// Pair adjacent triangles back into quads — Blender's "Tris to Quads",
    /// for glTF and STL imports, which carry triangles only. Returns how many
    /// quads were formed.
    ///
    /// Two triangles sharing an edge (with opposite windings along it) are a
    /// candidate unless the fold between them exceeds `max_face_angle`, the
    /// quad is not strictly convex, or a corner is further than
    /// `max_shape_angle` from 90°. Candidates are ranked by Blender's
    /// `join_triangles` error — normal angle / π + summed corner deviation /
    /// 2π + `1 − min/max` of the two diagonal splits' areas — and taken
    /// greedily, ties broken by triangle number, so the result is
    /// deterministic. Corners are matched on position and every per-vertex
    /// attribute except `normal`: a UV seam is a boundary, a flat-shading
    /// normal split is not.
    ///
    /// `max_face_angle` bounds the fold between the two triangles and
    /// `max_shape_angle` how far any quad corner may be from 90°, both in
    /// radians; [`DEFAULT_INFER_QUADS_ANGLE`] (40°) is Blender's default for
    /// each.
    ///
    /// Only the index is touched: each quad's two triangles are moved to be
    /// consecutive (the first-numbered one first, each keeping its own corner
    /// order), every other triangle keeps its relative order, and the
    /// [polygon grouping](Self::polygons) is set to match. No vertex is
    /// added, removed or changed, so normals, UVs and every other attribute
    /// stay exactly as they were. Unindexed geometry gains an index over its
    /// existing vertices. Polygons the geometry already groups are left whole
    /// and in place; only its single triangles are candidates.
    ///
    /// Because triangles move, sidecars keyed by triangle number follow: a
    /// `brep` surface table is permuted to match (and a pair is never made
    /// across two surfaces), while a `mesh-bvh` bounds tree is dropped, as any
    /// index write drops it. When no pair is found the geometry is untouched.
    pub fn infer_quads(&mut self, max_face_angle: f32, max_shape_angle: f32) -> usize {
        let triangles = self.triangle_count();
        let Some(position) = self.get_attribute("position").filter(|p| p.item_size == 3) else {
            return 0;
        };
        let vertices = position.count();
        let point = |v: u32| -> P3 {
            let at = v as usize * 3;
            [
                position.array[at],
                position.array[at + 1],
                position.array[at + 2],
            ]
        };
        let tris: Vec<[u32; 3]> = (0..triangles).map(|t| self.triangle(t)).collect();
        if tris.iter().flatten().any(|&v| v as usize >= vertices) {
            return 0;
        }

        // Which triangles are free to pair: those that are a polygon alone.
        let ranges: Vec<std::ops::Range<usize>> = self.polygon_ranges().collect();
        let mut free = vec![false; triangles];
        for r in &ranges {
            if r.len() == 1 {
                free[r.start] = true;
            }
        }

        // One id per distinct (position, attributes except `normal`).
        let canonical = |x: f32| if x == 0.0 { 0u32 } else { x.to_bits() };
        let mut names: Vec<&String> = self
            .attributes
            .iter()
            .filter(|(name, a)| {
                name.as_str() != "normal" && a.count() == vertices && a.item_size > 0
            })
            .map(|(name, _)| name)
            .collect();
        names.sort();
        let mut ids: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut weld = Vec::with_capacity(vertices);
        for v in 0..vertices {
            let mut key = Vec::new();
            for name in &names {
                let a = &self.attributes[*name];
                let at = v * a.item_size;
                key.extend(a.array[at..at + a.item_size].iter().map(|&x| canonical(x)));
            }
            let next = ids.len() as u32;
            weld.push(*ids.entry(key).or_insert(next));
        }

        #[cfg(feature = "brep")]
        let surface_of = {
            let table = self.surface_table();
            move |t: usize| table.and_then(|s| s.surface_index_of(t))
        };
        #[cfg(not(feature = "brep"))]
        let surface_of = |_: usize| None::<usize>;

        // Directed welded edges of the free, non-degenerate triangles.
        let mut normals: Vec<Option<P3>> = vec![None; triangles];
        let mut directed: HashMap<(u32, u32), Vec<(usize, usize)>> = HashMap::new();
        for (t, tri) in tris.iter().enumerate() {
            if !free[t] {
                continue;
            }
            let w = tri.map(|v| weld[v as usize]);
            if w[0] == w[1] || w[1] == w[2] || w[2] == w[0] {
                continue;
            }
            let [a, b, c] = tri.map(point);
            let n = cross(sub(b, a), sub(c, a));
            let len = length(n);
            if !(len > 0.0) {
                continue;
            }
            normals[t] = Some([n[0] / len, n[1] / len, n[2] / len]);
            for k in 0..3 {
                directed
                    .entry((w[k], w[(k + 1) % 3]))
                    .or_default()
                    .push((t, k));
            }
        }

        // Candidates: an edge used once each way and by nothing else.
        let mut candidates: Vec<(f32, usize, usize)> = Vec::new();
        for (t1, tri) in tris.iter().enumerate() {
            let Some(n1) = normals[t1] else { continue };
            let w = tri.map(|v| weld[v as usize]);
            for k in 0..3 {
                let (wu, wv) = (w[k], w[(k + 1) % 3]);
                let (Some(forward), Some(back)) =
                    (directed.get(&(wu, wv)), directed.get(&(wv, wu)))
                else {
                    continue;
                };
                if forward.len() != 1 || back.len() != 1 {
                    continue;
                }
                let (t2, k2) = back[0];
                if t2 <= t1 || surface_of(t1) != surface_of(t2) {
                    continue; // each pair once, from its lower triangle
                }
                let Some(n2) = normals[t2] else { continue };
                let u = tri[k];
                let v = tri[(k + 1) % 3];
                let a = tri[(k + 2) % 3];
                let b = tris[t2][(k2 + 2) % 3];
                let wb = weld[b as usize];
                if wb == w[(k + 2) % 3] {
                    continue; // the two triangles are one folded over the other
                }
                let quad = [point(u), point(b), point(v), point(a)];
                if let Some(e) = quad_error(quad, n1, n2, max_face_angle, max_shape_angle) {
                    candidates.push((e, t1, t2));
                }
            }
        }
        if candidates.is_empty() {
            return 0;
        }
        candidates.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
        let mut partner: Vec<Option<usize>> = vec![None; triangles];
        let mut quads = 0;
        for (_, t1, t2) in candidates {
            if partner[t1].is_none() && partner[t2].is_none() {
                partner[t1] = Some(t2);
                partner[t2] = Some(t1);
                quads += 1;
            }
        }

        // The new order: polygons as they come, a quad's partner pulled up
        // behind its first triangle.
        let mut order: Vec<usize> = Vec::with_capacity(triangles);
        let mut counts: Vec<u32> = Vec::with_capacity(ranges.len());
        let mut placed = vec![false; triangles];
        for r in &ranges {
            if r.len() > 1 {
                order.extend(r.clone());
                counts.push(r.len() as u32);
                continue;
            }
            let t = r.start;
            if placed[t] {
                continue;
            }
            placed[t] = true;
            order.push(t);
            match partner[t] {
                Some(p) => {
                    placed[p] = true;
                    order.push(p);
                    counts.push(2);
                }
                None => counts.push(1),
            }
        }

        #[cfg(feature = "brep")]
        let surfaces = self.surface_table().and_then(|table| {
            let tri_face: Option<Vec<u32>> = order
                .iter()
                .map(|&t| table.surface_index_of(t).map(|s| s as u32))
                .collect();
            crate::brep::SurfaceTable::new(table.surfaces().to_vec(), tri_face?)
        });

        let index: Vec<u32> = order.iter().flat_map(|&t| tris[t]).collect();
        self.set_index(index);
        self.set_polygons(counts);
        #[cfg(feature = "brep")]
        if let Some(table) = surfaces {
            self.set_surfaces(table);
        }
        quads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BufferAttribute;

    const DEG: f32 = std::f32::consts::PI / 180.0;

    /// An `n × n` grid of unit cells in z = 0, each split along a diagonal
    /// chosen by `diagonal(x, y)`, triangles in the order `order` gives.
    fn grid(n: u32, diagonal: impl Fn(u32, u32) -> bool) -> (BufferGeometry, Vec<[u32; 3]>) {
        let mut positions = Vec::new();
        let mut uvs = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                positions.extend_from_slice(&[x as f32, y as f32, 0.0]);
                uvs.extend_from_slice(&[x as f32 / n as f32, y as f32 / n as f32]);
            }
        }
        let at = |x: u32, y: u32| y * (n + 1) + x;
        let mut tris = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let (a, b, c, d) = (at(x, y), at(x + 1, y), at(x + 1, y + 1), at(x, y + 1));
                if diagonal(x, y) {
                    tris.push([a, b, c]);
                    tris.push([a, c, d]);
                } else {
                    tris.push([a, b, d]);
                    tris.push([b, c, d]);
                }
            }
        }
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_attribute("uv", BufferAttribute::new(uvs, 2));
        (g, tris)
    }

    /// A deterministic shuffle (xorshift), so no dependency and no flake.
    fn shuffle<T>(items: &mut [T], mut seed: u64) {
        for i in (1..items.len()).rev() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            items.swap(i, (seed % (i as u64 + 1)) as usize);
        }
    }

    fn cells_recovered(g: &BufferGeometry) -> Vec<Vec<u32>> {
        let mut cells: Vec<Vec<u32>> = g
            .polygon_ranges()
            .map(|r| {
                let mut c = g.polygon_loop(r).expect("a recoverable loop");
                c.sort();
                c
            })
            .collect();
        cells.sort();
        cells
    }

    #[test]
    fn a_shuffled_grid_with_both_diagonals_comes_back_as_its_cells() {
        let n = 6;
        for (name, diagonal) in [
            (
                "all one way",
                Box::new(|_, _| true) as Box<dyn Fn(u32, u32) -> bool>,
            ),
            ("all the other way", Box::new(|_, _| false)),
            ("alternating", Box::new(|x, y| (x + y) % 2 == 0)),
        ] {
            let (mut g, mut tris) = grid(n, diagonal);
            shuffle(&mut tris, 0x9e3779b97f4a7c15);
            let uv_before = g.get_attribute("uv").unwrap().array.clone();
            let pos_before = g.get_attribute("position").unwrap().array.clone();
            g.set_index(tris.iter().flatten().copied().collect());
            let mut before: Vec<[u32; 3]> = tris.clone();
            before.sort();

            let quads = g.infer_quads(DEFAULT_INFER_QUADS_ANGLE, DEFAULT_INFER_QUADS_ANGLE);
            assert_eq!(quads, (n * n) as usize, "{name}");
            assert_eq!(
                g.polygon_counts(),
                Some(&vec![2u32; (n * n) as usize][..]),
                "{name}"
            );
            // Exactly the cells.
            let at = |x: u32, y: u32| y * (n + 1) + x;
            let mut expected: Vec<Vec<u32>> = (0..n)
                .flat_map(|y| (0..n).map(move |x| (x, y)))
                .map(|(x, y)| {
                    let mut c = vec![at(x, y), at(x + 1, y), at(x + 1, y + 1), at(x, y + 1)];
                    c.sort();
                    c
                })
                .collect();
            expected.sort();
            assert_eq!(cells_recovered(&g), expected, "{name}");
            // Index reorder only: the same triangles, corner order kept, and
            // every attribute untouched.
            let mut after: Vec<[u32; 3]> = (0..g.triangle_count()).map(|t| g.triangle(t)).collect();
            after.sort();
            assert_eq!(after, before, "{name}");
            assert_eq!(g.get_attribute("uv").unwrap().array, uv_before);
            assert_eq!(g.get_attribute("position").unwrap().array, pos_before);
        }
    }

    #[test]
    fn it_is_deterministic() {
        let run = || {
            let (mut g, mut tris) = grid(5, |x, y| (x * 7 + y * 3) % 2 == 0);
            shuffle(&mut tris, 42);
            g.set_index(tris.iter().flatten().copied().collect());
            g.infer_quads(DEFAULT_INFER_QUADS_ANGLE, DEFAULT_INFER_QUADS_ANGLE);
            (g.index.clone(), g.polygons.clone())
        };
        assert_eq!(run(), run());
    }

    /// A triangle soup — every triangle its own three vertices, as STL loads —
    /// pairs on position, and gains an index over the same vertices.
    #[test]
    fn an_unindexed_soup_pairs_on_position() {
        let (g, tris) = grid(3, |_, _| true);
        let p = &g.get_attribute("position").unwrap().array;
        let soup: Vec<f32> = tris
            .iter()
            .flatten()
            .flat_map(|&v| p[v as usize * 3..v as usize * 3 + 3].to_vec())
            .collect();
        let mut s = BufferGeometry::new();
        s.set_attribute("position", BufferAttribute::new(soup.clone(), 3));
        assert_eq!(
            s.infer_quads(DEFAULT_INFER_QUADS_ANGLE, DEFAULT_INFER_QUADS_ANGLE),
            9
        );
        assert_eq!(s.get_attribute("position").unwrap().array, soup);
        assert_eq!(s.polygon_counts(), Some(&[2u32; 9][..]));
    }

    /// A UV seam is a real boundary: the same edge in space, different UVs
    /// either side, no pair. A normal-only split is not.
    #[test]
    fn a_uv_seam_is_respected_and_a_normal_split_is_not() {
        let quad = |uv_b: [f32; 2], normal_b: [f32; 3]| {
            // Two triangles over the unit square, the shared diagonal's
            // vertices duplicated for the second triangle.
            let mut g = BufferGeometry::new();
            g.set_attribute(
                "position",
                BufferAttribute::new(
                    vec![
                        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, // a b c
                        0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, // a' c' d
                    ],
                    3,
                ),
            );
            let mut uv = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
            uv[6] = uv_b[0];
            uv[7] = uv_b[1];
            g.set_attribute("uv", BufferAttribute::new(uv, 2));
            let mut n = [0.0, 0.0, 1.0].repeat(6);
            n[9..12].copy_from_slice(&normal_b);
            g.set_attribute("normal", BufferAttribute::new(n, 3));
            g.infer_quads(DEFAULT_INFER_QUADS_ANGLE, DEFAULT_INFER_QUADS_ANGLE)
        };
        assert_eq!(quad([0.0, 0.0], [0.0, 0.0, 1.0]), 1, "welded");
        assert_eq!(quad([0.0, 0.0], [0.0, 0.6, 0.8]), 1, "normals do not split");
        assert_eq!(quad([0.5, 0.0], [0.0, 0.0, 1.0]), 0, "a uv seam does");
    }

    fn pair(d: P3) -> BufferGeometry {
        // Triangles (a, b, c) and (a, c, d) over a, b, c fixed.
        let mut g = BufferGeometry::new();
        let mut p = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        p.extend_from_slice(&d);
        g.set_attribute("position", BufferAttribute::new(p, 3));
        g.set_index(vec![0, 1, 2, 0, 2, 3]);
        g
    }

    #[test]
    fn folded_concave_and_skewed_pairs_are_refused() {
        let ok = |d: P3| pair(d).infer_quads(DEFAULT_INFER_QUADS_ANGLE, DEFAULT_INFER_QUADS_ANGLE);
        assert_eq!(ok([0.0, 1.0, 0.0]), 1, "a square");
        assert_eq!(ok([0.0, 1.0, 0.3]), 1, "a little out of plane");
        assert_eq!(ok([0.0, 1.0, 1.5]), 0, "folded past 40°");
        assert_eq!(ok([0.7, 0.3, 0.0]), 0, "d behind the diagonal: faces flip");
        assert_eq!(ok([-0.5, -0.1, 0.0]), 0, "concave at a");
        assert_eq!(
            pair([-0.5, -0.1, 0.0]).infer_quads(179.0 * DEG, 179.0 * DEG),
            0
        );
        assert_eq!(
            ok([-1.5, 1.0, 0.0]),
            0,
            "convex but corners 56° from square"
        );
        // The thresholds are the caller's: a loose shape limit takes the skew.
        assert_eq!(
            pair([-1.5, 1.0, 0.0]).infer_quads(40.0 * DEG, 89.0 * DEG),
            1
        );
        // And a tight fold limit refuses the slight bend.
        assert_eq!(pair([0.0, 1.0, 0.3]).infer_quads(5.0 * DEG, 40.0 * DEG), 0);
    }

    /// A refused geometry is left alone — not even re-indexed.
    #[test]
    fn nothing_found_changes_nothing() {
        let mut g = pair([-0.5, -0.1, 0.0]);
        let version = g.geometry_version;
        assert_eq!(
            g.infer_quads(DEFAULT_INFER_QUADS_ANGLE, DEFAULT_INFER_QUADS_ANGLE),
            0
        );
        assert_eq!(g.geometry_version, version);
        assert!(g.polygons.is_none());
    }

    /// A UV sphere: the cells of every ring become quads again; the pole fans
    /// stay triangles. With the UVs gone the pole triangles do share edges,
    /// and the quads they would make are refused on shape, not on welding.
    #[test]
    fn a_sphere_recovers_its_cells_and_keeps_its_pole_fans() {
        let (w, h) = (16usize, 10usize);
        for keep_uv in [true, false] {
            let mut g = crate::geometries::SphereGeometry::new(1.0, w, h);
            if !keep_uv {
                g.attributes.remove("uv");
            }
            let index = g.index.clone().unwrap();
            g.set_index(index); // drop the generator's grouping
            let quads = g.infer_quads(DEFAULT_INFER_QUADS_ANGLE, DEFAULT_INFER_QUADS_ANGLE);
            assert_eq!(quads, w * (h - 2), "keep_uv {keep_uv}");
            let counts = g.polygon_counts().unwrap();
            assert_eq!(
                counts.iter().filter(|&&c| c == 1).count(),
                2 * w,
                "pole triangles"
            );
            assert!(g.polygon_ranges().all(|r| g.polygon_loop(r).is_some()));
        }
    }

    /// The glTF fixture: a UV sphere exported by another tool, triangles only.
    #[test]
    fn a_gltf_sphere_comes_back_as_quads() {
        use crate::core::ObjectKind;
        let scene = crate::loaders::GltfLoader::parse_glb(include_bytes!(
            "../loaders/testdata/sphere_plain.glb"
        ))
        .expect("parses");
        let mut g = scene
            .arena
            .nodes
            .iter()
            .find_map(|(_, o)| match &o.kind {
                ObjectKind::Mesh(m) => Some((*m.geometry).clone()),
                _ => None,
            })
            .expect("a mesh");
        let triangles = g.triangle_count();
        assert_eq!(triangles, 720);
        let quads = g.infer_quads(DEFAULT_INFER_QUADS_ANGLE, DEFAULT_INFER_QUADS_ANGLE);
        let counts = g.polygon_counts().unwrap().to_vec();
        let singles = counts.iter().filter(|&&c| c == 1).count();
        assert_eq!(quads * 2 + singles, triangles);
        // 720 = 24 segments × 14 quad rings × 2 + 2 × 24 pole triangles.
        assert_eq!((quads, singles), (336, 48), "counts {counts:?}");
        assert!(g.polygon_ranges().all(|r| g.polygon_loop(r).is_some()));
    }

    /// Existing polygons are kept whole; only the single triangles pair.
    #[test]
    fn existing_polygons_stay() {
        let (mut g, tris) = grid(2, |_, _| true);
        g.set_index(tris.iter().flatten().copied().collect());
        // Cell 0 already a quad; the other three cells loose.
        g.set_polygons(vec![2, 1, 1, 1, 1, 1, 1]);
        assert_eq!(
            g.infer_quads(DEFAULT_INFER_QUADS_ANGLE, DEFAULT_INFER_QUADS_ANGLE),
            3
        );
        assert_eq!(g.polygon_counts(), Some(&[2u32, 2, 2, 2][..]));
        assert_eq!(&g.index.as_ref().unwrap()[..6], &[0, 1, 4, 0, 4, 3]);
    }

    /// Provenance follows the triangles it describes.
    #[cfg(feature = "brep")]
    #[test]
    fn a_surface_table_is_permuted_with_the_triangles() {
        let (mut g, mut tris) = grid(3, |_, _| true);
        shuffle(&mut tris, 7);
        g.set_index(tris.iter().flatten().copied().collect());
        // Left column of cells on one plane, the rest on another (same plane
        // geometrically — the point is that the ids follow).
        let column = |t: [u32; 3]| t.iter().all(|&v| v % 4 <= 1);
        let tri_face: Vec<u32> = tris
            .iter()
            .map(|&t| if column(t) { 0 } else { 1 })
            .collect();
        let plane = crate::brep::Surface::plane([0.0; 3], [0.0, 0.0, 1.0]);
        g.set_surfaces(
            crate::brep::SurfaceTable::new(vec![plane.clone(), plane], tri_face).unwrap(),
        );
        assert_eq!(
            g.infer_quads(DEFAULT_INFER_QUADS_ANGLE, DEFAULT_INFER_QUADS_ANGLE),
            9
        );
        let table = g.surface_table().expect("kept");
        for t in 0..g.triangle_count() {
            let expected = if column(g.triangle(t)) { 0 } else { 1 };
            assert_eq!(table.surface_index_of(t), Some(expected));
        }
    }
}
