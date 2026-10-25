//! Checking a triangle mesh before it goes to a printer, and tidying what a
//! contour leaves behind.
//!
//! Three questions, each answered from the index alone or the index and the
//! positions:
//!
//! - [`edge_report`] — is it closed? Every edge used exactly twice, once each
//!   way round, or the slicer has to guess.
//! - [`triangle_quality`] — how well shaped are the triangles? Needles (long
//!   and thin) are harmless to a slicer but ruin simulation meshes, normals
//!   and any later simplification.
//! - [`split_pieces`] — what is it made of? A contoured random field leaves
//!   specks of a few cells floating in the void, and a sealed bubble inside a
//!   solid is a separate, inside-out piece.
//!
//! [`remove_loose_pieces`] acts on the last of these. Its rules are the ones a
//! lattice wants, which are not "keep the biggest piece": a woven or a
//! spinodal structure can be several interlocked pieces by design, and a
//! closed-cell foam is mostly sealed cavities. So specks go only while they
//! are a negligible share of the solid, and cavities stay.

use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::Vector3;
use std::collections::HashMap;

/// Whether every edge is shared by exactly two triangles, once each way round.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EdgeReport {
    /// Edges with no partner running the other way: holes, or flipped
    /// triangles.
    pub open: usize,
    /// Edges used more than once in the same direction, or by more than two
    /// triangles: the surface is not a manifold there.
    pub non_manifold: usize,
}

impl EdgeReport {
    /// Closed and consistently wound — what a slicer needs.
    pub fn is_closed(&self) -> bool {
        self.open == 0 && self.non_manifold == 0
    }
}

/// Count open and non-manifold edges of an indexed mesh.
///
/// Vertices are compared by index, not position: a mesh with coincident but
/// unwelded vertices reports its seams as open, which is the truth about it.
pub fn edge_report(geom: &BufferGeometry) -> EdgeReport {
    let Some(index) = geom.index.as_ref() else {
        return EdgeReport::default();
    };
    // Each undirected edge: (uses a→b, uses b→a) with a < b.
    let mut edges: HashMap<(u32, u32), (u32, u32)> = HashMap::with_capacity(index.len());
    for t in index.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let entry = edges.entry((a.min(b), a.max(b))).or_default();
            if a < b {
                entry.0 += 1;
            } else {
                entry.1 += 1;
            }
        }
    }
    let mut report = EdgeReport::default();
    for &(forward, back) in edges.values() {
        if forward > 1 || back > 1 {
            report.non_manifold += 1;
        } else if forward != back {
            report.open += 1;
        }
    }
    report
}

/// How well shaped a mesh's triangles are.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TriangleQuality {
    pub triangles: usize,
    /// Triangles whose height over their longest edge is under 0.05 — about
    /// a 3° sliver or worse. Zero-area triangles count.
    pub needles: usize,
    /// `needles / triangles`.
    pub needle_fraction: f32,
    /// The fifth-percentile smallest angle, in degrees: what the worst
    /// twentieth of the mesh looks like. An equilateral mesh reads 60.
    pub min_angle_p5: f32,
    /// The mean of every triangle's smallest angle, in degrees.
    pub mean_min_angle: f32,
}

/// Height over longest edge below which a triangle counts as a needle.
const NEEDLE: f32 = 0.05;

/// Measure the triangles of an indexed mesh.
pub fn triangle_quality(geom: &BufferGeometry) -> TriangleQuality {
    let (Some(index), Some(position)) = (geom.index.as_ref(), geom.get_attribute("position"))
    else {
        return TriangleQuality::default();
    };
    let p = &position.array;
    let at = |i: u32| {
        let i = i as usize * 3;
        Vector3::new(p[i], p[i + 1], p[i + 2])
    };
    let mut angles = Vec::with_capacity(index.len() / 3);
    let mut needles = 0;
    for t in index.chunks_exact(3) {
        let (a, b, c) = (at(t[0]), at(t[1]), at(t[2]));
        let edges = [(b - a).length(), (c - b).length(), (a - c).length()];
        let longest = edges[0].max(edges[1]).max(edges[2]);
        let area2 = (b - a).cross(c - a).length();
        if longest <= 0.0 || area2 / longest < NEEDLE * longest {
            needles += 1;
        }
        // The smallest angle is opposite the shortest edge.
        let smallest = if longest > 0.0 && area2 > 0.0 {
            let shortest = edges[0].min(edges[1]).min(edges[2]);
            let [x, y] = {
                let mut others: Vec<f32> = edges.to_vec();
                let k = others.iter().position(|&e| e == shortest).unwrap_or(0);
                others.remove(k);
                [others[0], others[1]]
            };
            // Law of cosines, clamped against rounding.
            let cos = ((x * x + y * y - shortest * shortest) / (2.0 * x * y)).clamp(-1.0, 1.0);
            cos.acos().to_degrees()
        } else {
            0.0
        };
        angles.push(smallest);
    }
    let triangles = angles.len();
    if triangles == 0 {
        return TriangleQuality::default();
    }
    let mean_min_angle = angles.iter().sum::<f32>() / triangles as f32;
    angles.sort_by(f32::total_cmp);
    TriangleQuality {
        triangles,
        needles,
        needle_fraction: needles as f32 / triangles as f32,
        min_angle_p5: angles[(triangles - 1) * 5 / 100],
        mean_min_angle,
    }
}

/// One connected piece of a mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct Piece {
    /// The piece's triangles, as indices into the mesh's triangle list.
    pub triangles: Vec<usize>,
    /// Enclosed volume, positive for a solid and negative for a sealed cavity
    /// (an inside-out shell). Meaningful only for a closed piece.
    pub volume: f32,
    pub area: f32,
}

impl Piece {
    /// An inside-out shell: the boundary of a bubble sealed inside a solid.
    pub fn is_cavity(&self) -> bool {
        self.volume < 0.0
    }
}

/// Split an indexed mesh into pieces joined through shared vertices, largest
/// enclosed volume first.
pub fn split_pieces(geom: &BufferGeometry) -> Vec<Piece> {
    let (Some(index), Some(position)) = (geom.index.as_ref(), geom.get_attribute("position"))
    else {
        return Vec::new();
    };
    let p = &position.array;
    let vertices = p.len() / 3;
    let mut parent: Vec<u32> = (0..vertices as u32).collect();
    fn root(parent: &mut [u32], mut v: u32) -> u32 {
        while parent[v as usize] != v {
            let up = parent[parent[v as usize] as usize];
            parent[v as usize] = up;
            v = up;
        }
        v
    }
    for t in index.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2])] {
            let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
            if ra != rb {
                parent[ra.max(rb) as usize] = ra.min(rb);
            }
        }
    }
    let at = |i: u32| {
        let i = i as usize * 3;
        Vector3::new(p[i], p[i + 1], p[i + 2])
    };
    let mut slot: HashMap<u32, usize> = HashMap::new();
    let mut pieces: Vec<(Vec<usize>, f64, f64, Vector3)> = Vec::new();
    for (n, t) in index.chunks_exact(3).enumerate() {
        let r = root(&mut parent, t[0]);
        let k = *slot.entry(r).or_insert_with(|| {
            // Volumes are summed about a point of the piece itself, so a small
            // piece far from the origin is not lost in cancellation.
            pieces.push((Vec::new(), 0.0, 0.0, at(t[0])));
            pieces.len() - 1
        });
        let o = pieces[k].3;
        let (a, b, c) = (at(t[0]) - o, at(t[1]) - o, at(t[2]) - o);
        pieces[k].0.push(n);
        pieces[k].1 += a.dot(b.cross(c)) as f64 / 6.0;
        pieces[k].2 += (b - a).cross(c - a).length() as f64 * 0.5;
    }
    let mut out: Vec<Piece> = pieces
        .into_iter()
        .map(|(triangles, volume, area, _)| Piece {
            triangles,
            volume: volume as f32,
            area: area as f32,
        })
        .collect();
    out.sort_by(|a, b| b.volume.total_cmp(&a.volume));
    out
}

/// What [`remove_loose_pieces`] keeps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoosePieces {
    /// Everything.
    Keep,
    /// Everything but solid pieces smaller than `min_volume` — the specks a
    /// contour leaves around a random field. If those add up to more than
    /// `max_fraction` of the solid volume they are not specks but part of the
    /// design, and nothing is dropped. Cavities smaller than `min_volume` go
    /// too; larger ones stay.
    Specks { min_volume: f32, max_fraction: f32 },
    /// The largest solid piece and the cavities sealed inside it.
    Largest,
}

/// What [`remove_loose_pieces`] found and did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PieceReport {
    /// Solid pieces and cavities in the input.
    pub solids: usize,
    pub cavities: usize,
    /// Pieces dropped, and the solid volume they held.
    pub dropped: usize,
    pub dropped_volume: f32,
    /// `Specks` found more than its share to drop and so dropped nothing.
    pub capped: bool,
}

/// Drop the pieces `mode` says to, and report what was there.
///
/// Positions and every other attribute are carried over for the vertices that
/// stay; the index is renumbered to match.
pub fn remove_loose_pieces(
    geom: &BufferGeometry,
    mode: LoosePieces,
) -> (BufferGeometry, PieceReport) {
    let pieces = split_pieces(geom);
    let mut report = PieceReport {
        solids: pieces.iter().filter(|p| !p.is_cavity()).count(),
        cavities: pieces.iter().filter(|p| p.is_cavity()).count(),
        ..Default::default()
    };
    let keep: Vec<bool> = match mode {
        LoosePieces::Keep => vec![true; pieces.len()],
        LoosePieces::Specks {
            min_volume,
            max_fraction,
        } => {
            let solid: f32 = pieces
                .iter()
                .filter(|p| !p.is_cavity())
                .map(|p| p.volume)
                .sum();
            let small = |p: &Piece| p.volume.abs() < min_volume;
            let specks: f32 = pieces
                .iter()
                .filter(|p| !p.is_cavity() && small(p))
                .map(|p| p.volume)
                .sum();
            if specks > max_fraction * solid {
                report.capped = true;
                vec![true; pieces.len()]
            } else {
                pieces.iter().map(|p| !small(p)).collect()
            }
        }
        LoosePieces::Largest => {
            let index = geom.index.as_deref().unwrap_or(&[]);
            let p = &geom
                .get_attribute("position")
                .map(|a| a.array.as_slice())
                .unwrap_or(&[]);
            match pieces.iter().position(|p| !p.is_cavity()) {
                None => vec![true; pieces.len()],
                Some(big) => pieces
                    .iter()
                    .enumerate()
                    .map(|(n, piece)| {
                        n == big
                            || (piece.is_cavity()
                                && inside(&pieces[big], index, p, vertex_of(piece, index, p)))
                    })
                    .collect(),
            }
        }
    };
    for (piece, &k) in pieces.iter().zip(&keep) {
        if !k {
            report.dropped += 1;
            if !piece.is_cavity() {
                report.dropped_volume += piece.volume;
            }
        }
    }
    if report.dropped == 0 {
        return (geom.clone(), report);
    }
    let mut kept_triangles: Vec<usize> = pieces
        .iter()
        .zip(&keep)
        .filter(|(_, &k)| k)
        .flat_map(|(p, _)| p.triangles.iter().copied())
        .collect();
    kept_triangles.sort_unstable();
    (subset(geom, &kept_triangles), report)
}

/// Any vertex of a piece.
fn vertex_of(piece: &Piece, index: &[u32], p: &[f32]) -> Vector3 {
    let v = index[piece.triangles[0] * 3] as usize * 3;
    Vector3::new(p[v], p[v + 1], p[v + 2])
}

/// Whether `q` is inside a closed piece, by the parity of a ray's crossings.
/// The ray is skewed off the axes so it does not run along a lattice's grid
/// planes and graze edges.
fn inside(piece: &Piece, index: &[u32], p: &[f32], q: Vector3) -> bool {
    let dir = Vector3::new(0.577_215_7, 0.316_227_8, 0.752_355).normalize();
    let at = |i: u32| {
        let i = i as usize * 3;
        Vector3::new(p[i], p[i + 1], p[i + 2])
    };
    let mut crossings = 0;
    for &t in &piece.triangles {
        let (a, b, c) = (at(index[t * 3]), at(index[t * 3 + 1]), at(index[t * 3 + 2]));
        // Möller–Trumbore.
        let (e1, e2) = (b - a, c - a);
        let h = dir.cross(e2);
        let det = e1.dot(h);
        if det.abs() < 1e-12 {
            continue;
        }
        let s = q - a;
        let u = s.dot(h) / det;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let w = s.cross(e1);
        let v = dir.dot(w) / det;
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        if e2.dot(w) / det > 0.0 {
            crossings += 1;
        }
    }
    crossings % 2 == 1
}

/// The triangles listed (ascending), with only the vertices they use.
fn subset(geom: &BufferGeometry, triangles: &[usize]) -> BufferGeometry {
    let index = geom.index.as_deref().unwrap_or(&[]);
    let vertices = geom
        .get_attribute("position")
        .map_or(0, |a| a.array.len() / 3);
    let mut remap = vec![u32::MAX; vertices];
    let mut order: Vec<u32> = Vec::new();
    let mut new_index = Vec::with_capacity(triangles.len() * 3);
    for &t in triangles {
        for &v in &index[t * 3..t * 3 + 3] {
            if remap[v as usize] == u32::MAX {
                remap[v as usize] = order.len() as u32;
                order.push(v);
            }
            new_index.push(remap[v as usize]);
        }
    }
    let mut out = BufferGeometry::new();
    for (name, attribute) in &geom.attributes {
        let size = attribute.item_size;
        let mut data = Vec::with_capacity(order.len() * size);
        for &v in &order {
            data.extend_from_slice(&attribute.array[v as usize * size..(v as usize + 1) * size]);
        }
        out.set_attribute(name, BufferAttribute::new(data, size));
    }
    out.set_index(new_index);
    out
}

/// The numbers that plan a print: how tall, what needs support, and how much
/// of the part sits on the plate. See [`print_report`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrintReport {
    /// Extent along `up`.
    pub height: f32,
    /// Area of downward-facing triangles by how far they tilt from horizontal,
    /// in 15° bands: `[0°, 15°)` (a flat ceiling) to `[75°, 90°]` (nearly a
    /// wall). Faces resting on the plate are left out.
    pub downward_bands: [f32; 6],
    /// Area of downward faces tilted less than the overhang limit — what a
    /// powder-bed or extrusion process has to hold up.
    pub support_area: f32,
    /// How much of the plate that support stands on: the area the faces cover
    /// seen from above, overlaps counted once.
    pub support_footprint: f32,
    /// Area of the downward faces lying on the plate itself — within a
    /// thousandth of the part's height of its lowest point.
    pub plate_contact: f32,
}

/// Measure a mesh for printing with `up` as the build direction, and faces
/// tilted less than `overhang_limit` degrees from horizontal needing support
/// (45° is the usual figure for metal powder bed).
///
/// The footprint is rasterised over the part's plan on a `raster`-square grid,
/// so it is good to about one pixel along its outline; the other numbers are
/// sums over the triangles and exact.
pub fn print_report(
    geom: &BufferGeometry,
    up: Vector3,
    overhang_limit: f32,
    raster: usize,
) -> PrintReport {
    let (Some(index), Some(position)) = (geom.index.as_ref(), geom.get_attribute("position"))
    else {
        return PrintReport::default();
    };
    let up = up.normalize();
    // A frame on the plate.
    let seed = if up.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let (eu, ev) = {
        let u = up.cross(seed).normalize();
        (u, up.cross(u))
    };
    let p = &position.array;
    let at = |i: u32| {
        let i = i as usize * 3;
        Vector3::new(p[i], p[i + 1], p[i + 2])
    };
    let heights: Vec<f32> = p
        .chunks_exact(3)
        .map(|c| Vector3::new(c[0], c[1], c[2]).dot(up))
        .collect();
    let (low, high) = heights
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &z| {
            (l.min(z), h.max(z))
        });
    if !low.is_finite() {
        return PrintReport::default();
    }
    let height = high - low;
    // On the plate: within a thousandth of the height of the lowest point —
    // twenty microns on a 20 mm part, well under a first layer, and loose
    // enough that a simplifier leaving a vertex a micron low does not lift
    // the whole face off.
    let on_plate = |i: u32| heights[i as usize] - low <= 1e-3 * height.max(1e-12);

    let mut report = PrintReport {
        height,
        ..Default::default()
    };
    let mut support: Vec<[Vector3; 3]> = Vec::new();
    let limit = overhang_limit.to_radians();
    for t in index.chunks_exact(3) {
        let (a, b, c) = (at(t[0]), at(t[1]), at(t[2]));
        let n = (b - a).cross(c - a);
        let area2 = n.length();
        if area2 <= 0.0 {
            continue;
        }
        let down = -n.dot(up) / area2; // cos of the angle to straight down
        if down <= 0.0 {
            continue;
        }
        let area = 0.5 * area2;
        if on_plate(t[0]) && on_plate(t[1]) && on_plate(t[2]) {
            report.plate_contact += area;
            continue;
        }
        let tilt = down.clamp(0.0, 1.0).acos(); // 0 = facing straight down
        report.downward_bands[((tilt.to_degrees() / 15.0) as usize).min(5)] += area;
        if tilt < limit {
            report.support_area += area;
            support.push([a, b, c]);
        }
    }

    // Footprint: pixel centres on the plate covered by any projected support
    // triangle, over the plan of the part.
    if !support.is_empty() && raster > 0 {
        let plan = |v: Vector3| (v.dot(eu), v.dot(ev));
        let (mut u0, mut v0, mut u1, mut v1) = (
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        );
        for tri in &support {
            for &q in tri {
                let (u, v) = plan(q);
                (u0, v0, u1, v1) = (u0.min(u), v0.min(v), u1.max(u), v1.max(v));
            }
        }
        let size = (u1 - u0).max(v1 - v0).max(1e-12);
        let px = size / raster as f32;
        let (w, h) = (
            ((u1 - u0) / px).ceil() as usize + 1,
            ((v1 - v0) / px).ceil() as usize + 1,
        );
        let mut covered = vec![false; w * h];
        for tri in &support {
            let q = tri.map(plan);
            let (lo_u, hi_u) = (
                q[0].0.min(q[1].0).min(q[2].0),
                q[0].0.max(q[1].0).max(q[2].0),
            );
            let (lo_v, hi_v) = (
                q[0].1.min(q[1].1).min(q[2].1),
                q[0].1.max(q[1].1).max(q[2].1),
            );
            let edge = |a: (f32, f32), b: (f32, f32), x: f32, y: f32| {
                (b.0 - a.0) * (y - a.1) - (b.1 - a.1) * (x - a.0)
            };
            let i0 = ((lo_u - u0) / px - 0.5).floor().max(0.0) as usize;
            let j0 = ((lo_v - v0) / px - 0.5).floor().max(0.0) as usize;
            let i1 = (((hi_u - u0) / px - 0.5).ceil() as usize).min(w - 1);
            let j1 = (((hi_v - v0) / px - 0.5).ceil() as usize).min(h - 1);
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let (x, y) = (u0 + (i as f32 + 0.5) * px, v0 + (j as f32 + 0.5) * px);
                    let (e0, e1, e2) = (
                        edge(q[0], q[1], x, y),
                        edge(q[1], q[2], x, y),
                        edge(q[2], q[0], x, y),
                    );
                    if (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0)
                        || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0)
                    {
                        covered[j * w + i] = true;
                    }
                }
            }
        }
        report.support_footprint = covered.iter().filter(|&&c| c).count() as f32 * px * px;
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometries::lattice::{Lattice, LatticeKind, Stochastic, Tpms};

    /// An octahedron of radius `r` about `c`, outward or inside out.
    fn octahedron(c: Vector3, r: f32, inside_out: bool) -> BufferGeometry {
        let v = [
            c + Vector3::new(r, 0.0, 0.0),
            c + Vector3::new(-r, 0.0, 0.0),
            c + Vector3::new(0.0, r, 0.0),
            c + Vector3::new(0.0, -r, 0.0),
            c + Vector3::new(0.0, 0.0, r),
            c + Vector3::new(0.0, 0.0, -r),
        ];
        let mut index: Vec<u32> = vec![
            0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5,
        ];
        if inside_out {
            for t in index.chunks_exact_mut(3) {
                t.swap(1, 2);
            }
        }
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(v.iter().flat_map(|p| [p.x, p.y, p.z]).collect(), 3),
        );
        g.set_index(index);
        g
    }

    fn merge(parts: &[BufferGeometry]) -> BufferGeometry {
        crate::utils::merge_geometries(parts).expect("merge")
    }

    #[test]
    fn an_octahedron_is_closed_and_measured() {
        let g = octahedron(Vector3::ZERO, 1.0, false);
        assert!(edge_report(&g).is_closed());
        let pieces = split_pieces(&g);
        assert_eq!(pieces.len(), 1);
        // Volume 4/3 r³, area 4√3 r².
        assert!(
            (pieces[0].volume - 4.0 / 3.0).abs() < 1e-5,
            "{}",
            pieces[0].volume
        );
        assert!((pieces[0].area - 4.0 * 3f32.sqrt()).abs() < 1e-4);
        // Equilateral faces: no needles, every smallest angle 60°.
        let q = triangle_quality(&g);
        assert_eq!(q.needles, 0);
        assert!((q.min_angle_p5 - 60.0).abs() < 0.01 && (q.mean_min_angle - 60.0).abs() < 0.01);
    }

    #[test]
    fn a_missing_triangle_leaves_three_open_edges() {
        let mut g = octahedron(Vector3::ZERO, 1.0, false);
        let mut index = g.index.clone().unwrap();
        index.truncate(index.len() - 3);
        g.set_index(index);
        let r = edge_report(&g);
        assert_eq!((r.open, r.non_manifold), (3, 0));
        assert!(!r.is_closed());
    }

    #[test]
    fn a_needle_is_counted() {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.01, 0.0, 0.0, 1.0, 0.0],
                3,
            ),
        );
        g.set_index(vec![0, 1, 2, 0, 1, 3]);
        let q = triangle_quality(&g);
        assert_eq!((q.triangles, q.needles), (2, 1));
        assert!(q.min_angle_p5 < 2.0, "{}", q.min_angle_p5);
    }

    #[test]
    fn specks_go_and_cavities_stay() {
        let body = octahedron(Vector3::ZERO, 3.0, false);
        let bubble = octahedron(Vector3::ZERO, 1.0, true);
        let speck = octahedron(Vector3::new(10.0, 0.0, 0.0), 0.1, false);
        let g = merge(&[body, bubble, speck]);
        let (out, report) = remove_loose_pieces(
            &g,
            LoosePieces::Specks {
                min_volume: 0.5,
                max_fraction: 0.02,
            },
        );
        assert_eq!((report.solids, report.cavities, report.dropped), (2, 1, 1));
        assert!(!report.capped);
        let pieces = split_pieces(&out);
        assert_eq!(pieces.len(), 2, "body and bubble");
        assert!(pieces.iter().any(Piece::is_cavity));
        assert!(edge_report(&out).is_closed());
    }

    #[test]
    fn specks_that_are_the_design_are_left_alone() {
        // Ten equal small pieces: dropping them would drop everything.
        let parts: Vec<BufferGeometry> = (0..10)
            .map(|i| octahedron(Vector3::new(i as f32 * 3.0, 0.0, 0.0), 0.5, false))
            .collect();
        let (out, report) = remove_loose_pieces(
            &merge(&parts),
            LoosePieces::Specks {
                min_volume: 1.0,
                max_fraction: 0.02,
            },
        );
        assert!(report.capped);
        assert_eq!(report.dropped, 0);
        assert_eq!(split_pieces(&out).len(), 10);
    }

    #[test]
    fn largest_keeps_its_own_cavities_only() {
        let body = octahedron(Vector3::ZERO, 3.0, false);
        let inner = octahedron(Vector3::new(0.3, 0.2, 0.1), 1.0, true);
        let other = octahedron(Vector3::new(12.0, 0.0, 0.0), 2.0, false);
        let other_bubble = octahedron(Vector3::new(12.0, 0.0, 0.0), 0.5, true);
        let (out, report) = remove_loose_pieces(
            &merge(&[body, inner, other, other_bubble]),
            LoosePieces::Largest,
        );
        assert_eq!(report.dropped, 2);
        let pieces = split_pieces(&out);
        assert_eq!(pieces.len(), 2);
        assert!(
            (pieces[0].volume - 36.0).abs() < 1e-3,
            "{}",
            pieces[0].volume
        );
        assert!(pieces[1].is_cavity());
    }

    #[test]
    fn lattices_come_out_closed() {
        // The check is only worth having if the meshes it is meant for pass.
        for kind in [
            LatticeKind::Tpms(Tpms::Gyroid),
            LatticeKind::Stochastic(Stochastic::Voronoi),
            LatticeKind::Stochastic(Stochastic::Spinodal),
        ] {
            let g = Lattice::new(kind)
                .size(Vector3::new(4.0, 4.0, 4.0))
                .cells([2, 2, 2])
                .thickness(0.3)
                .resolution(10)
                .build();
            assert!(
                edge_report(&g).is_closed(),
                "{kind:?}: {:?}",
                edge_report(&g)
            );
            let q = triangle_quality(&g);
            assert!(q.triangles > 0 && q.min_angle_p5 >= 0.0);
            let solid: f32 = split_pieces(&g).iter().map(|p| p.volume).sum();
            assert!(solid > 0.0);
        }
    }

    #[test]
    fn an_octahedron_on_its_point_overhangs_at_its_face_angle() {
        // Lower faces tilt acos(1/√3) ≈ 54.7° from horizontal: self-supporting
        // at a 45° limit, all in need of support at 60°.
        let g = octahedron(Vector3::ZERO, 1.0, false);
        let up = Vector3::new(0.0, 0.0, 1.0);
        let lenient = print_report(&g, up, 45.0, 256);
        assert_eq!(lenient.support_area, 0.0);
        let strict = print_report(&g, up, 60.0, 512);
        // Four faces of area √3/2.
        assert!(
            (strict.support_area - 2.0 * 3f32.sqrt()).abs() < 1e-4,
            "{}",
            strict.support_area
        );
        assert!(
            (strict.downward_bands[3] - 2.0 * 3f32.sqrt()).abs() < 1e-4,
            "{:?}",
            strict.downward_bands
        );
        // Seen from above they cover the square with a diagonal of 2: area 2.
        assert!(
            (strict.support_footprint - 2.0).abs() < 0.02,
            "{}",
            strict.support_footprint
        );
        assert!((strict.height - 2.0).abs() < 1e-6);
        // On its point, it touches the plate nowhere with any area.
        assert_eq!(strict.plate_contact, 0.0);
    }

    #[test]
    fn a_box_rests_on_its_bottom_face() {
        let g = crate::geometries::BoxGeometry::new(2.0, 3.0, 1.0);
        let r = print_report(&g, Vector3::new(0.0, 0.0, 1.0), 45.0, 128);
        assert!((r.plate_contact - 6.0).abs() < 1e-5, "{}", r.plate_contact);
        assert_eq!(r.support_area, 0.0);
        assert!((r.height - 1.0).abs() < 1e-6);
        // Stood on a 2×1 face instead: that face is the contact, and its
        // opposite faces up, so still nothing needs holding up.
        let side = print_report(&g, Vector3::new(0.0, 1.0, 0.0), 45.0, 128);
        assert!((side.plate_contact - 2.0).abs() < 1e-5);
        assert_eq!(side.support_area, 0.0);
        assert!((side.height - 3.0).abs() < 1e-6);
    }
}
