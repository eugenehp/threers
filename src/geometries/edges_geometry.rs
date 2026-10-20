use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::Vector3;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

pub struct EdgesGeometry;

/// Decimal places [`EdgesGeometry`] rounds positions to before matching
/// corners: three.js's `precisionPoints`.
const PRECISION_POINTS: i32 = 4;

/// A corner's identity for [`EdgesGeometry`]'s edge matching: its position
/// scaled by `10^PRECISION_POINTS` and rounded, as three.js's `EdgesGeometry`
/// hashes `Math.round(x * precision)`. The scaling is done in `f64`, as
/// JavaScript does on the `f32` attribute value, and rounds half up like
/// `Math.round`.
///
/// Not `BufferGeometry::corner_keys`, which matches exact positions: that is
/// the rule for polygon outlines and for three.js's `WireframeGeometry`, and
/// this one is three.js's for `EdgesGeometry`. They differ for points closer
/// than the precision, which are one corner here and two there.
type RoundedKey = [i64; 3];

fn rounded_key(p: Vector3) -> RoundedKey {
    let precision = 10f64.powi(PRECISION_POINTS);
    let round = |v: f32| (v as f64 * precision + 0.5).floor() as i64;
    [round(p.x), round(p.y), round(p.z)]
}

/// One directed triangle edge waiting for the triangle on its other side.
struct OpenEdge {
    from: u32,
    to: u32,
    triangle: usize,
    normal: [f64; 3],
}

impl EdgesGeometry {
    /// Extract sharp edges from `source`: edges whose adjacent triangles' face
    /// normals differ by at least `threshold_angle` radians are emitted as a
    /// line-list. Border edges (only one adjacent triangle) are always emitted.
    /// An edge between two triangles of the same polygon (see
    /// [`BufferGeometry::polygons`]) is never emitted, however bent the
    /// polygon: it is a triangulation artefact, not an edge of the model.
    /// Returns a non-indexed line geometry with `position` only.
    ///
    /// This is three.js's `EdgesGeometry` algorithm (the angle is in radians
    /// here, degrees there). Corners are matched by *position* rounded to four
    /// decimal places, not by vertex index, so indexed and unindexed copies of
    /// one mesh — and meshes whose vertices are split along seams for normals
    /// or UVs — give the same edges. Points closer than that precision are
    /// one corner; points further apart are distinct, so a crack between them
    /// draws as two border edges. Each triangle edge pairs with the first
    /// unpaired edge running the other way between the same two corners;
    /// triangles with two corners at the same rounded position are skipped.
    pub fn new(source: &BufferGeometry, threshold_angle: f32) -> BufferGeometry {
        Self::with_threshold_dot(source, (threshold_angle as f64).cos())
    }

    /// [`new`](Self::new) with the threshold in degrees, as three.js takes
    /// it, converted the way three.js converts it
    /// (`Math.cos(MathUtils.DEG2RAD * thresholdAngle)`), so the cut-off is
    /// the same to the last bit.
    pub fn with_threshold_degrees(source: &BufferGeometry, threshold_angle: f64) -> BufferGeometry {
        Self::with_threshold_dot(source, ((std::f64::consts::PI / 180.0) * threshold_angle).cos())
    }

    fn with_threshold_dot(source: &BufferGeometry, threshold_dot: f64) -> BufferGeometry {
        let positions: Vec<Vector3> = match source.positions() {
            Some(it) => it.collect(),
            None => return empty_lines(),
        };

        let triangles: Vec<[u32; 3]> = match &source.index {
            Some(idx) => idx.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
            None => (0..positions.len() as u32 / 3)
                .map(|i| [i * 3, i * 3 + 1, i * 3 + 2])
                .collect(),
        };

        let polygon_of = source.triangle_polygon_ids();
        let same_polygon = |a: usize, b: usize| {
            let (pa, pb) = (polygon_of.get(a), polygon_of.get(b));
            pa.is_some() && pa == pb
        };

        let mut out_positions = Vec::new();
        let mut push = |a: Vector3, b: Vector3| {
            out_positions.extend_from_slice(&[a.x, a.y, a.z, b.x, b.y, b.z]);
        };

        // Directed edges by corner keys, in first-seen order so the output is
        // deterministic. `None` marks an edge that has found its sibling.
        let mut open: HashMap<(RoundedKey, RoundedKey), Option<usize>> = HashMap::new();
        let mut edges: Vec<OpenEdge> = Vec::new();

        for (ti, t) in triangles.iter().enumerate() {
            if t.iter().any(|&v| v as usize >= positions.len()) {
                continue;
            }
            let p = t.map(|v| positions[v as usize]);
            let keys = p.map(rounded_key);
            if keys[0] == keys[1] || keys[1] == keys[2] || keys[2] == keys[0] {
                continue; // degenerate at this precision
            }
            let normal = face_normal(p);

            for j in 0..3 {
                let k = (j + 1) % 3;
                let forward = (keys[j], keys[k]);
                let reverse = (keys[k], keys[j]);
                match open.get(&reverse).copied().flatten() {
                    Some(sibling) => {
                        let other = &edges[sibling];
                        let d = dot(normal, other.normal);
                        if !same_polygon(ti, other.triangle) && d <= threshold_dot {
                            push(p[j], p[k]);
                        }
                        open.insert(reverse, None);
                    }
                    None => {
                        if let Entry::Vacant(slot) = open.entry(forward) {
                            slot.insert(Some(edges.len()));
                            edges.push(OpenEdge {
                                from: t[j],
                                to: t[k],
                                triangle: ti,
                                normal,
                            });
                        }
                    }
                }
            }
        }

        // Whatever never found a sibling is a border edge.
        for (i, e) in edges.iter().enumerate() {
            let (pa, pb) = (positions[e.from as usize], positions[e.to as usize]);
            if open.get(&(rounded_key(pa), rounded_key(pb))) == Some(&Some(i)) {
                push(pa, pb);
            }
        }

        let mut geom = BufferGeometry::new();
        geom.set_attribute("position", BufferAttribute::new(out_positions, 3));
        geom
    }
}

/// The unit normal of a triangle, in `f64` like three.js's
/// `Triangle.getNormal`; zero for a zero-area triangle.
fn face_normal(p: [Vector3; 3]) -> [f64; 3] {
    let v = |q: Vector3| [q.x as f64, q.y as f64, q.z as f64];
    let (a, b, c) = (v(p[0]), v(p[1]), v(p[2]));
    let (u, w) = (
        [c[0] - b[0], c[1] - b[1], c[2] - b[2]],
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
    );
    let n = [
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
    ];
    let len_sq = dot(n, n);
    if len_sq > 0.0 {
        let s = 1.0 / len_sq.sqrt();
        [n[0] * s, n[1] * s, n[2] * s]
    } else {
        [0.0; 3]
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub struct WireframeGeometry;

impl WireframeGeometry {
    /// Every triangle edge as a line, each once. Edges are deduplicated by
    /// their endpoints' exact positions, in either direction, as three.js's
    /// `WireframeGeometry` does — so unindexed input, or vertices split along
    /// seams, still draws each edge once — and come out in triangle order.
    ///
    /// A geometry that carries polygons (see [`BufferGeometry::polygons`])
    /// draws their outlines instead, without the fan diagonals.
    pub fn new(source: &BufferGeometry) -> BufferGeometry {
        let positions: Vec<Vector3> = match source.positions() {
            Some(it) => it.collect(),
            None => return empty_lines(),
        };
        let mut out_positions = Vec::new();
        let mut push = |a: u32, b: u32| {
            if let (Some(pa), Some(pb)) = (positions.get(a as usize), positions.get(b as usize)) {
                out_positions.extend_from_slice(&[pa.x, pa.y, pa.z, pb.x, pb.y, pb.z]);
            }
        };
        if source.polygon_counts().is_some() {
            for [a, b] in source.polygon_edges() {
                push(a, b);
            }
        } else {
            // three.js keys `isUniqueEdge` on the coordinates' strings, where
            // `-0` prints as `0`; `corner_keys` makes the same identification.
            let key = source.corner_keys();
            let mut seen = std::collections::HashSet::new();
            // Indexed geometry is walked group by group, as three.js does —
            // which decides the order, and leaves out what no group covers.
            let ranges: Vec<std::ops::Range<usize>> = match &source.index {
                Some(index) if !source.groups.is_empty() => source
                    .groups
                    .iter()
                    .map(|g| {
                        let start = (g.start as usize).min(index.len());
                        let end = (g.start as usize + g.count as usize).min(index.len());
                        start / 3..end.div_ceil(3).max(start / 3)
                    })
                    .collect(),
                _ => std::iter::once(0..source.triangle_count()).collect(),
            };
            for t in ranges.into_iter().flatten() {
                if t >= source.triangle_count() {
                    continue;
                }
                let tri = source.triangle(t);
                for j in 0..3 {
                    let (a, b) = (tri[j], tri[(j + 1) % 3]);
                    let (ka, kb) = (key(a), key(b));
                    if seen.insert(if ka <= kb { (ka, kb) } else { (kb, ka) }) {
                        push(a, b);
                    }
                }
            }
        }
        let mut geom = BufferGeometry::new();
        geom.set_attribute("position", BufferAttribute::new(out_positions, 3));
        geom
    }
}

fn empty_lines() -> BufferGeometry {
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(Vec::new(), 3));
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quad folded along its fan diagonal, by more than the threshold.
    fn bent_quad(polygon: bool) -> BufferGeometry {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                3,
            ),
        );
        g.set_index(vec![0, 1, 2, 0, 2, 3]);
        if polygon {
            g.set_polygons(vec![2]);
        }
        g
    }

    fn segments(g: &BufferGeometry) -> usize {
        g.get_attribute("position").unwrap().array.len() / 6
    }

    #[test]
    fn edges_skip_the_diagonal_inside_a_polygon() {
        let threshold = 1f32.to_radians();
        assert_eq!(
            segments(&EdgesGeometry::new(&bent_quad(false), threshold)),
            5
        );
        assert_eq!(
            segments(&EdgesGeometry::new(&bent_quad(true), threshold)),
            4
        );
    }

    /// The same triangles with one vertex per corner, as an unindexed loader
    /// would produce them.
    fn unindexed(g: &BufferGeometry) -> BufferGeometry {
        let p = &g.get_attribute("position").unwrap().array;
        let idx = g.index.as_ref().expect("indexed input");
        let flat: Vec<f32> = idx
            .iter()
            .flat_map(|&i| p[i as usize * 3..i as usize * 3 + 3].to_vec())
            .collect();
        let mut out = BufferGeometry::new();
        out.set_attribute("position", BufferAttribute::new(flat, 3));
        out
    }

    /// Each segment as a pair of rounded endpoints, direction-free, sorted.
    fn edge_set(g: &BufferGeometry) -> Vec<[[i64; 3]; 2]> {
        let p = &g.get_attribute("position").unwrap().array;
        let mut out: Vec<_> = p
            .chunks_exact(6)
            .map(|s| {
                let a = rounded_key(Vector3::new(s[0], s[1], s[2]));
                let b = rounded_key(Vector3::new(s[3], s[4], s[5]));
                if a <= b {
                    [a, b]
                } else {
                    [b, a]
                }
            })
            .collect();
        out.sort();
        out
    }

    const DEFAULT: f32 = std::f32::consts::PI / 180.0;

    #[test]
    fn a_box_has_twelve_edges_indexed_or_not() {
        let indexed = crate::geometries::BoxGeometry::new(1.0, 2.0, 3.0);
        let flat = unindexed(&indexed);
        let a = edge_set(&EdgesGeometry::new(&indexed, DEFAULT));
        let b = edge_set(&EdgesGeometry::new(&flat, DEFAULT));
        assert_eq!(a.len(), 12, "box vertices are split per face");
        assert_eq!(a, b);
        a.windows(2)
            .for_each(|w| assert_ne!(w[0], w[1], "no duplicates"));
    }

    #[test]
    fn a_sphere_gives_the_same_edges_indexed_or_not() {
        let indexed = crate::geometries::SphereGeometry::new(1.0, 8, 6);
        let flat = unindexed(&indexed);
        for threshold in [DEFAULT, 20f32.to_radians(), 60f32.to_radians()] {
            let a = edge_set(&EdgesGeometry::new(&indexed, threshold));
            let b = edge_set(&EdgesGeometry::new(&flat, threshold));
            assert_eq!(a, b, "threshold {threshold}");
        }
        // At 1 degree the rings and meridians are creases and the planar
        // quads' diagonals are not: fewer segments than triangle edges.
        let n = segments(&EdgesGeometry::new(&flat, DEFAULT));
        assert!(n > 0 && n < flat.triangle_count() * 3 / 2, "{n}");
    }

    #[test]
    fn the_fbx_cube_has_twelve_edges() {
        let mut g =
            crate::loaders::FbxLoader::parse(include_bytes!("../loaders/fbx/testdata/blender.fbx"))
                .expect("parses");
        assert!(g.index.is_none(), "the fixture is unindexed");
        assert_eq!(segments(&EdgesGeometry::new(&g, DEFAULT)), 12);
        // The flat diagonals stay out by angle alone, without the polygons.
        g.polygons = None;
        assert_eq!(segments(&EdgesGeometry::new(&g, DEFAULT)), 12);
    }

    #[test]
    fn an_open_boundary_is_still_drawn() {
        // A flat unit square as two unindexed triangles: four border edges,
        // and the shared diagonal is flat.
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(
                vec![
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, //
                    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                ],
                3,
            ),
        );
        assert_eq!(segments(&EdgesGeometry::new(&g, DEFAULT)), 4);
    }

    #[test]
    fn corners_merge_only_within_the_precision() {
        let square = |eps: f32| {
            let mut g = BufferGeometry::new();
            g.set_attribute(
                "position",
                BufferAttribute::new(
                    vec![
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        0.0, //
                        eps,
                        0.0,
                        0.0,
                        1.0 + eps,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                    ],
                    3,
                ),
            );
            segments(&EdgesGeometry::new(&g, DEFAULT))
        };
        assert_eq!(square(0.0), 4);
        assert_eq!(square(1e-6), 4, "below 1e-4: the same corner");
        assert_eq!(square(1e-3), 6, "above it: a crack, both sides drawn");
    }

    #[test]
    fn indexed_threshold_behaviour_is_unchanged() {
        // Two triangles folded 30 degrees along a shared indexed edge.
        let (s, c) = 30f32.to_radians().sin_cos();
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -c, s],
                3,
            ),
        );
        g.set_index(vec![0, 1, 2, 1, 0, 3]);
        assert_eq!(segments(&EdgesGeometry::new(&g, 20f32.to_radians())), 5);
        assert_eq!(segments(&EdgesGeometry::new(&g, 40f32.to_radians())), 4);
    }

    #[test]
    fn wireframe_draws_polygon_outlines() {
        assert_eq!(segments(&WireframeGeometry::new(&bent_quad(false))), 5);
        assert_eq!(segments(&WireframeGeometry::new(&bent_quad(true))), 4);
    }

    /// three.js dedupes wireframe edges by position, so two unindexed
    /// triangles sharing a diagonal draw five segments, not six, and an
    /// indexed copy with split seam vertices draws the same five.
    #[test]
    fn wireframe_dedupes_by_position_like_three() {
        let mut flat = BufferGeometry::new();
        flat.set_attribute(
            "position",
            BufferAttribute::new(
                vec![
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, //
                    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, -0.0, 1.0, 0.0,
                ],
                3,
            ),
        );
        let w = WireframeGeometry::new(&flat);
        assert_eq!(segments(&w), 5, "the shared diagonal once");
        // First-seen order, as three.js emits them: the first triangle's three
        // edges, then the second's two new ones.
        let p = &w.get_attribute("position").unwrap().array;
        assert_eq!(&p[..6], &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(&p[24..], &[-0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        // The same picture indexed, with the shared corners split per triangle.
        let mut split = flat.clone();
        split.set_index(vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(segments(&WireframeGeometry::new(&split)), 5);
        // A box: 12 edges plus a diagonal per face, its 24 vertices split per
        // face but sharing positions along every edge.
        let mut cube = crate::geometries::BoxGeometry::new(1.0, 1.0, 1.0);
        cube.polygons = None;
        assert_eq!(segments(&WireframeGeometry::new(&cube)), 18);
        assert_eq!(segments(&WireframeGeometry::new(&unindexed(&cube))), 18);
    }
}
