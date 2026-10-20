//! A quad from four arbitrary corners, meshed as a bilinear patch.
//!
//! [`PlaneGeometry`](super::PlaneGeometry) is always an axis-aligned rectangle
//! in XY. `QuadGeometry` takes any four corners in winding order and meshes
//!
//! ```text
//! P(u, v) = (1-u)(1-v)·p0 + u(1-v)·p1 + uv·p2 + (1-u)v·p3
//! ```
//!
//! so `p0..p3` sit at UV `(0,0)`, `(1,0)`, `(1,1)`, `(0,1)`. Those are the UVs
//! `PlaneGeometry` puts at its bottom-left, bottom-right, top-right and
//! top-left corners, and the vertex order, index pattern and normal side match
//! it too: passing a plane's corners counter-clockwise from bottom-left gives
//! the same buffers, so swapping one for the other flips nothing.
//!
//! Subdividing a non-planar quad samples the bilinear patch, so it comes out as
//! a smooth saddle rather than two folded triangles. Normals are the patch's
//! analytic `∂P/∂u × ∂P/∂v`, which points to the side the corners wind
//! counter-clockwise around (right-hand rule), like a triangle's.
//!
//! Each grid cell is split along the diagonal that folds least: the one whose
//! two triangles' normals agree best. On a convex planar cell both agree
//! exactly and the tie goes to the shorter diagonal, then to `p0–p2`, which is
//! `PlaneGeometry`'s. On a concave cell only the diagonal through the reflex
//! corner keeps both triangles front-facing, so that is the one taken; on a
//! skew cell it picks the split closer to the patch.
//!
//! Concave and bow-tie quads: the bilinear map of a concave quad folds over
//! near the reflex corner, so with segments > 1 the cells there come out
//! wound backwards. Unsubdivided, the least-fold split meshes a planar concave
//! quad correctly. A bow-tie (self-intersecting) quad has no consistent side:
//! it meshes as two overlapping halves facing opposite ways, and is not
//! supported beyond being finite. Use [`QuadGeometry::is_convex`] to check.

use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::Vector3;

pub struct QuadGeometry;

impl QuadGeometry {
    /// Two triangles over `corners`, given in winding order. See the module
    /// docs for the corner/UV mapping and which diagonal is used.
    pub fn new(corners: [Vector3; 4]) -> BufferGeometry {
        Self::with_segments(corners, 1, 1)
    }

    /// The bilinear patch over `corners`, sampled on a
    /// `width_segments × height_segments` grid (`u` along `p0→p1`, `v` along
    /// `p0→p3`). Zero segments are treated as one.
    pub fn with_segments(
        corners: [Vector3; 4],
        width_segments: usize,
        height_segments: usize,
    ) -> BufferGeometry {
        let grid_x = width_segments.max(1);
        let grid_y = height_segments.max(1);
        let grid_x1 = grid_x + 1;
        let grid_y1 = grid_y + 1;
        let area = area_vector(&corners);

        let mut points = Vec::with_capacity(grid_x1 * grid_y1);
        let mut positions = Vec::with_capacity(grid_x1 * grid_y1 * 3);
        let mut normals = Vec::with_capacity(grid_x1 * grid_y1 * 3);
        let mut uvs = Vec::with_capacity(grid_x1 * grid_y1 * 2);

        // Row 0 is v = 1, as in `PlaneGeometry`, so the buffers line up with it.
        for iy in 0..grid_y1 {
            let v = 1.0 - iy as f32 / grid_y as f32;
            for ix in 0..grid_x1 {
                let u = ix as f32 / grid_x as f32;
                let p = Self::point_at(&corners, u, v);
                let mut n = Self::normal_at(&corners, u, v);
                // Only a folded patch (concave or bow-tie) faces away from its
                // own area vector; flip so shading stays on one side.
                if n.dot(area) < 0.0 {
                    n = -n;
                }
                points.push(p);
                positions.extend_from_slice(&[p.x, p.y, p.z]);
                normals.extend_from_slice(&[n.x, n.y, n.z]);
                uvs.extend_from_slice(&[u, v]);
            }
        }

        let mut indices = Vec::with_capacity(grid_x * grid_y * 6);
        for iy in 0..grid_y {
            for ix in 0..grid_x {
                // a = top-left, b = bottom-left, c = bottom-right, d = top-right
                // in UV, i.e. cell corners p3, p0, p1, p2.
                let a = (ix + grid_x1 * iy) as u32;
                let b = (ix + grid_x1 * (iy + 1)) as u32;
                let c = (ix + 1 + grid_x1 * (iy + 1)) as u32;
                let d = (ix + 1 + grid_x1 * iy) as u32;
                let cell = [
                    points[b as usize],
                    points[c as usize],
                    points[d as usize],
                    points[a as usize],
                ];
                if split_along_02(&cell) {
                    indices.extend_from_slice(&[a, b, d, b, c, d]);
                } else {
                    indices.extend_from_slice(&[a, b, c, a, c, d]);
                }
            }
        }

        let mut geom = BufferGeometry::new();
        geom.set_attribute("position", BufferAttribute::new(positions, 3));
        geom.set_attribute("normal", BufferAttribute::new(normals, 3));
        geom.set_attribute("uv", BufferAttribute::new(uvs, 2));
        geom.set_index(indices);
        // Each cell is one quad: its two triangles are consecutive.
        geom.set_polygons(vec![2; grid_x * grid_y]);
        #[cfg(feature = "brep")]
        crate::brep::primitives::tag_quad(&mut geom, &corners);
        geom
    }

    /// The bilinear patch at `(u, v)`; `(0,0)` is `p0`, `(1,1)` is `p2`.
    pub fn point_at(corners: &[Vector3; 4], u: f32, v: f32) -> Vector3 {
        let [p0, p1, p2, p3] = *corners;
        p0 * ((1.0 - u) * (1.0 - v)) + p1 * (u * (1.0 - v)) + p2 * (u * v) + p3 * ((1.0 - u) * v)
    }

    /// Unit normal `∂P/∂u × ∂P/∂v` of the patch at `(u, v)`.
    ///
    /// Where that vanishes — a corner collapsed onto a neighbour zeroes `∂P/∂u`
    /// or `∂P/∂v` along a whole edge — it is taken a little way towards the
    /// centre instead (the limit from inside), then from the quad's area
    /// vector, then `+Z` if all four corners are collinear. Never NaN.
    pub fn normal_at(corners: &[Vector3; 4], u: f32, v: f32) -> Vector3 {
        let eps = degenerate_eps(corners);
        let n = partial_cross(corners, u, v);
        if n.length_sq() > eps {
            return n.normalize();
        }
        const NUDGE: f32 = 1e-2;
        let n = partial_cross(corners, u + (0.5 - u) * NUDGE, v + (0.5 - v) * NUDGE);
        if n.length_sq() > eps * NUDGE * NUDGE {
            return n.normalize();
        }
        let n = area_vector(corners);
        if n.length_sq() > eps {
            return n.normalize();
        }
        Vector3::new(0.0, 0.0, 1.0)
    }

    /// Whether all four corners lie within `tol` of a common plane.
    ///
    /// Measured as the distance of one corner from the plane of the other
    /// three, using the triple with the largest area. Collinear or coincident
    /// corners are planar.
    pub fn is_planar(corners: &[Vector3; 4], tol: f32) -> bool {
        let mut best = (0.0f32, Vector3::ZERO, 0usize);
        for skip in 0..4 {
            let [a, b, c] = others(corners, skip);
            let n = (b - a).cross(c - a);
            let len = n.length();
            if len > best.0 {
                best = (len, n, skip);
            }
        }
        let (len, n, skip) = best;
        if len <= degenerate_eps(corners).sqrt() {
            return true;
        }
        let [a, _, _] = others(corners, skip);
        ((corners[skip] - a).dot(n) / len).abs() <= tol
    }

    /// Whether the quad is strictly convex: every corner turns the same way
    /// about the area vector. Coincident corners, collinear edges and bow-ties
    /// are not convex. For a non-planar quad this tests its projection along
    /// the area vector.
    pub fn is_convex(corners: &[Vector3; 4]) -> bool {
        let n = area_vector(corners);
        let eps = degenerate_eps(corners);
        if n.length_sq() <= eps {
            return false;
        }
        (0..4).all(|i| {
            let prev = corners[(i + 3) % 4];
            let next = corners[(i + 1) % 4];
            let turn = (corners[i] - prev).cross(next - corners[i]);
            turn.dot(n) > eps.sqrt() * n.length()
        })
    }
}

/// `∂P/∂u × ∂P/∂v`, unnormalised.
fn partial_cross(corners: &[Vector3; 4], u: f32, v: f32) -> Vector3 {
    let [p0, p1, p2, p3] = *corners;
    let du = (p1 - p0) * (1.0 - v) + (p2 - p3) * v;
    let dv = (p3 - p0) * (1.0 - u) + (p2 - p1) * u;
    du.cross(dv)
}

/// Twice the quad's vector area (Newell): `(p2 - p0) × (p3 - p1)`. Points to
/// the counter-clockwise side; zero for a fully degenerate or symmetric bow-tie.
fn area_vector(corners: &[Vector3; 4]) -> Vector3 {
    (corners[2] - corners[0]).cross(corners[3] - corners[1])
}

/// A squared-length threshold for cross products below which a normal is
/// treated as zero, relative to the quad's size so units don't matter.
fn degenerate_eps(corners: &[Vector3; 4]) -> f32 {
    let mut scale = 0.0f32;
    for i in 0..4 {
        scale = scale.max(corners[i].distance_to(corners[(i + 2) % 4]));
        scale = scale.max(corners[i].distance_to(corners[(i + 1) % 4]));
    }
    let s2 = scale * scale;
    (s2 * 1e-6) * (s2 * 1e-6)
}

fn others(corners: &[Vector3; 4], skip: usize) -> [Vector3; 3] {
    [
        corners[(skip + 1) % 4],
        corners[(skip + 2) % 4],
        corners[(skip + 3) % 4],
    ]
}

/// Whether to split `cell` (corners in winding order) along `p0–p2` rather
/// than `p1–p3`: the diagonal whose two triangles' unit normals agree more,
/// then the shorter one, then `p0–p2`.
fn split_along_02(cell: &[Vector3; 4]) -> bool {
    let [p0, p1, p2, p3] = *cell;
    let fold = |a: Vector3, b: Vector3| a.normalize().dot(b.normalize());
    let agree_02 = fold((p1 - p0).cross(p2 - p0), (p2 - p0).cross(p3 - p0));
    let agree_13 = fold((p2 - p1).cross(p3 - p1), (p3 - p1).cross(p0 - p1));
    const EPS: f32 = 1e-5;
    if agree_13 > agree_02 + EPS {
        return false;
    }
    if agree_02 > agree_13 + EPS {
        return true;
    }
    let d02 = p0.distance_to(p2);
    let d13 = p1.distance_to(p3);
    d13 >= d02 * (1.0 - EPS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometries::PlaneGeometry;

    fn v(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3::new(x, y, z)
    }

    fn attr(g: &BufferGeometry, name: &str) -> Vec<f32> {
        g.get_attribute(name).expect(name).array.clone()
    }

    fn vec3s(g: &BufferGeometry, name: &str) -> Vec<Vector3> {
        attr(g, name)
            .chunks(3)
            .map(|c| v(c[0], c[1], c[2]))
            .collect()
    }

    fn close(a: Vector3, b: Vector3, tol: f32) -> bool {
        (a - b).length() <= tol
    }

    /// `PlaneGeometry::new(w, h)`'s corners counter-clockwise from bottom-left.
    fn plane_corners(w: f32, h: f32) -> [Vector3; 4] {
        [
            v(-w / 2.0, -h / 2.0, 0.0),
            v(w / 2.0, -h / 2.0, 0.0),
            v(w / 2.0, h / 2.0, 0.0),
            v(-w / 2.0, h / 2.0, 0.0),
        ]
    }

    fn saddle() -> [Vector3; 4] {
        [
            v(0.0, 0.0, 0.0),
            v(1.0, 0.0, 0.5),
            v(1.0, 1.0, 0.0),
            v(0.0, 1.0, 0.5),
        ]
    }

    #[test]
    fn corners_and_uvs_are_exact() {
        let c = [
            v(0.3, -1.2, 0.7),
            v(2.1, -0.4, 0.1),
            v(1.8, 1.9, -0.6),
            v(-0.5, 1.1, 0.2),
        ];
        let g = QuadGeometry::with_segments(c, 3, 2);
        let pos = vec3s(&g, "position");
        let uv = attr(&g, "uv");
        let row = 4;
        // Grid index of each corner: row 0 is v = 1.
        let at = [2 * row, 2 * row + 3, 3, 0];
        let want_uv = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        for k in 0..4 {
            assert_eq!(pos[at[k]], c[k], "corner {k}");
            assert_eq!((uv[at[k] * 2], uv[at[k] * 2 + 1]), want_uv[k], "uv {k}");
        }
    }

    #[test]
    fn index_count_is_six_per_cell() {
        for (sx, sy) in [(1, 1), (1, 5), (4, 3), (7, 7)] {
            let g = QuadGeometry::with_segments(saddle(), sx, sy);
            assert_eq!(g.index.as_ref().unwrap().len(), sx * sy * 6);
            assert_eq!(
                g.get_attribute("position").unwrap().count(),
                (sx + 1) * (sy + 1)
            );
        }
        // One quad polygon per cell, whichever diagonal each was split on.
        let g = QuadGeometry::with_segments(saddle(), 3, 2);
        assert_eq!(g.polygon_counts(), Some(&[2u32; 6][..]));
        let tris = g.polygon_ranges().next().unwrap();
        assert_eq!(g.polygon_loop(tris).map(|l| l.len()), Some(4));
        // Zero segments clamp to one, like `PlaneGeometry`.
        let g = QuadGeometry::with_segments(saddle(), 0, 0);
        assert_eq!(g.index.as_ref().unwrap().len(), 6);
    }

    #[test]
    fn matches_plane_geometry_on_its_corners() {
        for (sx, sy) in [(1, 1), (3, 2), (5, 4)] {
            let q = QuadGeometry::with_segments(plane_corners(3.0, 2.2), sx, sy);
            let p = PlaneGeometry::with_segments(3.0, 2.2, sx, sy);
            for name in ["position", "normal", "uv"] {
                let (a, b) = (attr(&q, name), attr(&p, name));
                assert_eq!(a.len(), b.len(), "{name}");
                for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                    assert!((x - y).abs() < 1e-5, "{name}[{i}] = {x}, plane {y}");
                }
            }
            assert_eq!(q.index, p.index, "index {sx}x{sy}");
        }
    }

    #[test]
    fn planar_normals_equal_face_normal_and_follow_winding() {
        // A tilted, non-rectangular planar quad.
        let c = [
            v(0.0, 0.0, 0.0),
            v(2.0, 0.0, 1.0),
            v(2.5, 1.5, 1.25),
            v(-0.5, 1.0, -0.25),
        ];
        assert!(QuadGeometry::is_planar(&c, 1e-5));
        let face = (c[1] - c[0]).cross(c[3] - c[0]).normalize();
        let g = QuadGeometry::with_segments(c, 4, 3);
        for n in vec3s(&g, "normal") {
            assert!(close(n, face, 1e-5), "{n:?} vs {face:?}");
        }
        // Every triangle, as indexed, winds the same way as the normals.
        let pos = vec3s(&g, "position");
        for t in g.index.as_ref().unwrap().chunks(3) {
            let (a, b, d) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
            assert!((b - a).cross(d - a).dot(face) > 0.0);
        }
        // Reversed winding flips the normal.
        let r = QuadGeometry::new([c[0], c[3], c[2], c[1]]);
        for n in vec3s(&r, "normal") {
            assert!(close(n, -face, 1e-5));
        }
    }

    #[test]
    fn saddle_normals_are_unit_and_vary_smoothly() {
        let n = 16;
        let g = QuadGeometry::with_segments(saddle(), n, n);
        let normals = vec3s(&g, "normal");
        for m in &normals {
            assert!((m.length() - 1.0).abs() < 1e-5);
        }
        // Not planar, so the normals must actually vary...
        let spread = normals
            .iter()
            .map(|m| m.dot(normals[0]))
            .fold(1.0f32, f32::min);
        assert!(spread < 0.9, "saddle normals barely vary: {spread}");
        // ...but neighbours differ by a small angle, shrinking with the grid.
        let w = n + 1;
        for iy in 0..w {
            for ix in 0..w {
                let here = normals[iy * w + ix];
                if ix + 1 < w {
                    assert!(here.dot(normals[iy * w + ix + 1]) > 0.99);
                }
                if iy + 1 < w {
                    assert!(here.dot(normals[(iy + 1) * w + ix]) > 0.99);
                }
            }
        }
        // And they are the patch's, not some average: check against finite
        // differences of `point_at` at an interior vertex.
        let (u, vv) = (0.25, 0.625);
        let h = 1e-3;
        let du = QuadGeometry::point_at(&saddle(), u + h, vv)
            - QuadGeometry::point_at(&saddle(), u - h, vv);
        let dv = QuadGeometry::point_at(&saddle(), u, vv + h)
            - QuadGeometry::point_at(&saddle(), u, vv - h);
        let fd = du.cross(dv).normalize();
        let idx = (n - (vv * n as f32) as usize) * w + (u * n as f32) as usize;
        assert!(
            close(normals[idx], fd, 1e-3),
            "{:?} vs {fd:?}",
            normals[idx]
        );
    }

    #[test]
    fn degenerate_corners_give_no_nan() {
        let cases: [[Vector3; 4]; 4] = [
            // Triangle-like: p2 == p3.
            [
                v(0.0, 0.0, 0.0),
                v(1.0, 0.0, 0.0),
                v(0.5, 1.0, 0.0),
                v(0.5, 1.0, 0.0),
            ],
            // Triangle-like, non-planar neighbour.
            [
                v(0.0, 0.0, 0.0),
                v(1.0, 0.0, 0.3),
                v(1.0, 1.0, 0.0),
                v(0.0, 0.0, 0.0),
            ],
            // All collinear.
            [
                v(0.0, 0.0, 0.0),
                v(1.0, 0.0, 0.0),
                v(2.0, 0.0, 0.0),
                v(3.0, 0.0, 0.0),
            ],
            // All coincident.
            [v(1.0, 2.0, 3.0); 4],
        ];
        for (k, c) in cases.iter().enumerate() {
            for segs in [1, 4] {
                let g = QuadGeometry::with_segments(*c, segs, segs);
                for name in ["position", "normal", "uv"] {
                    assert!(
                        attr(&g, name).iter().all(|x| x.is_finite()),
                        "case {k} {name}"
                    );
                }
                for m in vec3s(&g, "normal") {
                    assert!((m.length() - 1.0).abs() < 1e-4, "case {k}: {m:?}");
                }
            }
        }
        // The planar triangle-like quad keeps its face normal everywhere,
        // including along the collapsed edge.
        let g = QuadGeometry::with_segments(cases[0], 4, 4);
        for m in vec3s(&g, "normal") {
            assert!(close(m, v(0.0, 0.0, 1.0), 1e-4), "{m:?}");
        }
    }

    #[test]
    fn diagonal_choice() {
        // Planar rhombus: both splits are flat, so the shorter diagonal wins.
        let rhombus = [
            v(0.0, -1.0, 0.0),
            v(3.0, 0.0, 0.0),
            v(0.0, 1.0, 0.0),
            v(-3.0, 0.0, 0.0),
        ];
        let idx = QuadGeometry::new(rhombus).index.unwrap();
        // Vertex order: 0 = p3, 1 = p2, 2 = p0, 3 = p1. Shorter is p0–p2.
        assert_eq!(idx, vec![0, 2, 1, 2, 3, 1]);
        let rhombus = [rhombus[1], rhombus[2], rhombus[3], rhombus[0]];
        let idx = QuadGeometry::new(rhombus).index.unwrap();
        assert_eq!(idx, vec![0, 2, 3, 0, 3, 1], "shorter is now p1–p3");

        // Concave dart, reflex corner p3: the short p0–p2 diagonal lies
        // outside, so the split must go through p3 and both triangles face +Z.
        let dart = [
            v(0.0, 0.0, 0.0),
            v(10.0, 0.5, 0.0),
            v(0.0, 1.0, 0.0),
            v(0.5, 0.5, 0.0),
        ];
        assert!(!QuadGeometry::is_convex(&dart));
        let g = QuadGeometry::new(dart);
        let pos = vec3s(&g, "position");
        for t in g.index.as_ref().unwrap().chunks(3) {
            let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
            assert!((b - a).cross(c - a).z > 0.0);
        }
        for m in vec3s(&g, "normal") {
            assert!(close(m, v(0.0, 0.0, 1.0), 1e-5), "{m:?}");
        }
    }

    #[test]
    fn planarity_and_convexity() {
        assert!(QuadGeometry::is_planar(&plane_corners(2.0, 1.0), 1e-6));
        assert!(QuadGeometry::is_convex(&plane_corners(2.0, 1.0)));
        assert!(!QuadGeometry::is_planar(&saddle(), 0.1));
        // Each corner sits 0.82 off the plane of the other three.
        assert!(QuadGeometry::is_planar(&saddle(), 0.85));
        let bowtie = [
            v(0.0, 0.0, 0.0),
            v(1.0, 1.0, 0.0),
            v(1.0, 0.0, 0.0),
            v(0.0, 1.0, 0.0),
        ];
        assert!(QuadGeometry::is_planar(&bowtie, 1e-6));
        assert!(!QuadGeometry::is_convex(&bowtie));
        let g = QuadGeometry::with_segments(bowtie, 3, 3);
        assert!(attr(&g, "normal").iter().all(|x| x.is_finite()));
        // Clockwise winding is still convex: it is judged against its own
        // area vector.
        let c = plane_corners(2.0, 1.0);
        assert!(QuadGeometry::is_convex(&[c[3], c[2], c[1], c[0]]));
    }

    /// Ray-cast truth: rays through `point_at(u, v)` along the patch normal
    /// hit the subdivided mesh at that point, and rays outside miss.
    #[test]
    fn raycaster_hits_the_patch_where_expected() {
        use crate::core::{Mesh, Object3D, ObjectArena, Raycaster};
        use crate::materials::{BasicMaterial, Material};
        use crate::math::{Color, Matrix4, Ray};

        let c = [
            v(-1.0, -1.0, 0.0),
            v(1.5, -1.0, 0.4),
            v(1.0, 1.0, 0.0),
            v(-1.0, 1.2, 0.3),
        ];
        let n = 24;
        let mut geom = QuadGeometry::with_segments(c, n, n);
        geom.compute_bounding_sphere();
        let mesh = Mesh::new(
            geom,
            Material::Basic(BasicMaterial::new(Color::from_hex(0xffffff))),
        );
        let mut arena = ObjectArena::new();
        let root = arena.insert(Object3D::group());
        let id = arena.insert(Object3D::mesh(mesh));
        arena.add_child(root, id);
        arena.update_world_matrices(root, Matrix4::identity());

        for &(u, vv) in &[(0.5, 0.5), (0.1, 0.8), (0.9, 0.2), (0.33, 0.66)] {
            let p = QuadGeometry::point_at(&c, u, vv);
            let nrm = QuadGeometry::normal_at(&c, u, vv);
            let rc = Raycaster {
                ray: Ray::new(p + nrm * 2.0, -nrm),
                ..Raycaster::default()
            };
            let hits = rc.intersect_objects(&arena, root, true);
            assert!(!hits.is_empty(), "no hit at ({u}, {vv})");
            // Flat facets sit within the patch's sagitta of it; at 24 segments
            // that is well under a millimetre on this two-unit quad.
            assert!(
                hits[0].point.distance_to(p) < 2e-3,
                "({u}, {vv}): hit {:?}, patch {p:?}",
                hits[0].point
            );
        }
        let rc = Raycaster {
            ray: Ray::new(v(3.0, 0.0, 2.0), v(0.0, 0.0, -1.0)),
            ..Raycaster::default()
        };
        assert!(rc.intersect_objects(&arena, root, true).is_empty());
    }
}
