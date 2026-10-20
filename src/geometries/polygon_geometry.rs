//! Filled and extruded polygons from 2D rings — building footprints, regions
//! and countries.

use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::{polygon_points, triangulate_polygon};

/// A polygon with holes lying on the ground, optionally extruded upward.
///
/// 2D point `(u, v)` becomes `(u, y, -v)`: Y is up, and +v ("north") points
/// toward −Z, the top of the screen for a [`MapControls`](crate::MapControls)
/// camera at bearing 0. Rings may wind either way and may repeat their first
/// point, as GeoJSON's do.
///
/// With `height` 0 it is the fill alone, facing up. With a height it is a
/// closed solid: the fill at `base + height`, the fill again at `base`
/// facing down, and flat-shaded walls around the outline and every hole.
///
/// ```
/// use threers::PolygonGeometry;
/// let footprint = [[0.0, 0.0], [10.0, 0.0], [10.0, 6.0], [0.0, 6.0]];
/// let building = PolygonGeometry::new(&footprint, &[], 0.0, 25.0);
/// assert_eq!(building.triangle_count(), 2 + 2 + 4 * 2);
/// ```
pub struct PolygonGeometry;

impl PolygonGeometry {
    pub fn new(
        outer: &[[f64; 2]],
        holes: &[Vec<[f64; 2]>],
        base: f32,
        height: f32,
    ) -> BufferGeometry {
        let tris = triangulate_polygon(outer, holes);
        let points = polygon_points(outer, holes);
        let mut pos: Vec<f32> = Vec::new();
        let mut nrm: Vec<f32> = Vec::new();
        let to3 = |p: [f64; 2], y: f32| [p[0] as f32, y, -(p[1] as f32)];
        let mut push_tri = |a: [f32; 3], b: [f32; 3], c: [f32; 3], n: [f32; 3]| {
            // Wind each triangle to face along its normal, whatever the
            // ring's winding was.
            let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let facing = cross[0] * n[0] + cross[1] * n[1] + cross[2] * n[2];
            let (b, c) = if facing < 0.0 { (c, b) } else { (b, c) };
            for v in [a, b, c] {
                pos.extend_from_slice(&v);
                nrm.extend_from_slice(&n);
            }
        };
        let top = base + height;
        for t in &tris {
            let [a, b, c] = t.map(|i| to3(points[i as usize], top));
            push_tri(a, b, c, [0.0, 1.0, 0.0]);
        }
        if height != 0.0 {
            for t in &tris {
                let [a, b, c] = t.map(|i| to3(points[i as usize], base));
                push_tri(a, b, c, [0.0, -1.0, 0.0]);
            }
            let rings =
                std::iter::once((outer, true)).chain(holes.iter().map(|h| (h.as_slice(), false)));
            for (ring, is_outer) in rings {
                let ring = crate::math::open_ring(ring);
                if ring.len() < 2 {
                    continue;
                }
                let area: f64 = (0..ring.len())
                    .map(|i| {
                        let (p, q) = (ring[i], ring[(i + 1) % ring.len()]);
                        p[0] * q[1] - q[0] * p[1]
                    })
                    .sum();
                // Away from the solid: to the right of a counter-clockwise
                // outline, to the left of a counter-clockwise hole.
                let right = (area > 0.0) == is_outer;
                for i in 0..ring.len() {
                    let (p, q) = (ring[i], ring[(i + 1) % ring.len()]);
                    let (du, dv) = (q[0] - p[0], q[1] - p[1]);
                    let len = (du * du + dv * dv).sqrt();
                    if len == 0.0 {
                        continue;
                    }
                    let (nu, nv) = if right {
                        (dv / len, -du / len)
                    } else {
                        (-dv / len, du / len)
                    };
                    let n = [nu as f32, 0.0, -(nv as f32)];
                    let (p0, q0, p1, q1) = (to3(p, base), to3(q, base), to3(p, top), to3(q, top));
                    push_tri(p0, q0, q1, n);
                    push_tri(p0, q1, p1, n);
                }
            }
        }
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(pos, 3));
        g.set_attribute("normal", BufferAttribute::new(nrm, 3));
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri_normals(g: &BufferGeometry) -> Vec<([f32; 3], [f32; 3])> {
        let p = &g.get_attribute("position").unwrap().array;
        let n = &g.get_attribute("normal").unwrap().array;
        (0..p.len() / 9)
            .map(|t| {
                let v = |k: usize| [p[t * 9 + k * 3], p[t * 9 + k * 3 + 1], p[t * 9 + k * 3 + 2]];
                let (a, b, c) = (v(0), v(1), v(2));
                let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let cross = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                (cross, [n[t * 9], n[t * 9 + 1], n[t * 9 + 2]])
            })
            .collect()
    }

    #[test]
    fn every_face_winds_and_points_outward() {
        // Clockwise outline with a counter-clockwise hole: both windings that
        // can go wrong.
        let outer = [[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0], [0.0, 0.0]];
        let hole = vec![[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
        let g = PolygonGeometry::new(&outer, &[hole], 0.0, 2.0);
        let p = &g.get_attribute("position").unwrap().array;
        for (t, (cross, n)) in tri_normals(&g).into_iter().enumerate() {
            let dot = cross[0] * n[0] + cross[1] * n[1] + cross[2] * n[2];
            assert!(dot > 0.0, "triangle {t} winds against its normal");
            if n[1] == 0.0 {
                // A wall: its normal points away from the solid. The solid is
                // the ring between |x-2| or |z+2| in 1..2, so outward is away
                // from the centre on the outline and toward it on the hole.
                let cx = (p[t * 9] + p[t * 9 + 3] + p[t * 9 + 6]) / 3.0 - 2.0;
                let cz = (p[t * 9 + 2] + p[t * 9 + 5] + p[t * 9 + 8]) / 3.0 + 2.0;
                let radial = cx * n[0] + cz * n[2];
                let on_outline = cx.abs().max(cz.abs()) > 1.5;
                assert_eq!(radial > 0.0, on_outline, "wall {t} faces the wrong way");
            }
        }
        // 8 fill triangles each side, 4 + 4 walls of two triangles.
        assert_eq!(g.triangle_count(), 8 * 2 + 8 * 2);
    }

    #[test]
    fn flat_fill_faces_up() {
        let g = PolygonGeometry::new(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], &[], 0.5, 0.0);
        assert_eq!(g.triangle_count(), 1);
        let y = &g.get_attribute("position").unwrap().array;
        assert!(y.chunks(3).all(|v| v[1] == 0.5));
        assert!(tri_normals(&g)
            .iter()
            .all(|(c, n)| c[1] > 0.0 && n[1] == 1.0));
    }
}
