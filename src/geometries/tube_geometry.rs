use crate::core::{BufferAttribute, BufferGeometry};
use crate::curves::Curve3;
use crate::math::Vector3;
use super::frenet::compute_frenet_frames;

pub struct TubeGeometry;

impl TubeGeometry {
    /// Extrude a circular cross-section of `radius` along `path`, with
    /// three.js's Frenet frames (`Curve.computeFrenetFrames`).
    pub fn new(
        path: &dyn Curve3,
        tubular_segments: usize,
        radius: f32,
        radial_segments: usize,
        closed: bool,
    ) -> BufferGeometry {
        let ts = tubular_segments.max(2);
        let to64 = |v: Vector3| [v.x as f64, v.y as f64, v.z as f64];
        let points: Vec<[f64; 3]> = path.get_points(ts).into_iter().map(to64).collect();
        let tangents: Vec<[f64; 3]> = (0..=ts)
            .map(|i| to64(path.get_tangent(i as f32 / ts as f32).normalize()))
            .collect();
        Self::from_samples(&points, &tangents, radius as f64, radial_segments, closed)
    }

    /// three.js r165 `new TubeGeometry(path, tubularSegments, radius,
    /// radialSegments, closed)` over a three.js-exact curve — any
    /// [`ThreeCurve`](crate::curves::ThreeCurve), a JavaScript `Curve`
    /// subclass included. Returns the geometry and the Frenet frames three.js
    /// keeps on it (`tangents`, `normals`, `binormals`).
    pub fn from_curve(
        path: &crate::curves::ThreeCurve,
        tubular_segments: usize,
        radius: f64,
        radial_segments: usize,
        closed: bool,
    ) -> (BufferGeometry, super::frenet::FrenetFrames) {
        let frames = path.compute_frenet_frames(tubular_segments, closed);
        let points: Vec<[f64; 3]> = (0..=tubular_segments)
            .map(|i| {
                path.get_point_at(i as f64 / tubular_segments as f64)
                    .unwrap_or([f64::NAN; 3])
            })
            .collect();
        let g = Self::from_samples(&points, &frames.tangents, radius, radial_segments, closed);
        (g, frames)
    }

    /// three.js r165 `TubeGeometry` from a curve the caller has sampled:
    /// `points[i] = path.getPointAt(i / tubularSegments)` and
    /// `tangents[i] = path.getTangentAt(i / tubularSegments)` for `i` in
    /// `0..=tubularSegments`. The frames are built here, exactly as
    /// `computeFrenetFrames` builds them, and the arithmetic is `f64` as in
    /// JavaScript, so the buffers match r165's.
    ///
    /// Sampling is the caller's because the curve may be a JavaScript
    /// `Curve` subclass this side cannot evaluate.
    pub fn from_samples(
        points: &[[f64; 3]],
        tangents: &[[f64; 3]],
        radius: f64,
        radial_segments: usize,
        closed: bool,
    ) -> BufferGeometry {
        let n = points.len().min(tangents.len());
        if n < 2 {
            return BufferGeometry::new();
        }
        let ts = n - 1;
        let rs = radial_segments.max(1);
        let frames = compute_frenet_frames(&tangents[..n], closed);

        let mut positions = Vec::with_capacity((ts + 1) * (rs + 1) * 3);
        let mut normals = Vec::with_capacity((ts + 1) * (rs + 1) * 3);
        let mut uvs = Vec::with_capacity((ts + 1) * (rs + 1) * 2);
        let mut indices = Vec::new();

        // Rings 0..ts, then the last one: at the end of the curve when open,
        // a copy of the first when closed (`generateSegment(0)`).
        for ring in 0..=ts {
            let i = if ring == ts && closed { 0 } else { ring };
            let p = points[i];
            let (nn, bb) = (frames.normals[i], frames.binormals[i]);
            for j in 0..=rs {
                let v = j as f64 / rs as f64 * std::f64::consts::PI * 2.0;
                let sin = v.sin();
                let cos = -v.cos();
                let mut nx = cos * nn[0] + sin * bb[0];
                let mut ny = cos * nn[1] + sin * bb[1];
                let mut nz = cos * nn[2] + sin * bb[2];
                // `normal.normalize()`: a multiply by the reciprocal.
                let l = (nx * nx + ny * ny + nz * nz).sqrt();
                let inv = 1.0 / if l == 0.0 { 1.0 } else { l };
                nx *= inv;
                ny *= inv;
                nz *= inv;
                normals.extend_from_slice(&[nx as f32, ny as f32, nz as f32]);
                positions.extend_from_slice(&[
                    (p[0] + radius * nx) as f32,
                    (p[1] + radius * ny) as f32,
                    (p[2] + radius * nz) as f32,
                ]);
            }
        }
        for i in 0..=ts {
            for j in 0..=rs {
                uvs.extend_from_slice(&[(i as f64 / ts as f64) as f32, (j as f64 / rs as f64) as f32]);
            }
        }

        let row = (rs + 1) as u32;
        let mut polygons = super::polygons::CellPolygons::new();
        for i in 0..ts as u32 {
            for j in 0..rs as u32 {
                let a = i * row + j;
                let b = (i + 1) * row + j;
                let c = (i + 1) * row + j + 1;
                let d = i * row + j + 1;
                indices.extend_from_slice(&[a, b, d]);
                indices.extend_from_slice(&[b, c, d]);
                polygons.quad(&positions, [a, b, c, d]);
            }
        }

        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_attribute("normal", BufferAttribute::new(normals, 3));
        g.set_attribute("uv", BufferAttribute::new(uvs, 2));
        g.set_index(indices);
        polygons.apply(&mut g);
        #[cfg(feature = "brep")]
        {
            let v3 = |a: [f64; 3]| Vector3::new(a[0] as f32, a[1] as f32, a[2] as f32);
            let ring = |i: usize| if i == ts && closed { 0 } else { i };
            let centers: Vec<Vector3> = (0..=ts).map(|i| v3(points[ring(i)])).collect();
            let fr: Vec<(Vector3, Vector3)> = (0..=ts)
                .map(|i| (v3(frames.normals[ring(i)]), v3(frames.binormals[ring(i)])))
                .collect();
            crate::brep::primitives::tag_tube(&mut g, &centers, &fr, radius as f32);
        }
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curves::{CatmullRomCurve3, LineCurve3};
    use std::collections::HashMap;

    fn positions(g: &BufferGeometry) -> &[f32] {
        &g.get_attribute("position").unwrap().array
    }

    /// Edges not shared by exactly two triangles, after welding by position.
    fn open_edges(g: &BufferGeometry) -> usize {
        let pos = positions(g);
        let mut ids: HashMap<[i64; 3], usize> = HashMap::new();
        let welded: Vec<usize> = pos
            .chunks_exact(3)
            .map(|p| {
                let n = ids.len();
                *ids.entry([0, 1, 2].map(|k| (p[k] * 1e4).round() as i64)).or_insert(n)
            })
            .collect();
        let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
        for t in g.index.as_deref().unwrap().chunks_exact(3) {
            let w = [0, 1, 2].map(|k| welded[t[k] as usize]);
            for k in 0..3 {
                let (a, b) = (w[k], w[(k + 1) % 3]);
                *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        edges.values().filter(|&&n| n != 2).count()
    }

    #[test]
    fn every_tubular_segment_is_emitted() {
        // three.js r165: tubularSegments + 1 rings, tubularSegments * radialSegments
        // quads, open or closed.
        let path = LineCurve3::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 4.0));
        let (ts, rs) = (8, 6);
        for closed in [false, true] {
            let g = TubeGeometry::new(&path, ts, 0.25, rs, closed);
            assert_eq!(positions(&g).len() / 3, (ts + 1) * (rs + 1), "closed={closed}");
            assert_eq!(g.index.as_deref().unwrap().len(), ts * rs * 6, "closed={closed}");
        }
    }

    #[test]
    fn open_tube_reaches_the_end_of_the_curve() {
        let path = LineCurve3::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 4.0));
        let g = TubeGeometry::new(&path, 8, 0.25, 6, false);
        let pos = positions(&g);
        let idx = g.index.as_deref().unwrap();
        let max_z = idx.iter().map(|&v| pos[v as usize * 3 + 2]).fold(f32::MIN, f32::max);
        assert!((max_z - 4.0).abs() < 1e-5, "indexed tube ends at z = {max_z}");
        // An open tube's only boundary is its two end circles.
        assert_eq!(open_edges(&g), 2 * 6);
    }

    #[test]
    fn closed_tube_is_watertight() {
        let mut path = CatmullRomCurve3::new(vec![
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(0.0, 0.5, 2.0),
            Vector3::new(-2.0, 0.0, 0.0),
            Vector3::new(0.0, -0.5, -2.0),
        ]);
        path.closed = true;
        let (ts, rs) = (24, 8);
        let g = TubeGeometry::new(&path, ts, 0.3, rs, true);
        let pos = positions(&g);
        // The last ring is a copy of the first.
        let row = (rs + 1) * 3;
        assert_eq!(&pos[ts * row..(ts + 1) * row], &pos[..row]);
        assert_eq!(open_edges(&g), 0);
    }
}
