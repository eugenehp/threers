//! Triangulated polygon layers as plain data, built on any thread.

use super::geo::{FlatGround, GroundProjection};
use super::PolygonFeature;
use crate::math::{polygon_points, triangulate_polygon};

/// The mesh a [`PolygonMarks`](super::PolygonMarks) draws: every feature's
/// roof and walls, triangulated, with nothing GPU-side in it.
///
/// Building is the expensive part — ear clipping, walls, and on a globe the
/// subdivision that bends large polygons over the surface — so it is kept
/// apart from the layer: build it on a worker thread (it is `Send`), or in
/// a Web Worker and pass [`to_bytes`](Self::to_bytes) across, then hand it
/// to [`PolygonMarks::set_mesh`](super::PolygonMarks::set_mesh), which only
/// uploads.
///
/// ```
/// use threers::marks::{PolygonFeature, PolygonMarks, PolygonMesh};
/// let square = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
/// let mesh = std::thread::spawn(move || {
///     PolygonMesh::build(&[PolygonFeature::new(square)], 0.0)
/// })
/// .join()
/// .unwrap();
/// let bytes = mesh.to_bytes();
/// let mut layer = PolygonMarks::new();
/// layer.set_mesh(PolygonMesh::from_bytes(&bytes).unwrap());
/// assert_eq!(layer.feature_count(), 1);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PolygonMesh {
    /// What `positions` are relative to.
    pub origin: [f64; 3],
    pub positions: Vec<[f32; 3]>,
    /// Normal, and 1 where the vertex rises with its feature's elevation.
    pub normal_top: Vec<[f32; 4]>,
    /// The feature each vertex belongs to.
    pub feature: Vec<u32>,
    /// Each vertex's point in its feature — outer ring then holes, closing
    /// points dropped — or `u32::MAX` for one added to follow a curved
    /// ground: what [`PolygonMarks::set_point_colors`](super::PolygonMarks::set_point_colors)
    /// reads.
    pub source: Vec<u32>,
    pub index: Vec<u32>,
    /// Each feature's vertices, `start..end`.
    pub ranges: Vec<(u32, u32)>,
    /// The centre extrusion rises away from, for a mesh on a sphere.
    pub radial: Option<[f64; 3]>,
}

const MAGIC: &[u8; 4] = b"TPM2";

impl PolygonMesh {
    /// Triangulate `features` on the flat ground, at height `base`.
    pub fn build(features: &[PolygonFeature], base: f64) -> Self {
        Self::build_with(features, base, &FlatGround)
    }

    /// Triangulate `features`, whose rings are in `projection`'s ground
    /// coordinates, at height `base`. A projection with a
    /// [`max_edge`](GroundProjection::max_edge) has its rings densified and
    /// its roofs subdivided to it, so they follow a curved surface; edges are
    /// split by their own length alone, so neighbouring triangles split a
    /// shared edge identically and leave no cracks.
    pub fn build_with(
        features: &[PolygonFeature],
        base: f64,
        projection: &dyn GroundProjection,
    ) -> Self {
        let max_edge = projection.max_edge();
        let centre = projection.centre();
        let up_at = |w: [f64; 3]| -> [f64; 3] {
            match centre {
                None => [0.0, 1.0, 0.0],
                Some(c) => normalize([w[0] - c[0], w[1] - c[1], w[2] - c[2]]),
            }
        };
        let mut pos: Vec<[f64; 3]> = Vec::new();
        let mut nt: Vec<[f32; 4]> = Vec::new();
        let mut feature: Vec<u32> = Vec::new();
        let mut source: Vec<u32> = Vec::new();
        let mut index: Vec<u32> = Vec::new();
        // Walls are drawn before roofs, all features' walls first, as
        // deck.gl draws its side model before its top one: of translucent
        // buildings, that order decides which faces blend.
        let mut roof_index: Vec<u32> = Vec::new();
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        for (k, f) in features.iter().enumerate() {
            let start = pos.len();
            let Some(&anchor) = f.outer.first() else {
                ranges.push((start as u32, start as u32));
                continue;
            };
            // Ear clipping on coordinates near the anchor: map coordinates in
            // the millions square to numbers whose low bits are noise.
            let local = |r: &[[f64; 2]]| -> Vec<[f64; 2]> {
                let open = crate::math::open_ring(r);
                let pts: Vec<[f64; 2]> = open
                    .iter()
                    .map(|p| [p[0] - anchor[0], p[1] - anchor[1]])
                    .collect();
                match max_edge {
                    Some(m) => densify(&pts, m),
                    None => pts,
                }
            };
            let outer = local(&f.outer);
            let holes: Vec<Vec<[f64; 2]>> = f.holes.iter().map(|h| local(h)).collect();
            // Given triangles index the points as they are, so only stand
            // where no points were added.
            let tris = match (&f.triangles, max_edge) {
                (Some(t), None) => t.clone(),
                _ => triangulate_polygon(&outer, &holes),
            };
            let ground = |p: [f64; 2]| [p[0] + anchor[0], p[1] + anchor[1]];

            // Roof: on a curved ground, triangles subdivided to the maximum
            // edge, each vertex facing out of the surface.
            let points = polygon_points(&outer, &holes);
            let mut roof: Vec<[[f64; 2]; 3]> =
                tris.iter().map(|t| t.map(|i| points[i as usize])).collect();
            if let Some(m) = max_edge {
                roof = subdivide(roof, m);
                for t in &roof {
                    for p in t {
                        let w = projection.world(ground(*p), base);
                        let u = up_at(w);
                        roof_index.push(pos.len() as u32);
                        source.push(u32::MAX);
                        pos.push(w);
                        nt.push([u[0] as f32, u[1] as f32, u[2] as f32, 1.0]);
                    }
                }
            } else {
                for (j, p) in points.iter().enumerate() {
                    source.push(j as u32);
                    pos.push(projection.world(ground(*p), base));
                    nt.push([0.0, 1.0, 0.0, 1.0]);
                }
                for t in &tris {
                    roof_index.extend(t.iter().map(|&i| start as u32 + i));
                }
            }

            // Walls: a quad per ring edge, facing out of the solid.
            let rings = std::iter::once((&outer, false)).chain(holes.iter().map(|h| (h, true)));
            let mut first_point = 0u32;
            for (ring, hole) in rings {
                let first = first_point;
                first_point += ring.len() as u32;
                if ring.len() < 3 {
                    continue;
                }
                // Each ring point's index among the feature's points.
                let mut ids: Vec<u32> = match max_edge {
                    None => (first..first + ring.len() as u32).collect(),
                    Some(_) => vec![u32::MAX; ring.len()],
                };
                // deck.gl's winding: outer rings clockwise, holes counter-
                // clockwise (y up), reversed in place when not — which sets
                // the order walls are drawn in, and so how translucent ones
                // blend.
                let signed: f64 = (0..ring.len())
                    .map(|i| {
                        let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
                        a[0] * b[1] - b[0] * a[1]
                    })
                    .sum();
                let reversed: Vec<[f64; 2]>;
                let ring = if (signed > 0.0) != hole {
                    // As deck.gl reverses a closed ring: the first point
                    // stays first.
                    reversed = std::iter::once(ring[0]).chain(ring[1..].iter().rev().copied()).collect();
                    ids[1..].reverse();
                    &reversed
                } else {
                    ring
                };
                let area: f64 = (0..ring.len())
                    .map(|i| {
                        let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
                        a[0] * b[1] - b[0] * a[1]
                    })
                    .sum();
                // A counter-clockwise ring has its inside on the left, so its
                // outside on the right; a hole's solid is outside it.
                let s = if (area > 0.0) != hole { 1.0 } else { -1.0 };
                for i in 0..ring.len() {
                    let (a, b) = (ground(ring[i]), ground(ring[(i + 1) % ring.len()]));
                    if a == b {
                        continue;
                    }
                    let (wa, wb) = (projection.world(a, base), projection.world(b, base));
                    let e = [wb[0] - wa[0], wb[1] - wa[1], wb[2] - wa[2]];
                    let mid = [
                        (wa[0] + wb[0]) * 0.5,
                        (wa[1] + wb[1]) * 0.5,
                        (wa[2] + wb[2]) * 0.5,
                    ];
                    // The wall's outward normal: across the edge, level with
                    // the ground. On the flat ground this is the edge's right
                    // in (x, y), (dv, -du), with world z being -y.
                    let n = normalize(cross(e, up_at(mid)));
                    let d = [b[0] - a[0], b[1] - a[1]];
                    let n = match projection.wall_normal(a, [s * d[1], -s * d[0]]) {
                        Some(m) => [m[0] as f32, m[1] as f32, m[2] as f32],
                        None => [(s * n[0]) as f32, (s * n[1]) as f32, (s * n[2]) as f32],
                    };
                    let w = pos.len() as u32;
                    let (ia, ib) = (ids[i], ids[(i + 1) % ring.len()]);
                    for (p, top, id) in [(wa, 0.0, ia), (wb, 0.0, ib), (wb, 1.0, ib), (wa, 1.0, ia)] {
                        source.push(id);
                        pos.push(p);
                        nt.push([n[0], n[1], n[2], top]);
                    }
                    // Wound to face the normal: (bottom a, bottom b, top b)
                    // faces e × up, the outside only when `s` is positive —
                    // a ring run clockwise, as deck.gl's winding leaves an
                    // outer one, needs its quads turned over, or a scene
                    // renderer culls the walls (and their shadows) away.
                    if s > 0.0 {
                        index.extend_from_slice(&[w, w + 1, w + 2, w, w + 2, w + 3]);
                    } else {
                        index.extend_from_slice(&[w, w + 2, w + 1, w, w + 3, w + 2]);
                    }
                }
            }
            feature.extend(std::iter::repeat_n(k as u32, pos.len() - start));
            ranges.push((start as u32, pos.len() as u32));
        }
        index.extend(roof_index);
        // Stored local to the middle of the layer's bounds: half the bytes of
        // absolute f64 positions, and exact enough across any extent one
        // layer spans.
        let origin = match pos.first() {
            None => [0.0; 3],
            Some(&p0) => {
                let (mut lo, mut hi) = (p0, p0);
                for p in &pos {
                    for k in 0..3 {
                        lo[k] = lo[k].min(p[k]);
                        hi[k] = hi[k].max(p[k]);
                    }
                }
                [
                    (lo[0] + hi[0]) * 0.5,
                    (lo[1] + hi[1]) * 0.5,
                    (lo[2] + hi[2]) * 0.5,
                ]
            }
        };
        let positions = pos
            .iter()
            .map(|p| {
                [
                    (p[0] - origin[0]) as f32,
                    (p[1] - origin[1]) as f32,
                    (p[2] - origin[2]) as f32,
                ]
            })
            .collect();
        Self {
            origin,
            positions,
            normal_top: nt,
            feature,
            source,
            index,
            ranges,
            radial: centre,
        }
    }

    /// The mesh as bytes, to cross a boundary that only passes buffers — a
    /// Web Worker's `postMessage`. Little-endian.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            64 + self.positions.len() * 32 + self.index.len() * 4 + self.ranges.len() * 8,
        );
        out.extend_from_slice(MAGIC);
        for v in self.origin {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.push(u8::from(self.radial.is_some()));
        for v in self.radial.unwrap_or_default() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for n in [self.positions.len(), self.index.len(), self.ranges.len()] {
            out.extend_from_slice(&(n as u32).to_le_bytes());
        }
        let f32s = |out: &mut Vec<u8>, v: &[f32]| {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        };
        let u32s = |out: &mut Vec<u8>, v: &[u32]| {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        };
        f32s(&mut out, bytemuck::cast_slice(&self.positions));
        f32s(&mut out, bytemuck::cast_slice(&self.normal_top));
        u32s(&mut out, &self.feature);
        u32s(&mut out, &self.source);
        u32s(&mut out, &self.index);
        for &(a, b) in &self.ranges {
            u32s(&mut out, &[a, b]);
        }
        out
    }

    /// [`to_bytes`](Self::to_bytes) back.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(4)? != MAGIC {
            return Err("not a PolygonMesh".into());
        }
        let origin = [r.f64()?, r.f64()?, r.f64()?];
        let has_radial = r.take(1)?[0] != 0;
        let radial = [r.f64()?, r.f64()?, r.f64()?];
        let (n, ni, nr) = (r.u32()? as usize, r.u32()? as usize, r.u32()? as usize);
        let positions = (0..n)
            .map(|_| Ok([r.f32()?, r.f32()?, r.f32()?]))
            .collect::<Result<_, String>>()?;
        let normal_top = (0..n)
            .map(|_| Ok([r.f32()?, r.f32()?, r.f32()?, r.f32()?]))
            .collect::<Result<_, String>>()?;
        let feature = (0..n).map(|_| r.u32()).collect::<Result<_, _>>()?;
        let source = (0..n).map(|_| r.u32()).collect::<Result<_, _>>()?;
        let index: Vec<u32> = (0..ni).map(|_| r.u32()).collect::<Result<_, _>>()?;
        let ranges = (0..nr)
            .map(|_| Ok((r.u32()?, r.u32()?)))
            .collect::<Result<_, String>>()?;
        if r.at != bytes.len() {
            return Err(format!("{} bytes left over", bytes.len() - r.at));
        }
        if index.iter().any(|&i| i as usize >= n) {
            return Err("an index past the last vertex".into());
        }
        Ok(Self {
            origin,
            positions,
            normal_top,
            feature,
            source,
            index,
            ranges,
            radial: has_radial.then_some(radial),
        })
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let s = self
            .bytes
            .get(self.at..self.at + n)
            .ok_or("PolygonMesh bytes end early")?;
        self.at += n;
        Ok(s)
    }
    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l == 0.0 {
        [0.0, 1.0, 0.0]
    } else {
        [v[0] / l, v[1] / l, v[2] / l]
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn mid(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    // Symmetric in its arguments, so both triangles on an edge agree.
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

/// A closed ring (no repeated end) with points inserted so no edge exceeds
/// `max`.
fn densify(ring: &[[f64; 2]], max: f64) -> Vec<[f64; 2]> {
    let mut out = Vec::with_capacity(ring.len());
    for i in 0..ring.len() {
        let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
        out.push(a);
        let n = (dist(a, b) / max).ceil() as usize;
        for j in 1..n {
            let t = j as f64 / n as f64;
            out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
        }
    }
    out
}

/// Split every triangle edge longer than `max` at its midpoint, until none
/// is. Whether an edge splits depends on the edge alone, so the triangles
/// either side of it agree and no T-junction opens.
fn subdivide(mut tris: Vec<[[f64; 2]; 3]>, max: f64) -> Vec<[[f64; 2]; 3]> {
    let mut done = Vec::with_capacity(tris.len());
    let mut guard = 0;
    while let Some(t) = tris.pop() {
        guard += 1;
        let long = [0, 1, 2].map(|i| dist(t[i], t[(i + 1) % 3]) > max);
        if guard > 4_000_000 || !long.iter().any(|&l| l) {
            done.push(t);
            continue;
        }
        let m = [0, 1, 2].map(|i| mid(t[i], t[(i + 1) % 3]));
        let [a, b, c] = t;
        match long {
            [true, true, true] => tris.extend([
                [a, m[0], m[2]],
                [m[0], b, m[1]],
                [m[2], m[1], c],
                [m[0], m[1], m[2]],
            ]),
            [true, false, false] => tris.extend([[a, m[0], c], [m[0], b, c]]),
            [false, true, false] => tris.extend([[a, b, m[1]], [a, m[1], c]]),
            [false, false, true] => tris.extend([[a, b, m[2]], [m[2], b, c]]),
            [true, true, false] => tris.extend([[a, m[0], c], [m[0], b, m[1]], [m[0], m[1], c]]),
            [false, true, true] => tris.extend([[a, b, m[2]], [m[2], b, m[1]], [m[2], m[1], c]]),
            [true, false, true] => tris.extend([[a, m[0], m[2]], [m[0], b, c], [m[2], m[0], c]]),
            [false, false, false] => unreachable!(),
        }
    }
    done
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marks::geo::GlobeGround;

    fn square(x: f64, y: f64, s: f64) -> Vec<[f64; 2]> {
        vec![[x, y], [x + s, y], [x + s, y + s], [x, y + s]]
    }

    #[test]
    fn bytes_round_trip_exactly_and_reject_garbage() {
        let mut f = PolygonFeature::new(square(1e6, 2e6, 30.0));
        f.holes.push(square(1e6 + 10.0, 2e6 + 10.0, 5.0));
        let mesh = PolygonMesh::build(&[f, PolygonFeature::new(square(0.0, 0.0, 3.0))], 2.0);
        let back = PolygonMesh::from_bytes(&mesh.to_bytes()).unwrap();
        assert_eq!(back, mesh);
        let b = mesh.to_bytes();
        assert!(PolygonMesh::from_bytes(&b[..b.len() - 1]).is_err());
        assert!(PolygonMesh::from_bytes(b"nope").is_err());
        fn send<T: Send>() {}
        send::<PolygonMesh>();
    }

    /// Every triangle, roof and walls, is wound to face its own normal once
    /// extruded — what a scene renderer culls by (`to_buffer_geometry`) —
    /// whichever way the rings were given, holes included, flat and on a
    /// globe.
    #[test]
    fn triangles_face_their_normals_whichever_way_rings_run() {
        let globe = GlobeGround {
            radius: 1000.0,
            centre: [0.0; 3],
            max_edge_degrees: 2.0,
        };
        let ccw = square(0.0, 0.0, 10.0);
        let cw: Vec<[f64; 2]> = ccw.iter().rev().copied().collect();
        let hole = square(3.0, 3.0, 2.0);
        let hole_cw: Vec<[f64; 2]> = hole.iter().rev().copied().collect();
        for (outer, h) in [(&ccw, &hole), (&cw, &hole), (&ccw, &hole_cw), (&cw, &hole_cw)] {
            let mut f = PolygonFeature::new(outer.clone());
            f.holes.push(h.clone());
            for on_globe in [false, true] {
                let mesh = if on_globe {
                    PolygonMesh::build_with(&[f.clone()], 0.0, &globe)
                } else {
                    PolygonMesh::build(&[f.clone()], 0.0)
                };
                // Extruded 5 up: tops rise along the vertex's up.
                let at = |i: u32| {
                    let i = i as usize;
                    let p = mesh.positions[i];
                    let w = [p[0] as f64 + mesh.origin[0], p[1] as f64 + mesh.origin[1], p[2] as f64 + mesh.origin[2]];
                    let up = match mesh.radial {
                        Some(c) => {
                            let d = [w[0] - c[0], w[1] - c[1], w[2] - c[2]];
                            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                            [d[0] / l, d[1] / l, d[2] / l]
                        }
                        None => [0.0, 1.0, 0.0],
                    };
                    let rise = 5.0 * mesh.normal_top[i][3] as f64;
                    [w[0] + up[0] * rise, w[1] + up[1] * rise, w[2] + up[2] * rise]
                };
                for t in mesh.index.chunks(3) {
                    let (a, b, c) = (at(t[0]), at(t[1]), at(t[2]));
                    let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                    let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                    let g = cross(e1, e2);
                    let n = mesh.normal_top[t[0] as usize];
                    let facing = g[0] * n[0] as f64 + g[1] * n[1] as f64 + g[2] * n[2] as f64;
                    assert!(facing > 0.0, "globe {on_globe}, outer {outer:?}, hole {h:?}: triangle {t:?} faces away from {n:?}");
                }
            }
        }
    }

    #[test]
    fn a_globe_mesh_follows_the_surface_without_cracks() {
        let globe = GlobeGround {
            radius: 1000.0,
            centre: [0.0; 3],
            max_edge_degrees: 2.0,
        };
        // Nineteen degrees on a side, so no ring edge lands exactly on the
        // 2° limit: flat triangles would cut far inside the sphere.
        let mesh =
            PolygonMesh::build_with(&[PolygonFeature::new(square(0.0, 0.0, 19.0))], 0.0, &globe);
        assert_eq!(mesh.radial, Some([0.0; 3]));
        let world = |i: usize| {
            let p = mesh.positions[i];
            [
                p[0] as f64 + mesh.origin[0],
                p[1] as f64 + mesh.origin[1],
                p[2] as f64 + mesh.origin[2],
            ]
        };
        // Every vertex lies on the sphere, and every roof triangle is small.
        for i in 0..mesh.positions.len() {
            let w = world(i);
            let r = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
            assert!((r - 1000.0).abs() < 0.01, "vertex {i} at radius {r}");
        }
        let roof = mesh.normal_top.iter().filter(|n| n[3] == 1.0).count();
        assert!(roof > 100, "subdivided: {roof} roof vertices");
        // Crack-free: every interior roof edge is shared by exactly two
        // triangles, so each edge appears an even number of times or on the
        // boundary.
        use std::collections::HashMap;
        let key = |i: u32| {
            let w = world(i as usize);
            (
                (w[0] * 1e3).round() as i64,
                (w[1] * 1e3).round() as i64,
                (w[2] * 1e3).round() as i64,
            )
        };
        let mut edges: HashMap<_, u32> = HashMap::new();
        let roof_tris = mesh
            .index
            .chunks(3)
            .filter(|t| {
                mesh.normal_top[t[0] as usize][3] == 1.0
                    && mesh.normal_top[t[1] as usize][3] == 1.0
                    && mesh.normal_top[t[2] as usize][3] == 1.0
            })
            .filter(|t| {
                // Walls' top vertices are flagged too; a roof triangle's three
                // vertices all point straight out of the sphere.
                t.iter().all(|&i| {
                    let w = world(i as usize);
                    let n = mesh.normal_top[i as usize];
                    let r = 1000.0;
                    ((n[0] as f64 - w[0] / r).abs() + (n[1] as f64 - w[1] / r).abs()) < 1e-3
                })
            });
        for t in roof_tris {
            for i in 0..3 {
                let (a, b) = (key(t[i]), key(t[(i + 1) % 3]));
                let e = if a < b { (a, b) } else { (b, a) };
                *edges.entry(e).or_default() += 1;
            }
        }
        let over = edges.values().filter(|&&n| n > 2).count();
        assert_eq!(over, 0, "no edge is shared by more than two triangles");
        let boundary = edges.values().filter(|&&n| n == 1).count();
        // The boundary is the densified ring: 4 sides of 19° in 10 pieces each.
        assert_eq!(
            boundary, 40,
            "only the outline is open: a T-junction would add more"
        );
    }
}
