use crate::core::{BufferAttribute, BufferGeometry};
use crate::curves::{earcut, Curve2, Shape};

pub struct ExtrudeGeometry;

impl ExtrudeGeometry {
    /// Simplified extrusion: take the outline of `shape`, sample it into a
    /// 2D contour, then sweep it along Z by `depth`. The cap faces are
    /// triangulated with a fan from the first vertex (assumes a convex
    /// outline — three.js's earcut-based triangulation lands later).
    pub fn new(shape: &Shape, depth: f32, samples_per_segment: usize) -> BufferGeometry {
        let pts2 = shape
            .outline
            .curve_path
            .get_points(samples_per_segment.max(2));
        let n = pts2.len();
        if n < 3 {
            return BufferGeometry::new();
        }
        let mut positions = Vec::with_capacity(2 * n * 3 + 6 * n);
        let mut normals = Vec::with_capacity(2 * n * 3 + 6 * n);
        let mut uvs = Vec::with_capacity(2 * n * 2 + 4 * n);
        let mut indices = Vec::new();

        // Bottom cap (z = 0).
        for p in &pts2 {
            positions.extend_from_slice(&[p.x, p.y, 0.0]);
            normals.extend_from_slice(&[0.0, 0.0, -1.0]);
            uvs.extend_from_slice(&[p.x, p.y]);
        }
        // Top cap (z = depth).
        for p in &pts2 {
            positions.extend_from_slice(&[p.x, p.y, depth]);
            normals.extend_from_slice(&[0.0, 0.0, 1.0]);
            uvs.extend_from_slice(&[p.x, p.y]);
        }
        // Earcut caps (handles concave outlines).
        let tris = earcut::earcut(&pts2, &[]);
        // Read only by the `brep` tagging below; without that feature nothing
        // consumes it, and it is not worth cfg-ing the arithmetic itself.
        #[cfg_attr(not(feature = "brep"), allow(unused_variables))]
        let cap_triangles = tris.len() / 3;
        for tri in tris.chunks_exact(3) {
            indices.extend_from_slice(&[tri[0], tri[2], tri[1]]);
            let off = n as u32;
            indices.extend_from_slice(&[off + tri[0], off + tri[1], off + tri[2]]);
        }

        // Caps first, bottom and top triangles interleaved, so no cap is a run
        // of consecutive triangles: they stay separate polygons.
        let mut polygons = super::polygons::CellPolygons::new();
        polygons.triangles(indices.len() / 3);
        // Sides — each contour edge becomes a quad (4 unique vertices for flat shading).
        let side_base = positions.len() / 3;
        let _ = side_base;
        for i in 0..n {
            let a = pts2[i];
            let b = pts2[(i + 1) % n];
            // Per-quad normal (outward in XY plane).
            let edge_x = b.x - a.x;
            let edge_y = b.y - a.y;
            let nx_raw = edge_y;
            let ny_raw = -edge_x;
            let nlen = (nx_raw * nx_raw + ny_raw * ny_raw).sqrt().max(1e-8);
            let nx = nx_raw / nlen;
            let ny = ny_raw / nlen;
            let base = (positions.len() / 3) as u32;
            positions.extend_from_slice(&[a.x, a.y, 0.0]);
            positions.extend_from_slice(&[b.x, b.y, 0.0]);
            positions.extend_from_slice(&[b.x, b.y, depth]);
            positions.extend_from_slice(&[a.x, a.y, depth]);
            for _ in 0..4 {
                normals.extend_from_slice(&[nx, ny, 0.0]);
            }
            uvs.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            polygons.quad(&positions, [base, base + 1, base + 2, base + 3]);
        }

        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_attribute("normal", BufferAttribute::new(normals, 3));
        g.set_attribute("uv", BufferAttribute::new(uvs, 2));
        g.set_index(indices);
        polygons.apply(&mut g);
        #[cfg(feature = "brep")]
        crate::brep::primitives::tag_extrude(&mut g, &pts2, depth, cap_triangles);
        g
    }
}

/// Options of three.js r165's `ExtrudeGeometry` (and so of `TextGeometry`),
/// with three.js's defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtrudeOptions {
    /// Points per curve when the shapes are sampled. Default 12.
    pub curve_segments: usize,
    /// Layers along the extrusion (and samples of `extrude_path`). Default 1.
    pub steps: usize,
    /// Extrusion depth along +Z. Default 1. Ignored with `extrude_path`.
    pub depth: f64,
    /// Default true. Forced off with `extrude_path`, as in three.js.
    pub bevel_enabled: bool,
    /// How deep into the shape the bevel goes. Default 0.2.
    pub bevel_thickness: f64,
    /// How far from the outline the bevel reaches. Default
    /// `bevel_thickness - 0.1` (`None` means that default).
    pub bevel_size: Option<f64>,
    /// Where the bevel starts relative to the outline. Default 0.
    pub bevel_offset: f64,
    /// Bevel layers. Default 3.
    pub bevel_segments: usize,
    /// three.js's `extrudePath`, already sampled — see [`ExtrudePath`].
    pub extrude_path: Option<ExtrudePath>,
}

impl Default for ExtrudeOptions {
    fn default() -> Self {
        Self {
            curve_segments: 12,
            steps: 1,
            depth: 1.0,
            bevel_enabled: true,
            bevel_thickness: 0.2,
            bevel_size: None,
            bevel_offset: 0.0,
            bevel_segments: 3,
            extrude_path: None,
        }
    }
}

/// A curve to extrude along, given as three.js reads it: `steps + 1` points
/// from `extrudePath.getSpacedPoints(steps)` and as many tangents from
/// `extrudePath.getTangentAt(i / steps)`. Sampling is the caller's, so any
/// curve works — including one defined in JavaScript.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExtrudePath {
    pub points: Vec<[f64; 3]>,
    pub tangents: Vec<[f64; 3]>,
}

impl ExtrudePath {
    /// Sample `extrudePath` as three.js does: `getSpacedPoints(steps)` and
    /// the tangents of `computeFrenetFrames(steps, false)` (`getTangentAt`).
    pub fn from_curve(curve: &crate::curves::ThreeCurve, steps: usize) -> ExtrudePath {
        let points = curve
            .get_spaced_points(steps)
            .into_iter()
            .map(|p| p.unwrap_or([f64::NAN; 3]))
            .collect();
        let tangents = curve.compute_frenet_frames(steps, false).tangents;
        ExtrudePath { points, tangents }
    }
}

/// Which UV generator call made a run of vertices: a cap triangle (three
/// vertices, `generateTopUV`) or a side quad (six, `generateSideWallUV`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtrudeFace {
    Top,
    Side,
}

/// What three.js's `ExtrudeGeometry` builds before it becomes buffers: the
/// non-indexed vertices and UVs in JavaScript precision, the material groups
/// (0 caps, 1 sides, per shape), and the face sequence — what a custom
/// `UVGenerator` is called on, in order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExtrudeBuffers {
    pub positions: Vec<f64>,
    pub uvs: Vec<f64>,
    pub groups: Vec<(u32, u32, u32)>,
    pub faces: Vec<ExtrudeFace>,
}

impl ExtrudeBuffers {
    /// The geometry three.js ends with: `position` and `uv` as `f32`, groups,
    /// and `computeVertexNormals()`'s face normals.
    pub fn into_geometry(self) -> BufferGeometry {
        let positions: Vec<f32> = self.positions.iter().map(|&v| v as f32).collect();
        let normals = non_indexed_vertex_normals(&positions);
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_attribute("uv", BufferAttribute::new(self.uvs.iter().map(|&v| v as f32).collect(), 2));
        g.set_attribute("normal", BufferAttribute::new(normals, 3));
        for (s, c, m) in self.groups {
            g.add_group(s, c, m);
        }
        g
    }
}

/// three.js `BufferGeometry.computeVertexNormals()` on non-indexed `f32`
/// positions: each triangle's `(c - b) × (a - b)` on its three vertices, then
/// `normalizeNormals()`, with the intermediate stored as `f32` as three.js
/// stores it.
pub(crate) fn non_indexed_vertex_normals(positions: &[f32]) -> Vec<f32> {
    let mut normals = vec![0.0f32; positions.len()];
    for (tri, out) in positions.chunks_exact(9).zip(normals.chunks_exact_mut(9)) {
        let p = |k: usize| [tri[k * 3] as f64, tri[k * 3 + 1] as f64, tri[k * 3 + 2] as f64];
        let (a, b, c) = (p(0), p(1), p(2));
        let cb = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
        let ab = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let n = [
            cb[1] * ab[2] - cb[2] * ab[1],
            cb[2] * ab[0] - cb[0] * ab[2],
            cb[0] * ab[1] - cb[1] * ab[0],
        ];
        for v in 0..3 {
            out[v * 3] = n[0] as f32;
            out[v * 3 + 1] = n[1] as f32;
            out[v * 3 + 2] = n[2] as f32;
        }
    }
    for n in normals.chunks_exact_mut(3) {
        let (x, y, z) = (n[0] as f64, n[1] as f64, n[2] as f64);
        let l = (x * x + y * y + z * z).sqrt();
        let inv = 1.0 / if l == 0.0 { 1.0 } else { l };
        n[0] = (x * inv) as f32;
        n[1] = (y * inv) as f32;
        n[2] = (z * inv) as f32;
    }
    normals
}

type V2 = crate::curves::shapes::Vec2d;

/// `getBevelVec(inPt, inPrev, inNext)`.
fn bevel_vec(pt: V2, prev: V2, next: V2) -> V2 {
    let (v_trans_x, v_trans_y, shrink_by);
    let v_prev_x = pt.x - prev.x;
    let v_prev_y = pt.y - prev.y;
    let v_next_x = next.x - pt.x;
    let v_next_y = next.y - pt.y;
    let v_prev_lensq = v_prev_x * v_prev_x + v_prev_y * v_prev_y;
    let collinear0 = v_prev_x * v_next_y - v_prev_y * v_next_x;
    if collinear0.abs() > f64::EPSILON {
        let v_prev_len = v_prev_lensq.sqrt();
        let v_next_len = (v_next_x * v_next_x + v_next_y * v_next_y).sqrt();
        let pt_prev_shift_x = prev.x - v_prev_y / v_prev_len;
        let pt_prev_shift_y = prev.y + v_prev_x / v_prev_len;
        let pt_next_shift_x = next.x - v_next_y / v_next_len;
        let pt_next_shift_y = next.y + v_next_x / v_next_len;
        let sf = ((pt_next_shift_x - pt_prev_shift_x) * v_next_y
            - (pt_next_shift_y - pt_prev_shift_y) * v_next_x)
            / (v_prev_x * v_next_y - v_prev_y * v_next_x);
        v_trans_x = pt_prev_shift_x + v_prev_x * sf - pt.x;
        v_trans_y = pt_prev_shift_y + v_prev_y * sf - pt.y;
        let v_trans_lensq = v_trans_x * v_trans_x + v_trans_y * v_trans_y;
        if v_trans_lensq <= 2.0 {
            return V2::new(v_trans_x, v_trans_y);
        }
        shrink_by = (v_trans_lensq / 2.0).sqrt();
    } else {
        // `Math.sign` equality: 0 and -0 agree, NaN never does.
        let js_sign = |v: f64| if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { v };
        let direction_eq = if v_prev_x > f64::EPSILON {
            v_next_x > f64::EPSILON
        } else if v_prev_x < -f64::EPSILON {
            v_next_x < -f64::EPSILON
        } else {
            js_sign(v_prev_y) == js_sign(v_next_y)
        };
        if direction_eq {
            v_trans_x = -v_prev_y;
            v_trans_y = v_prev_x;
            shrink_by = v_prev_lensq.sqrt();
        } else {
            v_trans_x = v_prev_x;
            v_trans_y = v_prev_y;
            shrink_by = (v_prev_lensq / 2.0).sqrt();
        }
    }
    V2::new(v_trans_x / shrink_by, v_trans_y / shrink_by)
}

fn movements(contour: &[V2]) -> Vec<V2> {
    let il = contour.len();
    (0..il)
        .map(|i| {
            let j = if i == 0 { il - 1 } else { i - 1 };
            let k = if i + 1 == il { 0 } else { i + 1 };
            bevel_vec(contour[i], contour[j], contour[k])
        })
        .collect()
}

/// `scalePt2(pt, vec, size)`: `pt.clone().addScaledVector(vec, size)`.
fn scale_pt2(pt: V2, vec: V2, size: f64) -> V2 {
    V2::new(pt.x + vec.x * size, pt.y + vec.y * size)
}

impl ExtrudeGeometry {
    /// three.js r165 `new ExtrudeGeometry(shapes, options)`: bevels, steps,
    /// holes, an extrusion path, world UVs, groups 0 (caps) and 1 (sides) per
    /// shape, unindexed with face normals — buffer for buffer what three.js
    /// builds.
    pub fn from_shapes(
        shapes: &[crate::curves::shapes::Shape],
        options: &ExtrudeOptions,
    ) -> BufferGeometry {
        Self::build(shapes, options).into_geometry()
    }

    /// [`from_shapes`](Self::from_shapes) stopped before the buffers: for a
    /// caller with its own UV generator to rewrite `uvs` over `faces`.
    pub fn build(shapes: &[crate::curves::shapes::Shape], options: &ExtrudeOptions) -> ExtrudeBuffers {
        let mut out = ExtrudeBuffers::default();
        for shape in shapes {
            add_shape(shape, options, &mut out);
        }
        out
    }
}

fn add_shape(shape: &crate::curves::shapes::Shape, o: &ExtrudeOptions, out: &mut ExtrudeBuffers) {
    use crate::curves::shapes::shape_utils;
    let curve_segments = o.curve_segments;
    let steps = o.steps;
    let depth = o.depth;
    let mut bevel_enabled = o.bevel_enabled;
    let mut bevel_thickness = o.bevel_thickness;
    let mut bevel_size = o.bevel_size.unwrap_or(bevel_thickness - 0.1);
    let mut bevel_offset = o.bevel_offset;
    let mut bevel_segments = o.bevel_segments;

    let mut path_frames = None;
    if let Some(path) = &o.extrude_path {
        bevel_enabled = false;
        let frames = super::frenet::compute_frenet_frames(&path.tangents, false);
        path_frames = Some((path, frames));
    }
    if !bevel_enabled {
        bevel_segments = 0;
        bevel_thickness = 0.0;
        bevel_size = 0.0;
        bevel_offset = 0.0;
    }

    let points = shape.extract_points(curve_segments);
    let mut vertices = points.shape;
    let mut holes = points.holes;
    if !shape_utils::is_clock_wise(&vertices) {
        vertices.reverse();
        for h in holes.iter_mut() {
            if shape_utils::is_clock_wise(h) {
                h.reverse();
            }
        }
    }
    let faces = shape_utils::triangulate_shape(&mut vertices, &mut holes);
    let contour = vertices.clone();
    for h in &holes {
        vertices.extend_from_slice(h);
    }
    let vlen = vertices.len();
    let flen = faces.len();

    let contour_movements = movements(&contour);
    let holes_movements: Vec<Vec<V2>> = holes.iter().map(|h| movements(h)).collect();
    let mut vertices_movements = contour_movements.clone();
    for m in &holes_movements {
        vertices_movements.extend_from_slice(m);
    }

    let mut placeholder: Vec<f64> = Vec::new();
    let mut v = |x: f64, y: f64, z: f64| placeholder.extend_from_slice(&[x, y, z]);

    let half_pi_of = |b: usize| (b as f64 / bevel_segments as f64) * std::f64::consts::PI / 2.0;

    // Back bevel.
    for b in 0..bevel_segments {
        let a = half_pi_of(b);
        let z = bevel_thickness * a.cos();
        let bs = bevel_size * a.sin() + bevel_offset;
        for (i, &c) in contour.iter().enumerate() {
            let vert = scale_pt2(c, contour_movements[i], bs);
            v(vert.x, vert.y, -z);
        }
        for (h, hole) in holes.iter().enumerate() {
            for (i, &p) in hole.iter().enumerate() {
                let vert = scale_pt2(p, holes_movements[h][i], bs);
                v(vert.x, vert.y, -z);
            }
        }
    }

    let bs = bevel_size + bevel_offset;
    let path_vertex = |s: usize, vert: V2| -> [f64; 3] {
        let (path, frames) = path_frames.as_ref().unwrap();
        let n = frames.normals[s];
        let bn = frames.binormals[s];
        let p = path.points[s];
        let nx = [n[0] * vert.x, n[1] * vert.x, n[2] * vert.x];
        let by = [bn[0] * vert.y, bn[1] * vert.y, bn[2] * vert.y];
        [p[0] + nx[0] + by[0], p[1] + nx[1] + by[1], p[2] + nx[2] + by[2]]
    };
    for i in 0..vlen {
        let vert = if bevel_enabled {
            scale_pt2(vertices[i], vertices_movements[i], bs)
        } else {
            vertices[i]
        };
        if path_frames.is_none() {
            v(vert.x, vert.y, 0.0);
        } else {
            let p = path_vertex(0, vert);
            v(p[0], p[1], p[2]);
        }
    }
    for s in 1..=steps {
        for i in 0..vlen {
            let vert = if bevel_enabled {
                scale_pt2(vertices[i], vertices_movements[i], bs)
            } else {
                vertices[i]
            };
            if path_frames.is_none() {
                v(vert.x, vert.y, depth / steps as f64 * s as f64);
            } else {
                let p = path_vertex(s, vert);
                v(p[0], p[1], p[2]);
            }
        }
    }

    // Front bevel. (three.js's extrude-path branch for holes here is
    // unreachable: a path turns bevels off.)
    for b in (0..bevel_segments).rev() {
        let a = half_pi_of(b);
        let z = bevel_thickness * a.cos();
        let bs = bevel_size * a.sin() + bevel_offset;
        for (i, &c) in contour.iter().enumerate() {
            let vert = scale_pt2(c, contour_movements[i], bs);
            v(vert.x, vert.y, depth + z);
        }
        for (h, hole) in holes.iter().enumerate() {
            for (i, &p) in hole.iter().enumerate() {
                let vert = scale_pt2(p, holes_movements[h][i], bs);
                v(vert.x, vert.y, depth + z);
            }
        }
    }

    let positions = &mut out.positions;
    let uvs = &mut out.uvs;
    let face_log = &mut out.faces;
    let add_vertex = |positions: &mut Vec<f64>, index: usize| {
        positions.extend_from_slice(&placeholder[index * 3..index * 3 + 3]);
    };
    let mut f3 = |positions: &mut Vec<f64>, uvs: &mut Vec<f64>, a: usize, b: usize, c: usize| {
        add_vertex(positions, a);
        add_vertex(positions, b);
        add_vertex(positions, c);
        let n = positions.len() / 3;
        for k in [n - 3, n - 2, n - 1] {
            uvs.extend_from_slice(&[positions[k * 3], positions[k * 3 + 1]]);
        }
        face_log.push(ExtrudeFace::Top);
    };

    // Lids.
    let start = (positions.len() / 3) as u32;
    if bevel_enabled {
        for f in &faces {
            f3(positions, uvs, f[2] as usize, f[1] as usize, f[0] as usize);
        }
        let offset = vlen * (steps + bevel_segments * 2);
        for f in &faces {
            f3(positions, uvs, f[0] as usize + offset, f[1] as usize + offset, f[2] as usize + offset);
        }
    } else {
        for f in &faces {
            f3(positions, uvs, f[2] as usize, f[1] as usize, f[0] as usize);
        }
        let offset = vlen * steps;
        for f in &faces {
            f3(positions, uvs, f[0] as usize + offset, f[1] as usize + offset, f[2] as usize + offset);
        }
    }
    let _ = flen;
    let end = (positions.len() / 3) as u32;
    out.groups.push((start, end - start, 0));

    // Sides.
    let start = end;
    let mut layer_offset = 0;
    let mut sidewalls = |positions: &mut Vec<f64>, uvs: &mut Vec<f64>, len: usize, layer_offset: usize| {
        let mut i = len as isize;
        loop {
            i -= 1;
            if i < 0 {
                break;
            }
            let j = i as usize;
            let k = if i == 0 { len - 1 } else { j - 1 };
            for s in 0..(steps + bevel_segments * 2) {
                let slen1 = vlen * s;
                let slen2 = vlen * (s + 1);
                let a = layer_offset + j + slen1;
                let b = layer_offset + k + slen1;
                let c = layer_offset + k + slen2;
                let d = layer_offset + j + slen2;
                for idx in [a, b, d, b, c, d] {
                    add_vertex(positions, idx);
                }
                let n = positions.len() / 3;
                let uv = side_wall_uv(positions, n - 6, n - 3, n - 2, n - 1);
                for q in [0, 1, 3, 1, 2, 3] {
                    uvs.extend_from_slice(&uv[q]);
                }
                face_log.push(ExtrudeFace::Side);
            }
        }
    };
    sidewalls(positions, uvs, contour.len(), layer_offset);
    layer_offset += contour.len();
    for h in &holes {
        sidewalls(positions, uvs, h.len(), layer_offset);
        layer_offset += h.len();
    }
    let end = (positions.len() / 3) as u32;
    out.groups.push((start, end - start, 1));
}

/// `WorldUVGenerator.generateSideWallUV`.
fn side_wall_uv(v: &[f64], a: usize, b: usize, c: usize, d: usize) -> [[f64; 2]; 4] {
    let p = |i: usize| (v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
    let (ax, ay, az) = p(a);
    let (bx, by, bz) = p(b);
    let (cx, cy, cz) = p(c);
    let (dx, dy, dz) = p(d);
    if (ay - by).abs() < (ax - bx).abs() {
        [[ax, 1.0 - az], [bx, 1.0 - bz], [cx, 1.0 - cz], [dx, 1.0 - dz]]
    } else {
        [[ay, 1.0 - az], [by, 1.0 - bz], [cy, 1.0 - cz], [dy, 1.0 - dz]]
    }
}
