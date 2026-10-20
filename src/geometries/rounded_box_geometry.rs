//! three.js r165 `RoundedBoxGeometry` (examples/jsm/geometries/RoundedBoxGeometry.js).
//!
//! three.js builds it by bending a unit `BoxGeometry` with `2·segments + 1`
//! divisions a side, unindexed, so this builds that box the way three.js's
//! `BoxGeometry` does — in `f64`, stored as `f32` — and then bends it the same
//! way, reading the stored `f32` values back as three.js reads its
//! `Float32Array`.

use crate::core::{BufferAttribute, BufferGeometry};

type V3 = [f64; 3];

/// JavaScript's `Math.sign`: 0 for ±0 (keeping the sign), NaN for NaN.
fn js_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

fn normalize(v: V3) -> V3 {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let l = if l == 0.0 { 1.0 } else { l };
    [v[0] / l, v[1] / l, v[2] / l]
}

/// `Vector3.angleTo`.
fn angle_to(a: V3, b: V3) -> f64 {
    let den = ((a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) * (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]))
        .sqrt();
    if den == 0.0 {
        return std::f64::consts::PI / 2.0;
    }
    let theta = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / den;
    theta.clamp(-1.0, 1.0).acos()
}

/// three.js `BoxGeometry(width, height, depth, wx, hy, dz)`'s buffers and
/// groups: `(positions, normals, uvs, indices, groups)`.
#[allow(clippy::type_complexity)]
pub(crate) fn three_box(
    width: f64,
    height: f64,
    depth: f64,
    segs: [usize; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<[u32; 3]>) {
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut groups = Vec::new();
    let mut number_of_vertices = 0u32;
    let mut group_start = 0u32;
    let [ws, hs, ds] = segs.map(|s| s.max(1));
    // (u, v, w, udir, vdir, width, height, depth, gridX, gridY)
    let planes: [(usize, usize, usize, f64, f64, f64, f64, f64, usize, usize); 6] = [
        (2, 1, 0, -1.0, -1.0, depth, height, width, ds, hs),
        (2, 1, 0, 1.0, -1.0, depth, height, -width, ds, hs),
        (0, 2, 1, 1.0, 1.0, width, depth, height, ws, ds),
        (0, 2, 1, 1.0, -1.0, width, depth, -height, ws, ds),
        (0, 1, 2, 1.0, -1.0, width, height, depth, ws, hs),
        (0, 1, 2, -1.0, -1.0, width, height, -depth, ws, hs),
    ];
    for (material, &(u, v, w, udir, vdir, pw, ph, pd, gx, gy)) in planes.iter().enumerate() {
        let segment_width = pw / gx as f64;
        let segment_height = ph / gy as f64;
        let (wh, hh, dh) = (pw / 2.0, ph / 2.0, pd / 2.0);
        let (gx1, gy1) = (gx + 1, gy + 1);
        let mut counter = 0u32;
        let mut group_count = 0u32;
        for iy in 0..gy1 {
            let y = iy as f64 * segment_height - hh;
            for ix in 0..gx1 {
                let x = ix as f64 * segment_width - wh;
                let mut p = [0.0f64; 3];
                p[u] = x * udir;
                p[v] = y * vdir;
                p[w] = dh;
                vertices.extend(p.iter().map(|&c| c as f32));
                let mut n = [0.0f64; 3];
                n[w] = if pd > 0.0 { 1.0 } else { -1.0 };
                normals.extend(n.iter().map(|&c| c as f32));
                uvs.push((ix as f64 / gx as f64) as f32);
                uvs.push((1.0 - iy as f64 / gy as f64) as f32);
                counter += 1;
            }
        }
        let gx1 = gx1 as u32;
        for iy in 0..gy as u32 {
            for ix in 0..gx as u32 {
                let a = number_of_vertices + ix + gx1 * iy;
                let b = number_of_vertices + ix + gx1 * (iy + 1);
                let c = number_of_vertices + (ix + 1) + gx1 * (iy + 1);
                let d = number_of_vertices + (ix + 1) + gx1 * iy;
                indices.extend_from_slice(&[a, b, d, b, c, d]);
                group_count += 6;
            }
        }
        groups.push([group_start, group_count, material as u32]);
        group_start += group_count;
        number_of_vertices += counter;
    }
    (vertices, normals, uvs, indices, groups)
}

/// three.js r165 `BoxGeometry(width, height, depth, widthSegments,
/// heightSegments, depthSegments)` exactly: indexed, six groups, no polygon
/// grouping (unlike [`BoxGeometry`](super::BoxGeometry), whose faces are
/// quads). What `EdgesGeometry` / `WireframeGeometry` see for a three.js box.
pub fn three_box_geometry(width: f64, height: f64, depth: f64, segments: [usize; 3]) -> BufferGeometry {
    let (pos, nrm, uv, idx, groups) = three_box(width, height, depth, segments);
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(pos, 3));
    g.set_attribute("normal", BufferAttribute::new(nrm, 3));
    g.set_attribute("uv", BufferAttribute::new(uv, 2));
    g.set_index(idx);
    for [s, c, m] in groups {
        g.add_group(s, c, m);
    }
    g
}

fn get_uv(
    face_dir: V3,
    normal: V3,
    uv_axis: usize,
    projection_axis: usize,
    radius: f64,
    side_length: f64,
) -> f64 {
    let tot_arc_length = 2.0 * std::f64::consts::PI * radius / 4.0;
    let center_length = (side_length - 2.0 * radius).max(0.0);
    let half_arc = std::f64::consts::PI / 4.0;
    let mut t = normal;
    t[projection_axis] = 0.0;
    let t = normalize(t);
    let arc_uv_ratio = 0.5 * tot_arc_length / (tot_arc_length + center_length);
    let arc_angle_ratio = 1.0 - (angle_to(t, face_dir) / half_arc);
    if js_sign(t[uv_axis]) == 1.0 {
        arc_angle_ratio * arc_uv_ratio
    } else {
        let len_uv = center_length / (tot_arc_length + center_length);
        len_uv + arc_uv_ratio + arc_uv_ratio * (1.0 - arc_angle_ratio)
    }
}

pub struct RoundedBoxGeometry;

impl RoundedBoxGeometry {
    /// three.js r165 `new RoundedBoxGeometry(width, height, depth, segments, radius)`.
    ///
    /// Unindexed with the box's six groups. As in three.js, `segments = 0`
    /// gives an indexed *unit* box, whatever the dimensions.
    pub fn new(width: f64, height: f64, depth: f64, segments: usize, radius: f64) -> BufferGeometry {
        let segments = segments * 2 + 1;
        let radius = (width / 2.0).min(height / 2.0).min(depth / 2.0).min(radius);
        let (pos, nrm, uv, idx, groups) = three_box(1.0, 1.0, 1.0, [segments; 3]);
        let mut g = BufferGeometry::new();
        if segments == 1 {
            g.set_attribute("position", BufferAttribute::new(pos, 3));
            g.set_attribute("normal", BufferAttribute::new(nrm, 3));
            g.set_attribute("uv", BufferAttribute::new(uv, 2));
            g.set_index(idx);
            for [s, c, m] in groups {
                g.add_group(s, c, m);
            }
            return g;
        }
        // toNonIndexed.
        let mut positions: Vec<f32> = Vec::with_capacity(idx.len() * 3);
        let mut normals: Vec<f32> = Vec::with_capacity(idx.len() * 3);
        let mut uvs: Vec<f32> = Vec::with_capacity(idx.len() * 2);
        for &i in &idx {
            let i = i as usize;
            positions.extend_from_slice(&pos[i * 3..i * 3 + 3]);
            normals.extend_from_slice(&nrm[i * 3..i * 3 + 3]);
            uvs.extend_from_slice(&uv[i * 2..i * 2 + 2]);
        }

        let half = |v: f64| v / 2.0 - radius;
        let bx = [half(width), half(height), half(depth)];
        let face_tris = positions.len() / 6;
        let half_segment_size = 0.5 / segments as f64;
        let (x, y, z) = (0, 1, 2);
        let mut i = 0;
        let mut j = 0;
        while i < positions.len() {
            let p = [positions[i] as f64, positions[i + 1] as f64, positions[i + 2] as f64];
            let mut n = p;
            for v in n.iter_mut() {
                *v -= js_sign(*v) * half_segment_size;
            }
            let n = normalize(n);
            for k in 0..3 {
                positions[i + k] = (bx[k] * js_sign(p[k]) + n[k] * radius) as f32;
                normals[i + k] = n[k] as f32;
            }
            let side = i / face_tris;
            let (u, v) = match side {
                0 => (
                    get_uv([1.0, 0.0, 0.0], n, z, y, radius, depth),
                    1.0 - get_uv([1.0, 0.0, 0.0], n, y, z, radius, height),
                ),
                1 => (
                    1.0 - get_uv([-1.0, 0.0, 0.0], n, z, y, radius, depth),
                    1.0 - get_uv([-1.0, 0.0, 0.0], n, y, z, radius, height),
                ),
                2 => (
                    1.0 - get_uv([0.0, 1.0, 0.0], n, x, z, radius, width),
                    get_uv([0.0, 1.0, 0.0], n, z, x, radius, depth),
                ),
                3 => (
                    1.0 - get_uv([0.0, -1.0, 0.0], n, x, z, radius, width),
                    1.0 - get_uv([0.0, -1.0, 0.0], n, z, x, radius, depth),
                ),
                4 => (
                    1.0 - get_uv([0.0, 0.0, 1.0], n, x, y, radius, width),
                    1.0 - get_uv([0.0, 0.0, 1.0], n, y, x, radius, height),
                ),
                5 => (
                    get_uv([0.0, 0.0, -1.0], n, x, y, radius, width),
                    1.0 - get_uv([0.0, 0.0, -1.0], n, y, x, radius, height),
                ),
                _ => (uvs[j] as f64, uvs[j + 1] as f64),
            };
            uvs[j] = u as f32;
            uvs[j + 1] = v as f32;
            i += 3;
            j += 2;
        }
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_attribute("normal", BufferAttribute::new(normals, 3));
        g.set_attribute("uv", BufferAttribute::new(uvs, 2));
        for [s, c, m] in groups {
            g.add_group(s, c, m);
        }
        g
    }
}
