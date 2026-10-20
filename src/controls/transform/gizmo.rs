//! three.js r165's `TransformControlsGizmo` as it is built: the shared
//! materials, the reusable geometries, the gizmo / picker / helper tables, and
//! `setupGizmo` baking each handle's offset into a clone of its geometry.
//!
//! The geometry generators here are three.js's (`CylinderGeometry`,
//! `BoxGeometry`, `PlaneGeometry`, `TorusGeometry`, `SphereGeometry`,
//! `OctahedronGeometry`), computed in `f64` and stored as `f32` at the points
//! three.js stores them in a `Float32BufferAttribute`, so every handle's
//! buffers are the ones three.js builds.

use super::math::{self, js, M4, V3};
use std::f64::consts::PI;

/// A handle's buffers: `position` (and, for meshes, `normal` / `uv` and maybe
/// an index), as three.js's `Float32BufferAttribute`s hold them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GizmoGeometry {
    pub position: Vec<f32>,
    pub normal: Vec<f32>,
    pub uv: Vec<f32>,
    pub index: Option<Vec<u32>>,
}

impl GizmoGeometry {
    fn from_f64(position: &[f64], normal: &[f64], uv: &[f64], index: Option<Vec<u32>>) -> Self {
        let f = |a: &[f64]| a.iter().map(|&v| v as f32).collect::<Vec<f32>>();
        Self {
            position: f(position),
            normal: f(normal),
            uv: f(uv),
            index,
        }
    }

    /// `BufferGeometry.applyMatrix4`: positions through the matrix, normals
    /// through its normal matrix and renormalised.
    pub fn apply_matrix4(&mut self, m: &M4) {
        for p in self.position.chunks_exact_mut(3) {
            let v = math::apply_m4([p[0] as f64, p[1] as f64, p[2] as f64], m);
            p[0] = v[0] as f32;
            p[1] = v[1] as f32;
            p[2] = v[2] as f32;
        }
        if !self.normal.is_empty() {
            let nm = math::normal_matrix(m);
            for n in self.normal.chunks_exact_mut(3) {
                let v =
                    math::normalize(math::apply_m3([n[0] as f64, n[1] as f64, n[2] as f64], &nm));
                n[0] = v[0] as f32;
                n[1] = v[1] as f32;
                n[2] = v[2] as f32;
            }
        }
    }

    fn translate(mut self, x: f64, y: f64, z: f64) -> Self {
        self.apply_matrix4(&math::make_translation(x, y, z));
        self
    }
}

/// three.js `CylinderGeometry(radiusTop, radiusBottom, height, radialSegments)`
/// (one height segment, closed, full turn).
pub fn cylinder(
    radius_top: f64,
    radius_bottom: f64,
    height: f64,
    radial_segments: usize,
) -> GizmoGeometry {
    let rs = radial_segments;
    let hs = 1usize;
    let (theta_start, theta_length) = (0.0, PI * 2.0);
    let (mut vertices, mut normals, mut uvs, mut indices) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut index = 0u32;
    let half_height = height / 2.0;
    let mut index_array: Vec<Vec<u32>> = Vec::new();
    let slope = (radius_bottom - radius_top) / height;
    for y in 0..=hs {
        let mut row = Vec::new();
        let v = y as f64 / hs as f64;
        let radius = v * (radius_bottom - radius_top) + radius_top;
        for x in 0..=rs {
            let u = x as f64 / rs as f64;
            let theta = u * theta_length + theta_start;
            let (sin_t, cos_t) = (js::sin(theta), js::cos(theta));
            vertices.extend_from_slice(&[
                radius * sin_t,
                -v * height + half_height,
                radius * cos_t,
            ]);
            let n = math::normalize([sin_t, slope, cos_t]);
            normals.extend_from_slice(&n);
            uvs.extend_from_slice(&[u, 1.0 - v]);
            row.push(index);
            index += 1;
        }
        index_array.push(row);
    }
    for x in 0..rs {
        for y in 0..hs {
            let a = index_array[y][x];
            let b = index_array[y + 1][x];
            let c = index_array[y + 1][x + 1];
            let d = index_array[y][x + 1];
            indices.extend_from_slice(&[a, b, d, b, c, d]);
        }
    }
    let mut cap = |top: bool, index: &mut u32| {
        let center_start = *index;
        let radius = if top { radius_top } else { radius_bottom };
        let sign = if top { 1.0 } else { -1.0 };
        for _ in 1..=rs {
            vertices.extend_from_slice(&[0.0, half_height * sign, 0.0]);
            normals.extend_from_slice(&[0.0, sign, 0.0]);
            uvs.extend_from_slice(&[0.5, 0.5]);
            *index += 1;
        }
        let center_end = *index;
        for x in 0..=rs {
            let u = x as f64 / rs as f64;
            let theta = u * theta_length + theta_start;
            let (cos_t, sin_t) = (js::cos(theta), js::sin(theta));
            vertices.extend_from_slice(&[radius * sin_t, half_height * sign, radius * cos_t]);
            normals.extend_from_slice(&[0.0, sign, 0.0]);
            uvs.extend_from_slice(&[cos_t * 0.5 + 0.5, sin_t * 0.5 * sign + 0.5]);
            *index += 1;
        }
        for x in 0..rs as u32 {
            let c = center_start + x;
            let i = center_end + x;
            if top {
                indices.extend_from_slice(&[i, i + 1, c]);
            } else {
                indices.extend_from_slice(&[i + 1, i, c]);
            }
        }
    };
    if radius_top > 0.0 {
        cap(true, &mut index);
    }
    if radius_bottom > 0.0 {
        cap(false, &mut index);
    }
    GizmoGeometry::from_f64(&vertices, &normals, &uvs, Some(indices))
}

/// three.js `BoxGeometry(width, height, depth)` (one segment per side).
pub fn box_geometry(width: f64, height: f64, depth: f64) -> GizmoGeometry {
    let (mut vertices, mut normals, mut uvs, mut indices) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut count = 0u32;
    #[allow(clippy::too_many_arguments)]
    let mut plane = |u: usize,
                     v: usize,
                     w: usize,
                     udir: f64,
                     vdir: f64,
                     width: f64,
                     height: f64,
                     depth: f64| {
        let (grid_x, grid_y) = (1usize, 1usize);
        let seg_w = width / grid_x as f64;
        let seg_h = height / grid_y as f64;
        let (wh, hh, dh) = (width / 2.0, height / 2.0, depth / 2.0);
        let (gx1, gy1) = (grid_x + 1, grid_y + 1);
        let mut counter = 0u32;
        for iy in 0..gy1 {
            let y = iy as f64 * seg_h - hh;
            for ix in 0..gx1 {
                let x = ix as f64 * seg_w - wh;
                let mut vector = [0.0; 3];
                vector[u] = x * udir;
                vector[v] = y * vdir;
                vector[w] = dh;
                vertices.extend_from_slice(&vector);
                let mut n = [0.0; 3];
                n[w] = if depth > 0.0 { 1.0 } else { -1.0 };
                normals.extend_from_slice(&n);
                uvs.push(ix as f64 / grid_x as f64);
                uvs.push(1.0 - iy as f64 / grid_y as f64);
                counter += 1;
            }
        }
        let gx1 = gx1 as u32;
        for iy in 0..grid_y as u32 {
            for ix in 0..grid_x as u32 {
                let a = count + ix + gx1 * iy;
                let b = count + ix + gx1 * (iy + 1);
                let c = count + (ix + 1) + gx1 * (iy + 1);
                let d = count + (ix + 1) + gx1 * iy;
                indices.extend_from_slice(&[a, b, d, b, c, d]);
            }
        }
        count += counter;
    };
    let (x, y, z) = (0, 1, 2);
    plane(z, y, x, -1.0, -1.0, depth, height, width);
    plane(z, y, x, 1.0, -1.0, depth, height, -width);
    plane(x, z, y, 1.0, 1.0, width, depth, height);
    plane(x, z, y, 1.0, -1.0, width, depth, -height);
    plane(x, y, z, 1.0, -1.0, width, height, depth);
    plane(x, y, z, -1.0, -1.0, width, height, -depth);
    GizmoGeometry::from_f64(&vertices, &normals, &uvs, Some(indices))
}

/// three.js `PlaneGeometry(width, height, widthSegments, heightSegments)`.
pub fn plane_geometry(
    width: f64,
    height: f64,
    width_segments: usize,
    height_segments: usize,
) -> GizmoGeometry {
    let (wh, hh) = (width / 2.0, height / 2.0);
    let (gx, gy) = (width_segments, height_segments);
    let (gx1, gy1) = (gx + 1, gy + 1);
    let (sw, sh) = (width / gx as f64, height / gy as f64);
    let (mut vertices, mut normals, mut uvs, mut indices) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for iy in 0..gy1 {
        let y = iy as f64 * sh - hh;
        for ix in 0..gx1 {
            let x = ix as f64 * sw - wh;
            vertices.extend_from_slice(&[x, -y, 0.0]);
            normals.extend_from_slice(&[0.0, 0.0, 1.0]);
            uvs.push(ix as f64 / gx as f64);
            uvs.push(1.0 - iy as f64 / gy as f64);
        }
    }
    let gx1 = gx1 as u32;
    for iy in 0..gy as u32 {
        for ix in 0..gx as u32 {
            let a = ix + gx1 * iy;
            let b = ix + gx1 * (iy + 1);
            let c = (ix + 1) + gx1 * (iy + 1);
            let d = (ix + 1) + gx1 * iy;
            indices.extend_from_slice(&[a, b, d, b, c, d]);
        }
    }
    GizmoGeometry::from_f64(&vertices, &normals, &uvs, Some(indices))
}

/// three.js `TorusGeometry(radius, tube, radialSegments, tubularSegments, arc)`.
pub fn torus(
    radius: f64,
    tube: f64,
    radial_segments: usize,
    tubular_segments: usize,
    arc: f64,
) -> GizmoGeometry {
    let (rs, ts) = (radial_segments, tubular_segments);
    let (mut vertices, mut normals, mut uvs, mut indices) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for j in 0..=rs {
        for i in 0..=ts {
            let u = i as f64 / ts as f64 * arc;
            let v = j as f64 / rs as f64 * PI * 2.0;
            let vertex = [
                (radius + tube * js::cos(v)) * js::cos(u),
                (radius + tube * js::cos(v)) * js::sin(u),
                tube * js::sin(v),
            ];
            vertices.extend_from_slice(&vertex);
            let center = [radius * js::cos(u), radius * js::sin(u), 0.0];
            normals.extend_from_slice(&math::normalize(math::sub(vertex, center)));
            uvs.push(i as f64 / ts as f64);
            uvs.push(j as f64 / rs as f64);
        }
    }
    let row = (ts + 1) as u32;
    for j in 1..=rs as u32 {
        for i in 1..=ts as u32 {
            let a = row * j + i - 1;
            let b = row * (j - 1) + i - 1;
            let c = row * (j - 1) + i;
            let d = row * j + i;
            indices.extend_from_slice(&[a, b, d, b, c, d]);
        }
    }
    GizmoGeometry::from_f64(&vertices, &normals, &uvs, Some(indices))
}

/// three.js `SphereGeometry(radius, widthSegments, heightSegments)`.
pub fn sphere(radius: f64, width_segments: usize, height_segments: usize) -> GizmoGeometry {
    let ws = width_segments.max(3);
    let hs = height_segments.max(2);
    let (phi_start, phi_length, theta_start, theta_length) = (0.0, PI * 2.0, 0.0, PI);
    let theta_end = (theta_start + theta_length).min(PI);
    let (mut vertices, mut normals, mut uvs, mut indices) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut grid: Vec<Vec<u32>> = Vec::new();
    let mut index = 0u32;
    for iy in 0..=hs {
        let mut row = Vec::new();
        let v = iy as f64 / hs as f64;
        let mut u_offset = 0.0;
        if iy == 0 && theta_start == 0.0 {
            u_offset = 0.5 / ws as f64;
        } else if iy == hs && theta_end == PI {
            u_offset = -0.5 / ws as f64;
        }
        for ix in 0..=ws {
            let u = ix as f64 / ws as f64;
            let vertex = [
                -radius
                    * js::cos(phi_start + u * phi_length)
                    * js::sin(theta_start + v * theta_length),
                radius * js::cos(theta_start + v * theta_length),
                radius
                    * js::sin(phi_start + u * phi_length)
                    * js::sin(theta_start + v * theta_length),
            ];
            vertices.extend_from_slice(&vertex);
            normals.extend_from_slice(&math::normalize(vertex));
            uvs.extend_from_slice(&[u + u_offset, 1.0 - v]);
            row.push(index);
            index += 1;
        }
        grid.push(row);
    }
    for iy in 0..hs {
        for ix in 0..ws {
            let a = grid[iy][ix + 1];
            let b = grid[iy][ix];
            let c = grid[iy + 1][ix];
            let d = grid[iy + 1][ix + 1];
            if iy != 0 || theta_start > 0.0 {
                indices.extend_from_slice(&[a, b, d]);
            }
            if iy != hs - 1 || theta_end < PI {
                indices.extend_from_slice(&[b, c, d]);
            }
        }
    }
    GizmoGeometry::from_f64(&vertices, &normals, &uvs, Some(indices))
}

/// three.js `OctahedronGeometry(radius, detail)` — `PolyhedronGeometry`'s
/// subdivision, projection, UVs and normals (flat for detail 0, smooth
/// otherwise).
pub fn octahedron(radius: f64, detail: usize) -> GizmoGeometry {
    const VERTICES: [f64; 18] = [
        1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0,
    ];
    const INDICES: [usize; 24] = [
        0, 2, 4, 0, 4, 3, 0, 3, 5, 0, 5, 2, 1, 2, 5, 1, 5, 3, 1, 3, 4, 1, 4, 2,
    ];
    let vtx = |i: usize| [VERTICES[i * 3], VERTICES[i * 3 + 1], VERTICES[i * 3 + 2]];
    let lerp = |a: V3, b: V3, t: f64| -> V3 {
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ]
    };
    let mut vb: Vec<f64> = Vec::new();
    for f in INDICES.chunks_exact(3) {
        let (a, b, c) = (vtx(f[0]), vtx(f[1]), vtx(f[2]));
        let cols = detail + 1;
        let mut v: Vec<Vec<V3>> = Vec::new();
        for i in 0..=cols {
            let aj = lerp(a, c, i as f64 / cols as f64);
            let bj = lerp(b, c, i as f64 / cols as f64);
            let rows = cols - i;
            let mut row = Vec::new();
            for j in 0..=rows {
                if j == 0 && i == cols {
                    row.push(aj);
                } else {
                    row.push(lerp(aj, bj, j as f64 / rows as f64));
                }
            }
            v.push(row);
        }
        for i in 0..cols {
            for j in 0..(2 * (cols - i) - 1) {
                let k = j / 2;
                let tri = if j % 2 == 0 {
                    [v[i][k + 1], v[i + 1][k], v[i][k]]
                } else {
                    [v[i][k + 1], v[i + 1][k + 1], v[i + 1][k]]
                };
                for p in tri {
                    vb.extend_from_slice(&p);
                }
            }
        }
    }
    for p in vb.chunks_exact_mut(3) {
        let n = math::scale(math::normalize([p[0], p[1], p[2]]), radius);
        p.copy_from_slice(&n);
    }
    let azimuth = |v: V3| js::atan2(v[2], -v[0]);
    let inclination = |v: V3| js::atan2(-v[1], (v[0] * v[0] + v[2] * v[2]).sqrt());
    let mut uv: Vec<f64> = Vec::new();
    for p in vb.chunks_exact(3) {
        let v = [p[0], p[1], p[2]];
        let u = azimuth(v) / 2.0 / PI + 0.5;
        let w = inclination(v) / PI + 0.5;
        uv.extend_from_slice(&[u, 1.0 - w]);
    }
    // correctUVs
    let mut j = 0;
    for t in vb.chunks_exact(9) {
        let a = [t[0], t[1], t[2]];
        let b = [t[3], t[4], t[5]];
        let c = [t[6], t[7], t[8]];
        let uvs = [uv[j], uv[j + 2], uv[j + 4]];
        let centroid = math::scale(math::add(math::add(a, b), c), 1.0 / 3.0);
        let azi = azimuth(centroid);
        for (k, p) in [a, b, c].into_iter().enumerate() {
            let stride = j + k * 2;
            if azi < 0.0 && uvs[k] == 1.0 {
                uv[stride] = uvs[k] - 1.0;
            }
            if p[0] == 0.0 && p[2] == 0.0 {
                uv[stride] = azi / 2.0 / PI + 0.5;
            }
        }
        j += 6;
    }
    // correctSeam
    for t in uv.chunks_exact_mut(6) {
        let (x0, x1, x2) = (t[0], t[2], t[4]);
        let max = x0.max(x1).max(x2);
        let min = x0.min(x1).min(x2);
        if max > 0.9 && min < 0.1 {
            if x0 < 0.2 {
                t[0] += 1.0;
            }
            if x1 < 0.2 {
                t[2] += 1.0;
            }
            if x2 < 0.2 {
                t[4] += 1.0;
            }
        }
    }
    let mut g = GizmoGeometry::from_f64(&vb, &vb, &uv, None);
    if detail == 0 {
        // computeVertexNormals (non-indexed): the face cross product, stored,
        // then normalizeNormals.
        for (t, n) in g.position.chunks_exact(9).zip(g.normal.chunks_exact_mut(9)) {
            let p = |i: usize| [t[i] as f64, t[i + 1] as f64, t[i + 2] as f64];
            let (pa, pb, pc) = (p(0), p(3), p(6));
            let cb = math::cross(math::sub(pc, pb), math::sub(pa, pb));
            for k in 0..3 {
                n[k * 3] = cb[0] as f32;
                n[k * 3 + 1] = cb[1] as f32;
                n[k * 3 + 2] = cb[2] as f32;
            }
        }
    }
    for n in g.normal.chunks_exact_mut(3) {
        let v = math::normalize([n[0] as f64, n[1] as f64, n[2] as f64]);
        n[0] = v[0] as f32;
        n[1] = v[1] as f32;
        n[2] = v[2] as f32;
    }
    g
}

/// A line's positions (`new Float32BufferAttribute([...], 3)`).
fn line(points: &[f64]) -> GizmoGeometry {
    GizmoGeometry {
        position: points.iter().map(|&v| v as f32).collect(),
        ..Default::default()
    }
}

/// The gizmo's materials, one per three.js material object (clones are
/// separate entries; handles sharing one share its highlight).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialDef {
    /// `LineBasicMaterial` rather than `MeshBasicMaterial`.
    pub line: bool,
    pub color: u32,
    pub opacity: f64,
}

/// Indices into [`MATERIALS`].
mod mat {
    pub const INVISIBLE: usize = 0;
    pub const HELPER: usize = 1;
    pub const RED: usize = 2;
    pub const GREEN: usize = 3;
    pub const BLUE: usize = 4;
    pub const RED_T: usize = 5;
    pub const GREEN_T: usize = 6;
    pub const BLUE_T: usize = 7;
    pub const YELLOW_T: usize = 8;
    pub const GRAY: usize = 9;
    pub const T_XYZ: usize = 10;
    pub const T_XY: usize = 11;
    pub const T_YZ: usize = 12;
    pub const T_XZ: usize = 13;
    pub const HT_X: usize = 14;
    pub const HT_Y: usize = 15;
    pub const HT_Z: usize = 16;
    pub const HR_AXIS: usize = 17;
    pub const S_XYZ: usize = 18;
    pub const HS_X: usize = 19;
    pub const HS_Y: usize = 20;
    pub const HS_Z: usize = 21;
}

const fn mesh_mat(color: u32, opacity: f64) -> MaterialDef {
    MaterialDef {
        line: false,
        color,
        opacity,
    }
}
const HELPER_MAT: MaterialDef = MaterialDef {
    line: true,
    color: 0xffffff,
    opacity: 0.5,
};

/// Every material three.js's gizmo uses. All are `depthTest: false`,
/// `depthWrite: false`, `fog: false`, `toneMapped: false`,
/// `transparent: true`.
pub const MATERIALS: [MaterialDef; 22] = [
    mesh_mat(0xffffff, 0.15), // matInvisible
    HELPER_MAT,               // matHelper
    mesh_mat(0xff0000, 1.0),  // matRed
    mesh_mat(0x00ff00, 1.0),  // matGreen
    mesh_mat(0x0000ff, 1.0),  // matBlue
    mesh_mat(0xff0000, 0.5),  // matRedTransparent
    mesh_mat(0x00ff00, 0.5),  // matGreenTransparent
    mesh_mat(0x0000ff, 0.5),  // matBlueTransparent
    mesh_mat(0xffff00, 0.25), // matYellowTransparent
    mesh_mat(0x787878, 1.0),  // matGray
    mesh_mat(0xffffff, 0.25), // matWhiteTransparent.clone() — translate XYZ
    mesh_mat(0x0000ff, 0.5),  // matBlueTransparent.clone() — translate XY
    mesh_mat(0xff0000, 0.5),  // matRedTransparent.clone() — translate YZ
    mesh_mat(0x00ff00, 0.5),  // matGreenTransparent.clone() — translate XZ
    HELPER_MAT,               // matHelper.clone() — translate helper X
    HELPER_MAT,               // … Y
    HELPER_MAT,               // … Z
    HELPER_MAT,               // rotate helper AXIS
    mesh_mat(0xffffff, 0.25), // matWhiteTransparent.clone() — scale XYZ
    HELPER_MAT,               // scale helper X
    HELPER_MAT,               // … Y
    HELPER_MAT,               // … Z
];

/// Which of the gizmo's nine groups a handle is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Group {
    Gizmo,
    Picker,
    Helper,
}

/// `translate`, `rotate`, `scale` — the index into three.js's
/// `gizmo[mode]` / `picker[mode]` / `helper[mode]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Translate,
    Rotate,
    Scale,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "translate" => Some(Mode::Translate),
            "rotate" => Some(Mode::Rotate),
            "scale" => Some(Mode::Scale),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Mode::Translate => "translate",
            Mode::Rotate => "rotate",
            Mode::Scale => "scale",
        }
    }
}

/// One handle as `setupGizmo` leaves it: in its group, named, maybe tagged
/// `helper`, drawn with a material from [`MATERIALS`], with its baked
/// geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct Handle {
    pub group: Group,
    pub mode: Mode,
    pub name: &'static str,
    pub helper: bool,
    /// A `Line` (strip) rather than a `Mesh`.
    pub line: bool,
    pub material: usize,
    pub geometry: GizmoGeometry,
}

#[derive(Clone, Copy)]
enum Geo {
    Arrow,
    ScaleHandle,
    Line,
    Line2,
    Circle(f64, f64),
    TranslateHelper,
    Octa(f64, usize),
    Box(f64, f64, f64),
    Cyl(f64, f64, f64, usize),
    Sphere(f64, usize, usize),
    Torus(f64, f64, usize, usize),
}

/// One entry of a gizmo table: `[object, position, rotation, scale, tag]`.
struct Entry {
    geo: Geo,
    mat: usize,
    line: bool,
    pos: Option<V3>,
    rot: Option<V3>,
    scale: Option<V3>,
    helper: bool,
}

fn m(geo: Geo, mat: usize, pos: Option<V3>, rot: Option<V3>) -> Entry {
    Entry {
        geo,
        mat,
        line: false,
        pos,
        rot,
        scale: None,
        helper: false,
    }
}

fn helper_mesh(geo: Geo, mat: usize) -> Entry {
    Entry {
        geo,
        mat,
        line: false,
        pos: None,
        rot: None,
        scale: None,
        helper: true,
    }
}

fn helper_line(geo: Geo, mat: usize, pos: Option<V3>, rot: Option<V3>, scale: Option<V3>) -> Entry {
    Entry {
        geo,
        mat,
        line: true,
        pos,
        rot,
        scale,
        helper: true,
    }
}

fn build(geo: Geo) -> GizmoGeometry {
    match geo {
        Geo::Arrow => cylinder(0.0, 0.04, 0.1, 12).translate(0.0, 0.05, 0.0),
        Geo::ScaleHandle => box_geometry(0.08, 0.08, 0.08).translate(0.0, 0.04, 0.0),
        Geo::Line => line(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        Geo::Line2 => cylinder(0.0075, 0.0075, 0.5, 3).translate(0.0, 0.25, 0.0),
        Geo::Circle(radius, arc) => {
            let mut g = torus(radius, 0.0075, 3, 64, arc * PI * 2.0);
            g.apply_matrix4(&math::make_rotation_y(PI / 2.0));
            g.apply_matrix4(&math::make_rotation_x(PI / 2.0));
            g
        }
        Geo::TranslateHelper => line(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        Geo::Octa(r, d) => octahedron(r, d),
        Geo::Box(w, h, d) => box_geometry(w, h, d),
        Geo::Cyl(rt, rb, h, s) => cylinder(rt, rb, h, s),
        Geo::Sphere(r, w, h) => sphere(r, w, h),
        Geo::Torus(r, t, rs, ts) => torus(r, t, rs, ts, PI * 2.0),
    }
}

type Table = Vec<(&'static str, Vec<Entry>)>;

fn tables() -> Vec<(Group, Mode, Table)> {
    use mat::*;
    let h = PI / 2.0;
    let s = Some;
    let picker_cyl = Geo::Cyl(0.2, 0.0, 0.6, 4);
    let plane_handles = |picker: bool, xy: usize, yz: usize, xz: usize| -> Table {
        let g = if picker {
            Geo::Box(0.2, 0.2, 0.01)
        } else {
            Geo::Box(0.15, 0.15, 0.01)
        };
        vec![
            ("XY", vec![m(g, xy, s([0.15, 0.15, 0.0]), None)]),
            ("YZ", vec![m(g, yz, s([0.0, 0.15, 0.15]), s([0.0, h, 0.0]))]),
            (
                "XZ",
                vec![m(g, xz, s([0.15, 0.0, 0.15]), s([-h, 0.0, 0.0]))],
            ),
        ]
    };
    let axis_pickers = || -> Table {
        vec![
            (
                "X",
                vec![
                    m(picker_cyl, INVISIBLE, s([0.3, 0.0, 0.0]), s([0.0, 0.0, -h])),
                    m(picker_cyl, INVISIBLE, s([-0.3, 0.0, 0.0]), s([0.0, 0.0, h])),
                ],
            ),
            (
                "Y",
                vec![
                    m(picker_cyl, INVISIBLE, s([0.0, 0.3, 0.0]), None),
                    m(
                        picker_cyl,
                        INVISIBLE,
                        s([0.0, -0.3, 0.0]),
                        s([0.0, 0.0, PI]),
                    ),
                ],
            ),
            (
                "Z",
                vec![
                    m(picker_cyl, INVISIBLE, s([0.0, 0.0, 0.3]), s([h, 0.0, 0.0])),
                    m(
                        picker_cyl,
                        INVISIBLE,
                        s([0.0, 0.0, -0.3]),
                        s([-h, 0.0, 0.0]),
                    ),
                ],
            ),
        ]
    };
    let axis_lines = |x: usize, y: usize, z: usize| -> Table {
        let big = s([1e6, 1.0, 1.0]);
        vec![
            (
                "X",
                vec![helper_line(Geo::Line, x, s([-1e3, 0.0, 0.0]), None, big)],
            ),
            (
                "Y",
                vec![helper_line(
                    Geo::Line,
                    y,
                    s([0.0, -1e3, 0.0]),
                    s([0.0, 0.0, h]),
                    big,
                )],
            ),
            (
                "Z",
                vec![helper_line(
                    Geo::Line,
                    z,
                    s([0.0, 0.0, -1e3]),
                    s([0.0, -h, 0.0]),
                    big,
                )],
            ),
        ]
    };

    let mut gizmo_translate: Table = vec![
        (
            "X",
            vec![
                m(Geo::Arrow, RED, s([0.5, 0.0, 0.0]), s([0.0, 0.0, -h])),
                m(Geo::Arrow, RED, s([-0.5, 0.0, 0.0]), s([0.0, 0.0, h])),
                m(Geo::Line2, RED, s([0.0, 0.0, 0.0]), s([0.0, 0.0, -h])),
            ],
        ),
        (
            "Y",
            vec![
                m(Geo::Arrow, GREEN, s([0.0, 0.5, 0.0]), None),
                m(Geo::Arrow, GREEN, s([0.0, -0.5, 0.0]), s([PI, 0.0, 0.0])),
                m(Geo::Line2, GREEN, None, None),
            ],
        ),
        (
            "Z",
            vec![
                m(Geo::Arrow, BLUE, s([0.0, 0.0, 0.5]), s([h, 0.0, 0.0])),
                m(Geo::Arrow, BLUE, s([0.0, 0.0, -0.5]), s([-h, 0.0, 0.0])),
                m(Geo::Line2, BLUE, None, s([h, 0.0, 0.0])),
            ],
        ),
        (
            "XYZ",
            vec![m(Geo::Octa(0.1, 0), T_XYZ, s([0.0, 0.0, 0.0]), None)],
        ),
    ];
    gizmo_translate.extend(plane_handles(false, T_XY, T_YZ, T_XZ));

    let mut picker_translate = axis_pickers();
    picker_translate.push(("XYZ", vec![m(Geo::Octa(0.2, 0), INVISIBLE, None, None)]));
    picker_translate.extend(plane_handles(true, INVISIBLE, INVISIBLE, INVISIBLE));

    let mut helper_translate: Table = vec![
        ("START", vec![helper_mesh(Geo::Octa(0.01, 2), HELPER)]),
        ("END", vec![helper_mesh(Geo::Octa(0.01, 2), HELPER)]),
        (
            "DELTA",
            vec![helper_line(Geo::TranslateHelper, HELPER, None, None, None)],
        ),
    ];
    helper_translate.extend(axis_lines(HT_X, HT_Y, HT_Z));

    let gizmo_rotate: Table = vec![
        (
            "XYZE",
            vec![m(Geo::Circle(0.5, 1.0), GRAY, None, s([0.0, h, 0.0]))],
        ),
        ("X", vec![m(Geo::Circle(0.5, 0.5), RED, None, None)]),
        (
            "Y",
            vec![m(Geo::Circle(0.5, 0.5), GREEN, None, s([0.0, 0.0, -h]))],
        ),
        (
            "Z",
            vec![m(Geo::Circle(0.5, 0.5), BLUE, None, s([0.0, h, 0.0]))],
        ),
        (
            "E",
            vec![m(Geo::Circle(0.75, 1.0), YELLOW_T, None, s([0.0, h, 0.0]))],
        ),
    ];
    let helper_rotate: Table = vec![(
        "AXIS",
        vec![helper_line(
            Geo::Line,
            HR_AXIS,
            s([-1e3, 0.0, 0.0]),
            None,
            s([1e6, 1.0, 1.0]),
        )],
    )];
    let picker_rotate: Table = vec![
        (
            "XYZE",
            vec![m(Geo::Sphere(0.25, 10, 8), INVISIBLE, None, None)],
        ),
        (
            "X",
            vec![m(
                Geo::Torus(0.5, 0.1, 4, 24),
                INVISIBLE,
                s([0.0, 0.0, 0.0]),
                s([0.0, -h, -h]),
            )],
        ),
        (
            "Y",
            vec![m(
                Geo::Torus(0.5, 0.1, 4, 24),
                INVISIBLE,
                s([0.0, 0.0, 0.0]),
                s([h, 0.0, 0.0]),
            )],
        ),
        (
            "Z",
            vec![m(
                Geo::Torus(0.5, 0.1, 4, 24),
                INVISIBLE,
                s([0.0, 0.0, 0.0]),
                s([0.0, 0.0, -h]),
            )],
        ),
        (
            "E",
            vec![m(Geo::Torus(0.75, 0.1, 2, 24), INVISIBLE, None, None)],
        ),
    ];

    let mut gizmo_scale: Table = vec![
        (
            "X",
            vec![
                m(Geo::ScaleHandle, RED, s([0.5, 0.0, 0.0]), s([0.0, 0.0, -h])),
                m(Geo::Line2, RED, s([0.0, 0.0, 0.0]), s([0.0, 0.0, -h])),
                m(Geo::ScaleHandle, RED, s([-0.5, 0.0, 0.0]), s([0.0, 0.0, h])),
            ],
        ),
        (
            "Y",
            vec![
                m(Geo::ScaleHandle, GREEN, s([0.0, 0.5, 0.0]), None),
                m(Geo::Line2, GREEN, None, None),
                m(
                    Geo::ScaleHandle,
                    GREEN,
                    s([0.0, -0.5, 0.0]),
                    s([0.0, 0.0, PI]),
                ),
            ],
        ),
        (
            "Z",
            vec![
                m(Geo::ScaleHandle, BLUE, s([0.0, 0.0, 0.5]), s([h, 0.0, 0.0])),
                m(Geo::Line2, BLUE, s([0.0, 0.0, 0.0]), s([h, 0.0, 0.0])),
                m(
                    Geo::ScaleHandle,
                    BLUE,
                    s([0.0, 0.0, -0.5]),
                    s([-h, 0.0, 0.0]),
                ),
            ],
        ),
    ];
    gizmo_scale.extend(plane_handles(false, BLUE_T, RED_T, GREEN_T));
    gizmo_scale.push(("XYZ", vec![m(Geo::Box(0.1, 0.1, 0.1), S_XYZ, None, None)]));

    let mut picker_scale = axis_pickers();
    picker_scale.extend(plane_handles(true, INVISIBLE, INVISIBLE, INVISIBLE));
    picker_scale.push((
        "XYZ",
        vec![m(
            Geo::Box(0.2, 0.2, 0.2),
            INVISIBLE,
            s([0.0, 0.0, 0.0]),
            None,
        )],
    ));

    let helper_scale = axis_lines(HS_X, HS_Y, HS_Z);

    // The order three.js adds them in: gizmo ×3, picker ×3, helper ×3.
    vec![
        (Group::Gizmo, Mode::Translate, gizmo_translate),
        (Group::Gizmo, Mode::Rotate, gizmo_rotate),
        (Group::Gizmo, Mode::Scale, gizmo_scale),
        (Group::Picker, Mode::Translate, picker_translate),
        (Group::Picker, Mode::Rotate, picker_rotate),
        (Group::Picker, Mode::Scale, picker_scale),
        (Group::Helper, Mode::Translate, helper_translate),
        (Group::Helper, Mode::Rotate, helper_rotate),
        (Group::Helper, Mode::Scale, helper_scale),
    ]
}

/// Every handle, in the order three.js's `setupGizmo` adds them: groups in
/// creation order, names in table order, entries within a name last first.
pub fn handles() -> Vec<Handle> {
    let mut out = Vec::new();
    for (group, mode, table) in tables() {
        for (name, entries) in table {
            for e in entries.iter().rev() {
                let mut geometry = build(e.geo);
                let q = match e.rot {
                    Some([x, y, z]) => math::quat_from_euler_xyz(x, y, z),
                    None => math::IDENTITY_Q,
                };
                let matrix = math::compose(
                    e.pos.unwrap_or([0.0; 3]),
                    q,
                    e.scale.unwrap_or([1.0, 1.0, 1.0]),
                );
                geometry.apply_matrix4(&matrix);
                out.push(Handle {
                    group,
                    mode,
                    name,
                    helper: e.helper,
                    line: e.line,
                    material: e.mat,
                    geometry,
                });
            }
        }
    }
    out
}

/// `TransformControlsPlane`'s geometry: `PlaneGeometry(100000, 100000, 2, 2)`.
pub fn plane() -> GizmoGeometry {
    plane_geometry(100000.0, 100000.0, 2, 2)
}
