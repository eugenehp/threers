//! three.js r165's debug helpers as data: the world-space line segments (and,
//! where three.js draws one, the filled mesh) each helper puts on screen.
//!
//! three.js builds these helpers as small `Object3D` trees — a `Line` that
//! `lookAt`s the light's target, a wireframe `Mesh` scaled to the cone, a
//! `LineSegments` whose matrix *is* the camera's `matrixWorld` — and lets the
//! renderer compose the transforms. Here the same trees are composed once, in
//! `f64` as JavaScript does, and flattened: every helper becomes a line list in
//! world space (`Line` strips are expanded into segments, wireframe meshes
//! into the edges WebGL's `getWireframeAttribute` draws), plus a triangle list
//! for the parts three.js fills (`ArrowHelper`'s cone, `PlaneHelper`'s and
//! `RectAreaLightHelper`'s quads). The browser shim draws these with an
//! identity transform and recomputes them in `update()`.
//!
//! Matrices are three.js `Matrix4.elements`: column-major, 16 numbers.
//! Every helper is taken to sit at the scene root, which is where three.js's
//! own examples put them (a helper parented elsewhere composes with its
//! parent's transform in three.js, and with nothing here).

/// Column-major 4×4 matrix, as three.js's `Matrix4.elements`.
pub type Mat4 = [f64; 16];

/// Identity.
pub const IDENTITY: Mat4 = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// What a helper draws, in world space.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HelperGeometry {
    /// Line list: pairs of `xyz` points.
    pub lines: Vec<f32>,
    /// Per-point `rgb` for `lines`, or empty when the helper draws in one
    /// material colour.
    pub colors: Vec<f32>,
    /// Triangle list (`xyz` per corner) for the filled part, or empty.
    pub mesh: Vec<f32>,
}

// ---------------------------------------------------------------- math ----

type V3 = [f64; 3];
type Quat = [f64; 4];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn length_sq(a: V3) -> f64 {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}
/// `Vector3.normalize`: divide by the length, or by 1 when it is 0.
fn normalize(a: V3) -> V3 {
    let l = length_sq(a).sqrt();
    let l = if l == 0.0 { 1.0 } else { l };
    [a[0] / l, a[1] / l, a[2] / l]
}

/// `Matrix4.multiplyMatrices(a, b)`.
pub fn multiply(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [0.0; 16];
    for row in 0..4 {
        for col in 0..4 {
            // Same summation order as three.js: a_r1*b_1c + a_r2*b_2c + …
            out[col * 4 + row] = a[row] * b[col * 4]
                + a[4 + row] * b[col * 4 + 1]
                + a[8 + row] * b[col * 4 + 2]
                + a[12 + row] * b[col * 4 + 3];
        }
    }
    out
}

/// `Matrix4.invert`, operation for operation (a singular matrix gives zeros).
pub fn invert(te: &Mat4) -> Mat4 {
    let (n11, n21, n31, n41) = (te[0], te[1], te[2], te[3]);
    let (n12, n22, n32, n42) = (te[4], te[5], te[6], te[7]);
    let (n13, n23, n33, n43) = (te[8], te[9], te[10], te[11]);
    let (n14, n24, n34, n44) = (te[12], te[13], te[14], te[15]);
    let t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43
        - n23 * n32 * n44
        + n22 * n33 * n44;
    let t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43
        + n13 * n32 * n44
        - n12 * n33 * n44;
    let t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43
        - n13 * n22 * n44
        + n12 * n23 * n44;
    let t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33
        + n13 * n22 * n34
        - n12 * n23 * n34;
    let det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
    if det == 0.0 {
        return [0.0; 16];
    }
    let d = 1.0 / det;
    [
        t11 * d,
        (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44
            - n21 * n33 * n44)
            * d,
        (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44
            + n21 * n32 * n44)
            * d,
        (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43
            - n21 * n32 * n43)
            * d,
        t12 * d,
        (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44
            + n11 * n33 * n44)
            * d,
        (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44
            - n11 * n32 * n44)
            * d,
        (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43
            + n11 * n32 * n43)
            * d,
        t13 * d,
        (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44
            - n11 * n23 * n44)
            * d,
        (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44
            + n11 * n22 * n44)
            * d,
        (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43
            - n11 * n22 * n43)
            * d,
        t14 * d,
        (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34
            + n11 * n23 * n34)
            * d,
        (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34
            - n11 * n22 * n34)
            * d,
        (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33
            + n11 * n22 * n33)
            * d,
    ]
}

/// `Vector3.applyMatrix4`, perspective divide included.
pub fn apply(e: &Mat4, p: V3) -> V3 {
    let [x, y, z] = p;
    let w = 1.0 / (e[3] * x + e[7] * y + e[11] * z + e[15]);
    [
        (e[0] * x + e[4] * y + e[8] * z + e[12]) * w,
        (e[1] * x + e[5] * y + e[9] * z + e[13]) * w,
        (e[2] * x + e[6] * y + e[10] * z + e[14]) * w,
    ]
}

/// `Matrix4.compose(position, quaternion, scale)`.
pub fn compose(p: V3, q: Quat, s: V3) -> Mat4 {
    let [x, y, z, w] = q;
    let (x2, y2, z2) = (x + x, y + y, z + z);
    let (xx, xy, xz) = (x * x2, x * y2, x * z2);
    let (yy, yz, zz) = (y * y2, y * z2, z * z2);
    let (wx, wy, wz) = (w * x2, w * y2, w * z2);
    [
        (1.0 - (yy + zz)) * s[0],
        (xy + wz) * s[0],
        (xz - wy) * s[0],
        0.0,
        (xy - wz) * s[1],
        (1.0 - (xx + zz)) * s[1],
        (yz + wx) * s[1],
        0.0,
        (xz + wy) * s[2],
        (yz - wx) * s[2],
        (1.0 - (xx + yy)) * s[2],
        0.0,
        p[0],
        p[1],
        p[2],
        1.0,
    ]
}

/// `Matrix4.extractRotation`.
fn extract_rotation(m: &Mat4) -> Mat4 {
    let col = |c: usize| (m[c * 4] * m[c * 4] + m[c * 4 + 1] * m[c * 4 + 1] + m[c * 4 + 2] * m[c * 4 + 2]).sqrt();
    let (sx, sy, sz) = (1.0 / col(0), 1.0 / col(1), 1.0 / col(2));
    [
        m[0] * sx,
        m[1] * sx,
        m[2] * sx,
        0.0,
        m[4] * sy,
        m[5] * sy,
        m[6] * sy,
        0.0,
        m[8] * sz,
        m[9] * sz,
        m[10] * sz,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

/// `Quaternion.setFromRotationMatrix`.
fn quat_from_rotation(te: &Mat4) -> Quat {
    let (m11, m12, m13) = (te[0], te[4], te[8]);
    let (m21, m22, m23) = (te[1], te[5], te[9]);
    let (m31, m32, m33) = (te[2], te[6], te[10]);
    let trace = m11 + m22 + m33;
    if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        [(m32 - m23) * s, (m13 - m31) * s, (m21 - m12) * s, 0.25 / s]
    } else if m11 > m22 && m11 > m33 {
        let s = 2.0 * (1.0 + m11 - m22 - m33).sqrt();
        [0.25 * s, (m12 + m21) / s, (m13 + m31) / s, (m32 - m23) / s]
    } else if m22 > m33 {
        let s = 2.0 * (1.0 + m22 - m11 - m33).sqrt();
        [(m12 + m21) / s, 0.25 * s, (m23 + m32) / s, (m13 - m31) / s]
    } else {
        let s = 2.0 * (1.0 + m33 - m11 - m22).sqrt();
        [(m13 + m31) / s, (m23 + m32) / s, 0.25 * s, (m21 - m12) / s]
    }
}

/// `Quaternion.multiplyQuaternions(a, b)`.
fn quat_mul(a: Quat, b: Quat) -> Quat {
    let [qax, qay, qaz, qaw] = a;
    let [qbx, qby, qbz, qbw] = b;
    [
        qax * qbw + qaw * qbx + qay * qbz - qaz * qby,
        qay * qbw + qaw * qby + qaz * qbx - qax * qbz,
        qaz * qbw + qaw * qbz + qax * qby - qay * qbx,
        qaw * qbw - qax * qbx - qay * qby - qaz * qbz,
    ]
}

/// `Matrix4.lookAt(eye, target, up)` — the rotation part only, as three.js
/// writes it into a scratch matrix whose other entries do not matter here.
fn look_at_matrix(eye: V3, target: V3, up: V3) -> Mat4 {
    let mut z = sub(eye, target);
    if length_sq(z) == 0.0 {
        z[2] = 1.0;
    }
    z = normalize(z);
    let mut x = cross(up, z);
    if length_sq(x) == 0.0 {
        if up[2].abs() == 1.0 {
            z[0] += 0.0001;
        } else {
            z[2] += 0.0001;
        }
        z = normalize(z);
        x = cross(up, z);
    }
    x = normalize(x);
    let y = cross(z, x);
    let mut te = IDENTITY;
    te[0] = x[0];
    te[4] = y[0];
    te[8] = z[0];
    te[1] = x[1];
    te[5] = y[1];
    te[9] = z[1];
    te[2] = x[2];
    te[6] = y[2];
    te[10] = z[2];
    te
}

fn position_of(m: &Mat4) -> V3 {
    [m[12], m[13], m[14]]
}

const UP: V3 = [0.0, 1.0, 0.0];

/// The world matrix of a child of `parent` (a plain `Object3D`, not a camera
/// or light) at `position` with `scale`, after `child.lookAt(target)` — the
/// quaternion three.js computes, with the parent's rotation divided out, then
/// composed back under the parent.
fn child_looking_at(parent: &Mat4, position: V3, scale: V3, target: V3) -> Mat4 {
    let world_pos = position_of(&multiply(parent, &compose(position, [0.0, 0.0, 0.0, 1.0], scale)));
    let mut q = quat_from_rotation(&look_at_matrix(target, world_pos, UP));
    let pq = quat_from_rotation(&extract_rotation(parent));
    // `premultiply(_q1.invert())`: the conjugate, then on the left.
    q = quat_mul([-pq[0], -pq[1], -pq[2], pq[3]], q);
    multiply(parent, &compose(position, q, scale))
}

/// `Vector3.applyQuaternion`.
fn apply_quat(v: V3, q: Quat) -> V3 {
    let [vx, vy, vz] = v;
    let [qx, qy, qz, qw] = q;
    let tx = 2.0 * (qy * vz - qz * vy);
    let ty = 2.0 * (qz * vx - qx * vz);
    let tz = 2.0 * (qx * vy - qy * vx);
    [
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    ]
}

/// A `Float32BufferAttribute` round trip.
fn f32v(v: V3) -> V3 {
    [v[0] as f32 as f64, v[1] as f32 as f64, v[2] as f32 as f64]
}

fn push(out: &mut Vec<f32>, p: V3) {
    out.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32]);
}

/// Points of a `Line` strip under `m`, expanded into segments.
fn strip(out: &mut Vec<f32>, m: &Mat4, points: &[V3]) {
    for w in points.windows(2) {
        push(out, apply(m, f32v(w[0])));
        push(out, apply(m, f32v(w[1])));
    }
}

/// Points of a `LineSegments` list under `m`.
fn segments(out: &mut Vec<f32>, m: &Mat4, points: &[V3]) {
    for p in points {
        push(out, apply(m, f32v(*p)));
    }
}

/// The edges WebGL draws for a wireframe mesh (`getWireframeAttribute`):
/// `a b, b c, c a` per triangle.
fn wireframe(out: &mut Vec<f32>, m: &Mat4, positions: &[V3], triangles: &[[usize; 3]]) {
    for &[a, b, c] in triangles {
        for (i, j) in [(a, b), (b, c), (c, a)] {
            push(out, apply(m, positions[i]));
            push(out, apply(m, positions[j]));
        }
    }
}

// ------------------------------------------------------------- helpers ----

/// `DirectionalLightHelper(light, size)`: a square facing the target and a
/// line to it, under `light.matrixWorld`.
pub fn directional_light(light_world: &Mat4, target_world: V3, size: f64) -> HelperGeometry {
    let v1 = position_of(light_world);
    let v3 = sub(target_world, v1);
    let mut lines = Vec::new();
    let plane = child_looking_at(light_world, [0.0; 3], [1.0; 3], target_world);
    strip(
        &mut lines,
        &plane,
        &[
            [-size, size, 0.0],
            [size, size, 0.0],
            [size, -size, 0.0],
            [-size, -size, 0.0],
            [-size, size, 0.0],
        ],
    );
    let target_line = child_looking_at(light_world, [0.0; 3], [1.0, 1.0, length_sq(v3).sqrt()], target_world);
    strip(&mut lines, &target_line, &[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    HelperGeometry {
        lines,
        ..Default::default()
    }
}

/// `OctahedronGeometry(radius)` as `PolyhedronGeometry` builds it at detail 0,
/// non-indexed: each face `(a, b, c)` comes out as `(b, c, a)`.
fn octahedron(radius: f64) -> Vec<V3> {
    let v: [V3; 6] = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let faces = [
        [0, 2, 4],
        [0, 4, 3],
        [0, 3, 5],
        [0, 5, 2],
        [1, 2, 5],
        [1, 5, 3],
        [1, 3, 4],
        [1, 4, 2],
    ];
    let mut out = Vec::with_capacity(24);
    for [a, b, c] in faces {
        for i in [b, c, a] {
            let n = normalize(v[i]);
            out.push(f32v([n[0] * radius, n[1] * radius, n[2] * radius]));
        }
    }
    out
}

/// `HemisphereLightHelper(light, size)`: an octahedron wireframe turned
/// `π/2` about Y, sky colour on its first half of vertices and ground colour
/// on the rest, looking away from the light's position. With a fixed `color`,
/// pass it as both `sky` and `ground`.
pub fn hemisphere_light(light_world: &Mat4, size: f64, sky: [f32; 3], ground: [f32; 3]) -> HelperGeometry {
    // `geometry.rotateY(Math.PI * 0.5)`, through the Float32 attribute.
    let (s, c) = (std::f64::consts::PI * 0.5).sin_cos();
    let positions: Vec<V3> = octahedron(size)
        .into_iter()
        .map(|[x, y, z]| f32v([c * x + s * z, y, -s * x + c * z]))
        .collect();
    let n = positions.len();
    let lp = position_of(light_world);
    let mesh_world = child_looking_at(light_world, [0.0; 3], [1.0; 3], [-lp[0], -lp[1], -lp[2]]);
    let triangles: Vec<[usize; 3]> = (0..n / 3).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect();
    let mut lines = Vec::new();
    wireframe(&mut lines, &mesh_world, &positions, &triangles);
    let color = |i: usize| if i < n / 2 { sky } else { ground };
    let mut colors = Vec::new();
    for &[a, b, cc] in &triangles {
        for (i, j) in [(a, b), (b, cc), (cc, a)] {
            colors.extend_from_slice(&color(i));
            colors.extend_from_slice(&color(j));
        }
    }
    HelperGeometry {
        lines,
        colors,
        mesh: Vec::new(),
    }
}

/// `SphereGeometry(radius, widthSegments, heightSegments)` positions and its
/// index, as three.js lays them out (full sphere).
fn sphere(radius: f64, ws: usize, hs: usize) -> (Vec<V3>, Vec<[usize; 3]>) {
    use std::f64::consts::PI;
    let (phi_len, theta_len) = (PI * 2.0, PI);
    let mut positions = Vec::new();
    let mut grid = Vec::new();
    for iy in 0..=hs {
        let v = iy as f64 / hs as f64;
        let mut row = Vec::new();
        for ix in 0..=ws {
            let u = ix as f64 / ws as f64;
            positions.push(f32v([
                -radius * (u * phi_len).cos() * (v * theta_len).sin(),
                radius * (v * theta_len).cos(),
                radius * (u * phi_len).sin() * (v * theta_len).sin(),
            ]));
            row.push(positions.len() - 1);
        }
        grid.push(row);
    }
    let mut tris = Vec::new();
    for iy in 0..hs {
        for ix in 0..ws {
            let a = grid[iy][ix + 1];
            let b = grid[iy][ix];
            let c = grid[iy + 1][ix];
            let d = grid[iy + 1][ix + 1];
            if iy != 0 {
                tris.push([a, b, d]);
            }
            if iy != hs - 1 {
                tris.push([b, c, d]);
            }
        }
    }
    (positions, tris)
}

/// `PointLightHelper(light, sphereSize)`: a 4×2 sphere wireframe under
/// `light.matrixWorld`.
pub fn point_light(light_world: &Mat4, sphere_size: f64) -> HelperGeometry {
    let (positions, tris) = sphere(sphere_size, 4, 2);
    let mut lines = Vec::new();
    wireframe(&mut lines, light_world, &positions, &tris);
    HelperGeometry {
        lines,
        ..Default::default()
    }
}

/// `SpotLightHelper(light)`: five spokes and a 32-segment rim, scaled to
/// `distance` (1000 when 0) and `tan(angle)`, looking at the target.
pub fn spot_light(light_world: &Mat4, target_world: V3, distance: f64, angle: f64) -> HelperGeometry {
    let mut pts: Vec<V3> = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, -1.0, 1.0],
    ];
    let l = 32;
    for i in 0..l {
        let p1 = i as f64 / l as f64 * std::f64::consts::PI * 2.0;
        let p2 = (i + 1) as f64 / l as f64 * std::f64::consts::PI * 2.0;
        pts.push([p1.cos(), p1.sin(), 1.0]);
        pts.push([p2.cos(), p2.sin(), 1.0]);
    }
    let cone_length = if distance != 0.0 { distance } else { 1000.0 };
    let cone_width = cone_length * angle.tan();
    let cone = child_looking_at(light_world, [0.0; 3], [cone_width, cone_width, cone_length], target_world);
    let mut lines = Vec::new();
    segments(&mut lines, &cone, &pts);
    HelperGeometry {
        lines,
        ..Default::default()
    }
}

/// three.js's `CameraHelper` default colours — frustum, cone, up, target,
/// cross — as linear RGB (`new Color(0xffaa00)` and so on).
pub fn camera_default_colors() -> [[f32; 3]; 5] {
    let lin = |h: u32| {
        let c = crate::math::Color::from_hex(h);
        [c.r, c.g, c.b]
    };
    [lin(0xffaa00), lin(0xff0000), lin(0x00aaff), lin(0xffffff), lin(0x333333)]
}

/// `CameraHelper(camera)`: frustum, cone, up triangle, target and cross
/// lines, unprojected through `projectionMatrixInverse` and placed by the
/// camera's `matrixWorld`; `colors` as `setColors(frustum, cone, up,
/// target, cross)`.
pub fn camera(projection_inverse: &Mat4, camera_world: &Mat4, colors: [[f32; 3]; 5]) -> HelperGeometry {
    let (w, h) = (1.0, 1.0);
    let point = |id: &str| -> V3 {
        let ndc = match id {
            "c" => [0.0, 0.0, -1.0],
            "t" => [0.0, 0.0, 1.0],
            "n1" => [-w, -h, -1.0],
            "n2" => [w, -h, -1.0],
            "n3" => [-w, h, -1.0],
            "n4" => [w, h, -1.0],
            "f1" => [-w, -h, 1.0],
            "f2" => [w, -h, 1.0],
            "f3" => [-w, h, 1.0],
            "f4" => [w, h, 1.0],
            "u1" => [w * 0.7, h * 1.1, -1.0],
            "u2" => [-w * 0.7, h * 1.1, -1.0],
            "u3" => [0.0, h * 2.0, -1.0],
            "cf1" => [-w, 0.0, 1.0],
            "cf2" => [w, 0.0, 1.0],
            "cf3" => [0.0, -h, 1.0],
            "cf4" => [0.0, h, 1.0],
            "cn1" => [-w, 0.0, -1.0],
            "cn2" => [w, 0.0, -1.0],
            "cn3" => [0.0, -h, -1.0],
            "cn4" => [0.0, h, -1.0],
            // `p` is never set: it stays at the origin.
            _ => return [0.0; 3],
        };
        // `unproject` with an identity-world camera, then the Float32 store.
        f32v(apply(&IDENTITY, apply(projection_inverse, ndc)))
    };
    let [frustum, cone, up, target, cross_c] = colors;
    let pairs: [(&str, &str, [f32; 3]); 25] = [
        ("n1", "n2", frustum),
        ("n2", "n4", frustum),
        ("n4", "n3", frustum),
        ("n3", "n1", frustum),
        ("f1", "f2", frustum),
        ("f2", "f4", frustum),
        ("f4", "f3", frustum),
        ("f3", "f1", frustum),
        ("n1", "f1", frustum),
        ("n2", "f2", frustum),
        ("n3", "f3", frustum),
        ("n4", "f4", frustum),
        ("p", "n1", cone),
        ("p", "n2", cone),
        ("p", "n3", cone),
        ("p", "n4", cone),
        ("u1", "u2", up),
        ("u2", "u3", up),
        ("u3", "u1", up),
        ("c", "t", target),
        ("p", "c", cross_c),
        ("cn1", "cn2", cross_c),
        ("cn3", "cn4", cross_c),
        ("cf1", "cf2", cross_c),
        ("cf3", "cf4", cross_c),
    ];
    let mut lines = Vec::with_capacity(pairs.len() * 6);
    let mut cols = Vec::with_capacity(pairs.len() * 6);
    for (a, b, c) in pairs {
        push(&mut lines, apply(camera_world, point(a)));
        push(&mut lines, apply(camera_world, point(b)));
        cols.extend_from_slice(&c);
        cols.extend_from_slice(&c);
    }
    HelperGeometry {
        lines,
        colors: cols,
        mesh: Vec::new(),
    }
}

/// `PerspectiveCamera.updateProjectionMatrix` (WebGL clip space). `view` is
/// `[fullWidth, fullHeight, offsetX, offsetY, width, height]` when enabled.
#[allow(clippy::too_many_arguments)]
pub fn perspective_projection(
    fov: f64,
    aspect: f64,
    near: f64,
    far: f64,
    zoom: f64,
    film_offset: f64,
    film_width: f64,
    view: Option<[f64; 6]>,
) -> Mat4 {
    let mut top = near * (std::f64::consts::PI / 180.0 * 0.5 * fov).tan() / zoom;
    let mut height = 2.0 * top;
    let mut width = aspect * height;
    let mut left = -0.5 * width;
    if let Some([full_w, full_h, ox, oy, vw, vh]) = view {
        left += ox * width / full_w;
        top -= oy * height / full_h;
        width *= vw / full_w;
        height *= vh / full_h;
    }
    if film_offset != 0.0 {
        left += near * film_offset / film_width;
    }
    make_perspective(left, left + width, top, top - height, near, far)
}

fn make_perspective(left: f64, right: f64, top: f64, bottom: f64, near: f64, far: f64) -> Mat4 {
    let x = 2.0 * near / (right - left);
    let y = 2.0 * near / (top - bottom);
    let a = (right + left) / (right - left);
    let b = (top + bottom) / (top - bottom);
    let c = -(far + near) / (far - near);
    let d = (-2.0 * far * near) / (far - near);
    [
        x, 0.0, 0.0, 0.0, 0.0, y, 0.0, 0.0, a, b, c, -1.0, 0.0, 0.0, d, 0.0,
    ]
}

/// `OrthographicCamera.updateProjectionMatrix` (WebGL clip space).
#[allow(clippy::too_many_arguments)]
pub fn orthographic_projection(
    left: f64,
    right: f64,
    top: f64,
    bottom: f64,
    near: f64,
    far: f64,
    zoom: f64,
    view: Option<[f64; 6]>,
) -> Mat4 {
    let dx = (right - left) / (2.0 * zoom);
    let dy = (top - bottom) / (2.0 * zoom);
    let cx = (right + left) / 2.0;
    let cy = (top + bottom) / 2.0;
    let (mut l, mut r, mut t, mut b) = (cx - dx, cx + dx, cy + dy, cy - dy);
    if let Some([full_w, full_h, ox, oy, vw, vh]) = view {
        let scale_w = (right - left) / full_w / zoom;
        let scale_h = (top - bottom) / full_h / zoom;
        l += scale_w * ox;
        r = l + scale_w * vw;
        t -= scale_h * oy;
        b = t - scale_h * vh;
    }
    let w = 1.0 / (r - l);
    let h = 1.0 / (t - b);
    let p = 1.0 / (far - near);
    let x = (r + l) * w;
    let y = (t + b) * h;
    let z = (far + near) * p;
    let z_inv = -2.0 * p;
    [
        2.0 * w, 0.0, 0.0, 0.0, 0.0, 2.0 * h, 0.0, 0.0, 0.0, 0.0, z_inv, 0.0, -x, -y, -z, 1.0,
    ]
}

/// `ArrowHelper.setDirection(dir)`: the helper's quaternion.
pub fn arrow_quaternion(dir: V3) -> Quat {
    if dir[1] > 0.99999 {
        [0.0, 0.0, 0.0, 1.0]
    } else if dir[1] < -0.99999 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        let axis = normalize([dir[2], 0.0, -dir[0]]);
        let radians = dir[1].acos();
        let half = radians / 2.0;
        let s = half.sin();
        [axis[0] * s, axis[1] * s, axis[2] * s, half.cos()]
    }
}

/// `CylinderGeometry(0, 0.5, 1, 5, 1)` translated by `(0, -0.5, 0)` — the
/// arrow head — as positions and its index.
fn arrow_cone() -> (Vec<V3>, Vec<[usize; 3]>) {
    let (radius_top, radius_bottom, height, radial, hs) = (0.0f64, 0.5f64, 1.0f64, 5usize, 1usize);
    let half = height / 2.0;
    let mut positions: Vec<V3> = Vec::new();
    let mut tris = Vec::new();
    let mut rows = Vec::new();
    for y in 0..=hs {
        let v = y as f64 / hs as f64;
        let radius = v * (radius_bottom - radius_top) + radius_top;
        let mut row = Vec::new();
        for x in 0..=radial {
            let u = x as f64 / radial as f64;
            let theta = u * std::f64::consts::PI * 2.0;
            positions.push(f32v([radius * theta.sin(), -v * height + half, radius * theta.cos()]));
            row.push(positions.len() - 1);
        }
        rows.push(row);
    }
    for x in 0..radial {
        for y in 0..hs {
            let a = rows[y][x];
            let b = rows[y + 1][x];
            let c = rows[y + 1][x + 1];
            let d = rows[y][x + 1];
            tris.push([a, b, d]);
            tris.push([b, c, d]);
        }
    }
    // Bottom cap only: three.js skips the top one when `radiusTop` is 0.
    let center_start = positions.len();
    for _ in 1..=radial {
        positions.push([0.0, -half, 0.0]);
    }
    let center_end = positions.len();
    for x in 0..=radial {
        let theta = x as f64 / radial as f64 * std::f64::consts::PI * 2.0;
        positions.push(f32v([radius_bottom * theta.sin(), -half, radius_bottom * theta.cos()]));
    }
    for x in 0..radial {
        let c = center_start + x;
        let i = center_end + x;
        tris.push([i + 1, i, c]);
    }
    // `translate(0, -0.5, 0)` on the Float32 attribute.
    let positions = positions
        .into_iter()
        .map(|[x, y, z]| f32v([x, y - 0.5, z]))
        .collect();
    (positions, tris)
}

/// `ArrowHelper(dir, origin, length, color, headLength, headWidth)`: the
/// shaft as a line and the head as a filled cone. `helper_world` is the
/// helper's own `matrixWorld` — `compose(origin, arrow_quaternion(dir), 1)`
/// for one at the scene root that nobody has moved since.
pub fn arrow(helper_world: &Mat4, length: f64, head_length: f64, head_width: f64) -> HelperGeometry {
    let line = multiply(
        helper_world,
        &compose([0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0, (length - head_length).max(0.0001), 1.0]),
    );
    let mut lines = Vec::new();
    strip(&mut lines, &line, &[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let cone = multiply(
        helper_world,
        &compose([0.0, length, 0.0], [0.0, 0.0, 0.0, 1.0], [head_width, head_length, head_width]),
    );
    let (positions, tris) = arrow_cone();
    let mut mesh = Vec::with_capacity(tris.len() * 9);
    for t in tris {
        for i in t {
            push(&mut mesh, apply(&cone, positions[i]));
        }
    }
    HelperGeometry {
        lines,
        colors: Vec::new(),
        mesh,
    }
}

/// `PlaneHelper(plane, size)`: the square outline with its diagonal (a
/// `Line` strip) and the translucent fill, oriented by `plane.normal` and
/// pushed out by `-plane.constant`.
pub fn plane(normal: V3, constant: f64, size: f64) -> HelperGeometry {
    let scale = [0.5 * size, 0.5 * size, 1.0];
    // `lookAt(plane.normal)` from the origin, then `translateZ(-constant)`.
    let q = quat_from_rotation(&look_at_matrix(normal, [0.0; 3], UP));
    let z = apply_quat([0.0, 0.0, 1.0], q);
    let position = [z[0] * -constant, z[1] * -constant, z[2] * -constant];
    let world = compose(position, q, scale);
    let mut lines = Vec::new();
    strip(
        &mut lines,
        &world,
        &[
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
    );
    let mut mesh = Vec::new();
    segments(
        &mut mesh,
        &world,
        &[
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
        ],
    );
    HelperGeometry {
        lines,
        colors: Vec::new(),
        mesh,
    }
}

/// `SkeletonHelper(object)`: a blue-to-green line from each bone whose
/// parent is a bone to that parent. `bone_worlds[i]` is bone `i`'s
/// `matrixWorld` in `getBoneList` order; `parents[i]` the index of its parent
/// bone, or negative when its parent is not a bone.
pub fn skeleton(root_world: &Mat4, bone_worlds: &[Mat4], parents: &[i32]) -> HelperGeometry {
    let inv = invert(root_world);
    let mut lines = Vec::new();
    let mut colors = Vec::new();
    for (i, world) in bone_worlds.iter().enumerate() {
        let Some(&p) = parents.get(i) else { continue };
        if p < 0 || p as usize >= bone_worlds.len() {
            continue;
        }
        for m in [world, &bone_worlds[p as usize]] {
            let local = f32v(position_of(&multiply(&inv, m)));
            push(&mut lines, apply(root_world, local));
        }
        colors.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    }
    HelperGeometry {
        lines,
        colors,
        mesh: Vec::new(),
    }
}

/// The edge list `BoxHelper` and `Box3Helper` index their eight corners with.
const BOX_EDGES: [usize; 24] = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7];

/// `BoxHelper(object)`: the twelve edges of the world-space bounding box
/// `min`..`max` (nothing for an empty box).
pub fn box_lines(min: V3, max: V3) -> HelperGeometry {
    if max[0] < min[0] || max[1] < min[1] || max[2] < min[2] {
        return HelperGeometry::default();
    }
    let corners: [V3; 8] = [
        [max[0], max[1], max[2]],
        [min[0], max[1], max[2]],
        [min[0], min[1], max[2]],
        [max[0], min[1], max[2]],
        [max[0], max[1], min[2]],
        [min[0], max[1], min[2]],
        [min[0], min[1], min[2]],
        [max[0], min[1], min[2]],
    ];
    let mut lines = Vec::with_capacity(72);
    for i in BOX_EDGES {
        push(&mut lines, f32v(corners[i]));
    }
    HelperGeometry {
        lines,
        ..Default::default()
    }
}

/// `new Box3().setFromObject(object)` (not `precise`): the union over the
/// object's meshes of each geometry's bounding box, transformed by the mesh's
/// `matrixWorld` — `None` when nothing has a vertex. `positions[i]` are mesh
/// `i`'s `xyz` vertices, `worlds[i]` its world matrix.
pub fn object_bounds(worlds: &[Mat4], positions: &[&[f32]]) -> Option<(V3, V3)> {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for (world, pos) in worlds.iter().zip(positions) {
        if pos.len() < 3 {
            continue;
        }
        // `computeBoundingBox`.
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in pos.chunks_exact(3) {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k] as f64);
                hi[k] = hi[k].max(p[k] as f64);
            }
        }
        // `Box3.applyMatrix4`: the eight corners, transformed, re-boxed.
        for i in 0..8 {
            let c = [
                if i & 1 == 0 { lo[0] } else { hi[0] },
                if i & 2 == 0 { lo[1] } else { hi[1] },
                if i & 4 == 0 { lo[2] } else { hi[2] },
            ];
            let w = apply(world, c);
            for k in 0..3 {
                min[k] = min[k].min(w[k]);
                max[k] = max[k].max(w[k]);
            }
        }
    }
    (min[0] <= max[0]).then_some((min, max))
}

/// `Box3Helper(box)`: the unit cube scaled to the box's half size and moved
/// to its centre (nothing for an empty box).
pub fn box3(min: V3, max: V3) -> HelperGeometry {
    if max[0] < min[0] || max[1] < min[1] || max[2] < min[2] {
        return HelperGeometry::default();
    }
    let center = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5];
    let half = [(max[0] - min[0]) * 0.5, (max[1] - min[1]) * 0.5, (max[2] - min[2]) * 0.5];
    let world = compose(center, [0.0, 0.0, 0.0, 1.0], half);
    let unit: [V3; 8] = [
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
    ];
    let mut lines = Vec::with_capacity(72);
    for i in BOX_EDGES {
        push(&mut lines, apply(&world, unit[i]));
    }
    HelperGeometry {
        lines,
        ..Default::default()
    }
}

/// `VertexNormalsHelper(object, size)`: from every vertex, in world space,
/// along its normal (through the normal matrix, normalised) for `size`.
pub fn vertex_normals(object_world: &Mat4, positions: &[f32], normals: &[f32], size: f64) -> HelperGeometry {
    // `Matrix3.getNormalMatrix`: the inverse transpose of the upper 3×3.
    let m = object_world;
    let (n11, n21, n31) = (m[0], m[1], m[2]);
    let (n12, n22, n32) = (m[4], m[5], m[6]);
    let (n13, n23, n33) = (m[8], m[9], m[10]);
    let t11 = n33 * n22 - n32 * n23;
    let t12 = n32 * n13 - n33 * n12;
    let t13 = n23 * n12 - n22 * n13;
    let det = n11 * t11 + n21 * t12 + n31 * t13;
    let inv: [f64; 9] = if det == 0.0 {
        [0.0; 9]
    } else {
        let d = 1.0 / det;
        [
            t11 * d,
            (n31 * n23 - n33 * n21) * d,
            (n32 * n21 - n31 * n22) * d,
            t12 * d,
            (n33 * n11 - n31 * n13) * d,
            (n31 * n12 - n32 * n11) * d,
            t13 * d,
            (n21 * n13 - n23 * n11) * d,
            (n22 * n11 - n21 * n12) * d,
        ]
    };
    // Transposed: e[r + 3c] of the result is inv[c + 3r].
    let e = [inv[0], inv[3], inv[6], inv[1], inv[4], inv[7], inv[2], inv[5], inv[8]];
    let count = (positions.len() / 3).min(normals.len() / 3);
    let mut lines = Vec::with_capacity(count * 6);
    for j in 0..count {
        let p = apply(
            object_world,
            [positions[j * 3] as f64, positions[j * 3 + 1] as f64, positions[j * 3 + 2] as f64],
        );
        let (x, y, z) = (normals[j * 3] as f64, normals[j * 3 + 1] as f64, normals[j * 3 + 2] as f64);
        let n = normalize([
            e[0] * x + e[3] * y + e[6] * z,
            e[1] * x + e[4] * y + e[7] * z,
            e[2] * x + e[5] * y + e[8] * z,
        ]);
        push(&mut lines, p);
        push(&mut lines, [n[0] * size + p[0], n[1] * size + p[1], n[2] * size + p[2]]);
    }
    HelperGeometry {
        lines,
        ..Default::default()
    }
}

/// `RectAreaLightHelper(light)`: the light's rectangle outline and its
/// back-facing fill, rotated and placed as the light but ignoring its scale.
pub fn rect_area_light(light_world: &Mat4, width: f64, height: f64) -> HelperGeometry {
    let mut world = extract_rotation(light_world);
    let scale = [0.5 * width, 0.5 * height, 1.0];
    for c in 0..3 {
        for r in 0..4 {
            world[c * 4 + r] *= scale[c];
        }
    }
    world[12] = light_world[12];
    world[13] = light_world[13];
    world[14] = light_world[14];
    let mut lines = Vec::new();
    strip(
        &mut lines,
        &world,
        &[
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
    );
    let mut mesh = Vec::new();
    segments(
        &mut mesh,
        &world,
        &[
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
        ],
    );
    HelperGeometry {
        lines,
        colors: Vec::new(),
        mesh,
    }
}

/// `RectAreaLightHelper`'s colour without a fixed one: the light's colour
/// times its intensity, scaled back into range so it keeps its hue.
pub fn rect_area_light_color(color: [f32; 3], intensity: f32) -> [f32; 3] {
    let mut c = [color[0] * intensity, color[1] * intensity, color[2] * intensity];
    let max = c[0].max(c[1]).max(c[2]);
    if max > 1.0 {
        let s = 1.0 / max;
        c = [c[0] * s, c[1] * s, c[2] * s];
    }
    c
}

#[cfg(test)]
mod tests;
