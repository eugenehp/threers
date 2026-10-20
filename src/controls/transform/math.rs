//! The slice of three.js r165's math the transform gizmo needs, in `f64` and
//! operation for operation (`Vector3`, `Quaternion`, `Matrix4`, `Matrix3`),
//! so that a drag computed here lands on the same numbers JavaScript does.
//!
//! Vectors are `[x, y, z]`, quaternions `[x, y, z, w]`, matrices three.js's
//! column-major `elements`.

/// The transcendental functions, as JavaScript's `Math` computes them. In
/// the browser that is the engine's own `Math` — its `sin` and Rust's `libm`
/// can differ in the last bit, and a drag that rotates by a different last
/// bit is not the drag three.js computed. Elsewhere, Rust's.
pub mod js {
    #[cfg(target_arch = "wasm32")]
    mod imp {
        pub fn sin(x: f64) -> f64 {
            js_sys::Math::sin(x)
        }
        pub fn cos(x: f64) -> f64 {
            js_sys::Math::cos(x)
        }
        pub fn tan(x: f64) -> f64 {
            js_sys::Math::tan(x)
        }
        pub fn acos(x: f64) -> f64 {
            js_sys::Math::acos(x)
        }
        pub fn atan2(y: f64, x: f64) -> f64 {
            js_sys::Math::atan2(y, x)
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    mod imp {
        pub fn sin(x: f64) -> f64 {
            x.sin()
        }
        pub fn cos(x: f64) -> f64 {
            x.cos()
        }
        pub fn tan(x: f64) -> f64 {
            x.tan()
        }
        pub fn acos(x: f64) -> f64 {
            x.acos()
        }
        pub fn atan2(y: f64, x: f64) -> f64 {
            y.atan2(x)
        }
    }
    pub use imp::*;
}

pub type V3 = [f64; 3];
pub type Q = [f64; 4];
pub type M4 = [f64; 16];
pub type M3 = [f64; 9];

pub const IDENTITY_Q: Q = [0.0, 0.0, 0.0, 1.0];
pub const IDENTITY_M4: M4 = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `multiplyScalar`.
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// `multiply` (component-wise).
pub fn mul(a: V3, b: V3) -> V3 {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

/// `divide` (component-wise).
pub fn div(a: V3, b: V3) -> V3 {
    [a[0] / b[0], a[1] / b[1], a[2] / b[2]]
}

pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn length_sq(a: V3) -> f64 {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}

pub fn length(a: V3) -> f64 {
    length_sq(a).sqrt()
}

/// `distanceTo`.
pub fn distance(a: V3, b: V3) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// `crossVectors(a, b)`.
pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `normalize`: `divideScalar(length || 1)`, a multiply by the reciprocal.
pub fn normalize(a: V3) -> V3 {
    let l = length(a);
    scale(a, 1.0 / if l == 0.0 { 1.0 } else { l })
}

/// `angleTo`.
pub fn angle_to(a: V3, b: V3) -> f64 {
    let denominator = (length_sq(a) * length_sq(b)).sqrt();
    if denominator == 0.0 {
        return std::f64::consts::PI / 2.0;
    }
    let theta = dot(a, b) / denominator;
    js::acos(theta.clamp(-1.0, 1.0))
}

/// `applyQuaternion` (r165's form).
pub fn apply_quat(v: V3, q: Q) -> V3 {
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

/// `applyMatrix4` (with the perspective divide).
pub fn apply_m4(v: V3, e: &M4) -> V3 {
    let [x, y, z] = v;
    let w = 1.0 / (e[3] * x + e[7] * y + e[11] * z + e[15]);
    [
        (e[0] * x + e[4] * y + e[8] * z + e[12]) * w,
        (e[1] * x + e[5] * y + e[9] * z + e[13]) * w,
        (e[2] * x + e[6] * y + e[10] * z + e[14]) * w,
    ]
}

/// `transformDirection`.
pub fn transform_direction(v: V3, e: &M4) -> V3 {
    let [x, y, z] = v;
    normalize([
        e[0] * x + e[4] * y + e[8] * z,
        e[1] * x + e[5] * y + e[9] * z,
        e[2] * x + e[6] * y + e[10] * z,
    ])
}

/// `applyMatrix3`.
pub fn apply_m3(v: V3, e: &M3) -> V3 {
    let [x, y, z] = v;
    [
        e[0] * x + e[3] * y + e[6] * z,
        e[1] * x + e[4] * y + e[7] * z,
        e[2] * x + e[5] * y + e[8] * z,
    ]
}

/// `setFromMatrixPosition`.
pub fn matrix_position(e: &M4) -> V3 {
    [e[12], e[13], e[14]]
}

/// `Quaternion.setFromEuler` for the default `XYZ` order.
pub fn quat_from_euler_xyz(x: f64, y: f64, z: f64) -> Q {
    let c1 = js::cos(x / 2.0);
    let c2 = js::cos(y / 2.0);
    let c3 = js::cos(z / 2.0);
    let s1 = js::sin(x / 2.0);
    let s2 = js::sin(y / 2.0);
    let s3 = js::sin(z / 2.0);
    [
        s1 * c2 * c3 + c1 * s2 * s3,
        c1 * s2 * c3 - s1 * c2 * s3,
        c1 * c2 * s3 + s1 * s2 * c3,
        c1 * c2 * c3 - s1 * s2 * s3,
    ]
}

/// `setFromAxisAngle`.
pub fn quat_from_axis_angle(axis: V3, angle: f64) -> Q {
    let half = angle / 2.0;
    let s = js::sin(half);
    [axis[0] * s, axis[1] * s, axis[2] * s, js::cos(half)]
}

/// `setFromRotationMatrix` (reads the upper 3×3).
pub fn quat_from_rotation_matrix(te: &M4) -> Q {
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

/// `multiplyQuaternions(a, b)`.
pub fn quat_mul(a: Q, b: Q) -> Q {
    let [qax, qay, qaz, qaw] = a;
    let [qbx, qby, qbz, qbw] = b;
    [
        qax * qbw + qaw * qbx + qay * qbz - qaz * qby,
        qay * qbw + qaw * qby + qaz * qbx - qax * qbz,
        qaz * qbw + qaw * qbz + qax * qby - qay * qbx,
        qaw * qbw - qax * qbx - qay * qby - qaz * qbz,
    ]
}

/// `invert` (the conjugate, as three.js has it).
pub fn quat_invert(q: Q) -> Q {
    [-q[0], -q[1], -q[2], q[3]]
}

/// `normalize`.
pub fn quat_normalize(q: Q) -> Q {
    let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if l == 0.0 {
        IDENTITY_Q
    } else {
        let l = 1.0 / l;
        [q[0] * l, q[1] * l, q[2] * l, q[3] * l]
    }
}

pub fn quat_dot(a: Q, b: Q) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// `Matrix4.compose`.
pub fn compose(p: V3, q: Q, s: V3) -> M4 {
    let [x, y, z, w] = q;
    let (x2, y2, z2) = (x + x, y + y, z + z);
    let (xx, xy, xz) = (x * x2, x * y2, x * z2);
    let (yy, yz, zz) = (y * y2, y * z2, z * z2);
    let (wx, wy, wz) = (w * x2, w * y2, w * z2);
    let [sx, sy, sz] = s;
    [
        (1.0 - (yy + zz)) * sx,
        (xy + wz) * sx,
        (xz - wy) * sx,
        0.0,
        (xy - wz) * sy,
        (1.0 - (xx + zz)) * sy,
        (yz + wx) * sy,
        0.0,
        (xz + wy) * sz,
        (yz - wx) * sz,
        (1.0 - (xx + yy)) * sz,
        0.0,
        p[0],
        p[1],
        p[2],
        1.0,
    ]
}

/// `Matrix4.determinant`.
pub fn determinant(te: &M4) -> f64 {
    let (n11, n12, n13, n14) = (te[0], te[4], te[8], te[12]);
    let (n21, n22, n23, n24) = (te[1], te[5], te[9], te[13]);
    let (n31, n32, n33, n34) = (te[2], te[6], te[10], te[14]);
    let (n41, n42, n43, n44) = (te[3], te[7], te[11], te[15]);
    n41 * (n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34
        - n12 * n23 * n34)
        + n42
            * (n11 * n23 * n34 - n11 * n24 * n33 + n14 * n21 * n33 - n13 * n21 * n34
                + n13 * n24 * n31
                - n14 * n23 * n31)
        + n43
            * (n11 * n24 * n32 - n11 * n22 * n34 - n14 * n21 * n32
                + n12 * n21 * n34
                + n14 * n22 * n31
                - n12 * n24 * n31)
        + n44
            * (-n13 * n22 * n31 - n11 * n23 * n32 + n11 * n22 * n33 + n13 * n21 * n32
                - n12 * n21 * n33
                + n12 * n23 * n31)
}

/// `Matrix4.decompose` → `(position, quaternion, scale)`.
pub fn decompose(te: &M4) -> (V3, Q, V3) {
    let mut sx = length([te[0], te[1], te[2]]);
    let sy = length([te[4], te[5], te[6]]);
    let sz = length([te[8], te[9], te[10]]);
    if determinant(te) < 0.0 {
        sx = -sx;
    }
    let position = [te[12], te[13], te[14]];
    let mut m = *te;
    let (ix, iy, iz) = (1.0 / sx, 1.0 / sy, 1.0 / sz);
    m[0] *= ix;
    m[1] *= ix;
    m[2] *= ix;
    m[4] *= iy;
    m[5] *= iy;
    m[6] *= iy;
    m[8] *= iz;
    m[9] *= iz;
    m[10] *= iz;
    (position, quat_from_rotation_matrix(&m), [sx, sy, sz])
}

/// `Matrix4.multiplyMatrices(a, b)`.
pub fn multiply(ae: &M4, be: &M4) -> M4 {
    let (a11, a12, a13, a14) = (ae[0], ae[4], ae[8], ae[12]);
    let (a21, a22, a23, a24) = (ae[1], ae[5], ae[9], ae[13]);
    let (a31, a32, a33, a34) = (ae[2], ae[6], ae[10], ae[14]);
    let (a41, a42, a43, a44) = (ae[3], ae[7], ae[11], ae[15]);
    let (b11, b12, b13, b14) = (be[0], be[4], be[8], be[12]);
    let (b21, b22, b23, b24) = (be[1], be[5], be[9], be[13]);
    let (b31, b32, b33, b34) = (be[2], be[6], be[10], be[14]);
    let (b41, b42, b43, b44) = (be[3], be[7], be[11], be[15]);
    let mut te = [0.0; 16];
    te[0] = a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41;
    te[4] = a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42;
    te[8] = a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43;
    te[12] = a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44;
    te[1] = a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41;
    te[5] = a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42;
    te[9] = a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43;
    te[13] = a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44;
    te[2] = a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41;
    te[6] = a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42;
    te[10] = a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43;
    te[14] = a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44;
    te[3] = a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41;
    te[7] = a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42;
    te[11] = a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43;
    te[15] = a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44;
    te
}

/// `Matrix4.invert`.
pub fn invert(te: &M4) -> M4 {
    let (n11, n21, n31, n41) = (te[0], te[1], te[2], te[3]);
    let (n12, n22, n32, n42) = (te[4], te[5], te[6], te[7]);
    let (n13, n23, n33, n43) = (te[8], te[9], te[10], te[11]);
    let (n14, n24, n34, n44) = (te[12], te[13], te[14], te[15]);
    let t11 =
        n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44
            + n22 * n33 * n44;
    let t12 =
        n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44
            - n12 * n33 * n44;
    let t13 =
        n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44
            + n12 * n23 * n44;
    let t14 =
        n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34
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

/// `Matrix4.lookAt(eye, target, up)` written over `te` (only the rotation
/// part changes, as in three.js).
pub fn look_at(te: &mut M4, eye: V3, target: V3, up: V3) {
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
    te[0] = x[0];
    te[4] = y[0];
    te[8] = z[0];
    te[1] = x[1];
    te[5] = y[1];
    te[9] = z[1];
    te[2] = x[2];
    te[6] = y[2];
    te[10] = z[2];
}

/// `Matrix4.set` takes rows; this builds the column-major elements.
fn from_rows(r: [f64; 16]) -> M4 {
    [
        r[0], r[4], r[8], r[12], r[1], r[5], r[9], r[13], r[2], r[6], r[10], r[14], r[3], r[7],
        r[11], r[15],
    ]
}

pub fn make_translation(x: f64, y: f64, z: f64) -> M4 {
    from_rows([
        1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z, 0.0, 0.0, 0.0, 1.0,
    ])
}

pub fn make_rotation_x(theta: f64) -> M4 {
    let (c, s) = (js::cos(theta), js::sin(theta));
    from_rows([
        1.0, 0.0, 0.0, 0.0, 0.0, c, -s, 0.0, 0.0, s, c, 0.0, 0.0, 0.0, 0.0, 1.0,
    ])
}

pub fn make_rotation_y(theta: f64) -> M4 {
    let (c, s) = (js::cos(theta), js::sin(theta));
    from_rows([
        c, 0.0, s, 0.0, 0.0, 1.0, 0.0, 0.0, -s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0,
    ])
}

/// `Matrix3.getNormalMatrix(m)`: `setFromMatrix4(m).invert().transpose()`.
pub fn normal_matrix(m: &M4) -> M3 {
    let te = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
    let (n11, n21, n31) = (te[0], te[1], te[2]);
    let (n12, n22, n32) = (te[3], te[4], te[5]);
    let (n13, n23, n33) = (te[6], te[7], te[8]);
    let t11 = n33 * n22 - n32 * n23;
    let t12 = n32 * n13 - n33 * n12;
    let t13 = n23 * n12 - n22 * n13;
    let det = n11 * t11 + n21 * t12 + n31 * t13;
    let inv = if det == 0.0 {
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
    // transpose
    [
        inv[0], inv[3], inv[6], inv[1], inv[4], inv[7], inv[2], inv[5], inv[8],
    ]
}
