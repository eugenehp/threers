use super::{Quaternion, Vector3};

/// A 4x4 column-major matrix (same layout as three.js's `Matrix4`).
///
/// `elements[0..4]` is column 0, `elements[4..8]` is column 1, etc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix4 {
    pub elements: [f32; 16],
}

impl Default for Matrix4 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Matrix4 {
    pub const fn identity() -> Self {
        Self {
            elements: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    pub fn translation(v: Vector3) -> Self {
        let mut m = Self::identity();
        m.elements[12] = v.x;
        m.elements[13] = v.y;
        m.elements[14] = v.z;
        m
    }

    pub fn scale(v: Vector3) -> Self {
        let mut m = Self::identity();
        m.elements[0] = v.x;
        m.elements[5] = v.y;
        m.elements[10] = v.z;
        m
    }

    pub fn from_quaternion(q: Quaternion) -> Self {
        let x2 = q.x + q.x;
        let y2 = q.y + q.y;
        let z2 = q.z + q.z;
        let xx = q.x * x2;
        let xy = q.x * y2;
        let xz = q.x * z2;
        let yy = q.y * y2;
        let yz = q.y * z2;
        let zz = q.z * z2;
        let wx = q.w * x2;
        let wy = q.w * y2;
        let wz = q.w * z2;

        Self {
            elements: [
                1.0 - (yy + zz),
                xy + wz,
                xz - wy,
                0.0,
                xy - wz,
                1.0 - (xx + zz),
                yz + wx,
                0.0,
                xz + wy,
                yz - wx,
                1.0 - (xx + yy),
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        }
    }

    /// Compose a transformation from position, rotation, and scale (same order as three.js).
    pub fn compose(position: Vector3, quaternion: Quaternion, scale: Vector3) -> Self {
        let mut m = Self::from_quaternion(quaternion);
        m.elements[0] *= scale.x;
        m.elements[1] *= scale.x;
        m.elements[2] *= scale.x;
        m.elements[4] *= scale.y;
        m.elements[5] *= scale.y;
        m.elements[6] *= scale.y;
        m.elements[8] *= scale.z;
        m.elements[9] *= scale.z;
        m.elements[10] *= scale.z;
        m.elements[12] = position.x;
        m.elements[13] = position.y;
        m.elements[14] = position.z;
        m
    }

    pub fn multiply(&self, other: &Self) -> Self {
        let a = &self.elements;
        let b = &other.elements;
        let mut r = [0.0f32; 16];
        for col in 0..4 {
            for row in 0..4 {
                r[col * 4 + row] = a[row] * b[col * 4]
                    + a[row + 4] * b[col * 4 + 1]
                    + a[row + 8] * b[col * 4 + 2]
                    + a[row + 12] * b[col * 4 + 3];
            }
        }
        Self { elements: r }
    }

    /// Right-handed perspective projection into the **WebGPU** clip space —
    /// depth 0 at the near plane, 1 at the far plane — matching three.js's
    /// `makePerspective` under `WebGPUCoordinateSystem`.
    ///
    /// This used to emit the OpenGL form, `(far + near) * nf` and
    /// `2 * far * near * nf`, which puts the near plane at z = -1. Nothing else
    /// in the crate agreed with it: [`orthographic`](Self::orthographic)
    /// directly below already produces 0..1, wgpu clips at 0..1, and
    /// `shadow_factor` in the shader rejects `ndc.z < 0.0` outright.
    ///
    /// Two things followed, both silent. The real near plane was not `near` but
    /// the harmonic mean `2 * near * far / (near + far)`, and everything closer
    /// than that was clipped — invisible at the usual `near = 0.1`, where the
    /// true plane lands at 0.2, and very visible the moment you fit the depth
    /// range to a scene to buy back precision: near 324 / far 952 clips at 483
    /// and takes the front off the subject. And half the depth buffer went
    /// unused, which is precisely the precision such a fit is trying to win.
    /// Spot lights lost the near half of their shadows for the same reason.
    pub fn perspective(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y_rad / 2.0).tan();
        let nf = 1.0 / (near - far);
        Self {
            elements: [
                f / aspect,
                0.0,
                0.0,
                0.0,
                0.0,
                f,
                0.0,
                0.0,
                0.0,
                0.0,
                far * nf,
                -1.0,
                0.0,
                0.0,
                far * near * nf,
                0.0,
            ],
        }
    }

    /// Perspective with lens shift (Blender/three.js film offset style).
    /// `shift_x` / `shift_y` are fractions of the sensor (±0.5 ≈ half-frame).
    pub fn perspective_with_shift(
        fov_y_rad: f32,
        aspect: f32,
        near: f32,
        far: f32,
        shift_x: f32,
        shift_y: f32,
    ) -> Self {
        let mut m = Self::perspective(fov_y_rad, aspect, near, far);
        // Skew the projection by offsetting the frustum centre.
        m.elements[8] += shift_x * 2.0;
        m.elements[9] += shift_y * 2.0;
        m
    }

    pub fn orthographic(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) -> Self {
        let w = 1.0 / (right - left);
        let h = 1.0 / (top - bottom);
        let p = 1.0 / (far - near);
        Self {
            elements: [
                2.0 * w,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0 * h,
                0.0,
                0.0,
                0.0,
                0.0,
                -p,
                0.0,
                -(right + left) * w,
                -(top + bottom) * h,
                -near * p,
                1.0,
            ],
        }
    }

    /// Inverse via cofactor expansion. Returns identity if singular.
    pub fn invert(&self) -> Self {
        let m = &self.elements;
        let mut inv = [0.0f32; 16];
        inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
            + m[9] * m[7] * m[14]
            + m[13] * m[6] * m[11]
            - m[13] * m[7] * m[10];
        inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
            - m[8] * m[7] * m[14]
            - m[12] * m[6] * m[11]
            + m[12] * m[7] * m[10];
        inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
            + m[8] * m[7] * m[13]
            + m[12] * m[5] * m[11]
            - m[12] * m[7] * m[9];
        inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
            - m[8] * m[6] * m[13]
            - m[12] * m[5] * m[10]
            + m[12] * m[6] * m[9];
        inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
            - m[9] * m[3] * m[14]
            - m[13] * m[2] * m[11]
            + m[13] * m[3] * m[10];
        inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
            + m[8] * m[3] * m[14]
            + m[12] * m[2] * m[11]
            - m[12] * m[3] * m[10];
        inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
            - m[8] * m[3] * m[13]
            - m[12] * m[1] * m[11]
            + m[12] * m[3] * m[9];
        inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
            + m[8] * m[2] * m[13]
            + m[12] * m[1] * m[10]
            - m[12] * m[2] * m[9];
        inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
            + m[5] * m[3] * m[14]
            + m[13] * m[2] * m[7]
            - m[13] * m[3] * m[6];
        inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
            - m[4] * m[3] * m[14]
            - m[12] * m[2] * m[7]
            + m[12] * m[3] * m[6];
        inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
            + m[4] * m[3] * m[13]
            + m[12] * m[1] * m[7]
            - m[12] * m[3] * m[5];
        inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
            - m[4] * m[2] * m[13]
            - m[12] * m[1] * m[6]
            + m[12] * m[2] * m[5];
        inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
            - m[5] * m[3] * m[10]
            - m[9] * m[2] * m[7]
            + m[9] * m[3] * m[6];
        inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
            + m[4] * m[3] * m[10]
            + m[8] * m[2] * m[7]
            - m[8] * m[3] * m[6];
        inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
            - m[4] * m[3] * m[9]
            - m[8] * m[1] * m[7]
            + m[8] * m[3] * m[5];
        inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
            + m[4] * m[2] * m[9]
            + m[8] * m[1] * m[6]
            - m[8] * m[2] * m[5];

        let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
        if det == 0.0 {
            return Self::identity();
        }
        let inv_det = 1.0 / det;
        for x in inv.iter_mut() {
            *x *= inv_det;
        }
        Self { elements: inv }
    }

    /// Extract the rotation as a pure-rotation Matrix4 (normalizes the basis axes).
    pub fn extract_rotation(&self) -> Self {
        let m = &self.elements;
        let sx_inv = 1.0 / Vector3::new(m[0], m[1], m[2]).length();
        let sy_inv = 1.0 / Vector3::new(m[4], m[5], m[6]).length();
        let sz_inv = 1.0 / Vector3::new(m[8], m[9], m[10]).length();
        Self {
            elements: [
                m[0] * sx_inv,
                m[1] * sx_inv,
                m[2] * sx_inv,
                0.0,
                m[4] * sy_inv,
                m[5] * sy_inv,
                m[6] * sy_inv,
                0.0,
                m[8] * sz_inv,
                m[9] * sz_inv,
                m[10] * sz_inv,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        }
    }

    /// Decompose into translation, rotation, scale (matches three.js).
    pub fn decompose(&self) -> (Vector3, Quaternion, Vector3) {
        let m = &self.elements;
        let mut sx = Vector3::new(m[0], m[1], m[2]).length();
        let sy = Vector3::new(m[4], m[5], m[6]).length();
        let sz = Vector3::new(m[8], m[9], m[10]).length();
        if self.determinant() < 0.0 {
            sx = -sx;
        }
        let position = Vector3::new(m[12], m[13], m[14]);
        let rotation_only = Self::identity().with_basis(
            Vector3::new(m[0], m[1], m[2]) * (1.0 / sx),
            Vector3::new(m[4], m[5], m[6]) * (1.0 / sy),
            Vector3::new(m[8], m[9], m[10]) * (1.0 / sz),
        );
        let quaternion = Quaternion::from_rotation_matrix(&rotation_only);
        (position, quaternion, Vector3::new(sx, sy, sz))
    }

    fn with_basis(mut self, x: Vector3, y: Vector3, z: Vector3) -> Self {
        self.elements[0] = x.x;
        self.elements[1] = x.y;
        self.elements[2] = x.z;
        self.elements[4] = y.x;
        self.elements[5] = y.y;
        self.elements[6] = y.z;
        self.elements[8] = z.x;
        self.elements[9] = z.y;
        self.elements[10] = z.z;
        self
    }

    pub fn determinant(&self) -> f32 {
        let m = &self.elements;
        let n11 = m[0];
        let n12 = m[4];
        let n13 = m[8];
        let n14 = m[12];
        let n21 = m[1];
        let n22 = m[5];
        let n23 = m[9];
        let n24 = m[13];
        let n31 = m[2];
        let n32 = m[6];
        let n33 = m[10];
        let n34 = m[14];
        let n41 = m[3];
        let n42 = m[7];
        let n43 = m[11];
        let n44 = m[15];
        n41 * (n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33
            + n12 * n24 * n33
            + n13 * n22 * n34
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

    /// Build an Object3D-style orientation matrix where local -Z points from
    /// `eye` toward `target`. Matches three.js's `Matrix4.lookAt` (which is the
    /// inverse convention of a view matrix). Use [`Self::look_at`] for the view
    /// matrix variant.
    pub fn target_look_at(eye: Vector3, target: Vector3, up: Vector3) -> Self {
        let mut z = eye - target;
        if z.length_sq() == 0.0 {
            z = Vector3::new(0.0, 0.0, 1.0);
        }
        let z = z.normalize();
        let mut x = up.cross(z);
        if x.length_sq() == 0.0 {
            // up is parallel to z — perturb up slightly.
            let perturb = if up.z.abs() == 1.0 {
                Vector3::new(0.0001, 0.0, 1.0)
            } else {
                Vector3::new(0.0, 0.0, 0.0001)
            };
            x = (up + perturb).cross(z);
        }
        let x = x.normalize();
        let y = z.cross(x);
        let mut m = Self::identity();
        m.elements[0] = x.x;
        m.elements[1] = x.y;
        m.elements[2] = x.z;
        m.elements[4] = y.x;
        m.elements[5] = y.y;
        m.elements[6] = y.z;
        m.elements[8] = z.x;
        m.elements[9] = z.y;
        m.elements[10] = z.z;
        m
    }

    /// Build a view matrix looking from `eye` at `target`, with `up` as the up axis.
    /// A view matrix looking from `eye` at `target`.
    ///
    /// When `up` is parallel to the view direction there is no unique answer,
    /// and the cross product that normally gives the right-hand axis is zero.
    /// `Vector3::normalize` returns zero rather than NaN for that, so the
    /// result used to be a matrix whose x and y basis vectors were both zero —
    /// rank one, projecting the entire scene onto a line, which renders as a
    /// black frame. It is reachable by ordinary means: orbit straight over a
    /// planet's pole with the default `up` of +Y and the camera passes exactly
    /// through it.
    ///
    /// Any perpendicular will do in that case, so pick the world axis furthest
    /// from the view direction and build from that. `OrbitControls` separately
    /// keeps its polar angle off the poles, which is what stops the roll
    /// flipping as you cross one; this is the backstop for everything else.
    pub fn look_at(eye: Vector3, target: Vector3, up: Vector3) -> Self {
        let mut z = (eye - target).normalize();
        if z.length_sq() < 1e-12 {
            // Degenerate the other way: eye and target coincide.
            z = Vector3::new(0.0, 0.0, 1.0);
        }
        let mut x = up.cross(z);
        if x.length_sq() < 1e-12 {
            let alt = if z.x.abs() < 0.9 {
                Vector3::new(1.0, 0.0, 0.0)
            } else {
                Vector3::new(0.0, 1.0, 0.0)
            };
            x = alt.cross(z);
        }
        let x = x.normalize();
        let y = z.cross(x);
        let mut m = Self::identity();
        m.elements[0] = x.x;
        m.elements[1] = y.x;
        m.elements[2] = z.x;
        m.elements[4] = x.y;
        m.elements[5] = y.y;
        m.elements[6] = z.y;
        m.elements[8] = x.z;
        m.elements[9] = y.z;
        m.elements[10] = z.z;
        m.elements[12] = -x.dot(eye);
        m.elements[13] = -y.dot(eye);
        m.elements[14] = -z.dot(eye);
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_times_identity() {
        let i = Matrix4::identity();
        assert_eq!(i.multiply(&i), i);
    }

    #[test]
    fn translation_then_invert_is_negative() {
        let t = Matrix4::translation(Vector3::new(2.0, 3.0, 4.0));
        let inv = t.invert();
        assert!((inv.elements[12] + 2.0).abs() < 1e-5);
        assert!((inv.elements[13] + 3.0).abs() < 1e-5);
        assert!((inv.elements[14] + 4.0).abs() < 1e-5);
    }
}

#[cfg(test)]
mod clip_space {
    use super::Matrix4;

    /// Project a view-space point and return NDC z.
    fn ndc_z(m: &Matrix4, z_view: f32) -> f32 {
        let e = &m.elements;
        // column-major: clip = M * (0, 0, z, 1)
        let zc = e[8] * 0.0 + e[9] * 0.0 + e[10] * z_view + e[14];
        let wc = e[11] * z_view + e[15];
        zc / wc
    }

    // THE CRATE PROJECTS INTO WEBGPU CLIP SPACE: depth 0 at the near plane, 1 at
    // the far. `perspective` used to emit the OpenGL form (-1 at near) while
    // `orthographic` beside it emitted 0..1 and every consumer assumed 0..1 —
    // wgpu clips there, and the shader's shadow lookup rejects ndc.z < 0. The
    // visible near plane was therefore the harmonic mean 2nf/(n+f) rather than
    // n, and half the depth buffer went unaddressed. These pin the convention.

    #[test]
    fn perspective_puts_the_near_plane_at_zero() {
        let m = Matrix4::perspective(1.0, 1.0, 0.5, 100.0);
        assert!(ndc_z(&m, -0.5).abs() < 1e-5, "{}", ndc_z(&m, -0.5));
    }

    #[test]
    fn perspective_puts_the_far_plane_at_one() {
        let m = Matrix4::perspective(1.0, 1.0, 0.5, 100.0);
        assert!(
            (ndc_z(&m, -100.0) - 1.0).abs() < 1e-5,
            "{}",
            ndc_z(&m, -100.0)
        );
    }

    #[test]
    fn nothing_visible_is_clipped_between_near_and_far() {
        // The bug in one assertion: at n=324, f=952 the OpenGL form put the real
        // clip plane at 2nf/(n+f) = 483, so a subject starting at 339 lost its
        // front half.
        let m = Matrix4::perspective(0.6, 1.78, 324.0, 952.0);
        for d in [324.0_f32, 340.0, 483.0, 700.0, 952.0] {
            let z = ndc_z(&m, -d);
            assert!((-1e-5..=1.0 + 1e-5).contains(&z), "depth {d} -> ndc {z}");
        }
    }

    #[test]
    fn orthographic_agrees_with_perspective_about_the_convention() {
        let m = Matrix4::orthographic(-1.0, 1.0, 1.0, -1.0, 0.5, 100.0);
        let e = &m.elements;
        let z_at = |zv: f32| e[10] * zv + e[14];
        assert!(z_at(-0.5).abs() < 1e-5);
        assert!((z_at(-100.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn depth_is_monotonic_with_distance() {
        let m = Matrix4::perspective(1.0, 1.0, 1.0, 1000.0);
        let (mut prev, mut d) = (-1.0, 1.0_f32);
        while d <= 1000.0 {
            let z = ndc_z(&m, -d);
            assert!(z > prev, "not monotonic at {d}");
            prev = z;
            d *= 1.5;
        }
    }
}
