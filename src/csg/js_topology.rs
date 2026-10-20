//! three.js r165's vector, plane, matrix and triangle math in f64, as
//! three-bvh-csg and three-mesh-bvh run it in JavaScript — operand order and
//! all, since the CSG's exact-equality tests see the last bit.

use crate::core::BufferAttribute;
use crate::math::{Matrix3, Matrix4, Triangle, Vector3};

pub const TOPO_EPSILON: f64 = 1e-10;

/// Mirrors three.js `Vector3` (IEEE f64 components).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct JsVec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl JsVec3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn from_v(v: Vector3) -> Self {
        Self::new(v.x as f64, v.y as f64, v.z as f64)
    }

    pub fn to_v(self) -> Vector3 {
        Vector3::new(self.x as f32, self.y as f32, self.z as f32)
    }

    /// Round-trip through f32 (matches JS `Float32Array` / `BufferAttribute` precision).
    pub fn snap_f32(self) -> Self {
        Self::from_v(self.to_v())
    }

    pub fn copy(mut self, other: Self) -> Self {
        self.x = other.x;
        self.y = other.y;
        self.z = other.z;
        self
    }

    pub fn set(mut self, x: f64, y: f64, z: f64) -> Self {
        self.x = x;
        self.y = y;
        self.z = z;
        self
    }

    pub fn sub_vectors(mut self, a: Self, b: Self) -> Self {
        self.x = a.x - b.x;
        self.y = a.y - b.y;
        self.z = a.z - b.z;
        self
    }

    pub fn cross_vectors(mut self, a: Self, b: Self) -> Self {
        let ax = a.x;
        let ay = a.y;
        let az = a.z;
        let bx = b.x;
        let by = b.y;
        let bz = b.z;
        self.x = ay * bz - az * by;
        self.y = az * bx - ax * bz;
        self.z = ax * by - ay * bx;
        self
    }

    pub fn dot(self, v: Self) -> f64 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    pub fn length_sq(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// `Vector3.length`: `sqrt(x² + y² + z²)`.
    pub fn length(self) -> f64 {
        self.length_sq().sqrt()
    }

    /// `Vector3.normalize`: `divideScalar(length || 1)`, which three.js does
    /// as a multiply by the reciprocal — a different rounding from dividing.
    pub fn normalize(mut self) -> Self {
        let l = self.length();
        let inv = 1.0 / if l == 0.0 { 1.0 } else { l };
        self.x *= inv;
        self.y *= inv;
        self.z *= inv;
        self
    }

    /// `Vector3.distanceTo`: `sqrt(distanceToSquared)`.
    pub fn distance_to(self, v: Self) -> f64 {
        self.distance_to_squared(v).sqrt()
    }

    pub fn distance_to_squared(self, v: Self) -> f64 {
        let dx = self.x - v.x;
        let dy = self.y - v.y;
        let dz = self.z - v.z;
        dx * dx + dy * dy + dz * dz
    }

    /// Mirrors `Vector3.angleTo`.
    pub fn angle_to(self, v: Self) -> f64 {
        let denom = (self.length_sq() * v.length_sq()).sqrt();
        if denom == 0.0 {
            return std::f64::consts::PI / 2.0;
        }
        // three.js computes:
        //   acos( clamp( dot / sqrt( lenSqA * lenSqB ), -1, 1 ) )
        let cos = self.dot(v) / denom;
        // Match three.js: only clamp to [-1, 1] before acos.
        cos.clamp(-1.0, 1.0).acos()
    }

    pub fn lerp_vectors(mut self, a: Self, b: Self, t: f64) -> Self {
        self.x = a.x + (b.x - a.x) * t;
        self.y = a.y + (b.y - a.y) * t;
        self.z = a.z + (b.z - a.z) * t;
        self
    }

    /// Mirrors `Vector3.addScaledVector` / `Vector4.addScaledVector`.
    pub fn add_scaled_vector(mut self, v: Self, s: f64) -> Self {
        self.x += v.x * s;
        self.y += v.y * s;
        self.z += v.z * s;
        self
    }

    /// Mirrors three.js `Vector3.applyMatrix4` (f64 arithmetic, matches `threejs-shim.js`).
    pub fn apply_matrix4(self, m: &JsMatrix4) -> Self {
        let e = &m.elements;
        let x = self.x;
        let y = self.y;
        let z = self.z;
        let w = e[3] * x + e[7] * y + e[11] * z + e[15];
        let inv_w = if w == 0.0 { 1.0 } else { 1.0 / w };
        Self::new(
            (e[0] * x + e[4] * y + e[8] * z + e[12]) * inv_w,
            (e[1] * x + e[5] * y + e[9] * z + e[13]) * inv_w,
            (e[2] * x + e[6] * y + e[10] * z + e[14]) * inv_w,
        )
    }

    /// Mirrors three.js `Vector3.applyNormalMatrix`.
    pub fn apply_normal_matrix(self, m: &JsMatrix3) -> Self {
        let e = &m.elements;
        Self::new(
            e[0] * self.x + e[3] * self.y + e[6] * self.z,
            e[1] * self.x + e[4] * self.y + e[7] * self.z,
            e[2] * self.x + e[5] * self.y + e[8] * self.z,
        )
        .normalize()
    }

    pub fn multiply_scalar(mut self, s: f64) -> Self {
        self.x *= s;
        self.y *= s;
        self.z *= s;
        self
    }
}

/// Mirrors three.js `Matrix4` (f64 elements, matches `threejs-shim.js` arithmetic).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct JsMatrix4 {
    pub elements: [f64; 16],
}

impl JsMatrix4 {
    /// three.js `Matrix4.determinant`, in f64.
    pub fn determinant(&self) -> f64 {
        let te = &self.elements;
        let (n11, n12, n13, n14) = (te[0], te[4], te[8], te[12]);
        let (n21, n22, n23, n24) = (te[1], te[5], te[9], te[13]);
        let (n31, n32, n33, n34) = (te[2], te[6], te[10], te[14]);
        let (n41, n42, n43, n44) = (te[3], te[7], te[11], te[15]);
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

    pub fn from_matrix4(m: &Matrix4) -> Self {
        Self {
            elements: m.elements.map(|e| e as f64),
        }
    }

    pub fn invert(self) -> Self {
        let m = &self.elements;
        let mut inv = [0.0f64; 16];
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

    pub fn multiply(self, other: Self) -> Self {
        let a = &self.elements;
        let b = &other.elements;
        let mut r = [0.0f64; 16];
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

    pub const fn identity() -> Self {
        Self {
            elements: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        }
    }
}

/// Mirrors three.js `Matrix3` (f64 elements).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct JsMatrix3 {
    pub elements: [f64; 9],
}

impl JsMatrix3 {
    /// three.js `Matrix3.getNormalMatrix(m)`: the upper 3×3 of `m`, inverted
    /// and transposed, in f64. A singular matrix gives zero, as three.js's.
    pub fn normal_matrix(m: &JsMatrix4) -> Self {
        let me = &m.elements;
        let (n11, n21, n31) = (me[0], me[1], me[2]);
        let (n12, n22, n32) = (me[4], me[5], me[6]);
        let (n13, n23, n33) = (me[8], me[9], me[10]);
        let t11 = n33 * n22 - n32 * n23;
        let t12 = n32 * n13 - n33 * n12;
        let t13 = n23 * n12 - n22 * n13;
        let det = n11 * t11 + n21 * t12 + n31 * t13;
        if det == 0.0 {
            return Self { elements: [0.0; 9] };
        }
        let inv = 1.0 / det;
        // The inverse, column-major, then transposed.
        let i = [
            t11 * inv,
            (n31 * n23 - n33 * n21) * inv,
            (n32 * n21 - n31 * n22) * inv,
            t12 * inv,
            (n33 * n11 - n31 * n13) * inv,
            (n31 * n12 - n32 * n11) * inv,
            t13 * inv,
            (n21 * n13 - n23 * n11) * inv,
            (n22 * n11 - n21 * n12) * inv,
        ];
        Self {
            elements: [i[0], i[3], i[6], i[1], i[4], i[7], i[2], i[5], i[8]],
        }
    }

    pub fn from_matrix3(m: &Matrix3) -> Self {
        Self {
            elements: m.elements.map(|e| e as f64),
        }
    }

    pub fn multiply_scalar(mut self, s: f64) -> Self {
        for e in &mut self.elements {
            *e *= s;
        }
        self
    }
}

/// Read a position attribute vertex as f64 (matches `Vector3.fromBufferAttribute`).
pub fn read_position_js(attr: &BufferAttribute, index: usize) -> JsVec3 {
    let i = index * attr.item_size;
    JsVec3::new(
        attr.array[i] as f64,
        attr.array[i + 1] as f64,
        attr.array[i + 2] as f64,
    )
}

/// `b.matrixWorld.invert().multiply(a.matrixWorld)` with JS f64 brush transforms.
pub fn matrix_a_to_b_brushes(a: &super::brush::CsgBrush, b: &super::brush::CsgBrush) -> JsMatrix4 {
    b.js_matrix_world().invert().multiply(a.js_matrix_world())
}

/// `b.matrixWorld.invert().multiply(a.matrixWorld)` in f64 (legacy matrix path).
pub fn matrix_a_to_b_js(a: &Matrix4, b: &Matrix4) -> JsMatrix4 {
    JsMatrix4::from_matrix4(b)
        .invert()
        .multiply(JsMatrix4::from_matrix4(a))
}

pub fn js_tri_from_indices(
    attr: &BufferAttribute,
    i0: usize,
    i1: usize,
    i2: usize,
    matrix: Option<&JsMatrix4>,
) -> JsTriangle {
    let mut a = read_position_js(attr, i0);
    let mut b = read_position_js(attr, i1);
    let mut c = read_position_js(attr, i2);
    if let Some(m) = matrix {
        a = a.apply_matrix4(m);
        b = b.apply_matrix4(m);
        c = c.apply_matrix4(m);
    }
    JsTriangle { a, b, c }
}

/// Barycentric world position via `addScaledVector` (matches `pushBarycoordInterpolatedValues`).
pub fn interp_bary_js(v0: JsVec3, v1: JsVec3, v2: JsVec3, b: [f64; 3]) -> JsVec3 {
    JsVec3::new(0.0, 0.0, 0.0)
        .add_scaled_vector(v0, b[0])
        .add_scaled_vector(v1, b[1])
        .add_scaled_vector(v2, b[2])
}

/// Mirrors three.js `Plane`.
#[derive(Debug, Clone, Copy, Default)]
pub struct JsPlane {
    pub normal: JsVec3,
    pub constant: f64,
}

impl JsPlane {
    pub fn set_from_normal_and_coplanar_point(mut self, normal: JsVec3, point: JsVec3) -> Self {
        self.normal = normal;
        self.constant = -normal.dot(point);
        self
    }

    pub fn distance_to_point(self, p: JsVec3) -> f64 {
        self.normal.dot(p) + self.constant
    }

    /// three.js `Plane.intersectLine`: the point where the segment crosses
    /// the plane, by `t = −(start·n + c) / (n·dir)` within `[0, 1]`; a segment
    /// in the plane gives its start. `target` is left as it was on a miss, as
    /// three.js leaves it — callers read it afterwards.
    pub fn intersect_line(self, start: JsVec3, end: JsVec3, target: &mut JsVec3) -> Option<JsVec3> {
        let direction = JsVec3::default().sub_vectors(end, start);
        let denominator = self.normal.dot(direction);
        if denominator == 0.0 {
            if self.distance_to_point(start) == 0.0 {
                *target = start;
                return Some(start);
            }
            return None;
        }
        let t = -(start.dot(self.normal) + self.constant) / denominator;
        if !(0.0..=1.0).contains(&t) {
            return None;
        }
        *target = start.add_scaled_vector(direction, t);
        Some(*target)
    }
}

/// Mirrors three.js `Line3`.
#[derive(Debug, Clone, Copy, Default)]
pub struct JsLine3 {
    pub start: JsVec3,
    pub end: JsVec3,
}

impl JsLine3 {
    pub fn distance(self) -> f64 {
        self.start.distance_to(self.end)
    }
}

/// Mirrors three.js `Triangle` + `ExtendedTriangle.intersectsTriangle`.
#[derive(Debug, Clone, Copy, Default)]
pub struct JsTriangle {
    pub a: JsVec3,
    pub b: JsVec3,
    pub c: JsVec3,
}

impl JsTriangle {
    pub fn from_triangle(t: Triangle) -> Self {
        Self {
            a: JsVec3::from_v(t.a),
            b: JsVec3::from_v(t.b),
            c: JsVec3::from_v(t.c),
        }
    }

    pub fn to_triangle(self) -> Triangle {
        Triangle::new(self.a.to_v(), self.b.to_v(), self.c.to_v())
    }

    pub fn copy(mut self, other: Self) -> Self {
        self.a = other.a;
        self.b = other.b;
        self.c = other.c;
        self
    }

    /// three.js `Triangle.getNormal`: `(c − b) × (a − b)`, scaled by
    /// `1 / sqrt(lengthSq)`, or zero for a degenerate triangle. The operand
    /// order sets the rounding, which exact-equality tests downstream see.
    pub fn get_normal(self) -> JsVec3 {
        let cb = JsVec3::default().sub_vectors(self.c, self.b);
        let ab = JsVec3::default().sub_vectors(self.a, self.b);
        let n = JsVec3::default().cross_vectors(cb, ab);
        let len_sq = n.length_sq();
        if len_sq > 0.0 {
            n.multiply_scalar(1.0 / len_sq.sqrt())
        } else {
            JsVec3::default()
        }
    }

    pub fn get_plane(self) -> JsPlane {
        let n = self.get_normal();
        JsPlane::default().set_from_normal_and_coplanar_point(n, self.a)
    }

    /// Mirrors `Triangle.getBarycoord` / `target.set(1 - u - v, v, u)`.
    pub fn get_barycoord(self, point: JsVec3) -> [f64; 3] {
        let v0 = JsVec3::default().sub_vectors(self.c, self.a);
        let v1 = JsVec3::default().sub_vectors(self.b, self.a);
        let v2 = JsVec3::default().sub_vectors(point, self.a);
        let dot00 = v0.dot(v0);
        let dot01 = v0.dot(v1);
        let dot02 = v0.dot(v2);
        let dot11 = v1.dot(v1);
        let dot12 = v1.dot(v2);
        let denom = dot00 * dot11 - dot01 * dot01;
        if denom == 0.0 {
            return [-2.0, -1.0, -1.0];
        }
        let inv = 1.0 / denom;
        let u = (dot11 * dot02 - dot01 * dot12) * inv;
        let v = (dot00 * dot12 - dot01 * dot02) * inv;
        [1.0 - u - v, v, u]
    }

    /// three.js `Triangle.getMidpoint`: `(a + b) + c`, times `1/3`.
    pub fn get_midpoint(self) -> JsVec3 {
        let third = 1.0 / 3.0;
        JsVec3::new(
            (self.a.x + self.b.x + self.c.x) * third,
            (self.a.y + self.b.y + self.c.y) * third,
            (self.a.z + self.b.z + self.c.z) * third,
        )
    }

    /// Mirrors `ExtendedTriangle.intersectsTriangle` in `web/mesh-bvh-impl.js`.
    /// three.js `Triangle.containsPoint`: inside or on the boundary, by
    /// barycentric coordinates; a degenerate triangle contains nothing.
    pub fn contains_point(self, point: JsVec3) -> bool {
        let v0 = JsVec3::default().sub_vectors(self.c, self.a);
        let v1 = JsVec3::default().sub_vectors(self.b, self.a);
        let v2 = JsVec3::default().sub_vectors(point, self.a);
        let dot00 = v0.dot(v0);
        let dot01 = v0.dot(v1);
        let dot02 = v0.dot(v2);
        let dot11 = v1.dot(v1);
        let dot12 = v1.dot(v2);
        let denom = dot00 * dot11 - dot01 * dot01;
        if denom == 0.0 {
            return false;
        }
        let inv = 1.0 / denom;
        let u = (dot11 * dot02 - dot01 * dot12) * inv;
        let v = (dot00 * dot12 - dot01 * dot02) * inv;
        let (x, y) = (1.0 - u - v, v);
        x >= 0.0 && y >= 0.0 && x + y <= 1.0
    }

    /// three-mesh-bvh 0.7.6 `ExtendedTriangle.intersectsTriangle`, in f64 as
    /// JavaScript runs it: a separating-axis test for (near-)parallel planes,
    /// otherwise each triangle's crossing of the other's plane, and whether
    /// the two crossing segments overlap. Touching counts. three-bvh-csg calls
    /// it to collect intersecting pairs and again before each plane split.
    pub fn intersects_triangle_js(self, other: Self) -> bool {
        const ZERO_EPSILON: f64 = 1e-15;
        let near_zero = |v: f64| v.abs() < ZERO_EPSILON;
        let plane1 = self.get_plane();
        let plane2 = other.get_plane();

        if plane1.normal.dot(plane2.normal).abs() > 1.0 - 1e-10 {
            // Separating axes: each normal and edge, then the edges' crosses.
            let axes = |t: Self| {
                [
                    t.get_normal(),
                    JsVec3::default().sub_vectors(t.a, t.b),
                    JsVec3::default().sub_vectors(t.b, t.c),
                    JsVec3::default().sub_vectors(t.c, t.a),
                ]
            };
            let bounds = |axis: JsVec3, t: Self| {
                let mut min = f64::INFINITY;
                let mut max = f64::NEG_INFINITY;
                for p in [t.a, t.b, t.c] {
                    let v = axis.dot(p);
                    min = if v < min { v } else { min };
                    max = if v > max { v } else { max };
                }
                (min, max)
            };
            let separated = |x: (f64, f64), y: (f64, f64)| x.0 > y.1 || y.0 > x.1;
            let (axes1, axes2) = (axes(self), axes(other));
            for axis in axes1 {
                if separated(bounds(axis, self), bounds(axis, other)) {
                    return false;
                }
            }
            for axis in axes2 {
                if separated(bounds(axis, other), bounds(axis, self)) {
                    return false;
                }
            }
            for a1 in axes1 {
                for a2 in axes2 {
                    let axis = JsVec3::default().cross_vectors(a1, a2);
                    if separated(bounds(axis, self), bounds(axis, other)) {
                        return false;
                    }
                }
            }
            return true;
        }

        // three.js `Plane.intersectLine`.
        let intersect_line = |plane: JsPlane, start: JsVec3, end: JsVec3| -> Option<JsVec3> {
            let dir = JsVec3::default().sub_vectors(end, start);
            let denominator = plane.normal.dot(dir);
            if denominator == 0.0 {
                return (plane.distance_to_point(start) == 0.0).then_some(start);
            }
            let t = -(start.dot(plane.normal) + plane.constant) / denominator;
            if !(0.0..=1.0).contains(&t) {
                return None;
            }
            Some(start.add_scaled_vector(dir, t))
        };
        // `triIntersectPlane`: the segment where `tri` crosses `plane`.
        let tri_intersect_plane = |tri: Self, plane: JsPlane| -> (u32, JsLine3) {
            let points = [tri.a, tri.b, tri.c];
            let mut target = JsLine3::default();
            let mut count = 0u32;
            let mut start_point_intersection: i32 = -1;
            for i in 0..3 {
                let start = points[i];
                let end = points[(i + 1) % 3];
                let dir = JsVec3::default().sub_vectors(end, start);
                let start_intersects = near_zero(plane.distance_to_point(start));
                if near_zero(plane.normal.dot(dir)) && start_intersects {
                    target = JsLine3 { start, end };
                    count = 2;
                    break;
                }
                let hit = intersect_line(plane, start, end);
                let point = match (hit, start_intersects) {
                    (Some(p), _) => Some(p),
                    (None, true) => Some(start),
                    (None, false) => None,
                };
                let Some(point) = point else { continue };
                if near_zero(point.distance_to(end)) {
                    continue;
                }
                if count <= 1 {
                    if count == 1 {
                        target.start = point;
                    } else {
                        target.end = point;
                    }
                    if start_intersects {
                        start_point_intersection = count as i32;
                    }
                } else {
                    if start_point_intersection == 1 {
                        target.start = point;
                    } else {
                        target.end = point;
                    }
                    count = 2;
                    break;
                }
                count += 1;
                if count == 2 && start_point_intersection == -1 {
                    break;
                }
            }
            (count, target)
        };

        let (count1, edge1) = tri_intersect_plane(self, plane2);
        if count1 == 1 && other.contains_point(edge1.end) {
            return true;
        } else if count1 != 2 {
            return false;
        }
        let (count2, mut edge2) = tri_intersect_plane(other, plane1);
        if count2 == 1 && self.contains_point(edge2.end) {
            return true;
        } else if count2 != 2 {
            return false;
        }
        let dir1 = JsVec3::default().sub_vectors(edge1.end, edge1.start);
        let dir2 = JsVec3::default().sub_vectors(edge2.end, edge2.start);
        if dir1.dot(dir2) < 0.0 {
            std::mem::swap(&mut edge2.start, &mut edge2.end);
        }
        let s1 = edge1.start.dot(dir1);
        let e1 = edge1.end.dot(dir1);
        let s2 = edge2.start.dot(dir1);
        let e2 = edge2.end.dot(dir1);
        let separated1 = e1 < s2;
        let separated2 = s1 < e2;
        !(s1 != e2 && s2 != e1 && separated1 == separated2)
    }
}

/// three-bvh-csg's `isTriDegenerate`: an angle, or an edge's squared length,
/// under `eps`.
pub fn is_tri_degenerate(tri: JsTriangle, eps: f64) -> bool {
    let ab = JsVec3::default().sub_vectors(tri.b, tri.a);
    let ac = JsVec3::default().sub_vectors(tri.c, tri.a);
    let cb = JsVec3::default().sub_vectors(tri.b, tri.c);

    let angle1 = ab.angle_to(ac);
    let angle2 = ab.angle_to(cb);
    let angle3 = std::f64::consts::PI - angle1 - angle2;

    angle1.abs() < eps
        || angle2.abs() < eps
        || angle3.abs() < eps
        || tri.a.distance_to_squared(tri.b) < eps
        || tri.a.distance_to_squared(tri.c) < eps
        || tri.b.distance_to_squared(tri.c) < eps
}

/// Barycentric triple in f64 (matches JS `getBarycoord`).
pub type SplitBary = ([f64; 3], [f64; 3], [f64; 3]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Triangle, Vector3};

    #[test]
    fn intersect_line_reuse_matches_js() {
        let tri = JsTriangle::from_triangle(Triangle::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(0.0, 2.0, 0.0),
        ));
        let clip = JsTriangle::from_triangle(Triangle::new(
            Vector3::new(0.5, -1.0, 0.0),
            Vector3::new(0.5, 2.0, 0.0),
            Vector3::new(0.5, 0.0, 1.0),
        ));
        let plane = clip.get_plane();
        let arr = [tri.a, tri.b, tri.c];
        let mut hit_vec = JsVec3::default();
        let mut intersects = 0;
        for t in 0..3 {
            let start = arr[t];
            let end = arr[(t + 1) % 3];
            let start_dist = plane.distance_to_point(start);
            if start_dist.abs() < TOPO_EPSILON {
                continue;
            }
            let did = plane.intersect_line(start, end, &mut hit_vec).is_some();
            if did && hit_vec.distance_to(start) >= TOPO_EPSILON {
                intersects += 1;
            }
        }
        assert_eq!(intersects, 2);
    }
}
