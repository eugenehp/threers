//! Mesh exporters — Wavefront OBJ, STL (ASCII and binary) and Stanford PLY
//! (ASCII and binary) — written to match three.js r165's
//! `examples/jsm/exporters/{OBJ,STL,PLY}Exporter.js` byte for byte.
//!
//! Byte for byte means the arithmetic too. three.js reads each `Float32Array`
//! value into a JavaScript number (an `f64`), transforms it by the object's
//! `matrixWorld` and normals by its normal matrix in `f64`, and prints the
//! result with `Number.prototype.toString`. So do these: the transforms run in
//! `f64` in three.js's operation order, and [`js_number`] prints the way
//! JavaScript does. [`NumberFormat::Float32`] is the compact alternative —
//! each value rounded back to `f32` and printed shortest — which is what the
//! OpenSCAD front end writes (`geometry_to_obj` and friends, with `openscad`, are thin
//! wrappers over these writers).
//!
//! The input is a flat list of [`ExportObject`]s in traversal order, each with
//! its geometry and world matrix: the scene graph is the caller's. In the
//! browser that list comes from walking three.js-shaped JS objects.
//!
//! threers extension: [`ObjOptions::polygons`] and [`PlyOptions::polygons`]
//! write each polygon a geometry remembers ([`BufferGeometry::faces`]) as one
//! n-gon face instead of its triangles.
//!
//! glTF lives in [`crate::loaders::gltf_write`].

mod number;
mod obj;
mod ply;
mod stl;

use std::sync::Arc;

use crate::core::BufferGeometry;

pub use number::js_number;
pub use obj::{write_obj, ObjOptions};
pub use ply::{write_ply, PlyOptions};
pub use stl::{write_stl_ascii, write_stl_binary};

/// What an exported object is, which decides what each format writes for it
/// (three.js's `isMesh` / `isLine` / `isPoints`, and `type` for lines).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportKind {
    #[default]
    Mesh,
    /// A line strip (`type === 'Line'`): OBJ writes one `l` through every vertex.
    Line,
    /// Vertex pairs (`type === 'LineSegments'`): OBJ writes an `l` per pair.
    LineSegments,
    /// A closed strip (`type === 'LineLoop'`): three.js's OBJ writer emits its
    /// vertices and no `l` element, and so does this.
    LineLoop,
    /// A point cloud: OBJ `p`, PLY vertices without faces.
    Points,
}

/// How numbers are printed in the text formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NumberFormat {
    /// `f64` printed as JavaScript prints it — three.js byte parity.
    #[default]
    JavaScript,
    /// Rounded to `f32` and printed shortest: `0.1`, not `0.10000000149011612`.
    Float32,
}

impl NumberFormat {
    pub(crate) fn write(self, out: &mut String, v: f64) {
        match self {
            NumberFormat::JavaScript => out.push_str(&js_number(v)),
            NumberFormat::Float32 => {
                use std::fmt::Write;
                let v = v as f32;
                // `-0` reads as `0` everywhere; JavaScript prints it so too.
                let v = if v == 0.0 { 0.0 } else { v };
                let _ = write!(out, "{v}");
            }
        }
    }
}

/// One object to export: its geometry in object space and where it is.
#[derive(Debug, Clone)]
pub struct ExportObject {
    pub name: String,
    pub kind: ExportKind,
    pub geometry: Arc<BufferGeometry>,
    /// `matrixWorld.elements`: column-major, as three.js stores it.
    pub matrix_world: [f64; 16],
    /// The material's name; OBJ writes `usemtl` when it is non-empty.
    pub material_name: String,
}

/// The 4×4 identity, column-major.
pub const IDENTITY: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

impl ExportObject {
    /// A mesh at the origin, untransformed.
    pub fn mesh(name: impl Into<String>, geometry: impl Into<Arc<BufferGeometry>>) -> Self {
        Self {
            name: name.into(),
            kind: ExportKind::Mesh,
            geometry: geometry.into(),
            matrix_world: IDENTITY,
            material_name: String::new(),
        }
    }

    pub fn with_matrix_world(mut self, m: [f64; 16]) -> Self {
        self.matrix_world = m;
        self
    }

    pub fn with_kind(mut self, kind: ExportKind) -> Self {
        self.kind = kind;
        self
    }

    /// Every object in `arena` under `roots`, depth first in child order — the
    /// order `traverse` visits — with world matrices composed from each
    /// object's position, quaternion and scale. Meshes, lines and points only.
    pub fn from_arena(
        arena: &crate::core::ObjectArena,
        roots: &[crate::core::ObjectId],
    ) -> Vec<ExportObject> {
        fn visit(
            arena: &crate::core::ObjectArena,
            id: crate::core::ObjectId,
            parent: &[f64; 16],
            out: &mut Vec<ExportObject>,
        ) {
            let Some(o) = arena.get(id) else { return };
            let local = compose(
                [o.position.x as f64, o.position.y as f64, o.position.z as f64],
                [
                    o.quaternion.x as f64,
                    o.quaternion.y as f64,
                    o.quaternion.z as f64,
                    o.quaternion.w as f64,
                ],
                [o.scale.x as f64, o.scale.y as f64, o.scale.z as f64],
            );
            let world = multiply(parent, &local);
            if let crate::core::ObjectKind::Mesh(m) = &o.kind {
                out.push(ExportObject {
                    name: o.name.clone(),
                    kind: ExportKind::Mesh,
                    geometry: m.geometry.clone(),
                    matrix_world: world,
                    material_name: String::new(),
                });
            }
            for c in &o.children {
                visit(arena, *c, &world, out);
            }
        }
        let mut out = Vec::new();
        for r in roots {
            visit(arena, *r, &IDENTITY, &mut out);
        }
        out
    }
}

/// three.js `Matrix4.compose`, column-major.
pub fn compose(p: [f64; 3], q: [f64; 4], s: [f64; 3]) -> [f64; 16] {
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

/// three.js `Matrix4.multiplyMatrices(a, b)`, column-major.
pub fn multiply(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut te = [0.0; 16];
    for r in 0..4 {
        for c in 0..4 {
            te[c * 4 + r] = a[r] * b[c * 4]
                + a[4 + r] * b[c * 4 + 1]
                + a[8 + r] * b[c * 4 + 2]
                + a[12 + r] * b[c * 4 + 3];
        }
    }
    te
}

/// `Vector3.applyMatrix4`, operation for operation.
pub(crate) fn apply_matrix4(e: &[f64; 16], v: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = v;
    let w = 1.0 / (e[3] * x + e[7] * y + e[11] * z + e[15]);
    [
        (e[0] * x + e[4] * y + e[8] * z + e[12]) * w,
        (e[1] * x + e[5] * y + e[9] * z + e[13]) * w,
        (e[2] * x + e[6] * y + e[10] * z + e[14]) * w,
    ]
}

/// `new Matrix3().getNormalMatrix(m)`: the inverse transpose of the upper 3×3,
/// column-major, computed as `Matrix3.invert` does (all zeroes when singular).
pub(crate) fn normal_matrix(m: &[f64; 16]) -> [f64; 9] {
    // setFromMatrix4: column-major upper 3×3.
    let te = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
    let (n11, n21, n31) = (te[0], te[1], te[2]);
    let (n12, n22, n32) = (te[3], te[4], te[5]);
    let (n13, n23, n33) = (te[6], te[7], te[8]);
    let t11 = n33 * n22 - n32 * n23;
    let t12 = n32 * n13 - n33 * n12;
    let t13 = n23 * n12 - n22 * n13;
    let det = n11 * t11 + n21 * t12 + n31 * t13;
    if det == 0.0 {
        return [0.0; 9];
    }
    let d = 1.0 / det;
    let inv = [
        t11 * d,
        (n31 * n23 - n33 * n21) * d,
        (n32 * n21 - n31 * n22) * d,
        t12 * d,
        (n33 * n11 - n31 * n13) * d,
        (n31 * n12 - n32 * n11) * d,
        t13 * d,
        (n21 * n13 - n23 * n11) * d,
        (n22 * n11 - n21 * n12) * d,
    ];
    // transpose
    [
        inv[0], inv[3], inv[6], inv[1], inv[4], inv[7], inv[2], inv[5], inv[8],
    ]
}

/// `Vector3.applyMatrix3(m).normalize()`.
pub(crate) fn transform_normal(e: &[f64; 9], v: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = v;
    let v = [
        e[0] * x + e[3] * y + e[6] * z,
        e[1] * x + e[4] * y + e[7] * z,
        e[2] * x + e[5] * y + e[8] * z,
    ];
    normalize(v)
}

/// `Vector3.normalize`: `divideScalar(length() || 1)`, which multiplies by the
/// reciprocal.
pub(crate) fn normalize(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let s = 1.0 / if l == 0.0 || l.is_nan() { 1.0 } else { l };
    [v[0] * s, v[1] * s, v[2] * s]
}

/// three.js `LinearToSRGB`.
pub(crate) fn linear_to_srgb(c: f64) -> f64 {
    if c < 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(0.41666) - 0.055
    }
}

/// Item `i` of a named attribute as `f64`s, `None` past its end or when absent.
pub(crate) fn attr3(g: &BufferGeometry, name: &str, i: usize) -> [f64; 3] {
    let a = g.get_attribute(name).expect("checked by the caller");
    let s = a.item_size;
    let get = |k: usize| {
        if k < s {
            a.array.get(i * s + k).map(|v| *v as f64).unwrap_or(f64::NAN)
        } else {
            0.0
        }
    };
    [get(0), get(1), get(2)]
}

/// The vertex count three.js's `attribute.count` reports.
pub(crate) fn count(g: &BufferGeometry, name: &str) -> Option<usize> {
    g.get_attribute(name)
        .map(|a| a.array.len() / a.item_size.max(1))
}

#[cfg(test)]
mod tests;
