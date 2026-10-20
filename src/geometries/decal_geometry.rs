//! three.js r165 `DecalGeometry` (examples/jsm/geometries/DecalGeometry.js):
//! the part of a mesh inside a box projector, clipped against the box's six
//! planes, with UVs across the box's face. Ported operation for operation in
//! `f64`, as JavaScript computes it.

use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::{Quaternion, Vector3};

type V3 = [f64; 3];
/// Column-major, as three.js's `Matrix4.elements`.
type M4 = [f64; 16];

/// Euler rotation order, three.js's `Euler.order`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EulerOrder {
    #[default]
    Xyz,
    Yxz,
    Zxy,
    Zyx,
    Yzx,
    Xzy,
}

impl EulerOrder {
    /// `"XYZ"`, `"YXZ"`… as three.js spells them; anything else is `XYZ`.
    pub fn parse(s: &str) -> Self {
        match s {
            "YXZ" => Self::Yxz,
            "ZXY" => Self::Zxy,
            "ZYX" => Self::Zyx,
            "YZX" => Self::Yzx,
            "XZY" => Self::Xzy,
            _ => Self::Xyz,
        }
    }
}

/// `Matrix4.makeRotationFromEuler`.
fn rotation_from_euler(e: V3, order: EulerOrder) -> M4 {
    let mut te = [0.0; 16];
    let (x, y, z) = (e[0], e[1], e[2]);
    let (a, b) = (x.cos(), x.sin());
    let (c, d) = (y.cos(), y.sin());
    let (e, f) = (z.cos(), z.sin());
    match order {
        EulerOrder::Xyz => {
            let (ae, af, be, bf) = (a * e, a * f, b * e, b * f);
            te[0] = c * e;
            te[4] = -c * f;
            te[8] = d;
            te[1] = af + be * d;
            te[5] = ae - bf * d;
            te[9] = -b * c;
            te[2] = bf - ae * d;
            te[6] = be + af * d;
            te[10] = a * c;
        }
        EulerOrder::Yxz => {
            let (ce, cf, de, df) = (c * e, c * f, d * e, d * f);
            te[0] = ce + df * b;
            te[4] = de * b - cf;
            te[8] = a * d;
            te[1] = a * f;
            te[5] = a * e;
            te[9] = -b;
            te[2] = cf * b - de;
            te[6] = df + ce * b;
            te[10] = a * c;
        }
        EulerOrder::Zxy => {
            let (ce, cf, de, df) = (c * e, c * f, d * e, d * f);
            te[0] = ce - df * b;
            te[4] = -a * f;
            te[8] = de + cf * b;
            te[1] = cf + de * b;
            te[5] = a * e;
            te[9] = df - ce * b;
            te[2] = -a * d;
            te[6] = b;
            te[10] = a * c;
        }
        EulerOrder::Zyx => {
            let (ae, af, be, bf) = (a * e, a * f, b * e, b * f);
            te[0] = c * e;
            te[4] = be * d - af;
            te[8] = ae * d + bf;
            te[1] = c * f;
            te[5] = bf * d + ae;
            te[9] = af * d - be;
            te[2] = -d;
            te[6] = b * c;
            te[10] = a * c;
        }
        EulerOrder::Yzx => {
            let (ac, ad, bc, bd) = (a * c, a * d, b * c, b * d);
            te[0] = c * e;
            te[4] = bd - ac * f;
            te[8] = bc * f + ad;
            te[1] = f;
            te[5] = a * e;
            te[9] = -b * e;
            te[2] = -d * e;
            te[6] = ad * f + bc;
            te[10] = ac - bd * f;
        }
        EulerOrder::Xzy => {
            let (ac, ad, bc, bd) = (a * c, a * d, b * c, b * d);
            te[0] = c * e;
            te[4] = -f;
            te[8] = d * e;
            te[1] = ac * f + bd;
            te[5] = a * e;
            te[9] = ad * f - bc;
            te[2] = bc * f - ad;
            te[6] = b * e;
            te[10] = bd * f + ac;
        }
    }
    te[15] = 1.0;
    te
}

/// `Matrix4.invert`, the same cofactor expansion.
fn invert(te: &M4) -> M4 {
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
    let di = 1.0 / det;
    [
        t11 * di,
        (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44
            - n21 * n33 * n44)
            * di,
        (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44
            + n21 * n32 * n44)
            * di,
        (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43
            - n21 * n32 * n43)
            * di,
        t12 * di,
        (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44
            + n11 * n33 * n44)
            * di,
        (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44
            - n11 * n32 * n44)
            * di,
        (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43
            + n11 * n32 * n43)
            * di,
        t13 * di,
        (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44
            - n11 * n23 * n44)
            * di,
        (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44
            + n11 * n22 * n44)
            * di,
        (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43
            - n11 * n22 * n43)
            * di,
        t14 * di,
        (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34
            + n11 * n23 * n34)
            * di,
        (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34
            - n11 * n22 * n34)
            * di,
        (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33
            + n11 * n22 * n33)
            * di,
    ]
}

/// `Vector3.applyMatrix4`, perspective divide included.
fn apply_matrix4(v: V3, e: &M4) -> V3 {
    let (x, y, z) = (v[0], v[1], v[2]);
    let w = 1.0 / (e[3] * x + e[7] * y + e[11] * z + e[15]);
    [
        (e[0] * x + e[4] * y + e[8] * z + e[12]) * w,
        (e[1] * x + e[5] * y + e[9] * z + e[13]) * w,
        (e[2] * x + e[6] * y + e[10] * z + e[14]) * w,
    ]
}

/// `Vector3.transformDirection`.
fn transform_direction(v: V3, e: &M4) -> V3 {
    let (x, y, z) = (v[0], v[1], v[2]);
    let r = [
        e[0] * x + e[4] * y + e[8] * z,
        e[1] * x + e[5] * y + e[9] * z,
        e[2] * x + e[6] * y + e[10] * z,
    ];
    let l = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    let l = if l == 0.0 { 1.0 } else { l };
    [r[0] / l, r[1] / l, r[2] / l]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[derive(Debug, Clone, Copy)]
struct DecalVertex {
    position: V3,
    normal: V3,
}

fn clip(v0: DecalVertex, v1: DecalVertex, p: V3, s: f64) -> DecalVertex {
    let d0 = dot(v0.position, p) - s;
    let d1 = dot(v1.position, p) - s;
    let s0 = d0 / (d0 - d1);
    let lerp = |a: V3, b: V3| {
        [
            a[0] + s0 * (b[0] - a[0]),
            a[1] + s0 * (b[1] - a[1]),
            a[2] + s0 * (b[2] - a[2]),
        ]
    };
    DecalVertex {
        position: lerp(v0.position, v1.position),
        normal: lerp(v0.normal, v1.normal),
    }
}

fn clip_geometry(input: &[DecalVertex], plane: V3, size: V3) -> Vec<DecalVertex> {
    let mut out = Vec::with_capacity(input.len());
    let s = 0.5 * dot(size, plane).abs();
    for tri in input.chunks_exact(3) {
        let d = [
            dot(tri[0].position, plane) - s,
            dot(tri[1].position, plane) - s,
            dot(tri[2].position, plane) - s,
        ];
        let o = [d[0] > 0.0, d[1] > 0.0, d[2] > 0.0];
        match o.iter().filter(|&&b| b).count() {
            0 => out.extend_from_slice(tri),
            1 => {
                if o[1] {
                    let (n1, n2) = (tri[0], tri[2]);
                    let n3 = clip(tri[1], n1, plane, s);
                    let n4 = clip(tri[1], n2, plane, s);
                    out.extend_from_slice(&[n3, n2, n1, n2, n3, n4]);
                } else {
                    let (n1, n2, n3, n4) = if o[0] {
                        let (n1, n2) = (tri[1], tri[2]);
                        (n1, n2, clip(tri[0], n1, plane, s), clip(tri[0], n2, plane, s))
                    } else {
                        let (n1, n2) = (tri[0], tri[1]);
                        (n1, n2, clip(tri[2], n1, plane, s), clip(tri[2], n2, plane, s))
                    };
                    out.extend_from_slice(&[n1, n2, n3, n4, n3, n2]);
                }
            }
            2 => {
                for (k, &out_k) in o.iter().enumerate() {
                    if !out_k {
                        let n1 = tri[k];
                        let n2 = clip(n1, tri[(k + 1) % 3], plane, s);
                        let n3 = clip(n1, tri[(k + 2) % 3], plane, s);
                        out.extend_from_slice(&[n1, n2, n3]);
                    }
                }
            }
            _ => {}
        }
    }
    out
}

pub struct DecalGeometry;

impl DecalGeometry {
    /// three.js r165 `new DecalGeometry(mesh, position, orientation, size)`.
    ///
    /// `positions` / `normals` / `index` are the mesh geometry's buffers (in
    /// its local space) and `matrix_world` the mesh's `matrixWorld`
    /// (column-major); `orientation` is an Euler angle in `order`. The result
    /// is in world space, unindexed, with `position`, `normal` and `uv`.
    #[allow(clippy::too_many_arguments)]
    pub fn project(
        positions: &[f32],
        normals: &[f32],
        index: Option<&[u32]>,
        matrix_world: &[f64; 16],
        position: [f64; 3],
        orientation: [f64; 3],
        order: EulerOrder,
        size: [f64; 3],
    ) -> BufferGeometry {
        let mut projector = rotation_from_euler(orientation, order);
        projector[12] = position[0];
        projector[13] = position[1];
        projector[14] = position[2];
        Self::with_projector(positions, normals, index, matrix_world, &projector, size)
    }

    fn with_projector(
        positions: &[f32],
        normals: &[f32],
        index: Option<&[u32]>,
        matrix_world: &[f64; 16],
        projector: &M4,
        size: [f64; 3],
    ) -> BufferGeometry {
        let inverse = invert(projector);
        let count = positions.len() / 3;
        let read = |a: &[f32], i: usize| -> V3 {
            if i * 3 + 2 < a.len() {
                [a[i * 3] as f64, a[i * 3 + 1] as f64, a[i * 3 + 2] as f64]
            } else {
                [0.0; 3]
            }
        };
        let make = |i: usize| DecalVertex {
            position: apply_matrix4(apply_matrix4(read(positions, i), matrix_world), &inverse),
            normal: transform_direction(read(normals, i), matrix_world),
        };
        let mut verts: Vec<DecalVertex> = match index {
            Some(index) => index.iter().map(|&i| make(i as usize)).collect(),
            None => (0..count).map(make).collect(),
        };
        for plane in [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ] {
            verts = clip_geometry(&verts, plane, size);
        }
        let mut out_pos = Vec::with_capacity(verts.len() * 3);
        let mut out_nrm = Vec::with_capacity(verts.len() * 3);
        let mut out_uv = Vec::with_capacity(verts.len() * 2);
        for v in &verts {
            out_uv.extend_from_slice(&[
                (0.5 + v.position[0] / size[0]) as f32,
                (0.5 + v.position[1] / size[1]) as f32,
            ]);
            let p = apply_matrix4(v.position, projector);
            out_pos.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32]);
            out_nrm.extend_from_slice(&[v.normal[0] as f32, v.normal[1] as f32, v.normal[2] as f32]);
        }
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(out_pos, 3));
        g.set_attribute("normal", BufferAttribute::new(out_nrm, 3));
        g.set_attribute("uv", BufferAttribute::new(out_uv, 2));
        g
    }

    /// A decal on world-space triangles with a quaternion projector — the
    /// Rust-side convenience. Normals are the triangles' face normals, since
    /// there are none to read.
    pub fn new(
        base_positions: &[Vector3],
        base_indices: Option<&[u32]>,
        position: Vector3,
        orientation: Quaternion,
        size: Vector3,
    ) -> BufferGeometry {
        let tris: Vec<[usize; 3]> = match base_indices {
            Some(idx) => idx
                .chunks_exact(3)
                .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
                .collect(),
            None => (0..base_positions.len() / 3).map(|i| [i * 3, i * 3 + 1, i * 3 + 2]).collect(),
        };
        let mut positions = Vec::with_capacity(tris.len() * 9);
        let mut normals = Vec::with_capacity(tris.len() * 9);
        for t in tris {
            let [a, b, c] = t.map(|i| base_positions[i]);
            let n = (b - a).cross(c - a).normalize();
            for p in [a, b, c] {
                positions.extend_from_slice(&[p.x, p.y, p.z]);
                normals.extend_from_slice(&[n.x, n.y, n.z]);
            }
        }
        let (x, y, z, w) = (
            orientation.x as f64,
            orientation.y as f64,
            orientation.z as f64,
            orientation.w as f64,
        );
        // Matrix4.compose with unit scale.
        let (x2, y2, z2) = (x + x, y + y, z + z);
        let (xx, xy, xz) = (x * x2, x * y2, x * z2);
        let (yy, yz, zz) = (y * y2, y * z2, z * z2);
        let (wx, wy, wz) = (w * x2, w * y2, w * z2);
        let projector = [
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
            position.x as f64,
            position.y as f64,
            position.z as f64,
            1.0,
        ];
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        Self::with_projector(
            &positions,
            &normals,
            None,
            &identity,
            &projector,
            [size.x as f64, size.y as f64, size.z as f64],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_of_a_rotation_translation_round_trips() {
        let mut m = rotation_from_euler([0.3, -0.7, 1.1], EulerOrder::Yxz);
        m[12] = 1.0;
        m[13] = 2.0;
        m[14] = -3.0;
        let inv = invert(&m);
        let p = apply_matrix4(apply_matrix4([0.5, -0.25, 2.0], &m), &inv);
        for (a, b) in p.iter().zip([0.5, -0.25, 2.0]) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
