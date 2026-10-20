use crate::core::{BufferAttribute, BufferGeometry};
use std::f64::consts::PI;

pub struct PolyhedronGeometry;

type V = [f64; 3];

impl PolyhedronGeometry {
    /// three.js `PolyhedronGeometry(vertices, indices, radius, detail)`: each
    /// face subdivided `detail` times, projected onto a sphere of `radius`,
    /// as a non-indexed soup with spherical uvs — flat normals at detail 0,
    /// smooth above. `vertices` is flat `x, y, z, …`.
    pub fn new(vertices: &[f32], indices: &[u32], radius: f32, detail: usize) -> BufferGeometry {
        let v: Vec<f64> = vertices.iter().map(|&x| super::js_number(x)).collect();
        Self::build(&v, indices, super::js_number(radius), detail)
    }

    /// three.js's algorithm step for step, in f64, rounding once into the
    /// attributes — and its normals computed from those rounded attributes,
    /// as `computeVertexNormals` / `normalizeNormals` do.
    pub(crate) fn build(
        vertices: &[f64],
        indices: &[u32],
        radius: f64,
        detail: usize,
    ) -> BufferGeometry {
        let at = |i: u32| -> V {
            let s = i as usize * 3;
            [vertices[s], vertices[s + 1], vertices[s + 2]]
        };
        // `Vector3.lerp`: x += (v.x − x)·α.
        let lerp = |a: V, b: V, t: f64| -> V {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            ]
        };
        let mut buffer: Vec<V> = Vec::new();

        // `subdivide` / `subdivideFace`.
        let cols = detail + 1;
        for tri in indices.chunks_exact(3) {
            let (a, b, c) = (at(tri[0]), at(tri[1]), at(tri[2]));
            let mut grid: Vec<Vec<V>> = Vec::with_capacity(cols + 1);
            for i in 0..=cols {
                let aj = lerp(a, c, i as f64 / cols as f64);
                let bj = lerp(b, c, i as f64 / cols as f64);
                let rows = cols - i;
                grid.push(
                    (0..=rows)
                        .map(|j| {
                            if j == 0 && i == cols {
                                aj
                            } else {
                                lerp(aj, bj, j as f64 / rows as f64)
                            }
                        })
                        .collect(),
                );
            }
            for i in 0..cols {
                for j in 0..(2 * (cols - i) - 1) {
                    let k = j / 2;
                    if j % 2 == 0 {
                        buffer.extend_from_slice(&[grid[i][k + 1], grid[i + 1][k], grid[i][k]]);
                    } else {
                        buffer.extend_from_slice(&[
                            grid[i][k + 1],
                            grid[i + 1][k + 1],
                            grid[i + 1][k],
                        ]);
                    }
                }
            }
        }

        // `applyRadius`: `vertex.normalize().multiplyScalar(radius)`.
        for v in &mut buffer {
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let inv = 1.0 / if len == 0.0 { 1.0 } else { len };
            *v = [
                v[0] * inv * radius,
                v[1] * inv * radius,
                v[2] * inv * radius,
            ];
        }

        // `generateUVs`, `correctUVs`, `correctSeam`.
        let azimuth = |v: V| v[2].atan2(-v[0]);
        let inclination = |v: V| (-v[1]).atan2((v[0] * v[0] + v[2] * v[2]).sqrt());
        let mut uvs: Vec<f64> = Vec::with_capacity(buffer.len() * 2);
        for &v in &buffer {
            let u = azimuth(v) / 2.0 / PI + 0.5;
            let w = inclination(v) / PI + 0.5;
            uvs.extend_from_slice(&[u, 1.0 - w]);
        }
        for (f, face) in buffer.chunks_exact(3).enumerate() {
            let j = f * 6;
            let (ua, ub, uc) = (uvs[j], uvs[j + 2], uvs[j + 4]);
            // `copy(a).add(b).add(c).divideScalar(3)`.
            let third = 1.0 / 3.0;
            let centroid = [
                (face[0][0] + face[1][0] + face[2][0]) * third,
                (face[0][1] + face[1][1] + face[2][1]) * third,
                (face[0][2] + face[1][2] + face[2][2]) * third,
            ];
            let azi = azimuth(centroid);
            for (k, (uv_x, v)) in [(ua, face[0]), (ub, face[1]), (uc, face[2])]
                .into_iter()
                .enumerate()
            {
                let stride = j + 2 * k;
                if azi < 0.0 && uv_x == 1.0 {
                    uvs[stride] = uv_x - 1.0;
                }
                if v[0] == 0.0 && v[2] == 0.0 {
                    uvs[stride] = azi / 2.0 / PI + 0.5;
                }
            }
        }
        for j in (0..uvs.len()).step_by(6) {
            let (x0, x1, x2) = (uvs[j], uvs[j + 2], uvs[j + 4]);
            if x0.max(x1).max(x2) > 0.9 && x0.min(x1).min(x2) < 0.1 {
                for k in [j, j + 2, j + 4] {
                    if uvs[k] < 0.2 {
                        uvs[k] += 1.0;
                    }
                }
            }
        }

        let positions: Vec<f32> = buffer.iter().flat_map(|v| v.map(|c| c as f32)).collect();
        // The normal attribute starts as a copy of the positions.
        let mut normals = positions.clone();
        if detail == 0 {
            // `computeVertexNormals` on a soup: (c − b) × (a − b), stored,
            // then normalised.
            for t in 0..positions.len() / 9 {
                let p = |k: usize| -> V {
                    let i = t * 9 + k * 3;
                    [
                        positions[i] as f64,
                        positions[i + 1] as f64,
                        positions[i + 2] as f64,
                    ]
                };
                let (pa, pb, pc) = (p(0), p(1), p(2));
                let cb = [pc[0] - pb[0], pc[1] - pb[1], pc[2] - pb[2]];
                let ab = [pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]];
                let n = [
                    cb[1] * ab[2] - cb[2] * ab[1],
                    cb[2] * ab[0] - cb[0] * ab[2],
                    cb[0] * ab[1] - cb[1] * ab[0],
                ];
                for k in 0..3 {
                    normals[t * 9 + k * 3..t * 9 + k * 3 + 3].copy_from_slice(&n.map(|c| c as f32));
                }
            }
        }
        // `normalizeNormals`, on the stored values.
        for n in normals.chunks_exact_mut(3) {
            let (x, y, z) = (n[0] as f64, n[1] as f64, n[2] as f64);
            let len = (x * x + y * y + z * z).sqrt();
            let inv = 1.0 / if len == 0.0 { 1.0 } else { len };
            n.copy_from_slice(&[(x * inv) as f32, (y * inv) as f32, (z * inv) as f32]);
        }

        let mut geom = BufferGeometry::new();
        geom.set_attribute("position", BufferAttribute::new(positions, 3));
        geom.set_attribute("normal", BufferAttribute::new(normals, 3));
        geom.set_attribute(
            "uv",
            BufferAttribute::new(uvs.into_iter().map(|u| u as f32).collect(), 2),
        );
        geom
    }
}

pub struct TetrahedronGeometry;
impl TetrahedronGeometry {
    pub fn new(radius: f32, detail: usize) -> BufferGeometry {
        Self::from_f64(super::js_number(radius), detail)
    }

    /// [`new`](Self::new) with JavaScript's radius, as three.js takes it.
    pub fn from_f64(radius: f64, detail: usize) -> BufferGeometry {
        let v = [
            1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
        ];
        let i = [2u32, 1, 0, 0, 3, 2, 1, 3, 0, 2, 3, 1];
        PolyhedronGeometry::build(&v, &i, radius, detail)
    }
}

pub struct OctahedronGeometry;
impl OctahedronGeometry {
    pub fn new(radius: f32, detail: usize) -> BufferGeometry {
        Self::from_f64(super::js_number(radius), detail)
    }

    /// [`new`](Self::new) with JavaScript's radius, as three.js takes it.
    pub fn from_f64(radius: f64, detail: usize) -> BufferGeometry {
        let v = [
            1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            -1.0,
        ];
        let i = [
            0u32, 2, 4, 0, 4, 3, 0, 3, 5, 0, 5, 2, 1, 2, 5, 1, 5, 3, 1, 3, 4, 1, 4, 2,
        ];
        PolyhedronGeometry::build(&v, &i, radius, detail)
    }
}

pub struct IcosahedronGeometry;
impl IcosahedronGeometry {
    pub fn new(radius: f32, detail: usize) -> BufferGeometry {
        Self::from_f64(super::js_number(radius), detail)
    }

    /// [`new`](Self::new) with JavaScript's radius, as three.js takes it.
    pub fn from_f64(radius: f64, detail: usize) -> BufferGeometry {
        let t = (1.0 + 5f64.sqrt()) / 2.0;
        let v = [
            -1.0, t, 0.0, 1.0, t, 0.0, -1.0, -t, 0.0, 1.0, -t, 0.0, 0.0, -1.0, t, 0.0, 1.0, t, 0.0,
            -1.0, -t, 0.0, 1.0, -t, t, 0.0, -1.0, t, 0.0, 1.0, -t, 0.0, -1.0, -t, 0.0, 1.0,
        ];
        let i = [
            0u32, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10,
            7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2,
            10, 8, 6, 7, 9, 8, 1,
        ];
        PolyhedronGeometry::build(&v, &i, radius, detail)
    }
}

pub struct DodecahedronGeometry;
impl DodecahedronGeometry {
    pub fn new(radius: f32, detail: usize) -> BufferGeometry {
        Self::from_f64(super::js_number(radius), detail)
    }

    /// [`new`](Self::new) with JavaScript's radius, as three.js takes it.
    pub fn from_f64(radius: f64, detail: usize) -> BufferGeometry {
        let t = (1.0 + 5f64.sqrt()) / 2.0;
        let r = 1.0 / t;
        let v = [
            // (+-1, +-1, +-1)
            -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0,
            1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, // (0, +-1/phi, +-phi)
            0.0, -r, -t, 0.0, -r, t, 0.0, r, -t, 0.0, r, t, // (+-1/phi, +-phi, 0)
            -r, -t, 0.0, -r, t, 0.0, r, -t, 0.0, r, t, 0.0, // (+-phi, 0, +-1/phi)
            -t, 0.0, -r, t, 0.0, -r, -t, 0.0, r, t, 0.0, r,
        ];
        let i = [
            3u32, 11, 7, 3, 7, 15, 3, 15, 13, 7, 19, 17, 7, 17, 6, 7, 6, 15, 17, 4, 8, 17, 8, 10,
            17, 10, 6, 8, 0, 16, 8, 16, 2, 8, 2, 10, 0, 12, 1, 0, 1, 18, 0, 18, 16, 6, 10, 2, 6, 2,
            13, 6, 13, 15, 2, 16, 18, 2, 18, 3, 2, 3, 13, 18, 1, 9, 18, 9, 11, 18, 11, 3, 4, 14,
            12, 4, 12, 0, 4, 0, 8, 11, 9, 5, 11, 5, 19, 11, 19, 7, 19, 5, 14, 19, 14, 4, 19, 4, 17,
            1, 12, 14, 1, 14, 5, 1, 5, 9,
        ];
        PolyhedronGeometry::build(&v, &i, radius, detail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icosahedron_detail0_matches_threejs() {
        let g = IcosahedronGeometry::new(1.1, 0);
        let pos = g.get_attribute("position").expect("position");
        assert_eq!(pos.count(), 60, "20 triangles × 3 vertices");
        let data = &pos.array;
        let expected: [f32; 9] = [
            -0.9357, 0.0, 0.5783, 0.0, 0.5783, 0.9357, -0.5783, 0.9357, 0.0,
        ];
        for (i, (&a, &b)) in data[..9].iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 0.001, "pos[{i}] = {a}, expected {b}");
        }
    }
}
