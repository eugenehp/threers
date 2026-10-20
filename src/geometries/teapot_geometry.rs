//! three.js r165 `TeapotGeometry` (examples/jsm/geometries/TeapotGeometry.js):
//! Newell's bicubic Bézier patches evaluated on a `segments × segments` grid,
//! with the same matrix formulation, in `f64`, stored as `f32`.

use super::teapot_data::{PATCHES, VERTICES};
use crate::core::{BufferAttribute, BufferGeometry};

/// Column-major, as `Matrix4.elements`.
type M4 = [f64; 16];

/// `Matrix4.set` takes its arguments row by row.
fn set(r: [f64; 16]) -> M4 {
    [
        r[0], r[4], r[8], r[12], r[1], r[5], r[9], r[13], r[2], r[6], r[10], r[14], r[3], r[7],
        r[11], r[15],
    ]
}

/// `Matrix4.multiplyMatrices(a, b)`, term for term.
fn mul(ae: &M4, be: &M4) -> M4 {
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

/// `Vector4.applyMatrix4`.
fn apply4(v: [f64; 4], e: &M4) -> [f64; 4] {
    let [x, y, z, w] = v;
    [
        e[0] * x + e[4] * y + e[8] * z + e[12] * w,
        e[1] * x + e[5] * y + e[9] * z + e[13] * w,
        e[2] * x + e[6] * y + e[10] * z + e[14] * w,
        e[3] * x + e[7] * y + e[11] * z + e[15] * w,
    ]
}

fn dot4(a: [f64; 4], b: [f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn transpose(m: &M4) -> M4 {
    let mut t = *m;
    for r in 0..4 {
        for c in 0..4 {
            t[c * 4 + r] = m[r * 4 + c];
        }
    }
    t
}

/// The options of three.js's constructor, defaults included.
#[derive(Debug, Clone, Copy)]
pub struct TeapotOptions {
    pub size: f64,
    pub segments: usize,
    pub bottom: bool,
    pub lid: bool,
    pub body: bool,
    pub fit_lid: bool,
    pub blinn: bool,
}

impl Default for TeapotOptions {
    fn default() -> Self {
        Self {
            size: 50.0,
            segments: 10,
            bottom: true,
            lid: true,
            body: true,
            fit_lid: true,
            blinn: true,
        }
    }
}

pub struct TeapotGeometry;

impl TeapotGeometry {
    /// three.js r165 `new TeapotGeometry(size, segments, bottom, lid, body, fitLid, blinn)`.
    ///
    /// Like three.js, the index buffer is sized for every triangle and the
    /// degenerate ones (at the poles) are skipped rather than written, so it
    /// ends in zeros — triangles `(0, 0, 0)`, which draw nothing.
    pub fn new(o: TeapotOptions) -> BufferGeometry {
        let segments = o.segments.max(2);
        let blinn_scale = 1.3;
        let max_height = 3.15 * if o.blinn { 1.0 } else { blinn_scale };
        let max_height2 = max_height / 2.0;
        let true_size = o.size / max_height2;

        let mut num_triangles = if o.bottom { (8 * segments - 4) * segments } else { 0 };
        num_triangles += if o.lid { (16 * segments - 4) * segments } else { 0 };
        num_triangles += if o.body { 40 * segments * segments } else { 0 };
        let mut indices = vec![0u32; num_triangles * 3];
        let mut num_vertices = if o.bottom { 4 } else { 0 };
        num_vertices += if o.lid { 8 } else { 0 };
        num_vertices += if o.body { 20 } else { 0 };
        num_vertices *= (segments + 1) * (segments + 1);
        let mut vertices = vec![0f32; num_vertices * 3];
        let mut normals = vec![0f32; num_vertices * 3];
        let mut uvs = vec![0f32; num_vertices * 2];

        let ms = set([
            -1.0, 3.0, -3.0, 1.0, 3.0, -6.0, 3.0, 0.0, -3.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ]);
        let mst = transpose(&ms);
        let not_degenerate = |v: &[f32], a: usize, b: usize, c: usize| {
            let same = |p: usize, q: usize| {
                v[p * 3] == v[q * 3] && v[p * 3 + 1] == v[q * 3 + 1] && v[p * 3 + 2] == v[q * 3 + 2]
            };
            !(same(a, b) || same(a, c) || same(b, c))
        };

        let min_patches = if o.body { 0 } else { 20 };
        let max_patches = if o.bottom { 32 } else { 28 };
        let vert_per_row = segments + 1;
        let (mut surf_count, mut vert_count, mut norm_count, mut uv_count, mut index_count) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut mgm = [[0.0f64; 16]; 3];
        for surf in min_patches..max_patches {
            if !(o.lid || !(20..28).contains(&surf)) {
                continue;
            }
            for (i, m) in mgm.iter_mut().enumerate() {
                let mut g = [0.0f64; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        let mut v = VERTICES[PATCHES[surf * 16 + r * 4 + c] as usize * 3 + i];
                        if o.fit_lid && (20..28).contains(&surf) && i != 2 {
                            v *= 1.077;
                        }
                        if !o.blinn && i == 2 {
                            v *= blinn_scale;
                        }
                        g[c * 4 + r] = v;
                    }
                }
                let gmx = set(g);
                let tmtx = mul(&gmx, &ms);
                *m = mul(&mst, &tmtx);
            }
            for sstep in 0..=segments {
                let s = sstep as f64 / segments as f64;
                for tstep in 0..=segments {
                    let t = tstep as f64 / segments as f64;
                    let (mut sp, mut tp, mut dsp, mut dtp) = ([0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4]);
                    let (mut sval, mut tval, mut dsval, mut dtval) = (1.0f64, 1.0f64, 0.0f64, 0.0f64);
                    for p in (0..4).rev() {
                        sp[p] = sval;
                        tp[p] = tval;
                        sval *= s;
                        tval *= t;
                        if p == 3 {
                            dsp[p] = 0.0;
                            dtp[p] = 0.0;
                            dsval = 1.0;
                            dtval = 1.0;
                        } else {
                            dsp[p] = dsval * (3 - p) as f64;
                            dtp[p] = dtval * (3 - p) as f64;
                            dsval *= s;
                            dtval *= t;
                        }
                    }
                    let (mut vert, mut sdir, mut tdir) = ([0.0; 3], [0.0; 3], [0.0; 3]);
                    for i in 0..3 {
                        vert[i] = dot4(apply4(sp, &mgm[i]), tp);
                        sdir[i] = dot4(apply4(dsp, &mgm[i]), tp);
                        tdir[i] = dot4(apply4(sp, &mgm[i]), dtp);
                    }
                    // norm = tdir × sdir, normalised.
                    let n = [
                        tdir[1] * sdir[2] - tdir[2] * sdir[1],
                        tdir[2] * sdir[0] - tdir[0] * sdir[2],
                        tdir[0] * sdir[1] - tdir[1] * sdir[0],
                    ];
                    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                    let l = if l == 0.0 { 1.0 } else { l };
                    let n = [n[0] / l, n[1] / l, n[2] / l];
                    let norm_out = if vert[0] == 0.0 && vert[1] == 0.0 {
                        [0.0, if vert[2] > max_height2 { 1.0 } else { -1.0 }, 0.0]
                    } else {
                        [n[0], n[2], -n[1]]
                    };
                    vertices[vert_count] = (true_size * vert[0]) as f32;
                    vertices[vert_count + 1] = (true_size * (vert[2] - max_height2)) as f32;
                    vertices[vert_count + 2] = (-true_size * vert[1]) as f32;
                    vert_count += 3;
                    for k in 0..3 {
                        normals[norm_count + k] = norm_out[k] as f32;
                    }
                    norm_count += 3;
                    uvs[uv_count] = (1.0 - t) as f32;
                    uvs[uv_count + 1] = (1.0 - s) as f32;
                    uv_count += 2;
                }
            }
            for sstep in 0..segments {
                for tstep in 0..segments {
                    let v1 = surf_count * vert_per_row * vert_per_row + sstep * vert_per_row + tstep;
                    let v2 = v1 + 1;
                    let v3 = v2 + vert_per_row;
                    let v4 = v1 + vert_per_row;
                    for tri in [[v1, v2, v3], [v1, v3, v4]] {
                        if not_degenerate(&vertices, tri[0], tri[1], tri[2]) {
                            for v in tri {
                                // A typed array ignores writes past its end.
                                if let Some(slot) = indices.get_mut(index_count) {
                                    *slot = v as u32;
                                }
                                index_count += 1;
                            }
                        }
                    }
                }
            }
            surf_count += 1;
        }

        let mut geom = BufferGeometry::new();
        geom.set_attribute("position", BufferAttribute::new(vertices, 3));
        geom.set_attribute("normal", BufferAttribute::new(normals, 3));
        geom.set_attribute("uv", BufferAttribute::new(uvs, 2));
        geom.set_index(indices);
        geom
    }
}
