use super::cube_uv::{
    atlas_lod_origin, extract_cube_faces_from_atlas_lod, pmrem_face_uv, pmrem_get_direction,
    sample_atlas_pixels_bilinear_f32,
};
use crate::math::Vector3;
use crate::textures::{CubeTexture, CubeUvAtlas, TextureFormat};
use std::f32::consts::PI;
use std::sync::Arc;

/// Number of PMREM LOD planes — matches three.js for a 256³ cube (lodMax=8).
pub const PMREM_MIP_LEVELS: u32 = 11;

const LOD_MIN: u32 = 4;
const EXTRA_LOD_SIGMA: [f32; 6] = [0.125, 0.215, 0.35, 0.446, 0.526, 0.582];
const MAX_BLUR_SAMPLES: u32 = 20;

const PHI: f32 = 1.618_034;
const INV_PHI: f32 = 0.618_034;

/// Dodecahedron axis directions used by three.js PMREM blur passes.
const BLUR_POLE_AXES: [Vector3; 10] = [
    Vector3 {
        x: -PHI,
        y: INV_PHI,
        z: 0.0,
    },
    Vector3 {
        x: PHI,
        y: INV_PHI,
        z: 0.0,
    },
    Vector3 {
        x: -INV_PHI,
        y: 0.0,
        z: PHI,
    },
    Vector3 {
        x: INV_PHI,
        y: 0.0,
        z: PHI,
    },
    Vector3 {
        x: 0.0,
        y: PHI,
        z: -INV_PHI,
    },
    Vector3 {
        x: 0.0,
        y: PHI,
        z: INV_PHI,
    },
    Vector3 {
        x: -1.0,
        y: 1.0,
        z: -1.0,
    },
    Vector3 {
        x: 1.0,
        y: 1.0,
        z: -1.0,
    },
    Vector3 {
        x: -1.0,
        y: 1.0,
        z: 1.0,
    },
    Vector3 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    },
];

/// Prefiltered MipMap Radiance Environment generator. Builds a CPU-side
/// roughness mip chain and uploads it as a cube texture pyramid for IBL.
pub struct PmremGenerator;

impl PmremGenerator {
    pub fn from_cube(input: Arc<CubeTexture>) -> Arc<CubeTexture> {
        input
    }

    /// Build a PMREM CubeUV atlas from `input` at the requested base face size.
    pub fn generate_pmrem(input: &CubeTexture, size: u32) -> CubeTexture {
        let cube_size = size.clamp(16, 256);
        let base = prepare_pmrem_cube(input, cube_size);

        let lod_max = (cube_size as f32).log2().floor() as u32;
        let (size_lods, sigmas) = pmrem_lod_chain(lod_max);
        let level_count = size_lods.len();

        let mut current = base;
        if current.size != size_lods[0] {
            current = resize_cube(&current, size_lods[0]);
        }

        let mip0_cube = current.clone();

        let width = 3 * cube_size.max(16 * 7);
        let height = 4 * cube_size;
        let pixel_count = (width * height * 4) as usize;
        let mut atlas = vec![0.0f32; pixel_count];
        let mut ping = vec![0.0f32; pixel_count];

        rasterize_mip0_cube_uv_to_atlas_f32(&mut atlas, width, &mip0_cube, cube_size, lod_max);

        let mut mip_faces: Vec<[Vec<u8>; 6]> = vec![clone_cube_faces(&mip0_cube)];
        let mut mip_sizes: Vec<u32> = vec![size_lods[0]];

        let mut atlas_bytes = vec![0u8; pixel_count];
        for i in 1..level_count {
            let target_size = size_lods[i];
            let delta = (sigmas[i] * sigmas[i] - sigmas[i - 1] * sigmas[i - 1])
                .max(0.0)
                .sqrt();
            if delta > 0.0 {
                let pole = BLUR_POLE_AXES[(level_count - i - 1) % BLUR_POLE_AXES.len()];
                // Do not copy the full atlas into ping — three.js only renders the blur viewport
                // into the ping target; stale mip0 in ping was leaking into long-pass samples.
                blur_atlas_half(
                    &atlas,
                    &mut ping,
                    width,
                    height,
                    cube_size,
                    lod_max,
                    i as u32 - 1,
                    i as u32,
                    size_lods[i - 1],
                    target_size,
                    delta,
                    true,
                    pole,
                );
                blur_atlas_half(
                    &ping,
                    &mut atlas,
                    width,
                    height,
                    cube_size,
                    lod_max,
                    i as u32,
                    i as u32,
                    target_size,
                    target_size,
                    delta,
                    false,
                    pole,
                );
            }
            quantize_atlas_f32_to_u8(&atlas, &mut atlas_bytes);
            mip_faces.push(extract_cube_faces_from_atlas_lod(
                &atlas_bytes,
                width,
                cube_size,
                lod_max,
                i as u32,
                target_size,
            ));
            mip_sizes.push(target_size);
        }

        quantize_atlas_f32_to_u8(&atlas, &mut atlas_bytes);
        let atlas = CubeUvAtlas {
            width,
            height,
            cube_size,
            lod_max,
            texel_width: 1.0 / width as f32,
            texel_height: 1.0 / height as f32,
            pixels: Arc::new(atlas_bytes),
            pixels_f32: Some(Arc::new(atlas)),
        };

        let level0 = &mip_faces[0];
        CubeTexture::new(
            mip_sizes[0],
            TextureFormat::Rgba8Unorm,
            [
                level0[0].clone(),
                level0[1].clone(),
                level0[2].clone(),
                level0[3].clone(),
                level0[4].clone(),
                level0[5].clone(),
            ],
        )
        .with_pmrem_mips(mip_faces, mip_sizes)
        .with_cube_uv_atlas(atlas)
    }

    /// Prefilter `input` into a series of cube faces at a single roughness level.
    pub fn prefilter_cube(input: &CubeTexture, size: u32, roughness: f32) -> CubeTexture {
        let pmrem = Self::generate_pmrem(input, size.max(16));
        let mip = roughness_to_mip_index(roughness);
        let idx = mip.floor() as usize;
        let frac = mip - idx as f32;
        let idx1 = (idx + 1).min(pmrem.pmrem_sizes.as_ref().map(|s| s.len()).unwrap_or(1) - 1);
        if frac <= 0.0 || idx == idx1 {
            return cube_from_mip(&pmrem, idx);
        }
        blend_cube_mips(&pmrem, idx, idx1, frac)
    }

    /// Convert a **linear f32** equirectangular image (e.g. from
    /// [`crate::loaders::HdrLoader::parse_f32`]) into an HDR `CubeTexture`.
    ///
    /// This is the entry point for real image-based lighting: the returned cube
    /// keeps values above 1.0, and [`Self::generate_pmrem`] will prefilter them
    /// into a `Rgba16Float` atlas rather than clamping.
    pub fn from_equirect_f32(
        equirect: &[f32],
        src_w: u32,
        src_h: u32,
        cube_size: u32,
    ) -> CubeTexture {
        let n = cube_size.max(1) as usize;
        let face_len = n * n * 4;
        if equirect.is_empty() || src_w == 0 || src_h == 0 {
            return CubeTexture::new_f32(cube_size, std::array::from_fn(|_| vec![0.0; face_len]));
        }
        let (sw, sh) = (src_w as usize, src_h as usize);

        let faces: [Vec<f32>; 6] = std::array::from_fn(|face| {
            let mut out = vec![0.0f32; face_len];
            for y in 0..n {
                for x in 0..n {
                    let u = (x as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                    let v = (y as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                    // Normalise before the lookup. `cube_face_direction`
                    // returns a point on the *cube*, not the sphere — its
                    // longest component is 1 and the others run to ±1 — so
                    // `asin(d.y)` is not the latitude unless it is normalised
                    // first. Skipping that returns asin(±1) across the whole
                    // +Y and -Y faces, painting each of them a single flat
                    // colour, and skews the latitude on the other four
                    // everywhere except their vertical centre line. The 8-bit
                    // `from_equirect` below has always normalised; this is the
                    // HDR path, and so the one that mattered.
                    let d = cube_face_direction(face, u, v).normalize();

                    // Equirect lookup: longitude from atan2, latitude from asin.
                    let lon = d.z.atan2(d.x);
                    let lat = d.y.clamp(-1.0, 1.0).asin();
                    let su = (lon / (2.0 * PI) + 0.5).rem_euclid(1.0);
                    let sv = (0.5 - lat / PI).clamp(0.0, 1.0);

                    let fx = (su * sw as f32 - 0.5).clamp(0.0, (sw - 1) as f32);
                    let fy = (sv * sh as f32 - 0.5).clamp(0.0, (sh - 1) as f32);
                    let x0 = fx.floor() as usize;
                    let y0 = fy.floor() as usize;
                    let x1 = (x0 + 1) % sw; // wrap in longitude
                    let y1 = (y0 + 1).min(sh - 1);
                    let tx = fx - x0 as f32;
                    let ty = fy - y0 as f32;

                    let o = (y * n + x) * 4;
                    for c in 0..4 {
                        let p = |xx: usize, yy: usize| {
                            equirect.get((yy * sw + xx) * 4 + c).copied().unwrap_or(0.0)
                        };
                        let top = p(x0, y0) + (p(x1, y0) - p(x0, y0)) * tx;
                        let bot = p(x0, y1) + (p(x1, y1) - p(x0, y1)) * tx;
                        out[o + c] = top + (bot - top) * ty;
                    }
                }
            }
            out
        });
        CubeTexture::new_f32(cube_size, faces)
    }

    /// Convert an equirectangular HDR texture into a CubeTexture.
    pub fn from_equirect(
        equirect_rgba: &[u8],
        src_w: u32,
        src_h: u32,
        cube_size: u32,
    ) -> CubeTexture {
        let face_pixels = (cube_size as usize) * (cube_size as usize) * 4;
        if equirect_rgba.is_empty() || src_w == 0 || src_h == 0 {
            return CubeTexture::new(
                cube_size,
                TextureFormat::Rgba8UnormSrgb,
                [
                    vec![0u8; face_pixels],
                    vec![0u8; face_pixels],
                    vec![0u8; face_pixels],
                    vec![0u8; face_pixels],
                    vec![0u8; face_pixels],
                    vec![0u8; face_pixels],
                ],
            );
        }
        let sample = |u: f32, v: f32| -> [u8; 4] {
            let x = (u * src_w as f32).floor() as usize;
            let y = (v * src_h as f32).floor() as usize;
            let x = x.min(src_w as usize - 1);
            let y = y.min(src_h as usize - 1);
            let off = (y * src_w as usize + x) * 4;
            [
                equirect_rgba[off],
                equirect_rgba[off + 1],
                equirect_rgba[off + 2],
                equirect_rgba[off + 3],
            ]
        };
        let mut faces: [Vec<u8>; 6] = Default::default();
        for (face, slot) in faces.iter_mut().enumerate() {
            let mut data = Vec::with_capacity(face_pixels);
            for yi in 0..cube_size {
                for xi in 0..cube_size {
                    let u = (xi as f32 + 0.5) / cube_size as f32 * 2.0 - 1.0;
                    let v = (yi as f32 + 0.5) / cube_size as f32 * 2.0 - 1.0;
                    let dir = match face {
                        0 => [1.0, -v, -u],
                        1 => [-1.0, -v, u],
                        2 => [u, 1.0, v],
                        3 => [u, -1.0, -v],
                        4 => [u, -v, 1.0],
                        _ => [-u, -v, -1.0],
                    };
                    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
                    let nx = dir[0] / len;
                    let ny = dir[1] / len;
                    let nz = dir[2] / len;
                    let lon = nz.atan2(nx) / (2.0 * PI) + 0.5;
                    let lat = ny.asin() / PI + 0.5;
                    let p = sample(lon, lat);
                    data.extend_from_slice(&p);
                }
            }
            *slot = data;
        }
        let [f0, f1, f2, f3, f4, f5] = faces;
        CubeTexture::new(
            cube_size,
            TextureFormat::Rgba8UnormSrgb,
            [f0, f1, f2, f3, f4, f5],
        )
    }
}

fn pmrem_lod_chain(lod_max: u32) -> (Vec<u32>, Vec<f32>) {
    let mut sizes = Vec::new();
    let mut sigmas = Vec::new();
    let mut lod = lod_max;
    let total = lod_max - LOD_MIN + 1 + EXTRA_LOD_SIGMA.len() as u32;
    for i in 0..total {
        sizes.push(1u32 << lod);
        let sigma = if i == 0 {
            0.0
        } else if i > lod_max - LOD_MIN {
            EXTRA_LOD_SIGMA[(i - (lod_max - LOD_MIN) - 1) as usize]
        } else {
            1.0 / sizes[i as usize] as f32
        };
        sigmas.push(sigma);
        if lod > LOD_MIN {
            lod -= 1;
        }
    }
    (sizes, sigmas)
}

fn roughness_to_mip_index(roughness: f32) -> f32 {
    let mip = if roughness >= 0.8 {
        (1.0 - roughness) * (-1.0 - (-2.0)) / (1.0 - 0.8) + (-2.0)
    } else if roughness >= 0.4 {
        (0.8 - roughness) * (2.0 - (-1.0)) / (0.8 - 0.4) + (-1.0)
    } else if roughness >= 0.305 {
        (0.4 - roughness) * (3.0 - 2.0) / (0.4 - 0.305) + 2.0
    } else if roughness >= 0.21 {
        (0.305 - roughness) * (4.0 - 3.0) / (0.305 - 0.21) + 3.0
    } else {
        -2.0 * (1.16 * roughness.max(0.001)).log2()
    };
    ((mip + 2.0) / 10.0 * (PMREM_MIP_LEVELS - 1) as f32).clamp(0.0, PMREM_MIP_LEVELS as f32 - 1.0)
}

fn cube_from_mip(pmrem: &CubeTexture, idx: usize) -> CubeTexture {
    let mips = pmrem.pmrem_mips.as_ref().expect("pmrem mips");
    let sizes = pmrem.pmrem_sizes.as_ref().expect("pmrem sizes");
    let faces = &mips[idx];
    let size = sizes[idx];
    CubeTexture::new(
        size,
        pmrem.format,
        [
            faces[0].as_ref().clone(),
            faces[1].as_ref().clone(),
            faces[2].as_ref().clone(),
            faces[3].as_ref().clone(),
            faces[4].as_ref().clone(),
            faces[5].as_ref().clone(),
        ],
    )
}

fn blend_cube_mips(pmrem: &CubeTexture, a: usize, b: usize, t: f32) -> CubeTexture {
    let mips = pmrem.pmrem_mips.as_ref().expect("pmrem mips");
    let sizes = pmrem.pmrem_sizes.as_ref().expect("pmrem sizes");
    let size = sizes[a].max(sizes[b]);
    let mut out: [Vec<u8>; 6] = Default::default();
    for face in 0..6 {
        let fa = resize_face_bytes(mips[a][face].as_ref(), sizes[a], size, pmrem.format);
        let fb = resize_face_bytes(mips[b][face].as_ref(), sizes[b], size, pmrem.format);
        let mut data = Vec::with_capacity(fa.len());
        for (ca, cb) in fa.chunks(4).zip(fb.chunks(4)) {
            for c in 0..4 {
                let v = ca[c] as f32 * (1.0 - t) + cb[c] as f32 * t;
                data.push(v.round() as u8);
            }
        }
        out[face] = data;
    }
    let [f0, f1, f2, f3, f4, f5] = out;
    CubeTexture::new(size, pmrem.format, [f0, f1, f2, f3, f4, f5])
}

fn resize_face_bytes(data: &[u8], src_size: u32, dst_size: u32, format: TextureFormat) -> Vec<u8> {
    if src_size == dst_size {
        return data.to_vec();
    }
    let mut out = Vec::with_capacity((dst_size * dst_size * 4) as usize);
    for y in 0..dst_size {
        for x in 0..dst_size {
            let u = (x as f32 + 0.5) / dst_size as f32;
            let v = (y as f32 + 0.5) / dst_size as f32;
            let fx = (u * src_size as f32).clamp(0.0, (src_size as f32 - 1.001).max(0.0));
            let fy = (v * src_size as f32).clamp(0.0, (src_size as f32 - 1.001).max(0.0));
            let x0 = fx.floor() as usize;
            let y0 = fy.floor() as usize;
            let x1 = (x0 + 1).min(src_size as usize - 1);
            let y1 = (y0 + 1).min(src_size as usize - 1);
            let tx = fx - x0 as f32;
            let ty = fy - y0 as f32;
            for c in 0..3 {
                let sample = |sx: usize, sy: usize| -> f32 {
                    let v = data[(sy * src_size as usize + sx) * 4 + c] as f32 / 255.0;
                    if format == TextureFormat::Rgba8Unorm {
                        v
                    } else {
                        srgb_to_linear(v)
                    }
                };
                let v0 = sample(x0, y0) * (1.0 - tx) + sample(x1, y0) * tx;
                let v1 = sample(x0, y1) * (1.0 - tx) + sample(x1, y1) * tx;
                let lin = v0 * (1.0 - ty) + v1 * ty;
                out.push(if format == TextureFormat::Rgba8Unorm {
                    linear_to_byte(lin)
                } else {
                    linear_to_srgb_byte(lin)
                });
            }
            out.push(255);
        }
    }
    out
}

fn clone_cube_faces(cube: &CubeTexture) -> [Vec<u8>; 6] {
    [
        cube.faces[0].as_ref().clone(),
        cube.faces[1].as_ref().clone(),
        cube.faces[2].as_ref().clone(),
        cube.faces[3].as_ref().clone(),
        cube.faces[4].as_ref().clone(),
        cube.faces[5].as_ref().clone(),
    ]
}

fn quantize_atlas_f32_to_u8(src: &[f32], dst: &mut [u8]) {
    for (px, chunk) in src.chunks(4).zip(dst.chunks_mut(4)) {
        if chunk.len() < 4 {
            break;
        }
        chunk[0] = linear_to_byte(px[0]);
        chunk[1] = linear_to_byte(px.get(1).copied().unwrap_or(0.0));
        chunk[2] = linear_to_byte(px.get(2).copied().unwrap_or(0.0));
        chunk[3] = 255;
    }
}

/// Rasterize mip0 like three.js `_textureToCubeUV` (direction × flipEnvMap cubemap sample).
fn rasterize_mip0_cube_uv_to_atlas_f32(
    atlas: &mut [f32],
    atlas_w: u32,
    cube: &CubeTexture,
    cube_size: u32,
    lod_max: u32,
) {
    let face_size = cube_size;
    let (x_off, y_off) = atlas_lod_origin(0, face_size, cube_size, lod_max);
    for pmrem_slot in 0..6u32 {
        let col = pmrem_slot % 3;
        let row = super::cube_uv::pmrem_slot_row(pmrem_slot as usize);
        let dst_x = x_off + col * face_size;
        let dst_y = y_off + row * face_size;
        for yi in 0..face_size {
            for xi in 0..face_size {
                let (u, v) = pmrem_face_uv(face_size, xi, yi);
                let d = pmrem_get_direction(pmrem_slot, u, v);
                let rgb = sample_cube_linear(cube, Vector3::new(-d[0], d[1], d[2]));
                let dst_off =
                    (((dst_y + (face_size - 1 - yi)) * atlas_w + dst_x + xi) * 4) as usize;
                if dst_off + 3 < atlas.len() {
                    atlas[dst_off] = rgb[0];
                    atlas[dst_off + 1] = rgb[1];
                    atlas[dst_off + 2] = rgb[2];
                    atlas[dst_off + 3] = 1.0;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn blur_atlas_half(
    src: &[f32],
    dst: &mut [f32],
    width: u32,
    height: u32,
    cube_size: u32,
    lod_max: u32,
    lod_in: u32,
    lod_out: u32,
    kernel_size: u32,
    output_size: u32,
    sigma: f32,
    latitudinal: bool,
    pole_axis: Vector3,
) {
    if sigma <= 0.0 || output_size <= 1 {
        return;
    }

    let pixels = (kernel_size.saturating_sub(1)) as f32;
    let radians_per_pixel = if pixels > 0.0 {
        PI / (2.0 * pixels)
    } else {
        2.0 * PI / (2.0 * MAX_BLUR_SAMPLES as f32 - 1.0)
    };
    let sigma_pixels = sigma / radians_per_pixel;
    let sample_count = ((1.0 + 3.0 * sigma_pixels).floor() as u32 + 1).min(MAX_BLUR_SAMPLES);

    let mut weights = vec![0.0f32; MAX_BLUR_SAMPLES as usize];
    let mut sum = 0.0f32;
    for i in 0..MAX_BLUR_SAMPLES {
        let x = i as f32 / sigma_pixels.max(1.0);
        let w = (-x * x / 2.0).exp();
        weights[i as usize] = w;
        if i == 0 {
            sum += w;
        } else if i < sample_count {
            sum += 2.0 * w;
        }
    }
    for w in &mut weights {
        *w /= sum;
    }

    let mip_int = lod_max as f32 - lod_in as f32;
    let pole_axis = pole_axis.normalize();
    let (dst_x0, dst_y0) =
        super::cube_uv::atlas_lod_origin(lod_out, output_size, cube_size, lod_max);

    for pmrem_face in 0..6u32 {
        let col = pmrem_face % 3;
        let row = super::cube_uv::pmrem_slot_row(pmrem_face as usize);
        let face_x0 = dst_x0 + col * output_size;
        let face_y0 = dst_y0 + row * output_size;

        for yi in 0..output_size {
            for xi in 0..output_size {
                let (u, v) = pmrem_face_uv(output_size, xi, yi);
                let dir = pmrem_get_direction(pmrem_face, u, v);
                let dir_vec = Vector3::new(dir[0], dir[1], dir[2]);

                let mut axis = if latitudinal {
                    pole_axis
                } else {
                    let a = pole_axis.cross(dir_vec);
                    if a.length_sq() < 1e-8 {
                        Vector3::new(dir_vec.z, 0.0, -dir_vec.x)
                    } else {
                        a.normalize()
                    }
                };
                if axis.length_sq() < 1e-8 {
                    axis = Vector3::new(dir_vec.z, 0.0, -dir_vec.x);
                } else {
                    axis = axis.normalize();
                }

                let mut total = [0.0f32; 3];
                let d_theta = radians_per_pixel;
                for i in 0..sample_count {
                    let theta = d_theta * i as f32;
                    let w = weights[i as usize];
                    if i == 0 {
                        let sd0 = rotate_dir_raw(axis, dir_vec, 0.0);
                        let rgb0 = sample_atlas_pixels_bilinear_f32(
                            src,
                            width,
                            height,
                            lod_max,
                            [sd0[0], sd0[1], sd0[2]],
                            mip_int,
                        );
                        total[0] += rgb0[0] * w;
                        total[1] += rgb0[1] * w;
                        total[2] += rgb0[2] * w;
                    } else {
                        let sn = rotate_dir_raw(axis, dir_vec, -theta);
                        let sp = rotate_dir_raw(axis, dir_vec, theta);
                        let rgb_n = sample_atlas_pixels_bilinear_f32(
                            src,
                            width,
                            height,
                            lod_max,
                            [sn[0], sn[1], sn[2]],
                            mip_int,
                        );
                        let rgb_p = sample_atlas_pixels_bilinear_f32(
                            src,
                            width,
                            height,
                            lod_max,
                            [sp[0], sp[1], sp[2]],
                            mip_int,
                        );
                        total[0] += (rgb_n[0] + rgb_p[0]) * w;
                        total[1] += (rgb_n[1] + rgb_p[1]) * w;
                        total[2] += (rgb_n[2] + rgb_p[2]) * w;
                    }
                }

                let dst_off =
                    (((face_y0 + (output_size - 1 - yi)) * width + face_x0 + xi) * 4) as usize;
                if dst_off + 3 < dst.len() {
                    dst[dst_off] = total[0];
                    dst[dst_off + 1] = total[1];
                    dst[dst_off + 2] = total[2];
                    dst[dst_off + 3] = 1.0;
                }
            }
        }
    }
}

#[cfg(test)]
fn blur_cube_resample(
    cube: &CubeTexture,
    lod_in_size: u32,
    out_size: u32,
    sigma: f32,
    pole_axis: Vector3,
) -> CubeTexture {
    if out_size <= 1 || sigma <= 0.0 {
        return if cube.size == out_size {
            cube.clone()
        } else {
            resize_cube(cube, out_size)
        };
    }
    let lat = half_blur_resample(cube, out_size, lod_in_size, sigma, true, pole_axis);
    half_blur_resample(&lat, out_size, out_size, sigma, false, pole_axis)
}

#[cfg(test)]
fn half_blur_resample(
    cube: &CubeTexture,
    out_size: u32,
    kernel_size: u32,
    sigma: f32,
    latitudinal: bool,
    pole_axis: Vector3,
) -> CubeTexture {
    let pixels = (kernel_size.saturating_sub(1)) as f32;
    let radians_per_pixel = if sigma > 0.0 && pixels > 0.0 {
        PI / (2.0 * pixels)
    } else {
        2.0 * PI / (2.0 * MAX_BLUR_SAMPLES as f32 - 1.0)
    };
    let sigma_pixels = if sigma > 0.0 {
        sigma / radians_per_pixel
    } else {
        0.0
    };
    let sample_count = if sigma > 0.0 {
        (1.0 + 3.0 * sigma_pixels).floor() as u32 + 1
    } else {
        1
    }
    .min(MAX_BLUR_SAMPLES);

    let mut weights = Vec::with_capacity(sample_count as usize);
    let mut sum = 0.0f32;
    for i in 0..sample_count {
        let x = i as f32 / sigma_pixels.max(1.0);
        let w = (-x * x / 2.0).exp();
        weights.push(w);
        sum += if i == 0 { w } else { 2.0 * w };
    }
    for w in &mut weights {
        *w /= sum;
    }

    let pole_axis = pole_axis.normalize();
    let face_pixels = (out_size as usize) * (out_size as usize) * 4;
    let mut faces: [Vec<u8>; 6] = Default::default();
    for (face, slot) in faces.iter_mut().enumerate() {
        let mut data = Vec::with_capacity(face_pixels);
        for yi in 0..out_size {
            for xi in 0..out_size {
                let u = (xi as f32 + 0.5) / out_size as f32 * 2.0 - 1.0;
                let v = (yi as f32 + 0.5) / out_size as f32 * 2.0 - 1.0;
                let dir = face_dir(face, u, v).normalize();
                let mut axis = if latitudinal {
                    pole_axis
                } else {
                    let a = pole_axis.cross(dir);
                    if a.length_sq() < 1e-8 {
                        Vector3::new(dir.z, 0.0, -dir.x)
                    } else {
                        a.normalize()
                    }
                };
                if axis.length_sq() < 1e-8 {
                    axis = Vector3::new(dir.z, 0.0, -dir.x);
                } else {
                    axis = axis.normalize();
                }

                let mut total = [0.0f32; 3];
                let d_theta = radians_per_pixel;
                for i in 0..sample_count {
                    let theta = d_theta * i as f32;
                    let w = weights[i as usize];
                    if i == 0 {
                        let sd0 = rotate_dir_raw(axis, dir, 0.0);
                        let rgb = sample_cube_linear(cube, Vector3::new(sd0[0], sd0[1], sd0[2]));
                        for c in 0..3 {
                            total[c] += rgb[c] * w;
                        }
                    } else {
                        let sn = rotate_dir_raw(axis, dir, -theta);
                        let sp = rotate_dir_raw(axis, dir, theta);
                        let rgb_n = sample_cube_linear(cube, Vector3::new(sn[0], sn[1], sn[2]));
                        let rgb_p = sample_cube_linear(cube, Vector3::new(sp[0], sp[1], sp[2]));
                        for c in 0..3 {
                            total[c] += (rgb_n[c] + rgb_p[c]) * w;
                        }
                    }
                }
                data.extend_from_slice(&[
                    linear_to_byte(total[0]),
                    linear_to_byte(total[1]),
                    linear_to_byte(total[2]),
                    255,
                ]);
            }
        }
        *slot = data;
    }
    let [f0, f1, f2, f3, f4, f5] = faces;
    CubeTexture::new(
        out_size,
        TextureFormat::Rgba8Unorm,
        [f0, f1, f2, f3, f4, f5],
    )
}

fn rotate_dir_raw(axis: Vector3, dir: Vector3, angle: f32) -> [f32; 3] {
    let cos = angle.cos();
    let sin = angle.sin();
    let r = dir * cos + axis.cross(dir) * sin + axis * axis.dot(dir) * (1.0 - cos);
    [r.x, r.y, r.z]
}

/// Linearize LDR cubemap texels for PMREM (sRGB → linear HalfFloat-equivalent working space).
fn prepare_pmrem_cube(input: &CubeTexture, size: u32) -> CubeTexture {
    let src = if input.size == size {
        input.clone()
    } else {
        resize_cube(input, size)
    };
    // HDR faces are already linear and are what `sample_cube_linear` reads, so
    // there is nothing to decode — and rebuilding the CubeTexture below would
    // drop them.
    if src.faces_f32.is_some() || src.format == TextureFormat::Rgba8Unorm {
        return src;
    }
    let mut faces: [Vec<u8>; 6] = Default::default();
    for (slot, data) in faces.iter_mut().zip(&src.faces) {
        let mut out = Vec::with_capacity(data.len());
        for px in data.chunks(4) {
            if px.len() < 3 {
                out.extend_from_slice(px);
                continue;
            }
            out.push(linear_to_byte(srgb_to_linear(px[0] as f32 / 255.0)));
            out.push(linear_to_byte(srgb_to_linear(px[1] as f32 / 255.0)));
            out.push(linear_to_byte(srgb_to_linear(px[2] as f32 / 255.0)));
            out.push(px.get(3).copied().unwrap_or(255));
        }
        *slot = out;
    }
    let [f0, f1, f2, f3, f4, f5] = faces;
    CubeTexture::new(size, TextureFormat::Rgba8Unorm, [f0, f1, f2, f3, f4, f5])
}

fn resize_cube(input: &CubeTexture, size: u32) -> CubeTexture {
    let face_pixels = (size as usize) * (size as usize) * 4;
    let mut faces: [Vec<u8>; 6] = Default::default();
    for (slot, source) in faces.iter_mut().zip(&input.faces) {
        *slot = resize_face_bytes(source.as_ref(), input.size, size, input.format);
        debug_assert_eq!(slot.len(), face_pixels);
    }
    let [f0, f1, f2, f3, f4, f5] = faces;
    let mut out = CubeTexture::new(size, input.format, [f0, f1, f2, f3, f4, f5]);
    // Carry the HDR faces through the resize too, else the f32 data is silently
    // dropped here and the atlas falls back to the clamped 8-bit copy.
    if let Some(src_f32) = input.faces_f32.as_ref() {
        out.faces_f32 = Some(std::array::from_fn(|i| {
            Arc::new(resize_face_f32(src_f32[i].as_ref(), input.size, size))
        }));
    }
    out
}

/// Bilinear resize of one linear f32 RGBA face.
fn resize_face_f32(src: &[f32], src_size: u32, dst_size: u32) -> Vec<f32> {
    let (sn, dn) = (src_size.max(1) as usize, dst_size.max(1) as usize);
    let mut out = vec![0.0f32; dn * dn * 4];
    if src.len() < sn * sn * 4 {
        return out;
    }
    for y in 0..dn {
        for x in 0..dn {
            // Sample at destination texel centres mapped into source space.
            let fx = ((x as f32 + 0.5) / dn as f32 * sn as f32 - 0.5).clamp(0.0, (sn - 1) as f32);
            let fy = ((y as f32 + 0.5) / dn as f32 * sn as f32 - 0.5).clamp(0.0, (sn - 1) as f32);
            let x0 = fx.floor() as usize;
            let y0 = fy.floor() as usize;
            let x1 = (x0 + 1).min(sn - 1);
            let y1 = (y0 + 1).min(sn - 1);
            let tx = fx - x0 as f32;
            let ty = fy - y0 as f32;
            for c in 0..4 {
                let p = |xx: usize, yy: usize| src[(yy * sn + xx) * 4 + c];
                let top = p(x0, y0) + (p(x1, y0) - p(x0, y0)) * tx;
                let bot = p(x0, y1) + (p(x1, y1) - p(x0, y1)) * tx;
                out[(y * dn + x) * 4 + c] = top + (bot - top) * ty;
            }
        }
    }
    out
}

/// Direction through texel (u, v) ∈ [-1, 1]² of cube face `face`, in the
/// `+X, -X, +Y, -Y, +Z, -Z` order [`CubeTexture`] documents. Not normalised.
fn cube_face_direction(face: usize, u: f32, v: f32) -> Vector3 {
    match face {
        0 => Vector3::new(1.0, -v, -u),
        1 => Vector3::new(-1.0, -v, u),
        2 => Vector3::new(u, 1.0, v),
        3 => Vector3::new(u, -1.0, -v),
        4 => Vector3::new(u, -v, 1.0),
        _ => Vector3::new(-u, -v, -1.0),
    }
}

#[cfg(test)]
fn face_dir(face: usize, u: f32, v: f32) -> Vector3 {
    cube_face_direction(face, u, v)
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_byte(c: f32) -> u8 {
    (c.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn linear_to_srgb_byte(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

fn sample_cube_linear(cube: &CubeTexture, dir: Vector3) -> [f32; 3] {
    // HDR path: f32 faces are already linear and unbounded, so sample them
    // directly. This is the single choke point that feeds the atlas — the rest
    // of the blur chain is f32 already, so values above 1.0 survive from here.
    if cube.faces_f32.is_some() {
        return sample_cube_raw_f32(cube, dir);
    }
    let rgb = sample_cube_raw(cube, dir);
    if matches!(cube.format, TextureFormat::Rgba8Unorm) {
        rgb
    } else {
        [
            srgb_to_linear(rgb[0]),
            srgb_to_linear(rgb[1]),
            srgb_to_linear(rgb[2]),
        ]
    }
}

/// Bilinear sample of the linear f32 faces. Mirrors `sample_cube_raw`'s face
/// selection exactly so the HDR and 8-bit paths agree on orientation.
fn sample_cube_raw_f32(cube: &CubeTexture, dir: Vector3) -> [f32; 3] {
    let Some(faces) = cube.faces_f32.as_ref() else {
        return [0.0; 3];
    };
    let abs_x = dir.x.abs();
    let abs_y = dir.y.abs();
    let abs_z = dir.z.abs();
    let (face, sc, tc, ma) = if abs_x >= abs_y && abs_x >= abs_z {
        if dir.x > 0.0 {
            (0, -dir.z, -dir.y, abs_x)
        } else {
            (1, dir.z, -dir.y, abs_x)
        }
    } else if abs_y >= abs_z {
        if dir.y > 0.0 {
            (2, dir.x, dir.z, abs_y)
        } else {
            (3, dir.x, -dir.z, abs_y)
        }
    } else if dir.z > 0.0 {
        (4, dir.x, -dir.y, abs_z)
    } else {
        (5, -dir.x, -dir.y, abs_z)
    };
    let ma = ma.max(1e-8);
    let s = ((sc / ma) * 0.5 + 0.5).clamp(0.0, 1.0);
    let t = ((tc / ma) * 0.5 + 0.5).clamp(0.0, 1.0);

    let n = cube.size.max(1) as usize;
    let data = &faces[face];
    if n <= 1 || data.len() < 4 {
        return [
            data.first().copied().unwrap_or(0.0),
            data.get(1).copied().unwrap_or(0.0),
            data.get(2).copied().unwrap_or(0.0),
        ];
    }
    // Identical texel mapping to `sample_cube_raw` — the two must agree or the
    // HDR and 8-bit paths disagree about where a face's edge is, which shows up
    // as ghosting near face seams once the source has real dynamic range.
    let max_coord = n as f32 - 1.001;
    let fx = (s * n as f32).clamp(0.0, max_coord);
    let fy = (t * n as f32).clamp(0.0, max_coord);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(n - 1);
    let y1 = (y0 + 1).min(n - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;

    let at = |x: usize, y: usize, c: usize| -> f32 {
        data.get((y * n + x) * 4 + c).copied().unwrap_or(0.0)
    };
    let mut out = [0.0f32; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let top = at(x0, y0, c) + (at(x1, y0, c) - at(x0, y0, c)) * tx;
        let bot = at(x0, y1, c) + (at(x1, y1, c) - at(x0, y1, c)) * tx;
        *o = top + (bot - top) * ty;
    }
    out
}

fn sample_cube_raw(cube: &CubeTexture, dir: Vector3) -> [f32; 3] {
    let abs_x = dir.x.abs();
    let abs_y = dir.y.abs();
    let abs_z = dir.z.abs();
    let (face, sc, tc, ma) = if abs_x >= abs_y && abs_x >= abs_z {
        if dir.x > 0.0 {
            (0, -dir.z, -dir.y, abs_x)
        } else {
            (1, dir.z, -dir.y, abs_x)
        }
    } else if abs_y >= abs_z {
        if dir.y > 0.0 {
            (2, dir.x, dir.z, abs_y)
        } else {
            (3, dir.x, -dir.z, abs_y)
        }
    } else if dir.z > 0.0 {
        (4, dir.x, -dir.y, abs_z)
    } else {
        (5, -dir.x, -dir.y, abs_z)
    };
    let s = ((sc / ma) * 0.5 + 0.5).clamp(0.0, 1.0);
    let t = ((tc / ma) * 0.5 + 0.5).clamp(0.0, 1.0);
    let face_data = &cube.faces[face];
    if cube.size <= 1 {
        let off = if face_data.len() >= 4 {
            0
        } else {
            return [0.0; 3];
        };
        return [
            face_data[off] as f32 / 255.0,
            face_data[off + 1] as f32 / 255.0,
            face_data[off + 2] as f32 / 255.0,
        ];
    }
    let max_coord = cube.size as f32 - 1.001;
    let fx = (s * cube.size as f32).clamp(0.0, max_coord);
    let fy = (t * cube.size as f32).clamp(0.0, max_coord);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(cube.size as usize - 1);
    let y1 = (y0 + 1).min(cube.size as usize - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let c00 = face_data[(y0 * cube.size as usize + x0) * 4 + c] as f32 / 255.0;
        let c10 = face_data[(y0 * cube.size as usize + x1) * 4 + c] as f32 / 255.0;
        let c01 = face_data[(y1 * cube.size as usize + x0) * 4 + c] as f32 / 255.0;
        let c11 = face_data[(y1 * cube.size as usize + x1) * 4 + c] as f32 / 255.0;
        let v0 = c00 * (1.0 - tx) + c10 * tx;
        let v1 = c01 * (1.0 - tx) + c11 * tx;
        out[c] = v0 * (1.0 - ty) + v1 * ty;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::textures::{CubeTexture, TextureFormat};

    fn solid(size: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut d = vec![0u8; (size * size * 4) as usize];
        for i in 0..(size * size) as usize {
            d[i * 4..i * 4 + 4].copy_from_slice(&[r, g, b, 255]);
        }
        d
    }

    /// Probe the prefiltered environment around a single bright sun and report
    /// how many distinct peaks it has. There must be exactly one.
    #[test]
    fn pmrem_hdr_sun_does_not_ghost() {
        let size = 64u32;
        let n = size as usize;
        // One tight, very bright sun along +Z; everything else near-black.
        let sun = [0.0f32, 0.0, 1.0];
        let faces: [Vec<f32>; 6] = std::array::from_fn(|face| {
            let mut out = vec![0.0f32; n * n * 4];
            for y in 0..n {
                for x in 0..n {
                    let u = (x as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                    let v = (y as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                    let d = cube_face_direction(face, u, v);
                    let l = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt().max(1e-6);
                    let cos = (d.x * sun[0] + d.y * sun[1] + d.z * sun[2]) / l;
                    let val = if cos > 0.995 { 400.0 } else { 0.01 };
                    let o = (y * n + x) * 4;
                    out[o] = val;
                    out[o + 1] = val;
                    out[o + 2] = val;
                    out[o + 3] = 1.0;
                }
            }
            out
        });
        let cube = CubeTexture::new_f32(size, faces);
        let pmrem = PmremGenerator::generate_pmrem(&cube, size);
        let atlas = pmrem.cube_uv_atlas.as_ref().expect("atlas");

        // Sweep a great circle through the sun and count local maxima that are
        // a meaningful fraction of the global peak.
        let mut samples = Vec::new();
        for i in 0..360 {
            let a = (i as f32).to_radians();
            let dir = [a.sin(), 0.0, a.cos()];
            let c = crate::extras::cube_uv::sample_cube_uv_env(atlas, dir, 0.25);
            samples.push(c[0]);
        }
        let peak = samples.iter().cloned().fold(0.0f32, f32::max);
        let mut peaks = 0;
        for i in 0..samples.len() {
            let prev = samples[(i + samples.len() - 1) % samples.len()];
            let next = samples[(i + 1) % samples.len()];
            if samples[i] >= prev && samples[i] > next && samples[i] > peak * 0.15 {
                peaks += 1;
            }
        }
        assert_eq!(
            peaks, 1,
            "prefiltered HDR sun shows {peaks} peaks around a great circle \
             (expected 1); a second peak is a ghost duplicated across a cube face seam"
        );
    }

    /// HDR faces must survive prefiltering. The blur chain is f32 internally
    /// and the atlas uploads as Rgba16Float, so a sun far above 1.0 should
    /// still be above 1.0 after PMREM — if this regresses to <= 1.0 something
    /// has quietly routed through the clamped 8-bit faces again.
    #[test]
    fn pmrem_preserves_values_above_one_from_f32_faces() {
        let size = 32u32;
        let n = (size * size) as usize;
        // Uniform, very bright environment: every direction reads 50.0.
        let faces: [Vec<f32>; 6] =
            std::array::from_fn(|_| (0..n).flat_map(|_| [50.0f32, 50.0, 50.0, 1.0]).collect());
        let cube = CubeTexture::new_f32(size, faces);
        assert!(cube.faces_f32.is_some(), "f32 faces should be retained");

        let pmrem = PmremGenerator::generate_pmrem(&cube, size);
        let atlas = pmrem.cube_uv_atlas.as_ref().expect("atlas");
        let px = atlas.pixels_f32.as_ref().expect("f32 atlas");
        let peak = px.chunks(4).map(|c| c[0]).fold(0.0f32, f32::max);
        assert!(
            peak > 10.0,
            "HDR range lost in PMREM: peak {peak} (expected ~50, <=1 means it \
             went through the 8-bit faces)"
        );
    }

    #[test]
    fn pmrem_blur_step01_mip0_plus_z_exact() {
        use super::super::cube_uv::sample_atlas_pixels_bilinear_f32;
        let size = 32;
        let lod_max = 5;
        let base = prepare_pmrem_cube(&parity_cube(size), size);
        let width = 3 * size.max(16 * 7);
        let height = 4 * size;
        let mut atlas = vec![0.0f32; (width * height * 4) as usize];
        rasterize_mip0_cube_uv_to_atlas_f32(&mut atlas, width, &base, size, lod_max);
        let c =
            sample_atlas_pixels_bilinear_f32(&atlas, width, height, lod_max, [0.0, 0.0, 1.0], 5.0);
        eprintln!("step01 mip0 exact +Z {c:?}");
        assert!(c[1] < 0.08 && c[1] > 0.03, "mip0 +Z green {}", c[1]);
    }

    #[test]
    fn pmrem_blur_step02_mip0_at_blur_output_pixel() {
        use super::super::cube_uv::{
            pmrem_face_uv, pmrem_get_direction, sample_atlas_pixels_bilinear_f32,
        };
        let size = 32;
        let lod_max = 5;
        let base = prepare_pmrem_cube(&parity_cube(size), size);
        let width = 3 * size.max(16 * 7);
        let height = 4 * size;
        let mut atlas = vec![0.0f32; (width * height * 4) as usize];
        rasterize_mip0_cube_uv_to_atlas_f32(&mut atlas, width, &base, size, lod_max);
        let exact =
            sample_atlas_pixels_bilinear_f32(&atlas, width, height, lod_max, [0.0, 0.0, 1.0], 5.0);
        for (xi, yi) in [(7u32, 7), (8, 8), (7, 8), (8, 7)] {
            let (u, v) = pmrem_face_uv(16, xi, yi);
            let d = pmrem_get_direction(2, u, v);
            let c = sample_atlas_pixels_bilinear_f32(&atlas, width, height, lod_max, d, 5.0);
            eprintln!("step02 mip0 sample at blur pixel ({xi},{yi}) dir {d:?} -> {c:?}");
        }
        eprintln!("step02 exact +Z {exact:?}");
    }

    #[test]
    fn pmrem_blur_step03_face_blur_first_pass() {
        let size = 32;
        let base = prepare_pmrem_cube(&parity_cube(size), size);
        let (_, sigmas) = pmrem_lod_chain(5);
        let delta = (sigmas[1] * sigmas[1] - sigmas[0] * sigmas[0])
            .max(0.0)
            .sqrt();
        let pole = BLUR_POLE_AXES[6];
        let blurred = blur_cube_resample(&base, 32, 16, delta, pole);
        let face = &blurred.faces[4];
        let g = face[((8 * 16 + 8) * 4 + 1) as usize] as f32 / 255.0;
        eprintln!("step03 face blur +Z center G={g:.4} (full lat+long)");
        let lat = half_blur_resample(&base, 16, 32, delta, true, pole);
        let g_lat = lat.faces[4][((8 * 16 + 8) * 4 + 1) as usize] as f32 / 255.0;
        eprintln!("step03 face blur lat-only +Z center G={g_lat:.4}");
    }

    #[test]
    fn pmrem_blur_step04_atlas_first_lat_only() {
        use super::super::cube_uv::sample_atlas_pixels_bilinear_f32;
        let size = 32;
        let cube_size = size;
        let lod_max = 5;
        let (size_lods, sigmas) = pmrem_lod_chain(lod_max);
        let base = prepare_pmrem_cube(&parity_cube(size), size);
        let width = 3 * cube_size.max(16 * 7);
        let height = 4 * cube_size;
        let mut atlas = vec![0.0f32; (width * height * 4) as usize];
        let mut ping = vec![0.0f32; atlas.len()];
        rasterize_mip0_cube_uv_to_atlas_f32(&mut atlas, width, &base, cube_size, lod_max);
        let delta = (sigmas[1] * sigmas[1] - sigmas[0] * sigmas[0])
            .max(0.0)
            .sqrt();
        let pole = BLUR_POLE_AXES[6];
        blur_atlas_half(
            &atlas,
            &mut ping,
            width,
            height,
            cube_size,
            lod_max,
            0,
            1,
            size_lods[0],
            size_lods[1],
            delta,
            true,
            pole,
        );
        let c =
            sample_atlas_pixels_bilinear_f32(&ping, width, height, lod_max, [0.0, 0.0, 1.0], 4.0);
        eprintln!("step04 atlas lat-only +Z mip_int=4 {c:?}");
    }

    #[test]
    fn pmrem_blur_step05_atlas_long_pass_delta() {
        use super::super::cube_uv::sample_atlas_pixels_bilinear_f32;
        let size = 32;
        let cube_size = size;
        let lod_max = 5;
        let (size_lods, sigmas) = pmrem_lod_chain(lod_max);
        let base = prepare_pmrem_cube(&parity_cube(size), size);
        let width = 3 * cube_size.max(16 * 7);
        let height = 4 * cube_size;
        let mut atlas = vec![0.0f32; (width * height * 4) as usize];
        let mut ping = vec![0.0f32; atlas.len()];
        rasterize_mip0_cube_uv_to_atlas_f32(&mut atlas, width, &base, cube_size, lod_max);
        let delta = (sigmas[1] * sigmas[1] - sigmas[0] * sigmas[0])
            .max(0.0)
            .sqrt();
        let pole = BLUR_POLE_AXES[6];
        blur_atlas_half(
            &atlas,
            &mut ping,
            width,
            height,
            cube_size,
            lod_max,
            0,
            1,
            size_lods[0],
            size_lods[1],
            delta,
            true,
            pole,
        );
        let lat =
            sample_atlas_pixels_bilinear_f32(&ping, width, height, lod_max, [0.0, 0.0, 1.0], 4.0);
        blur_atlas_half(
            &ping,
            &mut atlas,
            width,
            height,
            cube_size,
            lod_max,
            1,
            1,
            size_lods[1],
            size_lods[1],
            delta,
            false,
            pole,
        );
        let full =
            sample_atlas_pixels_bilinear_f32(&atlas, width, height, lod_max, [0.0, 0.0, 1.0], 4.0);
        eprintln!("step05 lat {lat:?} after long {full:?}");
    }

    fn parity_cube(size: u32) -> CubeTexture {
        CubeTexture::new(
            size,
            TextureFormat::Rgba8UnormSrgb,
            [
                solid(size, 255, 64, 64),
                solid(size, 64, 255, 64),
                solid(size, 64, 64, 255),
                solid(size, 255, 255, 64),
                solid(size, 255, 64, 255),
                solid(size, 64, 255, 255),
            ],
        )
    }

    #[test]
    fn pmrem_atlas_has_color_data() {
        let size = 32;
        let src = CubeTexture::new(
            size,
            TextureFormat::Rgba8UnormSrgb,
            [
                solid(size, 255, 64, 64),
                solid(size, 64, 255, 64),
                solid(size, 64, 64, 255),
                solid(size, 255, 255, 64),
                solid(size, 255, 64, 255),
                solid(size, 64, 255, 255),
            ],
        );
        let pmrem = PmremGenerator::generate_pmrem(&src, size);
        let atlas = pmrem.cube_uv_atlas.as_ref().expect("atlas");
        assert_eq!(atlas.width, 336);
        assert_eq!(atlas.height, 128);
        assert!((atlas.texel_width - 1.0 / 336.0).abs() < 1e-6);
        let non_zero = atlas
            .pixels
            .chunks(4)
            .filter(|c| c[0] | c[1] | c[2] > 0)
            .count();
        assert!(non_zero > 1000, "atlas mostly empty: {non_zero}");
        let sample = super::super::cube_uv::sample_atlas_bilinear(atlas, [0.0, 0.0, 1.0], 1.0);
        assert!(
            sample[0] + sample[1] + sample[2] > 0.5,
            "atlas sample black: {sample:?}"
        );
    }

    #[test]
    fn pmrem_cube_uv_samples_non_black_at_roughness() {
        let size = 32;
        let src = CubeTexture::new(
            size,
            TextureFormat::Rgba8UnormSrgb,
            [
                solid(size, 255, 64, 64),
                solid(size, 64, 255, 64),
                solid(size, 64, 64, 255),
                solid(size, 255, 255, 64),
                solid(size, 255, 64, 255),
                solid(size, 64, 255, 255),
            ],
        );
        let pmrem = PmremGenerator::generate_pmrem(&src, size);
        let atlas = pmrem.cube_uv_atlas.as_ref().expect("atlas");
        let mut non_zero = 0;
        for y in 32..64 {
            for x in 144..192 {
                let i = ((y * atlas.width + x) * 4) as usize;
                if atlas.pixels[i] | atlas.pixels[i + 1] | atlas.pixels[i + 2] > 0 {
                    non_zero += 1;
                }
            }
        }
        assert!(
            non_zero > 50,
            "filter_int strip empty at x=144: {non_zero} px"
        );

        let mip: f32 = if 0.45_f32 >= 0.8 {
            (1.0 - 0.45) * 1.0 / 0.2 - 2.0
        } else if 0.45_f32 >= 0.4 {
            (0.8 - 0.45) * 3.0 / 0.4 - 1.0
        } else {
            0.0
        };
        let mip_i = mip.floor();
        let mip_f = mip - mip_i;
        let c0 = super::super::cube_uv::sample_atlas_bilinear(atlas, [0.0, 0.0, 1.0], mip_i);
        let c1 = super::super::cube_uv::sample_atlas_bilinear(atlas, [0.0, 0.0, 1.0], mip_i + 1.0);
        let sample = [
            c0[0] * (1.0 - mip_f) + c1[0] * mip_f,
            c0[1] * (1.0 - mip_f) + c1[1] * mip_f,
            c0[2] * (1.0 - mip_f) + c1[2] * mip_f,
        ];
        let lum = sample[0] + sample[1] + sample[2];
        assert!(
            lum > 0.5,
            "roughness 0.45 sample too dark: {sample:?} mip={mip}"
        );
    }

    #[test]
    fn pmrem_direction_color_diag() {
        let size = 32;
        let src = CubeTexture::new(
            size,
            TextureFormat::Rgba8UnormSrgb,
            [
                solid(size, 255, 64, 64),
                solid(size, 64, 255, 64),
                solid(size, 64, 64, 255),
                solid(size, 255, 255, 64),
                solid(size, 255, 64, 255),
                solid(size, 64, 255, 255),
            ],
        );
        let pmrem = PmremGenerator::generate_pmrem(&src, size);
        let atlas = pmrem.cube_uv_atlas.as_ref().unwrap();
        let dirs = [
            ("+X", [1.0, 0.0, 0.0]),
            ("-X", [-1.0, 0.0, 0.0]),
            ("+Y", [0.0, 1.0, 0.0]),
            ("-Y", [0.0, -1.0, 0.0]),
            ("+Z", [0.0, 0.0, 1.0]),
            ("-Z", [0.0, 0.0, -1.0]),
        ];
        let rough = 0.45_f32;
        let mip = if rough >= 0.4 {
            (0.8 - rough) * 3.0 / 0.4 - 1.0
        } else {
            0.0
        };
        let mip_i = mip.floor();
        let mip_f = mip - mip_i;
        eprintln!("mip={mip} i={mip_i} f={mip_f}");
        for (name, d) in dirs {
            let c0 = super::super::cube_uv::sample_atlas_bilinear(atlas, d, mip_i);
            let c1 = super::super::cube_uv::sample_atlas_bilinear(atlas, d, mip_i + 1.0);
            let c = [
                c0[0] * (1.0 - mip_f) + c1[0] * mip_f,
                c0[1] * (1.0 - mip_f) + c1[1] * mip_f,
                c0[2] * (1.0 - mip_f) + c1[2] * mip_f,
            ];
            let rgb = [
                (c[0] * 255.0) as u8,
                (c[1] * 255.0) as u8,
                (c[2] * 255.0) as u8,
            ];
            eprintln!("{name} atlas {rgb:?}");
        }
        let refl = [0.0, 0.0, 1.0];
        let c0 = super::super::cube_uv::sample_atlas_bilinear(atlas, refl, mip_i);
        let c1 = super::super::cube_uv::sample_atlas_bilinear(atlas, refl, mip_i + 1.0);
        let c = [
            c0[0] * (1.0 - mip_f) + c1[0] * mip_f,
            c0[1] * (1.0 - mip_f) + c1[1] * mip_f,
            c0[2] * (1.0 - mip_f) + c1[2] * mip_f,
        ];
        eprintln!(
            "reflect +Z atlas linear {c:?} srgb approx {:?}",
            [
                (c[0] * 255.0) as u8,
                (c[1] * 255.0) as u8,
                (c[2] * 255.0) as u8
            ]
        );

        // Irradiance sample used by multiscattering (roughness = 1.0 → mip -2).
        let irr = super::super::cube_uv::sample_atlas_bilinear(atlas, [0.0, 0.0, 1.0], -2.0);
        eprintln!(
            "irradiance +Z mip-2 linear {irr:?} srgb {:?}",
            [
                (irr[0] * 255.0) as u8,
                (irr[1] * 255.0) as u8,
                (irr[2] * 255.0) as u8
            ]
        );

        // Simulate IBL for metal r=0.45 at sphere center.
        let radiance = c;
        let dot_nv = 1.0_f32;
        let c0 = [-1.0_f32, -0.0275, -0.572, 0.022];
        let c1 = [1.0, 0.0425, 1.04, -0.04];
        let rv = [
            rough * c0[0] + c1[0],
            rough * c0[1] + c1[1],
            rough * c0[2] + c1[2],
            rough * c0[3] + c1[3],
        ];
        let a004 = (rv[0] * rv[0]).min(2f32.powf(-9.28 * dot_nv)) * rv[0] + rv[1];
        let fab = [-1.04_f32 * a004 + rv[2], 1.04 * a004 + rv[3]];
        let fss_ess = fab[0] + fab[1];
        let ems = 1.0 - fab[0] - fab[1];
        let favg = 1.0_f32;
        let fms = fss_ess * favg / (1.0 - ems * favg);
        let cosine_weighted_irr = irr; // PI * sample / PI
        let lit = [
            radiance[0] * fss_ess + fms * ems * cosine_weighted_irr[0],
            radiance[1] * fss_ess + fms * ems * cosine_weighted_irr[1],
            radiance[2] * fss_ess + fms * ems * cosine_weighted_irr[2],
        ];
        eprintln!(
            "IBL simulated linear {lit:?} srgb {:?}",
            [
                (lit[0] * 255.0) as u8,
                (lit[1] * 255.0) as u8,
                (lit[2] * 255.0) as u8
            ]
        );

        for test_r in [0.35_f32, 0.45, 0.55] {
            let m = if test_r >= 0.4 {
                (0.8 - test_r) * 3.0 / 0.4 - 1.0
            } else {
                0.0
            };
            let mi = m.floor();
            let mf = m - mi;
            let c0 = super::super::cube_uv::sample_atlas_bilinear(atlas, [0.0, 0.0, 1.0], mi);
            let c1 = super::super::cube_uv::sample_atlas_bilinear(atlas, [0.0, 0.0, 1.0], mi + 1.0);
            let s = [
                c0[0] * (1.0 - mf) + c1[0] * mf,
                c0[1] * (1.0 - mf) + c1[1] * mf,
                c0[2] * (1.0 - mf) + c1[2] * mf,
            ];
            eprintln!(
                "+Z rough={test_r} mip={m:.3} rgb {:?}",
                [
                    (s[0] * 255.0) as u8,
                    (s[1] * 255.0) as u8,
                    (s[2] * 255.0) as u8
                ]
            );
        }
    }

    #[test]
    fn pmrem_atlas_threejs_uv_probe() {
        use super::super::cube_uv::{read_atlas_texel, sample_atlas_bilinear, sample_cube_uv_env};
        let size = 32;
        let src = CubeTexture::new(
            size,
            TextureFormat::Rgba8UnormSrgb,
            [
                solid(size, 255, 64, 64),
                solid(size, 64, 255, 64),
                solid(size, 64, 64, 255),
                solid(size, 255, 255, 64),
                solid(size, 255, 64, 255),
                solid(size, 64, 255, 255),
            ],
        );
        let pmrem = PmremGenerator::generate_pmrem(&src, size);
        let atlas = pmrem.cube_uv_atlas.as_ref().unwrap();
        if let (Some(mips), Some(sizes)) = (&pmrem.pmrem_mips, &pmrem.pmrem_sizes) {
            for (i, (faces, fs)) in mips.iter().zip(sizes.iter()).enumerate().take(8) {
                let face = &faces[4];
                let mid = ((fs / 2) * fs + fs / 2) as usize * 4;
                let g = face[mid + 1] as f32 / 255.0;
                eprintln!("mip_faces[{i}] size={fs} +Z center G={g:.4}");
            }
        }
        let env = sample_cube_uv_env(atlas, [0.0, 0.0, 1.0], 0.45);
        let mip1 = sample_atlas_bilinear(atlas, [0.0, 0.0, 1.0], 1.0);
        let mip2 = sample_atlas_bilinear(atlas, [0.0, 0.0, 1.0], 2.0);
        eprintln!("our +Z rough0.45 env {env:?}");
        eprintln!("our mip1 {mip1:?} mip2 {mip2:?}");
        for (u, v) in [(0.547619_f32, 0.4375_f32), (0.404762_f32, 0.4375_f32)] {
            let t = read_atlas_texel(&atlas.pixels, atlas.width, atlas.height, u, v);
            eprintln!("texel u={u:.6} v={v:.6} linear {t:?}");
        }
        let px = |x: u32, y: u32| {
            let i = ((y * atlas.width + x) * 4) as usize;
            [
                atlas.pixels[i] as f32 / 255.0,
                atlas.pixels[i + 1] as f32 / 255.0,
                atlas.pixels[i + 2] as f32 / 255.0,
            ]
        };
        eprintln!("our byte (183,56) {:?}", px(183, 56));
        eprintln!("our byte (136,56) {:?}", px(136, 56));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn gpu_cube_uv_atlas_upload_matches_cpu() {
        use super::super::cube_uv::{atlas_bilinear_uv, sample_cube_uv_env};
        use crate::renderer::gpu_texture::f16_bits_to_f32;
        use crate::renderer::gpu_texture::GpuCubeTexture;

        let size = 32;
        let src = CubeTexture::new(
            size,
            TextureFormat::Rgba8UnormSrgb,
            [
                solid(size, 255, 64, 64),
                solid(size, 64, 255, 64),
                solid(size, 64, 64, 255),
                solid(size, 255, 255, 64),
                solid(size, 255, 64, 255),
                solid(size, 64, 255, 255),
            ],
        );
        let pmrem = PmremGenerator::generate_pmrem(&src, size);
        let atlas = pmrem.cube_uv_atlas.as_ref().expect("atlas");

        let cpu = sample_cube_uv_env(atlas, [0.0, 0.0, 1.0], 0.45);
        let (u, v) = atlas_bilinear_uv(
            atlas.width,
            atlas.height,
            atlas.lod_max,
            [0.0, 0.0, 1.0],
            1.0,
        );
        let f32_px = atlas.pixels_f32.as_ref().expect("f32 atlas");
        let x = (u * (atlas.width - 1) as f32).round() as u32;
        let y = (v * (atlas.height - 1) as f32).round() as u32;
        let fi = ((y * atlas.width + x) * 4) as usize;
        let cpu_texel = [f32_px[fi], f32_px[fi + 1], f32_px[fi + 2]];

        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::all();
        let instance = wgpu::Instance::new(instance_desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("pmrem atlas test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        ))
        .expect("device");

        let gpu = GpuCubeTexture::upload(&device, &queue, &pmrem);
        let tex = gpu.cube_uv_texture.as_ref().expect("gpu atlas");

        let unpadded_bpr = atlas.width * 8;
        let padded_bpr = unpadded_bpr.div_ceil(256) * 256;
        let read_size = (padded_bpr * atlas.height) as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atlas readback"),
            size: read_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("atlas readback"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(atlas.height),
                },
            },
            wgpu::Extent3d {
                width: atlas.width,
                height: atlas.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map channel").expect("map async");
        let data = slice.get_mapped_range().expect("buffer range is mapped");

        let off = (y * padded_bpr + x * 8) as usize;
        let gpu_texel = [
            f16_bits_to_f32(u16::from_le_bytes([data[off], data[off + 1]])),
            f16_bits_to_f32(u16::from_le_bytes([data[off + 2], data[off + 3]])),
            f16_bits_to_f32(u16::from_le_bytes([data[off + 4], data[off + 5]])),
        ];

        eprintln!("cpu sample_cube_uv_env {cpu:?}");
        eprintln!("cpu texel at ({x},{y}) uv=({u:.6},{v:.6}) {cpu_texel:?}");
        eprintln!("gpu texel at ({x},{y}) {gpu_texel:?}");

        for i in 0..3 {
            assert!(
                (gpu_texel[i] - cpu_texel[i]).abs() < 0.002,
                "channel {i}: cpu={} gpu={}",
                cpu_texel[i],
                gpu_texel[i]
            );
        }
    }

    /// Render a 1×1 pass that runs the same CubeUV bilinear + mip blend as the PBR shader.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn gpu_cube_uv_shader_sample_matches_cpu() {
        use super::super::cube_uv::sample_cube_uv_env;
        use crate::renderer::gpu_texture::GpuCubeTexture;
        use wgpu::util::DeviceExt;

        let size = 32;
        let src = CubeTexture::new(
            size,
            TextureFormat::Rgba8UnormSrgb,
            [
                solid(size, 255, 64, 64),
                solid(size, 64, 255, 64),
                solid(size, 64, 64, 255),
                solid(size, 255, 255, 64),
                solid(size, 255, 64, 255),
                solid(size, 64, 255, 255),
            ],
        );
        let pmrem = PmremGenerator::generate_pmrem(&src, size);
        let atlas = pmrem.cube_uv_atlas.as_ref().expect("atlas");
        let cpu = sample_cube_uv_env(atlas, [0.0, 0.0, 1.0], 0.45);

        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::all();
        let instance = wgpu::Instance::new(instance_desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("cube uv sample test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        ))
        .expect("device");

        let gpu = GpuCubeTexture::upload(&device, &queue, &pmrem);
        let uv_view = gpu.cube_uv_view.as_ref().expect("uv view");
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("test env sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cube uv sample"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                r#"
struct Params { texel_w: f32, texel_h: f32, lod_max: f32, roughness: f32, _pad: vec2<f32> }
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> p: Params;

const CUBEUV_MIN_MIP_LEVEL: f32 = 4.0;
const CUBEUV_MIN_TILE_SIZE: f32 = 16.0;

fn cube_uv_get_face(direction: vec3<f32>) -> f32 {
    let abs_dir = abs(direction);
    if (abs_dir.x > abs_dir.z) {
        if (abs_dir.x > abs_dir.y) { return select(3.0, 0.0, direction.x > 0.0); }
        return select(4.0, 1.0, direction.y > 0.0);
    }
    if (abs_dir.z > abs_dir.y) { return select(5.0, 2.0, direction.z > 0.0); }
    return select(4.0, 1.0, direction.y > 0.0);
}

fn cube_uv_get_uv(direction: vec3<f32>, face: f32) -> vec2<f32> {
    var uv = vec2<f32>(0.0);
    if (face == 0.0) { uv = vec2(direction.z, direction.y) / abs(direction.x); }
    else if (face == 1.0) { uv = vec2(-direction.x, -direction.z) / abs(direction.y); }
    else if (face == 2.0) { uv = vec2(-direction.x, direction.y) / abs(direction.z); }
    else if (face == 3.0) { uv = vec2(-direction.z, direction.y) / abs(direction.x); }
    else if (face == 4.0) { uv = vec2(-direction.x, direction.z) / abs(direction.y); }
    else { uv = vec2(direction.x, direction.y) / abs(direction.z); }
    return 0.5 * (uv + 1.0);
}

fn roughness_to_mip(roughness: f32) -> f32 {
    var mip = 0.0;
    if (roughness >= 0.8) { mip = (1.0 - roughness) * 1.0 / 0.2 - 2.0; }
    else if (roughness >= 0.4) { mip = (0.8 - roughness) * 3.0 / 0.4 - 1.0; }
    else if (roughness >= 0.305) { mip = (0.4 - roughness) * 1.0 / 0.095 + 2.0; }
    else if (roughness >= 0.21) { mip = (0.305 - roughness) * 1.0 / 0.095 + 3.0; }
    else { mip = -2.0 * log2(1.16 * max(roughness, 0.001)); }
    return mip;
}

fn bilinear_cube_uv(direction: vec3<f32>, mip_int: f32) -> vec3<f32> {
    var face = cube_uv_get_face(direction);
    let filter_int = max(CUBEUV_MIN_MIP_LEVEL - mip_int, 0.0);
    var mip_i = max(mip_int, CUBEUV_MIN_MIP_LEVEL);
    let face_size = exp2(mip_i);
    var uv = cube_uv_get_uv(direction, face) * (face_size - 2.0) + 1.0;
    if (face > 2.0) { uv.y += face_size; face -= 3.0; }
    uv.x += face * face_size;
    uv.x += filter_int * 3.0 * CUBEUV_MIN_TILE_SIZE;
    uv.y += 4.0 * (exp2(p.lod_max) - face_size);
    uv.x *= p.texel_w;
    uv.y *= p.texel_h;
    uv.y = 1.0 - uv.y;
    return textureSampleLevel(tex, samp, uv, 0.0).rgb;
}

@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
    return vec4(pos[vi], 0.0, 1.0);
}

@fragment fn fs() -> @location(0) vec4<f32> {
    let dir = vec3(0.0, 0.0, 1.0);
    let mip = clamp(roughness_to_mip(p.roughness), -2.0, p.lod_max);
    let mip_f = fract(mip);
    let mip_i = floor(mip);
    let c0 = bilinear_cube_uv(dir, mip_i);
    if (mip_f <= 0.0) { return vec4(c0, 1.0); }
    let c1 = bilinear_cube_uv(dir, mip_i + 1.0);
    return vec4(mix(c0, c1, mip_f), 1.0);
}
"#,
            )),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cube uv sample bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            texel_w: f32,
            texel_h: f32,
            lod_max: f32,
            roughness: f32,
            _pad: [f32; 2],
        }

        let params = Params {
            texel_w: atlas.texel_width,
            texel_h: atlas.texel_height,
            lod_max: atlas.lod_max as f32,
            roughness: 0.45,
            _pad: [0.0; 2],
        };
        let param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cube uv sample bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(uv_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("out"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&Default::default());

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cube uv sample"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        let padded_bpr = 256u32;
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: padded_bpr as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &read_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = read_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().expect("buffer range is mapped");
        let gpu = [
            data[0] as f32 / 255.0,
            data[1] as f32 / 255.0,
            data[2] as f32 / 255.0,
        ];

        eprintln!("cpu sample_cube_uv_env {cpu:?}");
        eprintln!("gpu shader sample {gpu:?}");

        for i in 0..3 {
            assert!(
                (gpu[i] - cpu[i]).abs() < 0.005,
                "channel {i}: cpu={} gpu={}",
                cpu[i],
                gpu[i]
            );
        }
    }
}
