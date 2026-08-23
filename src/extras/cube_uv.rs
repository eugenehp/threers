use crate::textures::CubeUvAtlas;
use std::sync::Arc;

const LOD_MIN: u32 = 4;

/// three.js PMREM slot order (+X,+Y,+Z,-X,-Y,-Z) → `CubeTexture` face index (+X,-X,+Y,-Y,+Z,-Z).
pub const PMREM_SLOT_TO_FACE: [usize; 6] = [0, 2, 4, 1, 3, 5];

/// Mip0 uses `flipEnvMap = -1` for regular cubemaps (swaps ±X in CubeUV layout).
pub const PMREM_SLOT_TO_FACE_MIP0: [usize; 6] = [1, 2, 5, 0, 3, 4];

/// Row within a CubeUV face tile.
///
/// three.js PMREM renders into a WebGL framebuffer (v=0 at bottom). CPU writes must
/// flip `yi` so CubeUV sampling — which applies `v = 1 - uv/h` for top-first storage —
/// reads the same texels. Without this, axis probes (+Z center) still pass but grazing
/// angles sample the wrong face row (scene parity was ~1.3% before the fix).
fn cube_uv_atlas_row(tile_y: u32, face_size: u32, yi: u32) -> u32 {
    tile_y + (face_size - 1 - yi)
}

/// three.js PMREM plane layout: slots 0–2 bottom row, 3–5 top row (GL y-up).
pub fn pmrem_slot_row(pmrem_slot: usize) -> u32 {
    if pmrem_slot >= 3 {
        0
    } else {
        1
    }
}

fn pmrem_slot_to_face(lod: usize) -> [usize; 6] {
    if lod == 0 {
        PMREM_SLOT_TO_FACE_MIP0
    } else {
        PMREM_SLOT_TO_FACE
    }
}

/// Blit six cube faces (CubeTexture order) into one CubeUV atlas LOD tile.
pub fn blit_cube_faces_to_atlas_lod(
    atlas: &mut [u8],
    atlas_w: u32,
    cube_size: u32,
    lod_max: u32,
    lod_out: u32,
    face_size: u32,
    faces: &[Vec<u8>; 6],
) {
    let (x_off, y_off) = atlas_lod_origin(lod_out, face_size, cube_size, lod_max);
    blit_cube_faces_to_tile(
        atlas,
        atlas_w,
        x_off,
        y_off,
        face_size,
        faces,
        lod_out as usize,
    );
}

/// Read six cube faces back out of a CubeUV atlas LOD tile (CubeTexture order).
pub fn extract_cube_faces_from_atlas_lod(
    atlas: &[u8],
    atlas_w: u32,
    cube_size: u32,
    lod_max: u32,
    lod_out: u32,
    face_size: u32,
) -> [Vec<u8>; 6] {
    let (x_off, y_off) = atlas_lod_origin(lod_out, face_size, cube_size, lod_max);
    let slot_to_face = pmrem_slot_to_face(lod_out as usize);
    let mut faces: [Vec<u8>; 6] = Default::default();
    let face_bytes = (face_size * face_size * 4) as usize;
    for face in &mut faces {
        *face = vec![0u8; face_bytes];
    }
    for (pmrem_slot, &cube_face) in slot_to_face.iter().enumerate() {
        let col = (pmrem_slot % 3) as u32;
        let row = pmrem_slot_row(pmrem_slot);
        let src_x = x_off + col * face_size;
        let src_y = y_off + row * face_size;
        for y in 0..face_size {
            for x in 0..face_size {
                let src_off =
                    (((cube_uv_atlas_row(src_y, face_size, y)) * atlas_w + src_x + x) * 4) as usize;
                let dst_off = ((y * face_size + x) * 4) as usize;
                if src_off + 3 < atlas.len() && dst_off + 3 < faces[cube_face].len() {
                    faces[cube_face][dst_off..dst_off + 4]
                        .copy_from_slice(&atlas[src_off..src_off + 4]);
                }
            }
        }
    }
    faces
}

/// Direction for PMREM face `face` at UV in `[0,1]`² — matches three.js `getDirection()`.
pub fn pmrem_get_direction(face: u32, u: f32, v: f32) -> [f32; 3] {
    let su = u * 2.0 - 1.0;
    let sv = v * 2.0 - 1.0;
    let d = match face {
        0 => [1.0, sv, su],
        1 => [-su, 1.0, -sv],
        2 => [-su, sv, 1.0],
        3 => [-1.0, sv, -su],
        4 => [-su, -1.0, sv],
        _ => [su, sv, -1.0],
    };
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-8);
    [d[0] / len, d[1] / len, d[2] / len]
}

/// UV coordinates for pixel `(xi, yi)` including the 1-texel gutter three.js uses.
pub fn pmrem_face_uv(face_size: u32, xi: u32, yi: u32) -> (f32, f32) {
    let texel = 1.0 / (face_size as f32 - 2.0);
    let min = -texel;
    let max = 1.0 + texel;
    let span = max - min;
    let u = min + (xi as f32 + 0.5) / face_size as f32 * span;
    let v = min + (yi as f32 + 0.5) / face_size as f32 * span;
    (u, v)
}

pub fn cube_uv_get_face(dir: [f32; 3]) -> f32 {
    let abs_dir = [dir[0].abs(), dir[1].abs(), dir[2].abs()];
    if abs_dir[0] > abs_dir[2] {
        if abs_dir[0] > abs_dir[1] {
            if dir[0] > 0.0 {
                0.0
            } else {
                3.0
            }
        } else if dir[1] > 0.0 {
            1.0
        } else {
            4.0
        }
    } else if abs_dir[2] > abs_dir[1] {
        if dir[2] > 0.0 {
            2.0
        } else {
            5.0
        }
    } else if dir[1] > 0.0 {
        1.0
    } else {
        4.0
    }
}

/// Bilinear sample from a PMREM-order six-face cube (`faces[0]` = +X … `faces[5]` = -Z).
pub fn sample_pmrem_faces(
    faces: [&[u8]; 6],
    face_size: u32,
    linear: bool,
    dir: [f32; 3],
) -> [f32; 3] {
    let face = cube_uv_get_face(dir) as usize;
    let uv = cube_uv_get_uv_cpu(dir, face as f32);
    sample_face_bilinear(faces[face], face_size, uv[0], uv[1], linear)
}

fn sample_face_bilinear(data: &[u8], size: u32, u: f32, v: f32, linear: bool) -> [f32; 3] {
    if size <= 1 {
        return if data.len() >= 3 {
            decode_rgb(data, 0, linear)
        } else {
            [0.0; 3]
        };
    }
    let max_coord = size as f32 - 1.001;
    let fx = (u * size as f32).clamp(0.0, max_coord);
    let fy = (v * size as f32).clamp(0.0, max_coord);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(size as usize - 1);
    let y1 = (y0 + 1).min(size as usize - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let mut out = [0.0f32; 3];
    for (c, channel) in out.iter_mut().enumerate() {
        let off = |x: usize, y: usize| (y * size as usize + x) * 4 + c;
        let c00 = decode_px(data, off(x0, y0), linear);
        let c10 = decode_px(data, off(x1, y0), linear);
        let c01 = decode_px(data, off(x0, y1), linear);
        let c11 = decode_px(data, off(x1, y1), linear);
        let v0 = c00 * (1.0 - tx) + c10 * tx;
        let v1 = c01 * (1.0 - tx) + c11 * tx;
        *channel = v0 * (1.0 - ty) + v1 * ty;
    }
    out
}

fn decode_px(data: &[u8], off: usize, linear: bool) -> f32 {
    if off >= data.len() {
        return 0.0;
    }
    let s = data[off] as f32 / 255.0;
    if linear {
        s
    } else {
        srgb_to_linear(s)
    }
}

fn decode_rgb(data: &[u8], off: usize, linear: bool) -> [f32; 3] {
    [
        decode_px(data, off, linear),
        decode_px(data, off + 1, linear),
        decode_px(data, off + 2, linear),
    ]
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Pack six-face cubemap LOD levels into a CubeUV 2D atlas.
/// Input `levels[*]` faces use CubeTexture order; atlas slots use PMREM order.
pub fn pack_cube_uv_atlas(
    levels: &[[Vec<u8>; 6]],
    size_lods: &[u32],
    cube_size: u32,
    lod_max: u32,
) -> CubeUvAtlas {
    let width = 3 * cube_size.max(16 * 7);
    let height = 4 * cube_size;
    let mut pixels = vec![0u8; (width * height * 4) as usize];

    for (lod_out, (faces, face_size)) in levels.iter().zip(size_lods.iter()).enumerate() {
        let output_size = *face_size;
        let x_off = 3 * output_size * (lod_out as u32).saturating_sub(lod_max - LOD_MIN);
        let y_bottom = 4 * (cube_size - output_size);
        let y_off = height - y_bottom - 2 * output_size;
        blit_cube_faces_to_tile(
            &mut pixels,
            width,
            x_off,
            y_off,
            output_size,
            faces,
            lod_out,
        );
    }

    let texel_width = 1.0 / width as f32;
    let texel_height = 1.0 / height as f32;

    CubeUvAtlas {
        width,
        height,
        cube_size,
        lod_max,
        texel_width,
        texel_height,
        pixels: Arc::new(pixels),
        pixels_f32: None,
    }
}

fn blit_cube_faces_to_tile(
    atlas: &mut [u8],
    atlas_w: u32,
    tile_x: u32,
    tile_y: u32,
    face_size: u32,
    faces: &[Vec<u8>; 6],
    lod: usize,
) {
    let slot_to_face = pmrem_slot_to_face(lod);
    for pmrem_slot in 0..6usize {
        let data = &faces[slot_to_face[pmrem_slot]];
        let col = (pmrem_slot % 3) as u32;
        let row = pmrem_slot_row(pmrem_slot);
        let dst_x = tile_x + col * face_size;
        let dst_y = tile_y + row * face_size;
        for y in 0..face_size {
            for x in 0..face_size {
                let src_off = ((y * face_size + x) * 4) as usize;
                if src_off + 3 >= data.len() {
                    continue;
                }
                let dst_off =
                    (((cube_uv_atlas_row(dst_y, face_size, y)) * atlas_w + dst_x + x) * 4) as usize;
                if dst_off + 3 >= atlas.len() {
                    continue;
                }
                atlas[dst_off..dst_off + 4].copy_from_slice(&data[src_off..src_off + 4]);
            }
        }
        if face_size >= 2 {
            copy_edge_gutter(atlas, atlas_w, dst_x, dst_y, face_size);
        }
    }
}

fn copy_edge_gutter(atlas: &mut [u8], atlas_w: u32, x0: u32, y0: u32, s: u32) {
    let copy_px = |atlas: &mut [u8], dx: u32, dy: u32, sx: u32, sy: u32| {
        let d = ((dy * atlas_w + dx) * 4) as usize;
        let src = ((sy * atlas_w + sx) * 4) as usize;
        if d + 3 < atlas.len() && src + 3 < atlas.len() && d != src {
            let px = [atlas[src], atlas[src + 1], atlas[src + 2], atlas[src + 3]];
            atlas[d..d + 4].copy_from_slice(&px);
        }
    };
    for x in 0..s {
        copy_px(atlas, x0 + x, y0, x0 + x, y0 + 1);
        copy_px(atlas, x0 + x, y0 + s - 1, x0 + x, y0 + s - 2);
    }
    for y in 0..s {
        copy_px(atlas, x0, y0 + y, x0 + 1, y0 + y);
        copy_px(atlas, x0 + s - 1, y0 + y, x0 + s - 2, y0 + y);
    }
}

fn sample_atlas_pixel(pixels: &[u8], width: u32, height: u32, u: f32, v: f32) -> [f32; 3] {
    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    let w = width as usize;
    let h = height as usize;
    let x = u * (width - 1) as f32;
    let y = v * (height - 1) as f32;
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let mut out = [0.0f32; 3];
    for (c, channel) in out.iter_mut().enumerate() {
        let c00 = pixels[(y0 * w + x0) * 4 + c] as f32 / 255.0;
        let c10 = pixels[(y0 * w + x1) * 4 + c] as f32 / 255.0;
        let c01 = pixels[(y1 * w + x0) * 4 + c] as f32 / 255.0;
        let c11 = pixels[(y1 * w + x1) * 4 + c] as f32 / 255.0;
        let v0 = c00 * (1.0 - tx) + c10 * tx;
        let v1 = c01 * (1.0 - tx) + c11 * tx;
        *channel = v0 * (1.0 - ty) + v1 * ty;
    }
    out
}

pub fn atlas_lod_origin(
    lod_out: u32,
    output_size: u32,
    cube_size: u32,
    lod_max: u32,
) -> (u32, u32) {
    let height = 4 * cube_size;
    let x_off = 3 * output_size * lod_out.saturating_sub(lod_max - LOD_MIN);
    let y_bottom = 4 * (cube_size - output_size);
    let y_off = height - y_bottom - 2 * output_size;
    (x_off, y_off)
}

fn sample_atlas_pixel_f32(pixels: &[f32], width: u32, height: u32, u: f32, v: f32) -> [f32; 3] {
    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    let w = width as usize;
    let h = height as usize;
    let x = u * (width - 1) as f32;
    let y = v * (height - 1) as f32;
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let mut out = [0.0f32; 3];
    for (c, channel) in out.iter_mut().enumerate() {
        let off = |x: usize, y: usize| (y * w + x) * 4 + c;
        let c00 = pixels[off(x0, y0)];
        let c10 = pixels[off(x1, y0)];
        let c01 = pixels[off(x0, y1)];
        let c11 = pixels[off(x1, y1)];
        let v0 = c00 * (1.0 - tx) + c10 * tx;
        let v1 = c01 * (1.0 - tx) + c11 * tx;
        *channel = v0 * (1.0 - ty) + v1 * ty;
    }
    out
}

/// Bilinear CubeUV sample from a linear f32 RGBA atlas (PMREM blur uses float like three.js HalfFloat).
pub fn sample_atlas_pixels_bilinear_f32(
    pixels: &[f32],
    width: u32,
    height: u32,
    lod_max: u32,
    direction: [f32; 3],
    mip_int: f32,
) -> [f32; 3] {
    const CUBEUV_MIN_MIP_LEVEL: f32 = 4.0;
    const CUBEUV_MIN_TILE_SIZE: f32 = 16.0;

    let dir = direction;
    let mut face = cube_uv_get_face(dir);
    let mut uv = cube_uv_get_uv_cpu(dir, face);
    let filter_int = (CUBEUV_MIN_MIP_LEVEL - mip_int).max(0.0);
    let mip_i = mip_int.max(CUBEUV_MIN_MIP_LEVEL);
    let face_size = 2f32.powf(mip_i);
    uv[0] = uv[0] * (face_size - 2.0) + 1.0;
    uv[1] = uv[1] * (face_size - 2.0) + 1.0;
    if face > 2.0 {
        uv[1] += face_size;
        face -= 3.0;
    }
    uv[0] += face * face_size;
    uv[0] += filter_int * 3.0 * CUBEUV_MIN_TILE_SIZE;
    uv[1] += 4.0 * (2f32.powf(lod_max as f32) - face_size);

    let texel_w = 1.0 / width as f32;
    let texel_h = 1.0 / height as f32;
    let u = uv[0] * texel_w;
    let v = 1.0 - uv[1] * texel_h;
    sample_atlas_pixel_f32(pixels, width, height, u, v)
}

/// Blit six cube faces into a linear f32 CubeUV atlas LOD tile.
pub fn blit_cube_faces_to_atlas_lod_f32(
    atlas: &mut [f32],
    atlas_w: u32,
    cube_size: u32,
    lod_max: u32,
    lod_out: u32,
    face_size: u32,
    faces: &[Vec<u8>; 6],
) {
    let (x_off, y_off) = atlas_lod_origin(lod_out, face_size, cube_size, lod_max);
    let slot_to_face = pmrem_slot_to_face(lod_out as usize);
    for pmrem_slot in 0..6usize {
        let data = &faces[slot_to_face[pmrem_slot]];
        let col = (pmrem_slot % 3) as u32;
        let row = pmrem_slot_row(pmrem_slot);
        let dst_x = x_off + col * face_size;
        let dst_y = y_off + row * face_size;
        for y in 0..face_size {
            for x in 0..face_size {
                let src_off = ((y * face_size + x) * 4) as usize;
                if src_off + 3 >= data.len() {
                    continue;
                }
                let dst_off =
                    (((cube_uv_atlas_row(dst_y, face_size, y)) * atlas_w + dst_x + x) * 4) as usize;
                if dst_off + 3 >= atlas.len() {
                    continue;
                }
                for c in 0..4 {
                    atlas[dst_off + c] = data[src_off + c] as f32 / 255.0;
                }
            }
        }
        if face_size >= 2 {
            copy_edge_gutter_f32(atlas, atlas_w, dst_x, dst_y, face_size);
        }
    }
}

fn copy_edge_gutter_f32(atlas: &mut [f32], atlas_w: u32, x0: u32, y0: u32, s: u32) {
    let copy_px = |atlas: &mut [f32], dx: u32, dy: u32, sx: u32, sy: u32| {
        let d = ((dy * atlas_w + dx) * 4) as usize;
        let src = ((sy * atlas_w + sx) * 4) as usize;
        if d + 3 < atlas.len() && src + 3 < atlas.len() && d != src {
            for c in 0..4 {
                atlas[d + c] = atlas[src + c];
            }
        }
    };
    for x in 0..s {
        copy_px(atlas, x0 + x, y0, x0 + x, y0 + 1);
        copy_px(atlas, x0 + x, y0 + s - 1, x0 + x, y0 + s - 2);
    }
    for y in 0..s {
        copy_px(atlas, x0, y0 + y, x0 + 1, y0 + y);
        copy_px(atlas, x0 + s - 1, y0 + y, x0 + s - 2, y0 + y);
    }
}

/// CPU port of WGSL `bilinear_cube_uv` / three.js `bilinearCubeUV`.
pub fn sample_atlas_pixels_bilinear(
    pixels: &[u8],
    width: u32,
    height: u32,
    lod_max: u32,
    direction: [f32; 3],
    mip_int: f32,
) -> [f32; 3] {
    const CUBEUV_MIN_MIP_LEVEL: f32 = 4.0;
    const CUBEUV_MIN_TILE_SIZE: f32 = 16.0;

    let dir = direction;
    let mut face = cube_uv_get_face(dir);
    let mut uv = cube_uv_get_uv_cpu(dir, face);
    let filter_int = (CUBEUV_MIN_MIP_LEVEL - mip_int).max(0.0);
    let mip_i = mip_int.max(CUBEUV_MIN_MIP_LEVEL);
    let face_size = 2f32.powf(mip_i);
    uv[0] = uv[0] * (face_size - 2.0) + 1.0;
    uv[1] = uv[1] * (face_size - 2.0) + 1.0;
    if face > 2.0 {
        uv[1] += face_size;
        face -= 3.0;
    }
    uv[0] += face * face_size;
    uv[0] += filter_int * 3.0 * CUBEUV_MIN_TILE_SIZE;
    uv[1] += 4.0 * (2f32.powf(lod_max as f32) - face_size);

    let texel_w = 1.0 / width as f32;
    let texel_h = 1.0 / height as f32;
    let u = uv[0] * texel_w;
    let v = 1.0 - uv[1] * texel_h;
    sample_atlas_pixel(pixels, width, height, u, v)
}

/// Normalized atlas UV (OpenGL-style) for CubeUV bilinear sampling.
pub fn atlas_bilinear_uv(
    width: u32,
    height: u32,
    lod_max: u32,
    direction: [f32; 3],
    mip_int: f32,
) -> (f32, f32) {
    const CUBEUV_MIN_MIP_LEVEL: f32 = 4.0;
    const CUBEUV_MIN_TILE_SIZE: f32 = 16.0;

    let dir = direction;
    let mut face = cube_uv_get_face(dir);
    let mut uv = cube_uv_get_uv_cpu(dir, face);
    let filter_int = (CUBEUV_MIN_MIP_LEVEL - mip_int).max(0.0);
    let mip_i = mip_int.max(CUBEUV_MIN_MIP_LEVEL);
    let face_size = 2f32.powf(mip_i);
    uv[0] = uv[0] * (face_size - 2.0) + 1.0;
    uv[1] = uv[1] * (face_size - 2.0) + 1.0;
    if face > 2.0 {
        uv[1] += face_size;
        face -= 3.0;
    }
    uv[0] += face * face_size;
    uv[0] += filter_int * 3.0 * CUBEUV_MIN_TILE_SIZE;
    uv[1] += 4.0 * (2f32.powf(lod_max as f32) - face_size);
    let u = uv[0] / width as f32;
    let v = 1.0 - uv[1] / height as f32;
    (u.clamp(0.0, 1.0), v.clamp(0.0, 1.0))
}

/// Read one atlas texel (no filtering) from CPU pixel bytes.
pub fn read_atlas_texel(pixels: &[u8], width: u32, height: u32, u: f32, v: f32) -> [f32; 3] {
    let x = (u * (width - 1) as f32).round() as u32;
    let y = (v * (height - 1) as f32).round() as u32;
    let i = ((y * width + x) * 4) as usize;
    [
        pixels[i] as f32 / 255.0,
        pixels[i + 1] as f32 / 255.0,
        pixels[i + 2] as f32 / 255.0,
    ]
}

/// three.js / WGSL `roughnessToMip` for CubeUV env maps.
pub fn roughness_to_mip_cube_uv(roughness: f32) -> f32 {
    if roughness >= 0.8 {
        (1.0 - roughness) * (-1.0 - (-2.0)) / (1.0 - 0.8) + (-2.0)
    } else if roughness >= 0.4 {
        (0.8 - roughness) * (2.0 - (-1.0)) / (0.8 - 0.4) + (-1.0)
    } else if roughness >= 0.305 {
        (0.4 - roughness) * (3.0 - 2.0) / (0.4 - 0.305) + 2.0
    } else if roughness >= 0.21 {
        (0.305 - roughness) * (4.0 - 3.0) / (0.305 - 0.21) + 3.0
    } else {
        -2.0 * (1.16 * roughness.max(0.001)).log2()
    }
}

/// CPU port of WGSL `texture_cube_uv` / three.js `textureCubeUV`.
pub fn sample_cube_uv_env(atlas: &CubeUvAtlas, direction: [f32; 3], roughness: f32) -> [f32; 3] {
    let mip = roughness_to_mip_cube_uv(roughness).clamp(-2.0, atlas.lod_max as f32);
    let mip_i = mip.floor();
    let mip_f = mip - mip_i;
    let c0 = sample_atlas_bilinear(atlas, direction, mip_i);
    if mip_f <= 0.0 {
        return c0;
    }
    let c1 = sample_atlas_bilinear(atlas, direction, mip_i + 1.0);
    [
        c0[0] * (1.0 - mip_f) + c1[0] * mip_f,
        c0[1] * (1.0 - mip_f) + c1[1] * mip_f,
        c0[2] * (1.0 - mip_f) + c1[2] * mip_f,
    ]
}

/// CPU port of the WGSL `bilinear_cube_uv` path for unit tests.
pub fn sample_atlas_bilinear(atlas: &CubeUvAtlas, direction: [f32; 3], mip_int: f32) -> [f32; 3] {
    if let Some(f32_px) = &atlas.pixels_f32 {
        return sample_atlas_pixels_bilinear_f32(
            f32_px,
            atlas.width,
            atlas.height,
            atlas.lod_max,
            direction,
            mip_int,
        );
    }
    sample_atlas_pixels_bilinear(
        &atlas.pixels,
        atlas.width,
        atlas.height,
        atlas.lod_max,
        direction,
        mip_int,
    )
}

fn cube_uv_get_uv_cpu(direction: [f32; 3], face: f32) -> [f32; 2] {
    let uv = if face == 0.0 {
        [
            direction[2] / direction[0].abs(),
            direction[1] / direction[0].abs(),
        ]
    } else if face == 1.0 {
        [
            -direction[0] / direction[1].abs(),
            -direction[2] / direction[1].abs(),
        ]
    } else if face == 2.0 {
        [
            -direction[0] / direction[2].abs(),
            direction[1] / direction[2].abs(),
        ]
    } else if face == 3.0 {
        [
            -direction[2] / direction[0].abs(),
            direction[1] / direction[0].abs(),
        ]
    } else if face == 4.0 {
        [
            -direction[0] / direction[1].abs(),
            direction[2] / direction[1].abs(),
        ]
    } else {
        [
            direction[0] / direction[2].abs(),
            direction[1] / direction[2].abs(),
        ]
    };
    [0.5 * (uv[0] + 1.0), 0.5 * (uv[1] + 1.0)]
}
