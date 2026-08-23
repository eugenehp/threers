//! Pack probe + G-buffer + reference into the planar layout the network reads.
//!
//! Colour and depth go through [`crate::compress`]. Albedo and normals are
//! already bounded. Training tiles are independent crops; inference uses
//! overlapping regions — see [`crate::infer::ProbeNet::reconstruct`].

use anyhow::{ensure, Result};

use crate::ddgi::{self, DdgiContext};
use crate::{compress, IN_CHANNELS, OUT_CHANNELS};
use crate::model::ALBEDO_FLOOR;

/// Optional world-space extras for DDGI and encoded position planes.
pub struct PlaneExtras<'a> {
    pub world: &'a [[f32; 3]],
}

/// Both frame dimensions must divide by this when building quarter-res probe planes.
pub const PROBE_QUARTER_SCALE: usize = 4;

/// Index of the compressed depth plane in a packed input buffer.
pub const DEPTH_PLANE: usize = 9;

/// Compressed depth at or above this value is treated as a sky / miss (no geometry).
pub const SKY_DEPTH_COMPRESSED: f32 = 0.999;

/// `true` where plane 9 indicates a first-hit on geometry (not sky).
pub fn geometry_hits(input: &[f32], pixels: usize) -> Vec<bool> {
    input[DEPTH_PLANE * pixels..(DEPTH_PLANE + 1) * pixels]
        .iter()
        .map(|&d| d < SKY_DEPTH_COMPRESSED)
        .collect()
}

/// Planar `[IN, H, W]` input and `[OUT, H, W]` target for one frame.
#[allow(clippy::too_many_arguments)]
pub fn planes(
    width: usize,
    height: usize,
    probe_rgba: &[f32],
    albedo: &[[f32; 3]],
    normal: &[[f32; 3]],
    depth: &[f32],
    reference_rgba: &[f32],
    scene_scale: f32,
    extras: Option<PlaneExtras<'_>>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let n = width * height;
    ensure!(probe_rgba.len() == n * 4, "probe is {} floats, expected {}", probe_rgba.len(), n * 4);
    ensure!(reference_rgba.len() == n * 4, "reference is {} floats, expected {}", reference_rgba.len(), n * 4);
    ensure!(albedo.len() == n && normal.len() == n && depth.len() == n, "guide length mismatch");
    ensure!(
        width.is_multiple_of(PROBE_QUARTER_SCALE) && height.is_multiple_of(PROBE_QUARTER_SCALE),
        "{width}x{height} must be a multiple of {PROBE_QUARTER_SCALE} for quarter-res probe planes"
    );
    let scale = scene_scale.max(1e-6);
    let probe_q = probe_quarter(probe_rgba, width, height);
    let (ddgi, world_enc) = match extras {
        Some(ex) => {
            let (min, max) = ddgi::bounds_from_world(ex.world);
            let (center, enc_scale) = ddgi::encode_frame(ex.world);
            let ctx = DdgiContext {
                world: ex.world,
                bounds_min: min,
                bounds_max: max,
            };
            let vis_blend = std::env::var("DDGI_VIS_BLEND")
                .ok()
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
            if vis_blend {
                let (mut colour, vis) =
                    ddgi::ddgi_planes_with_visibility(width, height, probe_rgba, &ctx);
                for i in 0..n {
                    let gate = 0.5 + 0.5 * vis[i];
                    for c in 0..3 {
                        colour[c * n + i] *= gate;
                    }
                }
                (colour, ddgi::encode_world_planes(ex.world, center, enc_scale))
            } else {
                (
                    ddgi::ddgi_planes(width, height, probe_rgba, &ctx),
                    ddgi::encode_world_planes(ex.world, center, enc_scale),
                )
            }
        }
        None => (vec![0.0; OUT_CHANNELS * n], vec![0.0; OUT_CHANNELS * n]),
    };
    let mut input = vec![0.0f32; IN_CHANNELS * n];
    let mut target = vec![0.0f32; OUT_CHANNELS * n];
    for i in 0..n {
        for c in 0..3 {
            let probe_c = probe_rgba[i * 4 + c].max(0.0);
            let alb = albedo[i][c].max(0.0);
            input[c * n + i] = compress(probe_c);
            input[(3 + c) * n + i] = alb;
            input[(6 + c) * n + i] = normal[i][c];
            input[(10 + c) * n + i] = compress(probe_c / alb.max(ALBEDO_FLOOR));
            input[(13 + c) * n + i] = probe_q[c * n + i];
            input[(16 + c) * n + i] = ddgi[c * n + i];
            input[(19 + c) * n + i] = world_enc[c * n + i];
            target[c * n + i] = compress(reference_rgba[i * 4 + c]);
        }
        input[9 * n + i] = compress(depth[i] / scale);
    }
    Ok((input, target))
}

/// Planar `[IN, H, W]` input only — for live inference without a reference.
#[allow(clippy::too_many_arguments)]
pub fn planes_input(
    width: usize,
    height: usize,
    probe_rgba: &[f32],
    albedo: &[[f32; 3]],
    normal: &[[f32; 3]],
    depth: &[f32],
    scene_scale: f32,
    extras: Option<PlaneExtras<'_>>,
) -> Result<Vec<f32>> {
    let n = width * height;
    let dummy = vec![0.0f32; n * 4];
    let (input, _) = planes(
        width,
        height,
        probe_rgba,
        albedo,
        normal,
        depth,
        &dummy,
        scene_scale,
        extras,
    )?;
    Ok(input)
}

/// Box-average probe RGB to ¼ resolution, then nearest-upsample to full size.
fn probe_quarter(probe_rgba: &[f32], width: usize, height: usize) -> Vec<f32> {
    let n = width * height;
    let s = PROBE_QUARTER_SCALE;
    let mut out = vec![0.0f32; OUT_CHANNELS * n];
    for qy in 0..(height / s) {
        for qx in 0..(width / s) {
            for c in 0..3 {
                let mut sum = 0.0f32;
                for dy in 0..s {
                    for dx in 0..s {
                        let y = qy * s + dy;
                        let x = qx * s + dx;
                        sum += probe_rgba[(y * width + x) * 4 + c].max(0.0);
                    }
                }
                let v = compress(sum / (s * s) as f32);
                for dy in 0..s {
                    for dx in 0..s {
                        let y = qy * s + dy;
                        let x = qx * s + dx;
                        out[c * n + y * width + x] = v;
                    }
                }
            }
        }
    }
    out
}

/// First three planes of a planar input — the probe colour.
pub fn colour_planes(input: &[f32], pixels: usize) -> &[f32] {
    &input[..OUT_CHANNELS * pixels]
}

/// Non-overlapping tiles of a packed frame, concatenated for a [`crate::Dataset`].
pub fn tiles(
    input: &[f32],
    target: &[f32],
    width: usize,
    height: usize,
    tile: usize,
) -> Result<Vec<f32>> {
    tiles_with(input, target, IN_CHANNELS, OUT_CHANNELS, width, height, tile)
}

/// Tile arbitrary channel counts (e.g. NRC features).
pub fn tiles_with(
    input: &[f32],
    target: &[f32],
    in_channels: usize,
    out_channels: usize,
    width: usize,
    height: usize,
    tile: usize,
) -> Result<Vec<f32>> {
    ensure!(tile > 0 && width.is_multiple_of(tile) && height.is_multiple_of(tile), "{width}x{height} is not a grid of {tile}");
    let tiles_x = width / tile;
    let tiles_y = height / tile;
    let mut out = Vec::with_capacity(
        tiles_x * tiles_y * (in_channels + out_channels) * tile * tile,
    );
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            out.extend(extract(input, in_channels, height, width, tile, tx, ty));
            out.extend(extract(target, out_channels, height, width, tile, tx, ty));
        }
    }
    Ok(out)
}

/// One tile from a planar `[C, H, W]` buffer on the non-overlapping grid.
pub fn extract(
    src: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    tile: usize,
    tx: usize,
    ty: usize,
) -> Vec<f32> {
    let x0 = tx * tile;
    let y0 = ty * tile;
    extract_region(src, channels, height, width, tile, x0, y0)
}

/// A `tile × tile` window anchored at `(x0, y0)`, clamping at the frame edge.
///
/// Used at inference so a kernel near a tile border can gather from the
/// neighbouring region instead of replicate-pad alone.
pub fn extract_region(
    src: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    tile: usize,
    x0: usize,
    y0: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; channels * tile * tile];
    for c in 0..channels {
        for ty in 0..tile {
            let sy = (y0 + ty).min(height - 1);
            for tx in 0..tile {
                let sx = (x0 + tx).min(width - 1);
                let src_i = c * height * width + sy * width + sx;
                out[c * tile * tile + ty * tile + tx] = src[src_i];
            }
        }
    }
    out
}

/// Write one tile into a planar `[C, H, W]` buffer.
#[allow(clippy::too_many_arguments)]
pub fn stamp(
    dst: &mut [f32],
    channels: usize,
    height: usize,
    width: usize,
    tile: usize,
    tx: usize,
    ty: usize,
    src: &[f32],
) {
    for c in 0..channels {
        for y in 0..tile {
            for x in 0..tile {
                let dst_i = c * height * width + (ty * tile + y) * width + tx * tile + x;
                dst[dst_i] = src[c * tile * tile + y * tile + x];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_hits_distinguish_sky_from_surfaces() {
        let pixels = 2;
        let mut input = vec![0.0f32; IN_CHANNELS * pixels];
        input[9 * pixels] = 0.5;
        input[9 * pixels + 1] = SKY_DEPTH_COMPRESSED;
        let hits = geometry_hits(&input, pixels);
        assert!(hits[0]);
        assert!(!hits[1]);
    }

    #[test]
    fn a_known_pixel_lands_in_the_right_plane() {
        let w = 4;
        let h = 4;
        let n = w * h;
        let mut probe = vec![0.0f32; n * 4];
        for i in 0..n {
            probe[i * 4] = 8.0;
        }
        let albedo = vec![[0.2, 0.3, 0.4]; n];
        let normal = vec![[0.0, 1.0, 0.0]; n];
        let depth = vec![2.0f32; n];
        let reference = vec![1.0f32; n * 4];
        let (input, target) =
            planes(w, h, &probe, &albedo, &normal, &depth, &reference, 2.0, None).unwrap();
        assert!((input[0] - compress(8.0)).abs() < 1e-6);
        assert!((input[3 * n] - 0.2).abs() < 1e-6);
        assert!((input[7 * n] - 1.0).abs() < 1e-6);
        assert!((input[9 * n] - compress(1.0)).abs() < 1e-6);
        assert!((input[10 * n] - compress(8.0 / 0.2)).abs() < 1e-6);
        assert!((input[13 * n] - compress(8.0)).abs() < 1e-6);
        assert!((target[n] - compress(1.0)).abs() < 1e-6);
        assert_eq!(colour_planes(&input, n).len(), OUT_CHANNELS * n);
    }

    #[test]
    fn quarter_res_probe_is_piecewise_constant_over_four_by_four() {
        let w = 8;
        let h = 8;
        let n = w * h;
        let mut probe = vec![0.0f32; n * 4];
        for y in 0..h {
            for x in 0..w {
                let v = (x / 4 + y / 4) as f32;
                let i = (y * w + x) * 4;
                probe[i] = v;
                probe[i + 1] = v * 0.5;
                probe[i + 2] = v * 0.25;
            }
        }
        let albedo = vec![[0.5, 0.5, 0.5]; n];
        let normal = vec![[0.0, 1.0, 0.0]; n];
        let depth = vec![1.0; n];
        let reference = vec![1.0f32; n * 4];
        let (input, _) =
            planes(w, h, &probe, &albedo, &normal, &depth, &reference, 1.0, None).unwrap();
        let block00 = input[13];
        let block01 = input[13 + 4];
        assert!((input[13 + 1] - block00).abs() < 1e-6);
        assert!((input[13 + w] - block00).abs() < 1e-6);
        assert!((block01 - block00).abs() > 1e-6);
    }

    #[test]
    fn tiles_round_trip_through_extract_and_stamp() {
        let w = 8;
        let n = w * w;
        let mut src = vec![0.0f32; 2 * n];
        for (i, slot) in src.iter_mut().enumerate() {
            *slot = i as f32;
        }
        let mut dst = vec![0.0f32; 2 * n];
        let tile = 4;
        for ty in 0..2 {
            for tx in 0..2 {
                let t = extract(&src, 2, w, w, tile, tx, ty);
                stamp(&mut dst, 2, w, w, tile, tx, ty, &t);
            }
        }
        assert_eq!(src, dst);
    }

    #[test]
    fn extract_region_clamps_at_the_frame_edge() {
        let w = 4;
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let patch = extract_region(&src, 1, 1, w, 2, 3, 0);
        assert_eq!(patch, vec![4.0, 4.0, 4.0, 4.0]);
    }
}
