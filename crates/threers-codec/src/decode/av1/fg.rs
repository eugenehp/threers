//! Film grain synthesis on output pictures (dav1d `src/fg_apply_tmpl.c`,
//! `src/filmgrain_tmpl.c`).

use super::obu::{FilmGrainData, Layout, SeqHeader};
use super::picture::Picture;
use super::tables::GAUSSIAN_SEQUENCE;

const GW: usize = 82;
const GH: usize = 73;
const FG: usize = 32;
const SUB_GW: usize = 44;
const SUB_GH: usize = 38;

type Lut = Vec<[i32; GW]>;

fn get_random_number(bits: u32, state: &mut u32) -> i32 {
    let r = *state;
    let bit = ((r) ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
    *state = (r >> 1) | (bit << 15);
    ((*state >> (16 - bits)) & ((1 << bits) - 1)) as i32
}

#[inline]
fn round2(x: i32, shift: u64) -> i32 {
    (x + ((1i32 << shift) >> 1)) >> shift
}

fn generate_grain_y(d: &FilmGrainData, bd8: i32) -> Lut {
    let mut buf = vec![[0i32; GW]; GH + 1];
    let mut seed = d.seed;
    let shift = (4 - bd8 + d.grain_scale_shift) as u64;
    let ctr = 128 << bd8;
    for row in buf.iter_mut().take(GH) {
        for v in row.iter_mut() {
            *v = round2(GAUSSIAN_SEQUENCE[get_random_number(11, &mut seed) as usize] as i32, shift);
        }
    }
    let lag = d.ar_coeff_lag as isize;
    for y in 3..GH {
        for x in 3..GW - 3 {
            let mut k = 0;
            let mut sum = 0;
            'o: for dy in -lag..=0 {
                for dx in -lag..=lag {
                    if dx == 0 && dy == 0 {
                        break 'o;
                    }
                    sum += d.ar_coeffs_y[k] as i32 * buf[(y as isize + dy) as usize][(x as isize + dx) as usize];
                    k += 1;
                }
            }
            buf[y][x] = (buf[y][x] + round2(sum, d.ar_coeff_shift)).clamp(-ctr, ctr - 1);
        }
    }
    buf
}

fn generate_grain_uv(d: &FilmGrainData, buf_y: &Lut, uv: usize, subx: usize, suby: usize, bd8: i32) -> Lut {
    let mut buf = vec![[0i32; GW]; GH + 1];
    let mut seed = d.seed ^ if uv != 0 { 0x49d8 } else { 0xb524 };
    let shift = (4 - bd8 + d.grain_scale_shift) as u64;
    let ctr = 128 << bd8;
    let cw = if subx != 0 { SUB_GW } else { GW };
    let ch = if suby != 0 { SUB_GH } else { GH };
    for row in buf.iter_mut().take(ch) {
        for v in row.iter_mut().take(cw) {
            *v = round2(GAUSSIAN_SEQUENCE[get_random_number(11, &mut seed) as usize] as i32, shift);
        }
    }
    let lag = d.ar_coeff_lag as isize;
    for y in 3..ch {
        for x in 3..cw - 3 {
            let coeff = &d.ar_coeffs_uv[uv];
            let mut k = 0;
            let mut sum = 0;
            'o: for dy in -lag..=0 {
                for dx in -lag..=lag {
                    if dx == 0 && dy == 0 {
                        if d.num_y_points == 0 {
                            break 'o;
                        }
                        let mut luma = 0;
                        let lx = ((x - 3) << subx) + 3;
                        let ly = ((y - 3) << suby) + 3;
                        for i in 0..=suby {
                            for j in 0..=subx {
                                luma += buf_y[ly + i][lx + j];
                            }
                        }
                        luma = round2(luma, (subx + suby) as u64);
                        sum += luma * coeff[k] as i32;
                        break 'o;
                    }
                    sum += coeff[k] as i32 * buf[(y as isize + dy) as usize][(x as isize + dx) as usize];
                    k += 1;
                }
            }
            buf[y][x] = (buf[y][x] + round2(sum, d.ar_coeff_shift)).clamp(-ctr, ctr - 1);
        }
    }
    buf
}

fn generate_scaling(bitdepth: u32, points: &[[u8; 2]], num: usize) -> Vec<u8> {
    let shift_x = if bitdepth == 8 { 0 } else { bitdepth - 8 } as usize;
    let size = if bitdepth == 8 { 256 } else { 1usize << bitdepth };
    let mut s = vec![0u8; size];
    if num == 0 {
        return s;
    }
    let first = (points[0][0] as usize) << shift_x;
    s[..first].fill(points[0][1]);
    for i in 0..num - 1 {
        let (bx, by) = (points[i][0] as i32, points[i][1] as i32);
        let (ex, ey) = (points[i + 1][0] as i32, points[i + 1][1] as i32);
        let dx = ex - bx;
        let dy = ey - by;
        let delta = dy * ((0x10000 + (dx >> 1)) / dx);
        let mut dd = 0x8000;
        for x in 0..dx {
            s[((bx + x) as usize) << shift_x] = (by + (dd >> 16)) as u8;
            dd += delta;
        }
    }
    let n = (points[num - 1][0] as usize) << shift_x;
    s[n..].fill(points[num - 1][1]);
    if bitdepth != 8 {
        let pad = 1usize << shift_x;
        let rnd = (pad >> 1) as i32;
        for i in 0..num - 1 {
            let bx = (points[i][0] as usize) << shift_x;
            let ex = (points[i + 1][0] as usize) << shift_x;
            let dx = ex - bx;
            let mut x = 0;
            while x < dx {
                let range = s[bx + x + pad] as i32 - s[bx + x] as i32;
                let mut r = rnd;
                for n in 1..pad {
                    r += range;
                    s[bx + x + n] = (s[bx + x] as i32 + (r >> shift_x)) as u8;
                }
                x += pad;
            }
        }
    }
    s
}

#[inline]
fn sample_lut(lut: &Lut, offsets: &[[i32; 2]; 2], subx: usize, suby: usize, bx: usize, by: usize, x: usize, y: usize) -> i32 {
    let r = offsets[bx][by];
    let offx = 3 + (2 >> subx) * (3 + (r >> 4)) as usize;
    let offy = 3 + (2 >> suby) * (3 + (r & 0xf)) as usize;
    lut[offy + y + (FG >> suby) * by][offx + x + (FG >> subx) * bx]
}

/// Grain for one plane and one 32-row band (`fgy_32x32xn_c` /
/// `fguv_32x32xn_c`); `luma` is the pre-grain luma for chroma scaling.
#[allow(clippy::too_many_arguments)]
fn apply_band(out: &mut Picture, inp: &Picture, pl: usize, d: &FilmGrainData, scaling: &[u8], lut: &Lut, row: usize, is_id: bool, bitdepth_max: i32) {
    let bd8 = (32 - (bitdepth_max as u32).leading_zeros()) as i32 - 8;
    let (sx, sy) = if pl == 0 { (0usize, 0usize) } else { let (a, b) = inp.ss(1); (a as usize, b as usize) };
    let pw = ((inp.w as usize) + sx) >> sx;
    let lw = inp.w as usize;
    let bh = ((FG.min(inp.h as usize - row * FG)) + sy) >> sy;
    let rows = 1 + (d.overlap_flag && row > 0) as usize;
    let ctr = 128 << bd8;
    let (gmin, gmax) = (-ctr, ctr - 1);
    let (minv, maxv) = if d.clip_to_restricted_range {
        (16 << bd8, (if pl == 0 || is_id { 235 } else { 240 }) << bd8)
    } else {
        (0, bitdepth_max)
    };
    let mut seed = [0u32; 2];
    for (i, s) in seed.iter_mut().enumerate().take(rows) {
        let r = row as i32 - i as i32;
        *s = d.seed ^ ((((r * 37 + 178) & 0xff) as u32) << 8) ^ (((r * 173 + 105) & 0xff) as u32);
    }
    let stride = inp.pstride(pl);
    let ys = inp.stride[0];
    let y0 = row * (FG >> sy);
    let luma_y0 = row * FG;
    let mut offsets = [[0i32; 2]; 2];
    let w: [[[i32; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0, 0]]];
    let mut bx = 0usize;
    while bx < pw {
        let bw = (FG >> sx).min(pw - bx);
        if d.overlap_flag && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = get_random_number(8, &mut seed[i]);
        }
        let ystart = if d.overlap_flag && row != 0 { (2 >> sy).min(bh) } else { 0 };
        let xstart = if d.overlap_flag && bx != 0 { (2 >> sx).min(bw) } else { 0 };
        let mut add = |x: usize, y: usize, grain: i32| {
            let src = inp.planes[pl][(y0 + y) * stride + bx + x] as i32;
            let val = if pl == 0 {
                src
            } else {
                let lx = (bx + x) << sx;
                let lrow = (luma_y0 + (y << sy)) * ys;
                let mut avg = inp.planes[0][lrow + lx.min(lw - 1)] as i32;
                if sx != 0 {
                    avg = (avg + inp.planes[0][lrow + (lx + 1).min(lw - 1)] as i32 + 1) >> 1;
                }
                if d.chroma_scaling_from_luma {
                    avg
                } else {
                    let uv = pl - 1;
                    let combined = avg * d.uv_luma_mult[uv] + src * d.uv_mult[uv];
                    ((combined >> 6) + d.uv_offset[uv] * (1 << bd8)).clamp(0, bitdepth_max)
                }
            };
            let noise = round2(scaling[val as usize] as i32 * grain, d.scaling_shift as u64);
            out.planes[pl][(y0 + y) * stride + bx + x] = (src + noise).clamp(minv, maxv) as u16;
        };
        let wx = |x: usize| w[sx][x];
        let wy = |y: usize| w[sy][y];
        let blend = |old: i32, cur: i32, ww: [i32; 2]| round2(old * ww[0] + cur * ww[1], 5).clamp(gmin, gmax);
        for y in ystart..bh {
            for x in xstart..bw {
                add(x, y, sample_lut(lut, &offsets, sx, sy, 0, 0, x, y));
            }
            for x in 0..xstart {
                let g = sample_lut(lut, &offsets, sx, sy, 0, 0, x, y);
                let old = sample_lut(lut, &offsets, sx, sy, 1, 0, x, y);
                add(x, y, blend(old, g, wx(x)));
            }
        }
        for y in 0..ystart {
            for x in xstart..bw {
                let g = sample_lut(lut, &offsets, sx, sy, 0, 0, x, y);
                let old = sample_lut(lut, &offsets, sx, sy, 0, 1, x, y);
                add(x, y, blend(old, g, wy(y)));
            }
            for x in 0..xstart {
                let top = sample_lut(lut, &offsets, sx, sy, 0, 1, x, y);
                let old = sample_lut(lut, &offsets, sx, sy, 1, 1, x, y);
                let top = blend(old, top, wx(x));
                let g = sample_lut(lut, &offsets, sx, sy, 0, 0, x, y);
                let old = sample_lut(lut, &offsets, sx, sy, 1, 0, x, y);
                let g = blend(old, g, wx(x));
                add(x, y, blend(top, g, wy(y)));
            }
        }
        bx += FG >> sx;
    }
}

/// `dav1d_apply_grain`.
pub fn apply_grain(inp: &Picture, d: &FilmGrainData, seq: &SeqHeader) -> Picture {
    let mut out = inp.clone();
    let bitdepth_max = (1i32 << inp.bpc) - 1;
    let bd8 = inp.bpc as i32 - 8;
    let lut_y = generate_grain_y(d, bd8);
    let (sx, sy) = if inp.layout == Layout::I400 { (0, 0) } else { let (a, b) = inp.ss(1); (a as usize, b as usize) };
    let lut_u = if d.num_uv_points[0] != 0 || d.chroma_scaling_from_luma { Some(generate_grain_uv(d, &lut_y, 0, sx, sy, bd8)) } else { None };
    let lut_v = if d.num_uv_points[1] != 0 || d.chroma_scaling_from_luma { Some(generate_grain_uv(d, &lut_y, 1, sx, sy, bd8)) } else { None };
    let sc_y = generate_scaling(inp.bpc, &d.y_points, d.num_y_points as usize);
    let sc_u = generate_scaling(inp.bpc, &d.uv_points[0], d.num_uv_points[0] as usize);
    let sc_v = generate_scaling(inp.bpc, &d.uv_points[1], d.num_uv_points[1] as usize);
    let is_id = seq.mtrx == 0;
    let rows = (inp.h as usize).div_ceil(FG);
    for row in 0..rows {
        if d.num_y_points != 0 {
            apply_band(&mut out, inp, 0, d, &sc_y, &lut_y, row, is_id, bitdepth_max);
        }
        if inp.layout == Layout::I400 {
            continue;
        }
        if d.chroma_scaling_from_luma {
            apply_band(&mut out, inp, 1, d, &sc_y, lut_u.as_ref().unwrap(), row, is_id, bitdepth_max);
            apply_band(&mut out, inp, 2, d, &sc_y, lut_v.as_ref().unwrap(), row, is_id, bitdepth_max);
        } else {
            if d.num_uv_points[0] != 0 {
                apply_band(&mut out, inp, 1, d, &sc_u, lut_u.as_ref().unwrap(), row, is_id, bitdepth_max);
            }
            if d.num_uv_points[1] != 0 {
                apply_band(&mut out, inp, 2, d, &sc_v, lut_v.as_ref().unwrap(), row, is_id, bitdepth_max);
            }
        }
    }
    out
}
