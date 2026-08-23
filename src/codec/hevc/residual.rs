//! HEVC residual (transform-coefficient) CABAC coding — `residual_coding`
//! (H.265 §7.3.8.11, context derivation §9.3.4.2). The most intricate syntax in
//! HEVC: last-significant position, per-sub-block significance, greater-1/2
//! flags, signs, and Golomb-Rice remainders, each with position/neighbor-derived
//! contexts.
//!
//! Verified two ways: the `tests` module round-trips random coefficient blocks through the
//! encoder and a matching decoder (validates the *logic* independent of table
//! values), and the integration test decodes real frames with ffmpeg (validates
//! the *context init values* + conformance).
//!
//! # Known conformance gap (dense 8×8 chroma)
//!
//! Smooth / low-frequency content is externally conformant, but a dense 8×8
//! **chroma** block whose last significant coefficient lands in last-position
//! group 4 (coordinate 4 or 5 in the fast dimension) desyncs a conformant
//! decoder: ffmpeg loses the sub-block-(0,0) DC while decoding the AC correctly.
//! The bug is **symmetric** — the mirrored decoder in the `tests` module round-trips it —
//! so it lives in shared encode/decode logic, not the arithmetic engine (which
//! is byte-exact to §9.3.4.3). Isolation done: luma 16×16 group 4 is fine, only
//! 8×8 chroma fails; the failure is independent of every context init value
//! (brute-forced) and of the last-context `ctxShift`; the last-position
//! binarization matches the ffmpeg source. Not yet root-caused, so `residual`
//! stays off by default. Next step: a bit-exact CABAC trace from a reference
//! decoder (ffmpeg built with tracing, or libde265) to find the first bin the
//! reference reads differently.

use crate::codec::hevc::cabac::{CabacEncoder, CtxModel};

// ---- context init values (I-slice / initType 0, §9.3.2.2) ----
const INIT_LAST_X: [u8; 18] = [
    110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63,
];
const INIT_LAST_Y: [u8; 18] = INIT_LAST_X;
const INIT_CSBF: [u8; 4] = [91, 171, 134, 141];
// sig_coeff_flag init (§9.3.2.2, I-slice): 27 luma contexts (0..=26) then 15
// chroma contexts (27..=41). The chroma DC context is index 27.
const INIT_SIG: [u8; 42] = [
    // luma (ctxInc 0..=26)
    111, 111, 125, 110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107, 125, 141, 179,
    153, 125, 107, 125, 141, 179, 153, 125, //
    // chroma (ctxInc 27..=41)
    140, 139, 182, 182, 152, 136, 152, 136, 153, 136, 139, 111, 136, 139, 111,
];
const INIT_GT1: [u8; 24] = [
    140, 92, 137, 138, 140, 152, 138, 139, 153, 74, 149, 92, 139, 107, 122, 152, 140, 179, 166,
    182, 140, 227, 122, 197,
];
const INIT_GT2: [u8; 6] = [138, 153, 136, 167, 152, 152];

/// The coefficient-coding CABAC contexts for one slice.
pub struct ResidualCtx {
    last_x: Vec<CtxModel>,
    last_y: Vec<CtxModel>,
    csbf: Vec<CtxModel>,
    sig: Vec<CtxModel>,
    gt1: Vec<CtxModel>,
    gt2: Vec<CtxModel>,
}

impl ResidualCtx {
    pub fn new(qp: i32) -> Self {
        let mk = |t: &[u8]| t.iter().map(|&v| CtxModel::init(v, qp)).collect();
        Self {
            last_x: mk(&INIT_LAST_X),
            last_y: mk(&INIT_LAST_Y),
            csbf: mk(&INIT_CSBF),
            sig: mk(&INIT_SIG),
            gt1: mk(&INIT_GT1),
            gt2: mk(&INIT_GT2),
        }
    }
}

// ---- scan orders ----

/// Scan order (`0` diagonal up-right, `1` horizontal, `2` vertical) over a
/// `k × k` grid, as a list of `(x, y)`.
pub fn scan_kxk(k: usize, scan_idx: u8) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(k * k);
    match scan_idx {
        1 => {
            for y in 0..k {
                for x in 0..k {
                    v.push((x, y));
                }
            }
        }
        2 => {
            for x in 0..k {
                for y in 0..k {
                    v.push((x, y));
                }
            }
        }
        _ => {
            for d in 0..(2 * k - 1) {
                for x in 0..k {
                    let y = d as isize - x as isize;
                    if y >= 0 && (y as usize) < k {
                        v.push((x, y as usize));
                    }
                }
            }
        }
    }
    v
}

/// Full coefficient scan of an `n × n` transform block: sub-blocks (4×4) scanned
/// in `scan_idx`, and the 16 positions within each in the same order.
pub fn full_scan(n: usize, scan_idx: u8) -> Vec<(usize, usize)> {
    if n == 4 {
        return scan_kxk(4, scan_idx);
    }
    let sb = scan_kxk(n / 4, scan_idx);
    let inner = scan_kxk(4, scan_idx);
    let mut out = Vec::with_capacity(n * n);
    for &(sx, sy) in &sb {
        for &(px, py) in &inner {
            out.push((sx * 4 + px, sy * 4 + py));
        }
    }
    out
}

#[inline]
fn log2(n: usize) -> usize {
    n.trailing_zeros() as usize
}

/// `last_sig_coeff` group index and group base (§9.3.4.2.3 tables).
const GROUP_IDX: [usize; 16] = [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7];
const MIN_IN_GROUP: [usize; 8] = [0, 1, 2, 3, 4, 6, 8, 12];

fn last_ctx(bin_idx: usize, log2n: usize, chroma: bool) -> usize {
    if chroma {
        15 + (bin_idx >> (log2n - 2))
    } else {
        let offset = 3 * (log2n - 2) + ((log2n - 1) >> 2);
        let shift = (log2n + 1) >> 2;
        (bin_idx >> shift) + offset
    }
}

/// sig_coeff_flag context index (§9.3.4.2.5). `csbf_rb` = right + 2·below CSBF.
fn sig_ctx(
    xc: usize,
    yc: usize,
    log2n: usize,
    chroma: bool,
    sub_nonzero: bool,
    csbf_rb: u8,
) -> usize {
    if log2n == 2 {
        const MAP: [usize; 16] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8];
        let base = MAP[(yc << 2) + xc];
        return if chroma { 27 + base } else { base };
    }
    if xc + yc == 0 {
        return if chroma { 27 } else { 0 };
    }
    let (xp, yp) = (xc & 3, yc & 3);
    let mut s = match csbf_rb {
        0 => {
            if xp + yp == 0 {
                2
            } else if xp + yp < 3 {
                1
            } else {
                0
            }
        }
        1 => {
            if yp == 0 {
                2
            } else if yp == 1 {
                1
            } else {
                0
            }
        }
        2 => {
            if xp == 0 {
                2
            } else if xp == 1 {
                1
            } else {
                0
            }
        }
        _ => 2,
    };
    if !chroma {
        if sub_nonzero {
            s += 3;
        }
        s += if log2n == 3 { 9 } else { 21 };
        s
    } else {
        s += if log2n == 3 { 9 } else { 12 };
        27 + s
    }
}

/// Encode `coeff` (`n × n` quantized levels, row-major) for `residual_coding`.
/// `chroma` selects luma/chroma contexts; `scan_idx` the scan order. The block
/// must contain at least one nonzero level (caller codes `cbf`).
pub fn encode(
    cabac: &mut CabacEncoder,
    ctx: &mut ResidualCtx,
    coeff: &[i32],
    n: usize,
    chroma: bool,
    scan_idx: u8,
) {
    let log2n = log2(n);
    let scan = full_scan(n, scan_idx);
    let inner = scan_kxk(4, scan_idx);
    let sb_per_side = n / 4;

    // Last significant position (in scan order).
    let last_scan = (0..scan.len())
        .rev()
        .find(|&i| coeff[scan[i].1 * n + scan[i].0] != 0)
        .unwrap();
    let (last_x, last_y) = scan[last_scan];

    // --- last_sig_coeff: x_prefix, y_prefix, then x_suffix, y_suffix (§7.3.8.11) ---
    let gx = code_last_prefix(cabac, &mut ctx.last_x, last_x, log2n, chroma);
    let gy = code_last_prefix(cabac, &mut ctx.last_y, last_y, log2n, chroma);
    code_last_suffix(cabac, last_x, gx);
    code_last_suffix(cabac, last_y, gy);

    let last_sb = last_scan / 16;
    let last_pos_in_sb = last_scan % 16;

    // coded_sub_block_flag grid.
    let mut csbf = vec![vec![false; sb_per_side]; sb_per_side];
    let sb_scan = scan_kxk(sb_per_side, scan_idx);

    let mut c1_carry = 1u32; // greater1 state carried across sub-blocks

    for si in (0..=last_sb).rev() {
        let (sxs, sys) = sb_scan[si];
        let mut infer_dc = false;
        let is_first = si == 0;
        let is_last = si == last_sb;
        if !is_first && !is_last {
            let rb = csbf_right_below(&csbf, sxs, sys, sb_per_side);
            let ctx_i = (rb.min(1) as usize) + if chroma { 2 } else { 0 };
            let bit = block_has_sig(coeff, n, sxs, sys, &inner);
            cabac.encode_bin(&mut ctx.csbf[ctx_i], bit as u32);
            csbf[sys][sxs] = bit;
            if !bit {
                continue;
            }
            infer_dc = true;
        } else {
            csbf[sys][sxs] = true;
        }
        let csbf_rb = csbf_right_below(&csbf, sxs, sys, sb_per_side);

        // --- sig_coeff_flag for positions in this sub-block ---
        let start = if is_last {
            last_pos_in_sb as i32 - 1
        } else {
            15
        };
        let mut sig = [false; 16];
        if is_last {
            sig[last_pos_in_sb] = true;
        }
        let sub_nonzero = sxs + sys != 0;
        let mut num_sig = if is_last { 1 } else { 0 };
        for p in (0..=start).rev() {
            let (px, py) = inner[p as usize];
            let (xc, yc) = (sxs * 4 + px, sys * 4 + py);
            if p == 0 && infer_dc && num_sig == 0 {
                // inferred significant (sub-block coded but nothing yet).
                sig[0] = true;
                num_sig += 1;
                break;
            }
            let ci = sig_ctx(xc, yc, log2n, chroma, sub_nonzero, csbf_rb);
            let bit = coeff[yc * n + xc] != 0;
            cabac.encode_bin(&mut ctx.sig[ci], bit as u32);
            if bit {
                sig[p as usize] = true;
                num_sig += 1;
            }
        }
        if num_sig == 0 {
            continue;
        }

        // Significant coeffs in reverse scan order (high → low position).
        let mut abs_levels = Vec::with_capacity(num_sig);
        let mut positions = Vec::with_capacity(num_sig);
        for p in (0..16).rev() {
            if sig[p] {
                let (px, py) = inner[p];
                let (xc, yc) = (sxs * 4 + px, sys * 4 + py);
                abs_levels.push(coeff[yc * n + xc].unsigned_abs());
                positions.push((xc, yc));
            }
        }

        // --- coeff_abs_level_greater1_flag (first 8) ---
        let ctx_set_base = if si > 0 && !chroma { 2 } else { 0 };
        let ctx_set = ctx_set_base + if c1_carry == 0 { 1 } else { 0 };
        let mut c1 = 1u32;
        let mut first_gt1: i32 = -1;
        let n_gt1 = abs_levels.len().min(8);
        for idx in 0..n_gt1 {
            let bin = (abs_levels[idx] > 1) as u32;
            let base = if chroma { 16 } else { 0 };
            let ci = base + (ctx_set << 2) + c1 as usize;
            cabac.encode_bin(&mut ctx.gt1[ci], bin);
            if bin == 1 {
                c1 = 0;
                if first_gt1 < 0 {
                    first_gt1 = idx as i32;
                }
            } else if c1 > 0 && c1 < 3 {
                c1 += 1;
            }
        }
        c1_carry = c1;

        // --- coeff_abs_level_greater2_flag (first greater1 coeff only) ---
        if first_gt1 >= 0 {
            let bin = (abs_levels[first_gt1 as usize] > 2) as u32;
            let base = if chroma { 4 } else { 0 };
            cabac.encode_bin(&mut ctx.gt2[base + ctx_set], bin);
        }

        // --- coeff_sign_flag (bypass) ---
        for &(xc, yc) in &positions {
            let sign = (coeff[yc * n + xc] < 0) as u32;
            cabac.encode_bypass(sign);
        }

        // --- coeff_abs_level_remaining (bypass, Golomb-Rice) ---
        let mut first_coeff2 = 1u32;
        let mut rice = 0u32;
        for (idx, &abs) in abs_levels.iter().enumerate() {
            let base_level = if idx < 8 { 2 + first_coeff2 } else { 1 };
            if abs >= base_level {
                write_remaining(cabac, abs - base_level, rice);
                if abs > 3 * (1 << rice) {
                    rice = (rice + 1).min(4);
                }
            }
            if abs >= 2 {
                first_coeff2 = 0;
            }
        }
    }
}

/// Code the truncated-unary `last_sig_coeff_*_prefix` (each bin context-coded);
/// returns the group index for the suffix step.
fn code_last_prefix(
    cabac: &mut CabacEncoder,
    ctxs: &mut [CtxModel],
    pos: usize,
    log2n: usize,
    chroma: bool,
) -> usize {
    let group = GROUP_IDX[pos];
    let c_max = (log2n << 1) - 1;
    for b in 0..group {
        cabac.encode_bin(&mut ctxs[last_ctx(b, log2n, chroma)], 1);
    }
    if group < c_max {
        cabac.encode_bin(&mut ctxs[last_ctx(group, log2n, chroma)], 0);
    }
    group
}

/// Fixed-length `last_sig_coeff_*_suffix` (bypass) when the group spans a range.
fn code_last_suffix(cabac: &mut CabacEncoder, pos: usize, group: usize) {
    if group > 3 {
        let nbits = (group >> 1) - 1;
        let suffix = (pos - MIN_IN_GROUP[group]) as u32;
        cabac.encode_bypass_bits(suffix, nbits as u32);
    }
}

/// Golomb-Rice / exp-Golomb `coeff_abs_level_remaining` (§9.3.3.9), all bypass.
fn write_remaining(cabac: &mut CabacEncoder, value: u32, rice: u32) {
    let threshold = 3u32 << rice;
    if value < threshold {
        let prefix = value >> rice;
        // `prefix` ones then a 0.
        for _ in 0..prefix {
            cabac.encode_bypass(1);
        }
        cabac.encode_bypass(0);
        if rice > 0 {
            cabac.encode_bypass_bits(value & ((1 << rice) - 1), rice);
        }
    } else {
        let mut v = value - threshold;
        let mut len = rice;
        while v >= (1 << len) {
            v -= 1 << len;
            len += 1;
        }
        // Escape prefix: (3 + len − rice) ones then a 0, then `len` suffix bits.
        let prefix_ones = 3 + len - rice;
        for _ in 0..prefix_ones {
            cabac.encode_bypass(1);
        }
        cabac.encode_bypass(0);
        cabac.encode_bypass_bits(v, len);
    }
}

fn csbf_right_below(csbf: &[Vec<bool>], sx: usize, sy: usize, side: usize) -> u8 {
    let right = if sx + 1 < side {
        csbf[sy][sx + 1] as u8
    } else {
        0
    };
    let below = if sy + 1 < side {
        csbf[sy + 1][sx] as u8
    } else {
        0
    };
    right + 2 * below
}

fn block_has_sig(
    coeff: &[i32],
    n: usize,
    sxs: usize,
    sys: usize,
    inner: &[(usize, usize)],
) -> bool {
    inner
        .iter()
        .any(|&(px, py)| coeff[(sys * 4 + py) * n + (sxs * 4 + px)] != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::hevc::cabac::CtxModel;

    // A CABAC decoder + residual decoder mirror, to round-trip the coefficient
    // coding logic independent of whether the init values match the spec.
    include!("residual_decode_test.rs");
}
