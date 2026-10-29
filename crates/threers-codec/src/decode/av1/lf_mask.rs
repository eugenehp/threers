//! Loop-filter edge masks and levels (dav1d `src/lf_mask.c`).

use super::obu::{FrameHeader, Layout, LoopfilterModeRefDeltas};
use super::tables::{BLOCK_DIMENSIONS, TXFM_DIMENSIONS};

#[derive(Clone, Copy)]
pub struct Av1FilterLut {
    pub e: [u8; 64],
    pub i: [u8; 64],
    pub sharp: [u64; 2],
}

impl Default for Av1FilterLut {
    fn default() -> Self {
        Av1FilterLut { e: [0; 64], i: [0; 64], sharp: [0; 2] }
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct RestorationUnit {
    /// SGR: `RESTORATION_SGRPROJ + sgr_idx`.
    pub typ: u8,
    pub filter_h: [i8; 3],
    pub filter_v: [i8; 3],
    pub sgr_weights: [i8; 2],
}

/// One 128x128 area (1 or 4 superblocks), pre-superres.
#[derive(Clone, Copy)]
#[derive(Default)]
pub struct Av1Filter {
    /// `[col/row][pos][tx class][half]`, each bit one 4px line.
    pub filter_y: [[[[u16; 2]; 3]; 32]; 2],
    pub filter_uv: [[[[u16; 2]; 2]; 32]; 2],
    pub cdef_idx: [i8; 4],
    pub noskip_mask: [[u16; 2]; 16],
}


/// One 128x128 area, post-superres.
#[derive(Clone, Copy, Default)]
pub struct Av1Restoration {
    pub lr: [[RestorationUnit; 4]; 3],
}

/// `txa[edge][txsz|step][y][x]`.
type Txa = [[[[u8; 32]; 32]; 2]; 2];

#[allow(clippy::too_many_arguments)]
fn decomp_tx(txa: &mut Txa, oy: usize, ox: usize, from: usize, depth: usize, y_off: usize, x_off: usize, tx_masks: &[u16; 2]) {
    let t = &TXFM_DIMENSIONS[from];
    let is_split = if from == 0 || depth > 1 { false } else { (tx_masks[depth] >> (y_off * 4 + x_off)) & 1 != 0 };
    if is_split {
        let sub = t.sub as usize;
        let htw4 = (t.w >> 1) as usize;
        let hth4 = (t.h >> 1) as usize;
        decomp_tx(txa, oy, ox, sub, depth + 1, y_off * 2, x_off * 2, tx_masks);
        if t.w >= t.h {
            decomp_tx(txa, oy, ox + htw4, sub, depth + 1, y_off * 2, x_off * 2 + 1, tx_masks);
        }
        if t.h >= t.w {
            decomp_tx(txa, oy + hth4, ox, sub, depth + 1, y_off * 2 + 1, x_off * 2, tx_masks);
            if t.w >= t.h {
                decomp_tx(txa, oy + hth4, ox + htw4, sub, depth + 1, y_off * 2 + 1, x_off * 2 + 1, tx_masks);
            }
        }
    } else {
        let lw = t.lw.min(2);
        let lh = t.lh.min(2);
        let n = 1usize << t.lw;
        for y in 0..t.h as usize {
            for x in 0..n {
                txa[0][0][oy + y][ox + x] = lw;
                txa[1][0][oy + y][ox + x] = lh;
            }
            txa[0][1][oy + y][ox] = t.w;
        }
        for x in 0..n {
            txa[1][1][oy][ox + x] = t.h;
        }
    }
}

#[inline]
fn set_bit(m: &mut [u16; 2], mask: u32) {
    let sidx = (mask >= 0x10000) as usize;
    m[sidx] |= (mask >> (sidx << 4)) as u16;
}

#[allow(clippy::too_many_arguments)]
fn mask_edges_inter(masks: &mut [[[[u16; 2]; 3]; 32]; 2], by4: usize, bx4: usize, w4: usize, h4: usize, skip: bool, max_tx: usize, tx_masks: &[u16; 2], a: &mut [u8], l: &mut [u8]) {
    let t = &TXFM_DIMENSIONS[max_tx];
    let mut txa: Txa = [[[[0; 32]; 32]; 2]; 2];
    let mut y_off = 0;
    let mut y = 0;
    while y < h4 {
        let mut x_off = 0;
        let mut x = 0;
        while x < w4 {
            decomp_tx(&mut txa, y, x, max_tx, 0, y_off, x_off, tx_masks);
            x += t.w as usize;
            x_off += 1;
        }
        y += t.h as usize;
        y_off += 1;
    }
    let mut mask = 1u32 << by4;
    for y in 0..h4 {
        set_bit(&mut masks[0][bx4][txa[0][0][y][0].min(l[y]) as usize], mask);
        mask <<= 1;
    }
    mask = 1u32 << bx4;
    for x in 0..w4 {
        set_bit(&mut masks[1][by4][txa[1][0][0][x].min(a[x]) as usize], mask);
        mask <<= 1;
    }
    if !skip {
        mask = 1u32 << by4;
        for y in 0..h4 {
            let mut ltx = txa[0][0][y][0];
            let mut step = txa[0][1][y][0] as usize;
            let mut x = step;
            while x < w4 {
                let rtx = txa[0][0][y][x];
                set_bit(&mut masks[0][bx4 + x][rtx.min(ltx) as usize], mask);
                ltx = rtx;
                step = txa[0][1][y][x] as usize;
                x += step;
            }
            mask <<= 1;
        }
        mask = 1u32 << bx4;
        for x in 0..w4 {
            let mut ttx = txa[1][0][0][x];
            let mut step = txa[1][1][0][x] as usize;
            let mut y = step;
            while y < h4 {
                let btx = txa[1][0][y][x];
                set_bit(&mut masks[1][by4 + y][ttx.min(btx) as usize], mask);
                ttx = btx;
                step = txa[1][1][y][x] as usize;
                y += step;
            }
            mask <<= 1;
        }
    }
    for y in 0..h4 {
        l[y] = txa[0][0][y][w4 - 1];
    }
    a[..w4].copy_from_slice(&txa[1][0][h4 - 1][..w4]);
}

#[allow(clippy::too_many_arguments)]
fn mask_edges_intra(masks: &mut [[[[u16; 2]; 3]; 32]; 2], by4: usize, bx4: usize, w4: usize, h4: usize, tx: usize, a: &mut [u8], l: &mut [u8]) {
    let t = &TXFM_DIMENSIONS[tx];
    let twl4c = t.lw.min(2);
    let thl4c = t.lh.min(2);
    let mut mask = 1u32 << by4;
    for y in 0..h4 {
        set_bit(&mut masks[0][bx4][twl4c.min(l[y]) as usize], mask);
        mask <<= 1;
    }
    mask = 1u32 << bx4;
    for x in 0..w4 {
        set_bit(&mut masks[1][by4][thl4c.min(a[x]) as usize], mask);
        mask <<= 1;
    }
    let hstep = t.w as usize;
    let tt = 1u64 << by4;
    let inner = ((tt << h4) - tt) as u32;
    let (i1, i2) = ((inner & 0xffff) as u16, (inner >> 16) as u16);
    let mut x = hstep;
    while x < w4 {
        masks[0][bx4 + x][twl4c as usize][0] |= i1;
        masks[0][bx4 + x][twl4c as usize][1] |= i2;
        x += hstep;
    }
    let vstep = t.h as usize;
    let tt = 1u64 << bx4;
    let inner = ((tt << w4) - tt) as u32;
    let (i1, i2) = ((inner & 0xffff) as u16, (inner >> 16) as u16);
    let mut y = vstep;
    while y < h4 {
        masks[1][by4 + y][thl4c as usize][0] |= i1;
        masks[1][by4 + y][thl4c as usize][1] |= i2;
        y += vstep;
    }
    a[..w4].fill(thl4c);
    l[..h4].fill(twl4c);
}

#[allow(clippy::too_many_arguments)]
fn mask_edges_chroma(masks: &mut [[[[u16; 2]; 2]; 32]; 2], cby4: usize, cbx4: usize, cw4: usize, ch4: usize, skip_inter: bool, tx: usize, a: &mut [u8], l: &mut [u8], ss_hor: usize, ss_ver: usize) {
    let t = &TXFM_DIMENSIONS[tx];
    let twl4c = (t.lw != 0) as u8;
    let thl4c = (t.lh != 0) as u8;
    let vbits = 4 - ss_ver;
    let hbits = 4 - ss_hor;
    let vmask = 16 >> ss_ver;
    let hmask = 16 >> ss_hor;
    let vmax = 1u32 << vmask;
    let hmax = 1u32 << hmask;
    let mut mask = 1u32 << cby4;
    for y in 0..ch4 {
        let sidx = (mask >= vmax) as usize;
        masks[0][cbx4][twl4c.min(l[y]) as usize][sidx] |= (mask >> (sidx << vbits)) as u16;
        mask <<= 1;
    }
    mask = 1u32 << cbx4;
    for x in 0..cw4 {
        let sidx = (mask >= hmax) as usize;
        masks[1][cby4][thl4c.min(a[x]) as usize][sidx] |= (mask >> (sidx << hbits)) as u16;
        mask <<= 1;
    }
    if !skip_inter {
        let hstep = t.w as usize;
        let tt = 1u64 << cby4;
        let inner = ((tt << ch4) - tt) as u32;
        let (i1, i2) = ((inner & ((1 << vmask) - 1)) as u16, (inner >> vmask) as u16);
        let mut x = hstep;
        while x < cw4 {
            if i1 != 0 {
                masks[0][cbx4 + x][twl4c as usize][0] |= i1;
            }
            if i2 != 0 {
                masks[0][cbx4 + x][twl4c as usize][1] |= i2;
            }
            x += hstep;
        }
        let vstep = t.h as usize;
        let tt = 1u64 << cbx4;
        let inner = ((tt << cw4) - tt) as u32;
        let (i1, i2) = ((inner & ((1 << hmask) - 1)) as u16, (inner >> hmask) as u16);
        let mut y = vstep;
        while y < ch4 {
            if i1 != 0 {
                masks[1][cby4 + y][thl4c as usize][0] |= i1;
            }
            if i2 != 0 {
                masks[1][cby4 + y][thl4c as usize][1] |= i2;
            }
            y += vstep;
        }
    }
    a[..cw4].fill(thl4c);
    l[..ch4].fill(twl4c);
}

/// Edge contexts handed to the mask builders: (above, left) luma and chroma.
pub struct LfEdges<'a> {
    pub ay: &'a mut [u8],
    pub ly: &'a mut [u8],
    pub uv: Option<(&'a mut [u8], &'a mut [u8])>,
}

/// `dav1d_create_lf_mask_intra` / `_inter` (`inter` = Some((skip, tx_masks))).
#[allow(clippy::too_many_arguments)]
pub fn create_lf_mask(
    lflvl: &mut Av1Filter,
    level_cache: &mut [[u8; 4]],
    b4_stride: usize,
    filter_level: [u8; 4],
    bx: usize,
    by: usize,
    iw: usize,
    ih: usize,
    bs: usize,
    ytx: usize,
    inter: Option<(bool, [u16; 2])>,
    uvtx: usize,
    layout: Layout,
    e: LfEdges,
) {
    let b_dim = BLOCK_DIMENSIONS[bs];
    let bw4 = (iw - bx).min(b_dim[0] as usize);
    let bh4 = (ih - by).min(b_dim[1] as usize);
    let bx4 = bx & 31;
    let by4 = by & 31;
    if bw4 != 0 && bh4 != 0 {
        for y in 0..bh4 {
            let o = (by + y) * b4_stride + bx;
            for v in &mut level_cache[o..o + bw4] {
                v[0] = filter_level[0];
                v[1] = filter_level[1];
            }
        }
        match inter {
            None => mask_edges_intra(&mut lflvl.filter_y, by4, bx4, bw4, bh4, ytx, e.ay, e.ly),
            Some((skip, tx_masks)) => mask_edges_inter(&mut lflvl.filter_y, by4, bx4, bw4, bh4, skip, ytx, &tx_masks, e.ay, e.ly),
        }
    }
    let Some((auv, luv)) = e.uv else { return };
    let ss_ver = (layout == Layout::I420) as usize;
    let ss_hor = (layout != Layout::I444) as usize;
    let cbw4 = (((iw + ss_hor) >> ss_hor) - (bx >> ss_hor)).min((b_dim[0] as usize + ss_hor) >> ss_hor);
    let cbh4 = (((ih + ss_ver) >> ss_ver) - (by >> ss_ver)).min((b_dim[1] as usize + ss_ver) >> ss_ver);
    if cbw4 == 0 || cbh4 == 0 {
        return;
    }
    let cbx4 = bx4 >> ss_hor;
    let cby4 = by4 >> ss_ver;
    for y in 0..cbh4 {
        let o = ((by >> ss_ver) + y) * b4_stride + (bx >> ss_hor);
        for v in &mut level_cache[o..o + cbw4] {
            v[2] = filter_level[2];
            v[3] = filter_level[3];
        }
    }
    let skip_inter = inter.map(|(s, _)| s).unwrap_or(false);
    mask_edges_chroma(&mut lflvl.filter_uv, cby4, cbx4, cbw4, cbh4, skip_inter, uvtx, auv, luv, ss_hor, ss_ver);
}

pub fn calc_eih(lut: &mut Av1FilterLut, sharp: i32) {
    for level in 0..64 {
        let mut limit = level;
        if sharp > 0 {
            limit >>= (sharp + 3) >> 2;
            limit = limit.min(9 - sharp);
        }
        limit = limit.max(1);
        lut.i[level as usize] = limit as u8;
        lut.e[level as usize] = (2 * (level + 2) + limit) as u8;
    }
    lut.sharp[0] = ((sharp + 3) >> 2) as u64;
    lut.sharp[1] = if sharp != 0 { (9 - sharp) as u64 } else { 0xff };
}

/// `[seg][dir][ref][is_gmv]`.
pub type LfLevels = [[[[u8; 2]; 8]; 4]; 8];

fn calc_lf_value(v: &mut [[u8; 2]; 8], base_lvl: i32, lf_delta: i32, seg_delta: i32, mr: Option<&LoopfilterModeRefDeltas>) {
    let base = ((base_lvl + lf_delta).clamp(0, 63) + seg_delta).clamp(0, 63);
    match mr {
        None => *v = [[base as u8; 2]; 8],
        Some(mr) => {
            let sh = (base >= 32) as i32;
            let l0 = (base + mr.ref_delta[0] as i32 * (1 << sh)).clamp(0, 63) as u8;
            v[0] = [l0, l0];
            for r in 1..8 {
                for m in 0..2 {
                    let delta = mr.mode_delta[m] as i32 + mr.ref_delta[r] as i32;
                    v[r][m] = (base + delta * (1 << sh)).clamp(0, 63) as u8;
                }
            }
        }
    }
}

pub fn calc_lf_values(values: &mut LfLevels, hdr: &FrameHeader, lf_delta: [i8; 4]) {
    let n_seg = if hdr.segmentation.enabled { 8 } else { 1 };
    let lf = &hdr.loopfilter;
    if lf.level_y[0] == 0 && lf.level_y[1] == 0 {
        for v in values.iter_mut().take(n_seg) {
            *v = [[[0; 2]; 8]; 4];
        }
        return;
    }
    let mr = if lf.mode_ref_delta_enabled { Some(&lf.mode_ref_deltas) } else { None };
    let multi = hdr.delta.lf_multi;
    for s in 0..n_seg {
        let segd = if hdr.segmentation.enabled { Some(&hdr.segmentation.seg_data.d[s]) } else { None };
        let sd = |f: fn(&super::obu::SegmentationData) -> i8| segd.map(|d| f(d) as i32).unwrap_or(0);
        calc_lf_value(&mut values[s][0], lf.level_y[0] as i32, lf_delta[0] as i32, sd(|d| d.delta_lf_y_v), mr);
        calc_lf_value(&mut values[s][1], lf.level_y[1] as i32, lf_delta[if multi { 1 } else { 0 }] as i32, sd(|d| d.delta_lf_y_h), mr);
        for (k, (lvl, dl)) in [(lf.level_u, 2usize), (lf.level_v, 3)].into_iter().enumerate() {
            let seg = if k == 0 { sd(|d| d.delta_lf_u) } else { sd(|d| d.delta_lf_v) };
            if lvl == 0 {
                values[s][2 + k] = [[0; 2]; 8];
            } else {
                calc_lf_value(&mut values[s][2 + k], lvl as i32, lf_delta[if multi { dl } else { 0 }] as i32, seg, mr);
            }
        }
    }
}
