//! CAVLC macroblock layer parsing (§7.3.5 with `entropy_coding_mode_flag`
//! 0).

use super::params::SliceType;
use super::slice::{Result, SliceDec};
use super::tables::{BLK_XY, CBP_400, CBP_420, FIELD4, FIELD8, ZIGZAG4, ZIGZAG8};
use super::types::*;
use crate::decode::bits::BitReader;

/// Map a raw `mb_type` to its kind and the index within that kind's table.
pub fn classify(st: SliceType, t: u32) -> Result<(MbKind, u32)> {
    let intra = |t: u32| -> Result<(MbKind, u32)> {
        match t {
            0 => Ok((MbKind::I4x4, 0)),
            1..=24 => Ok((MbKind::I16x16, t)),
            25 => Ok((MbKind::IPcm, 25)),
            _ => Err(format!("mb_type {t} out of range")),
        }
    };
    match st {
        SliceType::I => intra(t),
        SliceType::P if t < 5 => Ok((MbKind::P, t)),
        SliceType::P => intra(t - 5),
        SliceType::B if t < 23 => Ok((MbKind::B, t)),
        SliceType::B => intra(t - 23),
    }
}

impl SliceDec<'_> {
    /// Set up `syn` for an Intra 16×16 `mb_type` (1-24).
    pub fn set_i16(&mut self, t: u32) {
        self.syn.i16_mode = ((t - 1) % 4) as u8;
        let cbp_c = ((t - 1) / 4) % 3;
        let cbp_l = if t >= 13 { 15 } else { 0 };
        self.syn.cbp = (cbp_c << 4) as u8 | cbp_l;
    }

    pub fn read_pcm(&mut self, r: &mut BitReader) {
        r.byte_align();
        for i in 0..256 {
            self.syn.pcm[i] = r.u(self.bd_y) as u16;
        }
        if self.chroma {
            for i in 0..if self.c422 { 256 } else { 128 } {
                self.syn.pcm[256 + i] = r.u(self.bd_c) as u16;
            }
        }
    }

    pub fn parse_mb_cavlc(&mut self, r: &mut BitReader) -> Result<()> {
        let st = self.hdr.slice_type;
        let (kind, t) = classify(st, r.ue())?;
        self.syn.kind = kind;
        self.syn.mb_type = t;
        if kind == MbKind::IPcm {
            self.read_pcm(r);
            return Ok(());
        }
        let mut no_sub_lt8x8 = true;
        let is_8x8 = (kind == MbKind::P && t >= 3) || (kind == MbKind::B && t == 22);
        let b_direct16 = kind == MbKind::B && t == 0;
        match kind {
            MbKind::I4x4 => {
                if self.pps.transform_8x8_mode && r.flag() {
                    self.syn.t8x8 = true;
                    self.syn.kind = MbKind::I8x8;
                }
                let n = if self.syn.t8x8 { 4 } else { 16 };
                for i in 0..n {
                    self.syn.prev_flag[i] = r.flag();
                    if !self.syn.prev_flag[i] {
                        self.syn.rem[i] = r.u(3) as u8;
                    }
                }
                if self.chroma {
                    self.syn.chroma_mode = self.read_chroma_mode(r)?;
                }
            }
            MbKind::I16x16 => {
                self.set_i16(t);
                if self.chroma {
                    self.syn.chroma_mode = self.read_chroma_mode(r)?;
                }
            }
            _ if b_direct16 => {}
            _ if is_8x8 => {
                for s in 0..4 {
                    let v = r.ue();
                    if (kind == MbKind::P && v > 3) || v > 12 {
                        return Err("sub_mb_type out of range".into());
                    }
                    self.syn.sub[s] = v as u8;
                }
                let subs: [(PredFlags, usize, usize); 4] = std::array::from_fn(|s| {
                    if kind == MbKind::B {
                        b_sub_layout(self.syn.sub[s])
                    } else {
                        p_sub_layout(self.syn.sub[s])
                    }
                });
                for (pred, w, h) in subs {
                    if pred == 0 {
                        if !self.sps.direct_8x8_inference {
                            no_sub_lt8x8 = false;
                        }
                    } else if w * h < 4 {
                        no_sub_lt8x8 = false;
                    }
                }
                let ref0 = kind == MbKind::P && t == 4;
                for l in 0..2 {
                    for s in 0..4 {
                        let pred = subs[s].0;
                        self.syn.ref_idx[l][s] = if pred & (1 << l) != 0 && !ref0 {
                            self.read_ref_idx(r, l)?
                        } else {
                            0
                        };
                    }
                }
                for l in 0..2 {
                    for s in 0..4 {
                        let (pred, w, h) = subs[s];
                        if pred & (1 << l) == 0 {
                            continue;
                        }
                        let (sx, sy) = ((s % 2) * 2, (s / 2) * 2);
                        for py in (0..2).step_by(h) {
                            for px in (0..2).step_by(w) {
                                let d = [r.se(), r.se()];
                                self.fill_mvd(l, sx + px, sy + py, w, h, d);
                            }
                        }
                    }
                }
            }
            _ => {
                let (shape, preds) = if kind == MbKind::B {
                    b_mb_layout(t)
                } else {
                    p_mb_layout(t)
                };
                let parts = shape.parts();
                for l in 0..2 {
                    for i in 0..parts.len() {
                        if preds[i] & (1 << l) != 0 {
                            self.syn.ref_idx[l][i] = self.read_ref_idx(r, l)?;
                        }
                    }
                }
                for l in 0..2 {
                    for (i, &(x, y, w, h)) in parts.iter().enumerate() {
                        if preds[i] & (1 << l) != 0 {
                            let d = [r.se(), r.se()];
                            self.fill_mvd(l, x, y, w, h, d);
                        }
                    }
                }
            }
        }
        if self.syn.kind != MbKind::I16x16 {
            let code = r.ue() as usize;
            let intra = self.syn.kind.is_intra();
            let cbp = if self.chroma {
                *CBP_420[(!intra) as usize].get(code).ok_or("coded_block_pattern out of range")?
            } else {
                *CBP_400[(!intra) as usize].get(code).ok_or("coded_block_pattern out of range")?
            };
            self.syn.cbp = cbp;
            if cbp & 15 != 0
                && self.pps.transform_8x8_mode
                && !intra
                && no_sub_lt8x8
                && (!b_direct16 || self.sps.direct_8x8_inference)
            {
                self.syn.t8x8 = r.flag();
            }
        }
        if self.syn.cbp != 0 || self.syn.kind == MbKind::I16x16 {
            self.syn.qp_delta = r.se();
            self.residual_cavlc(r)?;
        }
        Ok(())
    }

    fn read_chroma_mode(&mut self, r: &mut BitReader) -> Result<u8> {
        let m = r.ue();
        if m > 3 {
            return Err("intra_chroma_pred_mode out of range".into());
        }
        Ok(m as u8)
    }

    fn read_ref_idx(&mut self, r: &mut BitReader, l: usize) -> Result<i8> {
        let n = self.num_ref(l);
        let v = match n {
            0 | 1 => 0,
            2 => 1 - r.u(1),
            _ => r.ue(),
        };
        if v >= n.max(1) {
            return Err(format!("ref_idx_l{l} {v} out of range"));
        }
        Ok(v as i8)
    }

    /// Record an `mvd` over a partition's 4×4 blocks — CABAC context
    /// selection reads it back from neighbours, and motion derivation from
    /// the partition's top-left block.
    pub fn fill_mvd(&mut self, l: usize, x: usize, y: usize, w: usize, h: usize, d: [i32; 2]) {
        let abs = [d[0].unsigned_abs().min(255) as u8, d[1].unsigned_abs().min(255) as u8];
        let m = &mut self.pic.mbs[self.addr];
        for yy in y..y + h {
            for xx in x..x + w {
                self.syn.mvd[l][yy * 4 + xx] = d;
                m.mvd[l][yy * 4 + xx] = abs;
            }
        }
    }

    /// `TotalCoeff` of the luma 4×4 block at `(x, y)` relative to the
    /// macroblock, if available.
    fn nnz_luma(&self, x: isize, y: isize) -> Option<i32> {
        let (a, k) = self.blk_nb(x, y)?;
        Some(self.pic.mbs[a].nnz[k] as i32)
    }

    fn nnz_chroma(&self, c: usize, x: isize, y: isize) -> Option<i32> {
        let (a, k) = self.cblk_nb(x, y)?;
        Some(self.pic.mbs[a].nnz_c[c][k] as i32)
    }

    fn nc_of(a: Option<i32>, b: Option<i32>) -> i32 {
        match (a, b) {
            (Some(a), Some(b)) => (a + b + 1) >> 1,
            (Some(a), None) => a,
            (None, Some(b)) => b,
            _ => 0,
        }
    }

    fn residual_cavlc(&mut self, r: &mut BitReader) -> Result<()> {
        let (z4, z8) = if self.field_coded() { (&FIELD4, &FIELD8) } else { (&ZIGZAG4, &ZIGZAG8) };
        let cbp_l = self.syn.cbp & 15;
        let cbp_c = self.syn.cbp >> 4;
        let i16 = self.syn.kind == MbKind::I16x16;
        let t8 = self.syn.t8x8;
        let mut tmp;
        if i16 {
            let nc = Self::nc_of(self.nnz_luma(-1, 0), self.nnz_luma(0, -1));
            tmp = [0; 16];
            self.tables.residual_block(r, nc, 16, &mut tmp)?;
            for k in 0..16 {
                self.syn.dc_luma[z4[k]] = tmp[k];
            }
        }
        for b8 in 0..4 {
            for b4 in 0..4 {
                let blk = b8 * 4 + b4;
                let (bx, by) = BLK_XY[blk];
                let raster = by * 4 + bx;
                if cbp_l & (1 << b8) == 0 {
                    self.pic.mbs[self.addr].nnz[raster] = 0;
                    continue;
                }
                let nc = Self::nc_of(
                    self.nnz_luma(bx as isize - 1, by as isize),
                    self.nnz_luma(bx as isize, by as isize - 1),
                );
                tmp = [0; 16];
                let total = if i16 {
                    let n = self.tables.residual_block(r, nc, 15, &mut tmp)?;
                    for k in 0..15 {
                        self.syn.coef[blk][z4[k + 1]] = tmp[k];
                    }
                    n
                } else if !t8 {
                    let n = self.tables.residual_block(r, nc, 16, &mut tmp)?;
                    for k in 0..16 {
                        self.syn.coef[blk][z4[k]] = tmp[k];
                    }
                    n
                } else {
                    let n = self.tables.residual_block(r, nc, 16, &mut tmp)?;
                    for k in 0..16 {
                        self.syn.coef8[b8][z8[4 * k + b4]] = tmp[k];
                    }
                    if n > 0 {
                        self.syn.coded8 |= 1 << b8;
                    }
                    n
                };
                if total > 0 && !t8 {
                    self.syn.coded |= 1 << blk;
                }
                self.pic.mbs[self.addr].nnz[raster] = total as u8;
            }
        }
        if self.chroma && cbp_c != 0 {
            for c in 0..2 {
                tmp = [0; 16];
                let n = if self.c422 { 8 } else { 4 };
                self.tables.residual_block(r, if self.c422 { -2 } else { -1 }, n, &mut tmp)?;
                self.syn.dc_c[c][..n].copy_from_slice(&tmp[..n]);
                self.pic.mbs[self.addr].cbf_dc |=
                    ((tmp[..n].iter().any(|&v| v != 0)) as u8) << (1 + c);
            }
            if cbp_c & 2 != 0 {
                for c in 0..2 {
                    for blk in 0..if self.c422 { 8 } else { 4 } {
                        let (bx, by) = ((blk % 2) as isize, (blk / 2) as isize);
                        let nc = Self::nc_of(
                            self.nnz_chroma(c, bx - 1, by),
                            self.nnz_chroma(c, bx, by - 1),
                        );
                        tmp = [0; 16];
                        let n = self.tables.residual_block(r, nc, 15, &mut tmp)?;
                        for k in 0..15 {
                            self.syn.ac_c[c][blk][z4[k + 1]] = tmp[k];
                        }
                        if n > 0 {
                            self.syn.coded_c[c] |= 1 << blk;
                        }
                        self.pic.mbs[self.addr].nnz_c[c][blk] = n as u8;
                    }
                }
            }
        }
        Ok(())
    }
}
