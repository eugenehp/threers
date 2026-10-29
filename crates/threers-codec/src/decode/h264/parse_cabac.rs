//! CABAC macroblock layer parsing (§7.3.5 with `entropy_coding_mode_flag`
//! 1): binarisations (§9.3.2) and context selection (§9.3.3.1).

use super::params::SliceType;
use super::parse_cavlc::classify;
use super::slice::{Result, SliceDec};
use super::tables::{BLK_XY, LAST8, SIG8, SIG8_FIELD, FIELD4, FIELD8, ZIGZAG4, ZIGZAG8};
use super::types::*;
use crate::decode::bits::BitReader;
use crate::decode::cabac::Cabac;

const CBF_OFF: [usize; 5] = [0, 4, 8, 12, 16];
const SIG_OFF: [usize; 5] = [0, 15, 29, 44, 47];
const ABS_OFF: [usize; 5] = [0, 10, 20, 30, 39];

impl SliceDec<'_> {
    #[inline(always)]
    fn bin(&mut self, c: &mut Cabac, idx: usize) -> u32 {
        c.decision(&mut self.ctx[idx])
    }

    /// `mb_field_decoding_flag` (ctxIdx 70-72), read at a pair's top
    /// macroblock. The left term is the flag the pair inherited, as ffmpeg
    /// has it (the left pair's, within a slice).
    pub fn cabac_field_flag(&mut self, c: &mut Cabac) -> bool {
        let w = self.pic.w_mbs;
        let mut inc = (self.mb_field && self.mb_x > 0) as usize;
        if self.mb_y >= 2 {
            let m = &self.pic.mbs[self.addr - 2 * w];
            inc += (m.slice == self.slice_num && m.field) as usize;
        }
        self.bin(c, 70 + inc) == 1
    }

    pub fn cabac_skip_flag(&mut self, c: &mut Cabac) -> bool {
        let cond = |s: &Self, dx, dy| {
            s.mb_nb(dx, dy)
                .map_or(0, |a| !s.pic.mbs[a].kind.is_skip() as usize)
        };
        let inc = cond(self, -1, 0) + cond(self, 0, -1);
        let base = if self.hdr.slice_type == SliceType::P { 11 } else { 24 };
        self.bin(c, base + inc) == 1
    }

    /// `mb_type` for I macroblocks: the whole of it in an I slice
    /// (`base` 3), or the suffix after an intra prefix in P (17) / B (32).
    fn mb_type_i(&mut self, c: &mut Cabac, base: usize) -> u32 {
        let in_i = base == 3;
        let b0 = if in_i {
            let cond = |s: &Self, dx, dy| {
                s.mb_nb(dx, dy)
                    .map_or(0, |a| !s.pic.mbs[a].kind.is_nxn() as usize)
            };
            let inc = cond(self, -1, 0) + cond(self, 0, -1);
            self.bin(c, 3 + inc)
        } else {
            self.bin(c, base)
        };
        if b0 == 0 {
            return 0;
        }
        if c.terminate() == 1 {
            return 25;
        }
        let (luma, chroma, pred);
        if in_i {
            luma = self.bin(c, 3 + 3);
            chroma = if self.bin(c, 3 + 4) == 1 {
                1 + self.bin(c, 3 + 5)
            } else {
                0
            };
            pred = (self.bin(c, 3 + 6) << 1) | self.bin(c, 3 + 7);
        } else {
            luma = self.bin(c, base + 1);
            chroma = if self.bin(c, base + 2) == 1 {
                1 + self.bin(c, base + 2)
            } else {
                0
            };
            pred = (self.bin(c, base + 3) << 1) | self.bin(c, base + 3);
        }
        1 + pred + 4 * chroma + 12 * luma
    }

    fn mb_type_p(&mut self, c: &mut Cabac) -> u32 {
        if self.bin(c, 14) == 1 {
            return 5 + self.mb_type_i(c, 17);
        }
        if self.bin(c, 15) == 0 {
            if self.bin(c, 16) == 0 {
                0
            } else {
                3
            }
        } else if self.bin(c, 17) == 1 {
            1
        } else {
            2
        }
    }

    fn mb_type_b(&mut self, c: &mut Cabac) -> u32 {
        let cond = |s: &Self, dx, dy| {
            s.mb_nb(dx, dy)
                .map_or(0, |a| !s.pic.mbs[a].b_direct16 as usize)
        };
        let inc = cond(self, -1, 0) + cond(self, 0, -1);
        if self.bin(c, 27 + inc) == 0 {
            return 0;
        }
        if self.bin(c, 27 + 3) == 0 {
            return 1 + self.bin(c, 27 + 5);
        }
        let mut bits = self.bin(c, 27 + 4) << 3;
        bits |= self.bin(c, 27 + 5) << 2;
        bits |= self.bin(c, 27 + 5) << 1;
        bits |= self.bin(c, 27 + 5);
        match bits {
            0..=7 => bits + 3,
            13 => 23 + self.mb_type_i(c, 32),
            14 => 11,
            15 => 22,
            _ => ((bits << 1) | self.bin(c, 27 + 5)) - 4,
        }
    }

    fn sub_mb_type_p(&mut self, c: &mut Cabac) -> u8 {
        if self.bin(c, 21) == 1 {
            0
        } else if self.bin(c, 22) == 0 {
            1
        } else if self.bin(c, 23) == 1 {
            2
        } else {
            3
        }
    }

    fn sub_mb_type_b(&mut self, c: &mut Cabac) -> u8 {
        if self.bin(c, 36) == 0 {
            return 0;
        }
        if self.bin(c, 37) == 0 {
            return 1 + self.bin(c, 39) as u8;
        }
        let mut t = 3u8;
        if self.bin(c, 38) == 1 {
            if self.bin(c, 39) == 1 {
                return 11 + self.bin(c, 39) as u8;
            }
            t += 4;
        }
        t += 2 * self.bin(c, 39) as u8;
        t += self.bin(c, 39) as u8;
        t
    }

    /// Context increment term for `ref_idx_lX` from the neighbouring
    /// partition at `(x, y)`.
    fn ref_cond(&self, l: usize, x: isize, y: isize) -> usize {
        let Some((a, k)) = self.blk_nb(x, y) else {
            return 0;
        };
        let m = &self.pic.mbs[a];
        if m.kind.is_skip() || m.kind.is_intra() {
            return 0;
        }
        let k8 = (k / 8) * 2 + (k % 4) / 2;
        if m.direct8 & (1 << k8) != 0 {
            return 0;
        }
        let r = m.ref_idx[l][k8];
        // MBAFF: a field neighbour of a frame macroblock counts refIdx >> 1.
        let r = if self.mbaff && !self.mb_field && m.field { r >> 1 } else { r };
        (r > 0) as usize
    }

    fn ref_idx_cabac(&mut self, c: &mut Cabac, l: usize, x: usize, y: usize) -> Result<i8> {
        let inc = self.ref_cond(l, x as isize - 1, y as isize)
            + 2 * self.ref_cond(l, x as isize, y as isize - 1);
        let mut v = 0u32;
        if self.bin(c, 54 + inc) == 1 {
            v = 1;
            if self.bin(c, 54 + 4) == 1 {
                v = 2;
                while self.bin(c, 54 + 5) == 1 {
                    v += 1;
                    if v > 32 {
                        return Err("ref_idx too large".into());
                    }
                }
            }
        }
        if v >= self.num_ref(l) {
            return Err(format!("ref_idx_l{l} {v} out of range"));
        }
        Ok(v as i8)
    }

    fn mvd_abs_nb(&self, l: usize, comp: usize, x: isize, y: isize) -> u32 {
        let Some((a, k)) = self.blk_nb(x, y) else {
            return 0;
        };
        let m = &self.pic.mbs[a];
        let v = m.mvd[l][k][comp] as u32;
        // MBAFF: vertical mvd of a neighbour of the other kind, rescaled
        // (ffmpeg MAP_F2F, which only touches partitions using the list).
        if comp == 1 && self.mbaff && m.field != self.mb_field && m.ref_idx[l][(k / 8) * 2 + (k % 4) / 2] >= 0 {
            return if self.mb_field { v >> 1 } else { v << 1 };
        }
        v
    }

    fn mvd_cabac(&mut self, c: &mut Cabac, l: usize, comp: usize, x: usize, y: usize) -> Result<i32> {
        let sum = self.mvd_abs_nb(l, comp, x as isize - 1, y as isize)
            + self.mvd_abs_nb(l, comp, x as isize, y as isize - 1);
        let inc = if sum < 3 {
            0
        } else if sum <= 32 {
            1
        } else {
            2
        };
        let base = if comp == 0 { 40 } else { 47 };
        if self.bin(c, base + inc) == 0 {
            return Ok(0);
        }
        const INC: [usize; 8] = [3, 4, 5, 6, 6, 6, 6, 6];
        let mut abs = 1i32;
        while abs < 9 && self.bin(c, base + INC[abs as usize - 1]) == 1 {
            abs += 1;
        }
        if abs >= 9 {
            let mut k = 3;
            while c.bypass() == 1 {
                abs += 1 << k;
                k += 1;
                if k > 24 {
                    return Err("mvd suffix too long".into());
                }
            }
            abs += c.bypass_bits(k) as i32;
        }
        Ok(if c.bypass() == 1 { -abs } else { abs })
    }

    fn parse_mvd_pair(&mut self, c: &mut Cabac, l: usize, x: usize, y: usize, w: usize, h: usize) -> Result<()> {
        let dx = self.mvd_cabac(c, l, 0, x, y)?;
        let dy = self.mvd_cabac(c, l, 1, x, y)?;
        self.fill_mvd(l, x, y, w, h, [dx, dy]);
        Ok(())
    }

    /// Record parsed reference indices on the current macroblock so later
    /// partitions' context selection sees them.
    fn note_ref(&mut self, l: usize, x: usize, y: usize, w: usize, h: usize, r: i8) {
        let m = &mut self.pic.mbs[self.addr];
        for yy in (y..y + h).step_by(2) {
            for xx in (x..x + w).step_by(2) {
                m.ref_idx[l][(yy / 2) * 2 + xx / 2] = r;
            }
        }
    }

    pub fn parse_mb_cabac(&mut self, c: &mut Cabac) -> Result<()> {
        let st = self.hdr.slice_type;
        let raw = match st {
            SliceType::I => self.mb_type_i(c, 3),
            SliceType::P => self.mb_type_p(c),
            SliceType::B => self.mb_type_b(c),
        };
        let (kind, t) = classify(st, raw)?;
        self.syn.kind = kind;
        self.syn.mb_type = t;
        if kind == MbKind::IPcm {
            let start = c.aligned_byte_pos();
            let data = c.data();
            let mut r = BitReader::new(data);
            r.set_pos(start * 8);
            self.read_pcm(&mut r);
            c.restart(r.byte_pos());
            return Ok(());
        }
        let mut no_sub_lt8x8 = true;
        let is_8x8 = (kind == MbKind::P && t >= 3) || (kind == MbKind::B && t == 22);
        let b_direct16 = kind == MbKind::B && t == 0;
        if b_direct16 {
            self.pic.mbs[self.addr].b_direct16 = true;
            self.pic.mbs[self.addr].direct8 = 0xF;
        }
        match kind {
            MbKind::I4x4 => {
                if self.pps.transform_8x8_mode && self.t8x8_flag(c) {
                    self.syn.t8x8 = true;
                    self.syn.kind = MbKind::I8x8;
                }
                let n = if self.syn.t8x8 { 4 } else { 16 };
                for i in 0..n {
                    self.syn.prev_flag[i] = self.bin(c, 68) == 1;
                    if !self.syn.prev_flag[i] {
                        let b0 = self.bin(c, 69);
                        let b1 = self.bin(c, 69);
                        let b2 = self.bin(c, 69);
                        self.syn.rem[i] = (b0 | (b1 << 1) | (b2 << 2)) as u8;
                    }
                }
                if self.chroma {
                    self.syn.chroma_mode = self.chroma_mode_cabac(c);
                }
            }
            MbKind::I16x16 => {
                self.set_i16(t);
                if self.chroma {
                    self.syn.chroma_mode = self.chroma_mode_cabac(c);
                }
            }
            _ if b_direct16 => {}
            _ if is_8x8 => {
                for s in 0..4 {
                    self.syn.sub[s] = if kind == MbKind::B {
                        self.sub_mb_type_b(c)
                    } else {
                        self.sub_mb_type_p(c)
                    };
                }
                let subs: [(PredFlags, usize, usize); 4] = std::array::from_fn(|s| {
                    if kind == MbKind::B {
                        b_sub_layout(self.syn.sub[s])
                    } else {
                        p_sub_layout(self.syn.sub[s])
                    }
                });
                let mut direct = 0u8;
                for (s, &(pred, w, h)) in subs.iter().enumerate() {
                    if pred == 0 {
                        direct |= 1 << s;
                        if !self.sps.direct_8x8_inference {
                            no_sub_lt8x8 = false;
                        }
                    } else if w * h < 4 {
                        no_sub_lt8x8 = false;
                    }
                }
                self.pic.mbs[self.addr].direct8 = direct;
                let ref0 = kind == MbKind::P && t == 4;
                for l in 0..2 {
                    for s in 0..4 {
                        let (sx, sy) = ((s % 2) * 2, (s / 2) * 2);
                        if subs[s].0 & (1 << l) != 0 {
                            let r = if ref0 || self.num_ref(l) <= 1 {
                                0
                            } else {
                                self.ref_idx_cabac(c, l, sx, sy)?
                            };
                            self.syn.ref_idx[l][s] = r;
                            self.note_ref(l, sx, sy, 2, 2, r);
                        }
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
                                self.parse_mvd_pair(c, l, sx + px, sy + py, w, h)?;
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
                    for (i, &(x, y, w, h)) in parts.iter().enumerate() {
                        if preds[i] & (1 << l) != 0 {
                            let r = if self.num_ref(l) <= 1 {
                                0
                            } else {
                                self.ref_idx_cabac(c, l, x, y)?
                            };
                            self.syn.ref_idx[l][i] = r;
                            self.note_ref(l, x, y, w, h, r);
                        }
                    }
                }
                for l in 0..2 {
                    for (i, &(x, y, w, h)) in parts.iter().enumerate() {
                        if preds[i] & (1 << l) != 0 {
                            self.parse_mvd_pair(c, l, x, y, w, h)?;
                        }
                    }
                }
            }
        }
        if self.syn.kind != MbKind::I16x16 {
            self.syn.cbp = self.cbp_cabac(c);
            if self.syn.cbp & 15 != 0
                && self.pps.transform_8x8_mode
                && !self.syn.kind.is_intra()
                && no_sub_lt8x8
                && (!b_direct16 || self.sps.direct_8x8_inference)
            {
                self.syn.t8x8 = self.t8x8_flag(c);
            }
        }
        // Visible to neighbours' cbf contexts from here on.
        self.pic.mbs[self.addr].cbp = self.syn.cbp;
        self.pic.mbs[self.addr].t8x8 = self.syn.t8x8;
        self.pic.mbs[self.addr].kind = self.syn.kind;
        if self.syn.cbp != 0 || self.syn.kind == MbKind::I16x16 {
            self.syn.qp_delta = self.qp_delta_cabac(c)?;
            self.residual_cabac(c)?;
        }
        Ok(())
    }

    fn t8x8_flag(&mut self, c: &mut Cabac) -> bool {
        let cond = |s: &Self, dx, dy| s.mb_nb(dx, dy).map_or(0, |a| s.pic.mbs[a].t8x8 as usize);
        let inc = cond(self, -1, 0) + cond(self, 0, -1);
        self.bin(c, 399 + inc) == 1
    }

    fn chroma_mode_cabac(&mut self, c: &mut Cabac) -> u8 {
        let cond = |s: &Self, dx, dy| {
            s.mb_nb(dx, dy).map_or(0, |a| {
                let m = &s.pic.mbs[a];
                (m.kind.is_intra() && m.kind != MbKind::IPcm && m.chroma_mode != 0) as usize
            })
        };
        let inc = cond(self, -1, 0) + cond(self, 0, -1);
        if self.bin(c, 64 + inc) == 0 {
            0
        } else if self.bin(c, 64 + 3) == 0 {
            1
        } else if self.bin(c, 64 + 3) == 0 {
            2
        } else {
            3
        }
    }

    fn cbp_cabac(&mut self, c: &mut Cabac) -> u8 {
        // The neighbouring 8×8 block's bit, located through its 4×4 block
        // (which macroblock of an MBAFF pair, and which row, can differ per
        // 8×8 row).
        let nb8 = |s: &Self, x: isize, y: isize| {
            s.blk_nb(x, y).map(|(a, k)| (s.pic.mbs[a].cbp >> ((k / 8) * 2 + (k % 4) / 2)) & 1)
        };
        let mut cbp = 0u8;
        for b8 in 0..4usize {
            // condTerm is 1 when the neighbouring 8×8 had no coefficients.
            let a = if b8 % 2 == 1 {
                (cbp >> (b8 - 1)) & 1 == 0
            } else {
                nb8(self, -1, 2 * (b8 / 2) as isize) == Some(0)
            };
            let b = if b8 >= 2 {
                (cbp >> (b8 - 2)) & 1 == 0
            } else {
                nb8(self, 2 * (b8 % 2) as isize, -1) == Some(0)
            };
            let inc = a as usize + 2 * b as usize;
            cbp |= (self.bin(c, 73 + inc) as u8) << b8;
        }
        if self.chroma {
            let left = self.mb_nb(-1, 0).map(|a| self.pic.mbs[a].cbp);
            let top = self.mb_nb(0, -1).map(|a| self.pic.mbs[a].cbp);
            let ca = left.map_or(0, |l| l >> 4);
            let cb = top.map_or(0, |t| t >> 4);
            let inc = (ca > 0) as usize + 2 * (cb > 0) as usize;
            if self.bin(c, 77 + inc) == 1 {
                let inc = (ca == 2) as usize + 2 * (cb == 2) as usize;
                let v = 1 + self.bin(c, 77 + 4 + inc) as u8;
                cbp |= v << 4;
            }
        }
        cbp
    }

    fn qp_delta_cabac(&mut self, c: &mut Cabac) -> Result<i32> {
        let inc = self.prev_qp_nz as usize;
        if self.bin(c, 60 + inc) == 0 {
            return Ok(0);
        }
        let mut k = 1i32;
        if self.bin(c, 60 + 2) == 1 {
            k = 2;
            while self.bin(c, 60 + 3) == 1 {
                k += 1;
                if k > 200 {
                    return Err("mb_qp_delta too long".into());
                }
            }
        }
        Ok(if k & 1 == 1 { (k + 1) / 2 } else { -(k / 2) })
    }

    /// `coded_block_flag` context term from a neighbour: `None` when the
    /// neighbouring macroblock is unavailable.
    fn cbf_term(&self, v: Option<bool>) -> usize {
        match v {
            Some(b) => b as usize,
            None => self.syn.kind.is_intra() as usize,
        }
    }

    fn cbf_luma4(&self, x: isize, y: isize) -> Option<bool> {
        let (a, k) = self.blk_nb(x, y)?;
        Some(self.pic.mbs[a].nnz[k] != 0)
    }

    fn cbf_chroma_ac(&self, comp: usize, x: isize, y: isize) -> Option<bool> {
        let (a, k) = self.cblk_nb(x, y)?;
        Some(self.pic.mbs[a].nnz_c[comp][k] != 0)
    }

    fn cbf_dc(&self, bit: u8, dx: isize, dy: isize) -> Option<bool> {
        let a = self.mb_nb(dx, dy)?;
        Some(self.pic.mbs[a].cbf_dc & (1 << bit) != 0)
    }

    /// One `residual_block_cabac` (§7.3.5.3.3). `cbf_inc` is the
    /// `coded_block_flag` increment, `None` for 8×8 blocks (no flag).
    /// Levels land in `out` by list index; returns how many are non-zero.
    fn block_cabac(&mut self, c: &mut Cabac, cat: usize, max: usize, cbf_inc: Option<usize>, out: &mut [i32]) -> Result<usize> {
        if let Some(inc) = cbf_inc {
            if self.bin(c, 85 + CBF_OFF[cat] + inc) == 0 {
                return Ok(0);
            }
        }
        let field = self.field_coded();
        let (sig_base, last_base, abs_base) = match (cat == 5, field) {
            (true, false) => (402, 417, 426),
            (true, true) => (436, 451, 426),
            (false, false) => (105 + SIG_OFF[cat], 166 + SIG_OFF[cat], 227 + ABS_OFF[cat]),
            (false, true) => (277 + SIG_OFF[cat], 338 + SIG_OFF[cat], 227 + ABS_OFF[cat]),
        };
        let sig8 = if field { &SIG8_FIELD } else { &SIG8 };
        let mut sig = [0u8; 64];
        let mut n = 0usize;
        let mut last = max - 1;
        for i in 0..max - 1 {
            let (si, li) = match cat {
                // Chroma DC: Min(numDecod / NumC8x8, 2), NumC8x8 = max / 4.
                3 => ((i / (max / 4)).min(2), (i / (max / 4)).min(2)),
                5 => (sig8[i] as usize, LAST8[i] as usize),
                _ => (i, i),
            };
            if self.bin(c, sig_base + si) == 1 {
                sig[n] = i as u8;
                n += 1;
                if self.bin(c, last_base + li) == 1 {
                    last = i;
                    break;
                }
            }
        }
        if last == max - 1 && (n == 0 || sig[n - 1] as usize != max - 1) {
            sig[n] = (max - 1) as u8;
            n += 1;
        }
        let mut gt1 = 0usize;
        let mut eq1 = 0usize;
        let gt1_cap = if cat == 3 { 3 } else { 4 };
        for j in (0..n).rev() {
            let inc0 = if gt1 != 0 { 0 } else { (1 + eq1).min(4) };
            let mut v = 0i32;
            if self.bin(c, abs_base + inc0) == 1 {
                v = 1;
                let inc = 5 + gt1.min(gt1_cap);
                while v < 14 && self.bin(c, abs_base + inc) == 1 {
                    v += 1;
                }
                if v == 14 {
                    let mut k = 0;
                    while c.bypass() == 1 {
                        v += 1 << k;
                        k += 1;
                        if k > 24 {
                            return Err("coeff_abs_level suffix too long".into());
                        }
                    }
                    v += c.bypass_bits(k) as i32;
                }
            }
            let abs = v + 1;
            if abs == 1 {
                eq1 += 1;
            } else {
                gt1 += 1;
            }
            out[sig[j] as usize] = if c.bypass() == 1 { -abs } else { abs };
        }
        Ok(n)
    }

    fn residual_cabac(&mut self, c: &mut Cabac) -> Result<()> {
        let (z4, z8) = if self.field_coded() { (&FIELD4, &FIELD8) } else { (&ZIGZAG4, &ZIGZAG8) };
        let cbp_l = self.syn.cbp & 15;
        let cbp_c = self.syn.cbp >> 4;
        let i16 = self.syn.kind == MbKind::I16x16;
        let t8 = self.syn.t8x8;
        let mut tmp = [0i32; 64];
        if i16 {
            let inc = self.cbf_term(self.cbf_dc(0, -1, 0))
                + 2 * self.cbf_term(self.cbf_dc(0, 0, -1));
            tmp[..16].fill(0);
            let n = self.block_cabac(c, 0, 16, Some(inc), &mut tmp)?;
            for k in 0..16 {
                self.syn.dc_luma[z4[k]] = tmp[k];
            }
            if n > 0 {
                self.pic.mbs[self.addr].cbf_dc |= 1;
            }
        }
        for b8 in 0..4 {
            if cbp_l & (1 << b8) == 0 {
                continue;
            }
            if t8 {
                tmp.fill(0);
                let n = self.block_cabac(c, 5, 64, None, &mut tmp)?;
                for k in 0..64 {
                    self.syn.coef8[b8][z8[k]] = tmp[k];
                }
                if n > 0 {
                    self.syn.coded8 |= 1 << b8;
                }
                let (bx, by) = ((b8 % 2) * 2, (b8 / 2) * 2);
                for d in [0, 1, 4, 5] {
                    self.pic.mbs[self.addr].nnz[by * 4 + bx + d] = n as u8;
                }
                continue;
            }
            for b4 in 0..4 {
                let blk = b8 * 4 + b4;
                let (bx, by) = BLK_XY[blk];
                let inc = self.cbf_term(self.cbf_luma4(bx as isize - 1, by as isize))
                    + 2 * self.cbf_term(self.cbf_luma4(bx as isize, by as isize - 1));
                tmp[..16].fill(0);
                let n = if i16 {
                    let n = self.block_cabac(c, 1, 15, Some(inc), &mut tmp)?;
                    for k in 0..15 {
                        self.syn.coef[blk][z4[k + 1]] = tmp[k];
                    }
                    n
                } else {
                    let n = self.block_cabac(c, 2, 16, Some(inc), &mut tmp)?;
                    for k in 0..16 {
                        self.syn.coef[blk][z4[k]] = tmp[k];
                    }
                    n
                };
                if n > 0 {
                    self.syn.coded |= 1 << blk;
                }
                self.pic.mbs[self.addr].nnz[by * 4 + bx] = n as u8;
            }
        }
        if self.chroma && cbp_c != 0 {
            for comp in 0..2 {
                let bit = 1 + comp as u8;
                let inc = self.cbf_term(self.cbf_dc(bit, -1, 0))
                    + 2 * self.cbf_term(self.cbf_dc(bit, 0, -1));
                let nmax = if self.c422 { 8 } else { 4 };
                tmp[..nmax].fill(0);
                let n = self.block_cabac(c, 3, nmax, Some(inc), &mut tmp)?;
                self.syn.dc_c[comp][..nmax].copy_from_slice(&tmp[..nmax]);
                if n > 0 {
                    self.pic.mbs[self.addr].cbf_dc |= 1 << bit;
                }
            }
            if cbp_c & 2 != 0 {
                for comp in 0..2 {
                    for blk in 0..if self.c422 { 8 } else { 4 } {
                        let (bx, by) = ((blk % 2) as isize, (blk / 2) as isize);
                        let inc = self.cbf_term(self.cbf_chroma_ac(comp, bx - 1, by))
                            + 2 * self.cbf_term(self.cbf_chroma_ac(comp, bx, by - 1));
                        tmp[..16].fill(0);
                        let n = self.block_cabac(c, 4, 15, Some(inc), &mut tmp)?;
                        for k in 0..15 {
                            self.syn.ac_c[comp][blk][z4[k + 1]] = tmp[k];
                        }
                        if n > 0 {
                            self.syn.coded_c[comp] |= 1 << blk;
                        }
                        self.pic.mbs[self.addr].nnz_c[comp][blk] = n as u8;
                    }
                }
            }
        }
        Ok(())
    }
}
