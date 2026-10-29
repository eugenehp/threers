//! VP9 motion vector prediction and coding (ffmpeg `vp9mvs.c`).

use super::block::FrameDec;
use super::data::{MV_CLASS_TREE, MV_FP_TREE, MV_JOINT_TREE};
use super::*;

const MV_REF_BLK_OFF: [[[i8; 2]; 8]; 13] = [
    [[3, -1], [-1, 3], [4, -1], [-1, 4], [-1, -1], [0, -1], [-1, 0], [6, -1]],
    [[0, -1], [-1, 0], [4, -1], [-1, 2], [-1, -1], [0, -3], [-3, 0], [2, -1]],
    [[-1, 0], [0, -1], [-1, 4], [2, -1], [-1, -1], [-3, 0], [0, -3], [-1, 2]],
    [[1, -1], [-1, 1], [2, -1], [-1, 2], [-1, -1], [0, -3], [-3, 0], [-3, -3]],
    [[0, -1], [-1, 0], [2, -1], [-1, -1], [-1, 1], [0, -3], [-3, 0], [-3, -3]],
    [[-1, 0], [0, -1], [-1, 2], [-1, -1], [1, -1], [-3, 0], [0, -3], [-3, -3]],
    [[0, -1], [-1, 0], [1, -1], [-1, 1], [-1, -1], [0, -3], [-3, 0], [-3, -3]],
    [[0, -1], [-1, 0], [1, -1], [-1, -1], [0, -2], [-2, 0], [-2, -1], [-1, -2]],
    [[-1, 0], [0, -1], [-1, 1], [-1, -1], [-2, 0], [0, -2], [-1, -2], [-2, -1]],
    [[0, -1], [-1, 0], [-1, -1], [0, -2], [-2, 0], [-1, -2], [-2, -1], [-2, -2]],
    [[0, -1], [-1, 0], [-1, -1], [0, -2], [-2, 0], [-1, -2], [-2, -1], [-2, -2]],
    [[0, -1], [-1, 0], [-1, -1], [0, -2], [-2, 0], [-1, -2], [-2, -1], [-2, -2]],
    [[0, -1], [-1, 0], [-1, -1], [0, -2], [-2, 0], [-1, -2], [-2, -1], [-2, -2]],
];

const INVALID_MV: u32 = 0x8000_8000;

fn unpack(m: u32) -> Mv {
    Mv { x: m as u16 as i16, y: (m >> 16) as u16 as i16 }
}

/// The search state of `find_ref_mvs` (its `mem` / `mem_sub8x8`).
struct Search {
    idx: bool,
    sb: i32,
    mem: u32,
    mem_sub8x8: u32,
}

impl FrameDec<'_> {
    fn clamp_mv(&self, m: Mv) -> Mv {
        Mv { x: (m.x as i32).clamp(self.min_mv.0, self.max_mv.0) as i16, y: (m.y as i32).clamp(self.min_mv.1, self.max_mv.1) as i16 }
    }

    /// `RETURN_DIRECT_MV`: `Some(result)` when the search ends.
    fn direct_mv(s: &mut Search, m: Mv) -> Option<Mv> {
        let p = m.packed();
        if !s.idx {
            return Some(m);
        } else if s.mem == INVALID_MV {
            s.mem = p;
        } else if p != s.mem {
            return Some(m);
        }
        None
    }

    /// `RETURN_MV`.
    fn ret_mv(&self, s: &mut Search, mv: Mv) -> Option<Mv> {
        if s.sb > 0 {
            if s.mem_sub8x8 == INVALID_MV {
                let tmp = self.clamp_mv(mv);
                if tmp.packed() != s.mem {
                    return Some(tmp);
                }
                s.mem_sub8x8 = mv.packed();
            } else if s.mem_sub8x8 != mv.packed() {
                let tmp = self.clamp_mv(mv);
                if tmp.packed() != s.mem {
                    return Some(tmp);
                }
                // BUG (libvpx-compatible): zero rather than the next candidate.
                return Some(Mv::default());
            }
        } else {
            let p = mv.packed();
            if !s.idx {
                return Some(self.clamp_mv(mv));
            } else if s.mem == INVALID_MV {
                s.mem = p;
            } else if p != s.mem {
                return Some(self.clamp_mv(mv));
            }
        }
        None
    }

    /// `find_ref_mvs`.
    fn find_ref_mvs(&self, r#ref: i8, z: usize, idx: bool, sb: i32) -> Mv {
        let b = &self.b;
        let (row, col, row7) = (self.row as isize, self.col as isize, self.row7);
        let p = &MV_REF_BLK_OFF[b.bs as usize];
        let mut s = Search { idx, sb, mem: INVALID_MV, mem_sub8x8: INVALID_MV };
        let stride = self.sb_cols as isize * 8;
        let mvs = &self.cur.mv;
        macro_rules! ret {
            ($e:expr) => {
                if let Some(m) = $e {
                    return m;
                }
            };
        }
        let mut i = 0;
        if sb >= 0 {
            if sb == 2 || sb == 1 {
                ret!(Self::direct_mv(&mut s, b.mv[0][z]));
            } else if sb == 3 {
                ret!(Self::direct_mv(&mut s, b.mv[2][z]));
                ret!(Self::direct_mv(&mut s, b.mv[1][z]));
                ret!(Self::direct_mv(&mut s, b.mv[0][z]));
            }
            if row > 0 {
                let mv = &mvs[((row - 1) * stride + col) as usize];
                let a = self.above_mv_ctx[(2 * col + (sb & 1) as isize) as usize];
                if mv.r#ref[0] == r#ref {
                    ret!(self.ret_mv(&mut s, a[0]));
                } else if mv.r#ref[1] == r#ref {
                    ret!(self.ret_mv(&mut s, a[1]));
                }
            }
            if col > self.tile_col_start as isize {
                let mv = &mvs[(row * stride + col - 1) as usize];
                let l = self.left_mv_ctx[2 * row7 + (sb >> 1) as usize];
                if mv.r#ref[0] == r#ref {
                    ret!(self.ret_mv(&mut s, l[0]));
                } else if mv.r#ref[1] == r#ref {
                    ret!(self.ret_mv(&mut s, l[1]));
                }
            }
            i = 2;
        }
        let in_range = |c: isize, r: isize| c >= self.tile_col_start as isize && c < self.cols as isize && r >= 0 && r < self.rows as isize;
        // Previously coded MVs in this neighbourhood, same reference.
        for k in i..8 {
            let (c, r) = (p[k][0] as isize + col, p[k][1] as isize + row);
            if in_range(c, r) {
                let mv = &mvs[(r * stride + c) as usize];
                if mv.r#ref[0] == r#ref {
                    ret!(self.ret_mv(&mut s, mv.mv[0]));
                } else if mv.r#ref[1] == r#ref {
                    ret!(self.ret_mv(&mut s, mv.mv[1]));
                }
            }
        }
        let last = if self.h.use_last_frame_mvs { self.mvpair } else { None };
        // At this position in the previous frame, same reference.
        if let Some(lf) = last {
            let mv = &lf.mv[(row * stride + col) as usize];
            if mv.r#ref[0] == r#ref {
                ret!(self.ret_mv(&mut s, mv.mv[0]));
            } else if mv.r#ref[1] == r#ref {
                ret!(self.ret_mv(&mut s, mv.mv[1]));
            }
        }
        let sbias = &self.h.signbias;
        let scaled = |m: Mv, other: i8| if sbias[other as usize] != sbias[r#ref as usize] { Mv { x: m.x.wrapping_neg(), y: m.y.wrapping_neg() } } else { m };
        // Different reference.
        for k in 0..8 {
            let (c, r) = (p[k][0] as isize + col, p[k][1] as isize + row);
            if in_range(c, r) {
                let mv = &mvs[(r * stride + c) as usize];
                if mv.r#ref[0] != r#ref && mv.r#ref[0] >= 0 {
                    ret!(self.ret_mv(&mut s, scaled(mv.mv[0], mv.r#ref[0])));
                }
                // BUG (libvpx): this condition regardless of whether the
                // first ref MV was used.
                if mv.r#ref[1] != r#ref && mv.r#ref[1] >= 0 && mv.mv[0].packed() != mv.mv[1].packed() {
                    ret!(self.ret_mv(&mut s, scaled(mv.mv[1], mv.r#ref[1])));
                }
            }
        }
        if let Some(lf) = last {
            let mv = &lf.mv[(row * stride + col) as usize];
            if mv.r#ref[0] != r#ref && mv.r#ref[0] >= 0 {
                ret!(self.ret_mv(&mut s, scaled(mv.mv[0], mv.r#ref[0])));
            }
            if mv.r#ref[1] != r#ref && mv.r#ref[1] >= 0 && mv.mv[0].packed() != mv.mv[1].packed() {
                ret!(self.ret_mv(&mut s, scaled(mv.mv[1], mv.r#ref[1])));
            }
        }
        let _ = unpack;
        self.clamp_mv(Mv::default())
    }

    /// `read_mv_component`.
    fn read_mv_component(&mut self, idx: usize, hp: bool) -> i32 {
        let pc = self.p.mv_comp[idx];
        let sign = self.c().get_prob(pc.sign);
        let c = self.c().get_tree(&MV_CLASS_TREE, &pc.classes) as usize;
        self.counts.mv_comp[idx].sign[sign as usize] += 1;
        self.counts.mv_comp[idx].classes[c] += 1;
        let mut n: i32;
        if c != 0 {
            n = 0;
            for m in 0..c {
                let bit = self.c().get_prob(pc.bits[m]);
                n |= (bit as i32) << m;
                self.counts.mv_comp[idx].bits[m][bit as usize] += 1;
            }
            n <<= 3;
            let bit = self.c().get_tree(&MV_FP_TREE, &pc.fp) as i32;
            n |= bit << 1;
            self.counts.mv_comp[idx].fp[bit as usize] += 1;
            if hp {
                let bit = self.c().get_prob(pc.hp);
                self.counts.mv_comp[idx].hp[bit as usize] += 1;
                n |= bit as i32;
            } else {
                n |= 1;
                // libvpx counts it even when not coded.
                self.counts.mv_comp[idx].hp[1] += 1;
            }
            n += 8 << c;
        } else {
            n = self.c().get_prob(pc.class0) as i32;
            self.counts.mv_comp[idx].class0[n as usize] += 1;
            let bit = self.c().get_tree(&MV_FP_TREE, &pc.class0_fp[n as usize]) as i32;
            self.counts.mv_comp[idx].class0_fp[n as usize][bit as usize] += 1;
            n = (n << 3) | (bit << 1);
            if hp {
                let bit = self.c().get_prob(pc.class0_hp);
                self.counts.mv_comp[idx].class0_hp[bit as usize] += 1;
                n |= bit as i32;
            } else {
                n |= 1;
                self.counts.mv_comp[idx].class0_hp[1] += 1;
            }
        }
        if sign {
            -(n + 1)
        } else {
            n + 1
        }
    }

    /// `ff_vp9_fill_mv` into `b.mv[bi]`.
    pub(super) fn fill_mv(&mut self, bi: usize, mode: u8, sb: i32) {
        if mode == ZEROMV {
            self.b.mv[bi] = [Mv::default(); 2];
            return;
        }
        let ncomp = if self.b.comp { 2 } else { 1 };
        for z in 0..ncomp {
            let r = self.b.r#ref[z] as i8;
            let mut mv = self.find_ref_mvs(r, z, mode == NEARMV, if mode == NEWMV { -1 } else { sb });
            let mut hp = false;
            if mode == NEWMV || sb == -1 {
                hp = self.h.highprecisionmvs && (mv.x as i32).abs() < 64 && (mv.y as i32).abs() < 64;
                if !hp {
                    if mv.y & 1 != 0 {
                        mv.y += if mv.y < 0 { 1 } else { -1 };
                    }
                    if mv.x & 1 != 0 {
                        mv.x += if mv.x < 0 { 1 } else { -1 };
                    }
                }
            }
            if mode == NEWMV {
                let pj = self.p.mv_joint;
                let j = self.c().get_tree(&MV_JOINT_TREE, &pj);
                self.counts.mv_joint[j as usize] += 1;
                if j >= 2 {
                    mv.y = (mv.y as i32 + self.read_mv_component(0, hp)) as i16;
                }
                if j & 1 != 0 {
                    mv.x = (mv.x as i32 + self.read_mv_component(1, hp)) as i16;
                }
            }
            self.b.mv[bi][z] = mv;
        }
    }
}
