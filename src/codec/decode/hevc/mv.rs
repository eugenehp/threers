//! HEVC motion vector prediction (§8.5.3.2): merge mode, AMVP, and the
//! temporal (collocated) candidate.

use super::ctu::{PartMode, SliceDec};
use super::params::SliceType;
use super::pic::MvField;

/// `distScaleFactor` scaling of a vector by POC distances (8-182 to 8-186).
fn scale_mv(mv: [i16; 2], td: i32, tb: i32) -> [i16; 2] {
    let td = td.clamp(-128, 127);
    let tb = tb.clamp(-128, 127);
    if td == 0 {
        return mv;
    }
    let tx = (16384 + (td.abs() >> 1)) / td;
    let dsf = ((tb * tx + 32) >> 6).clamp(-4096, 4095);
    let s = |v: i16| {
        let p = dsf * v as i32;
        (p.signum() * ((p.abs() + 127) >> 8)).clamp(-32768, 32767) as i16
    };
    [s(mv[0]), s(mv[1])]
}

impl SliceDec<'_> {
    /// Prediction block availability (§6.4.2) of the neighbour at
    /// `(xn, yn)`, and its motion if it is inter.
    #[allow(clippy::too_many_arguments)]
    fn pb_nb(
        &self,
        xcb: usize,
        ycb: usize,
        ncb: usize,
        xpb: usize,
        ypb: usize,
        w: usize,
        h: usize,
        part: usize,
        xn: isize,
        yn: isize,
    ) -> Option<MvField> {
        let (xcb, ycb, ncb) = (xcb as isize, ycb as isize, ncb as isize);
        let same_cb = xcb <= xn && ycb <= yn && xcb + ncb > xn && ycb + ncb > yn;
        let ok = if !same_cb {
            self.avail(xpb as isize, ypb as isize, xn, yn)
        } else {
            !((w << 1) as isize == ncb
                && (h << 1) as isize == ncb
                && part == 1
                && ycb + h as isize <= yn
                && xcb + w as isize > xn)
        };
        if !ok {
            return None;
        }
        let f = self.pic.mv[self.pic.idx4(xn as usize, yn as usize)];
        if f.ref_idx == [-1, -1] {
            None
        } else {
            Some(f)
        }
    }

    /// The collocated vector for list `lx` and reference `ref_idx`
    /// (§8.5.3.2.8), or `None`.
    fn tmvp(&self, xpb: usize, ypb: usize, w: usize, h: usize, lx: usize, ref_idx: i8) -> Option<[i16; 2]> {
        let col = self.col?;
        let log2 = self.sps.log2_ctb;
        let (xbr, ybr) = (xpb + w, ypb + h);
        if (ypb >> log2) == (ybr >> log2) && ybr < self.pic.height && xbr < self.pic.width {
            if let Some(mv) = self.col_mv(col, (xbr >> 4) << 4, (ybr >> 4) << 4, lx, ref_idx) {
                return Some(mv);
            }
        }
        let (xc, yc) = (xpb + (w >> 1), ypb + (h >> 1));
        self.col_mv(col, (xc >> 4) << 4, (yc >> 4) << 4, lx, ref_idx)
    }

    fn col_mv(&self, col: usize, x: usize, y: usize, lx: usize, ref_idx: i8) -> Option<[i16; 2]> {
        let cp = &self.dpb[col];
        let m = &cp.motion;
        let f = m.mv[(y >> 2) * m.w4 + (x >> 2)];
        if f.ref_idx == [-1, -1] {
            return None;
        }
        let lc = if f.ref_idx[0] < 0 {
            1
        } else if f.ref_idx[1] < 0 {
            0
        } else if self.no_backward_pred {
            lx
        } else {
            self.sh.collocated_from_l0 as usize
        };
        let slice = m.ctb_slice[(y >> m.log2_ctb) * m.ctb_w + (x >> m.log2_ctb)];
        let (col_ref_poc, col_ref_lt) = *m
            .slices
            .get(slice as usize)?
            .refs[lc]
            .get(f.ref_idx[lc] as usize)?;
        let cur = self.refs[lx].get(ref_idx as usize)?;
        if cur.long_term != col_ref_lt {
            return None;
        }
        let col_diff = cp.poc - col_ref_poc;
        let cur_diff = self.pic.poc - cur.poc;
        let mv = f.mv[lc];
        if cur.long_term || col_diff == cur_diff {
            Some(mv)
        } else {
            Some(scale_mv(mv, col_diff, cur_diff))
        }
    }

    /// Merge mode motion (§8.5.3.2.2).
    #[allow(clippy::too_many_arguments)]
    pub fn merge_motion(
        &self,
        xcb: usize,
        ycb: usize,
        ncb: usize,
        xpb: usize,
        ypb: usize,
        w: usize,
        h: usize,
        part: usize,
        merge_idx: usize,
    ) -> MvField {
        let (ow, oh) = (w, h);
        let pml = self.pps.log2_parallel_merge_level;
        let (xpb, ypb, w, h, part) = if pml > 2 && ncb == 8 {
            (xcb, ycb, ncb, ncb, 0)
        } else {
            (xpb, ypb, w, h, part)
        };
        let pm = self.part_mode;
        let nb = |xn: isize, yn: isize| -> Option<MvField> {
            if (xpb as isize >> pml) == (xn >> pml) && (ypb as isize >> pml) == (yn >> pml) {
                return None;
            }
            self.pb_nb(xcb, ycb, ncb, xpb, ypb, w, h, part, xn, yn)
        };
        let (xi, yi, wi, hi) = (xpb as isize, ypb as isize, w as isize, h as isize);
        // Pruning compares against a neighbour's motion whenever that
        // neighbour is *available* — whether or not it was itself pruned.
        let a1r = if part == 1 && matches!(pm, PartMode::PNx2N | PartMode::PnLx2N | PartMode::PnRx2N) {
            None
        } else {
            nb(xi - 1, yi + hi - 1)
        };
        let b1r = if part == 1 && matches!(pm, PartMode::P2NxN | PartMode::P2NxnU | PartMode::P2NxnD) {
            None
        } else {
            nb(xi + wi - 1, yi - 1)
        };
        let a1 = a1r;
        let b1 = b1r.filter(|f| a1r != Some(*f));
        let b0 = nb(xi + wi, yi - 1).filter(|f| b1r != Some(*f));
        let a0 = nb(xi - 1, yi + hi).filter(|f| a1r != Some(*f));
        let b2 = if [a0, a1, b0, b1].iter().all(|c| c.is_some()) {
            None
        } else {
            nb(xi - 1, yi - 1).filter(|f| a1r != Some(*f) && b1r != Some(*f))
        };
        let max = self.sh.max_merge_cand as usize;
        let mut list: Vec<MvField> = [a1, b1, b0, a0, b2].into_iter().flatten().collect();
        let is_b = self.sh.slice_type == SliceType::B;
        // The temporal candidate is needed whenever it could be picked or
        // feed a combined candidate — past `max` spatial ones it cannot.
        if self.sh.temporal_mvp && list.len() < max {
            let l0 = self.tmvp(xpb, ypb, w, h, 0, 0);
            let l1 = if is_b { self.tmvp(xpb, ypb, w, h, 1, 0) } else { None };
            if l0.is_some() || l1.is_some() {
                list.push(MvField {
                    mv: [l0.unwrap_or([0, 0]), l1.unwrap_or([0, 0])],
                    ref_idx: [if l0.is_some() { 0 } else { -1 }, if l1.is_some() { 0 } else { -1 }],
                });
            }
        }
        let orig = list.len();
        if is_b && orig > 1 && orig < max {
            const PAIRS: [(usize, usize); 12] = [
                (0, 1),
                (1, 0),
                (0, 2),
                (2, 0),
                (1, 2),
                (2, 1),
                (0, 3),
                (3, 0),
                (1, 3),
                (3, 1),
                (2, 3),
                (3, 2),
            ];
            for &(i0, i1) in PAIRS.iter().take(orig * (orig - 1)) {
                if list.len() >= max {
                    break;
                }
                let (c0, c1) = (list[i0], list[i1]);
                if c0.ref_idx[0] >= 0 && c1.ref_idx[1] >= 0 {
                    let p0 = self.refs[0].get(c0.ref_idx[0] as usize).map(|e| e.poc);
                    let p1 = self.refs[1].get(c1.ref_idx[1] as usize).map(|e| e.poc);
                    if p0 != p1 || c0.mv[0] != c1.mv[1] {
                        list.push(MvField {
                            mv: [c0.mv[0], c1.mv[1]],
                            ref_idx: [c0.ref_idx[0], c1.ref_idx[1]],
                        });
                    }
                }
            }
        }
        let num_ref = if is_b {
            self.sh.num_ref_idx[0].min(self.sh.num_ref_idx[1])
        } else {
            self.sh.num_ref_idx[0]
        } as i8;
        let mut zero = 0i8;
        while list.len() < max {
            let r = if zero < num_ref { zero } else { 0 };
            list.push(MvField {
                mv: [[0; 2]; 2],
                ref_idx: [r, if is_b { r } else { -1 }],
            });
            zero += 1;
        }
        let mut m = list[merge_idx.min(list.len() - 1)];
        if m.ref_idx[0] >= 0 && m.ref_idx[1] >= 0 && ow + oh == 12 {
            m.ref_idx[1] = -1;
            m.mv[1] = [0, 0];
        }
        m
    }

    /// AMVP predictor (§8.5.3.2.6) for list `lx`, reference `ref_idx`.
    #[allow(clippy::too_many_arguments)]
    pub fn amvp(
        &self,
        xcb: usize,
        ycb: usize,
        ncb: usize,
        xpb: usize,
        ypb: usize,
        w: usize,
        h: usize,
        part: usize,
        lx: usize,
        ref_idx: i8,
        mvp_flag: usize,
    ) -> [i16; 2] {
        let ly = 1 - lx;
        let target = self.refs[lx][ref_idx as usize];
        let cur_poc = self.pic.poc;
        let nb = |xn: isize, yn: isize| self.pb_nb(xcb, ycb, ncb, xpb, ypb, w, h, part, xn, yn);
        let ref_of = |l: usize, f: &MvField| self.refs[l].get(f.ref_idx[l] as usize).copied();
        let (xi, yi, wi, hi) = (xpb as isize, ypb as isize, w as isize, h as isize);

        // Same picture, no scaling.
        let same_pic = |f: &MvField| -> Option<[i16; 2]> {
            for l in [lx, ly] {
                if f.pred(l) && ref_of(l, f).is_some_and(|e| e.poc == target.poc) {
                    return Some(f.mv[l]);
                }
            }
            None
        };
        // Any picture of the same long-term-ness, scaled when short-term.
        let scaled = |f: &MvField| -> Option<[i16; 2]> {
            for l in [lx, ly] {
                if !f.pred(l) {
                    continue;
                }
                let e = ref_of(l, f)?;
                if e.long_term == target.long_term {
                    let mv = f.mv[l];
                    return Some(if !e.long_term && !target.long_term {
                        scale_mv(mv, cur_poc - e.poc, cur_poc - target.poc)
                    } else {
                        mv
                    });
                }
            }
            None
        };

        let a = [nb(xi - 1, yi + hi), nb(xi - 1, yi + hi - 1)];
        let is_scaled = a.iter().any(|c| c.is_some());
        let mut mv_a = a.iter().flatten().find_map(&same_pic);
        if mv_a.is_none() {
            mv_a = a.iter().flatten().find_map(&scaled);
        }
        let b = [nb(xi + wi, yi - 1), nb(xi + wi - 1, yi - 1), nb(xi - 1, yi - 1)];
        let mut mv_b = b.iter().flatten().find_map(same_pic);
        if !is_scaled {
            if mv_b.is_some() && mv_a.is_none() {
                mv_a = mv_b;
            }
            mv_b = b.iter().flatten().find_map(scaled);
        }
        let mut list: Vec<[i16; 2]> = Vec::with_capacity(3);
        if let Some(a) = mv_a {
            list.push(a);
        }
        if let Some(b) = mv_b {
            if mv_a != Some(b) {
                list.push(b);
            }
        }
        if list.len() < 2 && mvp_flag >= list.len() {
            if let Some(c) = self.tmvp(xpb, ypb, w, h, lx, ref_idx) {
                list.push(c);
            }
        }
        while list.len() < 2 {
            list.push([0, 0]);
        }
        list[mvp_flag]
    }
}
