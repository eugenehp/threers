//! VP9 block reconstruction (ffmpeg `vp9recon.c`, `vp9_mc_template.c`).

use super::block::FrameDec;
use super::data::{BWH_TAB, INTRA_TXFM_TYPE};
use super::dsp;
use super::*;

/// `mode_conv[mode][have_left][have_top]`.
const MODE_CONV: [[[usize; 2]; 2]; 10] = [
    [[DC_127_PRED, VERT_PRED], [DC_127_PRED, VERT_PRED]],
    [[DC_129_PRED, DC_129_PRED], [HOR_PRED, HOR_PRED]],
    [[DC_128_PRED, TOP_DC_PRED], [LEFT_DC_PRED, DC_PRED]],
    [[DC_127_PRED, DIAG_DOWN_LEFT_PRED], [DC_127_PRED, DIAG_DOWN_LEFT_PRED]],
    [[DIAG_DOWN_RIGHT_PRED, DIAG_DOWN_RIGHT_PRED], [DIAG_DOWN_RIGHT_PRED, DIAG_DOWN_RIGHT_PRED]],
    [[VERT_RIGHT_PRED, VERT_RIGHT_PRED], [VERT_RIGHT_PRED, VERT_RIGHT_PRED]],
    [[HOR_DOWN_PRED, HOR_DOWN_PRED], [HOR_DOWN_PRED, HOR_DOWN_PRED]],
    [[DC_127_PRED, VERT_LEFT_PRED], [DC_127_PRED, VERT_LEFT_PRED]],
    [[DC_129_PRED, DC_129_PRED], [HOR_UP_PRED, HOR_UP_PRED]],
    [[DC_129_PRED, VERT_PRED], [HOR_PRED, TM_VP8_PRED]],
];

/// `edges[mode]`: (needs_left, needs_top, needs_topleft, needs_topright,
/// invert_left).
const EDGES: [(bool, bool, bool, bool, bool); 15] = [
    (false, true, false, false, false), // VERT
    (true, false, false, false, false), // HOR
    (true, true, false, false, false),  // DC
    (false, true, false, true, false),  // DIAG_DOWN_LEFT
    (true, true, true, false, false),   // DIAG_DOWN_RIGHT
    (true, true, true, false, false),   // VERT_RIGHT
    (true, true, true, false, false),   // HOR_DOWN
    (false, true, false, true, false),  // VERT_LEFT
    (true, false, false, false, true),  // HOR_UP
    (true, true, true, false, false),   // TM
    (true, false, false, false, false), // LEFT_DC
    (false, true, false, false, false), // TOP_DC
    (false, false, false, false, false),
    (false, false, false, false, false),
    (false, false, false, false, false),
];

fn rounded_div(a: i32, b: i32) -> i32 {
    if a >= 0 {
        (a + (b >> 1)) / b
    } else {
        (a - (b >> 1)) / b
    }
}

fn avg2(a: Mv, b: Mv) -> Mv {
    Mv { x: rounded_div(a.x as i32 + b.x as i32, 2) as i16, y: rounded_div(a.y as i32 + b.y as i32, 2) as i16 }
}

fn avg4(a: Mv, b: Mv, c: Mv, d: Mv) -> Mv {
    Mv {
        x: rounded_div(a.x as i32 + b.x as i32 + c.x as i32 + d.x as i32, 4) as i16,
        y: rounded_div(a.y as i32 + b.y as i32 + c.y as i32 + d.y as i32, 4) as i16,
    }
}

impl FrameDec<'_> {
    /// `check_intra_mode` + the prediction: predict the `4 << tx` block at
    /// `(px, py)` (samples) of plane `p`. `x`, `y` are its 4×4 position in
    /// the block, `w` the block width in 4×4 units.
    #[allow(clippy::too_many_arguments)]
    fn intra_block(&mut self, mode: usize, p: usize, px: usize, py: usize, x: usize, y: usize, w: usize, tx: u8) {
        let (ss_h, ss_v) = if p == 0 { (0, 0) } else { (self.ss_h, self.ss_v) };
        let (row, col) = (self.row, self.col);
        let bpp = self.bpp;
        let have_top = row > 0 || y > 0;
        let have_left = col > self.tile_col_start || x > 0;
        let have_right = x < w - 1;
        let mode = MODE_CONV[mode][have_left as usize][have_top as usize];
        let (needs_left, needs_top, needs_topleft, needs_topright, invert_left) = EDGES[mode];
        let n = 4usize << tx;
        let plane = &self.cur.planes[p];
        let stride = plane.width;
        let base = 128i32 << (bpp - 8);
        let mut top = [0i32; 65];
        let mut left = [0i32; 32];
        if needs_top {
            let n_px_need = n;
            let n_px_have = ((((self.cols - col) << (1 - ss_h)) as isize - x as isize) * 4).max(0) as usize;
            let n_px_need_tr = if tx == TX_4X4 && needs_topright && have_right { 4 } else { 0 };
            let at = |xx: usize| plane.data[(py - 1) * stride + xx] as i32;
            if have_top && (!needs_topleft || have_left) && (tx != TX_4X4 || !needs_topright || have_right) && n_px_need + n_px_need_tr <= n_px_have {
                // ffmpeg reads the row above in place (`*a = top`).
                for i in 0..n + n_px_need_tr {
                    top[1 + i] = at(px + i);
                }
                if needs_topleft {
                    top[0] = at(px - 1);
                }
                if tx == TX_4X4 && needs_topright && n_px_need_tr == 0 {
                    // Not reached: the in-place path requires have_right.
                    for i in 4..8 {
                        top[1 + i] = top[4];
                    }
                }
            } else {
                if have_top {
                    if n_px_need <= n_px_have {
                        for i in 0..n_px_need {
                            top[1 + i] = at(px + i);
                        }
                    } else {
                        for i in 0..n_px_have {
                            top[1 + i] = at(px + i);
                        }
                        let v = top[n_px_have];
                        for i in n_px_have..n_px_need {
                            top[1 + i] = v;
                        }
                    }
                } else {
                    for i in 0..n_px_need {
                        top[1 + i] = base - 1;
                    }
                }
                if needs_topleft {
                    top[0] = if have_left && have_top { at(px - 1) } else { base + if have_top { 1 } else { -1 } };
                }
                if tx == TX_4X4 && needs_topright {
                    if have_top && have_right && n_px_need + n_px_need_tr <= n_px_have {
                        for i in 4..8 {
                            top[1 + i] = at(px + i);
                        }
                    } else {
                        for i in 4..8 {
                            top[1 + i] = top[4];
                        }
                    }
                }
            }
        }
        if needs_left {
            if have_left {
                let n_px_need = n;
                let n_px_have = ((((self.rows - row) << (1 - ss_v)) as isize - y as isize) * 4).max(0) as usize;
                let lv = |i: usize| plane.data[(py + i) * stride + px - 1] as i32;
                if invert_left {
                    let k = n_px_need.min(n_px_have);
                    for i in 0..k {
                        left[i] = lv(i);
                    }
                    for i in k..n_px_need {
                        left[i] = left[k - 1];
                    }
                } else if n_px_need <= n_px_have {
                    for i in 0..n_px_need {
                        left[n_px_need - 1 - i] = lv(i);
                    }
                } else {
                    for i in 0..n_px_have {
                        left[n_px_need - 1 - i] = lv(i);
                    }
                    let v = left[n_px_need - n_px_have];
                    for i in 0..n_px_need - n_px_have {
                        left[i] = v;
                    }
                }
            } else {
                for v in left.iter_mut().take(n) {
                    *v = base + 1;
                }
            }
        }
        let plane = &mut self.cur.planes[p];
        let stride = plane.width;
        dsp::intra_pred(tx, mode, &mut plane.data, py * stride + px, stride, &left, &top, bpp);
    }

    /// `intra_recon`.
    pub(super) fn intra_recon(&mut self) {
        let b = self.b;
        let (row, col) = (self.row, self.col);
        let bs = b.bs as usize;
        let lossless = self.h.lossless as usize;
        let mut w4 = (BWH_TAB[1][bs][0] as usize) << 1;
        let step1d = 1usize << b.tx;
        let step = 1usize << (b.tx * 2);
        let mut end_x = (2 * (self.cols - col)).min(w4);
        let mut end_y = (2 * (self.rows - row)).min((BWH_TAB[1][bs][1] as usize) << 1);
        let tx = 4 * lossless + b.tx as usize;
        let uvtx = b.uvtx as usize + 4 * lossless;
        let bpp = self.bpp;
        let (x0, y0) = (col * 8, row * 8);
        let mut n = 0;
        let mut y = 0;
        while y < end_y {
            let mut x = 0;
            while x < end_x {
                let mode = b.mode[if b.bs > BS_8X8 && b.tx == TX_4X4 { y * 2 + x } else { 0 }] as usize;
                let txtp = INTRA_TXFM_TYPE[mode] as usize;
                let eob = if b.skip { 0 } else { self.eob[n] as usize };
                let (px, py) = (x0 + 4 * x, y0 + 4 * y);
                self.intra_block(mode, 0, px, py, x, y, w4, b.tx);
                if eob != 0 {
                    let pl = &mut self.cur.planes[0];
                    let stride = pl.width;
                    dsp::itxfm_add(tx, txtp, &mut pl.data, py * stride + px, stride, &mut self.block[16 * n..], eob, bpp);
                }
                x += step1d;
                n += step;
            }
            y += step1d;
        }
        // U/V.
        w4 >>= self.ss_h;
        end_x >>= self.ss_h;
        end_y >>= self.ss_v;
        let uvstep1d = 1usize << b.uvtx;
        let uvstep = 1usize << (b.uvtx * 2);
        let (cx0, cy0) = ((col * 8) >> self.ss_h, (row * 8) >> self.ss_v);
        for p in 0..2 {
            let mut n = 0;
            let mut y = 0;
            while y < end_y {
                let mut x = 0;
                while x < end_x {
                    let eob = if b.skip { 0 } else { self.uveob[p][n] as usize };
                    let (px, py) = (cx0 + 4 * x, cy0 + 4 * y);
                    self.intra_block(b.uvmode as usize, p + 1, px, py, x, y, w4, b.uvtx);
                    if eob != 0 {
                        let pl = &mut self.cur.planes[1 + p];
                        let stride = pl.width;
                        dsp::itxfm_add(uvtx, DCT_DCT, &mut pl.data, py * stride + px, stride, &mut self.uvblock[p][16 * n..], eob, bpp);
                    }
                    x += uvstep1d;
                    n += uvstep;
                }
                y += uvstep1d;
            }
        }
    }

    /// One luma prediction: `mc_luma_unscaled` / `mc_luma_scaled`. `(x, y)`
    /// is the block's sample position, `(px, py, pw, ph)` its sub-block
    /// offset and the whole block size (for the scaled MV clamp).
    #[allow(clippy::too_many_arguments)]
    fn mc_luma(&mut self, refi: usize, x: usize, y: usize, mv: Mv, px: i32, py: i32, pw: i32, ph: i32, bw: usize, bh: usize, avg: bool) {
        let rf = self.refs[self.b.r#ref[refi] as usize].clone().unwrap();
        let (w, h) = (rf.width as usize, rf.height as usize);
        let filter = self.b.filter;
        let bpp = self.bpp;
        let dst = &mut self.cur.planes[0];
        let dstride = dst.width;
        let doff = y * dstride + x;
        let scale = self.mvscale[self.b.r#ref[refi] as usize];
        let mut win = Vec::new();
        if scale[0] == 0 {
            let (mx, my) = (mv.x as i32, mv.y as i32);
            let yy = y as isize + (my >> 3) as isize;
            let xx = x as isize + (mx >> 3) as isize;
            let (mx, my) = ((mx & 7) << 1, (my & 7) << 1);
            dsp::fetch(&rf.planes[0], xx - 3, yy - 3, bw + 8, bh + 8, w, h, &mut win);
            dsp::mc_put(&mut dst.data, doff, dstride, &win, bw + 8, bw, bh, filter, mx, my, avg, bpp);
        } else {
            let step = self.mvstep[self.b.r#ref[refi] as usize];
            let sm = |n: i64, d: usize| ((n * scale[d] as i64) >> 14) as i32;
            let (x, y) = (x as i32, y as i32);
            let mvx = (mv.x as i32).clamp(-(x + pw - px + 4) * 8, (self.cols as i32 * 8 - x + px + 3) * 8);
            let mvy = (mv.y as i32).clamp(-(y + ph - py + 4) * 8, (self.rows as i32 * 8 - y + py + 3) * 8);
            // BUG (libvpx-compatible): the components scale separately.
            let mx = sm(mvx as i64 * 2, 0) + sm(x as i64 * 16, 0);
            let my = sm(mvy as i64 * 2, 1) + sm(y as i64 * 16, 1);
            let (xx, yy) = (mx >> 4, my >> 4);
            let (mx, my) = (mx & 15, my & 15);
            let refbw_m1 = (((bw as i32 - 1) * step[0] as i32 + mx) >> 4) as usize;
            let refbh_m1 = (((bh as i32 - 1) * step[1] as i32 + my) >> 4) as usize;
            dsp::fetch(&rf.planes[0], xx as isize - 3, yy as isize - 3, refbw_m1 + 8, refbh_m1 + 8, w, h, &mut win);
            dsp::mc_scaled(&mut dst.data, doff, dstride, &win, refbw_m1 + 8, bw, bh, filter, mx, my, step[0] as i32, step[1] as i32, avg, bpp);
        }
    }

    /// One chroma prediction (both planes): `mc_chroma_unscaled` /
    /// `mc_chroma_scaled`. `(x, y)` in chroma samples; `w`, `h` the
    /// reference's chroma size.
    #[allow(clippy::too_many_arguments)]
    fn mc_chroma(&mut self, refi: usize, x: usize, y: usize, mv: Mv, px: i32, py: i32, pw: i32, ph: i32, bw: usize, bh: usize, avg: bool) {
        let rf = self.refs[self.b.r#ref[refi] as usize].clone().unwrap();
        let (ss_h, ss_v) = (self.ss_h, self.ss_v);
        let w = ((rf.width as usize) + ss_h) >> ss_h;
        let h = ((rf.height as usize) + ss_v) >> ss_v;
        let filter = self.b.filter;
        let bpp = self.bpp;
        let scale = self.mvscale[self.b.r#ref[refi] as usize];
        let mut win = Vec::new();
        if scale[0] == 0 {
            let mx = mv.x as i32 * (1 << (1 - ss_h));
            let my = mv.y as i32 * (1 << (1 - ss_v));
            let yy = y as isize + (my >> 4) as isize;
            let xx = x as isize + (mx >> 4) as isize;
            let (mx, my) = (mx & 15, my & 15);
            for p in 1..3 {
                dsp::fetch(&rf.planes[p], xx - 3, yy - 3, bw + 8, bh + 8, w, h, &mut win);
                let dst = &mut self.cur.planes[p];
                let dstride = dst.width;
                dsp::mc_put(&mut dst.data, y * dstride + x, dstride, &win, bw + 8, bw, bh, filter, mx, my, avg, bpp);
            }
        } else {
            let step = self.mvstep[self.b.r#ref[refi] as usize];
            let sm = |n: i64, d: usize| ((n * scale[d] as i64) >> 14) as i32;
            let (x, y) = (x as i32, y as i32);
            let mx = if ss_h != 0 {
                // BUG (libvpx-compatible, webm issue 820).
                let mvx = (mv.x as i32).clamp(-(x + pw - px + 4) * 16, (self.cols as i32 * 4 - x + px + 3) * 16);
                sm(mvx as i64, 0) + (sm(x as i64 * 16, 0) & !15) + (sm(x as i64 * 32, 0) & 15)
            } else {
                let mvx = (mv.x as i32).clamp(-(x + pw - px + 4) * 8, (self.cols as i32 * 8 - x + px + 3) * 8);
                sm(mvx as i64 * 2, 0) + sm(x as i64 * 16, 0)
            };
            let my = if ss_v != 0 {
                let mvy = (mv.y as i32).clamp(-(y + ph - py + 4) * 16, (self.rows as i32 * 4 - y + py + 3) * 16);
                sm(mvy as i64, 1) + (sm(y as i64 * 16, 1) & !15) + (sm(y as i64 * 32, 1) & 15)
            } else {
                let mvy = (mv.y as i32).clamp(-(y + ph - py + 4) * 8, (self.rows as i32 * 8 - y + py + 3) * 8);
                sm(mvy as i64 * 2, 1) + sm(y as i64 * 16, 1)
            };
            let (xx, yy) = (mx >> 4, my >> 4);
            let (mx, my) = (mx & 15, my & 15);
            let refbw_m1 = (((bw as i32 - 1) * step[0] as i32 + mx) >> 4) as usize;
            let refbh_m1 = (((bh as i32 - 1) * step[1] as i32 + my) >> 4) as usize;
            for p in 1..3 {
                dsp::fetch(&rf.planes[p], xx as isize - 3, yy as isize - 3, refbw_m1 + 8, refbh_m1 + 8, w, h, &mut win);
                let dst = &mut self.cur.planes[p];
                let dstride = dst.width;
                dsp::mc_scaled(&mut dst.data, y as usize * dstride + x as usize, dstride, &win, refbw_m1 + 8, bw, bh, filter, mx, my, step[0] as i32, step[1] as i32, avg, bpp);
            }
        }
    }

    /// `inter_pred` (unscaled and scaled templates).
    fn inter_pred(&mut self) {
        let b = self.b;
        let (row, col) = (self.row, self.col);
        let (ss_h, ss_v) = (self.ss_h, self.ss_v);
        let scaled = self.mvscale[b.r#ref[0] as usize][0] != 0 || (b.comp && self.mvscale[b.r#ref[1] as usize][0] != 0);
        let refs = if b.comp { 2 } else { 1 };
        if b.bs > BS_8X8 {
            for i in 0..refs {
                let avg = i == 1;
                let mv = |k: usize| b.mv[k][i];
                if b.bs == BS_8X4 && !scaled {
                    self.mc_luma(i, col * 8, row * 8, mv(0), 0, 0, 0, 0, 8, 4, avg);
                    self.mc_luma(i, col * 8, row * 8 + 4, mv(2), 0, 0, 0, 0, 8, 4, avg);
                    if ss_v != 0 {
                        let uv = avg2(mv(0), mv(2));
                        self.mc_chroma(i, (col * 8) >> ss_h, row * 4, uv, 0, 0, 0, 0, 8 >> ss_h, 4, avg);
                    } else {
                        self.mc_chroma(i, (col * 8) >> ss_h, row * 8, mv(0), 0, 0, 0, 0, 8 >> ss_h, 4, avg);
                        // BUG (libvpx, webm issue 993): 4:2:2 8x8 bottom
                        // block's chroma uses the wrong motion vector.
                        let uv = if ss_h == 0 { mv(2) } else { avg2(mv(0), mv(2)) };
                        self.mc_chroma(i, (col * 8) >> ss_h, row * 8 + 4, uv, 0, 0, 0, 0, 8 >> ss_h, 4, avg);
                    }
                } else if b.bs == BS_4X8 && !scaled {
                    self.mc_luma(i, col * 8, row * 8, mv(0), 0, 0, 0, 0, 4, 8, avg);
                    self.mc_luma(i, col * 8 + 4, row * 8, mv(1), 0, 0, 0, 0, 4, 8, avg);
                    if ss_h != 0 {
                        let uv = avg2(mv(0), mv(1));
                        self.mc_chroma(i, col * 4, (row * 8) >> ss_v, uv, 0, 0, 0, 0, 4, 8 >> ss_v, avg);
                    } else {
                        self.mc_chroma(i, col * 8, (row * 8) >> ss_v, mv(0), 0, 0, 0, 0, 4, 8 >> ss_v, avg);
                        self.mc_chroma(i, col * 8 + 4, (row * 8) >> ss_v, mv(1), 0, 0, 0, 0, 4, 8 >> ss_v, avg);
                    }
                } else {
                    // 4×4 (and, when scaled, 8×4 / 4×8).
                    let (x8, y8) = (col * 8, row * 8);
                    self.mc_luma(i, x8, y8, mv(0), 0, 0, 8, 8, 4, 4, avg);
                    self.mc_luma(i, x8 + 4, y8, mv(1), 4, 0, 8, 8, 4, 4, avg);
                    self.mc_luma(i, x8, y8 + 4, mv(2), 0, 4, 8, 8, 4, 4, avg);
                    self.mc_luma(i, x8 + 4, y8 + 4, mv(3), 4, 4, 8, 8, 4, 4, avg);
                    if ss_v != 0 {
                        if ss_h != 0 {
                            let uv = avg4(mv(0), mv(1), mv(2), mv(3));
                            self.mc_chroma(i, col * 4, row * 4, uv, 0, 0, 4, 4, 4, 4, avg);
                        } else {
                            let uv = avg2(mv(0), mv(2));
                            self.mc_chroma(i, col * 8, row * 4, uv, 0, 0, 8, 4, 4, 4, avg);
                            let uv = avg2(mv(1), mv(3));
                            self.mc_chroma(i, col * 8 + 4, row * 4, uv, 4, 0, 8, 4, 4, 4, avg);
                        }
                    } else if ss_h != 0 {
                        let uv = avg2(mv(0), mv(1));
                        self.mc_chroma(i, col * 4, row * 8, uv, 0, 0, 4, 8, 4, 4, avg);
                        // BUG (libvpx, webm issue 993).
                        let uv = avg2(mv(1), mv(2));
                        self.mc_chroma(i, col * 4, row * 8 + 4, uv, 0, 4, 4, 8, 4, 4, avg);
                    } else {
                        self.mc_chroma(i, col * 8, row * 8, mv(0), 0, 0, 8, 8, 4, 4, avg);
                        self.mc_chroma(i, col * 8 + 4, row * 8, mv(1), 4, 0, 8, 8, 4, 4, avg);
                        self.mc_chroma(i, col * 8, row * 8 + 4, mv(2), 0, 4, 8, 8, 4, 4, avg);
                        self.mc_chroma(i, col * 8 + 4, row * 8 + 4, mv(3), 4, 4, 8, 8, 4, 4, avg);
                    }
                }
            }
        } else {
            let bs = b.bs as usize;
            let bw = BWH_TAB[0][bs][0] as usize * 4;
            let bh = BWH_TAB[0][bs][1] as usize * 4;
            let uvbw = BWH_TAB[ss_h][bs][0] as usize * 4;
            let uvbh = BWH_TAB[ss_v][bs][1] as usize * 4;
            for i in 0..refs {
                let avg = i == 1;
                let mv = b.mv[0][i];
                self.mc_luma(i, col * 8, row * 8, mv, 0, 0, bw as i32, bh as i32, bw, bh, avg);
                self.mc_chroma(i, (col * 8) >> ss_h, (row * 8) >> ss_v, mv, 0, 0, uvbw as i32, uvbh as i32, uvbw, uvbh, avg);
            }
        }
    }

    /// `inter_recon`.
    pub(super) fn inter_recon(&mut self) {
        let b = self.b;
        if self.mvscale[b.r#ref[0] as usize][0] == 0xFFFF || (b.comp && self.mvscale[b.r#ref[1] as usize][0] == 0xFFFF) {
            return;
        }
        self.inter_pred();
        if b.skip {
            return;
        }
        let (row, col) = (self.row, self.col);
        let bs = b.bs as usize;
        let w4 = (BWH_TAB[1][bs][0] as usize) << 1;
        let h4 = (BWH_TAB[1][bs][1] as usize) << 1;
        let step1d = 1usize << b.tx;
        let step = 1usize << (b.tx * 2);
        let mut end_x = (2 * (self.cols - col)).min(w4);
        let mut end_y = (2 * (self.rows - row)).min(h4);
        let lossless = self.h.lossless as usize;
        let tx = 4 * lossless + b.tx as usize;
        let uvtx = b.uvtx as usize + 4 * lossless;
        let bpp = self.bpp;
        let (x0, y0) = (col * 8, row * 8);
        let mut n = 0;
        let mut y = 0;
        while y < end_y {
            let mut x = 0;
            while x < end_x {
                let eob = self.eob[n] as usize;
                if eob != 0 {
                    let pl = &mut self.cur.planes[0];
                    let stride = pl.width;
                    dsp::itxfm_add(tx, DCT_DCT, &mut pl.data, (y0 + 4 * y) * stride + x0 + 4 * x, stride, &mut self.block[16 * n..], eob, bpp);
                }
                x += step1d;
                n += step;
            }
            y += step1d;
        }
        end_x >>= self.ss_h;
        end_y >>= self.ss_v;
        let uvstep1d = 1usize << b.uvtx;
        let uvstep = 1usize << (b.uvtx * 2);
        let (cx0, cy0) = ((col * 8) >> self.ss_h, (row * 8) >> self.ss_v);
        for p in 0..2 {
            let mut n = 0;
            let mut y = 0;
            while y < end_y {
                let mut x = 0;
                while x < end_x {
                    let eob = self.uveob[p][n] as usize;
                    if eob != 0 {
                        let pl = &mut self.cur.planes[1 + p];
                        let stride = pl.width;
                        dsp::itxfm_add(uvtx, DCT_DCT, &mut pl.data, (cy0 + 4 * y) * stride + cx0 + 4 * x, stride, &mut self.uvblock[p][16 * n..], eob, bpp);
                    }
                    x += uvstep1d;
                    n += uvstep;
                }
                y += uvstep1d;
            }
        }
    }
}
