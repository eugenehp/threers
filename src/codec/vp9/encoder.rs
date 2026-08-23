//! VP9 intra encoder with 4×4 DCT residual coding.
//!
//! Recursively `PARTITION_SPLIT`s each 64×64 superblock down to 8×8 (including
//! partial edge superblocks), picks a DC/V/H mode by SAD, then transform-codes
//! each 4×4 residual with the matching hybrid TX. Intra prediction is per-TX.
//! A conformant decoder must reproduce our reconstruction bit-for-bit
//! (see `tests/webm.rs`).
//!
//! Frame width/height must be multiples of 8 (MI size) and at least 8; 4:2:0
//! also needs even chroma dims, which that implies.

use crate::codec::bitstream::BitWriter;
use crate::codec::hevc::Yuv420Frame;
use crate::codec::vp9::bool_coder::BoolEncoder;
use crate::codec::vp9::intra::{predict, sad};
use crate::codec::vp9::quant::{ac_quant, dc_quant, quantize};
use crate::codec::vp9::tables::{
    bsl_from_px, partition_ctx_index, DC_PRED, DEFAULT_SKIP_PROBS, INTRA_MODE_TREE,
    KF_PARTITION_PROBS, KF_UV_MODE_PROB, KF_Y_MODE_PROB, PARTITION_CONTEXT_LOOKUP, PARTITION_NONE,
    PARTITION_SPLIT, PARTITION_TREE, PRED_MODES,
};
use crate::codec::vp9::tokens::write_coefs;
use crate::codec::vp9::transform::{fht4x4, iht4x4_add, tx_type_from_mode, TxType};

const FRAME_SYNC_CODE: u32 = 0x0049_8342;
const CS_BT_601: u32 = 1;
const MAX_TILE_WIDTH_SB: u32 = 64;
/// Must match `base_q_idx` in the uncompressed header.
const QINDEX: usize = 128;

/// Reconstruction a conformant decoder produces.
#[derive(Clone, Debug)]
pub struct Reconstruction {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

/// Encode one VP9 intra keyframe (4×4 DCT residual). Returns `(bitstream, recon)`.
///
/// `frame.width` / `height` must be ≥ 8 and multiples of 8.
pub fn encode_intra_frame(frame: &Yuv420Frame) -> (Vec<u8>, Reconstruction) {
    assert!(
        frame.width >= 8 && frame.height >= 8 && frame.width.is_multiple_of(8) && frame.height.is_multiple_of(8),
        "VP9 encoder needs dims that are multiples of 8 (got {}×{})",
        frame.width,
        frame.height
    );

    let mut enc = FrameEnc::new(frame);
    let compressed = compressed_header();
    let tile = enc.encode_tile();

    let mut wb = BitWriter::new();
    write_uncompressed_header(&mut wb, frame.width, frame.height, compressed.len() as u32);
    let mut out = wb.finish();
    out.extend_from_slice(&compressed);
    out.extend_from_slice(&tile);

    let recon = Reconstruction {
        width: frame.width,
        height: frame.height,
        y: enc.recon_y,
        u: enc.recon_u,
        v: enc.recon_v,
    };
    (crate::codec::vp9::finalize_frame(out), recon)
}

struct MiInfo {
    mode: i8,
    skip: bool,
}

struct FrameEnc<'a> {
    src: &'a Yuv420Frame,
    mi_cols: usize,
    mi_rows: usize,
    recon_y: Vec<u8>,
    recon_u: Vec<u8>,
    recon_v: Vec<u8>,
    mi: Vec<MiInfo>,
    above_ctx: Vec<u8>,
    left_ctx: [u8; 8],
    /// Y entropy contexts: 2 per MI column (4×4 TX).
    above_ent_y: Vec<u8>,
    left_ent_y: [u8; 16],
    above_ent_u: Vec<u8>,
    left_ent_u: [u8; 8],
    above_ent_v: Vec<u8>,
    left_ent_v: [u8; 8],
    bool: BoolEncoder,
    dc_q: i32,
    ac_q: i32,
}

impl<'a> FrameEnc<'a> {
    fn new(src: &'a Yuv420Frame) -> Self {
        let (w, h) = (src.width as usize, src.height as usize);
        // VP9 MI grid is 8×8; dims are required to be multiples of 8 so this
        // matches `(size + 7) >> 3` without padding.
        let mi_cols = w / 8;
        let mi_rows = h / 8;
        let cw = w / 2;
        let ch = h / 2;
        Self {
            src,
            mi_cols,
            mi_rows,
            recon_y: vec![0; w * h],
            recon_u: vec![128; cw * ch],
            recon_v: vec![128; cw * ch],
            mi: (0..mi_cols * mi_rows)
                .map(|_| MiInfo {
                    mode: DC_PRED,
                    skip: true,
                })
                .collect(),
            above_ctx: vec![0; mi_cols],
            left_ctx: [0; 8],
            above_ent_y: vec![0; mi_cols * 2],
            left_ent_y: [0; 16],
            above_ent_u: vec![0; mi_cols],
            left_ent_u: [0; 8],
            above_ent_v: vec![0; mi_cols],
            left_ent_v: [0; 8],
            bool: BoolEncoder::new(),
            dc_q: dc_quant(QINDEX),
            ac_q: ac_quant(QINDEX),
        }
    }

    fn encode_tile(&mut self) -> Vec<u8> {
        let mut mi_row = 0;
        while mi_row < self.mi_rows {
            self.left_ctx = [0; 8];
            self.left_ent_y = [0; 16];
            self.left_ent_u = [0; 8];
            self.left_ent_v = [0; 8];
            let mut mi_col = 0;
            while mi_col < self.mi_cols {
                self.encode_partition(mi_row, mi_col, 64);
                mi_col += 8; // next superblock column (may be partial)
            }
            mi_row += 8;
        }
        std::mem::take(&mut self.bool).finish()
    }

    fn encode_partition(&mut self, mi_row: usize, mi_col: usize, bsize_px: u32) {
        if mi_row >= self.mi_rows || mi_col >= self.mi_cols {
            return;
        }
        let mi_half = (bsize_px as usize / 8) / 2;
        let has_rows = mi_row + mi_half < self.mi_rows;
        let has_cols = mi_col + mi_half < self.mi_cols;

        // Match libvpx `read_partition`: when neither half fits, SPLIT is forced
        // with no bits (recursive children no-op out of bounds). Never NONE above
        // 8×8 — our residual path only emits 8×8 blocks.
        let partition = if bsize_px == 8 {
            PARTITION_NONE
        } else if has_rows && has_cols {
            PARTITION_SPLIT
        } else if has_cols || has_rows {
            PARTITION_SPLIT
        } else {
            PARTITION_SPLIT
        };

        self.write_partition(mi_row, mi_col, bsize_px, has_rows, has_cols, partition);

        if bsize_px == 8 {
            self.encode_block(mi_row, mi_col, bsize_px);
            self.update_partition_context(mi_row, mi_col, bsize_px, bsize_px);
            return;
        }

        let sub_px = bsize_px / 2;
        // SPLIT into four (OOB recursive calls return immediately).
        self.encode_partition(mi_row, mi_col, sub_px);
        self.encode_partition(mi_row, mi_col + mi_half, sub_px);
        self.encode_partition(mi_row + mi_half, mi_col, sub_px);
        self.encode_partition(mi_row + mi_half, mi_col + mi_half, sub_px);
    }

    fn write_partition(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bsize_px: u32,
        has_rows: bool,
        has_cols: bool,
        partition: i8,
    ) {
        let ctx = self.partition_context(mi_row, mi_col, bsize_px);
        let probs = &KF_PARTITION_PROBS[ctx];
        if has_rows && has_cols {
            write_tree(&mut self.bool, &PARTITION_TREE, probs, partition);
        } else if !has_rows && has_cols {
            // false = VERT, true = SPLIT (libvpx).
            self.bool.put_bool(probs[1], partition == PARTITION_SPLIT);
        } else if has_rows && !has_cols {
            self.bool.put_bool(probs[2], partition == PARTITION_SPLIT);
        }
        // else: neither half fits → SPLIT forced, no bits (libvpx).
    }

    fn partition_context(&self, mi_row: usize, mi_col: usize, bsize_px: u32) -> usize {
        let bsl = bsl_from_px(bsize_px);
        let above = (self.above_ctx[mi_col] >> bsl) & 1;
        let left = (self.left_ctx[mi_row & 7] >> bsl) & 1;
        bsl * 4 + (left as usize) * 2 + above as usize
    }

    fn update_partition_context(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        sub_px: u32,
        bsize_px: u32,
    ) {
        let bs = bsize_px as usize / 8;
        let (above_v, left_v) = PARTITION_CONTEXT_LOOKUP[partition_ctx_index(sub_px)];
        for i in 0..bs {
            if mi_col + i < self.mi_cols {
                self.above_ctx[mi_col + i] = above_v;
            }
            self.left_ctx[(mi_row + i) & 7] = left_v;
        }
    }

    fn encode_block(&mut self, mi_row: usize, mi_col: usize, bsize_px: u32) {
        debug_assert_eq!(bsize_px, 8, "residual path encodes 8×8 blocks only");
        let (px, py) = (mi_col * 8, mi_row * 8);
        let w = self.src.width as usize;
        let h = self.src.height as usize;
        let (cpx, cpy) = (px / 2, py / 2);
        let cw = w / 2;
        let ch = h / 2;

        let mi_above = mi_row > 0;
        let mi_left = mi_col > 0;
        let above_mode = if mi_above {
            self.mi[(mi_row - 1) * self.mi_cols + mi_col].mode
        } else {
            DC_PRED
        };
        let left_mode = if mi_left {
            self.mi[mi_row * self.mi_cols + (mi_col - 1)].mode
        } else {
            DC_PRED
        };

        // Mode decision on the 8×8 with outer neighbors (one mode for all TXs).
        let above_y = gather_above(&self.recon_y, w, px, py, 8, mi_above);
        let left_y = gather_left(&self.recon_y, w, h, px, py, 8, mi_left);
        let best_y = pick_mode(
            8,
            mi_above,
            mi_left,
            &above_y,
            &left_y,
            &self.src.y[py * w + px..],
            w,
        );
        let y_tx = tx_type_from_mode(best_y);

        let above_u = gather_above(&self.recon_u, cw, cpx, cpy, 4, mi_above);
        let left_u = gather_left(&self.recon_u, cw, ch, cpx, cpy, 4, mi_left);
        let above_v = gather_above(&self.recon_v, cw, cpx, cpy, 4, mi_above);
        let left_v = gather_left(&self.recon_v, cw, ch, cpx, cpy, 4, mi_left);
        let best_uv = pick_uv_mode(
            4,
            mi_above,
            mi_left,
            &above_u,
            &left_u,
            &above_v,
            &left_v,
            &self.src.u[cpy * cw + cpx..],
            &self.src.v[cpy * cw + cpx..],
            cw,
        );
        // UV plane always uses DCT_DCT transform/scan (libvpx `get_tx_type`).
        let uv_tx = TxType::DctDct;

        // Pass 1: per-TX predict → hybrid TX → quant.
        let mut y_q = [[0i32; 16]; 4];
        let mut y_dq = [[0i32; 16]; 4];
        let mut any = false;
        let mut local_y = [0u8; 64];

        for ty in 0..2 {
            for tx in 0..2 {
                let ti = ty * 2 + tx;
                let (have_a, have_l) = (mi_above || ty > 0, mi_left || tx > 0);
                let (above, left) =
                    neighbors_y(&self.recon_y, &local_y, w, px, py, tx, ty, have_a, have_l);
                let mut pred = [0u8; 16];
                predict(best_y, 4, have_a, have_l, &above, &left, &mut pred);
                let residual = residual_packed4(&pred, &self.src.y, w, px + tx * 4, py + ty * 4);
                let mut coeff = [0i32; 16];
                fht4x4(&residual, &mut coeff, y_tx);
                let (q, dq) = quantize(&coeff, QINDEX);
                any |= q.iter().any(|&c| c != 0);
                y_q[ti] = q;
                y_dq[ti] = dq;
                let mut block = pred;
                iht4x4_add(&dq, &mut block, 4, y_tx);
                store_local4(&mut local_y, tx, ty, &block);
            }
        }

        let (u_q, u_dq) = {
            let (have_a, have_l) = (mi_above, mi_left);
            let above = gather_above(&self.recon_u, cw, cpx, cpy, 4, have_a);
            let left = gather_left(&self.recon_u, cw, ch, cpx, cpy, 4, have_l);
            let mut pred = [0u8; 16];
            predict(best_uv, 4, have_a, have_l, &above, &left, &mut pred);
            let residual = residual_packed4(&pred, &self.src.u, cw, cpx, cpy);
            let mut coeff = [0i32; 16];
            fht4x4(&residual, &mut coeff, uv_tx);
            let (q, dq) = quantize(&coeff, QINDEX);
            any |= q.iter().any(|&c| c != 0);
            (q, dq)
        };
        let (v_q, v_dq) = {
            let (have_a, have_l) = (mi_above, mi_left);
            let above = gather_above(&self.recon_v, cw, cpx, cpy, 4, have_a);
            let left = gather_left(&self.recon_v, cw, ch, cpx, cpy, 4, have_l);
            let mut pred = [0u8; 16];
            predict(best_uv, 4, have_a, have_l, &above, &left, &mut pred);
            let residual = residual_packed4(&pred, &self.src.v, cw, cpx, cpy);
            let mut coeff = [0i32; 16];
            fht4x4(&residual, &mut coeff, uv_tx);
            let (q, dq) = quantize(&coeff, QINDEX);
            any |= q.iter().any(|&c| c != 0);
            (q, dq)
        };

        let skip = !any;
        let skip_ctx = self.skip_context(mi_row, mi_col);
        self.bool.put_bool(DEFAULT_SKIP_PROBS[skip_ctx], skip);
        write_tree(
            &mut self.bool,
            &INTRA_MODE_TREE,
            &KF_Y_MODE_PROB[above_mode as usize][left_mode as usize],
            best_y,
        );
        write_tree(
            &mut self.bool,
            &INTRA_MODE_TREE,
            &KF_UV_MODE_PROB[best_y as usize],
            best_uv,
        );

        // Pass 2: emit tokens (if any) and rebuild recon.
        for ty in 0..2 {
            for tx in 0..2 {
                let ti = ty * 2 + tx;
                let x = px + tx * 4;
                let y = py + ty * 4;
                let (have_a, have_l) = (mi_above || ty > 0, mi_left || tx > 0);
                let above = gather_above(&self.recon_y, w, x, y, 4, have_a);
                let left = gather_left(&self.recon_y, w, h, x, y, 4, have_l);
                let mut pred = [0u8; 16];
                predict(best_y, 4, have_a, have_l, &above, &left, &mut pred);

                if !skip {
                    let a = self.above_ent_y[mi_col * 2 + tx];
                    let l = self.left_ent_y[(mi_row & 7) * 2 + ty];
                    let ctx = (a != 0) as usize + (l != 0) as usize;
                    let eob = write_coefs(
                        &mut self.bool,
                        &y_q[ti],
                        true,
                        false,
                        ctx,
                        y_tx,
                        self.dc_q,
                        self.ac_q,
                    );
                    let has = (eob > 0) as u8;
                    self.above_ent_y[mi_col * 2 + tx] = has;
                    self.left_ent_y[(mi_row & 7) * 2 + ty] = has;
                    iht4x4_add(&y_dq[ti], &mut pred, 4, y_tx);
                }
                store_4x4(&mut self.recon_y, w, x, y, &pred);
            }
        }

        {
            let (have_a, have_l) = (mi_above, mi_left);
            let above = gather_above(&self.recon_u, cw, cpx, cpy, 4, have_a);
            let left = gather_left(&self.recon_u, cw, ch, cpx, cpy, 4, have_l);
            let mut pred = [0u8; 16];
            predict(best_uv, 4, have_a, have_l, &above, &left, &mut pred);
            if !skip {
                let a = self.above_ent_u[mi_col];
                let l = self.left_ent_u[mi_row & 7];
                let ctx = (a != 0) as usize + (l != 0) as usize;
                let eob = write_coefs(
                    &mut self.bool,
                    &u_q,
                    false,
                    false,
                    ctx,
                    uv_tx,
                    self.dc_q,
                    self.ac_q,
                );
                let has = (eob > 0) as u8;
                self.above_ent_u[mi_col] = has;
                self.left_ent_u[mi_row & 7] = has;
                iht4x4_add(&u_dq, &mut pred, 4, uv_tx);
            }
            store_4x4(&mut self.recon_u, cw, cpx, cpy, &pred);
        }
        {
            let (have_a, have_l) = (mi_above, mi_left);
            let above = gather_above(&self.recon_v, cw, cpx, cpy, 4, have_a);
            let left = gather_left(&self.recon_v, cw, ch, cpx, cpy, 4, have_l);
            let mut pred = [0u8; 16];
            predict(best_uv, 4, have_a, have_l, &above, &left, &mut pred);
            if !skip {
                let a = self.above_ent_v[mi_col];
                let l = self.left_ent_v[mi_row & 7];
                let ctx = (a != 0) as usize + (l != 0) as usize;
                let eob = write_coefs(
                    &mut self.bool,
                    &v_q,
                    false,
                    false,
                    ctx,
                    uv_tx,
                    self.dc_q,
                    self.ac_q,
                );
                let has = (eob > 0) as u8;
                self.above_ent_v[mi_col] = has;
                self.left_ent_v[mi_row & 7] = has;
                iht4x4_add(&v_dq, &mut pred, 4, uv_tx);
            }
            store_4x4(&mut self.recon_v, cw, cpx, cpy, &pred);
        }

        if skip {
            for t in 0..2 {
                self.above_ent_y[mi_col * 2 + t] = 0;
                self.left_ent_y[(mi_row & 7) * 2 + t] = 0;
            }
            self.above_ent_u[mi_col] = 0;
            self.left_ent_u[mi_row & 7] = 0;
            self.above_ent_v[mi_col] = 0;
            self.left_ent_v[mi_row & 7] = 0;
        }

        self.mi[mi_row * self.mi_cols + mi_col] = MiInfo { mode: best_y, skip };
    }

    fn skip_context(&self, mi_row: usize, mi_col: usize) -> usize {
        let above = if mi_row > 0 {
            self.mi[(mi_row - 1) * self.mi_cols + mi_col].skip as usize
        } else {
            0
        };
        let left = if mi_col > 0 {
            self.mi[mi_row * self.mi_cols + (mi_col - 1)].skip as usize
        } else {
            0
        };
        above + left
    }
}

fn gather_above(plane: &[u8], stride: usize, x: usize, y: usize, bs: usize, have: bool) -> Vec<u8> {
    let mut above = vec![127u8; bs];
    if have && y > 0 {
        let row = &plane[(y - 1) * stride..];
        for i in 0..bs {
            above[i] = row[(x + i).min(stride - 1)];
        }
    }
    above
}

fn gather_left(
    plane: &[u8],
    stride: usize,
    height: usize,
    x: usize,
    y: usize,
    bs: usize,
    have: bool,
) -> Vec<u8> {
    let mut left = vec![129u8; bs];
    if have && x > 0 {
        for i in 0..bs {
            let yy = (y + i).min(height - 1);
            left[i] = plane[yy * stride + (x - 1)];
        }
    }
    left
}

/// Above/left for a Y 4×4 inside an 8×8, reading prior TXs from `local` (8×8).
fn neighbors_y(
    recon: &[u8],
    local: &[u8; 64],
    stride: usize,
    px: usize,
    py: usize,
    tx: usize,
    ty: usize,
    have_a: bool,
    have_l: bool,
) -> ([u8; 4], [u8; 4]) {
    let mut above = [127u8; 4];
    let mut left = [129u8; 4];
    if have_a {
        if ty == 0 {
            if py > 0 {
                let row = &recon[(py - 1) * stride..];
                for i in 0..4 {
                    above[i] = row[px + tx * 4 + i];
                }
            }
        } else {
            for i in 0..4 {
                above[i] = local[(ty * 4 - 1) * 8 + tx * 4 + i];
            }
        }
    }
    if have_l {
        if tx == 0 {
            if px > 0 {
                for i in 0..4 {
                    left[i] = recon[(py + ty * 4 + i) * stride + (px - 1)];
                }
            }
        } else {
            for i in 0..4 {
                left[i] = local[(ty * 4 + i) * 8 + (tx * 4 - 1)];
            }
        }
    }
    (above, left)
}

fn store_local4(local: &mut [u8; 64], tx: usize, ty: usize, block: &[u8; 16]) {
    for r in 0..4 {
        local[(ty * 4 + r) * 8 + tx * 4..(ty * 4 + r) * 8 + tx * 4 + 4]
            .copy_from_slice(&block[r * 4..r * 4 + 4]);
    }
}

fn store_4x4(plane: &mut [u8], stride: usize, x: usize, y: usize, block: &[u8; 16]) {
    for r in 0..4 {
        plane[(y + r) * stride + x..(y + r) * stride + x + 4]
            .copy_from_slice(&block[r * 4..r * 4 + 4]);
    }
}

fn residual_packed4(
    pred: &[u8; 16],
    src: &[u8],
    src_stride: usize,
    x: usize,
    y: usize,
) -> [i16; 16] {
    let mut r = [0i16; 16];
    for row in 0..4 {
        for col in 0..4 {
            let s = src[(y + row) * src_stride + x + col] as i16;
            r[row * 4 + col] = s - pred[row * 4 + col] as i16;
        }
    }
    r
}

fn pick_mode(
    bs: usize,
    have_above: bool,
    have_left: bool,
    above: &[u8],
    left: &[u8],
    src: &[u8],
    src_stride: usize,
) -> i8 {
    let mut best_mode = DC_PRED;
    let mut best_sad = u32::MAX;
    for &mode in &PRED_MODES {
        let mut pred = vec![0u8; bs * bs];
        predict(mode, bs, have_above, have_left, above, left, &mut pred);
        let s = sad(src, src_stride, &pred, bs);
        if s < best_sad {
            best_sad = s;
            best_mode = mode;
        }
    }
    best_mode
}

fn pick_uv_mode(
    bs: usize,
    have_above: bool,
    have_left: bool,
    above_u: &[u8],
    left_u: &[u8],
    above_v: &[u8],
    left_v: &[u8],
    src_u: &[u8],
    src_v: &[u8],
    stride: usize,
) -> i8 {
    let mut best_mode = DC_PRED;
    let mut best_sad = u32::MAX;
    for &mode in &PRED_MODES {
        let mut pu = vec![0u8; bs * bs];
        let mut pv = vec![0u8; bs * bs];
        predict(mode, bs, have_above, have_left, above_u, left_u, &mut pu);
        predict(mode, bs, have_above, have_left, above_v, left_v, &mut pv);
        let s = sad(src_u, stride, &pu, bs) + sad(src_v, stride, &pv, bs);
        if s < best_sad {
            best_sad = s;
            best_mode = mode;
        }
    }
    best_mode
}

fn write_tree(e: &mut BoolEncoder, tree: &[i8], probs: &[u8], token: i8) {
    let path = tree_path(tree, 0, token).expect("token not in tree");
    for (node, bit) in path {
        e.put_bool(probs[node >> 1], bit);
    }
}

fn tree_path(tree: &[i8], node: usize, token: i8) -> Option<Vec<(usize, bool)>> {
    for bit in 0..2 {
        let child = tree[node + bit];
        if child <= 0 {
            if -child == token {
                return Some(vec![(node, bit == 1)]);
            }
        } else if let Some(mut rest) = tree_path(tree, child as usize, token) {
            rest.insert(0, (node, bit == 1));
            return Some(rest);
        }
    }
    None
}

fn compressed_header() -> Vec<u8> {
    let mut e = BoolEncoder::new();
    e.put_literal(0, 2); // tx_mode = ONLY_4X4
    e.put_bit(false); // coef update TX_4X4 = 0
    for _ in 0..3 {
        e.put_bool(252, false); // skip probs unchanged
    }
    e.finish()
}

fn write_uncompressed_header(w: &mut BitWriter, width: u32, height: u32, header_size: u32) {
    w.write_bits(2, 2);
    w.write_bit(0);
    w.write_bit(0);
    w.write_bit(0);
    w.write_bit(0);
    w.write_bit(1);
    w.write_bit(0);

    w.write_bits(FRAME_SYNC_CODE, 24);
    w.write_bits(CS_BT_601, 3);
    w.write_bit(0);

    w.write_bits(width - 1, 16);
    w.write_bits(height - 1, 16);
    w.write_bit(0);

    w.write_bit(0);
    w.write_bit(1);
    w.write_bits(0, 2);

    w.write_bits(0, 6);
    w.write_bits(0, 3);
    w.write_bit(0);

    w.write_bits(QINDEX as u32, 8); // base_q_idx
    w.write_bit(0);
    w.write_bit(0);
    w.write_bit(0);

    w.write_bit(0);

    write_tile_info(w, width);

    w.write_bits(header_size, 16);
}

fn write_tile_info(w: &mut BitWriter, width: u32) {
    // libvpx `vp9_get_tile_n_bits` / `write_tile_info`.
    const MIN_TILE_WIDTH_SB: u32 = 4;
    let mi_cols = width / 8;
    let sb_cols = mi_cols.div_ceil(8);
    let mut min_log2 = 0u32;
    while (MAX_TILE_WIDTH_SB << min_log2) < sb_cols {
        min_log2 += 1;
    }
    let mut max_log2 = 1u32;
    while (sb_cols >> max_log2) >= MIN_TILE_WIDTH_SB {
        max_log2 += 1;
    }
    max_log2 -= 1;
    // Always use log2_tile_cols = min_log2 (one tile): no leading ones, then a
    // terminating 0 only when min < max (room to request more columns).
    if min_log2 < max_log2 {
        w.write_bit(0);
    }
    w.write_bit(0); // tile_rows_log2 = 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn busy_frame_uses_directional_modes() {
        // A strong horizontal ramp should prefer H_PRED over DC for interior blocks.
        let mut f = Yuv420Frame::new(64, 64);
        for j in 0..64u32 {
            for i in 0..64u32 {
                f.y[(j * 64 + i) as usize] = (i * 4).min(255) as u8;
            }
        }
        let (_, recon) = encode_intra_frame(&f);
        // Recon should track the ramp better than flat mid-gray.
        let mut err = 0u64;
        for (a, b) in f.y.iter().zip(&recon.y) {
            err += a.abs_diff(*b) as u64;
        }
        let flat: u64 = f.y.iter().map(|&x| x.abs_diff(128) as u64).sum();
        assert!(err < flat / 3, "H-friendly ramp err={err} flat={flat}");
    }

    #[test]
    fn gray_frame_recons_to_mid_gray() {
        let mut f = Yuv420Frame::new(64, 64);
        f.y.fill(128);
        f.u.fill(128);
        f.v.fill(128);
        let (bits, recon) = encode_intra_frame(&f);
        assert!(bits.len() > 12);
        assert!(recon.y.iter().all(|&x| x == 128));
        assert!(recon.u.iter().all(|&x| x == 128));
        assert!(recon.v.iter().all(|&x| x == 128));
    }

    #[test]
    fn partial_superblock_sizes_encode() {
        for &(w, h) in &[(8u32, 8), (16, 8), (80, 48), (96, 72), (128, 64)] {
            let mut f = Yuv420Frame::new(w, h);
            for j in 0..h {
                for i in 0..w {
                    f.y[(j * w + i) as usize] = ((i + j) & 0xFF) as u8;
                }
            }
            let (bits, recon) = encode_intra_frame(&f);
            assert!(bits.len() > 8, "{w}x{h} bitstream too small");
            assert_eq!(recon.y.len(), (w * h) as usize);
            assert_eq!(recon.u.len(), (w * h / 4) as usize);
        }
    }

    #[test]
    fn busy_frame_residual_beats_mid_gray() {
        let mut f = Yuv420Frame::new(64, 64);
        for j in 0..64u32 {
            for i in 0..64u32 {
                f.y[(j * 64 + i) as usize] = ((i.wrapping_mul(3) ^ j.wrapping_mul(5)) & 0xFF) as u8;
            }
        }
        let (bits, recon) = encode_intra_frame(&f);
        assert!(
            bits.len() > 50,
            "residual bitstream unexpectedly tiny: {}",
            bits.len()
        );
        let mut err = 0u64;
        for (a, b) in f.y.iter().zip(&recon.y) {
            err += a.abs_diff(*b) as u64;
        }
        let flat: u64 = f.y.iter().map(|&x| x.abs_diff(128) as u64).sum();
        assert!(
            err < flat / 2,
            "residual recon err={err} should beat flat={flat}"
        );
        // Recon should use the full sample range, not just {127,128,129}.
        let min = *recon.y.iter().min().unwrap();
        let max = *recon.y.iter().max().unwrap();
        assert!(max - min > 10, "recon range too narrow: {min}..{max}");
    }
}
