//! VP9 probability contexts, symbol counts, and backward adaptation
//! (ffmpeg `vp9dec.h` ProbContext / counts, `vp9prob.c`).

use super::data::DEFAULT_COEF_PROBS;
use super::{CompPred, FilterMode, Header, DC_PRED, DIAG_DOWN_LEFT_PRED, DIAG_DOWN_RIGHT_PRED, HOR_DOWN_PRED, HOR_PRED, HOR_UP_PRED, TM_VP8_PRED, TX_SWITCHABLE, VERT_LEFT_PRED, VERT_PRED, VERT_RIGHT_PRED};

#[derive(Clone, Copy, Debug)]
pub struct MvCompProbs {
    pub sign: u8,
    pub classes: [u8; 10],
    pub class0: u8,
    pub bits: [u8; 10],
    pub class0_fp: [[u8; 3]; 2],
    pub fp: [u8; 3],
    pub class0_hp: u8,
    pub hp: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct ProbContext {
    pub y_mode: [[u8; 9]; 4],
    pub uv_mode: [[u8; 9]; 10],
    pub filter: [[u8; 2]; 4],
    pub mv_mode: [[u8; 3]; 7],
    pub intra: [u8; 4],
    pub comp: [u8; 5],
    pub single_ref: [[u8; 2]; 5],
    pub comp_ref: [u8; 5],
    pub tx32p: [[u8; 3]; 2],
    pub tx16p: [[u8; 2]; 2],
    pub tx8p: [u8; 2],
    pub skip: [u8; 3],
    pub mv_joint: [u8; 3],
    pub mv_comp: [MvCompProbs; 2],
    pub partition: [[[u8; 3]; 4]; 4],
}

/// `[tx][plane][inter][band][ctx][n]` coefficient probabilities or counts.
pub type CoefArr<T, const N: usize> = [[[[[[T; N]; 6]; 6]; 2]; 2]; 4];

/// A saved frame context: modes and the 3 coded coefficient probabilities.
#[derive(Clone)]
pub struct SavedCtx {
    pub p: ProbContext,
    pub coef: CoefArr<u8, 3>,
}

impl SavedCtx {
    pub fn default_ctx() -> Self {
        SavedCtx { p: super::data::DEFAULT_PROBS, coef: DEFAULT_COEF_PROBS }
    }
}

#[derive(Clone, Copy, Default)]
pub struct MvCompCounts {
    pub sign: [u32; 2],
    pub classes: [u32; 11],
    pub class0: [u32; 2],
    pub bits: [[u32; 2]; 10],
    pub class0_fp: [[u32; 4]; 2],
    pub fp: [u32; 4],
    pub class0_hp: [u32; 2],
    pub hp: [u32; 2],
}

#[derive(Clone)]
pub struct Counts {
    pub y_mode: [[u32; 10]; 4],
    pub uv_mode: [[u32; 10]; 10],
    pub filter: [[u32; 3]; 4],
    pub mv_mode: [[u32; 4]; 7],
    pub intra: [[u32; 2]; 4],
    pub comp: [[u32; 2]; 5],
    pub single_ref: [[[u32; 2]; 2]; 5],
    pub comp_ref: [[u32; 2]; 5],
    pub tx32p: [[u32; 4]; 2],
    pub tx16p: [[u32; 3]; 2],
    pub tx8p: [[u32; 2]; 2],
    pub skip: [[u32; 2]; 3],
    pub mv_joint: [u32; 4],
    pub mv_comp: [MvCompCounts; 2],
    pub partition: [[[u32; 4]; 4]; 4],
    pub coef: CoefArr<u32, 3>,
    pub eob: CoefArr<u32, 2>,
}

impl Counts {
    pub fn new() -> Box<Self> {
        Box::new(Counts {
            y_mode: [[0; 10]; 4],
            uv_mode: [[0; 10]; 10],
            filter: [[0; 3]; 4],
            mv_mode: [[0; 4]; 7],
            intra: [[0; 2]; 4],
            comp: [[0; 2]; 5],
            single_ref: [[[0; 2]; 2]; 5],
            comp_ref: [[0; 2]; 5],
            tx32p: [[0; 4]; 2],
            tx16p: [[0; 3]; 2],
            tx8p: [[0; 2]; 2],
            skip: [[0; 2]; 3],
            mv_joint: [0; 4],
            mv_comp: [MvCompCounts::default(); 2],
            partition: [[[0; 4]; 4]; 4],
            coef: [[[[[[0; 3]; 6]; 6]; 2]; 2]; 4],
            eob: [[[[[[0; 2]; 6]; 6]; 2]; 2]; 4],
        })
    }
}

/// `adapt_prob`.
#[inline]
fn adapt(p: &mut u8, ct0: u32, ct1: u32, max_count: u32, update_factor: u32) {
    let ct = ct0 + ct1;
    if ct == 0 {
        return;
    }
    let uf = (update_factor * ct.min(max_count) / max_count) as i64;
    let p1 = *p as i64;
    let p2 = (((ct0 as i64) << 8) + (ct as i64 >> 1)) / ct as i64;
    let p2 = p2.clamp(1, 255);
    *p = (p1 + (((p2 - p1) * uf + 128) >> 8)) as u8;
}

/// `ff_vp9_adapt_probs`: fold this frame's counts into the saved context
/// `ctx` (the frame's `framectxid`). `cur` is the probability set the frame
/// was decoded with; `last_keyframe` whether the previous frame was a key.
pub fn adapt_probs(ctx: &mut SavedCtx, cur: &ProbContext, counts: &Counts, h: &Header, last_keyframe: bool) {
    let uf = if h.keyframe || h.intraonly || !last_keyframe { 112 } else { 128 };
    for i in 0..4 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..6 {
                    for m in 0..6 {
                        if l == 0 && m >= 3 {
                            break;
                        }
                        let pp = &mut ctx.coef[i][j][k][l][m];
                        let e = &counts.eob[i][j][k][l][m];
                        let c = &counts.coef[i][j][k][l][m];
                        adapt(&mut pp[0], e[0], e[1], 24, uf);
                        adapt(&mut pp[1], c[0], c[1] + c[2], 24, uf);
                        adapt(&mut pp[2], c[1], c[2], 24, uf);
                    }
                }
            }
        }
    }
    let p = &mut ctx.p;
    if h.keyframe || h.intraonly {
        p.skip = cur.skip;
        p.tx32p = cur.tx32p;
        p.tx16p = cur.tx16p;
        p.tx8p = cur.tx8p;
        return;
    }
    for i in 0..3 {
        adapt(&mut p.skip[i], counts.skip[i][0], counts.skip[i][1], 20, 128);
    }
    for i in 0..4 {
        adapt(&mut p.intra[i], counts.intra[i][0], counts.intra[i][1], 20, 128);
    }
    if h.comppredmode == CompPred::Switchable {
        for i in 0..5 {
            adapt(&mut p.comp[i], counts.comp[i][0], counts.comp[i][1], 20, 128);
        }
    }
    if h.comppredmode != CompPred::Single {
        for i in 0..5 {
            adapt(&mut p.comp_ref[i], counts.comp_ref[i][0], counts.comp_ref[i][1], 20, 128);
        }
    }
    if h.comppredmode != CompPred::Comp {
        for i in 0..5 {
            let c = &counts.single_ref[i];
            adapt(&mut p.single_ref[i][0], c[0][0], c[0][1], 20, 128);
            adapt(&mut p.single_ref[i][1], c[1][0], c[1][1], 20, 128);
        }
    }
    for i in 0..4 {
        for j in 0..4 {
            let c = &counts.partition[i][j];
            let pp = &mut p.partition[i][j];
            adapt(&mut pp[0], c[0], c[1] + c[2] + c[3], 20, 128);
            adapt(&mut pp[1], c[1], c[2] + c[3], 20, 128);
            adapt(&mut pp[2], c[2], c[3], 20, 128);
        }
    }
    if h.txfmmode == TX_SWITCHABLE {
        for i in 0..2 {
            let (c16, c32) = (&counts.tx16p[i], &counts.tx32p[i]);
            adapt(&mut p.tx8p[i], counts.tx8p[i][0], counts.tx8p[i][1], 20, 128);
            adapt(&mut p.tx16p[i][0], c16[0], c16[1] + c16[2], 20, 128);
            adapt(&mut p.tx16p[i][1], c16[1], c16[2], 20, 128);
            adapt(&mut p.tx32p[i][0], c32[0], c32[1] + c32[2] + c32[3], 20, 128);
            adapt(&mut p.tx32p[i][1], c32[1], c32[2] + c32[3], 20, 128);
            adapt(&mut p.tx32p[i][2], c32[2], c32[3], 20, 128);
        }
    }
    if h.filtermode == FilterMode::Switchable {
        for i in 0..4 {
            let c = &counts.filter[i];
            adapt(&mut p.filter[i][0], c[0], c[1] + c[2], 20, 128);
            adapt(&mut p.filter[i][1], c[1], c[2], 20, 128);
        }
    }
    for i in 0..7 {
        let c = &counts.mv_mode[i];
        let pp = &mut p.mv_mode[i];
        adapt(&mut pp[0], c[2], c[1] + c[0] + c[3], 20, 128);
        adapt(&mut pp[1], c[0], c[1] + c[3], 20, 128);
        adapt(&mut pp[2], c[1], c[3], 20, 128);
    }
    {
        let c = &counts.mv_joint;
        adapt(&mut p.mv_joint[0], c[0], c[1] + c[2] + c[3], 20, 128);
        adapt(&mut p.mv_joint[1], c[1], c[2] + c[3], 20, 128);
        adapt(&mut p.mv_joint[2], c[2], c[3], 20, 128);
    }
    for i in 0..2 {
        let cc = &counts.mv_comp[i];
        let pc = &mut p.mv_comp[i];
        adapt(&mut pc.sign, cc.sign[0], cc.sign[1], 20, 128);
        let c = &cc.classes;
        let pp = &mut pc.classes;
        let mut sum = c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7] + c[8] + c[9] + c[10];
        adapt(&mut pp[0], c[0], sum, 20, 128);
        sum -= c[1];
        adapt(&mut pp[1], c[1], sum, 20, 128);
        sum -= c[2] + c[3];
        adapt(&mut pp[2], c[2] + c[3], sum, 20, 128);
        adapt(&mut pp[3], c[2], c[3], 20, 128);
        sum -= c[4] + c[5];
        adapt(&mut pp[4], c[4] + c[5], sum, 20, 128);
        adapt(&mut pp[5], c[4], c[5], 20, 128);
        sum -= c[6];
        adapt(&mut pp[6], c[6], sum, 20, 128);
        adapt(&mut pp[7], c[7] + c[8], c[9] + c[10], 20, 128);
        adapt(&mut pp[8], c[7], c[8], 20, 128);
        adapt(&mut pp[9], c[9], c[10], 20, 128);
        adapt(&mut pc.class0, cc.class0[0], cc.class0[1], 20, 128);
        for j in 0..10 {
            adapt(&mut pc.bits[j], cc.bits[j][0], cc.bits[j][1], 20, 128);
        }
        for j in 0..2 {
            let c = &cc.class0_fp[j];
            let pp = &mut pc.class0_fp[j];
            adapt(&mut pp[0], c[0], c[1] + c[2] + c[3], 20, 128);
            adapt(&mut pp[1], c[1], c[2] + c[3], 20, 128);
            adapt(&mut pp[2], c[2], c[3], 20, 128);
        }
        let c = &cc.fp;
        adapt(&mut pc.fp[0], c[0], c[1] + c[2] + c[3], 20, 128);
        adapt(&mut pc.fp[1], c[1], c[2] + c[3], 20, 128);
        adapt(&mut pc.fp[2], c[2], c[3], 20, 128);
        if h.highprecisionmvs {
            adapt(&mut pc.class0_hp, cc.class0_hp[0], cc.class0_hp[1], 20, 128);
            adapt(&mut pc.hp, cc.hp[0], cc.hp[1], 20, 128);
        }
    }
    let modes = |pp: &mut [u8; 9], c: &[u32; 10]| {
        let mut sum = c[0] + c[1] + c[3] + c[4] + c[5] + c[6] + c[7] + c[8] + c[9];
        adapt(&mut pp[0], c[DC_PRED], sum, 20, 128);
        sum -= c[TM_VP8_PRED];
        adapt(&mut pp[1], c[TM_VP8_PRED], sum, 20, 128);
        sum -= c[VERT_PRED];
        adapt(&mut pp[2], c[VERT_PRED], sum, 20, 128);
        let mut s2 = c[HOR_PRED] + c[DIAG_DOWN_RIGHT_PRED] + c[VERT_RIGHT_PRED];
        sum -= s2;
        adapt(&mut pp[3], s2, sum, 20, 128);
        s2 -= c[HOR_PRED];
        adapt(&mut pp[4], c[HOR_PRED], s2, 20, 128);
        adapt(&mut pp[5], c[DIAG_DOWN_RIGHT_PRED], c[VERT_RIGHT_PRED], 20, 128);
        sum -= c[DIAG_DOWN_LEFT_PRED];
        adapt(&mut pp[6], c[DIAG_DOWN_LEFT_PRED], sum, 20, 128);
        sum -= c[VERT_LEFT_PRED];
        adapt(&mut pp[7], c[VERT_LEFT_PRED], sum, 20, 128);
        adapt(&mut pp[8], c[HOR_DOWN_PRED], c[HOR_UP_PRED], 20, 128);
    };
    for i in 0..4 {
        modes(&mut p.y_mode[i], &counts.y_mode[i]);
    }
    for i in 0..10 {
        modes(&mut p.uv_mode[i], &counts.uv_mode[i]);
    }
}
