//! CDF context set-up and end-of-frame update (dav1d `src/cdf.c`).

pub use super::cdf_data::*;

impl CdfContext {
    /// `dav1d_cdf_thread_copy` from the static tables for `qidx`.
    pub fn default_for(qidx: u8) -> Box<CdfContext> {
        let qcat = (qidx > 20) as usize + (qidx > 60) as usize + (qidx > 120) as usize;
        Box::new(CdfContext {
            coef: DEFAULT_COEF[qcat],
            m: DEFAULT_MODE,
            mv: CdfMvContext { comp: [DEFAULT_MV_COMP; 2], joint: DEFAULT_MV_JOINT },
            kfym: DEFAULT_KFYM,
        })
    }

    /// `dav1d_cdf_thread_update`: `self` starts as a copy of the frame's
    /// input CDFs; take the adapted state of `src` (the context tile) for
    /// the fields dav1d saves, with their counters reset.
    pub fn update_from(&mut self, src: &CdfContext, key_or_intra: bool) {
        macro_rules! cp {
            ($($f:ident).+ ; $n:expr) => {
                self.$($f).+ = src.$($f).+;
                zero(&mut self.$($f).+, $n);
            };
        }
        // Everything up to `m.intrabc` is copied.
        cp!(coef.eob_bin_16; 4);
        cp!(coef.eob_bin_32; 5);
        cp!(coef.eob_bin_64; 6);
        cp!(coef.eob_bin_128; 7);
        cp!(coef.eob_bin_256; 8);
        cp!(coef.eob_bin_512; 9);
        cp!(coef.eob_bin_1024; 10);
        cp!(coef.eob_base_tok; 2);
        cp!(coef.base_tok; 3);
        cp!(coef.br_tok; 3);
        cp!(coef.eob_hi_bit; 1);
        cp!(coef.skip; 1);
        cp!(coef.dc_sign; 1);

        // uv_mode: N_UV_INTRA_PRED_MODES - 1 - !k symbols.
        self.m.uv_mode = src.m.uv_mode;
        for (k, a) in self.m.uv_mode.iter_mut().enumerate() {
            zero(a, 13 - (k == 0) as usize);
        }
        self.m.partition = src.m.partition;
        for (bl, p) in self.m.partition.iter_mut().enumerate() {
            let n = match bl {
                0 => 7,
                4 => 3,
                _ => 9,
            };
            zero(p, n);
        }
        cp!(m.cfl_alpha; 15);
        cp!(m.txtp_inter1; 15);
        cp!(m.txtp_inter2; 11);
        cp!(m.txtp_intra1; 6);
        cp!(m.txtp_intra2; 4);
        cp!(m.cfl_sign; 7);
        cp!(m.angle_delta; 6);
        cp!(m.filter_intra; 4);
        cp!(m.seg_id; 7);
        cp!(m.pal_sz; 6);
        self.m.color_map = src.m.color_map;
        // The counter position follows the palette size (second index).
        for a in self.m.color_map.iter_mut() {
            for (k, b) in a.iter_mut().enumerate() {
                zero(b, k + 1);
            }
        }
        self.m.txsz = src.m.txsz;
        for (k, a) in self.m.txsz.iter_mut().enumerate() {
            zero(a, (k + 1).min(2));
        }
        cp!(m.delta_q; 3);
        cp!(m.delta_lf; 3);
        cp!(m.restore_switchable; 2);
        cp!(m.restore_wiener; 1);
        cp!(m.restore_sgrproj; 1);
        cp!(m.txtp_inter3; 1);
        cp!(m.use_filter_intra; 1);
        cp!(m.txpart; 1);
        cp!(m.skip; 1);
        cp!(m.pal_y; 1);
        cp!(m.pal_uv; 1);

        if key_or_intra {
            return;
        }

        // `m.y_mode` up to `kfym` (intrabc and kfym are never saved).
        cp!(m.y_mode; 12);
        cp!(m.wedge_idx; 15);
        cp!(m.comp_inter_mode; 7);
        cp!(m.filter; 2);
        cp!(m.interintra_mode; 3);
        cp!(m.motion_mode; 2);
        cp!(m.skip_mode; 1);
        cp!(m.newmv_mode; 1);
        cp!(m.globalmv_mode; 1);
        cp!(m.refmv_mode; 1);
        cp!(m.drl_bit; 1);
        cp!(m.intra; 1);
        cp!(m.comp; 1);
        cp!(m.comp_dir; 1);
        cp!(m.jnt_comp; 1);
        cp!(m.mask_comp; 1);
        cp!(m.wedge_comp; 1);
        cp!(m.r#ref; 1);
        cp!(m.comp_fwd_ref; 1);
        cp!(m.comp_bwd_ref; 1);
        cp!(m.comp_uni_ref; 1);
        cp!(m.seg_pred; 1);
        cp!(m.interintra; 1);
        cp!(m.interintra_wedge; 1);
        cp!(m.obmc; 1);
        for k in 0..2 {
            let (d, s) = (&mut self.mv.comp[k], &src.mv.comp[k]);
            *d = *s;
            zero(&mut d.classes, 10);
            zero(&mut d.sign, 1);
            zero(&mut d.class0, 1);
            zero(&mut d.class0_fp, 3);
            zero(&mut d.class0_hp, 1);
            zero(&mut d.classN, 1);
            zero(&mut d.classN_fp, 3);
            zero(&mut d.classN_hp, 1);
        }
        cp!(mv.joint; 3);
    }
}

/// Reset the adaptation counter (`cdf[n]`) of every innermost CDF.
fn zero<T: CdfArray + ?Sized>(a: &mut T, n: usize) {
    a.zero_counter(n);
}

pub trait CdfArray {
    fn zero_counter(&mut self, n: usize);
}

impl<const N: usize> CdfArray for [u16; N] {
    fn zero_counter(&mut self, n: usize) {
        self[n] = 0;
    }
}

impl<T, const M: usize, const N: usize> CdfArray for [[T; M]; N]
where
    [T; M]: CdfArray,
{
    fn zero_counter(&mut self, n: usize) {
        for x in self.iter_mut() {
            x.zero_counter(n);
        }
    }
}
