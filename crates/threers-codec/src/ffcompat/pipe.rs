//! Which decoded frames the ffmpeg n7.1 CLI writes for
//! `[-ss A -i f -ss B] -f image2pipe -pix_fmt rgb24 -` (moviepy's reader),
//! without running ffmpeg. Ported from fftools: input timestamps shifted by
//! `-(A + start)`, accurate-seek input trim at 0, output trim at `B`,
//! `adjust_frame_pts_to_encoder_tb`, and `video_sync_process` in VSCFR
//! (single-stream input) or CFR mode at `1 / r_frame_rate`, including the
//! end-of-stream median padding. Matched ffmpeg's real pipe on 180 of 180
//! (file, seek) sequences (`/Users/Shared/tribev2-ref/scripts/sim_cfr.py`).

use super::mov::{rescale, MovVideo};

/// Indices into `v.pts` (presentation order) of the frames ffmpeg outputs,
/// for input/output seeks `-ss A -i f -ss B` given as `(A, B)` in
/// microseconds (`None`: no seek).
pub fn pipe_frames(v: &MovVideo, seek: Option<(i64, i64)>) -> Vec<usize> {
    let ts = v.timescale as i64;
    let (fr_num, fr_den) = v.r_frame_rate;
    // tb_out = 1 / r_frame_rate = fr_den / fr_num.
    let (tb_num, tb_den) = (fr_den, fr_num);
    let vscfr = v.n_streams == 1;
    let (a_us, b_us) = seek.unwrap_or((0, 0));
    let in_off_us = -(a_us + v.start_us);
    let shift = rescale(in_off_us, ts, 1_000_000);
    let fpts: Vec<i64> = v.pts.iter().map(|p| p + shift).collect();
    let mut sel: Vec<usize> = (0..fpts.len()).collect();
    if seek.is_some() {
        sel.retain(|&i| fpts[i] >= 0); // input trim (accurate seek)
        let b = rescale(b_us, ts, 1_000_000);
        sel.retain(|&i| fpts[i] >= b); // output -ss trim
    }
    let out_ts_off = b_us;
    let extra = (29 - (64 - (tb_den as u64).leading_zeros() as i64 - 1)).clamp(0, 16);
    let den_x = tb_den << extra;

    let mut next_pts: i64 = 0;
    let mut frame_number: i64 = 0;
    let mut hist = [0i64; 3];
    let mut out = Vec::new();
    let mut prev: Option<usize> = None;
    for &i in &sel {
        // adjust_frame_pts_to_encoder_tb(frame, tb_out, ts_offset)
        let mut float_pts = (rescale(fpts[i], den_x, ts * tb_num)
            - rescale(out_ts_off, den_x, 1_000_000 * tb_num)) as f64
            / (1i64 << extra) as f64;
        if float_pts != float_pts.round_ties_even() {
            float_pts += float_pts.signum() / (1 << 17) as f64;
        }
        let mut duration = v.durations[i] as f64 / ts as f64 / (tb_num as f64 / tb_den as f64);
        let mut sync_ipts = float_pts;
        let mut delta0 = sync_ipts - next_pts as f64;
        let mut delta = delta0 + duration;
        let mut nb_prev = 0i64;
        let mut nb = 1i64;
        if delta0 < 0.0 && delta > 0.0 {
            sync_ipts = next_pts as f64;
            duration += delta0;
            delta0 = 0.0;
        }
        if vscfr && frame_number == 0 && delta0 >= 0.5 {
            delta = duration;
            delta0 = 0.0;
            next_pts = sync_ipts.round_ties_even() as i64;
        }
        if delta < -1.1 {
            nb = 0;
        } else if delta > 1.1 {
            nb = delta.round_ties_even() as i64;
            if delta0 > 1.1 {
                nb_prev = (delta0 - 0.6).round_ties_even() as i64;
            }
        }
        hist = [nb_prev, hist[0], hist[1]];
        for k in 0..nb {
            let src = match prev {
                Some(p) if k < nb_prev => p,
                _ => i,
            };
            out.push(src);
            frame_number += 1;
            next_pts += 1;
        }
        prev = Some(i);
    }
    // EOF: mid_pred of the last three nb_frames_prev.
    let mut h = hist;
    h.sort_unstable();
    if let Some(p) = prev {
        out.extend(std::iter::repeat_n(p, h[1].max(0) as usize));
    }
    out
}
