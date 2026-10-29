//! What ffmpeg n7.1's matroska demuxer and `avformat_find_stream_info`
//! report for a VP9 WebM/Matroska video, as [`MovVideo`] timing for the
//! pipe (`libavformat/matroskadec.c`, `demux.c`):
//!
//! - time base `TimestampScale / 1e9` (1 ms by default); packet pts = block
//!   timestamps.
//! - With `DefaultDuration` (or the deprecated `FrameRate`): `avg_frame_rate = 1e9 / DefaultDuration`
//!   (reduced to a denominator ≤ 30000), also `r_frame_rate` when between 5
//!   and 1000; packet durations `DefaultDuration / TimestampScale`
//!   (truncated), or the `BlockDuration`.
//! - Without: `find_stream_info` reads until 40 durations are seen (the
//!   time base is coarser than 0.5 ms, doubling the usual 20) or 5 s of
//!   them after 30 frames, then `ff_rfps_calculate`. `r_frame_rate` falls
//!   back to `1 / time base`; `avg_frame_rate` copies `r_frame_rate` when
//!   the mean step agrees within one tick. Packet durations left at zero
//!   become `1 / r_frame_rate`, rounded down.
//! - `Duration:` is the segment `Duration`, else the last packet's end;
//!   `start` the earliest first packet of any track.

use super::mov::{banner_fps, rescale, rfps, MovVideo, Orientation};
use crate::mkv_demux::{Mkv, TrackKind};
use crate::mp4_demux::{Sample, TrackCodec, VideoTrack};

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// `av_reduce(num, den, max)`: the best approximation with both terms ≤ max
/// (exact when the reduced fraction fits, which is the case that matters).
fn av_reduce(num: u64, den: u64, max: u64) -> (i64, i64) {
    let g = gcd(num, den).max(1);
    let (mut n, mut d) = (num / g, den / g);
    if n <= max && d <= max {
        return (n as i64, d as i64);
    }
    // Continued-fraction convergents (libavutil/rational.c).
    let (mut a0n, mut a0d, mut a1n, mut a1d) = (0u64, 1u64, 1u64, 0u64);
    while d != 0 {
        let x = n / d;
        let next_d = n - d * x;
        let a2n = x * a1n + a0n;
        let a2d = x * a1d + a0d;
        if a2n > max || a2d > max {
            let mut x2 = x;
            if let Some(v) = (max - a0n).checked_div(a1n) {
                x2 = v;
            }
            if let Some(v) = (max - a0d).checked_div(a1d) {
                x2 = x2.min(v);
            }
            // libavutil compares against the current remainders.
            if d * (2 * x2 * a1d + a0d) > n * a1d {
                a1n = x2 * a1n + a0n;
                a1d = x2 * a1d + a0d;
            }
            break;
        }
        a0n = a1n;
        a0d = a1d;
        a1n = a2n;
        a1d = a2d;
        n = d;
        d = next_d;
    }
    (a1n as i64, a1d as i64)
}

/// Parse a Matroska/WebM file: the video timing and its packets, laid out
/// back to back in the returned buffer (`VideoTrack::samples` index it).
pub fn parse(data: &[u8]) -> Result<(MovVideo, VideoTrack, Vec<u8>), String> {
    let mkv = Mkv::parse(data)?;
    let t = mkv.track(TrackKind::Video).ok_or("no video track")?;
    let codec = match t.codec_id.as_str() {
        "V_VP9" => TrackCodec::Vp9,
        "V_AV1" => TrackCodec::Av1,
        other => return Err(format!("Matroska video codec {other} is not supported")),
    };
    if 1_000_000_000 % mkv.timestamp_scale != 0 {
        return Err(format!("TimestampScale {} is not supported", mkv.timestamp_scale));
    }
    let ts = (1_000_000_000 / mkv.timestamp_scale) as i64;
    let frames: Vec<_> = mkv.frames_of(t.number).collect();
    if frames.is_empty() {
        return Err("video track has no frames".into());
    }
    // Packets that show a picture (a lone hidden frame shows none, and the
    // CLI never sees a frame for it).
    let shows: Vec<bool> = frames
        .iter()
        .map(|f| codec != TrackCodec::Vp9 || crate::decode::vp9_packet::packet_shows(&f.data))
        .collect();
    let all_pts: Vec<i64> = frames.iter().map(|f| f.pts).collect();
    let pts: Vec<i64> = frames.iter().zip(&shows).filter(|(_, &s)| s).map(|(f, _)| f.pts).collect();
    let shown: Vec<usize> = (0..frames.len()).filter(|&i| shows[i]).collect();

    // Rates.
    // matroskadec: without a DefaultDuration, the deprecated FrameRate
    // gives one (`1e9 / frame_rate`, truncated).
    let default_duration = t.default_duration.filter(|&d| d > 0).or_else(|| {
        t.frame_rate.filter(|&f| f > 0.0).map(|f| 1e9 / f).filter(|&d| d < u64::MAX as f64).map(|d| d as u64)
    });
    let mut r_frame_rate = None;
    let mut avg = None;
    if let Some(dd) = default_duration {
        let a = av_reduce(1_000_000_000, dd, 30000);
        avg = Some(a);
        if a.0 < a.1 * 1000 && a.0 > a.1 * 5 {
            r_frame_rate = Some(a);
        }
    }
    let probed = {
        // The dts find_stream_info feeds ff_rfps_add_frame.
        let mut n = 0;
        let mut count = 0;
        while n < all_pts.len() && count < 40 {
            if n > 30 && rescale(all_pts[n] - all_pts[0], 1_000_000, ts) >= 5_000_000 {
                break;
            }
            if n > 0 && all_pts[n] > all_pts[n - 1] {
                count += 1;
            }
            n += 1;
        }
        n
    };
    if r_frame_rate.is_none() {
        r_frame_rate = rfps(&all_pts[..probed], ts);
    }
    let r = r_frame_rate.unwrap_or((ts, 1));
    if avg.is_none() {
        // "Setting avg frame rate based on r frame rate".
        let steps: Vec<i64> = all_pts[..probed].windows(2).filter(|w| w[1] > w[0]).map(|w| w[1] - w[0]).collect();
        if steps.len() > 2 && r_frame_rate.is_some() {
            let mean = steps.iter().sum::<i64>() as f64 / steps.len() as f64;
            if ((ts as f64 * r.1 as f64 / r.0 as f64) - mean).abs() <= 1.0 {
                avg = Some(r);
            }
        }
    }
    let fps = avg.unwrap_or(r);

    // Packet durations.
    let dd_ticks = default_duration.map(|d| (d / mkv.timestamp_scale) as i64).unwrap_or(0);
    let fallback = (ts * r.1) / r.0;
    let pkt_durations: Vec<i64> = frames
        .iter()
        .map(|f| {
            let d = f.duration.filter(|&d| d > 0).unwrap_or(dd_ticks);
            if d > 0 {
                d
            } else {
                fallback
            }
        })
        .collect();

    let durations: Vec<i64> = shown.iter().map(|&i| pkt_durations[i]).collect();

    // Start and duration.
    let start = mkv
        .tracks
        .iter()
        .filter_map(|tr| mkv.frames_of(tr.number).next().map(|f| f.pts))
        .min()
        .unwrap_or(0);
    let start_us = rescale(start, 1_000_000, ts);
    let dur_us = match mkv.duration {
        Some(d) => (d * mkv.timestamp_scale as f64 * 1000.0 / 1_000_000.0) as i64,
        None => rescale(all_pts.last().unwrap() + pkt_durations.last().unwrap() - start, 1_000_000, ts),
    };
    let banner_duration = ((dur_us + 5000) / 10_000) as f64 / 100.0;

    // Packets back to back.
    let mut buf = Vec::new();
    let mut samples = Vec::with_capacity(frames.len());
    for (i, f) in frames.iter().enumerate() {
        samples.push(Sample {
            offset: buf.len() as u64,
            size: f.data.len() as u32,
            duration: pkt_durations[i] as u32,
            dts: f.pts,
            pts: f.pts,
            sync: f.keyframe,
        });
        buf.extend_from_slice(&f.data);
    }
    let n = frames.len();
    if pts.is_empty() {
        return Err("no frame of the video track is shown".into());
    }
    let mov = MovVideo {
        codec,
        timescale: ts as u32,
        pts,
        durations,
        sample: shown,
        decode_range: (0, n),
        start_us,
        r_frame_rate: r,
        banner_fps: banner_fps(fps.0 as f64 / fps.1 as f64),
        banner_duration,
        n_streams: mkv.tracks.len(),
        has_ctts: false,
        orientation: Orientation::Identity,
        width: t.width,
        height: t.height,
    };
    let track = VideoTrack { codec, width: t.width, height: t.height, config: Vec::new(), timescale: ts as u32, samples };
    Ok((mov, track, buf))
}
