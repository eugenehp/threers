//! What ffmpeg n7.1's mov demuxer reports for an H.264 or HEVC MP4/MOV, computed from
//! the boxes: the values moviepy reads from ffmpeg's banner (rounded fps,
//! centisecond duration, size), and the timing of every frame the CLI hands
//! its filter graph. Each rule below was fitted against ffmpeg 7.1 on 532
//! real clips (`/Users/Shared/tribev2-ref/scripts/{fit_banner,fit_tbr}.py`).
//!
//! - `start`: the empty edits, rescaled to the track timescale.
//! - `avg_frame_rate` (banner `fps`): `n_samples * timescale / sum(stts)`.
//! - `r_frame_rate` (banner `tbr`, the CLI's output rate): `timescale /
//!   stts[0]` for single-rate tracks, else `ff_rfps_calculate` over the first
//!   99 index dts (mov feeds them while building its index).
//! - duration: `start + sum(stts)`.
//! - last packet duration: `st->duration - dts`, with `st->duration =
//!   min(mdhd, sum(stts), sum(edits))`; if not positive, one frame period of
//!   the SPS VUI frame rate.
//! - with a `ctts` (B-frames): `mov_fix_index` (the keyframe search before the
//!   edit's media time, discarded frames, rewritten dts, `min_corrected_pts`),
//!   packet pts `dts + dts_shift + ctts`, and packet durations left to
//!   `ff_compute_frame_duration` (`1 / r_frame_rate`).

// Ported loop structure is kept as in the C reference.
#![allow(clippy::needless_range_loop, clippy::manual_memcpy, clippy::unnecessary_cast, clippy::manual_is_multiple_of)]

type Result<T> = std::result::Result<T, String>;

trait Context<T> {
    fn context(self, msg: &str) -> Result<T>;
}
impl<T> Context<T> for Option<T> {
    fn context(self, msg: &str) -> Result<T> {
        self.ok_or_else(|| msg.to_string())
    }
}

/// The video track as ffmpeg sees it.
#[derive(Debug, Clone)]
pub struct MovVideo {
    /// The codec of the sample entry.
    pub codec: crate::mp4_demux::TrackCodec,
    /// Ticks per second for `pts` and `durations`.
    pub timescale: u32,
    /// Frames the decoder outputs (discarded ones removed), in presentation
    /// order: pts in the track timescale including the empty-edit shift
    /// (ffmpeg's frame pts before the CLI's offsets).
    pub pts: Vec<i64>,
    /// Frame durations as ffmpeg's demuxer sets them.
    pub durations: Vec<i64>,
    /// For each output frame, its sample index (decode order, file order).
    pub sample: Vec<usize>,
    /// Samples `[from, to)` the demuxer delivers (decode order); frames in
    /// that range missing from [`sample`](Self::sample) are decoded and
    /// discarded.
    pub decode_range: (usize, usize),
    /// `ic->start_time` in microseconds (the earliest track start).
    pub start_us: i64,
    /// `r_frame_rate` (num, den).
    pub r_frame_rate: (i64, i64),
    /// `avg_frame_rate` as moviepy parses it from the banner (`%.2f`-rounded).
    pub banner_fps: f64,
    /// Banner `Duration:` in seconds (centiseconds).
    pub banner_duration: f64,
    /// Number of tracks (ffmpeg streams); 1 selects VSCFR in the CLI.
    pub n_streams: usize,
    /// Whether the track has a `ctts` (composition offsets, so B-frames).
    pub has_ctts: bool,
    /// The CLI's autorotate filters for the composed tkhd·mvhd display matrix.
    pub orientation: Orientation,
    /// Coded width from the sample entry.
    pub width: u32,
    /// Coded height from the sample entry.
    pub height: u32,
}

struct Boxes<'a> {
    d: &'a [u8],
}

impl<'a> Boxes<'a> {
    /// `(path, payload_start, end)` for the containers TRIBE needs.
    fn walk(&self, off: usize, end: usize, path: &str, out: &mut Vec<(String, usize, usize)>) {
        let d = self.d;
        let mut off = off;
        while off + 8 <= end {
            let mut size = u32::from_be_bytes(d[off..off + 4].try_into().unwrap()) as usize;
            let kind = String::from_utf8_lossy(&d[off + 4..off + 8]).into_owned();
            let mut hdr = 8;
            if size == 1 && off + 16 <= end {
                size = u64::from_be_bytes(d[off + 8..off + 16].try_into().unwrap()) as usize;
                hdr = 16;
            } else if size == 0 {
                size = end - off;
            }
            if size < hdr || off + size > end {
                break;
            }
            let p = format!("{path}/{kind}");
            out.push((p.clone(), off + hdr, off + size));
            if matches!(kind.as_str(), "moov" | "trak" | "mdia" | "minf" | "stbl" | "edts") {
                self.walk(off + hdr, off + size, &p, out);
            }
            if kind == "stsd" && off + hdr + 8 <= off + size {
                // sample entries: skip version/flags + count, then avc1 (78-byte header) children.
                let mut e = off + hdr + 8;
                while e + 8 <= off + size {
                    let esz = u32::from_be_bytes(d[e..e + 4].try_into().unwrap()) as usize;
                    if esz < 8 || e + esz > off + size {
                        break;
                    }
                    let ek = String::from_utf8_lossy(&d[e + 4..e + 8]).into_owned();
                    out.push((format!("{p}/{ek}"), e + 8, e + esz));
                    if matches!(ek.as_str(), "avc1" | "avc3" | "hvc1" | "hev1" | "vp09" | "av01") {
                        self.walk(e + 8 + 78, e + esz, &format!("{p}/{ek}"), out);
                    }
                    e += esz;
                }
            }
            off += size;
        }
    }
}

fn be32(d: &[u8], a: usize) -> u32 {
    u32::from_be_bytes(d[a..a + 4].try_into().unwrap())
}
fn be64(d: &[u8], a: usize) -> u64 {
    u64::from_be_bytes(d[a..a + 8].try_into().unwrap())
}

/// `(timescale, duration)` of mvhd/mdhd (version 0/1).
fn time_header(d: &[u8], a: usize) -> (u64, u64) {
    if d[a] == 1 {
        (be32(d, a + 20) as u64, be64(d, a + 24))
    } else {
        (be32(d, a + 12) as u64, be32(d, a + 16) as u64)
    }
}

/// av_rescale(a, b, c) with AV_ROUND_NEAR_INF (half away from zero).
pub fn rescale(a: i64, b: i64, c: i64) -> i64 {
    let n = a as i128 * b as i128;
    let c = c as i128;
    (if n >= 0 { (n + c / 2) / c } else { -((-n + c / 2) / c) }) as i64
}

// ---- SPS VUI timing ----------------------------------------------------

struct Bits<'a> {
    b: &'a [u8],
    p: usize,
}
impl Bits<'_> {
    fn u(&mut self, n: u32) -> Result<u64> {
        let mut v = 0u64;
        for _ in 0..n {
            let byte = *self.b.get(self.p >> 3).context("SPS truncated")?;
            v = (v << 1) | ((byte >> (7 - (self.p & 7))) & 1) as u64;
            self.p += 1;
        }
        Ok(v)
    }
    fn ue(&mut self) -> Result<u64> {
        let mut z = 0;
        while self.u(1)? == 0 {
            z += 1;
            if z > 31 {
                return Err("bad exp-golomb".into());
            }
        }
        Ok((1u64 << z) - 1 + self.u(z)?)
    }
    fn se(&mut self) -> Result<i64> {
        let k = self.ue()? as i64;
        Ok(if k & 1 == 1 { (k + 1) / 2 } else { -(k / 2) })
    }
}

fn unescape(n: &[u8]) -> Vec<u8> {
    let mut o = Vec::with_capacity(n.len());
    let mut zeros = 0;
    for &c in n {
        if zeros >= 2 && c == 3 {
            zeros = 0;
            continue;
        }
        o.push(c);
        zeros = if c == 0 { zeros + 1 } else { 0 };
    }
    o
}

/// What ffmpeg takes from an SPS VUI.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vui {
    /// `(num_units_in_tick, time_scale)`.
    pub timing: Option<(u64, u64)>,
    /// `matrix_coefficients` (2 = unspecified when absent).
    pub matrix: u32,
    /// `video_full_range_flag`.
    pub full_range: bool,
}

/// `(num_units_in_tick, time_scale)` from an SPS NAL (with header byte).
pub fn sps_vui_timing(sps: &[u8]) -> Result<Option<(u64, u64)>> {
    Ok(sps_vui(sps)?.timing)
}

/// The VUI fields ffmpeg uses, from an SPS NAL (with header byte).
pub fn sps_vui(sps: &[u8]) -> Result<Vui> {
    let mut vui = Vui { timing: None, matrix: 2, full_range: false };
    let rbsp = unescape(&sps[1..]);
    let mut r = Bits { b: &rbsp, p: 0 };
    let profile = r.u(8)?;
    r.u(16)?;
    r.ue()?;
    if matches!(profile, 100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135) {
        let cf = r.ue()?;
        if cf == 3 {
            r.u(1)?;
        }
        r.ue()?;
        r.ue()?;
        r.u(1)?;
        if r.u(1)? == 1 {
            for i in 0..(if cf != 3 { 8 } else { 12 }) {
                if r.u(1)? == 1 {
                    let (mut last, mut next) = (8i64, 8i64);
                    for _ in 0..(if i < 6 { 16 } else { 64 }) {
                        if next != 0 {
                            next = (last + r.se()? + 256) % 256;
                        }
                        last = if next == 0 { last } else { next };
                    }
                }
            }
        }
    }
    r.ue()?;
    let poc = r.ue()?;
    if poc == 0 {
        r.ue()?;
    } else if poc == 1 {
        r.u(1)?;
        r.se()?;
        r.se()?;
        for _ in 0..r.ue()? {
            r.se()?;
        }
    }
    r.ue()?;
    r.u(1)?;
    r.ue()?;
    r.ue()?;
    if r.u(1)? == 0 {
        r.u(1)?;
    }
    r.u(1)?;
    if r.u(1)? == 1 {
        r.ue()?;
        r.ue()?;
        r.ue()?;
        r.ue()?;
    }
    if r.u(1)? == 0 {
        return Ok(vui);
    }
    if r.u(1)? == 1 && r.u(8)? == 255 {
        r.u(32)?;
    }
    if r.u(1)? == 1 {
        r.u(1)?;
    }
    if r.u(1)? == 1 {
        r.u(3)?;
        vui.full_range = r.u(1)? == 1;
        if r.u(1)? == 1 {
            r.u(16)?;
            vui.matrix = r.u(8)? as u32;
        }
    }
    if r.u(1)? == 1 {
        r.ue()?;
        r.ue()?;
    }
    if r.u(1)? == 1 {
        let nu = r.u(32)?;
        let ts = r.u(32)?;
        vui.timing = Some((nu, ts));
    }
    Ok(vui)
}

// ---- ff_rfps_calculate (mov: first 99 index dts) ------------------------

fn std_framerate(i: usize) -> i64 {
    if i < 360 {
        return (i as i64 + 1) * 1001;
    }
    let i = i - 360;
    if i < 30 {
        return (i as i64 + 31) * 1001 * 12;
    }
    let i = i - 30;
    if i < 3 {
        return [80, 120, 240][i] * 1001 * 12;
    }
    [24, 30, 60, 12, 15, 48][i - 3] * 1000 * 12
}
const N_STD: usize = 360 + 30 + 3 + 6;

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

fn reduce(n: i64, d: i64) -> (i64, i64) {
    let g = gcd(n, d).max(1);
    (n / g, d / g)
}

/// `ff_rfps_add_frame` over `dts` then `ff_rfps_calculate` (H.264: the time
/// base is always "unreliable"; no codec_info_duration yet).
pub(crate) fn rfps(dts: &[i64], timescale: i64) -> Option<(i64, i64)> {
    let tb = 1.0 / timescale as f64;
    let mut err = vec![[[0f64; N_STD]; 2]; 2];
    let (mut count, mut rsum, mut g) = (0i64, 0i64, 0i64);
    let mut last: Option<i64> = None;
    for &t in dts {
        if let Some(l) = last.filter(|&l| t > l) {
            let dsec = t as f64 * tb;
            let dur = t - l;
            for i in 0..N_STD {
                if err[0][1][i] < 1e10 {
                    let sdts = dsec * std_framerate(i) as f64 / (1001 * 12) as f64;
                    for j in 0..2 {
                        let ticks = (sdts + j as f64 * 0.5).round_ties_even();
                        let e = sdts - ticks + j as f64 * 0.5;
                        err[j][0][i] += e;
                        err[j][1][i] += e * e;
                    }
                }
            }
            count += 1;
            rsum += dur;
            if count % 10 == 0 {
                let n = count as f64;
                for i in 0..N_STD {
                    if err[0][1][i] < 1e10 {
                        let a0 = err[0][0][i] / n;
                        let e0 = err[0][1][i] / n - a0 * a0;
                        let a1 = err[1][0][i] / n;
                        let e1 = err[1][1][i] / n - a1 * a1;
                        if e0 > 0.04 && e1 > 0.04 {
                            err[0][1][i] = 2e10;
                            err[1][1][i] = 2e10;
                        }
                    }
                }
            }
            if count > 3 {
                g = gcd(g, dur);
            }
        }
        last = Some(t);
    }
    if count > 15 && g > 1.max(timescale / 500) {
        return Some(reduce(timescale, g));
    }
    if count > 1 {
        let (mut num, mut best) = (0i64, 0.01f64);
        for j in 0..N_STD {
            if std_framerate(j) < 1001 * 12 {
                continue; // no codec_info_duration
            }
            if tb * rsum as f64 / (count as f64) < (1001.0 * 12.0 * 0.8) / std_framerate(j) as f64 {
                continue;
            }
            for k in 0..2 {
                let n = count as f64;
                let a = err[k][0][j] / n;
                let e = err[k][1][j] / n - a * a;
                if e < best && best > 0.000000001 {
                    best = e;
                    num = std_framerate(j);
                }
            }
        }
        if num != 0 && (num as f64 / (12.0 * 1001.0)) < 1.01 * timescale as f64 {
            return Some(reduce(num, 12 * 1001));
        }
    }
    None
}

/// ffmpeg's `print_fps` as moviepy parses it back.
pub(crate) fn banner_fps(d: f64) -> f64 {
    // lrintf(d * 100): the double is converted to float, rounded half to even.
    let v = ((d * 100.0) as f32).round_ties_even() as i64;
    let s = if v == 0 {
        format!("{d:.4}")
    } else if v % 100 != 0 {
        format!("{d:.2}")
    } else if v % (100 * 1000) != 0 {
        format!("{d:.0}")
    } else {
        format!("{:.0}000", d / 1000.0)
    };
    s.parse().unwrap_or(d)
}

/// What `ffmpeg_filter.c` inserts for a display matrix (`get_rotation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// No filter.
    Identity,
    /// `transpose=<dir>`: 0 cclock_flip, 1 clock, 2 cclock, 3 clock_flip.
    Transpose(u8),
    /// `hflip` and/or `vflip` (180 degrees, or a mirrored identity).
    Flip {
        /// Mirror left to right.
        h: bool,
        /// Mirror top to bottom.
        v: bool,
    },
    /// The `rotate` filter for a non-right angle (degrees): not supported.
    Arbitrary(i64),
}

/// `get_rotation` + the autorotate branch of `configure_input_video_filter`.
fn orientation(m: Option<&[i32; 9]>) -> Orientation {
    let Some(m) = m else { return Orientation::Identity };
    let fp = |x: i32| x as f64 / 65536.0;
    let s0 = fp(m[0]).hypot(fp(m[3]));
    let s1 = fp(m[1]).hypot(fp(m[4]));
    let rot = if s0 == 0.0 || s1 == 0.0 {
        f64::NAN
    } else {
        -((fp(m[1]) / s1).atan2(fp(m[0]) / s0) * 180.0 / std::f64::consts::PI)
    };
    let mut theta = -rot.round();
    theta -= 360.0 * (theta / 360.0 + 0.9 / 360.0).floor();
    if (theta - 90.0).abs() < 1.0 {
        Orientation::Transpose(if m[3] > 0 { 0 } else { 1 })
    } else if (theta - 180.0).abs() < 1.0 {
        Orientation::Flip { h: m[0] < 0, v: m[4] < 0 }
    } else if (theta - 270.0).abs() < 1.0 {
        Orientation::Transpose(if m[3] < 0 { 3 } else { 2 })
    } else if theta.abs() > 1.0 {
        Orientation::Arbitrary(theta as i64)
    } else if m[4] < 0 {
        Orientation::Flip { h: false, v: true }
    } else {
        Orientation::Identity
    }
}

/// `mov_read_tkhd`: the tkhd matrix times the mvhd one, or `None` for identity.
fn display_matrix(tkhd: &[i32; 9], mvhd: &[i32; 9]) -> Option<[i32; 9]> {
    let sh = [16, 16, 30];
    let mut r = [0i32; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc: i32 = 0;
            for e in 0..3 {
                acc = acc.wrapping_add(((tkhd[i * 3 + e] as i64 * mvhd[e * 3 + j] as i64) >> sh[e]) as i32);
            }
            r[i * 3 + j] = acc;
        }
    }
    let ident = [1 << 16, 0, 0, 0, 1 << 16, 0, 0, 0, 1 << 30];
    (r != ident).then_some(r)
}

/// One `mov_fix_index` result entry.
struct Fixed {
    sample: usize,
    dts: i64,
    discard: bool,
}

/// `mov_fix_index` for a video track with zero or more leading empty edits
/// and one media edit. `dts`/`ctts` per sample in decode order, edit
/// durations already in the track timescale. Returns the kept entries and
/// `min_corrected_pts` (already applied to the entries' dts).
#[allow(clippy::too_many_arguments)]
fn fix_index(
    dts: &[i64],
    ctts: Option<&[i64]>,
    key: &[bool],
    dts_shift: i64,
    edits: &[(i64, i64)],
) -> Result<(Vec<Fixed>, i64)> {
    let n = dts.len();
    let mut out = Vec::new();
    let mut entry_end = -dts_shift;
    let mut empty_sum = 0;
    let mut min_corrected: i64 = -1;
    let mut found_non_empty = false;
    for &(dur, media_time) in edits {
        let counter_start = entry_end;
        entry_end += dur;
        if !found_non_empty && media_time == -1 {
            empty_sum += dur;
            continue;
        }
        if found_non_empty {
            return Err("multiple edit list entries are not supported".into());
        }
        found_non_empty = true;
        // find_prev_closest_index(media_time): last keyframe with dts <=
        // media_time - dts_shift, then (with ctts) back to a keyframe whose
        // pts <= that.
        let target = media_time - dts_shift;
        let mut idx = (0..n).rev().find(|&i| dts[i] <= target && key[i]);
        if let (Some(c), Some(i0)) = (ctts, idx) {
            let mut i = i0 as isize;
            while i >= 0 {
                if dts[i as usize] + c[i as usize] <= target && key[i as usize] {
                    break;
                }
                i -= 1;
            }
            idx = (i >= 0).then_some(i as usize);
        }
        let start = idx.or_else(|| (0..n).rev().find(|&i| dts[i] <= target)).unwrap_or(0);
        let mut counter = counter_start;
        let mut encountered = false;
        let mut discarded_begin: Vec<i64> = Vec::new();
        let mut found_key_after = false;
        for i in start..n {
            let frame_duration = if i + 1 < n { dts[i + 1] - dts[i] } else { dur };
            let c = ctts.map_or(0, |c| c[i]);
            let curr_cts = dts[i] + dts_shift + c;
            let discard = curr_cts < media_time || curr_cts >= dur + media_time;
            if discard {
                if !encountered {
                    discarded_begin.push(frame_duration);
                }
            } else {
                let p = counter + c + dts_shift;
                min_corrected = if min_corrected < 0 { p } else { min_corrected.min(p) };
                if !encountered {
                    encountered = true;
                    // fix_index_entry_timestamps: rewrite the discarded run backwards.
                    let mut t = counter;
                    let base = out.len();
                    for (k, d) in discarded_begin.iter().rev().enumerate() {
                        t -= d;
                        let e: &mut Fixed = &mut out[base - 1 - k];
                        e.dts = t;
                    }
                }
            }
            out.push(Fixed { sample: i, dts: counter, discard });
            if encountered {
                counter += frame_duration;
            }
            if curr_cts + frame_duration >= dur + media_time && key[i] {
                if ctts.is_some() && !found_key_after {
                    found_key_after = true;
                    continue;
                }
                break;
            }
        }
    }
    min_corrected -= empty_sum;
    if min_corrected > 0 {
        for e in &mut out {
            e.dts -= min_corrected;
        }
    }
    Ok((out, min_corrected))
}

/// `(start, duration)` of one track in its own timescale as the mov demuxer
/// leaves `st->start_time` / `st->duration`: start = the leading empty
/// edits; duration = `min(mdhd, sum(stts), sum(edits))`.
fn stream_timing(d: &[u8], t: &[(String, usize, usize)], movie_ts: i64) -> Option<(i64, i64, i64)> {
    let get = |suffix: &str| t.iter().find(|(p, ..)| p.ends_with(suffix)).cloned();
    let (ts, mdhd_dur) = time_header(d, get("/mdia/mdhd")?.1);
    let ts = ts as i64;
    if ts == 0 {
        return None;
    }
    let mut dur = mdhd_dur as i64;
    if let Some((_, sa, _)) = get("/stts") {
        let n = be32(d, sa + 4) as usize;
        let sum: i64 = (0..n).map(|i| be32(d, sa + 8 + 8 * i) as i64 * be32(d, sa + 12 + 8 * i) as i64).sum();
        if sum != 0 {
            dur = dur.min(sum);
        }
    }
    let rs = |du: i64| (du * ts + movie_ts / 2) / movie_ts;
    let mut start = 0;
    if let Some((_, a, _)) = get("/elst") {
        let v1 = d[a] == 1;
        let cnt = be32(d, a + 4) as usize;
        let mut o = a + 8;
        let mut total = 0;
        let mut leading = true;
        for _ in 0..cnt {
            let (du, mt) = if v1 {
                (be64(d, o) as i64, be64(d, o + 8) as i64)
            } else {
                (be32(d, o) as i64, be32(d, o + 4) as i32 as i64)
            };
            o += if v1 { 20 } else { 12 };
            if leading && mt == -1 {
                start += rs(du);
            } else {
                leading = false;
            }
            total += rs(du);
        }
        if cnt > 0 {
            dur = dur.min(total);
        }
    }
    Some((ts, start, dur))
}

/// Parse the first H.264 or HEVC video track of an MP4/MOV.
pub fn parse(data: &[u8]) -> Result<MovVideo> {
    use crate::mp4_demux::TrackCodec;
    let bx = Boxes { d: data };
    let mut all = Vec::new();
    bx.walk(0, data.len(), "", &mut all);
    let d = data;
    if all.iter().any(|(p, ..)| p == "/moof") {
        return Err("fragmented MP4 (moof) is not supported".into());
    }
    let (mts, _) = all
        .iter()
        .find(|(p, ..)| p == "/moov/mvhd")
        .map(|(_, a, _)| time_header(d, *a))
        .context("no mvhd")?;
    // Split per trak.
    let mut traks: Vec<Vec<(String, usize, usize)>> = Vec::new();
    for b in &all {
        if b.0 == "/moov/trak" {
            traks.push(Vec::new());
        } else if b.0.starts_with("/moov/trak/") {
            if let Some(t) = traks.last_mut() {
                t.push(b.clone());
            }
        }
    }
    let n_streams = traks.len();
    let get = |t: &[(String, usize, usize)], suffix: &str| t.iter().find(|(p, ..)| p.ends_with(suffix)).cloned();
    let trak = traks
        .iter()
        .find(|t| get(t, "/mdia/hdlr").is_some_and(|(_, a, _)| &d[a + 8..a + 12] == b"vide"))
        .context("no video track")?;
    let (codec, entry) = if let Some(e) = get(trak, "/stsd/avc1").or_else(|| get(trak, "/stsd/avc3")) {
        (TrackCodec::H264, e)
    } else if let Some(e) = get(trak, "/stsd/hvc1").or_else(|| get(trak, "/stsd/hev1")) {
        (TrackCodec::Hevc, e)
    } else if let Some(e) = get(trak, "/stsd/vp09") {
        (TrackCodec::Vp9, e)
    } else if let Some(e) = get(trak, "/stsd/av01") {
        (TrackCodec::Av1, e)
    } else {
        return Err("video codec is not H.264, HEVC, VP9 or AV1".into());
    };
    if trak.iter().filter(|(p, ..)| p.contains("/stsd/") && p.matches('/').count() == 7).count() > 1 {
        return Err("more than one sample description is not supported".into());
    }
    let (width, height) = (
        u16::from_be_bytes([d[entry.1 + 24], d[entry.1 + 25]]) as u32,
        u16::from_be_bytes([d[entry.1 + 26], d[entry.1 + 27]]) as u32,
    );
    let (ts, mdhd_dur) = time_header(d, get(trak, "/mdia/mdhd").context("no mdhd")?.1);
    let ts = ts as i64;
    let read_m = |at: usize| -> [i32; 9] { std::array::from_fn(|i| be32(d, at + 4 * i) as i32) };
    let mvhd_at = all.iter().find(|(p, ..)| p == "/moov/mvhd").map(|b| b.1).unwrap();
    // mvhd: times (20 or 32 bytes), rate 4, volume 2, reserved 10, matrix.
    let mvhd_m = read_m(mvhd_at + if d[mvhd_at] == 1 { 32 } else { 20 } + 16);
    let orientation = match get(trak, "/tkhd") {
        Some((_, a, _)) => orientation(display_matrix(&read_m(a + if d[a] == 1 { 52 } else { 40 }), &mvhd_m).as_ref()),
        None => Orientation::Identity,
    };
    let (_, sa, _) = get(trak, "/stts").context("no stts")?;
    let n = be32(d, sa + 4) as usize;
    let stts: Vec<(u32, u32)> = (0..n).map(|i| (be32(d, sa + 8 + 8 * i), be32(d, sa + 12 + 8 * i))).collect();
    let durs: Vec<i64> = stts.iter().flat_map(|&(c, du)| std::iter::repeat_n(du as i64, c as usize)).collect();
    if durs.is_empty() {
        return Err("empty video track".into());
    }
    let n_samples = durs.len();
    // ctts (mov_read_ctts: signed offsets; negative ones raise dts_shift).
    let ctts: Option<Vec<i64>> = get(trak, "/ctts").map(|(_, a, _)| {
        let cnt = be32(d, a + 4) as usize;
        let mut v = Vec::with_capacity(n_samples);
        for i in 0..cnt {
            let c = be32(d, a + 8 + 8 * i) as usize;
            let off = be32(d, a + 12 + 8 * i) as i32 as i64;
            v.extend(std::iter::repeat_n(off, c));
        }
        v.resize(n_samples, *v.last().unwrap_or(&0));
        v
    });
    let has_ctts = ctts.is_some();
    let dts_shift = ctts.as_ref().map_or(0, |c| c.iter().map(|&x| -x).max().unwrap_or(0).max(0));
    let key: Vec<bool> = match get(trak, "/stss") {
        Some((_, a, _)) => {
            let mut k = vec![false; n_samples];
            for i in 0..be32(d, a + 4) as usize {
                let s = be32(d, a + 8 + 4 * i) as usize;
                if (1..=n_samples).contains(&s) {
                    k[s - 1] = true;
                }
            }
            k
        }
        None => vec![true; n_samples],
    };
    // Edit list: (duration in movie timescale, media_time).
    let edits: Vec<(i64, i64)> = get(trak, "/elst")
        .map(|(_, a, _)| {
            let v1 = d[a] == 1;
            let cnt = be32(d, a + 4) as usize;
            let mut o = a + 8;
            (0..cnt)
                .map(|_| {
                    let e = if v1 {
                        (be64(d, o) as i64, be64(d, o + 8) as i64)
                    } else {
                        (be32(d, o) as i64, be32(d, o + 4) as i32 as i64)
                    };
                    o += if v1 { 20 } else { 12 };
                    e
                })
                .collect()
        })
        .unwrap_or_default();
    let rs = |du: i64| (du * ts + mts as i64 / 2) / mts as i64;
    let shift: i64 = edits.iter().take_while(|e| e.1 == -1).map(|e| rs(e.0)).sum();
    let sum: i64 = durs.iter().sum();
    let mut odts = Vec::with_capacity(n_samples);
    let mut acc = 0;
    for &du in &durs {
        odts.push(acc);
        acc += du;
    }
    // update_stream_timings over every track: ic->start_time = min start,
    // ic->duration = max(start + duration) - start (AV_TIME_BASE units).
    let (mut f_start, mut f_end) = (i64::MAX, i64::MIN);
    for t in &traks {
        if let Some((tts, st, du)) = stream_timing(d, t, mts as i64) {
            let s_us = rescale(st, 1_000_000, tts);
            f_start = f_start.min(s_us);
            f_end = f_end.max(s_us + rescale(du, 1_000_000, tts));
        }
    }
    let start_us = if f_start == i64::MAX { rescale(shift, 1_000_000, ts) } else { f_start };

    // VUI frame period (fills a non-positive last packet duration).
    let vui = match codec {
        TrackCodec::H264 => {
            let sps = get(trak, "/avcC").and_then(|(_, a, e)| {
                let n_sps = (d.get(a + 5)? & 31) as usize;
                if n_sps == 0 || a + 8 > e {
                    return None;
                }
                let len = u16::from_be_bytes([d[a + 6], d[a + 7]]) as usize;
                d.get(a + 8..a + 8 + len).map(<[u8]>::to_vec)
            });
            match &sps {
                Some(s) => sps_vui(s)?.timing,
                None => None,
            }
        }
        TrackCodec::Hevc | TrackCodec::Vp9 | TrackCodec::Av1 => None,
    };

    let r_frame_rate = if stts.len() == 1 || (stts.len() == 2 && stts[1].0 == 1) {
        reduce(ts, stts[0].1 as i64)
    } else {
        let take = durs.len().min(99);
        match rfps(&odts[..take], ts) {
            Some(r) => r,
            None => match vui {
                Some((nu, tsc)) => reduce(tsc as i64, 2 * nu as i64),
                None => (ts, 1),
            },
        }
    };

    let (frames_pts, frames_dur, frames_sample, decode_range) = if !has_ctts {
        // Every frame shown once, in file order (no reordering, no fix_index
        // discards for a whole-track edit).
        let mut pts = Vec::with_capacity(n_samples);
        let mut acc = shift;
        for &du in &durs {
            pts.push(acc);
            acc += du;
        }
        let mut durations = durs.clone();
        let mut st_dur = (mdhd_dur as i64).min(sum);
        if !edits.is_empty() {
            st_dur = st_dur.min(edits.iter().map(|e| rs(e.0)).sum());
        }
        let last = *pts.last().unwrap();
        let dl = st_dur - last;
        *durations.last_mut().unwrap() = if dl > 0 {
            dl
        } else {
            let (nu, tsc) = vui.context("last frame needs the VUI frame rate, SPS has none")?;
            (ts * 2 * nu as i64) / tsc as i64
        };
        (pts, durations, (0..n_samples).collect::<Vec<_>>(), (0, n_samples))
    } else {
        let c = ctts.as_ref().unwrap();
        let edits_ts: Vec<(i64, i64)> = if edits.is_empty() {
            vec![(sum, 0)]
        } else {
            edits.iter().map(|&(du, mt)| (rs(du), mt)).collect()
        };
        let (fixed, _) = fix_index(&odts, Some(c), &key, dts_shift, &edits_ts)?;
        let from = fixed.first().map_or(0, |f| f.sample);
        let to = fixed.last().map_or(0, |f| f.sample + 1);
        // ff_compute_frame_duration: 1 / r_frame_rate, rounded down.
        let fdur = (ts * r_frame_rate.1) / r_frame_rate.0;
        let mut shown: Vec<(i64, usize)> = fixed
            .iter()
            .filter(|f| !f.discard)
            .map(|f| (f.dts + dts_shift + c[f.sample], f.sample))
            .collect();
        shown.sort();
        let pts: Vec<i64> = shown.iter().map(|s| s.0).collect();
        let sample: Vec<usize> = shown.iter().map(|s| s.1).collect();
        (pts, vec![fdur; sample.len()], sample, (from, to))
    };
    if frames_pts.is_empty() {
        return Err("no frames inside the edit list".into());
    }
    let avg = durs.len() as f64 * ts as f64 / sum as f64;
    // Banner `Duration:` (ic->duration + 5000 us, printed in centiseconds).
    let dur_us = f_end - start_us + 5000;
    let banner_duration = (dur_us / 10_000) as f64 / 100.0;
    Ok(MovVideo {
        codec,
        timescale: ts as u32,
        pts: frames_pts,
        durations: frames_dur,
        sample: frames_sample,
        decode_range,
        start_us,
        r_frame_rate,
        banner_fps: banner_fps(avg),
        banner_duration,
        n_streams,
        has_ctts,
        orientation,
        width,
        height,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banner_fps_formatting() {
        assert_eq!(banner_fps(51.42857), 51.43);
        assert_eq!(banner_fps(54.0), 54.0);
        assert_eq!(banner_fps(30000.0 / 1001.0), 29.97);
    }

    #[test]
    fn rescale_rounds_half_away_from_zero() {
        assert_eq!(rescale(66016, 15360, 1_000_000), 1014);
        assert_eq!(rescale(-66016, 15360, 1_000_000), -1014);
        assert_eq!(rescale(1014, 1_000_000, 15360), 66016);
    }
}
