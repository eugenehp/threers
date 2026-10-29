//! An MP4/MOV video read the way the ffmpeg n7.1 CLI pipes it as rgb24:
//! [`super::mov`] timing, [`super::pipe`] frame selection, the native
//! decoders, and [`super::swscale`].

use super::mov::{self, MovVideo};
use super::swscale::Yuv2Rgb;
use crate::decode::NativeDecoder;
use crate::mp4_demux::{demux_video, VideoTrack};
use std::collections::{BTreeMap, BTreeSet};

/// A video file opened for reading as `ffmpeg -i <file> -f rawvideo -pix_fmt rgb24 -`
/// would: same frames, same order, same pixels.
pub struct VideoReader {
    data: Vec<u8>,
    /// The track's timing as ffmpeg computes it.
    pub mov: MovVideo,
    track: VideoTrack,
}

impl VideoReader {
    /// `Err` for anything outside the supported subset (codec, rotation,
    /// fragmented files, multiple sample descriptions or edits).
    pub fn open(data: Vec<u8>) -> Result<Self, String> {
        if data.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
            // Matroska / WebM: the packets are copied out back to back.
            let (mov, track, packets) = super::mkv::parse(&data)?;
            return Ok(Self { data: packets, mov, track });
        }
        let mov = mov::parse(&data)?;
        let track = demux_video(&data)?;
        Ok(Self { data, mov, track })
    }

    /// Size of the piped pictures (after autorotation).
    pub fn size(&self) -> (u32, u32) {
        let (w, h) = (self.track.width, self.track.height);
        match self.mov.orientation {
            mov::Orientation::Transpose(_) => (h, w),
            _ => (w, h),
        }
    }

    /// Indices (into [`MovVideo::pts`]) of the frames the CLI writes for
    /// `-ss A -i f -ss B`, `(A, B)` in microseconds.
    /// Where decoding can start for the samples in `need` (decode index ->
    /// frame): one sync sample before the last sync sample at or before the
    /// first needed one. The extra GOP covers open-GOP leading pictures (a
    /// B-frame shown before its I-frame references the previous GOP); with
    /// fewer sync samples than that, the stream start. Without this every
    /// call decodes from the start of the stream, which makes extracting a
    /// long video chunk by chunk quadratic.
    /// Whether [`decode_rgb`](Self::decode_rgb) of `wanted` starts after the
    /// stream start (diagnostics / tests).
    pub fn seeks_for(&self, wanted: &BTreeSet<usize>) -> bool {
        let need: BTreeMap<usize, usize> = wanted.iter().filter_map(|&i| self.mov.sample.get(i).map(|&s| (s, i))).collect();
        self.seek_start(self.mov.decode_range.0, &need) > self.mov.decode_range.0
    }

    fn seek_start(&self, from: usize, need: &BTreeMap<usize, usize>) -> usize {
        let Some(&first) = need.keys().next() else { return from };
        let mut last = None;
        let mut prev = None;
        for si in (from..=first.min(self.track.samples.len().saturating_sub(1))).rev() {
            if self.track.samples[si].sync {
                if last.is_none() {
                    last = Some(si);
                } else {
                    prev = Some(si);
                    break;
                }
            }
        }
        prev.unwrap_or(from)
    }

    /// Which frames the CLI writes to its pipe, as indices into `mov.pts`, for
    /// input and output seeks `-ss A -i f -ss B` given as `(A, B)` in
    /// microseconds.
    pub fn pipe(&self, seek: Option<(i64, i64)>) -> Vec<usize> {
        super::pipe::pipe_frames(&self.mov, seek)
    }

    /// Decode once, calling `f(index, rgb24)` for each wanted frame index
    /// (presentation order, ascending).
    pub fn decode_rgb(
        &self,
        wanted: &BTreeSet<usize>,
        mut f: impl FnMut(usize, &[u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let Some(&last) = wanted.iter().next_back() else { return Ok(()) };
        // Decode-order position of every output frame we need.
        let mut need: BTreeMap<usize, usize> = BTreeMap::new(); // sample -> frame index
        for &i in wanted {
            need.insert(*self.mov.sample.get(i).ok_or("frame index out of range")?, i);
        }
        let last_sample = (0..=last).map(|i| self.mov.sample[i]).max().unwrap_or(0);
        let mut dec = NativeDecoder::new(&self.track)?;
        let mut conv: Option<Yuv2Rgb> = None;
        let mut generic: Option<super::sws_generic::GenericScaler> = None;
        let mut ready: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
        let mut next_out = wanted.iter().copied();
        let mut pending = next_out.next();
        let (from, to) = self.mov.decode_range;
        let start = self.seek_start(from, &need);
        for si in start..to.min(last_sample + 1) {
            let s = &self.track.samples[si];
            let bytes = self
                .data
                .get(s.offset as usize..s.offset as usize + s.size as usize)
                .ok_or("sample outside the file")?;
            dec.push(bytes, s.pts, s.dts, s.duration as i64)?;
            // (A VP9 packet holding only a hidden frame shows nothing.)
            let Some(&idx) = need.get(&si) else {
                let _ = dec.take_frame(s.pts);
                continue;
            };
            let frame = dec.take_frame(s.pts).ok_or("decoder produced no picture")?;
            use crate::decode::ChromaFormat;
            if (frame.width, frame.height) != (self.track.width, self.track.height) {
                return Err(format!(
                    "the picture size changes mid-stream ({}x{} at sample {si}, {}x{} at the start): the CLI rescales it with swscale's bicubic filter, which is not ported",
                    frame.width, frame.height, self.track.width, self.track.height
                ));
            }
            if frame.chroma == ChromaFormat::Monochrome && self.mov.orientation == mov::Orientation::Identity {
                // swscale's gray -> rgb24: R = G = B = Y, high bit depths
                // rounded down to 8 bits (measured: 10-bit is (y + 2) >> 2).
                let sh = frame.bit_depth as u32 - 8;
                let rnd = if sh > 0 { 1u32 << (sh - 1) } else { 0 };
                let mut rgb = Vec::with_capacity(frame.y.len() * 3);
                for &v in &frame.y {
                    let g = ((v as u32 + rnd) >> sh).min(255) as u8;
                    rgb.extend_from_slice(&[g, g, g]);
                }
                ready.insert(idx, rgb);
                while let Some(p) = pending {
                    let Some(rgb) = ready.remove(&p) else { break };
                    f(p, &rgb)?;
                    pending = next_out.next();
                }
                continue;
            }
            if frame.color.vui_matrix == Some(0) && frame.chroma == ChromaFormat::Yuv444 && self.mov.orientation == mov::Orientation::Identity {
                // Planar G, B, R (VP9 RGB): swscale's gbrp -> rgb24 is a copy.
                if frame.bit_depth != 8 {
                    return Err(format!("{}-bit RGB to rgb24 is not supported yet", frame.bit_depth));
                }
                let n = frame.width as usize * frame.height as usize;
                let mut rgb = vec![0u8; 3 * n];
                for i in 0..n {
                    rgb[3 * i] = frame.v[i] as u8;
                    rgb[3 * i + 1] = frame.y[i] as u8;
                    rgb[3 * i + 2] = frame.u[i] as u8;
                }
                ready.insert(idx, rgb);
                while let Some(p) = pending {
                    let Some(rgb) = ready.remove(&p) else { break };
                    f(p, &rgb)?;
                    pending = next_out.next();
                }
                continue;
            }
            let (log2_cw, log2_ch) = match frame.chroma {
                ChromaFormat::Yuv420 => (1, 1),
                ChromaFormat::Yuv422 => (1, 0),
                ChromaFormat::Yuv444 => (0, 0),
                ChromaFormat::Yuv440 => (0, 1),
                ChromaFormat::Monochrome => return Err("monochrome to rgb24 is not supported yet".into()),
            };
            let (w, h) = (frame.width as usize, frame.height as usize);
            let c = conv.get_or_insert_with(|| Yuv2Rgb::new(frame.color.vui_matrix.unwrap_or(2), frame.color.full_range));
            let (cw, ch) = ((w + (1 << log2_cw) - 1) >> log2_cw, (h + (1 << log2_ch) - 1) >> log2_ch);
            let o = self.mov.orientation;
            let transposed = matches!(o, mov::Orientation::Transpose(_));
            if (transposed || matches!(o, mov::Orientation::Arbitrary(_))) && log2_cw != log2_ch {
                return Err(format!("{:?} with rotation is not supported yet", frame.chroma));
            }
            let (ow, oh) = if transposed { (h, w) } else { (w, h) };
            // swscale's special converter: 8-bit 4:2:0 / 4:2:2, even height.
            let fast = frame.bit_depth == 8 && log2_cw == 1 && oh % 2 == 0;
            if !fast {
                if matches!(o, mov::Orientation::Arbitrary(_)) {
                    return Err(format!("{}-bit {:?} with an arbitrary rotation is not supported yet", frame.bit_depth, frame.chroma));
                }
                let (y, u, v) = (orient(&frame.y, w, h, o), orient(&frame.u, cw, ch, o), orient(&frame.v, cw, ch, o));
                let fmt = super::sws_generic::SrcFormat { w: ow, h: oh, log2_cw, log2_ch, bit_depth: frame.bit_depth as u32 };
                let sc = generic.get_or_insert_with(|| super::sws_generic::GenericScaler::new(fmt));
                ready.insert(idx, sc.convert(c, &y, &u, &v));
            } else {
                let to8 = |p: &[u16]| -> Vec<u8> { p.iter().map(|&v| v as u8).collect() };
                let (y, u, v) = match o {
                    mov::Orientation::Arbitrary(deg) => {
                        // ff_draw_color(black): Y 16 (limited) or 0 (full), chroma 128.
                        let fy = if frame.color.full_range { 0 } else { 16 };
                        let a = rotate_angle(deg);
                        (
                            rotate_plane(&to8(&frame.y), w, h, a, fy),
                            rotate_plane(&to8(&frame.u), cw, ch, a, 128),
                            rotate_plane(&to8(&frame.v), cw, ch, a, 128),
                        )
                    }
                    _ => (orient(&to8(&frame.y), w, h, o), orient(&to8(&frame.u), cw, ch, o), orient(&to8(&frame.v), cw, ch, o)),
                };
                ready.insert(idx, if log2_ch == 1 { c.convert_420(&y, &u, &v, ow, oh) } else { c.convert_422(&y, &u, &v, ow, oh) });
            }
            while let Some(p) = pending {
                let Some(rgb) = ready.remove(&p) else { break };
                f(p, &rgb)?;
                pending = next_out.next();
            }
        }
        if pending.is_some() {
            return Err("stream ended before every wanted frame was decoded".into());
        }
        Ok(())
    }
}

/// `vf_transpose` / `vf_hflip` / `vf_vflip` on one `w x h` plane.
fn orient<T: Copy + Default>(p: &[T], w: usize, h: usize, o: mov::Orientation) -> Vec<T> {
    match o {
        mov::Orientation::Identity | mov::Orientation::Arbitrary(_) => p.to_vec(),
        mov::Orientation::Flip { h: hf, v: vf } => {
            let mut out = vec![T::default(); w * h];
            for y in 0..h {
                let sy = if vf { h - 1 - y } else { y };
                for x in 0..w {
                    let sx = if hf { w - 1 - x } else { x };
                    out[y * w + x] = p[sy * w + sx];
                }
            }
            out
        }
        mov::Orientation::Transpose(dir) => {
            // dst' = transpose(src'), src' rows read bottom-up when dir & 1,
            // dst rows written bottom-up when dir & 2. dst is h x w (w' = h).
            let (dw, dh) = (h, w);
            let mut out = vec![T::default(); dw * dh];
            for dy in 0..dh {
                let ry = if dir & 2 != 0 { dh - 1 - dy } else { dy };
                for dx in 0..dw {
                    // transposed: dst'(ry, dx) = src'(dx, ry) -> src row dx, col ry.
                    let sy = if dir & 1 != 0 { h - 1 - dx } else { dx };
                    out[dy * dw + dx] = p[sy * w + ry];
                }
            }
            out
        }
    }
}

const FIXP: i64 = 1 << 16;
const FIXP2: i64 = 1 << 20;
const INT_PI: i64 = 3294199;

/// `vf_rotate`'s `int_sin`: input scaled by 2^20, output by 2^16.
fn int_sin(mut a: i64) -> i64 {
    if a < 0 {
        a = INT_PI - a;
    }
    a %= 2 * INT_PI;
    if a >= INT_PI * 3 / 2 {
        a -= 2 * INT_PI;
    }
    if a >= INT_PI / 2 {
        a = INT_PI - a;
    }
    let a2 = (a * a) / FIXP2;
    let mut res = 0;
    let mut i = 2;
    while i < 11 {
        res += a;
        a = -a * a2 / (FIXP2 * i * (i + 1));
        i += 2;
    }
    (res + 8) >> 4
}

/// The angle `av_expr` evaluates for the CLI's `"%f*PI/180"`.
fn rotate_angle(deg: i64) -> f64 {
    let theta: f64 = format!("{:.6}", deg as f64).parse().unwrap();
    theta * std::f64::consts::PI / 180.0
}

/// `vf_rotate` (bilinear, output size = input size, black fill) on one plane.
fn rotate_plane(p: &[u8], w: usize, h: usize, angle: f64, fill: u8) -> Vec<u8> {
    let angle_int = (angle * FIXP as f64 * 16.0) as i64;
    let s = int_sin(angle_int) as i32 as i64;
    let c = int_sin(angle_int + INT_PI / 2) as i32 as i64;
    let (inw, inh, outw, outh) = (w as i64, h as i64, w as i64, h as i64);
    // C integer division truncates toward zero.
    let xi = -(outw - 1) * c / 2;
    let yi = (outw - 1) * s / 2;
    let mut xprime = -(outh - 1) * s / 2;
    let mut yprime = -(outh - 1) * c / 2;
    let mut out = vec![fill; w * h];
    let (max_x, max_y) = (inw - 1, inh - 1);
    for j in 0..outh {
        let mut x = xprime + xi + FIXP * (inw - 1) / 2;
        let mut y = yprime + yi + FIXP * (inh - 1) / 2;
        for i in 0..outw {
            let (x1, y1) = (x >> 16, y >> 16);
            if x1 >= -1 && x1 <= inw && y1 >= -1 && y1 <= inh {
                // interpolate_bilinear8
                let ix = (x >> 16).clamp(0, max_x);
                let iy = (y >> 16).clamp(0, max_y);
                let (fx, fy) = (x & 0xffff, y & 0xffff);
                let ix1 = (ix + 1).min(max_x);
                let iy1 = (iy + 1).min(max_y);
                let at = |xx: i64, yy: i64| p[(yy * inw + xx) as usize] as i64;
                let s0 = (((1 << 16) - fx) * at(ix, iy) + fx * at(ix1, iy)) as i32 as i64;
                let s1 = (((1 << 16) - fx) * at(ix, iy1) + fx * at(ix1, iy1)) as i32 as i64;
                out[(j * outw + i) as usize] = ((((1 << 16) - fy) * s0 + fy * s1) >> 32) as u8;
            }
            x += c;
            y -= s;
        }
        xprime += s;
        yprime += c;
    }
    out
}
