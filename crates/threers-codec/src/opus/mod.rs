//! Opus, decoded as ffmpeg n7.1's native decoder does (`libavcodec/opus`,
//! the decoder the ffmpeg CLI picks: "opus (native)"), bit for bit on
//! aarch64: SILK (resampled to 48 kHz by its own libswresample context,
//! `filter_size 16`), CELT (av_tx's PFA inverse MDCT over the NEON
//! power-of-two codelets), hybrid mode with the CELT delay FIFO,
//! redundancy frames, multistream channel mapping (families 0, 1, 2, 255)
//! and the OpusHead output gain. [`decode_webm`] drives it the way the CLI
//! does for a WebM/Matroska file: pre-skip from the header, `DiscardPadding`
//! trimmed off the end, and the decoder drained at EOF.

pub mod celt;
pub mod mdct15;
pub mod rc;
pub mod silk;
#[rustfmt::skip]
/// SILK and CELT tables, generated from ffmpeg n7.1 `opus/tab.c`.
pub mod tab;

use crate::swr::SwrStream;
use celt::Celt;
use rc::Rc;
use silk::Silk;
use tab::{CELT_BAND_END, CELT_WINDOW2};

const MODE_SILK: u8 = 0;
const MODE_HYBRID: u8 = 1;
const MODE_CELT: u8 = 2;

const SILK_FRAME_DURATION_MS: [usize; 16] = [10, 20, 40, 60, 10, 20, 40, 60, 10, 20, 40, 60, 10, 20, 10, 20];
const SILK_RESAMPLE_DELAY: [usize; 5] = [4, 8, 11, 11, 11];
const FRAME_DURATION: [usize; 32] = [
    480, 960, 1920, 2880, 480, 960, 1920, 2880, 480, 960, 1920, 2880, 480, 960, 480, 960, 120, 240, 480, 960, 120, 240, 480, 960, 120,
    240, 480, 960, 120, 240, 480, 960,
];
const MAX_FRAME_SIZE: usize = 1275;

/// `OpusPacket`.
#[derive(Clone, Debug, Default)]
pub struct Packet {
    /// Bytes in the packet, padding included.
    pub packet_size: usize,
    /// Bytes of frame data, padding excluded.
    pub data_size: usize,
    /// The TOC's frame-count code, 0 to 3 (RFC 6716 §3.2).
    pub code: u8,
    /// Whether the TOC's stereo bit is set.
    pub stereo: bool,
    /// Whether a code-3 packet's frames have their own sizes.
    pub vbr: bool,
    /// The TOC configuration number, 0 to 31.
    pub config: usize,
    /// Frames in the packet.
    pub frame_count: usize,
    /// Where each frame starts, in bytes from the start of the packet.
    pub frame_offset: Vec<usize>,
    /// Each frame's size in bytes.
    pub frame_size: Vec<usize>,
    /// Samples per frame at 48 kHz.
    pub frame_duration: usize,
    /// 0 for SILK, 1 for hybrid, 2 for CELT.
    pub mode: u8,
    /// Audio bandwidth: 0 narrowband to 4 fullband.
    pub bandwidth: usize,
}

fn lacing16(b: &[u8], p: &mut usize, end: usize) -> Option<usize> {
    if *p >= end {
        return None;
    }
    let mut v = b[*p] as usize;
    *p += 1;
    if v >= 252 {
        if *p >= end {
            return None;
        }
        v += 4 * b[*p] as usize;
        *p += 1;
    }
    Some(v)
}

fn lacing_full(b: &[u8], p: &mut usize, end: usize) -> Option<usize> {
    let mut v = 0usize;
    loop {
        if *p >= end {
            return None;
        }
        let next = b[*p] as usize;
        *p += 1;
        v += next;
        if next < 255 {
            break;
        }
        v -= 1;
    }
    Some(v)
}

/// `ff_opus_parse_packet`.
pub fn parse_packet(buf: &[u8], self_delimiting: bool) -> Result<Packet, String> {
    let fail = || Err::<Packet, String>("Error parsing the packet header.".into());
    let mut pkt = Packet::default();
    let mut end = buf.len();
    let mut buf_size = buf.len();
    if buf_size < 1 {
        return fail();
    }
    let mut p = 0usize;
    let toc = buf[p];
    p += 1;
    pkt.code = toc & 3;
    pkt.stereo = (toc >> 2) & 1 == 1;
    pkt.config = (toc >> 3) as usize;
    if pkt.code >= 2 && buf_size < 2 {
        return fail();
    }
    let mut padding = 0usize;
    match pkt.code {
        0 => {
            pkt.frame_count = 1;
            if self_delimiting {
                let Some(len) = lacing16(buf, &mut p, end) else { return fail() };
                if len > end - p {
                    return fail();
                }
                end = p + len;
                buf_size = end;
            }
            let fb = end - p;
            if fb > MAX_FRAME_SIZE {
                return fail();
            }
            pkt.frame_offset = vec![p];
            pkt.frame_size = vec![fb];
        }
        1 => {
            pkt.frame_count = 2;
            if self_delimiting {
                let Some(len) = lacing16(buf, &mut p, end) else { return fail() };
                if 2 * len > end - p {
                    return fail();
                }
                end = p + 2 * len;
                buf_size = end;
            }
            let fb = end - p;
            if fb & 1 == 1 || fb >> 1 > MAX_FRAME_SIZE {
                return fail();
            }
            pkt.frame_offset = vec![p, p + fb / 2];
            pkt.frame_size = vec![fb / 2, fb / 2];
        }
        2 => {
            pkt.frame_count = 2;
            pkt.vbr = true;
            let Some(fb0) = lacing16(buf, &mut p, end) else { return fail() };
            if self_delimiting {
                let Some(len) = lacing16(buf, &mut p, end) else { return fail() };
                if len + fb0 > end - p {
                    return fail();
                }
                end = p + fb0 + len;
                buf_size = end;
            }
            if fb0 > end - p {
                return fail();
            }
            let fb1 = end - p - fb0;
            if fb1 > MAX_FRAME_SIZE {
                return fail();
            }
            pkt.frame_offset = vec![p, p + fb0];
            pkt.frame_size = vec![fb0, fb1];
        }
        _ => {
            let i = buf[p];
            p += 1;
            pkt.frame_count = (i & 0x3f) as usize;
            let has_padding = (i >> 6) & 1 == 1;
            pkt.vbr = i >> 7 == 1;
            if pkt.frame_count == 0 || pkt.frame_count > 48 {
                return fail();
            }
            if has_padding {
                let Some(v) = lacing_full(buf, &mut p, end) else { return fail() };
                padding = v;
            }
            pkt.frame_size = vec![0; pkt.frame_count];
            pkt.frame_offset = vec![0; pkt.frame_count];
            if pkt.vbr {
                let mut total = 0;
                for i in 0..pkt.frame_count - 1 {
                    let Some(fb) = lacing16(buf, &mut p, end) else { return fail() };
                    pkt.frame_size[i] = fb;
                    total += fb;
                }
                if self_delimiting {
                    let Some(len) = lacing16(buf, &mut p, end) else { return fail() };
                    if len + total + padding > end - p {
                        return fail();
                    }
                    end = p + total + len + padding;
                    buf_size = end;
                }
                if end < p + padding {
                    return fail();
                }
                let fb = end - p - padding;
                if total > fb {
                    return fail();
                }
                pkt.frame_offset[0] = p;
                for i in 1..pkt.frame_count {
                    pkt.frame_offset[i] = pkt.frame_offset[i - 1] + pkt.frame_size[i - 1];
                }
                pkt.frame_size[pkt.frame_count - 1] = fb - total;
            } else {
                let fb;
                if self_delimiting {
                    let Some(v) = lacing16(buf, &mut p, end) else { return fail() };
                    if pkt.frame_count * v + padding > end - p {
                        return fail();
                    }
                    end = p + pkt.frame_count * v + padding;
                    buf_size = end;
                    fb = v;
                } else {
                    if end < p + padding {
                        return fail();
                    }
                    let t = end - p - padding;
                    if !t.is_multiple_of(pkt.frame_count) || t / pkt.frame_count > MAX_FRAME_SIZE {
                        return fail();
                    }
                    fb = t / pkt.frame_count;
                }
                pkt.frame_offset[0] = p;
                pkt.frame_size[0] = fb;
                for i in 1..pkt.frame_count {
                    pkt.frame_offset[i] = pkt.frame_offset[i - 1] + pkt.frame_size[i - 1];
                    pkt.frame_size[i] = fb;
                }
            }
        }
    }
    pkt.packet_size = buf_size;
    pkt.data_size = buf_size - padding;
    pkt.frame_duration = FRAME_DURATION[pkt.config];
    if pkt.frame_duration * pkt.frame_count > 5760 {
        return fail();
    }
    if pkt.config < 12 {
        pkt.mode = MODE_SILK;
        pkt.bandwidth = pkt.config >> 2;
    } else if pkt.config < 16 {
        pkt.mode = MODE_HYBRID;
        pkt.bandwidth = 3 + (pkt.config >= 14) as usize;
    } else {
        pkt.mode = MODE_CELT;
        pkt.bandwidth = (pkt.config - 16) >> 2;
        if pkt.bandwidth != 0 {
            pkt.bandwidth += 1;
        }
    }
    Ok(pkt)
}

/// One output channel's source (`ChannelMap`).
#[derive(Clone, Copy, Debug, Default)]
struct ChannelMap {
    stream_idx: usize,
    channel_idx: usize,
    copy: Option<usize>,
    silence: bool,
}

/// `ff_vorbis_channel_layout_offsets`.
const VORBIS_OFFSETS: [[usize; 8]; 8] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 2, 1, 0, 0, 0, 0, 0],
    [0, 1, 2, 3, 0, 0, 0, 0],
    [0, 2, 1, 3, 4, 0, 0, 0],
    [0, 2, 1, 4, 5, 3, 0, 0],
    [0, 2, 1, 5, 6, 4, 3, 0],
    [0, 2, 1, 6, 7, 4, 5, 3],
];

/// The parsed `OpusHead` (`ff_opus_parse_extradata`).
#[derive(Clone, Debug)]
pub struct Head {
    /// Output channels.
    pub channels: usize,
    /// Samples at 48 kHz to drop from the start of the decode.
    pub pre_skip: usize,
    /// Output gain, Q7.8 dB.
    pub gain_i: i16,
    /// Elementary streams in the multistream packet.
    pub nb_streams: usize,
    /// How many of them are coupled stereo streams.
    pub nb_stereo_streams: usize,
    maps: Vec<ChannelMap>,
}

/// Parse an `OpusHead` (the Matroska `CodecPrivate`, or the first Ogg packet).
pub fn parse_head(extradata: &[u8]) -> Result<Head, String> {
    if extradata.len() < 19 {
        return Err(format!("Invalid extradata size: {}", extradata.len()));
    }
    if extradata[8] > 15 {
        return Err(format!("Extradata version {}", extradata[8]));
    }
    let pre_skip = u16::from_le_bytes([extradata[10], extradata[11]]) as usize;
    let channels = extradata[9] as usize;
    if channels == 0 {
        return Err("Zero channel count specified in the extradata".into());
    }
    let gain_i = i16::from_le_bytes([extradata[16], extradata[17]]);
    let map_type = extradata[18];
    let (streams, stereo_streams, channel_map, vorbis): (usize, usize, Vec<u8>, bool) = match map_type {
        0 => {
            if channels > 2 {
                return Err("Channel mapping 0 is only specified for up to 2 channels".into());
            }
            (1, channels - 1, vec![0, 1], false)
        }
        1 | 2 | 255 => {
            if extradata.len() < 21 + channels {
                return Err(format!("Invalid extradata size: {}", extradata.len()));
            }
            let (s, ss) = (extradata[19] as usize, extradata[20] as usize);
            if s == 0 || ss > s || s + ss > 255 {
                return Err(format!("Invalid stream/stereo stream count: {s}/{ss}"));
            }
            if map_type == 1 && channels > 8 {
                return Err("Channel mapping 1 is only specified for up to 8 channels".into());
            }
            (s, ss, extradata[21..21 + channels].to_vec(), map_type == 1)
        }
        t => return Err(format!("Mapping type {t}")),
    };
    let reorder = |i: usize| if vorbis { VORBIS_OFFSETS[channels - 1][i] } else { i };
    let mut maps = vec![ChannelMap::default(); channels];
    for i in 0..channels {
        let idx = channel_map[reorder(i)] as usize;
        if idx == 255 {
            maps[i].silence = true;
            continue;
        }
        if idx >= streams + stereo_streams {
            return Err(format!("Invalid channel map for output channel {i}: {idx}"));
        }
        maps[i].copy = (0..i).find(|&j| channel_map[reorder(j)] as usize == idx);
        if idx < 2 * stereo_streams {
            maps[i].stream_idx = idx / 2;
            maps[i].channel_idx = idx & 1;
        } else {
            maps[i].stream_idx = idx - stereo_streams;
            maps[i].channel_idx = 0;
        }
    }
    Ok(Head { channels, pre_skip, gain_i, nb_streams: streams, nb_stereo_streams: stereo_streams, maps })
}

/// `opus_fade(out, in1, in2, window, len)`: `in2*w + in1*(1.0 - w)`, the
/// second product in double.
#[inline]
fn fade1(in1: f32, in2: f32, w: f32) -> f32 {
    ((in2 * w) as f64 + in1 as f64 * (1.0 - w as f64)) as f32
}

/// `OpusStreamContext`.
struct Stream {
    output_channels: usize,
    silk: Silk,
    celt: Celt,
    swr: Option<SwrStream>,
    silk_samplerate: u32,
    delayed_samples: usize,
    celt_delay: [Vec<f32>; 2],
    sync_buffer: [Vec<f32>; 2],
    redundancy_idx: usize,
    packet: Packet,
    decoded_samples: usize,
    silk_out: [Vec<f32>; 2],
    celt_out: [Vec<f32>; 2],
    red_out: [Vec<f32>; 2],
    /// The stream's output planes for the current packet and the write cursor.
    out: [Vec<f32>; 2],
    cur: usize,
}

impl Stream {
    fn new(output_channels: usize, apply_phase_inv: bool) -> Self {
        Stream {
            output_channels,
            silk: Silk::new(output_channels),
            celt: Celt::new(output_channels, apply_phase_inv),
            swr: None,
            silk_samplerate: 0,
            delayed_samples: 0,
            celt_delay: [Vec::new(), Vec::new()],
            sync_buffer: [Vec::new(), Vec::new()],
            redundancy_idx: 0,
            packet: Packet::default(),
            decoded_samples: 0,
            silk_out: [vec![0.0; 960], vec![0.0; 960]],
            celt_out: [vec![0.0; 960], vec![0.0; 960]],
            red_out: [vec![0.0; 960], vec![0.0; 960]],
            out: [Vec::new(), Vec::new()],
            cur: 0,
        }
    }

    fn redundancy_fade(&mut self) {
        let idx = self.redundancy_idx;
        if idx != 0 {
            for c in 0..self.output_channels {
                for i in 0..120 - idx {
                    let o = &mut self.out[c][self.cur + i];
                    *o = fade1(*o, self.red_out[c][120 + idx + i], CELT_WINDOW2[idx + i]);
                }
            }
            self.redundancy_idx = 0;
        }
    }

    /// `opus_flush_resample`.
    fn flush_resample(&mut self, nb_samples: usize) -> Result<(), String> {
        let swr = self.swr.as_mut().unwrap();
        let mut ys = vec![Vec::new(); self.output_channels];
        let got = swr.flush(&mut ys, nb_samples);
        if got != nb_samples {
            return Err(format!("Wrong number of flushed samples: {got}"));
        }
        for c in 0..self.output_channels {
            self.out[c][self.cur..self.cur + got].copy_from_slice(&ys[c]);
        }
        let celt_size = self.celt_delay[0].len();
        if celt_size != 0 {
            if celt_size != nb_samples {
                return Err("Wrong number of CELT delay samples.".into());
            }
            for c in 0..self.output_channels {
                let d = std::mem::take(&mut self.celt_delay[c]);
                for (o, &v) in self.out[c][self.cur..self.cur + nb_samples].iter_mut().zip(&d) {
                    *o = v.mul_add(1.0, *o);
                }
            }
        }
        self.redundancy_fade();
        self.cur += nb_samples;
        Ok(())
    }

    /// `opus_decode_redundancy`.
    fn decode_redundancy(&mut self, data: &[u8], size: usize) -> Result<(), String> {
        let mut rc = Rc::new(data, size);
        rc.raw_init(size, size as u32);
        let n = self.output_channels;
        let (a, b) = self.red_out.split_at_mut(1);
        let mut outs: Vec<&mut [f32]> = vec![&mut a[0][..], &mut b[0][..]];
        outs.truncate(n);
        self.celt
            .decode_frame(&mut rc, &mut outs, self.packet.stereo as usize + 1, 240, 0, CELT_BAND_END[self.packet.bandwidth] as usize)
            .map_err(|e| format!("Error decoding the redundancy frame: {e}"))
    }

    /// `opus_decode_frame`: `data` runs from the frame start to the end of
    /// the packet (the range decoder may read past `size`).
    fn decode_frame(&mut self, data: &[u8], size: usize) -> Result<usize, String> {
        let fd = self.packet.frame_duration;
        let mut samples = fd;
        let delayed_samples = self.delayed_samples;
        let mut size = size;
        let mut rc = Rc::new(data, size);
        let mode = self.packet.mode;
        let nch = self.output_channels;

        if mode == MODE_SILK || mode == MODE_HYBRID {
            if self.swr.is_none() {
                let mut swr = SwrStream::new(self.silk_samplerate, 48000, nch, 16)?;
                let z = vec![0f32; SILK_RESAMPLE_DELAY[self.packet.bandwidth]];
                let zs: Vec<&[f32]> = (0..nch).map(|_| z.as_slice()).collect();
                swr.push(&zs);
                self.swr = Some(swr);
            }
            let n = self.silk.decode_superframe(
                &mut rc,
                &mut self.silk_out,
                self.packet.bandwidth.min(2),
                self.packet.stereo as usize + 1,
                SILK_FRAME_DURATION_MS[self.packet.config],
            )?;
            let ins: Vec<&[f32]> = (0..nch).map(|c| &self.silk_out[c][..n]).collect();
            let mut ys = vec![Vec::new(); nch];
            samples = self.swr.as_mut().unwrap().convert(&mut ys, fd, &ins);
            for c in 0..nch {
                self.out[c][self.cur..self.cur + samples].copy_from_slice(&ys[c]);
            }
            self.delayed_samples += fd - samples;
        } else {
            self.silk.flush();
        }

        let consumed = rc.tell() as usize;
        let mut redundancy = false;
        if mode == MODE_HYBRID && consumed + 37 <= size * 8 {
            redundancy = rc.dec_log(12) != 0;
        } else if mode == MODE_SILK && consumed + 17 <= size * 8 {
            redundancy = true;
        }
        let (mut redundancy_pos, mut redundancy_size) = (false, 0usize);
        if redundancy {
            redundancy_pos = rc.dec_log(1) != 0;
            redundancy_size = if mode == MODE_HYBRID { rc.dec_uint(256) as usize + 2 } else { size - consumed.div_ceil(8) };
            if redundancy_size > size {
                return Err("Invalid redundancy frame size.".into());
            }
            size -= redundancy_size;
            celt::stat(if redundancy_pos { 10 } else { 11 });
            if redundancy_pos {
                self.decode_redundancy(&data[size..], redundancy_size)?;
                self.celt.flush();
            }
        }

        if mode == MODE_CELT || mode == MODE_HYBRID {
            let mut out_off = self.cur;
            let mut celt_output_samples = samples;
            let delay = self.celt_delay[0].len();
            if delay != 0 {
                if mode == MODE_HYBRID {
                    for c in 0..nch {
                        let d = std::mem::take(&mut self.celt_delay[c]);
                        for (o, &v) in self.out[c][out_off..out_off + delay].iter_mut().zip(&d) {
                            *o = v.mul_add(1.0, *o);
                        }
                    }
                    out_off += delay;
                    celt_output_samples -= delay;
                } else {
                    for c in 0..2 {
                        self.celt_delay[c].clear();
                    }
                }
            }
            rc.raw_init(size, size as u32);
            let start = if mode == MODE_HYBRID { 17 } else { 0 };
            let end = CELT_BAND_END[self.packet.bandwidth] as usize;
            let channels = self.packet.stereo as usize + 1;
            if mode == MODE_CELT {
                let cur = self.cur;
                let (a, b) = self.out.split_at_mut(1);
                let mut outs: Vec<&mut [f32]> = vec![&mut a[0][cur..], &mut b[0][cur..]];
                outs.truncate(nch);
                self.celt.decode_frame(&mut rc, &mut outs, channels, fd, start, end)?;
            } else {
                let (a, b) = self.celt_out.split_at_mut(1);
                let mut outs: Vec<&mut [f32]> = vec![&mut a[0][..], &mut b[0][..]];
                outs.truncate(nch);
                self.celt.decode_frame(&mut rc, &mut outs, channels, fd, start, end)?;
                for c in 0..nch {
                    for k in 0..celt_output_samples {
                        let o = &mut self.out[c][out_off + k];
                        *o = self.celt_out[c][k].mul_add(1.0, *o);
                    }
                    let (celt_out, delay) = (&self.celt_out[c], &mut self.celt_delay[c]);
                    delay.extend_from_slice(&celt_out[celt_output_samples..fd]);
                }
            }
        } else {
            self.celt.flush();
        }

        self.redundancy_fade();
        if redundancy {
            if !redundancy_pos {
                self.celt.flush();
                self.decode_redundancy(&data[size..], redundancy_size)?;
                let at = self.cur + samples - 120 + delayed_samples;
                for c in 0..nch {
                    for i in 0..120 - delayed_samples {
                        let o = &mut self.out[c][at + i];
                        *o = fade1(*o, self.red_out[c][120 + i], CELT_WINDOW2[i]);
                    }
                    if delayed_samples != 0 {
                        self.redundancy_idx = 120 - delayed_samples;
                    }
                }
            } else {
                let at = self.cur + delayed_samples;
                for c in 0..nch {
                    self.out[c][at..at + 120].copy_from_slice(&self.red_out[c][..120]);
                    for i in 0..120 {
                        let o = &mut self.out[c][at + 120 + i];
                        *o = fade1(self.red_out[c][120 + i], *o, CELT_WINDOW2[i]);
                    }
                }
            }
        }
        Ok(samples)
    }

    /// `opus_decode_subpacket`. `buf` starts at the sub-packet and runs to
    /// the end of the packet (`None` drains).
    fn decode_subpacket(&mut self, buf: Option<&[u8]>) -> Result<usize, String> {
        let mut output_samples = 0;
        let mut flush_needed = false;
        if let Some(swr) = &self.swr {
            flush_needed =
                if buf.is_some() { self.packet.mode == MODE_CELT || swr.in_rate != self.silk_samplerate } else { self.delayed_samples != 0 };
        }
        if buf.is_none() && !flush_needed {
            return Ok(0);
        }
        if flush_needed {
            self.flush_resample(self.delayed_samples).map_err(|e| format!("Error flushing the resampler: {e}"))?;
            self.swr = None;
            output_samples += self.delayed_samples;
            self.delayed_samples = 0;
            if buf.is_none() {
                return Ok(output_samples);
            }
        }
        let buf = buf.unwrap();
        for i in 0..self.packet.frame_count {
            let (off, size) = (self.packet.frame_offset[i], self.packet.frame_size[i]);
            let samples = match self.decode_frame(&buf[off..], size) {
                Ok(s) => s,
                Err(_) => {
                    let fd = self.packet.frame_duration;
                    for c in 0..self.output_channels {
                        self.out[c][self.cur..self.cur + fd].iter_mut().for_each(|v| *v = 0.0);
                    }
                    fd
                }
            };
            output_samples += samples;
            self.cur += samples;
        }
        Ok(output_samples)
    }
}

/// ffmpeg's native Opus decoder.
pub struct Decoder {
    /// The `OpusHead` the decoder was opened with.
    pub head: Head,
    streams: Vec<Stream>,
    gain: f32,
}

impl Decoder {
    /// `opus_decode_init` from the `OpusHead` (Matroska `CodecPrivate`).
    pub fn new(extradata: &[u8]) -> Result<Self, String> {
        let head = parse_head(extradata)?;
        let gain = if head.gain_i != 0 { 10f64.powf(head.gain_i as f64 / (20.0 * 256.0)) as f32 } else { 0.0 };
        let streams = (0..head.nb_streams).map(|i| Stream::new(if i < head.nb_stereo_streams { 2 } else { 1 }, true)).collect();
        Ok(Decoder { head, streams, gain })
    }

    /// `opus_decode_packet`: planar output, one Vec per channel (empty when
    /// nothing came out). `None` drains.
    pub fn decode(&mut self, pkt: Option<&[u8]>) -> Result<Vec<Vec<f32>>, String> {
        let nb = self.streams.len();
        let mut delayed = 0;
        for s in &self.streams {
            delayed = delayed.max(s.delayed_samples + s.sync_buffer[0].len());
        }
        let mut coded_samples = 0;
        if let Some(buf) = pkt {
            let p = parse_packet(buf, nb > 1)?;
            coded_samples = p.frame_count * p.frame_duration;
            self.streams[0].silk_samplerate = silk_samplerate(p.config);
            self.streams[0].packet = p;
        }
        let nb_samples = coded_samples + delayed;
        let channels = self.head.channels;
        if nb_samples == 0 {
            return Ok(vec![Vec::new(); channels]);
        }
        for s in self.streams.iter_mut() {
            let n = s.sync_buffer[0].len();
            for c in 0..2 {
                s.out[c].clear();
                s.out[c].resize(nb_samples + 960, 0.0);
                let sb = std::mem::take(&mut s.sync_buffer[c]);
                s.out[c][..sb.len()].copy_from_slice(&sb);
            }
            s.cur = n;
        }
        let mut decoded_samples = usize::MAX;
        let mut at = 0usize;
        for i in 0..nb {
            if let Some(buf) = pkt {
                if i > 0 {
                    let p = parse_packet(&buf[at..], i != nb - 1)?;
                    if coded_samples != p.frame_count * p.frame_duration {
                        return Err(format!("Mismatching coded sample count in substream {i}."));
                    }
                    self.streams[i].silk_samplerate = silk_samplerate(p.config);
                    self.streams[i].packet = p;
                }
            }
            let s = &mut self.streams[i];
            let ret = s.decode_subpacket(pkt.map(|b| &b[at..]))?;
            s.decoded_samples = ret;
            decoded_samples = decoded_samples.min(ret);
            if pkt.is_some() {
                at += s.packet.packet_size;
            }
        }
        for s in self.streams.iter_mut() {
            let extra = s.decoded_samples - decoded_samples;
            if extra != 0 {
                let base = s.cur - s.decoded_samples + decoded_samples;
                for c in 0..2 {
                    s.sync_buffer[c] = s.out[c][base..base + extra].to_vec();
                }
            }
        }
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); channels];
        for (i, m) in self.head.maps.iter().enumerate() {
            out[i] = if let Some(j) = m.copy {
                out[j].clone()
            } else if m.silence {
                vec![0.0; decoded_samples]
            } else {
                self.streams[m.stream_idx].out[m.channel_idx][..decoded_samples].to_vec()
            };
            if self.head.gain_i != 0 && decoded_samples > 0 {
                for v in out[i].iter_mut() {
                    *v *= self.gain;
                }
            }
        }
        Ok(out)
    }
}

fn silk_samplerate(config: usize) -> u32 {
    if config < 4 {
        8000
    } else if config < 8 {
        12000
    } else {
        16000
    }
}

/// A decoded Opus track: planar 48 kHz float and the per-frame sample counts
/// the CLI saw (after pre-skip and discard padding).
pub struct Decoded {
    /// Always 48000.
    pub sample_rate: u32,
    /// One `Vec` per channel.
    pub planes: Vec<Vec<f32>>,
    /// Samples in each decoded frame, in order.
    pub frame_lens: Vec<usize>,
}

/// Collects decoded frames through `decode.c`'s `discard_samples`.
struct Sink {
    skip: usize,
    planes: Vec<Vec<f32>>,
    frame_lens: Vec<usize>,
}

impl Sink {
    fn push(&mut self, frame: Vec<Vec<f32>>, discard: usize) {
        let n = frame.first().map_or(0, |p| p.len());
        if n == 0 {
            return;
        }
        let mut lo = 0;
        if self.skip > 0 {
            if n <= self.skip {
                self.skip -= n;
                return;
            }
            lo = self.skip;
            self.skip = 0;
        }
        let mut hi = n;
        if discard > 0 && discard <= n - lo {
            if discard == n - lo {
                return;
            }
            hi = n - discard;
        }
        for (p, f) in self.planes.iter_mut().zip(&frame) {
            p.extend_from_slice(&f[lo..hi]);
        }
        self.frame_lens.push(hi - lo);
    }
}

/// Decode a WebM/Matroska file's first Opus track as
/// `ffmpeg -i IN -f f32le -` does (ffmpeg n7.1, native decoder).
pub fn decode_webm(data: &[u8]) -> Result<Decoded, String> {
    decode_matroska(data)?.ok_or_else(|| "no audio track".to_string())
}

/// The audio of a WebM/Matroska file as the ffmpeg CLI decodes its default
/// audio stream: `Ok(None)` without an audio track; an error for codecs
/// other than Opus (ffmpeg's native Vorbis decoder is not ported — the
/// `vorbis` module is libvorbis, which `soundfile` uses).
pub fn decode_matroska(data: &[u8]) -> Result<Option<Decoded>, String> {
    use crate::mkv_demux::{Mkv, TrackKind};
    let mkv = Mkv::parse(data)?;
    let audio: Vec<_> = mkv.tracks.iter().filter(|t| t.kind == TrackKind::Audio).collect();
    let Some(track) = audio.iter().find(|t| t.flag_default).or(audio.first()).copied() else {
        return Ok(None);
    };
    if !track.codec_id.starts_with("A_OPUS") {
        return Err(format!("{}: only Opus audio is decoded from Matroska/WebM", track.codec_id));
    }
    let mut dec = Decoder::new(&track.codec_private)?;
    // matroskadec rescales DiscardPadding by `codecpar->sample_rate`, which
    // avformat_find_stream_info has replaced with the decoder's 48 kHz by the
    // time the (final) padded packets are read.
    let rate = 48000i128;
    let mut sink = Sink { skip: dec.head.pre_skip, planes: vec![Vec::new(); dec.head.channels], frame_lens: Vec::new() };
    for f in mkv.frames_of(track.number) {
        let mut discard = 0usize;
        if f.discard_padding != 0 {
            // av_rescale_q(ns, 1/1e9, 1/rate), to nearest; the side data
            // also replaces skip_samples with its first field.
            let num = f.discard_padding as i128 * rate;
            let v = if num >= 0 { (num + 500_000_000) / 1_000_000_000 } else { -((-num + 500_000_000) / 1_000_000_000) } as i64;
            if v > 0 {
                discard = v as usize;
                sink.skip = 0;
            } else {
                sink.skip = (-v) as usize;
            }
        }
        let frame = dec.decode(Some(&f.data))?;
        sink.push(frame, discard);
    }
    loop {
        let frame = dec.decode(None)?;
        if frame.first().map_or(0, |p| p.len()) == 0 {
            break;
        }
        sink.push(frame, 0);
    }
    Ok(Some(Decoded { sample_rate: 48000, planes: sink.planes, frame_lens: sink.frame_lens }))
}
