//! Native, dependency-free H.264, HEVC, VP9 and AV1 *decoding* — the software
//! twin of `videotoolbox_decode::VideoToolboxDecoder`.
//!
//! VideoToolbox only exists on Apple platforms, and on the web there is no
//! system decoder a `wasm32` build can reach synchronously at all. This is
//! the same job done in pure Rust: no OS calls, no threads, no I/O, so it
//! builds and runs everywhere the rest of [`crate`] does.
//!
//! The decoders need the `decode` feature; an app that only encodes does not
//! compile them. The bit reader and the H.264 and
//! HEVC parameter-set parsers (`bits`, `h264::params`, `hevc::params`) are
//! here either way, because the MP4 muxer reads a stream's SPS to describe
//! its colour.
//!
//! [`NativeDecoder`] has the same shape as the VideoToolbox session, so code
//! written against one works with the other: open it from a demuxed track,
//! push samples in decode order, and take pictures back by presentation
//! time.
//!
//! ```no_run
//! # #[cfg(feature = "decode")] {
//! use threers_codec::decode::NativeDecoder;
//! use threers_codec::mp4_demux::demux_video;
//!
//! let file = std::fs::read("clip.mp4").unwrap();
//! let track = demux_video(&file).unwrap();
//! let mut dec = NativeDecoder::new(&track).unwrap();
//! for s in &track.samples {
//!     let bytes = &file[s.offset as usize..][..s.size as usize];
//!     dec.push(bytes, s.pts, s.dts, s.duration as i64).unwrap();
//!     if let Some(bgra) = dec.take(s.pts) {
//!         let _ = bgra; // width * height * 4
//!     }
//! }
//! # }
//! ```
//!
//! [`NativeDecoder::take_frame`] returns the planes instead of BGRA, at the
//! stream's own bit depth — what a renderer uploading YUV textures wants,
//! and what the conformance tests compare against a reference decoder
//! sample for sample.
//!
//! What each decoder covers is in its module docs: [`h264`], [`hevc`], `vp9`
//! and `av1`.
//! A stream using something outside that is refused with an error naming
//! the feature, not decoded into garbage.

// The decoders follow the specifications' own notation: loops over an index
// that addresses several parallel arrays at once (`bs[dir][e][s]`, the two
// reference lists, the 16 4×4 blocks). Iterator chains would hide that
// correspondence for no gain.
#![allow(clippy::needless_range_loop)]

#[cfg(feature = "decode")]
pub mod av1;
pub mod bits;
#[cfg(feature = "decode")]
pub mod cabac;
pub mod h264;
pub mod hevc;
#[cfg(feature = "decode")]
pub(crate) mod jobs;
pub mod picture;
#[cfg(feature = "decode")]
pub mod platform;
#[cfg(feature = "decode")]
pub mod vp9;
pub mod vp9_packet;

#[cfg(feature = "decode")]
pub use h264::H264Decoder;
#[cfg(feature = "decode")]
pub use hevc::HevcDecoder;
#[cfg(feature = "decode")]
pub use picture::{ChromaFormat, ColorInfo, DecodedFrame, Matrix, Plane};
#[cfg(feature = "decode")]
pub use platform::{Backend, Decoder};

#[cfg(feature = "decode")]
use crate::mp4_demux::{TrackCodec, VideoTrack};

#[cfg(feature = "decode")]
enum Codec {
    H264(Box<H264Decoder>),
    Hevc(Box<HevcDecoder>),
    Vp9(Box<vp9::Vp9Decoder>),
    Av1(Box<av1::Av1Decoder>),
}

#[cfg(feature = "decode")]
impl Codec {
    fn decode_nal(&mut self, nal: &[u8]) -> Result<(), String> {
        match self {
            Codec::H264(d) => d.decode_nal(nal),
            Codec::Hevc(d) => d.decode_nal(nal),
            Codec::Vp9(_) => Err("VP9 has no NAL units".into()),
            Codec::Av1(_) => Err("AV1 has no NAL units".into()),
        }
    }

    /// End of an access unit: finish the picture in progress and return
    /// everything completed, in decode order.
    fn flush(&mut self) -> Vec<DecodedFrame> {
        match self {
            Codec::H264(d) => {
                d.finish_picture();
                d.take_pictures().into_iter().map(|p| p.frame).collect()
            }
            Codec::Hevc(d) => {
                d.finish_picture();
                d.take_pictures().into_iter().map(|p| p.frame).collect()
            }
            Codec::Vp9(_) | Codec::Av1(_) => Vec::new(),
        }
    }
}

#[cfg(feature = "decode")]
/// Split an `avcC` payload into `(parameter set NALs, NAL length size)`.
pub(crate) fn split_avcc(cfg: &[u8]) -> Result<(Vec<&[u8]>, usize), String> {
    if cfg.len() < 7 {
        return Err("avcC too short".into());
    }
    let nal_len = (cfg[4] & 3) as usize + 1;
    let mut sets = Vec::new();
    let mut at = 5;
    for pass in 0..2 {
        let count = if pass == 0 {
            (cfg[at] & 31) as usize
        } else {
            cfg.get(at).copied().unwrap_or(0) as usize
        };
        at += 1;
        for _ in 0..count {
            if at + 2 > cfg.len() {
                return Err("avcC ended inside a parameter set".into());
            }
            let len = u16::from_be_bytes([cfg[at], cfg[at + 1]]) as usize;
            at += 2;
            if at + len > cfg.len() {
                return Err("avcC parameter set runs past the box".into());
            }
            sets.push(&cfg[at..at + len]);
            at += len;
        }
        if at >= cfg.len() {
            break;
        }
    }
    Ok((sets, nal_len))
}

#[cfg(feature = "decode")]
/// Split an `hvcC` payload into `(parameter set NALs, NAL length size)`.
pub(crate) fn split_hvcc(cfg: &[u8]) -> Result<(Vec<&[u8]>, usize), String> {
    if cfg.len() < 23 {
        return Err("hvcC too short".into());
    }
    let nal_len = (cfg[21] & 3) as usize + 1;
    let arrays = cfg[22] as usize;
    let mut sets = Vec::new();
    let mut at = 23;
    for _ in 0..arrays {
        if at + 3 > cfg.len() {
            return Err("hvcC ended inside an array header".into());
        }
        let count = u16::from_be_bytes([cfg[at + 1], cfg[at + 2]]) as usize;
        at += 3;
        for _ in 0..count {
            if at + 2 > cfg.len() {
                return Err("hvcC ended inside a parameter set".into());
            }
            let len = u16::from_be_bytes([cfg[at], cfg[at + 1]]) as usize;
            at += 2;
            if at + len > cfg.len() {
                return Err("hvcC parameter set runs past the box".into());
            }
            sets.push(&cfg[at..at + len]);
            at += len;
        }
    }
    Ok((sets, nal_len))
}

#[cfg(feature = "decode")]
/// A software decoder, fed one coded frame at a time — the same interface as
/// `videotoolbox_decode::VideoToolboxDecoder`.
pub struct NativeDecoder {
    codec: Codec,
    nal_len: usize,
    width: u32,
    height: u32,
    ready: Vec<(i64, DecodedFrame)>,
}

#[cfg(feature = "decode")]
impl NativeDecoder {
    /// Open a decoder for `track`'s codec and configuration.
    pub fn new(track: &VideoTrack) -> Result<Self, String> {
        Self::from_config(track.codec, &track.config, track.width, track.height)
    }

    /// Open a decoder from a decoder configuration alone — an `avcC` or
    /// `hvcC` payload — for streams that do not come from
    /// [`demux_video`](crate::mp4_demux::demux_video): RTSP, a
    /// container of your own. `width` and `height` are what
    /// [`size`](Self::size) reports.
    pub fn from_config(
        codec: TrackCodec,
        config: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let (mut codec, (sets, nal_len)) = match codec {
            TrackCodec::H264 => (Codec::H264(Box::default()), split_avcc(config)?),
            TrackCodec::Hevc => (Codec::Hevc(Box::default()), split_hvcc(config)?),
            TrackCodec::Vp9 => (Codec::Vp9(Box::default()), (Vec::new(), 0)),
            TrackCodec::Av1 => (Codec::Av1(Box::default()), (Vec::new(), 0)),
        };
        for nal in sets {
            codec.decode_nal(nal)?;
        }
        Ok(NativeDecoder {
            codec,
            nal_len,
            width,
            height,
            ready: Vec::new(),
        })
    }

    /// Size of the track, as the container states it.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Hand one coded frame to the decoder, in **decode order**: the bytes
    /// the demuxer pointed at, length-prefixed NAL units as they sit in the
    /// file. The picture it carries is decoded before this returns and
    /// queued under `pts`.
    pub fn push(
        &mut self,
        sample: &[u8],
        pts: i64,
        _dts: i64,
        _duration: i64,
    ) -> Result<(), String> {
        if let Codec::Av1(d) = &mut self.codec {
            // A temporal unit shows at most one picture.
            for (i, f) in d.decode(sample)?.into_iter().enumerate() {
                self.ready.push((pts + i as i64, f.to_decoded()));
            }
            return Ok(());
        }
        if let Codec::Vp9(d) = &mut self.codec {
            // A packet (superframe) shows at most one picture.
            for (i, f) in d.decode(sample)?.into_iter().enumerate() {
                self.ready.push((pts + i as i64, f.to_decoded()?));
            }
            return Ok(());
        }
        for nal in bits::split_length_prefixed(sample, self.nal_len)? {
            self.codec.decode_nal(nal)?;
        }
        for (i, frame) in self.codec.flush().into_iter().enumerate() {
            // One picture per sample is the rule for frame-coded streams; a
            // second would be a muxing error, and is kept just after it.
            self.ready.push((pts + i as i64, frame));
        }
        Ok(())
    }

    /// Take the picture for presentation time `pts` as tightly packed BGRA,
    /// `width * height * 4`, if it has been decoded. Frames before `pts` are
    /// dropped on the way past: they have been shown.
    pub fn take(&mut self, pts: i64) -> Option<Vec<u8>> {
        self.take_frame(pts).map(|f| f.to_bgra())
    }

    /// [`take`](Self::take), but the decoded planes rather than BGRA.
    pub fn take_frame(&mut self, pts: i64) -> Option<DecodedFrame> {
        let hit = self.ready.iter().position(|(p, _)| *p == pts)?;
        let (_, frame) = self.ready.remove(hit);
        self.ready.retain(|(p, _)| *p > pts);
        Some(frame)
    }

    /// Decode H.264 or HEVC pictures on the thread pool; see
    /// [`H264Decoder::set_threads`].
    fn set_threads(&mut self, threads: usize) {
        match &mut self.codec {
            Codec::H264(d) => d.set_threads(threads),
            Codec::Hevc(d) => d.set_threads(threads),
            _ => {}
        }
    }

    fn threaded(&self) -> bool {
        match &self.codec {
            Codec::H264(d) => d.threaded(),
            Codec::Hevc(d) => d.threaded(),
            _ => false,
        }
    }

    fn next_output_id(&self) -> u64 {
        match &self.codec {
            Codec::H264(d) => d.next_output_id(),
            Codec::Hevc(d) => d.next_output_id(),
            _ => 0,
        }
    }

    /// One sample's NAL units, and the end of its picture.
    fn push_nals(&mut self, sample: &[u8]) -> Result<(), String> {
        for nal in bits::split_length_prefixed(sample, self.nal_len)? {
            self.codec.decode_nal(nal)?;
        }
        match &mut self.codec {
            Codec::H264(d) => d.finish_picture(),
            Codec::Hevc(d) => d.finish_picture(),
            _ => {}
        }
        Ok(())
    }

    fn take_finished(&mut self) -> Vec<(u64, Option<DecodedFrame>)> {
        match &mut self.codec {
            Codec::H264(d) => d
                .take_finished()
                .into_iter()
                .map(|(id, p)| (id, p.map(|p| p.frame)))
                .collect(),
            Codec::Hevc(d) => d
                .take_finished()
                .into_iter()
                .map(|(id, p)| (id, p.map(|p| p.frame)))
                .collect(),
            _ => Vec::new(),
        }
    }

    fn wait_for_pictures(&mut self) {
        match &mut self.codec {
            Codec::H264(d) => d.wait_for_pictures(),
            Codec::Hevc(d) => d.wait_for_pictures(),
            _ => {}
        }
    }

    fn take_thread_error(&mut self) -> Option<String> {
        match &mut self.codec {
            Codec::H264(d) => d.take_thread_error(),
            Codec::Hevc(d) => d.take_thread_error(),
            _ => None,
        }
    }

    /// How many decoded pictures are waiting to be taken.
    pub fn queued(&self) -> usize {
        self.ready.len()
    }
}

#[cfg(feature = "decode")]
/// Decode a whole Annex B elementary stream (start-code delimited, as `.264`
/// and `.265` files are) and return its pictures in display order.
pub fn decode_annexb(codec: TrackCodec, stream: &[u8]) -> Result<Vec<DecodedFrame>, String> {
    // (starts a new POC period, POC, frame), in decode order.
    let mut pics: Vec<(bool, i32, DecodedFrame)> = Vec::new();
    match codec {
        TrackCodec::H264 => {
            let mut d = H264Decoder::new();
            d.set_threads(crate::frame_threads());
            for nal in bits::split_annexb(stream) {
                d.decode_nal(nal)?;
            }
            d.finish_picture();
            d.wait_for_pictures();
            d.take_thread_error().map_or(Ok(()), Err)?;
            pics.extend(
                d.take_pictures()
                    .into_iter()
                    .map(|p| (p.idr, p.poc, p.frame)),
            );
        }
        TrackCodec::Vp9 => return Err("VP9 is not an Annex B stream".into()),
        TrackCodec::Av1 => return Err("AV1 is not an Annex B H.26x stream".into()),
        TrackCodec::Hevc => {
            // HEVC defines its output order normatively (§C.5.2), including
            // pictures it discards; follow it rather than sorting by POC.
            let mut d = HevcDecoder::new();
            d.set_threads(crate::frame_threads());
            for nal in bits::split_annexb(stream) {
                d.decode_nal(nal)?;
            }
            d.flush_output();
            d.take_thread_error().map_or(Ok(()), Err)?;
            let mut by_id: std::collections::HashMap<u64, DecodedFrame> = d
                .take_pictures()
                .into_iter()
                .map(|p| (p.id, p.frame))
                .collect();
            return Ok(d
                .take_output_order()
                .into_iter()
                .filter_map(|id| by_id.remove(&id))
                .collect());
        }
    }
    let mut period = 0u32;
    let mut keyed: Vec<((u32, i32), DecodedFrame)> = Vec::with_capacity(pics.len());
    for (i, (starts, poc, frame)) in pics.into_iter().enumerate() {
        if starts && i > 0 {
            period += 1;
        }
        keyed.push(((period, poc), frame));
    }
    keyed.sort_by_key(|(k, _)| *k);
    Ok(keyed.into_iter().map(|(_, f)| f).collect())
}

/// [`decode_video`], with frames in display order: by time, and as they
/// arrive where the stream has none. Only the frames that could still be
/// overtaken are held, so memory does not grow with the stream.
///
/// `each` returns `Ok(false)` to stop early.
#[cfg(feature = "decode")]
pub fn decode_video_in_order(
    data: &[u8],
    mut each: impl FnMut(Option<f64>, DecodedFrame) -> Result<bool, String>,
) -> Result<usize, String> {
    // No codec here reorders further than its largest picture buffer.
    const MAX_REORDER: usize = 16;
    const STOPPED: &str = "\0stopped";
    let mut held: Vec<(f64, DecodedFrame)> = Vec::new();
    let mut emit = |t: Option<f64>, f: DecodedFrame| match each(t, f) {
        Ok(true) => Ok(()),
        Ok(false) => Err(STOPPED.to_string()),
        Err(e) => Err(e),
    };
    let earliest = |held: &[(f64, DecodedFrame)]| {
        (0..held.len())
            .min_by(|&a, &b| held[a].0.total_cmp(&held[b].0))
            .unwrap()
    };
    let mut decoded = decode_video(data, |t, f| {
        let Some(t) = t else { return emit(None, f) };
        held.push((t, f));
        if held.len() > MAX_REORDER {
            let (t, f) = held.remove(earliest(&held));
            emit(Some(t), f)?;
        }
        Ok(())
    });
    while decoded.is_ok() && !held.is_empty() {
        let (t, f) = held.remove(earliest(&held));
        if let Err(e) = emit(Some(t), f) {
            decoded = Err(e);
        }
    }
    match decoded {
        Err(e) if e == STOPPED => Ok(0),
        r => r,
    }
}

/// A whole AV1 stream, frames decoded concurrently where `parallel` allows.
#[cfg(feature = "decode")]
fn decode_av1_units(
    units: &[&[u8]],
    time: impl Fn(usize) -> f64,
    mut each: impl FnMut(Option<f64>, DecodedFrame) -> Result<(), String>,
) -> Result<usize, String> {
    av1::Av1Decoder::new().decode_each(units, crate::frame_threads(), |i, f| {
        each(Some(time(i)), f.to_decoded())
    })
}

/// A stream's samples, one at a time, skipping any that will not decode.
/// Returns how many were skipped, or the first error if nothing decoded.
#[cfg(feature = "decode")]
fn decode_samples<'a>(
    dec: &mut NativeDecoder,
    samples: impl Iterator<Item = Result<(&'a [u8], i64, i64, i64, f64), String>>,
    mut each: impl FnMut(Option<f64>, DecodedFrame) -> Result<(), String>,
) -> Result<usize, String> {
    let (mut shown, mut skipped, mut first_error) = (0usize, 0usize, None);
    for s in samples {
        let (bytes, pts, dts, duration, time) = s?;
        match dec.push(bytes, pts, dts, duration) {
            Ok(()) => {
                if let Some(f) = dec.take_frame(pts) {
                    shown += 1;
                    each(Some(time), f)?;
                }
            }
            Err(e) => {
                skipped += 1;
                first_error.get_or_insert(e);
            }
        }
    }
    match first_error {
        Some(e) if shown == 0 => Err(e),
        _ => Ok(skipped),
    }
}

#[cfg(feature = "decode")]
/// Decode every picture of a video file, whatever holds it: MP4/MOV (H.264,
/// HEVC, AV1), WebM/Matroska (VP9, AV1), IVF (VP9, AV1), or an Annex B
/// H.264/HEVC elementary stream, told apart by its parameter sets.
///
/// `each` gets the pictures in **decode order**, with their presentation time
/// in seconds from the first picture shown; sort by it to show them.
/// Elementary streams carry no timing, so there the time is `None` and the
/// pictures already come in display order.
pub fn decode_video(
    data: &[u8],
    mut each: impl FnMut(Option<f64>, DecodedFrame) -> Result<(), String>,
) -> Result<usize, String> {
    if data.starts_with(b"DKIF") {
        let (fourcc, w, h, (num, den), packets) = crate::mkv_demux::parse_ivf(data)?;
        let codec = match &fourcc {
            b"VP90" => TrackCodec::Vp9,
            b"AV01" => TrackCodec::Av1,
            other => {
                return Err(format!(
                    "IVF codec {} is not supported",
                    String::from_utf8_lossy(other)
                ))
            }
        };
        let first = packets.iter().map(|(pts, _)| *pts).min().unwrap_or(0);
        let time = |pts: i64| (pts - first) as f64 * num as f64 / den.max(1) as f64;
        if codec == TrackCodec::Av1 {
            let units: Vec<&[u8]> = packets.iter().map(|(_, p)| &p[..]).collect();
            return decode_av1_units(&units, |i| time(packets[i].0), each);
        }
        let mut dec = NativeDecoder::from_config(codec, &[], w, h)?;
        return decode_samples(
            &mut dec,
            packets
                .iter()
                .map(|(pts, p)| Ok((&p[..], *pts, *pts, 0, time(*pts)))),
            each,
        );
    }
    if data.starts_with(&[0, 0, 1]) || data.starts_with(&[0, 0, 0, 1]) {
        let codec =
            sniff_annexb(data).ok_or("no H.264 SPS or HEVC VPS in the elementary stream")?;
        for f in decode_annexb(codec, data)? {
            each(None, f)?;
        }
        return Ok(0);
    }
    let (track, packets) = if data.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        // Matroska: the packets come back laid end to end, and the track's
        // sample offsets point into them.
        let (_, track, packets) = crate::ffcompat::mkv::parse(data)?;
        (track, std::borrow::Cow::Owned(packets))
    } else {
        (
            crate::mp4_demux::demux_video(data)?,
            std::borrow::Cow::Borrowed(data),
        )
    };
    let first = track.samples.iter().map(|s| s.pts).min().unwrap_or(0);
    let timescale = track.timescale.max(1) as f64;
    if track.codec == TrackCodec::Av1 {
        let units = track
            .samples
            .iter()
            .map(|s| {
                packets
                    .get(s.offset as usize..s.offset as usize + s.size as usize)
                    .ok_or("sample runs past the end of the file".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        return decode_av1_units(
            &units,
            |i| (track.samples[i].pts - first) as f64 / timescale,
            each,
        );
    }
    let mut dec = NativeDecoder::new(&track)?;
    dec.set_threads(crate::frame_threads());
    let samples = track.samples.iter().map(|s| {
        let bytes = packets
            .get(s.offset as usize..s.offset as usize + s.size as usize)
            .ok_or("sample runs past the end of the file".to_string())?;
        Ok((
            bytes,
            s.pts,
            s.dts,
            s.duration as i64,
            (s.pts - first) as f64 / timescale,
        ))
    });
    if dec.threaded() {
        return decode_samples_threaded(&mut dec, samples, each);
    }
    decode_samples(&mut dec, samples, each)
}

/// [`decode_samples`] with picture jobs: a picture comes out when it has
/// finished, still in decode order, and keeps the time of the sample it was in.
#[cfg(feature = "decode")]
fn decode_samples_threaded<'a>(
    dec: &mut NativeDecoder,
    samples: impl Iterator<Item = Result<(&'a [u8], i64, i64, i64, f64), String>>,
    mut each: impl FnMut(Option<f64>, DecodedFrame) -> Result<(), String>,
) -> Result<usize, String> {
    let mut times = std::collections::HashMap::new();
    let (mut shown, mut skipped, mut first_error) = (0usize, 0usize, None);
    let mut emit = |dec: &mut NativeDecoder,
                    times: &mut std::collections::HashMap<u64, f64>,
                    skipped: &mut usize|
     -> Result<(), String> {
        for (id, frame) in dec.take_finished() {
            let time = times.remove(&id);
            match frame {
                Some(f) => {
                    shown += 1;
                    each(time, f)?;
                }
                None => *skipped += 1,
            }
        }
        Ok(())
    };
    for s in samples {
        let (bytes, _, _, _, time) = s?;
        let first = dec.next_output_id();
        if let Err(e) = dec.push_nals(bytes) {
            skipped += 1;
            first_error.get_or_insert(e);
        }
        for id in first..dec.next_output_id() {
            times.insert(id, time);
        }
        emit(dec, &mut times, &mut skipped)?;
    }
    dec.wait_for_pictures();
    emit(dec, &mut times, &mut skipped)?;
    let error = first_error.or_else(|| dec.take_thread_error());
    match error {
        Some(e) if shown == 0 => Err(e),
        _ => Ok(skipped),
    }
}

/// H.264 or HEVC, from the first NAL unit that can only be one of them: an
/// H.264 SPS (type 7) or an HEVC VPS (type 32, base layer). Each reads as a
/// reserved or unspecified type in the other.
#[cfg(feature = "decode")]
fn sniff_annexb(stream: &[u8]) -> Option<TrackCodec> {
    bits::split_annexb(stream)
        .into_iter()
        .take(64)
        .find_map(|nal| match nal {
            [0x40, 0x01, ..] => Some(TrackCodec::Hevc),
            [b, ..] if b & 0x9f == 0x07 => Some(TrackCodec::H264),
            _ => None,
        })
}
