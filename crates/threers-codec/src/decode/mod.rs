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

pub mod bits;
#[cfg(feature = "decode")]
pub mod cabac;
pub mod h264;
pub mod hevc;
#[cfg(feature = "decode")]
pub mod av1;
#[cfg(feature = "decode")]
pub mod vp9;
pub mod vp9_packet;
#[cfg(feature = "decode")]
pub mod picture;
#[cfg(feature = "decode")]
pub mod platform;

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
    pub fn from_config(codec: TrackCodec, config: &[u8], width: u32, height: u32) -> Result<Self, String> {
        let (mut codec, (sets, nal_len)) = match codec {
            TrackCodec::H264 => (
                Codec::H264(Box::default()),
                split_avcc(config)?,
            ),
            TrackCodec::Hevc => (
                Codec::Hevc(Box::default()),
                split_hvcc(config)?,
            ),
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
    pub fn push(&mut self, sample: &[u8], pts: i64, _dts: i64, _duration: i64) -> Result<(), String> {
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
            for nal in bits::split_annexb(stream) {
                d.decode_nal(nal)?;
            }
            d.finish_picture();
            pics.extend(d.take_pictures().into_iter().map(|p| (p.idr, p.poc, p.frame)));
        }
        TrackCodec::Vp9 => return Err("VP9 is not an Annex B stream".into()),
        TrackCodec::Av1 => return Err("AV1 is not an Annex B H.26x stream".into()),
        TrackCodec::Hevc => {
            // HEVC defines its output order normatively (§C.5.2), including
            // pictures it discards; follow it rather than sorting by POC.
            let mut d = HevcDecoder::new();
            for nal in bits::split_annexb(stream) {
                d.decode_nal(nal)?;
            }
            d.flush_output();
            let mut by_id: std::collections::HashMap<u64, DecodedFrame> =
                d.take_pictures().into_iter().map(|p| (p.id, p.frame)).collect();
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
