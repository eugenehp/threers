//! A video file's pictures one at a time, in display order, for playback.
//!
//! [`decode_video`](super::decode_video) walks a whole file and calls back
//! with every picture, in decode order. A player wants the opposite: the next
//! picture to show, when it is due, and nothing held that is not needed. That
//! is [`VideoReader`]: open a file, call [`next_frame`](VideoReader::next_frame)
//! for each picture in display order, and [`rewind`](VideoReader::rewind) to go
//! round again.
//!
//! ```no_run
//! use threers_codec::decode::VideoReader;
//!
//! let mut video = VideoReader::open(std::fs::read("clip.mp4").unwrap()).unwrap();
//! while let Some((seconds, frame)) = video.next_frame().unwrap() {
//!     let rgba = frame.to_rgba();
//!     let _ = (seconds, rgba);
//! }
//! ```
//!
//! The containers are those [`decode_video`](super::decode_video) reads:
//! MP4/MOV (H.264, HEVC, AV1), WebM/Matroska (VP9, AV1) and IVF (VP9, AV1).
//! An Annex B elementary stream has no timing and no index, so it is decoded
//! whole when opened and shown at 30 frames a second.
//!
//! Pictures decode in hardware where the build and machine have it:
//! VideoToolbox on macOS (`videotoolbox`; H.264, HEVC, AV1 on M3 and later,
//! VP9), NVDEC on NVIDIA (`nvidia`) and VAAPI on Linux (`vaapi`), in that
//! order, with the native decoders behind them for anything the hardware will
//! not take. [`backend`](VideoReader::backend) says which is in use;
//! [`open_software`](VideoReader::open_software) keeps to the native ones.
//!
//! Display order comes from the container's timestamps, which are all known
//! before the first picture is decoded: a picture goes out as soon as every
//! sample that could show earlier has been decoded. A stream with B-frames
//! holds back only the pictures it reorders, not a fixed window.

use super::{Backend, DecodedFrame, NativeDecoder, Picture};
use crate::mp4_demux::TrackCodec;
use std::collections::VecDeque;
use crate::bytes::{Bytes, FramePool};

/// Frame rate assumed for streams that carry no timing.
const UNTIMED_FPS: f64 = 30.0;

/// One coded sample: where its bytes are, its timestamps, and when it shows.
#[derive(Debug, Clone, Copy)]
pub struct CodedSample {
    /// Where its bytes start in [`Demuxed::bytes`].
    pub start: usize,
    /// Where they end.
    pub end: usize,
    /// Presentation time, in the track's units.
    pub pts: i64,
    /// Decode time, in the track's units.
    pub dts: i64,
    /// Duration, in the track's units; 0 where the container has none.
    pub duration: i64,
    /// When it shows, in seconds from the first picture shown.
    pub seconds: f64,
}

type Sample = CodedSample;

/// A video track as a decoder is fed it: the codec and its configuration
/// record, and every coded sample in decode order with when it shows. What
/// [`VideoReader`] reads, and what a platform decoder outside this crate (the
/// browser's WebCodecs, say) can be given instead; see
/// [`webcodecs`](super::webcodecs).
#[derive(Debug, Clone)]
pub struct Demuxed {
    /// The codec.
    pub codec: TrackCodec,
    /// The configuration record (`avcC`, `hvcC`, `av1C`, `vpcC`) as the
    /// container carries it; empty where it carries none (IVF, VP9 in
    /// Matroska).
    pub config: Vec<u8>,
    /// Picture width the container states.
    pub width: u32,
    /// Picture height.
    pub height: u32,
    /// The samples' bytes: the file itself for MP4 (mapped, if it was
    /// opened that way), the packets laid end to end for Matroska and IVF.
    pub bytes: Bytes,
    /// Every sample, in decode order.
    pub samples: Vec<CodedSample>,
}

impl Demuxed {
    /// Sample `i`'s bytes.
    pub fn sample(&self, i: usize) -> &[u8] {
        let s = &self.samples[i];
        &self.bytes[s.start..s.end]
    }

    /// From the first picture shown to the end of the last, in seconds.
    pub fn duration(&self) -> f64 {
        clip_length(&self.samples)
    }
}

/// Demux MP4/MOV, WebM/Matroska or IVF held in memory. An Annex B stream
/// has no container to demux and is refused.
pub fn demux(data: impl Into<Bytes>) -> Result<Demuxed, String> {
    let data: Bytes = data.into();
    let (codec, config, width, height, bytes, samples) = if data.starts_with(b"DKIF") {
        let (fourcc, w, h, (num, den), packets) = crate::mkv_demux::index_ivf(&data)?;
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
        let tick = num as f64 / den.max(1) as f64;
        // The packets stay where they are in the file.
        let samples = packets
            .iter()
            .map(|(pts, at)| Sample {
                start: at.start,
                end: at.end,
                pts: *pts,
                dts: *pts,
                duration: 0,
                seconds: (pts - first) as f64 * tick,
            })
            .collect();
        (codec, Vec::new(), w, h, data, samples)
    } else if data.starts_with(&[0, 0, 1]) || data.starts_with(&[0, 0, 0, 1]) {
        return Err("an Annex B elementary stream has no container to demux".into());
    } else {
        let (track, bytes) = if data.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
            // Matroska: the track's sample offsets point into the file,
            // or, for a track with header stripping, into its packets laid
            // end to end.
            let (_, track, packets) = crate::ffcompat::mkv::parse_in_place(&data)?;
            (track, packets.map_or(data, Bytes::from))
        } else {
            (crate::mp4_demux::demux_video(&data)?, data)
        };
        let first = track.samples.iter().map(|s| s.pts).min().unwrap_or(0);
        let timescale = track.timescale.max(1) as f64;
        let samples = track
            .samples
            .iter()
            .map(|s| {
                let start = s.offset as usize;
                let end = start + s.size as usize;
                if end > bytes.len() {
                    return Err("sample runs past the end of the file".to_string());
                }
                Ok(Sample {
                    start,
                    end,
                    pts: s.pts,
                    dts: s.dts,
                    duration: s.duration as i64,
                    seconds: (s.pts - first) as f64 / timescale,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        (track.codec, track.config, track.width, track.height, bytes, samples)
    };
    Ok(Demuxed {
        codec,
        config,
        width,
        height,
        bytes,
        samples,
    })
}

/// What the pictures come from.
enum Source {
    /// A demuxed track, decoded a sample at a time.
    Samples {
        bytes: Bytes,
        samples: Vec<Sample>,
        /// The earliest display time among `samples[i..]`, and infinity past
        /// the end: what a waiting picture has to be earlier than to go out.
        earliest_after: Vec<f64>,
        codec: TrackCodec,
        config: Vec<u8>,
        decoder: SampleDecoder,
        next: usize,
        /// Decoded, not yet shown, earliest first.
        waiting: VecDeque<(f64, Picture)>,
    },
    /// An elementary stream, decoded whole when opened, and again on a rewind.
    Decoded {
        data: Bytes,
        codec: TrackCodec,
        frames: Vec<Option<DecodedFrame>>,
        next: usize,
    },
}

/// The decoder samples go through: the hardware's, or the native one.
enum SampleDecoder {
    Native(Box<NativeDecoder>),
    /// The native H.264 or HEVC decoder reconstructing several pictures at
    /// once on the thread pool. Pictures come out as they finish, so each is
    /// keyed by its output id until then.
    Threaded {
        decoder: Box<NativeDecoder>,
        /// Output id to display time, for pictures not finished yet.
        in_flight: std::collections::HashMap<u64, f64>,
    },
    #[cfg(all(feature = "videotoolbox", target_os = "macos"))]
    VideoToolbox(crate::videotoolbox_decode::VideoToolboxDecoder),
    #[cfg(all(
        feature = "nvidia",
        any(target_os = "linux", target_os = "windows"),
        target_pointer_width = "64"
    ))]
    Nvdec(Box<crate::nvidia::NvdecDecoder>),
    #[cfg(all(feature = "vaapi", target_os = "linux"))]
    Vaapi(Box<crate::vaapi::VaapiDecoder>),
}

impl SampleDecoder {
    /// The hardware's where it is wanted and takes the stream, else native.
    /// `first` is the first coded sample, which describes a VP9 stream that
    /// came with no `vpcC` (Matroska carries none).
    fn open(
        codec: TrackCodec,
        config: &[u8],
        width: u32,
        height: u32,
        hardware: bool,
        first: Option<&[u8]>,
        pool: &FramePool,
    ) -> Result<Self, String> {
        #[cfg(all(feature = "videotoolbox", target_os = "macos"))]
        if hardware {
            let described = match (codec, config.is_empty(), first) {
                (TrackCodec::Vp9, true, Some(frame)) => vp9_record(frame, width, height).unwrap_or_default(),
                _ => config.to_vec(),
            };
            let track = crate::mp4_demux::VideoTrack {
                codec,
                width,
                height,
                config: described,
                timescale: TIMESCALE,
                samples: Vec::new(),
            };
            if let Ok(mut d) = crate::videotoolbox_decode::VideoToolboxDecoder::new_planar(&track) {
                d.set_pool(pool.clone());
                return Ok(Self::VideoToolbox(d));
            }
        }
        // NVDEC, then VAAPI: both are loaded at run time and refuse when the
        // machine has no driver.
        #[cfg(any(
            all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"),
            all(feature = "vaapi", target_os = "linux")
        ))]
        if hardware {
            let track = crate::mp4_demux::VideoTrack {
                codec,
                width,
                height,
                config: config.to_vec(),
                timescale: 90_000,
                samples: Vec::new(),
            };
            #[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
            if Backend::Nvdec.available() {
                if let Ok(mut d) = crate::nvidia::NvdecDecoder::new(&track) {
                    d.set_pool(pool.clone());
                    return Ok(Self::Nvdec(Box::new(d)));
                }
            }
            #[cfg(all(feature = "vaapi", target_os = "linux"))]
            if Backend::Vaapi.available() {
                if let Ok(mut d) = crate::vaapi::VaapiDecoder::new(&track) {
                    d.set_pool(pool.clone());
                    return Ok(Self::Vaapi(Box::new(d)));
                }
            }
        }
        let _ = (hardware, first, pool);
        let mut decoder = NativeDecoder::from_config(codec, config, width, height)?;
        decoder.set_threads(crate::frame_threads());
        if decoder.threaded() {
            return Ok(Self::Threaded {
                decoder: Box::new(decoder),
                in_flight: Default::default(),
            });
        }
        Ok(Self::Native(Box::new(decoder)))
    }

    /// The earliest display time among pictures still being reconstructed:
    /// nothing later can go out before it has.
    fn earliest_in_flight(&self) -> f64 {
        match self {
            Self::Threaded { in_flight, .. } => in_flight.values().copied().fold(f64::INFINITY, f64::min),
            _ => f64::INFINITY,
        }
    }

    /// Pictures that finished on the thread pool since last asked, with
    /// their display times.
    fn finished(&mut self) -> Vec<(f64, Picture)> {
        let Self::Threaded { decoder, in_flight } = self else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (id, frame) in decoder.take_finished() {
            let at = in_flight.remove(&id);
            if let (Some(at), Some(frame)) = (at, frame) {
                out.push((at, Picture::Planes(frame)));
            }
        }
        out
    }

    /// Wait for every picture in flight, when there is nothing left to push.
    fn wait(&mut self) -> bool {
        match self {
            Self::Threaded { decoder, in_flight } if !in_flight.is_empty() => {
                decoder.wait_for_pictures();
                true
            }
            _ => false,
        }
    }

    fn backend(&self) -> Backend {
        match self {
            Self::Native(_) | Self::Threaded { .. } => Backend::Native,
            #[cfg(all(feature = "videotoolbox", target_os = "macos"))]
            Self::VideoToolbox(_) => Backend::VideoToolbox,
            #[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
            Self::Nvdec(_) => Backend::Nvdec,
            #[cfg(all(feature = "vaapi", target_os = "linux"))]
            Self::Vaapi(_) => Backend::Vaapi,
        }
    }

    /// `seconds` is when the sample shows, for a threaded decoder to key
    /// its pictures by.
    fn push(&mut self, sample: &[u8], pts: i64, dts: i64, duration: i64, seconds: f64) -> Result<(), String> {
        match self {
            Self::Native(d) => d.push(sample, pts, dts, duration),
            Self::Threaded { decoder, in_flight } => {
                let first = decoder.next_output_id();
                let pushed = decoder.push_nals(sample);
                for id in first..decoder.next_output_id() {
                    in_flight.insert(id, seconds);
                }
                pushed
            }
            #[cfg(all(feature = "videotoolbox", target_os = "macos"))]
            Self::VideoToolbox(d) => d.push(sample, pts, dts, duration),
            #[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
            Self::Nvdec(d) => d.push(sample, pts, dts, duration),
            #[cfg(all(feature = "vaapi", target_os = "linux"))]
            Self::Vaapi(d) => d.push(sample, pts, dts, duration),
        }
    }

    /// The picture for `pts`, packed as the hardware left it where it
    /// did.
    fn take_frame(&mut self, pts: i64) -> Option<Picture> {
        match self {
            Self::Native(d) => d.take_frame(pts).map(Picture::Planes),
            // Pictures come out through `finished`.
            Self::Threaded { .. } => None,
            #[cfg(all(feature = "videotoolbox", target_os = "macos"))]
            Self::VideoToolbox(d) => d.take_packed(pts).map(Picture::Packed),
            #[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
            Self::Nvdec(d) => d.take_packed(pts).map(Picture::Packed),
            #[cfg(all(feature = "vaapi", target_os = "linux"))]
            Self::Vaapi(d) => d.take_picture(pts),
        }
    }
}

/// A `vpcC` record for a profile 0 stream (8-bit 4:2:0, the usual VP9) from
/// its first frame's header; `None` for the other profiles, which stay on
/// the native decoder. Colour is left unspecified, as the frame header has it
/// in a form the record does not.
pub(crate) fn vp9_record(frame: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let b = *frame.first()?;
    // frame_marker (2 bits, 0b10), profile_low_bit, profile_high_bit.
    if b >> 6 != 0b10 || (b >> 5) & 1 != 0 || (b >> 4) & 1 != 0 {
        return None;
    }
    let luma = width as u64 * height as u64;
    let level = if luma <= 2_228_224 {
        41
    } else if luma <= 8_912_896 {
        51
    } else {
        62
    };
    // Version 1, no flags; profile 0; level; 8 bits, 4:2:0 colocated, video
    // range; primaries, transfer and matrix unspecified (2); no init data.
    Some(vec![1, 0, 0, 0, 0, level, (8 << 4) | (1 << 1), 2, 2, 2, 0, 0])
}

/// The timescale handed to a hardware session. Only `pts` round-trips
/// through it, as a key, so any will do.
#[cfg(all(feature = "videotoolbox", target_os = "macos"))]
const TIMESCALE: u32 = 90_000;

/// A video file, read a picture at a time in display order. See the
/// [module docs](self).
pub struct VideoReader {
    source: Source,
    width: u32,
    height: u32,
    duration: f64,
    frames: usize,
    hardware: bool,
    /// The hardware decoders' picture buffers, given back by the caller.
    pool: FramePool,
}

impl VideoReader {
    /// Open a file held in memory, decoding in hardware where this build and
    /// machine can. Fails if the container or codec is not one this crate
    /// reads, or the stream's configuration does not parse.
    pub fn open(data: Vec<u8>) -> Result<Self, String> {
        Self::open_with(data.into(), true)
    }

    /// Open the file at `path`, mapped rather than read: the operating
    /// system pages in what decoding touches as it touches it, so a film of
    /// many gigabytes starts at once and is never all in memory. See
    /// [`Bytes::map`].
    pub fn open_file(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let data = Bytes::map(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::open_with(data, true)
    }

    /// As [`open`](Self::open), holding the bytes shared rather than owned:
    /// for a caller that reads them again, as a player decoding the
    /// soundtrack does, without a second copy of the file.
    pub fn open_shared(data: impl Into<Bytes>) -> Result<Self, String> {
        Self::open_with(data.into(), true)
    }

    /// As [`open`](Self::open), with the native decoders only.
    pub fn open_software(data: Vec<u8>) -> Result<Self, String> {
        Self::open_with(data.into(), false)
    }

    fn open_with(data: Bytes, hardware: bool) -> Result<Self, String> {
        if data.starts_with(&[0, 0, 1]) || data.starts_with(&[0, 0, 0, 1]) {
            let codec = super::sniff_annexb(&data)
                .ok_or("no H.264 SPS or HEVC VPS in the elementary stream")?;
            let frames: Vec<_> = super::decode_annexb(codec, &data)?.into_iter().map(Some).collect();
            let (width, height) = frames
                .first()
                .and_then(|f| f.as_ref())
                .map_or((0, 0), |f| (f.width, f.height));
            let count = frames.len();
            return Ok(Self {
                source: Source::Decoded {
                    data,
                    codec,
                    frames,
                    next: 0,
                },
                width,
                height,
                duration: count as f64 / UNTIMED_FPS,
                frames: count,
                hardware: false,
                pool: FramePool::default(),
            });
        }

        let Demuxed {
            codec,
            config,
            width,
            height,
            bytes,
            samples,
        } = demux(data)?;

        let mut earliest_after = vec![f64::INFINITY; samples.len() + 1];
        for i in (0..samples.len()).rev() {
            earliest_after[i] = earliest_after[i + 1].min(samples[i].seconds);
        }
        let duration = clip_length(&samples);
        let frames = samples.len();
        let first = samples.first().map(|s| &bytes[s.start..s.end]);
        let pool = FramePool::default();
        let decoder = SampleDecoder::open(codec, &config, width, height, hardware, first, &pool)?;
        Ok(Self {
            source: Source::Samples {
                bytes,
                samples,
                earliest_after,
                codec,
                config,
                decoder,
                next: 0,
                waiting: VecDeque::new(),
            },
            width,
            height,
            duration,
            frames,
            hardware,
            pool,
        })
    }

    /// The pool the hardware decoders copy pictures into. Give a
    /// [`Picture::Packed`]'s buffers back here once it has been shown
    /// ([`recycle`](Self::recycle) does, or keep this handle on another
    /// thread and call [`FramePool::give`]): a few buffers then serve the
    /// whole film, instead of a fresh allocation a picture, which at 8K costs
    /// more than the copy. Giving nothing back is fine, only slower.
    pub fn frame_pool(&self) -> FramePool {
        self.pool.clone()
    }

    /// Give a picture's buffers back to [`frame_pool`](Self::frame_pool).
    pub fn recycle(&self, picture: Picture) {
        if let Picture::Packed(frame) = picture {
            self.pool.give(frame.y);
            self.pool.give(frame.uv);
        }
    }

    /// Which decoder is doing the work.
    pub fn backend(&self) -> Backend {
        match &self.source {
            Source::Samples { decoder, .. } => decoder.backend(),
            Source::Decoded { .. } => Backend::Native,
        }
    }

    /// The picture size the container states.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// How long the clip runs, in seconds: from the first picture shown to the
    /// end of the last.
    pub fn duration(&self) -> f64 {
        self.duration
    }

    /// How many coded pictures the file holds.
    pub fn frame_count(&self) -> usize {
        self.frames
    }

    /// The next picture in display order and when it shows, in seconds from
    /// the first; `None` at the end. A sample that fails to decode is skipped,
    /// as a player would, unless nothing has decoded at all.
    pub fn next_frame(&mut self) -> Result<Option<(f64, DecodedFrame)>, String> {
        Ok(self.next_picture()?.map(|(at, p)| (at, p.into_decoded())))
    }

    /// [`next_frame`](Self::next_frame), with the picture as the decoder
    /// left it: packed NV12 or P010 from a hardware decoder, which a GPU can
    /// convert without a pass over it on the CPU; planes otherwise.
    pub fn next_picture(&mut self) -> Result<Option<(f64, Picture)>, String> {
        let (width, height) = (self.width, self.height);
        match &mut self.source {
            Source::Decoded { frames, next, .. } => {
                if *next >= frames.len() {
                    return Ok(None);
                }
                let at = *next;
                *next += 1;
                // Taken rather than cloned; `rewind` decodes the stream again.
                Ok(frames[at].take().map(|f| (at as f64 / UNTIMED_FPS, Picture::Planes(f))))
            }
            Source::Samples {
                bytes,
                samples,
                earliest_after,
                codec,
                config,
                decoder,
                next,
                waiting,
            } => loop {
                for (at, frame) in decoder.finished() {
                    let i = waiting.partition_point(|(t, _)| *t <= at);
                    waiting.insert(i, (at, frame));
                }
                // The earliest waiting picture can go once nothing still to
                // decode, or still being reconstructed, shows before it.
                if let Some((seconds, _)) = waiting.front() {
                    if (*seconds < earliest_after[*next] || *next == samples.len())
                        && *seconds <= decoder.earliest_in_flight()
                    {
                        return Ok(waiting.pop_front());
                    }
                }
                if *next == samples.len() {
                    if decoder.wait() {
                        continue;
                    }
                    return Ok(waiting.pop_front());
                }
                let s = samples[*next];
                *next += 1;
                let mut pushed = decoder.push(&bytes[s.start..s.end], s.pts, s.dts, s.duration, s.seconds);
                let mut frame = pushed.as_ref().ok().and_then(|_| decoder.take_frame(s.pts));
                // A hardware decoder that takes the stream and then gives
                // nothing for its first picture (NVDEC with VP9 below its
                // smallest size does) is no use: the native one starts over.
                if *next == 1 && frame.is_none() && decoder.backend() != Backend::Native {
                    *decoder = SampleDecoder::open(*codec, config, width, height, false, None, &self.pool)?;
                    pushed = decoder.push(&bytes[s.start..s.end], s.pts, s.dts, s.duration, s.seconds);
                    frame = pushed.as_ref().ok().and_then(|_| decoder.take_frame(s.pts));
                }
                match pushed {
                    Ok(()) => {
                        if let Some(frame) = frame {
                            let at = waiting.partition_point(|(t, _)| *t <= s.seconds);
                            waiting.insert(at, (s.seconds, frame));
                        }
                    }
                    // Nothing shown yet and nothing left: say why.
                    Err(e) if *next == samples.len() && waiting.is_empty() => return Err(e),
                    Err(_) => {}
                }
            },
        }
    }

    /// Back to the first picture.
    pub fn rewind(&mut self) -> Result<(), String> {
        match &mut self.source {
            Source::Decoded {
                data,
                codec,
                frames,
                next,
            } => {
                if *next > 0 {
                    // The pictures handed out were moved, not copied.
                    *frames = super::decode_annexb(*codec, data)?.into_iter().map(Some).collect();
                    *next = 0;
                }
            }
            Source::Samples {
                bytes,
                samples,
                codec,
                config,
                decoder,
                next,
                waiting,
                ..
            } => {
                let first = samples.first().map(|s| &bytes[s.start..s.end]);
                *decoder = SampleDecoder::open(*codec, config, self.width, self.height, self.hardware, first, &self.pool)?;
                *next = 0;
                waiting.clear();
            }
        }
        Ok(())
    }
}

/// The span from the first picture shown to the end of the last: the last
/// display time plus one frame, the frame being the last sample's stated
/// duration where it has one and the typical spacing otherwise.
fn clip_length(samples: &[Sample]) -> f64 {
    let mut times: Vec<f64> = samples.iter().map(|s| s.seconds).collect();
    times.sort_by(f64::total_cmp);
    let Some(&last) = times.last() else { return 0.0 };
    let mut gaps: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).filter(|g| *g > 0.0).collect();
    gaps.sort_by(f64::total_cmp);
    let frame = gaps.get(gaps.len() / 2).copied().unwrap_or(1.0 / UNTIMED_FPS);
    last + frame
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Matroska and IVF samples are found where they lie in the file: the
    /// demuxed bytes are the file's own, not a copy.
    #[test]
    fn webm_and_ivf_demux_in_place() {
        use crate::webm::{mux_webm, WebmCodec, WebmFrame, WebmParams};
        let packets: [&[u8]; 3] = [&[1, 2, 3, 4], &[5, 6], &[7, 8, 9]];
        let frames: Vec<WebmFrame> = packets
            .iter()
            .enumerate()
            .map(|(i, &data)| WebmFrame {
                data,
                alpha: None,
                timecode: i as u64 * 40,
                keyframe: i == 0,
            })
            .collect();
        let webm = Bytes::from(mux_webm(&WebmParams {
            width: 64,
            height: 48,
            codec: WebmCodec::Av1,
            timecode_scale_ns: 1_000_000,
            frames: &frames,
        }));

        let mut ivf = b"DKIF\0\0\x20\0AV01\x40\0\x30\0\x19\0\0\0\x01\0\0\0".to_vec();
        ivf.resize(32, 0);
        for (i, p) in packets.iter().enumerate() {
            ivf.extend_from_slice(&(p.len() as u32).to_le_bytes());
            ivf.extend_from_slice(&(i as u64).to_le_bytes());
            ivf.extend_from_slice(p);
        }
        let ivf = Bytes::from(ivf);

        for file in [webm, ivf] {
            let d = demux(file.clone()).unwrap();
            assert_eq!(d.bytes.as_ptr(), file.as_ptr(), "the file's own bytes");
            let got: Vec<&[u8]> = d.samples.iter().map(|s| &d.bytes[s.start..s.end]).collect();
            assert_eq!(got, packets);
            assert!((d.samples[2].seconds - 0.08).abs() < 1e-9, "{:?}", d.samples);
        }
    }
}
