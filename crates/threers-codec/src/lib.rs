//! Pure-Rust media codecs: video and audio encoders, decoders, muxers and
//! demuxers, with hardware acceleration where the platform has it and no C
//! dependencies anywhere else.
//!
//! Everything builds for `wasm32` as well as native and depends on nothing
//! beyond `std` (`rayon` with `parallel`). Outside the opt-in hardware
//! modules (`videotoolbox`, `nvidia`, `vaapi`) there are no OS, process or
//! filesystem calls: encoders return `Vec<u8>`, and the caller owns I/O. It
//! is the media half of [threers](https://crates.io/crates/threers), which
//! re-exports it as `threers::codec`, split out so programs that only encode
//! or decode media do not link a renderer.
//!
//! # Start here
//!
//! [`prelude`] has the types most programs need. Three entry points cover
//! the common jobs:
//!
//! ```no_run
//! # #[cfg(feature = "decode")] {
//! use threers_codec::prelude::*;
//!
//! // Encode RGBA frames to MP4 — on the hardware encoder where there is one.
//! let mut enc = Mp4Encoder::new(640, 360, EncodeOptions::default()).unwrap();
//! enc.push_rgba(&vec![128; 640 * 360 * 4]).unwrap();
//! std::fs::write("out.mp4", enc.finish().unwrap()).unwrap();
//!
//! // Decode any audio file this crate reads to planar f32.
//! let pcm = audio_file::read(&std::fs::read("song.flac").unwrap()).unwrap();
//! println!("{} channels at {} Hz", pcm.channels.len(), pcm.sample_rate);
//!
//! // Decode a video track (feature `decode`), again hardware first.
//! let file = std::fs::read("clip.mp4").unwrap();
//! let track = demux_video(&file).unwrap();
//! let mut dec = Decoder::new(&track).unwrap();
//! for s in &track.samples {
//!     dec.push(&file[s.offset as usize..][..s.size as usize], s.pts, s.dts, s.duration as i64)
//!         .unwrap();
//!     if let Some(_bgra) = dec.take(s.pts) {}
//! }
//! # }
//! ```
//!
//! # What is here
//!
//! | Kind | Modules |
//! |------|---------|
//! | Video encoders | [`h264`] (CAVLC/CABAC, intra + inter), [`hevc`] (intra, alpha), [`vp9`] (intra + inter, alpha) |
//! | Video decoders | [`decode`]: H.264, HEVC, VP9, AV1 — bit-exact against the reference decoders (feature `decode`) |
//! | Audio decoders | [`aac`] (LC, HE-AAC SBR + PS), [`mp3`], [`opus`] (SILK, CELT, hybrid), [`vorbis`], [`flac`], [`pcm`] (WAV/RF64, AIFF); [`audio_file`] sniffs and dispatches |
//! | Animated images | [`gif`] (encode and decode, feature `gif`), [`apng`]; [`animation`] picks one for a browser |
//! | Containers | [`mp4`] / [`mp4_stream`] (write), [`mp4_demux`] (read), [`webm`] (write), [`mkv_demux`] (read, plus IVF), [`ogg`] (read) |
//! | Hardware | [`hw`]: one encoder and one decoder over VideoToolbox, NVENC/NVDEC, VAAPI and the native codecs; `videotoolbox`, `videotoolbox_decode` (macOS), `nvidia` (Linux, Windows) and `vaapi` (Linux) directly |
//! | Rate control | [`rate`]: a quantizer or a bitrate target |
//! | Conversion | [`swr`] (resampling and sample formats), [`ffcompat`] (`swscale`, `.mov`/`.mkv` readers, the shapes ffmpeg's tools produce) |
//! | Captions | [`captions`]: tracks and cues, SRT and WebVTT, muxed by `mp4` and `webm` as soft subtitles |
//! | Primitives | [`bitstream`] (bit writer, Exp-Golomb, emulation prevention), [`contrast`] (WCAG luminance) |
//!
//! # Deep access
//!
//! Every stage of every codec is a public module, so the building blocks are
//! there for research, tooling and tests as well as the high-level calls: an
//! entropy coder ([`h264::cabac`], [`hevc::cabac`], [`vp9::bool_coder`],
//! [`decode::av1::msac`]), a transform ([`hevc::transform`],
//! [`decode::av1::itx`]), a loop filter ([`h264::deblock`],
//! [`decode::vp9::lpf`]), a parameter-set parser ([`decode::h264::params`],
//! [`decode::hevc::params`]), an MDCT ([`aac::dsp`], [`opus::celt`]). Each
//! module's docs say which section of which specification it implements.
//! These inner modules follow the structure of their specifications rather
//! than a stability promise; the high-level types re-exported from each
//! codec's root and from [`prelude`] are the stable surface.
//!
//! # Hardware acceleration
//!
//! [`hw::Mp4Encoder`] and [`decode::Decoder`] use VideoToolbox on macOS (the
//! `videotoolbox` feature), an NVIDIA GPU's NVENC and NVDEC engines on
//! Linux and Windows (the `nvidia` feature), and VA drivers — AMD, Intel —
//! on Linux (the `vaapi` feature), and fall back to the
//! pure-Rust codecs when the feature is off, the machine has no such
//! hardware, or the hardware refuses a stream. The NVIDIA and VA libraries
//! are loaded at run time, so a binary built with those features still runs,
//! natively, where they are missing.
//! The software paths use every core with `parallel`: frames of an all-intra
//! stream are independent, so batches of them encode at once with
//! byte-identical output (see [`set_frame_memory_budget`]).
//!
//! # Features
//!
//! | Feature | Default | What it adds |
//! |---------|---------|--------------|
//! | `decode` | **on** | The video decoders in [`decode`] (H.264, HEVC, VP9, AV1) |
//! | `gif` | **on** | GIF89a encode and decode ([`gif`]), and GIF in [`animation`] |
//! | `parallel` | | Frame-parallel encoding on `rayon` |
//! | `videotoolbox` | | VideoToolbox encode and decode on macOS (links Apple's frameworks; inert elsewhere) |
//! | `nvidia` | | NVENC encode and NVDEC decode on Linux and Windows (driver loaded at run time; implies `decode`; inert elsewhere) |
//! | `vaapi` | | VAAPI HEVC encode and H.264/HEVC/VP9/AV1 decode on Linux (libva loaded at run time; implies `decode`; inert elsewhere) |
//!
//! Encoders, audio decoders, muxers, demuxers and captions are always
//! compiled; none of them has a dependency to make optional.

// The codecs follow their specifications' notation: loops over indices that
// address several parallel arrays, argument lists that mirror the spec's
// processes, and the reference decoders' own nested types.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop, clippy::type_complexity)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod aac;
pub mod animation;
pub mod apng;
pub mod audio_file;
pub mod bitstream;
pub mod captions;
pub mod contrast;
pub mod decode;
pub mod ffcompat;
pub mod flac;
#[cfg(feature = "gif")]
pub mod gif;
pub mod h264;
pub mod hevc;
pub mod hw;
pub mod mkv_demux;
pub mod mp3;
pub mod mp4;
pub mod mp4_demux;
pub mod mp4_stream;
#[cfg(all(feature = "nvidia", any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
pub mod nvidia;
#[cfg(all(feature = "vaapi", target_os = "linux"))]
pub mod vaapi;
pub mod ogg;
pub mod opus;
pub mod pcm;
pub mod prelude;
pub mod rate;
pub mod swr;
#[cfg(all(feature = "videotoolbox", target_os = "macos"))]
pub mod videotoolbox;
#[cfg(all(feature = "videotoolbox", target_os = "macos"))]
pub mod videotoolbox_decode;
pub mod vorbis;
pub mod vp9;
pub mod webm;

pub use animation::{
    encode_animation_rgba, encode_animation_rgba_with_progress, format_animation_progress,
    AnimationEncodeError, AnimationEncodeOptions, AnimationExportPhase, AnimationExportProgress,
    BrowserCodec,
};
pub use apng::ApngEncoder;
pub use bitstream::{emulation_prevention, rbsp_trailing_bits, BitWriter};
#[cfg(feature = "gif")]
pub use gif::{
    decode_gif, encode_gif, DisposalMethod, DisposalMode, GifDecoder, GifEncoder, GifError,
    GifFrame, GifFrameMeta, GifInfo, GifOptions, GifVersion, GifWriter, LzwClearMode, PaletteMode,
    QuantizerKind,
};
#[cfg(feature = "gif")]
#[allow(deprecated)]
pub use gif::DecodedFrame;
pub use vp9::{encode_intra_frame, encode_intra_gray, Reconstruction};
pub use webm::{encode_gray_webm, encode_webm, mux_webm, WebmCodec, WebmFrame, WebmParams};

// --------------------------------------------------------------------------
// Frame-level parallelism
// --------------------------------------------------------------------------

/// How many frames to hold in flight at once.
///
/// Every frame of an all-intra stream is independent, so the only thing bounding
/// the batch is memory: an 8K frame is 50 MB of input and the encoder allocates
/// a reconstruction of its own alongside it. Encoding every frame at once would
/// undo the streaming work that got export memory down in the first place, so
/// the batch is whichever is smaller — the thread count, or what fits in a
/// fixed working-set budget.
pub(crate) fn frames_in_flight(width: u32, height: u32) -> usize {
    frames_by_memory(width, height).min(worker_threads())
}

/// How many frames the memory budget allows in flight, irrespective of how many
/// cores there are.
///
/// Rate control sizes its groups from this rather than from
/// [`frames_in_flight`], because the group is how often the controller gets to
/// correct itself — and a clip must not encode to a different size on a machine
/// with more cores.
pub(crate) fn frames_by_memory(width: u32, height: u32) -> usize {
    (frame_memory_budget() / frame_working_set(width, height).max(1)).max(1)
}

/// Roughly what one frame costs while it is being encoded: the input planes,
/// the reconstruction, and the padded copies the encoder makes of both.
pub(crate) fn frame_working_set(width: u32, height: u32) -> usize {
    (width as usize * height as usize * 3 / 2) * 5 / 2
}

/// Default working-set budget for frame-parallel encoding.
///
/// Deliberately conservative: it binds only at 4K and above, and only there
/// because a frame is large enough that holding a core's worth of them is real
/// memory. See [`set_frame_memory_budget`].
pub const DEFAULT_FRAME_MEMORY_BUDGET: usize = 256 << 20;

static FRAME_MEMORY_BUDGET: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(DEFAULT_FRAME_MEMORY_BUDGET);

/// How much memory frame-parallel encoding may hold in flight.
pub fn frame_memory_budget() -> usize {
    FRAME_MEMORY_BUDGET.load(std::sync::atomic::Ordering::Relaxed)
}

/// Set how much memory frame-parallel encoding may hold in flight.
///
/// Frames of an all-intra stream are independent, so the only thing limiting how
/// many encode at once is memory — about 4.5 bytes per pixel per frame in
/// flight, which is 149 MB at 8K. The default of 256 MB keeps peak usage close
/// to where the streaming work left it, at the cost of capping 8K to two frames
/// at a time.
///
/// Raising it is the cheapest speedup available at high resolution and it costs
/// nothing in bitrate: measured on a 4-frame 8K clip, going from 256 MB to 1 GB
/// took the encode from 49.9 s to 33.0 s with byte-identical output. Tiling the
/// picture would reach the same parallelism at a bitrate cost; this does not.
///
/// ```
/// // Let an 8K export use eight frames at a time.
/// threers_codec::set_frame_memory_budget(1024 << 20);
/// ```
pub fn set_frame_memory_budget(bytes: usize) {
    FRAME_MEMORY_BUDGET.store(bytes.max(1), std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "parallel")]
fn worker_threads() -> usize {
    rayon::current_num_threads().max(1)
}

#[cfg(not(feature = "parallel"))]
fn worker_threads() -> usize {
    1
}

/// Encode a batch of frames, in order, using every available thread.
///
/// The output is byte-identical to mapping `encode` over the frames one at a
/// time — the `parallel` feature changes how long it takes and nothing else.
#[cfg(feature = "parallel")]
pub(crate) fn map_frames<T, R>(frames: &[T], encode: impl Fn(&T) -> R + Sync + Send) -> Vec<R>
where
    T: Sync,
    R: Send,
{
    use rayon::prelude::*;
    frames.par_iter().map(encode).collect()
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn map_frames<T, R>(frames: &[T], encode: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    frames.iter().map(encode).collect()
}

/// `map_frames`, public so benchmarks and callers with their own frame source
/// can use the same batching the export paths do.
pub fn map_frames_pub<T, R>(frames: &[T], encode: impl Fn(&T) -> R + Sync + Send) -> Vec<R>
where
    T: Sync,
    R: Send,
{
    map_frames(frames, encode)
}

/// [`map_frames`] where the work needs to know its own index — tiles do, to
/// tell the last one from the rest.
#[cfg(feature = "parallel")]
pub(crate) fn map_indexed<T, R>(items: &[T], f: impl Fn(usize, &T) -> R + Sync + Send) -> Vec<R>
where
    T: Sync,
    R: Send,
{
    use rayon::prelude::*;
    items.par_iter().enumerate().map(|(i, t)| f(i, t)).collect()
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn map_indexed<T, R>(items: &[T], f: impl Fn(usize, &T) -> R + Sync + Send) -> Vec<R> {
    items.iter().enumerate().map(|(i, t)| f(i, t)).collect()
}

