//! Native, dependency-free media codecs (behind the `native-codec` feature).
//!
//! Unlike the optional `video` module (which streams frames to a system
//! `ffmpeg` process), everything here is pure Rust with no OS, process,
//! thread, or filesystem dependency — so it builds and runs on `wasm32` as
//! well as native. Encoders return `Vec<u8>` (or write via [`GifWriter`]); the
//! caller owns I/O.
//!
//! - [`crate::codec::bitstream`] — MSB-first bit writer, Exp-Golomb, RBSP/emulation-prevention.
//! - [`crate::codec::h264`] — from-scratch H.264 / AVC encoder (see its module docs).
//! - [`crate::codec::hevc`] — from-scratch HEVC / H.265 encoder (see its module docs).
//! - [`crate::codec::vp9`] — from-scratch VP9 encoder (intra + inter; alpha via WebM).
//! - [`crate::codec::mp4`] / [`crate::codec::webm`] — ISOBMFF and Matroska/WebM container
//!   muxers (H.264 `avc1`, HEVC `hvc1`, VP9 alpha via `BlockAdditional`).
//! - [`crate::codec::apng`] / [`crate::codec::gif`] — animated PNG and GIF89a encode
//!   (GIF also decodes).

pub mod animation;
pub mod apng;
pub mod bitstream;
pub mod gif;
pub mod h264;
pub mod rate;
pub mod hevc;
pub mod mp4;
pub mod vp9;
pub mod webm;

pub use animation::{
    encode_animation_rgba, encode_animation_rgba_with_progress, format_animation_progress,
    AnimationEncodeError, AnimationEncodeOptions, AnimationExportPhase, AnimationExportProgress,
    BrowserCodec,
};
pub use apng::ApngEncoder;
pub use bitstream::{emulation_prevention, rbsp_trailing_bits, BitWriter};
pub use gif::{
    decode_gif, encode_gif, DecodedFrame, DisposalMethod, DisposalMode, GifDecoder, GifEncoder,
    GifError, GifFrameMeta, GifInfo, GifOptions, GifVersion, GifWriter, LzwClearMode, PaletteMode,
    QuantizerKind,
};
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
/// threers::codec::set_frame_memory_budget(1024 << 20);
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

/// [`map_frames`], public so benchmarks and callers with their own frame source
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

