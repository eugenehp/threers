//! Native video export (behind the `video` feature). Renders a frame sequence
//! and encodes it — by default with the system `ffmpeg`. When both `video` and
//! `native-codec` are enabled, [`VideoCodec::Gif`], [`VideoCodec::Apng`], and
//! [`VideoCodec::H264`] are encoded in-process (no ffmpeg); other codecs still
//! require ffmpeg.
//!
//! For browser / wasm downloads (GIF, APNG, WebM, MP4) use
//! [`crate::encode_animation_rgba`] with [`crate::BrowserCodec`] instead — that
//! API returns bytes and never shells out.
//!
//! Engine-agnostic: you supply a closure that returns the RGBA bytes for a frame
//! index (e.g. from [`HeadlessRenderer::render_to_rgba`](crate::HeadlessRenderer)).
//!
//! Trace progress with [`VideoExporter`] events (mirrors the JS API):
//!
//! ```no_run
//! use threers::{VideoCodec, VideoExporter, VideoExportEvent};
//! # fn render(_i: usize) -> Vec<u8> { vec![0; 4] }
//! VideoExporter::new("out.gif")
//!     .size(64, 64)
//!     .frames(8)
//!     .fps(10)
//!     .codec(VideoCodec::Gif)
//!     .on_event(|ev| match ev {
//!         VideoExportEvent::Progress(p) => eprintln!("{}", p.message),
//!         VideoExportEvent::Complete { output, .. } => eprintln!("wrote {output}"),
//!         _ => {}
//!     })
//!     .export(render)
//!     .unwrap();
//! ```
//!
//! # Subtitles and captions
//!
//! Attach a [`crate::captions::CaptionTrack`] and pick how it ships:
//!
//! ```no_run
//! use threers::{CaptionMode, CaptionTrack, VideoCodec, VideoExporter};
//! # fn render(_i: usize) -> Vec<u8> { vec![0; 1280 * 720 * 4] }
//! let track = CaptionTrack::parse_srt(&std::fs::read_to_string("dialogue.srt").unwrap()).unwrap();
//! VideoExporter::new("out.mp4")
//!     .size(1280, 720)
//!     .frames(300)
//!     .fps(30)
//!     .codec(VideoCodec::H264)
//!     .captions(track)
//!     // Burn the text into the pixels *and* drop `out.srt` next to the video.
//!     .caption_mode(CaptionMode::BurnAndSidecar)
//!     .export(render)
//!     .unwrap();
//! ```
//!
//! | [`CaptionMode`] | Result |
//! |------|--------|
//! | [`Burn`](CaptionMode::Burn) (default) | text composited into the frames; works with every codec |
//! | [`Sidecar`](CaptionMode::Sidecar) | a separate `out.srt` / `out.vtt` |
//! | [`Embed`](CaptionMode::Embed) | a soft subtitle track in the container (MP4 `mov_text`, WebM `webvtt`) |
//! | [`BurnAndSidecar`](CaptionMode::BurnAndSidecar) | both of the first two |
//!
//! Burned-in text uses the built-in bitmap face unless you supply a TrueType
//! font with [`VideoOptions::caption_font`], and scales itself to the export
//! height unless you pin a [`crate::captions::CaptionStyle`].
//!
//! See examples `export_h264`, `export_hevc`, `export_vp9`, `export_gif`,
//! `captions_video`, … and the web demo `web/examples/export-video.html`.
//!
//! ```no_run
//! use threers::{HeadlessRenderer, VideoOptions, VideoCodec, export_video, Scene, PerspectiveCamera};
//! let (w, h) = (1280, 720);
//! let mut hr = HeadlessRenderer::builder().size(w, h).build().unwrap();
//! let mut scene = Scene::new();
//! let cam = PerspectiveCamera::new(50.0, w as f32 / h as f32, 0.1, 1000.0);
//! let opts = VideoOptions::new("out.mp4").fps(30).codec(VideoCodec::H264);
//! export_video(w, h, 90, &opts, |_frame| {
//!     // ...advance the scene for this frame...
//!     hr.render_to_rgba(&mut scene, &cam)
//! }).unwrap();
//! ```

/// Called with export progress as frames are written.
type ProgressCallback = Box<dyn FnMut(&VideoExportProgress) + Send>;
/// Called for each notable event during an export.
type EventCallback = Box<dyn FnMut(&VideoExportEvent) + Send>;

use std::io::Write;
use std::process::{Command, Stdio};

use crate::captions::{CaptionFont, CaptionFormat, CaptionPainter, CaptionStyle, CaptionTrack};

/// How a caption track reaches the exported video.
///
/// Burned-in captions are pixels — they survive any player, any codec, and any
/// re-encode, but the viewer cannot switch them off. Soft captions (sidecar or
/// embedded) stay selectable and searchable, which is what accessibility
/// guidelines ask for, but need a player that reads them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CaptionMode {
    /// Composite the text into the frames ("open captions"). Works with every
    /// codec, including GIF and APNG.
    #[default]
    Burn,
    /// Write a separate `.srt` / `.vtt` file next to the video.
    Sidecar,
    /// Mux a soft subtitle track into the container — MP4 gets `mov_text`
    /// (`tx3g`), WebM gets `webvtt`. Not available for GIF or APNG, which have
    /// no subtitle track.
    Embed,
    /// Burn the text in *and* write a sidecar, so the captions are both always
    /// visible and machine-readable.
    BurnAndSidecar,
}

impl CaptionMode {
    /// Whether this mode paints the text into the frames.
    pub fn burns(self) -> bool {
        matches!(self, CaptionMode::Burn | CaptionMode::BurnAndSidecar)
    }

    /// Whether this mode writes a separate subtitle file.
    pub fn writes_sidecar(self) -> bool {
        matches!(self, CaptionMode::Sidecar | CaptionMode::BurnAndSidecar)
    }

    /// Whether this mode muxes a subtitle track into the container.
    pub fn embeds(self) -> bool {
        matches!(self, CaptionMode::Embed)
    }
}

/// A caption track plus how to render and deliver it.
///
/// Build one with [`VideoOptions::captions`] and refine it with the
/// `caption_*` builders.
#[derive(Clone, Debug)]
pub struct CaptionExport {
    /// The cues to show.
    pub track: CaptionTrack,
    /// How the cues reach the output.
    pub mode: CaptionMode,
    /// Look of burned-in text. `None` uses [`CaptionStyle::default`] rescaled
    /// to the export height, which keeps captions proportionate at any
    /// resolution.
    pub style: Option<CaptionStyle>,
    /// TrueType font bytes for burned-in text. `None` uses the built-in
    /// bitmap face (see [`CaptionFont::builtin`]).
    pub font: Option<Vec<u8>>,
    /// Sidecar file format.
    pub format: CaptionFormat,
    /// Sidecar path. `None` derives it from the video path — `out.mp4` becomes
    /// `out.srt`.
    pub sidecar_path: Option<String>,
}

impl CaptionExport {
    /// Burn `track` into the frames using the default style.
    pub fn new(track: CaptionTrack) -> Self {
        Self {
            track,
            mode: CaptionMode::default(),
            style: None,
            font: None,
            format: CaptionFormat::Srt,
            sidecar_path: None,
        }
    }

    /// Where the sidecar goes for a given video `output` path.
    pub fn sidecar_for(&self, output: &str) -> String {
        self.sidecar_path
            .clone()
            .unwrap_or_else(|| replace_extension(output, self.format.extension()))
    }
}

/// `"a/b.mp4"` + `"srt"` → `"a/b.srt"`. Paths with no extension gain one.
fn replace_extension(path: &str, extension: &str) -> String {
    // Only treat a dot in the last path segment as an extension separator.
    let start = path.rfind(['/', '\\']).map(|i| i + 1).unwrap_or(0);
    match path[start..].rfind('.') {
        Some(dot) if dot > 0 => format!("{}.{extension}", &path[..start + dot]),
        _ => format!("{path}.{extension}"),
    }
}

/// Output video codec / container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoCodec {
    /// H.264 (`libx264`) → `.mp4`. Widely compatible.
    H264,
    /// H.265 / HEVC (`libx265`) → `.mp4` (`hvc1` tag).
    Hevc,
    /// Hardware HEVC on Apple platforms (`hevc_videotoolbox`) — fast.
    HevcVideoToolbox,
    /// VP9 (`libvpx-vp9`) → `.webm`. Set `transparent` for an alpha channel.
    Vp9,
    /// Animated GIF. With `native-codec`, encoded in-process; otherwise ffmpeg.
    Gif,
    /// Animated PNG. With `native-codec`, encoded in-process; otherwise ffmpeg.
    Apng,
}

impl VideoCodec {
    fn is_crf_based(self) -> bool {
        matches!(self, VideoCodec::H264 | VideoCodec::Hevc | VideoCodec::Vp9)
    }

    /// Short label for progress messages (`"gif"`, `"h264"`, …).
    pub fn label(self) -> &'static str {
        match self {
            VideoCodec::H264 => "h264",
            VideoCodec::Hevc => "hevc",
            VideoCodec::HevcVideoToolbox => "hevc-vt",
            VideoCodec::Vp9 => "webm",
            VideoCodec::Gif => "gif",
            VideoCodec::Apng => "apng",
        }
    }

    /// The ffmpeg subtitle encoder this container takes, or `None` when it has
    /// no subtitle track at all (GIF, APNG).
    ///
    /// ```
    /// use threers::VideoCodec;
    /// assert_eq!(VideoCodec::H264.subtitle_encoder(), Some("mov_text"));
    /// assert_eq!(VideoCodec::Vp9.subtitle_encoder(), Some("webvtt"));
    /// assert_eq!(VideoCodec::Gif.subtitle_encoder(), None);
    /// ```
    pub fn subtitle_encoder(self) -> Option<&'static str> {
        match self {
            // MP4 carries 3GPP timed text; ffmpeg calls that `mov_text`.
            VideoCodec::H264 | VideoCodec::Hevc | VideoCodec::HevcVideoToolbox => Some("mov_text"),
            VideoCodec::Vp9 => Some("webvtt"),
            VideoCodec::Gif | VideoCodec::Apng => None,
        }
    }
}

/// Encoding quality — a target bitrate or a constant-rate-factor.
#[derive(Clone, Debug)]
pub enum VideoQuality {
    /// Codec default.
    Default,
    /// Target bitrate string, e.g. `"12M"`.
    Bitrate(String),
    /// Constant rate factor (lower = higher quality; ~18–28 typical). Ignored by
    /// hardware/GIF codecs, which fall back to bitrate/default.
    Crf(u32),
}

/// Options for [`export_video`].
#[derive(Clone, Debug)]
pub struct VideoOptions {
    /// Output file path (extension should match the codec/container).
    pub output: String,
    /// Frames per second.
    pub fps: u32,
    /// Codec / container.
    pub codec: VideoCodec,
    /// Quality target.
    pub quality: VideoQuality,
    /// Preserve the alpha channel where the codec supports it (VP9 → yuva420p,
    /// GIF → transparent index, APNG → RGBA).
    pub transparent: bool,
    /// Extra raw ffmpeg args appended before the output path (escape hatch).
    pub extra_args: Vec<String>,
    /// GIF palette size when using the native encoder (`2..=256`, default 256).
    pub gif_colors: u16,
    /// Subtitles / captions to deliver with the video. See
    /// [`VideoOptions::captions`].
    pub captions: Option<CaptionExport>,
    /// Distance between keyframes, in frames. `None` leaves the codec default
    /// (x264/x265 use 250; VideoToolbox is much shorter). See
    /// [`VideoOptions::keyframe_interval`] — this is the single biggest lever on
    /// file size for rendered animation.
    pub keyframe_interval: Option<u32>,
    /// Software-encoder speed/efficiency preset (`"medium"`, `"slower"`, …).
    /// `None` uses the encoder default. Ignored by the hardware and native
    /// codecs.
    pub preset: Option<String>,
}

impl VideoOptions {
    /// New options for `output` with sensible defaults (30 fps, H.264).
    pub fn new(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            fps: 30,
            codec: VideoCodec::H264,
            quality: VideoQuality::Default,
            transparent: false,
            extra_args: Vec::new(),
            gif_colors: 256,
            captions: None,
            keyframe_interval: None,
            preset: None,
        }
    }
    /// Frames per second (minimum 1).
    pub fn fps(mut self, fps: u32) -> Self {
        self.fps = fps.max(1);
        self
    }
    /// Output codec / container.
    pub fn codec(mut self, codec: VideoCodec) -> Self {
        self.codec = codec;
        self
    }
    /// Target bitrate string for ffmpeg (e.g. `"12M"`).
    pub fn bitrate(mut self, bitrate: impl Into<String>) -> Self {
        self.quality = VideoQuality::Bitrate(bitrate.into());
        self
    }
    /// Constant rate factor (lower = higher quality).
    pub fn crf(mut self, crf: u32) -> Self {
        self.quality = VideoQuality::Crf(crf);
        self
    }
    /// Request an alpha channel where the codec supports it.
    pub fn transparent(mut self, transparent: bool) -> Self {
        self.transparent = transparent;
        self
    }
    /// Extra ffmpeg CLI args inserted before the output path.
    pub fn extra_args(mut self, args: Vec<String>) -> Self {
        self.extra_args = args;
        self
    }
    /// Native GIF palette size (`2..=256`).
    pub fn gif_colors(mut self, n: u16) -> Self {
        self.gif_colors = n.clamp(2, 256);
        self
    }

    /// Distance between keyframes, in frames (minimum 1; `1` is all-intra).
    ///
    /// Rendered animation is highly redundant frame to frame, so the encoder
    /// only pays for a keyframe when it cannot predict — forcing them often is
    /// what makes a high-fps export balloon. A keyframe every few seconds
    /// (`fps * 5` or more) is plenty unless the file is being seeked hard or
    /// segmented for streaming.
    ///
    /// ```
    /// use threers::VideoOptions;
    /// let opts = VideoOptions::new("out.mp4").fps(60).keyframe_interval(300);
    /// assert_eq!(opts.keyframe_interval, Some(300));
    /// ```
    pub fn keyframe_interval(mut self, frames: u32) -> Self {
        self.keyframe_interval = Some(frames.max(1));
        self
    }

    /// Software-encoder preset (`"veryfast"` … `"veryslow"`). Slower presets
    /// spend more CPU searching and produce a smaller file at the same quality.
    pub fn preset(mut self, preset: impl Into<String>) -> Self {
        self.preset = Some(preset.into());
        self
    }

    /// Attach a subtitle / caption track, burned into the frames by default.
    ///
    /// ```
    /// use threers::{CaptionTrack, VideoOptions};
    /// let track = CaptionTrack::parse_srt(
    ///     "1\n00:00:00,000 --> 00:00:02,000\nHello\n",
    /// ).unwrap();
    /// let opts = VideoOptions::new("out.mp4").captions(track);
    /// assert!(opts.captions.is_some());
    /// ```
    ///
    /// Follow with [`caption_mode`](Self::caption_mode) to ship the cues as a
    /// sidecar or an embedded soft-subtitle track instead.
    pub fn captions(mut self, track: CaptionTrack) -> Self {
        self.captions = Some(CaptionExport::new(track));
        self
    }

    /// How the captions reach the output. No-op without a caption track.
    pub fn caption_mode(mut self, mode: CaptionMode) -> Self {
        if let Some(c) = self.captions.as_mut() {
            c.mode = mode;
        }
        self
    }

    /// Look of burned-in captions. Set explicitly, the style is used verbatim —
    /// use [`CaptionStyle::for_height`] if you want it to track the export size.
    pub fn caption_style(mut self, style: CaptionStyle) -> Self {
        if let Some(c) = self.captions.as_mut() {
            c.style = Some(style);
        }
        self
    }

    /// TrueType font for burned-in captions. Without this the built-in bitmap
    /// face is used.
    pub fn caption_font(mut self, ttf: impl Into<Vec<u8>>) -> Self {
        if let Some(c) = self.captions.as_mut() {
            c.font = Some(ttf.into());
        }
        self
    }

    /// Sidecar / embedded subtitle format (default [`CaptionFormat::Srt`]).
    pub fn caption_format(mut self, format: CaptionFormat) -> Self {
        if let Some(c) = self.captions.as_mut() {
            c.format = format;
        }
        self
    }

    /// Explicit sidecar path, instead of deriving it from the video path.
    pub fn caption_sidecar_path(mut self, path: impl Into<String>) -> Self {
        if let Some(c) = self.captions.as_mut() {
            c.sidecar_path = Some(path.into());
        }
        self
    }
}

/// Progress phase — mirrors the JS `VideoExportPhase`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoExportPhase {
    Capture,
    Encode,
    Done,
}

impl VideoExportPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Capture => "capture",
            Self::Encode => "encode",
            Self::Done => "done",
        }
    }
}

/// Progress tick — mirrors the JS `VideoExportProgress`.
#[derive(Clone, Debug)]
pub struct VideoExportProgress {
    pub phase: VideoExportPhase,
    /// 1-based capture index, or encode cursor.
    pub frame: u32,
    pub frames: u32,
    /// Overall 0..=1 estimate.
    pub ratio: f32,
    pub codec: VideoCodec,
    /// Ready-to-display status line from [`format_video_progress`].
    pub message: String,
}

/// Lifecycle / progress events — mirrors the JS `VideoExportEvent` names.
#[derive(Clone, Debug)]
pub enum VideoExportEvent {
    Start {
        phase: VideoExportPhase,
        frames: u32,
        codec: VideoCodec,
    },
    Progress(VideoExportProgress),
    Capture(VideoExportProgress),
    Encode(VideoExportProgress),
    Complete {
        output: String,
        frames: u32,
        codec: VideoCodec,
    },
    Error {
        message: String,
    },
}

/// Ready-to-display progress line (e.g. `"Rendering 12/30 (40%)"`).
pub fn format_video_progress(info: &VideoExportProgress) -> String {
    let pct = (info.ratio.clamp(0.0, 1.0) * 100.0).round() as i32;
    match info.phase {
        VideoExportPhase::Capture => {
            format!("Rendering {}/{} ({pct}%)", info.frame, info.frames)
        }
        VideoExportPhase::Encode => {
            format!(
                "Encoding {}… ({pct}%)",
                info.codec.label().to_ascii_uppercase()
            )
        }
        VideoExportPhase::Done => format!("Done ({} frames)", info.frames),
    }
}

fn make_progress(
    phase: VideoExportPhase,
    frame: u32,
    frames: u32,
    ratio: f32,
    codec: VideoCodec,
) -> VideoExportProgress {
    let mut info = VideoExportProgress {
        phase,
        frame,
        frames,
        ratio: ratio.clamp(0.0, 1.0),
        codec,
        message: String::new(),
    };
    info.message = format_video_progress(&info);
    info
}

/// Collects progress / lifecycle listeners for [`VideoExporter`].
#[derive(Default)]
struct VideoExportTracer {
    on_progress: Option<ProgressCallback>,
    on_event: Option<EventCallback>,
}

impl VideoExportTracer {
    fn emit_event(&mut self, event: VideoExportEvent) {
        if let VideoExportEvent::Progress(ref p)
        | VideoExportEvent::Capture(ref p)
        | VideoExportEvent::Encode(ref p) = event
        {
            if let Some(cb) = self.on_progress.as_mut() {
                cb(p);
            }
        }
        if let Some(cb) = self.on_event.as_mut() {
            cb(&event);
        }
    }

    fn emit_progress(&mut self, info: VideoExportProgress) {
        match info.phase {
            VideoExportPhase::Capture => {
                self.emit_event(VideoExportEvent::Progress(info.clone()));
                self.emit_event(VideoExportEvent::Capture(info));
            }
            VideoExportPhase::Encode => {
                self.emit_event(VideoExportEvent::Progress(info.clone()));
                self.emit_event(VideoExportEvent::Encode(info));
            }
            VideoExportPhase::Done => {
                self.emit_event(VideoExportEvent::Progress(info));
            }
        }
    }
}

/// Fluent video exporter with JS-style progress events.
///
/// ```no_run
/// use threers::{VideoCodec, VideoExporter, VideoExportEvent};
/// # fn render(_: usize) -> Vec<u8> { vec![0; 8 * 8 * 4] }
/// VideoExporter::new("out.gif")
///     .size(8, 8)
///     .frames(4)
///     .fps(10)
///     .codec(VideoCodec::Gif)
///     .on_progress(|p| println!("{}", p.message))
///     .export(render)
///     .unwrap();
/// ```
pub struct VideoExporter {
    options: VideoOptions,
    width: u32,
    height: u32,
    frames: usize,
    tracer: VideoExportTracer,
}

impl VideoExporter {
    /// Start a fluent export targeting `output`.
    pub fn new(output: impl Into<String>) -> Self {
        Self {
            options: VideoOptions::new(output),
            width: 0,
            height: 0,
            frames: 0,
            tracer: VideoExportTracer::default(),
        }
    }

    /// Frame size in pixels.
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Number of frames to capture / encode.
    pub fn frames(mut self, frames: usize) -> Self {
        self.frames = frames;
        self
    }

    pub fn fps(mut self, fps: u32) -> Self {
        self.options = self.options.fps(fps);
        self
    }

    pub fn codec(mut self, codec: VideoCodec) -> Self {
        self.options = self.options.codec(codec);
        self
    }

    pub fn bitrate(mut self, bitrate: impl Into<String>) -> Self {
        self.options = self.options.bitrate(bitrate);
        self
    }

    pub fn crf(mut self, crf: u32) -> Self {
        self.options = self.options.crf(crf);
        self
    }

    pub fn transparent(mut self, transparent: bool) -> Self {
        self.options = self.options.transparent(transparent);
        self
    }

    pub fn gif_colors(mut self, n: u16) -> Self {
        self.options = self.options.gif_colors(n);
        self
    }

    pub fn extra_args(mut self, args: Vec<String>) -> Self {
        self.options = self.options.extra_args(args);
        self
    }

    /// Attach a subtitle / caption track. See [`VideoOptions::captions`].
    pub fn captions(mut self, track: CaptionTrack) -> Self {
        self.options = self.options.captions(track);
        self
    }

    /// How the captions reach the output.
    pub fn caption_mode(mut self, mode: CaptionMode) -> Self {
        self.options = self.options.caption_mode(mode);
        self
    }

    /// Look of burned-in captions.
    pub fn caption_style(mut self, style: CaptionStyle) -> Self {
        self.options = self.options.caption_style(style);
        self
    }

    /// TrueType font for burned-in captions.
    pub fn caption_font(mut self, ttf: impl Into<Vec<u8>>) -> Self {
        self.options = self.options.caption_font(ttf);
        self
    }

    /// Sidecar / embedded subtitle format.
    pub fn caption_format(mut self, format: CaptionFormat) -> Self {
        self.options = self.options.caption_format(format);
        self
    }

    /// Explicit sidecar path.
    pub fn caption_sidecar_path(mut self, path: impl Into<String>) -> Self {
        self.options = self.options.caption_sidecar_path(path);
        self
    }

    /// Replace options wholesale (keeps size / frame count / listeners).
    pub fn options(mut self, options: VideoOptions) -> Self {
        self.options = options;
        self
    }

    /// Progress callback (JS `onProgress` / `progress` event).
    pub fn on_progress(mut self, cb: impl FnMut(&VideoExportProgress) + Send + 'static) -> Self {
        self.tracer.on_progress = Some(Box::new(cb));
        self
    }

    /// Lifecycle + progress callback (JS `addEventListener` / `.on(...)`).
    pub fn on_event(mut self, cb: impl FnMut(&VideoExportEvent) + Send + 'static) -> Self {
        self.tracer.on_event = Some(Box::new(cb));
        self
    }

    /// Capture frames via `frame(index)` and encode to the configured output.
    pub fn export<F>(mut self, frame: F) -> Result<(), VideoError>
    where
        F: FnMut(usize) -> Vec<u8>,
    {
        if self.width == 0 || self.height == 0 {
            return Err(VideoError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "VideoExporter: size(width, height) required",
            )));
        }
        export_video_traced(
            self.width,
            self.height,
            self.frames,
            &self.options,
            frame,
            &mut self.tracer,
        )
    }
}

/// Errors from [`export_video`].
#[derive(Debug)]
pub enum VideoError {
    /// `ffmpeg` could not be launched (is it installed / on `PATH`?).
    Spawn(std::io::Error),
    /// Failed to write a frame to ffmpeg's stdin.
    Write(std::io::Error),
    /// Failed to write the output file (native animation path).
    Io(std::io::Error),
    /// A frame closure returned the wrong number of bytes (`expected`, `got`).
    FrameSize {
        frame: usize,
        expected: usize,
        got: usize,
    },
    /// ffmpeg exited with a non-zero status.
    Ffmpeg(Option<i32>),
    /// `frames` was zero.
    NoFrames,
    /// The caption track could not be applied — an unreadable font, or an
    /// [`CaptionMode::Embed`] request for a container with no subtitle track.
    Captions(String),
}

impl std::fmt::Display for VideoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VideoError::Spawn(e) => {
                write!(f, "failed to launch ffmpeg (installed and on PATH?): {e}")
            }
            VideoError::Write(e) => write!(f, "failed to write a frame to ffmpeg: {e}"),
            VideoError::Io(e) => write!(f, "failed to write output: {e}"),
            VideoError::FrameSize {
                frame,
                expected,
                got,
            } => write!(
                f,
                "frame {frame} returned {got} bytes, expected {expected} (width*height*4)"
            ),
            VideoError::Ffmpeg(code) => write!(f, "ffmpeg exited with status {code:?}"),
            VideoError::NoFrames => write!(f, "frame count was zero"),
            VideoError::Captions(m) => write!(f, "captions could not be applied: {m}"),
        }
    }
}
impl std::error::Error for VideoError {}

/// Encode `frames` frames of `width × height` RGBA into `options.output`.
///
/// `frame` is called with each index `0..frames` and must return exactly
/// `width * height * 4` bytes (tightly-packed RGBA8, top-left origin).
///
/// Prefer [`VideoExporter`] when you want progress events.
pub fn export_video<F>(
    width: u32,
    height: u32,
    frames: usize,
    options: &VideoOptions,
    frame: F,
) -> Result<(), VideoError>
where
    F: FnMut(usize) -> Vec<u8>,
{
    let mut tracer = VideoExportTracer::default();
    export_video_traced(width, height, frames, options, frame, &mut tracer)
}

/// Like [`export_video`], but reports progress through `on_progress`.
pub fn export_video_with_progress<F, P>(
    width: u32,
    height: u32,
    frames: usize,
    options: &VideoOptions,
    frame: F,
    on_progress: P,
) -> Result<(), VideoError>
where
    F: FnMut(usize) -> Vec<u8>,
    P: FnMut(&VideoExportProgress) + Send + 'static,
{
    VideoExporter::new(options.output.clone())
        .size(width, height)
        .frames(frames)
        .options(options.clone())
        .on_progress(on_progress)
        .export(frame)
}

fn export_video_traced<F>(
    width: u32,
    height: u32,
    frames: usize,
    options: &VideoOptions,
    frame: F,
    tracer: &mut VideoExportTracer,
) -> Result<(), VideoError>
where
    F: FnMut(usize) -> Vec<u8>,
{
    if frames == 0 {
        let err = VideoError::NoFrames;
        tracer.emit_event(VideoExportEvent::Error {
            message: err.to_string(),
        });
        return Err(err);
    }
    let expected = (width as usize) * (height as usize) * 4;
    let frames_u = frames as u32;
    let codec = options.codec;

    // Set up captions before anything is captured, so a bad font or an
    // impossible embed request fails immediately instead of after encoding.
    let mut burner = match CaptionBurner::prepare(options, width, height) {
        Ok(b) => b,
        Err(e) => {
            tracer.emit_event(VideoExportEvent::Error {
                message: e.to_string(),
            });
            return Err(e);
        }
    };
    // Every path below sees frames that already have the captions painted in.
    let mut source = frame;
    let mut frame = |i: usize| {
        let mut buf = source(i);
        if let Some(b) = burner.as_mut() {
            b.apply(&mut buf, i);
        }
        buf
    };

    let result = (|| {
        // The soft-subtitle track needs a real file for ffmpeg to read; it is
        // removed once the child exits.
        let embedded = EmbeddedSubtitles::prepare(options)?;

        #[cfg(feature = "native-codec")]
        if matches!(options.codec, VideoCodec::Gif | VideoCodec::Apng) {
            export_native_animation(width, height, frames, options, &mut frame, tracer)?;
            return write_sidecar(options);
        }

        #[cfg(feature = "native-codec")]
        if matches!(options.codec, VideoCodec::H264) {
            export_native_h264(width, height, frames, options, &mut frame, tracer)?;
            return write_sidecar(options);
        }

        tracer.emit_event(VideoExportEvent::Start {
            phase: VideoExportPhase::Capture,
            frames: frames_u,
            codec,
        });

        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-y")
            .args(["-f", "rawvideo"])
            .args(["-pixel_format", "rgba"])
            .args(["-video_size", &format!("{width}x{height}")])
            .args(["-framerate", &options.fps.to_string()])
            .args(["-i", "-"]);
        if let Some(sub) = &embedded {
            sub.append_input(&mut cmd);
        }
        append_codec_args(&mut cmd, options);
        if let Some(sub) = &embedded {
            sub.append_output_args(&mut cmd);
        }
        for a in &options.extra_args {
            cmd.arg(a);
        }
        cmd.arg(&options.output);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());

        let mut child = cmd.spawn().map_err(VideoError::Spawn)?;
        {
            let mut stdin = child.stdin.take().expect("ffmpeg stdin");
            for i in 0..frames {
                let buf = frame(i);
                if buf.len() != expected {
                    let _ = child.kill();
                    return Err(VideoError::FrameSize {
                        frame: i,
                        expected,
                        got: buf.len(),
                    });
                }
                stdin.write_all(&buf).map_err(VideoError::Write)?;
                let done = (i + 1) as u32;
                tracer.emit_progress(make_progress(
                    VideoExportPhase::Capture,
                    done,
                    frames_u,
                    done as f32 / (frames_u as f32 + 1.0),
                    codec,
                ));
            }
        }
        let status = child.wait().map_err(VideoError::Spawn)?;
        if !status.success() {
            return Err(VideoError::Ffmpeg(status.code()));
        }
        write_sidecar(options)?;
        tracer.emit_progress(make_progress(
            VideoExportPhase::Done,
            frames_u,
            frames_u,
            1.0,
            codec,
        ));
        Ok(())
    })();

    match &result {
        Ok(()) => tracer.emit_event(VideoExportEvent::Complete {
            output: options.output.clone(),
            frames: frames_u,
            codec,
        }),
        Err(e) => tracer.emit_event(VideoExportEvent::Error {
            message: e.to_string(),
        }),
    }
    result
}

#[cfg(feature = "native-codec")]
/// Read a bitrate like `"5M"`, `"800k"` or `"2500000"` into bits per second.
///
/// The same spellings ffmpeg takes, because that is what a caller who has used
/// `-b:v` will reach for. `M` and `k` are decimal here, as they are there.
fn parse_bitrate(s: &str) -> Option<u64> {
    let t = s.trim();
    let (digits, scale) = match t.chars().last()? {
        'k' | 'K' => (&t[..t.len() - 1], 1_000f64),
        'm' | 'M' => (&t[..t.len() - 1], 1_000_000f64),
        'g' | 'G' => (&t[..t.len() - 1], 1_000_000_000f64),
        _ => (t, 1.0),
    };
    let v: f64 = digits.trim().parse().ok()?;
    (v > 0.0).then_some((v * scale) as u64)
}

// The only caller is behind this same flag; without it the `crate::codec`
// imports below refer to a module that was configured out, and `--features
// video` alone stops compiling.
#[cfg(feature = "native-codec")]
fn export_native_h264<F>(
    width: u32,
    height: u32,
    frames: usize,
    options: &VideoOptions,
    frame: &mut F,
    tracer: &mut VideoExportTracer,
) -> Result<(), VideoError>
where
    F: FnMut(usize) -> Vec<u8>,
{
    use crate::codec::h264::write_compressed_mp4_quality;
    use crate::codec::rate::Quality;
    use crate::codec::hevc::Yuv420Frame;

    if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(VideoError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "native H.264 export requires even width and height (4:2:0)",
        )));
    }
    if options.transparent {
        return Err(VideoError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "native H.264 export does not support transparency yet",
        )));
    }

    // Quantization for the in-process encoder. H.264's QP and x264's CRF sit on
    // comparable scales, so a caller's CRF carries over directly; a bitrate
    // target has no meaning without rate control, so it falls back to the
    // default and says so.
    let quality = match &options.quality {
        VideoQuality::Crf(crf) => Quality::Qp((*crf as i32).clamp(0, 51)),
        VideoQuality::Default => Quality::Qp(26),
        VideoQuality::Bitrate(b) => match parse_bitrate(b) {
            Some(bps) => Quality::Bitrate(bps),
            None => {
                eprintln!(
                    "threers: could not read {b:?} as a bitrate — expected something like \
                     \"5M\", \"800k\" or a plain number of bits per second. Encoding at QP 26."
                );
                Quality::Qp(26)
            }
        },
    };

    let expected = (width as usize) * (height as usize) * 4;
    let frames_u = frames as u32;
    let codec = options.codec;
    let captions = options.captions.as_ref().and_then(|c| {
        if c.mode.embeds() && !c.track.is_empty() {
            Some(&c.track)
        } else {
            None
        }
    });

    tracer.emit_event(VideoExportEvent::Start {
        phase: VideoExportPhase::Capture,
        frames: frames_u,
        codec,
    });

    tracer.emit_event(VideoExportEvent::Start {
        phase: VideoExportPhase::Encode,
        frames: frames_u,
        codec,
    });

    // Render, convert and encode one frame at a time. Collecting them first
    // would hold `width * height * 1.5` bytes each — 7.5 GB for ten seconds of
    // 4K60, and the encoder only ever looks at one.
    let mut size_error = None;
    let file = std::fs::File::create(&options.output).map_err(VideoError::Io)?;
    let mut sink = std::io::BufWriter::new(file);
    let source = (0..frames).map(|i| {
        let buf = frame(i);
        if buf.len() != expected && size_error.is_none() {
            size_error = Some(VideoError::FrameSize {
                frame: i,
                expected,
                got: buf.len(),
            });
            // Keep the shape valid so the encoder can finish; the error is
            // returned below and the output discarded.
            return Yuv420Frame::new(width, height);
        }
        let done = (i + 1) as u32;
        tracer.emit_progress(make_progress(
            VideoExportPhase::Capture,
            done,
            frames_u,
            done as f32 / (frames_u as f32 + 1.0),
            codec,
        ));
        Yuv420Frame::from_rgba(width, height, &buf)
    });
    // Written as it is assembled, so the finished file is never held in memory
    // alongside the samples it was built from.
    let wrote = write_compressed_mp4_quality(
        &mut sink,
        width,
        height,
        options.fps,
        quality,
        captions,
        source,
        Some(frames_u),
    )
    .and_then(|()| sink.flush());
    // A frame of the wrong size is only discovered mid-stream, by which point
    // part of the file exists. Do not leave that behind for the caller to
    // mistake for output.
    if let Some(e) = size_error {
        drop(sink);
        let _ = std::fs::remove_file(&options.output);
        return Err(e);
    }
    wrote.map_err(VideoError::Io)?;

    tracer.emit_progress(make_progress(
        VideoExportPhase::Done,
        frames_u,
        frames_u,
        1.0,
        codec,
    ));
    Ok(())
}

#[cfg(feature = "native-codec")]
fn export_native_animation<F>(
    width: u32,
    height: u32,
    frames: usize,
    options: &VideoOptions,
    frame: &mut F,
    tracer: &mut VideoExportTracer,
) -> Result<(), VideoError>
where
    F: FnMut(usize) -> Vec<u8>,
{
    use crate::codec::animation::{
        encode_animation_rgba_with_progress, AnimationEncodeError, AnimationEncodeOptions,
        AnimationExportPhase, BrowserCodec,
    };

    let codec = match options.codec {
        VideoCodec::Gif => BrowserCodec::Gif,
        VideoCodec::Apng => BrowserCodec::Apng,
        _ => unreachable!("native animation export only handles GIF and APNG"),
    };
    let opts = AnimationEncodeOptions {
        width,
        height,
        fps: options.fps,
        codec,
        transparent: options.transparent,
        gif_colors: options.gif_colors,
    };
    let expected = (width as usize) * (height as usize) * 4;
    let frames_u = frames as u32;
    let vcodec = options.codec;

    tracer.emit_event(VideoExportEvent::Start {
        phase: VideoExportPhase::Capture,
        frames: frames_u,
        codec: vcodec,
    });

    let mut captured = Vec::with_capacity(frames);
    for i in 0..frames {
        let buf = frame(i);
        if buf.len() != expected {
            return Err(VideoError::FrameSize {
                frame: i,
                expected,
                got: buf.len(),
            });
        }
        captured.push(buf);
        let done = (i + 1) as u32;
        tracer.emit_progress(make_progress(
            VideoExportPhase::Capture,
            done,
            frames_u,
            done as f32 / (frames_u as f32 + 1.0),
            vcodec,
        ));
    }

    tracer.emit_event(VideoExportEvent::Start {
        phase: VideoExportPhase::Encode,
        frames: frames_u,
        codec: vcodec,
    });

    let bytes = encode_animation_rgba_with_progress(&opts, captured, |p| {
        let phase = match p.phase {
            AnimationExportPhase::Encode => VideoExportPhase::Encode,
            AnimationExportPhase::Done => VideoExportPhase::Done,
            AnimationExportPhase::Capture => VideoExportPhase::Capture,
        };
        tracer.emit_progress(make_progress(phase, p.frame, p.frames, p.ratio, vcodec));
    })
    .map_err(|error| match error {
        AnimationEncodeError::Empty => VideoError::NoFrames,
        AnimationEncodeError::FrameSize {
            frame,
            expected,
            got,
        } => VideoError::FrameSize {
            frame,
            expected,
            got,
        },
        AnimationEncodeError::Dimension => VideoError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid native animation dimensions or options",
        )),
        AnimationEncodeError::Io(message) => VideoError::Io(std::io::Error::other(message)),
    })?;
    std::fs::write(&options.output, bytes).map_err(VideoError::Io)?;
    tracer.emit_progress(make_progress(
        VideoExportPhase::Done,
        frames_u,
        frames_u,
        1.0,
        vcodec,
    ));
    Ok(())
}

/// Paints the active cues into each captured frame.
struct CaptionBurner<'a> {
    painter: CaptionPainter,
    track: &'a CaptionTrack,
    fps: u32,
    width: u32,
    height: u32,
}

impl<'a> CaptionBurner<'a> {
    /// Build a burner if `options` asks for burned-in captions.
    fn prepare(
        options: &'a VideoOptions,
        width: u32,
        height: u32,
    ) -> Result<Option<Self>, VideoError> {
        let Some(captions) = options.captions.as_ref() else {
            return Ok(None);
        };
        if captions.mode.embeds() && options.codec.subtitle_encoder().is_none() {
            return Err(VideoError::Captions(format!(
                "{} has no subtitle track — use CaptionMode::Burn or ::Sidecar",
                options.codec.label().to_ascii_uppercase()
            )));
        }
        if !captions.mode.burns() || captions.track.is_empty() {
            return Ok(None);
        }

        let font = match &captions.font {
            Some(bytes) => CaptionFont::from_ttf_bytes(bytes).map_err(|e| {
                VideoError::Captions(format!("caption font could not be parsed: {e:?}"))
            })?,
            None => CaptionFont::builtin(),
        };
        // An unset style tracks the export height so captions stay
        // proportionate whether you render 480p or 4K.
        let style = captions
            .style
            .clone()
            .unwrap_or_else(|| CaptionStyle::default().for_height(height));

        Ok(Some(Self {
            painter: CaptionPainter::with_font(font).style(style),
            track: &captions.track,
            fps: options.fps.max(1),
            width,
            height,
        }))
    }

    /// Paint frame `index`, whose display time is `index / fps`.
    fn apply(&mut self, frame: &mut [u8], index: usize) {
        let time = index as f64 / self.fps as f64;
        self.painter
            .burn_in(frame, self.width, self.height, self.track, time);
    }
}

/// Write the sidecar subtitle file when the mode asks for one.
fn write_sidecar(options: &VideoOptions) -> Result<(), VideoError> {
    let Some(captions) = options.captions.as_ref() else {
        return Ok(());
    };
    if !captions.mode.writes_sidecar() || captions.track.is_empty() {
        return Ok(());
    }
    let path = captions.sidecar_for(&options.output);
    std::fs::write(path, captions.track.to_format(captions.format)).map_err(VideoError::Io)
}

/// A temporary subtitle file handed to ffmpeg as a second input, deleted when
/// the export finishes.
struct EmbeddedSubtitles {
    path: std::path::PathBuf,
    encoder: &'static str,
    language: String,
    label: String,
}

impl EmbeddedSubtitles {
    fn prepare(options: &VideoOptions) -> Result<Option<Self>, VideoError> {
        let Some(captions) = options.captions.as_ref() else {
            return Ok(None);
        };
        if !captions.mode.embeds() || captions.track.is_empty() {
            return Ok(None);
        }
        let encoder = options.codec.subtitle_encoder().ok_or_else(|| {
            VideoError::Captions(format!(
                "{} has no subtitle track — use CaptionMode::Burn or ::Sidecar",
                options.codec.label().to_ascii_uppercase()
            ))
        })?;
        // WebM only takes WebVTT; MP4 timed text is fed from SubRip.
        let format = match encoder {
            "webvtt" => CaptionFormat::Vtt,
            _ => CaptionFormat::Srt,
        };
        let path = std::env::temp_dir().join(format!(
            "threers-captions-{}.{}",
            std::process::id(),
            format.extension()
        ));
        std::fs::write(&path, captions.track.to_format(format)).map_err(VideoError::Io)?;
        Ok(Some(Self {
            path,
            encoder,
            language: String::from_utf8_lossy(&crate::captions::iso639_2(&captions.track.language))
                .into_owned(),
            label: captions.track.label.clone(),
        }))
    }

    /// The subtitle file as ffmpeg input #1.
    fn append_input(&self, cmd: &mut Command) {
        cmd.arg("-i").arg(&self.path);
    }

    /// Map both streams and tag the subtitle track.
    fn append_output_args(&self, cmd: &mut Command) {
        cmd.args(["-map", "0:v:0", "-map", "1:s:0"]);
        cmd.args(["-c:s", self.encoder]);
        cmd.args(["-metadata:s:s:0", &format!("language={}", self.language)]);
        if !self.label.is_empty() {
            cmd.args(["-metadata:s:s:0", &format!("title={}", self.label)]);
        }
    }
}

impl Drop for EmbeddedSubtitles {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn append_codec_args(cmd: &mut Command, opts: &VideoOptions) {
    // x265 takes keyint through its own parameter string rather than `-g`.
    let x265_params = opts
        .keyframe_interval
        .map(|g| format!("keyint={g}:min-keyint={g}"));
    match opts.codec {
        VideoCodec::H264 => {
            cmd.args(["-c:v", "libx264", "-pix_fmt", "yuv420p"]);
        }
        VideoCodec::Hevc => {
            cmd.args(["-c:v", "libx265", "-pix_fmt", "yuv420p", "-tag:v", "hvc1"]);
            if let Some(p) = &x265_params {
                cmd.args(["-x265-params", p]);
            }
        }
        VideoCodec::HevcVideoToolbox => {
            cmd.args([
                "-c:v",
                "hevc_videotoolbox",
                "-tag:v",
                "hvc1",
                "-pix_fmt",
                "yuv420p",
            ]);
        }
        VideoCodec::Vp9 => {
            let pix = if opts.transparent {
                "yuva420p"
            } else {
                "yuv420p"
            };
            cmd.args(["-c:v", "libvpx-vp9", "-pix_fmt", pix]);
        }
        VideoCodec::Gif => {
            cmd.args([
                "-vf",
                "split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse",
            ]);
        }
        VideoCodec::Apng => {
            cmd.args(["-plays", "0", "-f", "apng"]);
        }
    }
    // Keyframe distance. x265 was handled above via `-x265-params`; everything
    // else takes `-g`. Left unset, x264/x265 use 250 frames while VideoToolbox
    // picks something far shorter, so an unset value is not a neutral default.
    if let Some(g) = opts.keyframe_interval {
        if !matches!(opts.codec, VideoCodec::Hevc | VideoCodec::Gif | VideoCodec::Apng) {
            cmd.args(["-g", &g.to_string()]);
        }
    }
    if let Some(preset) = &opts.preset {
        if matches!(opts.codec, VideoCodec::H264 | VideoCodec::Hevc | VideoCodec::Vp9) {
            cmd.args(["-preset", preset]);
        }
    }
    match &opts.quality {
        VideoQuality::Default => {}
        VideoQuality::Bitrate(b) => {
            cmd.args(["-b:v", b]);
        }
        VideoQuality::Crf(crf) => {
            if opts.codec.is_crf_based() {
                cmd.args(["-crf", &crf.to_string()]);
            }
        }
    }
}

#[cfg(test)]
mod caption_tests {
    use super::*;

    #[test]
    fn sidecar_paths_swap_the_extension() {
        assert_eq!(replace_extension("out.mp4", "srt"), "out.srt");
        assert_eq!(replace_extension("a/b/clip.webm", "vtt"), "a/b/clip.vtt");
        // No extension → gain one rather than clobbering the name.
        assert_eq!(replace_extension("out/movie", "srt"), "out/movie.srt");
        // A dot in a directory name is not an extension.
        assert_eq!(replace_extension("v1.2/clip", "srt"), "v1.2/clip.srt");
        // A dotfile keeps its leading dot.
        assert_eq!(replace_extension(".hidden", "srt"), ".hidden.srt");
        assert_eq!(
            replace_extension("C:\\vids\\a.mov", "vtt"),
            "C:\\vids\\a.vtt"
        );
    }

    #[test]
    fn sidecar_for_honors_format_and_explicit_path() {
        let mut export = CaptionExport::new(CaptionTrack::new());
        assert_eq!(export.sidecar_for("out.mp4"), "out.srt");
        export.format = CaptionFormat::Vtt;
        assert_eq!(export.sidecar_for("out.mp4"), "out.vtt");
        export.sidecar_path = Some("elsewhere/subs.vtt".into());
        assert_eq!(export.sidecar_for("out.mp4"), "elsewhere/subs.vtt");
    }

    #[test]
    fn caption_modes_describe_what_they_do() {
        assert!(CaptionMode::Burn.burns());
        assert!(!CaptionMode::Burn.writes_sidecar());
        assert!(!CaptionMode::Burn.embeds());
        assert!(CaptionMode::Sidecar.writes_sidecar());
        assert!(!CaptionMode::Sidecar.burns());
        assert!(CaptionMode::Embed.embeds());
        assert!(CaptionMode::BurnAndSidecar.burns());
        assert!(CaptionMode::BurnAndSidecar.writes_sidecar());
        // Burn is the default: captions are visible without player support.
        assert_eq!(CaptionMode::default(), CaptionMode::Burn);
    }

    #[test]
    fn caption_builders_are_no_ops_without_a_track() {
        // Setting caption options before `captions()` must not panic.
        let opts = VideoOptions::new("out.mp4")
            .caption_mode(CaptionMode::Embed)
            .caption_format(CaptionFormat::Vtt);
        assert!(opts.captions.is_none());
    }

    #[test]
    fn unset_caption_style_scales_itself_to_the_export_height() {
        let opts = VideoOptions::new("out.mp4").captions(CaptionTrack::new().cue(0.0, 1.0, "x"));
        let burner = CaptionBurner::prepare(&opts, 1280, 540).unwrap().unwrap();
        // Default style is authored for 1080p; 540 is half that.
        assert!((burner.painter.style.font_size - 17.0).abs() < 1e-4);

        // An explicit style is used verbatim.
        let opts = opts.caption_style(CaptionStyle::default().font_size(40.0));
        let burner = CaptionBurner::prepare(&opts, 1280, 540).unwrap().unwrap();
        assert_eq!(burner.painter.style.font_size, 40.0);
    }
}

#[cfg(all(test, feature = "native-codec"))]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut out = vec![0u8; (w * h * 4) as usize];
        for px in out.chunks_exact_mut(4) {
            px.copy_from_slice(&[r, g, b, 255]);
        }
        out
    }

    #[test]
    fn video_exporter_emits_progress_events() {
        let dir = std::env::temp_dir().join("threers-video-events");
        let _ = std::fs::create_dir_all(&dir);
        let out = dir.join("cube.gif");
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let ev2 = events.clone();
        let prog = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let p2 = prog.clone();

        VideoExporter::new(out.to_string_lossy().into_owned())
            .size(8, 8)
            .frames(4)
            .fps(10)
            .codec(VideoCodec::Gif)
            .gif_colors(32)
            .on_progress(move |p| p2.lock().unwrap().push(p.message.clone()))
            .on_event(move |e| {
                let label = match e {
                    VideoExportEvent::Start { phase, .. } => format!("start:{}", phase.as_str()),
                    VideoExportEvent::Progress(p) => format!("progress:{}", p.phase.as_str()),
                    VideoExportEvent::Capture(_) => "capture".into(),
                    VideoExportEvent::Encode(_) => "encode".into(),
                    VideoExportEvent::Complete { .. } => "complete".into(),
                    VideoExportEvent::Error { message } => format!("error:{message}"),
                };
                ev2.lock().unwrap().push(label);
            })
            .export(|i| {
                let c = ((i * 40) % 255) as u8;
                solid(8, 8, c, 80, 200)
            })
            .expect("export");

        let events = events.lock().unwrap().clone();
        assert!(events.iter().any(|e| e.starts_with("start:")), "{events:?}");
        assert!(events.iter().any(|e| e == "capture"), "{events:?}");
        assert!(events.iter().any(|e| e == "encode"), "{events:?}");
        assert!(events.iter().any(|e| e == "complete"), "{events:?}");
        let prog = prog.lock().unwrap().clone();
        assert!(!prog.is_empty(), "expected progress messages");
        assert!(out.is_file());
        let bytes = std::fs::read(&out).unwrap();
        assert_eq!(&bytes[..3], b"GIF");
    }
}
