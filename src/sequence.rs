//! Rendering a sequence: still frames, a video, or both from one pass.
//!
//! [`export_video`](crate::video::export_video) already turns a frame closure
//! into an encoded file, and [`encode_png`] already writes a still. What is
//! missing between them is the part every animation needs anyway — render once,
//! keep the frames *and* the video, know how long it has left, and do not
//! re-render the whole shot because the encoder was misconfigured.
//!
//! ```no_run
//! use threers::{SequenceOptions, render_sequence};
//!
//! let opts = SequenceOptions::new(120)
//!     .fps(30)
//!     .frames_to("out/shot")
//!     .video_to("out/shot.mp4");
//! let report = render_sequence(1920, 1080, &opts, |frame| {
//!     // ... render and return RGBA8 ...
//!     vec![0u8; 1920 * 1080 * 4]
//! })?;
//! println!("{:.1}s, {:.2} s/frame", report.elapsed_secs, report.seconds_per_frame());
//! # Ok::<(), threers::SequenceError>(())
//! ```
//!
//! # Keeping the frames
//!
//! Stills are worth writing even when a video is the deliverable. A path-traced
//! shot can cost minutes a frame, and every reason to re-encode — wrong codec,
//! wrong frame rate, wrong colour range, a denoiser comparison — is a reason
//! that does not need the renderer to run again. PNG is lossless, so a
//! re-encode from frames is indistinguishable from one straight out of the
//! renderer.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::utils::png::encode_png;

#[cfg(all(feature = "video", not(target_arch = "wasm32")))]
use crate::video::{VideoError, VideoOptions};

/// What to produce, and where.
#[derive(Debug, Clone)]
pub struct SequenceOptions {
    /// Number of frames to render.
    pub frames: usize,
    /// Frame rate for the video. Ignored when no video is written.
    pub fps: u32,
    /// Directory for numbered PNG stills. `None` writes no stills.
    pub frame_dir: Option<PathBuf>,
    /// Stem for still filenames — `{stem}_{index:05}.png`.
    pub frame_stem: String,
    /// Where the video goes. `None` writes no video.
    pub video_path: Option<String>,
    /// Constant rate factor handed to the encoder. Lower is better quality.
    pub crf: u32,
    /// Print a progress line as frames complete.
    pub progress: bool,
    /// Suppress the encoder's own chatter, leaving only its errors.
    pub quiet_encoder: bool,
}

impl SequenceOptions {
    /// `frames` frames, 30 fps, no output configured yet.
    pub fn new(frames: usize) -> Self {
        Self {
            frames,
            fps: 30,
            frame_dir: None,
            frame_stem: "frame".into(),
            video_path: None,
            // 18 rather than the encoder's usual 23. A render has flat gradients
            // — sky, soft shadows, defocus — and those are exactly what a
            // mid-range CRF bands. The file is larger; the render cost far more.
            crf: 18,
            progress: true,
            // ffmpeg reports its build configuration, every stream it opens and
            // a per-frame status line, which buries a progress line and any
            // real error along with it. Errors still come through.
            quiet_encoder: true,
        }
    }

    pub fn fps(mut self, fps: u32) -> Self {
        self.fps = fps.max(1);
        self
    }

    /// Write numbered PNG stills into `dir`, creating it if needed.
    pub fn frames_to(mut self, dir: impl AsRef<Path>) -> Self {
        self.frame_dir = Some(dir.as_ref().to_path_buf());
        self
    }

    /// Stem for still filenames.
    pub fn frame_stem(mut self, stem: impl Into<String>) -> Self {
        self.frame_stem = stem.into();
        self
    }

    /// Encode a video at this path. The extension picks the container.
    pub fn video_to(mut self, path: impl Into<String>) -> Self {
        self.video_path = Some(path.into());
        self
    }

    /// Encoder quality; lower is better. 0 is lossless for H.264.
    pub fn crf(mut self, crf: u32) -> Self {
        self.crf = crf;
        self
    }

    pub fn progress(mut self, on: bool) -> Self {
        self.progress = on;
        self
    }

    /// Let the encoder print everything it normally would.
    pub fn verbose_encoder(mut self) -> Self {
        self.quiet_encoder = false;
        self
    }

    fn frame_path(&self, dir: &Path, index: usize) -> PathBuf {
        dir.join(format!("{}_{index:05}.png", self.frame_stem))
    }
}

/// What a finished sequence produced.
#[derive(Debug, Clone, Default)]
pub struct SequenceReport {
    pub frames: usize,
    pub width: u32,
    pub height: u32,
    pub elapsed_secs: f64,
    /// Stills written, in order.
    pub frame_paths: Vec<PathBuf>,
    /// The encoded video, if one was asked for and produced.
    pub video_path: Option<String>,
}

impl SequenceReport {
    pub fn seconds_per_frame(&self) -> f64 {
        if self.frames == 0 {
            0.0
        } else {
            self.elapsed_secs / self.frames as f64
        }
    }
}

/// Why a sequence stopped.
#[derive(Debug)]
pub enum SequenceError {
    /// No frames were asked for.
    NoFrames,
    /// A frame closure returned the wrong number of bytes.
    FrameSize {
        frame: usize,
        got: usize,
        expected: usize,
    },
    Io(std::io::Error),
    /// The encoder failed. The stills, if any, are still on disk.
    Video(String),
    /// A video was asked for from a build without the `video` feature.
    VideoUnavailable,
}

impl std::fmt::Display for SequenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SequenceError::NoFrames => write!(f, "sequence has no frames"),
            SequenceError::FrameSize {
                frame,
                got,
                expected,
            } => write!(
                f,
                "frame {frame} returned {got} bytes, expected {expected} (width*height*4)"
            ),
            SequenceError::Io(e) => write!(f, "{e}"),
            SequenceError::Video(m) => write!(f, "video encode failed: {m}"),
            SequenceError::VideoUnavailable => {
                write!(
                    f,
                    "a video was requested but this build lacks the `video` feature"
                )
            }
        }
    }
}

impl std::error::Error for SequenceError {}

impl From<std::io::Error> for SequenceError {
    fn from(e: std::io::Error) -> Self {
        SequenceError::Io(e)
    }
}

#[cfg(all(feature = "video", not(target_arch = "wasm32")))]
impl From<VideoError> for SequenceError {
    fn from(e: VideoError) -> Self {
        SequenceError::Video(e.to_string())
    }
}

/// Render `options.frames` frames, writing stills and/or a video.
///
/// `frame` is called once per index in order and must return exactly
/// `width * height * 4` bytes of tightly-packed RGBA8, top-left origin.
///
/// The closure runs once per frame however many outputs are configured — the
/// expensive half happens once and both sinks are fed from the same pixels.
pub fn render_sequence<F>(
    width: u32,
    height: u32,
    options: &SequenceOptions,
    mut frame: F,
) -> Result<SequenceReport, SequenceError>
where
    F: FnMut(usize) -> Vec<u8>,
{
    if options.frames == 0 {
        return Err(SequenceError::NoFrames);
    }
    // BEFORE the availability check, not after.
    //
    // The encoder opens its own output and fails with a broken pipe if the
    // directory is not there — after every frame has been rendered. Creating it
    // up front costs nothing when it is already right. Doing it below the
    // `VideoUnavailable` return meant the one build that cannot encode is also
    // the one that leaves the caller's output directory missing, so a caller
    // who handles the missing encoder and writes frames anyway still fails.
    if let Some(path) = &options.video_path {
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
    }
    if options.video_path.is_some() && !video_available() {
        return Err(SequenceError::VideoUnavailable);
    }
    let expected = width as usize * height as usize * 4;
    if let Some(dir) = &options.frame_dir {
        std::fs::create_dir_all(dir)?;
    }

    let started = Instant::now();
    let mut report = SequenceReport {
        frames: options.frames,
        width,
        height,
        ..Default::default()
    };

    // Errors from inside the encoder's callback cannot travel out through it,
    // so they are parked here and re-raised once it returns.
    let mut failure: Option<SequenceError> = None;
    let mut paths: Vec<PathBuf> = Vec::with_capacity(options.frames);

    {
        let mut produce = |index: usize| -> Vec<u8> {
            if failure.is_some() {
                return vec![0u8; expected];
            }
            let rgba = frame(index);
            if rgba.len() != expected {
                failure = Some(SequenceError::FrameSize {
                    frame: index,
                    got: rgba.len(),
                    expected,
                });
                return vec![0u8; expected];
            }
            if let Some(dir) = &options.frame_dir {
                let path = options.frame_path(dir, index);
                if let Err(e) = std::fs::write(&path, encode_png(width, height, &rgba)) {
                    failure = Some(SequenceError::Io(e));
                    return vec![0u8; expected];
                }
                paths.push(path);
            }
            if options.progress {
                report_progress(index, options.frames, started);
            }
            rgba
        };

        match &options.video_path {
            Some(path) => encode(width, height, options, path, &mut produce)?,
            None => {
                for i in 0..options.frames {
                    let _ = produce(i);
                }
            }
        }
    }

    if let Some(e) = failure {
        return Err(e);
    }
    report.frame_paths = paths;
    report.video_path = options.video_path.clone();
    report.elapsed_secs = started.elapsed().as_secs_f64();
    if options.progress {
        println!(
            "\r{} frames in {:.1}s ({:.2}s/frame)          ",
            report.frames,
            report.elapsed_secs,
            report.seconds_per_frame()
        );
    }
    Ok(report)
}

/// Whether this build can encode video at all.
pub fn video_available() -> bool {
    cfg!(all(feature = "video", not(target_arch = "wasm32")))
}

#[cfg(all(feature = "video", not(target_arch = "wasm32")))]
fn encode<F>(
    width: u32,
    height: u32,
    options: &SequenceOptions,
    path: &str,
    produce: &mut F,
) -> Result<(), SequenceError>
where
    F: FnMut(usize) -> Vec<u8>,
{
    let mut video = VideoOptions::new(path).fps(options.fps).crf(options.crf);
    if options.quiet_encoder {
        video = video.extra_args(vec!["-loglevel".into(), "error".into()]);
    }
    crate::video::export_video(width, height, options.frames, &video, produce)?;
    Ok(())
}

#[cfg(not(all(feature = "video", not(target_arch = "wasm32"))))]
fn encode<F>(
    _width: u32,
    _height: u32,
    _options: &SequenceOptions,
    _path: &str,
    _produce: &mut F,
) -> Result<(), SequenceError>
where
    F: FnMut(usize) -> Vec<u8>,
{
    Err(SequenceError::VideoUnavailable)
}

/// One rewritten line: what is done, and what is left.
fn report_progress(index: usize, frames: usize, started: Instant) {
    use std::io::Write;
    let done = index + 1;
    let elapsed = started.elapsed().as_secs_f64();
    // `done` frames have finished, so the estimate is over frames actually
    // timed rather than assuming the current one is free.
    let per_frame = elapsed / done as f64;
    let remaining = per_frame * (frames - done) as f64;
    print!(
        "\r  frame {done}/{frames}  {:.2}s/frame  {} left     ",
        per_frame,
        clock(remaining)
    );
    let _ = std::io::stdout().flush();
}

fn clock(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "?".into();
    }
    let total = seconds.round() as u64;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}h{m:02}m")
    } else if m > 0 {
        format!("{m}m{s:02}s")
    } else {
        format!("{s}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames_of(w: u32, h: u32, value: u8) -> Vec<u8> {
        vec![value; w as usize * h as usize * 4]
    }

    #[test]
    fn a_sequence_with_no_frames_is_an_error() {
        let opts = SequenceOptions::new(0);
        let err = render_sequence(4, 4, &opts, |_| frames_of(4, 4, 0));
        assert!(matches!(err, Err(SequenceError::NoFrames)));
    }

    #[test]
    fn stills_are_written_and_reported_in_order() {
        let dir = std::env::temp_dir().join("threers_seq_stills");
        let _ = std::fs::remove_dir_all(&dir);
        let opts = SequenceOptions::new(3)
            .frames_to(&dir)
            .frame_stem("shot")
            .progress(false);
        let report = render_sequence(4, 4, &opts, |i| frames_of(4, 4, i as u8)).expect("sequence");

        assert_eq!(report.frames, 3);
        assert_eq!(report.frame_paths.len(), 3);
        for (i, path) in report.frame_paths.iter().enumerate() {
            assert!(path.exists(), "{path:?} was not written");
            assert!(
                path.to_string_lossy()
                    .ends_with(&format!("shot_{i:05}.png")),
                "unexpected name {path:?}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A closure returning the wrong size has to be caught rather than written
    /// out as a malformed PNG or handed to the encoder.
    #[test]
    fn a_short_frame_is_rejected_with_its_index() {
        let opts = SequenceOptions::new(4).progress(false);
        let err = render_sequence(4, 4, &opts, |i| {
            if i == 2 {
                vec![0u8; 3]
            } else {
                frames_of(4, 4, 0)
            }
        });
        match err {
            Err(SequenceError::FrameSize {
                frame,
                got,
                expected,
            }) => {
                assert_eq!(frame, 2);
                assert_eq!(got, 3);
                assert_eq!(expected, 64);
            }
            other => panic!("expected a FrameSize error, got {other:?}"),
        }
    }

    /// The renderer is the expensive half, so it must run once per frame no
    /// matter how many sinks are attached.
    #[test]
    fn the_frame_closure_runs_once_per_frame() {
        let dir = std::env::temp_dir().join("threers_seq_once");
        let _ = std::fs::remove_dir_all(&dir);
        let mut calls = 0usize;
        let opts = SequenceOptions::new(5).frames_to(&dir).progress(false);
        render_sequence(2, 2, &opts, |_| {
            calls += 1;
            frames_of(2, 2, 1)
        })
        .expect("sequence");
        assert_eq!(calls, 5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The encoder opens its own output, so a missing directory surfaces as a
    /// broken pipe only after the whole shot has been rendered. The directory
    /// has to exist before the first frame.
    #[test]
    fn the_videos_directory_is_created_before_rendering() {
        let root = std::env::temp_dir().join("threers_seq_video_dir");
        let _ = std::fs::remove_dir_all(&root);
        let nested = root.join("a").join("b");
        let opts = SequenceOptions::new(1)
            .video_to(nested.join("shot.mp4").to_string_lossy().to_string())
            .progress(false);

        // Without the `video` feature this reports that instead, and either way
        // it must not be a missing directory.
        let _ = render_sequence(2, 2, &opts, |_| frames_of(2, 2, 0));
        assert!(nested.exists(), "{nested:?} was not created");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A bare filename has no parent to create, and must not be treated as one.
    #[test]
    fn a_video_path_with_no_directory_is_fine() {
        let opts = SequenceOptions::new(1).video_to("shot.mp4").progress(false);
        let result = render_sequence(2, 2, &opts, |_| frames_of(2, 2, 0));
        // Whatever happens next, it is not an IO error about a directory.
        if let Err(SequenceError::Io(e)) = &result {
            panic!("bare filename produced an IO error: {e}");
        }
        let _ = std::fs::remove_file("shot.mp4");
    }

    #[test]
    fn a_clock_reads_in_the_largest_useful_unit() {
        assert_eq!(clock(9.0), "9s");
        assert_eq!(clock(75.0), "1m15s");
        assert_eq!(clock(3700.0), "1h01m");
        assert_eq!(clock(f64::NAN), "?");
    }

    #[test]
    fn seconds_per_frame_survives_an_empty_report() {
        assert_eq!(SequenceReport::default().seconds_per_frame(), 0.0);
    }
}
