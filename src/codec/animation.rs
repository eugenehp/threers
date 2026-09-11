//! High-level encoders for browser-friendly animated image/video containers.

use std::fmt;

use crate::codec::{
    apng::ApngEncoder,
    gif::{encode_gif, GifOptions, PaletteMode},
    h264::{
        encode_compressed_mp4_from_iter as encode_h264_mp4_from_iter, encode_mp4_with_captions,
    },
    hevc::{encode_compressed_mp4, encode_compressed_mp4_from_iter, Yuv420Frame},
    webm::encode_webm,
};

/// Browser-friendly animation containers (no ffmpeg).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrowserCodec {
    Gif,
    Apng,
    Webm,
    /// H.264 in an MP4 container (`avc1` + `avcC`). Universally playable.
    ///
    /// Transform-coded intra, roughly a hundredth the size of the `I_PCM` path
    /// this used to take. [`Mp4Hevc`](Self::Mp4Hevc) is smaller again where its
    /// narrower playback reach is acceptable.
    Mp4,
    /// HEVC in an MP4 container (`hvc1` + `hvcC`), transform-coded.
    ///
    /// Two orders of magnitude smaller than [`Mp4`](Self::Mp4) — 0.16 MB against
    /// 47 MB on the 17-frame clip in `docs/codec-benchmark.md`. The cost is
    /// reach: HEVC in MP4 plays in Safari everywhere and in Chrome/Edge on
    /// hardware that supports it, but not universally the way H.264 does.
    Mp4Hevc,
}

impl BrowserCodec {
    pub fn label(self) -> &'static str {
        match self {
            Self::Gif => "gif",
            Self::Apng => "apng",
            Self::Webm => "webm",
            Self::Mp4 | Self::Mp4Hevc => "mp4",
        }
    }
}

/// Progress phase for [`encode_animation_rgba_with_progress`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimationExportPhase {
    Capture,
    Encode,
    Done,
}

/// Progress tick for browser animation encoding.
#[derive(Clone, Debug)]
pub struct AnimationExportProgress {
    pub phase: AnimationExportPhase,
    pub frame: u32,
    pub frames: u32,
    pub ratio: f32,
    pub codec: BrowserCodec,
    pub message: String,
}

/// Ready-to-display animation progress line.
pub fn format_animation_progress(info: &AnimationExportProgress) -> String {
    let pct = (info.ratio.clamp(0.0, 1.0) * 100.0).round() as i32;
    match info.phase {
        AnimationExportPhase::Capture => {
            format!("Rendering {}/{} ({pct}%)", info.frame, info.frames)
        }
        AnimationExportPhase::Encode => {
            format!(
                "Encoding {}… ({pct}%)",
                info.codec.label().to_ascii_uppercase()
            )
        }
        AnimationExportPhase::Done => format!("Done ({} frames)", info.frames),
    }
}

fn make_anim_progress(
    phase: AnimationExportPhase,
    frame: u32,
    frames: u32,
    ratio: f32,
    codec: BrowserCodec,
) -> AnimationExportProgress {
    let mut info = AnimationExportProgress {
        phase,
        frame,
        frames,
        ratio: ratio.clamp(0.0, 1.0),
        codec,
        message: String::new(),
    };
    info.message = format_animation_progress(&info);
    info
}

/// Errors produced while encoding a browser animation.
#[derive(Debug)]
pub enum AnimationEncodeError {
    Empty,
    FrameSize {
        frame: usize,
        expected: usize,
        got: usize,
    },
    Dimension,
    Io(String),
}

impl fmt::Display for AnimationEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "animation needs at least one frame"),
            Self::FrameSize {
                frame,
                expected,
                got,
            } => write!(
                f,
                "frame {frame} has {got} bytes, expected {expected} (width*height*4)"
            ),
            Self::Dimension => write!(f, "invalid dimensions, frame rate, or codec options"),
            Self::Io(message) => write!(f, "animation encoding I/O failed: {message}"),
        }
    }
}

impl std::error::Error for AnimationEncodeError {}

/// Options for [`encode_animation_rgba`].
pub struct AnimationEncodeOptions {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub codec: BrowserCodec,
    pub transparent: bool,
    /// GIF palette size (`2..=256`; normally `256`).
    pub gif_colors: u16,
}

impl Default for AnimationEncodeOptions {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            fps: 30,
            codec: BrowserCodec::Gif,
            transparent: false,
            gif_colors: 256,
        }
    }
}

/// Encode tightly-packed RGBA8 frames into a browser-friendly animation.
pub fn encode_animation_rgba(
    opts: &AnimationEncodeOptions,
    frames: impl IntoIterator<Item = Vec<u8>>,
) -> Result<Vec<u8>, AnimationEncodeError> {
    encode_animation_rgba_with_progress(opts, frames, |_| {})
}

/// Like [`encode_animation_rgba`], with encode-phase progress callbacks.
pub fn encode_animation_rgba_with_progress<I, P>(
    opts: &AnimationEncodeOptions,
    frames: I,
    mut on_progress: P,
) -> Result<Vec<u8>, AnimationEncodeError>
where
    I: IntoIterator<Item = Vec<u8>>,
    P: FnMut(AnimationExportProgress),
{
    let pixels = opts
        .width
        .checked_mul(opts.height)
        .and_then(|n| n.checked_mul(4))
        .ok_or(AnimationEncodeError::Dimension)?;
    if opts.width == 0
        || opts.height == 0
        || opts.fps == 0
        || opts.fps > u16::MAX as u32
        || (matches!(opts.codec, BrowserCodec::Gif) && !(2..=256).contains(&opts.gif_colors))
    {
        return Err(AnimationEncodeError::Dimension);
    }
    if matches!(opts.codec, BrowserCodec::Gif)
        && (opts.width > u16::MAX as u32 || opts.height > u16::MAX as u32)
    {
        return Err(AnimationEncodeError::Dimension);
    }
    if matches!(opts.codec, BrowserCodec::Webm)
        && (!opts.width.is_multiple_of(8) || !opts.height.is_multiple_of(8))
    {
        return Err(AnimationEncodeError::Dimension);
    }
    if matches!(opts.codec, BrowserCodec::Mp4 | BrowserCodec::Mp4Hevc)
        && (!opts.width.is_multiple_of(2) || !opts.height.is_multiple_of(2))
    {
        return Err(AnimationEncodeError::Dimension);
    }

    let expected = pixels as usize;

    // The MP4 codecs can consume frames as they arrive. Everything else needs
    // them all at once (GIF quantizes a palette across the set, WebM and APNG
    // take slices), so only these two avoid the collect — which matters most
    // here, since an RGBA frame is 4 bytes per pixel: 133 MB each at 8K.
    if matches!(opts.codec, BrowserCodec::Mp4 | BrowserCodec::Mp4Hevc) && !opts.transparent {
        return encode_mp4_streaming(opts, frames, expected, on_progress);
    }

    let mut frames: Vec<Vec<u8>> = frames.into_iter().collect();
    if frames.is_empty() {
        return Err(AnimationEncodeError::Empty);
    }
    for (frame, rgba) in frames.iter().enumerate() {
        if rgba.len() != expected {
            return Err(AnimationEncodeError::FrameSize {
                frame,
                expected,
                got: rgba.len(),
            });
        }
    }

    let total = frames.len() as u32;
    on_progress(make_anim_progress(
        AnimationExportPhase::Encode,
        0,
        total,
        0.0,
        opts.codec,
    ));

    let delay_den = opts.fps as u16;
    let bytes = match opts.codec {
        BrowserCodec::Gif => {
            let gif_opts = GifOptions::default()
                .colors(opts.gif_colors)
                .transparency(opts.transparent)
                .dither(true)
                .diff_rects(true)
                .palette_mode(PaletteMode::Local);
            encode_gif(
                opts.width,
                opts.height,
                0,
                &gif_opts,
                frames.into_iter().map(|rgba| (rgba, 1, delay_den)),
            )
        }
        BrowserCodec::Apng => {
            let mut encoder = ApngEncoder::new(opts.width, opts.height, 0);
            let n = frames.len();
            for (i, mut rgba) in frames.into_iter().enumerate() {
                if !opts.transparent {
                    for pixel in rgba.chunks_exact_mut(4) {
                        pixel[3] = 255;
                    }
                }
                encoder.add_frame(&rgba, 1, delay_den);
                let done = (i + 1) as u32;
                on_progress(make_anim_progress(
                    AnimationExportPhase::Encode,
                    done,
                    n as u32,
                    done as f32 / n as f32,
                    opts.codec,
                ));
            }
            encoder.finish()
        }
        BrowserCodec::Webm => {
            let yuv: Vec<Yuv420Frame> = frames
                .drain(..)
                .map(|rgba| {
                    let mut frame = Yuv420Frame::from_rgba(opts.width, opts.height, &rgba);
                    if !opts.transparent {
                        frame.alpha = None;
                    }
                    frame
                })
                .collect();
            let refs: Vec<&Yuv420Frame> = yuv.iter().collect();
            encode_webm(&refs, opts.fps)
        }
        BrowserCodec::Mp4 | BrowserCodec::Mp4Hevc => {
            if opts.transparent {
                return Err(AnimationEncodeError::Dimension);
            }
            let n = frames.len();
            let mut yuv = Vec::with_capacity(n);
            for rgba in frames.drain(..) {
                yuv.push(Yuv420Frame::from_rgba(opts.width, opts.height, &rgba));
                let done = yuv.len() as u32;
                on_progress(make_anim_progress(
                    AnimationExportPhase::Encode,
                    done,
                    n as u32,
                    done as f32 / n as f32,
                    opts.codec,
                ));
            }
            if matches!(opts.codec, BrowserCodec::Mp4Hevc) {
                // QP 26 is the middle of the usable range; see `encode_mp4`.
                encode_compressed_mp4(opts.width, opts.height, opts.fps, 26, &yuv)
            } else {
                encode_mp4_with_captions(opts.width, opts.height, opts.fps, &yuv, None)
            }
        }
    };

    on_progress(make_anim_progress(
        AnimationExportPhase::Done,
        total,
        total,
        1.0,
        opts.codec,
    ));
    Ok(bytes)
}




/// Encode the MP4 codecs without holding every source frame.
///
/// Frames are pulled from the iterator, converted, encoded and dropped one at a
/// time, so peak memory is one RGBA frame plus one YUV frame plus the encoded
/// samples — rather than every RGBA frame at once. Output is byte-identical to
/// the collecting path.
fn encode_mp4_streaming<I, P>(
    opts: &AnimationEncodeOptions,
    frames: I,
    expected: usize,
    mut on_progress: P,
) -> Result<Vec<u8>, AnimationEncodeError>
where
    I: IntoIterator<Item = Vec<u8>>,
    P: FnMut(AnimationExportProgress),
{
    let mut iter = frames.into_iter().peekable();
    if iter.peek().is_none() {
        return Err(AnimationEncodeError::Empty);
    }
    // Only used to report a ratio; an unsized iterator just reports against the
    // count so far, which still moves monotonically.
    let hint = iter.size_hint().1.unwrap_or(0) as u32;

    let mut err = None;
    let mut done = 0u32;
    let (w, h) = (opts.width, opts.height);
    let codec = opts.codec;
    let yuv = std::iter::from_fn(|| {
        if err.is_some() {
            return None;
        }
        let rgba = iter.next()?;
        if rgba.len() != expected {
            err = Some(AnimationEncodeError::FrameSize {
                frame: done as usize,
                expected,
                got: rgba.len(),
            });
            return None;
        }
        done += 1;
        on_progress(make_anim_progress(
            AnimationExportPhase::Encode,
            done,
            hint.max(done),
            done as f32 / hint.max(done) as f32,
            codec,
        ));
        Some(Yuv420Frame::from_rgba(w, h, &rgba))
    });

    let bytes = if matches!(codec, BrowserCodec::Mp4Hevc) {
        // QP 26 is the middle of the usable range; see `encode_mp4`.
        encode_compressed_mp4_from_iter(w, h, opts.fps, 26, None, yuv)
    } else {
        // QP 26 is the middle of the usable range, matching the HEVC path.
        encode_h264_mp4_from_iter(w, h, opts.fps, 26, None, yuv)
    };
    if let Some(e) = err {
        return Err(e);
    }

    on_progress(make_anim_progress(
        AnimationExportPhase::Done,
        done,
        done,
        1.0,
        codec,
    ));
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let mut frame = vec![0; 8 * 8 * 4];
        for pixel in frame.chunks_exact_mut(4) {
            pixel.copy_from_slice(&[r, g, b, a]);
        }
        frame
    }

    #[test]
    fn encodes_browser_animation_headers() {
        for (codec, header) in [
            (BrowserCodec::Gif, b"GIF8".as_slice()),
            (BrowserCodec::Apng, &[0x89, b'P', b'N', b'G']),
            (BrowserCodec::Webm, &[0x1a, 0x45, 0xdf, 0xa3]),
            (BrowserCodec::Mp4, b"ftyp".as_slice()),
        ] {
            let transparent = !matches!(codec, BrowserCodec::Mp4);
            let options = AnimationEncodeOptions {
                width: 8,
                height: 8,
                fps: 10,
                codec,
                transparent,
                gif_colors: 256,
            };
            let bytes =
                encode_animation_rgba(&options, [solid(255, 0, 0, 255), solid(0, 0, 255, 128)])
                    .unwrap();
            assert!(bytes.len() > header.len());
            if matches!(codec, BrowserCodec::Mp4) {
                assert_eq!(&bytes[4..8], header);
            } else {
                assert_eq!(&bytes[..header.len()], header);
            }
        }
    }
}
