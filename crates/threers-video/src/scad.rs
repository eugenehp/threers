//! Video from an OpenSCAD animation: [`ScadVideo`] gives
//! [`ScadRender`] the `export_video` it had before video export left the
//! core (the `openscad` feature).

use std::path::Path;

use threers::openscad::animate::{ScadAnimation, ScadRender};

use crate::export::{export_video, VideoCodec, VideoOptions};

/// Rendering a [`ScadAnimation`] straight to a video file.
///
/// ```no_run
/// use threers::openscad::animate::{ScadAnimation, ScadCamera, ScadRender};
/// use threers_video::ScadVideo;
///
/// // A model that opens over one loop of $t.
/// let mut animation = ScadAnimation::from_file("hinge.scad").frames(120).fps(30);
/// ScadRender::new(1280, 720)
///     .camera(ScadCamera::turntable())
///     .export_video(&mut animation, "hinge.mp4")
///     .unwrap();
/// ```
pub trait ScadVideo {
    /// Render `animation` and encode it to `output`, picking the codec from the
    /// file extension (`.mp4` → H.264, `.webm` → VP9, `.gif`, `.apng`, `.mov`
    /// → HEVC).
    ///
    /// For control over quality, captions, or the codec, build your own
    /// [`VideoOptions`] and use [`export_video_with`](Self::export_video_with).
    fn export_video(
        &self,
        animation: &mut ScadAnimation,
        output: impl AsRef<Path>,
    ) -> Result<(), String>;

    /// Render `animation` and encode it with explicit [`VideoOptions`] — the
    /// way to set quality, a codec that does not match the extension, or
    /// subtitles.
    ///
    /// The options' `fps` is used as given; set it from
    /// [`ScadAnimation::frame_rate`] to keep playback at the authored rate.
    fn export_video_with(
        &self,
        animation: &mut ScadAnimation,
        options: &VideoOptions,
    ) -> Result<(), String>;
}

impl ScadVideo for ScadRender {
    fn export_video(
        &self,
        animation: &mut ScadAnimation,
        output: impl AsRef<Path>,
    ) -> Result<(), String> {
        let path = output.as_ref();
        let options = VideoOptions::new(path.to_string_lossy().into_owned())
            .fps(animation.frame_rate())
            .codec(codec_for_path(path));
        self.export_video_with(animation, &options)
    }

    fn export_video_with(
        &self,
        animation: &mut ScadAnimation,
        options: &VideoOptions,
    ) -> Result<(), String> {
        let frames = self.render_frames(animation)?;
        if frames.is_empty() {
            return Err("animation produced no frames".into());
        }
        let (width, height) = self.size();
        export_video(width, height, frames.len(), options, |i| frames[i].clone())
            .map_err(|e| e.to_string())
    }
}

/// Pick a codec from an output path's extension.
fn codec_for_path(path: &Path) -> VideoCodec {
    match path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .as_deref()
    {
        Some("webm") => VideoCodec::Vp9,
        Some("gif") => VideoCodec::Gif,
        Some("apng" | "png") => VideoCodec::Apng,
        Some("mov") => VideoCodec::Hevc,
        // `.mp4` and anything unrecognized: the most widely playable option.
        _ => VideoCodec::H264,
    }
}
