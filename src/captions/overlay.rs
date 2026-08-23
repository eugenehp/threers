//! A frame-sized caption overlay that only re-rasterizes when it has to.
//!
//! On-screen captions are redrawn every frame but change only when the cue
//! changes — typically a few times per minute. [`CaptionOverlay`] keeps the
//! rasterized RGBA buffer and rebuilds it only when the active cue set or the
//! frame size differs from last time, so per-frame cost is a texture upload
//! rather than a text layout.
//!
//! Pair it with
//! [`Renderer::draw_caption_overlay`](crate::renderer::Renderer::draw_caption_overlay)
//! for wgpu output, or read [`rgba`](CaptionOverlay::rgba) directly to
//! composite yourself.
//!
//! ```
//! use threers::captions::{CaptionOverlay, CaptionTrack};
//! let mut overlay = CaptionOverlay::new(CaptionTrack::new().cue(0.0, 2.0, "Live"));
//! overlay.set_size(640, 360);
//! assert!(overlay.update(1.0));       // rasterized
//! assert!(!overlay.update(1.5));      // same cue, buffer reused
//! assert!(overlay.is_visible());
//! assert!(overlay.update(3.0));       // cue ended, buffer cleared
//! assert!(!overlay.is_visible());
//! ```

use super::{CaptionPainter, CaptionStyle, CaptionTrack};

/// Caches a rasterized caption overlay across frames.
#[derive(Debug)]
pub struct CaptionOverlay {
    /// The cues to display.
    pub track: CaptionTrack,
    /// The face and style used to draw them.
    pub painter: CaptionPainter,
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    /// Indices of the cues the current buffer was drawn from.
    active: Vec<usize>,
    /// Whether `active` reflects a completed rasterization.
    valid: bool,
    /// Style to rescale from when `auto_scale` is on.
    base_style: CaptionStyle,
    auto_scale: bool,
}

impl CaptionOverlay {
    /// An overlay for `track`, using the built-in face and default style.
    pub fn new(track: CaptionTrack) -> Self {
        Self::with_painter(track, CaptionPainter::new())
    }

    /// An overlay drawing `track` with `painter`.
    pub fn with_painter(track: CaptionTrack, painter: CaptionPainter) -> Self {
        let base_style = painter.style.clone();
        Self {
            track,
            painter,
            width: 0,
            height: 0,
            rgba: Vec::new(),
            active: Vec::new(),
            valid: false,
            base_style,
            auto_scale: false,
        }
    }

    /// Rescale the style to the frame height on every resize, treating the
    /// current style as authored for 1080p. Off by default.
    ///
    /// ```
    /// use threers::captions::{CaptionOverlay, CaptionTrack};
    /// let mut o = CaptionOverlay::new(CaptionTrack::new()).auto_scale(true);
    /// o.set_size(1920, 540);
    /// assert!((o.painter.style.font_size - 17.0).abs() < 1e-4); // half of 34
    /// ```
    pub fn auto_scale(mut self, enabled: bool) -> Self {
        self.set_auto_scale(enabled);
        self
    }

    /// Turn [`auto_scale`](Self::auto_scale) on or off in place.
    pub fn set_auto_scale(&mut self, enabled: bool) {
        if self.auto_scale == enabled {
            return;
        }
        self.auto_scale = enabled;
        if self.height > 0 {
            self.painter.style = if enabled {
                self.base_style.for_height(self.height)
            } else {
                self.base_style.clone()
            };
            self.valid = false;
        }
    }

    /// The style as authored, before any [`auto_scale`](Self::auto_scale)
    /// rescaling — the value [`set_style`](Self::set_style) last took.
    ///
    /// Read `overlay.painter.style` instead for the values actually being
    /// drawn with.
    pub fn style(&self) -> &CaptionStyle {
        &self.base_style
    }

    /// Replace the style. Also becomes the new [`auto_scale`](Self::auto_scale)
    /// baseline.
    pub fn set_style(&mut self, style: CaptionStyle) {
        self.base_style = style.clone();
        self.painter.style = if self.auto_scale && self.height > 0 {
            style.for_height(self.height)
        } else {
            style
        };
        self.valid = false;
    }

    /// Replace the cues, invalidating the cached raster.
    pub fn set_track(&mut self, track: CaptionTrack) {
        self.track = track;
        self.valid = false;
    }

    /// Set the overlay resolution — normally the render target size.
    pub fn set_size(&mut self, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;
        self.rgba = vec![0u8; (width as usize) * (height as usize) * 4];
        if self.auto_scale && height > 0 {
            self.painter.style = self.base_style.for_height(height);
        }
        self.valid = false;
    }

    /// Overlay resolution in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Bring the buffer up to date for `time` seconds.
    ///
    /// Returns `true` when the buffer changed and needs re-uploading — this is
    /// the signal to skip the upload on the frames in between.
    pub fn update(&mut self, time: f64) -> bool {
        if self.width == 0 || self.height == 0 {
            return false;
        }
        let active: Vec<usize> = self
            .track
            .cues
            .iter()
            .enumerate()
            .filter(|(_, c)| c.is_active_at(time))
            .map(|(i, _)| i)
            .collect();
        if self.valid && active == self.active {
            return false;
        }
        self.active = active;
        self.valid = true;
        self.rgba.fill(0);
        if !self.active.is_empty() {
            self.painter
                .burn_in(&mut self.rgba, self.width, self.height, &self.track, time);
        }
        true
    }

    /// Force the next [`update`](Self::update) to re-rasterize.
    pub fn invalidate(&mut self) {
        self.valid = false;
    }

    /// Whether a cue is currently showing.
    pub fn is_visible(&self) -> bool {
        !self.active.is_empty()
    }

    /// The RGBA8 overlay buffer, `width * height * 4` bytes, transparent where
    /// there is no caption. Empty until [`set_size`](Self::set_size) is called.
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }

    /// Composite the overlay onto an RGBA frame of the same size. A no-op when
    /// nothing is showing or the sizes disagree.
    pub fn composite(&self, frame: &mut [u8]) -> bool {
        if !self.is_visible() || frame.len() != self.rgba.len() {
            return false;
        }
        for (dst, src) in frame.chunks_exact_mut(4).zip(self.rgba.chunks_exact(4)) {
            let sa = src[3] as u32;
            if sa == 0 {
                continue;
            }
            if sa == 255 {
                dst.copy_from_slice(src);
                continue;
            }
            let da = dst[3] as u32;
            let out_a = sa + da * (255 - sa) / 255;
            for c in 0..3 {
                let numerator = src[c] as u32 * sa + dst[c] as u32 * da * (255 - sa) / 255;
                dst[c] = numerator.checked_div(out_a).map_or(0, |v| v.min(255) as u8);
            }
            dst[3] = out_a.min(255) as u8;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::captions::CaptionStyle;

    fn overlay() -> CaptionOverlay {
        let track = CaptionTrack::new()
            .cue(0.0, 2.0, "first")
            .cue(2.0, 4.0, "second");
        let mut o = CaptionOverlay::new(track);
        o.set_style(CaptionStyle::default().font_size(12.0).margin(6.0));
        o.set_size(240, 135);
        o
    }

    #[test]
    fn rasterizes_once_per_cue_change() {
        let mut o = overlay();
        assert!(o.update(0.5), "first update must rasterize");
        assert!(!o.update(1.0), "same cue must reuse the buffer");
        assert!(!o.update(1.9));
        assert!(o.update(2.5), "cue change must re-rasterize");
        assert!(o.update(9.0), "cue ending must clear the buffer");
        assert!(!o.update(9.5), "still nothing showing");
    }

    #[test]
    fn buffer_is_transparent_when_nothing_shows() {
        let mut o = overlay();
        o.update(9.0);
        assert!(!o.is_visible());
        assert!(o.rgba().iter().all(|&b| b == 0));

        o.update(0.5);
        assert!(o.is_visible());
        assert!(o.rgba().chunks_exact(4).any(|px| px[3] > 0));
    }

    #[test]
    fn resizing_reallocates_and_invalidates() {
        let mut o = overlay();
        o.update(0.5);
        assert_eq!(o.rgba().len(), (240 * 135 * 4) as usize);
        o.set_size(320, 180);
        assert_eq!(o.rgba().len(), (320 * 180 * 4) as usize);
        assert!(o.update(0.5), "resize must force a redraw");
    }

    #[test]
    fn auto_scale_tracks_the_frame_height() {
        let mut o = CaptionOverlay::new(CaptionTrack::new().cue(0.0, 1.0, "x")).auto_scale(true);
        o.set_size(1920, 1080);
        let full = o.painter.style.font_size;
        o.set_size(960, 540);
        assert!((o.painter.style.font_size - full * 0.5).abs() < 1e-4);
    }

    #[test]
    fn auto_scale_toggles_back_to_the_authored_style() {
        let mut o = CaptionOverlay::new(CaptionTrack::new().cue(0.0, 1.0, "x"));
        o.set_style(CaptionStyle::default().font_size(40.0));
        o.set_size(1920, 540); // half of 1080

        assert_eq!(o.painter.style.font_size, 40.0, "off by default");
        o.set_auto_scale(true);
        assert_eq!(o.painter.style.font_size, 20.0);
        o.set_auto_scale(false);
        assert_eq!(o.painter.style.font_size, 40.0, "must restore the original");

        // `style()` always reports the authored values, never the scaled ones.
        o.set_auto_scale(true);
        assert_eq!(o.style().font_size, 40.0);
        assert_eq!(o.painter.style.font_size, 20.0);
    }

    #[test]
    fn style_and_track_changes_invalidate_the_cache() {
        let mut o = overlay();
        o.update(0.5);
        o.set_style(CaptionStyle::default().font_size(20.0));
        assert!(o.update(0.5), "style change must redraw");
        o.update(0.5);
        o.set_track(CaptionTrack::new().cue(0.0, 5.0, "replaced"));
        assert!(o.update(0.5), "track change must redraw");
        o.update(0.5);
        o.invalidate();
        assert!(o.update(0.5), "explicit invalidate must redraw");
    }

    #[test]
    fn composite_writes_only_where_the_overlay_is_opaque() {
        let mut o = overlay();
        o.update(0.5);
        let mut frame = vec![0u8; o.rgba().len()];
        for px in frame.chunks_exact_mut(4) {
            px.copy_from_slice(&[10, 20, 30, 255]);
        }
        assert!(o.composite(&mut frame));
        let untouched = frame
            .chunks_exact(4)
            .filter(|px| px[..3] == [10, 20, 30])
            .count();
        let changed = frame.chunks_exact(4).count() - untouched;
        assert!(changed > 0, "expected caption pixels");
        assert!(untouched > changed, "expected most of the frame untouched");
        assert!(frame.chunks_exact(4).all(|px| px[3] == 255));
    }

    #[test]
    fn composite_is_a_noop_on_mismatched_buffers() {
        let mut o = overlay();
        o.update(0.5);
        let mut wrong = vec![0u8; 8];
        assert!(!o.composite(&mut wrong));
        assert!(wrong.iter().all(|&b| b == 0));
    }

    #[test]
    fn zero_sized_overlay_never_rasterizes() {
        let mut o = CaptionOverlay::new(CaptionTrack::new().cue(0.0, 1.0, "x"));
        assert!(!o.update(0.5));
        assert!(o.rgba().is_empty());
    }
}
