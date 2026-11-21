//! When the UI draws: every frame its rate allows, or only when something in
//! it changed.
//!
//! [`UiRedraw`] decides, frame by frame, whether drawing the UI is worth it,
//! and tells the event loop when to wake next. In
//! [`RedrawMode::OnChange`] a still window draws nothing at all: it draws when
//! the tree changed or animates ([`Ui::needs_frame`]), when the app says so
//! ([`request`](UiRedraw::request): input, a resize, the OS asking), or when
//! a 3D view it shows has a new frame. In [`RedrawMode::Continuous`] it draws
//! at its rate regardless, which a UI showing a clock or video wants. Either
//! way the rate is capped.
//!
//! With winit:
//!
//! ```ignore
//! let mut redraw = UiRedraw::new(RedrawMode::OnChange, 60.0);
//! // A new 3D frame wakes the event loop.
//! let w = window.clone();
//! view.frames().on_frame(move || w.request_redraw());
//!
//! // In the event handler:
//! WindowEvent::RedrawRequested => {
//!     if !redraw.should_draw(&ui, &[view.frames()], Instant::now()) { return; }
//!     // ... lay out, update views, draw, present ...
//! }
//! other => { redraw.request(); /* handle input */ }
//! Event::AboutToWait => match redraw.next_wake(&ui, &[view.frames()]) {
//!     Some(at) if at <= Instant::now() => window.request_redraw(),
//!     Some(at) => target.set_control_flow(ControlFlow::WaitUntil(at)),
//!     None => target.set_control_flow(ControlFlow::Wait),
//! }
//! ```

use web_time::Instant;

use crate::renderer::{FramePacer, ViewFrames};
use crate::ui::Ui;

/// When a [`UiRedraw`] draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RedrawMode {
    /// Every frame the rate allows.
    #[default]
    Continuous,
    /// Only when something changed: the tree, an animation, input, a view's
    /// frame.
    OnChange,
}

/// Decides when the UI draws. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct UiRedraw {
    mode: RedrawMode,
    pacer: FramePacer,
    /// Something asked since the last frame drawn.
    pending: bool,
    drawn: u64,
    skipped: u64,
}

impl UiRedraw {
    /// Draws as `mode` says, at most `max_fps` frames a second (0 for no
    /// limit).
    pub fn new(mode: RedrawMode, max_fps: f32) -> Self {
        Self {
            mode,
            pacer: FramePacer::new(max_fps),
            // The first frame always draws.
            pending: true,
            drawn: 0,
            skipped: 0,
        }
    }

    pub fn mode(&self) -> RedrawMode {
        self.mode
    }

    /// Changing mode draws a frame, so the switch shows.
    pub fn set_mode(&mut self, mode: RedrawMode) {
        self.mode = mode;
        self.pending = true;
    }

    pub fn set_max_fps(&mut self, fps: f32) {
        self.pacer.set_fps(fps);
    }

    /// The cap, `None` for none.
    pub fn max_fps(&self) -> Option<f32> {
        self.pacer.fps()
    }

    /// Something outside the tree changed: input arrived, the window resized
    /// or was uncovered. The next frame draws.
    pub fn request(&mut self) {
        self.pending = true;
    }

    /// Whether anything wants a frame, cap aside.
    pub fn wanted(&self, ui: &Ui, views: &[&ViewFrames]) -> bool {
        self.mode == RedrawMode::Continuous
            || self.pending
            || ui.needs_frame()
            || views.iter().any(|v| v.has_new_frame())
    }

    /// Whether to draw a frame at `now`. When it says yes, the frame counts
    /// as drawn. When something wants a frame the cap holds back, it stays
    /// wanted, for [`next_wake`](Self::next_wake).
    pub fn should_draw(&mut self, ui: &Ui, views: &[&ViewFrames], now: Instant) -> bool {
        if !self.wanted(ui, views) {
            self.skipped += 1;
            return false;
        }
        if !self.pacer.due(now) {
            self.pending = true;
            return false;
        }
        self.pending = false;
        self.drawn += 1;
        true
    }

    /// When the event loop should wake to draw: `Some` of a time (now or
    /// past means at once), or `None` to sleep until an event.
    pub fn next_wake(&self, ui: &Ui, views: &[&ViewFrames]) -> Option<Instant> {
        self.wanted(ui, views).then(|| self.pacer.next())
    }

    /// Frames drawn, and frames asked for that had nothing to draw.
    pub fn counts(&self) -> (u64, u64) {
        (self.drawn, self.skipped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn on_change_draws_once_then_only_when_something_changes() {
        let mut ui = Ui::new();
        let root = ui.root();
        let text = ui.add_text(root, "a");
        ui.layout(100.0, 100.0, &mut crate::ui::FixedWidthText(7.0));
        let mut r = UiRedraw::new(RedrawMode::OnChange, 0.0);
        let now = Instant::now();
        // The first frame draws; then a still tree draws nothing.
        assert!(r.should_draw(&ui, &[], now));
        assert!(!r.should_draw(&ui, &[], now));
        assert_eq!(r.next_wake(&ui, &[]), None);
        // A change to the tree draws (until laid out again).
        ui.set_text(text, "b");
        assert!(r.should_draw(&ui, &[], now));
        ui.layout(100.0, 100.0, &mut crate::ui::FixedWidthText(7.0));
        assert!(!r.should_draw(&ui, &[], now));
        // Input draws once.
        r.request();
        assert!(r.should_draw(&ui, &[], now));
        assert!(!r.should_draw(&ui, &[], now));
    }

    #[test]
    fn the_cap_defers_rather_than_drops() {
        let ui = Ui::new();
        let mut r = UiRedraw::new(RedrawMode::OnChange, 10.0);
        let t0 = r.pacer.next();
        assert!(r.should_draw(&ui, &[], t0));
        r.request();
        // Too soon: held back, still wanted, wake at the next slot.
        assert!(!r.should_draw(&ui, &[], t0 + Duration::from_millis(10)));
        assert_eq!(r.next_wake(&ui, &[]), Some(t0 + Duration::from_millis(100)));
        assert!(r.should_draw(&ui, &[], t0 + Duration::from_millis(100)));
    }

    #[test]
    fn continuous_always_wants_a_frame() {
        let ui = Ui::new();
        let mut r = UiRedraw::new(RedrawMode::Continuous, 0.0);
        let now = Instant::now();
        assert!(r.should_draw(&ui, &[], now) && r.should_draw(&ui, &[], now));
        assert!(r.next_wake(&ui, &[]).is_some());
    }
}
