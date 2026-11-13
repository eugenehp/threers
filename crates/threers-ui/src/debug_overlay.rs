//! Frame statistics painted over the UI:
//! how long frames take, how long layout takes, and how much work each did.
//!
//! The app times its own frames (this crate has no clock that works in
//! every target) and records them with what [`Ui`] says it did; the
//! overlay keeps the last [`FrameStats::CAPACITY`] and paints a panel with
//! the latest numbers, their 95th percentile and a bar per frame.
//!
//! ```
//! use threers_ui::debug_overlay::{FrameSample, FrameStats};
//! use threers_ui::scene::Scene;
//!
//! let mut stats = FrameStats::default();
//! for i in 0..10 {
//!     stats.record(FrameSample { frame_ms: 4.0 + i as f32, layout_ms: 0.5, ..Default::default() });
//! }
//! assert_eq!(stats.percentile(0.5, |s| s.frame_ms), 9.0);
//! let mut scene = Scene::default();
//! // Painted as the last thing in a frame.
//! stats.paint(&mut scene.recorder(), 8.0, 8.0);
//! assert!(!scene.items.is_empty());
//! ```

use std::collections::VecDeque;

use super::paint::{Painter, TextStyle};
use super::style::{FontWeight, Rgba, Style};
use super::tree::{Rect, Ui};

/// One frame's numbers.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FrameSample {
    /// The whole frame, in milliseconds.
    pub frame_ms: f32,
    /// [`Ui::layout`], in milliseconds.
    pub layout_ms: f32,
    /// Nodes laid out rather than taken from the layout cache.
    pub laid_out: usize,
    /// Nodes in the tree.
    pub nodes: usize,
    /// Items painted (from a [`Scene`](super::scene::Scene)), or 0.
    pub painted: usize,
}

impl FrameSample {
    /// A sample with what `ui` reports of its last layout filled in.
    pub fn of(ui: &Ui, frame_ms: f32, layout_ms: f32) -> Self {
        FrameSample {
            frame_ms,
            layout_ms,
            laid_out: ui.laid_out_last(),
            nodes: ui.live_nodes(),
            painted: 0,
        }
    }
}

/// The last frames' samples. See the [module docs](self).
#[derive(Debug, Clone, Default)]
pub struct FrameStats {
    samples: VecDeque<FrameSample>,
}

impl FrameStats {
    /// Frames kept.
    pub const CAPACITY: usize = 120;

    pub fn record(&mut self, sample: FrameSample) {
        if self.samples.len() == Self::CAPACITY {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    pub fn latest(&self) -> Option<FrameSample> {
        self.samples.back().copied()
    }

    pub fn samples(&self) -> impl Iterator<Item = &FrameSample> {
        self.samples.iter()
    }

    /// The `p` quantile (0..=1) of `field` over the kept frames.
    pub fn percentile(&self, p: f32, field: impl Fn(&FrameSample) -> f32) -> f32 {
        let mut v: Vec<f32> = self.samples.iter().map(field).collect();
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(f32::total_cmp);
        v[((v.len() - 1) as f32 * p.clamp(0.0, 1.0)).round() as usize]
    }

    /// Paints the panel with its top-left corner at `(x, y)`.
    pub fn paint(&self, painter: &mut dyn Painter, x: f32, y: f32) {
        const W: f32 = 196.0;
        const BARS: f32 = 36.0;
        const BG: Rgba = [0.05, 0.06, 0.08, 0.86];
        const BUDGET: f32 = 1000.0 / 60.0;
        let Some(last) = self.latest() else { return };
        let lines = [
            format!(
                "frame {:5.2} ms  p95 {:5.2}",
                last.frame_ms,
                self.percentile(0.95, |s| s.frame_ms)
            ),
            format!(
                "layout {:4.2} ms  p95 {:4.2}",
                last.layout_ms,
                self.percentile(0.95, |s| s.layout_ms)
            ),
            format!("laid out {} of {} nodes", last.laid_out, last.nodes),
            format!("painted {} items", last.painted),
        ];
        let line_h = 14.0;
        let h = 8.0 + lines.len() as f32 * line_h + 6.0 + BARS + 8.0;
        painter.fill_rounded_rect(Rect { x, y, width: W, height: h }, 4.0, 4.0, BG);
        let style = TextStyle::of(&Style {
            font_family: "monospace".into(),
            font_size: 11.0,
            font_weight: FontWeight::Normal,
            color: [0.86, 0.88, 0.92, 1.0],
            ..Style::default()
        });
        for (i, text) in lines.iter().enumerate() {
            let line = Rect {
                x: x + 8.0,
                y: y + 6.0 + i as f32 * line_h,
                width: W - 16.0,
                height: line_h,
            };
            painter.text(line, text, &style);
        }
        // A bar per frame, scaled so a 60 Hz frame is half height; green
        // under budget, amber under two, red past.
        let base = y + h - 8.0;
        let bar_w = (W - 16.0) / Self::CAPACITY as f32;
        let start = Self::CAPACITY - self.samples.len();
        for (i, s) in self.samples.iter().enumerate() {
            let bh = (s.frame_ms / (2.0 * BUDGET) * BARS).clamp(1.0, BARS);
            let color = if s.frame_ms <= BUDGET {
                [0.36, 0.78, 0.45, 0.9]
            } else if s.frame_ms <= 2.0 * BUDGET {
                [0.95, 0.72, 0.25, 0.9]
            } else {
                [0.94, 0.33, 0.31, 0.9]
            };
            let r = Rect {
                x: x + 8.0 + (start + i) as f32 * bar_w,
                y: base - bh,
                width: bar_w.max(1.0) * 0.8,
                height: bh,
            };
            painter.fill_rect(r, color);
        }
        // The budget line.
        painter.fill_rect(
            Rect {
                x: x + 8.0,
                y: base - BARS / 2.0,
                width: W - 16.0,
                height: 1.0,
            },
            [1.0, 1.0, 1.0, 0.35],
        );
    }
}
