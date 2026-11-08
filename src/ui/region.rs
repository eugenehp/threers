//! Painting one rectangle of a window.
//!
//! [`Culled`] wraps another [`Painter`] and passes on only what could reach a rectangle. What is passed on is
//! painted exactly where and how a whole-window paint would paint it: not shifted (a shifted position rounds
//! differently near a half pixel) and not clipped (clipping a plain rectangle rounds off the corners of the
//! cut). Paint into an overlay the size of the window, then copy the rectangle out. Pair it with
//! [`damage`](super::damage), which says which rectangles need painting.
//!
//! ```
//! use threers::ui::region::Culled;
//! use threers::ui::{FixedWidthText, Rect, Style, Ui};
//!
//! // A painter that only counts what reaches it.
//! struct Count(usize);
//! impl threers::ui::TextMeasure for Count {
//!     fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
//!         FixedWidthText(7.0).measure(text, style)
//!     }
//! }
//! impl threers::ui::Painter for Count {
//!     fn fill_rect(&mut self, _: Rect, _: threers::ui::Rgba) {
//!         self.0 += 1;
//!     }
//!     fn text(&mut self, _: Rect, _: &str, _: &threers::ui::TextStyle) {}
//!     fn path(&mut self, _: Rect, _: &[String], _: [f32; 4], _: Option<f32>, _: bool, _: threers::ui::Rgba) {}
//! }
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! for x in [0.0, 500.0] {
//!     ui.add_box(root, Style {
//!         position: threers::ui::Position::Absolute,
//!         left: threers::ui::Length::Px(x),
//!         width: threers::ui::Length::Px(50.0),
//!         height: threers::ui::Length::Px(50.0),
//!         background: Some([1.0, 0.0, 0.0, 1.0]),
//!         ..Style::default()
//!     });
//! }
//! ui.layout(800.0, 600.0, &mut FixedWidthText(7.0));
//! let mut counter = Count(0);
//! let view = Rect { x: 480.0, y: 0.0, width: 100.0, height: 100.0 };
//! ui.paint(&mut Culled::new(&mut counter, view));
//! assert_eq!(counter.0, 1); // only the box that reaches the rectangle
//! ```

use super::{Painter, Rect, Rgba, TextMeasure, TextStyle};

/// A painter that leaves out what lies away from `view`. See the [module documentation](self).
pub struct Culled<'a> {
    inner: &'a mut dyn Painter,
    view: Rect,
}

/// Room around the rectangle for ink that spills past its box: a bold line is drawn twice, half a pixel apart.
const SPILL: f32 = 2.0;

impl<'a> Culled<'a> {
    pub fn new(inner: &'a mut dyn Painter, view: Rect) -> Self {
        Self { inner, view }
    }

    /// Whether anything drawn in `r` can reach the rectangle.
    fn visible(&self, r: Rect) -> bool {
        let v = self.view;
        r.x < v.x + v.width + SPILL && r.x + r.width > v.x - SPILL && r.y < v.y + v.height + SPILL && r.y + r.height > v.y - SPILL
    }
}

impl TextMeasure for Culled<'_> {
    fn measure(&mut self, text: &str, style: &super::Style) -> (f32, f32) {
        self.inner.measure(text, style)
    }
}

impl Painter for Culled<'_> {
    fn fill_rect(&mut self, rect: Rect, color: Rgba) {
        if self.visible(rect) {
            self.inner.fill_rect(rect, color);
        }
    }

    fn fill_rounded_rect(&mut self, rect: Rect, rx: f32, ry: f32, color: Rgba) {
        if self.visible(rect) {
            self.inner.fill_rounded_rect(rect, rx, ry, color);
        }
    }

    fn fill_linear_gradient(&mut self, rect: Rect, stops: &[(f32, Rgba)], blur: f32) {
        if self.visible(rect) {
            self.inner.fill_linear_gradient(rect, stops, blur);
        }
    }

    fn text(&mut self, line: Rect, text: &str, style: &TextStyle) {
        if self.visible(line) {
            self.inner.text(line, text, style);
        }
    }

    fn path(&mut self, rect: Rect, d: &[String], view_box: [f32; 4], stroke: Option<f32>, fill: bool, color: Rgba) {
        if self.visible(rect) {
            self.inner.path(rect, d, view_box, stroke, fill, color);
        }
    }

    fn push_clip(&mut self, rect: Rect) {
        self.inner.push_clip(rect);
    }

    fn push_clip_rounded(&mut self, rect: Rect, rx: f32, ry: f32) {
        self.inner.push_clip_rounded(rect, rx, ry);
    }

    fn pop_clip(&mut self) {
        self.inner.pop_clip();
    }

    fn set_subpixel(&mut self, on: bool) {
        self.inner.set_subpixel(on);
    }
}
