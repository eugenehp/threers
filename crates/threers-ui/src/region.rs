//! Painting one rectangle of a window.
//!
//! [`Culled`] wraps another [`Painter`] and passes on only what could reach a rectangle. What is passed on is
//! painted exactly where and how a whole-window paint would paint it: not shifted (a shifted position rounds
//! differently near a half pixel) and not clipped (clipping a plain rectangle rounds off the corners of the
//! cut). Paint into an overlay the size of the window, then copy the rectangle out. Pair it with
//! [`damage`](super::damage), which says which rectangles need painting.
//!
//! ```
//! use threers_ui::region::Culled;
//! use threers_ui::{FixedWidthText, Rect, Style, Ui};
//!
//! // A painter that only counts what reaches it.
//! struct Count(usize);
//! impl threers_ui::TextMeasure for Count {
//!     fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
//!         FixedWidthText(7.0).measure(text, style)
//!     }
//! }
//! impl threers_ui::Painter for Count {
//!     fn fill_rect(&mut self, _: Rect, _: threers_ui::Rgba) {
//!         self.0 += 1;
//!     }
//!     fn text(&mut self, _: Rect, _: &str, _: &threers_ui::TextStyle) {}
//!     fn path(&mut self, _: Rect, _: &[String], _: [f32; 4], _: Option<f32>, _: bool, _: threers_ui::Rgba) {}
//! }
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! for x in [0.0, 500.0] {
//!     ui.add_box(root, Style {
//!         position: threers_ui::Position::Absolute,
//!         left: threers_ui::Length::Px(x),
//!         width: threers_ui::Length::Px(50.0),
//!         height: threers_ui::Length::Px(50.0),
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
        r.x < v.x + v.width + SPILL
            && r.x + r.width > v.x - SPILL
            && r.y < v.y + v.height + SPILL
            && r.y + r.height > v.y - SPILL
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

    fn fill_gradient(
        &mut self,
        rect: Rect,
        shape: crate::GradientShape,
        stops: &[(f32, Rgba)],
        blur: f32,
    ) {
        if self.visible(rect) {
            self.inner.fill_gradient(rect, shape, stops, blur);
        }
    }

    fn text(&mut self, line: Rect, text: &str, style: &TextStyle) {
        if self.visible(line) {
            self.inner.text(line, text, style);
        }
    }

    fn path(
        &mut self,
        rect: Rect,
        d: &[String],
        view_box: [f32; 4],
        stroke: Option<f32>,
        fill: bool,
        color: Rgba,
    ) {
        if self.visible(rect) {
            self.inner.path(rect, d, view_box, stroke, fill, color);
        }
    }

    fn stroke_arc(&mut self, arc: crate::paint::ArcShape, width: f32, color: Rgba) {
        if self.visible(arc.bounds(width / 2.0 + 1.0)) {
            self.inner.stroke_arc(arc, width, color);
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

    fn push_transform(&mut self, m: [f32; 6]) {
        self.inner.push_transform(m);
    }

    fn pop_transform(&mut self) {
        self.inner.pop_transform();
    }

    fn push_projective(&mut self, h: [f32; 9], affine: [f32; 6]) {
        self.inner.push_projective(h, affine);
    }

    fn push_blur(&mut self, radius: f32) {
        self.inner.push_blur(radius);
    }

    fn pop_blur(&mut self) {
        self.inner.pop_blur();
    }

    fn push_effect(&mut self, rect: Rect, effect: &crate::shader::Effect, time: f32) {
        self.inner.push_effect(rect, effect, time);
    }

    fn pop_effect(&mut self) {
        self.inner.pop_effect();
    }

    fn set_subpixel(&mut self, on: bool) {
        self.inner.set_subpixel(on);
    }

    fn shadow(&mut self, rect: Rect, radius: f32, blur: f32, color: Rgba) {
        // The blur reaches three deviations past the box.
        let reach = 1.5 * blur;
        let wide = Rect {
            x: rect.x - reach,
            y: rect.y - reach,
            width: rect.width + 2.0 * reach,
            height: rect.height + 2.0 * reach,
        };
        if self.visible(wide) {
            self.inner.shadow(rect, radius, blur, color);
        }
    }

    fn underline(&mut self, x: f32, y: f32, width: f32, thickness: f32, color: Rgba, wavy: bool) {
        let reach = thickness * (crate::paint::WAVE_AMPLITUDE + 1.0);
        if self.visible(Rect {
            x,
            y: y - reach,
            width,
            height: 2.0 * reach,
        }) {
            self.inner.underline(x, y, width, thickness, color, wavy);
        }
    }

    fn surface(&mut self, rect: Rect, id: u64, opacity: f32) {
        if self.visible(rect) {
            self.inner.surface(rect, id, opacity);
        }
    }

    fn backdrop(&mut self, rect: Rect, radius: f32, blur: f32, saturate: f32) {
        if self.visible(rect) {
            self.inner.backdrop(rect, radius, blur, saturate);
        }
    }

    fn image(&mut self, rect: Rect, image: &crate::image::ImageRef, opacity: f32) {
        if self.visible(rect) {
            self.inner.image(rect, image, opacity);
        }
    }
}
