//! The retained UI — the [`threers-ui`](https://docs.rs/threers-ui) crate,
//! re-exported — and [`OverlayPainter`], which paints it onto threers' own
//! CPU [`Overlay`] for [`Renderer::draw_overlay`](crate::Renderer::draw_overlay)
//! to composite over a frame.
//!
//! For large or high-density UIs, [`UiPass`](crate::renderer::UiPass) draws
//! the same tree on the GPU from its [`Ui::scene`] instead, in one instanced
//! draw call and without uploading an overlay.
//!
//! [`paint_to_rgba`] paints a UI into an image on the CPU, for screenshots
//! and golden-image tests without a GPU.
//!
//! Everything but the painter lives in `threers-ui`, which knows nothing of
//! the renderer: depend on it directly for the layout, controls and
//! accessibility tree alone, or to paint through a [`Painter`] of your own.
//!
//! ```
//! use threers::overlay::Overlay;
//! use threers::ui::{OverlayPainter, Ui};
//! use threers::style;
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! ui.set_style(root, style! { background: #222; });
//! let mut overlay = Overlay::new(40, 30);
//! let mut painter = OverlayPainter::new(&mut overlay);
//! ui.layout(40.0, 30.0, &mut painter);
//! ui.paint(&mut painter);
//! assert_eq!(&overlay.pixels()[..4], &[34, 34, 34, 255]);
//! ```

pub use threers_ui::*;

// It draws with the headless GPU renderer, which wasm lacks.
#[cfg(all(feature = "render", not(target_arch = "wasm32")))]
pub mod model_view;

use crate::math::Color;
use crate::overlay::Overlay;

/// `ui` as last laid out, painted on the CPU into a `width × height` image
/// at `scale` device pixels per UI pixel, over transparency: RGBA8,
/// sRGB-encoded, straight alpha, rows from the top. For screenshots and
/// golden-image tests without a GPU; [`UiPass::render_to_rgba`](crate::renderer::UiPass)
/// is the GPU's.
pub fn paint_to_rgba(ui: &Ui, scale: f32, width: u32, height: u32) -> Vec<u8> {
    let mut overlay = Overlay::new(width.max(1), height.max(1));
    ui.paint(&mut OverlayPainter::with_scale(&mut overlay, scale));
    overlay.pixels().to_vec()
}

/// sRGB `0..=1` to the linear [`Color`] the overlay mixes in.
pub(crate) fn linear(c: Rgba) -> Color {
    let f = |v: f32| {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    Color::new(f(c[0]), f(c[1]), f(c[2]))
}

/// A [`Painter`] onto threers' CPU [`Overlay`], with the caption font.
///
/// Text is the overlay's face at the requested size, weight and italic (the
/// family is not looked up), synthesized where the font lacks them; only
/// the bitmap face, which has no outlines to embolden, fakes bold by drawing
/// twice, half a pixel apart.
///
/// Clipping (`overflow: hidden`, scrolling boxes) is to rectangles:
/// rectangles are cut exactly, a rounded rectangle keeps its corners only
/// when wholly inside, paths are drawn whole unless wholly outside, and text
/// loses the characters that do not fit (whole characters, not pixels).
pub struct OverlayPainter<'a> {
    pub overlay: &'a mut Overlay,
    /// Overlay pixels per UI pixel (the display's scale factor).
    pub scale: f32,
    /// The clip stack, in overlay pixels: each entry already intersected
    /// with the one below.
    clips: Vec<Rect>,
}

impl<'a> OverlayPainter<'a> {
    pub fn new(overlay: &'a mut Overlay) -> Self {
        Self {
            overlay,
            scale: 1.0,
            clips: Vec::new(),
        }
    }

    /// Paints a UI laid out in logical pixels onto an overlay `scale` times
    /// as dense.
    pub fn with_scale(overlay: &'a mut Overlay, scale: f32) -> Self {
        Self {
            overlay,
            scale,
            clips: Vec::new(),
        }
    }

    fn px(&self, r: Rect) -> Rect {
        let k = self.scale;
        Rect {
            x: r.x * k,
            y: r.y * k,
            width: r.width * k,
            height: r.height * k,
        }
    }

    /// `r` cut to the current clip; `None` when nothing is left.
    fn clipped(&self, r: Rect) -> Option<Rect> {
        let Some(c) = self.clips.last() else {
            return Some(r);
        };
        intersect(r, *c)
    }
}

fn intersect(a: Rect, b: Rect) -> Option<Rect> {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.width).min(b.x + b.width);
    let y1 = (a.y + a.height).min(b.y + b.height);
    (x1 > x0 && y1 > y0).then_some(Rect {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    })
}

impl OverlayPainter<'_> {
    fn draw_image(&mut self, rect: Rect, image: &image::Image, opacity: f32) {
        let r = self.px(rect);
        let clip = self
            .clips
            .last()
            .map(|c| (c.x, c.y, c.x + c.width, c.y + c.height));
        self.overlay.image(
            r.x,
            r.y,
            r.width,
            r.height,
            &image.rgba,
            image.width,
            image.height,
            opacity,
            clip,
        );
    }
}

impl TextMeasure for OverlayPainter<'_> {
    fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
        self.overlay.text_size(style.font_size * self.scale);
        let weight = TextStyle::of(style).weight();
        let real = self.overlay.text_weight(weight);
        self.overlay.text_italic(style.italic);
        // The caption layout trims a line's ends, as subtitles want; a UI
        // run ending in a space (before the next run) needs that space. So
        // such a run is measured between two bars, and the bars taken off.
        let w = if text.starts_with(char::is_whitespace) || text.ends_with(char::is_whitespace) {
            let bar = self.overlay.measure_line("|").0;
            self.overlay.measure_line(&format!("|{text}|")).0 - 2.0 * bar
        } else {
            self.overlay.measure_line(text).0
        };
        let bold = if weight >= 600 && !real { 0.5 } else { 0.0 };
        (
            w / self.scale + bold,
            style.line_height.unwrap_or(style.font_size * 1.2),
        )
    }
}

impl Painter for OverlayPainter<'_> {
    /// A radial, conic or turned linear gradient, pixel by pixel: each the
    /// colour at its centre, its edges covered by how far the box reaches
    /// into them, cut to the clip.
    fn fill_gradient(
        &mut self,
        rect: Rect,
        shape: crate::ui::GradientShape,
        stops: &[(f32, Rgba)],
        blur: f32,
    ) {
        use threers_math::srgb::{linear_to_srgb, srgb_to_linear, srgb_to_linear_f32};
        if shape.is_across() || stops.is_empty() {
            return self.fill_linear_gradient(rect, stops, blur);
        }
        let r = self.px(rect);
        let (mut x0, mut y0, mut x1, mut y1) = (r.x, r.y, r.x + r.width, r.y + r.height);
        if let Some(c) = self.clips.last() {
            x0 = x0.max(c.x);
            y0 = y0.max(c.y);
            x1 = x1.min(c.x + c.width);
            y1 = y1.min(c.y + c.height);
        }
        let (cw, ch) = self.overlay.size();
        let (bx0, by0) = (x0.floor().max(0.0) as i32, y0.floor().max(0.0) as i32);
        let (bx1, by1) = (
            x1.ceil().min(cw as f32) as i32,
            y1.ceil().min(ch as f32) as i32,
        );
        if bx1 <= bx0 || by1 <= by0 {
            return;
        }
        let (bw, bh) = ((bx1 - bx0) as usize, (by1 - by0) as usize);
        let mut buf = vec![0u8; bw * bh * 4];
        let src = self.overlay.pixels();
        for row in 0..bh {
            let from = ((by0 as usize + row) * cw as usize + bx0 as usize) * 4;
            buf[row * bw * 4..(row + 1) * bw * 4].copy_from_slice(&src[from..from + bw * 4]);
        }
        for row in 0..bh {
            let y = (by0 + row as i32) as f32;
            let cover_y = (y1.min(y + 1.0) - y0.max(y)).clamp(0.0, 1.0);
            for col in 0..bw {
                let x = (bx0 + col as i32) as f32;
                let cover = cover_y * (x1.min(x + 1.0) - x0.max(x)).clamp(0.0, 1.0);
                if cover <= 0.0 {
                    continue;
                }
                let c = crate::ui::gradient_at(stops, shape.t(r, x + 0.5, y + 0.5));
                let sa = c[3] * cover;
                if sa <= 0.0 {
                    continue;
                }
                let px = &mut buf[(row * bw + col) * 4..][..4];
                let da = px[3] as f32 / 255.0;
                let out_a = sa + da * (1.0 - sa);
                for i in 0..3 {
                    let d = srgb_to_linear(px[i]);
                    px[i] = linear_to_srgb(
                        (srgb_to_linear_f32(c[i]) * sa + d * da * (1.0 - sa)) / out_a.max(1e-6),
                    );
                }
                px[3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        self.overlay
            .put_image_data(bx0, by0, bw as u32, bh as u32, &buf);
    }

    /// From its distance field, pixel by pixel: each pixel covered by how
    /// far inside the stroke its centre is, cut to the clip, blended in
    /// linear light.
    fn stroke_arc(&mut self, arc: crate::ui::ArcShape, width: f32, color: Rgba) {
        use threers_math::srgb::{linear_to_srgb, srgb_to_linear, srgb_to_linear_f32};
        let k = self.scale;
        let b = self.px(arc.bounds(width / 2.0 + 1.0));
        let (mut x0, mut y0, mut x1, mut y1) = (b.x, b.y, b.x + b.width, b.y + b.height);
        if let Some(c) = self.clips.last() {
            x0 = x0.max(c.x);
            y0 = y0.max(c.y);
            x1 = x1.min(c.x + c.width);
            y1 = y1.min(c.y + c.height);
        }
        let (cw, ch) = self.overlay.size();
        let (bx0, by0) = (x0.floor().max(0.0) as i32, y0.floor().max(0.0) as i32);
        let (bx1, by1) = (
            x1.ceil().min(cw as f32) as i32,
            y1.ceil().min(ch as f32) as i32,
        );
        if bx1 <= bx0 || by1 <= by0 || color[3] <= 0.0 {
            return;
        }
        // In device pixels.
        let dev = crate::ui::ArcShape {
            cx: arc.cx * k,
            cy: arc.cy * k,
            radius: arc.radius * k,
            ..arc
        };
        let w = width * k;
        let (bw, bh) = ((bx1 - bx0) as usize, (by1 - by0) as usize);
        let mut buf = vec![0u8; bw * bh * 4];
        let src = self.overlay.pixels();
        for row in 0..bh {
            let from = ((by0 as usize + row) * cw as usize + bx0 as usize) * 4;
            buf[row * bw * 4..(row + 1) * bw * 4].copy_from_slice(&src[from..from + bw * 4]);
        }
        let lin = [0, 1, 2].map(|i| srgb_to_linear_f32(color[i]));
        for row in 0..bh {
            let y = (by0 + row as i32) as f32 + 0.5;
            for col in 0..bw {
                let x = (bx0 + col as i32) as f32 + 0.5;
                let cover = (0.5 - dev.distance(w, x, y)).clamp(0.0, 1.0);
                let sa = color[3] * cover;
                if sa <= 0.0 {
                    continue;
                }
                let px = &mut buf[(row * bw + col) * 4..][..4];
                let da = px[3] as f32 / 255.0;
                let out_a = sa + da * (1.0 - sa);
                for i in 0..3 {
                    let d = srgb_to_linear(px[i]);
                    px[i] = linear_to_srgb((lin[i] * sa + d * da * (1.0 - sa)) / out_a.max(1e-6));
                }
                px[3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        self.overlay
            .put_image_data(bx0, by0, bw as u32, bh as u32, &buf);
    }

    /// Each device pixel the colour at its centre (stops mixed as encoded,
    /// as CSS mixes them; the end colours carried out to the edges), its
    /// top and bottom rows covered by how far the box reaches into them,
    /// cut to the clip: as the GPU pass draws it, smooth where bands of flat
    /// colour would step. Blended here, pixel by pixel.
    fn fill_linear_gradient(&mut self, rect: Rect, stops: &[(f32, Rgba)], _blur: f32) {
        use threers_math::srgb::{linear_to_srgb, srgb_to_linear, srgb_to_linear_f32};
        if stops.is_empty() {
            return;
        }
        let r = self.px(rect);
        if r.width <= 0.0 || r.height <= 0.0 {
            return;
        }
        let at = |t: f32| -> Rgba {
            let (first, last) = (stops[0], stops[stops.len() - 1]);
            if t <= first.0 {
                return first.1;
            }
            if t >= last.0 {
                return last.1;
            }
            let i = stops
                .windows(2)
                .position(|w| t >= w[0].0 && t < w[1].0)
                .unwrap_or(0);
            let ((t0, c0), (t1, c1)) = (stops[i], stops[i + 1]);
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            std::array::from_fn(|c| c0[c] + (c1[c] - c0[c]) * f)
        };
        let (mut x0, mut y0, mut x1, mut y1) = (r.x, r.y, r.x + r.width, r.y + r.height);
        if let Some(c) = self.clips.last() {
            x0 = x0.max(c.x);
            y0 = y0.max(c.y);
            x1 = x1.min(c.x + c.width);
            y1 = y1.min(c.y + c.height);
        }
        let (cw, ch) = self.overlay.size();
        let (bx0, by0) = (x0.floor().max(0.0) as i32, y0.floor().max(0.0) as i32);
        let (bx1, by1) = (
            x1.ceil().min(cw as f32) as i32,
            y1.ceil().min(ch as f32) as i32,
        );
        if bx1 <= bx0 || by1 <= by0 {
            return;
        }
        let (bw, bh) = ((bx1 - bx0) as usize, (by1 - by0) as usize);
        let mut buf = vec![0u8; bw * bh * 4];
        let src = self.overlay.pixels();
        for row in 0..bh {
            let from = ((by0 as usize + row) * cw as usize + bx0 as usize) * 4;
            buf[row * bw * 4..(row + 1) * bw * 4].copy_from_slice(&src[from..from + bw * 4]);
        }
        for col in 0..bw {
            let centre = (bx0 + col as i32) as f32 + 0.5;
            if centre < x0 || centre >= x1 {
                continue;
            }
            let c = at((centre - r.x) / r.width);
            let lin = [0, 1, 2].map(|i| srgb_to_linear_f32(c[i]));
            for row in 0..bh {
                let y = (by0 + row as i32) as f32;
                let cover = (y1.min(y + 1.0) - y0.max(y)).clamp(0.0, 1.0);
                let sa = c[3] * cover;
                if sa <= 0.0 {
                    continue;
                }
                let px = &mut buf[(row * bw + col) * 4..][..4];
                let da = px[3] as f32 / 255.0;
                let out_a = sa + da * (1.0 - sa);
                for i in 0..3 {
                    let d = srgb_to_linear(px[i]);
                    px[i] = linear_to_srgb((lin[i] * sa + d * da * (1.0 - sa)) / out_a.max(1e-6));
                }
                px[3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        self.overlay
            .put_image_data(bx0, by0, bw as u32, bh as u32, &buf);
    }

    fn fill_rect(&mut self, rect: Rect, color: Rgba) {
        let Some(r) = self.clipped(self.px(rect)) else {
            return;
        };
        let o = &mut *self.overlay;
        o.no_stroke();
        o.fill_alpha(linear(color), color[3]);
        o.rect(r.x, r.y, r.width, r.height);
    }

    fn fill_rounded_rect(&mut self, rect: Rect, rx: f32, ry: f32, color: Rgba) {
        let (r, k) = (self.px(rect), self.scale);
        let Some(c) = self.clipped(r) else { return };
        let o = &mut *self.overlay;
        o.no_stroke();
        o.fill_alpha(linear(color), color[3]);
        // Uncut by the clip, to within float error: the intersection of a
        // box at a fractional place comes out a hair off its own rect, and
        // an exact compare drew every such box (inside any scrolling or
        // `overflow: hidden` parent) with square corners.
        let uncut = (c.x - r.x).abs() < 0.01
            && (c.y - r.y).abs() < 0.01
            && (c.width - r.width).abs() < 0.01
            && (c.height - r.height).abs() < 0.01;
        if uncut {
            o.rounded_rect(r.x, r.y, r.width, r.height, rx.min(ry) * k);
        } else {
            // cut by a scrolling box's edge: the part in view, its corners
            // still round (a plain rect here squared every corner of a box
            // half scrolled out)
            o.rounded_rect_clipped(
                r.x,
                r.y,
                r.width,
                r.height,
                rx.min(ry) * k,
                [c.x, c.y, c.width, c.height],
            );
        }
    }

    fn push_clip(&mut self, rect: Rect) {
        let r = self.px(rect);
        let c = match self.clips.last() {
            Some(top) => intersect(r, *top).unwrap_or(Rect {
                x: r.x,
                y: r.y,
                width: 0.0,
                height: 0.0,
            }),
            None => r,
        };
        self.clips.push(c);
    }

    fn pop_clip(&mut self) {
        self.clips.pop();
    }

    fn text(&mut self, line: Rect, text: &str, style: &TextStyle) {
        let (mut line, k) = (self.px(line), self.scale);
        // On a whole pixel down, a quarter across, as the GPU pass puts it
        // (a baseline between rows greys its horizontal strokes).
        line.y = line.y.round();
        line.x = (line.x * 4.0).round() / 4.0;
        let clip = self.clips.last().copied();
        let o = &mut *self.overlay;
        o.no_stroke();
        o.fill_alpha(linear(style.color), style.color[3]);
        o.text_size(style.font_size * k);
        let real_weight = o.text_weight(style.weight());
        o.text_italic(style.italic);
        // The caption layout drops a line's leading spaces; a run that
        // starts with one (after the run before it) begins that much
        // further on.
        let trimmed = text.trim_start();
        if trimmed.len() < text.len() {
            let lead = &text[..text.len() - trimmed.len()];
            let bar = o.measure_line("|").0;
            line.x += o.measure_line(&format!("|{lead}|")).0 - 2.0 * bar;
        }
        let text = trimmed;
        // Clipped: a line whose middle is outside goes; one running past the
        // right edge loses characters from its end; one starting left of it,
        // from its start (both with half a pixel to spare: the quarter-pixel
        // rounding above can start a line a hair left of its own box).
        let mut owned;
        let mut text = text;
        if let Some(c) = clip {
            let mid = line.y + line.height / 2.0;
            if mid < c.y || mid > c.y + c.height || line.x >= c.x + c.width {
                return;
            }
            let chars: Vec<char> = text.chars().collect();
            let mut start = 0;
            while start < chars.len() && line.x < c.x - 0.5 {
                let w = o.measure_line(&chars[start].to_string()).0;
                line.x += w;
                start += 1;
            }
            let mut end = chars.len();
            while end > start {
                owned = chars[start..end].iter().collect::<String>();
                if line.x + o.measure_line(&owned).0 <= c.x + c.width + 0.5 {
                    break;
                }
                end -= 1;
            }
            owned = chars[start..end].iter().collect::<String>();
            if owned.is_empty() {
                return;
            }
            text = &owned;
        }
        let w = o.measure_line(text).0;
        // One line's height, not the run's: the overlay measures text as a
        // caption, wrapped at a caption's width, so a long UI line came
        // back two lines tall and was centred half a line too high, onto
        // the line above.
        let h = o.measure_line("Hg").1;
        let y = line.y + (line.height - h) / 2.0;
        o.text(line.x, y, text);
        if style.weight() >= 600 && !real_weight {
            o.text(line.x + 0.5 * k, y, text);
        }
        if style.line_through {
            let mid = (line.y + line.height / 2.0).floor();
            o.rect(line.x, mid, w, k.max(1.0));
        }
    }

    fn image(&mut self, rect: Rect, image: &image::ImageRef, opacity: f32) {
        self.draw_image(rect, image, opacity);
    }

    fn shadow(&mut self, rect: Rect, radius: f32, blur: f32, color: Rgba) {
        let r = self.px(rect);
        if self.clipped(r).is_none() && blur <= 0.0 {
            return;
        }
        let k = self.scale;
        let o = &mut *self.overlay;
        o.no_stroke();
        o.fill_alpha(linear(color), color[3]);
        o.shadow(r.x, r.y, r.width, r.height, radius * k, blur * k / 2.0);
    }

    fn underline(&mut self, x: f32, y: f32, width: f32, thickness: f32, color: Rgba, wavy: bool) {
        let k = self.scale;
        let (x, y, width, thickness) = (x * k, y * k, width * k, thickness * k);
        let o = &mut *self.overlay;
        if wavy {
            o.no_fill();
            o.stroke_alpha(linear(color), color[3]);
            o.stroke_weight(thickness);
            o.wavy_line(
                x,
                y,
                x + width,
                thickness * WAVE_AMPLITUDE,
                thickness * WAVE_PERIOD,
            );
        } else {
            o.no_stroke();
            o.fill_alpha(linear(color), color[3]);
            o.rect(x, y - thickness / 2.0, width, thickness);
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
        let rect = self.px(rect);
        if self.clipped(rect).is_none() {
            return;
        }
        let [vx, vy, vw, vh] = view_box;
        let (sx, sy) = (rect.width / vw.max(1e-6), rect.height / vh.max(1e-6));
        let o = &mut *self.overlay;
        for data in d {
            let Ok(subpaths) = crate::curves::parse_svg_subpaths(data) else {
                continue;
            };
            for sub in subpaths {
                let closed = sub.curve_path.auto_close;
                let points: Vec<_> = sub
                    .get_points(curve_divisions(rect))
                    .into_iter()
                    .map(|p| {
                        crate::math::Vector2::new(
                            rect.x + (p.x - vx) * sx,
                            rect.y + (p.y - vy) * sy,
                        )
                    })
                    .collect();
                if points.len() < 2 {
                    continue;
                }
                if fill {
                    o.no_stroke();
                    o.fill_alpha(linear(color), color[3]);
                    o.polygon(&points);
                }
                if let Some(w) = stroke {
                    o.no_fill();
                    o.stroke_alpha(linear(color), color[3]);
                    o.stroke_weight(w * sx.min(sy));
                    let mut pts = points;
                    if closed {
                        pts.push(pts[0]);
                    }
                    o.polyline(&pts);
                }
            }
        }
    }
}

/// How many pieces each curve of a path drawn over `rect` (device pixels)
/// is cut into: a piece every six pixels of the path's larger side, 12 to
/// 96, so a large curve stays round and a small one costs little. (Circular
/// arcs need none: they are drawn from their distance field.)
pub(crate) fn curve_divisions(rect: Rect) -> usize {
    (rect.width.max(rect.height) / 6.0).ceil().clamp(12.0, 96.0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_paths_parse_as_svg() {
        crate::curves::parse_svg_subpaths(&circle_path(9.0, 6.0, 2.0)).unwrap();
    }

    #[test]
    fn spaces_at_either_end_have_width() {
        let mut overlay = Overlay::new(400, 100);
        let mut p = OverlayPainter::new(&mut overlay);
        let st = threers_ui::Style::default();
        let (a, a_sp, sp_a, sp) = (
            p.measure("a", &st).0,
            p.measure("a ", &st).0,
            p.measure(" a", &st).0,
            p.measure(" ", &st).0,
        );
        assert!(sp > 2.0, "a space alone is {sp} wide");
        assert!(
            (a_sp - a - sp).abs() < 0.5,
            "'a ' is {a_sp}, 'a' {a}, ' ' {sp}"
        );
        assert!((sp_a - a - sp).abs() < 0.5, "' a' is {sp_a}");
    }

    /// A sentence split into runs at its spaces lands where it does whole:
    /// the spaces at the runs' ends are measured and drawn, not dropped.
    #[test]
    fn runs_meeting_at_spaces_draw_as_the_whole_line() {
        let draw = |runs: Option<Vec<usize>>| {
            let mut ui = Ui::new();
            let root = ui.root();
            ui.set_style(
                root,
                crate::style! { background: #ffffff; color: #000000; font-size: 16px; },
            );
            let t = ui.add_text(root, "a keyword, a word and a link");
            if let Some(runs) = runs {
                ui.set_text_runs(
                    t,
                    runs.into_iter()
                        .map(crate::ui::rich_text::TextRun::new)
                        .collect(),
                );
            }
            let mut overlay = Overlay::new(300, 40);
            ui.layout(300.0, 40.0, &mut OverlayPainter::new(&mut overlay));
            paint_to_rgba(&ui, 1.0, 300, 40)
        };
        let whole = draw(None);
        // "a " | "keyword," | " a word" | " and a " | "link"
        let split = draw(Some(vec![2, 8, 7, 7, 4]));
        // Ink, column by column: where each column has any.
        let ink = |img: &[u8]| {
            (0..300)
                .map(|x| (0..40).any(|y| img[(y * 300 + x) * 4] < 128))
                .collect::<Vec<bool>>()
        };
        let (a, b) = (ink(&whole), ink(&split));
        let off = a.iter().zip(&b).filter(|(p, q)| p != q).count();
        assert!(off <= 4, "{off} columns inked differently");
    }

    /// A clipping box a fraction of a pixel right of a whole one keeps its
    /// first character: the line starts a hair left of the box, rounded to
    /// a quarter pixel, and that is not a character to drop.
    #[test]
    fn a_clipped_line_keeps_its_first_character() {
        let draw = |clip: bool| {
            let mut ui = Ui::new();
            let root = ui.root();
            ui.set_style(
                root,
                crate::style! { background: #ffffff; color: #000000; font-size: 16px; },
            );
            let b = ui.add_box(root, crate::style! { margin-left: 10.1px; white-space: nowrap; });
            if clip {
                ui.set_style(
                    b,
                    crate::style! { margin-left: 10.1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; },
                );
            }
            ui.add_text(b, "loading");
            let mut overlay = Overlay::new(120, 30);
            ui.layout(120.0, 30.0, &mut OverlayPainter::new(&mut overlay));
            paint_to_rgba(&ui, 1.0, 120, 30)
        };
        let first = |img: &[u8]| (0..120).find(|&x| (0..30).any(|y| img[(y * 120 + x) * 4] < 128));
        assert_eq!(first(&draw(true)), first(&draw(false)));
    }

    #[test]
    fn overlay_painter_fills_pixels() {
        let mut overlay = Overlay::new(40, 30);
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(
            root,
            Style {
                background: Some([34.0 / 255.0, 34.0 / 255.0, 34.0 / 255.0, 1.0]),
                ..Style::default()
            },
        );
        let icon = ui.add_path(
            root,
            Style {
                width: Length::Px(16.0),
                height: Length::Px(16.0),
                ..Style::default()
            },
            vec!["M0 0 L24 0 L24 24 L0 24 Z".into()],
            [0.0, 0.0, 24.0, 24.0],
            None,
            true,
        );
        let mut p = OverlayPainter::new(&mut overlay);
        ui.layout(40.0, 30.0, &mut p);
        ui.paint(&mut p);
        assert_eq!(ui.rect(icon).width, 16.0);
        let px = |x: usize, y: usize| overlay.pixels()[(y * 40 + x) * 4..][..4].to_vec();
        assert_eq!(px(30, 25), vec![34, 34, 34, 255]);
        assert_eq!(px(8, 8), vec![255, 255, 255, 255]);
    }
}
