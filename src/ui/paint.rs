//! Painting a laid-out [`Ui`]: backgrounds, borders, text and paths, in tree
//! order, through a [`Painter`].

use super::layout::TextMeasure;
use super::style::{Display, FontWeight, Rgba, Style};
use super::tree::{NodeId, NodeKind, Rect, Ui};
use crate::math::Color;
use crate::overlay::Overlay;

/// The font of a text run, resolved from its node's style.
#[derive(Debug, Clone, PartialEq)]
pub struct TextStyle {
    pub font_family: String,
    pub font_size: f32,
    pub font_weight: FontWeight,
    /// Non-premultiplied sRGB, alpha already multiplied by the opacity of the
    /// node and its ancestors.
    pub color: Rgba,
    /// Struck through (`text-decoration-line: line-through`).
    pub line_through: bool,
    /// Turned a quarter clockwise (`writing-mode: vertical-lr`); `line` is
    /// then the run's box, taller than wide.
    pub vertical: bool,
    pub italic: bool,
}

impl TextStyle {
    pub fn of(style: &Style) -> Self {
        Self {
            font_family: style.font_family.clone(),
            font_size: style.font_size,
            font_weight: style.font_weight,
            color: style.color,
            line_through: style.line_through,
            vertical: style.vertical,
            italic: style.italic,
        }
    }

    /// The weight as a CSS number (`normal` 400, `bold` 700).
    pub fn weight(&self) -> u16 {
        match self.font_weight {
            FontWeight::Normal => 400,
            FontWeight::Bold => 700,
            FontWeight::Weight(w) => w,
        }
    }

    /// A CSS `font` shorthand, as a 2D canvas takes it.
    pub fn css_font(&self) -> String {
        let style = if self.italic { "italic " } else { "" };
        format!("{style}{} {}px {}", self.weight(), self.font_size, self.font_family)
    }
}

/// Where a [`Ui`] is drawn. Colours are non-premultiplied sRGB in `0..=1`;
/// coordinates are pixels from the top-left.
///
/// A painter also measures text, so that layout and painting agree on every
/// string's width.
pub trait Painter: TextMeasure {
    fn fill_rect(&mut self, rect: Rect, color: Rgba);

    /// A rectangle with elliptical corners of radii `rx` × `ry`
    /// (`border-radius`). Painters without curves may fill the plain rectangle.
    fn fill_rounded_rect(&mut self, rect: Rect, rx: f32, ry: f32, color: Rgba) {
        let _ = (rx, ry);
        self.fill_rect(rect, color);
    }

    /// A left-to-right linear gradient through `stops` (fractions of the
    /// width, ascending), blurred by `blur` px (Gaussian standard deviation).
    /// By default: bands of interpolated colour, unblurred.
    fn fill_linear_gradient(&mut self, rect: Rect, stops: &[(f32, Rgba)], blur: f32) {
        let _ = blur;
        for pair in stops.windows(2) {
            let ((t0, c0), (t1, c1)) = (pair[0], pair[1]);
            if t1 <= t0 {
                continue;
            }
            let steps = if c0 == c1 { 1 } else { 16 };
            for k in 0..steps {
                let (a, b) = (k as f32 / steps as f32, (k + 1) as f32 / steps as f32);
                let mid = (a + b) / 2.0;
                let c = std::array::from_fn(|i| c0[i] + (c1[i] - c0[i]) * mid);
                let x0 = rect.x + rect.width * (t0 + (t1 - t0) * a);
                let x1 = rect.x + rect.width * (t0 + (t1 - t0) * b);
                self.fill_rect(Rect { x: x0, y: rect.y, width: x1 - x0, height: rect.height }, c);
            }
        }
    }

    /// One line of text in the line box `line` (its top-left and height):
    /// vertically centred on the line box, as CSS places a glyph run.
    fn text(&mut self, line: Rect, text: &str, style: &TextStyle);

    /// SVG path data in `view_box` scaled to `rect`, stroked `stroke` pixels
    /// wide in view-box units and/or filled (non-zero), in `color`.
    fn path(&mut self, rect: Rect, d: &[String], view_box: [f32; 4], stroke: Option<f32>, fill: bool, color: Rgba);

    /// Restricts painting to `rect` until the matching
    /// [`pop_clip`](Self::pop_clip). Painters that cannot clip may ignore it.
    fn push_clip(&mut self, _rect: Rect) {}

    /// Like [`push_clip`](Self::push_clip), to a box with elliptical
    /// corners (`overflow: hidden` with `border-radius`). By default, the
    /// plain box.
    fn push_clip_rounded(&mut self, rect: Rect, rx: f32, ry: f32) {
        let _ = (rx, ry);
        self.push_clip(rect);
    }

    fn pop_clip(&mut self) {}

    /// Inside (`true`) or out of a [`Style::subpixel`] subtree: boxes there
    /// keep their fractional edges. Painters that always snap may ignore it.
    fn set_subpixel(&mut self, _on: bool) {}
}

impl Ui {
    /// Paints the tree as last laid out: each box's background, then its
    /// borders, then its text or path, then its children in order.
    pub fn paint(&self, painter: &mut dyn Painter) {
        self.paint_node(self.root(), painter, 1.0, false);
    }

    fn paint_node(&self, id: NodeId, painter: &mut dyn Painter, opacity: f32, subpixel: bool) {
        let node = &self.nodes[id.0];
        let s = &node.style;
        if s.display == Display::None {
            return;
        }

        let opacity = opacity * s.opacity;
        if opacity <= 0.0 {
            return;
        }
        let enter = s.subpixel && !subpixel;
        if enter {
            painter.set_subpixel(true);
        }
        let subpixel = subpixel || s.subpixel;
        let fade = |c: Rgba| [c[0], c[1], c[2], c[3] * opacity];
        let r = node.rect;
        // `background-clip: content-box`.
        let bg_rect = if s.background_clip_content {
            Rect {
                x: r.x + s.border.left + s.padding.left,
                y: r.y + s.border.top + s.padding.top,
                width: (r.width - s.border.horizontal() - s.padding.horizontal()).max(0.0),
                height: (r.height - s.border.vertical() - s.padding.vertical()).max(0.0),
            }
        } else {
            r
        };
        let b = s.border;
        let uniform = b.top > 0.0 && b.top == b.right && b.top == b.bottom && b.top == b.left;
        let (orx, ory) = (
            s.border_radius.resolve(Some(r.width)).unwrap_or(0.0).min(r.width / 2.0),
            s.border_radius.resolve(Some(r.height)).unwrap_or(0.0).min(r.height / 2.0),
        );
        // A rounded box with a uniform border over a background: the border
        // colour over the whole rounded box, the background inset from it
        // with the inner radii (`radius − border`).
        let rounded_border = uniform && orx > 0.0 && ory > 0.0 && s.border_color[3] > 0.0 && s.background.is_some() && !s.background_clip_content;
        if rounded_border {
            painter.fill_rounded_rect(r, orx, ory, fade(s.border_color));
            let inner = Rect { x: r.x + b.left, y: r.y + b.top, width: (r.width - b.horizontal()).max(0.0), height: (r.height - b.vertical()).max(0.0) };
            let (irx, iry) = ((orx - b.left).max(0.0), (ory - b.top).max(0.0));
            let bg = fade(s.background.unwrap());
            if irx > 0.0 && iry > 0.0 {
                painter.fill_rounded_rect(inner, irx, iry, bg);
            } else {
                painter.fill_rect(inner, bg);
            }
        } else if let Some(bg) = s.background {
            let r = bg_rect;
            let rx = s.border_radius.resolve(Some(r.width)).unwrap_or(0.0).min(r.width / 2.0);
            let ry = s.border_radius.resolve(Some(r.height)).unwrap_or(0.0).min(r.height / 2.0);
            if rx > 0.0 && ry > 0.0 {
                painter.fill_rounded_rect(r, rx, ry, fade(bg));
            } else {
                painter.fill_rect(r, fade(bg));
            }
        }
        if !s.background_gradient.is_empty() {
            let stops: Vec<(f32, Rgba)> = s.background_gradient.iter().map(|&(t, c)| (t, fade(c))).collect();
            painter.fill_linear_gradient(r, &stops, s.blur);
        }
        if !rounded_border && s.border_color[3] > 0.0 && (b.top > 0.0 || b.right > 0.0 || b.bottom > 0.0 || b.left > 0.0) {
            let c = fade(s.border_color);
            let inner_h = (r.height - b.top - b.bottom).max(0.0);
            for side in [
                Rect { x: r.x, y: r.y, width: r.width, height: b.top },
                Rect { x: r.x, y: r.y + r.height - b.bottom, width: r.width, height: b.bottom },
                Rect { x: r.x, y: r.y + b.top, width: b.left, height: inner_h },
                Rect { x: r.x + r.width - b.right, y: r.y + b.top, width: b.right, height: inner_h },
            ] {
                if side.width > 0.0 && side.height > 0.0 {
                    painter.fill_rect(side, c);
                }
            }
        }
        let content = Rect {
            x: r.x + b.left + s.padding.left,
            y: r.y + b.top + s.padding.top,
            width: (r.width - s.border.horizontal() - s.padding.horizontal()).max(0.0),
            height: (r.height - s.border.vertical() - s.padding.vertical()).max(0.0),
        };
        match &node.kind {
            NodeKind::Box => {}
            NodeKind::Text(t) => {
                let mut style = TextStyle::of(s);
                style.color = fade(style.color);
                if node.lines.len() > 1 {
                    // A wrapped run: one line box each.
                    let h = content.height / node.lines.len() as f32;
                    for (i, line) in node.lines.iter().enumerate() {
                        let r = Rect { y: content.y + i as f32 * h, height: h, ..content };
                        painter.text(r, line.trim_end_matches([' ', '\u{200b}']), &style);
                    }
                } else {
                    painter.text(content, t, &style);
                }
            }
            NodeKind::Path { d, view_box, stroke, fill } => painter.path(content, d, *view_box, *stroke, *fill, fade(s.color)),
        }
        if s.clip {
            let rx = s.border_radius.resolve(Some(r.width)).unwrap_or(0.0).min(r.width / 2.0);
            let ry = s.border_radius.resolve(Some(r.height)).unwrap_or(0.0).min(r.height / 2.0);
            if rx > 0.0 && ry > 0.0 {
                painter.push_clip_rounded(r, rx, ry);
            } else {
                painter.push_clip(r);
            }
        }
        for c in self.paint_order(id) {
            self.paint_node(c, painter, opacity, subpixel);
        }
        if s.clip {
            painter.pop_clip();
        }
        if enter {
            painter.set_subpixel(false);
        }
    }
}

/// An SVG `<circle>` as path data, in the one form [`parse_circle`] reads
/// back: two half arcs from the rightmost point.
///
/// A painter that can draw a true oval should, for these: browsers stroke a
/// `<circle>` as an oval, which rasterizes differently from the same circle
/// given as arcs.
pub fn circle_path(cx: f32, cy: f32, r: f32) -> String {
    format!("M{} {}A{r} {r} 0 1 1 {} {}A{r} {r} 0 1 1 {} {}Z", cx + r, cy, cx - r, cy, cx + r, cy)
}

/// The centre and radius of path data made by [`circle_path`].
pub fn parse_circle(d: &str) -> Option<(f32, f32, f32)> {
    let rest = d.strip_prefix('M')?.strip_suffix('Z')?;
    let mut parts = rest.split('A');
    let start: Vec<f32> = parts.next()?.split(' ').map(str::parse).collect::<Result<_, _>>().ok()?;
    let a: Vec<f32> = parts.next()?.split(' ').map(str::parse).collect::<Result<_, _>>().ok()?;
    let b: Vec<f32> = parts.next()?.split(' ').map(str::parse).collect::<Result<_, _>>().ok()?;
    if parts.next().is_some() || start.len() != 2 || a.len() != 7 || b.len() != 7 {
        return None;
    }
    let r = a[0];
    let (cx, cy) = (start[0] - r, start[1]);
    (a[1] == r && b[0] == r && b[1] == r && a[5] == cx - r && a[6] == cy && b[5] == start[0] && b[6] == cy).then_some((cx, cy, r))
}

/// sRGB `0..=1` to the linear [`Color`] the overlay mixes in.
fn linear(c: Rgba) -> Color {
    let f = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
    Color::new(f(c[0]), f(c[1]), f(c[2]))
}

/// A [`Painter`] onto threers' CPU [`Overlay`], with the caption font.
///
/// Text is the overlay's face at the requested size (the family is not
/// looked up); bold is drawn twice, half a pixel apart. Clipping is not
/// supported.
pub struct OverlayPainter<'a> {
    pub overlay: &'a mut Overlay,
    /// Overlay pixels per UI pixel (the display's scale factor).
    pub scale: f32,
}

impl<'a> OverlayPainter<'a> {
    pub fn new(overlay: &'a mut Overlay) -> Self {
        Self { overlay, scale: 1.0 }
    }

    /// Paints a UI laid out in logical pixels onto an overlay `scale` times
    /// as dense.
    pub fn with_scale(overlay: &'a mut Overlay, scale: f32) -> Self {
        Self { overlay, scale }
    }

    fn px(&self, r: Rect) -> Rect {
        let k = self.scale;
        Rect { x: r.x * k, y: r.y * k, width: r.width * k, height: r.height * k }
    }
}

impl TextMeasure for OverlayPainter<'_> {
    fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
        self.overlay.text_size(style.font_size * self.scale);
        let (w, _) = self.overlay.measure_text(text);
        let bold = if TextStyle::of(style).weight() >= 600 { 0.5 } else { 0.0 };
        (w / self.scale + bold, style.line_height.unwrap_or(style.font_size * 1.2))
    }
}

impl Painter for OverlayPainter<'_> {
    fn fill_rect(&mut self, rect: Rect, color: Rgba) {
        let r = self.px(rect);
        let o = &mut *self.overlay;
        o.no_stroke();
        o.fill_alpha(linear(color), color[3]);
        o.rect(r.x, r.y, r.width, r.height);
    }

    fn fill_rounded_rect(&mut self, rect: Rect, rx: f32, ry: f32, color: Rgba) {
        let (r, k) = (self.px(rect), self.scale);
        let o = &mut *self.overlay;
        o.no_stroke();
        o.fill_alpha(linear(color), color[3]);
        o.rounded_rect(r.x, r.y, r.width, r.height, rx.min(ry) * k);
    }

    fn text(&mut self, line: Rect, text: &str, style: &TextStyle) {
        let (line, k) = (self.px(line), self.scale);
        let o = &mut *self.overlay;
        o.no_stroke();
        o.fill_alpha(linear(style.color), style.color[3]);
        o.text_size(style.font_size * k);
        let (w, h) = o.measure_text(text);
        let y = line.y + (line.height - h) / 2.0;
        o.text(line.x, y, text);
        if style.weight() >= 600 {
            o.text(line.x + 0.5 * k, y, text);
        }
        if style.line_through {
            let mid = (line.y + line.height / 2.0).floor();
            o.rect(line.x, mid, w, k.max(1.0));
        }
    }

    fn path(&mut self, rect: Rect, d: &[String], view_box: [f32; 4], stroke: Option<f32>, fill: bool, color: Rgba) {
        let rect = self.px(rect);
        let [vx, vy, vw, vh] = view_box;
        let (sx, sy) = (rect.width / vw.max(1e-6), rect.height / vh.max(1e-6));
        let o = &mut *self.overlay;
        for data in d {
            let Ok(subpaths) = crate::curves::parse_svg_subpaths(data) else { continue };
            for sub in subpaths {
                let closed = sub.curve_path.auto_close;
                let points: Vec<_> = sub
                    .get_points(12)
                    .into_iter()
                    .map(|p| crate::math::Vector2::new(rect.x + (p.x - vx) * sx, rect.y + (p.y - vy) * sy))
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

#[cfg(test)]
mod tests {
    use super::super::*;

    #[derive(Default)]
    struct Record(Vec<String>);

    impl TextMeasure for Record {
        fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
            FixedWidthText(7.0).measure(text, style)
        }
    }

    impl Painter for Record {
        fn fill_rect(&mut self, r: Rect, c: Rgba) {
            self.0.push(format!("rect {} {} {} {} {:?}", r.x, r.y, r.width, r.height, c));
        }
        fn fill_rounded_rect(&mut self, r: Rect, rx: f32, ry: f32, c: Rgba) {
            self.0.push(format!("round {} {} {} {} {rx} {ry} {:?}", r.x, r.y, r.width, r.height, c));
        }
        fn text(&mut self, r: Rect, t: &str, s: &TextStyle) {
            self.0.push(format!("text {} {} {t} {}", r.x, r.y, s.css_font()));
        }
        fn path(&mut self, r: Rect, d: &[String], _: [f32; 4], _: Option<f32>, _: bool, _: Rgba) {
            self.0.push(format!("path {} {} {}", r.x, r.y, d.len()));
        }
    }

    #[test]
    fn paints_background_borders_then_content_in_tree_order() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, Style::flex(FlexDirection::Row));
        let item = ui.add_box(
            root,
            Style {
                background: Some([0.0, 0.0, 0.0, 1.0]),
                border: Edges::all(1.0),
                border_color: [0.5, 0.5, 0.5, 1.0],
                padding: Edges::symmetric(0.0, 2.0),
                font_size: 13.0,
                opacity: 0.5,
                ..Style::default()
            },
        );
        ui.add_text(item, "ab");
        let mut rec = Record::default();
        ui.layout(100.0, 20.0, &mut rec);
        ui.paint(&mut rec);
        assert_eq!(rec.0[0], "rect 0 0 20 20 [0.0, 0.0, 0.0, 0.5]");
        assert_eq!(rec.0.len(), 6, "{:?}", rec.0);
        assert_eq!(rec.0[5], "text 3 1 ab 400 13px sans-serif");
    }

    #[test]
    fn rounded_borders_are_the_box_then_the_background_inset() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, Style::flex(FlexDirection::Row));
        ui.add_box(
            root,
            Style {
                width: Length::Px(13.0),
                height: Length::Px(13.0),
                box_sizing: BoxSizing::BorderBox,
                background: Some([1.0; 4]),
                border: Edges::all(1.0),
                border_color: [0.5, 0.5, 0.5, 1.0],
                border_radius: Length::Px(2.0),
                ..Style::default()
            },
        );
        let mut rec = Record::default();
        ui.layout(100.0, 20.0, &mut rec);
        ui.paint(&mut rec);
        assert_eq!(rec.0, ["round 0 0 13 13 2 2 [0.5, 0.5, 0.5, 1.0]", "round 1 1 11 11 1 1 [1.0, 1.0, 1.0, 1.0]"]);
    }

    #[test]
    fn circles_round_trip_through_path_data() {
        let d = super::super::circle_path(9.0, 6.0, 2.0);
        assert_eq!(d, "M11 6A2 2 0 1 1 7 6A2 2 0 1 1 11 6Z");
        assert_eq!(super::super::parse_circle(&d), Some((9.0, 6.0, 2.0)));
        assert_eq!(super::super::parse_circle("M11 6A2 2 0 1 1 7 6Z"), None);
        crate::curves::parse_svg_subpaths(&d).unwrap();
    }

    #[test]
    fn overlay_painter_fills_pixels() {
        let mut overlay = crate::overlay::Overlay::new(40, 30);
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, Style { background: Some([34.0 / 255.0, 34.0 / 255.0, 34.0 / 255.0, 1.0]), ..Style::default() });
        let icon = ui.add_path(
            root,
            Style { width: Length::Px(16.0), height: Length::Px(16.0), ..Style::default() },
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
