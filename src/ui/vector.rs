//! A UI drawn without a GPU, as vectors: [`VectorPainter`] records what a
//! [`Ui`] paints as an [`SvgDocument`], the drawing every vector exporter
//! reads, so the same interface goes out as SVG ([`ui_to_svg`]), as PDF
//! ([`ui_to_pdf`]) and, drawn on the CPU, as PNG ([`ui_to_png`]). For a
//! report or a print, a screenshot without a window, or a golden image in a
//! test that runs anywhere.
//!
//! Text is measured for layout in the same face it is drawn in
//! ([`DrawingFonts`]; DejaVu Sans unless told otherwise), so lines break where
//! the drawing puts them, and in the PDF it stays text: embedded, selectable
//! and searchable. Boxes, rounded corners, borders, linear and radial
//! gradients, clips (rounded too), transforms, images, icons' path data and
//! underlines are drawn as they are. What a drawing cannot hold is drawn as
//! close as it can be:
//!
//! - bold and italic are made from the regular face (outlined a little
//!   thicker, slanted), as browsers do for a family that has no bold;
//! - a blurred shadow is a stack of rounded rectangles, fading out over the
//!   blur;
//! - a conic gradient is wedges, 1.4° each;
//! - a gradient's blur, a box's backdrop blur, layer blur, custom shaders and
//!   GPU surfaces are left out.
//!
//! ```
//! use threers::style;
//! use threers::ui::Ui;
//! use threers::ui::vector::{ui_to_document, ui_to_pdf};
//! use threers::exporters::DrawingFonts;
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! ui.set_style(root, style! { background: #f5f5f4; padding: 16px; });
//! ui.add_text(root, "Quarterly report");
//! let fonts = DrawingFonts::new();
//! let doc = ui_to_document(&mut ui, 400.0, 200.0, &fonts);
//! assert_eq!((doc.width, doc.height), (400.0, 200.0));
//! let pdf = ui_to_pdf(&mut ui, 400.0, 200.0, &fonts);
//! assert!(pdf.starts_with(b"%PDF"));
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::exporters::{
    rasterize_svg, svg_to_png, write_pdf_pages_with, write_svg, DrawingFonts, PdfOptions,
    PdfText, RasterOptions, SvgDocument, SvgFillRule, SvgGradient, SvgGradientKind, SvgLineCap,
    SvgLineJoin, SvgNode, SvgPaint, SvgPath, SvgStop, SvgStyle, SvgTextAnchor, SvgTransform,
    SvgUnit,
};
use crate::ui::image::{Image, ImageRef};
use crate::ui::{GradientShape, Painter, Rect, Rgba, Style, TextMeasure, TextStyle, Ui};
use crate::utils::png::PngImage;

/// Lays `ui` out in a `width × height` viewport (CSS pixels), measuring text
/// in `fonts`, and records what it paints as a drawing of that size.
pub fn ui_to_document(ui: &mut Ui, width: f32, height: f32, fonts: &DrawingFonts) -> SvgDocument {
    let mut painter = VectorPainter::new(width, height, fonts.clone());
    ui.layout(width, height, &mut painter);
    ui.paint(&mut painter);
    painter.finish()
}

/// [`ui_to_document`], written as SVG.
pub fn ui_to_svg(ui: &mut Ui, width: f32, height: f32, fonts: &DrawingFonts) -> String {
    write_svg(&ui_to_document(ui, width, height, fonts))
}

/// [`ui_to_document`] as a one-page PDF, its text set in `fonts`, embedded.
pub fn ui_to_pdf(ui: &mut Ui, width: f32, height: f32, fonts: &DrawingFonts) -> Vec<u8> {
    let options = PdfOptions { text: PdfText::Embedded(fonts.clone()), ..PdfOptions::default() };
    write_pdf_pages_with(&[ui_to_document(ui, width, height, fonts)], &options)
}

/// [`ui_to_document`] drawn on the CPU at `scale` device pixels a CSS pixel,
/// as RGBA.
pub fn ui_to_image(ui: &mut Ui, width: f32, height: f32, scale: f32, fonts: &DrawingFonts) -> PngImage {
    let options = RasterOptions { dpi: 96.0 * scale as f64, background: None, fonts: fonts.clone() };
    rasterize_svg(&ui_to_document(ui, width, height, fonts), &options)
}

/// [`ui_to_image`], encoded as a PNG that records its resolution.
pub fn ui_to_png(ui: &mut Ui, width: f32, height: f32, scale: f32, fonts: &DrawingFonts) -> Vec<u8> {
    let options = RasterOptions { dpi: 96.0 * scale as f64, background: None, fonts: fonts.clone() };
    svg_to_png(&ui_to_document(ui, width, height, fonts), &options)
}

/// How much thicker a synthetic bold is drawn, in ems: an outline this wide
/// in the text's own colour, half of it outside the glyph.
const SYNTHETIC_BOLD: f64 = 0.04;
/// A synthetic italic's slant, x per y: 12°, about what browsers use.
const SYNTHETIC_ITALIC: f64 = 0.2126;
/// Layers a blurred shadow is drawn in.
const SHADOW_LAYERS: usize = 16;
/// Wedges a conic gradient is drawn in.
const CONIC_WEDGES: usize = 256;

/// What a [`Ui`] paints, recorded as an [`SvgDocument`]; also the
/// [`TextMeasure`] its layout must use (see [`ui_to_document`], which pairs
/// the two).
pub struct VectorPainter {
    fonts: DrawingFonts,
    doc: SvgDocument,
    /// What is being painted into now, and the groups it will close into.
    current: Vec<SvgNode>,
    open: Vec<Open>,
    /// Images already converted, by the UI image they came from.
    images: HashMap<*const Image, (ImageRef, Arc<PngImage>)>,
}

struct Open {
    outer: Vec<SvgNode>,
    transform: Option<SvgTransform>,
    style: SvgStyle,
}

impl VectorPainter {
    /// A painter for a `width × height` CSS-pixel page, measuring and drawing
    /// text in `fonts`.
    pub fn new(width: f32, height: f32, fonts: DrawingFonts) -> Self {
        let mut doc = SvgDocument::new(width as f64, height as f64, SvgUnit::Pixels);
        doc.precision = Some(3);
        VectorPainter { fonts, doc, current: vec![], open: vec![], images: HashMap::new() }
    }

    /// The drawing, with any group left open closed.
    pub fn finish(mut self) -> SvgDocument {
        while !self.open.is_empty() {
            self.close();
        }
        self.doc.children = std::mem::take(&mut self.current);
        self.doc
    }

    fn push(&mut self, node: SvgNode) {
        self.current.push(node);
    }

    fn open(&mut self, transform: Option<SvgTransform>, style: SvgStyle) {
        let outer = std::mem::take(&mut self.current);
        self.open.push(Open { outer, transform, style });
    }

    fn close(&mut self) {
        let Some(Open { outer, transform, style }) = self.open.pop() else { return };
        let children = std::mem::replace(&mut self.current, outer);
        if !children.is_empty() {
            self.current.push(SvgNode::Group { id: None, transform, style, children });
        }
    }

    fn png(&mut self, image: &ImageRef) -> Arc<PngImage> {
        let key = Arc::as_ptr(image);
        if let Some((_, png)) = self.images.get(&key) {
            return png.clone();
        }
        let png = Arc::new(PngImage { width: image.width, height: image.height, rgba: image.rgba.clone() });
        // Holding the image keeps its address from naming another.
        self.images.insert(key, (image.clone(), png.clone()));
        png
    }

    fn gradient(&mut self, kind: SvgGradientKind, stops: &[(f32, Rgba)]) -> SvgPaint {
        let stops = stops
            .iter()
            .map(|&(offset, c)| SvgStop { offset: offset as f64, color: rgb(c), opacity: c[3] as f64 })
            .collect();
        self.doc.gradient(SvgGradient { kind, stops, transform: None })
    }
}

fn rgb(c: Rgba) -> [u8; 3] {
    [0, 1, 2].map(|i| (c[i].clamp(0.0, 1.0) * 255.0).round() as u8)
}

fn below_one(a: f32) -> Option<f64> {
    (a < 1.0).then(|| a.max(0.0) as f64)
}

fn fill(c: Rgba) -> SvgStyle {
    SvgStyle { fill: Some(SvgPaint::Color(rgb(c))), fill_opacity: below_one(c[3]), ..SvgStyle::default() }
}

/// A rectangle with elliptical corners, clockwise from the top left's end.
fn rounded(r: Rect, rx: f32, ry: f32) -> SvgPath {
    let (x, y, w, h) = (r.x as f64, r.y as f64, r.width as f64, r.height as f64);
    let rx = (rx as f64).clamp(0.0, w / 2.0);
    let ry = (ry as f64).clamp(0.0, h / 2.0);
    let mut p = SvgPath::new();
    if rx <= 0.0 || ry <= 0.0 {
        return SvgPath::polygon(&[[x, y], [x + w, y], [x + w, y + h], [x, y + h]], true);
    }
    let corner = |p: &mut SvgPath, to: [f64; 2]| {
        p.arc_to([rx, ry], 0.0, false, true, to);
    };
    p.move_to([x + rx, y]);
    p.line_to([x + w - rx, y]);
    corner(&mut p, [x + w, y + ry]);
    p.line_to([x + w, y + h - ry]);
    corner(&mut p, [x + w - rx, y + h]);
    p.line_to([x + rx, y + h]);
    corner(&mut p, [x, y + h - ry]);
    p.line_to([x, y + ry]);
    corner(&mut p, [x + rx, y]);
    p.close();
    p
}

impl TextMeasure for VectorPainter {
    fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
        let w = self.fonts.advance(Some(&style.font_family), text, style.font_size as f64);
        (w as f32, style.line_height.unwrap_or(style.font_size * 1.2))
    }
}

impl Painter for VectorPainter {
    fn fill_rect(&mut self, rect: Rect, color: Rgba) {
        if color[3] <= 0.0 || rect.width <= 0.0 || rect.height <= 0.0 {
            return;
        }
        let origin = [rect.x as f64, rect.y as f64];
        self.push(SvgNode::Rect { origin, size: [rect.width as f64, rect.height as f64], corner_radius: 0.0, style: fill(color) });
    }

    fn fill_rounded_rect(&mut self, rect: Rect, rx: f32, ry: f32, color: Rgba) {
        if color[3] <= 0.0 || rect.width <= 0.0 || rect.height <= 0.0 {
            return;
        }
        self.push(SvgNode::Path { path: rounded(rect, rx, ry), style: fill(color) });
    }

    fn fill_linear_gradient(&mut self, rect: Rect, stops: &[(f32, Rgba)], _blur: f32) {
        self.fill_gradient(rect, GradientShape::Linear(90.0), stops, 0.0);
    }

    fn fill_gradient(&mut self, rect: Rect, shape: GradientShape, stops: &[(f32, Rgba)], _blur: f32) {
        if rect.width <= 0.0 || rect.height <= 0.0 || stops.is_empty() {
            return;
        }
        let (x, y, w, h) = (rect.x as f64, rect.y as f64, rect.width as f64, rect.height as f64);
        let box_path = || SvgPath::polygon(&[[x, y], [x + w, y], [x + w, y + h], [x, y + h]], true);
        let kind = match shape {
            GradientShape::Linear(angle) => {
                // CSS: 0° points up, 90° right; the line runs through the
                // middle, long enough that its ends' perpendiculars meet the
                // corners.
                let a = (angle as f64).to_radians();
                let dir = [a.sin(), -a.cos()];
                let half = (w * a.sin().abs() + h * a.cos().abs()) / 2.0;
                let c = [x + w / 2.0, y + h / 2.0];
                SvgGradientKind::Linear {
                    start: [c[0] - dir[0] * half, c[1] - dir[1] * half],
                    end: [c[0] + dir[0] * half, c[1] + dir[1] * half],
                }
            }
            GradientShape::Radial { cx, cy } => {
                let c = [x + cx as f64 * w, y + cy as f64 * h];
                let far = [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]
                    .iter()
                    .map(|p| ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).sqrt())
                    .fold(0.0, f64::max);
                SvgGradientKind::Radial { center: c, radius: far.max(1e-6), focus: None }
            }
            GradientShape::Conic { cx, cy, from } => {
                // Wedges about the centre, each its middle's colour, out past
                // the box and clipped to it.
                let c = [x + cx as f64 * w, y + cy as f64 * h];
                let reach = 2.0 * (w + h);
                let clip = self.doc.clip_path(box_path(), SvgFillRule::NonZero);
                self.open(None, SvgStyle { clip: Some(clip), ..SvgStyle::default() });
                for k in 0..CONIC_WEDGES {
                    let t0 = k as f64 / CONIC_WEDGES as f64;
                    let t1 = (k + 1) as f64 / CONIC_WEDGES as f64;
                    let color = crate::ui::gradient_at(stops, ((t0 + t1) / 2.0) as f32);
                    // Clockwise from straight up, `from` degrees on; a hair
                    // wider than its share, so neighbours leave no seam.
                    let at = |t: f64| {
                        let a = (from as f64).to_radians() + t * std::f64::consts::TAU;
                        [c[0] + reach * a.sin(), c[1] - reach * a.cos()]
                    };
                    let path = SvgPath::polygon(&[c, at(t0 - 0.0005), at(t1 + 0.0005)], true);
                    self.push(SvgNode::Path { path, style: fill(color) });
                }
                self.close();
                return;
            }
        };
        let paint = self.gradient(kind, stops);
        self.push(SvgNode::Path { path: box_path(), style: SvgStyle { fill: Some(paint), ..SvgStyle::default() } });
    }

    fn text(&mut self, line: Rect, text: &str, style: &TextStyle) {
        if text.is_empty() || style.color[3] <= 0.0 {
            return;
        }
        let family = style.font_family.clone();
        let size = style.font_size as f64;
        let (ascent, descent) = self.fonts.vertical_metrics(Some(&family), size);
        let (pos, rotation) = if style.vertical {
            // A quarter clockwise: the glyphs' tops face right, the run's
            // ascent-and-descent box centred across the line box.
            let base = line.x as f64 + line.width as f64 / 2.0 + (descent - ascent) / 2.0;
            ([base, line.y as f64], 90.0)
        } else {
            // Its ascent-and-descent box centred in the line box, as CSS
            // places a glyph run.
            let base = line.y as f64 + (line.height as f64 - (ascent + descent)) / 2.0 + ascent;
            ([line.x as f64, base], 0.0)
        };
        let mut paint = fill(style.color);
        if style.weight() >= 600 {
            paint.stroke = Some(SvgPaint::Color(rgb(style.color)));
            paint.stroke_opacity = below_one(style.color[3]);
            paint.stroke_width = Some(size * SYNTHETIC_BOLD);
            paint.line_join = Some(SvgLineJoin::Round);
        }
        let node = SvgNode::Text {
            pos,
            text: text.to_string(),
            size,
            anchor: SvgTextAnchor::Start,
            rotation,
            font_family: Some(family.clone()),
            style: paint,
        };
        if style.italic && !style.vertical {
            // Slanted about the baseline: x' = x − k·(y − baseline).
            let k = SYNTHETIC_ITALIC;
            self.push(SvgNode::Group {
                id: None,
                transform: Some([1.0, 0.0, -k, 1.0, k * pos[1], 0.0]),
                style: SvgStyle::default(),
                children: vec![node],
            });
        } else {
            self.push(node);
        }
        if style.line_through {
            let w = self.fonts.advance(Some(&family), text, size) as f32;
            let t = (style.font_size / 16.0).max(1.0);
            let r = if style.vertical {
                Rect { x: line.x + line.width / 2.0 - t / 2.0, y: line.y, width: t, height: w }
            } else {
                Rect { x: line.x, y: line.y + line.height / 2.0 - t / 2.0, width: w, height: t }
            };
            self.fill_rect(r, style.color);
        }
    }

    fn path(&mut self, rect: Rect, d: &[String], view_box: [f32; 4], stroke: Option<f32>, fill_it: bool, color: Rgba) {
        if color[3] <= 0.0 || view_box[2] <= 0.0 || view_box[3] <= 0.0 {
            return;
        }
        let Some(path) = SvgPath::from_data(&d.join(" ")) else { return };
        let sx = rect.width as f64 / view_box[2] as f64;
        let sy = rect.height as f64 / view_box[3] as f64;
        let transform = [sx, 0.0, 0.0, sy, rect.x as f64 - view_box[0] as f64 * sx, rect.y as f64 - view_box[1] as f64 * sy];
        let c = SvgPaint::Color(rgb(color));
        let style = SvgStyle {
            fill: Some(if fill_it { c } else { SvgPaint::None }),
            fill_opacity: below_one(color[3]),
            stroke: Some(if stroke.is_some() { c } else { SvgPaint::None }),
            stroke_width: stroke.map(|s| s as f64),
            stroke_opacity: below_one(color[3]),
            line_cap: Some(SvgLineCap::Round),
            line_join: Some(SvgLineJoin::Round),
            ..SvgStyle::default()
        };
        self.push(SvgNode::Group { id: None, transform: Some(transform), style: SvgStyle::default(), children: vec![SvgNode::Path { path, style }] });
    }

    fn push_clip(&mut self, rect: Rect) {
        self.push_clip_rounded(rect, 0.0, 0.0);
    }

    fn push_clip_rounded(&mut self, rect: Rect, rx: f32, ry: f32) {
        let clip = self.doc.clip_path(rounded(rect, rx, ry), SvgFillRule::NonZero);
        self.open(None, SvgStyle { clip: Some(clip), ..SvgStyle::default() });
    }

    fn pop_clip(&mut self) {
        self.close();
    }

    fn push_transform(&mut self, m: [f32; 6]) {
        self.open(Some(m.map(|v| v as f64)), SvgStyle::default());
    }

    fn pop_transform(&mut self) {
        self.close();
    }

    fn image(&mut self, rect: Rect, image: &ImageRef, opacity: f32) {
        if opacity <= 0.0 || rect.width <= 0.0 || rect.height <= 0.0 {
            return;
        }
        let png = self.png(image);
        self.push(SvgNode::Image {
            origin: [rect.x as f64, rect.y as f64],
            size: [rect.width as f64, rect.height as f64],
            image: png,
            style: SvgStyle { opacity: below_one(opacity), ..SvgStyle::default() },
        });
    }

    fn shadow(&mut self, rect: Rect, radius: f32, blur: f32, color: Rgba) {
        if color[3] <= 0.0 {
            return;
        }
        if blur <= 0.0 {
            return self.fill_rounded_rect(rect, radius, radius, color);
        }
        // CSS's blur radius is twice the Gaussian's deviation; the shadow
        // fades over two deviations either side of the box's edge. Each layer
        // is the box grown by its offset at an equal share of the alpha, so
        // where k layers overlap the cover is 1 − (1 − a)^k: from nearly the
        // full colour inside to nothing outside.
        let sigma = blur / 2.0;
        let a = 1.0 - (1.0 - color[3]).powf(1.0 / SHADOW_LAYERS as f32);
        for i in 0..SHADOW_LAYERS {
            let o = sigma * (-2.0 + 4.0 * (i as f32 + 0.5) / SHADOW_LAYERS as f32);
            let r = Rect { x: rect.x - o, y: rect.y - o, width: rect.width + 2.0 * o, height: rect.height + 2.0 * o };
            if r.width > 0.0 && r.height > 0.0 {
                let corner = (radius + o).max(0.0);
                self.fill_rounded_rect(r, corner, corner, [color[0], color[1], color[2], a]);
            }
        }
    }

    fn shadow_around(
        &mut self,
        rect: Rect,
        radius: f32,
        blur: f32,
        color: Rgba,
        hole: Rect,
        hole_radius: f32,
    ) {
        // Everything but the hole, by the even-odd rule.
        let mut p = rounded(hole, hole_radius, hole_radius);
        let (x, y, far) = (rect.x as f64 - 1e4, rect.y as f64 - 1e4, 2e4);
        p.move_to([x, y]);
        p.line_to([x + far + rect.width as f64, y]);
        p.line_to([x + far + rect.width as f64, y + far + rect.height as f64]);
        p.line_to([x, y + far + rect.height as f64]);
        p.close();
        let clip = self.doc.clip_path(p, SvgFillRule::EvenOdd);
        self.open(None, SvgStyle { clip: Some(clip), ..SvgStyle::default() });
        self.shadow(rect, radius, blur, color);
        self.close();
    }

    fn underline(&mut self, x: f32, y: f32, width: f32, thickness: f32, color: Rgba, wavy: bool) {
        if !wavy {
            return self.fill_rect(Rect { x, y: y - thickness / 2.0, width, height: thickness }, color);
        }
        // Half periods of a sine as quadratic arches, up and down.
        let amp = (crate::ui::WAVE_AMPLITUDE * thickness) as f64;
        let half = (crate::ui::WAVE_PERIOD * thickness) as f64 / 2.0;
        let (x, y, end) = (x as f64, y as f64, (x + width) as f64);
        let mut p = SvgPath::new();
        p.move_to([x, y]);
        let mut at = x;
        let mut up = true;
        while at < end {
            let next = (at + half).min(end);
            let peak = if up { y - 2.0 * amp } else { y + 2.0 * amp };
            p.quad_to([(at + next) / 2.0, peak], [next, y]);
            (at, up) = (next, !up);
        }
        let style = SvgStyle {
            stroke_width: Some(thickness as f64),
            stroke_opacity: below_one(color[3]),
            line_cap: Some(SvgLineCap::Butt),
            ..SvgStyle::stroke(rgb(color), thickness as f64)
        };
        self.push(SvgNode::Path { path: p, style });
    }
}
