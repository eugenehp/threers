//! An [`SvgDocument`] drawn into pixels on the CPU ([`rasterize_svg`]), and
//! so into a PNG ([`svg_to_png`]): what the SVG would show, for a preview, a
//! thumbnail or a golden-image test, with no GPU and no SVG library.
//!
//! The drawing is read as `drawing.rs` reads it for every format. Paths are
//! flattened in their own coordinates to a tenth of a pixel once
//! transformed. Fills are scan-converted with exact horizontal coverage and
//! sixteen sub-scanlines a pixel, under either fill rule. Strokes are turned
//! into an outline first (a quad per segment, a join at each vertex, a cap
//! at each open end, after the dash pattern is cut) and filled nonzero, so a
//! translucent stroke is one layer where it crosses itself, as SVG paints it.
//! The outline is built before the transform, so a non-uniformly scaled
//! stroke thickens the way SVG's does. Colours are composited source-over
//! in sRGB, as browsers and librsvg do.
//!
//! Text is drawn from the glyph outlines of the face [`RasterOptions::fonts`]
//! gives its family (DejaVu Sans, which threers carries, unless told
//! otherwise), laid out as `fonts.rs` lays it out for every format.
//!
//! ```
//! use threers::exporters::{rasterize_svg, RasterOptions, SvgDocument, SvgStyle, SvgUnit};
//!
//! let mut doc = SvgDocument::new(40.0, 20.0, SvgUnit::Pixels);
//! doc.rect([0.0, 0.0], [20.0, 20.0], 0.0, SvgStyle::fill([255, 0, 0]));
//! let img = rasterize_svg(&doc, &RasterOptions::default());
//! assert_eq!((img.width, img.height), (40, 20));
//! assert_eq!(img.pixel(5, 5), [255, 0, 0, 255]);
//! assert_eq!(img.pixel(30, 5), [0, 0, 0, 0]);
//! ```

use super::drawing::{self, Affine, Item, Paint, Resolved, Seg};
use super::fonts::{self, DrawingFonts, Face};
use super::svg::{SvgDocument, SvgFillRule, SvgGradient, SvgGradientKind, SvgLineCap, SvgLineJoin, SvgStop, SvgTextAnchor};
use crate::utils::png::PngImage;

/// How [`rasterize_svg`] draws.
#[derive(Debug, Clone, PartialEq)]
pub struct RasterOptions {
    /// Pixels an inch: 96 makes one CSS pixel one pixel.
    pub dpi: f64,
    /// What is under the drawing; transparent when `None`.
    pub background: Option<[u8; 4]>,
    /// The faces text is drawn in.
    pub fonts: DrawingFonts,
}

impl Default for RasterOptions {
    fn default() -> Self {
        RasterOptions { dpi: 96.0, background: None, fonts: DrawingFonts::new() }
    }
}

/// The drawing as 8-bit RGBA, the page's size at `options.dpi`, rounded to
/// whole pixels.
pub fn rasterize_svg(doc: &SvgDocument, options: &RasterOptions) -> PngImage {
    let inches = drawing::page_inches(doc);
    let w = (inches[0] * options.dpi).round().max(1.0) as usize;
    let h = (inches[1] * options.dpi).round().max(1.0) as usize;
    let mut canvas = Canvas::new(w, h, options.background);
    if let Some(view) = drawing::viewport(doc, [w as f64, h as f64]) {
        drawing::walk(doc, &mut |item| match item {
            Item::Shape { ctm, style, segs } => canvas.shape(doc, &drawing::compose(&view, &ctm), &style, &segs),
            Item::Text { ctm, style, pos, text, size, anchor, rotation, font_family } => {
                let segs = text_outline(&options.fonts.face(font_family), pos, text, size, anchor, rotation);
                canvas.shape(doc, &drawing::compose(&view, &ctm), &Resolved { fill_rule: SvgFillRule::NonZero, ..style }, &segs);
            }
            Item::Image { ctm, origin, size, image } => canvas.image(&drawing::compose(&view, &ctm), origin, size, image),
            Item::Push { ctm, opacity, clip } => canvas.push(&drawing::compose(&view, &ctm), opacity, clip),
            Item::Pop => canvas.pop(),
        });
    }
    canvas.finish()
}

/// [`rasterize_svg`], encoded as a PNG that records its resolution.
pub fn svg_to_png(doc: &SvgDocument, options: &RasterOptions) -> Vec<u8> {
    let img = rasterize_svg(doc, options);
    crate::utils::png::encode_png_with_dpi(img.width, img.height, &img.rgba, options.dpi)
}

/// How far a flattened curve may stray, in pixels.
const TOLERANCE: f64 = 0.1;
/// Sub-scanlines a pixel row.
const SUBSAMPLES: usize = 16;

/// Premultiplied RGBA, 0 to 1.
type Px = [f32; 4];

/// The pixels being drawn into, and the layers under them: each with the
/// opacity and clip coverage it is composited with when it ends.
struct Canvas {
    w: usize,
    h: usize,
    px: Vec<Px>,
    stack: Vec<(Vec<Px>, f32, Option<Mask>)>,
}

/// Coverage, 0 to 1, over the rows and columns a shape touches.
struct Mask {
    row0: usize,
    col0: usize,
    cols: usize,
    cover: Vec<f32>,
}

impl Mask {
    fn rows(&self) -> usize {
        self.cover.len().checked_div(self.cols).unwrap_or(0)
    }

    /// Coverage at a pixel of the canvas, 0 outside.
    fn at(&self, x: usize, y: usize) -> f32 {
        if x < self.col0 || y < self.row0 || x >= self.col0 + self.cols || y >= self.row0 + self.rows() {
            return 0.0;
        }
        self.cover[(y - self.row0) * self.cols + x - self.col0]
    }
}

/// What a covered pixel is painted with: premultiplied colour at its centre.
enum Shader<'a> {
    Solid(Px),
    Gradient {
        /// From the canvas to the gradient's own coordinates.
        inverse: Affine,
        gradient: &'a SvgGradient,
        stops: Vec<SvgStop>,
    },
    Image {
        /// From the canvas to the image's pixels.
        inverse: Affine,
        image: &'a PngImage,
    },
}

impl Shader<'_> {
    fn at(&self, x: usize, y: usize) -> Px {
        let p = [x as f64 + 0.5, y as f64 + 0.5];
        match self {
            Shader::Solid(c) => *c,
            Shader::Gradient { inverse, gradient, stops } => {
                let q = drawing::apply(inverse, p);
                stop_color(stops, gradient_t(&gradient.kind, q))
            }
            Shader::Image { inverse, image } => sample(image, drawing::apply(inverse, p)),
        }
    }
}

impl Canvas {
    fn new(w: usize, h: usize, background: Option<[u8; 4]>) -> Self {
        let bg = background.map_or([0.0; 4], |c| premultiply(c[0..3].try_into().unwrap(), c[3] as f32 / 255.0));
        Canvas { w, h, px: vec![bg; w * h], stack: Vec::new() }
    }

    /// A paint's shader under `m` (user space to canvas).
    fn shader<'a>(&self, doc: &'a SvgDocument, paint: Paint, m: &Affine) -> Option<Shader<'a>> {
        match paint {
            Paint::Color(c) => Some(Shader::Solid(premultiply(c, 1.0))),
            Paint::Gradient(k) => {
                let gradient = doc.gradients.get(k)?;
                let to_canvas = match &gradient.transform {
                    Some(t) => drawing::compose(m, t),
                    None => *m,
                };
                Some(Shader::Gradient { inverse: drawing::invert(&to_canvas)?, gradient, stops: drawing::stops(&gradient.stops) })
            }
        }
    }

    fn shape(&mut self, doc: &SvgDocument, m: &Affine, style: &Resolved, segs: &[Seg]) {
        let tol = TOLERANCE / drawing::max_scale(m).max(1e-12);
        let subpaths = flatten(segs, tol);
        if style.fills() {
            let polys: Vec<Vec<[f64; 2]>> =
                subpaths.iter().filter(|s| s.points.len() > 2).map(|s| s.points.iter().map(|p| drawing::apply(m, *p)).collect()).collect();
            if let (Some(mask), Some(shader)) = (self.coverage(&polys, style.fill_rule), style.fill.and_then(|p| self.shader(doc, p, m))) {
                self.paint(&mask, &shader, style.fill_opacity as f32);
            }
        }
        if style.strokes() {
            let mut pieces = Vec::new();
            let half = style.stroke_width / 2.0;
            for sp in &subpaths {
                if !style.dash.is_empty() {
                    for d in dash(sp, &style.dash) {
                        stroke_polyline(&d, half, style, tol, &mut pieces);
                    }
                } else {
                    stroke_polyline(sp, half, style, tol, &mut pieces);
                }
            }
            let polys: Vec<Vec<[f64; 2]>> = pieces.iter().map(|p| p.iter().map(|q| drawing::apply(m, *q)).collect()).collect();
            if let (Some(mask), Some(shader)) = (self.coverage(&polys, SvgFillRule::NonZero), style.stroke.and_then(|p| self.shader(doc, p, m))) {
                self.paint(&mask, &shader, style.stroke_opacity as f32);
            }
        }
    }

    /// An image stretched over the rectangle at `origin` of `size`, under `m`.
    fn image(&mut self, m: &Affine, origin: [f64; 2], size: [f64; 2], image: &PngImage) {
        // image pixels to user space, then to the canvas
        let place = [size[0] / image.width as f64, 0.0, 0.0, size[1] / image.height as f64, origin[0], origin[1]];
        let to_canvas = drawing::compose(m, &place);
        let Some(inverse) = drawing::invert(&to_canvas) else { return };
        let (w, h) = (image.width as f64, image.height as f64);
        let quad: Vec<[f64; 2]> = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]].iter().map(|p| drawing::apply(&to_canvas, *p)).collect();
        if let Some(mask) = self.coverage(&[quad], SvgFillRule::NonZero) {
            self.paint(&mask, &Shader::Image { inverse, image }, 1.0);
        }
    }

    /// A layer begins: drawing goes to fresh pixels until [`Canvas::pop`].
    fn push(&mut self, m: &Affine, opacity: f64, clip: Option<(Vec<Seg>, SvgFillRule)>) {
        let mask = clip.map(|(segs, rule)| {
            let tol = TOLERANCE / drawing::max_scale(m).max(1e-12);
            let polys: Vec<Vec<[f64; 2]>> = flatten(&segs, tol)
                .into_iter()
                .filter(|s| s.points.len() > 2)
                .map(|s| s.points.iter().map(|p| drawing::apply(m, *p)).collect())
                .collect();
            // a clip that covers nothing hides everything
            self.coverage(&polys, rule).unwrap_or(Mask { row0: 0, col0: 0, cols: 0, cover: Vec::new() })
        });
        let under = std::mem::replace(&mut self.px, vec![[0.0; 4]; self.w * self.h]);
        self.stack.push((under, opacity as f32, mask));
    }

    /// A layer ends: composited onto what was under it.
    fn pop(&mut self) {
        let Some((under, opacity, mask)) = self.stack.pop() else { return };
        let layer = std::mem::replace(&mut self.px, under);
        for (k, (d, s)) in self.px.iter_mut().zip(&layer).enumerate() {
            let a = opacity * mask.as_ref().map_or(1.0, |m| m.at(k % self.w, k / self.w));
            if a <= 0.0 || s[3] <= 0.0 {
                continue;
            }
            let k = 1.0 - s[3] * a;
            *d = [s[0] * a + d[0] * k, s[1] * a + d[1] * k, s[2] * a + d[2] * k, s[3] * a + d[3] * k];
        }
    }

    /// Polygons (closed implicitly) scan-converted to coverage.
    fn coverage(&self, polys: &[Vec<[f64; 2]>], rule: SvgFillRule) -> Option<Mask> {
        let mut edges: Vec<Edge> = Vec::new();
        let (mut ymin, mut ymax, mut xmin, mut xmax) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for poly in polys {
            // all of a polygon or none of it: one edge dropped unbalances the
            // winding along every row it spans
            if poly.iter().any(|p| !(p[0].is_finite() && p[1].is_finite())) {
                continue;
            }
            for k in 0..poly.len() {
                let (a, b) = (poly[k], poly[(k + 1) % poly.len()]);
                xmin = xmin.min(a[0]);
                xmax = xmax.max(a[0]);
                if a[1] == b[1] {
                    continue;
                }
                let (top, bot, dir) = if a[1] < b[1] { (a, b, 1) } else { (b, a, -1) };
                ymin = ymin.min(top[1]);
                ymax = ymax.max(bot[1]);
                edges.push(Edge { y0: top[1], y1: bot[1], x0: top[0], slope: (bot[0] - top[0]) / (bot[1] - top[1]), dir });
            }
        }
        if edges.is_empty() {
            return None;
        }
        edges.sort_by(|a, b| a.y0.total_cmp(&b.y0));
        let row0 = ymin.floor().max(0.0) as usize;
        let row1 = (ymax.ceil().min(self.h as f64)).max(0.0) as usize;
        let col0 = xmin.floor().max(0.0) as usize;
        let col1 = (xmax.ceil().min(self.w as f64)).max(0.0) as usize;
        if row0 >= row1 || col0 >= col1 {
            return None;
        }
        let cols = col1 - col0;
        let mut cover = vec![0f32; cols * (row1 - row0)];
        let mut active: Vec<usize> = Vec::new();
        let mut next = 0;
        let mut xs: Vec<(f64, i32)> = Vec::new();
        let weight = 1.0 / SUBSAMPLES as f64;
        for row in row0..row1 {
            let line = &mut cover[(row - row0) * cols..(row - row0 + 1) * cols];
            for sub in 0..SUBSAMPLES {
                let y = row as f64 + (sub as f64 + 0.5) / SUBSAMPLES as f64;
                while next < edges.len() && edges[next].y0 <= y {
                    active.push(next);
                    next += 1;
                }
                active.retain(|&e| edges[e].y1 > y);
                xs.clear();
                for &e in &active {
                    let ed = &edges[e];
                    if ed.y0 <= y {
                        xs.push((ed.x0 + (y - ed.y0) * ed.slope, ed.dir));
                    }
                }
                xs.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut winding = 0;
                for k in 0..xs.len().saturating_sub(1) {
                    winding += xs[k].1;
                    let inside = match rule {
                        SvgFillRule::NonZero => winding != 0,
                        SvgFillRule::EvenOdd => winding % 2 != 0,
                    };
                    if inside {
                        add_span(line, col0, xs[k].0, xs[k + 1].0, weight);
                    }
                }
            }
            for c in line.iter_mut() {
                *c = c.min(1.0);
            }
        }
        Some(Mask { row0, col0, cols, cover })
    }

    /// `shader` composited source-over where `mask` covers, at `opacity`.
    fn paint(&mut self, mask: &Mask, shader: &Shader, opacity: f32) {
        for r in 0..mask.rows() {
            let y = mask.row0 + r;
            for c in 0..mask.cols {
                let cov = mask.cover[r * mask.cols + c] * opacity;
                if cov <= 0.0 {
                    continue;
                }
                let x = mask.col0 + c;
                let s = shader.at(x, y);
                let d = &mut self.px[y * self.w + x];
                let k = 1.0 - s[3] * cov;
                *d = [s[0] * cov + d[0] * k, s[1] * cov + d[1] * k, s[2] * cov + d[2] * k, s[3] * cov + d[3] * k];
            }
        }
    }

    fn finish(mut self) -> PngImage {
        while !self.stack.is_empty() {
            self.pop();
        }
        let mut rgba = Vec::with_capacity(self.px.len() * 4);
        for p in &self.px {
            let a = p[3].clamp(0.0, 1.0);
            if a <= 0.0 {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
                continue;
            }
            for c in &p[..3] {
                rgba.push(((c / a).clamp(0.0, 1.0) * 255.0).round() as u8);
            }
            rgba.push((a * 255.0).round() as u8);
        }
        PngImage { width: self.w as u32, height: self.h as u32, rgba }
    }
}

/// An sRGB colour at `alpha`, premultiplied.
fn premultiply(c: [u8; 3], alpha: f32) -> Px {
    [c[0] as f32 / 255.0 * alpha, c[1] as f32 / 255.0 * alpha, c[2] as f32 / 255.0 * alpha, alpha]
}

/// Where `p` falls along a gradient, padded to 0 to 1.
fn gradient_t(kind: &SvgGradientKind, p: [f64; 2]) -> f64 {
    let t = match kind {
        SvgGradientKind::Linear { start, end } => {
            let d = [end[0] - start[0], end[1] - start[1]];
            let len2 = d[0] * d[0] + d[1] * d[1];
            if len2 == 0.0 {
                // SVG: a zero-length vector paints the last stop
                1.0
            } else {
                ((p[0] - start[0]) * d[0] + (p[1] - start[1]) * d[1]) / len2
            }
        }
        SvgGradientKind::Radial { center, radius, focus } => {
            if *radius <= 0.0 {
                1.0
            } else {
                let f = drawing::radial_focus(*center, *radius, *focus);
                // the t whose circle, from the focus (radius 0) to the outer
                // circle (radius r), passes through p: |p - f - t(c - f)| = t r
                let (cd, pd) = ([center[0] - f[0], center[1] - f[1]], [p[0] - f[0], p[1] - f[1]]);
                let a = cd[0] * cd[0] + cd[1] * cd[1] - radius * radius;
                let b = pd[0] * cd[0] + pd[1] * cd[1];
                let c = pd[0] * pd[0] + pd[1] * pd[1];
                if a.abs() < 1e-12 {
                    if b.abs() < 1e-12 { 0.0 } else { c / (2.0 * b) }
                } else {
                    // the focus is inside the circle, so a < 0 and the larger
                    // root is the one circle of the family through p
                    let disc = (b * b - a * c).max(0.0).sqrt();
                    ((b + disc) / a).max((b - disc) / a)
                }
            }
        }
    };
    t.clamp(0.0, 1.0)
}

/// The colour at `t` of stops in order, interpolated in sRGB with alpha
/// alongside (not premultiplied, as PDF interpolates a shading and its soft
/// mask), then premultiplied.
fn stop_color(stops: &[SvgStop], t: f64) -> Px {
    let Some(first) = stops.first() else { return [0.0; 4] };
    let last = stops.last().unwrap();
    let pick = |s: &SvgStop| ([s.color[0] as f64, s.color[1] as f64, s.color[2] as f64], s.opacity);
    let (c, a) = if t <= first.offset {
        pick(first)
    } else if t >= last.offset {
        pick(last)
    } else {
        let k = stops.windows(2).position(|w| t >= w[0].offset && t <= w[1].offset).unwrap_or(0);
        let (s0, s1) = (&stops[k], &stops[k + 1]);
        let span = s1.offset - s0.offset;
        let u = if span > 0.0 { (t - s0.offset) / span } else { 1.0 };
        let (c0, a0) = pick(s0);
        let (c1, a1) = pick(s1);
        ([c0[0] + (c1[0] - c0[0]) * u, c0[1] + (c1[1] - c0[1]) * u, c0[2] + (c1[2] - c0[2]) * u], a0 + (a1 - a0) * u)
    };
    let a = a.clamp(0.0, 1.0) as f32;
    [c[0] as f32 / 255.0 * a, c[1] as f32 / 255.0 * a, c[2] as f32 / 255.0 * a, a]
}

/// An image sampled bilinearly at `p` (in pixels, the first pixel's centre
/// at (0.5, 0.5)), edges clamped, premultiplied.
fn sample(image: &PngImage, p: [f64; 2]) -> Px {
    let (w, h) = (image.width as i64, image.height as i64);
    let fx = p[0] - 0.5;
    let fy = p[1] - 0.5;
    let (x0, y0) = (fx.floor() as i64, fy.floor() as i64);
    let (tx, ty) = ((fx - x0 as f64) as f32, (fy - y0 as f64) as f32);
    let px = |x: i64, y: i64| -> Px {
        let (x, y) = (x.clamp(0, w - 1) as usize, y.clamp(0, h - 1) as usize);
        let o = (y * image.width as usize + x) * 4;
        let c = &image.rgba[o..o + 4];
        premultiply([c[0], c[1], c[2]], c[3] as f32 / 255.0)
    };
    let (a, b, c, d) = (px(x0, y0), px(x0 + 1, y0), px(x0, y0 + 1), px(x0 + 1, y0 + 1));
    let mut out = [0.0; 4];
    for k in 0..4 {
        let top = a[k] + (b[k] - a[k]) * tx;
        let bottom = c[k] + (d[k] - c[k]) * tx;
        out[k] = top + (bottom - top) * ty;
    }
    out
}

struct Edge {
    y0: f64,
    y1: f64,
    x0: f64,
    slope: f64,
    dir: i32,
}

/// Coverage `weight` over `[x0, x1)` into `cover`, which starts at `col0`,
/// the end pixels by the fraction of them inside.
fn add_span(cover: &mut [f32], col0: usize, x0: f64, x1: f64, weight: f64) {
    let lo = col0 as f64;
    let hi = lo + cover.len() as f64;
    let (x0, x1) = (x0.max(lo), x1.min(hi));
    if x1 <= x0 {
        return;
    }
    let (i0, i1) = (x0.floor() as usize, (x1.ceil() as usize).min(col0 + cover.len()));
    if i1 - i0 == 1 {
        cover[i0 - col0] += ((x1 - x0) * weight) as f32;
        return;
    }
    cover[i0 - col0] += (((i0 + 1) as f64 - x0) * weight) as f32;
    for c in &mut cover[i0 + 1 - col0..i1 - 1 - col0] {
        *c += weight as f32;
    }
    cover[i1 - 1 - col0] += ((x1 - (i1 - 1) as f64) * weight) as f32;
}

/// A subpath flattened: its points, whether it was closed, and whether
/// anything but a move made it (a lone move draws nothing, a zero-length
/// line still gets its caps).
struct Polyline {
    points: Vec<[f64; 2]>,
    closed: bool,
    drawn: bool,
}

fn flatten(segs: &[Seg], tol: f64) -> Vec<Polyline> {
    let mut out: Vec<Polyline> = Vec::new();
    let mut cur: Option<Polyline> = None;
    let mut start = [0.0, 0.0];
    let mut last = [0.0, 0.0];
    let begin = |cur: &mut Option<Polyline>, out: &mut Vec<Polyline>, at: [f64; 2]| {
        if let Some(c) = cur.take() {
            out.push(c);
        }
        *cur = Some(Polyline { points: vec![at], closed: false, drawn: false });
    };
    for s in segs {
        match *s {
            Seg::Move(p) => {
                begin(&mut cur, &mut out, p);
                start = p;
                last = p;
            }
            Seg::Line(p) => {
                if cur.is_none() {
                    begin(&mut cur, &mut out, start);
                }
                let c = cur.as_mut().unwrap();
                c.points.push(p);
                c.drawn = true;
                last = p;
            }
            Seg::Cubic(c1, c2, p) => {
                if cur.is_none() {
                    begin(&mut cur, &mut out, start);
                }
                let c = cur.as_mut().unwrap();
                flatten_cubic(last, c1, c2, p, tol, &mut c.points);
                c.drawn = true;
                last = p;
            }
            Seg::Close => {
                if let Some(mut c) = cur.take() {
                    c.closed = true;
                    c.drawn = true;
                    out.push(c);
                }
                last = start;
            }
        }
    }
    if let Some(c) = cur {
        out.push(c);
    }
    for p in &mut out {
        p.points.dedup();
        // a closed subpath that came back to its start: the closing segment is implied
        if p.closed && p.points.len() > 1 && p.points.first() == p.points.last() {
            p.points.pop();
        }
    }
    out
}

/// A cubic as uniform steps, as many as Wang's bound says keep it within
/// `tol`.
fn flatten_cubic(p0: [f64; 2], p1: [f64; 2], p2: [f64; 2], p3: [f64; 2], tol: f64, out: &mut Vec<[f64; 2]>) {
    let dd = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| ((a[0] - 2.0 * b[0] + c[0]).powi(2) + (a[1] - 2.0 * b[1] + c[1]).powi(2)).sqrt();
    let d = dd(p0, p1, p2).max(dd(p1, p2, p3));
    let n = ((0.75 * d / tol).sqrt().ceil() as usize).clamp(1, 1000);
    for k in 1..=n {
        let t = k as f64 / n as f64;
        let u = 1.0 - t;
        let (a, b, c, e) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
        out.push([a * p0[0] + b * p1[0] + c * p2[0] + e * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + e * p3[1]]);
    }
}

/// The dashes of `sp`, each an open polyline; a closed subpath is dashed
/// around its closing segment too.
fn dash(sp: &Polyline, pattern: &[f64]) -> Vec<Polyline> {
    if !sp.drawn {
        return Vec::new();
    }
    let mut pts = sp.points.clone();
    if sp.closed && pts.len() > 1 {
        pts.push(pts[0]);
    }
    let mut out = Vec::new();
    let (mut k, mut left, mut on) = (0usize, pattern[0], true);
    let mut piece: Vec<[f64; 2]> = vec![pts[0]];
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        let mut at = 0.0;
        // `>=`: a dash that starts exactly at the end still has its caps
        while len - at >= left {
            at += left;
            let t = at / len;
            let p = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
            if on {
                piece.push(p);
                out.push(Polyline { points: std::mem::take(&mut piece), closed: false, drawn: true });
            } else {
                piece = vec![p];
            }
            on = !on;
            k = (k + 1) % pattern.len();
            left = pattern[k];
        }
        left -= len - at;
        if on {
            piece.push(b);
        }
    }
    if on && !piece.is_empty() {
        out.push(Polyline { points: piece, closed: false, drawn: true });
    }
    for p in &mut out {
        p.points.dedup();
    }
    out
}

/// The outline of a stroke along `sp`, as polygons all wound the same way,
/// so that filled nonzero they are one shape.
fn stroke_polyline(sp: &Polyline, half: f64, style: &Resolved, tol: f64, out: &mut Vec<Vec<[f64; 2]>>) {
    if !sp.drawn || half <= 0.0 {
        return;
    }
    let pts = &sp.points;
    let closed = sp.closed && pts.len() > 2;
    if pts.len() == 1 {
        // a zero-length subpath: just its caps
        match style.line_cap {
            SvgLineCap::Round => push_oriented(out, circle(pts[0], half, tol)),
            SvgLineCap::Square => {
                let [x, y] = pts[0];
                push_oriented(out, vec![[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]]);
            }
            SvgLineCap::Butt => {}
        }
        return;
    }
    let n = pts.len();
    let seg_count = if closed { n } else { n - 1 };
    let dir = |k: usize| {
        let (a, b) = (pts[k % n], pts[(k + 1) % n]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l = (dx * dx + dy * dy).sqrt();
        [dx / l, dy / l]
    };
    for k in 0..seg_count {
        let (mut a, mut b) = (pts[k], pts[(k + 1) % n]);
        let d = dir(k);
        if !closed && style.line_cap == SvgLineCap::Square {
            if k == 0 {
                a = [a[0] - d[0] * half, a[1] - d[1] * half];
            }
            if k + 1 == seg_count {
                b = [b[0] + d[0] * half, b[1] + d[1] * half];
            }
        }
        let nrm = [-d[1] * half, d[0] * half];
        push_oriented(out, vec![[a[0] + nrm[0], a[1] + nrm[1]], [b[0] + nrm[0], b[1] + nrm[1]], [b[0] - nrm[0], b[1] - nrm[1]], [a[0] - nrm[0], a[1] - nrm[1]]]);
    }
    // joins at the inner vertices, and at the start of a closed one
    let joins: Vec<usize> = if closed { (0..n).collect() } else { (1..n - 1).collect() };
    for v in joins {
        let d0 = dir((v + n - 1) % n);
        let d1 = dir(v % n);
        join(pts[v], d0, d1, half, style, tol, out);
    }
    if !closed && style.line_cap == SvgLineCap::Round {
        push_oriented(out, circle(pts[0], half, tol));
        push_oriented(out, circle(pts[n - 1], half, tol));
    }
}

fn join(v: [f64; 2], d0: [f64; 2], d1: [f64; 2], half: f64, style: &Resolved, tol: f64, out: &mut Vec<Vec<[f64; 2]>>) {
    let cross = d0[0] * d1[1] - d0[1] * d1[0];
    let dot = d0[0] * d1[0] + d0[1] * d1[1];
    if cross.abs() < 1e-12 && dot > 0.0 {
        return;
    }
    if style.line_join == SvgLineJoin::Round {
        push_oriented(out, circle(v, half, tol));
        return;
    }
    // the outer side is the one the path turns away from
    let s = if cross > 0.0 { -1.0 } else { 1.0 };
    let n0 = [-d0[1] * half * s, d0[0] * half * s];
    let n1 = [-d1[1] * half * s, d1[0] * half * s];
    let o0 = [v[0] + n0[0], v[1] + n0[1]];
    let o1 = [v[0] + n1[0], v[1] + n1[1]];
    if style.line_join == SvgLineJoin::Miter {
        // miter length / stroke width = 1 / sin(θ/2), θ the angle between the segments
        let ratio = 1.0 / ((1.0 + dot) / 2.0).max(0.0).sqrt();
        if ratio <= style.miter_limit {
            let bis = [n0[0] + n1[0], n0[1] + n1[1]];
            let bl = (bis[0] * bis[0] + bis[1] * bis[1]).sqrt();
            if bl > 0.0 {
                let m = [v[0] + bis[0] / bl * half * ratio, v[1] + bis[1] / bl * half * ratio];
                push_oriented(out, vec![v, o0, m, o1]);
                return;
            }
        }
    }
    push_oriented(out, vec![v, o0, o1]);
}

fn circle(c: [f64; 2], r: f64, tol: f64) -> Vec<[f64; 2]> {
    let step = if r > tol { 2.0 * (1.0 - tol / r).acos() } else { std::f64::consts::PI / 2.0 };
    let n = ((std::f64::consts::TAU / step).ceil() as usize).clamp(8, 1024);
    (0..n)
        .map(|k| {
            let a = k as f64 / n as f64 * std::f64::consts::TAU;
            [c[0] + r * a.cos(), c[1] + r * a.sin()]
        })
        .collect()
}

/// `poly`, reversed if need be so that its signed area is positive.
fn push_oriented(out: &mut Vec<Vec<[f64; 2]>>, mut poly: Vec<[f64; 2]>) {
    let mut area = 0.0;
    for k in 0..poly.len() {
        let (a, b) = (poly[k], poly[(k + 1) % poly.len()]);
        area += a[0] * b[1] - b[0] * a[1];
    }
    if area < 0.0 {
        poly.reverse();
    }
    if area != 0.0 {
        out.push(poly);
    }
}

/// A text run's glyph outlines as one path in user units: the baseline at
/// `pos`, `size` units to the em, shifted for its anchor, then SVG's
/// `rotate(rotation pos)`.
fn text_outline(face: &Face, pos: [f64; 2], text: &str, size: f64, anchor: SvgTextAnchor, rotation: f64) -> Vec<Seg> {
    let f = &face.ttf;
    let scale = size / face.units_per_em();
    let run = fonts::layout(face, text);
    let x = run.advance * scale;
    let shift = match anchor {
        SvgTextAnchor::Start => 0.0,
        SvgTextAnchor::Middle => -x / 2.0,
        SvgTextAnchor::End => -x,
    };
    let (sin, cos) = rotation.to_radians().sin_cos();
    // font units are y-up, the user space y-down
    let to_user = |gx: f64, gy: f64, x0: f64| {
        let (u, v) = (x0 + shift + gx * scale, -gy * scale);
        [pos[0] + cos * u - sin * v, pos[1] + sin * u + cos * v]
    };
    let mut segs = Vec::new();
    for (g, _, x0) in run.glyphs {
        let x0 = x0 * scale;
        for (starts, a, control, b) in f.outline_segments(g) {
            if starts {
                if !segs.is_empty() {
                    segs.push(Seg::Close);
                }
                segs.push(Seg::Move(to_user(a.x as f64, a.y as f64, x0)));
            }
            let pa = to_user(a.x as f64, a.y as f64, x0);
            let pb = to_user(b.x as f64, b.y as f64, x0);
            match control {
                Some(c) => {
                    let pc = to_user(c.x as f64, c.y as f64, x0);
                    let c1 = [pa[0] + 2.0 / 3.0 * (pc[0] - pa[0]), pa[1] + 2.0 / 3.0 * (pc[1] - pa[1])];
                    let c2 = [pb[0] + 2.0 / 3.0 * (pc[0] - pb[0]), pb[1] + 2.0 / 3.0 * (pc[1] - pb[1])];
                    segs.push(Seg::Cubic(c1, c2, pb));
                }
                None => segs.push(Seg::Line(pb)),
            }
        }
    }
    if !segs.is_empty() {
        segs.push(Seg::Close);
    }
    segs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exporters::{SvgNode, SvgPaint, SvgStyle, SvgUnit};

    fn render(doc: &SvgDocument) -> PngImage {
        rasterize_svg(doc, &RasterOptions::default())
    }

    #[test]
    fn edges_are_antialiased_by_area() {
        let mut doc = SvgDocument::new(4.0, 1.0, SvgUnit::Pixels);
        doc.rect([0.0, 0.0], [1.5, 1.0], 0.0, SvgStyle::fill([0, 0, 0]));
        let img = render(&doc);
        assert_eq!(img.pixel(0, 0)[3], 255);
        assert_eq!(img.pixel(1, 0)[3], 128, "half covered");
        assert_eq!(img.pixel(2, 0)[3], 0);
    }

    #[test]
    fn fill_rules_differ_on_a_self_overlap() {
        // two concentric squares wound the same way
        let mut doc = SvgDocument::new(10.0, 10.0, SvgUnit::Pixels);
        let mut p = crate::exporters::SvgPath::new();
        p.move_to([0.0, 0.0]).line_to([10.0, 0.0]).line_to([10.0, 10.0]).line_to([0.0, 10.0]).close();
        p.move_to([3.0, 3.0]).line_to([7.0, 3.0]).line_to([7.0, 7.0]).line_to([3.0, 7.0]).close();
        doc.path(p.clone(), SvgStyle::fill([0, 0, 0]));
        assert_eq!(render(&doc).pixel(5, 5)[3], 255);
        doc.children.clear();
        doc.path(p, SvgStyle::fill([0, 0, 0]).with_fill_rule(SvgFillRule::EvenOdd));
        assert_eq!(render(&doc).pixel(5, 5)[3], 0);
    }

    #[test]
    fn a_translucent_stroke_is_one_layer_where_it_overlaps_itself() {
        let mut doc = SvgDocument::new(20.0, 20.0, SvgUnit::Pixels);
        doc.polyline(&[[2.0, 10.0], [18.0, 10.0], [10.0, 2.0], [10.0, 18.0]], false, SvgStyle::stroke([0, 0, 0], 4.0).with_opacity(0.5));
        let img = render(&doc);
        assert_eq!(img.pixel(10, 10)[3], 128, "the crossing");
        assert_eq!(img.pixel(5, 10)[3], 128);
    }

    #[test]
    fn caps_and_dashes() {
        let mut doc = SvgDocument::new(30.0, 10.0, SvgUnit::Pixels);
        let butt = SvgStyle { line_cap: Some(SvgLineCap::Butt), ..SvgStyle::stroke([0, 0, 0], 2.0) };
        doc.line([5.0, 5.0], [25.0, 5.0], butt.clone().with_dash(&[4.0, 4.0]));
        let img = render(&doc);
        assert_eq!(img.pixel(6, 5)[3], 255, "first dash");
        assert_eq!(img.pixel(10, 5)[3], 0, "first gap");
        assert_eq!(img.pixel(14, 5)[3], 255, "second dash");
        assert_eq!(img.pixel(4, 5)[3], 0, "a butt cap stops at the end");
        doc.children.clear();
        doc.line([5.0, 5.0], [25.0, 5.0], SvgStyle { line_cap: Some(SvgLineCap::Square), ..butt });
        assert_eq!(render(&doc).pixel(4, 5)[3], 255, "a square cap reaches past it");
    }

    #[test]
    fn a_dash_starting_at_the_end_is_a_dot() {
        // dashes at 0, 25, 50, 75 and a zero-length one at 100
        let mut doc = SvgDocument::new(130.0, 10.0, SvgUnit::Pixels);
        let style = SvgStyle { line_cap: Some(SvgLineCap::Round), ..SvgStyle::stroke([0, 0, 0], 6.0) }.with_dash(&[15.0, 10.0]);
        doc.line([10.0, 5.0], [110.0, 5.0], style);
        let img = render(&doc);
        assert_eq!(img.pixel(110, 5)[3], 255, "the dot at the end");
        assert_eq!(img.pixel(104, 5)[3], 0, "the gap before it");
    }

    #[test]
    fn a_closed_path_back_at_its_start_strokes_no_extra_segment() {
        let mut doc = SvgDocument::new(40.0, 30.0, SvgUnit::Pixels);
        let mut p = crate::exporters::SvgPath::new();
        p.move_to([5.0, 20.0]).quad_to([20.0, -10.0], [35.0, 20.0]).quad_to([20.0, 30.0], [5.0, 20.0]).close();
        doc.path(p, SvgStyle { fill: Some(SvgPaint::None), ..SvgStyle::stroke([0, 0, 0], 1.0) });
        let img = render(&doc);
        assert_eq!(img.pixel(20, 20)[3], 0, "inside, between the two curves");
        assert_eq!(img.pixel(2, 20)[3], 0, "no stray span outside");
    }

    #[test]
    fn opacity_composites_over_the_background() {
        let mut doc = SvgDocument::new(2.0, 2.0, SvgUnit::Pixels);
        doc.rect([0.0, 0.0], [2.0, 2.0], 0.0, SvgStyle::fill([0, 0, 0]).with_opacity(0.5));
        let img = rasterize_svg(&doc, &RasterOptions { background: Some([255, 255, 255, 255]), ..Default::default() });
        assert_eq!(img.pixel(0, 0), [128, 128, 128, 255]);
    }

    #[test]
    fn text_draws_inside_its_advance() {
        let mut doc = SvgDocument::new(60.0, 20.0, SvgUnit::Pixels);
        doc.children.push(SvgNode::Text {
            pos: [30.0, 15.0],
            text: "HH".into(),
            size: 12.0,
            anchor: SvgTextAnchor::Middle,
            rotation: 0.0,
            font_family: None,
            style: SvgStyle { fill: Some(SvgPaint::Color([0, 0, 0])), ..Default::default() },
        });
        let img = render(&doc);
        let ink: Vec<u32> = (0..60).filter(|&x| (0..20).any(|y| img.pixel(x, y)[3] > 0)).collect();
        let (lo, hi) = (*ink.first().unwrap(), *ink.last().unwrap());
        // centred on 30, two H's of DejaVu Sans at 12 px are about 18 px
        assert!((lo as i32 + hi as i32 - 60).abs() <= 2, "{lo}..{hi}");
        assert!((14..=20).contains(&(hi - lo)), "{lo}..{hi}");
    }

    #[test]
    fn dpi_scales_the_page() {
        let doc = SvgDocument::new(1.0, 0.5, SvgUnit::Inches);
        let img = rasterize_svg(&doc, &RasterOptions { dpi: 300.0, ..Default::default() });
        assert_eq!((img.width, img.height), (300, 150));
    }
}
