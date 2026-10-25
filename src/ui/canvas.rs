//! A [`Painter`] for the browser's 2D canvas: the page rasterizes the boxes,
//! paths and text, as it does its own DOM. **wasm32 only.**
//!
//! Text can go two ways. On the canvas (`fillText`, the default), or, with
//! [`CanvasPainter::with_dom_text`], as positioned DOM text over it: Chrome
//! rasterizes canvas text and DOM text differently (gamma, contrast,
//! subpixel positioning), so a UI that must match a DOM page pixel for
//! pixel draws its text as the DOM does. Keeping the painting order then
//! takes layers: text is laid on the layer above the canvas it was painted
//! on, and anything painted after it starts a new canvas above that text,
//! so a popup still covers what lies under it. Clips apply to both (the
//! text through a clipping box around each run).
//!
//! ```no_run
//! # #[cfg(target_arch = "wasm32")]
//! # fn demo(ctx: web_sys::CanvasRenderingContext2d, ui: &threers::ui::Ui) {
//! use threers::ui::canvas::CanvasPainter;
//! let mut painter = CanvasPainter::new(ctx, 1.0);
//! painter.begin(800, 600, 1.0);
//! ui.paint(&mut painter);
//! painter.finish();
//! # }
//! ```

use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, Element, HtmlCanvasElement};

use super::{Painter, Rect, Rgba, Style, TextMeasure, TextStyle};

fn css(c: Rgba) -> String {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("rgba({},{},{},{})", b(c[0]), b(c[1]), b(c[2]), c[3])
}

/// A clip as pushed: its (snapped) box and corner radii.
#[derive(Clone, Copy)]
struct Clip {
    rect: (f64, f64, f64, f64),
    radii: Option<(f64, f64)>,
}

/// One layer of the DOM-text stack: a canvas and the text over it.
struct Layer {
    canvas: HtmlCanvasElement,
    ctx: CanvasRenderingContext2d,
    text: Element,
    /// Runs shown this frame, and the elements kept for reuse.
    used: usize,
    runs: Vec<Element>,
}

/// DOM text: the layers over the first canvas.
struct DomText {
    /// What the layers' elements go in (the first canvas's parent).
    parent: Element,
    /// The first canvas's own text layer, then each further layer.
    first_text: Element,
    first_used: usize,
    first_runs: Vec<Element>,
    layers: Vec<Layer>,
    /// The layer painted on now: 0 the first canvas.
    current: usize,
    /// Whether text was laid on the current layer (a shape after it moves
    /// up a layer).
    text_on_current: bool,
    /// Device size and ratio of this frame.
    size: (u32, u32),
    /// The layers' style: the first canvas's box.
    style: String,
    /// Each run's baseline below its top as the page lays it out
    /// (`line-height: normal`), by font and text.
    baselines: std::collections::HashMap<(String, String), f64>,
}

/// A [`Painter`] on a 2D canvas context, in CSS pixels (the context is
/// scaled by the device pixel ratio).
pub struct CanvasPainter {
    first: CanvasRenderingContext2d,
    /// Device pixels per CSS pixel.
    pub dpr: f64,
    /// Inside a `transform`ed subtree: boxes keep fractional edges.
    pub subpixel: bool,
    clips: Vec<Clip>,
    dom: Option<DomText>,
    /// Which runs go to the DOM (the rest on the canvas).
    dom_for: fn(&TextStyle) -> bool,
    /// Where blurred fills are drawn before they go on the page's canvas.
    scratch: Option<(HtmlCanvasElement, CanvasRenderingContext2d)>,
    /// The page's kerning of pairs with a space, which canvas text leaves
    /// out — see [`CanvasPainter::space_kerning`] — by font and pair, and the
    /// hidden span it is measured with.
    space_kerns: std::collections::HashMap<(String, char, char), f64>,
    probe: Option<Element>,
}

impl CanvasPainter {
    /// Painting on `ctx` (text included).
    pub fn new(ctx: CanvasRenderingContext2d, dpr: f64) -> Self {
        Self {
            first: ctx,
            dpr,
            subpixel: false,
            clips: vec![],
            dom: None,
            dom_for: |_| true,
            scratch: None,
            space_kerns: Default::default(),
            probe: None,
        }
    }

    /// [`with_dom_text`](Self::with_dom_text) for the runs `which` picks;
    /// the rest on the canvas. (Chrome's canvas text matches a page's in
    /// some cases — opaque colours over its own opaque canvas layer, say —
    /// and DOM text over a canvas in others, such as translucent colours.)
    pub fn with_dom_text_for(ctx: CanvasRenderingContext2d, dpr: f64, which: fn(&TextStyle) -> bool) -> Self {
        let mut painter = Self::with_dom_text(ctx, dpr);
        painter.dom_for = which;
        painter
    }

    /// Painting on `ctx`, text as DOM over it: the layers go in the
    /// canvas's parent, styled as the canvas is (its `style` attribute: so
    /// they cover the same box), and take no pointer events.
    pub fn with_dom_text(ctx: CanvasRenderingContext2d, dpr: f64) -> Self {
        let mut painter = Self::new(ctx, dpr);
        let canvas = painter.first.canvas();
        let parent = canvas.as_ref().and_then(|c| c.parent_element());
        if let (Some(canvas), Some(parent)) = (canvas, parent) {
            let style = format!("{};pointer-events:none;overflow:hidden", canvas.get_attribute("style").unwrap_or_else(|| "position:absolute;left:0;top:0;width:100%;height:100%".into()));
            if let Some(text) = new_layer_element(&parent, "div", &style) {
                let _ = canvas.after_with_node_1(&text);
                painter.dom = Some(DomText { parent, first_text: text, first_used: 0, first_runs: vec![], layers: vec![], current: 0, text_on_current: false, size: (0, 0), style, baselines: Default::default() });
            }
        }
        painter
    }

    /// A gradient blurred as a page blurs an element (`filter: blur()`): on
    /// a surface of its own, whole device pixels from the page's origin,
    /// then composited; a `filter` on the canvas itself blurs within the
    /// clip and comes out a few levels off along the way.
    fn fill_blurred(&mut self, r: Rect, stops: &[(f32, Rgba)], blur: f32) {
        let d = self.dpr;
        let m = (3.0 * blur as f64 * d).ceil() + 2.0;
        let (x, y) = ((r.x as f64 * d - m).round(), (r.y as f64 * d - m).round());
        let (w, h) = (r.width as f64 * d, r.height as f64 * d);
        if self.scratch.is_none() {
            let canvas = web_sys::window()
                .and_then(|w| w.document())
                .and_then(|doc| doc.create_element("canvas").ok())
                .and_then(|e| e.dyn_into::<HtmlCanvasElement>().ok());
            let ctx = canvas.as_ref().and_then(|c| c.get_context("2d").ok().flatten()).and_then(|c| c.dyn_into::<CanvasRenderingContext2d>().ok());
            if let (Some(canvas), Some(ctx)) = (canvas, ctx) {
                self.scratch = Some((canvas, ctx));
            }
        }
        let Some((canvas, sc)) = &self.scratch else { return };
        // Sizing the canvas clears it and resets its state.
        canvas.set_width((w + 2.0 * m).ceil() as u32);
        canvas.set_height((h + 2.0 * m).ceil() as u32);
        let g = sc.create_linear_gradient(m, 0.0, m + w, 0.0);
        for &(t, color) in stops {
            let _ = g.add_color_stop(t.clamp(0.0, 1.0), &css(color));
        }
        sc.set_filter(&format!("blur({}px)", blur as f64 * d));
        sc.set_fill_style_canvas_gradient(&g);
        sc.fill_rect(m, m, w, h);
        let c = self.ctx();
        c.save();
        let _ = c.set_transform(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let _ = c.draw_image_with_html_canvas_element(canvas, x, y);
        c.restore();
    }

    /// The context painted on now.
    fn ctx(&self) -> &CanvasRenderingContext2d {
        match &self.dom {
            Some(d) if d.current > 0 => &d.layers[d.current - 1].ctx,
            _ => &self.first,
        }
    }

    /// Clears the canvas (`width` × `height` device pixels, and the layers)
    /// and scales the context to CSS pixels.
    pub fn begin(&mut self, width: u32, height: u32, dpr: f64) {
        self.dpr = dpr;
        self.clips.clear();
        clear(&self.first, width, height, dpr);
        if let Some(d) = &mut self.dom {
            d.current = 0;
            d.text_on_current = false;
            d.first_used = 0;
            d.size = (width, height);
            for l in &mut d.layers {
                l.used = 0;
            }
        }
    }

    /// [`begin`](Self::begin), then `backdrop` (another canvas, a WebGPU
    /// one say) drawn across the whole canvas under what is painted next.
    ///
    /// With a context made opaque (`getContext('2d', {alpha: false})`),
    /// this is how text comes out as on a page: Chrome rasterizes text in a
    /// transparent canvas (or layer) with other contrast and gamma than over
    /// an opaque background, so a UI painted over a GPU canvas through a
    /// transparent overlay never quite matches the page's text.
    pub fn begin_with_backdrop(&mut self, width: u32, height: u32, dpr: f64, backdrop: &HtmlCanvasElement) {
        self.begin(width, height, dpr);
        let c = &self.first;
        let _ = c.set_transform(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let _ = c.draw_image_with_html_canvas_element_and_dw_and_dh(backdrop, 0.0, 0.0, width as f64, height as f64);
        let _ = c.set_transform(dpr, 0.0, 0.0, dpr, 0.0, 0.0);
    }

    /// Ends a frame: the text runs and layers not used this time are
    /// hidden. Nothing to do without DOM text.
    pub fn finish(&mut self) {
        let Some(d) = &mut self.dom else { return };
        for r in &d.first_runs[d.first_used..] {
            let _ = r.set_attribute("style", "display:none");
        }
        for (k, l) in d.layers.iter_mut().enumerate() {
            let shown = k < d.current;
            let hide = if shown { "" } else { "display:none;" };
            let _ = l.canvas.set_attribute("style", &format!("{hide}{}", d.style));
            let _ = l.text.set_attribute("style", &format!("{hide}{}", d.style));
            for r in &l.runs[l.used..] {
                let _ = r.set_attribute("style", "display:none");
            }
        }
    }

    /// Before a shape: above any text on this layer, a new layer.
    fn before_shape(&mut self) {
        let Some(d) = &mut self.dom else { return };
        if !d.text_on_current {
            return;
        }
        // The clips in force come off the context left (they go on the new
        // one below).
        let old = if d.current == 0 { self.first.clone() } else { d.layers[d.current - 1].ctx.clone() };
        for _ in &self.clips {
            old.restore();
        }
        d.current += 1;
        d.text_on_current = false;
        if d.layers.len() < d.current {
            let Some(canvas) = new_layer_element(&d.parent, "canvas", &d.style).and_then(|e| e.dyn_into::<HtmlCanvasElement>().ok()) else { return };
            let Some(ctx) = canvas.get_context("2d").ok().flatten().and_then(|c| c.dyn_into::<CanvasRenderingContext2d>().ok()) else { return };
            let Some(text) = new_layer_element(&d.parent, "div", &d.style) else { return };
            d.layers.push(Layer { canvas, ctx, text, used: 0, runs: vec![] });
        }
        let (w, h) = d.size;
        let l = &d.layers[d.current - 1];
        if l.canvas.width() != w || l.canvas.height() != h {
            l.canvas.set_width(w);
            l.canvas.set_height(h);
        }
        clear(&l.ctx, w, h, self.dpr);
        let _ = l.canvas.set_attribute("style", &d.style);
        let _ = l.text.set_attribute("style", &d.style);
        // The clips in force, on the new context too.
        let ctx = l.ctx.clone();
        for c in &self.clips {
            apply_clip(&ctx, c);
        }
    }

    /// A box snapped to device pixels as Chrome paints one: each edge
    /// rounded to the nearest pixel. In CSS pixels.
    fn snap(&self, r: Rect) -> (f64, f64, f64, f64) {
        if self.subpixel {
            return (r.x as f64, r.y as f64, r.width as f64, r.height as f64);
        }
        let d = self.dpr;
        let s = |v: f32| ((v as f64 * d) + 0.5).floor() / d;
        let (x0, y0, x1, y1) = (s(r.x), s(r.y), s(r.x + r.width), s(r.y + r.height));
        (x0, y0, x1 - x0, y1 - y0)
    }

    /// The page's kerning of the pair `a b` in `font`, one of them a space.
    ///
    /// A page kerns a space with its neighbours — Arial, Helvetica, Times,
    /// Trebuchet and Avenir all tighten "␣A", "␣T", "␣V", "␣W", "␣Y" — and
    /// a canvas does not, whole string or not, `fontKerning` or not (Chrome
    /// 156). So a line of canvas text ran 0.6–2.6 pixels longer than the
    /// page's, and every word after such a space sat that much off. Measured
    /// from the page once per font and pair, with a hidden span, at 64 times
    /// the size: kerning scales with it, and the page's widths are rounded
    /// to 1/64-pixel layout units, which at the size itself put a line with
    /// several such spaces a tenth of a pixel out.
    fn space_kerning(&mut self, style: &TextStyle, a: char, b: char) -> f64 {
        const ENLARGE: f32 = 64.0;
        let font = style.css_font();
        let key = (font.clone(), a, b);
        if let Some(&k) = self.space_kerns.get(&key) {
            return k;
        }
        if self.probe.is_none() {
            self.probe = web_sys::window()
                .and_then(|w| w.document())
                .and_then(|doc| {
                    let e = doc.create_element("span").ok()?;
                    let _ = e.set_attribute("style", "position:absolute;left:0;top:0;visibility:hidden;white-space:pre;pointer-events:none");
                    doc.body()?.append_child(&e).ok()?;
                    Some(e)
                });
        }
        let Some(probe) = &self.probe else { return 0.0 };
        let big = TextStyle { font_size: style.font_size * ENLARGE, ..style.clone() }.css_font();
        let _ = probe.set_attribute(
            "style",
            &format!("position:absolute;left:0;top:0;visibility:hidden;white-space:pre;pointer-events:none;font:{big}"),
        );
        // The pair from the page; each glyph alone from the canvas, whose
        // widths are not rounded to the page's 1/64-pixel layout units — one
        // rounding per pair rather than three.
        probe.set_text_content(Some(&format!("{a}{b}")));
        let pair = probe.get_bounding_client_rect().width();
        self.first.set_font(&big);
        let alone = |c: char| self.first.measure_text(&c.to_string()).map_or(0.0, |m| m.width());
        let k = (pair - alone(a) - alone(b)) / ENLARGE as f64;
        // Below what a page could position: no kerning, not a rounding.
        let k = if k.abs() < 0.01 { 0.0 } else { k };
        self.space_kerns.insert(key, k);
        k
    }

    /// `text` in pieces between its spaces, each with how far right of
    /// where the canvas would put it the page's space kerning moves it:
    /// `(start, end, shift)` in bytes and CSS pixels. One piece, unshifted,
    /// when no pair around a space kerns — the common case, and then the
    /// run is drawn whole, as it was.
    fn space_pieces(&mut self, text: &str, style: &TextStyle) -> Vec<(usize, usize, f64)> {
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let mut pieces = Vec::new();
        let (mut start, mut shift, mut moved) = (0usize, 0.0f64, false);
        for (k, &(at, c)) in chars.iter().enumerate() {
            if c != ' ' {
                continue;
            }
            let before = k.checked_sub(1).map(|j| chars[j].1).filter(|&p| p != ' ');
            let after = chars.get(k + 1).map(|&(_, n)| n).filter(|&n| n != ' ');
            let end = at + 1;
            pieces.push((start, at, shift));
            if let Some(p) = before {
                shift += self.space_kerning(style, p, ' ');
            }
            pieces.push((at, end, shift));
            if let Some(n) = after {
                shift += self.space_kerning(style, ' ', n);
            }
            moved |= shift != 0.0;
            start = end;
        }
        pieces.push((start, text.len(), shift));
        if !moved {
            return vec![(0, text.len(), 0.0)];
        }
        pieces.retain(|p| p.1 > p.0);
        pieces
    }

    /// Advance width, and the font's ascent and descent (`line-height:
    /// normal` spans both).
    fn metrics(&self, text: &str, font: &str) -> (f64, f64, f64) {
        let ctx = &self.first;
        ctx.set_font(font);
        match ctx.measure_text(text) {
            Ok(m) => (m.width(), m.font_bounding_box_ascent(), m.font_bounding_box_descent()),
            Err(_) => (0.0, 0.0, 0.0),
        }
    }

    /// A run as DOM text: a box clipped to the clips in force, the run in
    /// it at `line-height: normal`, placed so its baseline is where the
    /// canvas path puts it.
    fn dom_text(&mut self, line: Rect, text: &str, style: &TextStyle) {
        // Where the canvas path puts the baseline (half-leading above the
        // fonts' rounded ascent and descent, floored, as Blink centres a
        // line): the run is placed so its own baseline lands there.
        let (_, a, de) = self.metrics(text, &style.css_font());
        let baseline = ((line.height as f64 - (a.round() + de.round())) / 2.0).floor() + a.round();
        let clip = self.clips.iter().fold(None::<(f64, f64, f64, f64)>, |acc, c| {
            let (x, y, w, h) = c.rect;
            Some(match acc {
                None => (x, y, x + w, y + h),
                Some((x0, y0, x1, y1)) => (x0.max(x), y0.max(y), x1.min(x + w), y1.min(y + h)),
            })
        });
        let Some(d) = &mut self.dom else { return };
        d.text_on_current = true;
        let (layer_text, runs, used) = match d.current {
            0 => (&d.first_text, &mut d.first_runs, &mut d.first_used),
            k => {
                let l = &mut d.layers[k - 1];
                (&l.text, &mut l.runs, &mut l.used)
            }
        };
        if runs.len() <= *used {
            let Some(document) = web_sys::window().and_then(|w| w.document()) else { return };
            let Ok(outer) = document.create_element("div") else { return };
            let Ok(inner) = document.create_element("span") else { return };
            let _ = outer.append_child(&inner);
            let _ = layer_text.append_child(&outer);
            runs.push(outer);
        }
        let outer = &runs[*used];
        *used += 1;
        // The clip box (the page, unclipped), and the run within it.
        let (cx0, cy0, cx1, cy1) = clip.unwrap_or((-1e6, -1e6, 1e6, 1e6));
        let (cw, ch) = ((cx1 - cx0).max(0.0), (cy1 - cy0).max(0.0));
        let _ = outer.set_attribute("style", &format!("position:absolute;left:{cx0}px;top:{cy0}px;width:{cw}px;height:{ch}px;overflow:hidden;pointer-events:none"));
        let decoration = if style.line_through { "text-decoration:line-through;" } else { "" };
        let run_style = if style.vertical {
            format!(
                "position:absolute;left:{}px;top:{}px;width:{}px;height:{}px;writing-mode:vertical-lr;font:{};color:{};white-space:pre;{decoration}",
                line.x as f64 - cx0,
                line.y as f64 - cy0,
                line.width,
                line.height,
                style.css_font(),
                css(style.color)
            )
        } else {
            format!(
                "position:absolute;left:{}px;top:{}px;line-height:normal;font:{};color:{};white-space:pre;{decoration}",
                line.x as f64 - cx0,
                line.y as f64 - cy0,
                style.css_font(),
                css(style.color)
            )
        };
        let Some(inner) = outer.first_element_child() else { return };
        let _ = inner.set_attribute("style", &run_style);
        inner.set_text_content(Some(text));
        if !style.vertical {
            // The run's baseline below its top, measured once: a
            // zero-size box on the baseline after the text.
            let key = (style.css_font(), text.to_owned());
            let offset = *d.baselines.entry(key).or_insert_with(|| {
                let Some(marker) = web_sys::window().and_then(|w| w.document()).and_then(|doc| doc.create_element("span").ok()) else { return 0.0 };
                let _ = marker.set_attribute("style", "display:inline-block;width:0;height:0;vertical-align:baseline");
                let _ = inner.append_child(&marker);
                let at = marker.get_bounding_client_rect().top() - inner.get_bounding_client_rect().top();
                let _ = inner.remove_child(&marker);
                at
            });
            let top = line.y as f64 - cy0 + baseline - offset;
            let _ = inner.set_attribute("style", &run_style.replacen(&format!("top:{}px;", line.y as f64 - cy0), &format!("top:{top}px;"), 1));
        }
    }
}

fn new_layer_element(parent: &Element, tag: &str, style: &str) -> Option<Element> {
    let document = web_sys::window()?.document()?;
    let e = document.create_element(tag).ok()?;
    let _ = e.set_attribute("style", style);
    parent.append_child(&e).ok()?;
    Some(e)
}

fn clear(ctx: &CanvasRenderingContext2d, width: u32, height: u32, dpr: f64) {
    let _ = ctx.set_transform(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    ctx.clear_rect(0.0, 0.0, width as f64, height as f64);
    let _ = ctx.set_transform(dpr, 0.0, 0.0, dpr, 0.0, 0.0);
}

fn rounded_path(c: &CanvasRenderingContext2d, (x, y, w, h): (f64, f64, f64, f64), (rx, ry): (f64, f64)) {
    use std::f64::consts::{FRAC_PI_2, PI};
    c.begin_path();
    c.move_to(x + rx, y);
    c.line_to(x + w - rx, y);
    let _ = c.ellipse(x + w - rx, y + ry, rx, ry, 0.0, -FRAC_PI_2, 0.0);
    c.line_to(x + w, y + h - ry);
    let _ = c.ellipse(x + w - rx, y + h - ry, rx, ry, 0.0, 0.0, FRAC_PI_2);
    c.line_to(x + rx, y + h);
    let _ = c.ellipse(x + rx, y + h - ry, rx, ry, 0.0, FRAC_PI_2, PI);
    c.line_to(x, y + ry);
    let _ = c.ellipse(x + rx, y + ry, rx, ry, 0.0, PI, 1.5 * PI);
    c.close_path();
}

fn apply_clip(c: &CanvasRenderingContext2d, clip: &Clip) {
    c.save();
    match clip.radii {
        Some(r) => rounded_path(c, clip.rect, r),
        None => {
            c.begin_path();
            let (x, y, w, h) = clip.rect;
            c.rect(x, y, w, h);
        }
    }
    c.clip();
}

impl TextMeasure for CanvasPainter {
    fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
        // Blink rounds the font's ascent and descent to whole pixels.
        let ts = TextStyle::of(style);
        let (w, a, d) = self.metrics(text, &ts.css_font());
        // As wide as the page sets it, space kerning and all.
        let shift = self.space_pieces(text, &ts).last().map_or(0.0, |p| p.2);
        ((w + shift) as f32, style.line_height.unwrap_or((a.round() + d.round()) as f32))
    }
}

impl Painter for CanvasPainter {
    fn set_subpixel(&mut self, on: bool) {
        self.subpixel = on;
    }

    fn fill_rect(&mut self, r: Rect, color: Rgba) {
        self.before_shape();
        let (x, y, w, h) = self.snap(r);
        let c = self.ctx();
        c.set_fill_style_str(&css(color));
        c.fill_rect(x, y, w, h);
    }

    fn fill_rounded_rect(&mut self, r: Rect, rx: f32, ry: f32, color: Rgba) {
        self.before_shape();
        let rect = self.snap(r);
        let c = self.ctx();
        rounded_path(c, rect, (rx as f64, ry as f64));
        c.set_fill_style_str(&css(color));
        c.fill();
    }

    fn fill_linear_gradient(&mut self, r: Rect, stops: &[(f32, Rgba)], blur: f32) {
        self.before_shape();
        if blur > 0.0 {
            return self.fill_blurred(r, stops, blur);
        }
        let c = self.ctx();
        let g = c.create_linear_gradient(r.x as f64, 0.0, (r.x + r.width) as f64, 0.0);
        for &(t, color) in stops {
            let _ = g.add_color_stop(t.clamp(0.0, 1.0), &css(color));
        }
        c.save();
        c.set_fill_style_canvas_gradient(&g);
        c.fill_rect(r.x as f64, r.y as f64, r.width as f64, r.height as f64);
        c.restore();
    }

    fn text(&mut self, line: Rect, text: &str, style: &TextStyle) {
        if self.dom.is_some() && (self.dom_for)(style) {
            return self.dom_text(line, text, style);
        }
        // On the canvas, above any DOM text painted before it.
        self.before_shape();
        if style.vertical {
            // `vertical-lr`: the run turned a quarter clockwise, glyph tops
            // to the right; laid out as a horizontal line `line.width` tall.
            // The baseline (across the screen here) and the run's start on
            // device pixels: Chrome positions sideways glyphs without
            // subpixel offsets.
            let d = self.dpr;
            let snap = |v: f32| (v as f64 * d + 0.5).floor() / d;
            let c = self.ctx().clone();
            c.save();
            let _ = c.translate(snap(line.x + line.width), snap(line.y));
            let _ = c.rotate(std::f64::consts::FRAC_PI_2);
            let mut horizontal = style.clone();
            horizontal.vertical = false;
            self.text(Rect { x: 0.0, y: 0.0, width: line.height, height: line.width }, text, &horizontal);
            c.restore();
            return;
        }
        let f = style.css_font();
        let (w, a, d) = self.metrics(text, &f);
        let pieces = self.space_pieces(text, style);
        let w = w + pieces.last().map_or(0.0, |p| p.2);
        // Where a page would put the run: CSS lengths become `LayoutUnit`s
        // (1/64 px, truncated) before glyphs are placed on quarter pixels,
        // so 27.8837 px is drawn at 27.875 there; `fillText` takes the
        // float as it is.
        let line = Rect { x: super::layout_unit(line.x), y: super::layout_unit(line.y), ..line };
        // Half-leading above and below the font's ascent and descent, as
        // Blink lays a line out: metrics rounded, the leading above floored.
        let (a, d) = (a.round(), d.round());
        let baseline = line.y as f64 + ((line.height as f64 - (a + d)) / 2.0).floor() + a;
        let c = self.ctx();
        c.set_font(&f);
        c.set_text_baseline("alphabetic");
        c.set_text_align("left");
        c.set_fill_style_str(&css(style.color));
        if let [(_, _, shift)] = pieces[..] {
            let _ = c.fill_text(text, line.x as f64 + shift, baseline);
        } else {
            // Each piece where the canvas would put it within the run, moved
            // by the page's space kerning before it.
            for &(start, end, shift) in &pieces {
                let x = line.x as f64 + c.measure_text(&text[..start]).map_or(0.0, |m| m.width()) + shift;
                let _ = c.fill_text(&text[start..end], x, baseline);
            }
        }
        if style.line_through {
            // Through the middle of the x-height, one pixel thick.
            let y = (baseline - style.font_size as f64 * 0.3).round();
            c.fill_rect(line.x as f64, y - 0.5, w, 1.0);
        }
    }

    fn path(&mut self, r: Rect, d: &[String], view_box: [f32; 4], stroke: Option<f32>, fill: bool, color: Rgba) {
        self.before_shape();
        let c = self.ctx();
        let [vx, vy, vw, vh] = view_box.map(|v| v as f64);
        c.save();
        // An `<svg>` is painted at a whole-pixel origin.
        let dpr = self.dpr;
        let _ = c.translate((r.x as f64 * dpr + 0.5).floor() / dpr, (r.y as f64 * dpr + 0.5).floor() / dpr);
        let _ = c.scale(r.width as f64 / vw, r.height as f64 / vh);
        let _ = c.translate(-vx, -vy);
        let style = css(color);
        for data in d {
            // A `<circle>` as an oval (one full arc), as SVG strokes it.
            let path = match super::parse_circle(data) {
                Some((cx, cy, r)) => web_sys::Path2d::new().ok().inspect(|p| {
                    let _ = p.arc(cx as f64, cy as f64, r as f64, 0.0, std::f64::consts::TAU);
                }),
                None => web_sys::Path2d::new_with_path_string(data).ok(),
            };
            let Some(path) = path else { continue };
            if fill {
                c.set_fill_style_str(&style);
                c.fill_with_path_2d(&path);
            }
            if let Some(w) = stroke {
                c.set_line_width(w as f64);
                c.set_line_cap("round");
                c.set_line_join("round");
                c.set_stroke_style_str(&style);
                c.stroke_with_path(&path);
            }
        }
        c.restore();
    }

    fn push_clip(&mut self, r: Rect) {
        // `overflow: hidden` clips to the pixel-snapped box.
        let clip = Clip { rect: self.snap(r), radii: None };
        apply_clip(self.ctx(), &clip);
        self.clips.push(clip);
    }

    fn push_clip_rounded(&mut self, r: Rect, rx: f32, ry: f32) {
        let clip = Clip { rect: self.snap(r), radii: Some((rx as f64, ry as f64)) };
        apply_clip(self.ctx(), &clip);
        self.clips.push(clip);
    }

    fn pop_clip(&mut self) {
        self.clips.pop();
        self.ctx().restore();
    }
}
