//! Immediate-mode 2D drawing over the 3D view — p5's `rect`, `circle`, `line`
//! and `text`, for HUDs, labels, axes, plots and debug read-outs.
//!
//! Every project reaches this point: the render is right and now something has
//! to be written on top of it. The frame counter, the name of the thing under
//! the cursor, a legend for the colour ramp, a box around the selection. None
//! of that is 3D, and doing it in 3D — a textured quad per glyph, billboarded,
//! depth-disabled, re-uploaded when the string changes — is a disproportionate
//! amount of machinery for a line of text.
//!
//! So this is a CPU canvas of straight RGBA8 that
//! [`Renderer::draw_overlay`](crate::Renderer::draw_overlay) composites over an
//! already-rendered frame, uploading only the rows that changed.
//! [`crate::sketch`] wires it up for you — clear, draw, composite, once per
//! frame.
//!
//! ```
//! use threers::overlay::Overlay;
//! use threers::prelude::*;
//!
//! let mut o = Overlay::new(320, 200);
//! o.fill_alpha(Color::BLACK, 0.5);
//! o.rect(8.0, 8.0, 140.0, 48.0);
//! o.no_fill();
//! o.stroke(Color::from_hex(0x33ff99));
//! o.stroke_weight(2.0);
//! o.circle(80.0, 120.0, 30.0);
//! o.line(8.0, 190.0, 312.0, 190.0);
//! assert!(o.touched());
//! assert_eq!(o.pixels().len(), 320 * 200 * 4);
//! ```
//!
//! # State, not arguments
//!
//! `fill`/`stroke`/`stroke_weight` are sticky, the way p5's are, so a run of
//! shapes in one style is a run of one-line calls. The alternative — colour and
//! width on every call — reads worse at the call site for the common case of
//! drawing twenty things the same way.
//!
//! # Two things it gets right that are easy to get wrong
//!
//! Edges are anti-aliased from a signed distance field rather than sampled:
//! each primitive is one distance function, coverage is `0.5 - d` clamped, and
//! every shape goes through the same rasteriser. One path to get right, and no
//! supersampling cost.
//!
//! Compositing happens in linear light. [`Color`] holds linear values and the
//! buffer holds sRGB bytes — what the GPU pass wants — so overlapping shapes
//! decode, mix and re-encode. Blending coverage in sRGB directly is the classic
//! source of dark fringes on anti-aliased edges and of muddy half-transparent
//! fills.

use crate::math::{Color, Vector2};

/// sRGB-encoded byte to linear float, for the 256 values a byte can hold.
fn srgb_decode_table() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            let s = i as f32 / 255.0;
            *v = if s <= 0.040_45 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            };
        }
        t
    })
}

/// Linear float to an sRGB-encoded byte, the same curve [`Color::to_hex`] uses.
///
/// By table rather than by `powf`, and *exactly* rather than approximately. The
/// answer is one of 256 bytes, so the question is not "what is the encoded
/// value" but "which byte does this round to" — and that is answered by the 255
/// linear values where the rounding changes over. Byte `b` is chosen when `v`
/// reaches `decode((b - 0.5) / 255)`; the thresholds rise, so finding `b` is a
/// search and not a transcendental function.
///
/// Worth the trouble because it is the inner loop of every overlapping shape —
/// a panel with text on it encodes three channels per pixel — and it measures
/// 2.29 ns a call against 6.00 ns for the curve
/// (`cargo test --release overlay::tests::the_encode_table_is_faster -- --ignored
/// --nocapture`). End to end the gain is smaller than that ratio suggests,
/// because the encode interleaves with the loads and stores around it rather
/// than serialising; two overlapping full-viewport fills measure ~9 ms either
/// way on a loaded machine, so this is a faster function and not a measured
/// frame-time win.
fn srgb_encode(v: f32) -> u8 {
    let thresholds = srgb_encode_thresholds();
    let v = v.clamp(0.0, 1.0);
    // Branchless binary search: eight steps, each committing one bit of the
    // largest index whose threshold `v` has reached. `b` ends at 255 at most,
    // so `probe` is always in range.
    let mut b = 0usize;
    let mut step = 128usize;
    while step > 0 {
        let probe = b + step;
        if v >= thresholds[probe] {
            b = probe;
        }
        step >>= 1;
    }
    b as u8
}

/// Linear to encoded byte by the curve itself. The table above is built from
/// this and replaces it everywhere else.
fn srgb_encode_by_curve(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

/// The linear value at which the encoded byte becomes `b`, for each `b`.
/// `[0]` is zero so the search always has somewhere to start.
///
/// Each threshold is the *smallest `f32`* the curve encodes to `b` or above,
/// found by bisecting the curve over the bit patterns of `0.0..=1.0` — which
/// are ordered for non-negative floats, so bisecting them walks representable
/// values rather than a real interval. Inverting the formula analytically
/// instead, `decode((b - 0.5) / 255)`, is the obvious way and lands on the
/// wrong side of 14 of the 255 switchover points: the midpoint between two
/// bytes is exactly where `f32` rounding decides the answer, so the only way to
/// agree with the curve there is to ask it.
///
/// Built once, lazily, from about seven thousand `powf` calls.
fn srgb_encode_thresholds() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (b, v) in t.iter_mut().enumerate().skip(1) {
            let (mut lo, mut hi) = (0u32, 1.0f32.to_bits());
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if srgb_encode_by_curve(f32::from_bits(mid)) as usize >= b {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            *v = f32::from_bits(lo);
        }
        t
    })
}

/// A colour resolved for painting: linear for mixing, encoded for the fast
/// path, and an alpha.
#[derive(Debug, Clone, Copy)]
struct Paint {
    linear: [f32; 3],
    encoded: [u8; 3],
    alpha: f32,
}

impl Paint {
    fn new(color: Color, alpha: f32) -> Self {
        Self {
            linear: [color.r, color.g, color.b],
            encoded: [
                srgb_encode(color.r),
                srgb_encode(color.g),
                srgb_encode(color.b),
            ],
            alpha: alpha.clamp(0.0, 1.0),
        }
    }
}

/// A 2D canvas drawn on the CPU and composited over a rendered frame.
///
/// Coordinates are pixels from the top-left, matching the pointer position in
/// [`sketch::Frame`](crate::sketch::Frame) so a cursor read-out needs no
/// conversion. Fractional coordinates are honoured — edges are anti-aliased, so
/// a rectangle at `x = 10.5` lands half a pixel over.
pub struct Overlay {
    width: u32,
    height: u32,
    /// Straight (non-premultiplied) sRGB-encoded RGBA8, row-major from the
    /// top-left — exactly what the compositing pass uploads.
    pixels: Vec<u8>,
    fill: Option<Paint>,
    stroke: Option<Paint>,
    stroke_weight: f32,
    text_size: f32,
    /// Whether anything has been drawn since the last [`Overlay::clear`]. Lets
    /// a caller skip the texture upload on a frame that drew nothing.
    touched: bool,
    /// Rows whose pixels differ from what the GPU was last given, as one band.
    /// Cleared only by [`Overlay::mark_uploaded`], so skipping a frame's
    /// composite does not lose track of what changed during it.
    dirty: Option<(u32, u32)>,
    /// Rows drawn on since the last [`Overlay::clear`] — which is the band the
    /// GPU will hold ink in once this frame is uploaded, and therefore the band
    /// the *next* clear has to mark dirty. Without it, erasing a panel would
    /// leave it on screen: the CPU buffer goes to zero and nothing tells the
    /// texture.
    content: Option<(u32, u32)>,
    #[cfg(feature = "captions")]
    painter: Option<crate::captions::CaptionPainter>,
}

impl Overlay {
    /// A transparent canvas of `width × height` pixels.
    pub fn new(width: u32, height: u32) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        Self {
            width,
            height,
            pixels: vec![0; width as usize * height as usize * 4],
            fill: Some(Paint::new(Color::WHITE, 1.0)),
            stroke: None,
            stroke_weight: 1.0,
            text_size: 16.0,
            touched: false,
            dirty: None,
            content: None,
            #[cfg(feature = "captions")]
            painter: None,
        }
    }

    /// Resize, discarding the contents. A no-op if the size is unchanged, so
    /// calling it every frame from a resize handler costs nothing.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        self.pixels = vec![0; width as usize * height as usize * 4];
        self.touched = false;
        // A new size means a new texture on the other side, which is uploaded
        // whole, so there is nothing outstanding to describe.
        self.dirty = None;
        self.content = None;
    }

    /// Erase everything back to transparent.
    pub fn clear(&mut self) {
        self.pixels.fill(0);
        self.touched = false;
        // Rows that had ink have none now. The CPU buffer knows; the texture
        // does not, so they stay dirty until something uploads the zeroes.
        //
        // `dirty` only — not through `mark_rows`, which would put them back
        // into `content` as well. Doing that kept every row ever drawn on in
        // `content` for the life of the overlay, so a band that touched the
        // bottom of the canvas once stayed full height forever.
        if let Some((a, b)) = self.content.take() {
            self.dirty = widen(self.dirty, a, b);
        }
    }

    /// Rows that have changed since the last [`Overlay::mark_uploaded`], as a
    /// half-open range, or `None` if the texture is already current.
    ///
    /// One band rather than a set of rectangles: a read-out is one region, rows
    /// are contiguous in the buffer so a band is a contiguous upload with no
    /// stride to get wrong, and tracking anything finer costs more bookkeeping
    /// than the upload it would save.
    ///
    /// [`Renderer::draw_overlay`](crate::Renderer::draw_overlay) uses this and
    /// calls [`mark_uploaded`](Overlay::mark_uploaded) itself, which is the
    /// reason to prefer it over doing the two by hand.
    pub fn dirty_rows(&self) -> Option<(u32, u32)> {
        self.dirty
            .map(|(a, b)| (a.min(self.height), b.min(self.height)))
            .filter(|(a, b)| a < b)
    }

    /// Say that the texture now matches the buffer.
    pub fn mark_uploaded(&mut self) {
        self.dirty = None;
    }

    /// Record that `y0..y1` was drawn on: outstanding for the texture, and part
    /// of this frame's ink.
    fn mark_rows(&mut self, y0: u32, y1: u32) {
        if y0 >= y1 {
            return;
        }
        self.dirty = widen(self.dirty, y0, y1);
        self.content = widen(self.content, y0, y1);
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The raw buffer: straight-alpha sRGB RGBA8, row-major from the top-left.
    ///
    /// For compositing, [`Renderer::draw_overlay`](crate::Renderer::draw_overlay)
    /// takes the whole overlay and uploads only what changed;
    /// [`draw_overlay_rgba8`](crate::Renderer::draw_overlay_rgba8) takes these
    /// bytes and uploads all of them.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Whether anything has been drawn since the last [`Overlay::clear`].
    pub fn touched(&self) -> bool {
        self.touched
    }

    /// Fill subsequent shapes with `color`, opaque.
    pub fn fill(&mut self, color: Color) {
        self.fill = Some(Paint::new(color, 1.0));
    }

    /// Fill subsequent shapes with `color` at `alpha` (`0.0`–`1.0`).
    pub fn fill_alpha(&mut self, color: Color, alpha: f32) {
        self.fill = Some(Paint::new(color, alpha));
    }

    /// Draw subsequent shapes unfilled.
    pub fn no_fill(&mut self) {
        self.fill = None;
    }

    /// Outline subsequent shapes in `color`, opaque.
    pub fn stroke(&mut self, color: Color) {
        self.stroke = Some(Paint::new(color, 1.0));
    }

    /// Outline subsequent shapes in `color` at `alpha` (`0.0`–`1.0`).
    pub fn stroke_alpha(&mut self, color: Color, alpha: f32) {
        self.stroke = Some(Paint::new(color, alpha));
    }

    /// Draw subsequent shapes without an outline.
    pub fn no_stroke(&mut self) {
        self.stroke = None;
    }

    /// Outline width in pixels, centred on the shape's edge.
    pub fn stroke_weight(&mut self, pixels: f32) {
        self.stroke_weight = pixels.max(0.0);
    }

    /// Draw text in `font` from now on, instead of the UI face
    /// ([`CaptionFont::ui`](crate::captions::CaptionFont::ui): SF Compact on
    /// macOS, the bitmap face elsewhere).
    #[cfg(feature = "captions")]
    pub fn set_font(&mut self, font: crate::captions::CaptionFont) {
        self.painter = Some(crate::captions::CaptionPainter::with_font(font));
    }

    /// Type size in pixels for [`Overlay::text`].
    pub fn text_size(&mut self, pixels: f32) {
        self.text_size = pixels.max(1.0);
    }

    /// An axis-aligned rectangle with its top-left corner at `(x, y)`.
    pub fn rect(&mut self, x: f32, y: f32, width: f32, height: f32) {
        self.rounded_rect(x, y, width, height, 0.0);
    }

    /// A rectangle with corners rounded by `radius` pixels.
    pub fn rounded_rect(&mut self, x: f32, y: f32, width: f32, height: f32, radius: f32) {
        let (hx, hy) = (width * 0.5, height * 0.5);
        let (cx, cy) = (x + hx, y + hy);
        let r = radius.clamp(0.0, hx.abs().min(hy.abs()));
        let sdf = move |px: f32, py: f32| sd_round_box(px, py, cx, cy, hx.abs(), hy.abs(), r);
        self.shape(
            cx - hx.abs(),
            cy - hy.abs(),
            cx + hx.abs(),
            cy + hy.abs(),
            sdf,
        );
    }

    /// A circle of `radius` centred on `(cx, cy)`.
    pub fn circle(&mut self, cx: f32, cy: f32, radius: f32) {
        self.ellipse(cx, cy, radius, radius);
    }

    /// An axis-aligned ellipse centred on `(cx, cy)`.
    ///
    /// The distance field is the cheap radial approximation, exact for a circle
    /// and slightly off for a very eccentric ellipse — where it shows up as
    /// anti-aliasing a fraction of a pixel soft on the flatter sides, not as a
    /// misplaced edge.
    pub fn ellipse(&mut self, cx: f32, cy: f32, rx: f32, ry: f32) {
        let (rx, ry) = (rx.abs().max(1e-6), ry.abs().max(1e-6));
        let sdf = move |px: f32, py: f32| {
            let (dx, dy) = ((px - cx) / rx, (py - cy) / ry);
            ((dx * dx + dy * dy).sqrt() - 1.0) * rx.min(ry)
        };
        self.shape(cx - rx, cy - ry, cx + rx, cy + ry, sdf);
    }

    /// A line from `(x0, y0)` to `(x1, y1)`, in the stroke colour and width.
    ///
    /// Unlike the closed shapes this has no inside, so it ignores the fill and
    /// draws nothing when there is no stroke.
    pub fn line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        let Some(paint) = self.stroke else {
            return;
        };
        let half = self.stroke_weight * 0.5;
        let sdf = move |px: f32, py: f32| sd_segment(px, py, x0, y0, x1, y1) - half;
        let pad = half + 1.0;
        self.paint(
            x0.min(x1) - pad,
            y0.min(y1) - pad,
            x0.max(x1) + pad,
            y0.max(y1) + pad,
            &paint,
            sdf,
        );
    }

    /// A connected run of line segments, open — the last point is not joined
    /// back to the first. The shape for a plot or an axis.
    pub fn polyline(&mut self, points: &[Vector2]) {
        for w in points.windows(2) {
            self.line(w[0].x, w[0].y, w[1].x, w[1].y);
        }
    }

    /// A closed polygon, filled and stroked. Winding does not matter.
    ///
    /// Hand it a [`voronoi_cells`](crate::math::voronoi_cells) cell or a
    /// [`convex_hull`](crate::math::convex_hull) outline directly.
    pub fn polygon(&mut self, points: &[Vector2]) {
        if points.len() < 3 {
            // Two points are a line and one is nothing; neither has an inside,
            // and silently drawing half of what was asked for would be worse.
            if points.len() == 2 {
                self.line(points[0].x, points[0].y, points[1].x, points[1].y);
            }
            return;
        }
        let (mut x0, mut y0) = (f32::MAX, f32::MAX);
        let (mut x1, mut y1) = (f32::MIN, f32::MIN);
        for p in points {
            x0 = x0.min(p.x);
            y0 = y0.min(p.y);
            x1 = x1.max(p.x);
            y1 = y1.max(p.y);
        }
        // Borrowed, not copied into the closure: `shape` does not outlive this
        // call, and a `move` here was an allocation per polygon per frame.
        self.shape(x0, y0, x1, y1, |px, py| sd_polygon(px, py, points));
    }

    /// A string with its top-left corner at `(x, y)`, in the fill colour at the
    /// current [`text_size`](Overlay::text_size).
    ///
    /// A stroke, if set, becomes an outline around the glyphs — which is what
    /// makes small text readable over a busy render, and the reason this takes
    /// the stroke at all.
    ///
    /// Returns whether anything was drawn: `false` for an empty string or one
    /// placed entirely off-canvas.
    #[cfg(feature = "captions")]
    pub fn text(&mut self, x: f32, y: f32, text: &str) -> bool {
        self.draw_text(x, y, text, crate::captions::CaptionAlign::Left)
    }

    /// A string with its **right** edge at `x`, and its top at `y`.
    ///
    /// What a numeric read-out wants: the digits grow leftwards from a fixed
    /// edge instead of pushing the rest of the line around as the value
    /// changes. The alternative at the call site is
    /// [`measure_text`](Overlay::measure_text) and a subtraction, which is what
    /// this does and what every HUD ends up writing twice.
    #[cfg(feature = "captions")]
    pub fn text_right(&mut self, x: f32, y: f32, text: &str) -> bool {
        self.draw_text(x, y, text, crate::captions::CaptionAlign::Right)
    }

    /// A string centred horizontally on `x`, with its top at `y`.
    #[cfg(feature = "captions")]
    pub fn text_centered(&mut self, x: f32, y: f32, text: &str) -> bool {
        self.draw_text(x, y, text, crate::captions::CaptionAlign::Center)
    }

    /// The one implementation behind [`text`](Overlay::text),
    /// [`text_right`](Overlay::text_right) and
    /// [`text_centered`](Overlay::text_centered).
    ///
    /// `set_style` then draw, rather than a helper handing back a styled
    /// painter: the draw needs `self.painter` and `self.pixels` borrowed mutably
    /// at once, which the compiler allows for two fields of `self` named in the
    /// same body and not for a method that borrows all of it.
    #[cfg(feature = "captions")]
    fn draw_text(
        &mut self,
        x: f32,
        y: f32,
        text: &str,
        align: crate::captions::CaptionAlign,
    ) -> bool {
        use crate::captions::CaptionAlign;
        let style = self.text_style();
        let (w, h) = (self.width, self.height);
        let painter = self
            .painter
            .get_or_insert_with(crate::captions::CaptionPainter::new);
        painter.set_style(style);
        let pixels = &mut self.pixels;
        let drawn = match align {
            CaptionAlign::Left => painter.draw_text_at(pixels, w, h, x, y, text),
            CaptionAlign::Right => painter.draw_text_at_right(pixels, w, h, x, y, text),
            CaptionAlign::Center => painter.draw_text_at_center(pixels, w, h, x, y, text),
        };
        // Asked, not guessed: the painter re-wraps against the space left of
        // the anchor and slides the block up off the bottom edge, so the rows a
        // string lands on are not `y..y + measured height`.
        let rows = painter.last_rows();
        self.touched |= drawn;
        if let Some((y0, y1)) = rows {
            self.mark_rows(y0, y1);
        }
        drawn
    }

    /// Width and height of `text` in pixels, without drawing it — for
    /// right-aligning a read-out or sizing a box behind a label.
    #[cfg(feature = "captions")]
    pub fn measure_text(&mut self, text: &str) -> (f32, f32) {
        let style = self.text_style();
        let w = self.width;
        let painter = self
            .painter
            .get_or_insert_with(crate::captions::CaptionPainter::new);
        painter.set_style(style);
        let laid = painter.measure(text, w);
        (laid.width, laid.height)
    }

    /// The current fill, stroke and type size as a caption style. Text with no
    /// fill set would be invisible rather than absent, so it falls back to
    /// white — `no_fill` is about shapes, and a `text` call is an instruction to
    /// draw something.
    #[cfg(feature = "captions")]
    fn text_style(&self) -> crate::captions::CaptionStyle {
        let rgba = |p: &Paint| {
            [
                p.encoded[0],
                p.encoded[1],
                p.encoded[2],
                (p.alpha * 255.0 + 0.5) as u8,
            ]
        };
        let fill = self.fill.unwrap_or_else(|| Paint::new(Color::WHITE, 1.0));
        crate::captions::CaptionStyle {
            font_size: self.text_size,
            color: rgba(&fill),
            // No box and no shadow: this is a drawing primitive, and anything
            // behind the text is a `rect` the caller can place themselves.
            background: [0, 0, 0, 0],
            shadow_color: [0, 0, 0, 0],
            outline_color: self.stroke.as_ref().map_or([0, 0, 0, 0], rgba),
            outline_width: match self.stroke {
                Some(_) => self.stroke_weight * 0.5,
                None => 0.0,
            },
            padding: 0.0,
            margin: 0.0,
            ..Default::default()
        }
    }

    /// Fill then stroke one distance field — the shared body of every closed
    /// primitive. `sdf` is negative inside, in pixels.
    fn shape<F>(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, sdf: F)
    where
        F: Fn(f32, f32) -> f32,
    {
        if let Some(paint) = self.fill {
            self.paint(x0 - 1.0, y0 - 1.0, x1 + 1.0, y1 + 1.0, &paint, &sdf);
        }
        if let Some(paint) = self.stroke {
            let half = self.stroke_weight * 0.5;
            // The outline is the band within half a weight of the edge, which
            // is the same field folded about zero.
            let banded = |px: f32, py: f32| sdf(px, py).abs() - half;
            let pad = half + 1.0;
            self.paint(x0 - pad, y0 - pad, x1 + pad, y1 + pad, &paint, banded);
        }
    }

    /// Rasterise one distance field over a bounding box.
    fn paint<F>(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, paint: &Paint, sdf: F)
    where
        F: Fn(f32, f32) -> f32,
    {
        if paint.alpha <= 0.0 {
            return;
        }
        let lo_x = x0.floor().max(0.0) as u32;
        let lo_y = y0.floor().max(0.0) as u32;
        let hi_x = (x1.ceil().max(0.0) as u32 + 1).min(self.width);
        let hi_y = (y1.ceil().max(0.0) as u32 + 1).min(self.height);
        if lo_x >= hi_x || lo_y >= hi_y {
            return;
        }
        // Once per primitive, from its bounding box, rather than per pixel in
        // the loop below. Conservative by at most a row of coverage either side
        // of the shape, which is a row of upload and not a wrong picture.
        self.mark_rows(lo_y, hi_y);
        let table = srgb_decode_table();
        // Walked in tiles so whole tiles can be settled from one evaluation.
        // A signed distance field is 1-Lipschitz — moving a pixel changes it by
        // at most a pixel — so if the centre of a tile is further than the
        // tile's own radius from the edge, every sample in it is on the same
        // side and none of them needs evaluating. Coverage ramps over
        // `d` in `-0.5..0.5`, so that is the margin to clear.
        //
        // This is most of the cost of a large shape. `polygon` measures the
        // distance to every edge at every pixel, and a Voronoi cell covering an
        // eighth of a 1000x640 view is 80,000 pixels of which perhaps 3,000 are
        // near an edge. Eight such cells, filled and stroked, with the Delaunay
        // edges over the top: 96 ms a frame before tiling, 5 ms after. A
        // read-out and a gradient legend — which is what most frames draw — is
        // 183 us either way, so this costs nothing in the common case.
        const TILE: u32 = 8;
        for ty in (lo_y..hi_y).step_by(TILE as usize) {
            let ty1 = (ty + TILE).min(hi_y);
            for tx in (lo_x..hi_x).step_by(TILE as usize) {
                let tx1 = (tx + TILE).min(hi_x);
                // Furthest a sample centre in this tile can be from its middle.
                // The tile may be clipped, so this is measured and not assumed.
                let rx = (tx1 - tx) as f32 * 0.5 - 0.5;
                let ry = (ty1 - ty) as f32 * 0.5 - 0.5;
                let radius = (rx * rx + ry * ry).sqrt();
                let middle = sdf((tx + tx1) as f32 * 0.5, (ty + ty1) as f32 * 0.5);
                if middle - radius > 0.5 {
                    continue; // wholly outside
                }
                // Wholly inside, so every sample is fully covered and the
                // distance field never has to be asked again.
                let solid = middle + radius < -0.5;

                for py in ty..ty1 {
                    let row = py as usize * self.width as usize;
                    for px in tx..tx1 {
                        // Sampled at the pixel centre, and coverage is the
                        // distance to the edge over one pixel — the standard
                        // analytic approximation, and exact for an edge through
                        // the centre.
                        let alpha = if solid {
                            paint.alpha
                        } else {
                            let d = sdf(px as f32 + 0.5, py as f32 + 0.5);
                            let coverage = (0.5 - d).clamp(0.0, 1.0);
                            if coverage <= 0.0 {
                                continue;
                            }
                            paint.alpha * coverage
                        };
                        let i = (row + px as usize) * 4;
                        let dst = &mut self.pixels[i..i + 4];
                        let da = dst[3] as f32 / 255.0;
                        if da <= 0.0 {
                            // Nothing underneath, so there is nothing to mix
                            // with and the encoded colour goes straight down.
                            // The common case on a cleared canvas, and it skips
                            // both transfer functions.
                            dst[0] = paint.encoded[0];
                            dst[1] = paint.encoded[1];
                            dst[2] = paint.encoded[2];
                            dst[3] = (alpha * 255.0 + 0.5) as u8;
                        } else {
                            let out_a = alpha + da * (1.0 - alpha);
                            for c in 0..3 {
                                let under = table[dst[c] as usize];
                                let mixed = (paint.linear[c] * alpha
                                    + under * da * (1.0 - alpha))
                                    / out_a;
                                dst[c] = srgb_encode(mixed);
                            }
                            dst[3] = (out_a * 255.0 + 0.5) as u8;
                        }
                        self.touched = true;
                    }
                }
            }
        }
    }
}

/// Grow a row band to cover `y0..y1`.
fn widen(band: Option<(u32, u32)>, y0: u32, y1: u32) -> Option<(u32, u32)> {
    if y0 >= y1 {
        return band;
    }
    Some(match band {
        Some((a, b)) => (a.min(y0), b.max(y1)),
        None => (y0, y1),
    })
}

/// Signed distance to a box, negative inside.
fn sd_box(px: f32, py: f32, cx: f32, cy: f32, hx: f32, hy: f32) -> f32 {
    let qx = (px - cx).abs() - hx;
    let qy = (py - cy).abs() - hy;
    let (ox, oy) = (qx.max(0.0), qy.max(0.0));
    (ox * ox + oy * oy).sqrt() + qx.max(qy).min(0.0)
}

/// Signed distance to a box with rounded corners, negative inside.
fn sd_round_box(px: f32, py: f32, cx: f32, cy: f32, hx: f32, hy: f32, r: f32) -> f32 {
    sd_box(px, py, cx, cy, (hx - r).max(0.0), (hy - r).max(0.0)) - r
}

/// Unsigned distance to a line segment.
fn sd_segment(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let (vx, vy) = (bx - ax, by - ay);
    let (wx, wy) = (px - ax, py - ay);
    let len2 = vx * vx + vy * vy;
    let t = if len2 > 0.0 {
        ((wx * vx + wy * vy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (dx, dy) = (wx - vx * t, wy - vy * t);
    (dx * dx + dy * dy).sqrt()
}

/// Signed distance to a closed polygon, negative inside. The sign comes from a
/// crossing count rather than the winding, so a clockwise polygon and a
/// counter-clockwise one of the same outline give the same answer.
fn sd_polygon(px: f32, py: f32, pts: &[Vector2]) -> f32 {
    let n = pts.len();
    let mut best = f32::MAX;
    let mut inside = false;
    for i in 0..n {
        let j = (i + n - 1) % n;
        let (a, b) = (pts[i], pts[j]);
        best = best.min(sd_segment(px, py, a.x, a.y, b.x, b.y));
        // Ray crossing to the right: the edge straddles this row, and the
        // crossing is to the right of the point.
        if (py >= a.y) != (py >= b.y) {
            let t = (py - a.y) / (b.y - a.y);
            if a.x + t * (b.x - a.x) > px {
                inside = !inside;
            }
        }
    }
    if inside {
        -best
    } else {
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straight-alpha RGBA at a pixel.
    fn at(o: &Overlay, x: u32, y: u32) -> [u8; 4] {
        let (w, _) = o.size();
        let i = (y as usize * w as usize + x as usize) * 4;
        let p = o.pixels();
        [p[i], p[i + 1], p[i + 2], p[i + 3]]
    }

    /// The table has to give the same byte as the curve it replaces, not a
    /// close one — every pixel of every overlapping shape goes through it, and
    /// an off-by-one would be a banding artefact nobody could trace back here.
    /// Swept finely enough to land either side of all 255 switchover points.
    #[test]
    fn the_encode_table_matches_the_curve_it_replaces() {
        let mut differing = Vec::new();
        const STEPS: u32 = 200_000;
        for i in 0..=STEPS {
            let v = i as f32 / STEPS as f32;
            if srgb_encode(v) != srgb_encode_by_curve(v) {
                differing.push(v);
            }
        }
        // Then every switchover and the representable value either side of it,
        // which is where the two can actually disagree and which a sweep of any
        // density will mostly step over.
        for &t in srgb_encode_thresholds().iter() {
            for v in [
                f32::from_bits(t.to_bits().saturating_sub(1)),
                t,
                f32::from_bits(t.to_bits() + 1),
            ] {
                if srgb_encode(v) != srgb_encode_by_curve(v) {
                    differing.push(v);
                }
            }
        }
        assert!(
            differing.is_empty(),
            "{} values encode differently, e.g. {:?} -> {} by table, {} by curve",
            differing.len(),
            differing[0],
            srgb_encode(differing[0]),
            srgb_encode_by_curve(differing[0])
        );
        // And the ends, which a sweep can miss.
        assert_eq!(srgb_encode(0.0), 0);
        assert_eq!(srgb_encode(1.0), 255);
        assert_eq!(srgb_encode(-5.0), 0, "clamped");
        assert_eq!(srgb_encode(5.0), 255, "clamped");
        // Round trip against the decode table, which is the other half of the
        // pair: every byte must encode back to itself.
        let decode = srgb_decode_table();
        for b in 0..=255u8 {
            assert_eq!(
                srgb_encode(decode[b as usize]),
                b,
                "byte {b} did not survive decode then encode"
            );
        }
    }

    /// Where the 2.29 ns against 6.00 ns in [`srgb_encode`]'s own docs comes
    /// from. Ignored by default rather than asserted: the ratio is stable but
    /// the absolute numbers are not, and a timing assertion in a test suite
    /// fails for reasons that have nothing to do with the code.
    #[test]
    #[ignore]
    fn the_encode_table_is_faster_than_the_curve() {
        let vals: Vec<f32> = (0..1 << 16).map(|i| (i as f32) / ((1 << 16) as f32)).collect();
        let reps = 200;
        for (label, f) in [
            ("table", srgb_encode as fn(f32) -> u8),
            ("curve", srgb_encode_by_curve as fn(f32) -> u8),
        ] {
            let mut best = f64::MAX;
            for _ in 0..7 {
                let t = std::time::Instant::now();
                let mut acc = 0u64;
                for _ in 0..reps {
                    for &v in &vals {
                        acc += f(v) as u64;
                    }
                }
                let ns = t.elapsed().as_secs_f64() * 1e9 / (reps * vals.len()) as f64;
                best = best.min(ns);
                std::hint::black_box(acc);
            }
            eprintln!("{label}: {best:.2} ns/call");
        }
    }

    #[test]
    fn a_new_overlay_is_transparent_and_untouched() {
        let o = Overlay::new(8, 4);
        assert_eq!(o.size(), (8, 4));
        assert_eq!(o.pixels().len(), 8 * 4 * 4);
        assert!(o.pixels().iter().all(|&b| b == 0));
        assert!(!o.touched());
    }

    #[test]
    fn a_filled_rect_is_opaque_inside_and_untouched_outside() {
        let mut o = Overlay::new(32, 32);
        o.fill(Color::WHITE);
        o.no_stroke();
        o.rect(8.0, 8.0, 16.0, 16.0);
        assert!(o.touched());
        assert_eq!(at(&o, 16, 16), [255, 255, 255, 255], "centre");
        assert_eq!(at(&o, 2, 2), [0, 0, 0, 0], "outside stays clear");
        assert_eq!(at(&o, 30, 30), [0, 0, 0, 0], "outside stays clear");
    }

    /// The reason the rasteriser is a distance field: a half-pixel offset has
    /// to land as half coverage, not as a snapped edge.
    #[test]
    fn edges_are_anti_aliased_rather_than_snapped() {
        let mut o = Overlay::new(16, 16);
        o.fill(Color::WHITE);
        o.no_stroke();
        // Left edge exactly through the centre of column 4.
        o.rect(4.5, 0.0, 8.0, 16.0);
        let a = at(&o, 4, 8)[3];
        assert!(
            (a as i32 - 128).abs() <= 4,
            "an edge through a pixel centre should be about half covered, got {a}"
        );
        assert_eq!(at(&o, 6, 8)[3], 255, "well inside");
        assert_eq!(at(&o, 2, 8)[3], 0, "well outside");
    }

    /// Alpha has to be straight (non-premultiplied) and the colour has to
    /// survive at low alpha, because the compositing pass assumes both.
    #[test]
    fn alpha_is_straight_so_colour_survives_transparency() {
        let mut o = Overlay::new(8, 8);
        o.no_stroke();
        o.fill_alpha(Color::WHITE, 0.25);
        o.rect(0.0, 0.0, 8.0, 8.0);
        let p = at(&o, 4, 4);
        assert_eq!([p[0], p[1], p[2]], [255, 255, 255], "colour not premultiplied");
        assert!((p[3] as i32 - 64).abs() <= 2, "alpha {} should be ~64", p[3]);
    }

    /// Overlapping half-transparent fills must mix in linear light. Two 50%
    /// mid-greys over each other land on a different byte in each space, so
    /// this distinguishes them rather than just asserting the implementation.
    #[test]
    fn overlapping_fills_mix_in_linear_light() {
        let mut o = Overlay::new(8, 8);
        o.no_stroke();
        o.fill_alpha(Color::WHITE, 0.5);
        o.rect(0.0, 0.0, 8.0, 8.0);
        let under = at(&o, 4, 4);
        o.fill_alpha(Color::BLACK, 0.5);
        o.rect(0.0, 0.0, 8.0, 8.0);
        let got = at(&o, 4, 4);

        // The first layer's alpha is already a byte by the time the second one
        // reads it, and 0.5 rounds up to 128/255, so the stack lands a shade
        // above three quarters. Derived from what is actually in the buffer
        // rather than from the nominal halves, or this is off by one.
        let da = under[3] as f32 / 255.0;
        let out_a = 0.5 + da * 0.5;
        assert_eq!(got[3], (out_a * 255.0 + 0.5) as u8, "two 50% layers stack");

        // Black at 0.5 over white at `da`, mixed in linear light.
        let want = srgb_encode((0.0 * 0.5 + 1.0 * da * 0.5) / out_a);
        assert_eq!(got[0], want, "linear mix");
        // And it is not the answer sRGB-space blending would give, so the
        // assertion above is testing the space and not just the arithmetic.
        let naive = (0.0 * 0.5 + under[0] as f32 * da * 0.5) / out_a;
        assert!(
            (got[0] as f32 - naive).abs() > 8.0,
            "linear {} should differ from the sRGB-space mix {naive:.1}",
            got[0]
        );
    }

    #[test]
    fn a_circle_is_round_and_centred() {
        let mut o = Overlay::new(64, 64);
        o.fill(Color::WHITE);
        o.no_stroke();
        o.circle(32.0, 32.0, 20.0);
        assert_eq!(at(&o, 32, 32)[3], 255, "centre");
        for (x, y) in [(32u32, 14u32), (32, 50), (14, 32), (50, 32)] {
            assert_eq!(at(&o, x, y)[3], 255, "inside at ({x}, {y})");
        }
        for (x, y) in [(32u32, 8u32), (32, 56), (8, 32), (56, 32), (16, 16)] {
            assert_eq!(at(&o, x, y)[3], 0, "outside at ({x}, {y})");
        }
        // Area, as a check that the radius is a radius and not a diameter.
        let covered: f32 = o.pixels().chunks(4).map(|p| p[3] as f32 / 255.0).sum();
        let want = std::f32::consts::PI * 20.0 * 20.0;
        assert!(
            (covered - want).abs() < want * 0.02,
            "covered {covered:.1} against pi*r^2 = {want:.1}"
        );
    }

    #[test]
    fn an_unfilled_shape_draws_only_its_outline() {
        let mut o = Overlay::new(64, 64);
        o.no_fill();
        o.stroke(Color::WHITE);
        o.stroke_weight(4.0);
        o.circle(32.0, 32.0, 20.0);
        assert_eq!(at(&o, 32, 32)[3], 0, "the middle stays clear");
        assert_eq!(at(&o, 32, 12)[3], 255, "on the ring");
        assert_eq!(at(&o, 32, 2)[3], 0, "outside the ring");
        // A ring of width w at radius r has area 2*pi*r*w.
        let covered: f32 = o.pixels().chunks(4).map(|p| p[3] as f32 / 255.0).sum();
        let want = std::f32::consts::TAU * 20.0 * 4.0;
        assert!(
            (covered - want).abs() < want * 0.05,
            "covered {covered:.1} against 2*pi*r*w = {want:.1}"
        );
    }

    #[test]
    fn a_line_needs_a_stroke_and_ignores_the_fill() {
        let mut o = Overlay::new(32, 32);
        o.fill(Color::WHITE);
        o.no_stroke();
        o.line(2.0, 16.0, 30.0, 16.0);
        assert!(!o.touched(), "no stroke means no line");

        o.stroke(Color::WHITE);
        o.stroke_weight(2.0);
        o.line(2.0, 16.0, 30.0, 16.0);
        assert!(o.touched());
        assert_eq!(at(&o, 16, 16)[3], 255, "on the line");
        assert_eq!(at(&o, 16, 24)[3], 0, "off the line");
    }

    #[test]
    fn a_polygon_fills_its_inside_whichever_way_it_is_wound() {
        let square = [
            Vector2::new(8.0, 8.0),
            Vector2::new(24.0, 8.0),
            Vector2::new(24.0, 24.0),
            Vector2::new(8.0, 24.0),
        ];
        let mut reversed: Vec<Vector2> = square.to_vec();
        reversed.reverse();
        for pts in [square.to_vec(), reversed] {
            let mut o = Overlay::new(32, 32);
            o.fill(Color::WHITE);
            o.no_stroke();
            o.polygon(&pts);
            assert_eq!(at(&o, 16, 16)[3], 255, "inside");
            assert_eq!(at(&o, 2, 2)[3], 0, "outside");
            let covered: f32 = o.pixels().chunks(4).map(|p| p[3] as f32 / 255.0).sum();
            assert!(
                (covered - 256.0).abs() < 8.0,
                "a 16x16 square should cover ~256 px, got {covered:.1}"
            );
        }
    }

    /// A concave polygon is where a winding-sign distance field and a crossing
    /// count disagree, so the notch has to come out empty.
    #[test]
    fn a_concave_polygon_keeps_its_notch_empty() {
        // An L, 20x20 with a 10x10 bite out of the bottom-right.
        let l = [
            Vector2::new(4.0, 4.0),
            Vector2::new(24.0, 4.0),
            Vector2::new(24.0, 14.0),
            Vector2::new(14.0, 14.0),
            Vector2::new(14.0, 24.0),
            Vector2::new(4.0, 24.0),
        ];
        let mut o = Overlay::new(32, 32);
        o.fill(Color::WHITE);
        o.no_stroke();
        o.polygon(&l);
        assert_eq!(at(&o, 8, 8)[3], 255, "the top-left arm");
        assert_eq!(at(&o, 20, 8)[3], 255, "the top-right arm");
        assert_eq!(at(&o, 8, 20)[3], 255, "the bottom-left arm");
        assert_eq!(at(&o, 20, 20)[3], 0, "the notch stays empty");
        let covered: f32 = o.pixels().chunks(4).map(|p| p[3] as f32 / 255.0).sum();
        assert!(
            (covered - 300.0).abs() < 10.0,
            "20x20 less 10x10 is 300 px, got {covered:.1}"
        );
    }

    #[test]
    fn drawing_off_canvas_is_clipped_rather_than_a_panic() {
        let mut o = Overlay::new(16, 16);
        o.fill(Color::WHITE);
        o.stroke(Color::BLACK);
        o.rect(-100.0, -100.0, 40.0, 40.0);
        o.rect(1e6, 1e6, 10.0, 10.0);
        o.circle(-5.0, 8.0, 4.0);
        o.line(-50.0, -50.0, 500.0, 500.0);
        o.polygon(&[
            Vector2::new(-10.0, -10.0),
            Vector2::new(-20.0, -10.0),
            Vector2::new(-15.0, -20.0),
        ]);
        // Negative and zero sizes have no inside and must not wrap around.
        o.rect(4.0, 4.0, 0.0, 8.0);
        o.rect(4.0, 4.0, -8.0, -8.0);
        o.circle(8.0, 8.0, 0.0);
        o.polygon(&[Vector2::new(1.0, 1.0)]);
        assert_eq!(o.pixels().len(), 16 * 16 * 4);
    }

    /// The upload band has to describe the difference between the buffer and
    /// the texture, which is not the same as "what this frame drew": erasing a
    /// panel changes rows without drawing on them.
    #[test]
    fn the_dirty_band_tracks_what_the_texture_is_missing() {
        let mut o = Overlay::new(64, 200);
        assert_eq!(o.dirty_rows(), None, "nothing has happened yet");

        o.no_stroke();
        o.fill(Color::WHITE);
        o.rect(0.0, 20.0, 64.0, 20.0);
        let (a, b) = o.dirty_rows().expect("the rect is outstanding");
        assert!(a <= 20 && b >= 40, "rows {a}..{b} should cover 20..40");
        assert!(b - a < 30, "rows {a}..{b} is far more than the rect");

        o.mark_uploaded();
        assert_eq!(o.dirty_rows(), None, "acknowledged");

        // Erasing is a change. Without this the panel would stay on screen: the
        // buffer goes to zero and nothing would tell the texture.
        o.clear();
        let (a, b) = o.dirty_rows().expect("the cleared rows are outstanding");
        assert!(a <= 20 && b >= 40, "rows {a}..{b} should still cover 20..40");

        // Drawing somewhere else unions with it, because both bands differ from
        // the texture.
        o.rect(0.0, 150.0, 64.0, 20.0);
        let (a, b) = o.dirty_rows().expect("both");
        assert!(a <= 20 && b >= 170, "rows {a}..{b} should span 20..170");

        o.mark_uploaded();
        o.clear();
        let (a, b) = o.dirty_rows().expect("the second rect's rows");
        assert!(
            a >= 140 && b >= 170,
            "rows {a}..{b} should be the second rect only — the first was \
             already erased and acknowledged"
        );

        // A frame that draws nothing leaves the erase outstanding rather than
        // dropping it, so skipping a composite cannot lose it.
        o.clear();
        assert!(o.dirty_rows().is_some(), "still outstanding");
    }

    /// Text is drawn by the caption painter straight into the buffer, so the
    /// overlay has to ask it which rows it used rather than guess.
    #[cfg(feature = "captions")]
    #[test]
    fn text_marks_the_rows_it_actually_landed_on() {
        let mut o = Overlay::new(300, 200);
        o.text_size(20.0);
        o.fill(Color::WHITE);
        assert!(o.text(10.0, 80.0, "a label"));
        let (a, b) = o.dirty_rows().expect("the text is outstanding");
        // Found by scanning the buffer, which is the ground truth.
        let (mut first, mut last) = (u32::MAX, 0u32);
        for (i, px) in o.pixels().chunks(4).enumerate() {
            if px[3] > 0 {
                let row = i as u32 / 300;
                first = first.min(row);
                last = last.max(row);
            }
        }
        assert!(first != u32::MAX, "no ink");
        assert!(
            a <= first && b > last,
            "band {a}..{b} should contain the ink, which is in {first}..={last}"
        );
        assert!(b - a < 60, "band {a}..{b} is much wider than the line");
    }

    #[test]
    fn clear_and_resize_reset_the_canvas() {
        let mut o = Overlay::new(16, 16);
        o.fill(Color::WHITE);
        o.rect(0.0, 0.0, 16.0, 16.0);
        assert!(o.touched());
        o.clear();
        assert!(!o.touched());
        assert!(o.pixels().iter().all(|&b| b == 0));

        o.rect(0.0, 0.0, 16.0, 16.0);
        o.resize(8, 32);
        assert_eq!(o.size(), (8, 32));
        assert_eq!(o.pixels().len(), 8 * 32 * 4);
        assert!(o.pixels().iter().all(|&b| b == 0));
        assert!(!o.touched());

        // Same size is a no-op and keeps the contents.
        o.fill(Color::WHITE);
        o.rect(0.0, 0.0, 8.0, 32.0);
        o.resize(8, 32);
        assert!(o.touched(), "resizing to the same size should not clear");
    }

    /// Each alignment has to put the ink somewhere different, and in the right
    /// direction. Measured as the horizontal centre of mass of the ink, which
    /// needs no assumption about the font.
    #[cfg(feature = "captions")]
    #[test]
    fn text_alignment_moves_the_ink_the_right_way() {
        let centre_of_ink = |o: &Overlay| {
            let (w, _) = o.size();
            let (mut sum, mut weight) = (0.0f64, 0.0f64);
            for (i, px) in o.pixels().chunks(4).enumerate() {
                let a = px[3] as f64;
                if a > 0.0 {
                    sum += (i % w as usize) as f64 * a;
                    weight += a;
                }
            }
            assert!(weight > 0.0, "no ink");
            sum / weight
        };
        let draw = |f: &dyn Fn(&mut Overlay) -> bool| {
            let mut o = Overlay::new(400, 48);
            o.text_size(20.0);
            o.fill(Color::WHITE);
            assert!(f(&mut o));
            centre_of_ink(&o)
        };
        let anchor = 200.0f32;
        let left = draw(&|o| o.text(anchor, 8.0, "measure me"));
        let centre = draw(&|o| o.text_centered(anchor, 8.0, "measure me"));
        let right = draw(&|o| o.text_right(anchor, 8.0, "measure me"));

        assert!(
            right < centre && centre < left,
            "right {right:.1} should sit left of centred {centre:.1}, which sits \
             left of left-aligned {left:.1}"
        );
        // Left-aligned starts at the anchor, right-aligned ends at it, and
        // centred straddles it — so the centred one is about the midpoint of
        // the other two.
        assert!(
            ((left + right) * 0.5 - centre).abs() < 6.0,
            "centred {centre:.1} should be near the midpoint of {left:.1} and {right:.1}"
        );
        let anchor = anchor as f64;
        assert!(centre > anchor - 60.0 && centre < anchor + 60.0, "centred {centre:.1}");
    }

    #[cfg(feature = "captions")]
    #[test]
    fn text_marks_ink_and_measures_without_drawing() {
        let mut o = Overlay::new(256, 64);
        o.text_size(24.0);
        o.fill(Color::WHITE);
        let (w, h) = o.measure_text("Frame 120");
        assert!(w > 0.0 && h > 0.0, "measured {w}x{h}");
        assert!(!o.touched(), "measuring must not draw");

        assert!(o.text(8.0, 8.0, "Frame 120"));
        assert!(o.touched());
        let ink: usize = o.pixels().chunks(4).filter(|p| p[3] > 0).count();
        assert!(ink > 20, "only {ink} pixels of ink");

        // An empty string is not an error and leaves no ink.
        let mut blank = Overlay::new(64, 32);
        assert!(!blank.text(0.0, 0.0, "   "));
        assert!(!blank.touched());
    }
}
