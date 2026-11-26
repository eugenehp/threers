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
#[cfg(feature = "render")]
pub mod surface_view;
// When the UI draws: continuously, or only when something changed.
#[cfg(feature = "render")]
pub mod redraw;
// Text inputs' copy and paste through the system clipboard.
#[cfg(all(
    feature = "clipboard",
    any(target_os = "macos", target_os = "windows", target_os = "linux", target_os = "freebsd")
))]
pub mod system_clipboard;
// A UI recorded as a drawing: SVG, PDF and PNG with no GPU.
pub mod vector;

use crate::math::Color;
use crate::overlay::Overlay;

/// `ui` as last laid out, painted on the CPU into a `width × height` image
/// at `scale` device pixels per UI pixel, over transparency: RGBA8,
/// sRGB-encoded, straight alpha, rows from the top. For screenshots and
/// golden-image tests without a GPU; [`UiPass::render_to_rgba`](crate::renderer::UiPass)
/// is the GPU's.
pub fn paint_to_rgba(ui: &Ui, scale: f32, width: u32, height: u32) -> Vec<u8> {
    paint_to_rgba_in(ui, scale, width, height, crate::overlay::BlendSpace::Linear)
}

/// [`paint_to_rgba`], see-through colours mixed in `space` (as CSS mixes
/// them, with [`BlendSpace::Srgb`](crate::overlay::BlendSpace::Srgb)).
pub fn paint_to_rgba_in(
    ui: &Ui,
    scale: f32,
    width: u32,
    height: u32,
    space: crate::overlay::BlendSpace,
) -> Vec<u8> {
    let mut overlay = Overlay::new(width.max(1), height.max(1));
    overlay.set_blend_space(space);
    ui.paint(&mut OverlayPainter::with_scale(&mut overlay, scale));
    overlay.pixels().to_vec()
}

/// An sRGB `0..=1` channel as `space` mixes it.
fn to_space(space: crate::overlay::BlendSpace, c: f32) -> f32 {
    match space {
        crate::overlay::BlendSpace::Linear => threers_math::srgb::srgb_to_linear_f32(c),
        crate::overlay::BlendSpace::Srgb => c,
    }
}

/// An sRGB-encoded byte as `space` mixes it.
fn byte_to_space(space: crate::overlay::BlendSpace, b: u8) -> f32 {
    match space {
        crate::overlay::BlendSpace::Linear => threers_math::srgb::srgb_to_linear(b),
        crate::overlay::BlendSpace::Srgb => b as f32 / 255.0,
    }
}

/// A value mixed in `space` back to an sRGB-encoded byte.
fn space_to_byte(space: crate::overlay::BlendSpace, v: f32) -> u8 {
    match space {
        crate::overlay::BlendSpace::Linear => threers_math::srgb::linear_to_srgb(v),
        crate::overlay::BlendSpace::Srgb => (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
    }
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
    /// Beside each clip, its rounded corners (its box and radius, in overlay
    /// pixels) when it has them.
    rounds: Vec<Option<(Rect, f32)>>,
    /// What is in a turned box, kept until its end: then drawn flat on a
    /// canvas of its own, and that through the transform.
    turned: Option<Turned>,
}

struct Turned {
    scene: crate::ui::scene::Scene,
    /// Where the box's points are seen, both in UI pixels.
    h: [f32; 9],
    /// The transforms inside it.
    inside: usize,
}

impl<'a> OverlayPainter<'a> {
    pub fn new(overlay: &'a mut Overlay) -> Self {
        Self {
            overlay,
            scale: 1.0,
            clips: Vec::new(),
            rounds: Vec::new(),
            turned: None,
        }
    }

    /// Paints a UI laid out in logical pixels onto an overlay `scale` times
    /// as dense.
    pub fn with_scale(overlay: &'a mut Overlay, scale: f32) -> Self {
        // SF Pro's sizes are points, as the rest of iOS draws it.
        if cfg!(any(target_os = "ios", target_os = "tvos", target_os = "visionos")) {
            overlay.text_optical_scale(scale);
        }
        Self {
            overlay,
            scale,
            clips: Vec::new(),
            rounds: Vec::new(),
            turned: None,
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
        // A see-through one (an opaque one is written, not mixed, and quick).
        #[cfg(all(metal_backend, feature = "ui"))]
        if r.width * r.height >= crate::metal::warp::WORTH
            && image.rgba.len() <= 16 << 20
            && (opacity < 1.0 || image.rgba.chunks_exact(4).any(|p| p[3] < 255))
        {
            let what = crate::metal::warp::Fill::Image {
                rect: [r.x, r.y, r.width, r.height],
                rgba: &image.rgba,
                size: (image.width, image.height),
                opacity,
            };
            let (fw, fh) = self.overlay.size();
            let (cx0, cy0, cx1, cy1) = clip.unwrap_or((0.0, 0.0, fw as f32, fh as f32));
            let (x0, y0) = (r.x.max(cx0).max(0.0), r.y.max(cy0).max(0.0));
            let (x1, y1) = ((r.x + r.width).min(cx1), (r.y + r.height).min(cy1));
            let at = (x0.floor() as i32, y0.floor() as i32, x1.ceil() as i32, y1.ceil() as i32);
            if self.fill_on_gpu(&what, r.width * r.height, at, [x0, y0, x1, y1], false) {
                return;
            }
        }
        // (a large one drawn as it was a moment ago, moved or not: scaled once)
        let kept = if opacity >= 1.0 { scaled(r, image) } else { None };
        self.overlay.image_from(
            r.x,
            r.y,
            r.width,
            r.height,
            &image.rgba,
            image.width,
            image.height,
            opacity,
            clip,
            kept.as_ref().map(|(k, x, y)| (&k.0, *x, *y)),
        );
    }
}

/// A large opaque image, drawn to a size at a fraction of a pixel.
struct Scaled(Overlay);

/// Which: the image, its size (device pixels, bits), and the fraction of a
/// pixel it's at (64ths).
type ScaledKey = (u64, u32, u32, i32, i32);

thread_local! {
    /// Those drawn lately, the latest last: each once more than once.
    static SCALED: std::cell::RefCell<Vec<(ScaledKey, Option<std::rc::Rc<Scaled>>)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// From this many device pixels an image is worth keeping scaled, and no
/// more than so many of them in all.
const SCALED_LEAST: f32 = 60_000.0;
const SCALED_MOST: usize = 12_000_000;

/// `image` drawn at `r` as it was, if it has been (at this size and fraction
/// of a pixel), and where its canvas's corner goes: sliding it, or painting
/// over it again, is then a copy of its rows, not a sample at every pixel.
fn scaled(r: Rect, image: &image::Image) -> Option<(std::rc::Rc<Scaled>, i32, i32)> {
    let area = r.width * r.height;
    if !(area >= SCALED_LEAST && area <= SCALED_MOST as f32 / 2.0) {
        return None;
    }
    let (qx, qy) = ((r.x * 64.0).round() as i32, (r.y * 64.0).round() as i32);
    let (bx, by) = (qx.div_euclid(64), qy.div_euclid(64));
    let (fx, fy) = (qx.rem_euclid(64), qy.rem_euclid(64));
    let key: ScaledKey = (image.id(), r.width.to_bits(), r.height.to_bits(), fx, fy);
    SCALED.with_borrow_mut(|all| {
        let had = all.iter().position(|e| e.0 == key).map(|i| all.remove(i));
        let Some((_, had)) = had else {
            // its first time: remembered, not kept (a zoom's every step is one)
            all.push((key, None));
            if all.len() > 12 {
                all.remove(0);
            }
            return None;
        };
        let had = match had {
            Some(s) => Some(s),
            None if image.rgba.chunks_exact(4).all(|p| p[3] == 255) => {
                let (x, y) = (fx as f32 / 64.0, fy as f32 / 64.0);
                let (w, h) = ((x + r.width).ceil() as u32 + 1, (y + r.height).ceil() as u32 + 1);
                let mut o = Overlay::new(w, h);
                o.image(x, y, r.width, r.height, &image.rgba, image.width, image.height, 1.0, None);
                Some(std::rc::Rc::new(Scaled(o)))
            }
            // (one that's seen through: as it is each time)
            None => return None,
        };
        all.push((key, had.clone()));
        let size = |e: &(ScaledKey, Option<std::rc::Rc<Scaled>>)| e.1.as_ref().map_or(0, |s| s.0.pixels().len() / 4);
        while all.iter().map(size).sum::<usize>() > SCALED_MOST && all.len() > 1 {
            all.remove(0);
        }
        had.map(|s| (s, bx, by))
    })
}

/// [`OverlayPainter::round_cover`], for these clips.
fn round_cover(rounds: &[Option<(Rect, f32)>], x: f32, y: f32) -> f32 {
    rounds.iter().flatten().fold(1.0, |cover, &(r, radius)| {
        let dx = (r.x + radius - x).max(x - (r.x + r.width - radius)).max(0.0);
        let dy = (r.y + radius - y).max(y - (r.y + r.height - radius)).max(0.0);
        if dx > 0.0 && dy > 0.0 {
            cover * (radius - (dx * dx + dy * dy).sqrt() + 0.5).clamp(0.0, 1.0)
        } else {
            cover
        }
    })
}

impl OverlayPainter<'_> {
    /// How much of the pixel centred at `(x, y)` (overlay pixels) the
    /// rounded clips leave: 1 away from their corners, fading across each
    /// corner's arc.
    fn round_cover(&self, x: f32, y: f32) -> f32 {
        round_cover(&self.rounds, x, y)
    }
}

/// `r`, `by` larger each way.
fn grown(r: Rect, by: f32) -> Rect {
    Rect { x: r.x - by, y: r.y - by, width: r.width + 2.0 * by, height: r.height + 2.0 * by }
}

/// Where a line of text may draw: its glyphs reach past its box.
fn text_reach(line: Rect, style: &TextStyle) -> Rect {
    let reach = grown(line, style.font_size);
    // (a run can be wider than its box says)
    Rect { width: reach.width * 1.25, ..reach }
}

#[cfg(all(metal_backend, feature = "ui"))]
impl OverlayPainter<'_> {
    fn rounded_clips(&self) -> Vec<([f32; 4], f32)> {
        self.rounds.iter().flatten().map(|&(r, radius)| ([r.x, r.y, r.width, r.height], radius)).collect()
    }

    /// `what` by the GPU, if there is one and `area` (all of it, in pixels,
    /// whatever part is drawn now: a part drawn again is as it was) is worth
    /// it: within the pixels `x0, y0, x1, y1`, as far as `clip` covers each.
    fn fill_on_gpu(&mut self, what: &crate::metal::warp::Fill, area: f32, (x0, y0, x1, y1): (i32, i32, i32, i32), clip: [f32; 4], rounded: bool) -> bool {
        use crate::metal::warp;
        if area < warp::WORTH {
            return false;
        }
        let (fw, fh) = self.overlay.size();
        let (wx0, wy0, wx1, wy1) = self.overlay.window_or_all();
        let (x0, y0) = (x0.max(wx0 as i32), y0.max(wy0 as i32));
        let (x1, y1) = (x1.min(wx1.min(fw) as i32), y1.min(wy1.min(fh) as i32));
        if x1 <= x0 || y1 <= y0 {
            // (drawn: there is nothing of it)
            return true;
        }
        let (bw, bh) = ((x1 - x0) as usize, (y1 - y0) as usize);
        let mut buf = vec![0u8; bw * bh * 4];
        let under = self.overlay.pixels();
        for (row, to) in buf.chunks_exact_mut(bw * 4).enumerate() {
            let from = ((y0 as usize + row) * fw as usize + x0 as usize) * 4;
            to.copy_from_slice(&under[from..from + bw * 4]);
        }
        let space = self.overlay.blend_space();
        let decode: [f32; 256] = std::array::from_fn(|i| byte_to_space(space, i as u8));
        let rounds = if rounded { self.rounded_clips() } else { Vec::new() };
        let onto = warp::Onto {
            clip,
            rounds: &rounds,
            at: (x0, y0),
            size: (bw as u32, bh as u32),
            decode: &decode,
            linear: space == crate::overlay::BlendSpace::Linear,
        };
        if !warp::fill(what, &onto, &mut buf) {
            return false;
        }
        self.overlay.put_image_data(x0, y0, bw as u32, bh as u32, &buf);
        true
    }

    /// A shadow by the GPU: its box and the hole in overlay pixels.
    fn shadow_on_gpu(&mut self, r: Rect, radius: f32, sigma: f32, color: Rgba, hole: Option<([f32; 4], f32)>) -> bool {
        // (a sharp one is a box)
        if sigma < 0.5 || r.width <= 0.0 || r.height <= 0.0 || color[3] <= 0.0 {
            return false;
        }
        let reach = 3.0 * sigma;
        let all = grown(r, reach);
        let space = self.overlay.blend_space();
        let what = crate::metal::warp::Fill::Shadow {
            rect: [r.x, r.y, r.x + r.width, r.y + r.height],
            sigma,
            radius,
            hole,
            color: [to_space(space, color[0]), to_space(space, color[1]), to_space(space, color[2]), color[3]],
        };
        let (mut x0, mut y0) = (all.x.floor() as i32, all.y.floor() as i32);
        let (mut x1, mut y1) = ((all.x + all.width).ceil() as i32, (all.y + all.height).ceil() as i32);
        // cut as `narrow_window` cuts it, to the nearest pixel
        if let Some(c) = self.clips.last() {
            (x0, y0) = (x0.max(c.x.round() as i32), y0.max(c.y.round() as i32));
            (x1, y1) = (x1.min((c.x + c.width).round() as i32), y1.min((c.y + c.height).round() as i32));
        }
        let anywhere = [-1e9, -1e9, 1e9, 1e9];
        self.fill_on_gpu(&what, all.width * all.height, (x0, y0, x1, y1), anywhere, false)
    }
}

/// The pixels rounded clips cut, as they were: where, how much of what is
/// drawn there is to stay, and the pixel.
type Corners = Vec<(i32, i32, f32, [u8; 4])>;

impl OverlayPainter<'_> {
    /// The pixels of `r` (overlay pixels) the rounded clips cut, before
    /// something is drawn there that knows only their boxes.
    fn keep_corners(&self, r: Rect) -> Corners {
        let (fw, fh) = self.overlay.size();
        let (wx0, wy0, wx1, wy1) = self.overlay.window_or_all();
        let px = self.overlay.pixels();
        let mut out = Corners::new();
        let mut seen: Vec<[i32; 4]> = Vec::new();
        for &(c, radius) in self.rounds.iter().flatten() {
            let (right, bottom) = (c.x + c.width - radius, c.y + c.height - radius);
            for (cx, cy) in [(c.x, c.y), (right, c.y), (c.x, bottom), (right, bottom)] {
                let x0 = (cx.max(r.x).floor() as i32).max(wx0 as i32);
                let y0 = (cy.max(r.y).floor() as i32).max(wy0 as i32);
                let x1 = ((cx + radius).min(r.x + r.width).ceil() as i32).min(wx1.min(fw) as i32);
                let y1 = ((cy + radius).min(r.y + r.height).ceil() as i32).min(wy1.min(fh) as i32);
                for y in y0..y1 {
                    for x in x0..x1 {
                        // (one corner in another's: once)
                        if seen.iter().any(|s| x >= s[0] && y >= s[1] && x < s[2] && y < s[3]) {
                            continue;
                        }
                        let cover = self.round_cover(x as f32 + 0.5, y as f32 + 0.5);
                        if cover < 1.0 {
                            let i = (y as usize * fw as usize + x as usize) * 4;
                            out.push((x, y, cover, [px[i], px[i + 1], px[i + 2], px[i + 3]]));
                        }
                    }
                }
                seen.push([x0, y0, x1, y1]);
            }
        }
        out
    }

    /// When rounded clips cut something drawn within `r` (overlay pixels):
    /// it is to be drawn as if they did not, then [`uncut`](Self::uncut).
    fn cut(&mut self, r: Rect) -> Option<(Corners, Vec<Option<(Rect, f32)>>)> {
        if self.rounds.iter().all(Option::is_none) {
            return None;
        }
        let corners = self.keep_corners(r);
        if corners.is_empty() {
            return None;
        }
        let rounds = vec![None; self.rounds.len()];
        Some((corners, std::mem::replace(&mut self.rounds, rounds)))
    }

    fn uncut(&mut self, (corners, rounds): (Corners, Vec<Option<(Rect, f32)>>)) {
        self.rounds = rounds;
        self.put_back(corners);
    }

    /// What was drawn over `corners` kept only as far as the clips leave it.
    fn put_back(&mut self, corners: Corners) {
        let space = self.overlay.blend_space();
        let fw = self.overlay.size().0 as usize;
        for (x, y, cover, was) in corners {
            let i = (y as usize * fw + x as usize) * 4;
            let p = self.overlay.pixels();
            let now = [p[i], p[i + 1], p[i + 2], p[i + 3]];
            if now == was {
                continue;
            }
            let (a0, a1) = (was[3] as f32 / 255.0 * (1.0 - cover), now[3] as f32 / 255.0 * cover);
            let a = a0 + a1;
            let mut out = [0u8; 4];
            if a > 0.0 {
                for n in 0..3 {
                    out[n] = space_to_byte(space, (byte_to_space(space, was[n]) * a0 + byte_to_space(space, now[n]) * a1) / a);
                }
                out[3] = (a * 255.0).round() as u8;
            }
            self.overlay.put_image_data(x, y, 1, 1, &out);
        }
    }
}

impl TextMeasure for OverlayPainter<'_> {
    fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
        self.overlay.text_size(style.font_size * self.scale);
        // (`TextStyle::of(style).weight()`, without copying the family's name)
        let weight = match style.font_weight {
            threers_ui::FontWeight::Normal => 400,
            threers_ui::FontWeight::Bold => 700,
            threers_ui::FontWeight::Weight(w) => w,
        };
        let real = self.overlay.text_weight(weight);
        self.overlay.text_italic(style.italic);
        self.overlay
            .text_letter_spacing(style.letter_spacing * self.scale);
        // The caption layout trims a line's ends, as subtitles want; a UI
        // run ending in a space (before the next run) needs that space. So
        // such a run is measured between two bars, and the bars taken off.
        let w = if text.starts_with(char::is_whitespace) || text.ends_with(char::is_whitespace) {
            // (each bar brings one more spacing between characters)
            let bar = self.overlay.measure_line("|").0;
            let spacing = 2.0 * style.letter_spacing * self.scale;
            self.overlay.measure_line(&format!("|{text}|")).0 - 2.0 * bar - spacing
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
        if let Some(t) = &mut self.turned {
            return t.scene.recorder().fill_gradient(rect, shape, stops, blur);
        }
        let space = self.overlay.blend_space();
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
        #[cfg(all(metal_backend, feature = "ui"))]
        {
            let what = crate::metal::warp::Fill::Gradient { rect: [r.x, r.y, r.width, r.height], shape, stops };
            if self.fill_on_gpu(&what, r.width * r.height, (bx0, by0, bx1, by1), [x0, y0, x1, y1], true) {
                return;
            }
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
                let sa = c[3] * cover * self.round_cover(x + 0.5, y + 0.5);
                if sa <= 0.0 {
                    continue;
                }
                let px = &mut buf[(row * bw + col) * 4..][..4];
                let da = px[3] as f32 / 255.0;
                let out_a = sa + da * (1.0 - sa);
                for i in 0..3 {
                    let d = byte_to_space(space, px[i]);
                    px[i] = space_to_byte(space, 
                        (to_space(space, c[i]) * sa + d * da * (1.0 - sa)) / out_a.max(1e-6),
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
        if let Some(t) = &mut self.turned {
            return t.scene.recorder().stroke_arc(arc, width, color);
        }
        if let Some(corners) = self.cut(self.px(arc.bounds(width / 2.0 + 1.0))) {
            self.stroke_arc(arc, width, color);
            return self.uncut(corners);
        }
        let space = self.overlay.blend_space();
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
        let lin = [0, 1, 2].map(|i| to_space(space, color[i]));
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
                    let d = byte_to_space(space, px[i]);
                    px[i] = space_to_byte(space, (lin[i] * sa + d * da * (1.0 - sa)) / out_a.max(1e-6));
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
        if let Some(t) = &mut self.turned {
            return t.scene.recorder().fill_linear_gradient(rect, stops, _blur);
        }
        let space = self.overlay.blend_space();
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
        #[cfg(all(metal_backend, feature = "ui"))]
        {
            let shape = GradientShape::Linear(90.0);
            let what = crate::metal::warp::Fill::Gradient { rect: [r.x, r.y, r.width, r.height], shape, stops };
            // (a column is in or out, by its middle)
            let clip = [(x0 - 0.5).ceil(), y0, (x1 - 0.5).ceil(), y1];
            if self.fill_on_gpu(&what, r.width * r.height, (bx0, by0, bx1, by1), clip, true) {
                return;
            }
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
            let lin = [0, 1, 2].map(|i| to_space(space, c[i]));
            for row in 0..bh {
                let y = (by0 + row as i32) as f32;
                let cover = (y1.min(y + 1.0) - y0.max(y)).clamp(0.0, 1.0);
                let sa = c[3] * cover * self.round_cover(centre, y + 0.5);
                if sa <= 0.0 {
                    continue;
                }
                let px = &mut buf[(row * bw + col) * 4..][..4];
                let da = px[3] as f32 / 255.0;
                let out_a = sa + da * (1.0 - sa);
                for i in 0..3 {
                    let d = byte_to_space(space, px[i]);
                    px[i] = space_to_byte(space, (lin[i] * sa + d * da * (1.0 - sa)) / out_a.max(1e-6));
                }
                px[3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        self.overlay
            .put_image_data(bx0, by0, bw as u32, bh as u32, &buf);
    }

    fn fill_rect(&mut self, rect: Rect, color: Rgba) {
        if let Some(t) = &mut self.turned {
            return t.scene.recorder().fill_rect(rect, color);
        }
        if let Some(corners) = self.cut(self.px(rect)) {
            self.fill_rect(rect, color);
            return self.uncut(corners);
        }
        let Some(r) = self.clipped(self.px(rect)) else {
            return;
        };
        let o = &mut *self.overlay;
        o.no_stroke();
        o.fill_alpha(linear(color), color[3]);
        o.rect(r.x, r.y, r.width, r.height);
    }

    fn fill_rounded_rect(&mut self, rect: Rect, rx: f32, ry: f32, color: Rgba) {
        if let Some(t) = &mut self.turned {
            return t.scene.recorder().fill_rounded_rect(rect, rx, ry, color);
        }
        if let Some(corners) = self.cut(self.px(rect)) {
            self.fill_rounded_rect(rect, rx, ry, color);
            return self.uncut(corners);
        }
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
        if let Some(t) = &mut self.turned {
            return t.scene.recorder().push_clip(rect);
        }
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
        self.rounds.push(None);
    }

    fn push_clip_rounded(&mut self, rect: Rect, rx: f32, ry: f32) {
        if let Some(t) = &mut self.turned {
            return t.scene.recorder().push_clip_rounded(rect, rx, ry);
        }
        self.push_clip(rect);
        let (radius, at) = (rx.min(ry) * self.scale, self.px(rect));
        if radius > 0.0
            && let Some(last) = self.rounds.last_mut()
        {
            *last = Some((at, radius));
        }
    }

    fn pop_clip(&mut self) {
        if let Some(t) = &mut self.turned {
            return t.scene.recorder().pop_clip();
        }
        self.clips.pop();
        self.rounds.pop();
    }

    fn text(&mut self, line: Rect, text: &str, style: &TextStyle) {
        if let Some(t) = &mut self.turned {
            return t.scene.recorder().text(line, text, style);
        }
        if let Some(corners) = self.cut(self.px(text_reach(line, style))) {
            self.text(line, text, style);
            return self.uncut(corners);
        }
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
        o.text_letter_spacing(style.letter_spacing * k);
        // The caption layout drops a line's leading spaces; a run that
        // starts with one (after the run before it) begins that much
        // further on.
        // (right to left, what is first across is what is typed last)
        let rtl = threers_ui::bidi::has_rtl(text)
            && threers_ui::bidi::base(text, Default::default())
                == threers_ui::environment::LayoutDirection::RightToLeft;
        let trimmed = text.trim_start();
        let lead = if rtl {
            &text[text.trim_end().len()..]
        } else {
            &text[..text.len() - trimmed.len()]
        };
        if !lead.is_empty() {
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
        // the frame's own edges clip too: the caption painter fits a line
        // to the room between its start and the frame's right edge, so one
        // running off the screen (a bar dragged half off it) was wrapped
        // there, a letter at a time, under itself
        let (fw, fh) = o.size();
        let frame = Rect { x: 0.0, y: 0.0, width: fw as f32, height: fh as f32 };
        let clip = Some(match clip {
            Some(c) => {
                let (x0, y0) = (c.x.max(0.0), c.y.max(0.0));
                let (x1, y1) = ((c.x + c.width).min(frame.width), (c.y + c.height).min(frame.height));
                Rect { x: x0, y: y0, width: (x1 - x0).max(0.0), height: (y1 - y0).max(0.0) }
            }
            None => frame,
        });
        if let Some(c) = clip {
            let mid = line.y + line.height / 2.0;
            if mid < c.y || mid > c.y + c.height || line.x >= c.x + c.width {
                return;
            }
            let chars: Vec<char> = text.chars().collect();
            let (mut start, mut end) = (0, chars.len());
            while start < end && line.x < c.x - 0.5 {
                let first = if rtl { chars[end - 1] } else { chars[start] };
                line.x += o.measure_line(&first.to_string()).0;
                if rtl {
                    end -= 1;
                } else {
                    start += 1;
                }
            }
            while end > start {
                owned = chars[start..end].iter().collect::<String>();
                if line.x + o.measure_line(&owned).0 <= c.x + c.width + 0.5 {
                    break;
                }
                if rtl {
                    start += 1;
                } else {
                    end -= 1;
                }
            }
            owned = chars[start..end].iter().collect::<String>();
            if owned.is_empty() {
                return;
            }
            // (it reads as it did whole)
            if rtl {
                owned = bidi::marked(&owned, environment::LayoutDirection::RightToLeft).into_owned();
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
        if let Some(t) = &mut self.turned {
            return t.scene.recorder().image(rect, image, opacity);
        }
        let r = self.px(rect);
        let corners = self.keep_corners(r);
        self.draw_image(rect, image, opacity);
        self.put_back(corners);
    }

    fn shadow(&mut self, rect: Rect, radius: f32, blur: f32, color: Rgba) {
        if let Some(t) = &mut self.turned {
            return t.scene.recorder().shadow(rect, radius, blur, color);
        }
        if let Some(corners) = self.cut(self.px(grown(rect, 1.5 * blur + 2.0))) {
            self.shadow(rect, radius, blur, color);
            return self.uncut(corners);
        }
        let r = self.px(rect);
        if self.clipped(r).is_none() && blur <= 0.0 {
            return;
        }
        let k = self.scale;
        #[cfg(all(metal_backend, feature = "ui"))]
        if self.shadow_on_gpu(r, radius * k, blur * k / 2.0, color, None) {
            return;
        }
        let clip = self.clips.last().copied();
        let o = &mut *self.overlay;
        let window = o.narrow_window(clip);
        o.no_stroke();
        o.fill_alpha(linear(color), color[3]);
        o.shadow(r.x, r.y, r.width, r.height, radius * k, blur * k / 2.0);
        o.set_window(window);
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
        if let Some(t) = &mut self.turned {
            return t.scene.recorder().shadow_around(rect, radius, blur, color, hole, hole_radius);
        }
        if let Some(corners) = self.cut(self.px(grown(rect, 1.5 * blur + 2.0))) {
            self.shadow_around(rect, radius, blur, color, hole, hole_radius);
            return self.uncut(corners);
        }
        let (r, h) = (self.px(rect), self.px(hole));
        let k = self.scale;
        #[cfg(all(metal_backend, feature = "ui"))]
        if self.shadow_on_gpu(r, radius * k, blur * k / 2.0, color, Some(([h.x, h.y, h.width, h.height], hole_radius * k))) {
            return;
        }
        let clip = self.clips.last().copied();
        let o = &mut *self.overlay;
        let window = o.narrow_window(clip);
        o.no_stroke();
        o.fill_alpha(linear(color), color[3]);
        let hole = (h.x, h.y, h.width, h.height, hole_radius * k);
        o.shadow_around(r.x, r.y, r.width, r.height, radius * k, blur * k / 2.0, Some(hole));
        o.set_window(window);
    }

    fn underline(&mut self, x: f32, y: f32, width: f32, thickness: f32, color: Rgba, wavy: bool) {
        if let Some(t) = &mut self.turned {
            return t.scene.recorder().underline(x, y, width, thickness, color, wavy);
        }
        if let Some(corners) = self.cut(self.px(Rect { x, y: y - 3.0 * thickness, width, height: 6.0 * thickness })) {
            self.underline(x, y, width, thickness, color, wavy);
            return self.uncut(corners);
        }
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
        if let Some(t) = &mut self.turned {
            return t.scene.recorder().path(rect, d, view_box, stroke, fill, color);
        }
        if let Some(corners) = self.cut(self.px(grown(rect, stroke.unwrap_or(0.0) * (rect.width / view_box[2].max(1e-6)).max(rect.height / view_box[3].max(1e-6)) + 2.0))) {
            self.path(rect, d, view_box, stroke, fill, color);
            return self.uncut(corners);
        }
        let rect = self.px(rect);
        if self.clipped(rect).is_none() {
            return;
        }
        // a small one (an icon): drawn once, then its picture put there
        if rect.width.max(rect.height) <= ICON && color[3] > 0.0 {
            let clip = self.clips.last().map(|c| (c.x, c.y, c.x + c.width, c.y + c.height));
            let (x0, y0) = (rect.x.floor(), rect.y.floor());
            let at = Rect { x: rect.x - x0 + PAD, y: rect.y - y0 + PAD, ..rect };
            let icon = icon(at, d, view_box, stroke, fill, color);
            let (w, h) = (icon.0 as f32, icon.1 as f32);
            self.overlay.image(x0 - PAD, y0 - PAD, w, h, &icon.2, icon.0, icon.1, 1.0, clip);
            return;
        }
        draw_path(self.overlay, rect, d, view_box, stroke, fill, color);
    }

    fn push_transform(&mut self, m: [f32; 6]) {
        self.push_projective([m[0], m[2], m[4], m[1], m[3], m[5], 0.0, 0.0, 1.0], m);
    }

    /// In true perspective: what follows is kept, and drawn through `h` at
    /// the end of it.
    fn push_projective(&mut self, h: [f32; 9], affine: [f32; 6]) {
        match &mut self.turned {
            Some(t) => {
                t.inside += 1;
                t.scene.recorder().push_projective(h, affine);
            }
            None => self.turned = Some(Turned { scene: Default::default(), h, inside: 0 }),
        }
    }

    fn pop_transform(&mut self) {
        match &mut self.turned {
            Some(t) if t.inside > 0 => {
                t.inside -= 1;
                t.scene.recorder().pop_transform();
            }
            _ => {
                if let Some(t) = self.turned.take() {
                    self.draw_turned(t);
                }
            }
        }
    }
}

/// The largest canvas a turned box is drawn on, a side.
const TURNED_MAX: f32 = 4096.0;

impl OverlayPainter<'_> {
    /// A turned box's insides: drawn flat on a canvas of their own, as large
    /// as they are seen, then each pixel of the overlay they reach looked up
    /// on it through the transform.
    fn draw_turned(&mut self, t: Turned) {
        let k = self.scale;
        let (fw, fh) = self.overlay.size();
        let far = 4.0 * (fw.max(fh) as f32 / k + 1000.0);
        let all = Rect { x: -far, y: -far, width: 2.0 * far, height: 2.0 * far };
        let Some(b) = t.scene.bounds(all) else { return };
        let h = t.h;
        let w_at = |x: f32, y: f32| h[6] * x + h[7] * y + h[8];
        let corners = [(b.x, b.y), (b.x + b.width, b.y), (b.x + b.width, b.y + b.height), (b.x, b.y + b.height)];
        // where it is seen, in overlay pixels: anywhere, if it reaches
        // behind the eye
        let mut seen = (0.0f32, 0.0f32, fw as f32, fh as f32);
        let mut stretch = 1.0f32;
        if corners.iter().all(|&(x, y)| w_at(x, y) > 1e-4) {
            let q = corners.map(|(x, y)| crate::ui::depth::apply_homography(h, x, y)).map(|(x, y)| (x * k, y * k));
            let lo = |f: fn(&(f32, f32)) -> f32| q.iter().map(f).fold(f32::MAX, f32::min);
            let hi = |f: fn(&(f32, f32)) -> f32| q.iter().map(f).fold(f32::MIN, f32::max);
            seen = (lo(|p| p.0), lo(|p| p.1), hi(|p| p.0), hi(|p| p.1));
            // how much larger than flat, along its longest-drawn side
            let side = |i: usize, j: usize, flat: f32| ((q[i].0 - q[j].0).hypot(q[i].1 - q[j].1)) / (flat * k).max(1e-3);
            stretch = side(0, 1, b.width).max(side(2, 3, b.width)).max(side(1, 2, b.height)).max(side(3, 0, b.height));
        }
        let (wx0, wy0, wx1, wy1) = self.overlay.window_or_all();
        let mut to = (wx0 as f32, wy0 as f32, wx1 as f32, wy1 as f32);
        to = (to.0.max(seen.0 - 1.0), to.1.max(seen.1 - 1.0), to.2.min(seen.2 + 1.0), to.3.min(seen.3 + 1.0));
        if let Some(c) = self.clips.last() {
            to = (to.0.max(c.x), to.1.max(c.y), to.2.min(c.x + c.width), to.3.min(c.y + c.height));
        }
        let (x0, y0) = (to.0.floor().max(0.0) as i32, to.1.floor().max(0.0) as i32);
        let (x1, y1) = (to.2.ceil().min(fw as f32) as i32, to.3.ceil().min(fh as f32) as i32);
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        // The canvas: a pixel of it about one of the overlay's (as fine, at
        // least, as flat; no more than twice), a pixel's room round it.
        let most = (TURNED_MAX / (b.width.max(b.height) * k).max(1.0)).min(2.0);
        let s = k * stretch.max(1.0).min(most);
        let origin = ((b.x * s).floor() - 1.0, (b.y * s).floor() - 1.0);
        let lw = ((b.x + b.width) * s - origin.0).ceil().max(1.0) as u32 + 1;
        let lh = ((b.y + b.height) * s - origin.1).ceil().max(1.0) as u32 + 1;
        let mut layer = Overlay::new(lw, lh);
        let space = self.overlay.blend_space();
        layer.set_blend_space(space);
        #[cfg(feature = "captions")]
        layer.swap_text(self.overlay);
        let mut scene = t.scene;
        scene.translate(-origin.0 / s, -origin.1 / s);
        // (not `with_scale`: text keeps the optical sizes it has here)
        let mut painter = OverlayPainter::new(&mut layer);
        painter.scale = s;
        scene.replay(&mut painter);
        #[cfg(feature = "captions")]
        layer.swap_text(self.overlay);

        // From overlay pixels to the canvas's.
        let dev = [
            h[0] * k, h[1] * k, h[2] * k * s, //
            h[3] * k, h[4] * k, h[5] * k * s, //
            h[6], h[7], h[8] * s,
        ];
        let Some(inv) = inverse3(dev) else { return };
        let table: [f32; 256] = std::array::from_fn(|i| byte_to_space(space, i as u8));
        // As many looks at the canvas a pixel as it has of its pixels there,
        // each way (one, at most, where it is as fine as the overlay).
        // (asked at the middle of where it is seen, whatever part is drawn now)
        let (mx, my) = ((seen.0 + seen.2) / 2.0, (seen.1 + seen.3) / 2.0);
        let on_canvas = |x: f32, y: f32| {
            let w = inv[6] * x + inv[7] * y + inv[8];
            ((inv[0] * x + inv[1] * y + inv[2]) / w, (inv[3] * x + inv[4] * y + inv[5]) / w)
        };
        let step = |dx: f32, dy: f32| {
            let (a, b) = (on_canvas(mx - dx / 2.0, my - dy / 2.0), on_canvas(mx + dx / 2.0, my + dy / 2.0));
            let d = (a.0 - b.0).hypot(a.1 - b.1);
            if d.is_finite() { ((d - 0.2).ceil() as i32).clamp(1, 3) } else { 2 }
        };
        let (nx, ny) = (step(1.0, 0.0), step(0.0, 1.0));
        let looks: Vec<(f32, f32)> = (0..ny)
            .flat_map(|j| (0..nx).map(move |i| ((i as f32 + 0.5) / nx as f32, (j as f32 + 0.5) / ny as f32)))
            .collect();
        let each = 1.0 / looks.len() as f32;
        let (bw, bh) = ((x1 - x0) as usize, (y1 - y0) as usize);
        let mut buf = vec![0u8; bw * bh * 4];
        {
            let under = self.overlay.pixels();
            for row in 0..bh {
                let from = ((y0 as usize + row) * fw as usize + x0 as usize) * 4;
                buf[row * bw * 4..(row + 1) * bw * 4].copy_from_slice(&under[from..from + bw * 4]);
            }
        }
        let clip = self.clips.last().copied();
        let eye = [dev[6], dev[7], dev[8]];
        // A large one's pixels by the GPU, if there is one; by how large it
        // is seen, not by what is drawn now (a part drawn again is as it was).
        #[cfg(all(metal_backend, feature = "ui"))]
        if (seen.2 - seen.0) * (seen.3 - seen.1) >= crate::metal::warp::WORTH {
            let rounds = self.rounded_clips();
            let all = [0.0, 0.0, fw as f32, fh as f32];
            let drawn = crate::metal::warp::warp(
                &crate::metal::warp::Warp {
                    inverse: inv,
                    eye,
                    canvas: layer.pixels(),
                    canvas_size: layer.size(),
                    origin,
                    looks: &looks,
                    decode: &table,
                    linear: space == crate::overlay::BlendSpace::Linear,
                    clip: clip.map_or(all, |c| [c.x, c.y, c.x + c.width, c.y + c.height]),
                    rounds: &rounds,
                    at: (x0, y0),
                    size: (bw as u32, bh as u32),
                },
                &mut buf,
            );
            if drawn {
                self.overlay.put_image_data(x0, y0, bw as u32, bh as u32, &buf);
                return;
            }
        }
        // Straight alpha to premultiplied, as `space` mixes it.
        let src: Vec<[f32; 4]> = layer
            .pixels()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| {
                let a = p[3] as f32 / 255.0;
                [table[p[0] as usize] * a, table[p[1] as usize] * a, table[p[2] as usize] * a, a]
            })
            .collect();
        let (lw, lh) = (lw as i32, lh as i32);
        // (its outermost pixels are clear: what is between them is all of it)
        let (most_u, most_v, stride) = ((lw - 1) as f32, (lh - 1) as f32, lw as usize);
        let rounded = self.rounds.iter().any(Option::is_some);
        let rounds = &self.rounds;
        let draw = |row: usize, out: &mut [u8]| {
            let y = (y0 + row as i32) as f32;
            let cover_y = clip.map_or(1.0, |r| ((r.y + r.height).min(y + 1.0) - r.y.max(y)).clamp(0.0, 1.0));
            if cover_y <= 0.0 {
                return;
            }
            for (col, px) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let x = (x0 + col as i32) as f32;
                let mut c = [0.0f32; 4];
                for &(dx, dy) in &looks {
                    let (sx, sy) = (x + dx, y + dy);
                    let w = 1.0 / (inv[6] * sx + inv[7] * sy + inv[8]);
                    let (u, v) = ((inv[0] * sx + inv[1] * sy + inv[2]) * w, (inv[3] * sx + inv[4] * sy + inv[5]) * w);
                    // canvas pixel `i` is `i` to `i + 1`, from its origin
                    let (cu, cv) = (u - origin.0 - 0.5, v - origin.1 - 0.5);
                    // (a point behind the eye comes back upside down, not at all)
                    if !(cu >= 0.0 && cv >= 0.0 && cu < most_u && cv < most_v && eye[0] * u + eye[1] * v + eye[2] > 0.0) {
                        continue;
                    }
                    let (i, j) = (cu as usize, cv as usize);
                    let (tx, ty) = (cu - i as f32, cv - j as f32);
                    let at = j * stride + i;
                    let (Some(&[a, b]), Some(&[d, e])) = (src[at..].first_chunk::<2>(), src[at + stride..].first_chunk::<2>()) else {
                        continue;
                    };
                    let (top, foot) = ((1.0 - ty) * each, ty * each);
                    let (wa, wb, wd, we) = (top * (1.0 - tx), top * tx, foot * (1.0 - tx), foot * tx);
                    for n in 0..4 {
                        c[n] += a[n] * wa + b[n] * wb + d[n] * wd + e[n] * we;
                    }
                }
                if c[3] <= 0.0 {
                    continue;
                }
                let mut cover = cover_y;
                if let Some(r) = clip {
                    cover *= ((r.x + r.width).min(x + 1.0) - r.x.max(x)).clamp(0.0, 1.0);
                }
                if rounded {
                    cover *= round_cover(rounds, x + 0.5, y + 0.5);
                }
                if cover <= 0.0 {
                    continue;
                }
                let sa = c[3] * cover;
                // nothing under it shows
                if sa >= 0.9995 {
                    for n in 0..3 {
                        px[n] = space_to_byte(space, c[n]);
                    }
                    px[3] = 255;
                    continue;
                }
                let da = px[3] as f32 / 255.0;
                let a = sa + da * (1.0 - sa);
                for n in 0..3 {
                    let d = table[px[n] as usize] * da * (1.0 - sa);
                    px[n] = space_to_byte(space, (c[n] * cover + d) / a.max(1e-6));
                }
                px[3] = (a * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        };
        // A large one's rows by several threads.
        let threads = if cfg!(target_arch = "wasm32") || bw * bh < 60_000 {
            1
        } else {
            std::thread::available_parallelism().map_or(1, |n| n.get()).min(6)
        };
        if threads <= 1 {
            for (row, out) in buf.chunks_exact_mut(bw * 4).enumerate() {
                draw(row, out);
            }
        } else {
            let rows = bh.div_ceil(threads);
            std::thread::scope(|s| {
                for (n, part) in buf.chunks_mut(rows * bw * 4).enumerate() {
                    let draw = &draw;
                    s.spawn(move || {
                        for (k, out) in part.chunks_exact_mut(bw * 4).enumerate() {
                            draw(n * rows + k, out);
                        }
                    });
                }
            });
        }
        self.overlay.put_image_data(x0, y0, bw as u32, bh as u32, &buf);
    }
}

/// The inverse of a 3 × 3 matrix by rows (`None`: it has none, a box turned
/// edge on).
fn inverse3(m: [f32; 9]) -> Option<[f32; 9]> {
    let m = m.map(f64::from);
    let co = |a: usize, b: usize, c: usize, d: usize| m[a] * m[d] - m[b] * m[c];
    let (c0, c1, c2) = (co(4, 5, 7, 8), -co(3, 5, 6, 8), co(3, 4, 6, 7));
    let det = m[0] * c0 + m[1] * c1 + m[2] * c2;
    let size = m.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    if det.abs() < 1e-12 * size.max(1e-6) {
        return None;
    }
    let inv = [
        c0, -co(1, 2, 7, 8), co(1, 2, 4, 5), //
        c1, co(0, 2, 6, 8), -co(0, 2, 3, 5), //
        c2, -co(0, 1, 6, 7), co(0, 1, 3, 4),
    ];
    Some(inv.map(|v| (v / det) as f32))
}

impl Overlay {
    /// Cuts the window to `clip` (to the nearest pixel): the one there was,
    /// to put back.
    fn narrow_window(&mut self, clip: Option<Rect>) -> Option<(u32, u32, u32, u32)> {
        let before = self.window();
        if let Some(c) = clip {
            let (x0, y0, x1, y1) = self.window_or_all();
            let at = |v: f32| v.round().max(0.0) as u32;
            let (cx0, cy0, cx1, cy1) = (at(c.x), at(c.y), at(c.x + c.width), at(c.y + c.height));
            let (x0, y0) = (x0.max(cx0), y0.max(cy0));
            self.set_window(Some((x0, y0, x1.min(cx1).max(x0), y1.min(cy1).max(y0))));
        }
        before
    }
}

/// Paths no larger than this (device pixels) are kept as pictures, and the
/// room round each (its strokes reach past its box).
const ICON: f32 = 96.0;
const PAD: f32 = 6.0;

/// A small path's picture: its size, its pixels.
type Icon = std::rc::Rc<(u32, u32, Vec<u8>)>;

thread_local! {
    static ICONS: std::cell::RefCell<std::collections::HashMap<(Vec<String>, Vec<u32>), Icon>> = Default::default();
}

/// `d` drawn at `rect` (and no further right or down than its box and the
/// room round it), as it was before if it has been: an icon's strokes are
/// distances worked out for each pixel, a hundred times what putting its
/// picture costs.
fn icon(rect: Rect, d: &[String], view_box: [f32; 4], stroke: Option<f32>, fill: bool, color: Rgba) -> Icon {
    let mut n: Vec<u32> = [rect.x, rect.y, rect.width, rect.height, stroke.unwrap_or(-1.0), fill as u32 as f32].map(f32::to_bits).to_vec();
    n.extend(view_box.iter().chain(&color).map(|v| v.to_bits()));
    let key = (d.to_vec(), n);
    if let Some(i) = ICONS.with_borrow(|m| m.get(&key).cloned()) {
        return i;
    }
    let (w, h) = ((rect.x + rect.width + PAD).ceil() as u32 + 1, (rect.y + rect.height + PAD).ceil() as u32 + 1);
    let mut o = Overlay::new(w, h);
    draw_path(&mut o, rect, d, view_box, stroke, fill, color);
    let i: Icon = std::rc::Rc::new((w, h, o.pixels().to_vec()));
    ICONS.with_borrow_mut(|m| {
        // (colours and places come and go: not kept for ever)
        if m.len() > 600 {
            m.clear();
        }
        m.insert(key, i.clone());
    });
    i
}

fn draw_path(o: &mut Overlay, rect: Rect, d: &[String], view_box: [f32; 4], stroke: Option<f32>, fill: bool, color: Rgba) {
    let [vx, vy, vw, vh] = view_box;
    let (sx, sy) = (rect.width / vw.max(1e-6), rect.height / vh.max(1e-6));
    for data in d {
        let Ok(subpaths) = crate::curves::parse_svg_subpaths(data) else {
            continue;
        };
        for sub in subpaths {
            let closed = sub.curve_path.auto_close;
            let points: Vec<_> = drawn_points(&sub, sx.max(sy), curve_divisions(rect))
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

/// How many pieces each curve of a path drawn over `rect` (device pixels)
/// is cut into: a piece every six pixels of the path's larger side, 12 to
/// 96, so a large curve stays round and a small one costs little. (Circular
/// arcs need none: they are drawn from their distance field.)
pub(crate) fn curve_divisions(rect: Rect) -> usize {
    (rect.width.max(rect.height) / 6.0).ceil().clamp(12.0, 96.0) as usize
}

/// The points a path is drawn through: a straight piece's ends, and a curve
/// cut by its length as drawn (`scale` device pixels a unit), into a piece
/// every two pixels, 8 to `most`.
pub(crate) fn drawn_points(path: &crate::curves::Path, scale: f32, most: usize) -> Vec<crate::math::Vector2> {
    use crate::curves::PathSegment;
    let mut out = vec![];
    for c in &path.curve_path.curves {
        let straight = matches!(c.svg_segments().as_deref(), Some([PathSegment::Line { .. }]));
        let n = if straight {
            1
        } else {
            let n = (c.get_length(8) * scale / 2.0).ceil() as usize;
            n.clamp(8, most.max(8))
        };
        // (it starts where the last ended)
        let from = usize::from(out.last() == Some(&c.get_point(0.0)));
        out.extend((from..=n).map(|i| c.get_point(i as f32 / n as f32)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What cutting a chart's line takes, whole pieces and the old way.
    #[test]
    #[ignore = "a timing"]
    fn time_of_a_chart_line() {
        let line: String = (0..240).map(|i| format!("{}{i} {} ", if i == 0 { "M" } else { "L" }, i * 37 % 60)).collect();
        let path = &crate::curves::parse_svg_subpaths(&line).unwrap()[0];
        let time = |f: &dyn Fn() -> usize| {
            (0..50)
                .map(|_| {
                    let t = std::time::Instant::now();
                    std::hint::black_box(f());
                    t.elapsed().as_secs_f64() * 1e6
                })
                .fold(f64::MAX, f64::min)
        };
        let each = |n: usize| (0..=n).map(|i| crate::curves::Curve2::get_point(&path.curve_path, i as f32 / n as f32)).count();
        println!("whole {:.1} us", time(&|| drawn_points(path, 2.0, 96).len()));
        println!("cut {:.1} us", time(&|| path.get_points(96).len()));
        println!("cut, each alone (a tenth) {:.1} us", time(&|| each(2300)));
    }

    /// Straight pieces are drawn by their ends; curves as near the true curve
    /// as the many points they were cut into.
    #[test]
    fn a_path_is_drawn_through_its_corners() {
        let subs = crate::curves::parse_svg_subpaths("M0 0 L10 0 L10 10 Q10 40 40 40 Z").unwrap();
        let pts = drawn_points(&subs[0], 1.0, 96);
        for corner in [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (40.0, 40.0)] {
            assert!(pts.iter().any(|p| (p.x, p.y) == corner), "{corner:?} in {pts:?}");
        }
        assert!(pts.len() < 40, "{}", pts.len());
        let many: String = (0..240).map(|i| format!("{}{i} {} ", if i == 0 { "M" } else { "L" }, i % 7)).collect();
        assert_eq!(drawn_points(&crate::curves::parse_svg_subpaths(&many).unwrap()[0], 2.0, 96).len(), 240);
        // Sampled finely, the curve stays within a tenth of a pixel.
        let curve = &crate::curves::parse_svg_subpaths("M0 0 C0 60 60 60 60 0").unwrap()[0];
        let cut = drawn_points(curve, 1.0, 96);
        for p in curve.get_points(500) {
            let d = cut
                .windows(2)
                .map(|w| crate::sdf::d2::segment(p, w[0], w[1]))
                .fold(f32::MAX, f32::min);
            assert!(d < 0.1, "{p:?} is {d} off");
        }
    }

    /// A large image drawn again, moved by whole pixels or cut, is as drawn
    /// anew (to a byte's rounding); and one seen through isn't kept.
    #[test]
    fn a_large_image_is_scaled_once() {
        let px: Vec<u8> = (0..400 * 300u32).flat_map(|i| [(i % 251) as u8, (i / 400 % 253) as u8, (i % 7 * 30) as u8, 255]).collect();
        let image = image::Image::new(400, 300, px.clone()).unwrap();
        let rect = |dx: f32| Rect { x: 20.25 + dx, y: 10.5, width: 350.0, height: 262.5 };
        let clip = Rect { x: 40.0, y: 30.0, width: 300.0, height: 200.0 };
        for dx in [0.0, 0.0, 0.0, 13.0, -47.0, 150.0, -300.0] {
            for cut in [false, true] {
                let mut a = Overlay::new(420, 320);
                let mut p = OverlayPainter::new(&mut a);
                if cut {
                    p.push_clip(clip);
                }
                p.image(rect(dx), &image, 1.0);
                let r = rect(dx);
                let mut b = Overlay::new(420, 320);
                let c = cut.then_some((40.0, 30.0, 340.0, 230.0));
                b.image(r.x, r.y, r.width, r.height, &px, 400, 300, 1.0, c);
                let far = a.pixels().iter().zip(b.pixels()).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
                assert!(far <= 1, "{dx} {cut}: {far}");
            }
        }
        let mut glass = px;
        glass[3] = 100;
        let image = image::Image::new(400, 300, glass.clone()).unwrap();
        // (by the CPU, to a byte: the GPU's is, but for its rounding)
        #[cfg(metal_backend)]
        crate::metal::warp::OFF.set(true);
        for _ in 0..3 {
            let mut a = Overlay::new(420, 320);
            OverlayPainter::new(&mut a).image(rect(0.0), &image, 1.0);
            let r = rect(0.0);
            let mut b = Overlay::new(420, 320);
            b.image(r.x, r.y, r.width, r.height, &glass, 400, 300, 1.0, None);
            assert!(a.pixels() == b.pixels());
        }
        #[cfg(metal_backend)]
        crate::metal::warp::OFF.set(false);
    }

    /// An icon from its picture looks as one drawn, and costs far less.
    #[test]
    fn an_icon_is_kept_as_a_picture() {
        let d = vec!["M4 4H10.5V10.5H4ZM13.5 4H20V10.5H13.5Z".to_owned()];
        let vb = [0.0, 0.0, 24.0, 24.0];
        let color = [1.0, 0.5, 0.2, 1.0];
        for (x, y) in [(10.0, 12.0), (10.25, 11.5), (0.0, 0.0)] {
            let rect = Rect { x, y, width: 36.0, height: 36.0 };
            let mut a = Overlay::new(80, 80);
            draw_path(&mut a, rect, &d, vb, Some(2.0), false, color);
            let mut b = Overlay::new(80, 80);
            OverlayPainter::new(&mut b).path(rect, &d, vb, Some(2.0), false, color);
            let far = a.pixels().chunks(4).zip(b.pixels().chunks(4)).filter(|(p, q)| p[3].abs_diff(q[3]) > 3).count();
            assert_eq!(far, 0, "at {x},{y}");
            assert!(b.pixels().chunks(4).any(|p| p[3] > 200));
        }
        let mut o = Overlay::new(80, 80);
        let t = std::time::Instant::now();
        for _ in 0..200 {
            OverlayPainter::new(&mut o).path(Rect { x: 10.0, y: 12.0, width: 36.0, height: 36.0 }, &d, vb, Some(2.0), false, color);
        }
        eprintln!("an icon: {:.3} ms", t.elapsed().as_secs_f64() * 5.0);
    }

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

    /// A see-through wash mixes in linear light by default, and as CSS
    /// does (the encoded values) with `BlendSpace::Srgb`: 14 % of green
    /// over black is about a third of the way in light, 14 % in sRGB.
    #[test]
    fn washes_mix_in_the_space_asked_for() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, crate::style! { background: #000000; });
        let mut st = threers_ui::Style::default();
        for (n, v) in [
            ("position", "absolute"),
            ("left", "0px"),
            ("top", "0px"),
            ("width", "20px"),
            ("height", "20px"),
            ("background", "rgba(0, 200, 0, 0.14)"),
        ] {
            assert!(threers_ui::css::try_apply(&mut st, n, v), "{n}");
        }
        ui.add_box(root, st);
        ui.layout(20.0, 20.0, &mut threers_ui::FixedWidthText(7.0));
        let green = |space| paint_to_rgba_in(&ui, 1.0, 20, 20, space)[(10 * 20 + 10) * 4 + 1];
        let (light, css) = (
            green(crate::overlay::BlendSpace::Linear),
            green(crate::overlay::BlendSpace::Srgb),
        );
        assert!((27..=29).contains(&css), "as CSS: 200 × 0.14 ≈ 28, got {css}");
        assert!(light > 60, "in light, much more: {light}");
    }

    /// A gradient in a rounded box stays inside its corners, as its
    /// background would.
    #[test]
    fn a_gradient_keeps_to_its_rounded_corners() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, crate::style! { background: #000000; });
        let mut st = threers_ui::Style::default();
        for (n, v) in [
            ("position", "absolute"),
            ("left", "10px"),
            ("top", "10px"),
            ("width", "100px"),
            ("height", "60px"),
            ("border-radius", "18px"),
            ("background", "linear-gradient(90deg, #ff0000, #0000ff)"),
        ] {
            assert!(threers_ui::css::try_apply(&mut st, n, v), "{n}");
        }
        ui.add_box(root, st);
        ui.layout(120.0, 80.0, &mut threers_ui::FixedWidthText(7.0));
        let rgba = paint_to_rgba(&ui, 1.0, 120, 80);
        let at = |x: u32, y: u32| &rgba[((y * 120 + x) * 4) as usize..][..3];
        assert_eq!(at(11, 11), [0, 0, 0], "the corner, outside the arc");
        assert!(at(60, 40)[0] > 100 && at(60, 40)[2] > 100, "the middle, mixed");
        assert!(at(11, 40)[0] > 200, "the left edge, away from a corner");
    }

    /// Letter-spacing widens a run by that much between its characters.
    #[test]
    fn letter_spacing_widens_a_run() {
        let mut overlay = Overlay::new(400, 100);
        let mut p = OverlayPainter::new(&mut overlay);
        let plain = threers_ui::Style::default();
        let spaced = threers_ui::Style { letter_spacing: 2.0, ..plain.clone() };
        let (a, b) = (p.measure("TITLE", &plain).0, p.measure("TITLE", &spaced).0);
        assert!((b - a - 8.0).abs() < 1.0, "{a} → {b}: four gaps of 2px");
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

    /// A turned box is drawn where its corners are seen, its near side
    /// taller, and nothing of it where it was.
    #[test]
    fn a_turned_box_is_drawn_turned() {
        use crate::ui::depth::{apply_homography, Transform3d};
        let mut ui = Ui::new();
        let root = ui.root();
        let card = ui.add_box(root, style! { position: absolute; left: 50px; top: 40px; width: 200px; height: 120px; background: #ffffff; });
        let t = Transform3d { rotate_y: 0.9, perspective: 300.0, ..Default::default() };
        ui.set_transform3d(card, t);
        ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
        for k in [1u32, 2] {
            let (w, h) = (300 * k, 200 * k);
            let px = paint_to_rgba(&ui, k as f32, w, h);
            let alpha = |x: f32, y: f32| px[(((y * k as f32) as u32 * w + (x * k as f32) as u32) * 4 + 3) as usize];
            let hm = t.homography(ui.rect(card));
            let r = ui.rect(card);
            let seen = |x, y| apply_homography(hm, x, y);
            for (x, y, dx, dy) in [(r.x, r.y, 1.0, 1.0), (r.x + r.width, r.y, -1.0, 1.0), (r.x, r.y + r.height, 1.0, -1.0), (r.x + r.width, r.y + r.height, -1.0, -1.0)] {
                // just inside each corner, and just outside
                let (ix, iy) = seen(x + dx * 6.0, y + dy * 6.0);
                assert!(alpha(ix, iy) > 240, "{k}: in at {x},{y}");
                let (ox, oy) = seen(x - dx * 4.0, y - dy * 4.0);
                if ox > 0.0 && oy > 0.0 && ox < 299.0 && oy < 199.0 {
                    assert!(alpha(ox, oy) < 16, "{k}: out at {x},{y}");
                }
            }
            assert_eq!(alpha(r.x + r.width - 2.0, r.y + 2.0), 0, "not where it was");
        }
        // Edge on, nothing; and no panic.
        ui.set_transform3d(card, Transform3d { rotate_y: std::f32::consts::FRAC_PI_2, ..t });
        paint_to_rgba(&ui, 2.0, 600, 400);
    }

    /// An image in a rounded box that clips has its corners cut, as a
    /// gradient has; one away from them is as it was.
    #[test]
    fn an_image_keeps_to_its_rounded_clip() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, style! { background: #ffffff; });
        let card = ui.add_box(root, style! { position: absolute; left: 10px; top: 10px; width: 60px; height: 40px; border-radius: 16px; overflow: hidden; });
        let red = image::Image::new(2, 2, [255, 0, 0, 255].repeat(4)).unwrap();
        ui.add_image(card, style! { width: 60px; height: 40px; }, red);
        ui.layout(100.0, 60.0, &mut FixedWidthText(7.0));
        for k in [1u32, 2] {
            let px = paint_to_rgba(&ui, k as f32, 100 * k, 60 * k);
            let at = |x: u32, y: u32| &px[((y * k * 100 * k + x * k) * 4) as usize..][..4];
            for (x, y) in [(10, 10), (69, 10), (10, 49), (69, 49), (12, 11)] {
                assert_eq!(at(x, y), [255, 255, 255, 255], "{k}: cut at {x},{y}");
            }
            for (x, y) in [(40, 10), (10, 30), (40, 30), (69, 30), (17, 17)] {
                assert_eq!(at(x, y), [255, 0, 0, 255], "{k}: there at {x},{y}");
            }
            // smooth: some of each along the arc
            let mixed = px.chunks(4).filter(|p| p[1] > 20 && p[1] < 235).count();
            assert!(mixed > 20, "{k}: {mixed}");
        }
    }

    /// Everything in a rounded box that clips has its corners cut: a plain
    /// box, a rounded one, a path.
    #[test]
    fn all_keeps_to_a_rounded_clip() {
        for kind in 0..3 {
            let mut ui = Ui::new();
            let root = ui.root();
            ui.set_style(root, style! { background: #ffffff; });
            let card = ui.add_box(root, style! { position: absolute; left: 10px; top: 10px; width: 60px; height: 40px; border-radius: 16px; overflow: hidden; });
            match kind {
                0 => ui.add_box(card, style! { width: 60px; height: 40px; background: #ff0000; }),
                1 => ui.add_box(card, style! { width: 60px; height: 40px; background: #ff0000; border-radius: 3px; }),
                _ => ui.add_path(card, style! { width: 60px; height: 40px; color: #ff0000; }, vec!["M0 0 L60 0 L60 40 L0 40 Z".into()], [0.0, 0.0, 60.0, 40.0], None, true),
            };
            ui.layout(100.0, 60.0, &mut FixedWidthText(7.0));
            let px = paint_to_rgba(&ui, 2.0, 200, 120);
            let at = |x: usize, y: usize| &px[(y * 2 * 200 + x * 2) * 4..][..4];
            for (x, y) in [(10, 10), (69, 10), (10, 49), (69, 49), (12, 11)] {
                assert_eq!(at(x, y), [255, 255, 255, 255], "{kind}: cut at {x},{y}");
            }
            for (x, y) in [(40, 12), (12, 30), (40, 30), (67, 30), (17, 17)] {
                assert_eq!(at(x, y), [255, 0, 0, 255], "{kind}: there at {x},{y}");
            }
        }
    }

    /// An image's own `border-radius` rounds it, inside its padding.
    #[test]
    fn an_images_own_radius_rounds_it() {
        for (pad, clear) in [(0.0, [10, 10]), (6.0, [16, 16])] {
            let mut ui = Ui::new();
            let root = ui.root();
            let red = image::Image::new(2, 2, [255, 0, 0, 255].repeat(4)).unwrap();
            let i = ui.add_image(root, style! { position: absolute; left: 10px; top: 10px; width: 60px; height: 40px; box-sizing: border-box; border-radius: 16px; }, red);
            ui.update_style(i, |s| s.padding = Edges::all(pad));
            ui.layout(100.0, 60.0, &mut FixedWidthText(7.0));
            let px = paint_to_rgba(&ui, 2.0, 200, 120);
            let at = |x: usize, y: usize| px[(y * 2 * 200 + x * 2) * 4 + 3];
            assert_eq!(at(clear[0], clear[1]), 0, "{pad}");
            assert_eq!(at(40, 30), 255);
            assert_eq!(at(40, 10 + pad as usize), 255);
        }
    }

    /// What the GPU fills is what the CPU would: each kind, plain and cut by
    /// a clip, square and round, as both blend.
    #[cfg(metal_backend)]
    #[test]
    fn the_gpu_paints_what_the_cpu_does() {
        use crate::metal::warp::{DRAWN, OFF};
        use crate::overlay::BlendSpace;
        use crate::ui::depth::Transform3d;
        let picture = |a: bool| {
            let px = (0..60 * 40u32).flat_map(|i| [(i % 60 * 4) as u8, (i / 60 * 6) as u8, 120, if a { (i % 256) as u8 } else { 255 }]).collect();
            image::Image::new(60, 40, px).unwrap()
        };
        let gradient = |css: &'static str| move |ui: &mut Ui, b: NodeId| ui.update_style(b, |s| css::apply(s, "background", css));
        type Make<'a> = Box<dyn Fn(&mut Ui, NodeId) + 'a>;
        let kinds: Vec<(&str, Make)> = vec![
            ("shadow", Box::new(|ui, b| { let st = style! { box-shadow: 3px 8px 24px rgba(20,0,90,0.6); }; ui.update_style(b, |s| { s.box_shadow = st.box_shadow.clone(); s.border_radius = Length::Px(20.0); }) })),
            ("across", Box::new(gradient("linear-gradient(90deg, #ff0000, rgba(0,0,255,0.3))"))),
            ("turned", Box::new(gradient("linear-gradient(30deg, #ff0000 10%, #00ff00 40%, #00ff00 60%, #0000ff)"))),
            ("radial", Box::new(gradient("radial-gradient(#ffffff, #102040)"))),
            ("conic", Box::new(gradient("conic-gradient(#ff0000, #0000ff 50%, #ff0000)"))),
            ("see-through", Box::new(|ui, b| { let i = picture(true); ui.add_image(b, style! { width: 100%; height: 100%; }, i); })),
            ("faded", Box::new(|ui, b| { let i = picture(false); ui.add_image(b, style! { width: 100%; height: 100%; opacity: 0.5; }, i); })),
            ("warped", Box::new(|ui, b| { let i = picture(false); ui.add_image(b, style! { width: 100%; height: 100%; }, i); ui.set_transform3d(b, Transform3d { rotate_y: 0.6, rotate_x: 0.2, perspective: 500.0, ..Default::default() }); })),
        ];
        for (name, make) in &kinds {
            for clipped in [false, true] {
                let mut ui = Ui::new();
                let root = ui.root();
                ui.set_style(root, style! { background: #ddd8cc; });
                let frame = ui.add_box(root, style! { position: absolute; left: 10.3px; top: 12.6px; width: 170px; height: 130px; });
                if clipped {
                    ui.update_style(frame, |s| (s.clip, s.border_radius) = (true, Length::Px(40.0)));
                }
                let b = ui.add_box(frame, style! { position: absolute; left: 12px; top: 9px; width: 190px; height: 150px; });
                make(&mut ui, b);
                ui.layout(230.0, 190.0, &mut FixedWidthText(7.0));
                for space in [BlendSpace::Linear, BlendSpace::Srgb] {
                    let draw = |off: bool| {
                        OFF.set(off);
                        let before = DRAWN.get();
                        let px = paint_to_rgba_in(&ui, 2.0, 460, 380, space);
                        OFF.set(false);
                        (px, DRAWN.get() - before)
                    };
                    let ((gpu, n), (cpu, none)) = (draw(false), draw(true));
                    assert!(none == 0, "{name}");
                    if n == 0 {
                        // no GPU here
                        return;
                    }
                    let off: Vec<u8> = gpu.chunks(4).zip(cpu.chunks(4)).map(|(a, b)| (0..4).map(|i| a[i].abs_diff(b[i])).max().unwrap()).collect();
                    let far = off.iter().filter(|d| **d > 3).count();
                    // (a picture's own edges are softened a little apart)
                    let most = if name.starts_with("see") || name.starts_with("faded") { 2200 } else { 60 };
                    assert!(far <= most, "{name} clipped {clipped} {space:?}: {far} pixels off, worst {}", off.iter().max().unwrap());
                }
            }
        }
    }
}
