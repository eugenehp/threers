//! Which of a UI's images the GPU draws ([`MetalUi`](super::MetalUi)), where,
//! and the holes its overlay has for them.

use std::collections::{HashMap, HashSet};

use crate::ui::{OverlayPainter, Rect, Ui};

use super::quads::Quad;
use super::ui::clip;

/// An image, by identity, and the pixels it covers.
pub(super) type Place = (u64, [u32; 4]);

/// The images of the last paint: the GPU's, which of them the overlay has a
/// hole for, those painted on the CPU, and each one's opacity, found once.
#[derive(Default)]
pub(super) struct GpuImages {
    quads: Vec<Quad>,
    holes: HashSet<Place>,
    cpu: HashSet<Place>,
    opaque: HashMap<u64, bool>,
}

impl GpuImages {
    pub(super) fn quads(&self) -> &[Quad] {
        &self.quads
    }

    pub(super) fn holes(&self) -> &HashSet<Place> {
        &self.holes
    }

    /// The images `ui` shows, where (frame pixels) and cut to what: a paint
    /// of it that draws nothing else. Whether one is now painted the other
    /// way (an image that came, went or changed is damage, which paints its
    /// hole; one that is as it was is not, so all is to be painted again).
    pub(super) fn record(&mut self, ui: &Ui, frame: (u32, u32), scale: f32) -> bool {
        let mut r = Recorder {
            scale,
            frame,
            clips: Vec::new(),
            opaque: &mut self.opaque,
            seen: Default::default(),
            quads: Vec::new(),
            painted: Vec::new(),
            cpu: Default::default(),
            turned: 0,
            rounds: Vec::new(),
        };
        ui.paint(&mut r);
        let (seen, quads, cpu) = (r.seen, r.quads, r.cpu);
        self.opaque.retain(|id, _| seen.contains(id));
        let holes: HashSet<Place> = quads.iter().map(|q| (q.image.id(), q.clip)).collect();
        let swapped = !holes.is_disjoint(&self.cpu) || !cpu.is_disjoint(&self.holes);
        self.holes = holes;
        self.cpu = cpu;
        self.quads = quads;
        swapped
    }
}

/// The pixel rectangle (`x, y, width, height`) an image drawn to `dest`
/// (frame pixels) covers within `clip`: its hole, and the GPU's scissor.
fn covered(dest: Rect, clip: Option<Rect>, frame: (u32, u32)) -> Option<[u32; 4]> {
    let (fw, fh) = (frame.0 as f32, frame.1 as f32);
    let (mut x0, mut y0, mut x1, mut y1) = (dest.x, dest.y, dest.x + dest.width, dest.y + dest.height);
    if let Some(c) = clip {
        (x0, y0, x1, y1) = (x0.max(c.x), y0.max(c.y), x1.min(c.x + c.width), y1.min(c.y + c.height));
    }
    let (x0, y0) = (x0.round().clamp(0.0, fw), y0.round().clamp(0.0, fh));
    let (x1, y1) = (x1.round().clamp(0.0, fw), y1.round().clamp(0.0, fh));
    (x1 > x0 && y1 > y0).then(|| [x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32])
}

/// `a` within `b`, or an empty rectangle.
fn within(a: Rect, b: Option<&Rect>) -> Rect {
    let Some(b) = b else { return a };
    clip(a, *b)
}

/// A paint that draws nothing but notes the opaque images, as the overlay's
/// painter would draw them: where, cut by the clips around them (the
/// overlay's painter clips to rectangles).
struct Recorder<'a> {
    scale: f32,
    frame: (u32, u32),
    clips: Vec<Rect>,
    opaque: &'a mut HashMap<u64, bool>,
    seen: HashSet<u64>,
    quads: Vec<Quad>,
    /// Where the CPU paints, so far (frame pixels).
    painted: Vec<Rect>,
    /// The images it paints.
    cpu: HashSet<Place>,
    /// The transforms in force.
    turned: usize,
    /// Beside each clip, its rounded corners (its box and radius, frame
    /// pixels) when it has them.
    rounds: Vec<Option<(Rect, f32)>>,
}

impl Recorder<'_> {
    /// Something the CPU paints, reaching `grow` UI pixels past `r`.
    fn cpu(&mut self, r: Rect, grow: f32) {
        let k = self.scale;
        let r = Rect { x: (r.x - grow) * k, y: (r.y - grow) * k, width: (r.width + 2.0 * grow) * k, height: (r.height + 2.0 * grow) * k };
        let r = within(r, self.clips.last());
        if r.width > 0.0 && r.height > 0.0 {
            self.painted.push(r);
        }
    }

    /// Everywhere (what it paints depends on what is under it).
    fn cpu_everywhere(&mut self) {
        self.painted.push(Rect { x: 0.0, y: 0.0, width: self.frame.0 as f32, height: self.frame.1 as f32 });
    }
}

impl crate::ui::TextMeasure for Recorder<'_> {
    fn measure(&mut self, _text: &str, _style: &crate::ui::Style) -> (f32, f32) {
        (0.0, 0.0)
    }
}

impl crate::ui::Painter for Recorder<'_> {
    fn fill_rect(&mut self, rect: Rect, color: crate::ui::Rgba) {
        if color[3] > 0.0 {
            self.cpu(rect, 0.0);
        }
    }
    fn fill_rounded_rect(&mut self, rect: Rect, _rx: f32, _ry: f32, color: crate::ui::Rgba) {
        self.fill_rect(rect, color);
    }
    fn fill_rounded_rect_corners(&mut self, rect: Rect, _radii: [f32; 4], color: crate::ui::Rgba) {
        self.fill_rect(rect, color);
    }
    fn fill_linear_gradient(&mut self, rect: Rect, _stops: &[(f32, crate::ui::Rgba)], blur: f32) {
        self.cpu(rect, blur);
    }
    fn fill_gradient(&mut self, rect: Rect, _shape: crate::ui::GradientShape, _stops: &[(f32, crate::ui::Rgba)], blur: f32) {
        self.cpu(rect, blur);
    }
    fn text(&mut self, line: Rect, _text: &str, style: &crate::ui::TextStyle) {
        self.cpu(line, style.font_size);
    }
    fn path(&mut self, rect: Rect, _d: &[String], _view_box: [f32; 4], stroke: Option<f32>, _fill: bool, _color: crate::ui::Rgba) {
        self.cpu(rect, stroke.unwrap_or(0.0).max(1.0));
    }
    fn stroke_arc(&mut self, arc: crate::ui::ArcShape, width: f32, _color: crate::ui::Rgba) {
        let b = arc.bounds(0.0);
        self.cpu(b, width);
    }
    fn shadow(&mut self, rect: Rect, _radius: f32, blur: f32, _color: crate::ui::Rgba) {
        self.cpu(rect, blur * 2.0);
    }
    fn underline(&mut self, x: f32, y: f32, width: f32, thickness: f32, _color: crate::ui::Rgba, _wavy: bool) {
        self.cpu(Rect { x, y, width, height: thickness }, thickness * 4.0);
    }
    fn backdrop(&mut self, _rect: Rect, _radius: f32, _blur: f32, _saturate: f32) {
        self.cpu_everywhere();
    }
    fn push_transform(&mut self, _m: [f32; 6]) {
        self.turned += 1;
        self.cpu_everywhere();
    }
    fn pop_transform(&mut self) {
        self.turned = self.turned.saturating_sub(1);
    }
    fn push_blur(&mut self, _radius: f32) {
        self.cpu_everywhere();
    }
    fn push_effect(&mut self, _rect: Rect, _effect: &crate::ui::shader::Effect, _time: f32) {
        self.cpu_everywhere();
    }

    fn push_clip(&mut self, rect: Rect) {
        let k = self.scale;
        let r = Rect { x: rect.x * k, y: rect.y * k, width: rect.width * k, height: rect.height * k };
        let c = within(r, self.clips.last());
        self.clips.push(c);
        self.rounds.push(None);
    }

    fn push_clip_rounded(&mut self, rect: Rect, rx: f32, ry: f32) {
        self.push_clip(rect);
        let k = self.scale;
        let radius = rx.min(ry) * k;
        if radius > 0.0
            && let Some(last) = self.rounds.last_mut()
        {
            *last = Some((Rect { x: rect.x * k, y: rect.y * k, width: rect.width * k, height: rect.height * k }, radius));
        }
    }

    fn pop_clip(&mut self) {
        self.clips.pop();
        self.rounds.pop();
    }

    fn image(&mut self, rect: Rect, image: &crate::ui::image::ImageRef, opacity: f32) {
        let id = image.id();
        self.seen.insert(id);
        let opaque = *self.opaque.entry(id).or_insert_with(|| image.rgba.chunks_exact(4).all(|p| p[3] == 255));
        let k = self.scale;
        let dest = Rect { x: rect.x * k, y: rect.y * k, width: rect.width * k, height: rect.height * k };
        let Some(clip) = covered(dest, self.clips.last().copied(), self.frame) else { return };
        // a hole there loses nothing: it is opaque, or there is nothing but
        // the GPU's under it
        let [x, y, w, h] = clip.map(|v| v as f32);
        let over_paint = || self.painted.iter().any(|p| p.x < x + w && p.x + p.width > x && p.y < y + h && p.y + p.height > y);
        // (a turned one is the overlay's, which turns it)
        // (nor one a round corner cuts: the GPU's are cut to rectangles)
        let cut = self.rounds.iter().flatten().any(|&(c, radius)| {
            let (right, bottom) = (c.x + c.width - radius, c.y + c.height - radius);
            [(c.x, c.y), (right, c.y), (c.x, bottom), (right, bottom)]
                .iter()
                .any(|&(cx, cy)| x < cx + radius && cx < x + w && y < cy + radius && cy < y + h)
        });
        if self.turned == 0 && !cut && opacity >= 0.999 && (opaque || !over_paint()) {
            self.quads.push(Quad { image: image.clone(), dest: [dest.x, dest.y, dest.width, dest.height], clip, opacity: 1.0 });
        } else {
            self.cpu.insert((id, clip));
            self.cpu(rect, 0.0);
        }
    }
}

/// The overlay's painter, but for the images the GPU draws: a hole (clear
/// pixels) where each goes, the same pixels its quad covers.
pub(super) struct Holes<'o, 'h> {
    inner: OverlayPainter<'o>,
    holes: Option<&'h HashSet<Place>>,
    clips: Vec<Rect>,
    turned: usize,
}

impl<'o, 'h> Holes<'o, 'h> {
    pub(super) fn new(inner: OverlayPainter<'o>, holes: Option<&'h HashSet<Place>>) -> Self {
        Self { inner, holes, clips: Vec::new(), turned: 0 }
    }
}

impl crate::ui::TextMeasure for Holes<'_, '_> {
    fn measure(&mut self, text: &str, style: &crate::ui::Style) -> (f32, f32) {
        self.inner.measure(text, style)
    }
}

impl crate::ui::Painter for Holes<'_, '_> {
    fn fill_rect(&mut self, rect: Rect, color: crate::ui::Rgba) {
        self.inner.fill_rect(rect, color)
    }
    fn fill_rounded_rect(&mut self, rect: Rect, rx: f32, ry: f32, color: crate::ui::Rgba) {
        self.inner.fill_rounded_rect(rect, rx, ry, color)
    }
    fn fill_rounded_rect_corners(&mut self, rect: Rect, radii: [f32; 4], color: crate::ui::Rgba) {
        self.inner.fill_rounded_rect_corners(rect, radii, color)
    }
    fn fill_linear_gradient(&mut self, rect: Rect, stops: &[(f32, crate::ui::Rgba)], blur: f32) {
        self.inner.fill_linear_gradient(rect, stops, blur)
    }
    fn fill_gradient(&mut self, rect: Rect, shape: crate::ui::GradientShape, stops: &[(f32, crate::ui::Rgba)], blur: f32) {
        self.inner.fill_gradient(rect, shape, stops, blur)
    }
    fn text(&mut self, line: Rect, text: &str, style: &crate::ui::TextStyle) {
        self.inner.text(line, text, style)
    }
    fn path(&mut self, rect: Rect, d: &[String], view_box: [f32; 4], stroke: Option<f32>, fill: bool, color: crate::ui::Rgba) {
        self.inner.path(rect, d, view_box, stroke, fill, color)
    }
    fn stroke_arc(&mut self, arc: crate::ui::ArcShape, width: f32, color: crate::ui::Rgba) {
        self.inner.stroke_arc(arc, width, color)
    }
    fn push_clip(&mut self, rect: Rect) {
        let k = self.inner.scale;
        let r = Rect { x: rect.x * k, y: rect.y * k, width: rect.width * k, height: rect.height * k };
        let c = within(r, self.clips.last());
        self.clips.push(c);
        self.inner.push_clip(rect)
    }
    fn push_clip_rounded(&mut self, rect: Rect, rx: f32, ry: f32) {
        // (images are cut to the rectangle; the overlay's painter rounds
        // what it fills pixel by pixel)
        let k = self.inner.scale;
        let r = Rect { x: rect.x * k, y: rect.y * k, width: rect.width * k, height: rect.height * k };
        let c = within(r, self.clips.last());
        self.clips.push(c);
        self.inner.push_clip_rounded(rect, rx, ry)
    }
    fn pop_clip(&mut self) {
        self.clips.pop();
        self.inner.pop_clip()
    }
    fn push_transform(&mut self, m: [f32; 6]) {
        self.turned += 1;
        self.inner.push_transform(m)
    }
    fn backdrop(&mut self, rect: Rect, radius: f32, blur: f32, saturate: f32) {
        self.inner.backdrop(rect, radius, blur, saturate)
    }
    fn push_projective(&mut self, h: [f32; 9], affine: [f32; 6]) {
        self.turned += 1;
        self.inner.push_projective(h, affine)
    }
    fn pop_transform(&mut self) {
        self.turned = self.turned.saturating_sub(1);
        self.inner.pop_transform()
    }
    fn push_blur(&mut self, radius: f32) {
        self.inner.push_blur(radius)
    }
    fn pop_blur(&mut self) {
        self.inner.pop_blur()
    }
    fn push_effect(&mut self, rect: Rect, effect: &crate::ui::shader::Effect, time: f32) {
        self.inner.push_effect(rect, effect, time)
    }
    fn pop_effect(&mut self) {
        self.inner.pop_effect()
    }
    fn set_subpixel(&mut self, on: bool) {
        self.inner.set_subpixel(on)
    }
    fn image(&mut self, rect: Rect, image: &crate::ui::image::ImageRef, opacity: f32) {
        if let Some(holes) = self.holes
            && opacity >= 0.999
            && self.turned == 0
        {
            let k = self.inner.scale;
            let dest = Rect { x: rect.x * k, y: rect.y * k, width: rect.width * k, height: rect.height * k };
            let frame = self.inner.overlay.size();
            if let Some(clip) = covered(dest, self.clips.last().copied(), frame)
                && holes.contains(&(image.id(), clip))
            {
                let [x, y, w, h] = clip;
                ZEROS.with(|z| {
                    let mut z = z.borrow_mut();
                    let n = w as usize * h as usize * 4;
                    if z.len() < n {
                        z.resize(n, 0);
                    }
                    self.inner.overlay.put_image_data(x as i32, y as i32, w, h, &z[..n]);
                });
                return;
            }
        }
        self.inner.image(rect, image, opacity)
    }
    fn surface(&mut self, rect: Rect, id: u64, opacity: f32) {
        self.inner.surface(rect, id, opacity)
    }
    fn shadow(&mut self, rect: Rect, radius: f32, blur: f32, color: crate::ui::Rgba) {
        self.inner.shadow(rect, radius, blur, color)
    }
    fn shadow_around(
        &mut self,
        rect: Rect,
        radius: f32,
        blur: f32,
        color: crate::ui::Rgba,
        hole: Rect,
        hole_radius: f32,
    ) {
        self.inner.shadow_around(rect, radius, blur, color, hole, hole_radius)
    }
    fn underline(&mut self, x: f32, y: f32, width: f32, thickness: f32, color: crate::ui::Rgba, wavy: bool) {
        self.inner.underline(x, y, width, thickness, color, wavy)
    }
}

thread_local!(static ZEROS: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) });
