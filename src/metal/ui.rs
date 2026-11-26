//! A threers-ui [`Ui`] over a Metal frame.
//!
//! The wgpu renderer draws a UI on the GPU ([`UiPass`](crate::renderer::UiPass));
//! this backend paints it on the CPU, with the same painter the overlay uses
//! ([`OverlayPainter`]), into an [`Overlay`] the size of the frame, and
//! blends that over the frame ([`MetalRenderer::draw_overlay`]). Only what
//! changed is painted again: each update compares the laid-out tree with the
//! last one ([`damage`]) and repaints just the damaged rectangles, so a UI
//! that sits still costs a blend a frame and no painting, and a button that
//! lights up costs the button.
//!
//! The pixels are also there to take ([`MetalUi::pixels`]): a UI drawn alone,
//! as an image for something else to show.
//!
//! ```no_run
//! # #[cfg(all(feature = "metal", feature = "ui", target_os = "macos"))] {
//! # use threers::metal::{MetalRenderer, MetalUi, PassAttachments};
//! # fn frame(renderer: &mut MetalRenderer, ui: &mut threers::ui::Ui, scene: &mut threers::Scene,
//! #          camera: &threers::PerspectiveCamera, pass: PassAttachments) -> Result<(), threers::metal::MetalError> {
//! let mut hud = MetalUi::new();
//! hud.update(ui, pass.size().0, pass.size().1, 2.0);
//! renderer.render(scene, camera, &pass.without_drawable())?;
//! hud.draw(renderer, &pass)?; // blends, then presents
//! # Ok(()) }
//! # }
//! ```

use crate::overlay::Overlay;
use crate::ui::damage::{self, Damage};
use crate::ui::region::Culled;
use crate::ui::{NodeId, OverlayPainter, Rect, Ui};

use super::quads::MetalQuads;
use super::ui_images::{GpuImages, Holes};
use super::{MetalError, MetalRenderer, PassAttachments};

/// A UI's pixels, kept from frame to frame and repainted where they change.
pub struct MetalUi {
    overlay: Overlay,
    /// The tree as last painted, and the scale it was painted at.
    last: Vec<damage::Node>,
    scale: f32,
    /// Repaint everything next time.
    stale: bool,
    /// Where a damaged rectangle is painted before it is copied in.
    scratch: Overlay,
    /// The scrolling boxes as last painted.
    scrolls: Vec<Scrolled>,
    /// Device pixels painted by the last update (what it cost).
    painted: usize,
    /// The last update's layout and paint (ms), and whether it painted all.
    times: (f32, f32, bool),
    /// Images drawn by the GPU ([`set_gpu_images`](Self::set_gpu_images)):
    /// which, and their textures.
    gpu_images: bool,
    images: GpuImages,
    gpu: MetalQuads,
}

/// A scrolling box as painted: where it shows its content (UI pixels), how
/// far down it is scrolled, and what its pixels were moved short of that (a
/// fraction of a device pixel, carried to its next move).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Scrolled {
    node: NodeId,
    view: Rect,
    top: f32,
    carry: f32,
    /// how much of its scrollbar shows (0 with none), and where its track
    /// and thumb are
    bar: f32,
    track: Option<Rect>,
    thumb: Option<Rect>,
}

impl Default for MetalUi {
    fn default() -> Self {
        Self::new()
    }
}

impl MetalUi {
    pub fn new() -> Self {
        Self {
            overlay: Overlay::new(1, 1),
            last: Vec::new(),
            scale: 0.0,
            stale: true,
            scratch: Overlay::new(1, 1),
            scrolls: Vec::new(),
            painted: 0,
            times: (0.0, 0.0, false),
            gpu_images: false,
            images: GpuImages::default(),
            gpu: MetalQuads::default(),
        }
    }

    /// Images drawn by the GPU: opaque ones (every pixel's alpha whole, shown
    /// wholly opaque), and see-through ones nothing but such images is under.
    /// They go under the overlay, which has a hole where each goes: each is
    /// uploaded once, and a box of them scrolled, or resized (a page
    /// zoomed), moved there rather than painted again. Its
    /// [`pixels`](Self::pixels) then lack them.
    pub fn set_gpu_images(&mut self, on: bool) {
        if on != self.gpu_images {
            self.gpu_images = on;
            self.stale = true;
        }
    }

    /// Lay `ui` out to fill `width × height` pixels at `scale` pixels per UI
    /// pixel, and paint what changed since the last call. Whether anything
    /// was painted.
    pub fn update(&mut self, ui: &mut Ui, width: u32, height: u32, scale: f32) -> bool {
        let (width, height, scale) = (width.max(1), height.max(1), scale.max(0.01));
        if self.overlay.size() != (width, height) {
            self.overlay.resize(width, height);
            self.stale = true;
        }
        if scale != self.scale {
            self.scale = scale;
            self.stale = true;
        }
        let (w, h) = (width as f32 / scale, height as f32 / scale);
        let t0 = std::time::Instant::now();
        ui.layout(w, h, &mut OverlayPainter::with_scale(&mut self.overlay, scale));
        let laid = t0.elapsed().as_secs_f32() * 1e3;
        let t1 = std::time::Instant::now();
        let all = self.stale;
        if self.gpu_images {
            self.record_images(ui, width, height, scale);
        }
        let r = self.paint_damage(ui, w, h, width, height, scale);
        self.times = (laid, t1.elapsed().as_secs_f32() * 1e3, all || self.painted >= (width * height) as usize);
        r
    }

    /// The last update's layout and paint times (ms), and whether it
    /// painted everything (what a slow frame spent, where).
    pub fn last_times(&self) -> (f32, f32, bool) {
        self.times
    }

    fn paint_damage(&mut self, ui: &mut Ui, w: f32, h: f32, width: u32, height: u32, scale: f32) -> bool {
        let now = damage::collect(ui);
        // a box scrolled: its pixels moved with it, and what was last painted
        // there taken as moved too, so only the strip it uncovers is new
        let mut uncovered = if self.stale {
            self.scrolls = scrollers(ui, w, h);
            vec![]
        } else {
            self.scroll(ui, w, h, &now)
        };
        // a box that only moved (a key row the keyboard pushed up): its
        // pixels copied across, the last tree's nodes moved with it, and
        // only the place it left painted (not it, again, where it went)
        if !self.stale {
            uncovered.extend(self.move_blocks(&now, scale));
        }
        // `repaint` draws what reaches a rectangle whole and keeps the rectangle: no growing
        let mut damage = if self.stale { Damage::Full } else { damage::diff_cropped(&self.last, &now, w, h) };
        if !uncovered.is_empty() {
            damage = match damage {
                Damage::None => Damage::Regions(uncovered),
                Damage::Regions(mut r) => {
                    r.extend(uncovered);
                    Damage::Regions(r)
                }
                Damage::Full => Damage::Full,
            };
        }
        self.last = now;
        self.stale = false;
        self.painted = 0;
        match damage {
            Damage::None => false,
            Damage::Full => {
                self.painted = (width * height) as usize;
                self.overlay.clear();
                let holes = self.gpu_images.then_some(self.images.holes());
                ui.paint(&mut Holes::new(OverlayPainter::with_scale(&mut self.overlay, scale), holes));
                true
            }
            Damage::Regions(regions) => {
                for r in merged(regions) {
                    self.repaint(ui, r);
                }
                true
            }
        }
    }

    /// The boxes that only moved since the last update (damage::moved_blocks),
    /// each by whole device pixels from a place on whole ones: their pixels
    /// copied across, and the last tree's nodes for them moved (so nothing
    /// there is damaged); what each left uncovered, to paint.
    fn move_blocks(&mut self, now: &[damage::Node], k: f32) -> Vec<Rect> {
        let whole = |v: f32| (v * k - (v * k).round()).abs() < 0.01;
        let (fw, fh) = self.overlay.size();
        let mut left = Vec::new();
        for m in damage::moved_blocks(&self.last, now) {
            let (old, new) = (self.last[m.prev].visible, now[m.next].visible);
            if ![old.x, old.y, old.width, old.height, m.dx, m.dy].into_iter().all(whole) {
                continue;
            }
            let (sx, sy) = ((old.x * k).round() as i32, (old.y * k).round() as i32);
            let (w, h) = ((old.width * k).round() as u32, (old.height * k).round() as u32);
            let (dx, dy) = ((m.dx * k).round() as i32, (m.dy * k).round() as i32);
            // (on the frame at both ends: else painted as before)
            let inside = |x: i32, y: i32| x >= 0 && y >= 0 && x as u32 + w <= fw && y as u32 + h <= fh;
            if !inside(sx, sy) || !inside(sx + dx, sy + dy) {
                continue;
            }
            self.overlay.copy_rect(sx, sy, w, h, sx + dx, sy + dy);
            for i in 0..m.size {
                self.last[m.prev + i] = now[m.next + i].clone();
            }
            left.extend(damage::subtract(old, new));
        }
        left
    }

    /// The scrolling boxes moved since the last update, each by whole device
    /// pixels: their pixels shifted, the last tree's nodes inside them moved
    /// as far, and what each uncovers (or, once still, the whole box, for
    /// the fraction of a pixel its pixels were moved short) to paint.
    fn scroll(&mut self, ui: &Ui, w: f32, h: f32, now_nodes: &[damage::Node]) -> Vec<Rect> {
        let k = self.scale;
        let mut now = scrollers(ui, w, h);
        let mut paint = vec![];
        for c in &mut now {
            let Some(p) = self
                .scrolls
                .iter()
                .find(|p| p.node == c.node && p.view == c.view)
                // (a tree built anew, with more before it: the box that was there)
                .or_else(|| self.scrolls.iter().find(|p| p.view == c.view))
                .copied()
            else {
                continue;
            };
            // its scrollbar came or went: its track painted, where it was and
            // is; it faded, or its thumb moved (the track's pixels stay put):
            // the thumb, where it was and is (the track too, if it has a colour)
            let clear_track = ui.style(c.node).scrollbar_color.is_none_or(|(_, track)| track[3] == 0.0);
            if c.track != p.track || (c.bar != p.bar && !clear_track) {
                paint.extend(c.track.into_iter().chain(p.track));
            } else if c.thumb != p.thumb || c.bar != p.bar {
                paint.extend(c.thumb.into_iter().chain(p.thumb));
            }
            let dy = c.top - p.top;
            if dy == 0.0 {
                // still again after moving short of a pixel: painted where it is
                if p.carry != 0.0 {
                    paint.push(c.view);
                }
                continue;
            }
            if dy.abs() >= c.view.height {
                continue;
            }
            // the pixels moved by whole device pixels, the rest carried
            let exact = dy * k + p.carry;
            let px = exact.round();
            c.carry = exact - px;
            // all of it moved, its scrollbar's column too (what is under the
            // bar moved): the thumb, painted where it is (above), and where
            // its pixels were moved to
            self.shift_pixels(c.view, px as i32);
            // what is over the box, and did not move with it: where it is,
            // and where its pixels went
            for r in over(ui, c.node, c.view) {
                let went = Rect { y: r.y - px / k, ..r };
                paint.extend([r, went].map(|r| clip(r, c.view)).into_iter().filter(|r| r.width > 0.0 && r.height > 0.0));
            }
            if let Some(t) = p.thumb {
                paint.push(Rect { y: t.y - px / k, ..t });
            }
            // (a track with a colour of its own moved with it: painted whole)
            if !clear_track {
                paint.extend(c.track);
            }
            self.move_nodes(ui, c.node, dy, now_nodes);
            // the strip uncovered: below when scrolled down, above when up
            let strip = dy.abs() + 1.0;
            paint.push(if dy > 0.0 {
                Rect { x: c.view.x, y: c.view.y + c.view.height - strip, width: c.view.width, height: strip }
            } else {
                Rect { x: c.view.x, y: c.view.y, width: c.view.width, height: strip }
            });
        }
        self.scrolls = now;
        paint
    }

    /// The last tree's nodes in the scrolling box `node` moved up by `dy`
    /// with its pixels, so what only moved is not damage.
    fn move_nodes(&mut self, ui: &Ui, node: NodeId, dy: f32, now: &[damage::Node]) {
        // where the tree's own record clips what it holds: its padding box,
        // cut as the box itself shows (by the boxes around it)
        let r = ui.rect(node);
        let b = ui.style(node).border;
        let pad = Rect { x: r.x + b.left, y: r.y + b.top, width: (r.width - b.horizontal()).max(0.0), height: (r.height - b.vertical()).max(0.0) };
        let shown = now.iter().find(|n| n.style.scroll && n.rect == r).map_or(pad, |n| n.visible);
        let holds = clip(pad, shown);
        if self.other_rows(r, dy, now) {
            return self.move_box(r, dy, holds, now);
        }
        // the same tree, in the same order
        if self.last.len() != now.len() {
            return;
        }
        // each node that now sits exactly `dy` higher, where the box shows it:
        // its content, however tall
        for (n, next) in self.last.iter_mut().zip(now) {
            let r = n.rect;
            let moved = (next.rect.y - (r.y - dy)).abs() < 1e-3
                && next.rect.x == r.x
                && next.rect.width == r.width
                && next.rect.height == r.height;
            let shows = n.visible.width > 0.0 || next.visible.width > 0.0;
            if moved && shows && overlaps(r, holds) {
                // what of it shows changed only at the box's edges: what it
                // uncovers is the strip painted, what went out of sight needs
                // nothing
                n.rect.y -= dy;
                n.visible = next.visible;
            } else if n.style.scroll && n.rect == next.rect && n.style.scroll_top != next.style.scroll_top {
                // the box itself: only how far it is scrolled differs,
                // which the pixels moved already show
                n.style.scroll_top = next.style.scroll_top;
            }
        }
    }

    /// The last tree's nodes in the scrolling box at `r`, moved up by `dy`
    /// with its pixels, where the tree is another now (a list built only
    /// where it shows): what's in the same place then is the same.
    fn move_box(&mut self, r: Rect, dy: f32, holds: Rect, now: &[damage::Node]) {
        let Some((at, size)) = damage::scroll_box(&self.last, r) else { return };
        for n in &mut self.last[at + 1..at + size] {
            n.rect.y -= dy;
            n.visible = clip(Rect { y: n.visible.y - dy, ..n.visible }, holds);
        }
        if let Some((k, _)) = damage::scroll_box(now, r) {
            self.last[at].style.scroll_top = now[k].style.scroll_top;
        }
    }

    /// Whether the scrolling box at `r` holds other rows than it did, not
    /// the same ones `dy` higher.
    fn other_rows(&self, r: Rect, dy: f32, now: &[damage::Node]) -> bool {
        let (Some((a, n)), Some((b, m))) = (damage::scroll_box(&self.last, r), damage::scroll_box(now, r)) else { return false };
        if self.last.len() != now.len() || n != m {
            return true;
        }
        let pairs = || self.last[a + 1..a + n].iter().zip(&now[b + 1..b + m]).filter(|(x, y)| x.visible.width > 0.0 || y.visible.width > 0.0);
        let moved = pairs().filter(|(x, y)| (y.rect.y - (x.rect.y - dy)).abs() < 1e-3).count();
        moved * 2 < pairs().count()
    }

    /// `view`'s pixels (UI pixels) moved up by `px` device pixels (down when
    /// negative); what moves out of it is dropped.
    fn shift_pixels(&mut self, view: Rect, px: i32) {
        let k = self.scale;
        let (fw, fh) = self.overlay.size();
        let x0 = ((view.x * k).round() as i32).clamp(0, fw as i32);
        let x1 = (((view.x + view.width) * k).round() as i32).clamp(0, fw as i32);
        let y0 = ((view.y * k).round() as i32).clamp(0, fh as i32);
        let y1 = (((view.y + view.height) * k).round() as i32).clamp(0, fh as i32);
        let (w, h) = ((x1 - x0).max(0) as usize, (y1 - y0).max(0) as usize);
        if w == 0 || h <= px.unsigned_abs() as usize {
            return;
        }
        // moved within the frame's own pixels, row by row (no copy of the
        // whole box out and back in: that was most of a scroll's frame); the
        // rows uncovered are painted after, by `repaint`
        let keep = (h - px.unsigned_abs() as usize) as u32;
        if px > 0 {
            self.overlay.copy_rect(x0, y0 + px, w as u32, keep, x0, y0);
        } else {
            self.overlay.copy_rect(x0, y0, w as u32, keep, x0, y0 - px);
        }
    }

    /// Paint `r` (UI pixels) again: what reaches it, whole, into the
    /// scratch overlay, then that rectangle of it over the frame's. Painted
    /// under a clip instead, a rounded box the clip cuts loses its corners:
    /// the painter fills a cut one as a plain rectangle. (`r` is grown by
    /// [`SHADOW`] already.)
    fn repaint(&mut self, ui: &Ui, r: Rect) {
        let k = self.scale;
        let (fw, fh) = self.overlay.size();
        let (x0, y0) = (((r.x * k).floor() as i32).max(0), ((r.y * k).floor() as i32).max(0));
        let (x1, y1) = ((((r.x + r.width) * k).ceil() as i32).min(fw as i32), (((r.y + r.height) * k).ceil() as i32).min(fh as i32));
        let (w, h) = ((x1 - x0).max(0) as u32, (y1 - y0).max(0) as u32);
        if w == 0 || h == 0 {
            return;
        }
        self.painted += (w * h) as usize;
        if self.scratch.size() != (fw, fh) {
            self.scratch.resize(fw, fh);
        }
        self.scratch.put_image_data(x0, y0, w, h, &vec![0; w as usize * h as usize * 4]);
        // what reaches the whole pixels just cleared, in UI pixels
        let reach = Rect { x: x0 as f32 / k, y: y0 as f32 / k, width: w as f32 / k, height: h as f32 / k };
        // (a box that reaches them filled there only, not whole)
        self.scratch.set_window(Some((x0 as u32, y0 as u32, x1 as u32, y1 as u32)));
        let holes = self.gpu_images.then_some(self.images.holes());
        ui.paint(&mut Culled::new(&mut Holes::new(OverlayPainter::with_scale(&mut self.scratch, k), holes), reach));
        self.scratch.set_window(None);
        let row = fw as usize * 4;
        let src = self.scratch.pixels();
        let mut out = Vec::with_capacity(w as usize * h as usize * 4);
        for y in y0 as usize..y1 as usize {
            out.extend_from_slice(&src[y * row + x0 as usize * 4..y * row + x1 as usize * 4]);
        }
        self.overlay.put_image_data(x0, y0, w, h, &out);
    }

    /// Paint everything at the next update (after something outside the
    /// tree changed how it looks: a font, an image's pixels).
    pub fn invalidate(&mut self) {
        self.stale = true;
    }

    /// The UI's pixels: straight-alpha sRGB RGBA8, rows from the top.
    pub fn pixels(&self) -> &[u8] {
        self.overlay.pixels()
    }

    pub fn size(&self) -> (u32, u32) {
        self.overlay.size()
    }

    /// Device pixels the last update painted (what it cost).
    pub fn painted(&self) -> usize {
        self.painted
    }

    /// Blend the UI over the frame in `pass`, then present its drawable if
    /// it has one (draw the scene into `pass.without_drawable()` first).
    pub fn draw(&mut self, renderer: &mut MetalRenderer, pass: &PassAttachments) -> Result<(), MetalError> {
        // the GPU's images first, in the overlay's holes
        if self.gpu_images {
            self.gpu.draw(renderer.device(), self.images.quads(), &pass.without_drawable())?;
        }
        renderer.draw_overlay(&mut self.overlay, pass)
    }

    /// The images the GPU draws of the tree now.
    fn record_images(&mut self, ui: &Ui, width: u32, height: u32, scale: f32) {
        if self.images.record(ui, (width, height), scale) {
            self.stale = true;
        }
    }
}

/// Where what is drawn over `node` (not in it: what comes after it, or is
/// deferred) reaches into `view`, a pixel past its edges.
fn over(ui: &Ui, node: NodeId, view: Rect) -> Vec<Rect> {
    let mut out = vec![];
    let mut passed = false;
    // in the order it is painted; what `node` holds left out
    let mut stack = vec![ui.root()];
    while let Some(n) = stack.pop() {
        if n == node {
            passed = true;
            continue;
        }
        let s = ui.style(n);
        if s.display == crate::ui::Display::None {
            continue;
        }
        let drawn = s.background.is_some() || s.border != crate::ui::Edges::default() || ui.children(n).is_empty();
        if drawn && (passed || ui.deferred_priority(n).is_some()) {
            let r = ui.rect(n);
            let r = clip(Rect { x: r.x - 1.0, y: r.y - 1.0, width: r.width + 2.0, height: r.height + 2.0 }, view);
            if r.width > 0.0 && r.height > 0.0 {
                out.push(r);
            }
        }
        stack.extend(ui.children(n).iter().rev().copied());
    }
    out
}

/// The boxes that scroll up and down in `ui`: where each shows its content
/// (its padding box, within the `w × h` viewport) and how far it is scrolled.
fn scrollers(ui: &Ui, w: f32, h: f32) -> Vec<Scrolled> {
    let mut out = vec![];
    // (each with whether a box around it is turned: its pixels are not where
    // its box is, so cannot be moved as it scrolls; nor those of one holding
    // turned boxes, a drum's rows, which turn as it does)
    let mut stack = vec![(ui.root(), false)];
    while let Some((n, turned)) = stack.pop() {
        let turned = turned || ui.is_transformed(n);
        stack.extend(ui.children(n).iter().map(|&c| (c, turned)));
        if !ui.style(n).scroll || turned || holds_turned(ui, n) {
            continue;
        }
        let r = ui.rect(n);
        let b = ui.style(n).border;
        let (_, shown, top) = ui.scroll_extent(n);
        let view = clip(Rect { x: r.x + b.left, y: r.y + b.top, width: (r.width - b.horizontal()).max(0.0), height: shown }, Rect { x: 0.0, y: 0.0, width: w, height: h });
        if view.width > 0.0 && view.height > 0.0 {
            let geometry = ui.scrollbar(n);
            let bar = if geometry.is_some() { ui.scrollbar_opacity(n) } else { 0.0 };
            let (track, thumb) = (geometry.map(|g| g.track), geometry.map(|g| g.thumb));
            out.push(Scrolled { node: n, view, top, carry: 0.0, bar, track, thumb });
        }
    }
    out
}

/// Whether a box inside `n` is turned.
fn holds_turned(ui: &Ui, n: NodeId) -> bool {
    ui.children(n).iter().any(|&c| ui.is_transformed(c) || holds_turned(ui, c))
}

/// A box's shadow reaches past the box the damage is made of: a thumb's that
/// moved would stay behind.
const SHADOW: f32 = 12.0;

/// Each region grown by [`SHADOW`], and those near one another painted as
/// one: a repaint walks the whole tree, so a dozen slivers a scrolled frame
/// leaves (rows' edges at the view's end) cost a dozen walks where their
/// bounds, a few more pixels, cost one.
fn merged(regions: Vec<Rect>) -> Vec<Rect> {
    // pixels a walk is worth, in UI pixels²
    const WALK: f32 = 96.0 * 96.0;
    let area = |r: &Rect| r.width * r.height;
    let union = |a: &Rect, b: &Rect| {
        let (x0, y0) = (a.x.min(b.x), a.y.min(b.y));
        let (x1, y1) = ((a.x + a.width).max(b.x + b.width), (a.y + a.height).max(b.y + b.height));
        Rect { x: x0, y: y0, width: x1 - x0, height: y1 - y0 }
    };
    let mut out: Vec<Rect> = regions
        .into_iter()
        .map(|r| Rect { x: r.x - SHADOW, y: r.y - SHADOW, width: r.width + 2.0 * SHADOW, height: r.height + 2.0 * SHADOW })
        .collect();
    'again: loop {
        for i in 0..out.len() {
            for j in i + 1..out.len() {
                let u = union(&out[i], &out[j]);
                if area(&u) <= area(&out[i]) + area(&out[j]) + WALK {
                    out[i] = u;
                    out.swap_remove(j);
                    continue 'again;
                }
            }
        }
        return out;
    }
}

/// Whether `a` and `b` share any area.
fn overlaps(a: Rect, b: Rect) -> bool {
    a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
}

/// `r` cut to `to` (empty, at `to`'s corner, when they do not meet).
pub(super) fn clip(r: Rect, to: Rect) -> Rect {
    let x0 = r.x.max(to.x);
    let y0 = r.y.max(to.y);
    let x1 = (r.x + r.width).min(to.x + to.width);
    let y1 = (r.y + r.height).min(to.y + to.height);
    if x1 <= x0 || y1 <= y0 {
        return Rect { x: to.x, y: to.y, width: 0.0, height: 0.0 };
    }
    Rect { x: x0, y: y0, width: x1 - x0, height: y1 - y0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::style;

    fn ui_with(blue: bool) -> Ui {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.add_box(root, style! { width: 40px; height: 20px; background: #ff0000; border-radius: 6px; });
        if blue {
            ui.add_box(root, style! { width: 40px; height: 20px; background: #0000ff; });
        } else {
            ui.add_box(root, style! { width: 40px; height: 20px; background: #00ff00; });
        }
        ui
    }

    #[test]
    fn a_change_repaints_as_a_whole_paint_would() {
        let mut hud = MetalUi::new();
        let mut ui = ui_with(false);
        assert!(hud.update(&mut ui, 100, 100, 2.0));
        assert!(!hud.update(&mut ui, 100, 100, 2.0), "nothing changed, nothing painted");
        let mut ui = ui_with(true);
        assert!(hud.update(&mut ui, 100, 100, 2.0));
        let whole = crate::ui::paint_to_rgba(&ui, 2.0, 100, 100);
        assert_eq!(hud.pixels(), &whole[..]);
    }

    /// Turned boxes, changed: turned further, what is in one, and one that
    /// scrolls inside one, or holds ones that turn as it scrolls.
    #[test]
    fn turned_boxes_repaint_as_a_whole_paint_would() {
        use crate::ui::depth::Transform3d;
        let build = |angle: f32, blue: bool, top: f32, left: f32| {
            let mut ui = Ui::new();
            let root = ui.root();
            let card = ui.add_box(root, style! { position: absolute; left: 30px; top: 20px; width: 90px; height: 60px; background: #ffffff; border-radius: 10px; overflow: hidden; });
            ui.set_transform3d(card, Transform3d { rotate_y: angle, depth: 40.0, shadow: true, glare: true, perspective: 300.0, ..Default::default() });
            let dot = ui.add_box(card, style! { position: absolute; left: 60px; top: 10px; width: 20px; height: 20px; });
            ui.update_style(dot, |s| s.background = Some(if blue { [0.0, 0.0, 1.0, 1.0] } else { [0.0, 1.0, 0.0, 1.0] }));
            let turned = ui.add_box(root, style! { position: absolute; left: 150px; top: 10px; width: 80px; height: 90px; overflow-y: auto; display: flex; flex-direction: column; });
            ui.set_transform3d(turned, Transform3d { rotate_x: 0.6, rotate_z: 0.2, perspective: 300.0, ..Default::default() });
            let strip = ui.add_box(root, style! { position: absolute; left: 20px; top: 130px; width: 200px; height: 60px; overflow-x: auto; display: flex; flex-direction: row; gap: 8px; });
            let drum = ui.add_box(root, style! { position: absolute; left: 240px; top: 10px; width: 50px; height: 100px; overflow-y: auto; display: flex; flex-direction: column; });
            for i in 0..12 {
                let c = [i as f32 / 12.0, 0.3, 1.0 - i as f32 / 12.0, 1.0];
                for (p, w, h) in [(turned, 80.0, 24.0), (strip, 60.0, 50.0), (drum, 50.0, 20.0)] {
                    let b = ui.add_box(p, style! { flex-shrink: 0; });
                    ui.update_style(b, |s| (s.width, s.height, s.background) = (crate::ui::Length::Px(w), crate::ui::Length::Px(h), Some(c)));
                }
            }
            ui.update_style(turned, |s| s.scroll_top = top);
            ui.update_style(drum, |s| s.scroll_top = top);
            ui.update_style(strip, |s| s.scroll_left = left);
            ui.carousel_3d(strip);
            ui.cylinder_3d(drum);
            ui
        };
        let mut hud = MetalUi::new();
        let mut first = build(0.7, false, 0.0, 0.0);
        hud.update(&mut first, 600, 440, 2.0);
        for (angle, blue, top, left) in [
            (0.7, true, 0.0, 0.0),
            (0.2, true, 0.0, 0.0),
            (-0.9, true, 0.0, 0.0),
            (-0.9, false, 14.0, 0.0),
            (-0.9, false, 33.5, 40.0),
            (0.0, false, 0.0, 110.0),
        ] {
            let mut ui = build(angle, blue, top, left);
            assert!(hud.update(&mut ui, 600, 440, 2.0));
            let whole = crate::ui::paint_to_rgba(&ui, 2.0, 600, 440);
            let off = hud.pixels().chunks(4).zip(whole.chunks(4)).filter(|(a, b)| a != b).count();
            assert_eq!(off, 0, "{angle} {blue} {top} {left}");
        }
    }

    /// A box with a shadow, moved: the shadow it left goes too.
    fn shadow_at(x: f32) -> Ui {
        let mut ui = Ui::new();
        let root = ui.root();
        let b = ui.add_box(root, style! { position: absolute; top: 20px; width: 20px; height: 20px; border-radius: 10px; background: #ffffff; box-shadow: 0 1px 4px rgba(0,0,0,0.4); });
        ui.update_style(b, |s| s.left = crate::ui::Length::Px(x));
        ui
    }

    #[test]
    fn a_moved_shadow_leaves_nothing_behind() {
        let mut hud = MetalUi::new();
        let mut ui = shadow_at(10.0);
        assert!(hud.update(&mut ui, 160, 120, 2.0));
        let mut ui = shadow_at(60.0);
        assert!(hud.update(&mut ui, 160, 120, 2.0));
        let whole = crate::ui::paint_to_rgba(&ui, 2.0, 160, 120);
        assert_eq!(hud.pixels(), &whole[..]);
    }

    /// A box changing over a rounded box's corner: the corner, which the
    /// damaged rectangle cuts, keeps its curve.
    fn corner_with(blue: bool) -> Ui {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.add_box(root, style! { position: absolute; left: 10px; top: 10px; width: 60px; height: 40px; background: #ffffff; border-radius: 16px; });
        let over = ui.add_box(root, style! { position: absolute; left: 0px; top: 0px; width: 16px; height: 16px; });
        ui.update_style(over, |s| s.background = Some(if blue { [0.0, 0.0, 1.0, 0.5] } else { [0.0, 1.0, 0.0, 0.5] }));
        ui
    }

    /// A list of rows in a box that scrolls, `top` pixels down.
    fn list_at(top: f32) -> Ui {
        let mut ui = Ui::new();
        let root = ui.root();
        let list = ui.add_box(root, style! { position: absolute; left: 10px; top: 10px; width: 80px; height: 200px; overflow-y: auto; display: flex; flex-direction: column; });
        ui.update_style(list, |s| s.scroll_top = top);
        for i in 0..60 {
            let row = ui.add_box(list, style! { height: 12px; flex-shrink: 0; });
            ui.update_style(row, |s| s.background = Some(if i % 2 == 0 { [0.9, 0.2, 0.2, 1.0] } else { [0.2, 0.2, 0.9, 1.0] }));
        }
        ui
    }

    #[test]
    fn a_scrolled_box_moves_its_pixels_and_paints_what_it_uncovers() {
        let mut hud = MetalUi::new();
        let mut ui = list_at(0.0);
        assert!(hud.update(&mut ui, 120, 240, 2.0));
        let list = ui.children(ui.root())[0];
        // scrolled by whole pixels, down twice and back up: as a whole paint
        for top in [10.0, 23.0, 7.0] {
            ui.update_style(list, |s| s.scroll_top = top);
            assert!(hud.update(&mut ui, 120, 240, 2.0));
            // the strip uncovered (with a shadow's margin), not the list
            let list_px = 80 * 200 * 4;
            assert!(hud.painted < list_px / 3, "scrolled to {top}: {} px painted, the list is {list_px}", hud.painted);
            let whole = crate::ui::paint_to_rgba(&ui, 2.0, 120, 240);
            assert_eq!(hud.pixels(), &whole[..], "scrolled to {top}");
        }
        // by a fraction of a device pixel: once still, painted where it is
        ui.update_style(list, |s| s.scroll_top = 7.3);
        hud.update(&mut ui, 120, 240, 2.0);
        hud.update(&mut ui, 120, 240, 2.0);
        let whole = crate::ui::paint_to_rgba(&ui, 2.0, 120, 240);
        assert_eq!(hud.pixels(), &whole[..], "still again: painted where it is");
    }

    /// What is over a box that scrolls stays where it is as the box's
    /// pixels move under it.
    #[test]
    fn what_is_over_a_scrolled_box_stays() {
        use crate::ui::{Length, Position, Style};
        let mut hud = MetalUi::new();
        let mut ui = list_at(0.0);
        let root = ui.root();
        let list = ui.children(root)[0];
        let over = Style { position: Position::Absolute, left: Length::Px(20.0), top: Length::Px(90.0), width: Length::Px(30.0), height: Length::Px(20.0), background: Some([1.0, 0.0, 0.0, 1.0]), ..Style::default() };
        let badge = ui.add_box(root, over);
        ui.add_text(badge, "!");
        hud.update(&mut ui, 120, 240, 2.0);
        for top in [10.0, 23.0, 7.0, 40.0] {
            ui.update_style(list, |s| s.scroll_top = top);
            hud.update(&mut ui, 120, 240, 2.0);
            let whole = crate::ui::paint_to_rgba(&ui, 2.0, 120, 240);
            assert!(hud.pixels() == &whole[..], "scrolled to {top}");
        }
    }

    /// [`list_at`], built only where it shows: rows `from..to`, room for the
    /// rest, and a mark over it.
    fn rows_at(top: f32, from: usize, to: usize, mark: f32) -> Ui {
        let mut ui = Ui::new();
        let root = ui.root();
        let list = ui.add_box(root, style! { position: absolute; left: 10px; top: 10px; width: 80px; height: 200px; overflow-y: auto; display: flex; flex-direction: column; });
        ui.update_style(list, |s| s.scroll_top = top);
        let room = |ui: &mut Ui, n: usize| {
            if n > 0 {
                let b = ui.add_box(list, style! { flex-shrink: 0; });
                ui.update_style(b, |s| s.height = crate::ui::Length::Px(12.0 * n as f32));
            }
        };
        room(&mut ui, from);
        for i in from..to {
            let row = ui.add_box(list, style! { height: 12px; flex-shrink: 0; });
            ui.update_style(row, |s| s.background = Some([i as f32 / 100.0, 0.2, 1.0 - i as f32 / 100.0, 1.0]));
            ui.add_box(row, style! { width: 30px; height: 6px; background: #ffffff; });
        }
        room(&mut ui, 100 - to);
        let m = ui.add_box(root, style! { position: absolute; left: 70px; width: 6px; height: 40px; background: #00ff00; });
        ui.update_style(m, |s| s.top = crate::ui::Length::Px(mark));
        ui
    }

    #[test]
    fn a_list_built_only_where_it_shows_is_moved_though_its_rows_are_others() {
        let mut hud = MetalUi::new();
        hud.update(&mut rows_at(100.0, 0, 40, 20.0), 120, 240, 2.0);
        // other rows, as many or more or fewer, moved and not
        for (top, from, to) in [(110.0, 4, 44), (124.0, 4, 50), (124.0, 8, 48), (96.0, 0, 40), (180.0, 10, 60)] {
            let mut ui = rows_at(top, from, to, 10.0 + top / 8.0);
            hud.update(&mut ui, 120, 240, 2.0);
            let whole = crate::ui::paint_to_rgba(&ui, 2.0, 120, 240);
            assert_eq!(hud.pixels(), &whole[..], "{top} {from}..{to}");
            if top < 180.0 {
                assert!(hud.painted < 80 * 200 * 4 * 2 / 3, "{top} {from}..{to}: {} px painted", hud.painted);
            }
        }
    }

    #[test]
    fn a_rounded_corner_cut_by_the_damage_stays_round() {
        let mut hud = MetalUi::new();
        let mut ui = corner_with(false);
        assert!(hud.update(&mut ui, 160, 120, 2.0));
        let mut ui = corner_with(true);
        assert!(hud.update(&mut ui, 160, 120, 2.0));
        let whole = crate::ui::paint_to_rgba(&ui, 2.0, 160, 120);
        assert_eq!(hud.pixels(), &whole[..]);
    }
}

#[cfg(test)]
mod gpu_image_tests {
    use super::*;
    use crate::ui::image::Image;
    use crate::ui::{Length, Position, Style};

    fn at(x: f32, y: f32, w: f32, h: f32) -> Style {
        Style { position: Position::Absolute, left: Length::Px(x), top: Length::Px(y), width: Length::Px(w), height: Length::Px(h), ..Style::default() }
    }

    fn picture(alpha: u8) -> crate::ui::image::ImageRef {
        Image::new(4, 4, [200, 10, 10, alpha].repeat(16)).unwrap()
    }

    fn pixel(m: &MetalUi, x: usize, y: usize) -> [u8; 4] {
        let i = (y * m.size().0 as usize + x) * 4;
        m.pixels()[i..i + 4].try_into().unwrap()
    }

    fn metal() -> MetalUi {
        let mut m = MetalUi::new();
        m.set_gpu_images(true);
        m
    }

    #[test]
    fn an_opaque_image_is_the_gpus_and_a_hole_is_left_for_it() {
        let mut ui = Ui::new();
        let root = ui.root();
        let mut under = at(0.0, 0.0, 40.0, 40.0);
        under.background = Some([0.0, 1.0, 0.0, 1.0]);
        ui.add_box(root, under);
        ui.add_image(root, at(10.0, 10.0, 20.0, 20.0), picture(255));
        let mut over = at(15.0, 15.0, 4.0, 4.0);
        over.background = Some([0.0, 0.0, 1.0, 1.0]);
        ui.add_box(root, over);
        let mut m = metal();
        m.update(&mut ui, 40, 40, 1.0);
        assert_eq!(m.images.quads().len(), 1);
        assert_eq!(m.images.quads()[0].clip, [10, 10, 20, 20]);
        assert_eq!(pixel(&m, 12, 12)[3], 0, "the hole");
        assert_eq!(pixel(&m, 5, 5)[3], 255, "what is round it");
        assert_eq!(pixel(&m, 16, 16)[3], 255, "what is over it");
    }

    #[test]
    fn a_see_through_image_is_the_gpus_only_over_the_gpus() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.add_image(root, at(0.0, 0.0, 40.0, 40.0), picture(255));
        ui.add_image(root, at(5.0, 5.0, 10.0, 10.0), picture(128));
        let mut m = metal();
        m.update(&mut ui, 80, 40, 1.0);
        assert_eq!(m.images.quads().len(), 2, "over an opaque image");
        // something painted under another: that one stays the CPU's
        let mut under = at(50.0, 0.0, 20.0, 20.0);
        under.background = Some([0.0, 1.0, 0.0, 1.0]);
        ui.add_box(root, under);
        ui.add_image(root, at(55.0, 5.0, 10.0, 10.0), picture(128));
        m.update(&mut ui, 80, 40, 1.0);
        assert_eq!(m.images.quads().len(), 2);
        assert_eq!(pixel(&m, 58, 8)[3], 255, "blended over the box");
    }

    #[test]
    fn an_image_painted_the_other_way_is_painted_again() {
        let mut ui = Ui::new();
        let root = ui.root();
        let under = ui.add_box(root, at(0.0, 0.0, 20.0, 20.0));
        ui.add_image(root, at(5.0, 5.0, 10.0, 10.0), picture(128));
        let mut m = metal();
        m.update(&mut ui, 40, 40, 1.0);
        assert_eq!((m.images.quads().len(), pixel(&m, 8, 8)[3]), (1, 0));
        let mut st = ui.style(under).clone();
        st.background = Some([0.0, 1.0, 0.0, 1.0]);
        ui.set_style(under, st);
        m.update(&mut ui, 40, 40, 1.0);
        assert_eq!((m.images.quads().len(), pixel(&m, 8, 8)[3]), (0, 255), "no hole left behind");
    }
}
