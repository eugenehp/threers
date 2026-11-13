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
use crate::ui::{OverlayPainter, Rect, Ui};

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
}

impl Default for MetalUi {
    fn default() -> Self {
        Self::new()
    }
}

impl MetalUi {
    pub fn new() -> Self {
        Self { overlay: Overlay::new(1, 1), last: Vec::new(), scale: 0.0, stale: true, scratch: Overlay::new(1, 1) }
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
        ui.layout(w, h, &mut OverlayPainter::with_scale(&mut self.overlay, scale));
        let now = damage::collect(ui);
        let damage = if self.stale { Damage::Full } else { damage::diff(&self.last, &now, w, h) };
        self.last = now;
        self.stale = false;
        match damage {
            Damage::None => false,
            Damage::Full => {
                self.overlay.clear();
                ui.paint(&mut OverlayPainter::with_scale(&mut self.overlay, scale));
                true
            }
            Damage::Regions(regions) => {
                for r in regions {
                    self.repaint(ui, r);
                }
                true
            }
        }
    }

    /// Paint `r` (UI pixels) again: what reaches it, whole, into the
    /// scratch overlay, then that rectangle of it over the frame's. Painted
    /// under a clip instead, a rounded box the clip cuts loses its corners:
    /// the painter fills a cut one as a plain rectangle.
    fn repaint(&mut self, ui: &Ui, r: Rect) {
        // a box's shadow reaches past the box the damage is made of: a
        // thumb's that moved would stay behind
        const SHADOW: f32 = 12.0;
        let r = Rect { x: r.x - SHADOW, y: r.y - SHADOW, width: r.width + 2.0 * SHADOW, height: r.height + 2.0 * SHADOW };
        let k = self.scale;
        let (fw, fh) = self.overlay.size();
        let (x0, y0) = (((r.x * k).floor() as i32).max(0), ((r.y * k).floor() as i32).max(0));
        let (x1, y1) = ((((r.x + r.width) * k).ceil() as i32).min(fw as i32), (((r.y + r.height) * k).ceil() as i32).min(fh as i32));
        let (w, h) = ((x1 - x0).max(0) as u32, (y1 - y0).max(0) as u32);
        if w == 0 || h == 0 {
            return;
        }
        if self.scratch.size() != (fw, fh) {
            self.scratch.resize(fw, fh);
        }
        self.scratch.put_image_data(x0, y0, w, h, &vec![0; w as usize * h as usize * 4]);
        // what reaches the whole pixels just cleared, in UI pixels
        let reach = Rect { x: x0 as f32 / k, y: y0 as f32 / k, width: w as f32 / k, height: h as f32 / k };
        ui.paint(&mut Culled::new(&mut OverlayPainter::with_scale(&mut self.scratch, k), reach));
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

    /// Blend the UI over the frame in `pass`, then present its drawable if
    /// it has one (draw the scene into `pass.without_drawable()` first).
    pub fn draw(&mut self, renderer: &mut MetalRenderer, pass: &PassAttachments) -> Result<(), MetalError> {
        renderer.draw_overlay(&mut self.overlay, pass)
    }
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
