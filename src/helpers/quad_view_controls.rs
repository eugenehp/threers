//! Pointer input for a [`QuadView`]: pure logic, no windowing dependency.
//!
//! Feed it pointer events in **canvas coordinates** — logical pixels, origin
//! at the **top left**, `y` down, which is what DOM `offsetX`/`offsetY` and
//! winit's cursor position (over the scale factor) both give — and it routes
//! each gesture to the pane it started in ([`QuadView::pane_at`]):
//!
//! | pane          | primary drag | secondary / middle drag | wheel                 |
//! |---------------|--------------|-------------------------|-----------------------|
//! | perspective   | orbit        | pan                     | dolly                 |
//! | orthographic  | pan          | pan                     | zoom about the cursor |
//!
//! The three orthographic panes keep **one** scale — a zoom in any of them is
//! [`QuadView::zoom`], which all three read — and **one** centre, so panning
//! along X in the top pane pans X in the front pane too (the third-angle
//! layout shares that axis). Only the perspective camera is ever orbited.
//!
//! A double-click maximises the pane under the pointer, or restores the four.
//! [`frame_box`](QuadViewControls::frame_box) /
//! [`frame_all`](QuadViewControls::frame_all) aim every pane at a box.
//!
//! ```
//! use threers::helpers::{Pane, PointerButton, QuadView, QuadViewControls};
//! use threers::{PerspectiveCamera, Vector3};
//! let mut quad = QuadView::new();
//! let mut cam = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
//! cam.position = Vector3::new(4.0, 3.0, 5.0);
//! let mut ctl = QuadViewControls::new(&cam);
//! ctl.set_size(800.0, 600.0);
//! // A drag in the top-right (perspective) pane orbits the camera.
//! assert_eq!(ctl.pointer_down(&quad, &cam, 600.0, 150.0, PointerButton::Primary),
//!            Some(Pane::Perspective));
//! assert!(ctl.pointer_move(&mut quad, &mut cam, 640.0, 150.0));
//! ctl.pointer_up();
//! ```

use super::quad_view::{scene_bounds, Pane, QuadView};
use crate::cameras::PerspectiveCamera;
use crate::controls::{OrbitControls, PointerEvent};
use crate::math::Box3;
use crate::scene::Scene;

/// Which button a pointer gesture uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerButton {
    /// Left mouse, a touch, a pen.
    Primary,
    Middle,
    /// Right mouse.
    Secondary,
}

impl PointerButton {
    /// From a DOM `PointerEvent.button`: 0 primary, 1 middle, 2 secondary.
    pub fn from_dom(button: i32) -> Option<Self> {
        match button {
            0 => Some(Self::Primary),
            1 => Some(Self::Middle),
            2 => Some(Self::Secondary),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum DragMode {
    Orbit,
    Pan,
}

#[derive(Clone, Copy, Debug)]
struct Drag {
    pane: Pane,
    mode: DragMode,
    last: (f32, f32),
}

/// Orbit/pan/zoom for the four panes of a [`QuadView`]. See the
/// [module docs](self).
#[derive(Clone, Debug)]
pub struct QuadViewControls {
    /// The perspective pane's orbit state. Reseeded from the camera at the
    /// start of every gesture in that pane, so a camera moved by other code
    /// in between is picked up rather than snapped back.
    pub orbit: OrbitControls,
    /// Wheel sensitivity: a wheel delta of `d` (DOM `deltaY`, pixels) scales
    /// by `exp(d · 0.001 · zoom_speed)`. Positive (scroll down) zooms out, as
    /// in three.js.
    pub zoom_speed: f32,
    /// Off: every call is ignored and returns "nothing changed".
    pub enabled: bool,
    width: f32,
    height: f32,
    drag: Option<Drag>,
}

impl QuadViewControls {
    pub fn new(camera: &PerspectiveCamera) -> Self {
        Self {
            orbit: OrbitControls::new(camera),
            zoom_speed: 1.0,
            enabled: true,
            width: 1.0,
            height: 1.0,
            drag: None,
        }
    }

    /// The canvas in logical pixels — the size the [`QuadView`] is rendered
    /// at, so the panes hit-test where they are drawn.
    pub fn set_size(&mut self, width: f32, height: f32) {
        self.width = width.max(1.0);
        self.height = height.max(1.0);
    }

    pub fn size(&self) -> (f32, f32) {
        (self.width, self.height)
    }

    /// The pane under a top-left-origin pointer position, `None` over a gap.
    pub fn pane_at(&self, quad: &QuadView, x: f32, y: f32) -> Option<Pane> {
        quad.pane_at(x, self.height - y, self.width, self.height)
    }

    /// The pane being dragged, if a gesture is in progress.
    pub fn dragging(&self) -> Option<Pane> {
        self.drag.map(|d| d.pane)
    }

    /// Start a gesture. Returns the pane it belongs to — `None` over a gap,
    /// when nothing will follow.
    pub fn pointer_down(
        &mut self,
        quad: &QuadView,
        camera: &PerspectiveCamera,
        x: f32,
        y: f32,
        button: PointerButton,
    ) -> Option<Pane> {
        if !self.enabled {
            return None;
        }
        let pane = self.pane_at(quad, x, y)?;
        let mode = match (pane, button) {
            (Pane::Perspective, PointerButton::Primary) => DragMode::Orbit,
            _ => DragMode::Pan,
        };
        if pane == Pane::Perspective {
            self.orbit.reseed_from_camera(camera);
        }
        self.drag = Some(Drag {
            pane,
            mode,
            last: (x, y),
        });
        Some(pane)
    }

    /// Continue the gesture. Returns whether anything moved (so a caller
    /// rendering on demand knows to redraw). The gesture stays with the pane
    /// it started in even when the pointer leaves it.
    pub fn pointer_move(
        &mut self,
        quad: &mut QuadView,
        camera: &mut PerspectiveCamera,
        x: f32,
        y: f32,
    ) -> bool {
        if !self.enabled {
            return false;
        }
        let Some(drag) = self.drag.as_mut() else {
            return false;
        };
        let (dx, dy) = (x - drag.last.0, y - drag.last.1);
        drag.last = (x, y);
        if dx == 0.0 && dy == 0.0 {
            return false;
        }
        let (pane, mode) = (drag.pane, drag.mode);
        match pane {
            Pane::Perspective => {
                let size = self.pane_size(quad, pane);
                self.orbit.update(
                    PointerEvent {
                        dx,
                        dy,
                        wheel: 0.0,
                        rotating: mode == DragMode::Orbit,
                        panning: mode == DragMode::Pan,
                    },
                    camera,
                    size,
                );
                true
            }
            _ => self.pan_ortho(quad, pane, dx, dy),
        }
    }

    /// End the gesture.
    pub fn pointer_up(&mut self) {
        self.drag = None;
    }

    /// Zoom at the pointer. `delta` is DOM `WheelEvent.deltaY` in pixels —
    /// positive (scroll down) zooms out. In an orthographic pane the zoom is
    /// the shared [`QuadView::zoom`], and the point under the cursor stays
    /// put; in the perspective pane the camera dollies towards its target.
    pub fn wheel(
        &mut self,
        quad: &mut QuadView,
        camera: &mut PerspectiveCamera,
        x: f32,
        y: f32,
        delta: f32,
    ) -> bool {
        if !self.enabled || delta == 0.0 || !delta.is_finite() {
            return false;
        }
        let Some(pane) = self.pane_at(quad, x, y) else {
            return false;
        };
        let factor = (delta * 0.001 * self.zoom_speed).exp();
        if pane == Pane::Perspective {
            if self.drag.is_none() {
                self.orbit.reseed_from_camera(camera);
            }
            // `OrbitControls` multiplies the distance by `1 − wheel·0.001`.
            let wheel = (1.0 - factor) / (0.001 * self.orbit.zoom_speed.max(1e-6));
            let size = self.pane_size(quad, pane);
            self.orbit.update(
                PointerEvent {
                    wheel,
                    ..Default::default()
                },
                camera,
                size,
            );
            return true;
        }
        let (w, h) = (self.width, self.height);
        let (Some(rect), Some(upp), Some((right, up))) = (
            quad.pane_rect(pane, w, h),
            quad.units_per_pixel(w, h),
            QuadView::pane_axes(pane),
        ) else {
            return false;
        };
        // The cursor's offset from the pane centre, pixels, y up.
        let ox = x - (rect.x + rect.width * 0.5);
        let oy = (h - y) - (rect.y + rect.height * 0.5);
        let zoom = (quad.zoom / factor).clamp(1e-4, 1e5);
        let upp_after = upp * quad.zoom / zoom;
        quad.zoom = zoom;
        // Keep the world point under the cursor where it is: the centre moves
        // by the cursor offset times the change in scale.
        let shift = upp - upp_after;
        quad.pan_by(right * (ox * shift) + up * (oy * shift));
        true
    }

    /// Maximise the pane under the pointer, or — when it already is — show
    /// all four again. Returns the pane, `None` over a gap.
    pub fn dblclick(&mut self, quad: &mut QuadView, x: f32, y: f32) -> Option<Pane> {
        if !self.enabled {
            return None;
        }
        let pane = self.pane_at(quad, x, y)?;
        quad.toggle_maximize(pane);
        self.drag = None;
        Some(pane)
    }

    /// Aim every pane at `bounds`: the orthographic panes framed on it at
    /// zoom 1, and the perspective camera moved along its current line of
    /// sight until the box's bounding sphere fits its field of view.
    pub fn frame_box(
        &mut self,
        quad: &mut QuadView,
        camera: &mut PerspectiveCamera,
        bounds: &Box3,
    ) {
        if bounds.is_empty() {
            return;
        }
        quad.frame_box(bounds);
        let center = bounds.center();
        let radius = (bounds.size().length() * 0.5).max(1e-4);
        let half_fov = (camera.fov * 0.5).clamp(1e-3, 1.5);
        let aspect = camera.aspect.max(1e-3);
        // The narrower of the two fields of view decides.
        let half = if aspect < 1.0 {
            (half_fov.tan() * aspect).atan()
        } else {
            half_fov
        };
        let distance = radius / half.sin();
        let dir = camera.position - camera.target;
        let dir = if dir.length() > 1e-6 {
            dir.normalize()
        } else {
            crate::math::Vector3::new(0.0, 0.0, 1.0)
        };
        camera.target = center;
        camera.position = center + dir * distance;
        self.orbit.reseed_from_camera(camera);
    }

    /// [`frame_box`](Self::frame_box) on every visible mesh in `scene`.
    /// Returns the box, `None` when there is nothing to frame.
    pub fn frame_all(
        &mut self,
        quad: &mut QuadView,
        camera: &mut PerspectiveCamera,
        scene: &mut Scene,
    ) -> Option<Box3> {
        let b = scene_bounds(scene)?;
        self.frame_box(quad, camera, &b);
        Some(b)
    }

    fn pane_size(&self, quad: &QuadView, pane: Pane) -> (f32, f32) {
        quad.pane_rect(pane, self.width, self.height)
            .map(|r| (r.width.max(1.0), r.height.max(1.0)))
            .unwrap_or((self.width, self.height))
    }

    /// Drag the picture with the pointer: a pointer move of `(dx, dy)`
    /// (pixels, y down) moves the shared centre the opposite way along the
    /// pane's screen axes.
    fn pan_ortho(&self, quad: &mut QuadView, pane: Pane, dx: f32, dy: f32) -> bool {
        let (Some(upp), Some((right, up))) = (
            quad.units_per_pixel(self.width, self.height),
            QuadView::pane_axes(pane),
        ) else {
            return false;
        };
        quad.pan_by(right * (-dx * upp) + up * (dy * upp));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cameras::OrthographicCamera;
    use crate::math::Vector3;

    const W: f32 = 800.0;
    const H: f32 = 600.0;

    fn setup() -> (QuadView, PerspectiveCamera, QuadViewControls) {
        let mut quad = QuadView::new();
        quad.gap = 0.0;
        quad.frame_box(&Box3::new(
            Vector3::new(-1.0, -1.0, -1.0),
            Vector3::new(1.0, 1.0, 1.0),
        ));
        quad.fit(W, H);
        let mut cam = PerspectiveCamera::new(45.0, 4.0 / 3.0, 0.1, 100.0);
        cam.position = Vector3::new(4.0, 3.0, 5.0);
        cam.look_at(Vector3::ZERO);
        let mut ctl = QuadViewControls::new(&cam);
        ctl.set_size(W, H);
        (quad, cam, ctl)
    }

    /// Canvas (top-left origin) centre of a pane.
    fn centre(quad: &QuadView, pane: Pane) -> (f32, f32) {
        let r = quad.pane_rect(pane, W, H).unwrap();
        (r.x + r.width * 0.5, H - (r.y + r.height * 0.5))
    }

    fn height(c: &OrthographicCamera) -> f32 {
        c.top - c.bottom
    }

    #[test]
    fn routes_by_pane_with_a_top_left_origin() {
        let (quad, cam, mut ctl) = setup();
        // Third-angle: top pane at the top left of the canvas.
        assert_eq!(
            ctl.pointer_down(&quad, &cam, 10.0, 10.0, PointerButton::Primary),
            Some(Pane::Top)
        );
        ctl.pointer_up();
        assert_eq!(
            ctl.pointer_down(&quad, &cam, 10.0, H - 10.0, PointerButton::Primary),
            Some(Pane::Front)
        );
        ctl.pointer_up();
        assert_eq!(ctl.pane_at(&quad, W - 10.0, 10.0), Some(Pane::Perspective));
        assert_eq!(ctl.pane_at(&quad, W - 10.0, H - 10.0), Some(Pane::Side));
    }

    #[test]
    fn zoom_in_the_top_pane_scales_every_ortho_pane_alike() {
        let (mut quad, mut cam, mut ctl) = setup();
        let before = [height(&quad.top), height(&quad.front), height(&quad.side)];
        let (x, y) = centre(&quad, Pane::Top);
        assert!(ctl.wheel(&mut quad, &mut cam, x, y, -200.0), "scroll up");
        quad.fit(W, H);
        let after = [height(&quad.top), height(&quad.front), height(&quad.side)];
        let ratio = after[0] / before[0];
        assert!(ratio < 0.9, "scroll up zooms in: {ratio}");
        for i in 1..3 {
            assert!(
                (after[i] / before[i] - ratio).abs() < 1e-5,
                "{after:?} {before:?}"
            );
        }
        assert!((quad.zoom - 1.0 / ratio).abs() < 1e-4);
        // At the pane centre the centre does not move.
        assert!((quad.target() - Vector3::ZERO).length() < 1e-5);
        // And the perspective camera is untouched.
        assert_eq!(cam.position, Vector3::new(4.0, 3.0, 5.0));
    }

    #[test]
    fn zoom_keeps_the_point_under_the_cursor() {
        let (mut quad, mut cam, mut ctl) = setup();
        // 50 px right of the front pane's centre and 40 px up.
        let (cx, cy) = centre(&quad, Pane::Front);
        let (x, y) = (cx + 50.0, cy - 40.0);
        let upp0 = quad.units_per_pixel(W, H).unwrap();
        let world = |q: &QuadView, upp: f32| q.target() + Vector3::new(50.0 * upp, 40.0 * upp, 0.0);
        let before = world(&quad, upp0);
        ctl.wheel(&mut quad, &mut cam, x, y, -300.0);
        let upp1 = quad.units_per_pixel(W, H).unwrap();
        let after = world(&quad, upp1);
        assert!((before.x - after.x).abs() < 1e-4 && (before.y - after.y).abs() < 1e-4);
    }

    #[test]
    fn pan_in_the_top_pane_moves_x_and_z_and_the_front_pane_follows_in_x() {
        let (mut quad, mut cam, mut ctl) = setup();
        let upp = quad.units_per_pixel(W, H).unwrap();
        let front_before = quad.front.position;
        let (x, y) = centre(&quad, Pane::Top);
        ctl.pointer_down(&quad, &cam, x, y, PointerButton::Primary);
        // Drag right 30 px and down 20 px: the picture follows the pointer,
        // so the view centre moves the other way — left (−X) and up the
        // screen, which in the top pane is −Z.
        assert!(ctl.pointer_move(&mut quad, &mut cam, x + 30.0, y + 20.0));
        ctl.pointer_up();
        let t = quad.target();
        assert!((t.x - (-30.0 * upp)).abs() < 1e-5, "{t:?}");
        assert!((t.z - (-20.0 * upp)).abs() < 1e-5, "{t:?}");
        assert!(t.y.abs() < 1e-6);
        // Front pane: X moved with it, Y did not, and its depth stayed on the
        // framed box (the Z part of the pan is along its line of sight).
        let f = quad.front.position;
        assert!((f.x - (front_before.x - 30.0 * upp)).abs() < 1e-5, "{f:?}");
        assert!((f.y - front_before.y).abs() < 1e-6);
        assert!((f.z - front_before.z).abs() < 1e-6);
        // Side pane (screen right is −Z) shows the Z move across, keeps X.
        assert!((quad.side.target.z - t.z).abs() < 1e-5);
        assert!(quad.side.target.x.abs() < 1e-6);
        // Pan never orbits.
        assert_eq!(cam.position, Vector3::new(4.0, 3.0, 5.0));
    }

    #[test]
    fn pan_in_the_front_pane_moves_x_and_y() {
        let (mut quad, mut cam, mut ctl) = setup();
        let upp = quad.units_per_pixel(W, H).unwrap();
        let (x, y) = centre(&quad, Pane::Front);
        ctl.pointer_down(&quad, &cam, x, y, PointerButton::Middle);
        ctl.pointer_move(&mut quad, &mut cam, x - 10.0, y - 25.0);
        let t = quad.target();
        // Pointer left and up: the view goes right (+X) and down (−Y).
        assert!((t.x - 10.0 * upp).abs() < 1e-5 && (t.y + 25.0 * upp).abs() < 1e-5);
        assert!(t.z.abs() < 1e-6);
        // The top pane shares X.
        assert!((quad.top.target.x - t.x).abs() < 1e-6);
    }

    #[test]
    fn orbit_only_moves_the_perspective_camera() {
        let (mut quad, mut cam, mut ctl) = setup();
        let ortho = (
            quad.top.clone(),
            quad.front.clone(),
            quad.side.clone(),
            quad.zoom,
        );
        let target = quad.target();
        let (x, y) = centre(&quad, Pane::Perspective);
        let before = cam.position;
        ctl.pointer_down(&quad, &cam, x, y, PointerButton::Primary);
        assert!(ctl.pointer_move(&mut quad, &mut cam, x + 40.0, y + 10.0));
        ctl.pointer_up();
        assert!((cam.position - before).length() > 0.1, "orbited");
        // Same distance from the target: an orbit, not a dolly.
        assert!(((cam.position - cam.target).length() - before.length()).abs() < 1e-3);
        assert_eq!(quad.target(), target);
        assert_eq!(quad.zoom, ortho.3);
        assert_eq!(quad.top.position, ortho.0.position);
        assert_eq!(quad.front.position, ortho.1.position);
        assert_eq!(quad.side.position, ortho.2.position);
        // The wheel in the perspective pane dollies the camera only.
        let d0 = (cam.position - cam.target).length();
        ctl.wheel(&mut quad, &mut cam, x, y, 100.0);
        let d1 = (cam.position - cam.target).length();
        assert!(d1 > d0 * 1.05, "scroll down dollies out: {d0} → {d1}");
        assert_eq!(quad.zoom, ortho.3);
    }

    #[test]
    fn a_gesture_stays_with_its_pane_and_gaps_start_none() {
        let (mut quad, mut cam, mut ctl) = setup();
        quad.gap = 10.0;
        // The vertical gap between the columns.
        let r = quad.pane_rect(Pane::Front, W, H).unwrap();
        let gx = r.x + r.width + 5.0;
        assert_eq!(
            ctl.pointer_down(&quad, &cam, gx, H - 10.0, PointerButton::Primary),
            None
        );
        assert!(!ctl.pointer_move(&mut quad, &mut cam, gx + 50.0, H - 10.0));
        // Started in the top pane, dragged across into the perspective one:
        // still a pan of the ortho panes.
        let (x, y) = centre(&quad, Pane::Top);
        ctl.pointer_down(&quad, &cam, x, y, PointerButton::Primary);
        ctl.pointer_move(&mut quad, &mut cam, W - 20.0, y);
        assert_eq!(ctl.dragging(), Some(Pane::Top));
        assert_eq!(cam.position, Vector3::new(4.0, 3.0, 5.0));
        assert!(quad.target().x < 0.0);
    }

    #[test]
    fn double_click_maximises_and_restores() {
        let (mut quad, _cam, mut ctl) = setup();
        let (x, y) = centre(&quad, Pane::Side);
        assert_eq!(ctl.dblclick(&mut quad, x, y), Some(Pane::Side));
        assert_eq!(quad.maximized, Some(Pane::Side));
        // Maximised, the side pane is everywhere; a double-click anywhere
        // restores the four.
        assert_eq!(ctl.dblclick(&mut quad, 5.0, 5.0), Some(Pane::Side));
        assert_eq!(quad.maximized, None);
    }

    #[test]
    fn frame_box_resets_zoom_and_pan_and_fits_the_perspective_camera() {
        let (mut quad, mut cam, mut ctl) = setup();
        quad.zoom = 3.0;
        quad.pan_by(Vector3::new(5.0, 0.0, 0.0));
        let b = Box3::new(Vector3::new(9.0, -1.0, -2.0), Vector3::new(11.0, 1.0, 2.0));
        ctl.frame_box(&mut quad, &mut cam, &b);
        assert_eq!(quad.zoom, 1.0);
        assert_eq!(quad.target(), Vector3::new(10.0, 0.0, 0.0));
        assert_eq!(cam.target, Vector3::new(10.0, 0.0, 0.0));
        // The box's bounding sphere fits the vertical field of view.
        let r = b.size().length() * 0.5;
        let d = (cam.position - cam.target).length();
        assert!((d * (cam.fov * 0.5).sin() - r).abs() < 1e-3, "{d}");
        // From the same direction as before.
        let dir = (cam.position - cam.target).normalize();
        assert!((dir - Vector3::new(4.0, 3.0, 5.0).normalize()).length() < 1e-4);
    }

    #[test]
    fn disabled_controls_do_nothing() {
        let (mut quad, mut cam, mut ctl) = setup();
        ctl.enabled = false;
        let (x, y) = centre(&quad, Pane::Top);
        assert_eq!(
            ctl.pointer_down(&quad, &cam, x, y, PointerButton::Primary),
            None
        );
        assert!(!ctl.wheel(&mut quad, &mut cam, x, y, -100.0));
        assert_eq!(ctl.dblclick(&mut quad, x, y), None);
        assert_eq!(quad.zoom, 1.0);
    }
}
