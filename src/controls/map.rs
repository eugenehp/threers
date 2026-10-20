//! Map-style camera controls: drag the ground, zoom toward the cursor, tilt
//! and turn.

use super::PointerEvent;
use crate::cameras::PerspectiveCamera;
use crate::math::Vector3;

/// A camera that looks at a point on a horizontal ground plane, the way a map
/// application's does.
///
/// The state is the look-at point, the distance to it, a bearing and a pitch,
/// and the camera is derived from those on every update — so a map can be put
/// anywhere by setting four numbers, and the controls never drift.
///
/// - **Pan** (`ev.panning`) grabs the ground: the point under the cursor stays
///   under the cursor however far the camera is tilted.
/// - **Zoom** (`ev.wheel`) moves toward the point under the cursor, not the
///   middle of the view.
/// - **Rotate** (`ev.rotating`) turns the bearing with horizontal motion and
///   tilts with vertical motion.
///
/// Y is up. Bearing 0 looks toward −Z, and increases clockwise seen from
/// above; pitch 0 looks straight down.
///
/// Unlike three.js's `MapControls`, which is `OrbitControls` with the buttons
/// swapped, this needs the cursor position, because two of its three gestures
/// are about the point under it.
#[derive(Debug, Clone, Copy)]
pub struct MapControls {
    /// The look-at point. Kept on the ground plane.
    pub target: Vector3,
    pub distance: f32,
    /// Radians clockwise from looking toward −Z.
    pub bearing: f32,
    /// Radians from straight down.
    pub pitch: f32,
    pub min_distance: f32,
    pub max_distance: f32,
    pub max_pitch: f32,
    /// Height of the ground plane.
    pub ground: f32,
    pub rotate_speed: f32,
    pub zoom_speed: f32,
}

impl MapControls {
    /// Looking straight down at `target` from `distance` away.
    pub fn new(target: Vector3, distance: f32) -> Self {
        Self {
            target,
            distance,
            bearing: 0.0,
            pitch: 0.0,
            min_distance: 1e-3,
            max_distance: 1e9,
            max_pitch: 85f32.to_radians(),
            ground: target.y,
            rotate_speed: 1.0,
            zoom_speed: 1.0,
        }
    }

    /// The camera's forward and up vectors.
    fn basis(&self) -> (Vector3, Vector3) {
        let (sb, cb) = self.bearing.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let forward = Vector3::new(sb * sp, -cp, -cb * sp);
        // Perpendicular to forward, and the horizontal "ahead" when looking
        // straight down — never parallel to the view, which world up would be.
        let up = Vector3::new(sb * cp, sp, -cb * cp);
        (forward, up)
    }

    /// Put `camera` where the state says.
    pub fn apply(&self, camera: &mut PerspectiveCamera) {
        let (forward, up) = self.basis();
        camera.target = self.target;
        camera.position = self.target - forward * self.distance;
        camera.up = up;
    }

    /// Set `camera`'s near and far planes to what this view can see: near a
    /// fraction of the distance, far growing as the view tilts towards the
    /// horizon (capped at 1000 distances). Keeps the depth buffer's far ÷
    /// near — which sets its precision — near 10⁴. Call after
    /// [`apply`](Self::apply).
    pub fn fit_clip_planes(&self, camera: &mut PerspectiveCamera) {
        let half = camera.fov * 0.5;
        // The far edge of the view meets the ground at the pitch plus half
        // the field of view; past 90° it never does.
        let edge = (self.pitch + half).min(std::f32::consts::FRAC_PI_2 - 1e-3);
        let height = self.distance * self.pitch.cos();
        let reach = (height / edge.cos().max(1e-3)).min(self.distance * 1000.0);
        camera.far = (reach * 1.5).max(self.distance * 2.0);
        camera.near = (self.distance * 0.01).max(camera.far * 1e-4);
    }

    /// Where the ray through pixel `(x, y)` (top-left origin) meets the
    /// ground, or `None` when it points at the sky.
    pub fn ground_point(
        &self,
        camera: &PerspectiveCamera,
        x: f32,
        y: f32,
        viewport: (f32, f32),
    ) -> Option<Vector3> {
        let (forward, up) = self.basis();
        let right = forward.cross(up);
        let origin = self.target - forward * self.distance;
        let t = (camera.fov * 0.5).tan();
        let nx = (x / viewport.0) * 2.0 - 1.0;
        let ny = 1.0 - (y / viewport.1) * 2.0;
        let dir = forward + right * (nx * t * camera.aspect) + up * (ny * t);
        if dir.y >= -1e-6 {
            return None;
        }
        let s = (self.ground - origin.y) / dir.y;
        Some(origin + dir * s)
    }

    /// Apply one frame of input. `cursor` is where the pointer is now; the
    /// event's `dx`, `dy` are how far it moved to get there.
    pub fn update(
        &mut self,
        ev: PointerEvent,
        cursor: (f32, f32),
        camera: &mut PerspectiveCamera,
        viewport: (f32, f32),
    ) {
        if !(ev.panning || ev.rotating || ev.wheel != 0.0) {
            return;
        }
        let (w, h) = viewport;
        if ev.panning && (ev.dx != 0.0 || ev.dy != 0.0) {
            let before = self.ground_point(camera, cursor.0 - ev.dx, cursor.1 - ev.dy, viewport);
            let after = self.ground_point(camera, cursor.0, cursor.1, viewport);
            if let (Some(b), Some(a)) = (before, after) {
                self.target = self.target + (b - a);
            }
        }
        if ev.rotating {
            self.bearing += ev.dx / w * std::f32::consts::TAU * self.rotate_speed;
            self.pitch = (self.pitch + ev.dy / h * std::f32::consts::PI * self.rotate_speed)
                .clamp(0.0, self.max_pitch);
        }
        if ev.wheel != 0.0 {
            let factor = (1.0 - ev.wheel * 0.001 * self.zoom_speed).clamp(0.1, 10.0);
            let anchor = self.ground_point(camera, cursor.0, cursor.1, viewport);
            self.distance = (self.distance * factor).clamp(self.min_distance, self.max_distance);
            if let (Some(before), Some(after)) = (
                anchor,
                self.ground_point(camera, cursor.0, cursor.1, viewport),
            ) {
                self.target = self.target + (before - after);
            }
        }
        self.target.y = self.ground;
        self.apply(camera);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (MapControls, PerspectiveCamera) {
        let cam = PerspectiveCamera::new(50.0, 1.5, 0.1, 1000.0);
        let mut m = MapControls::new(Vector3::ZERO, 100.0);
        m.pitch = 0.6;
        m.bearing = 0.3;
        (m, cam)
    }

    const VP: (f32, f32) = (600.0, 400.0);

    fn close(a: Vector3, b: Vector3) -> bool {
        (a - b).length() < 1e-2
    }

    #[test]
    fn clip_planes_reach_further_as_the_view_tilts() {
        let (mut m, mut cam) = setup();
        m.pitch = 0.0;
        m.apply(&mut cam);
        m.fit_clip_planes(&mut cam);
        let flat = cam.far;
        assert!(flat >= m.distance * 2.0 && cam.near <= m.distance * 0.01 + 1e-3);
        m.pitch = 1.3;
        m.fit_clip_planes(&mut cam);
        assert!(cam.far > flat * 2.0, "tilted: {} vs {flat}", cam.far);
        assert!(cam.far / cam.near <= 1.01e4);
    }

    #[test]
    fn panning_keeps_the_grabbed_point_under_the_cursor() {
        let (mut m, mut cam) = setup();
        m.apply(&mut cam);
        let start = (200.0, 250.0);
        let grabbed = m.ground_point(&cam, start.0, start.1, VP).unwrap();
        let ev = PointerEvent {
            dx: 40.0,
            dy: -25.0,
            panning: true,
            ..Default::default()
        };
        let now = (start.0 + 40.0, start.1 - 25.0);
        m.update(ev, now, &mut cam, VP);
        let under = m.ground_point(&cam, now.0, now.1, VP).unwrap();
        assert!(close(grabbed, under), "{grabbed:?} vs {under:?}");
    }

    #[test]
    fn zoom_keeps_the_point_under_the_cursor() {
        let (mut m, mut cam) = setup();
        m.apply(&mut cam);
        let cursor = (500.0, 120.0);
        let before = m.ground_point(&cam, cursor.0, cursor.1, VP).unwrap();
        let ev = PointerEvent {
            wheel: 300.0,
            ..Default::default()
        };
        m.update(ev, cursor, &mut cam, VP);
        assert!(m.distance < 100.0);
        let after = m.ground_point(&cam, cursor.0, cursor.1, VP).unwrap();
        assert!(close(before, after), "{before:?} vs {after:?}");
    }

    #[test]
    fn pitch_is_clamped_and_the_camera_looks_at_the_target() {
        let (mut m, mut cam) = setup();
        let ev = PointerEvent {
            dy: 10_000.0,
            rotating: true,
            ..Default::default()
        };
        m.update(ev, (0.0, 0.0), &mut cam, VP);
        assert_eq!(m.pitch, m.max_pitch);
        assert!(close(cam.target, m.target));
        assert!(((cam.position - cam.target).length() - m.distance).abs() < 1e-3);
        // The middle of the screen is the target.
        let mid = m.ground_point(&cam, VP.0 / 2.0, VP.1 / 2.0, VP).unwrap();
        assert!(close(mid, m.target));
    }

    #[test]
    fn straight_down_has_north_at_the_top() {
        let mut m = MapControls::new(Vector3::ZERO, 10.0);
        let cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        let top = m.ground_point(&cam, 50.0, 0.0, (100.0, 100.0)).unwrap();
        assert!(top.z < -1.0 && top.x.abs() < 1e-3, "{top:?}");
        m.bearing = std::f32::consts::FRAC_PI_2;
        let top = m.ground_point(&cam, 50.0, 0.0, (100.0, 100.0)).unwrap();
        // Turned clockwise a quarter: the top of the screen faces +X (east).
        assert!(top.x > 1.0, "{top:?}");
    }
}
