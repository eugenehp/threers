//! Globe navigation: drag the planet, zoom toward the cursor, tilt and turn.

use crate::cameras::PerspectiveCamera;
use crate::controls::PointerEvent;
use crate::marks::Origin;
use crate::math::Vector3;

/// Camera controls for a planet: what [`MapControls`](super::MapControls)
/// is for a flat map, on a sphere.
///
/// The state is a point on the surface (longitude and latitude in degrees,
/// in the convention of
/// [`marks::geo::lon_lat_to_globe`](crate::marks::geo::lon_lat_to_globe):
/// Y through the north pole, longitude 0 towards +Z), the camera's distance
/// from it, and a bearing and pitch in that point's own east-north-up frame
/// — bearing 0 looks north, pitch 0 straight down at the surface.
///
/// - **Pan** drags the globe: the point grabbed stays under the cursor.
/// - **Wheel** zooms toward the point under the cursor.
/// - **Rotate** turns with horizontal motion and tilts with vertical.
///
/// Everything is worked out in `f64`. A camera placed in absolute
/// coordinates on an Earth-sized sphere has half-metre steps, so place it
/// with [`apply_relative`](Self::apply_relative) against an [`Origin`] at the
/// look-at point ([`origin`](Self::origin)): the camera then sits within a
/// view distance of it, and stays exact from orbit down to a metre off the
/// ground. Draw marks and origin-relative meshes against the same origin.
#[derive(Debug, Clone, Copy)]
pub struct GlobeControls {
    pub centre: [f64; 3],
    pub radius: f64,
    /// The look-at point: longitude, degrees east.
    pub lon: f64,
    /// The look-at point: latitude, degrees north.
    pub lat: f64,
    /// From the camera to the look-at point.
    pub distance: f64,
    /// Radians clockwise from north.
    pub bearing: f32,
    /// Radians from straight down.
    pub pitch: f32,
    pub min_distance: f64,
    pub max_distance: f64,
    pub max_pitch: f32,
    pub rotate_speed: f32,
    pub zoom_speed: f32,
}

type V3 = [f64; 3];

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: V3) -> V3 {
    scale(a, 1.0 / dot(a, a).sqrt().max(1e-300))
}
fn v32(a: V3) -> Vector3 {
    Vector3::new(a[0] as f32, a[1] as f32, a[2] as f32)
}

impl GlobeControls {
    /// Looking straight down at `(lon, lat)` from `distance` above it.
    pub fn new(radius: f64, lon: f64, lat: f64, distance: f64) -> Self {
        Self {
            centre: [0.0; 3],
            radius,
            lon,
            lat,
            distance,
            bearing: 0.0,
            pitch: 0.0,
            min_distance: radius * 1e-7,
            max_distance: radius * 20.0,
            max_pitch: 80f32.to_radians(),
            rotate_speed: 1.0,
            zoom_speed: 1.0,
        }
    }

    /// The look-at point on the surface, and its east, north and up.
    fn frame(&self) -> (V3, V3, V3, V3) {
        let (lon, lat) = (self.lon.to_radians(), self.lat.to_radians());
        let up = [lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos()];
        let east = [lon.cos(), 0.0, -lon.sin()];
        let north = cross(up, east);
        (add(self.centre, scale(up, self.radius)), east, north, up)
    }

    /// The camera's position (absolute), forward and up.
    fn pose(&self) -> (V3, V3, V3) {
        let (target, east, north, up) = self.frame();
        let (sb, cb) = (self.bearing as f64).sin_cos();
        let (sp, cp) = (self.pitch as f64).sin_cos();
        let forward = add(
            add(scale(east, sb * sp), scale(north, cb * sp)),
            scale(up, -cp),
        );
        let cam_up = add(
            add(scale(east, sb * cp), scale(north, cb * cp)),
            scale(up, sp),
        );
        (add(target, scale(forward, -self.distance)), forward, cam_up)
    }

    /// The look-at point on the surface, absolute.
    pub fn target(&self) -> [f64; 3] {
        self.frame().0
    }

    /// The camera's position, absolute.
    pub fn eye(&self) -> [f64; 3] {
        self.pose().0
    }

    /// An origin at the look-at point: what to place the camera, the marks
    /// and nearby meshes against. Move it whenever convenient — every frame
    /// is fine, since marks hold absolute positions.
    pub fn origin(&self) -> Origin {
        let t = self.target();
        Origin::new(t[0], t[1], t[2])
    }

    /// Put `camera` where the state says, in absolute coordinates — exact
    /// only for a small sphere near zero; see
    /// [`apply_relative`](Self::apply_relative).
    pub fn apply(&self, camera: &mut PerspectiveCamera) {
        self.apply_relative(camera, &Origin::ZERO);
    }

    /// Put `camera` where the state says, in coordinates relative to
    /// `origin`, the subtraction done in `f64`.
    pub fn apply_relative(&self, camera: &mut PerspectiveCamera, origin: &Origin) {
        let (pos, _, up) = self.pose();
        let o = origin.position;
        let rel = |p: V3| v32([p[0] - o[0], p[1] - o[1], p[2] - o[2]]);
        camera.position = rel(pos);
        camera.target = rel(self.frame().0);
        camera.up = v32(up);
    }

    /// Set `camera`'s near and far planes to just what this view can see:
    /// near a fraction of the altitude, far a little past the horizon.
    ///
    /// Depth precision is set by far ÷ near; a planet seen with a fixed near
    /// plane of a metre and a far one at the horizon z-fights across whole
    /// mountains. Fitting both every frame keeps the ratio to about 10⁴
    /// whatever the altitude.
    pub fn fit_clip_planes(&self, camera: &mut PerspectiveCamera) {
        let r = self.radius;
        let d = self.pose().0;
        let c = self.centre;
        let h = dot(
            [d[0] - c[0], d[1] - c[1], d[2] - c[2]],
            [d[0] - c[0], d[1] - c[1], d[2] - c[2]],
        )
        .sqrt();
        let altitude = (h - r).max(r * 1e-8);
        // Past the horizon nothing on the sphere is visible; a little more
        // for whatever stands on it.
        let horizon = (h * h - r * r).max(0.0).sqrt();
        let far = (horizon + altitude) * 1.1 + r * 0.01;
        let near = (altitude * 0.1).max(far * 1e-4);
        camera.near = near as f32;
        camera.far = far as f32;
    }

    /// `[longitude, latitude]` of the surface under pixel `(x, y)` (top-left
    /// origin), or `None` when the ray misses the planet. Uses `camera` only
    /// for its field of view and aspect.
    pub fn surface_point(
        &self,
        camera: &PerspectiveCamera,
        x: f32,
        y: f32,
        viewport: (f32, f32),
    ) -> Option<[f64; 2]> {
        let (origin, forward, up) = self.pose();
        let right = cross(forward, up);
        // `fov` is stored in radians.
        let t = (camera.fov as f64 * 0.5).tan();
        let nx = (x / viewport.0) as f64 * 2.0 - 1.0;
        let ny = 1.0 - (y / viewport.1) as f64 * 2.0;
        let dir = norm(add(
            add(forward, scale(right, nx * t * camera.aspect as f64)),
            scale(up, ny * t),
        ));
        let c = self.centre;
        let oc = [origin[0] - c[0], origin[1] - c[1], origin[2] - c[2]];
        let b = dot(oc, dir);
        let r = self.radius;
        // The nearest root, written so that a short ray from just above a
        // planet-sized sphere does not cancel: `c / q` rather than `-b - √`.
        let cc = dot(oc, oc) - r * r;
        let disc = b * b - cc;
        if disc < 0.0 {
            return None;
        }
        let q = -b + disc.sqrt();
        let s = if q.abs() > 0.0 { cc / q } else { -b };
        if s < 0.0 || b > 0.0 {
            return None;
        }
        let p = add(oc, scale(dir, s));
        let ll = crate::marks::geo::globe_to_lon_lat(p, r);
        Some([ll[0], ll[1]])
    }

    /// Apply one frame of input: `cursor` is where the pointer is now, the
    /// event's `dx`, `dy` how far it moved to get there. Leaves `camera`
    /// alone; place it with [`apply_relative`](Self::apply_relative).
    pub fn update(
        &mut self,
        ev: PointerEvent,
        cursor: (f32, f32),
        camera: &PerspectiveCamera,
        viewport: (f32, f32),
    ) {
        if !(ev.panning || ev.rotating || ev.wheel != 0.0) {
            return;
        }
        let (w, h) = viewport;
        // Keep the surface point at `before` under the cursor at `after`, by
        // moving the look-at point the other way. Two passes: the first gets
        // most of it, the second the curvature the first did not account for.
        let hold = |s: &mut Self, before: [f64; 2], at: (f32, f32)| {
            for _ in 0..2 {
                let Some(now) = s.surface_point(camera, at.0, at.1, viewport) else {
                    return;
                };
                let mut dlon = before[0] - now[0];
                if dlon > 180.0 {
                    dlon -= 360.0;
                } else if dlon < -180.0 {
                    dlon += 360.0;
                }
                s.lon = (s.lon + dlon + 540.0).rem_euclid(360.0) - 180.0;
                s.lat = (s.lat + before[1] - now[1]).clamp(-89.9, 89.9);
            }
        };
        if ev.panning && (ev.dx != 0.0 || ev.dy != 0.0) {
            if let Some(grabbed) =
                self.surface_point(camera, cursor.0 - ev.dx, cursor.1 - ev.dy, viewport)
            {
                hold(self, grabbed, cursor);
            }
        }
        if ev.rotating {
            self.bearing += ev.dx / w * std::f32::consts::TAU * self.rotate_speed;
            self.pitch = (self.pitch + ev.dy / h * std::f32::consts::PI * self.rotate_speed)
                .clamp(0.0, self.max_pitch);
        }
        if ev.wheel != 0.0 {
            let anchor = self.surface_point(camera, cursor.0, cursor.1, viewport);
            let factor = (1.0 - ev.wheel as f64 * 0.001 * self.zoom_speed as f64).clamp(0.1, 10.0);
            self.distance = (self.distance * factor).clamp(self.min_distance, self.max_distance);
            if let Some(a) = anchor {
                hold(self, a, cursor);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marks::MarkView;

    const VP: (f32, f32) = (800.0, 600.0);

    fn setup() -> (GlobeControls, PerspectiveCamera) {
        let cam = PerspectiveCamera::new(50.0, VP.0 / VP.1, 1.0, 1e6);
        let mut g = GlobeControls::new(1000.0, 10.0, 45.0, 1500.0);
        g.pitch = 0.4;
        g.bearing = 0.3;
        (g, cam)
    }

    fn close(a: [f64; 2], b: [f64; 2], tol: f64) -> bool {
        (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol
    }

    #[test]
    fn straight_down_the_centre_of_the_screen_is_the_look_at_point() {
        let (mut g, mut cam) = setup();
        g.pitch = 0.0;
        g.apply(&mut cam);
        let p = g.surface_point(&cam, VP.0 * 0.5, VP.1 * 0.5, VP).unwrap();
        assert!(close(p, [10.0, 45.0], 1e-3), "{p:?}");
        // The camera is `distance` above the surface, on the radial line.
        let r = cam.position.length();
        assert!((r - 2500.0).abs() < 0.5, "{r}");
    }

    #[test]
    fn panning_keeps_the_grabbed_point_under_the_cursor() {
        let (mut g, mut cam) = setup();
        g.apply(&mut cam);
        let start = (300.0, 350.0);
        let grabbed = g.surface_point(&cam, start.0, start.1, VP).unwrap();
        let ev = PointerEvent {
            dx: 60.0,
            dy: -40.0,
            panning: true,
            ..Default::default()
        };
        let now = (start.0 + 60.0, start.1 - 40.0);
        g.update(ev, now, &cam, VP);
        let under = g.surface_point(&cam, now.0, now.1, VP).unwrap();
        assert!(close(grabbed, under, 0.05), "{grabbed:?} vs {under:?}");
    }

    #[test]
    fn zoom_keeps_the_point_under_the_cursor_and_nears_the_ground() {
        let (mut g, mut cam) = setup();
        g.apply(&mut cam);
        let cursor = (550.0, 250.0);
        let before = g.surface_point(&cam, cursor.0, cursor.1, VP).unwrap();
        let ev = PointerEvent {
            wheel: 300.0,
            ..Default::default()
        };
        g.update(ev, cursor, &cam, VP);
        assert!(g.distance < 1500.0);
        let after = g.surface_point(&cam, cursor.0, cursor.1, VP).unwrap();
        assert!(close(before, after, 0.05), "{before:?} vs {after:?}");
    }

    #[test]
    fn clip_planes_follow_altitude_and_reach_the_horizon() {
        let r = 6_371_000.0;
        let mut cam = PerspectiveCamera::new(50.0, 1.5, 1.0, 1e9);
        for alt in [100.0, 10_000.0, 1_000_000.0, 20_000_000.0] {
            let g = GlobeControls::new(r, 0.0, 0.0, alt);
            g.apply(&mut cam);
            g.fit_clip_planes(&mut cam);
            let horizon = ((r + alt).powi(2) - r.powi(2)).sqrt() as f32;
            assert!(
                cam.far > horizon,
                "{alt}: far {} short of the horizon {horizon}",
                cam.far
            );
            assert!(
                cam.near <= alt as f32 * 0.1 + 1.0,
                "{alt}: near {}",
                cam.near
            );
            assert!(
                cam.far / cam.near <= 1.01e4,
                "{alt}: ratio {}",
                cam.far / cam.near
            );
        }
    }

    #[test]
    fn the_sky_is_not_a_surface() {
        let (mut g, mut cam) = setup();
        g.pitch = g.max_pitch;
        g.apply(&mut cam);
        let hit = g.surface_point(&cam, VP.0 * 0.5, 0.0, VP);
        assert!(
            hit.is_none(),
            "the top edge looks past the horizon: {hit:?}"
        );
    }

    /// Ten metres above the Earth, tilted: placed against an origin at the
    /// look-at point, a point 10 cm east of it lands where the pinhole says,
    /// to a hundredth of a pixel. Placed absolutely, `f32`'s half-metre steps
    /// put it pixels off.
    #[test]
    fn a_camera_metres_off_the_earth_stays_exact_against_its_origin() {
        let mut g = GlobeControls::new(crate::marks::geo::EARTH_MEAN_RADIUS, -122.4, 37.8, 10.0);
        g.pitch = 0.6;
        g.bearing = 1.1;
        let vp = (800u32, 600u32);
        let mut cam = PerspectiveCamera::new(50.0, 800.0 / 600.0, 0.1, 100.0);
        let (target, east, _, _) = g.frame();
        let probe = add(target, scale(east, 0.1));
        // The truth, in f64: the probe in the camera's frame through the
        // pinhole.
        let (eye, fwd, up) = g.pose();
        let right = cross(fwd, up);
        let d = [probe[0] - eye[0], probe[1] - eye[1], probe[2] - eye[2]];
        let t = (cam.fov as f64 * 0.5).tan();
        let (z, x, y) = (dot(d, fwd), dot(d, right), dot(d, up));
        let truth = [
            (x / z / (t * 800.0 / 600.0) * 0.5 + 0.5) * 800.0,
            (0.5 - y / z / t * 0.5) * 600.0,
        ];
        let origin = g.origin();
        g.apply_relative(&mut cam, &origin);
        g.fit_clip_planes(&mut cam);
        let p = MarkView::new(&cam, origin, vp).project(probe).unwrap();
        let err = ((p[0] as f64 - truth[0]).powi(2) + (p[1] as f64 - truth[1]).powi(2)).sqrt();
        assert!(err < 0.01, "relative: {err} px off ({p:?} vs {truth:?})");
        g.apply(&mut cam);
        let p = MarkView::new(&cam, Origin::ZERO, vp).project(probe);
        let err = p.map_or(f64::INFINITY, |p| {
            ((p[0] as f64 - truth[0]).powi(2) + (p[1] as f64 - truth[1]).powi(2)).sqrt()
        });
        assert!(
            err > 1.0,
            "absolute placement should show f32 steps: {err} px"
        );
    }

    /// Drags and zooms work as well at the same height.
    #[test]
    fn panning_a_metre_off_the_ground_holds_the_point() {
        let mut g = GlobeControls::new(crate::marks::geo::EARTH_MEAN_RADIUS, 2.35, 48.85, 5.0);
        g.pitch = 0.5;
        let cam = PerspectiveCamera::new(50.0, VP.0 / VP.1, 0.1, 100.0);
        let start = (380.0, 360.0);
        let grabbed = g.surface_point(&cam, start.0, start.1, VP).unwrap();
        let ev = PointerEvent {
            dx: 25.0,
            dy: -15.0,
            panning: true,
            ..Default::default()
        };
        let now = (start.0 + 25.0, start.1 - 15.0);
        g.update(ev, now, &cam, VP);
        let under = g.surface_point(&cam, now.0, now.1, VP).unwrap();
        // A millimetre is about 1e-8 degrees.
        assert!(close(grabbed, under, 1e-7), "{grabbed:?} vs {under:?}");
    }
}
