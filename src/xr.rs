//! Stereo views for headsets: what an XR device reports for each eye (a
//! projection, a pose, a rectangle of the shared framebuffer) turned into the
//! cameras this renderer draws with.
//!
//! The device speaks OpenGL conventions: its projections put the near plane
//! at clip z = -1, and its poses are in a reference space whose origin the
//! app may move (a camera rig, a teleport). [`XrEye::new`] converts the
//! first to this renderer's 0..1 depth and places the second under the
//! app's `origin`; [`XrEye::camera`] then gives a [`PerspectiveCamera`]
//! that keeps the app camera's lens and layers.
//!
//! The layers, so that one app runs on any headset platform:
//! - a platform gives each frame as an [`XrSource`] — the head, the eyes,
//!   the inputs ([`XrInput`]) and what they did ([`XrEvent`]). The browser's
//!   is `web::XrSession` (WebXR, wasm only), visionOS's is
//!   `visionos::ImmersiveSession` (feature `xr-visionos`); another native
//!   runtime (OpenXR) implements the same trait;
//! - above it nothing is platform code: any scene drawn per eye by
//!   [`Renderer::render_xr`](crate::renderer::Renderer::render_xr) (or
//!   [`XrEye::camera`] for a renderer of your own), and a flat UI as a
//!   [`Panel`] (input as a screen gives it) drawn by [`PanelPass`], alone
//!   in a room or standing in the scene.
//!
//! ```
//! use threers::xr::{XrEye, depth_zero_to_one};
//! use threers::{Matrix4, PerspectiveCamera, Vector3};
//!
//! // A symmetric 90° projection as a device sends it (OpenGL depth).
//! let (n, f) = (0.1f32, 100.0f32);
//! let gl = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
//!           0.0, 0.0, -(f + n) / (f - n), -1.0, 0.0, 0.0, -2.0 * f * n / (f - n), 0.0];
//! // The left eye 3 cm left of a head 1.6 m up.
//! let pose = Matrix4::translation(Vector3::new(-0.03, 1.6, 0.0));
//! let eye = XrEye::new(gl, &Matrix4::identity(), &pose, [0, 0, 1000, 1000]);
//! let camera = eye.camera(&PerspectiveCamera::new(50.0, 1.0, n, f));
//! assert!((camera.position.y - 1.6).abs() < 1e-6);
//! assert_eq!(camera.projection_override, Some(depth_zero_to_one(gl)));
//! ```

use crate::cameras::{PerspectiveCamera, ViewOverride};

// What a headset reports, the same from every platform, and the trait
// that gives it.
mod input;
pub use input::*;
// An app in a few lines: a scene built once and changed each frame, run
// by a platform's runner (`visionos::run`, `visionos_app!`).
mod app;
pub use app::*;
// A flat UI on a panel in a headset, and its drawing (wgpu).
mod panel;
pub use panel::*;
mod panel_pass;
pub use panel_pass::*;
// Any scene, once per eye (`Renderer::render_xr`).
mod stereo;
// A phone as a window into a scene where there is no immersive WebXR (the
// motion sensors, the camera, taps): `phone::PhoneView` in the browser.
pub mod phone;
// The browser session (WebXR), for apps and the three.js shim alike.
#[cfg(target_arch = "wasm32")]
pub mod web;
// A native visionOS immersive space (CompositorServices and ARKit).
#[cfg(all(target_os = "visionos", feature = "xr-visionos"))]
pub mod visionos;
use crate::math::{Matrix4, Vector3};

/// An OpenGL projection (clip z from -1 at the near plane to 1 at the far)
/// remapped to 0..1, this renderer's (and WebGPU's) depth range: z' = (z + w) / 2.
pub fn depth_zero_to_one(gl: [f32; 16]) -> [f32; 16] {
    let mut m = gl;
    // Column-major: row 2 of each column becomes half of rows 2 and 3.
    for c in 0..4 {
        m[c * 4 + 2] = 0.5 * (gl[c * 4 + 2] + gl[c * 4 + 3]);
    }
    m
}

/// One eye of a frame: where it is, what it sees, and where in the shared
/// framebuffer it is drawn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XrEye {
    /// The projection in this renderer's 0..1 depth.
    pub projection: [f32; 16],
    /// The eye's world matrix: the app's origin times the device's pose.
    pub world: Matrix4,
    /// `[x, y, width, height]` in framebuffer pixels, from the bottom left
    /// as the device gives it (and as viewports are set here).
    pub viewport: [i32; 4],
}

impl XrEye {
    /// An eye from the device's projection (OpenGL depth), its pose in the
    /// reference space, and its framebuffer rectangle. `origin` places the
    /// reference space in the world: identity for a fixed rig, the parent's
    /// world matrix of a camera the app moves.
    pub fn new(gl_projection: [f32; 16], origin: &Matrix4, pose: &Matrix4, viewport: [i32; 4]) -> Self {
        XrEye { projection: depth_zero_to_one(gl_projection), world: origin.multiply(pose), viewport }
    }

    /// The eye's vertical field of view, in radians, from its projection
    /// (an asymmetric frustum's total angle).
    pub fn fov(&self) -> f32 {
        let p = &self.projection;
        // tan of the top and bottom half-angles: (1 ± p[9]) / p[5].
        ((1.0 + p[9]) / p[5]).atan() + ((1.0 - p[9]) / p[5]).atan()
    }

    /// The camera to draw this eye with: `lens`'s depth of field, layers and
    /// clipping range, at the eye's pose, through the eye's projection.
    pub fn camera(&self, lens: &PerspectiveCamera) -> PerspectiveCamera {
        let mut c = lens.clone();
        let (position, target, up, view) = pose_from_world(&self.world, (lens.target - lens.position).length());
        (c.position, c.target, c.up, c.view_override) = (position, target, up, view);
        c.projection_override = Some(self.projection);
        c.fov = self.fov();
        c.aspect = self.viewport[2].max(1) as f32 / self.viewport[3].max(1) as f32;
        c.shift_x = 0.0;
        c.shift_y = 0.0;
        c
    }
}

/// The camera pose of a world matrix (looking down its -Z, up its +Y): the
/// position, a target `distance` ahead, the up vector, and, when the matrix
/// is not a rotation and translation (a scaled rig), the view to draw with.
pub fn pose_from_world(m: &Matrix4, distance: f32) -> (Vector3, Vector3, Vector3, Option<ViewOverride>) {
    let e = &m.elements;
    let position = Vector3::new(e[12], e[13], e[14]);
    let forward = Vector3::new(-e[8], -e[9], -e[10]).normalize();
    let up = Vector3::new(e[4], e[5], e[6]).normalize();
    let d = if distance.is_finite() && distance > 1e-6 { distance } else { 1.0 };
    let target = position + forward * d;
    let (x, y, z) = (Vector3::new(e[0], e[1], e[2]), Vector3::new(e[4], e[5], e[6]), Vector3::new(e[8], e[9], e[10]));
    const EPS: f32 = 1e-5;
    let rigid = [x, y, z].iter().all(|v| (v.length() - 1.0).abs() < EPS)
        && x.dot(y).abs() < EPS
        && y.dot(z).abs() < EPS
        && z.dot(x).abs() < EPS;
    let view = (!rigid).then(|| ViewOverride { view: m.invert(), position, target, up });
    (position, target, up, view)
}

/// The WebXR hand joints in the order a hand lists them (and three.js names
/// them): the wrist, then each finger from its base to its tip.
pub const HAND_JOINTS: [&str; 25] = [
    "wrist",
    "thumb-metacarpal",
    "thumb-phalanx-proximal",
    "thumb-phalanx-distal",
    "thumb-tip",
    "index-finger-metacarpal",
    "index-finger-phalanx-proximal",
    "index-finger-phalanx-intermediate",
    "index-finger-phalanx-distal",
    "index-finger-tip",
    "middle-finger-metacarpal",
    "middle-finger-phalanx-proximal",
    "middle-finger-phalanx-intermediate",
    "middle-finger-phalanx-distal",
    "middle-finger-tip",
    "ring-finger-metacarpal",
    "ring-finger-phalanx-proximal",
    "ring-finger-phalanx-intermediate",
    "ring-finger-phalanx-distal",
    "ring-finger-tip",
    "pinky-finger-metacarpal",
    "pinky-finger-phalanx-proximal",
    "pinky-finger-phalanx-intermediate",
    "pinky-finger-phalanx-distal",
    "pinky-finger-tip",
];

/// Which controller slot each input source holds. Sources come and go (a
/// controller put down, hands taken up); an app's controller `i` keeps its
/// source until that source leaves, and a new source takes the first free
/// slot, as three.js assigns them.
#[derive(Debug, Clone, Default)]
pub struct InputSlots<T> {
    slots: Vec<Option<T>>,
}

impl<T: PartialEq> InputSlots<T> {
    pub fn new() -> Self {
        InputSlots { slots: vec![] }
    }

    /// Puts `source` in the first free slot, returning it.
    pub fn connect(&mut self, source: T) -> usize {
        if let Some(i) = self.slot_of(&source) {
            return i;
        }
        match self.slots.iter().position(Option::is_none) {
            Some(i) => {
                self.slots[i] = Some(source);
                i
            }
            None => {
                self.slots.push(Some(source));
                self.slots.len() - 1
            }
        }
    }

    /// Frees `source`'s slot, returning it.
    pub fn disconnect(&mut self, source: &T) -> Option<usize> {
        let i = self.slot_of(source)?;
        self.slots[i] = None;
        Some(i)
    }

    pub fn slot_of(&self, source: &T) -> Option<usize> {
        self.slots.iter().position(|s| s.as_ref() == Some(source))
    }

    /// The occupied slots and their sources.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, s)| s.as_ref().map(|s| (i, s)))
    }

    pub fn clear(&mut self) {
        self.slots.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cameras::Camera;

    fn gl_frustum(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> [f32; 16] {
        [
            2.0 * n / (r - l), 0.0, 0.0, 0.0,
            0.0, 2.0 * n / (t - b), 0.0, 0.0,
            (r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n), -1.0,
            0.0, 0.0, -2.0 * f * n / (f - n), 0.0,
        ]
    }

    fn clip(m: &[f32; 16], p: [f32; 3]) -> [f32; 4] {
        let mut o = [0.0; 4];
        for (r, v) in o.iter_mut().enumerate() {
            *v = m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r];
        }
        o
    }

    #[test]
    fn near_and_far_land_on_zero_and_one() {
        let (n, f) = (0.1, 50.0);
        let p = depth_zero_to_one(gl_frustum(-0.12, 0.08, -0.1, 0.1, n, f));
        for (z, want) in [(-n, 0.0), (-f, 1.0)] {
            let c = clip(&p, [0.0, 0.0, z]);
            assert!((c[2] / c[3] - want).abs() < 1e-5, "z {z}: {}", c[2] / c[3]);
        }
        // x and y are untouched.
        let (g, c) = (clip(&gl_frustum(-0.12, 0.08, -0.1, 0.1, n, f), [0.3, -0.2, -2.0]), clip(&p, [0.3, -0.2, -2.0]));
        assert_eq!((g[0], g[1], g[3]), (c[0], c[1], c[3]));
    }

    #[test]
    fn a_symmetric_projection_matches_the_renderers_own() {
        let (n, f) = (0.1f32, 100.0f32);
        let t = n * (30f32.to_radians()).tan();
        let eye = XrEye::new(gl_frustum(-t, t, -t, t, n, f), &Matrix4::identity(), &Matrix4::identity(), [0, 0, 8, 8]);
        let ours = Matrix4::perspective(60f32.to_radians(), 1.0, n, f);
        for (a, b) in eye.projection.iter().zip(ours.elements) {
            assert!((a - b).abs() < 1e-4, "{:?} vs {:?}", eye.projection, ours.elements);
        }
        assert!((eye.fov() - 60f32.to_radians()).abs() < 1e-5);
    }

    #[test]
    fn the_eye_sits_under_the_origin_and_keeps_the_lens() {
        let origin = Matrix4::translation(Vector3::new(10.0, 0.0, 0.0));
        let pose = Matrix4::translation(Vector3::new(0.032, 1.6, 0.5));
        let eye = XrEye::new(gl_frustum(-0.1, 0.1, -0.1, 0.1, 0.1, 10.0), &origin, &pose, [960, 0, 960, 1080]);
        let mut lens = PerspectiveCamera::new(50.0, 1.0, 0.1, 10.0);
        lens.aperture = 0.02;
        let c = eye.camera(&lens);
        assert!((c.position - Vector3::new(10.032, 1.6, 0.5)).length() < 1e-5);
        assert!(c.view_override.is_none(), "a rigid pose views by look_at");
        assert_eq!(c.aperture, 0.02);
        assert_eq!(c.projection_matrix().elements, eye.projection);
        // The view takes the eye's own position to the view-space origin.
        let v = c.view_matrix();
        let at = clip(&v.elements, [10.032, 1.6, 0.5]);
        assert!(at[..3].iter().all(|x| x.abs() < 1e-4), "{at:?}");
    }

    #[test]
    fn a_scaled_rig_views_through_its_inverse() {
        let origin = Matrix4::scale(Vector3::new(2.0, 2.0, 2.0));
        let eye = XrEye::new(gl_frustum(-0.1, 0.1, -0.1, 0.1, 0.1, 10.0), &origin, &Matrix4::identity(), [0, 0, 8, 8]);
        let c = eye.camera(&PerspectiveCamera::new(50.0, 1.0, 0.1, 10.0));
        let v = c.view_matrix();
        let p = clip(&v.elements, [0.0, 0.0, -2.0]);
        assert!((p[2] + 1.0).abs() < 1e-5, "world -2 is one unit ahead of a doubled rig: {p:?}");
    }

    #[test]
    fn slots_keep_their_sources() {
        let mut s = InputSlots::new();
        assert_eq!(s.connect("left"), 0);
        assert_eq!(s.connect("right"), 1);
        assert_eq!(s.disconnect(&"left"), Some(0));
        assert_eq!(s.slot_of(&"right"), Some(1), "the right stays put");
        assert_eq!(s.connect("hand"), 0, "a new source takes the free slot");
        assert_eq!(s.connect("hand"), 0, "connecting twice is once");
        assert_eq!(s.iter().map(|(i, _)| i).collect::<Vec<_>>(), [0, 1]);
    }
}
