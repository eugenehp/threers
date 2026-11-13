//! What any headset platform reports each frame, in one shape: the head,
//! the eyes, and the controllers and hands with what they did. An
//! [`XrSource`] gives it; the browser's is `web::XrSession` (WebXR), and a
//! native one (OpenXR, visionOS's CompositorServices) gives the same, so
//! [`Panel`](super::Panel), [`PanelPass`](super::PanelPass) and an app's
//! own code run on any of them unchanged.

use super::XrEye;
use crate::math::Matrix4;

/// Which hand an input is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Handedness {
    #[default]
    None,
    Left,
    Right,
}

impl Handedness {
    /// WebXR's and OpenXR's names: `"left"`, `"right"`, anything else none.
    pub fn parse(name: &str) -> Self {
        match name {
            "left" => Handedness::Left,
            "right" => Handedness::Right,
            _ => Handedness::None,
        }
    }
}

/// How an input points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TargetRayMode {
    /// A controller or a tracked hand: always there, and it can hover.
    #[default]
    TrackedPointer,
    /// From the head, along the gaze.
    Gaze,
    /// A tap on a phone's screen in AR: there only while it touches.
    Screen,
    /// There only while it is used, as Apple Vision Pro's look-and-pinch.
    TransientPointer,
}

impl TargetRayMode {
    /// WebXR's names (`tracked-pointer`, `gaze`, `screen`, `transient-pointer`).
    pub fn parse(name: &str) -> Self {
        match name {
            "gaze" => TargetRayMode::Gaze,
            "screen" => TargetRayMode::Screen,
            "transient-pointer" => TargetRayMode::TransientPointer,
            _ => TargetRayMode::TrackedPointer,
        }
    }

    /// Whether the input exists only while it presses (a pinch, a tap): an
    /// app treats it as a finger rather than a mouse.
    pub fn is_transient(self) -> bool {
        matches!(self, TargetRayMode::Screen | TargetRayMode::TransientPointer)
    }
}

/// A controller or hand this frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct XrInput {
    /// Which controller it is (three.js's `getController(slot)`); kept
    /// while the source stays connected.
    pub slot: usize,
    pub handedness: Handedness,
    pub target_ray_mode: TargetRayMode,
    /// Where it points from, looking down its -Z, in the reference space;
    /// `None` while untracked.
    pub ray: Option<[f32; 16]>,
    /// Where it is held, in the reference space.
    pub grip: Option<[f32; 16]>,
    /// A tracked hand's joints, 16 floats each in
    /// [`HAND_JOINTS`](super::HAND_JOINTS) order, and their radii.
    pub joints: Option<(Vec<f32>, Vec<f32>)>,
    /// The gamepad's axes (thumbsticks: `[2]` and `[3]` on most
    /// controllers) and buttons (`pressed`, `value`).
    pub axes: Vec<f32>,
    pub buttons: Vec<(bool, f32)>,
}

impl XrInput {
    /// The ray's start and direction, if tracked.
    pub fn ray_line(&self) -> Option<(crate::math::Vector3, crate::math::Vector3)> {
        self.ray.as_ref().map(ray_of)
    }

    /// The thumbstick: the second pair of axes where there are two (a
    /// controller's touchpad comes first), else the first.
    pub fn thumbstick(&self) -> (f32, f32) {
        let a = &self.axes;
        if a.len() >= 4 {
            (a[2], a[3])
        } else if a.len() >= 2 {
            (a[0], a[1])
        } else {
            (0.0, 0.0)
        }
    }
}

/// A pose's position and its -Z (where a ray points).
pub fn ray_of(m: &[f32; 16]) -> (crate::math::Vector3, crate::math::Vector3) {
    use crate::math::Vector3;
    (Vector3::new(m[12], m[13], m[14]), Vector3::new(-m[8], -m[9], -m[10]).normalize())
}

/// A ray's pose: at `start`, looking down `direction` (its -Z), with its
/// +Y as near to straight up as it can be — what [`ray_of`] reads back.
/// For a ray an app makes itself: a pinch's selection ray, a tap on a
/// screen.
pub fn ray_pose(start: crate::math::Vector3, direction: crate::math::Vector3) -> [f32; 16] {
    use crate::math::Vector3;
    let z = -direction.normalize();
    let up = if z.y.abs() > 0.99 { Vector3::new(0.0, 0.0, 1.0) } else { Vector3::new(0.0, 1.0, 0.0) };
    let x = up.cross(z).normalize();
    let y = z.cross(x);
    [x.x, x.y, x.z, 0.0, y.x, y.y, y.z, 0.0, z.x, z.y, z.z, 0.0, start.x, start.y, start.z, 1.0]
}

/// What an input did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XrEventKind {
    Connected,
    Disconnected,
    /// The primary action (trigger, pinch, tap): begun, ended, and done
    /// (ended without being cancelled).
    SelectStart,
    SelectEnd,
    Select,
    /// The grip.
    SqueezeStart,
    SqueezeEnd,
    Squeeze,
}

impl XrEventKind {
    /// WebXR's event names.
    pub fn name(self) -> &'static str {
        match self {
            XrEventKind::Connected => "connected",
            XrEventKind::Disconnected => "disconnected",
            XrEventKind::SelectStart => "selectstart",
            XrEventKind::SelectEnd => "selectend",
            XrEventKind::Select => "select",
            XrEventKind::SqueezeStart => "squeezestart",
            XrEventKind::SqueezeEnd => "squeezeend",
            XrEventKind::Squeeze => "squeeze",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "connected" => XrEventKind::Connected,
            "disconnected" => XrEventKind::Disconnected,
            "selectstart" => XrEventKind::SelectStart,
            "selectend" => XrEventKind::SelectEnd,
            "select" => XrEventKind::Select,
            "squeezestart" => XrEventKind::SqueezeStart,
            "squeezeend" => XrEventKind::SqueezeEnd,
            "squeeze" => XrEventKind::Squeeze,
            _ => return None,
        })
    }
}

/// Something an input did since the last frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XrEvent {
    pub kind: XrEventKind,
    pub slot: usize,
}

/// A headset's frame, from whatever platform runs it: read inside the
/// platform's frame callback.
pub trait XrSource {
    /// The head's pose in the reference space, `None` while tracking is lost.
    fn head(&self) -> Option<[f32; 16]>;
    /// The eyes, with the reference space at `origin` in the world.
    fn eyes(&self, origin: &Matrix4) -> Vec<XrEye>;
    /// The framebuffer the eyes' viewports are in, `(width, height)`.
    fn framebuffer_size(&self) -> (u32, u32);
    /// The connected inputs.
    fn inputs(&self) -> Vec<XrInput>;
    /// What they did since the last call.
    fn take_events(&mut self) -> Vec<XrEvent>;
    /// Whether the world shows through (AR): draw no background then.
    fn see_through(&self) -> bool;
    /// Where to draw what the system highlights when it is looked at, this
    /// frame: Apple Vision Pro keeps the gaze to itself and lights up the
    /// areas an app marks, by drawing each one's value
    /// ([`declare_hover`](Self::declare_hover)) where it shows, in every
    /// eye's viewport. `None` where the platform has no such thing.
    fn hover_target(&self) -> Option<HoverTarget> {
        None
    }
    /// Marks an area `id` (not 0; the same from frame to frame for the
    /// same thing) as highlighted when looked at, this frame: the value to
    /// draw for it into the [`hover_target`](Self::hover_target), or `None`
    /// if there is none (or no more room).
    fn declare_hover(&self, id: u64) -> Option<u32> {
        let _ = id;
        None
    }
}

/// See [`XrSource::hover_target`]: a texture of integers, the size of the
/// framebuffer the eyes' viewports are in, cleared to 0 (nothing) by
/// whoever draws into it.
#[derive(Debug, Clone)]
pub struct HoverTarget {
    pub view: wgpu::TextureView,
    pub format: wgpu::TextureFormat,
    pub size: (u32, u32),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vector3;

    #[test]
    fn a_ray_pose_reads_back_as_its_ray() {
        for (start, dir) in [(Vector3::new(1.0, 2.0, 3.0), Vector3::new(0.0, 0.0, -2.0)), (Vector3::ZERO, Vector3::new(0.0, 1.0, 0.0)), (Vector3::ZERO, Vector3::new(0.3, -0.2, 0.9))] {
            let (s, d) = ray_of(&ray_pose(start, dir));
            assert!((s - start).length() < 1e-6 && (d - dir.normalize()).length() < 1e-5, "{start:?} {dir:?}");
        }
    }
}
