//! An XR app in a few lines: build a scene once, change it each frame; a
//! platform's runner does the rest — the session, the frame loop, drawing
//! each eye, putting the frame on the headset, the inputs.
//!
//! ```
//! use threers::prelude::*;
//! use threers::xr::{App, Frame};
//!
//! struct Spinner {
//!     cube: ObjectId,
//! }
//!
//! impl App for Spinner {
//!     fn setup(scene: &mut Scene) -> Self {
//!         scene.add_light(HemisphereLight::new(Color::from_hex(0xffffff), Color::from_hex(0x444444), 1.0));
//!         let mut cube = Object3D::mesh(Mesh::new(BoxGeometry::new(0.3, 0.3, 0.3), StandardMaterial::new(Color::from_hex(0x4fa3e8)).into()));
//!         cube.position = Vector3::new(0.0, 1.4, -1.0);
//!         Spinner { cube: scene.add(cube) }
//!     }
//!
//!     fn update(&mut self, scene: &mut Scene, frame: &Frame) {
//!         if let Some(cube) = scene.get_mut(self.cube) {
//!             cube.quaternion = Euler::new(0.0, frame.time as f32, 0.0).to_quaternion();
//!         }
//!     }
//! }
//!
//! // On Apple Vision Pro, one line makes it the app (see `visionos_app!`):
//! // threers::visionos_app!(Spinner);
//! # let mut scene = Scene::new();
//! # let mut app = Spinner::setup(&mut scene);
//! # app.update(&mut scene, &Frame::default());
//! ```

use super::{XrEvent, XrEventKind, XrInput, XrSource};
use crate::cameras::PerspectiveCamera;
use crate::math::{Matrix4, Vector3};
use crate::scene::Scene;

/// An app a headset runs: see the module docs.
pub trait App: Sized + 'static {
    /// The scene, built once before the first frame.
    fn setup(scene: &mut Scene) -> Self;

    /// Each frame, before it is drawn: the time, the head, the hands and
    /// controllers and what they did are in `frame`.
    fn update(&mut self, _scene: &mut Scene, _frame: &Frame) {}

    /// The lens the eyes see through: its near and far planes (the
    /// headset's projections use them), depth of field, layers.
    fn lens(&self) -> PerspectiveCamera {
        PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0)
    }

    /// Where the viewer's space is in the world: identity to stand still,
    /// a rig's world matrix to walk, fly or ride through the scene.
    fn origin(&self) -> Matrix4 {
        Matrix4::identity()
    }

    /// Draws the frame into `gpu.target`, each of `gpu.source`'s eyes in
    /// its viewport, after [`update`](Self::update). By default the scene,
    /// through [`lens`](Self::lens) at [`origin`](Self::origin); override it
    /// to draw more — a UI on a [`Panel`](super::Panel) standing in the
    /// scene ([`PanelPass::draw_in_scene`](super::PanelPass::draw_in_scene))
    /// — or to draw differently. Headset runners call it, and so do
    /// threers-ios's views, as one eye (with no head in a window: draw a
    /// UI flat over it there).
    fn draw(&mut self, scene: &mut Scene, _frame: &Frame, gpu: &mut Gpu<'_>) {
        let (lens, origin) = (self.lens(), self.origin());
        gpu.renderer.render_xr(scene, &lens, gpu.source, &origin, gpu.target);
    }

    /// Where a flat screen's camera looks, with nobody's head to go by (an
    /// app in a window: threers-ios's orbit view): a centre and a
    /// radius to fit in view. `None` until there is something to frame; the
    /// view frames it the first time it is `Some`.
    fn focus(&self) -> Option<(Vector3, f32)> {
        None
    }
}

/// What [`App::draw`] draws with: the GPU, the renderer (made by the
/// runner, for the target's format), the frame's texture, and the
/// platform's eyes.
pub struct Gpu<'a> {
    pub device: &'a std::sync::Arc<wgpu::Device>,
    pub queue: &'a std::sync::Arc<wgpu::Queue>,
    /// The target's format.
    pub format: wgpu::TextureFormat,
    pub target: &'a wgpu::TextureView,
    pub renderer: &'a mut crate::renderer::Renderer,
    /// Where each eye is, and the framebuffer they share.
    pub source: &'a dyn XrSource,
}

/// A frame, as an [`App`] sees it.
#[derive(Debug, Clone, Default)]
pub struct Frame {
    /// Seconds since the app started, and since the last frame.
    pub time: f64,
    pub dt: f64,
    /// The head's pose in the viewer's space, when known.
    pub head: Option<[f32; 16]>,
    /// The hands and controllers now (see [`XrInput`]).
    pub inputs: Vec<XrInput>,
    /// What they did since the last frame.
    pub events: Vec<XrEvent>,
    /// The world shows through (AR, a mixed space): draw no background.
    pub see_through: bool,
}

impl Frame {
    /// This frame from `source` — its head, inputs, and the events since the
    /// last (taken: call it once a frame) — at `time`, `dt` seconds after the
    /// last. What a runner hands [`App::update`].
    pub fn take(source: &mut (impl XrSource + ?Sized), time: f64, dt: f64) -> Frame {
        Frame { time, dt, head: source.head(), inputs: source.inputs(), events: source.take_events(), see_through: source.see_through() }
    }

    /// The rays (start, direction, in the viewer's space) of the inputs
    /// that selected this frame — a pinch let go, a trigger pulled and
    /// released: what most apps mean by "tapped".
    pub fn selects(&self) -> Vec<(Vector3, Vector3)> {
        self.events
            .iter()
            .filter(|e| e.kind == XrEventKind::Select)
            .filter_map(|e| self.inputs.iter().find(|i| i.slot == e.slot)?.ray_line())
            .collect()
    }

    /// Where the head is, in the viewer's space.
    pub fn head_position(&self) -> Option<Vector3> {
        self.head.map(|m| Vector3::new(m[12], m[13], m[14]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xr::TargetRayMode;

    #[test]
    fn a_select_gives_its_ray() {
        let ray = Matrix4::translation(Vector3::new(0.0, 1.5, 0.0)).elements;
        let frame = Frame {
            inputs: vec![XrInput { slot: 3, target_ray_mode: TargetRayMode::TransientPointer, ray: Some(ray), ..Default::default() }],
            events: vec![XrEvent { kind: XrEventKind::SelectStart, slot: 3 }, XrEvent { kind: XrEventKind::Select, slot: 3 }],
            ..Default::default()
        };
        let s = frame.selects();
        assert_eq!(s.len(), 1, "one select, not the start");
        assert!((s[0].0 - Vector3::new(0.0, 1.5, 0.0)).length() < 1e-6);
        assert!((s[0].1 - Vector3::new(0.0, 0.0, -1.0)).length() < 1e-6, "looking down -Z");
    }
}
