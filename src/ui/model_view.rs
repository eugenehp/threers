//! A threers scene shown in a UI box and turned there: [`ModelView`] renders
//! it with the GPU renderer from where an [`Orbit`] puts the camera, at the
//! box's size, into the box's image, so it shows with any painter and sits
//! among the rest of the UI (scrolling, tilting, clipped). The orbit is the
//! same one [`model_view`](crate::ui::model3d::model_view) uses: keep it in
//! the state and turn it from the pointer, a finger or the keys.
//!
//! Each [`render`](ModelView::render) waits for its frame to come back from
//! the GPU. Set [`pipelined`](ModelView::pipelined) to show the frame before
//! instead, so the wait overlaps the next frame. Under the GPU
//! [`UiPass`](crate::renderer::UiPass) there is no copy to wait for:
//! [`SurfaceView`](crate::ui::surface_view::SurfaceView) shows a view
//! drawn on a thread of its own, straight from its texture.
//!
//! ```no_run
//! use threers::prelude::*;
//! use threers::ui::model3d::Orbit;
//! use threers::ui::model_view::ModelView;
//! use threers::style;
//! use threers::ui::Ui;
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let mut view = ModelView::new(&mut ui, root, style! { width: 320px; height: 240px; });
//! let mut scene = Scene::new();
//! // ... meshes and lights ...
//! ui.layout(800.0, 600.0, &mut threers::ui::FixedWidthText(7.0));
//! view.render(&mut ui, &mut scene, &Orbit::default(), Vector3::new(0.0, 0.0, 0.0), 2.0).unwrap();
//! ```

use crate::math::Vector3;
use crate::prelude::{Camera, PerspectiveCamera, Scene};
use crate::renderer::HeadlessRenderer;
use crate::ui::image::Image;
use crate::ui::model3d::Orbit;
use crate::ui::{NodeId, Style, Ui};

/// A box showing a scene, and the renderer that draws it at the box's size.
pub struct ModelView {
    pub node: NodeId,
    renderer: Option<HeadlessRenderer>,
    size: (u32, u32),
    /// The vertical field of view, in degrees.
    pub fov: f32,
    /// Show the frame before the one just drawn, read back while this one
    /// draws, rather than wait for this one. The box lags a frame, and the
    /// first render after a resize leaves the last picture in place. Off by
    /// default.
    pub pipelined: bool,
}

impl ModelView {
    /// An image box under `parent`, styled by `style` (give it a size).
    pub fn new(ui: &mut Ui, parent: NodeId, style: Style) -> Self {
        let blank = Image::new(1, 1, vec![0; 4]).expect("sized");
        let node = ui.add_image(parent, style, blank);
        Self {
            node,
            renderer: None,
            size: (0, 0),
            fov: 40.0,
            pipelined: false,
        }
    }

    /// The camera `orbit` puts round `target`, for a box `aspect` wide to
    /// tall.
    pub fn camera(&self, orbit: &Orbit, target: Vector3, aspect: f32) -> PerspectiveCamera {
        let e = orbit.eye();
        let mut camera = PerspectiveCamera::new(self.fov, aspect, 0.05, 200.0);
        camera.position = Vector3::new(target.x + e[0], target.y + e[1], target.z + e[2]);
        camera.look_at(target);
        camera
    }

    /// Renders `scene` from `orbit` round `target` into the box, at `scale`
    /// device pixels per UI pixel, as the box is laid out now. The renderer
    /// is made (or remade) at the box's size.
    pub fn render(
        &mut self,
        ui: &mut Ui,
        scene: &mut Scene,
        orbit: &Orbit,
        target: Vector3,
        scale: f32,
    ) -> Result<(), String> {
        let r = ui.rect(self.node);
        let size = (
            ((r.width * scale).round() as u32).max(1),
            ((r.height * scale).round() as u32).max(1),
        );
        if self.renderer.is_none() || self.size != size {
            let mut hr = HeadlessRenderer::builder().size(size.0, size.1).build()?;
            hr.set_msaa(4);
            self.renderer = Some(hr);
            self.size = size;
        }
        let camera = self.camera(orbit, target, size.0 as f32 / size.1 as f32);
        let hr = self.renderer.as_mut().expect("made above");
        if self.pipelined {
            hr.render(scene, &camera as &dyn Camera);
            // The frame before this one, at the size it was drawn at: a
            // resize remakes the renderer, so nothing older is pending.
            if let Some(rgba) = hr.read_rgba_pipelined() {
                ui.set_image(self.node, Image::new(size.0, size.1, rgba)?);
            }
        } else {
            let rgba = hr.render_to_rgba(scene, &camera as &dyn Camera);
            ui.set_image(self.node, Image::new(size.0, size.1, rgba)?);
        }
        Ok(())
    }
}
