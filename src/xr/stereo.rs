//! Any threers scene in a headset: [`Renderer::render_xr`] draws it once
//! per eye of an [`XrSource`], whichever platform gives the eyes, into
//! whatever texture the platform presents (on the web, the canvas the
//! session copies to the headset).

use super::XrSource;
use crate::cameras::PerspectiveCamera;
use crate::math::Matrix4;
use crate::renderer::Renderer;
use crate::scene::Scene;

impl Renderer {
    /// Draws `scene` into `target` once per eye of `source`, each in its
    /// viewport (cleared first), through `camera`'s lens — its depth of
    /// field, layers and near/far — at the eye's pose, with the reference
    /// space at `origin` in the world (identity for a fixed rig; a moving
    /// rig's world matrix to walk or fly the viewer through the scene).
    ///
    /// The renderer is sized to the source's framebuffer; viewport,
    /// scissor, auto-clear and pixel ratio are put back after. Returns
    /// whether any eye was drawn (none while tracking is lost).
    pub fn render_xr(
        &mut self,
        scene: &mut Scene,
        camera: &PerspectiveCamera,
        source: &(impl XrSource + ?Sized),
        origin: &Matrix4,
        target: &wgpu::TextureView,
    ) -> bool {
        self.render_xr_with(scene, camera, source, origin, target, |_, _, _| {})
    }

    /// [`render_xr`](Self::render_xr), calling `after_eye(renderer, i,
    /// eye)` once each eye is drawn, while the renderer still holds that
    /// eye's depth ([`overlay_attachments`](Self::overlay_attachments)):
    /// for drawing more into the eye, depth-tested against the scene, as
    /// [`PanelPass::draw_in_scene`](super::PanelPass::draw_in_scene) does.
    pub fn render_xr_with(
        &mut self,
        scene: &mut Scene,
        camera: &PerspectiveCamera,
        source: &(impl XrSource + ?Sized),
        origin: &Matrix4,
        target: &wgpu::TextureView,
        mut after_eye: impl FnMut(&Renderer, usize, &super::XrEye),
    ) -> bool {
        let eyes = source.eyes(origin);
        if eyes.is_empty() {
            return false;
        }
        let (w, h) = source.framebuffer_size();
        self.resize(w, h);
        let saved = (
            self.viewport(),
            self.scissor(),
            self.scissor_test(),
            self.auto_clear(),
            self.pixel_ratio(),
        );
        self.set_pixel_ratio(1.0);
        self.set_auto_clear(true);
        self.set_scissor_test(true);
        for (i, eye) in eyes.iter().enumerate() {
            let [x, y, vw, vh] = eye.viewport.map(|v| v as f32);
            self.set_viewport(x, y, vw, vh);
            self.set_scissor(x, y, vw, vh);
            self.render(scene, &eye.camera(camera), target, false);
            after_eye(self, i, eye);
        }
        let (viewport, scissor, scissor_test, auto_clear, pixel_ratio) = saved;
        match viewport {
            Some(v) => self.set_viewport(v.x, v.y, v.width, v.height),
            None => self.reset_viewport(),
        }
        match scissor {
            Some(v) => self.set_scissor(v.x, v.y, v.width, v.height),
            None => self.reset_scissor(),
        }
        self.set_scissor_test(scissor_test);
        self.set_auto_clear(auto_clear);
        self.set_pixel_ratio(pixel_ratio);
        true
    }
}
