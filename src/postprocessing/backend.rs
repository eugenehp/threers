//! What the composer asks of a renderer, so a chain of passes runs on any
//! backend that can answer it.
//!
//! A pass describes its work as [`PostFxOp`]s (one fullscreen draw of the
//! post-fx shader's kinds), scene renders into a [`Target`], and a handful of
//! uploads. Everything it needs from the GPU goes through this trait: the wgpu
//! [`Renderer`] implements it, and so does the native Metal backend
//! (`crate::metal::MetalPostBackend`), which runs the same post-fx shader,
//! translated, over Metal textures.

use super::target::Target;
use crate::cameras::Camera;
use crate::renderer::{PostFxOp, PostFxOutput, Renderer};
use crate::scene::Scene;

/// A renderer an [`EffectComposer`](super::EffectComposer) can drive.
pub trait PostBackend {
    /// A new `width × height` target of `format`, registered under a fresh id
    /// so [`PostFxOp`]s can name it.
    fn new_target(&mut self, width: u32, height: u32, format: wgpu::TextureFormat) -> Target;

    /// The target registered under `id` by someone else — a texture that is
    /// already a render target.
    fn registered_target(&self, id: u32) -> Option<Target>;

    /// Forget a target, freeing its allocation.
    fn unregister_render_target(&mut self, id: u32);

    /// Overwrite a half-float target's texels, row by row.
    fn upload_render_target_f16(&mut self, id: u32, texels: &[[f32; 4]]);

    /// The kernel the SSAO, bloom and convolution kinds read.
    fn set_postfx_kernel(&mut self, data: &[[f32; 4]]);

    /// The SSAO kind's rotation noise.
    fn set_ssao_noise(&mut self, noise: &[f32]);

    /// The glitch kinds' displacement map, `size × size`.
    fn set_glitch_disp(&mut self, data: &[f32], size: u32);

    /// One post-fx draw.
    fn run_postfx(&mut self, op: &PostFxOp, output: PostFxOutput<'_>);

    /// Clear an output to `color` (linear).
    fn clear_postfx_output(&mut self, output: PostFxOutput<'_>, color: [f64; 4], display: bool);

    /// three.js `renderer.autoClear`.
    fn auto_clear(&self) -> bool;
    fn set_auto_clear(&mut self, enabled: bool);

    /// A sub-pixel projection offset for the next scene renders, in pixels
    /// (y up); `None` puts it back.
    fn set_jitter_override(&mut self, offset: Option<(f32, f32)>);

    /// Draw `scene` into `target`.
    fn render_to_target(&mut self, scene: &mut Scene, camera: &dyn Camera, target: &Target);

    /// Draw `scene` into a wgpu view — a composer's screen given as one.
    /// Only the wgpu backend has such a screen; the others ignore it.
    fn render_to_view(&mut self, scene: &mut Scene, camera: &dyn Camera, view: &wgpu::TextureView);

    /// The wgpu renderer, for a pass that encodes raw wgpu commands itself
    /// (`ShaderPass` with a draw callback). `None` on other backends, which
    /// skip such a pass.
    fn wgpu(&mut self) -> Option<&mut Renderer> {
        None
    }
}

impl PostBackend for Renderer {
    fn new_target(&mut self, width: u32, height: u32, format: wgpu::TextureFormat) -> Target {
        Target::new(self, width, height, format)
    }

    fn registered_target(&self, id: u32) -> Option<Target> {
        self.registered_render_target(id)
            .map(|rt| Target::from_registered(id, rt))
    }

    fn unregister_render_target(&mut self, id: u32) {
        Renderer::unregister_render_target(self, id)
    }

    fn upload_render_target_f16(&mut self, id: u32, texels: &[[f32; 4]]) {
        Renderer::upload_render_target_f16(self, id, texels)
    }

    fn set_postfx_kernel(&mut self, data: &[[f32; 4]]) {
        Renderer::set_postfx_kernel(self, data)
    }

    fn set_ssao_noise(&mut self, noise: &[f32]) {
        Renderer::set_ssao_noise(self, noise)
    }

    fn set_glitch_disp(&mut self, data: &[f32], size: u32) {
        Renderer::set_glitch_disp(self, data, size)
    }

    fn run_postfx(&mut self, op: &PostFxOp, output: PostFxOutput<'_>) {
        Renderer::run_postfx(self, op, output)
    }

    fn clear_postfx_output(&mut self, output: PostFxOutput<'_>, color: [f64; 4], display: bool) {
        Renderer::clear_postfx_output(self, output, color, display)
    }

    fn auto_clear(&self) -> bool {
        Renderer::auto_clear(self)
    }

    fn set_auto_clear(&mut self, enabled: bool) {
        Renderer::set_auto_clear(self, enabled)
    }

    fn set_jitter_override(&mut self, offset: Option<(f32, f32)>) {
        Renderer::set_jitter_override(self, offset)
    }

    fn render_to_target(&mut self, scene: &mut Scene, camera: &dyn Camera, target: &Target) {
        if let Some(rt) = target.wgpu_render_target() {
            self.render_to(scene, camera, rt);
        }
    }

    fn render_to_view(&mut self, scene: &mut Scene, camera: &dyn Camera, view: &wgpu::TextureView) {
        self.render(scene, camera, view, false);
    }

    fn wgpu(&mut self) -> Option<&mut Renderer> {
        Some(self)
    }
}
