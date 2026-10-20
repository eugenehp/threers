//! Post-processing: three.js's `EffectComposer` and its passes.
//!
//! The same implementation serves native Rust and the browser — the
//! JavaScript shim's `EffectComposer`, `RenderPass`, `UnrealBloomPass` and
//! the rest are thin wrappers over these, forwarding their properties and
//! uniforms through [`Pass::get`] / [`Pass::set`].
//!
//! Every pass draws through [`Renderer::run_postfx`](crate::renderer::Renderer::run_postfx):
//! one fullscreen module (`POSTFX_SHADER`) with a branch per three.js shader,
//! the passes' sequencing — buffers, ping-pong, mip chains, prepasses — here.
//! The composer's buffers are half-float, as three.js's are, so a chain
//! carries linear light; the last pass writes the screen the way three.js's
//! does, which on WebGL2 and WebGPU alike stores the same bytes.
//!
//! ```no_run
//! use threers::postprocessing::{EffectComposer, RenderPass, UnrealBloomPass, OutputPass};
//! # fn demo(renderer: &mut threers::Renderer, scene: &mut threers::Scene,
//! #         camera: &threers::PerspectiveCamera, view: &threers::wgpu::TextureView) {
//! let mut composer = EffectComposer::with_size(1280, 720);
//! composer.add_pass(RenderPass::new());
//! composer.add_pass(UnrealBloomPass::new(1280, 720, 1.5, 0.4, 0.85));
//! composer.add_pass(OutputPass::new());
//! composer.set_screen_format(threers::wgpu::TextureFormat::Bgra8Unorm);
//! composer.render(renderer, scene, camera, view);
//! # }
//! ```

mod composer;
mod context;
mod pass;
mod passes;
pub mod params;
pub mod random;
mod target;
mod upload;

pub use composer::EffectComposer;
pub use context::{kinds, Dest, Fx, PassContext, SceneDraw, Screen, Visibility};
pub use params::{ParamInfo, ParamKind, ParamValue};
pub use pass::{MaskRole, Pass, PassState, PassThreading};
pub use passes::*;
pub use target::{ensure as ensure_target, release as release_target, Target, INTERMEDIATE_FORMAT};
pub use upload::{cube_atlas, texels, texture_target};

/// `ShaderPass(FXAAShader)` sized to `width × height` — the name the native
/// API had for it.
pub type FxaaPass = ShaderPass;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
