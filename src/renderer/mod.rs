//! wgpu rendering: scene traversal, material pipelines, shadows, and post-fx.
//!
//! [`Renderer`] is the main entry point on native. It owns GPU pipelines for
//! each topology (triangles, lines, points, sprites, skinned meshes), uploads
//! geometry and textures through internal caches, and exposes [`Renderer::apply_postfx`]
//! for fullscreen effects used by both native [`crate::postprocessing`] and the
//! wasm `WebRenderer` path.
//!
//! [`RenderTarget`] / [`CubeRenderTarget`] are offscreen color (+ depth) buffers
//! compatible with three.js `WebGLRenderTarget` usage in the JS shim.
//!
//! On native targets, [`crate::renderer::headless`] provides [`HeadlessRenderer`] for
//! offscreen RGBA readback without managing a window surface.

/// Bloom as compute passes over a finished frame. See [`bloom`].
pub mod bloom;
mod caption_pass;
pub mod downsample;
mod gpu_mesh;
pub mod gpu_texture;
mod render_target;
/// Pack RGBA to tightly-packed RGB on the GPU, so the alpha never crosses the
/// readback. See [`rgb_pack`].
pub mod rgb_pack;
/// Depth-only ambient occlusion as a compute pass. See [`ssao`].
pub mod ssao;
// The renderer itself, in a file named after it; this module is the facade.
#[allow(clippy::module_inception)]
mod renderer;
// Public so the WGSL sources can be parse-validated by `tests/shader_validation.rs`
// without a GPU. Not part of the stable API surface.
#[doc(hidden)]
pub mod shader;

// Batteries-included headless offscreen renderer (native-only: pollster).
#[cfg(not(target_arch = "wasm32"))]
pub mod headless;

pub use downsample::Downsampler;
pub use gpu_texture::GpuCubeTexture;
#[cfg(not(target_arch = "wasm32"))]
pub use headless::{HeadlessBuilder, HeadlessConfig, HeadlessRenderer};
pub use render_target::{CubeRenderTarget, RenderTarget};
pub use renderer::{PostFxCamera, Renderer, ToneMapping};
