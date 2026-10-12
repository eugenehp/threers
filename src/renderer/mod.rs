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

/// What the device can do — compute and storage, which WebGL2 lacks. See
/// [`caps`].
pub mod caps;
// No outer doc comments on these three: each carries `//!` docs of its own,
// and an outer `///` here makes the intra-doc links inside them resolve in
// *this* scope instead of the module's. The warning for that names no file.
pub mod colorspace;
pub mod fullscreen;
pub mod readback;
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
/// GPU-side timing for the main pass. See [`gpu_timer`].
pub mod gpu_timer;
// The renderer itself, in a file named after it; this module is the facade.
#[allow(clippy::module_inception)]
mod renderer;
// Shader-material assembly and validation, for `ShaderMaterial::validate`.
// Not part of the public surface: the module layout is an implementation
// detail and the only supported entry point is that method.
pub(crate) use renderer::{
    custom_preamble_lines, custom_shader_module_source, validate_custom_shader,
};
// Public so the WGSL sources can be parse-validated by `tests/shader_validation.rs`
// without a GPU. Not part of the stable API surface.
#[doc(hidden)]
pub mod shader;

// Batteries-included headless offscreen renderer (native-only: pollster).
#[cfg(not(target_arch = "wasm32"))]
pub mod headless;

// Multi-sample accumulation for the offline renderer. Only `headless` reads it,
// and that is native-only, so this follows the same gate — otherwise it is dead
// code on wasm32 and weight in the bundle for nothing.
#[cfg(not(target_arch = "wasm32"))]
mod accumulate;

/// Whether the *shader* has to apply the sRGB transfer curve itself.
///
/// Three storage conventions live side by side and the difference matters
/// everywhere a value is written, sampled, blended or resolved:
///
/// * `Rgba16Float` stores linear light. Nothing converts, in either direction.
/// * An sRGB-suffixed format stores encoded bytes and the *hardware* converts:
///   it encodes on write, decodes on sample, and — the reason this predicate
///   exists — decodes, averages and re-encodes on a multisample resolve.
/// * A plain `Unorm` format stores encoded bytes and the hardware does not know
///   it. The shader has to encode on the way out and decode on the way in, and
///   any hardware operation over those bytes is a mean over a curve.
///
/// Only the third needs the shader's help, and it is one predicate rather than
/// two because the write and the read always agree: whoever encodes decodes.
/// It used to be spelled `format != Rgba16Float` in two places, which quietly
/// made an sRGB target *double*-encode — a 50% coverage edge stored 0.7379
/// where the light was 0.5 — and double-decode on the way back.
pub(crate) fn shader_applies_transfer(format: wgpu::TextureFormat) -> bool {
    format != wgpu::TextureFormat::Rgba16Float && !format.is_srgb()
}
pub use downsample::{DownsampleFilter, Downsampler};
pub use gpu_texture::GpuCubeTexture;
#[cfg(not(target_arch = "wasm32"))]
pub use headless::{HeadlessBuilder, HeadlessConfig, HeadlessRenderer};
pub use render_target::{CubeRenderTarget, RenderTarget};
pub use renderer::{PostFxCamera, Renderer, ToneMapping};
pub use renderer::{halton, stratified_jitter};
