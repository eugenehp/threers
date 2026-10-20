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
// `set_viewport` / `set_scissor` state and the partial clear behind them.
pub mod viewport;
/// Pack RGBA to tightly-packed RGB on the GPU, so the alpha never crosses the
/// readback. See [`rgb_pack`].
pub mod rgb_pack;
/// Depth-only ambient occlusion as a compute pass. See [`ssao`].
pub mod ssao;
/// [`ssao`] composited onto a live surface. See [`ssao_composite`].
pub mod ssao_composite;
/// GPU-side timing for the main pass. See [`gpu_timer`].
pub mod gpu_timer;
// The renderer itself, in a file named after it; this module is the facade.
#[allow(clippy::module_inception)]
mod renderer;
// Material depth/colour-write state as pipeline variants.
pub mod material_maps;
mod pipeline_state;
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

/// The value to clear a `format` target to so it ends up holding the
/// *linear* colour `color` — `scene.background`, a `clear()` colour, a
/// viewport gap — the way everything else drawn into it does.
///
/// [`Color`](crate::math::Color) is linear ([`from_hex`](crate::math::Color::from_hex)
/// decodes, as three.js does with colour management on), so this follows
/// [`shader_applies_transfer`]: an sRGB or float target takes the linear value
/// and the hardware (or a later pass) encodes it; a plain `Unorm` target —
/// a WebGPU canvas is usually `bgra8unorm` — is written verbatim, so the
/// value is encoded here, exactly as the shader encodes its own output.
/// A hex background therefore lands on the image as that hex on either kind,
/// which is three.js's behaviour (`WebGLBackground` converts the clear colour
/// to the output colour space before `gl.clearColor`).
///
/// The opposite rule — `powf(2.2)` on sRGB targets, verbatim on `Unorm` —
/// treats the colour as if it were still sRGB-encoded: a `from_hex`
/// background is decoded twice and comes out near black.
pub(crate) fn clear_value(
    format: wgpu::TextureFormat,
    color: crate::math::Color,
    alpha: f32,
) -> wgpu::Color {
    let c = |v: f32| {
        if shader_applies_transfer(format) {
            crate::utils::srgb::linear_to_srgb_f32(v) as f64
        } else {
            v as f64
        }
    };
    wgpu::Color {
        r: c(color.r),
        g: c(color.g),
        b: c(color.b),
        a: alpha as f64,
    }
}
pub use downsample::{DownsampleFilter, Downsampler};
pub use gpu_texture::GpuCubeTexture;
#[cfg(not(target_arch = "wasm32"))]
pub use headless::{HeadlessBuilder, HeadlessConfig, HeadlessRenderer};
pub use render_target::{CubeRenderTarget, RenderTarget};
pub use viewport::{PixelRect, ShadowMap, ViewportRect};
pub use renderer::{PostFxCamera, Renderer, ToneMapping};
pub use renderer::{PostFxBlend, PostFxNoise, PostFxOp, PostFxOutput};
pub use renderer::{halton, stratified_jitter};

#[cfg(test)]
mod clear_value_tests {
    use super::clear_value;
    use crate::math::Color;
    use wgpu::TextureFormat as F;

    /// The canvas formats a browser hands out, which the headless tests cannot
    /// build: plain `bgra8unorm` must get the encoded value, the sRGB variant
    /// the linear one, and both must come out as the hex.
    #[test]
    fn a_hex_clear_lands_as_the_hex_on_canvas_formats() {
        for hex in [0x20262c_u32, 0x0a0b0c, 0xff8800, 0x808080] {
            let c = Color::from_hex(hex);
            let want = [(hex >> 16) & 0xff, (hex >> 8) & 0xff, hex & 0xff];
            let plain = clear_value(F::Bgra8Unorm, c, 1.0);
            for (v, w) in [plain.r, plain.g, plain.b].into_iter().zip(want) {
                assert!((v * 255.0 - w as f64).abs() < 0.01, "#{hex:06x}: {v}");
            }
            for f in [F::Bgra8UnormSrgb, F::Rgba16Float] {
                let lin = clear_value(f, c, 1.0);
                assert_eq!([lin.r, lin.g, lin.b], [c.r as f64, c.g as f64, c.b as f64]);
            }
        }
    }
}
