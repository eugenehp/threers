//! A Metal renderer for Apple platforms, written against the Objective-C
//! runtime directly — no `metal-rs`, no `objc` crate, no build script.
//!
//! Behind the `metal` feature, and compiled only for macOS and iOS. The wgpu
//! [`Renderer`](crate::renderer::Renderer) remains the default and the complete
//! one; this is a second, self-contained path to the same scene graph for
//! callers who want to be on Metal with nothing between them and it.
//!
//! # Why you might want it
//!
//! - **No dependencies.** The whole backend is `objc_msgSend` plus
//!   `#[link(name = "Metal", kind = "framework")]`. Nothing to audit but this
//!   directory, and no C or C++ toolchain in the build.
//! - **Metal-native objects.** [`crate::metal::PassAttachments::from_raw`] takes textures an
//!   app already owns, so a scene can be drawn into someone else's frame —
//!   an AVFoundation pipeline, a SceneKit view, a Core Image chain.
//! - **A small surface to read.** One MSL file, one draw loop.
//!
//! # Getting a frame
//!
//! Offscreen, which needs no window and no main thread:
//!
//! ```no_run
//! # #[cfg(all(feature = "metal", target_os = "macos"))] {
//! use threers::metal::MetalHeadlessRenderer;
//! use threers::{Color, Material, Mesh, Object3D, PerspectiveCamera, Scene};
//! use threers::geometries::BoxGeometry;
//! use threers::materials::StandardMaterial;
//! use threers::lights::DirectionalLight;
//!
//! let mut hr = MetalHeadlessRenderer::builder().size(512, 512).msaa(4).build()?;
//! let mut scene = Scene::new();
//! scene.add(Object3D::mesh(Mesh::new(
//!     BoxGeometry::new(1.0, 1.0, 1.0),
//!     Material::Standard(StandardMaterial::new(Color::from_hex(0xff8844))),
//! )));
//! scene.add_light(DirectionalLight::new(Color::WHITE, 3.0));
//!
//! let mut camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
//! camera.position = threers::math::Vector3::new(2.0, 2.0, 3.0);
//! let rgba = hr.render_to_rgba(&mut scene, &camera)?;
//! std::fs::write("cube.png", threers::encode_png(512, 512, &rgba))?;
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! On screen, given an `NSView` from a window toolkit:
//!
//! ```no_run
//! # #[cfg(all(feature = "metal", target_os = "macos"))] {
//! # use threers::metal::{MetalDevice, MetalRenderer, MetalSurface};
//! # use threers::{PerspectiveCamera, Scene};
//! # fn demo(ns_view: *mut std::ffi::c_void, scene: &mut Scene, camera: &PerspectiveCamera)
//! # -> Result<(), threers::metal::MetalError> {
//! let device = MetalDevice::new()?;
//! let mut renderer = MetalRenderer::with_device(device.clone())?;
//! let surface = unsafe { MetalSurface::from_ns_view(&device, ns_view, 1280, 720, 4)? };
//!
//! if let Some(frame) = surface.next_frame() {
//!     renderer.render(scene, camera, &frame.attachments())?;   // presents the drawable
//! }
//! # Ok(()) }
//! # }
//! ```
//!
//! # Stereo, and visionOS
//!
//! [`crate::metal::MetalRenderer::render_views`] takes a view per eye. When they are
//! *layered* — one texture array, a slice each, same viewport — both are drawn
//! in **one pass**: the vertex stage reads its eye out of the instance id and
//! writes `render_target_array_index`, so the second eye costs pixels and not a
//! second traversal. Eyes with a texture or a viewport of their own get a pass
//! each instead, and the clear happens once per target.
//!
//! [`crate::metal::PassAttachments::with_reverse_z`] flips the depth convention to near-at-1,
//! which is the only one visionOS accepts.
//!
//! The `visionos` feature builds on both: see [`crate::metal::visionos`] for the
//! CompositorServices frame loop and ARKit world tracking.
//!
//! # What it draws
//!
//! Meshes, instanced meshes, line segments, points and sprites; `Basic`,
//! `Lambert`, `Phong`, `Standard`, `Physical`, `Normal`, `Depth`, `Toon`,
//! `Matcap`, `Line`, `Points` and `Sprite` materials; one base-colour map per
//! material; ambient, directional, point, spot and hemisphere lights; linear
//! and exponential fog; alpha blending, alpha test, per-vertex colour,
//! wireframe, and MSAA.
//!
//! Deliberately not here — reach for [`crate::renderer::Renderer`] instead:
//! shadow maps, post-processing, environment maps and IBL, skinning, morph
//! targets, and the normal / roughness / metalness / AO / emissive map slots.
//! Materials outside the list draw unlit in their base colour rather than
//! failing, so a scene authored for the wgpu path still renders.
//!
//! # Conventions
//!
//! Shared with the wgpu backend, so the two agree on a scene: clip-space depth
//! in `[0, 1]`, counter-clockwise front faces, top-left texture origin (with
//! [`Texture::flip_y`](crate::textures::Texture::flip_y) applied at upload),
//! light colours premultiplied by intensity, and three.js's punctual
//! attenuation.
//!
//! # Threads
//!
//! Every type here holds Objective-C pointers and none is `Send` or `Sync`.
//! Drive a renderer from the thread that made it; [`crate::metal::MetalSurface::from_ns_view`]
//! additionally needs the main thread, because attaching a layer to a view is
//! an AppKit call.
//!
//! # Layout
//!
//! | Module | Role |
//! |--------|------|
//! | [`crate::metal::objc`] | `objc_msgSend`, selectors, retain/release, autorelease pools |
//! | [`crate::metal::enums`] | Metal enumerants and by-value structs |
//! | [`crate::metal::resources`] | Textures, samplers, pipeline states |
//! | — | [`crate::metal::MetalDevice`], [`crate::metal::MetalRenderer`], [`crate::metal::MetalRenderTarget`], [`crate::metal::MetalSurface`] |

pub mod objc;

pub mod enums;
pub mod resources;

/// visionOS: the CompositorServices frame loop and ARKit world tracking.
///
/// Behind the `visionos` feature. The value types compile for every Apple
/// target; `ImmersiveRenderer` and
/// `WorldTracking` exist only on visionOS itself.
#[cfg(feature = "visionos")]
pub mod visionos;

mod device;
mod headless;
mod renderer;
mod surface;
mod target;

pub use device::{MetalDevice, MetalError};
pub use headless::{MetalHeadlessBuilder, MetalHeadlessConfig, MetalHeadlessRenderer};
pub use renderer::{
    MetalRenderStats, MetalRenderer, PassAttachments, RenderView, DEFAULT_CACHE_RETENTION,
    MAX_VIEWS,
};
pub use surface::{MetalSurface, SurfaceFrame};
pub use target::MetalRenderTarget;

/// The Metal Shading Language source compiled by [`MetalDevice::new`].
///
/// Exposed so it can be inspected, diffed against the WGSL, or fed to
/// [`MetalDevice::with_shader_source`] with local edits — the entry points
/// (`vs_mesh`, `fs_mesh`, `vs_point`, `fs_point`) and the buffer indices are
/// the contract.
pub const SHADER_SOURCE: &str = include_str!("shaders.metal");

#[cfg(test)]
mod tests {
    #[test]
    fn shader_source_declares_the_entry_points() {
        for entry in ["vs_mesh", "fs_mesh", "vs_point", "fs_point"] {
            assert!(
                super::SHADER_SOURCE.contains(entry),
                "shaders.metal is missing `{entry}`"
            );
        }
    }
}
