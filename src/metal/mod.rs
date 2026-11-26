//! A Metal renderer for Apple platforms, written against the Objective-C
//! runtime directly — no `metal-rs`, no `objc` crate, no build script.
//!
//! Behind the `metal` feature, and compiled only for Apple's Metal platforms:
//! macOS, iOS and iPadOS (Mac Catalyst included), tvOS and visionOS. The wgpu
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
//! The `visionos` feature builds on both: see the `visionos` module for the
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
//! Render on any thread. [`crate::metal::MetalDevice`] is `Send + Sync` — share
//! one between threads, and its command queue with it — and the renderers,
//! targets, surfaces and post chains are `Send`: build one where it is
//! convenient and move it to the thread that draws. Each one is used by one
//! thread at a time, which is what `&mut self` already says.
//!
//! Every call that makes Objective-C objects drains its own autorelease pool,
//! dropping a value included, so a plain `std::thread` with no pool of its own
//! does not leak.
//!
//! Two calls want the main thread, because they touch a view:
//! [`crate::metal::MetalSurface::from_ns_view`] and, on UIKit platforms,
//! `MetalSurface::from_ui_view`. Both say so with an error
//! rather than leaving it to AppKit or UIKit. Everything after — `resize`,
//! `next_frame`, `render` — runs on the render thread, the way `MTKView`-free
//! engines drive a `CAMetalLayer`.
//!
//! For a fixed set of render threads that are reused — sized for the
//! performance cores, and capped low on iOS, tvOS and visionOS where cores and
//! thermal headroom are scarce — use [`crate::metal::MetalRenderPool`]: each
//! worker keeps its renderer and its caches from job to job, and the bounded
//! queue in front of them makes a producer wait or drop a frame rather than
//! run ahead of the GPU.
//!
//! A command buffer the system refuses — iOS rejects GPU work from an app in
//! the background — comes back from a readback as
//! [`crate::metal::MetalError::CommandBuffer`] rather than as stale pixels.
//!
//! # Layout
//!
//! | Module | Role |
//! |--------|------|
//! | [`crate::metal::objc`] | `objc_msgSend`, selectors, retain/release, autorelease pools |
//! | [`crate::metal::enums`] | Metal enumerants and by-value structs |
//! | [`crate::metal::resources`] | Textures, samplers, pipeline states |
//! | — | [`crate::metal::MetalDevice`], [`crate::metal::MetalRenderer`], [`crate::metal::MetalRenderTarget`], [`crate::metal::MetalSurface`] |
//! | — | [`crate::metal::MetalRenderPool`]: reusable render threads |
//! | — | [`crate::metal::MetalUi`]: a threers-ui UI over a frame (`ui`) |

pub mod objc;

pub mod enums;
#[cfg(metal_backend)]
pub mod resources;

#[cfg(feature = "visionos")]
pub mod visionos;

// Built with `metal-fluid` alone as well as with `metal`: the device, the
// Objective-C glue, textures and fullscreen passes. Everything else here is
// the scene renderer, and needs `metal` (and so wgpu's half of the crate).
mod device;
mod pass;

#[cfg(metal_backend)]
mod composer;
#[cfg(metal_backend)]
mod headless;
#[cfg(metal_backend)]
mod overlay;
#[cfg(metal_backend)]
mod pool;
#[cfg(metal_backend)]
mod postfx;
#[cfg(metal_backend)]
mod renderer;
#[cfg(metal_backend)]
mod surface;
#[cfg(metal_backend)]
mod target;
#[cfg(all(metal_backend, feature = "ui"))]
mod quads;
#[cfg(all(metal_backend, feature = "ui"))]
pub(crate) mod warp;
#[cfg(all(metal_backend, feature = "ui"))]
mod ui;
#[cfg(all(metal_backend, feature = "ui"))]
mod ui_images;
#[cfg(metal_backend)]
mod video;
#[cfg(metal_backend)]
mod wgpu_bridge;
#[cfg(metal_backend)]
pub use wgpu_bridge::{wgpu_on, wgpu_texture};
#[cfg(metal_backend)]
pub mod viewport;

#[cfg(metal_backend)]
pub use composer::{MetalPostBackend, MetalPostFx};
pub use device::{MetalDevice, MetalError};
#[cfg(metal_backend)]
pub(crate) use device::compile_library;
#[cfg(metal_backend)]
pub use headless::{MetalHeadlessBuilder, MetalHeadlessConfig, MetalHeadlessRenderer};
#[cfg(metal_backend)]
pub use pool::{
    MetalJob, MetalJobError, MetalRenderPool, MetalRenderPoolConfig, MetalWorker, MetalWorkerQos,
};
#[cfg(metal_backend)]
pub use renderer::{
    MetalRenderStats, MetalRenderer, PassAttachments, RenderView, DEFAULT_CACHE_RETENTION,
    MAX_VIEWS,
};
#[cfg(metal_backend)]
pub use postfx::{MetalBloom, MetalDownsampler, MetalSsao, MetalUpscaler, PostFxChain, SsaoSettings};
#[cfg(metal_backend)]
pub use surface::{MetalSurface, SurfaceFrame};
#[cfg(metal_backend)]
pub use target::MetalRenderTarget;
#[cfg(all(metal_backend, feature = "ui"))]
pub use ui::MetalUi;

// Moving a renderer to another thread.
//
// SAFETY: what these types hold is `MTLDevice`, `MTLCommandQueue`,
// `MTLLibrary`, `MTLFunction`, `MTLBuffer`, `MTLTexture`, `MTLSamplerState`,
// `MTLRenderPipelineState`, `MTLDepthStencilState`, a committed
// `MTLCommandBuffer` waiting to be read back, and (the surface) a
// `CAMetalLayer` — every one of which Apple documents as usable from any
// thread. Encoders and uncommitted command buffers, which are not, never
// outlive the call that made them. The raw `Id`s kept in a field
// (`MetalRenderer::capture_bound`) point into objects the same value owns.
//
// `Send` and not `Sync`: a renderer's caches are `&mut` state, and two threads
// encoding into one target would race on the GPU however Rust saw it. The
// device is the exception, and is `Sync` too — see `device.rs`.
#[cfg(metal_backend)]
unsafe impl Send for MetalRenderer {}
#[cfg(metal_backend)]
unsafe impl Send for MetalRenderTarget {}
#[cfg(metal_backend)]
unsafe impl Send for MetalHeadlessRenderer {}
#[cfg(metal_backend)]
unsafe impl Send for MetalSurface {}
#[cfg(metal_backend)]
unsafe impl Send for MetalSsao {}
#[cfg(metal_backend)]
unsafe impl Send for MetalBloom {}
#[cfg(metal_backend)]
unsafe impl Send for MetalDownsampler {}
#[cfg(metal_backend)]
unsafe impl Send for MetalUpscaler {}
#[cfg(metal_backend)]
unsafe impl Send for PostFxChain {}
#[cfg(metal_backend)]
unsafe impl Send for postfx::PostFxPipelines {}
#[cfg(any(feature = "fluid", feature = "metal-fluid"))]
unsafe impl Send for MetalFluid {}
#[cfg(feature = "metal-fluid")]
unsafe impl Send for MetalFluid3d {}

#[cfg(metal_backend)]
/// The hand-written part of the default library: meshes, surfaces and the
/// post passes not generated. [`shader_source`] is the whole library,
/// compiled by [`MetalDevice::new`].
pub const SHADER_SOURCE: &str = concat!(
    include_str!("shaders.metal"),
    "\n",
    include_str!("fx_surface.metal"),
    "\n",
    include_str!("postfx.metal"),
);

/// The default library [`MetalDevice::new`] compiles: [`SHADER_SOURCE`] and
/// the post passes written once with [`passes!`](crate::passes) for both
/// backends — bloom's threshold and blur — and the shared functions in
/// place of their marker.
#[cfg(metal_backend)]
pub fn shader_source() -> &'static str {
    static SOURCE: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        let bloom = crate::renderer::bloom::BLOOM.msl("BloomParams", "bloom_").expect("bloom translates");
        use crate::renderer::shader::{SHARED_FUNCTIONS, SHARED_FUNCTIONS_MARKER};
        let shared = SHARED_FUNCTIONS.msl().expect("the shared functions translate");
        let source = SHADER_SOURCE.replacen(SHARED_FUNCTIONS_MARKER, &shared, 1);
        format!("{source}\n{bloom}")
    });
    &SOURCE
}

/// With `metal-fluid` alone there is no scene renderer, and the default
/// library is only what every Metal source starts with. The fluid's is
/// [`fluid_shader_source`].
#[cfg(not(metal_backend))]
pub fn shader_source() -> &'static str {
    "#include <metal_stdlib>\nusing namespace metal;\n"
}

/// The fluid's frame-rate read-out in MSL — the one part of the fluid's
/// shaders written by hand. The passes are generated from the `passes!`
/// source the wgpu module also runs; [`fluid_shader_source`] has the whole
/// thing.
#[cfg(any(feature = "fluid", feature = "metal-fluid"))]
pub const FLUID_SHADER_SOURCE: &str = include_str!("fluid.metal");

/// The default library plus the fluid's, for a device that wants both.
///
/// ```no_run
/// # #[cfg(all(feature = "metal", feature = "fluid"))]
/// # fn demo() -> Result<(), threers::metal::MetalError> {
/// use threers::metal::{fluid_shader_source, MetalDevice};
/// let device = MetalDevice::with_shader_source(&fluid_shader_source())?;
/// # Ok(()) }
/// ```
///
/// Kept out of the default library because most Metal work is a scene and a
/// post chain, and a library is compiled once per device: a caller who will
/// never run a fluid should not pay for its twenty fragment functions.
#[cfg(any(feature = "fluid", feature = "metal-fluid"))]
pub fn fluid_shader_source() -> String {
    let passes = crate::fluid::shader::FLUID
        .msl(crate::fluid::shader::UNIFORMS, "fluid_")
        .expect("the fluid's passes translate");
    format!("{}\n{passes}\n{FLUID_SHADER_SOURCE}", shader_source())
}

#[cfg(any(feature = "fluid", feature = "metal-fluid"))]
pub mod fluid;
#[cfg(any(feature = "fluid", feature = "metal-fluid"))]
pub use fluid::{FluidSplat, MetalFieldFormats, MetalFluid};
#[cfg(feature = "metal-fluid")]
pub mod fluid3d;
#[cfg(feature = "metal-fluid")]
pub use fluid3d::{fluid3d_shader_source, pixel_format_rgba8, MetalFluid3d};

#[cfg(all(test, metal_backend))]
mod tests {
    #[test]
    fn renderers_can_move_between_threads() {
        fn send<T: Send>() {}
        fn sync<T: Sync>() {}
        send::<super::MetalDevice>();
        sync::<super::MetalDevice>();
        send::<super::MetalRenderer>();
        send::<super::MetalRenderTarget>();
        send::<super::MetalHeadlessRenderer>();
        send::<super::MetalSurface>();
        send::<super::PostFxChain>();
        send::<super::MetalRenderPool>();
        sync::<super::MetalRenderPool>();
        #[cfg(feature = "fluid")]
        send::<super::MetalFluid>();
        // What a render thread is handed along with them.
        send::<crate::scene::Scene>();
        send::<crate::cameras::PerspectiveCamera>();
    }

    #[test]
    fn shader_source_declares_the_entry_points() {
        for entry in [
            "vs_mesh",
            "fs_mesh",
            "vs_line",
            "vs_point",
            "fs_point",
            "fs_point_encoded_blend",
            "vs_fx_surface",
            "fs_fx_surface",
            "bloom_fs_threshold",
            "bloom_fs_blur",
            "fs_downsample",
            "fs_upscale",
            "fs_bloom_composite",
            "fs_overlay",
            "fs_overlay_premultiplied",
            "fs_mesh_encoded_add",
            "fs_point_encoded_add",
        ] {
            assert!(
                super::shader_source().contains(entry),
                "shader source is missing `{entry}`"
            );
        }
    }
}
