//! A physically-based path tracer, beside the rasteriser rather than inside it.
//!
//! [`Renderer`](crate::renderer::Renderer) draws a scene the way a GPU
//! pipeline does: project the triangles, shade each fragment from a fixed set
//! of lights, and approximate everything else — shadows with a depth map,
//! reflections by marching the screen, ambient occlusion from the depth buffer.
//! Those approximations are what make it run at 60 fps and what make it wrong
//! in specific, familiar ways: light does not bounce, a mirror cannot show what
//! is behind the camera, and a glass ball does not bend the room.
//!
//! This module answers the same question by simulating light transport
//! directly. Rays leave the camera, scatter off surfaces according to their
//! BSDF, and either find a light or die trying; the image is the average of
//! millions of such paths. Global illumination, soft shadows, glossy
//! interreflection, refraction with dispersion-free Snell bending, and depth of
//! field all fall out of that rather than being features — they are what the
//! integral says. The cost is time: a frame is seconds to minutes, not
//! milliseconds.
//!
//! # Quick start
//!
//! ```no_run
//! use threers::raytrace::{RaytraceRenderer, RaytraceSettings};
//! # use threers::{Scene, PerspectiveCamera};
//! # let mut scene = Scene::new();
//! # let camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
//! let mut renderer = RaytraceRenderer::new(800, 600);
//! renderer.set_settings(RaytraceSettings::default().with_samples(128));
//! let rgba = renderer.render_to_rgba(&mut scene, &camera);
//! std::fs::write("out.png", threers::encode_png(800, 600, &rgba)).unwrap();
//! ```
//!
//! The scene is an ordinary [`Scene`](crate::Scene) — the same one the raster
//! renderer takes, with the same [`Material`](crate::materials::Material)s and
//! [`Light`](crate::lights::Light)s. Nothing needs to be authored twice.
//!
//! # Progressive rendering
//!
//! Prepare once, then add samples in batches and resolve whenever you want to
//! show progress:
//!
//! ```no_run
//! # use threers::raytrace::RaytraceRenderer;
//! # use threers::{Scene, PerspectiveCamera};
//! # let mut scene = Scene::new();
//! # let camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
//! let mut renderer = RaytraceRenderer::new(800, 600);
//! renderer.prepare(&mut scene, &camera);
//! for _ in 0..20 {
//!     renderer.accumulate(8).unwrap();
//!     let preview = renderer.resolve_rgba();  // refines each time
//!     # let _ = preview;
//! }
//! ```
//!
//! # Backends
//!
//! [`CpuBackend`] is the reference: portable, `wasm32` included, and the
//! definition of correct for everything here. [`crate::raytrace::gpu::GpuBackend`] runs the same
//! integrator as a wgpu compute kernel over the same
//! [`RaytraceScene`] — typically one to two orders of magnitude faster on a
//! discrete GPU. Both implement [`RaytraceBackend`], so switching is one line:
//!
//! ```text
//! use threers::raytrace::{gpu::GpuBackend, RaytraceRenderer};
//! let backend = GpuBackend::headless().expect("no GPU");
//! let mut renderer = RaytraceRenderer::with_backend(800, 600, Box::new(backend));
//! ```
//!
//! # What is and is not simulated
//!
//! Simulated: multi-bounce diffuse and glossy global illumination; the
//! principled BSDF (Lambert + anisotropic GGX + rough dielectric transmission +
//! clearcoat) with Kulla–Conty energy compensation; emissive geometry as area
//! lights; `Directional`/`Point`/`Spot`/`RectArea` lights, with optional
//! angular or spherical size for soft shadows; `Ambient` and `Hemisphere`
//! lights and `scene.environment` as image-based lighting; alpha cutouts with
//! correctly-shaped shadows; Beer–Lambert absorption inside transmissive
//! solids; a thin-lens camera.
//!
//! Not simulated: participating media (fog, smoke), subsurface scattering,
//! spectral dispersion, and caustics through NEE — light that reaches a diffuse
//! surface through glass arrives only along BSDF-sampled paths, so a caustic is
//! noisy and a bidirectional method would do better.
//!
//! Materials with no physical reading — `MeshNormalMaterial`,
//! `MeshToonMaterial`, `ShaderMaterial` and friends — are mapped to the nearest
//! surface that does have one, and each choice is recorded in
//! [`crate::raytrace::BuildReport::approximated`] rather than made silently.

mod backend;
mod bsdf;
mod bvh;
mod camera;
mod denoise;
/// The trained denoiser's forward pass, dependency-free and wasm-capable.
pub mod denoise_net;
mod distribution;
mod film;
mod integrator;
/// The trained denoiser. Needs `learned-denoise`.
#[cfg(all(feature = "learned-denoise", not(target_arch = "wasm32")))]
pub mod learned;
mod lights;
mod render;
mod sampler;
mod scene;
mod settings;
mod texture;

/// The wgpu compute backend — see [`crate::raytrace::gpu::GpuBackend`].
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu;

pub use backend::{probe_focus_distance, CpuBackend, RaytraceBackend, RaytraceError};
pub use bsdf::{Bsdf, BsdfSample, Surface};
pub use bvh::{RtBvh, RtHit};
pub use camera::RtCamera;
pub use denoise::{denoise, DenoiseExample, DenoiseGuides, DenoiseParams};
pub use denoise_net::{DenoiseError, Denoiser as NetDenoiser, Widths as NetWidths};
pub use distribution::{
    direction_from_uv, uv_from_direction, EnvDistribution, ENV_HEIGHT, ENV_WIDTH,
};
pub use film::{tone_map, Film, Pixel};
pub use integrator::{Integrator, PathResult, Scratch};
pub use lights::{
    distance_attenuation, emissive_pdf, sample_analytic, sample_emissive, sample_world, world_pdf,
    EmitterSample, LightSample,
};
pub use render::{ProgressiveFrame, ProgressiveOptions, RaytraceRenderer};
pub use sampler::{
    cosine_hemisphere, cosine_hemisphere_pdf, power_heuristic, uniform_cone, uniform_cone_pdf,
    uniform_sphere, Onb, Rng,
};
pub use scene::{
    BuildReport, EmissiveSet, EmissiveTri, RaytraceScene, RtHemisphere, RtLight, RtMaterial,
    TriShading, World,
};
pub use settings::{Aov, BackgroundMode, RaytraceSettings};
pub use texture::{CpuTexture, TextureCache};
