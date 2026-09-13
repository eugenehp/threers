//! The public front end: point it at a [`Scene`] and a [`Camera`] and get
//! pixels.
//!
//! Two ways to drive it, and they share one film:
//!
//! - **One shot.** [`RaytraceRenderer::render_to_rgba`] traces
//!   `samples_per_pixel` paths and returns tone-mapped RGBA8, the same layout
//!   [`HeadlessRenderer::render_to_rgba`](crate::HeadlessRenderer::render_to_rgba)
//!   produces — so a path-traced frame drops straight into anything built
//!   around the raster renderer, including the video exporters.
//! - **Progressive.** [`RaytraceRenderer::prepare`] once, then
//!   [`RaytraceRenderer::accumulate`] in batches, resolving whenever you want
//!   to show what has converged so far. This is the interactive path: the image
//!   refines instead of appearing all at once, and the scene is only flattened
//!   and its BVH built once.

use crate::cameras::Camera;
use crate::core::Layers;
use crate::scene::Scene;

use super::backend::{intersect_box, probe_focus_distance, CpuBackend, RaytraceBackend, RaytraceError, RenderRect};
use crate::math::{Box3, Vector3};
use super::camera::RtCamera;
use super::checkpoint::{encode_exr_rgba, FilmCheckpoint};
use super::denoise::{self, DenoiseGuides, DenoiseParams};
use super::fingerprint::SceneFingerprint;
use super::film::Film;
use super::scene::{BuildReport, RaytraceScene};
use super::settings::{Aov, RaytraceSettings};

/// How [`RaytraceRenderer::render_progressive`] paces its updates.
#[derive(Debug, Clone)]
pub struct ProgressiveOptions {
    /// Samples to trace between updates.
    pub batch: u32,
    /// Do not report or denoise until this many samples have landed.
    ///
    /// Denoising a single sample gives an image, but a smeared one, and for a
    /// viewport that first frame sets the impression. Cycles exposes the same
    /// control as `denoise_start_sample`.
    pub start_sample: u32,
    /// Minimum samples between reported frames.
    ///
    /// Once an image is close to converged each extra batch changes it less
    /// than the denoise costs to run, so the updates spread out. Cycles does
    /// this on a timer past 20 samples; counting samples instead keeps the
    /// behaviour the same whatever the machine's speed, which is what makes a
    /// test of it repeatable.
    pub min_step: u32,
    /// Whether to denoise each reported frame.
    pub denoise: bool,
    /// Stop early when this fraction of pixels have converged (0 = disabled).
    pub stop_when_converged: f32,
    /// Tile size for unconverged-tile scheduling when sample redistribution is on.
    /// `0` picks [`RaytraceRenderer::recommended_tile_size`].
    pub tile_size: u32,
    /// Cap tiles traced per batch (0 = all unconverged tiles).
    pub max_tiles_per_batch: u32,
    /// Soft time budget in milliseconds for each progressive batch (0 = unlimited).
    /// The scheduler learns tile cost from prior batches and takes fewer tiles
    /// when a frame would otherwise overrun.
    pub max_ms: u32,
    /// Write a film checkpoint every N accumulated samples (0 = off).
    pub autosave_every: u32,
    /// Destination for [`Self::autosave_every`] checkpoints.
    pub autosave_path: Option<String>,
}

impl Default for ProgressiveOptions {
    fn default() -> Self {
        Self {
            batch: 1,
            start_sample: 1,
            min_step: 1,
            denoise: true,
            stop_when_converged: 0.0,
            tile_size: 0,
            max_tiles_per_batch: 0,
            max_ms: 0,
            autosave_every: 0,
            autosave_path: None,
        }
    }
}

impl ProgressiveOptions {
    pub fn with_batch(mut self, batch: u32) -> Self {
        self.batch = batch.max(1);
        self
    }

    pub fn with_start_sample(mut self, n: u32) -> Self {
        self.start_sample = n.max(1);
        self
    }

    pub fn with_min_step(mut self, n: u32) -> Self {
        self.min_step = n.max(1);
        self
    }

    pub fn with_denoise(mut self, on: bool) -> Self {
        self.denoise = on;
        self
    }

    pub fn with_stop_when_converged(mut self, fraction: f32) -> Self {
        self.stop_when_converged = fraction.clamp(0.0, 1.0);
        self
    }

    pub fn with_tile_size(mut self, tile_size: u32) -> Self {
        self.tile_size = tile_size;
        self
    }

    pub fn with_max_tiles_per_batch(mut self, max: u32) -> Self {
        self.max_tiles_per_batch = max;
        self
    }

    pub fn with_max_ms(mut self, max_ms: u32) -> Self {
        self.max_ms = max_ms;
        self
    }

    pub fn with_autosave(mut self, every: u32, path: impl Into<String>) -> Self {
        self.autosave_every = every;
        self.autosave_path = if every > 0 {
            Some(path.into())
        } else {
            None
        };
        self
    }
}

/// One update from [`RaytraceRenderer::render_progressive`].
pub struct ProgressiveFrame<'a> {
    /// Samples per pixel accumulated so far.
    pub samples: u32,
    /// Samples this render is working toward.
    pub total: u32,
    /// Whether `image` went through the denoiser.
    pub denoised: bool,
    /// Tone-mapped sRGB RGBA8, as [`RaytraceRenderer::resolve_rgba`] returns.
    pub image: &'a [u8],
    /// Fraction of the adaptive sample budget actually traced.
    pub efficiency: f32,
    /// Fewest and most samples any pixel took.
    pub sample_range: (u32, u32),
    /// Share of pixels that met the adaptive convergence threshold.
    pub converged_fraction: f32,
    /// Unconverged tiles remaining at `tile_size` (0 when adaptive is off).
    pub unconverged_tile_count: u32,
    /// One-line scene build summary, when the renderer is prepared.
    pub report_summary: Option<String>,
}

impl ProgressiveFrame<'_> {
    /// Whether this is the last update of the render.
    pub fn is_last(&self) -> bool {
        self.samples >= self.total || self.converged_fraction >= 1.0
    }
}

/// Adaptive sampling and build stats for a progressive update.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderStats {
    pub efficiency: f32,
    pub sample_range: (u32, u32),
    pub converged_fraction: f32,
    pub unconverged_tile_count: u32,
    pub report_summary: Option<String>,
}

/// A path-tracing renderer.
pub struct RaytraceRenderer {
    settings: RaytraceSettings,
    denoise_params: DenoiseParams,
    backend: Box<dyn RaytraceBackend>,
    film: Film,
    /// The flattened scene, kept between calls so a progressive render does not
    /// rebuild the BVH every batch.
    scene: Option<RaytraceScene>,
    camera: Option<RtCamera>,
    /// Previous camera pose for motion blur when only the view changed.
    prev_camera: Option<RtCamera>,
    scene_fingerprint: Option<SceneFingerprint>,
    /// Rolling estimate of milliseconds per unconverged tile (interactive pacing).
    last_tile_ms: f32,
    /// A trained denoiser, used in place of the À-Trous filter when set.
    ///
    /// Behind a `RefCell` because running a compiled graph binds its parameters
    /// and so needs `&mut`, while resolving a film is otherwise a read.
    #[cfg(all(feature = "learned-denoise", not(target_arch = "wasm32")))]
    learned: Option<std::cell::RefCell<super::learned::LearnedDenoiser>>,
}

impl std::fmt::Debug for RaytraceRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RaytraceRenderer")
            .field("backend", &self.backend.name())
            .field("size", &(self.film.width(), self.film.height()))
            .field("samples", &self.film.samples())
            .field("prepared", &self.scene.is_some())
            .finish()
    }
}

impl RaytraceRenderer {
    /// A renderer of `width × height`, on the CPU backend.
    pub fn new(width: u32, height: u32) -> Self {
        Self::with_backend(width, height, Box::new(CpuBackend::new()))
    }

    /// A renderer on a backend of your choosing — see
    /// [`RaytraceBackend`], and [`GpuBackend`](super::gpu::GpuBackend) for the
    /// compute-shader one.
    pub fn with_backend(width: u32, height: u32, backend: Box<dyn RaytraceBackend>) -> Self {
        let mut r = Self {
            settings: RaytraceSettings::default(),
            denoise_params: DenoiseParams::default(),
            backend,
            film: Film::new(1, 1),
            scene: None,
            camera: None,
            prev_camera: None,
            scene_fingerprint: None,
            last_tile_ms: 0.0,
            #[cfg(all(feature = "learned-denoise", not(target_arch = "wasm32")))]
            learned: None,
        };
        r.set_size(width, height);
        r
    }

    /// Name of the backend in use.
    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Intersect a world-space AABB along a ray.
    ///
    /// Same as [`Raycaster::intersect_box`] — exposed on the renderer so path
    /// tracing and raster picking share one intersection implementation.
    pub fn intersect_box(
        origin: Vector3,
        direction: Vector3,
        near: f32,
        far: f32,
        bounds: &Box3,
    ) -> Option<f32> {
        intersect_box(origin, direction, near, far, bounds)
    }

    /// Device limits when the GPU backend is active.
    pub fn gpu_caps(&self) -> Option<super::gpu::GpuCaps> {
        self.backend.gpu_caps()
    }

    /// Maximum film pixels on the GPU backend, if known.
    pub fn max_film_pixels(&self) -> Option<u64> {
        self.gpu_caps().map(|c| c.max_accum_pixels)
    }

    /// Maximum film side on the GPU backend (~√`max_film_pixels`), if known.
    pub fn max_film_side(&self) -> Option<u32> {
        self.gpu_caps().map(|c| c.max_film_side())
    }

    /// Check whether `width × height` fits the GPU accum buffer.
    pub fn check_film_size(&self, width: u32, height: u32) -> Result<(), RaytraceError> {
        if let Some(caps) = self.gpu_caps() {
            caps.check_film(width.max(1), height.max(1))?;
        }
        Ok(())
    }

    /// Clamp a requested size to GPU limits. No-op on CPU backends.
    pub fn clamp_film_size(&self, width: u32, height: u32) -> (u32, u32) {
        self.gpu_caps()
            .map(|c| c.clamp_film_size(width, height))
            .unwrap_or((width.max(1), height.max(1)))
    }

    pub fn settings(&self) -> &RaytraceSettings {
        &self.settings
    }

    /// Replace the settings and discard anything accumulated — most settings
    /// change what a sample *means*, so mixing old and new ones would produce
    /// an image that is neither.
    pub fn set_settings(&mut self, settings: RaytraceSettings) {
        self.settings = settings;
        self.film.clear();
        // Materials and lights are baked into the flattened scene using the
        // settings, so it has to go too.
        self.scene = None;
        self.scene_fingerprint = None;
        self.prev_camera = None;
        self.backend.invalidate();
    }

    /// Pick a fresh random sample seed and start the film over.
    ///
    /// Same as [`Self::set_settings`] with [`RaytraceSettings::random_seed`],
    /// but keeps every other setting as-is.
    pub fn reseed_random(&mut self) {
        let mut settings = self.settings.clone();
        settings.seed = RaytraceSettings::random_seed();
        self.set_settings(settings);
    }

    /// Tune the À-Trous denoiser.
    pub fn set_denoise_params(&mut self, params: DenoiseParams) {
        self.denoise_params = params;
    }

    /// Enable or disable the post-process denoiser without clearing the film.
    /// Denoising does not change what samples mean — only how they are shown.
    pub fn set_denoise(&mut self, on: bool) {
        self.settings.denoise = on;
    }

    /// Change exposure without clearing the film (tone mapping is resolve-time).
    pub fn set_exposure(&mut self, exposure: f32) {
        self.settings.exposure = exposure.max(0.0);
    }

    /// Select which channel [`Self::resolve_rgba`] reports, without clearing the film.
    /// AOVs are derived from auxiliaries already stored per sample.
    pub fn set_aov(&mut self, aov: Aov) {
        self.settings.aov = aov;
    }

    /// Adaptive threshold for future samples / convergence tests. Does not clear
    /// the film — already-traced pixels keep their accumulators.
    pub fn set_adaptive(&mut self, threshold: f32) {
        self.settings.adaptive_threshold = threshold.max(0.0);
    }

    /// Steer leftover sample budget toward noisy pixels (tile scheduling).
    pub fn set_sample_redistribution(&mut self, on: bool) {
        self.settings.sample_redistribution = on;
    }

    /// Minimum samples before a pixel may be declared converged.
    pub fn set_adaptive_min_samples(&mut self, n: u32) {
        self.settings.adaptive_min_samples = n.max(1);
    }

    /// Raise or lower the sample budget without clearing the film.
    /// Lowering below the current count simply stops further accumulation.
    pub fn set_sample_budget(&mut self, spp: u32) {
        self.settings.samples_per_pixel = spp.max(1);
    }

    /// Indirect firefly clamp for future paths (`0` = off). Does not clear the film.
    pub fn set_clamp_indirect(&mut self, clamp: f32) {
        self.settings.clamp_indirect = clamp.max(0.0);
    }

    /// Direct-light clamp for future paths (`0` = off). Does not clear the film.
    pub fn set_clamp_direct(&mut self, clamp: f32) {
        self.settings.clamp_direct = clamp.max(0.0);
    }

    /// Biased glass-shadow caustic hops for future paths.
    pub fn set_caustic_glass_shadows(&mut self, on: bool) {
        self.settings.caustic_glass_shadows = on;
    }

    /// GPU only: samples per compute dispatch. Returns `false` on CPU backends.
    pub fn set_samples_per_dispatch(&mut self, n: u32) -> bool {
        if let Some(gpu) = self
            .backend
            .as_any_mut()
            .downcast_mut::<super::gpu::GpuBackend>()
        {
            gpu.set_samples_per_dispatch(n);
            true
        } else {
            false
        }
    }

    /// GPU only: defer readback after tile/region traces until resolve or
    /// [`Self::flush_gpu_film`]. Full-frame traces always sync immediately.
    pub fn set_defer_gpu_readback(&mut self, on: bool) -> bool {
        if let Some(gpu) = self
            .backend
            .as_any_mut()
            .downcast_mut::<super::gpu::GpuBackend>()
        {
            gpu.set_defer_readback(on);
            true
        } else {
            false
        }
    }

    /// Pull a deferred GPU film onto the host. No-op on CPU backends.
    pub fn flush_gpu_film(&mut self) -> Result<(), RaytraceError> {
        self.sync_gpu_film()
    }

    fn sync_gpu_film(&mut self) -> Result<(), RaytraceError> {
        if let Some(gpu) = self
            .backend
            .as_any_mut()
            .downcast_mut::<super::gpu::GpuBackend>()
        {
            gpu.sync_film(&mut self.film)?;
        }
        Ok(())
    }

    /// Awaited twin of [`Self::sync_gpu_film`].
    #[cfg(target_arch = "wasm32")]
    async fn sync_gpu_film_async(&mut self) -> Result<(), RaytraceError> {
        if let Some(gpu) = self
            .backend
            .as_any_mut()
            .downcast_mut::<super::gpu::GpuBackend>()
        {
            gpu.sync_film_async(&mut self.film).await?;
        }
        Ok(())
    }

    /// Resolve to RGBA8 without blocking on the film readback.
    ///
    /// The wasm counterpart of [`Self::resolve_rgba_denoised`]. Tracing itself
    /// never blocks — `GpuBackend::render` leaves the accumulator on the device
    /// and only flags a pending readback — so this is the single call that had
    /// to become awaited for the compute backend to work in a browser at all.
    #[cfg(target_arch = "wasm32")]
    pub async fn resolve_rgba_denoised_async(&mut self, denoise: bool) -> Vec<u8> {
        let _ = self.sync_gpu_film_async().await;
        if self.settings.aov != Aov::Beauty {
            return self.film.resolve_rgba8(&self.settings);
        }
        // Past the readback the work is all on the host, so it can reuse the
        // existing synchronous resolve rather than duplicating the pipeline.
        let hdr = self.resolve_hdr_denoised(denoise);
        self.film.beauty_rgba8(&hdr, &self.settings)
    }

    /// Re-seed the device accum buffer on the next trace without repacking BVH
    /// data. Used when only the host film was cleared or replaced.
    fn reset_gpu_accum(&mut self) {
        if let Some(gpu) = self
            .backend
            .as_any_mut()
            .downcast_mut::<super::gpu::GpuBackend>()
        {
            gpu.reset_accum();
        }
    }

    /// Resolve a specific AOV without mutating settings.
    pub fn resolve_rgba_aov(&mut self, aov: Aov) -> Vec<u8> {
        let _ = self.sync_gpu_film();
        if aov == Aov::Beauty {
            return self.resolve_rgba();
        }
        let mut settings = self.settings.clone();
        settings.aov = aov;
        self.film.resolve_rgba8(&settings)
    }

    /// Grayscale map of samples taken per pixel (bright = more samples).
    pub fn resolve_rgba_samples(&mut self) -> Vec<u8> {
        let _ = self.sync_gpu_film();
        let counts = self.film.resolve_sample_counts();
        let max = counts.iter().copied().max().unwrap_or(1).max(1);
        let mut out = Vec::with_capacity(counts.len() * 4);
        for n in counts {
            let t = n as f32 / max as f32;
            let b = (t * 255.0 + 0.5) as u8;
            out.extend_from_slice(&[b, b, b, 255]);
        }
        out
    }

    /// Resolve beauty RGBA8, optionally tinting noisy pixels red for debugging.
    ///
    /// `strength` is how strongly the relative-error heatmap mixes in (`0` =
    /// beauty only). Infinite/unknown error lights up at full strength.
    pub fn resolve_rgba_error_overlay(&mut self, strength: f32) -> Vec<u8> {
        let mut rgba = self.resolve_rgba();
        let strength = strength.clamp(0.0, 1.0);
        if strength <= 0.0 {
            return rgba;
        }
        let err = self.film.resolve_error();
        for (px, e) in rgba.chunks_exact_mut(4).zip(err.iter()) {
            let t = if !e.is_finite() {
                1.0
            } else {
                (e / (e + 0.05)).clamp(0.0, 1.0)
            };
            let mix = t * strength;
            px[0] = ((1.0 - mix) * px[0] as f32 + mix * 255.0) as u8;
            px[1] = ((1.0 - mix) * px[1] as f32) as u8;
            px[2] = ((1.0 - mix) * px[2] as f32) as u8;
        }
        rgba
    }

    /// Denoise with a trained network instead of the À-Trous filter.
    ///
    /// Takes effect wherever `settings.denoise` already applies, so nothing
    /// else about the call sequence changes. `None` restores the filter.
    #[cfg(all(feature = "learned-denoise", not(target_arch = "wasm32")))]
    pub fn set_learned_denoiser(&mut self, denoiser: Option<super::learned::LearnedDenoiser>) {
        self.learned = denoiser.map(std::cell::RefCell::new);
    }

    /// Whether a trained denoiser is in use.
    #[cfg(all(feature = "learned-denoise", not(target_arch = "wasm32")))]
    pub fn has_learned_denoiser(&self) -> bool {
        self.learned.is_some()
    }

    pub fn size(&self) -> (u32, u32) {
        (self.film.width(), self.film.height())
    }

    /// Resize, discarding the film. On the GPU backend, sizes above
    /// [`Self::max_film_side`] are clamped to fit the device accum buffer.
    /// Returns the film size actually used.
    pub fn set_size(&mut self, width: u32, height: u32) -> (u32, u32) {
        let (width, height) = self.clamp_film_size(width, height);
        if (width, height) != self.size() {
            self.film = Film::new(width, height);
            // Deliberately NOT `backend.invalidate()`. That means "the scene
            // changed" and makes the GPU backend drop its packed geometry and
            // BVH — the most expensive thing it builds — when all that moved is
            // the film. `ensure_resources` compares the film size itself and
            // rebuilds only the accumulator, so the resolution can change
            // without repacking. `invalidate` is a no-op on the CPU backend.
        } else {
            self.film.clear();
            self.reset_gpu_accum();
        }
        (width, height)
    }

    /// Whether the GPU backend still has the flattened scene on the device.
    pub fn gpu_scene_cached(&mut self) -> bool {
        self.backend
            .as_any_mut()
            .downcast_mut::<super::gpu::GpuBackend>()
            .is_some_and(|gpu| gpu.scene_on_device())
    }

    /// Drop all backend caches, including GPU scene geometry. Camera-only updates
    /// should prefer [`Self::prepare_if_changed`], which keeps the BVH resident.
    /// How many times the GPU backend has packed a scene, if it is one.
    pub fn gpu_packs(&mut self) -> Option<usize> {
        self.backend
            .as_any_mut()
            .downcast_mut::<super::gpu::GpuBackend>()
            .map(|g| g.packs())
    }

    pub fn backend_invalidate(&mut self) {
        self.backend.invalidate();
    }

    /// Samples *issued* per pixel so far.
    pub fn samples(&self) -> u32 {
        self.film.samples()
    }

    /// Fewest and most samples any pixel actually took. Equal when adaptive
    /// sampling is off; a wide range means the budget went where the noise was.
    pub fn sample_range(&self) -> (u32, u32) {
        self.film.sample_range()
    }

    /// Total samples traced, against what a non-adaptive render of the same
    /// budget would have cost. `1.0` means nothing was saved.
    pub fn sample_efficiency(&self) -> f32 {
        self.render_stats().efficiency
    }

    /// Adaptive sampling and build stats for the current film.
    pub fn render_stats(&self) -> RenderStats {
        self.render_stats_for_tile_size(RenderRect::recommended_tile_size(
            self.film.width(),
            self.film.height(),
        ))
    }

    /// [`Self::render_stats`] with an explicit tile size for unconverged counts.
    pub fn render_stats_for_tile_size(&self, tile_size: u32) -> RenderStats {
        let efficiency = {
            let counts = self.film.resolve_sample_counts();
            if counts.is_empty() || self.film.samples() == 0 {
                1.0
            } else {
                let taken: u64 = counts.iter().map(|&n| n as u64).sum();
                let budget = counts.len() as u64 * self.film.samples() as u64;
                taken as f32 / budget.max(1) as f32
            }
        };
        let sample_range = self.film.sample_range();
        let converged_fraction = self
            .film
            .converged_fraction(self.settings.adaptive_threshold, self.settings.adaptive_min_samples);
        RenderStats {
            efficiency,
            sample_range,
            converged_fraction,
            unconverged_tile_count: self.unconverged_tile_count(tile_size),
            report_summary: self.report().map(|r| r.summary()),
        }
    }

    /// Tile size that balances viewport responsiveness and GPU dispatch overhead.
    pub fn recommended_tile_size(&self) -> u32 {
        let (w, h) = self.size();
        RenderRect::recommended_tile_size(w, h)
    }

    pub fn film(&self) -> &Film {
        &self.film
    }

    /// What the scene conversion did, once [`Self::prepare`] has run.
    pub fn report(&self) -> Option<&BuildReport> {
        self.scene.as_ref().map(|s| &s.report)
    }

    /// The flattened scene, for callers that want to inspect or reuse it.
    pub fn traced_scene(&self) -> Option<&RaytraceScene> {
        self.scene.as_ref()
    }

    /// Flatten `scene` for `camera` and reset the film. Call this whenever the
    /// scene, its transforms, or the camera change; [`Self::accumulate`] then
    /// adds samples to the result.
    pub fn prepare(&mut self, scene: &mut Scene, camera: &dyn Camera) {
        let layers = camera.layers();
        self.prepare_for_layers(scene, camera, layers);
    }

    /// As [`Self::prepare`], with an explicit layer mask.
    pub fn prepare_for_layers(&mut self, scene: &mut Scene, camera: &dyn Camera, layers: Layers) {
        let fp = SceneFingerprint::of(scene, &self.settings);
        self.rebuild_scene(scene, camera, layers);
        self.scene_fingerprint = Some(fp);
    }

    /// Rebuild only when the scene fingerprint changed; always refresh the camera.
    ///
    /// Returns `true` when the acceleration structure was rebuilt. When `false`,
    /// only the camera moved and the existing BVH is reused — the film is cleared.
    pub fn prepare_if_changed(&mut self, scene: &mut Scene, camera: &dyn Camera) -> bool {
        let fp = SceneFingerprint::of(scene, &self.settings);
        let layers = camera.layers();
        if self.scene_fingerprint == Some(fp) && self.scene.is_some() {
            self.update_camera(scene, camera);
            return false;
        }
        self.rebuild_scene(scene, camera, layers);
        self.scene_fingerprint = Some(fp);
        true
    }

    fn rebuild_scene(&mut self, scene: &mut Scene, camera: &dyn Camera, layers: Layers) {
        let flat = RaytraceScene::build_for_layers(scene, &self.settings, layers);
        self.camera = Some(self.build_camera(camera, &flat));
        self.scene = Some(flat);
        self.film.clear();
        self.backend.invalidate();
        self.last_tile_ms = 0.0;
    }

    fn update_camera(&mut self, scene: &mut Scene, camera: &dyn Camera) {
        let _ = scene;
        if let Some(cur) = self.camera.take() {
            self.prev_camera = Some(cur);
        }
        let flat = self.scene.as_ref().expect("prepared");
        self.camera = Some(self.build_camera(camera, flat));
        self.film.clear();
        self.reset_gpu_accum();
        self.last_tile_ms = 0.0;
    }

    fn build_camera(&self, camera: &dyn Camera, flat: &RaytraceScene) -> RtCamera {
        let mut rt_camera = RtCamera::new(camera, &self.settings);
        if self.settings.aperture > 0.0 && self.settings.focus_distance <= 0.0 {
            let d = probe_focus_distance(flat, &rt_camera).unwrap_or_else(|| flat.scale());
            rt_camera = rt_camera.with_focus_distance(d);
        }
        if self.settings.motion_blur_shutter > 0.0 {
            if let Some(prev) = self.prev_camera.as_ref() {
                rt_camera = rt_camera.with_motion_previous(prev.clone(), self.settings.motion_blur_shutter);
            }
        }
        rt_camera
    }

    /// Add `samples` samples per pixel to the film. Requires a previous
    /// [`Self::prepare`].
    pub fn accumulate(&mut self, samples: u32) -> Result<(), RaytraceError> {
        let (Some(scene), Some(camera)) = (self.scene.as_ref(), self.camera.as_ref()) else {
            return Err(RaytraceError::NotPrepared);
        };
        let first = self.film.samples();
        self.backend.render(
            scene,
            camera,
            &self.settings,
            &mut self.film,
            first,
            samples,
        )
    }

    /// Add samples only inside a screen rectangle. Supported on CPU and GPU
    /// backends; others fall back to a full-frame trace.
    pub fn accumulate_region(
        &mut self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        samples: u32,
    ) -> Result<(), RaytraceError> {
        let (Some(scene), Some(camera)) = (self.scene.as_ref(), self.camera.as_ref()) else {
            return Ok(());
        };
        let rect = RenderRect {
            x,
            y,
            width,
            height,
        };
        let first = self.film.samples();
        if let Some(gpu) = self
            .backend
            .as_any_mut()
            .downcast_mut::<super::gpu::GpuBackend>()
        {
            gpu.render_rect(
                scene,
                camera,
                &self.settings,
                &mut self.film,
                first,
                samples,
                rect,
            )
        } else if let Some(cpu) = self
            .backend
            .as_any_mut()
            .downcast_mut::<CpuBackend>()
        {
            cpu.render_rect(
                scene,
                camera,
                &self.settings,
                &mut self.film,
                first,
                samples,
                rect,
            )
        } else {
            self.backend.render(
                scene,
                camera,
                &self.settings,
                &mut self.film,
                first,
                samples,
            )
        }
    }

    /// Trace `samples` per pixel across a grid of tiles. Useful for viewport
    /// schedulers that want to spread work in spatial chunks.
    pub fn accumulate_tiles(
        &mut self,
        tile_size: u32,
        samples: u32,
    ) -> Result<(), RaytraceError> {
        let (w, h) = self.size();
        for rect in RenderRect::tile_grid(w, h, tile_size) {
            self.accumulate_region(rect.x, rect.y, rect.width, rect.height, samples)?;
        }
        Ok(())
    }

    /// Like [`Self::accumulate_tiles`], but skip tiles whose pixels have all
    /// converged. This is the GPU-friendly form of sample redistribution:
    /// work goes only where noise remains.
    pub fn accumulate_unconverged_tiles(
        &mut self,
        tile_size: u32,
        samples: u32,
    ) -> Result<(), RaytraceError> {
        self.accumulate_unconverged_tiles_limited(tile_size, samples, 0)
    }

    fn tile_worklist(&self, tile_size: u32, max_tiles: u32) -> Vec<RenderRect> {
        let tile_size = if tile_size == 0 {
            self.recommended_tile_size()
        } else {
            tile_size
        };
        let threshold = self.settings.adaptive_threshold;
        let min = self.settings.adaptive_min_samples;
        // Skip converged tiles whenever adaptive sampling is on — redistribution
        // only changes how leftover budget is spent *inside* a tile.
        let mut tiles: Vec<_> = if threshold > 0.0 {
            RenderRect::unconverged_tiles(&self.film, threshold, min, tile_size).collect()
        } else {
            RenderRect::tile_grid(self.film.width(), self.film.height(), tile_size).collect()
        };
        if threshold > 0.0 {
            tiles.sort_by(|a, b| {
                b.max_relative_error(&self.film)
                    .partial_cmp(&a.max_relative_error(&self.film))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        if max_tiles > 0 {
            tiles.truncate(max_tiles as usize);
        }
        tiles
    }

    fn accumulate_tile_regions(
        &mut self,
        tiles: &[RenderRect],
        samples: u32,
    ) -> Result<(), RaytraceError> {
        if tiles.is_empty() || samples == 0 {
            return Ok(());
        }
        let (Some(scene), Some(camera)) = (self.scene.as_ref(), self.camera.as_ref()) else {
            return Ok(());
        };
        let first = self.film.samples();
        if tiles.len() == 1 {
            let r = tiles[0];
            return self.accumulate_region(r.x, r.y, r.width, r.height, samples);
        }
        if let Some(gpu) = self
            .backend
            .as_any_mut()
            .downcast_mut::<super::gpu::GpuBackend>()
        {
            gpu.render_regions(
                scene,
                camera,
                &self.settings,
                &mut self.film,
                first,
                samples,
                tiles,
            )
        } else if let Some(cpu) = self
            .backend
            .as_any_mut()
            .downcast_mut::<CpuBackend>()
        {
            cpu.render_regions(
                scene,
                camera,
                &self.settings,
                &mut self.film,
                first,
                samples,
                tiles,
            )
        } else {
            self.backend.render(
                scene,
                camera,
                &self.settings,
                &mut self.film,
                first,
                samples,
            )
        }
    }

    /// Like [`Self::accumulate_tiles`], but skip converged tiles. Noisiest tiles
    /// first; `max_tiles` caps work per call (`0` = no cap).
    pub fn accumulate_unconverged_tiles_limited(
        &mut self,
        tile_size: u32,
        samples: u32,
        max_tiles: u32,
    ) -> Result<(), RaytraceError> {
        self.accumulate_unconverged_tiles_budgeted(tile_size, samples, max_tiles, 0)
            .map(|_| ())
    }

    /// Trace unconverged tiles under a soft time budget.
    ///
    /// Returns how many tiles were traced. When `max_ms > 0`, the number of
    /// tiles is chosen from a rolling cost estimate so interactive viewports
    /// stay responsive; the whole batch is still one GPU/CPU submission so the
    /// film sample counter advances only once.
    pub fn accumulate_unconverged_tiles_budgeted(
        &mut self,
        tile_size: u32,
        samples: u32,
        max_tiles: u32,
        max_ms: u32,
    ) -> Result<u32, RaytraceError> {
        if samples == 0 {
            return Ok(0);
        }
        let mut cap = max_tiles;
        if max_ms > 0 && self.last_tile_ms > 0.0 {
            let fits = (max_ms as f32 / self.last_tile_ms).floor() as u32;
            let fits = fits.max(1);
            cap = if cap == 0 { fits } else { cap.min(fits) };
        } else if max_ms > 0 && cap == 0 {
            // First frame with a budget: start small until we have a measurement.
            cap = 4;
        }
        let tiles = self.tile_worklist(tile_size, cap);
        if tiles.is_empty() {
            return Ok(0);
        }
        let n = tiles.len() as u32;
        let t0 = std::time::Instant::now();
        self.accumulate_tile_regions(&tiles, samples)?;
        let elapsed_ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.last_tile_ms = elapsed_ms / n.max(1) as f32;
        Ok(n)
    }

    /// One interactive step: refresh the camera when the scene is unchanged,
    /// then trace unconverged tiles under optional tile and time caps.
    pub fn accumulate_interactive(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
        tile_size: u32,
        samples: u32,
        max_tiles: u32,
        max_ms: u32,
    ) -> Result<u32, RaytraceError> {
        self.prepare_if_changed(scene, camera);
        self.accumulate_unconverged_tiles_budgeted(tile_size, samples, max_tiles, max_ms)
    }

    /// Rolling estimate of milliseconds spent per tile (0 until the first batch).
    pub fn last_tile_ms(&self) -> f32 {
        self.last_tile_ms
    }

    /// Whether tracing should continue: sample budget remains and, when adaptive
    /// sampling is on, some pixels are still noisy.
    pub fn needs_more_samples(&self) -> bool {
        let total = self.settings.samples_per_pixel;
        if self.film.samples() >= total {
            return false;
        }
        let threshold = self.settings.adaptive_threshold;
        if threshold > 0.0
            && self.film.converged_fraction(threshold, self.settings.adaptive_min_samples) >= 1.0
        {
            return false;
        }
        true
    }

    /// True when every pixel met the adaptive threshold (or adaptive is off and
    /// the sample budget is spent).
    pub fn is_converged(&self) -> bool {
        let threshold = self.settings.adaptive_threshold;
        if threshold <= 0.0 {
            return self.film.samples() >= self.settings.samples_per_pixel;
        }
        self.film.converged_fraction(threshold, self.settings.adaptive_min_samples) >= 1.0
    }

    /// Prepare and trace `samples_per_pixel` in one call.
    pub fn render(&mut self, scene: &mut Scene, camera: &dyn Camera) -> Result<(), RaytraceError> {
        self.prepare(scene, camera);
        self.accumulate(self.settings.samples_per_pixel)
    }

    /// Trace `samples_per_pixel` in batches, denoising the accumulated film
    /// after each one and handing the result to `on_frame`.
    ///
    /// The denoiser is a post-process over whatever the film holds, so it can
    /// run at any point rather than only at the end — a viewer gets a usable
    /// image after the first batch and watches it sharpen, instead of watching
    /// noise for the whole render and getting one clean frame at the end.
    ///
    /// `on_frame` receives the samples accumulated so far and the resolved
    /// image, and returns whether to keep going, so a viewport can stop on
    /// camera movement without waiting for the sample budget.
    ///
    /// This mirrors what Cycles does in the viewport
    /// (`integrator/render_scheduler.cpp`): denoise once the start sample is
    /// reached, then again on each update, and throttle once the image is
    /// converged enough that denoising would cost more than it shows. The
    /// difference from a final render is only *when* the filter runs — it is
    /// the same single forward pass over the same buffer, not a refinement
    /// loop, so a progressive render costs one extra denoise per batch and
    /// nothing else.
    pub fn render_progressive(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
        opts: &ProgressiveOptions,
        mut on_frame: impl FnMut(ProgressiveFrame<'_>) -> bool,
    ) -> Result<(), RaytraceError> {
        self.prepare(scene, camera);
        let total = self.settings.samples_per_pixel;
        let batch = opts.batch.max(1);
        let use_tiles = self.settings.adaptive_threshold > 0.0;
        let mut shown = 0u32;
        let mut last_saved = 0u32;
        while self.needs_more_samples() {
            let remaining = total.saturating_sub(self.film.samples()).max(1);
            let want = batch.min(remaining);
            let traced = if use_tiles {
                let tile = if opts.tile_size == 0 {
                    self.recommended_tile_size()
                } else {
                    opts.tile_size
                };
                self.accumulate_unconverged_tiles_budgeted(
                    tile,
                    want,
                    opts.max_tiles_per_batch,
                    opts.max_ms,
                )?
            } else {
                self.accumulate(want)?;
                want
            };
            let samples = self.film.samples();
            if opts.autosave_every > 0 {
                if let Some(path) = opts.autosave_path.as_ref() {
                    if samples >= last_saved + opts.autosave_every {
                        if let Err(e) = self.save_film(path) {
                            eprintln!("raytrace: autosave failed ({e})");
                        }
                        last_saved = samples;
                    }
                }
            }
            // Empty tile worklist or spent budget both count as finished.
            let last = !self.needs_more_samples() || (use_tiles && traced == 0);
            // Below the start sample there is nothing worth showing; past it,
            // every batch updates until the throttle takes over. The last batch
            // always reports, so the caller never misses the finished image.
            if !last && samples < opts.start_sample {
                continue;
            }
            if !last && shown > 0 && samples < shown + opts.min_step.max(1) {
                continue;
            }
            shown = samples;
            let denoise = opts.denoise && samples >= opts.start_sample;
            let image = self.resolve_rgba_denoised(denoise);
            let tile = if opts.tile_size == 0 {
                self.recommended_tile_size()
            } else {
                opts.tile_size
            };
            let stats = self.render_stats_for_tile_size(tile);
            if !on_frame(ProgressiveFrame {
                samples,
                total,
                denoised: denoise,
                image: &image,
                efficiency: stats.efficiency,
                sample_range: stats.sample_range,
                converged_fraction: stats.converged_fraction,
                unconverged_tile_count: stats.unconverged_tile_count,
                report_summary: stats.report_summary.clone(),
            }) {
                break;
            }
            if opts.stop_when_converged > 0.0
                && stats.converged_fraction >= opts.stop_when_converged
            {
                break;
            }
            if last {
                break;
            }
        }
        Ok(())
    }

    /// Render and resolve to tone-mapped sRGB RGBA8.
    ///
    /// Errors from the backend are not swallowed: a failure returns the film as
    /// it stands, which for a GPU backend that could not start is a black
    /// frame. Use [`Self::render`] directly if you need to see the error.
    pub fn render_to_rgba(&mut self, scene: &mut Scene, camera: &dyn Camera) -> Vec<u8> {
        let _ = self.render(scene, camera);
        self.resolve_rgba()
    }

    /// Render and resolve to linear RGBA f32 — no tone curve, no sRGB, values
    /// above 1 intact. This is the one to feed a compositor or an EXR writer.
    pub fn render_to_hdr(&mut self, scene: &mut Scene, camera: &dyn Camera) -> Vec<f32> {
        let _ = self.render(scene, camera);
        self.resolve_hdr()
    }

    /// Render and encode the beauty pass as a ZIP-compressed linear EXR.
    pub fn render_to_exr(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
    ) -> Result<Vec<u8>, String> {
        self.render(scene, camera)
            .map_err(|e| e.to_string())?;
        self.encode_exr()
    }

    /// Encode the current film as linear ZIP EXR (no additional tracing).
    pub fn encode_exr(&mut self) -> Result<Vec<u8>, String> {
        self.sync_gpu_film().map_err(|e| e.to_string())?;
        let (w, h) = self.size();
        encode_exr_rgba(w, h, &self.resolve_hdr())
    }

    /// Write the current film to an EXR file.
    pub fn save_exr(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        let bytes = self.encode_exr()?;
        std::fs::write(path.as_ref(), bytes).map_err(|e| e.to_string())
    }

    /// Save the accumulated film to a checkpoint file.
    pub fn save_film(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        self.sync_gpu_film().map_err(|e| e.to_string())?;
        std::fs::write(path.as_ref(), self.film_checkpoint().to_bytes()).map_err(|e| e.to_string())
    }

    /// Raw film checkpoint for in-memory save/resume (browser, tests, IPC).
    pub fn film_checkpoint(&mut self) -> FilmCheckpoint {
        let _ = self.sync_gpu_film();
        FilmCheckpoint::from_film(&self.film)
    }

    /// Restore the film from a checkpoint file.
    pub fn load_film(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        let bytes = std::fs::read(path.as_ref()).map_err(|e| e.to_string())?;
        self.restore_film_bytes(&bytes)
    }

    /// Restore accumulated samples from a checkpoint.
    pub fn restore_film(&mut self, cp: &FilmCheckpoint) -> Result<(), String> {
        if self.size() != (cp.width(), cp.height()) {
            self.film = Film::new(cp.width(), cp.height());
        }
        cp.apply_to(&mut self.film)?;
        self.reset_gpu_accum();
        Ok(())
    }

    /// Restore from [`FilmCheckpoint::to_bytes`] output.
    pub fn restore_film_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        let cp = FilmCheckpoint::from_bytes(bytes)?;
        self.restore_film(&cp)
    }

    /// Count tiles that still need samples (0 when adaptive is off).
    pub fn unconverged_tile_count(&self, tile_size: u32) -> u32 {
        let threshold = self.settings.adaptive_threshold;
        if threshold <= 0.0 {
            return 0;
        }
        RenderRect::unconverged_tiles(
            &self.film,
            threshold,
            self.settings.adaptive_min_samples,
            tile_size,
        )
        .count() as u32
    }

    /// Resolve what has been accumulated so far to RGBA8, denoising first if
    /// the settings ask for it.
    pub fn resolve_rgba(&mut self) -> Vec<u8> {
        if self.settings.aov != Aov::Beauty {
            // The AOVs are already noise-free by construction; running them
            // through a colour-guided filter would only blur them.
            let _ = self.sync_gpu_film();
            return self.film.resolve_rgba8(&self.settings);
        }
        let hdr = self.resolve_hdr();
        self.film.beauty_rgba8(&hdr, &self.settings)
    }

    /// Resolve to RGBA8, choosing whether to denoise regardless of the
    /// settings.
    ///
    /// Denoising is a post-process over the finished film, so both answers come
    /// from the *same* samples — which is what makes a with/against comparison
    /// mean anything. Going through [`Self::set_settings`] instead would clear
    /// the film and rebuild the scene, so the two images would differ by their
    /// noise as well as by the filter.
    pub fn resolve_rgba_denoised(&mut self, denoise: bool) -> Vec<u8> {
        if self.settings.aov != Aov::Beauty {
            let _ = self.sync_gpu_film();
            return self.film.resolve_rgba8(&self.settings);
        }
        let hdr = self.resolve_hdr_denoised(denoise);
        self.film.beauty_rgba8(&hdr, &self.settings)
    }

    /// Resolve to linear RGBA f32, denoised if enabled.
    pub fn resolve_hdr(&mut self) -> Vec<f32> {
        self.resolve_hdr_denoised(self.settings.denoise)
    }

    /// Resolve to linear RGBA f32, choosing whether to denoise regardless of
    /// the settings. See [`Self::resolve_rgba_denoised`].
    pub fn resolve_hdr_denoised(&mut self, denoise: bool) -> Vec<f32> {
        let _ = self.sync_gpu_film();
        let hdr = self.film.resolve_hdr();
        if !denoise || self.film.samples() == 0 {
            return hdr;
        }
        let scale = self.scene.as_ref().map(|s| s.scale()).unwrap_or(1.0);

        #[cfg(all(feature = "learned-denoise", not(target_arch = "wasm32")))]
        if let Some(learned) = &self.learned {
            match learned.borrow_mut().apply(&self.film, scale) {
                Ok(out) => return out,
                // A trained denoiser that cannot run is worth saying so about,
                // but not worth losing the render over — the unfiltered image
                // is still the render.
                Err(e) => eprintln!(
                    "raytrace: learned denoiser failed ({e}); leaving the film unfiltered"
                ),
            }
        }

        // No global "how converged is this render" guess: the colour threshold
        // is measured per pixel from its own accumulated variance, so a settled
        // region and a noisy one in the same image are filtered differently.
        let albedo = self.film.resolve_albedo();
        let normal = self.film.resolve_normal();
        let depth = self.film.resolve_depth();
        let variance = self.film.resolve_variance();
        let sample_counts = self.film.resolve_sample_counts();
        denoise::denoise(
            self.film.width(),
            self.film.height(),
            &hdr,
            &DenoiseGuides {
                albedo: &albedo,
                normal: &normal,
                depth: &depth,
                variance: &variance,
                sample_counts: Some(&sample_counts),
                scene_scale: scale,
            },
            &self.denoise_params,
        )
    }

    /// Point a new camera at the scene already flattened, and start the film
    /// over.
    ///
    /// The acceleration structure describes the geometry, not the view, so a
    /// shot that only moves the camera has no reason to rebuild it — and for a
    /// scene of any size the build dominates a low-sample frame. Returns
    /// `false` if there is nothing prepared yet, in which case the caller still
    /// needs [`Self::prepare`].
    ///
    /// The scene's transforms are *not* re-read. Anything that moves in the
    /// scene itself still needs a full `prepare`.
    pub fn set_camera(&mut self, camera: &dyn Camera) -> bool {
        let Some(scene) = self.scene.as_ref() else {
            return false;
        };
        if let Some(cur) = self.camera.take() {
            self.prev_camera = Some(cur);
        }
        self.camera = Some(self.build_camera(camera, scene));
        self.film.clear();
        self.reset_gpu_accum();
        self.last_tile_ms = 0.0;
        true
    }

    /// Discard accumulated samples but keep the flattened scene — for when only
    /// the exposure or the tone curve changed and the geometry did not.
    pub fn reset_film(&mut self) {
        self.film.clear();
        self.reset_gpu_accum();
        self.last_tile_ms = 0.0;
    }

    /// Fold another film's samples into this one (distributed / multi-process).
    /// Sizes must match; returns `false` when they do not.
    pub fn merge_film(&mut self, other: &Film) -> bool {
        if other.width() != self.film.width() || other.height() != self.film.height() {
            return false;
        }
        self.film.merge(other);
        self.reset_gpu_accum();
        true
    }

    /// Merge a checkpoint's pixels into the current film.
    pub fn merge_film_checkpoint(&mut self, cp: &FilmCheckpoint) -> Result<(), String> {
        let mut tmp = Film::new(cp.width(), cp.height());
        cp.apply_to(&mut tmp)?;
        if !self.merge_film(&tmp) {
            return Err(format!(
                "checkpoint size {}×{} does not match film {}×{}",
                cp.width(),
                cp.height(),
                self.film.width(),
                self.film.height()
            ));
        }
        Ok(())
    }

    /// Per-pixel sample counts (for debug overlays).
    pub fn resolve_sample_counts(&mut self) -> Vec<u32> {
        let _ = self.sync_gpu_film();
        self.film.resolve_sample_counts()
    }

    /// Per-pixel relative error estimates (infinite where unknown).
    pub fn resolve_error(&mut self) -> Vec<f32> {
        let _ = self.sync_gpu_film();
        self.film.resolve_error()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cameras::PerspectiveCamera;
    use crate::raytrace::gpu::GpuBackend;
    use crate::core::Object3D;
    use crate::geometries::BoxGeometry;
    use crate::lights::AmbientLight;
    use crate::materials::{Material, StandardMaterial};
    use crate::math::{Color, Vector3};

    fn scene_and_camera() -> (Scene, PerspectiveCamera) {
        let mut scene = Scene::new();
        scene.background = Color::new(0.05, 0.05, 0.08);
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            BoxGeometry::new(2.0, 2.0, 2.0),
            Material::Standard(StandardMaterial::new(Color::new(0.7, 0.4, 0.2))),
        )));
        scene.add_light(AmbientLight::new(Color::WHITE, 0.7));
        let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(3.0, 2.5, 4.0);
        cam.target = Vector3::ZERO;
        (scene, cam)
    }

    #[test]
    fn renders_an_image_of_the_right_size() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(24, 16);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(4)
                .with_denoise(false),
        );
        let rgba = r.render_to_rgba(&mut scene, &cam);
        assert_eq!(rgba.len(), 24 * 16 * 4);
        assert_eq!(r.samples(), 4);
        assert_eq!(r.backend_name(), "cpu");
    }

    #[test]
    fn the_subject_is_distinguishable_from_the_background() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(32, 32);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(16)
                .with_denoise(false),
        );
        let rgba = r.render_to_rgba(&mut scene, &cam);
        let px = |x: usize, y: usize| rgba[(y * 32 + x) * 4];
        // The corner sees background; the centre sees the lit box.
        assert!(
            px(16, 16) > px(0, 0) + 20,
            "centre {} vs corner {}",
            px(16, 16),
            px(0, 0)
        );
    }

    #[test]
    fn progressive_accumulation_converges_toward_the_one_shot_render() {
        let (mut scene, cam) = scene_and_camera();
        let settings = RaytraceSettings::default()
            .with_samples(32)
            .with_denoise(false);

        let mut a = RaytraceRenderer::new(16, 16);
        a.set_settings(settings.clone());
        a.render(&mut scene, &cam).unwrap();

        let mut b = RaytraceRenderer::new(16, 16);
        b.set_settings(settings);
        b.prepare(&mut scene, &cam);
        for _ in 0..4 {
            b.accumulate(8).unwrap();
        }
        assert_eq!(b.samples(), 32);

        let ha = a.resolve_hdr();
        let hb = b.resolve_hdr();
        for (x, y) in ha.iter().zip(hb.iter()) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    /// A progressive render has to end at the same image a one-shot render of
    /// the same budget produces. If it does not, the batching is losing or
    /// double-counting samples and every progressive frame is of a different
    /// render than the one the caller asked for.
    #[test]
    fn a_progressive_render_ends_where_the_one_shot_render_does() {
        let (mut scene, cam) = scene_and_camera();
        let settings = RaytraceSettings::default()
            .with_samples(32)
            .with_denoise(false);

        let mut one_shot = RaytraceRenderer::new(16, 16);
        one_shot.set_settings(settings.clone());
        one_shot.render(&mut scene, &cam).unwrap();

        let mut progressive = RaytraceRenderer::new(16, 16);
        progressive.set_settings(settings);
        let mut seen = Vec::new();
        progressive
            .render_progressive(
                &mut scene,
                &cam,
                &ProgressiveOptions {
                    batch: 8,
                    denoise: false,
                    ..Default::default()
                },
                |frame| {
                    seen.push(frame.samples);
                    true
                },
            )
            .unwrap();

        assert_eq!(seen, vec![8, 16, 24, 32], "one frame per batch");
        assert_eq!(progressive.samples(), 32);
        let a = one_shot.resolve_hdr();
        let b = progressive.resolve_hdr();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    /// `start_sample` suppresses the early frames and `min_step` spreads out
    /// the later ones, but neither may swallow the finished image — a caller
    /// that only draws what it is handed would otherwise never see the render
    /// it asked for.
    #[test]
    fn pacing_never_swallows_the_final_frame() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(8, 8);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(10)
                .with_denoise(false),
        );

        let mut seen = Vec::new();
        r.render_progressive(
            &mut scene,
            &cam,
            &ProgressiveOptions {
                batch: 1,
                start_sample: 4,
                min_step: 3,
                denoise: false,
                ..Default::default()
            },
            |frame| {
                assert!(frame.samples >= 4 || frame.is_last());
                seen.push(frame.samples);
                true
            },
        )
        .unwrap();

        assert_eq!(
            seen,
            vec![4, 7, 10],
            "start at 4, then every 3, ending at 10"
        );
        assert!(
            seen.last().copied() == Some(10),
            "the last frame is the render"
        );
    }

    /// Returning false stops the render where it stands. A viewport drops the
    /// remaining samples when the camera moves, and the film has to be left
    /// holding exactly what was traced rather than the full budget.
    #[test]
    fn a_progressive_render_stops_when_the_caller_asks() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(8, 8);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(64)
                .with_denoise(false),
        );

        let mut frames = 0;
        r.render_progressive(
            &mut scene,
            &cam,
            &ProgressiveOptions {
                batch: 4,
                denoise: false,
                ..Default::default()
            },
            |_| {
                frames += 1;
                frames < 2
            },
        )
        .unwrap();

        assert_eq!(frames, 2);
        assert_eq!(
            r.samples(),
            8,
            "stopped after two batches, not at the budget"
        );
    }

    #[test]
    fn changing_the_settings_discards_the_film() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(8, 8);
        r.set_settings(RaytraceSettings::default().with_samples(4));
        r.render(&mut scene, &cam).unwrap();
        assert_eq!(r.samples(), 4);
        r.set_settings(RaytraceSettings::default().with_samples(2));
        assert_eq!(r.samples(), 0);
        assert!(
            r.report().is_none(),
            "the flattened scene should be dropped too"
        );
    }

    #[test]
    fn resizing_clears_the_film() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(8, 8);
        r.set_settings(RaytraceSettings::default().with_samples(2));
        r.render(&mut scene, &cam).unwrap();
        r.set_size(12, 6);
        assert_eq!(r.size(), (12, 6));
        assert_eq!(r.samples(), 0);
    }

    #[test]
    fn accumulate_without_prepare_returns_not_prepared() {
        let mut r = RaytraceRenderer::new(4, 4);
        assert!(matches!(r.accumulate(4), Err(RaytraceError::NotPrepared)));
        assert_eq!(r.samples(), 0);
    }

    #[test]
    fn prepare_if_changed_camera_move_keeps_gpu_scene_cache() {
        let gpu = match GpuBackend::headless() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::with_backend(16, 16, Box::new(gpu));
        r.set_settings(RaytraceSettings::default().with_samples(8).with_denoise(false));
        r.prepare(&mut scene, &cam);
        assert!(!r.gpu_scene_cached(), "BVH uploads on first trace, not prepare");
        r.accumulate(1).unwrap();
        assert!(r.gpu_scene_cached());
        let mut cam2 = cam;
        cam2.position.x += 0.5;
        assert!(!r.prepare_if_changed(&mut scene, &cam2));
        assert!(r.gpu_scene_cached(), "camera-only update must not repack BVH");
        assert_eq!(r.samples(), 0);
    }

    #[test]
    fn film_checkpoint_roundtrip_through_renderer() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(8, 8);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(16)
                .with_denoise(false),
        );
        r.prepare(&mut scene, &cam);
        r.accumulate(8).unwrap();
        let bytes = r.film_checkpoint().to_bytes();
        r.accumulate(8).unwrap();
        assert_eq!(r.samples(), 16);
        let target = r.resolve_hdr();
        r.restore_film_bytes(&bytes).unwrap();
        assert_eq!(r.samples(), 8);
        r.accumulate(8).unwrap();
        let resumed = r.resolve_hdr();
        for (a, b) in target.iter().zip(resumed.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn redistribution_can_finish_before_the_sample_budget() {
        let mut scene = Scene::new();
        scene.background = Color::new(0.45, 0.45, 0.45);
        let cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        let mut r = RaytraceRenderer::new(16, 16);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(128)
                .with_adaptive(0.08)
                .with_adaptive_min_samples(4)
                .with_sample_redistribution(true)
                .with_denoise(false),
        );
        r.prepare(&mut scene, &cam);
        for _ in 0..32 {
            if !r.needs_more_samples() {
                break;
            }
            r.accumulate_unconverged_tiles(8, 4).unwrap();
        }
        assert!(r.is_converged(), "uniform background should converge");
        assert!(
            r.samples() < 128,
            "stopped at {} spp, not the full budget",
            r.samples()
        );
    }

    #[test]
    fn multi_tile_accumulate_advances_the_budget_once() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(16, 16);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(64)
                .with_adaptive(0.0)
                .with_denoise(false),
        );
        r.prepare(&mut scene, &cam);
        r.accumulate_unconverged_tiles_limited(8, 1, 4).unwrap();
        assert_eq!(r.samples(), 1, "four tiles share one batch counter");
    }

    #[test]
    fn budgeted_scheduler_learns_tile_cost() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(32, 32);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(64)
                .with_adaptive(0.05)
                .with_sample_redistribution(true)
                .with_denoise(false),
        );
        r.prepare(&mut scene, &cam);
        assert_eq!(r.last_tile_ms(), 0.0);
        let n = r
            .accumulate_unconverged_tiles_budgeted(8, 1, 4, 50)
            .unwrap();
        assert!(n >= 1);
        assert!(r.last_tile_ms() > 0.0);
        assert_eq!(r.samples(), 1);
    }

    #[test]
    fn zero_tile_size_uses_recommended() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(128, 128);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(4)
                .with_adaptive(0.0)
                .with_denoise(false),
        );
        r.prepare(&mut scene, &cam);
        let rec = r.recommended_tile_size();
        assert!(rec >= 8);
        r.accumulate_unconverged_tiles_limited(0, 1, 1).unwrap();
        assert_eq!(r.samples(), 1);
        assert!(r.unconverged_tile_count(rec) > 0 || r.samples() > 0);
    }

    #[test]
    fn set_denoise_preserves_the_film() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(8, 8);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(8)
                .with_denoise(false),
        );
        r.prepare(&mut scene, &cam);
        r.accumulate(4).unwrap();
        assert_eq!(r.samples(), 4);
        r.set_denoise(true);
        assert_eq!(r.samples(), 4);
        assert!(r.settings().denoise);
    }

    #[test]
    fn set_exposure_preserves_the_film() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(8, 8);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(8)
                .with_denoise(false),
        );
        r.prepare(&mut scene, &cam);
        r.accumulate(2).unwrap();
        r.set_exposure(2.0);
        assert_eq!(r.samples(), 2);
        assert!((r.settings().exposure - 2.0).abs() < 1e-6);
    }

    #[test]
    fn error_overlay_keeps_rgba_size() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(8, 8);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(4)
                .with_denoise(false),
        );
        r.prepare(&mut scene, &cam);
        r.accumulate(2).unwrap();
        let overlay = r.resolve_rgba_error_overlay(0.5);
        assert_eq!(overlay.len(), 8 * 8 * 4);
        assert!(overlay.iter().any(|&c| c > 0));
    }

    #[test]
    fn aov_resolve_does_not_require_settings_mutation() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(8, 8);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(4)
                .with_denoise(false),
        );
        r.prepare(&mut scene, &cam);
        r.accumulate(2).unwrap();
        assert_eq!(r.settings().aov, Aov::Beauty);
        let albedo = r.resolve_rgba_aov(Aov::Albedo);
        let samples = r.resolve_rgba_samples();
        assert_eq!(albedo.len(), 8 * 8 * 4);
        assert_eq!(samples.len(), 8 * 8 * 4);
        assert_eq!(r.settings().aov, Aov::Beauty);
        r.set_aov(Aov::Normal);
        assert_eq!(r.settings().aov, Aov::Normal);
        assert_eq!(r.samples(), 2);
    }

    #[test]
    fn adaptive_toggles_preserve_the_film() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(8, 8);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(8)
                .with_adaptive(0.02)
                .with_sample_redistribution(true)
                .with_denoise(false),
        );
        r.prepare(&mut scene, &cam);
        r.accumulate(4).unwrap();
        r.set_adaptive(0.0);
        r.set_sample_redistribution(false);
        r.set_adaptive_min_samples(8);
        assert_eq!(r.samples(), 4);
        assert_eq!(r.settings().adaptive_threshold, 0.0);
        assert!(!r.settings().sample_redistribution);
    }

    #[test]
    fn sample_budget_extends_without_clearing() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(8, 8);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(4)
                .with_denoise(false),
        );
        r.prepare(&mut scene, &cam);
        r.accumulate(4).unwrap();
        assert!(!r.needs_more_samples());
        r.set_sample_budget(8);
        assert!(r.needs_more_samples());
        assert_eq!(r.samples(), 4);
        r.accumulate(4).unwrap();
        assert_eq!(r.samples(), 8);
    }

    #[test]
    fn adaptive_stops_without_redistribution() {
        let mut scene = Scene::new();
        scene.background = Color::new(0.45, 0.45, 0.45);
        let cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        let mut r = RaytraceRenderer::new(16, 16);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(128)
                .with_adaptive(0.08)
                .with_adaptive_min_samples(4)
                .with_sample_redistribution(false)
                .with_denoise(false),
        );
        r.prepare(&mut scene, &cam);
        for _ in 0..32 {
            if !r.needs_more_samples() {
                break;
            }
            r.accumulate_unconverged_tiles(8, 4).unwrap();
        }
        assert!(r.is_converged());
        assert!(r.samples() < 128, "stopped at {} spp", r.samples());
    }

    #[test]
    fn encode_exr_from_partial_film() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(8, 8);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(4)
                .with_denoise(false),
        );
        r.prepare(&mut scene, &cam);
        r.accumulate(2).unwrap();
        let exr = r.encode_exr().expect("exr");
        assert!(exr.len() > 64);
        assert_eq!(&exr[..4], b"\x76\x2f\x31\x01");
    }

    #[test]
    fn merge_film_sums_partial_renders() {
        let (mut scene, cam) = scene_and_camera();
        let settings = RaytraceSettings::default()
            .with_samples(16)
            .with_denoise(false);
        let mut a = RaytraceRenderer::new(8, 8);
        a.set_settings(settings.clone());
        a.prepare(&mut scene, &cam);
        a.accumulate(4).unwrap();
        let cp = a.film_checkpoint();

        let mut b = RaytraceRenderer::new(8, 8);
        b.set_settings(settings);
        b.prepare(&mut scene, &cam);
        b.accumulate(4).unwrap();
        b.merge_film_checkpoint(&cp).unwrap();
        assert_eq!(b.samples(), 8);
        assert!(b.film().pixels().iter().all(|p| p.samples == 8));
        let hdr = b.resolve_hdr();
        assert!(hdr.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn the_build_report_describes_the_scene() {
        let (mut scene, cam) = scene_and_camera();
        let mut r = RaytraceRenderer::new(4, 4);
        r.prepare(&mut scene, &cam);
        let report = r.report().expect("prepared");
        assert_eq!(report.triangles, 12);
        assert_eq!(report.materials, 1);
        assert!(report.summary().contains("12 triangles"));
    }

    #[test]
    fn hdr_output_is_linear_and_unclamped() {
        let mut scene = Scene::new();
        scene.background = Color::new(4.0, 4.0, 4.0);
        let cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        let mut r = RaytraceRenderer::new(4, 4);
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(2)
                .with_denoise(false),
        );
        let hdr = r.render_to_hdr(&mut scene, &cam);
        assert!((hdr[0] - 4.0).abs() < 1e-4, "{}", hdr[0]);
        // The 8-bit path tone-maps it down.
        let rgba = r.resolve_rgba();
        assert!(rgba[0] < 255);
    }

    #[test]
    fn the_aov_channels_render() {
        let (mut scene, cam) = scene_and_camera();
        for aov in [Aov::Albedo, Aov::Normal, Aov::Depth] {
            let mut r = RaytraceRenderer::new(16, 16);
            r.set_settings(RaytraceSettings::default().with_samples(4).with_aov(aov));
            let rgba = r.render_to_rgba(&mut scene, &cam);
            assert_eq!(rgba.len(), 16 * 16 * 4);
            assert!(
                rgba.chunks_exact(4).any(|p| p[0] != rgba[0]),
                "{aov:?} produced a constant image"
            );
        }
    }
}
