//! Browser bindings for progressive path tracing.
//!
//! [`WebPathTracer`] wraps [`crate::raytrace::RaytraceRenderer`] with a CPU
//! backend by default. Call [`WebPathTracer::new_gpu`] when WebGPU is available
//! for the wgpu compute kernel, or [`WebPathTracer::new_gpu_from_renderer`] to
//! share the rasteriser's device.

use wasm_bindgen::prelude::*;

use crate::raytrace::{CpuBackend, DenoiseParams, RaytraceRenderer};

use crate::wasm::{WebCamera, WebRenderer, WebScene};

/// Progressive path tracer for the browser.
#[wasm_bindgen]
pub struct WebPathTracer {
    inner: RaytraceRenderer,
}

#[wasm_bindgen]
impl WebPathTracer {
    /// CPU path tracer — works everywhere, including workers without WebGPU.
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> WebPathTracer {
        Self {
            inner: RaytraceRenderer::new(width.max(1), height.max(1)),
        }
    }

    /// WebGPU compute path tracer. Async because adapter acquisition is async.
    pub async fn new_gpu(width: u32, height: u32) -> Result<WebPathTracer, JsValue> {
        let backend = crate::raytrace::gpu::GpuBackend::headless_browser()
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self {
            inner: RaytraceRenderer::with_backend(
                width.max(1),
                height.max(1),
                Box::new(backend),
            ),
        })
    }

    /// Build on the same WebGPU device as a [`WebRenderer`], so resources are
    /// not duplicated.
    pub fn new_gpu_from_renderer(
        renderer: &WebRenderer,
        width: u32,
        height: u32,
    ) -> WebPathTracer {
        let (device, queue) = renderer.gpu_device_queue();
        let backend = crate::raytrace::gpu::GpuBackend::with_device(device, queue);
        Self {
            inner: RaytraceRenderer::with_backend(
                width.max(1),
                height.max(1),
                Box::new(backend),
            ),
        }
    }

    #[wasm_bindgen(js_name = setSize)]
    pub fn set_size(&mut self, width: u32, height: u32) -> Vec<u32> {
        let (w, h) = self.inner.set_size(width.max(1), height.max(1));
        vec![w, h]
    }

    /// Largest film side the GPU accum buffer supports (`0` on CPU).
    #[wasm_bindgen(js_name = maxFilmSide)]
    pub fn max_film_side(&self) -> u32 {
        self.inner.max_film_side().unwrap_or(0)
    }

    /// Maximum film pixel count on GPU (`0` on CPU).
    #[wasm_bindgen(js_name = maxAccumPixels)]
    pub fn max_accum_pixels(&self) -> f64 {
        self.inner.max_film_pixels().unwrap_or(0) as f64
    }

    /// Texture atlas side used when packing maps on GPU (`0` on CPU).
    #[wasm_bindgen(js_name = atlasSide)]
    pub fn atlas_side(&self) -> u32 {
        self.inner.gpu_caps().map(|c| c.atlas_side).unwrap_or(0)
    }

    #[wasm_bindgen(js_name = setSamples)]
    pub fn set_samples(&mut self, spp: u32) {
        let s = self.inner.settings().clone().with_samples(spp.max(1));
        self.inner.set_settings(s);
    }

    #[wasm_bindgen(js_name = setBounces)]
    pub fn set_bounces(&mut self, n: u32) {
        let s = self.inner.settings().clone().with_bounces(n.max(1));
        self.inner.set_settings(s);
    }

    #[wasm_bindgen(js_name = setSeed)]
    pub fn set_seed(&mut self, seed: u64) {
        let s = self.inner.settings().clone().with_seed(seed);
        self.inner.set_settings(s);
    }

    #[wasm_bindgen(js_name = setDenoise)]
    pub fn set_denoise(&mut self, on: bool) {
        self.inner.set_denoise(on);
    }

    /// Tune À-Trous widths and iteration count (see [`DenoiseParams`]).
    #[wasm_bindgen(js_name = setDenoiseParams)]
    pub fn set_denoise_params(
        &mut self,
        iterations: u32,
        sigma_color: f32,
        sigma_normal: f32,
        sigma_depth: f32,
        sigma_albedo: f32,
        min_samples: u32,
    ) {
        self.inner.set_denoise_params(DenoiseParams {
            iterations,
            sigma_color,
            sigma_normal,
            sigma_depth,
            sigma_albedo,
            min_samples,
        });
    }

    #[wasm_bindgen(js_name = setExposure)]
    pub fn set_exposure(&mut self, exposure: f32) {
        self.inner.set_exposure(exposure);
    }

    /// `"beauty" | "albedo" | "normal" | "depth"` — resolve-time, keeps the film.
    #[wasm_bindgen(js_name = setAov)]
    pub fn set_aov(&mut self, name: &str) -> Result<(), JsValue> {
        let aov = match name.trim().to_ascii_lowercase().as_str() {
            "beauty" | "colour" | "color" => crate::raytrace::Aov::Beauty,
            "albedo" => crate::raytrace::Aov::Albedo,
            "normal" => crate::raytrace::Aov::Normal,
            "depth" => crate::raytrace::Aov::Depth,
            other => {
                return Err(JsValue::from_str(&format!(
                    "unknown aov '{other}' (beauty|albedo|normal|depth)"
                )))
            }
        };
        self.inner.set_aov(aov);
        Ok(())
    }

    /// Resolve a named AOV without changing the default resolve channel.
    #[wasm_bindgen(js_name = resolveRgbaAov)]
    pub fn resolve_rgba_aov(&mut self, name: &str) -> Result<Vec<u8>, JsValue> {
        let aov = match name.trim().to_ascii_lowercase().as_str() {
            "beauty" | "colour" | "color" => crate::raytrace::Aov::Beauty,
            "albedo" => crate::raytrace::Aov::Albedo,
            "normal" => crate::raytrace::Aov::Normal,
            "depth" => crate::raytrace::Aov::Depth,
            other => {
                return Err(JsValue::from_str(&format!(
                    "unknown aov '{other}' (beauty|albedo|normal|depth)"
                )))
            }
        };
        Ok(self.inner.resolve_rgba_aov(aov))
    }

    /// Grayscale sample-density map.
    #[wasm_bindgen(js_name = resolveRgbaSamples)]
    pub fn resolve_rgba_samples(&mut self) -> Vec<u8> {
        self.inner.resolve_rgba_samples()
    }

    /// Fewest and most samples any pixel took: `[min, max]`.
    #[wasm_bindgen(js_name = sampleRange)]
    pub fn sample_range(&self) -> Vec<u32> {
        let (lo, hi) = self.inner.sample_range();
        vec![lo, hi]
    }

    /// Scene build summary after prepare, or empty.
    #[wasm_bindgen(js_name = reportSummary)]
    pub fn report_summary(&self) -> String {
        self.inner
            .report()
            .map(|r| r.summary())
            .unwrap_or_default()
    }

    /// Beauty with a relative-error heatmap mix-in (`strength` 0..=1).
    #[wasm_bindgen(js_name = resolveRgbaErrorOverlay)]
    pub fn resolve_rgba_error_overlay(&mut self, strength: f32) -> Vec<u8> {
        self.inner.resolve_rgba_error_overlay(strength)
    }

    #[wasm_bindgen(js_name = setAdaptive)]
    pub fn set_adaptive(&mut self, threshold: f32) {
        self.inner.set_adaptive(threshold);
    }

    #[wasm_bindgen(js_name = setSampleRedistribution)]
    pub fn set_sample_redistribution(&mut self, on: bool) {
        self.inner.set_sample_redistribution(on);
    }

    #[wasm_bindgen(js_name = setAdaptiveMinSamples)]
    pub fn set_adaptive_min_samples(&mut self, n: u32) {
        self.inner.set_adaptive_min_samples(n);
    }

    /// Raise the sample budget without clearing the film.
    #[wasm_bindgen(js_name = setSampleBudget)]
    pub fn set_sample_budget(&mut self, spp: u32) {
        self.inner.set_sample_budget(spp);
    }

    /// Indirect firefly clamp (`0` = off). Keeps the film.
    #[wasm_bindgen(js_name = setClampIndirect)]
    pub fn set_clamp_indirect(&mut self, clamp: f32) {
        self.inner.set_clamp_indirect(clamp);
    }

    /// Biased glass caustic shadow hops. Keeps the film.
    #[wasm_bindgen(js_name = setCausticGlassShadows)]
    pub fn set_caustic_glass_shadows(&mut self, on: bool) {
        self.inner.set_caustic_glass_shadows(on);
    }

    /// GPU only: samples per compute dispatch.
    #[wasm_bindgen(js_name = setSamplesPerDispatch)]
    pub fn set_samples_per_dispatch(&mut self, n: u32) -> bool {
        self.inner.set_samples_per_dispatch(n)
    }

    #[wasm_bindgen(js_name = setMotionBlur)]
    pub fn set_motion_blur(&mut self, shutter: f32) {
        let s = self.inner.settings().clone().with_motion_blur(shutter);
        self.inner.set_settings(s);
    }

    /// Encode the current film as linear ZIP EXR bytes.
    #[wasm_bindgen(js_name = encodeExr)]
    pub fn encode_exr(&mut self) -> Result<Vec<u8>, JsValue> {
        self.inner
            .encode_exr()
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Suggested tile size for this resolution (multiple of 8).
    #[wasm_bindgen(js_name = recommendedTileSize)]
    pub fn recommended_tile_size(&self) -> u32 {
        self.inner.recommended_tile_size()
    }

    /// Flatten the scene and build acceleration structures.
    pub fn prepare(&mut self, scene: &mut WebScene, camera: &WebCamera) {
        camera.prepare_path_tracer(scene, &mut self.inner);
    }

    /// Rebuild only when geometry changed; refresh the camera otherwise.
    #[wasm_bindgen(js_name = prepareIfChanged)]
    pub fn prepare_if_changed(&mut self, scene: &mut WebScene, camera: &WebCamera) -> bool {
        camera.prepare_path_tracer_if_changed(scene, &mut self.inner)
    }

    /// Add `samples` paths per pixel to the film.
    pub fn accumulate(&mut self, samples: u32) -> Result<(), JsValue> {
        self.inner
            .accumulate(samples)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Add samples inside a screen rectangle (CPU backend traces only that region).
    #[wasm_bindgen(js_name = accumulateRegion)]
    pub fn accumulate_region(
        &mut self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        samples: u32,
    ) -> Result<(), JsValue> {
        self.inner
            .accumulate_region(x, y, width, height, samples)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Add samples per pixel across a grid of tiles.
    #[wasm_bindgen(js_name = accumulateTiles)]
    pub fn accumulate_tiles(&mut self, tile_size: u32, samples: u32) -> Result<(), JsValue> {
        self.inner
            .accumulate_tiles(tile_size, samples)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Like [`accumulateTiles`], but skip tiles whose pixels have all converged.
    #[wasm_bindgen(js_name = accumulateUnconvergedTiles)]
    pub fn accumulate_unconverged_tiles(
        &mut self,
        tile_size: u32,
        samples: u32,
    ) -> Result<(), JsValue> {
        self.inner
            .accumulate_unconverged_tiles(tile_size, samples)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Whether the sample budget remains or adaptive pixels are still noisy.
    #[wasm_bindgen(js_name = needsMoreSamples)]
    pub fn needs_more_samples(&self) -> bool {
        self.inner.needs_more_samples()
    }

    /// Like [`accumulateUnconvergedTiles`], but cap tiles per call (noisiest first).
    #[wasm_bindgen(js_name = accumulateUnconvergedTilesLimited)]
    pub fn accumulate_unconverged_tiles_limited(
        &mut self,
        tile_size: u32,
        samples: u32,
        max_tiles: u32,
    ) -> Result<(), JsValue> {
        self.inner
            .accumulate_unconverged_tiles_limited(tile_size, samples, max_tiles)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Trace under a soft millisecond budget; returns tiles actually traced.
    #[wasm_bindgen(js_name = accumulateUnconvergedTilesBudgeted)]
    pub fn accumulate_unconverged_tiles_budgeted(
        &mut self,
        tile_size: u32,
        samples: u32,
        max_tiles: u32,
        max_ms: u32,
    ) -> Result<u32, JsValue> {
        self.inner
            .accumulate_unconverged_tiles_budgeted(tile_size, samples, max_tiles, max_ms)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Rolling ms/tile estimate used by the budgeted scheduler.
    #[wasm_bindgen(js_name = lastTileMs)]
    pub fn last_tile_ms(&self) -> f32 {
        self.inner.last_tile_ms()
    }

    /// Film width in pixels.
    pub fn width(&self) -> u32 {
        self.inner.size().0
    }

    /// Film height in pixels.
    pub fn height(&self) -> u32 {
        self.inner.size().1
    }

    /// True when every pixel met the adaptive threshold.
    #[wasm_bindgen(js_name = isConverged)]
    pub fn is_converged(&self) -> bool {
        self.inner.is_converged()
    }

    /// Serialize the accumulated film for save/resume (see [`restoreFilmBytes`]).
    #[wasm_bindgen(js_name = filmCheckpointBytes)]
    pub fn film_checkpoint_bytes(&mut self) -> Vec<u8> {
        self.inner.film_checkpoint().to_bytes()
    }

    /// Restore a film checkpoint produced by [`filmCheckpointBytes`].
    #[wasm_bindgen(js_name = restoreFilmBytes)]
    pub fn restore_film_bytes(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        self.inner
            .restore_film_bytes(bytes)
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Tiles that still contain noisy pixels (0 when adaptive is off).
    #[wasm_bindgen(js_name = unconvergedTileCount)]
    pub fn unconverged_tile_count(&self, tile_size: u32) -> u32 {
        self.inner.unconverged_tile_count(tile_size)
    }

    /// Tone-mapped sRGB RGBA8 for the accumulated film so far.
    #[wasm_bindgen(js_name = resolveRgba)]
    pub fn resolve_rgba(&mut self) -> Vec<u8> {
        self.inner.resolve_rgba()
    }

    /// Resolve with an explicit denoise toggle (ignores the settings flag).
    #[wasm_bindgen(js_name = resolveRgbaDenoised)]
    pub fn resolve_rgba_denoised(&mut self, denoise: bool) -> Vec<u8> {
        self.inner.resolve_rgba_denoised(denoise)
    }

    /// Linear HDR RGBA f32 — no tone curve.
    #[wasm_bindgen(js_name = resolveHdr)]
    pub fn resolve_hdr(&mut self) -> Vec<f32> {
        self.inner.resolve_hdr()
    }

    /// Pull a deferred GPU film onto the host (no-op on CPU).
    #[wasm_bindgen(js_name = flushGpuFilm)]
    pub fn flush_gpu_film(&mut self) -> Result<(), JsValue> {
        self.inner
            .flush_gpu_film()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Defer GPU readback after tile traces until resolve/flush (GPU only).
    #[wasm_bindgen(js_name = setDeferGpuReadback)]
    pub fn set_defer_gpu_readback(&mut self, on: bool) -> bool {
        self.inner.set_defer_gpu_readback(on)
    }

    /// Samples per pixel accumulated so far.
    pub fn samples(&self) -> u32 {
        self.inner.film().samples()
    }

    /// Backend name: `"cpu"` or `"gpu"`.
    #[wasm_bindgen(js_name = backendName)]
    pub fn backend_name(&self) -> String {
        self.inner.backend_name().to_string()
    }

    /// Fraction of pixels that met the adaptive convergence threshold.
    #[wasm_bindgen(js_name = convergedFraction)]
    pub fn converged_fraction(&self) -> f32 {
        self.inner.render_stats().converged_fraction
    }

    /// Adaptive sampling efficiency (`1.0` = no savings).
    #[wasm_bindgen(js_name = sampleEfficiency)]
    pub fn sample_efficiency(&self) -> f32 {
        self.inner.sample_efficiency()
    }

    /// Clear the accumulated film.
    pub fn reset(&mut self) {
        self.inner.reset_film();
    }

    /// Merge another checkpoint into the current film (sizes must match).
    #[wasm_bindgen(js_name = mergeFilmBytes)]
    pub fn merge_film_bytes(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        let cp = crate::raytrace::FilmCheckpoint::from_bytes(bytes)
            .map_err(|e| JsValue::from_str(&e))?;
        self.inner
            .merge_film_checkpoint(&cp)
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Per-pixel sample counts for debug overlays.
    #[wasm_bindgen(js_name = resolveSampleCounts)]
    pub fn resolve_sample_counts(&mut self) -> Vec<u32> {
        self.inner.resolve_sample_counts()
    }

    /// Per-pixel relative error (debug / adaptive visualization).
    #[wasm_bindgen(js_name = resolveError)]
    pub fn resolve_error(&mut self) -> Vec<f32> {
        self.inner.resolve_error()
    }
}

impl WebPathTracer {
    /// Switch to CPU rendering (e.g. when WebGPU is unavailable).
    pub fn use_cpu(&mut self) {
        let (w, h) = self.inner.size();
        let settings = self.inner.settings().clone();
        let mut r = RaytraceRenderer::with_backend(w, h, Box::new(CpuBackend::new()));
        r.set_settings(settings);
        self.inner = r;
    }
}
