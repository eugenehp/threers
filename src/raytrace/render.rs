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

use super::backend::{probe_focus_distance, CpuBackend, RaytraceBackend, RaytraceError};
use super::camera::RtCamera;
use super::denoise::{self, DenoiseGuides, DenoiseParams};
use super::film::Film;
use super::scene::{BuildReport, RaytraceScene};
use super::settings::{Aov, RaytraceSettings};

/// How [`RaytraceRenderer::render_progressive`] paces its updates.
#[derive(Debug, Clone, Copy)]
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
}

impl Default for ProgressiveOptions {
    fn default() -> Self {
        Self {
            batch: 1,
            start_sample: 1,
            min_step: 1,
            denoise: true,
        }
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
}

impl ProgressiveFrame<'_> {
    /// Whether this is the last update of the render.
    pub fn is_last(&self) -> bool {
        self.samples >= self.total
    }
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
        Self {
            settings: RaytraceSettings::default(),
            denoise_params: DenoiseParams::default(),
            backend,
            film: Film::new(width, height),
            scene: None,
            camera: None,
            #[cfg(all(feature = "learned-denoise", not(target_arch = "wasm32")))]
            learned: None,
        }
    }

    /// Name of the backend in use.
    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
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
        self.backend.invalidate();
    }

    /// Tune the À-Trous denoiser.
    pub fn set_denoise_params(&mut self, params: DenoiseParams) {
        self.denoise_params = params;
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

    /// Resize, discarding the film.
    pub fn set_size(&mut self, width: u32, height: u32) {
        if (width, height) != self.size() {
            self.film = Film::new(width, height);
        } else {
            self.film.clear();
        }
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
        let counts = self.film.resolve_sample_counts();
        if counts.is_empty() || self.film.samples() == 0 {
            return 1.0;
        }
        let taken: u64 = counts.iter().map(|&n| n as u64).sum();
        let budget = counts.len() as u64 * self.film.samples() as u64;
        taken as f32 / budget.max(1) as f32
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
        let flat = RaytraceScene::build_for_layers(scene, &self.settings, layers);
        let mut rt_camera = RtCamera::new(camera, &self.settings);
        if self.settings.aperture > 0.0 && self.settings.focus_distance <= 0.0 {
            // Autofocus on whatever is in the centre of frame — the behaviour a
            // "focus distance of 0" most usefully means. Nothing there and the
            // scene's own extent stands in, so the lens still does something.
            let d = probe_focus_distance(&flat, &rt_camera).unwrap_or_else(|| flat.scale());
            rt_camera = rt_camera.with_focus_distance(d);
        }
        self.scene = Some(flat);
        self.camera = Some(rt_camera);
        self.film.clear();
        self.backend.invalidate();
    }

    /// Add `samples` samples per pixel to the film. Requires a previous
    /// [`Self::prepare`].
    pub fn accumulate(&mut self, samples: u32) -> Result<(), RaytraceError> {
        let (Some(scene), Some(camera)) = (self.scene.as_ref(), self.camera.as_ref()) else {
            return Ok(());
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
        let mut shown = 0u32;
        while self.film.samples() < total {
            let want = batch.min(total - self.film.samples());
            self.accumulate(want)?;
            let samples = self.film.samples();
            let last = samples >= total;
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
            if !on_frame(ProgressiveFrame {
                samples,
                total,
                denoised: denoise,
                image: &image,
            }) {
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

    /// Resolve what has been accumulated so far to RGBA8, denoising first if
    /// the settings ask for it.
    pub fn resolve_rgba(&self) -> Vec<u8> {
        if self.settings.aov != Aov::Beauty {
            // The AOVs are already noise-free by construction; running them
            // through a colour-guided filter would only blur them.
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
    pub fn resolve_rgba_denoised(&self, denoise: bool) -> Vec<u8> {
        if self.settings.aov != Aov::Beauty {
            return self.film.resolve_rgba8(&self.settings);
        }
        let hdr = self.resolve_hdr_denoised(denoise);
        self.film.beauty_rgba8(&hdr, &self.settings)
    }

    /// Resolve to linear RGBA f32, denoised if enabled.
    pub fn resolve_hdr(&self) -> Vec<f32> {
        self.resolve_hdr_denoised(self.settings.denoise)
    }

    /// Resolve to linear RGBA f32, choosing whether to denoise regardless of
    /// the settings. See [`Self::resolve_rgba_denoised`].
    pub fn resolve_hdr_denoised(&self, denoise: bool) -> Vec<f32> {
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
        denoise::denoise(
            self.film.width(),
            self.film.height(),
            &hdr,
            &DenoiseGuides {
                albedo: &self.film.resolve_albedo(),
                normal: &self.film.resolve_normal(),
                depth: &self.film.resolve_depth(),
                variance: &self.film.resolve_variance(),
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
        let mut rt_camera = RtCamera::new(camera, &self.settings);
        if self.settings.aperture > 0.0 && self.settings.focus_distance <= 0.0 {
            let d = probe_focus_distance(scene, &rt_camera).unwrap_or_else(|| scene.scale());
            rt_camera = rt_camera.with_focus_distance(d);
        }
        self.camera = Some(rt_camera);
        self.film.clear();
        self.backend.invalidate();
        true
    }

    /// Discard accumulated samples but keep the flattened scene — for when only
    /// the exposure or the tone curve changed and the geometry did not.
    pub fn reset_film(&mut self) {
        self.film.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cameras::PerspectiveCamera;
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
    fn accumulate_without_prepare_is_a_no_op_rather_than_a_panic() {
        let mut r = RaytraceRenderer::new(4, 4);
        r.accumulate(4).unwrap();
        assert_eq!(r.samples(), 0);
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
