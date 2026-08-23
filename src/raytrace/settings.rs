//! Everything the tracer is told before it starts: how many samples, how deep
//! to follow a path, what the camera's lens does, and what an escaping ray sees.

use crate::renderer::ToneMapping;

/// What a camera ray that hits nothing returns.
///
/// three.js keeps `scene.background` and `scene.environment` separate — one is
/// what you see behind the subject, the other is what lights it — and so does
/// this. Changing the mode never changes the lighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackgroundMode {
    /// `scene.background`, at `scene.background_alpha`.
    #[default]
    Color,
    /// `scene.environment`, so the reflections and the backdrop agree. Falls
    /// back to the colour when the scene has no environment.
    Environment,
    /// Alpha 0, no radiance — the subject on a clear plate, for compositing.
    Transparent,
}

/// Which quantity the film reports. The extra channels are the ones a denoiser
/// needs, and they are free: the tracer already knows all three at the first
/// hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Aov {
    /// Path-traced radiance — the actual render.
    #[default]
    Beauty,
    /// Base colour at the first hit, unlit. Noise-free.
    Albedo,
    /// Shading normal at the first hit, mapped to `0.5 * n + 0.5`.
    Normal,
    /// Distance from the camera to the first hit, normalised over the scene's
    /// depth range.
    Depth,
}

/// Path tracer configuration.
///
/// The defaults are a preview: 32 samples, 6 bounces, denoiser on. That
/// resolves a lit interior in seconds rather than minutes and is what you want
/// while framing a shot. For a final frame raise `samples_per_pixel` — noise
/// falls as `1/sqrt(n)`, so four times the samples is half the noise — and turn
/// the denoiser off if you would rather not have it smooth over fine detail.
#[derive(Debug, Clone)]
pub struct RaytraceSettings {
    /// Paths traced per pixel.
    pub samples_per_pixel: u32,
    /// Hard ceiling on path length. 0 gives direct camera visibility only (no
    /// shading at all); 1 is direct lighting; every step above that adds a
    /// bounce of indirect light.
    pub max_bounces: u32,
    /// Bounces taken before Russian roulette may kill a path. Roulette is
    /// unbiased but noisy, and killing paths that have barely started shows up
    /// as speckle in the shadows.
    pub min_bounces: u32,
    /// How many purely transparent surfaces (alpha < 1) a ray may pass through.
    /// These do not count as bounces — foliage cards would exhaust a path
    /// budget on the first leaf otherwise.
    pub transparent_max_bounces: u32,
    /// Ceiling on the radiance a single *indirect* sample may contribute. A
    /// path that finds a bright light through a near-specular chain lands a
    /// value thousands of times the mean, and one such sample is a permanent
    /// white dot. Clamping biases the result slightly darker in exchange for
    /// removing it. 0 disables.
    pub clamp_indirect: f32,
    /// Same, for directly-visible light. Usually left at 0: clamping direct
    /// light dims the lights themselves.
    pub clamp_direct: f32,
    /// How far a secondary ray is pushed off the surface it starts from, to
    /// stop it re-hitting that surface through floating-point error. In world
    /// units; raise it if a large scene shows shadow acne.
    pub ray_epsilon: f32,
    /// Seed for the sample sequence. Two renders with the same seed, settings
    /// and scene are identical, whatever the thread count.
    pub seed: u64,
    /// Applied when the film is converted to 8-bit. Path-traced output is
    /// genuinely high dynamic range, so [`ToneMapping::AcesFilmic`] is the
    /// default here — unlike the raster renderer, which defaults to none to
    /// match three.js.
    pub tone_mapping: ToneMapping,
    /// Linear exposure multiplier applied before the tone curve.
    pub exposure: f32,
    /// What a camera ray that escapes the scene returns.
    pub background: BackgroundMode,
    /// Multiplier on the environment map's contribution to lighting *and*
    /// background.
    pub environment_intensity: f32,
    /// Angular *radius* of every directional light's disc, in radians. 0 keeps
    /// them mathematically parallel and their shadows hard. The real sun is
    /// 0.00465; anything up to ~0.1 reads as an overcast sky.
    pub sun_angular_radius: f32,
    /// World-space radius given to every point and spot light, turning them
    /// from delta sources into spheres with penumbrae. 0 keeps them point-like.
    pub light_radius: f32,
    /// Lens radius in world units. 0 is a pinhole — everything in focus.
    pub aperture: f32,
    /// Distance to the focal plane. 0 focuses on whatever the centre of the
    /// frame hits, and on the far plane if that is nothing.
    pub focus_distance: f32,
    /// Stop sampling a pixel once its estimate's relative standard error falls
    /// below this. 0 disables, and every pixel takes the full budget.
    ///
    /// Noise falls as `1/sqrt(n)`, so the last halving of the error costs three
    /// quarters of the render — and most of an image reaches any given error
    /// long before its worst pixels do. Spending the budget where the number is
    /// still high is typically a third to a half off the wall clock for an
    /// image that is indistinguishable.
    ///
    /// It does not bias the result: a pixel that stops early is still the mean
    /// of its own samples, and the film divides by each pixel's own count.
    pub adaptive_threshold: f32,
    /// Samples a pixel must take before its error estimate is trusted. Below a
    /// dozen or so the sample variance is itself too noisy to threshold on, and
    /// a pixel that happened to draw similar values twice would stop
    /// immediately.
    pub adaptive_min_samples: u32,
    /// Run the AOV-guided denoiser over the accumulated film before output.
    pub denoise: bool,
    /// Which channel [`crate::raytrace::RaytraceRenderer::render_to_rgba`]
    /// writes. The AOVs are diagnostics — [`Aov::Beauty`] is the render.
    pub aov: Aov,
}

impl Default for RaytraceSettings {
    fn default() -> Self {
        Self {
            samples_per_pixel: 32,
            max_bounces: 6,
            min_bounces: 3,
            transparent_max_bounces: 8,
            clamp_indirect: 10.0,
            clamp_direct: 0.0,
            ray_epsilon: 1e-4,
            seed: 0x5eed_1234_abcd_0001,
            tone_mapping: ToneMapping::AcesFilmic,
            exposure: 1.0,
            background: BackgroundMode::Color,
            environment_intensity: 1.0,
            sun_angular_radius: 0.0,
            light_radius: 0.0,
            aperture: 0.0,
            focus_distance: 0.0,
            adaptive_threshold: 0.01,
            adaptive_min_samples: 16,
            denoise: true,
            aov: Aov::Beauty,
        }
    }
}

impl RaytraceSettings {
    /// Fast and noisy — for framing, not for output.
    pub fn preview() -> Self {
        Self {
            samples_per_pixel: 16,
            max_bounces: 4,
            denoise: true,
            ..Self::default()
        }
    }

    /// Converged, denoiser off, no indirect clamp, every pixel taking its full
    /// budget: the reference image, and the one to compare against when you
    /// want to know what the approximations cost you.
    pub fn final_quality() -> Self {
        Self {
            samples_per_pixel: 512,
            max_bounces: 12,
            min_bounces: 4,
            clamp_indirect: 0.0,
            adaptive_threshold: 0.0,
            denoise: false,
            ..Self::default()
        }
    }

    pub fn with_samples(mut self, n: u32) -> Self {
        self.samples_per_pixel = n.max(1);
        self
    }

    pub fn with_bounces(mut self, n: u32) -> Self {
        self.max_bounces = n;
        self.min_bounces = self.min_bounces.min(n);
        self
    }

    pub fn with_background(mut self, mode: BackgroundMode) -> Self {
        self.background = mode;
        self
    }

    pub fn with_tone_mapping(mut self, mode: ToneMapping, exposure: f32) -> Self {
        self.tone_mapping = mode;
        self.exposure = exposure;
        self
    }

    /// Depth of field. `aperture` is the lens radius in world units — larger
    /// blurs more — and `focus_distance` is where the plane of sharp focus
    /// sits.
    pub fn with_depth_of_field(mut self, aperture: f32, focus_distance: f32) -> Self {
        self.aperture = aperture.max(0.0);
        self.focus_distance = focus_distance.max(0.0);
        self
    }

    /// Stop pixels once their relative standard error drops below `threshold`.
    /// Pass 0 to give every pixel the full sample budget.
    pub fn with_adaptive(mut self, threshold: f32) -> Self {
        self.adaptive_threshold = threshold.max(0.0);
        self
    }

    pub fn with_denoise(mut self, on: bool) -> Self {
        self.denoise = on;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_aov(mut self, aov: Aov) -> Self {
        self.aov = aov;
        self
    }

    /// Soft shadows: give directional lights an angular radius and point/spot
    /// lights a world-space one.
    pub fn with_soft_shadows(mut self, sun_angular_radius: f32, light_radius: f32) -> Self {
        self.sun_angular_radius = sun_angular_radius.max(0.0);
        self.light_radius = light_radius.max(0.0);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_compose() {
        let s = RaytraceSettings::default()
            .with_samples(64)
            .with_bounces(2)
            .with_depth_of_field(0.05, 7.0)
            .with_denoise(false);
        assert_eq!(s.samples_per_pixel, 64);
        assert_eq!(s.max_bounces, 2);
        // min_bounces must not outrun max_bounces, or roulette never runs.
        assert!(s.min_bounces <= s.max_bounces);
        assert_eq!(s.aperture, 0.05);
        assert!(!s.denoise);
    }

    #[test]
    fn zero_samples_is_clamped_up() {
        assert_eq!(
            RaytraceSettings::default()
                .with_samples(0)
                .samples_per_pixel,
            1
        );
    }

    #[test]
    fn final_quality_drops_every_approximation() {
        let s = RaytraceSettings::final_quality();
        assert_eq!(s.clamp_indirect, 0.0);
        assert_eq!(
            s.adaptive_threshold, 0.0,
            "a reference render must not stop early"
        );
        assert!(!s.denoise);
    }

    #[test]
    fn adaptive_can_be_turned_off() {
        assert_eq!(
            RaytraceSettings::default()
                .with_adaptive(0.0)
                .adaptive_threshold,
            0.0
        );
        assert!(RaytraceSettings::default().adaptive_threshold > 0.0);
    }
}
