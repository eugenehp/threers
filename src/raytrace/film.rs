//! Where samples land: a running mean per pixel, plus the auxiliary channels
//! the denoiser and compositing need.
//!
//! The film accumulates rather than averages as it goes, so rendering is
//! *progressive*: 16 samples can be resolved and shown, then 16 more added to
//! the same film without restarting. That is what makes an interactive preview
//! possible and what lets a long render be checkpointed.

use crate::math::Vector3;
use crate::renderer::ToneMapping;

use super::settings::{Aov, RaytraceSettings};

const LUM_R: f64 = 0.2126;
const LUM_G: f64 = 0.7152;
const LUM_B: f64 = 0.0722;

/// Luminance of a single radiance sample.
#[inline]
pub(crate) fn luminance_sample(radiance: Vector3) -> f32 {
    0.2126 * radiance.x + 0.7152 * radiance.y + 0.0722 * radiance.z
}

/// Mean luminance of a pixel's accumulated radiance (`f64` for cancellation safety).
#[inline]
pub(crate) fn pixel_mean_luminance(p: &Pixel) -> f64 {
    if p.samples == 0 {
        return 0.0;
    }
    let inv = 1.0 / p.samples as f64;
    LUM_R * p.color[0] as f64 * inv + LUM_G * p.color[1] as f64 * inv + LUM_B * p.color[2] as f64 * inv
}

/// Variance of the pixel's mean luminance estimate. `INFINITY` when unknown.
///
/// [`Pixel::lum_sq`] stores Welford's `M₂ = Σ(x − μ)²`. The variance of the
/// mean is then `M₂ / n²`, which stays accurate when `μ` is large and the
/// spread is tiny — the case where `E[x²] − μ²` cancels in f32.
#[inline]
pub(crate) fn pixel_mean_variance(p: &Pixel) -> f32 {
    if p.samples < 2 {
        return f32::INFINITY;
    }
    let n = p.samples as f64;
    let m2 = p.lum_sq as f64;
    if !m2.is_finite() {
        return f32::INFINITY;
    }
    let var = (m2.max(0.0) / (n * n)) as f32;
    if !var.is_finite() {
        return f32::INFINITY;
    }
    var
}

/// Convert Welford `M₂` to the legacy `Σx²` used on disk in checkpoints.
#[inline]
pub(crate) fn sum_sq_from_m2(p: &Pixel) -> f32 {
    if p.samples == 0 {
        return 0.0;
    }
    let n = p.samples as f64;
    let mean = pixel_mean_luminance(p);
    let sum_sq = p.lum_sq as f64 + n * mean * mean;
    if !sum_sq.is_finite() {
        return f32::INFINITY;
    }
    sum_sq.max(0.0) as f32
}

/// Convert a checkpoint/`Σx²` value into Welford `M₂`.
#[inline]
pub(crate) fn m2_from_sum_sq(color: [f32; 3], samples: u32, sum_sq: f32) -> f32 {
    if samples == 0 {
        return 0.0;
    }
    if !sum_sq.is_finite() {
        return f32::INFINITY;
    }
    let n = samples as f64;
    let mean = LUM_R * color[0] as f64 / n
        + LUM_G * color[1] as f64 / n
        + LUM_B * color[2] as f64 / n;
    let m2 = sum_sq as f64 - n * mean * mean;
    if !m2.is_finite() {
        return f32::INFINITY;
    }
    m2.max(0.0) as f32
}

/// One pixel's accumulated state. Kept as an array-of-structs so a row is one
/// contiguous slice — which is what lets the renderer hand disjoint rows to
/// different threads without any locking at all.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Pixel {
    /// Summed radiance.
    pub color: [f32; 3],
    /// Summed alpha — the fraction of samples that hit something (or hit an
    /// opaque background).
    pub alpha: f32,
    /// Summed first-hit base colour, unlit.
    pub albedo: [f32; 3],
    /// Summed first-hit shading normal, world space.
    pub normal: [f32; 3],
    /// Summed distance to the first hit. Misses contribute nothing and are
    /// counted separately.
    pub depth: f32,
    /// How many samples contributed a depth (i.e. actually hit geometry).
    pub depth_samples: u32,
    /// Samples this pixel has taken. Per pixel rather than per film, because
    /// adaptive sampling stops pixels at different times — and resolving with a
    /// single global count would then scale most of the image wrongly.
    pub samples: u32,
    /// Welford `M₂ = Σ(x − μ)²` for sample luminances. Used with the running
    /// colour sum to estimate the variance of the mean without the f32
    /// cancellation of `E[x²] − μ²` on bright pixels.
    ///
    /// Checkpoints still serialise the legacy `Σx²` form; see
    /// [`sum_sq_from_m2`] / [`m2_from_sum_sq`].
    pub lum_sq: f32,
}

impl Pixel {
    /// Fold one path's result in.
    #[inline]
    pub fn add(&mut self, radiance: Vector3, alpha: f32) {
        let lum = luminance_sample(radiance);
        let n_old = self.samples as f32;
        let n_new = n_old + 1.0;
        // Welford one-pass update before the colour sum changes the mean.
        let mean_old = if self.samples == 0 {
            0.0
        } else {
            pixel_mean_luminance(self) as f32
        };
        let mut delta = lum - mean_old;
        // Firefly paths should not dominate M₂ — cap only when the jump is
        // absurd relative to the running spread (or vs the mean when spread=0).
        if self.samples >= 2 {
            let m2 = self.lum_sq.max(0.0);
            let std_sample = if m2 > 0.0 {
                (m2 / (n_old - 1.0).max(1.0)).sqrt()
            } else {
                0.0
            };
            let cap = if std_sample > 1e-6 {
                (std_sample * 4.0).max(mean_old.abs() * 0.25 + 1e-4)
            } else {
                mean_old.abs().max(1e-3) * 8.0 + 1e-3
            };
            if cap.is_finite() && delta.abs() > cap {
                delta = delta.clamp(-cap, cap);
            }
        }
        let lum_eff = mean_old + delta;
        let mean_new = mean_old + delta / n_new;
        self.lum_sq += delta * (lum_eff - mean_new);

        self.color[0] += radiance.x;
        self.color[1] += radiance.y;
        self.color[2] += radiance.z;
        self.alpha += alpha;
        self.samples += 1;
    }

    /// Fold another pixel's samples in (Chan / parallel Welford merge).
    #[inline]
    pub fn merge_pixel(&mut self, other: &Pixel) {
        if other.samples == 0 {
            return;
        }
        if self.samples == 0 {
            *self = *other;
            return;
        }
        let n_a = self.samples as f64;
        let n_b = other.samples as f64;
        let n = n_a + n_b;
        let mean_a = pixel_mean_luminance(self);
        let mean_b = pixel_mean_luminance(other);
        let delta = mean_b - mean_a;
        let m2 = self.lum_sq as f64 + other.lum_sq as f64 + delta * delta * n_a * n_b / n;
        for c in 0..3 {
            self.color[c] += other.color[c];
            self.albedo[c] += other.albedo[c];
            self.normal[c] += other.normal[c];
        }
        self.alpha += other.alpha;
        self.depth += other.depth;
        self.depth_samples += other.depth_samples;
        self.samples += other.samples;
        self.lum_sq = if m2.is_finite() {
            m2.max(0.0) as f32
        } else {
            f32::INFINITY
        };
    }

    /// Standard error of this pixel's mean, relative to the mean itself.
    ///
    /// This is the quantity adaptive sampling thresholds on. It falls as
    /// `1/sqrt(n)`, so halving it costs four times the samples — which is
    /// exactly why spending samples only where the number is still high is
    /// worth the bookkeeping.
    ///
    /// The denominator has a floor: a pixel that is nearly black has nearly no
    /// absolute error to remove, and dividing by its mean would keep it
    /// sampling forever chasing a relative target it can never meet.
    #[inline]
    pub fn relative_error(&self) -> f32 {
        let var = pixel_mean_variance(self);
        if var.is_infinite() {
            return f32::INFINITY;
        }
        if var <= 0.0 {
            return 0.0;
        }
        let mean = pixel_mean_luminance(self).abs().max(1e-3) as f32;
        var.sqrt() / mean
    }

    /// Whether this pixel has reached `threshold` relative error, having taken
    /// at least `min_samples`. A threshold of 0 means "never stop".
    #[inline]
    pub fn is_converged(&self, threshold: f32, min_samples: u32) -> bool {
        threshold > 0.0 && self.samples >= min_samples.max(2) && self.relative_error() < threshold
    }

    /// Fold in the first-hit auxiliaries. Only the primary hit contributes, so
    /// these stay noise-free however deep the paths go.
    #[inline]
    pub fn add_aux(&mut self, albedo: Vector3, normal: Vector3, depth: f32) {
        self.albedo[0] += albedo.x;
        self.albedo[1] += albedo.y;
        self.albedo[2] += albedo.z;
        self.normal[0] += normal.x;
        self.normal[1] += normal.y;
        self.normal[2] += normal.z;
        if depth.is_finite() {
            self.depth += depth;
            self.depth_samples += 1;
        }
    }
}

/// An accumulating frame buffer.
#[derive(Debug, Clone)]
pub struct Film {
    width: u32,
    height: u32,
    pixels: Vec<Pixel>,
    samples: u32,
}

impl Film {
    pub fn new(width: u32, height: u32) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        Self {
            width,
            height,
            pixels: vec![Pixel::default(); (width as usize) * (height as usize)],
            samples: 0,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Samples *issued* per pixel so far.
    ///
    /// With adaptive sampling on, a pixel that converged early will have taken
    /// fewer — see [`Self::sample_range`]. Resolving always divides by each
    /// pixel's own count, so this is a budget figure, not a divisor.
    pub fn samples(&self) -> u32 {
        self.samples
    }

    /// Discard everything accumulated. Call this when the scene or camera
    /// moves — the film has no way to know that on its own.
    pub fn clear(&mut self) {
        self.pixels.fill(Pixel::default());
        self.samples = 0;
    }

    /// Record that `n` more samples per pixel have been added.
    pub fn advance(&mut self, n: u32) {
        self.samples += n;
    }

    /// Replace the global sample counter — used when restoring a checkpoint.
    pub fn set_samples(&mut self, samples: u32) {
        self.samples = samples;
    }

    /// Share of pixels that met the adaptive convergence threshold.
    pub fn converged_fraction(&self, threshold: f32, min_samples: u32) -> f32 {
        if threshold <= 0.0 || self.pixels.is_empty() {
            return 0.0;
        }
        let converged = self
            .pixels
            .iter()
            .filter(|p| p.is_converged(threshold, min_samples))
            .count();
        converged as f32 / self.pixels.len() as f32
    }

    /// Rows as disjoint mutable slices, for handing to worker threads.
    pub fn rows_mut(&mut self) -> impl Iterator<Item = (usize, &mut [Pixel])> {
        let w = self.width as usize;
        self.pixels.chunks_mut(w).enumerate()
    }

    /// The backing store, one row after another.
    pub fn pixels(&self) -> &[Pixel] {
        &self.pixels
    }

    pub fn pixels_mut(&mut self) -> &mut [Pixel] {
        &mut self.pixels
    }

    /// Add another film's samples into this one — the reduction step when work
    /// was split across processes or devices rather than rows.
    pub fn merge(&mut self, other: &Film) {
        if other.width != self.width || other.height != self.height {
            return;
        }
        for (a, b) in self.pixels.iter_mut().zip(other.pixels.iter()) {
            a.merge_pixel(b);
        }
        self.samples += other.samples;
    }

    /// Mean radiance and alpha, linear and untone-mapped. Four floats a pixel,
    /// row-major from the top-left — the layout an EXR writer or a compositor
    /// wants.
    pub fn resolve_hdr(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.pixels.len() * 4);
        for p in &self.pixels {
            let inv = inv_samples(p);
            out.push(p.color[0] * inv);
            out.push(p.color[1] * inv);
            out.push(p.color[2] * inv);
            out.push((p.alpha * inv).clamp(0.0, 1.0));
        }
        out
    }

    /// Per-pixel variance of the mean, in luminance.
    ///
    /// This is what the denoiser should threshold on: it is the *measured*
    /// noise level of each pixel, so a converged pixel is barely filtered and a
    /// noisy one is filtered hard, with no global guess about how converged the
    /// image is. Pixels with too few samples to estimate it report `INFINITY`,
    /// which reads as "assume it is noisy".
    /// Variance of each pixel's mean.
    ///
    /// `INFINITY` where a pixel has fewer than two samples: with one sample the
    /// spread is genuinely unknown, and a zero there would read as "converged"
    /// — the opposite of the truth. Never NaN.
    ///
    /// Every consumer has to handle the infinity. `denoise` filters
    /// freely when it sees one; anything that *compresses* the value must map
    /// it rather than divide by it, because `inf / (1 + inf)` is NaN and a
    /// single NaN plane propagates silently through a convolution stack.
    pub fn resolve_variance(&self) -> Vec<f32> {
        self.pixels.iter().map(pixel_mean_variance).collect()
    }

    /// Per-pixel relative standard error — the adaptive sampler's own view of
    /// the image, and a useful thing to look at when a render will not settle.
    pub fn resolve_error(&self) -> Vec<f32> {
        self.pixels.iter().map(|p| p.relative_error()).collect()
    }

    /// Samples actually taken, per pixel. Equal to [`Self::samples`] everywhere
    /// unless adaptive sampling stopped some pixels early.
    pub fn resolve_sample_counts(&self) -> Vec<u32> {
        self.pixels.iter().map(|p| p.samples).collect()
    }

    /// Fewest and most samples any pixel took.
    pub fn sample_range(&self) -> (u32, u32) {
        self.pixels.iter().fold((u32::MAX, 0), |(lo, hi), p| {
            (lo.min(p.samples), hi.max(p.samples))
        })
    }

    /// Mean albedo, for the denoiser.
    pub fn resolve_albedo(&self) -> Vec<[f32; 3]> {
        self.pixels
            .iter()
            .map(|p| {
                let inv = inv_samples(p);
                [p.albedo[0] * inv, p.albedo[1] * inv, p.albedo[2] * inv]
            })
            .collect()
    }

    /// Mean normal, renormalised. Averaging unit vectors shortens them wherever
    /// samples disagree, which is exactly the edge signal the denoiser wants —
    /// but it needs the direction back as a unit vector.
    pub fn resolve_normal(&self) -> Vec<[f32; 3]> {
        let mut out = Vec::with_capacity(self.pixels.len());
        for p in &self.pixels {
            let inv = if p.samples > 0 {
                1.0 / p.samples as f32
            } else {
                1.0
            };
            let v = Vector3::new(
                p.normal[0] * inv,
                p.normal[1] * inv,
                p.normal[2] * inv,
            );
            let v = if v.length_sq() > 1e-12 {
                v.normalize()
            } else {
                Vector3::ZERO
            };
            out.push([v.x, v.y, v.z]);
        }
        out
    }

    /// Mean first-hit distance. Pixels that never hit anything report
    /// `INFINITY`.
    pub fn resolve_depth(&self) -> Vec<f32> {
        self.pixels
            .iter()
            .map(|p| {
                if p.depth_samples == 0 {
                    f32::INFINITY
                } else {
                    p.depth / p.depth_samples as f32
                }
            })
            .collect()
    }

    /// Tone-map and encode to tightly-packed sRGB RGBA8 — the same layout
    /// [`crate::encode_png`] and
    /// [`HeadlessRenderer::render_to_rgba`](crate::HeadlessRenderer::render_to_rgba)
    /// use, so a path-traced frame drops into any pipeline built around the
    /// raster one.
    pub fn resolve_rgba8(&self, settings: &RaytraceSettings) -> Vec<u8> {
        match settings.aov {
            Aov::Beauty => self.beauty_rgba8(&self.resolve_hdr(), settings),
            Aov::Albedo => {
                let a = self.resolve_albedo();
                let mut out = Vec::with_capacity(a.len() * 4);
                for c in a {
                    out.extend_from_slice(&[
                        encode_srgb(c[0]),
                        encode_srgb(c[1]),
                        encode_srgb(c[2]),
                        255,
                    ]);
                }
                out
            }
            Aov::Normal => {
                let n = self.resolve_normal();
                let mut out = Vec::with_capacity(n.len() * 4);
                for v in n {
                    // Signed direction into an unsigned byte.
                    let enc = |x: f32| ((x * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    out.extend_from_slice(&[enc(v[0]), enc(v[1]), enc(v[2]), 255]);
                }
                out
            }
            Aov::Depth => {
                let d = self.resolve_depth();
                let (near, far) = d
                    .iter()
                    .filter(|v| v.is_finite())
                    .fold((f32::INFINITY, 0.0f32), |(lo, hi), &v| {
                        (lo.min(v), hi.max(v))
                    });
                let span = (far - near).max(1e-6);
                let mut out = Vec::with_capacity(d.len() * 4);
                for v in d {
                    // Near is white, far is black, a miss is black.
                    let t = if v.is_finite() {
                        1.0 - ((v - near) / span).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let b = (t * 255.0 + 0.5) as u8;
                    out.extend_from_slice(&[b, b, b, 255]);
                }
                out
            }
        }
    }

    /// Encode already-resolved linear RGBA — used both by
    /// [`Self::resolve_rgba8`] and by the denoiser, which needs to tone-map its
    /// own output rather than the raw film.
    pub fn beauty_rgba8(&self, hdr: &[f32], settings: &RaytraceSettings) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixels.len() * 4);
        for px in hdr.chunks_exact(4) {
            let c = tone_map(
                Vector3::new(px[0], px[1], px[2]) * settings.exposure,
                settings.tone_mapping,
            );
            out.extend_from_slice(&[
                encode_srgb(c.x),
                encode_srgb(c.y),
                encode_srgb(c.z),
                (px[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            ]);
        }
        out
    }
}

/// Reciprocal of a pixel's own sample count, or 0 if it has none.
#[inline]
fn inv_samples(p: &Pixel) -> f32 {
    if p.samples == 0 {
        0.0
    } else {
        1.0 / p.samples as f32
    }
}

/// Apply a tone curve to linear radiance. Mirrors the WGSL in
/// [`crate::renderer`], so the two renderers agree on what an over-bright
/// highlight becomes.
pub fn tone_map(c: Vector3, mode: ToneMapping) -> Vector3 {
    match mode {
        ToneMapping::None => Vector3::new(
            c.x.clamp(0.0, 1.0),
            c.y.clamp(0.0, 1.0),
            c.z.clamp(0.0, 1.0),
        ),
        ToneMapping::Linear => Vector3::new(
            c.x.clamp(0.0, 1.0),
            c.y.clamp(0.0, 1.0),
            c.z.clamp(0.0, 1.0),
        ),
        ToneMapping::AcesFilmic => {
            // Narkowicz's fit to the ACES RRT+ODT: a rational curve that keeps
            // hue as it rolls off, which is why bright metal still reads as
            // metal instead of turning white.
            let f = |x: f32| {
                let (a, b, cc, d, e) = (2.51f32, 0.03f32, 2.43f32, 0.59f32, 0.14f32);
                ((x * (a * x + b)) / (x * (cc * x + d) + e)).clamp(0.0, 1.0)
            };
            Vector3::new(f(c.x), f(c.y), f(c.z))
        }
    }
}

/// Linear → sRGB, IEC 61966-2-1, then to a byte.
pub fn encode_srgb(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The variance contract: infinite where unknown, never NaN. A NaN here
    /// reaches a denoiser's input plane and takes the whole frame with it.
    #[test]
    fn variance_is_infinite_when_unknown_and_never_nan() {
        let mut film = Film::new(2, 2);
        // Nothing accumulated: every pixel is unknown, not converged.
        for v in film.resolve_variance() {
            assert!(v.is_infinite() && v > 0.0, "empty film gave {v}");
        }

        // A firefly: a sample large enough that its square overflows f32 must
        // not turn the estimate into NaN. `lum_sq` goes infinite, and
        // `inf - mean*mean` is what would produce one.
        film.pixels_mut()[0].add(Vector3::new(1e30, 1e30, 1e30), 1.0);
        film.pixels_mut()[0].add(Vector3::ZERO, 1.0);
        for v in film.resolve_variance() {
            assert!(!v.is_nan(), "variance was NaN");
        }
    }

    /// Large means plus tiny spread must not collapse to zero variance in f32.
    #[test]
    fn variance_stays_stable_at_high_dynamic_range() {
        let mut film = Film::new(1, 1);
        let p = film.pixels_mut();
        // Bright mean with a tiny spread — the kind of cancellation f32 variance
        // arithmetic loses when `(sum/n)²` ≈ `sum(x²)/n`.
        for _ in 0..99 {
            p[0].add(Vector3::new(1000.0, 1000.0, 1000.0), 1.0);
        }
        p[0].add(Vector3::new(1010.0, 1010.0, 1010.0), 1.0);
        let v = film.resolve_variance()[0];
        assert!(v.is_finite() && v > 0.0, "variance collapsed to {v}");
        assert!(film.pixels()[0].relative_error().is_finite());
    }

    /// Welford M₂ matches the classical second-moment formula on mild data,
    /// and checkpoint I/O still speaks Σx².
    #[test]
    fn fireflies_do_not_inflate_variance() {
        let mut film = Film::new(1, 1);
        for _ in 0..32 {
            film.pixels_mut()[0].add(Vector3::new(1.0, 1.0, 1.0), 1.0);
        }
        // One absurd path — should not dominate M₂.
        film.pixels_mut()[0].add(Vector3::new(1e4, 1e4, 1e4), 1.0);
        let v = film.resolve_variance()[0];
        assert!(
            v.is_finite() && v < 0.5,
            "firefly inflated variance to {v}"
        );
    }

    /// Adaptive stop must not get stuck open because one capped firefly landed.
    #[test]
    fn firefly_does_not_block_adaptive_convergence() {
        let mut p = Pixel::default();
        for _ in 0..64 {
            p.add(Vector3::new(1.0, 1.0, 1.0), 1.0);
        }
        assert!(
            p.is_converged(0.02, 16),
            "quiet pixel not converged: err={}",
            p.relative_error()
        );
        p.add(Vector3::new(1e5, 1e5, 1e5), 1.0);
        assert!(
            p.is_converged(0.05, 16),
            "firefly kept pixel open: err={}",
            p.relative_error()
        );
    }

    /// Legitimate high-variance samples must still keep a pixel open — the
    /// firefly cap is not a free pass to stop early on mirrors / caustics.
    #[test]
    fn high_variance_still_blocks_convergence() {
        let mut p = Pixel::default();
        for i in 0..32u32 {
            let v = if i % 2 == 0 { 0.2 } else { 2.0 };
            p.add(Vector3::new(v, v, v), 1.0);
        }
        assert!(
            !p.is_converged(0.02, 8),
            "bimodal pixel converged early: err={}",
            p.relative_error()
        );
    }

    #[test]
    fn welford_matches_second_moment_on_mild_data() {
        let mut film = Film::new(1, 1);
        for i in 0..16u32 {
            let v = 0.5 + (i as f32) * 0.01;
            film.pixels_mut()[0].add(Vector3::new(v, v, v), 1.0);
        }
        let p = &film.pixels()[0];
        let sum_sq = sum_sq_from_m2(p);
        let n = p.samples as f64;
        let mean = pixel_mean_luminance(p);
        let classic = ((sum_sq as f64 / n - mean * mean).max(0.0) / n) as f32;
        let welford = pixel_mean_variance(p);
        assert!(
            (classic - welford).abs() < 1e-5,
            "classic {classic} vs welford {welford}"
        );
        let round = m2_from_sum_sq(p.color, p.samples, sum_sq);
        assert!((round - p.lum_sq).abs() < 1e-4);
    }

    #[test]
    fn merge_combines_welford_m2() {
        let mut a = Film::new(1, 1);
        let mut b = Film::new(1, 1);
        for _ in 0..8 {
            a.pixels_mut()[0].add(Vector3::new(1.0, 1.0, 1.0), 1.0);
            b.pixels_mut()[0].add(Vector3::new(2.0, 2.0, 2.0), 1.0);
        }
        a.advance(8);
        b.advance(8);
        let mut both = Film::new(1, 1);
        for _ in 0..8 {
            both.pixels_mut()[0].add(Vector3::new(1.0, 1.0, 1.0), 1.0);
        }
        for _ in 0..8 {
            both.pixels_mut()[0].add(Vector3::new(2.0, 2.0, 2.0), 1.0);
        }
        a.merge(&b);
        let va = a.resolve_variance()[0];
        let vb = both.resolve_variance()[0];
        assert!(
            (va - vb).abs() < 1e-5,
            "merged {va} vs sequential {vb}"
        );
    }

    #[test]
    fn accumulation_averages() {
        let mut film = Film::new(2, 2);
        for i in 0..4u32 {
            for p in film.pixels_mut() {
                p.add(Vector3::new(i as f32, 0.0, 0.0), 1.0);
            }
            film.advance(1);
        }
        let hdr = film.resolve_hdr();
        // Mean of 0,1,2,3 is 1.5.
        assert!((hdr[0] - 1.5).abs() < 1e-6, "{}", hdr[0]);
        assert!((hdr[3] - 1.0).abs() < 1e-6);
        assert_eq!(film.samples(), 4);
    }

    #[test]
    fn an_empty_film_resolves_to_black_rather_than_nan() {
        let film = Film::new(3, 3);
        let hdr = film.resolve_hdr();
        assert!(hdr.iter().all(|v| v.is_finite() && *v == 0.0));
        let rgba = film.resolve_rgba8(&RaytraceSettings::default());
        assert_eq!(rgba.len(), 9 * 4);
    }

    #[test]
    fn clear_resets_everything() {
        let mut film = Film::new(2, 1);
        film.pixels_mut()[0].add(Vector3::new(5.0, 5.0, 5.0), 1.0);
        film.advance(1);
        film.clear();
        assert_eq!(film.samples(), 0);
        assert_eq!(film.resolve_hdr()[0], 0.0);
    }

    #[test]
    fn merge_sums_two_partial_renders() {
        let mut a = Film::new(2, 1);
        let mut b = Film::new(2, 1);
        for p in a.pixels_mut() {
            p.add(Vector3::new(1.0, 0.0, 0.0), 1.0);
        }
        a.advance(1);
        for p in b.pixels_mut() {
            p.add(Vector3::new(3.0, 0.0, 0.0), 1.0);
        }
        b.advance(1);
        a.merge(&b);
        assert_eq!(a.samples(), 2);
        assert!((a.resolve_hdr()[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn merge_rejects_a_mismatched_film() {
        let mut a = Film::new(2, 2);
        let b = Film::new(4, 4);
        a.merge(&b);
        assert_eq!(a.samples(), 0);
    }

    #[test]
    fn rows_are_disjoint_and_cover_the_film() {
        let mut film = Film::new(5, 3);
        let mut count = 0;
        for (y, row) in film.rows_mut() {
            assert_eq!(row.len(), 5);
            assert!(y < 3);
            count += row.len();
        }
        assert_eq!(count, 15);
    }

    #[test]
    fn srgb_encoding_hits_the_known_anchors() {
        assert_eq!(encode_srgb(0.0), 0);
        assert_eq!(encode_srgb(1.0), 255);
        // Linear 0.5 is 188 in 8-bit sRGB.
        assert_eq!(encode_srgb(0.5), 188);
        // Out-of-range input clamps rather than wrapping.
        assert_eq!(encode_srgb(-1.0), 0);
        assert_eq!(encode_srgb(50.0), 255);
    }

    #[test]
    fn aces_rolls_off_instead_of_clipping() {
        let a = tone_map(Vector3::new(1.0, 1.0, 1.0), ToneMapping::AcesFilmic);
        let b = tone_map(Vector3::new(3.0, 3.0, 3.0), ToneMapping::AcesFilmic);
        // Input 1.0 maps well below white, which is what leaves headroom for
        // the highlights above it.
        assert!(a.x < 0.9, "{a:?}");
        assert!(b.x < 1.0 && b.x > a.x, "{a:?} -> {b:?}");
        // Plain clamping does not.
        let c = tone_map(Vector3::new(8.0, 8.0, 8.0), ToneMapping::None);
        assert_eq!(c.x, 1.0);
    }

    #[test]
    fn normals_are_renormalised_on_resolve() {
        let mut film = Film::new(1, 1);
        for _ in 0..4 {
            film.pixels_mut()[0].add_aux(
                Vector3::new(0.5, 0.5, 0.5),
                Vector3::new(0.0, 0.0, 1.0),
                2.0,
            );
            film.advance(1);
        }
        let n = film.resolve_normal();
        assert!((n[0][2] - 1.0).abs() < 1e-6, "{:?}", n[0]);
        let d = film.resolve_depth();
        assert!((d[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn pixels_that_never_hit_report_infinite_depth() {
        let mut film = Film::new(1, 1);
        film.pixels_mut()[0].add_aux(Vector3::ZERO, Vector3::ZERO, f32::INFINITY);
        film.advance(1);
        assert!(film.resolve_depth()[0].is_infinite());
        // And the depth AOV renders them black rather than NaN.
        let s = RaytraceSettings {
            aov: Aov::Depth,
            ..Default::default()
        };
        let rgba = film.resolve_rgba8(&s);
        assert_eq!(&rgba[..3], &[0, 0, 0]);
    }
}
