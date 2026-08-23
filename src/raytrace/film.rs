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

/// One pixel's accumulated state. Kept as an array-of-structs so a row is one
/// contiguous slice — which is what lets the renderer hand disjoint rows to
/// different threads without any locking at all.
#[derive(Debug, Clone, Copy, Default)]
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
    /// Summed *squared* luminance. Together with the running mean this gives
    /// the sample variance, which is how a pixel knows whether it has
    /// converged.
    pub lum_sq: f32,
}

impl Pixel {
    /// Fold one path's result in.
    #[inline]
    pub fn add(&mut self, radiance: Vector3, alpha: f32) {
        self.color[0] += radiance.x;
        self.color[1] += radiance.y;
        self.color[2] += radiance.z;
        self.alpha += alpha;
        let lum = 0.2126 * radiance.x + 0.7152 * radiance.y + 0.0722 * radiance.z;
        self.lum_sq += lum * lum;
        self.samples += 1;
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
        if self.samples < 2 {
            return f32::INFINITY;
        }
        let n = self.samples as f32;
        let mean = (0.2126 * self.color[0] + 0.7152 * self.color[1] + 0.0722 * self.color[2]) / n;
        let variance = (self.lum_sq / n - mean * mean).max(0.0);
        if variance <= 0.0 {
            return 0.0;
        }
        (variance / n).sqrt() / mean.max(1e-3)
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
            for c in 0..3 {
                a.color[c] += b.color[c];
                a.albedo[c] += b.albedo[c];
                a.normal[c] += b.normal[c];
            }
            a.alpha += b.alpha;
            a.depth += b.depth;
            a.depth_samples += b.depth_samples;
            a.samples += b.samples;
            a.lum_sq += b.lum_sq;
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
        self.pixels
            .iter()
            .map(|p| {
                if p.samples < 2 {
                    return f32::INFINITY;
                }
                let n = p.samples as f32;
                let mean = (0.2126 * p.color[0] + 0.7152 * p.color[1] + 0.0722 * p.color[2]) / n;
                let variance = (p.lum_sq / n - mean * mean).max(0.0);
                variance / n
            })
            .collect()
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
            let v = Vector3::new(p.normal[0], p.normal[1], p.normal[2]);
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
