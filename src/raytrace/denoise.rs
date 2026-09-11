//! Edge-avoiding À-Trous wavelet denoising (Dammertz et al. 2010), guided by
//! the albedo, normal and depth channels the film already carries.
//!
//! Monte-Carlo noise is zero-mean: neighbouring pixels are each wrong by a
//! different amount and their average is close to right. A blur exploits that
//! and destroys everything else, so this one is told where the edges are by
//! channels that have no noise in them — the first-hit base colour, normal and
//! distance are the same for every sample of a pixel. Where those agree, the
//! filter averages hard; where they disagree, it barely averages at all.
//!
//! The À-Trous part is what makes it cheap: instead of one wide filter, five
//! passes of the same 5×5 kernel with the taps spaced 1, 2, 4, 8 and 16 pixels
//! apart. That reaches a 65-pixel radius for the cost of 125 taps a pixel
//! rather than 4225.
//!
//! It is a *post-process*, not part of the estimator. The result is biased —
//! it is no longer the integral, it is a smoothed version of it — which is why
//! [`RaytraceSettings::denoise`](super::settings::RaytraceSettings::denoise) is
//! something you turn off for a reference image.

use crate::math::Vector3;

/// How aggressively each guide channel rejects a neighbour. Larger is more
/// permissive — a bigger difference is still considered "the same surface".
#[derive(Debug, Clone, Copy)]
pub struct DenoiseParams {
    /// Passes of the À-Trous kernel. Each doubles the reach.
    pub iterations: u32,
    /// Colour similarity at the first pass, in units of the pixel's own
    /// standard deviation.
    ///
    /// The threshold is *relative to the measured noise*: a difference of one
    /// sigma between two pixels is what noise alone would produce, so anything
    /// much larger is real detail and must not be averaged away. Because the
    /// noise level is known per pixel rather than guessed, a converged region
    /// is barely touched and a noisy one is filtered hard within the same
    /// image — which a single global threshold cannot do.
    pub sigma_color: f32,
    /// Normal similarity, in radians-ish (it compares dot products).
    pub sigma_normal: f32,
    /// Depth similarity, as a fraction of the scene's scale.
    pub sigma_depth: f32,
    /// Albedo similarity — this is what preserves texture detail that the
    /// colour channel is too noisy to see.
    pub sigma_albedo: f32,
    /// Do not trust a per-pixel variance estimate until at least this many
    /// samples have landed — filter more aggressively below it.
    pub min_samples: u32,
}

impl Default for DenoiseParams {
    /// Fitted, not guessed.
    ///
    /// These came out of [`Self::fit`] against pairs the renderer generated
    /// itself — an area light, a mirror over a chequered floor, and glass
    /// beside rough metal, judged on a held-out interior. The hand-picked
    /// values they replaced were worse on every one of those scenes, and on the
    /// area light they were worse than not denoising at all. `sigma_normal` in
    /// particular was nearly six times too loose, which no amount of looking at
    /// one test image was going to reveal.
    ///
    /// Re-run `cargo run --release --example denoise_fit --features raytrace`
    /// after changing anything the guides depend on.
    fn default() -> Self {
        // Fitted 2026-08 (`denoise_fit`, REFERENCE=8192) for colour/depth under
        // log-lum weights + Welford guides. `sigma_albedo` kept at the prior
        // tight width: blowing it open (~23) scored the same on area lights and
        // only risks softening textured floors (see `examples/denoise_ab`).
        Self {
            iterations: 5,
            sigma_color: 1.297,
            sigma_normal: 0.566,
            sigma_depth: 0.0007,
            sigma_albedo: 0.252,
            min_samples: 4,
        }
    }
}

/// Relative standard error at which a pixel counts as fully converged, and the
/// filter leaves it alone. Matches the adaptive sampler's default target, which
/// is the level a render is deliberately taken to.
const CONVERGED_ERROR: f32 = 0.01;

/// The B3-spline kernel the À-Trous scheme is built on, as a 5×5 separable
/// outer product of `[1, 4, 6, 4, 1] / 16`.
const KERNEL: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// The noise-free channels the filter is steered by. All row-major, all the
/// same dimensions as the image.
#[derive(Debug, Clone, Copy)]
pub struct DenoiseGuides<'a> {
    /// First-hit base colour, 3 floats a pixel.
    pub albedo: &'a [[f32; 3]],
    /// First-hit shading normal, 3 floats a pixel.
    pub normal: &'a [[f32; 3]],
    /// First-hit distance, 1 float a pixel; `INFINITY` where nothing was hit.
    pub depth: &'a [f32],
    /// Variance of each pixel's mean — the measured noise level, from
    /// [`Film::resolve_variance`](super::Film::resolve_variance).
    pub variance: &'a [f32],
    /// Per-pixel sample counts when adaptive sampling stopped pixels early.
    /// Pixels below [`DenoiseParams::min_samples`] are filtered harder because
    /// their variance estimate is not yet trustworthy.
    pub sample_counts: Option<&'a [u32]>,
    /// The scene's diagonal, so the depth threshold means the same thing in a
    /// model measured in millimetres and one measured in kilometres.
    pub scene_scale: f32,
}

/// Denoise linear RGBA.
///
/// `color` is RGBA, 4 floats a pixel. Alpha is carried through untouched — it
/// is coverage, not radiance, and blurring it would soften the matte.
pub fn denoise(
    width: u32,
    height: u32,
    color: &[f32],
    guides: &DenoiseGuides<'_>,
    params: &DenoiseParams,
) -> Vec<f32> {
    let (albedo, normal, depth) = (guides.albedo, guides.normal, guides.depth);
    let variance = guides.variance;
    let sample_counts = guides.sample_counts;
    let scene_scale = guides.scene_scale;
    let (w, h) = (width as usize, height as usize);
    let n = w * h;
    if n == 0 || color.len() < n * 4 {
        return color.to_vec();
    }

    let mut src: Vec<[f32; 3]> = (0..n)
        .map(|i| [color[i * 4], color[i * 4 + 1], color[i * 4 + 2]])
        .collect();
    let mut dst = src.clone();

    let depth_scale = (params.sigma_depth * scene_scale.max(1e-3)).max(1e-4);
    let min_samples = params.min_samples.max(2);

    for pass in 0..params.iterations {
        let step = 1usize << pass;
        let sigma_color = params.sigma_color;

        filter_pass(
            w,
            h,
            step,
            sigma_color,
            params,
            depth_scale,
            min_samples,
            &src,
            &mut dst,
            albedo,
            normal,
            depth,
            variance,
            sample_counts,
        );
        std::mem::swap(&mut src, &mut dst);
    }

    let mut out = Vec::with_capacity(n * 4);
    for (i, c) in src.iter().enumerate() {
        out.push(c[0]);
        out.push(c[1]);
        out.push(c[2]);
        out.push(color[i * 4 + 3]);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn filter_pass(
    w: usize,
    h: usize,
    step: usize,
    sigma_color: f32,
    params: &DenoiseParams,
    depth_scale: f32,
    min_samples: u32,
    src: &[[f32; 3]],
    dst: &mut [[f32; 3]],
    albedo: &[[f32; 3]],
    normal: &[[f32; 3]],
    depth: &[f32],
    variance: &[f32],
    sample_counts: Option<&[u32]>,
) {
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    {
        use rayon::prelude::*;
        dst.par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, out) in row.iter_mut().enumerate() {
                    *out = filter_pixel(
                        x,
                        y,
                        w,
                        h,
                        step,
                        sigma_color,
                        params,
                        depth_scale,
                        min_samples,
                        src,
                        albedo,
                        normal,
                        depth,
                        variance,
                        sample_counts,
                    );
                }
            });
    }
    #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
    {
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                dst[i] = filter_pixel(
                    x,
                    y,
                    w,
                    h,
                    step,
                    sigma_color,
                    params,
                    depth_scale,
                    min_samples,
                    src,
                    albedo,
                    normal,
                    depth,
                    variance,
                    sample_counts,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn filter_pixel(
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    step: usize,
    sigma_color: f32,
    params: &DenoiseParams,
    depth_scale: f32,
    min_samples: u32,
    src: &[[f32; 3]],
    albedo: &[[f32; 3]],
    normal: &[[f32; 3]],
    depth: &[f32],
    variance: &[f32],
    sample_counts: Option<&[u32]>,
) -> [f32; 3] {
    let i = y * w + x;
    let c_i = src[i];
    let n_i = normal[i];
    let a_i = albedo[i];
    let d_i = depth[i];
    let v_i = variance.get(i).copied().unwrap_or(f32::INFINITY);

    let mut sum = [0.0f32; 3];
    let mut weight_sum = 0.0f32;

    for (ky, kv) in KERNEL.iter().enumerate() {
        let sy = y as isize + (ky as isize - 2) * step as isize;
        if sy < 0 || sy >= h as isize {
            continue;
        }
        for (kx, ku) in KERNEL.iter().enumerate() {
            let sx = x as isize + (kx as isize - 2) * step as isize;
            if sx < 0 || sx >= w as isize {
                continue;
            }
            let j = sy as usize * w + sx as usize;
            let c_j = src[j];

            let v_j = variance.get(j).copied().unwrap_or(f32::INFINITY);
            let mean_l = (luminance(c_i).abs() + luminance(c_j).abs()) * 0.5;
            let w_c = if v_i.is_finite() && v_j.is_finite() {
                let scale = (1.0 + mean_l).max(1e-3);
                let li = (1.0 + luminance(c_i).max(0.0)).ln();
                let lj = (1.0 + luminance(c_j).max(0.0)).ln();
                let dl_log = li - lj;
                let dl_lin = (luminance(c_i) - luminance(c_j)).abs() / scale;
                // The tighter of log-relative and linear-relative distance: HDR
                // walls keep log-lum; specular noise can still blur when linear
                // says neighbours agree.
                let dl = dl_log.abs().min(dl_lin);
                let noise_lin =
                    (v_i + v_j).max((mean_l * 0.01).powi(2).max(1e-12));
                let noise = noise_lin / (scale * scale);
                (-(dl * dl) / (sigma_color * sigma_color * noise).max(1e-9)).exp()
            } else {
                1.0
            };

            let dot =
                (n_i[0] * normal[j][0] + n_i[1] * normal[j][1] + n_i[2] * normal[j][2])
                    .clamp(-1.0, 1.0);
            let dn = (1.0 - dot).max(0.0);
            let w_n = (-dn / (params.sigma_normal * params.sigma_normal).max(1e-8)).exp();

            let w_d = if d_i.is_finite() && depth[j].is_finite() {
                let dd = (d_i - depth[j]).abs() / depth_scale;
                (-dd * dd).exp()
            } else if d_i.is_finite() == depth[j].is_finite() {
                1.0
            } else {
                0.0
            };

            let da = sq_dist(a_i, albedo[j]);
            let w_a = (-da / (params.sigma_albedo * params.sigma_albedo).max(1e-8)).exp();

            let weight = kv * ku * w_c * w_n * w_d * w_a;
            if weight <= 0.0 {
                continue;
            }
            sum[0] += c_j[0] * weight;
            sum[1] += c_j[1] * weight;
            sum[2] += c_j[2] * weight;
            weight_sum += weight;
        }
    }

    let filtered = if weight_sum > 1e-8 {
        [
            sum[0] / weight_sum,
            sum[1] / weight_sum,
            sum[2] / weight_sum,
        ]
    } else {
        c_i
    };

    let mut trust = if v_i.is_finite() {
        // Relative SE in linear space, floored — same units as adaptive sampling.
        let mean = luminance(c_i).abs().max(1e-3);
        (v_i.sqrt() / mean / CONVERGED_ERROR).min(1.0)
    } else {
        1.0
    };

    if let Some(counts) = sample_counts {
        let n = counts.get(i).copied().unwrap_or(0);
        if n < min_samples {
            let boost = 1.0 - (n as f32 / min_samples as f32);
            trust = trust.max(boost);
        }
    }

    blend_denoised(c_i, filtered, trust)
}

/// Blend filtered radiance toward the noisy pixel. Luminance is mixed in log
/// space; chroma linearly — so HDR highlights denoise without hue shifts.
fn blend_denoised(c_i: [f32; 3], filtered: [f32; 3], trust: f32) -> [f32; 3] {
    if trust <= 0.0 {
        return c_i;
    }
    if trust >= 1.0 - 1e-6 {
        return filtered;
    }
    let li = luminance(c_i).max(0.0);
    let lf = luminance(filtered).max(0.0);
    if li <= 1e-8 && lf <= 1e-8 {
        return [
            c_i[0] + (filtered[0] - c_i[0]) * trust,
            c_i[1] + (filtered[1] - c_i[1]) * trust,
            c_i[2] + (filtered[2] - c_i[2]) * trust,
        ];
    }
    let log_lo =
        (1.0 + li).ln() + trust * ((1.0 + lf).ln() - (1.0 + li).ln());
    let lo = log_lo.exp() - 1.0;
    let norm_i = chroma_dir(c_i, li);
    let norm_f = chroma_dir(filtered, lf);
    let norm = [
        norm_i[0] + trust * (norm_f[0] - norm_i[0]),
        norm_i[1] + trust * (norm_f[1] - norm_i[1]),
        norm_i[2] + trust * (norm_f[2] - norm_i[2]),
    ];
    let nl = luminance(norm).max(1e-8);
    let scale = lo / nl;
    [norm[0] * scale, norm[1] * scale, norm[2] * scale]
}

fn chroma_dir(c: [f32; 3], l: f32) -> [f32; 3] {
    if l > 1e-8 {
        [c[0] / l, c[1] / l, c[2] / l]
    } else {
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    }
}

/// One training example: a noisy render, the guides that came with it, and the
/// converged answer it should have been.
///
/// The renderer produces both halves itself — render a scene at 32 samples and
/// again at 2048 — so a training set costs nothing but time.
pub struct DenoiseExample<'a> {
    pub width: u32,
    pub height: u32,
    /// Linear RGBA, as [`Film::resolve_hdr`](super::Film::resolve_hdr) returns.
    pub noisy: &'a [f32],
    pub guides: DenoiseGuides<'a>,
    /// The converged render, same layout.
    pub reference: &'a [f32],
}

impl DenoiseExample<'_> {
    /// Relative mean squared error of `image` against this example's reference.
    ///
    /// `mean((a - b)² / (b² + 0.01))`, the metric the denoising literature
    /// settled on, and not plain RMS. Plain RMS in linear radiance is dominated
    /// by whichever pixels are brightest: fitting to it produces a filter that
    /// polishes a highlight and leaves visible grain across everything darker,
    /// which is the opposite of what the eye notices. Dividing by the
    /// reference's own magnitude asks how wrong each pixel is *relative to what
    /// it should be*, which is much closer to how an image reads — and makes
    /// scenes of different brightness comparable, so several can be averaged.
    ///
    /// The 0.01 floor stops near-black pixels, where the relative error is
    /// unbounded and meaningless, from deciding the whole fit.
    pub fn relative_rms(&self, image: &[f32]) -> f32 {
        let mut sum = 0.0f64;
        let mut n = 0usize;
        for (a, b) in image.chunks_exact(4).zip(self.reference.chunks_exact(4)) {
            for k in 0..3 {
                let d = (a[k] - b[k]) as f64;
                let r = b[k] as f64;
                sum += d * d / (r * r + 0.01);
                n += 1;
            }
        }
        if n == 0 {
            return 0.0;
        }
        (sum / n as f64).sqrt() as f32
    }

    fn error_with(&self, params: &DenoiseParams) -> f32 {
        let out = denoise(self.width, self.height, self.noisy, &self.guides, params);
        self.relative_rms(&out)
    }
}

impl DenoiseParams {
    /// Mean relative error these parameters achieve over a set of examples.
    pub fn error_over(&self, examples: &[DenoiseExample<'_>]) -> f32 {
        if examples.is_empty() {
            return 0.0;
        }
        examples.iter().map(|e| e.error_with(self)).sum::<f32>() / examples.len() as f32
    }

    /// Fit the filter's parameters to a set of examples.
    ///
    /// This is what "training a denoiser" amounts to for a filter of this
    /// shape. The model is the filter, its parameters are four widths and an
    /// iteration count, and the training data is pairs the renderer generates
    /// itself. There is no network and nothing to ship but five numbers.
    ///
    /// Coordinate descent in log space, rather than gradient descent: with four
    /// continuous parameters the derivative buys nothing a direct search does
    /// not already get, and a direct search needs no autodiff framework, no
    /// optional dependency and no second implementation of the filter to
    /// differentiate through. (A *kernel-predicting network* — a small CNN that
    /// emits per-pixel filter weights — is where gradients would start to earn
    /// their place, and that is a different piece of work with a different
    /// dependency footprint.)
    ///
    /// Deterministic: the same examples and starting point always give the same
    /// answer.
    pub fn fit(examples: &[DenoiseExample<'_>], start: DenoiseParams) -> (DenoiseParams, f32) {
        Self::fit_impl(examples, start, true)
    }

    /// As [`Self::fit`], but leaving `iterations` alone.
    ///
    /// The iteration count is not really a free parameter — it sets how far the
    /// filter can *reach*, and reach is what removes the broad, low-frequency
    /// blotches that a relative error metric barely notices and the eye
    /// immediately does. Left free, the fit reliably cuts it to two passes and
    /// scores well on an image that still visibly crawls. So the reach is a
    /// choice about the model, and this fits the widths given it.
    pub fn fit_widths(
        examples: &[DenoiseExample<'_>],
        start: DenoiseParams,
    ) -> (DenoiseParams, f32) {
        Self::fit_impl(examples, start, false)
    }

    fn fit_impl(
        examples: &[DenoiseExample<'_>],
        start: DenoiseParams,
        search_iterations: bool,
    ) -> (DenoiseParams, f32) {
        let mut best = start;
        let mut best_error = best.error_over(examples);
        if examples.is_empty() {
            return (best, best_error);
        }

        // Multiplicative steps: these parameters span orders of magnitude, and
        // an additive step that suits `sigma_depth` at 0.02 is meaningless for
        // `sigma_color` at 1.0.
        //
        // The iteration count is searched *inside* the loop rather than after
        // it, because it trades directly against the widths — fewer passes want
        // looser thresholds and vice versa. Optimising it once at the end lands
        // on whichever local optimum the starting point happened to sit in, and
        // the answer then depends on where you started, which is not a fit.
        let mut step = 2.0f32;
        for _ in 0..12 {
            let mut improved = false;

            for iterations in 1..=6u32 {
                if !search_iterations || iterations == best.iterations {
                    continue;
                }
                let mut trial = best;
                trial.iterations = iterations;
                let error = trial.error_over(examples);
                if error < best_error {
                    best = trial;
                    best_error = error;
                    improved = true;
                }
            }

            for axis in 0..4 {
                for factor in [step, 1.0 / step] {
                    let mut trial = best;
                    let field = match axis {
                        0 => &mut trial.sigma_color,
                        1 => &mut trial.sigma_normal,
                        2 => &mut trial.sigma_depth,
                        _ => &mut trial.sigma_albedo,
                    };
                    *field = (*field * factor).clamp(1e-4, 1e3);
                    let error = trial.error_over(examples);
                    if error < best_error {
                        best = trial;
                        best_error = error;
                        improved = true;
                    }
                }
            }

            if !improved {
                step = step.sqrt();
                if step < 1.02 {
                    break;
                }
            }
        }
        (best, best_error)
    }
}

fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn sq_dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = Vector3::new(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    d.length_sq()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raytrace::sampler::Rng;

    /// Colour, albedo, normal, depth, variance — what a test image needs.
    type TestImage = (Vec<f32>, Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<f32>, Vec<f32>);

    /// Build a test image: a flat left half and a flat right half, with noise
    /// added, plus matching guide channels.
    fn split_image(w: usize, h: usize, noise: f32) -> TestImage {
        let mut rng = Rng::new(1, 2);
        let mut color = Vec::with_capacity(w * h * 4);
        let mut albedo = Vec::with_capacity(w * h);
        let mut normal = Vec::with_capacity(w * h);
        let mut depth = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let left = x < w / 2;
                let base = if left { 0.2 } else { 0.8 };
                let n = (rng.next_f32() - 0.5) * 2.0 * noise;
                color.extend_from_slice(&[base + n, base + n, base + n, 1.0]);
                albedo.push([base, base, base]);
                normal.push(if left {
                    [0.0, 0.0, 1.0]
                } else {
                    [1.0, 0.0, 0.0]
                });
                // Both halves are the same distance away, so only the normal and
                // albedo separate them — which is what this image is testing.
                depth.push(5.0);
                let _ = y;
            }
        }
        // The noise was injected uniformly, so every pixel has the same
        // variance — which is what the filter is told.
        let variance = vec![noise * noise / 3.0; w * h];
        (color, albedo, normal, depth, variance)
    }

    fn spatial_variance(v: &[f32], w: usize, h: usize, x0: usize, x1: usize) -> f32 {
        let mut vals = Vec::new();
        for y in 1..h - 1 {
            for x in x0..x1 {
                vals.push(v[(y * w + x) * 4]);
            }
        }
        let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
        vals.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / vals.len() as f32
    }

    #[test]
    fn noise_is_reduced_in_a_flat_region() {
        let (w, h) = (48, 48);
        let (color, albedo, normal, depth, variance) = split_image(w, h, 0.25);
        let out = denoise(
            w as u32,
            h as u32,
            &color,
            &DenoiseGuides {
                albedo: &albedo,
                normal: &normal,
                depth: &depth,
                variance: &variance,
                sample_counts: None,
                scene_scale: 10.0,
            },
            &DenoiseParams::default(),
        );
        let before = spatial_variance(&color, w, h, 2, w / 2 - 2);
        let after = spatial_variance(&out, w, h, 2, w / 2 - 2);
        assert!(
            after < before * 0.35,
            "variance went {before} -> {after}; the filter is not working"
        );
    }

    #[test]
    fn the_edge_between_two_surfaces_survives() {
        let (w, h) = (48, 48);
        let (color, albedo, normal, depth, variance) = split_image(w, h, 0.1);
        let out = denoise(
            w as u32,
            h as u32,
            &color,
            &DenoiseGuides {
                albedo: &albedo,
                normal: &normal,
                depth: &depth,
                variance: &variance,
                sample_counts: None,
                scene_scale: 10.0,
            },
            &DenoiseParams::default(),
        );
        let row = h / 2;
        let left = out[(row * w + w / 2 - 1) * 4];
        let right = out[(row * w + w / 2) * 4];
        assert!(
            right - left > 0.4,
            "the 0.2/0.8 step was smoothed to {left} / {right}"
        );
    }

    #[test]
    fn alpha_passes_through_untouched() {
        let (w, h) = (8, 8);
        let mut color = vec![0.0f32; w * h * 4];
        for (i, px) in color.chunks_exact_mut(4).enumerate() {
            px[3] = (i % 2) as f32;
        }
        let albedo = vec![[0.0f32; 3]; w * h];
        let normal = vec![[0.0f32, 0.0, 1.0]; w * h];
        let depth = vec![1.0f32; w * h];
        let variance = vec![0.02f32; w * h];
        let out = denoise(
            w as u32,
            h as u32,
            &color,
            &DenoiseGuides {
                albedo: &albedo,
                normal: &normal,
                depth: &depth,
                variance: &variance,
                sample_counts: None,
                scene_scale: 1.0,
            },
            &DenoiseParams::default(),
        );
        for i in 0..w * h {
            assert_eq!(out[i * 4 + 3], (i % 2) as f32);
        }
    }

    #[test]
    fn background_and_foreground_do_not_bleed_into_each_other() {
        let (w, h) = (16, 16);
        let mut color = vec![0.0f32; w * h * 4];
        let albedo = vec![[0.5f32; 3]; w * h];
        let normal = vec![[0.0f32, 0.0, 1.0]; w * h];
        let mut depth = vec![f32::INFINITY; w * h];
        let variance = vec![0.02f32; w * h];
        // Left half is geometry at depth 3 and bright; right half is background.
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x < w / 2 {
                    depth[i] = 3.0;
                    color[i * 4] = 1.0;
                }
                color[i * 4 + 3] = 1.0;
            }
        }
        let out = denoise(
            w as u32,
            h as u32,
            &color,
            &DenoiseGuides {
                albedo: &albedo,
                normal: &normal,
                depth: &depth,
                variance: &variance,
                sample_counts: None,
                scene_scale: 10.0,
            },
            &DenoiseParams::default(),
        );
        // The first background column must still be black.
        let i = (h / 2 * w + w / 2) * 4;
        assert!(
            out[i] < 0.05,
            "background picked up {} from the object",
            out[i]
        );
    }

    #[test]
    fn blend_denoised_preserves_chroma_at_hdr() {
        let c = [1000.0, 1000.0, 1000.0];
        let f = [1000.0, 1100.0, 1000.0];
        // Partial trust: log-lum + linear-chroma beats a straight RGB lerp on hue.
        let out = blend_denoised(c, f, 0.5);
        let linear_g = c[1] + 0.5 * (f[1] - c[1]);
        assert!(
            (out[0] - out[1]).abs() < (c[0] - linear_g).abs(),
            "chroma drift {:?} vs linear green {linear_g}",
            out
        );
        let neutral = blend_denoised(c, [1000.0, 1000.0, 1000.0], 1.0);
        assert!((neutral[0] - neutral[1]).abs() < 1e-3);
    }

    #[test]
    fn zero_iterations_is_a_no_op() {
        let (w, h) = (8, 8);
        let (color, albedo, normal, depth, variance) = split_image(w, h, 0.2);
        let params = DenoiseParams {
            iterations: 0,
            ..Default::default()
        };
        let out = denoise(
            w as u32,
            h as u32,
            &color,
            &DenoiseGuides {
                albedo: &albedo,
                normal: &normal,
                depth: &depth,
                variance: &variance,
                sample_counts: None,
                scene_scale: 1.0,
            },
            &params,
        );
        assert_eq!(out, color);
    }

    #[test]
    fn low_sample_count_filters_harder() {
        let (w, h) = (16, 16);
        let (color, albedo, normal, depth, mut variance) = split_image(w, h, 0.4);
        variance[w / 2] = 1e-8;
        let counts = vec![1u32; w * h];
        let params = DenoiseParams {
            min_samples: 8,
            ..Default::default()
        };
        let out = denoise(
            w as u32,
            h as u32,
            &color,
            &DenoiseGuides {
                albedo: &albedo,
                normal: &normal,
                depth: &depth,
                variance: &variance,
                sample_counts: Some(&counts),
                scene_scale: 10.0,
            },
            &params,
        );
        let idx = (h / 2 * w + w / 2) * 4;
        let center = out[idx];
        let noisy = color[idx];
        assert!(
            (center - noisy).abs() > 0.05,
            "low-sample pixel should have been filtered: {noisy} -> {center}"
        );
    }

    #[test]
    fn an_empty_image_is_handled() {
        let out = denoise(
            0,
            0,
            &[],
            &DenoiseGuides {
                albedo: &[],
                normal: &[],
                depth: &[],
                variance: &[],
                sample_counts: None,
                scene_scale: 1.0,
            },
            &DenoiseParams::default(),
        );
        assert!(out.is_empty());
    }
}
