//! Importance sampling for the environment.
//!
//! A cosine-weighted guess is the right way to sample a *smooth* sky and the
//! wrong way to sample a real one. An HDRI of an outdoor scene puts most of its
//! energy into the sun — a disc a quarter of a degree across, about 6·10⁻⁵ of
//! the sphere — so a cosine-weighted direction finds it roughly one sample in
//! twenty thousand, and each one that does carries thousands of times the mean.
//! That is not slow convergence, it is an image of white speckles that does not
//! resolve at any sample count you would wait for.
//!
//! The fix is to build a distribution over the sky's own brightness and draw
//! from that: a piecewise-constant 2D density over an equirectangular
//! projection, sampled by two binary searches. The sun then gets the share of
//! the samples it has of the energy, and multiple importance sampling weighs
//! the result against the BSDF's own guess so neither strategy is trusted where
//! it is poor.
//!
//! The density is deliberately *coarse* — 256×128 over the whole sphere. It
//! does not need to resolve the sun, only to know which texel the sun is in;
//! the radiance itself is read from the full-resolution cube afterwards. What
//! it does need is to be non-zero wherever the cube is, which is why each texel
//! is built from a 4×4 supersample rather than one point: a sun that fell
//! between single samples would be a direction with radiance and no density,
//! and dividing by that density is how a renderer produces infinities.

use crate::math::Vector3;
use std::f32::consts::PI;

/// Width of the sampling grid, in texels. Height is half.
pub const ENV_WIDTH: usize = 256;
/// Height of the sampling grid.
pub const ENV_HEIGHT: usize = ENV_WIDTH / 2;
/// Samples per texel per axis when building the grid.
const SUPERSAMPLE: usize = 4;

/// A piecewise-constant density over the sphere, in equirectangular
/// coordinates.
///
/// The layout is flat rather than a vector of per-row distributions, because
/// the GPU backend uploads these arrays verbatim and a nested structure would
/// have to be flattened for it anyway.
#[derive(Debug, Clone)]
pub struct EnvDistribution {
    width: usize,
    height: usize,
    /// Luminance × sinθ per texel, row-major. The sine is the Jacobian of the
    /// equirectangular projection: without it the poles, which occupy almost no
    /// solid angle, would be sampled as heavily as the equator.
    func: Vec<f32>,
    /// Per-row CDFs over u, `width + 1` entries each.
    conditional_cdf: Vec<f32>,
    /// Each row's integral — which is also the marginal distribution's own
    /// function.
    row_integral: Vec<f32>,
    /// CDF over v, `height + 1` entries.
    marginal_cdf: Vec<f32>,
    /// Integral of the whole function. Zero for a black environment.
    total: f32,
}

impl EnvDistribution {
    /// Build from anything that can report radiance in a direction.
    ///
    /// Returns `None` when the environment is uniformly black, in which case
    /// there is nothing to importance-sample and the caller should fall back to
    /// cosine-weighted sampling.
    pub fn build(radiance: impl Fn(Vector3) -> Vector3) -> Option<Self> {
        Self::build_sized(ENV_WIDTH, ENV_HEIGHT, radiance)
    }

    /// As [`Self::build`], at an explicit resolution. Exposed for tests.
    pub fn build_sized(
        width: usize,
        height: usize,
        radiance: impl Fn(Vector3) -> Vector3,
    ) -> Option<Self> {
        let (width, height) = (width.max(2), height.max(2));
        let mut func = vec![0.0f32; width * height];
        let inv_ss = 1.0 / (SUPERSAMPLE * SUPERSAMPLE) as f32;

        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0f32;
                for sy in 0..SUPERSAMPLE {
                    let v = (y as f32 + (sy as f32 + 0.5) / SUPERSAMPLE as f32) / height as f32;
                    for sx in 0..SUPERSAMPLE {
                        let u = (x as f32 + (sx as f32 + 0.5) / SUPERSAMPLE as f32) / width as f32;
                        sum += luminance(radiance(direction_from_uv(u, v)));
                    }
                }
                // The sine is taken at the texel centre; the texel is narrow
                // enough in v that the variation across it does not matter.
                let theta = (y as f32 + 0.5) / height as f32 * PI;
                func[y * width + x] = (sum * inv_ss).max(0.0) * theta.sin();
            }
        }

        let mut conditional_cdf = vec![0.0f32; height * (width + 1)];
        let mut row_integral = vec![0.0f32; height];
        for y in 0..height {
            let row = &func[y * width..(y + 1) * width];
            let cdf = &mut conditional_cdf[y * (width + 1)..(y + 1) * (width + 1)];
            row_integral[y] = build_cdf(row, cdf);
        }

        let mut marginal_cdf = vec![0.0f32; height + 1];
        let total = build_cdf(&row_integral, &mut marginal_cdf);
        if total <= 0.0 {
            return None;
        }

        Some(Self {
            width,
            height,
            func,
            conditional_cdf,
            row_integral,
            marginal_cdf,
            total,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// The raw arrays, in the order a device buffer wants them: function,
    /// conditional CDFs, row integrals, marginal CDF.
    pub fn arrays(&self) -> (&[f32], &[f32], &[f32], &[f32]) {
        (
            &self.func,
            &self.conditional_cdf,
            &self.row_integral,
            &self.marginal_cdf,
        )
    }

    /// Integral of the sampling function over the unit square.
    pub fn total(&self) -> f32 {
        self.total
    }

    /// Draw a direction. Returns it with its solid-angle density.
    pub fn sample(&self, u1: f32, u2: f32) -> (Vector3, f32) {
        // Marginal over rows first, then the conditional within the row it
        // picked. Two binary searches, both over arrays small enough to stay in
        // cache.
        let (v, iv) = sample_cdf(&self.marginal_cdf, self.height, u2);
        let row_start = iv * (self.width + 1);
        let (u, iu) = sample_cdf(
            &self.conditional_cdf[row_start..row_start + self.width + 1],
            self.width,
            u1,
        );

        let dir = direction_from_uv(u, v);
        (dir, self.pdf_at(iu, iv, v))
    }

    /// Solid-angle density this distribution assigns to `dir`.
    pub fn pdf(&self, dir: Vector3) -> f32 {
        let (u, v) = uv_from_direction(dir);
        let iu = ((u * self.width as f32) as usize).min(self.width - 1);
        let iv = ((v * self.height as f32) as usize).min(self.height - 1);
        self.pdf_at(iu, iv, v)
    }

    fn pdf_at(&self, iu: usize, iv: usize, v: f32) -> f32 {
        let sin_theta = (v * PI).sin();
        if sin_theta <= 0.0 || self.total <= 0.0 {
            return 0.0;
        }
        // Density over the unit square, then the Jacobian of
        // (u, v) -> direction: dω = 2π² sinθ du dv.
        let pdf_uv = self.func[iv * self.width + iu] / self.total;
        pdf_uv / (2.0 * PI * PI * sin_theta)
    }
}

/// Build a normalised CDF over `f`, returning its integral.
///
/// A zero function gets a uniform CDF so that sampling it still terminates —
/// the caller checks the integral to decide whether the result means anything.
fn build_cdf(f: &[f32], cdf: &mut [f32]) -> f32 {
    let n = f.len();
    cdf[0] = 0.0;
    for i in 0..n {
        cdf[i + 1] = cdf[i] + f[i] / n as f32;
    }
    let integral = cdf[n];
    if integral <= 0.0 {
        for (i, c) in cdf.iter_mut().enumerate().take(n + 1) {
            *c = i as f32 / n as f32;
        }
    } else {
        for c in cdf.iter_mut().take(n + 1) {
            *c /= integral;
        }
    }
    integral
}

/// Invert a CDF at `u`, returning the continuous position in `[0, 1)` and the
/// interval it fell in.
fn sample_cdf(cdf: &[f32], n: usize, u: f32) -> (f32, usize) {
    // `partition_point` is the binary search: the last index whose CDF value is
    // still at or below u.
    let i = cdf
        .partition_point(|&c| c <= u)
        .saturating_sub(1)
        .min(n - 1);
    let lo = cdf[i];
    let hi = cdf[i + 1];
    let du = if hi > lo { (u - lo) / (hi - lo) } else { 0.5 };
    (((i as f32 + du) / n as f32).clamp(0.0, 0.999_999), i)
}

/// Equirectangular UV to direction. `v` runs from the +Y pole to the -Y pole,
/// `u` around the Y axis from +X through +Z.
pub fn direction_from_uv(u: f32, v: f32) -> Vector3 {
    let theta = v * PI;
    let phi = u * 2.0 * PI;
    let (sin_t, cos_t) = theta.sin_cos();
    let (sin_p, cos_p) = phi.sin_cos();
    Vector3::new(sin_t * cos_p, cos_t, sin_t * sin_p)
}

/// The inverse of [`direction_from_uv`].
///
/// θ comes from `atan2(hypot(x, z), y)` rather than `acos(y)`. They agree
/// mathematically and not numerically: near a pole, `acos` maps a float's worth
/// of error in `y` into a large relative error in θ — and θ enters the density
/// through `sinθ`, so a direction and its own round trip could disagree on
/// their probability by several percent. `atan2` stays well conditioned there.
pub fn uv_from_direction(dir: Vector3) -> (f32, f32) {
    let v = (dir.x * dir.x + dir.z * dir.z).sqrt().atan2(dir.y) / PI;
    let mut u = dir.z.atan2(dir.x) / (2.0 * PI);
    if u < 0.0 {
        u += 1.0;
    }
    (u.clamp(0.0, 0.999_999), v.clamp(0.0, 0.999_999))
}

fn luminance(c: Vector3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raytrace::sampler::{uniform_sphere, Rng};

    /// A sky that is black except for a small bright disc around `axis` — the
    /// case cosine sampling cannot resolve.
    fn sun(axis: Vector3, radius: f32, power: f32) -> impl Fn(Vector3) -> Vector3 {
        let axis = axis.normalize();
        move |d: Vector3| {
            if d.normalize().dot(axis) > radius.cos() {
                Vector3::new(power, power, power)
            } else {
                Vector3::ZERO
            }
        }
    }

    #[test]
    fn uv_and_direction_round_trip() {
        let mut rng = Rng::new(1, 2);
        for _ in 0..5000 {
            let d = uniform_sphere(rng.next_f32(), rng.next_f32());
            let (u, v) = uv_from_direction(d);
            let back = direction_from_uv(u, v);
            assert!((back - d).length() < 1e-3, "{d:?} -> ({u},{v}) -> {back:?}");
        }
    }

    #[test]
    fn a_black_environment_has_no_distribution() {
        assert!(EnvDistribution::build(|_| Vector3::ZERO).is_none());
    }

    /// The density must integrate to 1 over the sphere, or every estimate that
    /// divides by it is scaled wrong.
    #[test]
    fn pdf_integrates_to_one_over_the_sphere() {
        let d = EnvDistribution::build(sun(Vector3::new(0.3, 0.8, 0.2), 0.3, 50.0)).unwrap();
        let mut rng = Rng::new(3, 4);
        let n = 400_000;
        let mut sum = 0.0f64;
        for _ in 0..n {
            let dir = uniform_sphere(rng.next_f32(), rng.next_f32());
            sum += d.pdf(dir) as f64;
        }
        // Uniform-sphere density is 1/4π.
        let integral = sum / n as f64 * (4.0 * PI) as f64;
        assert!(
            (integral - 1.0).abs() < 0.03,
            "pdf integrates to {integral}"
        );
    }

    /// `sample` and `pdf` have to agree, or the MIS weights are computed from a
    /// different density than the samples were drawn from.
    #[test]
    fn sample_and_pdf_agree() {
        let d = EnvDistribution::build(sun(Vector3::new(0.0, 1.0, 0.0), 0.25, 30.0)).unwrap();
        let mut rng = Rng::new(5, 6);
        let mut checked = 0;
        for _ in 0..20_000 {
            let (dir, pdf) = d.sample(rng.next_f32(), rng.next_f32());
            if pdf <= 0.0 {
                continue;
            }
            let looked_up = d.pdf(dir);
            assert!(
                (looked_up - pdf).abs() <= 1e-3 * pdf.max(1.0),
                "sampled pdf {pdf}, lookup {looked_up} for {dir:?}"
            );
            checked += 1;
        }
        assert!(checked > 19_000, "only {checked} samples were usable");
    }

    /// The whole point: samples must land on the bright part of the sky.
    #[test]
    fn samples_concentrate_on_the_bright_region() {
        let axis = Vector3::new(0.2, 0.9, -0.3).normalize();
        let radius = 0.2f32;
        let d = EnvDistribution::build(sun(axis, radius, 100.0)).unwrap();
        let mut rng = Rng::new(7, 8);
        let n = 20_000;
        let mut on_target = 0;
        for _ in 0..n {
            let (dir, _) = d.sample(rng.next_f32(), rng.next_f32());
            // Allow a texel's slack: the grid is coarse on purpose.
            if dir.dot(axis) > (radius + 0.05).cos() {
                on_target += 1;
            }
        }
        let frac = on_target as f32 / n as f32;
        assert!(
            frac > 0.9,
            "only {frac} of samples found a disc covering {:.4} of the sphere",
            (1.0 - radius.cos()) / 2.0
        );
    }

    /// Estimating the sky's total irradiance on a surface must converge to the
    /// same number whether the directions came from this distribution or from a
    /// uniform sphere — with far fewer samples.
    #[test]
    fn the_estimator_is_unbiased_and_far_less_noisy() {
        let axis = Vector3::new(0.0, 1.0, 0.0);
        let radius = 0.1f32;
        let power = 200.0f32;
        let sky = sun(axis, radius, power);
        let d = EnvDistribution::build(&sky).unwrap();
        let normal = Vector3::new(0.0, 1.0, 0.0);

        // Reference: E = L * Omega * cos, for a small disc straight overhead.
        let solid_angle = 2.0 * PI * (1.0 - radius.cos());
        let reference = power * solid_angle;

        let mut rng = Rng::new(9, 10);
        let n = 20_000;
        let mut importance = 0.0f64;
        for _ in 0..n {
            let (dir, pdf) = d.sample(rng.next_f32(), rng.next_f32());
            if pdf <= 0.0 {
                continue;
            }
            let cos = dir.dot(normal).max(0.0);
            importance += (luminance(sky(dir)) * cos / pdf) as f64;
        }
        let importance = importance / n as f64;
        assert!(
            (importance - reference as f64).abs() < 0.05 * reference as f64,
            "importance-sampled {importance} vs analytic {reference}"
        );

        // The same budget spent on cosine-weighted directions barely finds it.
        let mut hits = 0;
        for _ in 0..n {
            let dir = uniform_sphere(rng.next_f32(), rng.next_f32());
            if dir.dot(axis) > radius.cos() {
                hits += 1;
            }
        }
        assert!(
            hits * 20 < n,
            "the test disc is not small enough to make the point ({hits} hits)"
        );
    }

    /// A uniform sky should give a near-uniform density — but only away from
    /// the poles.
    ///
    /// The function is piecewise-constant in `(u, v)` with `sin θ` sampled at
    /// each texel's centre, while the projection's Jacobian uses the *actual*
    /// `sin θ`. Within one texel the two differ, and near a pole they differ a
    /// lot, because `sin θ` changes by a factor of several across the top row.
    /// The density is still exactly normalised — that is what
    /// `pdf_integrates_to_one_over_the_sphere` checks — it just is not flat
    /// there, and no piecewise-constant equirectangular density is.
    #[test]
    fn a_uniform_sky_gives_a_near_uniform_density_away_from_the_poles() {
        let d = EnvDistribution::build(|_| Vector3::new(1.0, 1.0, 1.0)).unwrap();
        let mut rng = Rng::new(11, 12);
        let mut checked = 0;
        for _ in 0..4000 {
            let dir = uniform_sphere(rng.next_f32(), rng.next_f32());
            if dir.y.abs() > 0.9 {
                continue;
            }
            let pdf = d.pdf(dir);
            assert!(
                (pdf - 1.0 / (4.0 * PI)).abs() < 2e-3,
                "pdf {pdf} for a uniform sky at y = {}",
                dir.y
            );
            checked += 1;
        }
        assert!(checked > 2000, "only {checked} directions were tested");
    }

    #[test]
    fn the_poles_are_not_oversampled() {
        // A uniform sky: every direction is equally likely, so the fraction of
        // samples near a pole must match that cap's share of the sphere. Without
        // the sinθ Jacobian the grid's rows are equal-probability and the poles
        // collect far more than their share.
        let d = EnvDistribution::build(|_| Vector3::new(1.0, 1.0, 1.0)).unwrap();
        let mut rng = Rng::new(13, 14);
        let n = 40_000;
        let cap = 0.95f32; // cos of the cap half-angle
        let mut in_cap = 0;
        for _ in 0..n {
            let (dir, _) = d.sample(rng.next_f32(), rng.next_f32());
            if dir.y > cap {
                in_cap += 1;
            }
        }
        let expected = (1.0 - cap) / 2.0;
        let got = in_cap as f32 / n as f32;
        assert!(
            (got - expected).abs() < 0.01,
            "cap holds {expected} of the sphere but took {got} of the samples"
        );
    }
}
