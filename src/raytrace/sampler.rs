//! Random numbers and the sampling routines the integrator draws from.
//!
//! Every path is seeded from `(pixel, sample_index)` alone, so a tile rendered
//! on one thread and the same tile rendered on another produce identical
//! numbers — the image does not depend on how the work was split, and a render
//! is reproducible from its seed.

use crate::math::Vector3;
use std::f32::consts::PI;

/// PCG-XSH-RR 64/32 — O'Neill's permuted congruential generator.
///
/// A 64-bit LCG whose output is a xorshift-then-rotate of the high bits. It is
/// two multiplies and a shift per number, has a 2^64 period, and passes
/// TestU01's BigCrush; a plain LCG's low bits do not, and the low bits are
/// exactly what a `[0,1)` float ends up made of.
#[derive(Debug, Clone, Copy)]
pub struct Rng {
    state: u64,
    inc: u64,
    /// Which sample of the pixel this is. Only meaningful when `stratified`.
    sample_index: u32,
    /// Per-pixel scramble, so neighbouring pixels use different permutations
    /// of the same sequence and the low-discrepancy structure does not show up
    /// as a visible grid.
    pixel_seed: u32,
    /// Which pair of dimensions the next [`Self::next_2d`] will draw.
    dim: u32,
    /// Whether 2D draws come from the Sobol sequence or from the PCG stream.
    stratified: bool,
}

impl Rng {
    /// `stream` selects one of 2^63 non-overlapping sequences; `seed` positions
    /// within it. Distinct pixels take distinct streams, so their paths are
    /// uncorrelated without the generator ever being re-seeded mid-render.
    pub fn new(stream: u64, seed: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (stream << 1) | 1,
            sample_index: 0,
            pixel_seed: 0,
            dim: 0,
            stratified: false,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(hash64(seed));
        rng.next_u32();
        rng
    }

    /// Seed from a pixel coordinate and a sample index.
    ///
    /// A sampler built this way draws its 2D pairs from an Owen-scrambled
    /// Sobol (0,2)-sequence rather than from the PCG stream — see
    /// [`Self::next_2d`]. Its 1D draws stay on PCG.
    pub fn for_sample(x: u32, y: u32, sample: u32, seed: u64) -> Self {
        let pixel = (y as u64) << 32 | x as u64;
        let mut rng = Self::new(hash64(pixel ^ seed.rotate_left(17)), sample as u64);
        rng.sample_index = sample;
        rng.pixel_seed = (hash64(pixel ^ seed) >> 32) as u32;
        rng.stratified = true;
        rng
    }

    /// Point the sampler at the block of dimensions belonging to bounce `b`.
    ///
    /// Stratification only helps if a given decision draws the *same*
    /// dimension on every sample of a pixel. Path tracing makes that awkward —
    /// how many random numbers a path consumes depends on what it hits — so
    /// dimensions are allocated in a fixed block per bounce and the counter is
    /// reset at the start of each one. Two samples that follow different paths
    /// then still agree about which dimension the first-bounce light sample
    /// lives in, which is where most of the variance is.
    pub fn set_bounce(&mut self, bounce: u32) {
        self.dim = bounce.saturating_mul(DIMENSIONS_PER_BOUNCE);
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in `[0, 1)`. Built from the top 24 bits, which is every bit a
    /// binade-aligned f32 can hold — dividing a full u32 would round to exactly
    /// 1.0 for the largest inputs and put a sample outside the domain.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    /// A 2D sample.
    ///
    /// For a sampler built by [`Self::for_sample`] this is the next pair of an
    /// Owen-scrambled Sobol (0,2)-sequence — a *stratified* pair, so twenty
    /// samples of a pixel cover the unit square evenly instead of clumping the
    /// way twenty independent draws do. Nearly every decision a path makes is
    /// two-dimensional (a point on a light, a direction off a BSDF, a point on
    /// the lens), so routing them all through here stratifies the whole
    /// renderer without a single call site knowing about it.
    ///
    /// Error falls closer to `1/n` than to `1/sqrt(n)` on the smooth part of an
    /// integrand, which is most of it.
    pub fn next_2d(&mut self) -> (f32, f32) {
        if !self.stratified {
            return (self.next_f32(), self.next_f32());
        }
        let d = self.dim;
        self.dim = self.dim.wrapping_add(1);
        sobol02_owen(
            self.sample_index,
            self.pixel_seed ^ hash32(d.wrapping_add(1)),
        )
    }

    /// Uniform integer in `[0, n)`, or 0 when `n == 0`.
    pub fn next_index(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        ((self.next_f32() as f64 * n as f64) as usize).min(n - 1)
    }
}

/// Thomas Wang-style 64-bit integer avalanche (splitmix64's finaliser). Turns
/// adjacent pixel indices into seeds that share no structure.
fn hash64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// A right-handed orthonormal basis with `n` as its local +Z.
///
/// Built by Duff et al.'s branchless method: it is exact for every unit vector
/// including the poles, where the textbook "cross with whichever axis is least
/// aligned" construction loses precision.
#[derive(Debug, Clone, Copy)]
pub struct Onb {
    pub t: Vector3,
    pub b: Vector3,
    pub n: Vector3,
}

impl Onb {
    pub fn new(n: Vector3) -> Self {
        let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
        let a = -1.0 / (sign + n.z);
        let b = n.x * n.y * a;
        Self {
            t: Vector3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
            b: Vector3::new(b, sign + n.y * n.y * a, -n.y),
            n,
        }
    }

    /// Local (tangent-space, +Z = normal) → world.
    pub fn to_world(&self, v: Vector3) -> Vector3 {
        self.t * v.x + self.b * v.y + self.n * v.z
    }

    /// World → local.
    pub fn to_local(&self, v: Vector3) -> Vector3 {
        Vector3::new(v.dot(self.t), v.dot(self.b), v.dot(self.n))
    }
}

/// Cosine-weighted direction on the +Z hemisphere, by Malley's method: a
/// uniform point on the disk lifted onto the sphere.
pub fn cosine_hemisphere(u1: f32, u2: f32) -> Vector3 {
    let (dx, dy) = concentric_disk(u1, u2);
    let z = (1.0 - dx * dx - dy * dy).max(0.0).sqrt();
    Vector3::new(dx, dy, z)
}

/// Solid-angle density of [`cosine_hemisphere`] for a direction at `cos_theta`.
pub fn cosine_hemisphere_pdf(cos_theta: f32) -> f32 {
    (cos_theta / PI).max(0.0)
}

/// Shirley–Chiu concentric mapping: square → disk with far less distortion
/// than the polar mapping, which matters because a stratified square stays
/// stratified through it.
pub fn concentric_disk(u1: f32, u2: f32) -> (f32, f32) {
    let ox = 2.0 * u1 - 1.0;
    let oy = 2.0 * u2 - 1.0;
    if ox == 0.0 && oy == 0.0 {
        return (0.0, 0.0);
    }
    let (r, theta) = if ox.abs() > oy.abs() {
        (ox, PI / 4.0 * (oy / ox))
    } else {
        (oy, PI / 2.0 - PI / 4.0 * (ox / oy))
    };
    (r * theta.cos(), r * theta.sin())
}

/// Uniform direction on the whole sphere.
pub fn uniform_sphere(u1: f32, u2: f32) -> Vector3 {
    let z = 1.0 - 2.0 * u1;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    Vector3::new(r * phi.cos(), r * phi.sin(), z)
}

/// Uniform direction inside a cone of half-angle `acos(cos_theta_max)` around
/// local +Z. This is how a light with an angular size is sampled — a sun disc,
/// or a point light given a radius.
pub fn uniform_cone(u1: f32, u2: f32, cos_theta_max: f32) -> Vector3 {
    let cos_theta = 1.0 - u1 * (1.0 - cos_theta_max);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    Vector3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

/// Solid-angle density of [`uniform_cone`].
pub fn uniform_cone_pdf(cos_theta_max: f32) -> f32 {
    let solid_angle = 2.0 * PI * (1.0 - cos_theta_max);
    if solid_angle <= 0.0 {
        0.0
    } else {
        1.0 / solid_angle
    }
}

/// Sample a GGX microfacet normal from the *visible* normal distribution
/// (Heitz 2018), in a local frame whose +Z is the shading normal.
///
/// Sampling the plain NDF generates microfacets the viewer cannot see and then
/// throws those samples away, which at grazing angles is most of them. The VNDF
/// generates only visible ones, so the estimator's variance stops climbing as
/// the view flattens out — the difference between a clean and a fizzing edge on
/// a rough sphere.
///
/// `ax`/`ay` are the anisotropic roughness values (α, i.e. roughness²).
pub fn sample_ggx_vndf(view: Vector3, ax: f32, ay: f32, u1: f32, u2: f32) -> Vector3 {
    // Stretch the view so the ellipsoid becomes a hemisphere.
    let vh = Vector3::new(ax * view.x, ay * view.y, view.z).normalize();
    // Orthonormal basis around the stretched view.
    let len_sq = vh.x * vh.x + vh.y * vh.y;
    let t1 = if len_sq > 0.0 {
        Vector3::new(-vh.y, vh.x, 0.0) * (1.0 / len_sq.sqrt())
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let t2 = vh.cross(t1);
    // Uniform point on the projected area: a disk, warped so the lower half
    // matches the hemisphere's projection.
    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;
    let p1 = r * phi.cos();
    let mut p2 = r * phi.sin();
    let s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * p2;
    let nh = t1 * p1 + t2 * p2 + vh * (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt();
    // Unstretch.
    Vector3::new(ax * nh.x, ay * nh.y, nh.z.max(1e-6)).normalize()
}

/// Dimension pairs reserved for each bounce. Generous enough that the light,
/// emitter, world and BSDF samples of one bounce never run into the next
/// bounce's block.
const DIMENSIONS_PER_BOUNCE: u32 = 8;

/// A 2D sample from an Owen-scrambled Sobol (0,2)-sequence.
///
/// The (0,2)-sequence is the classic pair of low-discrepancy dimensions: any
/// power-of-two prefix of it has exactly one sample in each cell of every
/// elongated dyadic partition of the unit square. Owen scrambling randomises it
/// per pixel without destroying that property — unlike the cheaper XOR
/// scramble, it also removes the structured correlation that otherwise shows up
/// as faint patterns across an image.
pub fn sobol02_owen(index: u32, seed: u32) -> (f32, f32) {
    // Shuffling the *index* as well as the values is what decorrelates two
    // pixels that happen to draw the same dimension: without it every pixel
    // walks the same sequence in the same order.
    let shuffled = owen_scramble(index, hash32(seed ^ 0x9e37_79b9));
    let x = owen_scramble(sobol_x(shuffled), hash32(seed ^ 0x6c50_b47c));
    let y = owen_scramble(sobol_y(shuffled), hash32(seed ^ 0xb82f_1e52));
    (to_unit(x), to_unit(y))
}

/// Hash-based nested uniform (Owen) scramble, after Burley 2020.
///
/// A true Owen scramble permutes every node of the binary tree of digits
/// independently, which is prohibitive; this reproduces its statistics with
/// four rounds of a self-multiplying mix, sandwiched between two bit
/// reversals so that the mixing propagates from the high digits down.
fn owen_scramble(x: u32, seed: u32) -> u32 {
    let mut x = x.reverse_bits();
    x = x.wrapping_add(seed);
    x ^= x.wrapping_mul(0x6c50_b47c);
    x ^= x.wrapping_mul(0xb82f_1e52);
    x ^= x.wrapping_mul(0xc7af_e638);
    x ^= x.wrapping_mul(0x8d22_f6e6);
    x.reverse_bits()
}

/// First Sobol dimension: the van der Corput sequence, which is just the index
/// with its bits reversed.
fn sobol_x(index: u32) -> u32 {
    index.reverse_bits()
}

/// Second Sobol dimension, by the Gray-code recurrence.
fn sobol_y(mut index: u32) -> u32 {
    let mut result = 0u32;
    let mut v = 1u32 << 31;
    while index != 0 {
        if index & 1 != 0 {
            result ^= v;
        }
        index >>= 1;
        v ^= v >> 1;
    }
    result
}

/// 32-bit integer avalanche, for deriving independent scramble seeds.
fn hash32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Top 24 bits to a float in `[0, 1)`, the same conversion `next_f32` uses.
fn to_unit(x: u32) -> f32 {
    (x >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Veach's power heuristic with β = 2 — the MIS weight for combining a light
/// sample against a BSDF sample.
pub fn power_heuristic(n_f: f32, pdf_f: f32, n_g: f32, pdf_g: f32) -> f32 {
    let f = n_f * pdf_f;
    let g = n_g * pdf_g;
    let denom = f * f + g * g;
    if denom <= 0.0 {
        0.0
    } else {
        (f * f) / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defining property of a (0,2)-sequence: any power-of-two prefix has
    /// exactly one point in every cell of every elongated dyadic partition of
    /// the unit square. For 64 points that means a 1x64, 2x32, 4x16, 8x8,
    /// 16x4, 32x2 and 64x1 grid all hold exactly one point per cell.
    ///
    /// This is the property that makes the sequence worth having, and Owen
    /// scrambling is chosen over the cheaper XOR scramble partly because it
    /// preserves it. If this test fails the sampler is merely random.
    #[test]
    fn sobol_prefixes_are_stratified_nets() {
        for seed in [1u32, 0xdead_beef, 12345] {
            let n = 64usize;
            let points: Vec<(f32, f32)> = (0..n).map(|i| sobol02_owen(i as u32, seed)).collect();
            for a in 0..=6u32 {
                let (nx, ny) = (1usize << a, 1usize << (6 - a));
                let mut cells = vec![0u32; nx * ny];
                for &(x, y) in &points {
                    let cx = ((x * nx as f32) as usize).min(nx - 1);
                    let cy = ((y * ny as f32) as usize).min(ny - 1);
                    cells[cy * nx + cx] += 1;
                }
                assert!(
                    cells.iter().all(|&c| c == 1),
                    "seed {seed}, {nx}x{ny} grid: cells hold {:?}",
                    cells
                        .iter()
                        .copied()
                        .collect::<std::collections::BTreeSet<_>>()
                );
            }
        }
    }

    #[test]
    fn sobol_stays_in_the_unit_square() {
        for seed in [0u32, 7, 0xffff_ffff] {
            for i in 0..10_000u32 {
                let (x, y) = sobol02_owen(i, seed);
                assert!(
                    (0.0..1.0).contains(&x) && (0.0..1.0).contains(&y),
                    "{x},{y}"
                );
            }
        }
    }

    /// Different pixels have to walk different permutations, or the sequence's
    /// structure shows up as a pattern across the image.
    #[test]
    fn different_seeds_decorrelate() {
        let a: Vec<(f32, f32)> = (0..64).map(|i| sobol02_owen(i, 11)).collect();
        let b: Vec<(f32, f32)> = (0..64).map(|i| sobol02_owen(i, 12)).collect();
        let matching = a
            .iter()
            .zip(&b)
            .filter(|(p, q)| (p.0 - q.0).abs() < 1e-6)
            .count();
        assert!(matching < 8, "{matching} of 64 samples coincided");
    }

    /// Stratification must not come at the cost of a shifted mean: the marginal
    /// distribution of each axis is still uniform.
    #[test]
    fn sobol_is_unbiased() {
        let n = 1 << 14;
        let mut sum = (0.0f64, 0.0f64);
        for i in 0..n {
            let (x, y) = sobol02_owen(i, 0x51633e2d);
            sum.0 += x as f64;
            sum.1 += y as f64;
        }
        let mean = (sum.0 / n as f64, sum.1 / n as f64);
        assert!((mean.0 - 0.5).abs() < 1e-3, "x mean {}", mean.0);
        assert!((mean.1 - 0.5).abs() < 1e-3, "y mean {}", mean.1);
    }

    /// The point of the exercise: integrating a smooth function converges far
    /// faster than with independent draws. Error falls closer to 1/n than to
    /// 1/sqrt(n) on the smooth part of an integrand, which is most of it.
    #[test]
    fn sobol_converges_faster_than_independent_sampling() {
        // A smooth integrand with a known value: the integral of
        // cos(pi*x/2) * cos(pi*y/2) over the unit square is (2/pi)^2.
        let f = |x: f32, y: f32| {
            (std::f32::consts::FRAC_PI_2 * x).cos() * (std::f32::consts::FRAC_PI_2 * y).cos()
        };
        let exact = (2.0 / PI as f64).powi(2);

        let n = 256u32;
        let trials = 64;
        let mut sobol_err = 0.0f64;
        let mut random_err = 0.0f64;
        for t in 0..trials {
            let mut a = 0.0f64;
            for i in 0..n {
                let (x, y) = sobol02_owen(i, 1000 + t);
                a += f(x, y) as f64;
            }
            sobol_err += (a / n as f64 - exact).abs();

            let mut rng = Rng::new(t as u64, 99);
            let mut b = 0.0f64;
            for _ in 0..n {
                let (x, y) = (rng.next_f32(), rng.next_f32());
                b += f(x, y) as f64;
            }
            random_err += (b / n as f64 - exact).abs();
        }
        sobol_err /= trials as f64;
        random_err /= trials as f64;
        assert!(
            sobol_err * 5.0 < random_err,
            "sobol mean error {sobol_err:.3e} vs independent {random_err:.3e} — \
             not the improvement this is for"
        );
    }

    /// A sampler built for a pixel draws stratified pairs; a bare one keeps the
    /// independent stream, because the sequence needs a sample index to walk.
    #[test]
    fn only_pixel_samplers_are_stratified() {
        let mut plain = Rng::new(1, 2);
        let a = plain.next_2d();
        let b = plain.next_2d();
        assert_ne!(a, b);

        // Two samples of the same pixel must differ, and the same sample of two
        // pixels must differ.
        let p0 = Rng::for_sample(4, 9, 0, 77).next_2d();
        let p1 = Rng::for_sample(4, 9, 1, 77).next_2d();
        let q0 = Rng::for_sample(5, 9, 0, 77).next_2d();
        assert_ne!(p0, p1);
        assert_ne!(p0, q0);
    }

    /// Dimension blocks have to line up, or a decision draws a different
    /// dimension on different samples and the stratification is wasted.
    #[test]
    fn bounce_blocks_are_deterministic() {
        let mut a = Rng::for_sample(3, 3, 7, 5);
        let mut b = Rng::for_sample(3, 3, 7, 5);
        a.set_bounce(2);
        // `b` gets there by a different route: a different number of draws
        // before the reset.
        b.set_bounce(0);
        let _ = b.next_2d();
        let _ = b.next_2d();
        let _ = b.next_2d();
        b.set_bounce(2);
        assert_eq!(a.next_2d(), b.next_2d());
    }

    #[test]
    fn rng_stays_in_unit_interval() {
        let mut rng = Rng::new(7, 11);
        for _ in 0..100_000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "{v} out of range");
        }
    }

    #[test]
    fn rng_mean_is_close_to_half() {
        let mut rng = Rng::new(1, 2);
        let n = 200_000;
        let mean: f64 = (0..n).map(|_| rng.next_f32() as f64).sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 5e-3, "mean {mean}");
    }

    #[test]
    fn distinct_pixels_decorrelate() {
        let a = Rng::for_sample(10, 20, 0, 99).next_f32();
        let b = Rng::for_sample(11, 20, 0, 99).next_f32();
        assert!((a - b).abs() > 1e-6, "adjacent pixels produced {a} and {b}");
    }

    #[test]
    fn onb_is_orthonormal_at_the_poles() {
        for n in [
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.3, -0.5, 0.81).normalize(),
        ] {
            let onb = Onb::new(n);
            assert!(onb.t.dot(onb.b).abs() < 1e-5);
            assert!(onb.t.dot(onb.n).abs() < 1e-5);
            assert!(onb.b.dot(onb.n).abs() < 1e-5);
            assert!((onb.t.length() - 1.0).abs() < 1e-5);
            assert!((onb.b.length() - 1.0).abs() < 1e-5);
            // Round trip.
            let v = Vector3::new(0.2, 0.7, -0.4);
            let rt = onb.to_local(onb.to_world(v));
            assert!((rt - v).length() < 1e-5);
        }
    }

    /// ∫ pdf dω over the hemisphere must be 1. Estimated by sampling the
    /// hemisphere uniformly and averaging pdf / (uniform pdf).
    #[test]
    fn cosine_hemisphere_pdf_integrates_to_one() {
        let mut rng = Rng::new(3, 4);
        let n = 200_000;
        let mut sum = 0.0f64;
        for _ in 0..n {
            let (u1, u2) = rng.next_2d();
            let d = uniform_sphere(u1, u2);
            if d.z <= 0.0 {
                continue;
            }
            // Uniform-sphere pdf is 1/4π; we only counted the upper half.
            sum += (cosine_hemisphere_pdf(d.z) * 4.0 * PI) as f64;
        }
        let integral = sum / n as f64;
        assert!((integral - 1.0).abs() < 0.02, "integral {integral}");
    }

    #[test]
    fn cone_pdf_matches_its_solid_angle() {
        let cos_max = 0.99f32;
        let mut rng = Rng::new(5, 6);
        let pdf = uniform_cone_pdf(cos_max);
        for _ in 0..1000 {
            let (u1, u2) = rng.next_2d();
            let d = uniform_cone(u1, u2, cos_max);
            assert!(d.z >= cos_max - 1e-4, "sample escaped the cone: {}", d.z);
            assert!((d.length() - 1.0).abs() < 1e-4);
        }
        assert!((pdf * 2.0 * PI * (1.0 - cos_max) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn power_heuristic_weights_sum_to_one() {
        let (a, b) = (0.3f32, 1.7f32);
        let wa = power_heuristic(1.0, a, 1.0, b);
        let wb = power_heuristic(1.0, b, 1.0, a);
        assert!((wa + wb - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vndf_samples_face_the_normal() {
        let mut rng = Rng::new(9, 10);
        for _ in 0..2000 {
            let (u1, u2) = rng.next_2d();
            let v = Vector3::new(0.6, 0.1, 0.79).normalize();
            let h = sample_ggx_vndf(v, 0.25, 0.25, u1, u2);
            assert!(h.z > 0.0, "half vector below the surface: {h:?}");
            assert!((h.length() - 1.0).abs() < 1e-4);
        }
    }
}
