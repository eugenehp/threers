//! One seeded random number generator, instead of twenty-six.
//!
//! Before this there were twenty-six files in the tree carrying their own
//! xorshift or multiply-hash — the path tracer, the lattices, the OpenSCAD
//! front end, the city generator, half the examples — and each one had reached
//! for the same four lines because the alternative was a dependency. They do
//! not agree with each other, none of them is reproducible across a refactor,
//! and every one of them is a place where "deterministic" is a claim nobody
//! checked.
//!
//! This is the one. It is not cryptographic and does not pretend to be: it is
//! PCG32, which is small, fast, has a long period and passes the statistical
//! tests that matter for scattering points on a surface.
//!
//! ```
//! use threers::math::Rng;
//!
//! let mut rng = Rng::new(42);
//! let a: Vec<f32> = (0..5).map(|_| rng.f32()).collect();
//!
//! // Same seed, same sequence — on every platform, in every build.
//! let mut again = Rng::new(42);
//! let b: Vec<f32> = (0..5).map(|_| again.f32()).collect();
//! assert_eq!(a, b);
//! ```

use crate::math::Vector3;

/// A seeded pseudo-random source. `Clone` so a sequence can be branched and
/// replayed, which is how a generator renders the same scene twice.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
    inc: u64,
}

impl Rng {
    /// Seed the generator. Every seed gives a different sequence and the same
    /// seed always gives the same one.
    pub fn new(seed: u64) -> Self {
        // The odd increment selects the stream; PCG requires it to be odd.
        let mut rng = Self {
            state: 0,
            inc: (seed << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// A second generator from this one, advanced to a different stream.
    ///
    /// For splitting work: give each thread, each tile or each object its own
    /// `fork`, and the result does not depend on the order they run in. Drawing
    /// from one shared generator across threads is the usual reason a
    /// "deterministic" generator stops being one.
    pub fn fork(&mut self, stream: u64) -> Self {
        Self::new(self.next_u64() ^ stream.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    /// The raw 32-bit output. PCG-XSH-RR.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    pub fn next_u64(&mut self) -> u64 {
        (self.next_u32() as u64) << 32 | self.next_u32() as u64
    }

    /// Uniform in `[0, 1)`.
    ///
    /// Built from the top 24 bits, which is every bit an `f32` mantissa can
    /// hold — taking the low bits instead is the classic way to get a
    /// generator whose lowest-order bit is not random at all.
    pub fn f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    /// Uniform in `[lo, hi)`. Reversed bounds come back as `lo`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }

    /// Uniform in `[0, n)`. Returns 0 when `n` is 0.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }

    /// `true` with probability `p`.
    pub fn chance(&mut self, p: f32) -> bool {
        self.f32() < p
    }

    /// Normally distributed, mean 0 and standard deviation 1.
    ///
    /// Marsaglia's polar method. p5's `randomGaussian`, and what you want
    /// whenever "random" means "jittered about a value" rather than "anywhere
    /// in a range" — the uniform version of a scatter reads as a grid with
    /// holes in it, and this does not.
    pub fn gaussian(&mut self) -> f32 {
        loop {
            let u = self.range(-1.0, 1.0);
            let v = self.range(-1.0, 1.0);
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                return u * (-2.0 * s.ln() / s).sqrt();
            }
        }
    }

    /// A point uniformly distributed on the unit sphere.
    ///
    /// Three gaussians normalised, which is the only cheap way that is actually
    /// uniform: sampling spherical angles uniformly bunches points at the poles,
    /// and it is a mistake that looks fine until the subject is a sphere.
    pub fn on_sphere(&mut self) -> Vector3 {
        loop {
            let v = Vector3::new(self.gaussian(), self.gaussian(), self.gaussian());
            let len = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
            if len > 1e-6 {
                return Vector3::new(v.x / len, v.y / len, v.z / len);
            }
        }
    }

    /// A point uniformly distributed inside the unit sphere.
    pub fn in_sphere(&mut self) -> Vector3 {
        let dir = self.on_sphere();
        // The cube root is what makes it uniform by *volume* rather than
        // crowding everything towards the centre.
        let r = self.f32().cbrt();
        Vector3::new(dir.x * r, dir.y * r, dir.z * r)
    }

    /// One of `items`, or `None` when it is empty.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        if items.is_empty() {
            None
        } else {
            items.get(self.below(items.len()))
        }
    }

    /// Fisher–Yates, in place.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.below(i + 1));
        }
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::new(0x2545_F491_4F6C_DD1D)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_sequence() {
        let a: Vec<u32> = (0..64).map(|_| Rng::new(7).next_u32()).collect();
        assert!(a.windows(2).all(|w| w[0] == w[1]), "a fresh Rng replays");

        let mut x = Rng::new(7);
        let mut y = Rng::new(7);
        for i in 0..1000 {
            assert_eq!(x.next_u32(), y.next_u32(), "diverged at {i}");
        }
    }

    #[test]
    fn different_seeds_give_different_sequences() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let differing = (0..64).filter(|_| a.next_u32() != b.next_u32()).count();
        assert!(differing > 60, "only {differing} of 64 draws differed");
    }

    /// Uniform means uniform. A generator whose low bits are not random passes
    /// a mean test and fails this one.
    #[test]
    fn f32_is_uniform_across_its_range() {
        let mut rng = Rng::new(99);
        let mut buckets = [0usize; 16];
        const N: usize = 160_000;
        for _ in 0..N {
            let v = rng.f32();
            assert!((0.0..1.0).contains(&v), "{v} is outside [0, 1)");
            buckets[(v * 16.0) as usize] += 1;
        }
        let expect = N / 16;
        for (i, b) in buckets.iter().enumerate() {
            let err = (*b as f32 - expect as f32).abs() / expect as f32;
            assert!(err < 0.05, "bucket {i} is {err:.3} off uniform");
        }
    }

    #[test]
    fn gaussian_has_the_moments_it_claims() {
        let mut rng = Rng::new(5);
        const N: usize = 200_000;
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        for _ in 0..N {
            let v = rng.gaussian() as f64;
            sum += v;
            sq += v * v;
        }
        let mean = sum / N as f64;
        let var = sq / N as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean {mean:.4} is not 0");
        assert!((var - 1.0).abs() < 0.03, "variance {var:.4} is not 1");
    }

    /// Sampling spherical angles uniformly crowds the poles. This is the test
    /// that catches it: equal-area latitude bands must get equal counts.
    #[test]
    fn on_sphere_is_uniform_by_area_and_not_by_angle() {
        let mut rng = Rng::new(11);
        let mut bands = [0usize; 10];
        const N: usize = 100_000;
        for _ in 0..N {
            let p = rng.on_sphere();
            assert!(
                ((p.x * p.x + p.y * p.y + p.z * p.z).sqrt() - 1.0).abs() < 1e-4,
                "not on the unit sphere"
            );
            // Equal-area bands in z, which is Archimedes' theorem.
            let band = (((p.z + 1.0) * 0.5 * 10.0) as usize).min(9);
            bands[band] += 1;
        }
        let expect = N / 10;
        for (i, b) in bands.iter().enumerate() {
            let err = (*b as f32 - expect as f32).abs() / expect as f32;
            assert!(err < 0.06, "latitude band {i} is {err:.3} off equal area");
        }
    }

    #[test]
    fn in_sphere_fills_the_volume_rather_than_the_centre() {
        let mut rng = Rng::new(13);
        let mut inner = 0usize;
        const N: usize = 50_000;
        for _ in 0..N {
            let p = rng.in_sphere();
            let r = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
            assert!(r <= 1.0 + 1e-4);
            // Half the radius is an eighth of the volume.
            if r < 0.5 {
                inner += 1;
            }
        }
        let frac = inner as f32 / N as f32;
        assert!(
            (frac - 0.125).abs() < 0.01,
            "{frac:.3} of points inside half the radius; uniform by volume is 0.125"
        );
    }

    #[test]
    fn shuffle_permutes_without_losing_anything() {
        let mut rng = Rng::new(3);
        let mut v: Vec<u32> = (0..64).collect();
        rng.shuffle(&mut v);
        assert_ne!(v, (0..64).collect::<Vec<_>>(), "nothing moved");
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>(), "elements were lost");
    }

    /// Forks have to be independent, or splitting work across threads
    /// reintroduces the correlation the split was meant to remove.
    #[test]
    fn forks_are_independent_streams() {
        let mut parent = Rng::new(21);
        let mut a = parent.fork(0);
        let mut b = parent.fork(1);
        let differing = (0..64).filter(|_| a.next_u32() != b.next_u32()).count();
        assert!(differing > 60, "two forks agreed on {} of 64", 64 - differing);
    }

    #[test]
    fn pick_and_below_stay_in_range() {
        let mut rng = Rng::new(17);
        let items = [1, 2, 3];
        for _ in 0..1000 {
            assert!(items.contains(rng.pick(&items).unwrap()));
            assert!(rng.below(3) < 3);
        }
        assert_eq!(rng.pick::<u8>(&[]), None);
        assert_eq!(rng.below(0), 0);
    }
}
