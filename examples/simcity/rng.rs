//! Part of the `simcity` example; see `mod.rs`.
#![allow(dead_code)]

// Self-contained: no shared imports needed.
#[allow(unused_imports)]
use super::*;

// ---------------------------------------------------------------------------
// Deterministic RNG — splitmix64. Same seed, same city, on every platform.
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 32) as u32
    }

    /// Uniform in `[0, 1)`.
    pub(crate) fn f(&mut self) -> f32 {
        self.next_u32() as f32 / 4_294_967_296.0
    }

    pub(crate) fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }

    pub(crate) fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            self.next_u32() as usize % n
        }
    }

    pub(crate) fn chance(&mut self, p: f32) -> bool {
        self.f() < p
    }
}
