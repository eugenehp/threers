//! `Math.random`, as the post-processing passes draw from it.
//!
//! Several three.js passes are random by construction: GlitchPass picks its
//! displacement map and every frame's glitch with `Math.random`, SSAOPass its
//! sample kernel and rotation noise. To match three.js frame for frame a run
//! has to draw the same numbers in the same order — so the passes here draw
//! from one shared stream, as three.js's do from `Math.random`, and that
//! stream can be seeded.
//!
//! Seeded, it is the 32-bit LCG the parity scenes install as `Math.random`
//! (`s = s * 1664525 + 1013904223`, returned as `s / 2^32`); the browser shim
//! points `Math.random` at [`random`] when it seeds, so JavaScript and these
//! passes share the one stream exactly as three.js's code shares
//! `Math.random`. Unseeded, it is the platform's: `Math.random` itself in a
//! browser, a time-seeded generator natively.

use std::sync::Mutex;

enum Source {
    /// The parity LCG, from a seed.
    Lcg(u32),
    /// xorshift64*, seeded from the clock on first use (native only; a
    /// browser uses `Math.random`).
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    Native(u64),
    Unset,
}

static STATE: Mutex<Source> = Mutex::new(Source::Unset);

/// Seed the stream: every [`random`] after this follows the LCG from `seed`,
/// as the parity scenes' `seedRandom` makes `Math.random` do.
pub fn seed_random(seed: u32) {
    *STATE.lock().unwrap_or_else(|e| e.into_inner()) = Source::Lcg(seed);
}

/// Back to the platform's generator.
pub fn unseed_random() {
    *STATE.lock().unwrap_or_else(|e| e.into_inner()) = Source::Unset;
}

/// Whether [`seed_random`] is in effect.
pub fn is_seeded() -> bool {
    matches!(
        *STATE.lock().unwrap_or_else(|e| e.into_inner()),
        Source::Lcg(_)
    )
}

/// A number in `[0, 1)` — three.js's `Math.random()`.
pub fn random() -> f64 {
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    match &mut *state {
        Source::Lcg(s) => {
            *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *s as f64 / 4294967296.0
        }
        Source::Native(x) => xorshift(x),
        Source::Unset => {
            #[cfg(target_arch = "wasm32")]
            {
                // Not under the lock: `Math.random` may be the shim's,
                // which calls back in here — only while seeded, but a
                // re-entrant lock would deadlock rather than tell.
                drop(state);
                js_sys::Math::random()
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0x9e37_79b9_7f4a_7c15);
                let mut x = nanos | 1;
                let r = xorshift(&mut x);
                *state = Source::Native(x);
                r
            }
        }
    }
}

fn xorshift(x: &mut u64) -> f64 {
    *x ^= *x >> 12;
    *x ^= *x << 25;
    *x ^= *x >> 27;
    let v = x.wrapping_mul(0x2545_f491_4f6c_dd1d);
    (v >> 11) as f64 / (1u64 << 53) as f64
}

/// three.js `MathUtils.randFloat(low, high)`.
pub fn rand_float(low: f64, high: f64) -> f64 {
    low + random() * (high - low)
}

/// three.js `MathUtils.randInt(low, high)`, inclusive.
pub fn rand_int(low: i64, high: i64) -> i64 {
    low + (random() * (high - low + 1) as f64).floor() as i64
}

/// three.js's `SimplexNoise` (examples/jsm/math), which SSAOPass builds its
/// rotation noise with. Its permutation table comes from [`random`], 256
/// draws, as three.js's does from `Math.random`.
pub struct SimplexNoise {
    perm: [u8; 512],
}

const GRAD3: [[f64; 3]; 12] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
];

impl Default for SimplexNoise {
    fn default() -> Self {
        Self::new()
    }
}

impl SimplexNoise {
    pub fn new() -> Self {
        let mut p = [0u8; 256];
        for v in p.iter_mut() {
            *v = (random() * 256.0).floor() as u8;
        }
        let mut perm = [0u8; 512];
        for i in 0..512 {
            perm[i] = p[i & 255];
        }
        Self { perm }
    }

    fn dot3(g: &[f64; 3], x: f64, y: f64, z: f64) -> f64 {
        g[0] * x + g[1] * y + g[2] * z
    }

    /// `noise3d`, in double precision as JavaScript computes it.
    pub fn noise3d(&self, xin: f64, yin: f64, zin: f64) -> f64 {
        let f3 = 1.0 / 3.0;
        let g3 = 1.0 / 6.0;
        let s = (xin + yin + zin) * f3;
        let i = (xin + s).floor();
        let j = (yin + s).floor();
        let k = (zin + s).floor();
        let t = (i + j + k) * g3;
        let x0 = xin - (i - t);
        let y0 = yin - (j - t);
        let z0 = zin - (k - t);
        let (i1, j1, k1, i2, j2, k2);
        if x0 >= y0 {
            if y0 >= z0 {
                (i1, j1, k1, i2, j2, k2) = (1, 0, 0, 1, 1, 0);
            } else if x0 >= z0 {
                (i1, j1, k1, i2, j2, k2) = (1, 0, 0, 1, 0, 1);
            } else {
                (i1, j1, k1, i2, j2, k2) = (0, 0, 1, 1, 0, 1);
            }
        } else if y0 < z0 {
            (i1, j1, k1, i2, j2, k2) = (0, 0, 1, 0, 1, 1);
        } else if x0 < z0 {
            (i1, j1, k1, i2, j2, k2) = (0, 1, 0, 0, 1, 1);
        } else {
            (i1, j1, k1, i2, j2, k2) = (0, 1, 0, 1, 1, 0);
        }
        let x1 = x0 - i1 as f64 + g3;
        let y1 = y0 - j1 as f64 + g3;
        let z1 = z0 - k1 as f64 + g3;
        let x2 = x0 - i2 as f64 + 2.0 * g3;
        let y2 = y0 - j2 as f64 + 2.0 * g3;
        let z2 = z0 - k2 as f64 + 2.0 * g3;
        let x3 = x0 - 1.0 + 3.0 * g3;
        let y3 = y0 - 1.0 + 3.0 * g3;
        let z3 = z0 - 1.0 + 3.0 * g3;
        // `& 255` on JavaScript numbers: the integer's two's-complement bits.
        let ii = (i as i64 & 255) as usize;
        let jj = (j as i64 & 255) as usize;
        let kk = (k as i64 & 255) as usize;
        let p = |n: usize| self.perm[n] as usize;
        let gi0 = p(ii + p(jj + p(kk))) % 12;
        let gi1 = p(ii + i1 + p(jj + j1 + p(kk + k1))) % 12;
        let gi2 = p(ii + i2 + p(jj + j2 + p(kk + k2))) % 12;
        let gi3 = p(ii + 1 + p(jj + 1 + p(kk + 1))) % 12;
        let corner = |t: f64, g: usize, x: f64, y: f64, z: f64| {
            if t < 0.0 {
                0.0
            } else {
                let t = t * t;
                t * t * Self::dot3(&GRAD3[g], x, y, z)
            }
        };
        let n0 = corner(0.6 - x0 * x0 - y0 * y0 - z0 * z0, gi0, x0, y0, z0);
        let n1 = corner(0.6 - x1 * x1 - y1 * y1 - z1 * z1, gi1, x1, y1, z1);
        let n2 = corner(0.6 - x2 * x2 - y2 * y2 - z2 * z2, gi2, x2, y2, z2);
        let n3 = corner(0.6 - x3 * x3 - y3 * y3 - z3 * z3, gi3, x3, y3, z3);
        32.0 * (n0 + n1 + n2 + n3)
    }
}

/// Tests that seed or draw from the stream take this, so they do not
/// interleave their draws.
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The seeded stream is the parity scenes' LCG, value for value: the
    /// first draws after `seedRandom(42)` in a browser.
    #[test]
    fn the_seeded_stream_is_the_parity_lcg() {
        let _guard = test_lock();
        seed_random(42);
        let mut s: u32 = 42;
        for _ in 0..1000 {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            assert_eq!(random(), s as f64 / 4294967296.0);
        }
        unseed_random();
        assert!(!is_seeded());
        let r = random();
        assert!((0.0..1.0).contains(&r));
    }
}
