//! three.js r165's `SimplexNoise` (`examples/jsm/math/SimplexNoise.js`),
//! operation for operation and in `f64`, so a field seeded the same way gives
//! the same numbers as three.js.
//!
//! This is not [`SimplexNoise`](super::SimplexNoise): that one is unseeded,
//! rescaled into [-1, 1] and builds 3D noise from 2D slices, which is what
//! the crate's own consumers want. This one exists to be three.js's — the
//! permutation is drawn from a caller's `random()` exactly as three.js draws
//! it, and the 2D/3D/4D scales (70, 32, 27) are three.js's.

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

const GRAD4: [[f64; 4]; 32] = [
    [0.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0, -1.0],
    [0.0, 1.0, -1.0, 1.0],
    [0.0, 1.0, -1.0, -1.0],
    [0.0, -1.0, 1.0, 1.0],
    [0.0, -1.0, 1.0, -1.0],
    [0.0, -1.0, -1.0, 1.0],
    [0.0, -1.0, -1.0, -1.0],
    [1.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0, -1.0],
    [1.0, 0.0, -1.0, 1.0],
    [1.0, 0.0, -1.0, -1.0],
    [-1.0, 0.0, 1.0, 1.0],
    [-1.0, 0.0, 1.0, -1.0],
    [-1.0, 0.0, -1.0, 1.0],
    [-1.0, 0.0, -1.0, -1.0],
    [1.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, -1.0],
    [1.0, -1.0, 0.0, 1.0],
    [1.0, -1.0, 0.0, -1.0],
    [-1.0, 1.0, 0.0, 1.0],
    [-1.0, 1.0, 0.0, -1.0],
    [-1.0, -1.0, 0.0, 1.0],
    [-1.0, -1.0, 0.0, -1.0],
    [1.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, -1.0, 0.0],
    [1.0, -1.0, 1.0, 0.0],
    [1.0, -1.0, -1.0, 0.0],
    [-1.0, 1.0, 1.0, 0.0],
    [-1.0, 1.0, -1.0, 0.0],
    [-1.0, -1.0, 1.0, 0.0],
    [-1.0, -1.0, -1.0, 0.0],
];

#[rustfmt::skip]
const SIMPLEX: [[u8; 4]; 64] = [
    [0, 1, 2, 3], [0, 1, 3, 2], [0, 0, 0, 0], [0, 2, 3, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 3, 0],
    [0, 2, 1, 3], [0, 0, 0, 0], [0, 3, 1, 2], [0, 3, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 3, 2, 0],
    [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
    [1, 2, 0, 3], [0, 0, 0, 0], [1, 3, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 3, 0, 1], [2, 3, 1, 0],
    [1, 0, 2, 3], [1, 0, 3, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 3, 1], [0, 0, 0, 0], [2, 1, 3, 0],
    [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
    [2, 0, 1, 3], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [3, 0, 1, 2], [3, 0, 2, 1], [0, 0, 0, 0], [3, 1, 2, 0],
    [2, 1, 0, 3], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [3, 1, 0, 2], [0, 0, 0, 0], [3, 2, 0, 1], [3, 2, 1, 0],
];

/// three.js's `SimplexNoise`: 2D, 3D and 4D simplex noise over a seeded
/// permutation.
#[derive(Debug, Clone)]
pub struct ThreeSimplexNoise {
    perm: [u8; 512],
}

/// JavaScript's `i & 255` on a `Math.floor` result.
fn wrap(i: f64) -> usize {
    ((i as i64) & 255) as usize
}

impl ThreeSimplexNoise {
    /// From the 256 entries three.js draws as `Math.floor(r.random() * 256)`.
    pub fn from_permutation(p: &[u8; 256]) -> Self {
        let mut perm = [0u8; 512];
        for (i, v) in perm.iter_mut().enumerate() {
            *v = p[i & 255];
        }
        Self { perm }
    }

    /// Seeded from `random`, called 256 times in three.js's order; each value
    /// in `[0, 1)`.
    pub fn from_random(mut random: impl FnMut() -> f64) -> Self {
        let mut p = [0u8; 256];
        for v in p.iter_mut() {
            *v = (random() * 256.0).floor().clamp(0.0, 255.0) as u8;
        }
        Self::from_permutation(&p)
    }

    fn perm(&self, i: usize) -> usize {
        self.perm[i] as usize
    }

    /// `noise(x, y)`: 2D, scaled by 70.
    pub fn noise(&self, xin: f64, yin: f64) -> f64 {
        let f2 = 0.5 * (3.0f64.sqrt() - 1.0);
        let s = (xin + yin) * f2;
        let i = (xin + s).floor();
        let j = (yin + s).floor();
        let g2 = (3.0 - 3.0f64.sqrt()) / 6.0;
        let t = (i + j) * g2;
        let x0 = xin - (i - t);
        let y0 = yin - (j - t);
        let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };
        let x1 = x0 - i1 as f64 + g2;
        let y1 = y0 - j1 as f64 + g2;
        let x2 = x0 - 1.0 + 2.0 * g2;
        let y2 = y0 - 1.0 + 2.0 * g2;
        let (ii, jj) = (wrap(i), wrap(j));
        let gi0 = self.perm(ii + self.perm(jj)) % 12;
        let gi1 = self.perm(ii + i1 + self.perm(jj + j1)) % 12;
        let gi2 = self.perm(ii + 1 + self.perm(jj + 1)) % 12;
        let corner = |x: f64, y: f64, g: usize| {
            let mut t = 0.5 - x * x - y * y;
            if t < 0.0 {
                0.0
            } else {
                t *= t;
                t * t * (GRAD3[g][0] * x + GRAD3[g][1] * y)
            }
        };
        70.0 * (corner(x0, y0, gi0) + corner(x1, y1, gi1) + corner(x2, y2, gi2))
    }

    /// `noise3d(x, y, z)`: 3D, scaled by 32.
    pub fn noise3d(&self, xin: f64, yin: f64, zin: f64) -> f64 {
        let f3 = 1.0 / 3.0;
        let s = (xin + yin + zin) * f3;
        let i = (xin + s).floor();
        let j = (yin + s).floor();
        let k = (zin + s).floor();
        let g3 = 1.0 / 6.0;
        let t = (i + j + k) * g3;
        let x0 = xin - (i - t);
        let y0 = yin - (j - t);
        let z0 = zin - (k - t);
        let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
            if y0 >= z0 {
                (1, 0, 0, 1, 1, 0)
            } else if x0 >= z0 {
                (1, 0, 0, 1, 0, 1)
            } else {
                (0, 0, 1, 1, 0, 1)
            }
        } else if y0 < z0 {
            (0, 0, 1, 0, 1, 1)
        } else if x0 < z0 {
            (0, 1, 0, 0, 1, 1)
        } else {
            (0, 1, 0, 1, 1, 0)
        };
        let x1 = x0 - i1 as f64 + g3;
        let y1 = y0 - j1 as f64 + g3;
        let z1 = z0 - k1 as f64 + g3;
        let x2 = x0 - i2 as f64 + 2.0 * g3;
        let y2 = y0 - j2 as f64 + 2.0 * g3;
        let z2 = z0 - k2 as f64 + 2.0 * g3;
        let x3 = x0 - 1.0 + 3.0 * g3;
        let y3 = y0 - 1.0 + 3.0 * g3;
        let z3 = z0 - 1.0 + 3.0 * g3;
        let (ii, jj, kk) = (wrap(i), wrap(j), wrap(k));
        let p = |a: usize| self.perm(a);
        let gi0 = p(ii + p(jj + p(kk))) % 12;
        let gi1 = p(ii + i1 + p(jj + j1 + p(kk + k1))) % 12;
        let gi2 = p(ii + i2 + p(jj + j2 + p(kk + k2))) % 12;
        let gi3 = p(ii + 1 + p(jj + 1 + p(kk + 1))) % 12;
        let corner = |x: f64, y: f64, z: f64, g: usize| {
            let mut t = 0.6 - x * x - y * y - z * z;
            if t < 0.0 {
                0.0
            } else {
                t *= t;
                t * t * (GRAD3[g][0] * x + GRAD3[g][1] * y + GRAD3[g][2] * z)
            }
        };
        32.0 * (corner(x0, y0, z0, gi0)
            + corner(x1, y1, z1, gi1)
            + corner(x2, y2, z2, gi2)
            + corner(x3, y3, z3, gi3))
    }

    /// `noise4d(x, y, z, w)`: 4D, scaled by 27.
    pub fn noise4d(&self, x: f64, y: f64, z: f64, w: f64) -> f64 {
        let f4 = (5.0f64.sqrt() - 1.0) / 4.0;
        let g4 = (5.0 - 5.0f64.sqrt()) / 20.0;
        let s = (x + y + z + w) * f4;
        let i = (x + s).floor();
        let j = (y + s).floor();
        let k = (z + s).floor();
        let l = (w + s).floor();
        let t = (i + j + k + l) * g4;
        let x0 = x - (i - t);
        let y0 = y - (j - t);
        let z0 = z - (k - t);
        let w0 = w - (l - t);
        let c = (if x0 > y0 { 32 } else { 0 })
            + (if x0 > z0 { 16 } else { 0 })
            + (if y0 > z0 { 8 } else { 0 })
            + (if x0 > w0 { 4 } else { 0 })
            + (if y0 > w0 { 2 } else { 0 })
            + (if z0 > w0 { 1 } else { 0 });
        let sc = SIMPLEX[c];
        let step = |n: u8| -> [usize; 4] { sc.map(|v| usize::from(v >= n)) };
        let [i1, j1, k1, l1] = step(3);
        let [i2, j2, k2, l2] = step(2);
        let [i3, j3, k3, l3] = step(1);
        let off = |d: f64, o: usize, n: f64| d - o as f64 + n * g4;
        let c1 = [off(x0, i1, 1.0), off(y0, j1, 1.0), off(z0, k1, 1.0), off(w0, l1, 1.0)];
        let c2 = [off(x0, i2, 2.0), off(y0, j2, 2.0), off(z0, k2, 2.0), off(w0, l2, 2.0)];
        let c3 = [off(x0, i3, 3.0), off(y0, j3, 3.0), off(z0, k3, 3.0), off(w0, l3, 3.0)];
        let c4 = [
            x0 - 1.0 + 4.0 * g4,
            y0 - 1.0 + 4.0 * g4,
            z0 - 1.0 + 4.0 * g4,
            w0 - 1.0 + 4.0 * g4,
        ];
        let (ii, jj, kk, ll) = (wrap(i), wrap(j), wrap(k), wrap(l));
        let p = |a: usize| self.perm(a);
        let gi0 = p(ii + p(jj + p(kk + p(ll)))) % 32;
        let gi1 = p(ii + i1 + p(jj + j1 + p(kk + k1 + p(ll + l1)))) % 32;
        let gi2 = p(ii + i2 + p(jj + j2 + p(kk + k2 + p(ll + l2)))) % 32;
        let gi3 = p(ii + i3 + p(jj + j3 + p(kk + k3 + p(ll + l3)))) % 32;
        let gi4 = p(ii + 1 + p(jj + 1 + p(kk + 1 + p(ll + 1)))) % 32;
        let corner = |[x, y, z, w]: [f64; 4], g: usize| {
            let mut t = 0.6 - x * x - y * y - z * z - w * w;
            if t < 0.0 {
                0.0
            } else {
                t *= t;
                let gr = GRAD4[g];
                t * t * (gr[0] * x + gr[1] * y + gr[2] * z + gr[3] * w)
            }
        };
        27.0 * (corner([x0, y0, z0, w0], gi0)
            + corner(c1, gi1)
            + corner(c2, gi2)
            + corner(c3, gi3)
            + corner(c4, gi4))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shim's `seedRandom` LCG — the generator the parity scenes seed
    /// `Math.random` with.
    fn lcg(seed: u32) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            s as f64 / 4_294_967_296.0
        }
    }

    /// Values from three.js r165's `SimplexNoise` in node, constructed with
    /// `{ random: lcg(42) }`.
    #[test]
    fn matches_three_js() {
        let n = ThreeSimplexNoise::from_random(lcg(42));
        let want: &[(&str, [f64; 4], f64)] = include!("testdata/three_simplex_r165.in");
        for (kind, a, v) in want {
            let got = match *kind {
                "2" => n.noise(a[0], a[1]),
                "3" => n.noise3d(a[0], a[1], a[2]),
                _ => n.noise4d(a[0], a[1], a[2], a[3]),
            };
            assert!((got - v).abs() <= 1e-12, "noise{kind}({a:?}) = {got}, three.js {v}");
        }
    }
}
