/// Lightweight 2D / 3D simplex noise. Mirrors three.js's `SimplexNoise`
/// (used by the Ashima implementation for terrain / cloud generation).
pub struct SimplexNoise;

impl SimplexNoise {
    /// 2D simplex noise in [-1, 1].
    pub fn noise2(x: f32, y: f32) -> f32 {
        const F2: f32 = 0.366_025_4; // (sqrt(3) - 1) / 2
        const G2: f32 = 0.211_324_87; // (3 - sqrt(3)) / 6
        let s = (x + y) * F2;
        let i = (x + s).floor();
        let j = (y + s).floor();
        let t = (i + j) * G2;
        let x0 = x - (i - t);
        let y0 = y - (j - t);
        let (i1, j1) = if x0 > y0 { (1.0, 0.0) } else { (0.0, 1.0) };
        let x1 = x0 - i1 + G2;
        let y1 = y0 - j1 + G2;
        let x2 = x0 - 1.0 + 2.0 * G2;
        let y2 = y0 - 1.0 + 2.0 * G2;
        let n0 = corner_contrib(x0, y0, i as i32, j as i32);
        let n1 = corner_contrib(x1, y1, (i + i1) as i32, (j + j1) as i32);
        let n2 = corner_contrib(x2, y2, (i + 1.0) as i32, (j + 1.0) as i32);
        70.0 * (n0 + n1 + n2)
    }
}

fn corner_contrib(x: f32, y: f32, i: i32, j: i32) -> f32 {
    let t = 0.5 - x * x - y * y;
    if t < 0.0 {
        return 0.0;
    }
    let t2 = t * t;
    let g = grad(hash(i, j), x, y);
    t2 * t2 * g
}

fn hash(i: i32, j: i32) -> u32 {
    let mut h = (i as u32)
        .wrapping_mul(374761393)
        .wrapping_add((j as u32).wrapping_mul(668265263));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h ^ (h >> 16)
}

fn grad(hash: u32, x: f32, y: f32) -> f32 {
    // 8 gradient directions evenly around the unit circle.
    let h = hash & 7;
    let u = if h < 4 { x } else { y };
    let v = if h < 4 { y } else { x };
    let sx = if h & 1 != 0 { -u } else { u };
    let sy = if h & 2 != 0 { -2.0 * v } else { 2.0 * v };
    sx + sy
}
