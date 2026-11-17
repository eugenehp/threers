//! Back-to-front ordering of Gaussian splats: a stable radix sort of their
//! centres by depth along an axis. See the README for an example.
//!
//! A frame's sort is three sweeps over the splats:
//!
//! 1. **Keys.** Depth along the axis, quantised to 24 bits over the range
//!    the bounding box spans on it (worked out from its corners, so no sweep
//!    is spent finding it). The centres are kept as three separate arrays
//!    and the loop runs over fixed-width chunks with no branches, which the
//!    compiler makes SIMD — NEON, SSE, or wasm `simd128` where enabled.
//! 2. **Low digit.** Both digits' histograms are counted, and then the splats
//!    scattered by their low 12 bits, key and index together.
//! 3. **High digit.** The (key, index) pairs read back in order — no gather
//!    by index — and scattered by their high 12 bits, which leaves the
//!    indices in order.

/// Bits a depth is quantised to.
const BITS: u32 = 24;
/// Bits a pass sorts by.
const RADIX: u32 = BITS / 2;
const BUCKETS: usize = 1 << RADIX;
const MASK: u32 = BUCKETS as u32 - 1;
/// The largest key, as a float.
const MAX: f32 = ((1u32 << BITS) - 1) as f32;
/// Splats the key loop handles at once: wide enough for any SIMD width.
const LANES: usize = 16;

/// Splat centres, kept ready to sort along any axis.
#[derive(Debug, Clone, Default)]
pub struct DepthSort {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    /// The bounding box of the finite centres.
    min: [f32; 3],
    max: [f32; 3],
    keys: Vec<u32>,
    /// `key << 32 | index`, after the low-digit pass.
    pairs: Vec<u64>,
    order: Vec<u32>,
}

impl DepthSort {
    /// Ready to sort `centres`. Its order starts as theirs.
    pub fn new(centres: &[[f32; 3]]) -> Self {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for p in centres {
            if p.iter().all(|v| v.is_finite()) {
                for k in 0..3 {
                    min[k] = min[k].min(p[k]);
                    max[k] = max[k].max(p[k]);
                }
            }
        }
        Self {
            x: centres.iter().map(|p| p[0]).collect(),
            y: centres.iter().map(|p| p[1]).collect(),
            z: centres.iter().map(|p| p[2]).collect(),
            min,
            max,
            keys: Vec::new(),
            pairs: Vec::new(),
            order: (0..centres.len() as u32).collect(),
        }
    }

    /// Ready to sort centres given as three columns of equal length — the
    /// form [`columns`](Self::columns) hands out, to sort on another thread.
    ///
    /// # Panics
    ///
    /// If the columns' lengths differ.
    pub fn from_columns(x: Vec<f32>, y: Vec<f32>, z: Vec<f32>) -> Self {
        assert!(x.len() == y.len() && y.len() == z.len(), "columns of different lengths");
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for i in 0..x.len() {
            let p = [x[i], y[i], z[i]];
            if p.iter().all(|v| v.is_finite()) {
                for k in 0..3 {
                    min[k] = min[k].min(p[k]);
                    max[k] = max[k].max(p[k]);
                }
            }
        }
        let n = x.len() as u32;
        Self { x, y, z, min, max, keys: Vec::new(), pairs: Vec::new(), order: (0..n).collect() }
    }

    /// The centres, as their `x`, `y` and `z` columns.
    pub fn columns(&self) -> [&[f32]; 3] {
        [&self.x, &self.y, &self.z]
    }

    /// How many centres there are.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// The last sort's order: indices into the centres.
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// The order as floats, which hold indices below 2²⁴ exactly — the form a
    /// shader's float storage takes — into `out`, resized to fit.
    pub fn order_f32(&self, out: &mut Vec<f32>) {
        out.clear();
        out.extend(self.order.iter().map(|&i| i as f32));
    }

    /// Order the centres by ascending `axis · centre`, stably. With the view
    /// axis (row 2 of a model-view matrix, along which view-space z grows
    /// towards the camera) that is farthest first. A centre that is not
    /// finite sorts with the farthest (key 0), and so is drawn first.
    pub fn sort(&mut self, axis: [f32; 3]) {
        let n = self.len();
        if n == 0 {
            return;
        }
        // The range the box spans along the axis: each term at whichever end
        // of the box makes it least, or most.
        let (mut lo, mut hi) = (0.0f32, 0.0f32);
        for k in 0..3 {
            let (a, b) = (axis[k] * self.min[k], axis[k] * self.max[k]);
            lo += a.min(b);
            hi += a.max(b);
        }
        let scale = if hi > lo && (hi - lo).is_finite() { MAX / (hi - lo) } else { 0.0 };
        self.make_keys(axis, lo, scale);

        let mut low = vec![0u32; BUCKETS];
        let mut high = vec![0u32; BUCKETS];
        for &k in &self.keys {
            low[(k & MASK) as usize] += 1;
            high[(k >> RADIX) as usize] += 1;
        }
        exclusive_sums(&mut low);
        exclusive_sums(&mut high);

        self.pairs.resize(n, 0);
        for (i, &k) in self.keys.iter().enumerate() {
            let slot = &mut low[(k & MASK) as usize];
            self.pairs[*slot as usize] = (k as u64) << 32 | i as u64;
            *slot += 1;
        }
        self.order.resize(n, 0);
        for &p in &self.pairs {
            let slot = &mut high[(p >> (32 + RADIX)) as usize];
            self.order[*slot as usize] = p as u32;
            *slot += 1;
        }
    }

    /// `keys[i] = ((axis · centre_i − lo) · scale)` clamped to `0..=MAX`, a
    /// non-finite depth to 0.
    fn make_keys(&mut self, axis: [f32; 3], lo: f32, scale: f32) {
        let n = self.len();
        self.keys.resize(n, 0);
        let [ax, ay, az] = axis;
        let key = |x: f32, y: f32, z: f32| {
            let d = ((ax * x + ay * y + az * z) - lo) * scale;
            // Not `clamp`, which keeps a NaN: `max` takes 0 over one. The cast
            // saturates.
            #[allow(clippy::manual_clamp)]
            let k = d.max(0.0).min(MAX) as u32;
            k
        };
        let (kc, kr) = self.keys.as_chunks_mut::<LANES>();
        let (xc, xr) = self.x.as_chunks::<LANES>();
        let (yc, yr) = self.y.as_chunks::<LANES>();
        let (zc, zr) = self.z.as_chunks::<LANES>();
        for (((k, x), y), z) in kc.iter_mut().zip(xc).zip(yc).zip(zc) {
            for l in 0..LANES {
                k[l] = key(x[l], y[l], z[l]);
            }
        }
        for (((k, &x), &y), &z) in kr.iter_mut().zip(xr).zip(yr).zip(zr) {
            *k = key(x, y, z);
        }
    }
}

fn exclusive_sums(counts: &mut [u32]) {
    let mut sum = 0;
    for c in counts {
        let here = *c;
        *c = sum;
        sum += here;
    }
}
