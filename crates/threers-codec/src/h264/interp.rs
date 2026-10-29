//! H.264 luma sub-sample interpolation (§8.4.2.2.1), for the encoder.
//!
//! Half-sample positions come from the 6-tap filter `(1, −5, 20, 20, −5, 1)`:
//! `b` horizontally, `h` vertically, `j` (the centre) vertically over the
//! unrounded horizontal sums. Quarter-sample positions are the rounded
//! averages of their two nearest integer and half samples. Samples outside
//! the picture are its nearest edge sample, as a decoder reads them.
//!
//! The decoder has its own, vectorised copy (`decode::h264::inter`); the
//! encoder's must agree with it bit for bit, which `tests/h264_inter.rs`
//! checks through both threers' decoder and ffmpeg.

/// An 8-bit plane, read with edge clamping.
#[derive(Clone, Copy)]
pub struct PlaneRef<'a> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
}

impl PlaneRef<'_> {
    #[inline(always)]
    fn at(&self, x: isize, y: isize) -> i32 {
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        self.data[y * self.width + x] as i32
    }

    /// The horizontal 6-tap sum at half-sample `(x + ½, y)`, unrounded.
    #[inline(always)]
    fn tap_h(&self, x: isize, y: isize) -> i32 {
        self.at(x - 2, y) - 5 * self.at(x - 1, y) + 20 * self.at(x, y) + 20 * self.at(x + 1, y) - 5 * self.at(x + 2, y) + self.at(x + 3, y)
    }

    #[inline(always)]
    fn tap_v(&self, x: isize, y: isize) -> i32 {
        self.at(x, y - 2) - 5 * self.at(x, y - 1) + 20 * self.at(x, y) + 20 * self.at(x, y + 1) - 5 * self.at(x, y + 2) + self.at(x, y + 3)
    }

    fn b(&self, x: isize, y: isize) -> i32 {
        ((self.tap_h(x, y) + 16) >> 5).clamp(0, 255)
    }

    fn h(&self, x: isize, y: isize) -> i32 {
        ((self.tap_v(x, y) + 16) >> 5).clamp(0, 255)
    }

    fn j(&self, x: isize, y: isize) -> i32 {
        let t = |r: isize| self.tap_h(x, y + r);
        let j1 = t(-2) - 5 * t(-1) + 20 * t(0) + 20 * t(1) - 5 * t(2) + t(3);
        ((j1 + 512) >> 10).clamp(0, 255)
    }

    /// The sample at full position `(x, y)` plus quarter fraction `(xf, yf)`.
    pub fn sample(&self, x: isize, y: isize, xf: usize, yf: usize) -> i32 {
        let avg = |a: i32, b: i32| (a + b + 1) >> 1;
        let g = || self.at(x, y);
        match (xf, yf) {
            (0, 0) => g(),
            (1, 0) => avg(g(), self.b(x, y)),
            (2, 0) => self.b(x, y),
            (3, 0) => avg(self.b(x, y), self.at(x + 1, y)),
            (0, 1) => avg(g(), self.h(x, y)),
            (0, 2) => self.h(x, y),
            (0, 3) => avg(self.h(x, y), self.at(x, y + 1)),
            (2, 2) => self.j(x, y),
            (2, 1) => avg(self.b(x, y), self.j(x, y)),
            (2, 3) => avg(self.j(x, y), self.b(x, y + 1)),
            (1, 2) => avg(self.h(x, y), self.j(x, y)),
            (3, 2) => avg(self.j(x, y), self.h(x + 1, y)),
            // Diagonals: the two nearest half samples.
            (xf, yf) => avg(self.b(x, y + (yf == 3) as isize), self.h(x + (xf == 3) as isize, y)),
        }
    }

    /// A `w × h` block at full position `(x, y)` plus quarter fraction.
    pub fn block(&self, x: isize, y: isize, xf: usize, yf: usize, w: usize, h: usize, out: &mut [i32]) {
        for r in 0..h {
            for c in 0..w {
                out[r * w + c] = self.sample(x + c as isize, y + r as isize, xf, yf);
            }
        }
    }
}

/// A reference picture's half-sample planes, computed once: `b`
/// (horizontal), `h` (vertical) and `j` (centre) for every position, so any
/// quarter sample is a lookup and an average. Positions outside the picture
/// are clamped — past three samples out, every tap reads the edge, so an
/// 8-sample margin holds every value an outside position can have.
pub struct HalfPlanes {
    width: usize,
    height: usize,
    stride: usize,
    g: Vec<u8>,
    b: Vec<u8>,
    h: Vec<u8>,
    j: Vec<u8>,
}

const PAD: isize = 8;

impl HalfPlanes {
    /// From an 8-bit plane, on up to `threads` threads.
    pub fn new(data: &[u8], width: usize, height: usize, threads: usize) -> HalfPlanes {
        let p = PlaneRef { data, width, height };
        let (sw, sh) = (width + 2 * PAD as usize, height + 2 * PAD as usize);
        let mut g = vec![0u8; sw * sh];
        let mut b = vec![0u8; sw * sh];
        let mut h = vec![0u8; sw * sh];
        let mut j = vec![0u8; sw * sh];
        // Unrounded horizontal sums over three more rows each side, for `j`.
        let b1_rows = sh + 5;
        let mut b1 = vec![0i32; sw * b1_rows];
        let rows_each = |n: usize| n.div_ceil(threads.max(1)).max(1);
        let fill = |out: &mut [i32], first_row: usize, f: &(dyn Fn(isize, isize) -> i32 + Sync)| {
            for (k, v) in out.iter_mut().enumerate() {
                let (r, c) = (first_row + k / sw, k % sw);
                *v = f(c as isize - PAD, r as isize);
            }
        };
        let tap_h = |x: isize, r: isize| p.tap_h(x, r - PAD - 2);
        std::thread::scope(|scope| {
            for (i, chunk) in b1.chunks_mut(rows_each(b1_rows) * sw).enumerate() {
                let first = i * rows_each(b1_rows);
                scope.spawn(move || fill(chunk, first, &tap_h));
            }
        });
        let b1 = &b1;
        let n = rows_each(sh);
        std::thread::scope(|scope| {
            for (i, (((gc, bc), hc), jc)) in g.chunks_mut(n * sw).zip(b.chunks_mut(n * sw)).zip(h.chunks_mut(n * sw)).zip(j.chunks_mut(n * sw)).enumerate() {
                scope.spawn(move || {
                    for k in 0..gc.len() {
                        let (r, c) = (i * n + k / sw, k % sw);
                        let (x, y) = (c as isize - PAD, r as isize - PAD);
                        gc[k] = p.at(x, y) as u8;
                        bc[k] = ((b1[(r + 2) * sw + c] + 16) >> 5).clamp(0, 255) as u8;
                        hc[k] = ((p.tap_v(x, y) + 16) >> 5).clamp(0, 255) as u8;
                        let t = |d: usize| b1[(r + d) * sw + c];
                        let j1 = t(0) - 5 * t(1) + 20 * t(2) + 20 * t(3) - 5 * t(4) + t(5);
                        jc[k] = ((j1 + 512) >> 10).clamp(0, 255) as u8;
                    }
                });
            }
        });
        HalfPlanes { width, height, stride: sw, g, b, h, j }
    }

    #[inline(always)]
    fn idx(&self, x: isize, y: isize) -> usize {
        let x = x.clamp(-PAD + 1, self.width as isize + PAD - 2);
        let y = y.clamp(-PAD + 1, self.height as isize + PAD - 2);
        (y + PAD) as usize * self.stride + (x + PAD) as usize
    }

    /// The same as [`PlaneRef::sample`], from the tables.
    #[inline(always)]
    pub fn sample(&self, x: isize, y: isize, xf: usize, yf: usize) -> i32 {
        let avg = |a: u8, b: u8| (a as i32 + b as i32 + 1) >> 1;
        let i = self.idx(x, y);
        let (g, b, h, j) = (&self.g, &self.b, &self.h, &self.j);
        match (xf, yf) {
            (0, 0) => g[i] as i32,
            (1, 0) => avg(g[i], b[i]),
            (2, 0) => b[i] as i32,
            (3, 0) => avg(b[i], g[self.idx(x + 1, y)]),
            (0, 1) => avg(g[i], h[i]),
            (0, 2) => h[i] as i32,
            (0, 3) => avg(h[i], g[self.idx(x, y + 1)]),
            (2, 2) => j[i] as i32,
            (2, 1) => avg(b[i], j[i]),
            (2, 3) => avg(j[i], b[self.idx(x, y + 1)]),
            (1, 2) => avg(h[i], j[i]),
            (3, 2) => avg(j[i], h[self.idx(x + 1, y)]),
            (xf, yf) => avg(b[self.idx(x, y + (yf == 3) as isize)], h[self.idx(x + (xf == 3) as isize, y)]),
        }
    }

    /// A `w × h` block at full position `(x, y)` plus quarter fraction.
    pub fn block(&self, x: isize, y: isize, xf: usize, yf: usize, w: usize, h: usize, out: &mut [i32]) {
        for r in 0..h {
            for c in 0..w {
                out[r * w + c] = self.sample(x + c as isize, y + r as isize, xf, yf);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_match_direct_interpolation_everywhere() {
        let (w, h) = (37usize, 29usize);
        let data: Vec<u8> = (0..w * h).map(|i| ((i * 97 + (i / w) * 31) % 256) as u8).collect();
        let direct = PlaneRef { data: &data, width: w, height: h };
        let tables = HalfPlanes::new(&data, w, h, 3);
        for y in -20..h as isize + 20 {
            for x in -20..w as isize + 20 {
                for xf in 0..4 {
                    for yf in 0..4 {
                        assert_eq!(tables.sample(x, y, xf, yf), direct.sample(x, y, xf, yf), "({x}, {y}) + ({xf}, {yf})/4");
                    }
                }
            }
        }
    }

    #[test]
    fn flat_stays_flat_and_halves_average() {
        let data = vec![100u8; 32 * 32];
        let p = PlaneRef { data: &data, width: 32, height: 32 };
        for xf in 0..4 {
            for yf in 0..4 {
                assert_eq!(p.sample(10, 10, xf, yf), 100, "({xf}, {yf})");
            }
        }
        // A step: the half sample between 0 and 200 is 100.
        let step: Vec<u8> = (0..32 * 32).map(|i| if i % 32 < 16 { 0 } else { 200 }).collect();
        let p = PlaneRef { data: &step, width: 32, height: 32 };
        assert_eq!(p.sample(15, 5, 2, 0), 100);
        assert_eq!(p.sample(15, 5, 0, 2), 0, "vertical half of a vertical edge column");
    }

    #[test]
    fn edges_clamp() {
        let data: Vec<u8> = (0..16 * 16).map(|i| (i % 16 * 10) as u8).collect();
        let p = PlaneRef { data: &data, width: 16, height: 16 };
        assert_eq!(p.sample(-5, -5, 0, 0), 0);
        assert_eq!(p.sample(40, 3, 0, 0), 150);
        let _ = p.sample(-3, 20, 3, 3);
    }
}
