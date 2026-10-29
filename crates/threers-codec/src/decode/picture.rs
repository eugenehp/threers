//! Sample planes and the decoded frame the decoders hand back.

/// One plane of samples. Stored as `u16` whatever the bit depth, so 8-bit and
/// 10-bit streams share every reconstruction path.
#[derive(Clone, Debug)]
pub struct Plane {
    /// Samples, row by row, `width * height` of them.
    pub data: Vec<u16>,
    /// Width in samples.
    pub width: usize,
    /// Height in samples.
    pub height: usize,
}

impl Plane {
    /// A `width × height` plane with every sample set to `fill`.
    pub fn new(width: usize, height: usize, fill: u16) -> Self {
        Plane {
            data: vec![fill; width * height],
            width,
            height,
        }
    }

    /// The sample at `(x, y)`.
    #[inline(always)]
    pub fn at(&self, x: usize, y: usize) -> u16 {
        self.data[y * self.width + x]
    }

    /// Set the sample at `(x, y)`.
    #[inline(always)]
    pub fn set(&mut self, x: usize, y: usize, v: u16) {
        self.data[y * self.width + x] = v;
    }

    /// The sample at `(x, y)` with the coordinates clamped into the plane —
    /// what motion compensation reads when a vector points off the edge.
    #[inline(always)]
    pub fn clamped(&self, x: isize, y: isize) -> u16 {
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        self.data[y * self.width + x]
    }
}

/// Chroma subsampling of a decoded frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaFormat {
    /// Luma only.
    Monochrome,
    /// Chroma halved both ways.
    Yuv420,
    /// Chroma halved horizontally.
    Yuv422,
    /// Full-resolution chroma.
    Yuv444,
    /// Chroma halved vertically only (VP9 profile 1).
    Yuv440,
}

impl ChromaFormat {
    /// From H.264/HEVC `chroma_format_idc`, 0 to 3.
    pub fn from_idc(idc: u32) -> Option<Self> {
        Some(match idc {
            0 => ChromaFormat::Monochrome,
            1 => ChromaFormat::Yuv420,
            2 => ChromaFormat::Yuv422,
            3 => ChromaFormat::Yuv444,
            _ => return None,
        })
    }

    /// `(SubWidthC, SubHeightC)`.
    pub fn subsampling(self) -> (usize, usize) {
        match self {
            ChromaFormat::Yuv420 => (2, 2),
            ChromaFormat::Yuv422 => (2, 1),
            ChromaFormat::Yuv440 => (1, 2),
            _ => (1, 1),
        }
    }
}

/// The YCbCr → RGB matrix a stream says it was coded with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Matrix {
    /// ITU-R BT.601 (SD).
    Bt601,
    /// ITU-R BT.709 (HD).
    Bt709,
    /// ITU-R BT.2020 (UHD, non-constant luminance).
    Bt2020,
}

/// What the VUI says about colour, reduced to what conversion needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorInfo {
    /// The YCbCr → RGB matrix.
    pub matrix: Matrix,
    /// Full-range (0–255 at 8 bits) rather than video range (16–235).
    pub full_range: bool,
    /// The VUI's `matrix_coefficients` as coded (`None` without a colour
    /// description) — what ffmpeg's swscale keys its tables on.
    pub vui_matrix: Option<u32>,
}

impl ColorInfo {
    /// From the VUI's `matrix_coefficients` and `video_full_range_flag`.
    ///
    /// An unspecified matrix (2, or no VUI at all) is a guess either way;
    /// this makes the same one players do — BT.709 for HD, BT.601 below it.
    pub fn from_vui(matrix_coefficients: Option<u32>, full_range: bool, height: u32) -> Self {
        let matrix = match matrix_coefficients {
            Some(1) => Matrix::Bt709,
            Some(5) | Some(6) => Matrix::Bt601,
            Some(9) | Some(10) => Matrix::Bt2020,
            _ if height >= 720 => Matrix::Bt709,
            _ => Matrix::Bt601,
        };
        ColorInfo { matrix, full_range, vui_matrix: matrix_coefficients }
    }
}

/// A decoded picture, cropped to its display size.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// Display width in pixels.
    pub width: u32,
    /// Display height in pixels.
    pub height: u32,
    /// Bits per sample, 8 to 12.
    pub bit_depth: u8,
    /// Chroma subsampling.
    pub chroma: ChromaFormat,
    /// Luma, `width * height`, tightly packed.
    pub y: Vec<u16>,
    /// Chroma planes, [`chroma_size`](Self::chroma_size) each. Empty for
    /// monochrome.
    pub u: Vec<u16>,
    /// Cr, laid out like `u`.
    pub v: Vec<u16>,
    /// What the stream says about its colour.
    pub color: ColorInfo,
}

impl DecodedFrame {
    /// Size of each chroma plane.
    pub fn chroma_size(&self) -> (u32, u32) {
        if self.chroma == ChromaFormat::Monochrome {
            return (0, 0);
        }
        let (sw, sh) = self.chroma.subsampling();
        (
            self.width.div_ceil(sw as u32),
            self.height.div_ceil(sh as u32),
        )
    }

    /// Crop planes out of the coded picture.
    pub(crate) fn from_planes(
        planes: &[Plane; 3],
        chroma: ChromaFormat,
        bit_depth: u8,
        crop: [usize; 4],
        color: ColorInfo,
    ) -> Self {
        let [left, right, top, bottom] = crop;
        let w = planes[0].width.saturating_sub(left + right);
        let h = planes[0].height.saturating_sub(top + bottom);
        let cut = |p: &Plane, x0: usize, y0: usize, w: usize, h: usize| {
            let mut out = Vec::with_capacity(w * h);
            for y in y0..y0 + h {
                out.extend_from_slice(&p.data[y * p.width + x0..y * p.width + x0 + w]);
            }
            out
        };
        let y = cut(&planes[0], left, top, w, h);
        let (u, v) = if chroma == ChromaFormat::Monochrome {
            (Vec::new(), Vec::new())
        } else {
            let (sw, sh) = chroma.subsampling();
            let (cw, ch) = (w.div_ceil(sw), h.div_ceil(sh));
            (
                cut(&planes[1], left / sw, top / sh, cw, ch),
                cut(&planes[2], left / sw, top / sh, cw, ch),
            )
        };
        DecodedFrame {
            width: w as u32,
            height: h as u32,
            bit_depth,
            chroma,
            y,
            u,
            v,
            color,
        }
    }

    /// The planes as ffmpeg's `rawvideo` writes them for the stream's native
    /// pixel format: one byte a sample at 8 bits, two (little-endian) above.
    pub fn to_raw_yuv(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for plane in [&self.y, &self.u, &self.v] {
            if self.bit_depth <= 8 {
                out.extend(plane.iter().map(|&s| s as u8));
            } else {
                for &s in plane.iter() {
                    out.extend_from_slice(&s.to_le_bytes());
                }
            }
        }
        out
    }

    /// Convert to tightly packed 8-bit BGRA, alpha 255, with the stream's
    /// matrix and range ([`ColorInfo`]) and bilinear chroma upsampling.
    /// Large pictures are converted on several threads (not on `wasm32`).
    pub fn to_bgra(&self) -> Vec<u8> {
        self.to_rgb_order([2, 1, 0])
    }

    /// [`to_bgra`](Self::to_bgra), in RGBA order.
    pub fn to_rgba(&self) -> Vec<u8> {
        self.to_rgb_order([0, 1, 2])
    }

    fn to_rgb_order(&self, order: [usize; 3]) -> Vec<u8> {
        let (w, h) = (self.width as usize, self.height as usize);
        let mut out = vec![255u8; w * h * 4];
        if w == 0 || h == 0 {
            return out;
        }
        let k = Coefficients::new(self);
        let threads = conversion_threads(w, h);
        if threads <= 1 {
            self.convert_rows(0..h, &mut out, order, &k);
            return out;
        }
        let rows_each = h.div_ceil(threads);
        std::thread::scope(|scope| {
            for (n, chunk) in out.chunks_mut(rows_each * w * 4).enumerate() {
                let (k, start) = (&k, n * rows_each);
                let rows = start..(start + rows_each).min(h);
                scope.spawn(move || self.convert_rows(rows, chunk, order, k));
            }
        });
        out
    }

    /// Rows `rows` of the picture into `out`, which holds exactly those rows.
    ///
    /// Chroma is upsampled bilinearly from where the samples sit by default
    /// in H.264 and HEVC (`chroma_sample_loc_type` 0): level with the even
    /// luma columns, and halfway between luma rows for 4:2:0. Each output
    /// pixel's chroma is a weighted sum scaled by 8 — vertical weights 3:1
    /// (or 4), horizontal 1:1 at odd columns (or 2) — so the arithmetic stays
    /// in integers.
    fn convert_rows(&self, rows: std::ops::Range<usize>, out: &mut [u8], order: [usize; 3], k: &Coefficients) {
        let w = self.width as usize;
        let mono = self.chroma == ChromaFormat::Monochrome;
        let (sw, sh) = self.chroma.subsampling();
        let (cw, ch) = self.chroma_size();
        let (cw, ch) = (cw as usize, ch as usize);
        let mid = k.c_mid;
        // One chroma row, interpolated vertically, scaled by 4, centred on 0.
        let mut ru = vec![0i32; cw];
        let mut rv = vec![0i32; cw];
        for (row, y) in rows.enumerate() {
            if !mono {
                let (a, b) = if sh == 2 {
                    let c = (y / 2).min(ch - 1);
                    let other = if y % 2 == 0 { c.saturating_sub(1) } else { (c + 1).min(ch - 1) };
                    (c, other)
                } else {
                    (y.min(ch - 1), y.min(ch - 1))
                };
                let (wa, wb) = if sh == 2 { (3, 1) } else { (4, 0) };
                let (ua, ub) = (&self.u[a * cw..][..cw], &self.u[b * cw..][..cw]);
                let (va, vb) = (&self.v[a * cw..][..cw], &self.v[b * cw..][..cw]);
                for i in 0..cw {
                    ru[i] = wa * ua[i] as i32 + wb * ub[i] as i32 - 4 * mid;
                    rv[i] = wa * va[i] as i32 + wb * vb[i] as i32 - 4 * mid;
                }
            }
            let luma = &self.y[y * w..][..w];
            let line = &mut out[row * w * 4..][..w * 4];
            for (x, (px, &l)) in line.chunks_exact_mut(4).zip(luma).enumerate() {
                let (cb, cr) = if mono {
                    (0, 0)
                } else if sw == 2 {
                    let j = x / 2;
                    if x % 2 == 0 {
                        (2 * ru[j], 2 * rv[j])
                    } else {
                        let n = (j + 1).min(cw - 1);
                        (ru[j] + ru[n], rv[j] + rv[n])
                    }
                } else {
                    (2 * ru[x], 2 * rv[x])
                };
                let yv = k.y * (l as i32 - k.y_off);
                let to8 = |v: i32| ((v + (1 << 15)) >> 16).clamp(0, 255) as u8;
                px[order[0]] = to8(yv + k.rv * cr);
                px[order[1]] = to8(yv - k.gu * cb - k.gv * cr);
                px[order[2]] = to8(yv + k.bu * cb);
            }
        }
    }
}

/// YCbCr → 8-bit RGB in 16.16 fixed point. Chroma coefficients include the
/// ÷8 of [`DecodedFrame::convert_rows`]' interpolation weights.
struct Coefficients {
    y: i32,
    y_off: i32,
    c_mid: i32,
    rv: i32,
    gu: i32,
    gv: i32,
    bu: i32,
}

impl Coefficients {
    fn new(f: &DecodedFrame) -> Self {
        let max = ((1u32 << f.bit_depth) - 1) as f64;
        let s = (1u32 << (f.bit_depth - 8)) as f64;
        // Y in [0, 1] and C in [-0.5, 0.5], then to 0..255.
        let (y_off, y_scale, c_scale) = if f.color.full_range {
            (0.0, 1.0 / max, 1.0 / max)
        } else {
            (16.0 * s, 1.0 / (219.0 * s), 1.0 / (224.0 * s))
        };
        let (kr, kb) = match f.color.matrix {
            Matrix::Bt601 => (0.299, 0.114),
            Matrix::Bt709 => (0.2126, 0.0722),
            Matrix::Bt2020 => (0.2627, 0.0593),
        };
        let kg = 1.0 - kr - kb;
        let fixed = |v: f64| (v * 65536.0).round() as i32;
        let c = 255.0 * c_scale / 8.0;
        Coefficients {
            y: fixed(255.0 * y_scale),
            y_off: y_off as i32,
            c_mid: 1 << (f.bit_depth - 1),
            rv: fixed(2.0 * (1.0 - kr) * c),
            gu: fixed(2.0 * kb * (1.0 - kb) / kg * c),
            gv: fixed(2.0 * kr * (1.0 - kr) / kg * c),
            bu: fixed(2.0 * (1.0 - kb) * c),
        }
    }
}

/// How many threads to convert a `w × h` picture on: one for small
/// pictures, where starting threads costs more than it saves, and always one
/// on `wasm32`, which has none.
fn conversion_threads(w: usize, h: usize) -> usize {
    if cfg!(target_arch = "wasm32") || w * h < 1 << 17 {
        return 1;
    }
    std::thread::available_parallelism().map_or(1, |n| n.get()).min(8).min(h / 16).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: u32, h: u32, y: Vec<u16>, u: Vec<u16>, v: Vec<u16>, matrix: Matrix) -> DecodedFrame {
        DecodedFrame {
            width: w,
            height: h,
            bit_depth: 8,
            chroma: ChromaFormat::Yuv420,
            y,
            u,
            v,
            color: ColorInfo { matrix, full_range: false, vui_matrix: None },
        }
    }

    #[test]
    fn flat_colours_convert_like_the_formula() {
        // BT.601 limited: (81, 90, 240) is pure red, (16, 128, 128) black,
        // (235, 128, 128) white.
        for (yuv, rgb) in [((81, 90, 240), [255, 0, 0]), ((16, 128, 128), [0, 0, 0]), ((235, 128, 128), [255, 255, 255])] {
            let f = frame(4, 4, vec![yuv.0; 16], vec![yuv.1; 4], vec![yuv.2; 4], Matrix::Bt601);
            for px in f.to_rgba().chunks_exact(4) {
                for c in 0..3 {
                    assert!((px[c] as i32 - rgb[c]).abs() <= 1, "{yuv:?} → {px:?}, want {rgb:?}");
                }
                assert_eq!(px[3], 255);
            }
        }
    }

    #[test]
    fn chroma_is_interpolated_between_samples() {
        // Cb steps from 128 to 228 between chroma columns 0 and 1 (luma 0–1
        // and 2–3). Column 1 sits halfway: its blue is between its
        // neighbours', not equal to one of them.
        let (w, h) = (4u32, 2u32);
        let f = frame(w, h, vec![128; 8], vec![128, 228], vec![128, 128], Matrix::Bt709);
        let bgra = f.to_bgra();
        let blue: Vec<u8> = (0..w as usize).map(|x| bgra[x * 4]).collect();
        assert!(blue[0] < blue[1] && blue[1] < blue[2], "{blue:?}");
        assert_eq!(blue[2], blue[3], "past the last sample, held: {blue:?}");
    }

    #[test]
    fn threaded_conversion_matches_one_thread() {
        let (w, h) = (640u32, 480u32);
        let n = (w * h) as usize;
        let y: Vec<u16> = (0..n).map(|i| (i * 7 % 220 + 16) as u16).collect();
        let u: Vec<u16> = (0..n / 4).map(|i| (i * 13 % 224 + 16) as u16).collect();
        let v: Vec<u16> = (0..n / 4).map(|i| (i * 29 % 224 + 16) as u16).collect();
        let f = frame(w, h, y, u, v, Matrix::Bt709);
        assert!(conversion_threads(w as usize, h as usize) > 1 || cfg!(target_arch = "wasm32") || std::thread::available_parallelism().map_or(true, |p| p.get() == 1));
        let mut single = vec![255u8; n * 4];
        f.convert_rows(0..h as usize, &mut single, [2, 1, 0], &Coefficients::new(&f));
        assert!(f.to_bgra() == single);
    }
}
