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

/// Refuse a picture too large to decode, as an error. A damaged header can
/// claim any size, and allocation failure aborts: on wasm, with memory capped,
/// that takes the whole instance down. No level of AV1, H.264, HEVC or VP9
/// allows more luma samples than this, and the memory for a few such pictures
/// has to be there before anything is sized from the header.
pub(crate) fn check_picture_size(width: u32, height: u32) -> Result<(), String> {
    const MAX_LUMA_SAMPLES: u64 = 35_651_584;
    let luma = width as u64 * height as u64;
    if luma > MAX_LUMA_SAMPLES {
        return Err(format!("{width}x{height} is larger than any level allows"));
    }
    // Four 4:4:4 pictures of 16-bit samples.
    let mut probe = Vec::<u16>::new();
    probe
        .try_reserve_exact(luma as usize * 3 * 4)
        .map_err(|_| format!("not enough memory for {width}x{height} pictures"))
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
        ColorInfo {
            matrix,
            full_range,
            vui_matrix: matrix_coefficients,
        }
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
    #[cfg(feature = "decode")]
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
    /// Large pictures are converted on several threads with `parallel`.
    pub fn to_bgra(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.convert_into::<true>(&mut out);
        out
    }

    /// [`to_bgra`](Self::to_bgra), in RGBA order.
    pub fn to_rgba(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.to_rgba_into(&mut out);
        out
    }

    /// [`to_rgba`](Self::to_rgba) into `out`, replacing what it holds. Reusing
    /// one buffer across frames saves allocating and clearing a whole picture
    /// each time.
    pub fn to_rgba_into(&self, out: &mut Vec<u8>) {
        self.convert_into::<false>(out);
    }

    fn convert_into<const BGR: bool>(&self, out: &mut Vec<u8>) {
        let (w, h) = (self.width as usize, self.height as usize);
        out.resize(w * h * 4, 0);
        if w == 0 || h == 0 {
            return;
        }
        let k = Coefficients::new(self);
        let rows_each = rows_per_band(w, h);
        if rows_each >= h {
            self.convert_rows::<BGR>(0..h, out, &k);
            return;
        }
        let bands: Vec<_> = out.chunks_mut(rows_each * w * 4).enumerate().collect();
        crate::for_each_owned(bands, |(n, band)| {
            let start = n * rows_each;
            self.convert_rows::<BGR>(start..(start + rows_each).min(h), band, &k);
        });
    }

    /// Rows `rows` of the picture into `out`, which holds exactly those rows.
    ///
    /// Chroma is upsampled bilinearly from where the samples sit by default
    /// in H.264 and HEVC (`chroma_sample_loc_type` 0): level with the even
    /// luma columns, and halfway between luma rows for 4:2:0. Each output
    /// pixel's chroma is a weighted sum scaled by 8 — vertical weights 3:1
    /// (or 4), horizontal 1:1 at odd columns (or 2) — so the arithmetic stays
    /// in integers.
    ///
    /// A row's chroma is upsampled first, so the pixel loop has no branches
    /// and vectorises; on wasm, where this runs on one thread beside the
    /// decoder, it is otherwise the slowest stage of decoding to RGBA.
    fn convert_rows<const BGR: bool>(
        &self,
        rows: std::ops::Range<usize>,
        out: &mut [u8],
        k: &Coefficients,
    ) {
        let w = self.width as usize;
        let mono = self.chroma == ChromaFormat::Monochrome;
        let (sw, sh) = self.chroma.subsampling();
        let (cw, ch) = self.chroma_size();
        let (cw, ch) = (cw as usize, ch as usize);
        let mid = k.c_mid;
        // One chroma row, interpolated vertically, scaled by 4, centred on 0.
        let mut ru = vec![0i32; cw];
        let mut rv = vec![0i32; cw];
        // The same across the picture's width, scaled by 8.
        let mut cb = vec![0i32; w];
        let mut cr = vec![0i32; w];
        for (row, y) in rows.enumerate() {
            if !mono {
                let (a, b) = if sh == 2 {
                    let c = (y / 2).min(ch - 1);
                    let other = if y % 2 == 0 {
                        c.saturating_sub(1)
                    } else {
                        (c + 1).min(ch - 1)
                    };
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
                if sw == 2 {
                    upsample_row(&ru, &mut cb);
                    upsample_row(&rv, &mut cr);
                } else {
                    for x in 0..w {
                        cb[x] = 2 * ru[x];
                        cr[x] = 2 * rv[x];
                    }
                }
            }
            let luma = &self.y[y * w..][..w];
            let line = &mut out[row * w * 4..][..w * 4];
            let to8 = |v: i32| ((v + (1 << 15)) >> 16).clamp(0, 255) as u32;
            for (((px, &l), &cb), &cr) in line.chunks_exact_mut(4).zip(luma).zip(&cb).zip(&cr) {
                let yv = k.y * (l as i32 - k.y_off);
                let r = to8(yv + k.rv * cr);
                let g = to8(yv - k.gu * cb - k.gv * cr);
                let b = to8(yv + k.bu * cb);
                let (first, last) = if BGR { (b, r) } else { (r, b) };
                px.copy_from_slice(&(first | g << 8 | last << 16 | 255 << 24).to_le_bytes());
            }
        }
    }
}

/// `row`, sited on even columns, across twice the width: each even column as
/// it is and doubled, each odd one the sum of its neighbours. The picture's
/// last column may be even and past the last pair.
fn upsample_row(row: &[i32], out: &mut [i32]) {
    let last = row.len() - 1;
    for (j, pair) in out.chunks_mut(2).enumerate() {
        pair[0] = 2 * row[j];
        if let Some(odd) = pair.get_mut(1) {
            *odd = row[j] + row[(j + 1).min(last)];
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

/// Rows to convert at a time on the thread pool: enough that a band outweighs
/// handing it over, and all of a small picture.
fn rows_per_band(w: usize, h: usize) -> usize {
    if !crate::threaded() || w * h < 1 << 17 {
        return h;
    }
    ((1 << 16) / w).max(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(
        w: u32,
        h: u32,
        y: Vec<u16>,
        u: Vec<u16>,
        v: Vec<u16>,
        matrix: Matrix,
    ) -> DecodedFrame {
        DecodedFrame {
            width: w,
            height: h,
            bit_depth: 8,
            chroma: ChromaFormat::Yuv420,
            y,
            u,
            v,
            color: ColorInfo {
                matrix,
                full_range: false,
                vui_matrix: None,
            },
        }
    }

    #[test]
    fn flat_colours_convert_like_the_formula() {
        // BT.601 limited: (81, 90, 240) is pure red, (16, 128, 128) black,
        // (235, 128, 128) white.
        for (yuv, rgb) in [
            ((81, 90, 240), [255, 0, 0]),
            ((16, 128, 128), [0, 0, 0]),
            ((235, 128, 128), [255, 255, 255]),
        ] {
            let f = frame(
                4,
                4,
                vec![yuv.0; 16],
                vec![yuv.1; 4],
                vec![yuv.2; 4],
                Matrix::Bt601,
            );
            for px in f.to_rgba().chunks_exact(4) {
                for c in 0..3 {
                    assert!(
                        (px[c] as i32 - rgb[c]).abs() <= 1,
                        "{yuv:?} → {px:?}, want {rgb:?}"
                    );
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
        let f = frame(
            w,
            h,
            vec![128; 8],
            vec![128, 228],
            vec![128, 128],
            Matrix::Bt709,
        );
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
        assert!(rows_per_band(w as usize, h as usize) < h as usize || !crate::threaded());
        let mut single = vec![255u8; n * 4];
        f.convert_rows::<true>(0..h as usize, &mut single, &Coefficients::new(&f));
        assert!(f.to_bgra() == single);
    }

    /// The conversion pixel by pixel, as it reads in the definition: the
    /// reference for the vectorised rows.
    fn reference_rgba(f: &DecodedFrame) -> Vec<u8> {
        let k = Coefficients::new(f);
        let (w, h) = (f.width as usize, f.height as usize);
        let mono = f.chroma == ChromaFormat::Monochrome;
        let (sw, sh) = f.chroma.subsampling();
        let (cw, ch) = f.chroma_size();
        let (cw, ch) = (cw as usize, ch as usize);
        let to8 = |v: i32| ((v + (1 << 15)) >> 16).clamp(0, 255) as u8;
        let mut out = vec![255u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let (mut cb, mut cr) = (0, 0);
                if !mono {
                    let (a, b) = if sh == 2 {
                        let c = (y / 2).min(ch - 1);
                        (
                            c,
                            if y % 2 == 0 {
                                c.saturating_sub(1)
                            } else {
                                (c + 1).min(ch - 1)
                            },
                        )
                    } else {
                        (y.min(ch - 1), y.min(ch - 1))
                    };
                    let (wa, wb) = if sh == 2 { (3, 1) } else { (4, 0) };
                    let vert = |p: &[u16], i: usize| {
                        wa * p[a * cw + i] as i32 + wb * p[b * cw + i] as i32 - 4 * k.c_mid
                    };
                    (cb, cr) = if sw == 2 && x % 2 == 1 {
                        let (j, n) = (x / 2, (x / 2 + 1).min(cw - 1));
                        (vert(&f.u, j) + vert(&f.u, n), vert(&f.v, j) + vert(&f.v, n))
                    } else {
                        let j = if sw == 2 { x / 2 } else { x };
                        (2 * vert(&f.u, j), 2 * vert(&f.v, j))
                    };
                }
                let yv = k.y * (f.y[y * w + x] as i32 - k.y_off);
                let px = &mut out[(y * w + x) * 4..][..3];
                px[0] = to8(yv + k.rv * cr);
                px[1] = to8(yv - k.gu * cb - k.gv * cr);
                px[2] = to8(yv + k.bu * cb);
            }
        }
        out
    }

    #[test]
    fn rows_match_the_per_pixel_definition() {
        let mut x = 0x9e37_79b9u32;
        let mut rnd = move |n: u32| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x % n
        };
        for chroma in [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
            ChromaFormat::Monochrome,
        ] {
            for bit_depth in [8u8, 10, 12] {
                for (w, h) in [(1, 1), (2, 2), (3, 5), (17, 9), (64, 33), (333, 21)] {
                    let mut f = frame(w, h, Vec::new(), Vec::new(), Vec::new(), Matrix::Bt709);
                    (f.chroma, f.bit_depth) = (chroma, bit_depth);
                    let (cw, ch) = f.chroma_size();
                    let max = 1u32 << bit_depth;
                    f.y = (0..w * h).map(|_| rnd(max) as u16).collect();
                    f.u = (0..cw * ch).map(|_| rnd(max) as u16).collect();
                    f.v = (0..cw * ch).map(|_| rnd(max) as u16).collect();
                    let want = reference_rgba(&f);
                    assert!(f.to_rgba() == want, "{chroma:?} {bit_depth}-bit {w}x{h}");
                    let bgr: Vec<u8> = want
                        .chunks(4)
                        .flat_map(|p| [p[2], p[1], p[0], p[3]])
                        .collect();
                    assert!(
                        f.to_bgra() == bgr,
                        "{chroma:?} {bit_depth}-bit {w}x{h} BGRA"
                    );
                }
            }
        }
    }
}
