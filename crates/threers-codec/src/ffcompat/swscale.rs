//! Port of ffmpeg n7.1 libswscale's C `yuv2rgb` (`yuv2rgb_c_24_rgb`), the
//! converter ffmpeg uses for moviepy's `-pix_fmt rgb24` from 8-bit 4:2:0 and
//! 4:2:2 at even output heights on
//! CPUs without an accelerated path ("No accelerated colorspace conversion
//! found from yuv420p to rgb24", e.g. Apple Silicon). Chroma is nearest
//! (one sample per 2x2 block); each channel is a lookup `y_table[Y + off]`
//! with integer offsets from `ff_yuv2rgb_c_init_tables`.

/// `ff_yuv2rgb_coeffs[colorspace]` (crv, cbu, cgu, cgv).
pub fn coefficients(colorspace: u32) -> [i64; 4] {
    match colorspace {
        1 => [117489, 138438, 13975, 34925],         // BT.709
        4 => [104448, 132798, 24759, 53109],         // FCC
        7 => [117579, 136230, 16907, 35559],         // SMPTE 240M
        9 | 10 => [110013, 140363, 12277, 42626],    // BT.2020
        _ => [104597, 132201, 25675, 53279],         // BT.601 / default
    }
}

const HEADROOM: i64 = 512;
const LUMA_HEADROOM: i64 = 512;

fn clip_u8(v: i64) -> i64 {
    v.clamp(0, 255)
}

/// Tables for one (colorspace, range); brightness 0, contrast and saturation 1.
pub struct Yuv2Rgb {
    pub(crate) y_table: Vec<u8>,
    pub(crate) r_v: Vec<i64>,
    pub(crate) g_u: Vec<i64>,
    pub(crate) g_v: Vec<i64>,
    pub(crate) b_u: Vec<i64>,
    /// `yuv2rgb_{y_coeff, y_offset, v2r, v2g, u2g, u2b}` for the
    /// full-chroma output functions.
    pub(crate) full: [i32; 6],
}

/// `roundToInt16` then the `(int16_t)` cast.
fn round_to_i16(f: i64) -> i32 {
    let r = (f + (1 << 15)) >> 16;
    let u: u16 = if r < -0x7fff {
        0x8000
    } else if r > 0x7fff {
        0x7fff
    } else {
        r as u16
    };
    u as i16 as i32
}

impl Yuv2Rgb {
    /// swscale's YUV → RGB tables for a VUI `matrix_coefficients` value
    /// (`colorspace`) and range.
    pub fn new(colorspace: u32, full_range: bool) -> Self {
        let inv = coefficients(colorspace);
        let (contrast, saturation, brightness) = (1i64 << 16, 1i64 << 16, 0i64);
        let yoffs = if full_range { 384 } else { 326 } + LUMA_HEADROOM;
        let table_plane_size = 1024 + 2 * LUMA_HEADROOM;
        let (mut crv, mut cbu, mut cgu, mut cgv) = (inv[0], inv[1], -inv[2], -inv[3]);
        let (mut cy, mut oy) = (1i64 << 16, 0i64);
        if !full_range {
            cy = (cy * 255) / 219;
            oy = 16 << 16;
        } else {
            crv = (crv * 224) / 255;
            cbu = (cbu * 224) / 255;
            cgu = (cgu * 224) / 255;
            cgv = (cgv * 224) / 255;
        }
        cy = (cy * contrast) >> 16;
        crv = (crv * contrast * saturation) >> 32;
        cbu = (cbu * contrast * saturation) >> 32;
        cgu = (cgu * contrast * saturation) >> 32;
        cgv = (cgv * contrast * saturation) >> 32;
        oy -= 256 * brightness;
        let full = [
            round_to_i16(cy * (1 << 13)),
            round_to_i16(oy * (1 << 9)),
            round_to_i16(crv * (1 << 13)),
            round_to_i16(cgv * (1 << 13)),
            round_to_i16(cgu * (1 << 13)),
            round_to_i16(cbu * (1 << 13)),
        ];
        // Scale by cy (C truncating division).
        let d = cy.max(1);
        crv = ((crv * (1 << 16)) + 0x8000) / d;
        cbu = ((cbu * (1 << 16)) + 0x8000) / d;
        cgu = ((cgu * (1 << 16)) + 0x8000) / d;
        cgv = ((cgv * (1 << 16)) + 0x8000) / d;

        let mut yb = -(384i64 << 16) - LUMA_HEADROOM * cy - oy;
        let mut y_table = Vec::with_capacity(table_plane_size as usize);
        for _ in 0..table_plane_size {
            y_table.push(clip_u8((yb + 0x8000) >> 16) as u8);
            yb += cy;
        }
        let n = (256 + 2 * HEADROOM) as usize;
        // fill_table: offsets into y_table of `y_table + yoffs - (inc >> 9) + (cb >> 16)`.
        let fill = |inc: i64| -> Vec<i64> {
            (0..n as i64)
                .map(|i| yoffs - (inc >> 9) + ((clip_u8(i - HEADROOM) * inc) >> 16))
                .collect()
        };
        let fill_gv = |inc: i64| -> Vec<i64> {
            (0..n as i64).map(|i| -(inc >> 9) + ((clip_u8(i - HEADROOM) * inc) >> 16)).collect()
        };
        Self { y_table, r_v: fill(crv), g_u: fill(cgu), g_v: fill_gv(cgv), b_u: fill(cbu), full }
    }

    /// Convert 8-bit 4:2:0 planes (tightly packed, even `w` and `h`) to rgb24.
    pub fn convert_420(&self, y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8> {
        let cw = w.div_ceil(2);
        let mut out = vec![0u8; w * h * 3];
        let h0 = HEADROOM as usize;
        for yy in (0..h - h % 2).step_by(2) {
            for x in (0..w - w % 2).step_by(2) {
                let ci = (yy / 2) * cw + x / 2;
                let (uu, vv) = (u[ci] as usize + h0, v[ci] as usize + h0);
                let r = self.r_v[vv];
                let g = self.g_u[uu] + self.g_v[vv];
                let b = self.b_u[uu];
                for (row, col) in [(yy, x), (yy, x + 1), (yy + 1, x), (yy + 1, x + 1)] {
                    let yv = y[row * w + col] as i64;
                    let o = (row * w + col) * 3;
                    out[o] = self.y_table[(r + yv) as usize];
                    out[o + 1] = self.y_table[(g + yv) as usize];
                    out[o + 2] = self.y_table[(b + yv) as usize];
                }
            }
        }
        out
    }

    /// The same converter's 8-bit 4:2:2 variant (`h` even): every row uses
    /// its own chroma row (`pu_2` for the second line of each pair).
    pub fn convert_422(&self, y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8> {
        let cw = w.div_ceil(2);
        let mut out = vec![0u8; w * h * 3];
        let h0 = HEADROOM as usize;
        for yy in 0..h - h % 2 {
            for x in (0..w - w % 2).step_by(2) {
                let ci = yy * cw + x / 2;
                let (uu, vv) = (u[ci] as usize + h0, v[ci] as usize + h0);
                let r = self.r_v[vv];
                let g = self.g_u[uu] + self.g_v[vv];
                let b = self.b_u[uu];
                for (row, col) in [(yy, x), (yy, x + 1)] {
                    let yv = y[row * w + col] as i64;
                    let o = (row * w + col) * 3;
                    out[o] = self.y_table[(r + yv) as usize];
                    out[o + 1] = self.y_table[(g + yv) as usize];
                    out[o + 2] = self.y_table[(b + yv) as usize];
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grey_maps_to_grey_full_range() {
        let t = Yuv2Rgb::new(1, true);
        let (w, h) = (4, 2);
        let y = vec![0u8, 64, 128, 255, 10, 20, 30, 40];
        let rgb = t.convert_420(&y, &[128, 128], &[128, 128], w, h);
        for (i, &yv) in y.iter().enumerate() {
            assert_eq!(&rgb[i * 3..i * 3 + 3], &[yv, yv, yv], "Y {yv}");
        }
    }
}
