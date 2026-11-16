//! A face at a size, as `FT_Set_Char_Size` makes it: the 16.16 scales from
//! font units to 26.6 pixels, the pixels per em, and the scaled metrics.

use crate::fixed::{div_fix, mul_fix, pix_ceil, pix_floor, pix_round};
use crate::sfnt::Face;

/// `FT_Size_Metrics`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizeMetrics {
    pub x_ppem: u16,
    pub y_ppem: u16,
    pub x_scale: i64,
    pub y_scale: i64,
    pub ascender: i64,
    pub descender: i64,
    pub height: i64,
    pub max_advance: i64,
}

/// A face's size: the metrics a caller sees, and those hinting uses
/// (whole pixels per em for a face that asks for them).
#[derive(Clone, Copy, Debug)]
pub struct Size {
    pub metrics: SizeMetrics,
    pub hinted: SizeMetrics,
    /// The resolution of the axis with more pixels per em, for the point
    /// size `MPS` gives.
    pub resolution: u32,
}

/// The face's line metrics in font units as FreeType reads them: `hhea`,
/// else the `OS/2` typographic ones, else the Windows ones.
pub fn face_line_metrics(face: &Face) -> (i64, i64, i64) {
    let (mut asc, mut desc, mut gap) = (i64::from(face.ascender), i64::from(face.descender), i64::from(face.line_gap));

    if asc == 0 && desc == 0
        && let Some(os2) = face.os2 {
            if os2.typo_ascender != 0 || os2.typo_descender != 0 {
                asc = i64::from(os2.typo_ascender);
                desc = i64::from(os2.typo_descender);
                gap = i64::from(os2.typo_line_gap);
            } else {
                asc = i64::from(os2.win_ascent);
                desc = -i64::from(os2.win_descent);
                gap = 0;
            }
        }

    (asc, desc, asc - desc + gap)
}

impl Size {
    /// `FT_Set_Char_Size( face, width, height, hres, vres )` with sizes in
    /// 26.6 points; zero sizes and resolutions take the other's, both zero
    /// resolutions 72 dpi.
    pub fn char_size(face: &Face, mut width: i64, mut height: i64, mut hres: u32, mut vres: u32) -> Size {
        if width == 0 {
            width = height;
        } else if height == 0 {
            height = width;
        }

        if hres == 0 {
            hres = vres;
        } else if vres == 0 {
            vres = hres;
        }

        if hres == 0 {
            hres = 72;
            vres = 72;
        }

        // FT_REQUEST_WIDTH / HEIGHT
        let scaled_w = (width * i64::from(hres) + 36) / 72;
        let scaled_h = (height * i64::from(vres) + 36) / 72;
        let upem = i64::from(face.units_per_em);

        let x_scale = div_fix(scaled_w, upem);
        let y_scale = div_fix(scaled_h, upem);
        let x_ppem = ((mul_fix(upem, x_scale) + 32) >> 6) as u16;
        let y_ppem = ((mul_fix(upem, y_scale) + 32) >> 6) as u16;

        let (asc, desc, line_height) = face_line_metrics(face);
        let max_advance = i64::from(face.advance_width_max);

        // ft_recompute_scaled_metrics
        let metrics = SizeMetrics {
            x_ppem,
            y_ppem,
            x_scale,
            y_scale,
            ascender: pix_ceil(mul_fix(asc, y_scale)),
            descender: pix_floor(mul_fix(desc, y_scale)),
            height: pix_round(mul_fix(line_height, y_scale)),
            max_advance: pix_round(mul_fix(max_advance, x_scale)),
        };

        // tt_size_reset: a face that wants whole pixels per em is hinted at
        // them
        let mut hinted = metrics;

        if face.head_flags & 8 != 0 {
            hinted.ascender = pix_round(mul_fix(asc, y_scale));
            hinted.descender = pix_round(mul_fix(desc, y_scale));
            hinted.height = pix_round(mul_fix(line_height, y_scale));
            hinted.x_scale = div_fix(i64::from(x_ppem) << 6, upem);
            hinted.y_scale = div_fix(i64::from(y_ppem) << 6, upem);
        }

        let resolution = if x_ppem > y_ppem { hres } else { vres };
        Size { metrics, hinted, resolution }
    }
}
