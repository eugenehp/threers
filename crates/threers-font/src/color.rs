//! Colour glyphs, in every form a face may have them:
//!
//! - **`sbix`** (Apple Color Emoji): a PNG for each glyph at a few sizes.
//! - **`CBDT`/`CBLC`** (Noto Color Emoji): the same as embedded bitmaps.
//! - **`COLR`/`CPAL`** version 0 (Segoe UI Emoji, icon fonts): a glyph as
//!   other glyphs' outlines layered, each in a colour of the palette (or
//!   the text's own).
//!
//! [`glyph`] says which a glyph is, if any. (`COLR` version 1's gradients
//! and transforms and the `SVG ` table are not read: a glyph with only
//! those is drawn from its outline, where it has one.)
//!
//! ```no_run
//! # let data = std::fs::read("/System/Library/Fonts/Apple Color Emoji.ttc").unwrap();
//! let face = threers_font::Face::new(data, 0).unwrap();
//! match threers_font::color::glyph(&face, 1000, 64.0) {
//!     Some(threers_font::color::Color::Bitmap(b)) => { /* b.png at b.ppem, bottom left (b.x, b.y) */ }
//!     Some(threers_font::color::Color::Layers(ls)) => { /* each outline in its colour */ }
//!     None => { /* its outline, in the text's colour */ }
//! }
//! ```

use crate::Face;

/// A glyph's picture: PNG bytes made for `ppem` pixels an em, whose bottom
/// left is `x` right of the pen and `y` above the baseline, in those pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bitmap<'a> {
    pub ppem: u16,
    pub x: i16,
    pub y: i16,
    pub png: &'a [u8],
}

/// One layer of a layered glyph: another glyph's outline, in a colour
/// (`[r, g, b, a]`), or `None`: the text's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layer {
    pub glyph: u16,
    pub color: Option<[u8; 4]>,
}

/// How a glyph is coloured.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Color<'a> {
    Bitmap(Bitmap<'a>),
    /// Drawn in order, the first at the bottom.
    Layers(Vec<Layer>),
}

/// `glyph` as its face colours it at `ppem` pixels an em: its layers, else
/// its picture; `None` for a glyph drawn from its outline alone.
pub fn glyph(face: &Face, glyph: u16, ppem: f32) -> Option<Color<'_>> {
    if let Some(colr) = face.table(b"COLR")
        && let Some(layers) = colr_layers(colr, face.table(b"CPAL"), glyph)
    {
        return Some(Color::Layers(layers));
    }
    sbix(face, glyph, ppem).or_else(|| cbdt(face.table(b"CBLC")?, face.table(b"CBDT")?, glyph, ppem)).map(Color::Bitmap)
}

fn u8_at(d: &[u8], at: usize) -> Option<u8> {
    d.get(at).copied()
}

fn u16_at(d: &[u8], at: usize) -> Option<u16> {
    d.get(at..at + 2).map(|b| u16::from_be_bytes([b[0], b[1]]))
}

fn u32_at(d: &[u8], at: usize) -> Option<u32> {
    d.get(at..at + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

/// Of sizes, the nearest at or above `ppem`, else the largest.
fn nearest<T: Copy>(mut sizes: Vec<(u16, T)>, ppem: f32) -> Option<(u16, T)> {
    sizes.sort_by_key(|s| s.0);
    sizes.iter().find(|s| f32::from(s.0) >= ppem).or(sizes.last()).copied()
}

/// `glyph`'s picture in the face's `sbix`, from the strike nearest above
/// `ppem` (else the largest). `None` where the face has none for it (or
/// only another kind than PNG).
pub fn sbix<'a>(face: &'a Face, glyph: u16, ppem: f32) -> Option<Bitmap<'a>> {
    let t = face.table(b"sbix")?;
    sbix_in(t, face.num_glyphs, glyph, ppem)
}

fn sbix_in(t: &[u8], num_glyphs: u16, glyph: u16, ppem: f32) -> Option<Bitmap<'_>> {
    let strikes = u32_at(t, 4)? as usize;
    let all: Vec<(u16, usize)> = (0..strikes.min(64))
        .filter_map(|i| {
            let at = u32_at(t, 8 + 4 * i)? as usize;
            Some((u16_at(t, at)?, at))
        })
        .collect();
    let (size, strike) = nearest(all, ppem)?;
    if glyph >= num_glyphs {
        return None;
    }
    let g = usize::from(glyph);
    let start = strike + u32_at(t, strike + 4 + 4 * g)? as usize;
    let end = strike + u32_at(t, strike + 4 + 4 * (g + 1))? as usize;
    if end < start + 8 || t.get(start + 4..start + 8)? != b"png " {
        return None;
    }
    let x = u16_at(t, start)? as i16;
    let y = u16_at(t, start + 2)? as i16;
    Some(Bitmap { ppem: size, x, y, png: t.get(start + 8..end)? })
}

/// `glyph`'s picture in `CBDT` (found through `CBLC`), from the size
/// nearest above `ppem`: PNG images only (formats 17, 18, 19).
pub fn cbdt<'a>(cblc: &[u8], cbdt: &'a [u8], glyph: u16, ppem: f32) -> Option<Bitmap<'a>> {
    let sizes = u32_at(cblc, 4)? as usize;
    // each size: its ppem, and where its glyphs are
    let all: Vec<(u16, usize)> = (0..sizes.min(64))
        .filter_map(|i| {
            let rec = 8 + 48 * i;
            let (start, end) = (u16_at(cblc, rec + 40)?, u16_at(cblc, rec + 42)?);
            (start..=end).contains(&glyph).then_some((u16::from(u8_at(cblc, rec + 45)?), rec))
        })
        .collect();
    let (ppem, rec) = nearest(all, ppem)?;
    let array = u32_at(cblc, rec)? as usize;
    let tables = u32_at(cblc, rec + 8)? as usize;
    for k in 0..tables {
        let e = array + 8 * k;
        let (first, last) = (u16_at(cblc, e)?, u16_at(cblc, e + 2)?);
        if !(first..=last).contains(&glyph) {
            continue;
        }
        let sub = array + u32_at(cblc, e + 4)? as usize;
        let (index_format, image_format, data) = (u16_at(cblc, sub)?, u16_at(cblc, sub + 2)?, u32_at(cblc, sub + 4)? as usize);
        let i = usize::from(glyph - first);
        // where the glyph's data is in CBDT, and its metrics if the index has them
        let (at, big): (usize, Option<usize>) = match index_format {
            1 => (data + u32_at(cblc, sub + 8 + 4 * i)? as usize, None),
            3 => (data + usize::from(u16_at(cblc, sub + 8 + 2 * i)?), None),
            2 => (data + u32_at(cblc, sub + 8)? as usize * i, Some(sub + 12)),
            4 | 5 => {
                let (n, ids) = if index_format == 4 { (u32_at(cblc, sub + 8)? as usize, sub + 12) } else { (u32_at(cblc, sub + 20)? as usize, sub + 24) };
                if index_format == 4 {
                    // pairs of (glyph, offset)
                    let j = (0..n).find(|&j| u16_at(cblc, ids + 4 * j) == Some(glyph))?;
                    (data + usize::from(u16_at(cblc, ids + 4 * j + 2)?), None)
                } else {
                    let j = (0..n).find(|&j| u16_at(cblc, ids + 2 * j) == Some(glyph))?;
                    (data + u32_at(cblc, sub + 8)? as usize * j, Some(sub + 12))
                }
            }
            _ => return None,
        };
        // the metrics (height, width, bearing x, bearing y), then the PNG's length and bytes
        let (h, w, bx, by, png_at) = match image_format {
            17 => (u8_at(cbdt, at)?, u8_at(cbdt, at + 1)?, u8_at(cbdt, at + 2)? as i8, u8_at(cbdt, at + 3)? as i8, at + 5),
            18 => (u8_at(cbdt, at)?, u8_at(cbdt, at + 1)?, u8_at(cbdt, at + 2)? as i8, u8_at(cbdt, at + 3)? as i8, at + 8),
            19 => {
                let m = big?;
                (u8_at(cblc, m)?, u8_at(cblc, m + 1)?, u8_at(cblc, m + 2)? as i8, u8_at(cblc, m + 3)? as i8, at)
            }
            _ => return None,
        };
        let _ = w;
        let len = u32_at(cbdt, png_at)? as usize;
        let png = cbdt.get(png_at + 4..png_at + 4 + len)?;
        // the bearing's y is to the top: the bottom is the height below it
        return Some(Bitmap { ppem, x: i16::from(bx), y: i16::from(by) - i16::from(h), png });
    }
    None
}

/// `glyph`'s layers in `COLR` version 0, their colours from `CPAL`'s first
/// palette (`0xFFFF`, or no `CPAL`: the text's colour).
pub fn colr_layers(colr: &[u8], cpal: Option<&[u8]>, glyph: u16) -> Option<Vec<Layer>> {
    let bases = usize::from(u16_at(colr, 2)?);
    let base_at = u32_at(colr, 4)? as usize;
    let layer_at = u32_at(colr, 8)? as usize;
    // the base glyph records are sorted: a binary search
    let (mut lo, mut hi) = (0usize, bases);
    let rec = loop {
        if lo >= hi {
            return None;
        }
        let mid = (lo + hi) / 2;
        let g = u16_at(colr, base_at + 6 * mid)?;
        match g.cmp(&glyph) {
            std::cmp::Ordering::Equal => break base_at + 6 * mid,
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
        }
    };
    let (first, n) = (usize::from(u16_at(colr, rec + 2)?), usize::from(u16_at(colr, rec + 4)?));
    let colour = |index: u16| -> Option<[u8; 4]> {
        if index == 0xFFFF {
            return None;
        }
        let cpal = cpal?;
        let records = u32_at(cpal, 8)? as usize;
        let start = usize::from(u16_at(cpal, 12)?);
        let at = records + 4 * (start + usize::from(index));
        // stored blue, green, red, alpha
        let b = cpal.get(at..at + 4)?;
        Some([b[2], b[1], b[0], b[3]])
    };
    let layers = (0..n)
        .filter_map(|i| {
            let at = layer_at + 4 * (first + i);
            Some(Layer { glyph: u16_at(colr, at)?, color: colour(u16_at(colr, at + 2)?) })
        })
        .collect::<Vec<_>>();
    (!layers.is_empty()).then_some(layers)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn be16(v: &mut Vec<u8>, x: u16) {
        v.extend_from_slice(&x.to_be_bytes());
    }

    fn be32(v: &mut Vec<u8>, x: u32) {
        v.extend_from_slice(&x.to_be_bytes());
    }

    #[test]
    fn apple_color_emoji_has_pngs() {
        let Ok(data) = std::fs::read("/System/Library/Fonts/Apple Color Emoji.ttc") else { return };
        let face = Face::new(data, 0).unwrap();
        let g = face.glyph_index('😀' as u32);
        assert!(g != 0);
        let Some(Color::Bitmap(b)) = glyph(&face, g, 40.0) else { panic!("a picture") };
        assert!(b.ppem >= 40 && b.png.starts_with(b"\x89PNG"));
    }

    #[test]
    fn colr_layers_take_cpal_colours_and_the_texts() {
        // glyph 5 as glyphs 6 (palette 1) and 7 (the text's colour)
        let mut colr = Vec::new();
        be16(&mut colr, 0); // version
        be16(&mut colr, 1); // base glyphs
        be32(&mut colr, 14); // base records at
        be32(&mut colr, 20); // layer records at
        be16(&mut colr, 2); // layers
        be16(&mut colr, 5);
        be16(&mut colr, 0);
        be16(&mut colr, 2);
        be16(&mut colr, 6);
        be16(&mut colr, 1);
        be16(&mut colr, 7);
        be16(&mut colr, 0xFFFF);
        let mut cpal = Vec::new();
        be16(&mut cpal, 0);
        be16(&mut cpal, 2); // entries
        be16(&mut cpal, 1); // palettes
        be16(&mut cpal, 2); // colour records
        be32(&mut cpal, 14); // records at
        be16(&mut cpal, 0); // palette 0 starts at record 0
        cpal.extend_from_slice(&[0, 0, 255, 255]); // red (BGRA)
        cpal.extend_from_slice(&[255, 0, 0, 128]); // blue, half
        let ls = colr_layers(&colr, Some(&cpal), 5).unwrap();
        assert_eq!(ls, vec![Layer { glyph: 6, color: Some([0, 0, 255, 128]) }, Layer { glyph: 7, color: None }]);
        assert!(colr_layers(&colr, Some(&cpal), 4).is_none());
    }

    #[test]
    fn cbdt_finds_a_png_by_its_index() {
        // one size (109 ppem) for glyphs 3..=4, index format 1, image format 17
        let png = b"\x89PNGfake";
        let mut cbdt = Vec::new();
        be32(&mut cbdt, 0x0003_0000);
        let g3 = cbdt.len();
        cbdt.extend_from_slice(&[100, 90, 2, 88, 100]); // height, width, bearing x, bearing y, advance
        be32(&mut cbdt, png.len() as u32);
        cbdt.extend_from_slice(png);
        let g4 = cbdt.len();
        let mut cblc = Vec::new();
        be32(&mut cblc, 0x0003_0000);
        be32(&mut cblc, 1); // sizes
        let array = 8 + 48;
        be32(&mut cblc, array as u32);
        be32(&mut cblc, 0);
        be32(&mut cblc, 1); // index tables
        be32(&mut cblc, 0);
        cblc.extend_from_slice(&[0; 24]); // line metrics
        be16(&mut cblc, 3);
        be16(&mut cblc, 4);
        cblc.extend_from_slice(&[109, 109, 32, 1]);
        // the index array, then its one table
        be16(&mut cblc, 3);
        be16(&mut cblc, 4);
        be32(&mut cblc, 8);
        be16(&mut cblc, 1); // index format
        be16(&mut cblc, 17); // image format
        be32(&mut cblc, 0); // image data at
        be32(&mut cblc, g3 as u32);
        be32(&mut cblc, g4 as u32);
        be32(&mut cblc, g4 as u32);
        let b = super::cbdt(&cblc, &cbdt, 3, 20.0).unwrap();
        assert_eq!((b.ppem, b.x, b.y, b.png), (109, 2, 88 - 100, &png[..]));
        assert!(super::cbdt(&cblc, &cbdt, 9, 20.0).is_none());
    }
}
