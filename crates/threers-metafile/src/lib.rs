//! Windows metafiles — EMF and the older WMF, the vector pictures Office
//! documents carry (logos, charts, equations pasted from old editors) —
//! read into a [`Drawing`] of paths, text and bitmaps in points, and
//! written out as PDF ([`Drawing::to_pdf`], which CoreGraphics, browsers and
//! printers draw; several as a PDF's pages, [`pdf_pages`]) or SVG
//! ([`Drawing::to_svg`]).
//!
//! What is read: the coordinate mapping (window and viewport, map modes,
//! world transforms), pens, brushes and fonts, lines, polygons, Béziers,
//! rectangles, ellipses, paths, text, and device-independent bitmaps.
//! Clipping, raster operations other than copy, and EMF+ records (the GDI
//! records beside them are drawn instead) are not.
//!
//! ```
//! let emf = threers_metafile::tests_support::sample_emf();
//! let d = threers_metafile::read(&emf).unwrap();
//! assert!(d.width > 0.0 && !d.ops.is_empty());
//! assert!(d.to_pdf().starts_with(b"%PDF"));
//! let two = threers_metafile::pdf_pages(&[d.clone(), d]);
//! assert!(String::from_utf8_lossy(&two).contains("/Count 2"));
//! ```

pub mod symbol;
mod emf;
mod gdi;
mod out;

pub use out::pdf_pages;
mod wmf;

#[doc(hidden)]
pub mod tests_support;

/// `[r, g, b, a]`, 0–255.
pub type Rgba = [u8; 4];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Seg {
    Move(f32, f32),
    Line(f32, f32),
    Cubic(f32, f32, f32, f32, f32, f32),
    Close,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// A path filled and/or stroked (width in points).
    Path { segs: Vec<Seg>, fill: Option<Rgba>, stroke: Option<(Rgba, f32)>, even_odd: bool },
    /// Text from its baseline's left (after alignment), `size` in points,
    /// `angle` in degrees anticlockwise, its characters `stretch` times as
    /// wide as the font has them (1: as they are).
    Text { x: f32, y: f32, text: String, size: f32, color: Rgba, bold: bool, italic: bool, family: String, angle: f32, stretch: f32 },
    /// A bitmap, RGBA rows from the top, put in a rectangle.
    Image { x: f32, y: f32, w: f32, h: f32, width: u32, height: u32, rgba: Vec<u8> },
}

/// A picture: its size in points, what is in it (y down, from the top left).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Drawing {
    pub width: f32,
    pub height: f32,
    pub ops: Vec<Op>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Neither EMF nor WMF.
    NotAMetafile,
    Truncated,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NotAMetafile => f.write_str("not an EMF or WMF metafile"),
            Error::Truncated => f.write_str("the metafile is cut short"),
        }
    }
}

impl std::error::Error for Error {}

/// Whether `bytes` look like a metafile.
pub fn is_metafile(bytes: &[u8]) -> bool {
    emf::is_emf(bytes) || wmf::is_wmf(bytes)
}

/// An EMF or WMF file.
pub fn read(bytes: &[u8]) -> Result<Drawing, Error> {
    if emf::is_emf(bytes) {
        emf::read(bytes)
    } else if wmf::is_wmf(bytes) {
        wmf::read(bytes)
    } else {
        Err(Error::NotAMetafile)
    }
}

/// A little-endian reader that never panics.
#[derive(Clone, Copy)]
pub(crate) struct R<'a>(pub &'a [u8]);

impl R<'_> {
    pub fn u8(&self, at: usize) -> u8 {
        self.0.get(at).copied().unwrap_or(0)
    }
    pub fn u16(&self, at: usize) -> u16 {
        u16::from_le_bytes([self.u8(at), self.u8(at + 1)])
    }
    pub fn i16(&self, at: usize) -> i16 {
        self.u16(at) as i16
    }
    pub fn u32(&self, at: usize) -> u32 {
        u32::from_le_bytes([self.u8(at), self.u8(at + 1), self.u8(at + 2), self.u8(at + 3)])
    }
    pub fn i32(&self, at: usize) -> i32 {
        self.u32(at) as i32
    }
    pub fn f32(&self, at: usize) -> f32 {
        f32::from_bits(self.u32(at))
    }
    pub fn slice(&self, at: usize, len: usize) -> &[u8] {
        let a = at.min(self.0.len());
        let b = at.saturating_add(len).min(self.0.len());
        &self.0[a..b]
    }
}

/// A `COLORREF` (`0x00BBGGRR`) as RGBA.
pub(crate) fn colorref(c: u32) -> Rgba {
    [(c & 0xFF) as u8, (c >> 8 & 0xFF) as u8, (c >> 16 & 0xFF) as u8, 255]
}

/// A device-independent bitmap (a `BITMAPINFO` and its bits) as RGBA rows
/// from the top.
pub(crate) fn dib(info: &[u8], bits: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let r = R(info);
    let header = r.u32(0) as usize;
    if header < 12 {
        return None;
    }
    let (w, h, count, compression, used) = if header == 12 {
        (i32::from(r.i16(4)), i32::from(r.i16(6)), r.u16(10), 0, 0)
    } else {
        (r.i32(4), r.i32(8), r.u16(14), r.u32(16), r.u32(32))
    };
    if w <= 0 || h == 0 || w > 16384 || h.unsigned_abs() > 16384 || !(compression == 0 || compression == 3) {
        return None;
    }
    let (width, height) = (w as u32, h.unsigned_abs());
    let top_down = h < 0;
    let palette_entries = if count <= 8 { if used > 0 { used as usize } else { 1 << count } } else { 0 };
    let entry = if header == 12 { 3 } else { 4 };
    let pal_at = header + if compression == 3 && header == 40 { 12 } else { 0 };
    let palette: Vec<Rgba> = (0..palette_entries)
        .map(|i| {
            let at = pal_at + i * entry;
            [r.u8(at + 2), r.u8(at + 1), r.u8(at), 255]
        })
        .collect();
    let stride = ((width as usize * count as usize + 31) / 32) * 4;
    if bits.len() < stride * (height as usize - 1) {
        return None;
    }
    let b = R(bits);
    let mut out = vec![0u8; (width * height * 4) as usize];
    for row in 0..height as usize {
        let src = if top_down { row } else { height as usize - 1 - row } * stride;
        for x in 0..width as usize {
            let px: Rgba = match count {
                1 => palette.get((b.u8(src + x / 8) >> (7 - x % 8) & 1) as usize).copied().unwrap_or([0, 0, 0, 255]),
                4 => palette.get((b.u8(src + x / 2) >> if x % 2 == 0 { 4 } else { 0 } & 0xF) as usize).copied().unwrap_or([0, 0, 0, 255]),
                8 => palette.get(b.u8(src + x) as usize).copied().unwrap_or([0, 0, 0, 255]),
                16 => {
                    let v = b.u16(src + 2 * x);
                    let (r5, g, b5) = ((v >> 10 & 31) as u32, (v >> 5 & 31) as u32, (v & 31) as u32);
                    [(r5 * 255 / 31) as u8, (g * 255 / 31) as u8, (b5 * 255 / 31) as u8, 255]
                }
                24 => [b.u8(src + 3 * x + 2), b.u8(src + 3 * x + 1), b.u8(src + 3 * x), 255],
                32 => [b.u8(src + 4 * x + 2), b.u8(src + 4 * x + 1), b.u8(src + 4 * x), 255],
                _ => return None,
            };
            let o = (row * width as usize + x) * 4;
            out[o..o + 4].copy_from_slice(&px);
        }
    }
    Some((width, height, out))
}

/// An ellipse in the box, as four Béziers.
pub(crate) fn ellipse(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<Seg> {
    let (cx, cy, rx, ry) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (x1 - x0).abs() / 2.0, (y1 - y0).abs() / 2.0);
    let k = 0.552_284_8;
    vec![
        Seg::Move(cx + rx, cy),
        Seg::Cubic(cx + rx, cy + k * ry, cx + k * rx, cy + ry, cx, cy + ry),
        Seg::Cubic(cx - k * rx, cy + ry, cx - rx, cy + k * ry, cx - rx, cy),
        Seg::Cubic(cx - rx, cy - k * ry, cx - k * rx, cy - ry, cx, cy - ry),
        Seg::Cubic(cx + k * rx, cy - ry, cx + rx, cy - k * ry, cx + rx, cy),
        Seg::Close,
    ]
}

/// A rectangle with rounded corners (`rx`, `ry` the corners' radii).
pub(crate) fn round_rect(x0: f32, y0: f32, x1: f32, y1: f32, rx: f32, ry: f32) -> Vec<Seg> {
    let (x0, x1) = (x0.min(x1), x0.max(x1));
    let (y0, y1) = (y0.min(y1), y0.max(y1));
    let rx = rx.min((x1 - x0) / 2.0).max(0.0);
    let ry = ry.min((y1 - y0) / 2.0).max(0.0);
    if rx == 0.0 || ry == 0.0 {
        return vec![Seg::Move(x0, y0), Seg::Line(x1, y0), Seg::Line(x1, y1), Seg::Line(x0, y1), Seg::Close];
    }
    let k = 0.552_284_8;
    vec![
        Seg::Move(x0 + rx, y0),
        Seg::Line(x1 - rx, y0),
        Seg::Cubic(x1 - rx + k * rx, y0, x1, y0 + ry - k * ry, x1, y0 + ry),
        Seg::Line(x1, y1 - ry),
        Seg::Cubic(x1, y1 - ry + k * ry, x1 - rx + k * rx, y1, x1 - rx, y1),
        Seg::Line(x0 + rx, y1),
        Seg::Cubic(x0 + rx - k * rx, y1, x0, y1 - ry + k * ry, x0, y1 - ry),
        Seg::Line(x0, y0 + ry),
        Seg::Cubic(x0, y0 + ry - k * ry, x0 + rx - k * rx, y0, x0 + rx, y0),
        Seg::Close,
    ]
}

/// An elliptical arc in the box from the ray through `start` to the ray
/// through `end`, anticlockwise on screen (as GDI draws them): its points.
pub(crate) fn arc_points(x0: f32, y0: f32, x1: f32, y1: f32, start: (f32, f32), end: (f32, f32)) -> Vec<(f32, f32)> {
    let (cx, cy, rx, ry) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (x1 - x0).abs() / 2.0, (y1 - y0).abs() / 2.0);
    let a0 = (-(start.1 - cy)).atan2(start.0 - cx);
    let mut a1 = (-(end.1 - cy)).atan2(end.0 - cx);
    if a1 <= a0 {
        a1 += std::f32::consts::TAU;
    }
    let n = (((a1 - a0) / 0.1).ceil() as usize).clamp(2, 400);
    (0..=n)
        .map(|i| {
            let a = a0 + (a1 - a0) * i as f32 / n as f32;
            (cx + rx * a.cos(), cy - ry * a.sin())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_small_emf_is_read_and_written() {
        let d = read(&tests_support::sample_emf()).unwrap();
        assert!((d.width - 72.0).abs() < 0.5, "{}", d.width);
        // a filled red rectangle, a blue line, text
        assert!(d.ops.iter().any(|o| matches!(o, Op::Path { fill: Some([255, 0, 0, 255]), .. })));
        assert!(d.ops.iter().any(|o| matches!(o, Op::Path { stroke: Some(([0, 0, 255, 255], _)), .. })));
        assert!(d.ops.iter().any(|o| matches!(o, Op::Text { text, .. } if text == "Hi")));
        let pdf = d.to_pdf();
        let s = String::from_utf8_lossy(&pdf);
        assert!(s.contains("/MediaBox") && s.contains("/FlateDecode") && s.ends_with("%%EOF\n"));
        // the page's content, inflated: the text drawn
        let start = pdf.windows(7).position(|w| w == b"stream\n").unwrap() + 7;
        let end = start + pdf[start..].windows(10).position(|w| w == b"\nendstream").unwrap();
        let content = miniz_oxide::inflate::decompress_to_vec_zlib(&pdf[start..end]).unwrap();
        assert!(String::from_utf8_lossy(&content).contains("(Hi) Tj"));
        assert!(d.to_svg().contains("<path"));
    }

    #[test]
    fn a_small_wmf_is_read() {
        let d = read(&tests_support::sample_wmf()).unwrap();
        assert!((d.width - 72.0).abs() < 0.5 && (d.height - 36.0).abs() < 0.5, "{} {}", d.width, d.height);
        let rects = d.ops.iter().filter(|o| matches!(o, Op::Path { fill: Some(_), .. })).count();
        assert_eq!(rects, 1);
    }

    #[test]
    fn bitmaps_are_decoded() {
        // 2×2, 24 bits, bottom up: blue red / green white
        let mut info = vec![0u8; 40];
        info[0] = 40;
        info[4] = 2;
        info[8] = 2;
        info[12] = 1;
        info[14] = 24;
        let bits = [0, 255, 0, 255, 255, 255, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0];
        let (w, h, px) = dib(&info, &bits).unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(&px[0..4], &[0, 0, 255, 255]);
        assert_eq!(&px[12..16], &[255, 255, 255, 255]);
    }

    #[test]
    fn rubbish_is_not_a_metafile() {
        assert_eq!(read(b"hello"), Err(Error::NotAMetafile));
        assert!(!is_metafile(&[0; 4]));
    }
}
