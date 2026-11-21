//! The faces drawings are lettered in ([`DrawingFonts`]), how a run of text
//! is laid out in one, and the subset of a face a PDF embeds.
//!
//! Text is laid out once, here, for every format that draws glyphs: each
//! character's glyph (the face's `?` where it has none), placed by advance
//! widths and pair kerning. [`render_svg`](super::rasterize_svg) fills those
//! glyphs' outlines and [`write_pdf`](super::write_pdf) embeds the same face
//! and shows the same glyphs at the same places, so a label in the PNG and
//! the PDF is the same label, anchored the same.
//!
//! A family name picks a face: one registered under that exact name (case
//! aside), else the generic family it names (`mono`/`courier` monospace;
//! `serif`/`times` without `sans` serif; anything else sans-serif). Every
//! generic family is DejaVu Sans, which threers carries, unless a face is
//! registered for it or [`DrawingFonts::system`] found one, so output does
//! not depend on the machine unless asked to.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use crate::loaders::TtfFont;

/// A TrueType face: its file and what parsing it gave.
pub struct Face {
    bytes: Arc<[u8]>,
    /// Where the face's table directory starts (non-zero in a collection).
    offset: usize,
    pub(crate) ttf: TtfFont,
    /// The PostScript name (`name` ID 6), for the PDF's font name.
    pub(crate) postscript_name: String,
}

impl std::fmt::Debug for Face {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Face").field("postscript_name", &self.postscript_name).field("bytes", &self.bytes.len()).finish()
    }
}

impl Face {
    /// The `index`th face of a TrueType file or collection. `None` when it is
    /// not one this reads, or has no `glyf` outlines (CFF faces cannot be
    /// embedded as TrueType).
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>, index: u32) -> Option<Face> {
        let bytes: Arc<[u8]> = bytes.into();
        let offset = if bytes.get(..4)? == b"ttcf" { be32(&bytes, 12 + 4 * index as usize)? as usize } else { 0 };
        let ttf = TtfFont::parse_face(&bytes, index, &[]).ok()?;
        let mut face = Face { bytes, offset, ttf, postscript_name: String::new() };
        face.table(b"glyf")?;
        face.table(b"loca")?;
        face.postscript_name = face.read_postscript_name().unwrap_or_else(|| "Font".into());
        Some(face)
    }

    /// A table's bytes.
    pub(crate) fn table(&self, tag: &[u8; 4]) -> Option<&[u8]> {
        let b = &self.bytes;
        let n = be16(b, self.offset + 4)? as usize;
        (0..n).find_map(|k| {
            let rec = self.offset + 12 + 16 * k;
            (b.get(rec..rec + 4)? == tag).then(|| {
                let (off, len) = (be32(b, rec + 8)? as usize, be32(b, rec + 12)? as usize);
                b.get(off..off + len)
            })?
        })
    }

    fn read_postscript_name(&self) -> Option<String> {
        let t = self.table(b"name")?;
        let (count, strings) = (be16(t, 2)? as usize, be16(t, 4)? as usize);
        let mut mac = None;
        for k in 0..count {
            let r = 6 + 12 * k;
            let (platform, id) = (be16(t, r)?, be16(t, r + 6)?);
            if id != 6 {
                continue;
            }
            let (len, off) = (be16(t, r + 8)? as usize, be16(t, r + 10)? as usize);
            let raw = t.get(strings + off..strings + off + len)?;
            if platform == 3 {
                let units: Vec<u16> = raw.as_chunks::<2>().0.iter().map(|&c| u16::from_be_bytes(c)).collect();
                return Some(sanitize(&String::from_utf16_lossy(&units)));
            }
            if platform == 1 {
                mac = Some(sanitize(&raw.iter().map(|&c| c as char).collect::<String>()));
            }
        }
        mac
    }

    pub(crate) fn units_per_em(&self) -> f64 {
        self.ttf.units_per_em.max(1) as f64
    }
}

/// A PostScript name keeps printable ASCII without spaces or delimiters.
fn sanitize(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_graphic() && !"[](){}<>/%".contains(*c)).collect()
}

fn be16(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*b.get(at)?, *b.get(at + 1)?]))
}

fn be32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes([*b.get(at)?, *b.get(at + 1)?, *b.get(at + 2)?, *b.get(at + 3)?]))
}

/// DejaVu Sans, which threers carries, parsed once.
pub(crate) fn bundled() -> Arc<Face> {
    static FACE: OnceLock<Arc<Face>> = OnceLock::new();
    FACE.get_or_init(|| {
        let bytes: &'static [u8] = include_bytes!("../openscad/DejaVuSans.ttf");
        Arc::new(Face::from_bytes(bytes, 0).expect("the bundled DejaVu Sans parses"))
    })
    .clone()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Generic {
    Sans,
    Serif,
    Mono,
}

impl Generic {
    fn of(family: &str) -> Generic {
        let f = family.to_ascii_lowercase();
        if f.contains("mono") || f.contains("courier") {
            Generic::Mono
        } else if (f.contains("serif") || f.contains("times")) && !f.contains("sans") {
            Generic::Serif
        } else {
            Generic::Sans
        }
    }
}

/// The faces a drawing's text may use, by family name.
#[derive(Clone, Debug, Default)]
pub struct DrawingFonts {
    named: BTreeMap<String, Arc<Face>>,
    generic: BTreeMap<Generic, Arc<Face>>,
}

impl PartialEq for DrawingFonts {
    /// The same faces under the same names.
    fn eq(&self, other: &Self) -> bool {
        let same = |a: &Arc<Face>, b: &Arc<Face>| Arc::ptr_eq(a, b);
        self.named.len() == other.named.len()
            && self.generic.len() == other.generic.len()
            && self.named.iter().zip(&other.named).all(|((k, a), (l, b))| k == l && same(a, b))
            && self.generic.iter().zip(&other.generic).all(|((k, a), (l, b))| k == l && same(a, b))
    }
}

impl DrawingFonts {
    /// DejaVu Sans for every family.
    pub fn new() -> Self {
        Self::default()
    }

    /// `face` for the family `name` (matched without regard to case), and
    /// for the generic family that name falls in when `generic` is set.
    pub fn with_face(mut self, name: &str, face: Face, generic: bool) -> Self {
        let face = Arc::new(face);
        if generic {
            self.generic.insert(Generic::of(name), face.clone());
        }
        self.named.insert(name.to_ascii_lowercase(), face);
        self
    }

    /// The operating system's sans-serif, serif and monospace TrueType faces
    /// for the generic families, where one of the usual files is installed
    /// (macOS, Linux distributions, Windows); DejaVu Sans for the others.
    pub fn system() -> Self {
        const CANDIDATES: [(Generic, &[&str]); 3] = [
            (
                Generic::Sans,
                &[
                    "/System/Library/Fonts/Supplemental/Arial.ttf",
                    "/Library/Fonts/Arial.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/TTF/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    "C:\\Windows\\Fonts\\arial.ttf",
                ],
            ),
            (
                Generic::Serif,
                &[
                    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
                    "/Library/Fonts/Times New Roman.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
                    "/usr/share/fonts/TTF/DejaVuSerif.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
                    "C:\\Windows\\Fonts\\times.ttf",
                ],
            ),
            (
                Generic::Mono,
                &[
                    "/System/Library/Fonts/Supplemental/Courier New.ttf",
                    "/Library/Fonts/Courier New.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                    "C:\\Windows\\Fonts\\cour.ttf",
                ],
            ),
        ];
        let mut fonts = DrawingFonts::default();
        for (generic, paths) in CANDIDATES {
            if let Some(face) = paths.iter().find_map(|p| std::fs::read(p).ok().and_then(|b| Face::from_bytes(b, 0))) {
                fonts.generic.insert(generic, Arc::new(face));
            }
        }
        fonts
    }

    /// The face for a family (none given: sans-serif).
    /// How far `text` in `family` at `size` advances, kerned, in the units
    /// of `size`: the width every format lays it out at, for setting text
    /// out (a label's box, a table's columns) the way it will be drawn.
    pub fn advance(&self, family: Option<&str>, text: &str, size: f64) -> f64 {
        let face = self.face(family);
        layout(&face, text).advance * size / face.units_per_em()
    }

    /// The face's ascent and descent at `size` (`hhea`, both positive), in
    /// the units of `size`: a line's glyphs reach `ascent` above the baseline
    /// and `descent` below.
    pub fn vertical_metrics(&self, family: Option<&str>, size: f64) -> (f64, f64) {
        let face = self.face(family);
        let k = size / face.units_per_em();
        (face.ttf.ascender as f64 * k, -(face.ttf.descender as f64) * k)
    }

    pub(crate) fn face(&self, family: Option<&str>) -> Arc<Face> {
        let family = family.unwrap_or("sans-serif");
        if let Some(f) = self.named.get(&family.to_ascii_lowercase()) {
            return f.clone();
        }
        self.generic.get(&Generic::of(family)).cloned().unwrap_or_else(bundled)
    }
}

/// A run of text laid out: each glyph, the character it stands for, and
/// its pen position, and the whole advance, in font units.
pub(crate) struct Run {
    pub glyphs: Vec<(u16, char, f64)>,
    pub advance: f64,
}

/// `text` in `face`, one glyph a character (the face's `?` for one it
/// lacks; nothing for one it lacks when it has no `?` either), kerned.
pub(crate) fn layout(face: &Face, text: &str) -> Run {
    let f = &face.ttf;
    let mut glyphs = Vec::new();
    let mut x = 0.0;
    let mut prev: Option<u16> = None;
    for c in text.chars() {
        let Some(g) = f.glyph_id(c) else { continue };
        if let Some(p) = prev {
            x += f.kern(p, g) as f64;
        }
        glyphs.push((g, c, x));
        x += f.advance(g) as f64;
        prev = Some(g);
    }
    Run { glyphs, advance: x }
}

/// A TrueType file holding only the glyphs `used` (and the components of
/// composite ones) with every glyph ID kept, so a PDF can address them as
/// its CIDs; the rest of `glyf` is emptied. The tables a PDF viewer needs
/// are kept (`head`, `hhea`, `maxp`, `hmtx`, `loca`, `glyf`, and the hinting
/// programs and values when there are any); `loca` is rewritten long.
pub(crate) fn subset(face: &Face, used: &std::collections::BTreeSet<u16>) -> Option<Vec<u8>> {
    let head = face.table(b"head")?;
    let loca = face.table(b"loca")?;
    let glyf = face.table(b"glyf")?;
    let long = be16(head, 50)? == 1;
    let n_glyphs = be16(face.table(b"maxp")?, 4)? as usize;
    let span = |g: usize| -> Option<(usize, usize)> {
        if long {
            Some((be32(loca, 4 * g)? as usize, be32(loca, 4 * g + 4)? as usize))
        } else {
            Some((be16(loca, 2 * g)? as usize * 2, be16(loca, 2 * g + 2)? as usize * 2))
        }
    };
    // the glyphs, their components, and .notdef
    let mut keep: std::collections::BTreeSet<u16> = used.clone();
    keep.insert(0);
    let mut todo: Vec<u16> = keep.iter().copied().collect();
    while let Some(g) = todo.pop() {
        let Some((a, b)) = span(g as usize) else { continue };
        let Some(data) = glyf.get(a..b) else { continue };
        if data.len() < 10 || (be16(data, 0)? as i16) >= 0 {
            continue;
        }
        let mut p = 10;
        loop {
            let (flags, comp) = (be16(data, p)?, be16(data, p + 2)?);
            if keep.insert(comp) {
                todo.push(comp);
            }
            p += 4 + if flags & 0x1 != 0 { 4 } else { 2 };
            p += if flags & 0x8 != 0 {
                2
            } else if flags & 0x40 != 0 {
                4
            } else if flags & 0x80 != 0 {
                8
            } else {
                0
            };
            if flags & 0x20 == 0 {
                break;
            }
        }
    }
    let mut new_glyf = Vec::new();
    let mut new_loca = Vec::with_capacity(4 * (n_glyphs + 1));
    for g in 0..n_glyphs {
        new_loca.extend_from_slice(&(new_glyf.len() as u32).to_be_bytes());
        if keep.contains(&(g as u16)) {
            let (a, b) = span(g)?;
            new_glyf.extend_from_slice(glyf.get(a..b)?);
            while new_glyf.len() % 4 != 0 {
                new_glyf.push(0);
            }
        }
    }
    new_loca.extend_from_slice(&(new_glyf.len() as u32).to_be_bytes());
    let mut new_head = head.to_vec();
    new_head[8..12].fill(0); // checkSumAdjustment, set below
    new_head[50..52].copy_from_slice(&1u16.to_be_bytes());
    let mut tables: Vec<([u8; 4], Vec<u8>)> = vec![
        (*b"glyf", new_glyf),
        (*b"head", new_head),
        (*b"hhea", face.table(b"hhea")?.to_vec()),
        (*b"hmtx", face.table(b"hmtx")?.to_vec()),
        (*b"loca", new_loca),
        (*b"maxp", face.table(b"maxp")?.to_vec()),
    ];
    for tag in [b"cvt ", b"fpgm", b"prep"] {
        if let Some(t) = face.table(tag) {
            tables.push((*tag, t.to_vec()));
        }
    }
    tables.sort_by_key(|t| t.0);
    Some(write_sfnt(tables))
}

/// A TrueType file of `tables` (sorted by tag), with checksums and
/// `head.checkSumAdjustment`.
fn write_sfnt(tables: Vec<([u8; 4], Vec<u8>)>) -> Vec<u8> {
    let n = tables.len() as u16;
    let pow = 1u16 << (15 - n.leading_zeros() as u16).min(15);
    let search = pow * 16;
    let mut out = Vec::new();
    out.extend_from_slice(&0x0001_0000u32.to_be_bytes());
    for v in [n, search, pow.trailing_zeros() as u16, n * 16 - search] {
        out.extend_from_slice(&v.to_be_bytes());
    }
    let checksum = |d: &[u8]| -> u32 {
        d.chunks(4).fold(0u32, |s, c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            s.wrapping_add(u32::from_be_bytes(w))
        })
    };
    let mut offset = 12 + 16 * tables.len();
    let mut head_at = None;
    for (tag, data) in &tables {
        out.extend_from_slice(tag);
        out.extend_from_slice(&checksum(data).to_be_bytes());
        out.extend_from_slice(&(offset as u32).to_be_bytes());
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        if tag == b"head" {
            head_at = Some(offset);
        }
        offset += data.len().div_ceil(4) * 4;
    }
    for (_, data) in &tables {
        out.extend_from_slice(data);
        while out.len() % 4 != 0 {
            out.push(0);
        }
    }
    if let Some(h) = head_at {
        let adj = 0xB1B0_AFBAu32.wrapping_sub(checksum(&out));
        out[h + 8..h + 12].copy_from_slice(&adj.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn families_fall_back_to_the_bundled_face() {
        let fonts = DrawingFonts::new();
        assert_eq!(fonts.face(Some("monospace")).postscript_name, "DejaVuSans");
        assert_eq!(fonts.face(None).postscript_name, "DejaVuSans");
    }

    #[test]
    fn a_named_face_wins_over_its_generic() {
        let bytes: &'static [u8] = include_bytes!("../openscad/DejaVuSans.ttf");
        let fonts = DrawingFonts::new().with_face("Label Face", Face::from_bytes(bytes, 0).unwrap(), false);
        assert!(!Arc::ptr_eq(&fonts.face(Some("label face")), &bundled()));
        assert!(Arc::ptr_eq(&fonts.face(Some("sans-serif")), &bundled()));
    }

    #[test]
    fn layout_kerns_pairs() {
        let face = bundled();
        let av = layout(&face, "AV");
        let (a, v) = (face.ttf.glyph_id('A').unwrap(), face.ttf.glyph_id('V').unwrap());
        let kern = face.ttf.kern(a, v) as f64;
        assert!(kern < 0.0, "DejaVu kerns AV");
        assert_eq!(av.glyphs[1].2, face.ttf.advance(a) as f64 + kern);
    }

    /// A table of a plain TrueType file.
    fn table<'a>(b: &'a [u8], tag: &[u8; 4]) -> &'a [u8] {
        let n = be16(b, 4).unwrap() as usize;
        (0..n)
            .find_map(|k| {
                let rec = 12 + 16 * k;
                (&b[rec..rec + 4] == tag).then(|| &b[be32(b, rec + 8).unwrap() as usize..][..be32(b, rec + 12).unwrap() as usize])
            })
            .unwrap()
    }

    /// Glyph `g`'s bytes in a file with long `loca`, or as `face` has them.
    fn glyph(loca: &[u8], glyf: &[u8], g: usize, long: bool) -> Vec<u8> {
        let (a, b) = if long {
            (be32(loca, 4 * g).unwrap() as usize, be32(loca, 4 * g + 4).unwrap() as usize)
        } else {
            (be16(loca, 2 * g).unwrap() as usize * 2, be16(loca, 2 * g + 2).unwrap() as usize * 2)
        };
        glyf[a..b].to_vec()
    }

    #[test]
    fn a_subset_keeps_the_glyphs_used_at_their_ids() {
        let face = bundled();
        // é is a composite in DejaVu: its components come along
        let used: std::collections::BTreeSet<u16> = "Hé".chars().filter_map(|c| face.ttf.glyph_id(c)).collect();
        let sub = subset(&face, &used).unwrap();
        assert!(sub.len() < 80_000, "{} bytes", sub.len());
        let (loca, glyf) = (table(&sub, b"loca"), table(&sub, b"glyf"));
        let n = be16(table(&sub, b"maxp"), 4).unwrap() as usize;
        assert_eq!(n, face.ttf.glyphs.len(), "every glyph id kept");
        assert_eq!(be16(table(&sub, b"head"), 50), Some(1), "loca is long");
        let long = be16(face.table(b"head").unwrap(), 50) == Some(1);
        let (oloca, oglyf) = (face.table(b"loca").unwrap(), face.table(b"glyf").unwrap());
        let mut kept = 0;
        for g in 0..n {
            let (new, old) = (glyph(loca, glyf, g, true), glyph(oloca, oglyf, g, long));
            if new.is_empty() {
                assert!(!used.contains(&(g as u16)), "glyph {g} was used");
            } else {
                assert_eq!(&new[..old.len()], &old[..], "glyph {g}");
                kept += 1;
            }
        }
        assert!(kept > used.len() + 1, "the composite's components and .notdef too: {kept}");
        // the whole file sums to 0xB1B0AFBA
        let sum = sub.chunks(4).fold(0u32, |s, c| s.wrapping_add(u32::from_be_bytes(c.try_into().unwrap())));
        assert_eq!(sum, 0xB1B0_AFBA);
    }
}
