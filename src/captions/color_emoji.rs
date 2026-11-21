//! Colour emoji from an `sbix` font (Apple Color Emoji): each glyph is a
//! PNG per strike (a pixel size), drawn as its own colours rather than
//! as coverage in the text's colour.
//!
//! The system's file is ~180 MB, so it is never read whole: the table
//! directory, `cmap`, `hmtx` and each strike's offsets are read at open
//! (a few hundred kilobytes), and a glyph's PNG only when it's drawn.
//! Only characters shown as emoji by default (Unicode's
//! `Emoji_Presentation`) are taken: digits, `#`, `©`, arrows and the like
//! stay with the text faces.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::font::GlyphBitmap;

/// How far below where its offsets say the system draws an `sbix` image,
/// in ems: Core Text puts Apple's emoji an eighth of an em lower, measured
/// (see `the_system_draws_emoji_too`).
const DROP: f32 = 0.125;

/// A strike: its pixels per em, and where each glyph's data is.
struct Strike {
    ppem: u16,
    /// Absolute file offset of the strike.
    at: u64,
    /// Per glyph (and one past the last), from the strike's start.
    offsets: Vec<u32>,
}

/// A decoded glyph image at its strike's size, straight-alpha RGBA, and
/// where its bottom-left corner is from the pen (strike pixels, y up).
struct Picture {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    origin: (i16, i16),
    ppem: u16,
}

type Pictures = HashMap<(u16, u16), Option<Arc<Picture>>>;

/// An `sbix` colour font, read as glyphs are drawn.
pub struct ColorGlyphs {
    file: Mutex<std::fs::File>,
    units_per_em: u16,
    ascender: i16,
    descender: i16,
    /// Codepoint → glyph, emoji only.
    cmap: HashMap<u32, u16>,
    /// Advance per glyph, font units.
    advances: Vec<u16>,
    /// Where each glyph's `glyf` entry is (from `loca`), and that table: its
    /// box's lower-left is where the image goes (with `sbix`'s offsets).
    loca: Vec<u32>,
    glyf_at: u64,
    strikes: Vec<Strike>,
    /// Images decoded, by (glyph, strike's pixels per em); `None`: none.
    pictures: Mutex<Pictures>,
    /// Emoji sequences met (👨‍👩‍👧, 👍🏽, 🇯🇵, ❤️), each a glyph of its own:
    /// the `n`th is glyph [`SEQUENCES`]` + n`.
    sequences: Mutex<Vec<String>>,
}

/// Where glyphs for emoji sequences start: past every codepoint, below the
/// glyph index's face tag.
pub const SEQUENCES: u32 = 0x20_0000;

const ZWJ: char = '\u{200D}';
const VS16: char = '\u{FE0F}';

/// How many of `chars` make the emoji sequence they start with: a pair of
/// regional indicators (a flag), a keycap (`1️⃣`), or an emoji (or a
/// symbol asked to be one with U+FE0F) with its skin tone, tags, and more
/// joined on with U+200D. 0 for no sequence of two or more.
pub fn sequence_len(chars: &[char]) -> usize {
    let at = |i: usize| chars.get(i).copied().unwrap_or('\0');
    let flag = |c: char| ('\u{1F1E6}'..='\u{1F1FF}').contains(&c);
    let tone = |c: char| ('\u{1F3FB}'..='\u{1F3FF}').contains(&c);
    let tag = |c: char| ('\u{E0020}'..='\u{E007F}').contains(&c);
    if flag(at(0)) && flag(at(1)) {
        return 2;
    }
    if matches!(at(0), '0'..='9' | '#' | '*') {
        let j = if at(1) == VS16 { 2 } else { 1 };
        return if at(j) == '\u{20E3}' { j + 1 } else { 0 };
    }
    // one element: an emoji, or a symbol (not ASCII) made one, then its
    // presentation selector and skin tone
    let element = |i: usize, joined: bool| -> Option<usize> {
        let c = at(i);
        let pictographic = is_emoji_presentation(c) || (c as u32 >= 0xA9 && (at(i + 1) == VS16 || joined && c as u32 >= 0x2000));
        if !pictographic || c.is_whitespace() {
            return None;
        }
        let mut j = i + 1;
        if at(j) == VS16 {
            j += 1;
        }
        if tone(at(j)) {
            j += 1;
        }
        Some(j)
    };
    let Some(mut j) = element(0, false) else { return 0 };
    while tag(at(j)) {
        j += 1;
    }
    while at(j) == ZWJ {
        match element(j + 1, true) {
            Some(k) => j = k,
            None => break,
        }
    }
    if j >= 2 { j } else { 0 }
}

impl std::fmt::Debug for ColorGlyphs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColorGlyphs")
            .field("glyphs", &self.cmap.len())
            .field("strikes", &self.strikes.iter().map(|s| s.ppem).collect::<Vec<_>>())
            .finish()
    }
}

fn be16(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes(b.get(at..at + 2)?.try_into().ok()?))
}

fn be32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

/// Whether `c` is shown as an emoji by default (`Emoji_Presentation=Yes`,
/// Unicode 16), rather than as text.
pub fn is_emoji_presentation(c: char) -> bool {
    const RANGES: &[(u32, u32)] = &[
        (0x231A, 0x231B), (0x23E9, 0x23EC), (0x23F0, 0x23F0), (0x23F3, 0x23F3), (0x25FD, 0x25FE),
        (0x2614, 0x2615), (0x2648, 0x2653), (0x267F, 0x267F), (0x2693, 0x2693), (0x26A1, 0x26A1),
        (0x26AA, 0x26AB), (0x26BD, 0x26BE), (0x26C4, 0x26C5), (0x26CE, 0x26CE), (0x26D4, 0x26D4),
        (0x26EA, 0x26EA), (0x26F2, 0x26F3), (0x26F5, 0x26F5), (0x26FA, 0x26FA), (0x26FD, 0x26FD),
        (0x2705, 0x2705), (0x270A, 0x270B), (0x2728, 0x2728), (0x274C, 0x274C), (0x274E, 0x274E),
        (0x2753, 0x2755), (0x2757, 0x2757), (0x2795, 0x2797), (0x27B0, 0x27B0), (0x27BF, 0x27BF),
        (0x2B1B, 0x2B1C), (0x2B50, 0x2B50), (0x2B55, 0x2B55),
        (0x1F004, 0x1F004), (0x1F0CF, 0x1F0CF), (0x1F18E, 0x1F18E), (0x1F191, 0x1F19A),
        (0x1F1E6, 0x1F1FF), (0x1F201, 0x1F201), (0x1F21A, 0x1F21A), (0x1F22F, 0x1F22F),
        (0x1F232, 0x1F236), (0x1F238, 0x1F23A), (0x1F250, 0x1F251), (0x1F300, 0x1F320),
        (0x1F32D, 0x1F335), (0x1F337, 0x1F37C), (0x1F37E, 0x1F393), (0x1F3A0, 0x1F3CA),
        (0x1F3CF, 0x1F3D3), (0x1F3E0, 0x1F3F0), (0x1F3F4, 0x1F3F4), (0x1F3F8, 0x1F43E),
        (0x1F440, 0x1F440), (0x1F442, 0x1F4FC), (0x1F4FF, 0x1F53D), (0x1F54B, 0x1F54E),
        (0x1F550, 0x1F567), (0x1F57A, 0x1F57A), (0x1F595, 0x1F596), (0x1F5A4, 0x1F5A4),
        (0x1F5FB, 0x1F64F), (0x1F680, 0x1F6C5), (0x1F6CC, 0x1F6CC), (0x1F6D0, 0x1F6D2),
        (0x1F6D5, 0x1F6D7), (0x1F6DC, 0x1F6DF), (0x1F6EB, 0x1F6EC), (0x1F6F4, 0x1F6FC),
        (0x1F7E0, 0x1F7EB), (0x1F7F0, 0x1F7F0), (0x1F90C, 0x1F93A), (0x1F93C, 0x1F945),
        (0x1F947, 0x1F9FF), (0x1FA70, 0x1FA7C), (0x1FA80, 0x1FA89), (0x1FA8F, 0x1FAC6),
        (0x1FACE, 0x1FADC), (0x1FADF, 0x1FAE9), (0x1FAF0, 0x1FAF8),
    ];
    let c = c as u32;
    RANGES
        .binary_search_by(|&(lo, hi)| {
            if hi < c {
                std::cmp::Ordering::Less
            } else if lo > c {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

impl ColorGlyphs {
    /// The system's colour emoji font, where there is one (Apple
    /// platforms; on the iOS simulator, inside its runtime).
    pub fn system() -> Option<Self> {
        let mut paths = Vec::new();
        if cfg!(target_os = "macos") {
            paths.push(std::path::PathBuf::from("/System/Library/Fonts/Apple Color Emoji.ttc"));
        } else if cfg!(any(target_os = "ios", target_os = "visionos", target_os = "tvos")) {
            let base = std::env::var("IPHONE_SIMULATOR_ROOT")
                .or_else(|_| std::env::var("SIMULATOR_ROOT"))
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| "/".into())
                .join("System/Library/Fonts");
            for dir in ["Core", "CoreAddition"] {
                for name in ["AppleColorEmoji-160px.ttc", "AppleColorEmoji.ttc", "Apple Color Emoji.ttc", "AppleColorEmoji@2x.ttc"] {
                    paths.push(base.join(dir).join(name));
                }
            }
        }
        paths.into_iter().find_map(|p| Self::open(&p).ok())
    }

    /// An `sbix` font's file (a `.ttc` gives its first face).
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let read = |at: u64, len: usize| -> Result<Vec<u8>, String> { read_at(&file, at, len) };
        let head = read(0, 12)?;
        let font_at = if &head[0..4] == b"ttcf" { u64::from(be32(&read(12, 4)?, 0).unwrap_or(0)) } else { 0 };
        let dir = read(font_at, 12)?;
        let n = be16(&dir, 4).ok_or("no tables")? as usize;
        let records = read(font_at + 12, n * 16)?;
        let mut tables = HashMap::new();
        for i in 0..n {
            let r = &records[i * 16..i * 16 + 16];
            let tag = [r[0], r[1], r[2], r[3]];
            tables.insert(tag, (u64::from(be32(r, 8).unwrap_or(0)), be32(r, 12).unwrap_or(0) as usize));
        }
        let table = |tag: &[u8; 4]| -> Result<(u64, usize), String> {
            tables.get(tag).copied().ok_or_else(|| format!("no {} table", String::from_utf8_lossy(tag)))
        };
        let (sbix_at, _) = table(b"sbix")?;
        let (at, len) = table(b"head")?;
        let head = read(at, len.min(64))?;
        let units_per_em = be16(&head, 18).ok_or("bad head")?;
        let long_loca = be16(&head, 50).unwrap_or(0) == 1;
        let (at, len) = table(b"hhea")?;
        let hhea = read(at, len.min(64))?;
        let ascender = be16(&hhea, 4).ok_or("bad hhea")? as i16;
        let descender = be16(&hhea, 6).ok_or("bad hhea")? as i16;
        let long_metrics = be16(&hhea, 34).ok_or("bad hhea")? as usize;
        let (at, len) = table(b"maxp")?;
        let glyphs = be16(&read(at, len.min(8))?, 4).ok_or("bad maxp")? as usize;
        let (at, len) = table(b"hmtx")?;
        let hmtx = read(at, len.min(long_metrics * 4))?;
        let mut advances: Vec<u16> = (0..long_metrics).map(|i| be16(&hmtx, i * 4).unwrap_or(0)).collect();
        let last = advances.last().copied().unwrap_or(units_per_em);
        advances.resize(glyphs, last);
        let (glyf_at, loca) = match (table(b"glyf"), table(b"loca")) {
            (Ok((glyf_at, _)), Ok((at, len))) => {
                let b = read(at, len)?;
                let loca: Vec<u32> = if long_loca {
                    (0..=glyphs).map(|g| be32(&b, g * 4).unwrap_or(0)).collect()
                } else {
                    (0..=glyphs).map(|g| u32::from(be16(&b, g * 2).unwrap_or(0)) * 2).collect()
                };
                (glyf_at, loca)
            }
            _ => (0, Vec::new()),
        };
        let (at, len) = table(b"cmap")?;
        let cmap = parse_cmap(&read(at, len)?);
        let sbix = read(sbix_at, 8)?;
        let count = be32(&sbix, 4).ok_or("bad sbix")? as usize;
        let offsets = read(sbix_at + 8, count * 4)?;
        let mut strikes = Vec::new();
        for i in 0..count {
            let at = sbix_at + u64::from(be32(&offsets, i * 4).unwrap_or(0));
            let b = read(at, 4 + (glyphs + 1) * 4)?;
            strikes.push(Strike {
                ppem: be16(&b, 0).unwrap_or(0),
                at,
                offsets: (0..=glyphs).map(|g| be32(&b, 4 + g * 4).unwrap_or(0)).collect(),
            });
        }
        strikes.sort_by_key(|s| s.ppem);
        if strikes.is_empty() {
            return Err("no strikes".into());
        }
        Ok(Self {
            file: Mutex::new(file),
            units_per_em: units_per_em.max(1),
            ascender,
            descender,
            cmap: cmap.into_iter().filter(|&(c, _)| char::from_u32(c).is_some_and(is_emoji_presentation)).collect(),
            advances,
            loca,
            glyf_at,
            strikes,
            pictures: Mutex::new(HashMap::new()),
            sequences: Mutex::new(Vec::new()),
        })
    }

    /// The glyph for the emoji sequence `seq` (see [`sequence_len`]): one
    /// image, as the system draws it. `None` where only the system can
    /// (Core Text, on Apple's platforms) and it isn't here.
    pub fn sequence_glyph(&self, seq: &[char]) -> Option<u32> {
        if !cfg!(any(target_os = "macos", target_os = "ios", target_os = "visionos", target_os = "tvos")) {
            return None;
        }
        let s: String = seq.iter().collect();
        let mut all = self.sequences.lock().unwrap_or_else(|e| e.into_inner());
        let n = match all.iter().position(|x| *x == s) {
            Some(n) => n,
            None => {
                all.push(s);
                all.len() - 1
            }
        };
        Some(SEQUENCES + n as u32)
    }

    fn sequence(&self, gid: u32) -> Option<String> {
        let n = gid.checked_sub(SEQUENCES)? as usize;
        self.sequences.lock().unwrap_or_else(|e| e.into_inner()).get(n).cloned()
    }

    /// The advance of glyph `gid` (a codepoint, or a sequence's) at `size`.
    pub fn advance_of(&self, gid: u32, size: f32) -> f32 {
        if let Some(c) = char::from_u32(gid) {
            return self.advance(c, size);
        }
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "visionos", target_os = "tvos"))]
        if let Some(s) = self.sequence(gid) {
            return coretext::line(&s, size, 0.0, false).map_or(0.0, |g| g.advance);
        }
        0.0
    }

    /// Glyph `gid` (a codepoint, or a sequence's) at `size`, `shift`
    /// pixels right of the pen.
    pub fn raster_of(&self, gid: u32, size: f32, shift: f32) -> GlyphBitmap {
        if let Some(c) = char::from_u32(gid) {
            return self.raster(c, size, shift);
        }
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "visionos", target_os = "tvos"))]
        if let Some(g) = self.sequence(gid).and_then(|s| coretext::line(&s, size, shift, true)) {
            return g;
        }
        GlyphBitmap::default()
    }

    /// Whether `c` is one of its emoji.
    pub fn has(&self, c: char) -> bool {
        self.cmap.contains_key(&(c as u32))
    }

    /// Ascent and descent at `size` pixels per em.
    pub fn metrics(&self, size: f32) -> (f32, f32) {
        let k = size / self.units_per_em as f32;
        (self.ascender as f32 * k, -(self.descender as f32) * k)
    }

    /// How far `c` moves the pen at `size`.
    pub fn advance(&self, c: char, size: f32) -> f32 {
        let Some(&g) = self.cmap.get(&(c as u32)) else { return 0.0 };
        self.advances.get(g as usize).copied().unwrap_or(self.units_per_em) as f32 * size / self.units_per_em as f32
    }

    /// Glyph `g`'s box's lower-left (`xMin`, `yMin`), font units; zero
    /// for one with no outline.
    fn corner(&self, g: u16) -> (i16, i16) {
        let (Some(&a), Some(&b)) = (self.loca.get(g as usize), self.loca.get(g as usize + 1)) else { return (0, 0) };
        if b < a + 10 {
            return (0, 0);
        }
        let file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        match read_at(&file, self.glyf_at + u64::from(a), 10) {
            Ok(h) => (be16(&h, 2).unwrap_or(0) as i16, be16(&h, 4).unwrap_or(0) as i16),
            Err(_) => (0, 0),
        }
    }

    /// The strike drawn from at `size`: the smallest at least as large,
    /// else the largest.
    fn strike_for(&self, size: f32) -> &Strike {
        self.strikes.iter().find(|s| s.ppem as f32 >= size).unwrap_or_else(|| self.strikes.last().unwrap())
    }

    /// Glyph `g`'s image in `strike`, decoded (kept for next time).
    fn picture(&self, strike: &Strike, g: u16, depth: u8) -> Option<Arc<Picture>> {
        let key = (g, strike.ppem);
        if let Some(p) = self.pictures.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return p.clone();
        }
        let (a, b) = (*strike.offsets.get(g as usize)?, *strike.offsets.get(g as usize + 1)?);
        let p = (b > a + 8)
            .then(|| {
                let file = self.file.lock().unwrap_or_else(|e| e.into_inner());
                read_at(&file, strike.at + u64::from(a), (b - a) as usize).ok()
            })
            .flatten()
            .and_then(|data| {
                let origin = (be16(&data, 0)? as i16, be16(&data, 2)? as i16);
                match &data[4..8] {
                    b"png " => {
                        let img = crate::utils::png::decode_png(&data[8..]).ok()?;
                        Some(Arc::new(Picture { width: img.width, height: img.height, rgba: img.rgba, origin, ppem: strike.ppem }))
                    }
                    // the same image as another glyph's
                    b"dupe" if depth < 2 => self.picture(strike, be16(&data, 8)?, depth + 1),
                    // another glyph's image, mirrored (facing the other way)
                    b"flip" if depth < 2 => {
                        let p = self.picture(strike, be16(&data, 8)?, depth + 1)?;
                        let mut rgba = p.rgba.clone();
                        for row in rgba.chunks_exact_mut(p.width as usize * 4) {
                            row.as_chunks_mut::<4>().0.reverse();
                        }
                        Some(Arc::new(Picture { width: p.width, height: p.height, rgba, origin: p.origin, ppem: p.ppem }))
                    }
                    _ => None,
                }
            });
        let mut cache = self.pictures.lock().unwrap_or_else(|e| e.into_inner());
        if cache.len() > 512 {
            cache.clear();
        }
        cache.insert(key, p.clone());
        p
    }

    /// `c` at `size` pixels per em, its image scaled to it, `shift` pixels
    /// right of the pen: premultiplied RGBA in [`GlyphBitmap::color`], its
    /// alpha as the coverage.
    pub fn raster(&self, c: char, size: f32, shift: f32) -> GlyphBitmap {
        let advance = self.advance(c, size);
        let blank = GlyphBitmap { advance, ..GlyphBitmap::default() };
        let Some(&g) = self.cmap.get(&(c as u32)) else { return blank };
        let Some(p) = self.picture(self.strike_for(size), g, 0) else {
            // not a PNG: iOS's font keeps its images in Apple's own
            // format (`emjc`), which the system draws
            #[cfg(any(target_os = "macos", target_os = "ios", target_os = "visionos", target_os = "tvos"))]
            if let Some(g) = coretext::raster(c, size, shift) {
                return GlyphBitmap { advance, ..g };
            }
            return blank;
        };
        let k = size / p.ppem as f32;
        // the image's lower-left is the outline box's, moved by the strike's
        // offsets; then its top-left, pen at the baseline, y down
        let (cx, cy) = self.corner(g);
        let units = size / self.units_per_em as f32;
        let left = cx as f32 * units + p.origin.0 as f32 * k + shift;
        let top = -(cy as f32 * units + (p.origin.1 as f32 + p.height as f32) * k) + DROP * size;
        GlyphBitmap { advance, ..resample(&p, left, top, k) }
    }
}

/// `p` scaled by `k` with its top-left at (`left`, `top`) from the pen
/// (fractions and all), each pixel the source pixels under it weighted by
/// overlap: premultiplied, on whole pixels.
fn resample(p: &Picture, left: f32, top: f32, k: f32) -> GlyphBitmap {
    let (x0, y0) = (left.floor() as i32, top.floor() as i32);
    let w = ((left + p.width as f32 * k).ceil() as i32 - x0).max(1) as u32;
    let h = ((top + p.height as f32 * k).ceil() as i32 - y0).max(1) as u32;
    let mut color = vec![0u8; (w * h * 4) as usize];
    // the source span under output pixel `o` of an axis starting at `from`
    let span = |o: u32, start: i32, from: f32| (((start + o as i32) as f32 - from) / k, ((start + o as i32 + 1) as f32 - from) / k);
    for oy in 0..h {
        let (sy0, sy1) = span(oy, y0, top);
        for ox in 0..w {
            let (sx0, sx1) = span(ox, x0, left);
            let mut acc = [0f32; 4];
            let mut area = 0f32;
            let mut sy = sy0.floor().max(0.0) as i32;
            while (sy as f32) < sy1 && sy < p.height as i32 {
                let fy = (sy1.min(sy as f32 + 1.0) - sy0.max(sy as f32)).max(0.0);
                let mut sx = sx0.floor().max(0.0) as i32;
                while (sx as f32) < sx1 && sx < p.width as i32 {
                    let f = (sx1.min(sx as f32 + 1.0) - sx0.max(sx as f32)).max(0.0) * fy;
                    let i = ((sy as u32 * p.width + sx as u32) * 4) as usize;
                    let a = p.rgba[i + 3] as f32 / 255.0;
                    for (c, v) in acc.iter_mut().take(3).enumerate() {
                        *v += p.rgba[i + c] as f32 * a * f;
                    }
                    acc[3] += p.rgba[i + 3] as f32 * f;
                    sx += 1;
                }
                area += fy * (sx1 - sx0);
                sy += 1;
            }
            if area > 0.0 {
                let o = ((oy * w + ox) * 4) as usize;
                for (c, v) in acc.iter().enumerate() {
                    color[o + c] = (v / area).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    GlyphBitmap {
        width: w,
        height: h,
        coverage: color.as_chunks::<4>().0.iter().map(|p| p[3]).collect(),
        offset_x: x0,
        offset_y: y0,
        advance: 0.0,
        color: Some(Arc::new(color)),
    }
}

/// Emoji drawn by the system (Core Text): the images only it can read
/// (iOS's), and sequences it joins into one (👨‍👩‍👧, 👍🏽, 🇯🇵).
#[cfg(any(target_os = "macos", target_os = "ios", target_os = "visionos", target_os = "tvos"))]
pub(crate) mod coretext {
    use std::ffi::c_void;
    use std::sync::Arc;

    use super::GlyphBitmap;

    type Ref = *const c_void;

    #[repr(C)]
    struct DictCallbacks([usize; 6]);

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFStringCreateWithBytes(alloc: Ref, bytes: *const u8, len: isize, encoding: u32, external: bool) -> Ref;
        fn CFAttributedStringCreate(alloc: Ref, s: Ref, attributes: Ref) -> Ref;
        fn CFDictionaryCreate(alloc: Ref, keys: *const Ref, values: *const Ref, n: isize, kc: *const DictCallbacks, vc: *const DictCallbacks) -> Ref;
        fn CFRelease(r: Ref);
        static kCFTypeDictionaryKeyCallBacks: DictCallbacks;
        static kCFTypeDictionaryValueCallBacks: DictCallbacks;
    }

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGColorSpaceCreateDeviceRGB() -> Ref;
        fn CGBitmapContextCreate(data: *mut c_void, w: usize, h: usize, bits: usize, row: usize, space: Ref, info: u32) -> Ref;
        fn CGContextSetTextPosition(ctx: Ref, x: f64, y: f64);
    }

    #[link(name = "CoreText", kind = "framework")]
    unsafe extern "C" {
        fn CTFontCreateWithName(name: Ref, size: f64, matrix: *const c_void) -> Ref;
        fn CTFontGetGlyphsForCharacters(font: Ref, chars: *const u16, glyphs: *mut u16, count: isize) -> bool;
        fn CTLineCreateWithAttributedString(s: Ref) -> Ref;
        fn CTLineGetTypographicBounds(line: Ref, ascent: *mut f64, descent: *mut f64, leading: *mut f64) -> f64;
        fn CTLineDraw(line: Ref, ctx: Ref);
        static kCTFontAttributeName: Ref;
    }

    /// A Core Foundation object (a string, font, line, context, colour
    /// space), released when dropped; never null.
    struct Cf(Ref);

    impl Cf {
        fn new(r: Ref) -> Option<Self> {
            (!r.is_null()).then_some(Self(r))
        }
    }

    impl Drop for Cf {
        fn drop(&mut self) {
            // SAFETY: an object this owns one reference to (`new` took a
            // Create call's result).
            unsafe { CFRelease(self.0) }
        }
    }

    fn string(s: &str) -> Option<Cf> {
        const UTF8: u32 = 0x0800_0100;
        // SAFETY: the bytes are valid UTF-8 for their length.
        Cf::new(unsafe { CFStringCreateWithBytes(std::ptr::null(), s.as_ptr(), s.len() as isize, UTF8, false) })
    }

    /// The system's emoji font at `size`.
    fn font(size: f32) -> Option<Cf> {
        let name = string("AppleColorEmoji")?;
        // SAFETY: a valid name; no matrix.
        Cf::new(unsafe { CTFontCreateWithName(name.0, size as f64, std::ptr::null()) })
    }

    /// Whether the emoji font itself has `c` (Core Text would otherwise
    /// draw it from another font).
    fn has(font: &Cf, c: char) -> bool {
        let mut units = [0u16; 2];
        let units = c.encode_utf16(&mut units);
        let mut glyphs = [0u16; 2];
        // SAFETY: `units` and `glyphs` hold `units.len()` each.
        unsafe { CTFontGetGlyphsForCharacters(font.0, units.as_ptr(), glyphs.as_mut_ptr(), units.len() as isize) && glyphs[0] != 0 }
    }

    /// `text` set as a line in `font`.
    fn set(text: &str, font: &Cf) -> Option<Cf> {
        let text = string(text)?;
        // SAFETY: valid objects; the dictionary and string retain what they're given.
        unsafe {
            let (keys, values) = ([kCTFontAttributeName], [font.0]);
            let attrs = Cf::new(CFDictionaryCreate(
                std::ptr::null(),
                keys.as_ptr(),
                values.as_ptr(),
                1,
                &kCFTypeDictionaryKeyCallBacks,
                &kCFTypeDictionaryValueCallBacks,
            ))?;
            let attributed = Cf::new(CFAttributedStringCreate(std::ptr::null(), text.0, attrs.0))?;
            Cf::new(CTLineCreateWithAttributedString(attributed.0))
        }
    }

    /// `text` (an emoji, or a sequence) as the system sets it in its emoji
    /// font at `size`, `shift` pixels right of the pen: its advance, and
    /// with `draw` its image (premultiplied RGBA) in a box with room around
    /// it.
    pub(crate) fn line(text: &str, size: f32, shift: f32, draw: bool) -> Option<GlyphBitmap> {
        const PREMULTIPLIED_LAST: u32 = 1;
        let line = set(text, &font(size)?)?;
        let (mut ascent, mut descent, mut leading) = (0.0, 0.0, 0.0);
        // SAFETY: a valid line; the bounds are written to locals.
        let advance = unsafe { CTLineGetTypographicBounds(line.0, &mut ascent, &mut descent, &mut leading) } as f32;
        if !draw {
            return Some(GlyphBitmap { advance, ..GlyphBitmap::default() });
        }
        // room for ink past the line's box
        let pad = (size * 0.25).ceil() as i32;
        let left = shift.floor() as i32 - pad;
        let bottom = -(descent.ceil() as i32) - pad;
        let w = (advance.ceil() as i32 + 2 * pad + 1) as usize;
        let h = (ascent.ceil() as i32 + descent.ceil() as i32 + 2 * pad) as usize;
        let mut px = vec![0u8; w * h * 4];
        // SAFETY: `px` is `w × h` RGBA and outlives the context, dropped
        // (released) at the end of this block.
        unsafe {
            let space = Cf::new(CGColorSpaceCreateDeviceRGB())?;
            let ctx = Cf::new(CGBitmapContextCreate(px.as_mut_ptr().cast(), w, h, 8, w * 4, space.0, PREMULTIPLIED_LAST))?;
            CGContextSetTextPosition(ctx.0, shift as f64 - left as f64, -bottom as f64);
            CTLineDraw(line.0, ctx.0);
        }
        Some(GlyphBitmap {
            width: w as u32,
            height: h as u32,
            coverage: px.as_chunks::<4>().0.iter().map(|p| p[3]).collect(),
            offset_x: left,
            // the bitmap's first row is its top
            offset_y: -(bottom + h as i32),
            advance,
            color: Some(Arc::new(px)),
        })
    }

    /// `c` at `size` pixels per em as the system draws it (see [`line`]);
    /// `None` if its emoji font hasn't it.
    pub(crate) fn raster(c: char, size: f32, shift: f32) -> Option<GlyphBitmap> {
        if !has(&font(size)?, c) {
            return None;
        }
        line(c.encode_utf8(&mut [0; 4]), size, shift, true)
    }
}

/// `len` bytes of `file` from `at`.
fn read_at(file: &std::fs::File, at: u64, len: usize) -> Result<Vec<u8>, String> {
    let mut buf = vec![0u8; len];
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(&mut buf, at).map_err(|e| e.to_string())?;
    }
    #[cfg(not(unix))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = file;
        f.seek(SeekFrom::Start(at)).map_err(|e| e.to_string())?;
        f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    }
    Ok(buf)
}

/// A `cmap` table's Unicode mapping (format 12, else format 4).
fn parse_cmap(t: &[u8]) -> HashMap<u32, u16> {
    let mut out = HashMap::new();
    let n = be16(t, 2).unwrap_or(0) as usize;
    let subtables: Vec<usize> = (0..n).filter_map(|i| be32(t, 4 + i * 8 + 4).map(|o| o as usize)).collect();
    let format = |o: usize| be16(t, o).unwrap_or(0);
    if let Some(&o) = subtables.iter().find(|&&o| format(o) == 12) {
        let groups = be32(t, o + 12).unwrap_or(0) as usize;
        for g in 0..groups {
            let r = o + 16 + g * 12;
            let (Some(start), Some(end), Some(gid)) = (be32(t, r), be32(t, r + 4), be32(t, r + 8)) else { break };
            for c in start..=end.min(start + 0x10000) {
                out.insert(c, (gid + c - start) as u16);
            }
        }
    } else if let Some(&o) = subtables.iter().find(|&&o| format(o) == 4) {
        let segs = be16(t, o + 6).unwrap_or(0) as usize / 2;
        let (ends, starts, deltas, ranges) = (o + 14, o + 16 + segs * 2, o + 16 + segs * 4, o + 16 + segs * 6);
        for s in 0..segs {
            let (Some(end), Some(start), Some(delta), Some(range)) =
                (be16(t, ends + s * 2), be16(t, starts + s * 2), be16(t, deltas + s * 2), be16(t, ranges + s * 2))
            else {
                break;
            };
            for c in start..=end {
                if c == 0xFFFF {
                    break;
                }
                let g = if range == 0 {
                    c.wrapping_add(delta)
                } else {
                    let at = ranges + s * 2 + range as usize + (c - start) as usize * 2;
                    match be16(t, at) {
                        Some(0) | None => 0,
                        Some(g) => g.wrapping_add(delta),
                    }
                };
                if g != 0 {
                    out.insert(u32::from(c), g);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequences() {
        let len = |s: &str| sequence_len(&s.chars().collect::<Vec<_>>());
        assert_eq!(len("👨‍👩‍👧 x"), 5, "a family");
        assert_eq!(len("👍🏽!"), 2, "a skin tone");
        assert_eq!(len("🇯🇵🇺"), 2, "a flag, then the next");
        assert_eq!(len("1️⃣"), 3, "a keycap");
        assert_eq!(len("❤️ love"), 2, "a symbol made an emoji");
        assert_eq!(len("👩🏽‍💻"), 4);
        assert_eq!(len("🏴󠁧󠁢󠁳󠁣󠁴󠁿"), 7, "a tag sequence (Scotland)");
        assert_eq!(len("😀"), 0, "one character: no sequence");
        assert_eq!(len("😀😀"), 0);
        assert_eq!(len("1 2"), 0);
        assert_eq!(len("a\u{FE0F}"), 0);
        assert_eq!(len("→\u{200D}"), 0);
    }

    /// A sequence the system joins draws as one emoji: one image about an
    /// em wide, not its pieces side by side.
    #[cfg(target_os = "macos")]
    #[test]
    fn sequences_draw_as_one() {
        let e = ColorGlyphs::system().unwrap();
        for s in ["👨‍👩‍👧", "👍🏽", "🇯🇵", "❤️", "1️⃣"] {
            let chars: Vec<char> = s.chars().collect();
            let g = e.sequence_glyph(&chars).unwrap();
            assert_eq!(e.sequence_glyph(&chars), Some(g), "the same glyph again");
            let adv = e.advance_of(g, 32.0);
            assert!((28.0..=44.0).contains(&adv), "{s}: {adv}");
            let b = e.raster_of(g, 32.0, 0.0);
            let color = b.color.as_ref().unwrap();
            let ink = color.chunks_exact(4).filter(|p| p[3] > 200).count();
            assert!(ink > 200, "{s}: {ink}");
            // (Apple's families and keycaps are grey: the rest have colour)
            let colourful = color.chunks_exact(4).filter(|p| p[3] > 200 && p[0].abs_diff(p[1]).max(p[1].abs_diff(p[2])) > 40).count();
            assert!(matches!(s, "👨‍👩‍👧" | "1️⃣") || colourful > 30, "{s}: in colour ({colourful})");
            if let Some(dir) = std::env::var_os("THREERS_EMOJI_OUT") {
                let straight: Vec<u8> = color
                    .chunks_exact(4)
                    .flat_map(|p| if p[3] == 0 { [0, 0, 0, 0] } else { [(p[0] as u32 * 255 / p[3] as u32) as u8, (p[1] as u32 * 255 / p[3] as u32) as u8, (p[2] as u32 * 255 / p[3] as u32) as u8, p[3]] })
                    .collect();
                let path = std::path::Path::new(&dir).join(format!("seq-{}.png", s.chars().next().unwrap() as u32));
                std::fs::write(path, crate::utils::png::encode_png(b.width, b.height, &straight)).unwrap();
            }
        }
    }

    #[test]
    fn emoji_presentation() {
        for c in ['😀', '🦊', '⭐', '⚓', '🚀', '✅', '🇺'] {
            assert!(is_emoji_presentation(c), "{c}");
        }
        for c in ['a', '1', '#', '©', '→', '✻', '➤', '❤', '☀'] {
            assert!(!is_emoji_presentation(c), "{c}");
        }
    }

    #[test]
    fn the_systems_emoji_in_colour() {
        let Some(e) = ColorGlyphs::system() else {
            assert!(!cfg!(target_os = "macos"), "a Mac has Apple Color Emoji");
            return;
        };
        assert!(e.has('😀') && !e.has('1'), "{e:?}");
        let g = e.raster('🦊', 32.0, 0.0);
        let color = g.color.as_ref().expect("in colour");
        assert!(g.width >= 24 && g.width <= 40 && g.height >= 24 && g.height <= 40, "{}×{}", g.width, g.height);
        // an orange fox: some pixels far more red than blue
        let orange = color.chunks_exact(4).filter(|p| p[3] > 200 && p[0] as i32 - p[2] as i32 > 100).count();
        assert!(orange > 20, "{orange}");
        // around the baseline: mostly above it
        assert!(g.offset_y < -(g.height as i32) / 2, "{}", g.offset_y);
    }

    /// Another `sbix` font (`THREERS_EMOJI_FONT`: iOS's, from a simulator
    /// runtime), read as the system's is.
    #[test]
    fn a_given_emoji_font() {
        let Some(path) = std::env::var_os("THREERS_EMOJI_FONT") else { return };
        let e = ColorGlyphs::open(std::path::Path::new(&path)).unwrap();
        let g = e.raster('🦊', 32.0, 0.0);
        let color = g.color.as_ref().expect("in colour");
        let orange = color.chunks_exact(4).filter(|p| p[3] > 200 && p[0] as i32 - p[2] as i32 > 100).count();
        assert!(orange > 20, "{e:?}: {orange}");
        eprintln!("{e:?}: 🦊 {}×{} at {}", g.width, g.height, g.offset_y);
    }

    /// The system draws an emoji (what iOS's font needs: its images
    /// aren't PNGs), and the PNG ones land where it puts them.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_system_draws_emoji_too() {
        let g = coretext::raster('🦊', 32.0, 0.0).expect("Core Text has the fox");
        let color = g.color.as_ref().unwrap();
        let orange = color.chunks_exact(4).filter(|p| p[3] > 200 && p[0] as i32 - p[2] as i32 > 100).count();
        assert!(orange > 20, "{orange}");
        assert!(coretext::raster('a', 32.0, 0.0).is_none_or(|g| g.color.is_some()));
        // the rows and columns with ink, from the pen at the baseline
        let ink = |g: &GlyphBitmap| {
            let on = |x: u32, y: u32| g.coverage[(y * g.width + x) as usize] > 60;
            let rows: Vec<i32> = (0..g.height).filter(|&y| (0..g.width).any(|x| on(x, y))).map(|y| g.offset_y + y as i32).collect();
            let cols: Vec<i32> = (0..g.width).filter(|&x| (0..g.height).any(|y| on(x, y))).map(|x| g.offset_x + x as i32).collect();
            [rows[0], *rows.last().unwrap(), cols[0], *cols.last().unwrap()]
        };
        let e = ColorGlyphs::system().unwrap();
        for size in [32.0, 48.0, 64.0] {
            for c in ['🦊', '🚀', '⭐', '😀', '🍉', '✅', '🇺', '🫠'] {
                let (ct, png) = (ink(&coretext::raster(c, size, 0.0).unwrap()), ink(&e.raster(c, size, 0.0)));
                for k in 0..4 {
                    assert!((ct[k] - png[k]).abs() <= 1, "{c} at {size}: {ct:?} by the system, {png:?} from the PNG");
                }
            }
        }
    }

    /// Text with an emoji: the letters in the text's colour, the emoji in
    /// its own.
    #[test]
    fn text_draws_emoji_in_colour() {
        if !crate::captions::CaptionFont::use_color_emoji() {
            assert!(!cfg!(target_os = "macos"), "a Mac has Apple Color Emoji");
            return;
        }
        let (w, h) = (200u32, 60u32);
        let mut frame = vec![0u8; (w * h * 4) as usize];
        let mut p = crate::captions::CaptionPainter::new();
        let mut style = p.style.clone();
        style.font_size = 32.0;
        style.color = [255, 255, 255, 255];
        style.background = [0, 0, 0, 0];
        style.shadow_color = [0, 0, 0, 0];
        style.outline_color = [0, 0, 0, 0];
        p.set_style(style);
        assert!(p.draw_text_at(&mut frame, w, h, 4.0, 4.0, "ab🦊"));
        let px: Vec<&[u8]> = frame.chunks_exact(4).filter(|p| p[3] > 200).collect();
        let white = px.iter().filter(|p| p[0] > 200 && p[1] > 200 && p[2] > 200).count();
        let orange = px.iter().filter(|p| p[0] as i32 - p[2] as i32 > 100).count();
        assert!(white > 20 && orange > 20, "white {white}, orange {orange}");
    }
}
