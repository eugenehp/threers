//! Whether a run of a face's glyphs composites by **maximum** rather than by union.
//!
//! Two glyphs of a run genuinely share edge pixels once they are placed at subpixel
//! positions, and the union `a + b - ab/255` is what two partial coverages of the same
//! pixel come to. That is right for Latin, where neighbouring letters only feather into
//! each other — and wrong for any script whose letters **join**, where the two glyphs
//! cover the *same* stroke and the union darkens it. Arabic is 30 to 57 levels too heavy
//! at every joint.
//!
//! Core Graphics has both composites and picks one per font, which is not a guess: it was
//! read off `CoreText`'s `EnumerateOverlappingGlyphs(CGContext*, TFont const&, …,
//! void (CFRange, bool) block_pointer)`, whose `bool` selects between
//!
//! ```text
//! ripc_DrawGlyphs -> draw_glyph_bitmaps -> render_glyphs          per-glyph blits
//! ripc_DrawGlyphs -> draw_glyph_bitmap_run_with_compositing       the maximum
//! ```
//!
//! and asked for directly with `CTFontDrawGlyphs`, which takes a glyph array and a
//! position array — so the reference can be asked for glyph A alone, glyph B alone and
//! the two together at the same pens, and the rule read off its own operands:
//!
//! ```text
//! SFNS      A 174  B  42  together 188   = 255 - floor((255-A)(255-B)/255)   a union
//! SFArabic  A 126  B 113  together 126   = max(A, B)
//! ```
//!
//! It really is a maximum and not a geometric union: offsetting one copy of a glyph
//! *vertically* puts a sliver of each in one cell side by side, where a union would be
//! their sum, and the maximum still wins 663 of 663 cells.
//!
//! # The chain this reproduces
//!
//! A `MERG` table decides alone; otherwise `TBaseFont::SupportsConnectedLanguage()`:
//!
//! | term | reads | decides |
//! | --- | --- | --- |
//! | a PostScript name of `.SFNS*`, or PingFang | `name` id 6 | **no** |
//! | `IsConnectedLanguage` of any design language | `meta`'s `dlng` | **yes** |
//! | one of three hard-coded names | `name` id 6 | **yes** |
//! | `GetOS2UnicodeRanges` ∩ eight bits | the **`cmap`** | **yes** |
//! | `TBaseFont::HasCursiveAttachment` | `GPOS` `curs` | **yes** |
//!
//! Three of those could not have been guessed. The ranges are *derived from the `cmap`*
//! rather than read from `OS/2`, so Rockwell — a Latin slab serif — composites because its
//! `cmap` covers one Arabic codepoint, `U+060B`, while Menlo *claims* the Arabic bit in
//! `OS/2` and has no Arabic at all. The 28 connected languages are a different set from
//! the 8 ranges. And `curs` is not unused: `Kannada MN` and `Kannada Sangam MN` carry it
//! and are the only fonts whose answer comes from that term.
//!
//! Checked against every font file on a macOS machine with the real path read off a
//! breakpoint rather than inferred from pixels: 239 of 239. Ported from `swiftui-font`'s
//! `composites_run`, where the reading is recorded in full.

/// The eight `OS/2` Unicode-range bits Core Text treats as connected scripts, as the
/// codepoint ranges they stand for: 13 Arabic, 14 N'Ko, 15 Devanagari, 16 Bengali,
/// 17 Gurmukhi, 53 Syriac, 71 Thaana, 81 Mongolian.
const CONNECTED_RANGES: &[(u32, u32)] = &[
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x07C0, 0x07FF),
    (0x0900, 0x097F),
    (0x0980, 0x09FF),
    (0x0A00, 0x0A7F),
    (0x0700, 0x074F),
    (0x0780, 0x07BF),
    (0x1800, 0x18AF),
];

/// The languages `IsConnectedLanguage` answers yes to, swept by calling it.
///
/// Arabic script, Syriac, N'Ko, Mongolian, Adlam, Hanifi Rohingya, and the Devanagari,
/// Bengali and Gurmukhi group. **Not** Hebrew, Thai, Lao, Khmer, Burmese, Tibetan,
/// Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam, Sinhala, Chinese, Japanese or
/// Korean.
const CONNECTED_LANGUAGES: &[&str] = &[
    "ar", "fa", "ur", "ps", "sd", "ckb", "ug", "bal", "syr", "nqo", "mn", "hi", "bn", "pa",
    "mr", "ne", "sa", "as", "kok", "doi", "brx", "mai", "bho", "mni", "new", "ks", "adlm",
    "rhg",
];

/// Core Text's hard-coded set, for connected scripts whose `meta` does not name theirs.
const CONNECTED_NAMES: &[&str] =
    &["NotoSansAdlam", "NotoSansHanifiRohingya", "NotoSansMongolian"];

fn u16_at(d: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*d.get(at)?, *d.get(at + 1)?]))
}

fn u32_at(d: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes([
        *d.get(at)?,
        *d.get(at + 1)?,
        *d.get(at + 2)?,
        *d.get(at + 3)?,
    ]))
}

/// The decision, from the tables a face has already parsed.
///
/// `covers` answers whether the `cmap` maps a codepoint, which is how the `OS/2` ranges
/// are *derived* — see the module comment for why reading `OS/2` itself is wrong.
pub(crate) fn composites_run(
    merg: Option<&[u8]>,
    meta: Option<&[u8]>,
    name: Option<&[u8]>,
    gpos: &[u8],
    covers: impl Fn(u32) -> bool,
) -> bool {
    if let Some(m) = merg {
        // `version`, `mergeClassCount`, `mergeDataOffset`: the matrix is `count * count`
        // bytes and its first entry is class 0 against itself. All six faces that have
        // one are Apple's `SF` family and all six say *do not merge*.
        let count = u16_at(m, 2).unwrap_or(0);
        let off = u16_at(m, 4).unwrap_or(0) as usize;
        return count > 0 && m.get(off).is_some_and(|b| b & 1 != 0);
    }
    if let Some(ps) = name.and_then(postscript_name) {
        if ps.starts_with(".SFNS") || ps.contains("PingFang") {
            return false;
        }
        if CONNECTED_NAMES.contains(&ps.as_str()) {
            return true;
        }
    }
    if let Some(dlng) = meta.and_then(design_languages) {
        // A comma-separated BCP-47 list; only the language subtag is compared, so
        // `zh-Hant` is asked as `zh` — and is not connected either way.
        for lang in dlng.split(',') {
            let base = lang.trim().split('-').next().unwrap_or("");
            if CONNECTED_LANGUAGES.contains(&base) {
                return true;
            }
        }
    }
    for (lo, hi) in CONNECTED_RANGES {
        // Every codepoint, not a sample: Rockwell's single `U+060B` is not at either end
        // of its range nor in the middle. A range is at most 256 wide and this runs once
        // per face.
        if (*lo..=*hi).any(&covers) {
            return true;
        }
    }
    has_curs(gpos)
}

/// Whether `GPOS`'s `FeatureList` names `curs` — `TBaseFont::HasCursiveAttachment`.
fn has_curs(gpos: &[u8]) -> bool {
    let Some(list) = u16_at(gpos, 6).map(usize::from) else {
        return false;
    };
    let Some(count) = u16_at(gpos, list).map(usize::from) else {
        return false;
    };
    (0..count).any(|i| gpos.get(list + 2 + i * 6..list + 6 + i * 6) == Some(b"curs"))
}

/// The `meta` table's `dlng` payload — the design languages, comma separated.
fn design_languages(meta: &[u8]) -> Option<&str> {
    let count = u32_at(meta, 12)? as usize;
    for i in 0..count {
        let rec = 16 + i * 12;
        if meta.get(rec..rec + 4)? != b"dlng" {
            continue;
        }
        let off = u32_at(meta, rec + 4)? as usize;
        let len = u32_at(meta, rec + 8)? as usize;
        return core::str::from_utf8(meta.get(off..off.checked_add(len)?)?).ok();
    }
    None
}

/// `name` id 6, the PostScript name. A `u16` encoding is read as its low bytes, which is
/// right for the ASCII a PostScript name is by definition.
fn postscript_name(name: &[u8]) -> Option<String> {
    let count = u16_at(name, 2)? as usize;
    let storage = u16_at(name, 4)? as usize;
    for i in 0..count {
        let rec = 6 + i * 12;
        if u16_at(name, rec + 6)? != 6 {
            continue;
        }
        let platform = u16_at(name, rec)?;
        let len = u16_at(name, rec + 8)? as usize;
        let off = u16_at(name, rec + 10)? as usize;
        let raw = name.get(storage + off..storage.checked_add(off)?.checked_add(len)?)?;
        return Some(if matches!(platform, 0 | 3) {
            raw.chunks(2).filter_map(|c| c.get(1).map(|b| *b as char)).collect()
        } else {
            raw.iter().map(|b| *b as char).collect()
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `MERG` table whose first matrix entry has bit 0 set, and one where it does not.
    fn merg(entry: u8) -> Vec<u8> {
        let mut t = vec![0u8; 12];
        t[2..4].copy_from_slice(&1u16.to_be_bytes()); // mergeClassCount
        t[4..6].copy_from_slice(&11u16.to_be_bytes()); // mergeDataOffset
        t[11] = entry;
        t
    }

    /// A `meta` table carrying one `dlng` string.
    fn meta(dlng: &str) -> Vec<u8> {
        let mut t = vec![0u8; 28];
        t[12..16].copy_from_slice(&1u32.to_be_bytes()); // dataMapsCount
        t[16..20].copy_from_slice(b"dlng");
        t[20..24].copy_from_slice(&28u32.to_be_bytes()); // offset
        t[24..28].copy_from_slice(&(dlng.len() as u32).to_be_bytes());
        t.extend_from_slice(dlng.as_bytes());
        t
    }

    /// A `GPOS` whose `FeatureList` holds one feature with the given tag.
    ///
    /// Header: version, then the script, feature and lookup list offsets. A
    /// `FeatureRecord` is a four-byte tag and a two-byte offset, so the first tag sits
    /// two bytes past the list.
    fn gpos(tag: &[u8; 4]) -> Vec<u8> {
        let mut t = vec![0u8; 10];
        t[6..8].copy_from_slice(&10u16.to_be_bytes()); // featureListOffset
        t.extend_from_slice(&1u16.to_be_bytes()); // featureCount
        t.extend_from_slice(tag);
        t.extend_from_slice(&0u16.to_be_bytes()); // featureOffset
        t
    }

    #[test]
    fn merg_decides_alone() {
        // Present and set: composite, whatever else the face says.
        assert!(composites_run(Some(&merg(1)), None, None, &[], |_| false));
        // Present and clear: do not, even for a face that covers Arabic. This is the
        // case that keeps the system font off the maximum.
        assert!(!composites_run(Some(&merg(0x22)), None, None, &[], |cp| cp == 0x628));
    }

    #[test]
    fn a_connected_design_language_is_enough() {
        for lang in ["ar", "fa,en", "en,hi", "ur-Arab"] {
            assert!(
                composites_run(None, Some(&meta(lang)), None, &[], |_| false),
                "{lang}"
            );
        }
        // Hebrew, Thai and Chinese are not connected, and a subtag is not a prefix match:
        // `zh-Hant` is asked as `zh`.
        for lang in ["he", "th", "zh-Hant", "ja,ko", "en"] {
            assert!(
                !composites_run(None, Some(&meta(lang)), None, &[], |_| false),
                "{lang}"
            );
        }
    }

    #[test]
    fn one_covered_codepoint_is_enough() {
        // Rockwell's case: a Latin face whose `cmap` covers `U+060B` and nothing else
        // connected. Sampling the ends or the middle of the range would miss it.
        assert!(composites_run(None, None, None, &[], |cp| cp == 0x60B));
        // And a face covering only Hebrew is not connected, which is the control.
        assert!(!composites_run(None, None, None, &[], |cp| (0x590..=0x5FF).contains(&cp)));
    }

    #[test]
    fn cursive_attachment_is_the_last_term() {
        assert!(composites_run(None, None, None, &gpos(b"curs"), |_| false));
        assert!(!composites_run(None, None, None, &gpos(b"kern"), |_| false));
    }
}
