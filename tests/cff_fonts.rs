//! `CFF` fonts load and draw, and a collection's faces are reachable.
//!
//! Sixty of the 369 font files macOS ships keep their outlines in `CFF ` or `CFF2` rather
//! than `glyf`, and they are the ones Asian and Indic text needs: every Hiragino face,
//! Apple SD Gothic Neo, Hiragino Sans GB and the whole Kohinoor family. All of them ship
//! as `.ttc` collections, so both pieces are needed before any of them loads.
//!
//! These assert that they **load with outlines**. Whether the outlines are *right* is a
//! different question with a different instrument: `rswift`'s
//! `swiftui-map/examples/cff_oracle` compares every curve against a reader that is
//! pixel-exact against `CTFontCreatePathForGlyph`, and reports 2407 of 2407 glyphs over
//! seven faces agreeing to 0.00017 font units — which is `f32` storage, not a difference
//! in reading. That comparison cannot live here: threers cannot depend on rswift.
//!
//! Returns rather than fails where a font is missing, so a machine without it still runs.

use threers::loaders::TtfFont;

/// `(path, what it is, outlines expected)`.
const FONTS: &[(&str, &str, bool)] = &[
    ("/System/Library/Fonts/LastResort.otf", "CFF, single face", true),
    ("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "CFF in a collection", true),
    ("/System/Library/Fonts/AppleSDGothicNeo.ttc", "CFF, Korean", true),
    ("/System/Library/Fonts/Kohinoor.ttc", "CFF, Devanagari", true),
    ("/System/Library/Fonts/SFIndia.ttc", "CFF2, variable", true),
    ("/System/Library/Fonts/SFNS.ttf", "glyf, for contrast", true),
];

#[test]
fn postscript_outlines_load_and_draw() {
    for (path, what, inked) in FONTS {
        let Ok(bytes) = std::fs::read(path) else { continue };
        let font = TtfFont::parse(&bytes).unwrap_or_else(|e| panic!("{what} ({path}): {e:?}"));
        assert!(font.units_per_em > 0, "{what}: no em");
        assert!(!font.glyphs.is_empty(), "{what}: no glyphs");
        let with_ink = font
            .glyphs
            .iter()
            .filter(|g| !g.shape.outline.curve_path.curves.is_empty())
            .count();
        if *inked {
            // A **majority**, not "nearly all". Last Resort is 4 of 7 and that is correct:
            // the oracle reports 7 of 7 agreeing, so the three without ink have none in
            // the reference either. A tighter ratio here asserts something about the font
            // rather than about the reader.
            assert!(
                with_ink * 2 > font.glyphs.len(),
                "{what}: only {with_ink} of {} glyphs drew anything",
                font.glyphs.len()
            );
        }
    }
}

#[test]
fn a_collection_reaches_past_its_first_face() {
    let path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc";
    let Ok(bytes) = std::fs::read(path) else { return };
    let first = TtfFont::parse_face(&bytes, 0, &[]).expect("face 0");
    assert!(!first.glyphs.is_empty());
    // Face 1 may or may not exist; what must not happen is a panic or a wrong face.
    match TtfFont::parse_face(&bytes, 1, &[]) {
        Ok(second) => assert!(!second.glyphs.is_empty(), "face 1 parsed as empty"),
        Err(e) => assert!(
            matches!(e, threers::loaders::TtfError::UnsupportedFormat),
            "face 1 failed for the wrong reason: {e:?}"
        ),
    }
    // And a face that cannot exist is refused rather than read from nowhere.
    assert!(TtfFont::parse_face(&bytes, 9999, &[]).is_err());
}

#[test]
fn a_cff_font_maps_characters_to_glyphs_that_draw() {
    let path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc";
    let Ok(bytes) = std::fs::read(path) else { return };
    let font = TtfFont::parse(&bytes).expect("parses");
    // Kana and a Han character, which is what this face exists for.
    for c in ['あ', 'ア', '日', '本', '語'] {
        let gid = font.cmap.get(&(c as u32)).copied().unwrap_or(0);
        assert!(gid != 0, "{c:?} is not in the cmap");
        let g = &font.glyphs[gid as usize];
        assert!(!g.shape.outline.curve_path.curves.is_empty(), "{c:?} drew nothing");
        assert!(g.advance_width > 0, "{c:?} has no advance");
    }
}
