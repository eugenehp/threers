//! Arabic letters join, and the joining depends on the neighbours.
//!
//! The property that needs no oracle: an Arabic letter has up to four shapes, so an
//! **unshaped** pipeline draws the same glyph for `ب` whatever surrounds it, and a shaped
//! one does not. Every assertion here fails on a one-codepoint-one-glyph pipeline and
//! passes on a correct one, without having to know which glyph is right.
//!
//! Which glyph *is* right is measured elsewhere, against Core Text: `rswift`'s
//! `swiftui-shape/examples/ct_parity` reports 16 of 16 corpus words and 1987 of 2000
//! fuzzed Arabic words exact, and `latn_parity` 0 differing of 20,000 fuzzed Latin
//! strings in each of six system fonts. That comparison cannot live here — threers has
//! no Core Text binding and should not grow one.
//!
//! Returns rather than fails without the font, so a machine without it still runs.

use threers::captions::font::CaptionFont;
use threers::captions::order;

const ARABIC: &str = "/System/Library/Fonts/SFArabic.ttf";

fn face() -> Option<CaptionFont> {
    CaptionFont::from_ttf_bytes(&std::fs::read(ARABIC).ok()?).ok()
}

/// The glyph ids `text` draws as, in drawing order.
fn gids(font: &CaptionFont, text: &str) -> Vec<u32> {
    order::visual(text, font, 24.0).into_iter().map(|p| p.gid).collect()
}

#[test]
fn a_letter_takes_a_different_form_from_its_neighbours() {
    let Some(f) = face() else { return };
    // Beh alone is isolated; between two behs it is medial; at the start, initial.
    let alone = gids(&f, "\u{0628}");
    let pair = gids(&f, "\u{0628}\u{0628}");
    assert_eq!(alone.len(), 1, "one letter drew {} glyphs", alone.len());
    assert!(!pair.is_empty());
    // Drawing order is right to left, so the *last* glyph of the pair is the first letter.
    let first_of_pair = *pair.last().expect("non-empty");
    assert_ne!(
        alone[0], first_of_pair,
        "beh drew the same glyph alone and before another letter: not shaped"
    );
}

#[test]
fn alef_breaks_the_join() {
    let Some(f) = face() else { return };
    // Alef joins to its right only, so the beh after it cannot be medial.
    let through = gids(&f, "\u{0628}\u{0628}\u{0628}");
    let broken = gids(&f, "\u{0628}\u{627}\u{0628}");
    assert_ne!(through, broken, "an alef in the middle changed nothing");
}

#[test]
fn lam_alef_becomes_one_glyph() {
    let Some(f) = face() else { return };
    // The one ligature every Arabic font has, and it is mandatory rather than optional.
    let two = gids(&f, "\u{0644}\u{0627}");
    assert_eq!(two.len(), 1, "lam-alef drew {} glyphs, not a ligature", two.len());
}

#[test]
fn a_shaped_run_still_reads_right_to_left() {
    let Some(f) = face() else { return };
    // Shaping must not undo the reordering: the clusters descend.
    let run = order::visual("\u{0645}\u{0631}\u{062D}\u{0628}\u{0627}", &f, 24.0);
    assert!(run.len() >= 5, "only {} glyphs", run.len());
    assert!(run.iter().all(|p| p.rtl), "an Arabic run was not marked right to left");
    let first = run.first().expect("non-empty").cluster;
    let last = run.last().expect("non-empty").cluster;
    assert!(first > last, "clusters {first} then {last}: not reversed");
}

#[test]
fn a_character_the_face_cannot_draw_still_takes_a_place() {
    let Some(f) = face() else { return };
    // SF Arabic has no Latin at all, so this is the missing-glyph case — and it must come
    // back as glyph 0 per character rather than as nothing. The shaper this was ported
    // from drops an uncovered character, which is right behind a fallback cascade and
    // wrong for a single face: the text would silently vanish instead of drawing notdef.
    for text in ["abc", "Hello", "123"] {
        let shaped = gids(&f, text);
        let naive: Vec<u32> = text.chars().map(|c| f.gid(c)).collect();
        assert_eq!(shaped, naive, "{text:?} did not survive shaping");
        assert_eq!(shaped.len(), text.chars().count());
    }
}

#[test]
fn the_width_matches_the_drawing() {
    let Some(mut f) = face() else { return };
    // `measure` and the raster walk must agree, which they only do by walking the same
    // order. A ligature makes this concrete: two characters, one advance.
    let lam_alef = f.measure("\u{0644}\u{0627}", 24.0, 0.0);
    let lam = f.measure("\u{0644}", 24.0, 0.0);
    let alef = f.measure("\u{0627}", 24.0, 0.0);
    assert!(
        lam_alef < lam + alef,
        "the ligature measured {lam_alef} against {} for the two letters apart",
        lam + alef
    );
}

#[test]
fn a_dot_is_a_mark_and_hangs_off_its_skeleton() {
    let Some(f) = face() else { return };
    // `كتاب` is four letters with nothing combining in the string, and it still has a
    // mark in it: SF Arabic's `rclt` decomposes the teh into a skeleton plus its two
    // dots, and the dots are mark glyphs with no codepoint of their own.
    let run = order::visual("\u{0643}\u{062A}\u{0627}\u{0628}", &f, 24.0);
    let attached: Vec<usize> =
        (0..run.len()).filter(|k| run[*k].attach.is_some()).collect();
    assert!(
        !attached.is_empty(),
        "no glyph of كتاب attached to anything, so the dots sit at the skeleton's origin"
    );
    for k in attached {
        let (to, dx, _dy) = run[k].attach.expect("attached");
        assert!(to < run.len(), "attached to glyph {to} of {}", run.len());
        assert_ne!(to, k, "a glyph attached to itself");
        // Measured in rswift against CTRunGetPositions: the dot sits 0.3857pt right of
        // its base at 24pt. The anchor delta is what puts it there.
        assert!(dx.abs() > 0.01, "the anchor delta is {dx}, so nothing was read");
    }
}

#[test]
fn the_dot_lands_where_core_text_puts_it() {
    let Some(f) = face() else { return };
    // The number itself, which is the one thing a property test cannot check. rswift
    // reports 0.3857pt at 24pt for this word's dot, exact against CTRunGetPositions.
    let run = order::visual("\u{0643}\u{062A}\u{0627}\u{0628}", &f, 24.0);
    let dx = run
        .iter()
        .filter_map(|p| p.attach.map(|(_, dx, _)| dx))
        .find(|dx| dx.abs() > 0.01)
        .expect("an attached glyph with an offset");
    assert!(
        (dx - 0.3857).abs() < 0.01,
        "the dot is {dx:.4}pt from its base; Core Text says 0.3857"
    );
}

#[test]
fn a_face_without_mark_positioning_pays_nothing_for_it() {
    let Ok(bytes) = std::fs::read("/System/Library/Fonts/SFNS.ttf") else { return };
    let f = CaptionFont::from_ttf_bytes(&bytes).expect("parses");
    // SF Pro registers `kern` and one `ss09` lookup and no mark positioning, so the
    // GPOS and GDEF bytes are not kept at all — and nothing attaches.
    let run = order::visual("Hamburgefonstiv", &f, 24.0);
    assert!(run.iter().all(|p| p.attach.is_none()));
}
