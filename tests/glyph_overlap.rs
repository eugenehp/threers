//! Which composite a run of a face's glyphs gets, and that a joint stops darkening.
//!
//! Two glyphs of a run share edge pixels once they are placed at subpixel positions, and
//! the union `a + b - ab/255` is what two partial coverages of one pixel come to. For a
//! script whose letters **join** that is wrong: the tail of one letter and the head of the
//! next are the *same* stroke, drawn twice, and a union darkens every joint. See
//! `src/loaders/overlap.rs` for the rule and where it was read.
//!
//! **The pixels are not asserted here, on purpose.** Which composite is correct was
//! measured against Core Text, and that comparison cannot live in this crate — threers has
//! no Core Text binding and should not grow one, for the same reason `arabic_shaping.rs`
//! says so about shaping. `rswift`'s `swiftui-native/tests/fallback_pixels.rs` is the
//! pixel guarantee for the identical rule: 0 differing cells of 9,365,885 over Arabic,
//! Hebrew and mixed runs at 1x, 2x and 3x. What is checked here is that each face is
//! classified the way that measurement classified it.
//!
//! Returns rather than fails without the fonts.

use threers::captions::font::CaptionFont;

const ARABIC: &str = "/System/Library/Fonts/SFArabic.ttf";
const LATIN: &str = "/System/Library/Fonts/SFNS.ttf";
const HEBREW: &str = "/System/Library/Fonts/SFHebrew.ttf";

fn face(path: &str) -> Option<CaptionFont> {
    CaptionFont::from_ttf_bytes(&std::fs::read(path).ok()?).ok()
}

#[test]
fn a_joining_face_takes_the_maximum_and_a_latin_one_the_union() {
    // Arabic joins; Latin and Hebrew do not. Hebrew is the control that keeps the rule
    // from being "any right-to-left script" — its letters are separate.
    let expected = [(ARABIC, true), (LATIN, false), (HEBREW, false)];
    let mut seen = 0;
    for (path, want) in expected {
        let Some(f) = face(path) else { continue };
        seen += 1;
        assert_eq!(f.composites_run(), want, "{path}");
    }
    if seen == 0 {
        return;
    }
    assert!(seen >= 2, "only {seen} of the three faces were readable");
}
