//! The UI's bidi (which places carets and styled pieces) orders lines as
//! the one that draws them does.

use threers::captions::bidi::{self, Direction};
use threers::ui::environment::LayoutDirection;

#[test]
fn the_ui_orders_lines_as_they_are_drawn() {
    let parts: Vec<char> = "abz אבש \u{5b0}ابي\u{64b}٣ 0123 +-,.:/%$#()[]!?\"–…·×".chars().collect();
    let mut seed = 0x2545_f491u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for k in 0..40_000 {
        let n = 1 + next() as usize % 14;
        let s: String = (0..n).map(|_| parts[next() as usize % parts.len()]).collect();
        // (a mark straight on a bracket is as the bracket, by N0's last
        // note: what draws leaves it as it was)
        let c: Vec<char> = s.chars().collect();
        let mark = |c: char| matches!(c, '\u{5b0}' | '\u{64b}');
        if c.windows(2).any(|w| "()[]".contains(w[0]) && mark(w[1])) {
            continue;
        }
        for (a, b) in [
            (LayoutDirection::LeftToRight, Direction::Ltr),
            (LayoutDirection::RightToLeft, Direction::Rtl),
        ] {
            let ours = threers::ui::bidi::levels(&s, a);
            let theirs = bidi::resolve(&s, b);
            let mut theirs = theirs.levels;
            // Spaces at the end are the paragraph's (L1), in either.
            let chars: Vec<char> = s.chars().collect();
            for (c, l) in chars.iter().zip(&mut theirs).rev() {
                if *c != ' ' {
                    break;
                }
                *l = u8::from(a == LayoutDirection::RightToLeft);
            }
            assert_eq!(ours, theirs, "{k}: {s:?} {a:?}");
        }
    }
}

/// The marks that tell a painter a line's direction take no room.
#[test]
fn marks_are_not_drawn() {
    let font = threers::CaptionFont::builtin();
    let n = |s: &str| threers::captions::order::visual(s, &font, 16.0).len();
    assert_eq!(n("\u{200f}ab"), 2);
    assert_eq!(n("\u{200e}"), 0);
    let mut font = threers::CaptionFont::builtin();
    assert_eq!(font.measure("\u{200f}ab", 16.0, 1.0), font.measure("ab", 16.0, 1.0));
}
