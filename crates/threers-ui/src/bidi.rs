//! Which way text reads: the Unicode bidirectional algorithm (UAX #9) for a
//! line at a time, without explicit embeddings. Carets, selections and
//! styled pieces are placed by it; the painters order the glyphs of each
//! line themselves, told the paragraph's direction by [`marked`].
//!
//! ```
//! use threers_ui::bidi::{base, levels};
//! use threers_ui::environment::LayoutDirection::*;
//!
//! // Hebrew with a Latin word and a number: both inside read left to right.
//! let text = "\u{5d0}\u{5d1} ab 12";
//! assert_eq!(base(text, LeftToRight), RightToLeft);
//! assert_eq!(levels(text, RightToLeft), [1, 1, 1, 2, 2, 2, 2, 2]);
//! // In a layout that reads right to left, a paragraph with any such
//! // letter does too, whatever it starts with; one with none does not.
//! assert_eq!(base("ab \u{5d0}", RightToLeft), RightToLeft);
//! assert_eq!(base("ab.", RightToLeft), LeftToRight);
//! ```

use std::borrow::Cow;
use std::ops::Range;

use crate::environment::LayoutDirection;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    L,
    R,
    Al,
    En,
    An,
    Es,
    Et,
    Cs,
    Nsm,
    Bn,
    Ws,
    On,
}

fn class(c: char) -> Class {
    use Class::*;
    match c as u32 {
        0x30..=0x39
        | 0x6F0..=0x6F9
        | 0xB2
        | 0xB3
        | 0xB9
        | 0x2070
        | 0x2074..=0x2079
        | 0x2080..=0x2089
        | 0xFF10..=0xFF19 => En,
        0x660..=0x669 | 0x66B | 0x66C | 0x600..=0x605 | 0x6DD | 0x8E2 => An,
        0x2B | 0x2D | 0x207A | 0x207B | 0x208A | 0x208B | 0x2212 => Es,
        0x23..=0x25
        | 0xA2..=0xA5
        | 0xB0
        | 0xB1
        | 0x58F
        | 0x609
        | 0x60A
        | 0x66A
        | 0x2030..=0x2034
        | 0x20A0..=0x20CF
        | 0x2213 => Et,
        0x2C | 0x2E | 0x2F | 0x3A | 0xA0 | 0x60C | 0x202F | 0x2044 => Cs,
        0x300..=0x36F
        | 0x591..=0x5BD
        | 0x5BF
        | 0x5C1
        | 0x5C2
        | 0x5C4
        | 0x5C5
        | 0x5C7
        | 0x610..=0x61A
        | 0x64B..=0x65F
        | 0x670
        | 0x6D6..=0x6DC
        | 0x6DF..=0x6E4
        | 0x6E7
        | 0x6E8
        | 0x6EA..=0x6ED
        | 0x711
        | 0x730..=0x74A
        | 0x7A6..=0x7B0
        | 0x8D3..=0x8E1
        | 0x8E3..=0x902
        | 0x200D
        | 0xFB1E
        | 0xFE00..=0xFE0F
        | 0x20D0..=0x20F0 => Nsm,
        0x200E => L,
        0x61C => Al,
        0x200F => R,
        0x0..=0x8
        | 0xE..=0x1B
        | 0x7F..=0x84
        | 0x86..=0x9F
        | 0xAD
        | 0x200B
        | 0x200C
        | 0x2060..=0x2064
        | 0xFEFF => Bn,
        0x9..=0xD
        | 0x20
        | 0x85
        | 0x1C..=0x1F
        | 0x2000..=0x200A
        | 0x2028
        | 0x2029
        | 0x205F
        | 0x3000 => Ws,
        0x606 | 0x607 | 0x60E | 0x60F | 0x6DE | 0x6E9 | 0xFD3E | 0xFD3F => On,
        0x590..=0x5FF
        | 0x7C0..=0x85F
        | 0xFB1D..=0xFB4F
        | 0x10800..=0x10CFF
        | 0x10D40..=0x10EBF
        | 0x10F00..=0x10F2F
        | 0x10F70..=0x10FFF
        | 0x1E800..=0x1EC6F
        | 0x1ECC0..=0x1ECFF
        | 0x1ED50..=0x1EDFF
        | 0x1EF00..=0x1EFFF => R,
        0x606..=0x7BF
        | 0x860..=0x8FF
        | 0xFB50..=0xFDFF
        | 0xFE70..=0xFEFC
        | 0x10D00..=0x10D3F
        | 0x10EC0..=0x10EFF
        | 0x10F30..=0x10F6F
        | 0x1EC70..=0x1ECBF
        | 0x1ED00..=0x1ED4F
        | 0x1EE00..=0x1EEFF => Al,
        _ if c.is_alphabetic() => L,
        _ => On,
    }
}

/// Whether any of `text` reads right to left.
pub fn has_rtl(text: &str) -> bool {
    // Nothing before U+0590 does, nor anything whose UTF-8 starts lower.
    text.bytes().any(|b| b >= 0xD6)
        && text
            .chars()
            .any(|c| matches!(class(c), Class::R | Class::Al | Class::An))
}

/// The first letter's direction, if any has one.
pub(crate) fn first_strong(text: &str) -> Option<LayoutDirection> {
    text.chars().find_map(|c| match class(c) {
        Class::L => Some(LayoutDirection::LeftToRight),
        Class::R | Class::Al => Some(LayoutDirection::RightToLeft),
        _ => None,
    })
}

/// Which way paragraph `text` reads in a layout that reads `layout`: as its
/// first letter; or right to left in such a layout if any of it does.
pub fn base(text: &str, layout: LayoutDirection) -> LayoutDirection {
    if !has_rtl(text) {
        return LayoutDirection::LeftToRight;
    }
    match layout {
        LayoutDirection::RightToLeft => LayoutDirection::RightToLeft,
        LayoutDirection::LeftToRight => first_strong(text).unwrap_or_default(),
    }
}

/// `line` of a paragraph that reads `base`, for a painter that finds a
/// line's direction by its first letter: an invisible mark (LRM, RLM) put
/// before it where that would be the other.
pub fn marked(line: &str, base: LayoutDirection) -> Cow<'_, str> {
    if !has_rtl(line) && base == LayoutDirection::LeftToRight || first_strong(line) == Some(base) {
        return Cow::Borrowed(line);
    }
    let mark = match base {
        LayoutDirection::LeftToRight => '\u{200e}',
        LayoutDirection::RightToLeft => '\u{200f}',
    };
    Cow::Owned(format!("{mark}{line}"))
}

/// `key` in text that reads right to left: the left arrow goes on through
/// it, and the right one back.
pub(crate) fn reading_key<'a>(text: &str, key: &'a str) -> &'a str {
    match (key, base(text, LayoutDirection::LeftToRight)) {
        ("ArrowLeft", LayoutDirection::RightToLeft) => "ArrowRight",
        ("ArrowRight", LayoutDirection::RightToLeft) => "ArrowLeft",
        _ => key,
    }
}

fn closing(c: char) -> Option<char> {
    Some(match c {
        '(' => ')',
        '[' => ']',
        '{' => '}',
        '\u{2329}' => '\u{232A}',
        '\u{3008}' => '\u{3009}',
        '\u{300C}' => '\u{300D}',
        '\u{FF08}' => '\u{FF09}',
        _ => return None,
    })
}

/// The embedding level of each character of `text`: even read left to
/// right, odd right to left.
pub fn levels(text: &str, base: LayoutDirection) -> Vec<u8> {
    use Class::*;
    let chars: Vec<char> = text.chars().collect();
    let level = u8::from(base == LayoutDirection::RightToLeft);
    let edge = if level == 1 { R } else { L };
    let original: Vec<Class> = chars.iter().map(|c| class(*c)).collect();
    let mut cls = original.clone();
    let n = cls.len();
    // W1: a mark is as what it is on.
    let mut before = edge;
    for c in &mut cls {
        match *c {
            Nsm => *c = before,
            Bn => {}
            o => before = o,
        }
    }
    // W2, W3: numbers after Arabic letters, which are right to left.
    let mut strong = edge;
    for c in &mut cls {
        match *c {
            L | R | Al => strong = *c,
            En if strong == Al => *c = An,
            _ => {}
        }
        if *c == Al {
            *c = R;
        }
    }
    // W4: a separator between two numbers.
    let seen = |cls: &[Class], i: usize, back: bool| {
        let mut k = i;
        loop {
            k = if back { k.checked_sub(1)? } else { k + 1 };
            match *cls.get(k)? {
                Bn => {}
                c => return Some(c),
            }
        }
    };
    for i in 0..n {
        let (a, b) = (seen(&cls, i, true), seen(&cls, i, false));
        match (cls[i], a, b) {
            (Es | Cs, Some(En), Some(En)) => cls[i] = En,
            (Cs, Some(An), Some(An)) => cls[i] = An,
            _ => {}
        }
    }
    // W5, W6: signs beside a number are part of it; others are not.
    let mut i = 0;
    while i < n {
        if !matches!(cls[i], Et | Bn) {
            i += 1;
            continue;
        }
        let from = i;
        while i < n && matches!(cls[i], Et | Bn) {
            i += 1;
        }
        let beside = from > 0 && cls[from - 1] == En || i < n && cls[i] == En;
        if beside && cls[from..i].contains(&Et) {
            cls[from..i].fill(En);
        }
    }
    for c in &mut cls {
        if matches!(c, Es | Et | Cs) {
            *c = On;
        }
    }
    // W7: a number after Latin is of it.
    let mut strong = edge;
    for c in &mut cls {
        match *c {
            L | R => strong = *c,
            En if strong == L => *c = L,
            _ => {}
        }
    }
    let dir = |c: Class| match c {
        L => Some(L),
        R | En | An => Some(R),
        _ => None,
    };
    // N0: a pair of brackets as what is in it, or before it.
    let mut open: Vec<(char, usize)> = vec![];
    let mut pairs: Vec<(usize, usize)> = vec![];
    for (i, &c) in chars.iter().enumerate() {
        if cls[i] != On {
            continue;
        }
        if let Some(to) = closing(c) {
            open.push((to, i));
        } else if let Some(k) = open.iter().rposition(|(to, _)| *to == c) {
            pairs.push((open[k].1, i));
            open.truncate(k);
        }
    }
    pairs.sort_unstable();
    let opposite = if edge == R { L } else { R };
    for (a, b) in pairs {
        let inside = |cls: &[Class], want| cls[a + 1..b].iter().any(|c| dir(*c) == Some(want));
        let to = if inside(&cls, edge) {
            edge
        } else if inside(&cls, opposite) {
            match cls[..a].iter().rev().find_map(|c| dir(*c)).unwrap_or(edge) {
                d if d == opposite => opposite,
                _ => edge,
            }
        } else {
            continue;
        };
        for i in [a, b] {
            cls[i] = to;
            // Marks on a bracket follow it.
            for k in i + 1..n {
                match original[k] {
                    Nsm => cls[k] = to,
                    Bn => {}
                    _ => break,
                }
            }
        }
    }
    // N1, N2: what has no direction, between two of the same, has theirs.
    let mut i = 0;
    while i < n {
        if !matches!(cls[i], On | Ws | Bn | Nsm) {
            i += 1;
            continue;
        }
        let from = i;
        while i < n && matches!(cls[i], On | Ws | Bn | Nsm) {
            i += 1;
        }
        let a = from
            .checked_sub(1)
            .and_then(|k| dir(cls[k]))
            .unwrap_or(edge);
        let b = cls.get(i).and_then(|c| dir(*c)).unwrap_or(edge);
        cls[from..i].fill(if a == b { a } else { edge });
    }
    // I1, I2.
    let mut out: Vec<u8> = cls
        .iter()
        .map(|c| match (*c, level) {
            (L, 0) | (R, 1) => level,
            (L, _) | (R, _) => level + 1,
            (_, 0) => 2,
            _ => 2,
        })
        .collect();
    // L1: spaces at the line's end are the paragraph's.
    for (k, o) in original.iter().enumerate().rev() {
        match o {
            Ws | Bn => out[k] = level,
            _ => break,
        }
    }
    out
}

/// A stretch of a line that reads one way, as bytes of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Run {
    pub bytes: Range<usize>,
    pub rtl: bool,
}

/// `line`'s runs from left to right as drawn.
pub fn runs(line: &str, base: LayoutDirection) -> Vec<Run> {
    let levels = levels(line, base);
    let mut runs: Vec<(Range<usize>, u8)> = vec![];
    for ((i, c), l) in line.char_indices().zip(&levels) {
        let end = i + c.len_utf8();
        match runs.last_mut() {
            Some((r, k)) if k == l => r.end = end,
            _ => runs.push((i..end, *l)),
        }
    }
    // L2, of whole runs.
    let top = levels.iter().max().copied().unwrap_or(0);
    for at in (1..=top).rev() {
        let mut i = 0;
        while i < runs.len() {
            let from = i;
            while i < runs.len() && runs[i].1 >= at {
                i += 1;
            }
            runs[from..i].reverse();
            i = i.max(from + 1);
        }
    }
    runs.into_iter()
        .map(|(bytes, l)| Run {
            bytes,
            rtl: l % 2 == 1,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use LayoutDirection::{LeftToRight as Ltr, RightToLeft as Rtl};

    fn drawn(s: &str, base: LayoutDirection) -> String {
        runs(s, base)
            .iter()
            .map(|r| {
                let t = &s[r.bytes.clone()];
                if r.rtl {
                    t.chars().rev().collect()
                } else {
                    t.to_owned()
                }
            })
            .collect()
    }

    #[test]
    fn runs_are_in_the_order_they_are_drawn() {
        assert_eq!(drawn("abc", Ltr), "abc");
        assert_eq!(drawn("אבג", Rtl), "גבא");
        assert_eq!(drawn("א 12 ב", Rtl), "ב 12 א");
        assert_eq!(drawn("abc אב def", Ltr), "abc בא def");
        // The full stop of a right-to-left line is at its left.
        assert_eq!(drawn("אב ab.", Rtl), ".ab בא");
        // Brackets round Latin in Hebrew stay together.
        assert_eq!(drawn("אב (ab) ג", Rtl), "ג )ab( בא");
        assert_eq!(drawn("x (אב)", Ltr), "x (בא)");
    }

    #[test]
    fn a_line_is_marked_only_where_it_would_be_taken_for_the_other() {
        assert_eq!(marked("abc", Ltr), "abc");
        assert_eq!(marked("אב", Rtl), "אב");
        assert_eq!(marked("ab א", Rtl), "\u{200f}ab א");
        assert_eq!(marked("12.", Rtl), "\u{200f}12.");
        assert_eq!(marked("א ab", Ltr), "\u{200e}א ab");
        assert_eq!(base("12 + 3", Rtl), Ltr);
    }
}
