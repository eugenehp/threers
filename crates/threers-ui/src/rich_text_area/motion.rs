//! Where the caret goes: by character (a whole grapheme: an emoji with its
//! skin tone, a flag, a letter with its accents, an Indic or Southeast
//! Asian letter with its vowel signs and the consonants a virama joins to
//! it), by word, by line (keeping a column), by paragraph, page and the
//! whole text.

use super::{Document, Pos};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Motion {
    Left,
    Right,
    WordLeft,
    WordRight,
    Up,
    Down,
    LineStart,
    LineEnd,
    ParaStart,
    ParaEnd,
    DocStart,
    DocEnd,
    PageUp,
    PageDown,
}

/// Part of a word, for moving and selecting by words.
pub fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '\'' || c == '’' || threers_spell::segment::script_of(c).is_some()
}

/// Where words end in Thai, Lao, Khmer or Burmese in `text` (none in other text).
fn sea_ends(text: &str) -> Vec<usize> {
    if text.chars().any(|c| threers_spell::segment::script_of(c).is_some()) {
        threers_spell::segment::boundaries(text)
    } else {
        Vec::new()
    }
}

/// Whether `c` goes on in the grapheme before it (Unicode's Extend and
/// SpacingMark: combining marks, vowel signs, viramas, variation
/// selectors, skin tones).
fn extends(c: u32) -> bool {
    matches!(c,
        0x300..=0x36F | 0x483..=0x489 | 0x591..=0x5BD | 0x5BF | 0x5C1..=0x5C2 | 0x5C4..=0x5C5 | 0x5C7 | 0x610..=0x61A | 0x64B..=0x65F | 0x670
        | 0x6D6..=0x6DC | 0x6DF..=0x6E4 | 0x6E7..=0x6E8 | 0x6EA..=0x6ED
        // Devanagari to Malayalam
        | 0x900..=0x903 | 0x93A..=0x93C | 0x93E..=0x94F | 0x951..=0x957 | 0x962..=0x963
        | 0x981..=0x983 | 0x9BC | 0x9BE..=0x9C4 | 0x9C7..=0x9C8 | 0x9CB..=0x9CD | 0x9D7 | 0x9E2..=0x9E3 | 0x9FE
        | 0xA01..=0xA03 | 0xA3C | 0xA3E..=0xA42 | 0xA47..=0xA48 | 0xA4B..=0xA4D | 0xA51 | 0xA70..=0xA71 | 0xA75
        | 0xA81..=0xA83 | 0xABC | 0xABE..=0xAC5 | 0xAC7..=0xAC9 | 0xACB..=0xACD | 0xAE2..=0xAE3 | 0xAFA..=0xAFF
        | 0xB01..=0xB03 | 0xB3C | 0xB3E..=0xB44 | 0xB47..=0xB48 | 0xB4B..=0xB4D | 0xB55..=0xB57 | 0xB62..=0xB63
        | 0xB82 | 0xBBE..=0xBC2 | 0xBC6..=0xBC8 | 0xBCA..=0xBCD | 0xBD7
        | 0xC00..=0xC04 | 0xC3C | 0xC3E..=0xC44 | 0xC46..=0xC48 | 0xC4A..=0xC4D | 0xC55..=0xC56 | 0xC62..=0xC63
        | 0xC81..=0xC83 | 0xCBC | 0xCBE..=0xCC4 | 0xCC6..=0xCC8 | 0xCCA..=0xCCD | 0xCD5..=0xCD6 | 0xCE2..=0xCE3
        | 0xD00..=0xD03 | 0xD3B..=0xD3C | 0xD3E..=0xD44 | 0xD46..=0xD48 | 0xD4A..=0xD4D | 0xD57 | 0xD62..=0xD63
        // Sinhala, Thai, Lao, Tibetan, Myanmar, Khmer
        | 0xD81..=0xD83 | 0xDCA | 0xDCF..=0xDD4 | 0xDD6 | 0xDD8..=0xDDF | 0xDF2..=0xDF3
        | 0xE31 | 0xE33..=0xE3A | 0xE47..=0xE4E | 0xEB1 | 0xEB3..=0xEBC | 0xEC8..=0xECE
        | 0xF18..=0xF19 | 0xF35 | 0xF37 | 0xF39 | 0xF3E..=0xF3F | 0xF71..=0xF84 | 0xF86..=0xF87 | 0xF8D..=0xFBC | 0xFC6
        | 0x102B..=0x103E | 0x1056..=0x1059 | 0x105E..=0x1060 | 0x1062..=0x1064 | 0x1067..=0x106D | 0x1071..=0x1074 | 0x1082..=0x108D | 0x108F | 0x109A..=0x109D
        | 0x17B4..=0x17D3 | 0x17DD | 0x1CD0..=0x1CF9
        | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x200C | 0x20D0..=0x20FF | 0xFE00..=0xFE0F | 0xFE20..=0xFE2F | 0x1F3FB..=0x1F3FF | 0xE0020..=0xE007F)
}

/// A virama that joins the consonant after it into one grapheme (Unicode's
/// conjunct linkers, and Khmer's COENG and Myanmar's stacking virama).
fn links(c: u32) -> bool {
    matches!(c, 0x94D | 0x9CD | 0xACD | 0xB4D | 0xC4D | 0xD4D | 0x17D2 | 0x1039)
}

/// A letter a virama may join on: in the virama's own block.
fn same_block_letter(virama: u32, c: u32) -> bool {
    let consonant = (0x900..0xD80).contains(&c) && matches!(c & 0x7F, 0x15..=0x39 | 0x58..=0x5F | 0x78..=0x7F);
    consonant && c & !0x7F == virama & !0x7F || (0x1780..=0x17A2).contains(&c) && virama == 0x17D2 || (0x1000..=0x1021).contains(&c) && virama == 0x1039
}

/// The end of the grapheme at `off` (its marks, joiners and modifiers with it).
pub fn next_boundary(s: &str, off: usize) -> usize {
    let mut it = s[off..].char_indices().peekable();
    let Some((_, first)) = it.next() else { return s.len() };
    let mut end = off + first.len_utf8();
    let regional = |c: char| (0x1F1E6..=0x1F1FF).contains(&(c as u32));
    let mut pairs = regional(first);
    let mut last = first as u32;
    while let Some(&(i, c)) = it.peek() {
        let cu = c as u32;
        let joins = extends(cu) || links(last) && same_block_letter(last, cu);
        if joins {
            end = off + i + c.len_utf8();
            it.next();
            last = cu;
        } else if cu == 0x200D {
            // a joiner takes what follows it too
            it.next();
            end = off + i + c.len_utf8();
            if let Some((j, d)) = it.next() {
                end = off + j + d.len_utf8();
            }
        } else if pairs && regional(c) {
            end = off + i + c.len_utf8();
            it.next();
            pairs = false;
        } else {
            break;
        }
    }
    end
}

/// The start of the grapheme before `off`.
pub fn prev_boundary(s: &str, off: usize) -> usize {
    if off == 0 {
        return 0;
    }
    let mut start = s[..off].char_indices().next_back().map_or(0, |(i, _)| i);
    for _ in 0..16 {
        if start == 0 {
            break;
        }
        let before = s[..start].char_indices().next_back().map_or(0, |(i, _)| i);
        if next_boundary(s, before) >= off {
            start = before;
        } else {
            break;
        }
    }
    start
}

/// The word round `off` (or the run of spaces or punctuation there): byte range.
pub fn word_at(text: &str, off: usize) -> (usize, usize) {
    let off = off.min(text.len());
    let at = text[off..].chars().next().or_else(|| text[..off].chars().next_back());
    let Some(c) = at else { return (off, off) };
    let class = |ch: char| if is_word_char(ch) { 0 } else if ch.is_whitespace() { 1 } else { 2 };
    let k = class(c);
    let mut a = off;
    for (i, ch) in text[..off].char_indices().rev() {
        if class(ch) != k {
            break;
        }
        a = i;
    }
    let mut b = off;
    for (i, ch) in text[off..].char_indices() {
        if class(ch) != k {
            break;
        }
        b = off + i + ch.len_utf8();
    }
    // in text without spaces between words: the word the list finds
    let ends = sea_ends(text);
    if let Some(&e) = ends.iter().find(|&&e| e > off && e <= b) {
        b = e;
    }
    if let Some(&s) = ends.iter().rev().find(|&&e| e <= off && e > a) {
        a = s;
    }
    (a, b)
}

/// The start of the word before `off` (spaces and punctuation passed over).
pub fn word_left(text: &str, off: usize) -> usize {
    let chars: Vec<(usize, char)> = text[..off].char_indices().collect();
    let mut i = chars.len();
    while i > 0 && !is_word_char(chars[i - 1].1) {
        i -= 1;
    }
    while i > 0 && is_word_char(chars[i - 1].1) {
        i -= 1;
    }
    let at = chars.get(i).map_or(0, |c| c.0);
    sea_ends(text).into_iter().rev().find(|&e| e < off && e > at).unwrap_or(at)
}

/// The end of the word after `off`.
pub fn word_right(text: &str, off: usize) -> usize {
    let mut it = text[off..].char_indices().peekable();
    while it.peek().is_some_and(|(_, c)| !is_word_char(*c)) {
        it.next();
    }
    let start = it.peek().map_or(text.len(), |(i, _)| off + i);
    while it.peek().is_some_and(|(_, c)| is_word_char(*c)) {
        it.next();
    }
    let at = it.peek().map_or(text.len(), |(i, _)| off + i);
    sea_ends(text).into_iter().find(|&e| e > start && e < at).unwrap_or(at)
}

/// Where `m` takes the caret from `p`, and whether it is drawn at a wrapped
/// line's end. `goal_x` is the column up and down keep.
pub fn step(doc: &dyn Document, p: Pos, upstream: bool, m: Motion, goal_x: f32) -> (Pos, bool) {
    let text = doc.text(p.para);
    let last = doc.paragraphs().saturating_sub(1);
    match m {
        Motion::Left => {
            if p.off == 0 {
                if p.para == 0 {
                    return (p, false);
                }
                return (Pos::new(p.para - 1, doc.text(p.para - 1).len()), false);
            }
            let mut off = prev_boundary(&text, p.off);
            while off > 0 && doc.hidden(Pos::new(p.para, off)) {
                off = prev_boundary(&text, off);
            }
            (Pos::new(p.para, off), false)
        }
        Motion::Right => {
            if p.off >= text.len() {
                if p.para >= last {
                    return (p, false);
                }
                return (Pos::new(p.para + 1, 0), false);
            }
            let mut off = next_boundary(&text, p.off);
            while off < text.len() && doc.hidden(Pos::new(p.para, off)) {
                off = next_boundary(&text, off);
            }
            (Pos::new(p.para, off), false)
        }
        Motion::WordLeft if p.off == 0 => step(doc, p, upstream, Motion::Left, goal_x),
        Motion::WordLeft => (Pos::new(p.para, word_left(&text, p.off)), false),
        Motion::WordRight if p.off >= text.len() => step(doc, p, upstream, Motion::Right, goal_x),
        Motion::WordRight => (Pos::new(p.para, word_right(&text, p.off)), false),
        Motion::Up | Motion::Down => match doc.line_step(p, upstream, m == Motion::Down, goal_x) {
            Some(r) => r,
            None if m == Motion::Up => (Pos::new(p.para, 0), false),
            None => (Pos::new(p.para, text.len()), false),
        },
        Motion::PageUp | Motion::PageDown => {
            // a page's worth of lines
            let r = doc.caret_rect(p, upstream);
            let mut at = (p, upstream);
            for _ in 0..400 {
                let Some(next) = doc.line_step(at.0, at.1, m == Motion::PageDown, goal_x) else { break };
                at = next;
                if (doc.caret_rect(next.0, next.1).y - r.y).abs() > 600.0 {
                    break;
                }
            }
            at
        }
        Motion::LineStart => (doc.line_ends(p, upstream).0, false),
        Motion::LineEnd => {
            let end = doc.line_ends(p, upstream).1;
            // a wrapped line's end is drawn there
            let wraps = end.off < text.len() && end.para == p.para;
            (end, wraps)
        }
        Motion::ParaStart => (Pos::new(p.para, 0), false),
        Motion::ParaEnd => (Pos::new(p.para, text.len()), false),
        Motion::DocStart => (Pos::default(), false),
        Motion::DocEnd => (Pos::new(last, doc.text(last).len()), false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thai_words_are_found_by_the_word_list() {
        let t = "ภาษาไทยง่าย";
        assert_eq!(word_at(t, 0), (0, "ภาษา".len()));
        assert_eq!(word_at(t, "ภาษาไ".len()), ("ภาษา".len(), "ภาษาไทย".len()));
        assert_eq!(word_right(t, 0), "ภาษา".len());
        assert_eq!(word_left(t, t.len()), "ภาษาไทย".len());
    }

    #[test]
    fn indic_letters_keep_their_vowel_signs_and_conjuncts() {
        fn graphemes(s: &str) -> Vec<&str> {
            let mut out = Vec::new();
            let mut i = 0;
            while i < s.len() {
                let j = next_boundary(s, i);
                out.push(&s[i..j]);
                i = j;
            }
            out
        }
        assert_eq!(graphemes("नमस्ते"), ["न", "म", "स्ते"]);
        assert_eq!(graphemes("क्षत्रिय"), ["क्ष", "त्रि", "य"]);
        assert_eq!(graphemes("தமிழ்"), ["த", "மி", "ழ்"]);
        assert_eq!(graphemes("น้ำ"), ["น้ำ"]);
        assert_eq!(graphemes("ស្រី"), ["ស្រី"]);
        assert_eq!(prev_boundary("नमस्ते", "नमस्ते".len()), "नम".len());
        // a virama before a digit or a space joins nothing
        assert_eq!(graphemes("क्१"), ["क्", "१"]);
    }

    #[test]
    fn graphemes_move_whole() {
        let s = "a👍🏽b🇫🇷é";
        let after_thumb = next_boundary(s, 1);
        assert_eq!(&s[1..after_thumb], "👍🏽");
        let flag_end = next_boundary(s, after_thumb + 1);
        assert_eq!(&s[after_thumb + 1..flag_end], "🇫🇷");
        assert_eq!(prev_boundary(s, flag_end), after_thumb + 1);
        assert_eq!(prev_boundary(s, after_thumb), 1);
        let e = "e\u{301}x";
        assert_eq!(next_boundary(e, 0), 3);
    }

    #[test]
    fn words() {
        let t = "Hello, world's end";
        assert_eq!(word_at(t, 8), (7, 14));
        assert_eq!(word_left(t, 14), 7);
        assert_eq!(word_right(t, 5), 14);
        assert_eq!(word_at(t, 5), (5, 6));
    }
}
