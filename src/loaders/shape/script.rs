//! Which script a character belongs to, and where a run of one ends.
//!
//! Shaping needs a script tag, because a feature tag means different lookups under
//! different scripts and taking the union draws the wrong glyph. But a character does not
//! always say: Unicode gives a digit, a space and a quotation mark the **common** script
//! and a combining mark the **inherited** one, so `"عربي 2024"` is one Arabic run and not
//! three, and a mark after a Devanagari letter is Devanagari.
//!
//! So a run is split by *resolved* script, where common and inherited characters take the
//! script of what surrounds them — preferring what precedes, which is what a trailing
//! digit or a mark belongs to.


use super::script_table::SCRIPTS;

/// Unicode's *common* script: a digit, a space, most punctuation.
pub const COMMON: [u8; 4] = *b"zyyy";
/// Unicode's *inherited* script: a combining mark, which belongs to its base.
pub const INHERITED: [u8; 4] = *b"zinh";
/// What a run of nothing but common characters is shaped as.
pub const DEFAULT: [u8; 4] = *b"DFLT";

/// The script tag of one character, unresolved — possibly [`COMMON`] or [`INHERITED`].
#[must_use]
pub fn script_of(c: char) -> [u8; 4] {
    let v = c as u32;
    let mut lo = 0usize;
    let mut hi = SCRIPTS.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let (first, last, tag) = SCRIPTS[mid];
        if v < first {
            hi = mid;
        } else if v > last {
            lo = mid + 1;
        } else {
            return tag;
        }
    }
    DEFAULT
}

/// Whether a character carries a script of its own.
#[must_use]
pub fn is_real(tag: [u8; 4]) -> bool {
    tag != COMMON && tag != INHERITED && tag != DEFAULT
}

/// `text` split into runs of one resolved script, as `(start, end, tag)`.
///
/// A common or inherited character extends the run before it. A leading group of them —
/// `"2024 عربي"` starts with four digits and a space — takes the script of the first real
/// character that follows, because shaping the digits as `DFLT` and the letters as `arab`
/// would split a line that Core Text keeps whole.
#[must_use]
pub fn runs(text: &[char]) -> Vec<(usize, usize, [u8; 4])> {
    let mut out: Vec<(usize, usize, [u8; 4])> = Vec::new();
    for (i, c) in text.iter().enumerate() {
        let tag = script_of(*c);
        match out.last_mut() {
            // A real script continues its own run; anything else continues whatever run
            // it finds itself in.
            Some((_, end, prev)) if !is_real(tag) || *prev == tag => *end = i + 1,
            Some((_, end, prev)) if !is_real(*prev) => {
                // The run so far was all common: it belongs to this script after all.
                *prev = tag;
                *end = i + 1;
            }
            _ => out.push((i, i + 1, tag)),
        }
    }
    for r in &mut out {
        if !is_real(r.2) {
            r.2 = DEFAULT;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn split(s: &str) -> Vec<(usize, usize, String)> {
        runs(&s.chars().collect::<Vec<_>>())
            .into_iter()
            .map(|(a, b, t)| (a, b, String::from_utf8_lossy(&t).into_owned()))
            .collect()
    }

    #[test]
    fn a_character_knows_its_script() {
        assert_eq!(script_of('A'), *b"latn");
        assert_eq!(script_of('\u{0628}'), *b"arab");
        assert_eq!(script_of('\u{05D0}'), *b"hebr");
        assert_eq!(script_of('\u{0915}'), *b"deva");
        assert_eq!(script_of('\u{4E00}'), *b"hani");
        assert_eq!(script_of('0'), COMMON);
        assert_eq!(script_of('\u{0301}'), INHERITED);
    }

    #[test]
    fn a_digit_joins_the_run_before_it() {
        assert_eq!(split("\u{0628}\u{0628} 2024"), vec![(0, 7, "arab".into())]);
    }

    #[test]
    fn a_mark_joins_its_base() {
        assert_eq!(split("a\u{0301}"), vec![(0, 2, "latn".into())]);
    }

    #[test]
    fn a_leading_run_of_digits_takes_what_follows() {
        assert_eq!(split("2024 \u{0628}"), vec![(0, 6, "arab".into())]);
    }

    #[test]
    fn two_scripts_split() {
        assert_eq!(
            split("ab \u{0628}\u{0628} cd"),
            vec![(0, 3, "latn".into()), (3, 6, "arab".into()), (6, 8, "latn".into())]
        );
    }

    #[test]
    fn nothing_but_common_is_the_default() {
        assert_eq!(split("1 + 2 = 3"), vec![(0, 9, "DFLT".into())]);
        assert_eq!(split(""), vec![]);
    }
}
