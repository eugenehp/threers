//! Unicode canonical ordering and composition — what has to happen to a caption's text
//! before a single character can become a single glyph.
//!
//! Ported from `rswift`'s `swiftui-normalize`, where it is measured against Core Text.
//!
//! # Why a renderer needs this
//!
//! `t` followed by `U+0308 COMBINING DIAERESIS` is **one glyph** in every font tried:
//! Core Text composes the pair to `U+1E97` and draws that. Mapping the two characters
//! separately draws a `t` and a floating diaeresis at the wrong place, and it is not a
//! near miss — it is a different glyph count at a different width. Measured against Core
//! Text over 2000 fuzzed strings per font, the combining marks were the whole residual
//! once shaping was in: about 47 of 2000 in every one of six fonts. A caption reading
//! `"Noe\u{0308}l"` drew a floating diaeresis to the right of the `e` before this.
//!
//! # What it does, and what it deliberately does not
//!
//! Canonical **ordering** and **composition** — the second half of NFC — applied to the
//! string as given:
//!
//! 1. Marks are sorted by canonical combining class, stably, within each run of marks.
//! 2. Each starter absorbs the following characters it has a primary composite with,
//!    skipping any that a higher-or-equal class in between **blocks**.
//! 3. Hangul composes arithmetically rather than from a table.
//!
//! It does **not** decompose first, which full NFC would. That is not an omission for
//! convenience: Core Text does not either, and it shows. `Arial` maps `U+2126 OHM SIGN`
//! to its own glyph 159, where the singleton decomposition to `U+03A9` would have given
//! 525. So a renderer that normalises properly draws the wrong glyph here, and one that
//! only composes draws Core Text's.
//!
//! # Cluster indices survive
//!
//! [`compose`] returns each resulting character with the index of the character it
//! started from, because everything downstream is addressed by cluster: a selection
//! highlights characters, a caret lands between them, and a caption's own timing is
//! per character. Composing without carrying the mapping would make the result
//! unattributable to the text the author wrote.

use super::compose_tables as tables;

/// The canonical combining class of a character — 0 for a starter.
#[must_use]
pub fn ccc(c: char) -> u8 {
    let v = c as u32;
    let t = tables::CCC;
    let mut lo = 0usize;
    let mut hi = t.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let (first, last, k) = t[mid];
        if v < first {
            hi = mid;
        } else if v > last {
            lo = mid + 1;
        } else {
            return k;
        }
    }
    0
}

/// Hangul's three jamo ranges and the syllable block, from Unicode's own names.
mod hangul {
    pub const L_BASE: u32 = 0x1100;
    pub const V_BASE: u32 = 0x1161;
    pub const T_BASE: u32 = 0x11A7;
    pub const S_BASE: u32 = 0xAC00;
    pub const L_COUNT: u32 = 19;
    pub const V_COUNT: u32 = 21;
    pub const T_COUNT: u32 = 28;
    pub const S_COUNT: u32 = L_COUNT * V_COUNT * T_COUNT;
}

/// The primary composite of `a` and `b`, if they have one.
///
/// Composition exclusions are absent from the table rather than filtered here, because
/// `unorm2_composePair` already refused them when it was generated.
#[must_use]
pub fn compose_pair(a: char, b: char) -> Option<char> {
    use hangul::{L_BASE, L_COUNT, S_BASE, S_COUNT, T_BASE, T_COUNT, V_BASE, V_COUNT};
    let (x, y) = (a as u32, b as u32);
    // Hangul, arithmetically: leading + vowel, then syllable + trailing.
    let l = x.wrapping_sub(L_BASE);
    let v = y.wrapping_sub(V_BASE);
    if l < L_COUNT && v < V_COUNT {
        return char::from_u32(S_BASE + (l * V_COUNT + v) * T_COUNT);
    }
    let s = x.wrapping_sub(S_BASE);
    let t = y.wrapping_sub(T_BASE);
    if s < S_COUNT && s % T_COUNT == 0 && t > 0 && t < T_COUNT {
        return char::from_u32(x + t);
    }
    let table = tables::PAIRS;
    let mut lo = 0usize;
    let mut hi = table.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let (pa, pb, pc) = table[mid];
        match (pa, pb).cmp(&(x, y)) {
            core::cmp::Ordering::Less => lo = mid + 1,
            core::cmp::Ordering::Greater => hi = mid,
            core::cmp::Ordering::Equal => return char::from_u32(pc),
        }
    }
    None
}

/// Canonical ordering: sort each run of marks by combining class, stably.
///
/// The insertion sort is not laziness. The ordering must be **stable** within a class —
/// two marks of class 230 keep the order they were typed in, because they are rendered
/// outward in that order — and a run of marks is almost always one or two characters, so
/// an insertion sort is both correct by construction and the fastest thing here.
fn reorder(s: &mut [(char, usize)]) {
    for i in 1..s.len() {
        let k = ccc(s[i].0);
        if k == 0 {
            continue;
        }
        let mut j = i;
        while j > 0 {
            let prev = ccc(s[j - 1].0);
            if prev == 0 || prev <= k {
                break;
            }
            s.swap(j - 1, j);
            j -= 1;
        }
    }
}

/// Canonically order and compose `text`, keeping each output's source index.
///
/// The index is the position in `text` of the character the output **started** as, so a
/// composite is attributed to its base and a mark that did not compose keeps its own.
#[must_use]
pub fn compose(text: &[char]) -> Vec<(char, usize)> {
    let mut out: Vec<(char, usize)> = text.iter().copied().zip(0..).collect();
    reorder(&mut out);

    let mut result: Vec<(char, usize)> = Vec::with_capacity(out.len());
    // The last starter in `result`, which is what a later character composes onto — and
    // the highest class seen since, which is what decides whether it is blocked.
    let mut starter: Option<usize> = None;
    let mut blocker = 0u8;
    for (c, at) in out {
        let k = ccc(c);
        if let Some(s) = starter {
            // A character is blocked from the starter when the one before it has a class
            // **greater than or equal** to its own. Strictly greater is wrong and the
            // fuzz found it: `r` + diaeresis + acute are both class 230, so the acute is
            // blocked and stays loose — with `>` it reached the `r` and composed to
            // `U+0155`, a real glyph and the wrong one, in five words of 20,000.
            //
            // `blocker` is the highest class seen since the starter, which after
            // canonical ordering is also the *last* one — the classes of a mark run are
            // non-decreasing, and a mark that composes is removed rather than recorded.
            // A starter after a starter is blocked only if a mark intervened, which is
            // how Hangul and `U+09C7 + U+09BE` compose at all.
            let blocked = if k == 0 { blocker != 0 } else { blocker >= k };
            if !blocked {
                if let Some(joined) = compose_pair(result[s].0, c) {
                    result[s].0 = joined;
                    continue;
                }
            }
        }
        if k == 0 {
            starter = Some(result.len());
            blocker = 0;
        } else {
            blocker = blocker.max(k);
        }
        result.push((c, at));
    }
    result
}

/// Just the characters, for a caller that does not need the mapping.
#[must_use]
pub fn composed(text: &[char]) -> Vec<char> {
    compose(text).into_iter().map(|(c, _)| c).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(s: &str) -> Vec<char> {
        composed(&s.chars().collect::<Vec<_>>())
    }

    #[test]
    fn a_caption_with_a_decomposed_accent_is_one_glyph_per_letter() {
        // The reason this is here: four characters in, four letters out, not five.
        assert_eq!(run("Noe\u{0308}l"), "Noël".chars().collect::<Vec<_>>());
        assert_eq!(run("e\u{0301}te\u{0301}"), "été".chars().collect::<Vec<_>>());
    }

    #[test]
    fn a_pair_composes() {
        assert_eq!(run("t\u{0308}"), vec!['\u{1E97}']);
        assert_eq!(run("a\u{0301}"), vec!['\u{00E1}']);
        assert_eq!(run("A\u{030A}\u{0301}"), vec!['\u{01FA}']);
    }

    #[test]
    fn an_equal_class_blocks() {
        // Two marks of class 230: the first composes, the second cannot reach the starter.
        assert_eq!(run("a\u{0301}\u{0308}"), vec!['\u{00E1}', '\u{0308}']);
    }

    #[test]
    fn a_lower_class_does_not_block() {
        // Cedilla is 202 and diaeresis 230, so after reordering the cedilla comes first
        // and does not block the diaeresis from the `a`.
        assert_eq!(run("a\u{0308}\u{0327}"), vec!['\u{00E4}', '\u{0327}']);
        assert_eq!(run("a\u{0327}\u{0308}"), vec!['\u{00E4}', '\u{0327}']);
    }

    #[test]
    fn a_composed_mark_stops_blocking() {
        // Both marks are class 230, so the second looks blocked by the first — and is
        // not, because the first is *gone*: composing it into the starter removes it from
        // the sequence, which leaves the second adjacent to the starter. So `a` + 0308 +
        // 0304 is a single `U+01DF`, not `ä` and a loose macron. I expected the latter.
        assert_eq!(run("a\u{0308}\u{0304}"), vec!['\u{01DF}']);
        // Where the composite does not exist the mark stays and *then* blocks.
        assert_eq!(run("a\u{0301}\u{0308}"), vec!['\u{00E1}', '\u{0308}']);
    }

    #[test]
    fn reordering_is_stable_within_a_class() {
        // Two marks of the same class must keep the order they were typed in, because
        // they are drawn outward in that order. `U+0F71` (129) and `U+0F74` (132) differ,
        // so they sort; `U+0F74` twice must not.
        let a: Vec<char> = "\u{0F74}\u{0F71}".chars().collect();
        assert_eq!(compose(&a), vec![('\u{0F71}', 1), ('\u{0F74}', 0)]);
        let b: Vec<char> = "\u{0F71}\u{0F71}".chars().collect();
        assert_eq!(compose(&b), vec![('\u{0F71}', 0), ('\u{0F71}', 1)]);
    }

    #[test]
    fn hangul_composes_arithmetically() {
        assert_eq!(run("\u{1100}\u{1161}"), vec!['\u{AC00}']);
        assert_eq!(run("\u{1100}\u{1161}\u{11A8}"), vec!['\u{AC01}']);
        assert_eq!(run("\u{AC00}\u{11A8}"), vec!['\u{AC01}']);
        // A trailing jamo on an already-trailing syllable does not compose.
        assert_eq!(run("\u{AC01}\u{11A8}"), vec!['\u{AC01}', '\u{11A8}']);
    }

    #[test]
    fn a_composition_exclusion_stays_apart() {
        // `U+0344` is excluded, and `U+0958` is a composite that NFC decomposes — neither
        // may be produced here.
        assert_eq!(run("\u{0915}\u{093C}"), vec!['\u{0915}', '\u{093C}']);
    }

    #[test]
    fn nothing_is_decomposed() {
        // Core Text does not decompose, and nor does this: `Arial` maps the ohm sign to
        // its own glyph, not to omega's.
        assert_eq!(run("\u{2126}"), vec!['\u{2126}']);
        assert_eq!(run("\u{212B}"), vec!['\u{212B}']);
    }

    #[test]
    fn the_source_index_follows_the_base() {
        let text: Vec<char> = "xt\u{0308}y".chars().collect();
        let got = compose(&text);
        assert_eq!(got, vec![('x', 0), ('\u{1E97}', 1), ('y', 3)]);
    }

    #[test]
    fn a_lone_mark_survives() {
        assert_eq!(run("\u{0308}"), vec!['\u{0308}']);
        assert_eq!(run(""), Vec::<char>::new());
    }

    #[test]
    fn the_combining_class_table_reads() {
        assert_eq!(ccc('a'), 0);
        assert_eq!(ccc('\u{0308}'), 230);
        assert_eq!(ccc('\u{0327}'), 202);
        assert_eq!(ccc('\u{05B0}'), 10);
    }
}
