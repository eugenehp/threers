//! **L4**: mirroring a character in a right-to-left run.
//!
//! A bracket is drawn as its mirror when it sits at an odd level: `(` as `)`, `<` as `>`,
//! `⟨` as `⟩`. This is a *rendering* step and not a reordering one — the character does
//! not change, only the glyph chosen for it — which is why it is the last rule and why it
//! is here rather than in [`crate::resolve`].
//!
//! The table is generated from `u_charMirror`; see `examples/gen_mirror.rs`.

use super::mirror_table::PAIRS;

/// The character to draw in place of `c` at `level`.
///
/// `c` itself at an even level, and its mirror at an odd one. A character with no mirror
/// comes back unchanged at any level.
#[must_use]
pub fn at_level(c: char, level: u8) -> char {
    if level.is_multiple_of(2) {
        return c;
    }
    mirror(c).unwrap_or(c)
}

/// `c`'s mirror, if it has one that differs from itself.
#[must_use]
pub fn mirror(c: char) -> Option<char> {
    let key = c as u32;
    let mut lo = 0usize;
    let mut hi = PAIRS.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let (a, b) = PAIRS[mid];
        if key < a {
            hi = mid;
        } else if key > a {
            lo = mid + 1;
        } else {
            return if b == key { None } else { char::from_u32(b) };
        }
    }
    None
}

/// Whether `c` is mirrored at all, including the few that map to themselves.
#[must_use]
pub fn is_mirrored(c: char) -> bool {
    let key = c as u32;
    PAIRS.binary_search_by_key(&key, |(a, _)| *a).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brackets_mirror_and_letters_do_not() {
        assert_eq!(mirror('('), Some(')'));
        assert_eq!(mirror(')'), Some('('));
        assert_eq!(mirror('<'), Some('>'));
        assert_eq!(mirror('['), Some(']'));
        assert_eq!(mirror('{'), Some('}'));
        assert_eq!(mirror('a'), None);
        assert_eq!(mirror('\u{05D0}'), None); // aleph
    }

    #[test]
    fn mirroring_only_happens_at_an_odd_level() {
        assert_eq!(at_level('(', 0), '(');
        assert_eq!(at_level('(', 1), ')');
        assert_eq!(at_level('(', 2), '(');
        assert_eq!(at_level('a', 1), 'a');
    }

    #[test]
    fn the_table_is_sorted_so_the_search_is_valid() {
        for w in PAIRS.windows(2) {
            assert!(w[0].0 < w[1].0, "{:?} then {:?}", w[0], w[1]);
        }
    }
}
