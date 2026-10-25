//! Which form an Arabic letter takes, from its neighbours.
//!
//! An Arabic letter has up to four shapes — isolated, initial, medial, final — and which
//! one is drawn depends on whether the letters either side of it can join. That is not a
//! property of the character but of its **context**, which is why `"ا"` and `"اب"` draw
//! the same first letter differently and why mapping codepoints to glyphs one at a time
//! cannot render Arabic at all.
//!
//! The rule, from the joining types:
//!
//! - a letter **joins to the right** (towards the previous letter) if the previous
//!   non-transparent character is dual-joining, left-joining or join-causing;
//! - it **joins to the left** (towards the next) if the next non-transparent character is
//!   dual-joining, right-joining or join-causing;
//! - and the form follows: joined on both sides is medial, on the right only is final, on
//!   the left only is initial, on neither is isolated.
//!
//! A **transparent** character — a vowel mark or a shadda — is skipped when looking at
//! neighbours and keeps no form of its own. That is what makes `"بَب"` join across the
//! fatha rather than breaking at it.
//!
//! The directions read oddly in a left-to-right source file: "right" means earlier in the
//! string, because Arabic runs right to left. The names follow the script, not the file.

use super::{joining_table, Joining};

/// The joining type of `c`.
#[must_use]
pub fn joining(c: char) -> Joining {
    let key = c as u32;
    let r = joining_table::RANGES;
    let mut lo = 0usize;
    let mut hi = r.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let (first, last, k) = r[mid];
        if key < first {
            hi = mid;
        } else if key > last {
            lo = mid + 1;
        } else {
            return k;
        }
    }
    Joining::NonJoining
}

/// Which of the four shapes a letter takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// Joined on neither side. The `isol` feature.
    Isolated,
    /// Joined towards the next letter only. The `init` feature.
    Initial,
    /// Joined on both sides. The `medi` feature.
    Medial,
    /// Joined towards the previous letter only. The `fina` feature.
    Final,
}

impl Form {
    /// The OpenType feature tag that selects this form.
    #[must_use]
    pub fn tag(self) -> [u8; 4] {
        match self {
            Self::Isolated => *b"isol",
            Self::Initial => *b"init",
            Self::Medial => *b"medi",
            Self::Final => *b"fina",
        }
    }
}

/// The form of every character in `text`, in logical order.
///
/// A character that takes no form — a mark, a space, Latin — gets [`Form::Isolated`],
/// which is also what the `isol` feature would do to it: nothing, since it has no
/// alternate.
#[must_use]
pub fn forms(text: &[char]) -> Vec<Form> {
    let kinds: Vec<Joining> = text.iter().copied().map(joining).collect();
    let prev = |i: usize| -> Option<Joining> {
        (0..i).rev().map(|j| kinds[j]).find(|k| *k != Joining::Transparent)
    };
    let next = |i: usize| -> Option<Joining> {
        kinds[i + 1..].iter().copied().find(|k| *k != Joining::Transparent)
    };
    (0..text.len())
        .map(|i| {
            if matches!(kinds[i], Joining::Transparent | Joining::NonJoining) {
                return Form::Isolated;
            }
            // Can this letter take a join on each side at all? A right-joining letter
            // joins only towards the previous character, a left-joining one only towards
            // the next, and a dual-joining one either way.
            //
            // **Join-causing counts as either.** `U+0640 ARABIC TATWEEL` is the case: it
            // is not a letter but a connecting stroke, so it joins on both sides and
            // takes the medial form itself — which is what makes it elongate a word
            // rather than interrupt it.
            let can_right = matches!(
                kinds[i],
                Joining::DualJoining | Joining::RightJoining | Joining::JoinCausing
            );
            let can_left = matches!(
                kinds[i],
                Joining::DualJoining | Joining::LeftJoining | Joining::JoinCausing
            );
            let joined_right = can_right
                && matches!(
                    prev(i),
                    Some(Joining::DualJoining | Joining::LeftJoining | Joining::JoinCausing)
                );
            let joined_left = can_left
                && matches!(
                    next(i),
                    Some(Joining::DualJoining | Joining::RightJoining | Joining::JoinCausing)
                );
            match (joined_right, joined_left) {
                (true, true) => Form::Medial,
                (true, false) => Form::Final,
                (false, true) => Form::Initial,
                (false, false) => Form::Isolated,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn forms_of(s: &str) -> Vec<Form> {
        forms(&s.chars().collect::<Vec<char>>())
    }

    #[test]
    fn the_joining_types_are_what_unicode_says() {
        assert_eq!(joining('\u{628}'), Joining::DualJoining); // beh
        assert_eq!(joining('\u{627}'), Joining::RightJoining); // alef
        assert_eq!(joining('\u{62F}'), Joining::RightJoining); // dal
        assert_eq!(joining('\u{64B}'), Joining::Transparent); // fathatan
        assert_eq!(joining('\u{640}'), Joining::JoinCausing); // tatweel
        assert_eq!(joining('a'), Joining::NonJoining);
        assert_eq!(joining(' '), Joining::NonJoining);
    }

    #[test]
    fn a_lone_letter_is_isolated() {
        assert_eq!(forms_of("\u{628}"), vec![Form::Isolated]);
        assert_eq!(forms_of("\u{627}"), vec![Form::Isolated]);
    }

    #[test]
    fn three_dual_joining_letters_make_an_initial_a_medial_and_a_final() {
        // beh beh beh: the first joins only forwards, the last only backwards.
        assert_eq!(
            forms_of("\u{628}\u{628}\u{628}"),
            vec![Form::Initial, Form::Medial, Form::Final]
        );
    }

    #[test]
    fn a_right_joining_letter_stops_the_chain() {
        // Alef joins only towards the previous letter, so what follows it starts again.
        // beh alef beh: initial, final, isolated.
        assert_eq!(
            forms_of("\u{628}\u{627}\u{628}"),
            vec![Form::Initial, Form::Final, Form::Isolated]
        );
    }

    #[test]
    fn a_mark_is_transparent_and_the_letters_join_across_it() {
        // beh fatha beh: the mark does not break the join.
        assert_eq!(
            forms_of("\u{628}\u{64E}\u{628}"),
            vec![Form::Initial, Form::Isolated, Form::Final]
        );
    }

    #[test]
    fn a_space_breaks_the_chain() {
        assert_eq!(
            forms_of("\u{628}\u{628} \u{628}\u{628}"),
            vec![Form::Initial, Form::Final, Form::Isolated, Form::Initial, Form::Final]
        );
    }

    #[test]
    fn tatweel_causes_a_join_without_being_a_letter() {
        // The elongation character joins on both sides, which is its whole purpose.
        assert_eq!(
            forms_of("\u{628}\u{640}\u{628}"),
            vec![Form::Initial, Form::Medial, Form::Final]
        );
    }

    #[test]
    fn the_table_covers_every_codepoint_without_gaps() {
        let mut next = 0u32;
        for (first, last, _) in joining_table::RANGES {
            assert_eq!(*first, next, "a gap or overlap before {first:#X}");
            next = last + 1;
        }
        assert_eq!(next, 0x11_0000);
    }
}
