//! Text from input methods: composition shown in place before it is
//! committed, as the platforms' IME APIs model it.
//!
//! Typing Chinese, Japanese or Korean, or an accented letter through a dead
//! key, goes through an input method. Keys build up *preedit* text that the
//! field shows where the caret is, usually underlined, while the method's
//! candidate window offers what it could become, and a *commit* replaces it
//! with the final text. [`Event::Ime`](super::input::Event::Ime) carries
//! both (winit's `Ime::Preedit` and `Ime::Commit` map onto it one to one),
//! and [`TextField`](super::TextField) and [`TextArea`](super::TextArea)
//! keep the preedit as a marked range of their text.
//!
//! ```
//! use threers_ui::TextField;
//!
//! let mut field = TextField::new("I said ");
//! field.preedit("にほ", Some((6, 6)));
//! assert_eq!(field.text, "I said にほ");
//! assert_eq!(field.marked, Some((7, 9)));
//! field.preedit("日本", Some((6, 6)));
//! field.commit("日本");
//! assert_eq!(field.text, "I said 日本");
//! assert_eq!((field.marked, field.caret), (None, 9));
//! ```

/// What an input method sends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ime {
    /// The text being composed, replacing any before it, and where the
    /// method's cursor (or selection) is in it, in bytes; `None` hides the
    /// cursor. Empty text ends composing without committing.
    Preedit {
        text: String,
        cursor: Option<(usize, usize)>,
    },
    /// The final text, in place of the preedit.
    Commit(String),
}

/// Characters in `text` before byte `i`.
pub(crate) fn chars_before(text: &str, i: usize) -> usize {
    let mut i = i.min(text.len());
    while !text.is_char_boundary(i) {
        i -= 1;
    }
    text[..i].chars().count()
}

/// Composition for a text model with `text`, `anchor`, `caret` (in chars),
/// `marked` and an `insert` that replaces the selection.
macro_rules! composition {
    ($t:ty) => {
        impl $t {
            /// Shows `text` being composed in place of what was being
            /// composed (or of the selection, to start), marked, with the
            /// caret (or selection) at `cursor`'s bytes within it.
            pub fn preedit(&mut self, text: &str, cursor: Option<(usize, usize)>) {
                if let Some((a, z)) = self.marked.take() {
                    self.anchor = a;
                    self.caret = z;
                }
                let (start, _) = self.selection();
                self.insert(text);
                let n = text.chars().count();
                if n == 0 {
                    return;
                }
                self.marked = Some((start, start + n));
                let (c0, c1) = cursor.map_or((n, n), |(a, b)| {
                    (
                        $crate::ime::chars_before(text, a),
                        $crate::ime::chars_before(text, b),
                    )
                });
                self.anchor = start + c0;
                self.caret = start + c1;
            }

            /// Replaces what is being composed (or the selection) with
            /// `text`, the caret after it.
            pub fn commit(&mut self, text: &str) {
                if let Some((a, z)) = self.marked.take() {
                    self.anchor = a;
                    self.caret = z;
                }
                self.insert(text);
            }

            /// Drops what is being composed, as Escape in a candidate
            /// window does.
            pub fn cancel_preedit(&mut self) {
                self.preedit("", None);
            }

            /// Applies what an input method sent.
            pub fn ime(&mut self, ime: &$crate::ime::Ime) {
                match ime {
                    $crate::ime::Ime::Preedit { text, cursor } => self.preedit(text, *cursor),
                    $crate::ime::Ime::Commit(text) => self.commit(text),
                }
            }
        }
    };
}

pub(crate) use composition;

#[cfg(test)]
mod tests {
    use crate::{TextArea, TextField};

    #[test]
    fn preedit_replaces_the_selection_then_itself() {
        let mut f = TextField::new("abc");
        f.anchor = 1;
        f.caret = 2;
        f.preedit("x", None);
        assert_eq!((f.text.as_str(), f.marked), ("axc", Some((1, 2))));
        f.preedit("xyz", Some((1, 2)));
        assert_eq!(
            (f.text.as_str(), f.marked, f.anchor, f.caret),
            ("axyzc", Some((1, 4)), 2, 3)
        );
        f.cancel_preedit();
        assert_eq!((f.text.as_str(), f.marked, f.caret), ("ac", None, 1));
    }

    #[test]
    fn a_text_area_composes_on_its_line() {
        let mut t = TextArea::new("one\ntwo");
        t.move_to(4, false);
        t.preedit("ñ", None);
        t.commit("ñ");
        assert_eq!(t.text, "one\nñtwo");
        assert_eq!(t.marked, None);
    }
}
