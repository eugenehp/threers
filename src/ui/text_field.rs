//! A single-line text field's editing state: what an `<input type=text>`
//! does with keys, without the drawing (an app lays out its text, selection
//! and caret with [`TextField::spans`]).

/// The modifier keys held.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KeyModifiers {
    pub shift: bool,
    pub control: bool,
    pub alt: bool,
    pub meta: bool,
}

/// What a key did to a field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKey {
    /// The text changed.
    Edited,
    /// Only the caret or selection moved.
    Moved,
    /// Enter.
    Commit,
    /// Escape.
    Cancel,
    /// Tab, or Shift+Tab (`backward`).
    Tab { backward: bool },
    /// A key the field does not use (a shortcut, say).
    Ignored,
}

/// Text and a selection `anchor..caret` (in characters; equal for a bare
/// caret).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TextField {
    pub text: String,
    pub anchor: usize,
    pub caret: usize,
}

impl TextField {
    /// `text` with the caret at its end.
    pub fn new(text: impl Into<String>) -> Self {
        let text = text.into();
        let n = text.chars().count();
        Self { text, anchor: n, caret: n }
    }

    /// `text`, all selected (as a field focused by `select()`).
    pub fn selected(text: impl Into<String>) -> Self {
        let mut f = Self::new(text);
        f.anchor = 0;
        f
    }

    pub fn len(&self) -> usize {
        self.text.chars().count()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// The selection as an ordered range.
    pub fn selection(&self) -> (usize, usize) {
        (self.anchor.min(self.caret), self.anchor.max(self.caret))
    }

    pub fn select_all(&mut self) {
        self.anchor = 0;
        self.caret = self.len();
    }

    /// Replaces the selection with `s`.
    pub fn insert(&mut self, s: &str) {
        let (a, z) = self.selection();
        let chars: Vec<char> = self.text.chars().collect();
        self.text = chars[..a].iter().copied().chain(s.chars()).chain(chars[z..].iter().copied()).collect();
        self.caret = a + s.chars().count();
        self.anchor = self.caret;
    }

    fn move_to(&mut self, caret: usize, extend: bool) {
        self.caret = caret.min(self.len());
        if !extend {
            self.anchor = self.caret;
        }
    }

    /// A key (DOM `KeyboardEvent.key`) pressed in the field; characters
    /// `accept` refuses are dropped, as an `input` handler filtering them
    /// would.
    pub fn key(&mut self, key: &str, m: KeyModifiers, accept: impl Fn(char) -> bool) -> FieldKey {
        let command = m.control || m.meta;
        let n = self.len();
        let (a, z) = self.selection();
        match key {
            "Enter" => FieldKey::Commit,
            "Escape" => FieldKey::Cancel,
            "Tab" => FieldKey::Tab { backward: m.shift },
            "ArrowLeft" => {
                let to = if a != z && !m.shift { a } else { self.caret.saturating_sub(1) };
                self.move_to(if command { 0 } else { to }, m.shift);
                FieldKey::Moved
            }
            "ArrowRight" => {
                let to = if a != z && !m.shift { z } else { (self.caret + 1).min(n) };
                self.move_to(if command { n } else { to }, m.shift);
                FieldKey::Moved
            }
            "Home" | "ArrowUp" => {
                self.move_to(0, m.shift);
                FieldKey::Moved
            }
            "End" | "ArrowDown" => {
                self.move_to(n, m.shift);
                FieldKey::Moved
            }
            "a" if command => {
                self.select_all();
                FieldKey::Moved
            }
            "Backspace" | "Delete" => {
                if a == z {
                    if key == "Backspace" && a > 0 {
                        self.anchor = a - 1;
                    } else if key == "Delete" && z < n {
                        self.caret = z + 1;
                    } else {
                        return FieldKey::Moved;
                    }
                }
                self.insert("");
                FieldKey::Edited
            }
            k if !command && k.chars().count() == 1 => {
                let c = k.chars().next().unwrap();
                if accept(c) {
                    self.insert(k);
                    FieldKey::Edited
                } else {
                    FieldKey::Moved
                }
            }
            _ => FieldKey::Ignored,
        }
    }

    /// The text before the selection, the selection, and after it: for an
    /// app to measure where the selection (or caret) is drawn.
    pub fn spans(&self) -> (String, String, String) {
        let (a, z) = self.selection();
        let chars: Vec<char> = self.text.chars().collect();
        (chars[..a].iter().collect(), chars[a..z].iter().collect(), chars[z..].iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edits_as_an_input_does() {
        let any = |_: char| true;
        let mut f = TextField::selected("1000");
        assert_eq!(f.key("5", KeyModifiers::default(), any), FieldKey::Edited);
        assert_eq!(f.text, "5");
        f.key("x", KeyModifiers::default(), |c| c.is_ascii_digit());
        assert_eq!(f.text, "5");
        f.key("2", KeyModifiers::default(), any);
        f.key("ArrowLeft", KeyModifiers { shift: true, ..Default::default() }, any);
        assert_eq!(f.spans(), ("5".into(), "2".into(), "".into()));
        f.key("Backspace", KeyModifiers::default(), any);
        assert_eq!((f.text.as_str(), f.caret), ("5", 1));
        f.key("Home", KeyModifiers::default(), any);
        f.key("Delete", KeyModifiers::default(), any);
        assert!(f.is_empty());
        assert_eq!(f.key("Tab", KeyModifiers { shift: true, ..Default::default() }, any), FieldKey::Tab { backward: true });
        assert_eq!(f.key("c", KeyModifiers { meta: true, ..Default::default() }, any), FieldKey::Ignored);
    }
}
