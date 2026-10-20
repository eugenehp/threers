//! A multi-line text editor's editing state (a `<textarea>`, or CodeMirror
//! with its default keys), without the drawing: lines, a selection
//! `anchor..caret` in characters, and what keys do to them.

use super::{FieldKey, KeyModifiers};

/// Text and a selection `anchor..caret` in characters (equal for a bare
/// caret), and the column Up/Down keep to (`goal`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TextArea {
    pub text: String,
    pub anchor: usize,
    pub caret: usize,
    goal: Option<usize>,
    /// Spaces Tab inserts (CodeMirror's `indentUnit`).
    pub indent_unit: usize,
}

impl TextArea {
    /// `text` with the caret at its start (as CodeMirror's `setValue`).
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into(), anchor: 0, caret: 0, goal: None, indent_unit: 2 }
    }

    pub fn len(&self) -> usize {
        self.text.chars().count()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// The lines (without their `\n`).
    pub fn lines(&self) -> Vec<&str> {
        self.text.split('\n').collect()
    }

    /// The (line, column) of character index `i`.
    pub fn line_col(&self, i: usize) -> (usize, usize) {
        let mut line = 0;
        let mut col = 0;
        for (k, c) in self.text.chars().enumerate() {
            if k == i {
                break;
            }
            if c == '\n' {
                line += 1;
                col = 0;
            } else {
                col += 1;
            }
        }
        (line, col)
    }

    /// The character index of (line, column), the column clamped to the
    /// line.
    pub fn index_of(&self, line: usize, col: usize) -> usize {
        let lines = self.lines();
        let line = line.min(lines.len() - 1);
        let start: usize = lines[..line].iter().map(|l| l.chars().count() + 1).sum();
        start + col.min(lines[line].chars().count())
    }

    /// The selection as an ordered range.
    pub fn selection(&self) -> (usize, usize) {
        (self.anchor.min(self.caret), self.anchor.max(self.caret))
    }

    /// The selected columns of `line`, if the selection touches it: (start,
    /// end, and whether it runs on past the line's end).
    pub fn selection_in_line(&self, line: usize) -> Option<(usize, usize, bool)> {
        let (a, z) = self.selection();
        if a == z {
            return None;
        }
        let (la, ca) = self.line_col(a);
        let (lz, cz) = self.line_col(z);
        if line < la || line > lz {
            return None;
        }
        let len = self.lines().get(line).map_or(0, |l| l.chars().count());
        let start = if line == la { ca } else { 0 };
        let end = if line == lz { cz } else { len };
        Some((start, end, line < lz))
    }

    pub fn select_all(&mut self) {
        self.anchor = 0;
        self.caret = self.len();
        self.goal = None;
    }

    /// Replaces the selection with `s`.
    pub fn insert(&mut self, s: &str) {
        let (a, z) = self.selection();
        let chars: Vec<char> = self.text.chars().collect();
        self.text = chars[..a].iter().copied().chain(s.chars()).chain(chars[z..].iter().copied()).collect();
        self.caret = a + s.chars().count();
        self.anchor = self.caret;
        self.goal = None;
    }

    /// Puts the caret at `i` (extending the selection if `extend`).
    pub fn move_to(&mut self, i: usize, extend: bool) {
        self.caret = i.min(self.len());
        if !extend {
            self.anchor = self.caret;
        }
    }

    fn vertical(&mut self, lines: isize, extend: bool) {
        let (line, col) = self.line_col(self.caret);
        let goal = *self.goal.get_or_insert(col);
        let n = self.lines().len() as isize;
        let target = line as isize + lines;
        let i = if target < 0 {
            0
        } else if target >= n {
            self.len()
        } else {
            self.index_of(target as usize, goal)
        };
        let keep = self.goal;
        self.move_to(i, extend);
        self.goal = keep;
    }

    /// A key (DOM `KeyboardEvent.key`) pressed in the editor.
    pub fn key(&mut self, key: &str, m: KeyModifiers) -> FieldKey {
        let command = m.control || m.meta;
        let (a, z) = self.selection();
        if !matches!(key, "ArrowUp" | "ArrowDown" | "PageUp" | "PageDown") {
            self.goal = None;
        }
        match key {
            "Escape" => FieldKey::Cancel,
            "Enter" => {
                // `newlineAndIndent`, keeping the line's indentation.
                let (line, _) = self.line_col(a);
                let indent: String = self.lines()[line].chars().take_while(|c| *c == ' ' || *c == '\t').collect();
                self.insert(&format!("\n{indent}"));
                FieldKey::Edited
            }
            "Tab" => {
                self.insert(&" ".repeat(self.indent_unit));
                FieldKey::Edited
            }
            "ArrowLeft" => {
                let to = if a != z && !m.shift { a } else { self.caret.saturating_sub(1) };
                self.move_to(to, m.shift);
                FieldKey::Moved
            }
            "ArrowRight" => {
                let to = if a != z && !m.shift { z } else { (self.caret + 1).min(self.len()) };
                self.move_to(to, m.shift);
                FieldKey::Moved
            }
            "ArrowUp" | "ArrowDown" | "PageUp" | "PageDown" => {
                let step = match key {
                    "ArrowUp" => -1,
                    "ArrowDown" => 1,
                    "PageUp" => -20,
                    _ => 20,
                };
                if command && matches!(key, "ArrowUp" | "ArrowDown") {
                    self.move_to(if step < 0 { 0 } else { self.len() }, m.shift);
                } else {
                    self.vertical(step, m.shift);
                }
                FieldKey::Moved
            }
            "Home" | "End" => {
                let to = if command {
                    if key == "Home" { 0 } else { self.len() }
                } else {
                    let (line, _) = self.line_col(self.caret);
                    self.index_of(line, if key == "Home" { 0 } else { usize::MAX })
                };
                self.move_to(to, m.shift);
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
                    } else if key == "Delete" && z < self.len() {
                        self.caret = z + 1;
                    } else {
                        return FieldKey::Moved;
                    }
                }
                self.insert("");
                FieldKey::Edited
            }
            k if !command && k.chars().count() == 1 => {
                self.insert(k);
                FieldKey::Edited
            }
            _ => FieldKey::Ignored,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edits_lines_as_an_editor_does() {
        let none = KeyModifiers::default();
        let mut t = TextArea::new("{\n  \"a\": 1,\n  \"bb\": 2\n}");
        assert_eq!(t.line_col(t.index_of(2, 9)), (2, 9));
        t.move_to(t.index_of(1, 9), false);
        t.key("ArrowDown", none);
        assert_eq!(t.line_col(t.caret), (2, 9));
        t.key("ArrowDown", none);
        assert_eq!(t.line_col(t.caret), (3, 1));
        t.key("ArrowUp", none);
        assert_eq!(t.line_col(t.caret), (2, 9), "the goal column is kept");
        t.key("End", none);
        t.key("Enter", none);
        assert_eq!(t.lines()[3], "  ");
        t.key("x", none);
        assert_eq!(t.lines()[3], "  x");
        t.key("Home", KeyModifiers { shift: true, ..none });
        assert_eq!(t.selection_in_line(3), Some((0, 3, false)));
        t.key("Backspace", none);
        assert_eq!(t.lines()[3], "");
        t.key("a", KeyModifiers { meta: true, ..none });
        assert_eq!(t.selection_in_line(1), Some((0, 9, true)));
        assert_eq!(t.key("Escape", none), FieldKey::Cancel);
    }
}
