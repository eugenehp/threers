//! Text and the pointer: which character is under a point, where a caret
//! goes, and the boxes a selection covers. With these, a click
//! places the caret, a drag selects, two clicks select a word and three a
//! line, across wrapped lines and styled runs.
//!
//! Positions are measured with the app's [`TextMeasure`], the one layout
//! used (a [`TextCache`](super::text_cache::TextCache) in front of it makes
//! repeated questions cheap). Indices are in characters.
//!
//! ```
//! use threers_ui::{FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let t = ui.add_text(root, "hello world");
//! ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
//! let m = &mut FixedWidthText(7.0);
//! assert_eq!(ui.text_index_at(t, 38.0, 5.0, m), 5); // nearer the space's left edge
//! assert_eq!(ui.text_caret(t, 6, m).x, 42.0);
//! assert_eq!(ui.text_selection_rects(t, 6, 11, m)[0].width, 35.0);
//! ```

use super::layout::TextMeasure;
use super::rich_text::split;
use super::style::Style;
use super::tree::{NodeId, NodeKind, Rect, Ui};

/// A drawn line: its box, and the bytes of the text it shows.
struct Line {
    rect: Rect,
    start: usize,
    end: usize,
}

impl Ui {
    /// The node's drawn lines, as painting places them.
    fn text_lines(&self, id: NodeId) -> (String, Vec<Line>) {
        let n = self.node(id);
        let NodeKind::Text(text) = &n.kind else {
            return (String::new(), vec![]);
        };
        let s = &n.style;
        let r = n.rect;
        let content = Rect {
            x: r.x + s.border.left + s.padding.left,
            y: r.y + s.border.top + s.padding.top,
            width: (r.width - s.border.horizontal() - s.padding.horizontal()).max(0.0),
            height: (r.height - s.border.vertical() - s.padding.vertical()).max(0.0),
        };
        let wrapped: Vec<&String> = if n.lines.len() > 1 && n.lines.concat() == *text {
            n.lines.iter().collect()
        } else {
            vec![]
        };
        if wrapped.is_empty() {
            return (text.clone(), vec![Line { rect: content, start: 0, end: text.len() }]);
        }
        let h = content.height / wrapped.len() as f32;
        let mut at = 0;
        let lines = wrapped
            .iter()
            .enumerate()
            .map(|(i, l)| {
                // A line's break (its trailing space) belongs to it, but is
                // not where a caret at its end goes.
                let shown = l.trim_end_matches([' ', '\u{200b}']).len();
                let line = Line {
                    rect: Rect {
                        y: content.y + i as f32 * h,
                        height: h,
                        ..content
                    },
                    start: at,
                    end: at + shown,
                };
                at += l.len();
                line
            })
            .collect();
        (text.clone(), lines)
    }

    /// How wide `text[start..end]` draws, each run in its own style.
    fn span_width(&self, id: NodeId, text: &str, start: usize, end: usize, measure: &mut dyn TextMeasure) -> f32 {
        if end <= start {
            return 0.0;
        }
        let n = self.node(id);
        let style: &Style = &n.style;
        match &n.runs {
            None => measure.measure(&text[start..end], style).0,
            Some(runs) => split(text, runs, start, end)
                .into_iter()
                .map(|(a, b, k)| {
                    let st = runs.get(k as usize).map_or_else(|| style.clone(), |r| r.style(style));
                    measure.measure(&text[a..b], &st).0
                })
                .sum(),
        }
    }

    /// The character index nearest `(x, y)` in text node `id`: on the line
    /// under `y` (the first or last above or below the text), at the
    /// character boundary closest to `x`.
    /// Turns selecting text on or off for the whole UI (on by default);
    /// `user-select: none` turns it off for a part of it.
    pub fn set_text_selection(&mut self, on: bool) {
        self.text_selection = on;
    }

    pub fn text_selection(&self) -> bool {
        self.text_selection
    }

    /// The text node under `(x, y)`, if its text can be selected (see
    /// [`UserSelect`](crate::UserSelect)): the nearest of it and its
    /// ancestors that says `text` or `none` decides; a control on the way
    /// says `none`; nothing saying, it can.
    pub fn selectable_text_at(&self, x: f32, y: f32) -> Option<NodeId> {
        if !self.text_selection {
            return None;
        }
        let hit = self.hit_test(x, y)?;
        if !matches!(self.kind(hit), crate::NodeKind::Text(_)) {
            return None;
        }
        use crate::{Role, UserSelect};
        for n in self.ancestors(hit) {
            match self.style(n).user_select {
                UserSelect::Text => return Some(hit),
                UserSelect::None => return None,
                UserSelect::Auto => {}
            }
            let control = self.semantics(n).is_some_and(|s| {
                matches!(
                    s.role,
                    Role::Button | Role::Tab | Role::MenuItem | Role::Checkbox | Role::Switch | Role::Radio | Role::Slider | Role::Option
                )
            });
            if control {
                return None;
            }
        }
        Some(hit)
    }

    pub fn text_index_at(&self, id: NodeId, x: f32, y: f32, measure: &mut dyn TextMeasure) -> usize {
        let (text, lines) = self.text_lines(id);
        let Some(line) = lines
            .iter()
            .find(|l| y < l.rect.y + l.rect.height)
            .or(lines.last())
        else {
            return 0;
        };
        let target = x - line.rect.x;
        let (mut best, mut best_d) = (line.start, f32::INFINITY);
        let bounds = text[line.start..line.end]
            .char_indices()
            .map(|(i, _)| line.start + i)
            .chain(std::iter::once(line.end));
        for b in bounds {
            let w = self.span_width(id, &text, line.start, b, measure);
            let d = (w - target).abs();
            if d < best_d {
                (best, best_d) = (b, d);
            }
            if w > target {
                break;
            }
        }
        text[..best].chars().count()
    }

    /// The caret before character `index`: a box a pixel wide and a line
    /// tall.
    pub fn text_caret(&self, id: NodeId, index: usize, measure: &mut dyn TextMeasure) -> Rect {
        let (text, lines) = self.text_lines(id);
        let byte = text.char_indices().nth(index).map_or(text.len(), |(b, _)| b);
        let Some(line) = lines
            .iter()
            .find(|l| byte <= l.end)
            .or(lines.last())
        else {
            return Rect::default();
        };
        let x = line.rect.x + self.span_width(id, &text, line.start, byte.clamp(line.start, line.end), measure);
        Rect {
            x,
            width: 1.0,
            ..line.rect
        }
    }

    /// The boxes characters `a..b` cover, one per line they are on: what to
    /// fill behind a selection.
    pub fn text_selection_rects(&self, id: NodeId, a: usize, b: usize, measure: &mut dyn TextMeasure) -> Vec<Rect> {
        let (a, b) = (a.min(b), a.max(b));
        let (text, lines) = self.text_lines(id);
        let byte = |i: usize| text.char_indices().nth(i).map_or(text.len(), |(b, _)| b);
        let (ba, bb) = (byte(a), byte(b));
        let mut out = vec![];
        for (k, line) in lines.iter().enumerate() {
            let next = lines.get(k + 1).map_or(text.len(), |l| l.start);
            let (lo, hi) = (ba.max(line.start), bb.min(next));
            if lo >= hi {
                continue;
            }
            let (lo, hi) = (lo.min(line.end), hi.min(line.end));
            let x0 = self.span_width(id, &text, line.start, lo, measure);
            let mut x1 = self.span_width(id, &text, line.start, hi, measure);
            // A selection running on past the line's end shows a space's
            // worth, as editors mark the newline.
            if bb > line.end && k + 1 < lines.len() {
                x1 += measure.measure(" ", &self.node(id).style).0;
            }
            out.push(Rect {
                x: line.rect.x + x0,
                width: (x1 - x0).max(0.0),
                ..line.rect
            });
        }
        out
    }
}

/// What a press selects by: two presses a word, three a line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Granularity {
    #[default]
    Character,
    Word,
    Line,
}

impl Granularity {
    /// For the `count`th press in a row.
    pub fn of_clicks(count: u32) -> Self {
        match count {
            0 | 1 => Granularity::Character,
            2 => Granularity::Word,
            _ => Granularity::Line,
        }
    }
}

/// The word (a run of letters, digits and `_`), run of spaces, or single
/// other character around character `i` of `chars`, as a character range.
pub fn word_at(chars: &[char], i: usize) -> (usize, usize) {
    if chars.is_empty() {
        return (0, 0);
    }
    let i = i.min(chars.len() - 1);
    let class = |c: char| {
        if c.is_alphanumeric() || c == '_' {
            0
        } else if c.is_whitespace() {
            1
        } else {
            2
        }
    };
    let k = class(chars[i]);
    if k == 2 {
        return (i, i + 1);
    }
    let mut a = i;
    while a > 0 && class(chars[a - 1]) == k {
        a -= 1;
    }
    let mut b = i + 1;
    while b < chars.len() && class(chars[b]) == k {
        b += 1;
    }
    (a, b)
}

/// The line (between newlines) around character `i`, as a character
/// range, without its newline.
pub fn line_at(chars: &[char], i: usize) -> (usize, usize) {
    let i = i.min(chars.len());
    let a = chars[..i].iter().rposition(|&c| c == '\n').map_or(0, |p| p + 1);
    let b = chars[i..].iter().position(|&c| c == '\n').map_or(chars.len(), |p| i + p);
    (a, b)
}

/// Selecting with the pointer, for a text model with `text`, `anchor`,
/// `caret` and a `pointer` field.
macro_rules! pointer_selection {
    ($t:ty) => {
        impl $t {
            /// A press at character `index`, the `clicks`th in a row:
            /// places the caret, or selects the word or line there; with
            /// `extend` (Shift held), moves the selection's end there.
            pub fn pointer_down(&mut self, index: usize, clicks: u32, extend: bool) {
                let chars: Vec<char> = self.text.chars().collect();
                let index = index.min(chars.len());
                let unit = $crate::text_hit::Granularity::of_clicks(clicks);
                let (a, b) = match unit {
                    $crate::text_hit::Granularity::Character => (index, index),
                    $crate::text_hit::Granularity::Word => $crate::text_hit::word_at(&chars, index),
                    $crate::text_hit::Granularity::Line => $crate::text_hit::line_at(&chars, index),
                };
                if extend && unit == $crate::text_hit::Granularity::Character {
                    self.caret = index;
                } else {
                    self.anchor = a;
                    self.caret = b;
                }
                self.pointer = Some((self.anchor, self.caret, unit));
            }

            /// The pointer dragged to character `index` since
            /// [`pointer_down`](Self::pointer_down): the selection runs from
            /// where it was pressed, by the unit it was pressed with.
            pub fn pointer_drag(&mut self, index: usize) {
                let Some((a, b, unit)) = self.pointer else { return };
                let chars: Vec<char> = self.text.chars().collect();
                let index = index.min(chars.len());
                let (lo, hi) = match unit {
                    $crate::text_hit::Granularity::Character => (index, index),
                    $crate::text_hit::Granularity::Word => $crate::text_hit::word_at(&chars, index),
                    $crate::text_hit::Granularity::Line => $crate::text_hit::line_at(&chars, index),
                };
                if index < a {
                    self.anchor = b;
                    self.caret = lo;
                } else {
                    self.anchor = a;
                    self.caret = hi.max(b);
                }
            }

            /// The pointer released: the selection stays.
            pub fn pointer_up(&mut self) {
                self.pointer = None;
            }
        }
    };
}

pub(crate) use pointer_selection;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FixedWidthText, TextField};

    #[test]
    fn words_lines_and_drags_by_unit() {
        let chars: Vec<char> = "let x_1 = a.b;\nnext".chars().collect();
        assert_eq!(word_at(&chars, 5), (4, 7));
        assert_eq!(word_at(&chars, 3), (3, 4), "a run of spaces");
        assert_eq!(word_at(&chars, 11), (11, 12), "punctuation alone");
        assert_eq!(line_at(&chars, 17), (15, 19));

        let mut f = TextField::new("alpha beta gamma");
        f.pointer_down(7, 2, false);
        assert_eq!(f.selection(), (6, 10));
        f.pointer_drag(13); // into "gamma": by words
        assert_eq!(f.selection(), (6, 16));
        f.pointer_drag(1); // back before the start: the word pressed stays
        assert_eq!((f.anchor, f.caret), (10, 0));
        f.pointer_up();
        f.pointer_down(3, 1, true);
        assert_eq!(f.selection(), (3, 10), "shift extends");
    }

    #[test]
    fn hits_wrapped_lines() {
        let mut ui = Ui::new();
        let root = ui.root();
        let para = ui.add_box(root, crate::style! { width: 50px; white-space: normal; });
        let t = ui.add_text(para, "abc def ghi");
        ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
        let m = &mut FixedWidthText(7.0);
        let lh = ui.text_caret(t, 0, m).height;
        // Lines "abc def " and "ghi": the second line's start is index 8.
        assert_eq!(ui.text_index_at(t, 0.0, lh * 1.5, m), 8);
        assert_eq!(ui.text_index_at(t, 1000.0, 1.0, m), 7, "past a line's end: before its break");
        let c = ui.text_caret(t, 9, m);
        assert_eq!((c.x, c.y), (7.0, lh));
        let rects = ui.text_selection_rects(t, 4, 10, m);
        assert_eq!(rects.len(), 2);
        assert_eq!((rects[0].x, rects[1].x, rects[1].width), (28.0, 0.0, 14.0));
    }
}
