//! Styled runs within one text node: a keyword in another colour, a link underlined, a search hit
//! highlighted, a word in bold, all in one line of text that wraps as one.
//!
//! A run covers a byte range of the node's text and changes what it sets.
//! What it leaves `None` stays as the node's style says. Layout measures each
//! run in its own style, so a bold word takes a bold word's width, and
//! wrapping breaks the text by those widths.
//!
//! ```
//! use threers_ui::rich_text::TextRun;
//! use threers_ui::{FixedWidthText, FontWeight, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let line = ui.add_text(root, "let x = 1;");
//! ui.set_text_runs(line, vec![
//!     TextRun::new(3).color([0.8, 0.4, 0.9, 1.0]).weight(FontWeight::Bold),
//!     TextRun::new(7),
//! ]);
//! ui.layout(300.0, 100.0, &mut FixedWidthText(7.0));
//! let painted: Vec<_> = ui.scene().items.into_iter().filter_map(|i| match i {
//!     threers_ui::scene::Item::Text { text, .. } => Some(text),
//!     _ => None,
//! }).collect();
//! assert_eq!(painted, ["let", " x = 1;"]);
//! ```

use super::style::{FontWeight, Rgba, Style};
use super::tree::{NodeId, NodeKind, Ui};

/// A styled stretch of a text node. See the [module docs](self).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextRun {
    /// Bytes of text it covers, from where the run before it ended.
    pub len: usize,
    pub color: Option<Rgba>,
    pub weight: Option<FontWeight>,
    pub italic: Option<bool>,
    pub underline: Option<bool>,
    pub wavy: Option<bool>,
    pub decoration_color: Option<Rgba>,
    pub line_through: Option<bool>,
    /// A highlight behind the run's text, the height of its line.
    pub background: Option<Rgba>,
    /// Where the run links to: a click or tap on it is the app's to follow
    /// (see [`Ui::link_at`]).
    pub link: Option<String>,
}

impl TextRun {
    /// `len` bytes in the node's own style.
    pub fn new(len: usize) -> Self {
        TextRun {
            len,
            ..Default::default()
        }
    }
    pub fn color(mut self, c: Rgba) -> Self {
        self.color = Some(c);
        self
    }
    pub fn weight(mut self, w: FontWeight) -> Self {
        self.weight = Some(w);
        self
    }
    pub fn italic(mut self, on: bool) -> Self {
        self.italic = Some(on);
        self
    }
    /// Underlined; `wavy` for a wave (a misspelling, a diagnostic).
    pub fn underline(mut self, wavy: bool) -> Self {
        self.underline = Some(true);
        self.wavy = Some(wavy);
        self
    }
    pub fn decoration_color(mut self, c: Rgba) -> Self {
        self.decoration_color = Some(c);
        self
    }
    pub fn line_through(mut self, on: bool) -> Self {
        self.line_through = Some(on);
        self
    }
    pub fn background(mut self, c: Rgba) -> Self {
        self.background = Some(c);
        self
    }
    /// Makes the run a link to `target`.
    pub fn link(mut self, target: impl Into<String>) -> Self {
        self.link = Some(target.into());
        self
    }

    /// `base` with what this run sets.
    pub fn style(&self, base: &Style) -> Style {
        let mut s = base.clone();
        if let Some(c) = self.color {
            s.color = c;
        }
        if let Some(w) = self.weight {
            s.font_weight = w;
        }
        if let Some(i) = self.italic {
            s.italic = i;
        }
        if let Some(u) = self.underline {
            s.underline = u;
        }
        if let Some(w) = self.wavy {
            s.decoration_wavy = w;
        }
        if self.decoration_color.is_some() {
            s.decoration_color = self.decoration_color;
        }
        if let Some(l) = self.line_through {
            s.line_through = l;
        }
        s
    }
}

/// A run's piece of one drawn line, placed by layout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Piece {
    pub line: u32,
    /// From the line's start, and how wide.
    pub x: f32,
    pub width: f32,
    /// Bytes of the node's text.
    pub start: u32,
    pub end: u32,
    /// Which run; `u32::MAX` past the last one (the node's own style).
    pub run: u32,
}

impl Ui {
    /// Styles stretches of a text node's text. The runs' lengths should add
    /// up to the text's; text past the last run keeps the node's style, and
    /// runs past the text are ignored. A run boundary inside a character is
    /// moved to the character's end.
    #[track_caller]
    pub fn set_text_runs(&mut self, id: NodeId, runs: Vec<TextRun>) {
        assert!(matches!(self.kind(id), NodeKind::Text(_)), "runs are for text nodes");
        if self.node(id).runs.as_deref() != Some(runs.as_slice()) {
            self.touch(id);
            self.nodes[id.0].runs = Some(runs);
        }
    }

    /// Back to one style for the whole node.
    pub fn clear_text_runs(&mut self, id: NodeId) {
        if self.node(id).runs.is_some() {
            self.touch(id);
            self.nodes[id.0].runs = None;
        }
    }

    pub fn text_runs(&self, id: NodeId) -> Option<&[TextRun]> {
        self.node(id).runs.as_deref()
    }

    /// The boxes of text node `id`'s links as laid out (a link wrapping
    /// over two lines has two), each with where it links to.
    pub fn link_rects(&self, id: NodeId, measure: &mut dyn crate::TextMeasure) -> Vec<(crate::Rect, String)> {
        let (Some(runs), NodeKind::Text(text)) = (self.text_runs(id), self.kind(id)) else {
            return vec![];
        };
        let mut out = vec![];
        let mut at = 0;
        for run in runs {
            let end = (at + run.len).min(text.len());
            if let Some(target) = &run.link {
                let chars = |b: usize| text.get(..b).map_or(0, |t| t.chars().count());
                for r in self.text_selection_rects(id, chars(at), chars(end), measure) {
                    out.push((r, target.clone()));
                }
            }
            at = end;
        }
        out
    }

    /// The link under `(x, y)` in any of `nodes`' text, by
    /// [`link_rects`](Self::link_rects).
    pub fn link_at(&self, nodes: &[NodeId], x: f32, y: f32, measure: &mut dyn crate::TextMeasure) -> Option<String> {
        nodes
            .iter()
            .flat_map(|&n| self.link_rects(n, measure))
            .find(|(r, _)| r.contains(x, y))
            .map(|(_, target)| target)
    }
}

/// `start..end` of `text` cut where `runs` change, each piece with its run
/// (`u32::MAX` past the last).
pub(crate) fn split(text: &str, runs: &[TextRun], start: usize, end: usize) -> Vec<(usize, usize, u32)> {
    let mut out = vec![];
    let mut at = 0usize;
    let fix = |i: usize| {
        let mut i = i.min(text.len());
        while !text.is_char_boundary(i) {
            i += 1;
        }
        i
    };
    for (k, run) in runs.iter().enumerate() {
        let (a, b) = (fix(at), fix(at + run.len));
        at += run.len;
        let (lo, hi) = (a.max(start), b.min(end));
        if lo < hi {
            out.push((lo, hi, k as u32));
        }
        if b >= end {
            return out;
        }
    }
    let lo = fix(at).max(start);
    if lo < end {
        out.push((lo, end, u32::MAX));
    }
    out
}
