//! A range of the page's own text, from one text node to the next.
//!
//! [`PageText`] is the text it can run through — the text nodes of the
//! scrolling box it began in, or of the overlay (an absolutely placed box:
//! a dialog), else of the window, in reading order — as one string, with
//! points turned into places in it and places into carets and boxes.
//! [`PageRange`] is a selection in that, for a UI that is built anew as it
//! changes: node ids mean nothing in the next tree, so
//! [`refind`](PageRange::refind) finds it there by what its text says and
//! where it was. With [`RangeSelect`](crate::range_drag::RangeSelect) it is
//! pulled by its ends and scrolls at an edge ([`PageRange::content`]).
//! [`TextEditing`](crate::text_editing) selects in the same `PageText`.
//!
//! ```
//! use threers_ui::page_range::PageRange;
//! use threers_ui::{style, FixedWidthText, Ui};
//!
//! let build = || {
//!     let mut ui = Ui::new();
//!     let root = ui.root();
//!     for line in ["one two", "three four"] {
//!         let b = ui.add_box(root, style! { height: 20px; });
//!         ui.add_text(b, line);
//!     }
//!     ui.layout(300.0, 100.0, &mut FixedWidthText(10.0));
//!     ui
//! };
//! let m = &mut FixedWidthText(10.0);
//! let any = |_: &Ui, _| false;
//! let mut ui = build();
//! let mut range = PageRange::begin(&ui, 41.0, 10.0, &any, m).unwrap();
//! range.select_to(&ui, 51.0, 30.0, m);
//! assert_eq!(range.text(), "two\nthree");
//! // the UI built again: the same text selected in it
//! ui = build();
//! assert!(range.refind(&ui, &any));
//! assert_eq!(range.text(), "two\nthree");
//! assert_eq!(range.rects(&ui, m).len(), 2);
//! ```

use crate::range_drag::{RangeContent, RangeEnd};
use crate::text_hit::{line_at, word_at, Granularity};
use crate::tree::{NodeId, NodeKind, Rect, Ui};
use crate::{Display, Position, TextMeasure};

/// A text node a range can run through: where its text is in the whole
/// (characters from, and how many).
#[derive(Debug, Clone, PartialEq)]
struct Piece {
    node: NodeId,
    text: String,
    from: usize,
    len: usize,
}

/// Parts of the UI a range leaves out (with all in them): controls, say.
/// Text that can't be selected ([`Ui::text_selectable`]) is left out anyway.
pub type Skip<'a> = &'a dyn Fn(&Ui, NodeId) -> bool;

/// Text nodes as one text. See the [module docs](self).
#[derive(Debug, Clone, PartialEq)]
pub struct PageText {
    scope: NodeId,
    /// Where the scope is: each ancestor's place among its siblings.
    path: Vec<usize>,
    pieces: Vec<Piece>,
    text: String,
}

/// The part of the UI whose text goes with `node`'s: the scrolling box or
/// the overlay it is in, else the window.
pub fn scope_of(ui: &Ui, node: NodeId) -> NodeId {
    ui.ancestors(node).find(|&n| own_scope(ui, n)).unwrap_or(ui.root())
}

fn own_scope(ui: &Ui, n: NodeId) -> bool {
    let s = ui.style(n);
    s.scroll || s.position == Position::Absolute
}

fn path_of(ui: &Ui, node: NodeId) -> Vec<usize> {
    let up: Vec<NodeId> = ui.ancestors(node).collect();
    up.windows(2)
        .map(|w| ui.children(w[1]).iter().position(|&c| c == w[0]).unwrap_or(0))
        .collect()
}

/// The box that clips `node` (itself too): the intersection of what clips
/// it, or the window.
fn clip_of(ui: &Ui, node: NodeId) -> Rect {
    let mut r = ui.rect(ui.root());
    for a in ui.ancestors(node).filter(|&a| ui.style(a).clip || ui.style(a).scroll) {
        r = meet(r, ui.rect(a)).unwrap_or(Rect { width: 0.0, height: 0.0, ..r });
    }
    r
}

fn meet(a: Rect, b: Rect) -> Option<Rect> {
    let (x0, y0) = (a.x.max(b.x), a.y.max(b.y));
    let (x1, y1) = ((a.x + a.width).min(b.x + b.width), (a.y + a.height).min(b.y + b.height));
    (x1 > x0 && y1 > y0).then_some(Rect { x: x0, y: y0, width: x1 - x0, height: y1 - y0 })
}

impl PageText {
    /// The text under `scope`: each node's on a line of its own, unless it
    /// goes on beside the one before (then a space apart).
    pub fn of(ui: &Ui, scope: NodeId, skip: Skip) -> PageText {
        let mut nodes = vec![];
        let mut stack = vec![scope];
        while let Some(n) = stack.pop() {
            let others = n != scope && own_scope(ui, n);
            if others || ui.style(n).display == Display::None || skip(ui, n) {
                continue;
            }
            if ui.text_selectable(n) && ui.rect(n).height > 0.0 {
                nodes.push(n);
            }
            stack.extend(ui.children(n).iter().rev());
        }
        let (mut pieces, mut text): (Vec<Piece>, String) = (vec![], String::new());
        let mut from = 0;
        for node in nodes {
            let NodeKind::Text(t) = ui.kind(node) else { continue };
            if let Some(p) = pieces.last() {
                let (a, b) = (ui.rect(p.node), ui.rect(node));
                let beside = b.y < a.y + a.height / 2.0 && b.y + b.height > a.y && b.x >= a.x;
                text.push(if beside { ' ' } else { '\n' });
                from += 1;
            }
            let len = t.chars().count();
            text.push_str(t);
            pieces.push(Piece { node, text: t.clone(), from, len });
            from += len;
        }
        PageText { scope, path: path_of(ui, scope), pieces, text }
    }

    /// The whole text.
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn scope(&self) -> NodeId {
        self.scope
    }

    /// Where `node`'s text is in the whole, if it is in it.
    pub fn span(&self, node: NodeId) -> Option<(usize, usize)> {
        self.pieces.iter().find(|p| p.node == node).map(|p| (p.from, p.from + p.len))
    }

    /// The span of the node the caret before character `i` is in.
    pub fn span_at(&self, i: usize) -> (usize, usize) {
        self.piece(i).map_or((0, 0), |p| (p.from, p.from + p.len))
    }

    fn piece(&self, i: usize) -> Option<&Piece> {
        let k = self.pieces.partition_point(|p| p.from <= i);
        self.pieces.get(k.saturating_sub(1))
    }

    /// The caret before character `i`.
    pub fn caret(&self, ui: &Ui, i: usize, measure: &mut dyn TextMeasure) -> Rect {
        self.piece(i).map_or(Rect::default(), |p| ui.text_caret(p.node, i.saturating_sub(p.from).min(p.len), measure))
    }

    /// The character nearest `(x, y)`: in the node whose lines are beside
    /// it, the nearest across, else the node nearest up or down.
    pub fn index(&self, ui: &Ui, x: f32, y: f32, measure: &mut dyn TextMeasure) -> usize {
        let away = |from: f32, size: f32, at: f32| (from - at).max(at - (from + size)).max(0.0);
        self.pieces
            .iter()
            .map(|p| (p, ui.rect(p.node)))
            .filter(|(_, r)| r.width > 0.0 && r.height > 0.0)
            .min_by(|(_, a), (_, b)| {
                let d = |r: &Rect| (away(r.y, r.height, y), away(r.x, r.width, x));
                let (a, b) = (d(a), d(b));
                a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1))
            })
            .map_or(0, |(p, _)| p.from + ui.text_index_at(p.node, x, y, measure).min(p.len))
    }

    /// The boxes characters `lo..hi` cover, as they show.
    pub fn rects(&self, ui: &Ui, lo: usize, hi: usize, measure: &mut dyn TextMeasure) -> Vec<Rect> {
        self.pieces
            .iter()
            .filter(|p| p.from < hi && p.from + p.len > lo)
            .flat_map(|p| {
                let clip = clip_of(ui, p.node);
                let (a, b) = (lo.saturating_sub(p.from), (hi - p.from).min(p.len));
                ui.text_selection_rects(p.node, a, b, measure)
                    .into_iter()
                    .filter_map(move |r| meet(r, clip))
            })
            .collect()
    }

    /// Where its scope shows.
    pub fn view(&self, ui: &Ui) -> Rect {
        clip_of(ui, self.scope)
    }

    /// Whether `ui` still has it as it is.
    pub fn holds(&self, ui: &Ui) -> bool {
        ui.contains(self.scope)
            && self
                .pieces
                .iter()
                .all(|p| ui.contains(p.node) && matches!(ui.kind(p.node), NodeKind::Text(t) if *t == p.text))
    }

    /// The place `i` in `new` (this text as it has become): in the node with
    /// the same text, the same one of its likes; else in the one in its
    /// place that begins with it up to there.
    pub(crate) fn place_in(&self, new: &PageText, i: usize) -> Option<usize> {
        let p = self.piece(i)?;
        let local = i.saturating_sub(p.from).min(p.len);
        let nth = self.pieces.iter().filter(|q| q.from < p.from && q.text == p.text).count();
        if let Some(q) = new.pieces.iter().filter(|q| q.text == p.text).nth(nth) {
            return Some(q.from + local);
        }
        // text that's added to: the one in its place that goes on from it
        let head: String = p.text.chars().take(local).collect();
        let k = self.pieces.iter().position(|q| q.from == p.from)?;
        new.pieces.get(k).filter(|q| local > 0 && q.text.starts_with(&head)).map(|q| q.from + local)
    }
}

/// A range selected in the page's text. See the [module docs](self).
#[derive(Debug, Clone, PartialEq)]
pub struct PageRange {
    page: PageText,
    /// The end that stays, and the one that moves, as characters.
    anchor: usize,
    caret: usize,
    /// What it began with, and by what it goes on.
    from: (usize, usize),
    unit: Granularity,
}

impl PageRange {
    /// Begun at `(x, y)`, empty, if there is text there to select.
    pub fn begin(ui: &Ui, x: f32, y: f32, skip: Skip, measure: &mut dyn TextMeasure) -> Option<PageRange> {
        let node = ui.selectable_text_at(x, y)?;
        if ui.ancestors(node).any(|n| skip(ui, n)) {
            return None;
        }
        let page = PageText::of(ui, scope_of(ui, node), skip);
        let (from, to) = page.span(node)?;
        let i = (from + ui.text_index_at(node, x, y, measure)).min(to);
        Some(PageRange { page, anchor: i, caret: i, from: (i, i), unit: Granularity::Character })
    }

    /// From here on by words or lines, beginning with the one it is at.
    pub fn by(&mut self, unit: Granularity) {
        self.unit = unit;
        self.from = self.unit_at(self.caret);
        (self.anchor, self.caret) = self.from;
    }

    fn unit_at(&self, i: usize) -> (usize, usize) {
        if self.unit == Granularity::Character {
            return (i, i);
        }
        let chars: Vec<char> = self.page.text.chars().collect();
        if chars.is_empty() {
            return (0, 0);
        }
        let i = i.min(chars.len() - 1);
        match self.unit {
            Granularity::Word => word_at(&chars, i),
            _ => line_at(&chars, i),
        }
    }

    pub fn page(&self) -> &PageText {
        &self.page
    }

    /// The selection, in the page's text.
    pub fn selection(&self) -> (usize, usize) {
        (self.anchor.min(self.caret), self.anchor.max(self.caret))
    }

    pub fn is_empty(&self) -> bool {
        self.anchor == self.caret
    }

    /// What is selected.
    pub fn text(&self) -> String {
        let (lo, hi) = self.selection();
        self.page.text.chars().skip(lo).take(hi - lo).collect()
    }

    /// `lo..hi` selected.
    pub fn select(&mut self, lo: usize, hi: usize) {
        (self.anchor, self.caret) = (lo, hi);
        (self.from, self.unit) = ((lo, lo), Granularity::Character);
    }

    /// Everything in its scope.
    pub fn select_all(&mut self) {
        self.select(0, self.page.text.chars().count());
    }

    /// All of the text node it begins in.
    pub fn select_piece(&mut self) {
        let (lo, hi) = self.page.span_at(self.selection().0);
        self.select(lo, hi);
    }

    /// Whether all of the text node it begins in is selected.
    pub fn covers_piece(&self) -> bool {
        let (lo, hi) = self.selection();
        let (a, b) = self.page.span_at(lo);
        lo <= a && hi >= b
    }

    /// The drag that began it, on to `(x, y)`: by its unit.
    pub fn select_to(&mut self, ui: &Ui, x: f32, y: f32, measure: &mut dyn TextMeasure) {
        let i = self.page.index(ui, x, y, measure);
        let (lo, hi) = self.unit_at(i);
        let (a, b) = self.from;
        (self.anchor, self.caret) = if i < a { (b, lo) } else { (a, hi.max(b)) };
    }

    /// The boxes it covers, as they show.
    pub fn rects(&self, ui: &Ui, measure: &mut dyn TextMeasure) -> Vec<Rect> {
        let (lo, hi) = self.selection();
        self.page.rects(ui, lo, hi, measure)
    }

    /// Its ends' carets, none for one scrolled out of view; `None` empty.
    pub fn ends(&self, ui: &Ui, measure: &mut dyn TextMeasure) -> Option<(Option<Rect>, Option<Rect>)> {
        if self.is_empty() {
            return None;
        }
        let (lo, hi) = self.selection();
        let view = self.page.view(ui);
        let mut caret = |i: usize| {
            let c = self.page.caret(ui, i, measure);
            let middle = c.y + c.height / 2.0;
            (middle >= view.y && middle <= view.y + view.height).then_some(c)
        };
        Some((caret(lo), caret(hi)))
    }

    /// Found in `ui`, which may have been built anew or changed: in the scope
    /// in the same place, else any that has its ends' text. `false`: it's
    /// not there any more (or says something else).
    pub fn refind(&mut self, ui: &Ui, skip: Skip) -> bool {
        let mut scopes = vec![];
        let mut stack = vec![ui.root()];
        while let Some(n) = stack.pop() {
            if n == ui.root() || own_scope(ui, n) {
                scopes.push(n);
            }
            stack.extend(ui.children(n).iter().rev());
        }
        // (the one in its place first)
        scopes.sort_by_key(|&n| path_of(ui, n) != self.page.path);
        let was = self.text();
        for scope in scopes {
            let new = PageText::of(ui, scope, skip);
            let places = [self.anchor, self.caret, self.from.0, self.from.1].map(|i| self.page.place_in(&new, i));
            let [Some(anchor), Some(caret), from, to] = places else { continue };
            let found = PageRange { from: (from.unwrap_or(anchor), to.unwrap_or(anchor)), anchor, caret, page: new, unit: self.unit };
            // the same selected: what's between may have been added to, no more
            if self.is_empty() || found.text() == was || (!found.is_empty() && found.text().len() > was.len()) {
                *self = found;
                return true;
            }
        }
        false
    }

    /// It, as [`RangeSelect`](crate::range_drag::RangeSelect)'s content.
    pub fn content<'a>(&'a mut self, ui: &'a mut Ui, measure: &'a mut dyn TextMeasure) -> PageContent<'a> {
        PageContent { range: self, ui, measure: std::cell::RefCell::new(measure) }
    }
}

/// A [`PageRange`] in its UI, for [`RangeSelect`](crate::range_drag::RangeSelect).
pub struct PageContent<'a> {
    range: &'a mut PageRange,
    ui: &'a mut Ui,
    measure: std::cell::RefCell<&'a mut dyn TextMeasure>,
}

impl RangeContent for PageContent<'_> {
    fn view(&self) -> Rect {
        self.range.page.view(self.ui)
    }

    /// (A page scrolls by pixels.)
    fn line_height(&self) -> f32 {
        1.0
    }

    fn ends(&self) -> Option<(Option<Rect>, Option<Rect>)> {
        self.range.ends(self.ui, *self.measure.borrow_mut())
    }

    fn hold(&mut self, end: RangeEnd) {
        let (lo, hi) = self.range.selection();
        (self.range.anchor, self.range.caret) = match end {
            RangeEnd::Start => (hi, lo),
            RangeEnd::Finish => (lo, hi),
        };
    }

    fn hold_nearer(&mut self, x: f32, y: f32) -> bool {
        if self.range.is_empty() {
            return false;
        }
        let i = self.range.page.index(self.ui, x, y, *self.measure.borrow_mut());
        let (lo, hi) = self.range.selection();
        self.hold(if i.abs_diff(hi) <= i.abs_diff(lo) { RangeEnd::Finish } else { RangeEnd::Start });
        true
    }

    fn move_held(&mut self, x: f32, y: f32) {
        let (x, y) = self.inside(x, y);
        let i = self.range.page.index(self.ui, x, y, *self.measure.borrow_mut());
        if i != self.range.anchor {
            self.range.caret = i;
        }
    }

    fn select_to(&mut self, x: f32, y: f32) {
        let (x, y) = self.inside(x, y);
        self.range.select_to(self.ui, x, y, *self.measure.borrow_mut());
    }

    fn let_go(&mut self) {}

    fn scroll(&mut self, pixels: i32) -> bool {
        let scope = self.range.page.scope;
        if !self.ui.contains(scope) || !self.ui.style(scope).scroll {
            return false;
        }
        let (content, view, _) = self.ui.scroll_extent(scope);
        let now = self.ui.style(scope).scroll_top;
        let to = (now + pixels as f32).clamp(0.0, (content - view).max(0.0));
        if to != now {
            self.ui.set_scroll_top(scope, to);
        }
        to != now
    }
}

impl PageContent<'_> {
    /// A point in the view: past its edge, only what shows is selected (what
    /// scrolls in is, as it does).
    fn inside(&self, x: f32, y: f32) -> (f32, f32) {
        let v = self.view();
        if !self.ui.style(self.range.page.scope).scroll || v.height < 2.0 {
            return (x, y);
        }
        (x, y.clamp(v.y + 1.0, v.y + v.height - 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_drag::RangeSelect;
    use crate::{style, FixedWidthText};

    /// A heading, a scrolling list of lines and a dialog over them.
    fn build(lines: &[&str], scrolled: f32, banner: bool) -> (Ui, NodeId) {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.update_style(root, |s| s.flex_direction = crate::FlexDirection::Column);
        if banner {
            let b = ui.add_box(root, style! { height: 20px; });
            ui.add_text(b, "A banner come since");
        }
        let h = ui.add_box(root, style! { height: 20px; });
        ui.add_text(h, "Heading");
        let list = ui.add_box(root, style! { display: flex; flex-direction: column; width: 300px; height: 100px; overflow-y: auto; });
        for l in lines {
            let b = ui.add_box(list, style! { height: 20px; flex-shrink: 0; });
            ui.add_text(b, *l);
        }
        ui.style_mut(list).scroll_top = scrolled;
        let over = ui.add_box(root, style! { position: absolute; left: 200px; top: 150px; width: 150px; height: 40px; });
        ui.add_text(over, "In a dialog");
        ui.layout(400.0, 300.0, &mut FixedWidthText(10.0));
        (ui, list)
    }

    const NONE: Skip = &|_, _| false;

    #[test]
    fn a_range_is_found_in_the_tree_built_anew() {
        let m = &mut FixedWidthText(10.0);
        let lines = ["same", "same", "other", "same"];
        let (ui, _) = build(&lines, 0.0, false);
        let mut r = PageRange::begin(&ui, 11.0, 50.0, NONE, m).unwrap();
        r.select_to(&ui, 31.0, 70.0, m);
        assert_eq!(r.text(), "ame\noth");
        // the second of the same, moved down by a banner, text added after
        let (ui, _) = build(&["same", "same", "other", "same", "more"], 0.0, true);
        assert!(r.refind(&ui, NONE));
        assert_eq!((r.text().as_str(), r.selection()), ("ame\noth", (6, 13)));
        assert_eq!(r.rects(&ui, m)[0].y, 60.0);
        // the last line growing as it's streamed: still there
        let (ui, _) = build(&["same", "same", "otherwise", "same"], 0.0, false);
        assert!(r.refind(&ui, NONE));
        assert_eq!(r.text(), "ame\noth");
        // its text gone: so is it
        let (ui, _) = build(&["new"], 0.0, false);
        assert!(!r.refind(&ui, NONE));
    }

    #[test]
    fn a_scrolling_box_an_overlay_and_the_window_are_each_their_own() {
        let m = &mut FixedWidthText(10.0);
        let (ui, _) = build(&["a line", "b line"], 0.0, false);
        let all = |x: f32, y: f32, m: &mut FixedWidthText| {
            let mut r = PageRange::begin(&ui, x, y, NONE, m).unwrap();
            r.select_all();
            r.text()
        };
        assert_eq!(all(11.0, 30.0, m), "a line\nb line");
        assert_eq!(all(11.0, 10.0, m), "Heading");
        assert_eq!(all(211.0, 155.0, m), "In a dialog");
        assert!(PageRange::begin(&ui, 11.0, 30.0, &|ui: &Ui, n| ui.style(n).scroll, m).is_none(), "left out");
    }

    #[test]
    fn words_pieces_and_an_end_held_at_the_edge() {
        let m = &mut FixedWidthText(10.0);
        let text: Vec<String> = (0..40).map(|i| format!("line {i:02}")).collect();
        let lines: Vec<&str> = text.iter().map(String::as_str).collect();
        let (mut ui, list) = build(&lines, 0.0, false);
        let mut r = PageRange::begin(&ui, 11.0, 50.0, NONE, m).unwrap();
        r.by(Granularity::Word);
        assert_eq!(r.text(), "line");
        assert!(!r.covers_piece());
        r.select_piece();
        assert!(r.covers_piece());
        assert_eq!(r.text(), "line 01");
        // its last end's handle, pulled to the bottom and held
        let mut s = RangeSelect::new();
        let end = r.ends(&ui, m).unwrap().1.unwrap();
        assert!(s.press(&mut r.content(&mut ui, m), end.x, end.y + end.height + 6.0));
        s.drag(&mut r.content(&mut ui, m), 60.0, 118.0);
        for k in 0..60 {
            s.tick(&mut r.content(&mut ui, m), k as f64 / 60.0);
            ui.layout(400.0, 300.0, &mut FixedWidthText(10.0));
        }
        assert!(ui.style(list).scroll_top > 100.0);
        assert!(r.text().lines().count() > 8, "{}", r.text());
        assert!(s.release(&mut r.content(&mut ui, m)));
        assert!(r.ends(&ui, m).unwrap().0.is_none(), "its first end is out of view");
    }
}
