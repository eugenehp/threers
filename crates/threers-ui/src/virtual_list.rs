//! Lists that build only the rows in view.
//!
//! A list of 100,000 rows as 100,000 nodes is laid out, painted and
//! hit-tested whole every frame. A [`VirtualList`] keeps a scrolling box
//! with just the rows that show, plus a few either side, between two
//! spacers standing in for the rest. The box's content height, and so its
//! scroll extent, is the whole list's.
//!
//! Rows are either one known height ([`RowHeight::Uniform`]), when the range
//! in view is a division, or measured as they come into view
//! ([`RowHeight::Measured`]), from an estimate until then. Measured heights
//! are kept in a Fenwick tree, so finding a row's offset is O(log n) however
//! long the list. When rows above the view turn out taller or shorter than
//! estimated, the scroll position is moved to match, so the row being
//! looked at stays put.
//!
//! ```
//! use threers_ui::virtual_list::{RowHeight, VirtualList};
//! use threers_ui::{style, FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let mut list = VirtualList::new(&mut ui, root, style! { height: 200px; }, 100_000, RowHeight::Uniform(20.0));
//! let mut measure = FixedWidthText(7.0);
//! // Lay out, build what shows, and lay out again if that changed anything.
//! ui.layout(300.0, 200.0, &mut measure);
//! while list.sync(&mut ui, |ui, row, i| { ui.add_text(row, format!("row {i}")); }) {
//!     ui.layout(300.0, 200.0, &mut measure);
//! }
//! assert!(list.built().len() < 20);
//! assert_eq!(ui.scroll_extent(list.container()).0, 2_000_000.0);
//! ```

use std::ops::Range;

use super::style::{Display, Length, Style};
use super::scrolling::ScrollStrategy;
use super::tree::{NodeId, Ui};

/// How tall a list's rows are.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RowHeight {
    /// Every row this tall.
    Uniform(f32),
    /// Each row as tall as its content, this tall until it has been seen.
    Measured { estimate: f32 },
}

/// Prefix sums over row heights.
#[derive(Debug, Clone, Default)]
struct Fenwick(Vec<f32>);

impl Fenwick {
    fn new(n: usize, value: f32) -> Self {
        let mut f = Fenwick(vec![0.0; n + 1]);
        for i in 0..n {
            f.add(i, value);
        }
        f
    }
    fn add(&mut self, i: usize, delta: f32) {
        let mut i = i + 1;
        while i < self.0.len() {
            self.0[i] += delta;
            i += i.isolate_lowest_one();
        }
    }
    /// The sum of the first `n` values.
    fn prefix(&self, n: usize) -> f32 {
        let mut i = n.min(self.0.len() - 1);
        let mut s = 0.0;
        while i > 0 {
            s += self.0[i];
            i -= i.isolate_lowest_one();
        }
        s
    }
    /// The first index whose end lies below `offset`: the row containing it.
    fn find(&self, offset: f32) -> usize {
        let n = self.0.len() - 1;
        let (mut at, mut left) = (0, offset);
        let mut step = n.next_power_of_two();
        while step > 0 {
            let next = at + step;
            if next <= n && self.0[next] <= left {
                at = next;
                left -= self.0[next];
            }
            step >>= 1;
        }
        at.min(n.saturating_sub(1))
    }
}

/// A scrolling list that builds only its visible rows. See the
/// [module docs](self).
#[derive(Debug, Clone)]
pub struct VirtualList {
    container: NodeId,
    top: NodeId,
    bottom: NodeId,
    count: usize,
    height: RowHeight,
    /// Rows built beyond each edge of the view, so a small scroll shows
    /// rows that already exist.
    pub overscan: usize,
    /// Measured heights (for [`RowHeight::Measured`]), `None` until seen.
    measured: Vec<Option<f32>>,
    sums: Fenwick,
    built: Vec<(usize, NodeId)>,
    /// A row asked to be shown, and whether at the top of the view: held
    /// until syncing settles, since measuring rows moves where it is.
    reveal: Option<(usize, ScrollStrategy)>,
}

impl VirtualList {
    /// A list of `count` rows in a new scrolling box under `parent`, styled
    /// `style` (give it a height, or a flex size, to scroll within).
    pub fn new(ui: &mut Ui, parent: NodeId, style: Style, count: usize, height: RowHeight) -> Self {
        let container = ui.add_box(
            parent,
            Style {
                scroll: true,
                clip: true,
                display: Display::Block,
                ..style
            },
        );
        let spacer = Style::default();
        let top = ui.add_box(container, spacer.clone());
        let bottom = ui.add_box(container, spacer);
        let mut list = VirtualList {
            container,
            top,
            bottom,
            count,
            height,
            overscan: 3,
            measured: vec![],
            sums: Fenwick::default(),
            built: vec![],
            reveal: None,
        };
        list.reset_heights();
        list.size_spacers(ui, 0..0);
        list
    }

    fn reset_heights(&mut self) {
        let h = self.estimate();
        self.measured = match self.height {
            RowHeight::Uniform(_) => vec![],
            RowHeight::Measured { .. } => vec![None; self.count],
        };
        self.sums = Fenwick::new(self.count, h);
    }

    fn estimate(&self) -> f32 {
        match self.height {
            RowHeight::Uniform(h) | RowHeight::Measured { estimate: h } => h.max(0.0),
        }
    }

    /// The scrolling box.
    pub fn container(&self) -> NodeId {
        self.container
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The rows that exist as nodes now, by index, with their boxes.
    pub fn built(&self) -> &[(usize, NodeId)] {
        &self.built
    }

    /// The box of row `index`, if it is built.
    pub fn row(&self, index: usize) -> Option<NodeId> {
        self.built.iter().find(|(i, _)| *i == index).map(|(_, n)| *n)
    }

    /// Where row `index` starts, from the top of the content.
    pub fn offset_of(&self, index: usize) -> f32 {
        match self.height {
            RowHeight::Uniform(h) => index.min(self.count) as f32 * h,
            RowHeight::Measured { .. } => self.sums.prefix(index),
        }
    }

    /// The whole list's height (as far as it is known).
    pub fn content_height(&self) -> f32 {
        self.offset_of(self.count)
    }

    /// The row at `offset` from the top of the content.
    pub fn index_at(&self, offset: f32) -> usize {
        if self.count == 0 {
            return 0;
        }
        match self.height {
            RowHeight::Uniform(h) if h > 0.0 => ((offset.max(0.0) / h) as usize).min(self.count - 1),
            RowHeight::Uniform(_) => 0,
            RowHeight::Measured { .. } => self.sums.find(offset.max(0.0)),
        }
    }

    /// The rows the view shows as last laid out, without overscan.
    pub fn visible(&self, ui: &Ui) -> Range<usize> {
        if self.count == 0 {
            return 0..0;
        }
        // Where layout put it: `scroll_top` clamped to the content.
        let (_, view, top) = ui.scroll_extent(self.container);
        let first = self.index_at(top);
        let last = self.index_at(top + view.max(1.0) - 1e-3);
        first..(last + 1).min(self.count)
    }

    /// Changes how many rows there are. Rows past the end are removed;
    /// measured heights of rows that remain are kept.
    /// Removes row `index`: the rows after it move up one, keeping their
    /// nodes and measured heights, so nothing is built again and the list
    /// does not jump. The next [`sync`](Self::sync) builds whatever comes
    /// into view at the end.
    pub fn remove(&mut self, ui: &mut Ui, index: usize) {
        if index >= self.count {
            return;
        }
        let mut measured = std::mem::take(&mut self.measured);
        if index < measured.len() {
            measured.remove(index);
        }
        self.count -= 1;
        self.reset_heights();
        for (i, h) in measured.into_iter().enumerate() {
            if let Some(h) = h {
                self.record(i, h);
            }
        }
        self.built.retain_mut(|(i, n)| {
            if *i == index {
                ui.remove(*n);
                return false;
            }
            if *i > index {
                *i -= 1;
            }
            true
        });
        let range = self.built_range();
        self.size_spacers(ui, range);
    }

    pub fn set_len(&mut self, ui: &mut Ui, count: usize) {
        if count == self.count {
            return;
        }
        let measured: Vec<Option<f32>> = self.measured.iter().copied().take(count).collect();
        self.count = count;
        self.reset_heights();
        for (i, h) in measured.into_iter().enumerate() {
            if let Some(h) = h {
                self.record(i, h);
            }
        }
        self.built.retain(|&(i, n)| {
            let keep = i < count;
            if !keep {
                ui.remove(n);
            }
            keep
        });
        let range = self.built_range();
        self.size_spacers(ui, range);
    }

    /// Throws away the built rows, so the next [`sync`](Self::sync) builds
    /// them again (after the data behind them changed).
    pub fn rebuild(&mut self, ui: &mut Ui) {
        for (_, n) in self.built.drain(..) {
            ui.remove(n);
        }
        self.size_spacers(ui, 0..0);
    }

    /// Scrolls so row `index` is in view: at the top if it is above, at
    /// the bottom if below, not at all if it already shows.
    pub fn scroll_to(&mut self, ui: &mut Ui, index: usize) {
        self.scroll_to_with(ui, index, ScrollStrategy::Nearest);
    }

    /// Scrolls row `index` to where `strategy` says: into view by the
    /// least, or to the top, the middle or the bottom of the list.
    pub fn scroll_to_with(&mut self, ui: &mut Ui, index: usize, strategy: ScrollStrategy) {
        let index = index.min(self.count.saturating_sub(1));
        let strategy = match strategy {
            ScrollStrategy::Nearest => {
                let (_, view, top) = ui.scroll_extent(self.container);
                let (start, end) = (self.offset_of(index), self.offset_of(index + 1));
                if start < top {
                    ScrollStrategy::Top
                } else if end > top + view {
                    ScrollStrategy::Bottom
                } else {
                    return;
                }
            }
            s => s,
        };
        self.reveal = Some((index, strategy));
        self.apply_reveal(ui);
    }

    /// Scrolls to the row asked for by [`scroll_to`](Self::scroll_to), by
    /// the heights known now.
    fn apply_reveal(&self, ui: &mut Ui) {
        let Some((index, strategy)) = self.reveal else { return };
        let (_, view, _) = ui.scroll_extent(self.container);
        let (start, end) = (self.offset_of(index), self.offset_of(index + 1));
        let to = match strategy {
            ScrollStrategy::Bottom => end - view,
            ScrollStrategy::Center => (start + end - view) / 2.0,
            _ => start,
        };
        let to = to.clamp(0.0, (self.content_height() - view).max(0.0));
        if ui.style(self.container).scroll_top != to {
            ui.style_mut(self.container).scroll_top = to;
        }
    }

    fn record(&mut self, index: usize, height: f32) {
        let old = self.measured[index].unwrap_or(self.estimate());
        if old != height {
            self.sums.add(index, height - old);
        }
        self.measured[index] = Some(height);
    }

    fn built_range(&self) -> Range<usize> {
        match (self.built.first(), self.built.last()) {
            (Some(a), Some(b)) => a.0..b.0 + 1,
            _ => 0..0,
        }
    }

    fn size_spacers(&self, ui: &mut Ui, range: Range<usize>) {
        let above = self.offset_of(range.start);
        let below = self.content_height() - self.offset_of(range.end);
        for (node, h) in [(self.top, above), (self.bottom, below.max(0.0))] {
            if ui.style(node).height != Length::Px(h) {
                ui.style_mut(node).height = Length::Px(h);
            }
        }
    }

    /// Brings the built rows in line with the view as last laid out:
    /// records the heights of measured rows, builds the rows that came into
    /// view with `build` (given the row's box and its index), and removes
    /// those that left. Returns whether anything changed, in which case lay
    /// out again and sync again, until it returns `false`.
    pub fn sync(&mut self, ui: &mut Ui, mut build: impl FnMut(&mut Ui, NodeId, usize)) -> bool {
        let mut changed = false;
        // Heights of rows seen, and the scroll position held on the row at
        // the top of the view while rows above it change size.
        if let RowHeight::Measured { .. } = self.height {
            let (_, _, top) = ui.scroll_extent(self.container);
            let anchor = self.index_at(top);
            let within = top - self.offset_of(anchor);
            for &(i, n) in &self.built.clone() {
                let h = ui.rect(n).height;
                if self.measured[i] != Some(h) {
                    self.record(i, h);
                    changed = true;
                }
            }
            if changed && self.reveal.is_none() {
                let to = self.offset_of(anchor) + within;
                if to != top {
                    ui.style_mut(self.container).scroll_top = to;
                }
            }
        }
        // A row asked for is placed by what is measured now; once nothing
        // moves, it has arrived.
        if self.reveal.is_some() {
            let before = ui.style(self.container).scroll_top;
            self.apply_reveal(ui);
            changed |= ui.style(self.container).scroll_top != before;
        }
        let visible = self.visible(ui);
        let want = visible.start.saturating_sub(self.overscan)..(visible.end + self.overscan).min(self.count);
        let before = self.built.len();
        self.built.retain(|&(i, n)| {
            let keep = want.contains(&i);
            if !keep {
                ui.remove(n);
            }
            keep
        });
        changed |= self.built.len() != before;
        let row_style = match self.height {
            RowHeight::Uniform(h) => Style {
                height: Length::Px(h),
                clip: true,
                ..Style::default()
            },
            RowHeight::Measured { .. } => Style::default(),
        };
        for i in want.clone() {
            if self.built.iter().any(|(b, _)| *b == i) {
                continue;
            }
            let row = ui.add_box(self.container, row_style.clone());
            build(ui, row, i);
            // Its place in the whole list, which only some rows of are
            // built: "5 of 100,000" to a screen reader. A row the app gave
            // no role is a list item.
            let sem = ui.semantics_mut(row);
            if sem.role == crate::Role::Generic {
                sem.role = crate::Role::ListItem;
            }
            // A row keeps its layout while it scrolls: scrolling marks the
            // list changed every frame, and without this every row in view
            // would be measured again (its text wrapped again) each time.
            ui.cache_layout(row, true);
            self.built.push((i, row));
            changed = true;
        }
        // Every built row's place, kept right as rows go and the list's
        // length changes.
        for &(i, row) in &self.built {
            let want = Some((i + 1, self.count));
            if ui.semantics(row).is_some_and(|s| s.position != want) {
                ui.semantics_mut(row).position = want;
            }
        }
        if changed {
            self.built.sort_by_key(|(i, _)| *i);
            // Spacer, rows in order, spacer.
            ui.move_child(self.container, self.top, 0);
            for (k, &(_, n)) in self.built.iter().enumerate() {
                ui.move_child(self.container, n, k + 1);
            }
            ui.move_child(self.container, self.bottom, self.built.len() + 1);
        }
        let range = self.built_range();
        self.size_spacers(ui, range);
        if !changed {
            self.reveal = None;
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fenwick_sums_and_finds() {
        let mut f = Fenwick::new(10, 2.0);
        assert_eq!(f.prefix(10), 20.0);
        f.add(3, 8.0); // row 3 is 10 tall: rows start 0 2 4 6 16 18 …
        assert_eq!(f.prefix(4), 16.0);
        assert_eq!(f.find(0.0), 0);
        assert_eq!(f.find(5.9), 2);
        assert_eq!(f.find(6.0), 3);
        assert_eq!(f.find(15.9), 3);
        assert_eq!(f.find(16.0), 4);
        assert_eq!(f.find(1e9), 9);
    }
}
