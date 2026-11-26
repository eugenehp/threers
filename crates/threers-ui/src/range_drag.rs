//! Changing a selected range by its ends: what every selection here does,
//! and ready for content an app draws and scrolls itself (a terminal, a log,
//! a table, a canvas of text). It knows nothing of the content, which
//! answers through [`RangeContent`]: where its range's ends are drawn, and
//! moving one to a point.
//!
//! - [`RangeSelect`] is the whole behaviour: an end pulled by its handle
//!   (past the other too), a drag that selects, Shift's click, and scrolling
//!   on while any of them is held at an edge; [`draw_handles`] and
//!   [`draw_bar`] show it.
//! - Its parts, for other uses: [`RangeHandles`] (the knobs, which a finger
//!   takes, and the caret following it without a jump), [`EdgeScroll`]
//!   (faster the further past the edge) and [`place_bar`] (room for a bar
//!   clear of the range).
//!
//! [`TextEditing`](crate::text_editing)'s fields and page text, and
//! [`Input`](crate::input::Input)'s drags, use the same parts.
//!
//! ```
//! use threers_ui::Rect;
//! use threers_ui::range_drag::{RangeContent, RangeEnd, RangeSelect};
//!
//! /// A hundred lines of 20 px, whole lines selected, ten in view.
//! struct Lines { top: i32, range: (i32, i32), held: RangeEnd }
//!
//! impl RangeContent for Lines {
//!     fn view(&self) -> Rect { Rect { x: 0.0, y: 0.0, width: 100.0, height: 200.0 } }
//!     fn line_height(&self) -> f32 { 20.0 }
//!     fn ends(&self) -> Option<(Option<Rect>, Option<Rect>)> {
//!         let caret = |line: i32, x| {
//!             let y = (line - self.top) as f32 * 20.0;
//!             (0.0..200.0).contains(&y).then_some(Rect { x, y, width: 0.0, height: 20.0 })
//!         };
//!         Some((caret(self.range.0, 10.0), caret(self.range.1, 90.0)))
//!     }
//!     fn hold(&mut self, end: RangeEnd) { self.held = end }
//!     fn hold_nearer(&mut self, _x: f32, _y: f32) -> bool { false }
//!     fn move_held(&mut self, _x: f32, y: f32) {
//!         let line = self.top + (y.clamp(0.0, 199.0) / 20.0) as i32;
//!         match self.held {
//!             RangeEnd::Start => self.range.0 = line.min(self.range.1),
//!             RangeEnd::Finish => self.range.1 = line.max(self.range.0),
//!         }
//!     }
//!     fn select_to(&mut self, _x: f32, _y: f32) {}
//!     fn let_go(&mut self) {}
//!     fn scroll(&mut self, lines: i32) -> bool {
//!         let to = (self.top + lines).clamp(0, 90);
//!         std::mem::replace(&mut self.top, to) != to
//!     }
//! }
//!
//! let mut lines = Lines { top: 0, range: (2, 4), held: RangeEnd::Start };
//! let mut select = RangeSelect::new();
//! // a finger on the last end's handle, pulled to the bottom and held there
//! assert!(select.press(&mut lines, 90.0, 106.0));
//! select.drag(&mut lines, 90.0, 198.0);
//! for frame in 0..60 {
//!     select.tick(&mut lines, frame as f64 / 60.0);
//! }
//! assert!(lines.top > 5 && lines.range.1 > 15, "scrolled and selected on");
//! select.release(&mut lines);
//! ```

use crate::tree::{NodeId, Rect, Ui};
use crate::{style, Display, Length, Role, Semantics, Style};

/// An end of the range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeEnd {
    Start,
    Finish,
}

/// How near handles are taken, and their knobs' size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandleConfig {
    /// A knob's diameter.
    pub knob: f32,
    /// How near a knob's middle a finger takes it.
    pub reach: f32,
    /// How far beside an end's bar a finger takes it.
    pub bar_reach: f32,
}

impl Default for HandleConfig {
    fn default() -> Self {
        HandleConfig { knob: 12.0, reach: 24.0, bar_reach: 12.0 }
    }
}

/// The two handles of a range. See the [module docs](self).
#[derive(Debug, Clone, Default)]
pub struct RangeHandles {
    pub config: HandleConfig,
    /// The end held, and the caret's middle from the finger.
    held: Option<(RangeEnd, (f32, f32))>,
}

impl RangeHandles {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: HandleConfig) -> Self {
        RangeHandles { config, held: None }
    }

    /// An end's knob's middle: above the start's line, below the finish's,
    /// and inside `view` (an end on its first or last line).
    pub fn knob(&self, end: RangeEnd, caret: Rect, view: Rect) -> (f32, f32) {
        let r = self.config.knob / 2.0;
        let y = match end {
            RangeEnd::Start => caret.y - r,
            RangeEnd::Finish => caret.y + caret.height + r,
        };
        let low = (view.y + view.height - r).max(view.y + r);
        let right = (view.x + view.width - r).max(view.x + r);
        (caret.x.clamp(view.x + r, right), y.clamp(view.y + r, low))
    }

    /// The end a finger at (`x`, `y`) would take: near its knob, or by its
    /// bar on the knob's side of the line (a fingertip lands a little off
    /// what it aims at, and the knob is small). Across the middle of the
    /// line the touch is the content's: a tap on a short selected word is
    /// near both.
    pub fn end_at(&self, x: f32, y: f32, start: Option<Rect>, finish: Option<Rect>, view: Rect) -> Option<RangeEnd> {
        let HandleConfig { knob, reach, bar_reach } = self.config;
        let near = |end: RangeEnd, caret: Option<Rect>| {
            let c = caret?;
            let (kx, ky) = self.knob(end, c, view);
            let d = (x - kx).hypot(y - ky);
            let (knob_half, knob_side) = match end {
                RangeEnd::Start => (y < c.y + c.height / 2.0, y < c.y + c.height * 0.4),
                RangeEnd::Finish => (y > c.y + c.height / 2.0, y > c.y + c.height * 0.6),
            };
            let by_bar = (x - c.x).abs() <= bar_reach && knob_side && y >= c.y - knob - reach && y <= c.y + c.height + knob + reach;
            ((d <= reach && knob_half) || by_bar).then_some((d, end))
        };
        [near(RangeEnd::Start, start), near(RangeEnd::Finish, finish)]
            .into_iter()
            .flatten()
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, end)| end)
    }

    /// A finger down: the end it takes, if one.
    pub fn press(&mut self, x: f32, y: f32, start: Option<Rect>, finish: Option<Rect>, view: Rect) -> Option<RangeEnd> {
        let end = self.end_at(x, y, start, finish, view)?;
        let c = match end {
            RangeEnd::Start => start,
            RangeEnd::Finish => finish,
        }?;
        self.held = Some((end, (c.x - x, c.y + c.height / 2.0 - y)));
        Some(end)
    }

    /// The finger moved: where the caret's middle is now.
    pub fn drag(&self, x: f32, y: f32) -> Option<(f32, f32)> {
        self.held.map(|(_, (dx, dy))| (x + dx, y + dy))
    }

    /// The end held (as it was when taken: dragged past the other, it's the
    /// range's other end now).
    pub fn held(&self) -> Option<RangeEnd> {
        self.held.map(|h| h.0)
    }

    /// The finger lifted; whether an end was held.
    pub fn release(&mut self) -> bool {
        self.held.take().is_some()
    }
}

/// How a drag held at an edge scrolls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeConfig {
    /// How far inside the view its edge begins (0: past it only, a mouse's).
    pub inside: f32,
    /// How far from where it began a drag goes toward an edge before that
    /// edge scrolls: begun at an edge, it doesn't at once.
    pub slop: f32,
    /// Pixels a second as the edge is reached, and `full` pixels past that.
    pub slow: f32,
    pub fast: f32,
    pub full: f32,
}

impl EdgeConfig {
    /// A mouse's: only past the view's edge, and as soon as it is.
    pub fn past() -> Self {
        EdgeConfig { inside: 0.0, slop: f32::NEG_INFINITY, ..Self::default() }
    }
}

impl Default for EdgeConfig {
    fn default() -> Self {
        EdgeConfig { inside: 28.0, slop: 8.0, slow: 80.0, fast: 1400.0, full: 90.0 }
    }
}

/// Scrolling while a drag is held at an edge. See the [module docs](self).
#[derive(Debug, Clone, Default)]
pub struct EdgeScroll {
    pub config: EdgeConfig,
    /// Where the drag began (its height), and what of a line is left over.
    from: f32,
    carry: f32,
}

impl EdgeScroll {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: EdgeConfig) -> Self {
        EdgeScroll { config, ..Self::default() }
    }

    /// A drag begins at (`x`, `y`).
    pub fn begin(&mut self, _x: f32, y: f32) {
        (self.from, self.carry) = (y, 0.0);
    }

    /// How fast the content scrolls for a drag at height `y`: pixels a
    /// second, positive toward its end (0: not at an edge).
    pub fn speed(&self, view: Rect, y: f32) -> f32 {
        let c = &self.config;
        // (a view too short for two edges: past it only)
        let inside = if view.height > c.inside * 3.0 { c.inside } else { 0.0 };
        let over = |d: f32| c.slow + (c.fast - c.slow) * (d / c.full.max(1.0)).clamp(0.0, 1.0);
        let (top, bottom) = (view.y + inside - y, y - (view.y + view.height - inside));
        if top > 0.0 && y < self.from - c.slop {
            -over(top)
        } else if bottom > 0.0 && y > self.from + c.slop {
            over(bottom)
        } else {
            0.0
        }
    }

    /// `dt` seconds on, the whole lines (`line` high) to scroll, positive
    /// toward the content's end; parts of a line are kept for the next.
    pub fn lines(&mut self, view: Rect, y: f32, line: f32, dt: f32) -> i32 {
        let v = self.speed(view, y);
        if v == 0.0 || line <= 0.0 {
            self.carry = 0.0;
            return 0;
        }
        if v * self.carry < 0.0 {
            self.carry = 0.0;
        }
        self.carry += v * dt / line;
        let n = self.carry.trunc();
        self.carry -= n;
        n as i32
    }
}

/// Where a bar of `w` × `h` goes: above the range's start, else below its
/// finish, else (a range filling the view) across its middle; `gap` clear of
/// the line (the handles' room), inside `view`.
pub fn place_bar(start: Option<Rect>, finish: Option<Rect>, w: f32, h: f32, gap: f32, view: Rect) -> (f32, f32) {
    let middle = match (start, finish) {
        (Some(a), Some(b)) if (a.y - b.y).abs() < 1.0 => (a.x + b.x) / 2.0,
        (Some(c), _) | (None, Some(c)) => c.x,
        (None, None) => view.x + view.width / 2.0,
    };
    let x = (middle - w / 2.0).min(view.x + view.width - w - 8.0).max(view.x + 8.0);
    let above = start.map(|c| c.y - gap - h).filter(|y| *y >= view.y);
    let below = finish.map(|c| c.y + c.height + gap).filter(|y| y + h <= view.y + view.height);
    (x, above.or(below).unwrap_or(view.y + (view.height - h) / 2.0))
}

/// Content with a range selected in it, for [`RangeSelect`]. Points are the
/// UI's.
pub trait RangeContent {
    /// Where the content shows.
    fn view(&self) -> Rect;
    /// A line's height: what [`scroll`](Self::scroll) counts by.
    fn line_height(&self) -> f32;
    /// The range's ends, as drawn: a caret box each, none for one out of
    /// view. `None` with nothing selected.
    fn ends(&self) -> Option<(Option<Rect>, Option<Rect>)>;
    /// This end moves from here on; the other stays.
    fn hold(&mut self, end: RangeEnd);
    /// The end nearer (`x`, `y`) moves from here on, if anything is selected.
    fn hold_nearer(&mut self, x: f32, y: f32) -> bool;
    /// The held end to the place nearest (`x`, `y`) (past the other end, it
    /// is the range's other end; never to nothing).
    fn move_held(&mut self, x: f32, y: f32);
    /// A selection the app began, on to (`x`, `y`).
    fn select_to(&mut self, x: f32, y: f32);
    /// The end held, or the selection begun, let go.
    fn let_go(&mut self);
    /// Scrolled by lines, positive toward the content's end; whether it was.
    fn scroll(&mut self, lines: i32) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Drag {
    /// An end, by its handle.
    Handle,
    /// An end, by a Shift-click.
    End,
    /// A selection the app began.
    Selection,
}

/// Selecting a range and changing it. See the [module docs](self).
#[derive(Debug, Clone, Default)]
pub struct RangeSelect {
    pub handles: RangeHandles,
    /// How a finger's drag scrolls at an edge, and a mouse's.
    pub finger_edge: EdgeConfig,
    pub mouse_edge: Option<EdgeConfig>,
    edge: EdgeScroll,
    drag: Option<Drag>,
    at: Option<(f32, f32)>,
    time: Option<f64>,
}

impl RangeSelect {
    pub fn new() -> Self {
        Self::default()
    }

    /// The end a finger at (`x`, `y`) would take.
    pub fn end_at(&self, content: &dyn RangeContent, x: f32, y: f32) -> Option<RangeEnd> {
        let (start, finish) = content.ends()?;
        self.handles.end_at(x, y, start, finish, content.view())
    }

    /// A finger down: on a handle, it takes that end. Whether it did.
    pub fn press(&mut self, content: &mut dyn RangeContent, x: f32, y: f32) -> bool {
        let Some((start, finish)) = content.ends() else { return false };
        let Some(end) = self.handles.press(x, y, start, finish, content.view()) else { return false };
        content.hold(end);
        self.start(Drag::Handle, x, y, true);
        true
    }

    /// The app began a selection at (`x`, `y`) (a long press, a mouse's
    /// button), which [`drag`](Self::drag) goes on with.
    pub fn begin(&mut self, x: f32, y: f32, finger: bool) {
        self.start(Drag::Selection, x, y, finger);
    }

    /// A Shift-click: the range's nearer end to there, then on with the
    /// drag. Whether anything was selected.
    pub fn extend(&mut self, content: &mut dyn RangeContent, x: f32, y: f32) -> bool {
        if !content.hold_nearer(x, y) {
            return false;
        }
        content.move_held(x, y);
        self.start(Drag::End, x, y, false);
        true
    }

    fn start(&mut self, drag: Drag, x: f32, y: f32, finger: bool) {
        let config = if finger { self.finger_edge } else { self.mouse_edge.unwrap_or_else(EdgeConfig::past) };
        self.edge = EdgeScroll::with_config(config);
        self.edge.begin(x, y);
        (self.drag, self.at) = (Some(drag), None);
    }

    /// The finger or pointer moved to (`x`, `y`).
    pub fn drag(&mut self, content: &mut dyn RangeContent, x: f32, y: f32) {
        let Some(drag) = self.drag else { return };
        self.at = Some((x, y));
        match drag {
            Drag::Handle => {
                let (cx, cy) = self.handles.drag(x, y).unwrap_or((x, y));
                content.move_held(cx, cy);
            }
            Drag::End => content.move_held(x, y),
            Drag::Selection => content.select_to(x, y),
        }
    }

    /// Lifted. Whether anything was being dragged.
    pub fn release(&mut self, content: &mut dyn RangeContent) -> bool {
        self.handles.release();
        self.at = None;
        let was = self.drag.take().is_some();
        if was {
            content.let_go();
        }
        was
    }

    /// Whether a drag is on (the bar keeps out of its way).
    pub fn dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// Whether the drag is held at an edge: frames are wanted.
    pub fn scrolling(&self, content: &dyn RangeContent) -> bool {
        self.drag.is_some() && self.at.is_some_and(|(_, y)| self.edge.speed(content.view(), y) != 0.0)
    }

    /// Time passing (each frame): a drag held at an edge scrolls the
    /// content, and the range goes on. Whether it scrolled.
    pub fn tick(&mut self, content: &mut dyn RangeContent, time: f64) -> bool {
        let dt = self.time.map_or(0.0, |t| (time - t).clamp(0.0, 0.1)) as f32;
        self.time = Some(time);
        let (Some(_), Some((x, y))) = (self.drag, self.at) else { return false };
        let n = self.edge.lines(content.view(), y, content.line_height(), dt);
        if n == 0 || !content.scroll(n) {
            return false;
        }
        self.drag(content, x, y);
        true
    }
}

/// A selection's handles in `parent` (the root: they are placed in the UI's
/// points), in `accent`, for a range whose ends are drawn at `start` and
/// `finish`. They take no presses: [`RangeSelect::press`] does.
pub fn draw_handles(
    ui: &mut Ui,
    parent: NodeId,
    handles: &RangeHandles,
    start: Option<Rect>,
    finish: Option<Rect>,
    view: Rect,
    accent: [f32; 4],
) {
    let size = handles.config.knob;
    for (end, caret) in [(RangeEnd::Start, start), (RangeEnd::Finish, finish)] {
        let Some(c) = caret else { continue };
        let (kx, ky) = handles.knob(end, c, view);
        let mut part = |x: f32, y: f32, w: f32, h: f32, round: f32| {
            let mut s = style! { position: absolute; pointer-events: none; };
            (s.left, s.top) = (Length::Px(x), Length::Px(y));
            (s.width, s.height) = (Length::Px(w), Length::Px(h));
            crate::css::apply(&mut s, "border-radius", &format!("{round}px"));
            s.background = Some(accent);
            ui.add_box(parent, s);
        };
        part(c.x - 1.0, c.y, 2.0, c.height, 0.0);
        part(kx - size / 2.0, ky - size / 2.0, size, size, size / 2.0);
    }
}

/// About how wide a bar of these labels is (14 px text).
pub fn bar_width(labels: &[&str]) -> f32 {
    labels.iter().map(|l| 26.0 + 8.2 * l.chars().count() as f32).sum::<f32>() + 8.0
}

/// A bar of actions at `at` ([`place_bar`]), `size` large, in `look` (its
/// colours, border and corners): the bar, and each label's button.
pub fn draw_bar(ui: &mut Ui, parent: NodeId, at: (f32, f32), size: (f32, f32), labels: &[&str], look: Style) -> (NodeId, Vec<NodeId>) {
    let mut s = look;
    crate::css::apply(&mut s, "position", "absolute");
    crate::css::apply(&mut s, "align-items", "center");
    crate::css::apply(&mut s, "justify-content", "space-between");
    crate::css::apply(&mut s, "padding", "0 4px");
    crate::css::apply(&mut s, "cursor", "pointer");
    crate::css::apply(&mut s, "user-select", "none");
    s.display = Display::Flex;
    (s.left, s.top) = (Length::Px(at.0), Length::Px(at.1));
    (s.width, s.height) = (Length::Px(size.0), Length::Px(size.1));
    let bar = ui.add_box(parent, s);
    ui.set_semantics(bar, Semantics::new(Role::Menu).label("Selection"));
    let items = labels
        .iter()
        .map(|l| {
            let b = ui.add_box(bar, style! { padding: 8px 10px; border-radius: 8px; cursor: pointer; font-size: 14px; white-space: nowrap; });
            ui.add_text(b, *l);
            ui.set_semantics(b, Semantics::new(Role::MenuItem).label(*l));
            b
        })
        .collect();
    (bar, items)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VIEW: Rect = Rect { x: 0.0, y: 100.0, width: 400.0, height: 300.0 };

    fn caret(x: f32, y: f32) -> Rect {
        Rect { x, y, width: 0.0, height: 20.0 }
    }

    #[test]
    fn the_nearer_end_is_taken_and_follows_without_a_jump() {
        let mut h = RangeHandles::new();
        // a word: its ends close together
        let (a, b) = (Some(caret(100.0, 200.0)), Some(caret(120.0, 200.0)));
        assert_eq!(h.end_at(98.0, 190.0, a, b, VIEW), Some(RangeEnd::Start));
        assert_eq!(h.end_at(122.0, 228.0, a, b, VIEW), Some(RangeEnd::Finish));
        assert_eq!(h.end_at(250.0, 210.0, a, b, VIEW), None);
        assert_eq!(h.end_at(110.0, 210.0, a, b, VIEW), None, "the word's middle: a tap on it");
        assert_eq!(h.end_at(98.0, 190.0, None, None, VIEW), None);
        assert_eq!(h.press(124.0, 230.0, a, b, VIEW), Some(RangeEnd::Finish));
        assert_eq!(h.drag(124.0, 230.0), Some((120.0, 210.0)), "not moved: where it was");
        assert!(h.release() && !h.release());
        assert_eq!(h.drag(0.0, 0.0), None);
    }

    #[test]
    fn knobs_stay_in_the_view() {
        let h = RangeHandles::new();
        assert_eq!(h.knob(RangeEnd::Start, caret(0.0, 100.0), VIEW), (6.0, 106.0));
        assert_eq!(h.knob(RangeEnd::Finish, caret(400.0, 380.0), VIEW), (394.0, 394.0));
    }

    #[test]
    fn an_edge_scrolls_faster_further_out_and_only_once_gone_toward() {
        let mut e = EdgeScroll::new();
        e.begin(0.0, 250.0);
        assert_eq!(e.speed(VIEW, 250.0), 0.0);
        let (near, far) = (e.speed(VIEW, 380.0), e.speed(VIEW, 440.0));
        assert!(near > 0.0 && far > near, "{near} {far}");
        assert!(e.speed(VIEW, 110.0) < 0.0);
        // begun on the first line: not until it's pulled up
        e.begin(0.0, 110.0);
        assert_eq!(e.speed(VIEW, 110.0), 0.0);
        assert!(e.speed(VIEW, 95.0) < 0.0);
        // parts of a line add up
        e.begin(0.0, 250.0);
        let n: i32 = (0..60).map(|_| e.lines(VIEW, 380.0, 20.0, 1.0 / 60.0)).sum();
        assert_eq!(n, (near / 20.0) as i32);
    }

    #[test]
    fn a_bar_goes_above_then_below_then_in_the_middle() {
        let bar = |a, b| place_bar(a, b, 200.0, 40.0, 18.0, VIEW);
        assert_eq!(bar(Some(caret(200.0, 300.0)), Some(caret(240.0, 300.0))), (120.0, 242.0));
        assert_eq!(bar(Some(caret(10.0, 120.0)), Some(caret(40.0, 160.0))), (8.0, 198.0));
        assert_eq!(bar(None, None), (100.0, 230.0));
    }
}
