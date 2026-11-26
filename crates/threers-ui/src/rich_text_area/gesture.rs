//! The pointer on the text: the mouse's clicks and drags (in `lib.rs`), and
//! a finger's taps, long press, handles and edit menu (here).

use super::{motion, rect, union, Clip, Document, Effect, Loupe, Menu, Overlay, Pos, Rect, Selection, RichTextArea};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerKind {
    Mouse,
    Touch,
    Pen,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Down,
    Move,
    Up,
    Cancel,
}

/// A pointer event in the document's frame, at a time in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pointer {
    pub kind: PointerKind,
    pub phase: Phase,
    pub x: f32,
    pub y: f32,
    pub time: f64,
    pub shift: bool,
}

impl Pointer {
    pub fn new(kind: PointerKind, phase: Phase, x: f32, y: f32, time: f64) -> Pointer {
        Pointer { kind, phase, x, y, time, shift: false }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    /// Down, not yet a tap, a press or a scroll.
    Pending,
    /// Dragging the selection's start (`false`) or end.
    Handle(bool),
    /// The caret following the finger (a long press).
    Caret,
    /// On a menu item.
    Menu(usize),
    /// Moving: the scroll view's, not ours.
    Scroll,
}

#[derive(Clone, Debug, PartialEq)]
enum Item {
    Cut,
    Copy,
    Paste,
    Select,
    SelectAll,
    Fix(String),
    LookUp,
}

impl Item {
    fn label(&self) -> String {
        use crate::i18n::{word, Word};
        match self {
            Item::Cut => word(Word::Cut),
            Item::Copy => word(Word::Copy),
            Item::Paste => word(Word::Paste),
            Item::Select => word(Word::Select),
            Item::SelectAll => word(Word::SelectAll),
            Item::Fix(s) => s.clone(),
            Item::LookUp => word(Word::LookUp),
        }
    }
}

#[derive(Default)]
pub(crate) struct Touch {
    down: Option<(f32, f32, f64, Mode)>,
    finger: Option<(f32, f32)>,
    pub(super) taps: Vec<(f64, f32, f32)>,
    menu: Option<Vec<Item>>,
    menu_rects: Vec<Rect>,
    handles: bool,
    /// A finger has been used (a phone's text area: handles, suggestions).
    pub used: bool,
}

impl Touch {
    pub fn dismiss(&mut self) {
        self.menu = None;
        self.handles = false;
    }

    pub fn pending(&self) -> bool {
        matches!(self.down, Some((_, _, _, Mode::Pending)))
    }

    fn captured(&self) -> bool {
        matches!(self.down, Some((_, _, _, Mode::Handle(_) | Mode::Caret | Mode::Menu(_))))
    }
}

const LONG_PRESS: f64 = 0.45;
const TAP_GAP: f64 = 0.35;

impl RichTextArea {
    /// Whether a finger now is the text's (dragging a handle or the caret,
    /// on the menu): the scroll view should leave it alone.
    pub fn touch_captured(&self) -> bool {
        self.touch.captured()
    }

    /// Whether the edit menu is showing.
    pub fn menu_shown(&self) -> bool {
        self.touch.menu.is_some()
    }

    /// Shows the edit menu over the selection (or hides it).
    pub fn show_menu(&mut self, doc: &mut dyn Document, show: bool) {
        if show {
            open_menu(self, doc);
        } else {
            self.touch.menu = None;
        }
    }
}

/// Handle knobs: where each end's knob is, and how near counts as on it.
fn knobs(area: &RichTextArea, doc: &dyn Document) -> Option<[(f32, f32); 2]> {
    let s = doc.selection();
    if s.is_caret() || !area.touch.handles {
        return None;
    }
    let (a, b) = s.range();
    let (ra, rb) = (doc.caret_rect(a, false), doc.caret_rect(b, true));
    let k = 6.0 / area.scale;
    Some([(ra.x, ra.y - k), (rb.x, rb.y + rb.height + k)])
}

pub(crate) fn wants(area: &RichTextArea, doc: &dyn Document, x: f32, y: f32) -> bool {
    let slop = 24.0 / area.scale;
    if area.touch.menu.is_some() && area.touch.menu_rects.iter().any(|r| r.contains(x, y)) {
        return true;
    }
    knobs(area, doc).is_some_and(|ks| ks.iter().any(|&(kx, ky)| (kx - x).abs() < slop && (ky - y).abs() < slop * 1.5))
}

fn open_menu(area: &mut RichTextArea, doc: &mut dyn Document) {
    let s = doc.selection();
    let mut items = Vec::new();
    if s.is_caret() {
        items.extend([Item::Select, Item::SelectAll, Item::Paste]);
    } else {
        // a misspelt word: its fixes first
        let word = area.selected_text(doc);
        if !word.contains(char::is_whitespace) && word.chars().count() > 1 && !area.autocorrect.is_correct(&word) {
            for f in area.autocorrect.suggestions(&word, 3) {
                items.push(Item::Fix(f));
            }
        }
        items.extend([Item::Cut, Item::Copy, Item::Paste, Item::LookUp]);
    }
    area.touch.menu = Some(items);
}

pub(crate) fn touch(area: &mut RichTextArea, doc: &mut dyn Document, p: super::Pointer) -> Vec<Effect> {
    area.touch.used = true;
    let slop = 10.0 / area.scale;
    match p.phase {
        Phase::Down => {
            let mode = if let Some(i) = area.touch.menu_rects.iter().position(|r| r.contains(p.x, p.y)).filter(|_| area.touch.menu.is_some()) {
                Mode::Menu(i)
            } else if let Some(ks) = knobs(area, doc) {
                let near = |k: (f32, f32)| (k.0 - p.x).hypot(k.1 - p.y);
                let (d0, d1) = (near(ks[0]), near(ks[1]));
                if d0.min(d1) < 30.0 / area.scale { Mode::Handle(d1 < d0) } else { Mode::Pending }
            } else {
                Mode::Pending
            };
            area.touch.down = Some((p.x, p.y, p.time, mode));
            if matches!(mode, Mode::Handle(_)) {
                area.touch.finger = Some((p.x, p.y));
                area.touch.menu = None;
            }
            vec![Effect::Tick]
        }
        Phase::Move => {
            let Some((x0, y0, t0, mode)) = area.touch.down else { return Vec::new() };
            match mode {
                Mode::Pending if (p.x - x0).hypot(p.y - y0) > slop => {
                    area.touch.down = Some((x0, y0, t0, Mode::Scroll));
                    Vec::new()
                }
                Mode::Handle(end) => {
                    // the finger covers what it moves: aim a little above it
                    let (at, _) = doc.hit(p.x, p.y - 12.0 / area.scale);
                    let (a, b) = doc.selection().range();
                    let s = if end {
                        Selection { anchor: a, head: at.max(motion_next(doc, a)), upstream: false }
                    } else {
                        Selection { anchor: b, head: at.min(motion_prev(doc, b)), upstream: false }
                    };
                    doc.set_selection(s);
                    area.touch.finger = Some((p.x, p.y));
                    vec![Effect::Redraw]
                }
                Mode::Caret => {
                    let (at, up) = doc.hit(p.x, p.y - 12.0 / area.scale);
                    doc.set_selection(Selection { anchor: at, head: at, upstream: up });
                    area.touch.finger = Some((p.x, p.y));
                    vec![Effect::Redraw]
                }
                _ => Vec::new(),
            }
        }
        Phase::Up => {
            let Some((x0, y0, t0, mode)) = area.touch.down.take() else { return Vec::new() };
            area.touch.finger = None;
            match mode {
                Mode::Pending if p.time - t0 < LONG_PRESS && (p.x - x0).hypot(p.y - y0) <= slop => tap(area, doc, p),
                Mode::Handle(_) | Mode::Caret => {
                    open_menu(area, doc);
                    area.touch.handles = !doc.selection().is_caret();
                    vec![Effect::Redraw]
                }
                Mode::Menu(i) if area.touch.menu_rects.get(i).is_some_and(|r| r.contains(p.x, p.y)) => choose(area, doc, i),
                _ => vec![Effect::Redraw],
            }
        }
        Phase::Cancel => {
            area.touch.down = None;
            area.touch.finger = None;
            vec![Effect::Redraw]
        }
    }
}

fn motion_next(doc: &dyn Document, p: Pos) -> Pos {
    motion::step(doc, p, false, super::Motion::Right, 0.0).0
}

fn motion_prev(doc: &dyn Document, p: Pos) -> Pos {
    motion::step(doc, p, false, super::Motion::Left, 0.0).0
}

fn tap(area: &mut RichTextArea, doc: &mut dyn Document, p: super::Pointer) -> Vec<Effect> {
    let reach = 30.0 / area.scale;
    area.touch.taps.retain(|t| p.time - t.0 < TAP_GAP && (t.1 - p.x).hypot(t.2 - p.y) < reach);
    area.touch.taps.push((p.time, p.x, p.y));
    let n = area.touch.taps.len().min(3) as u32;
    let before = doc.selection();
    match n {
        1 => {
            let (at, up) = doc.hit(p.x, p.y);
            // on the caret: the menu, or away
            let on_caret = before.is_caret() && {
                let r = doc.caret_rect(before.head, before.upstream);
                (r.x - p.x).abs() < reach && p.y >= r.y - reach / 2.0 && p.y <= r.y + r.height + reach / 2.0
            };
            if on_caret {
                if area.touch.menu.is_some() {
                    area.touch.menu = None;
                } else {
                    open_menu(area, doc);
                }
                return vec![Effect::Redraw];
            }
            // on a misspelt word: it, and its fixes
            let m = area.misspelled(doc, at.para);
            if let Some(&(a, b)) = m.iter().find(|(a, b)| *a < at.off && at.off < *b) {
                area.select(doc, Pos::new(at.para, a), Pos::new(at.para, b));
                area.touch.handles = true;
                open_menu(area, doc);
                return vec![Effect::Redraw];
            }
            area.press(doc, p.x, p.y, 1, false);
            doc.set_selection(Selection { anchor: at, head: at, upstream: up });
            area.touch.dismiss();
            vec![Effect::Redraw, Effect::ShowCaret]
        }
        n => {
            area.press(doc, p.x, p.y, n, false);
            area.touch.handles = true;
            open_menu(area, doc);
            vec![Effect::Redraw]
        }
    }
}

fn choose(area: &mut RichTextArea, doc: &mut dyn Document, i: usize) -> Vec<Effect> {
    let Some(item) = area.touch.menu.as_ref().and_then(|m| m.get(i)).cloned() else { return Vec::new() };
    area.touch.menu = None;
    match item {
        Item::Cut => vec![Effect::Clipboard(Clip::Cut)],
        Item::Copy => vec![Effect::Clipboard(Clip::Copy), Effect::Redraw],
        Item::Paste => vec![Effect::Clipboard(Clip::Paste)],
        Item::Select => {
            let h = doc.selection().head;
            let (a, b) = motion::word_at(&doc.text(h.para), h.off);
            area.select(doc, Pos::new(h.para, a), Pos::new(h.para, b));
            area.touch.handles = true;
            open_menu(area, doc);
            vec![Effect::Redraw]
        }
        Item::SelectAll => {
            let end = motion::step(doc, Pos::default(), false, super::Motion::DocEnd, 0.0).0;
            area.select(doc, Pos::default(), end);
            area.touch.handles = true;
            open_menu(area, doc);
            vec![Effect::Redraw]
        }
        Item::Fix(s) => {
            doc.checkpoint();
            let (a, b) = doc.selection().range();
            let end = doc.replace(a, b, &s, false);
            doc.set_selection(Selection::caret(end));
            area.touch.handles = false;
            vec![Effect::Redraw, Effect::ShowCaret]
        }
        Item::LookUp => vec![Effect::LookUp(area.selected_text(doc))],
    }
}

pub(crate) fn tick(area: &mut RichTextArea, doc: &mut dyn Document, time: f64) -> Vec<Effect> {
    if let Some((x, y, t0, Mode::Pending)) = area.touch.down
        && time - t0 >= LONG_PRESS
    {
        // a long press: the caret under the finger, a loupe over it
        area.touch.down = Some((x, y, t0, Mode::Caret));
        area.touch.menu = None;
        area.touch.handles = false;
        let (at, up) = doc.hit(x, y);
        doc.set_selection(Selection { anchor: at, head: at, upstream: up });
        area.touch.finger = Some((x, y));
        return vec![Effect::Redraw];
    }
    if area.touch.pending() || area.focused { vec![Effect::Tick] } else { Vec::new() }
}

pub(crate) fn overlay(area: &mut RichTextArea, doc: &dyn Document, o: &mut Overlay) {
    let k = 1.0 / area.scale;
    if let Some(ks) = knobs(area, doc) {
        let (a, b) = doc.selection().range();
        for (i, p) in [a, b].into_iter().enumerate() {
            let r = doc.caret_rect(p, i == 1);
            o.handles.push((rect(r.x - k, r.y, 2.0 * k, r.height), ks[i]));
        }
    }
    if let Some((fx, fy)) = area.touch.finger {
        let s = doc.selection();
        let r = doc.caret_rect(s.head, s.upstream);
        let radius = 48.0 * k;
        // above the finger, or below it where the text's top leaves no room
        let above = fy - 80.0 * k;
        let y = if above - radius < 0.0 { fy + 80.0 * k } else { above };
        o.loupe = Some(Loupe { x: fx.max(radius), y, focus_x: fx, focus_y: r.y + r.height / 2.0, radius, zoom: 1.5 });
    }
    area.touch.menu_rects.clear();
    if let Some(items) = &area.touch.menu {
        let size = 14.0 * k;
        let (pad, h) = (12.0 * k, 34.0 * k);
        let s = doc.selection();
        let anchor = if s.is_caret() {
            doc.caret_rect(s.head, s.upstream)
        } else {
            let (a, b) = s.range();
            doc.selection_rects(a, b).iter().fold(None::<Rect>, |acc, r| Some(acc.map_or(*r, |u| union(&u, r)))).unwrap_or_default()
        };
        let widths: Vec<f32> = items.iter().map(|i| doc.measure(&i.label(), size) + 2.0 * pad).collect();
        let total: f32 = widths.iter().sum();
        let x0 = (anchor.x + anchor.width / 2.0 - total / 2.0).max(0.0);
        let mut y0 = anchor.y - 10.0 * k - h;
        if y0 < 0.0 {
            y0 = anchor.y + anchor.height + 24.0 * k;
        }
        let mut x = x0;
        let mut out = Vec::new();
        for (item, w) in items.iter().zip(&widths) {
            let r = rect(x, y0, *w, h);
            out.push((item.label(), r));
            area.touch.menu_rects.push(r);
            x += w;
        }
        o.menu = Some(Menu { rect: rect(x0, y0, total, h), items: out, font_size: size });
    }
    // the correction autocorrect would make, under the word (on a phone)
    if area.touch.used
        && let Some((a, _, fix)) = area.autocorrect.pending(doc)
    {
        let r = doc.caret_rect(a, false);
        o.suggestion = Some((rect(r.x, r.y + r.height + 2.0 * k, doc.measure(&fix, 13.0 * k) + 12.0 * k, 20.0 * k), fix));
    }
}
