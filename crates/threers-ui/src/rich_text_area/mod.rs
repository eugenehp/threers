//! A rich text area's editing (a word processor's, a notes app's), whatever holds the text: paragraphs of it,
//! laid out however the app likes.
//!
//! [`RichTextArea`] is everything about editing text that is not the text
//! itself or how it is drawn:
//!
//! - **the caret and the selection** — moving by character (whole
//!   graphemes), word, line, paragraph, page and document; the caret at a
//!   wrapped line's end drawn there, not at the next line's start; up and
//!   down keeping their column; the caret blinking, steady while typing;
//! - **the mouse** — a click places the caret, two select a word, three a
//!   paragraph, a drag extends by what the click selected, shift extends;
//! - **the finger** — a tap places the caret (on it: the edit menu), two
//!   taps a word, three a paragraph, a long press the caret under a loupe,
//!   a selection's two handles dragging its ends, and an edit menu (cut,
//!   copy, paste, select, select all, spelling fixes) laid out and
//!   hit-tested here, drawn by the app ([`Overlay`]);
//! - **the keyboard** — each platform's shortcuts ([`keys`]): moving,
//!   selecting and deleting by word and line, Emacs keys on the Mac, the
//!   clipboard, undo, and the formatting a rich-text editor takes, all
//!   rebindable;
//! - **autocorrect** ([`autocorrect`]) — misspellings fixed as a word ends
//!   (threers-spell), sentences capitalised, smart quotes and dashes, a
//!   double space a full stop; Backspace straight after puts it back;
//!   suggestions for the word being typed;
//! - **input methods** — composing (marked) text, replaced as it changes;
//! - **spelling** — the words to underline;
//! - **find and replace**, and when to begin an undo step (a word a step).
//!
//! The app's text implements [`Document`]: its paragraphs' text, replacing a
//! stretch, and where things are on the screen. Commands that are the
//! app's (a new paragraph, bold, a page break) come back as [`Effect`]s,
//! as do the clipboard's (which only the platform reaches).
//!
//! ```
//! use threers_ui::rich_text_area::{plain::PlainText, Pos, RichTextArea};
//!
//! let mut doc = PlainText::new("Hello wrold");
//! let mut area = RichTextArea::new();
//! area.select(&mut doc, Pos::new(0, 11), Pos::new(0, 11));
//! area.text(&mut doc, " ", 0.0); // the word ends: corrected
//! assert_eq!(doc.text_of(0), "Hello world ");
//! area.key(&mut doc, "Backspace", Default::default(), 0.1); // and put back
//! assert_eq!(doc.text_of(0), "Hello wrold");
//! ```

pub mod autocorrect;
mod gesture;
pub mod keys;
pub mod motion;
pub mod plain;

use std::collections::HashMap;

pub use gesture::{Phase, Pointer, PointerKind};
pub use keys::{Clip, Command, KeyAction, Keymap, Mods, Platform};
pub use motion::Motion;

/// A place in the text: a paragraph, a byte in its UTF-8 text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Pos {
    pub para: usize,
    pub off: usize,
}

impl Pos {
    pub fn new(para: usize, off: usize) -> Pos {
        Pos { para, off }
    }
}

pub use crate::Rect;

/// A rectangle.
pub fn rect(x: f32, y: f32, width: f32, height: f32) -> Rect {
    Rect { x, y, width, height }
}

/// The smallest rectangle holding both.
pub fn union(a: &Rect, b: &Rect) -> Rect {
    let (x0, y0) = (a.x.min(b.x), a.y.min(b.y));
    let (x1, y1) = ((a.x + a.width).max(b.x + b.width), (a.y + a.height).max(b.y + b.height));
    rect(x0, y0, x1 - x0, y1 - y0)
}

/// The selection: where it began and where its moving end is, and whether
/// a caret at a wrapped line's end is drawn there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Selection {
    pub anchor: Pos,
    pub head: Pos,
    pub upstream: bool,
}

impl Selection {
    pub fn caret(p: Pos) -> Selection {
        Selection { anchor: p, head: p, upstream: false }
    }

    /// Its ends in order.
    pub fn range(&self) -> (Pos, Pos) {
        if self.anchor <= self.head { (self.anchor, self.head) } else { (self.head, self.anchor) }
    }

    pub fn is_caret(&self) -> bool {
        self.anchor == self.head
    }
}
/// The text a [`RichTextArea`] edits, and where it is on the screen (any
/// frame the app likes, y down).
pub trait Document {
    fn paragraphs(&self) -> usize;
    /// A paragraph's text (what is not text, a picture, as U+FFFC).
    fn text(&self, para: usize) -> String;
    fn selection(&self) -> Selection;
    fn set_selection(&mut self, s: Selection);
    /// Puts `text` in place of `a..b` (a newline in it begins a paragraph);
    /// where it ends. `typing`: as a keystroke (undone with its word).
    fn replace(&mut self, a: Pos, b: Pos, text: &str, typing: bool) -> Pos;
    /// Takes out `a..b`, `backward` as Backspace does (a document may do
    /// more at a paragraph's start: take a list's number off).
    fn delete(&mut self, a: Pos, b: Pos, backward: bool) -> Pos {
        let _ = backward;
        self.replace(a, b, "", true)
    }
    /// The place nearest a point, and whether it is a wrapped line's end.
    fn hit(&self, x: f32, y: f32) -> (Pos, bool);
    /// The caret's box at a place.
    fn caret_rect(&self, p: Pos, upstream: bool) -> Rect;
    /// Boxes over `a..b`, a line each.
    fn selection_rects(&self, a: Pos, b: Pos) -> Vec<Rect>;
    /// The line holding `p`: its first and last places.
    fn line_ends(&self, p: Pos, upstream: bool) -> (Pos, Pos);
    /// The place at `x` on the line above (below) the one holding `p`, and
    /// whether it is a wrapped line's end; `None` at the text's ends.
    fn line_step(&self, p: Pos, upstream: bool, down: bool, x: f32) -> Option<(Pos, bool)>;
    /// How the line at `p` is turned: 0 level, 1 read upwards (its lines
    /// following rightwards), 2 read downwards (its lines following
    /// leftwards, as East Asian vertical text). The arrow keys turn with
    /// it, and `line_step`'s `x` is then the caret's `y`, along the line.
    fn turned(&self, _p: Pos) -> u8 {
        0
    }
    /// Whether the character at `p` is not seen (a bookmark): the caret goes past it.
    fn hidden(&self, p: Pos) -> bool {
        let _ = p;
        false
    }
    /// How wide `text` is at `size` (for the edit menu's labels).
    fn measure(&self, text: &str, size: f32) -> f32 {
        text.chars().count() as f32 * size * 0.55
    }
    /// `a..b` moved to `to` (a place outside it), as dragging the
    /// selection moves it: where it now stands. By default its text taken
    /// out and put in again; a document with formatting keeps it (and may
    /// record the move).
    fn move_range(&mut self, a: Pos, b: Pos, to: Pos) -> Option<(Pos, Pos)> {
        if to >= a && to <= b {
            return None;
        }
        let mut text = String::new();
        for p in a.para..=b.para {
            let t = self.text(p);
            let (x, y) = (if p == a.para { a.off } else { 0 }, if p == b.para { b.off } else { t.len() });
            text.push_str(&t[x..y]);
            if p < b.para {
                text.push('\n');
            }
        }
        if to > b {
            let end = self.replace(to, to, &text, false);
            self.replace(a, b, "", false);
            // what was after `b` in its paragraph moved back by what went
            let shift = |p: Pos| if p.para == b.para { Pos::new(a.para, a.off + p.off - b.off) } else { Pos::new(p.para - (b.para - a.para), p.off) };
            Some((shift(to), shift(end)))
        } else {
            self.replace(a, b, "", false);
            let end = self.replace(to, to, &text, false);
            Some((to, end))
        }
    }

    /// Begins an undo step: what is changed next is undone apart from what
    /// was changed before.
    fn checkpoint(&mut self) {}
    fn undo(&mut self) {}
    fn redo(&mut self) {}
}

/// What the app is to do after a call.
#[derive(Clone, Debug, PartialEq)]
pub enum Effect {
    /// Something changed: draw again.
    Redraw,
    /// The caret moved or the text changed: bring the caret into view.
    ShowCaret,
    /// A command that is the app's (a new paragraph, bold).
    Command(Command),
    /// The clipboard: put the selection there, or what is there here
    /// ([`RichTextArea::paste`]).
    Clipboard(Clip),
    /// Look the selection up, or share it (the platform's).
    LookUp(String),
    /// Draw again soon: the caret blinks, a long press is timing.
    Tick,
}

/// What the app draws over (and under) its text.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Overlay {
    /// Under the text.
    pub selection: Vec<Rect>,
    /// The caret, when it is showing (it blinks).
    pub caret: Option<Rect>,
    /// Text being composed: underlines.
    pub marked: Vec<Rect>,
    /// Misspelt words: wavy underlines from `x0` to `x1` at `y`.
    pub squiggles: Vec<(f32, f32, f32)>,
    /// A selection's handles (on a finger): each a bar and a knob's centre.
    pub handles: Vec<(Rect, (f32, f32))>,
    /// The loupe: where its circle is, what it shows (centre), radius, magnification.
    pub loupe: Option<Loupe>,
    pub menu: Option<Menu>,
    /// The word autocorrect would put in place of the one being typed (on a phone).
    pub suggestion: Option<(Rect, String)>,
    /// Lengths in the document's frame for this many points on the screen.
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Loupe {
    pub x: f32,
    pub y: f32,
    pub focus_x: f32,
    pub focus_y: f32,
    pub radius: f32,
    pub zoom: f32,
}

/// The edit menu: its box and its items' (each a label and a box).
#[derive(Clone, Debug, PartialEq)]
pub struct Menu {
    pub rect: Rect,
    pub items: Vec<(String, Rect)>,
    /// Its labels' size.
    pub font_size: f32,
}

/// What a step of editing was, for undo: a word typed is one step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Last {
    None,
    Typing,
    Deleting,
}

/// A rich text area's editing state. See the crate's documentation.
pub struct RichTextArea {
    keymap: Keymap,
    /// The column up and down keep.
    goal_x: Option<f32>,
    /// What a drag extends by (1 characters, 2 words, 3 paragraphs), and
    /// what the press that began it selected.
    granularity: u32,
    origin: (Pos, Pos),
    dragging: bool,
    /// A press inside the selection: it may be dragged away (where it was
    /// pressed, where it is now dropped).
    moving: Option<((f32, f32), Option<Pos>)>,
    clicks: Vec<(f64, f32, f32)>,
    last: Last,
    /// When the caret last moved (it shows steadily for a moment after).
    moved_at: f64,
    pub autocorrect: autocorrect::Autocorrect,
    /// Underline misspelt words.
    pub spellcheck: bool,
    misspelled: HashMap<String, Vec<(usize, usize)>>,
    marked: Option<(Pos, Pos)>,
    touch: gesture::Touch,
    /// Screen points a unit of the document's frame (sizes on the screen
    /// that don't zoom: handles, the menu, slop).
    pub scale: f32,
    /// The text the app copied last (pasting it back keeps its formatting:
    /// the app knows).
    pub focused: bool,
}

impl Default for RichTextArea {
    fn default() -> Self {
        Self::new()
    }
}

const BLINK: f64 = 0.53;

impl RichTextArea {
    pub fn new() -> RichTextArea {
        RichTextArea::with_platform(Platform::current())
    }

    pub fn with_platform(p: Platform) -> RichTextArea {
        RichTextArea {
            keymap: Keymap::defaults(p),
            goal_x: None,
            granularity: 1,
            origin: (Pos::default(), Pos::default()),
            dragging: false,
            moving: None,
            clicks: Vec::new(),
            last: Last::None,
            moved_at: 0.0,
            autocorrect: autocorrect::Autocorrect::new(),
            spellcheck: true,
            misspelled: HashMap::new(),
            marked: None,
            touch: gesture::Touch::default(),
            scale: 1.0,
            focused: true,
        }
    }

    pub fn keymap(&mut self) -> &mut Keymap {
        &mut self.keymap
    }

    /// Selects `a..b` (the head at `b`).
    pub fn select(&mut self, doc: &mut dyn Document, a: Pos, b: Pos) {
        doc.set_selection(Selection { anchor: a, head: b, upstream: false });
        self.after_move();
    }

    fn after_move(&mut self) {
        self.last = Last::None;
        self.marked = None;
        self.autocorrect.forget();
    }

    /// The selection's text.
    pub fn selected_text(&self, doc: &dyn Document) -> String {
        let (a, b) = doc.selection().range();
        text_between(doc, a, b)
    }

    // --- the keyboard

    /// A key (DOM `KeyboardEvent.key`: `ArrowLeft`, `Backspace`, `a`), at a
    /// time in seconds. Whether it was taken, and what the app is to do.
    pub fn key(&mut self, doc: &mut dyn Document, key: &str, mods: Mods, time: f64) -> (bool, Vec<Effect>) {
        self.moved_at = time;
        self.touch.dismiss();
        let Some(action) = self.keymap.action(key, mods) else { return (false, Vec::new()) };
        let fx = self.act(doc, action, time);
        (true, fx)
    }

    /// Carries out an action (as a key or a menu would).
    pub fn act(&mut self, doc: &mut dyn Document, action: KeyAction, time: f64) -> Vec<Effect> {
        self.moved_at = time;
        match action {
            KeyAction::Move(m, extend) => {
                self.move_caret(doc, m, extend);
                vec![Effect::Redraw, Effect::ShowCaret]
            }
            KeyAction::Delete(m) => self.delete(doc, m),
            KeyAction::SelectAll => {
                let end = motion::step(doc, Pos::default(), false, Motion::DocEnd, 0.0).0;
                self.select(doc, Pos::default(), end);
                vec![Effect::Redraw]
            }
            KeyAction::Clipboard(c) => vec![Effect::Clipboard(c)],
            KeyAction::Command(Command::Undo) => {
                self.after_move();
                doc.undo();
                vec![Effect::Redraw, Effect::ShowCaret]
            }
            KeyAction::Command(Command::Redo) => {
                self.after_move();
                doc.redo();
                vec![Effect::Redraw, Effect::ShowCaret]
            }
            KeyAction::Command(c) => {
                if matches!(c, Command::Enter | Command::LineBreak | Command::Tab | Command::PageBreak) {
                    // a word ends here too: corrected first
                    self.autocorrect.word_ends(doc);
                    self.last = Last::None;
                    self.marked = None;
                } else {
                    self.after_move();
                }
                vec![Effect::Command(c), Effect::Redraw, Effect::ShowCaret]
            }
            KeyAction::Text(t) => self.text(doc, &t, time),
        }
    }

    /// Moves the caret, or (with `extend`) the selection's moving end.
    pub fn move_caret(&mut self, doc: &mut dyn Document, m: Motion, extend: bool) {
        let s = doc.selection();
        // a turned line: the arrows as it reads (along it, and to the lines beside it)
        let turned = doc.turned(s.head);
        let m = match (turned, m) {
            (2, Motion::Down) | (1, Motion::Up) => Motion::Right,
            (2, Motion::Up) | (1, Motion::Down) => Motion::Left,
            (2, Motion::Left) | (1, Motion::Right) => Motion::Down,
            (2, Motion::Right) | (1, Motion::Left) => Motion::Up,
            _ => m,
        };
        let vertical = matches!(m, Motion::Up | Motion::Down | Motion::PageUp | Motion::PageDown);
        if vertical && self.goal_x.is_none() {
            let r = doc.caret_rect(s.head, s.upstream);
            self.goal_x = Some(if turned != 0 { r.y } else { r.x });
        }
        let (to, upstream) = if !extend && !s.is_caret() && matches!(m, Motion::Left | Motion::Right) {
            // a selection collapses to its end that way
            let (a, b) = s.range();
            (if m == Motion::Left { a } else { b }, false)
        } else {
            motion::step(doc, s.head, s.upstream, m, self.goal_x.unwrap_or(0.0))
        };
        let goal = self.goal_x;
        let anchor = if extend { s.anchor } else { to };
        doc.set_selection(Selection { anchor, head: to, upstream });
        self.after_move();
        self.goal_x = if vertical { goal } else { None };
        self.granularity = 1;
    }

    /// Deletes by a motion from the caret, or the selection.
    fn delete(&mut self, doc: &mut dyn Document, m: Motion) -> Vec<Effect> {
        let s = doc.selection();
        let backward = matches!(m, Motion::Left | Motion::WordLeft | Motion::LineStart | Motion::ParaStart);
        // Backspace straight after a correction: the word as it was typed
        if backward && m == Motion::Left && s.is_caret() && self.autocorrect.revert(doc) {
            self.last = Last::None;
            return vec![Effect::Redraw, Effect::ShowCaret];
        }
        let (a, b) = if s.is_caret() {
            let (to, _) = motion::step(doc, s.head, s.upstream, m, 0.0);
            if to <= s.head { (to, s.head) } else { (s.head, to) }
        } else {
            s.range()
        };
        if a == b && s.is_caret() {
            return Vec::new();
        }
        // a character at a time goes on deleting in one step; anything else is its own
        let continuing = s.is_caret() && matches!(m, Motion::Left | Motion::Right) && self.last == Last::Deleting;
        if !continuing {
            doc.checkpoint();
        }
        self.marked = None;
        self.autocorrect.forget();
        let at = doc.delete(a, b, backward);
        doc.set_selection(Selection::caret(at));
        self.last = if s.is_caret() && matches!(m, Motion::Left | Motion::Right) { Last::Deleting } else { Last::None };
        self.goal_x = None;
        vec![Effect::Redraw, Effect::ShowCaret]
    }

    /// Text typed (or dictated, or a key's character): over the selection,
    /// autocorrected as words end.
    pub fn text(&mut self, doc: &mut dyn Document, text: &str, time: f64) -> Vec<Effect> {
        self.moved_at = time;
        self.touch.dismiss();
        if text.is_empty() {
            return Vec::new();
        }
        if let Some((a, b)) = self.marked.take() {
            // the composed text, final
            doc.set_selection(Selection { anchor: a, head: b, upstream: false });
        }
        let single = text.chars().count() == 1 && !text.contains('\n');
        if !single || self.last != Last::Typing || !doc.selection().is_caret() {
            doc.checkpoint();
        }
        let text = if single { self.autocorrect.typed(doc, text) } else { text.to_string() };
        let (a, b) = doc.selection().range();
        let end = doc.replace(a, b, &text, single);
        doc.set_selection(Selection::caret(end));
        // a space ends a word, and its undo step
        self.last = if single && !text.chars().all(char::is_whitespace) { Last::Typing } else { Last::None };
        self.goal_x = None;
        vec![Effect::Redraw, Effect::ShowCaret]
    }

    /// Pastes text (the app pastes formatted text itself, then calls
    /// [`select`](Self::select) where it ends).
    pub fn paste(&mut self, doc: &mut dyn Document, text: &str) -> Vec<Effect> {
        self.marked = None;
        self.autocorrect.forget();
        doc.checkpoint();
        let (a, b) = doc.selection().range();
        let end = doc.replace(a, b, &text.replace("\r\n", "\n").replace('\r', "\n"), false);
        doc.set_selection(Selection::caret(end));
        self.last = Last::None;
        vec![Effect::Redraw, Effect::ShowCaret]
    }

    /// The selection taken out (after the app copied it: a cut).
    pub fn cut(&mut self, doc: &mut dyn Document) -> Vec<Effect> {
        let (a, b) = doc.selection().range();
        if a == b {
            return Vec::new();
        }
        doc.checkpoint();
        let at = doc.delete(a, b, true);
        doc.set_selection(Selection::caret(at));
        self.after_move();
        vec![Effect::Redraw, Effect::ShowCaret]
    }

    // --- input methods

    /// Composing text (an input method's), in place of what was being
    /// composed (or the selection).
    pub fn set_marked(&mut self, doc: &mut dyn Document, text: &str) {
        let (a, b) = match self.marked.take() {
            Some(m) => m,
            None => {
                doc.checkpoint();
                doc.selection().range()
            }
        };
        let text = text.replace(['\n', '\r'], " ");
        let end = doc.replace(a, b, &text, true);
        doc.set_selection(Selection::caret(end));
        self.marked = if text.is_empty() { None } else { Some((a, end)) };
        self.last = Last::Typing;
    }

    /// The composed text kept as it is.
    pub fn unmark(&mut self) {
        self.marked = None;
    }

    pub fn marked(&self) -> Option<(Pos, Pos)> {
        self.marked
    }

    // --- the mouse and the finger

    /// A pointer event (document frame). What it does depends on its kind:
    /// see the crate's documentation.
    pub fn pointer(&mut self, doc: &mut dyn Document, p: Pointer) -> Vec<Effect> {
        self.moved_at = p.time;
        match p.kind {
            PointerKind::Mouse | PointerKind::Pen => self.mouse(doc, p),
            PointerKind::Touch => gesture::touch(self, doc, p),
        }
    }

    /// Whether a finger coming down here is the text's (a handle, the menu,
    /// the caret's line) rather than the scroll view's.
    pub fn wants_touch(&self, doc: &dyn Document, x: f32, y: f32) -> bool {
        gesture::wants(self, doc, x, y)
    }

    fn mouse(&mut self, doc: &mut dyn Document, p: Pointer) -> Vec<Effect> {
        match p.phase {
            Phase::Down => {
                // two or three presses close in time and place
                self.clicks.retain(|c| p.time - c.0 < 0.5 && (c.1 - p.x).abs() < 5.0 && (c.2 - p.y).abs() < 5.0);
                self.clicks.push((p.time, p.x, p.y));
                let count = (self.clicks.len() as u32).min(3);
                if count == 3 {
                    self.clicks.clear();
                    self.clicks.push((p.time, p.x, p.y));
                    self.clicks.push((p.time, p.x, p.y));
                }
                // a press in the selection: it may be dragged to move it
                let s = doc.selection();
                let (at, _) = doc.hit(p.x, p.y);
                let (a, b) = s.range();
                if count == 1 && !p.shift && !s.is_caret() && at > a && at < b {
                    self.moving = Some(((p.x, p.y), None));
                    return vec![Effect::Redraw];
                }
                self.press(doc, p.x, p.y, count, p.shift);
                self.dragging = true;
                vec![Effect::Redraw]
            }
            Phase::Move if self.moving.is_some() => {
                let ((x0, y0), _) = self.moving.unwrap();
                if (p.x - x0).hypot(p.y - y0) > 4.0 {
                    let (to, _) = doc.hit(p.x, p.y);
                    self.moving = Some(((x0, y0), Some(to)));
                }
                vec![Effect::Redraw]
            }
            Phase::Move if self.dragging => {
                self.drag(doc, p.x, p.y);
                vec![Effect::Redraw]
            }
            Phase::Up if self.moving.is_some() => {
                let ((x0, y0), to) = self.moving.take().unwrap();
                match to {
                    Some(to) if p.phase == Phase::Up => {
                        let (a, b) = doc.selection().range();
                        doc.checkpoint();
                        if let Some((na, nb)) = doc.move_range(a, b, to) {
                            doc.set_selection(Selection { anchor: na, head: nb, upstream: false });
                            self.last = Last::None;
                        }
                    }
                    // not dragged: a click, the caret where it was pressed
                    _ => self.press(doc, x0, y0, 1, false),
                }
                vec![Effect::Redraw, Effect::ShowCaret]
            }
            Phase::Up | Phase::Cancel => {
                self.moving = None;
                self.dragging = false;
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    /// Presses and taps so far counted as new: the next is a first (after
    /// the document changed what it shows under the pointer).
    pub fn forget_clicks(&mut self) {
        self.clicks.clear();
        self.touch.taps.clear();
    }

    /// A press (`count` the clicks), selecting by its unit.
    pub fn press(&mut self, doc: &mut dyn Document, x: f32, y: f32, count: u32, extend: bool) {
        let (p, upstream) = doc.hit(x, y);
        self.goal_x = None;
        self.granularity = count.clamp(1, 3);
        let unit = self.unit_at(doc, p, self.granularity);
        if extend {
            let (a, b) = doc.selection().range();
            let (anchor, head) = if unit.0 < a { (b, unit.0) } else { (a, unit.1) };
            doc.set_selection(Selection { anchor, head, upstream: false });
            self.origin = (anchor, anchor);
        } else {
            self.origin = unit;
            doc.set_selection(Selection { anchor: unit.0, head: unit.1, upstream: upstream && self.granularity == 1 });
        }
        self.after_move();
    }

    /// The pointer dragged: the selection's end follows it by the unit the
    /// press selected.
    pub fn drag(&mut self, doc: &mut dyn Document, x: f32, y: f32) {
        let (p, upstream) = doc.hit(x, y);
        let unit = self.unit_at(doc, p, self.granularity);
        let (oa, ob) = self.origin;
        let s = if unit.0 < oa {
            Selection { anchor: ob, head: unit.0, upstream: false }
        } else {
            Selection { anchor: oa, head: unit.1.max(ob), upstream: upstream && self.granularity == 1 }
        };
        doc.set_selection(s);
        self.marked = None;
    }

    /// What a press selects at `p`: the place, its word, its paragraph.
    fn unit_at(&self, doc: &dyn Document, p: Pos, granularity: u32) -> (Pos, Pos) {
        match granularity {
            2 => {
                let (a, b) = motion::word_at(&doc.text(p.para), p.off);
                (Pos::new(p.para, a), Pos::new(p.para, b))
            }
            3 => (Pos::new(p.para, 0), Pos::new(p.para, doc.text(p.para).len())),
            _ => (p, p),
        }
    }

    /// Time passing (a long press, the caret's blink): call while
    /// [`needs_tick`](Self::needs_tick).
    pub fn tick(&mut self, doc: &mut dyn Document, time: f64) -> Vec<Effect> {
        gesture::tick(self, doc, time)
    }

    pub fn needs_tick(&self) -> bool {
        self.focused || self.touch.pending()
    }

    // --- spelling

    /// The misspelt words of a paragraph: byte ranges.
    pub fn misspelled(&mut self, doc: &dyn Document, para: usize) -> Vec<(usize, usize)> {
        let text = doc.text(para);
        if let Some(m) = self.misspelled.get(&text) {
            return m.clone();
        }
        let m = self.autocorrect.misspelled(&text);
        if self.misspelled.len() > 20000 {
            self.misspelled.clear();
        }
        self.misspelled.insert(text, m.clone());
        m
    }

    /// The misspelt word at a point, selected, and fixes for it.
    pub fn spelling_at(&mut self, doc: &mut dyn Document, x: f32, y: f32) -> Option<(String, Vec<String>)> {
        let (p, _) = doc.hit(x, y);
        let m = self.misspelled(doc, p.para);
        let &(a, b) = m.iter().find(|(a, b)| *a <= p.off && p.off <= *b)?;
        let word = doc.text(p.para)[a..b].to_string();
        self.select(doc, Pos::new(p.para, a), Pos::new(p.para, b));
        let fixes = self.autocorrect.suggestions(&word, 6);
        Some((word, fixes))
    }

    pub fn learn(&mut self, word: &str) {
        self.autocorrect.learn(word);
        self.misspelled.clear();
    }

    // --- find

    /// Selects the next (or previous) match of `query` from the selection,
    /// round the end; whether there is one.
    pub fn find(&mut self, doc: &mut dyn Document, query: &str, backward: bool, case: bool) -> bool {
        if query.is_empty() {
            return false;
        }
        let norm = |s: &str| if case { s.to_string() } else { s.to_lowercase() };
        let q = norm(query);
        let (a, b) = doc.selection().range();
        let n = doc.paragraphs();
        let find_in = |t: &str, from: usize, to: usize, back: bool| -> Option<usize> {
            let hay = norm(t);
            // lowercasing changed lengths: compare as written
            let (hay, q) = if hay.len() == t.len() { (hay, q.clone()) } else { (t.to_string(), query.to_string()) };
            let (from, to) = (from.min(hay.len()), to.min(hay.len()));
            if back {
                hay[..to].rfind(&q).filter(|&i| i >= from)
            } else {
                hay.get(from..)?.find(&q).map(|i| i + from)
            }
        };
        for k in 0..=n {
            let hit = if backward {
                let p = (a.para + n - k % n) % n;
                let to = if k == 0 { a.off } else { usize::MAX };
                find_in(&doc.text(p), 0, to, true).map(|i| (p, i))
            } else {
                let p = (b.para + k) % n;
                let from = if k == 0 { b.off } else { 0 };
                find_in(&doc.text(p), from, usize::MAX, false).map(|i| (p, i))
            };
            if let Some((p, i)) = hit {
                self.select(doc, Pos::new(p, i), Pos::new(p, i + query.len()));
                return true;
            }
        }
        false
    }

    /// The selection, when it is a match, replaced; then the next match.
    pub fn replace(&mut self, doc: &mut dyn Document, query: &str, with: &str, case: bool) -> bool {
        let sel = self.selected_text(doc);
        if !sel.is_empty() && (if case { sel == query } else { sel.to_lowercase() == query.to_lowercase() }) {
            doc.checkpoint();
            let (a, b) = doc.selection().range();
            let end = doc.replace(a, b, with, false);
            doc.set_selection(Selection::caret(end));
        }
        self.find(doc, query, false, case)
    }

    /// Every match replaced, one undo step; how many.
    pub fn replace_all(&mut self, doc: &mut dyn Document, query: &str, with: &str, case: bool) -> usize {
        if query.is_empty() {
            return 0;
        }
        let mut count = 0;
        for p in 0..doc.paragraphs() {
            let text = doc.text(p);
            let hay = if case { text.clone() } else { text.to_lowercase() };
            if hay.len() != text.len() {
                continue;
            }
            let q = if case { query.to_string() } else { query.to_lowercase() };
            let hits: Vec<usize> = hay.match_indices(&q).map(|(i, _)| i).collect();
            for &i in hits.iter().rev() {
                if count == 0 {
                    doc.checkpoint();
                }
                doc.replace(Pos::new(p, i), Pos::new(p, i + q.len()), with, false);
                count += 1;
            }
        }
        self.after_move();
        count
    }

    // --- what to draw

    /// Whether the caret shows now (steady for a moment after it moves).
    pub fn caret_on(&self, time: f64) -> bool {
        let since = time - self.moved_at;
        since < BLINK || ((since / BLINK) as i64) % 2 == 0
    }

    /// The selection, caret, handles, loupe, menu, marked text and
    /// misspellings, at a time (the caret blinks), over paragraphs `visible`.
    pub fn overlay(&mut self, doc: &dyn Document, time: f64, visible: std::ops::Range<usize>) -> Overlay {
        let s = doc.selection();
        let (a, b) = s.range();
        let mut o = Overlay { scale: self.scale, ..Default::default() };
        if !s.is_caret() {
            o.selection = doc.selection_rects(a, b);
        } else if self.focused && self.caret_on(time) && self.marked.is_none() {
            o.caret = Some(doc.caret_rect(s.head, s.upstream));
        }
        // where a dragged selection would be dropped
        if let Some((_, Some(to))) = self.moving {
            o.caret = Some(doc.caret_rect(to, false));
        }
        if let Some((ma, mb)) = self.marked {
            o.marked = doc.selection_rects(ma, mb);
        }
        if self.spellcheck {
            for para in visible.start..visible.end.min(doc.paragraphs()) {
                for (x, y) in self.misspelled(doc, para) {
                    // not the word being typed
                    if s.is_caret() && s.head.para == para && s.head.off >= x && s.head.off <= y {
                        continue;
                    }
                    // under the text: by the caret's box where it is in the
                    // line's (a line made tall by a picture has its text low)
                    let caret = doc.caret_rect(Pos::new(para, x), false);
                    for (k, r) in doc.selection_rects(Pos::new(para, x), Pos::new(para, y)).into_iter().enumerate() {
                        let inside = caret.y >= r.y - 0.5 && caret.y + caret.height <= r.y + r.height + 0.5;
                        let under = if k == 0 && inside { caret.y + caret.height * 0.86 } else { r.y + r.height * 0.86 };
                        o.squiggles.push((r.x, r.x + r.width, under));
                    }
                }
            }
        }
        gesture::overlay(self, doc, &mut o);
        o
    }
}

/// The text between two places, paragraphs a newline apart.
pub fn text_between(doc: &dyn Document, a: Pos, b: Pos) -> String {
    let mut s = String::new();
    for p in a.para..=b.para {
        let t = doc.text(p);
        let (x, y) = (if p == a.para { a.off.min(t.len()) } else { 0 }, if p == b.para { b.off.min(t.len()) } else { t.len() });
        s.push_str(&t[x..y.max(x)]);
        if p < b.para {
            s.push('\n');
        }
    }
    s
}
