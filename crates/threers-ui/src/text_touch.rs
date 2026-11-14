//! Editing text with a finger, as iOS's text fields do:
//!
//! - a tap puts the caret where it lands; on the caret, it shows the edit
//!   menu;
//! - two taps select the word, three everything; the second tap held and
//!   dragged extends the selection word by word;
//! - a finger held still (or pressed firmly, where the screen reports
//!   force) takes the caret, which then follows it until it lifts and
//!   shows the menu; a finger dragged straight away moves the caret too;
//! - a selection has two handles (a bar with a knob), each dragging its end;
//! - the edit menu offers, as iOS's does: over a selection, cut, copy,
//!   paste, select all, look up, translate, share, and changing case; at a
//!   caret, paste, select and select all; over text that is only read
//!   ([`TouchSelection::read_only`]), copy, select all, look up, translate
//!   and share. Then the app's own actions
//!   ([`add_action`](TouchSelection::add_action)). It comes in pages a
//!   chevron turns ([`menu_page`](TouchSelection::menu_page)).
//!
//! The keyboard's own cursor control (its space bar held, or a firm press
//! on it) moves the platform input's caret, which the app copies into the
//! field.
//!
//! The field owns its text, selection and composition (a
//! [`TextField`]), and the platform's text input only delivers what is typed
//! into it; everything shown and touched is the app's. [`TouchSelection`]
//! is the touch half: the app gives it the field's finger events (with ways
//! to find a character from a point and a caret's box from a character), and
//! draws the [`handles`](TouchSelection::handles) and
//! [`menu`](TouchSelection::menu) it reports. [`TouchSelection::apply`]
//! carries out a menu choice; the clipboard is the app's to reach.
//!
//! ```
//! use threers_ui::input::TouchPhase;
//! use threers_ui::text_touch::{EditAction, TouchSelection};
//! use threers_ui::{Rect, TextField};
//!
//! let mut field = TextField::new("one two three");
//! let mut touch = TouchSelection::new();
//! // 10 px a character, one 20 px line.
//! let mut at = |x: f32, _y: f32| ((x / 10.0).round().max(0.0) as usize).min(13);
//! let mut caret = |i: usize| Rect { x: i as f32 * 10.0, y: 0.0, width: 1.0, height: 20.0 };
//! // Two taps on "two" select it, with the menu up.
//! for t in [0.0, 0.2] {
//!     touch.touch(&mut field, TouchPhase::Start, 45.0, 10.0, t, &mut at, &mut caret);
//!     touch.touch(&mut field, TouchPhase::End, 45.0, 10.0, t + 0.05, &mut at, &mut caret);
//! }
//! assert_eq!(field.selection(), (4, 7));
//! assert_eq!(touch.menu(&field).unwrap()[1], EditAction::Copy);
//! let copied = touch.apply(&mut field, EditAction::Copy);
//! assert_eq!(copied, Some(threers_ui::text_touch::Request::Copy("two".into())));
//! ```

use crate::input::TouchPhase;
use crate::text_field::TextField;
use crate::text_hit::word_at;
use crate::tree::Rect;

/// How touch editing behaves: its gestures, their timing and reach, and
/// the edit menu's built-in actions. The default is iOS's.
#[derive(Debug, Clone, PartialEq)]
pub struct TouchConfig {
    /// How far a finger may move and still tap, in pixels.
    pub tap_slop: f32,
    /// Taps closer in time than this, in seconds, on about the same place,
    /// count together.
    pub multi_tap: f64,
    /// A finger held this long, in seconds, without moving takes the caret.
    pub long_press: f64,
    /// A press this firm (of 1, where the screen measures force), and
    /// `firm_rise` firmer than it came down, selects the word under it.
    /// Firmer than it came down: some screens report one middling force for
    /// every touch.
    pub firm_press: f32,
    pub firm_rise: f32,
    /// How near a handle's knob a finger must come down to take it, and how
    /// far beside its bar.
    pub handle_reach: f32,
    pub bar_reach: f32,
    /// A handle's knob: its diameter, and how far it stands off the line.
    pub knob: f32,
    /// Two taps select the word (and the second, held, drags by words).
    pub double_tap: bool,
    /// Three taps select everything.
    pub triple_tap: bool,
    /// A finger dragged on the text carries the caret.
    pub drag_caret: bool,
    /// A finger held still takes the caret.
    pub hold_caret: bool,
    /// A firm press selects the word.
    pub firm_press_selects: bool,
    /// The edit menu's built-in actions, in order: over a selection, at a
    /// bare caret, and over text that is only read.
    pub selection_actions: Vec<EditAction>,
    pub caret_actions: Vec<EditAction>,
    pub read_actions: Vec<EditAction>,
}

impl Default for TouchConfig {
    fn default() -> Self {
        use EditAction::*;
        Self {
            tap_slop: 10.0,
            multi_tap: 0.35,
            long_press: 0.45,
            firm_press: 0.5,
            firm_rise: 0.3,
            handle_reach: 24.0,
            bar_reach: 14.0,
            knob: 10.0,
            double_tap: true,
            triple_tap: true,
            drag_caret: true,
            hold_caret: true,
            firm_press_selects: true,
            selection_actions: vec![
                Cut, Copy, Paste, SelectAll, LookUp, Translate, Share, Uppercase, Lowercase,
                Capitalize,
            ],
            caret_actions: vec![Paste, Select, SelectAll],
            read_actions: vec![Copy, SelectAll, LookUp, Translate, Share],
        }
    }
}

/// An edit menu's choices: iOS's, and the app's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditAction {
    Cut,
    Copy,
    Paste,
    /// Select the word at the caret.
    Select,
    SelectAll,
    /// Look the selection up (a dictionary, the web).
    LookUp,
    Translate,
    /// Share the selection (a phone's share sheet).
    Share,
    /// The selection in capitals, small letters, or each word capitalised.
    Uppercase,
    Lowercase,
    Capitalize,
    /// One of the app's own (see [`TouchSelection::add_action`]), which
    /// [`TouchSelection::apply`] leaves to the app.
    Custom(usize),
}

impl EditAction {
    /// The label of a built-in action (an app's own is
    /// [`TouchSelection::label`]'s).
    pub fn label(self) -> &'static str {
        match self {
            EditAction::Cut => "Cut",
            EditAction::Copy => "Copy",
            EditAction::Paste => "Paste",
            EditAction::Select => "Select",
            EditAction::SelectAll => "Select All",
            EditAction::LookUp => "Look Up",
            EditAction::Translate => "Translate",
            EditAction::Share => "Share…",
            EditAction::Uppercase => "Uppercase",
            EditAction::Lowercase => "Lowercase",
            EditAction::Capitalize => "Capitalize",
            EditAction::Custom(_) => "",
        }
    }
}

/// When an app's own action is in the edit menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shows {
    /// Over a selection.
    WithSelection,
    /// At a bare caret.
    AtCaret,
    Always,
}

/// One page of the edit menu: its actions, and whether a chevron leads back
/// to the page before or on to more.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MenuPage {
    pub actions: Vec<EditAction>,
    pub back: bool,
    pub more: bool,
}

/// What a menu choice asks of the platform, which the text cannot do
/// itself: the clipboard, a dictionary, a translator, a share sheet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    /// Put this on the clipboard (cut or copied).
    Copy(String),
    /// Read the clipboard and [`TextField::insert`] it.
    Paste,
    /// Look this up.
    LookUp(String),
    /// Translate this.
    Translate(String),
    /// Share this.
    Share(String),
}

/// A selection's two handles: the caret box at each end, and the knob a
/// finger drags it by (above the start, below the end).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Handles {
    pub start: Rect,
    pub end: Rect,
    pub start_knob: (f32, f32),
    pub end_knob: (f32, f32),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Press {
    /// On the text: a tap, a long press, or a drag moving the caret.
    /// `words` is the word a second tap came down on: held and dragged, it
    /// selects by words from there. `long` is set once the press took the
    /// caret (held) or selected (pressed firmly), so lifting it is no tap.
    Text {
        x: f32,
        y: f32,
        time: f64,
        index: usize,
        moved: bool,
        long: bool,
        words: Option<(usize, usize)>,
        base_force: Option<f32>,
    },
    /// Dragging a handle; the finger's offset from the line's middle.
    Handle { dy: f32 },
}

/// Touch editing for one text field. See the [module docs](self).
#[derive(Debug, Clone, Default)]
pub struct TouchSelection {
    press: Option<Press>,
    /// The last tap: when, at which character, and how many in a row.
    last_tap: Option<(f64, usize, u32)>,
    menu: bool,
    /// The page of the menu showing.
    page: usize,
    /// The app's own actions: their labels, and when they show.
    custom: Vec<(String, Shows)>,
    /// Text that is read, not edited: the menu offers copying, not cutting
    /// or pasting.
    read_only: bool,
    config: TouchConfig,
}

impl TouchSelection {
    pub fn new() -> Self {
        Self::default()
    }

    /// Touch editing that behaves as `config` says.
    pub fn with_config(config: TouchConfig) -> Self {
        Self {
            config,
            ..Self::default()
        }
    }

    pub fn config(&self) -> &TouchConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut TouchConfig {
        &mut self.config
    }

    /// Adds an action of the app's own to the edit menu, after the built-in
    /// ones: `label`, shown `shows`. Returns it, for the app to recognise
    /// when it is chosen.
    pub fn add_action(&mut self, label: impl Into<String>, shows: Shows) -> EditAction {
        self.custom.push((label.into(), shows));
        EditAction::Custom(self.custom.len() - 1)
    }

    /// For text that is read, not edited (a page's selectable text): the
    /// menu offers Copy and Select All (and the app's actions), not Cut,
    /// Paste or Select.
    pub fn read_only() -> Self {
        Self {
            read_only: true,
            ..Self::default()
        }
    }

    /// This, for text that is only read (see [`read_only`](Self::read_only)).
    pub fn into_read_only(self) -> Self {
        Self {
            read_only: true,
            ..self
        }
    }

    /// Selects the word at character `index` and shows the menu, as a long
    /// press on a page's text does.
    pub fn select_word(&mut self, field: &mut TextField, index: usize) {
        let chars: Vec<char> = field.text.chars().collect();
        if chars.is_empty() {
            return;
        }
        let (a, b) = word_at(&chars, index.min(chars.len() - 1));
        field.anchor = a;
        field.caret = b;
        self.menu = true;
        self.page = 0;
    }

    /// An action's label.
    pub fn label(&self, action: EditAction) -> &str {
        match action {
            EditAction::Custom(i) => self.custom.get(i).map_or("", |c| c.0.as_str()),
            a => a.label(),
        }
    }

    /// The page of the edit menu showing, `per_page` actions to a page (a
    /// chevron takes one of their places where there is one).
    pub fn menu_page(&self, field: &TextField, per_page: usize) -> Option<MenuPage> {
        let all = self.menu(field)?;
        let per_page = per_page.max(1);
        let pages = all.len().div_ceil(per_page).max(1);
        let page = self.page.min(pages - 1);
        Some(MenuPage {
            actions: all
                .iter()
                .copied()
                .skip(page * per_page)
                .take(per_page)
                .collect(),
            back: page > 0,
            more: page + 1 < pages,
        })
    }

    /// A chevron tapped: the menu's next page, or the one before.
    pub fn turn_page(&mut self, forward: bool) {
        self.page = if forward {
            self.page + 1
        } else {
            self.page.saturating_sub(1)
        };
    }

    /// Whether a finger coming down at `(x, y)` is this field's: on the
    /// text (`inside`, the field's box, says so) or on a handle's knob.
    pub fn wants(
        &self,
        field: &TextField,
        x: f32,
        y: f32,
        inside: bool,
        caret_at: &mut dyn FnMut(usize) -> Rect,
    ) -> bool {
        inside || self.handle_at(field, x, y, caret_at).is_some()
    }

    /// The handle whose knob is under `(x, y)`: `false` the start, `true`
    /// the end.
    fn handle_at(
        &self,
        field: &TextField,
        x: f32,
        y: f32,
        caret_at: &mut dyn FnMut(usize) -> Rect,
    ) -> Option<bool> {
        let h = self.handles(field, caret_at)?;
        let d = |(kx, ky): (f32, f32)| (kx - x).hypot(ky - y);
        // Near the knob, or by the bar on the knob's half of the line: a
        // fingertip lands a little off what it aims at, and the knob is
        // small. Across the middle of the line, the touch is the text's (a
        // tap on a short selected word is near both handles).
        let (reach, bar, knob) = (
            self.config.handle_reach,
            self.config.bar_reach,
            self.config.knob,
        );
        let by_bar = |r: Rect, below: bool| {
            let on_knob_side = if below {
                y > r.y + r.height * 0.6
            } else {
                y < r.y + r.height * 0.4
            };
            (x - r.x).abs() <= bar
                && on_knob_side
                && y >= r.y - knob - reach
                && y <= r.y + r.height + knob + reach
        };
        let start = (d(h.start_knob) <= reach && y < h.start.y + h.start.height / 2.0)
            || by_bar(h.start, false);
        let end =
            (d(h.end_knob) <= reach && y > h.end.y + h.end.height / 2.0) || by_bar(h.end, true);
        match (start, end) {
            (true, true) => Some(d(h.end_knob) < d(h.start_knob)),
            (true, false) => Some(false),
            (false, true) => Some(true),
            (false, false) => None,
        }
    }

    /// A finger event on the field (the app routes to it every touch that
    /// [`wants`](Self::wants) claimed at its start). `index_at` finds the
    /// character nearest a point, `caret_at` the caret's box before a
    /// character. Returns whether the field changed.
    #[allow(clippy::too_many_arguments)]
    pub fn touch(
        &mut self,
        field: &mut TextField,
        phase: TouchPhase,
        x: f32,
        y: f32,
        time: f64,
        index_at: &mut dyn FnMut(f32, f32) -> usize,
        caret_at: &mut dyn FnMut(usize) -> Rect,
    ) -> bool {
        let before = (field.anchor, field.caret, field.text.len());
        match phase {
            TouchPhase::Start => {
                if let Some(end) = self.handle_at(field, x, y, caret_at) {
                    // The other end stays put; this one follows the finger.
                    let (lo, hi) = field.selection();
                    let (fixed, moving) = if end { (lo, hi) } else { (hi, lo) };
                    field.anchor = fixed;
                    field.caret = moving;
                    let r = caret_at(moving);
                    self.press = Some(Press::Handle {
                        dy: y - (r.y + r.height / 2.0),
                    });
                    self.menu = false;
                } else {
                    let index = index_at(x, y);
                    // The second of two taps: held and dragged, it selects
                    // by words.
                    let words = match self.last_tap {
                        Some((t, i, 1))
                            if self.config.double_tap
                                && time - t < self.config.multi_tap
                                && i.abs_diff(index) <= 1 =>
                        {
                            let chars: Vec<char> = field.text.chars().collect();
                            (!chars.is_empty()).then(|| word_at(&chars, index.min(chars.len() - 1)))
                        }
                        _ => None,
                    };
                    self.press = Some(Press::Text {
                        x,
                        y,
                        time,
                        index,
                        moved: false,
                        long: false,
                        words,
                        base_force: None,
                    });
                }
            }
            TouchPhase::Move => match &mut self.press {
                Some(Press::Handle { dy }) => {
                    let i = index_at(x, y - *dy);
                    // Never past the other end: a handle dragged over it
                    // stops there.
                    field.caret = if field.caret >= field.anchor {
                        i.max(field.anchor)
                    } else {
                        i.min(field.anchor)
                    };
                }
                Some(Press::Text {
                    x: x0,
                    y: y0,
                    moved,
                    long,
                    words,
                    ..
                }) if *moved || *long || (x - *x0).hypot(y - *y0) > self.config.tap_slop => {
                    *moved = true;
                    self.menu = false;
                    let i = index_at(x, y);
                    if let Some((a, b)) = *words {
                        // From the word tapped twice, by whole words.
                        let chars: Vec<char> = field.text.chars().collect();
                        let (c, d) = if chars.is_empty() {
                            (0, 0)
                        } else {
                            word_at(&chars, i.min(chars.len() - 1))
                        };
                        if i < a {
                            field.anchor = b;
                            field.caret = c;
                        } else {
                            field.anchor = a;
                            field.caret = d.max(b);
                        }
                    } else if self.config.drag_caret || *long {
                        // The caret follows the finger.
                        field.anchor = i;
                        field.caret = i;
                    }
                }
                _ => {}
            },
            TouchPhase::End => {
                match self.press.take() {
                    Some(Press::Handle { .. }) => self.menu = field.anchor != field.caret,
                    // The caret was held and carried, or a word was pressed
                    // firmly, or words dragged: the menu, where it ended.
                    Some(Press::Text { long: true, .. })
                    | Some(Press::Text {
                        words: Some(_),
                        moved: true,
                        ..
                    }) => {
                        self.menu = true;
                        self.last_tap = None;
                    }
                    Some(Press::Text { moved: true, .. }) => self.last_tap = None,
                    Some(Press::Text {
                        index,
                        moved: false,
                        long: false,
                        ..
                    }) => {
                        let count = match self.last_tap {
                            Some((t, i, n))
                                if time - t < self.config.multi_tap && i.abs_diff(index) <= 1 =>
                            {
                                n + 1
                            }
                            _ => 1,
                        };
                        let most = if !self.config.double_tap {
                            1
                        } else if !self.config.triple_tap {
                            2
                        } else {
                            3
                        };
                        let count = if count > most { 1 } else { count };
                        self.last_tap = Some((time, index, count));
                        let chars: Vec<char> = field.text.chars().collect();
                        match count {
                            1 => {
                                let on_caret = field.anchor == field.caret && field.caret == index;
                                field.anchor = index;
                                field.caret = index;
                                // A tap on the caret shows the menu, as on
                                // a phone; anywhere else hides it.
                                self.menu = on_caret && !self.menu;
                            }
                            2 => {
                                let (a, b) =
                                    word_at(&chars, index.min(chars.len().saturating_sub(1)));
                                field.anchor = a;
                                field.caret = b;
                                self.menu = true;
                            }
                            _ => {
                                field.select_all();
                                self.menu = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
            TouchPhase::Cancel => self.press = None,
        }
        if phase == TouchPhase::End {
            // A menu shown anew starts at its first page.
            self.page = 0;
        }
        (field.anchor, field.caret, field.text.len()) != before
    }

    /// Time passing with a finger down: held still long enough, it takes
    /// the caret, which goes where the finger is and then follows it.
    /// Returns whether the field changed.
    pub fn tick(&mut self, field: &mut TextField, time: f64) -> bool {
        if let Some(Press::Text {
            time: t0,
            index,
            moved: false,
            long,
            words: None,
            ..
        }) = &mut self.press
            && !*long
            && self.config.hold_caret
            && time - *t0 >= self.config.long_press
        {
            *long = true;
            self.menu = false;
            let changed = (field.anchor, field.caret) != (*index, *index);
            field.anchor = *index;
            field.caret = *index;
            return changed;
        }
        false
    }

    /// How firmly the finger is pressing, 0 to 1 (a touch's `force`, where
    /// the screen measures it). Pressed firmly before moving, it selects the
    /// word under it. Returns whether the field changed.
    pub fn force(&mut self, field: &mut TextField, force: f32) -> bool {
        let Some(Press::Text {
            index,
            moved: false,
            long,
            base_force,
            ..
        }) = &mut self.press
        else {
            return false;
        };
        let base = *base_force.get_or_insert(force);
        let c = &self.config;
        if c.firm_press_selects && !*long && force >= c.firm_press && force - base >= c.firm_rise {
            *long = true;
            let chars: Vec<char> = field.text.chars().collect();
            if chars.is_empty() {
                return false;
            }
            let (a, b) = word_at(&chars, (*index).min(chars.len() - 1));
            field.anchor = a;
            field.caret = b;
            return true;
        }
        false
    }

    /// A tap the field did not get as a touch (the one that gave it focus),
    /// at character `index`: the next tap there counts as the second.
    pub fn note_tap(&mut self, time: f64, index: usize) {
        self.last_tap = Some((time, index, 1));
    }

    /// Whether a frame is wanted to notice a long press.
    pub fn needs_frame(&self) -> bool {
        self.config.hold_caret
            && matches!(
                self.press,
                Some(Press::Text {
                    moved: false,
                    long: false,
                    words: None,
                    ..
                })
            )
    }

    /// The selection's handles, while there is a selection.
    pub fn handles(
        &self,
        field: &TextField,
        caret_at: &mut dyn FnMut(usize) -> Rect,
    ) -> Option<Handles> {
        let (lo, hi) = field.selection();
        if lo == hi {
            return None;
        }
        let (start, end) = (caret_at(lo), caret_at(hi));
        Some(Handles {
            start,
            end,
            start_knob: (start.x, start.y - self.config.knob / 2.0),
            end_knob: (end.x, end.y + end.height + self.config.knob / 2.0),
        })
    }

    /// The edit menu's choices, while it shows.
    pub fn menu(&self, field: &TextField) -> Option<Vec<EditAction>> {
        if !self.menu || self.press.is_some() {
            return None;
        }
        let selection = field.anchor != field.caret;
        let c = &self.config;
        let mut all = if self.read_only {
            c.read_actions.clone()
        } else if selection {
            c.selection_actions.clone()
        } else {
            c.caret_actions.clone()
        };
        all.extend(
            self.custom
                .iter()
                .enumerate()
                .filter_map(|(i, (_, shows))| {
                    let shown = match shows {
                        Shows::WithSelection => selection,
                        Shows::AtCaret => !selection,
                        Shows::Always => true,
                    };
                    shown.then_some(EditAction::Custom(i))
                }),
        );
        Some(all)
    }

    /// Hides the menu (the field was typed in, or lost focus).
    pub fn dismiss(&mut self) {
        self.menu = false;
        self.press = None;
        self.page = 0;
    }

    /// Carries out a menu choice on `field`. Returns what the platform must
    /// do, if anything (the clipboard, looking up, translating, sharing).
    pub fn apply(&mut self, field: &mut TextField, action: EditAction) -> Option<Request> {
        self.page = 0;
        let (lo, hi) = field.selection();
        let selected: String = field.text.chars().skip(lo).take(hi - lo).collect();
        match action {
            EditAction::Cut => {
                self.menu = false;
                field.insert("");
                Some(Request::Copy(selected))
            }
            EditAction::Copy => {
                self.menu = false;
                Some(Request::Copy(selected))
            }
            EditAction::Paste => {
                self.menu = false;
                Some(Request::Paste)
            }
            EditAction::LookUp => {
                self.menu = false;
                Some(Request::LookUp(selected))
            }
            EditAction::Translate => {
                self.menu = false;
                Some(Request::Translate(selected))
            }
            EditAction::Share => {
                self.menu = false;
                Some(Request::Share(selected))
            }
            EditAction::Uppercase | EditAction::Lowercase | EditAction::Capitalize => {
                self.menu = false;
                let text: String = match action {
                    EditAction::Uppercase => selected.to_uppercase(),
                    EditAction::Lowercase => selected.to_lowercase(),
                    _ => {
                        let mut start = true;
                        selected
                            .chars()
                            .flat_map(|c| {
                                let up = start;
                                start = !c.is_alphanumeric();
                                if up {
                                    c.to_uppercase().collect::<Vec<_>>()
                                } else {
                                    vec![c]
                                }
                            })
                            .collect()
                    }
                };
                // Rewritten, and still selected.
                field.insert(&text);
                field.anchor = lo;
                field.caret = lo + text.chars().count();
                None
            }
            EditAction::Select => {
                let chars: Vec<char> = field.text.chars().collect();
                if !chars.is_empty() {
                    let (a, b) = word_at(&chars, field.caret.min(chars.len() - 1));
                    field.anchor = a;
                    field.caret = b;
                }
                None
            }
            EditAction::SelectAll => {
                field.select_all();
                None
            }
            // The app's to carry out.
            EditAction::Custom(_) => {
                self.menu = false;
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 10 px a character on one 20 px line.
    fn geometry(n: usize) -> (impl FnMut(f32, f32) -> usize, impl FnMut(usize) -> Rect) {
        (
            move |x: f32, _y: f32| ((x / 10.0).round().max(0.0) as usize).min(n),
            |i: usize| Rect {
                x: i as f32 * 10.0,
                y: 0.0,
                width: 1.0,
                height: 20.0,
            },
        )
    }

    fn tap(t: &mut TouchSelection, f: &mut TextField, x: f32, time: f64) {
        let (mut at, mut caret) = geometry(f.len());
        t.touch(f, TouchPhase::Start, x, 10.0, time, &mut at, &mut caret);
        t.touch(
            f,
            TouchPhase::End,
            x,
            10.0,
            time + 0.05,
            &mut at,
            &mut caret,
        );
    }

    #[test]
    fn taps_place_the_caret_select_words_and_all() {
        let mut f = TextField::new("one two three");
        let mut t = TouchSelection::new();
        tap(&mut t, &mut f, 52.0, 0.0);
        assert_eq!((f.anchor, f.caret), (5, 5));
        assert!(t.menu(&f).is_none());
        // On the caret: the menu, for a bare caret.
        tap(&mut t, &mut f, 50.0, 1.0);
        assert_eq!(
            t.menu(&f).unwrap(),
            [EditAction::Paste, EditAction::Select, EditAction::SelectAll]
        );
        // Three in a row: the word, then everything.
        tap(&mut t, &mut f, 52.0, 2.0);
        tap(&mut t, &mut f, 52.0, 2.2);
        assert_eq!(f.selection(), (4, 7));
        tap(&mut t, &mut f, 52.0, 2.4);
        assert_eq!(f.selection(), (0, 13));
        assert_eq!(t.menu(&f).unwrap()[0], EditAction::Cut);
    }

    #[test]
    fn a_held_finger_takes_the_caret_and_a_firm_press_selects() {
        let mut f = TextField::new("one two three");
        let mut t = TouchSelection::new();
        let (mut at, mut caret) = geometry(f.len());
        // Held: the caret comes to the finger, follows it, and the menu
        // shows where it is let go.
        t.touch(
            &mut f,
            TouchPhase::Start,
            95.0,
            10.0,
            0.0,
            &mut at,
            &mut caret,
        );
        assert!(t.needs_frame());
        assert!(!t.tick(&mut f, 0.2));
        assert!(t.tick(&mut f, 0.5));
        assert_eq!((f.anchor, f.caret), (10, 10));
        t.touch(
            &mut f,
            TouchPhase::Move,
            98.0,
            10.0,
            0.6,
            &mut at,
            &mut caret,
        );
        t.touch(
            &mut f,
            TouchPhase::Move,
            31.0,
            11.0,
            0.7,
            &mut at,
            &mut caret,
        );
        assert_eq!(
            (f.anchor, f.caret),
            (3, 3),
            "even a small move after holding"
        );
        t.touch(
            &mut f,
            TouchPhase::End,
            31.0,
            11.0,
            0.8,
            &mut at,
            &mut caret,
        );
        assert_eq!(t.menu(&f).unwrap()[0], EditAction::Paste);

        // Dragged straight away: the caret too, and no menu.
        t.touch(
            &mut f,
            TouchPhase::Start,
            10.0,
            10.0,
            2.0,
            &mut at,
            &mut caret,
        );
        t.touch(
            &mut f,
            TouchPhase::Move,
            40.0,
            10.0,
            2.1,
            &mut at,
            &mut caret,
        );
        t.touch(
            &mut f,
            TouchPhase::Move,
            62.0,
            12.0,
            2.2,
            &mut at,
            &mut caret,
        );
        assert_eq!((f.anchor, f.caret), (6, 6));
        t.touch(
            &mut f,
            TouchPhase::End,
            62.0,
            12.0,
            2.3,
            &mut at,
            &mut caret,
        );
        assert!(t.menu(&f).is_none());
        assert_eq!((f.anchor, f.caret), (6, 6), "a drag is not a tap");

        // Pressed firmly: the word. (A screen reporting the same middling
        // force throughout is no firm press.)
        t.touch(
            &mut f,
            TouchPhase::Start,
            52.0,
            10.0,
            3.0,
            &mut at,
            &mut caret,
        );
        assert!(!t.force(&mut f, 0.5) && !t.force(&mut f, 0.5));
        t.touch(
            &mut f,
            TouchPhase::End,
            52.0,
            10.0,
            3.05,
            &mut at,
            &mut caret,
        );
        assert_eq!(f.selection(), (5, 5), "a tap");
        t.touch(
            &mut f,
            TouchPhase::Start,
            52.0,
            10.0,
            4.0,
            &mut at,
            &mut caret,
        );
        assert!(!t.force(&mut f, 0.2));
        assert!(t.force(&mut f, 0.7));
        assert_eq!(f.selection(), (4, 7));
        t.touch(
            &mut f,
            TouchPhase::End,
            52.0,
            10.0,
            4.1,
            &mut at,
            &mut caret,
        );
        assert_eq!(f.selection(), (4, 7), "lifting it is no tap");
        assert_eq!(t.menu(&f).unwrap()[0], EditAction::Cut);
    }

    #[test]
    fn a_second_tap_held_and_dragged_selects_by_words() {
        let mut f = TextField::new("one two three four");
        let mut t = TouchSelection::new();
        let (mut at, mut caret) = geometry(f.len());
        tap(&mut t, &mut f, 52.0, 0.0);
        t.touch(
            &mut f,
            TouchPhase::Start,
            52.0,
            10.0,
            0.15,
            &mut at,
            &mut caret,
        );
        assert!(!t.needs_frame(), "no long press on the second tap");
        t.touch(
            &mut f,
            TouchPhase::Move,
            100.0,
            10.0,
            0.25,
            &mut at,
            &mut caret,
        );
        assert_eq!(f.selection(), (4, 13), "two through three, whole");
        t.touch(
            &mut f,
            TouchPhase::Move,
            15.0,
            10.0,
            0.35,
            &mut at,
            &mut caret,
        );
        assert_eq!(f.selection(), (0, 7), "back over one: one through two");
        t.touch(
            &mut f,
            TouchPhase::End,
            15.0,
            10.0,
            0.4,
            &mut at,
            &mut caret,
        );
        assert!(t.menu(&f).is_some());
    }

    #[test]
    fn handles_drag_the_ends_and_stop_at_each_other() {
        let mut f = TextField::new("one two three");
        f.anchor = 4;
        f.caret = 7;
        let mut t = TouchSelection::new();
        let (mut at, mut caret) = geometry(f.len());
        let h = t.handles(&f, &mut caret).unwrap();
        assert_eq!((h.start.x, h.end.x), (40.0, 70.0));
        assert!(
            t.wants(&f, 70.0, 28.0, false, &mut caret),
            "the end's knob, under the line"
        );
        // A fingertip landing a little high, by the bar's lower half, still
        // takes it; the middle of the line is the text's.
        assert_eq!(t.handle_at(&f, 76.0, 13.0, &mut caret), Some(true));
        assert_eq!(t.handle_at(&f, 36.0, 6.0, &mut caret), Some(false));
        assert_eq!(t.handle_at(&f, 55.0, 10.0, &mut caret), None);
        assert!(!t.wants(&f, 140.0, 80.0, false, &mut caret));
        // The end handle out to the end of the text.
        t.touch(
            &mut f,
            TouchPhase::Start,
            70.0,
            26.0,
            0.0,
            &mut at,
            &mut caret,
        );
        t.touch(
            &mut f,
            TouchPhase::Move,
            128.0,
            26.0,
            0.1,
            &mut at,
            &mut caret,
        );
        assert_eq!(f.selection(), (4, 13));
        // Back past the start: it stops at the start.
        t.touch(
            &mut f,
            TouchPhase::Move,
            10.0,
            26.0,
            0.2,
            &mut at,
            &mut caret,
        );
        assert_eq!(f.selection(), (4, 4));
        t.touch(
            &mut f,
            TouchPhase::Move,
            90.0,
            26.0,
            0.3,
            &mut at,
            &mut caret,
        );
        t.touch(
            &mut f,
            TouchPhase::End,
            90.0,
            26.0,
            0.4,
            &mut at,
            &mut caret,
        );
        assert_eq!(f.selection(), (4, 9));
        assert!(t.menu(&f).is_some());
        // The start handle, above the line.
        t.touch(
            &mut f,
            TouchPhase::Start,
            40.0,
            -6.0,
            1.0,
            &mut at,
            &mut caret,
        );
        t.touch(
            &mut f,
            TouchPhase::Move,
            0.0,
            -6.0,
            1.1,
            &mut at,
            &mut caret,
        );
        t.touch(&mut f, TouchPhase::End, 0.0, -6.0, 1.2, &mut at, &mut caret);
        assert_eq!(f.selection(), (0, 9));
    }

    #[test]
    fn cut_copy_and_paste_edit_the_field() {
        let mut f = TextField::new("one two three");
        f.anchor = 4;
        f.caret = 8;
        let mut t = TouchSelection::new();
        assert_eq!(
            t.apply(&mut f, EditAction::Cut),
            Some(Request::Copy("two ".into()))
        );
        assert_eq!(f.text, "one three");
        assert_eq!(t.apply(&mut f, EditAction::Paste), Some(Request::Paste));
        f.insert("two ");
        assert_eq!(f.text, "one two three");
        f.caret = 1;
        f.anchor = 1;
        assert_eq!(t.apply(&mut f, EditAction::Select), None);
        assert_eq!(f.selection(), (0, 3));
    }

    #[test]
    fn the_apps_actions_follow_the_built_in_ones_in_pages() {
        let mut f = TextField::new("one two three");
        let mut t = TouchSelection::new();
        let stamp = t.add_action("Stamp", Shows::WithSelection);
        let date = t.add_action("Insert Date", Shows::AtCaret);
        assert_eq!(t.label(stamp), "Stamp");
        // Over "two": iOS's ten, then the app's one; four to a page.
        tap(&mut t, &mut f, 52.0, 0.0);
        tap(&mut t, &mut f, 52.0, 0.2);
        let p = t.menu_page(&f, 4).unwrap();
        assert_eq!(
            (p.actions.clone(), p.back, p.more),
            (
                vec![
                    EditAction::Cut,
                    EditAction::Copy,
                    EditAction::Paste,
                    EditAction::SelectAll
                ],
                false,
                true
            )
        );
        t.turn_page(true);
        let p = t.menu_page(&f, 4).unwrap();
        assert_eq!(
            (p.actions.clone(), p.back, p.more),
            (
                vec![
                    EditAction::LookUp,
                    EditAction::Translate,
                    EditAction::Share,
                    EditAction::Uppercase
                ],
                true,
                true
            )
        );
        t.turn_page(true);
        let p = t.menu_page(&f, 4).unwrap();
        assert_eq!(
            (p.actions.clone(), p.back, p.more),
            (
                vec![EditAction::Lowercase, EditAction::Capitalize, stamp],
                true,
                false
            )
        );
        t.turn_page(false);
        assert!(t.menu_page(&f, 4).unwrap().back);
        // An app's action is the app's: the menu closes, nothing else.
        assert_eq!(t.apply(&mut f, stamp), None);
        assert!(t.menu(&f).is_none());
        assert_eq!(f.selection(), (4, 7));
        // The platform's: a request; the case: done, still selected.
        assert_eq!(
            t.apply(&mut f, EditAction::LookUp),
            Some(Request::LookUp("two".into()))
        );
        assert_eq!(t.apply(&mut f, EditAction::Capitalize), None);
        assert_eq!((f.text.as_str(), f.selection()), ("one Two three", (4, 7)));
        assert_eq!(t.apply(&mut f, EditAction::Uppercase), None);
        assert_eq!(f.text, "one TWO three");
        // Text only read: copying, not cutting or pasting.
        let r = TouchSelection::read_only();
        let mut g = TextField::new("read me");
        let mut r = r;
        r.select_word(&mut g, 1);
        assert_eq!(
            r.menu(&g).unwrap(),
            [
                EditAction::Copy,
                EditAction::SelectAll,
                EditAction::LookUp,
                EditAction::Translate,
                EditAction::Share
            ]
        );
        // At a caret, the caret's.
        tap(&mut t, &mut f, 120.0, 2.0);
        tap(&mut t, &mut f, 120.0, 3.0);
        assert_eq!(t.menu(&f).unwrap().last(), Some(&date));
    }

    #[test]
    fn the_gestures_and_the_menu_follow_the_config() {
        let mut f = TextField::new("one two three");
        let config = TouchConfig {
            double_tap: false,
            hold_caret: false,
            selection_actions: vec![EditAction::Copy],
            ..TouchConfig::default()
        };
        let mut t = TouchSelection::with_config(config);
        tap(&mut t, &mut f, 52.0, 0.0);
        tap(&mut t, &mut f, 52.0, 0.2);
        assert_eq!(f.selection(), (5, 5), "two taps are two taps");
        let (mut at, mut caret) = geometry(f.len());
        t.touch(
            &mut f,
            TouchPhase::Start,
            95.0,
            10.0,
            1.0,
            &mut at,
            &mut caret,
        );
        assert!(!t.needs_frame());
        assert!(!t.tick(&mut f, 2.0), "no hold");
        t.touch(
            &mut f,
            TouchPhase::End,
            95.0,
            10.0,
            2.1,
            &mut at,
            &mut caret,
        );
        f.anchor = 4;
        f.caret = 7;
        t.config_mut().selection_actions.push(EditAction::Share);
        t.touch(
            &mut f,
            TouchPhase::Start,
            70.0,
            26.0,
            3.0,
            &mut at,
            &mut caret,
        );
        t.touch(
            &mut f,
            TouchPhase::End,
            70.0,
            26.0,
            3.1,
            &mut at,
            &mut caret,
        );
        assert_eq!(t.menu(&f).unwrap(), [EditAction::Copy, EditAction::Share]);
    }
}
