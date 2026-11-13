//! An on-screen keyboard, drawn by the UI itself: for a touch screen with
//! no keyboard of its own (a kiosk, a car, a TV), or where the platform's
//! will not come up. It rises from the bottom while a text box, text area
//! or search box has focus and it is on — or for one field when that
//! field's keyboard button asks ([`VirtualKeyboard::attach_button`]) — and
//! goes when focus leaves. Its keys type into whatever has focus, as a
//! hardware keyboard's do (the same key events), and pressing them leaves
//! the focus where it is.
//!
//! It has the letters of a [`KeyboardLayout`] (English, Spanish, German,
//! French, Russian, Ukrainian and Greek built in; the first chosen by the
//! UI's [`Locale`](crate::i18n::Locale)), and a 🌐 key moves to the next of
//! those it is given; Shift (once, or twice for caps lock), a page of
//! digits and symbols, Backspace, Space and Return. A bar over the keys
//! says which field it types into, with Done to put it away.
//!
//! ```
//! use threers_ui::input::Input;
//! use threers_ui::virtual_keyboard::{KeyboardLayout, VirtualKeyboard};
//! use threers_ui::{style, FixedWidthText, Role, Semantics, TextField, Ui};
//!
//! let mut ui = Ui::new();
//! let mut input: Input<TextField> = Input::new();
//! let mut field = TextField::new("");
//! let root = ui.root();
//! let b = ui.add_box(root, style! { height: 20px; });
//! ui.set_semantics(b, Semantics::new(Role::TextBox).label("Name"));
//! input.on_key(b, |f: &mut TextField, _, key, m, _| !matches!(f.key(key, m, |_| true), threers_ui::FieldKey::Ignored));
//! let kb = VirtualKeyboard::new(&mut ui, &mut input);
//! kb.set_enabled(true);
//! kb.set_layouts(&mut ui, vec![KeyboardLayout::german(), KeyboardLayout::english()]);
//! input.set_focus(&mut ui, &mut field, Some(b));
//! input.tick(&mut ui, &mut field, 0.0);
//! ui.layout(400.0, 600.0, &mut FixedWidthText(7.0));
//! assert!(kb.showing(&ui));
//! for k in ["ü", "ß"] {
//!     kb.press(&mut ui, &mut input, &mut field, k);
//! }
//! assert_eq!(field.text, "üß");
//! assert_eq!(input.focused(), Some(b), "focus stays in the field");
//! ```

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::environment::Token;
use crate::input::{Event, Input, Phase};
use crate::transitions::Transition;
use crate::tree::{NodeId, Ui};
use crate::{style, Display, KeyModifiers, Role, Semantics};

/// The letters of a keyboard, three rows of them, and its name.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyboardLayout {
    /// Shown on the space bar and said to assistive technology.
    pub name: String,
    /// A language tag it is for (`en`, `de`, …).
    pub language: String,
    /// Its letters, row by row, lower case.
    pub rows: [String; 3],
    /// Letters with accents the three rows leave out, as a row over them
    /// (French's ê, ô, ï…); empty for none.
    pub accents: String,
}

impl KeyboardLayout {
    pub fn new(name: &str, language: &str, rows: [&str; 3]) -> Self {
        Self { name: name.into(), language: language.into(), rows: rows.map(String::from), accents: String::new() }
    }

    /// With a row of accented letters over the others.
    pub fn with_accents(mut self, accents: &str) -> Self {
        self.accents = accents.into();
        self
    }

    pub fn english() -> Self {
        Self::new("English", "en", ["qwertyuiop", "asdfghjkl", "zxcvbnm"])
    }
    pub fn spanish() -> Self {
        Self::new("Español", "es", ["qwertyuiop", "asdfghjklñ", "zxcvbnm"])
    }
    pub fn german() -> Self {
        Self::new("Deutsch", "de", ["qwertzuiopü", "asdfghjklöä", "yxcvbnmß"])
    }
    pub fn french() -> Self {
        Self::new("Français", "fr", ["azertyuiop", "qsdfghjklm", "wxcvbnéèàç"]).with_accents("êëâùûüôîïœ«»")
    }
    pub fn russian() -> Self {
        Self::new("Русский", "ru", ["йцукенгшщзх", "фывапролджэ", "ячсмитьбю"])
    }
    pub fn ukrainian() -> Self {
        Self::new("Українська", "uk", ["йцукенгшщзхї", "фівапролджє", "ячсмитьбюґ"])
    }
    pub fn greek() -> Self {
        Self::new("Ελληνικά", "el", ["ςερτυθιοπ", "ασδφγηξκλ", "ζχψωβνμ"])
    }

    /// Those built in.
    pub fn all() -> Vec<Self> {
        vec![Self::english(), Self::spanish(), Self::german(), Self::french(), Self::russian(), Self::ukrainian(), Self::greek()]
    }

    /// The one for `locale`'s language (English for one not built in).
    pub fn for_locale(locale: &str) -> Self {
        let lang = locale.split(['-', '_']).next().unwrap_or("").to_lowercase();
        Self::all().into_iter().find(|l| l.language == lang).unwrap_or_else(Self::english)
    }
}

/// The text of a text node (a key's label).
fn text_of(ui: &Ui, id: NodeId) -> String {
    match ui.kind(id) {
        crate::NodeKind::Text(t) => t.clone(),
        _ => String::new(),
    }
}

/// Digits and symbols, the keyboard's second page.
const SYMBOLS: [&str; 3] = ["1234567890", "-/:;()$&@\"", ".,?!'#%*"];

/// Shift: off, for the next letter, or locked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shift {
    Off,
    Once,
    Locked,
}

/// The keyboard's state, shared by its keys and its hook.
struct Board {
    layouts: Vec<KeyboardLayout>,
    current: usize,
    shift: Shift,
    symbols: bool,
    /// The letter keys: each key, its label's text node, and its row and
    /// place.
    letters: Vec<(NodeId, NodeId, usize, usize)>,
    /// The rows holding the letter keys (rebuilt for a new layout).
    rows: Vec<NodeId>,
    /// The layout the rows were built for, and the page.
    built: Option<(usize, bool)>,
}

/// The keyboard: built once, shown while it is on (or summoned) and a text
/// control has focus.
#[derive(Clone)]
pub struct VirtualKeyboard {
    panel: NodeId,
    enabled: Rc<Cell<bool>>,
    /// Asked for by a field's keyboard button, until focus leaves it.
    summoned: Rc<Cell<bool>>,
    board: Rc<RefCell<Board>>,
    /// Keys pressed, waiting to be typed (pressed in listeners, which
    /// cannot dispatch; the input types them after).
    queue: Rc<RefCell<Vec<String>>>,
    /// The bar's line saying which field it types into.
    field_label: NodeId,
    space_label: NodeId,
    shift_key: NodeId,
    symbols_label: NodeId,
}

/// A key's height, in pixels.
const KEY_HEIGHT: f32 = 42.0;

fn key_box(ui: &mut Ui, row: NodeId, label: &str, grow: f32) -> (NodeId, NodeId) {
    let k = ui.add_box(row, style! { display: flex; align-items: center; justify-content: center; border-radius: 6px; cursor: pointer; font-size: 18px; min-width: 0; flex-basis: 0px; box-shadow: 0 1px 0 rgba(0,0,0,0.35); });
    let s = ui.style_mut(k);
    s.flex_grow = grow;
    s.height = crate::Length::Px(KEY_HEIGHT);
    ui.token_style(k, &[("background", Token::Raised)]);
    ui.active_style(k, "opacity: 0.55");
    let t = ui.add_text(k, label);
    ui.set_semantics(k, Semantics::new(Role::Button).label(label).focusable(false));
    (k, t)
}

/// `f` when `k` is pressed, without taking the focus from the field.
fn on_press<S: 'static>(input: &mut Input<S>, k: NodeId, f: Rc<dyn Fn(&mut Ui)>) {
    input.on(k, Phase::Bubble, move |_, ui, e, cx| {
        if let Event::PointerDown { .. } = e {
            cx.prevent_default();
            f(ui);
        }
    });
}

impl VirtualKeyboard {
    pub fn new<S: 'static>(ui: &mut Ui, input: &mut Input<S>) -> Self {
        let root = ui.root();
        // Centred under a window wider than a keyboard needs, its top
        // corners rounded, over the bottom of what is behind it.
        let panel = ui.add_box(
            root,
            style! {
                position: absolute; left: 0px; right: 0px; bottom: 0px; display: none; flex-direction: column; align-items: center;
                user-select: none; pointer-events: none;
            },
        );
        ui.defer(panel, 990);
        let sheet = ui.add_box(
            panel,
            style! {
                display: flex; flex-direction: column; gap: 8px; width: 100%; max-width: 760px; padding: 8px 6px 10px 6px; box-sizing: border-box;
                border-radius: 14px; box-shadow: 0 -6px 24px rgba(0,0,0,0.35); pointer-events: auto;
            },
        );
        ui.token_style(sheet, &[("background", Token::Separator), ("color", Token::Text)]);
        ui.pad_safe_area(sheet, [false, false, true, false]);
        ui.set_semantics(sheet, Semantics::new(Role::Group).label("Keyboard"));
        // The bar: the field typed into, and Done.
        let bar = ui.add_box(sheet, style! { display: flex; align-items: center; gap: 8px; padding: 0 6px; min-height: 28px; });
        let fl = ui.add_box(bar, style! { flex-grow: 1; font-size: 12px; min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; });
        ui.token_style(fl, &[("color", Token::SecondaryText)]);
        let field_label = ui.add_text(fl, "");
        let done = ui.add_box(bar, style! { padding: 4px 12px; border-radius: 6px; font-weight: bold; cursor: pointer; min-height: 24px; box-sizing: border-box; });
        ui.token_style(done, &[("color", Token::Accent)]);
        ui.add_text(done, "Done");
        ui.set_semantics(done, Semantics::new(Role::Button).label("Done").focusable(false));
        let letters_box = ui.add_box(sheet, style! { display: flex; flex-direction: column; gap: 8px; });
        // The bottom row: Shift, 123, 🌐, space, Return, Backspace.
        let row = ui.add_box(sheet, style! { display: flex; gap: 6px; });
        let (shift_key, _) = key_box(ui, row, "⇧", 1.4);
        ui.semantics_mut(shift_key).label = Some("Shift".into());
        let (sym, symbols_label) = key_box(ui, row, "123", 1.4);
        ui.semantics_mut(sym).label = Some("Numbers and symbols".into());
        let (globe, _) = key_box(ui, row, "🌐", 1.0);
        ui.semantics_mut(globe).label = Some("Next keyboard".into());
        let (space, space_label) = key_box(ui, row, "space", 4.5);
        ui.update_style(space, |s| s.font_size = 13.0);
        ui.semantics_mut(space).label = Some("Space".into());
        let (ret, _) = key_box(ui, row, "return", 1.8);
        ui.update_style(ret, |s| s.font_size = 13.0);
        ui.semantics_mut(ret).label = Some("Return".into());
        let (back, _) = key_box(ui, row, "⌫", 1.4);
        ui.semantics_mut(back).label = Some("Delete".into());

        let queue: Rc<RefCell<Vec<String>>> = Rc::default();
        let lang = ui.environment::<crate::i18n::Locale>(root).map_or_else(|| "en".to_owned(), |l| l.0);
        let first = KeyboardLayout::for_locale(&lang);
        let mut layouts = vec![first.clone()];
        layouts.extend(KeyboardLayout::all().into_iter().filter(|l| *l != first));
        let board = Rc::new(RefCell::new(Board { layouts, current: 0, shift: Shift::Off, symbols: false, letters: vec![], rows: vec![letters_box], built: None }));
        let kb = Self {
            panel,
            enabled: Rc::default(),
            summoned: Rc::default(),
            board: board.clone(),
            queue: queue.clone(),
            field_label,
            space_label,
            shift_key,
            symbols_label,
        };
        for (k, name) in [(space, " "), (ret, "Enter"), (back, "Backspace")] {
            let q = queue.clone();
            on_press(input, k, Rc::new(move |_| q.borrow_mut().push(name.to_owned())));
        }
        {
            let b = board.clone();
            let me = kb.clone();
            on_press(
                input,
                shift_key,
                Rc::new(move |ui: &mut Ui| {
                    {
                        let mut bd = b.borrow_mut();
                        bd.shift = match bd.shift {
                            Shift::Off => Shift::Once,
                            Shift::Once => Shift::Locked,
                            Shift::Locked => Shift::Off,
                        };
                    }
                    me.relabel(ui);
                }),
            );
        }
        {
            let b = board.clone();
            on_press(input, sym, Rc::new(move |_| {
                let mut bd = b.borrow_mut();
                bd.symbols = !bd.symbols;
            }));
        }
        {
            let b = board.clone();
            on_press(input, globe, Rc::new(move |_| {
                let mut bd = b.borrow_mut();
                bd.current = (bd.current + 1) % bd.layouts.len().max(1);
                bd.symbols = false;
            }));
        }
        let hide = Rc::new(Cell::new(false));
        let h = hide.clone();
        on_press(input, done, Rc::new(move |_| h.set(true)));

        let me = kb.clone();
        let mut shown = false;
        // While it shows, the root's bottom padding is its height: the page
        // above gets shorter, as a phone's does for its keyboard, and what
        // has focus is scrolled into what is left.
        let mut own_bottom: Option<f32> = None;
        let mut reveal = false;
        input.on_after_dispatch(move |input, ui, s| {
            if !ui.contains(panel) {
                return false;
            }
            if hide.take() {
                me.summoned.set(false);
                input.set_focus(ui, s, None);
            }
            // The letter keys, for the layout and page now.
            me.build_letters(ui, input);
            // Type what was pressed, as keys, at what has focus.
            let keys: Vec<String> = me.queue.borrow_mut().drain(..).collect();
            for k in keys {
                input.dispatch(ui, s, Event::Key { key: k, modifiers: KeyModifiers::default() });
            }
            let typing = input.wants_ime(ui);
            if !typing {
                me.summoned.set(false);
            }
            let want = typing && (me.enabled.get() || me.summoned.get());
            let root = ui.root();
            if want != shown {
                shown = want;
                if want {
                    ui.update_style(panel, |st| st.display = Display::Flex);
                    ui.appear(panel, Transition::from_below(80.0), 0.2);
                    own_bottom.get_or_insert(ui.style(root).padding.bottom);
                } else {
                    ui.update_style(panel, |st| st.display = Display::None);
                    if let Some(b) = own_bottom.take() {
                        ui.update_style(root, |st| st.padding.bottom = b);
                    }
                }
            }
            if shown {
                // Which field it types into.
                let name = input
                    .focused()
                    .and_then(|f| ui.semantics(f).and_then(|s| s.label.clone()))
                    .map_or_else(String::new, |n| format!("Typing in {n}"));
                if text_of(ui, me.field_label) != name {
                    ui.set_text(me.field_label, name);
                }
                // Its height, once laid out.
                let h = ui.rect(panel).height;
                let want = own_bottom.unwrap_or(0.0) + h;
                if h > 0.0 && ui.style(root).padding.bottom != want {
                    ui.update_style(root, |st| st.padding.bottom = want);
                    reveal = true;
                } else if reveal && h > 0.0 {
                    reveal = false;
                    if let Some(f) = input.focused().filter(|&f| ui.contains(f)) {
                        ui.scroll_into_view(f);
                    }
                }
            }
            true
        });
        kb
    }

    /// The letter rows built (again) when the layout or the page changed.
    fn build_letters<S: 'static>(&self, ui: &mut Ui, input: &mut Input<S>) {
        let (want, layout) = {
            let b = self.board.borrow();
            ((b.current, b.symbols), b.layouts.get(b.current).cloned().unwrap_or_else(KeyboardLayout::english))
        };
        if self.board.borrow().built == Some(want) {
            return;
        }
        let holder = self.board.borrow().rows[0];
        for c in ui.children(holder).to_vec() {
            ui.remove(c);
        }
        let symbols = want.1;
        let mut letters = vec![];
        // The accents' row first (letters only), then the three.
        let accents = (!symbols && !layout.accents.is_empty()).then(|| layout.accents.clone());
        let pages: Vec<String> = accents
            .into_iter()
            .chain(SYMBOLS.iter().enumerate().map(|(r, sym)| if symbols { sym.to_string() } else { layout.rows[r].clone() }))
            .collect();
        for (r, page) in pages.iter().enumerate() {
            let chars: Vec<char> = page.chars().collect();
            let row = ui.add_box(holder, style! { display: flex; gap: 6px; justify-content: center; });
            for (c, ch) in chars.into_iter().enumerate() {
                let (k, t) = key_box(ui, row, &ch.to_string(), 1.0);
                letters.push((k, t, r, c));
                let (q, b, me) = (self.queue.clone(), self.board.clone(), self.clone());
                on_press(
                    input,
                    k,
                    Rc::new(move |ui: &mut Ui| {
                        let typed = text_of(ui, t);
                        if typed.is_empty() {
                            return;
                        }
                        q.borrow_mut().push(typed);
                        let once = b.borrow().shift == Shift::Once;
                        if once {
                            b.borrow_mut().shift = Shift::Off;
                            me.relabel(ui);
                        }
                    }),
                );
            }
        }
        {
            let mut b = self.board.borrow_mut();
            b.letters = letters;
            b.built = Some(want);
        }
        ui.set_text(self.space_label, if symbols { "space".to_owned() } else { layout.name.clone() });
        ui.set_text(self.symbols_label, if symbols { "ABC" } else { "123" });
        ui.semantics_mut(self.panel).label = Some(format!("Keyboard: {}", layout.name));
        self.relabel(ui);
    }

    /// The letter keys in upper or lower case, and Shift lit, as it stands.
    fn relabel(&self, ui: &mut Ui) {
        let b = self.board.borrow();
        let upper = !b.symbols && b.shift != Shift::Off;
        for &(k, t, _, _) in &b.letters {
            let now = text_of(ui, t);
            let want = if upper { now.to_uppercase() } else { now.to_lowercase() };
            if want != now {
                ui.set_text(t, want.clone());
                ui.semantics_mut(k).label = Some(want);
            }
        }
        ui.token_style(self.shift_key, &[("background", if b.shift != Shift::Off { Token::Accent } else { Token::Raised })]);
    }

    /// Turns it on or off. On, it shows while a text control has focus.
    pub fn set_enabled(&self, on: bool) {
        self.enabled.set(on);
    }

    pub fn enabled(&self) -> bool {
        self.enabled.get()
    }

    /// The layouts the 🌐 key moves through, the first showing.
    pub fn set_layouts(&self, ui: &mut Ui, layouts: Vec<KeyboardLayout>) {
        let _ = ui;
        let mut b = self.board.borrow_mut();
        if !layouts.is_empty() {
            b.layouts = layouts;
        }
        b.current = 0;
        b.built = None;
    }

    /// The layout showing.
    pub fn layout(&self) -> KeyboardLayout {
        let b = self.board.borrow();
        b.layouts.get(b.current).cloned().unwrap_or_else(KeyboardLayout::english)
    }

    /// Whether it is showing now.
    pub fn showing(&self, ui: &Ui) -> bool {
        ui.style(self.panel).display != Display::None
    }

    /// Its panel (to measure, or keep content clear of).
    pub fn panel(&self) -> NodeId {
        self.panel
    }

    /// Gives `field` (a text box, text area or search box, positioned) a
    /// keyboard button at its right end: pressed, focus goes to the field
    /// and this keyboard comes up for it, on or not, until focus leaves it.
    /// Returns the button.
    pub fn attach_button<S: 'static>(&self, ui: &mut Ui, input: &mut Input<S>, field: NodeId) -> NodeId {
        let b = ui.add_box(
            field,
            style! { position: absolute; right: 2px; top: 0px; bottom: 0px; width: 28px; display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 15px; },
        );
        ui.token_style(b, &[("color", Token::SecondaryText)]);
        ui.hover_style(b, "opacity: 0.7");
        ui.add_text(b, "⌨");
        ui.set_semantics(b, Semantics::new(Role::Button).label("Show the keyboard").focusable(false));
        // Room for it: the field's text stops short of it.
        ui.update_style(field, |s| s.padding.right = s.padding.right.max(30.0));
        let summoned = self.summoned.clone();
        input.on(b, Phase::Bubble, move |s, ui, e, cx| {
            if let Event::PointerDown { .. } = e {
                cx.prevent_default();
                cx.stop_propagation();
                summoned.set(true);
                cx.focus(field);
                let _ = (s, ui);
            }
        });
        b
    }

    /// Brings it up for the text control with focus, on or not (as a
    /// field's keyboard button does).
    pub fn summon(&self) {
        self.summoned.set(true);
    }

    /// Presses `key` (a character, `"Backspace"`, `"Enter"`, `" "`) as its
    /// key on the panel would: typed at what has focus.
    pub fn press<S: 'static>(&self, ui: &mut Ui, input: &mut Input<S>, state: &mut S, key: &str) {
        input.dispatch(ui, state, Event::Key { key: key.to_owned(), modifiers: KeyModifiers::default() });
    }
}
