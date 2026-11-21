//! What a keyboard has on it, where it sits and how it looks.
//!
//! A [`KeyboardConfig`] is pages of rows of [`Key`]s: the letters, the
//! digits and punctuation (`123`), and more symbols (`#+=`), as Apple's
//! keyboards have them, with rows over and under every page. A key is as
//! wide as its [`KeySize`] in units, one unit a letter key, so a short row
//! of letters sits centred under a long one as it does on an iPhone. Where
//! a row says [`Key::letters`], the language's letters go
//! ([`KeyboardLayout`](super::KeyboardLayout)), so one template serves every
//! language.
//!
//! The presets follow Apple's: [`iphone`](KeyboardConfig::iphone),
//! [`ipad`](KeyboardConfig::ipad), [`ipad_pro`](KeyboardConfig::ipad_pro)
//! (a number row, Tab and Caps Lock), the iPad's
//! [`floating`](KeyboardConfig::ipad_floating) one, and the keyboard types
//! a field asks for: [`number_pad`](KeyboardConfig::number_pad),
//! [`decimal_pad`](KeyboardConfig::decimal_pad),
//! [`phone_pad`](KeyboardConfig::phone_pad), [`email`](KeyboardConfig::email)
//! and [`url`](KeyboardConfig::url). Each is plain data, to change.
//!
//! ```
//! use threers_ui::virtual_keyboard::{Align, Key, KeyboardConfig, Placement, ReturnKey};
//!
//! // An iPhone keyboard for a search box, a row of arrows under it, kept
//! // to the right for a right thumb.
//! let config = KeyboardConfig::iphone()
//!     .return_key(ReturnKey::Search)
//!     .with_row_at_bottom(vec![
//!         Key::named("ArrowLeft", "←"),
//!         Key::named("ArrowRight", "→"),
//!     ])
//!     .placement(Placement::one_handed(Align::Right));
//! assert_eq!(config.bottom_rows.len(), 1);
//! ```

use super::context::AutoCapitalize;
use super::feedback::KeyFeedback;
use super::types::Device;
use crate::environment::Token;
use crate::Rgba;

/// What pressing a key does.
#[derive(Debug, Clone, PartialEq)]
pub enum KeyAction {
    /// Stands for the language's letters in row `n` (0 to 2): one letter
    /// key each, on the letters page; nothing on the others.
    Letters(usize),
    /// Types the text: upper case (or the key's
    /// [`shifted`](Key::shifted) text) while Shift is on. More than one
    /// character types each in turn.
    Text(String),
    /// Sends a key by its DOM name: `"Backspace"`, `"Enter"`, `"Tab"`,
    /// `"Escape"`, `"ArrowLeft"`, … or any name the app listens for.
    Named(String),
    /// Shift: once for the next letter, twice quickly for caps lock.
    Shift,
    /// Caps lock, on or off.
    CapsLock,
    /// Shows a page.
    Page(Page),
    /// The next of the keyboard's languages (🌐). Left out while there is
    /// only one.
    NextLayout,
    /// Puts the keyboard away, and the focus with it.
    Dismiss,
    /// Moves the keyboard (one-handed, docked, floating…).
    Place(Placement),
    /// Control, Alt or Meta for the next key (a terminal's ⌃C): pressed
    /// once it's held for the next key typed, pressed again let go.
    Modifier(Modifier),
    /// Calls [`VirtualKeyboard::on_action`](super::VirtualKeyboard::on_action)
    /// with this id when the finger lifts: an emoji page, a voice button, a
    /// macro.
    Action(String),
    /// Nothing: a gap the width of the key.
    None,
}

/// A page of keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Page {
    Letters,
    /// Digits and common punctuation (`123`).
    Symbols,
    /// The rest of the symbols (`#+=`).
    More,
}

/// How wide a key is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeySize {
    /// So many letter keys wide (gaps included).
    Units(f32),
    /// At least so many units, and the room the row leaves (a space bar,
    /// an iPad's Return).
    Fill(f32),
}

/// What colours a key takes from the [`KeyboardTheme`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyLook {
    /// A letter's: the lightest.
    Character,
    /// Shift, Backspace, `123`, Return…: darker.
    Function,
    /// The accent: a Go or Search Return.
    Accent,
    /// No key drawn, only its label (a number pad's Backspace).
    Clear,
}

/// A key: what it does, what it says, how wide it is.
#[derive(Debug, Clone, PartialEq)]
pub struct Key {
    pub action: KeyAction,
    /// What it shows.
    pub label: String,
    /// What a text key types, and shows, while Shift is on (upper case
    /// when `None`).
    pub shifted: Option<String>,
    /// Shows [`shifted`](Self::shifted) small over the label, as an iPad's
    /// `,` key shows `!` and a number row's `1` shows `!`.
    pub hint: bool,
    /// A line under the label, as a number pad's `2` has `ABC`.
    pub caption: Option<String>,
    /// What assistive technology calls it (the label when `None`).
    pub name: Option<String>,
    pub size: KeySize,
    pub look: KeyLook,
    /// Offered by holding the key (accented letters). Letters and common
    /// punctuation have Apple's when [`KeyboardConfig::alternates`] is on.
    pub alternates: Vec<String>,
    /// Pressed again and again while held (Backspace).
    pub repeats: bool,
    /// Its label's size, over the theme's.
    pub font_size: Option<f32>,
}

impl Key {
    pub fn new(action: KeyAction, label: &str, look: KeyLook) -> Self {
        Self {
            action,
            label: label.into(),
            shifted: None,
            hint: false,
            caption: None,
            name: None,
            size: KeySize::Units(1.0),
            look,
            alternates: vec![],
            repeats: false,
            font_size: None,
        }
    }

    /// The language's letters of row `n` (see [`KeyAction::Letters`]).
    pub fn letters(row: usize) -> Self {
        Self::new(KeyAction::Letters(row), "", KeyLook::Character)
    }

    /// A key typing `text`.
    pub fn text(text: &str) -> Self {
        Self::new(KeyAction::Text(text.into()), text, KeyLook::Character)
    }

    /// A key typing `text`, or `shifted` under Shift, showing both.
    pub fn dual(text: &str, shifted: &str) -> Self {
        let mut k = Self::text(text);
        k.shifted = Some(shifted.into());
        k.hint = true;
        k
    }

    /// A key sending the key named `key` (see [`KeyAction::Named`]); the
    /// arrows repeat while held.
    pub fn named(key: &str, label: &str) -> Self {
        let mut k = Self::new(KeyAction::Named(key.into()), label, KeyLook::Function).name(key);
        k.repeats = matches!(key, "ArrowLeft" | "ArrowRight" | "ArrowUp" | "ArrowDown");
        k
    }

    pub fn shift() -> Self {
        Self::new(KeyAction::Shift, "⇧", KeyLook::Function).name("Shift")
    }

    pub fn caps_lock() -> Self {
        Self::new(KeyAction::CapsLock, "caps lock", KeyLook::Function).name("Caps Lock")
    }

    /// Backspace, repeating while held.
    pub fn backspace() -> Self {
        let mut k = Self::named("Backspace", "⌫").name("Delete");
        k.repeats = true;
        k
    }

    /// Return; its label follows [`KeyboardConfig::return_key`].
    pub fn enter() -> Self {
        Self::named("Enter", "return").name("Return")
    }

    pub fn tab() -> Self {
        Self::named("Tab", "tab")
    }

    /// The space bar, taking the room its row leaves.
    pub fn space() -> Self {
        Self::new(KeyAction::Text(" ".into()), "space", KeyLook::Character)
            .name("Space")
            .fill(3.0)
    }

    /// A key showing `page`.
    pub fn page(page: Page, label: &str) -> Self {
        let name = match page {
            Page::Letters => "Letters",
            Page::Symbols => "Numbers and symbols",
            Page::More => "More symbols",
        };
        Self::new(KeyAction::Page(page), label, KeyLook::Function).name(name)
    }

    /// `123`: to the digits and punctuation.
    pub fn symbols() -> Self {
        Self::page(Page::Symbols, "123")
    }

    /// `ABC`: back to the letters.
    pub fn letters_page() -> Self {
        Self::page(Page::Letters, "ABC")
    }

    /// `#+=`: to the rest of the symbols.
    pub fn more_symbols() -> Self {
        Self::page(Page::More, "#+=")
    }

    /// 🌐: the next language.
    pub fn globe() -> Self {
        Self::new(KeyAction::NextLayout, "🌐", KeyLook::Function).name("Next keyboard")
    }

    /// Control held for the next key (`ctrl`).
    pub fn control() -> Self {
        Self::new(KeyAction::Modifier(Modifier::Control), "ctrl", KeyLook::Function)
    }

    /// Alt (Option) held for the next key (`alt`).
    pub fn alt() -> Self {
        Self::new(KeyAction::Modifier(Modifier::Alt), "alt", KeyLook::Function)
    }

    /// Escape (`esc`).
    pub fn escape() -> Self {
        Self::named("Escape", "esc").look(KeyLook::Function)
    }

    /// Puts the keyboard away (an iPad's ⌨).
    pub fn dismiss() -> Self {
        Self::new(KeyAction::Dismiss, "⌨", KeyLook::Function).name("Hide keyboard")
    }

    /// Moves the keyboard to `placement`.
    pub fn place(placement: Placement, label: &str, name: &str) -> Self {
        Self::new(KeyAction::Place(placement), label, KeyLook::Function).name(name)
    }

    /// A key calling [`VirtualKeyboard::on_action`](super::VirtualKeyboard::on_action)
    /// with `id`.
    pub fn action(id: &str, label: &str) -> Self {
        Self::new(KeyAction::Action(id.into()), label, KeyLook::Function)
    }

    /// A gap `units` wide.
    pub fn gap(units: f32) -> Self {
        Self::new(KeyAction::None, "", KeyLook::Clear).units(units)
    }

    /// `units` letter keys wide.
    pub fn units(mut self, units: f32) -> Self {
        self.size = KeySize::Units(units);
        self
    }

    /// At least `min` units wide, and what the row leaves.
    pub fn fill(mut self, min: f32) -> Self {
        self.size = KeySize::Fill(min);
        self
    }

    pub fn look(mut self, look: KeyLook) -> Self {
        self.look = look;
        self
    }

    pub fn label(mut self, label: &str) -> Self {
        self.label = label.into();
        self
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn shifted(mut self, text: &str) -> Self {
        self.shifted = Some(text.into());
        self
    }

    pub fn caption(mut self, caption: &str) -> Self {
        self.caption = Some(caption.into());
        self
    }

    pub fn alternates(mut self, alternates: &[&str]) -> Self {
        self.alternates = alternates.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn font_size(mut self, size: f32) -> Self {
        self.font_size = Some(size);
        self
    }

    /// Its width in units, and whether it takes what the row leaves.
    pub(crate) fn units_and_fill(&self) -> (f32, bool) {
        match self.size {
            KeySize::Units(u) => (u.max(0.0), false),
            KeySize::Fill(u) => (u.max(0.0), true),
        }
    }
}

/// A key for each character of `text`.
pub fn keys(text: &str) -> Vec<Key> {
    text.chars().map(|c| Key::text(&c.to_string())).collect()
}

/// What the Return key says, as a field's `returnKeyType` has it: Go,
/// Search, Send, Join and Done in the accent.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ReturnKey {
    #[default]
    Return,
    Go,
    Search,
    Send,
    Next,
    Done,
    Join,
    Continue,
    /// Your own words, in the function keys' colour.
    Label(String),
}

impl ReturnKey {
    pub fn label(&self) -> String {
        match self {
            ReturnKey::Return => "return",
            ReturnKey::Go => "go",
            ReturnKey::Search => "search",
            ReturnKey::Send => "send",
            ReturnKey::Next => "next",
            ReturnKey::Done => "done",
            ReturnKey::Join => "join",
            ReturnKey::Continue => "continue",
            ReturnKey::Label(s) => s,
        }
        .to_owned()
    }

    /// Whether it is in the accent, as Apple's are.
    pub fn accent(&self) -> bool {
        matches!(
            self,
            ReturnKey::Go | ReturnKey::Search | ReturnKey::Send | ReturnKey::Join | ReturnKey::Done
        )
    }
}

/// A side, or the middle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Align {
    Left,
    #[default]
    Center,
    Right,
}

/// Where the keyboard sits.
///
/// - [`docked`](Self::docked): across the bottom (up to a width), the page
///   above made shorter to make room.
/// - [`one_handed`](Self::one_handed): docked, its keys pushed to one side
///   for that thumb, a strip on the other to bring it back, as an iPhone's.
/// - [`anchored`](Self::anchored): docked at a width, to one side or in the
///   middle (a kiosk's).
/// - [`floating`](Self::floating): a small keyboard over the page, moved by
///   the handle under it, as an iPad's.
/// - [`split`](Self::split): its rows parted in the middle, each half at its
///   side, for two thumbs on a tablet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    pub align: Align,
    /// Its width; the window's when `None`.
    pub width: Option<f32>,
    pub max_width: Option<f32>,
    /// Keys to the side of `align`, a strip on the other.
    pub one_handed: bool,
    /// Over the page, not under it; moved by its handle.
    pub floating: bool,
    /// From the window's edges, floating.
    pub margin: f32,
    /// The parting between a split keyboard's halves, in pixels (0 for
    /// none).
    pub split: f32,
}

impl Default for Placement {
    fn default() -> Self {
        Self::docked()
    }
}

impl Placement {
    /// Across the bottom.
    pub fn docked() -> Self {
        Self {
            align: Align::Center,
            width: None,
            max_width: None,
            one_handed: false,
            floating: false,
            margin: 0.0,
            split: 0.0,
        }
    }

    /// Keys to the `side`, for one thumb.
    pub fn one_handed(side: Align) -> Self {
        Self {
            align: side,
            one_handed: true,
            ..Self::docked()
        }
    }

    /// Docked at `width`, to the side or in the middle.
    pub fn anchored(align: Align, width: f32) -> Self {
        Self {
            align,
            width: Some(width),
            ..Self::docked()
        }
    }

    /// Docked at `width` in the middle.
    pub fn centered(width: f32) -> Self {
        Self::anchored(Align::Center, width)
    }

    /// Over the page at `width`, starting at the bottom in the middle.
    pub fn floating(width: f32) -> Self {
        Self {
            width: Some(width),
            floating: true,
            margin: 12.0,
            ..Self::docked()
        }
    }

    /// Parted `gap` pixels in the middle.
    pub fn split(mut self, gap: f32) -> Self {
        self.split = gap.max(0.0);
        self
    }

    pub fn max_width(mut self, width: f32) -> Self {
        self.max_width = Some(width);
        self
    }
}

/// A colour: a theme token (light and dark as the UI's theme has them),
/// a fixed colour, or one for each appearance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeyColor {
    Token(Token),
    Fixed(Rgba),
    Adaptive { light: Rgba, dark: Rgba },
}

/// How the keyboard looks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyboardTheme {
    /// Behind the keys.
    pub background: KeyColor,
    pub key: KeyColor,
    pub key_text: KeyColor,
    pub function_key: KeyColor,
    pub function_text: KeyColor,
    pub accent_key: KeyColor,
    pub accent_text: KeyColor,
    /// Shift while it is on.
    pub lit_key: KeyColor,
    pub lit_text: KeyColor,
    /// A line under each key, 1 px down.
    pub shadow: Option<KeyColor>,
    pub key_radius: f32,
    /// The keyboard's corners (a floating one's at least 14).
    pub radius: f32,
    /// A one-character label's size.
    pub letter_size: f32,
    /// A word's (`return`, `space`, `123`).
    pub label_size: f32,
    /// A hint's and a caption's.
    pub hint_size: f32,
}

fn hex(v: u32) -> Rgba {
    let c = |s: u32| ((v >> s) & 0xff) as f32 / 255.0;
    [c(16), c(8), c(0), 1.0]
}

impl KeyboardTheme {
    /// The UI's own theme tokens.
    pub fn system() -> Self {
        Self {
            background: KeyColor::Token(Token::Separator),
            key: KeyColor::Token(Token::Raised),
            key_text: KeyColor::Token(Token::Text),
            function_key: KeyColor::Token(Token::Border),
            function_text: KeyColor::Token(Token::Text),
            accent_key: KeyColor::Token(Token::Accent),
            accent_text: KeyColor::Token(Token::OnAccent),
            lit_key: KeyColor::Token(Token::Accent),
            lit_text: KeyColor::Token(Token::OnAccent),
            shadow: Some(KeyColor::Fixed([0.0, 0.0, 0.0, 0.35])),
            key_radius: 6.0,
            radius: 14.0,
            letter_size: 18.0,
            label_size: 13.0,
            hint_size: 11.0,
        }
    }

    /// Apple's: white letter keys and grey function keys on light grey,
    /// greys on near-black in dark; Shift lit white.
    pub fn apple() -> Self {
        let both = |light: u32, dark: u32| KeyColor::Adaptive {
            light: hex(light),
            dark: hex(dark),
        };
        Self {
            background: both(0xd1d4da, 0x2b2b2d),
            key: both(0xffffff, 0x6b6b6d),
            key_text: both(0x000000, 0xffffff),
            function_key: both(0xabb0ba, 0x464648),
            function_text: both(0x000000, 0xffffff),
            accent_key: both(0x007aff, 0x0a84ff),
            accent_text: KeyColor::Fixed(hex(0xffffff)),
            lit_key: both(0xffffff, 0xf2f2f2),
            lit_text: KeyColor::Fixed(hex(0x000000)),
            shadow: Some(KeyColor::Adaptive {
                light: [0.53, 0.54, 0.55, 1.0],
                dark: [0.0, 0.0, 0.0, 0.6],
            }),
            key_radius: 8.0,
            radius: 0.0,
            letter_size: 22.0,
            label_size: 16.0,
            hint_size: 12.0,
        }
    }
}

/// Everything a keyboard is: its pages, where it sits, how it looks.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyboardConfig {
    /// The letters page, row by row.
    pub letters: Vec<Vec<Key>>,
    /// The digits and punctuation page (`123`).
    pub symbols: Vec<Vec<Key>>,
    /// The rest of the symbols (`#+=`).
    pub more: Vec<Vec<Key>>,
    /// Over every page (a number row).
    pub top_rows: Vec<Vec<Key>>,
    /// Under every page (arrows, a row of shortcuts).
    pub bottom_rows: Vec<Vec<Key>>,
    pub placement: Placement,
    pub theme: KeyboardTheme,
    /// A key's height, in pixels.
    pub key_height: f32,
    /// Between keys in a row.
    pub gap: f32,
    /// Between rows.
    pub row_gap: f32,
    /// Around the keys: top, right, bottom, left.
    pub padding: [f32; 4],
    /// A bar over the keys saying which field it types into, with Done.
    pub bar: bool,
    /// How presses show, sound and feel.
    pub feedback: KeyFeedback,
    /// Holding a letter offers its accented forms.
    pub alternates: bool,
    pub return_key: ReturnKey,
    /// The page it opens on.
    pub start_page: Page,
    /// When Shift comes on by itself (needs
    /// [`set_context`](super::VirtualKeyboard::set_context)).
    pub auto_capitalize: AutoCapitalize,
    /// Two spaces after a word type “. ”.
    pub period_shortcut: bool,
    /// Back to the letters after a space typed on a symbols page.
    pub return_to_letters: bool,
    /// Holding the space bar makes the keyboard a trackpad: sliding moves
    /// the caret.
    pub space_trackpad: bool,
    /// Every key that types repeats while it's held (a terminal's: held
    /// letters, digits and space go on, as on a hardware keyboard), not
    /// only Backspace and the arrows; holding offers no alternates then.
    pub repeat_typing: bool,
    /// How long a repeating key is held before it repeats (seconds)…
    pub repeat_delay: f64,
    /// … and how often it repeats then.
    pub repeat_interval: f64,
}

/// A modifier a key holds for the next one ([`KeyAction::Modifier`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modifier {
    Control,
    Alt,
    Meta,
}

impl Default for KeyboardConfig {
    fn default() -> Self {
        Self::iphone()
    }
}

impl KeyboardConfig {
    /// The keyboard of `device`: an iPhone's, an iPad's or an iPad Pro's
    /// (`Auto` taken as a phone; the keyboard itself resolves it by the
    /// window).
    pub fn for_device(device: Device) -> Self {
        match device {
            Device::Phone | Device::Auto => Self::iphone(),
            Device::Tablet => Self::ipad(),
            Device::LargeTablet => Self::ipad_pro(),
        }
    }

    /// With this feedback.
    pub fn feedback(mut self, feedback: KeyFeedback) -> Self {
        self.feedback = feedback;
        self
    }

    /// With key clicks on or off.
    pub fn sound(mut self, on: bool) -> Self {
        self.feedback.sound = on;
        self
    }

    /// With taps on or off.
    pub fn haptics(mut self, on: bool) -> Self {
        self.feedback.haptics = on;
        self
    }

    pub fn auto_capitalize(mut self, mode: AutoCapitalize) -> Self {
        self.auto_capitalize = mode;
        self
    }
}

/// An iPhone's bottom row: `page`, 🌐, space, return.
fn phone_bottom(page: Key) -> Vec<Key> {
    vec![
        page.units(1.25),
        Key::globe().units(1.25),
        Key::space(),
        Key::enter().units(2.5),
    ]
}

/// A terminal phone's third row on its symbol pages: `page`, five wide
/// keys of a shell's punctuation, Backspace.
fn shell_punctuation(page: Key, chars: &str) -> Vec<Key> {
    let mut row = vec![page.units(1.3), Key::gap(0.2)];
    row.extend(keys(chars).into_iter().map(|k| k.units(1.4)));
    row.extend([Key::gap(0.2), Key::backspace().units(1.3)]);
    row
}

/// An iPhone's third row on the symbol pages: `page`, five wide
/// punctuation keys, Backspace.
fn phone_punctuation(page: Key) -> Vec<Key> {
    let mut row = vec![page.units(1.3), Key::gap(0.2)];
    row.extend(keys(".,?!'").into_iter().map(|k| k.units(1.4)));
    row.extend([Key::gap(0.2), Key::backspace().units(1.3)]);
    row
}

/// A number pad's row: digits with their letters under them.
fn pad_row(digits: [(&str, &str); 3]) -> Vec<Key> {
    digits
        .iter()
        .map(|(d, c)| {
            let k = Key::text(d);
            if c.is_empty() {
                k
            } else {
                k.caption(c)
            }
        })
        .collect()
}

impl KeyboardConfig {
    /// An iPhone's, portrait: ten letters across, Shift and Backspace by
    /// the third row, `123`, 🌐, space and return under them; the digits
    /// and `#+=` pages as iOS has them; the pressed letter shown large.
    /// A terminal's, for a phone: an iPhone's letters, with a row over
    /// every page of what a shell needs that a phone's keyboard hides
    /// (esc, tab, ctrl, alt, the arrows, `| ~ / -`); symbol pages of a
    /// shell's punctuation (`\ | & $ " '`, `_ = * < >`, `` ` ``) rather
    /// than prose's; every key repeating while held; no capitals at the
    /// start of a sentence, no ". " for two spaces. A tablet's:
    /// [`terminal_tablet`](Self::terminal_tablet).
    pub fn terminal() -> Self {
        let mut c = Self::iphone();
        c.top_rows = vec![vec![
            Key::escape(),
            Key::tab().look(KeyLook::Function),
            Key::control(),
            Key::alt(),
            Key::named("ArrowLeft", "←").look(KeyLook::Function),
            Key::named("ArrowUp", "↑").look(KeyLook::Function),
            Key::named("ArrowDown", "↓").look(KeyLook::Function),
            Key::named("ArrowRight", "→").look(KeyLook::Function),
            Key::text("|"),
            Key::text("~"),
            Key::text("/"),
            Key::text("-"),
        ]];
        // the whole width, edge to edge (a phone's is capped for prose)
        c.placement = Placement::docked();
        c.symbols = vec![
            keys("1234567890"),
            keys("-/\\:;|&$\"'"),
            shell_punctuation(Key::more_symbols(), "._=*?"),
            phone_bottom(Key::letters_page()),
        ];
        c.more = vec![
            keys("[]{}()<>#%"),
            keys("`~^+@!,'\"\\"),
            shell_punctuation(Key::symbols(), "._=*;"),
            phone_bottom(Key::letters_page()),
        ];
        c.terminal_behaviour()
    }

    /// A terminal's, for a tablet: a laptop's keys across the whole width.
    /// The number row with its shifted symbols, tab before `q`, ctrl where
    /// caps lock is (as on a Unix keyboard), `[ ] \ ; ' , . /` in their
    /// places, and esc, alt, the arrows along the bottom: everything a
    /// shell needs on the one page. Every key repeats while held.
    pub fn terminal_tablet() -> Self {
        let mut c = Self::ipad_pro();
        let number_row = [
            ("`", "~"), ("1", "!"), ("2", "@"), ("3", "#"), ("4", "$"), ("5", "%"), ("6", "^"),
            ("7", "&"), ("8", "*"), ("9", "("), ("0", ")"), ("-", "_"), ("=", "+"),
        ];
        let mut numbers: Vec<Key> = number_row.iter().map(|(a, b)| Key::dual(a, b)).collect();
        numbers.push(Key::backspace().fill(1.5));
        let arrow = |name: &str, label: &str| Key::named(name, label).look(KeyLook::Function);
        c.letters = vec![
            numbers,
            vec![
                Key::tab().units(1.5),
                Key::letters(0),
                Key::dual("[", "{"),
                Key::dual("]", "}"),
                Key::dual("\\", "|"),
            ],
            vec![
                Key::control().units(1.75),
                Key::letters(1),
                Key::dual(";", ":"),
                Key::dual("'", "\""),
                Key::enter().fill(1.75),
            ],
            vec![
                Key::shift().units(2.25),
                Key::letters(2),
                Key::dual(",", "<"),
                Key::dual(".", ">"),
                Key::dual("/", "?"),
                Key::shift().fill(2.25),
            ],
            vec![
                Key::escape().units(1.25),
                Key::alt().units(1.25),
                Key::symbols().label(".?123").units(1.5),
                Key::space(),
                arrow("ArrowLeft", "←"),
                arrow("ArrowUp", "↑"),
                arrow("ArrowDown", "↓"),
                arrow("ArrowRight", "→"),
            ],
        ];
        // its symbols page: the same keys' other halves, and the rest
        c.symbols = vec![
            keys("1234567890-="),
            [vec![Key::tab().units(1.5)], keys("[]{}#%^*+=\\|~")].concat(),
            [vec![Key::control().units(1.75)], keys("-/:;()$&@\"'"), vec![Key::enter().fill(1.75)]].concat(),
            [vec![Key::shift().units(2.25)], keys("._,?!`<>"), vec![Key::backspace().fill(2.25)]].concat(),
            vec![
                Key::escape().units(1.25),
                Key::alt().units(1.25),
                Key::letters_page().label("ABC").units(1.5),
                Key::space(),
                arrow("ArrowLeft", "←"),
                arrow("ArrowUp", "↑"),
                arrow("ArrowDown", "↓"),
                arrow("ArrowRight", "→"),
            ],
        ];
        c.more = c.symbols.clone();
        c.top_rows = vec![];
        c.placement = Placement::docked();
        c.terminal_behaviour()
    }

    /// What makes a keyboard a terminal's: keys repeat, no accents on
    /// holding, nothing capitalised or turned into ". " for it.
    fn terminal_behaviour(mut self) -> Self {
        self.repeat_typing = true;
        self.alternates = false;
        self.auto_capitalize = AutoCapitalize::None;
        self.period_shortcut = false;
        self.return_to_letters = false;
        self
    }

    pub fn iphone() -> Self {
        let mut shift_row = vec![Key::shift().units(1.3), Key::gap(0.2), Key::letters(2)];
        shift_row.extend([Key::gap(0.2), Key::backspace().units(1.3)]);
        Self {
            letters: vec![
                vec![Key::letters(0)],
                vec![Key::letters(1)],
                shift_row,
                phone_bottom(Key::symbols()),
            ],
            symbols: vec![
                keys("1234567890"),
                keys("-/:;()$&@\""),
                phone_punctuation(Key::more_symbols()),
                phone_bottom(Key::letters_page()),
            ],
            more: vec![
                keys("[]{}#%^*+="),
                keys("_\\|~<>€£¥•"),
                phone_punctuation(Key::symbols()),
                phone_bottom(Key::letters_page()),
            ],
            top_rows: vec![],
            bottom_rows: vec![],
            placement: Placement::docked().max_width(760.0),
            theme: KeyboardTheme::apple(),
            key_height: 43.0,
            gap: 6.0,
            row_gap: 11.0,
            padding: [8.0, 3.0, 4.0, 3.0],
            bar: true,
            feedback: KeyFeedback::default(),
            alternates: true,
            return_key: ReturnKey::Return,
            start_page: Page::Letters,
            auto_capitalize: AutoCapitalize::Sentences,
            period_shortcut: true,
            return_to_letters: true,
            space_trackpad: true,
            repeat_typing: false,
            repeat_delay: 0.45,
            repeat_interval: 0.09,
        }
    }

    /// An iPad's: Backspace ending the first row and Return the second,
    /// Shift at both ends of the third with `,!` and `.?`, then
    /// `.?123`, 🌐, space, `.?123` and ⌨ to put it away. Bigger keys, no
    /// preview, no bar.
    pub fn ipad() -> Self {
        let bottom = |page: fn() -> Key, label: &str| {
            vec![
                page().label(label).units(1.5),
                Key::globe(),
                Key::space(),
                page().label(label).units(1.5),
                Key::dismiss(),
            ]
        };
        let mut theme = KeyboardTheme::apple();
        theme.letter_size = 24.0;
        theme.label_size = 17.0;
        theme.key_radius = 7.0;
        Self {
            letters: vec![
                vec![Key::letters(0), Key::backspace().units(1.5)],
                vec![Key::gap(0.5), Key::letters(1), Key::enter().fill(2.0)],
                vec![
                    Key::shift(),
                    Key::letters(2),
                    Key::dual(",", "!"),
                    Key::dual(".", "?"),
                    Key::shift().fill(1.5),
                ],
                bottom(Key::symbols, ".?123"),
            ],
            symbols: vec![
                [keys("1234567890"), vec![Key::backspace().units(1.5)]].concat(),
                [
                    vec![Key::gap(0.5)],
                    keys("@#$&*()'\""),
                    vec![Key::enter().fill(2.0)],
                ]
                .concat(),
                [
                    vec![Key::more_symbols()],
                    keys("%-+=/;:,."),
                    vec![Key::more_symbols().fill(1.5)],
                ]
                .concat(),
                bottom(Key::letters_page, "ABC"),
            ],
            more: vec![
                [keys("€£¥_^[]{}`"), vec![Key::backspace().units(1.5)]].concat(),
                [
                    vec![Key::gap(0.5)],
                    keys("§|~…\\<>!?"),
                    vec![Key::enter().fill(2.0)],
                ]
                .concat(),
                [
                    vec![Key::symbols()],
                    keys("¡¿•°«»‹›¢"),
                    vec![Key::symbols().fill(1.5)],
                ]
                .concat(),
                bottom(Key::letters_page, "ABC"),
            ],
            top_rows: vec![],
            bottom_rows: vec![],
            placement: Placement::docked(),
            theme,
            key_height: 56.0,
            gap: 11.0,
            row_gap: 9.0,
            padding: [8.0, 5.0, 6.0, 5.0],
            bar: false,
            feedback: KeyFeedback {
                preview: false,
                ..KeyFeedback::default()
            },
            alternates: true,
            return_key: ReturnKey::Return,
            start_page: Page::Letters,
            auto_capitalize: AutoCapitalize::Sentences,
            period_shortcut: true,
            return_to_letters: true,
            space_trackpad: true,
            repeat_typing: false,
            repeat_delay: 0.45,
            repeat_interval: 0.09,
        }
    }

    /// The larger iPad Pro's: a number row (`!` over `1`…), Tab, Caps
    /// Lock, and `[ ] \ ; ' , . /` beside the letters, as a hardware
    /// keyboard has them.
    pub fn ipad_pro() -> Self {
        let number_row = [
            ("`", "~"),
            ("1", "!"),
            ("2", "@"),
            ("3", "#"),
            ("4", "$"),
            ("5", "%"),
            ("6", "^"),
            ("7", "&"),
            ("8", "*"),
            ("9", "("),
            ("0", ")"),
            ("-", "_"),
            ("=", "+"),
        ];
        let mut numbers: Vec<Key> = number_row.iter().map(|(a, b)| Key::dual(a, b)).collect();
        numbers.push(Key::backspace().fill(1.5));
        let bottom = |page: fn() -> Key, label: &str| {
            vec![
                Key::globe(),
                page().label(label).units(1.5),
                Key::space(),
                page().label(label).units(1.5),
                Key::dismiss(),
            ]
        };
        let mut c = Self::ipad();
        c.letters = vec![
            numbers.clone(),
            vec![
                Key::tab().units(1.5),
                Key::letters(0),
                Key::dual("[", "{"),
                Key::dual("]", "}"),
                Key::dual("\\", "|"),
            ],
            vec![
                Key::caps_lock().units(1.75),
                Key::letters(1),
                Key::dual(";", ":"),
                Key::dual("'", "\""),
                Key::enter().fill(1.75),
            ],
            vec![
                Key::shift().units(2.25),
                Key::letters(2),
                Key::dual(",", "<"),
                Key::dual(".", ">"),
                Key::dual("/", "?"),
                Key::shift().fill(2.25),
            ],
            bottom(Key::symbols, ".?123"),
        ];
        c.symbols = vec![
            numbers,
            [vec![Key::tab().units(1.5)], keys("[]{}#%^*+=\\|~")].concat(),
            [
                vec![Key::caps_lock().units(1.75)],
                keys("-/:;()$&@\"'"),
                vec![Key::enter().fill(1.75)],
            ]
            .concat(),
            [
                vec![Key::more_symbols().units(2.25)],
                keys(".,?!_…•€£¥"),
                vec![Key::more_symbols().fill(2.25)],
            ]
            .concat(),
            bottom(Key::letters_page, "ABC"),
        ];
        c.key_height = 52.0;
        c.theme.letter_size = 22.0;
        c.theme.label_size = 15.0;
        c
    }

    /// An iPad's floating keyboard: an iPhone's, 320 pixels wide, over the
    /// page, moved by its handle.
    pub fn ipad_floating() -> Self {
        let mut c = Self::iphone();
        c.placement = Placement::floating(320.0);
        c.key_height = 38.0;
        c.gap = 5.0;
        c.row_gap = 9.0;
        c.theme.letter_size = 19.0;
        c.theme.label_size = 14.0;
        c.bar = false;
        c
    }

    /// Digits, with their letters as a phone's have them, and Backspace:
    /// no Return, so the bar's Done puts it away.
    pub fn number_pad() -> Self {
        Self::pad(Key::gap(1.0))
    }

    /// A number pad with a decimal point.
    pub fn decimal_pad() -> Self {
        Self::pad(Key::text(".").look(KeyLook::Clear))
    }

    /// A phone's dial pad: `+*#` for `*`, `#`, `+` and the rest.
    pub fn phone_pad() -> Self {
        let mut c = Self::pad(Key::page(Page::Symbols, "+*#"));
        c.symbols = vec![
            keys("*#+"),
            keys("(),"),
            keys(";/-"),
            vec![
                Key::page(Page::Letters, "123").name("Digits"),
                Key::text("0"),
                Key::backspace().look(KeyLook::Clear),
            ],
        ];
        c
    }

    fn pad(corner: Key) -> Self {
        let mut theme = KeyboardTheme::apple();
        theme.letter_size = 26.0;
        theme.hint_size = 10.0;
        let digits = vec![
            pad_row([("1", ""), ("2", "ABC"), ("3", "DEF")]),
            pad_row([("4", "GHI"), ("5", "JKL"), ("6", "MNO")]),
            pad_row([("7", "PQRS"), ("8", "TUV"), ("9", "WXYZ")]),
            vec![
                corner,
                Key::text("0"),
                Key::backspace().look(KeyLook::Clear),
            ],
        ];
        Self {
            letters: digits.clone(),
            symbols: digits.clone(),
            more: digits,
            top_rows: vec![],
            bottom_rows: vec![],
            placement: Placement::docked().max_width(480.0),
            theme,
            key_height: 46.0,
            gap: 6.0,
            row_gap: 7.0,
            padding: [6.0, 6.0, 4.0, 6.0],
            bar: true,
            feedback: KeyFeedback {
                preview: false,
                ..KeyFeedback::default()
            },
            alternates: false,
            return_key: ReturnKey::Done,
            start_page: Page::Letters,
            auto_capitalize: AutoCapitalize::None,
            period_shortcut: false,
            return_to_letters: false,
            space_trackpad: false,
            repeat_typing: false,
            repeat_delay: 0.45,
            repeat_interval: 0.09,
        }
    }

    /// For an email address: `@` and `.` beside a shorter space bar.
    pub fn email() -> Self {
        let bottom = |page: Key| {
            vec![
                page.units(1.25),
                Key::globe().units(1.25),
                Key::space().fill(2.0),
                Key::text("@").units(1.25),
                Key::text(".").units(1.25),
                Key::enter().units(2.0),
            ]
        };
        Self::iphone().with_bottom_rows(bottom)
    }

    /// For a web address: `.`, `/` and `.com` where the space bar was.
    pub fn url() -> Self {
        let bottom = |page: Key| {
            vec![
                page.units(1.25),
                Key::globe().units(1.25),
                Key::text(".").units(1.5),
                Key::text("/").units(1.5),
                Key::text(".com").units(2.0),
                Key::enter().units(2.5),
            ]
        };
        Self::iphone()
            .with_bottom_rows(bottom)
            .return_key(ReturnKey::Go)
    }

    /// Each page's last row made by `row` from that page's page key.
    fn with_bottom_rows(mut self, row: impl Fn(Key) -> Vec<Key>) -> Self {
        for (page, key) in [
            (&mut self.letters, Key::symbols()),
            (&mut self.symbols, Key::letters_page()),
            (&mut self.more, Key::letters_page()),
        ] {
            if let Some(last) = page.last_mut() {
                *last = row(key);
            }
        }
        self
    }

    pub fn placement(mut self, placement: Placement) -> Self {
        self.placement = placement;
        self
    }

    pub fn theme(mut self, theme: KeyboardTheme) -> Self {
        self.theme = theme;
        self
    }

    pub fn return_key(mut self, key: ReturnKey) -> Self {
        self.return_key = key;
        self
    }

    pub fn key_height(mut self, height: f32) -> Self {
        self.key_height = height;
        self
    }

    pub fn bar(mut self, on: bool) -> Self {
        self.bar = on;
        self
    }

    /// With a row over every page.
    pub fn with_row_on_top(mut self, row: Vec<Key>) -> Self {
        self.top_rows.push(row);
        self
    }

    /// With a row under every page.
    pub fn with_row_at_bottom(mut self, row: Vec<Key>) -> Self {
        self.bottom_rows.push(row);
        self
    }

    /// With digits over every page, their symbols under Shift (`!` over
    /// `1`), as an iPad Pro's.
    pub fn with_number_row(self) -> Self {
        let row = "1!2@3#4$5%6^7&8*9(0)"
            .chars()
            .collect::<Vec<_>>()
            .chunks(2)
            .map(|p| Key::dual(&p[0].to_string(), &p[1].to_string()))
            .collect();
        self.with_row_on_top(row)
    }
}
