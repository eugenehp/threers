//! An on-screen keyboard, drawn by the UI itself: for a touch screen with
//! no keyboard of its own (a kiosk, a car, a TV), or where the platform's
//! will not come up. It rises from the bottom while a text box, text area
//! or search box has focus and it is on — or for one field when that
//! field's keyboard button asks ([`VirtualKeyboard::attach_button`]) — and
//! goes when focus leaves. Its keys type into whatever has focus, as a
//! hardware keyboard's do (the same key events), and pressing them leaves
//! the focus where it is.
//!
//! What is on it, where it sits and how it looks is a [`KeyboardConfig`]:
//! pages of rows of [`Key`]s, a [`Placement`] (docked, one-handed, anchored
//! to a side, floating, split) and a [`KeyboardTheme`]. The presets are
//! Apple's — an iPhone's (the default), an iPad's, an iPad Pro's with its
//! number row, the iPad's floating one — and the keyboards a field asks
//! for: a number, decimal or phone pad, an email address, a web address.
//! A field can have its own ([`VirtualKeyboard::set_config_for`]): a PIN
//! box gets a number pad, the rest letters.
//!
//! It behaves as Apple's do: a letter types when the finger lifts, shown
//! large over it on a phone; holding one offers its accented forms
//! ([`alternates_for`]), chosen by sliding to one; Backspace repeats while
//! held; Shift is for one letter, tapped twice quickly it locks; 🌐 moves
//! to the next [`KeyboardLayout`] (English, Spanish, German, French,
//! Russian, Ukrainian and Greek built in; the first chosen by the UI's
//! [`Locale`](crate::i18n::Locale)) and is left out while there is one;
//! Return says Go, Search or Send when the field wants it.
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
//!
//! An iPad's, floating, for one field and an iPhone's for the rest:
//!
//! ```
//! use threers_ui::input::Input;
//! use threers_ui::virtual_keyboard::{Align, KeyboardConfig, Placement, VirtualKeyboard};
//! use threers_ui::Ui;
//!
//! let mut ui = Ui::new();
//! let mut input: Input<()> = Input::new();
//! let notes = ui.add_box(ui.root(), Default::default());
//! let kb = VirtualKeyboard::with_config(
//!     &mut ui,
//!     &mut input,
//!     KeyboardConfig::iphone().placement(Placement::one_handed(Align::Left)),
//! );
//! kb.set_config_for(notes, Some(KeyboardConfig::ipad_floating()));
//! ```

mod config;
mod context;
mod feedback;
mod layouts;
#[cfg(all(feature = "sound", not(target_arch = "wasm32")))]
mod sound;
mod spec;
#[cfg(feature = "swipe")]
mod swipe;
mod types;

pub use config::{
    keys, Align, Key, KeyAction, KeyColor, KeyLook, KeySize, KeyboardConfig, KeyboardTheme, Modifier,
    Page, Placement, ReturnKey,
};
pub use context::{current_word, period_shortcut, wants_capital, AutoCapitalize};
pub use feedback::{Haptic, KeyFeedback, KeyPress, KeySound, PressStyle};
pub use layouts::{alternates_for, KeyboardLayout};
#[cfg(all(feature = "sound", not(target_arch = "wasm32")))]
pub use sound::ClickPlayer;
pub use spec::{row, try_row, KeyBox};
#[cfg(feature = "swipe")]
pub use swipe::{Swipe, SwipeKey, SwipePoint, SWIPE_START};
pub use types::{Device, KeyboardType};

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::environment::Token;
use crate::input::{Event, EventCx, Input, Phase, TimerId};
use crate::states::Appearance;
use crate::transitions::Transition;
use crate::tree::{NodeId, Ui};
use crate::{
    style, AlignItems, BoxShadow, Display, JustifyContent, KeyModifiers, Length, Rgba, Role,
    Semantics,
};

/// The text of a text node (a key's label).
fn text_of(ui: &Ui, id: NodeId) -> String {
    match ui.kind(id) {
        crate::NodeKind::Text(t) => t.clone(),
        _ => String::new(),
    }
}

/// `text` in upper case, unless that is longer (ß is not SS on a key).
/// The letter `text` types, if it is one letter (what a swipe crosses).
#[cfg(feature = "swipe")]
fn swipe_letter(text: &str) -> Option<char> {
    let mut it = text.chars();
    let c = it.next()?;
    (it.next().is_none() && c.is_alphabetic()).then_some(c)
}

fn upper(text: &str) -> String {
    let u = text.to_uppercase();
    if u.chars().count() == text.chars().count() {
        u
    } else {
        text.to_owned()
    }
}

/// Holding a key this long offers its alternates (a repeating key's
/// timing is its config's: `repeat_delay`, `repeat_interval`).
const HOLD: f64 = 0.45;
/// Backspace held this many repeats deletes words.
const WORDS_AFTER: u32 = 12;
/// Pixels the finger moves on the space bar's trackpad per caret step.
const TRACKPAD_STEP: f32 = 12.0;
/// The most steps of one move: no finger goes further, a bad coordinate does.
const TRACKPAD_MOST: usize = 500;

/// The modifier that makes Backspace delete a word on this platform's
/// text shortcuts: ⌥ on Apple's, Ctrl on the rest.
fn word_modifier() -> KeyModifiers {
    match crate::text_shortcuts::Platform::current() {
        crate::text_shortcuts::Platform::Mac => KeyModifiers {
            alt: true,
            ..KeyModifiers::default()
        },
        crate::text_shortcuts::Platform::Other => KeyModifiers {
            control: true,
            ..KeyModifiers::default()
        },
    }
}

/// Two taps of Shift this close lock it.
const DOUBLE_TAP: f64 = 0.35;
/// The strip beside a one-handed keyboard.
const STRIP: f32 = 56.0;

/// Shift: off, for the next letter, or locked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shift {
    Off,
    Once,
    Locked,
}

/// Changes every config shown gets.
type Tweak = Rc<dyn Fn(&mut KeyboardConfig)>;
/// Told of each press.
type FeedbackListener = Rc<dyn Fn(&KeyPress)>;
/// Told of each action key's id.
type ActionListener = Rc<dyn Fn(&str, &mut Ui)>;
/// Words for the one being typed.
type Suggester = Rc<dyn Fn(&str) -> Vec<String>>;
/// Words from the language and the text before the caret.
type Predictor = Rc<dyn Fn(&str, &str) -> Vec<String>>;
/// Words a swipe may be, best first.
#[cfg(feature = "swipe")]
type SwipeDecoder = Rc<dyn Fn(&Swipe) -> Vec<String>>;
/// Reads the text before the caret from the input's state `S`.
type ContextReader<S> = Box<dyn Fn(&S, NodeId) -> Option<String>>;

/// A key on the keyboard now.
struct Built {
    cell: NodeId,
    face: NodeId,
    label: NodeId,
    key: Key,
    /// The page it is on (`None`: beside every page, as the one-handed
    /// strip's).
    page: Option<Page>,
    /// What it shows now: its label and colours, so a change of Shift
    /// touches only the keys it changes.
    shown: Option<(String, Option<KeyColor>, KeyColor)>,
    /// Its preview bubble and the bubble's text: laid out over it once and
    /// shown by its opacity, so a press is drawn without a layout pass.
    bubble: Option<(NodeId, NodeId)>,
}

/// The alternates offered by a held key.
struct Popup {
    options: Vec<String>,
    chosen: usize,
}

/// A key held down.
struct Press {
    index: usize,
    /// Typed when the finger lifts (a letter), or already acted on (Shift).
    types: bool,
    popup: Option<Popup>,
    /// Where the finger is.
    at: (f32, f32),
    /// The space bar held, the keyboard a trackpad: where the caret last
    /// stepped from.
    trackpad: Option<(f32, f32)>,
    /// A letter pressed with a swipe decoder set: the path so far, and how
    /// far it has gone.
    #[cfg(feature = "swipe")]
    path: Option<(Vec<SwipePoint>, f32)>,
    /// The path went far enough to be a swipe: no letter types, the
    /// decoder's word does.
    #[cfg(feature = "swipe")]
    swiping: bool,
}

/// Seconds of a swipe's path its trail shows.
#[cfg(feature = "swipe")]
pub const TRAIL_SECONDS: f64 = 1.2;
/// Points of a swipe's path its trail shows at most.
#[cfg(feature = "swipe")]
const TRAIL_POINTS: usize = 240;
/// Seconds a trail takes to fade once the finger lifts.
#[cfg(feature = "swipe")]
pub const TRAIL_FADE: f64 = 0.3;

/// Alternates a held key can offer at once.
const MOST_ALTERNATES: usize = 10;
/// Words offered at once.
const SUGGESTIONS: usize = 3;

/// What a listener asks of the input, done after its dispatch.
enum Request {
    Hold(usize),
    Release,
    ShiftTap,
}

/// The keyboard's state, shared by its keys and its hook.
struct Board {
    /// The config for fields without their own, when no device is chosen.
    default: KeyboardConfig,
    /// The device the configs are for (`None`: `default` as given), and
    /// what it came to for the window.
    device: Option<Device>,
    resolved: Device,
    /// Changes made to every config shown ([`VirtualKeyboard::customize`]).
    tweaks: Vec<Tweak>,
    per_field: Vec<(NodeId, FieldKeyboard)>,
    /// The config showing: the focused field's, or the default.
    active: KeyboardConfig,
    active_field: Option<NodeId>,
    /// The per-field configs changed: take up the focused field's again.
    refresh: bool,
    /// Where a key or the app moved it, over the config's.
    placement: Option<Placement>,
    layouts: Vec<KeyboardLayout>,
    current: usize,
    shift: Shift,
    /// Control, Alt, Meta pressed: held for the next key typed.
    mods: KeyModifiers,
    page: Page,
    /// Counts changes to the config or the placement.
    generation: u64,
    /// What the keys were built for.
    built: Option<(u64, usize, Page, Appearance)>,
    keys: Vec<Built>,
    press: Option<Press>,
    requests: Vec<Request>,
    timers: Vec<TimerId>,
    shift_tapped: Rc<Cell<bool>>,
    /// A floating keyboard being dragged: where the pointer was.
    drag: Option<(f32, f32)>,
    /// How far a floating keyboard was dragged.
    offset: (f32, f32),
    hide: bool,
    /// Told of each press.
    feedback: Vec<FeedbackListener>,
    /// Told of each action key.
    actions: Vec<ActionListener>,
    /// Reads the text before the caret: a [`ContextReader`] for the
    /// input's state `S`.
    context: Option<Rc<dyn Any>>,
    /// The text before the caret as last read.
    before: Option<String>,
    /// Shift came on by itself (a sentence starting), not by a tap.
    auto_shift: bool,
    /// What this keyboard typed lately, for the period shortcut without a
    /// context.
    recent: String,
    /// The focused field holds a secret: nothing of it is read, kept,
    /// predicted, swiped or previewed.
    secret: bool,
    /// Words offered for the one being typed.
    suggest: Option<Suggester>,
    /// Words offered from the text before the caret, ahead of `suggest`.
    predict: Option<Predictor>,
    /// The word they were offered for, and them.
    offered: (String, Vec<String>),
    /// The generation whose config was checked.
    checked: Option<u64>,
    /// What had focus last.
    focused: Option<NodeId>,
    /// Each page's column of rows, all built, the showing one shown.
    columns: Vec<(Page, NodeId)>,
    shown_page: Option<Page>,
    /// The preview bubble showing.
    lit_bubble: Option<NodeId>,
    /// Turns swipes into words ([`VirtualKeyboard::set_swipe_decoder`]).
    #[cfg(feature = "swipe")]
    swipe: Option<SwipeDecoder>,
    /// The input's clock at the last dispatch, for a swipe's times.
    #[cfg(feature = "swipe")]
    clock: f64,
    /// The word a swipe typed and the others it may have been, offered
    /// in the bar while that word is the one being typed.
    #[cfg(feature = "swipe")]
    swiped: Option<(String, Vec<String>)>,
    /// The space after a swiped word was the keyboard's own: punctuation
    /// typed next takes its place ("word." rather than "word .").
    #[cfg(feature = "swipe")]
    auto_space: bool,
    /// The last swipe's path and when the finger lifted, its trail fading.
    #[cfg(feature = "swipe")]
    trail: Option<(Vec<SwipePoint>, f64)>,
}

/// A field's own keyboard: a type for the device, or a config as given.
#[derive(Clone)]
enum FieldKeyboard {
    Type(KeyboardType),
    Config(Box<KeyboardConfig>),
}

impl Board {
    /// The config `field` shows (the default for `None`), before the tweaks:
    /// those are the app's, to be run with the board let go.
    fn config_for(&self, field: Option<NodeId>) -> KeyboardConfig {
        let own = field.and_then(|f| self.per_field.iter().find(|(n, _)| *n == f));
        match own.map(|(_, k)| k) {
            Some(FieldKeyboard::Config(c)) => (**c).clone(),
            Some(FieldKeyboard::Type(t)) => t.config(self.resolved),
            None if self.device.is_some() => KeyboardConfig::for_device(self.resolved),
            None => self.default.clone(),
        }
    }

    fn placement(&self) -> Placement {
        self.placement.unwrap_or(self.active.placement)
    }

    fn layout(&self) -> KeyboardLayout {
        self.layouts
            .get(self.current)
            .cloned()
            .unwrap_or_else(KeyboardLayout::english)
    }
}

/// A UI's own keyboard, on or not: set on its root
/// ([`Ui::environment`](crate::Ui::environment)) by [`VirtualKeyboard`].
#[derive(Clone)]
pub struct OwnKeyboard(Rc<Cell<bool>>);

impl OwnKeyboard {
    /// Whether it types into the UI's text controls now.
    pub fn on(&self) -> bool {
        self.0.get()
    }
}

/// The keyboard: built once, shown while it is on (or summoned) and a text
/// control has focus.
#[derive(Clone)]
pub struct VirtualKeyboard {
    panel: NodeId,
    sheet: NodeId,
    bar: NodeId,
    body: NodeId,
    handle: NodeId,
    /// The bar's line saying which field it types into, and its box.
    field_label: NodeId,
    label_box: NodeId,
    /// The bar's suggestions: the row, and each chip and its text.
    chips: NodeId,
    chip_nodes: Rc<[(NodeId, NodeId)]>,
    /// Over the keys, made once and shown when wanted, so a press never
    /// changes the tree's shape (which would repaint the whole window): a
    /// held key's alternates.
    popup: NodeId,
    options: Rc<[(NodeId, NodeId)]>,
    enabled: Rc<Cell<bool>>,
    /// Asked for by a field's keyboard button, until focus leaves it.
    summoned: Rc<Cell<bool>>,
    board: Rc<RefCell<Board>>,
    /// Keys pressed, waiting to be typed (pressed in listeners, which
    /// cannot dispatch; the input types them after).
    queue: Rc<RefCell<Vec<(String, KeyModifiers)>>>,
    /// A swipe decoder is set: letters type when lifted (a stroke across
    /// them is a word), even on a keyboard whose keys repeat.
    swiping: Rc<Cell<bool>>,
}

/// Gives `node` `color` for `property` (`background` or `color`).
fn paint(ui: &mut Ui, node: NodeId, property: &str, color: KeyColor, appearance: Appearance) {
    let fixed = match color {
        KeyColor::Token(t) => {
            ui.token_style(node, &[(property, t)]);
            return;
        }
        KeyColor::Fixed(c) => c,
        KeyColor::Adaptive { light, dark } => match appearance {
            Appearance::Light => light,
            Appearance::Dark => dark,
        },
    };
    ui.clear_token_style(node, property);
    ui.update_style(node, |s| match property {
        "background" => s.background = Some(fixed),
        _ => {
            // Set, so the label in it takes it (as CSS's inheritance).
            s.color = fixed;
            s.color_set = true;
        }
    });
}

/// Which of a theme's two looks the keys take: the dark one where
/// the UI's words are light (a dark appearance, or glass, whose theme has
/// light words either way), the light one where they are dark, so the
/// keyboard's bar (in the UI's colours) reads on it.
fn drawn_for(ui: &Ui) -> Appearance {
    let [r, g, b, _] = ui.token(Token::Text);
    if crate::contrast_ratio([r, g, b], [0.0; 3]) > crate::contrast_ratio([r, g, b], [1.0; 3]) {
        Appearance::Dark
    } else {
        Appearance::Light
    }
}

/// A colour as it shows now (a token as black: for shadows).
fn resolve(color: KeyColor, appearance: Appearance) -> Rgba {
    match color {
        KeyColor::Token(_) => [0.0, 0.0, 0.0, 0.35],
        KeyColor::Fixed(c) => c,
        KeyColor::Adaptive { light, dark } => match appearance {
            Appearance::Light => light,
            Appearance::Dark => dark,
        },
    }
}

impl VirtualKeyboard {
    /// The keyboard for the window: an iPhone's, an iPad's or an iPad
    /// Pro's by its size, again when it changes ([`Device::Auto`]).
    pub fn new<S: 'static>(ui: &mut Ui, input: &mut Input<S>) -> Self {
        Self::with_device(ui, input, Device::Auto)
    }

    /// The keyboard of `device` (each field's [`KeyboardType`] for it).
    pub fn with_device<S: 'static>(ui: &mut Ui, input: &mut Input<S>, device: Device) -> Self {
        Self::make(ui, input, KeyboardConfig::for_device(device), Some(device))
    }

    /// A keyboard as `config` has it, wherever it runs.
    pub fn with_config<S: 'static>(
        ui: &mut Ui,
        input: &mut Input<S>,
        config: KeyboardConfig,
    ) -> Self {
        Self::make(ui, input, config, None)
    }

    fn make<S: 'static>(
        ui: &mut Ui,
        input: &mut Input<S>,
        config: KeyboardConfig,
        device: Option<Device>,
    ) -> Self {
        let root = ui.root();
        // Across the bottom, over what is behind it; the sheet in it placed
        // as the placement says.
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
            style! { display: flex; flex-direction: column; position: relative; box-sizing: border-box; pointer-events: auto; },
        );
        ui.set_semantics(sheet, Semantics::new(Role::Group).label("Keyboard"));
        // Typed on with two thumbs: a finger landing ends the last one's
        // press (its key types), rather than the two making a pinch.
        input.set_rollover(sheet, true);
        // The bar: the field typed into, and Done.
        let bar = ui.add_box(sheet, style! { display: flex; align-items: center; gap: 8px; padding: 0 6px; min-height: 28px; });
        let fl = ui.add_box(bar, style! { flex-grow: 1; font-size: 12px; min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; });
        ui.token_style(fl, &[("color", Token::SecondaryText)]);
        let field_label = ui.add_text(fl, "");
        // Words for the one being typed, in place of the line when there
        // are any.
        let chips = ui.add_box(
            bar,
            style! { display: none; flex-grow: 1; min-width: 0; gap: 6px; },
        );
        let chip_nodes: Rc<[(NodeId, NodeId)]> = (0..SUGGESTIONS)
            .map(|_| {
                let chip = ui.add_box(
                    chips,
                    style! { display: none; justify-content: center; align-items: center; flex-grow: 1; flex-basis: 0px; min-width: 0; padding: 4px 6px; border-radius: 6px; font-size: 15px; cursor: pointer; overflow: hidden; white-space: nowrap; },
                );
                ui.token_style(chip, &[("color", Token::Text)]);
                ui.active_style(chip, "opacity: 0.6");
                (chip, ui.add_text(chip, ""))
            })
            .collect();
        let done = ui.add_box(bar, style! { padding: 4px 12px; border-radius: 6px; font-weight: bold; font-size: 15px; cursor: pointer; min-height: 24px; box-sizing: border-box; });
        ui.token_style(done, &[("color", Token::AccentText)]);
        ui.add_text(done, "Done");
        ui.set_semantics(
            done,
            Semantics::new(Role::Button).label("Done").focusable(false),
        );
        let body = ui.add_box(sheet, style! { display: flex; });
        // A floating keyboard's handle, to move it by.
        let handle = ui.add_box(
            sheet,
            style! { display: none; justify-content: center; padding: 8px 0 2px; cursor: pointer; },
        );
        let pill = ui.add_box(
            handle,
            style! { width: 40px; height: 5px; border-radius: 3px; background: rgba(128, 128, 128, 0.6); },
        );
        ui.set_semantics(
            handle,
            Semantics::new(Role::Button)
                .label("Move keyboard")
                .focusable(false),
        );
        let _ = pill;
        // The alternates, over the keys when shown.
        let popup = ui.add_box(
            panel,
            style! { position: absolute; display: none; padding: 6px; box-sizing: border-box; pointer-events: none; },
        );
        let options: Rc<[(NodeId, NodeId)]> = (0..MOST_ALTERNATES)
            .map(|_| {
                let o = ui.add_box(
                    popup,
                    style! { display: none; align-items: center; justify-content: center; pointer-events: none; },
                );
                let text = ui.add_text(o, "");
                ui.update_style(text, |s| s.pointer_events = false);
                (o, text)
            })
            .collect();

        let lang = ui
            .environment::<crate::i18n::Locale>(root)
            .map_or_else(|| "en".to_owned(), |l| l.0);
        let first = KeyboardLayout::for_locale(&lang);
        let mut layouts = vec![first.clone()];
        layouts.extend(KeyboardLayout::all().into_iter().filter(|l| *l != first));
        let board = Rc::new(RefCell::new(Board {
            default: config.clone(),
            device,
            resolved: Device::Phone,
            tweaks: vec![],
            per_field: vec![],
            active: config,
            active_field: None,
            refresh: false,
            placement: None,
            layouts,
            current: 0,
            shift: Shift::Off,
            mods: KeyModifiers::default(),
            page: Page::Letters,
            generation: 0,
            built: None,
            keys: vec![],
            press: None,
            requests: vec![],
            timers: vec![],
            shift_tapped: Rc::default(),
            drag: None,
            offset: (0.0, 0.0),
            hide: false,
            feedback: vec![],
            actions: vec![],
            context: None,
            before: None,
            auto_shift: false,
            recent: String::new(),
            secret: false,
            suggest: None,
            predict: None,
            offered: (String::new(), vec![]),
            checked: None,
            focused: None,
            columns: vec![],
            shown_page: None,
            lit_bubble: None,
            #[cfg(feature = "swipe")]
            swipe: None,
            #[cfg(feature = "swipe")]
            clock: 0.0,
            #[cfg(feature = "swipe")]
            trail: None,
            #[cfg(feature = "swipe")]
            swiped: None,
            #[cfg(feature = "swipe")]
            auto_space: false,
        }));
        let kb = Self {
            panel,
            sheet,
            bar,
            body,
            handle,
            field_label,
            label_box: fl,
            chips,
            chip_nodes: chip_nodes.clone(),
            popup,
            options,
            enabled: Rc::default(),
            summoned: Rc::default(),
            board,
            queue: Rc::default(),
            swiping: Rc::default(),
        };
        // What a native layer reads: while it is on, the UI's text controls
        // are typed into here, not through a platform keyboard
        ui.set_environment(root, OwnKeyboard(kb.enabled.clone()));
        {
            let b = kb.board.clone();
            input.on(done, Phase::Bubble, move |_, _, e, cx| {
                if let Event::PointerDown { .. } = e {
                    cx.prevent_default();
                    b.borrow_mut().hide = true;
                }
            });
        }
        {
            let me = kb.clone();
            input.on(handle, Phase::Bubble, move |_, ui, e, cx| {
                me.drag(ui, e, cx);
            });
        }
        for (i, &(chip, _)) in chip_nodes.iter().enumerate() {
            let me = kb.clone();
            input.on(chip, Phase::Bubble, move |_, ui, e, cx| {
                if let Event::PointerDown { .. } = e {
                    cx.prevent_default();
                    let (word, words) = me.board.borrow().offered.clone();
                    if let Some(w) = words.get(i) {
                        me.take_suggestion(ui, word.chars().count(), w);
                    }
                }
            });
        }

        let me = kb.clone();
        let mut shown = false;
        // While it shows docked, the root's bottom padding is its height:
        // the page above gets shorter, as a phone's does for its keyboard,
        // and what has focus is scrolled into what is left.
        let mut own_bottom: Option<f32> = None;
        let mut reveal = false;
        input.on_after_dispatch(move |input, ui, s| {
            if !ui.contains(panel) {
                return false;
            }
            #[cfg(feature = "swipe")]
            {
                me.board.borrow_mut().clock = input.now();
            }
            if std::mem::take(&mut me.board.borrow_mut().hide) {
                me.summoned.set(false);
                me.release(ui);
                input.set_focus(ui, s, None);
            }
            let window = ui.rect(ui.root());
            let secret = input.focused_text_is_secret(ui);
            me.follow(input.focused(), secret, window.width, window.height);
            me.serve_requests(input);
            // The keys, for the config, layout, page and appearance now.
            me.build(ui, input);
            // Type what was pressed, as keys, at what has focus.
            let keys: Vec<(String, KeyModifiers)> = me.queue.borrow_mut().drain(..).collect();
            for (key, modifiers) in keys {
                input.dispatch(ui, s, Event::Key { key, modifiers });
            }
            let typing = input.wants_ime(ui);
            if !typing {
                me.summoned.set(false);
            }
            // What the text before the caret says now: Shift for a new
            // sentence, words for the one being typed.
            if typing && let Some(f) = input.focused() {
                me.read_context(ui, s, f);
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
                    me.release(ui);
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
                // Its height, once laid out, unless it floats over the page.
                let floating = me.board.borrow().placement().floating;
                let h = if floating { 0.0 } else { ui.rect(panel).height };
                let want = own_bottom.unwrap_or(0.0) + h;
                if (h > 0.0 || floating) && ui.style(root).padding.bottom != want {
                    ui.update_style(root, |st| st.padding.bottom = want);
                    reveal = !floating;
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

    /// The focused field's config, or the default, for the device the
    /// window is, showing.
    fn follow(&self, focused: Option<NodeId>, secret: bool, width: f32, height: f32) {
        let mut b = self.board.borrow_mut();
        let device = b
            .device
            .map_or(b.resolved, |d| d.resolve(width.max(1.0), height.max(1.0)));
        if focused != b.focused || secret != b.secret {
            // Another field: what was typed and read was the last one's.
            b.focused = focused;
            b.secret = secret;
            b.recent.clear();
            b.before = None;
            #[cfg(feature = "swipe")]
            {
                b.swiped = None;
                b.auto_space = false;
            }
            if b.auto_shift {
                b.shift = Shift::Off;
                b.auto_shift = false;
                b.built = None;
            }
        }
        let field = focused.filter(|f| b.per_field.iter().any(|(n, _)| n == f));
        if field == b.active_field && device == b.resolved && !std::mem::take(&mut b.refresh) {
            return;
        }
        b.resolved = device;
        let mut config = b.config_for(field);
        let tweaks = b.tweaks.clone();
        drop(b);
        // (with the board let go: one may ask the keyboard something)
        for t in &tweaks {
            t(&mut config);
        }
        let mut b = self.board.borrow_mut();
        b.active = config;
        b.active_field = field;
        b.page = b.active.start_page;
        b.generation += 1;
    }

    /// Reads the text before the caret, if the app said how: Shift on for a
    /// new sentence (off again when it is not), words for the one typed.
    fn read_context<S: 'static>(&self, ui: &mut Ui, s: &S, field: NodeId) {
        if self.board.borrow().secret {
            // Nothing is read, and no words are offered.
            self.offer_words(ui, "");
            return;
        }
        let reader = self.board.borrow().context.clone();
        let Some(reader) = reader else {
            return;
        };
        let Some(read) = reader.downcast_ref::<ContextReader<S>>() else {
            return;
        };
        let before = read(s, field).unwrap_or_default();
        if self.board.borrow().before.as_deref() == Some(before.as_str()) {
            return;
        }
        let relabel = {
            let mut b = self.board.borrow_mut();
            b.before = Some(before.clone());
            let want = b.page == Page::Letters && wants_capital(b.active.auto_capitalize, &before);
            match b.shift {
                Shift::Off if want => {
                    b.shift = Shift::Once;
                    b.auto_shift = true;
                    true
                }
                Shift::Once if !want && b.auto_shift => {
                    b.shift = Shift::Off;
                    b.auto_shift = false;
                    true
                }
                _ => false,
            }
        };
        if relabel {
            self.relabel(ui);
        }
        self.offer_words(ui, current_word(&before));
    }

    /// The bar's suggestions for `word`, from the app's
    /// [`set_suggestions`](Self::set_suggestions), in its three chips.
    fn offer_words(&self, ui: &mut Ui, word: &str) {
        // What a chip replaces: the word being typed, or a swiped word
        // and the space after it.
        // (changed only by a swipe)
        #[cfg_attr(not(feature = "swipe"), allow(unused_mut))]
        let mut replaces = word.to_owned();
        #[cfg(feature = "swipe")]
        let swiped = {
            let mut b = self.board.borrow_mut();
            // A swipe's other words while its word is the one being typed,
            // or the one just before the caret and its space.
            let before = b.before.clone().unwrap_or_else(|| b.recent.clone());
            let keep = b.swiped.as_ref().map(|(w, _)| {
                if w == word {
                    Some(w.clone())
                } else if word.is_empty() && before.ends_with(&format!("{w} ")) {
                    Some(format!("{w} "))
                } else {
                    None
                }
            });
            match keep.flatten() {
                Some(r) => {
                    replaces = r;
                    b.swiped.clone().map(|(_, others)| others)
                }
                None => {
                    b.swiped = None;
                    None
                }
            }
        };
        #[cfg(not(feature = "swipe"))]
        let swiped: Option<Vec<String>> = None;
        let (suggest, predict) = {
            let b = self.board.borrow();
            (b.suggest.clone(), b.predict.clone())
        };
        let words = match (&swiped, &suggest) {
            // A secret: a predictor may learn from what it is asked about.
            _ if self.board.borrow().secret => vec![],
            (Some(others), _) => others.iter().take(SUGGESTIONS).cloned().collect(),
            (None, _) if predict.is_some() => {
                let f = predict.as_ref().expect("checked");
                let (language, before) = {
                    let b = self.board.borrow();
                    let before = b.before.clone().unwrap_or_else(|| b.recent.clone());
                    (b.layout().language, before)
                };
                let mut w = f(&language, &before);
                w.truncate(SUGGESTIONS);
                w
            }
            (None, Some(f)) if !word.is_empty() => {
                let mut w = f(word);
                w.truncate(SUGGESTIONS);
                w
            }
            _ => vec![],
        };
        {
            let b = self.board.borrow();
            if b.offered.0 == word && b.offered.1 == words {
                return;
            }
        }
        let any = !words.is_empty();
        if any != !self.board.borrow().offered.1.is_empty() {
            ui.update_style(self.chips, |st| {
                st.display = if any { Display::Flex } else { Display::None };
            });
            // The words in place of the line saying which field it is.
            ui.update_style(self.label_box, |st| {
                st.display = if any { Display::None } else { Display::Block };
            });
        }
        for (i, &(chip, text)) in self.chip_nodes.iter().enumerate() {
            let w = words.get(i);
            let display = if w.is_some() {
                Display::Flex
            } else {
                Display::None
            };
            if ui.style(chip).display != display {
                ui.update_style(chip, |st| st.display = display);
            }
            if let Some(w) = w
                && text_of(ui, text) != *w
            {
                ui.set_text(text, w.clone());
                ui.set_semantics(
                    chip,
                    Semantics::new(Role::Button)
                        .label(format!("Suggestion: {w}"))
                        .focusable(false),
                );
            }
        }
        self.board.borrow_mut().offered = (replaces, words);
    }

    /// `word` in place of the `typed` characters before the caret, and a
    /// space after it.
    fn take_suggestion(&self, ui: &mut Ui, typed: usize, word: &str) {
        #[cfg(feature = "swipe")]
        {
            self.board.borrow_mut().auto_space = false;
        }
        let mut keys = vec!["Backspace".to_owned(); typed];
        keys.extend(word.chars().map(|c| c.to_string()));
        keys.push(" ".into());
        self.send(keys);
        let key = Key::text(word).name(word);
        self.emit(&key, false, false, false);
        let _ = ui;
    }

    /// Starts and stops the timers the listeners asked for.
    fn serve_requests<S: 'static>(&self, input: &mut Input<S>) {
        let requests = std::mem::take(&mut self.board.borrow_mut().requests);
        for r in requests {
            match r {
                Request::Hold(i) => {
                    let me = self.clone();
                    let repeats = self
                        .board
                        .borrow()
                        .keys
                        .get(i)
                        .is_some_and(|k| k.key.repeats);
                    // as the config has it (a terminal's faster, say)
                    let (hold, every) = {
                        let b = self.board.borrow();
                        (b.active.repeat_delay.max(0.05), b.active.repeat_interval.max(0.01))
                    };
                    let id = if repeats {
                        // Every `every`, from `hold` on.
                        let mut waited = 0.0;
                        input.every(every, move |_, _| {
                            waited += every;
                            if waited + 1e-6 < hold {
                                return;
                            }
                            let held = {
                                let b = me.board.borrow();
                                match (&b.press, b.keys.get(i)) {
                                    (Some(p), Some(k)) if p.index == i => {
                                        Some((k.key.clone(), me.typed(&b, &k.key)))
                                    }
                                    _ => None,
                                }
                            };
                            if let Some((key, typed)) = held {
                                // Held a second, Backspace deletes words,
                                // a little slower, as Apple's does.
                                let repeats = ((waited - hold) / every).round() as u32;
                                if key.action == KeyAction::Named("Backspace".into())
                                    && repeats >= WORDS_AFTER
                                {
                                    if repeats.is_multiple_of(2) {
                                        me.queue
                                            .borrow_mut()
                                            .push(("Backspace".into(), word_modifier()));
                                        me.emit(&key, true, false, false);
                                    }
                                    return;
                                }
                                me.send(typed);
                                me.emit(&key, true, false, false);
                            }
                        })
                    } else {
                        input.after(HOLD, move |_, ui| me.offer_alternates(ui, i))
                    };
                    self.board.borrow_mut().timers.push(id);
                }
                Request::Release => {
                    let timers = std::mem::take(&mut self.board.borrow_mut().timers);
                    for t in timers {
                        input.cancel_timer(t);
                    }
                }
                Request::ShiftTap => {
                    let tapped = self.board.borrow().shift_tapped.clone();
                    tapped.set(true);
                    input.after(DOUBLE_TAP, move |_, _| tapped.set(false));
                }
            }
        }
    }

    /// The keys built (again) when the config, placement, layout or
    /// appearance changed: every page at once, so a page shows by being
    /// shown, without building keys or changing the tree's shape.
    fn build<S: 'static>(&self, ui: &mut Ui, input: &mut Input<S>) {
        let appearance = drawn_for(ui);
        let (want, page) = {
            let b = self.board.borrow();
            ((b.generation, b.current, Page::Letters, appearance), b.page)
        };
        if self.board.borrow().built == Some(want) {
            if self.board.borrow().shown_page != Some(page) {
                self.show_page(ui, page);
            }
            return;
        }
        let (cfg, placement, layout, many) = {
            let b = self.board.borrow();
            (
                b.active.clone(),
                b.placement(),
                b.layout(),
                b.layouts.len() > 1,
            )
        };
        self.release(ui);
        self.place(ui, &cfg, placement, appearance);
        ui.clear_children(self.body);

        // In a debug build, what is wrong with the config, once.
        if cfg!(debug_assertions) && self.board.borrow().checked != Some(want.0) {
            self.board.borrow_mut().checked = Some(want.0);
            for issue in cfg.validate() {
                eprintln!("threers-ui keyboard: {issue}");
            }
        }
        let languages = if many { 2 } else { 1 };
        let mut built = vec![];

        // The pages' columns, and a one-handed keyboard's strip beside them.
        let holder = ui.add_box(
            self.body,
            style! { display: flex; flex-direction: column; flex-grow: 1; min-width: 0; },
        );
        if placement.one_handed {
            let strip = ui.add_box(
                self.body,
                style! { display: flex; flex-direction: column; justify-content: center; flex-shrink: 0; },
            );
            ui.update_style(strip, |s| {
                s.width = Length::Px(STRIP);
                s.order = if placement.align == Align::Right {
                    -1
                } else {
                    1
                };
            });
            let back = Placement {
                one_handed: false,
                ..cfg.placement
            };
            let arrow = if placement.align == Align::Right {
                "‹"
            } else {
                "›"
            };
            let key = Key::place(back, arrow, "Full width keyboard")
                .look(KeyLook::Clear)
                .font_size(34.0);
            let row = ui.add_box(strip, style! { display: flex; });
            self.build_key(ui, input, row, key, 1.0, &cfg, appearance, None, &mut built);
        }
        let mut columns = vec![];
        for p in [Page::Letters, Page::Symbols, Page::More] {
            let rows = spec::page_rows(&cfg, p, &layout, languages);
            if rows.is_empty() {
                continue;
            }
            let total = spec::total_units(&rows);
            let column = ui.add_box(holder, style! { display: none; flex-direction: column; });
            ui.update_style(column, |s| s.gap = cfg.row_gap);
            for row in rows {
                let r = ui.add_box(column, style! { display: flex; justify-content: center; });
                if placement.split > 0.0 {
                    self.build_split_row(
                        ui,
                        input,
                        r,
                        row,
                        total,
                        placement.split,
                        &cfg,
                        appearance,
                        p,
                        &mut built,
                    );
                } else {
                    for k in row {
                        self.build_key(
                            ui,
                            input,
                            r,
                            k,
                            total,
                            &cfg,
                            appearance,
                            Some(p),
                            &mut built,
                        );
                    }
                }
            }
            columns.push((p, column));
        }
        {
            let mut b = self.board.borrow_mut();
            b.keys = built;
            b.built = Some(want);
            b.columns = columns;
            b.shown_page = None;
        }
        ui.semantics_mut(self.sheet).label = Some(format!("Keyboard: {}", layout.name));
        self.show_page(ui, page);
    }

    /// `page` shown, the others hidden, and its keys labelled for Shift.
    fn show_page(&self, ui: &mut Ui, page: Page) {
        let columns = self.board.borrow().columns.clone();
        // A page with no rows shows the letters.
        let page = if columns.iter().any(|(p, _)| *p == page) {
            page
        } else {
            Page::Letters
        };
        for (p, column) in columns {
            let display = if p == page {
                Display::Flex
            } else {
                Display::None
            };
            if ui.style(column).display != display {
                ui.update_style(column, |s| s.display = display);
            }
        }
        {
            let mut b = self.board.borrow_mut();
            b.page = page;
            b.shown_page = Some(page);
        }
        self.relabel(ui);
    }

    /// A row parted in the middle: the keys wholly left of it in one half
    /// (`q w e r t | y u i o p`, `⇧ z x c v | b n m`), the rest in the other, each half against its side; a key wide enough to
    /// straddle the parting (the space bar) in both.
    #[allow(clippy::too_many_arguments)]
    fn build_split_row<S: 'static>(
        &self,
        ui: &mut Ui,
        input: &mut Input<S>,
        row: NodeId,
        keys: Vec<Key>,
        total: f32,
        split: f32,
        cfg: &KeyboardConfig,
        appearance: Appearance,
        page: Page,
        built: &mut Vec<Built>,
    ) {
        let half = total / 2.0;
        let sum: f32 = keys.iter().map(|k| k.units_and_fill().0).sum();
        // What the row leaves goes to its fill keys, as it would unparted;
        // a row without one starts where it would centred.
        let fills = keys.iter().filter(|k| k.units_and_fill().1).count();
        let spare = (total - sum).max(0.0);
        let lead = if fills > 0 { 0.0 } else { spare / 2.0 };
        let keys: Vec<Key> = keys
            .into_iter()
            .map(|mut k| {
                if let (u, true) = k.units_and_fill() {
                    k.size = config::KeySize::Fill(u + spare / fills as f32);
                }
                k
            })
            .collect();
        let (mut left, mut right) = (vec![], vec![]);
        if lead > 0.0 {
            left.push(Key::gap(lead));
        }
        let mut x = lead;
        for k in keys {
            let (u, fill) = k.units_and_fill();
            if fill && x < half && x + u > half {
                let mut l = k.clone();
                l.size = config::KeySize::Fill(half - x);
                let mut r = k;
                r.size = config::KeySize::Fill(x + u - half);
                left.push(l);
                right.push(r);
            } else if x + u <= half + 1e-3 {
                left.push(k);
            } else {
                right.push(k);
            }
            x += u;
        }
        if lead > 0.0 {
            right.push(Key::gap(lead));
        }
        for (side, keys, justify) in [
            (0, left, JustifyContent::FlexStart),
            (1, right, JustifyContent::FlexEnd),
        ] {
            // Half the row each, less half the parting: a definite width
            // for its keys' percentages.
            let group = ui.add_box(row, style! { display: flex; flex-shrink: 1; min-width: 0; });
            ui.update_style(group, |s| {
                s.justify_content = justify;
                s.flex_basis = Length::Percent(50.0);
            });
            for k in keys {
                self.build_key(
                    ui,
                    input,
                    group,
                    k,
                    half,
                    cfg,
                    appearance,
                    Some(page),
                    built,
                );
            }
            if side == 0 {
                let gap = ui.add_box(row, style! { flex-shrink: 0; });
                ui.update_style(gap, |s| s.width = Length::Px(split));
            }
        }
    }

    /// A key: a cell `units / total` of its row wide that takes the
    /// pointer (the gaps round the key included, as on a phone), and the
    /// key drawn in it.
    #[allow(clippy::too_many_arguments)]
    fn build_key<S: 'static>(
        &self,
        ui: &mut Ui,
        input: &mut Input<S>,
        row: NodeId,
        key: Key,
        total: f32,
        cfg: &KeyboardConfig,
        appearance: Appearance,
        page: Option<Page>,
        built: &mut Vec<Built>,
    ) {
        let (u, fill) = key.units_and_fill();
        let cell = ui.add_box(
            row,
            style! { display: flex; flex-direction: column; box-sizing: border-box; min-width: 0; },
        );
        ui.update_style(cell, |s| {
            s.flex_basis = Length::Percent(u / total.max(1e-3) * 100.0);
            s.flex_grow = if fill { 1.0 } else { 0.0 };
            s.flex_shrink = 1.0;
            s.padding.left = cfg.gap / 2.0;
            s.padding.right = cfg.gap / 2.0;
            s.height = Length::Px(cfg.key_height);
        });
        if key.action == KeyAction::None {
            return;
        }
        let theme = &cfg.theme;
        let face = ui.add_box(
            cell,
            style! { display: flex; flex-direction: column; align-items: center; justify-content: center; flex-grow: 1; min-width: 0; overflow: hidden; cursor: pointer; },
        );
        let label_text = self.label_of(&key, cfg);
        let size = key.font_size.unwrap_or_else(|| {
            let glyphs = label_text.chars().count();
            if glyphs <= 1 || key.action == KeyAction::NextLayout {
                theme.letter_size
            } else {
                theme.label_size
            }
        });
        ui.update_style(face, |s| {
            s.border_radius = Length::Px(theme.key_radius);
            s.font_size = size;
            s.line_height = Some(size * 1.2);
            if key.look != KeyLook::Clear
                && let Some(c) = theme.shadow
            {
                s.box_shadow = vec![BoxShadow {
                    x: 0.0,
                    y: 1.0,
                    blur: 0.0,
                    spread: 0.0,
                    color: resolve(c, appearance),
                }];
            }
        });
        if key.hint
            && let Some(sh) = &key.shifted
        {
            let h = ui.add_box(face, style! { opacity: 0.6; });
            ui.update_style(h, |s| {
                s.font_size = theme.hint_size;
                s.line_height = Some(theme.hint_size * 1.2);
            });
            ui.add_text(h, sh.clone());
        }
        let label = ui.add_text(face, label_text.clone());
        if let Some(c) = &key.caption {
            let h = ui.add_box(face, style! { font-weight: bold; });
            ui.update_style(h, |s| {
                s.font_size = theme.hint_size;
                s.line_height = Some(theme.hint_size * 1.2);
            });
            ui.add_text(h, c.clone());
        }
        let name = key.name.clone().unwrap_or_else(|| label_text.clone());
        ui.set_semantics(
            face,
            Semantics::new(Role::Button).label(name).focusable(false),
        );
        // Pressed: a letter with a preview keeps its look; the rest as
        // the feedback says.
        let previewed = cfg.feedback.preview
            && key.look == KeyLook::Character
            && label_text.chars().count() == 1;
        if !previewed {
            match cfg.feedback.press {
                PressStyle::None => {}
                PressStyle::Dim => ui.active_style(face, "opacity: 0.6"),
                PressStyle::Swap => {
                    let other = match key.look {
                        KeyLook::Character => theme.function_key,
                        KeyLook::Function | KeyLook::Clear => theme.key,
                        KeyLook::Accent => theme.function_key,
                    };
                    match other {
                        // A token's colour is the theme's to give: fade instead.
                        KeyColor::Token(_) => ui.active_style(face, "opacity: 0.6"),
                        c => {
                            let [r, g, b, a] = resolve(c, appearance);
                            let css = format!(
                                "background: rgba({}, {}, {}, {a})",
                                (r * 255.0).round(),
                                (g * 255.0).round(),
                                (b * 255.0).round()
                            );
                            ui.active_style(face, &css);
                        }
                    }
                }
            }
        }
        // Its preview, laid out over it now so showing it later is only
        // drawn: half a key wider, a little taller, its letter larger.
        let bubble = previewed.then(|| {
            let h = cfg.key_height * 1.15;
            ui.update_style(cell, |s| s.position = crate::Position::Relative);
            let b = ui.add_box(
                cell,
                style! { position: absolute; display: flex; align-items: center; justify-content: center; pointer-events: none; opacity: 0; },
            );
            ui.update_style(b, |s| {
                s.left = Length::Px(cfg.gap / 2.0 - 10.0);
                s.right = Length::Px(cfg.gap / 2.0 - 10.0);
                s.top = Length::Px(-(h + 4.0));
                s.height = Length::Px(h);
                s.border_radius = Length::Px(theme.key_radius + 2.0);
                s.font_size = theme.letter_size * 1.6;
                s.line_height = Some(theme.letter_size * 1.6 * 1.2);
                s.box_shadow = vec![BoxShadow {
                    x: 0.0,
                    y: 2.0,
                    blur: 8.0,
                    spread: 0.0,
                    color: [0.0, 0.0, 0.0, 0.3],
                }];
            });
            paint(ui, b, "background", theme.key, appearance);
            paint(ui, b, "color", theme.key_text, appearance);
            // Decoration: the key itself is what assistive technology reads.
            ui.set_semantics(b, Semantics::new(Role::Generic).hidden());
            let text = ui.add_text(b, label_text.clone());
            // Laid out over the row above even while unseen: nothing in it
            // may take a press meant for the keys there (`pointer-events`
            // is not inherited here, as it is in CSS).
            ui.update_style(text, |s| s.pointer_events = false);
            (b, text)
        });
        // a terminal's keyboard: every key that types goes on while held
        // (but letters, while a swipe decoder reads strokes across them)
        let letter = matches!(&key.action, KeyAction::Text(t) if t.chars().count() == 1 && t.chars().all(char::is_alphabetic));
        let key = if cfg.repeat_typing && matches!(key.action, KeyAction::Text(_)) && !(letter && self.swiping.get()) {
            Key { repeats: true, alternates: Vec::new(), ..key }
        } else {
            key
        };
        let index = built.len();
        let me = self.clone();
        input.on(cell, Phase::Bubble, move |_, ui, e, cx| {
            me.on_key_event(ui, index, e, cx);
        });
        built.push(Built {
            cell,
            face,
            label,
            key,
            page,
            shown: None,
            bubble,
        });
    }

    /// What `key` shows, Shift aside.
    fn label_of(&self, key: &Key, cfg: &KeyboardConfig) -> String {
        match &key.action {
            KeyAction::Named(n) if n == "Enter" => cfg.return_key.label(),
            KeyAction::Text(t) if t == " " => {
                let b = self.board.borrow();
                if b.layouts.len() > 1 && b.page == Page::Letters && key.label == "space" {
                    b.layout().name
                } else {
                    key.label.clone()
                }
            }
            _ => key.label.clone(),
        }
    }

    /// The panel, the sheet and the handle placed as `placement` says.
    fn place(&self, ui: &mut Ui, cfg: &KeyboardConfig, p: Placement, appearance: Appearance) {
        let theme = &cfg.theme;
        let offset = if p.floating {
            self.board.borrow().offset
        } else {
            (0.0, 0.0)
        };
        ui.update_style(self.panel, |s| {
            s.align_items = match p.align {
                Align::Left => AlignItems::FlexStart,
                Align::Center => AlignItems::Center,
                Align::Right => AlignItems::FlexEnd,
            };
            let m = if p.floating { p.margin } else { 0.0 };
            s.bottom = Length::Px(m);
            s.padding.left = m;
            s.padding.right = m;
        });
        ui.update_style(self.sheet, |s| {
            s.width = match p.width {
                Some(w) if !p.one_handed => Length::Px(w),
                _ => Length::Percent(100.0),
            };
            s.max_width = p.max_width.map_or(Length::Auto, Length::Px);
            let radius = if p.floating {
                theme.radius.max(14.0)
            } else {
                theme.radius
            };
            s.border_radius = Length::Px(radius);
            s.box_shadow = if p.floating || radius > 0.0 {
                vec![BoxShadow {
                    x: 0.0,
                    y: if p.floating { 8.0 } else { -6.0 },
                    blur: 24.0,
                    spread: 0.0,
                    color: [0.0, 0.0, 0.0, 0.3],
                }]
            } else {
                vec![]
            };
            let [t, r, b, l] = cfg.padding;
            s.padding = crate::Edges {
                top: t,
                right: r,
                bottom: b,
                left: l,
            };
            s.gap = cfg.row_gap.min(8.0);
            s.left = Length::Px(offset.0);
            s.top = Length::Px(offset.1);
        });
        // Docked, it reaches the bottom edge, under the home indicator.
        ui.pad_safe_area(self.sheet, [false, false, !p.floating, false]);
        paint(ui, self.sheet, "background", theme.background, appearance);
        paint(ui, self.sheet, "color", theme.key_text, appearance);
        ui.update_style(self.bar, |s| {
            let b = self.board.borrow();
            s.display = if cfg.bar || b.suggest.is_some() || b.predict.is_some() {
                Display::Flex
            } else {
                Display::None
            };
        });
        ui.update_style(self.handle, |s| {
            s.display = if p.floating {
                Display::Flex
            } else {
                Display::None
            };
        });
    }

    /// Each key's label for Shift, and its colours: on the page showing,
    /// and only where they change.
    fn relabel(&self, ui: &mut Ui) {
        let appearance = drawn_for(ui);
        let mut b = self.board.borrow_mut();
        let theme = b.active.theme;
        let accent_return = b.active.return_key.accent();
        let (page, shift, mods) = (b.page, b.shift, b.mods);
        let up = page == Page::Letters && shift != Shift::Off;
        for k in b.keys.iter_mut() {
            if k.page.is_some_and(|p| p != page) {
                continue;
            }
            let lit = match k.key.action {
                KeyAction::Shift => shift != Shift::Off,
                KeyAction::CapsLock => shift == Shift::Locked,
                KeyAction::Modifier(Modifier::Control) => mods.control,
                KeyAction::Modifier(Modifier::Alt) => mods.alt,
                KeyAction::Modifier(Modifier::Meta) => mods.meta,
                _ => false,
            };
            let label = match &k.key.action {
                KeyAction::Text(_) if !k.key.hint && k.key.label != "space" => {
                    if up {
                        k.key.shifted.clone().unwrap_or_else(|| upper(&k.key.label))
                    } else {
                        k.key.label.clone()
                    }
                }
                KeyAction::Shift => if shift == Shift::Locked { "⇪" } else { "⇧" }.to_owned(),
                _ => text_of(ui, k.label),
            };
            let enter = matches!(&k.key.action, KeyAction::Named(n) if n == "Enter");
            let look = if enter && accent_return {
                KeyLook::Accent
            } else {
                k.key.look
            };
            let (bg, fg) = if lit {
                (Some(theme.lit_key), theme.lit_text)
            } else {
                match look {
                    KeyLook::Character => (Some(theme.key), theme.key_text),
                    KeyLook::Function => (Some(theme.function_key), theme.function_text),
                    KeyLook::Accent => (Some(theme.accent_key), theme.accent_text),
                    KeyLook::Clear => (None, theme.function_text),
                }
            };
            let was = k.shown.take();
            if was.as_ref().is_none_or(|w| w.0 != label) && text_of(ui, k.label) != label {
                ui.set_text(k.label, label.clone());
                if k.key.name.is_none() {
                    ui.semantics_mut(k.face).label = Some(label.clone());
                }
                if let Some((_, text)) = k.bubble {
                    ui.set_text(text, label.clone());
                }
            }
            if was.as_ref().is_none_or(|w| w.1 != bg || w.2 != fg) {
                match bg {
                    Some(c) => paint(ui, k.face, "background", c, appearance),
                    None => {
                        ui.clear_token_style(k.face, "background");
                        ui.update_style(k.face, |s| s.background = None);
                    }
                }
                paint(ui, k.face, "color", fg, appearance);
            }
            k.shown = Some((label, bg, fg));
        }
    }

    /// What pressing `key` types now: one key event per character of its
    /// text (upper case under Shift), or its name.
    fn typed(&self, b: &Board, key: &Key) -> Vec<String> {
        match &key.action {
            KeyAction::Text(t) => {
                let t = if b.page == Page::Letters && b.shift != Shift::Off {
                    key.shifted.clone().unwrap_or_else(|| upper(t))
                } else {
                    t.clone()
                };
                t.chars().map(|c| c.to_string()).collect()
            }
            KeyAction::Named(n) => vec![n.clone()],
            _ => vec![],
        }
    }

    /// Queues `keys` to type, unmodified.
    /// Queues `keys`, with Control/Alt/Meta if they were pressed for them
    /// (then let go); true when they were.
    fn send(&self, keys: impl IntoIterator<Item = String>) -> bool {
        let mods = std::mem::take(&mut self.board.borrow_mut().mods);
        self.queue.borrow_mut().extend(keys.into_iter().map(|k| (k, mods)));
        mods != KeyModifiers::default()
    }

    /// Types `key`: two spaces after a word as “. ”, back to the letters
    /// after a space on a symbols page, and Shift held for one letter let
    /// go.
    fn type_key(&self, ui: &mut Ui, key: &Key) {
        let mut typed = self.typed(&self.board.borrow(), key);
        if typed.is_empty() {
            return;
        }
        let once = {
            let mut b = self.board.borrow_mut();
            #[cfg(feature = "swipe")]
            if std::mem::take(&mut b.auto_space)
                && typed.len() == 1
                && typed[0].chars().count() == 1
                && ".,!?;:".contains(typed[0].as_str())
            {
                typed.insert(0, "Backspace".into());
            }
            if typed == [" "] && b.active.period_shortcut && !b.secret {
                // The field's text if the app tells it, else what this
                // keyboard typed.
                let before = b.before.clone().unwrap_or_else(|| b.recent.clone());
                if period_shortcut(&before) {
                    typed = vec!["Backspace".into(), ".".into(), " ".into()];
                }
            }
            // (what is typed of a secret is not kept)
            let kept = !b.secret;
            for t in typed.iter().filter(|_| kept) {
                if t == "Backspace" {
                    b.recent.pop();
                } else if t.chars().count() == 1 {
                    b.recent.push_str(t);
                }
            }
            let excess = b.recent.chars().count().saturating_sub(16);
            if excess > 0 {
                b.recent = b.recent.chars().skip(excess).collect();
            }
            if b.active.return_to_letters
                && b.page != Page::Letters
                && matches!(&key.action, KeyAction::Text(t) if t == " " || t == "'")
            {
                b.page = Page::Letters;
            }
            let once = b.shift == Shift::Once && matches!(key.action, KeyAction::Text(_));
            if once {
                b.shift = Shift::Off;
                b.auto_shift = false;
            }
            once
        };
        let modified = self.send(typed);
        if once || modified {
            self.relabel(ui);
        }
    }

    /// Tells the app's [`on_feedback`](Self::on_feedback) listeners of a
    /// press: its click, and a tap (a firmer one for a `strong` press).
    fn emit(&self, key: &Key, repeat: bool, strong: bool, silent: bool) {
        let (listeners, fb) = {
            let b = self.board.borrow();
            (b.feedback.clone(), b.active.feedback)
        };
        if listeners.is_empty() {
            return;
        }
        let press = KeyPress {
            name: key.name.clone().unwrap_or_else(|| key.label.clone()),
            action: key.action.clone(),
            sound: (fb.sound && !silent)
                .then(|| KeySound::of(&key.action))
                .flatten(),
            volume: fb.volume,
            haptic: fb.haptics.then_some(if strong {
                Haptic::Medium
            } else {
                Haptic::Light
            }),
            repeat,
        };
        for f in listeners {
            f(&press);
        }
    }

    /// The pointer on key `index`.
    fn on_key_event(&self, ui: &mut Ui, index: usize, e: &Event, cx: &mut EventCx) {
        match e {
            Event::PointerDown { x, y, .. } => {
                // The focus stays in the field.
                cx.prevent_default();
                cx.capture_pointer();
                // the key where it lands: another finger's key may hold the
                // pointer (typing with two thumbs)
                let index = self.index_at(ui, *x, *y).unwrap_or(index);
                self.down(ui, index, *x, *y);
            }
            Event::PointerMove { x, y } => {
                if self.board.borrow().press.is_some() {
                    self.slide(ui, *x, *y);
                }
            }
            Event::PointerUp { x, y, .. } => {
                cx.release_pointer();
                self.up(ui, *x, *y);
            }
            Event::PointerCancel => self.release(ui),
            // A held key offers its own alternates, not a context menu.
            Event::ContextMenu { .. } => cx.prevent_default(),
            _ => {}
        }
    }

    fn down(&self, ui: &mut Ui, index: usize, x: f32, y: f32) {
        // rollover: a key still down when the next comes down is typed now
        // (fast two-thumb typing lands the next finger before the last
        // lifts); anything else held (alternates, the trackpad) is let go
        let rolled = self
            .board
            .borrow()
            .press
            .as_ref()
            .filter(|p| p.types && p.popup.is_none() && p.trackpad.is_none())
            .map(|p| p.at);
        match rolled {
            Some((px, py)) => self.up(ui, px, py),
            None => self.release(ui),
        }
        let Some(key) = self.board.borrow().keys.get(index).map(|k| k.key.clone()) else {
            return;
        };
        let mut relabel = false;
        let mut types = false;
        {
            let mut b = self.board.borrow_mut();
            match &key.action {
                KeyAction::Shift => {
                    b.auto_shift = false;
                    let twice = b.shift_tapped.get();
                    b.shift = match b.shift {
                        Shift::Off if twice => Shift::Locked,
                        Shift::Off => Shift::Once,
                        Shift::Once if twice => Shift::Locked,
                        _ => Shift::Off,
                    };
                    b.requests.push(Request::ShiftTap);
                    relabel = true;
                }
                KeyAction::CapsLock => {
                    b.shift = if b.shift == Shift::Locked {
                        Shift::Off
                    } else {
                        Shift::Locked
                    };
                    relabel = true;
                }
                KeyAction::Modifier(m) => {
                    let on = match m {
                        Modifier::Control => &mut b.mods.control,
                        Modifier::Alt => &mut b.mods.alt,
                        Modifier::Meta => &mut b.mods.meta,
                    };
                    *on = !*on;
                    relabel = true;
                }
                KeyAction::Page(p) => b.page = *p,
                KeyAction::NextLayout => {
                    b.current = (b.current + 1) % b.layouts.len().max(1);
                    b.page = Page::Letters;
                }
                KeyAction::Dismiss => b.hide = true,
                KeyAction::Place(p) => {
                    b.placement = Some(*p);
                    b.offset = (0.0, 0.0);
                    b.generation += 1;
                }
                _ if key.repeats => {}
                KeyAction::Text(_) | KeyAction::Named(_) | KeyAction::Action(_) => types = true,
                _ => {}
            }
            // Only Shift twice in a row locks it.
            if key.action != KeyAction::Shift {
                b.shift_tapped.set(false);
            }
            #[cfg(feature = "swipe")]
            let path = {
                let letter = b.page == Page::Letters
                    && matches!(&key.action, KeyAction::Text(t) if swipe_letter(t).is_some());
                (types && letter && b.swipe.is_some() && !b.secret)
                    .then(|| (vec![SwipePoint { x, y, t: b.clock }], 0.0))
            };
            b.press = Some(Press {
                index,
                types,
                popup: None,
                at: (x, y),
                trackpad: None,
                #[cfg(feature = "swipe")]
                path,
                #[cfg(feature = "swipe")]
                swiping: false,
            });
            let space = key.action == KeyAction::Text(" ".into());
            if key.repeats
                || (types && !key.alternates.is_empty())
                || (space && b.active.space_trackpad)
            {
                b.requests.push(Request::Hold(index));
            }
        }
        if key.repeats {
            // Once now, then again while held.
            self.type_key(ui, &key);
        }
        if relabel {
            self.relabel(ui);
        }
        if types {
            self.preview(ui, index);
        }
        let locked = key.action == KeyAction::Shift && self.board.borrow().shift == Shift::Locked;
        self.emit(&key, false, locked, false);
    }

    /// The pointer moved while pressed: along a held key's alternates, or
    /// from key to key (the one under the finger types).
    /// The key on the page showing at (x, y).
    fn index_at(&self, ui: &Ui, x: f32, y: f32) -> Option<usize> {
        let b = self.board.borrow();
        b.keys
            .iter()
            .position(|k| k.page.is_none_or(|p| p == b.page) && ui.rect(k.cell).contains(x, y))
    }

    fn slide(&self, ui: &mut Ui, x: f32, y: f32) {
        if !x.is_finite() || !y.is_finite() {
            return;
        }
        let mut b = self.board.borrow_mut();
        let Some(press) = b.press.as_mut() else {
            return;
        };
        press.at = (x, y);
        if let Some((ax, ay)) = press.trackpad {
            // A caret step for every few pixels: across, and (further) up
            // and down.
            let across = ((x - ax) / TRACKPAD_STEP).trunc();
            let down = ((y - ay) / (TRACKPAD_STEP * 2.0)).trunc();
            press.trackpad = Some((ax + across * TRACKPAD_STEP, ay + down * TRACKPAD_STEP * 2.0));
            drop(b);
            let step = |n: f32, more: &str, less: &str| {
                let key = if n > 0.0 { more } else { less };
                std::iter::repeat_n(key.to_owned(), (n.abs() as usize).min(TRACKPAD_MOST))
            };
            self.send(step(across, "ArrowRight", "ArrowLeft"));
            self.send(step(down, "ArrowDown", "ArrowUp"));
            return;
        }
        if let Some(popup) = press.popup.as_mut() {
            let chosen = self.options[..popup.options.len()]
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, c)| {
                    let d = |n: NodeId| {
                        let r = ui.rect(n);
                        (r.x + r.width / 2.0 - x).abs()
                    };
                    d(a.0).total_cmp(&d(c.0))
                })
                .map_or(0, |(i, _)| i);
            if chosen != popup.chosen {
                let was = popup.chosen;
                popup.chosen = chosen;
                drop(b);
                self.light_options(ui, &[was, chosen], chosen);
            }
            return;
        }
        #[cfg(feature = "swipe")]
        let press = {
            let index = press.index;
            let clock = b.clock;
            let start_width = b.keys.get(index).map(|k| ui.rect(k.cell).width);
            let press = b.press.as_mut().unwrap();
            if let Some((path, travel)) = press.path.as_mut() {
                let last = *path.last().unwrap();
                *travel += (x - last.x).hypot(y - last.y);
                path.push(SwipePoint { x, y, t: clock });
                if press.swiping {
                    return;
                }
                if *travel > SWIPE_START * start_width.unwrap_or(f32::MAX) {
                    // A swipe: no letter types, no hold offers accents.
                    press.swiping = true;
                    b.requests.push(Request::Release);
                    drop(b);
                    self.unlight_bubble(ui);
                    return;
                }
            }
            press
        };
        if !press.types {
            return;
        }
        let at = press.index;
        let page = b.page;
        let under = b.keys.iter().position(|k| {
            matches!(k.key.action, KeyAction::Text(_) | KeyAction::Named(_))
                && !k.key.repeats
                && k.page.is_none_or(|p| p == page)
                && ui.rect(k.cell).contains(x, y)
        });
        let Some(to) = under.filter(|&to| to != at) else {
            return;
        };
        b.press.as_mut().unwrap().index = to;
        // The new key's hold counts from now.
        b.requests.push(Request::Release);
        if !b.keys[to].key.alternates.is_empty() {
            b.requests.push(Request::Hold(to));
        }
        drop(b);
        self.preview(ui, to);
    }

    fn up(&self, ui: &mut Ui, x: f32, y: f32) {
        let press = self.board.borrow_mut().press.take();
        let Some(press) = press else {
            return;
        };
        self.close(ui, &press);
        self.board.borrow_mut().requests.push(Request::Release);
        #[cfg(feature = "swipe")]
        if press.swiping
            && let Some((mut path, _)) = press.path
        {
            let t = self.board.borrow().clock;
            path.push(SwipePoint { x, y, t });
            self.board.borrow_mut().trail = Some((path.clone(), t));
            self.type_swipe(ui, path);
            return;
        }
        if press.trackpad.is_some() {
            // The caret was moved; the space is not typed.
            return;
        }
        if let Some(popup) = &press.popup {
            if let Some(text) = popup.options.get(popup.chosen) {
                let key = Key::text(text);
                // An accented capital stays as chosen.
                let shift = self.board.borrow().shift;
                self.board.borrow_mut().shift = Shift::Off;
                self.type_key(ui, &key);
                if shift == Shift::Once {
                    self.relabel(ui);
                } else {
                    self.board.borrow_mut().shift = shift;
                }
            }
            return;
        }
        if !press.types {
            return;
        }
        // The key under the finger as it lifts, as a phone has it; the one
        // pressed when it lifts off the keyboard's keys.
        let key = {
            let b = self.board.borrow();
            b.keys
                .iter()
                .find(|k| {
                    matches!(
                        k.key.action,
                        KeyAction::Text(_) | KeyAction::Named(_) | KeyAction::Action(_)
                    ) && !k.key.repeats
                        && k.page.is_none_or(|p| p == b.page)
                        && ui.rect(k.cell).contains(x, y)
                })
                .or_else(|| b.keys.get(press.index))
                .map(|k| k.key.clone())
        };
        match key {
            Some(Key {
                action: KeyAction::Action(id),
                ..
            }) => {
                let listeners = self.board.borrow().actions.clone();
                for f in listeners {
                    f(&id, ui);
                }
            }
            Some(key) => self.type_key(ui, &key),
            None => {}
        }
    }

    #[cfg(feature = "swipe")]
    /// A finished swipe: the decoder's best word typed with a space after
    /// it (and one before it after a word typed by tapping), capitals as
    /// Shift says; its other words in the bar.
    fn type_swipe(&self, ui: &mut Ui, points: Vec<SwipePoint>) {
        let (decoder, swipe, shift, before) = {
            let b = self.board.borrow();
            let Some(decoder) = b.swipe.clone() else {
                return;
            };
            let keys = b
                .keys
                .iter()
                .filter(|k| k.page.is_none_or(|p| p == b.page) && ui.contains(k.cell))
                .filter_map(|k| match &k.key.action {
                    KeyAction::Text(t) => swipe_letter(t).map(|c| {
                        let r = ui.rect(k.cell);
                        SwipeKey {
                            text: c.to_lowercase().collect(),
                            x: r.x,
                            y: r.y,
                            width: r.width,
                            height: r.height,
                        }
                    }),
                    _ => None,
                })
                .collect();
            let before = b.before.clone().unwrap_or_else(|| b.recent.clone());
            let swipe = Swipe {
                points,
                keys,
                language: b.layout().language,
                before: before.clone(),
            };
            (decoder, swipe, b.shift, before)
        };
        let words = decoder(&swipe);
        let Some(best) = words.first() else { return };
        let cased = |w: &str| match shift {
            Shift::Locked => upper(w),
            Shift::Once => {
                let mut c = w.chars();
                c.next()
                    .map_or_else(String::new, |f| upper(&f.to_string()) + c.as_str())
            }
            Shift::Off => w.to_owned(),
        };
        let word = cased(best);
        let others: Vec<String> = words[1..].iter().map(|w| cased(w)).collect();
        let mut keys: Vec<String> = vec![];
        if before
            .chars()
            .last()
            .is_some_and(|c| c.is_alphanumeric() || ".,!?;:".contains(c))
        {
            keys.push(" ".into());
        }
        keys.extend(word.chars().map(|c| c.to_string()));
        keys.push(" ".into());
        let relabel = {
            let mut b = self.board.borrow_mut();
            b.auto_space = true;
            for k in &keys {
                b.recent.push_str(k);
            }
            // The text before the caret as it will be once the keys are
            // in, so the bar can offer the word's alternatives now.
            if let Some(before) = b.before.as_mut() {
                before.push_str(&keys.concat());
            }
            let excess = b.recent.chars().count().saturating_sub(16);
            if excess > 0 {
                b.recent = b.recent.chars().skip(excess).collect();
            }
            b.swiped = Some((word.clone(), others));
            let once = b.shift == Shift::Once;
            if once {
                b.shift = Shift::Off;
                b.auto_shift = false;
            }
            once
        };
        self.send(keys);
        self.emit(&Key::text(&word).name(&word), false, false, true);
        if relabel {
            self.relabel(ui);
        }
        // The caret is past the word's space: no word being typed, the
        // swiped one and its space what a chip replaces.
        self.offer_words(ui, "");
    }

    /// Lets go of any key held, its preview and alternates closed, its
    /// timers stopped.
    fn release(&self, ui: &mut Ui) {
        let press = self.board.borrow_mut().press.take();
        if let Some(press) = press {
            self.close(ui, &press);
            self.board.borrow_mut().requests.push(Request::Release);
        }
    }

    fn close(&self, ui: &mut Ui, press: &Press) {
        self.unlight_bubble(ui);
        if press.trackpad.is_some() && ui.contains(self.body) {
            ui.update_style(self.body, |s| s.opacity = 1.0);
        }
        if press.popup.is_some() {
            self.hide(ui, self.popup);
        }
    }

    fn hide(&self, ui: &mut Ui, node: NodeId) {
        if ui.contains(node) && ui.style(node).display != Display::None {
            ui.update_style(node, |s| s.display = Display::None);
        }
    }

    /// The pressed letter, large over the finger (`feedback.preview`): its
    /// bubble made opaque, which is drawn without laying anything out.
    fn preview(&self, ui: &mut Ui, index: usize) {
        self.unlight_bubble(ui);
        if self.board.borrow().secret {
            // Not for whoever looks over a shoulder.
            return;
        }
        let bubble = self.board.borrow().keys.get(index).and_then(|k| k.bubble);
        if let Some((b, _)) = bubble {
            ui.update_style(b, |s| s.opacity = 1.0);
            self.board.borrow_mut().lit_bubble = Some(b);
        }
    }

    fn unlight_bubble(&self, ui: &mut Ui) {
        let lit = self.board.borrow_mut().lit_bubble.take();
        if let Some(b) = lit.filter(|&b| ui.contains(b)) {
            ui.update_style(b, |s| s.opacity = 0.0);
        }
    }

    /// Key `index` held: its alternates in a row over it, the first chosen,
    /// in the popup kept for them.
    fn offer_alternates(&self, ui: &mut Ui, index: usize) {
        let space = self
            .board
            .borrow()
            .keys
            .get(index)
            .is_some_and(|k| k.key.action == KeyAction::Text(" ".into()));
        if space {
            self.start_trackpad(ui, index);
            return;
        }
        let (face, options, theme) = {
            let b = self.board.borrow();
            if !b
                .press
                .as_ref()
                .is_some_and(|p| p.index == index && p.popup.is_none())
            {
                return;
            }
            let Some(k) = b.keys.get(index) else { return };
            let up = b.page == Page::Letters && b.shift != Shift::Off;
            let mut options: Vec<String> = if up {
                k.key.alternates.iter().map(|a| upper(a)).collect()
            } else {
                k.key.alternates.clone()
            };
            options.truncate(MOST_ALTERNATES);
            if options.is_empty() || !ui.contains(k.face) {
                return;
            }
            (k.face, options, b.active.theme)
        };
        self.unlight_bubble(ui);
        let appearance = drawn_for(ui);
        let (r, panel) = (ui.rect(face), ui.rect(self.panel));
        let w = r.width.max(28.0);
        let total = w * options.len() as f32 + 12.0;
        let left = (r.x - panel.x - 6.0).min(panel.width - total).max(0.0);
        ui.update_style(self.popup, |s| {
            s.display = Display::Flex;
            s.left = Length::Px(left);
            s.top = Length::Px(r.y - panel.y - r.height - 18.0);
            s.border_radius = Length::Px(theme.key_radius + 2.0);
            s.box_shadow = vec![BoxShadow {
                x: 0.0,
                y: 2.0,
                blur: 10.0,
                spread: 0.0,
                color: [0.0, 0.0, 0.0, 0.3],
            }];
        });
        paint(ui, self.popup, "background", theme.key, appearance);
        paint(ui, self.popup, "color", theme.key_text, appearance);
        for (i, &(o, text)) in self.options.iter().enumerate() {
            match options.get(i) {
                Some(t) => {
                    ui.update_style(o, |s| {
                        s.display = Display::Flex;
                        s.width = Length::Px(w);
                        s.height = Length::Px(r.height + 6.0);
                        s.border_radius = Length::Px(theme.key_radius);
                        s.font_size = theme.letter_size;
                    });
                    if text_of(ui, text) != *t {
                        ui.set_text(text, t.clone());
                    }
                }
                None => self.hide(ui, o),
            }
        }
        let all: Vec<usize> = (0..options.len()).collect();
        if let Some(p) = self.board.borrow_mut().press.as_mut() {
            p.popup = Some(Popup { options, chosen: 0 });
        }
        self.light_options(ui, &all, 0);
        // Felt, not heard, as the alternates come up.
        let key = self.board.borrow().keys.get(index).map(|k| k.key.clone());
        if let Some(key) = key {
            self.emit(&key, false, true, true);
        }
    }

    /// The space bar held: the keys fade, and sliding moves the caret.
    fn start_trackpad(&self, ui: &mut Ui, index: usize) {
        if !ui.contains(self.body) {
            return;
        }
        let key = {
            let mut b = self.board.borrow_mut();
            let Some(press) = b.press.as_mut().filter(|p| p.index == index) else {
                return;
            };
            press.trackpad = Some(press.at);
            b.keys.get(index).map(|k| k.key.clone())
        };
        self.unlight_bubble(ui);
        // Opacity alone: drawn, not laid out.
        ui.update_style(self.body, |s| s.opacity = 0.35);
        if let Some(key) = key {
            self.emit(&key, false, true, true);
        }
    }

    /// The alternates `which` lit if chosen, plain if not.
    fn light_options(&self, ui: &mut Ui, which: &[usize], chosen: usize) {
        let appearance = drawn_for(ui);
        let theme = self.board.borrow().active.theme;
        for &i in which {
            let Some(&(n, _)) = self.options.get(i) else {
                continue;
            };
            if i == chosen {
                paint(ui, n, "background", theme.accent_key, appearance);
                paint(ui, n, "color", theme.accent_text, appearance);
            } else {
                ui.clear_token_style(n, "background");
                ui.update_style(n, |s| s.background = None);
                paint(ui, n, "color", theme.key_text, appearance);
            }
        }
    }

    /// A floating keyboard dragged by its handle, kept in the window.
    fn drag(&self, ui: &mut Ui, e: &Event, cx: &mut EventCx) {
        match *e {
            Event::PointerDown { x, y, .. } => {
                cx.prevent_default();
                cx.capture_pointer();
                self.board.borrow_mut().drag = Some((x, y));
            }
            Event::PointerMove { x, y } => {
                let Some((px, py)) = self.board.borrow().drag else {
                    return;
                };
                if !x.is_finite() || !y.is_finite() {
                    return;
                }
                let (sheet, panel, root) =
                    (ui.rect(self.sheet), ui.rect(self.panel), ui.rect(ui.root()));
                let (ox, oy) = self.board.borrow().offset;
                // Where it would be undragged, and how far it may go.
                let (x0, y0) = (sheet.x - ox, sheet.y - oy);
                // (in a window smaller than it, its top left is what shows)
                let within = |v: f32, least: f32, most: f32| v.min(most).max(least);
                let nx = within(ox + x - px, panel.x - x0, panel.x + panel.width - sheet.width - x0);
                let ny = within(oy + y - py, root.y - y0, panel.y + panel.height - sheet.height - y0);
                {
                    let mut b = self.board.borrow_mut();
                    b.offset = (nx, ny);
                    b.drag = Some((x, y));
                }
                ui.update_style(self.sheet, |s| {
                    s.left = Length::Px(nx);
                    s.top = Length::Px(ny);
                });
            }
            Event::PointerUp { .. } | Event::PointerCancel => {
                cx.release_pointer();
                self.board.borrow_mut().drag = None;
            }
            _ => {}
        }
    }

    /// Turns it on or off. On, it shows while a text control has focus, and
    /// a native layer ([`native::controls`](crate::native::controls)) puts
    /// no platform keyboard's field over the UI's text controls: they are
    /// pressed to be typed into here.
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
        self.board.borrow().layout()
    }

    /// The keyboard every field without one of its own gets, wherever it
    /// runs (the device no longer chosen by the window).
    pub fn set_config(&self, config: KeyboardConfig) {
        let mut b = self.board.borrow_mut();
        b.default = config;
        b.device = None;
        b.refresh = true;
    }

    /// The keyboards for `device` from now on ([`Device::Auto`]: by the
    /// window's size).
    pub fn set_device(&self, device: Device) {
        let mut b = self.board.borrow_mut();
        b.device = Some(device);
        b.refresh = true;
    }

    /// The device the keyboard is shaped for now (`None` before it first
    /// shows, or when [`set_config`](Self::set_config) gave one config for
    /// all).
    pub fn device(&self) -> Option<Device> {
        let b = self.board.borrow();
        b.device.map(|_| b.resolved)
    }

    /// The keyboard showing now (or that would, for a field without its
    /// own).
    pub fn config(&self) -> KeyboardConfig {
        self.board.borrow().active.clone()
    }

    /// Changes every keyboard it shows — for any device, any field's type —
    /// with `f`: sound off, a theme, a row of shortcuts, everywhere.
    ///
    /// ```
    /// # use threers_ui::{input::Input, Ui};
    /// # use threers_ui::virtual_keyboard::{KeyboardTheme, VirtualKeyboard};
    /// # let (mut ui, mut input) = (Ui::new(), Input::<()>::new());
    /// let kb = VirtualKeyboard::new(&mut ui, &mut input);
    /// kb.customize(|c| {
    ///     c.feedback.sound = false;
    ///     c.theme = KeyboardTheme::system();
    /// });
    /// ```
    pub fn customize(&self, f: impl Fn(&mut KeyboardConfig) + 'static) {
        let mut b = self.board.borrow_mut();
        b.tweaks.push(Rc::new(f));
        b.refresh = true;
    }

    /// The keyboard `field` gets while it has focus (`None`: the default):
    /// a config of your own. For the usual kinds of field,
    /// [`set_type_for`](Self::set_type_for) is shorter and follows the
    /// device.
    pub fn set_config_for(&self, field: NodeId, config: Option<KeyboardConfig>) {
        self.set_field(field, config.map(|c| FieldKeyboard::Config(Box::new(c))));
    }

    /// What `field` takes — an email address, a PIN, a price — and so the
    /// keyboard it gets, shaped for the device.
    ///
    /// ```
    /// # use threers_ui::{input::Input, Ui};
    /// # use threers_ui::virtual_keyboard::{KeyboardType, VirtualKeyboard};
    /// # let (mut ui, mut input) = (Ui::new(), Input::<()>::new());
    /// # let (email, pin) = (ui.add_box(ui.root(), Default::default()), ui.add_box(ui.root(), Default::default()));
    /// let kb = VirtualKeyboard::new(&mut ui, &mut input);
    /// kb.set_type_for(email, KeyboardType::Email);
    /// kb.set_type_for(pin, KeyboardType::NumberPad);
    /// ```
    pub fn set_type_for(&self, field: NodeId, kind: KeyboardType) {
        self.set_field(field, Some(FieldKeyboard::Type(kind)));
    }

    fn set_field(&self, field: NodeId, keyboard: Option<FieldKeyboard>) {
        let mut b = self.board.borrow_mut();
        b.per_field.retain(|(n, _)| *n != field);
        if let Some(k) = keyboard {
            b.per_field.push((field, k));
        }
        // Taken up on the next dispatch, if it has focus.
        b.refresh = true;
    }

    /// Calls `f` for every key pressed (and Backspace again while held):
    /// the click to play, the tap to give. For your own audio or haptics;
    /// [`play_clicks`](Self::play_clicks) plays the clicks itself.
    ///
    /// ```
    /// # use threers_ui::{input::Input, Ui};
    /// # use threers_ui::virtual_keyboard::{Haptic, VirtualKeyboard};
    /// # let (mut ui, mut input) = (Ui::new(), Input::<()>::new());
    /// let kb = VirtualKeyboard::new(&mut ui, &mut input);
    /// kb.on_feedback(|press| {
    ///     if let Some(sound) = press.sound {
    ///         let _samples = sound.samples(48_000); // to your mixer
    ///     }
    ///     if press.haptic == Some(Haptic::Light) { /* a light tap */ }
    /// });
    /// ```
    pub fn on_feedback(&self, f: impl Fn(&KeyPress) + 'static) {
        self.board.borrow_mut().feedback.push(Rc::new(f));
    }

    /// Plays the key clicks on the default audio device (the `sound`
    /// feature), as [`KeyFeedback::sound`] and its volume say.
    ///
    /// # Errors
    ///
    /// If there is no device, or it will not open.
    #[cfg(all(feature = "sound", not(target_arch = "wasm32")))]
    pub fn play_clicks(&self) -> Result<(), String> {
        let player = Rc::new(ClickPlayer::open()?);
        self.on_feedback(move |p| {
            if let Some(s) = p.sound {
                player.play(s, p.volume);
            }
        });
        Ok(())
    }

    /// Calls `f` with the id of an action key ([`Key::action`],
    /// `{action:id:label}`) when it is pressed.
    pub fn on_action(&self, f: impl Fn(&str, &mut Ui) + 'static) {
        self.board.borrow_mut().actions.push(Rc::new(f));
    }

    /// How to read the text before the caret of the focused field from the
    /// input's state: sentences then start with a capital
    /// ([`KeyboardConfig::auto_capitalize`]), the period shortcut knows the
    /// field's text, and [`set_suggestions`](Self::set_suggestions) has a
    /// word to suggest for. `S` is the state the [`Input`] dispatches to.
    ///
    /// ```
    /// # use threers_ui::{input::Input, TextField, Ui};
    /// # use threers_ui::virtual_keyboard::VirtualKeyboard;
    /// # let (mut ui, mut input) = (Ui::new(), Input::<TextField>::new());
    /// let kb = VirtualKeyboard::new(&mut ui, &mut input);
    /// kb.set_context(|field: &TextField, _node| Some(field.before_caret()));
    /// ```
    pub fn set_context<S: 'static>(&self, f: impl Fn(&S, NodeId) -> Option<String> + 'static) {
        let reader: ContextReader<S> = Box::new(f);
        let mut b = self.board.borrow_mut();
        b.context = Some(Rc::new(reader));
        b.before = None;
    }

    /// Words to offer for the one being typed, up to three, shown in the
    /// bar over the keys; a tap puts one in its place with a space. Needs
    /// [`set_context`](Self::set_context) to know the word.
    ///
    /// ```
    /// # use threers_ui::{input::Input, TextField, Ui};
    /// # use threers_ui::virtual_keyboard::VirtualKeyboard;
    /// # let (mut ui, mut input) = (Ui::new(), Input::<TextField>::new());
    /// let words = ["hello", "help", "helmet"];
    /// let kb = VirtualKeyboard::new(&mut ui, &mut input);
    /// kb.set_context(|f: &TextField, _| Some(f.before_caret()));
    /// kb.set_suggestions(move |typed| {
    ///     words.iter().filter(|w| w.starts_with(typed)).map(|w| w.to_string()).collect()
    /// });
    /// ```
    pub fn set_suggestions(&self, f: impl Fn(&str) -> Vec<String> + 'static) {
        let mut b = self.board.borrow_mut();
        b.suggest = Some(Rc::new(f));
        b.before = None;
        b.built = None;
    }

    /// Words for the bar from the text before the caret, as a next-word
    /// model reads it: `f(language, before)` gives completions of the word
    /// being typed, or the next word after a space, up to three, best
    /// first. Asked ahead of [`set_suggestions`](Self::set_suggestions);
    /// the layout's language tag (`en`, `de`, …) comes first. Without
    /// [`set_context`](Self::set_context), `before` is what the keyboard
    /// typed lately. `threers-swipe` makes such a predictor from its
    /// next-word models.
    ///
    /// ```
    /// # use threers_ui::{input::Input, TextField, Ui};
    /// # use threers_ui::virtual_keyboard::VirtualKeyboard;
    /// # let (mut ui, mut input) = (Ui::new(), Input::<TextField>::new());
    /// let kb = VirtualKeyboard::new(&mut ui, &mut input);
    /// kb.set_context(|f: &TextField, _| Some(f.before_caret()));
    /// kb.set_predictions(|_language, before| {
    ///     if before.ends_with("going ") { vec!["to".into()] } else { vec![] }
    /// });
    /// ```
    pub fn set_predictions(&self, f: impl Fn(&str, &str) -> Vec<String> + 'static) {
        let mut b = self.board.borrow_mut();
        b.predict = Some(Rc::new(f));
        b.before = None;
        b.built = None;
    }

    /// Stops [`set_predictions`](Self::set_predictions)' words.
    pub fn clear_predictions(&self) {
        let mut b = self.board.borrow_mut();
        b.predict = None;
        b.before = None;
        b.built = None;
    }

    /// The finger's trail across the keys while it swipes, painted over
    /// what [`Ui::paint`] drew: a ribbon thickening and strengthening toward
    /// the finger (strokes stacked, each from further along), its last [`TRAIL_SECONDS`] showing, fading out over
    /// [`TRAIL_FADE`] once the finger lifts. `now` is the input's clock
    /// (what [`Input::tick`] was given). Whether it is still to be seen
    /// changing: frames are wanted while it is.
    ///
    /// ```ignore
    /// ui.paint(&mut painter);
    /// let fading = keyboard.paint_swipe_trail(&mut painter, now);
    /// ```
    #[cfg(feature = "swipe")]
    pub fn paint_swipe_trail(&self, painter: &mut dyn crate::Painter, now: f64) -> bool {
        let b = self.board.borrow();
        let live = b
            .press
            .as_ref()
            .filter(|p| p.swiping)
            .and_then(|p| p.path.as_ref())
            .map(|(path, _)| path.as_slice());
        let (path, fade) = match (live, &b.trail) {
            (Some(path), _) => (path, 1.0),
            (None, Some((path, lifted))) if now - lifted < TRAIL_FADE => {
                (path.as_slice(), 1.0 - ((now - lifted) / TRAIL_FADE) as f32)
            }
            _ => return false,
        };
        let newest = path.last().map_or(0.0, |p| p.t);
        let from = path
            .iter()
            .position(|p| newest - p.t <= TRAIL_SECONDS)
            .unwrap_or(0)
            .max(path.len().saturating_sub(TRAIL_POINTS));
        let pts = &path[from..];
        if pts.len() < 2 {
            return fade < 1.0;
        }
        // Round dots along it, near enough to be a stroke, larger toward the
        // finger: each costs its own few pixels, so a long swipe draws as
        // fast as a short one. (Strokes of all of it, stacked, worked out
        // every pixel round it against every piece of it, eight times over.)
        let length: f32 = pts.windows(2).map(|w| (w[1].x - w[0].x).hypot(w[1].y - w[0].y)).sum();
        // each place is under about this many of them
        const OVER: f32 = 4.0;
        let alpha = 1.0 - (1.0 - 0.8 * fade).powf(1.0 / OVER);
        let colour: crate::Rgba = [0.0, 0.48, 1.0, alpha];
        let size = |at: f32| 2.0 + 6.0 * at / length.max(1.0);
        let mut dot = |x: f32, y: f32, d: f32| {
            let r = crate::Rect { x: x - d / 2.0, y: y - d / 2.0, width: d, height: d };
            painter.fill_rounded_rect(r, d / 2.0, d / 2.0, colour);
        };
        let (mut along, mut next) = (0.0f32, 0.0f32);
        for w in pts.windows(2) {
            let (dx, dy) = (w[1].x - w[0].x, w[1].y - w[0].y);
            let l = dx.hypot(dy);
            while next <= along + l {
                let t = if l > 0.0 { (next - along) / l } else { 0.0 };
                let d = size(next);
                dot(w[0].x + dx * t, w[0].y + dy * t, d);
                next += (d / OVER).max(0.75);
            }
            along += l;
        }
        true
    }

    #[cfg(feature = "swipe")]
    /// Turns on gesture typing: a word drawn across the letter keys
    /// without lifting is handed to `f` (the path, and the keys as drawn),
    /// and the first word it gives is typed with a space after it, the
    /// next ones offered in the bar. `threers-swipe` makes such a decoder from a trained model.
    ///
    /// ```
    /// use threers_ui::input::Input;
    /// use threers_ui::virtual_keyboard::VirtualKeyboard;
    /// use threers_ui::Ui;
    ///
    /// let mut ui = Ui::new();
    /// let mut input: Input<()> = Input::new();
    /// let kb = VirtualKeyboard::new(&mut ui, &mut input);
    /// // Whatever was drawn, "hello"; a real decoder reads the path.
    /// kb.set_swipe_decoder(|swipe| {
    ///     assert!(swipe.keys.iter().any(|k| k.text == "h"));
    ///     vec!["hello".into(), "jello".into()]
    /// });
    /// ```
    pub fn set_swipe_decoder(&self, f: impl Fn(&Swipe) -> Vec<String> + 'static) {
        let mut b = self.board.borrow_mut();
        b.swipe = Some(Rc::new(f));
        self.swiping.set(true);
        // the letters made again: for strokes, not repeats
        b.generation += 1;
    }

    /// Turns gesture typing off: a slide types the key it ends on again.
    #[cfg(feature = "swipe")]
    pub fn clear_swipe_decoder(&self) {
        let mut b = self.board.borrow_mut();
        b.swipe = None;
        self.swiping.set(false);
        b.generation += 1;
    }

    /// Shows `page` (the letters, if the keyboard has no such page), as
    /// its page keys do.
    pub fn set_page(&self, page: Page) {
        self.board.borrow_mut().page = page;
    }

    /// The page showing.
    pub fn page(&self) -> Page {
        self.board.borrow().page
    }

    /// The page showing, drawn as text ([`KeyboardConfig::sketch`]): to
    /// see in a test or a log what is on it.
    pub fn sketch(&self) -> String {
        let b = self.board.borrow();
        b.active.sketch(b.page, &b.layout(), b.layouts.len())
    }

    /// Moves it, over what its config says, until set again (`None`: back
    /// to the config's), as its one-handed and dock keys do.
    pub fn set_placement(&self, placement: Option<Placement>) {
        let mut b = self.board.borrow_mut();
        b.placement = placement;
        b.offset = (0.0, 0.0);
        b.generation += 1;
    }

    /// Where it sits now.
    pub fn placement(&self) -> Placement {
        self.board.borrow().placement()
    }

    /// Whether it is showing now.
    pub fn showing(&self, ui: &Ui) -> bool {
        ui.contains(self.panel) && ui.style(self.panel).display != Display::None
    }

    /// Its panel (to measure, or keep content clear of).
    pub fn panel(&self) -> NodeId {
        self.panel
    }

    /// The keyboard itself, inside the panel (where it is drawn).
    pub fn sheet(&self) -> NodeId {
        self.sheet
    }

    /// Gives `field` (a text box, text area or search box, positioned) a
    /// keyboard button at its right end: pressed, focus goes to the field
    /// and this keyboard comes up for it, on or not, until focus leaves it.
    /// Returns the button.
    pub fn attach_button<S: 'static>(
        &self,
        ui: &mut Ui,
        input: &mut Input<S>,
        field: NodeId,
    ) -> NodeId {
        let b = ui.add_box(
            field,
            style! { position: absolute; right: 2px; top: 0px; bottom: 0px; width: 28px; display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 15px; },
        );
        ui.token_style(b, &[("color", Token::SecondaryText)]);
        ui.hover_style(b, "opacity: 0.7");
        ui.add_text(b, "⌨");
        ui.set_semantics(
            b,
            Semantics::new(Role::Button)
                .label("Show the keyboard")
                .focusable(false),
        );
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

    /// Puts it away, focus or not, until it's turned on (`set_enabled`) or
    /// summoned again: at the next dispatch or tick it goes, any key held
    /// let go (a held key stops repeating) and the room it took given back.
    pub fn dismiss(&self) {
        self.enabled.set(false);
        self.summoned.set(false);
    }

    /// Presses `key` (a character, `"Backspace"`, `"Enter"`, `" "`) as its
    /// key on the panel would: typed at what has focus.
    pub fn press<S: 'static>(&self, ui: &mut Ui, input: &mut Input<S>, state: &mut S, key: &str) {
        input.dispatch(
            ui,
            state,
            Event::Key {
                key: key.to_owned(),
                modifiers: KeyModifiers::default(),
            },
        );
    }
}
