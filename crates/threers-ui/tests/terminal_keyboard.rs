//! The terminal keyboard (`KeyboardConfig::terminal`): ctrl and alt held
//! for the next key, every key repeating while held, and the row of keys a
//! shell needs over every page.

use threers_ui::input::{Event, Input, PointerButton};
use threers_ui::virtual_keyboard::{KeyboardConfig, KeyboardLayout, VirtualKeyboard};
use threers_ui::{style, FixedWidthText, KeyModifiers, NodeId, Role, Semantics, Ui};

type Typed = Vec<(String, KeyModifiers)>;

struct Rig {
    ui: Ui,
    input: Input<Typed>,
    typed: Typed,
    _kb: VirtualKeyboard,
    t: f64,
    size: (f32, f32),
}

impl Rig {
    fn new() -> Self {
        Self::with(KeyboardConfig::terminal(), 400.0, 700.0)
    }

    fn with(config: KeyboardConfig, w: f32, h: f32) -> Self {
        let mut ui = Ui::new();
        let mut input: Input<Typed> = Input::new();
        let root = ui.root();
        let target = ui.add_box(root, style! { height: 20px; });
        ui.set_semantics(target, Semantics::new(Role::TextBox).label("Terminal"));
        input.on_key(target, |t: &mut Typed, _, key, m, _| {
            t.push((key.to_owned(), m));
            true
        });
        let kb = VirtualKeyboard::with_config(&mut ui, &mut input, config);
        kb.set_enabled(true);
        kb.set_layouts(&mut ui, vec![KeyboardLayout::english()]);
        let mut r = Rig { ui, input, typed: Vec::new(), _kb: kb, t: 0.0, size: (w, h) };
        r.input.set_focus(&mut r.ui, &mut r.typed, Some(target));
        r.settle();
        r
    }

    fn settle(&mut self) {
        for _ in 0..3 {
            self.tick(0.0);
            self.ui.step(0.3);
            self.ui.layout(self.size.0, self.size.1, &mut FixedWidthText(7.0));
        }
    }

    fn tick(&mut self, dt: f64) {
        self.t += dt;
        self.input.tick(&mut self.ui, &mut self.typed, self.t);
        self.ui.step(dt as f32);
    }

    fn key(&self, name: &str) -> NodeId {
        let tree = self.ui.accessibility_tree();
        tree.iter().find(|a| a.name == name).map(|a| a.id).unwrap_or_else(|| {
            panic!("no {name}: {:?}", tree.iter().filter(|a| a.role == Role::Button).map(|a| &a.name).collect::<Vec<_>>())
        })
    }

    fn center(&self, name: &str) -> (f32, f32) {
        let r = self.ui.rect(self.key(name));
        (r.x + r.width / 2.0, r.y + r.height / 2.0)
    }

    fn send(&mut self, e: Event) {
        self.input.dispatch(&mut self.ui, &mut self.typed, e);
        self.ui.layout(self.size.0, self.size.1, &mut FixedWidthText(7.0));
    }

    fn down(&mut self, name: &str) -> (f32, f32) {
        let (x, y) = self.center(name);
        self.send(Event::PointerMove { x, y });
        self.send(Event::PointerDown { x, y, button: PointerButton::Primary });
        (x, y)
    }

    fn up(&mut self, (x, y): (f32, f32)) {
        self.send(Event::PointerUp { x, y, button: PointerButton::Primary });
        self.settle();
    }

    fn tap(&mut self, name: &str) {
        let at = self.down(name);
        self.up(at);
    }
}

#[test]
fn ctrl_is_held_for_the_next_key_only() {
    let mut r = Rig::new();
    r.tap("ctrl");
    r.tap("c");
    r.tap("d");
    let ctrl = KeyModifiers { control: true, ..Default::default() };
    assert_eq!(r.typed, [("c".to_owned(), ctrl), ("d".to_owned(), KeyModifiers::default())]);
    // pressed twice: let go again, nothing held
    r.tap("alt");
    r.tap("alt");
    r.tap("x");
    assert_eq!(r.typed.last().unwrap().1, KeyModifiers::default());
}

#[test]
fn a_held_letter_repeats_and_stops_when_let_go() {
    let mut r = Rig::new();
    let at = r.down("a");
    for _ in 0..12 {
        r.tick(0.09);
    }
    let held = r.typed.len();
    assert!(held >= 4, "repeated while held: {:?}", r.typed);
    assert!(r.typed.iter().all(|(k, _)| k == "a"));
    r.up(at);
    for _ in 0..10 {
        r.tick(0.09);
    }
    assert!(r.typed.len() <= held + 1, "stops when let go");
}

#[test]
fn a_shells_keys_are_over_every_page() {
    let mut r = Rig::new();
    for k in ["esc", "ctrl", "alt", "←", "↑", "↓", "→", "|", "~"] {
        r.key(k);
    }
    r.tap("esc");
    r.tap("↑");
    let keys: Vec<&str> = r.typed.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["Escape", "ArrowUp"]);
}

#[test]
fn a_key_still_down_when_the_next_comes_is_typed_first() {
    let mut r = Rig::new();
    // two thumbs: s down, then d down before s lifts
    r.down("s");
    let d = r.down("d");
    r.up(d);
    let keys: Vec<&str> = r.typed.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["s", "d"]);
}


#[test]
fn a_tablets_terminal_keyboard_is_a_laptops() {
    let mut r = Rig::with(KeyboardConfig::terminal_tablet(), 1024.0, 768.0);
    // one page: the number row, ctrl where caps lock is, esc/alt/arrows below
    for k in ["`", "1", "0", "=", "Tab", "[", "\\", "ctrl", ";", "'", "Shift", "/", "Escape", "alt", "ArrowLeft", "ArrowRight"] {
        r.key(k);
    }
    let (c, a) = (r.ui.rect(r.key("ctrl")), r.ui.rect(r.key("a")));
    assert!((c.y - a.y).abs() < 1.0 && c.x < a.x, "ctrl just left of a, in its row");
    // shift for the number row's other half
    r.tap("Shift");
    r.tap("4");
    r.tap("ctrl");
    r.tap("c");
    let typed: Vec<(&str, bool)> = r.typed.iter().map(|(k, m)| (k.as_str(), m.control)).collect();
    assert_eq!(typed, [("$", false), ("c", true)]);
}

#[test]
fn a_phones_symbols_are_a_shells() {
    let mut r = Rig::new();
    r.tap("Numbers and symbols");
    for k in ["\\", "|", "&", "$", "_", "="] {
        r.key(k);
    }
    r.tap("|");
    assert_eq!(r.typed.last().map(|(k, _)| k.as_str()), Some("|"));
}
