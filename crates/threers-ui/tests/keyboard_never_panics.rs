//! The on-screen keyboard under whatever happens to it: fingers anywhere, a
//! window of any size, its config, layouts and placement changed while a key
//! is held, its field or the keyboard itself taken away.

use std::panic::{AssertUnwindSafe, catch_unwind};

use threers_ui::input::{Event, Input, PointerButton};
use threers_ui::virtual_keyboard::{
    Align, KeyboardConfig, KeyboardLayout, Page, Placement, ReturnKey, VirtualKeyboard,
};
use threers_ui::{FieldKey, FixedWidthText, NodeId, Role, Semantics, TextField, Ui, style};

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as usize % n.max(1)
    }
}

fn config(r: &mut Rng) -> KeyboardConfig {
    match r.below(8) {
        0 => KeyboardConfig::iphone(),
        1 => KeyboardConfig::ipad(),
        2 => KeyboardConfig::ipad_pro(),
        3 => KeyboardConfig::ipad_floating(),
        4 => KeyboardConfig::terminal(),
        5 => KeyboardConfig::phone_pad(),
        6 => KeyboardConfig::url(),
        _ => {
            let mut c = KeyboardConfig::number_pad();
            c.repeat_typing = true;
            c.return_key = ReturnKey::Search;
            c
        }
    }
}

fn placement(r: &mut Rng) -> Placement {
    Placement {
        floating: r.below(2) == 0,
        one_handed: r.below(3) == 0,
        split: [0.0, 80.0, 5000.0][r.below(3)],
        width: [None, Some(300.0), Some(2000.0)][r.below(3)],
        align: [Align::Left, Align::Center, Align::Right][r.below(3)],
        margin: [0.0, 12.0, 400.0][r.below(3)],
        ..Placement::docked()
    }
}

/// A run of `steps` random happenings; what was done, if it panicked.
fn run(seed: u64, steps: usize) -> Result<(), String> {
    let did = std::cell::RefCell::new(vec![]);
    catch_unwind(AssertUnwindSafe(|| {
        let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut ui = Ui::new();
        let mut input: Input<TextField> = Input::new();
        let root = ui.root();
        let field = |ui: &mut Ui, input: &mut Input<TextField>| -> NodeId {
            let n = ui.add_box(root, style! { height: 20px; });
            ui.set_semantics(n, Semantics::new(Role::TextBox).label("Name"));
            input.on_key(n, |f: &mut TextField, _, key, m, _| {
                !matches!(f.key(key, m, |_| true), FieldKey::Ignored)
            });
            n
        };
        let mut target = field(&mut ui, &mut input);
        let kb = VirtualKeyboard::with_config(&mut ui, &mut input, config(&mut r));
        kb.set_enabled(true);
        if r.below(2) == 0 {
            kb.set_context(|f: &TextField, _| Some(f.before_caret()));
            kb.set_suggestions(|w| (0..9).map(|i| format!("{w}{i}é")).collect());
        }
        #[cfg(feature = "swipe")]
        if r.below(2) == 0 {
            kb.set_swipe_decoder(|s| {
                if s.points.len() % 2 == 0 { vec![] } else { vec!["ünder".into(), String::new()] }
            });
        }
        let mut text = TextField::new("");
        let (mut w, mut h) = (400.0f32, 700.0f32);
        let mut t = 0.0;
        input.set_focus(&mut ui, &mut text, Some(target));
        for _ in 0..steps {
            let (x, y) = match r.below(14) {
                0 => (-50.0, 1e6),
                1 => (f32::NAN, 3.0),
                2 => (f32::INFINITY, f32::NEG_INFINITY),
                _ => (
                    r.below(w as usize + 1) as f32,
                    h - r.below((h as usize + 1).min(360)) as f32,
                ),
            };
            let what = r.below(30);
            did.borrow_mut().push((what, x, y, w, h));
            match what {
                0..=5 => {
                    input.dispatch(&mut ui, &mut text, Event::PointerDown { x, y, button: PointerButton::Primary });
                }
                6..=10 => {
                    input.dispatch(&mut ui, &mut text, Event::PointerMove { x, y });
                }
                11..=13 => {
                    input.dispatch(&mut ui, &mut text, Event::PointerUp { x, y, button: PointerButton::Primary });
                }
                14 => {
                    input.dispatch(&mut ui, &mut text, Event::PointerCancel);
                }
                15..=17 => t += [0.05, 0.6, 3.0][r.below(3)],
                18 => {
                    let sizes = [0.0, 1.0, 40.0, 250.0, 400.0, 900.0, 2400.0];
                    (w, h) = (sizes[r.below(7)], sizes[r.below(7)]);
                }
                19 => kb.set_config(config(&mut r)),
                20 => kb.set_placement((r.below(4) > 0).then(|| placement(&mut r))),
                21 => kb.set_layouts(
                    &mut ui,
                    (0..r.below(4)).map(|_| KeyboardLayout::english()).collect(),
                ),
                22 => kb.set_page([Page::Letters, Page::Symbols, Page::More][r.below(3)]),
                23 => {
                    input.set_focus(&mut ui, &mut text, (r.below(2) == 0).then_some(target));
                }
                24 => kb.set_enabled(r.below(3) > 0),
                25 => kb.dismiss(),
                26 if r.below(3) == 0 => {
                    ui.remove(target);
                    target = field(&mut ui, &mut input);
                    input.set_focus(&mut ui, &mut text, Some(target));
                }
                // The floating keyboard by its handle.
                27 => {
                    let s = ui.rect(kb.sheet());
                    let (hx, hy) = (s.x + s.width / 2.0, s.y + 8.0);
                    input.dispatch(&mut ui, &mut text, Event::PointerDown { x: hx, y: hy, button: PointerButton::Primary });
                    input.dispatch(&mut ui, &mut text, Event::PointerMove { x, y });
                    input.dispatch(&mut ui, &mut text, Event::PointerUp { x, y, button: PointerButton::Primary });
                }
                28 => {
                    kb.set_enabled(true);
                    kb.summon();
                    input.set_focus(&mut ui, &mut text, Some(target));
                }
                29 if r.below(40) == 0 && ui.contains(kb.panel()) => {
                    ui.remove(kb.panel());
                    return after(&mut ui, &mut input, &mut text, &kb, t);
                }
                _ => {}
            }
            t += 0.016;
            input.tick(&mut ui, &mut text, t);
            ui.step(0.016);
            ui.layout(w, h, &mut FixedWidthText(7.0));
            assert!(text.text.len() < 1_000_000, "typed without end");
            if text.text.len() > 500 {
                text = TextField::new("");
            }
        }
    }))
    .map_err(|e| {
        let why = e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()));
        let d = did.borrow();
        format!("seed {seed}: {why:?} after {:?}", &d[d.len().saturating_sub(6)..])
    })
}

/// The panel gone from the tree: events and timers still to come, and the
/// keyboard's own methods, do nothing.
fn after(ui: &mut Ui, input: &mut Input<TextField>, text: &mut TextField, kb: &VirtualKeyboard, mut t: f64) {
    for x in [10.0, 200.0] {
        for e in [
            Event::PointerDown { x, y: 600.0, button: PointerButton::Primary },
            Event::PointerMove { x: x + 50.0, y: 620.0 },
            Event::PointerUp { x, y: 600.0, button: PointerButton::Primary },
        ] {
            input.dispatch(ui, text, e);
            t += 0.7;
            input.tick(ui, text, t);
            ui.step(0.7);
            ui.layout(400.0, 700.0, &mut FixedWidthText(7.0));
        }
    }
    kb.set_page(Page::Symbols);
    kb.dismiss();
    assert!(!kb.showing(ui));
}

#[test]
fn whatever_happens_to_the_keyboard_it_does_not_panic() {
    let runs = std::env::var("KEYBOARD_RUNS").ok().and_then(|n| n.parse().ok()).unwrap_or(20);
    let bad: Vec<String> = (1..=runs).filter_map(|seed| run(seed, 300).err()).collect();
    assert!(bad.is_empty(), "{} of {runs}:\n{}", bad.len(), bad[..bad.len().min(6)].join("\n"));
}

#[test]
fn a_tweak_may_ask_the_keyboard_something() {
    let mut ui = Ui::new();
    let mut input: Input<TextField> = Input::new();
    let root = ui.root();
    let field = ui.add_box(root, style! { height: 20px; });
    ui.set_semantics(field, Semantics::new(Role::TextBox).label("Name"));
    input.on_key(field, |_: &mut TextField, _, _, _, _| true);
    let kb = VirtualKeyboard::with_config(&mut ui, &mut input, KeyboardConfig::iphone());
    kb.set_enabled(true);
    let me = kb.clone();
    kb.customize(move |c| c.period_shortcut = me.page() != Page::Letters);
    let mut text = TextField::new("");
    input.set_focus(&mut ui, &mut text, Some(field));
    input.tick(&mut ui, &mut text, 0.1);
    assert!(kb.showing(&ui) && !kb.config().period_shortcut);
}
