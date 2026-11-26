//! The on-screen keyboard as Apple's behave: its rows' geometry, typing on
//! release, holding for accents and for Backspace, Shift twice for caps
//! lock, its pages, a field's own keyboard, and where it sits.

use threers_ui::input::{Event, Input, PointerButton};
use threers_ui::virtual_keyboard::{
    Align, Key, KeyboardConfig, KeyboardLayout, Placement, ReturnKey, VirtualKeyboard,
};
use threers_ui::{style, FieldKey, FixedWidthText, NodeId, Rect, Role, Semantics, TextField, Ui};

const W: f32 = 400.0;
const H: f32 = 700.0;

struct Rig {
    ui: Ui,
    input: Input<TextField>,
    field: TextField,
    kb: VirtualKeyboard,
    target: NodeId,
    t: f64,
    size: (f32, f32),
}

impl Rig {
    fn new(config: KeyboardConfig) -> Self {
        Self::sized(config, W, H)
    }

    fn sized(config: KeyboardConfig, w: f32, h: f32) -> Self {
        Self::made(w, h, move |ui, input| {
            VirtualKeyboard::with_config(ui, input, config)
        })
    }

    /// With the keyboard `make` makes.
    fn made(
        w: f32,
        h: f32,
        make: impl FnOnce(&mut Ui, &mut Input<TextField>) -> VirtualKeyboard,
    ) -> Self {
        let mut ui = Ui::new();
        let mut input: Input<TextField> = Input::new();
        let root = ui.root();
        let target = ui.add_box(root, style! { height: 20px; });
        ui.set_semantics(target, Semantics::new(Role::TextBox).label("Name"));
        input.on_key(target, |f: &mut TextField, _, key, m, _| {
            !matches!(f.key(key, m, |_| true), FieldKey::Ignored)
        });
        let kb = make(&mut ui, &mut input);
        kb.set_enabled(true);
        kb.set_layouts(&mut ui, vec![KeyboardLayout::english()]);
        let mut r = Self {
            ui,
            input,
            field: TextField::new(""),
            kb,
            target,
            t: 0.0,
            size: (w, h),
        };
        r.input.set_focus(&mut r.ui, &mut r.field, Some(target));
        r.settle_sized(w, h);
        r
    }

    fn settle_sized(&mut self, w: f32, h: f32) {
        self.size = (w, h);
        for _ in 0..3 {
            self.tick(0.0);
            // Transitions finish; the input's clock (its timers) stays.
            self.ui.step(0.3);
            self.ui.layout(w, h, &mut FixedWidthText(7.0));
        }
    }

    fn settle(&mut self) {
        let (w, h) = self.size;
        self.settle_sized(w, h);
    }

    fn relayout(&mut self) {
        let (w, h) = self.size;
        self.ui.layout(w, h, &mut FixedWidthText(7.0));
    }

    fn tick(&mut self, dt: f64) {
        self.t += dt;
        self.input.tick(&mut self.ui, &mut self.field, self.t);
        self.ui.step(dt as f32);
    }

    fn named(&self, name: &str) -> NodeId {
        self.ui
            .accessibility_tree()
            .into_iter()
            .find(|a| a.name == name)
            .unwrap_or_else(|| panic!("no {name}: {:?}", self.names()))
            .id
    }

    fn all_named(&self, name: &str) -> Vec<NodeId> {
        self.ui
            .accessibility_tree()
            .into_iter()
            .filter(|a| a.name == name)
            .map(|a| a.id)
            .collect()
    }

    fn names(&self) -> Vec<String> {
        self.ui
            .accessibility_tree()
            .into_iter()
            .filter(|a| a.role == Role::Button)
            .map(|a| a.name)
            .collect()
    }

    fn rect(&self, name: &str) -> Rect {
        self.ui.rect(self.named(name))
    }

    fn send(&mut self, e: Event) {
        self.input.dispatch(&mut self.ui, &mut self.field, e);
        self.relayout();
    }

    fn down_at(&mut self, (x, y): (f32, f32)) {
        self.send(Event::PointerMove { x, y });
        let b = PointerButton::Primary;
        self.send(Event::PointerDown { x, y, button: b });
    }

    fn up_at(&mut self, (x, y): (f32, f32)) {
        let b = PointerButton::Primary;
        self.send(Event::PointerUp { x, y, button: b });
        self.settle();
    }

    fn center(&self, name: &str) -> (f32, f32) {
        let r = self.rect(name);
        (r.x + r.width / 2.0, r.y + r.height / 2.0)
    }

    fn tap(&mut self, name: &str) {
        let at = self.center(name);
        self.down_at(at);
        self.up_at(at);
    }
}

#[test]
fn a_short_row_has_keys_as_wide_as_a_long_one_centred_under_it() {
    let r = Rig::new(KeyboardConfig::iphone());
    let (q, p, a, l) = (r.rect("q"), r.rect("p"), r.rect("a"), r.rect("l"));
    assert!((q.width - a.width).abs() < 0.6, "{q:?} {a:?}");
    // Nine under ten: half a key in from each end.
    let half = (q.width + 6.0) / 2.0;
    assert!((a.x - q.x - half).abs() < 1.0, "{q:?} {a:?}");
    assert!(((p.x + p.width) - (l.x + l.width) - half).abs() < 1.0);
    // Shift and Backspace by the third row, wider than a letter.
    let (shift, z, back) = (r.rect("Shift"), r.rect("z"), r.rect("Delete"));
    assert!(shift.width > q.width * 1.2 && back.width > q.width * 1.2);
    assert!(shift.x < z.x && (shift.y - z.y).abs() < 0.5);
    // The space bar takes what its row leaves.
    let space = r.rect("Space");
    assert!(space.width > q.width * 4.0, "{space:?}");
}

#[test]
fn a_letter_types_when_the_finger_lifts_and_the_last_key_under_it_wins() {
    let mut r = Rig::new(KeyboardConfig::iphone());
    let (q, w) = (r.center("q"), r.center("w"));
    r.down_at(q);
    assert_eq!(r.field.text, "", "nothing typed on the press");
    // The key and its label, as before it was pressed: the preview is
    // hidden from assistive technology.
    assert_eq!(r.all_named("q").len(), 2, "the preview takes no name");
    // Slid over to w before lifting.
    r.send(Event::PointerMove { x: w.0, y: w.1 });
    r.up_at(w);
    assert_eq!(r.field.text, "w");
}

#[test]
fn holding_a_letter_offers_its_accents_and_sliding_picks_one() {
    let mut r = Rig::new(KeyboardConfig::iphone());
    let e = r.center("e");
    r.down_at(e);
    for _ in 0..6 {
        r.tick(0.1);
    }
    r.relayout();
    // Slide right: è é ê … — the second.
    let key = r.rect("e");
    let to = (e.0 + key.width.max(28.0) * 1.0, e.1 - key.height);
    r.send(Event::PointerMove { x: to.0, y: to.1 });
    r.up_at(to);
    assert_eq!(r.field.text, "é");
    // A quick tap types the letter itself.
    r.tap("e");
    assert_eq!(r.field.text, "ée");
}

#[test]
fn backspace_repeats_while_held() {
    let mut r = Rig::new(KeyboardConfig::iphone());
    r.field = TextField::new("abcdefghij");
    let back = r.center("Delete");
    r.down_at(back);
    r.tick(0.0);
    assert_eq!(r.field.text, "abcdefghi", "once on the press");
    for _ in 0..10 {
        r.tick(0.09);
    }
    let held = r.field.text.len();
    assert!(
        held < 9 && held > 0,
        "repeated while held: {:?}",
        r.field.text
    );
    r.up_at(back);
    for _ in 0..10 {
        r.tick(0.09);
    }
    assert_eq!(r.field.text.len(), held, "and stops when let go");
}

#[test]
fn shift_once_is_for_one_letter_and_twice_quickly_locks() {
    let mut r = Rig::new(KeyboardConfig::iphone());
    r.tap("Shift");
    r.tap("A");
    r.tap("b");
    assert_eq!(r.field.text, "Ab");
    r.tap("Shift");
    r.tap("Shift");
    r.tap("C");
    r.tap("D");
    assert_eq!(r.field.text, "AbCD", "locked");
    r.tap("Shift");
    r.tap("e");
    assert_eq!(r.field.text, "AbCDe");
    // Two taps apart are on, then off.
    r.tap("Shift");
    r.tick(0.5);
    r.tap("Shift");
    r.tap("f");
    assert_eq!(r.field.text, "AbCDef");
}

#[test]
fn the_pages_are_ios_s_and_the_globe_shows_for_two_languages() {
    let mut r = Rig::new(KeyboardConfig::iphone());
    assert!(r.all_named("Next keyboard").is_empty(), "one language");
    r.tap("Numbers and symbols");
    for k in ["1", "0", "@", "\"", "More symbols", "Letters"] {
        r.named(k);
    }
    r.tap("1");
    r.tap("More symbols");
    for k in ["[", "€", "•"] {
        r.named(k);
    }
    r.tap("€");
    r.tap("Letters");
    r.named("q");
    assert_eq!(r.field.text, "1€");
    r.kb.set_layouts(
        &mut r.ui,
        vec![KeyboardLayout::english(), KeyboardLayout::german()],
    );
    r.settle();
    r.tap("Next keyboard");
    r.named("ü");
}

#[test]
fn return_says_what_the_field_does() {
    let r = Rig::new(KeyboardConfig::iphone().return_key(ReturnKey::Search));
    let ret = r.named("Return");
    let label = r.ui.children(ret).to_vec();
    assert!(label
        .iter()
        .any(|&n| matches!(r.ui.kind(n), threers_ui::NodeKind::Text(t) if t == "search")));
    assert_ne!(
        r.ui.style(ret).background,
        r.ui.style(r.named("Shift")).background,
        "search in the accent"
    );
}

#[test]
fn an_ipad_s_dual_keys_type_their_second_under_shift() {
    let mut r = Rig::sized(KeyboardConfig::ipad(), 820.0, 1180.0);
    assert_eq!(r.all_named("Shift").len(), 2, "both ends");
    r.named("Hide keyboard");
    let comma = r.center(",");
    r.down_at(comma);
    r.up_at(comma);
    r.tap("Shift");
    let comma = r.center(",");
    r.down_at(comma);
    r.up_at(comma);
    assert_eq!(r.field.text, ",!");
    r.tap("Hide keyboard");
    assert_eq!(r.input.focused(), None);
}

#[test]
fn an_ipad_pro_has_a_number_row_tab_and_caps_lock() {
    let r = Rig::sized(KeyboardConfig::ipad_pro(), 1024.0, 1366.0);
    for k in [
        "`",
        "1",
        "0",
        "=",
        "Tab",
        "Caps Lock",
        "[",
        "\\",
        ";",
        "'",
        "/",
    ] {
        r.named(k);
    }
    let (one, q) = (r.rect("1"), r.rect("q"));
    assert!(one.y < q.y && (one.width - q.width).abs() < 0.6);
}

#[test]
fn a_field_can_have_its_own_keyboard() {
    let mut r = Rig::new(KeyboardConfig::iphone());
    let pin = r.ui.add_box(r.ui.root(), style! { height: 20px; });
    r.ui.set_semantics(pin, Semantics::new(Role::TextBox).label("PIN"));
    r.input.on_key(pin, |f: &mut TextField, _, key, m, _| {
        !matches!(f.key(key, m, |_| true), FieldKey::Ignored)
    });
    r.kb.set_config_for(pin, Some(KeyboardConfig::number_pad()));
    r.input.set_focus(&mut r.ui, &mut r.field, Some(pin));
    r.settle();
    assert!(r.all_named("q").is_empty(), "no letters");
    for k in ["1", "2", "5", "9"] {
        r.tap(k);
    }
    assert_eq!(r.field.text, "1259");
    assert_eq!(
        (r.rect("1").width / r.rect("2").width).round(),
        1.0,
        "three equal columns"
    );
    let target = r.target;
    r.input.set_focus(&mut r.ui, &mut r.field, Some(target));
    r.settle();
    r.named("q");
}

#[test]
fn one_handed_it_keeps_to_a_side_with_a_way_back() {
    let mut r = Rig::new(KeyboardConfig::iphone());
    let full = r.rect("q");
    r.kb.set_placement(Some(Placement::one_handed(Align::Left)));
    r.settle();
    let (q, p) = (r.rect("q"), r.rect("p"));
    assert!(q.width < full.width, "narrower: {q:?} {full:?}");
    assert!(p.x + p.width < W - 50.0, "room on the right: {p:?}");
    r.tap("Full width keyboard");
    let q = r.rect("q");
    assert!((q.width - full.width).abs() < 0.6, "back across");
    r.kb.set_placement(Some(Placement::one_handed(Align::Right)));
    r.settle();
    assert!(r.rect("q").x > 50.0, "keys on the right");
}

#[test]
fn anchored_and_floating_keyboards_sit_where_asked() {
    let mut r = Rig::sized(KeyboardConfig::iphone(), 1000.0, 700.0);
    r.kb.set_placement(Some(Placement::anchored(Align::Right, 420.0)));
    r.settle_sized(1000.0, 700.0);
    let sheet = r.ui.rect(r.kb.sheet());
    assert!((sheet.width - 420.0).abs() < 0.5 && (sheet.x + sheet.width - 1000.0).abs() < 0.5);
    assert!(
        r.ui.style(r.ui.root()).padding.bottom > 100.0,
        "the page made room"
    );

    r.kb.set_placement(Some(Placement::floating(320.0)));
    r.settle_sized(1000.0, 700.0);
    let sheet = r.ui.rect(r.kb.sheet());
    assert!((sheet.width - 320.0).abs() < 0.5);
    assert!(
        (sheet.x + sheet.width / 2.0 - 500.0).abs() < 1.0,
        "in the middle"
    );
    assert!(sheet.y + sheet.height < 700.0 - 5.0, "off the bottom");
    assert_eq!(r.ui.style(r.ui.root()).padding.bottom, 0.0, "over the page");
    // Dragged by its handle, kept on screen.
    let h = r.rect("Move keyboard");
    let at = (h.x + h.width / 2.0, h.y + h.height / 2.0);
    r.down_at(at);
    r.send(Event::PointerMove {
        x: at.0 - 2000.0,
        y: at.1 - 200.0,
    });
    r.up_at((at.0 - 2000.0, at.1 - 200.0));
    r.relayout();
    let moved = r.ui.rect(r.kb.sheet());
    assert!(moved.x.abs() < 0.5, "against the left edge: {moved:?}");
    assert!(
        (moved.y - (sheet.y - 200.0)).abs() < 0.5,
        "and up: {moved:?}"
    );
}

#[test]
fn a_split_keyboard_puts_its_halves_at_the_sides() {
    let r = Rig::sized(
        KeyboardConfig::ipad().placement(Placement::docked().split(200.0)),
        1024.0,
        768.0,
    );
    let (t, y) = (r.rect("t"), r.rect("y"));
    assert!(y.x - (t.x + t.width) > 190.0, "parted: {t:?} {y:?}");
    // The space bar is in both halves.
    let spaces = r.all_named("Space");
    assert_eq!(spaces.len(), 2);
}

#[test]
fn rows_can_be_added_over_and_under_every_page() {
    let mut r = Rig::new(
        KeyboardConfig::iphone()
            .with_number_row()
            .with_row_at_bottom(vec![Key::named("ArrowLeft", "←"), Key::text(".com")]),
    );
    let (one, q) = (r.rect("1"), r.rect("q"));
    assert!(one.y < q.y);
    r.tap("Shift");
    let one = r.center("1");
    r.down_at(one);
    r.up_at(one);
    r.tap(".com");
    assert_eq!(r.field.text, "!.com");
    r.tap("Numbers and symbols");
    r.named("ArrowLeft");
}

fn has(r: &Rig, name: &str) -> bool {
    !r.all_named(name).is_empty()
}

#[test]
fn each_press_reports_its_click_and_tap() {
    use std::cell::RefCell;
    use std::rc::Rc;
    use threers_ui::virtual_keyboard::{Haptic, KeyPress, KeySound};
    let mut r = Rig::new(KeyboardConfig::iphone());
    let heard: Rc<RefCell<Vec<KeyPress>>> = Rc::default();
    let h = heard.clone();
    r.kb.on_feedback(move |p| h.borrow_mut().push(p.clone()));
    r.tap("q");
    r.tap("Space");
    r.tap("Shift");
    r.tap("Shift");
    {
        let p = heard.borrow();
        assert_eq!(p[0].sound, Some(KeySound::Letter));
        assert_eq!(p[0].haptic, Some(Haptic::Light));
        assert_eq!(p[1].sound, Some(KeySound::Modifier));
        assert_eq!(p[3].haptic, Some(Haptic::Medium), "caps lock felt firmer");
    }
    // Held, Backspace clicks each time it deletes.
    heard.borrow_mut().clear();
    let back = r.center("Delete");
    r.down_at(back);
    for _ in 0..10 {
        r.tick(0.09);
    }
    r.up_at(back);
    let p = heard.borrow();
    assert!(p.len() > 3 && p.iter().all(|k| k.sound == Some(KeySound::Delete)));
    assert!(!p[0].repeat && p[p.len() - 1].repeat);
}

#[test]
fn feedback_can_be_turned_off_everywhere() {
    use std::cell::RefCell;
    use std::rc::Rc;
    let mut r = Rig::made(W, H, VirtualKeyboard::new);
    r.kb.customize(|c| {
        c.feedback.sound = false;
        c.feedback.haptics = false;
    });
    r.settle();
    let heard: Rc<RefCell<Vec<threers_ui::virtual_keyboard::KeyPress>>> = Rc::default();
    let h = heard.clone();
    r.kb.on_feedback(move |p| h.borrow_mut().push(p.clone()));
    r.tap("q");
    let p = heard.borrow();
    assert_eq!(p.len(), 1);
    assert_eq!((p[0].sound, p[0].haptic), (None, None));
}

#[test]
fn a_pressed_key_takes_the_other_keys_colour() {
    let mut r = Rig::sized(KeyboardConfig::ipad(), 820.0, 1180.0);
    let q = r.named("q");
    let before = r.ui.style(q).background;
    let back = r.ui.style(r.named("Delete")).background;
    let at = r.center("q");
    r.down_at(at);
    assert_eq!(r.ui.style(q).background, back, "grey while held");
    r.up_at(at);
    assert_eq!(r.ui.style(q).background, before);
}

#[test]
fn the_keyboard_follows_the_window() {
    let mut r = Rig::made(W, H, VirtualKeyboard::new);
    assert_eq!(
        r.kb.device(),
        Some(threers_ui::virtual_keyboard::Device::Phone)
    );
    assert!(!has(&r, "Hide keyboard"), "a phone's");
    r.settle_sized(820.0, 1180.0);
    r.settle();
    assert_eq!(
        r.kb.device(),
        Some(threers_ui::virtual_keyboard::Device::Tablet)
    );
    assert!(has(&r, "Hide keyboard"), "a tablet's");
    r.settle_sized(1024.0, 1366.0);
    r.settle();
    assert!(has(&r, "Caps Lock"), "a large tablet's");
}

#[test]
fn a_field_s_type_picks_its_keyboard_for_the_device() {
    use threers_ui::virtual_keyboard::KeyboardType;
    let mut r = Rig::made(W, H, VirtualKeyboard::new);
    let target = r.target;
    r.kb.set_type_for(target, KeyboardType::Email);
    r.settle();
    assert!(has(&r, "@") && has(&r, "."));
    r.kb.set_type_for(target, KeyboardType::NumberPad);
    r.settle();
    assert!(!has(&r, "q") && has(&r, "5"));
    r.kb.set_type_for(target, KeyboardType::Url);
    r.settle_sized(820.0, 1180.0);
    r.settle();
    assert!(
        has(&r, ".com") && has(&r, "Hide keyboard"),
        "a tablet's, with .com"
    );
}

/// A keyboard reading the field: sentences capitalised, two spaces a
/// period, words suggested.
fn reading_rig() -> Rig {
    Rig::made(W, H, |ui, input| {
        let kb = VirtualKeyboard::new(ui, input);
        kb.set_context(|f: &TextField, _| Some(f.before_caret()));
        kb.set_suggestions(|w| {
            ["hello", "help", "helmet"]
                .iter()
                .filter(|s| s.starts_with(&w.to_lowercase()))
                .map(|s| s.to_string())
                .collect()
        });
        kb
    })
}

#[test]
fn sentences_start_with_a_capital_and_two_spaces_end_one() {
    let mut r = reading_rig();
    r.settle();
    assert!(has(&r, "Q"), "an empty field starts a sentence");
    r.tap("H");
    assert!(has(&r, "q"), "then lower case");
    r.tap("i");
    r.tap("Space");
    r.tap("Space");
    assert_eq!(r.field.text, "Hi. ");
    r.settle();
    assert!(has(&r, "Q"), "a new sentence");
    // Shift tapped off stays off.
    r.tap("Shift");
    r.tap("o");
    assert_eq!(r.field.text, "Hi. o");
}

#[test]
fn words_are_suggested_and_a_tap_takes_one() {
    let mut r = reading_rig();
    r.settle();
    for k in ["H", "e", "l"] {
        r.tap(k);
        r.settle();
    }
    assert!(has(&r, "Suggestion: hello") && has(&r, "Suggestion: helmet"));
    r.tap("Suggestion: help");
    r.settle();
    assert_eq!(r.field.text, "help ");
    assert!(!has(&r, "Suggestion: hello"), "none for no word");
}

#[test]
fn the_next_word_is_predicted_after_a_space() {
    let mut r = Rig::made(W, H, |ui, input| {
        let kb = VirtualKeyboard::new(ui, input);
        kb.set_context(|f: &TextField, _| Some(f.before_caret()));
        kb.set_suggestions(|_| vec!["never".into()]);
        kb.set_predictions(|language, before| {
            assert_eq!(language, "en");
            match before.to_lowercase().as_str() {
                "go " => vec!["to".into(), "home".into()],
                "go h" => vec!["home".into()],
                _ => vec![],
            }
        });
        kb
    });
    r.settle();
    for k in ["G", "o", "Space"] {
        r.tap(k);
        r.settle();
    }
    assert!(has(&r, "Suggestion: to") && has(&r, "Suggestion: home"));
    assert!(!has(&r, "Suggestion: never"), "predictions come first");
    r.tap("h");
    r.settle();
    assert!(has(&r, "Suggestion: home") && !has(&r, "Suggestion: to"));
    r.tap("Suggestion: home");
    r.settle();
    assert_eq!(r.field.text, "Go home ", "the half-typed word replaced");
    r.kb.clear_predictions();
    r.tap("h");
    r.settle();
    assert!(has(&r, "Suggestion: never"), "the suggester again");
}

#[test]
fn a_space_on_the_symbols_page_goes_back_to_the_letters() {
    let mut r = Rig::new(KeyboardConfig::iphone());
    r.tap("Numbers and symbols");
    r.tap("1");
    r.tap("Space");
    r.settle();
    assert!(has(&r, "q"));
    assert_eq!(r.field.text, "1 ");
}

#[test]
fn action_keys_call_the_app_and_text_rows_build_keyboards() {
    use std::cell::RefCell;
    use std::rc::Rc;
    let config = KeyboardConfig::from_rows(&[
        "{row:0}",
        "{row:1}",
        "{shift@1.3} {row:2} {backspace@1.3}",
        "{123@1.25} {action:emoji:☺@1.25} {space} {return@2.5}",
    ]);
    assert!(config.validate().is_empty(), "{:?}", config.validate());
    let mut r = Rig::new(config);
    let called: Rc<RefCell<Vec<String>>> = Rc::default();
    let c = called.clone();
    r.kb.on_action(move |id, _| c.borrow_mut().push(id.to_owned()));
    r.tap("☺");
    assert_eq!(*called.borrow(), ["emoji"]);
    assert_eq!(r.field.text, "", "an action types nothing");
    let sketch = r.kb.sketch();
    assert!(sketch.contains("[ q ]") && sketch.contains("☺"), "{sketch}");
}

#[test]
fn backspace_held_long_deletes_whole_words() {
    let mut r = Rig::new(KeyboardConfig::iphone());
    r.field = TextField::new("one two three four five six seven eight nine ten");
    let back = r.center("Delete");
    r.down_at(back);
    for _ in 0..40 {
        r.tick(0.09);
    }
    r.up_at(back);
    // About thirteen characters one by one, then eleven words, two ticks
    // apart: everything (one at a time, it would leave half of it).
    assert_eq!(r.field.text, "", "words went");
}

#[test]
fn arrows_repeat_while_held() {
    let mut r =
        Rig::new(KeyboardConfig::iphone().with_row_at_bottom(vec![Key::named("ArrowLeft", "←")]));
    r.field = TextField::new("abcdefghij");
    let left = r.center("ArrowLeft");
    r.down_at(left);
    for _ in 0..10 {
        r.tick(0.09);
    }
    r.up_at(left);
    assert!(r.field.caret < 8, "moved more than once: {}", r.field.caret);
}

#[test]
fn holding_the_space_bar_makes_a_trackpad() {
    let mut r = Rig::new(KeyboardConfig::iphone());
    r.field = TextField::new("hello world");
    let space = r.center("Space");
    r.down_at(space);
    for _ in 0..6 {
        r.tick(0.1);
    }
    // Five steps to the left.
    let to = (space.0 - 5.0 * 12.0 - 1.0, space.1);
    r.send(Event::PointerMove { x: to.0, y: to.1 });
    r.up_at(to);
    assert_eq!(r.field.text, "hello world", "no space typed");
    assert_eq!(r.field.caret, 6);
    // A quick tap still types one.
    r.tap("Space");
    assert_eq!(r.field.text, "hello  world");
}

/// Every letter types itself, row by row: nothing laid out over a row (a
/// preview bubble of the row below, unseen) takes its presses.
#[test]
fn every_letter_types_itself() {
    for (config, size) in [
        (KeyboardConfig::iphone(), (W, H)),
        (KeyboardConfig::ipad(), (820.0, 1180.0)),
    ] {
        let mut r = Rig::sized(config, size.0, size.1);
        // The key pressed, as the finger lands (its click, preview and
        // accents are that key's), not only the one typed as it lifts.
        let pressed: std::rc::Rc<std::cell::RefCell<String>> = Default::default();
        let p = pressed.clone();
        r.kb.on_feedback(move |k| p.borrow_mut().push_str(&k.name));
        let all = "qwertyuiopasdfghjklzxcvbnm";
        for c in all.chars() {
            let at = r.center(&c.to_string());
            r.down_at(at);
            r.up_at(at);
        }
        assert_eq!(*pressed.borrow(), all, "pressed");
        assert_eq!(r.field.text, all, "typed");
    }
}

/// A number row over every page takes the place of a page's own row of
/// digits, as on a large iPad, rather than showing the digits twice.
#[test]
fn a_number_row_is_not_repeated_on_the_symbols_page() {
    use threers_ui::virtual_keyboard::Page;
    let config = KeyboardConfig::iphone().with_number_row();
    let digits = |page| {
        config
            .sketch(page, &KeyboardLayout::english(), 1)
            .lines()
            .filter(|l| l.starts_with("[ 1 ]"))
            .count()
    };
    assert_eq!(digits(Page::Letters), 1);
    assert_eq!(digits(Page::Symbols), 1);
    // And on the keyboard: one "1" on the page.
    let mut r = Rig::new(config);
    r.tap("Numbers and symbols");
    let ones =
        r.ui.accessibility_tree()
            .into_iter()
            .filter(|a| a.name == "1" && a.role == Role::Button)
            .count();
    assert_eq!(ones, 1);
    r.named("@");
}

/// Fast typing with two thumbs: the next finger lands before the last
/// lifts. Each key still types, in order, the earlier one as the next
/// lands (as on a phone), not lost to a pinch.
#[test]
fn overlapping_fingers_type_every_key_in_order() {
    use threers_ui::input::TouchPhase;
    let mut r = Rig::new(KeyboardConfig::iphone());
    let mut t = 0.0;
    let mut touch = |r: &mut Rig, id: u64, phase: TouchPhase, at: (f32, f32)| {
        t += 0.03;
        r.send(Event::Touch {
            id,
            phase,
            x: at.0,
            y: at.1,
            time: t,
        });
    };
    let word = "typing";
    let keys: Vec<(f32, f32)> = word.chars().map(|c| r.center(&c.to_string())).collect();
    // Each finger lands, the one before it lifts after, the last lifts.
    for (i, &at) in keys.iter().enumerate() {
        touch(&mut r, i as u64, TouchPhase::Start, at);
        if i > 0 {
            touch(&mut r, i as u64 - 1, TouchPhase::End, keys[i - 1]);
        }
    }
    touch(
        &mut r,
        keys.len() as u64 - 1,
        TouchPhase::End,
        keys[keys.len() - 1],
    );
    r.settle();
    assert_eq!(r.field.text, word);
    // One finger at a time types as before.
    let at = r.center("Space");
    touch(&mut r, 20, TouchPhase::Start, at);
    touch(&mut r, 20, TouchPhase::End, at);
    let at = r.center("a");
    touch(&mut r, 21, TouchPhase::Start, at);
    touch(&mut r, 21, TouchPhase::End, at);
    r.settle();
    assert_eq!(r.field.text, "typing a");
}

/// Gesture typing (feature `swipe`).
#[cfg(feature = "swipe")]
mod swiping {
    use super::*;
    /// A keyboard whose swipe decoder keeps what it was given and answers
    /// `words`.
    fn swiping_rig(
        words: &'static [&'static str],
    ) -> (
        Rig,
        std::rc::Rc<std::cell::RefCell<Vec<threers_ui::virtual_keyboard::Swipe>>>,
    ) {
        let seen: std::rc::Rc<std::cell::RefCell<Vec<threers_ui::virtual_keyboard::Swipe>>> =
            Default::default();
        let r = Rig::new(KeyboardConfig::iphone());
        let log = seen.clone();
        r.kb.set_swipe_decoder(move |s| {
            log.borrow_mut().push(s.clone());
            words.iter().map(|w| w.to_string()).collect()
        });
        (r, seen)
    }

    /// Down on the first letter's key, through the others' centres in small
    /// steps (the clock running), up on the last.
    fn swipe(r: &mut Rig, word: &str) {
        // Keys are named as labelled: upper case under Shift.
        let at: Vec<(f32, f32)> = word
            .chars()
            .map(|c| {
                let lower = c.to_string();
                let name = if has(r, &lower) {
                    lower
                } else {
                    c.to_uppercase().to_string()
                };
                r.center(&name)
            })
            .collect();
        r.down_at(at[0]);
        for w in at.windows(2) {
            for i in 1..=6 {
                let f = i as f32 / 6.0;
                let p = (
                    w[0].0 + (w[1].0 - w[0].0) * f,
                    w[0].1 + (w[1].1 - w[0].1) * f,
                );
                r.tick(0.016);
                r.send(Event::PointerMove { x: p.0, y: p.1 });
            }
        }
        r.up_at(*at.last().unwrap());
    }

    /// Counts what the trail paints: segments, their widths and alphas.
    #[derive(Default)]
    struct Strokes(Vec<(f32, f32)>);

    impl threers_ui::TextMeasure for Strokes {
        fn measure(&mut self, _: &str, _: &threers_ui::Style) -> (f32, f32) {
            (0.0, 0.0)
        }
    }

    impl threers_ui::Painter for Strokes {
        fn fill_rect(&mut self, _: Rect, _: threers_ui::Rgba) {}
        fn text(&mut self, _: Rect, _: &str, _: &threers_ui::TextStyle) {}
        fn path(
            &mut self,
            _: Rect,
            _: &[String],
            _: [f32; 4],
            stroke: Option<f32>,
            _: bool,
            c: threers_ui::Rgba,
        ) {
            self.0.push((stroke.unwrap_or(0.0), c[3]));
        }
        fn fill_rounded_rect(&mut self, r: threers_ui::Rect, _: f32, _: f32, c: threers_ui::Rgba) {
            self.0.push((r.width, c[3]));
        }
    }

    #[test]
    fn the_finger_leaves_a_trail_that_fades_once_it_lifts() {
        let (mut r, _) = swiping_rig(&["hello"]);
        let mut none = Strokes::default();
        assert!(!r.kb.paint_swipe_trail(&mut none, r.t), "no swipe yet");
        assert!(none.0.is_empty());
        let (h, e, l) = (r.center("h"), r.center("e"), r.center("l"));
        r.down_at(h);
        for (from, to) in [(h, e), (e, l)] {
            for i in 1..=6 {
                let f = i as f32 / 6.0;
                r.tick(0.016);
                r.send(Event::PointerMove {
                    x: from.0 + (to.0 - from.0) * f,
                    y: from.1 + (to.1 - from.1) * f,
                });
            }
        }
        let mut mid = Strokes::default();
        assert!(r.kb.paint_swipe_trail(&mut mid, r.t), "drawn while swiping");
        assert!(mid.0.len() > 8, "dots along it");
        let (first, last) = (mid.0[0], *mid.0.last().unwrap());
        assert!(last.0 > first.0, "wider toward the finger");
        // About four over each place: 0.8 opaque, so.
        let stacked = 1.0 - (1.0 - last.1).powi(4);
        assert!((stacked - 0.8).abs() < 1e-3, "{stacked}");
        r.send(Event::PointerUp {
            x: l.0,
            y: l.1,
            button: PointerButton::Primary,
        });
        let mut lifted = Strokes::default();
        assert!(r.kb.paint_swipe_trail(&mut lifted, r.t + 0.1), "fading");
        assert!(lifted.0.last().unwrap().1 < last.1, "dimmer as it fades");
        let mut gone = Strokes::default();
        assert!(!r.kb.paint_swipe_trail(&mut gone, r.t + 1.0));
        assert!(gone.0.is_empty());
        assert_eq!(r.field.text, "hello ");
    }

    #[test]
    fn a_swipe_types_the_decoder_s_word_and_offers_the_others() {
        let (mut r, seen) = swiping_rig(&["hello", "jello", "hells"]);
        swipe(&mut r, "helo");
        r.settle();
        assert_eq!(
            r.field.text, "hello ",
            "no letter typed: the word, and a space"
        );
        {
            let seen = seen.borrow();
            assert_eq!(seen.len(), 1);
            let s = &seen[0];
            assert_eq!(s.language, "en");
            assert_eq!(s.keys.len(), 26, "every letter, as drawn");
            let h = s.keys.iter().find(|k| k.text == "h").unwrap();
            let p0 = s.points[0];
            assert!(p0.x >= h.x && p0.x <= h.x + h.width && p0.y >= h.y && p0.y <= h.y + h.height);
            assert!(s.points.len() > 15);
            assert!(
                s.points.last().unwrap().t > p0.t,
                "the clock runs through it"
            );
        }
        assert!(has(&r, "Suggestion: jello") && has(&r, "Suggestion: hells"));
        // A second word gets its space before it.
        swipe(&mut r, "helo");
        r.settle();
        assert_eq!(r.field.text, "hello hello ", "one space between");
        // The bar's other word replaces the one swiped.
        r.tap("Suggestion: jello");
        r.settle();
        assert_eq!(
            r.field.text, "hello jello ",
            "the word and its space replaced"
        );
    }

    #[test]
    fn punctuation_after_a_swiped_word_takes_its_space() {
        let (mut r, _) = swiping_rig(&["hello"]);
        swipe(&mut r, "helo");
        r.settle();
        assert_eq!(r.field.text, "hello ");
        r.tap("Numbers and symbols");
        r.settle();
        r.tap(".");
        r.settle();
        assert_eq!(r.field.text, "hello.", "the period in the space's place");
        r.tap("Letters");
        r.settle();
        // The next swipe puts its space back.
        swipe(&mut r, "helo");
        r.settle();
        assert_eq!(r.field.text, "hello. hello ");
    }

    #[test]
    fn with_swiping_on_taps_and_short_slides_type_as_before() {
        let (mut r, seen) = swiping_rig(&["never"]);
        r.tap("a");
        // A slide to the neighbouring key, under a key's width: that key.
        let (from, to) = (r.center("s"), r.center("d"));
        r.down_at(from);
        r.send(Event::PointerMove {
            x: (from.0 + to.0) / 2.0 + 4.0,
            y: from.1,
        });
        r.up_at(((from.0 + to.0) / 2.0 + 4.0, from.1));
        r.settle();
        assert_eq!(r.field.text, "ad");
        assert!(seen.borrow().is_empty(), "no swipe");
        // Turned off, a long slide types where it ends.
        r.kb.clear_swipe_decoder();
        let (from, to) = (r.center("q"), r.center("p"));
        r.down_at(from);
        r.send(Event::PointerMove { x: to.0, y: to.1 });
        r.up_at(to);
        r.settle();
        assert_eq!(r.field.text, "adp");
    }

    #[test]
    fn a_swipe_starting_a_sentence_takes_a_capital() {
        let mut r = reading_rig();
        r.kb.set_swipe_decoder(|_| vec!["hello".into()]);
        r.settle();
        swipe(&mut r, "helo");
        r.settle();
        assert_eq!(r.field.text, "Hello ");
        assert!(has(&r, "q"), "Shift was for one word");
    }
}

/// A password's keyboard asks nobody about it, says nothing of it and keeps
/// nothing of it — nor for the next field.
#[test]
fn a_secret_is_not_read_predicted_nor_kept() {
    use std::cell::RefCell;
    use std::rc::Rc;
    let told: Rc<RefCell<Vec<String>>> = Default::default();
    // With the app telling the text and without (the keyboard's own record).
    for context in [false, true] {
        let mut r = Rig::new(KeyboardConfig::iphone());
        if context {
            let t = told.clone();
            r.kb.set_context(move |f: &TextField, _| {
                t.borrow_mut().push(format!("read {}", f.text));
                Some(f.before_caret())
            });
        }
        let t = told.clone();
        r.kb.set_predictions(move |_, before| {
            t.borrow_mut().push(format!("predict {before}"));
            vec!["offered".into()]
        });
        let t = told.clone();
        r.kb.set_suggestions(move |word| {
            t.borrow_mut().push(format!("suggest {word}"));
            vec!["offered".into()]
        });
        let offered = |r: &Rig| r.names().iter().any(|n| n.starts_with("Suggestion"));
        // An ordinary field is offered words: the hooks work.
        let a = if r.names().iter().any(|n| n == "a") { "a" } else { "A" };
        r.tap(a);
        assert_eq!(!told.borrow().is_empty(), context, "context {context}");
        assert_eq!(offered(&r), context);
        // The same field, a password now.
        r.ui.semantics_mut(r.target).secret = true;
        r.settle();
        assert!(!offered(&r), "what was offered goes");
        told.borrow_mut().clear();
        r.field = TextField::new("");
        for k in ["s", "e", "c"] {
            r.tap(k);
        }
        r.tap("Space");
        r.tap("Space");
        assert_eq!(r.field.text, "sec  ", "no capital, no full stop");
        assert!(told.borrow().is_empty(), "{:?}", told.borrow());
        assert!(!offered(&r));
        // Back to an ordinary one: nothing of it is left.
        r.ui.semantics_mut(r.target).secret = false;
        r.field = TextField::new("");
        r.settle();
        let x = if r.names().iter().any(|n| n == "x") { "x" } else { "X" };
        r.tap(x);
        assert!(told.borrow().iter().all(|t| !t.contains("sec")), "{:?}", told.borrow());
        told.borrow_mut().clear();
    }
}

#[cfg(feature = "swipe")]
#[test]
fn a_secret_is_not_swiped() {
    let seen: std::rc::Rc<std::cell::Cell<u32>> = Default::default();
    let mut r = Rig::new(KeyboardConfig::iphone());
    r.ui.semantics_mut(r.target).secret = true;
    r.settle();
    let n = seen.clone();
    r.kb.set_swipe_decoder(move |_| {
        n.set(n.get() + 1);
        vec!["hello".into()]
    });
    let (from, to) = (r.center("q"), r.center("p"));
    r.down_at(from);
    for i in 1..=12 {
        r.tick(0.016);
        let x = from.0 + (to.0 - from.0) * i as f32 / 12.0;
        r.send(Event::PointerMove { x, y: from.1 });
    }
    r.up_at(to);
    assert_eq!(seen.get(), 0);
    assert_eq!(r.field.text, "p", "the key under the finger, as without swiping");
}
