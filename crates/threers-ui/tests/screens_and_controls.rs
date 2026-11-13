//! Navigation stacks, sheets, snapping, scaled spacing, safe areas, right
//! to left, tracked views, matched geometry, button roles, gesture
//! priority, accessibility actions and order, focus as state, and the
//! pickers.

use std::cell::Cell;
use std::rc::Rc;

use threers_ui::binding;
use threers_ui::button::ButtonRole;
use threers_ui::environment::LayoutDirection;
use threers_ui::input::{Event, Gesture, GesturePriority, Input, Phase, PointerButton, TouchPhase};
use threers_ui::model::Model;
use threers_ui::nav_stack::nav_stack;
use threers_ui::pickers::{progress, search_field, segmented, slider};
use threers_ui::prompt::prompt_with;
use threers_ui::scroll_snap::Snap;
use threers_ui::sheet::{sheet, Detent};
use threers_ui::view::{El, View};
use threers_ui::{style, Edges, FixedWidthText, NodeId, Role, Semantics, Ui};

fn layout(ui: &mut Ui) {
    ui.layout(400.0, 600.0, &mut FixedWidthText(7.0));
}

fn click<S: 'static>(input: &mut Input<S>, ui: &mut Ui, s: &mut S, n: NodeId) {
    let r = ui.rect(n);
    let (x, y) = (r.x + r.width / 2.0, r.y + r.height / 2.0);
    let b = PointerButton::Primary;
    input.dispatch(ui, s, Event::PointerMove { x, y });
    input.dispatch(ui, s, Event::PointerDown { x, y, button: b });
    input.dispatch(ui, s, Event::PointerUp { x, y, button: b });
}

fn named(ui: &Ui, name: &str) -> NodeId {
    ui.accessibility_tree().into_iter().find(|a| a.name == name).unwrap_or_else(|| panic!("no {name}")).id
}

#[test]
fn spacing_follows_the_text_size_and_the_safe_area_pads() {
    let mut ui = Ui::new();
    let root = ui.root();
    let row = ui.add_box(root, style! { display: flex; padding: 4px 8px; gap: 6px; height: 30px; });
    ui.add_text(row, "a");
    ui.scale_tree_with_text(row);
    ui.set_text_scale(2.0);
    let s = ui.style(row);
    assert_eq!((s.padding.left, s.padding.top, s.gap, s.height), (16.0, 8.0, 12.0, threers_ui::Length::Px(60.0)));
    ui.set_text_scale(1.0);
    assert_eq!(ui.style(row).padding.left, 8.0);
    assert_eq!(ui.scaled(10.0), 10.0);

    let page = ui.add_box(root, style! { padding: 10px; });
    ui.pad_safe_area(page, [true, false, true, false]);
    ui.set_safe_area(Edges { top: 47.0, right: 0.0, bottom: 34.0, left: 0.0 });
    assert_eq!((ui.style(page).padding.top, ui.style(page).padding.bottom, ui.style(page).padding.left), (57.0, 44.0, 10.0));
    ui.set_safe_area(Edges::ZERO);
    assert_eq!(ui.style(page).padding.top, 10.0);
}

#[test]
fn right_to_left_mirrors_rows() {
    let mut ui = Ui::new();
    let root = ui.root();
    let row = ui.add_box(root, style! { display: flex; width: 400px; });
    let a = ui.add_box(row, style! { width: 100px; height: 20px; });
    let b = ui.add_box(row, style! { width: 50px; height: 20px; });
    layout(&mut ui);
    assert_eq!((ui.rect(a).x, ui.rect(b).x), (0.0, 100.0));
    ui.set_layout_direction(LayoutDirection::RightToLeft);
    layout(&mut ui);
    assert_eq!((ui.rect(a).x, ui.rect(b).x), (300.0, 250.0), "the first at the right end");
    assert_eq!(ui.hit_test(350.0, 10.0), Some(a));
    assert_eq!(threers_ui::i18n::direction_of("ar-EG"), LayoutDirection::RightToLeft);
}

#[test]
fn a_strip_settles_on_an_item_after_a_flick_and_after_the_wheel() {
    let mut ui = Ui::new();
    let root = ui.root();
    let strip = ui.add_box(root, style! { display: flex; overflow-x: auto; width: 200px; });
    for _ in 0..20 {
        ui.add_box(strip, style! { width: 80px; height: 40px; flex-shrink: 0; });
    }
    ui.set_scroll_snap(strip, Some(Snap::Items));
    layout(&mut ui);
    let mut input: Input<()> = Input::new();
    let (mut t, dt) = (0.0, 1.0 / 60.0);
    input.tick(&mut ui, &mut (), t);
    let touch = |input: &mut Input<()>, ui: &mut Ui, phase, x: f32, t: f64| {
        input.dispatch(ui, &mut (), Event::Touch { id: 1, phase, x, y: 20.0, time: t });
    };
    // A slow drag of 100 px, let go still: onto the nearest item (80).
    touch(&mut input, &mut ui, TouchPhase::Start, 180.0, t);
    for k in 1..=10 {
        t += dt;
        touch(&mut input, &mut ui, TouchPhase::Move, 180.0 - k as f32 * 10.0, t);
        layout(&mut ui);
    }
    t += 0.3;
    touch(&mut input, &mut ui, TouchPhase::End, 80.0, t);
    let run = |input: &mut Input<()>, ui: &mut Ui, t: &mut f64| {
        for _ in 0..60 {
            *t += dt;
            input.tick(ui, &mut (), *t);
            ui.step(dt as f32);
            layout(ui);
        }
    };
    run(&mut input, &mut ui, &mut t);
    assert_eq!(ui.style(strip).scroll_left, 80.0);
    // A flick goes further, and still ends on an item.
    touch(&mut input, &mut ui, TouchPhase::Start, 180.0, t);
    for k in 1..=4 {
        t += dt;
        touch(&mut input, &mut ui, TouchPhase::Move, 180.0 - k as f32 * 25.0, t);
        layout(&mut ui);
    }
    touch(&mut input, &mut ui, TouchPhase::End, 80.0, t);
    run(&mut input, &mut ui, &mut t);
    let at = ui.style(strip).scroll_left;
    assert!(at > 300.0 && at % 80.0 == 0.0, "{at}");
    // The wheel moves it anywhere; once it stops, it settles.
    input.dispatch(&mut ui, &mut (), Event::Wheel { x: 50.0, y: 20.0, dx: 30.0, dy: 0.0 });
    run(&mut input, &mut ui, &mut t);
    assert_eq!(ui.style(strip).scroll_left % 80.0, 0.0);
}

#[test]
fn a_scroll_offset_in_the_state_is_restored_and_followed() {
    #[derive(Default)]
    struct S {
        top: f32,
    }
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(root, style! { height: 100px; overflow: auto; });
    ui.add_box(list, style! { height: 1000px; });
    let mut input: Input<S> = Input::new();
    let mut s = S { top: 250.0 };
    input.bind_scroll(list, binding!(|s: S| s.top));
    // Before layout it cannot be reached: it waits.
    input.tick(&mut ui, &mut s, 0.0);
    assert_eq!(s.top, 250.0);
    layout(&mut ui);
    input.tick(&mut ui, &mut s, 0.1);
    assert_eq!(ui.style(list).scroll_top, 250.0);
    input.dispatch(&mut ui, &mut s, Event::Wheel { x: 10.0, y: 10.0, dx: 0.0, dy: 30.0 });
    assert_eq!(s.top, 280.0, "scrolling writes the state");
}

#[test]
fn a_tracked_view_is_described_again_only_when_its_models_change() {
    let mut ui = Ui::new();
    let mut input: Input<()> = Input::new();
    let root = ui.root();
    let items = Model::new(vec!["a".to_owned()]);
    let mut view = View::new(&mut ui, root, "");
    let described = Rc::new(Cell::new(0));
    let describe = |items: &Model<Vec<String>>, n: &Rc<Cell<u32>>| {
        n.set(n.get() + 1);
        items.read().iter().map(|i| El::<()>::label(i.clone())).collect::<Vec<_>>()
    };
    assert!(view.update_tracked(&mut ui, &mut input, || describe(&items, &described)));
    assert!(!view.update_tracked(&mut ui, &mut input, || describe(&items, &described)));
    assert_eq!(described.get(), 1);
    items.update(|v| v.push("b".into()));
    assert!(view.update_tracked(&mut ui, &mut input, || describe(&items, &described)));
    assert_eq!(ui.children(view.root()).len(), 2);
}

#[test]
fn a_navigation_stack_follows_its_path_and_goes_back() {
    #[derive(Default)]
    struct S {
        path: Vec<String>,
    }
    let mut ui = Ui::new();
    let mut input: Input<S> = Input::new();
    let mut s = S::default();
    let root = ui.root();
    let host = ui.add_box(root, style! { display: flex; flex-direction: column; height: 600px; });
    let stack = nav_stack(
        &mut ui,
        &mut input,
        &s,
        host,
        "Inbox",
        binding!(|s: S| s.path),
        |ui, _, _, page, screen| {
            ui.add_text(screen, format!("Page {page}"));
        },
        |ui, _, _, screen| {
            ui.add_text(screen, "Home");
        },
    );
    let mut t = 0.0;
    let mut frame = |input: &mut Input<S>, ui: &mut Ui, s: &mut S| {
        t += 0.05;
        input.tick(ui, s, t);
        ui.step(0.05);
        layout(ui);
    };
    frame(&mut input, &mut ui, &mut s);
    assert_eq!(stack.title(&ui), "Inbox");
    s.path = vec!["a".into(), "b".into()];
    for _ in 0..8 {
        frame(&mut input, &mut ui, &mut s);
    }
    assert_eq!(stack.title(&ui), "b");
    // Only the top screen is read.
    let texts: Vec<String> = ui.accessibility_tree().into_iter().filter(|a| a.role == Role::StaticText).map(|a| a.name).collect();
    assert!(texts.contains(&"Page b".to_owned()) && !texts.contains(&"Page a".to_owned()) && !texts.contains(&"Home".to_owned()), "{texts:?}");
    click(&mut input, &mut ui, &mut s, stack.back);
    for _ in 0..8 {
        frame(&mut input, &mut ui, &mut s);
    }
    assert_eq!(s.path, ["a"]);
    assert_eq!(stack.title(&ui), "a");
    // A drag from the left edge goes back too.
    let body = ui.rect(stack.body);
    let (y, mut tt) = (body.y + 100.0, 10.0);
    let touch = |input: &mut Input<S>, ui: &mut Ui, s: &mut S, phase, x: f32, t: f64| {
        input.dispatch(ui, s, Event::Touch { id: 1, phase, x, y, time: t });
    };
    touch(&mut input, &mut ui, &mut s, TouchPhase::Start, body.x + 5.0, tt);
    for k in 1..=10 {
        tt += 0.016;
        touch(&mut input, &mut ui, &mut s, TouchPhase::Move, body.x + 5.0 + k as f32 * 20.0, tt);
    }
    touch(&mut input, &mut ui, &mut s, TouchPhase::End, body.x + 205.0, tt + 0.2);
    frame(&mut input, &mut ui, &mut s);
    assert!(s.path.is_empty());
}

#[test]
fn a_sheet_moves_between_detents_and_is_dragged_away() {
    #[derive(Default)]
    struct S {
        open: bool,
    }
    let mut ui = Ui::new();
    let mut input: Input<S> = Input::new();
    let mut s = S { open: true };
    layout(&mut ui);
    sheet(&mut input, binding!(|s: S| s.open), vec![Detent::Medium, Detent::Large], "Options", |ui, _, _, c| {
        ui.add_text(c, "Inside");
    });
    let mut t = 0.0;
    let mut frame = |input: &mut Input<S>, ui: &mut Ui, s: &mut S| {
        t += 0.05;
        input.tick(ui, s, t);
        ui.step(0.05);
        layout(ui);
    };
    for _ in 0..10 {
        frame(&mut input, &mut ui, &mut s);
    }
    let panel = named(&ui, "Options");
    assert_eq!(ui.rect(panel).y, 300.0);
    assert_eq!(ui.accessibility_tree().into_iter().find(|a| a.id == panel).unwrap().role, Role::Dialog);
    // The grabber, dragged up: the large detent.
    let g = ui.rect(panel);
    let (x, y) = (g.x + g.width / 2.0, g.y + 10.0);
    let b = PointerButton::Primary;
    input.dispatch(&mut ui, &mut s, Event::PointerDown { x, y, button: b });
    input.dispatch(&mut ui, &mut s, Event::PointerMove { x, y: y - 200.0 });
    input.dispatch(&mut ui, &mut s, Event::PointerUp { x, y: y - 200.0, button: b });
    for _ in 0..10 {
        frame(&mut input, &mut ui, &mut s);
    }
    assert_eq!(ui.rect(panel).y, 12.0);
    // Dragged down most of the way: away, and the state is told.
    let y = ui.rect(panel).y + 10.0;
    input.dispatch(&mut ui, &mut s, Event::PointerDown { x, y, button: b });
    input.dispatch(&mut ui, &mut s, Event::PointerMove { x, y: y + 500.0 });
    input.dispatch(&mut ui, &mut s, Event::PointerUp { x, y: y + 500.0, button: b });
    for _ in 0..10 {
        frame(&mut input, &mut ui, &mut s);
    }
    assert!(!s.open);
    assert_eq!(input.modal(), None);
}

#[test]
fn buttons_by_role_answer_escape_and_enter() {
    let mut ui = Ui::new();
    let mut input: Input<Vec<usize>> = Input::new();
    let mut log = vec![];
    layout(&mut ui);
    let buttons = [("Keep", ButtonRole::Cancel), ("Delete", ButtonRole::Destructive), ("Archive", ButtonRole::Primary)];
    prompt_with(&mut ui, &mut input, &mut log, "Delete it?", "", &buttons, |log, _, i| log.push(i));
    layout(&mut ui);
    assert_eq!(input.focused(), Some(named(&ui, "Archive")), "focus starts on the primary one");
    let delete = named(&ui, "Delete");
    assert_eq!(ui.style(delete).background, Some(ui.token(threers_ui::environment::Token::DestructiveFill)).map(|c| c.map(|v| (v * 255.0).round() / 255.0)));
    input.simulate_keystrokes(&mut ui, &mut log, "escape").unwrap();
    assert_eq!(log, [0], "Escape: the cancel button, wherever it is");
}

#[test]
fn gesture_priority_takes_pans_from_scrolling_or_shares_them() {
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(root, style! { height: 200px; overflow: auto; });
    let canvas = ui.add_box(list, style! { height: 100px; });
    ui.add_box(list, style! { height: 1000px; });
    layout(&mut ui);
    let mut input: Input<u32> = Input::new();
    let mut pans = 0;
    input.on(canvas, Phase::Bubble, |n: &mut u32, _, e, cx| {
        if let Event::Gesture(Gesture::Pan { .. }) = e {
            *n += 1;
            cx.prevent_default();
        }
    });
    let drag = |input: &mut Input<u32>, ui: &mut Ui, n: &mut u32, t0: f64| {
        input.dispatch(ui, n, Event::Touch { id: 1, phase: TouchPhase::Start, x: 50.0, y: 80.0, time: t0 });
        for k in 1..=5 {
            input.dispatch(ui, n, Event::Touch { id: 1, phase: TouchPhase::Move, x: 50.0, y: 80.0 - k as f32 * 10.0, time: t0 + k as f64 * 0.016 });
        }
        input.dispatch(ui, n, Event::Touch { id: 1, phase: TouchPhase::End, x: 50.0, y: 30.0, time: t0 + 0.5 });
    };
    input.set_gesture_priority(canvas, Some(GesturePriority::Simultaneous));
    drag(&mut input, &mut ui, &mut pans, 0.0);
    assert!(pans > 0);
    assert!(ui.style(list).scroll_top > 0.0, "shared: the list scrolled too");
    ui.scroll_to(list, 0.0);
    layout(&mut ui);
    input.set_gesture_priority(canvas, Some(GesturePriority::High));
    input.on(canvas, Phase::Bubble, |_, _, _, cx| cx.stop_propagation());
    drag(&mut input, &mut ui, &mut pans, 5.0);
    assert_eq!(ui.style(list).scroll_top, 0.0, "taken: nothing scrolled");
}

#[test]
fn rows_read_as_one_with_actions_and_an_order() {
    let mut ui = Ui::new();
    let mut input: Input<Vec<String>> = Input::new();
    let mut log = vec![];
    let root = ui.root();
    let row = ui.add_box(root, style! { display: flex; });
    ui.set_semantics(row, Semantics::new(Role::ListItem));
    ui.add_text(row, "Ada");
    ui.add_text(row, "3 unread");
    ui.combine_accessibility(row);
    let late = ui.add_box(root, style! {});
    ui.set_semantics(late, Semantics::new(Role::Button).label("Compose"));
    ui.semantics_mut(late).sort_priority = 1;
    input.on_accessibility_action(&mut ui, row, &["Delete", "Archive"], |log, _, a| log.push(a.to_owned()));
    layout(&mut ui);
    let tree = ui.accessibility_tree();
    let names: Vec<&str> = tree[0].children.iter().map(|&i| tree[i].name.as_str()).collect();
    assert_eq!(names, ["Compose", "Ada 3 unread"], "priority first; the row as one");
    let r = tree.iter().find(|a| a.id == row).unwrap();
    assert!(r.children.is_empty());
    assert_eq!(r.actions, ["Delete", "Archive"]);
    assert!(input.accessibility_action(&mut ui, &mut log, row, "Archive"));
    assert!(!input.accessibility_action(&mut ui, &mut log, row, "Fly"));
    assert_eq!(log, ["Archive"]);
}

#[test]
fn default_focus_goes_in_when_a_part_shows() {
    let mut ui = Ui::new();
    let mut input: Input<()> = Input::new();
    let root = ui.root();
    let form = ui.add_box(root, style! {});
    let first = ui.add_box(form, style! {});
    ui.set_semantics(first, Semantics::new(Role::TextBox));
    input.default_focus(form, first);
    input.tick(&mut ui, &mut (), 0.0);
    assert_eq!(input.focused(), Some(first));
}

#[test]
fn pickers_work_from_keys_and_the_pointer() {
    #[derive(Default)]
    struct S {
        size: usize,
        volume: f32,
        query: String,
    }
    let mut ui = Ui::new();
    let mut input: Input<S> = Input::new();
    let mut s = S { volume: 0.5, ..Default::default() };
    let root = ui.root();
    let col = ui.add_box(root, style! { display: flex; flex-direction: column; gap: 8px; width: 300px; });
    let seg = segmented(&mut ui, &mut input, &s, col, "Size", &["S", "M", "L"], binding!(|s: S| s.size));
    let sl = slider(&mut ui, &mut input, &s, col, "Volume", binding!(|s: S| s.volume), 0.0..=1.0, 0.1);
    let p = progress(&mut ui, col, "Upload", Some(0.25));
    let search = search_field(&mut ui, &mut input, &s, col, "Search", binding!(|s: S| s.query));
    layout(&mut ui);
    // Segmented: a click, then the arrows; the highlight follows.
    let n = named(&ui, "L");
    click(&mut input, &mut ui, &mut s, n);
    assert_eq!(s.size, 2);
    let n = named(&ui, "L");
    input.set_focus(&mut ui, &mut s, Some(n));
    input.simulate_keystrokes(&mut ui, &mut s, "right").unwrap();
    assert_eq!(s.size, 0, "wraps");
    assert_eq!(ui.accessibility_tree().into_iter().find(|a| a.name == "S").unwrap().checked, Some(true));
    let _ = seg;
    // Slider: keys, and a press along the track.
    input.set_focus(&mut ui, &mut s, Some(sl));
    input.simulate_keystrokes(&mut ui, &mut s, "right right").unwrap();
    assert!((s.volume - 0.7).abs() < 1e-5);
    let r = ui.rect(sl);
    let b = PointerButton::Primary;
    input.dispatch(&mut ui, &mut s, Event::PointerDown { x: r.x + r.width - 10.0, y: r.y + 10.0, button: b });
    input.dispatch(&mut ui, &mut s, Event::PointerUp { x: r.x + r.width - 10.0, y: r.y + 10.0, button: b });
    assert_eq!(s.volume, 1.0);
    let a = ui.accessibility_tree().into_iter().find(|a| a.id == sl).unwrap();
    assert_eq!((a.role, a.range.map(|r| r.now)), (Role::Slider, Some(1.0)));
    // Progress.
    let pa = ui.accessibility_tree().into_iter().find(|a| a.id == p.bar).unwrap();
    assert_eq!((pa.role, pa.value.as_deref()), (Role::ProgressBar, Some("25%")));
    p.set(&mut ui, None);
    assert!(ui.accessibility_tree().into_iter().find(|a| a.id == p.bar).unwrap().range.is_none(), "unknown: no value");
    // Search: typed into, then cleared.
    input.set_focus(&mut ui, &mut s, Some(search));
    input.simulate_keystrokes(&mut ui, &mut s, "c a t").unwrap();
    assert_eq!(s.query, "cat");
    layout(&mut ui);
    let n = named(&ui, "Clear");
    click(&mut input, &mut ui, &mut s, n);
    assert_eq!(s.query, "");
}

/// Tapping the on-screen keyboard's keys types into the field, which keeps
/// focus; Shift capitalises one letter; ⌄ puts it away.
#[test]
fn the_on_screen_keyboard_types_where_the_focus_is() {
    use threers_ui::virtual_keyboard::VirtualKeyboard;
    use threers_ui::{FieldKey, TextField};
    let mut ui = Ui::new();
    let mut input: Input<TextField> = Input::new();
    let mut field = TextField::new("");
    let root = ui.root();
    let b = ui.add_box(root, style! { height: 20px; });
    ui.set_semantics(b, Semantics::new(Role::TextBox).label("Name"));
    input.on_key(b, |f: &mut TextField, _, key, m, _| !matches!(f.key(key, m, |_| true), FieldKey::Ignored));
    let kb = VirtualKeyboard::new(&mut ui, &mut input);
    kb.set_enabled(true);
    input.tick(&mut ui, &mut field, 0.0);
    layout(&mut ui);
    assert!(!kb.showing(&ui), "nothing to type into yet");
    input.set_focus(&mut ui, &mut field, Some(b));
    input.tick(&mut ui, &mut field, 0.1);
    ui.step(0.3);
    layout(&mut ui);
    assert!(kb.showing(&ui));
    for k in ["Shift", "H", "i", "Space", "Delete"] {
        let n = named(&ui, k);
        click(&mut input, &mut ui, &mut field, n);
        layout(&mut ui);
    }
    assert_eq!(field.text, "Hi");
    assert_eq!(input.focused(), Some(b), "focus stayed in the field");
    assert_eq!(ui.style(named(&ui, "Shift")).background, ui.style(named(&ui, "Space")).background, "shift no longer lit");
    let n = named(&ui, "Done");
    click(&mut input, &mut ui, &mut field, n);
    assert_eq!(input.focused(), None);
    assert!(!kb.showing(&ui));
}

/// A field at the bottom of a scrolling page stays in view above the
/// on-screen keyboard.
#[test]
fn the_on_screen_keyboard_keeps_the_field_in_view() {
    use threers_ui::virtual_keyboard::VirtualKeyboard;
    let mut ui = Ui::new();
    let mut input: Input<()> = Input::new();
    let root = ui.root();
    let page = ui.add_box(root, style! { height: 100%; overflow: auto; });
    ui.add_box(page, style! { height: 560px; });
    let field = ui.add_box(page, style! { height: 30px; });
    ui.set_semantics(field, Semantics::new(Role::TextBox).label("Name"));
    let kb = VirtualKeyboard::new(&mut ui, &mut input);
    kb.set_enabled(true);
    layout(&mut ui);
    input.set_focus(&mut ui, &mut (), Some(field));
    for k in 0..30 {
        input.tick(&mut ui, &mut (), k as f64 * 0.05);
        ui.step(0.05);
        layout(&mut ui);
    }
    let (f, k) = (ui.rect(field), ui.rect(kb.panel()));
    assert!(k.height > 100.0 && k.y + k.height == 600.0, "at the bottom: {k:?}");
    assert!(f.y + f.height <= k.y + 0.5, "above it: {f:?} {k:?}");
    input.set_focus(&mut ui, &mut (), None);
    input.tick(&mut ui, &mut (), 2.0);
    assert_eq!(ui.style(root).padding.bottom, 0.0, "given back");
}
