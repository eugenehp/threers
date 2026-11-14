//! Focus coming in and going out, overlays that let the wheel through,
//! drags moving over targets, files dropped from outside, context menus,
//! timers, and typing keystrokes in tests.

use std::cell::RefCell;
use std::rc::Rc;

use threers_ui::input::{DroppedFile, Event, HitBehavior, Input, PointerButton, TouchPhase};
use threers_ui::{style, FixedWidthText, Role, Semantics, Ui};

fn layout(ui: &mut Ui) {
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
}

type Log = Vec<String>;

#[test]
fn focus_in_and_out_are_heard_by_whole_subtrees() {
    let mut ui = Ui::new();
    let root = ui.root();
    let panel = ui.add_box(root, style! { height: 100px; });
    let a = ui.add_box(panel, style! { height: 20px; });
    let b = ui.add_box(panel, style! { height: 20px; });
    let outside = ui.add_box(root, style! { height: 20px; });
    for n in [a, b, outside] {
        ui.set_semantics(n, Semantics::new(Role::Button));
    }
    layout(&mut ui);
    let mut input: Input<Log> = Input::new();
    input.on_focus_in(panel, |log, _| log.push("in".into()));
    input.on_focus_out(panel, |log, _| log.push("out".into()));
    let mut log = vec![];
    input.set_focus(&mut ui, &mut log, Some(a));
    input.set_focus(&mut ui, &mut log, Some(b));
    assert_eq!(log, ["in"], "within the panel is no coming in or going out");
    input.set_focus(&mut ui, &mut log, Some(outside));
    input.set_focus(&mut ui, &mut log, None);
    assert_eq!(log, ["in", "out"]);
}

#[test]
fn an_overlay_can_let_the_wheel_scroll_what_is_under_it() {
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(root, style! { height: 100px; overflow: auto; });
    ui.add_box(list, style! { height: 1000px; });
    let overlay = ui.add_box(
        root,
        style! { position: absolute; left: 0px; top: 0px; width: 400px; height: 100px; },
    );
    layout(&mut ui);
    let mut input: Input<Log> = Input::new();
    input.smooth_wheel = false;
    let mut log = vec![];
    input.dispatch(
        &mut ui,
        &mut log,
        Event::Wheel {
            x: 10.0,
            y: 50.0,
            dx: 0.0,
            dy: 30.0,
        },
    );
    assert_eq!(ui.style(list).scroll_top, 0.0, "blocked by the overlay");
    input.set_hit_behavior(overlay, HitBehavior::BlockExceptScroll);
    input.dispatch(
        &mut ui,
        &mut log,
        Event::Wheel {
            x: 10.0,
            y: 50.0,
            dx: 0.0,
            dy: 30.0,
        },
    );
    assert_eq!(ui.style(list).scroll_top, 30.0, "through it");
}

#[test]
fn a_drag_tells_what_it_moves_over() {
    let mut ui = Ui::new();
    let root = ui.root();
    let source = ui.add_box(root, style! { height: 20px; });
    let target = ui.add_box(root, style! { height: 100px; });
    layout(&mut ui);
    let mut input: Input<Log> = Input::new();
    input.draggable(source, |_, _| Some(7u32));
    input.on_drag_move::<u32>(target, |log, _, n, _, y| log.push(format!("{n} at {y}")));
    input.on_drag_move::<String>(target, |log, _, _, _, _| log.push("not a string".into()));
    let mut log = vec![];
    let b = PointerButton::Primary;
    input.dispatch(
        &mut ui,
        &mut log,
        Event::PointerDown {
            x: 5.0,
            y: 5.0,
            button: b,
        },
    );
    input.dispatch(&mut ui, &mut log, Event::PointerMove { x: 5.0, y: 12.0 });
    input.dispatch(&mut ui, &mut log, Event::PointerMove { x: 5.0, y: 40.0 });
    input.dispatch(&mut ui, &mut log, Event::PointerMove { x: 5.0, y: 60.0 });
    input.dispatch(
        &mut ui,
        &mut log,
        Event::PointerUp {
            x: 5.0,
            y: 60.0,
            button: b,
        },
    );
    assert_eq!(log, ["7 at 40", "7 at 60"]);
}

#[test]
fn files_dropped_from_outside_go_to_the_node_that_takes_them() {
    let mut ui = Ui::new();
    let root = ui.root();
    let zone = ui.add_box(root, style! { height: 100px; });
    let inner = ui.add_box(zone, style! { height: 20px; });
    let elsewhere = ui.add_box(root, style! { height: 100px; });
    layout(&mut ui);
    let _ = (inner, elsewhere);
    let mut input: Input<Log> = Input::new();
    input.on_drop_files(zone, |log, _, files, _, _| {
        log.extend(files.iter().map(|f| format!("{} {}", f.name, f.size)))
    });
    let files = [DroppedFile {
        name: "a.png".into(),
        size: 12,
        kind: "image/png".into(),
    }];
    let mut log = vec![];
    assert_eq!(
        input.files_target(&ui, 5.0, 5.0),
        Some(zone),
        "from inside it too"
    );
    assert_eq!(
        input.drop_files(&mut ui, &mut log, 5.0, 5.0, &files),
        Some(zone)
    );
    assert_eq!(
        input.drop_files(&mut ui, &mut log, 5.0, 150.0, &files),
        None
    );
    assert_eq!(log, ["a.png 12"]);
}

#[test]
fn a_secondary_click_or_a_held_finger_asks_for_a_context_menu() {
    let mut ui = Ui::new();
    let root = ui.root();
    let row = ui.add_box(root, style! { height: 40px; });
    let plain = ui.add_box(root, style! { height: 40px; });
    layout(&mut ui);
    let mut input: Input<Log> = Input::new();
    input.on_context_menu(row, |log, _, x, y| log.push(format!("menu at {x},{y}")));
    input.on(
        root,
        threers_ui::input::Phase::Bubble,
        |log: &mut Log, _, e, _| {
            if let Event::Gesture(threers_ui::input::Gesture::LongPress { .. }) = e {
                log.push("long press".into());
            }
        },
    );
    let mut log = vec![];
    input.dispatch(
        &mut ui,
        &mut log,
        Event::PointerDown {
            x: 5.0,
            y: 10.0,
            button: PointerButton::Secondary,
        },
    );
    input.dispatch(
        &mut ui,
        &mut log,
        Event::PointerUp {
            x: 5.0,
            y: 10.0,
            button: PointerButton::Secondary,
        },
    );
    assert_eq!(log, ["menu at 5,10"]);
    log.clear();
    // Held on the row: its menu, and no long press. On the plain box: a
    // long press.
    for (y, t) in [(10.0, 1.0), (60.0, 3.0)] {
        input.dispatch(
            &mut ui,
            &mut log,
            Event::Touch {
                id: 1,
                phase: TouchPhase::Start,
                x: 5.0,
                y,
                time: t,
            },
        );
        input.tick(&mut ui, &mut log, t + 0.7);
        input.dispatch(
            &mut ui,
            &mut log,
            Event::Touch {
                id: 1,
                phase: TouchPhase::End,
                x: 5.0,
                y,
                time: t + 0.8,
            },
        );
    }
    assert_eq!(log, ["menu at 5,10", "long press"]);
    let _ = plain;
}

#[test]
fn timers_run_from_ticks_once_or_every_interval_until_cancelled() {
    let mut ui = Ui::new();
    let mut input: Input<Log> = Input::new();
    let mut log = vec![];
    input.tick(&mut ui, &mut log, 10.0);
    input.after(0.5, |log, _| log.push("once".into()));
    let every = input.every(0.2, |log, _| log.push("tick".into()));
    assert!((input.next_timer().unwrap() - 0.2).abs() < 1e-9);
    for k in 1..=7 {
        input.tick(&mut ui, &mut log, 10.0 + k as f64 * 0.1);
    }
    assert_eq!(
        log,
        ["tick", "tick", "once", "tick"],
        "at 10.2, 10.4, 10.5, 10.6"
    );
    assert!(input.cancel_timer(every));
    log.clear();
    input.tick(&mut ui, &mut log, 12.0);
    assert!(log.is_empty() && input.next_timer().is_none());
}

#[test]
fn keystrokes_type_as_a_keymap_spells_them() {
    let mut ui = Ui::new();
    let root = ui.root();
    let field = ui.add_box(root, style! { height: 20px; });
    ui.set_semantics(field, Semantics::new(Role::TextBox));
    layout(&mut ui);
    let typed = Rc::new(RefCell::new(vec![]));
    let mut input: Input<Log> = Input::new();
    let t = typed.clone();
    input.on_key(field, move |_, _, key, m, _| {
        t.borrow_mut()
            .push(format!("{}{key}", if m.meta { "⌘" } else { "" }));
        true
    });
    let mut log = vec![];
    input.set_focus(&mut ui, &mut log, Some(field));
    input
        .simulate_keystrokes(&mut ui, &mut log, "cmd-a h shift-i enter left")
        .unwrap();
    assert_eq!(*typed.borrow(), ["⌘a", "h", "I", "Enter", "ArrowLeft"]);
}

#[test]
fn shortcuts_are_labelled_as_each_platform_writes_them() {
    use threers_ui::keymap::{Keymap, Keystroke};
    #[derive(Debug, Clone)]
    struct Save;
    let k = |s: &str| Keystroke::parse(s).unwrap();
    assert_eq!(k("cmd-shift-s").label(true), "⇧⌘S");
    assert_eq!(k("ctrl-shift-s").label(false), "Ctrl+Shift+S");
    assert_eq!(k("alt-enter").label(true), "⌥↩");
    assert_eq!(k("ctrl-pageup").label(false), "Ctrl+PgUp");
    let mut keymap = Keymap::default();
    keymap.bind("cmd-k cmd-s", Save, None).unwrap();
    assert_eq!(keymap.label_for("Save", true).as_deref(), Some("⌘K ⌘S"));
    assert_eq!(keymap.label_for("Open", true), None);
}

/// The shortcuts listed where focus is are those whose actions something
/// there handles.
#[test]
fn shortcuts_here_are_the_handled_ones_in_force() {
    use threers_ui::input::Input;
    use threers_ui::{style, FixedWidthText, Ui};
    #[derive(Debug, Clone)]
    struct Save;
    #[derive(Debug, Clone)]
    struct Rename;
    #[derive(Debug, Clone)]
    struct Unhandled;
    let mut ui = Ui::new();
    let mut input: Input<()> = Input::new();
    let root = ui.root();
    let list = ui.add_box(root, style! { width: 100px; height: 100px; });
    let row = ui.add_box(list, style! { width: 100px; height: 20px; });
    ui.layout(200.0, 200.0, &mut FixedWidthText(7.0));
    input.keymap.bind("ctrl-s", Save, None).unwrap();
    input.keymap.bind("f2", Rename, Some("List")).unwrap();
    input.keymap.bind("ctrl-u", Unhandled, None).unwrap();
    input.set_key_context(list, "List");
    input.on_action(root, |_, _, _: &Save, _| {});
    input.on_action(list, |_, _, _: &Rename, _| {});
    let names = |input: &Input<()>, ui: &Ui| {
        input
            .shortcuts_here(ui, false)
            .into_iter()
            .map(|(k, a)| format!("{k}={a}"))
            .collect::<Vec<_>>()
    };
    assert_eq!(names(&input, &ui), ["Ctrl+S=Save"]);
    input.set_focus(&mut ui, &mut (), Some(row));
    assert_eq!(names(&input, &ui), ["Ctrl+S=Save", "F2=Rename"]);
}
