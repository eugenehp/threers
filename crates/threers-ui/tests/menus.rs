//! Context menus and popover menus: the keyboard, the pointer, submenus,
//! closing, and focus going back.

use std::cell::RefCell;
use std::rc::Rc;

use threers_ui::input::{Event, Input, PointerButton};
use threers_ui::menu::{open_menu, popover_menu, MenuAt, MenuHandle, MenuItem};
use threers_ui::{style, FixedWidthText, NodeId, Role, Semantics, Ui};

type Log = Vec<String>;

fn layout(ui: &mut Ui) {
    ui.layout(600.0, 400.0, &mut FixedWidthText(7.0));
}

fn items() -> Vec<MenuItem> {
    vec![
        MenuItem::header("Edit"),
        MenuItem::action("cut", "Cut").shortcut("⌘X"),
        MenuItem::action("copy", "Copy").shortcut("⌘C"),
        MenuItem::action("paste", "Paste").disabled(true),
        MenuItem::separator(),
        MenuItem::submenu(
            "Share",
            vec![
                MenuItem::action("mail", "Mail"),
                MenuItem::action("link", "Copy Link"),
            ],
        ),
    ]
}

fn names(ui: &Ui) -> Vec<String> {
    ui.accessibility_tree()
        .into_iter()
        .filter(|a| a.role == Role::MenuItem)
        .map(|a| a.name)
        .collect()
}

fn centre(ui: &Ui, name: &str) -> (f32, f32) {
    let b = ui
        .accessibility_tree()
        .into_iter()
        .find(|a| a.name == name)
        .unwrap_or_else(|| panic!("no {name}"))
        .bounds;
    (b.x + b.width / 2.0, b.y + b.height / 2.0)
}

fn click(input: &mut Input<Log>, ui: &mut Ui, log: &mut Log, (x, y): (f32, f32)) {
    let b = PointerButton::Primary;
    input.dispatch(ui, log, Event::PointerMove { x, y });
    input.dispatch(ui, log, Event::PointerDown { x, y, button: b });
    input.dispatch(ui, log, Event::PointerUp { x, y, button: b });
    layout(ui);
}

#[test]
fn a_right_click_opens_a_context_menu_the_keyboard_works() {
    let mut ui = Ui::new();
    let root = ui.root();
    let area = ui.add_box(root, style! { height: 300px; });
    let field = ui.add_box(root, style! { height: 20px; });
    ui.set_semantics(field, Semantics::new(Role::TextBox));
    layout(&mut ui);
    let mut input: Input<Log> = Input::new();
    let mut log = vec![];
    input.set_focus(&mut ui, &mut log, Some(field));
    // The menu opens from the context-menu listener (it needs the input:
    // the app opens it after the event).
    let wanted: Rc<RefCell<Option<(f32, f32)>>> = Rc::default();
    let w = wanted.clone();
    input.on_context_menu(area, move |_, _, x, y| *w.borrow_mut() = Some((x, y)));
    input.dispatch(
        &mut ui,
        &mut log,
        Event::PointerDown {
            x: 50.0,
            y: 60.0,
            button: PointerButton::Secondary,
        },
    );
    input.dispatch(
        &mut ui,
        &mut log,
        Event::PointerUp {
            x: 50.0,
            y: 60.0,
            button: PointerButton::Secondary,
        },
    );
    let (x, y) = wanted.borrow_mut().take().expect("asked for");
    // The press took focus from the field, as a press anywhere does; the
    // menu gives back whatever had it when it opened: the field, here.
    assert_eq!(input.focused(), None);
    input.set_focus(&mut ui, &mut log, Some(field));
    let menu = open_menu(
        &mut ui,
        &mut input,
        &mut log,
        MenuAt::Point(x, y),
        items(),
        |log, _, id| log.push(id.to_owned()),
    );
    layout(&mut ui);
    assert_eq!(names(&ui), ["Cut", "Copy", "Paste", "Share"]);
    let r = ui.rect(menu.root());
    assert_eq!((r.x, r.y), (50.0, 60.0), "at the click");
    // Down past the header to Cut, Copy; again skips disabled Paste to Share.
    input
        .simulate_keystrokes(&mut ui, &mut log, "down down down")
        .unwrap();
    // Right opens Share's submenu, with its first item: Mail. Left closes it.
    input
        .simulate_keystrokes(&mut ui, &mut log, "right")
        .unwrap();
    layout(&mut ui);
    assert!(names(&ui).contains(&"Mail".to_owned()));
    input
        .simulate_keystrokes(&mut ui, &mut log, "left")
        .unwrap();
    layout(&mut ui);
    assert!(
        !names(&ui).contains(&"Mail".to_owned()) && menu.is_open(&ui),
        "the submenu only"
    );
    // Into it again and pick the second: both menus close, focus goes back.
    input
        .simulate_keystrokes(&mut ui, &mut log, "right down enter")
        .unwrap();
    layout(&mut ui);
    assert_eq!(log, ["link"]);
    assert!(!menu.is_open(&ui) && names(&ui).is_empty());
    assert_eq!(input.focused(), Some(field));
}

#[test]
fn the_pointer_highlights_opens_submenus_picks_and_a_click_outside_closes() {
    let mut ui = Ui::new();
    let root = ui.root();
    ui.add_box(root, style! { height: 300px; });
    layout(&mut ui);
    let mut input: Input<Log> = Input::new();
    let mut log = vec![];
    let open = |ui: &mut Ui, input: &mut Input<Log>, log: &mut Log| -> MenuHandle {
        let m = open_menu(
            ui,
            input,
            log,
            MenuAt::Point(20.0, 20.0),
            items(),
            |log, _, id| log.push(id.to_owned()),
        );
        layout(ui);
        m
    };
    let menu = open(&mut ui, &mut input, &mut log);
    // Disabled: no pick, still open.
    let paste = centre(&ui, "Paste");
    click(&mut input, &mut ui, &mut log, paste);
    assert!(log.is_empty() && menu.is_open(&ui));
    // Over Share: its submenu opens; a click in it picks.
    let share = centre(&ui, "Share");
    input.dispatch(
        &mut ui,
        &mut log,
        Event::PointerMove {
            x: share.0,
            y: share.1,
        },
    );
    layout(&mut ui);
    let mail = centre(&ui, "Mail");
    click(&mut input, &mut ui, &mut log, mail);
    assert_eq!(log, ["mail"]);
    assert!(!menu.is_open(&ui));
    // Opened again: a click outside closes it, picking nothing.
    let menu = open(&mut ui, &mut input, &mut log);
    click(&mut input, &mut ui, &mut log, (500.0, 350.0));
    assert!(!menu.is_open(&ui) && log == ["mail"]);
}

#[test]
fn a_popover_menu_opens_below_its_button_and_closes_on_a_second_click() {
    let mut ui = Ui::new();
    let root = ui.root();
    let button = ui.add_box(root, style! { width: 80px; height: 24px; margin: 30px; });
    ui.set_semantics(button, Semantics::new(Role::Button).label("More"));
    layout(&mut ui);
    let mut input: Input<Log> = Input::new();
    popover_menu(
        &mut input,
        button,
        |_| {
            vec![
                MenuItem::action("a", "Alpha"),
                MenuItem::action("b", "Beta"),
            ]
        },
        |log, _, id| log.push(id.to_owned()),
    );
    let mut log = vec![];
    let at = centre(&ui, "More");
    click(&mut input, &mut ui, &mut log, at);
    let (bx, by) = centre(&ui, "Alpha");
    let b = ui.rect(button);
    assert!(
        by > b.y + b.height && (bx - (b.x + 40.0)).abs() < 80.0,
        "below the button"
    );
    click(&mut input, &mut ui, &mut log, at);
    assert!(names(&ui).is_empty(), "the second click closes it");
    click(&mut input, &mut ui, &mut log, at);
    input
        .simulate_keystrokes(&mut ui, &mut log, "down down enter")
        .unwrap();
    assert_eq!(log, ["b"]);
    let _: Option<NodeId> = None;
}
