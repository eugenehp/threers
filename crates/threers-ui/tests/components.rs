//! The components: switches, disclosures, tabs, avatars, badges, popovers,
//! split panes, list navigation, notifications, prompts and the command
//! palette, each worked by pointer and keyboard and read as assistive
//! technology reads it.

use threers_ui::a11y::AccessNode;
use threers_ui::binding::Binding;
use threers_ui::components::{avatar, badge, disclosure, popover, switch, tabs};
use threers_ui::input::{Event, Input, PointerButton};
use threers_ui::navigation::{ListNavigation, Nav};
use threers_ui::notifications::{Notice, Notifications, Tone};
use threers_ui::palette::{open_palette, Command};
use threers_ui::prompt::prompt;
use threers_ui::split::{split, Axis};
use threers_ui::{style, Display, FixedWidthText, NodeId, Role, Ui};

type Log = Vec<String>;

fn layout(ui: &mut Ui) {
    ui.layout(600.0, 400.0, &mut FixedWidthText(7.0));
}

fn node(ui: &Ui, id: NodeId) -> AccessNode {
    ui.accessibility_tree().into_iter().find(|a| a.id == id).expect("in the accessibility tree")
}

fn with_role(ui: &Ui, role: Role) -> Vec<AccessNode> {
    ui.accessibility_tree().into_iter().filter(|a| a.role == role).collect()
}

fn centre(ui: &Ui, id: NodeId) -> (f32, f32) {
    let r = ui.rect(id);
    (r.x + r.width / 2.0, r.y + r.height / 2.0)
}

fn click(input: &mut Input<Log>, ui: &mut Ui, log: &mut Log, (x, y): (f32, f32)) {
    let b = PointerButton::Primary;
    input.dispatch(ui, log, Event::PointerMove { x, y });
    input.dispatch(ui, log, Event::PointerDown { x, y, button: b });
    input.dispatch(ui, log, Event::PointerUp { x, y, button: b });
    layout(ui);
}

fn tap(input: &mut Input<Log>, ui: &mut Ui, log: &mut Log, id: NodeId) {
    let at = centre(ui, id);
    click(input, ui, log, at);
}

fn keys(input: &mut Input<Log>, ui: &mut Ui, log: &mut Log, k: &str) {
    input.simulate_keystrokes(ui, log, k).unwrap();
    layout(ui);
}

#[test]
fn a_switch_toggles_by_click_and_keys_and_says_so() {
    let mut ui = Ui::new();
    let mut input: Input<Log> = Input::new();
    let mut log = vec![];
    let root = ui.root();
    let s = switch(&mut ui, &mut input, &log, root, "Wi-Fi", Binding::local(false).on_set(|log: &mut Log, on| log.push(format!("{on}"))));
    layout(&mut ui);
    let a = node(&ui, s);
    assert_eq!((a.role, a.name.as_str(), a.checked, a.focusable), (Role::Switch, "Wi-Fi", Some(false), true));
    tap(&mut input, &mut ui, &mut log, s);
    assert_eq!(node(&ui, s).checked, Some(true));
    input.set_focus(&mut ui, &mut log, Some(s));
    keys(&mut input, &mut ui, &mut log, "space enter");
    assert_eq!(log, ["true", "false", "true"]);
    assert_eq!(node(&ui, s).checked, Some(true));
}

#[test]
fn a_disclosure_shows_and_hides_its_body() {
    let mut ui = Ui::new();
    let mut input: Input<Log> = Input::new();
    let mut log = vec![];
    let root = ui.root();
    let (head, body) = disclosure(&mut ui, &mut input, &log, root, "Advanced", Binding::local(false));
    ui.add_text(body, "hidden things");
    layout(&mut ui);
    assert_eq!(ui.style(body).display, Display::None);
    assert_eq!(node(&ui, head).expanded, Some(false));
    assert_eq!(node(&ui, head).role, Role::Button);
    let settle = |input: &mut Input<Log>, ui: &mut Ui, log: &mut Log, t: &mut f64| {
        for _ in 0..60 {
            *t += 1.0 / 60.0;
            ui.step(1.0 / 60.0);
            input.tick(ui, log, *t);
        }
    };
    let mut t = 0.0;
    tap(&mut input, &mut ui, &mut log, head);
    assert_ne!(ui.style(body).display, Display::None);
    assert_eq!(node(&ui, head).expanded, Some(true));
    // It unfolds from its top edge, then lies flat.
    assert!(ui.transform3d(body).is_some_and(|t| t.rotate_x < -1.0), "folded to start with");
    settle(&mut input, &mut ui, &mut log, &mut t);
    assert!(ui.transform3d(body).is_none(), "flat once open");
    input.set_focus(&mut ui, &mut log, Some(head));
    keys(&mut input, &mut ui, &mut log, "enter");
    assert_eq!(node(&ui, head).expanded, Some(false));
    assert_ne!(ui.style(body).display, Display::None, "still there while it folds");
    settle(&mut input, &mut ui, &mut log, &mut t);
    assert_eq!(ui.style(body).display, Display::None);
}

#[test]
fn tabs_select_by_click_and_arrows_with_one_tab_stop() {
    let mut ui = Ui::new();
    let mut input: Input<Log> = Input::new();
    let mut log = vec![];
    let root = ui.root();
    let t = tabs(&mut ui, &mut input, &log, root, &["One", "Two", "Three"], Binding::local(0).on_set(|log: &mut Log, i| log.push(i.to_string())));
    layout(&mut ui);
    assert_eq!(node(&ui, t.list).role, Role::TabList);
    let selected = |ui: &Ui| with_role(ui, Role::Tab).iter().map(|a| a.selected == Some(true)).collect::<Vec<_>>();
    assert_eq!(selected(&ui), [true, false, false]);
    let shown = |ui: &Ui| t.panels.iter().map(|&p| ui.style(p).display != Display::None).collect::<Vec<_>>();
    assert_eq!(shown(&ui), [true, false, false]);
    // Only the selected tab is in the tab order.
    assert_eq!(ui.focus_order(), [t.tabs[0]]);
    tap(&mut input, &mut ui, &mut log, t.tabs[2]);
    assert_eq!(selected(&ui), [false, false, true]);
    assert_eq!(shown(&ui), [false, false, true]);
    input.set_focus(&mut ui, &mut log, Some(t.tabs[2]));
    keys(&mut input, &mut ui, &mut log, "right");
    assert_eq!(selected(&ui), [true, false, false], "wraps round");
    assert_eq!(input.focused(), Some(t.tabs[0]), "focus follows");
    keys(&mut input, &mut ui, &mut log, "end left");
    assert_eq!(selected(&ui), [false, true, false]);
    assert_eq!(log, ["2", "0", "2", "1"]);
}

#[test]
fn avatars_show_initials_named_for_the_person() {
    let mut ui = Ui::new();
    let root = ui.root();
    let a = avatar(&mut ui, root, "ada lovelace", 32.0);
    let b = badge(&mut ui, root, "3", [0.9, 0.2, 0.2, 1.0]);
    layout(&mut ui);
    let n = node(&ui, a);
    assert_eq!((n.role, n.name.as_str()), (Role::Image, "ada lovelace"));
    assert_eq!(ui.text_lines_of(ui.children(a)[0]), ["AL"]);
    assert_eq!((ui.rect(a).width, ui.rect(a).height), (32.0, 32.0));
    assert!(ui.rect(b).width > 0.0);
}

#[test]
fn a_popover_takes_focus_and_gives_it_back_on_escape_or_outside() {
    let mut ui = Ui::new();
    let mut input: Input<Log> = Input::new();
    let mut log = vec![];
    let root = ui.root();
    let button = ui.add_box(root, style! { width: 80px; height: 24px; });
    ui.set_semantics(button, threers_ui::Semantics::new(Role::Button).label("Open"));
    ui.add_box(root, style! { height: 300px; });
    layout(&mut ui);
    input.set_focus(&mut ui, &mut log, Some(button));
    let p = popover(&mut ui, &mut input, &mut log, button, "Details", |ui, panel| {
        ui.add_text(panel, "Inside");
    });
    layout(&mut ui);
    assert_eq!(node(&ui, p.panel).role, Role::Dialog);
    assert_eq!(node(&ui, p.panel).name, "Details");
    assert_eq!(input.focused(), Some(p.panel));
    assert!(ui.rect(p.panel).y >= ui.rect(button).y + ui.rect(button).height, "under its anchor");
    keys(&mut input, &mut ui, &mut log, "escape");
    assert!(!p.is_open(&ui));
    assert_eq!(input.focused(), Some(button));
    // Again, closed by a press elsewhere.
    let p = popover(&mut ui, &mut input, &mut log, button, "Details", |_, _| {});
    layout(&mut ui);
    click(&mut input, &mut ui, &mut log, (300.0, 350.0));
    assert!(!p.is_open(&ui));
}

#[test]
fn a_split_handle_drags_keys_and_resets() {
    let mut ui = Ui::new();
    let mut input: Input<Log> = Input::new();
    let mut log = vec![];
    let root = ui.root();
    let host = ui.add_box(root, style! { display: flex; width: 606px; height: 200px; });
    let sp = split(&mut ui, &mut input, host, Axis::Horizontal, 200.0, 100.0);
    layout(&mut ui);
    // The first pane's size is a share of 600 (606 less the handle).
    assert_eq!(ui.rect(sp.first).width, 200.0);
    let h = node(&ui, sp.handle);
    assert_eq!((h.role, h.focusable), (Role::Separator, true));
    // Dragged to x = 403: the pane is 400 (the pointer at the handle's middle).
    let (hx, hy) = centre(&ui, sp.handle);
    let b = PointerButton::Primary;
    input.dispatch(&mut ui, &mut log, Event::PointerDown { x: hx, y: hy, button: b });
    input.dispatch(&mut ui, &mut log, Event::PointerMove { x: 403.0, y: hy });
    input.dispatch(&mut ui, &mut log, Event::PointerUp { x: 403.0, y: hy, button: b });
    layout(&mut ui);
    assert_eq!(ui.rect(sp.first).width, 400.0);
    assert_eq!(node(&ui, sp.handle).range.map(|r| r.now), Some(67.0));
    // Not past the second pane's minimum.
    input.dispatch(&mut ui, &mut log, Event::PointerDown { x: 403.0, y: hy, button: b });
    input.dispatch(&mut ui, &mut log, Event::PointerMove { x: 590.0, y: hy });
    input.dispatch(&mut ui, &mut log, Event::PointerUp { x: 590.0, y: hy, button: b });
    layout(&mut ui);
    assert_eq!(ui.rect(sp.first).width, 500.0);
    input.set_focus(&mut ui, &mut log, Some(sp.handle));
    keys(&mut input, &mut ui, &mut log, "left left");
    assert_eq!(ui.rect(sp.first).width, 468.0);
    keys(&mut input, &mut ui, &mut log, "home");
    assert_eq!(ui.rect(sp.first).width, 100.0);
    // A double-click puts it back (presses count together by the clock).
    input.tick(&mut ui, &mut log, 1.0);
    let (hx, hy) = centre(&ui, sp.handle);
    click(&mut input, &mut ui, &mut log, (hx, hy));
    click(&mut input, &mut ui, &mut log, (hx, hy));
    assert_eq!(ui.rect(sp.first).width, 200.0);
}

#[test]
fn list_navigation_moves_pages_and_activates() {
    let mut nav = ListNavigation::new(5);
    assert_eq!(nav.key("Enter", 2), None, "nothing highlighted yet");
    assert_eq!(nav.key("ArrowUp", 2), Some(Nav::Moved(4)), "from nothing, up is the last");
    assert_eq!(nav.key("ArrowDown", 2), Some(Nav::Moved(4)), "stays at the end");
    assert_eq!(nav.key("PageUp", 2), Some(Nav::Moved(2)));
    assert_eq!(nav.key("Home", 2), Some(Nav::Moved(0)));
    assert_eq!(nav.key(" ", 2), Some(Nav::Activated(0)));
    assert_eq!(nav.key("x", 2), None);
    assert_eq!(ListNavigation::new(0).key("ArrowDown", 2), None);
}

#[test]
fn notifications_stack_time_out_and_act() {
    let mut ui = Ui::new();
    let mut input: Input<Log> = Input::new();
    let mut log = vec![];
    input.tick(&mut ui, &mut log, 0.0);
    let mut stack: Notifications<Log> = Notifications::new(&mut ui);
    stack.max = 2;
    let a = stack.push(&mut ui, &mut input, Notice::new("Saved", "All changes"), None);
    let b = stack.push(&mut ui, &mut input, Notice::new("Failed", "Disk full").tone(Tone::Error).action("Retry"), Some(Box::new(|log: &mut Log, _: &mut Ui| log.push("retry".into()))));
    layout(&mut ui);
    let na = node(&ui, a);
    assert_eq!((na.role, na.live, na.name.as_str()), (Role::Status, true, "Saved. All changes"));
    assert_eq!(node(&ui, b).role, Role::Alert);
    // A third pushes out the oldest.
    let c = stack.push(&mut ui, &mut input, Notice::new("Third", "").timeout(Some(1.0)), None);
    assert_eq!(stack.shown(&ui), [b, c]);
    // The info one times out; the error stays.
    input.tick(&mut ui, &mut log, 6.0);
    input.tick(&mut ui, &mut log, 6.1);
    layout(&mut ui);
    assert_eq!(stack.shown(&ui), [b]);
    let retry = ui.accessibility_tree().into_iter().find(|n| n.name == "Retry").unwrap().id;
    tap(&mut input, &mut ui, &mut log, retry);
    assert_eq!(log, ["retry"]);
    assert!(stack.shown(&ui).is_empty(), "acted on, it closes");
}

#[test]
fn a_prompt_is_modal_and_answers_once() {
    let mut ui = Ui::new();
    let mut input: Input<Log> = Input::new();
    let mut log = vec![];
    let root = ui.root();
    let field = ui.add_box(root, style! { height: 20px; });
    ui.set_semantics(field, threers_ui::Semantics::new(Role::TextBox));
    layout(&mut ui);
    input.set_focus(&mut ui, &mut log, Some(field));
    let p = prompt(&mut ui, &mut input, &mut log, "Delete?", "It cannot be undone.", &["Delete", "Cancel"], |log, _, i| log.push(i.to_string()));
    layout(&mut ui);
    let d = node(&ui, p.dialog);
    assert_eq!((d.role, d.name.as_str(), d.description.as_deref()), (Role::AlertDialog, "Delete?", Some("It cannot be undone.")));
    assert_eq!(input.modal(), Some(p.dialog));
    ui.set_system_appearance(threers_ui::states::Appearance::Light);
    layout(&mut ui);
    assert_eq!((ui.rect(p.dialog).x, ui.rect(p.dialog).y), (140.0, 100.0), "centred, in either appearance");
    let buttons = with_role(&ui, Role::Button);
    assert_eq!(input.focused(), Some(buttons[0].id), "the first button");
    // Tab keeps to the dialog; Enter presses the focused button.
    keys(&mut input, &mut ui, &mut log, "tab");
    assert_eq!(input.focused(), Some(buttons[1].id));
    keys(&mut input, &mut ui, &mut log, "enter");
    assert_eq!(log, ["1"]);
    assert!(!ui.contains(p.dialog));
    assert_eq!(input.modal(), None);
    assert_eq!(input.focused(), Some(field), "focus goes back");
}

#[test]
fn the_palette_filters_navigates_and_runs() {
    let mut ui = Ui::new();
    let mut input: Input<Log> = Input::new();
    let mut log = vec![];
    layout(&mut ui);
    let commands = vec![
        Command::new("open", "Open File").shortcut("⌘O"),
        Command::new("sidebar", "Toggle Sidebar"),
        Command::new("settings", "Open Settings").shortcut("⌘,"),
    ];
    let panel = open_palette(&mut ui, &mut input, &mut log, commands.clone(), |log, _, id| log.push(id.to_owned()));
    layout(&mut ui);
    assert_eq!(node(&ui, panel).role, Role::Dialog);
    // Centred, in either appearance (its place is in the base under its
    // light style).
    ui.set_system_appearance(threers_ui::states::Appearance::Light);
    layout(&mut ui);
    assert_eq!((ui.rect(panel).x, ui.rect(panel).y), (80.0, 48.0));
    assert_eq!(with_role(&ui, Role::ComboBox).len(), 1);
    let options = |ui: &Ui| with_role(ui, Role::Option).into_iter().map(|a| (a.name, a.selected == Some(true))).collect::<Vec<_>>();
    assert_eq!(options(&ui).len(), 3);
    let opt = with_role(&ui, Role::Option);
    assert_eq!(opt[0].description.as_deref(), Some("⌘O"), "its shortcut read out");
    // To the byte: tokens reach a style as CSS colours.
    let shortcut = |ui: &Ui, k: usize| {
        let o = with_role(ui, Role::Option)[k].id;
        ui.color_of(ui.children(*ui.children(o).last().unwrap())[0]).map(|c| (c * 255.0).round() as u8)
    };
    let byte = |c: [f32; 4]| c.map(|c| (c * 255.0).round() as u8);
    assert_eq!(shortcut(&ui, 0), byte(ui.token(threers_ui::environment::Token::OnAccent)), "white on the highlight");
    assert_eq!(shortcut(&ui, 1), byte(ui.token(threers_ui::environment::Token::SecondaryText)));
    keys(&mut input, &mut ui, &mut log, "o p");
    let names: Vec<String> = options(&ui).into_iter().map(|o| o.0).collect();
    assert_eq!(names, ["Open File", "Open Settings"]);
    keys(&mut input, &mut ui, &mut log, "down");
    assert_eq!(options(&ui), [("Open File".to_owned(), false), ("Open Settings".to_owned(), true)]);
    keys(&mut input, &mut ui, &mut log, "enter");
    assert_eq!(log, ["settings"]);
    assert!(!ui.contains(panel));
    // Escape closes without running; a click on a row runs it.
    let panel = open_palette(&mut ui, &mut input, &mut log, commands.clone(), |log, _, id| log.push(id.to_owned()));
    layout(&mut ui);
    keys(&mut input, &mut ui, &mut log, "escape");
    assert!(!ui.contains(panel));
    let panel = open_palette(&mut ui, &mut input, &mut log, commands, |log, _, id| log.push(id.to_owned()));
    layout(&mut ui);
    let row = with_role(&ui, Role::Option)[1].id;
    tap(&mut input, &mut ui, &mut log, row);
    assert_eq!(log, ["settings", "sidebar"]);
    assert!(!ui.contains(panel));
}

/// Two switches bound to one value show it however it changes: by either,
/// or by the app.
#[test]
fn bound_controls_show_the_state_however_it_changes() {
    #[derive(Default)]
    struct S {
        wifi: bool,
        tab: usize,
    }
    let mut ui = Ui::new();
    let mut input: Input<S> = Input::new();
    let mut s = S::default();
    let root = ui.root();
    let a = switch(&mut ui, &mut input, &s, root, "Wi-Fi", threers_ui::binding!(|s: S| s.wifi));
    let b = switch(&mut ui, &mut input, &s, root, "Wi-Fi too", threers_ui::binding!(|s: S| s.wifi));
    let t = tabs(&mut ui, &mut input, &s, root, &["One", "Two"], threers_ui::binding!(|s: S| s.tab));
    ui.layout(600.0, 400.0, &mut FixedWidthText(7.0));
    let (x, y) = centre(&ui, a);
    input.dispatch(&mut ui, &mut s, Event::PointerDown { x, y, button: PointerButton::Primary });
    input.dispatch(&mut ui, &mut s, Event::PointerUp { x, y, button: PointerButton::Primary });
    assert!(s.wifi);
    assert_eq!((node(&ui, a).checked, node(&ui, b).checked), (Some(true), Some(true)));
    // The app changes it: shown on the next tick.
    s.wifi = false;
    s.tab = 1;
    input.tick(&mut ui, &mut s, 0.1);
    assert_eq!((node(&ui, a).checked, node(&ui, b).checked), (Some(false), Some(false)));
    assert_eq!(node(&ui, t.tabs[1]).selected, Some(true));
}

/// A prompt shown while the state says so: answering clears the flag;
/// clearing the flag takes it away, and modality and focus go back.
#[test]
fn a_prompt_presented_from_state() {
    #[derive(Default)]
    struct S {
        asking: bool,
        answers: Vec<usize>,
    }
    let mut ui = Ui::new();
    let mut input: Input<S> = Input::new();
    let mut s = S::default();
    let root = ui.root();
    let field = ui.add_box(root, style! { height: 20px; });
    ui.set_semantics(field, threers_ui::Semantics::new(Role::TextBox));
    ui.layout(600.0, 400.0, &mut FixedWidthText(7.0));
    input.set_focus(&mut ui, &mut s, Some(field));
    input.present(|s: &S| s.asking, |s| s.asking = false, |ui, input, s| {
        prompt(ui, input, s, "Sure?", "", &["Yes", "No"], |s: &mut S, _, i| s.answers.push(i)).dialog
    });
    let dialogs = |ui: &Ui| with_role(ui, Role::AlertDialog).len();
    input.tick(&mut ui, &mut s, 0.0);
    assert_eq!(dialogs(&ui), 0);
    s.asking = true;
    input.tick(&mut ui, &mut s, 0.1);
    ui.layout(600.0, 400.0, &mut FixedWidthText(7.0));
    assert_eq!(dialogs(&ui), 1);
    input.simulate_keystrokes(&mut ui, &mut s, "enter").unwrap();
    input.tick(&mut ui, &mut s, 0.2);
    assert_eq!((dialogs(&ui), s.asking, s.answers.clone()), (0, false, vec![0]), "answered: the flag is cleared");
    assert_eq!(input.focused(), Some(field));
    // Asked again, then withdrawn by the app.
    s.asking = true;
    input.tick(&mut ui, &mut s, 0.3);
    assert_eq!(dialogs(&ui), 1);
    assert!(input.modal().is_some());
    s.asking = false;
    input.tick(&mut ui, &mut s, 0.4);
    input.tick(&mut ui, &mut s, 0.5);
    assert_eq!(dialogs(&ui), 0);
    assert_eq!(input.modal(), None);
    assert_eq!(input.focused(), Some(field), "focus goes back");
    assert_eq!(s.answers, [0], "withdrawn, not answered");
}

/// Timers tied to a node go with it, and its going is heard.
#[test]
fn work_tied_to_a_node_ends_with_it() {
    let mut ui = Ui::new();
    let mut input: Input<Log> = Input::new();
    let mut log = vec![];
    let root = ui.root();
    let n = ui.add_box(root, style! {});
    input.tick(&mut ui, &mut log, 0.0);
    input.on_appear(n, |_, _, log| log.push("appeared".into()));
    input.every_for(n, 0.5, |log, _| log.push("tick".into()));
    input.on_disappear(n, |log, _| log.push("gone".into()));
    input.tick(&mut ui, &mut log, 0.6);
    ui.remove(n);
    input.tick(&mut ui, &mut log, 1.2);
    input.tick(&mut ui, &mut log, 2.0);
    assert_eq!(log, ["tick", "appeared", "gone"]);
    assert_eq!(input.next_timer(), None);
}

/// Keyed rows keep their nodes as they reorder; changed ones are patched.
#[test]
fn a_view_keeps_keyed_nodes_and_patches_what_changed() {
    use threers_ui::view::{El, View};
    let mut ui = Ui::new();
    let mut input: Input<Log> = Input::new();
    let root = ui.root();
    let mut view = View::new(&mut ui, root, "display: flex; flex-direction: column");
    let rows = |items: &[(&str, &str)]| -> Vec<El<Log>> {
        items.iter().map(|(k, t)| El::div("padding: 2px").key(*k).role(Role::ListItem, *t).text(*t)).collect()
    };
    view.update(&mut ui, &mut input, rows(&[("a", "Apple"), ("b", "Banana"), ("c", "Cherry")]));
    let before: Vec<NodeId> = ui.children(view.root()).to_vec();
    view.update(&mut ui, &mut input, rows(&[("c", "Cherry"), ("a", "Apricot")]));
    let after: Vec<NodeId> = ui.children(view.root()).to_vec();
    assert_eq!(after, [before[2], before[0]], "the same nodes, reordered; b's gone");
    assert!(!ui.contains(before[1]));
    let names: Vec<String> = with_role(&ui, Role::ListItem).into_iter().map(|a| a.name).collect();
    assert_eq!(names, ["Cherry", "Apricot"]);
    assert_eq!(ui.text_lines_of(ui.children(after[1])[0]), ["Apricot"]);
}

/// The components as previews: every role, box and colour, light and
/// dark, and with larger text, kept in `tests/snapshots/`.
#[test]
fn component_previews_match_their_snapshots() {
    use threers_ui::preview::{assert_snapshot, Preview};
    #[derive(Default)]
    struct S {
        on: bool,
        tab: usize,
    }
    let out = Preview::new().size(600.0, 400.0).text_scales(&[1.0, 1.5]).render(|ui, input, s: &mut S| {
        let root = ui.root();
        let col = ui.add_box(root, style! { display: flex; flex-direction: column; gap: 8px; padding: 8px; });
        switch(ui, input, s, col, "Wi-Fi", threers_ui::binding!(|s: S| s.on));
        tabs(ui, input, s, col, &["One", "Two"], threers_ui::binding!(|s: S| s.tab));
    });
    // A modal prompt is all assistive technology sees while it is open.
    let out = out + &Preview::new().size(600.0, 400.0).text_scales(&[1.0, 1.5]).render(|ui, input, s: &mut S| {
        prompt(ui, input, s, "Delete?", "It cannot be undone.", &["Delete", "Cancel"], |_, _, _| {});
    });
    assert_snapshot(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/snapshots/components.txt"), &out);
}

/// On a phone the palette is the window's width less a margin, a long name
/// is cut short with its shortcut still showing, and it follows the window
/// as it changes size.
#[test]
fn the_palette_fits_a_narrow_window_and_follows_it() {
    let mut ui = Ui::new();
    let mut input: Input<Log> = Input::new();
    let mut log = vec![];
    let small = |ui: &mut Ui| ui.layout(300.0, 500.0, &mut FixedWidthText(7.0));
    small(&mut ui);
    let commands = vec![Command::new("long", "Open the Recently Closed Project in a New Window").shortcut("⌘⇧O")];
    let panel = open_palette(&mut ui, &mut input, &mut log, commands, |_, _, _| {});
    small(&mut ui);
    let r = ui.rect(panel);
    assert_eq!((r.x, r.width), (8.0, 284.0), "the window less a margin each side");
    let option = with_role(&ui, Role::Option)[0].id;
    let row = ui.rect(option);
    let keys = ui.rect(*ui.children(option).last().unwrap());
    assert!(keys.x + keys.width <= row.x + row.width + 0.5, "the shortcut shows: {keys:?} in {row:?}");
    // Wider: back to its own width, centred.
    input.tick(&mut ui, &mut log, 0.0);
    ui.layout(1000.0, 600.0, &mut FixedWidthText(7.0));
    input.tick(&mut ui, &mut log, 0.1);
    ui.layout(1000.0, 600.0, &mut FixedWidthText(7.0));
    let r = ui.rect(panel);
    assert_eq!((r.x, r.width), (280.0, 440.0));
}
