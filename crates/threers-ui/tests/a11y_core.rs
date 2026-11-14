//! Core accessibility: modal dialogs keep assistive technology in, names
//! and descriptions from other nodes, the states a screen reader reads,
//! places in long lists, announcements, the focus ring, and the audit.

use threers_ui::a11y_core::Audit;
use threers_ui::input::Input;
use threers_ui::scene::Item;
use threers_ui::virtual_list::{RowHeight, VirtualList};
use threers_ui::{style, FixedWidthText, Role, Semantics, Ui};

fn layout(ui: &mut Ui) {
    ui.layout(600.0, 400.0, &mut FixedWidthText(7.0));
}

#[test]
fn a_modal_dialog_is_all_assistive_technology_sees() {
    let mut ui = Ui::new();
    let mut input: Input<()> = Input::new();
    let root = ui.root();
    let behind = ui.add_box(root, style! {});
    ui.set_semantics(behind, Semantics::new(Role::Button).label("Behind"));
    let dialog = ui.add_box(root, style! {});
    ui.set_semantics(dialog, Semantics::new(Role::Dialog).label("Settings"));
    let ok = ui.add_box(dialog, style! {});
    ui.set_semantics(ok, Semantics::new(Role::Button).label("OK"));
    layout(&mut ui);
    input.set_modal(&mut ui, &mut (), Some(dialog));
    let names: Vec<String> = ui
        .accessibility_tree()
        .into_iter()
        .map(|a| a.name)
        .collect();
    assert_eq!(names, ["", "Settings", "OK"]);
    assert!(ui.accessibility_tree()[1].modal);
    input.set_modal(&mut ui, &mut (), None);
    assert!(ui.accessibility_tree().iter().any(|a| a.name == "Behind"));
}

#[test]
fn names_and_descriptions_from_other_nodes_and_the_states_read() {
    let mut ui = Ui::new();
    let root = ui.root();
    let label = ui.add_box(root, style! {});
    ui.add_text(label, "Email address");
    let help = ui.add_box(root, style! {});
    ui.add_text(help, "We never share it");
    let field = ui.add_box(root, style! { width: 200px; height: 30px; });
    let mut sem = Semantics::new(Role::TextBox);
    sem.labelled_by = Some(label);
    sem.described_by = Some(help);
    sem.invalid = Some("Needs an @".into());
    sem.shortcut = Some("Alt+E".into());
    ui.set_semantics(field, sem);
    let list = ui.add_box(root, style! {});
    ui.set_redacted(list, true);
    ui.set_semantics(list, Semantics::new(Role::List).label("Results"));
    layout(&mut ui);
    let tree = ui.accessibility_tree();
    let f = tree.iter().find(|a| a.id == field).unwrap();
    assert_eq!(f.name, "Email address");
    assert_eq!(f.description.as_deref(), Some("We never share it"));
    assert_eq!(
        (f.invalid.as_deref(), f.shortcut.as_deref()),
        (Some("Needs an @"), Some("Alt+E"))
    );
    assert!(
        tree.iter().find(|a| a.id == list).unwrap().busy,
        "a placeholder is busy"
    );
}

#[test]
fn rows_of_a_virtual_list_know_their_place_in_all_of_it() {
    let mut ui = Ui::new();
    let root = ui.root();
    let mut list = VirtualList::new(
        &mut ui,
        root,
        style! { height: 100px; overflow: auto; },
        100_000,
        RowHeight::Uniform(20.0),
    );
    layout(&mut ui);
    list.sync(&mut ui, |ui, row, i| {
        ui.add_text(row, format!("Row {}", i + 1));
    });
    layout(&mut ui);
    let tree = ui.accessibility_tree();
    let first = tree.iter().find(|a| a.name == "Row 1").unwrap();
    assert_eq!(
        (first.role, first.position),
        (Role::ListItem, Some((1, 100_000)))
    );
    list.set_len(&mut ui, 50);
    list.sync(&mut ui, |ui, row, i| {
        ui.add_text(row, format!("Row {}", i + 1));
    });
    let first = ui
        .accessibility_tree()
        .into_iter()
        .find(|a| a.name == "Row 1")
        .unwrap();
    assert_eq!(
        first.position,
        Some((1, 50)),
        "kept right as the list changes"
    );
}

#[test]
fn announcements_are_kept_in_order_with_their_urgency() {
    let mut ui = Ui::new();
    ui.announce("Saved");
    ui.announce_assertive("Connection lost");
    let all = ui.announcements_since(0);
    assert_eq!(
        all.iter().map(|a| (a.1.as_str(), a.2)).collect::<Vec<_>>(),
        [("Saved", false), ("Connection lost", true)]
    );
    assert_eq!(ui.announcements_since(all[0].0).len(), 1);
    for i in 0..20 {
        ui.announce(format!("n{i}"));
    }
    assert_eq!(ui.announcements_since(0).len(), 8, "only a few are kept");
}

#[test]
fn a_ring_shows_round_a_node_focused_from_the_keyboard_without_its_own_style() {
    let mut ui = Ui::new();
    let mut input: Input<()> = Input::new();
    let root = ui.root();
    let a = ui.add_box(root, style! { width: 100px; height: 30px; });
    ui.set_semantics(a, Semantics::new(Role::Button).label("Plain"));
    let b = ui.add_box(root, style! { width: 100px; height: 30px; });
    ui.set_semantics(b, Semantics::new(Role::Button).label("Styled"));
    ui.focus_visible_style(b, "background: #ff0000");
    layout(&mut ui);
    let ring = |ui: &Ui| {
        let focus = ui.token(threers_ui::environment::Token::Focus);
        ui.scene()
            .items
            .iter()
            .any(|i| matches!(i, Item::Rect { color, .. } if *color == focus))
    };
    input.simulate_keystrokes(&mut ui, &mut (), "tab").unwrap();
    assert_eq!(input.focused(), Some(a));
    assert!(ring(&ui), "a ring round the plain one");
    input.simulate_keystrokes(&mut ui, &mut (), "tab").unwrap();
    assert!(!ring(&ui), "the styled one shows its own");
    ui.set_focus_ring(false);
    input
        .simulate_keystrokes(&mut ui, &mut (), "shift-tab")
        .unwrap();
    assert!(!ring(&ui));
}

#[test]
fn the_audit_finds_what_will_trouble_people() {
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { background: #ffffff; });
    let unnamed = ui.add_box(root, style! { width: 40px; height: 40px; });
    ui.set_semantics(unnamed, Semantics::new(Role::Button));
    let tiny = ui.add_box(root, style! { width: 16px; height: 16px; });
    ui.set_semantics(tiny, Semantics::new(Role::Button).label("Close"));
    let pic = ui.add_box(root, style! { width: 40px; height: 40px; });
    ui.set_semantics(pic, Semantics::new(Role::Image));
    let h1 = ui.add_box(root, style! {});
    ui.add_text(h1, "Title");
    ui.set_semantics(h1, Semantics::new(Role::Heading(1)));
    let h3 = ui.add_box(root, style! {});
    ui.add_text(h3, "Skipped");
    ui.set_semantics(h3, Semantics::new(Role::Heading(3)));
    let hidden = ui.add_box(root, style! { width: 40px; height: 40px; });
    let mut sem = Semantics::new(Role::Button).label("Ghost");
    sem.hidden = true;
    ui.set_semantics(hidden, sem);
    let faint = ui.add_box(root, style! { color: #eeeeee; });
    ui.add_text(faint, "faint");
    let fine = ui.add_box(root, style! { width: 60px; height: 30px; color: #000000; });
    ui.set_semantics(fine, Semantics::new(Role::Button).label("Fine"));
    ui.add_text(fine, "Fine");
    layout(&mut ui);
    let found = ui.accessibility_audit();
    let has = |k: Audit, id| found.iter().any(|i| i.kind == k && i.id == id);
    assert!(has(Audit::NoName, unnamed));
    assert!(has(Audit::SmallTarget, tiny));
    assert!(has(Audit::ImageNoName, pic));
    assert!(has(Audit::HeadingSkipsLevel, h3));
    assert!(has(Audit::HiddenFocusable, hidden));
    assert!(found.iter().any(|i| i.kind == Audit::LowContrast));
    assert!(
        !found.iter().any(|i| i.id == fine),
        "nothing wrong with this one: {found:?}"
    );
}
