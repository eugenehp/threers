//! A button taller than its content (stretched by its row, or given a
//! height) centres its content up and down, as a page's `<button>` does;
//! a plain box keeps it at the top, as a `<div>` does.

use threers_ui::{style, FixedWidthText, Role, Semantics, Ui};

/// A row of a tall button and a short `role`d box with `label`; its
/// label's top, from the short box's top, and the box's height.
fn short_in_row(role: Option<Role>, flex: bool) -> (f32, f32) {
    let mut ui = Ui::new();
    let root = ui.root();
    let row = ui.add_box(root, style! { display: flex; flex-direction: row; });
    let tall = ui.add_box(row, style! { padding: 20px; font-size: 20px; });
    ui.add_text(tall, "Tall");
    let short = ui.add_box(row, style! { padding: 4px; font-size: 12px; });
    if flex {
        ui.update_style(short, |s| s.display = threers_ui::Display::Flex);
    }
    if let Some(r) = role {
        ui.set_semantics(short, Semantics::new(r).label("Short"));
    }
    let text = ui.add_text(short, "Short");
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    let (b, t) = (ui.rect(short), ui.rect(text));
    (t.y - b.y, b.height)
}

#[test]
fn a_stretched_button_centres_its_label() {
    for flex in [false, true] {
        let (top, height) = short_in_row(Some(Role::Button), flex);
        let line = 12.0 * 1.2;
        assert!(
            (top - (height - line) / 2.0).abs() < 1.0,
            "flex {flex}: label at {top} in {height}"
        );
    }
}

#[test]
fn a_plain_box_keeps_its_content_at_the_top() {
    let (top, height) = short_in_row(None, false);
    assert!(height > 30.0);
    assert!((top - 4.0).abs() < 0.5, "at its padding: {top}");
}
