//! `white-space` is inherited, as in CSS: a box that does not set it wraps
//! (or keeps newlines) as its parent does, however deep, and a box that
//! sets it keeps its own.

use threers_ui::{style, FixedWidthText, Length, Ui};

const LONG: &str = "one two three four five six seven eight nine ten eleven twelve";

/// The lines of the text node in `b`, laid out at 120px wide.
fn lines(ui: &Ui, b: threers_ui::NodeId) -> usize {
    ui.text_lines_of(ui.children(b)[0]).len()
}

#[test]
fn a_nested_box_wraps_as_its_ancestor_says() {
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { white-space: normal; });
    let card = ui.add_box(root, style! { width: 120px; display: flex; flex-direction: column; });
    let p = ui.add_box(card, style! { font-size: 13px; });
    ui.add_text(p, LONG);
    let own = ui.add_box(card, style! { white-space: nowrap; });
    ui.add_text(own, LONG);
    ui.layout(400.0, 400.0, &mut FixedWidthText(7.0));
    assert!(lines(&ui, p) > 1, "inherited `normal` wraps");
    assert_eq!(lines(&ui, own), 1, "its own `nowrap` wins");
}

#[test]
fn restyling_the_ancestor_reaches_its_descendants() {
    let mut ui = Ui::new();
    let root = ui.root();
    let card = ui.add_box(root, style! { width: 120px; display: flex; flex-direction: column; });
    let p = ui.add_box(card, style! {});
    ui.add_text(p, LONG);
    ui.layout(400.0, 400.0, &mut FixedWidthText(7.0));
    assert_eq!(lines(&ui, p), 1, "nowrap by default");

    let mut s = ui.style(card).clone();
    s.width = Length::Px(120.0);
    threers_ui::css::apply(&mut s, "white-space", "normal");
    ui.set_style(card, s);
    ui.layout(400.0, 400.0, &mut FixedWidthText(7.0));
    assert!(lines(&ui, p) > 1, "the change came down to the grandchild's text");
}
