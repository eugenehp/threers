//! `align-items: baseline`: in a row, the first lines of text of items of
//! different sizes sit on one line; in a column it is `flex-start`.

use threers_ui::{style, FixedWidthText, Ui};

/// A row of a big and a small word (each in its own box, the small one
/// padded too), aligned `align`: the two boxes and their text nodes.
fn row(align: &str) -> (Ui, [threers_ui::NodeId; 4]) {
    let mut ui = Ui::new();
    let root = ui.root();
    let row = ui.add_box(root, style! { display: flex; flex-direction: row; });
    threers_ui::css::apply(ui.style_mut(row), "align-items", align);
    let big = ui.add_box(row, style! { font-size: 32px; });
    let big_text = ui.add_text(big, "Grand");
    let small = ui.add_box(row, style! { font-size: 14px; padding: 6px; });
    let small_text = ui.add_text(small, "petit");
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    (ui, [big, big_text, small, small_text])
}

/// Where a text node's baseline is, on the page.
fn baseline(ui: &Ui, text: threers_ui::NodeId) -> f32 {
    let s = ui.style(text);
    let lh = s.line_height.unwrap_or(s.font_size * 1.2);
    ui.rect(text).y + (lh - s.font_size) / 2.0 + s.font_size * 0.8
}

#[test]
fn baselines_line_up_in_a_row() {
    let (ui, [big, big_text, small, small_text]) = row("baseline");
    let (b, s) = (baseline(&ui, big_text), baseline(&ui, small_text));
    assert!((b - s).abs() < 0.5, "baselines {b} and {s}");
    // The small box is moved down, the big one stays at the top.
    assert_eq!(ui.rect(big).y, 0.0);
    assert!(ui.rect(small).y > 0.0);
    // Not so at flex-start: the tops line up, the baselines do not.
    let (ui, [_, big_text, _, small_text]) = row("flex-start");
    assert!((baseline(&ui, big_text) - baseline(&ui, small_text)).abs() > 2.0);
}

#[test]
fn baseline_in_a_column_is_flex_start() {
    let mut ui = Ui::new();
    let root = ui.root();
    let col = ui.add_box(
        root,
        style! { display: flex; flex-direction: column; width: 200px; },
    );
    threers_ui::css::apply(ui.style_mut(col), "align-items", "baseline");
    let a = ui.add_box(col, style! { width: 50px; height: 10px; });
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    assert_eq!(ui.rect(a).x, 0.0);
}
