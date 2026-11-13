//! A row that wraps, inside a column, is as tall as its lines: measured at
//! its own width (`width`, `max-width`), not the column's.

use threers_ui::{style, FixedWidthText, Ui};

#[test]
fn a_wrapping_row_narrower_than_its_column_is_as_tall_as_its_lines() {
    let mut ui = Ui::new();
    let column = ui.add_box(ui.root(), style! { display: flex; flex-direction: column; align-items: center; width: 800px; });
    let row = ui.add_box(column, style! { display: flex; flex-wrap: wrap; width: 300px; max-width: 100%; });
    for _ in 0..4 {
        ui.add_box(row, style! { width: 100px; height: 20px; flex-shrink: 0; });
    }
    let after = ui.add_box(column, style! { width: 50px; height: 10px; });
    ui.layout(800.0, 600.0, &mut FixedWidthText(8.0));
    // Three to a line: two lines.
    assert_eq!(ui.rect(row).height, 40.0);
    assert_eq!(ui.rect(after).y, ui.rect(row).y + 40.0);
}

/// Words as boxes in a wrapping row with no width of its own, centred in
/// a column: the row is as tall as the lines they make.
#[test]
fn a_row_of_word_boxes_that_wraps_is_as_tall_as_its_lines() {
    let mut ui = Ui::new();
    let column = ui.add_box(ui.root(), style! { display: flex; flex-direction: column; align-items: center; width: 200px; });
    let row = ui.add_box(column, style! { display: flex; flex-wrap: wrap; justify-content: center; column-gap: 8px; max-width: 100%; });
    for word in ["Tu", "écoutes", "___", "la", "radio."] {
        let w = ui.add_box(row, style! { font-size: 20px; white-space: nowrap; });
        ui.add_text(w, word);
    }
    let after = ui.add_box(column, style! { width: 50px; height: 10px; });
    ui.layout(400.0, 600.0, &mut FixedWidthText(10.0));
    let r = ui.rect(row);
    let last = ui.children(row)[4];
    assert!(ui.rect(last).y + ui.rect(last).height <= r.y + r.height + 0.01, "{:?} in {:?}", ui.rect(last), r);
    assert!(ui.rect(after).y >= r.y + r.height - 0.01);
}

/// A row sized to its content counts `column-gap` between its items, so
/// they fit on the line it is sized for.
#[test]
fn a_rows_own_width_counts_its_column_gap() {
    let mut ui = Ui::new();
    let column = ui.add_box(ui.root(), style! { display: flex; flex-direction: column; align-items: center; width: 400px; });
    let row = ui.add_box(column, style! { display: flex; flex-wrap: wrap; column-gap: 7px; });
    let mut words = vec![];
    for word in ["___", "coup"] {
        let w = ui.add_box(row, style! { white-space: nowrap; });
        ui.add_text(w, word);
        words.push(w);
    }
    ui.layout(400.0, 600.0, &mut FixedWidthText(10.0));
    assert_eq!(ui.rect(row).width, 30.0 + 7.0 + 40.0);
    assert_eq!(ui.rect(words[0]).y, ui.rect(words[1]).y, "one line");
}

/// A card narrower than its column holds what wraps inside it: measured at
/// its own width, not the column's, it is as tall as its content there.
#[test]
fn a_narrow_card_in_a_wide_column_holds_its_wrapped_content() {
    let mut ui = Ui::new();
    let page = ui.add_box(ui.root(), style! { display: flex; flex-direction: column; align-items: center; width: 1000px; });
    let card = ui.add_box(page, style! { display: flex; flex-direction: column; align-items: center; width: 300px; padding: 10px; box-sizing: border-box; });
    let row = ui.add_box(card, style! { display: flex; flex-wrap: wrap; gap: 10px; max-width: 100%; });
    for _ in 0..2 {
        ui.add_box(row, style! { width: 200px; height: 50px; flex-shrink: 0; });
    }
    let after = ui.add_box(page, style! { width: 10px; height: 10px; });
    ui.layout(1000.0, 800.0, &mut FixedWidthText(8.0));
    // Two lines of 50 with a gap of 10, and the card's padding.
    assert_eq!(ui.rect(row).height, 110.0);
    assert_eq!(ui.rect(card).height, 130.0);
    assert_eq!(ui.rect(after).y, ui.rect(card).y + 130.0);
}
