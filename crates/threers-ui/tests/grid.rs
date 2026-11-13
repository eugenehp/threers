//! CSS Grid, against what a browser lays out for the same CSS (worked by
//! hand from the spec's track sizing).

use threers_ui::{style, FixedWidthText, NodeId, Rect, Ui};

fn boxes(ui: &mut Ui, parent: NodeId, n: usize, css: threers_ui::Style) -> Vec<NodeId> {
    (0..n).map(|_| ui.add_box(parent, css.clone())).collect()
}

fn r(ui: &Ui, n: NodeId) -> (f32, f32, f32, f32) {
    let Rect { x, y, width, height } = ui.rect(n);
    ((x * 100.0).round() / 100.0, (y * 100.0).round() / 100.0, (width * 100.0).round() / 100.0, (height * 100.0).round() / 100.0)
}

fn lay(ui: &mut Ui) {
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
}

#[test]
fn fixed_and_fractional_columns_share_what_is_left() {
    let mut ui = Ui::new();
    let root = ui.root();
    let g = ui.add_box(root, style! { display: grid; grid-template-columns: 100px 1fr 2fr; gap: 10px; });
    let items = boxes(&mut ui, g, 5, style! { height: 20px; });
    lay(&mut ui);
    // 400 − 100 − 2×10 = 280 shared 1:2.
    assert_eq!(r(&ui, items[0]), (0.0, 0.0, 100.0, 20.0));
    assert_eq!(r(&ui, items[1]), (110.0, 0.0, 93.33, 20.0));
    assert_eq!(r(&ui, items[2]), (213.33, 0.0, 186.67, 20.0));
    // Auto-placed onto the next row, after the row gap.
    assert_eq!(r(&ui, items[3]), (0.0, 30.0, 100.0, 20.0));
    assert_eq!(ui.rect(g).height, 50.0);
}

#[test]
fn spans_and_explicit_lines_place_items_and_the_rest_flow_round_them() {
    let mut ui = Ui::new();
    let root = ui.root();
    let g = ui.add_box(root, style! { display: grid; grid-template-columns: repeat(3, 1fr); width: 300px; });
    let header = ui.add_box(g, style! { grid-column: 1 / -1; height: 10px; });
    let side = ui.add_box(g, style! { grid-column: 3; grid-row: 2 / span 2; });
    let rest = boxes(&mut ui, g, 4, style! { height: 20px; });
    let wide = ui.add_box(g, style! { grid-column: span 2; height: 5px; });
    lay(&mut ui);
    assert_eq!(r(&ui, header), (0.0, 0.0, 300.0, 10.0));
    // The side column takes rows 2 and 3 of column 3; the four flow into
    // columns 1 and 2 of those rows; the wide one wraps to row 4.
    assert_eq!(r(&ui, side), (200.0, 10.0, 100.0, 40.0), "stretched over its two rows");
    assert_eq!(r(&ui, rest[0]), (0.0, 10.0, 100.0, 20.0));
    assert_eq!(r(&ui, rest[1]), (100.0, 10.0, 100.0, 20.0));
    assert_eq!(r(&ui, rest[2]), (0.0, 30.0, 100.0, 20.0));
    assert_eq!(r(&ui, rest[3]), (100.0, 30.0, 100.0, 20.0));
    assert_eq!(r(&ui, wide), (0.0, 50.0, 200.0, 5.0));
}

#[test]
fn auto_tracks_fit_content_and_share_free_space_without_fr() {
    let mut ui = Ui::new();
    let root = ui.root();
    let g = ui.add_box(root, style! { display: grid; grid-template-columns: auto auto; });
    let a = ui.add_box(g, style! { width: 50px; height: 10px; });
    let b = ui.add_box(g, style! { width: 30px; height: 10px; });
    lay(&mut ui);
    // 400 − 80 = 320 free, 160 to each: the items keep their widths, at
    // the starts of 210- and 190-wide columns.
    assert_eq!(r(&ui, a), (0.0, 0.0, 50.0, 10.0));
    assert_eq!(r(&ui, b), (210.0, 0.0, 30.0, 10.0));
}

#[test]
fn minmax_keeps_a_minimum_and_rows_flex_in_a_definite_height() {
    let mut ui = Ui::new();
    let root = ui.root();
    let g = ui.add_box(root, style! {
        display: grid; width: 150px; height: 100px;
        grid-template-columns: minmax(100px, 1fr) minmax(100px, 1fr);
        grid-template-rows: 20px 1fr 3fr;
    });
    let items = boxes(&mut ui, g, 6, style! {});
    lay(&mut ui);
    // Each column wants at least 100: the grid overflows its 150.
    assert_eq!(r(&ui, items[1]).0, 100.0);
    // Rows: 20, then 80 shared 1:3.
    assert_eq!(r(&ui, items[2]), (0.0, 20.0, 100.0, 20.0));
    assert_eq!(r(&ui, items[4]), (0.0, 40.0, 100.0, 60.0));
}

#[test]
fn a_grid_without_a_width_is_as_wide_as_its_tracks() {
    let mut ui = Ui::new();
    let root = ui.root();
    let row = ui.add_box(root, style! { display: flex; });
    let g = ui.add_box(row, style! { display: grid; grid-template-columns: 40px auto; gap: 5px; });
    ui.add_box(g, style! { height: 10px; });
    ui.add_box(g, style! { width: 70px; height: 10px; });
    let after = ui.add_box(row, style! { width: 10px; height: 10px; });
    lay(&mut ui);
    assert_eq!(ui.rect(g).width, 115.0);
    assert_eq!(ui.rect(after).x, 115.0);
}

#[test]
fn grid_css_reads_as_written() {
    use threers_ui::{Track, TrackSize};
    let s = style! { grid-template-columns: 200px repeat(2, minmax(0, 1fr)) auto 25%; grid-row: 2 / span 3; row-gap: 4px; };
    assert_eq!(s.grid_columns.len(), 5);
    assert_eq!(s.grid_columns[1], Track { min: TrackSize::Px(0.0), max: TrackSize::Fr(1.0) });
    assert_eq!(s.grid_columns[4], Track { min: TrackSize::Percent(0.25), max: TrackSize::Percent(0.25) });
    assert_eq!((s.grid_row.start, s.grid_row.span), (Some(2), 3));
    assert_eq!(s.row_gap, Some(4.0));
}

/// A grid given its height by where it is placed (the root, a stretched
/// item) is definite there: its `fr` rows fill it.
#[test]
fn fr_rows_fill_a_placed_grid_without_a_set_height() {
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { display: grid; grid-template-rows: 50px 1fr; });
    let top = ui.add_box(root, style! {});
    let rest = ui.add_box(root, style! {});
    lay(&mut ui);
    assert_eq!(r(&ui, top).3, 50.0);
    assert_eq!(r(&ui, rest), (0.0, 50.0, 400.0, 250.0));
}

/// A flex item whose text wraps may shrink to its longest word (`min-width:
/// auto` is the min-content width) and stops at its `max-width`, as a chat
/// bubble does; text that does not wrap keeps its width.
#[test]
fn a_wrapping_flex_item_shrinks_to_its_row_and_its_max_width() {
    let text = "Sure the layout is done can you check scrolling on your phone it glides now.";
    let bubble = |max: Option<&str>| {
        let mut ui = Ui::new();
        let root = ui.root();
        let row = ui.add_box(root, style! { display: flex; width: 284px; });
        let b = ui.add_box(row, style! { padding: 0 10px; white-space: normal; });
        if let Some(m) = max {
            threers_ui::css::apply(ui.style_mut(b), "max-width", m);
        }
        ui.add_text(b, text);
        lay(&mut ui);
        r(&ui, b)
    };
    // 532 px on one line: shrunk to the row, and wrapped.
    let (_, _, w, h) = bubble(None);
    assert_eq!(w, 284.0);
    assert!(h > 19.2 * 2.0);
    // A content-box max-width of 78 %, plus the padding.
    assert_eq!(bubble(Some("78%")).2, 241.52);
    assert_eq!(bubble(Some("200px")).2, 220.0);
    // Without wrapping, the run is its own minimum: it overflows the row.
    let mut ui = Ui::new();
    let root = ui.root();
    let row = ui.add_box(root, style! { display: flex; width: 284px; });
    let b = ui.add_box(row, style! { padding: 0 10px; });
    ui.add_text(b, text);
    lay(&mut ui);
    assert_eq!(r(&ui, b).2, 552.0);
}

/// A grid item that clips or scrolls has no automatic minimum size (CSS):
/// a flexible row in a grid of known height takes what is left of it, not
/// the item's content height, and the item scrolls in it.
#[test]
fn a_scrolling_item_does_not_stretch_a_flexible_row() {
    use threers_ui::{style, FixedWidthText, Ui};
    let mut ui = Ui::new();
    let root = ui.root();
    let grid = ui.add_box(root, style! { display: grid; grid-template-columns: 1fr; grid-template-rows: 20px 1fr; height: 100%; });
    ui.add_box(grid, style! {});
    let panel = ui.add_box(grid, style! { overflow: auto; });
    ui.add_box(panel, style! { height: 1000px; });
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    assert_eq!(ui.rect(panel).height, 280.0);
    assert_eq!(ui.scroll_extent(panel).0, 1000.0, "it scrolls its content");
}
