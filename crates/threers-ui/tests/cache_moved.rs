//! A [`Ui::cache_layout`] boundary moved by whole pixels — a list above it
//! grew a row — is put back from its cache, moved, not laid out again; and
//! lands where laying it out would put it.

use threers_ui::{style, FixedWidthText, NodeId, Rect, Ui};

/// Rows above a cached section of wrapped text; with `cache`, the section a
/// boundary.
fn page(rows: usize, cache: bool) -> (Ui, NodeId) {
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(root, style! { display: flex; flex-direction: column; });
    for _ in 0..rows {
        ui.add_box(list, style! { height: 20px; });
    }
    let section = ui.add_box(root, style! { display: flex; flex-direction: column; padding: 8px; gap: 4px; });
    for k in 0..30 {
        let b = ui.add_box(section, style! { width: 100%; });
        ui.add_text(b, format!("line {k} of a section long enough to wrap in the width it has"));
    }
    ui.cache_layout(section, cache);
    (ui, list)
}

fn layout(ui: &mut Ui) {
    ui.layout(300.0, 2000.0, &mut FixedWidthText(7.0));
}

fn rects(ui: &Ui) -> Vec<Rect> {
    let mut out = vec![];
    let mut stack = vec![ui.root()];
    while let Some(n) = stack.pop() {
        out.push(ui.rect(n));
        stack.extend(ui.children(n).iter().rev());
    }
    out
}

#[test]
fn a_cached_section_moved_down_is_put_back_moved() {
    let (mut ui, list) = page(3, true);
    layout(&mut ui);
    let all = ui.laid_out_last();
    // a row more above it: the section moves down 20px
    ui.add_box(list, style! { height: 20px; });
    layout(&mut ui);
    assert!(ui.laid_out_last() < all / 4, "laid out {} of {all} again", ui.laid_out_last());
    let (mut fresh, _) = page(4, false);
    layout(&mut fresh);
    assert_eq!(rects(&ui), rects(&fresh));
    // and half a pixel at twice the density: a whole device pixel
    let (mut ui, list) = page(3, true);
    ui.set_pixel_ratio(2.0);
    layout(&mut ui);
    ui.add_box(list, style! { height: 0.5px; });
    layout(&mut ui);
    assert!(ui.laid_out_last() < all / 4, "a device pixel's move laid out {}", ui.laid_out_last());
}
