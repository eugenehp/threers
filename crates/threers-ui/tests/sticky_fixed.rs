//! `position: sticky` and `position: fixed`, and a scroll laid out by moving
//! what a box holds rather than laying the tree out again.

use threers_ui::{style, FixedWidthText, NodeId, Ui};

fn layout(ui: &mut Ui) {
    ui.layout(400.0, 300.0, &mut FixedWidthText(10.0));
}

/// A 100px list of two sections, each a 20px sticky header over ten 20px
/// rows (220px a section).
fn sections() -> (Ui, NodeId, [NodeId; 2]) {
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(root, style! { height: 100px; overflow: auto; });
    let mut headers = vec![];
    for _ in 0..2 {
        let section = ui.add_box(list, style! {});
        headers.push(ui.add_box(section, style! { height: 20px; position: sticky; top: 0px; }));
        for _ in 0..10 {
            ui.add_box(section, style! { height: 20px; });
        }
    }
    layout(&mut ui);
    (ui, list, [headers[0], headers[1]])
}

#[test]
fn a_sticky_header_stays_while_its_section_shows() {
    let (mut ui, list, [a, b]) = sections();
    assert_eq!((ui.rect(a).y, ui.rect(b).y), (0.0, 220.0));
    ui.scroll_to(list, 50.0);
    layout(&mut ui);
    assert_eq!(ui.rect(a).y, 0.0, "stuck to the top of the view");
    assert_eq!(ui.rect(b).y, 170.0, "in the flow, below");
    // Section a ends at 220 - 210 = 10: its header goes up with its end,
    // and b's takes the top.
    ui.scroll_to(list, 210.0);
    layout(&mut ui);
    assert_eq!(ui.rect(a).y, -10.0);
    assert_eq!(ui.rect(b).y, 10.0);
    ui.scroll_to(list, 230.0);
    layout(&mut ui);
    assert_eq!(ui.rect(b).y, 0.0);
    // and back
    ui.scroll_to(list, 0.0);
    layout(&mut ui);
    assert_eq!((ui.rect(a).y, ui.rect(b).y), (0.0, 220.0));
}

/// Every box's rect, in tree order.
fn rects(ui: &Ui) -> Vec<threers_ui::Rect> {
    let mut out = vec![];
    let mut stack = vec![ui.root()];
    while let Some(n) = stack.pop() {
        out.push(ui.rect(n));
        stack.extend(ui.children(n).iter().rev());
    }
    out
}

#[test]
fn a_scroll_moves_the_boxes_without_laying_the_tree_out_again() {
    let (mut fast, list, _) = sections();
    let (mut slow, slow_list, _) = sections();
    let passes = fast.layout_passes();
    for top in [12.0, 37.5, 210.0, 400.0, 3.0] {
        fast.scroll_to(list, top);
        layout(&mut fast);
        // the same offset, laid out whole
        slow.style_mut(slow_list).scroll_top = top;
        layout(&mut slow);
        assert_eq!(fast.scroll_extent(list), slow.scroll_extent(slow_list), "at {top}");
        assert_eq!(rects(&fast), rects(&slow), "at {top}");
    }
    assert_eq!(fast.layout_passes(), passes, "no layout ran");
}

#[test]
fn a_fixed_box_stays_in_the_window_and_over_what_scrolls() {
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(root, style! { margin-top: 50px; height: 100px; overflow: auto; });
    for _ in 0..20 {
        ui.add_box(list, style! { height: 20px; });
    }
    let badge = ui.add_box(
        list,
        style! { position: fixed; top: 10px; right: 10px; width: 40px; height: 20px; },
    );
    layout(&mut ui);
    let at = ui.rect(badge);
    assert_eq!((at.x, at.y, at.width, at.height), (350.0, 10.0, 40.0, 20.0));
    assert_eq!(ui.scroll_extent(list).0, 400.0, "takes no room in the list");
    assert_eq!(ui.deferred_priority(badge), Some(0), "painted over the rest");
    assert_eq!(ui.hit_test(360.0, 15.0), Some(badge));
    ui.scroll_to(list, 120.0);
    layout(&mut ui);
    assert_eq!(ui.rect(badge), at, "the list scrolled, the badge did not");
    // no longer fixed: in the flow again, painted in its place
    ui.style_mut(badge).position = threers_ui::Position::Static;
    layout(&mut ui);
    assert_eq!(ui.deferred_priority(badge), None);
    assert_eq!(ui.scroll_extent(list).0, 420.0);
}
