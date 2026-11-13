//! Flexbox beyond one line: wrapping (and from the far side), reversed
//! directions, `order`, every `justify-content` and `align-content`, and
//! separate row and column gaps.

use threers_ui::{style, FixedWidthText, NodeId, Rect, Ui};

fn layout(ui: &mut Ui) {
    ui.layout(400.0, 400.0, &mut FixedWidthText(7.0));
}

fn boxes(ui: &mut Ui, parent: NodeId, n: usize, w: f32, h: f32) -> Vec<NodeId> {
    (0..n)
        .map(|_| {
            let mut s = style! { flex-shrink: 0; };
            s.width = threers_ui::Length::Px(w);
            s.height = threers_ui::Length::Px(h);
            ui.add_box(parent, s)
        })
        .collect()
}

fn at(ui: &Ui, n: NodeId) -> (f32, f32) {
    let r: Rect = ui.rect(n);
    (r.x, r.y)
}

#[test]
fn items_wrap_onto_lines_and_the_box_grows_to_hold_them() {
    let mut ui = Ui::new();
    let root = ui.root();
    let row = ui.add_box(root, style! { display: flex; flex-wrap: wrap; width: 250px; gap: 10px; });
    let b = boxes(&mut ui, row, 5, 70.0, 30.0);
    layout(&mut ui);
    // 70 + 10 + 70 + 10 + 70 = 230 fits 250; the fourth goes down.
    assert_eq!([at(&ui, b[0]), at(&ui, b[2]), at(&ui, b[3]), at(&ui, b[4])], [(0.0, 0.0), (160.0, 0.0), (0.0, 40.0), (80.0, 40.0)]);
    assert_eq!(ui.rect(row).height, 70.0, "two lines and the gap between");
    // From the far side: the first line at the bottom.
    ui.update_style(row, |s| threers_ui::css::apply(s, "flex-wrap", "wrap-reverse"));
    layout(&mut ui);
    assert_eq!((at(&ui, b[0]), at(&ui, b[3])), ((0.0, 40.0), (0.0, 0.0)));
}

#[test]
fn reversed_directions_lay_items_from_the_end() {
    let mut ui = Ui::new();
    let root = ui.root();
    let row = ui.add_box(root, style! { display: flex; flex-direction: row-reverse; width: 300px; height: 40px; gap: 10px; });
    let b = boxes(&mut ui, row, 3, 50.0, 20.0);
    let col = ui.add_box(root, style! { display: flex; flex-direction: column-reverse; width: 50px; height: 200px; });
    let c = boxes(&mut ui, col, 2, 50.0, 30.0);
    layout(&mut ui);
    assert_eq!(b.iter().map(|&n| ui.rect(n).x).collect::<Vec<_>>(), [250.0, 190.0, 130.0]);
    let top = ui.rect(col).y;
    assert_eq!(c.iter().map(|&n| ui.rect(n).y - top).collect::<Vec<_>>(), [170.0, 140.0]);
}

#[test]
fn order_places_items_before_their_turn_in_the_tree() {
    let mut ui = Ui::new();
    let root = ui.root();
    let row = ui.add_box(root, style! { display: flex; width: 300px; });
    let b = boxes(&mut ui, row, 3, 50.0, 20.0);
    ui.update_style(b[2], |s| s.order = -1);
    ui.update_style(b[0], |s| s.order = 1);
    layout(&mut ui);
    assert_eq!(b.iter().map(|&n| ui.rect(n).x).collect::<Vec<_>>(), [100.0, 50.0, 0.0]);
}

#[test]
fn every_justify_content() {
    let cases = [("space-around", [25.0, 125.0, 225.0]), ("space-evenly", [37.5, 125.0, 212.5]), ("space-between", [0.0, 125.0, 250.0]), ("center", [75.0, 125.0, 175.0])];
    for (j, want) in cases {
        let mut ui = Ui::new();
        let root = ui.root();
        let row = ui.add_box(root, style! { display: flex; width: 300px; });
        ui.update_style(row, |s| threers_ui::css::apply(s, "justify-content", j));
        let b = boxes(&mut ui, row, 3, 50.0, 20.0);
        layout(&mut ui);
        assert_eq!(b.iter().map(|&n| ui.rect(n).x).collect::<Vec<_>>(), want, "{j}");
    }
}

#[test]
fn every_align_content() {
    // Two lines of 30 in 200: 140 to share.
    let cases = [
        ("flex-start", [0.0, 30.0], 30.0),
        ("center", [70.0, 100.0], 30.0),
        ("flex-end", [140.0, 170.0], 30.0),
        ("space-between", [0.0, 170.0], 30.0),
        ("space-around", [35.0, 135.0], 30.0),
        ("space-evenly", [46.666668, 123.333336], 30.0),
        ("stretch", [0.0, 100.0], 100.0),
    ];
    for (a, want, h) in cases {
        let mut ui = Ui::new();
        let root = ui.root();
        let row = ui.add_box(root, style! { display: flex; flex-wrap: wrap; width: 100px; height: 200px; });
        ui.update_style(row, |s| threers_ui::css::apply(s, "align-content", a));
        let b: Vec<NodeId> = (0..2)
            .map(|_| ui.add_box(row, style! { width: 80px; min-height: 30px; }))
            .collect();
        layout(&mut ui);
        let top = ui.rect(row).y;
        let got: Vec<f32> = b.iter().map(|&n| ui.rect(n).y - top).collect();
        assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-3), "{a}: {got:?}");
        assert_eq!(ui.rect(b[0]).height, h, "{a}: an item stretched to its line");
    }
}

#[test]
fn rows_and_columns_have_their_own_gaps() {
    let mut ui = Ui::new();
    let root = ui.root();
    let grid = ui.add_box(root, style! { display: flex; flex-flow: row wrap; width: 120px; gap: 20px 5px; });
    let b = boxes(&mut ui, grid, 4, 50.0, 10.0);
    layout(&mut ui);
    // 5 between columns, 20 between rows.
    assert_eq!([at(&ui, b[1]), at(&ui, b[2])], [(55.0, 0.0), (0.0, 30.0)]);
}

/// `aspect-ratio`: a box with no height of its own is as tall as its width
/// over the ratio, and follows its width; a height of its own wins.
#[test]
fn aspect_ratio_gives_the_height_from_the_width() {
    use threers_ui::{style, FixedWidthText, Ui};
    let mut ui = Ui::new();
    let root = ui.root();
    let col = ui.add_box(root, style! { display: flex; flex-direction: column; width: 320px; });
    let video = ui.add_box(col, style! { aspect-ratio: 16 / 9; });
    let below = ui.add_box(col, style! { height: 10px; });
    ui.layout(800.0, 600.0, &mut FixedWidthText(7.0));
    assert_eq!(ui.rect(video).height, 180.0);
    assert_eq!(ui.rect(below).y, 180.0, "what follows moves down");
    ui.update_style(col, |s| s.width = threers_ui::Length::Px(160.0));
    ui.layout(800.0, 600.0, &mut FixedWidthText(7.0));
    assert_eq!(ui.rect(video).height, 90.0);
    ui.update_style(video, |s| s.height = threers_ui::Length::Px(40.0));
    ui.layout(800.0, 600.0, &mut FixedWidthText(7.0));
    assert_eq!(ui.rect(video).height, 40.0);
}
