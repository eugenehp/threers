//! Right to left, left to right, and the one inside the other.

use threers_ui::environment::LayoutDirection::{LeftToRight, RightToLeft};
use threers_ui::{style, FixedWidthText, NodeId, Rect, Style, Ui};

fn ui() -> Ui {
    let mut ui = Ui::new();
    ui.set_layout_direction(RightToLeft);
    let root = ui.root();
    ui.set_style(root, style! { display: flex; flex-direction: column; });
    ui
}

fn lay(ui: &mut Ui) {
    ui.layout(400.0, 300.0, &mut FixedWidthText(10.0));
}

fn boxed(ui: &mut Ui, parent: NodeId, css: &str) -> NodeId {
    let b = ui.add_box(parent, Style::default());
    ui.apply(b, &threers_ui::patch::Patch::css(css));
    b
}

const ROW: &str =
    "display: flex; flex-direction: row; width: 200px; height: 20px; padding-left: 10px";

#[test]
fn a_part_can_read_the_other_way_whichever_is_around_it() {
    for (around, inside) in [
        (RightToLeft, "direction: ltr"),
        (LeftToRight, "direction: rtl"),
    ] {
        let mut ui = ui();
        ui.set_layout_direction(around);
        let root = ui.root();
        let outer = boxed(&mut ui, root, ROW);
        let o = boxed(&mut ui, outer, "width: 30px; height: 10px");
        let part = boxed(&mut ui, root, &format!("{ROW}; {inside}"));
        let a = boxed(&mut ui, part, "width: 30px; height: 10px");
        let b = boxed(&mut ui, part, "width: 30px; height: 10px");
        lay(&mut ui);
        let start = |ui: &Ui, p: NodeId, c: NodeId, rtl: bool| {
            let (p, c) = (ui.rect(p), ui.rect(c));
            if rtl {
                p.x + p.width - c.x - c.width
            } else {
                c.x - p.x
            }
        };
        let rtl = around == RightToLeft;
        assert_eq!(start(&ui, outer, o, rtl), 10.0);
        assert_eq!(start(&ui, part, a, !rtl), 10.0);
        assert_eq!(start(&ui, part, b, !rtl), 40.0);
        assert_eq!(ui.direction_of(a) == RightToLeft, !rtl);
        // Both boxes start where the outside does.
        assert_eq!(ui.rect(part).x, if rtl { 190.0 } else { 0.0 });
    }
}

#[test]
fn what_is_placed_by_the_screen_stays_there() {
    let mut ui = ui();
    let root = ui.root();
    let b = boxed(
        &mut ui,
        root,
        "position: absolute; left: 30px; top: 0; width: 10px; height: 10px",
    );
    lay(&mut ui);
    assert_eq!(ui.rect(b).x, 360.0);
    ui.update_style(b, |s| s.unmirrored = true);
    lay(&mut ui);
    assert_eq!(ui.rect(b).x, 30.0);
}

#[test]
fn logical_sides_in_css_are_the_styles() {
    let mut s = Style::default();
    for (n, v) in [
        ("margin-inline-start", "3px"),
        ("padding-inline", "4px 5px"),
        ("inset-inline-end", "6px"),
        ("direction", "rtl"),
    ] {
        threers_ui::css::apply(&mut s, n, v);
    }
    assert_eq!(
        (s.margin.left, s.padding.left, s.padding.right),
        (3.0, 4.0, 5.0)
    );
    assert_eq!(s.right, threers_ui::Length::Px(6.0));
    assert_eq!(s.direction, Some(RightToLeft));
}

/// The caret before each letter of Hebrew is at its right; before Latin in
/// it, at its left; and a press finds the letter it is on.
#[test]
fn carets_and_presses_follow_the_way_text_reads() {
    let mut ui = ui();
    let root = ui.root();
    let t = ui.add_text(root, "אב cd ה");
    lay(&mut ui);
    let m = &mut FixedWidthText(10.0);
    // Drawn, right to left from 400: א ב _ [cd] _ ה
    let x = |ui: &Ui, i| ui.text_caret(t, i, &mut FixedWidthText(10.0)).x;
    assert_eq!(
        (0..=7).map(|i| x(&ui, i)).collect::<Vec<_>>(),
        [400.0, 390.0, 380.0, 350.0, 360.0, 350.0, 340.0, 330.0]
    );
    assert_eq!(ui.text_index_at(t, 398.0, 5.0, m), 0);
    assert_eq!(ui.text_index_at(t, 389.0, 5.0, m), 1);
    assert_eq!(ui.text_index_at(t, 361.0, 5.0, m), 4);
    assert_eq!(ui.text_index_at(t, 331.0, 5.0, m), 7);
    // Selected from in the Hebrew into the Latin: two stretches.
    let r = ui.text_selection_rects(t, 1, 4, m);
    assert_eq!(
        r.iter().map(|r| (r.x, r.width)).collect::<Vec<_>>(),
        [(350.0, 10.0), (370.0, 20.0)]
    );
    // And English in the same layout as ever, at the right.
    let e = ui.add_text(root, "abc");
    lay(&mut ui);
    assert_eq!(ui.text_caret(e, 0, m).x, 370.0);
    assert_eq!(ui.text_caret(e, 3, m).x, 400.0);
}

#[derive(Default)]
struct Drawn {
    texts: Vec<(f32, String)>,
    rects: Vec<Rect>,
}

impl threers_ui::TextMeasure for Drawn {
    fn measure(&mut self, t: &str, s: &Style) -> (f32, f32) {
        FixedWidthText(10.0).measure(t, s)
    }
}

impl threers_ui::Painter for Drawn {
    fn path(
        &mut self,
        _: Rect,
        _: &[String],
        _: [f32; 4],
        _: Option<f32>,
        _: bool,
        _: threers_ui::Rgba,
    ) {
    }

    fn fill_rect(&mut self, r: Rect, _: threers_ui::Rgba) {
        self.rects.push(r);
    }
    fn fill_rounded_rect(&mut self, r: Rect, _: f32, _: f32, _: threers_ui::Rgba) {
        self.rects.push(r);
    }
    fn text(&mut self, r: Rect, t: &str, _: &threers_ui::TextStyle) {
        self.texts.push((r.x, t.to_owned()));
    }
}

/// A wrapped line that begins with Latin still reads as its paragraph; a
/// styled stretch is where the line's order puts it.
#[test]
fn lines_and_styled_stretches_read_as_their_paragraph() {
    let mut ui = ui();
    let root = ui.root();
    let b = boxed(&mut ui, root, "width: 80px; white-space: normal");
    ui.add_text(b, "אבגד הו Windows אב");
    let c = boxed(&mut ui, root, "width: 400px");
    let t = ui.add_text(c, "אב גד");
    ui.set_text_runs(
        t,
        vec![
            threers_ui::rich_text::TextRun::new(4),
            threers_ui::rich_text::TextRun::new(5).color([1.0, 0.0, 0.0, 1.0]),
        ],
    );
    lay(&mut ui);
    let mut d = Drawn::default();
    ui.paint(&mut d);
    assert!(
        d.texts
            .iter()
            .any(|(_, t)| t.starts_with("\u{200f}Windows")),
        "{:?}",
        d.texts
    );
    let at = |s: &str| {
        d.texts
            .iter()
            .find(|(_, t)| t.trim() == s)
            .unwrap_or_else(|| panic!("{s} in {:?}", d.texts))
            .0
    };
    assert!(at("אב") > at("גד"), "{:?}", d.texts);
    assert_eq!(at("גד"), 350.0);
}

#[test]
fn a_border_at_the_start_is_at_the_right() {
    let mut ui = ui();
    let root = ui.root();
    let b = boxed(
        &mut ui,
        root,
        "width: 100px; height: 20px; border-left-width: 4px; border-color: #ff0000",
    );
    lay(&mut ui);
    let mut d = Drawn::default();
    ui.paint(&mut d);
    let r = ui.rect(b);
    assert!(
        d.rects
            .iter()
            .any(|q| q.width == 4.0 && q.x == r.x + r.width - 4.0),
        "{:?}",
        d.rects
    );
}

#[test]
fn a_sideways_strip_scrolls_from_the_right() {
    let mut ui = ui();
    let root = ui.root();
    let strip = boxed(
        &mut ui,
        root,
        "display: flex; flex-direction: row; width: 100px; height: 20px; overflow-x: auto",
    );
    let items: Vec<_> = (0..4)
        .map(|_| boxed(&mut ui, strip, "width: 60px; height: 20px; flex-shrink: 0"))
        .collect();
    lay(&mut ui);
    assert_eq!(ui.rect(items[0]).x, 340.0);
    ui.scroll_into_view(items[3]);
    lay(&mut ui);
    assert_eq!(ui.rect(items[3]).x, 300.0);
    assert_eq!(ui.style(strip).scroll_left, 140.0);
}

#[test]
fn what_people_write_reads_as_it_is_written() {
    let mut ui = ui();
    let root = ui.root();
    let at = |ui: &mut Ui, text: &str| {
        let b = boxed(ui, root, "direction: auto; width: 200px; display: flex; flex-direction: row");
        let t = ui.add_text(b, text);
        lay(ui);
        let (b, t) = (ui.rect(b), ui.rect(t));
        (t.x == b.x, t.x + t.width == b.x + b.width)
    };
    assert_eq!(at(&mut ui, "12 abc"), (true, false));
    assert_eq!(at(&mut ui, "(אב) abc"), (false, true));
    // With no letters, as around it.
    assert_eq!(at(&mut ui, "12"), (false, true));
    ui.set_layout_direction(LeftToRight);
    assert_eq!(at(&mut ui, "12"), (true, false));
    assert_eq!(at(&mut ui, "אב"), (false, true));
}

/// Towards the start, where the row opens, is the right.
#[test]
fn a_row_is_swiped_towards_its_start() {
    use threers_ui::input::Input;
    let mut ui = ui();
    let root = ui.root();
    let list = boxed(&mut ui, root, "width: 400px; display: flex; flex-direction: column");
    let row = boxed(&mut ui, list, "height: 40px");
    boxed(&mut ui, row, "height: 40px");
    lay(&mut ui);
    let mut input: Input<()> = Input::new();
    input.on_swipe_delete(list, |_, _, _| {});
    for (dx, opens) in [(100.0, true), (-100.0, false)] {
        use threers_ui::input::TouchPhase::*;
        input.close_swipe();
        for (phase, x) in [(Start, 200.0), (Move, 200.0 + dx * 0.3), (Move, 200.0 + dx)] {
            input.dispatch(&mut ui, &mut (), threers_ui::input::Event::Touch { id: 1, phase, x: x, y: 20.0, time: 0.0 });
        }
        assert_eq!(input.swipe(list).is_some_and(|s| s.offset < -50.0), opens, "{dx}");
        input.dispatch(&mut ui, &mut (), threers_ui::input::Event::Touch { id: 1, phase: Cancel, x: 200.0, y: 20.0, time: 0.1 });
    }
}

/// The shape itself is drawn the other way round, on any painter.
#[test]
fn an_arrow_points_the_way_lines_read() {
    #[derive(Default)]
    struct Paths(Vec<String>);
    impl threers_ui::TextMeasure for Paths {
        fn measure(&mut self, t: &str, s: &Style) -> (f32, f32) {
            FixedWidthText(10.0).measure(t, s)
        }
    }
    impl threers_ui::Painter for Paths {
        fn fill_rect(&mut self, _: Rect, _: threers_ui::Rgba) {}
        fn fill_rounded_rect(&mut self, _: Rect, _: f32, _: f32, _: threers_ui::Rgba) {}
        fn text(&mut self, _: Rect, _: &str, _: &threers_ui::TextStyle) {}
        fn path(&mut self, _: Rect, d: &[String], _: [f32; 4], _: Option<f32>, _: bool, _: threers_ui::Rgba) {
            self.0.extend_from_slice(d);
        }
    }
    let drawn = |direction, css: &str| {
        let mut ui = ui();
        ui.set_layout_direction(direction);
        let root = ui.root();
        let b = boxed(&mut ui, root, css);
        threers_ui::svg::add_icon(&mut ui, b, style! { width: 24px; height: 24px; }, r#"<path d="M9 18l6-6-6-6"/>"#, 2.0);
        lay(&mut ui);
        let mut p = Paths::default();
        ui.paint(&mut p);
        p.0.concat().replace([',', 'M', 'L'], " ")
    };
    let first = |s: &str| s.split_whitespace().next().unwrap().parse::<f32>().unwrap();
    assert_eq!(first(&drawn(LeftToRight, "rtl-flip: on")), 9.0);
    assert_eq!(first(&drawn(RightToLeft, "width: auto")), 9.0);
    assert_eq!(first(&drawn(RightToLeft, "rtl-flip: on")), 15.0);
}

/// A gradient from the start runs from the right.
#[test]
fn a_gradient_starts_where_lines_do() {
    let mut ui = ui();
    let root = ui.root();
    for (css, at) in [("width: 100px", 0.0), ("width: 100px; direction: ltr", 1.0)] {
        let b = boxed(&mut ui, root, &format!("{css}; height: 20px; background: linear-gradient(90deg, #ff0000, #0000ff)"));
        lay(&mut ui);
        #[derive(Default)]
        struct Runs(Vec<(f32, threers_ui::Rgba)>);
        impl threers_ui::TextMeasure for Runs {
            fn measure(&mut self, t: &str, s: &Style) -> (f32, f32) {
                FixedWidthText(10.0).measure(t, s)
            }
        }
        impl threers_ui::Painter for Runs {
            fn fill_rect(&mut self, r: Rect, c: threers_ui::Rgba) {
                self.0.push((r.x, c));
            }
            fn fill_rounded_rect(&mut self, r: Rect, _: f32, _: f32, c: threers_ui::Rgba) {
                self.0.push((r.x, c));
            }
            fn text(&mut self, _: Rect, _: &str, _: &threers_ui::TextStyle) {}
            fn path(&mut self, _: Rect, _: &[String], _: [f32; 4], _: Option<f32>, _: bool, _: threers_ui::Rgba) {}
        }
        let mut p = Runs::default();
        ui.paint(&mut p);
        let r = ui.rect(b);
        let left = p.0.iter().filter(|(x, _)| *x >= r.x && *x <= r.x + 100.0).min_by(|a, b| a.0.total_cmp(&b.0));
        let (_, c) = left.expect("drawn");
        assert!((c[0] - at).abs() < 0.2, "{css}: {c:?}");
        ui.remove(b);
    }
}
