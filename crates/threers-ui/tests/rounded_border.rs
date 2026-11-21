//! A rounded border is laid down once everywhere: a faded or see-through
//! box shows no border colour behind its background, and a rounded border
//! without a background has rounded corners.

use threers_ui::{ArcShape, Painter, TextStyle};
use threers_ui::{style, FixedWidthText, Rect, Rgba, Style, TextMeasure, Ui};

/// What was painted, in order.
#[derive(Default)]
struct Record {
    rects: Vec<(Rect, Rgba)>,
    rounded: Vec<(Rect, f32, Rgba)>,
    arcs: Vec<(ArcShape, f32, Rgba)>,
}

impl TextMeasure for Record {
    fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
        FixedWidthText(7.0).measure(text, style)
    }
}

impl Painter for Record {
    fn fill_rect(&mut self, r: Rect, c: Rgba) {
        self.rects.push((r, c));
    }
    fn fill_rounded_rect(&mut self, r: Rect, rx: f32, _ry: f32, c: Rgba) {
        self.rounded.push((r, rx, c));
    }
    fn stroke_arc(&mut self, arc: ArcShape, width: f32, c: Rgba) {
        self.arcs.push((arc, width, c));
    }
    fn text(&mut self, _: Rect, _: &str, _: &TextStyle) {}
    fn path(&mut self, _: Rect, _: &[String], _: [f32; 4], _: Option<f32>, _: bool, _: Rgba) {}
}

const BORDER: Rgba = [1.0, 0.0, 0.0, 1.0];
const BG: Rgba = [0.0, 0.0, 1.0, 1.0];

fn paint(style: Style) -> Record {
    let mut ui = Ui::new();
    let root = ui.root();
    ui.add_box(root, style);
    ui.layout(200.0, 100.0, &mut FixedWidthText(7.0));
    let mut rec = Record::default();
    ui.paint(&mut rec);
    rec
}

/// 100 × 40 inside a 1 px border (content-box sizing): 102 × 42 in all.
fn boxed(opacity: f32, background: Option<Rgba>) -> Style {
    let mut s = style! { width: 100px; height: 40px; border: 1px solid; border-radius: 8px; };
    s.border_color = BORDER;
    s.background = background;
    s.opacity = opacity;
    s
}

#[test]
fn an_opaque_box_keeps_the_fast_path() {
    let rec = paint(boxed(1.0, Some(BG)));
    assert!(rec.arcs.is_empty());
    assert_eq!(rec.rounded.len(), 2, "border colour over the box, background inset");
    assert_eq!(rec.rounded[0].2, BORDER);
}

#[test]
fn a_faded_box_lays_no_border_colour_under_its_background() {
    let rec = paint(boxed(0.45, Some(BG)));
    // The background, once, inset by the border, with the inner radius.
    assert_eq!(rec.rounded.len(), 1);
    let (inner, radius, c) = rec.rounded[0];
    assert_eq!((inner.x, inner.y, inner.width, inner.height, radius), (1.0, 1.0, 100.0, 40.0, 7.0));
    assert_eq!(c, [0.0, 0.0, 1.0, 0.45]);
    // The border as four corners and four sides, none of them under the background.
    assert_eq!(rec.arcs.len(), 4);
    for (arc, width, c) in &rec.arcs {
        assert_eq!((arc.radius, *width, *c), (7.5, 1.0, [1.0, 0.0, 0.0, 0.45]));
    }
    assert_eq!(rec.rects.len(), 4);
    let border_area: f32 = rec.rects.iter().map(|(r, _)| r.width * r.height).sum();
    assert_eq!(border_area, 2.0 * 85.0 + 2.0 * 25.0, "sides stop half a border short of each arc");
}

#[test]
fn a_rounded_border_without_background_has_round_corners() {
    let rec = paint(boxed(1.0, None));
    assert_eq!(rec.arcs.len(), 4, "round corners, not square bars");
    let starts: Vec<f32> = rec.arcs.iter().map(|(a, ..)| a.start).collect();
    let pi = std::f32::consts::PI;
    assert_eq!(starts, [1.5 * pi, 0.0, 0.5 * pi, pi]);
    // Each corner's centre is a radius in from its two sides.
    let (tl, br) = (rec.arcs[0].0, rec.arcs[2].0);
    assert_eq!((tl.cx, tl.cy, br.cx, br.cy), (8.0, 8.0, 94.0, 34.0));
}
