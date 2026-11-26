//! What only moves is moved, not laid out; and is where a layout puts it.

use threers_ui::rich_text::TextRun;
use threers_ui::{Length, Style, TextMeasure, Ui};

/// `height: 20px; position: absolute`, as much of CSS as is wanted here.
fn boxed(css: &str) -> Style {
    let mut s = Style::default();
    for d in css.split(';') {
        let (k, v) = d.split_once(':').unwrap();
        let px = || Length::Px(v.trim().trim_end_matches("px").parse().unwrap());
        match k.trim() {
            "width" => s.width = px(),
            "height" => s.height = px(),
            "top" => s.top = px(),
            "bottom" => s.bottom = px(),
            "left" => s.left = px(),
            "position" => s.position = threers_ui::Position::Absolute,
            "overflow-y" => s.clip = true,
            k => panic!("{k}"),
        }
    }
    s
}

struct Mono;
impl TextMeasure for Mono {
    fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
        (text.chars().count() as f32 * style.font_size * 0.6, style.font_size * 1.2)
    }
}

#[test]
fn an_absolute_boxs_top_and_a_scroll_are_no_layout() {
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(root, boxed("overflow-y: auto; height: 100px; width: 100px"));
    for _ in 0..20 {
        ui.add_box(list, boxed("height: 20px"));
    }
    let bar = ui.add_box(root, boxed("position: absolute; top: 5px; left: 90px; width: 6px; height: 30px"));
    let inside = ui.add_box(bar, boxed("height: 10px"));
    ui.layout(200.0, 200.0, &mut Mono);
    let passes = ui.layout_passes();
    ui.update_style(bar, |s| s.top = Length::Px(42.0));
    ui.update_style(list, |s| s.scroll_top = 60.0);
    ui.layout(200.0, 200.0, &mut Mono);
    assert_eq!(ui.layout_passes(), passes, "moved, not laid out");
    let moved = (ui.rect(bar), ui.rect(inside), ui.scroll_extent(list));
    ui.invalidate_layout();
    ui.layout(200.0, 200.0, &mut Mono);
    assert!(ui.layout_passes() > passes);
    assert_eq!(moved, (ui.rect(bar), ui.rect(inside), ui.scroll_extent(list)), "where a layout has them");
    // one sized by both its ends, or in a box that clips, is laid out
    let both = ui.add_box(root, boxed("position: absolute; top: 5px; bottom: 5px; width: 6px"));
    ui.layout(200.0, 200.0, &mut Mono);
    let passes = ui.layout_passes();
    ui.update_style(both, |s| s.top = Length::Px(50.0));
    ui.layout(200.0, 200.0, &mut Mono);
    assert!(ui.layout_passes() > passes);
    assert_eq!(ui.rect(both).height, 145.0);
}

#[test]
fn runs_of_another_weight_wrap_anew_and_of_another_colour_the_same() {
    struct Bold;
    impl TextMeasure for Bold {
        fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
            let k = if style.font_weight == threers_ui::FontWeight::Bold { 2.0 } else { 1.0 };
            (text.chars().count() as f32 * 10.0 * k, 12.0)
        }
    }
    let mut ui = Ui::new();
    let root = ui.root();
    let b = ui.add_box(root, boxed("width: 100px"));
    let t = ui.add_text(b, "aaaa bbbb cccc");
    ui.update_style(t, |s| s.wrap = true);
    ui.set_text_runs(t, vec![TextRun::new(14)]);
    ui.layout(300.0, 300.0, &mut Bold);
    let plain = ui.rect(b).height;
    ui.set_text_runs(t, vec![TextRun::new(14).weight(threers_ui::FontWeight::Bold)]);
    ui.layout(300.0, 300.0, &mut Bold);
    assert!(ui.rect(b).height > plain);
    ui.set_text_runs(t, vec![TextRun::new(14).color([1.0, 0.0, 0.0, 1.0])]);
    ui.layout(300.0, 300.0, &mut Bold);
    assert_eq!(ui.rect(b).height, plain);
}
