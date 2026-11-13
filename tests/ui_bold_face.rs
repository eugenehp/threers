//! A UI face given a bold file draws bold text with it: more ink than the
//! regular grown for bold, in the bold's own letterforms. (Its own binary:
//! the UI face is set for the whole process.)
#![cfg(not(target_arch = "wasm32"))]

use threers::captions::CaptionFont;
use threers::style;
use threers::ui::{paint_to_rgba, FixedWidthText, Ui};

fn ink(bold: bool) -> f64 {
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { padding: 10px; background: #ffffff; color: #000000; });
    let t = ui.add_box(root, style! { font-size: 18px; });
    ui.update_style(t, |s| s.font_weight = if bold { threers::ui::FontWeight::Bold } else { threers::ui::FontWeight::Normal });
    ui.add_text(t, "Gauges and charts 64%");
    ui.layout(320.0, 60.0, &mut FixedWidthText(9.0));
    let px = paint_to_rgba(&ui, 1.0, 320, 60);
    px.chunks(4).map(|p| 255.0 - p[0] as f64).sum()
}

#[test]
fn a_bold_file_draws_bold_text() {
    CaptionFont::set_ui(include_bytes!("../src/openscad/DejaVuSans.ttf")).unwrap();
    let (regular, grown) = (ink(false), ink(true));
    CaptionFont::set_ui_style(true, false, include_bytes!("../web/ui-demo/DejaVuSans-Bold.ttf")).unwrap();
    let (regular2, real) = (ink(false), ink(true));
    assert!((regular - regular2).abs() < regular * 0.01, "the regular unchanged: {regular} {regular2}");
    assert!(real > regular * 1.3, "real bold: {real} against {regular}");
    // (A regular grown for bold adds about half again; the bold file,
    // with its own letterforms, more.)
    assert!(real > grown * 1.08, "more than a grown regular: {real} against {grown}");
}
