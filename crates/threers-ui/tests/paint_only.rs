//! A change to how a box looks is drawn without a layout; a change to where
//! boxes go is laid out.

use threers_ui::transform::Transformation;
use threers_ui::{style, FixedWidthText, Length, Ui};

#[test]
fn looks_repaint_and_sizes_lay_out() {
    let mut ui = Ui::new();
    let root = ui.root();
    let card = ui.add_box(root, style! { width: 120px; height: 40px; background: #334155; });
    let label = ui.add_text(card, "Hello");
    let icon = ui.add_path(card, style! { width: 16px; height: 16px; }, vec!["M0 0 L10 0 L10 10 Z".into()], [0.0, 0.0, 10.0, 10.0], None, true);
    let m = &mut FixedWidthText(7.0);
    ui.layout(400.0, 300.0, m);
    let laid = ui.layout_passes();
    assert!(!ui.needs_frame());

    // Colour, opacity, corners and a shape turned: drawn again, not laid out.
    let mut s = ui.style(card).clone();
    s.background = Some([0.9, 0.2, 0.2, 1.0]);
    s.opacity = 0.5;
    s.border_radius = Length::Px(8.0);
    ui.set_style(card, s);
    ui.update_style(label, |s| s.color = [0.1, 0.1, 0.1, 1.0]);
    ui.set_path_transform(icon, Transformation { rotate: 0.4, ..Default::default() });
    assert!(ui.needs_frame(), "a repaint is asked for");
    ui.layout(400.0, 300.0, m);
    assert_eq!(ui.layout_passes(), laid, "only the look changed");
    assert_eq!(ui.style(card).background, Some([0.9, 0.2, 0.2, 1.0]));
    assert!(!ui.needs_frame());

    // A size: laid out.
    ui.update_style(card, |s| s.width = Length::Px(200.0));
    ui.layout(400.0, 300.0, m);
    assert!(ui.layout_passes() > laid);
    assert_eq!(ui.rect(card).width, 200.0);
}
