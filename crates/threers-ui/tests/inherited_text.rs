//! Text takes its box's colour and font as CSS inherits them: white given
//! in CSS is a colour, and a box restyled carries its text along.

use threers_ui::{style, FontWeight, Ui};

#[test]
fn white_given_in_css_is_inherited() {
    let mut ui = Ui::new();
    let page = ui.add_box(ui.root(), style! { color: #222222; });
    let button = ui.add_box(page, style! { color: #ffffff; background: #2f7d5b; });
    let label = ui.add_text(button, "Check");
    assert_eq!(ui.color_of(label), [1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn a_restyled_box_carries_its_text() {
    let mut ui = Ui::new();
    let button = ui.add_box(ui.root(), style! { color: #888888; font-size: 15px; });
    let label = ui.add_text(button, "Check");
    ui.set_style(
        button,
        style! { color: #ffffff; font-size: 16px; font-weight: bold; white-space: normal; },
    );
    let s = ui.style(label);
    assert_eq!(
        (s.font_size, s.font_weight, s.wrap),
        (16.0, FontWeight::Bold, true)
    );
    assert_eq!(ui.color_of(label), [1.0, 1.0, 1.0, 1.0]);
    // A text given a style of its own keeps it.
    ui.set_style(label, style! { font-size: 30px; });
    ui.set_style(button, style! { font-size: 12px; });
    assert_eq!(ui.style(label).font_size, 30.0);
}

#[test]
fn text_align_is_parsed_inherited_and_centres_lines() {
    use threers_ui::{
        FixedWidthText, Painter, Rect, Rgba, Style, TextAlign, TextMeasure, TextStyle,
    };
    /// Where each line of text is drawn.
    struct Lines(Vec<(String, f32)>);
    impl Painter for Lines {
        fn fill_rect(&mut self, _: Rect, _: Rgba) {}
        fn text(&mut self, line: Rect, text: &str, _: &TextStyle) {
            self.0.push((text.to_owned(), line.x));
        }
        fn path(&mut self, _: Rect, _: &[String], _: [f32; 4], _: Option<f32>, _: bool, _: Rgba) {}
    }
    impl TextMeasure for Lines {
        fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
            FixedWidthText(10.0).measure(text, style)
        }
    }
    let mut ui = Ui::new();
    let card = ui.add_box(
        ui.root(),
        style! { width: 60px; text-align: center; white-space: normal; },
    );
    let text = ui.add_text(card, "aaaa bb");
    assert_eq!(ui.style(text).text_align, TextAlign::Center);
    ui.layout(400.0, 400.0, &mut FixedWidthText(10.0));
    let mut lines = Lines(vec![]);
    ui.paint(&mut lines);
    // "aaaa" (40 px) and "bb" (20 px), each centred in 60 px.
    assert_eq!(
        lines.0,
        [("aaaa".to_owned(), 10.0), ("bb".to_owned(), 20.0)]
    );
    // Restyled to the end, the text follows.
    ui.set_style(
        card,
        style! { width: 100px; text-align: right; white-space: normal; },
    );
    assert_eq!(ui.style(text).text_align, TextAlign::End);
}

/// A box described with a name, then without one: the name goes.
#[test]
fn a_name_taken_away_is_gone() {
    use threers_ui::input::Input;
    use threers_ui::view::{El, View};
    use threers_ui::{Role, Ui};
    let mut ui = Ui::new();
    let mut input: Input<()> = Input::new();
    let root = ui.root();
    let mut view = View::new(&mut ui, root, "display: flex");
    view.update(
        &mut ui,
        &mut input,
        vec![El::div("padding: 4px")
            .key("pill")
            .role(Role::Status, "Loading 40%")],
    );
    assert!(ui
        .accessibility_tree()
        .iter()
        .any(|a| a.name == "Loading 40%"));
    view.update(&mut ui, &mut input, vec![El::div("width: 0").key("pill")]);
    assert!(!ui
        .accessibility_tree()
        .iter()
        .any(|a| a.name == "Loading 40%"));
}

/// A box's font family reaches the boxes under it that have none of their
/// own, and their text, as CSS inherits it; changed, it follows; a box with
/// its own keeps it.
#[test]
fn font_family_is_inherited_through_boxes() {
    let mut ui = Ui::new();
    let page = ui.add_box(ui.root(), style! { font-family: "Roboto"; });
    let card = ui.add_box(page, style! { padding: 4px; });
    let button = ui.add_box(card, style! { font-weight: bold; });
    let label = ui.add_text(button, "Check");
    let mono = ui.add_box(card, style! { font-family: "monospace"; });
    let code = ui.add_text(mono, "x");
    assert_eq!(ui.style(label).font_family, "Roboto");
    assert_eq!(ui.style(code).font_family, "monospace");
    // Restyled with no family: still the page's (and no change seen).
    ui.set_style(button, style! { font-weight: bold; });
    assert_eq!(ui.style(button).font_family, "Roboto");
    // The page's changes: down it goes, but not into the one with its own.
    ui.set_style(page, style! { font-family: "serif"; });
    assert_eq!(ui.style(card).font_family, "serif");
    assert_eq!(ui.style(label).font_family, "serif");
    assert_eq!(ui.style(code).font_family, "monospace");
    ui.update_style(page, |s| s.font_family = "Inter".into());
    assert_eq!(ui.style(label).font_family, "Inter");
}

/// Size, weight and alignment pass down through boxes that do not set
/// them, as CSS has it; line height comes along with the size, and a box
/// with a size of its own keeps its own (normal) line height.
#[test]
fn font_size_weight_and_alignment_are_inherited() {
    use threers_ui::TextAlign;
    let mut ui = Ui::new();
    let small = ui.add_box(
        ui.root(),
        style! { font-size: 13px; line-height: 18px; font-weight: bold; text-align: center; },
    );
    let inner = ui.add_box(small, style! { padding: 2px; });
    let label = ui.add_text(inner, "Hint");
    let s = ui.style(label);
    assert_eq!(
        (s.font_size, s.line_height, s.font_weight, s.text_align),
        (13.0, Some(18.0), FontWeight::Bold, TextAlign::Center)
    );
    // A size of its own: not the parent's line height.
    let big = ui.add_box(small, style! { font-size: 30px; });
    let title = ui.add_text(big, "Title");
    assert_eq!(
        (ui.style(title).font_size, ui.style(title).line_height),
        (30.0, None)
    );
    // The parent restyled: down it goes.
    ui.set_style(small, style! { font-size: 15px; });
    assert_eq!(ui.style(label).font_size, 15.0);
    assert_eq!(ui.style(label).font_weight, FontWeight::Normal);
}
