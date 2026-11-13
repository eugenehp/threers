//! Clamped text, scrolling something to the top, middle or bottom, and
//! styles for the light and the dark appearance.

use threers_ui::scrolling::ScrollStrategy;
use threers_ui::states::{Appearance, Theme};
use threers_ui::virtual_list::{RowHeight, VirtualList};
use threers_ui::{style, FixedWidthText, NodeKind, Ui};

fn layout(ui: &mut Ui) {
    ui.layout(400.0, 300.0, &mut FixedWidthText(10.0));
}

#[test]
fn line_clamp_keeps_so_many_lines_and_ends_the_last_in_an_ellipsis() {
    let mut ui = Ui::new();
    let root = ui.root();
    let p = ui.add_box(root, style! { width: 100px; white-space: normal; line-height: 20px; line-clamp: 2; });
    let t = ui.add_text(p, "one two three four five six seven eight");
    let short = ui.add_box(root, style! { width: 100px; white-space: normal; line-height: 20px; line-clamp: 3; });
    let s = ui.add_text(short, "one two");
    layout(&mut ui);
    // The prefixed spelling reads the same.
    let mut prefixed = threers_ui::Style::default();
    threers_ui::css::apply(&mut prefixed, "-webkit-line-clamp", "2");
    assert_eq!(prefixed.line_clamp, Some(2));
    assert_eq!(ui.rect(p).height, 40.0, "two lines high");
    let lines = ui.text_lines_of(t);
    assert_eq!(lines.len(), 2);
    assert!(lines[1].ends_with('…') && lines[1].chars().count() <= 10, "{lines:?}");
    assert_eq!(ui.rect(short).height, 20.0, "fewer lines than the clamp: as they are");
    assert!(!ui.text_lines_of(s).iter().any(|l| l.contains('…')));
    let _ = NodeKind::Box;
}

#[test]
fn a_list_row_and_a_node_scroll_to_the_top_middle_or_bottom() {
    let mut ui = Ui::new();
    let root = ui.root();
    let mut list = VirtualList::new(&mut ui, root, style! { height: 100px; }, 100, RowHeight::Uniform(20.0));
    let build = |ui: &mut Ui, row: threers_ui::NodeId, i: usize| {
        ui.add_text(row, format!("{i}"));
    };
    layout(&mut ui);
    list.sync(&mut ui, build);
    let c = list.container();
    for (strategy, want) in [(ScrollStrategy::Top, 400.0), (ScrollStrategy::Center, 360.0), (ScrollStrategy::Bottom, 320.0)] {
        list.scroll_to_with(&mut ui, 20, strategy);
        assert_eq!(ui.style(c).scroll_top, want, "{strategy:?}");
    }
    // Nearest: already showing (as laid out), it stays.
    layout(&mut ui);
    list.scroll_to_with(&mut ui, 18, ScrollStrategy::Nearest);
    assert_eq!(ui.style(c).scroll_top, 320.0);
    list.scroll_to_with(&mut ui, 99, ScrollStrategy::Top);
    assert_eq!(ui.style(c).scroll_top, 1900.0, "not past the end");

    // Any node in a scrolling box.
    let mut ui = Ui::new();
    let root = ui.root();
    let b = ui.add_box(root, style! { height: 100px; overflow: auto; });
    let kids: Vec<_> = (0..20).map(|_| ui.add_box(b, style! { height: 20px; })).collect();
    layout(&mut ui);
    ui.scroll_into_view_with(kids[10], ScrollStrategy::Center);
    layout(&mut ui);
    assert_eq!(ui.style(b).scroll_top, 160.0);
}

#[test]
fn appearance_styles_follow_light_and_dark() {
    let mut ui = Ui::new();
    let root = ui.root();
    let panel = ui.add_box(root, style! { height: 20px; background: #101010; });
    ui.appearance_style(panel, Appearance::Light, "background: #f5f5f5; color: #111111");
    ui.hover_style(panel, "background: #2563eb");
    layout(&mut ui);
    assert_eq!((ui.theme(), ui.appearance()), (Theme::System, Appearance::Light), "the system's, light until told");
    let bg = |ui: &Ui| ui.style(panel).background.map(|c| (c[0] * 255.0).round() as u8);
    assert_eq!(bg(&ui), Some(0xf5), "light: its light style");
    ui.set_system_appearance(Appearance::Dark);
    assert_eq!(bg(&ui), Some(0x10), "the system went dark: its own");
    ui.set_theme(Theme::Light);
    assert_eq!(bg(&ui), Some(0xf5), "the light theme, whatever the system");
    ui.set_system_appearance(Appearance::Light);
    ui.set_theme(Theme::Dark);
    assert_eq!(bg(&ui), Some(0x10));
    ui.set_theme(Theme::System);
    assert_eq!(bg(&ui), Some(0xf5));
    // Text inside follows its box's colour for the appearance.
    let label = ui.add_text(panel, "label");
    assert_eq!(ui.color_of(label), [17.0 / 255.0, 17.0 / 255.0, 17.0 / 255.0, 1.0]);
    ui.set_theme(Theme::Dark);
    assert_eq!(ui.color_of(label), [1.0; 4], "dark: the panel's own colour, the default");
    // A state goes over the appearance's style.
    ui.set_interaction(threers_ui::states::Interaction { hovered: vec![panel, root], ..Default::default() });
    assert_eq!(bg(&ui), Some(0x25));
}

/// A node with state styles changed through `update_style` keeps the change
/// as states come and go; through `style_mut`, the next state puts it back.
#[test]
fn update_style_changes_what_states_go_over() {
    let mut ui = Ui::new();
    let root = ui.root();
    let b = ui.add_box(root, style! { height: 20px; });
    ui.hover_style(b, "background: #ff0000");
    let hover = |ui: &mut Ui, on: bool| {
        ui.set_interaction(threers_ui::states::Interaction { hovered: if on { vec![b, root] } else { vec![] }, ..Default::default() })
    };
    ui.update_style(b, |s| s.height = threers_ui::Length::Px(40.0));
    hover(&mut ui, true);
    hover(&mut ui, false);
    assert_eq!(ui.style(b).height, threers_ui::Length::Px(40.0), "kept");
    ui.style_mut(b).height = threers_ui::Length::Px(60.0);
    hover(&mut ui, true);
    assert_eq!(ui.style(b).height, threers_ui::Length::Px(40.0), "style_mut's is lost to the base");
}

/// A box scrolls sideways too: its content's width, the wheel's sideways
/// delta and a finger moving sideways.
#[test]
fn a_box_scrolls_sideways() {
    use threers_ui::input::{Event, Input, TouchPhase};
    let mut ui = Ui::new();
    let root = ui.root();
    let strip = ui.add_box(root, style! { width: 200px; height: 50px; display: flex; overflow-x: auto; });
    let tiles: Vec<_> = (0..10).map(|_| ui.add_box(strip, style! { width: 60px; height: 40px; flex-shrink: 0; })).collect();
    layout(&mut ui);
    assert!(ui.style(strip).scroll_x && !ui.style(strip).scroll, "overflow-x only");
    assert_eq!(ui.scroll_extent_x(strip), (600.0, 200.0, 0.0));
    ui.scroll_to_x(strip, 1000.0);
    layout(&mut ui);
    assert_eq!(ui.rect(tiles[9]).x + 60.0, 200.0, "the last tile at the right edge");
    let mut input: Input<()> = Input::new();
    input.smooth_wheel = false;
    input.dispatch(&mut ui, &mut (), Event::Wheel { x: 10.0, y: 10.0, dx: -100.0, dy: 0.0 });
    layout(&mut ui);
    assert_eq!(ui.scroll_extent_x(strip).2, 300.0);
    // A finger dragging left scrolls on to the right.
    input.dispatch(&mut ui, &mut (), Event::Touch { id: 1, phase: TouchPhase::Start, x: 150.0, y: 20.0, time: 0.0 });
    input.dispatch(&mut ui, &mut (), Event::Touch { id: 1, phase: TouchPhase::Move, x: 120.0, y: 21.0, time: 0.05 });
    input.dispatch(&mut ui, &mut (), Event::Touch { id: 1, phase: TouchPhase::Move, x: 100.0, y: 21.0, time: 0.1 });
    input.dispatch(&mut ui, &mut (), Event::Touch { id: 1, phase: TouchPhase::End, x: 100.0, y: 21.0, time: 0.5 });
    layout(&mut ui);
    assert_eq!(ui.scroll_extent_x(strip).2, 350.0);
    // `overflow` alone scrolls both ways.
    let mut s = threers_ui::Style::default();
    threers_ui::css::apply(&mut s, "overflow", "auto");
    assert!(s.scroll && s.scroll_x);
}
