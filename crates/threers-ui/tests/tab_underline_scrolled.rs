//! The tab underline stays under the selected tab when the tab row is
//! scrolled sideways.

use threers_ui::components::tabs;
use threers_ui::input::Input;
use threers_ui::{binding, style, FixedWidthText, Ui};

#[test]
fn the_underline_follows_a_tab_scrolled_into_view() {
    #[derive(Default)]
    struct S {
        tab: usize,
    }
    let mut ui = Ui::new();
    let mut input: Input<S> = Input::new();
    let mut s = S::default();
    let root = ui.root();
    let host = ui.add_box(root, style! { width: 200px; display: flex; flex-direction: column; });
    let t = tabs(&mut ui, &mut input, &s, host, &["People", "Split", "Actions", "Tasks", "Controls", "Screens"], binding!(|s: S| s.tab));
    let lay = |ui: &mut Ui| ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    lay(&mut ui);
    s.tab = 4;
    let mut time = 0.0;
    for _ in 0..30 {
        time += 0.05;
        input.tick(&mut ui, &mut s, time);
        ui.step(0.05);
        lay(&mut ui);
    }
    let tab = ui.rect(t.tabs[4]);
    let line = ui.rect(*ui.children(t.list).last().unwrap());
    assert!(ui.style(t.list).scroll_left > 0.0, "scrolled to show it");
    assert_eq!((line.x, line.width), (tab.x, tab.width), "under it");
}
