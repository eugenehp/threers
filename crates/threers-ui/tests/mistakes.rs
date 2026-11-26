//! A release build's way with mistakes (see `threers_ui::mistake`): bad
//! CSS and a gone node's id are reported and skipped, not a crash. (Its
//! own test binary: the setting is the process's.)

use std::sync::{Arc, Mutex};

use threers_ui::{Style, Ui, css, mistake};

#[test]
fn mistakes_are_reported_and_skipped() {
    mistake::set_panics(Some(false));
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let s2 = seen.clone();
    mistake::set_hook(move |m| s2.lock().unwrap().push(m.to_owned()));

    // CSS it doesn't take: the rest of the style still applies
    let mut s = Style::default();
    css::apply(&mut s, "z-index", "10");
    css::apply(&mut s, "opacity", "0.5");
    assert_eq!(s.opacity, 0.5);

    // a node removed, then used: read as an empty box, changes dropped
    let mut ui = Ui::new();
    let root = ui.root();
    let gone = ui.add_box(root, Style::default());
    ui.remove(gone);
    // (its slot taken by another)
    let kept = ui.add_box(root, Style::default());
    let _ = ui.rect(gone);
    ui.style_mut(gone).opacity = 0.5;
    ui.set_style(gone, Style::default());
    assert!(ui.contains(kept));
    assert_eq!(ui.style(kept).opacity, 1.0, "the new node untouched");

    let seen = seen.lock().unwrap();
    assert!(seen.iter().any(|m| m.contains("unsupported `z-index")), "{seen:?}");
    assert!(seen.iter().any(|m| m.contains("is not in this Ui")), "{seen:?}");
    // each once
    let stale = seen.iter().filter(|m| m.contains("is not in this Ui")).count();
    assert!(stale <= 3, "{seen:?}");
}
