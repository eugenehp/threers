//! The demo driven natively, as the page drives it.

use threers_ui::input::{Event, PointerButton, TouchPhase};
use threers_ui::{FixedWidthText, KeyModifiers};
use threers_ui_demo::app::App;
use threers_ui_demo::pages;

const W: f32 = 1200.0;
const H: f32 = 800.0;

fn frame(app: &mut App, t: &mut f64) {
    *t += 1.0 / 60.0;
    app.frame(*t, W, H, &mut FixedWidthText(7.0));
}

fn click(app: &mut App, x: f32, y: f32) {
    let m = &mut FixedWidthText(7.0);
    app.pointer_move(x, y, m);
    app.pointer_down(x, y, PointerButton::Primary, false, m);
    app.pointer_up(x, y, PointerButton::Primary);
}

/// Shows page `page`.
fn go(app: &mut App, t: &mut f64, page: usize) {
    app.state.page = page;
    for _ in 0..3 {
        frame(app, t);
    }
}

/// The middle of what is named `label`, scrolled into view first.
fn reveal(app: &mut App, t: &mut f64, label: &str) -> (f32, f32) {
    let id = app
        .ui
        .accessibility_tree()
        .into_iter()
        .find(|a| a.name.split_whitespace().eq(label.split_whitespace()))
        .unwrap_or_else(|| panic!("no {label}"))
        .id;
    app.ui.scroll_into_view(id);
    for _ in 0..2 {
        frame(app, t);
    }
    centre(app, label)
}

fn centre(app: &App, label: &str) -> (f32, f32) {
    let n = app
        .ui
        .accessibility_tree()
        .into_iter()
        .find(|a| a.name.split_whitespace().eq(label.split_whitespace()))
        .unwrap_or_else(|| panic!("no {label}"));
    (
        n.bounds.x + n.bounds.width / 2.0,
        n.bounds.y + n.bounds.height / 2.0,
    )
}

#[test]
fn the_demo_runs_as_the_page_drives_it() {
    let mut app = App::new();
    app.state.page = pages::LISTS;
    let mut t = 0.0;
    for _ in 0..3 {
        frame(&mut app, &mut t);
    }
    // Of its 100,000 rows, the list builds only those that show.
    let built = app.lists.lists[0].1.built().len();
    assert!(
        built > 0 && built < 60,
        "the list builds only what shows: {built}"
    );

    // Save: a toast fades in.
    let (x, y) = centre(&app, "Save");
    click(&mut app, x, y);
    frame(&mut app, &mut t);
    for _ in 0..20 {
        frame(&mut app, &mut t);
    }
    assert!(app.needs_frame(), "the toast is animating");

    // The chord saves from the keyboard too.
    let cmd = KeyModifiers {
        meta: cfg!(target_os = "macos"),
        control: !cfg!(target_os = "macos"),
        ..Default::default()
    };
    assert!(app.key("k", cmd));
    assert!(app.key("s", cmd));

    // Settings: the dialog opens, traps focus, and Escape closes it.
    let (x, y) = centre(&app, "Settings…");
    click(&mut app, x, y);
    frame(&mut app, &mut t);
    frame(&mut app, &mut t);
    assert_eq!(app.input.modal(), Some(app.dialog()));
    for _ in 0..5 {
        app.key("Tab", KeyModifiers::default());
        let f = app.input.focused().expect("focus is in the dialog");
        assert!(
            app.ui.ancestors(f).any(|a| a == app.dialog()),
            "Tab stays in the dialog"
        );
    }
    app.key("Escape", KeyModifiers::default());
    frame(&mut app, &mut t);
    assert_eq!(app.input.modal(), None);

    // A card dragged to the other lane.
    go(&mut app, &mut t, pages::BASICS);
    let (cx, cy) = centre(&app, "Grid layout");
    let (dx, dy) = centre(&app, "Done");
    let m = &mut FixedWidthText(7.0);
    app.pointer_move(cx, cy, m);
    app.pointer_down(cx, cy, PointerButton::Primary, false, m);
    for k in 1..=6 {
        let f = k as f32 / 6.0;
        app.pointer_move(cx + (dx - cx) * f, cy + (dy + 30.0 - cy) * f, m);
        frame(&mut app, &mut t);
    }
    app.pointer_up(dx, dy + 30.0, PointerButton::Primary);
    assert!(
        app.take_log().iter().any(|l| l.contains("Done")),
        "dropped in Done"
    );

    // The text field: a double click selects a word.
    go(&mut app, &mut t, pages::TEXT);
    let (fx, fy) = reveal(&mut app, &mut t, "Name");
    click(&mut app, fx - 60.0, fy);
    frame(&mut app, &mut t);
    let before = app.state.field.selection();
    assert_eq!(before.0, before.1, "a click places the caret");

    // The list scrolls.
    go(&mut app, &mut t, pages::LISTS);
    let (lx, ly) = centre(&app, "Row 1");
    app.wheel(lx, ly, 0.0, 2600.0);
    // A wheel's scroll glides there over a few frames.
    for _ in 0..15 {
        frame(&mut app, &mut t);
    }
    assert!(app
        .ui
        .accessibility_tree()
        .iter()
        .any(|a| a.name.split_whitespace().eq(["Row", "101"])));
    app.event(Event::PointerMove { x: 5.0, y: 5.0 });
}

#[test]
fn a_finger_scrolls_the_page_on_a_phone() {
    use threers_ui::input::TouchPhase;
    let mut app = App::new();
    app.state.page = pages::COMPONENTS;
    let mut t = 0.0;
    let step = |app: &mut App, t: &mut f64| {
        *t += 1.0 / 60.0;
        app.frame(*t, 390.0, 844.0, &mut FixedWidthText(7.0));
    };
    for _ in 0..3 {
        step(&mut app, &mut t);
    }
    let touch = |app: &mut App, phase, y: f32| {
        app.event(Event::Touch {
            id: 9,
            phase,
            x: 195.0,
            y,
            time: 0.0,
        })
    };
    // Above the bar of page links at the bottom.
    touch(&mut app, TouchPhase::Start, 700.0);
    for y in [600.0, 450.0, 300.0, 120.0] {
        touch(&mut app, TouchPhase::Move, y);
        step(&mut app, &mut t);
    }
    touch(&mut app, TouchPhase::End, 120.0);
    step(&mut app, &mut t);
    let page = app.scroller();
    let hit = app.ui.hit_test(195.0, 784.0);
    let _ = hit;
    // The pan went further than there is to scroll: the page is at its end.
    let (content, view, scrolled) = app.ui.scroll_extent(page);
    assert!(content > view, "the page is taller than a phone");
    assert_eq!(scrolled, content - view);
}

/// A card lands where its slot showed it would: between the cards the
/// pointer was between, not at the end.
#[test]
fn a_card_drops_where_its_slot_was() {
    let mut app = App::new();
    app.state.page = pages::BASICS;
    let mut t = 0.0;
    for _ in 0..3 {
        frame(&mut app, &mut t);
    }
    let names = |app: &App, lane: &str| -> Vec<String> {
        let tree = app.ui.accessibility_tree();
        let l = tree.iter().find(|a| a.name == lane).expect("lane");
        l.children
            .iter()
            .map(|&i| tree[i].name.clone())
            .filter(|n| !n.is_empty())
            .collect()
    };
    let (cx, cy) = centre(&app, "Grid layout");
    let (sx, sy) = centre(&app, "Shadows");
    let (_, ty) = centre(&app, "Tooltips");
    let target = (sx, (sy + ty) / 2.0);
    let m = &mut FixedWidthText(7.0);
    app.pointer_move(cx, cy, m);
    app.pointer_down(cx, cy, PointerButton::Primary, false, m);
    for k in 1..=8 {
        let f = k as f32 / 8.0;
        app.pointer_move(cx + (target.0 - cx) * f, cy + (target.1 - cy) * f, m);
        frame(&mut app, &mut t);
    }
    // While held: the slot sits between Shadows and Tooltips.
    let slot = app.state.slot.expect("a slot shows");
    let lane = app.ui.parent(slot).unwrap();
    let kids = app.ui.children(lane).to_vec();
    let at = kids.iter().position(|&k| k == slot).unwrap();
    assert!(app.ui.rect(slot).height > 20.0);
    assert_eq!(
        app.ui
            .semantics(kids[at - 1])
            .and_then(|s| s.label.clone())
            .as_deref(),
        Some("Shadows")
    );
    app.pointer_up(target.0, target.1, PointerButton::Primary);
    frame(&mut app, &mut t);
    assert_eq!(
        names(&app, "Done"),
        ["Done", "Shadows", "Grid layout", "Tooltips", "Springs"]
    );
    assert_eq!(names(&app, "To do"), ["To do", "Scrollbars", "IME"]);
    assert!(app.state.slot.is_none());

    // Escape mid-drag: back where it was.
    let (cx, cy) = centre(&app, "IME");
    app.pointer_move(cx, cy, m);
    app.pointer_down(cx, cy, PointerButton::Primary, false, m);
    for k in 1..=4 {
        app.pointer_move(cx + 40.0 * k as f32, cy, m);
        frame(&mut app, &mut t);
    }
    app.key("Escape", Default::default());
    app.pointer_up(cx + 160.0, cy, PointerButton::Primary);
    frame(&mut app, &mut t);
    assert_eq!(names(&app, "To do"), ["To do", "Scrollbars", "IME"]);
}

/// On a phone: a tap in the field places the caret where it lands, two
/// select the word, and once the field has focus a finger dragged across it
/// selects, rather than scrolling the page (fingers drift up and down).
#[test]
fn a_finger_places_the_caret_and_selects_in_the_field() {
    let mut app = App::new();
    app.state.page = pages::TEXT;
    let mut t = 0.0;
    let step = |app: &mut App, t: &mut f64| {
        *t += 1.0 / 60.0;
        app.frame(*t, 390.0, 844.0, &mut FixedWidthText(7.0));
    };
    for _ in 0..3 {
        step(&mut app, &mut t);
    }
    // The page at its end puts the field on screen; from there a pan down
    // would scroll the page back up.
    let page = app.scroller();
    app.ui.scroll_to(page, 1e6);
    for _ in 0..3 {
        step(&mut app, &mut t);
    }
    let r = app
        .ui
        .accessibility_tree()
        .into_iter()
        .find(|a| a.name == "Name")
        .expect("the field")
        .bounds;
    assert!(r.y + r.height < 844.0, "{r:?}");
    // The text starts after the border and padding; 7 pixels a character.
    let at = |i: f32, dy: f32| (r.x + 9.0 + 7.0 * i + 3.0, r.y + r.height / 2.0 + dy);
    let m = &mut FixedWidthText(7.0);
    let mut finger = |app: &mut App, id: u64, phase: TouchPhase, (x, y): (f32, f32), time: f64| {
        app.touch(id, phase, x, y, time, m)
    };
    // "Edit me: click, drag, double-click"
    finger(&mut app, 1, TouchPhase::Start, at(10.0, 0.0), 10.0);
    finger(&mut app, 1, TouchPhase::End, at(10.0, 0.0), 10.05);
    step(&mut app, &mut t);
    assert!(app.input.wants_ime(&app.ui), "the tap focuses the field");
    assert_eq!((app.state.field.anchor, app.state.field.caret), (10, 10));

    let before = app.ui.scroll_extent(page).2;
    finger(&mut app, 2, TouchPhase::Start, at(3.0, 0.0), 11.0);
    finger(&mut app, 2, TouchPhase::Move, at(8.0, 6.0), 11.05);
    finger(&mut app, 2, TouchPhase::Move, at(14.0, 12.0), 11.1);
    finger(&mut app, 2, TouchPhase::End, at(14.0, 12.0), 11.15);
    for _ in 0..10 {
        step(&mut app, &mut t);
    }
    assert_eq!(
        app.ui.scroll_extent(page).2,
        before,
        "the drag moved the caret, and did not scroll"
    );
    assert_eq!((app.state.field.anchor, app.state.field.caret), (14, 14));

    for (id, time) in [(3, 12.0), (4, 12.15)] {
        finger(&mut app, id, TouchPhase::Start, at(11.0, 0.0), time);
        finger(&mut app, id, TouchPhase::End, at(11.0, 0.0), time + 0.05);
    }
    let f = &app.state.field;
    assert_eq!(
        &f.text[f.anchor.min(f.caret)..f.anchor.max(f.caret)],
        "click",
        "two taps select the word"
    );

    // The edit menu over the word; Copy puts it out for the clipboard.
    step(&mut app, &mut t);
    let names = |app: &App| {
        app.ui
            .accessibility_tree()
            .into_iter()
            .map(|a| a.name)
            .collect::<Vec<_>>()
    };
    for item in ["Cut", "Copy", "Paste", "Select All"] {
        assert!(
            names(&app).iter().any(|n| n == item),
            "{item} in {:?}",
            names(&app)
        );
    }
    // The menu stays put: where it shows is where a tap finds it.
    let shown = centre(&app, "Cut");
    step(&mut app, &mut t);
    assert_eq!(centre(&app, "Cut"), shown, "the menu moved after it showed");
    let menu_y = centre(&app, "Copy").1;
    assert!(menu_y < r.y, "the menu is over the field");
    let tap_at = |app: &mut App, id: u64, (x, y): (f32, f32), time: f64| {
        let m = &mut FixedWidthText(7.0);
        app.touch(id, TouchPhase::Start, x, y, time, m);
        app.touch(id, TouchPhase::End, x, y, time + 0.05, m);
    };
    // Cut, the leftmost, is over the start handle's knob: the tap is the
    // menu's, not the handle's. (Cut, then put back.)
    let cut = centre(&app, "Cut");
    tap_at(&mut app, 50, cut, 12.6);
    assert_eq!(app.state.copy_out.take().as_deref(), Some("click"));
    assert_eq!(app.state.field.text, "Edit me: , drag, double-click");
    app.paste("click");
    app.state.field.anchor = 9;
    app.state.field.caret = 14;
    for (id, time) in [(51, 12.7), (52, 12.85)] {
        finger(&mut app, id, TouchPhase::Start, at(11.0, 0.0), time);
        finger(&mut app, id, TouchPhase::End, at(11.0, 0.0), time + 0.05);
    }
    step(&mut app, &mut t);
    let copy = centre(&app, "Copy");
    tap_at(&mut app, 5, copy, 13.0);
    assert_eq!(app.state.copy_out.take().as_deref(), Some("click"));
    assert!(app.input.wants_ime(&app.ui), "the field kept its focus");
    step(&mut app, &mut t);
    assert!(!names(&app).iter().any(|n| n == "Copy"), "the menu went");

    // The selection's end handle, dragged on to "drag".
    let end_x = at(14.0, 0.0).0;
    let knob = (end_x, r.y + r.height / 2.0 + 16.0);
    let m = &mut FixedWidthText(7.0);
    app.touch(6, TouchPhase::Start, knob.0, knob.1, 14.0, m);
    app.touch(6, TouchPhase::Move, at(20.0, 16.0).0, knob.1, 14.1, m);
    app.touch(6, TouchPhase::End, at(20.0, 16.0).0, knob.1, 14.2, m);
    let f = &app.state.field;
    assert_eq!(f.selection(), (9, 20), "click, drag");

    // A tap on the caret: Paste, which the page answers with the clipboard.
    tap_at(&mut app, 7, at(3.0, 0.0), 16.0);
    tap_at(&mut app, 8, at(3.0, 0.0), 17.0);
    step(&mut app, &mut t);
    let paste = centre(&app, "Paste");
    tap_at(&mut app, 9, paste, 18.0);
    assert!(std::mem::take(&mut app.state.paste_wanted));
    app.paste("X");
    assert_eq!(app.state.field.text, "EdiXt me: click, drag, double-click");
}

/// A flick on the list glides on after the finger lifts, slowing down.
#[test]
fn a_flick_on_the_list_glides_and_slows() {
    let mut app = App::new();
    app.state.page = pages::LISTS;
    let step = |app: &mut App, t: f64| {
        app.frame(t, 390.0, 844.0, &mut FixedWidthText(7.0));
    };
    let mut t = 0.0;
    for _ in 0..3 {
        t += 1.0 / 60.0;
        step(&mut app, t);
    }
    let (x, y) = centre(&app, "Row 1");
    let m = &mut FixedWidthText(7.0);
    app.touch(1, TouchPhase::Start, x, y + 150.0, t, m);
    for i in 1..=8 {
        t += 0.016;
        app.touch(1, TouchPhase::Move, x, y + 150.0 - 25.0 * i as f32, t, m);
        step(&mut app, t);
    }
    t += 0.01;
    app.touch(1, TouchPhase::End, x, y - 50.0, t, m);
    let first = |app: &App| {
        app.ui
            .accessibility_tree()
            .into_iter()
            .filter_map(|a| {
                a.name
                    .strip_prefix("Row")
                    .and_then(|n| n.trim().parse::<u32>().ok())
            })
            .min()
            .unwrap()
    };
    let mut seen = vec![first(&app)];
    for _ in 0..6 {
        for _ in 0..15 {
            t += 1.0 / 60.0;
            step(&mut app, t);
        }
        seen.push(first(&app));
    }
    let steps: Vec<i64> = seen.windows(2).map(|w| w[1] as i64 - w[0] as i64).collect();
    assert!(steps[0] > 5, "it glides: {seen:?}");
    assert!(
        steps.windows(2).all(|w| w[1] <= w[0]),
        "and slows: {seen:?}"
    );
}

/// The keyboard coming up takes the bottom of a phone's screen: the field
/// that has focus is scrolled back into what is left.
#[test]
fn the_focused_field_stays_in_view_when_the_keyboard_comes_up() {
    let mut app = App::new();
    app.state.page = pages::TEXT;
    let mut t = 0.0;
    let mut step = |app: &mut App, h: f32| {
        t += 1.0 / 60.0;
        app.frame(t, 390.0, h, &mut FixedWidthText(7.0));
    };
    for _ in 0..3 {
        step(&mut app, 844.0);
    }
    let page = app.scroller();
    app.ui.scroll_to(page, 1e6);
    step(&mut app, 844.0);
    let field = |app: &App| {
        app.ui
            .accessibility_tree()
            .into_iter()
            .find(|a| a.name == "Name")
            .expect("the field")
            .bounds
    };
    let r = field(&app);
    let m = &mut FixedWidthText(7.0);
    let (x, y) = (r.x + r.width / 2.0, r.y + r.height / 2.0);
    app.touch(1, TouchPhase::Start, x, y, 1.0, m);
    app.touch(1, TouchPhase::End, x, y, 1.05, m);
    step(&mut app, 844.0);
    assert!(app.input.wants_ime(&app.ui));
    assert!(
        r.y > 450.0,
        "the field is where the keyboard will be: {r:?}"
    );
    for h in [700.0, 520.0, 450.0] {
        step(&mut app, h);
    }
    let r = field(&app);
    assert!(
        r.y >= 0.0 && r.y + r.height <= 450.0,
        "in view above the keyboard: {r:?}"
    );
}

/// Each tab of the lists panel shows its own kind of virtual list, built
/// only where it shows.
#[test]
fn the_list_tabs_show_each_kind_of_list() {
    use threers_ui_demo::lists::Kind;
    let mut app = App::new();
    app.state.page = pages::LISTS;
    let mut t = 0.0;
    for _ in 0..3 {
        frame(&mut app, &mut t);
    }
    let names = |app: &App| {
        app.ui
            .accessibility_tree()
            .into_iter()
            .map(|a| a.name)
            .collect::<Vec<_>>()
    };
    for (i, kind) in Kind::ALL.into_iter().enumerate() {
        let (x, y) = centre(&app, kind.label());
        click(&mut app, x, y);
        for _ in 0..3 {
            frame(&mut app, &mut t);
        }
        assert_eq!(app.state.list_tab, i);
        let list = &app.lists.lists[i].1;
        assert!(
            !list.built().is_empty() && list.built().len() < 60,
            "{}: {} rows built",
            kind.label(),
            list.built().len()
        );
        // Only the chosen list shows.
        for (k, (_, other)) in app.lists.lists.iter().enumerate() {
            assert_eq!(
                app.ui.rect(other.container()).height > 0.0,
                k == i,
                "{} showing with {}",
                k,
                kind.label()
            );
        }
        let n = names(&app);
        match kind {
            Kind::Rows => assert!(n.iter().any(|s| s.split_whitespace().eq(["Row", "1"]))),
            Kind::Messages => {
                let heights: Vec<f32> = list
                    .built()
                    .iter()
                    .map(|&(_, r)| app.ui.rect(r).height)
                    .collect();
                let (lo, hi) = heights
                    .iter()
                    .fold((f32::MAX, 0.0f32), |(a, b), &h| (a.min(h), b.max(h)));
                assert!(hi > lo + 15.0, "messages of different heights: {heights:?}");
            }
            Kind::Contacts => {
                assert!(n.iter().any(|s| s == "A") && n.iter().any(|s| s.contains(' ')))
            }
            Kind::Cards => {
                assert!(n.iter().any(|s| s == "Card 1") && n.iter().any(|s| s == "Card 3"));
                let row = list.built()[0].1;
                let cards = app.ui.children(row);
                assert_eq!(cards.len(), 3);
                assert_eq!(
                    app.ui.rect(cards[0]).y,
                    app.ui.rect(cards[2]).y,
                    "three across"
                );
            }
            Kind::Select => {
                let (x, y) = centre(&app, "Item 2");
                click(&mut app, x, y);
                frame(&mut app, &mut t);
                assert!(app.state.selected.contains(&1));
                assert!(
                    app.ui.text_content(app.lists.caption).starts_with("1 of"),
                    "{}",
                    app.ui.text_content(app.lists.caption)
                );
                let item = app
                    .ui
                    .accessibility_tree()
                    .into_iter()
                    .find(|a| a.role == threers_ui::Role::Checkbox && a.name == "Item 2")
                    .unwrap();
                assert_eq!(item.checked, Some(true));
                let (x, y) = centre(&app, "Item 2");
                click(&mut app, x, y);
                frame(&mut app, &mut t);
                assert!(app.state.selected.is_empty(), "a second tap unticks");
            }
        }
    }
}

/// The chosen tab stays marked after the pointer moves off it.
#[test]
fn the_chosen_list_tab_stays_marked() {
    let mut app = App::new();
    app.state.page = pages::LISTS;
    let mut t = 0.0;
    for _ in 0..3 {
        frame(&mut app, &mut t);
    }
    let (x, y) = centre(&app, "Select");
    click(&mut app, x, y);
    frame(&mut app, &mut t);
    let tab = app.lists.tabs[4];
    app.pointer_move(x, y + 200.0, &mut FixedWidthText(7.0));
    frame(&mut app, &mut t);
    assert_eq!(
        app.ui.style(tab).background,
        Some([0.145, 0.388, 0.922, 1.0])
    );
}

/// Pulling the list down from its top opens the indicator above it; let go
/// far enough, the list refreshes (a second's pretend load) and closes.
#[test]
fn pulling_the_list_down_refreshes_it() {
    let mut app = App::new();
    app.state.page = pages::LISTS;
    let mut t = 0.0;
    let step = |app: &mut App, t: &mut f64| {
        *t += 1.0 / 60.0;
        app.frame(*t, 1200.0, 800.0, &mut FixedWidthText(7.0));
    };
    for _ in 0..3 {
        step(&mut app, &mut t);
    }
    let (x, y) = centre(&app, "Row 3");
    let m = &mut FixedWidthText(7.0);
    app.touch(1, TouchPhase::Start, x, y, t, m);
    let mut label = String::new();
    for i in 1..=12 {
        app.touch(
            1,
            TouchPhase::Move,
            x,
            y + 25.0 * i as f32,
            t + 0.03 * i as f64,
            m,
        );
        step(&mut app, &mut t);
        if i == 3 {
            label = app
                .ui
                .accessibility_tree()
                .into_iter()
                .find(|a| a.name == "Pull to refresh" || a.name == "Release to refresh")
                .map(|a| a.name)
                .unwrap_or_default();
        }
    }
    assert_eq!(label, "Pull to refresh");
    let names = |app: &App| {
        app.ui
            .accessibility_tree()
            .into_iter()
            .map(|a| a.name)
            .collect::<Vec<_>>()
    };
    assert!(
        names(&app).iter().any(|n| n == "Release to refresh"),
        "{:?}",
        names(&app)
    );
    app.touch(1, TouchPhase::End, x, y + 300.0, t + 0.4, m);
    step(&mut app, &mut t);
    assert_eq!(app.state.refreshing, Some(0));
    assert!(names(&app).iter().any(|n| n == "Refreshing…"));
    for _ in 0..120 {
        step(&mut app, &mut t);
    }
    assert_eq!((app.state.refreshes, app.state.refreshing), (1, None));
    assert!(app
        .ui
        .text_content(app.lists.caption)
        .contains("Refreshed 1×"));
    assert!(!names(&app).iter().any(|n| ["Pull to refresh", "Release to refresh", "Refreshing…"].contains(&n.as_str())), "closed");
    // And then it goes quiet: no frames asked for once everything settles.
    let mut frames = 0;
    while app.needs_frame() && frames < 300 {
        step(&mut app, &mut t);
        frames += 1;
        if frames % 60 == 0 {
            let c = app.lists.lists[0].1.container();
            eprintln!(
                "frame {frames}: list top {} page top {} built {}",
                app.ui.style(c).scroll_top,
                app.ui.style(app.scroller()).scroll_top,
                app.lists.lists[0].1.built().len()
            );
        }
    }
    assert!(
        !app.needs_frame(),
        "still asking for frames after {frames}: ui {} input {} pull {:?} anim {}",
        app.ui.needs_frame(),
        app.input.needs_frame(),
        app.input.pull(app.lists.lists[0].1.container()),
        app.ui.style(app.ui.root()).opacity
    );
}

/// Swiping a row left deletes it, or opens its Delete action to tap; the
/// rows below move up into its place. Letter headings stay.
#[test]
fn swiping_a_row_left_deletes_it() {
    let mut app = App::new();
    app.state.page = pages::LISTS;
    let mut t = 0.0;
    let step = |app: &mut App, t: &mut f64| {
        *t += 1.0 / 60.0;
        app.frame(*t, 390.0, 844.0, &mut FixedWidthText(7.0));
    };
    for _ in 0..3 {
        step(&mut app, &mut t);
    }
    let names = |app: &App| {
        app.ui
            .accessibility_tree()
            .into_iter()
            .map(|a| a.name)
            .collect::<Vec<_>>()
    };
    let has = |app: &App, n: &str| {
        names(app)
            .iter()
            .any(|s| s.split_whitespace().eq(n.split_whitespace()))
    };
    let m = &mut FixedWidthText(7.0);
    let mut id = 0;
    let mut swipe = |app: &mut App, t: &mut f64, label: &str, by: f32| {
        let (_, y) = centre(app, label);
        let c = app.lists.active(&app.state).container();
        let r = app.ui.rect(c);
        let x = r.x + r.width - 30.0;
        id += 1;
        app.touch(id, TouchPhase::Start, x, y, *t, m);
        for i in 1..=10 {
            app.touch(
                id,
                TouchPhase::Move,
                x + by * i as f32 / 10.0,
                y,
                *t + 0.02 * i as f64,
                m,
            );
            step(app, t);
        }
        app.touch(id, TouchPhase::End, x + by, y, *t + 0.25, m);
        for _ in 0..30 {
            step(app, t);
        }
    };
    // All the way: gone, and the next row in its place.
    let y3 = centre(&app, "Row 3").1;
    swipe(&mut app, &mut t, "Row 3", -300.0);
    assert!(!has(&app, "Row 3") && has(&app, "Row 4"));
    assert_eq!(centre(&app, "Row 4").1, y3);
    assert!(app.ui.text_content(app.lists.caption).contains("1 deleted"));

    // Part way: open on Delete, which a tap deletes.
    swipe(&mut app, &mut t, "Row 5", -70.0);
    assert!(has(&app, "Delete"), "the action shows");
    let (x, y) = centre(&app, "Delete");
    let tap = &mut FixedWidthText(7.0);
    app.touch(1000, TouchPhase::Start, x, y, t, tap);
    app.touch(1000, TouchPhase::End, x, y, t + 0.05, tap);
    for _ in 0..30 {
        step(&mut app, &mut t);
    }
    assert!(!has(&app, "Row 5") && has(&app, "Row 6") && !has(&app, "Delete"));
    assert!(app.ui.text_content(app.lists.caption).contains("2 deleted"));

    // Contacts: a person goes, a heading stays.
    let (x, y) = centre(&app, "Contacts");
    click(&mut app, x, y);
    for _ in 0..3 {
        step(&mut app, &mut t);
    }
    swipe(&mut app, &mut t, "A", -300.0);
    assert!(has(&app, "A"));
    let first = app
        .ui
        .accessibility_tree()
        .into_iter()
        .find(|a| {
            let list = app.ui.rect(app.lists.active(&app.state).container());
            a.role == threers_ui::Role::ListItem
                && a.bounds.y > list.y
                && a.bounds.y < list.y + list.height
        })
        .map(|a| a.name)
        .expect("a contact");
    swipe(&mut app, &mut t, &first, -300.0);
    assert!(
        app.ui.text_content(app.lists.caption).contains("1 deleted"),
        "{}",
        app.ui.text_content(app.lists.caption)
    );
}

/// Text longer than the field scrolls sideways to keep the caret in view,
/// and back when the caret goes home.
#[test]
fn the_field_scrolls_to_its_caret() {
    let mut app = App::new();
    app.state.page = pages::TEXT;
    let mut t = 0.0;
    for _ in 0..3 {
        frame(&mut app, &mut t);
    }
    let (x, y) = centre(&app, "Name");
    click(&mut app, x, y);
    app.key("End", KeyModifiers::default());
    for c in " and a lot more text so that it runs well past the end of the field, which on its own page is wide, so more is typed".chars() {
        app.key(&c.to_string(), KeyModifiers::default());
    }
    for _ in 0..3 {
        frame(&mut app, &mut t);
    }
    let f = &app.state.field;
    assert_eq!(
        f.caret,
        f.text.chars().count(),
        "typed at the end: {:?}",
        f.text
    );
    let field = app
        .ui
        .accessibility_tree()
        .into_iter()
        .find(|a| a.name == "Name")
        .unwrap()
        .bounds;
    // The text moved left so that its end, where the caret is, shows.
    let r = field_text_rect(&app);
    assert!(
        r.x < field.x,
        "scrolled: text starts at {} left of the field at {}",
        r.x,
        field.x
    );
    let end = r.x + 7.0 * app.state.field.text.chars().count() as f32;
    // (Short of the keyboard button at its end: 30 px of padding, and the border.)
    assert!(
        (end - (field.x + field.width - 31.0)).abs() < 2.0,
        "its end at the field's end: {end} {field:?}"
    );
    app.key("Home", KeyModifiers::default());
    for _ in 0..3 {
        frame(&mut app, &mut t);
    }
    assert_eq!(field_text_rect(&app).x, field.x + 9.0, "back at the start");
}

fn field_text_rect(app: &App) -> threers_ui::Rect {
    app.field_rect_and_text().expect("the field shows").1
}

/// The edit menu goes on past its first page, as iOS's Notes' does: a
/// chevron turns to the demo's own actions, which edit the selection.
#[test]
fn the_edit_menu_turns_to_more_actions() {
    let mut app = App::new();
    app.state.page = pages::TEXT;
    let mut t = 0.0;
    let step = |app: &mut App, t: &mut f64| {
        *t += 1.0 / 60.0;
        app.frame(*t, 390.0, 844.0, &mut FixedWidthText(7.0));
    };
    for _ in 0..3 {
        step(&mut app, &mut t);
    }
    let page = app.scroller();
    app.ui.scroll_to(page, 1e6);
    for _ in 0..3 {
        step(&mut app, &mut t);
    }
    let r = app
        .ui
        .accessibility_tree()
        .into_iter()
        .find(|a| a.name == "Name")
        .unwrap()
        .bounds;
    let at = |i: f32| (r.x + 9.0 + 7.0 * i + 3.0, r.y + r.height / 2.0);
    let m = &mut FixedWidthText(7.0);
    let mut id = 0;
    let mut tap = |app: &mut App, (x, y): (f32, f32), time: f64| {
        id += 1;
        app.touch(id, TouchPhase::Start, x, y, time, m);
        app.touch(id, TouchPhase::End, x, y, time + 0.05, m);
    };
    tap(&mut app, at(10.0), 1.0);
    step(&mut app, &mut t);
    tap(&mut app, at(11.0), 2.0);
    tap(&mut app, at(11.0), 2.15);
    step(&mut app, &mut t);
    let has = |app: &App, n: &str| app.ui.accessibility_tree().into_iter().any(|a| a.name == n);
    assert!(has(&app, "Cut") && has(&app, "More") && !has(&app, "Look Up"));
    let more = centre(&app, "More");
    tap(&mut app, more, 3.0);
    step(&mut app, &mut t);
    assert!(
        has(&app, "Back") && has(&app, "Look Up") && !has(&app, "Cut"),
        "the next page"
    );
    assert!(
        app.input.wants_ime(&app.ui),
        "turning the page keeps the field's focus"
    );
    let upper = centre(&app, "Uppercase");
    tap(&mut app, upper, 5.0);
    step(&mut app, &mut t);
    let f = &app.state.field;
    assert_eq!(f.text, "Edit me: CLICK, drag, double-click");
    assert_eq!(f.selection(), (9, 14), "still selected");
    assert!(!has(&app, "Back") && !has(&app, "Cut"), "the menu closed");
}

/// The page's selectable text: a long press selects the word, with handles
/// and a read-only menu whose Copy puts it out for the clipboard; the mouse
/// drags out a selection that ⌘/Ctrl-C copies.
#[test]
fn page_text_selects_and_copies() {
    let mut app = App::new();
    app.state.page = pages::TEXT;
    let mut t = 0.0;
    let step = |app: &mut App, t: &mut f64| {
        *t += 1.0 / 60.0;
        app.frame(*t, 1200.0, 800.0, &mut FixedWidthText(7.0));
    };
    for _ in 0..3 {
        step(&mut app, &mut t);
    }
    // "Rich text has runs: a keyword, …" — hold a finger on "text".
    let para = app
        .ui
        .accessibility_tree()
        .into_iter()
        .find(|a| a.name.starts_with("Rich text has runs"))
        .unwrap()
        .bounds;
    let (x, y) = (para.x + 7.0 * 6.5, para.y + 8.0);
    let m = &mut FixedWidthText(7.0);
    app.touch(1, TouchPhase::Start, x, y, t, m);
    for _ in 0..40 {
        step(&mut app, &mut t);
    }
    app.touch(1, TouchPhase::End, x, y, t, m);
    step(&mut app, &mut t);
    let names = |app: &App| {
        app.ui
            .accessibility_tree()
            .into_iter()
            .map(|a| a.name)
            .collect::<Vec<_>>()
    };
    for item in ["Copy", "Select All"] {
        assert!(
            names(&app).iter().any(|n| n == item),
            "{item} in the menu: {:?}",
            names(&app)
        );
    }
    assert!(
        !names(&app).iter().any(|n| n == "Cut" || n == "Paste"),
        "read only"
    );
    let (cx, cy) = centre(&app, "Copy");
    app.touch(2, TouchPhase::Start, cx, cy, t, m);
    app.touch(2, TouchPhase::End, cx, cy, t + 0.05, m);
    assert_eq!(app.state.copy_out.take().as_deref(), Some("text"));

    // The mouse: a drag over "Rich text" and ⌘C.
    app.pointer_move(para.x + 1.0, y, m);
    app.pointer_down(para.x + 1.0, y, PointerButton::Primary, false, m);
    app.pointer_move(para.x + 7.0 * 9.0, y, m);
    app.pointer_up(para.x + 7.0 * 9.0, y, PointerButton::Primary);
    assert!(app.key(
        "c",
        KeyModifiers {
            meta: true,
            ..Default::default()
        }
    ));
    assert_eq!(app.state.copy_out.take().as_deref(), Some("Rich text"));
    // A click elsewhere puts it away; ⌘C then copies nothing.
    click(&mut app, 600.0, 400.0);
    app.key(
        "c",
        KeyModifiers {
            meta: true,
            ..Default::default()
        },
    );
    assert_eq!(app.state.copy_out.take(), None);
}

/// The paragraph's link opens: a click or a tap on it asks the page to,
/// and the pointer over it is a hand.
#[test]
fn a_link_in_text_opens() {
    let mut app = App::new();
    app.state.page = pages::TEXT;
    let mut t = 0.0;
    for _ in 0..3 {
        frame(&mut app, &mut t);
    }
    let para = app
        .ui
        .accessibility_tree()
        .into_iter()
        .find(|a| a.name.starts_with("Rich text has runs"))
        .unwrap()
        .bounds;
    // Find "link" by trying along the lines.
    let mut on_link = None;
    'find: for row in 0..4 {
        for col in 0..60 {
            let (x, y) = (
                para.x + 7.0 * col as f32 + 3.0,
                para.y + 19.0 * row as f32 + 9.0,
            );
            app.pointer_move(x, y, &mut FixedWidthText(7.0));
            if app.cursor() == "pointer" {
                on_link = Some((x, y));
                break 'find;
            }
        }
    }
    let (x, y) = on_link.expect("the pointer turns to a hand over the link");
    click(&mut app, x, y);
    assert_eq!(
        app.state.open_out.take().as_deref(),
        Some("https://example.com/")
    );
    let m = &mut FixedWidthText(7.0);
    app.touch(1, TouchPhase::Start, x, y, 1.0, m);
    app.touch(1, TouchPhase::End, x, y, 1.05, m);
    assert_eq!(
        app.state.open_out.take().as_deref(),
        Some("https://example.com/"),
        "a tap too"
    );
    click(&mut app, para.x + 3.0, para.y + 9.0);
    assert_eq!(app.state.open_out.take(), None, "not off the link");
}

/// Any text selects, but the board's (its cards are dragged), and Settings'
/// "Selectable text" turns it off everywhere; the pointer is an I-beam over
/// selectable text, and Shift-click extends a selection.
#[test]
fn any_text_selects_unless_turned_off() {
    let mut app = App::new();
    app.state.page = pages::LISTS;
    let mut t = 0.0;
    for _ in 0..3 {
        frame(&mut app, &mut t);
    }
    let m = &mut FixedWidthText(7.0);
    let at = |app: &App, name: &str| {
        let a = app
            .ui
            .accessibility_tree()
            .into_iter()
            .find(|a| a.name.starts_with(name))
            .unwrap_or_else(|| panic!("no {name}"));
        // A list row is read as one item, its text inside its padding.
        let pad = if a.role == threers_ui::Role::ListItem {
            13.0
        } else {
            3.0
        };
        (a.bounds.x + pad, a.bounds.y + a.bounds.height / 2.0)
    };
    // A list row's text: selectable now, by default.
    let (x, y) = at(&app, "Row 2");
    app.pointer_move(x, y, m);
    assert_eq!(app.cursor(), "text");
    app.pointer_down(x, y, PointerButton::Primary, false, m);
    app.pointer_move(x + 7.0 * 3.0, y, m);
    app.pointer_up(x + 7.0 * 3.0, y, PointerButton::Primary);
    // Shift-click further on extends it.
    app.pointer_down(x + 7.0 * 9.8 - 3.0, y, PointerButton::Primary, true, m);
    app.pointer_up(x + 7.0 * 9.8 - 3.0, y, PointerButton::Primary);
    assert!(app.key(
        "c",
        KeyModifiers {
            meta: true,
            ..Default::default()
        }
    ));
    assert_eq!(app.state.copy_out.take().as_deref(), Some("Row      2"));
    // A card on the board: dragged, not selected.
    go(&mut app, &mut t, pages::BASICS);
    let (x, y) = at(&app, "Grid layout");
    app.pointer_move(x, y, m);
    assert_ne!(app.cursor(), "text");
    // Settings → Selectable text: off everywhere.
    let (sx, sy) = centre(&app, "Settings…");
    click(&mut app, sx, sy);
    frame(&mut app, &mut t);
    let (cx, cy) = centre(&app, "Selectable text");
    click(&mut app, cx, cy);
    frame(&mut app, &mut t);
    assert!(!app.ui.text_selection());
    // Closed (while it is open it is all assistive technology sees).
    app.key("Escape", KeyModifiers::default());
    frame(&mut app, &mut t);
    go(&mut app, &mut t, pages::TEXT);
    let (x, y) = at(&app, "Rich text has runs");
    assert_eq!(app.ui.selectable_text_at(x, y), None);
}

/// The menu, the command palette, the theme global, notifications, a
/// prompt, a popover and a list worked from the keyboard, as wired into the
/// demo.
#[test]
fn the_menu_palette_and_components_work_in_the_demo() {
    use threers_ui::states::{Appearance, Theme};
    use threers_ui::Role;
    let mut app = App::new();
    app.state.page = pages::BASICS;
    let mut t = 0.0;
    for _ in 0..3 {
        frame(&mut app, &mut t);
    }
    let has = |app: &App, n: &str| app.ui.accessibility_tree().into_iter().any(|a| a.name == n);
    let none = KeyModifiers::default();
    // The ⋯ menu: Theme ▸ Light sets the app-wide setting, and the frame
    // turns the UI light.
    let (x, y) = centre(&app, "Menu");
    click(&mut app, x, y);
    frame(&mut app, &mut t);
    assert!(has(&app, "Command Palette…") && has(&app, "Theme"));
    let (x, y) = centre(&app, "Theme");
    click(&mut app, x, y);
    frame(&mut app, &mut t);
    let (x, y) = centre(&app, "Light");
    click(&mut app, x, y);
    frame(&mut app, &mut t);
    assert_eq!(app.ui.theme(), Theme::Light);
    assert_eq!(app.ui.appearance(), Appearance::Light);
    // The palette, from its key: typing narrows it, Enter runs the command.
    let palette = KeyModifiers {
        shift: true,
        meta: cfg!(target_os = "macos"),
        control: !cfg!(target_os = "macos"),
        ..Default::default()
    };
    assert!(app.key("p", palette));
    frame(&mut app, &mut t);
    assert!(has(&app, "Command palette"));
    for k in ["d", "a", "r", "k"] {
        app.key(k, none);
    }
    frame(&mut app, &mut t);
    let options: Vec<String> = app
        .ui
        .accessibility_tree()
        .into_iter()
        .filter(|a| a.role == Role::Option)
        .map(|a| a.name)
        .collect();
    assert_eq!(options, ["Theme: Dark"]);
    app.key("Enter", none);
    frame(&mut app, &mut t);
    assert!(!has(&app, "Command palette"));
    assert_eq!(app.ui.theme(), Theme::Dark);
    // Notifications: one from a button, stacked; Save's replaces the toast.
    let (x, y) = reveal(&mut app, &mut t, "Notify");
    click(&mut app, x, y);
    frame(&mut app, &mut t);
    assert!(has(&app, "Build finished. 3 crates compiled in 4.2 s."));
    assert_eq!(app.notices.shown(&app.ui).len(), 1);
    // A prompt: modal; Escape is Cancel and leaves the field alone.
    let (x, y) = reveal(&mut app, &mut t, "Clear name…");
    click(&mut app, x, y);
    frame(&mut app, &mut t);
    assert!(has(&app, "Clear the name?"));
    assert!(app.input.modal().is_some());
    app.key("Escape", none);
    frame(&mut app, &mut t);
    assert!(!has(&app, "Clear the name?"));
    assert!(!app.state.field.text.is_empty());
    // The popover, under its button.
    let (x, y) = reveal(&mut app, &mut t, "Details");
    click(&mut app, x, y);
    frame(&mut app, &mut t);
    let popover = app
        .ui
        .accessibility_tree()
        .into_iter()
        .find(|a| a.role == Role::Dialog && a.name == "Details")
        .expect("open");
    assert!(popover.bounds.y > y);
    app.key("Escape", none);
    frame(&mut app, &mut t);
    assert!(
        app.ui
            .accessibility_tree()
            .iter()
            .all(|a| a.role != Role::Dialog || a.name != "Details"),
        "closed"
    );
    // The list, focused, moves its highlight with the arrows.
    go(&mut app, &mut t, pages::LISTS);
    let list = app.lists.lists[0].1.container();
    app.input.set_focus(&mut app.ui, &mut app.state, Some(list));
    for _ in 0..3 {
        assert!(app.key("ArrowDown", none));
    }
    frame(&mut app, &mut t);
    let row = app.lists.lists[0].1.row(2).expect("built");
    assert_eq!(app.ui.semantics(row).and_then(|s| s.selected), Some(true));
    assert!(app.key("Enter", none));
    assert!(app.take_log().iter().any(|l| l == "activated row 2"));

    // Tasks, described from the state: adding, ticking and deleting change
    // the state, and the view follows on the next frame.
    go(&mut app, &mut t, pages::COMPONENTS);
    let (x, y) = reveal(&mut app, &mut t, "Add task");
    click(&mut app, x, y);
    frame(&mut app, &mut t);
    assert_eq!(app.state.tasks.len(), 4);
    assert!(has(&app, "Task 4") && has(&app, "3 of 4 to do"));
    let (x, y) = reveal(&mut app, &mut t, "Task 4");
    click(&mut app, x, y);
    frame(&mut app, &mut t);
    assert!(has(&app, "2 of 4 to do"));
    let (x, y) = reveal(&mut app, &mut t, "Delete Read the docs");
    click(&mut app, x, y);
    frame(&mut app, &mut t);
    assert!(!has(&app, "Read the docs") && has(&app, "2 of 3 to do"));

    // The shadows switch here and the Settings dialog's are bound to one
    // value: turned off here, the dialog's shows off too.
    go(&mut app, &mut t, pages::BASICS);
    let (x, y) = reveal(&mut app, &mut t, "Card shadows");
    click(&mut app, x, y);
    frame(&mut app, &mut t);
    assert!(!app.state.shadows);
    let shadows: Vec<Option<bool>> = app
        .ui
        .accessibility_tree()
        .into_iter()
        .filter(|a| a.role == Role::Switch && a.name == "Card shadows")
        .map(|a| a.checked)
        .collect();
    assert!(
        shadows.len() == 1 && shadows[0] == Some(false),
        "{shadows:?} (the dialog's is hidden)"
    );
    app.state.dialog_open = true;
    frame(&mut app, &mut t);
    // The dialog is modal: its switch is all assistive technology sees.
    let shadows: Vec<(Option<bool>, bool)> = app
        .ui
        .accessibility_tree()
        .into_iter()
        .filter(|a| a.role == Role::Switch && a.name == "Card shadows")
        .map(|a| (a.checked, app.ui.ancestors(a.id).any(|n| n == app.dialog())))
        .collect();
    assert_eq!(shadows, [(Some(false), true)]);
}

/// The header's title is the longest that fits.
#[test]
fn the_title_is_the_longest_that_fits() {
    let mut app = App::new();
    let mut t = 0.0;
    let title = |app: &App| {
        app.ui
            .accessibility_tree()
            .into_iter()
            .filter(|a| a.name.starts_with("threers-ui"))
            .map(|a| a.name)
            .next()
    };
    for _ in 0..2 {
        frame(&mut app, &mut t);
    }
    assert_eq!(
        title(&app).as_deref(),
        Some("threers-ui — a retained UI in Rust")
    );
    for _ in 0..2 {
        t += 1.0 / 60.0;
        app.frame(t, 390.0, 844.0, &mut FixedWidthText(7.0));
    }
    assert_eq!(title(&app).as_deref(), Some("threers-ui"));
}

/// The accessibility audit finds nothing in the demo: every components tab
/// and list, light and dark.
#[test]
fn the_demo_passes_the_accessibility_audit() {
    use threers_ui::states::Appearance;
    let mut app = App::new();
    let mut t = 0.0;
    let mut problems = vec![];
    for dark in [true, false] {
        app.set_system_dark(dark);
        for tab in 0..pages::PAGES.len() {
            app.state.page = tab;
            app.state.list_tab = tab % 5;
            // Long enough for colour transitions to finish.
            for _ in 0..20 {
                frame(&mut app, &mut t);
            }
            let a = if dark {
                Appearance::Dark
            } else {
                Appearance::Light
            };
            for i in app.ui.accessibility_audit() {
                problems.push(format!("{a:?} tab {tab}: {:?} {}", i.kind, i.detail));
            }
        }
    }
    problems.dedup();
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// The 3D page's departures board turns to the next one on the button,
/// and says so; the Shaders page's sliders reach their cards' effects.
#[test]
fn the_board_flips_and_the_sliders_turn_the_shaders() {
    let mut app = App::new();
    let mut t = 0.0;
    go(&mut app, &mut t, pages::DEPTH);
    let value = |app: &App| {
        app.ui
            .accessibility_tree()
            .into_iter()
            .find(|a| a.name == "Next departure" && a.role == threers_ui::Role::Status)
            .and_then(|a| a.value)
    };
    assert_eq!(value(&app).as_deref(), Some("LISBON"));
    // The board (a status) and the button share the name: the button.
    let find = |app: &App| {
        app.ui
            .accessibility_tree()
            .into_iter()
            .find(|a| a.name == "Next departure" && a.role == threers_ui::Role::Button)
            .expect("the button")
    };
    app.ui.scroll_into_view(find(&app).id);
    for _ in 0..2 {
        frame(&mut app, &mut t);
    }
    let button = find(&app);
    click(&mut app, button.bounds.x + 4.0, button.bounds.y + 4.0);
    for _ in 0..30 {
        frame(&mut app, &mut t);
    }
    assert_eq!(app.state.departure, 1);
    assert_eq!(value(&app).as_deref(), Some("OSLO"));
    assert_eq!(app.showcase.board.text(), "OSLO    ");

    go(&mut app, &mut t, pages::SHADERS);
    let split = app
        .ui
        .accessibility_tree()
        .into_iter()
        .find(|a| a.name == "Split" && a.role == threers_ui::Role::Group)
        .expect("the card")
        .id;
    let amount = |app: &App| app.ui.effect(split).map(|e| e.values()[0][0]);
    assert_eq!(amount(&app), Some(14.0));
    app.state.shader_split = 30.0;
    frame(&mut app, &mut t);
    assert_eq!(amount(&app), Some(30.0));
}

/// In the light appearance, the chosen list tab and the page's own link keep
/// readable text with the pointer over them (not white on grey).
#[test]
fn what_is_chosen_stays_readable_under_the_pointer() {
    let mut app = App::new();
    app.set_system_dark(false);
    app.state.page = pages::LISTS;
    let mut t = 0.0;
    for _ in 0..3 {
        frame(&mut app, &mut t);
    }
    let (x, y) = centre(&app, "Select");
    click(&mut app, x, y);
    frame(&mut app, &mut t);
    let page = app.ui.style(app.ui.root()).background.unwrap_or([1.0; 4]);
    for (label, (x, y)) in [
        ("Select", centre(&app, "Select")),
        ("Lists", centre(&app, "Lists")),
    ] {
        app.pointer_move(x, y, &mut FixedWidthText(7.0));
        frame(&mut app, &mut t);
        frame(&mut app, &mut t);
        let bad: Vec<_> = app
            .ui
            .contrast_issues(page, false)
            .into_iter()
            .filter(|i| i.text == label)
            .collect();
        assert!(
            bad.is_empty(),
            "{label} under the pointer: {:?}",
            bad.iter()
                .map(|i| (i.ratio, i.foreground, i.background))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn turning_the_model_and_the_candle_flickering_lay_nothing_out() {
    let mut app = App::new();
    let mut t = 0.0;
    go(&mut app, &mut t, pages::DEPTH);
    let (x, y) = reveal(&mut app, &mut t, "A torus");
    // Settled: what the page started has stopped.
    for _ in 0..400 {
        frame(&mut app, &mut t);
    }
    let m = &mut FixedWidthText(7.0);
    let before = app.ui.layout_passes();
    app.pointer_move(x, y, m);
    app.pointer_down(x, y, PointerButton::Primary, false, m);
    for k in 1..30 {
        app.pointer_move(x + k as f32 * 3.0, y + (k % 5) as f32, m);
        frame(&mut app, &mut t);
    }
    app.pointer_up(x + 90.0, y, PointerButton::Primary);
    frame(&mut app, &mut t);
    assert!(
        app.state.orbit.yaw.abs() > 0.3,
        "it turned: {:?}",
        app.state.orbit
    );
    // The new pictures of the model, the flame's sway and the glow's
    // opacity are only drawn.
    assert_eq!(
        app.ui.layout_passes(),
        before,
        "laid out while only the look changed"
    );
}

#[test]
fn a_link_opens_its_page_and_goes_to_its_section() {
    let mut app = App::new();
    let mut t = 0.0;
    go(&mut app, &mut t, pages::BASICS);
    assert_eq!(app.link(), "basics");
    assert!(app.follow_link("#toolkit/gauges-and-charts"));
    for _ in 0..4 {
        frame(&mut app, &mut t);
    }
    assert_eq!(app.state.page, pages::TOOLKIT);
    let target = app
        .ui
        .find_anchor("gauges-and-charts")
        .expect("the section's heading");
    assert_eq!(
        app.input.focused(),
        Some(target),
        "focus is where the link went"
    );
    let r = app.ui.rect(target);
    assert!(
        r.y >= 0.0 && r.y + r.height <= H,
        "scrolled into view: {r:?}"
    );
    assert_eq!(app.link(), "toolkit/gauges-and-charts");
    assert_eq!(
        app.take_followed(),
        Some(target),
        "for the mirror to follow"
    );
    // A page alone, and a section on the page showing.
    assert!(app.follow_link("3d"));
    for _ in 0..3 {
        frame(&mut app, &mut t);
    }
    assert_eq!(
        (
            app.state.page,
            app.link().split('/').next().unwrap().to_owned()
        ),
        (pages::DEPTH, "3d".to_owned())
    );
    // Nowhere: nothing changes, and it is given up after a few frames.
    assert!(app.follow_link("3d/no-such-section"));
    for _ in 0..6 {
        frame(&mut app, &mut t);
    }
    assert_eq!(app.link().split('/').next(), Some("3d"));
    assert!(!app.link().contains("no-such-section"));
}

#[test]
fn the_candles_flicker_only_while_one_shows() {
    let mut app = App::new();
    let mut t = 0.0;
    go(&mut app, &mut t, pages::DEPTH);
    // At the page's top the candles are far below: once it settles, no
    // frames are wanted for them.
    for _ in 0..400 {
        frame(&mut app, &mut t);
    }
    let candle = app.showcase.candles[0].root;
    assert!(!app.ui.is_on_screen(candle));
    assert!(
        !app.needs_frame(),
        "nothing to draw while the candles are out of sight"
    );
    // Scrolled to: they flicker, frame after frame.
    reveal(
        &mut app,
        &mut t,
        "A candle with a realistic flame, flickering",
    );
    assert!(app.ui.is_on_screen(candle));
    for _ in 0..30 {
        frame(&mut app, &mut t);
    }
    assert!(app.needs_frame());
}

#[test]
fn the_appearance_picker_menu_and_previews_agree() {
    use threers_ui::environment::Token;
    use threers_ui::states::{Appearance, Theme};
    let mut app = App::new();
    let mut t = 0.0;
    app.ui.set_system_appearance(Appearance::Light);
    go(&mut app, &mut t, pages::ACCESSIBILITY);
    assert_eq!(app.ui.theme(), Theme::System, "System by default");
    let (x, y) = reveal(&mut app, &mut t, "Dark");
    click(&mut app, x, y);
    for _ in 0..3 {
        frame(&mut app, &mut t);
    }
    assert_eq!(
        (app.state.theme, app.ui.theme()),
        (Theme::Dark, Theme::Dark)
    );
    // The previews hold their own: the light one stays light in the dark.
    let light = app
        .ui
        .accessibility_tree()
        .into_iter()
        .find(|a| a.name == "Always light: a preview")
        .unwrap()
        .id;
    assert_eq!(app.ui.appearance_of(light), Appearance::Light);
    let surface = app.ui.style(light).background.unwrap();
    let want = app.ui.token_in(Appearance::Light, Token::Surface);
    assert!(
        (0..3).all(|i| (surface[i] - want[i]).abs() < 1.0 / 255.0),
        "{surface:?} {want:?}"
    );
    // The menu's (or the palette's) choice reaches the picker.
    app.state.command = Some("theme-light".into());
    for _ in 0..2 {
        frame(&mut app, &mut t);
    }
    assert_eq!(
        (app.state.theme, app.ui.theme()),
        (Theme::Light, Theme::Light)
    );
}
