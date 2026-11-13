//! Rich text, shadows and
//! underlines, input methods, drag and drop, tooltips, element state,
//! surfaces and random trees.

use std::cell::Cell;
use std::rc::Rc;

use threers_ui::ime::Ime;
use threers_ui::input::{Event, Input};
use threers_ui::rich_text::TextRun;
use threers_ui::scene::Item;
use threers_ui::state::ElementState;
use threers_ui::testing::{random_tree, Rng};
use threers_ui::{
    style, FixedWidthText, FontWeight, NodeId, NodeKind, Role, Semantics, Style, TextField, TextMeasure, Ui,
};

/// Bold text is wider: measuring by run matters.
struct Weighted;
impl TextMeasure for Weighted {
    fn measure(&mut self, text: &str, s: &Style) -> (f32, f32) {
        let per = if threers_ui::TextStyle::of(s).weight() >= 600 { 10.0 } else { 6.0 };
        (text.chars().count() as f32 * per, s.font_size * 1.2)
    }
}

fn texts(ui: &Ui) -> Vec<(String, f32, f32)> {
    ui.scene()
        .items
        .into_iter()
        .filter_map(|i| match i {
            Item::Text { line, text, .. } => Some((text, line.x, line.y)),
            _ => None,
        })
        .collect()
}

#[test]
fn runs_are_measured_in_their_own_style_and_wrap_by_it() {
    let mut ui = Ui::new();
    let root = ui.root();
    let para = ui.add_box(root, style! { width: 95px; white-space: normal; font-size: 10px; });
    let t = ui.add_text(para, "aaaa bbbb cccc");
    // The middle word bold: 4×10 instead of 4×6 wide.
    ui.set_text_runs(t, vec![TextRun::new(5), TextRun::new(4).weight(FontWeight::Bold), TextRun::new(5)]);
    ui.layout(400.0, 300.0, &mut Weighted);
    let got = texts(&ui);
    // "aaaa " 30 + "bbbb" 40 + " cccc" 30 = 100 is past 95, so "cccc"
    // wraps — though in one style (6 a character, 84) all three fit.
    assert_eq!(got[0], ("aaaa ".into(), 0.0, 0.0));
    assert_eq!(got[1], ("bbbb".into(), 30.0, 0.0));
    assert_eq!(got[2].0, "cccc");
    assert!(got[2].2 > 0.0, "on the second line: {got:?}");
    ui.clear_text_runs(t);
    ui.layout(400.0, 300.0, &mut Weighted);
    assert_eq!(texts(&ui).len(), 1, "one style, one line");
}

#[test]
fn highlights_and_underlines_go_where_their_runs_are() {
    let mut ui = Ui::new();
    let root = ui.root();
    let t = ui.add_text(root, "see the typo");
    ui.set_text_runs(t, vec![
        TextRun::new(8).background([1.0, 1.0, 0.0, 1.0]),
        TextRun::new(4).underline(true),
    ]);
    let plain = ui.add_box(root, style! { text-decoration: underline wavy #f00; });
    ui.add_text(plain, "all of it");
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    let items = ui.scene().items;
    let hl = items.iter().find_map(|i| match i {
        Item::Rect { rect, color, .. } if color[2] == 0.0 && color[0] == 1.0 && color[1] == 1.0 => Some(*rect),
        _ => None,
    });
    assert_eq!(hl.map(|r| (r.x, r.width)), Some((0.0, 56.0)));
    let unders: Vec<_> = items
        .iter()
        .filter_map(|i| match i {
            Item::Underline { x, width, wavy, color, .. } => Some((*x, *width, *wavy, *color)),
            _ => None,
        })
        .collect();
    assert_eq!(unders.len(), 2);
    assert_eq!((unders[0].0, unders[0].1, unders[0].2), (56.0, 28.0, true));
    assert_eq!((unders[1].1, unders[1].2), (63.0, true), "the whole of the inherited one");
    assert_eq!(unders[1].3, [1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn box_shadows_parse_and_paint_under_the_box() {
    let s = threers_ui::style! { box-shadow: 0 2px 8px rgba(0,0,0,0.5), 1px 1px #f00; };
    assert_eq!(s.box_shadow.len(), 2);
    assert_eq!((s.box_shadow[0].y, s.box_shadow[0].blur), (2.0, 8.0));
    assert_eq!(s.box_shadow[1].color, [1.0, 0.0, 0.0, 1.0]);
    let mut ui = Ui::new();
    let root = ui.root();
    ui.add_box(root, style! { width: 50px; height: 20px; background: #fff; border-radius: 4px; box-shadow: 0 2px 8px 1px #000; });
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    let items = ui.scene().items;
    match (&items[0], &items[1]) {
        (Item::Shadow { rect, radius, blur, .. }, Item::Rect { .. }) => {
            assert_eq!((rect.x, rect.y, rect.width, rect.height), (-1.0, 1.0, 52.0, 22.0));
            assert_eq!((*radius, *blur), (5.0, 8.0));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn an_input_method_composes_in_the_focused_field() {
    let mut ui = Ui::new();
    let root = ui.root();
    let field = ui.add_box(root, style! { height: 20px; });
    ui.set_semantics(field, Semantics::new(Role::TextBox));
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    let mut input: Input<TextField> = Input::new();
    input.on(field, threers_ui::input::Phase::Bubble, |f, _, e, _| {
        if let Event::Ime(ime) = e {
            f.ime(ime);
        }
    });
    let mut text = TextField::new("");
    assert!(!input.wants_ime(&ui));
    input.click(&mut ui, &mut text, 5.0, 5.0);
    assert!(input.wants_ime(&ui));
    for (i, s) in ["k", "か", "かん", "漢"].iter().enumerate() {
        let ime = if i < 3 {
            Ime::Preedit { text: s.to_string(), cursor: Some((s.len(), s.len())) }
        } else {
            Ime::Commit(s.to_string())
        };
        input.dispatch(&mut ui, &mut text, Event::Ime(ime));
        if i == 2 {
            assert_eq!((text.text.as_str(), text.marked), ("かん", Some((0, 2))));
        }
    }
    assert_eq!((text.text.as_str(), text.marked, text.caret), ("漢", None, 1));
}

#[derive(Debug, Clone, PartialEq)]
struct Card(u32);

#[test]
fn a_card_drags_to_a_column_that_takes_it() {
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { display: flex; });
    let left = ui.add_box(root, style! { width: 100px; height: 200px; });
    let card = ui.add_box(left, style! { height: 30px; });
    let right = ui.add_box(root, style! { width: 100px; height: 200px; });
    let label = ui.add_text(right, "drop here");
    let preview = ui.add_box(root, style! { display: none; position: absolute; width: 40px; height: 10px; });
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    let mut input: Input<Vec<String>> = Input::new();
    input.draggable(card, |_, _| Some(Card(7)));
    input.on_drop::<Card>(right, |log, _, c, _| log.push(format!("dropped {}", c.0)));
    input.on_click(card, |log, _, _| log.push("click".into()));
    input.set_drag_preview(&mut ui, Some(preview));
    let mut log = vec![];

    // Mid-drag: what is carried, where it would land, the preview shown
    // and following the pointer, and the preview not in the way.
    let b = Default::default();
    input.dispatch(&mut ui, &mut log, Event::PointerDown { x: 10.0, y: 10.0, button: b });
    input.dispatch(&mut ui, &mut log, Event::PointerMove { x: 150.0, y: 5.0 });
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    input.dispatch(&mut ui, &mut log, Event::PointerMove { x: 151.0, y: 5.0 });
    assert_eq!(input.dragging().and_then(|d| d.downcast_ref::<Card>()), Some(&Card(7)));
    assert_eq!(input.drag_over(), Some(right), "the label is inside the column");
    assert_eq!(ui.style(preview).display, threers_ui::Display::Block);
    assert_eq!((ui.rect(preview).x, ui.rect(preview).y), (160.0, 15.0));
    let out = input.dispatch(&mut ui, &mut log, Event::PointerUp { x: 151.0, y: 5.0, button: b });
    assert_eq!(out.dropped, Some(right));
    assert_eq!(log, ["dropped 7"], "and no click");
    assert_eq!(ui.style(preview).display, threers_ui::Display::None);
    let _ = label;

    // Dropped where nothing takes it, or cancelled: nothing happens.
    log.clear();
    let out = input.drag(&mut ui, &mut log, (10.0, 10.0), (50.0, 100.0));
    assert_eq!(out.dropped, None);
    input.dispatch(&mut ui, &mut log, Event::PointerDown { x: 10.0, y: 10.0, button: b });
    input.dispatch(&mut ui, &mut log, Event::PointerMove { x: 150.0, y: 5.0 });
    input.press(&mut ui, &mut log, "escape");
    assert!(input.dragging().is_none());
    input.dispatch(&mut ui, &mut log, Event::PointerUp { x: 150.0, y: 5.0, button: b });
    assert!(log.is_empty(), "{log:?}");
    // A short press is still a click.
    input.click(&mut ui, &mut log, 10.0, 10.0);
    assert_eq!(log, ["click"]);
}

#[test]
fn a_tooltip_shows_after_a_rest_and_goes_on_a_press() {
    let mut ui = Ui::new();
    let root = ui.root();
    let button = ui.add_box(root, style! { width: 60px; height: 20px; margin: 0 0 0 360px; });
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    let built = Rc::new(Cell::new(0));
    let mut input: Input<()> = Input::new();
    let count = built.clone();
    input.tooltip(button, move |_, ui, root| {
        count.set(count.get() + 1);
        let tip = ui.add_box(root, style! { position: absolute; width: 80px; height: 18px; });
        ui.add_text(tip, "Render");
        tip
    });
    input.dispatch(&mut ui, &mut (), Event::PointerMove { x: 370.0, y: 10.0 });
    input.tick(&mut ui, &mut (), 1.0);
    input.tick(&mut ui, &mut (), 1.3);
    assert_eq!(input.tooltip_shown(), None, "not yet");
    input.tick(&mut ui, &mut (), 1.6);
    let tip = input.tooltip_shown().expect("shown");
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    // Below the button, pushed left to stay in the window.
    let r = ui.rect(tip);
    assert_eq!((r.x, r.y), (320.0, 26.0));
    assert_eq!(ui.deferred_priority(tip), Some(i32::MAX - 1));
    // A press takes it away, and it does not come back until the pointer
    // leaves and returns.
    input.click(&mut ui, &mut (), 370.0, 10.0);
    assert!(!ui.contains(tip));
    input.tick(&mut ui, &mut (), 5.0);
    assert_eq!(input.tooltip_shown(), None);
    input.dispatch(&mut ui, &mut (), Event::PointerMove { x: 10.0, y: 200.0 });
    input.tick(&mut ui, &mut (), 6.0);
    input.dispatch(&mut ui, &mut (), Event::PointerMove { x: 370.0, y: 10.0 });
    input.tick(&mut ui, &mut (), 7.0);
    input.tick(&mut ui, &mut (), 7.6);
    assert!(input.tooltip_shown().is_some());
    assert_eq!(built.get(), 2);
}

#[test]
fn element_state_outlives_rebuilt_nodes() {
    #[derive(Default)]
    struct Open(bool);
    let mut ui = Ui::new();
    let mut state = ElementState::new();
    for frame in 0..5 {
        let root = ui.root();
        ui.clear_children(root);
        for i in 0..3 {
            let open = state.get_or_default::<Open>(("section", i));
            if frame == 1 && i == 2 {
                open.0 = true;
            }
            let shown = open.0;
            ui.add_box(root, Style { display: if shown { threers_ui::Display::Block } else { threers_ui::Display::None }, ..Style::default() });
        }
        state.finish_frame();
    }
    assert!(state.get::<Open>(("section", 2)).unwrap().0);
    assert!(!state.get::<Open>(("section", 1)).unwrap().0);
    assert_eq!(state.len(), 3);
}

#[test]
fn surfaces_size_like_images_and_reach_the_scene() {
    let mut ui = Ui::new();
    let root = ui.root();
    let video = ui.add_surface(root, style! { width: 160px; }, 42, (1920, 1080));
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    assert_eq!(ui.rect(video).height, 90.0);
    assert!(ui.scene().items.iter().any(|i| matches!(i, Item::Surface { id: 42, .. })));
    assert!(matches!(ui.kind(video), NodeKind::Surface { width: 1920, .. }));
}

#[test]
fn random_trees_lay_out_and_hit_test_consistently() {
    for seed in 0..10 {
        let (mut ui, nodes) = random_tree(&mut Rng::new(seed), 150);
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        let mut rng = Rng::new(seed + 100);
        for _ in 0..300 {
            let (x, y) = (rng.range(-20.0, 420.0), rng.range(-20.0, 320.0));
            assert_eq!(ui.hit_test(x, y), ui.hit_test_walk(x, y), "seed {seed}");
        }
        assert_eq!(nodes.len(), 151);
        let _: Option<NodeId> = None;
    }
}

/// On a touch screen a finger on a card drags it; elsewhere it scrolls.
#[test]
fn a_finger_drags_a_card_and_pans_elsewhere() {
    use threers_ui::input::TouchPhase;
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { display: flex; });
    let left = ui.add_box(root, style! { width: 100px; height: 200px; });
    let card = ui.add_box(left, style! { height: 30px; });
    let right = ui.add_box(root, style! { width: 100px; height: 200px; overflow: auto; });
    ui.add_box(right, style! { height: 600px; });
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    let mut input: Input<Vec<String>> = Input::new();
    input.draggable(card, |_, _| Some(Card(1)));
    input.on_drop::<Card>(right, |log, _, c, _| log.push(format!("dropped {}", c.0)));
    let mut log = vec![];
    let t = |input: &mut Input<Vec<String>>, ui: &mut Ui, log: &mut Vec<String>, phase, x, y| {
        input.dispatch(ui, log, Event::Touch { id: 1, phase, x, y, time: 0.0 });
    };
    t(&mut input, &mut ui, &mut log, TouchPhase::Start, 20.0, 10.0);
    for k in 1..=6 {
        t(&mut input, &mut ui, &mut log, TouchPhase::Move, 20.0 + k as f32 * 25.0, 10.0);
    }
    assert!(input.dragging().is_some(), "the finger is dragging the card");
    t(&mut input, &mut ui, &mut log, TouchPhase::End, 170.0, 10.0);
    assert_eq!(log, ["dropped 1"]);
    // On the scrolling box, a finger scrolls.
    t(&mut input, &mut ui, &mut log, TouchPhase::Start, 150.0, 150.0);
    t(&mut input, &mut ui, &mut log, TouchPhase::Move, 150.0, 100.0);
    t(&mut input, &mut ui, &mut log, TouchPhase::End, 150.0, 100.0);
    assert_eq!(ui.style(right).scroll_top, 50.0);
}

/// Restyling by a container query or a state keeps where a box is scrolled.
#[test]
fn restyling_keeps_the_scroll_position() {
    let mut ui = Ui::new();
    let root = ui.root();
    let page = ui.add_box(root, style! { height: 100px; overflow: auto; });
    ui.add_box(page, style! { height: 500px; });
    ui.container_query(page, root, vec![(10_000.0, style! { height: 50px; overflow: auto; })]);
    ui.hover_style(page, "background: #222");
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    ui.style_mut(page).scroll_top = 120.0;
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    assert_eq!(ui.scroll_extent(page).2, 120.0, "the query left it scrolled");
    let mut input: Input<()> = Input::new();
    input.dispatch(&mut ui, &mut (), Event::PointerMove { x: 10.0, y: 10.0 });
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    assert!(ui.style(page).background.is_some(), "hovered");
    assert_eq!(ui.scroll_extent(page).2, 120.0, "so did hovering it");
}

/// A run can link somewhere: its boxes as laid out, and the link under a
/// point.
#[test]
fn text_runs_link_and_are_found_under_a_point() {
    use threers_ui::rich_text::TextRun;
    let mut ui = Ui::new();
    let root = ui.root();
    let p = ui.add_box(root, style! { width: 100px; white-space: normal; });
    let t = ui.add_text(p, "see the docs here");
    ui.set_text_runs(t, vec![TextRun::new(8), TextRun::new(4).link("https://example.com/docs"), TextRun::new(5)]);
    let m = &mut FixedWidthText(10.0);
    ui.layout(400.0, 300.0, m);
    let links = ui.link_rects(t, m);
    assert_eq!(links.len(), 1);
    let (r, target) = &links[0];
    // "docs" does not fit after "see the ": it starts the second line.
    assert_eq!((r.x, r.y > 10.0, r.width, target.as_str()), (0.0, true, 40.0, "https://example.com/docs"));
    assert_eq!(ui.link_at(&[t], 15.0, r.y + 5.0, m).as_deref(), Some("https://example.com/docs"));
    assert_eq!(ui.link_at(&[t], 30.0, 5.0, m), None);
}

/// Any text can be selected, as in a browser, but a control's; `user-select`
/// decides for a part of the UI, and the whole UI can turn it off.
#[test]
fn text_is_selectable_unless_switched_off() {
    use threers_ui::{Role, Semantics, UserSelect};
    let mut ui = Ui::new();
    let root = ui.root();
    let para = ui.add_box(root, style! { height: 20px; });
    ui.add_text(para, "plain text");
    let button = ui.add_box(root, style! { height: 20px; });
    ui.add_text(button, "a button");
    ui.set_semantics(button, Semantics::new(Role::Button));
    let off = ui.add_box(root, style! { height: 20px; user-select: none; });
    let inner = ui.add_box(off, Style::default());
    ui.add_text(inner, "turned off");
    let back = ui.add_box(off, style! { user-select: text; });
    ui.add_text(back, "back on");
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    let at = |ui: &Ui, n: NodeId| {
        let r = ui.rect(n);
        ui.selectable_text_at(r.x + 3.0, r.y + 5.0).is_some()
    };
    assert!(at(&ui, para), "any text");
    assert!(!at(&ui, button), "not a control's");
    assert!(!at(&ui, inner), "not under user-select: none");
    assert!(at(&ui, back), "unless turned back on inside");
    assert_eq!(ui.style(back).user_select, UserSelect::Text);
    ui.set_text_selection(false);
    assert!(!at(&ui, para) && !at(&ui, back), "off everywhere");
}

/// `clip-path` cuts a box and all it draws to a circle, an ellipse, a pill
/// or a rounded inset: a clip of that shape round its painting.
#[test]
fn clip_path_shapes_become_clips() {
    use threers_ui::scene::Item;
    use threers_ui::{style, FixedWidthText, Rect, Ui};
    for (css, want) in [
        ("circle()", (Rect { x: 20.0, y: 0.0, width: 60.0, height: 60.0 }, 30.0, 30.0)),
        ("ellipse()", (Rect { x: 0.0, y: 0.0, width: 100.0, height: 60.0 }, 50.0, 30.0)),
        ("capsule", (Rect { x: 0.0, y: 0.0, width: 100.0, height: 60.0 }, 30.0, 30.0)),
        ("inset(10px round 6px)", (Rect { x: 10.0, y: 10.0, width: 80.0, height: 40.0 }, 6.0, 6.0)),
    ] {
        let mut ui = Ui::new();
        let root = ui.root();
        let b = ui.add_box(root, style! { width: 100px; height: 60px; background: #336699; });
        ui.update_style(b, |s| threers_ui::css::apply(s, "clip-path", css));
        assert!(ui.style(b).clip_path.is_some(), "{css} parses");
        ui.layout(200.0, 100.0, &mut FixedWidthText(7.0));
        let scene = ui.scene();
        let clip = scene.items.iter().find_map(|i| match i {
            Item::PushClip { rect, rx, ry } => Some((*rect, *rx, *ry)),
            _ => None,
        });
        assert_eq!(clip, Some(want), "{css}");
        // The background inside it.
        let at = scene.items.iter().position(|i| matches!(i, Item::PushClip { .. })).unwrap();
        assert!(matches!(scene.items[at + 1], Item::Rect { .. }), "{css}: {:?}", scene.items);
    }
}

/// `linear-`, `radial-` and `conic-gradient` parse to their shapes and
/// stops, and place the stops where CSS does.
#[test]
fn gradients_of_three_shapes() {
    use threers_ui::{css, GradientShape, Rect, Style};
    let parse = |v: &str| {
        let mut s = Style::default();
        css::apply(&mut s, "background", v);
        (s.gradient_shape, s.background_gradient)
    };
    let (shape, stops) = parse("linear-gradient(to bottom, #ff0000, #0000ff 80%)");
    assert_eq!(shape, GradientShape::Linear(180.0));
    assert_eq!(stops.iter().map(|s| s.0).collect::<Vec<_>>(), [0.0, 0.8]);
    let (shape, stops) = parse("radial-gradient(circle at 25% 75%, red, rgb(0, 255, 0), blue)");
    assert_eq!(shape, GradientShape::Radial { cx: 0.25, cy: 0.75 });
    assert_eq!(stops.len(), 3);
    assert_eq!(stops[1], (0.5, [0.0, 1.0, 0.0, 1.0]));
    let (shape, _) = parse("conic-gradient(from 90deg at center, red, blue)");
    assert_eq!(shape, GradientShape::Conic { cx: 0.5, cy: 0.5, from: 90.0 });
    // Where the stops fall.
    let r = Rect { x: 0.0, y: 0.0, width: 100.0, height: 50.0 };
    let down = GradientShape::Linear(180.0);
    assert!((down.t(r, 50.0, 0.0) - 0.0).abs() < 1e-4 && (down.t(r, 50.0, 50.0) - 1.0).abs() < 1e-4);
    let ring = GradientShape::Radial { cx: 0.5, cy: 0.5 };
    assert!(ring.t(r, 50.0, 25.0).abs() < 1e-4 && (ring.t(r, 100.0, 50.0) - 1.0).abs() < 1e-4);
    let round = GradientShape::Conic { cx: 0.5, cy: 0.5, from: 0.0 };
    assert!((round.t(r, 100.0, 25.0) - 0.25).abs() < 1e-4, "a quarter turn at the right");
    assert!((round.t(r, 50.0, 50.0) - 0.5).abs() < 1e-4, "half at the bottom");
}
