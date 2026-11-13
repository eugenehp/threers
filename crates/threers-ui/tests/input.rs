//! Event dispatch, focus and actions, driven as a person would: pointer at
//! coordinates, keys by name.

use threers_ui::input::{Event, EventCx, Input, Phase, PointerButton};
use threers_ui::{style, FixedWidthText, NodeId, Role, Semantics, Ui};

#[derive(Default)]
struct Log(Vec<String>);

/// A panel holding two buttons and a slider, over a scrolling list.
struct Fixture {
    ui: Ui,
    panel: NodeId,
    ok: NodeId,
    cancel: NodeId,
    slider: NodeId,
    list: NodeId,
    outer: NodeId,
}

fn fixture() -> Fixture {
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { display: flex; flex-direction: column; });
    let panel = ui.add_box(root, style! { display: flex; height: 30px; });
    let ok = ui.add_box(panel, style! { width: 50px; height: 30px; });
    ui.set_semantics(ok, Semantics::new(Role::Button).label("OK"));
    ui.add_text(ok, "OK");
    let cancel = ui.add_box(panel, style! { width: 50px; height: 30px; });
    ui.set_semantics(cancel, Semantics::new(Role::Button).label("Cancel"));
    let slider = ui.add_box(panel, style! { width: 100px; height: 30px; });
    ui.set_semantics(slider, Semantics::new(Role::Slider).range(0.0, 1.0, 0.5));
    // A scrolling box inside a scrolling box.
    let outer = ui.add_box(root, style! { height: 100px; overflow: auto; });
    let list = ui.add_box(outer, style! { height: 60px; overflow: auto; });
    for _ in 0..10 {
        ui.add_box(list, style! { height: 20px; });
    }
    ui.add_box(outer, style! { height: 200px; });
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    Fixture { ui, panel, ok, cancel, slider, list, outer }
}

fn logger(name: &'static str) -> impl FnMut(&mut Log, &mut Ui, &Event, &mut EventCx) {
    move |log, _, e, cx| {
        let what = match e {
            Event::PointerDown { .. } => "down",
            Event::PointerUp { .. } => "up",
            Event::Click { .. } => "click",
            Event::Focus => "focus",
            Event::Blur => "blur",
            Event::PointerEnter => "enter",
            Event::PointerLeave => "leave",
            Event::Key { .. } => "key",
            Event::Text(_) => "text",
            _ => return,
        };
        let phase = match cx.phase {
            Phase::Capture => "capture",
            Phase::Bubble => "bubble",
        };
        log.0.push(format!("{name} {what} {phase}"));
    }
}

#[test]
fn capture_runs_inward_then_bubble_outward() {
    let mut f = fixture();
    let mut input = Input::new();
    let root = f.ui.root();
    for (node, name) in [(root, "root"), (f.panel, "panel"), (f.ok, "ok")] {
        input.on(node, Phase::Capture, logger(name));
        input.on(node, Phase::Bubble, logger(name));
    }
    let mut log = Log::default();
    input.dispatch(&mut f.ui, &mut log, Event::PointerDown { x: 5.0, y: 5.0, button: PointerButton::Primary });
    let downs: Vec<_> = log.0.iter().filter(|l| l.contains("down")).cloned().collect();
    assert_eq!(
        downs,
        [
            "root down capture",
            "panel down capture",
            "ok down capture",
            "ok down bubble",
            "panel down bubble",
            "root down bubble"
        ]
    );
}

#[test]
fn stopping_an_event_keeps_it_from_the_nodes_further_on() {
    let mut f = fixture();
    let mut input = Input::new();
    input.on(f.panel, Phase::Capture, |_: &mut Log, _, _, cx| cx.stop_propagation());
    input.on(f.ok, Phase::Bubble, logger("ok"));
    let mut log = Log::default();
    let out = input.click(&mut f.ui, &mut log, 5.0, 5.0);
    // Entering is told to the node itself, so no ancestor can stop it.
    log.0.retain(|l| !l.contains("enter"));
    assert!(log.0.is_empty(), "{:?}", log.0);
    assert!(out.handled);
}

#[test]
fn a_click_lands_on_what_holds_both_press_and_release() {
    let mut f = fixture();
    let mut input = Input::new();
    let mut log = Log::default();
    let primary = PointerButton::Primary;
    // Pressed on OK, released on Cancel: the click is the panel's.
    input.dispatch(&mut f.ui, &mut log, Event::PointerDown { x: 5.0, y: 5.0, button: primary });
    let out = input.dispatch(&mut f.ui, &mut log, Event::PointerUp { x: 55.0, y: 5.0, button: primary });
    assert_eq!(out.clicked, Some(f.panel));
    // Pressed and released on OK's label: the click is the label's, and
    // reaches OK as it bubbles.
    let out = input.click(&mut f.ui, &mut log, 5.0, 5.0);
    let clicked = out.clicked.unwrap();
    assert_eq!(f.ui.parent(clicked), Some(f.ok));
    // The other button released: no click.
    input.dispatch(&mut f.ui, &mut log, Event::PointerDown { x: 5.0, y: 5.0, button: primary });
    let out = input.dispatch(
        &mut f.ui,
        &mut log,
        Event::PointerUp { x: 5.0, y: 5.0, button: PointerButton::Secondary },
    );
    assert_eq!(out.clicked, None);
}

#[test]
fn a_captured_pointer_keeps_reaching_the_slider_off_its_edge() {
    let mut f = fixture();
    let mut input: Input<Vec<f32>> = Input::new();
    let slider = f.slider;
    input.on(slider, Phase::Bubble, |seen, ui, e, cx| match e {
        Event::PointerDown { .. } => cx.capture_pointer(),
        Event::PointerMove { x, .. } => {
            let r = ui.rect(cx.current);
            seen.push(((x - r.x) / r.width).clamp(0.0, 1.0));
        }
        _ => {}
    });
    let mut seen = vec![];
    let b = PointerButton::Primary;
    input.dispatch(&mut f.ui, &mut seen, Event::PointerDown { x: 150.0, y: 10.0, button: b });
    input.dispatch(&mut f.ui, &mut seen, Event::PointerMove { x: 175.0, y: 10.0 });
    // Far outside it, and over another node entirely.
    input.dispatch(&mut f.ui, &mut seen, Event::PointerMove { x: 380.0, y: 200.0 });
    input.dispatch(&mut f.ui, &mut seen, Event::PointerUp { x: 380.0, y: 200.0, button: b });
    // Released: moves go back to whatever is under the pointer.
    input.dispatch(&mut f.ui, &mut seen, Event::PointerMove { x: 120.0, y: 10.0 });
    assert_eq!(seen, [0.75, 1.0, 0.2]);
}

#[test]
fn pressing_focuses_what_can_take_focus_and_pressing_elsewhere_blurs() {
    let mut f = fixture();
    let mut input = Input::new();
    input.on(f.ok, Phase::Bubble, logger("ok"));
    let mut log = Log::default();
    // On the text inside the button: the button takes focus.
    input.click(&mut f.ui, &mut log, 3.0, 3.0);
    assert_eq!(input.focused(), Some(f.ok));
    input.click(&mut f.ui, &mut log, 300.0, 250.0);
    assert_eq!(input.focused(), None);
    assert!(log.0.contains(&"ok focus bubble".to_string()));
    assert!(log.0.contains(&"ok blur bubble".to_string()));
}

#[test]
fn preventing_the_default_of_a_press_keeps_focus_where_it_was() {
    let mut f = fixture();
    let mut input = Input::new();
    input.on(f.cancel, Phase::Capture, |_: &mut (), _, e, cx| {
        if let Event::PointerDown { .. } = e {
            cx.prevent_default();
        }
    });
    input.click(&mut f.ui, &mut (), 5.0, 5.0);
    input.click(&mut f.ui, &mut (), 55.0, 5.0);
    assert_eq!(input.focused(), Some(f.ok));
}

#[test]
fn tab_cycles_and_enter_and_space_activate_like_a_page() {
    let mut f = fixture();
    let mut input = Input::new();
    let mut clicks: Vec<NodeId> = vec![];
    for b in [f.ok, f.cancel, f.slider] {
        input.on_click(b, move |c: &mut Vec<NodeId>, _, _| c.push(b));
    }
    input.press(&mut f.ui, &mut clicks, "tab");
    assert_eq!(input.focused(), Some(f.ok));
    input.press(&mut f.ui, &mut clicks, "tab tab");
    assert_eq!(input.focused(), Some(f.slider));
    input.press(&mut f.ui, &mut clicks, "tab");
    assert_eq!(input.focused(), Some(f.ok), "wraps round");
    input.press(&mut f.ui, &mut clicks, "shift-tab");
    assert_eq!(input.focused(), Some(f.slider));
    // A slider is not activated by Enter or Space; buttons are by both.
    input.press(&mut f.ui, &mut clicks, "enter space");
    input.press(&mut f.ui, &mut clicks, "tab enter space");
    assert_eq!(clicks, [f.ok, f.ok]);
}

#[test]
fn the_wheel_scrolls_the_innermost_box_then_chains_outward() {
    let mut f = fixture();
    let mut input = Input::new();
    // Notches at once, for this test of chaining (they animate otherwise).
    input.smooth_wheel = false;
    let (list, outer) = (f.list, f.outer);
    // Over the inner list: it scrolls first, to its end (200 − 60 = 140).
    let out = input.scroll(&mut f.ui, &mut (), 10.0, 40.0, 100.0);
    assert_eq!(out.scrolled, Some(list));
    f.ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    input.scroll(&mut f.ui, &mut (), 10.0, 40.0, 100.0);
    f.ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    assert_eq!(f.ui.scroll_extent(list).2, 140.0);
    // At its end, the next turn moves the box around it.
    let out = input.scroll(&mut f.ui, &mut (), 10.0, 40.0, 30.0);
    assert_eq!(out.scrolled, Some(outer));
}

#[test]
fn the_pointer_entering_and_leaving_tells_only_those_nodes() {
    let mut f = fixture();
    let mut input = Input::new();
    input.on(f.ok, Phase::Bubble, logger("ok"));
    input.on(f.panel, Phase::Bubble, logger("panel"));
    let mut log = Log::default();
    input.dispatch(&mut f.ui, &mut log, Event::PointerMove { x: 5.0, y: 5.0 });
    input.dispatch(&mut f.ui, &mut log, Event::PointerMove { x: 55.0, y: 5.0 });
    assert!(input.is_hovered(f.panel) && !input.is_hovered(f.ok));
    input.dispatch(&mut f.ui, &mut log, Event::PointerMove { x: 300.0, y: 250.0 });
    let hover: Vec<_> = log.0.iter().filter(|l| l.contains("enter") || l.contains("leave")).collect();
    assert_eq!(
        hover,
        ["panel enter bubble", "ok enter bubble", "ok leave bubble", "panel leave bubble"]
    );
}

#[derive(Debug, Clone, PartialEq)]
struct Save;
#[derive(Debug, Clone, PartialEq)]
struct Confirm(u8);

#[test]
fn keys_become_actions_handled_nearest_the_focus() {
    let mut f = fixture();
    let mut input: Input<Vec<String>> = Input::new();
    input.keymap.bind("ctrl-s", Save, None).unwrap();
    input.keymap.bind("enter", Confirm(1), None).unwrap();
    input.keymap.bind("enter", Confirm(2), Some("Dialog")).unwrap();
    input.keymap.bind("ctrl-k ctrl-s", Confirm(9), None).unwrap();
    let root = f.ui.root();
    input.set_key_context(f.panel, "Dialog");
    input.on_action(root, |log, _, _: &Save, _| log.push("root saved".into()));
    input.on_action(f.panel, |log, _, a: &Confirm, cx| {
        log.push(format!("panel confirm {}", a.0));
        cx.propagate();
    });
    input.on_action(root, |log, _, a: &Confirm, _| log.push(format!("root confirm {}", a.0)));
    input.on_key(root, |log, _, key, _, _| {
        log.push(format!("key {key}"));
        true
    });

    let mut log = vec![];
    // Nothing focused: the root's context, so the binding for everywhere.
    input.press(&mut f.ui, &mut log, "enter");
    // Focus in the dialog: its binding, its handler, then (propagated) the root's.
    input.click(&mut f.ui, &mut log, 55.0, 5.0);
    input.press(&mut f.ui, &mut log, "enter ctrl-s");
    // A chord, and a key nothing binds.
    input.press(&mut f.ui, &mut log, "ctrl-k ctrl-s x");
    assert_eq!(
        log,
        [
            "root confirm 1",
            "panel confirm 2",
            "root confirm 2",
            "root saved",
            "panel confirm 9",
            "root confirm 9",
            "key x"
        ]
    );
    let shown = input.keymap.keys_for::<Confirm>(&["Dialog"]).unwrap();
    assert_eq!(shown[0].to_string(), "enter");
}

#[test]
fn an_unhandled_action_falls_through_to_the_key() {
    let mut f = fixture();
    let mut input = Input::new();
    input.keymap.bind("enter", Save, None).unwrap();
    let mut clicks = 0;
    input.on_click(f.ok, |c: &mut i32, _, _| *c += 1);
    input.press(&mut f.ui, &mut clicks, "tab");
    let out = input.press(&mut f.ui, &mut clicks, "enter");
    assert!(out.action.is_some());
    assert_eq!(clicks, 1, "nobody handles Save, so Enter still activates the button");
}

#[test]
fn removing_a_node_drops_its_listeners_and_its_focus() {
    let mut f = fixture();
    let mut input = Input::new();
    input.on_click(f.ok, |c: &mut i32, _, _| *c += 1);
    let mut clicks = 0;
    input.click(&mut f.ui, &mut clicks, 5.0, 5.0);
    assert_eq!(input.focused(), Some(f.ok));
    f.ui.remove(f.ok);
    // The slot is taken by a new node in the same place, which must not
    // inherit the old one's listener or focus.
    let fresh = f.ui.add_box(f.panel, style! { width: 50px; height: 30px; });
    f.ui.move_child(f.panel, fresh, 0);
    f.ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    input.click(&mut f.ui, &mut clicks, 5.0, 5.0);
    assert_eq!(clicks, 1);
    assert_eq!(input.focused(), None);
}

#[test]
fn text_and_keys_go_to_the_focused_field() {
    let mut f = fixture();
    let mut input: Input<String> = Input::new();
    let field = f.ui.add_box(f.ui.root(), style! { height: 20px; });
    f.ui.set_semantics(field, Semantics::new(Role::TextBox));
    f.ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    input.on(field, Phase::Bubble, |text, _, e, _| match e {
        Event::Key { key, .. } if key.chars().count() == 1 => text.push_str(key),
        Event::Text(t) => text.push_str(t),
        _ => {}
    });
    let mut text = String::new();
    input.type_text(&mut f.ui, &mut text, "lost");
    assert_eq!(text, "", "nothing focused");
    f.ui.style_mut(field);
    input.set_focus(&mut f.ui, &mut text, Some(field));
    input.type_text(&mut f.ui, &mut text, "Hi");
    input.dispatch(&mut f.ui, &mut text, Event::Text("ñ".into()));
    assert_eq!(text, "Hiñ");
}

/// A mouse wheel's notches glide there, and keep adding up; a trackpad's
/// small steps move at once; a flung finger keeps the content going.
#[test]
fn notches_animate_and_flings_glide() {
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(root, style! { height: 100px; overflow: auto; });
    ui.add_box(list, style! { height: 5000px; });
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    let mut input: Input<()> = Input::new();
    let (mut t, dt) = (0.0, 1.0 / 60.0);
    input.tick(&mut ui, &mut (), t);
    let notch = |input: &mut Input<()>, ui: &mut Ui| {
        input.dispatch(ui, &mut (), Event::Wheel { x: 10.0, y: 10.0, dx: 0.0, dy: 100.0 });
    };
    notch(&mut input, &mut ui);
    notch(&mut input, &mut ui);
    let mut seen = vec![];
    for _ in 0..20 {
        t += dt;
        input.tick(&mut ui, &mut (), t);
        ui.step(dt as f32);
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        seen.push(ui.style(list).scroll_top);
    }
    assert!(seen[0] > 0.0 && seen[0] < 100.0, "a notch is not a jump: {seen:?}");
    assert!(seen.windows(2).all(|w| w[1] >= w[0]), "it only goes forward");
    assert_eq!(*seen.last().unwrap(), 200.0, "two notches add up");
    // A trackpad's small step moves at once.
    input.dispatch(&mut ui, &mut (), Event::Wheel { x: 10.0, y: 10.0, dx: 0.0, dy: 7.5 });
    assert_eq!(ui.style(list).scroll_top, 207.5);

    // A finger flicks up and lets go: the content keeps moving, slowing.
    use threers_ui::input::TouchPhase;
    let touch = |input: &mut Input<()>, ui: &mut Ui, phase, y: f32, t: f64| {
        input.dispatch(ui, &mut (), Event::Touch { id: 1, phase, x: 10.0, y, time: t });
    };
    touch(&mut input, &mut ui, TouchPhase::Start, 90.0, t);
    for k in 1..=5 {
        touch(&mut input, &mut ui, TouchPhase::Move, 90.0 - k as f32 * 15.0, t + k as f64 * dt);
    }
    t += 5.0 * dt;
    touch(&mut input, &mut ui, TouchPhase::End, 15.0, t);
    let lifted = ui.style(list).scroll_top;
    assert!(input.needs_frame(), "gliding");
    let mut steps = vec![];
    for _ in 0..240 {
        t += dt;
        input.tick(&mut ui, &mut (), t);
        steps.push(ui.style(list).scroll_top);
    }
    assert!(steps[0] > lifted, "it carries on");
    let gained: Vec<f32> = steps.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(gained[1] < gained[0] * 1.001 && gained[60] < gained[1], "and slows");
    assert!(!input.needs_frame(), "until it stops");
}

/// A flick's glide goes as far as a phone's: its release speed over the
/// friction, measured from the last of its movement, not the whole swipe.
#[test]
fn a_flick_glides_as_far_as_a_phone_would() {
    use threers_ui::input::TouchPhase;
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(root, style! { height: 300px; overflow: auto; });
    ui.add_box(list, style! { height: 100000px; });
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    let mut input: Input<()> = Input::new();
    let dt = 1.0 / 60.0;
    let mut t = 0.0;
    input.tick(&mut ui, &mut (), t);
    let touch = |input: &mut Input<()>, ui: &mut Ui, phase, y: f32, t: f64| {
        input.dispatch(ui, &mut (), Event::Touch { id: 1, phase, x: 10.0, y, time: t });
    };
    // A slow start, then 10 px a frame (600 px/s) for the last 100 ms.
    let mut y = 290.0;
    touch(&mut input, &mut ui, TouchPhase::Start, y, t);
    for k in 0..20 {
        t += dt;
        y -= if k < 14 { 2.0 } else { 10.0 };
        touch(&mut input, &mut ui, TouchPhase::Move, y, t);
    }
    touch(&mut input, &mut ui, TouchPhase::End, y, t);
    let lifted = ui.style(list).scroll_top;
    for _ in 0..600 {
        t += dt;
        input.tick(&mut ui, &mut (), t);
    }
    let glide = ui.style(list).scroll_top - lifted;
    // 600 px/s over a friction of 2: about 300 px (less the tail cut at 20 px/s).
    assert!((260.0..=310.0).contains(&glide), "glided {glide} px");

    // A second flick while it still glides adds to it.
    let fast = |input: &mut Input<()>, ui: &mut Ui, t: &mut f64| {
        let mut y = 290.0;
        touch(input, ui, TouchPhase::Start, y, *t);
        for _ in 0..6 {
            *t += dt;
            y -= 30.0;
            touch(input, ui, TouchPhase::Move, y, *t);
        }
        touch(input, ui, TouchPhase::End, y, *t);
    };
    fast(&mut input, &mut ui, &mut t);
    let a = ui.style(list).scroll_top;
    for _ in 0..10 {
        t += dt;
        input.tick(&mut ui, &mut (), t);
    }
    let one = ui.style(list).scroll_top - a;
    fast(&mut input, &mut ui, &mut t);
    let b = ui.style(list).scroll_top;
    for _ in 0..10 {
        t += dt;
        input.tick(&mut ui, &mut (), t);
    }
    let two = ui.style(list).scroll_top - b;
    assert!(two > one * 1.4, "the second flick goes faster: {one} then {two}");
}

/// A finger hovers only while it is down: after a tap the hover style is
/// gone, so the style the tap set shows.
#[test]
fn a_lifted_finger_hovers_nothing() {
    use threers_ui::input::TouchPhase;
    let mut ui = Ui::new();
    let root = ui.root();
    let b = ui.add_box(root, style! { width: 50px; height: 50px; background: #000000; });
    ui.hover_style(b, "background: #ff0000");
    ui.layout(200.0, 200.0, &mut FixedWidthText(7.0));
    let mut input: Input<()> = Input::new();
    input.dispatch(&mut ui, &mut (), Event::Touch { id: 1, phase: TouchPhase::Start, x: 10.0, y: 10.0, time: 0.0 });
    assert_eq!(input.hovered(), Some(b));
    input.dispatch(&mut ui, &mut (), Event::Touch { id: 1, phase: TouchPhase::End, x: 10.0, y: 10.0, time: 0.05 });
    assert_eq!(input.hovered(), None);
    ui.layout(200.0, 200.0, &mut FixedWidthText(7.0));
    assert_eq!(ui.style(b).background, Some([0.0, 0.0, 0.0, 1.0]));
}

/// A strip that scrolls sideways inside a page that scrolls down: a pan
/// that starts sideways keeps to the strip however it drifts, and a flick
/// glides on after the finger lifts.
#[test]
fn a_sideways_pan_keeps_to_its_strip_and_glides() {
    use threers_ui::input::TouchPhase;
    let mut ui = Ui::new();
    let root = ui.root();
    let page = ui.add_box(root, style! { height: 300px; overflow: auto; });
    let strip = ui.add_box(page, style! { display: flex; overflow-x: auto; width: 300px; });
    for _ in 0..40 {
        ui.add_box(strip, style! { width: 60px; height: 40px; flex-shrink: 0; });
    }
    ui.add_box(page, style! { height: 2000px; });
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    let mut input: Input<()> = Input::new();
    let (mut t, dt) = (0.0, 1.0 / 60.0);
    input.tick(&mut ui, &mut (), t);
    let touch = |input: &mut Input<()>, ui: &mut Ui, phase, x: f32, y: f32, t: f64| {
        input.dispatch(ui, &mut (), Event::Touch { id: 1, phase, x, y, time: t });
    };
    touch(&mut input, &mut ui, TouchPhase::Start, 250.0, 20.0, t);
    // Mostly left, drifting down as a thumb does.
    for k in 1..=6 {
        t += dt;
        touch(&mut input, &mut ui, TouchPhase::Move, 250.0 - k as f32 * 30.0, 20.0 + k as f32 * 8.0, t);
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    }
    assert_eq!(ui.style(strip).scroll_left, 180.0, "every pixel of the finger");
    assert_eq!(ui.style(page).scroll_top, 0.0, "the page stays put");
    touch(&mut input, &mut ui, TouchPhase::End, 70.0, 68.0, t);
    let lifted = ui.style(strip).scroll_left;
    for _ in 0..60 {
        t += dt;
        input.tick(&mut ui, &mut (), t);
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    }
    let glided = ui.style(strip).scroll_left - lifted;
    assert!(glided > 300.0, "it glides on: {glided}");
    // A pan that starts down scrolls the page, not the strip.
    let before = ui.style(strip).scroll_left;
    touch(&mut input, &mut ui, TouchPhase::Start, 200.0, 20.0, t);
    for k in 1..=4 {
        t += dt;
        touch(&mut input, &mut ui, TouchPhase::Move, 200.0 - k as f32 * 4.0, 20.0 - k as f32 * 20.0, t);
    }
    touch(&mut input, &mut ui, TouchPhase::End, 184.0, -60.0, t + 0.3);
    assert_eq!(ui.style(strip).scroll_left, before);
    assert!(ui.style(page).scroll_top > 0.0);
}
