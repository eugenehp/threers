//! What a host gets without asking: animations moved on by the input's
//! tick, and swipes heard by [`Input::on_swipe`].

use threers_ui::animation::{Animation, Property, Value};
use threers_ui::input::{Event, Input, TouchPhase, Toward};
use threers_ui::{style, FixedWidthText, Ui};

fn layout(ui: &mut Ui) {
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
}

/// A box fading in over a second.
fn fading() -> (Ui, threers_ui::NodeId) {
    let mut ui = Ui::new();
    let root = ui.root();
    let b = ui.add_box(root, style! { width: 50px; height: 50px; });
    ui.animate(b, Animation::new(Property::Opacity, Value::Number(0.0), Value::Number(1.0), 1.0));
    layout(&mut ui);
    (ui, b)
}

#[test]
fn the_tick_moves_animations_on() {
    let (mut ui, b) = fading();
    let mut input: Input<()> = Input::new();
    input.tick(&mut ui, &mut (), 0.0);
    input.tick(&mut ui, &mut (), 0.5);
    let half = ui.style(b).opacity;
    assert!((half - 0.5).abs() < 0.2, "half way: {half}");
    input.tick(&mut ui, &mut (), 1.5);
    assert_eq!(ui.style(b).opacity, 1.0);
}

#[test]
fn a_host_that_steps_them_is_not_stepped_twice() {
    let (mut ui, b) = fading();
    let (mut stepped, _) = fading();
    let mut input: Input<()> = Input::new();
    input.tick(&mut ui, &mut (), 0.0);
    for k in 1..=4 {
        ui.step(0.1);
        input.tick(&mut ui, &mut (), k as f64 * 0.1);
        stepped.step(0.1);
    }
    assert_eq!(ui.style(b).opacity, stepped.style(b).opacity);
}

/// A finger from `(x, y)` by `(dx, dy)`, in ten moves.
fn drag(ui: &mut Ui, input: &mut Input<u32>, n: &mut u32, (x, y): (f32, f32), (dx, dy): (f32, f32)) {
    let mut at = |ui: &mut Ui, phase, k: f32, time: f64| {
        input.dispatch(ui, n, Event::Touch { id: 1, phase, x: x + dx * k, y: y + dy * k, time });
    };
    at(ui, TouchPhase::Start, 0.0, 0.0);
    for i in 1..=10 {
        at(ui, TouchPhase::Move, i as f32 / 10.0, i as f64 * 0.016);
    }
    at(ui, TouchPhase::End, 1.0, 0.2);
}

#[test]
fn a_swipe_is_heard_once_and_not_from_what_scrolls_its_way() {
    let mut ui = Ui::new();
    let root = ui.root();
    let drawer = ui.add_box(root, style! { width: 300px; height: 300px; overflow-y: auto; });
    let wide = ui.add_box(drawer, style! { height: 40px; overflow-x: auto; display: flex; });
    ui.add_box(wide, style! { width: 900px; height: 40px; flex-shrink: 0; });
    for _ in 0..30 {
        ui.add_box(drawer, style! { height: 20px; });
    }
    layout(&mut ui);
    let mut input: Input<u32> = Input::new();
    input.on_swipe(drawer, Toward::Left, 60.0, |n, _| *n += 1);
    let mut shut = 0;
    // up and down: it scrolls, and that is all
    drag(&mut ui, &mut input, &mut shut, (150.0, 250.0), (5.0, -150.0));
    assert_eq!(shut, 0);
    // left: heard, once however far it goes
    drag(&mut ui, &mut input, &mut shut, (250.0, 100.0), (-200.0, 6.0));
    assert_eq!(shut, 1);
    // right: not its way
    drag(&mut ui, &mut input, &mut shut, (50.0, 100.0), (200.0, 0.0));
    assert_eq!(shut, 1);
    // left on the wide table: the table scrolls, the drawer stays
    drag(&mut ui, &mut input, &mut shut, (250.0, 20.0), (-200.0, 0.0));
    assert_eq!(shut, 1);
    layout(&mut ui);
    assert!(ui.scroll_extent_x(wide).2 > 0.0, "the table scrolled");
}

/// A button pressed: by a finger, shown once the finger has rested a
/// moment (not when it lands to scroll); by the mouse, at once.
#[test]
fn a_finger_shows_its_press_once_it_rests() {
    use threers_ui::input::{PointerButton, PRESS_SHOWS};
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(root, style! { height: 100px; overflow-y: auto; });
    let button = ui.add_box(list, style! { height: 40px; background: #ff0000; });
    for _ in 0..10 {
        ui.add_box(list, style! { height: 40px; });
    }
    ui.active_style(button, "opacity: 0.5");
    layout(&mut ui);
    let mut input: Input<u32> = Input::new();
    input.on_click(button, |n, _, _| *n += 1);
    let mut n = 0;
    let touch = |input: &mut Input<u32>, ui: &mut Ui, n: &mut u32, phase, y: f32, time: f64| {
        input.dispatch(ui, n, Event::Touch { id: 1, phase, x: 50.0, y, time });
        input.tick(ui, n, time);
    };
    // landed, then rested
    input.tick(&mut ui, &mut n, 1.0);
    touch(&mut input, &mut ui, &mut n, TouchPhase::Start, 20.0, 1.0);
    assert_eq!(ui.style(button).opacity, 1.0, "not as it lands");
    input.tick(&mut ui, &mut n, 1.0 + PRESS_SHOWS + 0.01);
    assert_eq!(ui.style(button).opacity, 0.5, "pressed, once it rests");
    touch(&mut input, &mut ui, &mut n, TouchPhase::End, 20.0, 1.2);
    assert_eq!((ui.style(button).opacity, n), (1.0, 1), "let go: clicked, and lit no more");
    // landed to scroll: never lit
    touch(&mut input, &mut ui, &mut n, TouchPhase::Start, 20.0, 2.0);
    touch(&mut input, &mut ui, &mut n, TouchPhase::Move, 50.0, 2.03);
    input.tick(&mut ui, &mut n, 2.2);
    assert_eq!(ui.style(button).opacity, 1.0, "a scroll does not light what it lands on");
    touch(&mut input, &mut ui, &mut n, TouchPhase::End, 50.0, 2.3);
    assert_eq!(n, 1, "and does not click it");
    // the mouse: at once
    layout(&mut ui);
    let y = ui.rect(button).y + 10.0;
    input.dispatch(&mut ui, &mut n, Event::PointerMove { x: 50.0, y });
    input.dispatch(&mut ui, &mut n, Event::PointerDown { x: 50.0, y, button: PointerButton::Primary });
    assert_eq!(ui.style(button).opacity, 0.5, "a mouse press shows at once");
}
