//! A number between − and + buttons: pressed, held (it repeats, faster as
//! it goes), stepped with the arrow keys, or typed.
//!
//! - [`Number`] is the value and its rules — a range, a step, decimals, a
//!   unit — with no UI: what a step, a key or typed text makes of it.
//! - [`draw`] draws one, stateless: give its buttons actions and draw it
//!   again with the new value (a UI built anew as it changes), with the
//!   text and caret of one being typed into.
//! - [`number_stepper`] is the whole control with its value in a
//!   [`Binding`](crate::binding::Binding).
//! - [`HoldRepeat`] says how many steps a button held makes as time passes
//!   (and times [`Input::on_repeat`], which does it for any control).
//!
//! ```
//! use threers_ui::number_stepper::{HoldRepeat, Number};
//!
//! let size = Number::new("Font size", 13.0, 8.0..=32.0).step(0.5).unit(" pt");
//! assert_eq!(size.shown(), "13 pt");
//! assert_eq!(size.stepped(3), 14.5);
//! assert_eq!(size.parse("40pt"), Some(32.0), "kept in its range");
//! assert_eq!(size.key("End"), Some(32.0));
//!
//! let mut hold = HoldRepeat::new();
//! hold.press(0.0);
//! assert_eq!(hold.steps(0.2), 0, "a tap isn't a hold");
//! assert!(hold.steps(1.0) > 3);
//! ```

use std::ops::RangeInclusive;

use crate::a11y::RangeValue;
use crate::binding::Binding;
use crate::input::Input;
use crate::tree::{NodeId, Ui};
use crate::{style, Role, Semantics, Style};

/// A value and its rules. See the [module docs](self).
#[derive(Debug, Clone, PartialEq)]
pub struct Number {
    pub label: String,
    pub value: f64,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    /// The most decimals it has (those the step needs, if not said).
    pub decimals: usize,
    /// Shown after it, as it is (`" px"`, `"%"`).
    pub unit: String,
}

impl Number {
    /// `value` in `range`, by ones.
    pub fn new(label: impl Into<String>, value: f64, range: RangeInclusive<f64>) -> Number {
        Number { label: label.into(), value, min: *range.start(), max: *range.end(), step: 1.0, decimals: 0, unit: String::new() }
    }

    /// Stepped by `step` (over nought), with the decimals that needs.
    pub fn step(mut self, step: f64) -> Number {
        self.step = if step > 0.0 { step } else { 1.0 };
        self.decimals = (0..=6).find(|&d| {
            let s = self.step * 10f64.powi(d as i32);
            (s - s.round()).abs() < 1e-6
        }).unwrap_or(6);
        self
    }

    pub fn decimals(mut self, decimals: usize) -> Number {
        self.decimals = decimals;
        self
    }

    pub fn unit(mut self, unit: impl Into<String>) -> Number {
        self.unit = unit.into();
        self
    }

    /// `v` as it may be: in the range, to its decimals.
    pub fn fit(&self, v: f64) -> f64 {
        let k = 10f64.powi(self.decimals as i32);
        ((v * k).round() / k).clamp(self.min, self.max)
    }

    /// `n` steps on (back, under nought). One between steps goes to the next
    /// on that way first: 13.3 by ones goes up to 14, down to 13.
    pub fn stepped(&self, n: i32) -> f64 {
        if n == 0 {
            return self.fit(self.value);
        }
        let at = self.value / self.step;
        let near = at.round();
        let on = (at - near).abs() < 1e-6;
        let from = match (on, n > 0) {
            (true, _) => near,
            (false, true) => at.floor(),
            (false, false) => at.ceil(),
        };
        self.fit((from + f64::from(n)) * self.step)
    }

    /// Whether it can be stepped back, and on.
    pub fn can(&self) -> (bool, bool) {
        (self.value > self.min, self.value < self.max)
    }

    /// Its digits, no more than they need.
    pub fn digits(&self) -> String {
        let s = format!("{:.*}", self.decimals, self.value);
        let s = if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.') } else { &s };
        if s == "-0" { "0".into() } else { s.into() }
    }

    /// Its digits and unit.
    pub fn shown(&self) -> String {
        format!("{}{}", self.digits(), self.unit)
    }

    /// What typed `text` makes it: a number (a comma for a point, its unit
    /// or spaces about it, as may be), kept in the range.
    pub fn parse(&self, text: &str) -> Option<f64> {
        let t = text.trim();
        let t = t.strip_suffix(self.unit.trim()).unwrap_or(t).trim().replace('−', "-").replace(',', ".");
        let v: f64 = t.parse().ok()?;
        v.is_finite().then(|| self.fit(v))
    }

    /// What a key makes it: an arrow a step, Page Up and Down ten, Home and
    /// End the range's ends.
    pub fn key(&self, key: &str) -> Option<f64> {
        Some(match key {
            "ArrowUp" | "ArrowRight" => self.stepped(1),
            "ArrowDown" | "ArrowLeft" => self.stepped(-1),
            "PageUp" => self.stepped(10),
            "PageDown" => self.stepped(-10),
            "Home" => self.min,
            "End" => self.max,
            _ => return None,
        })
    }
}

/// A button held: the steps it makes. See the [module docs](self).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoldRepeat {
    /// How long until it repeats, then how often, slowest and fastest, and
    /// over how long it gets there (seconds).
    pub delay: f64,
    pub slow: f64,
    pub fast: f64,
    pub over: f64,
    /// When it was pressed, and when the next step is due.
    held: Option<(f64, f64)>,
}

impl Default for HoldRepeat {
    fn default() -> Self {
        HoldRepeat { delay: 0.4, slow: 0.14, fast: 0.03, over: 2.0, held: None }
    }
}

impl HoldRepeat {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn press(&mut self, time: f64) {
        self.held = Some((time, time + self.delay));
    }

    pub fn release(&mut self) {
        self.held = None;
    }

    pub fn held(&self) -> bool {
        self.held.is_some()
    }

    /// Steps due by `time` since last asked (a long gap isn't caught up on).
    pub fn steps(&mut self, time: f64) -> u32 {
        let Some((from, mut due)) = self.held else { return 0 };
        let mut n = 0;
        while due <= time && n < 20 {
            n += 1;
            let t = ((due - from - self.delay) / self.over.max(0.01)).clamp(0.0, 1.0);
            due += (self.slow + (self.fast - self.slow) * t).max(0.005);
        }
        self.held = Some((from, if due <= time { time + self.fast } else { due }));
        n
    }
}

/// How [`draw`]'s control looks.
#[derive(Debug, Clone)]
pub struct StepperLook {
    /// The control's box: background, border, corners, text.
    pub frame: Style,
    /// Between its parts, and a caret.
    pub line: [f32; 4],
    pub caret: [f32; 4],
}

/// The parts drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumberStepper {
    pub root: NodeId,
    pub down: NodeId,
    pub value: NodeId,
    pub up: NodeId,
}

/// `number` between its buttons, one at an end of the range greyed (give it
/// no action); `typing`: the text being typed into it, and the caret's place
/// in it (bytes). See the [module docs](self).
pub fn draw(ui: &mut Ui, parent: NodeId, number: &Number, typing: Option<(&str, usize)>, look: &StepperLook) -> NumberStepper {
    let mut frame = look.frame.clone();
    for (k, v) in [("display", "flex"), ("align-items", "stretch"), ("overflow", "hidden"), ("flex-shrink", "0"), ("user-select", "none")] {
        crate::css::apply(&mut frame, k, v);
    }
    let root = ui.add_box(parent, frame);
    let mut sem = Semantics::new(Role::SpinButton).label(number.label.clone());
    sem.value = Some(number.shown());
    sem.range = Some(RangeValue { min: number.min as f32, max: number.max as f32, now: number.value as f32 });
    ui.set_semantics(root, sem);
    let (can_down, can_up) = number.can();
    let side = |ui: &mut Ui, glyph: &str, can: bool| {
        let b = ui.add_box(root, style! { display: flex; align-items: center; justify-content: center; width: 38px; min-height: 34px; font-size: 18px; flex-shrink: 0; });
        if can {
            crate::css::apply(ui.style_mut(b), "cursor", "pointer");
        } else {
            ui.style_mut(b).opacity = 0.35;
        }
        ui.add_text(b, glyph);
        // The control is the one thing; its buttons are for the pointer.
        ui.set_semantics(b, Semantics::new(Role::Generic).hidden());
        b
    };
    let rule = |ui: &mut Ui| {
        let r = ui.add_box(root, style! { width: 1px; flex-shrink: 0; });
        ui.style_mut(r).background = Some(look.line);
    };
    let down = side(ui, "−", can_down);
    rule(ui);
    let value = ui.add_box(
        root,
        style! { display: flex; align-items: center; justify-content: center; min-width: 72px; padding: 0 10px; font-size: 14px; cursor: text; white-space: pre; },
    );
    match typing {
        None => {
            ui.add_text(value, number.shown());
        }
        Some((text, caret)) => {
            let caret = (0..=caret.min(text.len())).rev().find(|&i| text.is_char_boundary(i)).unwrap_or(0);
            let (before, after) = text.split_at(caret);
            if !before.is_empty() {
                ui.add_text(value, before);
            }
            let c = ui.add_box(value, style! { width: 1px; height: 18px; flex-shrink: 0; });
            ui.style_mut(c).background = Some(look.caret);
            let rest = format!("{after}{}", number.unit);
            if !rest.is_empty() {
                ui.add_text(value, rest);
            }
        }
    }
    rule(ui);
    let up = side(ui, "+", can_up);
    NumberStepper { root, down, value, up }
}

/// The whole control, its value kept in a binding: its buttons, pressed or
/// held (they repeat, by [`Input::tick`]'s clock), and the keys while it has
/// the focus (a step, ten, the ends; digits typed, then Enter).
pub fn number_stepper<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    number: Number,
    value: Binding<S, f64>,
    look: &StepperLook,
) -> NumberStepper {
    let start = Number { value: value.get(state), ..number.clone() };
    let parts = draw(ui, parent, &start, None, look);
    let NumberStepper { root, down, value: shown, up } = parts;
    ui.semantics_mut(root).tab_index = Some(0);
    ui.focus_visible_style(root, "box-shadow: 0 0 0 2px var(--focus)");
    let typed: std::rc::Rc<std::cell::RefCell<String>> = Default::default();
    for (b, n) in [(down, -1), (up, 1)] {
        // (either may come to be at the end, and leave it)
        crate::css::apply(ui.style_mut(b), "cursor", "pointer");
        let (number, value, typed) = (number.clone(), value.clone(), typed.clone());
        let step = std::rc::Rc::new(move |s: &mut S| {
            typed.borrow_mut().clear();
            let now = Number { value: value.get(s), ..number.clone() };
            value.set(s, now.stepped(n));
        });
        let held = step.clone();
        input.on_click(b, move |s: &mut S, _, _| step(s));
        input.on_repeat(b, HoldRepeat::new(), move |s, _| held(s));
    }
    let (n, v, t) = (number.clone(), value.clone(), typed.clone());
    input.on_key(root, move |s, _, key, _, _| {
        let now = Number { value: v.get(s), ..n.clone() };
        let mut t = t.borrow_mut();
        match key {
            "Enter" if !t.is_empty() => {
                if let Some(x) = now.parse(&t) {
                    v.set(s, x);
                }
                t.clear();
            }
            "Escape" if !t.is_empty() => t.clear(),
            "Backspace" if !t.is_empty() => {
                t.pop();
            }
            k if k.len() == 1 && "0123456789.,-".contains(k) => t.push_str(k),
            k => match now.key(k) {
                Some(x) => {
                    t.clear();
                    v.set(s, x);
                }
                None => return false,
            },
        }
        true
    });
    let mut last = (f64::NAN, String::new());
    let mut show = move |ui: &mut Ui, now: f64| {
        let t = typed.borrow().clone();
        if last.0 == now && last.1 == t {
            return;
        }
        let number = Number { value: now, ..number.clone() };
        let text = if t.is_empty() { number.shown() } else { format!("{t}{}", number.unit) };
        if let Some(&node) = ui.children(shown).first() {
            ui.set_text(node, text);
        }
        let (can_down, can_up) = number.can();
        for (b, can) in [(down, can_down), (up, can_up)] {
            ui.update_style(b, |st| st.opacity = if can { 1.0 } else { 0.35 });
        }
        let sem = ui.semantics_mut(root);
        sem.value = Some(number.shown());
        if let Some(r) = &mut sem.range {
            r.now = now as f32;
        }
        last = (now, t);
    };
    show(ui, start.value);
    input.on_after_dispatch(move |_, ui, s| {
        if !ui.contains(root) {
            return false;
        }
        show(ui, value.get(s));
        true
    });
    parts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{Event, PointerButton, TouchPhase};
    use crate::{FixedWidthText, KeyModifiers};

    #[test]
    fn steps_snap_and_stay_in_the_range() {
        let n = |v: f64| Number::new("n", v, 0.0..=10.0).step(0.25);
        assert_eq!(n(1.0).stepped(1), 1.25);
        assert_eq!(n(1.1).stepped(1), 1.25, "between steps: the next on");
        assert_eq!(n(1.1).stepped(-1), 1.0);
        assert_eq!(n(9.9).stepped(5), 10.0);
        assert_eq!(n(0.0).stepped(-1), 0.0);
        assert_eq!(n(0.3).decimals, 2);
        // no dust from adding tenths
        let mut v = 0.0;
        for _ in 0..7 {
            v = Number::new("n", v, 0.0..=1.0).step(0.1).stepped(1);
        }
        assert_eq!(v, 0.7);
        assert_eq!(Number::new("n", v, 0.0..=1.0).step(0.1).can(), (true, true));
    }

    #[test]
    fn shown_and_typed() {
        let p = Number::new("Zoom", 110.0, 50.0..=300.0).step(10.0).unit("%");
        assert_eq!(p.shown(), "110%");
        assert_eq!(p.parse(" 125 % "), Some(125.0), "typed, it needn't be a step");
        assert_eq!(p.parse("5"), Some(50.0));
        assert_eq!((p.parse("abc"), p.parse(""), p.parse("inf")), (None, None, None));
        let f = Number::new("Size", 13.5, 8.0..=32.0).step(0.5);
        assert_eq!(f.shown(), "13.5");
        assert_eq!(f.parse("14,24"), Some(14.2));
        assert_eq!(Number::new("n", -0.0, -5.0..=5.0).shown(), "0");
        assert_eq!(Number::new("n", 0.0, -5.0..=5.0).parse("−3"), Some(-3.0));
        assert_eq!((p.key("PageUp"), p.key("Home"), p.key("x")), (Some(210.0), Some(50.0), None));
    }

    #[test]
    fn a_hold_repeats_and_speeds_up() {
        let mut h = HoldRepeat::new();
        assert_eq!(h.steps(5.0), 0);
        h.press(10.0);
        let mut count = |from: f64, to: f64| {
            let (mut n, mut t) = (0, from);
            while t < to {
                t += 1.0 / 60.0;
                n += h.steps(t);
            }
            n
        };
        assert_eq!(count(10.0, 10.3), 0);
        let (first, later) = (count(10.3, 11.3), count(12.3, 13.3));
        assert!(first > 4 && later > first * 2, "{first} then {later}");
        h.release();
        assert_eq!(h.steps(20.0), 0);
        // a frame that came late isn't caught up on for ever
        h.press(0.0);
        assert!(h.steps(100.0) <= 20 && h.steps(100.01) == 0);
    }

    #[test]
    fn the_control_steps_types_and_greys_its_ends() {
        let (mut ui, mut input, mut v) = (Ui::new(), Input::<f64>::new(), 9.0);
        let look = StepperLook { frame: Style::default(), line: [1.0; 4], caret: [1.0; 4] };
        let root = ui.root();
        let number = Number::new("Count", 0.0, 0.0..=10.0).unit(" x");
        let p = number_stepper(&mut ui, &mut input, &v, root, number, Binding::new(|s: &f64| *s, |s: &mut f64, v| *s = v), &look);
        let m = &mut FixedWidthText(8.0);
        ui.layout(400.0, 200.0, m);
        assert!(ui.rect(p.up).x > ui.rect(p.value).x && ui.rect(p.value).x > ui.rect(p.down).x);
        assert!(ui.rect(p.up).height >= 34.0);
        let mut click = |ui: &mut Ui, input: &mut Input<f64>, v: &mut f64, n: NodeId| {
            let r = ui.rect(n);
            let (x, y) = (r.x + r.width / 2.0, r.y + r.height / 2.0);
            input.dispatch(ui, v, Event::PointerDown { x, y, button: PointerButton::Primary });
            input.dispatch(ui, v, Event::PointerUp { x, y, button: PointerButton::Primary });
        };
        click(&mut ui, &mut input, &mut v, p.up);
        assert_eq!((v, ui.text_content(p.value).as_str()), (10.0, "10 x"));
        assert!(ui.style(p.up).opacity < 0.5 && ui.style(p.down).opacity == 1.0);
        click(&mut ui, &mut input, &mut v, p.up);
        assert_eq!(v, 10.0);
        input.set_focus(&mut ui, &mut v, Some(p.root));
        for k in ["ArrowDown", "4", "2"] {
            input.dispatch(&mut ui, &mut v, Event::Key { key: k.into(), modifiers: KeyModifiers::default() });
        }
        assert_eq!((v, ui.text_content(p.value).as_str()), (9.0, "42 x"), "typed, not yet taken");
        for k in ["Backspace", "Enter"] {
            input.dispatch(&mut ui, &mut v, Event::Key { key: k.into(), modifiers: KeyModifiers::default() });
        }
        assert_eq!((v, ui.text_content(p.value).as_str()), (4.0, "4 x"));
    }

    struct Held {
        ui: Ui,
        input: Input<f64>,
        v: f64,
        p: NumberStepper,
        t: f64,
    }

    impl Held {
        fn new(value: f64) -> Held {
            let (mut ui, mut input) = (Ui::new(), Input::<f64>::new());
            let root = ui.root();
            let list = ui.add_box(root, style! { display: flex; flex-direction: column; width: 300px; height: 100px; overflow-y: auto; });
            let look = StepperLook { frame: Style::default(), line: [1.0; 4], caret: [1.0; 4] };
            let row = ui.add_box(list, style! { height: 40px; flex-shrink: 0; display: flex; });
            let number = Number::new("Count", 0.0, 0.0..=100.0);
            let p = number_stepper(&mut ui, &mut input, &value, row, number, Binding::new(|s: &f64| *s, |s: &mut f64, v| *s = v), &look);
            ui.add_box(list, style! { height: 600px; flex-shrink: 0; });
            let mut h = Held { ui, input, v: value, p, t: 1.0 };
            h.pass(0.05);
            h
        }

        fn pass(&mut self, seconds: f64) {
            let end = self.t + seconds;
            while self.t < end {
                self.t += 1.0 / 60.0;
                self.input.tick(&mut self.ui, &mut self.v, self.t);
                self.ui.layout(400.0, 200.0, &mut FixedWidthText(8.0));
            }
        }

        fn middle(&self, n: NodeId) -> (f32, f32) {
            let r = self.ui.rect(n);
            (r.x + r.width / 2.0, r.y + r.height / 2.0)
        }

        fn mouse(&mut self, down: bool, (x, y): (f32, f32)) {
            let button = PointerButton::Primary;
            let e = if down { Event::PointerDown { x, y, button } } else { Event::PointerUp { x, y, button } };
            self.input.dispatch(&mut self.ui, &mut self.v, e);
        }

        fn finger(&mut self, phase: TouchPhase, (x, y): (f32, f32)) {
            self.input.dispatch(&mut self.ui, &mut self.v, Event::Touch { id: 1, phase, x, y, time: self.t });
        }
    }

    #[test]
    fn a_mouse_held_repeats_and_isnt_a_click_too() {
        let mut h = Held::new(50.0);
        let up = h.middle(h.p.up);
        h.mouse(true, up);
        assert!(h.input.needs_frame(), "an idle host goes on ticking");
        h.pass(0.2);
        assert_eq!(h.v, 50.0, "not yet");
        h.pass(0.8);
        let held = h.v;
        assert!(held > 53.0, "{held}");
        assert_eq!(h.ui.text_content(h.p.value), format!("{held}"), "shown as it goes");
        // off the button it waits, with nothing owed
        h.input.dispatch(&mut h.ui, &mut h.v, Event::PointerMove { x: 350.0, y: 150.0 });
        h.pass(1.0);
        assert_eq!(h.v, held);
        h.input.dispatch(&mut h.ui, &mut h.v, Event::PointerMove { x: up.0, y: up.1 });
        h.pass(0.1);
        assert!(h.v > held && h.v < held + 12.0, "{} then {}", held, h.v);
        let last = h.v;
        h.mouse(false, up);
        h.pass(1.0);
        assert_eq!(h.v, last, "let go: over, and no click");
        assert!(!h.input.needs_frame());
        // a short press is a click: one step
        h.mouse(true, up);
        h.pass(0.1);
        h.mouse(false, up);
        assert_eq!(h.v, last + 1.0);
        // held to its end
        h.mouse(true, up);
        h.pass(5.0);
        h.mouse(false, up);
        assert_eq!(h.v, 100.0);
        assert!(h.ui.style(h.p.up).opacity < 0.5);
    }

    #[test]
    fn a_finger_held_repeats_and_one_drawn_scrolls() {
        let mut h = Held::new(50.0);
        let down = h.middle(h.p.down);
        h.finger(TouchPhase::Start, down);
        h.pass(1.2);
        let held = h.v;
        assert!(held < 45.0, "past a long press's time, still going: {held}");
        h.finger(TouchPhase::End, down);
        h.pass(0.5);
        assert_eq!(h.v, held, "lifted: no tap too");
        h.finger(TouchPhase::Start, down);
        h.pass(0.1);
        h.finger(TouchPhase::End, down);
        assert_eq!(h.v, held - 1.0, "a tap");
        // begun on it and drawn away: the list scrolls, nothing steps
        let at = h.v;
        h.finger(TouchPhase::Start, down);
        for k in 1..=6 {
            h.pass(0.02);
            h.finger(TouchPhase::Move, (down.0, down.1 - 10.0 * k as f32));
        }
        h.pass(1.0);
        h.finger(TouchPhase::End, (down.0, down.1 - 60.0));
        h.pass(0.5);
        assert_eq!(h.v, at);
    }

    #[test]
    fn a_hosts_press_is_carried_from_tree_to_tree() {
        // a UI built anew at each step, its presses the host's
        let build = |v: f64| {
            let (mut ui, mut input) = (Ui::new(), Input::<f64>::new());
            let root = ui.root();
            let look = StepperLook { frame: Style::default(), line: [1.0; 4], caret: [1.0; 4] };
            let number = Number::new("Count", v, 0.0..=20.0);
            let p = draw(&mut ui, root, &number, None, &look);
            if number.can().1 {
                input.on_repeat_named(p.up, "up", HoldRepeat::new(), |s: &mut f64, _| *s += 1.0);
            }
            ui.layout(400.0, 200.0, &mut FixedWidthText(8.0));
            let r = ui.rect(p.up);
            (ui, input, (r.x + r.width / 2.0, r.y + r.height / 2.0))
        };
        let mut v = 0.0;
        let (mut ui, mut input, (x, y)) = build(v);
        assert!(!input.press_repeat(&ui, 390.0, 190.0), "nothing there repeats");
        assert!(input.press_repeat(&ui, x, y));
        let mut run = |ui: &mut Ui, input: &mut Input<f64>, v: &mut f64, t: f64| {
            for k in 0..60 {
                if input.tick_repeat(ui, v, t + k as f64 / 60.0) > 0 {
                    let (u, mut i, _) = build(*v);
                    i.carry_repeat(input);
                    (*ui, *input) = (u, i);
                }
            }
        };
        run(&mut ui, &mut input, &mut v, 1.0);
        assert!(v > 3.0 && input.repeating(), "{v}");
        // off it, it waits
        input.move_repeat(390.0, 190.0);
        let at = v;
        run(&mut ui, &mut input, &mut v, 2.0);
        assert_eq!(v, at);
        input.move_repeat(x, y);
        run(&mut ui, &mut input, &mut v, 3.0);
        run(&mut ui, &mut input, &mut v, 4.0);
        assert_eq!(v, 20.0, "to its end, where nothing has the name any more");
        assert!(!input.end_repeat());
        // let go in time: it repeated, so it's no click
        let (mut ui, mut input, (x, y)) = build(0.0);
        input.press_repeat(&ui, x, y);
        assert!(!input.end_repeat(), "at once: a click");
        input.press_repeat(&ui, x, y);
        let mut v = 0.0;
        input.tick_repeat(&mut ui, &mut v, 1.0);
        input.tick_repeat(&mut ui, &mut v, 1.5);
        assert!(v >= 1.0 && input.end_repeat());
    }
}
