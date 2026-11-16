//! More controls, each bound to the state (see [`binding`](mod@crate::binding)):
//! a [`segmented`] picker whose highlight slides to the choice, a
//! [`stepper`], a [`slider`], a [`progress`] bar (known or not), an
//! [`empty_state`], and a [`search_field`]. Each has its role, value and
//! keys for assistive technology and the keyboard.
//!
//! ```
//! use threers_ui::input::Input;
//! use threers_ui::pickers::stepper;
//! use threers_ui::{binding, FixedWidthText, Ui};
//!
//! #[derive(Default)]
//! struct S { guests: i64 }
//!
//! let mut ui = Ui::new();
//! let mut input: Input<S> = Input::new();
//! let mut s = S { guests: 2 };
//! let root = ui.root();
//! let st = stepper(&mut ui, &mut input, &s, root, "Guests", binding!(|s: S| s.guests), 1..=8, 1);
//! ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
//! input.set_focus(&mut ui, &mut s, Some(st));
//! input.simulate_keystrokes(&mut ui, &mut s, "up up end").unwrap();
//! assert_eq!(s.guests, 8);
//! ```

use std::ops::RangeInclusive;
use std::rc::Rc;

use crate::a11y::RangeValue;
use crate::animation::{Animation, Property, Repeat, Value};
use crate::binding::Binding;
use crate::environment::Token;
use crate::input::{Event, Input, Phase};
use crate::matched::MatchShape;
use crate::tree::{NodeId, Ui};
use crate::{style, Display, FieldKey, Length, Role, Semantics, TextField};

/// Runs `show` with the bound value now, and again whenever it changes.
fn follow<S: 'static, T: Clone + PartialEq + 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    node: NodeId,
    value: Binding<S, T>,
    mut show: impl FnMut(&mut Ui, &T) + 'static,
) {
    let mut shown = value.get(state);
    show(ui, &shown);
    input.on_after_dispatch(move |_, ui, s| {
        if !ui.contains(node) {
            return false;
        }
        let now = value.get(s);
        if now != shown {
            show(ui, &now);
            shown = now;
        }
        true
    });
}

/// A row of choices, one picked, its highlight sliding to the one picked
/// (a radio group to assistive technology; the arrows move the choice).
pub fn segmented<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    label: &str,
    choices: &[&str],
    selected: Binding<S, usize>,
) -> NodeId {
    let row = ui.add_box(
        parent,
        // Wrapping: options that do not fit the width go on to another row,
        // each row's sharing it out, rather than running off the edge.
        style! { position: relative; display: flex; flex-wrap: wrap; padding: 2px; border-radius: 8px; gap: 2px; },
    );
    ui.token_style(row, &[("background", Token::Separator)]);
    ui.set_semantics(row, Semantics::new(Role::RadioGroup).label(label));
    let pill = ui.add_box(
        row,
        style! { position: absolute; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.25); },
    );
    ui.token_style(pill, &[("background", Token::Raised)]);
    let n = choices.len();
    let items: Vec<NodeId> = choices
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let b = ui.add_box(row, style! { position: relative; flex-grow: 1; padding: 4px 12px; border-radius: 6px; cursor: pointer; display: flex; justify-content: center; });
            ui.add_text(b, *c);
            ui.set_semantics(b, Semantics::new(Role::Radio).label(*c).checked(false));
            ui.focus_visible_style(b, "box-shadow: 0 0 0 2px #93c5fd");
            let sel = selected.clone();
            input.on_click(b, move |s, _, _| sel.set(s, i));
            let sel = selected.clone();
            input.on_key(b, move |s, _, key, _, _| {
                let k = sel.get(s).min(n - 1);
                let to = match key {
                    "ArrowRight" | "ArrowDown" => (k + 1) % n,
                    "ArrowLeft" | "ArrowUp" => (k + n - 1) % n,
                    _ => return false,
                };
                sel.set(s, to);
                true
            });
            b
        })
        .collect();
    if let Some(&first) = items.first() {
        ui.match_geometry(pill, first, MatchShape::Fill, 0.22);
    }
    let all = items.clone();
    follow(ui, input, state, row, selected, move |ui, &k| {
        for (i, &b) in all.iter().enumerate() {
            let s = ui.semantics_mut(b);
            s.checked = Some(i == k);
            s.tab_index = Some(if i == k { 0 } else { -1 });
        }
        if let Some(&b) = all.get(k) {
            ui.set_match_target(pill, b);
        }
    });
    row
}

/// A number with − and + buttons, kept in `range` by `step` (a spin
/// button: up and down step it, Home and End go to the ends).
#[allow(clippy::too_many_arguments)]
pub fn stepper<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    label: &str,
    value: Binding<S, i64>,
    range: RangeInclusive<i64>,
    step: i64,
) -> NodeId {
    let (lo, hi) = (*range.start(), *range.end());
    let row = ui.add_box(
        parent,
        style! { display: flex; align-items: center; gap: 8px; },
    );
    let name = ui.add_box(row, style! { flex-grow: 1; });
    ui.add_text(name, label);
    let shown = ui.add_box(
        row,
        style! { min-width: 28px; display: flex; justify-content: center; },
    );
    let number = ui.add_text(shown, "");
    let buttons = ui.add_box(
        row,
        style! { display: flex; border-radius: 6px; overflow: hidden; },
    );
    ui.token_style(buttons, &[("background", Token::Separator)]);
    let mut sem = Semantics::new(Role::SpinButton).label(label);
    sem.range = Some(RangeValue {
        min: lo as f32,
        max: hi as f32,
        now: 0.0,
    });
    ui.set_semantics(row, sem);
    ui.focus_visible_style(row, "box-shadow: 0 0 0 2px #93c5fd");
    let bump = move |v: &Binding<S, i64>, s: &mut S, by: i64| {
        let now = v.get(s);
        v.set(s, (now + by).clamp(lo, hi));
    };
    let mut ends = vec![];
    for (sign, glyph) in [(-1, "−"), (1, "+")] {
        let b = ui.add_box(buttons, style! { padding: 4px 12px; cursor: pointer; });
        ui.add_text(b, glyph);
        // The row is the control; its buttons are for the pointer.
        ui.set_semantics(b, Semantics::new(Role::Generic).hidden());
        ui.hover_style(b, "background: rgba(128,138,160,0.3)");
        let v = value.clone();
        input.on_click(b, move |s, _, _| bump(&v, s, sign * step));
        ends.push(b);
    }
    let v = value.clone();
    input.on_key(row, move |s, _, key, _, _| {
        match key {
            "ArrowUp" | "ArrowRight" => bump(&v, s, step),
            "ArrowDown" | "ArrowLeft" => bump(&v, s, -step),
            "Home" => v.set(s, lo),
            "End" => v.set(s, hi),
            _ => return false,
        }
        true
    });
    follow(ui, input, state, row, value, move |ui, &v| {
        ui.set_text(number, v.to_string());
        let s = ui.semantics_mut(row);
        s.value = Some(v.to_string());
        if let Some(r) = &mut s.range {
            r.now = v as f32;
        }
        for (b, off) in [(ends[0], v <= lo), (ends[1], v >= hi)] {
            ui.update_style(b, |st| st.opacity = if off { 0.4 } else { 1.0 });
            ui.semantics_mut(b).disabled = off;
        }
    });
    row
}

/// A value in `range` set by dragging a thumb along a track (a slider: the
/// arrows step it by `step`, Page Up and Down by ten steps, Home and End to
/// the ends).
#[allow(clippy::too_many_arguments)]
pub fn slider<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    label: &str,
    value: Binding<S, f32>,
    range: RangeInclusive<f32>,
    step: f32,
) -> NodeId {
    let (lo, hi) = (*range.start(), *range.end());
    let span = (hi - lo).max(f32::EPSILON);
    let track = ui.add_box(
        parent,
        style! { position: relative; height: 24px; cursor: pointer; },
    );
    let rail = ui.add_box(track, style! { position: absolute; left: 0px; top: 10px; width: 100%; height: 4px; border-radius: 2px; });
    ui.token_style(rail, &[("background", Token::Separator)]);
    let fill = ui.add_box(
        track,
        style! { position: absolute; left: 0px; top: 10px; height: 4px; border-radius: 2px; },
    );
    ui.token_style(fill, &[("background", Token::Accent)]);
    let thumb = ui.add_box(track, style! { position: absolute; top: 2px; width: 20px; height: 20px; border-radius: 10px; background: #ffffff; box-shadow: 0 1px 4px rgba(0,0,0,0.4); });
    let mut sem = Semantics::new(Role::Slider).label(label);
    sem.range = Some(RangeValue {
        min: lo,
        max: hi,
        now: lo,
    });
    ui.set_semantics(track, sem);
    ui.focus_visible_style(track, "box-shadow: 0 0 0 2px #93c5fd");
    let snap = move |v: f32| {
        let v = if step > 0.0 {
            lo + ((v - lo) / step).round() * step
        } else {
            v
        };
        v.clamp(lo, hi)
    };
    let at = move |ui: &Ui, x: f32| {
        let r = ui.rect(track);
        let usable = (r.width - 20.0).max(1.0);
        snap(lo + ((x - r.x - 10.0) / usable).clamp(0.0, 1.0) * span)
    };
    let v = value.clone();
    let dragging = Rc::new(std::cell::Cell::new(false));
    let d = dragging.clone();
    // the value before the press: a cancelled one (a finger that went on to
    // scroll the page) puts it back
    let before = Rc::new(std::cell::Cell::new(None::<f32>));
    input.on(track, Phase::Bubble, move |s, ui, e, cx| match *e {
        Event::PointerDown { x, .. } => {
            d.set(true);
            cx.capture_pointer();
            before.set(Some(v.get(s)));
            v.set(s, at(ui, x));
        }
        Event::PointerMove { x, .. } if d.get() => v.set(s, at(ui, x)),
        Event::PointerUp { .. } => {
            d.set(false);
            before.set(None);
        }
        Event::PointerCancel => {
            d.set(false);
            if let Some(was) = before.take() {
                v.set(s, was);
            }
        }
        _ => {}
    });
    let v = value.clone();
    input.on_key(track, move |s, _, key, _, _| {
        let now = v.get(s);
        let to = match key {
            "ArrowRight" | "ArrowUp" => now + step,
            "ArrowLeft" | "ArrowDown" => now - step,
            "PageUp" => now + step * 10.0,
            "PageDown" => now - step * 10.0,
            "Home" => lo,
            "End" => hi,
            _ => return false,
        };
        v.set(s, snap(to));
        true
    });
    follow(ui, input, state, track, value, move |ui, &v| {
        let t = ((v - lo) / span).clamp(0.0, 1.0);
        // Placed by share of the width, so it needs no layout to be right.
        ui.update_style(thumb, |s| s.left = Length::Percent(t * 100.0));
        ui.update_style(thumb, |s| s.margin.left = -20.0 * t);
        ui.update_style(fill, |s| s.width = Length::Percent(t * 100.0));
        let sem = ui.semantics_mut(track);
        sem.value = Some(format!("{v}"));
        if let Some(r) = &mut sem.range {
            r.now = v;
        }
    });
    track
}

/// A progress bar.
#[derive(Debug, Clone, Copy)]
pub struct Progress {
    pub bar: NodeId,
    fill: NodeId,
}

impl Progress {
    /// How far along, 0 to 1; `None` while it is not known (a stripe runs
    /// along the bar).
    pub fn set(&self, ui: &mut Ui, value: Option<f32>) {
        let sem = ui.semantics_mut(self.bar);
        sem.range = value.map(|v| RangeValue {
            min: 0.0,
            max: 100.0,
            now: (v * 100.0).round(),
        });
        sem.value = value.map(|v| format!("{}%", (v * 100.0).round()));
        match value {
            Some(v) => {
                ui.stop_animations(self.fill, Some(Property::Left));
                ui.update_style(self.fill, |s| {
                    s.left = Length::Px(0.0);
                    s.width = Length::Percent(v.clamp(0.0, 1.0) * 100.0);
                });
            }
            None => {
                let w = ui.rect(self.bar).width.max(100.0);
                ui.update_style(self.fill, |s| s.width = Length::Percent(30.0));
                if !ui.is_animating(self.fill) {
                    ui.animate(
                        self.fill,
                        Animation::new(
                            Property::Left,
                            Value::Number(-0.3 * w),
                            Value::Number(w),
                            1.2,
                        )
                        .repeat(Repeat::Forever),
                    );
                }
            }
        }
    }
}

/// A bar showing how far along something is (`value` 0 to 1), or that it
/// is under way when `None`.
pub fn progress(ui: &mut Ui, parent: NodeId, label: &str, value: Option<f32>) -> Progress {
    let bar = ui.add_box(
        parent,
        style! { position: relative; height: 6px; border-radius: 3px; overflow: hidden; },
    );
    ui.token_style(bar, &[("background", Token::Separator)]);
    let fill = ui.add_box(
        bar,
        style! { position: absolute; top: 0px; height: 100%; border-radius: 3px; },
    );
    ui.token_style(fill, &[("background", Token::Accent)]);
    ui.set_semantics(bar, Semantics::new(Role::ProgressBar).label(label));
    let p = Progress { bar, fill };
    p.set(ui, value);
    p
}

/// What an [`empty_state`]'s button does.
pub type EmptyAction<S> = Box<dyn FnMut(&mut S, &mut Ui)>;

/// What shows where there is nothing yet: a symbol, a title, a line of
/// explanation, and perhaps a button to do something about it.
pub fn empty_state<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    parent: NodeId,
    symbol: &str,
    title: &str,
    message: &str,
    action: Option<(&str, EmptyAction<S>)>,
) -> NodeId {
    let b = ui.add_box(parent, style! { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 24px 12px; white-space: normal; });
    ui.set_semantics(
        b,
        Semantics::new(Role::Status).label(format!("{title}. {message}")),
    );
    let sym = ui.add_box(b, style! { font-size: 32px; });
    ui.token_style(sym, &[("color", Token::SecondaryText)]);
    ui.add_text(sym, symbol);
    let t = ui.add_box(b, style! { font-weight: bold; font-size: 15px; });
    ui.add_text(t, title);
    let m = ui.add_box(b, style! { font-size: 12px; });
    ui.token_style(m, &[("color", Token::SecondaryText)]);
    ui.add_text(m, message);
    if let Some((label, mut f)) = action {
        let a = ui.add_box(
            b,
            style! { margin-top: 4px; padding: 4px 12px; border-radius: 6px; cursor: pointer; },
        );
        ui.token_style(
            a,
            &[("background", Token::Accent), ("color", Token::OnAccent)],
        );
        ui.add_text(a, label);
        ui.set_semantics(a, Semantics::new(Role::Button).label(label));
        input.on_click(a, move |s, ui, _| f(s, ui));
    }
    b
}

/// A search box bound to `query`: typed into with the keyboard, a ✕ to
/// clear it while it has text, `placeholder` while it has none.
pub fn search_field<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    placeholder: &str,
    query: Binding<S, String>,
) -> NodeId {
    let b = ui.add_box(parent, style! { display: flex; align-items: center; gap: 6px; padding: 5px 8px; border-radius: 8px; cursor: text; });
    ui.token_style(b, &[("background", Token::Separator)]);
    ui.focus_style(b, "box-shadow: 0 0 0 2px #2563eb");
    ui.set_semantics(b, Semantics::new(Role::SearchBox).label(placeholder));
    // A magnifier, drawn (not every face has a glyph for one).
    let icon = ui.add_path(
        b,
        style! { width: 14px; height: 14px; flex-shrink: 0; },
        vec!["M10 2a7 7 0 1 0 4.2 12.6l5.6 5.6 1.4-1.4-5.6-5.6A7 7 0 0 0 10 2zm0 2a5 5 0 1 1 0 10 5 5 0 0 1 0-10z".into()],
        [0.0, 0.0, 22.0, 22.0],
        None,
        true,
    );
    ui.token_style(icon, &[("color", Token::SecondaryText)]);
    ui.semantics_mut(icon).hidden = true;
    let text_box = ui.add_box(
        b,
        style! { flex-grow: 1; min-width: 0; overflow: hidden; white-space: nowrap; },
    );
    let text = ui.add_text(text_box, "");
    let clear = ui.add_box(
        b,
        style! { padding: 0 2px; cursor: pointer; display: none; },
    );
    ui.token_style(clear, &[("color", Token::SecondaryText)]);
    ui.add_text(clear, "✕");
    ui.set_semantics(clear, Semantics::new(Role::Button).label("Clear"));
    let q = query.clone();
    input.on_click(clear, move |s, _, cx| {
        q.set(s, String::new());
        cx.stop_propagation();
    });
    let q = query.clone();
    let field = Rc::new(std::cell::RefCell::new(TextField::new(query.get(state))));
    let f = field.clone();
    input.on_key(b, move |s, _, key, m, _| {
        // Up, down and Enter are the results' (see `search::searchable`):
        // a line this short has nothing for them to do.
        if matches!(key, "ArrowUp" | "ArrowDown" | "Enter") {
            return false;
        }
        let mut field = f.borrow_mut();
        // From the state, in case it changed there.
        let now = q.get(s);
        if field.text != now {
            *field = TextField::new(&now);
        }
        let used = !matches!(
            field.key(key, m, |_| true),
            FieldKey::Ignored | FieldKey::Tab { .. }
        );
        if field.text != now {
            q.set(s, field.text.clone());
        }
        used
    });
    // A host's own text input (a phone's keyboard) edits it too.
    let (f, q) = (field.clone(), query.clone());
    let (f2, q2) = (field.clone(), query.clone());
    input.register_text(
        b,
        move |s| {
            let field = f.borrow();
            let now = q.get(s);
            if field.text == now {
                (now, field.anchor, field.caret)
            } else {
                let n = now.chars().count();
                (now, n, n)
            }
        },
        move |s, _, text, anchor, caret| {
            let mut field = f2.borrow_mut();
            *field = TextField::new(&text);
            let n = text.chars().count();
            (field.anchor, field.caret) = (anchor.min(n), caret.min(n));
            q2.set(s, text);
        },
    );
    let placeholder = placeholder.to_owned();
    follow(ui, input, state, b, query, move |ui, q: &String| {
        if q.is_empty() {
            ui.set_text(text, placeholder.clone());
            ui.token_style(text_box, &[("color", Token::SecondaryText)]);
        } else {
            ui.set_text(text, q.clone());
            ui.token_style(text_box, &[("color", Token::Text)]);
        }
        let d = if q.is_empty() {
            Display::None
        } else {
            Display::Flex
        };
        ui.update_style(clear, |s| s.display = d);
        ui.semantics_mut(b).value = Some(q.clone());
    });
    b
}

#[cfg(test)]
mod slider_touch {
    use super::*;
    use crate::input::TouchPhase;
    use crate::FixedWidthText;

    /// A page that scrolls, a slider in it, and much more below.
    fn page() -> (Ui, Input<(f32, f32)>, NodeId) {
        let mut ui = Ui::new();
        let mut input: Input<(f32, f32)> = Input::new();
        let root = ui.root();
        let page = ui.add_box(root, style! { width: 300px; height: 200px; overflow-y: auto; display: flex; flex-direction: column; });
        let v = Binding::new(|s: &(f32, f32)| s.0, |s: &mut (f32, f32), x| s.0 = x);
        let row = ui.add_box(page, style! { flex-shrink: 0; });
        slider(
            &mut ui,
            &mut input,
            &(0.5, 0.0),
            row,
            "Opacity",
            v,
            0.0..=1.0,
            0.01,
        );
        ui.add_box(page, style! { height: 2000px; flex-shrink: 0; });
        ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
        (ui, input, page)
    }

    fn touch(ui: &mut Ui, input: &mut Input<(f32, f32)>, s: &mut (f32, f32), path: &[(f32, f32)]) {
        let (x, y) = path[0];
        input.dispatch(
            ui,
            s,
            Event::Touch {
                id: 1,
                phase: TouchPhase::Start,
                x,
                y,
                time: 0.0,
            },
        );
        for (i, &(x, y)) in path.iter().enumerate().skip(1) {
            input.dispatch(
                ui,
                s,
                Event::Touch {
                    id: 1,
                    phase: TouchPhase::Move,
                    x,
                    y,
                    time: i as f64 * 0.016,
                },
            );
        }
        let &(x, y) = path.last().unwrap();
        input.dispatch(
            ui,
            s,
            Event::Touch {
                id: 1,
                phase: TouchPhase::End,
                x,
                y,
                time: path.len() as f64 * 0.016,
            },
        );
        ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
    }

    #[test]
    fn a_finger_up_a_slider_scrolls_the_page_and_leaves_the_value() {
        let (mut ui, mut input, page) = page();
        let mut s = (0.5, 0.0);
        let track = ui
            .accessibility_tree()
            .into_iter()
            .find(|a| a.role == Role::Slider)
            .unwrap()
            .bounds;
        let (x, y) = (track.x + track.width * 0.9, track.y + track.height / 2.0);
        touch(
            &mut ui,
            &mut input,
            &mut s,
            &[(x, y), (x, y - 20.0), (x, y - 60.0), (x, y - 120.0)],
        );
        assert_eq!(s.0, 0.5, "the slider kept its value");
        assert!(
            ui.scroll_extent(page).2 > 50.0,
            "the page scrolled: {}",
            ui.scroll_extent(page).2
        );
    }

    #[test]
    fn a_finger_along_a_slider_slides_it_and_a_tap_sets_it() {
        let (mut ui, mut input, page) = page();
        let mut s = (0.5, 0.0);
        let track = ui
            .accessibility_tree()
            .into_iter()
            .find(|a| a.role == Role::Slider)
            .unwrap()
            .bounds;
        let (y, x0) = (track.y + track.height / 2.0, track.x + track.width / 2.0);
        touch(
            &mut ui,
            &mut input,
            &mut s,
            &[
                (x0, y),
                (x0 + 20.0, y + 2.0),
                (track.x + track.width, y + 3.0),
            ],
        );
        assert!(s.0 > 0.95, "slid to the end: {}", s.0);
        assert_eq!(ui.scroll_extent(page).2, 0.0, "the page stayed");
        touch(&mut ui, &mut input, &mut s, &[(track.x + 10.0, y)]);
        assert!(s.0 < 0.05, "a tap at its start: {}", s.0);
    }
}

/// A text box of several lines (HTML's `<textarea>`): `rows` lines tall,
/// the text wrapped as written (its own line breaks kept), the placeholder
/// shown while it is empty. Typed into from the keyboard ([`TextArea`]'s
/// keys: Enter breaks the line) or a host's own text input (a phone's
/// keyboard, a native field over it); its text lives in the state at
/// `text`.
///
/// ```
/// use threers_ui::binding::Binding;
/// use threers_ui::input::Input;
/// use threers_ui::pickers::text_area;
/// use threers_ui::{FixedWidthText, Role, Ui};
///
/// let mut ui = Ui::new();
/// let mut input: Input<String> = Input::new();
/// let mut text = String::new();
/// let root = ui.root();
/// let all = Binding::new(|s: &String| s.clone(), |s: &mut String, v| *s = v);
/// let b = text_area(&mut ui, &mut input, &text, root, "region,value", all, 4);
/// ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
/// input.set_focus(&mut ui, &mut text, Some(b));
/// input.simulate_keystrokes(&mut ui, &mut text, "a enter b").unwrap();
/// assert_eq!(text, "a\nb");
/// assert_eq!(ui.semantics(b).map(|s| s.role), Some(Role::TextArea));
/// // a line each
/// ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
/// assert_eq!(ui.children(ui.children(b)[0]).len(), 2);
/// ```
pub fn text_area<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    placeholder: &str,
    text: Binding<S, String>,
    rows: u32,
) -> NodeId {
    let b = ui.add_box(parent, style! { display: flex; padding: 7px 8px; border-radius: 7px; cursor: text; overflow-y: auto; });
    let line = 12.0 * 1.4;
    ui.update_style(b, |s| {
        s.min_height = Length::Px(rows.max(1) as f32 * line + 14.0)
    });
    ui.token_style(b, &[("background", Token::Separator)]);
    ui.focus_style(b, "box-shadow: 0 0 0 2px #2563eb");
    ui.set_semantics(b, Semantics::new(Role::TextArea).label(placeholder));
    // a box per line, each wrapping (the layout breaks lines at spaces, not
    // at the text's own line breaks)
    let shown = ui.add_box(b, style! { flex-grow: 1; min-width: 0; display: flex; flex-direction: column; font-size: 12px; line-height: 1.4; });
    let field = Rc::new(std::cell::RefCell::new(crate::text_area::TextArea::new(
        text.get(state),
    )));
    let (f, t) = (field.clone(), text.clone());
    input.on_key(b, move |s, _, key, m, _| {
        // Tab moves on, as a form's would
        if key == "Tab" {
            return false;
        }
        let mut field = f.borrow_mut();
        let now = t.get(s);
        if field.text != now {
            *field = crate::text_area::TextArea::new(now.clone());
            field.move_to(now.chars().count(), false);
        }
        let used = !matches!(field.key(key, m), FieldKey::Ignored | FieldKey::Tab { .. });
        if field.text != now {
            t.set(s, field.text.clone());
        }
        used
    });
    let (f, t) = (field.clone(), text.clone());
    let (f2, t2) = (field, text.clone());
    input.register_text(
        b,
        move |s| {
            let field = f.borrow();
            let now = t.get(s);
            if field.text == now {
                (now, field.anchor, field.caret)
            } else {
                let n = now.chars().count();
                (now, n, n)
            }
        },
        move |s, _, new, anchor, caret| {
            let mut field = f2.borrow_mut();
            *field = crate::text_area::TextArea::new(new.clone());
            let n = new.chars().count();
            (field.anchor, field.caret) = (anchor.min(n), caret.min(n));
            t2.set(s, new);
        },
    );
    let placeholder = placeholder.to_owned();
    follow(ui, input, state, b, text, move |ui, now: &String| {
        ui.clear_children(shown);
        let lines: Vec<&str> = if now.is_empty() {
            placeholder.split('\n').collect()
        } else {
            now.split('\n').collect()
        };
        for l in lines {
            // (each its size: a box's font size does not reach the text of
            // a box inside it)
            let row = ui.add_box(
                shown,
                style! { min-height: 16.8px; font-size: 12px; line-height: 1.4; },
            );
            ui.add_text(row, l);
        }
        let color = if now.is_empty() {
            Token::SecondaryText
        } else {
            Token::Text
        };
        ui.token_style(shown, &[("color", color)]);
        ui.semantics_mut(b).value = Some(now.clone());
    });
    b
}
