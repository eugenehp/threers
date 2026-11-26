//! A sheet: a panel that rises from the bottom of the window over a dimmed
//! backdrop, shown while the state says so. It rests at one of its
//! [`Detent`]s (half the window, nearly all of it, a height), is dragged
//! between them by its grabber, and dragged down past the lowest, tapped
//! outside or sent Escape, it goes and the state is told. It is a modal
//! dialog for assistive technology, and gives focus back when it goes.
//!
//! ```
//! use threers_ui::input::Input;
//! use threers_ui::sheet::{sheet, Detent};
//! use threers_ui::{binding, FixedWidthText, Role, Ui};
//!
//! #[derive(Default)]
//! struct S { open: bool }
//!
//! let mut ui = Ui::new();
//! let mut input: Input<S> = Input::new();
//! let mut s = S::default();
//! ui.layout(400.0, 800.0, &mut FixedWidthText(7.0));
//! sheet(&mut input, binding!(|s: S| s.open), vec![Detent::Medium, Detent::Large], "Share", |ui, _, _, panel| {
//!     ui.add_text(panel, "Share to…");
//! });
//! s.open = true;
//! input.tick(&mut ui, &mut s, 0.0);
//! // It slides up from the bottom.
//! ui.step(0.3);
//! ui.layout(400.0, 800.0, &mut FixedWidthText(7.0));
//! let d = ui.accessibility_tree().into_iter().find(|a| a.role == Role::Dialog).unwrap();
//! assert_eq!(d.bounds.y, 400.0, "half the window");
//! input.simulate_keystrokes(&mut ui, &mut s, "escape").unwrap();
//! // It slides away, then goes, and the state is told.
//! ui.step(0.3);
//! ui.step(0.0);
//! input.tick(&mut ui, &mut s, 0.4);
//! assert!(!s.open);
//! ```

use std::cell::Cell;
use std::rc::Rc;

use crate::animation::{Animation, Property, Value};
use crate::binding::Binding;
use crate::environment::Token;
use crate::i18n::{word, Word};
use crate::input::{Event, Input, Phase};
use crate::transitions::Transition;
use crate::tree::{NodeId, Ui};
use crate::{style, Length, Role, Semantics};

/// A height a sheet rests at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Detent {
    /// Half the window.
    Medium,
    /// The window but a strip at the top (and the safe area).
    Large,
    /// This many pixels.
    Height(f32),
}

impl Detent {
    fn height(self, ui: &Ui) -> f32 {
        let h = ui.rect(ui.root()).height;
        match self {
            Detent::Medium => (h / 2.0).round(),
            Detent::Large => (h - ui.safe_area().top - 12.0).max(0.0),
            Detent::Height(px) => px.min(h),
        }
    }
}

type Fill<S> = dyn Fn(&mut Ui, &mut Input<S>, &S, NodeId);

/// Seconds a sheet takes to move to a detent, or away.
const MOVE: f32 = 0.25;

/// Shows a sheet while `presented` is true, resting first at the first of
/// `detents`; `fill` builds its content into the panel given.
pub fn sheet<S: 'static>(
    input: &mut Input<S>,
    presented: Binding<S, bool>,
    detents: Vec<Detent>,
    label: &str,
    fill: impl Fn(&mut Ui, &mut Input<S>, &S, NodeId) + 'static,
) {
    let label = label.to_owned();
    let detents = Rc::new(if detents.is_empty() {
        vec![Detent::Medium]
    } else {
        detents
    });
    let shown = presented.clone();
    input.present(
        move |s| shown.get(s),
        move |s| presented.set(s, false),
        move |ui, input, s| open(ui, input, s, detents.clone(), &label, &fill),
    );
}

fn open<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &mut S,
    detents: Rc<Vec<Detent>>,
    label: &str,
    fill: &Fill<S>,
) -> NodeId {
    let root = ui.root();
    // One box over the window holds the backdrop and the panel: it is what
    // is modal, and what goes.
    let layer = ui.add_box(
        root,
        style! { position: absolute; left: 0px; top: 0px; width: 100%; height: 100%; },
    );
    ui.defer(layer, 940);
    let backdrop = ui.add_box(layer, style! { position: absolute; left: 0px; top: 0px; width: 100%; height: 100%; background: rgba(0,0,0,0.4); });
    let panel = ui.add_box(
        layer,
        style! {
            position: absolute; left: 0px; width: 100%; display: flex; flex-direction: column; box-sizing: border-box;
            border-radius: 14px; padding: 0 16px 16px 16px; box-shadow: 0 -8px 30px rgba(0,0,0,0.35); overflow: hidden;
        },
    );
    ui.token_style(
        panel,
        &[("background", Token::Raised), ("color", Token::Text)],
    );
    let mut sem = Semantics::new(Role::Dialog).label(label).tab_index(-1);
    sem.description = Some(word(Word::SheetHint));
    ui.set_semantics(panel, sem);
    let grab = ui.add_box(panel, style! { display: flex; justify-content: center; padding: 8px 0 10px 0; cursor: grab; flex-shrink: 0; });
    let bar = ui.add_box(
        grab,
        style! { width: 36px; height: 5px; border-radius: 3px; },
    );
    ui.token_style(bar, &[("background", Token::Border)]);
    let content = ui.add_box(panel, style! { display: flex; flex-direction: column; gap: 8px; flex-grow: 1; min-height: 0; overflow: auto; });
    fill(ui, input, state, content);

    // Where it rests: an index into the detents; while dragged, how far.
    let at = Rc::new(Cell::new(0usize));
    let drag: Rc<Cell<Option<(f32, f32)>>> = Rc::default();
    let going = Rc::new(Cell::new(false));
    let place = {
        let (at, detents) = (at.clone(), detents.clone());
        move |ui: &mut Ui, animate: bool| {
            let window = ui.rect(ui.root()).height;
            let h = detents[at.get()].height(ui);
            let safe = ui.safe_area().bottom;
            let top = window - h;
            if animate {
                let from = ui.read(panel, Property::Top);
                ui.animate(
                    panel,
                    Animation::new(Property::Top, from, Value::Number(top), MOVE)
                        .easing(threers_math::easing::Easing::CubicOut),
                );
            } else if ui.style(panel).top != Length::Px(top) {
                ui.update_style(panel, |s| s.top = Length::Px(top));
            }
            if ui.style(panel).height != Length::Px(h + 20.0) {
                // Taller than it shows by its rounded bottom, kept off screen.
                ui.update_style(panel, |s| {
                    s.height = Length::Px(h + 20.0);
                    s.padding.bottom = 16.0 + safe + 20.0;
                });
            }
        }
    };
    let window = ui.rect(root).height;
    ui.update_style(panel, |s| s.top = Length::Px(window));
    place(ui, true);
    ui.appear(backdrop, Transition::Fade, MOVE);

    // Going: down and away, then removed (the state is told then).
    let dismiss = {
        let going = going.clone();
        move |ui: &mut Ui| {
            if going.replace(true) {
                return;
            }
            let window = ui.rect(ui.root()).height;
            let from = ui.read(panel, Property::Top);
            ui.animate(
                panel,
                Animation::new(Property::Top, from, Value::Number(window), MOVE),
            );
            ui.remove_with(layer, Transition::Fade, MOVE);
        }
    };
    let d = dismiss.clone();
    input.on_click(backdrop, move |_, ui, _| d(ui));
    let d = dismiss.clone();
    input.on_key(panel, move |_, ui, key, _, _| {
        if key == "Escape" {
            d(ui);
            return true;
        }
        false
    });
    // The grabber drags it: between detents, or down and away.
    let (dr, a, ds, d, pl) = (
        drag.clone(),
        at.clone(),
        detents.clone(),
        dismiss.clone(),
        place.clone(),
    );
    input.on(grab, Phase::Bubble, move |_, ui, e, cx| match *e {
        Event::PointerDown { y, .. } => {
            let top = match ui.style(panel).top {
                Length::Px(t) => t,
                _ => 0.0,
            };
            ui.stop_animations(panel, Some(Property::Top));
            dr.set(Some((y, top)));
            cx.capture_pointer();
        }
        Event::PointerMove { y, .. } => {
            if let Some((y0, top0)) = dr.get() {
                let window = ui.rect(ui.root()).height;
                let highest = window - ds.iter().map(|d| d.height(ui)).fold(0.0, f32::max);
                // Past the highest detent it resists, as a phone's does.
                let mut top = top0 + y - y0;
                if top < highest {
                    top = highest - (highest - top).sqrt() * 4.0;
                }
                ui.update_style(panel, |s| s.top = Length::Px(top));
            }
        }
        Event::PointerUp { .. } if dr.take().is_some() => {
            let window = ui.rect(ui.root()).height;
            let shown = window
                - match ui.style(panel).top {
                    Length::Px(t) => t,
                    _ => 0.0,
                };
            let heights: Vec<f32> = ds.iter().map(|d| d.height(ui)).collect();
            let lowest = heights.iter().copied().fold(f32::INFINITY, f32::min);
            if shown < lowest * 2.0 / 3.0 {
                d(ui);
            } else {
                let nearest = (0..heights.len())
                    .min_by(|&i, &j| {
                        (heights[i] - shown)
                            .abs()
                            .total_cmp(&(heights[j] - shown).abs())
                    })
                    .unwrap_or(0);
                a.set(nearest);
                pl(ui, true);
            }
        }
        _ => {}
    });

    // Modal, with focus in it; given back when it goes, however it goes.
    let before = input.focused();
    input.set_modal(ui, state, Some(layer));
    input.set_focus(ui, state, Some(panel));
    let pl = place;
    input.on_after_dispatch(move |input, ui, state| {
        if !ui.contains(layer) {
            if input.modal().is_none_or(|m| m == layer || !ui.contains(m)) {
                input.set_modal(ui, state, None);
            }
            input.set_focus(ui, state, before.filter(|&b| ui.contains(b)));
            return false;
        }
        // The window may have changed: it stays at its detent.
        if drag.get().is_none() && !going.get() && !ui.is_animating(panel) {
            pl(ui, false);
        }
        true
    });
    layer
}
