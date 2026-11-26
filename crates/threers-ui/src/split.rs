//! Two panes side by side (or one over the other) and a handle between
//! them to drag: the first pane's size follows the
//! handle, inside the minimum each pane keeps. A double-click puts it back.
//! The handle is a focusable `separator` whose value is the first pane's
//! share; the arrow keys move it, Home and End to its limits.

use std::cell::Cell;
use std::rc::Rc;

use crate::environment::LayoutDirection;
use crate::input::{Event, Input, Phase};
use crate::tree::{NodeId, Ui};
use crate::{style, Cursor, Length, Role, Semantics};

/// Which way the panes sit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// Side by side: the handle moves left and right.
    Horizontal,
    /// One over the other: the handle moves up and down.
    Vertical,
}

/// A split: the box holding it, its two panes to fill, and the handle.
#[derive(Debug, Clone, Copy)]
pub struct Split {
    pub container: NodeId,
    pub first: NodeId,
    pub second: NodeId,
    pub handle: NodeId,
}

/// How far the arrow keys move the handle, in pixels.
const STEP: f32 = 16.0;

/// Splits a new box under `parent` along `axis`: the first pane `size`
/// pixels, each pane at least `min` pixels. Give the container a size (it
/// grows to fill a flex parent).
pub fn split<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    parent: NodeId,
    axis: Axis,
    size: f32,
    min: f32,
) -> Split {
    let horizontal = axis == Axis::Horizontal;
    let container = ui.add_box(
        parent,
        style! { display: flex; flex-grow: 1; min-width: 0; min-height: 0; },
    );
    if !horizontal {
        ui.style_mut(container).flex_direction = crate::FlexDirection::Column;
    }
    let first = ui.add_box(container, style! { flex-shrink: 0; overflow: hidden; });
    let handle = ui.add_box(container, style! { flex-shrink: 0; background: #2a303c; });
    let second = ui.add_box(
        container,
        style! { flex-grow: 1; min-width: 0; min-height: 0; overflow: hidden; },
    );
    {
        let h = ui.style_mut(handle);
        if horizontal {
            h.width = Length::Px(6.0);
            h.cursor = Cursor::ResizeHorizontal;
        } else {
            h.height = Length::Px(6.0);
            h.cursor = Cursor::ResizeVertical;
        }
    }
    ui.token_style(
        handle,
        &[("background", crate::environment::Token::Separator)],
    );
    ui.hover_style(handle, "background: var(--accent)");
    ui.focus_visible_style(handle, "background: var(--accent)");
    let start = size;
    let current = Rc::new(Cell::new(size));
    // Sets the first pane's size, inside the minimums, and says it.
    let set = Rc::new(move |ui: &mut Ui, want: f32| -> f32 {
        let r = ui.rect(container);
        let room = if horizontal { r.width } else { r.height } - 6.0;
        let to = if room > 2.0 * min {
            want.clamp(min, room - min)
        } else {
            want.max(0.0)
        };
        ui.update_style(first, |s| {
            if horizontal {
                s.width = Length::Px(to);
            } else {
                s.height = Length::Px(to);
            }
        });
        let share = if room > 0.0 {
            (to / room * 100.0).round()
        } else {
            0.0
        };
        let mut sem = Semantics::new(Role::Separator).label("Resize").tab_index(0);
        sem.value = Some(format!("{share}%"));
        sem.range = Some(crate::a11y::RangeValue {
            min: 0.0,
            max: 100.0,
            now: share,
        });
        ui.set_semantics(handle, sem);
        to
    });
    let _ = set(ui, size);
    // Dragged: the handle takes the pointer, and the pane follows it.
    let dragging = Rc::new(Cell::new(false));
    let (d, cur, s2) = (dragging.clone(), current.clone(), set.clone());
    input.on(handle, Phase::Bubble, move |_, ui, e, cx| match *e {
        Event::PointerDown { .. } => {
            d.set(true);
            cx.capture_pointer();
        }
        Event::PointerMove { x, y } if d.get() => {
            let r = ui.rect(container);
            let along = match ui.direction_of(container) {
                LayoutDirection::LeftToRight => x - r.x,
                LayoutDirection::RightToLeft => r.x + r.width - x,
            };
            let at = if horizontal { along } else { y - r.y } - 3.0;
            cur.set(s2(ui, at));
        }
        Event::PointerUp { .. } => d.set(false),
        _ => {}
    });
    // A double-click puts it back (the input counts the clicks: asked
    // after the click, by the hook, which is given it).
    let clicked = Rc::new(Cell::new(false));
    let c = clicked.clone();
    input.on_click(handle, move |_, _, _| c.set(true));
    let (cur, s2) = (current.clone(), set.clone());
    input.on_after_dispatch(move |input, ui, _| {
        if clicked.take() && input.click_count() >= 2 {
            cur.set(s2(ui, start));
        }
        ui.contains(handle)
    });
    // The keys.
    let (cur, s2) = (current.clone(), set.clone());
    input.on_key(handle, move |_, ui, key, _, _| {
        let r = ui.rect(container);
        let room = if horizontal { r.width } else { r.height };
        let (less, more) = if horizontal {
            ("ArrowLeft", "ArrowRight")
        } else {
            ("ArrowUp", "ArrowDown")
        };
        let want = match ui.reading_key(container, key) {
            k if k == less => cur.get() - STEP,
            k if k == more => cur.get() + STEP,
            "Home" => 0.0,
            "End" => room,
            _ => return false,
        };
        cur.set(s2(ui, want));
        true
    });
    Split {
        container,
        first,
        second,
        handle,
    }
}
