//! Small components: a [`switch`], a
//! [`disclosure`], [`tabs`], an [`avatar`], a [`badge`], and a [`popover`]
//! (an anchored panel of anything). Each is ordinary nodes, styled like the
//! rest of a UI, with its role, name and state for assistive technology and
//! the keys a keyboard user expects.

use std::cell::Cell;
use std::rc::Rc;

use crate::binding::Binding;
use crate::environment::Token;
use crate::input::{Event, Input};
use crate::layers::{Anchor, AnchorTo, Corner, Fit};
use crate::states::Appearance;
use crate::tree::{NodeId, Ui};
use crate::{style, Display, Length, Role, Semantics};

/// An on/off switch with `label`, showing `value`: a track and a knob that
/// slides, toggled by a click, a tap, Space or Enter (role `switch`, its
/// state `checked`). It keeps no state of its own: it writes the binding,
/// and shows the bound value however it changes.
pub fn switch<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    label: &str,
    value: Binding<S, bool>,
) -> NodeId {
    // At least 24 px tall: a target a finger can hit (WCAG 2.2).
    let row = ui.add_box(
        parent,
        style! { display: flex; align-items: center; gap: 8px; cursor: pointer; min-height: 24px; },
    );
    let track = ui.add_box(row, style! { position: relative; width: 34px; height: 20px; border-radius: 10px; flex-shrink: 0; });
    let knob = ui.add_box(track, style! { position: absolute; top: 2px; width: 16px; height: 16px; border-radius: 8px; background: #ffffff; });
    ui.add_text(row, label);
    ui.set_semantics(
        row,
        Semantics::new(Role::Switch).label(label).checked(false),
    );
    ui.focus_visible_style(row, "box-shadow: 0 0 0 2px #93c5fd");
    let show = move |ui: &mut Ui, on: bool| {
        ui.token_style(
            track,
            &[("background", if on { Token::Accent } else { Token::Border })],
        );
        ui.style_mut(knob).left = Length::Px(if on { 16.0 } else { 2.0 });
        ui.semantics_mut(row).checked = Some(on);
    };
    let mut shown = value.get(state);
    show(ui, shown);
    let v = value.clone();
    input.on_click(row, move |s, _, _| {
        let on = !v.get(s);
        v.set(s, on);
    });
    input.on_after_dispatch(move |_, ui, s| {
        if !ui.contains(row) {
            return false;
        }
        let on = value.get(s);
        if on != shown {
            shown = on;
            show(ui, on);
        }
        true
    });
    row
}

/// A heading that shows and hides what is under it, as `open` says:
/// returns the heading (a button, its state `expanded`) and the body to
/// fill. A click, Space or Enter toggles it. [`Binding::local`] keeps the
/// state in the disclosure itself.
pub fn disclosure<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    label: &str,
    open: Binding<S, bool>,
) -> (NodeId, NodeId) {
    let head = ui.add_box(
        parent,
        style! { display: flex; align-items: center; gap: 6px; cursor: pointer; padding: 4px 0; min-height: 24px; box-sizing: border-box; },
    );
    let arrow = ui.add_box(head, style! { width: 12px; });
    let arrow_text = ui.add_text(arrow, "▸");
    ui.add_text(head, label);
    ui.set_semantics(head, Semantics::new(Role::Button).label(label));
    let body = ui.add_box(
        parent,
        style! { display: flex; flex-direction: column; gap: 6px; padding: 2px 0 4px 18px; },
    );
    // Opening unfolds the body from its top edge, as paper does; closing
    // folds it up, then hides it.
    let show = move |ui: &mut Ui, open: bool, animate: bool| {
        use crate::animation::Property;
        use crate::depth::Transform3d;
        ui.set_text(arrow_text, if open { "▾" } else { "▸" });
        ui.semantics_mut(head).expanded = Some(open);
        let folded = -std::f32::consts::FRAC_PI_2;
        if !animate || ui.reduce_motion() {
            ui.update_style(body, |s| {
                s.display = if open { Display::Flex } else { Display::None }
            });
            ui.clear_transform3d(body);
            return;
        }
        let spring = crate::anim::Spring::new(220.0, 26.0, 1.0);
        if open {
            ui.update_style(body, |s| s.display = Display::Flex);
            ui.set_transform3d(
                body,
                Transform3d {
                    rotate_x: folded,
                    origin: (0.5, 0.0),
                    shade: true,
                    ..Transform3d::default()
                },
            );
            ui.spring_to(body, Property::RotateX, 0.0, spring);
        } else {
            if ui.transform3d(body).is_none() {
                ui.set_transform3d(
                    body,
                    Transform3d {
                        origin: (0.5, 0.0),
                        shade: true,
                        ..Transform3d::default()
                    },
                );
            }
            ui.spring_to(body, Property::RotateX, folded, spring);
        }
    };
    let mut shown = open.get(state);
    show(ui, shown, false);
    let o = open.clone();
    input.on_click(head, move |s, _, _| {
        let now = !o.get(s);
        o.set(s, now);
    });
    input.on_after_dispatch(move |_, ui, s| {
        if !ui.contains(head) {
            return false;
        }
        let now = open.get(s);
        if now != shown {
            shown = now;
            show(ui, now, true);
        }
        // Folded shut: hidden, and flat again for next time; open and
        // unfolded: flat.
        if let Some(t) = ui.transform3d(body)
            && ui.spring_velocity(body, crate::animation::Property::RotateX) == 0.0
        {
            if !now && t.rotate_x <= -std::f32::consts::FRAC_PI_2 + 1e-3 {
                ui.update_style(body, |st| st.display = Display::None);
                ui.clear_transform3d(body);
            } else if now && t.rotate_x.abs() < 1e-4 {
                ui.clear_transform3d(body);
            }
        }
        true
    });
    (head, body)
}

/// A row of tabs over panels: returned, the tab list, the tabs, and a
/// panel under each to fill, the `selected` one showing. A click picks a
/// tab; with focus on the tabs, the arrow keys, Home and End move between
/// them (as ARIA's tabs pattern has it).
pub struct Tabs {
    pub list: NodeId,
    pub tabs: Vec<NodeId>,
    pub panels: Vec<NodeId>,
}

pub fn tabs<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    labels: &[&str],
    selected: Binding<S, usize>,
) -> Tabs {
    // Scrolls sideways when the tabs are wider than it (larger text, a
    // phone).
    let list = ui.add_box(parent, style! { position: relative; display: flex; gap: 2px; border-bottom: 1px solid #3a4150; overflow-x: auto; scrollbar-width: none; flex-shrink: 0; });
    ui.token_style(list, &[("border-color", Token::Separator)]);
    ui.set_semantics(list, Semantics::new(Role::TabList));
    let tab_nodes: Vec<NodeId> = labels
        .iter()
        .map(|l| {
            let t = ui.add_box(list, style! { padding: 6px 12px 8px 12px; cursor: pointer; flex-shrink: 0; white-space: nowrap; });
            ui.add_text(t, *l);
            t
        })
        .collect();
    let panels: Vec<NodeId> = labels
        .iter()
        .map(|l| {
            let p = ui.add_box(
                parent,
                style! { display: flex; flex-direction: column; gap: 8px; padding: 8px 0; },
            );
            ui.set_semantics(p, Semantics::new(Role::TabPanel).label(*l));
            p
        })
        .collect();
    // The selected tab's underline, sliding to the tab picked.
    let line = ui.add_box(list, style! { position: absolute; border-radius: 1px; });
    ui.token_style(line, &[("background", Token::Accent)]);
    ui.match_geometry(
        line,
        tab_nodes[0],
        crate::matched::MatchShape::Underline(2.0),
        0.22,
    );
    let (tabs_, panels_) = (tab_nodes.clone(), panels.clone());
    let labels: Vec<String> = labels.iter().map(|l| l.to_string()).collect();
    let select = move |ui: &mut Ui, k: usize| {
        for (i, (&t, &p)) in tabs_.iter().zip(&panels_).enumerate() {
            let on = i == k;
            ui.set_semantics(
                t,
                Semantics::new(Role::Tab)
                    .label(labels[i].clone())
                    .selected(on)
                    .tab_index(if on { 0 } else { -1 }),
            );
            ui.update_style(t, |s| {
                s.font_weight = if on {
                    crate::FontWeight::Bold
                } else {
                    crate::FontWeight::Normal
                }
            });
            if on {
                ui.set_match_target(line, t);
                ui.scroll_into_view(t);
            }
            ui.update_style(p, |s| {
                s.display = if on { Display::Flex } else { Display::None }
            });
        }
    };
    let n = tab_nodes.len();
    let mut shown = selected.get(state).min(n.saturating_sub(1));
    select(ui, shown);
    for (i, &t) in tab_nodes.iter().enumerate() {
        let sel = selected.clone();
        input.on_click(t, move |s, _, _| sel.set(s, i));
        // The arrows move the choice and the focus along the tabs.
        let (sel, all) = (selected.clone(), tab_nodes.clone());
        input.on_key(t, move |s, _, key, _, cx| {
            // From the tab with focus (which a click or Tab may have put on
            // one not selected).
            let k = i;
            let to = match key {
                "ArrowRight" | "ArrowDown" => (k + 1) % n,
                "ArrowLeft" | "ArrowUp" => (k + n - 1) % n,
                "Home" => 0,
                "End" => n - 1,
                _ => return false,
            };
            sel.set(s, to);
            cx.focus(all[to]);
            true
        });
    }
    input.on_after_dispatch(move |_, ui, s| {
        if !ui.contains(list) {
            return false;
        }
        let k = selected.get(s).min(n.saturating_sub(1));
        if k != shown {
            shown = k;
            select(ui, k);
        }
        true
    });
    Tabs {
        list,
        tabs: tab_nodes,
        panels,
    }
}

/// A round picture of `name`'s initials, its colour from the name, `size`
/// pixels across (role `img`, named `name`).
pub fn avatar(ui: &mut Ui, parent: NodeId, name: &str, size: f32) -> NodeId {
    // Each at least 4.5:1 against the white initials (WCAG AA for text).
    let hues = [
        [0.706, 0.137, 0.094],
        [0.180, 0.490, 0.196],
        [0.114, 0.306, 0.847],
        [0.706, 0.325, 0.035],
        [0.427, 0.157, 0.851],
        [0.059, 0.463, 0.431],
    ];
    let h = name
        .bytes()
        .fold(5381u32, |h, b| h.wrapping_mul(33) ^ b as u32);
    let c = hues[h as usize % hues.len()];
    let a = ui.add_box(parent, style! { display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-weight: bold; });
    let s = ui.style_mut(a);
    s.width = Length::Px(size);
    s.height = Length::Px(size);
    s.border_radius = Length::Px(size / 2.0);
    s.background = Some([c[0], c[1], c[2], 1.0]);
    s.font_size = (size * 0.4).round();
    s.color = [0.99, 0.99, 0.99, 1.0];
    let initials: String = name
        .split_whitespace()
        .filter_map(|w| w.chars().next())
        .take(2)
        .collect::<String>()
        .to_uppercase();
    ui.add_text(a, initials);
    ui.set_semantics(a, Semantics::new(Role::Image).label(name));
    a
}

/// A small pill of `text` in `color` (a count, a status).
pub fn badge(ui: &mut Ui, parent: NodeId, text: &str, color: [f32; 4]) -> NodeId {
    let b = ui.add_box(
        parent,
        style! { padding: 1px 7px; border-radius: 9px; font-size: 11px; font-weight: bold; },
    );
    ui.style_mut(b).background = Some(color);
    ui.style_mut(b).color = [0.99, 0.99, 0.99, 1.0];
    ui.add_text(b, text);
    b
}

/// An open popover: its panel.
#[derive(Clone)]
pub struct PopoverHandle {
    pub panel: NodeId,
    closing: Rc<Cell<bool>>,
}

impl PopoverHandle {
    pub fn is_open(&self, ui: &Ui) -> bool {
        ui.contains(self.panel)
    }

    /// Asks it to close (it does after the event, giving focus back).
    pub fn close(&self) {
        self.closing.set(true);
    }
}

/// A panel anchored under `anchor`, holding whatever `fill` puts in it (a
/// form, a preview, a colour picker): a non-modal dialog (role `dialog`,
/// named `label`) that takes focus, and closes on Escape or a click
/// outside it, giving focus back. Flips above when there is no room below.
pub fn popover<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &mut S,
    anchor: NodeId,
    label: &str,
    fill: impl FnOnce(&mut Ui, NodeId),
) -> PopoverHandle {
    let root = ui.root();
    let panel = ui.add_box(
        root,
        style! {
            position: absolute; display: flex; flex-direction: column; gap: 8px; padding: 12px; max-width: 320px;
            border-radius: 10px; background: #22262f; border: 1px solid #3a4150; color: #e5e7eb;
            box-shadow: 0 10px 28px rgba(0,0,0,0.5);
        },
    );
    ui.token_style(
        panel,
        &[
            ("background", Token::Raised),
            ("border-color", Token::Border),
            ("color", Token::Text),
        ],
    );
    ui.appearance_style(
        panel,
        Appearance::Light,
        "box-shadow: 0 10px 28px rgba(0,0,0,0.18)",
    );
    ui.set_semantics(
        panel,
        Semantics::new(Role::Dialog).label(label).tab_index(-1),
    );
    ui.defer(panel, 900);
    ui.anchor(
        panel,
        Anchor::new(AnchorTo::Node(anchor, Corner::BottomLeft), Corner::TopLeft)
            .offset(0.0, 6.0)
            .fit(Fit::Flip),
    );
    fill(ui, panel);
    ui.appear(panel, crate::transitions::Transition::Fade, 0.12);
    let closing = Rc::new(Cell::new(false));
    let c = closing.clone();
    input.on_key(panel, move |_, _, key, _, _| {
        if key == "Escape" {
            c.set(true);
            return true;
        }
        false
    });
    let c = closing.clone();
    input.on_click_outside(panel, move |_, ui, e, _| {
        // Not the press on the anchor that opened it: that one toggles.
        let on_anchor = match *e {
            Event::PointerDown { x, y, .. } => ui
                .hit_test(x, y)
                .is_some_and(|h| ui.ancestors(h).any(|a| a == anchor)),
            _ => false,
        };
        if !on_anchor {
            c.set(true);
        }
    });
    let before = input.focused();
    input.set_focus(ui, state, Some(panel));
    let c = closing.clone();
    input.on_after_dispatch(move |input, ui, state| {
        // Removed from outside (see `Input::present`): focus that was in
        // it has gone with it, and goes back too.
        let gone = !ui.contains(panel);
        if !gone && !c.get() {
            return true;
        }
        let had = gone
            || input
                .focused()
                .is_some_and(|f| ui.contains(f) && ui.ancestors(f).any(|a| a == panel));
        ui.remove(panel);
        if had
            && input
                .focused()
                .is_none_or(|f| !ui.contains(f) || ui.ancestors(f).any(|a| a == panel))
        {
            input.set_focus(ui, state, before.filter(|&b| ui.contains(b)));
        }
        false
    });
    PopoverHandle { panel, closing }
}
