//! Small components: a [`switch`], a
//! [`disclosure`], [`tabs`], an [`avatar`], a [`badge`], and a [`popover`]
//! (an anchored panel of anything); and the parts most screens are built
//! of: a [`button`] in one of five [`ButtonKind`]s, a [`card`] and a
//! titled [`section`], a row of [`choices`], a [`setting`] and a
//! [`switch_row`], a stateless stepper ([`steps`]), and an in-place
//! [`confirm`]ation. Each is ordinary nodes, styled like the
//! rest of a UI, with its role, name and state for assistive technology and
//! the keys a keyboard user expects.

use std::cell::Cell;
use std::rc::Rc;

use crate::binding::Binding;
use crate::environment::Token;
use crate::input::{Event, Input};
use crate::layers::{Anchor, AnchorTo, Corner, Fit};
use crate::patch::Patch;
use crate::states::Appearance;
use crate::tree::{NodeId, Ui};
use crate::{style, Display, Length, Role, Semantics};

/// A switch's track and knob: each 3:1 or more against what it is on, on or
/// off, in either appearance (WCAG 1.4.11).
pub(crate) fn switch_colors(on: bool) -> (Token, Token) {
    if on {
        (Token::Accent, Token::OnAccent)
    } else {
        (Token::SecondaryText, Token::Surface)
    }
}

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
    let knob = ui.add_box(track, style! { position: absolute; top: 2px; width: 16px; height: 16px; border-radius: 8px; });
    ui.add_text(row, label);
    ui.set_semantics(
        row,
        Semantics::new(Role::Switch).label(label).checked(false),
    );
    ui.focus_visible_style(row, "box-shadow: 0 0 0 2px var(--focus)");
    let show = move |ui: &mut Ui, on: bool| {
        let (fill, dot) = switch_colors(on);
        ui.token_style(track, &[("background", fill)]);
        ui.token_style(knob, &[("background", dot)]);
        // (through its base: its colours are states over it)
        ui.update_style(knob, |s| s.left = Length::Px(if on { 16.0 } else { 2.0 }));
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
        style! { display: flex; align-items: center; gap: 6px; cursor: pointer; padding: 4px 0; min-height: 24px; border-radius: 6px; box-sizing: border-box; },
    );
    ui.hover_style(head, "background: var(--hover)");
    // Drawn, not glyphs: the same in every font, and the closed one points
    // the way the language reads.
    let [closed, opened] = ["M9 18l6-6-6-6", "M6 9l6 6 6-6"].map(|d| {
        let mut s = style! { width: 12px; height: 12px; flex-shrink: 0; };
        s.flips_rtl = true;
        let icon = crate::svg::add_icon(ui, head, s, &format!(r#"<path d="{d}"/>"#), 2.4);
        ui.token_style(icon, &[("color", Token::SecondaryText)]);
        icon
    });
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
        for (icon, on) in [(closed, !open), (opened, open)] {
            ui.update_style(icon, |s| {
                s.display = if on { Display::Flex } else { Display::None }
            });
        }
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
    let list = ui.add_box(parent, style! { position: relative; display: flex; gap: 2px; border-bottom: 1px solid; overflow-x: auto; scrollbar-width: none; flex-shrink: 0; });
    ui.token_style(list, &[("border-color", Token::Separator)]);
    ui.set_semantics(list, Semantics::new(Role::TabList));
    let tab_nodes: Vec<NodeId> = labels
        .iter()
        .map(|l| {
            let t = ui.add_box(list, style! { padding: 6px 12px 8px 12px; border-radius: 6px 6px 0 0; cursor: pointer; flex-shrink: 0; white-space: nowrap; });
            ui.hover_style(t, "background: var(--hover)");
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
    if let Some(&first) = tab_nodes.first() {
        ui.match_geometry(
            line,
            first,
            crate::matched::MatchShape::Underline(2.0),
            0.22,
        );
    }
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
            if on {
                // The colour of the text around the tabs.
                ui.clear_token_style(t, "color");
            } else {
                ui.token_style(t, &[("color", Token::SecondaryText)]);
            }
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
        input.on_key(t, move |s, ui, key, _, cx| {
            // From the tab with focus (which a click or Tab may have put on
            // one not selected).
            let k = i;
            let to = match ui.reading_key(t, key) {
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

/// A small pill of `text` in `color` (a count, a status), its words light or
/// dark, whichever reads better on it.
pub fn badge(ui: &mut Ui, parent: NodeId, text: &str, color: [f32; 4]) -> NodeId {
    let b = ui.add_box(
        parent,
        style! { padding: 1px 7px; border-radius: 9px; font-size: 11px; font-weight: bold; },
    );
    let ratio = |c: [f32; 4]| crate::contrast_ratio([c[0], c[1], c[2]], [color[0], color[1], color[2]]);
    let (light, dark) = ([0.99, 0.99, 0.99, 1.0], [0.07, 0.08, 0.1, 1.0]);
    ui.style_mut(b).background = Some(color);
    ui.style_mut(b).color = if ratio(light) >= ratio(dark) { light } else { dark };
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

    /// Whether it has been asked to close, or has closed.
    pub(crate) fn is_closing(&self, ui: &Ui) -> bool {
        self.closing.get() || !self.is_open(ui)
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
            border-radius: 10px; border: 1px solid;
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

/// How much a button asks for attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ButtonKind {
    /// The one thing to do here, filled with the accent: Save, Send.
    Primary,
    /// An ordinary action, outlined.
    #[default]
    Secondary,
    /// Text alone, for what matters least: Cancel, Show more.
    Plain,
    /// It destroys something, outlined in the destructive colour.
    Destructive,
    /// The same, text alone (a row's delete).
    DestructivePlain,
}

impl ButtonKind {
    fn borderless(self) -> bool {
        matches!(self, ButtonKind::Plain | ButtonKind::DestructivePlain)
    }

    /// Its look at rest: its declarations and the properties the theme
    /// colours (shared by [`button`] and [`crate::view::button`]).
    pub fn css(self) -> (&'static str, Vec<(&'static str, Token)>) {
        const BOXED: &str =
            "padding: 4px 12px; border-radius: 6px; font-size: 12px; cursor: pointer; \
             border: 1px solid; border-color: transparent; background: transparent";
        const TEXT: &str =
            "padding: 4px 8px; border-radius: 6px; font-size: 12px; cursor: pointer; \
             border: 1px solid; border-color: transparent; background: transparent";
        match self {
            ButtonKind::Primary => (
                BOXED,
                vec![
                    ("background", Token::Prominent),
                    ("border-color", Token::Prominent),
                    ("color", Token::OnProminent),
                ],
            ),
            ButtonKind::Secondary => (
                BOXED,
                vec![("border-color", Token::Border), ("color", Token::Text)],
            ),
            ButtonKind::Plain => (TEXT, vec![("color", Token::SecondaryText)]),
            ButtonKind::Destructive => (
                BOXED,
                vec![
                    ("border-color", Token::Destructive),
                    ("color", Token::Destructive),
                ],
            ),
            ButtonKind::DestructivePlain => (TEXT, vec![("color", Token::Destructive)]),
        }
    }

    /// Its looks under the pointer and pressed, in the theme's colours by
    /// name (they follow it).
    pub fn state_css(self) -> (&'static str, &'static str) {
        match self {
            ButtonKind::Primary => ("opacity: 0.88", "opacity: 0.76"),
            ButtonKind::Destructive | ButtonKind::DestructivePlain => (
                "background: color-mix(in srgb, var(--destructive) 12%, transparent)",
                "background: color-mix(in srgb, var(--destructive) 20%, transparent)",
            ),
            _ => (
                "background: var(--hover)",
                "background: var(--hover); opacity: 0.8",
            ),
        }
    }

    /// Its look at rest, in the theme's colours: with [`Ui::apply`] it
    /// restyles any node as this kind of button (states not included; see
    /// [`button`]).
    pub fn look(self) -> Patch {
        let (css, tokens) = self.css();
        tokens
            .into_iter()
            .fold(Patch::css(css), |p, (prop, t)| p.token(prop, t))
    }
}

/// A button: `label` in `kind`'s look, with its hover, press, keyboard
/// focus and disabled looks, the role `button` and its name. It has no
/// action of its own: give it one with [`on_press`] (which a disabled
/// button ignores) or [`Input::on_click`]; Space and Enter press it too.
///
/// A borderless kind in a column pulls its padding back, so its label
/// lines up with the text above and below it.
pub fn button(ui: &mut Ui, parent: NodeId, label: &str, kind: ButtonKind) -> NodeId {
    let b = ui.add_box(parent, Default::default());
    ui.apply(b, &kind.look());
    if kind.borderless() && ui.style(parent).flex_direction == crate::FlexDirection::Column {
        ui.apply(b, &Patch::css("margin-left: -8px; margin-right: -8px"));
    }
    ui.add_text(b, label);
    // The state looks in the theme's colours, by name: they follow it.
    let (hover, press) = kind.state_css();
    ui.hover_style(b, hover);
    ui.active_style(b, press);
    ui.focus_visible_style(b, "box-shadow: 0 0 0 2px var(--focus)");
    ui.disabled_style(
        b,
        match kind {
            ButtonKind::Primary => {
                "color: var(--secondary-text); border-color: var(--separator); background: var(--separator); cursor: default; opacity: 1"
            }
            k if k.borderless() => "color: var(--secondary-text); background: transparent; cursor: default; opacity: 1",
            _ => "color: var(--secondary-text); border-color: var(--separator); background: transparent; cursor: default; opacity: 1",
        },
    );
    ui.set_semantics(b, Semantics::new(Role::Button).label(label));
    b
}

/// Whether `node` (a [`button`], a choice) can be used: one that cannot
/// shows so, is skipped by Tab, and is not pressed by [`on_press`].
pub fn set_enabled(ui: &mut Ui, node: NodeId, enabled: bool) {
    if let Some(s) = ui.semantics(node).cloned() {
        ui.set_semantics(node, s.disabled(!enabled));
        // Its disabled look follows.
        ui.restyle(node);
    }
}

/// Runs `f` when `node` is pressed (a click, a tap, Space or Enter), unless
/// it is disabled ([`set_enabled`]).
pub fn on_press<S: 'static>(
    input: &mut Input<S>,
    node: NodeId,
    mut f: impl FnMut(&mut S) + 'static,
) {
    input.on_click(node, move |s, ui, _| {
        if !ui.semantics(node).is_some_and(|m| m.disabled) {
            f(s);
        }
    });
}

/// A card: a raised panel in the theme's surface colour, its children in
/// a column 8 px apart.
pub fn card(ui: &mut Ui, parent: NodeId) -> NodeId {
    let c = ui.add_box(parent, Default::default());
    ui.apply(
        c,
        &Patch::css("border: 1px solid; border-radius: 10px; padding: 16px")
            .stack(8.0)
            .token("background", Token::Surface)
            .token("border-color", Token::Border),
    );
    c
}

/// A row of choices, one of which may be `selected` (the accent on it):
/// buttons that wrap onto more lines, a radio group to assistive
/// technology. Returns the group and each choice, to give each an action.
pub fn choices<L: AsRef<str>>(
    ui: &mut Ui,
    parent: NodeId,
    labels: &[L],
    selected: Option<usize>,
) -> (NodeId, Vec<NodeId>) {
    let group = ui.add_box(parent, Patch::new().flow(8.0).into());
    ui.set_semantics(group, Semantics::new(Role::RadioGroup));
    let items = labels
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let on = selected == Some(i);
            let b = button(
                ui,
                group,
                l.as_ref(),
                if on {
                    ButtonKind::Primary
                } else {
                    ButtonKind::Secondary
                },
            );
            ui.set_semantics(b, Semantics::new(Role::Radio).label(l.as_ref()).checked(on));
            b
        })
        .collect();
    (group, items)
}

/// A card titled `title` (a level-2 heading, in small capitals): a group
/// of settings, a panel of a screen. Its children go in a column.
pub fn section(ui: &mut Ui, parent: NodeId, title: &str) -> NodeId {
    let c = card(ui, parent);
    let h = ui.add_box(c, style! { font-size: 12px; font-weight: bold; });
    ui.token_style(h, &[("color", Token::SecondaryText)]);
    ui.add_text(h, title.to_uppercase());
    ui.set_semantics(h, Semantics::new(Role::Heading(2)).label(title));
    c
}

/// A setting's `name` (and its `value` now, at the end of the line) over
/// what it does (`description`, wrapping; none when empty). Returns the box:
/// its control goes in under them.
pub fn setting(
    ui: &mut Ui,
    parent: NodeId,
    name: &str,
    value: Option<&str>,
    description: &str,
) -> NodeId {
    let b = ui.add_box(
        parent,
        (Patch::new().stack(8.0) + Patch::css("padding: 8px 0")).into(),
    );
    let head = ui.add_box(
        b,
        style! { display: flex; flex-direction: row; justify-content: space-between; gap: 8px; },
    );
    let n = ui.add_box(
        head,
        style! { font-weight: bold; white-space: normal; min-width: 0; },
    );
    ui.add_text(n, name);
    if let Some(v) = value {
        let v_box = ui.add_box(head, style! { flex-shrink: 0; });
        ui.token_style(v_box, &[("color", Token::SecondaryText)]);
        ui.add_text(v_box, v);
    }
    describe(ui, b, description);
    b
}

/// `description` under a setting's name, muted and wrapping.
fn describe(ui: &mut Ui, parent: NodeId, description: &str) {
    if !description.is_empty() {
        let d = ui.add_box(parent, style! { font-size: 13px; white-space: normal; });
        ui.token_style(d, &[("color", Token::SecondaryText)]);
        ui.add_text(d, description);
    }
}

/// A setting that is on or off: its `label` and `description` beside a
/// switch showing `on`. The whole row is the control (role `switch`, its
/// state `checked`), pressed by a click, Space or Enter: give it its
/// action with [`on_press`] and draw it again with the new value. (For a
/// UI that rebuilds from a model; [`switch`] keeps its value in a
/// binding.)
pub fn switch_row(ui: &mut Ui, parent: NodeId, label: &str, description: &str, on: bool) -> NodeId {
    // At least 24 px tall: a target a finger can hit (WCAG 2.2).
    let row = ui.add_box(
        parent,
        style! { display: flex; flex-direction: row; align-items: center; gap: 12px; padding: 8px 0; min-height: 24px; cursor: pointer; },
    );
    let info = ui.add_box(
        row,
        (Patch::new().stack(2.0).grow() + Patch::css("min-width: 0")).into(),
    );
    let l = ui.add_box(info, style! { font-weight: bold; white-space: normal; });
    ui.add_text(l, label);
    describe(ui, info, description);
    let track = ui.add_box(
        row,
        style! { display: flex; flex-direction: row; width: 36px; height: 20px; border-radius: 10px; padding: 2px; box-sizing: border-box; flex-shrink: 0; },
    );
    let (fill, dot) = switch_colors(on);
    ui.token_style(track, &[("background", fill)]);
    if on {
        ui.update_style(track, |s| s.justify_content = crate::JustifyContent::FlexEnd);
    }
    let knob = ui.add_box(
        track,
        style! { width: 16px; height: 16px; border-radius: 8px; },
    );
    ui.token_style(knob, &[("background", dot)]);
    ui.set_semantics(row, Semantics::new(Role::Switch).label(label).checked(on));
    ui.focus_visible_style(row, "box-shadow: 0 0 0 2px var(--focus)");
    row
}

/// The parts of a [`steps`] control, to give the buttons their actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Steps {
    pub root: NodeId,
    pub down: NodeId,
    pub up: NodeId,
}

/// A value shown as `shown` between a − and a + button, named `names`
/// (`("Decrease", "Increase")`, in the UI's language) for assistive
/// technology; an end of the range (`can_down`, `can_up` false) greys its
/// button out. Stateless: give the buttons actions with [`on_press`] and
/// draw it again with the new value. ([`crate::pickers::stepper`] keeps
/// its value in a binding.)
pub fn steps(
    ui: &mut Ui,
    parent: NodeId,
    shown: &str,
    (can_down, can_up): (bool, bool),
    names: (&str, &str),
) -> Steps {
    let root = ui.add_box(parent, Patch::new().hstack(8.0).into());
    let down = button(ui, root, "−", ButtonKind::Secondary);
    let value = ui.add_box(
        root,
        style! { font-weight: bold; min-width: 72px; text-align: center; },
    );
    ui.add_text(value, shown);
    let up = button(ui, root, "+", ButtonKind::Secondary);
    for (b, name, can) in [(down, names.0, can_down), (up, names.1, can_up)] {
        ui.set_semantics(b, Semantics::new(Role::Button).label(name).disabled(!can));
        set_enabled(ui, b, can);
    }
    Steps { root, down, up }
}

/// The parts of a [`confirm`]ation, to give the buttons their actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Confirm {
    pub root: NodeId,
    pub cancel: NodeId,
    pub yes: NodeId,
}

/// "Delete this?" in place, before something is destroyed: the `question`
/// (it wraps in a narrow column), then the `cancel` button and the
/// destructive `yes`, in that order, so the safe choice comes first.
pub fn confirm(ui: &mut Ui, parent: NodeId, question: &str, cancel: &str, yes: &str) -> Confirm {
    let root = ui.add_box(parent, Patch::new().stack(4.0).into());
    ui.style_mut(root).min_width = Length::Px(0.0);
    let q = ui.add_box(root, style! { font-size: 12px; white-space: normal; });
    ui.token_style(q, &[("color", Token::Destructive)]);
    ui.add_text(q, question);
    ui.set_semantics(q, Semantics::new(Role::Alert).label(question));
    let row = ui.add_box(root, Patch::new().hstack(8.0).into());
    let cancel = button(ui, row, cancel, ButtonKind::Plain);
    let yes = button(ui, row, yes, ButtonKind::Destructive);
    Confirm { root, cancel, yes }
}
