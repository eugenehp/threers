//! The controls of a UI as a platform's own views would stand in for them:
//! where each one is, its shape, what it is and what it says. A shell that
//! shows the UI as a picture — a texture in a RealityKit scene, an image
//! beside a visionOS volume — lays a native view or entity over each, so
//! the system does what only it can:
//!
//! - **the look highlight.** visionOS never tells an app where its user is
//!   looking; it lights a control under the gaze only when the control is
//!   its own (SwiftUI's `hoverEffect`, RealityKit's `HoverEffectComponent`).
//! - **assistive technology.** VoiceOver, Switch Control and Full Keyboard
//!   Access find the buttons, not one picture.
//! - **exact presses.** A press on a native view is that control's, by its
//!   key, wherever the picture is drawn or however it is scaled.
//!
//! [`controls`] lists them from the accessibility tree (so a control is
//! whatever has a pressable [`Role`]: give a hand-built button
//! [`Semantics`](crate::Semantics)); [`version`] says when the list changed;
//! a press comes back as a click at the control's
//! [`centre`](Control::centre), through the
//! input the UI already has.
//!
//! ```
//! use threers_ui::native::{controls, Control};
//! use threers_ui::{style, FixedWidthText, Role, Semantics, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let b = ui.add_box(root, style! { width: 120px; height: 44px; border-radius: 22px; });
//! ui.set_semantics(b, Semantics::new(Role::Switch).label("Glass cortex").checked(true));
//! ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
//! let list = controls(&ui);
//! assert_eq!(list.len(), 1);
//! assert_eq!((list[0].label.as_str(), list[0].checked, list[0].radius), ("Glass cortex", Some(true), 22.0));
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::environment::Token;
use crate::tree::{NodeId, Rect, Ui};
use crate::{Rgba, Role};

/// A control, as a native view over it needs it. Rectangles are in the
/// UI's pixels (its layout's), from its top left.
#[derive(Debug, Clone, PartialEq)]
pub struct Control {
    /// Stays the same while the control does — across rebuilds of the
    /// tree, which give it a new [`NodeId`] — so a native view keeps its
    /// identity (and its highlight) when the UI is built again. Made of its
    /// role, its name and how many like it come before it.
    pub key: u64,
    pub node: NodeId,
    pub role: Role,
    /// Its accessible name.
    pub label: String,
    pub rect: Rect,
    /// Its corners' radius, for the highlight's shape.
    pub radius: f32,
    pub checked: Option<bool>,
    pub selected: Option<bool>,
    pub disabled: bool,
    /// Typed into ([`editable`]): a native text field stands over it, for
    /// the platform's keyboard, and sends what it holds back to the UI's
    /// own (`Input::set_focused_text`).
    pub editable: bool,
    /// Its accessible value: a text control's text, for the native field
    /// over it to start from (and to follow when the UI changes it).
    pub value: Option<String>,
    /// Slid ([`Role::Slider`]): a native view over it shows the look's
    /// highlight and passes the drag on to the UI.
    pub slider: bool,
    /// The UI's accent: the highlight's colour, as its own hover's.
    pub tint: Rgba,
}

impl Control {
    /// Where a press on it lands.
    pub fn centre(&self) -> (f32, f32) {
        (
            self.rect.x + self.rect.width / 2.0,
            self.rect.y + self.rect.height / 2.0,
        )
    }

    /// On, for what has a state: a ticked checkbox or switch, the picked
    /// tab or option.
    pub fn on(&self) -> bool {
        self.checked.or(self.selected).unwrap_or(false)
    }
}

/// Whether a role is something to press: a native view stands in for it.
pub fn pressable(role: Role) -> bool {
    matches!(
        role,
        Role::Button
            | Role::Link
            | Role::Checkbox
            | Role::Radio
            | Role::Switch
            | Role::Tab
            | Role::MenuItem
            | Role::Option
            | Role::ComboBox
    )
}

/// Whether a role is typed into: a native text field stands in for it.
pub fn editable(role: Role) -> bool {
    matches!(role, Role::TextBox | Role::TextArea | Role::SearchBox)
}

/// The controls `ui` shows, as last laid out, in the accessibility tree's
/// order. Only those a press at the centre would reach: one scrolled out
/// of view, clipped or covered is left out, as is anything with no size.
pub fn controls(ui: &Ui) -> Vec<Control> {
    let mut seen: HashMap<(Role, String), u32> = HashMap::new();
    let mut out = vec![];
    for a in ui.accessibility_tree() {
        if !(pressable(a.role) || editable(a.role) || a.role == Role::Slider) || a.bounds.width <= 0.0 || a.bounds.height <= 0.0 {
            continue;
        }
        let rect = a.bounds;
        let (cx, cy) = (rect.x + rect.width / 2.0, rect.y + rect.height / 2.0);
        if !reaches(ui, cx, cy, a.id) {
            continue;
        }
        let nth = seen.entry((a.role, a.name.clone())).or_insert(0);
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (a.role, &a.name, *nth).hash(&mut h);
        *nth += 1;
        let short = rect.width.min(rect.height);
        let radius = ui
            .style(a.id)
            .border_radius
            .resolve(Some(short))
            .unwrap_or(0.0)
            .clamp(0.0, short / 2.0);
        out.push(Control {
            key: h.finish(),
            node: a.id,
            role: a.role,
            label: a.name,
            rect,
            radius,
            checked: a.checked,
            selected: a.selected,
            disabled: a.disabled,
            editable: editable(a.role),
            value: a.value,
            slider: a.role == Role::Slider,
            tint: ui.token(Token::Accent),
        });
    }
    out
}

/// Whether a press at `(x, y)` lands on `node` or inside it.
fn reaches(ui: &Ui, x: f32, y: f32, node: NodeId) -> bool {
    let mut n = ui.hit_test(x, y);
    while let Some(id) = n {
        if id == node {
            return true;
        }
        n = ui.parent(id);
    }
    false
}

/// A number that changes when anything a native layer shows of `controls`
/// does: one is added or goes, moves, is renamed, turns on or off.
pub fn version(controls: &[Control]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for c in controls {
        c.key.hash(&mut h);
        c.label.hash(&mut h);
        for v in [c.rect.x, c.rect.y, c.rect.width, c.rect.height, c.radius] {
            v.to_bits().hash(&mut h);
        }
        (c.checked, c.selected, c.disabled, c.editable, &c.value, c.slider).hash(&mut h);
        c.tint.map(f32::to_bits).hash(&mut h);
    }
    h.finish()
}

/// The control with `key` among `controls`.
pub fn find(controls: &[Control], key: u64) -> Option<&Control> {
    controls.iter().find(|c| c.key == key)
}

/// `controls` as JSON, for a shell in another language: an array of
/// `{"key","role","label","x","y","w","h","r","on","toggle","disabled","edit","value","slide","tint"}`
/// (`value` a string, or null; `tint` `[r, g, b]`, 0–1, sRGB).
/// The key is a string (a 64-bit number does not survive a double);
/// `toggle` says whether it has an on and an off, `edit` whether it is
/// typed into.
pub fn to_json(controls: &[Control]) -> String {
    let mut s = String::from("[");
    for (i, c) in controls.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let r = c.rect;
        s.push_str(&format!(
            "{{\"key\":\"{}\",\"role\":\"{}\",\"label\":{},\"x\":{},\"y\":{},\"w\":{},\"h\":{},\"r\":{},\"on\":{},\"toggle\":{},\"disabled\":{},\"edit\":{},\"value\":{},\"slide\":{},\"tint\":[{},{},{}]}}",
            c.key,
            c.role.aria(),
            json_string(&c.label),
            r.x,
            r.y,
            r.width,
            r.height,
            c.radius,
            c.on(),
            c.checked.is_some() || c.selected.is_some(),
            c.disabled,
            c.editable,
            c.value.as_deref().map_or("null".to_owned(), json_string),
            c.slider,
            c.tint[0],
            c.tint[1],
            c.tint[2]
        ));
    }
    s.push(']');
    s
}

fn json_string(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{style, FixedWidthText, Semantics};

    fn card() -> (Ui, NodeId, NodeId) {
        let mut ui = Ui::new();
        let root = ui.root();
        let row = ui.add_box(root, style! { display: flex; gap: 10px; padding: 20px; });
        let a = ui.add_box(
            row,
            style! { width: 100px; height: 44px; border-radius: 22px; },
        );
        ui.set_semantics(
            a,
            Semantics::new(Role::Switch).label("Glass").checked(false),
        );
        let b = ui.add_box(
            row,
            style! { width: 100px; height: 44px; border-radius: 50%; },
        );
        ui.set_semantics(b, Semantics::new(Role::Button).label("Leave"));
        // not a control
        let t = ui.add_box(row, style! { width: 100px; height: 44px; });
        ui.add_text(t, "Title");
        ui.layout(500.0, 200.0, &mut FixedWidthText(7.0));
        (ui, a, b)
    }

    #[test]
    fn the_pressable_nodes_with_their_shapes_and_states() {
        let (ui, a, b) = card();
        let c = controls(&ui);
        assert_eq!(c.iter().map(|c| c.node).collect::<Vec<_>>(), [a, b]);
        assert_eq!(
            c[0].rect,
            Rect {
                x: 20.0,
                y: 20.0,
                width: 100.0,
                height: 44.0
            }
        );
        assert_eq!(
            (c[0].radius, c[1].radius),
            (22.0, 22.0),
            "a percentage of the short side, at most half"
        );
        assert_eq!((c[0].checked, c[0].on()), (Some(false), false));
        assert_eq!(c[1].centre(), (180.0, 42.0));
        let json = to_json(&c);
        assert!(
            json.contains("\"role\":\"switch\",\"label\":\"Glass\""),
            "{json}"
        );
        assert!(json.contains("\"toggle\":true") && json.contains("\"toggle\":false"));
    }

    #[test]
    fn keys_outlive_a_rebuild_and_the_version_follows_the_state() {
        let (ui, ..) = card();
        let before = controls(&ui);
        let (mut ui2, a2, _) = card();
        assert_eq!(
            controls(&ui2).iter().map(|c| c.key).collect::<Vec<_>>(),
            before.iter().map(|c| c.key).collect::<Vec<_>>()
        );
        assert_eq!(version(&controls(&ui2)), version(&before));
        ui2.set_semantics(
            a2,
            Semantics::new(Role::Switch).label("Glass").checked(true),
        );
        assert_ne!(version(&controls(&ui2)), version(&before), "switched on");
        assert!(find(&before, before[1].key).is_some());
    }

    #[test]
    fn a_search_box_is_typed_into() {
        use crate::binding::Binding;
        use crate::input::Input;
        let mut ui = Ui::new();
        let mut input: Input<String> = Input::new();
        let root = ui.root();
        let q = Binding::new(|s: &String| s.clone(), |s: &mut String, v| *s = v);
        crate::pickers::search_field(&mut ui, &mut input, &String::new(), root, "Search regions", q);
        ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
        let c = controls(&ui);
        let field = c.iter().find(|c| c.role == Role::SearchBox).expect("the search box");
        assert!(field.editable && !pressable(field.role));
        assert_eq!(field.label, "Search regions");
        assert!(to_json(&c).contains("\"role\":\"searchbox\",\"label\":\"Search regions\""));
        assert!(to_json(&c).contains("\"edit\":true"));
    }

    #[test]
    fn a_slider_is_listed_to_be_looked_at_and_slid() {
        use crate::binding::Binding;
        use crate::input::Input;
        let mut ui = Ui::new();
        let mut input: Input<f32> = Input::new();
        let root = ui.root();
        let v = Binding::new(|s: &f32| *s, |s: &mut f32, x| *s = x);
        crate::pickers::slider(&mut ui, &mut input, &0.5, root, "Opacity", v, 0.0..=1.0, 0.1);
        ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
        let c = controls(&ui);
        let s = c.iter().find(|c| c.role == Role::Slider).expect("the slider");
        assert!(s.slider && !s.editable);
        let json = to_json(&c);
        assert!(json.contains("\"slide\":true") && json.contains("\"tint\":["), "{json}");
    }

    #[test]
    fn a_covered_control_is_left_out() {
        let mut ui = Ui::new();
        let root = ui.root();
        let b = ui.add_box(root, style! { width: 100px; height: 44px; });
        ui.set_semantics(b, Semantics::new(Role::Button).label("Under"));
        ui.add_box(
            root,
            style! { position: absolute; left: 0px; top: 0px; width: 200px; height: 100px; },
        );
        ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
        assert!(controls(&ui).is_empty());
    }
}
