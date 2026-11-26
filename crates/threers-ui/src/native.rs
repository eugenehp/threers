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
    /// over it to start from (and to follow when the UI changes it). A
    /// [`secret`](Self::secret) one's is never given: `None` while there is
    /// one, empty when there is none (and the native field is emptied).
    pub value: Option<String>,
    /// It holds a password: the native field over it is a secure one.
    pub secret: bool,
    /// Slid ([`Role::Slider`]): a native view over it shows the look's
    /// highlight and passes the drag on to the UI.
    pub slider: bool,
    /// A slider's thumb, for a shell that shows a solid one: its box.
    pub thumb: Option<Rect>,
    /// What it is painted on, see-through where it has nothing of its own:
    /// for a shell that shows it raised.
    pub fill: Rgba,
    /// What of it such a shell raises, and its corners: a switch's track, a
    /// box to tick, a radio's ring; of any other, all of it.
    pub solid: Rect,
    pub solid_radius: f32,
    /// A slider's rail, for a shell that shows it solid: the thin bars it
    /// is drawn with (its track, then what of it is filled), each with its
    /// colour. Given with one of a rail's controls only (a range's two
    /// thumbs share theirs).
    pub rails: Vec<(Rect, Rgba)>,
    /// A knob's turn ([`pickers::knob`](crate::pickers::knob)): radians
    /// right of up, for a shell that shows a solid one. A slider that is
    /// itself round.
    pub dial: Option<f32>,
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
            | Role::TreeItem(_)
    )
}

/// Whether `id`, of a role that only sometimes is, is something to press: a
/// table's heading that sorts it, one of its rows.
fn pressed_in_a_table(ui: &Ui, id: NodeId, role: Role) -> bool {
    match role {
        Role::ColumnHeader => ui.style(id).cursor == crate::Cursor::Pointer,
        // (a heading row has no place among the rows)
        Role::Row => ui.semantics(id).is_some_and(|s| s.row_index.is_some()),
        _ => false,
    }
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
    let mut railed = std::collections::HashSet::new();
    for a in ui.accessibility_tree() {
        if !(pressable(a.role) || editable(a.role) || a.role == Role::Slider || pressed_in_a_table(ui, a.id, a.role))
            || a.bounds.width <= 0.0
            || a.bounds.height <= 0.0
        {
            continue;
        }
        // what shows of it: inside every box that clips or scrolls it (a
        // row scrolled half out of its list is half a control; one scrolled
        // away, none)
        let Some(rect) = visible(ui, a.id, a.bounds) else {
            continue;
        };
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
        let mark = matches!(a.role, Role::Switch | Role::Checkbox | Role::Radio)
            .then(|| mark(ui, a.id))
            .flatten();
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
            // (none where the UI has its own keyboard on: pressed, it has focus
            // and that keyboard comes up)
            editable: editable(a.role) && !own_keyboard(ui),
            // (its bullets are not its text: edited, they would replace it)
            value: a.value.filter(|v| !a.secret || v.is_empty()),
            secret: a.secret,
            slider: a.role == Role::Slider,
            dial: (a.role == Role::Slider).then(|| dial(ui, a.id)).flatten(),
            thumb: (a.role == Role::Slider && dial(ui, a.id).is_none()).then(|| thumb(ui, a.id)).flatten(),
            fill: ui.style(mark.unwrap_or(a.id)).background.unwrap_or([0.0; 4]),
            solid: mark.map_or(rect, |m| ui.rect(m)),
            solid_radius: mark.map_or(radius, |m| corner(ui, m)),
            rails: if a.role == Role::Slider { rails(ui, a.id, &mut railed) } else { vec![] },
            tint: ui.token(Token::Accent),
        });
    }
    out
}

/// A slider's rail: the thin painted bars in it, or round it (a range's
/// thumb is in its rail's box), not given before (in `seen`). Widest first.
fn rails(ui: &Ui, id: NodeId, seen: &mut std::collections::HashSet<NodeId>) -> Vec<(Rect, Rgba)> {
    fn bars(ui: &Ui, id: NodeId, out: &mut Vec<NodeId>) {
        let (r, s) = (ui.rect(id), ui.style(id));
        if s.background.is_some_and(|c| c[3] > 0.3) && (1.5..=10.0).contains(&r.height) && r.width >= 3.0 * r.height {
            out.push(id);
        }
        for &c in ui.children(id) {
            bars(ui, c, out);
        }
    }
    let mut at = Some(id);
    for _ in 0..4 {
        let Some(n) = at else { break };
        let mut found = vec![];
        bars(ui, n, &mut found);
        if !found.is_empty() {
            found.retain(|b| seen.insert(*b));
            let mut out: Vec<(Rect, Rgba)> = found.iter().map(|&b| (ui.rect(b), ui.style(b).background.unwrap_or([0.0; 4]))).collect();
            out.sort_by(|a, b| b.0.width.total_cmp(&a.0.width));
            return out;
        }
        at = ui.parent(n);
    }
    vec![]
}

/// A knob's turn: of a slider that is round (as wide as tall, its corners
/// half that), where its value has it.
fn dial(ui: &Ui, id: NodeId) -> Option<f32> {
    let r = ui.rect(id);
    let round = (r.width - r.height).abs() < 1.0
        && r.width >= 24.0
        && ui.style(id).border_radius.resolve(Some(r.width)).unwrap_or(0.0) >= r.width * 0.45;
    let range = ui.semantics(id)?.range?;
    round.then(|| {
        let t = ((range.now - range.min) / (range.max - range.min).max(f32::EPSILON)).clamp(0.0, 1.0);
        (t - 0.5) * crate::pickers::KNOB_SWEEP.to_radians()
    })
}

/// Whether `ui` types with its own keyboard now
/// ([`VirtualKeyboard`](crate::virtual_keyboard::VirtualKeyboard)).
fn own_keyboard(ui: &Ui) -> bool {
    ui.environment::<crate::virtual_keyboard::OwnKeyboard>(ui.root()).is_some_and(|k| k.on())
}

/// `id`'s corners' radius.
fn corner(ui: &Ui, id: NodeId) -> f32 {
    let r = ui.rect(id);
    let short = r.width.min(r.height);
    let radius = ui.style(id).border_radius.resolve(Some(short));
    radius.unwrap_or(0.0).clamp(0.0, short / 2.0)
}

/// What marks a row that is on or off (a switch's track, a box to tick, a
/// ring): the first small painted or bordered box in it, if the row is not
/// itself painted.
fn mark(ui: &Ui, id: NodeId) -> Option<NodeId> {
    if ui.style(id).background.is_some_and(|c| c[3] > 0.05) {
        return None;
    }
    fn find(ui: &Ui, id: NodeId) -> Option<NodeId> {
        let (r, s) = (ui.rect(id), ui.style(id));
        let painted = s.background.is_some_and(|c| c[3] > 0.05) || (s.border.horizontal() > 0.0 && s.border_color[3] > 0.05);
        if painted && (12.0..=64.0).contains(&r.width) && (12.0..=36.0).contains(&r.height) {
            return Some(id);
        }
        ui.children(id).iter().find_map(|&c| find(ui, c))
    }
    let row = ui.rect(id);
    let found = ui.children(id).iter().find_map(|&c| find(ui, c))?;
    // (not what is most of it: a chip's own box)
    (ui.rect(found).width < row.width * 0.75).then_some(found)
}

/// A slider's thumb: the round box in `id`, or the one `id` is the target
/// of.
fn thumb(ui: &Ui, id: NodeId) -> Option<Rect> {
    fn round(ui: &Ui, id: NodeId) -> Option<Rect> {
        let r = ui.rect(id);
        let s = ui.style(id);
        ((r.width - r.height).abs() < 1.0
            && (10.0..=32.0).contains(&r.width)
            && s.background.is_some_and(|c| c[3] > 0.5)
            && s.border_radius.resolve(Some(r.width)).unwrap_or(0.0) >= r.width * 0.45)
            .then_some(r)
    }
    // the last painted: over its rail
    fn within(ui: &Ui, id: NodeId) -> Option<Rect> {
        round(ui, id).or_else(|| ui.children(id).iter().rev().find_map(|&c| within(ui, c)))
    }
    within(ui, id).or_else(|| round(ui, ui.parent(id)?))
}

/// `rect` (node `id`'s) cut to what its clipping and scrolling ancestors
/// show; `None` when nothing of it does.
fn visible(ui: &Ui, id: NodeId, mut rect: Rect) -> Option<Rect> {
    let mut n = ui.parent(id);
    while let Some(p) = n {
        let s = ui.style(p);
        if s.clip || s.scroll || s.scroll_x || ui.parent(p).is_none() {
            let b = ui.rect(p);
            let (x0, y0) = (rect.x.max(b.x), rect.y.max(b.y));
            let (x1, y1) = (
                (rect.x + rect.width).min(b.x + b.width),
                (rect.y + rect.height).min(b.y + b.height),
            );
            if x1 <= x0 || y1 <= y0 {
                return None;
            }
            rect = Rect {
                x: x0,
                y: y0,
                width: x1 - x0,
                height: y1 - y0,
            };
        }
        n = ui.parent(p);
    }
    Some(rect)
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
        (
            c.checked, c.selected, c.disabled, c.editable, &c.value, c.slider, c.secret,
        )
            .hash(&mut h);
        c.tint.map(f32::to_bits).hash(&mut h);
        c.fill.map(f32::to_bits).hash(&mut h);
        [c.solid.x, c.solid.y, c.solid.width, c.solid.height, c.solid_radius].map(f32::to_bits).hash(&mut h);
        c.thumb.map(|r| [r.x, r.y, r.width, r.height].map(f32::to_bits)).hash(&mut h);
        c.dial.map(f32::to_bits).hash(&mut h);
        for (r, k) in &c.rails {
            [r.x, r.y, r.width, r.height, k[0], k[1], k[2], k[3]].map(f32::to_bits).hash(&mut h);
        }
    }
    h.finish()
}

/// The control with `key` among `controls`.
pub fn find(controls: &[Control], key: u64) -> Option<&Control> {
    controls.iter().find(|c| c.key == key)
}

/// `controls` as JSON, for a shell in another language: an array of
/// `{"key","role","label","x","y","w","h","r","on","toggle","disabled","edit","secret","value","slide","tint","fill","thumb","solid","rails","dial"}`
/// (`value` a string, or null; `tint` `[r, g, b]` and `fill` `[r, g, b, a]`,
/// 0–1, sRGB; `thumb` a slider's, `[x, y, w, h]`, or null; `solid` `[x, y, w, h, radius]`;
/// `rails` a slider's bars, `[x, y, w, h, r, g, b, a]` each, track first;
/// `dial` a knob's turn, radians right of up, or null).
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
            "{{\"key\":\"{}\",\"role\":\"{}\",\"label\":{},\"x\":{},\"y\":{},\"w\":{},\"h\":{},\"r\":{},\"on\":{},\"toggle\":{},\"disabled\":{},\"edit\":{},\"secret\":{},\"value\":{},\"slide\":{},\"tint\":[{},{},{}],\"fill\":[{},{},{},{}],\"thumb\":{},\"solid\":[{},{},{},{},{}],\"rails\":[{}],\"dial\":{}}}",
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
            c.secret,
            c.value.as_deref().map_or("null".to_owned(), json_string),
            c.slider,
            c.tint[0],
            c.tint[1],
            c.tint[2],
            c.fill[0],
            c.fill[1],
            c.fill[2],
            c.fill[3],
            c.thumb.map_or("null".to_owned(), |t| format!("[{},{},{},{}]", t.x, t.y, t.width, t.height)),
            c.solid.x,
            c.solid.y,
            c.solid.width,
            c.solid.height,
            c.solid_radius,
            c.rails.iter().map(|(r, k)| format!("[{},{},{},{},{},{},{},{}]", r.x, r.y, r.width, r.height, k[0], k[1], k[2], k[3])).collect::<Vec<_>>().join(","),
            c.dial.map_or("null".to_owned(), |d| d.to_string()),
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
    fn a_scrolled_list_gives_only_what_shows_of_its_rows() {
        let mut ui = Ui::new();
        let root = ui.root();
        let list = ui.add_box(
            root,
            style! { display: flex; flex-direction: column; height: 100px; overflow-y: auto; },
        );
        let rows: Vec<NodeId> = (0..5)
            .map(|i| {
                let r = ui.add_box(list, style! { height: 40px; flex-shrink: 0; });
                ui.set_semantics(r, Semantics::new(Role::Button).label(format!("Row {i}")));
                r
            })
            .collect();
        ui.layout(200.0, 300.0, &mut FixedWidthText(7.0));
        ui.update_style(list, |s| s.scroll_top = 50.0);
        ui.layout(200.0, 300.0, &mut FixedWidthText(7.0));
        let c = controls(&ui);
        // row 0 (−50..−10) scrolled away; row 1 (−10..30) cut at the top;
        // row 3 (70..110) cut at the foot; row 4 (110..150) below
        assert_eq!(c.iter().map(|c| c.node).collect::<Vec<_>>(), rows[1..4]);
        assert_eq!((c[0].rect.y, c[0].rect.height), (0.0, 30.0));
        assert_eq!((c[2].rect.y, c[2].rect.height), (70.0, 30.0));
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
        crate::pickers::search_field(
            &mut ui,
            &mut input,
            &String::new(),
            root,
            "Search regions",
            q,
        );
        ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
        let c = controls(&ui);
        let field = c
            .iter()
            .find(|c| c.role == Role::SearchBox)
            .expect("the search box");
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
        crate::pickers::slider(
            &mut ui,
            &mut input,
            &0.5,
            root,
            "Opacity",
            v,
            0.0..=1.0,
            0.1,
        );
        ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
        let c = controls(&ui);
        let s = c
            .iter()
            .find(|c| c.role == Role::Slider)
            .expect("the slider");
        assert!(s.slider && !s.editable);
        let json = to_json(&c);
        assert!(
            json.contains("\"slide\":true") && json.contains("\"tint\":["),
            "{json}"
        );
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
