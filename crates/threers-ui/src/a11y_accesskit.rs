//! The accessibility tree for native screen readers, through
//! [AccessKit](https://accesskit.dev): VoiceOver on macOS and iOS, Narrator
//! and other UI Automation clients on Windows, Orca on Linux. AccessKit's
//! platform adapters (`accesskit_winit` for a winit window) take the
//! [`tree_update`] each frame and hand back [`ActionRequest`]s, which
//! [`request`] turns into what this crate's input understands.
//!
//! It is `a11y_dom`'s counterpart outside a browser (that module exists
//! only in wasm builds): the
//! same [`Ui::accessibility_tree`], told to the platform rather than to the
//! page.
//!
//! ```
//! use threers_ui::a11y_accesskit::tree_update;
//! use threers_ui::{style, FixedWidthText, Role, Semantics, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let ok = ui.add_box(root, style! { width: 60px; height: 20px; });
//! ui.set_semantics(ok, Semantics::new(Role::Button).label("OK"));
//! ui.layout(200.0, 100.0, &mut FixedWidthText(7.0));
//! let update = tree_update(&ui, Some(ok));
//! assert_eq!(update.focus, threers_ui::a11y_accesskit::node_id(ok));
//! assert!(update.nodes.iter().any(|(_, n)| n.label() == Some("OK")));
//! ```

pub use accesskit::{Action, ActionRequest};

use super::a11y::Role;
use super::tree::{NodeId, Ui};

/// The AccessKit id of a node: its slot and generation, so an id never
/// names a node that took a removed one's place.
pub fn node_id(id: NodeId) -> accesskit::NodeId {
    accesskit::NodeId(((id.0 as u64) << 32) | id.1 as u64)
}

/// The node an AccessKit id names, if it is still in the tree.
pub fn ui_node(ui: &Ui, id: accesskit::NodeId) -> Option<NodeId> {
    let n = NodeId((id.0 >> 32) as usize, id.0 as u32);
    ui.contains(n).then_some(n)
}

fn role(r: Role) -> accesskit::Role {
    use accesskit::Role as A;
    match r {
        Role::None | Role::Generic => A::GenericContainer,
        Role::Group => A::Group,
        Role::StaticText => A::Label,
        Role::Heading(_) => A::Heading,
        Role::Button => A::Button,
        Role::Link => A::Link,
        Role::Checkbox => A::CheckBox,
        Role::Radio => A::RadioButton,
        Role::RadioGroup => A::RadioGroup,
        Role::ComboBox => A::ComboBox,
        Role::ListBox => A::ListBox,
        Role::Option => A::ListBoxOption,
        Role::Slider => A::Slider,
        Role::TextBox => A::TextInput,
        Role::TextArea => A::MultilineTextInput,
        Role::Image => A::Image,
        Role::List => A::List,
        Role::ListItem => A::ListItem,
        Role::Dialog => A::Dialog,
        Role::Tab => A::Tab,
        Role::TabList => A::TabList,
        Role::TabPanel => A::TabPanel,
        Role::Toolbar => A::Toolbar,
        Role::Menu => A::Menu,
        Role::MenuItem => A::MenuItem,
        Role::Status => A::Status,
        Role::Alert => A::Alert,
        Role::Region => A::Region,
        Role::Switch => A::Switch,
        Role::Separator => A::Splitter,
        Role::AlertDialog => A::AlertDialog,
        Role::SpinButton => A::SpinButton,
        Role::ProgressBar => A::ProgressIndicator,
        Role::SearchBox => A::SearchInput,
        Role::Navigation => A::Navigation,
        Role::Table => A::Table,
        Role::Row => A::Row,
        Role::Cell => A::Cell,
        Role::ColumnHeader => A::ColumnHeader,
    }
}

/// The whole accessibility tree as an AccessKit update, with `focus` (the
/// root when `None`) as the focused node. Send one whenever the tree or
/// focus changes; AccessKit works out what changed.
pub fn tree_update(ui: &Ui, focus: Option<NodeId>) -> accesskit::TreeUpdate {
    let tree = ui.accessibility_tree();
    let mut nodes = Vec::with_capacity(tree.len());
    for a in &tree {
        let mut n = accesskit::Node::new(role(a.role));
        if !a.name.is_empty() {
            n.set_label(a.name.clone());
        }
        if let Some(d) = &a.description {
            n.set_description(d.clone());
        }
        if let Some(v) = &a.value {
            n.set_value(v.clone());
        }
        if let Role::Heading(level) = a.role {
            n.set_level(level as usize);
        }
        if let Some(r) = a.range {
            n.set_numeric_value(r.now as f64);
            n.set_min_numeric_value(r.min as f64);
            n.set_max_numeric_value(r.max as f64);
            n.add_action(Action::Increment);
            n.add_action(Action::Decrement);
        }
        if let Some(c) = a.checked {
            n.set_toggled(if c {
                accesskit::Toggled::True
            } else {
                accesskit::Toggled::False
            });
        }
        if let Some(s) = a.selected {
            n.set_selected(s);
        }
        if let Some(e) = a.expanded {
            n.set_expanded(e);
        }
        if a.disabled {
            n.set_disabled();
        }
        if a.live {
            n.set_live(if a.role == Role::Alert {
                accesskit::Live::Assertive
            } else {
                accesskit::Live::Polite
            });
        }
        if let Some((pos, size)) = a.position {
            n.set_position_in_set(pos);
            n.set_size_of_set(size);
        }
        if a.busy {
            n.set_busy();
        }
        if a.invalid.is_some() {
            n.set_invalid(accesskit::Invalid::True);
        }
        if a.current {
            n.set_aria_current(accesskit::AriaCurrent::True);
        }
        if let Some(k) = &a.shortcut {
            n.set_keyboard_shortcut(k.clone());
        }
        if a.modal {
            n.set_modal();
        }
        match (a.role, a.row_index) {
            (Role::Table, Some((_, count))) => n.set_row_count(count),
            (_, Some((i, _))) => n.set_row_index(i.saturating_sub(1)),
            _ => {}
        }
        if let Some(up) = a.sort {
            n.set_sort_direction(if up {
                accesskit::SortDirection::Ascending
            } else {
                accesskit::SortDirection::Descending
            });
        }
        if a.focusable && !a.disabled {
            n.add_action(Action::Focus);
            n.add_action(Action::Click);
        }
        let b = a.bounds;
        n.set_bounds(accesskit::Rect {
            x0: b.x as f64,
            y0: b.y as f64,
            x1: (b.x + b.width) as f64,
            y1: (b.y + b.height) as f64,
        });
        n.set_children(
            a.children
                .iter()
                .map(|&i| node_id(tree[i].id))
                .collect::<Vec<_>>(),
        );
        nodes.push((node_id(a.id), n));
    }
    let root = node_id(ui.root());
    let mut info = accesskit::TreeInfo::new(root);
    info.toolkit_name = Some("threers-ui".into());
    info.toolkit_version = Some(env!("CARGO_PKG_VERSION").into());
    accesskit::TreeUpdate {
        nodes,
        tree: Some(info),
        tree_id: accesskit::TreeId::ROOT,
        focus: focus.filter(|&f| ui.contains(f)).map_or(root, node_id),
    }
}

/// What a screen reader asked for, in this crate's terms.
#[derive(Debug, Clone, PartialEq)]
pub enum AccessRequest {
    /// Move focus there ([`Input::set_focus`](super::input::Input::set_focus)).
    Focus(NodeId),
    /// Press it: dispatch a click, as Enter on it would.
    Activate(NodeId),
    /// Step a slider up or down.
    Increment(NodeId),
    Decrement(NodeId),
}

/// An AccessKit action request as an [`AccessRequest`], if it is one this
/// crate acts on and its node still exists.
pub fn request(ui: &Ui, req: &ActionRequest) -> Option<AccessRequest> {
    if req.target_tree != accesskit::TreeId::ROOT {
        return None;
    }
    let n = ui_node(ui, req.target_node)?;
    Some(match req.action {
        Action::Focus => AccessRequest::Focus(n),
        Action::Click => AccessRequest::Activate(n),
        Action::Increment => AccessRequest::Increment(n),
        Action::Decrement => AccessRequest::Decrement(n),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{style, FixedWidthText, Semantics};

    #[test]
    fn roles_states_and_ids_reach_accesskit() {
        let mut ui = Ui::new();
        let root = ui.root();
        let dialog = ui.add_box(root, style! { padding: 4px; });
        ui.set_semantics(dialog, Semantics::new(Role::Dialog).label("Settings"));
        let check = ui.add_box(dialog, style! { width: 16px; height: 16px; });
        ui.set_semantics(
            check,
            Semantics::new(Role::Checkbox)
                .label("Shadows")
                .checked(true),
        );
        let slider = ui.add_box(dialog, style! { width: 100px; height: 16px; });
        ui.set_semantics(
            slider,
            Semantics::new(Role::Slider)
                .label("Roughness")
                .range(0.0, 1.0, 0.25),
        );
        ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
        let u = tree_update(&ui, Some(check));
        let find = |id: NodeId| {
            u.nodes
                .iter()
                .find(|(n, _)| *n == node_id(id))
                .map(|(_, n)| n)
                .unwrap()
        };
        assert_eq!(find(dialog).role(), accesskit::Role::Dialog);
        assert_eq!(find(dialog).children(), [node_id(check), node_id(slider)]);
        assert_eq!(find(check).toggled(), Some(accesskit::Toggled::True));
        assert_eq!(find(slider).numeric_value(), Some(0.25));
        assert!(find(slider).supports_action(Action::Increment));
        let b = find(slider).bounds().unwrap();
        assert_eq!((b.x0, b.x1), (4.0, 104.0));
        assert_eq!(u.focus, node_id(check));
        // Requests come back as this crate's, and not for removed nodes.
        let req = ActionRequest {
            action: Action::Click,
            target_tree: accesskit::TreeId::ROOT,
            target_node: node_id(check),
            data: None,
        };
        assert_eq!(request(&ui, &req), Some(AccessRequest::Activate(check)));
        ui.remove(check);
        assert_eq!(request(&ui, &req), None);
    }
}
