//! Links into a UI, as `#fragment` links into a page: a node can be given
//! a name ([`Ui::set_anchor`], as an element its `id`), and a heading has
//! one already, its text as a slug ("Gauges and charts" is
//! `gauges-and-charts`). [`Input::go_to_anchor`] scrolls to the node a name
//! finds and moves focus there, as a browser does for a fragment: a screen
//! reader reads it, and Tab goes on from it. The DOM mirror
//! (`a11y_dom`, in the browser) gives each anchored element the name as
//! its `id`, so links in the page reach it too.
//!
//! ```
//! use threers_ui::input::Input;
//! use threers_ui::{style, FixedWidthText, Role, Semantics, Ui};
//!
//! let mut ui = Ui::new();
//! let mut input: Input<()> = Input::new();
//! let root = ui.root();
//! let h = ui.add_box(root, style! { height: 30px; });
//! ui.add_text(h, "Gauges and charts");
//! ui.set_semantics(h, Semantics::new(Role::Heading(2)));
//! let save = ui.add_box(root, style! { height: 30px; });
//! ui.set_anchor(save, "save");
//! ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
//! assert_eq!(ui.find_anchor("gauges-and-charts"), Some(h));
//! assert_eq!(input.go_to_anchor(&mut ui, &mut (), "save"), Some(save));
//! assert_eq!(input.focused(), Some(save));
//! ```

use crate::input::Input;
use crate::tree::{NodeId, Ui};
use crate::{Display, Role};

/// `text` as an anchor name: lower case, letters and digits kept (in any
/// script), everything else one `-` between them.
pub fn slug(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars().flat_map(char::to_lowercase) {
        if c.is_alphanumeric() {
            out.push(c);
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

impl Ui {
    /// Names `node` for links (`#name`), as an element's `id`; an empty
    /// name takes it off.
    pub fn set_anchor(&mut self, node: NodeId, name: impl Into<String>) {
        let name = name.into();
        if name.is_empty() {
            self.env.anchors.remove(&node);
        } else {
            self.env.anchors.insert(node, name);
        }
    }

    /// The name a link reaches `node` by: its own, else a heading's text as
    /// a [`slug`].
    pub fn anchor_name(&self, node: NodeId) -> Option<String> {
        if let Some(a) = self.env.anchors.get(&node) {
            return Some(a.clone());
        }
        match self.semantics(node) {
            Some(s) if matches!(s.role, Role::Heading(_)) => {
                let text = s.label.clone().unwrap_or_else(|| self.text_content(node));
                Some(slug(&text)).filter(|s| !s.is_empty())
            }
            _ => None,
        }
    }

    /// The node `name` reaches: of those that show, one given it, else the
    /// first heading whose text it is; if none shows, the same among the
    /// rest.
    pub fn find_anchor(&self, name: &str) -> Option<NodeId> {
        self.find_anchor_in(self.root(), name)
    }

    /// [`find_anchor`](Self::find_anchor) among `scope` and what is inside
    /// it (a page's section, where another page has one of the same name).
    pub fn find_anchor_in(&self, scope: NodeId, name: &str) -> Option<NodeId> {
        let name = name.trim_start_matches('#');
        if name.is_empty() {
            return None;
        }
        let inside = |n: NodeId| self.contains(n) && self.ancestors(n).any(|a| a == scope);
        let mut own: Vec<NodeId> = self.env.anchors.iter().filter(|(n, a)| a.as_str() == name && inside(**n)).map(|(n, _)| *n).collect();
        own.sort();
        // Headings, in tree order.
        let mut headings = vec![];
        let mut stack = vec![scope];
        while let Some(n) = stack.pop() {
            if matches!(self.semantics(n).map(|s| s.role), Some(Role::Heading(_))) && !self.env.anchors.contains_key(&n) && self.anchor_name(n).as_deref() == Some(name) {
                headings.push(n);
            }
            stack.extend(self.children(n).iter().rev().copied());
        }
        // What shows first, a name given before a heading's; then the rest.
        let all = || own.iter().chain(&headings).copied();
        all().find(|&n| self.anchor_shows(n)).or_else(|| all().next())
    }

    /// Every name a link can reach, with its node, in tree order: what a
    /// table of contents lists.
    pub fn anchors(&self) -> Vec<(String, NodeId)> {
        let mut out = vec![];
        let mut stack = vec![self.root()];
        while let Some(n) = stack.pop() {
            if let Some(a) = self.anchor_name(n) {
                out.push((a, n));
            }
            stack.extend(self.children(n).iter().rev().copied());
        }
        out
    }

    fn anchor_shows(&self, node: NodeId) -> bool {
        self.ancestors(node).all(|a| self.style(a).display != Display::None)
    }
}

impl<S> Input<S> {
    /// Follows a link to `name` (a leading `#` is fine): scrolls the node
    /// it reaches to the top of the view and moves focus there, as a browser does for a
    /// fragment, so assistive technology reads it and Tab goes on from it.
    /// Returns the node, or `None` if no anchor has the name. (Lay out
    /// first: it scrolls to where the node was laid out.)
    pub fn go_to_anchor(&mut self, ui: &mut Ui, state: &mut S, name: &str) -> Option<NodeId> {
        let node = ui.find_anchor(name)?;
        ui.scroll_into_view_with(node, crate::scrolling::ScrollStrategy::Top);
        self.set_focus(ui, state, Some(node));
        Some(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{style, FixedWidthText, Semantics};

    #[test]
    fn slugs_keep_letters_and_digits_of_any_script() {
        assert_eq!(slug("Gauges and charts"), "gauges-and-charts");
        assert_eq!(slug("  3D: a model, to turn!  "), "3d-a-model-to-turn");
        assert_eq!(slug("Größe — Ünïcode"), "größe-ünïcode");
        assert_eq!(slug("---"), "");
    }

    #[test]
    fn a_link_scrolls_to_its_node_and_focuses_it() {
        let mut ui = Ui::new();
        let mut input: Input<()> = Input::new();
        let root = ui.root();
        let list = ui.add_box(root, style! { height: 100px; overflow: scroll; flex-direction: column; });
        let mut rows = vec![];
        for i in 0..20 {
            let r = ui.add_box(list, style! { height: 40px; flex-shrink: 0; });
            ui.add_text(r, format!("Section {i}"));
            ui.set_semantics(r, Semantics::new(Role::Heading(2)));
            rows.push(r);
        }
        // A hidden twin: the shown one wins.
        let hidden = ui.add_box(root, style! { display: none; });
        ui.set_anchor(hidden, "section-15");
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        assert_eq!(ui.find_anchor("#section-15"), Some(rows[15]));
        assert_eq!(input.go_to_anchor(&mut ui, &mut (), "section-15"), Some(rows[15]));
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        let (r, view) = (ui.rect(rows[15]), ui.rect(list));
        assert!(r.y >= view.y - 0.5 && r.y + r.height <= view.y + view.height + 0.5, "{r:?} in {view:?}");
        assert_eq!(input.focused(), Some(rows[15]));
        assert_eq!(input.go_to_anchor(&mut ui, &mut (), "nowhere"), None);
        // In a scope, and listed.
        assert_eq!(ui.find_anchor_in(list, "section-2"), Some(rows[2]));
        assert_eq!(ui.find_anchor_in(hidden, "section-2"), None);
        assert_eq!(ui.anchors().iter().filter(|(a, _)| a.starts_with("section-")).count(), 21);
        // In the accessibility tree.
        let node = ui.accessibility_tree().into_iter().find(|a| a.id == rows[3]).unwrap();
        assert_eq!(node.anchor.as_deref(), Some("section-3"));
    }
}
