//! Container queries: a node styled by the width of a box around it, not of
//! the window, as CSS's `@container` does.
//!
//! A sidebar panel and the same panel in a wide main area want different
//! layouts, and the window's width says nothing about which one this is.
//! [`Ui::container_query`] gives a node styles for widths of a container:
//! after each layout, the node takes the style of the widest breakpoint
//! the container's content box reaches, and if that changed anything the
//! layout runs again with it (at most twice more). As in CSS, the container's
//! own width should not depend on the node's style, or the two could
//! disagree; with containment, the second layout is final.
//!
//! ```
//! use threers_ui::{style, FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let panel = ui.add_box(root, style! { width: 300px; });
//! let cards = ui.add_box(panel, style! { display: flex; flex-direction: column; });
//! ui.container_query(cards, panel, vec![
//!     (400.0, style! { display: flex; flex-direction: row; }),
//! ]);
//! ui.layout(800.0, 600.0, &mut FixedWidthText(7.0));
//! assert_eq!(ui.style(cards).flex_direction, threers_ui::FlexDirection::Column);
//! ui.style_mut(panel).width = threers_ui::Length::Px(500.0);
//! ui.layout(800.0, 600.0, &mut FixedWidthText(7.0));
//! assert_eq!(ui.style(cards).flex_direction, threers_ui::FlexDirection::Row);
//! ```

use super::style::Style;
use super::tree::{NodeId, Ui};

/// A node's styles by container width.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ContainerQuery {
    pub container: NodeId,
    /// The style below every breakpoint.
    pub base: Style,
    /// `(minimum content width, style)`, ascending.
    pub rules: Vec<(f32, Style)>,
}

impl Ui {
    /// Styles `node` by the content width of `container` (usually an
    /// ancestor): below every breakpoint, the style it has now; at or above
    /// a `rules` width, that rule's style (the widest that applies). See
    /// the [module docs](self).
    pub fn container_query(&mut self, node: NodeId, container: NodeId, mut rules: Vec<(f32, Style)>) {
        self.node(container);
        rules.sort_by(|a, b| a.0.total_cmp(&b.0));
        let base = self.style(node).clone();
        self.node_mut(node).query = Some(Box::new(ContainerQuery {
            container,
            base,
            rules,
        }));
        self.touch(node);
    }

    /// Stops styling `node` by its container, leaving the style it has.
    pub fn remove_container_query(&mut self, node: NodeId) {
        self.node_mut(node).query = None;
        self.touch(node);
    }

    /// Gives each queried node the style its container's width calls for.
    /// Returns whether any changed (and so the layout must run again).
    pub(crate) fn apply_container_queries(&mut self) -> bool {
        let queried: Vec<usize> = (0..self.nodes.len())
            .filter(|&i| !self.nodes[i].removed && self.nodes[i].query.is_some())
            .collect();
        let mut changed = false;
        for i in queried {
            let q = self.nodes[i].query.as_ref().expect("filtered").clone();
            if !self.contains(q.container) {
                continue;
            }
            let c = &self.nodes[q.container.0];
            let s = &c.style;
            let width = c.rect.width - s.padding.horizontal() - s.border.horizontal();
            let mut style = q
                .rules
                .iter()
                .rev()
                .find(|(min, _)| width >= *min)
                .map_or(&q.base, |(_, s)| s)
                .clone();
            let id = self.id_at(i);
            // Where a box is scrolled to is its state, not its style.
            style.scroll_top = self.style(id).scroll_top;
            style.scroll_left = self.style(id).scroll_left;
            if *self.style(id) != style {
                self.set_style(id, style);
                changed = true;
            }
        }
        changed
    }
}
