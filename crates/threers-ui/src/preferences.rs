//! Values passed up the tree: a node states a preference (a screen its
//! title, a list how many rows it has, a field that it needs the
//! keyboard), and an ancestor reads those stated inside it — the nearest
//! or deepest, or all of them in order — to show them or act on them.
//! The reverse of the [environment](crate::environment), which flows down.
//!
//! ```
//! use threers_ui::{style, Ui};
//!
//! #[derive(Clone, Debug, PartialEq)]
//! struct Title(&'static str);
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let screen = ui.add_box(root, style! {});
//! let detail = ui.add_box(screen, style! {});
//! ui.set_preference(screen, Title("Inbox"));
//! ui.set_preference(detail, Title("Message"));
//! assert_eq!(ui.preference::<Title>(root), Some(Title("Message")), "the deepest wins");
//! assert_eq!(ui.preferences::<Title>(root).len(), 2);
//! ```

use std::any::{Any, TypeId};
use std::rc::Rc;

use crate::tree::{NodeId, Ui};

impl Ui {
    /// States `value` at `node`, replacing a `T` stated there before.
    pub fn set_preference<T: 'static>(&mut self, node: NodeId, value: T) {
        let key = TypeId::of::<T>();
        let list = &mut self.prefs;
        list.retain(|(n, k, _)| !(*n == node && *k == key));
        list.push((node, key, Rc::new(value)));
    }

    /// Takes back the `T` stated at `node`.
    pub fn clear_preference<T: 'static>(&mut self, node: NodeId) {
        let key = TypeId::of::<T>();
        self.prefs.retain(|(n, k, _)| !(*n == node && *k == key));
    }

    /// Every `T` stated at `node` or inside it (and shown), in tree order.
    pub fn preferences<T: Clone + 'static>(&self, node: NodeId) -> Vec<(NodeId, T)> {
        let key = TypeId::of::<T>();
        let mut out = vec![];
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            if !self.contains(n) || self.style(n).display == crate::Display::None {
                continue;
            }
            for (m, k, v) in &self.prefs {
                if *m == n && *k == key
                    && let Some(v) = (v.as_ref() as &dyn Any).downcast_ref::<T>()
                {
                    out.push((n, v.clone()));
                }
            }
            stack.extend(self.children(n).iter().rev().copied());
        }
        out
    }

    /// The deepest `T` stated inside `node`: the last in tree order, as
    /// the innermost screen's title wins.
    pub fn preference<T: Clone + 'static>(&self, node: NodeId) -> Option<T> {
        self.preferences::<T>(node).pop().map(|(_, v)| v)
    }
}
