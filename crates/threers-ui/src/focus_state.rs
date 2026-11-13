//! Focus kept as state: which of a set of nodes has focus is a value in
//! the state, written as focus moves and moving focus when it is set (to
//! put the cursor in a form's first empty field, or take it out to put a
//! phone's keyboard away). And where focus goes first when a part of the
//! UI shows.
//!
//! ```
//! use threers_ui::input::Input;
//! use threers_ui::{binding, style, Role, Semantics, Ui};
//!
//! #[derive(Clone, Copy, Debug, PartialEq)]
//! enum Field { Name, Email }
//! #[derive(Default)]
//! struct S { focus: Option<Field> }
//!
//! let mut ui = Ui::new();
//! let mut input: Input<S> = Input::new();
//! let mut s = S::default();
//! let root = ui.root();
//! let name = ui.add_box(root, style! {});
//! let email = ui.add_box(root, style! {});
//! for n in [name, email] {
//!     ui.set_semantics(n, Semantics::new(Role::TextBox));
//! }
//! input.bind_focus(binding!(|s: S| s.focus), vec![(Field::Name, name), (Field::Email, email)]);
//! s.focus = Some(Field::Email);
//! input.tick(&mut ui, &mut s, 0.0);
//! assert_eq!(input.focused(), Some(email));
//! input.set_focus(&mut ui, &mut s, Some(name));
//! input.tick(&mut ui, &mut s, 0.1);
//! assert_eq!(s.focus, Some(Field::Name));
//! ```

use crate::binding::Binding;
use crate::input::Input;
use crate::tree::NodeId;

impl<S: 'static> Input<S> {
    /// Keeps `focus` (one of `fields`' keys, or `None`) and which of their
    /// nodes has focus the same, both ways.
    pub fn bind_focus<K: Clone + PartialEq + 'static>(&mut self, focus: Binding<S, Option<K>>, fields: Vec<(K, NodeId)>) {
        let mut agreed: Option<Option<K>> = None;
        self.on_after_dispatch(move |input, ui, s| {
            let has = input.focused().and_then(|f| fields.iter().find(|(_, n)| *n == f).map(|(k, _)| k.clone()));
            let want = focus.get(s);
            if agreed.as_ref() != Some(&want) {
                // The state moved it: focus follows.
                let node = want.as_ref().and_then(|k| fields.iter().find(|(f, _)| f == k).map(|(_, n)| *n)).filter(|&n| ui.contains(n));
                if node.is_some() || has.is_some() {
                    input.set_focus(ui, s, node);
                }
                agreed = Some(want);
            } else if has != want {
                // Focus moved: the state follows.
                focus.set(s, has.clone());
                agreed = Some(has);
            }
            true
        });
    }

    /// When `scope` shows (on the next dispatch or tick), focus goes to
    /// `node` unless it is already inside `scope`.
    pub fn default_focus(&mut self, scope: NodeId, node: NodeId) {
        self.on_appear(scope, move |input, ui, s| {
            let inside = input.focused().is_some_and(|f| ui.contains(f) && ui.ancestors(f).any(|a| a == scope));
            if !inside && ui.contains(node) {
                input.set_focus(ui, s, Some(node));
            }
        });
    }
}
