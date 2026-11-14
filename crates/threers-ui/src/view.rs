//! A UI described from state: each frame (or when
//! the state changes) the app says what the UI should be as a tree of
//! [`El`]s, and a [`View`] changes the nodes it built last time to match —
//! adding, removing and reordering children (matched by [`key`](El::key),
//! else by place and kind), and setting only the styles, text, semantics
//! and handlers that differ. Nothing is edited by hand, so nothing can be
//! left out of step with the state.
//!
//! ```
//! use threers_ui::input::Input;
//! use threers_ui::view::{El, View};
//! use threers_ui::{FixedWidthText, Role, Ui};
//!
//! #[derive(Default)]
//! struct Todo { items: Vec<String> }
//!
//! fn describe(s: &Todo) -> Vec<El<Todo>> {
//!     let mut rows: Vec<El<Todo>> = s.items.iter().map(|i| El::div("padding: 4px").key(i.as_str()).text(i.as_str())).collect();
//!     rows.push(El::div("padding: 4px").role(Role::Button, "Add").text("Add").on_click(|s: &mut Todo, _| s.items.push(format!("item {}", s.items.len()))));
//!     rows
//! }
//!
//! let mut ui = Ui::new();
//! let mut input: Input<Todo> = Input::new();
//! let mut todo = Todo::default();
//! let root = ui.root();
//! let mut view = View::new(&mut ui, root, "display: flex; flex-direction: column");
//! view.update(&mut ui, &mut input, describe(&todo));
//! ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
//! let add = ui.accessibility_tree().into_iter().find(|a| a.name == "Add").unwrap();
//! input.set_focus(&mut ui, &mut todo, Some(add.id));
//! input.simulate_keystrokes(&mut ui, &mut todo, "enter enter").unwrap();
//! view.update(&mut ui, &mut input, describe(&todo));
//! assert_eq!(ui.children(view.root()).len(), 3);
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::environment::Token;
use crate::input::Input;
use crate::tree::{NodeId, Ui};
use crate::{Role, Semantics, Style};

type Handler<S> = Rc<dyn Fn(&mut S, &mut Ui)>;

/// What a node should be.
pub struct El<S> {
    kind: Kind,
    css: String,
    key: Option<String>,
    semantics: Option<Semantics>,
    hover: Option<String>,
    tokens: Vec<(&'static str, Token)>,
    on_click: Option<Handler<S>>,
    children: Vec<El<S>>,
}

#[derive(Debug, Clone, PartialEq)]
enum Kind {
    Box,
    Text(String),
}

impl<S: 'static> El<S> {
    /// A box styled by `css` (declarations, as `style!` takes them).
    pub fn div(css: &str) -> Self {
        Self {
            kind: Kind::Box,
            css: css.to_owned(),
            key: None,
            semantics: None,
            hover: None,
            tokens: vec![],
            on_click: None,
            children: vec![],
        }
    }

    /// Text, styled as the box it is in.
    pub fn label(text: impl Into<String>) -> Self {
        Self {
            kind: Kind::Text(text.into()),
            ..Self::div("")
        }
    }

    /// Who it is among its siblings, kept as they come and go and move
    /// (a row's id): it is then changed, not rebuilt.
    pub fn key(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }

    pub fn child(mut self, child: El<S>) -> Self {
        self.children.push(child);
        self
    }

    pub fn children(mut self, children: impl IntoIterator<Item = El<S>>) -> Self {
        self.children.extend(children);
        self
    }

    /// A text child.
    pub fn text(self, text: impl Into<String>) -> Self {
        self.child(El::label(text))
    }

    pub fn semantics(mut self, semantics: Semantics) -> Self {
        self.semantics = Some(semantics);
        self
    }

    /// Its role and name for assistive technology.
    pub fn role(self, role: Role, name: impl Into<String>) -> Self {
        self.semantics(Semantics::new(role).label(name.into()))
    }

    /// Declarations while the pointer is over it.
    pub fn hover(mut self, css: &str) -> Self {
        self.hover = Some(css.to_owned());
        self
    }

    /// A CSS colour property by purpose (see [`Ui::token_style`]).
    pub fn token(mut self, property: &'static str, token: Token) -> Self {
        self.tokens.push((property, token));
        self
    }

    /// Called when it is clicked, tapped, or pressed with Enter or Space
    /// (as its role allows).
    pub fn on_click(mut self, f: impl Fn(&mut S, &mut Ui) + 'static) -> Self {
        self.on_click = Some(Rc::new(f));
        self
    }

    /// Its declarations, its hover's and every descendant's, rewritten by
    /// `f` (a dark palette for a tree written in light colours, say).
    pub fn map_css(mut self, f: &dyn Fn(&str) -> String) -> Self {
        self.css = f(&self.css);
        self.hover = self.hover.map(|h| f(&h));
        self.children = self.children.into_iter().map(|c| c.map_css(f)).collect();
        self
    }
}

/// What was built for an [`El`], to compare the next one with.
struct Built {
    node: NodeId,
    kind: Kind,
    key: Option<String>,
    css: String,
    semantics: Option<Semantics>,
    hover: Option<String>,
    tokens: Vec<(&'static str, Token)>,
    clickable: bool,
    children: Vec<Built>,
}

/// The nodes built from [`El`]s under a root, kept in step by
/// [`update`](Self::update).
pub struct View<S> {
    root: NodeId,
    built: Vec<Built>,
    handlers: Rc<RefCell<HashMap<NodeId, Handler<S>>>>,
    /// What the last tracked description read.
    reads: Option<crate::model::Reads>,
}

fn style_of(css: &str) -> Style {
    let mut s = Style::default();
    for (name, value) in crate::states::declarations(css) {
        crate::css::apply(&mut s, &name, &value);
    }
    s
}

impl<S: 'static> View<S> {
    /// A box under `parent`, styled by `css`, to hold what is described.
    pub fn new(ui: &mut Ui, parent: NodeId, css: &str) -> Self {
        let root = ui.add_box(parent, style_of(css));
        Self {
            root,
            built: vec![],
            handlers: Rc::default(),
            reads: None,
        }
    }

    pub fn root(&self) -> NodeId {
        self.root
    }

    /// Restyles the root (as [`new`](Self::new) styled it).
    pub fn set_root_css(&self, ui: &mut Ui, css: &str) {
        ui.set_style(self.root, style_of(css));
    }

    /// Describes the children with `describe` and makes the root's children
    /// match — but only if a [`Model`](crate::model::Model) it read last
    /// time has changed since (or on the first call): reads through models
    /// are tracked, so a view built from models costs nothing until they
    /// change. One that reads no model is described every time. Returns
    /// whether it was.
    pub fn update_tracked(
        &mut self,
        ui: &mut Ui,
        input: &mut Input<S>,
        describe: impl FnOnce() -> Vec<El<S>>,
    ) -> bool {
        if self
            .reads
            .as_ref()
            .is_some_and(|r| !r.is_empty() && !r.changed())
        {
            return false;
        }
        let (children, reads) = crate::model::track(describe);
        self.reads = Some(reads);
        self.update(ui, input, children);
        true
    }

    /// Makes the root's children what `children` describe.
    pub fn update(&mut self, ui: &mut Ui, input: &mut Input<S>, children: Vec<El<S>>) {
        let old = std::mem::take(&mut self.built);
        self.built = self.reconcile(ui, input, self.root, old, children);
        let live = &self.built;
        fn walk(b: &[Built], out: &mut Vec<NodeId>) {
            for x in b {
                out.push(x.node);
                walk(&x.children, out);
            }
        }
        let mut nodes = vec![];
        walk(live, &mut nodes);
        self.handlers.borrow_mut().retain(|n, _| nodes.contains(n));
    }

    fn reconcile(
        &mut self,
        ui: &mut Ui,
        input: &mut Input<S>,
        parent: NodeId,
        old: Vec<Built>,
        new: Vec<El<S>>,
    ) -> Vec<Built> {
        let mut old: Vec<Option<Built>> = old.into_iter().map(Some).collect();
        let mut out = Vec::with_capacity(new.len());
        for (i, el) in new.into_iter().enumerate() {
            // By key, else the next unkeyed one of the same kind in place.
            let found = match &el.key {
                Some(k) => old
                    .iter()
                    .position(|b| b.as_ref().is_some_and(|b| b.key.as_ref() == Some(k))),
                None => old.iter().position(|b| {
                    b.as_ref().is_some_and(|b| {
                        b.key.is_none()
                            && std::mem::discriminant(&b.kind) == std::mem::discriminant(&el.kind)
                    })
                }),
            };
            let built = match found.and_then(|f| old[f].take()) {
                Some(b) => self.patch(ui, input, b, el),
                None => self.create(ui, input, parent, el),
            };
            // In its place among the parent's children.
            let at = ui.children(parent).iter().position(|&c| c == built.node);
            if at != Some(i) {
                ui.move_child(parent, built.node, i);
            }
            out.push(built);
        }
        for b in old.into_iter().flatten() {
            ui.remove(b.node);
        }
        out
    }

    fn create(&mut self, ui: &mut Ui, input: &mut Input<S>, parent: NodeId, el: El<S>) -> Built {
        let node = match &el.kind {
            Kind::Box => ui.add_box(parent, style_of(&el.css)),
            Kind::Text(t) => ui.add_text(parent, t.clone()),
        };
        let mut b = Built {
            node,
            kind: el.kind.clone(),
            key: el.key.clone(),
            css: el.css.clone(),
            semantics: None,
            hover: None,
            tokens: vec![],
            clickable: false,
            children: vec![],
        };
        self.apply(ui, input, &mut b, &el, true);
        b.children = self.reconcile(ui, input, node, vec![], el.children);
        b
    }

    fn patch(&mut self, ui: &mut Ui, input: &mut Input<S>, mut b: Built, el: El<S>) -> Built {
        if b.kind != el.kind {
            if let Kind::Text(t) = &el.kind {
                ui.set_text(b.node, t.clone());
            }
            b.kind = el.kind.clone();
        }
        if b.css != el.css && el.kind == Kind::Box {
            ui.set_style(b.node, style_of(&el.css));
            b.css = el.css.clone();
        }
        self.apply(ui, input, &mut b, &el, false);
        let old = std::mem::take(&mut b.children);
        b.children = self.reconcile(ui, input, b.node, old, el.children);
        b
    }

    /// Semantics, hover, tokens and the click handler, where they differ.
    fn apply(&mut self, ui: &mut Ui, input: &mut Input<S>, b: &mut Built, el: &El<S>, new: bool) {
        if b.semantics != el.semantics {
            // Gone too: a box that had a name and now has none must not
            // keep announcing the old one.
            match &el.semantics {
                Some(s) => ui.set_semantics(b.node, s.clone()),
                None => ui.clear_semantics(b.node),
            }
            b.semantics = el.semantics.clone();
        }
        if b.hover != el.hover {
            if let Some(h) = &el.hover {
                ui.hover_style(b.node, h);
            }
            b.hover = el.hover.clone();
        }
        if b.tokens != el.tokens && !el.tokens.is_empty() {
            ui.token_style(b.node, &el.tokens);
            b.tokens = el.tokens.clone();
        }
        match &el.on_click {
            Some(f) => {
                self.handlers.borrow_mut().insert(b.node, f.clone());
                if !b.clickable || new {
                    let handlers = self.handlers.clone();
                    let node = b.node;
                    input.on_click(node, move |s, ui, _| {
                        let f = handlers.borrow().get(&node).cloned();
                        if let Some(f) = f {
                            f(s, ui);
                        }
                    });
                    b.clickable = true;
                }
            }
            None => {
                self.handlers.borrow_mut().remove(&b.node);
            }
        }
    }
}
