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

mod calendar_words;
mod kit;
mod kit_controls;
mod kit_more;
mod kit_style;
pub use calendar_words::CalendarWords;
pub use kit::*;
pub use kit_controls::*;
pub(crate) use kit_more::calendar_picking;
pub use kit_more::*;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::environment::Token;
use crate::input::{EventCx, Input};
use crate::tree::{NodeId, Ui};
use crate::{Role, Semantics, Style};

type Handler<S> = Rc<dyn Fn(&mut S, &mut Ui)>;
/// Hears a key: whether it used it.
type KeyHandler<S> = Rc<dyn Fn(&mut S, &mut Ui, &str, NodeId, &mut EventCx) -> bool>;
/// What fills a [`El::host`]'s box when it is made.
type Build<S> = Rc<dyn Fn(&mut Ui, &mut Input<S>, &S, NodeId)>;

/// What a node should be.
pub struct El<S> {
    kind: Kind,
    css: String,
    key: Option<String>,
    semantics: Option<Semantics>,
    hover: Option<String>,
    tokens: Vec<(&'static str, Token)>,
    on_click: Option<Handler<S>>,
    on_key: Option<KeyHandler<S>>,
    children: Vec<El<S>>,
    build: Option<Build<S>>,
    help: Option<String>,
    /// A row of a card, parted from the next by a line (see
    /// [`as_row`](Self::as_row)).
    pub(crate) row: bool,
    /// In a column, this much of its side padding pulled back.
    bleed: f32,
    /// Declarations added from a width of the view on.
    wide: Vec<(f32, String)>,
    /// Theme colours painted as something else in it.
    tints: Vec<(Token, crate::fill::Fill)>,
    /// Its text's colours, across each line of it.
    text_gradient: Option<crate::fill::Gradient>,
}

#[derive(Debug, Clone, PartialEq)]
enum Kind {
    Box,
    Text(String),
    /// Built by its [`El::host`] closure, kept as it is while its id stays.
    Host(String),
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
            on_key: None,
            children: vec![],
            build: None,
            help: None,
            row: false,
            bleed: 0.0,
            wide: vec![],
            tints: vec![],
            text_gradient: None,
        }
    }

    /// A row: in a card ([`inset_group`](super::view::inset_group)) there is
    /// a line between it and the row after it.
    pub fn as_row(mut self) -> Self {
        self.row = true;
        self
    }

    /// In a column, its side padding of `px` pulled back, so what is in it
    /// lines up with the text above and below (a text-only button); in a
    /// row it keeps its place.
    pub fn line_up_in_column(mut self, px: f32) -> Self {
        self.bleed = px;
        self
    }

    /// A tooltip of `text` when the pointer rests on it (SwiftUI's
    /// `.help`): what a control does, said where it is.
    pub fn help(mut self, text: impl Into<String>) -> Self {
        self.help = Some(text.into());
        self
    }

    /// A box filled by `build(ui, input, state, box)` when it is made — a
    /// control that keeps state of its own between updates (a text field's
    /// text, caret and focus): later updates keep what it built, and build
    /// again only when `id` changes. Built only through
    /// [`View::update_with`], which has the state to give it.
    pub fn host(
        id: impl Into<String>,
        build: impl Fn(&mut Ui, &mut Input<S>, &S, NodeId) + 'static,
    ) -> Self {
        Self {
            kind: Kind::Host(id.into()),
            build: Some(Rc::new(build)),
            ..Self::div("")
        }
    }

    /// Runs `also` after a [`host`](Self::host)'s build, on the same box
    /// (more listeners, say); nothing on another kind.
    pub fn after_build(
        mut self,
        also: impl Fn(&mut Ui, &mut Input<S>, &S, NodeId) + 'static,
    ) -> Self {
        if let Some(first) = self.build.take() {
            self.build = Some(Rc::new(
                move |ui: &mut Ui, input: &mut Input<S>, s: &S, n: NodeId| {
                    first(ui, input, s, n);
                    also(ui, input, s, n);
                },
            ));
        }
        self
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

    /// Called with a key (`"ArrowDown"`), and the node it is, while focus is
    /// in it; it says whether it used the key.
    pub fn on_key(
        mut self,
        f: impl Fn(&mut S, &mut Ui, &str, NodeId, &mut EventCx) -> bool + 'static,
    ) -> Self {
        self.on_key = Some(Rc::new(f));
        self
    }

    /// Adds `css` where the view it is in is at least `min` pixels wide
    /// (SwiftUI's size classes): written for the narrow window first, a
    /// sidebar can show its words where there is room for them.
    pub fn at_width(mut self, min: f32, css: &str) -> Self {
        self.wide.push((min, css.to_owned()));
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
/// A node's declarations, tokens and [`El::at_width`] rules.
type Queried = (String, Vec<(&'static str, Token)>, Vec<(f32, String)>);

struct Built {
    node: NodeId,
    kind: Kind,
    key: Option<String>,
    css: String,
    semantics: Option<Semantics>,
    hover: Option<String>,
    tokens: Vec<(&'static str, Token)>,
    /// What its width rules were made from.
    queried: Option<Queried>,
    clickable: bool,
    keyed: bool,
    children: Vec<Built>,
}

/// The nodes built from [`El`]s under a root, kept in step by
/// [`update`](Self::update).
pub struct View<S> {
    root: NodeId,
    built: Vec<Built>,
    handlers: Rc<RefCell<HashMap<NodeId, Handler<S>>>>,
    key_handlers: Rc<RefCell<HashMap<NodeId, KeyHandler<S>>>>,
    /// What the last tracked description read.
    reads: Option<crate::model::Reads>,
    /// Hosts are built anew at each update, but for the one with focus.
    rebuild_hosts: bool,
    /// The tooltip text of each node that has one.
    helps: Rc<RefCell<HashMap<NodeId, String>>>,
}

/// `el`'s declarations under `parent`: with its padding pulled back when it
/// asks for that and `parent` is a column.
fn css_in<S>(ui: &Ui, parent: NodeId, el: &El<S>) -> String {
    // `var(--token)` and `color-mix(…)` (in a gradient, a shadow) in the
    // theme's colours as it is built: a theme change builds it anew.
    let css = crate::environment::resolve_colors(&el.css, |t| ui.token(t)).into_owned();
    if el.bleed > 0.0 && ui.style(parent).flex_direction == crate::FlexDirection::Column {
        format!(
            "{css}; margin-left: -{b}px; margin-right: -{b}px",
            b = el.bleed
        )
    } else {
        css
    }
}

/// `tints` and the text's gradient on what is in `node`, once it is there.
fn paint_in(
    ui: &mut Ui,
    node: NodeId,
    tints: &[(Token, crate::fill::Fill)],
    text: Option<&crate::fill::Gradient>,
) {
    for (token, fill) in tints {
        ui.set_fill_in(node, *token, Some(fill.clone()));
    }
    let Some(g) = text else { return };
    let stops = g.resolve(ui);
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        stack.extend_from_slice(ui.children(n));
        let crate::NodeKind::Text(t) = ui.kind(n) else {
            continue;
        };
        // A colour a character, as far along as it is.
        let t = t.clone();
        let last = t.chars().count().saturating_sub(1).max(1) as f32;
        let runs = t
            .chars()
            .enumerate()
            .map(|(i, c)| {
                crate::rich_text::TextRun::new(c.len_utf8())
                    .color(crate::style::gradient_at(&stops, i as f32 / last))
            })
            .collect();
        ui.set_text_runs(n, runs);
    }
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
            key_handlers: Rc::default(),
            rebuild_hosts: false,
            helps: Rc::default(),
            reads: None,
        }
    }

    /// Builds each [`host`](El::host) anew at every update, except one that
    /// has focus or is open (so a field keeps its caret): for a view whose
    /// hosts take what they do from the description (a row's menu), which
    /// a host kept for its id would not follow.
    pub fn rebuilding_hosts(mut self) -> Self {
        self.rebuild_hosts = true;
        self
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

    /// Makes the root's children what `children` describe. ([`El::host`]s
    /// need [`update_with`](Self::update_with).)
    pub fn update(&mut self, ui: &mut Ui, input: &mut Input<S>, children: Vec<El<S>>) {
        self.update_in(ui, input, None, children);
    }

    /// [`update`](Self::update), with the `state` an [`El::host`] reads
    /// when it is built (a field's text to begin with).
    pub fn update_with(
        &mut self,
        ui: &mut Ui,
        input: &mut Input<S>,
        state: &S,
        children: Vec<El<S>>,
    ) {
        self.update_in(ui, input, Some(state), children);
    }

    fn update_in(
        &mut self,
        ui: &mut Ui,
        input: &mut Input<S>,
        state: Option<&S>,
        children: Vec<El<S>>,
    ) {
        let old = std::mem::take(&mut self.built);
        self.built = self.reconcile(ui, input, state, self.root, old, children);
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
        self.key_handlers
            .borrow_mut()
            .retain(|n, _| nodes.contains(n));
        self.helps.borrow_mut().retain(|n, _| nodes.contains(n));
    }

    fn reconcile(
        &mut self,
        ui: &mut Ui,
        input: &mut Input<S>,
        state: Option<&S>,
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
            // (one not kept goes with what is left over, below)
            let found = found.filter(|&f| {
                !(self.rebuild_hosts
                    && matches!(el.kind, Kind::Host(_))
                    && old[f].as_ref().is_some_and(|b| {
                        !input
                            .focused()
                            .is_some_and(|n| ui.contains(n) && ui.ancestors(n).any(|a| a == b.node))
                    }))
            });
            let built = match found.and_then(|f| old[f].take()) {
                Some(b) => self.patch(ui, input, state, b, el),
                None => self.create(ui, input, state, parent, el),
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

    fn create(
        &mut self,
        ui: &mut Ui,
        input: &mut Input<S>,
        state: Option<&S>,
        parent: NodeId,
        el: El<S>,
    ) -> Built {
        let css = css_in(ui, parent, &el);
        let node = match &el.kind {
            Kind::Box | Kind::Host(_) => ui.add_box(parent, style_of(&css)),
            Kind::Text(t) => ui.add_text(parent, t.clone()),
        };
        let mut b = Built {
            node,
            kind: el.kind.clone(),
            key: el.key.clone(),
            css,
            semantics: None,
            hover: None,
            tokens: vec![],
            queried: None,
            clickable: false,
            keyed: false,
            children: vec![],
        };
        self.apply(ui, input, &mut b, &el, true);
        if let Kind::Host(_) = el.kind {
            Self::build(ui, input, state, node, &el);
        } else {
            b.children = self.reconcile(ui, input, state, node, vec![], el.children);
        }
        paint_in(ui, node, &el.tints, el.text_gradient.as_ref());
        b
    }

    /// Fills a host's box, as its closure says.
    fn build(ui: &mut Ui, input: &mut Input<S>, state: Option<&S>, node: NodeId, el: &El<S>) {
        if let (Some(build), Some(state)) = (&el.build, state) {
            build(ui, input, state, node);
        }
    }

    fn patch(
        &mut self,
        ui: &mut Ui,
        input: &mut Input<S>,
        state: Option<&S>,
        mut b: Built,
        el: El<S>,
    ) -> Built {
        if b.kind != el.kind {
            match &el.kind {
                Kind::Text(t) => ui.set_text(b.node, t.clone()),
                // Another host here: what the last one built goes.
                Kind::Host(_) => {
                    ui.clear_children(b.node);
                    Self::build(ui, input, state, b.node, &el);
                }
                Kind::Box => {}
            }
            b.kind = el.kind.clone();
        }
        let css = ui
            .parent(b.node)
            .map_or_else(|| el.css.clone(), |p| css_in(ui, p, &el));
        if b.css != css && !matches!(el.kind, Kind::Text(_)) {
            ui.set_style(b.node, style_of(&css));
            b.css = css;
        }
        self.apply(ui, input, &mut b, &el, false);
        // A host keeps what it built.
        if !matches!(el.kind, Kind::Host(_)) {
            let old = std::mem::take(&mut b.children);
            b.children = self.reconcile(ui, input, state, b.node, old, el.children);
        }
        paint_in(ui, b.node, &el.tints, el.text_gradient.as_ref());
        b
    }

    /// Its styles by the view's width, when they or what they add to differ.
    fn query(&self, ui: &mut Ui, b: &mut Built, el: &El<S>) {
        if el.wide.is_empty() {
            if b.queried.take().is_some() {
                ui.remove_container_query(b.node);
                ui.set_style(b.node, style_of(&b.css));
                ui.token_style(b.node, &el.tokens);
            }
            return;
        }
        let now = (b.css.clone(), el.tokens.clone(), el.wide.clone());
        if b.queried.as_ref() == Some(&now) {
            return;
        }
        // Narrow first, in the theme's colours; each rule on top of that.
        ui.remove_container_query(b.node);
        ui.set_style(b.node, style_of(&b.css));
        ui.token_style(b.node, &el.tokens);
        let base = ui.style(b.node).clone();
        let mut rules = vec![];
        let mut css = b.css.clone();
        let mut wide = el.wide.clone();
        wide.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (min, more) in wide {
            css = format!("{css}; {more}");
            let mut s = style_of(&css);
            for (property, _) in &el.tokens {
                match *property {
                    "color" => s.color = base.color,
                    "background" => s.background = base.background,
                    "border-color" => s.border_color = base.border_color,
                    _ => {}
                }
            }
            rules.push((min, s));
        }
        ui.container_query(b.node, self.root, rules);
        b.queried = Some(now);
    }

    /// Semantics, hover, tokens and the click handler, where they differ.
    fn apply(&mut self, ui: &mut Ui, input: &mut Input<S>, b: &mut Built, el: &El<S>, new: bool) {
        match &el.help {
            Some(text) => {
                let had = self
                    .helps
                    .borrow_mut()
                    .insert(b.node, text.clone())
                    .is_some();
                if !had || new {
                    let (helps, node) = (self.helps.clone(), b.node);
                    input.tooltip(node, move |_, ui, root| {
                        let text = helps.borrow().get(&node).cloned().unwrap_or_default();
                        let tip = ui.add_box(
                            root,
                            crate::style! { padding: 4px 8px; border-radius: 6px; font-size: 12px; max-width: 280px; white-space: normal; },
                        );
                        ui.token_style(tip, &[("background", Token::Raised), ("color", Token::Text)]);
                        ui.add_text(tip, text);
                        tip
                    });
                }
            }
            None => {
                self.helps.borrow_mut().remove(&b.node);
            }
        }
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
            // In place of the one it had.
            ui.set_hover_style(b.node, el.hover.as_deref().unwrap_or(""));
            b.hover = el.hover.clone();
        }
        if b.tokens != el.tokens {
            // Those it no longer has go: a choice not made any more must
            // not stay drawn as made.
            for (property, _) in &b.tokens {
                if !el.tokens.iter().any(|(p, _)| p == property) {
                    ui.clear_token_style(b.node, property);
                }
            }
            if !el.tokens.is_empty() {
                ui.token_style(b.node, &el.tokens);
            }
            b.tokens = el.tokens.clone();
        }
        self.query(ui, b, el);
        match &el.on_key {
            Some(f) => {
                self.key_handlers.borrow_mut().insert(b.node, f.clone());
                if !b.keyed || new {
                    let handlers = self.key_handlers.clone();
                    let node = b.node;
                    input.on_key(node, move |s, ui, key, _, cx| {
                        let f = handlers.borrow().get(&node).cloned();
                        f.is_some_and(|f| f(s, ui, key, node, cx))
                    });
                    b.keyed = true;
                }
            }
            None => {
                self.key_handlers.borrow_mut().remove(&b.node);
            }
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
