//! The accessibility tree in the page, for screen readers. **wasm32 only.**
//!
//! A canvas is one opaque image to assistive technology. [`DomAccessibility`]
//! mirrors a [`Ui`]'s [`accessibility_tree`](Ui::accessibility_tree) into
//! elements over it — ARIA roles, names, values and states, each element on
//! its node's box, so a screen reader's cursor and a magnifier land where
//! the control is drawn. They are transparent and take no pointer events:
//! the mouse still reaches the canvas. What assistive technology does to
//! them comes back as [`A11yAction`]s for the app to apply:
//!
//! ```no_run
//! # #[cfg(target_arch = "wasm32")]
//! # fn demo(canvas: web_sys::HtmlCanvasElement, ui: &threers_ui::Ui) {
//! use threers_ui::a11y_dom::{A11yAction, DomAccessibility};
//! let mut mirror = DomAccessibility::new(&canvas).unwrap();
//! // Each frame, after layout:
//! mirror.update(ui);
//! for action in mirror.take_actions() {
//!     match action {
//!         A11yAction::Activate(node) => { /* as a click on `node` */ }
//!         A11yAction::Focus(node) => { /* draw the focus ring there */ }
//!         _ => {}
//!     }
//! }
//! # }
//! ```
//!
//! Keyboard focus goes through the page as a page's would: Tab and Shift+Tab
//! move between the focusable elements in reading order, and an app that
//! moves focus itself calls [`DomAccessibility::focus`] so the screen reader
//! follows.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;
use web_sys::{Element, HtmlCanvasElement, HtmlElement};

use super::a11y::{AccessNode, Role};
use super::tree::{NodeId, Ui};

/// What assistive technology (or the keyboard, through the mirror) asked
/// of a node.
#[derive(Debug, Clone, PartialEq)]
pub enum A11yAction {
    /// Focus moved to it.
    Focus(NodeId),
    /// Pressed: a click, Enter or Space (a checkbox toggles, a button acts).
    Activate(NodeId),
    /// One of its custom actions, by name (see
    /// [`Input::accessibility_action`](crate::input::Input::accessibility_action)).
    Custom(NodeId, String),
    /// A slider stepped up (arrow up or right) or down.
    Increment(NodeId),
    Decrement(NodeId),
}

/// A node's element, the attributes it was last given, and its text.
struct Mirrored {
    element: Element,
    attributes: Vec<(String, String)>,
    text: String,
    actions: Vec<String>,
}

/// The mirror: one element per node of the accessibility tree.
pub struct DomAccessibility {
    root: Element,
    /// The canvas mirrored, and the placement copied from its `style`: the
    /// mirror follows when the page moves or resizes the canvas.
    canvas: HtmlCanvasElement,
    placed: String,
    elements: HashMap<NodeId, Mirrored>,
    actions: Rc<RefCell<Vec<A11yAction>>>,
    _listeners: Vec<Closure<dyn FnMut(web_sys::Event)>>,
    /// Live regions the UI's announcements are read through: polite and
    /// assertive; and the last announcement read.
    live: Option<(Element, Element)>,
    announced: u64,
}

/// The attributes a node's element carries, sorted by name.
fn attributes(n: &AccessNode, parent: Option<&AccessNode>) -> Vec<(String, String)> {
    let mut a: Vec<(String, String)> = Vec::new();
    let mut put = |k: &str, v: String| a.push((k.to_owned(), v));
    let (dx, dy) = parent.map_or((0.0, 0.0), |p| (p.bounds.x, p.bounds.y));
    put(
        "style",
        format!(
            "position:absolute;left:{}px;top:{}px;width:{}px;height:{}px;margin:0;padding:0;border:0;\
             background:transparent;color:transparent;outline:none;overflow:visible;white-space:pre;\
             font-size:1px;pointer-events:none",
            n.bounds.x - dx,
            n.bounds.y - dy,
            n.bounds.width.max(1.0),
            n.bounds.height.max(1.0)
        ),
    );
    put("data-threers-node", format!("{}:{}", n.id.0, n.id.1));
    match n.role {
        // Plain text is read as text; it needs no role.
        Role::StaticText => {}
        Role::Heading(level) => {
            put("role", "heading".into());
            put("aria-level", level.clamp(1, 6).to_string());
        }
        Role::TreeItem(level) => {
            put("role", "treeitem".into());
            put("aria-level", level.max(1).to_string());
        }
        Role::TextArea => {
            put("role", "textbox".into());
            put("aria-multiline", "true".into());
        }
        r => put("role", r.aria().into()),
    }
    if !n.name.is_empty() && n.role != Role::StaticText {
        put("aria-label", n.name.clone());
    }
    if let (Some(d), None) = (&n.description, &n.invalid) {
        put("aria-description", d.clone());
    }
    if let Some(c) = n.checked {
        put("aria-checked", c.to_string());
    }
    if let Some(s) = n.selected {
        put("aria-selected", s.to_string());
    }
    if let Some(e) = n.expanded {
        put("aria-expanded", e.to_string());
    }
    if let Some(r) = n.range {
        put("aria-valuemin", r.min.to_string());
        put("aria-valuemax", r.max.to_string());
        put("aria-valuenow", r.now.to_string());
    }
    if n.disabled {
        put("aria-disabled", "true".into());
    }
    if n.focusable {
        put("tabindex", "0".into());
    }
    // A link's target: reached by `#name`, and able to take focus when
    // the link moves it there (not by Tab).
    if let Some(a) = &n.anchor {
        put("id", a.clone());
        if !n.focusable {
            put("tabindex", "-1".into());
        }
    }
    if n.live {
        put(
            "aria-live",
            if n.role == Role::Alert {
                "assertive"
            } else {
                "polite"
            }
            .into(),
        );
    }
    if let (Some(r), Some(v)) = (n.range, &n.value) {
        // A value read as words ("50%", "Medium"), beside the number.
        if v != &r.now.to_string() {
            put("aria-valuetext", v.clone());
        }
    }
    if let Some((pos, size)) = n.position {
        put("aria-posinset", pos.to_string());
        put("aria-setsize", size.to_string());
    }
    if n.busy {
        put("aria-busy", "true".into());
    }
    if let Some(why) = &n.invalid {
        put("aria-invalid", "true".into());
        // Read with it: there is no element of its own to point at.
        let d = n
            .description
            .as_ref()
            .map_or_else(|| why.clone(), |d| format!("{d}. {why}"));
        put("aria-description", d);
    }
    if n.current {
        put("aria-current", "true".into());
    }
    if let Some(k) = &n.shortcut {
        put("aria-keyshortcuts", k.clone());
    }
    if n.modal {
        put("aria-modal", "true".into());
    }
    // A row says where it is; the table, how many rows it has.
    match (n.role, n.row_index) {
        (Role::Table, Some((_, count))) => put("aria-rowcount", count.to_string()),
        (_, Some((i, _))) => put("aria-rowindex", i.to_string()),
        _ => {}
    }
    if let Some(up) = n.sort {
        put(
            "aria-sort",
            if up { "ascending" } else { "descending" }.into(),
        );
    }
    if matches!(n.role, Role::TextBox | Role::TextArea) && !n.focusable {
        put("aria-readonly", "true".into());
    }
    a.sort();
    a
}

/// What a node's element holds as text: static text's words, a text box's
/// or combo box's value (which ARIA reads from the content).
fn content(n: &AccessNode) -> String {
    match n.role {
        Role::StaticText => n.name.clone(),
        Role::TextBox | Role::TextArea | Role::ComboBox => n.value.clone().unwrap_or_default(),
        _ => String::new(),
    }
}

fn node_of(target: Option<web_sys::EventTarget>) -> Option<NodeId> {
    let e: Element = target?.dyn_into().ok()?;
    let e = e.closest("[data-threers-node]").ok()??;
    let attr = e.get_attribute("data-threers-node")?;
    let (slot, generation) = attr.split_once(':')?;
    Some(NodeId(slot.parse().ok()?, generation.parse().ok()?))
}

/// Where the canvas is, as its `style` attribute places it.
fn placement(canvas: &HtmlCanvasElement) -> String {
    canvas
        .get_attribute("style")
        .unwrap_or_else(|| "position:absolute;left:0;top:0;width:100%;height:100%".into())
}

impl DomAccessibility {
    /// A mirror over `canvas`: a layer in its parent, covering the box its
    /// `style` attribute gives it (as [`CanvasPainter::with_dom_text`]'s
    /// layers do). `None` without a parent to put it in.
    ///
    /// [`CanvasPainter::with_dom_text`]: super::canvas::CanvasPainter::with_dom_text
    pub fn new(canvas: &HtmlCanvasElement) -> Option<Self> {
        let document = web_sys::window()?.document()?;
        // `after` needs the canvas in a parent; without one, no mirror.
        canvas.parent_element()?;
        let root = document.create_element("div").ok()?;
        let place = placement(canvas);
        root.set_attribute(
            "style",
            &format!("{place};pointer-events:none;overflow:hidden"),
        )
        .ok()?;
        root.set_attribute("data-threers-a11y", "").ok()?;
        canvas.after_with_node_1(&root).ok()?;
        let actions: Rc<RefCell<Vec<A11yAction>>> = Rc::default();
        let mut listeners = Vec::new();
        let listen = |root: &Element,
                      kind: &str,
                      f: Box<dyn FnMut(web_sys::Event)>,
                      out: &mut Vec<Closure<dyn FnMut(web_sys::Event)>>| {
            let c = Closure::wrap(f);
            let _ = root.add_event_listener_with_callback(kind, c.as_ref().unchecked_ref());
            out.push(c);
        };
        let a = actions.clone();
        listen(
            &root,
            "focusin",
            Box::new(move |e: web_sys::Event| {
                if let Some(id) = node_of(e.target()) {
                    a.borrow_mut().push(A11yAction::Focus(id));
                }
            }),
            &mut listeners,
        );
        let a = actions.clone();
        // A screen reader's "press" is a click on the element.
        listen(
            &root,
            "click",
            Box::new(move |e: web_sys::Event| {
                // One of its custom actions' buttons.
                let action = e
                    .target()
                    .and_then(|t| t.dyn_into::<Element>().ok())
                    .and_then(|t| t.get_attribute("data-threers-action"));
                if let Some(id) = node_of(e.target()) {
                    a.borrow_mut().push(match action {
                        Some(name) => A11yAction::Custom(id, name),
                        None => A11yAction::Activate(id),
                    });
                }
            }),
            &mut listeners,
        );
        let a = actions.clone();
        listen(
            &root,
            "keydown",
            Box::new(move |e: web_sys::Event| {
                let Some(k) = e.dyn_ref::<web_sys::KeyboardEvent>() else {
                    return;
                };
                let Some(id) = node_of(e.target()) else {
                    return;
                };
                let target: Option<Element> = e.target().and_then(|t| t.dyn_into().ok());
                let role = target
                    .and_then(|t| t.get_attribute("role"))
                    .unwrap_or_default();
                let action = match (k.key().as_str(), role.as_str()) {
                    ("Enter" | " ", "slider") => None,
                    ("Enter" | " ", _) => Some(A11yAction::Activate(id)),
                    ("ArrowUp" | "ArrowRight", "slider" | "separator") => {
                        Some(A11yAction::Increment(id))
                    }
                    ("ArrowDown" | "ArrowLeft", "slider" | "separator") => {
                        Some(A11yAction::Decrement(id))
                    }
                    _ => None,
                };
                if let Some(action) = action {
                    e.prevent_default();
                    a.borrow_mut().push(action);
                }
            }),
            &mut listeners,
        );
        // Announcements are read through two live regions, kept apart from
        // the nodes' elements (which come and go).
        let region = |politeness: &str| -> Option<Element> {
            let e = document.create_element("div").ok()?;
            let _ = e.set_attribute("aria-live", politeness);
            let _ = e.set_attribute("aria-atomic", "true");
            let _ = e.set_attribute("style", "position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap");
            root.append_child(&e).ok()?;
            Some(e)
        };
        let live = region("polite").zip(region("assertive"));
        Some(Self {
            root,
            canvas: canvas.clone(),
            placed: place,
            elements: HashMap::new(),
            actions,
            _listeners: listeners,
            live,
            announced: 0,
        })
    }

    /// Brings the mirror up to `ui`'s accessibility tree as last laid out:
    /// elements added, changed and removed so it matches, the ones that did
    /// not change left alone (and focused ones keep their focus).
    pub fn update(&mut self, ui: &Ui) {
        let place = placement(&self.canvas);
        if place != self.placed {
            let _ = self.root.set_attribute(
                "style",
                &format!("{place};pointer-events:none;overflow:hidden"),
            );
            self.placed = place;
        }
        let tree = ui.accessibility_tree();
        let Some(document) = web_sys::window().and_then(|w| w.document()) else {
            return;
        };
        // New announcements, read through the live regions.
        let new = ui.announcements_since(self.announced);
        if let (Some((polite, assertive)), Some(last)) = (&self.live, new.last()) {
            self.announced = last.0;
            for (_, message, urgent) in new {
                let region = if urgent { assertive } else { polite };
                // Emptied first: the same message twice is read twice.
                region.set_text_content(None);
                region.set_text_content(Some(&message));
            }
        }
        let mut seen = std::collections::HashSet::new();
        // Parents before children: the tree's list is in that order.
        let mut parent_of: Vec<Option<usize>> = vec![None; tree.len()];
        for (i, n) in tree.iter().enumerate() {
            for &c in &n.children {
                parent_of[c] = Some(i);
            }
        }
        for (i, n) in tree.iter().enumerate().skip(1) {
            let parent = parent_of[i].filter(|&p| p != 0).map(|p| &tree[p]);
            let attrs = attributes(n, parent);
            let text = content(n);
            seen.insert(n.id);
            let container = match parent {
                Some(p) => self.elements.get(&p.id).map(|e| e.element.clone()),
                None => Some(self.root.clone()),
            };
            let Some(container) = container else { continue };
            let entry = match self.elements.remove(&n.id) {
                Some(e) => e,
                None => {
                    let Ok(element) = document.create_element("div") else {
                        continue;
                    };
                    Mirrored {
                        element,
                        attributes: Vec::new(),
                        text: String::new(),
                        actions: Vec::new(),
                    }
                }
            };
            let Mirrored {
                element,
                attributes: old_attrs,
                text: old_text,
                actions: old_actions,
            } = entry;
            if old_actions != n.actions {
                // Its custom actions, as buttons inside it: a screen reader
                // offers them, and their clicks come back as actions.
                if let Ok(Some(old)) = element.query_selector(":scope > [data-threers-actions]") {
                    old.remove();
                }
                if !n.actions.is_empty()
                    && let Ok(holder) = document.create_element("span")
                {
                    let _ = holder.set_attribute("data-threers-actions", "");
                    // Offered to a screen reader, not shown: the canvas draws
                    // the node.
                    let _ = holder.set_attribute("style", "position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap");
                    for name in &n.actions {
                        if let Ok(b) = document.create_element("button") {
                            let _ = b.set_attribute("data-threers-action", name);
                            let _ = b.set_attribute("tabindex", "-1");
                            b.set_text_content(Some(name));
                            let _ = holder.append_child(&b);
                        }
                    }
                    let _ = element.append_child(&holder);
                }
            }
            if old_attrs != attrs {
                for (k, _) in &old_attrs {
                    if !attrs.iter().any(|(n, _)| n == k) {
                        let _ = element.remove_attribute(k);
                    }
                }
                for (k, v) in &attrs {
                    if !old_attrs.iter().any(|(n, o)| n == k && o == v) {
                        let _ = element.set_attribute(k, v);
                    }
                }
            }
            if old_text != text {
                // Text goes in its own child, so the element's children
                // (other nodes' elements) are not replaced with it.
                let holder = match element
                    .first_element_child()
                    .filter(|c| c.has_attribute("data-threers-text"))
                {
                    Some(h) => h,
                    None => {
                        let Ok(h) = document.create_element("span") else {
                            continue;
                        };
                        let _ = h.set_attribute("data-threers-text", "");
                        let _ = element.prepend_with_node_1(&h);
                        h
                    }
                };
                holder.set_text_content(Some(&text));
            }
            if element.parent_element().as_ref() != Some(&container) {
                let _ = container.append_child(&element);
            }
            self.elements.insert(
                n.id,
                Mirrored {
                    element,
                    attributes: attrs,
                    text,
                    actions: n.actions.clone(),
                },
            );
        }
        // Gone from the tree: gone from the page.
        self.elements.retain(|id, m| {
            let keep = seen.contains(id);
            if !keep {
                m.element.remove();
            }
            keep
        });
    }

    /// Moves the page's focus to `node`'s element, so assistive technology
    /// follows focus the app moved itself (a Tab handled on the canvas).
    pub fn focus(&self, node: NodeId) {
        if let Some(m) = self.elements.get(&node)
            && let Some(h) = m.element.dyn_ref::<HtmlElement>()
        {
            let _ = h.focus();
        }
    }

    /// The element mirroring `node`, if it has one.
    pub fn element(&self, node: NodeId) -> Option<&Element> {
        self.elements.get(&node).map(|m| &m.element)
    }

    /// What was asked of the mirror since last time, oldest first.
    pub fn take_actions(&mut self) -> Vec<A11yAction> {
        std::mem::take(&mut self.actions.borrow_mut())
    }
}

impl Drop for DomAccessibility {
    fn drop(&mut self) {
        self.root.remove();
    }
}
