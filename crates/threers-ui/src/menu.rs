//! Menus: a context menu opened where the
//! pointer is (on [`Input::on_context_menu`]), and a popover menu opened
//! from a button ([`popover_menu`]). Items are actions with an optional
//! shortcut label (from [`Keymap::label_for`](crate::keymap::Keymap::label_for)),
//! separators, headers and submenus; disabled items show but are skipped.
//!
//! The keyboard works it as a platform's does: up and down move the
//! highlight (past separators and disabled items), right opens a submenu
//! and left closes it, Enter picks, Escape closes, and focus goes back to
//! where it was. The pointer highlights what it is over and picks with a
//! click; a click outside closes the menu.
//!
//! ```
//! use threers_ui::input::Input;
//! use threers_ui::menu::{open_menu, MenuAt, MenuItem};
//! use threers_ui::{FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let mut input: Input<Vec<String>> = Input::new();
//! let mut picked = vec![];
//! let items = vec![
//!     MenuItem::action("copy", "Copy").shortcut("⌘C"),
//!     MenuItem::separator(),
//!     MenuItem::action("delete", "Delete").disabled(true),
//! ];
//! let menu = open_menu(&mut ui, &mut input, &mut picked, MenuAt::Point(10.0, 10.0), items, |log, _, id| log.push(id.to_owned()));
//! ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
//! input.simulate_keystrokes(&mut ui, &mut picked, "down enter").unwrap();
//! assert_eq!(picked, ["copy"]);
//! assert!(!menu.is_open(&ui));
//! ```

use std::cell::RefCell;
use std::rc::Rc;

use crate::input::{Event, Input, Phase};
use crate::layers::{Anchor, AnchorTo, Corner, Fit};
use crate::tree::{NodeId, Ui};
use crate::{style, Role, Semantics};

/// An item of a menu.
#[derive(Debug, Clone, PartialEq)]
pub enum MenuItem {
    /// Something to do: `id` is handed to the menu's `on_pick`.
    /// `destructive`: it deletes or discards, and reads so (in red).
    Action { id: String, label: String, shortcut: Option<String>, disabled: bool, destructive: bool },
    Separator,
    /// A title over the items after it.
    Header(String),
    /// A menu inside the menu.
    Submenu { label: String, items: Vec<MenuItem> },
}

impl MenuItem {
    pub fn action(id: impl Into<String>, label: impl Into<String>) -> Self {
        MenuItem::Action { id: id.into(), label: label.into(), shortcut: None, disabled: false, destructive: false }
    }

    pub fn separator() -> Self {
        MenuItem::Separator
    }

    pub fn header(label: impl Into<String>) -> Self {
        MenuItem::Header(label.into())
    }

    pub fn submenu(label: impl Into<String>, items: Vec<MenuItem>) -> Self {
        MenuItem::Submenu { label: label.into(), items }
    }

    /// Shows `label` (`⌘C`) at the item's end.
    pub fn shortcut(mut self, label: impl Into<String>) -> Self {
        if let MenuItem::Action { shortcut, .. } = &mut self {
            *shortcut = Some(label.into());
        }
        self
    }

    /// Marks it as one that deletes or discards.
    pub fn destructive(mut self, on: bool) -> Self {
        if let MenuItem::Action { destructive, .. } = &mut self {
            *destructive = on;
        }
        self
    }

    pub fn disabled(mut self, on: bool) -> Self {
        if let MenuItem::Action { disabled, .. } = &mut self {
            *disabled = on;
        }
        self
    }

    /// Whether the keyboard and the pointer can land on it.
    fn active(&self) -> bool {
        matches!(self, MenuItem::Action { disabled: false, .. } | MenuItem::Submenu { .. })
    }
}

/// Where a menu opens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MenuAt {
    /// Its top-left corner at a point: where the pointer was.
    Point(f32, f32),
    /// Under a node (a button), its left edges lined up.
    Below(NodeId),
    /// Beside a node (a submenu's item), its top edges lined up.
    Beside(NodeId),
}

type Pick<S> = Rc<RefCell<dyn FnMut(&mut S, &mut Ui, &str)>>;

struct Open {
    root: NodeId,
    items: Vec<MenuItem>,
    rows: Vec<NodeId>,
    highlighted: Option<usize>,
    /// A submenu open from it, and which item it is from.
    child: Option<(usize, Rc<RefCell<Open>>)>,
    /// Where focus was before the menu took it.
    focus_before: Option<NodeId>,
    /// Asked for by its listeners, done after the event (with `Input`):
    /// closing; an item picked (the whole menu closes); a submenu to open,
    /// and whether it takes focus (opened from the keyboard).
    closing: bool,
    picked: bool,
    submenu_wanted: Option<usize>,
    submenu_focus: bool,
}

/// An open menu.
#[derive(Clone)]
pub struct MenuHandle {
    open: Rc<RefCell<Open>>,
}

impl MenuHandle {
    pub fn is_open(&self, ui: &Ui) -> bool {
        ui.contains(self.open.borrow().root)
    }

    /// The menu's box.
    pub fn root(&self) -> NodeId {
        self.open.borrow().root
    }

    /// Closes it (and its submenus), giving focus back.
    pub fn close<S: 'static>(&self, ui: &mut Ui, input: &mut Input<S>, state: &mut S) {
        close(&self.open, ui, input, state);
    }
}

/// The menu's box and its open submenus' boxes.
fn roots(open: &Rc<RefCell<Open>>) -> Vec<NodeId> {
    let o = open.borrow();
    let mut out = vec![o.root];
    if let Some((_, c)) = &o.child {
        out.extend(roots(c));
    }
    out
}

fn close<S: 'static>(open: &Rc<RefCell<Open>>, ui: &mut Ui, input: &mut Input<S>, state: &mut S) {
    // Focus in it or in a submenu of it goes back where it was before.
    let all = roots(open);
    let had_focus = input.focused().is_some_and(|f| ui.contains(f) && ui.ancestors(f).any(|a| all.contains(&a)));
    let (root, child, before) = {
        let mut o = open.borrow_mut();
        (o.root, o.child.take(), o.focus_before)
    };
    if let Some((_, c)) = child {
        close(&c, ui, input, state);
    }
    if ui.contains(root) {
        ui.remove(root);
        if had_focus {
            input.set_focus(ui, state, before.filter(|&b| ui.contains(b)));
        }
    }
}

/// Opens a menu of `items` at `at`, taking focus for its keys. `on_pick`
/// gets the id of the item picked; the menu closes then, on Escape, and on
/// a click outside it.
pub fn open_menu<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &mut S,
    at: MenuAt,
    items: Vec<MenuItem>,
    on_pick: impl FnMut(&mut S, &mut Ui, &str) + 'static,
) -> MenuHandle {
    let pick: Pick<S> = Rc::new(RefCell::new(on_pick));
    let open = build(ui, input, at, items, pick, input.focused(), None);
    let root = open.borrow().root;
    input.set_focus(ui, state, Some(root));
    MenuHandle { open }
}

/// Makes `trigger` (a button) open a menu below it when clicked, and close
/// it when clicked again. `items` builds the menu each
/// time it opens.
pub fn popover_menu<S: 'static>(
    input: &mut Input<S>,
    trigger: NodeId,
    mut items: impl FnMut(&S) -> Vec<MenuItem> + 'static,
    on_pick: impl FnMut(&mut S, &mut Ui, &str) + 'static,
) {
    let pick: Pick<S> = Rc::new(RefCell::new(on_pick));
    let shown: Rc<RefCell<Option<MenuHandle>>> = Rc::default();
    // Its listeners need `&mut Input`, which a listener does not get: the
    // trigger asks, and the next `Input::tick` or dispatch opens it.
    let wanted: Rc<RefCell<Option<Vec<MenuItem>>>> = Rc::default();
    let (w, s) = (wanted.clone(), shown.clone());
    input.on_click(trigger, move |state: &mut S, ui, _| {
        let open = s.borrow().as_ref().is_some_and(|m| m.is_open(ui));
        if open {
            if let Some(m) = s.borrow().as_ref() {
                m.open.borrow_mut().closing = true;
            }
        } else {
            *w.borrow_mut() = Some(items(state));
        }
    });
    input.on_after_dispatch(move |input, ui, state| {
        // Kept for as long as the button is there.
        let Some(items) = wanted.borrow_mut().take() else { return ui.contains(trigger) };
        let p = pick.clone();
        let open = build(ui, input, MenuAt::Below(trigger), items, p, input.focused(), Some(trigger));
        let root = open.borrow().root;
        input.set_focus(ui, state, Some(root));
        *shown.borrow_mut() = Some(MenuHandle { open });
        ui.contains(trigger)
    });
}

fn build<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    at: MenuAt,
    items: Vec<MenuItem>,
    pick: Pick<S>,
    focus_before: Option<NodeId>,
    owner: Option<NodeId>,
) -> Rc<RefCell<Open>> {
    let parent = ui.root();
    let root = ui.add_box(
        parent,
        style! {
            position: absolute; display: flex; flex-direction: column; min-width: 160px; padding: 4px;
            border-radius: 8px; background: #22262f; border: 1px solid #3a4150;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5); font-size: 13px; color: #e5e7eb;
        },
    );
    ui.set_semantics(root, Semantics::new(Role::Menu).tab_index(-1));
    // In the light appearance, a light menu (see `Ui::appearance_style`).
    ui.appearance_style(root, crate::states::Appearance::Light, "background: #ffffff; border: 1px solid #d5d9e0; color: #1f2329");
    ui.defer(root, 1000);
    let anchor = match at {
        MenuAt::Point(x, y) => Anchor::new(AnchorTo::Point(x, y), Corner::TopLeft),
        MenuAt::Below(n) => Anchor::new(AnchorTo::Node(n, Corner::BottomLeft), Corner::TopLeft).offset(0.0, 4.0),
        MenuAt::Beside(n) => Anchor::new(AnchorTo::Node(n, Corner::TopRight), Corner::TopLeft).offset(2.0, -5.0),
    };
    ui.anchor(root, anchor.fit(Fit::Flip));
    let mut rows = vec![];
    for (i, item) in items.iter().enumerate() {
        let row = match item {
            MenuItem::Separator => {
                let s = ui.add_box(root, style! { height: 1px; margin: 4px 6px; background: #3a4150; });
                ui.appearance_style(s, crate::states::Appearance::Light, "background: #e3e6eb");
                s
            }
            MenuItem::Header(label) => {
                let h = ui.add_box(root, style! { padding: 6px 10px 2px; font-size: 11px; font-weight: bold; color: #8b93a5; });
                ui.add_text(h, label.clone());
                h
            }
            MenuItem::Action { label, shortcut, disabled, destructive, .. } => {
                let r = ui.add_box(root, style! { display: flex; gap: 24px; justify-content: space-between; padding: 6px 10px; border-radius: 5px; });
                let t = ui.add_text(r, label.clone());
                if *destructive {
                    let mut st = ui.style(t).clone();
                    st.color = ui.token(crate::environment::Token::Destructive);
                    ui.set_style(t, st);
                }
                if let Some(s) = shortcut {
                    let k = ui.add_box(r, style! { color: #8b93a5; });
                    ui.add_text(k, s.clone());
                }
                let mut sem = Semantics::new(Role::MenuItem).label(label.clone());
                sem.disabled = *disabled;
                ui.set_semantics(r, sem);
                if *disabled {
                    ui.style_mut(r).color = [0.45, 0.48, 0.55, 1.0];
                }
                r
            }
            MenuItem::Submenu { label, .. } => {
                let r = ui.add_box(root, style! { display: flex; gap: 24px; justify-content: space-between; padding: 6px 10px; border-radius: 5px; });
                ui.add_text(r, label.clone());
                let k = ui.add_box(r, style! { color: #8b93a5; });
                ui.add_text(k, "›");
                ui.set_semantics(r, Semantics::new(Role::MenuItem).label(label.clone()));
                r
            }
        };
        let _ = i;
        rows.push(row);
    }
    let open = Rc::new(RefCell::new(Open {
        root,
        items,
        rows: rows.clone(),
        highlighted: None,
        child: None,
        focus_before,
        closing: false,
        picked: false,
        submenu_wanted: None,
        submenu_focus: false,
    }));
    // The pointer: over an item, it is highlighted (and a submenu opens);
    // a click picks it.
    for (i, &row) in rows.iter().enumerate() {
        if !open.borrow().items[i].active() {
            continue;
        }
        let o = open.clone();
        input.on(row, Phase::Bubble, move |_, ui, e, _| {
            if let Event::PointerEnter = e {
                highlight(ui, &o, Some(i));
                o.borrow_mut().want_submenu(i);
            }
        });
        let (o, p) = (open.clone(), pick.clone());
        input.on_click(row, move |state, ui, cx| {
            activate(state, ui, &o, &p, i);
            cx.stop_propagation();
        });
    }
    // The keyboard.
    let (o, p) = (open.clone(), pick.clone());
    input.on_key(root, move |state, ui, key, _, _| {
        let n = o.borrow().items.len();
        let step = |from: Option<usize>, down: bool| -> Option<usize> {
            let items = &o.borrow().items;
            let mut i = from;
            for _ in 0..n {
                let next = match (i, down) {
                    (None, true) => 0,
                    (None, false) => n - 1,
                    (Some(k), true) => (k + 1) % n,
                    (Some(k), false) => (k + n - 1) % n,
                };
                i = Some(next);
                if items[next].active() {
                    return Some(next);
                }
            }
            None
        };
        match key {
            "ArrowDown" | "ArrowUp" => {
                let h = step(o.borrow().highlighted, key == "ArrowDown");
                highlight(ui, &o, h);
            }
            "Home" => highlight(ui, &o, step(None, true)),
            "End" => highlight(ui, &o, step(None, false)),
            "Enter" | " " | "ArrowRight" => {
                let target = {
                    let b = o.borrow();
                    b.highlighted.filter(|&i| key != "ArrowRight" || matches!(b.items[i], MenuItem::Submenu { .. }))
                };
                if let Some(i) = target {
                    activate(state, ui, &o, &p, i);
                } else if key != "ArrowRight" {
                    return false;
                }
            }
            "Escape" | "ArrowLeft" => {
                // Closed by the frame (it needs `Input`): asked for here.
                o.borrow_mut().closing = true;
            }
            _ => return false,
        }
        true
    });
    // A click outside closes it (but not one in a submenu of it, nor on
    // the button that opened it: that click toggles it itself).
    let o = open.clone();
    input.on_click_outside(root, move |_, ui, e, _| {
        let (x, y) = match *e {
            Event::PointerDown { x, y, .. } | Event::PointerUp { x, y, .. } | Event::Click { x, y, .. } => (x, y),
            _ => return,
        };
        let hit = ui.hit_test(x, y);
        let inside = |n: NodeId| hit.is_some_and(|h| ui.contains(n) && ui.ancestors(h).any(|a| a == n));
        let in_child = roots(&o).iter().skip(1).any(|&r| inside(r));
        if !in_child && !owner.is_some_and(inside) {
            o.borrow_mut().closing = true;
        }
    });
    // What the listeners asked for that needs `Input`: closing, submenus.
    let (o, p) = (open.clone(), pick.clone());
    input.on_after_dispatch(move |input, ui, state| service(input, ui, state, &o, &p));
    open
}

/// Does what the menu's listeners asked for. Returns whether it is still
/// open.
fn service<S: 'static>(input: &mut Input<S>, ui: &mut Ui, state: &mut S, open: &Rc<RefCell<Open>>, pick: &Pick<S>) -> bool {
    if !ui.contains(open.borrow().root) {
        return false;
    }
    // A submenu that closed itself (Left, Escape) is gone.
    let child_gone = open.borrow().child.as_ref().is_some_and(|(_, c)| !ui.contains(c.borrow().root));
    if child_gone {
        open.borrow_mut().child = None;
    }
    let child_picked = open.borrow().child.as_ref().is_some_and(|(_, c)| c.borrow().picked);
    let (closing, picked, wanted, focus) = {
        let mut o = open.borrow_mut();
        (std::mem::take(&mut o.closing), o.picked, o.submenu_wanted.take(), std::mem::take(&mut o.submenu_focus))
    };
    if picked || child_picked {
        open.borrow_mut().picked = true;
        close(open, ui, input, state);
        return false;
    }
    if closing {
        close(open, ui, input, state);
        return false;
    }
    if let Some(i) = wanted {
        let current = open.borrow().child.as_ref().map(|(k, _)| *k);
        if current != Some(i) {
            if let Some((_, c)) = open.borrow_mut().child.take() {
                close(&c, ui, input, state);
            }
            let (item, row, root) = {
                let o = open.borrow();
                (o.items[i].clone(), o.rows[i], o.root)
            };
            if let MenuItem::Submenu { items, .. } = item {
                let child = build(ui, input, MenuAt::Beside(row), items, pick.clone(), Some(root), None);
                open.borrow_mut().child = Some((i, child));
            }
        }
        if focus && let Some((_, c)) = open.borrow().child.clone() {
            // Opened from the keys: into it, on its first item.
            let root = c.borrow().root;
            input.set_focus(ui, state, Some(root));
            let first = c.borrow().items.iter().position(MenuItem::active);
            highlight(ui, &c, first);
        }
    }
    true
}

impl Open {
    fn want_submenu(&mut self, i: usize) {
        self.submenu_wanted = Some(i);
    }
}

fn highlight(ui: &mut Ui, open: &Rc<RefCell<Open>>, to: Option<usize>) {
    let mut o = open.borrow_mut();
    if o.highlighted == to {
        return;
    }
    if let Some(r) = o.highlighted.and_then(|h| o.rows.get(h).copied()).filter(|&r| ui.contains(r)) {
        let st = ui.style_mut(r);
        st.background = None;
        // Back to the menu's colour (the default inherits it).
        st.color = [1.0; 4];
    }
    if let Some(r) = to.and_then(|h| o.rows.get(h).copied()).filter(|&r| ui.contains(r)) {
        let st = ui.style_mut(r);
        st.background = Some([0.145, 0.388, 0.922, 1.0]);
        // White on the blue, in either appearance (not the default white,
        // which would inherit the menu's).
        st.color = [0.99, 0.99, 0.99, 1.0];
    }
    o.highlighted = to;
}

fn activate<S: 'static>(state: &mut S, ui: &mut Ui, open: &Rc<RefCell<Open>>, pick: &Pick<S>, i: usize) {
    let item = open.borrow().items[i].clone();
    match item {
        MenuItem::Action { id, disabled: false, .. } => {
            (pick.borrow_mut())(state, ui, &id);
            // The whole menu closes, from the top.
            open.borrow_mut().picked = true;
        }
        MenuItem::Submenu { .. } => {
            open.borrow_mut().submenu_wanted = Some(i);
            open.borrow_mut().submenu_focus = true;
        }
        _ => {}
    }
}
