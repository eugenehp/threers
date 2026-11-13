//! Accessibility: what a [`Ui`] means, for assistive technology, and checks
//! against WCAG 2.2's success criteria.
//!
//! A `Ui` paints pixels — onto a canvas or the CPU overlay — and pixels
//! say nothing to a screen reader. So nodes carry [`Semantics`]: a
//! [`Role`], an accessible name, a value and states. The controls in
//! [`controls`](super::controls) set theirs; an app sets its own with
//! [`Ui::set_semantics`]. From those:
//!
//! - [`Ui::accessibility_tree`] is the tree assistive technology reads
//!   (WCAG 1.1.1, 1.3.1, 4.1.2): layout boxes with no meaning are left
//!   out and their content lifted to the nearest node that has one, and
//!   each node's name is computed as browsers compute it — its label, or
//!   the text inside it. It is platform-neutral: on the web,
//!   `a11y_dom::DomAccessibility` (wasm32) mirrors it into
//!   the page; natively, a host hands it to the platform (AccessKit takes
//!   the same shape).
//! - [`Ui::focus_order`] and [`Ui::next_focus`] make every focusable node
//!   reachable by keyboard, in reading order (2.1.1, 2.4.3).
//! - [`Ui::contrast_issues`] finds text whose contrast with what is behind
//!   it is under the minimum (1.4.3): 4.5:1, or 3:1 for large text.

use super::style::{Display, FontWeight, Rgba};
use super::tree::{NodeId, NodeKind, Rect, Ui};
pub use threers_captions::contrast::{contrast_ratio, relative_luminance};

/// What a node is, to assistive technology: ARIA's roles, those a UI of
/// boxes needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// Nothing of its own; its content belongs to its parent. The default.
    None,
    Generic,
    Group,
    StaticText,
    Heading(u8),
    Button,
    Link,
    Checkbox,
    Radio,
    RadioGroup,
    /// A drop-down list (`<select>`).
    ComboBox,
    ListBox,
    Option,
    Slider,
    TextBox,
    /// A text box of several lines.
    TextArea,
    Image,
    List,
    ListItem,
    Dialog,
    Tab,
    TabList,
    TabPanel,
    Toolbar,
    Menu,
    MenuItem,
    Status,
    Alert,
    Region,
    /// An on/off switch (`role="switch"`): a checkbox that acts at once.
    Switch,
    /// A divider, or (focusable, with a value) a splitter's handle.
    Separator,
    /// A dialog asking for an answer, which interrupts (`alertdialog`).
    AlertDialog,
    /// A number stepped up and down (`spinbutton`): a stepper.
    SpinButton,
    /// How far along something is (`progressbar`); no value while unknown.
    ProgressBar,
    /// A text box for a search (`searchbox`).
    SearchBox,
    /// A set of links or controls that moves between screens (`navigation`).
    Navigation,
    /// Rows and columns of data (`table`).
    Table,
    /// A row of a table (`row`).
    Row,
    /// A cell of a table (`cell`).
    Cell,
    /// A column's heading (`columnheader`).
    ColumnHeader,
}

impl Role {
    /// The ARIA role name.
    pub fn aria(self) -> &'static str {
        match self {
            Role::None => "none",
            Role::Generic => "generic",
            Role::Group => "group",
            Role::StaticText => "text",
            Role::Heading(_) => "heading",
            Role::Button => "button",
            Role::Link => "link",
            Role::Checkbox => "checkbox",
            Role::Radio => "radio",
            Role::RadioGroup => "radiogroup",
            Role::ComboBox => "combobox",
            Role::ListBox => "listbox",
            Role::Option => "option",
            Role::Slider => "slider",
            Role::TextBox | Role::TextArea => "textbox",
            Role::Image => "img",
            Role::List => "list",
            Role::ListItem => "listitem",
            Role::Dialog => "dialog",
            Role::Tab => "tab",
            Role::TabList => "tablist",
            Role::TabPanel => "tabpanel",
            Role::Toolbar => "toolbar",
            Role::Menu => "menu",
            Role::MenuItem => "menuitem",
            Role::Status => "status",
            Role::Alert => "alert",
            Role::Region => "region",
            Role::Switch => "switch",
            Role::Separator => "separator",
            Role::AlertDialog => "alertdialog",
            Role::SpinButton => "spinbutton",
            Role::ProgressBar => "progressbar",
            Role::SearchBox => "searchbox",
            Role::Navigation => "navigation",
            Role::Table => "table",
            Role::Row => "row",
            Role::Cell => "cell",
            Role::ColumnHeader => "columnheader",
        }
    }

    /// Whether the role takes its name from its content when it has no
    /// label (ARIA's "name from content"): a button's name is its text.
    pub fn named_by_content(self) -> bool {
        matches!(
            self,
            Role::Button
                | Role::Link
                | Role::Checkbox
                | Role::Switch
                | Role::Radio
                | Role::Heading(_)
                | Role::Option
                | Role::ListItem
                | Role::Cell
                | Role::ColumnHeader
                | Role::Tab
                | Role::MenuItem
                | Role::StaticText
        )
    }

    /// Whether a keyboard user reaches it with Tab by default.
    pub fn focusable_by_default(self) -> bool {
        matches!(
            self,
            Role::Button
                | Role::Link
                | Role::Checkbox
                | Role::Switch
                | Role::Radio
                | Role::ComboBox
                | Role::Slider
                | Role::TextBox
                | Role::TextArea
                | Role::Tab
                | Role::MenuItem
                | Role::SpinButton
                | Role::SearchBox
        )
    }
}

/// A slider's (or progress bar's) value: where it is between its ends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeValue {
    pub min: f32,
    pub max: f32,
    pub now: f32,
}

/// What a node means: its role, name and state.
#[derive(Debug, Clone, PartialEq)]
pub struct Semantics {
    pub role: Role,
    /// The accessible name (`aria-label`); `None` takes it from the text
    /// inside, where the role allows.
    pub label: Option<String>,
    /// More about it than the name says (`aria-description`).
    pub description: Option<String>,
    /// A text box's text, a combo box's choice.
    pub value: Option<String>,
    pub range: Option<RangeValue>,
    /// A checkbox's or radio button's state; `None` for roles without one.
    pub checked: Option<bool>,
    pub selected: Option<bool>,
    pub expanded: Option<bool>,
    pub disabled: bool,
    /// Reached by Tab; `None` follows the role.
    pub focusable: Option<bool>,
    /// Where Tab reaches it, as HTML's `tabindex`: nodes with a positive
    /// index first, lowest first, then those with 0 (or none) in reading
    /// order. A negative index keeps a node focusable — by a press, or by
    /// [`Input::set_focus`](super::input::Input::set_focus) — but out of
    /// the Tab sequence.
    pub tab_index: Option<i32>,
    /// Hidden from assistive technology, as decoration (`aria-hidden`).
    pub hidden: bool,
    /// Announced when it changes (`aria-live="polite"`).
    pub live: bool,
    /// Read as one item: its name is all the text inside, and what is
    /// inside is not read again (a row of a list, a card).
    pub combine: bool,
    /// Read before its siblings with a lower priority (0 by default),
    /// whatever their order on screen.
    pub sort_priority: i32,
    /// Things it can do besides being pressed, offered by name to
    /// assistive technology (a row's "Delete", "Archive"); see
    /// [`Input::on_accessibility_action`](super::input::Input::on_accessibility_action).
    pub actions: Vec<String>,
    /// Its place in a set and the set's size (`aria-posinset`,
    /// `aria-setsize`): "5 of 100,000" for a row of a virtual list.
    pub position: Option<(usize, usize)>,
    /// Being loaded or changed, not to be read yet (`aria-busy`).
    pub busy: bool,
    /// Not valid, and why (`aria-invalid`, with the reason read as its
    /// error message).
    pub invalid: Option<String>,
    /// The current one of a set: the page, the step, the tab's screen
    /// (`aria-current`).
    pub current: bool,
    /// Its keyboard shortcut, as a menu shows it (`aria-keyshortcuts`).
    pub shortcut: Option<String>,
    /// Named by the text of another node, a visible label
    /// (`aria-labelledby`); a [`label`](Self::label) wins over it.
    pub labelled_by: Option<NodeId>,
    /// Described by the text of another node (`aria-describedby`).
    pub described_by: Option<NodeId>,
    /// A table row's place among all the table's rows, from 1, and how many
    /// there are (`aria-rowindex`; on the table itself, the count is its
    /// `aria-rowcount`): a virtual table builds only some.
    pub row_index: Option<(usize, usize)>,
    /// How a column is sorted (`aria-sort`): `Some(true)` ascending.
    pub sort: Option<bool>,
}

impl Semantics {
    pub fn new(role: Role) -> Self {
        Self {
            role,
            label: None,
            description: None,
            value: None,
            range: None,
            checked: None,
            selected: None,
            expanded: None,
            disabled: false,
            focusable: None,
            tab_index: None,
            hidden: false,
            live: false,
            combine: false,
            sort_priority: 0,
            actions: vec![],
            position: None,
            busy: false,
            invalid: None,
            current: false,
            shortcut: None,
            labelled_by: None,
            described_by: None,
            row_index: None,
            sort: None,
        }
    }

    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn value(mut self, value: impl Into<String>) -> Self {
        self.value = Some(value.into());
        self
    }

    pub fn range(mut self, min: f32, max: f32, now: f32) -> Self {
        self.range = Some(RangeValue { min, max, now });
        self
    }

    pub fn checked(mut self, checked: bool) -> Self {
        self.checked = Some(checked);
        self
    }

    pub fn selected(mut self, selected: bool) -> Self {
        self.selected = Some(selected);
        self
    }

    pub fn expanded(mut self, expanded: bool) -> Self {
        self.expanded = Some(expanded);
        self
    }

    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    pub fn focusable(mut self, focusable: bool) -> Self {
        self.focusable = Some(focusable);
        self
    }

    /// Places it in the Tab sequence (see [`Semantics::tab_index`]). Setting
    /// one makes a node focusable, as `tabindex` does an HTML element.
    pub fn tab_index(mut self, index: i32) -> Self {
        self.tab_index = Some(index);
        self.focusable.get_or_insert(true);
        self
    }

    pub fn hidden(mut self) -> Self {
        self.hidden = true;
        self
    }

    pub fn live(mut self) -> Self {
        self.live = true;
        self
    }

    /// Whether Tab reaches it: as asked, else as its role does, and never
    /// when disabled.
    pub fn is_focusable(&self) -> bool {
        !self.disabled && !self.hidden && self.focusable.unwrap_or(self.role.focusable_by_default())
    }
}

/// A node of the accessibility tree.
#[derive(Debug, Clone, PartialEq)]
pub struct AccessNode {
    /// The `Ui` node it stands for.
    pub id: NodeId,
    pub role: Role,
    /// The accessible name, computed: the label, else the text inside for a
    /// role named by content.
    pub name: String,
    pub description: Option<String>,
    pub value: Option<String>,
    pub range: Option<RangeValue>,
    pub checked: Option<bool>,
    pub selected: Option<bool>,
    pub expanded: Option<bool>,
    pub disabled: bool,
    pub focusable: bool,
    pub live: bool,
    /// Its custom actions (see [`Semantics::actions`]).
    pub actions: Vec<String>,
    /// See [`Semantics::position`].
    pub position: Option<(usize, usize)>,
    pub busy: bool,
    /// Why it is not valid (see [`Semantics::invalid`]).
    pub invalid: Option<String>,
    pub current: bool,
    pub shortcut: Option<String>,
    /// It is a modal dialog, open: nothing outside it is in the tree.
    pub modal: bool,
    /// See [`Semantics::row_index`].
    pub row_index: Option<(usize, usize)>,
    /// See [`Semantics::sort`].
    pub sort: Option<bool>,
    /// The name a link reaches it by (see [`anchors`](crate::anchors)).
    pub anchor: Option<String>,
    /// Its border box, as last laid out.
    pub bounds: Rect,
    /// Its children's indices in the tree's list.
    pub children: Vec<usize>,
}

impl Ui {
    /// Gives `id` a meaning for assistive technology.
    pub fn set_semantics(&mut self, id: NodeId, semantics: Semantics) {
        self.nodes[id.0].semantics = Some(semantics);
    }

    /// Takes `id`'s meaning away (it is a plain box again).
    pub fn clear_semantics(&mut self, id: NodeId) {
        self.nodes[id.0].semantics = None;
    }

    /// `id`'s meaning, if it has one of its own (text nodes have one
    /// implicitly: [`Role::StaticText`]).
    pub fn semantics(&self, id: NodeId) -> Option<&Semantics> {
        self.nodes[id.0].semantics.as_ref()
    }

    /// Changes `id`'s semantics in place — a checkbox ticked, a value typed
    /// — giving it [`Role::Generic`] first if it had none.
    pub fn semantics_mut(&mut self, id: NodeId) -> &mut Semantics {
        self.nodes[id.0]
            .semantics
            .get_or_insert_with(|| Semantics::new(Role::Generic))
    }

    /// Whether `id` is shown: not removed, and neither it nor an ancestor
    /// `display: none` or fully transparent.
    fn shown(&self, id: NodeId) -> bool {
        self.ancestors(id).all(|n| {
            let node = &self.nodes[n.0];
            !node.removed && node.style.display != Display::None && node.style.opacity > 0.0
        })
    }

    /// The text inside `id`, in reading order, as one string — what a
    /// node named by its content is called. Hidden subtrees and nodes
    /// with labels of their own contribute their label instead.
    pub fn text_content(&self, id: NodeId) -> String {
        let mut parts: Vec<String> = Vec::new();
        self.collect_text(id, &mut parts, true);
        parts
            .join(" ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn collect_text(&self, id: NodeId, out: &mut Vec<String>, top: bool) {
        let node = &self.nodes[id.0];
        if node.removed || node.style.display == Display::None {
            return;
        }
        if let Some(s) = &node.semantics {
            if s.hidden {
                return;
            }
            if !top && let Some(l) = &s.label {
                out.push(l.clone());
                return;
            }
        }
        if let NodeKind::Text(t) = &node.kind {
            out.push(if node.lines.is_empty() {
                t.clone()
            } else {
                node.lines.join(" ")
            });
        }
        for &c in &node.children {
            self.collect_text(c, out, false);
        }
    }

    /// The tree assistive technology reads, its root first.
    ///
    /// Boxes without semantics are left out and their children lifted to
    /// the nearest node that has them; text nodes are [`Role::StaticText`]
    /// unless an ancestor is named by its content (a button's label is the
    /// button's name, not a separate thing to read). Hidden and
    /// `display: none` subtrees are left out entirely.
    pub fn accessibility_tree(&self) -> Vec<AccessNode> {
        let mut out = vec![AccessNode {
            id: self.root(),
            role: Role::Group,
            name: String::new(),
            description: None,
            value: None,
            range: None,
            checked: None,
            selected: None,
            expanded: None,
            disabled: false,
            focusable: false,
            live: false,
            actions: vec![],
            position: None,
            busy: false,
            invalid: None,
            current: false,
            shortcut: None,
            modal: false,
            row_index: None,
            sort: None,
            anchor: None,
            bounds: self.rect(self.root()),
            children: vec![],
        }];
        // With a modal dialog open, it is all there is: what is behind it
        // cannot be reached, by touch or by a screen reader's cursor.
        match self.a11y_modal.filter(|&m| self.contains(m)) {
            Some(m) => self.build(m, 0, false, &mut out),
            None => {
                for c in self.reading_order(self.children(self.root())) {
                    self.build(c, 0, false, &mut out);
                }
            }
        }
        out
    }

    fn build(&self, id: NodeId, parent: usize, inside_named: bool, out: &mut Vec<AccessNode>) {
        if !self.shown(id) {
            return;
        }
        let node = &self.nodes[id.0];
        let own = match (&node.semantics, &node.kind) {
            (Some(s), _) if s.hidden => return,
            (Some(s), _) if s.role != Role::None => Some(s.clone()),
            (None, NodeKind::Text(_)) if !inside_named => Some(Semantics::new(Role::StaticText)),
            _ => None,
        };
        let Some(s) = own else {
            for &c in &node.children {
                self.build(c, parent, inside_named, out);
            }
            return;
        };
        let name = match &s.label {
            Some(l) => l.clone(),
            None => match s.labelled_by.filter(|&l| self.contains(l)) {
                Some(l) => self.text_content(l),
                None if s.role.named_by_content() || s.combine => self.text_content(id),
                None => String::new(),
            },
        };
        let description = s.description.clone().or_else(|| s.described_by.filter(|&d| self.contains(d)).map(|d| self.text_content(d)));
        // A busy subtree (loading, a placeholder) says so.
        let busy = s.busy || self.is_redacted(id);
        let index = out.len();
        out.push(AccessNode {
            id,
            role: s.role,
            name,
            description,
            value: s.value.clone(),
            range: s.range,
            checked: s.checked,
            selected: s.selected,
            expanded: s.expanded,
            disabled: s.disabled,
            focusable: s.is_focusable(),
            live: s.live,
            actions: s.actions.clone(),
            position: s.position,
            busy,
            invalid: s.invalid.clone(),
            current: s.current,
            shortcut: s.shortcut.clone(),
            modal: self.a11y_modal == Some(id),
            row_index: s.row_index,
            sort: s.sort,
            anchor: self.anchor_name(id),
            bounds: node.rect,
            children: vec![],
        });
        out[parent].children.push(index);
        // A node named by its content has spoken its text already; its
        // text children are not read again.
        if s.combine {
            // Read as one: nothing inside is read again.
            return;
        }
        let named = inside_named
            || (s.role.named_by_content() && s.label.is_none())
            || s.role == Role::StaticText;
        for c in self.reading_order(&node.children) {
            self.build(c, index, named, out);
        }
    }

    /// `children` in the order assistive technology reads them: by
    /// [`Semantics::sort_priority`], highest first, else as they are.
    fn reading_order(&self, children: &[NodeId]) -> Vec<NodeId> {
        let mut kids = children.to_vec();
        let prio = |c: &NodeId| self.nodes[c.0].semantics.as_ref().map_or(0, |s| s.sort_priority);
        if kids.iter().any(|c| prio(c) != 0) {
            kids.sort_by_key(|c| -prio(c));
        }
        kids
    }

    /// Reads `node` and what is inside it as one item (see
    /// [`Semantics::combine`]).
    pub fn combine_accessibility(&mut self, node: NodeId) {
        self.semantics_mut(node).combine = true;
    }

    /// Every node Tab reaches, in the DOM's sequential focus order: those
    /// with a positive [`tab_index`](Semantics::tab_index) by index (ties in
    /// reading order), then the rest in reading order. Nodes with a negative
    /// index are left out.
    pub fn focus_order(&self) -> Vec<NodeId> {
        let mut order = self.focusable_in_reading_order();
        order.retain(|&id| self.nodes[id.0].semantics.as_ref().and_then(|s| s.tab_index).unwrap_or(0) >= 0);
        // Stable: equal indices keep reading order.
        order.sort_by_key(|&id| match self.nodes[id.0].semantics.as_ref().and_then(|s| s.tab_index) {
            Some(i) if i > 0 => (0, i),
            _ => (1, 0),
        });
        order
    }

    fn focusable_in_reading_order(&self) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut stack = vec![self.root()];
        while let Some(id) = stack.pop() {
            if !self.shown(id) {
                continue;
            }
            let node = &self.nodes[id.0];
            if let Some(s) = &node.semantics {
                if s.hidden {
                    continue;
                }
                if s.is_focusable() {
                    out.push(id);
                }
            }
            stack.extend(self.reading_order(&node.children).into_iter().rev());
        }
        out
    }

    /// The node Tab (or Shift+Tab, `backwards`) moves focus to from
    /// `current`, wrapping round; the first (or last) with nothing focused.
    /// `None` when nothing is focusable.
    pub fn next_focus(&self, current: Option<NodeId>, backwards: bool) -> Option<NodeId> {
        self.next_focus_within(None, current, backwards)
    }

    /// [`Ui::next_focus`] among the nodes inside `scope` only: a dialog's
    /// focus trap.
    pub fn next_focus_within(&self, scope: Option<NodeId>, current: Option<NodeId>, backwards: bool) -> Option<NodeId> {
        let mut order = self.focus_order();
        if let Some(scope) = scope {
            order.retain(|&n| self.ancestors(n).any(|a| a == scope));
        }
        if order.is_empty() {
            return None;
        }
        let n = order.len();
        let at = current.and_then(|c| order.iter().position(|&o| o == c));
        Some(match (at, backwards) {
            (None, false) => order[0],
            (None, true) => order[n - 1],
            (Some(i), false) => order[(i + 1) % n],
            (Some(i), true) => order[(i + n - 1) % n],
        })
    }

    /// Text whose contrast with what is painted behind it is under WCAG's
    /// minimum (success criterion 1.4.3, level AA): 4.5:1, or 3:1 for large
    /// text — 24 px, or 18.66 px bold. With `enhanced`, level AAA's 7:1 and
    /// 4.5:1 (1.4.6).
    ///
    /// The background is the node's ancestors' backgrounds composited over
    /// `page` (what is under the UI: a canvas's clear colour), each with its
    /// opacity; text colours with alpha are composited too. Boxes that
    /// overlap without being ancestors — a popup over a panel — are not
    /// looked through, and gradients count as their first stop. Disabled
    /// controls are exempt, as WCAG exempts them.
    pub fn contrast_issues(&self, page: Rgba, enhanced: bool) -> Vec<ContrastIssue> {
        let mut out = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if node.removed {
                continue;
            }
            let id = self.id_at(i);
            let NodeKind::Text(text) = &node.kind else {
                continue;
            };
            if text.trim().is_empty() || !self.shown(id) {
                continue;
            }
            // Disabled controls are exempt; text hidden from assistive
            // technology is still seen, and is not.
            if self.ancestors(id).any(|a| {
                self.nodes[a.0]
                    .semantics
                    .as_ref()
                    .is_some_and(|s| s.disabled)
            }) {
                continue;
            }
            let chain: Vec<NodeId> = self.ancestors(id).collect();
            let mut bg = [page[0], page[1], page[2]];
            for a in chain.iter().rev() {
                let st = &self.nodes[a.0].style;
                let fill = st
                    .background
                    .or_else(|| st.background_gradient.first().map(|s| s.1));
                if let Some(c) = fill {
                    let alpha = c[3] * st.opacity;
                    bg = blend([c[0], c[1], c[2]], bg, alpha);
                }
            }
            let opacity: f32 = chain
                .iter()
                .map(|a| self.nodes[a.0].style.opacity)
                .product();
            let st = &node.style;
            let color = self.color_of(id);
            let fg = blend([color[0], color[1], color[2]], bg, color[3] * opacity);
            let ratio = contrast_ratio(fg, bg);
            let bold = match st.font_weight {
                FontWeight::Bold => true,
                FontWeight::Weight(w) => w >= 700,
                FontWeight::Normal => false,
            };
            let large = st.font_size >= 24.0 || (bold && st.font_size >= 18.66);
            let required = match (large, enhanced) {
                (true, false) => 3.0,
                (false, false) | (true, true) => 4.5,
                (false, true) => 7.0,
            };
            if ratio < required {
                out.push(ContrastIssue {
                    id,
                    text: text.clone(),
                    ratio,
                    required,
                    foreground: fg,
                    background: bg,
                });
            }
        }
        out
    }
}

/// Text with too little contrast; see [`Ui::contrast_issues`].
#[derive(Debug, Clone, PartialEq)]
pub struct ContrastIssue {
    pub id: NodeId,
    pub text: String,
    /// The contrast it has, `(L1 + 0.05) / (L2 + 0.05)`.
    pub ratio: f32,
    /// The least it needs.
    pub required: f32,
    /// The colours compared, as seen (sRGB, composited).
    pub foreground: [f32; 3],
    pub background: [f32; 3],
}

/// `top` over `under` at `alpha`, as sRGB values are composited on the
/// web (in the encoding).
fn blend(top: [f32; 3], under: [f32; 3], alpha: f32) -> [f32; 3] {
    let a = alpha.clamp(0.0, 1.0);
    [0, 1, 2].map(|k| top[k] * a + under[k] * (1.0 - a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FixedWidthText;

    /// WCAG's reference values: black on white 21:1, and the grey that just
    /// passes AA on white, #767676, at 4.54:1.
    #[test]
    fn contrast_ratio_matches_wcag() {
        assert!((contrast_ratio([0.0; 3], [1.0; 3]) - 21.0).abs() < 1e-4);
        let grey = 0x76 as f32 / 255.0;
        assert!((contrast_ratio([grey; 3], [1.0; 3]) - 4.54).abs() < 0.01);
        let lighter = 0x77 as f32 / 255.0;
        assert!(contrast_ratio([lighter; 3], [1.0; 3]) < 4.5);
        assert_eq!(contrast_ratio([0.3, 0.2, 0.9], [0.3, 0.2, 0.9]), 1.0);
    }

    fn laid_out(ui: &mut Ui) {
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    }

    /// A button is one node named by its text; layout boxes vanish; plain
    /// text is static text; a labelled icon is named by its label.
    #[test]
    fn the_tree_keeps_meaning_and_drops_layout() {
        let mut ui = Ui::new();
        let root = ui.root();
        let row = ui.add_box(root, crate::style! { display: flex; });
        let b = ui.add_box(row, crate::style! { padding: 4px; });
        let inner = ui.add_box(b, Style::default());
        ui.add_text(inner, "Save");
        ui.add_text(inner, "file");
        ui.set_semantics(b, Semantics::new(Role::Button));
        ui.add_text(row, "Ready");
        let icon = ui.add_box(row, crate::style! { width: 16px; height: 16px; });
        ui.set_semantics(icon, Semantics::new(Role::Image).label("Warning"));
        let deco = ui.add_box(row, Style::default());
        ui.add_text(deco, "★");
        ui.set_semantics(deco, Semantics::new(Role::Generic).hidden());
        laid_out(&mut ui);
        let tree = ui.accessibility_tree();
        let roles: Vec<(Role, &str)> = tree[1..]
            .iter()
            .map(|n| (n.role, n.name.as_str()))
            .collect();
        assert_eq!(
            roles,
            vec![
                (Role::Button, "Save file"),
                (Role::StaticText, "Ready"),
                (Role::Image, "Warning")
            ]
        );
        // All three hang off the root: the flex row had no meaning.
        assert_eq!(tree[0].children, vec![1, 2, 3]);
        assert!(tree[1].focusable && !tree[2].focusable);
        assert_eq!(tree[1].bounds, ui.rect(b));
    }

    /// Tab walks focusable nodes in reading order and wraps; disabled and
    /// hidden ones are skipped; Shift+Tab goes back.
    #[test]
    fn tab_order_is_reading_order() {
        let mut ui = Ui::new();
        let root = ui.root();
        let a = ui.add_box(root, Style::default());
        let group = ui.add_box(root, Style::default());
        let b = ui.add_box(group, Style::default());
        let off = ui.add_box(group, Style::default());
        let c = ui.add_box(root, Style::default());
        let gone = ui.add_box(root, crate::style! { display: none; });
        ui.set_semantics(a, Semantics::new(Role::Button));
        ui.set_semantics(b, Semantics::new(Role::Checkbox).checked(false));
        ui.set_semantics(off, Semantics::new(Role::Button).disabled(true));
        ui.set_semantics(c, Semantics::new(Role::Slider).range(0.0, 1.0, 0.5));
        ui.set_semantics(gone, Semantics::new(Role::Button));
        ui.set_semantics(group, Semantics::new(Role::Group).label("Options"));
        assert_eq!(ui.focus_order(), vec![a, b, c]);
        assert_eq!(ui.next_focus(None, false), Some(a));
        assert_eq!(ui.next_focus(Some(a), false), Some(b));
        assert_eq!(ui.next_focus(Some(c), false), Some(a));
        assert_eq!(ui.next_focus(Some(a), true), Some(c));
    }

    /// Light grey text on white fails; on the dark panel it sits in, it
    /// passes; large text needs only 3:1; disabled text is exempt.
    #[test]
    fn contrast_issues_composite_the_backgrounds() {
        let grey = 0x99 as f32 / 255.0; // 2.85:1 on white
        let mut ui = Ui::new();
        let root = ui.root();
        let t = ui.add_box(
            root,
            crate::style! { color: {[grey, grey, grey, 1.0]}; font-size: 13px; },
        );
        ui.add_text(t, "faint");
        let panel = ui.add_box(root, crate::style! { background: {[0.1, 0.1, 0.1, 1.0]}; color: {[grey, grey, grey, 1.0]}; font-size: 13px; });
        ui.add_text(panel, "on dark");
        let big = ui.add_box(
            root,
            crate::style! { color: {[0.55, 0.55, 0.55, 1.0]}; font-size: 28px; },
        );
        ui.add_text(big, "large");
        let off = ui.add_box(
            root,
            crate::style! { color: {[grey, grey, grey, 1.0]}; font-size: 13px; },
        );
        ui.add_text(off, "disabled");
        ui.set_semantics(off, Semantics::new(Role::Button).disabled(true));
        // Translucent black text on white: #000 at 40% is a light grey.
        let ghost = ui.add_box(
            root,
            crate::style! { color: {[0.0, 0.0, 0.0, 0.4]}; font-size: 13px; },
        );
        ui.add_text(ghost, "ghost");
        laid_out(&mut ui);
        let issues = ui.contrast_issues([1.0; 4], false);
        let texts: Vec<&str> = issues.iter().map(|i| i.text.as_str()).collect();
        assert_eq!(texts, vec!["faint", "ghost"]);
        assert!((issues[0].ratio - 2.85).abs() < 0.02, "{}", issues[0].ratio);
        // The large grey is 3.36:1: enough for AA's 3, not AAA's 4.5.
        let aaa: Vec<String> = ui
            .contrast_issues([1.0; 4], true)
            .into_iter()
            .map(|i| i.text)
            .collect();
        assert!(aaa.contains(&"large".to_string()));
    }

    use crate::Style;
}
