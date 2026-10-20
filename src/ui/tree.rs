//! The box tree.

use super::layout::TextMeasure;
use super::style::Style;

/// A node of a [`Ui`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub(crate) usize);

/// What a node draws besides its box (background and borders).
#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    /// Only its box and children.
    Box,
    /// One line of text, in the node's font and colour (`white-space: nowrap`).
    Text(String),
    /// SVG path data in a `view_box` (x, y, width, height), scaled to the
    /// content box: stroked with `stroke` px lines and/or filled, in the
    /// node's colour (`currentColor`).
    Path {
        d: Vec<String>,
        view_box: [f32; 4],
        stroke: Option<f32>,
        fill: bool,
    },
}

/// A laid-out box: the border box, in pixels from the top-left of the UI.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x && y >= self.y && x < self.x + self.width && y < self.y + self.height
    }

    /// Whether a pointer at `(x, y)` is over the box, as Chrome hit-tests:
    /// the pixel `[x, x+1) × [y, y+1)` overlapping it, so a box starting at
    /// 60.95 is under a pointer at 60.
    pub fn hit(&self, x: f32, y: f32) -> bool {
        x + 1.0 > self.x && y + 1.0 > self.y && x < self.x + self.width && y < self.y + self.height
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Node {
    pub style: Style,
    pub kind: NodeKind,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    pub rect: Rect,
    pub removed: bool,
    /// For a scrolling box: the height of its content and how far it is
    /// scrolled, as last laid out.
    pub scroll_height: f32,
    pub scrolled: f32,
    /// A wrapped text node's lines, as last laid out.
    pub lines: Vec<String>,
}

/// A tree of styled boxes. Node 0 is the root.
#[derive(Debug, Clone)]
pub struct Ui {
    pub(crate) nodes: Vec<Node>,
}

impl Default for Ui {
    fn default() -> Self {
        Self::new()
    }
}

impl Ui {
    pub fn new() -> Self {
        Self {
            nodes: vec![Node {
                style: Style::default(),
                kind: NodeKind::Box,
                parent: None,
                children: vec![],
                rect: Rect::default(),
                removed: false,
                scroll_height: 0.0,
                scrolled: 0.0,
                lines: vec![],
            }],
        }
    }

    pub fn root(&self) -> NodeId {
        NodeId(0)
    }

    fn add(&mut self, parent: NodeId, style: Style, kind: NodeKind) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node { style, kind, parent: Some(parent), children: vec![], rect: Rect::default(), removed: false, scroll_height: 0.0, scrolled: 0.0, lines: vec![] });
        self.nodes[parent.0].children.push(id);
        id
    }

    /// A box child of `parent`.
    pub fn add_box(&mut self, parent: NodeId, style: Style) -> NodeId {
        self.add(parent, style, NodeKind::Box)
    }

    /// A line of text under `parent`, with `parent`'s font and colour (CSS
    /// inheritance of `color` and `font-*`).
    pub fn add_text(&mut self, parent: NodeId, text: impl Into<String>) -> NodeId {
        let p = &self.nodes[parent.0].style;
        let style = Style {
            color: p.color,
            font_family: p.font_family.clone(),
            font_size: p.font_size,
            font_weight: p.font_weight,
            line_height: p.line_height,
            line_through: p.line_through,
            wrap: p.wrap,
            italic: p.italic,
            vertical: p.vertical,
            ..Style::default()
        };
        self.add(parent, style, NodeKind::Text(text.into()))
    }

    /// An icon or other vector shape (see [`NodeKind::Path`]).
    pub fn add_path(&mut self, parent: NodeId, style: Style, d: Vec<String>, view_box: [f32; 4], stroke: Option<f32>, fill: bool) -> NodeId {
        self.add(parent, style, NodeKind::Path { d, view_box, stroke, fill })
    }

    /// Removes `id` and its subtree from layout and painting.
    pub fn remove(&mut self, id: NodeId) {
        if let Some(p) = self.nodes[id.0].parent {
            self.nodes[p.0].children.retain(|&c| c != id);
        }
        let mut stack = vec![id];
        while let Some(n) = stack.pop() {
            self.nodes[n.0].removed = true;
            stack.extend(self.nodes[n.0].children.clone());
        }
    }

    /// Moves `id` and everything inside it by `(dx, dy)` after layout (for
    /// an app placing something as a renderer would, by its own rounding).
    pub fn offset(&mut self, id: NodeId, dx: f32, dy: f32) {
        let mut stack = vec![id];
        while let Some(n) = stack.pop() {
            self.nodes[n.0].rect.x += dx;
            self.nodes[n.0].rect.y += dy;
            stack.extend(self.nodes[n.0].children.iter().copied());
        }
    }

    /// Moves `child` (a child of `parent`) to position `index` among its
    /// siblings (CSS `order`).
    pub fn move_child(&mut self, parent: NodeId, child: NodeId, index: usize) {
        let children = &mut self.nodes[parent.0].children;
        if let Some(i) = children.iter().position(|&c| c == child) {
            children.remove(i);
            children.insert(index.min(children.len()), child);
        }
    }

    /// Moves `child` from wherever it is to the end of `parent`'s children
    /// (the DOM's `appendChild` of an attached node).
    pub fn reparent(&mut self, child: NodeId, parent: NodeId) {
        if let Some(p) = self.nodes[child.0].parent {
            self.nodes[p.0].children.retain(|&c| c != child);
        }
        self.nodes[child.0].parent = Some(parent);
        self.nodes[parent.0].children.push(child);
    }

    /// Removes every child of `id`.
    pub fn clear_children(&mut self, id: NodeId) {
        for c in self.nodes[id.0].children.clone() {
            self.remove(c);
        }
    }

    pub fn style(&self, id: NodeId) -> &Style {
        &self.nodes[id.0].style
    }

    pub fn style_mut(&mut self, id: NodeId) -> &mut Style {
        &mut self.nodes[id.0].style
    }

    pub fn set_style(&mut self, id: NodeId, style: Style) {
        self.nodes[id.0].style = style;
    }

    pub fn kind(&self, id: NodeId) -> &NodeKind {
        &self.nodes[id.0].kind
    }

    /// Replaces a text node's text.
    pub fn set_text(&mut self, id: NodeId, text: impl Into<String>) {
        self.nodes[id.0].kind = NodeKind::Text(text.into());
    }

    pub fn children(&self, id: NodeId) -> &[NodeId] {
        &self.nodes[id.0].children
    }

    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        self.nodes[id.0].parent
    }

    /// The border box from the last [`layout`](Self::layout).
    pub fn rect(&self, id: NodeId) -> Rect {
        self.nodes[id.0].rect
    }

    /// Lays the tree out in a `width × height` viewport.
    pub fn layout(&mut self, width: f32, height: f32, text: &mut dyn TextMeasure) {
        super::layout::layout(self, width, height, text);
    }

    /// The deepest visible node containing `(x, y)`, last painted first.
    pub fn hit_test(&self, x: f32, y: f32) -> Option<NodeId> {
        self.hit(self.root(), x, y)
    }

    fn hit(&self, id: NodeId, x: f32, y: f32) -> Option<NodeId> {
        let node = &self.nodes[id.0];
        if node.style.display == super::style::Display::None {
            return None;
        }
        let inside = node.rect.hit(x, y);
        if node.style.clip && !inside {
            return None;
        }
        for c in self.paint_order(id).into_iter().rev() {
            if let Some(h) = self.hit(c, x, y) {
                return Some(h);
            }
        }
        inside.then_some(id)
    }

    /// `id`'s children in painting order: in-flow boxes first, then
    /// positioned ones (`relative`, `absolute`), each in tree order — CSS's
    /// order without `z-index`.
    pub fn paint_order(&self, id: NodeId) -> Vec<NodeId> {
        let children = &self.nodes[id.0].children;
        let positioned = |c: &NodeId| self.nodes[c.0].style.position != super::style::Position::Static;
        children.iter().copied().filter(|c| !positioned(c)).chain(children.iter().copied().filter(positioned)).collect()
    }

    /// A scrolling box's content height and the height it shows (its
    /// padding box), and how far it is scrolled, as last laid out.
    pub fn scroll_extent(&self, id: NodeId) -> (f32, f32, f32) {
        let n = &self.nodes[id.0];
        let s = &n.style;
        (n.scroll_height, (n.rect.height - s.border.vertical()).max(0.0), n.scrolled)
    }

    /// The innermost scrolling box among `id` and its ancestors.
    pub fn scroll_container(&self, id: NodeId) -> Option<NodeId> {
        self.ancestors(id).find(|n| self.nodes[n.0].style.scroll)
    }

    /// `id` and its ancestors, innermost first.
    pub fn ancestors(&self, id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        std::iter::successors(Some(id), move |n| self.nodes[n.0].parent)
    }
}
