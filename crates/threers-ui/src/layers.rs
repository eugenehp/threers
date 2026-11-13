//! Popups and what is hit: deferred nodes, anchored nodes, and the index
//! hit-testing goes through.
//!
//! A menu or tooltip belongs in the tree next to what opened it, since that
//! is where its state and its layout come from. But it has to be drawn over
//! everything and kept on screen. The two are separate here:
//!
//! - [`Ui::defer`]: the node is laid out where it is, but painted after the
//!   rest of the tree, by priority, outside any ancestor's clip, and
//!   hit-tested before the rest.
//! - [`Ui::anchor`]: after layout, the node is moved so one of its corners
//!   sits on a point or on a corner of another node. With [`Fit::Flip`] it
//!   turns to the other side when it would leave the window, and with
//!   [`Fit::Snap`] it is pushed back inside.
//!
//! ```
//! use threers_ui::layers::{Anchor, AnchorTo, Corner, Fit};
//! use threers_ui::{style, FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let button = ui.add_box(root, style! { width: 80px; height: 20px; margin: 0 0 0 320px; overflow: hidden; });
//! let menu = ui.add_box(button, style! { position: absolute; width: 100px; height: 60px; });
//! ui.defer(menu, 1);
//! ui.anchor(menu, Anchor::new(AnchorTo::Node(button, Corner::BottomLeft), Corner::TopLeft).fit(Fit::Flip));
//! ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
//! // It would run off the right edge, so it hangs from the button's right
//! // corner instead, and the button's clip does not cut it.
//! assert_eq!(ui.rect(menu).x, 300.0);
//! assert_eq!(ui.rect(menu).y, 20.0);
//! assert_eq!(ui.hit_test(350.0, 50.0), Some(menu));
//! ```
//!
//! Hit-testing is indexed: rather than walking the tree from the root,
//! [`Ui::hit_test`] looks the point up in a grid of the boxes as last laid
//! out, built the first time it is asked after a change. A point then
//! costs the few boxes near it, not the whole tree.

use std::collections::HashMap;

use super::style::Display;
use super::tree::{NodeId, Rect, Ui};

/// A corner of a box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

impl Corner {
    fn of(self, r: Rect) -> (f32, f32) {
        match self {
            Corner::TopLeft => (r.x, r.y),
            Corner::TopRight => (r.x + r.width, r.y),
            Corner::BottomLeft => (r.x, r.y + r.height),
            Corner::BottomRight => (r.x + r.width, r.y + r.height),
        }
    }

    fn flip_x(self) -> Self {
        match self {
            Corner::TopLeft => Corner::TopRight,
            Corner::TopRight => Corner::TopLeft,
            Corner::BottomLeft => Corner::BottomRight,
            Corner::BottomRight => Corner::BottomLeft,
        }
    }

    fn flip_y(self) -> Self {
        match self {
            Corner::TopLeft => Corner::BottomLeft,
            Corner::BottomLeft => Corner::TopLeft,
            Corner::TopRight => Corner::BottomRight,
            Corner::BottomRight => Corner::TopRight,
        }
    }
}

/// What an anchored node is placed against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnchorTo {
    /// A point in the UI: where the pointer was, for a context menu.
    Point(f32, f32),
    /// A corner of another node, as last laid out.
    Node(NodeId, Corner),
}

/// What an anchored node does when it would leave the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Fit {
    /// Nothing: it may hang off the edge.
    None,
    /// Turns to the other side of its anchor on the axis it overflows
    /// (and of the anchoring node, for [`AnchorTo::Node`]), then is pushed
    /// inside if it still does not fit.
    #[default]
    Flip,
    /// Pushed back inside the window, staying on its side.
    Snap,
}

/// Where an anchored node goes. See [`Ui::anchor`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anchor {
    pub to: AnchorTo,
    /// The node's corner put on the anchor.
    pub corner: Corner,
    /// Moved by this much from the anchor, away from it: a gap between a
    /// button and its menu. Mirrored when the node flips.
    pub offset: (f32, f32),
    pub fit: Fit,
}

impl Anchor {
    pub fn new(to: AnchorTo, corner: Corner) -> Self {
        Anchor {
            to,
            corner,
            offset: (0.0, 0.0),
            fit: Fit::Flip,
        }
    }

    pub fn offset(mut self, dx: f32, dy: f32) -> Self {
        self.offset = (dx, dy);
        self
    }

    pub fn fit(mut self, fit: Fit) -> Self {
        self.fit = fit;
        self
    }
}

impl Ui {
    /// Paints `id` after the rest of the tree — after other deferred nodes
    /// of lower `priority`, and of the same priority earlier in the tree —
    /// outside any ancestor's clip, and hit-tests it first. Layout is as if
    /// it were not deferred.
    pub fn defer(&mut self, id: NodeId, priority: i32) {
        self.node_mut(id).deferred = Some(priority);
        self.invalidate_hits();
    }

    /// `id`'s priority if it is deferred.
    pub fn deferred_priority(&self, id: NodeId) -> Option<i32> {
        self.node(id).deferred
    }

    /// Paints `id` in its place in the tree again.
    pub fn undefer(&mut self, id: NodeId) {
        self.node_mut(id).deferred = None;
        self.invalidate_hits();
    }

    /// Places `id` against an anchor after every layout. See the
    /// [module docs](self).
    pub fn anchor(&mut self, id: NodeId, anchor: Anchor) {
        if self.node(id).anchor != Some(anchor) {
            self.touch(id);
            self.nodes[id.0].anchor = Some(anchor);
        }
    }

    pub fn unanchor(&mut self, id: NodeId) {
        if self.node(id).anchor.is_some() {
            self.touch(id);
            self.nodes[id.0].anchor = None;
        }
    }

    /// Deferred nodes that show (no ancestor is `display: none`), in the
    /// order they are painted.
    pub(crate) fn deferred_order(&self) -> Vec<NodeId> {
        let mut out = vec![];
        let mut stack = vec![self.root()];
        while let Some(id) = stack.pop() {
            let n = &self.nodes[id.0];
            if n.style.display == Display::None {
                continue;
            }
            if n.deferred.is_some() && id.0 != 0 {
                out.push(id);
            }
            let order: Vec<NodeId> = self.paint_order_iter(id).collect();
            stack.extend(order.into_iter().rev());
        }
        // Stable: equal priorities keep the order they were met in.
        out.sort_by_key(|&id| self.nodes[id.0].deferred.unwrap_or(0));
        out
    }

    /// The opacity a node's ancestors leave it, for painting it outside
    /// their subtree.
    pub(crate) fn inherited_opacity(&self, id: NodeId) -> f32 {
        self.ancestors(id)
            .skip(1)
            .map(|n| self.nodes[n.0].style.opacity)
            .product()
    }

    /// Moves anchored nodes into place, in tree order (so one can anchor
    /// to another placed before it).
    pub(crate) fn place_anchored(&mut self) {
        let (vw, vh) = self.viewport;
        let mut stack = vec![self.root()];
        let mut anchored = vec![];
        while let Some(id) = stack.pop() {
            let n = &self.nodes[id.0];
            if n.style.display == Display::None {
                continue;
            }
            if let Some(a) = n.anchor {
                anchored.push((id, a));
            }
            stack.extend(n.children.iter().rev().copied());
        }
        for (id, a) in anchored {
            let r = self.nodes[id.0].rect;
            let place = |corner: Corner, to: AnchorTo, flip: (bool, bool)| -> Option<Rect> {
                let (mut ax, mut ay) = match to {
                    AnchorTo::Point(x, y) => (x, y),
                    AnchorTo::Node(n, c) => {
                        if !self.contains(n) {
                            return None;
                        }
                        let c = if flip.0 { c.flip_x() } else { c };
                        let c = if flip.1 { c.flip_y() } else { c };
                        c.of(self.nodes[n.0].rect)
                    }
                };
                let corner = if flip.0 { corner.flip_x() } else { corner };
                let corner = if flip.1 { corner.flip_y() } else { corner };
                // The offset points away from the anchor: rightward and
                // down for a node hanging from its top-left corner.
                let (sx, sy) = match corner {
                    Corner::TopLeft => (1.0, 1.0),
                    Corner::TopRight => (-1.0, 1.0),
                    Corner::BottomLeft => (1.0, -1.0),
                    Corner::BottomRight => (-1.0, -1.0),
                };
                ax += a.offset.0 * sx;
                ay += a.offset.1 * sy;
                // The corner's place within the node.
                let (cx, cy) = corner.of(Rect { x: 0.0, y: 0.0, ..r });
                Some(Rect {
                    x: ax - cx,
                    y: ay - cy,
                    ..r
                })
            };
            let Some(mut want) = place(a.corner, a.to, (false, false)) else {
                continue;
            };
            let over_x = |q: Rect| q.x < 0.0 || q.x + q.width > vw;
            let over_y = |q: Rect| q.y < 0.0 || q.y + q.height > vh;
            if a.fit == Fit::Flip {
                let (fx, fy) = (over_x(want), over_y(want));
                if (fx || fy)
                    && let Some(flipped) = place(a.corner, a.to, (fx, fy))
                {
                    // Keep a flip only on an axis where it helps.
                    if fx && !over_x(flipped) {
                        want.x = flipped.x;
                    }
                    if fy && !over_y(flipped) {
                        want.y = flipped.y;
                    }
                }
            }
            if a.fit != Fit::None {
                want.x = want.x.min(vw - want.width).max(0.0);
                want.y = want.y.min(vh - want.height).max(0.0);
            }
            let (dx, dy) = (want.x - r.x, want.y - r.y);
            if dx != 0.0 || dy != 0.0 {
                self.offset_laid_out(id, dx, dy);
            }
        }
    }

    /// Forgets the hit index; the next hit test builds it again.
    pub(crate) fn invalidate_hits(&mut self) {
        self.hits = std::sync::OnceLock::new();
    }

    /// The deepest visible node at `(x, y)`, last painted first, by walking
    /// the tree: the definition [`Ui::hit_test`]'s index must agree with.
    pub fn hit_test_walk(&self, x: f32, y: f32) -> Option<NodeId> {
        for &d in self.deferred_order().iter().rev() {
            if let Some(h) = self.hit_walk(d, x, y) {
                return Some(h);
            }
        }
        self.hit_walk(self.root(), x, y)
    }

    /// [`Ui::hit_test`] as if `except` and everything in it were not there:
    /// what is under a drag preview that follows the pointer.
    pub fn hit_test_except(&self, x: f32, y: f32, except: NodeId) -> Option<NodeId> {
        for &d in self.deferred_order().iter().rev() {
            if let Some(h) = self.hit_walk_except(d, x, y, Some(except)) {
                return Some(h);
            }
        }
        self.hit_walk_except(self.root(), x, y, Some(except))
    }

    fn hit_walk(&self, id: NodeId, x: f32, y: f32) -> Option<NodeId> {
        self.hit_walk_except(id, x, y, None)
    }

    pub(crate) fn hit_walk_except(&self, id: NodeId, x: f32, y: f32, except: Option<NodeId>) -> Option<NodeId> {
        let node = &self.nodes[id.0];
        if node.style.display == Display::None || Some(id) == except {
            return None;
        }
        let inside = node.rect.hit(x, y);
        if node.style.clip && !inside {
            return None;
        }
        // Paint order backwards, without collecting it: positioned children
        // were painted last, so they are hit first. Deferred children are
        // not here; they were tried first.
        let deferred = |c: &&NodeId| self.nodes[c.0].deferred.is_some();
        let children = &node.children;
        let topmost = children.iter().rev().filter(|c| self.is_positioned(**c));
        let below = children.iter().rev().filter(|c| !self.is_positioned(**c));
        for &c in topmost.chain(below).filter(|c| !deferred(c)) {
            if let Some(h) = self.hit_walk_except(c, x, y, except) {
                return Some(h);
            }
        }
        (inside && node.style.pointer_events).then_some(id)
    }
}

/// Grid cells, in UI pixels.
const CELL: f32 = 64.0;

/// Boxes as last laid out, in paint order, binned by grid cell.
#[derive(Debug, Clone, Default)]
pub(crate) struct HitIndex {
    /// Paint order: a node, and where its clip chain starts in `clips`.
    entries: Vec<(NodeId, Option<u32>)>,
    /// Clipping boxes, each with the one enclosing it.
    clips: Vec<(Rect, Option<u32>)>,
    cells: HashMap<(i32, i32), Vec<u32>>,
}

impl HitIndex {
    pub(crate) fn build(ui: &Ui) -> Self {
        let mut index = HitIndex::default();
        let mut roots = vec![ui.root()];
        roots.extend(ui.deferred_order());
        for (k, &root) in roots.iter().enumerate() {
            // Deferred subtrees ignore their ancestors' clips.
            let mut stack = vec![(root, None::<u32>)];
            while let Some((id, clip)) = stack.pop() {
                let n = &ui.nodes[id.0];
                if n.style.display == Display::None || (k > 0 && id != root && n.deferred.is_some()) {
                    continue;
                }
                if k == 0 && n.deferred.is_some() && id.0 != 0 {
                    continue;
                }
                let e = index.entries.len() as u32;
                index.entries.push((id, clip));
                // Conservatively: the box's hit region, cut by its clips;
                // nothing for a box the pointer passes through.
                let mut b = (n.rect.x - 1.0, n.rect.y - 1.0, n.rect.x + n.rect.width, n.rect.y + n.rect.height);
                if !n.style.pointer_events {
                    b = (0.0, 0.0, -1.0, -1.0);
                }
                let mut c = clip;
                while let Some(i) = c {
                    let (r, up) = index.clips[i as usize];
                    b = (b.0.max(r.x - 1.0), b.1.max(r.y - 1.0), b.2.min(r.x + r.width), b.3.min(r.y + r.height));
                    c = up;
                }
                if b.2 > b.0 && b.3 > b.1 && b.0.is_finite() && b.3.is_finite() {
                    let cell = |v: f32| (v / CELL).floor().clamp(-1e6, 1e6) as i32;
                    for cy in cell(b.1)..=cell(b.3) {
                        for cx in cell(b.0)..=cell(b.2) {
                            index.cells.entry((cx, cy)).or_default().push(e);
                        }
                    }
                }
                let child_clip = if n.style.clip {
                    index.clips.push((n.rect, clip));
                    Some(index.clips.len() as u32 - 1)
                } else {
                    clip
                };
                let order: Vec<NodeId> = ui.paint_order_iter(id).collect();
                stack.extend(order.into_iter().rev().map(|c| (c, child_clip)));
            }
        }
        index
    }

    pub(crate) fn hit(&self, ui: &Ui, x: f32, y: f32) -> Option<NodeId> {
        let key = ((x / CELL).floor() as i32, (y / CELL).floor() as i32);
        let candidates = self.cells.get(&key)?;
        // Latest painted first: the first that passes is the answer.
        candidates.iter().rev().find_map(|&e| {
            let (id, mut clip) = self.entries[e as usize];
            if !ui.nodes[id.0].rect.hit(x, y) {
                return None;
            }
            while let Some(i) = clip {
                let (r, up) = self.clips[i as usize];
                if !r.hit(x, y) {
                    return None;
                }
                clip = up;
            }
            Some(id)
        })
    }
}
