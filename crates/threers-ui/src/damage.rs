//! Which part of the window changed between two frames, so that only that part is painted again.
//!
//! An interface that is rebuilt every frame (as an immediate-mode one is) has no record of what moved, so the
//! change is found by comparing the new tree with the last one, node by node: a node
//! that differs in style, text or place marks its old and its new rectangle as damaged. The damage is then
//! widened so that clipping it can't cut through anything whose drawing depends on being whole (a line of text,
//! a rounded corner). When the trees don't have the same shape, everything is damaged.
//!
//! Paint each damaged rectangle with a [`Culled`](super::region::Culled) painter, which leaves out what lies
//! away from it and draws the rest exactly where a whole-window paint would, so the pixels come out the same.

use super::{Length, NodeId, NodeKind, Rect, Style, Ui};

/// A drawn path as painting sees it: its data, view box, stroke and fill.
pub type PathPaint = (Vec<String>, [f32; 4], Option<f32>, bool);

/// One node of a laid-out tree, as far as painting can tell.
#[derive(Clone, PartialEq)]
pub struct Node {
    pub rect: Rect,
    /// The part of `rect` the boxes that clip it let show: what painting it
    /// can touch. A list's rows scrolled out of sight show nothing, so they
    /// neither need repainting nor hold a repaint open to keep them whole.
    pub visible: Rect,
    pub style: Style,
    pub text: Option<String>,
    /// A drawn path, if the node is one.
    pub path: Option<PathPaint>,
    /// The image shown, by identity, if the node is one.
    pub image: Option<u64>,
    /// Painted after the rest, by this priority.
    pub deferred: Option<i32>,
    /// The transform it is drawn through, its own and those around it, if
    /// any: `visible` is then where it is seen.
    pub turned: Option<[f32; 9]>,
    pub children: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Damage {
    /// Nothing looks different.
    None,
    /// These rectangles, in whole pixels and sharing no pixel with each other.
    Regions(Vec<Rect>),
    /// Everything: the trees differ in shape.
    Full,
}

/// The tree's nodes, depth first. Call it after the tree is laid out.
pub fn collect(ui: &Ui) -> Vec<Node> {
    /// `clip`: where the clipping boxes around `id` let it show (`None`:
    /// anywhere), as painting cuts it.
    fn walk(ui: &Ui, out: &mut Vec<Node>, id: NodeId, clip: Option<Rect>, turned: Option<[f32; 9]>) {
        // Deferred nodes are painted outside their ancestors' clips.
        let clip = if ui.deferred_priority(id).is_some() {
            None
        } else {
            clip
        };
        // Hidden, it paints nothing, and nothing inside it does: a stand-in
        // for each node keeps the tree's shape (shown again, it damages
        // just its box) without copying styles nothing will paint.
        if ui.style(id).display == crate::Display::None {
            hidden(ui, out, id);
            return;
        }
        let rect = ui.rect(id);
        let own = ui.paint_transform(id).map(|(h, _)| h);
        let turned = match (turned, own) {
            (Some(a), Some(b)) => Some(std::array::from_fn(|i| (0..3).map(|k| a[i / 3 * 3 + k] * b[k * 3 + i % 3]).sum())),
            (a, b) => a.or(b),
        };
        let seen = match turned {
            // its own shadow, on what is behind, is not turned with it
            Some(h) => {
                let r = seen_through(h, rect);
                own.and_then(|_| ui.lift_shadow_box(id)).map_or(r, |s| union(r, s))
            }
            None => rect,
        };
        let visible = clip.map_or(seen, |c| intersection(seen, c));
        out.push(Node {
            rect,
            visible,
            // A flicker shows as the opacity it paints at.
            style: {
                let mut s = ui.style(id).clone();
                s.opacity = ui.painted_opacity(id);
                // (the colour it's painted in, its own or not: one inherited
                // can change, and nothing of the node itself)
                if matches!(ui.kind(id), NodeKind::Text(_) | NodeKind::Path { .. }) {
                    s.color = ui.color_of(id);
                }
                s
            },
            text: match ui.kind(id) {
                NodeKind::Text(t) => Some(t.clone()),
                _ => None,
            },
            path: match ui.kind(id) {
                NodeKind::Path {
                    d,
                    view_box,
                    stroke,
                    fill,
                } => Some((d.clone(), *view_box, *stroke, *fill)),
                _ => None,
            },
            image: match ui.kind(id) {
                NodeKind::Image { image } => Some(image.id()),
                // A surface's texture changes without the tree knowing: its
                // box is damaged by whoever updates it.
                NodeKind::Surface { id, .. } => Some(u64::MAX - id),
                _ => None,
            },
            deferred: ui.deferred_priority(id),
            turned,
            children: ui.children(id).len(),
        });
        // What its children can show in: cut to it if it clips (where it is
        // seen, when it is turned).
        let inner = if ui.style(id).clip {
            let r = turned.map_or(rect, |h| seen_through(h, rect));
            Some(clip.map_or(r, |c| intersection(r, c)))
        } else {
            clip
        };
        for &child in ui.children(id) {
            walk(ui, out, child, inner, turned);
        }
    }
    fn hidden(ui: &Ui, out: &mut Vec<Node>, id: NodeId) {
        let none = Rect::default();
        out.push(Node {
            rect: none,
            visible: none,
            style: Style {
                display: crate::Display::None,
                ..Style::default()
            },
            text: None,
            path: None,
            image: None,
            deferred: None,
            turned: None,
            children: ui.children(id).len(),
        });
        for &child in ui.children(id) {
            hidden(ui, out, child);
        }
    }
    let mut out = Vec::with_capacity(ui.nodes.len());
    walk(ui, &mut out, ui.root(), None, None);
    out
}

/// The box `r` is seen in through `h`; everywhere, if part of it is behind
/// the eye.
fn seen_through(h: [f32; 9], r: Rect) -> Rect {
    let (mut lo, mut hi) = ((f32::MAX, f32::MAX), (f32::MIN, f32::MIN));
    for (x, y) in [(r.x, r.y), (r.x + r.width, r.y), (r.x, r.y + r.height), (r.x + r.width, r.y + r.height)] {
        let w = h[6] * x + h[7] * y + h[8];
        if w <= 1e-6 {
            return Rect { x: -1e6, y: -1e6, width: 2e6, height: 2e6 };
        }
        let (px, py) = crate::depth::apply_homography(h, x, y);
        lo = (lo.0.min(px), lo.1.min(py));
        hi = (hi.0.max(px), hi.1.max(py));
    }
    Rect { x: lo.0, y: lo.1, width: hi.0 - lo.0, height: hi.1 - lo.1 }
}

/// Where `a` and `b` overlap; an empty rectangle where they do not.
fn intersection(a: Rect, b: Rect) -> Rect {
    let (x0, y0) = (a.x.max(b.x), a.y.max(b.y));
    let (x1, y1) = (
        (a.x + a.width).min(b.x + b.width),
        (a.y + a.height).min(b.y + b.height),
    );
    Rect {
        x: x0,
        y: y0,
        width: (x1 - x0).max(0.0),
        height: (y1 - y0).max(0.0),
    }
}

pub fn union(a: Rect, b: Rect) -> Rect {
    let (x0, y0) = (a.x.min(b.x), a.y.min(b.y));
    let (x1, y1) = (
        (a.x + a.width).max(b.x + b.width),
        (a.y + a.height).max(b.y + b.height),
    );
    Rect {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    }
}

fn intersects(a: Rect, b: Rect) -> bool {
    a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
}

fn contains(outer: Rect, inner: Rect) -> bool {
    inner.x >= outer.x
        && inner.y >= outer.y
        && inner.x + inner.width <= outer.x + outer.width
        && inner.y + inner.height <= outer.y + outer.height
}

/// Where nodes changed, old place and new, in whole pixels; `None` when the
/// tree's shape changed (everything is damaged).
fn changed(prev: &[Node], next: &[Node], width: f32, height: f32) -> Option<Vec<Rect>> {
    if prev.len() != next.len() || prev.iter().zip(next).any(|(a, b)| a.children != b.children) {
        return None;
    }
    let mut rects: Vec<Rect> = Vec::new();
    for (a, b) in prev.iter().zip(next) {
        if a != b {
            for r in [a.visible, b.visible] {
                let r = snap(r, width, height);
                if r.width > 0.0 && r.height > 0.0 {
                    rects.push(r);
                }
            }
        }
    }
    Some(rects)
}

/// Each node's subtree size (itself and all below it), from the depth-first
/// list and each node's number of children.
fn sizes(nodes: &[Node]) -> Vec<usize> {
    let mut size = vec![1usize; nodes.len()];
    // children follow their parent: from the end, a node's size is one plus
    // its children's, each child found after the one before's subtree
    for i in (0..nodes.len()).rev() {
        let mut at = i + 1;
        let mut total = 1;
        for _ in 0..nodes[i].children {
            if at >= nodes.len() {
                break;
            }
            total += size[at];
            at += size[at];
        }
        size[i] = total;
    }
    size
}

/// Whether a node draws nothing of its own: no text, path or image, no
/// background, border, shadow or blur, and no scrollbar (a box only
/// holding others, a pane the terminal shows through). Its place changing
/// changes no pixel but its children's.
fn paints_nothing(n: &Node) -> bool {
    let s = &n.style;
    let clear = |c: crate::Rgba| c[3] <= 0.0;
    n.text.is_none()
        && n.path.is_none()
        && n.image.is_none()
        && s.background.is_none_or(clear)
        && s.background_gradient.is_empty()
        && (s.border.horizontal() + s.border.vertical() == 0.0 || clear(s.border_color))
        && s.box_shadow.is_empty()
        && s.backdrop_blur == 0.0
        && s.mask.is_none()
        && !s.scroll
        && !s.scroll_x
}

/// Two nodes alike but for how many children they hold (and the style
/// that only places what's in them: in the same box, they look the same).
fn same_but_children(a: &Node, b: &Node) -> bool {
    a.rect == b.rect
        && a.visible == b.visible
        && a.text == b.text
        && a.path == b.path
        && a.image == b.image
        && a.deferred == b.deferred
        && a.turned == b.turned
        && looks_the_same(a, &b.style)
}

/// Whether node `a`, styled `b` instead, in the same box, looks the same:
/// they differ only in what lays it and its children out (its padding
/// too, where no text or background is drawn by it: a sidebar padded
/// above a keyboard).
fn looks_the_same(a: &Node, b: &Style) -> bool {
    let s = &a.style;
    if s == b {
        return true;
    }
    let mut x = b.clone();
    (x.position, x.box_sizing, x.flex_direction, x.flex_grow, x.flex_shrink, x.flex_basis) =
        (s.position, s.box_sizing, s.flex_direction, s.flex_grow, s.flex_shrink, s.flex_basis);
    (x.align_items, x.align_self, x.justify_content, x.gap, x.flex_wrap, x.flex_reverse, x.order, x.align_content) =
        (s.align_items, s.align_self, s.justify_content, s.gap, s.flex_wrap, s.flex_reverse, s.order, s.align_content);
    (x.column_gap, x.row_gap, x.width, x.height, x.min_width, x.min_height, x.max_width, x.max_height) =
        (s.column_gap, s.row_gap, s.width, s.height, s.min_width, s.min_height, s.max_width, s.max_height);
    (x.margin, x.margin_auto, x.left, x.top, x.right, x.bottom, x.aspect_ratio) = (s.margin, s.margin_auto, s.left, s.top, s.right, s.bottom, s.aspect_ratio);
    (x.grid_columns, x.grid_rows, x.grid_column, x.grid_row) = (s.grid_columns.clone(), s.grid_rows.clone(), s.grid_column, s.grid_row);
    (x.cursor, x.pointer_events, x.user_select) = (s.cursor, s.pointer_events, s.user_select);
    if a.text.is_none() && !s.background_clip_content {
        x.padding = s.padding;
    }
    *s == x
}

/// Whether `a` and `b` are in the same place.
fn at_once(a: &Node, b: &Node) -> bool {
    let near = |x: f32, y: f32| (x - y).abs() < 1e-3;
    near(a.rect.y, b.rect.y) && near(a.rect.x, b.rect.x) && near(a.rect.width, b.rect.width) && near(a.rect.height, b.rect.height)
}

/// Whether a list's children (each further down than the one before) are in
/// other places than those that came in their order before.
fn slid(prev: &[Node], next: &[Node], pk: &[usize], nk: &[usize]) -> bool {
    let down = |nodes: &[Node], k: &[usize]| k.windows(2).all(|w| nodes[w[0]].rect.y <= nodes[w[1]].rect.y);
    (pk.len() != nk.len() || pk.iter().zip(nk).any(|(&i, &j)| !at_once(&prev[i], &next[j]))) && down(prev, pk) && down(next, nk)
}

/// The nodes of the first scrolling box at `rect`: where it is, and how
/// many it and what it holds are.
pub fn scroll_box(nodes: &[Node], rect: Rect) -> Option<(usize, usize)> {
    let at = nodes.iter().position(|n| n.style.scroll && n.rect == rect)?;
    Some((at, sizes(&nodes[at..])[0]))
}

/// [`changed`], for trees of different shapes too, subtree by subtree: where
/// a node holds a different number of children, its children's subtrees are
/// paired from the front and from the back (each pair compared the same
/// way, all the way down), and only those between (in either tree) are
/// damaged where they show. A layer that came or went over the rest
/// (a toast, a card) is damaged where it is, not the whole window; a list
/// that grew a row, the row and what moved. `None` when the trees share
/// nothing (no root).
fn changed_tree(prev: &[Node], next: &[Node], width: f32, height: f32) -> Option<Vec<Rect>> {
    if prev.is_empty() || next.is_empty() {
        return None;
    }
    let (ps, ns) = (sizes(prev), sizes(next));
    let mut rects: Vec<Rect> = Vec::new();
    let mut damage = |r: Rect, rects: &mut Vec<Rect>| {
        let r = snap(r, width, height);
        if r.width > 0.0 && r.height > 0.0 {
            rects.push(r);
        }
    };
    // a subtree's nodes, all damaged where they show
    fn all(nodes: &[Node], at: usize, size: usize, out: &mut Vec<Rect>, damage: &mut dyn FnMut(Rect, &mut Vec<Rect>)) {
        for n in nodes[at..at + size].iter().filter(|n| !paints_nothing(n)) {
            damage(n.visible, out);
        }
    }
    // the children's subtrees' starts, of the node at `i`
    let kids = |nodes: &[Node], size: &[usize], i: usize| -> Vec<usize> {
        let mut v = Vec::with_capacity(nodes[i].children);
        let mut at = i + 1;
        for _ in 0..nodes[i].children {
            if at >= nodes.len() {
                break;
            }
            v.push(at);
            at += size[at];
        }
        v
    };
    let mut stack = vec![(0usize, 0usize)];
    while let Some((p, n)) = stack.pop() {
        let (a, b) = (&prev[p], &next[n]);
        // (a box that paints nothing itself, moved or resized: nothing of
        // its own to paint again; what's in it says what of it changed)
        if !same_but_children(a, b) && !(paints_nothing(a) && paints_nothing(b)) {
            damage(a.visible, &mut rects);
            damage(b.visible, &mut rects);
        }
        let (pk, nk) = (kids(prev, &ps, p), kids(next, &ns, n));
        // a list built only where it shows, other rows of it now: each row
        // paired with the one in its place, wherever it comes in the list
        if (a.style.scroll || b.style.scroll) && slid(prev, next, &pk, &nk) {
            let (mut x, mut y) = (0, 0);
            while x < pk.len() || y < nk.len() {
                match (pk.get(x).copied(), nk.get(y).copied()) {
                    (Some(i), Some(j)) if at_once(&prev[i], &next[j]) => {
                        stack.push((i, j));
                        (x, y) = (x + 1, y + 1);
                    }
                    (Some(i), j) if j.is_none_or(|j| prev[i].rect.y <= next[j].rect.y) => {
                        all(prev, i, ps[i], &mut rects, &mut damage);
                        x += 1;
                    }
                    (_, Some(j)) => {
                        all(next, j, ns[j], &mut rects, &mut damage);
                        y += 1;
                    }
                    (_, None) => break,
                }
            }
            continue;
        }
        if pk.len() == nk.len() {
            stack.extend(pk.into_iter().zip(nk));
            continue;
        }
        // children came or went: paired from the front while their subtrees
        // are the same size, and from the back; those between, damaged
        let shape = |x: usize, y: usize| ps[x] == ns[y];
        let head = pk.iter().zip(&nk).take_while(|(x, y)| shape(**x, **y)).count();
        let room = pk.len().min(nk.len()) - head;
        let tail = pk.iter().rev().zip(nk.iter().rev()).take(room).take_while(|(x, y)| shape(**x, **y)).count();
        for k in 0..head {
            stack.push((pk[k], nk[k]));
        }
        for k in 0..tail {
            stack.push((pk[pk.len() - 1 - k], nk[nk.len() - 1 - k]));
        }
        for &x in &pk[head..pk.len() - tail] {
            all(prev, x, ps[x], &mut rects, &mut damage);
        }
        for &y in &nk[head..nk.len() - tail] {
            all(next, y, ns[y], &mut rects, &mut damage);
        }
    }
    Some(rects)
}

/// A box that moved, whole and as it was, from one tree to the next: its
/// subtree's place in each, and how far (UI pixels). Its pixels can be
/// copied across rather than painted again.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moved {
    pub prev: usize,
    pub next: usize,
    pub size: usize,
    pub dx: f32,
    pub dy: f32,
}

/// The boxes that only moved, from `prev` to `next`: subtrees alike node for
/// node but for all of them being `dx, dy` further (a key row the keyboard
/// pushed up, a bar under a pane that grew). Only those whose pixels are
/// theirs alone: an opaque background, square corners, no shadow, not see-
/// through, nothing painted over them before or after, unclipped, and not
/// overlapping another such box.
pub fn moved_blocks(prev: &[Node], next: &[Node]) -> Vec<Moved> {
    if prev.is_empty() || next.is_empty() {
        return Vec::new();
    }
    let (ps, ns) = (sizes(prev), sizes(next));
    let kids = |nodes: &[Node], size: &[usize], i: usize| -> Vec<usize> {
        let mut v = Vec::with_capacity(nodes[i].children);
        let mut at = i + 1;
        for _ in 0..nodes[i].children {
            if at >= nodes.len() {
                break;
            }
            v.push(at);
            at += size[at];
        }
        v
    };
    let shift = |r: Rect, dx: f32, dy: f32| Rect { x: r.x + dx, y: r.y + dy, ..r };
    // node for node the same, `dx, dy` further
    let same_moved = |p: usize, n: usize, dx: f32, dy: f32| -> bool {
        ps[p] == ns[n]
            && (0..ps[p]).all(|k| {
                let (a, b) = (&prev[p + k], &next[n + k]);
                a.children == b.children
                    && same_but_place(&a.style, &b.style)
                    && a.text == b.text
                    && a.path == b.path
                    && a.image == b.image
                    && a.turned.is_none()
                    && b.turned.is_none()
                    && a.deferred.is_none()
                    && b.deferred.is_none()
                    && shift(a.rect, dx, dy) == b.rect
                    && shift(a.visible, dx, dy) == b.visible
            })
    };
    // its pixels its own: opaque, square, no shadow, unclipped
    let solid = |n: &Node| {
        n.style.background.is_some_and(|c| c[3] >= 1.0)
            && matches!(n.style.border_radius, Length::Auto | Length::Px(0.0))
            && n.style.box_shadow.is_empty()
            && n.style.opacity >= 1.0
            && n.visible == n.rect
    };
    let mut found = Vec::new();
    let mut stack = vec![(0usize, 0usize)];
    while let Some((p, n)) = stack.pop() {
        let (a, b) = (&prev[p], &next[n]);
        let (dx, dy) = (b.rect.x - a.rect.x, b.rect.y - a.rect.y);
        if (dx != 0.0 || dy != 0.0) && solid(a) && solid(b) && same_moved(p, n, dx, dy) {
            found.push(Moved { prev: p, next: n, size: ps[p], dx, dy });
            continue;
        }
        let (pk, nk) = (kids(prev, &ps, p), kids(next, &ns, n));
        if pk.len() == nk.len() {
            stack.extend(pk.into_iter().zip(nk));
        } else {
            let head = pk.iter().zip(&nk).take_while(|(x, y)| ps[**x] == ns[**y]).count();
            let room = pk.len().min(nk.len()) - head;
            let tail = pk.iter().rev().zip(nk.iter().rev()).take(room).take_while(|(x, y)| ps[**x] == ns[**y]).count();
            for k in 0..head {
                stack.push((pk[k], nk[k]));
            }
            for k in 0..tail {
                stack.push((pk[pk.len() - 1 - k], nk[nk.len() - 1 - k]));
            }
        }
    }
    // nothing painted over one, where it was or is: what's painted after it
    // (later in the tree, or deferred) mustn't reach either place
    let over = |nodes: &[Node], from: usize, r: Rect, own: (usize, usize)| {
        nodes.iter().enumerate().any(|(i, m)| (i >= from || m.deferred.is_some()) && !(own.0..own.1).contains(&i) && intersects(m.visible, r))
    };
    found.retain(|m| {
        let (old, new) = (prev[m.prev].visible, next[m.next].visible);
        !over(prev, m.prev + m.size, old, (m.prev, m.prev + m.size))
            && !over(prev, m.prev + m.size, new, (m.prev, m.prev + m.size))
            && !over(next, m.next + m.size, old, (m.next, m.next + m.size))
            && !over(next, m.next + m.size, new, (m.next, m.next + m.size))
    });
    // nor one moved where another was or is
    let places: Vec<(Rect, Rect)> = found.iter().map(|m| (prev[m.prev].visible, next[m.next].visible)).collect();
    let alone: Vec<bool> = (0..found.len())
        .map(|i| {
            (0..found.len()).all(|j| {
                j == i || {
                    let (a, b) = (places[i], places[j]);
                    !(intersects(a.0, b.0) || intersects(a.0, b.1) || intersects(a.1, b.0) || intersects(a.1, b.1))
                }
            })
        })
        .collect();
    found.into_iter().zip(alone).filter_map(|(m, ok)| ok.then_some(m)).collect()
}

/// Two styles alike but for where they put the box (its offsets: a box
/// moved by them looks the same).
fn same_but_place(a: &Style, b: &Style) -> bool {
    if a == b {
        return true;
    }
    let mut b2 = b.clone();
    (b2.top, b2.left, b2.right, b2.bottom) = (a.top, a.left, a.right, a.bottom);
    *a == b2
}

/// What of `a` `b` doesn't cover: up to four boxes.
pub fn subtract(a: Rect, b: Rect) -> Vec<Rect> {
    if !intersects(a, b) {
        return vec![a];
    }
    let (ax1, ay1, bx1, by1) = (a.x + a.width, a.y + a.height, b.x + b.width, b.y + b.height);
    let mut out = Vec::new();
    if b.y > a.y {
        out.push(Rect { x: a.x, y: a.y, width: a.width, height: b.y - a.y });
    }
    if by1 < ay1 {
        out.push(Rect { x: a.x, y: by1, width: a.width, height: ay1 - by1 });
    }
    let (y0, y1) = (a.y.max(b.y), ay1.min(by1));
    if b.x > a.x {
        out.push(Rect { x: a.x, y: y0, width: b.x - a.x, height: y1 - y0 });
    }
    if bx1 < ax1 {
        out.push(Rect { x: bx1, y: y0, width: ax1 - bx1, height: y1 - y0 });
    }
    out.retain(|r| r.width > 0.0 && r.height > 0.0);
    out
}

/// [`changed`], for trees of different shapes too: the nodes before the first
/// that holds a different number of children (and it) are paired in order,
/// and as many as match from the end; those between, in either tree, are
/// damaged where they show. A list that grew a row repaints the list and what
/// moved below it, not the window. `None` when the trees share nothing.
fn changed_aligned(prev: &[Node], next: &[Node], width: f32, height: f32) -> Option<Vec<Rect>> {
    let n = prev.len().min(next.len());
    let same = |a: &Node, b: &Node| a.children == b.children;
    let mut head = prev.iter().zip(next).take_while(|(a, b)| same(a, b)).count();
    if head == prev.len() && head == next.len() {
        return changed(prev, next, width, height);
    }
    // the first that differs in what it holds is the same node, its contents not
    head = (head + 1).min(n);
    let tail = prev.iter().rev().zip(next.iter().rev()).take(n - head).take_while(|(a, b)| same(a, b)).count();
    if head == 0 && tail == 0 {
        return None;
    }
    let mut rects: Vec<Rect> = Vec::new();
    let mut damage = |r: Rect| {
        let r = snap(r, width, height);
        if r.width > 0.0 && r.height > 0.0 {
            rects.push(r);
        }
    };
    let paired = prev[..head].iter().zip(&next[..head]).chain(prev[prev.len() - tail..].iter().zip(&next[next.len() - tail..]));
    for (a, b) in paired {
        if a != b {
            damage(a.visible);
            damage(b.visible);
        }
    }
    for a in prev[head..prev.len() - tail].iter().chain(&next[head..next.len() - tail]) {
        damage(a.visible);
    }
    Some(rects)
}

/// [`diff`] for a painter that draws every node reaching a rectangle whole,
/// then keeps only the rectangle's pixels (as `MetalUi` repaints): nothing
/// is cut by a clip, so the rectangles needn't grow to hold nodes whole.
/// That growth is what costs: in a staggered grid (a keyboard's rows) a
/// small change grown to the keys it touches reaches their neighbours,
/// and theirs, until it covers the grid. Overlapping pieces are joined.
pub fn diff_cropped(prev: &[Node], next: &[Node], width: f32, height: f32) -> Damage {
    let mut rects = match changed_tree(prev, next, width, height).or_else(|| changed_aligned(prev, next, width, height)) {
        Some(r) if r.is_empty() => return Damage::None,
        Some(r) => r,
        None => return Damage::Full,
    };
    // (joining thousands one by one costs more than painting their union:
    // banded first, then joined)
    if rects.len() > MOST_PIECES * 8 {
        rects = bands(rects);
    }
    merge(&mut rects);
    if rects.len() > MOST_PIECES {
        // too many to paint one by one: in bands across, each what's
        // changed at those heights (a tab strip's, a status line's), not one
        // box round them all (which, a strip at the top and a line further
        // down, is most of the window)
        rects = bands(rects);
        if rects.len() > MOST_PIECES {
            let whole = rects.iter().copied().reduce(union).unwrap_or(Rect { x: 0.0, y: 0.0, width, height });
            return Damage::Regions(vec![whole]);
        }
    }
    Damage::Regions(rects)
}

/// Pieces joined into bands: those whose heights overlap (or come within
/// a few pixels), each band the box round its pieces.
fn bands(mut rects: Vec<Rect>) -> Vec<Rect> {
    rects.sort_by(|a, b| a.y.total_cmp(&b.y));
    let mut out: Vec<Rect> = Vec::new();
    for r in rects {
        match out.last_mut() {
            Some(b) if r.y <= b.y + b.height + 4.0 => *b = union(*b, r),
            _ => out.push(r),
        }
    }
    out
}

/// A pixel of margin on every side of a damaged rectangle: the edge of a rounded shape or a stroke is
/// anti-aliased, and its faint pixels fall just outside the box it was laid out in.
const MARGIN: f32 = 1.0;

/// Whole pixels, outward with a margin, inside the window.
pub fn snap(r: Rect, width: f32, height: f32) -> Rect {
    let (x0, y0) = (
        (r.x.floor() - MARGIN).max(0.0),
        (r.y.floor() - MARGIN).max(0.0),
    );
    let (x1, y1) = (
        (r.x + r.width).ceil().min(width + MARGIN) + MARGIN,
        (r.y + r.height).ceil().min(height + MARGIN) + MARGIN,
    );
    let (x1, y1) = (x1.min(width), y1.min(height));
    Rect {
        x: x0,
        y: y0,
        width: (x1 - x0).max(0.0),
        height: (y1 - y0).max(0.0),
    }
}

/// What a node's drawing needs whole: text lines, and shapes with rounded corners (clipped, they are drawn square).
fn needs_whole(n: &Node) -> bool {
    n.text.is_some()
        || n.path.is_some()
        || n.image.is_some()
        || (n.style.background.is_some()
            && !matches!(n.style.border_radius, Length::Auto | Length::Px(0.0)))
}

/// Joins rectangles that overlap, until none do.
pub fn merge(rects: &mut Vec<Rect>) {
    let mut again = true;
    while again {
        again = false;
        'scan: for i in 0..rects.len() {
            for j in i + 1..rects.len() {
                if intersects(rects[i], rects[j]) {
                    rects[i] = union(rects[i], rects[j]);
                    rects.swap_remove(j);
                    again = true;
                    break 'scan;
                }
            }
        }
    }
}

/// Past this many pieces, one box round them all is cheaper than painting each.
const MOST_PIECES: usize = 48;

/// The damage between the last frame's nodes and this frame's, in a window `width` by `height`.
pub fn diff(prev: &[Node], next: &[Node], width: f32, height: f32) -> Damage {
    let mut rects = match changed(prev, next, width, height) {
        Some(r) if r.is_empty() => return Damage::None,
        Some(r) => r,
        None => return Damage::Full,
    };
    // Widen a piece until nothing that needs to be whole is cut by its edge, and join pieces that meet.
    for _ in 0..32 {
        merge(&mut rects);
        let mut grown = false;
        for region in rects.iter_mut() {
            for n in prev.iter().chain(next) {
                if !needs_whole(n) {
                    continue;
                }
                // Only the part inside the window counts: a node that runs past its edge (a scrolled
                // panel) could never be held whole by a region, which stops at the edge.
                let inside = snap(n.visible, width, height);
                if inside.width > 0.0
                    && inside.height > 0.0
                    && intersects(*region, inside)
                    && !contains(*region, inside)
                {
                    *region = snap(union(*region, inside), width, height);
                    grown = true;
                }
            }
        }
        if !grown {
            if rects.len() > MOST_PIECES {
                let whole = rects.iter().copied().reduce(union).unwrap_or(Rect {
                    x: 0.0,
                    y: 0.0,
                    width,
                    height,
                });
                return Damage::Regions(vec![whole]);
            }
            return Damage::Regions(rects);
        }
    }
    Damage::Full
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edges, FixedWidthText};

    fn tree(label: &str, bar: f32) -> Ui {
        let mut ui = Ui::new();
        let root = ui.root();
        let a = ui.add_box(
            root,
            Style {
                width: Length::Px(100.0),
                height: Length::Px(40.0),
                background: Some([1.0, 0.0, 0.0, 1.0]),
                border_radius: Length::Px(6.0),
                ..Style::default()
            },
        );
        ui.add_text(a, label);
        ui.add_box(
            root,
            Style {
                width: Length::Px(bar),
                height: Length::Px(20.0),
                margin: Edges {
                    top: 10.0,
                    ..Default::default()
                },
                background: Some([0.0, 1.0, 0.0, 1.0]),
                ..Style::default()
            },
        );
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        ui
    }

    #[test]
    fn the_same_tree_has_no_damage_and_a_changed_one_has_only_its_part() {
        let a = collect(&tree("x", 50.0));
        assert_eq!(
            diff(&a, &collect(&tree("x", 50.0)), 400.0, 300.0),
            Damage::None
        );
        // The bar got longer: only it (and what it overlaps) is damaged, not the rounded box above it.
        let Damage::Regions(rs) = diff(&a, &collect(&tree("x", 80.0)), 400.0, 300.0) else {
            panic!("regions")
        };
        assert_eq!(rs.len(), 1);
        assert!(rs[0].y >= 49.0 && rs[0].width <= 90.0, "{:?}", rs[0]);
    }

    #[test]
    fn a_cut_through_text_or_a_rounded_corner_widens_the_damage() {
        let a = collect(&tree("x", 50.0));
        // The label changed: the box that holds it has rounded corners, so the whole box is repainted.
        let Damage::Regions(rs) = diff(&a, &collect(&tree("y", 50.0)), 400.0, 300.0) else {
            panic!("regions")
        };
        assert!(
            rs[0].x <= 0.0 && rs[0].width >= 100.0 && rs[0].height >= 40.0,
            "{:?}",
            rs[0]
        );
    }

    #[test]
    fn a_path_whose_inherited_colour_changes_is_damaged() {
        let icon = |c: &str| {
            let mut ui = Ui::new();
            let root = ui.root();
            let b = ui.add_box(root, Default::default());
            ui.update_style(b, |s| crate::css::apply(s, "color", c));
            let size = crate::style! { width: 20px; height: 20px; };
            crate::svg::add_icon(&mut ui, b, size, r#"<path d="M5 12H19"/>"#, 2.0);
            ui.layout(100.0, 100.0, &mut crate::FixedWidthText(7.0));
            collect(&ui)
        };
        assert_eq!(diff(&icon("#ff0000"), &icon("#ff0000"), 100.0, 100.0), Damage::None);
        assert_ne!(diff(&icon("#ff0000"), &icon("#0000ff"), 100.0, 100.0), Damage::None);
    }

    #[test]
    fn a_different_shape_is_damaged_everywhere() {
        let a = collect(&tree("x", 50.0));
        let mut ui = tree("x", 50.0);
        let root = ui.root();
        ui.add_box(root, Style::default());
        assert_eq!(diff(&a, &collect(&ui), 400.0, 300.0), Damage::Full);
    }

    /// Cropped, a layer added over the rest (a toast) is damaged where it is,
    /// not the whole window; and taken away, the same.
    #[test]
    fn a_layer_added_over_the_rest_damages_only_it() {
        let ui = |toast: bool| {
            let mut ui = Ui::new();
            let root = ui.root();
            let side = ui.add_box(root, crate::style! { width: 100px; height: 300px; background: #ff0000; });
            ui.add_text(side, "sidebar");
            let main = ui.add_box(root, crate::style! { position: absolute; left: 100px; top: 0px; width: 300px; height: 300px; background: #00ff00; });
            ui.add_text(main, "main");
            if toast {
                let t = ui.add_box(root, crate::style! { position: absolute; left: 150px; top: 250px; width: 100px; height: 20px; background: #000000; });
                ui.add_text(t, "Copied");
            }
            ui.layout(400.0, 300.0, &mut crate::FixedWidthText(7.0));
            collect(&ui)
        };
        for (a, b) in [(ui(false), ui(true)), (ui(true), ui(false))] {
            let Damage::Regions(rs) = diff_cropped(&a, &b, 400.0, 300.0) else { panic!("regions, not everything") };
            let all = rs.iter().copied().reduce(union).unwrap();
            assert!(all.width <= 104.0 && all.height <= 24.0 && all.y >= 248.0, "the toast only: {all:?}");
        }
    }

    /// A solid bar pushed up (a keyboard came) is found as moved, whole; a
    /// rounded or see-through one isn't (its pixels aren't its own alone).
    #[test]
    fn a_solid_box_moved_whole_is_found() {
        let ui = |y: f32, round: bool| {
            let mut ui = Ui::new();
            let root = ui.root();
            let top = ui.add_box(root, crate::style! { position: absolute; left: 0px; top: 0px; width: 400px; height: 100px; background: #222222; });
            ui.add_text(top, "terminal");
            let mut bar = crate::style! { position: absolute; left: 0px; width: 400px; height: 40px; background: #333333; };
            bar.top = Length::Px(y);
            if round {
                bar.border_radius = Length::Px(8.0);
            }
            let b = ui.add_box(root, bar);
            ui.add_text(b, "esc ctrl alt");
            ui.layout(400.0, 600.0, &mut crate::FixedWidthText(7.0));
            collect(&ui)
        };
        let m = moved_blocks(&ui(500.0, false), &ui(300.0, false));
        assert_eq!(m.len(), 1);
        assert_eq!((m[0].dx, m[0].dy, m[0].size), (0.0, -200.0, 2));
        assert!(moved_blocks(&ui(500.0, true), &ui(300.0, true)).is_empty(), "rounded: not its own corners");
        assert!(moved_blocks(&ui(500.0, false), &ui(500.0, false)).is_empty(), "not moved");
        let left = subtract(Rect { x: 0.0, y: 500.0, width: 400.0, height: 40.0 }, Rect { x: 0.0, y: 520.0, width: 400.0, height: 40.0 });
        assert_eq!(left, vec![Rect { x: 0.0, y: 500.0, width: 400.0, height: 20.0 }]);
    }

    /// Cropped, many small changes far apart (a strip at the top, a line
    /// lower down) are painted as bands, not as one box round them all.
    #[test]
    fn many_scattered_changes_are_painted_in_bands() {
        let ui = |v: usize| {
            let mut ui = Ui::new();
            let root = ui.root();
            let top = ui.add_box(root, crate::style! { position: absolute; left: 0px; top: 0px; width: 800px; height: 20px; display: flex; });
            for i in 0..60 {
                ui.add_text(top, if (i + v) % 2 == 0 { "a" } else { "b" });
            }
            let low = ui.add_box(root, crate::style! { position: absolute; left: 0px; top: 500px; width: 800px; height: 20px; display: flex; });
            for i in 0..60 {
                ui.add_text(low, if (i + v) % 2 == 0 { "c" } else { "d" });
            }
            ui.layout(800.0, 600.0, &mut crate::FixedWidthText(7.0));
            collect(&ui)
        };
        let Damage::Regions(rs) = diff_cropped(&ui(0), &ui(1), 800.0, 600.0) else { panic!("regions") };
        let painted: f32 = rs.iter().map(|r| r.width * r.height).sum();
        assert!(painted < 800.0 * 600.0 * 0.2, "two bands, not the window: {rs:?}");
    }

    /// Cropped, a tree that grew a row is damaged where the row came and
    /// what it moved, not everywhere; the boxes beside it stay.
    #[test]
    fn a_row_added_damages_the_row_and_what_moved() {
        let list = |rows: usize| {
            let mut ui = Ui::new();
            let root = ui.root();
            ui.set_style(root, crate::style! { display: flex; });
            let side = ui.add_box(root, crate::style! { width: 100px; });
            for _ in 0..rows {
                ui.add_box(side, crate::style! { height: 20px; background: #ff0000; });
            }
            ui.add_box(side, crate::style! { height: 20px; background: #0000ff; });
            let other = ui.add_box(root, crate::style! { width: 100px; height: 300px; background: #00ff00; });
            ui.add_text(other, "beside");
            ui.layout(400.0, 300.0, &mut crate::FixedWidthText(7.0));
            collect(&ui)
        };
        let Damage::Regions(rs) = diff_cropped(&list(3), &list(4), 400.0, 300.0) else {
            panic!("regions, not everything")
        };
        let all = rs.iter().copied().reduce(union).unwrap();
        assert!(all.x <= 0.0 && all.y <= 60.0 && all.y + all.height >= 100.0, "the new row and the blue moved below it: {all:?}");
        assert!(all.x + all.width <= 102.0, "not the box beside: {all:?}");
    }

    #[test]
    fn separate_changes_stay_separate_and_overlapping_ones_join() {
        let mut rects = vec![
            Rect {
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 10.0,
            },
            Rect {
                x: 100.0,
                y: 100.0,
                width: 10.0,
                height: 10.0,
            },
            Rect {
                x: 5.0,
                y: 5.0,
                width: 10.0,
                height: 10.0,
            },
        ];
        merge(&mut rects);
        assert_eq!(rects.len(), 2);
        assert!(rects.iter().any(|r| r.width == 15.0 && r.height == 15.0));
    }

    #[test]
    fn snapping_goes_outward_and_stays_inside() {
        let r = snap(
            Rect {
                x: 10.4,
                y: -3.0,
                width: 5.2,
                height: 8.0,
            },
            100.0,
            100.0,
        );
        // One pixel more on each side, for the edge of anything antialiased.
        assert_eq!((r.x, r.y, r.width, r.height), (9.0, 0.0, 8.0, 6.0));
    }
}
