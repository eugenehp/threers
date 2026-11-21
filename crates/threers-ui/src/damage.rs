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
    fn walk(ui: &Ui, out: &mut Vec<Node>, id: NodeId, clip: Option<Rect>) {
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
        let visible = clip.map_or(rect, |c| intersection(rect, c));
        out.push(Node {
            rect,
            visible,
            // A flicker shows as the opacity it paints at.
            style: {
                let mut s = ui.style(id).clone();
                s.opacity = ui.painted_opacity(id);
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
            children: ui.children(id).len(),
        });
        // What its children can show in: cut to it if it clips; anywhere,
        // once transformed (their boxes are then not where they show).
        let inner = if ui.paint_transform(id).is_some() {
            None
        } else if ui.style(id).clip {
            Some(clip.map_or(rect, |c| intersection(rect, c)))
        } else {
            clip
        };
        for &child in ui.children(id) {
            walk(ui, out, child, inner);
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
            children: ui.children(id).len(),
        });
        for &child in ui.children(id) {
            hidden(ui, out, child);
        }
    }
    let mut out = Vec::with_capacity(ui.nodes.len());
    walk(ui, &mut out, ui.root(), None);
    out
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

/// [`diff`] for a painter that draws every node reaching a rectangle whole,
/// then keeps only the rectangle's pixels (as `MetalUi` repaints): nothing
/// is cut by a clip, so the rectangles needn't grow to hold nodes whole.
/// That growth is what costs: in a staggered grid (a keyboard's rows) a
/// small change grown to the keys it touches reaches their neighbours,
/// and theirs, until it covers the grid. Overlapping pieces are joined.
pub fn diff_cropped(prev: &[Node], next: &[Node], width: f32, height: f32) -> Damage {
    let mut rects = match changed(prev, next, width, height) {
        Some(r) if r.is_empty() => return Damage::None,
        Some(r) => r,
        None => return Damage::Full,
    };
    merge(&mut rects);
    if rects.len() > MOST_PIECES {
        let whole = rects.iter().copied().reduce(union).unwrap_or(Rect { x: 0.0, y: 0.0, width, height });
        return Damage::Regions(vec![whole]);
    }
    Damage::Regions(rects)
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
    fn a_different_shape_is_damaged_everywhere() {
        let a = collect(&tree("x", 50.0));
        let mut ui = tree("x", 50.0);
        let root = ui.root();
        ui.add_box(root, Style::default());
        assert_eq!(diff(&a, &collect(&ui), 400.0, 300.0), Damage::Full);
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
