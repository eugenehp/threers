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

/// One node of a laid-out tree, as far as painting can tell.
#[derive(Clone, PartialEq)]
pub struct Node {
    pub rect: Rect,
    pub style: Style,
    pub text: Option<String>,
    /// A drawn path: its data, view box, stroke and fill.
    pub path: Option<(Vec<String>, [f32; 4], Option<f32>, bool)>,
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
    fn walk(ui: &Ui, out: &mut Vec<Node>, id: NodeId) {
        out.push(Node {
            rect: ui.rect(id),
            style: ui.style(id).clone(),
            text: match ui.kind(id) {
                NodeKind::Text(t) => Some(t.clone()),
                _ => None,
            },
            path: match ui.kind(id) {
                NodeKind::Path { d, view_box, stroke, fill } => Some((d.clone(), *view_box, *stroke, *fill)),
                _ => None,
            },
            children: ui.children(id).len(),
        });
        for &child in ui.children(id) {
            walk(ui, out, child);
        }
    }
    let mut out = Vec::new();
    walk(ui, &mut out, ui.root());
    out
}

pub fn union(a: Rect, b: Rect) -> Rect {
    let (x0, y0) = (a.x.min(b.x), a.y.min(b.y));
    let (x1, y1) = ((a.x + a.width).max(b.x + b.width), (a.y + a.height).max(b.y + b.height));
    Rect { x: x0, y: y0, width: x1 - x0, height: y1 - y0 }
}

fn intersects(a: Rect, b: Rect) -> bool {
    a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
}

fn contains(outer: Rect, inner: Rect) -> bool {
    inner.x >= outer.x && inner.y >= outer.y && inner.x + inner.width <= outer.x + outer.width && inner.y + inner.height <= outer.y + outer.height
}

/// A pixel of margin on every side of a damaged rectangle: the edge of a rounded shape or a stroke is
/// anti-aliased, and its faint pixels fall just outside the box it was laid out in.
const MARGIN: f32 = 1.0;

/// Whole pixels, outward with a margin, inside the window.
pub fn snap(r: Rect, width: f32, height: f32) -> Rect {
    let (x0, y0) = ((r.x.floor() - MARGIN).max(0.0), (r.y.floor() - MARGIN).max(0.0));
    let (x1, y1) = ((r.x + r.width).ceil().min(width + MARGIN) + MARGIN, (r.y + r.height).ceil().min(height + MARGIN) + MARGIN);
    let (x1, y1) = (x1.min(width), y1.min(height));
    Rect { x: x0, y: y0, width: (x1 - x0).max(0.0), height: (y1 - y0).max(0.0) }
}

/// What a node's drawing needs whole: text lines, and shapes with rounded corners (clipped, they are drawn square).
fn needs_whole(n: &Node) -> bool {
    n.text.is_some() || n.path.is_some() || (n.style.background.is_some() && !matches!(n.style.border_radius, Length::Auto | Length::Px(0.0)))
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
    if prev.len() != next.len() || prev.iter().zip(next).any(|(a, b)| a.children != b.children) {
        return Damage::Full;
    }
    let mut rects: Vec<Rect> = Vec::new();
    for (a, b) in prev.iter().zip(next) {
        if a != b {
            for r in [a.rect, b.rect] {
                let r = snap(r, width, height);
                if r.width > 0.0 && r.height > 0.0 {
                    rects.push(r);
                }
            }
        }
    }
    if rects.is_empty() {
        return Damage::None;
    }
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
                let inside = snap(n.rect, width, height);
                if inside.width > 0.0 && inside.height > 0.0 && intersects(*region, inside) && !contains(*region, inside) {
                    *region = snap(union(*region, inside), width, height);
                    grown = true;
                }
            }
        }
        if !grown {
            if rects.len() > MOST_PIECES {
                let whole = rects.iter().copied().reduce(union).unwrap_or(Rect { x: 0.0, y: 0.0, width, height });
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
    use crate::ui::{Edges, FixedWidthText};

    fn tree(label: &str, bar: f32) -> Ui {
        let mut ui = Ui::new();
        let root = ui.root();
        let a = ui.add_box(root, Style { width: Length::Px(100.0), height: Length::Px(40.0), background: Some([1.0, 0.0, 0.0, 1.0]), border_radius: Length::Px(6.0), ..Style::default() });
        ui.add_text(a, label);
        ui.add_box(root, Style { width: Length::Px(bar), height: Length::Px(20.0), margin: Edges { top: 10.0, ..Default::default() }, background: Some([0.0, 1.0, 0.0, 1.0]), ..Style::default() });
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        ui
    }

    #[test]
    fn the_same_tree_has_no_damage_and_a_changed_one_has_only_its_part() {
        let a = collect(&tree("x", 50.0));
        assert_eq!(diff(&a, &collect(&tree("x", 50.0)), 400.0, 300.0), Damage::None);
        // The bar got longer: only it (and what it overlaps) is damaged, not the rounded box above it.
        let Damage::Regions(rs) = diff(&a, &collect(&tree("x", 80.0)), 400.0, 300.0) else { panic!("regions") };
        assert_eq!(rs.len(), 1);
        assert!(rs[0].y >= 49.0 && rs[0].width <= 90.0, "{:?}", rs[0]);
    }

    #[test]
    fn a_cut_through_text_or_a_rounded_corner_widens_the_damage() {
        let a = collect(&tree("x", 50.0));
        // The label changed: the box that holds it has rounded corners, so the whole box is repainted.
        let Damage::Regions(rs) = diff(&a, &collect(&tree("y", 50.0)), 400.0, 300.0) else { panic!("regions") };
        assert!(rs[0].x <= 0.0 && rs[0].width >= 100.0 && rs[0].height >= 40.0, "{:?}", rs[0]);
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
            Rect { x: 0.0, y: 0.0, width: 10.0, height: 10.0 },
            Rect { x: 100.0, y: 100.0, width: 10.0, height: 10.0 },
            Rect { x: 5.0, y: 5.0, width: 10.0, height: 10.0 },
        ];
        merge(&mut rects);
        assert_eq!(rects.len(), 2);
        assert!(rects.iter().any(|r| r.width == 15.0 && r.height == 15.0));
    }

    #[test]
    fn snapping_goes_outward_and_stays_inside() {
        let r = snap(Rect { x: 10.4, y: -3.0, width: 5.2, height: 8.0 }, 100.0, 100.0);
        // One pixel more on each side, for the edge of anything antialiased.
        assert_eq!((r.x, r.y, r.width, r.height), (9.0, 0.0, 8.0, 6.0));
    }
}
