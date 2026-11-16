//! OpenSCAD's 2D booleans and `offset()` on [`crate::math::polygon_ops`], as
//! OpenSCAD itself does them: in integer coordinates, polygons oriented
//! (outers counter-clockwise, holes clockwise) and filled non-zero, so that
//! touching and overlapping pieces merge exactly and offsets come out free
//! of self-intersections.

use super::{Poly, Shape};
use crate::math::polygon_ops::{
    is_positive, Boolean64, ClipType, EndType, FillRule, JoinType, Path64, PathOffset, Paths64,
    Point64, PolyPath64,
};

/// Bits the largest coordinate is scaled to: well inside the 62 the
/// algorithms take, and far finer than an `f32` coordinate.
const SCALE_BITS: i32 = 40;

/// How a corner is drawn by [`offset`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Join {
    /// `offset(r = …)`: arcs round the corners.
    Round,
    /// `offset(delta = …)`: corners kept sharp.
    Miter,
    /// `offset(delta = …, chamfer = true)`: corners cut square.
    Chamfer,
}

/// The largest coordinate of the shapes, in either axis.
fn extent(shapes: &[&Shape]) -> f64 {
    shapes
        .iter()
        .flat_map(|s| s.polys.iter())
        .flat_map(|p| p.outer.iter().chain(p.holes.iter().flatten()))
        .fold(0.0f64, |m, q| {
            m.max(f64::from(q[0].abs())).max(f64::from(q[1].abs()))
        })
}

/// The power of two coordinates up to `extent` are multiplied by.
fn scale_for(extent: f64) -> f64 {
    if extent <= 0.0 {
        return 2f64.powi(SCALE_BITS / 2);
    }

    2f64.powi(SCALE_BITS - extent.log2().ceil() as i32)
}

fn ring_path(ring: &[[f32; 2]], scale: f64, positive: bool) -> Path64 {
    let mut p: Path64 = ring
        .iter()
        .map(|q| {
            Point64::new(
                (f64::from(q[0]) * scale).round() as i64,
                (f64::from(q[1]) * scale).round() as i64,
            )
        })
        .collect();

    if p.len() >= 3 && is_positive(&p) != positive {
        p.reverse();
    }

    p
}

/// A shape's rings, oriented for non-zero filling.
fn shape_paths(sh: &Shape, scale: f64) -> Paths64 {
    let mut out = Paths64::new();

    for p in &sh.polys {
        out.push(ring_path(&p.outer, scale, true));
        out.extend(p.holes.iter().map(|h| ring_path(h, scale, false)));
    }

    out.retain(|p| p.len() >= 3);
    out
}

fn ring_of(path: &Path64, scale: f64) -> Vec<[f32; 2]> {
    path.iter()
        .map(|q| [(q.x as f64 / scale) as f32, (q.y as f64 / scale) as f32])
        .collect()
}

/// The polygons of a result tree: each outer with the holes directly in it,
/// and the islands in those holes as polygons of their own.
fn tree_shape(root: &PolyPath64, scale: f64) -> Shape {
    fn outers(node: &PolyPath64, scale: f64, polys: &mut Vec<Poly>) {
        for outer in &node.childs {
            polys.push(Poly {
                outer: ring_of(&outer.polygon, scale),
                holes: outer
                    .childs
                    .iter()
                    .map(|h| ring_of(&h.polygon, scale))
                    .collect(),
            });

            for hole in &outer.childs {
                outers(hole, scale, polys);
            }
        }
    }

    let mut polys = Vec::new();
    outers(root, scale, &mut polys);
    Shape { polys }
}

/// `subject op clip`, non-zero filled; a union of the subjects alone when
/// `clip` is empty.
pub(super) fn boolean(op: ClipType, subject: &[&Shape], clip: &[&Shape]) -> Shape {
    let all: Vec<&Shape> = subject.iter().chain(clip).copied().collect();
    let scale = scale_for(extent(&all));
    let mut c = Boolean64::new();
    c.add_subject(
        &subject
            .iter()
            .flat_map(|s| shape_paths(s, scale))
            .collect::<Paths64>(),
    );
    c.add_clip(
        &clip
            .iter()
            .flat_map(|s| shape_paths(s, scale))
            .collect::<Paths64>(),
    );
    let (_, tree, _) = c.execute_tree(op, FillRule::NonZero);
    tree_shape(&tree, scale)
}

/// `offset()`: the shape grown by `d` (shrunk where it is negative), round
/// corners within `arc_tolerance` of the true arc.
pub(super) fn offset(sh: &Shape, d: f32, join: Join, arc_tolerance: f64) -> Shape {
    // room for the growth
    let scale = scale_for(extent(&[sh]) + f64::from(d.abs()));
    let join = match join {
        Join::Round => JoinType::Round,
        Join::Miter => JoinType::Miter,
        Join::Chamfer => JoinType::Square,
    };

    // OpenSCAD's miter limit: corners stay sharp at any angle
    let mut o = PathOffset::new(1_000_000.0, arc_tolerance * scale, false, false);
    o.add_paths(&shape_paths(sh, scale), join, EndType::Polygon);
    let tree = o.execute_tree(f64::from(d) * scale);
    tree_shape(&tree, scale)
}

