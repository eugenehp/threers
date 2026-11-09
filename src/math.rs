//! Math primitives. Mirrors three.js's math module.
//!
//! All of it is the [`threers-math`](https://docs.rs/threers-math) crate,
//! re-exported, so it can be used without the rest of threers; the one thing
//! added here is [`triangulate_polygon`], which uses the ear clipper in
//! [`curves`](crate::curves).

pub use threers_math::*;

/// Triangles filling a polygon with holes, by ear clipping.
///
/// The polygon outline `outer` and every hole are rings of 2D points, in
/// either winding; a ring may repeat its first point at the end, as GeoJSON
/// rings do, and that closing point is dropped. Indices count the points of
/// `outer` and then of each hole in turn, *after* dropping it — the order
/// [`polygon_points`] returns them in.
///
/// Unlike [`delaunay`], which triangulates the convex hull of scattered
/// points, this keeps to the outline: a C-shaped country stays C-shaped.
///
/// ```
/// use threers::math::{polygon_points, triangulate_polygon};
/// let square = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]];
/// let hole = vec![[1.0, 1.0], [1.0, 3.0], [3.0, 3.0], [3.0, 1.0]];
/// let tris = triangulate_polygon(&square, &[hole.clone()]);
/// assert_eq!(polygon_points(&square, &[hole]).len(), 8);
/// assert_eq!(tris.len(), 8);
/// ```
pub fn triangulate_polygon(outer: &[[f64; 2]], holes: &[Vec<[f64; 2]>]) -> Vec<[u32; 3]> {
    use crate::curves::shapes::{shape_utils, Vec2d};
    let ring = |r: &[[f64; 2]]| -> Vec<Vec2d> {
        open_ring(r)
            .iter()
            .map(|p| Vec2d::new(p[0], p[1]))
            .collect()
    };
    let mut contour = ring(outer);
    let mut hs: Vec<Vec<Vec2d>> = holes.iter().map(|h| ring(h)).collect();
    if contour.len() < 3 {
        return Vec::new();
    }
    shape_utils::triangulate_shape(&mut contour, &mut hs)
}
