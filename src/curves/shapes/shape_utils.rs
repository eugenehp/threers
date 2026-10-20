//! three.js `ShapeUtils`.

use super::curve::Vec2d;
use super::earcut;

/// `ShapeUtils.area(contour)`: signed area, positive counter-clockwise.
pub fn area(contour: &[Vec2d]) -> f64 {
    let n = contour.len();
    let mut a = 0.0;
    if n == 0 {
        return a;
    }
    let mut p = n - 1;
    for q in 0..n {
        a += contour[p].x * contour[q].y - contour[q].x * contour[p].y;
        p = q;
    }
    a * 0.5
}

/// `ShapeUtils.isClockWise(pts)`.
pub fn is_clock_wise(pts: &[Vec2d]) -> bool {
    area(pts) < 0.0
}

/// three.js's `removeDupEndPts`: drop a closing point equal to the first.
/// Returns whether it dropped one.
pub fn remove_dup_end_pts(points: &mut Vec<Vec2d>) -> bool {
    let l = points.len();
    if l > 2 && points[l - 1] == points[0] {
        points.pop();
        return true;
    }
    false
}

/// `ShapeUtils.triangulateShape(contour, holes)`: Earcut over the contour
/// then the holes, as one index space. Like three.js it first removes a
/// repeated closing point from the contour and every hole, in place — callers
/// that go on to use those arrays (both shape geometries do) see them shorter.
pub fn triangulate_shape(contour: &mut Vec<Vec2d>, holes: &mut [Vec<Vec2d>]) -> Vec<[u32; 3]> {
    let mut vertices = Vec::new();
    let mut hole_indices = Vec::with_capacity(holes.len());
    remove_dup_end_pts(contour);
    for p in contour.iter() {
        vertices.push(p.x);
        vertices.push(p.y);
    }
    let mut hole_index = contour.len();
    for h in holes.iter_mut() {
        remove_dup_end_pts(h);
    }
    for h in holes.iter() {
        hole_indices.push(hole_index);
        hole_index += h.len();
        for p in h {
            vertices.push(p.x);
            vertices.push(p.y);
        }
    }
    earcut::triangulate(&vertices, &hole_indices, 2)
        .chunks_exact(3)
        .map(|t| [t[0], t[1], t[2]])
        .collect()
}
