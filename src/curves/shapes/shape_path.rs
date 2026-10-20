//! three.js `ShapePath`: subpaths drawn with a pen, sorted into shapes and
//! holes by `toShapes`.

use super::curve::Vec2d;
use super::path::{Path, Shape};
use super::shape_utils::is_clock_wise;

/// three.js `ShapePath`. `moveTo` starts a new subpath.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShapePath {
    pub sub_paths: Vec<Path>,
}

impl ShapePath {
    pub fn new() -> Self {
        Self::default()
    }

    fn current(&mut self) -> &mut Path {
        if self.sub_paths.is_empty() {
            // three.js throws here (no current path); start one at the origin.
            self.sub_paths.push(Path::new());
        }
        self.sub_paths.last_mut().unwrap()
    }

    pub fn move_to(&mut self, x: f64, y: f64) -> &mut Self {
        let mut p = Path::new();
        p.move_to(x, y);
        self.sub_paths.push(p);
        self
    }

    pub fn line_to(&mut self, x: f64, y: f64) -> &mut Self {
        self.current().line_to(x, y);
        self
    }

    pub fn quadratic_curve_to(&mut self, cpx: f64, cpy: f64, x: f64, y: f64) -> &mut Self {
        self.current().quadratic_curve_to(cpx, cpy, x, y);
        self
    }

    pub fn bezier_curve_to(
        &mut self,
        c1x: f64,
        c1y: f64,
        c2x: f64,
        c2y: f64,
        x: f64,
        y: f64,
    ) -> &mut Self {
        self.current().bezier_curve_to(c1x, c1y, c2x, c2y, x, y);
        self
    }

    pub fn spline_thru(&mut self, pts: &[Vec2d]) -> &mut Self {
        self.current().spline_thru(pts);
        self
    }

    /// `toShapes(isCCW)`.
    pub fn to_shapes(&self, is_ccw: bool) -> Vec<Shape> {
        to_shapes(&self.sub_paths, is_ccw)
    }
}

/// A shape made from a subpath: its curves only — three.js assigns
/// `shape.curves = path.curves`, so `autoClose` and the pen stay default.
fn shape_of(path: &Path) -> Shape {
    Shape::from_path(Path {
        curves: path.curves.clone(),
        ..Path::default()
    })
}

fn is_point_inside_polygon(in_pt: Vec2d, polygon: &[Vec2d]) -> bool {
    let len = polygon.len();
    let mut inside = false;
    if len == 0 {
        return false;
    }
    let mut p = len - 1;
    for q in 0..len {
        let mut low = polygon[p];
        let mut high = polygon[q];
        let mut dx = high.x - low.x;
        let mut dy = high.y - low.y;
        if dy.abs() > f64::EPSILON {
            if dy < 0.0 {
                low = polygon[q];
                dx = -dx;
                high = polygon[p];
                dy = -dy;
            }
            if in_pt.y < low.y || in_pt.y > high.y {
                p = q;
                continue;
            }
            if in_pt.y == low.y {
                if in_pt.x == low.x {
                    return true;
                }
            } else {
                let perp = dy * (in_pt.x - low.x) - dx * (in_pt.y - low.y);
                if perp == 0.0 {
                    return true;
                }
                if perp < 0.0 {
                    p = q;
                    continue;
                }
                inside = !inside;
            }
        } else {
            if in_pt.y != low.y {
                p = q;
                continue;
            }
            if (high.x <= in_pt.x && in_pt.x <= low.x) || (low.x <= in_pt.x && in_pt.x <= high.x) {
                return true;
            }
        }
        p = q;
    }
    inside
}

/// `ShapePath.toShapes(isCCW)` over `sub_paths`: solids become shapes, the
/// others holes of the shape that contains them.
pub fn to_shapes(sub_paths: &[Path], is_ccw: bool) -> Vec<Shape> {
    if sub_paths.is_empty() {
        return Vec::new();
    }
    if sub_paths.len() == 1 {
        return vec![shape_of(&sub_paths[0])];
    }
    let mut holes_first = !is_clock_wise(&sub_paths[0].get_points(12));
    if is_ccw {
        holes_first = !holes_first;
    }

    struct Solid {
        shape: Shape,
        points: Vec<Vec2d>,
    }
    struct Hole {
        path: usize,
        first: Vec2d,
    }
    let mut new_shapes: Vec<Option<Solid>> = vec![None];
    let mut new_shape_holes: Vec<Vec<Hole>> = vec![Vec::new()];
    let mut main_idx = 0usize;

    let ensure = |v: &mut Vec<Option<Solid>>, h: &mut Vec<Vec<Hole>>, i: usize| {
        while v.len() <= i {
            v.push(None);
        }
        while h.len() <= i {
            h.push(Vec::new());
        }
    };

    for (i, path) in sub_paths.iter().enumerate() {
        let points = path.get_points(12);
        let mut solid = is_clock_wise(&points);
        if is_ccw {
            solid = !solid;
        }
        if solid {
            if !holes_first && new_shapes.get(main_idx).is_some_and(|s| s.is_some()) {
                main_idx += 1;
            }
            ensure(&mut new_shapes, &mut new_shape_holes, main_idx);
            new_shapes[main_idx] = Some(Solid {
                shape: shape_of(path),
                points,
            });
            if holes_first {
                main_idx += 1;
            }
            ensure(&mut new_shapes, &mut new_shape_holes, main_idx);
            new_shape_holes[main_idx] = Vec::new();
        } else {
            ensure(&mut new_shapes, &mut new_shape_holes, main_idx);
            let first = points.first().copied().unwrap_or_default();
            new_shape_holes[main_idx].push(Hole { path: i, first });
        }
    }

    // Solids fill `new_shapes` from 0 without gaps; `ensure` may have added a
    // trailing empty slot JavaScript's array never had (holes collected there,
    // after the last solid when holes come first, are dropped as in three.js).
    let shapes_len = new_shapes
        .iter()
        .rposition(|s| s.is_some())
        .map_or(0, |i| i + 1);
    if new_shapes.first().is_none_or(|s| s.is_none()) {
        return sub_paths.iter().map(shape_of).collect();
    }
    new_shapes.truncate(shapes_len);

    if new_shapes.len() > 1 {
        let mut ambiguous = false;
        let mut to_change = 0;
        let mut better: Vec<Vec<Hole>> = (0..new_shapes.len()).map(|_| Vec::new()).collect();
        #[allow(clippy::needless_range_loop)]
        for s_idx in 0..new_shapes.len() {
            let sho = std::mem::take(&mut new_shape_holes[s_idx]);
            for ho in sho {
                let mut unassigned = true;
                let mut target = None;
                for (s2_idx, s2) in new_shapes.iter().enumerate() {
                    let s2 = s2.as_ref().expect("solid");
                    if is_point_inside_polygon(ho.first, &s2.points) {
                        if s_idx != s2_idx {
                            to_change += 1;
                        }
                        if unassigned {
                            unassigned = false;
                            target = Some(s2_idx);
                        } else {
                            ambiguous = true;
                        }
                    }
                }
                let dest = target.unwrap_or(s_idx);
                better[dest].push(Hole {
                    path: ho.path,
                    first: ho.first,
                });
                // Keep the original assignment too, for when the better one
                // is rejected below.
                new_shape_holes[s_idx].push(ho);
            }
        }
        if to_change > 0 && !ambiguous {
            new_shape_holes = better;
        }
    }

    let mut shapes = Vec::with_capacity(new_shapes.len());
    for (i, s) in new_shapes.into_iter().enumerate() {
        let mut shape = s.expect("solid").shape;
        if let Some(holes) = new_shape_holes.get(i) {
            for h in holes {
                shape.holes.push(sub_paths[h.path].clone());
            }
        }
        shapes.push(shape);
    }
    shapes
}
