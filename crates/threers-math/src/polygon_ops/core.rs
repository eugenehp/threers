//! Points, paths and rectangles in 64-bit integer coordinates (with a `z`
//! carried along for the caller), and the path utilities the boolean and
//! offset operations share.
//!
//! Floating point follows a clang build for arm64, which fuses a product
//! into the sum or difference it is an operand of within one expression
//! (`-ffp-contract=on`): in `x ± y` the product on the left if there is one,
//! else the one on the right, never across a call. Those fusions are written
//! out with `mul_add`, each where clang makes it, so that results are the
//! same to the last unit.

/// `Point64`: with the `z` that `USINGZ` adds. Two points are the same when
/// their `x` and `y` are; `z` goes along.
#[derive(Clone, Copy, Debug, Default)]
pub struct Point64 {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

impl PartialEq for Point64 {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl Eq for Point64 {}

impl Point64 {
    pub const fn new(x: i64, y: i64) -> Self {
        Point64 { x, y, z: 0 }
    }

    pub const fn with_z(x: i64, y: i64, z: i64) -> Self {
        Point64 { x, y, z }
    }

    /// `Point64( double, double )`: rounded half away from zero.
    pub fn from_f64(x: f64, y: f64) -> Self {
        Point64 {
            x: x.round() as i64,
            y: y.round() as i64,
            z: 0,
        }
    }
}

pub type Path64 = Vec<Point64>;
pub type Paths64 = Vec<Path64>;

/// `FillRule`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillRule {
    EvenOdd,
    NonZero,
    Positive,
    Negative,
}

/// `Rect64`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect64 {
    pub left: i64,
    pub top: i64,
    pub right: i64,
    pub bottom: i64,
}

impl Default for Rect64 {
    /// `Rect( true )`: all zero.
    fn default() -> Self {
        Rect64 {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        }
    }
}

impl Rect64 {
    pub fn mid_point(&self) -> Point64 {
        Point64::new((self.left + self.right) / 2, (self.top + self.bottom) / 2)
    }

    pub fn contains_rect(&self, rec: &Rect64) -> bool {
        rec.left >= self.left
            && rec.right <= self.right
            && rec.top >= self.top
            && rec.bottom <= self.bottom
    }

    pub fn is_empty(&self) -> bool {
        self.bottom <= self.top || self.right <= self.left
    }
}

/// `GetBounds( Path64 )`
pub fn get_bounds(path: &[Point64]) -> Rect64 {
    let (mut xmin, mut ymin, mut xmax, mut ymax) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);

    for p in path {
        if p.x < xmin {
            xmin = p.x;
        }

        if p.x > xmax {
            xmax = p.x;
        }

        if p.y < ymin {
            ymin = p.y;
        }

        if p.y > ymax {
            ymax = p.y;
        }
    }

    Rect64 {
        left: xmin,
        top: ymin,
        right: xmax,
        bottom: ymax,
    }
}

/// `Sqr`
#[inline]
pub fn sqr(v: f64) -> f64 {
    v * v
}

/// `nearbyint`, in the default rounding mode: half to even.
#[inline]
pub fn nearbyint(v: f64) -> f64 {
    v.round_ties_even()
}

/// `CrossProductSign`: exact, with 128-bit products.
pub fn cross_product_sign(pt1: Point64, pt2: Point64, pt3: Point64) -> i32 {
    let a = pt2.x.wrapping_sub(pt1.x) as i128;
    let b = pt3.y.wrapping_sub(pt2.y) as i128;
    let c = pt2.y.wrapping_sub(pt1.y) as i128;
    let d = pt3.x.wrapping_sub(pt2.x) as i128;
    let ab = a * b;
    let cd = c * d;

    match ab.cmp(&cd) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
    }
}

/// `IsCollinear`
pub fn is_collinear(pt1: Point64, shared: Point64, pt2: Point64) -> bool {
    let a = shared.x.wrapping_sub(pt1.x) as i128;
    let b = pt2.y.wrapping_sub(shared.y) as i128;
    let c = shared.y.wrapping_sub(pt1.y) as i128;
    let d = pt2.x.wrapping_sub(shared.x) as i128;
    a * b == c * d
}

/// `CrossProduct( pt1, pt2, pt3 )`
pub fn cross_product(pt1: Point64, pt2: Point64, pt3: Point64) -> f64 {
    ((pt2.x - pt1.x) as f64).mul_add(
        (pt3.y - pt2.y) as f64,
        -((pt2.y - pt1.y) as f64 * (pt3.x - pt2.x) as f64),
    )
}

/// `DotProduct( pt1, pt2, pt3 )`
pub fn dot_product(pt1: Point64, pt2: Point64, pt3: Point64) -> f64 {
    ((pt2.x - pt1.x) as f64).mul_add(
        (pt3.x - pt2.x) as f64,
        (pt2.y - pt1.y) as f64 * (pt3.y - pt2.y) as f64,
    )
}

/// `DistanceSqr`
pub fn distance_sqr(pt1: Point64, pt2: Point64) -> f64 {
    sqr((pt1.x - pt2.x) as f64) + sqr((pt1.y - pt2.y) as f64)
}

/// `PerpendicDistFromLineSqrd`
pub fn perpendic_dist_from_line_sqrd(pt: Point64, line1: Point64, line2: Point64) -> f64 {
    let a = (pt.x - line1.x) as f64;
    let b = (pt.y - line1.y) as f64;
    let c = (line2.x - line1.x) as f64;
    let d = (line2.y - line1.y) as f64;

    if c == 0.0 && d == 0.0 {
        return 0.0;
    }

    sqr(a.mul_add(d, -(c * b))) / c.mul_add(c, d * d)
}

/// `Area( Path64 )`: positive for a counter-clockwise path (y up).
pub fn area(path: &[Point64]) -> f64 {
    let cnt = path.len();

    if cnt < 3 {
        return 0.0;
    }

    let mut a = 0.0;
    let mut it2 = cnt - 1;
    let mut stop = it2;

    if cnt & 1 == 0 {
        stop += 1;
    }

    let mut it1 = 0;

    while it1 != stop {
        a = ((path[it2].y + path[it1].y) as f64).mul_add((path[it2].x - path[it1].x) as f64, a);
        it2 = it1 + 1;
        a = ((path[it1].y + path[it2].y) as f64).mul_add((path[it1].x - path[it2].x) as f64, a);
        it1 += 2;
    }

    if cnt & 1 == 1 {
        a = ((path[it2].y + path[it1].y) as f64).mul_add((path[it2].x - path[it1].x) as f64, a);
    }

    a * 0.5
}

/// `IsPositive`
pub fn is_positive(path: &[Point64]) -> bool {
    area(path) >= 0.0
}

/// `GetLineIntersectPt` (not `CLIPPER2_HI_PRECISION`): the point is cut
/// toward zero, and has no `z`.
pub fn get_line_intersect_pt(
    ln1a: Point64,
    ln1b: Point64,
    ln2a: Point64,
    ln2b: Point64,
    ip: &mut Point64,
) -> bool {
    let dx1 = (ln1b.x - ln1a.x) as f64;
    let dy1 = (ln1b.y - ln1a.y) as f64;
    let dx2 = (ln2b.x - ln2a.x) as f64;
    let dy2 = (ln2b.y - ln2a.y) as f64;
    let det = dy1.mul_add(dx2, -(dy2 * dx1));

    if det == 0.0 {
        return false;
    }

    let t = ((ln1a.x - ln2a.x) as f64).mul_add(dy2, -((ln1a.y - ln2a.y) as f64 * dx2)) / det;

    if t <= 0.0 {
        *ip = ln1a;
    } else if t >= 1.0 {
        *ip = ln1b;
    } else {
        ip.x = t.mul_add(dx1, ln1a.x as f64) as i64;
        ip.y = t.mul_add(dy1, ln1a.y as f64) as i64;
        ip.z = 0;
    }

    true
}

/// `SegmentsIntersect`, not inclusive.
pub fn segments_intersect(seg1a: Point64, seg1b: Point64, seg2a: Point64, seg2b: Point64) -> bool {
    let dy1 = (seg1b.y - seg1a.y) as f64;
    let dx1 = (seg1b.x - seg1a.x) as f64;
    let dy2 = (seg2b.y - seg2a.y) as f64;
    let dx2 = (seg2b.x - seg2a.x) as f64;
    let cp = dy1.mul_add(dx2, -(dy2 * dx1));

    // parallel
    if cp == 0.0 {
        return false;
    }

    let mut t = ((seg1a.x - seg2a.x) as f64).mul_add(dy2, -((seg1a.y - seg2a.y) as f64 * dx2));

    if t == 0.0 {
        return false;
    }

    if t > 0.0 {
        if cp < 0.0 || t >= cp {
            return false;
        }
    } else if cp > 0.0 || t <= cp {
        return false;
    }

    t = ((seg1a.x - seg2a.x) as f64).mul_add(dy1, -((seg1a.y - seg2a.y) as f64 * dx1));

    if t == 0.0 {
        return false;
    }

    if t > 0.0 {
        cp > 0.0 && t < cp
    } else {
        cp < 0.0 && t > cp
    }
}

/// `GetClosestPointOnSegment`, for integer points.
pub fn get_closest_point_on_segment(off_pt: Point64, seg1: Point64, seg2: Point64) -> Point64 {
    if seg1.x == seg2.x && seg1.y == seg2.y {
        return seg1;
    }

    let dx = (seg2.x - seg1.x) as f64;
    let dy = (seg2.y - seg1.y) as f64;
    let mut q = ((off_pt.x - seg1.x) as f64).mul_add(dx, (off_pt.y - seg1.y) as f64 * dy)
        / (sqr(dx) + sqr(dy));

    q = q.clamp(0.0, 1.0);

    Point64::new(
        seg1.x + nearbyint(q * dx) as i64,
        seg1.y + nearbyint(q * dy) as i64,
    )
}

/// `PointInPolygonResult`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointInPolygonResult {
    IsOn,
    IsInside,
    IsOutside,
}

/// `PointInPolygon`
pub fn point_in_polygon(pt: Point64, polygon: &[Point64]) -> PointInPolygonResult {
    if polygon.len() < 3 {
        return PointInPolygonResult::IsOutside;
    }

    let mut val = 0;
    let cbegin = 0;
    let mut cend = polygon.len();
    let mut first = cbegin;

    while first != cend && polygon[first].y == pt.y {
        first += 1;
    }

    // not a proper polygon
    if first == cend {
        return PointInPolygonResult::IsOutside;
    }

    let mut is_above = polygon[first].y < pt.y;
    let starting_above = is_above;
    let mut curr = first + 1;
    let mut prev;

    loop {
        if curr == cend {
            if cend == first || first == cbegin {
                break;
            }

            cend = first;
            curr = cbegin;
        }

        if is_above {
            while curr != cend && polygon[curr].y < pt.y {
                curr += 1;
            }

            if curr == cend {
                continue;
            }
        } else {
            while curr != cend && polygon[curr].y > pt.y {
                curr += 1;
            }

            if curr == cend {
                continue;
            }
        }

        // not cend, which might be first
        prev = if curr == cbegin {
            polygon.len() - 1
        } else {
            curr - 1
        };

        if polygon[curr].y == pt.y {
            if polygon[curr].x == pt.x
                || (polygon[curr].y == polygon[prev].y
                    && ((pt.x < polygon[prev].x) != (pt.x < polygon[curr].x)))
            {
                return PointInPolygonResult::IsOn;
            }

            curr += 1;

            if curr == first {
                break;
            }

            continue;
        }

        if pt.x < polygon[curr].x && pt.x < polygon[prev].x {
            // the edge is to the right of the point
        } else if pt.x > polygon[prev].x && pt.x > polygon[curr].x {
            val = 1 - val;
        } else {
            let d = cross_product_sign(polygon[prev], polygon[curr], pt);

            if d == 0 {
                return PointInPolygonResult::IsOn;
            }

            if (d < 0) == is_above {
                val = 1 - val;
            }
        }

        is_above = !is_above;
        curr += 1;
    }

    if is_above != starting_above {
        cend = polygon.len();

        if curr == cend {
            curr = cbegin;
        }

        prev = if curr == cbegin { cend - 1 } else { curr - 1 };

        let d = cross_product_sign(polygon[prev], polygon[curr], pt);

        if d == 0 {
            return PointInPolygonResult::IsOn;
        }

        if (d < 0) == is_above {
            val = 1 - val;
        }
    }

    if val == 0 {
        PointInPolygonResult::IsOutside
    } else {
        PointInPolygonResult::IsInside
    }
}

/// `Path2ContainsPath1`: whether `path1` lies inside `path2`.
pub fn path2_contains_path1(path1: &[Point64], path2: &[Point64]) -> bool {
    let mut pip = PointInPolygonResult::IsOn;

    for pt in path1 {
        match point_in_polygon(*pt, path2) {
            PointInPolygonResult::IsOutside => {
                if pip == PointInPolygonResult::IsOutside {
                    return false;
                }

                pip = PointInPolygonResult::IsOutside;
            }
            PointInPolygonResult::IsInside => {
                if pip == PointInPolygonResult::IsInside {
                    return true;
                }

                pip = PointInPolygonResult::IsInside;
            }
            PointInPolygonResult::IsOn => {}
        }
    }

    if pip != PointInPolygonResult::IsInside {
        return false;
    }

    // likely, but check the middle
    let mp1 = get_bounds(path1).mid_point();
    point_in_polygon(mp1, path2) == PointInPolygonResult::IsInside
}

fn get_next(mut current: usize, high: usize, flags: &[bool]) -> usize {
    current += 1;

    while current <= high && flags[current] {
        current += 1;
    }

    if current <= high {
        return current;
    }

    current = 0;

    while flags[current] {
        current += 1;
    }

    current
}

fn get_prior(mut current: usize, high: usize, flags: &[bool]) -> usize {
    if current == 0 {
        current = high;
    } else {
        current -= 1;
    }

    while current > 0 && flags[current] {
        current -= 1;
    }

    if !flags[current] {
        return current;
    }

    current = high;

    while flags[current] {
        current -= 1;
    }

    current
}

/// `SimplifyPath`
pub fn simplify_path(path: &[Point64], epsilon: f64, is_closed_path: bool) -> Path64 {
    let len = path.len();

    if len < 4 {
        return path.to_vec();
    }

    let high = len - 1;
    let eps_sqr = sqr(epsilon);

    let mut flags = vec![false; len];
    let mut dist_sqr = vec![0.0; len];
    let mut curr = 0;

    if is_closed_path {
        dist_sqr[0] = perpendic_dist_from_line_sqrd(path[0], path[high], path[1]);
        dist_sqr[high] = perpendic_dist_from_line_sqrd(path[high], path[0], path[high - 1]);
    } else {
        dist_sqr[0] = f64::MAX;
        dist_sqr[high] = f64::MAX;
    }

    for i in 1..high {
        dist_sqr[i] = perpendic_dist_from_line_sqrd(path[i], path[i - 1], path[i + 1]);
    }

    loop {
        if dist_sqr[curr] > eps_sqr {
            let start = curr;

            loop {
                curr = get_next(curr, high, &flags);

                if !(curr != start && dist_sqr[curr] > eps_sqr) {
                    break;
                }
            }

            if curr == start {
                break;
            }
        }

        let mut prior = get_prior(curr, high, &flags);
        let mut next = get_next(curr, high, &flags);

        if next == prior {
            break;
        }

        // flag for removal the smaller of adjacent distances
        let prior2;

        if dist_sqr[next] < dist_sqr[curr] {
            prior2 = prior;
            prior = curr;
            curr = next;
            next = get_next(next, high, &flags);
        } else {
            prior2 = get_prior(prior, high, &flags);
        }

        flags[curr] = true;
        curr = next;
        next = get_next(next, high, &flags);

        if is_closed_path || (curr != high && curr != 0) {
            dist_sqr[curr] = perpendic_dist_from_line_sqrd(path[curr], path[prior], path[next]);
        }

        if is_closed_path || (prior != 0 && prior != high) {
            dist_sqr[prior] = perpendic_dist_from_line_sqrd(path[prior], path[prior2], path[curr]);
        }
    }

    path.iter()
        .zip(&flags)
        .filter(|(_, removed)| !**removed)
        .map(|(pt, _)| *pt)
        .collect()
}

/// `SimplifyPaths`
pub fn simplify_paths(paths: &[Path64], epsilon: f64, is_closed_path: bool) -> Paths64 {
    paths
        .iter()
        .map(|path| simplify_path(path, epsilon, is_closed_path))
        .collect()
}
