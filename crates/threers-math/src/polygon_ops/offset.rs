//! Growing and shrinking paths and polygons: round, square, bevel and miter
//! joins, and open paths ended butt, square, round or joined; and
//! [`ellipse`], an integer ellipse at a given precision.
//!
//! Products fused into sums are written with `mul_add` (see `core`).

use std::cell::RefCell;
use std::f64::consts::PI;
use std::rc::Rc;

use super::boolean::{Boolean64, ClipType, PolyPath64, ZCallback64};
use super::core::*;

const FLOATING_POINT_TOLERANCE: f64 = 1e-12;
// 1/500
const ARC_CONST: f64 = 0.002;

/// `JoinType`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinType {
    Square,
    Bevel,
    Round,
    Miter,
}

/// `EndType`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndType {
    Polygon,
    Joined,
    Butt,
    Square,
    Round,
}

/// `PointD`, with the `z` of `USINGZ`.
#[derive(Clone, Copy, Debug, Default)]
struct PointD {
    x: f64,
    y: f64,
    z: i64,
}

impl PointD {
    fn new(x: f64, y: f64) -> Self {
        PointD { x, y, z: 0 }
    }

    fn with_z(x: f64, y: f64, z: i64) -> Self {
        PointD { x, y, z }
    }

    /// `PointD( Point64 )`
    fn from64(p: Point64) -> Self {
        PointD {
            x: p.x as f64,
            y: p.y as f64,
            z: p.z,
        }
    }

    /// `Point64( PointD )`: rounded half away from zero.
    fn to64(self) -> Point64 {
        Point64::with_z(self.x.round() as i64, self.y.round() as i64, self.z)
    }
}

/// `Point64( double, double, z )`
fn point64(x: f64, y: f64, z: i64) -> Point64 {
    Point64::with_z(x.round() as i64, y.round() as i64, z)
}

/// `Hypot`: `sqrt( x * x + y * y )`, fused.
fn hypot(x: f64, y: f64) -> f64 {
    x.mul_add(x, y * y).sqrt()
}

/// `CrossProduct( vec1, vec2 )`
fn cross_product_d(vec1: PointD, vec2: PointD) -> f64 {
    vec1.y.mul_add(vec2.x, -(vec2.y * vec1.x))
}

/// `DotProduct( vec1, vec2 )`
fn dot_product_d(vec1: PointD, vec2: PointD) -> f64 {
    vec1.x.mul_add(vec2.x, vec1.y * vec2.y)
}

fn get_unit_normal(pt1: Point64, pt2: Point64) -> PointD {
    if pt1 == pt2 {
        return PointD::new(0.0, 0.0);
    }

    let mut dx = (pt2.x - pt1.x) as f64;
    let mut dy = (pt2.y - pt1.y) as f64;
    let inverse_hypot = 1.0 / hypot(dx, dy);
    dx *= inverse_hypot;
    dy *= inverse_hypot;
    PointD::new(dy, -dx)
}

fn almost_zero(value: f64) -> bool {
    value.abs() < 0.001
}

fn normalize_vector(vec: PointD) -> PointD {
    let h = hypot(vec.x, vec.y);

    if almost_zero(h) {
        return PointD::new(0.0, 0.0);
    }

    let inverse_hypot = 1.0 / h;
    PointD::new(vec.x * inverse_hypot, vec.y * inverse_hypot)
}

fn get_avg_unit_vector(vec1: PointD, vec2: PointD) -> PointD {
    normalize_vector(PointD::new(vec1.x + vec2.x, vec1.y + vec2.y))
}

fn get_perpendic(pt: Point64, norm: PointD, delta: f64) -> Point64 {
    point64(
        norm.x.mul_add(delta, pt.x as f64),
        norm.y.mul_add(delta, pt.y as f64),
        pt.z,
    )
}

fn get_perpendic_d(pt: Point64, norm: PointD, delta: f64) -> PointD {
    PointD::with_z(
        norm.x.mul_add(delta, pt.x as f64),
        norm.y.mul_add(delta, pt.y as f64),
        pt.z,
    )
}

/// `TranslatePoint`: the offsets come in as values, not fused.
fn translate_point(pt: PointD, dx: f64, dy: f64) -> PointD {
    PointD::with_z(pt.x + dx, pt.y + dy, pt.z)
}

fn reflect_point(pt: PointD, pivot: PointD) -> PointD {
    PointD::with_z(pivot.x + (pivot.x - pt.x), pivot.y + (pivot.y - pt.y), pt.z)
}

/// `GetLineIntersectPt` for `PointD`.
fn get_line_intersect_pt_d(
    ln1a: PointD,
    ln1b: PointD,
    ln2a: PointD,
    ln2b: PointD,
    ip: &mut PointD,
) -> bool {
    let dx1 = ln1b.x - ln1a.x;
    let dy1 = ln1b.y - ln1a.y;
    let dx2 = ln2b.x - ln2a.x;
    let dy2 = ln2b.y - ln2a.y;
    let det = dy1.mul_add(dx2, -(dy2 * dx1));

    if det == 0.0 {
        return false;
    }

    let t = (ln1a.x - ln2a.x).mul_add(dy2, -((ln1a.y - ln2a.y) * dx2)) / det;

    if t <= 0.0 {
        *ip = ln1a;
    } else if t >= 1.0 {
        *ip = ln1b;
    } else {
        ip.x = t.mul_add(dx1, ln1a.x);
        ip.y = t.mul_add(dy1, ln1a.y);
        ip.z = 0;
    }

    true
}

/// An ellipse of integer points.
pub fn ellipse(center: Point64, radius_x: f64, mut radius_y: f64, mut steps: usize) -> Path64 {
    if radius_x <= 0.0 {
        return Vec::new();
    }

    if radius_y <= 0.0 {
        radius_y = radius_x;
    }

    if steps <= 2 {
        steps = (PI * ((radius_x + radius_y) / 2.0).sqrt()) as usize;
    }

    let si = (2.0 * PI / steps as f64).sin();
    let co = (2.0 * PI / steps as f64).cos();
    let (mut dx, mut dy) = (co, si);
    let mut result = Vec::with_capacity(steps);
    result.push(point64(center.x as f64 + radius_x, center.y as f64, 0));

    for _ in 1..steps {
        result.push(point64(
            radius_x.mul_add(dx, center.x as f64),
            radius_y.mul_add(dy, center.y as f64),
            0,
        ));
        let x = dx.mul_add(co, -(dy * si));
        dy = dy.mul_add(co, dx * si);
        dx = x;
    }

    result
}

/// `PathOffset::Group`
struct Group {
    paths_in: Paths64,
    lowest_path_idx: Option<usize>,
    is_reversed: bool,
    join_type: JoinType,
    end_type: EndType,
}

/// `GetLowestClosedPathInfo`
fn get_lowest_closed_path_info(paths: &[Path64], idx: &mut Option<usize>, is_neg_area: &mut bool) {
    *idx = None;
    let mut bot_pt = Point64::new(i64::MAX, i64::MIN);

    for (i, path) in paths.iter().enumerate() {
        let mut a = f64::MAX;

        for pt in path {
            if pt.y < bot_pt.y || (pt.y == bot_pt.y && pt.x >= bot_pt.x) {
                continue;
            }

            if a == f64::MAX {
                a = area(path);

                // invalid closed path, so break from inner loop
                if a == 0.0 {
                    break;
                }

                *is_neg_area = a < 0.0;
            }

            *idx = Some(i);
            bot_pt.x = pt.x;
            bot_pt.y = pt.y;
        }
    }
}

/// `StripDuplicates`: of runs of points at one place the first stays.
fn strip_duplicates(path: &mut Path64, is_closed_path: bool) {
    path.dedup();

    if is_closed_path {
        while path.len() > 1 && path.last() == path.first() {
            path.pop();
        }
    }
}

impl Group {
    fn new(paths: &[Path64], join_type: JoinType, end_type: EndType) -> Self {
        let is_joined = end_type == EndType::Polygon || end_type == EndType::Joined;
        let mut paths_in = paths.to_vec();

        for p in &mut paths_in {
            strip_duplicates(p, is_joined);
        }

        let mut lowest_path_idx = None;
        let mut is_reversed = false;

        if end_type == EndType::Polygon {
            let mut is_neg_area = false;
            get_lowest_closed_path_info(&paths_in, &mut lowest_path_idx, &mut is_neg_area);
            is_reversed = lowest_path_idx.is_some() && is_neg_area;
        }

        Group {
            paths_in,
            lowest_path_idx,
            is_reversed,
            join_type,
            end_type,
        }
    }
}

/// `PathOffset`
pub struct PathOffset {
    error_code: i32,
    delta: f64,
    group_delta: f64,
    temp_lim: f64,
    steps_per_rad: f64,
    step_sin: f64,
    step_cos: f64,
    norms: Vec<PointD>,
    path_out: Path64,
    solution: Paths64,
    groups: Vec<Group>,
    join_type: JoinType,
    end_type: EndType,
    pub miter_limit: f64,
    pub arc_tolerance: f64,
    pub preserve_collinear: bool,
    pub reverse_solution: bool,
    z_callback: Option<Rc<RefCell<ZCallback64>>>,
}

impl Default for PathOffset {
    fn default() -> Self {
        Self::new(2.0, 0.0, false, false)
    }
}

impl PathOffset {
    pub fn new(
        miter_limit: f64,
        arc_tolerance: f64,
        preserve_collinear: bool,
        reverse_solution: bool,
    ) -> Self {
        PathOffset {
            error_code: 0,
            delta: 0.0,
            group_delta: 0.0,
            temp_lim: 0.0,
            steps_per_rad: 0.0,
            step_sin: 0.0,
            step_cos: 0.0,
            norms: Vec::new(),
            path_out: Vec::new(),
            solution: Vec::new(),
            groups: Vec::new(),
            join_type: JoinType::Bevel,
            end_type: EndType::Polygon,
            miter_limit,
            arc_tolerance,
            preserve_collinear,
            reverse_solution,
            z_callback: None,
        }
    }

    pub fn error_code(&self) -> i32 {
        self.error_code
    }

    /// `SetZCallback`
    pub fn set_z_callback(&mut self, cb: ZCallback64) {
        self.z_callback = Some(Rc::new(RefCell::new(cb)));
    }

    pub fn add_path(&mut self, path: &[Point64], join_type: JoinType, end_type: EndType) {
        self.groups
            .push(Group::new(&[path.to_vec()], join_type, end_type));
    }

    pub fn add_paths(&mut self, paths: &[Path64], join_type: JoinType, end_type: EndType) {
        if paths.is_empty() {
            return;
        }

        self.groups.push(Group::new(paths, join_type, end_type));
    }

    pub fn clear(&mut self) {
        self.groups.clear();
        self.norms.clear();
    }

    fn build_normals(&mut self, path: &[Point64]) {
        self.norms.clear();

        if path.is_empty() {
            return;
        }

        for i in 0..path.len() - 1 {
            self.norms.push(get_unit_normal(path[i], path[i + 1]));
        }

        self.norms
            .push(get_unit_normal(path[path.len() - 1], path[0]));
    }

    fn do_bevel(&mut self, path: &[Point64], j: usize, k: usize) {
        let (pt1, pt2);
        let p = path[j];

        if j == k {
            let abs_delta = self.group_delta.abs();
            let n = self.norms[j];
            pt1 = PointD::with_z(
                (-abs_delta).mul_add(n.x, p.x as f64),
                (-abs_delta).mul_add(n.y, p.y as f64),
                p.z,
            );
            pt2 = PointD::with_z(
                abs_delta.mul_add(n.x, p.x as f64),
                abs_delta.mul_add(n.y, p.y as f64),
                p.z,
            );
        } else {
            let gd = self.group_delta;
            let (nk, nj) = (self.norms[k], self.norms[j]);
            pt1 = PointD::with_z(
                gd.mul_add(nk.x, p.x as f64),
                gd.mul_add(nk.y, p.y as f64),
                p.z,
            );
            pt2 = PointD::with_z(
                gd.mul_add(nj.x, p.x as f64),
                gd.mul_add(nj.y, p.y as f64),
                p.z,
            );
        }

        self.path_out.push(pt1.to64());
        self.path_out.push(pt2.to64());
    }

    fn do_square(&mut self, path: &[Point64], j: usize, k: usize) {
        let (nj, nk) = (self.norms[j], self.norms[k]);
        let gd = self.group_delta;

        let vec = if j == k {
            PointD::new(nj.y, -nj.x)
        } else {
            get_avg_unit_vector(PointD::new(-nk.y, nk.x), PointD::new(nj.y, -nj.x))
        };

        let abs_delta = gd.abs();

        // now offset the original vertex delta units along unit vector
        let mut pt_q = PointD::from64(path[j]);
        pt_q = translate_point(pt_q, abs_delta * vec.x, abs_delta * vec.y);

        // get perpendicular vertices
        let pt1 = translate_point(pt_q, gd * vec.y, gd * -vec.x);
        let pt2 = translate_point(pt_q, gd * -vec.y, gd * vec.x);

        // get 2 vertices along one edge offset
        let pt3 = get_perpendic_d(path[k], nk, gd);

        if j == k {
            let pt4 = PointD::new(vec.x.mul_add(gd, pt3.x), vec.y.mul_add(gd, pt3.y));
            let mut pt = pt_q;
            get_line_intersect_pt_d(pt1, pt2, pt3, pt4, &mut pt);

            // get the second intersect point through reflecion
            self.path_out.push(reflect_point(pt, pt_q).to64());
            self.path_out.push(pt.to64());
        } else {
            let pt4 = get_perpendic_d(path[j], nk, gd);
            let mut pt = pt_q;
            get_line_intersect_pt_d(pt1, pt2, pt3, pt4, &mut pt);
            self.path_out.push(pt.to64());

            // get the second intersect point through reflecion
            self.path_out.push(reflect_point(pt, pt_q).to64());
        }
    }

    fn do_miter(&mut self, path: &[Point64], j: usize, k: usize, cos_a: f64) {
        let q = self.group_delta / (cos_a + 1.0);
        let (nk, nj) = (self.norms[k], self.norms[j]);
        let p = path[j];
        self.path_out.push(point64(
            (nk.x + nj.x).mul_add(q, p.x as f64),
            (nk.y + nj.y).mul_add(q, p.y as f64),
            p.z,
        ));
    }

    fn do_round(&mut self, path: &[Point64], j: usize, k: usize, angle: f64) {
        let pt = path[j];
        let nk = self.norms[k];
        let mut offset_vec = PointD::new(nk.x * self.group_delta, nk.y * self.group_delta);

        if j == k {
            offset_vec.x = -offset_vec.x;
            offset_vec.y = -offset_vec.y;
        }

        self.path_out.push(point64(
            pt.x as f64 + offset_vec.x,
            pt.y as f64 + offset_vec.y,
            pt.z,
        ));

        // #448, #456
        let steps = (self.steps_per_rad * angle.abs()).ceil() as i32;

        // ie 1 less than steps
        for _ in 1..steps {
            offset_vec = PointD::new(
                offset_vec
                    .x
                    .mul_add(self.step_cos, -(self.step_sin * offset_vec.y)),
                offset_vec
                    .x
                    .mul_add(self.step_sin, offset_vec.y * self.step_cos),
            );

            self.path_out.push(point64(
                pt.x as f64 + offset_vec.x,
                pt.y as f64 + offset_vec.y,
                pt.z,
            ));
        }

        self.path_out
            .push(get_perpendic(path[j], self.norms[j], self.group_delta));
    }

    fn offset_point(&mut self, path: &[Point64], j: usize, k: usize) {
        if path[j] == path[k] {
            return;
        }

        // Let A = change in angle where edges join
        // A == 0: ie no change in angle (flat join)
        // A == PI: edges 'spike'
        // sin(A) < 0: right turning
        // cos(A) < 0: change in angle is more than 90 degree
        let (nj, nk) = (self.norms[j], self.norms[k]);
        let mut sin_a = cross_product_d(nj, nk);
        let cos_a = dot_product_d(nj, nk);
        sin_a = sin_a.clamp(-1.0, 1.0);

        if self.group_delta.abs() <= FLOATING_POINT_TOLERANCE {
            self.path_out.push(path[j]);
            return;
        }

        // test for concavity first (#593)
        if cos_a > -0.999 && sin_a * self.group_delta < 0.0 {
            // is concave
            // by far the simplest way to construct concave joins, especially
            // those joining very short segments, is to insert 3 points that
            // produce negative regions. These regions will be removed later
            // by the finishing union operation. This is also the best way to
            // ensure that path reversals (ie over-shrunk paths) are removed.
            let p1 = get_perpendic(path[j], nk, self.group_delta);
            self.path_out.push(Point64::with_z(p1.x, p1.y, path[j].z));

            // (#405, #873, #916)
            self.path_out.push(path[j]);

            let p2 = get_perpendic(path[j], nj, self.group_delta);
            self.path_out.push(Point64::with_z(p2.x, p2.y, path[j].z));
        } else if cos_a > 0.999 && self.join_type != JoinType::Round {
            // almost straight - less than 2.5 degree (#424, #482, #526 & #724)
            self.do_miter(path, j, k, cos_a);
        } else if self.join_type == JoinType::Miter {
            // miter unless the angle is sufficiently acute to exceed ML
            if cos_a > self.temp_lim - 1.0 {
                self.do_miter(path, j, k, cos_a);
            } else {
                self.do_square(path, j, k);
            }
        } else if self.join_type == JoinType::Round {
            self.do_round(path, j, k, sin_a.atan2(cos_a));
        } else if self.join_type == JoinType::Bevel {
            self.do_bevel(path, j, k);
        } else {
            self.do_square(path, j, k);
        }
    }

    fn offset_polygon(&mut self, path: &[Point64]) {
        self.path_out.clear();
        let len = path.len();
        let mut k = len.wrapping_sub(1);

        for j in 0..len {
            self.offset_point(path, j, k);
            k = j;
        }

        self.solution.push(self.path_out.clone());
    }

    fn offset_open_joined(&mut self, path: &[Point64]) {
        self.offset_polygon(path);

        let mut reverse_path = path.to_vec();
        reverse_path.reverse();

        // rebuild normals
        self.norms.reverse();
        let first = self.norms[0];
        self.norms.push(first);
        self.norms.remove(0);

        for n in &mut self.norms {
            n.x = -n.x;
            n.y = -n.y;
        }

        self.offset_polygon(&reverse_path);
    }

    fn offset_open_path(&mut self, path: &[Point64]) {
        // do the line start cap
        if self.group_delta.abs() <= FLOATING_POINT_TOLERANCE {
            self.path_out.push(path[0]);
        } else {
            match self.end_type {
                EndType::Butt => self.do_bevel(path, 0, 0),
                EndType::Round => self.do_round(path, 0, 0, PI),
                _ => self.do_square(path, 0, 0),
            }
        }

        let high_i = path.len() - 1;

        // offset the left side going forward
        for j in 1..high_i {
            self.offset_point(path, j, j - 1);
        }

        // reverse normals
        for i in (1..=high_i).rev() {
            self.norms[i] = PointD::new(-self.norms[i - 1].x, -self.norms[i - 1].y);
        }

        self.norms[0] = self.norms[high_i];

        // do the line end cap
        if self.group_delta.abs() <= FLOATING_POINT_TOLERANCE {
            self.path_out.push(path[high_i]);
        } else {
            match self.end_type {
                EndType::Butt => self.do_bevel(path, high_i, high_i),
                EndType::Round => self.do_round(path, high_i, high_i, PI),
                _ => self.do_square(path, high_i, high_i),
            }
        }

        let mut k = high_i;
        let mut j = high_i.wrapping_sub(1);

        while j > 0 && j != usize::MAX {
            self.offset_point(path, j, k);
            k = j;
            j -= 1;
        }

        self.solution.push(self.path_out.clone());
    }

    fn do_group_offset(&mut self, g: usize) {
        let (end_type, join_type, is_reversed, has_lowest) = {
            let group = &self.groups[g];
            (
                group.end_type,
                group.join_type,
                group.is_reversed,
                group.lowest_path_idx.is_some(),
            )
        };

        if end_type == EndType::Polygon {
            // a straight path (2 points) can now also be 'polygon' offset
            // where the ends will be treated as (180 deg.) joins
            if !has_lowest {
                self.delta = self.delta.abs();
            }

            self.group_delta = if is_reversed { -self.delta } else { self.delta };
        } else {
            self.group_delta = self.delta.abs();
        }

        let abs_delta = self.group_delta.abs();
        self.join_type = join_type;
        self.end_type = end_type;

        if join_type == JoinType::Round || end_type == EndType::Round {
            // calculate the number of steps required to approximate a circle
            // (see the trigonometry notes of the algorithm's author, A. Johnson)
            // arcTol - when arc_tolerance_ is undefined (0) then curve
            // imprecision will be relative to the size of the offset
            // (delta). Obviously very large offsets will almost always
            // require much less precision.
            let arc_tol = if self.arc_tolerance > FLOATING_POINT_TOLERANCE {
                abs_delta.min(self.arc_tolerance)
            } else {
                abs_delta * ARC_CONST
            };

            let steps_per_360 = (PI / (1.0 - arc_tol / abs_delta).acos()).min(abs_delta * PI);
            self.step_sin = (2.0 * PI / steps_per_360).sin();
            self.step_cos = (2.0 * PI / steps_per_360).cos();

            if self.group_delta < 0.0 {
                self.step_sin = -self.step_sin;
            }

            self.steps_per_rad = steps_per_360 / (2.0 * PI);
        }

        let paths_in = std::mem::take(&mut self.groups[g].paths_in);

        for path in &paths_in {
            let path_len = path.len();
            self.path_out.clear();

            // single point
            if path_len == 1 {
                if self.group_delta < 1.0 {
                    continue;
                }

                let pt = path[0];

                // single vertex so build a circle or square ...
                if join_type == JoinType::Round {
                    let radius = abs_delta;

                    // #617
                    let steps = if self.steps_per_rad > 0.0 {
                        (self.steps_per_rad * 2.0 * PI).ceil() as usize
                    } else {
                        0
                    };

                    self.path_out = ellipse(pt, radius, radius, steps);
                } else {
                    let d = abs_delta.ceil() as i32;
                    let d = i64::from(d);

                    self.path_out = vec![
                        Point64::new(pt.x - d, pt.y - d),
                        Point64::new(pt.x + d, pt.y - d),
                        Point64::new(pt.x + d, pt.y + d),
                        Point64::new(pt.x - d, pt.y + d),
                    ];
                }

                for p in &mut self.path_out {
                    p.z = pt.z;
                }

                self.solution.push(self.path_out.clone());
                continue;
            }

            if path_len == 2 && end_type == EndType::Joined {
                self.end_type = if join_type == JoinType::Round {
                    EndType::Round
                } else {
                    EndType::Square
                };
            }

            self.build_normals(path);

            match self.end_type {
                EndType::Polygon => self.offset_polygon(path),
                EndType::Joined => self.offset_open_joined(path),
                _ => self.offset_open_path(path),
            }
        }

        self.groups[g].paths_in = paths_in;
    }

    fn check_reverse_orientation(&self) -> bool {
        self.groups
            .iter()
            .find(|g| g.end_type == EndType::Polygon)
            .is_some_and(|g| g.is_reversed)
    }

    /// The union of the offset paths, as paths or as a tree.
    fn execute_internal(&mut self, delta: f64) -> Option<(bool, Boolean64)> {
        self.error_code = 0;
        self.solution.clear();

        if self.groups.is_empty() {
            return None;
        }

        // ie: offset is insignificant
        if delta.abs() < 0.5 {
            for group in &self.groups {
                self.solution.extend(group.paths_in.iter().cloned());
            }
        } else {
            self.temp_lim = if self.miter_limit <= 1.0 {
                2.0
            } else {
                2.0 / (self.miter_limit * self.miter_limit)
            };
            self.delta = delta;

            for g in 0..self.groups.len() {
                self.do_group_offset(g);

                if self.error_code != 0 {
                    self.solution.clear();
                }
            }
        }

        if self.solution.is_empty() {
            return None;
        }

        let paths_reversed = self.check_reverse_orientation();
        let mut c = Boolean64::new();
        c.preserve_collinear = self.preserve_collinear;
        c.reverse_solution = self.reverse_solution != paths_reversed;

        // ZCB: the z of matching ends wins, else the caller's callback
        let user = self.z_callback.clone();

        c.set_z_callback(Box::new(move |bot1, top1, bot2, top2, ip| {
            if bot1.z != 0 && (bot1.z == bot2.z || bot1.z == top2.z) {
                ip.z = bot1.z;
            } else if bot2.z != 0 && bot2.z == top1.z {
                ip.z = bot2.z;
            } else if top1.z != 0 && top1.z == top2.z {
                ip.z = top1.z;
            } else if let Some(cb) = user.as_ref() {
                (cb.borrow_mut())(bot1, top1, bot2, top2, ip);
            }
        }));

        c.add_subject(&self.solution);
        Some((paths_reversed, c))
    }

    /// `Execute( delta, Paths64& )`
    pub fn execute_paths(&mut self, delta: f64) -> Paths64 {
        match self.execute_internal(delta) {
            None => std::mem::take(&mut self.solution),
            Some((paths_reversed, mut c)) => {
                let fill = if paths_reversed {
                    FillRule::Negative
                } else {
                    FillRule::Positive
                };
                let (_, closed, _) = c.execute_paths(ClipType::Union, fill);
                closed
            }
        }
    }

    /// `Execute( delta, PolyTree64& )`
    pub fn execute_tree(&mut self, delta: f64) -> PolyPath64 {
        match self.execute_internal(delta) {
            None => PolyPath64::default(),
            Some((paths_reversed, mut c)) => {
                let fill = if paths_reversed {
                    FillRule::Negative
                } else {
                    FillRule::Positive
                };
                let (_, tree, _) = c.execute_tree(ClipType::Union, fill);
                tree
            }
        }
    }
}
