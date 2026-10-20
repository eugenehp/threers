//! three.js r165's curve hierarchy — `Curve` and its arc-length machinery,
//! the 2D curves (`LineCurve`, `QuadraticBezierCurve`, `CubicBezierCurve`,
//! `SplineCurve`, `EllipseCurve` / `ArcCurve`), the 3D curves (`LineCurve3`,
//! `QuadraticBezierCurve3`, `CubicBezierCurve3`, `CatmullRomCurve3`) and
//! `CurvePath` — operation for operation in `f64`, as JavaScript computes it.
//!
//! A [`ThreeCurve`] is one curve object. Its definition ([`CurveDef`]) can be
//! replaced in place, and it carries the per-object state three.js keeps on
//! the instance: `arcLengthDivisions`, the `cacheArcLengths` table with its
//! `needsUpdate` flag, and a `CurvePath`'s `cacheLengths`. Caches are kept and
//! invalidated exactly as three.js does — editing a curve's points does *not*
//! clear them, `needsUpdate` / `updateArcLengths()` does.
//!
//! [`CurveDef::Custom`] is a curve whose `getPoint` lives elsewhere — a
//! JavaScript `Curve` subclass in the browser, or any closure natively — and
//! runs the base-class algorithms (`getLengths`, `getUtoTmapping`,
//! `getPointAt`, `getTangent`, `computeFrenetFrames`, …) over it, which is
//! what three.js does for a subclass that only overrides `getPoint`.
//!
//! Points are `[x, y, z]`; a 2D curve's have `z = 0`, which leaves every
//! distance and normalisation bit-identical to three.js's `Vector2` maths.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use super::shapes::{cubic_bezier, quadratic_bezier, ShapeCurve, Vec2d};
use crate::geometries::frenet::{compute_frenet_frames, FrenetFrames};

/// A point or vector in JavaScript number precision.
pub type Vec3d = [f64; 3];

/// A curve point evaluated outside Rust: `None` is three.js's `null`.
pub type PointFn = Rc<dyn Fn(f64) -> Option<Vec3d>>;

/// three.js `Curve.arcLengthDivisions`' default.
pub const ARC_LENGTH_DIVISIONS: usize = 200;

const NAN3: Vec3d = [f64::NAN; 3];

fn sub(a: Vec3d, b: Vec3d) -> Vec3d {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: Vec3d, b: Vec3d) -> Vec3d {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// `Vector3.distanceToSquared`.
fn distance_sq(a: Vec3d, b: Vec3d) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// `Vector3.normalize`: a multiply by `1 / (length || 1)`.
fn normalize(a: Vec3d) -> Vec3d {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let inv = 1.0 / if l == 0.0 || l.is_nan() { 1.0 } else { l };
    [a[0] * inv, a[1] * inv, a[2] * inv]
}

fn planar(p: Vec2d) -> Vec3d {
    [p.x, p.y, 0.0]
}

fn v2(p: Vec3d) -> Vec2d {
    Vec2d::new(p[0], p[1])
}

/// `CatmullRomCurve3.curveType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatmullRomType {
    /// `'centripetal'` (the default): knot spacing `|Δp|^0.5`.
    Centripetal,
    /// `'chordal'`: knot spacing `|Δp|`.
    Chordal,
    /// `'catmullrom'`: uniform, with `tension`.
    CatmullRom,
}

impl CatmullRomType {
    /// From three.js's string. Anything else three.js would evaluate with
    /// whatever coefficients a previous curve left in its shared polynomial
    /// objects; that is not reproducible, and is treated as centripetal.
    pub fn parse(s: &str) -> Self {
        match s {
            "chordal" => CatmullRomType::Chordal,
            "catmullrom" => CatmullRomType::CatmullRom,
            _ => CatmullRomType::Centripetal,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            CatmullRomType::Centripetal => "centripetal",
            CatmullRomType::Chordal => "chordal",
            CatmullRomType::CatmullRom => "catmullrom",
        }
    }
}

/// What three.js's flags make of a [`CurveDef::Custom`] curve — a subclass of
/// a built-in keeps the built-in's `isLineCurve` / `isEllipseCurve` /
/// `isSplineCurve`, which change `getPointAt` and how a `CurvePath` samples
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CustomRole {
    /// A plain `Curve` subclass.
    #[default]
    Plain,
    /// `isLineCurve` / `isLineCurve3`: `getPointAt(u) = getPoint(u)`,
    /// `getTangentAt(u) = getTangent(u)`, one division in a path.
    Line,
    /// `isEllipseCurve`: twice the divisions in a path.
    Ellipse,
    /// `isSplineCurve` with this many points: `divisions × points` in a path.
    Spline(usize),
}

/// A curve given by callbacks — three.js's `Curve` base class over a
/// `getPoint` (and optionally `getTangent`) defined elsewhere.
#[derive(Clone)]
pub struct CustomCurve {
    pub get_point: PointFn,
    /// An overridden `getTangent`; `None` uses the base class's finite
    /// difference.
    pub get_tangent: Option<PointFn>,
    /// 2 when `getPoint` returns `Vector2`s, else 3.
    pub dims: u8,
    pub role: CustomRole,
}

impl std::fmt::Debug for CustomCurve {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomCurve")
            .field("dims", &self.dims)
            .field("role", &self.role)
            .field("get_tangent", &self.get_tangent.is_some())
            .finish()
    }
}

/// Which three.js curve a [`ThreeCurve`] is, and its parameters.
#[derive(Debug, Clone)]
pub enum CurveDef {
    /// `LineCurve`, `QuadraticBezierCurve`, `CubicBezierCurve`, `SplineCurve`,
    /// `EllipseCurve` and `ArcCurve` — the 2D curves a `Path` is made of.
    Planar(ShapeCurve),
    /// `LineCurve3`.
    Line3 { v1: Vec3d, v2: Vec3d },
    /// `QuadraticBezierCurve3`.
    QuadraticBezier3 { v0: Vec3d, v1: Vec3d, v2: Vec3d },
    /// `CubicBezierCurve3`.
    CubicBezier3 { v0: Vec3d, v1: Vec3d, v2: Vec3d, v3: Vec3d },
    /// `CatmullRomCurve3`.
    CatmullRom3 {
        points: Vec<Vec3d>,
        closed: bool,
        curve_type: CatmullRomType,
        tension: f64,
    },
    /// `CurvePath`: its curves, in order, and `autoClose`.
    Path { curves: Vec<Rc<ThreeCurve>>, auto_close: bool },
    /// A `Curve` subclass whose `getPoint` is a callback.
    Custom(CustomCurve),
}

impl CurveDef {
    /// three.js's `type`.
    pub fn type_name(&self) -> &'static str {
        match self {
            CurveDef::Planar(c) => match c {
                ShapeCurve::Line { .. } => "LineCurve",
                ShapeCurve::QuadraticBezier { .. } => "QuadraticBezierCurve",
                ShapeCurve::CubicBezier { .. } => "CubicBezierCurve",
                ShapeCurve::Spline { .. } => "SplineCurve",
                ShapeCurve::Ellipse { .. } => "EllipseCurve",
            },
            CurveDef::Line3 { .. } => "LineCurve3",
            CurveDef::QuadraticBezier3 { .. } => "QuadraticBezierCurve3",
            CurveDef::CubicBezier3 { .. } => "CubicBezierCurve3",
            CurveDef::CatmullRom3 { .. } => "CatmullRomCurve3",
            CurveDef::Path { .. } => "CurvePath",
            CurveDef::Custom(_) => "Curve",
        }
    }

    fn is_line(&self) -> bool {
        matches!(
            self,
            CurveDef::Planar(ShapeCurve::Line { .. })
                | CurveDef::Line3 { .. }
                | CurveDef::Custom(CustomCurve {
                    role: CustomRole::Line,
                    ..
                })
        )
    }

    /// The resolution `CurvePath.getPoints(divisions)` samples this at.
    fn path_resolution(&self, divisions: usize) -> usize {
        match self {
            CurveDef::Planar(ShapeCurve::Ellipse { .. }) => divisions * 2,
            CurveDef::Planar(ShapeCurve::Line { .. }) | CurveDef::Line3 { .. } => 1,
            CurveDef::Planar(ShapeCurve::Spline { points }) => divisions * points.len(),
            CurveDef::Custom(c) => match c.role {
                CustomRole::Ellipse => divisions * 2,
                CustomRole::Line => 1,
                CustomRole::Spline(n) => divisions * n,
                CustomRole::Plain => divisions,
            },
            _ => divisions,
        }
    }
}

/// One three.js curve object: a [`CurveDef`] plus the state three.js keeps on
/// the instance. Shared (`Rc`) so a `CurvePath` can hold curves that are also
/// used on their own, as JavaScript objects are.
#[derive(Debug)]
pub struct ThreeCurve {
    def: RefCell<CurveDef>,
    arc_length_divisions: Cell<usize>,
    cache_arc_lengths: RefCell<Option<Rc<[f64]>>>,
    needs_update: Cell<bool>,
    /// `CurvePath.cacheLengths`.
    cache_lengths: RefCell<Option<Rc<[f64]>>>,
}

/// three.js `CatmullRomCurve3.getPoint(t)`.
fn catmull_rom3(points: &[Vec3d], closed: bool, ty: CatmullRomType, tension: f64, t: f64) -> Vec3d {
    let l = points.len();
    if l == 0 {
        return NAN3;
    }
    let li = l as i64;
    let p = (l as f64 - if closed { 0.0 } else { 1.0 }) * t;
    let int_point_f = p.floor();
    if !int_point_f.is_finite() {
        return NAN3;
    }
    let mut weight = p - int_point_f;
    let mut ip = int_point_f as i64;
    if closed {
        ip += if ip > 0 { 0 } else { (ip.abs() / li + 1) * li };
    } else if weight == 0.0 && ip == li - 1 {
        ip = li - 2;
        weight = 1.0;
    }
    // `points[i % l]`, undefined (here NaN) for a negative index as in JS.
    let at = |i: i64| -> Option<Vec3d> {
        let r = i % li;
        if r < 0 {
            None
        } else {
            points.get(r as usize).copied()
        }
    };
    // three.js extrapolates both ends into one shared temporary, so when both
    // are extrapolated (two points, open), `p0` ends up equal to `p3`.
    let extrapolate_first = !(closed || ip > 0);
    let extrapolate_last = !(closed || ip + 2 < li);
    if (extrapolate_first || extrapolate_last) && l < 2 {
        return NAN3;
    }
    let last_ext = || add(sub(points[l - 1], points[l - 2]), points[l - 1]);
    let p0 = if extrapolate_first {
        if extrapolate_last {
            last_ext()
        } else {
            add(sub(points[0], points[1]), points[0])
        }
    } else {
        match at(ip - 1) {
            Some(p) => p,
            None => return NAN3,
        }
    };
    let (p1, p2) = match (at(ip), at(ip + 1)) {
        (Some(a), Some(b)) => (a, b),
        _ => return NAN3,
    };
    let p3 = if extrapolate_last {
        last_ext()
    } else {
        match at(ip + 2) {
            Some(p) => p,
            None => return NAN3,
        }
    };

    // CubicPoly: init(x0, x1, t0, t1), calc(t).
    let calc = |x0: f64, x1: f64, t0: f64, t1: f64, t: f64| {
        let c0 = x0;
        let c1 = t0;
        let c2 = -3.0 * x0 + 3.0 * x1 - 2.0 * t0 - t1;
        let c3 = 2.0 * x0 - 2.0 * x1 + t0 + t1;
        let t2 = t * t;
        let t3 = t2 * t;
        c0 + c1 * t + c2 * t2 + c3 * t3
    };
    let mut out = [0.0; 3];
    match ty {
        CatmullRomType::Centripetal | CatmullRomType::Chordal => {
            let pow = if ty == CatmullRomType::Chordal { 0.5 } else { 0.25 };
            let mut dt0 = distance_sq(p0, p1).powf(pow);
            let mut dt1 = distance_sq(p1, p2).powf(pow);
            let mut dt2 = distance_sq(p2, p3).powf(pow);
            if dt1 < 1e-4 {
                dt1 = 1.0;
            }
            if dt0 < 1e-4 {
                dt0 = dt1;
            }
            if dt2 < 1e-4 {
                dt2 = dt1;
            }
            for k in 0..3 {
                let (x0, x1, x2, x3) = (p0[k], p1[k], p2[k], p3[k]);
                let mut t1 = (x1 - x0) / dt0 - (x2 - x0) / (dt0 + dt1) + (x2 - x1) / dt1;
                let mut t2 = (x2 - x1) / dt1 - (x3 - x1) / (dt1 + dt2) + (x3 - x2) / dt2;
                t1 *= dt1;
                t2 *= dt1;
                out[k] = calc(x1, x2, t1, t2, weight);
            }
        }
        CatmullRomType::CatmullRom => {
            for k in 0..3 {
                let (x0, x1, x2, x3) = (p0[k], p1[k], p2[k], p3[k]);
                out[k] = calc(x1, x2, tension * (x2 - x0), tension * (x3 - x1), weight);
            }
        }
    }
    out
}

impl ThreeCurve {
    /// A curve object with three.js's defaults for the per-instance state.
    pub fn new(def: CurveDef) -> Rc<ThreeCurve> {
        Rc::new(ThreeCurve {
            def: RefCell::new(def),
            arc_length_divisions: Cell::new(ARC_LENGTH_DIVISIONS),
            cache_arc_lengths: RefCell::new(None),
            needs_update: Cell::new(false),
            cache_lengths: RefCell::new(None),
        })
    }

    /// `new LineCurve3(v1, v2)`.
    pub fn line3(v1: Vec3d, v2: Vec3d) -> Rc<ThreeCurve> {
        Self::new(CurveDef::Line3 { v1, v2 })
    }

    /// `new QuadraticBezierCurve3(v0, v1, v2)`.
    pub fn quadratic_bezier3(v0: Vec3d, v1: Vec3d, v2: Vec3d) -> Rc<ThreeCurve> {
        Self::new(CurveDef::QuadraticBezier3 { v0, v1, v2 })
    }

    /// `new CubicBezierCurve3(v0, v1, v2, v3)`.
    pub fn cubic_bezier3(v0: Vec3d, v1: Vec3d, v2: Vec3d, v3: Vec3d) -> Rc<ThreeCurve> {
        Self::new(CurveDef::CubicBezier3 { v0, v1, v2, v3 })
    }

    /// `new CatmullRomCurve3(points, closed, curveType, tension)`.
    pub fn catmull_rom3(
        points: Vec<Vec3d>,
        closed: bool,
        curve_type: CatmullRomType,
        tension: f64,
    ) -> Rc<ThreeCurve> {
        Self::new(CurveDef::CatmullRom3 {
            points,
            closed,
            curve_type,
            tension,
        })
    }

    /// One of the 2D curves.
    pub fn planar(c: ShapeCurve) -> Rc<ThreeCurve> {
        Self::new(CurveDef::Planar(c))
    }

    /// `new CurvePath()` holding `curves`.
    pub fn path(curves: Vec<Rc<ThreeCurve>>, auto_close: bool) -> Rc<ThreeCurve> {
        Self::new(CurveDef::Path { curves, auto_close })
    }

    /// A `Curve` subclass whose `getPoint(t)` is `get_point`.
    pub fn custom(get_point: impl Fn(f64) -> Option<Vec3d> + 'static) -> Rc<ThreeCurve> {
        Self::new(CurveDef::Custom(CustomCurve {
            get_point: Rc::new(get_point),
            get_tangent: None,
            dims: 3,
            role: CustomRole::Plain,
        }))
    }

    /// The definition (a copy).
    pub fn def(&self) -> CurveDef {
        self.def.borrow().clone()
    }

    /// Replace the definition in place. Caches stay, as in three.js where
    /// editing `points` leaves `cacheArcLengths` until `needsUpdate`.
    pub fn set_def(&self, def: CurveDef) {
        *self.def.borrow_mut() = def;
    }

    /// Edit the definition in place.
    pub fn with_def_mut<R>(&self, f: impl FnOnce(&mut CurveDef) -> R) -> R {
        f(&mut self.def.borrow_mut())
    }

    pub fn arc_length_divisions(&self) -> usize {
        self.arc_length_divisions.get()
    }

    pub fn set_arc_length_divisions(&self, n: usize) {
        self.arc_length_divisions.set(n);
    }

    pub fn needs_update(&self) -> bool {
        self.needs_update.get()
    }

    pub fn set_needs_update(&self, v: bool) {
        self.needs_update.set(v);
    }

    /// `cacheArcLengths`, if computed.
    pub fn cache_arc_lengths(&self) -> Option<Rc<[f64]>> {
        self.cache_arc_lengths.borrow().clone()
    }

    /// Drop every cache here and in a path's curves (after a failed
    /// evaluation, so that NaNs from it are not served later).
    pub fn clear_caches(&self) {
        *self.cache_arc_lengths.borrow_mut() = None;
        *self.cache_lengths.borrow_mut() = None;
        if let Some(children) = self.children() {
            for c in children {
                c.clear_caches();
            }
        }
    }

    /// 2 for the planar curves (and a path or custom curve of them), else 3.
    pub fn dims(&self) -> u8 {
        match &*self.def.borrow() {
            CurveDef::Planar(_) => 2,
            CurveDef::Custom(c) => c.dims,
            CurveDef::Path { curves, .. } => curves.first().map_or(3, |c| c.dims()),
            _ => 3,
        }
    }

    /// three.js's `type`.
    pub fn type_name(&self) -> &'static str {
        self.def.borrow().type_name()
    }

    fn children(&self) -> Option<Vec<Rc<ThreeCurve>>> {
        match &*self.def.borrow() {
            CurveDef::Path { curves, .. } => Some(curves.clone()),
            _ => None,
        }
    }

    fn custom_fns(&self) -> Option<(PointFn, Option<PointFn>)> {
        match &*self.def.borrow() {
            CurveDef::Custom(c) => Some((c.get_point.clone(), c.get_tangent.clone())),
            _ => None,
        }
    }

    fn is_line(&self) -> bool {
        self.def.borrow().is_line()
    }

    fn is_path(&self) -> bool {
        matches!(&*self.def.borrow(), CurveDef::Path { .. })
    }

    /// `getPoint(t)`; `None` where three.js returns `null` (a `CurvePath`
    /// past its end, a custom curve that returned nothing).
    pub fn get_point(&self, t: f64) -> Option<Vec3d> {
        // Callbacks and child curves run with no borrow held, so a callback
        // may use (and redefine) this curve re-entrantly.
        if let Some((get_point, _)) = self.custom_fns() {
            return get_point(t);
        }
        if self.is_path() {
            return self.path_get_point(t);
        }
        let def = self.def.borrow();
        Some(match &*def {
            CurveDef::Planar(c) => planar(c.get_point(t)),
            CurveDef::Line3 { v1, v2 } => {
                if t == 1.0 {
                    *v2
                } else {
                    let d = sub(*v2, *v1);
                    [d[0] * t + v1[0], d[1] * t + v1[1], d[2] * t + v1[2]]
                }
            }
            CurveDef::QuadraticBezier3 { v0, v1, v2 } => [
                quadratic_bezier(t, v0[0], v1[0], v2[0]),
                quadratic_bezier(t, v0[1], v1[1], v2[1]),
                quadratic_bezier(t, v0[2], v1[2], v2[2]),
            ],
            CurveDef::CubicBezier3 { v0, v1, v2, v3 } => [
                cubic_bezier(t, v0[0], v1[0], v2[0], v3[0]),
                cubic_bezier(t, v0[1], v1[1], v2[1], v3[1]),
                cubic_bezier(t, v0[2], v1[2], v2[2], v3[2]),
            ],
            CurveDef::CatmullRom3 {
                points,
                closed,
                curve_type,
                tension,
            } => catmull_rom3(points, *closed, *curve_type, *tension, t),
            CurveDef::Path { .. } | CurveDef::Custom(_) => unreachable!(),
        })
    }

    fn point_or_nan(&self, t: f64) -> Vec3d {
        self.get_point(t).unwrap_or(NAN3)
    }

    /// `getPointAt(u)`: by arc length (a line's `t` already is).
    pub fn get_point_at(&self, u: f64) -> Option<Vec3d> {
        if self.is_line() {
            self.get_point(u)
        } else {
            self.get_point(self.get_u_to_t_mapping(u, None))
        }
    }

    /// `getPoints(divisions)` (three.js's default is 5, a path's 12).
    pub fn get_points(&self, divisions: usize) -> Vec<Option<Vec3d>> {
        if let Some(curves) = self.children() {
            return self.path_get_points(&curves, divisions);
        }
        (0..=divisions)
            .map(|d| self.get_point(d as f64 / divisions as f64))
            .collect()
    }

    /// `getSpacedPoints(divisions)` (three.js's default is 5, a path's 40).
    pub fn get_spaced_points(&self, divisions: usize) -> Vec<Option<Vec3d>> {
        if self.is_path() {
            let mut points: Vec<Option<Vec3d>> = (0..=divisions)
                .map(|i| self.get_point(i as f64 / divisions as f64))
                .collect();
            if self.auto_close() {
                points.push(points[0]);
            }
            return points;
        }
        (0..=divisions)
            .map(|d| self.get_point_at(d as f64 / divisions as f64))
            .collect()
    }

    fn auto_close(&self) -> bool {
        matches!(&*self.def.borrow(), CurveDef::Path { auto_close: true, .. })
    }

    /// `getLength()`.
    pub fn get_length(&self) -> f64 {
        if self.is_path() {
            return self.get_curve_lengths().last().copied().unwrap_or(f64::NAN);
        }
        let lengths = self.get_lengths(self.arc_length_divisions.get());
        lengths.last().copied().unwrap_or(f64::NAN)
    }

    /// `getLengths(divisions)`: cumulative chord lengths at `divisions + 1`
    /// even `t`, cached until `needsUpdate` or a different `divisions`.
    pub fn get_lengths(&self, divisions: usize) -> Rc<[f64]> {
        if let Some(c) = self.cache_arc_lengths.borrow().as_ref() {
            if c.len() == divisions + 1 && !self.needs_update.get() {
                return c.clone();
            }
        }
        self.needs_update.set(false);
        let mut cache = Vec::with_capacity(divisions + 1);
        let mut last = self.point_or_nan(0.0);
        let mut sum = 0.0;
        cache.push(0.0);
        for p in 1..=divisions {
            let current = self.point_or_nan(p as f64 / divisions as f64);
            sum += distance_sq(current, last).sqrt();
            cache.push(sum);
            last = current;
        }
        let cache: Rc<[f64]> = cache.into();
        *self.cache_arc_lengths.borrow_mut() = Some(cache.clone());
        cache
    }

    /// `updateArcLengths()`.
    pub fn update_arc_lengths(&self) {
        self.needs_update.set(true);
        if self.is_path() {
            *self.cache_lengths.borrow_mut() = None;
            self.get_curve_lengths();
        } else {
            self.get_lengths(self.arc_length_divisions.get());
        }
    }

    /// `getUtoTmapping(u, distance)`: the `t` at arc-length fraction `u` (or
    /// at arc length `distance`, when truthy). Outside the curve — `u < 0` or
    /// `u > 1` — three.js reads past its table and returns NaN; so does this.
    pub fn get_u_to_t_mapping(&self, u: f64, distance: Option<f64>) -> f64 {
        let arc_lengths = self.get_lengths(self.arc_length_divisions.get());
        let il = arc_lengths.len();
        if il == 0 {
            return f64::NAN;
        }
        let target = match distance {
            Some(d) if d != 0.0 && !d.is_nan() => d,
            _ => u * arc_lengths[il - 1],
        };
        let (mut low, mut high) = (0_i64, il as i64 - 1);
        while low <= high {
            let i = low + (high - low) / 2;
            let comparison = arc_lengths[i as usize] - target;
            if comparison < 0.0 {
                low = i + 1;
            } else if comparison > 0.0 {
                high = i - 1;
            } else {
                high = i;
                break;
            }
        }
        let i = high;
        if i < 0 {
            return f64::NAN;
        }
        let i = i as usize;
        if arc_lengths[i] == target {
            return i as f64 / (il - 1) as f64;
        }
        let before = arc_lengths[i];
        let Some(&after) = arc_lengths.get(i + 1) else {
            return f64::NAN;
        };
        let segment_fraction = (target - before) / (after - before);
        (i as f64 + segment_fraction) / (il - 1) as f64
    }

    /// `getTangent(t)`: a line's direction, an overridden `getTangent`, or
    /// the base class's difference of points `0.0001` either side.
    pub fn get_tangent(&self, t: f64) -> Vec3d {
        if let Some((_, Some(get_tangent))) = self.custom_fns() {
            return get_tangent(t).unwrap_or(NAN3);
        }
        self.get_tangent_base(t)
    }

    /// `getTangent(t)` ignoring an overridden one: `Curve.prototype.getTangent`
    /// (or a line's), which an override may call as `super.getTangent`.
    pub fn get_tangent_base(&self, t: f64) -> Vec3d {
        {
            let def = self.def.borrow();
            match &*def {
                CurveDef::Planar(ShapeCurve::Line { v1, v2 }) => {
                    return normalize(sub(planar(*v2), planar(*v1)));
                }
                CurveDef::Line3 { v1, v2 } => return normalize(sub(*v2, *v1)),
                _ => {}
            }
        }
        let delta = 0.0001;
        let mut t1 = t - delta;
        let mut t2 = t + delta;
        if t1 < 0.0 {
            t1 = 0.0;
        }
        if t2 > 1.0 {
            t2 = 1.0;
        }
        let pt1 = self.point_or_nan(t1);
        let pt2 = self.point_or_nan(t2);
        normalize(sub(pt2, pt1))
    }

    /// `getTangentAt(u)`.
    pub fn get_tangent_at(&self, u: f64) -> Vec3d {
        if self.is_line() {
            self.get_tangent(u)
        } else {
            self.get_tangent(self.get_u_to_t_mapping(u, None))
        }
    }

    /// `computeFrenetFrames(segments, closed)`.
    pub fn compute_frenet_frames(&self, segments: usize, closed: bool) -> FrenetFrames {
        let tangents: Vec<Vec3d> = (0..=segments)
            .map(|i| self.get_tangent_at(i as f64 / segments as f64))
            .collect();
        compute_frenet_frames(&tangents, closed)
    }

    /// `CurvePath.getCurveLengths()` (empty for a lone curve).
    pub fn get_curve_lengths(&self) -> Rc<[f64]> {
        let Some(curves) = self.children() else {
            return Rc::from(Vec::new());
        };
        if let Some(c) = self.cache_lengths.borrow().as_ref() {
            if c.len() == curves.len() {
                return c.clone();
            }
        }
        let mut sums = 0.0;
        let lengths: Rc<[f64]> = curves
            .iter()
            .map(|c| {
                sums += c.get_length();
                sums
            })
            .collect::<Vec<_>>()
            .into();
        *self.cache_lengths.borrow_mut() = Some(lengths.clone());
        lengths
    }

    fn path_get_point(&self, t: f64) -> Option<Vec3d> {
        let curves = self.children()?;
        let d = t * self.get_length();
        let curve_lengths = self.get_curve_lengths();
        for (i, &len) in curve_lengths.iter().enumerate() {
            if len >= d {
                let diff = len - d;
                let curve = curves.get(i)?;
                let segment_length = curve.get_length();
                let u = if segment_length == 0.0 {
                    0.0
                } else {
                    1.0 - diff / segment_length
                };
                return curve.get_point_at(u);
            }
        }
        None
    }

    fn path_get_points(&self, curves: &[Rc<ThreeCurve>], divisions: usize) -> Vec<Option<Vec3d>> {
        let mut points: Vec<Option<Vec3d>> = Vec::new();
        let mut last: Option<Vec3d> = None;
        for curve in curves {
            let resolution = curve.def.borrow().path_resolution(divisions);
            for point in curve.get_points(resolution) {
                if let (Some(l), Some(p)) = (last, point) {
                    if l == p {
                        continue;
                    }
                }
                points.push(point);
                last = point;
            }
        }
        if self.auto_close() && points.len() > 1 && points[points.len() - 1] != points[0] {
            points.push(points[0]);
        }
        points
    }

    /// A deep copy: the definition (a path's curves copied too) and the
    /// divisions, without caches — three.js's `clone()`.
    pub fn duplicate(&self) -> Rc<ThreeCurve> {
        let def = match self.def() {
            CurveDef::Path { curves, auto_close } => CurveDef::Path {
                curves: curves.iter().map(|c| c.duplicate()).collect(),
                auto_close,
            },
            d => d,
        };
        let c = ThreeCurve::new(def);
        c.set_arc_length_divisions(self.arc_length_divisions());
        c
    }

    /// `CurvePath.closePath()`: a line from the end back to the start, when
    /// they differ. No-op on anything but a non-empty path.
    pub fn close_path(&self) {
        let Some(curves) = self.children() else { return };
        let (Some(first), Some(last)) = (curves.first(), curves.last()) else {
            return;
        };
        let start = first.point_or_nan(0.0);
        let end = last.point_or_nan(1.0);
        if start != end {
            let line = if first.dims() == 2 {
                ThreeCurve::planar(ShapeCurve::Line {
                    v1: v2(end),
                    v2: v2(start),
                })
            } else {
                ThreeCurve::line3(end, start)
            };
            self.with_def_mut(|d| {
                if let CurveDef::Path { curves, .. } = d {
                    curves.push(line);
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3d, b: Vec3d) -> bool {
        (0..3).all(|k| (a[k] - b[k]).abs() < 1e-12)
    }

    #[test]
    fn line3_is_linear_in_arc_length() {
        let c = ThreeCurve::line3([0.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        assert_eq!(c.get_point(1.0), Some([2.0, 0.0, 0.0]));
        assert_eq!(c.get_point_at(0.25), Some([0.5, 0.0, 0.0]));
        assert_eq!(c.get_tangent(0.3), [1.0, 0.0, 0.0]);
        assert!((c.get_length() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn catmull_rom_passes_through_points() {
        for ty in [
            CatmullRomType::Centripetal,
            CatmullRomType::Chordal,
            CatmullRomType::CatmullRom,
        ] {
            let pts = vec![
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 0.5],
                [3.0, 1.0, -1.0],
                [4.0, 3.0, 2.0],
            ];
            let c = ThreeCurve::catmull_rom3(pts.clone(), false, ty, 0.5);
            for (i, p) in pts.iter().enumerate() {
                assert!(close(c.get_point(i as f64 / 3.0).unwrap(), *p), "{ty:?} {i}");
            }
        }
    }

    #[test]
    fn two_point_catmull_rom_shares_the_temporary() {
        // Both ends extrapolated: three.js's p0 is overwritten by p3.
        let c = ThreeCurve::catmull_rom3(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            false,
            CatmullRomType::Centripetal,
            0.5,
        );
        let p = c.get_point(0.5).unwrap();
        assert!(p[0].is_finite());
    }

    #[test]
    fn custom_curve_uses_base_machinery() {
        let c = ThreeCurve::custom(|t| Some([t * 3.0, 0.0, 0.0]));
        assert!((c.get_length() - 3.0).abs() < 1e-12);
        assert!(close(c.get_point_at(0.5).unwrap(), [1.5, 0.0, 0.0]));
        assert!(close(c.get_tangent_at(0.5), [1.0, 0.0, 0.0]));
        let f = c.compute_frenet_frames(4, false);
        assert_eq!(f.normals.len(), 5);
    }

    #[test]
    fn caches_survive_edits_until_needs_update() {
        let c = ThreeCurve::line3([0.0; 3], [1.0, 0.0, 0.0]);
        assert!((c.get_length() - 1.0).abs() < 1e-12);
        c.set_def(CurveDef::Line3 {
            v1: [0.0; 3],
            v2: [2.0, 0.0, 0.0],
        });
        assert!((c.get_length() - 1.0).abs() < 1e-12);
        c.set_needs_update(true);
        assert!((c.get_length() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn path_walks_its_curves() {
        let a = ThreeCurve::line3([0.0; 3], [1.0, 0.0, 0.0]);
        let b = ThreeCurve::line3([1.0, 0.0, 0.0], [1.0, 1.0, 0.0]);
        let p = ThreeCurve::path(vec![a, b], false);
        assert!((p.get_length() - 2.0).abs() < 1e-9);
        assert!(close(p.get_point(0.75).unwrap(), [1.0, 0.5, 0.0]));
        let pts = p.get_points(12);
        assert_eq!(pts.len(), 3);
        p.close_path();
        assert_eq!(p.children().unwrap().len(), 3);
    }
}
