//! SVG path data — writing the crate's curves out, and reading them back in.
//!
//! A [`Path`] is already a sequence of Béziers, arcs and lines,
//! which is the same vocabulary SVG's `d` attribute speaks. Flattening one to a
//! polyline to draw it therefore throws away exactly the information the format
//! was built to carry: a circle becomes a 32-gon, and every downstream editor,
//! plotter and font tool sees a polygon instead of a curve.
//!
//! ```
//! use threers::prelude::*;
//! use threers::Path;
//!
//! let mut p = Path::new();
//! p.move_to(Vector2::new(0.0, 0.0));
//! p.bezier_curve_to(
//!     Vector2::new(0.0, 8.0),
//!     Vector2::new(10.0, 8.0),
//!     Vector2::new(10.0, 0.0),
//! );
//! assert_eq!(p.to_svg_path_data(3), "M0 0C0 8 10 8 10 0");
//! ```
//!
//! # Which way is up
//!
//! The `d` string carries the path's own coordinates, untouched. SVG's y axis
//! points *down* and this crate's 2D curves are y-up, so dropping the string
//! straight into a document draws the path mirrored. That is a property of the
//! coordinates, not a bug to fix here: flipping inside the serialiser would
//! make the output disagree with the numbers that went in. Wrap it in a
//! `transform="scale(1,-1)"`, or flip the `viewBox`, at the point where the
//! document is assembled — which is what
//! `openscad::schematic_to_svg` and the origami net
//! writers already do.

use super::{
    CubicBezierCurve, Curve2, CurvePath, EllipseCurve, LineCurve, Path, QuadraticBezierCurve,
    Shape, SplineCurve,
};
use crate::math::Vector2;
use crate::utils::format_svg_number;
use std::f32::consts::PI;

/// Divisions used when a curve has no exact SVG form and has to be flattened.
///
/// Only reached by curve types SVG cannot express — a NURBS curve, or a
/// [`Curve2`] implemented outside the crate. Everything built by [`Path`]'s own
/// methods has a closed form and is written exactly.
const FLATTEN_DIVISIONS: usize = 48;

/// One SVG drawing command, with its endpoint in absolute coordinates.
///
/// The starting point is deliberately absent: a segment only ever means
/// anything relative to wherever the pen already is, and carrying a copy of the
/// previous endpoint in every segment is one more thing that can disagree with
/// itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathSegment {
    /// `L`
    Line { to: Vector2 },
    /// `Q`
    Quadratic { control: Vector2, to: Vector2 },
    /// `C`
    Cubic {
        c1: Vector2,
        c2: Vector2,
        to: Vector2,
    },
    /// `A`. `x_rotation` is in radians here and written out in degrees, which
    /// is the one unit SVG measures in degrees.
    Arc {
        radii: Vector2,
        x_rotation: f32,
        large_arc: bool,
        sweep: bool,
        to: Vector2,
    },
}

impl PathSegment {
    /// Where the pen ends up.
    pub fn end(&self) -> Vector2 {
        match *self {
            PathSegment::Line { to }
            | PathSegment::Quadratic { to, .. }
            | PathSegment::Cubic { to, .. }
            | PathSegment::Arc { to, .. } => to,
        }
    }

    fn write(&self, out: &mut String, p: usize) {
        let n = |v: f32| format_svg_number(v, p);
        match *self {
            PathSegment::Line { to } => {
                out.push('L');
                out.push_str(&n(to.x));
                out.push(' ');
                out.push_str(&n(to.y));
            }
            PathSegment::Quadratic { control, to } => {
                out.push('Q');
                for (i, v) in [control.x, control.y, to.x, to.y].iter().enumerate() {
                    if i > 0 {
                        out.push(' ');
                    }
                    out.push_str(&n(*v));
                }
            }
            PathSegment::Cubic { c1, c2, to } => {
                out.push('C');
                for (i, v) in [c1.x, c1.y, c2.x, c2.y, to.x, to.y].iter().enumerate() {
                    if i > 0 {
                        out.push(' ');
                    }
                    out.push_str(&n(*v));
                }
            }
            PathSegment::Arc {
                radii,
                x_rotation,
                large_arc,
                sweep,
                to,
            } => {
                out.push('A');
                out.push_str(&n(radii.x));
                out.push(' ');
                out.push_str(&n(radii.y));
                out.push(' ');
                out.push_str(&n(x_rotation.to_degrees()));
                out.push(' ');
                out.push(if large_arc { '1' } else { '0' });
                out.push(' ');
                out.push(if sweep { '1' } else { '0' });
                out.push(' ');
                out.push_str(&n(to.x));
                out.push(' ');
                out.push_str(&n(to.y));
            }
        }
    }
}

/// Write a run of segments as a `d` string, starting the pen at `start`.
pub fn segments_to_path_data(
    start: Vector2,
    segments: &[PathSegment],
    precision: usize,
    close: bool,
) -> String {
    let mut out = String::with_capacity(segments.len() * 16 + 16);
    out.push('M');
    out.push_str(&format_svg_number(start.x, precision));
    out.push(' ');
    out.push_str(&format_svg_number(start.y, precision));
    for s in segments {
        s.write(&mut out, precision);
    }
    if close {
        out.push('Z');
    }
    out
}

// ======================================================================
//                       CURVES → EXACT SEGMENTS
// ======================================================================

pub(super) fn line_segments(c: &LineCurve) -> Vec<PathSegment> {
    vec![PathSegment::Line { to: c.v2 }]
}

pub(super) fn quadratic_segments(c: &QuadraticBezierCurve) -> Vec<PathSegment> {
    vec![PathSegment::Quadratic {
        control: c.v1,
        to: c.v2,
    }]
}

pub(super) fn cubic_segments(c: &CubicBezierCurve) -> Vec<PathSegment> {
    vec![PathSegment::Cubic {
        c1: c.v1,
        c2: c.v2,
        to: c.v3,
    }]
}

/// A Catmull-Rom spline is a cubic Hermite in disguise, and Hermite converts to
/// Bézier exactly: the control points sit a third of the way along each end
/// tangent. So this is a lossless rewrite, not an approximation.
pub(super) fn spline_segments(c: &SplineCurve) -> Vec<PathSegment> {
    let pts = &c.points;
    if pts.len() < 2 {
        return Vec::new();
    }
    const TENSION: f32 = 0.5;
    let n = pts.len();
    let at = |k: i32| pts[(k.clamp(0, n as i32 - 1)) as usize];
    (0..n - 1)
        .map(|i| {
            let i = i as i32;
            let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
            let m1 = (p2 - p0) * TENSION;
            let m2 = (p3 - p1) * TENSION;
            PathSegment::Cubic {
                c1: p1 + m1 * (1.0 / 3.0),
                c2: p2 - m2 * (1.0 / 3.0),
                to: p2,
            }
        })
        .collect()
}

/// An ellipse in SVG is given by its *endpoints* plus two flags, not by its
/// centre and angles, so this is the standard centre → endpoint conversion.
///
/// A sweep of a full turn or more cannot be written as one arc — the endpoints
/// would coincide and SVG draws nothing at all — so it goes out as two halves.
pub(super) fn ellipse_segments(c: &EllipseCurve) -> Vec<PathSegment> {
    let delta = ellipse_delta(c);
    if delta.abs() < 1e-6 || c.x_radius.abs() < 1e-9 || c.y_radius.abs() < 1e-9 {
        return Vec::new();
    }
    let radii = Vector2::new(c.x_radius.abs(), c.y_radius.abs());
    // SVG's sweep flag is "increasing angle" in the coordinate system the path
    // is read in, which is the same sense as this delta's sign.
    let sweep = delta > 0.0;
    let mut out = Vec::new();
    let halves = if delta.abs() > PI * 2.0 - 1e-6 { 2 } else { 1 };
    for k in 1..=halves {
        let t = k as f32 / halves as f32;
        let span = delta / halves as f32;
        out.push(PathSegment::Arc {
            radii,
            x_rotation: c.rotation,
            large_arc: span.abs() > PI,
            sweep,
            to: c.get_point(t),
        });
    }
    out
}

/// The signed sweep, resolved the same way [`EllipseCurve::get_point`] does —
/// including its wrap-around and its `clockwise` handling, which is where a
/// second implementation would quietly disagree.
fn ellipse_delta(c: &EllipseCurve) -> f32 {
    let two_pi = PI * 2.0;
    let mut delta = c.a_end - c.a_start;
    let same = delta.abs() < f32::EPSILON;
    while delta < 0.0 {
        delta += two_pi;
    }
    while delta > two_pi {
        delta -= two_pi;
    }
    if delta < f32::EPSILON {
        delta = if same { 0.0 } else { two_pi };
    }
    if c.clockwise && !same {
        delta = if (delta - two_pi).abs() < f32::EPSILON {
            -two_pi
        } else {
            delta - two_pi
        };
    }
    delta
}

/// Flatten a curve that has no exact form into line segments.
fn flatten(c: &dyn Curve2) -> Vec<PathSegment> {
    c.get_points(FLATTEN_DIVISIONS)
        .into_iter()
        .skip(1)
        .map(|to| PathSegment::Line { to })
        .collect()
}

// ======================================================================
//                        PATHS → `d` STRINGS
// ======================================================================

/// How far apart two endpoints must be before the pen is treated as having
/// jumped, and a new `M` is written rather than a continuing segment.
const GAP: f32 = 1e-5;

impl CurvePath {
    /// Every sub-curve as SVG path data, exactly where it has a closed form and
    /// flattened where it does not.
    ///
    /// A gap between one curve's end and the next one's start becomes a fresh
    /// `M`, so a path built with more than one `move_to` comes out as more than
    /// one subpath rather than joined by a line that was never drawn.
    pub fn to_svg_path_data(&self, precision: usize) -> String {
        let mut out = String::new();
        let mut cursor: Option<Vector2> = None;
        for curve in &self.curves {
            let start = curve.get_point(0.0);
            if cursor.is_none_or(|c| (c - start).length() > GAP) {
                out.push('M');
                out.push_str(&format_svg_number(start.x, precision));
                out.push(' ');
                out.push_str(&format_svg_number(start.y, precision));
            }
            let segments = curve
                .svg_segments()
                .unwrap_or_else(|| flatten(curve.as_ref()));
            for s in &segments {
                s.write(&mut out, precision);
            }
            cursor = segments.last().map(|s| s.end()).or(Some(start));
        }
        if self.auto_close && !out.is_empty() {
            out.push('Z');
        }
        out
    }
}

impl Path {
    /// This path as an SVG `d` string. See the [module docs](self) on which way
    /// up the coordinates are.
    pub fn to_svg_path_data(&self, precision: usize) -> String {
        self.curve_path.to_svg_path_data(precision)
    }
}

impl Shape {
    /// Outline then holes, each as its own closed subpath.
    ///
    /// Fill it with `fill-rule="evenodd"`. The default, `nonzero`, only leaves a
    /// hole where the hole winds opposite to the outline, and nothing in this
    /// crate or in three.js enforces that — a hole traced the same way round as
    /// its outline fills solid and the shape looks simply wrong.
    pub fn to_svg_path_data(&self, precision: usize) -> String {
        let mut out = self.outline.to_svg_path_data(precision);
        if !out.is_empty() && !out.ends_with('Z') {
            out.push('Z');
        }
        for hole in &self.holes {
            let d = hole.to_svg_path_data(precision);
            if d.is_empty() {
                continue;
            }
            out.push_str(&d);
            if !out.ends_with('Z') {
                out.push('Z');
            }
        }
        out
    }
}

// ======================================================================
//                        `d` STRINGS → PATHS
// ======================================================================

/// Why a `d` string could not be read, and where it gave up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvgPathError {
    /// What went wrong, in a sentence.
    pub message: String,
    /// Byte offset into the input.
    pub offset: usize,
}

impl std::fmt::Display for SvgPathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SVG path data at byte {}: {}", self.offset, self.message)
    }
}

impl std::error::Error for SvgPathError {}

/// Read SVG path data into one [`Path`] per subpath.
///
/// Handles the whole `d` grammar: `M L H V C S Q T A Z`, their relative
/// lowercase forms, the implicit repeat where a command's parameters recur
/// without the letter, and `S`/`T`'s reflected control points. Arcs become
/// [`EllipseCurve`]s through the spec's endpoint-to-centre conversion, so they
/// stay arcs rather than being sampled into a polyline.
///
/// A subpath closed with `Z` comes back with `auto_close` set on its
/// [`CurvePath`] rather than with an extra line segment appended, so writing it
/// out again produces the `Z` it started with.
pub fn parse_svg_subpaths(d: &str) -> Result<Vec<Path>, SvgPathError> {
    Parser::new(d).run()
}

impl Path {
    /// Read SVG path data. Subpaths are concatenated into this one path, with a
    /// `Z` becoming the line it draws — a single `Path` has one `auto_close`
    /// flag and cannot record "the second subpath was closed and the third was
    /// not". Use [`parse_svg_subpaths`] or [`Shape::from_svg_path_data`] when
    /// that distinction matters.
    pub fn from_svg_path_data(d: &str) -> Result<Path, SvgPathError> {
        let subpaths = parse_svg_subpaths(d)?;
        let mut out = Path::new();
        for sub in subpaths {
            let close_to = sub.curve_path.auto_close.then(|| {
                sub.curve_path
                    .curves
                    .first()
                    .map(|c| c.get_point(0.0))
                    .unwrap_or(Vector2::ZERO)
            });
            let mut end = out.current;
            for c in sub.curve_path.curves {
                end = c.get_point(1.0);
                out.curve_path.add(c);
            }
            out.current = end;
            if let Some(start) = close_to {
                if (end - start).length() > GAP {
                    out.line_to(start);
                }
            }
        }
        Ok(out)
    }
}

impl Shape {
    /// Read SVG path data as an outline plus holes: the first subpath is the
    /// outline and the rest are holes, which is the convention every SVG
    /// exporter writes and [`Shape::to_svg_path_data`] reverses.
    pub fn from_svg_path_data(d: &str) -> Result<Shape, SvgPathError> {
        let mut subpaths = parse_svg_subpaths(d)?.into_iter();
        let outline = subpaths.next().unwrap_or_default();
        let mut shape = Shape::from_path(outline);
        for hole in subpaths {
            shape.add_hole(hole);
        }
        Ok(shape)
    }
}

struct Parser<'a> {
    src: &'a [u8],
    i: usize,
    /// Pen position.
    cur: Vector2,
    /// Where the current subpath began, which `Z` returns to.
    start: Vector2,
    /// Last cubic control point, for `S`'s reflection. `None` unless the
    /// previous command was a cubic.
    last_cubic: Option<Vector2>,
    /// Last quadratic control point, for `T`'s reflection.
    last_quad: Option<Vector2>,
    out: Vec<Path>,
    open: Option<Path>,
}

impl<'a> Parser<'a> {
    fn new(d: &'a str) -> Self {
        Self {
            src: d.as_bytes(),
            i: 0,
            cur: Vector2::ZERO,
            start: Vector2::ZERO,
            last_cubic: None,
            last_quad: None,
            out: Vec::new(),
            open: None,
        }
    }

    fn err<T>(&self, message: impl Into<String>) -> Result<T, SvgPathError> {
        Err(SvgPathError {
            message: message.into(),
            offset: self.i,
        })
    }

    fn run(mut self) -> Result<Vec<Path>, SvgPathError> {
        self.skip_sep();
        if self.i < self.src.len() && !matches!(self.src[self.i], b'M' | b'm') {
            return self.err("path must start with a moveto");
        }
        let mut command = 0u8;
        while self.i < self.src.len() {
            self.skip_sep();
            if self.i >= self.src.len() {
                break;
            }
            let b = self.src[self.i];
            if b.is_ascii_alphabetic() {
                command = b;
                self.i += 1;
            } else if command == 0 {
                return self.err("expected a command letter");
            } else if matches!(command, b'M' | b'm') {
                // "If a moveto is followed by multiple pairs of coordinates,
                // the subsequent pairs are treated as implicit lineto."
                command = if command == b'M' { b'L' } else { b'l' };
            } else if matches!(command, b'Z' | b'z') {
                return self.err("closepath takes no parameters");
            }
            self.step(command)?;
        }
        self.finish_subpath();
        Ok(self.out)
    }

    fn step(&mut self, command: u8) -> Result<(), SvgPathError> {
        let rel = command.is_ascii_lowercase();
        let base = if rel { self.cur } else { Vector2::ZERO };
        match command.to_ascii_uppercase() {
            b'M' => {
                let p = self.point(base)?;
                self.finish_subpath();
                self.open = Some(Path::new());
                if let Some(path) = &mut self.open {
                    path.move_to(p);
                }
                self.cur = p;
                self.start = p;
                self.clear_reflection();
            }
            b'L' => {
                let p = self.point(base)?;
                self.line(p);
                self.clear_reflection();
            }
            b'H' => {
                let x = self.number()? + if rel { self.cur.x } else { 0.0 };
                let p = Vector2::new(x, self.cur.y);
                self.line(p);
                self.clear_reflection();
            }
            b'V' => {
                let y = self.number()? + if rel { self.cur.y } else { 0.0 };
                let p = Vector2::new(self.cur.x, y);
                self.line(p);
                self.clear_reflection();
            }
            b'C' => {
                let c1 = self.point(base)?;
                let c2 = self.point(base)?;
                let to = self.point(base)?;
                self.cubic(c1, c2, to);
            }
            b'S' => {
                // Reflect the previous cubic's second control point about the
                // pen. With no previous cubic the reflection is the pen itself.
                let c1 = self.reflect(self.last_cubic);
                let c2 = self.point(base)?;
                let to = self.point(base)?;
                self.cubic(c1, c2, to);
            }
            b'Q' => {
                let c = self.point(base)?;
                let to = self.point(base)?;
                self.quadratic(c, to);
            }
            b'T' => {
                let c = self.reflect(self.last_quad);
                let to = self.point(base)?;
                self.quadratic(c, to);
            }
            b'A' => {
                let rx = self.number()?;
                let ry = self.number()?;
                let rotation = self.number()?.to_radians();
                let large_arc = self.flag()?;
                let sweep = self.flag()?;
                let to = self.point(base)?;
                self.arc(rx, ry, rotation, large_arc, sweep, to);
                self.clear_reflection();
            }
            b'Z' => {
                if let Some(path) = &mut self.open {
                    path.curve_path.auto_close = true;
                }
                self.cur = self.start;
                self.clear_reflection();
            }
            other => return self.err(format!("unknown command {:?}", other as char)),
        }
        Ok(())
    }

    fn clear_reflection(&mut self) {
        self.last_cubic = None;
        self.last_quad = None;
    }

    /// The previous control point mirrored through the pen — `S`/`T`'s "smooth"
    /// rule, which is what makes a chain of them keep a continuous tangent.
    fn reflect(&self, previous: Option<Vector2>) -> Vector2 {
        match previous {
            Some(c) => self.cur * 2.0 - c,
            None => self.cur,
        }
    }

    fn ensure_open(&mut self) -> &mut Path {
        if self.open.is_none() {
            let mut p = Path::new();
            p.move_to(self.cur);
            self.open = Some(p);
        }
        self.open.as_mut().expect("just opened")
    }

    fn line(&mut self, p: Vector2) {
        self.ensure_open().line_to(p);
        self.cur = p;
    }

    fn cubic(&mut self, c1: Vector2, c2: Vector2, to: Vector2) {
        self.ensure_open().bezier_curve_to(c1, c2, to);
        self.cur = to;
        self.last_cubic = Some(c2);
        self.last_quad = None;
    }

    fn quadratic(&mut self, c: Vector2, to: Vector2) {
        self.ensure_open().quadratic_curve_to(c, to);
        self.cur = to;
        self.last_quad = Some(c);
        self.last_cubic = None;
    }

    fn finish_subpath(&mut self) {
        if let Some(path) = self.open.take() {
            if !path.curve_path.curves.is_empty() {
                self.out.push(path);
            }
        }
    }

    // ---- lexing ----

    fn skip_sep(&mut self) {
        while self.i < self.src.len()
            && matches!(self.src[self.i], b' ' | b'\t' | b'\r' | b'\n' | b',')
        {
            self.i += 1;
        }
    }

    fn point(&mut self, base: Vector2) -> Result<Vector2, SvgPathError> {
        let x = self.number()?;
        let y = self.number()?;
        Ok(Vector2::new(base.x + x, base.y + y))
    }

    /// `large-arc` and `sweep` are single digits and, uniquely in this grammar,
    /// need no separator: `a1 1 0 011 1` is a valid arc with both flags set.
    /// Reading them as ordinary numbers swallows the coordinates behind them.
    fn flag(&mut self) -> Result<bool, SvgPathError> {
        self.skip_sep();
        match self.src.get(self.i) {
            Some(b'0') => {
                self.i += 1;
                Ok(false)
            }
            Some(b'1') => {
                self.i += 1;
                Ok(true)
            }
            _ => self.err("arc flag must be 0 or 1"),
        }
    }

    fn number(&mut self) -> Result<f32, SvgPathError> {
        self.skip_sep();
        let start = self.i;
        if matches!(self.src.get(self.i), Some(b'+' | b'-')) {
            self.i += 1;
        }
        let int_digits = self.digits();
        let mut frac_digits = 0;
        if self.src.get(self.i) == Some(&b'.') {
            self.i += 1;
            frac_digits = self.digits();
        }
        if int_digits == 0 && frac_digits == 0 {
            self.i = start;
            return self.err("expected a number");
        }
        if matches!(self.src.get(self.i), Some(b'e' | b'E')) {
            let mark = self.i;
            self.i += 1;
            if matches!(self.src.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if self.digits() == 0 {
                // A trailing `e` that is not an exponent belongs to whatever
                // comes next, so give it back.
                self.i = mark;
            }
        }
        let text = std::str::from_utf8(&self.src[start..self.i]).unwrap_or("");
        match text.parse::<f32>() {
            Ok(v) if v.is_finite() => Ok(v),
            _ => {
                self.i = start;
                self.err(format!("{text:?} is not a finite number"))
            }
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.i;
        while matches!(self.src.get(self.i), Some(c) if c.is_ascii_digit()) {
            self.i += 1;
        }
        self.i - start
    }

    // ---- arcs ----

    /// SVG gives an arc by its endpoints; [`EllipseCurve`] wants a centre and
    /// two angles. This is the conversion from the spec's implementation notes
    /// (F.6.5), including the F.6.6 correction that grows radii too small to
    /// span the endpoints instead of failing.
    fn arc(&mut self, rx: f32, ry: f32, rotation: f32, large_arc: bool, sweep: bool, to: Vector2) {
        let from = self.cur;
        if (to - from).length() < GAP {
            // Coincident endpoints: the spec says draw nothing.
            return;
        }
        let (mut rx, mut ry) = (rx.abs(), ry.abs());
        if rx < GAP || ry < GAP {
            // Degenerate radii: the spec says draw a straight line.
            self.line(to);
            return;
        }

        let (sin_p, cos_p) = rotation.sin_cos();
        let dx = (from.x - to.x) * 0.5;
        let dy = (from.y - to.y) * 0.5;
        let x1 = cos_p * dx + sin_p * dy;
        let y1 = -sin_p * dx + cos_p * dy;

        // Radii that cannot reach across the chord are scaled up until they can.
        let lambda = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry);
        if lambda > 1.0 {
            let k = lambda.sqrt();
            rx *= k;
            ry *= k;
        }

        let num = (rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1).max(0.0);
        let den = rx * rx * y1 * y1 + ry * ry * x1 * x1;
        let coef = if den > 0.0 { (num / den).sqrt() } else { 0.0 };
        let sign = if large_arc == sweep { -1.0 } else { 1.0 };
        let cx1 = sign * coef * (rx * y1 / ry);
        let cy1 = sign * coef * -(ry * x1 / rx);

        let center = Vector2::new(
            cos_p * cx1 - sin_p * cy1 + (from.x + to.x) * 0.5,
            sin_p * cx1 + cos_p * cy1 + (from.y + to.y) * 0.5,
        );

        let start_angle = ((y1 - cy1) / ry).atan2((x1 - cx1) / rx);
        let end_angle = ((-y1 - cy1) / ry).atan2((-x1 - cx1) / rx);
        let mut delta = end_angle - start_angle;
        let two_pi = PI * 2.0;
        if sweep && delta < 0.0 {
            delta += two_pi;
        } else if !sweep && delta > 0.0 {
            delta -= two_pi;
        }

        let curve = EllipseCurve::new(
            center,
            rx,
            ry,
            start_angle,
            start_angle + delta,
            delta < 0.0,
            rotation,
        );
        self.ensure_open().curve_path.add(Box::new(curve));
        self.cur = to;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32) -> Vector2 {
        Vector2::new(x, y)
    }

    fn close(a: Vector2, b: Vector2) -> bool {
        (a - b).length() < 2e-3
    }

    /// Sample both paths at the same parameters and compare. Comparing the `d`
    /// strings instead would fail on formatting that means the same curve, and
    /// pass on an arc that parsed into the wrong half of an ellipse.
    fn same_shape(a: &Path, b: &Path) -> bool {
        (0..=40).all(|i| {
            let t = i as f32 / 40.0;
            close(a.curve_path.get_point(t), b.curve_path.get_point(t))
        })
    }

    // ---- writing ----

    #[test]
    fn writes_each_curve_in_its_own_form() {
        let mut p = Path::new();
        p.move_to(v(0.0, 0.0));
        p.line_to(v(10.0, 0.0));
        p.quadratic_curve_to(v(15.0, 5.0), v(10.0, 10.0));
        p.bezier_curve_to(v(6.0, 12.0), v(2.0, 12.0), v(0.0, 10.0));
        assert_eq!(p.to_svg_path_data(3), "M0 0L10 0Q15 5 10 10C6 12 2 12 0 10");
    }

    /// The whole point: a curve stays a curve. Flattening would emit dozens of
    /// `L`s and lose the control points for good.
    #[test]
    fn a_cubic_is_one_command_not_a_polyline() {
        let mut p = Path::new();
        p.move_to(v(0.0, 0.0));
        p.bezier_curve_to(v(0.0, 8.0), v(10.0, 8.0), v(10.0, 0.0));
        let d = p.to_svg_path_data(3);
        assert_eq!(d.matches('C').count(), 1);
        assert_eq!(d.matches('L').count(), 0);
    }

    #[test]
    fn a_second_moveto_starts_a_new_subpath() {
        let mut p = Path::new();
        p.move_to(v(0.0, 0.0));
        p.line_to(v(4.0, 0.0));
        p.move_to(v(10.0, 10.0));
        p.line_to(v(14.0, 10.0));
        assert_eq!(p.to_svg_path_data(2), "M0 0L4 0M10 10L14 10");
    }

    /// An arc goes out as `A`, and a full turn as two of them — one arc whose
    /// endpoints coincide is a no-op in SVG and would draw nothing.
    #[test]
    fn arcs_stay_arcs() {
        let mut half = Path::new();
        half.arc(v(0.0, 0.0), 5.0, 0.0, PI, false);
        assert_eq!(half.to_svg_path_data(3), "M5 0A5 5 0 0 1 -5 0");

        let mut circle = Path::new();
        circle.arc(v(0.0, 0.0), 5.0, 0.0, PI * 2.0, false);
        assert_eq!(circle.to_svg_path_data(3).matches('A').count(), 2);
    }

    #[test]
    fn clockwise_arcs_flip_the_sweep_flag() {
        let mut cw = Path::new();
        cw.arc(v(0.0, 0.0), 5.0, 0.0, PI, true);
        let d = cw.to_svg_path_data(3);
        assert!(d.contains(" 0 0 "), "expected sweep 0 in {d}");
    }

    /// Catmull-Rom is a cubic Hermite, so this is an exact rewrite rather than
    /// a fit: the curve through the sampled points must be unchanged.
    #[test]
    fn splines_convert_to_beziers_exactly() {
        let spline = SplineCurve::new(vec![v(0.0, 0.0), v(4.0, 6.0), v(9.0, 2.0), v(14.0, 7.0)]);
        let segments = spline.svg_segments().expect("exact form");
        assert_eq!(segments.len(), 3);

        let mut rebuilt = Path::new();
        rebuilt.move_to(v(0.0, 0.0));
        for s in &segments {
            match *s {
                PathSegment::Cubic { c1, c2, to } => {
                    rebuilt.bezier_curve_to(c1, c2, to);
                }
                _ => panic!("expected cubics, got {s:?}"),
            }
        }
        // Per span, because the two parameterise their arc length differently.
        for i in 0..3 {
            for k in 0..=8 {
                let local = k as f32 / 8.0;
                let want = spline.get_point((i as f32 + local) / 3.0);
                let got = rebuilt.curve_path.curves[i].get_point(local);
                assert!(close(want, got), "span {i} at {local}: {want:?} vs {got:?}");
            }
        }
    }

    /// A curve with no SVG form still exports — as a polyline, not as nothing.
    #[test]
    fn curves_without_an_exact_form_are_flattened() {
        struct Wiggle;
        impl Curve2 for Wiggle {
            fn get_point(&self, t: f32) -> Vector2 {
                Vector2::new(t * 10.0, (t * 6.0).sin())
            }
        }
        let mut cp = CurvePath::new();
        cp.add(Box::new(Wiggle));
        let d = cp.to_svg_path_data(2);
        assert!(d.starts_with("M0 0"));
        assert_eq!(d.matches('L').count(), FLATTEN_DIVISIONS);
    }

    #[test]
    fn shape_writes_holes_as_closed_subpaths() {
        let mut outline = Path::new();
        outline.move_to(v(0.0, 0.0));
        outline.line_to(v(10.0, 0.0));
        outline.line_to(v(10.0, 10.0));
        outline.line_to(v(0.0, 10.0));
        let mut hole = Path::new();
        hole.move_to(v(3.0, 3.0));
        hole.line_to(v(3.0, 7.0));
        hole.line_to(v(7.0, 7.0));
        hole.line_to(v(7.0, 3.0));
        let mut shape = Shape::from_path(outline);
        shape.add_hole(hole);
        assert_eq!(
            shape.to_svg_path_data(2),
            "M0 0L10 0L10 10L0 10ZM3 3L3 7L7 7L7 3Z"
        );
    }

    // ---- reading ----

    #[test]
    fn reads_absolute_and_relative_forms_alike() {
        let abs = Path::from_svg_path_data("M10 10 L20 10 L20 20").unwrap();
        let rel = Path::from_svg_path_data("m10 10 l10 0 l0 10").unwrap();
        assert!(same_shape(&abs, &rel));
    }

    #[test]
    fn reads_horizontal_and_vertical_shorthands() {
        let short = Path::from_svg_path_data("M0 0 H10 V10 h-10 v-10").unwrap();
        let long = Path::from_svg_path_data("M0 0 L10 0 L10 10 L0 10 L0 0").unwrap();
        assert!(same_shape(&short, &long));
    }

    /// A moveto followed by extra coordinate pairs is an implicit lineto — a
    /// rule that is easy to miss and silently turns a polygon into a point.
    #[test]
    fn a_moveto_with_extra_pairs_is_an_implicit_lineto() {
        let implicit = Path::from_svg_path_data("M0 0 10 0 10 10").unwrap();
        let explicit = Path::from_svg_path_data("M0 0 L10 0 L10 10").unwrap();
        assert!(same_shape(&implicit, &explicit));
        assert_eq!(implicit.curve_path.curves.len(), 2);
    }

    #[test]
    fn smooth_cubics_reflect_the_previous_control_point() {
        let smooth = Path::from_svg_path_data("M0 0 C0 5 5 5 5 0 S10 -5 10 0").unwrap();
        let spelled = Path::from_svg_path_data("M0 0 C0 5 5 5 5 0 C5 -5 10 -5 10 0").unwrap();
        assert!(same_shape(&smooth, &spelled));
    }

    #[test]
    fn smooth_quadratics_reflect_the_previous_control_point() {
        let smooth = Path::from_svg_path_data("M0 0 Q5 5 10 0 T20 0").unwrap();
        let spelled = Path::from_svg_path_data("M0 0 Q5 5 10 0 Q15 -5 20 0").unwrap();
        assert!(same_shape(&smooth, &spelled));
    }

    /// Arc flags are single characters and need no separator, so `0 011 1` is
    /// large-arc=0, sweep=1, then the point (1, 1). Lexing them as numbers eats
    /// the coordinates and silently draws the wrong arc.
    #[test]
    fn arc_flags_need_no_separator() {
        let packed = Path::from_svg_path_data("M0 0 a5 5 0 011 1").unwrap();
        let spaced = Path::from_svg_path_data("M0 0 a5 5 0 0 1 1 1").unwrap();
        assert!(same_shape(&packed, &spaced));
        assert!(close(packed.curve_path.get_point(1.0), v(1.0, 1.0)));
    }

    #[test]
    fn arcs_land_on_their_endpoint() {
        for (d, end) in [
            ("M0 0 A5 5 0 0 1 10 0", v(10.0, 0.0)),
            ("M0 0 A5 5 0 1 0 10 0", v(10.0, 0.0)),
            ("M10 10 A20 8 30 1 1 40 25", v(40.0, 25.0)),
        ] {
            let p = Path::from_svg_path_data(d).unwrap();
            let got = p.curve_path.get_point(1.0);
            assert!(close(got, end), "{d}: ended at {got:?}, wanted {end:?}");
        }
    }

    /// Radii too small to span the endpoints are scaled up rather than rejected
    /// — F.6.6 in the spec, and common in real files.
    #[test]
    fn undersized_arc_radii_are_grown_to_fit() {
        // Radii of 1 cannot span a chord of 10, so both grow to exactly 5 and
        // the arc becomes the semicircle on that chord.
        let p = Path::from_svg_path_data("M0 0 A1 1 0 0 1 10 0").unwrap();
        assert!(close(p.curve_path.get_point(1.0), v(10.0, 0.0)));
        assert!(close(p.curve_path.get_point(0.5), v(5.0, -5.0)));
    }

    /// The sweep flag picks which side of the chord the arc bulges to, and
    /// getting it backwards is the classic way to mirror imported artwork.
    #[test]
    fn the_sweep_flag_picks_the_side() {
        let one = Path::from_svg_path_data("M0 0 A5 5 0 0 1 10 0").unwrap();
        let zero = Path::from_svg_path_data("M0 0 A5 5 0 0 0 10 0").unwrap();
        assert!(close(one.curve_path.get_point(0.5), v(5.0, -5.0)));
        assert!(close(zero.curve_path.get_point(0.5), v(5.0, 5.0)));
    }

    #[test]
    fn degenerate_arcs_become_lines() {
        let p = Path::from_svg_path_data("M0 0 A0 0 0 0 1 10 0").unwrap();
        assert_eq!(p.curve_path.curves.len(), 1);
        assert!(close(p.curve_path.get_point(0.5), v(5.0, 0.0)));
    }

    #[test]
    fn reads_exponents_and_unseparated_signs() {
        let p = Path::from_svg_path_data("M0 0L1e2 0L1e2-50L-.5-.5").unwrap();
        assert!(close(p.curve_path.curves[0].get_point(1.0), v(100.0, 0.0)));
        assert!(close(
            p.curve_path.curves[1].get_point(1.0),
            v(100.0, -50.0)
        ));
        assert!(close(p.curve_path.curves[2].get_point(1.0), v(-0.5, -0.5)));
    }

    #[test]
    fn subpaths_split_and_record_whether_they_closed() {
        let subs = parse_svg_subpaths("M0 0L4 0L4 4Z M10 10L14 10").unwrap();
        assert_eq!(subs.len(), 2);
        assert!(subs[0].curve_path.auto_close);
        assert!(!subs[1].curve_path.auto_close);
    }

    #[test]
    fn shape_reads_outline_then_holes() {
        let shape = Shape::from_svg_path_data("M0 0L10 0L10 10L0 10Z M3 3L3 7L7 7L7 3Z").unwrap();
        assert_eq!(shape.holes.len(), 1);
        assert!(close(shape.outline.curve_path.get_point(0.0), v(0.0, 0.0)));
        assert!(close(shape.holes[0].curve_path.get_point(0.0), v(3.0, 3.0)));
    }

    #[test]
    fn bad_input_says_where() {
        assert!(Path::from_svg_path_data("L10 10").is_err());
        let e = Path::from_svg_path_data("M0 0 L10").unwrap_err();
        assert!(e.message.contains("number"), "{e}");
        let e = Path::from_svg_path_data("M0 0 A5 5 0 9 1 10 0").unwrap_err();
        assert!(e.message.contains("flag"), "{e}");
        assert!(Path::from_svg_path_data("M0 0 K5 5").is_err());
    }

    #[test]
    fn empty_input_is_an_empty_path() {
        assert!(Path::from_svg_path_data("")
            .unwrap()
            .curve_path
            .curves
            .is_empty());
        assert!(parse_svg_subpaths("   ").unwrap().is_empty());
    }

    // ---- both directions ----

    #[test]
    fn round_trips_through_text() {
        for d in [
            "M0 0L10 0L10 10L0 10Z",
            "M0 0C0 8 10 8 10 0",
            "M0 0Q5 5 10 0",
            "M5 0A5 5 0 0 1 -5 0",
            "M10 10A20 8 30 1 1 40 25",
            "M0 0L4 0M10 10L14 10",
        ] {
            let once = Path::from_svg_path_data(d).unwrap();
            let text = once.to_svg_path_data(4);
            let twice = Path::from_svg_path_data(&text).unwrap();
            assert!(
                same_shape(&once, &twice),
                "{d} -> {text} did not survive the trip"
            );
        }
    }

    #[test]
    fn shape_round_trips_with_its_holes() {
        let d = "M0 0L10 0L10 10L0 10ZM3 3L3 7L7 7L7 3Z";
        let shape = Shape::from_svg_path_data(d).unwrap();
        assert_eq!(shape.to_svg_path_data(3), d);
    }
}
