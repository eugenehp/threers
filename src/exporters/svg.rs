//! SVG drawings: an [`SvgDocument`] of paths, circles, rectangles, lines,
//! polylines, text and images, nested in groups with their own transforms
//! and styles, painted in colours or gradients, clipped and made translucent
//! as groups, written as SVG 1.1 ([`write_svg`]).
//!
//! The document is in user units with y down, as SVG has it; the page size
//! carries a unit (`mm`, `in`, `px`…) and the view box maps user units onto
//! it. [`SvgDocument::y_up`] flips the drawing so callers can stay in
//! y-up CAD coordinates. Numbers are printed with [`SvgDocument::precision`]
//! decimals, trailing zeros dropped, or shortest round-trip when it is
//! `None`.
//!
//! ```
//! use threers::exporters::{write_svg, SvgDocument, SvgPath, SvgStyle, SvgUnit};
//!
//! let mut doc = SvgDocument::new(100.0, 50.0, SvgUnit::Millimeters);
//! let outline = SvgStyle::stroke([0, 0, 0], 0.2);
//! doc.rect([5.0, 5.0], [90.0, 40.0], 2.0, outline.clone());
//! doc.circle([25.0, 25.0], 8.0, SvgStyle::fill([200, 52, 52]));
//! let mut slot = SvgPath::new();
//! slot.move_to([50.0, 20.0]).line_to([70.0, 20.0]).arc([70.0, 25.0], 5.0, -90.0, 180.0);
//! slot.line_to([50.0, 30.0]).arc([50.0, 25.0], 5.0, 90.0, 180.0).close();
//! doc.path(slot, outline);
//! let text = write_svg(&doc);
//! assert!(text.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg "));
//! assert!(text.ends_with("</svg>\n"));
//! ```

use std::fmt::Write;

use super::js_number;

/// The unit of the page size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SvgUnit {
    /// No unit: CSS pixels.
    #[default]
    None,
    Pixels,
    Millimeters,
    Centimeters,
    Inches,
    Points,
}

impl SvgUnit {
    pub fn suffix(self) -> &'static str {
        match self {
            SvgUnit::None => "",
            SvgUnit::Pixels => "px",
            SvgUnit::Millimeters => "mm",
            SvgUnit::Centimeters => "cm",
            SvgUnit::Inches => "in",
            SvgUnit::Points => "pt",
        }
    }
}

/// How a shape is filled or stroked: not at all, with a colour, or with a
/// gradient.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SvgPaint {
    #[default]
    None,
    /// sRGB, 0–255 per channel.
    Color([u8; 3]),
    /// [`SvgDocument::gradients`]`[k]`.
    Gradient(usize),
}

/// A gradient's geometry, in the user space of the shape it paints
/// (`gradientUnits="userSpaceOnUse"`).
#[derive(Debug, Clone, PartialEq)]
pub enum SvgGradientKind {
    /// Along the line from `start` to `end`.
    Linear { start: [f64; 2], end: [f64; 2] },
    /// From `focus` (the centre when `None`) out to the circle about `center`.
    Radial { center: [f64; 2], radius: f64, focus: Option<[f64; 2]> },
}

/// A colour stop: where along the gradient (0 to 1), its colour, and its
/// opacity (0 to 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvgStop {
    pub offset: f64,
    pub color: [u8; 3],
    pub opacity: f64,
}

/// A gradient, padded beyond its ends (`spreadMethod="pad"`), its colours
/// interpolated in sRGB.
#[derive(Debug, Clone, PartialEq)]
pub struct SvgGradient {
    pub kind: SvgGradientKind,
    pub stops: Vec<SvgStop>,
    /// `gradientTransform`.
    pub transform: Option<SvgTransform>,
}

/// A clipping path, in the user space of what it clips
/// (`clipPathUnits="userSpaceOnUse"`).
#[derive(Debug, Clone, PartialEq)]
pub struct SvgClip {
    pub path: SvgPath,
    pub rule: SvgFillRule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SvgLineCap {
    #[default]
    Butt,
    Round,
    Square,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SvgLineJoin {
    #[default]
    Miter,
    Round,
    Bevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SvgFillRule {
    #[default]
    NonZero,
    EvenOdd,
}

/// The presentation attributes of a shape or group. A field left at its
/// default is not written, so a shape inherits it from its group.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SvgStyle {
    pub fill: Option<SvgPaint>,
    pub stroke: Option<SvgPaint>,
    pub stroke_width: Option<f64>,
    /// 0–1.
    pub fill_opacity: Option<f64>,
    pub stroke_opacity: Option<f64>,
    pub line_cap: Option<SvgLineCap>,
    pub line_join: Option<SvgLineJoin>,
    /// `stroke-miterlimit`: the longest miter, in stroke widths, before a
    /// miter join is bevelled instead (4 when no group sets it).
    pub miter_limit: Option<f64>,
    /// Dash and gap lengths, in user units; empty for solid.
    pub dash: Vec<f64>,
    pub fill_rule: Option<SvgFillRule>,
    /// The element's or group's opacity as a whole, 0–1: a group is
    /// composited once, at this opacity. Not inherited.
    pub opacity: Option<f64>,
    /// [`SvgDocument::clip_paths`]`[k]` clips the element or group. Not
    /// inherited.
    pub clip: Option<usize>,
}

impl SvgStyle {
    /// Filled with `color`, not stroked.
    pub fn fill(color: [u8; 3]) -> Self {
        SvgStyle { fill: Some(SvgPaint::Color(color)), stroke: Some(SvgPaint::None), ..Default::default() }
    }

    /// Stroked with `color` at `width`, round caps and joins, not filled.
    pub fn stroke(color: [u8; 3], width: f64) -> Self {
        SvgStyle {
            fill: Some(SvgPaint::None),
            stroke: Some(SvgPaint::Color(color)),
            stroke_width: Some(width),
            line_cap: Some(SvgLineCap::Round),
            line_join: Some(SvgLineJoin::Round),
            ..Default::default()
        }
    }

    /// The same, with the fill and stroke opacity set.
    pub fn with_opacity(mut self, opacity: f64) -> Self {
        self.fill_opacity = Some(opacity);
        self.stroke_opacity = Some(opacity);
        self
    }

    pub fn with_fill_rule(mut self, rule: SvgFillRule) -> Self {
        self.fill_rule = Some(rule);
        self
    }

    pub fn with_dash(mut self, dash: &[f64]) -> Self {
        self.dash = dash.to_vec();
        self
    }
}

/// One command of a path, in absolute coordinates.
#[derive(Debug, Clone, PartialEq)]
pub enum PathCommand {
    MoveTo([f64; 2]),
    LineTo([f64; 2]),
    /// SVG's elliptical arc: radii, x-axis rotation in degrees, the two flags
    /// and the end point.
    ArcTo { radii: [f64; 2], rotation: f64, large_arc: bool, sweep: bool, to: [f64; 2] },
    CubicTo { c1: [f64; 2], c2: [f64; 2], to: [f64; 2] },
    QuadTo { c: [f64; 2], to: [f64; 2] },
    Close,
}

/// A path built command by command.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SvgPath {
    pub commands: Vec<PathCommand>,
}

impl SvgPath {
    pub fn new() -> Self {
        Self::default()
    }

    /// Through `points`, closed back to the first when `closed`.
    pub fn polygon(points: &[[f64; 2]], closed: bool) -> Self {
        let mut p = SvgPath::new();

        if let Some((first, rest)) = points.split_first() {
            p.move_to(*first);
            rest.iter().for_each(|q| {
                p.line_to(*q);
            });

            if closed {
                p.close();
            }
        }

        p
    }

    pub fn move_to(&mut self, p: [f64; 2]) -> &mut Self {
        self.commands.push(PathCommand::MoveTo(p));
        self
    }

    pub fn line_to(&mut self, p: [f64; 2]) -> &mut Self {
        self.commands.push(PathCommand::LineTo(p));
        self
    }

    pub fn arc_to(&mut self, radii: [f64; 2], rotation: f64, large_arc: bool, sweep: bool, to: [f64; 2]) -> &mut Self {
        self.commands.push(PathCommand::ArcTo { radii, rotation, large_arc, sweep, to });
        self
    }

    /// A circular arc about `center`, from the angle `start` (degrees, the
    /// angle SVG's y-down axes give: positive turns towards +y) through
    /// `sweep` degrees, its start joined to the pen by a line. Sweeps of a
    /// whole turn or more are split in two, since one SVG arc cannot close
    /// on itself.
    pub fn arc(&mut self, center: [f64; 2], radius: f64, start: f64, sweep: f64) -> &mut Self {
        let at = |deg: f64| {
            let a = deg.to_radians();
            [center[0] + radius * a.cos(), center[1] + radius * a.sin()]
        };

        let begin = at(start);

        match self.commands.last() {
            None | Some(PathCommand::Close) => self.move_to(begin),
            _ => self.line_to(begin),
        };

        let sweep = sweep.clamp(-360.0, 360.0);
        let pieces = if sweep.abs() >= 360.0 { 2 } else { 1 };
        let step = sweep / pieces as f64;

        for i in 1..=pieces {
            let end = at(start + step * i as f64);
            self.arc_to([radius, radius], 0.0, step.abs() > 180.0, step > 0.0, end);
        }

        self
    }

    pub fn cubic_to(&mut self, c1: [f64; 2], c2: [f64; 2], to: [f64; 2]) -> &mut Self {
        self.commands.push(PathCommand::CubicTo { c1, c2, to });
        self
    }

    pub fn quad_to(&mut self, c: [f64; 2], to: [f64; 2]) -> &mut Self {
        self.commands.push(PathCommand::QuadTo { c, to });
        self
    }

    pub fn close(&mut self) -> &mut Self {
        self.commands.push(PathCommand::Close);
        self
    }
}

impl SvgPath {
    /// SVG path data (a `d` attribute) as a path of absolute commands: `M L
    /// H V C S Q T A Z`, their relative forms, parameters repeated without
    /// the letter, `S` and `T`'s reflected control points, and arc flags
    /// written without separators (`a1 1 0 011 1`). `None` for data that does
    /// not parse; a valid prefix is not returned on its own.
    pub fn from_data(d: &str) -> Option<SvgPath> {
        let b = d.as_bytes();
        let mut i = 0;
        let skip = |i: &mut usize| {
            while *i < b.len() && (b[*i].is_ascii_whitespace() || b[*i] == b',') {
                *i += 1;
            }
        };
        let number = |i: &mut usize| -> Option<f64> {
            skip(i);
            let start = *i;
            if *i < b.len() && (b[*i] == b'+' || b[*i] == b'-') {
                *i += 1;
            }
            let mut dot = false;
            let mut digits = false;
            while *i < b.len() && (b[*i].is_ascii_digit() || (b[*i] == b'.' && !dot)) {
                dot |= b[*i] == b'.';
                digits |= b[*i].is_ascii_digit();
                *i += 1;
            }
            if *i < b.len() && (b[*i] == b'e' || b[*i] == b'E') && digits {
                let save = *i;
                *i += 1;
                if *i < b.len() && (b[*i] == b'+' || b[*i] == b'-') {
                    *i += 1;
                }
                if *i < b.len() && b[*i].is_ascii_digit() {
                    while *i < b.len() && b[*i].is_ascii_digit() {
                        *i += 1;
                    }
                } else {
                    *i = save;
                }
            }
            if !digits {
                return None;
            }
            d[start..*i].parse().ok()
        };
        let flag = |i: &mut usize| -> Option<bool> {
            skip(i);
            let f = match b.get(*i)? {
                b'0' => false,
                b'1' => true,
                _ => return None,
            };
            *i += 1;
            Some(f)
        };
        let mut path = SvgPath::new();
        let (mut cur, mut start) = ([0.0f64; 2], [0.0f64; 2]);
        // The last command's second control point, for S and T to reflect.
        let mut last_cubic: Option<[f64; 2]> = None;
        let mut last_quad: Option<[f64; 2]> = None;
        let mut cmd: Option<u8> = None;
        loop {
            skip(&mut i);
            if i >= b.len() {
                break;
            }
            if b[i].is_ascii_alphabetic() {
                cmd = Some(b[i]);
                i += 1;
            } else if cmd.is_none() {
                return None;
            }
            let c = cmd?;
            let rel = c.is_ascii_lowercase();
            let at = |p: [f64; 2], cur: [f64; 2]| if rel { [p[0] + cur[0], p[1] + cur[1]] } else { p };
            let pair = |i: &mut usize| -> Option<[f64; 2]> { Some([number(i)?, number(i)?]) };
            let reflect = |p: Option<[f64; 2]>, cur: [f64; 2]| p.map_or(cur, |p| [2.0 * cur[0] - p[0], 2.0 * cur[1] - p[1]]);
            let (mut cubic, mut quad) = (None, None);
            match c.to_ascii_uppercase() {
                b'M' => {
                    let p = at(pair(&mut i)?, cur);
                    path.move_to(p);
                    (cur, start) = (p, p);
                    // Pairs after the first are lines.
                    cmd = Some(if rel { b'l' } else { b'L' });
                }
                b'L' => {
                    cur = at(pair(&mut i)?, cur);
                    path.line_to(cur);
                }
                b'H' => {
                    let x = number(&mut i)?;
                    cur = [if rel { cur[0] + x } else { x }, cur[1]];
                    path.line_to(cur);
                }
                b'V' => {
                    let y = number(&mut i)?;
                    cur = [cur[0], if rel { cur[1] + y } else { y }];
                    path.line_to(cur);
                }
                b'C' => {
                    let (c1, c2, to) = (at(pair(&mut i)?, cur), at(pair(&mut i)?, cur), at(pair(&mut i)?, cur));
                    path.cubic_to(c1, c2, to);
                    (cur, cubic) = (to, Some(c2));
                }
                b'S' => {
                    let c1 = reflect(last_cubic, cur);
                    let (c2, to) = (at(pair(&mut i)?, cur), at(pair(&mut i)?, cur));
                    path.cubic_to(c1, c2, to);
                    (cur, cubic) = (to, Some(c2));
                }
                b'Q' => {
                    let (q, to) = (at(pair(&mut i)?, cur), at(pair(&mut i)?, cur));
                    path.quad_to(q, to);
                    (cur, quad) = (to, Some(q));
                }
                b'T' => {
                    let q = reflect(last_quad, cur);
                    let to = at(pair(&mut i)?, cur);
                    path.quad_to(q, to);
                    (cur, quad) = (to, Some(q));
                }
                b'A' => {
                    let radii = [number(&mut i)?, number(&mut i)?];
                    let rotation = number(&mut i)?;
                    let (large_arc, sweep) = (flag(&mut i)?, flag(&mut i)?);
                    let to = at(pair(&mut i)?, cur);
                    path.arc_to(radii, rotation, large_arc, sweep, to);
                    cur = to;
                }
                b'Z' => {
                    path.close();
                    cur = start;
                    // A command letter must follow.
                    cmd = None;
                }
                _ => return None,
            }
            (last_cubic, last_quad) = (cubic, quad);
        }
        Some(path)
    }
}

/// A 2D affine transform `[a, b, c, d, e, f]`: x' = a·x + c·y + e,
/// y' = b·x + d·y + f, SVG's `matrix()`.
pub type SvgTransform = [f64; 6];

/// Text anchored at its baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SvgTextAnchor {
    #[default]
    Start,
    Middle,
    End,
}

/// What a drawing holds.
#[derive(Debug, Clone, PartialEq)]
pub enum SvgNode {
    Path { path: SvgPath, style: SvgStyle },
    Circle { center: [f64; 2], radius: f64, style: SvgStyle },
    Rect { origin: [f64; 2], size: [f64; 2], corner_radius: f64, style: SvgStyle },
    Line { start: [f64; 2], end: [f64; 2], style: SvgStyle },
    Polyline { points: Vec<[f64; 2]>, closed: bool, style: SvgStyle },
    /// Font size in user units, rotation in degrees about the anchor.
    Text { pos: [f64; 2], text: String, size: f64, anchor: SvgTextAnchor, rotation: f64, font_family: Option<String>, style: SvgStyle },
    /// RGBA pixels stretched over the rectangle at `origin` of `size`
    /// (`preserveAspectRatio="none"`), the image's top row at `origin`'s y.
    /// Its style's `opacity` and `clip` apply.
    Image { origin: [f64; 2], size: [f64; 2], image: std::sync::Arc<crate::utils::png::PngImage>, style: SvgStyle },
    Group { id: Option<String>, transform: Option<SvgTransform>, style: SvgStyle, children: Vec<SvgNode> },
}

/// A drawing: its page and what is on it.
#[derive(Debug, Clone, PartialEq)]
pub struct SvgDocument {
    /// The page size in `unit`s.
    pub width: f64,
    pub height: f64,
    pub unit: SvgUnit,
    /// `[x, y, width, height]` in user units; the page in user units when
    /// `None`.
    pub view_box: Option<[f64; 4]>,
    /// The drawing in y-up coordinates: a group that flips y about the view
    /// box's middle wraps everything.
    pub y_up: bool,
    pub title: Option<String>,
    pub description: Option<String>,
    /// Decimals printed; shortest round-trip when `None`.
    pub precision: Option<usize>,
    pub children: Vec<SvgNode>,
    /// What [`SvgPaint::Gradient`] refers to.
    pub gradients: Vec<SvgGradient>,
    /// What [`SvgStyle::clip`] refers to.
    pub clip_paths: Vec<SvgClip>,
}

impl SvgDocument {
    pub fn new(width: f64, height: f64, unit: SvgUnit) -> Self {
        SvgDocument {
            width,
            height,
            unit,
            view_box: None,
            y_up: false,
            title: None,
            description: None,
            precision: Some(4),
            children: Vec::new(),
            gradients: Vec::new(),
            clip_paths: Vec::new(),
        }
    }

    pub fn path(&mut self, path: SvgPath, style: SvgStyle) {
        self.children.push(SvgNode::Path { path, style });
    }

    pub fn circle(&mut self, center: [f64; 2], radius: f64, style: SvgStyle) {
        self.children.push(SvgNode::Circle { center, radius, style });
    }

    pub fn rect(&mut self, origin: [f64; 2], size: [f64; 2], corner_radius: f64, style: SvgStyle) {
        self.children.push(SvgNode::Rect { origin, size, corner_radius, style });
    }

    pub fn line(&mut self, start: [f64; 2], end: [f64; 2], style: SvgStyle) {
        self.children.push(SvgNode::Line { start, end, style });
    }

    pub fn polyline(&mut self, points: &[[f64; 2]], closed: bool, style: SvgStyle) {
        self.children.push(SvgNode::Polyline { points: points.to_vec(), closed, style });
    }

    pub fn text(&mut self, pos: [f64; 2], text: impl Into<String>, size: f64, anchor: SvgTextAnchor, style: SvgStyle) {
        self.children.push(SvgNode::Text { pos, text: text.into(), size, anchor, rotation: 0.0, font_family: None, style });
    }

    /// A group; returns its children to fill.
    pub fn group(&mut self, id: Option<&str>, transform: Option<SvgTransform>, style: SvgStyle) -> &mut Vec<SvgNode> {
        self.children.push(SvgNode::Group { id: id.map(str::to_string), transform, style, children: Vec::new() });

        match self.children.last_mut() {
            Some(SvgNode::Group { children, .. }) => children,
            _ => unreachable!("just pushed a group"),
        }
    }

    /// Adds a gradient; its paint.
    pub fn gradient(&mut self, g: SvgGradient) -> SvgPaint {
        self.gradients.push(g);
        SvgPaint::Gradient(self.gradients.len() - 1)
    }

    /// Adds a clipping path; its index, for [`SvgStyle::clip`].
    pub fn clip_path(&mut self, path: SvgPath, rule: SvgFillRule) -> usize {
        self.clip_paths.push(SvgClip { path, rule });
        self.clip_paths.len() - 1
    }

    /// The view box, the page's own when none is set.
    pub fn effective_view_box(&self) -> [f64; 4] {
        self.view_box.unwrap_or([0.0, 0.0, self.width, self.height])
    }
}

/// Escapes text for element content, and quotes too when `attribute`.
pub fn xml_escape(s: &str, attribute: bool) -> String {
    let mut out = String::with_capacity(s.len());

    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' if attribute => out.push_str("&quot;"),
            '\n' if attribute => out.push_str("&#xA;"),
            '\t' if attribute => out.push_str("&#x9;"),
            '\r' => out.push_str("&#xD;"),
            _ => out.push(c),
        }
    }

    out
}

struct Writer {
    out: String,
    precision: Option<usize>,
}

impl Writer {
    /// A number: `precision` decimals with trailing zeros dropped, never `-0`.
    fn num(&self, v: f64) -> String {
        let v = if v == 0.0 { 0.0 } else { v };

        match self.precision {
            None => js_number(v),
            Some(p) => {
                let s = format!("{v:.p$}");
                let s = if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s };
                if s == "-0" { "0".to_string() } else { s }
            }
        }
    }

    fn pt(&self, p: [f64; 2]) -> String {
        format!("{},{}", self.num(p[0]), self.num(p[1]))
    }

    fn paint(p: SvgPaint) -> String {
        match p {
            SvgPaint::None => "none".into(),
            SvgPaint::Color([r, g, b]) => format!("#{r:02X}{g:02X}{b:02X}"),
            SvgPaint::Gradient(k) => format!("url(#gradient{k})"),
        }
    }

    fn style(&self, s: &SvgStyle) -> String {
        let mut a = String::new();

        if let Some(f) = s.fill {
            let _ = write!(a, " fill=\"{}\"", Self::paint(f));
        }

        if let Some(o) = s.fill_opacity {
            let _ = write!(a, " fill-opacity=\"{}\"", self.num(o));
        }

        if let Some(r) = s.fill_rule {
            a.push_str(match r {
                SvgFillRule::NonZero => " fill-rule=\"nonzero\"",
                SvgFillRule::EvenOdd => " fill-rule=\"evenodd\"",
            });
        }

        if let Some(st) = s.stroke {
            let _ = write!(a, " stroke=\"{}\"", Self::paint(st));
        }

        if let Some(w) = s.stroke_width {
            let _ = write!(a, " stroke-width=\"{}\"", self.num(w));
        }

        if let Some(o) = s.stroke_opacity {
            let _ = write!(a, " stroke-opacity=\"{}\"", self.num(o));
        }

        if let Some(c) = s.line_cap {
            a.push_str(match c {
                SvgLineCap::Butt => " stroke-linecap=\"butt\"",
                SvgLineCap::Round => " stroke-linecap=\"round\"",
                SvgLineCap::Square => " stroke-linecap=\"square\"",
            });
        }

        if let Some(j) = s.line_join {
            a.push_str(match j {
                SvgLineJoin::Miter => " stroke-linejoin=\"miter\"",
                SvgLineJoin::Round => " stroke-linejoin=\"round\"",
                SvgLineJoin::Bevel => " stroke-linejoin=\"bevel\"",
            });
        }

        if let Some(m) = s.miter_limit {
            let _ = write!(a, " stroke-miterlimit=\"{}\"", self.num(m));
        }

        if !s.dash.is_empty() {
            let list: Vec<String> = s.dash.iter().map(|d| self.num(*d)).collect();
            let _ = write!(a, " stroke-dasharray=\"{}\"", list.join(","));
        }

        if let Some(o) = s.opacity {
            let _ = write!(a, " opacity=\"{}\"", self.num(o));
        }

        if let Some(k) = s.clip {
            let _ = write!(a, " clip-path=\"url(#clip{k})\"");
        }

        a
    }

    fn path_data(&self, p: &SvgPath) -> String {
        let parts: Vec<String> = p
            .commands
            .iter()
            .map(|c| match c {
                PathCommand::MoveTo(q) => format!("M{}", self.pt(*q)),
                PathCommand::LineTo(q) => format!("L{}", self.pt(*q)),
                PathCommand::ArcTo { radii, rotation, large_arc, sweep, to } => format!(
                    "A{},{} {} {} {} {}",
                    self.num(radii[0]),
                    self.num(radii[1]),
                    self.num(*rotation),
                    u8::from(*large_arc),
                    u8::from(*sweep),
                    self.pt(*to)
                ),
                PathCommand::CubicTo { c1, c2, to } => format!("C{} {} {}", self.pt(*c1), self.pt(*c2), self.pt(*to)),
                PathCommand::QuadTo { c, to } => format!("Q{} {}", self.pt(*c), self.pt(*to)),
                PathCommand::Close => "Z".into(),
            })
            .collect();

        parts.join(" ")
    }

    fn node(&mut self, n: &SvgNode, depth: usize) {
        let indent = "  ".repeat(depth);

        let line = match n {
            SvgNode::Path { path, style } => format!("<path d=\"{}\"{}/>", self.path_data(path), self.style(style)),
            SvgNode::Circle { center, radius, style } => format!(
                "<circle cx=\"{}\" cy=\"{}\" r=\"{}\"{}/>",
                self.num(center[0]),
                self.num(center[1]),
                self.num(*radius),
                self.style(style)
            ),
            SvgNode::Rect { origin, size, corner_radius, style } => {
                // a negative size is drawn the other way round
                let x = origin[0].min(origin[0] + size[0]);
                let y = origin[1].min(origin[1] + size[1]);
                let rx = if *corner_radius > 0.0 { format!(" rx=\"{}\"", self.num(*corner_radius)) } else { String::new() };
                format!(
                    "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"{rx}{}/>",
                    self.num(x),
                    self.num(y),
                    self.num(size[0].abs()),
                    self.num(size[1].abs()),
                    self.style(style)
                )
            }
            SvgNode::Line { start, end, style } => format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"{}/>",
                self.num(start[0]),
                self.num(start[1]),
                self.num(end[0]),
                self.num(end[1]),
                self.style(style)
            ),
            SvgNode::Polyline { points, closed, style } => {
                let list: Vec<String> = points.iter().map(|p| self.pt(*p)).collect();
                let tag = if *closed { "polygon" } else { "polyline" };
                format!("<{tag} points=\"{}\"{}/>", list.join(" "), self.style(style))
            }
            SvgNode::Text { pos, text, size, anchor, rotation, font_family, style } => {
                let anchor = match anchor {
                    SvgTextAnchor::Start => "",
                    SvgTextAnchor::Middle => " text-anchor=\"middle\"",
                    SvgTextAnchor::End => " text-anchor=\"end\"",
                };
                let family = font_family.as_ref().map(|f| format!(" font-family=\"{}\"", xml_escape(f, true))).unwrap_or_default();
                let rotate = if *rotation != 0.0 {
                    format!(" transform=\"rotate({} {} {})\"", self.num(*rotation), self.num(pos[0]), self.num(pos[1]))
                } else {
                    String::new()
                };
                format!(
                    "<text x=\"{}\" y=\"{}\" font-size=\"{}\"{family}{anchor}{rotate}{}>{}</text>",
                    self.num(pos[0]),
                    self.num(pos[1]),
                    self.num(*size),
                    self.style(style),
                    xml_escape(text, false)
                )
            }
            SvgNode::Image { origin, size, image, style } => {
                let png = crate::utils::png::encode_png(image.width, image.height, &image.rgba);
                format!(
                    "<image x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" preserveAspectRatio=\"none\"{} href=\"data:image/png;base64,{}\"/>",
                    self.num(origin[0]),
                    self.num(origin[1]),
                    self.num(size[0]),
                    self.num(size[1]),
                    self.style(style),
                    base64(&png)
                )
            }
            SvgNode::Group { id, transform, style, children } => {
                let id = id.as_ref().map(|i| format!(" id=\"{}\"", xml_escape(i, true))).unwrap_or_default();
                let tf = transform
                    .map(|t| {
                        let v: Vec<String> = t.iter().map(|x| self.num(*x)).collect();
                        format!(" transform=\"matrix({})\"", v.join(" "))
                    })
                    .unwrap_or_default();
                let _ = writeln!(self.out, "{indent}<g{id}{tf}{}>", self.style(style));

                for c in children {
                    self.node(c, depth + 1);
                }

                let _ = writeln!(self.out, "{indent}</g>");
                return;
            }
        };

        let _ = writeln!(self.out, "{indent}{line}");
    }
}

/// The document as SVG 1.1 text.
pub fn write_svg(doc: &SvgDocument) -> String {
    let mut w = Writer { out: String::new(), precision: doc.precision };
    let vb = doc.effective_view_box();
    let unit = doc.unit.suffix();

    w.out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    let _ = writeln!(
        w.out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"{}{unit}\" height=\"{}{unit}\" viewBox=\"{} {} {} {}\">",
        w.num(doc.width),
        w.num(doc.height),
        w.num(vb[0]),
        w.num(vb[1]),
        w.num(vb[2]),
        w.num(vb[3])
    );

    if let Some(t) = &doc.title {
        let _ = writeln!(w.out, "  <title>{}</title>", xml_escape(t, false));
    }

    if let Some(d) = &doc.description {
        let _ = writeln!(w.out, "  <desc>{}</desc>", xml_escape(d, false));
    }

    if !doc.gradients.is_empty() || !doc.clip_paths.is_empty() {
        w.out.push_str("  <defs>\n");
        for (k, g) in doc.gradients.iter().enumerate() {
            let tf = g
                .transform
                .map(|t| format!(" gradientTransform=\"matrix({})\"", t.iter().map(|x| w.num(*x)).collect::<Vec<_>>().join(" ")))
                .unwrap_or_default();
            let (tag, geometry) = match &g.kind {
                SvgGradientKind::Linear { start, end } => (
                    "linearGradient",
                    format!("x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"", w.num(start[0]), w.num(start[1]), w.num(end[0]), w.num(end[1])),
                ),
                SvgGradientKind::Radial { center, radius, focus } => {
                    let f = focus.map(|f| format!(" fx=\"{}\" fy=\"{}\"", w.num(f[0]), w.num(f[1]))).unwrap_or_default();
                    ("radialGradient", format!("cx=\"{}\" cy=\"{}\" r=\"{}\"{f}", w.num(center[0]), w.num(center[1]), w.num(*radius)))
                }
            };
            let _ = writeln!(w.out, "    <{tag} id=\"gradient{k}\" gradientUnits=\"userSpaceOnUse\" {geometry}{tf}>");
            for st in &g.stops {
                let [r, gg, b] = st.color;
                let op = if st.opacity < 1.0 { format!(" stop-opacity=\"{}\"", w.num(st.opacity)) } else { String::new() };
                let _ = writeln!(w.out, "      <stop offset=\"{}\" stop-color=\"#{r:02X}{gg:02X}{b:02X}\"{op}/>", w.num(st.offset));
            }
            let _ = writeln!(w.out, "    </{tag}>");
        }
        for (k, c) in doc.clip_paths.iter().enumerate() {
            let rule = match c.rule {
                SvgFillRule::NonZero => "",
                SvgFillRule::EvenOdd => " clip-rule=\"evenodd\"",
            };
            let _ = writeln!(w.out, "    <clipPath id=\"clip{k}\"><path d=\"{}\"{rule}/></clipPath>", w.path_data(&c.path));
        }
        w.out.push_str("  </defs>\n");
    }

    if doc.y_up {
        // y' = (top + bottom) - y
        let flip = [1.0, 0.0, 0.0, -1.0, 0.0, 2.0 * vb[1] + vb[3]];
        let wrapper = SvgNode::Group { id: None, transform: Some(flip), style: SvgStyle::default(), children: doc.children.clone() };
        w.node(&wrapper, 1);
    } else {
        for c in &doc.children {
            w.node(c, 1);
        }
    }

    w.out.push_str("</svg>\n");
    w.out
}

/// Standard base64 with padding, for data URIs.
pub(crate) fn base64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            if k <= c.len() {
                s.push(A[(n >> (18 - 6 * k) & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    #[test]
    fn path_data_reads_every_command_absolute() {
        use PathCommand::*;
        let p = SvgPath::from_data("M10 20 l5,0 5 5 H40v-10 c1 2 3 4 5 6 s 1 1 2 2 Q50 50 60 60 t10 0 a5 5 0 011 1 z m1 1 1 1").unwrap();
        assert_eq!(
            p.commands,
            vec![
                MoveTo([10.0, 20.0]),
                LineTo([15.0, 20.0]),
                LineTo([20.0, 25.0]),
                LineTo([40.0, 25.0]),
                LineTo([40.0, 15.0]),
                CubicTo { c1: [41.0, 17.0], c2: [43.0, 19.0], to: [45.0, 21.0] },
                // reflected about (45, 21): the last control (43, 19) gives (47, 23)
                CubicTo { c1: [47.0, 23.0], c2: [46.0, 22.0], to: [47.0, 23.0] },
                QuadTo { c: [50.0, 50.0], to: [60.0, 60.0] },
                QuadTo { c: [70.0, 70.0], to: [70.0, 60.0] },
                // "011 1": large-arc 0, sweep 1, then x 1 (flags need no separators)
                ArcTo { radii: [5.0, 5.0], rotation: 0.0, large_arc: false, sweep: true, to: [71.0, 61.0] },
                Close,
                MoveTo([11.0, 21.0]),
                LineTo([12.0, 22.0]),
            ]
        );
        // Numbers run together: an exponent, a sign, a second decimal point.
        assert_eq!(
            SvgPath::from_data("M1e2-3.5.5.5").unwrap().commands,
            vec![MoveTo([100.0, -3.5]), LineTo([0.5, 0.5])]
        );
        assert!(SvgPath::from_data("10 10").is_none(), "no command");
        assert!(SvgPath::from_data("M1 1 L2").is_none());
        assert!(SvgPath::from_data("M0 0 X1 1").is_none());
    }

    use super::*;

    #[test]
    fn numbers_drop_trailing_zeros_and_negative_zero() {
        let w = Writer { out: String::new(), precision: Some(4) };
        assert_eq!(w.num(1.5), "1.5");
        assert_eq!(w.num(2.0), "2");
        assert_eq!(w.num(-0.00001), "0");
        assert_eq!(w.num(-0.0), "0");
        assert_eq!(w.num(1.23456), "1.2346");
        let w = Writer { out: String::new(), precision: None };
        assert_eq!(w.num(0.1 + 0.2), "0.30000000000000004");
    }

    #[test]
    fn header_and_page() {
        let mut doc = SvgDocument::new(210.0, 297.0, SvgUnit::Millimeters);
        doc.title = Some("a & b".into());
        let text = write_svg(&doc);
        assert!(text.contains("width=\"210mm\" height=\"297mm\" viewBox=\"0 0 210 297\""));
        assert!(text.contains("<title>a &amp; b</title>"));
    }

    #[test]
    fn escapes_text_and_attributes() {
        assert_eq!(xml_escape("<a \"b\">", false), "&lt;a \"b\"&gt;");
        assert_eq!(xml_escape("<a \"b\">", true), "&lt;a &quot;b&quot;&gt;");
    }

    #[test]
    fn arcs_set_the_flags() {
        let mut p = SvgPath::new();
        p.arc([0.0, 0.0], 1.0, 0.0, 270.0);
        match &p.commands[1] {
            PathCommand::ArcTo { large_arc, sweep, to, .. } => {
                assert!(*large_arc && *sweep);
                assert!((to[0]).abs() < 1e-12 && (to[1] + 1.0).abs() < 1e-12);
            }
            c => panic!("expected an arc, got {c:?}"),
        }

        // a full turn is two half turns
        let mut p = SvgPath::new();
        p.arc([0.0, 0.0], 1.0, 0.0, -360.0);
        assert_eq!(p.commands.len(), 3);
        assert!(p.commands[1..].iter().all(|c| matches!(c, PathCommand::ArcTo { large_arc: false, sweep: false, .. })));
    }

    #[test]
    fn styles_write_only_what_is_set() {
        let mut doc = SvgDocument::new(10.0, 10.0, SvgUnit::None);
        doc.circle([5.0, 5.0], 2.0, SvgStyle::fill([255, 0, 16]).with_opacity(0.5));
        doc.line([0.0, 0.0], [10.0, 10.0], SvgStyle::stroke([0, 0, 0], 0.1).with_dash(&[1.0, 0.5]));
        doc.circle([1.0, 1.0], 1.0, SvgStyle::default());
        let text = write_svg(&doc);
        assert!(text.contains("<circle cx=\"5\" cy=\"5\" r=\"2\" fill=\"#FF0010\" fill-opacity=\"0.5\" stroke=\"none\" stroke-opacity=\"0.5\"/>"));
        assert!(text.contains("stroke-dasharray=\"1,0.5\""));
        assert!(text.contains("<circle cx=\"1\" cy=\"1\" r=\"1\"/>"));
    }

    #[test]
    fn groups_nest_and_y_up_flips() {
        let mut doc = SvgDocument::new(100.0, 50.0, SvgUnit::Millimeters);
        doc.y_up = true;
        let g = doc.group(Some("layer F.Cu"), None, SvgStyle::fill([200, 52, 52]));
        g.push(SvgNode::Polyline { points: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]], closed: true, style: SvgStyle::default() });
        let text = write_svg(&doc);
        assert!(text.contains("<g transform=\"matrix(1 0 0 -1 0 50)\">"));
        assert!(text.contains("    <g id=\"layer F.Cu\" fill=\"#C83434\" stroke=\"none\">"));
        assert!(text.contains("      <polygon points=\"0,0 10,0 10,5\"/>"));
    }

    #[test]
    fn gradients_clips_opacity_and_images_go_in_defs_and_attributes() {
        let mut doc = SvgDocument::new(10.0, 10.0, SvgUnit::None);
        let paint = doc.gradient(SvgGradient {
            kind: SvgGradientKind::Linear { start: [0.0, 0.0], end: [10.0, 0.0] },
            stops: vec![SvgStop { offset: 0.0, color: [255, 0, 0], opacity: 1.0 }, SvgStop { offset: 1.0, color: [0, 0, 255], opacity: 0.5 }],
            transform: None,
        });
        let clip = doc.clip_path(SvgPath::polygon(&[[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]], true), SvgFillRule::EvenOdd);
        doc.rect([0.0, 0.0], [10.0, 10.0], 0.0, SvgStyle { fill: Some(paint), opacity: Some(0.5), clip: Some(clip), ..Default::default() });
        let img = crate::utils::png::PngImage { width: 1, height: 1, rgba: vec![1, 2, 3, 255] };
        doc.children.push(SvgNode::Image { origin: [1.0, 1.0], size: [2.0, 2.0], image: std::sync::Arc::new(img), style: SvgStyle::default() });
        let text = write_svg(&doc);
        assert!(text.contains("<linearGradient id=\"gradient0\" gradientUnits=\"userSpaceOnUse\" x1=\"0\" y1=\"0\" x2=\"10\" y2=\"0\">"), "{text}");
        assert!(text.contains("<stop offset=\"1\" stop-color=\"#0000FF\" stop-opacity=\"0.5\"/>"));
        assert!(text.contains("<clipPath id=\"clip0\"><path d=\"M0,0 L5,0 L5,5 Z\" clip-rule=\"evenodd\"/></clipPath>"));
        assert!(text.contains("fill=\"url(#gradient0)\" opacity=\"0.5\" clip-path=\"url(#clip0)\"/>"));
        assert!(text.contains("preserveAspectRatio=\"none\" href=\"data:image/png;base64,iVBORw0KGgo"));
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(base64(b"Ma"), "TWE=");
        assert_eq!(base64(b"M"), "TQ==");
    }

    #[test]
    fn negative_rectangles_are_normalised() {
        let mut doc = SvgDocument::new(10.0, 10.0, SvgUnit::None);
        doc.rect([5.0, 5.0], [-2.0, -3.0], 0.0, SvgStyle::default());
        assert!(write_svg(&doc).contains("<rect x=\"3\" y=\"2\" width=\"2\" height=\"3\"/>"));
    }

    #[test]
    fn text_is_escaped_and_rotated() {
        let mut doc = SvgDocument::new(10.0, 10.0, SvgUnit::None);
        doc.children.push(SvgNode::Text {
            pos: [1.0, 2.0],
            text: "R1 <5k>".into(),
            size: 1.5,
            anchor: SvgTextAnchor::Middle,
            rotation: 90.0,
            font_family: Some("sans-serif".into()),
            style: SvgStyle::default(),
        });
        let text = write_svg(&doc);
        assert!(text.contains(
            "<text x=\"1\" y=\"2\" font-size=\"1.5\" font-family=\"sans-serif\" text-anchor=\"middle\" transform=\"rotate(90 1 2)\">R1 &lt;5k&gt;</text>"
        ));
    }
}
