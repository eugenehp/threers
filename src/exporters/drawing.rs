//! An [`SvgDocument`] as the geometry it stands for, the one reading that
//! [`write_pdf`](super::write_pdf) and [`render_svg`](super::render_svg)
//! share, so that a drawing looks the same in all three formats.
//!
//! The reading is SVG 1.1's, of the text [`write_svg`](super::write_svg)
//! writes: styles cascade from groups (a field left `None` inherits, as an
//! attribute left out does), the initial values are SVG's (black fill, no
//! stroke, width 1, miter joins with a limit of 4, butt caps, nonzero), a
//! `y_up` drawing sits in the same flipping group, the view box is fitted
//! to the page `xMidYMid meet`, and every shape becomes moves, lines and
//! cubic Béziers in the order SVG draws it, which is where a dash pattern
//! starts. Arcs are converted as SVG's implementation notes (F.6) say,
//! radii scaled up when too small to reach.
//!
//! An element or group with an `opacity` below 1 or a `clip` is a layer:
//! [`Item::Push`] before it and [`Item::Pop`] after, the clip in the user
//! space of the element (its own transform applied, as `userSpaceOnUse`
//! has it). Neither is inherited. Opacity 0 skips the element; a gradient or
//! clip index the document does not have is no paint and no clip.

use super::svg::{PathCommand, SvgDocument, SvgFillRule, SvgLineCap, SvgLineJoin, SvgNode, SvgPaint, SvgPath, SvgStyle, SvgTextAnchor, SvgUnit};
use crate::utils::png::PngImage;

/// An affine transform `[a, b, c, d, e, f]` as [`SvgTransform`](super::SvgTransform):
/// x' = a·x + c·y + e, y' = b·x + d·y + f.
pub(crate) type Affine = [f64; 6];

pub(crate) const IDENTITY: Affine = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// `m` after `n`: `n` applied first.
pub(crate) fn compose(m: &Affine, n: &Affine) -> Affine {
    [
        m[0] * n[0] + m[2] * n[1],
        m[1] * n[0] + m[3] * n[1],
        m[0] * n[2] + m[2] * n[3],
        m[1] * n[2] + m[3] * n[3],
        m[0] * n[4] + m[2] * n[5] + m[4],
        m[1] * n[4] + m[3] * n[5] + m[5],
    ]
}

pub(crate) fn apply(m: &Affine, p: [f64; 2]) -> [f64; 2] {
    [m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5]]
}

/// `m`'s inverse, when it has one.
pub(crate) fn invert(m: &Affine) -> Option<Affine> {
    let det = m[0] * m[3] - m[1] * m[2];
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let (a, b, c, d) = (m[3] / det, -m[1] / det, -m[2] / det, m[0] / det);
    Some([a, b, c, d, -(a * m[4] + c * m[5]), -(b * m[4] + d * m[5])])
}

/// A gradient's stops as SVG reads them: offsets clamped to 0 to 1 and
/// each at least the one before, opacities clamped.
pub(crate) fn stops(stops: &[super::svg::SvgStop]) -> Vec<super::svg::SvgStop> {
    let mut last = 0.0f64;
    stops
        .iter()
        .map(|s| {
            last = s.offset.clamp(0.0, 1.0).max(last);
            super::svg::SvgStop { offset: last, color: s.color, opacity: s.opacity.clamp(0.0, 1.0) }
        })
        .collect()
}

/// A radial gradient's focus as SVG 1.1 places it: the centre when none is
/// given, moved onto the circle (just inside) when outside it.
pub(crate) fn radial_focus(center: [f64; 2], radius: f64, focus: Option<[f64; 2]>) -> [f64; 2] {
    let Some(f) = focus else { return center };
    let d = [f[0] - center[0], f[1] - center[1]];
    let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
    let max = radius * 0.999;
    if len > max && len > 0.0 {
        [center[0] + d[0] / len * max, center[1] + d[1] / len * max]
    } else {
        f
    }
}

/// How much `m` can stretch a length: the larger singular value.
pub(crate) fn max_scale(m: &Affine) -> f64 {
    let (a, b, c, d) = (m[0], m[1], m[2], m[3]);
    let s = a * a + b * b + c * c + d * d;
    let det = a * d - b * c;
    let disc = (s * s - 4.0 * det * det).max(0.0).sqrt();
    ((s + disc) / 2.0).sqrt()
}

/// One step of a path, in the coordinates of the shape it belongs to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Seg {
    Move([f64; 2]),
    Line([f64; 2]),
    Cubic([f64; 2], [f64; 2], [f64; 2]),
    Close,
}

/// What a shape is painted with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Paint {
    Color([u8; 3]),
    /// The document's gradient `k`.
    Gradient(usize),
}

/// A style with every property decided.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Resolved {
    pub fill: Option<Paint>,
    pub stroke: Option<Paint>,
    pub stroke_width: f64,
    pub fill_opacity: f64,
    pub stroke_opacity: f64,
    pub line_cap: SvgLineCap,
    pub line_join: SvgLineJoin,
    pub miter_limit: f64,
    /// Dash and gap lengths, an even number of them; empty for solid.
    pub dash: Vec<f64>,
    pub fill_rule: SvgFillRule,
}

impl Default for Resolved {
    /// SVG's initial values.
    fn default() -> Self {
        Resolved {
            fill: Some(Paint::Color([0, 0, 0])),
            stroke: None,
            stroke_width: 1.0,
            fill_opacity: 1.0,
            stroke_opacity: 1.0,
            line_cap: SvgLineCap::Butt,
            line_join: SvgLineJoin::Miter,
            miter_limit: 4.0,
            dash: Vec::new(),
            fill_rule: SvgFillRule::NonZero,
        }
    }
}

impl Resolved {
    /// `style` over `self`, which it inherits from, in a document of
    /// `gradients` gradients.
    pub fn with(&self, style: &SvgStyle, gradients: usize) -> Resolved {
        let paint = |p: Option<SvgPaint>, inherited: Option<Paint>| match p {
            None => inherited,
            Some(SvgPaint::None) => None,
            Some(SvgPaint::Color(c)) => Some(Paint::Color(c)),
            Some(SvgPaint::Gradient(k)) => (k < gradients).then_some(Paint::Gradient(k)),
        };
        let mut r = self.clone();
        r.fill = paint(style.fill, self.fill);
        r.stroke = paint(style.stroke, self.stroke);
        if let Some(w) = style.stroke_width {
            r.stroke_width = w;
        }
        if let Some(o) = style.fill_opacity {
            r.fill_opacity = o.clamp(0.0, 1.0);
        }
        if let Some(o) = style.stroke_opacity {
            r.stroke_opacity = o.clamp(0.0, 1.0);
        }
        if let Some(c) = style.line_cap {
            r.line_cap = c;
        }
        if let Some(j) = style.line_join {
            r.line_join = j;
        }
        if let Some(f) = style.fill_rule {
            r.fill_rule = f;
        }
        // SVG: a limit below 1 is an error, ignored here
        if let Some(m) = style.miter_limit.filter(|m| *m >= 1.0) {
            r.miter_limit = m;
        }
        if !style.dash.is_empty() {
            r.dash = dash_array(&style.dash);
        }
        r
    }

    /// Whether the stroke paints anything.
    pub fn strokes(&self) -> bool {
        self.stroke.is_some() && self.stroke_width > 0.0 && self.stroke_opacity > 0.0
    }

    pub fn fills(&self) -> bool {
        self.fill.is_some() && self.fill_opacity > 0.0
    }
}

/// `stroke-dasharray` as SVG takes it: a list with a negative value, or
/// summing to zero, draws solid; an odd list is repeated to make it even.
fn dash_array(d: &[f64]) -> Vec<f64> {
    if d.iter().any(|v| *v < 0.0 || !v.is_finite()) || d.iter().sum::<f64>() <= 0.0 {
        return Vec::new();
    }
    if d.len() % 2 == 1 {
        d.iter().chain(d.iter()).copied().collect()
    } else {
        d.to_vec()
    }
}

/// What [`walk`] hands over: a shape, a run of text or an image, with the
/// transform from its coordinates to the view box's and its style, and
/// where layers begin and end.
pub(crate) enum Item<'a> {
    /// A layer begins: what follows until its [`Item::Pop`] is composited
    /// as one, at `opacity`, inside `clip` (in `ctm`'s coordinates).
    Push { ctm: Affine, opacity: f64, clip: Option<(Vec<Seg>, SvgFillRule)> },
    Pop,
    /// RGBA pixels over the rectangle at `origin` of `size`, top row first.
    Image { ctm: Affine, origin: [f64; 2], size: [f64; 2], image: &'a PngImage },
    Shape { ctm: Affine, style: Resolved, segs: Vec<Seg> },
    Text {
        ctm: Affine,
        style: Resolved,
        pos: [f64; 2],
        text: &'a str,
        size: f64,
        anchor: SvgTextAnchor,
        rotation: f64,
        font_family: Option<&'a str>,
    },
}

/// Every shape and text of `doc` in painting order, in user units.
pub(crate) fn walk<'a>(doc: &'a SvgDocument, f: &mut dyn FnMut(Item<'a>)) {
    let cx = Context { gradients: doc.gradients.len(), clips: &doc.clip_paths };
    let root = Resolved::default();
    let ctm = if doc.y_up {
        // write_svg's wrapper: y' = (top + bottom) - y
        let vb = doc.effective_view_box();
        [1.0, 0.0, 0.0, -1.0, 0.0, 2.0 * vb[1] + vb[3]]
    } else {
        IDENTITY
    };
    for n in &doc.children {
        walk_node(&cx, n, &ctm, &root, f);
    }
}

/// What a node needs of its document.
struct Context<'a> {
    gradients: usize,
    clips: &'a [super::svg::SvgClip],
}

fn node_style(n: &SvgNode) -> &SvgStyle {
    match n {
        SvgNode::Path { style, .. }
        | SvgNode::Circle { style, .. }
        | SvgNode::Rect { style, .. }
        | SvgNode::Line { style, .. }
        | SvgNode::Polyline { style, .. }
        | SvgNode::Text { style, .. }
        | SvgNode::Image { style, .. }
        | SvgNode::Group { style, .. } => style,
    }
}

fn walk_node<'a>(cx: &Context<'a>, n: &'a SvgNode, ctm: &Affine, inherited: &Resolved, f: &mut dyn FnMut(Item<'a>)) {
    let own = node_style(n);
    let opacity = own.opacity.map_or(1.0, |o| o.clamp(0.0, 1.0));
    if opacity <= 0.0 {
        return;
    }
    // a layer's clip is in the element's own user space: a group's transform applies
    let inner = match n {
        SvgNode::Group { transform: Some(t), .. } => compose(ctm, t),
        _ => *ctm,
    };
    let clip = own.clip.and_then(|k| cx.clips.get(k)).map(|c| (path_segs(&c.path), c.rule));
    let layer = opacity < 1.0 || clip.is_some();
    if layer {
        f(Item::Push { ctm: inner, opacity, clip });
    }
    walk_content(cx, n, ctm, inherited, f);
    if layer {
        f(Item::Pop);
    }
}

fn walk_content<'a>(cx: &Context<'a>, n: &'a SvgNode, ctm: &Affine, inherited: &Resolved, f: &mut dyn FnMut(Item<'a>)) {
    match n {
        SvgNode::Group { transform, style, children, .. } => {
            let ctm = match transform {
                Some(t) => compose(ctm, t),
                None => *ctm,
            };
            let style = inherited.with(style, cx.gradients);
            for c in children {
                walk_node(cx, c, &ctm, &style, f);
            }
        }
        SvgNode::Text { pos, text, size, anchor, rotation, font_family, style } => {
            if *size <= 0.0 || text.is_empty() {
                return;
            }
            f(Item::Text {
                ctm: *ctm,
                style: inherited.with(style, cx.gradients),
                pos: *pos,
                text,
                size: *size,
                anchor: *anchor,
                rotation: *rotation,
                font_family: font_family.as_deref(),
            });
        }
        SvgNode::Image { origin, size, image, .. } => {
            if size[0] > 0.0 && size[1] > 0.0 && image.width > 0 && image.height > 0 {
                f(Item::Image { ctm: *ctm, origin: *origin, size: *size, image });
            }
        }
        other => {
            let (segs, style) = shape_segs(other);
            if !segs.is_empty() {
                f(Item::Shape { ctm: *ctm, style: inherited.with(style, cx.gradients), segs });
            }
        }
    }
}

/// A shape node as a path, as SVG defines each shape's equivalent path, and
/// its own style. Shapes SVG does not render (a zero radius or size) are
/// empty.
fn shape_segs(n: &SvgNode) -> (Vec<Seg>, &SvgStyle) {
    let mut s = Vec::new();
    match n {
        SvgNode::Path { path, style } => (path_segs(path), style),
        SvgNode::Circle { center, radius, style } => {
            if *radius > 0.0 {
                ellipse(*center, *radius, *radius, &mut s);
            }
            (s, style)
        }
        SvgNode::Rect { origin, size, corner_radius, style } => {
            let x = origin[0].min(origin[0] + size[0]);
            let y = origin[1].min(origin[1] + size[1]);
            let (w, h) = (size[0].abs(), size[1].abs());
            if w > 0.0 && h > 0.0 {
                rect(x, y, w, h, *corner_radius, &mut s);
            }
            (s, style)
        }
        SvgNode::Line { start, end, style } => {
            s.push(Seg::Move(*start));
            s.push(Seg::Line(*end));
            (s, style)
        }
        SvgNode::Polyline { points, closed, style } => {
            if let Some((first, rest)) = points.split_first() {
                s.push(Seg::Move(*first));
                s.extend(rest.iter().map(|p| Seg::Line(*p)));
                if *closed {
                    s.push(Seg::Close);
                }
            }
            (s, style)
        }
        SvgNode::Group { style, .. } | SvgNode::Text { style, .. } | SvgNode::Image { style, .. } => (s, style),
    }
}

/// A path's commands as segments.
fn path_segs(path: &SvgPath) -> Vec<Seg> {
    let mut s = Vec::new();
    let mut start = [0.0, 0.0];
    let mut cur: Option<[f64; 2]> = None;
    for c in &path.commands {
        match *c {
            PathCommand::MoveTo(p) => {
                s.push(Seg::Move(p));
                start = p;
                cur = Some(p);
            }
            // path data must begin with a moveto; SVG renders up to the first error
            _ if cur.is_none() => break,
            PathCommand::LineTo(p) => {
                s.push(Seg::Line(p));
                cur = Some(p);
            }
            PathCommand::CubicTo { c1, c2, to } => {
                s.push(Seg::Cubic(c1, c2, to));
                cur = Some(to);
            }
            PathCommand::QuadTo { c, to } => {
                let p0 = cur.unwrap_or(start);
                let c1 = [p0[0] + 2.0 / 3.0 * (c[0] - p0[0]), p0[1] + 2.0 / 3.0 * (c[1] - p0[1])];
                let c2 = [to[0] + 2.0 / 3.0 * (c[0] - to[0]), to[1] + 2.0 / 3.0 * (c[1] - to[1])];
                s.push(Seg::Cubic(c1, c2, to));
                cur = Some(to);
            }
            PathCommand::ArcTo { radii, rotation, large_arc, sweep, to } => {
                arc_to_cubics(cur.unwrap_or(start), radii, rotation, large_arc, sweep, to, &mut s);
                cur = Some(to);
            }
            PathCommand::Close => {
                s.push(Seg::Close);
                cur = Some(start);
            }
        }
    }
    s
}

/// Bézier circles: the control distance that puts each quarter's midpoint on
/// the circle.
const KAPPA: f64 = 0.552_284_749_830_793_4;

/// An ellipse as SVG's equivalent path: from (cx + rx, cy), toward +y first.
fn ellipse(c: [f64; 2], rx: f64, ry: f64, s: &mut Vec<Seg>) {
    let (kx, ky) = (rx * KAPPA, ry * KAPPA);
    let [x, y] = c;
    s.push(Seg::Move([x + rx, y]));
    s.push(Seg::Cubic([x + rx, y + ky], [x + kx, y + ry], [x, y + ry]));
    s.push(Seg::Cubic([x - kx, y + ry], [x - rx, y + ky], [x - rx, y]));
    s.push(Seg::Cubic([x - rx, y - ky], [x - kx, y - ry], [x, y - ry]));
    s.push(Seg::Cubic([x + kx, y - ry], [x + rx, y - ky], [x + rx, y]));
    s.push(Seg::Close);
}

/// A rectangle as SVG's equivalent path: clockwise on the page from the top
/// edge, corners rounded by `r` clamped to half of each side (`ry` taken
/// from `rx`, as the writer leaves it out).
fn rect(x: f64, y: f64, w: f64, h: f64, r: f64, s: &mut Vec<Seg>) {
    let (rx, ry) = if r > 0.0 { (r.min(w / 2.0), r.min(h / 2.0)) } else { (0.0, 0.0) };
    if rx == 0.0 {
        s.push(Seg::Move([x, y]));
        s.push(Seg::Line([x + w, y]));
        s.push(Seg::Line([x + w, y + h]));
        s.push(Seg::Line([x, y + h]));
        s.push(Seg::Close);
        return;
    }
    let (kx, ky) = (rx * KAPPA, ry * KAPPA);
    s.push(Seg::Move([x + rx, y]));
    s.push(Seg::Line([x + w - rx, y]));
    s.push(Seg::Cubic([x + w - rx + kx, y], [x + w, y + ry - ky], [x + w, y + ry]));
    s.push(Seg::Line([x + w, y + h - ry]));
    s.push(Seg::Cubic([x + w, y + h - ry + ky], [x + w - rx + kx, y + h], [x + w - rx, y + h]));
    s.push(Seg::Line([x + rx, y + h]));
    s.push(Seg::Cubic([x + rx - kx, y + h], [x, y + h - ry + ky], [x, y + h - ry]));
    s.push(Seg::Line([x, y + ry]));
    s.push(Seg::Cubic([x, y + ry - ky], [x + rx - kx, y], [x + rx, y]));
    s.push(Seg::Close);
}

/// SVG's elliptical arc from `p0` as cubics, at most a quarter turn each
/// (SVG 1.1 F.6.5 and F.6.6). Coincident ends draw nothing; a zero radius
/// draws a line.
pub(crate) fn arc_to_cubics(p0: [f64; 2], radii: [f64; 2], rotation: f64, large_arc: bool, sweep: bool, p1: [f64; 2], s: &mut Vec<Seg>) {
    if p0 == p1 {
        return;
    }
    let (mut rx, mut ry) = (radii[0].abs(), radii[1].abs());
    if rx == 0.0 || ry == 0.0 {
        s.push(Seg::Line(p1));
        return;
    }
    let phi = rotation.to_radians();
    let (sin, cos) = phi.sin_cos();
    let dx2 = (p0[0] - p1[0]) / 2.0;
    let dy2 = (p0[1] - p1[1]) / 2.0;
    let x1p = cos * dx2 + sin * dy2;
    let y1p = -sin * dx2 + cos * dy2;
    let lambda = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
    if lambda > 1.0 {
        let k = lambda.sqrt();
        rx *= k;
        ry *= k;
    }
    let num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p;
    let den = rx * rx * y1p * y1p + ry * ry * x1p * x1p;
    let mut coef = if den > 0.0 { (num / den).max(0.0).sqrt() } else { 0.0 };
    if large_arc == sweep {
        coef = -coef;
    }
    let cxp = coef * rx * y1p / ry;
    let cyp = -coef * ry * x1p / rx;
    let cx = cos * cxp - sin * cyp + (p0[0] + p1[0]) / 2.0;
    let cy = sin * cxp + cos * cyp + (p0[1] + p1[1]) / 2.0;
    let angle = |ux: f64, uy: f64, vx: f64, vy: f64| (ux * vy - uy * vx).atan2(ux * vx + uy * vy);
    let (ux, uy) = ((x1p - cxp) / rx, (y1p - cyp) / ry);
    let (vx, vy) = ((-x1p - cxp) / rx, (-y1p - cyp) / ry);
    let theta1 = angle(1.0, 0.0, ux, uy);
    let mut dtheta = angle(ux, uy, vx, vy);
    let tau = std::f64::consts::TAU;
    if !sweep && dtheta > 0.0 {
        dtheta -= tau;
    } else if sweep && dtheta < 0.0 {
        dtheta += tau;
    }
    let n = (dtheta.abs() / (tau / 4.0) - 1e-9).ceil().max(1.0) as usize;
    let delta = dtheta / n as f64;
    let t = 4.0 / 3.0 * (delta / 4.0).tan();
    let map = |x: f64, y: f64| [cx + rx * cos * x - ry * sin * y, cy + rx * sin * x + ry * cos * y];
    let mut a1 = theta1;
    for k in 0..n {
        let a2 = a1 + delta;
        let (s1, c1) = a1.sin_cos();
        let (s2, c2) = a2.sin_cos();
        let q1 = map(c1 - t * s1, s1 + t * c1);
        let q2 = map(c2 + t * s2, s2 - t * c2);
        let end = if k + 1 == n { p1 } else { map(c2, s2) };
        s.push(Seg::Cubic(q1, q2, end));
        a1 = a2;
    }
}

/// The page's size in inches.
pub(crate) fn page_inches(doc: &SvgDocument) -> [f64; 2] {
    let per_inch = match doc.unit {
        SvgUnit::None | SvgUnit::Pixels => 96.0,
        SvgUnit::Millimeters => 25.4,
        SvgUnit::Centimeters => 2.54,
        SvgUnit::Inches => 1.0,
        SvgUnit::Points => 72.0,
    };
    [doc.width / per_inch, doc.height / per_inch]
}

/// The view box fitted into a `size` viewport with y down, `xMidYMid meet`:
/// uniformly scaled to fit and centred. `None` when the view box is empty,
/// which disables rendering.
pub(crate) fn viewport(doc: &SvgDocument, size: [f64; 2]) -> Option<Affine> {
    let vb = doc.effective_view_box();
    if !(vb[2] > 0.0 && vb[3] > 0.0) {
        return None;
    }
    let s = (size[0] / vb[2]).min(size[1] / vb[3]);
    let tx = (size[0] - vb[2] * s) / 2.0 - vb[0] * s;
    let ty = (size[1] - vb[3] * s) / 2.0 - vb[1] * s;
    Some([s, 0.0, 0.0, s, tx, ty])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exporters::{SvgPath, SvgStyle};

    fn end(s: &[Seg]) -> [f64; 2] {
        match s.last().unwrap() {
            Seg::Cubic(_, _, p) | Seg::Line(p) | Seg::Move(p) => *p,
            Seg::Close => panic!("ends in close"),
        }
    }

    #[test]
    fn arcs_split_into_quarter_turns_and_end_exactly() {
        let mut s = Vec::new();
        arc_to_cubics([1.0, 0.0], [1.0, 1.0], 0.0, true, true, [0.0, -1.0], &mut s);
        assert_eq!(s.len(), 3, "a three-quarter turn");
        assert_eq!(end(&s), [0.0, -1.0]);
        // the first quarter ends at (0, 1): sweep is the +angle direction
        match s[0] {
            Seg::Cubic(_, _, p) => assert!((p[0]).abs() < 1e-12 && (p[1] - 1.0).abs() < 1e-12, "{p:?}"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn arcs_too_small_to_reach_are_scaled_up() {
        let mut s = Vec::new();
        arc_to_cubics([0.0, 0.0], [0.1, 0.1], 0.0, false, true, [2.0, 0.0], &mut s);
        // a half circle of radius 1 about (1, 0)
        assert_eq!(s.len(), 2);
        match s[0] {
            Seg::Cubic(_, _, p) => assert!((p[0] - 1.0).abs() < 1e-9 && (p[1].abs() - 1.0).abs() < 1e-9, "{p:?}"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn styles_inherit_and_none_turns_paint_off() {
        let group = Resolved::default().with(&SvgStyle::stroke([1, 2, 3], 0.5).with_dash(&[1.0]), 0);
        assert_eq!(group.fill, None);
        assert_eq!(group.stroke, Some(Paint::Color([1, 2, 3])));
        assert_eq!(group.dash, vec![1.0, 1.0], "an odd dash list is repeated");
        let child = group.with(&SvgStyle { stroke_width: Some(2.0), ..Default::default() }, 0);
        assert_eq!((child.stroke, child.stroke_width, child.line_cap), (Some(Paint::Color([1, 2, 3])), 2.0, SvgLineCap::Round));
        assert!(Resolved::default().with(&SvgStyle::default().with_dash(&[1.0, -1.0]), 0).dash.is_empty());
        // a gradient the document does not have paints nothing
        let missing = SvgStyle { fill: Some(SvgPaint::Gradient(3)), ..Default::default() };
        assert_eq!(Resolved::default().with(&missing, 3).fill, None);
        assert_eq!(Resolved::default().with(&missing, 4).fill, Some(Paint::Gradient(3)));
    }

    #[test]
    fn y_up_wraps_the_drawing_in_the_writers_flip() {
        let mut doc = SvgDocument::new(100.0, 50.0, SvgUnit::Millimeters);
        doc.y_up = true;
        doc.line([0.0, 0.0], [10.0, 0.0], SvgStyle::default());
        let mut got = None;
        walk(&doc, &mut |i| {
            if let Item::Shape { ctm, .. } = i {
                got = Some(ctm);
            }
        });
        assert_eq!(got, Some([1.0, 0.0, 0.0, -1.0, 0.0, 50.0]));
    }

    #[test]
    fn opacity_and_clips_make_layers_in_the_elements_own_space() {
        let mut doc = SvgDocument::new(10.0, 10.0, SvgUnit::None);
        let clip = doc.clip_path(SvgPath::polygon(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], true), SvgFillRule::NonZero);
        let g = doc.group(None, Some([2.0, 0.0, 0.0, 2.0, 0.0, 0.0]), SvgStyle { opacity: Some(0.5), clip: Some(clip), ..Default::default() });
        g.push(SvgNode::Circle { center: [1.0, 1.0], radius: 1.0, style: SvgStyle::default() });
        doc.circle([1.0, 1.0], 1.0, SvgStyle { opacity: Some(0.0), ..Default::default() });
        let mut seen = Vec::new();
        walk(&doc, &mut |i| {
            seen.push(match i {
                Item::Push { ctm, opacity, clip } => format!("push {ctm:?} {opacity} {}", clip.is_some()),
                Item::Pop => "pop".into(),
                Item::Shape { .. } => "shape".into(),
                _ => "other".into(),
            })
        });
        assert_eq!(seen, ["push [2.0, 0.0, 0.0, 2.0, 0.0, 0.0] 0.5 true", "shape", "pop"]);
    }

    #[test]
    fn paths_must_start_with_a_move() {
        let mut doc = SvgDocument::new(10.0, 10.0, SvgUnit::None);
        let mut p = SvgPath::new();
        p.line_to([1.0, 1.0]);
        doc.path(p, SvgStyle::default());
        let mut n = 0;
        walk(&doc, &mut |_| n += 1);
        assert_eq!(n, 0);
    }

    #[test]
    fn view_boxes_fit_and_centre() {
        let mut doc = SvgDocument::new(200.0, 100.0, SvgUnit::None);
        doc.view_box = Some([0.0, 0.0, 10.0, 10.0]);
        assert_eq!(viewport(&doc, [200.0, 100.0]), Some([10.0, 0.0, 0.0, 10.0, 50.0, 0.0]));
        doc.view_box = Some([0.0, 0.0, 0.0, 10.0]);
        assert_eq!(viewport(&doc, [200.0, 100.0]), None);
    }
}
