//! Painting a laid-out [`Ui`]: backgrounds, borders, text and paths, in tree
//! order, through a [`Painter`].

use super::layout::TextMeasure;
use super::style::{Display, FontWeight, Rgba, Style};
use super::tree::{NodeId, NodeKind, Rect, Ui};

/// The font of a text run, resolved from its node's style.
#[derive(Debug, Clone, PartialEq)]
pub struct TextStyle {
    pub font_family: String,
    pub font_size: f32,
    pub font_weight: FontWeight,
    /// Non-premultiplied sRGB, alpha already multiplied by the opacity of the
    /// node and its ancestors.
    pub color: Rgba,
    /// Struck through (`text-decoration-line: line-through`).
    pub line_through: bool,
    /// Turned a quarter clockwise (`writing-mode: vertical-lr`); `line` is
    /// then the run's box, taller than wide.
    pub vertical: bool,
    pub italic: bool,
}

impl TextStyle {
    pub fn of(style: &Style) -> Self {
        Self {
            font_family: style.font_family.clone(),
            font_size: style.font_size,
            font_weight: style.font_weight,
            color: style.color,
            line_through: style.line_through,
            vertical: style.vertical,
            italic: style.italic,
        }
    }

    /// The weight as a CSS number (`normal` 400, `bold` 700).
    pub fn weight(&self) -> u16 {
        match self.font_weight {
            FontWeight::Normal => 400,
            FontWeight::Bold => 700,
            FontWeight::Weight(w) => w,
        }
    }

    /// A CSS `font` shorthand, as a 2D canvas takes it.
    pub fn css_font(&self) -> String {
        let style = if self.italic { "italic " } else { "" };
        format!(
            "{style}{} {}px {}",
            self.weight(),
            self.font_size,
            self.font_family
        )
    }
}

/// Where a [`Ui`] is drawn. Colours are non-premultiplied sRGB in `0..=1`;
/// coordinates are pixels from the top-left.
///
/// A painter also measures text, so that layout and painting agree on every
/// string's width.
/// A circular arc: its centre and radius, where it starts and how far round
/// it goes (radians, clockwise from straight up on screen; a sweep of a
/// whole turn or more is the whole ring). Painters stroke it with
/// [`Painter::stroke_arc`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArcShape {
    pub cx: f32,
    pub cy: f32,
    pub radius: f32,
    pub start: f32,
    pub sweep: f32,
}

impl ArcShape {
    /// The box round its whole circle, `reach` wider each way.
    pub fn bounds(&self, reach: f32) -> Rect {
        let r = self.radius + reach;
        Rect {
            x: self.cx - r,
            y: self.cy - r,
            width: 2.0 * r,
            height: 2.0 * r,
        }
    }

    /// How far `(x, y)` is from the arc stroked `width` wide with round
    /// ends: negative inside, in the same units. A distance field, so a
    /// pixel's coverage is `0.5 - distance` (clamped) at any size.
    pub fn distance(&self, width: f32, x: f32, y: f32) -> f32 {
        let (px, py) = (x - self.cx, y - self.cy);
        let half = width / 2.0;
        if self.sweep.abs() >= std::f32::consts::TAU - 1e-4 {
            return ((px * px + py * py).sqrt() - self.radius).abs() - half;
        }
        // Turned so the arc's middle points along +y: then it is symmetric
        // about the y axis, and only |x| matters.
        let (start, sweep) = if self.sweep < 0.0 {
            (self.start + self.sweep, -self.sweep)
        } else {
            (self.start, self.sweep)
        };
        let mid = start + sweep / 2.0;
        let (ms, mc) = mid.sin_cos();
        // Along the middle (screen direction (sin, -cos)), and across it.
        let along = px * ms - py * mc;
        let across = (px * mc + py * ms).abs();
        let (hs, hc) = (sweep / 2.0).sin_cos();
        if hc * across > hs * along {
            // Past an end: from the end's point (a round cap).
            let (ex, ey) = (across - hs * self.radius, along - hc * self.radius);
            (ex * ex + ey * ey).sqrt() - half
        } else {
            ((across * across + along * along).sqrt() - self.radius).abs() - half
        }
    }

    /// The arc that SVG path data `d` draws, if it draws one: a move, then
    /// one or more circular arcs (`A r r 0 large sweep x y`, absolute, one
    /// after another round the same circle). A gauge's dial and a ring
    /// round an icon are such paths.
    pub fn from_svg(d: &str) -> Option<ArcShape> {
        let mut tokens = svg_tokens(d).into_iter();
        if tokens.next()? != SvgToken::Command('M') {
            return None;
        }
        let num = |t: &mut std::vec::IntoIter<SvgToken>| match t.next() {
            Some(SvgToken::Number(n)) => Some(n),
            _ => None,
        };
        let (mut x, mut y) = (num(&mut tokens)?, num(&mut tokens)?);
        let mut out: Option<ArcShape> = None;
        let mut arc_cmd = false;
        loop {
            match tokens.next() {
                None => break,
                Some(SvgToken::Command('A')) => arc_cmd = true,
                Some(SvgToken::Command(_)) => return None,
                Some(SvgToken::Number(rx)) if arc_cmd => {
                    let (ry, rot, large, sweep, x1, y1) = (
                        num(&mut tokens)?,
                        num(&mut tokens)?,
                        num(&mut tokens)?,
                        num(&mut tokens)?,
                        num(&mut tokens)?,
                        num(&mut tokens)?,
                    );
                    if (rx - ry).abs() > 1e-3 * rx.abs().max(1.0) || rot != 0.0 || rx <= 0.0 {
                        return None;
                    }
                    let a = circle_arc((x, y), (x1, y1), rx, large != 0.0, sweep != 0.0)?;
                    out = Some(match out {
                        None => a,
                        Some(o) => {
                            let tol = 1e-2 * o.radius.max(1.0);
                            let end = o.start + o.sweep;
                            let joins = (o.cx - a.cx).abs() < tol
                                && (o.cy - a.cy).abs() < tol
                                && (o.radius - a.radius).abs() < tol
                                && same_angle(end, a.start);
                            if !joins || o.sweep.signum() != a.sweep.signum() {
                                return None;
                            }
                            ArcShape {
                                sweep: o.sweep + a.sweep,
                                ..o
                            }
                        }
                    });
                    (x, y) = (x1, y1);
                }
                Some(SvgToken::Number(_)) => return None,
            }
        }
        out
    }

    /// Mapped from a view box to a box: scaled by `(sx, sy)` (alike, or it
    /// is no longer a circle) after moving its origin to `(ox, oy)`.
    pub fn mapped(&self, ox: f32, oy: f32, sx: f32, sy: f32, to: (f32, f32)) -> Option<ArcShape> {
        if (sx - sy).abs() > 1e-3 * sx.abs().max(1e-6) {
            return None;
        }
        Some(ArcShape {
            cx: to.0 + (self.cx - ox) * sx,
            cy: to.1 + (self.cy - oy) * sy,
            radius: self.radius * sx,
            ..*self
        })
    }
}

fn same_angle(a: f32, b: f32) -> bool {
    let d = (a - b).rem_euclid(std::f32::consts::TAU);
    d < 1e-3 || d > std::f32::consts::TAU - 1e-3
}

/// The circle arc from `p0` to `p1` of radius `r` (SVG's endpoint form):
/// the centre on the side `large` and `sweep` say; the radius grown if the
/// ends are further apart than it allows, as SVG does.
fn circle_arc(
    p0: (f32, f32),
    p1: (f32, f32),
    r: f32,
    large: bool,
    sweep: bool,
) -> Option<ArcShape> {
    let (dx, dy) = (p1.0 - p0.0, p1.1 - p0.1);
    let chord = (dx * dx + dy * dy).sqrt();
    if chord < 1e-6 {
        return None;
    }
    let r = r.max(chord / 2.0);
    let h = (r * r - chord * chord / 4.0).max(0.0).sqrt();
    let (mx, my) = ((p0.0 + p1.0) / 2.0, (p0.1 + p1.1) / 2.0);
    // The centre is to the right of the chord's direction (on screen, y
    // down) for a clockwise sweep that is the short way round.
    let (nx, ny) = (-dy / chord, dx / chord);
    let side = if large != sweep { 1.0 } else { -1.0 };
    let (cx, cy) = (mx + side * h * nx, my + side * h * ny);
    let angle = |(x, y): (f32, f32)| (x - cx).atan2(-(y - cy));
    let (a0, a1) = (angle(p0), angle(p1));
    let mut delta = a1 - a0;
    if sweep {
        if delta <= 0.0 {
            delta += std::f32::consts::TAU;
        }
    } else if delta >= 0.0 {
        delta -= std::f32::consts::TAU;
    }
    Some(ArcShape {
        cx,
        cy,
        radius: r,
        start: a0,
        sweep: delta,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SvgToken {
    Command(char),
    Number(f32),
}

/// SVG path data as commands and numbers.
fn svg_tokens(d: &str) -> Vec<SvgToken> {
    let mut out = vec![];
    let b = d.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_ascii_alphabetic() && c != 'e' && c != 'E' {
            out.push(SvgToken::Command(c));
            i += 1;
        } else if c.is_ascii_digit() || c == '-' || c == '+' || c == '.' {
            let start = i;
            i += 1;
            let mut dot = c == '.';
            while i < b.len() {
                let ch = b[i] as char;
                let exp_sign = (ch == '-' || ch == '+') && matches!(b[i - 1], b'e' | b'E');
                if ch.is_ascii_digit() || (ch == '.' && !dot) || ch == 'e' || ch == 'E' || exp_sign
                {
                    dot |= ch == '.';
                    i += 1;
                } else {
                    break;
                }
            }
            match d[start..i].parse::<f32>() {
                Ok(n) => out.push(SvgToken::Number(n)),
                Err(_) => return vec![],
            }
        } else {
            i += 1;
        }
    }
    out
}

pub trait Painter: TextMeasure {
    fn fill_rect(&mut self, rect: Rect, color: Rgba);

    /// A rectangle with elliptical corners of radii `rx` × `ry`
    /// (`border-radius`). Painters without curves may fill the plain rectangle.
    fn fill_rounded_rect(&mut self, rect: Rect, rx: f32, ry: f32, color: Rgba) {
        let _ = (rx, ry);
        self.fill_rect(rect, color);
    }

    /// A left-to-right linear gradient through `stops` (fractions of the
    /// width, ascending), blurred by `blur` px (Gaussian standard deviation).
    /// By default: bands of interpolated colour, unblurred.
    fn fill_linear_gradient(&mut self, rect: Rect, stops: &[(f32, Rgba)], blur: f32) {
        let _ = blur;
        for pair in stops.windows(2) {
            let ((t0, c0), (t1, c1)) = (pair[0], pair[1]);
            if t1 <= t0 {
                continue;
            }
            let steps = if c0 == c1 { 1 } else { 16 };
            for k in 0..steps {
                let (a, b) = (k as f32 / steps as f32, (k + 1) as f32 / steps as f32);
                let mid = (a + b) / 2.0;
                let c = std::array::from_fn(|i| c0[i] + (c1[i] - c0[i]) * mid);
                let x0 = rect.x + rect.width * (t0 + (t1 - t0) * a);
                let x1 = rect.x + rect.width * (t0 + (t1 - t0) * b);
                self.fill_rect(
                    Rect {
                        x: x0,
                        y: rect.y,
                        width: x1 - x0,
                        height: rect.height,
                    },
                    c,
                );
            }
        }
    }

    /// A gradient laid over `rect` as `shape` says (see
    /// [`GradientShape`](crate::GradientShape)). By default the left-to-right
    /// one is [`fill_linear_gradient`](Self::fill_linear_gradient), and the
    /// others are bands of their colours across, as close as a painter with
    /// only that can come.
    fn fill_gradient(
        &mut self,
        rect: Rect,
        shape: crate::GradientShape,
        stops: &[(f32, Rgba)],
        blur: f32,
    ) {
        if shape.is_across() {
            return self.fill_linear_gradient(rect, stops, blur);
        }
        let mid = (rect.x + rect.width / 2.0, rect.y + rect.height / 2.0);
        let at = |x: f32| crate::gradient_at(stops, shape.t(rect, x, mid.1));
        let across: Vec<(f32, Rgba)> = (0..=16)
            .map(|i| (i as f32 / 16.0, at(rect.x + rect.width * i as f32 / 16.0)))
            .collect();
        self.fill_linear_gradient(rect, &across, blur);
    }

    /// One line of text in the line box `line` (its top-left and height):
    /// vertically centred on the line box, as CSS places a glyph run.
    fn text(&mut self, line: Rect, text: &str, style: &TextStyle);

    /// SVG path data in `view_box` scaled to `rect`, stroked `stroke` pixels
    /// wide in view-box units and/or filled (non-zero), in `color`.
    fn path(
        &mut self,
        rect: Rect,
        d: &[String],
        view_box: [f32; 4],
        stroke: Option<f32>,
        fill: bool,
        color: Rgba,
    );

    /// `arc` stroked `width` pixels wide with round ends, in `color`. A
    /// painter that can should draw it from its distance field
    /// ([`ArcShape::distance`]): smooth at any size. By default it is a
    /// [`path`](Self::path) of a segment every pixel and a half or so.
    fn stroke_arc(&mut self, arc: ArcShape, width: f32, color: Rgba) {
        let turn = arc.sweep.abs().min(std::f32::consts::TAU);
        let n = ((arc.radius * turn / 1.5).ceil() as usize).clamp(8, 4096);
        let mut d = String::with_capacity(n * 16);
        for i in 0..=n {
            let a = arc.start + arc.sweep.signum() * turn * i as f32 / n as f32;
            let (x, y) = (arc.cx + arc.radius * a.sin(), arc.cy - arc.radius * a.cos());
            d.push_str(&format!("{}{x:.3} {y:.3} ", if i == 0 { 'M' } else { 'L' }));
        }
        let b = arc.bounds(0.0);
        self.path(
            b,
            &[d],
            [b.x, b.y, b.width.max(1e-6), b.height.max(1e-6)],
            Some(width),
            false,
            color,
        );
    }

    /// Restricts painting to `rect` until the matching
    /// [`pop_clip`](Self::pop_clip). Painters that cannot clip may ignore it.
    fn push_clip(&mut self, _rect: Rect) {}

    /// Like [`push_clip`](Self::push_clip), to a box with elliptical
    /// corners (`overflow: hidden` with `border-radius`). By default, the
    /// plain box.
    fn push_clip_rounded(&mut self, rect: Rect, rx: f32, ry: f32) {
        let _ = (rx, ry);
        self.push_clip(rect);
    }

    fn pop_clip(&mut self) {}

    /// Maps what is painted until the matching
    /// [`pop_transform`](Self::pop_transform) through the affine `m`
    /// (`[a, b, c, d, e, f]`: x' = a·x + c·y + e, y' = b·x + d·y + f), after
    /// any in force: a [3D effect](crate::depth) projected flat. Painters
    /// that cannot transform ignore it, and the effect shows flat.
    fn push_transform(&mut self, _m: [f32; 6]) {}

    /// What has been painted under `rect` (corners of `radius`), blurred by
    /// `blur` px (standard deviation) and its colour scaled by `saturate`,
    /// shown in the box: a material's backdrop, painted before the box's
    /// translucent background. Painters that cannot read back what they
    /// painted leave it out, and the background alone tints the box.
    fn backdrop(&mut self, _rect: Rect, _radius: f32, _blur: f32, _saturate: f32) {}

    /// [`push_transform`](Self::push_transform) for a box seen in true
    /// perspective: `h` maps it exactly (a 3 × 3 projective matrix, by rows),
    /// `affine` is the nearest 2D transform. By default, `affine`; a painter
    /// that can draw perspective (a GPU's) uses `h`. Ended by
    /// [`pop_transform`](Self::pop_transform).
    fn push_projective(&mut self, h: [f32; 9], affine: [f32; 6]) {
        let _ = h;
        self.push_transform(affine);
    }

    fn pop_transform(&mut self) {}

    /// Blurs what is painted until [`pop_blur`](Self::pop_blur) by `radius`
    /// pixels (a layer out of focus). Painters that cannot blur ignore it.
    fn push_blur(&mut self, _radius: f32) {}

    fn pop_blur(&mut self) {}

    /// What follows, until [`pop_effect`](Self::pop_effect), is the box at
    /// `rect` and its insides, to be drawn through `effect`'s shader (see
    /// [`shader`](crate::shader)) at `time` seconds. Painters that cannot
    /// run shaders ignore it and draw what follows as it is.
    fn push_effect(&mut self, _rect: Rect, _effect: &crate::shader::Effect, _time: f32) {}

    fn pop_effect(&mut self) {}

    /// Inside (`true`) or out of a [`Style::subpixel`] subtree: boxes there
    /// keep their fractional edges. Painters that always snap may ignore it.
    fn set_subpixel(&mut self, _on: bool) {}

    /// `image` scaled to fill `rect` (already fitted by `object-fit`; the
    /// clip trims what overflows the box), at `opacity`. Painters without
    /// images draw nothing.
    fn image(&mut self, _rect: Rect, _image: &crate::image::ImageRef, _opacity: f32) {}

    /// The renderer's texture `id` (see [`NodeKind::Surface`]) drawn to
    /// `rect` at `opacity`. Painters without such textures draw nothing.
    fn surface(&mut self, _rect: Rect, _id: u64, _opacity: f32) {}

    /// A box shadow: `rect` (the box moved and grown) with corner `radius`,
    /// blurred by `blur` (CSS's radius; a Gaussian of half that deviation).
    /// By default, the box itself, unblurred, which is better than nothing
    /// only for a sharp shadow.
    fn shadow(&mut self, rect: Rect, radius: f32, blur: f32, color: Rgba) {
        if blur <= 0.0 {
            if radius > 0.0 {
                self.fill_rounded_rect(rect, radius, radius, color);
            } else {
                self.fill_rect(rect, color);
            }
        }
    }

    /// A line under text, `thickness` thick and centred on `y`, from `x`
    /// for `width`: straight, or a wave of amplitude
    /// [`WAVE_AMPLITUDE`]` × thickness` and period [`WAVE_PERIOD`]` ×
    /// thickness`. By default the wave is drawn straight.
    fn underline(&mut self, x: f32, y: f32, width: f32, thickness: f32, color: Rgba, wavy: bool) {
        let _ = wavy;
        self.fill_rect(
            Rect {
                x,
                y: y - thickness / 2.0,
                width,
                height: thickness,
            },
            color,
        );
    }
}

/// A wavy underline's amplitude, in line thicknesses.
pub const WAVE_AMPLITUDE: f32 = 1.25;
/// A wavy underline's period, in line thicknesses.
pub const WAVE_PERIOD: f32 = 6.0;

/// Where a run's underline goes, and how thick it is, from its line box
/// and font size: a fixed share of the em below the centre, as fonts put
/// it near enough (this crate has no font metrics).
pub fn underline_position(line: Rect, font_size: f32) -> (f32, f32) {
    let thickness = (font_size / 14.0).max(1.0);
    (line.y + line.height / 2.0 + font_size * 0.4, thickness)
}

impl Ui {
    /// Paints the tree as last laid out: each box's background, then its
    /// borders, then its text or path, then its children in order.
    pub fn paint(&self, painter: &mut dyn Painter) {
        self.env.repaint.set(false);
        // What is outside the window (or a box that clips) is not drawn.
        let window = self.rect(self.root());
        self.paint_node(self.root(), painter, 1.0, false, window);
        // Deferred nodes last, outside their ancestors' clips, with the
        // opacity those ancestors would have given them.
        for d in self.deferred_order() {
            self.paint_node(d, painter, self.inherited_opacity(d), false, window);
        }
        self.paint_focus_ring(painter);
    }

    /// `id` and what is inside it, at `opacity` (a reflection's copy).
    pub(crate) fn paint_node_pub(&self, id: NodeId, painter: &mut dyn Painter, opacity: f32) {
        self.paint_node(
            id,
            painter,
            opacity / self.painted_opacity(id).max(1e-6),
            false,
            EVERYWHERE,
        );
    }

    /// Whether `id` can be left undrawn: nothing of it shows in `visible`
    /// (its box, its shadows' reach), and nothing inside it can show
    /// elsewhere (it has nothing inside, or it clips what it has).
    fn out_of_sight(&self, id: NodeId, s: &Style, visible: Rect) -> bool {
        if visible.width >= EVERYWHERE.width {
            return false;
        }
        let node = &self.nodes[id.0];
        if !(node.children.is_empty() || s.clip)
            || self.env.reflections.contains_key(&id)
            || self.env.effects.contains_key(&id)
            || self.transform3d(id).is_some()
        {
            return false;
        }
        let r = node.rect;
        let reach = s
            .box_shadow
            .iter()
            .map(|sh| sh.blur * 3.0 + sh.spread.max(0.0) + sh.x.abs().max(sh.y.abs()))
            .fold(1.0f32, f32::max);
        r.x - reach > visible.x + visible.width
            || r.y - reach > visible.y + visible.height
            || r.x + r.width + reach < visible.x
            || r.y + r.height + reach < visible.y
    }

    fn paint_node(
        &self,
        id: NodeId,
        painter: &mut dyn Painter,
        opacity: f32,
        subpixel: bool,
        visible: Rect,
    ) {
        let node = &self.nodes[id.0];
        // Text and shapes in the colour they inherit now, not the one they
        // were made with (see `Ui::color_of`).
        let inherited;
        let s = if matches!(node.kind, NodeKind::Text(_) | NodeKind::Path { .. })
            && self.color_of(id) != node.style.color
        {
            let mut st = node.style.clone();
            st.color = self.color_of(id);
            inherited = st;
            &inherited
        } else {
            &node.style
        };
        if s.display == Display::None {
            return;
        }

        let opacity = opacity * s.opacity * self.flicker_level(id);
        if opacity <= 0.0 {
            return;
        }
        if self.out_of_sight(id, s, visible) {
            return;
        }
        // In 3D: a face turned away is not drawn; a lifted box's shadow
        // goes under it, unturned; then it is drawn through its transform.
        if !self.faces_viewer(id) {
            return;
        }
        self.paint_lift_shadow(id, painter, opacity);
        let blur = self.depth_blur(id);
        if let Some(b) = blur {
            painter.push_blur(b);
        }
        let transform = self.paint_transform(id);
        if let Some((h, m)) = transform {
            painter.push_projective(h, m);
        }
        // In the box's own space, so an effect turns with it.
        let effect = self.push_effect(id, painter);
        // Cut to its shape: everything it draws, shadows too.
        let shape = s.clip_path.map(|c| c.place(node.rect));
        if let Some((r, rx, ry)) = shape {
            painter.push_clip_rounded(r, rx, ry);
        }
        let enter = s.subpixel && !subpixel;
        if enter {
            painter.set_subpixel(true);
        }
        let subpixel = subpixel || s.subpixel;
        let fade = |c: Rgba| [c[0], c[1], c[2], c[3] * opacity];
        let r = node.rect;
        // Shadows under everything, the first listed on top.
        for sh in s.box_shadow.iter().rev() {
            let rect = Rect {
                x: r.x + sh.x - sh.spread,
                y: r.y + sh.y - sh.spread,
                width: (r.width + 2.0 * sh.spread).max(0.0),
                height: (r.height + 2.0 * sh.spread).max(0.0),
            };
            let radius = s
                .border_radius
                .resolve(Some(r.width.min(r.height)))
                .map_or(0.0, |v| (v + sh.spread).max(0.0));
            painter.shadow(rect, radius, sh.blur, fade(sh.color));
        }
        // `background-clip: content-box`.
        let bg_rect = if s.background_clip_content {
            Rect {
                x: r.x + s.border.left + s.padding.left,
                y: r.y + s.border.top + s.padding.top,
                width: (r.width - s.border.horizontal() - s.padding.horizontal()).max(0.0),
                height: (r.height - s.border.vertical() - s.padding.vertical()).max(0.0),
            }
        } else {
            r
        };
        if s.backdrop_blur > 0.0 && opacity > 0.0 {
            let radius = s
                .border_radius
                .resolve(Some(r.width.min(r.height)))
                .unwrap_or(0.0)
                .min(r.width.min(r.height) / 2.0);
            painter.backdrop(r, radius, s.backdrop_blur, s.backdrop_saturate);
        }
        let b = s.border;
        let uniform = b.top > 0.0 && b.top == b.right && b.top == b.bottom && b.top == b.left;
        let (orx, ory) = (
            s.border_radius
                .resolve(Some(r.width))
                .unwrap_or(0.0)
                .min(r.width / 2.0),
            s.border_radius
                .resolve(Some(r.height))
                .unwrap_or(0.0)
                .min(r.height / 2.0),
        );
        // A rounded uniform border that is not wholly opaque — the box
        // faded (`opacity`), its background or border see-through, or no
        // background at all — is drawn as a ring: the background inset with
        // the inner radii, then the corners as arcs and the sides as bars,
        // nothing painted twice. The fast path below lays the border colour
        // over the whole box and the background over that, which shows the
        // border through anything translucent (a faded button kept ~70 % of
        // its colour at `opacity: 0.45`); and a rounded border with no
        // background used to get square corners.
        let w = b.top;
        let opaque = |c: Rgba| fade(c)[3] >= 1.0;
        let ring = uniform
            && w > 0.0
            && orx >= w
            && (orx - ory).abs() < 0.01
            && s.border_color[3] > 0.0
            && !s.border_dashed
            && !s.background_clip_content
            && !(s.background.is_some_and(opaque) && opaque(s.border_color));
        // A rounded box with a uniform border over a background: the border
        // colour over the whole rounded box, the background inset from it
        // with the inner radii (`radius − border`).
        let rounded_border = !ring
            && uniform
            && orx > 0.0
            && ory > 0.0
            && s.border_color[3] > 0.0
            && s.background.is_some()
            && !s.background_clip_content
            && !s.border_dashed;
        if ring {
            if let Some(bg) = s.background {
                let inner = Rect {
                    x: r.x + w,
                    y: r.y + w,
                    width: (r.width - 2.0 * w).max(0.0),
                    height: (r.height - 2.0 * w).max(0.0),
                };
                let ir = orx - w;
                if ir > 0.0 {
                    painter.fill_rounded_rect(inner, ir, ir, fade(bg));
                } else {
                    painter.fill_rect(inner, fade(bg));
                }
            }
            rounded_ring(painter, r, orx, w, fade(s.border_color));
        } else if rounded_border {
            painter.fill_rounded_rect(r, orx, ory, fade(s.border_color));
            let inner = Rect {
                x: r.x + b.left,
                y: r.y + b.top,
                width: (r.width - b.horizontal()).max(0.0),
                height: (r.height - b.vertical()).max(0.0),
            };
            let (irx, iry) = ((orx - b.left).max(0.0), (ory - b.top).max(0.0));
            let bg = fade(s.background.unwrap());
            if irx > 0.0 && iry > 0.0 {
                painter.fill_rounded_rect(inner, irx, iry, bg);
            } else {
                painter.fill_rect(inner, bg);
            }
        } else if let Some(bg) = s.background {
            let r = bg_rect;
            let rx = s
                .border_radius
                .resolve(Some(r.width))
                .unwrap_or(0.0)
                .min(r.width / 2.0);
            let ry = s
                .border_radius
                .resolve(Some(r.height))
                .unwrap_or(0.0)
                .min(r.height / 2.0);
            if rx > 0.0 && ry > 0.0 {
                painter.fill_rounded_rect(r, rx, ry, fade(bg));
            } else {
                painter.fill_rect(r, fade(bg));
            }
        }
        if !s.background_gradient.is_empty() {
            let stops: Vec<(f32, Rgba)> = s
                .background_gradient
                .iter()
                .map(|&(t, c)| (t, fade(c)))
                .collect();
            // Inside the box's rounded corners, as its background is.
            let round = orx > 0.0 && ory > 0.0;
            if round {
                painter.push_clip_rounded(r, orx, ory);
            }
            painter.fill_gradient(r, s.gradient_shape, &stops, s.blur);
            if round {
                painter.pop_clip();
            }
        }
        if s.border_dashed && uniform && s.border_color[3] > 0.0 {
            dashes(painter, r, orx.min(ory), b.top, fade(s.border_color));
        } else if !rounded_border
            && !ring
            && s.border_color[3] > 0.0
            && (b.top > 0.0 || b.right > 0.0 || b.bottom > 0.0 || b.left > 0.0)
        {
            let c = fade(s.border_color);
            let inner_h = (r.height - b.top - b.bottom).max(0.0);
            for side in [
                Rect {
                    x: r.x,
                    y: r.y,
                    width: r.width,
                    height: b.top,
                },
                Rect {
                    x: r.x,
                    y: r.y + r.height - b.bottom,
                    width: r.width,
                    height: b.bottom,
                },
                Rect {
                    x: r.x,
                    y: r.y + b.top,
                    width: b.left,
                    height: inner_h,
                },
                Rect {
                    x: r.x + r.width - b.right,
                    y: r.y + b.top,
                    width: b.right,
                    height: inner_h,
                },
            ] {
                if side.width > 0.0 && side.height > 0.0 {
                    painter.fill_rect(side, c);
                }
            }
        }
        let content = Rect {
            x: r.x + b.left + s.padding.left,
            y: r.y + b.top + s.padding.top,
            width: (r.width - s.border.horizontal() - s.padding.horizontal()).max(0.0),
            height: (r.height - s.border.vertical() - s.padding.vertical()).max(0.0),
        };
        let redacted = self.is_redacted(id);
        match &node.kind {
            NodeKind::Box => {}
            // A placeholder: a bar for each line of text, a block for a
            // picture or a shape.
            NodeKind::Text(_) if redacted => {
                let n = node.lines.len().max(1);
                let h = content.height / n as f32;
                let bar = fade(self.token(crate::environment::Token::Separator));
                for i in 0..n {
                    let w = node
                        .line_widths
                        .get(i)
                        .copied()
                        .unwrap_or(content.width)
                        .min(content.width);
                    let r = Rect {
                        x: content.x,
                        y: content.y + i as f32 * h + h * 0.2,
                        width: w,
                        height: h * 0.6,
                    };
                    painter.fill_rounded_rect(r, 3.0, 3.0, bar);
                }
            }
            NodeKind::Image { .. } | NodeKind::Surface { .. } | NodeKind::Path { .. }
                if redacted =>
            {
                painter.fill_rounded_rect(
                    content,
                    4.0,
                    4.0,
                    fade(self.token(crate::environment::Token::Separator)),
                );
            }
            NodeKind::Text(t) if !node.pieces.is_empty() => {
                // Styled runs: each piece where layout put it, in its own
                // style, over its highlight, under its decoration.
                let runs = node.runs.as_deref().unwrap_or(&[]);
                let n = node.lines.len().max(1);
                let h = content.height / n as f32;
                for p in &node.pieces {
                    let run = runs.get(p.run as usize);
                    let st = run.map_or_else(|| s.clone(), |r| r.style(s));
                    let line = Rect {
                        x: content.x + p.x,
                        y: content.y + p.line as f32 * h,
                        width: p.width,
                        height: if n > 1 { h } else { content.height },
                    };
                    if let Some(bg) = run.and_then(|r| r.background) {
                        painter.fill_rect(line, fade(bg));
                    }
                    let mut ts = TextStyle::of(&st);
                    ts.color = fade(ts.color);
                    painter.text(line, &t[p.start as usize..p.end as usize], &ts);
                    if st.underline {
                        let (y, thick) = underline_position(line, st.font_size);
                        let c = fade(st.decoration_color.unwrap_or(st.color));
                        painter.underline(line.x, y, p.width, thick, c, st.decoration_wavy);
                    }
                }
            }
            NodeKind::Text(t) => {
                let mut style = TextStyle::of(s);
                style.color = fade(style.color);
                let decorate = |painter: &mut dyn Painter, line: Rect, i: usize| {
                    if let (true, Some(&w)) = (s.underline, node.line_widths.get(i)) {
                        let (y, thick) = underline_position(line, s.font_size);
                        let c = fade(s.decoration_color.unwrap_or(s.color));
                        painter.underline(line.x, y, w, thick, c, s.decoration_wavy);
                    }
                };
                if node.lines.len() > 1 {
                    // A wrapped run: one line box each.
                    let h = content.height / node.lines.len() as f32;
                    for (i, line) in node.lines.iter().enumerate() {
                        // Across the box by `text-align`, by the line's width.
                        let shift = node
                            .line_widths
                            .get(i)
                            .map_or(0.0, |w| s.text_align.offset(content.width - w));
                        let r = Rect {
                            x: content.x + shift,
                            y: content.y + i as f32 * h,
                            width: content.width - shift,
                            height: h,
                        };
                        painter.text(r, line.trim_end_matches([' ', '\u{200b}']), &style);
                        decorate(painter, r, i);
                    }
                } else if let [line] = node.lines.as_slice() {
                    // An ellipsized (or once-wrapped) run.
                    let shift = node
                        .line_widths
                        .first()
                        .map_or(0.0, |w| s.text_align.offset(content.width - w));
                    let r = Rect {
                        x: content.x + shift,
                        width: content.width - shift,
                        ..content
                    };
                    painter.text(r, line, &style);
                    decorate(painter, r, 0);
                } else {
                    painter.text(content, t, &style);
                    decorate(painter, content, 0);
                }
            }
            NodeKind::Path {
                d,
                view_box,
                stroke,
                fill,
            } => match node.transform {
                Some(t) => {
                    let centre = (
                        view_box[0] + view_box[2] / 2.0,
                        view_box[1] + view_box[3] / 2.0,
                    );
                    let moved: Vec<String> = d
                        .iter()
                        .map(|p| crate::transform::transform_path(p, &t, centre))
                        .collect();
                    painter.path(content, &moved, *view_box, *stroke, *fill, fade(s.color))
                }
                None => {
                    // Circular arcs, stroked: from their distance field.
                    let (sx, sy) = (
                        content.width / view_box[2].max(1e-6),
                        content.height / view_box[3].max(1e-6),
                    );
                    let arcs: Option<Vec<ArcShape>> = match stroke {
                        Some(_) if !*fill && !d.is_empty() => d
                            .iter()
                            .map(|p| {
                                ArcShape::from_svg(p).and_then(|a| {
                                    a.mapped(
                                        view_box[0],
                                        view_box[1],
                                        sx,
                                        sy,
                                        (content.x, content.y),
                                    )
                                })
                            })
                            .collect(),
                        _ => None,
                    };
                    // Circles (`circle_path`'s): a filled one a disc, a
                    // rounded box's distance field; a stroked one a ring.
                    let uniform = (sx - sy).abs() <= 1e-3 * sx.abs().max(1e-6);
                    let circles: Option<Vec<(f32, f32, f32)>> = (uniform && !d.is_empty())
                        .then(|| d.iter().map(|p| parse_circle(p)).collect())
                        .flatten();
                    match (arcs, circles, stroke) {
                        (Some(arcs), _, Some(w)) => {
                            for a in arcs {
                                painter.stroke_arc(a, w * sx.min(sy), fade(s.color));
                            }
                        }
                        (None, Some(circles), _) => {
                            for (cx, cy, r) in circles {
                                let (x, y, r) = (
                                    content.x + (cx - view_box[0]) * sx,
                                    content.y + (cy - view_box[1]) * sy,
                                    r * sx,
                                );
                                if *fill {
                                    painter.fill_rounded_rect(
                                        Rect {
                                            x: x - r,
                                            y: y - r,
                                            width: 2.0 * r,
                                            height: 2.0 * r,
                                        },
                                        r,
                                        r,
                                        fade(s.color),
                                    );
                                }
                                if let Some(w) = stroke {
                                    let ring = ArcShape {
                                        cx: x,
                                        cy: y,
                                        radius: r,
                                        start: 0.0,
                                        sweep: std::f32::consts::TAU,
                                    };
                                    painter.stroke_arc(ring, w * sx, fade(s.color));
                                }
                            }
                        }
                        _ => painter.path(content, d, *view_box, *stroke, *fill, fade(s.color)),
                    }
                }
            },
            NodeKind::Image { .. } | NodeKind::Surface { .. } => {
                let (iw, ih) = match &node.kind {
                    NodeKind::Image { image } => (image.width, image.height),
                    NodeKind::Surface { width, height, .. } => (*width, *height),
                    _ => unreachable!(),
                };
                let dest = crate::image::fit(s.object_fit, content, iw as f32, ih as f32);
                let spills = dest.x < content.x
                    || dest.y < content.y
                    || dest.x + dest.width > content.x + content.width
                    || dest.y + dest.height > content.y + content.height;
                if spills {
                    painter.push_clip(content);
                }
                match &node.kind {
                    NodeKind::Image { image } => painter.image(dest, image, opacity),
                    NodeKind::Surface { id, .. } => painter.surface(dest, *id, opacity),
                    _ => unreachable!(),
                }
                if spills {
                    painter.pop_clip();
                }
            }
        }
        if s.clip {
            let rx = s
                .border_radius
                .resolve(Some(r.width))
                .unwrap_or(0.0)
                .min(r.width / 2.0);
            let ry = s
                .border_radius
                .resolve(Some(r.height))
                .unwrap_or(0.0)
                .min(r.height / 2.0);
            if rx > 0.0 && ry > 0.0 {
                painter.push_clip_rounded(r, rx, ry);
            } else {
                painter.push_clip(r);
            }
        }
        // What its children can show in: cut to it if it clips; anything,
        // once turned in 3D (their boxes are then not where they show).
        let inner = if transform.is_some() || self.env.parallax.contains_key(&id) {
            EVERYWHERE
        } else if s.clip {
            let (x0, y0) = (visible.x.max(r.x), visible.y.max(r.y));
            let (x1, y1) = (
                (visible.x + visible.width).min(r.x + r.width),
                (visible.y + visible.height).min(r.y + r.height),
            );
            Rect {
                x: x0,
                y: y0,
                width: (x1 - x0).max(0.0),
                height: (y1 - y0).max(0.0),
            }
        } else {
            visible
        };
        for c in self.paint_order_iter(id) {
            if self.nodes[c.0].deferred.is_some() {
                continue;
            }
            self.paint_node(c, painter, opacity, subpixel, inner);
        }
        if s.clip {
            painter.pop_clip();
        }
        if let Some(bar) = self
            .scrollbar(id)
            .filter(|_| self.scrollbar_opacity(id) > 0.0)
        {
            let shown = self.scrollbar_opacity(id);
            let (mut thumb, mut track) = self.scrollbar_colors(id);
            thumb[3] *= shown;
            track[3] *= shown;
            if track[3] > 0.0 {
                painter.fill_rect(bar.track, fade(track));
            }
            let r = bar.thumb.width / 2.0;
            painter.fill_rounded_rect(bar.thumb, r, r, fade(thumb));
        }
        if shape.is_some() {
            painter.pop_clip();
        }
        if effect {
            painter.pop_effect();
        }
        if transform.is_some() {
            self.paint_shade(id, painter, opacity);
            self.paint_glare(id, painter, opacity);
            painter.pop_transform();
        }
        if blur.is_some() {
            painter.pop_blur();
        }
        self.paint_reflection(id, painter, opacity);
        if enter {
            painter.set_subpixel(false);
        }
    }
}

/// An SVG `<circle>` as path data, in the one form [`parse_circle`] reads
/// back: two half arcs from the rightmost point.
///
/// A painter that can draw a true oval should, for these: browsers stroke a
/// `<circle>` as an oval, which rasterizes differently from the same circle
/// given as arcs.
pub fn circle_path(cx: f32, cy: f32, r: f32) -> String {
    format!(
        "M{} {}A{r} {r} 0 1 1 {} {}A{r} {r} 0 1 1 {} {}Z",
        cx + r,
        cy,
        cx - r,
        cy,
        cx + r,
        cy
    )
}

/// The centre and radius of path data made by [`circle_path`].
pub fn parse_circle(d: &str) -> Option<(f32, f32, f32)> {
    let rest = d.strip_prefix('M')?.strip_suffix('Z')?;
    let mut parts = rest.split('A');
    let start: Vec<f32> = parts
        .next()?
        .split(' ')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .ok()?;
    let a: Vec<f32> = parts
        .next()?
        .split(' ')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .ok()?;
    let b: Vec<f32> = parts
        .next()?
        .split(' ')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .ok()?;
    if parts.next().is_some() || start.len() != 2 || a.len() != 7 || b.len() != 7 {
        return None;
    }
    let r = a[0];
    let (cx, cy) = (start[0] - r, start[1]);
    (a[1] == r
        && b[0] == r
        && b[1] == r
        && a[5] == cx - r
        && a[6] == cy
        && b[5] == start[0]
        && b[6] == cy)
        .then_some((cx, cy, r))
}

/// A view that culls nothing.
const EVERYWHERE: Rect = Rect {
    x: -1e9,
    y: -1e9,
    width: 2e9,
    height: 2e9,
};

/// A rounded rectangle's border, `width` wide inside `r`, corners of radius
/// `radius`: four quarter arcs and four straight sides, none overlapping
/// (the sides stop half a border short of each arc, whose round cap covers
/// the rest), so a translucent colour is laid down once everywhere.
fn rounded_ring(painter: &mut dyn Painter, r: Rect, radius: f32, width: f32, color: Rgba) {
    use std::f32::consts::{FRAC_PI_2, PI};
    let (half, mid) = (width / 2.0, radius - width / 2.0);
    let (l, t, rt, bt) = (r.x + radius, r.y + radius, r.x + r.width - radius, r.y + r.height - radius);
    // Clockwise from straight up: top-left from the left, top-right from the
    // top, and on round.
    for (cx, cy, start) in [(l, t, 1.5 * PI), (rt, t, 0.0), (rt, bt, FRAC_PI_2), (l, bt, PI)] {
        painter.stroke_arc(
            ArcShape {
                cx,
                cy,
                radius: mid,
                start,
                sweep: FRAC_PI_2,
            },
            width,
            color,
        );
    }
    let across = (r.width - 2.0 * radius - width).max(0.0);
    let down = (r.height - 2.0 * radius - width).max(0.0);
    for side in [
        Rect { x: l + half, y: r.y, width: across, height: width },
        Rect { x: l + half, y: r.y + r.height - width, width: across, height: width },
        Rect { x: r.x, y: t + half, width, height: down },
        Rect { x: r.x + r.width - width, y: t + half, width, height: down },
    ] {
        if side.width > 0.0 && side.height > 0.0 {
            painter.fill_rect(side, color);
        }
    }
}

/// A dashed border `width` wide round `r` (its corners of `radius`):
/// dashes three widths long, as long apart, walked along the outline's
/// middle — straight edges and corner arcs alike — each a run of
/// `width`-square steps.
fn dashes(painter: &mut dyn Painter, r: Rect, radius: f32, width: f32, color: Rgba) {
    let half = width / 2.0;
    // the outline's middle: a rounded rectangle inset half the width
    let (x0, y0, x1, y1) = (
        r.x + half,
        r.y + half,
        r.x + r.width - half,
        r.y + r.height - half,
    );
    let rad = (radius - half).clamp(0.0, ((x1 - x0) / 2.0).min((y1 - y0) / 2.0));
    let (w, h) = (
        (x1 - x0 - 2.0 * rad).max(0.0),
        (y1 - y0 - 2.0 * rad).max(0.0),
    );
    let arc = std::f32::consts::FRAC_PI_2 * rad;
    let total = 2.0 * (w + h) + 4.0 * arc;
    // the point `t` along it, clockwise from the top edge's start
    let at = |mut t: f32| -> (f32, f32) {
        let corner = |cx: f32, cy: f32, from: f32, u: f32| {
            let a = from + u / rad.max(1e-6);
            (cx + rad * a.cos(), cy + rad * a.sin())
        };
        let pi = std::f32::consts::PI;
        let segs: [(f32, &dyn Fn(f32) -> (f32, f32)); 8] = [
            (w, &|u| (x0 + rad + u, y0)),
            (arc, &|u| corner(x1 - rad, y0 + rad, -pi / 2.0, u)),
            (h, &|u| (x1, y0 + rad + u)),
            (arc, &|u| corner(x1 - rad, y1 - rad, 0.0, u)),
            (w, &|u| (x1 - rad - u, y1)),
            (arc, &|u| corner(x0 + rad, y1 - rad, pi / 2.0, u)),
            (h, &|u| (x0, y1 - rad - u)),
            (arc, &|u| corner(x0 + rad, y0 + rad, pi, u)),
        ];
        for (len, f) in segs {
            if t <= len {
                return f(t);
            }
            t -= len;
        }
        (x0 + rad, y0)
    };
    let dash = (width * 3.0).max(2.0);
    // whole dash-and-gap periods round it, so the ends meet
    let periods = (total / (2.0 * dash)).round().max(1.0);
    let period = total / periods;
    let step = (width * 0.5).max(0.5);
    let mut t = 0.0;
    while t < total {
        if t % period < period / 2.0 {
            let (px, py) = at(t);
            painter.fill_rect(
                Rect {
                    x: px - half,
                    y: py - half,
                    width,
                    height: width,
                },
                color,
            );
        }
        t += step;
    }
}

#[cfg(test)]
mod arc_tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI, TAU};

    #[test]
    fn svg_arcs_are_read_as_circles() {
        // A quarter, clockwise from the top to the right.
        let a = ArcShape::from_svg("M50 10 A40 40 0 0 1 90 50").unwrap();
        assert!(
            (a.cx - 50.0).abs() < 1e-3
                && (a.cy - 50.0).abs() < 1e-3
                && (a.radius - 40.0).abs() < 1e-3,
            "{a:?}"
        );
        assert!(
            a.start.abs() < 1e-4 && (a.sweep - FRAC_PI_2).abs() < 1e-4,
            "{a:?}"
        );
        // The long way round, and anticlockwise.
        let long = ArcShape::from_svg("M50 10 A40 40 0 1 1 10 50").unwrap();
        assert!((long.sweep - 1.5 * PI).abs() < 1e-4, "{long:?}");
        let back = ArcShape::from_svg("M50 10 A40 40 0 0 0 10 50").unwrap();
        assert!((back.sweep + FRAC_PI_2).abs() < 1e-4, "{back:?}");
        // Two halves round one circle: the whole ring.
        let ring = ArcShape::from_svg(
            "M50.000 10.000 A40 40 0 0 1 50.000 90.000 A40 40 0 0 1 50.000 10.000",
        )
        .unwrap();
        assert!((ring.sweep - TAU).abs() < 1e-3);
        // Not an arc: lines, ellipses, a second circle.
        for d in [
            "M0 0 L10 10",
            "M50 10 A40 20 0 0 1 90 50",
            "M50 10 A40 40 0 0 1 90 50 A10 10 0 0 1 80 60",
            "M50 10 A40 40 0 0 1 90 50 L0 0",
            "",
        ] {
            assert_eq!(ArcShape::from_svg(d), None, "{d}");
        }
    }

    #[test]
    fn the_distance_field_has_round_ends_and_a_gap() {
        // The dial: three quarters from bottom left, clockwise.
        let a = ArcShape {
            cx: 0.0,
            cy: 0.0,
            radius: 40.0,
            start: -0.75 * PI,
            sweep: 1.5 * PI,
        };
        let w = 10.0;
        // On the ring, top: inside by half the width.
        assert!((a.distance(w, 0.0, -40.0) + 5.0).abs() < 1e-3);
        // Its edges.
        assert!(a.distance(w, 0.0, -45.0).abs() < 1e-3 && a.distance(w, 0.0, -35.0).abs() < 1e-3);
        // The gap at the bottom: far from it.
        assert!(a.distance(w, 0.0, 40.0) > 20.0);
        // Round ends: just past the start, along the ring, at most half the width out.
        let (ex, ey) = (40.0 * (-0.75 * PI).sin(), -40.0 * (-0.75 * PI).cos());
        assert!(
            (a.distance(w, ex, ey) + 5.0).abs() < 1e-3,
            "the end's centre"
        );
        // A ring: the same everywhere round.
        let ring = ArcShape { sweep: TAU, ..a };
        assert!((ring.distance(w, 0.0, 40.0) + 5.0).abs() < 1e-3);
        // Anticlockwise draws the same as its clockwise twin.
        let twin = ArcShape {
            start: 0.75 * PI,
            sweep: -1.5 * PI,
            ..a
        };
        for (x, y) in [(0.0, -40.0), (30.0, 30.0), (-30.0, 30.0), (5.0, 41.0)] {
            assert!(
                (twin.distance(w, x, y) - a.distance(w, x, y)).abs() < 1e-3,
                "({x}, {y})"
            );
        }
    }

    #[test]
    fn circle_paths_are_painted_as_discs_and_rings() {
        let mut ui = crate::Ui::new();
        let root = ui.root();
        let st = || crate::style! { width: 40px; height: 40px; };
        ui.add_path(
            root,
            st(),
            vec![circle_path(20.0, 20.0, 10.0)],
            [0.0, 0.0, 40.0, 40.0],
            None,
            true,
        );
        ui.add_path(
            root,
            st(),
            vec![circle_path(20.0, 20.0, 15.0)],
            [0.0, 0.0, 40.0, 40.0],
            Some(2.0),
            false,
        );
        ui.layout(200.0, 200.0, &mut crate::FixedWidthText(7.0));
        let mut scene = crate::scene::Scene::default();
        ui.paint(&mut scene.recorder());
        let discs = scene
            .items
            .iter()
            .filter(
                |i| matches!(i, crate::scene::Item::Rect { rx, .. } if (*rx - 10.0).abs() < 1e-3),
            )
            .count();
        let rings: Vec<_> = scene
            .items
            .iter()
            .filter_map(|i| match i {
                crate::scene::Item::Arc { arc, width, .. } => Some((arc.radius, arc.sweep, *width)),
                _ => None,
            })
            .collect();
        assert_eq!(discs, 1);
        assert_eq!(rings, vec![(15.0, std::f32::consts::TAU, 2.0)]);
        assert!(
            !scene
                .items
                .iter()
                .any(|i| matches!(i, crate::scene::Item::Path { .. })),
            "no path left to cut up"
        );
    }

    #[test]
    fn a_gauge_dial_is_painted_as_arcs() {
        use crate::charts::{gauge, GaugeKind};
        let mut ui = crate::Ui::new();
        let root = ui.root();
        let mut g = gauge(&mut ui, root, "Level", 0.0..=100.0, GaugeKind::Circular);
        ui.set_reduce_motion(true);
        g.set(&mut ui, 50.0);
        ui.layout(400.0, 300.0, &mut crate::FixedWidthText(7.0));
        let mut scene = crate::scene::Scene::default();
        ui.paint(&mut scene.recorder());
        let arcs: Vec<_> = scene
            .items
            .iter()
            .filter_map(|i| match i {
                crate::scene::Item::Arc { arc, width, .. } => Some((*arc, *width)),
                _ => None,
            })
            .collect();
        // The track and the fill, 96 px across: radius 40 of 100, 9.6 wide.
        assert_eq!(
            arcs.len(),
            2,
            "{:?}",
            scene
                .items
                .iter()
                .filter(|i| matches!(i, crate::scene::Item::Path { .. }))
                .count()
        );
        assert!(
            (arcs[0].0.radius - 38.4).abs() < 0.01 && (arcs[0].1 - 9.6).abs() < 0.01,
            "{arcs:?}"
        );
        assert!(
            (arcs[1].0.sweep - 0.75 * PI).abs() < 1e-3,
            "half of three quarters: {arcs:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[derive(Default)]
    struct Record(Vec<String>);

    impl TextMeasure for Record {
        fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
            FixedWidthText(7.0).measure(text, style)
        }
    }

    impl Painter for Record {
        fn fill_rect(&mut self, r: Rect, c: Rgba) {
            self.0.push(format!(
                "rect {} {} {} {} {:?}",
                r.x, r.y, r.width, r.height, c
            ));
        }
        fn fill_rounded_rect(&mut self, r: Rect, rx: f32, ry: f32, c: Rgba) {
            self.0.push(format!(
                "round {} {} {} {} {rx} {ry} {:?}",
                r.x, r.y, r.width, r.height, c
            ));
        }
        fn text(&mut self, r: Rect, t: &str, s: &TextStyle) {
            self.0
                .push(format!("text {} {} {t} {}", r.x, r.y, s.css_font()));
        }
        fn path(&mut self, r: Rect, d: &[String], _: [f32; 4], _: Option<f32>, _: bool, _: Rgba) {
            self.0.push(format!("path {} {} {}", r.x, r.y, d.len()));
        }
    }

    #[test]
    fn paints_background_borders_then_content_in_tree_order() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, Style::flex(FlexDirection::Row));
        let item = ui.add_box(
            root,
            Style {
                background: Some([0.0, 0.0, 0.0, 1.0]),
                border: Edges::all(1.0),
                border_color: [0.5, 0.5, 0.5, 1.0],
                padding: Edges::symmetric(0.0, 2.0),
                font_size: 13.0,
                opacity: 0.5,
                ..Style::default()
            },
        );
        ui.add_text(item, "ab");
        let mut rec = Record::default();
        ui.layout(100.0, 20.0, &mut rec);
        ui.paint(&mut rec);
        assert_eq!(rec.0[0], "rect 0 0 20 20 [0.0, 0.0, 0.0, 0.5]");
        assert_eq!(rec.0.len(), 6, "{:?}", rec.0);
        assert_eq!(rec.0[5], "text 3 1 ab 400 13px sans-serif");
    }

    #[test]
    fn rounded_borders_are_the_box_then_the_background_inset() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, Style::flex(FlexDirection::Row));
        ui.add_box(
            root,
            Style {
                width: Length::Px(13.0),
                height: Length::Px(13.0),
                box_sizing: BoxSizing::BorderBox,
                background: Some([1.0; 4]),
                border: Edges::all(1.0),
                border_color: [0.5, 0.5, 0.5, 1.0],
                border_radius: Length::Px(2.0),
                ..Style::default()
            },
        );
        let mut rec = Record::default();
        ui.layout(100.0, 20.0, &mut rec);
        ui.paint(&mut rec);
        assert_eq!(
            rec.0,
            [
                "round 0 0 13 13 2 2 [0.5, 0.5, 0.5, 1.0]",
                "round 1 1 11 11 1 1 [1.0, 1.0, 1.0, 1.0]"
            ]
        );
    }

    #[test]
    fn circles_round_trip_through_path_data() {
        let d = super::super::circle_path(9.0, 6.0, 2.0);
        assert_eq!(d, "M11 6A2 2 0 1 1 7 6A2 2 0 1 1 11 6Z");
        assert_eq!(super::super::parse_circle(&d), Some((9.0, 6.0, 2.0)));
        assert_eq!(super::super::parse_circle("M11 6A2 2 0 1 1 7 6Z"), None);
    }

    #[test]
    fn a_dashed_border_is_dashes_all_round() {
        let mut p = Record::default();
        crate::paint::dashes(
            &mut p,
            Rect {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 40.0,
            },
            8.0,
            1.0,
            [0.0, 0.0, 0.0, 1.0],
        );
        let rects: Vec<[f32; 2]> =
            p.0.iter()
                .map(|l| {
                    let v: Vec<f32> = l
                        .split(' ')
                        .skip(1)
                        .take(2)
                        .map(|x| x.parse().unwrap())
                        .collect();
                    [v[0], v[1]]
                })
                .collect();
        // on the outline everywhere
        assert!(rects.iter().all(|&[x, y]| x <= 1.0
            || y <= 1.0
            || x >= 98.0
            || y >= 38.0
            || (x < 9.0 || x > 90.0) && (y < 9.0 || y > 30.0)));
        // along the top, in dashes with gaps between
        let top: Vec<f32> = rects
            .iter()
            .filter(|r| r[1] < 0.5 && r[0] > 10.0 && r[0] < 90.0)
            .map(|r| r[0])
            .collect();
        assert!(
            top.len() > 40 && top.len() < 140,
            "{} steps along the top",
            top.len()
        );
        assert!(top.windows(2).any(|w| w[1] - w[0] > 2.0), "with gaps");
    }
}
