//! A beam of light running round a box's border: a short comet of colour
//! circling a card or a text field, or a glow sliding to and fro along a
//! search bar's bottom edge. It shows something is live (a field taking
//! input, an assistant thinking) without taking any room.
//!
//! Any box can have one. It is a style property, so it goes wherever styles
//! go: [`Ui::set_border_beam`], `style!`, a state style (on while hovered or
//! focused), or Tailwind classes:
//!
//! ```
//! use threers_ui::border_beam::{BeamPalette, BeamStyle};
//! use threers_ui::{style, FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! // From Rust: a neon beam, two comets, a 2px line, starting at the right.
//! let card = ui.add_box(root, style! { width: 240px; height: 120px; border-radius: 16px; background: #1c1c1c; });
//! ui.set_border_beam(card, BeamStyle::default().palette(BeamPalette::NEON).count(2).width(2.0).angle(90.0));
//!
//! // From CSS, with longhands for each setting.
//! let field = ui.add_box(root, style! {
//!     width: 300px; height: 44px; border-radius: 22px;
//!     border-beam: line neon;
//!     border-beam-speed: 1.5;
//! });
//! // While hovered only, fading in and out.
//! let button = ui.add_box(root, style! { width: 120px; height: 40px; border-radius: 8px; });
//! ui.hover_style(button, "border-beam: small sunset");
//! // From Tailwind.
//! let tw = ui.add_tw(root, "w-64 h-24 rounded-xl beam beam-aurora beam-speed-150 beam-w-2 focus:beam-sm");
//!
//! ui.layout(800.0, 600.0, &mut FixedWidthText(7.0));
//! assert!(ui.needs_frame(), "beams run round");
//! assert!(ui.border_beam(button).is_none(), "not hovered");
//! ```
//!
//! The GPU painter draws it in one shader effect on the box, over whatever
//! the box draws: a crisp line on the edge, a tight halo, and a wide soft
//! glow reaching in, laid down as a chain of round blobs along the beam's
//! path so corners come out round on any shape. It follows the box's own
//! `border-radius` unless given a [`BeamShape`]. Nothing reaches outside
//! the box. Taken off, a beam fades out, and once gone it costs nothing.
//! Under reduced motion it holds still where it is. A box with an effect
//! of its own ([`Ui::set_effect`]) keeps it and shows no beam. Painters
//! that cannot run shaders (the 2D canvas, the CPU overlay) draw the box
//! without it.
//!
//! # CSS
//!
//! `border-beam` is `none`, or any of: a kind (`large`, `small`, `line`), a
//! palette's name (`neon`) or colours (`#22d3ee #a855f7`), a direction
//! (`clockwise`, `counter-clockwise`), an edge for a line beam (`top`,
//! `bottom`, `left`, `right`), and a lap's time (`3s`). Its longhands,
//! each of which also turns the beam on:
//!
//! | property | value |
//! |---|---|
//! | `border-beam-kind` | `large`, `small`, `line` |
//! | `border-beam-colors` | a palette's name, or one to many colours |
//! | `border-beam-speed` | 1 as timed; 0 holds it still |
//! | `border-beam-duration` | a lap (or a line's there and back): `2.4s` |
//! | `border-beam-width` | the line on the edge: `2px` |
//! | `border-beam-length` | each comet, a share of the border: `30%` |
//! | `border-beam-glow` | how far the soft glow reaches in: `20px` |
//! | `border-beam-intensity` | `0.8` or `80%` |
//! | `border-beam-angle` | where the head starts, clockwise from the top: `90deg`, `0.25turn` |
//! | `border-beam-count` | comets spaced evenly round: `2` |
//! | `border-beam-direction` | `clockwise`, `counter-clockwise` |
//! | `border-beam-edge` | a line beam's edge |
//! | `border-beam-fade` | fading in and out: `600ms` |
//! | `border-beam-shape` | `auto` (the box's radius), `capsule`, `circle`, a radius |
//!
//! # Tailwind
//!
//! `beam`, `beam-sm`, `beam-line`, `beam-none`; colours `beam-neon` (a
//! palette), `beam-sky-400` (one colour), `beam-[#f00_#0f0_#00f]`; and
//! `beam-reverse`, `beam-top` (and the other edges), `beam-speed-150` (a
//! percentage), `beam-duration-3000` (ms), `beam-w-2` (px),
//! `beam-length-30` (%), `beam-glow-24` (px), `beam-intensity-80` (%),
//! `beam-angle-90` (degrees), `beam-count-2`, `beam-fade-300` (ms). Any
//! takes an arbitrary value: `beam-angle-[-45deg]`. With variants:
//! `focus-within:beam`, `hover:beam-speed-200`.

use std::cell::OnceCell;

use crate::shader::{Effect, Shader};
use crate::style::Length;
use crate::{style, NodeId, Rgba, Style, Ui};

/// What the beam does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BeamKind {
    /// Comets circling the whole border, with a soft glow reaching in
    /// (cards, text fields).
    #[default]
    Large,
    /// Longer, tighter comets for small controls (icon buttons, avatars).
    Small,
    /// A glow sliding to and fro along one edge (search bars).
    Line,
}

impl BeamKind {
    /// Each comet's length and the glow's reach, as designed.
    fn defaults(self) -> (f32, f32) {
        match self {
            BeamKind::Large => (0.36, 20.0),
            BeamKind::Small => (0.5, 4.0),
            BeamKind::Line => (0.28, 10.0),
        }
    }
}

/// The outline the beam follows.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum BeamShape {
    /// The box's own: its `border-radius`.
    #[default]
    Auto,
    /// Corners of this radius, in UI pixels.
    Rounded(f32),
    /// Its shorter sides fully round.
    Capsule,
    /// The circle inside the box, about its middle.
    Circle,
}

/// Which way the beam goes round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BeamDirection {
    /// Left to right along the top edge (and a line beam left to right, or
    /// down).
    #[default]
    Clockwise,
    /// Right to left along the top edge.
    CounterClockwise,
}

/// The edge a [`BeamKind::Line`] beam slides along.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BeamEdge {
    Top,
    #[default]
    Bottom,
    Left,
    Right,
}

/// Five colours a beam runs through, from its tail to its head (sRGB, 0 to
/// 1, as in CSS). A colour's alpha weighs it: a faint one lets the others
/// show through.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamPalette(pub [Rgba; 5]);

const fn hex(c: u32) -> Rgba {
    [
        ((c >> 16) & 255) as f32 / 255.0,
        ((c >> 8) & 255) as f32 / 255.0,
        (c & 255) as f32 / 255.0,
        1.0,
    ]
}

const fn faint(c: Rgba, a: f32) -> Rgba {
    [c[0], c[1], c[2], a]
}

impl BeamPalette {
    /// Green, blue, violet, pink, orange.
    pub const COLORFUL: BeamPalette = BeamPalette([
        hex(0x40d973),
        hex(0x3399f2),
        hex(0xb359f2),
        hex(0xff59a6),
        hex(0xffa640),
    ]);
    /// Blue through cyan and green to yellow and red.
    pub const SPECTRUM: BeamPalette = BeamPalette([
        hex(0x4280f5),
        hex(0x40d1d9),
        hex(0x40c766),
        hex(0xfabd33),
        hex(0xeb4740),
    ]);
    /// Cyan, blue, violet, magenta.
    pub const NEON: BeamPalette = BeamPalette([
        hex(0x00e5ff),
        hex(0x4d66ff),
        hex(0x8b5cf6),
        hex(0xd933ff),
        hex(0xff3399),
    ]);
    /// Gold, orange, coral, rose, purple.
    pub const SUNSET: BeamPalette = BeamPalette([
        hex(0xffc74d),
        hex(0xff804d),
        hex(0xff4d73),
        hex(0xd946ef),
        hex(0xb34de6),
    ]);
    /// Mint, sky, indigo, lilac.
    pub const AURORA: BeamPalette = BeamPalette([
        hex(0x4dffb3),
        hex(0x33ccff),
        hex(0x6b66ff),
        hex(0x8066ff),
        hex(0xe680ff),
    ]);
    /// A cool, quiet white-blue, faint at both ends.
    pub const SILVER: BeamPalette = BeamPalette([
        faint(hex(0xbfccff), 0.2),
        faint(hex(0xd4dcff), 0.6),
        hex(0xe6edff),
        faint(hex(0xd4dcff), 0.6),
        faint(hex(0xbfccff), 0.2),
    ]);
    /// White, faint at both ends.
    pub const MONO: BeamPalette = BeamPalette([
        faint(hex(0xffffff), 0.25),
        faint(hex(0xffffff), 0.6),
        faint(hex(0xffffff), 0.9),
        faint(hex(0xffffff), 0.6),
        faint(hex(0xffffff), 0.25),
    ]);

    /// Every named palette, with its name.
    pub const NAMED: [(&'static str, BeamPalette); 7] = [
        ("Colorful", BeamPalette::COLORFUL),
        ("Spectrum", BeamPalette::SPECTRUM),
        ("Neon", BeamPalette::NEON),
        ("Sunset", BeamPalette::SUNSET),
        ("Aurora", BeamPalette::AURORA),
        ("Silver", BeamPalette::SILVER),
        ("Mono", BeamPalette::MONO),
    ];

    /// The palette called `name` (any case), of [`NAMED`](Self::NAMED).
    pub fn named(name: &str) -> Option<BeamPalette> {
        BeamPalette::NAMED
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|&(_, p)| p)
    }

    /// One colour, fading in and out along the beam.
    pub fn single(c: Rgba) -> BeamPalette {
        let a = c[3];
        BeamPalette([
            faint(c, a * 0.3),
            faint(c, a * 0.7),
            c,
            faint(c, a * 0.7),
            faint(c, a * 0.3),
        ])
    }

    /// Any number of colours, spread evenly tail to head (sampled at five
    /// points between them). `None` for none.
    pub fn colors(colors: &[Rgba]) -> Option<BeamPalette> {
        let n = colors.len();
        if n == 0 {
            return None;
        }
        let stops: Vec<(f32, Rgba)> = colors
            .iter()
            .enumerate()
            .map(|(i, &c)| (i as f32 / (n.max(2) - 1) as f32, c))
            .collect();
        Some(BeamPalette(
            [0.0, 0.25, 0.5, 0.75, 1.0].map(|t| crate::gradient_at(&stops, t)),
        ))
    }

    /// From CSS colours (`#22d3ee`, `rgb(…)`, a name), spread evenly; one
    /// colour as [`single`](Self::single). `None` if one does not parse or
    /// there are none.
    pub fn css(colors: &[&str]) -> Option<BeamPalette> {
        let c: Vec<Rgba> = colors
            .iter()
            .map(|c| crate::css::color(c))
            .collect::<Option<_>>()?;
        match c.as_slice() {
            [one] => Some(BeamPalette::single(*one)),
            _ => BeamPalette::colors(&c),
        }
    }
}

impl From<crate::indicators::Palette> for BeamPalette {
    fn from(p: crate::indicators::Palette) -> Self {
        BeamPalette::colors(&p.0).unwrap()
    }
}

/// How a beam looks and moves. Start from [`BeamStyle::new`] (or
/// `default()`, a large one) and change what you like.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamStyle {
    pub kind: BeamKind,
    pub shape: BeamShape,
    pub direction: BeamDirection,
    /// Where a [`BeamKind::Line`] beam runs.
    pub edge: BeamEdge,
    pub palette: BeamPalette,
    /// How long each comet is, as a share of the border (of its own
    /// stretch of it, with several); a line beam's, of its edge.
    pub length: f32,
    /// How far the soft glow reaches in, in UI pixels.
    pub glow: f32,
    /// The crisp line on the edge, in UI pixels.
    pub width: f32,
    /// How bright, 0 to 1.
    pub intensity: f32,
    /// Seconds for a lap of the border.
    pub lap: f32,
    /// Seconds for a line beam to go along its edge and back.
    pub sweep: f32,
    /// Seconds to fade in and out when turned on and off.
    pub fade: f32,
    /// How fast it goes, 1 as timed by `lap` and `sweep`; 0 holds it still.
    pub speed: f32,
    /// Where the head starts, in degrees clockwise from straight up (seen
    /// from the box's middle); `None`, somewhere of its own, so beams side
    /// by side do not run in step.
    pub angle: Option<f32>,
    /// How many comets, spaced evenly round (1 to 8).
    pub count: u8,
}

impl BeamStyle {
    /// A [`BeamPalette::COLORFUL`] beam of `kind`, on the box's own corners,
    /// clockwise, a lap every 2.4 seconds.
    pub fn new(kind: BeamKind) -> Self {
        let (length, glow) = kind.defaults();
        Self {
            kind,
            shape: BeamShape::Auto,
            direction: BeamDirection::Clockwise,
            edge: BeamEdge::Bottom,
            palette: BeamPalette::COLORFUL,
            length,
            glow,
            width: 1.5,
            intensity: 1.0,
            lap: 2.4,
            sweep: 2.0,
            fade: 0.6,
            speed: 1.0,
            angle: None,
            count: 1,
        }
    }

    /// Of `kind`, with that kind's length and glow (where they were the
    /// last kind's).
    pub fn kind(self, kind: BeamKind) -> Self {
        let (l0, g0) = self.kind.defaults();
        let (l1, g1) = kind.defaults();
        Self {
            kind,
            length: if self.length == l0 { l1 } else { self.length },
            glow: if self.glow == g0 { g1 } else { self.glow },
            ..self
        }
    }

    pub fn shape(self, shape: BeamShape) -> Self {
        Self { shape, ..self }
    }

    pub fn direction(self, direction: BeamDirection) -> Self {
        Self { direction, ..self }
    }

    /// Counter-clockwise (a line beam from the other end).
    pub fn reversed(self) -> Self {
        self.direction(BeamDirection::CounterClockwise)
    }

    /// A line beam along `edge`.
    pub fn edge(self, edge: BeamEdge) -> Self {
        Self {
            edge,
            ..self.kind(BeamKind::Line)
        }
    }

    pub fn palette(self, palette: impl Into<BeamPalette>) -> Self {
        Self {
            palette: palette.into(),
            ..self
        }
    }

    /// Through `colors`, tail to head: one fades in and out along the beam.
    pub fn colors(self, colors: &[Rgba]) -> Self {
        let palette = match colors {
            [one] => Some(BeamPalette::single(*one)),
            _ => BeamPalette::colors(colors),
        };
        Self {
            palette: palette.unwrap_or(self.palette),
            ..self
        }
    }

    /// One colour, fading in and out along the beam.
    pub fn color(self, color: Rgba) -> Self {
        self.palette(BeamPalette::single(color))
    }

    pub fn length(self, length: f32) -> Self {
        Self {
            length: length.clamp(0.02, 0.98),
            ..self
        }
    }

    pub fn glow(self, glow: f32) -> Self {
        Self {
            glow: glow.max(0.0),
            ..self
        }
    }

    /// The crisp line on the edge, in UI pixels.
    pub fn width(self, width: f32) -> Self {
        Self {
            width: width.max(0.0),
            ..self
        }
    }

    pub fn intensity(self, intensity: f32) -> Self {
        Self {
            intensity: intensity.clamp(0.0, 1.0),
            ..self
        }
    }

    /// Seconds for a lap, and for a line beam's there and back.
    pub fn duration(self, seconds: f32) -> Self {
        Self {
            lap: seconds.max(0.01),
            sweep: seconds.max(0.01),
            ..self
        }
    }

    /// Seconds for a lap, and for a line beam's there and back, apart.
    pub fn timing(self, lap: f32, sweep: f32) -> Self {
        Self {
            lap: lap.max(0.01),
            sweep: sweep.max(0.01),
            ..self
        }
    }

    pub fn fade(self, seconds: f32) -> Self {
        Self {
            fade: seconds.max(0.0),
            ..self
        }
    }

    /// 1 as timed, 2 twice as fast, 0 still.
    pub fn speed(self, speed: f32) -> Self {
        Self {
            speed: speed.max(0.0),
            ..self
        }
    }

    /// Where the head starts, degrees clockwise from straight up. With a
    /// speed of 0, where it stays.
    pub fn angle(self, degrees: f32) -> Self {
        Self {
            angle: Some(degrees),
            ..self
        }
    }

    /// `n` comets, spaced evenly round (1 to 8).
    pub fn count(self, n: u8) -> Self {
        Self {
            count: n.clamp(1, 8),
            ..self
        }
    }

    /// Laps (or sweeps) a second.
    fn rate(&self) -> f32 {
        let period = match self.kind {
            BeamKind::Line => self.sweep,
            _ => self.lap,
        };
        self.speed / period.max(0.01)
    }

    /// The corner radius as the shader takes it, for a box styled `s`:
    /// pixels; -1 the circle inside; -2 - f a share f of the shorter side.
    fn radius_code(&self, s: &Style) -> f32 {
        match self.shape {
            BeamShape::Rounded(r) => r.max(0.0),
            BeamShape::Capsule => 1.0e4,
            BeamShape::Circle => -1.0,
            BeamShape::Auto => match s.border_radius {
                Length::Px(r) => r.max(0.0),
                Length::Percent(p) => -2.0 - (p / 100.0).max(0.0),
                _ => 0.0,
            },
        }
    }
}

impl Default for BeamStyle {
    fn default() -> Self {
        BeamStyle::new(BeamKind::Large)
    }
}

impl Ui {
    /// Puts a beam round `node`'s border (fading in, if it is on screen
    /// already), or takes it off with `None` (fading out). See the
    /// [module docs](crate::border_beam).
    pub fn set_border_beam(&mut self, node: NodeId, beam: impl Into<Option<BeamStyle>>) {
        let beam = beam.into();
        self.update_style(node, |s| s.border_beam = beam);
    }

    /// Changes `node`'s beam, putting one on first if it has none: it
    /// carries on from where it is round.
    pub fn update_border_beam(&mut self, node: NodeId, f: impl FnOnce(BeamStyle) -> BeamStyle) {
        let beam = f(self.style(node).border_beam.unwrap_or_default());
        self.set_border_beam(node, beam);
    }

    /// `node`'s beam as styled now (its state styles included).
    pub fn border_beam(&self, node: NodeId) -> Option<&BeamStyle> {
        self.style(node).border_beam.as_ref()
    }

    /// The beams' effects brought up to date with the styles: put on
    /// where a style has one, changed where it changed, faded out where it
    /// went. Run before each layout.
    pub(crate) fn sync_beams(&mut self) {
        let reduce = self.env.reduce_motion;
        let mut todo = vec![];
        for (i, n) in self.nodes.iter().enumerate() {
            if n.removed {
                continue;
            }
            let id = NodeId(i, n.generation);
            let had = self.env.beams.get(&id).copied();
            let want = n.style.border_beam;
            let code = want
                .or(had.map(|h| h.style))
                .map(|b| b.radius_code(&n.style));
            match (want, had, code) {
                (None, None, _) | (_, _, None) => {}
                (Some(b), Some(h), Some(c))
                    if h.on && h.style == b && h.still == reduce && h.radius == c => {}
                (None, Some(h), _) if !h.on => {}
                (want, had, Some(c)) => {
                    let unseen = had.is_none() && n.rect.width == 0.0 && n.rect.height == 0.0;
                    todo.push((id, want, had, c, unseen));
                }
            }
        }
        for (id, want, had, radius, unseen) in todo {
            let (style, on) = match (want, had) {
                (Some(b), _) => (b, true),
                (None, Some(h)) => (h.style, false),
                (None, None) => continue,
            };
            if apply(self, id, &style, radius, on, unseen) {
                self.env.beams.insert(
                    id,
                    BeamState {
                        style,
                        on,
                        still: reduce,
                        radius,
                    },
                );
            }
        }
    }

    /// Beams that have faded out, or whose boxes went, forgotten (their
    /// effects too, so they cost nothing).
    pub(crate) fn step_beams(&mut self) {
        if self.env.beams.is_empty() {
            return;
        }
        let now = self.clock as f32;
        let done: Vec<NodeId> = self
            .env
            .beams
            .iter()
            .filter(|(id, st)| {
                !self.contains(**id)
                    || (!st.on
                        && self
                            .effect(**id)
                            .is_none_or(|e| e.until.is_some_and(|t| now >= t)))
            })
            .map(|(id, _)| *id)
            .collect();
        for id in done {
            self.env.beams.remove(&id);
            if self.contains(id) && self.effect(id).is_some_and(is_beam) {
                self.set_effect(id, None);
            }
        }
    }
}

/// A beam as last put on a box's effect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BeamState {
    style: BeamStyle,
    /// Showing or fading in; else fading out.
    on: bool,
    /// Put on under reduced motion.
    still: bool,
    /// The radius it was put on with.
    radius: f32,
}

/// A beam on a box, to turn on and off and restyle; the same as setting
/// its style ([`Ui::set_border_beam`]), keeping the beam's style while it
/// is off.
#[derive(Debug, Clone)]
pub struct BorderBeam {
    /// The box it runs round: put what it holds in here.
    pub node: NodeId,
    pub style: BeamStyle,
}

impl BorderBeam {
    /// A beam on `node`, a box you already have: shown at once if the box
    /// has not been laid out yet, else fading in.
    pub fn on(ui: &mut Ui, node: NodeId, style: BeamStyle) -> BorderBeam {
        ui.set_border_beam(node, style);
        BorderBeam { node, style }
    }

    /// Fades it in, or out; once out it costs nothing.
    pub fn set_active(&self, ui: &mut Ui, active: bool) {
        if self.is_active(ui) != active {
            ui.set_border_beam(self.node, active.then_some(self.style));
        }
    }

    /// Whether it is on (or fading in).
    pub fn is_active(&self, ui: &Ui) -> bool {
        ui.border_beam(self.node).is_some()
    }

    /// How fast it goes (1 as timed): it speeds up or slows down from
    /// where it is, without jumping.
    pub fn set_speed(&mut self, ui: &mut Ui, speed: f32) {
        self.set_style(ui, self.style.speed(speed));
    }

    /// Restyles it, carrying on from where it is round, on or off as it was.
    pub fn set_style(&mut self, ui: &mut Ui, style: BeamStyle) {
        self.style = style;
        if self.is_active(ui) {
            ui.set_border_beam(self.node, style);
        }
    }
}

/// A box with a dark surface and a hairline border, a beam running round
/// it, in `parent`, its corners those of `style`'s shape (26 pixels for
/// [`BeamShape::Auto`]). Put what it holds in [`BorderBeam::node`]; it is as
/// big as that, as any box is, and can be restyled as any box can.
pub fn border_beam(ui: &mut Ui, parent: NodeId, style: BeamStyle) -> BorderBeam {
    let node = ui.add_box(
        parent,
        style! { position: relative; background: #1c1c1c; border: 1px solid rgba(255, 255, 255, 0.08); },
    );
    ui.update_style(node, |s| {
        s.border_radius = match style.shape {
            BeamShape::Auto => Length::Px(26.0),
            BeamShape::Rounded(r) => Length::Px(r.max(0.0)),
            BeamShape::Capsule | BeamShape::Circle => Length::Px(1.0e4),
        };
        if style.shape == BeamShape::Circle {
            s.aspect_ratio = 1.0;
        }
    });
    BorderBeam::on(ui, node, style)
}

thread_local! {
    static SHADER: OnceCell<Shader> = const { OnceCell::new() };
}

/// The beam's effect: the beam over what the box draws.
fn shader() -> Shader {
    SHADER.with(|s| s.get_or_init(|| Shader::new(source())).clone())
}

fn is_beam(e: &Effect) -> bool {
    e.shader == shader()
}

/// The effect's WGSL.
pub fn source() -> String {
    format!(
        "{}
fn border_beam(p: vec2<f32>, color: vec4<f32>, size: vec2<f32>, time: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, c3: vec4<f32>, c4: vec4<f32>, geom: vec4<f32>, look: vec4<f32>, run: vec4<f32>) -> vec4<f32> {{
    let beam = bb_beam(p, size, time, c0, c1, c2, c3, c4, geom, look, run);
    return beam + color * (1.0 - beam.a);
}}
",
        include_str!("border_beam.wgsl")
    )
}

/// Puts the beam's effect on `node`, `on` (fading in) or off (fading out),
/// carrying on from the one there: where round it is, and how far faded.
/// `unseen`: shown or gone at once, not faded. `false` when the box has an
/// effect of its own, which it keeps.
fn apply(ui: &mut Ui, node: NodeId, st: &BeamStyle, radius: f32, on: bool, unseen: bool) -> bool {
    let old = match ui.effect(node) {
        Some(e) if !is_beam(e) => return false,
        e => e.cloned(),
    };
    let now = ui.clock as f32;
    let fade = st.fade.max(0.0);
    let (phase, level) = match &old {
        Some(e) => {
            let (run, look) = (e.get("run").unwrap(), e.get("look").unwrap());
            let level = if unseen {
                f32::from(u8::from(on))
            } else {
                fade_level(run, now)
            };
            // Where round it is; back to the start at a new angle.
            let angle = st.angle.map_or(0.0, |a| a / 360.0);
            let phase = if look[3] == angle {
                (now * run[0] + run[1]).rem_euclid(1.0)
            } else {
                0.0
            };
            (phase, level)
        }
        // A new one starts somewhere of its own unless told where, so
        // beams side by side do not run in step.
        None => (
            if st.angle.is_some() { 0.0 } else { seed(node) },
            if on && !unseen {
                0.0
            } else {
                f32::from(u8::from(on))
            },
        ),
    };
    let rate = if ui.reduce_motion() { 0.0 } else { st.rate() };
    let offset = (phase - now * rate).rem_euclid(1.0);
    // How far into the fade towards `on` it already is.
    let into = smoothstep_inverse(if on { level } else { 1.0 - level });
    let start = now - into * fade;
    let run = [
        rate,
        offset,
        start,
        if on { fade + 1.0 } else { -(fade + 1.0) },
    ];
    let [c0, c1, c2, c3, c4] = st.palette.0.map(linear);
    let dir = match st.direction {
        BeamDirection::Clockwise => 1.0,
        BeamDirection::CounterClockwise => -1.0,
    };
    let edge = match (st.kind, st.edge) {
        (BeamKind::Line, BeamEdge::Top) => 1.0,
        (BeamKind::Line, BeamEdge::Bottom) => 2.0,
        (BeamKind::Line, BeamEdge::Left) => 3.0,
        (BeamKind::Line, BeamEdge::Right) => 4.0,
        _ => 0.0,
    };
    let mut effect = Effect::new(&shader())
        .arg("c0", c0)
        .arg("c1", c1)
        .arg("c2", c2)
        .arg("c3", c3)
        .arg("c4", c4)
        .arg(
            "geom",
            [
                radius,
                dir * f32::from(st.count.clamp(1, 8)),
                edge,
                st.width,
            ],
        )
        .arg(
            "look",
            [
                st.length,
                st.glow,
                st.intensity,
                st.angle.map_or(0.0, |a| a / 360.0),
            ],
        )
        .arg("run", run);
    // Off, or not moving: still once the fade is over.
    if !on || rate == 0.0 {
        effect = effect.until((start + fade).max(now));
    }
    if old.as_ref() != Some(&effect) {
        ui.set_effect(node, Some(effect));
    }
    true
}

/// How faded in a beam is at `now`, 0 to 1, from its `run` (as the shader
/// works it out).
fn fade_level(run: [f32; 4], now: f32) -> f32 {
    let fade = run[3].abs() - 1.0;
    let x = ((now - run[2]) / fade.max(1.0e-4)).clamp(0.0, 1.0);
    let eased = x * x * (3.0 - 2.0 * x);
    if run[3] > 0.0 {
        eased
    } else {
        1.0 - eased
    }
}

/// Where along `x * x * (3 - 2x)` the value `y` is.
fn smoothstep_inverse(y: f32) -> f32 {
    let y = y.clamp(0.0, 1.0);
    0.5 - ((1.0 - 2.0 * y).asin() / 3.0).sin()
}

/// Somewhere round the border of its own for `node`, 0 to 1.
fn seed(node: NodeId) -> f32 {
    let h = (node.0 as u32 ^ node.1.rotate_left(16)).wrapping_mul(0x9e37_79b9);
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// `c` decoded: effects mix and blend in linear light.
fn linear(c: Rgba) -> [f32; 4] {
    use threers_math::srgb::srgb_to_linear_f32 as d;
    [d(c[0]), d(c[1]), d(c[2]), c[3]]
}

/// Sets the `border-beam` property `name` (the shorthand or a longhand)
/// from CSS text; `None` when the value does not read.
pub(crate) fn css(s: &mut Style, name: &str, v: &str) -> Option<()> {
    use crate::css::{color, px, seconds, words};
    let v = v.trim();
    if name == "border-beam" {
        if v == "none" {
            s.border_beam = None;
            return Some(());
        }
        let w = words(v);
        let k = w.iter().find_map(|w| kind(w));
        let mut b = BeamStyle::new(k.unwrap_or_default());
        let mut colours = vec![];
        for word in w {
            if kind(word).is_some() {
                continue;
            }
            if let Some(d) = direction(word) {
                b = b.direction(d);
            } else if let Some(e) = edge(word) {
                b = b.edge(e);
            } else if let Some(p) = BeamPalette::named(word) {
                b = b.palette(p);
            } else if let Some(t) = seconds(word) {
                b = b.duration(t);
            } else {
                colours.push(color(word)?);
            }
        }
        if !colours.is_empty() {
            b = b.colors(&colours);
        }
        s.border_beam = Some(b);
        return Some(());
    }
    let b = s.border_beam.unwrap_or_default();
    let number = |v: &str| v.parse::<f32>().ok().filter(|n| n.is_finite());
    let share = |v: &str| match v.strip_suffix('%') {
        Some(p) => number(p.trim()).map(|p| p / 100.0),
        None => number(v),
    };
    let b = match name.strip_prefix("border-beam-")? {
        "kind" => b.kind(kind(v)?),
        "colors" | "color" => match BeamPalette::named(v) {
            Some(p) => b.palette(p),
            None => b.colors(
                &words(v)
                    .into_iter()
                    .map(color)
                    .collect::<Option<Vec<_>>>()?,
            ),
        },
        "speed" => b.speed(number(v)?),
        "duration" => b.duration(seconds(v)?),
        "width" => b.width(px(v)?),
        "length" => b.length(share(v)?),
        "glow" => b.glow(px(v)?),
        "intensity" => b.intensity(share(v)?),
        "angle" => b.angle(degrees(v)?),
        "count" => b.count(v.parse::<u8>().ok()?),
        "direction" => b.direction(direction(v)?),
        "edge" => b.edge(edge(v)?),
        "fade" => b.fade(seconds(v)?),
        "shape" => b.shape(match v {
            "auto" => BeamShape::Auto,
            "capsule" => BeamShape::Capsule,
            "circle" => BeamShape::Circle,
            r => BeamShape::Rounded(px(r)?),
        }),
        _ => return None,
    };
    s.border_beam = Some(b);
    Some(())
}

fn kind(v: &str) -> Option<BeamKind> {
    Some(match v {
        "large" => BeamKind::Large,
        "small" => BeamKind::Small,
        "line" => BeamKind::Line,
        _ => return None,
    })
}

fn direction(v: &str) -> Option<BeamDirection> {
    Some(match v {
        "clockwise" | "cw" => BeamDirection::Clockwise,
        "counter-clockwise" | "counterclockwise" | "ccw" | "reverse" => {
            BeamDirection::CounterClockwise
        }
        _ => return None,
    })
}

fn edge(v: &str) -> Option<BeamEdge> {
    Some(match v {
        "top" => BeamEdge::Top,
        "bottom" => BeamEdge::Bottom,
        "left" => BeamEdge::Left,
        "right" => BeamEdge::Right,
        _ => return None,
    })
}

/// A CSS angle in degrees: `90deg`, `0.25turn`, `1.5rad`, or a bare number
/// of degrees.
fn degrees(v: &str) -> Option<f32> {
    let n = |s: &str| s.trim().parse::<f32>().ok().filter(|n| n.is_finite());
    if let Some(d) = v.strip_suffix("deg") {
        n(d)
    } else if let Some(t) = v.strip_suffix("turn") {
        n(t).map(|t| t * 360.0)
    } else if let Some(r) = v.strip_suffix("rad") {
        n(r).map(f32::to_degrees)
    } else {
        n(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Scene;
    use crate::FixedWidthText;

    fn ui_with(style: Style) -> (Ui, NodeId) {
        let mut ui = Ui::new();
        let root = ui.root();
        let n = ui.add_box(root, style);
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        (ui, n)
    }

    /// Draws, then moves the clock on and lays out: whether a frame is
    /// needed is then whether the picture changed.
    fn frame(ui: &mut Ui, dt: f32) {
        ui.paint(&mut Scene::default().recorder());
        ui.step(dt);
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    }

    fn run(ui: &Ui, n: NodeId) -> [f32; 4] {
        ui.effect(n).and_then(|e| e.get("run")).unwrap()
    }

    #[test]
    fn the_shader_has_the_shape_of_an_effect() {
        let s = shader();
        assert_eq!(s.error(), None);
        assert_eq!(s.function(), "border_beam");
        assert!(s.animated());
    }

    #[test]
    fn css_sets_every_setting() {
        let (ui, n) = ui_with(crate::style! {
            width: 200px; height: 80px;
            border-beam: line neon top 3s counter-clockwise;
            border-beam-speed: 1.5;
            border-beam-width: 2.5px;
            border-beam-length: 40%;
            border-beam-glow: 12px;
            border-beam-intensity: 80%;
            border-beam-angle: 0.25turn;
            border-beam-count: 3;
            border-beam-fade: 200ms;
            border-beam-shape: capsule;
        });
        let b = ui.border_beam(n).unwrap();
        assert_eq!(b.kind, BeamKind::Line);
        assert_eq!(b.edge, BeamEdge::Top);
        assert_eq!(b.palette, BeamPalette::NEON);
        assert_eq!(b.direction, BeamDirection::CounterClockwise);
        assert_eq!((b.lap, b.sweep, b.speed), (3.0, 3.0, 1.5));
        assert_eq!(
            (b.width, b.length, b.glow, b.intensity),
            (2.5, 0.4, 12.0, 0.8)
        );
        assert_eq!((b.angle, b.count, b.fade), (Some(90.0), 3, 0.2));
        assert_eq!(b.shape, BeamShape::Capsule);
        // Colours in the shorthand; a longhand alone turns it on.
        let mut s = Style::default();
        assert!(crate::css::try_apply(
            &mut s,
            "border-beam",
            "small #f00 #00f"
        ));
        let b = s.border_beam.unwrap();
        assert_eq!(
            (b.kind, b.palette.0[0], b.palette.0[4]),
            (BeamKind::Small, [1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0])
        );
        let mut s = Style::default();
        assert!(crate::css::try_apply(
            &mut s,
            "border-beam-colors",
            "aurora"
        ));
        assert_eq!(s.border_beam.unwrap().palette, BeamPalette::AURORA);
        assert!(crate::css::try_apply(&mut s, "border-beam", "none"));
        assert_eq!(s.border_beam, None);
        assert!(!crate::css::try_apply(&mut s, "border-beam", "wobbly"));
        assert!(!crate::css::try_apply(&mut s, "border-beam-angle", "north"));
    }

    #[test]
    fn tailwind_classes_set_it() {
        let mut ui = Ui::new();
        let root = ui.root();
        let n = ui.add_tw(
            root,
            "w-64 h-24 rounded-xl beam-sm beam-neon beam-speed-150 beam-w-2 beam-angle-[-45deg] beam-count-2 beam-reverse beam-duration-3000",
        );
        let b = *ui.border_beam(n).unwrap();
        assert_eq!(
            (b.kind, b.palette, b.speed, b.width),
            (BeamKind::Small, BeamPalette::NEON, 1.5, 2.0)
        );
        assert_eq!(
            (b.angle, b.count, b.direction, b.lap),
            (Some(-45.0), 2, BeamDirection::CounterClockwise, 3.0)
        );
        let one = ui.add_tw(root, "beam-line beam-bottom beam-sky-400 beam-glow-8 beam-length-30 beam-intensity-50 beam-fade-100");
        let b = *ui.border_beam(one).unwrap();
        assert_eq!(
            (b.kind, b.edge, b.glow, b.length, b.intensity, b.fade),
            (BeamKind::Line, BeamEdge::Bottom, 8.0, 0.3, 0.5, 0.1)
        );
        let none = ui.add_tw(root, "beam beam-none");
        assert!(ui.border_beam(none).is_none());
        assert!(ui.try_tw(none, "beam-wobble-[x]").is_err());
    }

    #[test]
    fn a_hover_style_fades_it_in_and_out_and_then_it_is_gone() {
        use crate::input::{Event, Input};
        let (mut ui, n) =
            ui_with(crate::style! { width: 200px; height: 80px; border-radius: 12px; });
        ui.hover_style(n, "border-beam: large");
        frame(&mut ui, 0.1);
        assert!(ui.effect(n).is_none(), "no beam, no effect");
        let mut input = Input::new();
        input.dispatch(&mut ui, &mut (), Event::PointerMove { x: 50.0, y: 40.0 });
        frame(&mut ui, 0.0);
        assert!(ui.effect(n).is_some_and(is_beam));
        let r = run(&ui, n);
        assert!(
            r[3] > 0.0 && fade_level(r, ui.clock as f32) < 0.01,
            "fading in from nothing"
        );
        frame(&mut ui, 0.1);
        assert!(ui.needs_frame());
        input.dispatch(&mut ui, &mut (), Event::PointerMove { x: 350.0, y: 250.0 });
        frame(&mut ui, 0.0);
        assert!(run(&ui, n)[3] < 0.0, "fading out");
        for _ in 0..10 {
            frame(&mut ui, 0.1);
        }
        assert!(ui.effect(n).is_none(), "faded: the effect is gone");
        assert!(!ui.needs_frame());
    }

    #[test]
    fn set_before_layout_it_shows_at_once() {
        let mut ui = Ui::new();
        let root = ui.root();
        let n = ui.add_box(root, crate::style! { width: 100px; height: 40px; });
        ui.set_border_beam(n, BeamStyle::default());
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        assert_eq!(fade_level(run(&ui, n), ui.clock as f32), 1.0);
    }

    #[test]
    fn a_new_speed_carries_on_from_where_it_is() {
        let (mut ui, n) =
            ui_with(crate::style! { width: 200px; height: 80px; border-beam: large; });
        frame(&mut ui, 1.234);
        let at = |ui: &Ui| {
            let r = run(ui, n);
            (ui.clock as f32 * r[0] + r[1]).rem_euclid(1.0)
        };
        let before = at(&ui);
        ui.update_border_beam(n, |b| b.speed(3.0));
        frame(&mut ui, 0.0);
        assert!((at(&ui) - before).abs() < 1e-4);
        assert!((run(&ui, n)[0] - 3.0 / 2.4).abs() < 1e-6);
    }

    #[test]
    fn reduced_motion_holds_it_still_and_lets_it_go_again() {
        let (mut ui, n) =
            ui_with(crate::style! { width: 200px; height: 80px; border-beam: large; });
        ui.set_reduce_motion(true);
        frame(&mut ui, 0.0);
        frame(&mut ui, 0.1);
        assert_eq!(run(&ui, n)[0], 0.0);
        assert!(!ui.needs_frame(), "shown, and still");
        ui.set_reduce_motion(false);
        frame(&mut ui, 0.1);
        assert!(run(&ui, n)[0] > 0.0);
    }

    #[test]
    fn the_shape_follows_the_box() {
        let (ui, n) = ui_with(
            crate::style! { width: 200px; height: 80px; border-radius: 14px; border-beam: large; },
        );
        assert_eq!(ui.effect(n).unwrap().get("geom").unwrap()[0], 14.0);
        let (ui, n) = ui_with(
            crate::style! { width: 80px; height: 80px; border-radius: 50%; border-beam: small; },
        );
        assert_eq!(ui.effect(n).unwrap().get("geom").unwrap()[0], -2.5);
        let (mut ui, n) = ui_with(
            crate::style! { width: 80px; height: 80px; border-radius: 4px; border-beam: small; },
        );
        ui.update_style(n, |s| s.border_radius = Length::Px(9.0));
        frame(&mut ui, 0.0);
        assert_eq!(
            ui.effect(n).unwrap().get("geom").unwrap()[0],
            9.0,
            "a new radius, followed"
        );
    }

    #[test]
    fn a_box_with_its_own_effect_keeps_it() {
        let (mut ui, n) = ui_with(crate::style! { width: 200px; height: 80px; });
        let own =
            Shader::new("fn f(p: vec2<f32>, color: vec4<f32>) -> vec4<f32> { return color; }");
        ui.set_effect(n, Some(Effect::new(&own)));
        ui.set_border_beam(n, BeamStyle::default());
        frame(&mut ui, 0.1);
        assert_eq!(ui.effect(n).unwrap().shader, own);
    }

    #[test]
    fn beams_side_by_side_start_apart_unless_given_an_angle() {
        let mut ui = Ui::new();
        let root = ui.root();
        let a = border_beam(&mut ui, root, BeamStyle::default());
        let b = border_beam(&mut ui, root, BeamStyle::default());
        let c = border_beam(&mut ui, root, BeamStyle::default().angle(90.0).speed(0.0));
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        assert_ne!(run(&ui, a.node)[1], run(&ui, b.node)[1]);
        let r = run(&ui, c.node);
        assert_eq!((r[0], r[1]), (0.0, 0.0));
        assert_eq!(ui.effect(c.node).unwrap().get("look").unwrap()[3], 0.25);
    }

    #[test]
    fn the_handle_turns_it_off_and_on() {
        let mut ui = Ui::new();
        let root = ui.root();
        let mut beam = border_beam(&mut ui, root, BeamStyle::default());
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        beam.set_active(&mut ui, false);
        assert!(!beam.is_active(&ui));
        beam.set_speed(&mut ui, 2.0);
        assert!(
            ui.border_beam(beam.node).is_none(),
            "restyled while off: still off"
        );
        beam.set_active(&mut ui, true);
        assert_eq!(ui.border_beam(beam.node).unwrap().speed, 2.0);
    }

    #[test]
    fn palettes_spread_over_five_stops() {
        let p = BeamPalette::colors(&[[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]).unwrap();
        assert_eq!(p.0[0], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(p.0[2], [0.5, 0.0, 0.5, 1.0]);
        assert_eq!(p.0[4], [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(
            BeamPalette::css(&["#f00"]).unwrap().0[2],
            [1.0, 0.0, 0.0, 1.0]
        );
        assert!(BeamPalette::css(&[]).is_none());
        let from: BeamPalette = crate::indicators::Palette::MONO.into();
        assert_eq!(from.0[4], [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn the_inverse_smoothstep_inverts() {
        for y in [0.0, 0.1, 0.5, 0.77, 1.0] {
            let x = smoothstep_inverse(y);
            assert!((x * x * (3.0 - 2.0 * x) - y).abs() < 1e-5, "{y}");
        }
    }
}
