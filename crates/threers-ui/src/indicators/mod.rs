//! Progress indicators. Plain ones in the theme's colours ([`progress_view`]:
//! a bar, a ring, the spokes of an activity indicator), and glowing ones
//! drawn by signed distance fields ([`indicator`]): a
//! [`Shape::Bar`] or a [`Shape::Ring`], each [`Depth::Flat`] (a neon
//! stroke in 2D) or [`Depth::Solid`] (a glass tube with glowing liquid in
//! it, shaded in 3D: the bar's tube exactly, from its circular sections;
//! the ring's torus ray-marched, tipped back so its depth shows). Colour
//! runs through three stops along the length, light spills past the box
//! as a glow, and a shimmer may run along the fill. With no value the
//! indicator is busy: a comet runs along the bar, an arc chases round the
//! ring.
//!
//! The GPU painter draws the whole indicator in one shader effect per box,
//! a few dozen instructions a pixel flat and a pair of short ray marches
//! solid. Painters that cannot run shaders (the 2D canvas, the CPU
//! overlay) draw the same indicator plainly: a gradient bar or a stroked
//! arc. Only a busy or shimmering indicator takes the time, so a still one
//! asks for no frames.
//!
//! ```
//! use threers_ui::indicators::{indicator, Depth, IndicatorStyle, Palette, Shape};
//! use threers_ui::{FixedWidthText, Role, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let style = IndicatorStyle::new(Shape::Ring).depth(Depth::Solid).palette(Palette::AURORA);
//! let ring = indicator(&mut ui, root, "Upload", style, Some(0.4));
//! ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
//! let sem = ui.semantics(ring.node).unwrap();
//! assert_eq!((sem.role, sem.value.as_deref()), (Role::ProgressBar, Some("40%")));
//! ring.set(&mut ui, None); // busy: an arc chasing round
//! assert!(ui.needs_frame());
//! ```

use crate::animation::{Animation, Property, Repeat, Value};
use crate::shader::{Effect, Shader};
use crate::style::Length;
use crate::{style, BoxShadow, NodeId, Rgba, Role, Semantics, Ui};

mod label;
mod palette;
mod plain;
mod presets;
mod shaders;
#[cfg(test)]
mod tests;
mod theme;

use label::{place_label, Host, LabelNode};
pub use label::{Label, LabelAlign};
use palette::hex;
pub use palette::Palette;
pub use plain::{progress_view, PlainKind, PlainStyle, ProgressView, SPOKES};
pub use presets::Preset;
use shaders::{linear, shader};
pub use shaders::{source, source_filled};
pub use theme::{indicator_themed, progress_view_themed, IndicatorTheme};

/// What an indicator is drawn as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Shape {
    /// Along a line, as wide as its parent lets it be.
    #[default]
    Bar,
    /// Round a circle, clockwise from the top.
    Ring,
    /// A battery: a body with a terminal at its end, filled to its charge,
    /// coloured by how full it is (the palette's start empty, its end
    /// full). `size` is its height; it is a little over twice as wide.
    Battery,
}

/// What fills an indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Fill {
    /// Light (flat) or glowing liquid (solid).
    #[default]
    Neon,
    /// Electricity: bolts of lightning crackling along the fill, striking
    /// anew many times a second, plasma in the tube and sparks at the head.
    /// Always moving; flat or solid alike.
    Electric,
    /// Fire: flames licking up off the fill (out from a ring), coloured by
    /// their heat through the palette, embers rising.
    Fire,
    /// Water: caustic light drifting through it, bubbles rising.
    Water,
    /// Plasma: swirling bands of the palette's colours, glowing.
    Plasma,
    /// Frost: cracked ice, sparkling.
    Frost,
    /// Your own: see [`Fill::custom`].
    Custom(CustomFill),
}

/// A fill of your own, made by [`Fill::custom`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CustomFill(u32);

impl Fill {
    /// A fill of your own: WGSL whose last function takes `(along: f32, up:
    /// f32, half: f32, env: f32, cover: f32, u: f32, now: f32, c0:
    /// vec4<f32>, c1: vec4<f32>, c2: vec4<f32>)` and returns the pixel's
    /// colour, premultiplied — `along` pixels along the fill, `up` pixels off
    /// its middle (up a bar, out from a ring) in a stroke `half` thick, `env`
    /// how far within the fill's run (0 past its ends), `cover` how much of
    /// the pixel the fill covers, `u` where along the whole track (0 to 1),
    /// the time and the palette. The [`shader_lib`](crate::shader_lib)
    /// (noise, colour, shapes, fills) is in front of it: start from
    /// `ui_fire` and the rest. The same source makes the same fill.
    ///
    /// ```
    /// use threers_ui::indicators::{indicator, Fill, IndicatorStyle, Shape};
    /// use threers_ui::Ui;
    ///
    /// // Stripes sliding along, in the palette's colours.
    /// let stripes = Fill::custom("
    ///     fn stripes(along: f32, up: f32, half: f32, env: f32, cover: f32, u: f32, now: f32,
    ///                c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>) -> vec4<f32> {
    ///         let band = step(0.5, fract((along + up) / (half * 2.0) - now));
    ///         return vec4<f32>(ui_ramp3(c0, c1, c2, u) * (0.7 + 0.3 * band), 1.0) * cover;
    ///     }");
    /// assert_eq!(stripes.custom_error(), None);
    /// let mut ui = Ui::new();
    /// let root = ui.root();
    /// indicator(&mut ui, root, "Busy", IndicatorStyle::new(Shape::Bar).fill(stripes), Some(0.5));
    /// assert!(Fill::custom("fn nope(p: vec2<f32>) -> vec4<f32> { return vec4<f32>(); }").custom_error().is_some());
    /// ```
    pub fn custom(source: &str) -> Fill {
        Fill::Custom(shaders::register_custom(source))
    }

    /// What is wrong with a custom fill's source, if anything is (its shape;
    /// the GPU painter reports what its compiler finds).
    pub fn custom_error(self) -> Option<String> {
        match self {
            Fill::Custom(c) => shaders::custom_error(c),
            _ => None,
        }
    }
}

impl CustomFill {
    pub(super) fn id(self) -> u32 {
        self.0
    }

    pub(super) fn new(id: u32) -> Self {
        Self(id)
    }
}

impl Fill {
    /// Every fill, in order.
    pub const ALL: [Fill; 6] = [
        Fill::Neon,
        Fill::Electric,
        Fill::Fire,
        Fill::Water,
        Fill::Plasma,
        Fill::Frost,
    ];

    /// Its number in the effects' shader (`Electric` and `Neon` have their
    /// own).
    pub(super) fn kind(self) -> i32 {
        match self {
            Fill::Fire => 1,
            Fill::Water => 2,
            Fill::Plasma => 3,
            Fill::Frost => 4,
            Fill::Neon | Fill::Electric | Fill::Custom(_) => 0,
        }
    }

    /// Whether it moves on its own (and so asks for frames).
    pub(super) fn moves(self) -> bool {
        self != Fill::Neon
    }
}

pub use crate::motion::Motion;
pub use crate::solid::Depth;

/// A motion's number in the shaders (a bar's roll is a tumble, 8).
fn motion_code(m: Motion, shape: Shape) -> f32 {
    match (m, shape) {
        (Motion::Roll, Shape::Bar) => 8.0,
        _ => m as u8 as f32,
    }
}

/// How an indicator looks. Start from [`IndicatorStyle::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndicatorStyle {
    pub shape: Shape,
    pub depth: Depth,
    pub palette: Palette,
    /// How much light spills past it, 0 for none; 1 is a strong glow.
    pub glow: f32,
    /// A band of light running along the fill (asks for frames while it
    /// shows).
    pub shimmer: bool,
    /// A bar's height, a ring's width, in UI pixels.
    pub size: f32,
    /// A ring's stroke, as a share of its radius (0.05 to 0.6).
    pub thickness: f32,
    /// How it moves on its own.
    pub motion: Motion,
    /// How fast (1 as designed) and how far (1 as designed) it moves.
    pub motion_speed: f32,
    pub motion_amount: f32,
    /// Colour by phase rather than blended: below the first share the
    /// palette's first colour, below the second its middle one, else its
    /// last (a battery red when low, amber, green). `None` blends.
    pub phases: Option<[f32; 2]>,
    /// What fills it.
    pub fill: Fill,
}

impl IndicatorStyle {
    /// A flat [`Palette::AURORA`] indicator with a moderate glow: a bar 10
    /// pixels high or a ring 96 across.
    pub fn new(shape: Shape) -> Self {
        let size = match shape {
            Shape::Bar => 10.0,
            Shape::Ring => 96.0,
            Shape::Battery => 28.0,
        };
        Self {
            shape,
            depth: Depth::Flat,
            palette: Palette::AURORA,
            glow: 0.7,
            shimmer: false,
            size,
            thickness: 0.22,
            motion: Motion::None,
            motion_speed: 1.0,
            motion_amount: 1.0,
            phases: None,
            fill: Fill::Neon,
        }
    }

    pub fn depth(self, depth: Depth) -> Self {
        Self { depth, ..self }
    }

    pub fn palette(self, palette: Palette) -> Self {
        Self { palette, ..self }
    }

    pub fn glow(self, glow: f32) -> Self {
        Self {
            glow: glow.max(0.0),
            ..self
        }
    }

    pub fn shimmer(self, shimmer: bool) -> Self {
        Self { shimmer, ..self }
    }

    pub fn size(self, size: f32) -> Self {
        Self {
            size: size.max(2.0),
            ..self
        }
    }

    pub fn thickness(self, thickness: f32) -> Self {
        Self {
            thickness: thickness.clamp(0.05, 0.6),
            ..self
        }
    }

    /// Coloured by phase: the palette's first colour below `low`, its
    /// middle one below `mid`, its last from there (`.phases(0.2, 0.5)`:
    /// red under 20%, amber under 50%, green).
    pub fn fill(self, fill: Fill) -> Self {
        Self { fill, ..self }
    }

    /// Filled with electricity (see [`Fill::Electric`]).
    pub fn electric(self) -> Self {
        self.fill(Fill::Electric)
    }

    /// On fire (see [`Fill::Fire`]).
    pub fn fire(self) -> Self {
        self.fill(Fill::Fire)
    }

    pub fn phases(self, low: f32, mid: f32) -> Self {
        Self {
            phases: Some([low.min(mid), mid.max(low)]),
            ..self
        }
    }

    /// The colours to give the shader at `value`: the palette, or the one
    /// colour of its phase.
    fn colours_at(&self, value: Option<f32>) -> [Rgba; 3] {
        match (self.phases, value) {
            (Some([low, mid]), Some(v)) => {
                let c = if v < low {
                    self.palette.0[0]
                } else if v < mid {
                    self.palette.0[1]
                } else {
                    self.palette.0[2]
                };
                [c, c, c]
            }
            _ => self.palette.0,
        }
    }

    pub fn motion(self, motion: Motion) -> Self {
        Self { motion, ..self }
    }

    /// How fast (1 as designed) and how far (1 as designed) it moves.
    pub fn motion_tuned(self, speed: f32, amount: f32) -> Self {
        Self {
            motion_speed: speed.max(0.0),
            motion_amount: amount.max(0.0),
            ..self
        }
    }

    /// How far its glow reaches past its box, in UI pixels.
    pub fn glow_reach(&self) -> f32 {
        let stroke = match self.shape {
            Shape::Bar | Shape::Battery => self.size * 0.5,
            Shape::Ring => self.size * 0.25 * self.thickness,
        };
        if self.glow <= 0.0 {
            1.0
        } else {
            (stroke * 4.0 + 6.0) * self.glow.min(1.5)
        }
    }

    /// How far a hop goes, in UI pixels.
    fn hop(&self) -> f32 {
        match self.shape {
            Shape::Bar => (self.size * 1.2).max(8.0),
            Shape::Ring | Shape::Battery => self.size * 0.3,
        }
    }

    /// How far it draws past its box, in UI pixels: its glow, and room to
    /// move.
    pub fn overflow(&self) -> f32 {
        let a = self.motion_amount;
        let moves = match self.motion {
            Motion::None | Motion::Spin | Motion::Rock | Motion::Roll => 0.0,
            Motion::Bounce => self.hop() * a + self.size * 0.1 * a,
            Motion::Buzz => self.hop() * 0.15 * a,
            // A bar grows along its length too: room for a long one.
            Motion::Pulse => {
                self.size * 0.1 * a
                    + if self.shape == Shape::Bar {
                        16.0 * a
                    } else {
                        0.0
                    }
            }
            Motion::Float => self.hop() * 0.3 * a + self.size * 0.04,
        };
        // Flames rise four times the stroke's half above it.
        let flames = if self.fill == Fill::Fire {
            self.stroke() * 2.0
        } else {
            0.0
        };
        self.glow_reach().max(flames) + moves
    }

    /// The stroke's width in UI pixels (a ring's tube, a bar's height).
    fn stroke(&self) -> f32 {
        match self.shape {
            Shape::Bar | Shape::Battery => self.size,
            Shape::Ring => self.size * 0.5 * self.thickness,
        }
    }
}

/// An indicator's nodes: [`set`](Self::set) moves it on.
#[derive(Debug, Clone)]
pub struct Indicator {
    /// Its box, with its role and value for assistive technology.
    pub node: NodeId,
    /// The box the effect draws.
    fx: NodeId,
    /// What painters without shaders show: the fill (a box, or an arc).
    fill: NodeId,
    pub style: IndicatorStyle,
    label: Option<LabelNode>,
}

/// An indicator in `parent` showing `value` (0 to 1), or busy with `None`.
/// A bar is as wide as its parent lets it be; give it a width to change
/// that. See the [module docs](self).
pub fn indicator(
    ui: &mut Ui,
    parent: NodeId,
    label: &str,
    style: IndicatorStyle,
    value: Option<f32>,
) -> Indicator {
    let node = ui.add_box(parent, style! { position: relative; flex-shrink: 0; });
    ui.update_style(node, |s| match style.shape {
        Shape::Bar => {
            s.height = Length::Px(style.size);
            s.align_self = Some(crate::AlignItems::Stretch);
        }
        Shape::Ring => {
            s.width = Length::Px(style.size);
            s.height = Length::Px(style.size);
        }
        Shape::Battery => {
            s.width = Length::Px(battery_width(style.size));
            s.height = Length::Px(style.size);
        }
    });
    ui.set_semantics(node, Semantics::new(Role::ProgressBar).label(label));
    let fx = ui.add_box(
        node,
        style! { position: absolute; left: 0px; top: 0px; width: 100%; height: 100%; },
    );
    let [c0, c1, c2] = style.palette.0;
    let fill = match style.shape {
        Shape::Bar => {
            let r = style.size * 0.5;
            ui.update_style(fx, |s| {
                s.border_radius = Length::Px(r);
                s.clip = true;
                s.background = Some([c0[0], c0[1], c0[2], 0.18]);
            });
            let fill = ui.add_box(fx, style! { position: absolute; top: 0px; height: 100%; });
            ui.update_style(fill, |s| {
                s.border_radius = Length::Px(r);
                s.background = Some(c1);
                s.background_gradient = vec![(0.0, c0), (0.5, c1), (1.0, c2)];
                if style.glow > 0.0 {
                    s.box_shadow = vec![BoxShadow {
                        x: 0.0,
                        y: 0.0,
                        blur: style.size * 1.5 * style.glow,
                        spread: 0.0,
                        color: [c1[0], c1[1], c1[2], 0.6],
                    }];
                }
            });
            fill
        }
        Shape::Battery => {
            // A box with a border, the charge in it, the terminal beside.
            let h = style.size;
            let t = (h * 0.08).max(1.0);
            let nub = h * 0.14;
            ui.update_style(fx, |s| {
                s.width = Length::Px(battery_width(h) - nub);
                s.border = crate::Edges {
                    top: t,
                    right: t,
                    bottom: t,
                    left: t,
                };
                s.border_color = [0.82, 0.85, 0.9, 0.85];
                s.border_radius = Length::Px(h * 0.24);
                s.padding = crate::Edges {
                    top: t,
                    right: t,
                    bottom: t,
                    left: t,
                };
                s.box_sizing = crate::BoxSizing::BorderBox;
                s.display = crate::Display::Flex;
            });
            let fill = ui.add_box(fx, style! { height: 100%; flex-shrink: 0; });
            ui.update_style(fill, |s| {
                s.border_radius = Length::Px((h * 0.24 - 2.0 * t).max(1.0));
                s.background = Some(c2);
            });
            let terminal = ui.add_box(node, style! { position: absolute; });
            ui.update_style(terminal, |s| {
                s.left = Length::Px(battery_width(h) - nub - t * 0.5);
                s.top = Length::Px(h * 0.3);
                s.width = Length::Px(nub + t * 0.5);
                s.height = Length::Px(h * 0.4);
                s.border_radius = Length::Px(nub * 0.45);
                s.background = Some([0.82, 0.85, 0.9, 0.85]);
            });
            ui.semantics_mut(terminal).hidden = true;
            fill
        }
        Shape::Ring => {
            // Arcs on a 100-unit view box: the stroke is its share of that.
            let stroke = 100.0 * style.stroke() / style.size;
            let r = 50.0 - stroke * 0.5 - 100.0 / style.size;
            let whole =
                || style! { position: absolute; left: 0px; top: 0px; width: 100%; height: 100%; };
            let track = ui.add_path(
                fx,
                whole(),
                arc(r, 0.0, 1.0),
                [0.0, 0.0, 100.0, 100.0],
                Some(stroke),
                false,
            );
            ui.update_style(track, |s| {
                s.color = [c0[0], c0[1], c0[2], 0.2];
                s.color_set = true;
            });
            let fill = ui.add_path(
                fx,
                whole(),
                vec![],
                [0.0, 0.0, 100.0, 100.0],
                Some(stroke),
                false,
            );
            ui.update_style(fill, |s| {
                s.color = c1;
                s.color_set = true;
            });
            fill
        }
    };
    for n in [fx, fill] {
        ui.semantics_mut(n).hidden = true;
    }
    let ind = Indicator {
        node,
        fx,
        fill,
        style,
        label: None,
    };
    ind.set(ui, value);
    ind
}

/// A busy bar's `fill`, a `share` of the bar `bar` wide, sliding along it
/// every `period` seconds: moved as drawn, so nothing is laid out again.
fn slide(ui: &mut Ui, bar: NodeId, fill: NodeId, share: f32, period: f32) {
    ui.update_style(fill, |s| {
        s.left = Length::Px(0.0);
        s.width = Length::Percent(share * 100.0);
    });
    if !ui.is_animating(fill) {
        let w = ui.rect(bar).width.max(100.0);
        ui.animate(
            fill,
            Animation::new(
                Property::TranslateX,
                Value::Number(-share * w),
                Value::Number(w),
                period,
            )
            .easing(threers_math::easing::Easing::SineInOut)
            .repeat(Repeat::Forever),
        );
    }
}

/// `fill` back where it is laid out.
fn stop_sliding(ui: &mut Ui, fill: NodeId) {
    ui.stop_animations(fill, Some(Property::TranslateX));
    ui.clear_transform3d(fill);
    ui.update_style(fill, |s| s.left = Length::Px(0.0));
}

/// A battery `height` high: how wide, its terminal included.
fn battery_width(height: f32) -> f32 {
    height * 2.15
}

/// The ring's arc of radius `r` on a 100-unit view box, from `start` round
/// `turns` (both in turns, clockwise from the top).
fn arc(r: f32, start: f32, turns: f32) -> Vec<String> {
    let tau = std::f32::consts::TAU;
    let point = |t: f32| (50.0 + r * (t * tau).sin(), 50.0 - r * (t * tau).cos());
    let turns = turns.clamp(0.0, 1.0);
    if turns <= 1e-4 {
        return vec![];
    }
    let (x0, y0) = point(start);
    if turns >= 1.0 - 1e-4 {
        let (xm, ym) = point(start + 0.5);
        return vec![format!(
            "M{x0:.3} {y0:.3} A{r} {r} 0 0 1 {xm:.3} {ym:.3} A{r} {r} 0 0 1 {x0:.3} {y0:.3}"
        )];
    }
    let (x1, y1) = point(start + turns);
    let large = u8::from(turns > 0.5);
    vec![format!(
        "M{x0:.3} {y0:.3} A{r} {r} 0 {large} 1 {x1:.3} {y1:.3}"
    )]
}

impl Indicator {
    /// How far along, 0 to 1; `None` while it is not known (busy).
    pub fn set(&self, ui: &mut Ui, value: Option<f32>) {
        let v = value.map(|v| v.clamp(0.0, 1.0));
        let sem = ui.semantics_mut(self.node);
        sem.range = v.map(|v| crate::a11y::RangeValue {
            min: 0.0,
            max: 100.0,
            now: v * 100.0,
        });
        sem.value = v.map(|v| format!("{}%", (v * 100.0).round()));
        if let Some(l) = &self.label {
            ui.set_text(l.text, l.label.render(v));
        }
        let st = self.style;
        let busy = v.is_none();
        let motion = if ui.reduce_motion() {
            Motion::None
        } else {
            st.motion
        };
        let light = ui.appearance_of(self.node) == crate::states::Appearance::Light;
        let animated = busy || st.shimmer || motion != Motion::None || st.fill.moves();
        let sh = shader(st.shape, st.depth, st.fill, animated);
        let [c0, c1, c2] = st.colours_at(v).map(linear);
        let flag = |b: bool| if b { 1.0 } else { 0.0 };
        let effect = Effect::new(&sh)
            .arg("value", v.unwrap_or(0.0))
            .arg("c0", c0)
            .arg("c1", c1)
            .arg("c2", c2)
            .arg("look", [st.glow, flag(st.shimmer), flag(busy), st.stroke()])
            .arg(
                "reach",
                if light {
                    -st.glow_reach()
                } else {
                    st.glow_reach()
                },
            )
            .arg(
                "motion",
                [
                    motion_code(motion, st.shape),
                    st.motion_speed,
                    st.motion_amount,
                    st.hop(),
                ],
            )
            .overflow(st.overflow());
        if ui.effect(self.fx) != Some(&effect) {
            ui.set_effect(self.fx, Some(effect));
        }
        // What painters without shaders draw.
        match st.shape {
            Shape::Bar => match v {
                Some(v) => {
                    stop_sliding(ui, self.fill);
                    ui.update_style(self.fill, |s| s.width = Length::Percent(v * 100.0));
                }
                None => slide(ui, self.node, self.fill, 0.35, 1.4),
            },
            Shape::Battery => {
                let level = v.unwrap_or(0.35);
                let c = if st.phases.is_some() {
                    st.colours_at(Some(level))[0]
                } else {
                    st.palette.at(level)
                };
                ui.update_style(self.fill, |s| {
                    s.width = Length::Percent(level * 100.0);
                    s.background = Some(c);
                });
            }
            Shape::Ring => {
                let stroke = 100.0 * st.stroke() / st.size;
                let r = 50.0 - stroke * 0.5 - 100.0 / st.size;
                // Busy: a still quarter arc (the arc's turning is the
                // shader's; a path cannot be animated round).
                let (start, turns) = v.map_or((0.0, 0.3), |v| (0.0, v));
                ui.set_path_data(self.fill, arc(r, start, turns));
            }
        }
    }

    /// The box drawn through the shader (for tests and inspection).
    pub fn effect_node(&self) -> NodeId {
        self.fx
    }

    /// With `label` on it (see [`Label`]): how far along, or any text.
    pub fn labelled(mut self, ui: &mut Ui, label: Label) -> Self {
        self.set_label(ui, Some(label));
        self
    }

    /// Puts `label` on it, or takes its label off.
    pub fn set_label(&mut self, ui: &mut Ui, label: Option<Label>) {
        let host = match self.style.shape {
            Shape::Bar | Shape::Battery => Host::Bar {
                height: self.style.size,
            },
            Shape::Ring => Host::Round {
                size: self.style.size,
            },
        };
        let value = self.current(ui);
        match label {
            Some(l) => {
                self.label = Some(place_label(
                    ui,
                    self.node,
                    host,
                    l,
                    self.label.as_ref(),
                    value,
                ))
            }
            None => {
                if let Some(o) = self.label.take() {
                    ui.remove(o.node);
                }
            }
        }
    }

    /// What its label says, if it has one.
    pub fn label_text(&self, ui: &Ui) -> Option<String> {
        self.label.as_ref().and_then(|l| match ui.kind(l.text) {
            crate::NodeKind::Text(t) => Some(t.clone()),
            _ => None,
        })
    }

    /// Its value as last set (`None`: busy).
    pub fn current(&self, ui: &Ui) -> Option<f32> {
        ui.semantics(self.node)
            .and_then(|s| s.range.map(|r| r.now / 100.0))
    }
}
