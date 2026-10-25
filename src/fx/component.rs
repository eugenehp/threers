//! What an element *is*: its shape, its content and how it responds —
//! [`Component`], with [`Button`] and [`Figure`].

use std::any::Any;

use super::Shape;
use crate::math::{Color, Vector2};
use crate::overlay::Overlay;

/// Where a component sits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Anchor {
    /// Centred on a point, in CSS pixels.
    At(Vector2),
    /// Centred on a fraction of the viewport: `(0.5, 0.5)` is its middle.
    Fraction(Vector2),
}

impl Anchor {
    pub const CENTER: Anchor = Anchor::Fraction(Vector2::new(0.5, 0.5));

    pub fn resolve(&self, viewport: Vector2) -> Vector2 {
        match self {
            Anchor::At(p) => *p,
            Anchor::Fraction(f) => Vector2::new(viewport.x * f.x, viewport.y * f.y),
        }
    }
}

/// What a component gets to lay itself out: the viewport, and the overlay to
/// measure text with.
pub struct Layout<'a> {
    pub viewport: Vector2,
    pub overlay: &'a mut Overlay,
}

/// What a component gets each frame, after the stage has hit-tested the
/// pointer against its shape.
#[derive(Debug, Clone, Copy)]
pub struct ComponentInput {
    pub dt: f32,
    /// The pointer is over the shape.
    pub hovered: bool,
    /// The pointer went down over the shape and has not been released.
    pub pressed: bool,
}

/// An element's body: a [`Shape`], whatever is drawn on it, and how it
/// reacts to the pointer. Pair one with an [`Effect`](super::Effect) on a
/// [`Stage`](super::Stage).
///
/// Implement this for a new kind of element — a toggle, a slider thumb, an
/// avatar — and every effect works on it.
pub trait Component: Any {
    /// Place and size yourself for this viewport. Called every frame before
    /// anything else; cheap components need not cache.
    fn layout(&mut self, _layout: &mut Layout) {}
    /// The shape this frame, in CSS pixels.
    fn shape(&self) -> Shape;
    /// React to the pointer: animate a press, a hover.
    fn update(&mut self, _input: &ComponentInput) {}
    /// Draw content over the effect's surface and under its detail — a label,
    /// an icon. `overlay` is in device pixels, `dpr` of them per CSS pixel.
    fn paint(&self, _overlay: &mut Overlay, _dpr: f32) {}
    /// Whether the pointer interacts with it. A purely decorative component
    /// returns `false` and raises no events.
    fn interactive(&self) -> bool {
        true
    }
}

/// A push button: a pill sized to its label, that gives like a spring when
/// pressed. The stage raises [`EventKind::Click`](super::EventKind::Click) on
/// release over it.
#[derive(Debug, Clone)]
pub struct Button {
    pub label: String,
    pub anchor: Anchor,
    /// Pill height, CSS pixels.
    pub height: f32,
    /// Space either side of the label.
    pub padding: f32,
    pub font_size: f32,
    pub font_weight: u16,
    pub text: Color,
    /// How far it shrinks while held: 0.96 is 4%.
    pub press_scale: f32,
    width: Option<f32>,
    center: Vector2,
    scale: f32,
    velocity: f32,
}

impl Button {
    /// A 64 px pill with a 20 px bold label, at the middle of the viewport.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            anchor: Anchor::CENTER,
            height: 64.0,
            padding: 36.0,
            font_size: 20.0,
            font_weight: 700,
            text: Color::from_hex(0xf4f4f4),
            press_scale: 0.96,
            width: None,
            center: Vector2::ZERO,
            scale: 1.0,
            velocity: 0.0,
        }
    }

    pub fn at(mut self, anchor: Anchor) -> Self {
        self.anchor = anchor;
        self
    }

    pub fn with_text_color(mut self, color: Color) -> Self {
        self.text = color;
        self
    }

    /// The current press scale, 1 at rest.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    fn rest(&self) -> Shape {
        Shape::pill(self.center, self.width.unwrap_or(self.height * 3.0), self.height)
    }
}

impl Component for Button {
    fn layout(&mut self, layout: &mut Layout) {
        if self.width.is_none() {
            // Measuring the label needs the text engine, which is the `captions` feature
            // — `Overlay::measure_text` and `text_weight` are gated on it. `fx` is not a
            // feature and so cannot imply one the way `ui` and `video` do, so a build
            // without it falls back to a width from the label's length. Before this, the
            // two calls were unguarded and the crate simply did not compile without
            // `captions`; the default build has it, which is why that went unnoticed.
            #[cfg(feature = "captions")]
            {
                layout.overlay.text_weight(self.font_weight);
                layout.overlay.text_size(self.font_size);
                self.width =
                    Some(layout.overlay.measure_text(&self.label).0 + self.padding * 2.0);
            }
            #[cfg(not(feature = "captions"))]
            {
                // Half the type size per character is the usual average for a UI face,
                // which keeps a button the right order of size without a measurement.
                self.width = Some(
                    self.label.chars().count() as f32 * self.font_size * 0.5
                        + self.padding * 2.0,
                );
            }
        }
        self.center = self.anchor.resolve(layout.viewport);
    }

    fn shape(&self) -> Shape {
        self.rest().scaled(self.scale)
    }

    fn update(&mut self, input: &ComponentInput) {
        // a stiff, underdamped spring, as `whileTap` gives
        let target = if input.pressed { self.press_scale } else { 1.0 };
        let dt = input.dt.min(1.0 / 30.0);
        self.velocity += ((target - self.scale) * 420.0 - self.velocity * 18.0) * dt;
        self.scale += self.velocity * dt;
    }

    fn paint(&self, o: &mut Overlay, dpr: f32) {
        let c = self.center;
        o.no_stroke();
        // As in `layout`: the label is drawn only where there is a text engine to draw
        // it with. The button's shape comes from `shape` either way.
        #[cfg(not(feature = "captions"))]
        let _ = (c, dpr);
        #[cfg(feature = "captions")]
        {
        o.text_weight(self.font_weight);
        o.text_size(self.font_size * dpr * self.scale);
        let (_, th) = o.measure_text(&self.label);
        let (x, y) = (c.x * dpr, c.y * dpr - th / 2.0);
        // the text shadow a button has in a page
        o.fill_alpha(Color::from_hex(0x000000), 0.75);
        o.text_centered(x, y + dpr, &self.label);
        o.fill(self.text);
        o.text_centered(x, y, &self.label);
        }
    }
}

/// Any shape, with nothing drawn on it: a logo, a blob, a decoration. With an
/// anchor it is re-centred there every layout; without, it stays where its
/// shape says.
#[derive(Debug, Clone)]
pub struct Figure {
    pub shape: Shape,
    pub anchor: Option<Anchor>,
    /// Whether it raises events.
    pub interactive: bool,
    placed: Shape,
}

impl Figure {
    pub fn new(shape: Shape) -> Self {
        Self { placed: shape.clone(), shape, anchor: None, interactive: true }
    }

    pub fn at(mut self, anchor: Anchor) -> Self {
        self.anchor = Some(anchor);
        self
    }

    /// A figure that ignores the pointer for events (effects still see it).
    pub fn decorative(mut self) -> Self {
        self.interactive = false;
        self
    }
}

impl Component for Figure {
    fn layout(&mut self, layout: &mut Layout) {
        self.placed = match self.anchor {
            Some(a) => self.shape.centered_at(a.resolve(layout.viewport)),
            None => self.shape.clone(),
        };
    }

    fn shape(&self) -> Shape {
        self.placed.clone()
    }

    fn interactive(&self) -> bool {
        self.interactive
    }
}
