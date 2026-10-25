//! Interactive effects on 2D elements: glossy liquids that react to the
//! pointer, on buttons, figures and anything else with a shape.
//!
//! Three ideas, each a trait or a type you can add to:
//!
//! | | |
//! |---|---|
//! | [`Component`] | what an element *is*: its [`Shape`], its content, how it responds — [`Button`], [`Figure`] |
//! | [`Effect`] | what it *wears*: a behaviour that draws a [`Surface`] and paints detail — [`Venom`], [`Gloss`] |
//! | [`Stage`] | the elements together: layout, hit-testing and [`Event`]s, the scene quads, the paint order, one [`Cursor`] |
//!
//! Any component wears any effect, so a new component works with every
//! effect and a new effect with every component:
//!
//! ```no_run
//! use threers::fx::{Anchor, Backdrop, Button, Figure, Gloss, GlossStyle, Look, Shape, Stage, Venom, VenomStyle};
//! use threers::math::Vector2;
//! use threers::sketch::{sketch, Frame, SketchOptions};
//!
//! sketch(SketchOptions::new("fx"), |s| {
//!     let mut stage = Stage::new();
//!     stage.set_backdrop(s.scene, Backdrop::STUDIO);
//!     let go = stage.add(s.scene, Button::new("let's go"), Venom::default());
//!     let star = Shape::star(Vector2::ZERO, 5, 70.0, 30.0, 6.0);
//!     stage.add(
//!         s.scene,
//!         Figure::new(star).at(Anchor::Fraction(Vector2::new(0.5, 0.25))),
//!         Venom::new(VenomStyle::default().with_look(Look::GOLD)),
//!     );
//!     stage.add(s.scene, Figure::new(Shape::circle(Vector2::new(120.0, 120.0), 40.0)), Gloss::new(GlossStyle::default().with_look(Look::CHROME)));
//!     move |f: &mut Frame| {
//!         for event in stage.sketch_frame(f) {
//!             if event.element == go && event.kind == threers::fx::EventKind::Click {
//!                 println!("clicked");
//!             }
//!         }
//!     }
//! });
//! ```
//!
//! Everything is in **CSS pixels** — logical, top-left origin, y down — the
//! way a UI is laid out; the device pixel ratio converts at the edges.
//!
//! # Adding an effect
//!
//! Implement [`Effect`]: step in [`update`](Effect::update) from the
//! element's shape and the pointer, and return a [`Surface`] — the shape plus
//! whatever grows out of it (strands, spikes, droplets, a mound), shaded in a
//! [`Look`] — and optionally paint detail on the overlay or decorate the
//! [`Cursor`]. [`Outline`] walks any shape's edge by arc length, for effects
//! that root things on it. The surface renderer is shared, so a new effect
//! draws on every backend the existing ones do.
//!
//! # Adding a component
//!
//! Implement [`Component`]: lay out against the viewport, return a [`Shape`],
//! animate on hover and press, paint content. The stage does the
//! hit-testing and raises the events.
//!
//! # Backends
//!
//! Surfaces are [`ShaderMaterial`](crate::ShaderMaterial)s with their WGSL
//! built on [`crate::sdf::WGSL`], so they run on wgpu everywhere — Metal,
//! Vulkan, DX12, WebGPU, and WebGL2 through the preamble's texture fallback
//! for their arrays. threers' own Metal renderer, which runs no WGSL,
//! recognises them and draws them with its port of the shader. The overlay
//! detail is a CPU canvas and goes wherever the overlay does.

mod component;
mod cursor;
pub(crate) mod draw;
pub mod effects;
mod look;
mod shape;
mod stage;
pub mod surface;

use std::any::Any;

pub use component::{Anchor, Button, Component, ComponentInput, Figure, Layout};
pub use cursor::{Cursor, CursorDecor};
pub use effects::{Gloss, GlossStyle, Venom, VenomState, VenomStyle};
pub use look::Look;
pub use shape::{Edge, Outline, Shape, MAX_POLYGON_POINTS};
pub use stage::{ElementId, Event, EventKind, Stage, StageInput};
pub use surface::{Backdrop, Surface};

use crate::math::Vector2;
use crate::overlay::Overlay;

/// What an effect gets each frame.
#[derive(Debug, Clone, Copy)]
pub struct EffectInput<'a> {
    /// The element's shape this frame, in CSS pixels.
    pub shape: &'a Shape,
    /// Seconds since the last frame.
    pub dt: f32,
    /// Seconds since the stage started.
    pub time: f32,
    /// The pointer's hotspot, or `None` when there is none.
    pub pointer: Option<Vector2>,
    /// What stands in for the pointer, and the grips on it.
    pub cursor: Cursor,
    /// The pointer is over the element.
    pub hovered: bool,
    /// The element is held down.
    pub pressed: bool,
}

/// A behaviour an element wears: it steps with the pointer and draws a
/// [`Surface`], and may paint detail over the element's content and decorate
/// the cursor. See the [module docs](self) for adding one.
pub trait Effect: Any {
    fn update(&mut self, input: &EffectInput);
    /// The surface to draw this frame, or `None` for an effect that only
    /// paints.
    fn surface(&self) -> Option<Surface> {
        None
    }
    /// Detail over the element's content and under the cursor. `overlay` is
    /// in device pixels, `dpr` of them per CSS pixel.
    fn paint(&self, _overlay: &mut Overlay, _dpr: f32) {}
    /// Leave something on the cursor — a coat of the surface on the glove.
    fn decorate_cursor(&self, _decor: &mut CursorDecor) {}
    /// Detail over the cursor — strands lying on the glove.
    fn paint_over_cursor(&self, _overlay: &mut Overlay, _dpr: f32) {}
}
