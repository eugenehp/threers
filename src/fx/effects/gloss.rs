//! A still, glossy surface: [`Gloss`].
//!
//! The shape in a [`Look`], its reflections leaning with the pointer and its
//! surface swelling gently toward it when it comes near — the venom style
//! without the hunt, for anything that should look like the same material
//! but keep to itself: cards, badges, icons beside a venom button.

use crate::fx::surface::{Mound, Surface};
use crate::fx::{Effect, EffectInput, Look, Outline, Shape};
use crate::math::Vector2;

#[derive(Debug, Clone, PartialEq)]
pub struct GlossStyle {
    pub look: Look,
    /// Deepest the round profile goes, in CSS pixels.
    pub bevel: f32,
    /// How far the surface bulges toward a pointer in range, in CSS pixels.
    /// 0 keeps it still.
    pub swell: f32,
    /// Distance from the outline at which it starts to swell.
    pub reach: f32,
}

impl Default for GlossStyle {
    fn default() -> Self {
        Self { look: Look::LACQUER, bevel: 32.0, swell: 6.0, reach: 120.0 }
    }
}

impl GlossStyle {
    pub fn with_look(mut self, look: Look) -> Self {
        self.look = look;
        self
    }
}

pub struct Gloss {
    pub style: GlossStyle,
    shape: Option<(Shape, Outline, f32)>,
    lean: f32,
    /// The bulge, a damped spring, and where along the outline it is.
    bulge: f32,
    bulge_v: f32,
    at: f32,
    toward: Vector2,
}

impl Gloss {
    pub fn new(style: GlossStyle) -> Self {
        Self { style, shape: None, lean: 0.0, bulge: 0.0, bulge_v: 0.0, at: 0.0, toward: Vector2::ZERO }
    }
}

impl Default for Gloss {
    fn default() -> Self {
        Self::new(GlossStyle::default())
    }
}

impl Effect for Gloss {
    fn update(&mut self, input: &EffectInput) {
        if self.shape.as_ref().is_none_or(|(s, ..)| s != input.shape) {
            let s = input.shape.clone();
            let outline = Outline::new(&s);
            let inradius = s.inradius();
            self.shape = Some((s, outline, inradius));
        }
        let (shape, outline, _) = self.shape.as_ref().expect("just set");
        let dt = input.dt.clamp(1.0 / 240.0, 1.0 / 30.0);
        let (lo, hi) = shape.bounds();
        let target_lean = input.pointer.map_or(0.0, |p| ((p.x - (lo.x + hi.x) * 0.5) / ((hi.x - lo.x) * 1.5).max(1.0)).clamp(-1.0, 1.0));
        self.lean += (target_lean - self.lean) * (dt * 4.0).min(1.0);

        let mut target = 0.0;
        if let Some(p) = input.pointer {
            let anchor = input.cursor.center(p);
            let d = shape.distance(anchor).max(0.0);
            if d < self.style.reach {
                target = self.style.swell * (1.0 - d / self.style.reach);
                self.at += outline.loop_diff(self.at, outline.nearest(anchor)) * (dt * 8.0).min(1.0);
                self.toward = anchor;
            }
        }
        self.bulge_v += ((target - self.bulge) * 160.0 - self.bulge_v * 12.0) * dt;
        self.bulge += self.bulge_v * dt;
    }

    fn surface(&self) -> Option<Surface> {
        let (shape, outline, inradius) = self.shape.as_ref()?;
        let mut surface = Surface::new_with_thickness(shape.clone(), self.style.look, inradius.min(self.style.bevel));
        surface.lean = self.lean;
        if self.bulge > 0.3 {
            let e = outline.at(self.at);
            let dir = (e.normal * 0.6 + (self.toward - e.point).normalize() * 0.4).normalize();
            let r = 10.0 + self.bulge;
            surface.mound = Some(Mound { from: e.point - e.normal * 6.0, to: e.point + dir * self.bulge, base_radius: r, tip_radius: r * 0.6 });
        }
        Some(surface)
    }
}
