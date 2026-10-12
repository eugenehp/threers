use crate::math::Color;

/// Area light shaped like a rectangle — a softbox, a window, a strip.
///
/// Shaded as an area light, not as an approximation: the diffuse term is the
/// exact form factor over the quad, and the specular is a representative point
/// on the rectangle with the lobe widened by the light's angular size, so a
/// large panel gives a large soft highlight and a narrow one gives a hard
/// line. Anisotropy and iridescence apply, because it goes through the same
/// BRDF as every other light.
///
/// It emits along its local **−Z** and is single-sided, like three.js: a
/// surface behind it receives nothing. Orient it with the object's rotation.
///
/// This is what to reach for when a punctual light is doing nothing. Metal has
/// no diffuse term — it mirrors what is around it rather than responding to a
/// light — so a zero-area source contributes a near-delta specular that a
/// roughness of 0.25 all but swallows. An area light is something a rail can
/// actually reflect.
#[derive(Debug, Clone, Copy)]
pub struct RectAreaLight {
    pub color: Color,
    pub intensity: f32,
    pub width: f32,
    pub height: f32,
}

impl Default for RectAreaLight {
    fn default() -> Self {
        Self::new(Color::WHITE, 1.0, 10.0, 10.0)
    }
}

impl RectAreaLight {
    pub const fn new(color: Color, intensity: f32, width: f32, height: f32) -> Self {
        Self {
            color,
            intensity,
            width,
            height,
        }
    }
}
