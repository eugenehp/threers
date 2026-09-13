//! Lights. Mirror three.js's light hierarchy: ambient, directional, point,
//! spot, hemisphere, rect-area. Lights live in the scene graph as
//! `ObjectKind::Light` variants and contribute to the renderer's per-frame
//! lighting uniform buffer.

mod ambient;
mod directional;
mod hemisphere;
mod point;
mod rect_area;
mod shadow;
mod spot;

pub use ambient::AmbientLight;
pub use directional::DirectionalLight;
pub use hemisphere::HemisphereLight;
pub use point::PointLight;
pub use rect_area::RectAreaLight;
pub use shadow::ShadowSettings;
pub use spot::SpotLight;

use crate::math::Color;

/// Unified light type. Wrap an `AmbientLight`/`DirectionalLight`/etc. and add
/// it to the scene via `Object3D::light(...)`.
#[derive(Debug, Clone)]
pub enum Light {
    Ambient(AmbientLight),
    Directional(DirectionalLight),
    Point(PointLight),
    Spot(SpotLight),
    Hemisphere(HemisphereLight),
    RectArea(RectAreaLight),
}

impl Light {
    pub fn color(&self) -> Color {
        match self {
            Light::Ambient(l) => l.color,
            Light::Directional(l) => l.color,
            Light::Point(l) => l.color,
            Light::Spot(l) => l.color,
            Light::Hemisphere(l) => l.sky_color,
            Light::RectArea(l) => l.color,
        }
    }

    pub fn intensity(&self) -> f32 {
        match self {
            Light::Ambient(l) => l.intensity,
            Light::Directional(l) => l.intensity,
            Light::Point(l) => l.intensity,
            Light::Spot(l) => l.intensity,
            Light::Hemisphere(l) => l.intensity,
            Light::RectArea(l) => l.intensity,
        }
    }

    /// Set the colour, whichever kind of light this is.
    ///
    /// A hemisphere light has two, and this sets the sky: it is the one that
    /// corresponds to the single colour every other light has, and to what an
    /// animation driving "the light's colour" means.
    pub fn set_color(&mut self, color: Color) {
        match self {
            Light::Ambient(l) => l.color = color,
            Light::Directional(l) => l.color = color,
            Light::Point(l) => l.color = color,
            Light::Spot(l) => l.color = color,
            Light::Hemisphere(l) => l.sky_color = color,
            Light::RectArea(l) => l.color = color,
        }
    }

    pub fn set_intensity(&mut self, intensity: f32) {
        match self {
            Light::Ambient(l) => l.intensity = intensity,
            Light::Directional(l) => l.intensity = intensity,
            Light::Point(l) => l.intensity = intensity,
            Light::Spot(l) => l.intensity = intensity,
            Light::Hemisphere(l) => l.intensity = intensity,
            Light::RectArea(l) => l.intensity = intensity,
        }
    }
}

impl From<AmbientLight> for Light {
    fn from(l: AmbientLight) -> Self {
        Light::Ambient(l)
    }
}
impl From<DirectionalLight> for Light {
    fn from(l: DirectionalLight) -> Self {
        Light::Directional(l)
    }
}
impl From<PointLight> for Light {
    fn from(l: PointLight) -> Self {
        Light::Point(l)
    }
}
impl From<SpotLight> for Light {
    fn from(l: SpotLight) -> Self {
        Light::Spot(l)
    }
}
impl From<HemisphereLight> for Light {
    fn from(l: HemisphereLight) -> Self {
        Light::Hemisphere(l)
    }
}
impl From<RectAreaLight> for Light {
    fn from(l: RectAreaLight) -> Self {
        Light::RectArea(l)
    }
}
