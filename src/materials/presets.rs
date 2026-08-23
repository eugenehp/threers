//! Measured-reflectance [`PhysicalMaterial`] presets, with a spacecraft bias.
//!
//! For a **metal** (`metalness = 1.0`) the base color is not an albedo — it is
//! the F0 specular reflectance, i.e. the fraction of light reflected at normal
//! incidence per channel. Eyeballing a hex value for gold gets the hue wrong in
//! a way that is hard to unsee once the environment is reflecting properly, so
//! these use published linear-space values rather than `Color::from_hex`.
//!
//! Every preset here is just a starting point — the returned struct is plain
//! data, so tweak any field afterward:
//!
//! ```no_run
//! use threers::materials::presets;
//! let mut foil = presets::gold_foil();
//! foil.roughness = 0.4; // more diffuse, older / more crumpled
//! ```
//!
//! # These need an environment map
//!
//! Metals have no diffuse term. Without a [`crate::Scene::environment`] the
//! metal presets render near-black no matter how the lights are set — see the
//! `spacecraft_materials` example for the full setup.

use super::{PhysicalMaterial, TransparencyMode};
use crate::math::Color;

/// Gold multi-layer insulation (MLI) — the amber foil blanket on spacecraft
/// bodies and the JWST-style sunshield. F0 (1.000, 0.766, 0.336).
///
/// The crumpled read comes from a normal map, not from these numbers: assign a
/// tiled crinkle to `normal_map` and this goes from "snooker ball" to "foil".
pub fn gold_foil() -> PhysicalMaterial {
    PhysicalMaterial::new(Color::new(1.000, 0.766, 0.336))
        .with_metalness(1.0)
        .with_roughness(0.25)
}

/// Aluminized-kapton / silver MLI. F0 (0.972, 0.960, 0.915).
pub fn silver_foil() -> PhysicalMaterial {
    PhysicalMaterial::new(Color::new(0.972, 0.960, 0.915))
        .with_metalness(1.0)
        .with_roughness(0.18)
}

/// Bare structural aluminium — bus panels, trusses, brackets.
/// F0 (0.913, 0.921, 0.925).
pub fn aluminum() -> PhysicalMaterial {
    PhysicalMaterial::new(Color::new(0.913, 0.921, 0.925))
        .with_metalness(1.0)
        .with_roughness(0.35)
}

/// Machined aluminium with a circular-brushed finish: the same base metal, plus
/// an anisotropic streak. `rotation` aims the streak in UV tangent space.
pub fn brushed_aluminum(rotation: f32) -> PhysicalMaterial {
    aluminum()
        .with_roughness(0.3)
        .with_anisotropy(0.8, rotation)
}

/// Titanium — engine bells, high-temperature structure. F0 (0.542, 0.497, 0.449).
pub fn titanium() -> PhysicalMaterial {
    PhysicalMaterial::new(Color::new(0.542, 0.497, 0.449))
        .with_metalness(1.0)
        .with_roughness(0.4)
}

/// Heat-tinted titanium: the oxide layer that forms on hot titanium is a thin
/// film, which is exactly what the iridescence layer models. Vary
/// `thickness_nm` (roughly 150–500) to walk from straw to violet to blue.
pub fn anodized_titanium(thickness_nm: f32) -> PhysicalMaterial {
    titanium()
        .with_roughness(0.28)
        .with_iridescence(1.0, 1.8, thickness_nm)
}

/// A photovoltaic cell: dark silicon substrate under a specular cover glass.
///
/// The look is carried by three things stacked — a very dark blue base, a
/// clearcoat standing in for the cover glass, and an anisotropic streak along
/// the cell's busbar grid. Assign a grid texture to `map` for the interconnects.
pub fn solar_cell() -> PhysicalMaterial {
    PhysicalMaterial::new(Color::from_hex(0x0a1633))
        .with_metalness(0.1)
        .with_roughness(0.35)
        .with_clearcoat(1.0, 0.04)
        .with_anisotropy(0.6, 0.0)
}

/// Solar-array backing / structural panel behind the cells.
pub fn array_backing() -> PhysicalMaterial {
    PhysicalMaterial::new(Color::from_hex(0x1b1d24))
        .with_metalness(0.0)
        .with_roughness(0.8)
}

/// Optical glass — porthole, sensor window, lens.
///
/// Uses [`TransparencyMode::Refract`] (screen-space refraction), which needs an
/// environment set to read well. `dispersion` splits the refraction IOR per
/// channel for a rainbow edge fringe.
pub fn optical_glass() -> PhysicalMaterial {
    PhysicalMaterial::new(Color::from_hex(0xffffff))
        .with_transmission(1.0, 1.52, 0.4)
        .with_roughness(0.03)
        .with_attenuation(Color::from_hex(0xdff0ff), 2.0)
        .with_transparency(TransparencyMode::Refract)
}

/// Thermal-radiator white paint (e.g. AZ-93) — diffuse, slightly warm.
pub fn white_thermal_paint() -> PhysicalMaterial {
    PhysicalMaterial::new(Color::from_hex(0xf2f0ea))
        .with_metalness(0.0)
        .with_roughness(0.7)
}

/// Black kapton / anti-stray-light coating. Deliberately near-black with a
/// faint sheen so the silhouette still reads against a dark starfield.
pub fn black_kapton() -> PhysicalMaterial {
    PhysicalMaterial::new(Color::from_hex(0x0a0a0c))
        .with_metalness(0.0)
        .with_roughness(0.55)
        .with_sheen(0.35, Color::from_hex(0x3a4050), 0.7)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_presets_are_fully_metallic() {
        for m in [gold_foil(), silver_foil(), aluminum(), titanium()] {
            assert_eq!(m.metalness, 1.0);
        }
    }

    #[test]
    fn gold_f0_is_warm() {
        // Gold reflects red fully, green partially, blue least — if this
        // ordering ever inverts the preset has been mis-edited.
        let g = gold_foil().color;
        assert!(g.r > g.g && g.g > g.b);
    }

    #[test]
    fn glass_opts_into_refraction() {
        let g = optical_glass();
        assert_eq!(g.transparency, TransparencyMode::Refract);
        assert!(g.transmission > 0.0);
        // Attenuation must be finite, else the Beer-Lambert path stays off.
        assert!(g.attenuation_distance.is_finite());
    }

    #[test]
    fn anodized_titanium_enables_thin_film() {
        let t = anodized_titanium(300.0);
        assert!(t.iridescence > 0.0);
        assert_eq!(t.iridescence_thickness, 300.0);
    }
}
