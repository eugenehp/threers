//! How a surface is coloured: [`Look`].

use crate::math::Color;

/// A base colour under a Fresnel reflection of a small studio — a softbox
/// above, a window to the left, a fill to the right, the floor below and
/// whatever is behind the viewer — plus two sharp glints and a contact shadow.
/// Colours are sRGB `0..1`, as a CSS colour is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Look {
    /// What the surface is where it reflects nothing.
    pub base: [f32; 3],
    /// The colour of what it reflects.
    pub tint: [f32; 3],
    /// Reflectance facing the viewer (Schlick's F0): ~0.05 for a lacquer or a
    /// liquid, 0.5–0.9 for a metal.
    pub f0: f32,
    /// The studio's lights, as multipliers on their own levels.
    pub softbox: f32,
    pub window: f32,
    pub fill: f32,
    pub floor: f32,
    /// What is behind the viewer, reflected where the surface faces you. A
    /// flat-topped metal needs it or its middle reads as a hole; the lacquer
    /// has none, so its face stays black.
    pub front: f32,
    /// The two sharp specular glints.
    pub glints: f32,
    /// Opacity of the soft contact shadow under the shape, 0 for none.
    pub shadow: f32,
}

impl Look {
    /// Near-black lacquer.
    pub const LACQUER: Look = Look {
        base: [0.008, 0.008, 0.01],
        tint: [0.93, 0.94, 0.97],
        f0: 0.05,
        softbox: 1.0,
        window: 1.0,
        fill: 1.0,
        floor: 1.0,
        front: 0.0,
        glints: 1.0,
        shadow: 0.2,
    };
    /// Liquid metal: reflects everywhere, darkest where it faces you.
    pub const CHROME: Look = Look {
        base: [0.05, 0.05, 0.06],
        tint: [0.98, 0.98, 1.0],
        f0: 0.55,
        softbox: 1.1,
        window: 1.0,
        fill: 1.2,
        floor: 0.9,
        front: 0.55,
        glints: 1.2,
        shadow: 0.22,
    };
    /// Molten gold.
    pub const GOLD: Look = Look {
        base: [0.16, 0.08, 0.0],
        tint: [1.0, 0.76, 0.33],
        f0: 0.6,
        softbox: 1.0,
        window: 0.9,
        fill: 1.1,
        floor: 0.8,
        front: 0.5,
        glints: 1.0,
        shadow: 0.22,
    };
    /// Glossy green slime.
    pub const TOXIC: Look = Look {
        base: [0.12, 0.6, 0.05],
        tint: [0.9, 1.0, 0.85],
        f0: 0.04,
        softbox: 1.0,
        window: 1.0,
        fill: 1.0,
        floor: 0.6,
        front: 0.0,
        glints: 1.0,
        shadow: 0.18,
    };

    /// Every preset, with its name.
    pub const PRESETS: [(&'static str, Look); 4] =
        [("lacquer", Look::LACQUER), ("chrome", Look::CHROME), ("gold", Look::GOLD), ("toxic", Look::TOXIC)];

    /// A preset by name — `lacquer`, `chrome`, `gold`, `toxic` — for command
    /// lines and config files.
    pub fn named(name: &str) -> Option<Look> {
        Some(match name {
            "lacquer" | "black" => Look::LACQUER,
            "chrome" | "mercury" | "silver" => Look::CHROME,
            "gold" => Look::GOLD,
            "toxic" | "slime" | "green" => Look::TOXIC,
            _ => return None,
        })
    }

    /// The same look with a different base colour, from a `0xRRGGBB` hex.
    pub fn with_base_hex(mut self, rgb: u32) -> Look {
        self.base = hex3(rgb);
        self
    }

    /// The same look reflecting a different colour, from a `0xRRGGBB` hex.
    pub fn with_tint_hex(mut self, rgb: u32) -> Look {
        self.tint = hex3(rgb);
        self
    }

    /// The base scaled by `k` and lifted by `lift`, as a linear [`Color`] for
    /// the overlay — what fine detail drawn in the surface's colour uses.
    pub fn base_color(&self, k: f32, lift: f32) -> Color {
        shade(self.base, k, lift)
    }

    /// The tint, likewise.
    pub fn tint_color(&self, k: f32, lift: f32) -> Color {
        shade(self.tint, k, lift)
    }
}

fn hex3(rgb: u32) -> [f32; 3] {
    [((rgb >> 16) & 255) as f32 / 255.0, ((rgb >> 8) & 255) as f32 / 255.0, (rgb & 255) as f32 / 255.0]
}

fn shade(rgb: [f32; 3], k: f32, lift: f32) -> Color {
    let lin = |s: f32| {
        let s = (s * k + lift).clamp(0.0, 1.0);
        if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    };
    Color::new(lin(rgb[0]), lin(rgb[1]), lin(rgb[2]))
}
