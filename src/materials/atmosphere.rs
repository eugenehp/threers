use crate::math::Color;

/// A planetary atmosphere, shaded analytically.
///
/// Attach it to a sphere a little larger than the planet. The shader does not
/// treat that sphere as a surface: it intersects the view ray with the
/// atmosphere shell and with the planet, measures how much air the ray actually
/// passes through, and shades from that. So the glow thickens toward the limb
/// where the path is long, is cut off where the planet blocks it, and needs no
/// back-face trick or stack of nested shells to get a gradient.
///
/// Light comes from the first directional light in the scene, which is what
/// puts the day side bright, the night side dark, and a warm band along the
/// terminator where the sunlight has travelled furthest through the air.
///
/// ```
/// use threers::{AtmosphereMaterial, Color};
/// // Earth-like: a thin shell, 2.5% of the planet radius.
/// let air = AtmosphereMaterial::new(1.0, 1.025);
/// assert!(air.atmosphere_radius > air.planet_radius);
/// let mars = AtmosphereMaterial::new(1.0, 1.01).color(Color::from_hex(0xd8a06a));
/// assert_eq!(mars.color, Color::from_hex(0xd8a06a));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AtmosphereMaterial {
    /// Scattering tint. Earth's air scatters blue hardest, so the default is a
    /// sky blue; a dusty planet wants something warmer.
    pub color: Color,
    /// Colour the light takes on when it has grazed a long way through the air
    /// — the sunset band at the terminator.
    pub sunset_color: Color,
    /// Radius of the solid body the atmosphere sits on, in world units. The
    /// shader clips the view ray against this, which is what stops the glow
    /// bleeding across the planet's disc.
    pub planet_radius: f32,
    /// Outer radius of the atmosphere. The difference from `planet_radius` is
    /// the depth of air.
    pub atmosphere_radius: f32,
    /// Overall strength.
    pub intensity: f32,
    /// How sharply density falls off with altitude. Higher packs the glow
    /// closer to the surface; Earth's air is around `3`.
    pub falloff: f32,
    /// Opacity multiplier, on top of the computed optical depth.
    pub opacity: f32,
    /// Airglow: a faint band of light the upper atmosphere emits on its own,
    /// with no sun on it.
    ///
    /// Oxygen recombining at around 90 km, which on Earth is 1.4% of the
    /// radius, and it is why photographs from orbit show a thin green line
    /// tracing the night limb rather than the planet ending in black. Zero
    /// disables it.
    pub airglow: f32,
    /// Colour of that emission. Earth's is dominated by the 558 nm oxygen line.
    pub airglow_color: Color,
    /// Ozone absorption. 0 disables it.
    ///
    /// Rayleigh scattering alone makes a limb that whitens as it thickens,
    /// because scattering *adds* light at every wavelength and the blue
    /// saturates first. What keeps a real twilight blue is absorption: ozone's
    /// Chappuis band eats the middle of the spectrum — orange and green — while
    /// letting blue through, and the sunbeam's path near the terminator is long
    /// enough through the ozone layer for that to dominate. Without it the
    /// limb reads as haze rather than sky.
    pub ozone: f32,
    /// Aurora brightness. 0 disables it.
    ///
    /// Solar wind particles follow the field lines down and hit the upper
    /// atmosphere in a ring around each *geomagnetic* pole — not the spin axis,
    /// which is why the oval sits off-centre. Emission is atomic oxygen: the
    /// same 558 nm green as the airglow low down, with a red 630 nm crown above
    /// it where the air is thin enough for the slower transition to survive.
    pub aurora: f32,
    /// Colour of the green base of the emission.
    pub aurora_color: Color,
    /// Angular radius of the auroral oval, in degrees from the pole.
    ///
    /// About 23 on Earth in quiet conditions — the ring sits near 67 degrees of
    /// magnetic latitude — and it widens toward the equator as a storm builds.
    pub aurora_colatitude: f32,
}

impl Default for AtmosphereMaterial {
    fn default() -> Self {
        Self::new(1.0, 1.025)
    }
}

impl AtmosphereMaterial {
    /// An Earth-like atmosphere between the two radii, in world units.
    pub fn new(planet_radius: f32, atmosphere_radius: f32) -> Self {
        Self {
            color: Color::new(0.30, 0.55, 1.0),
            sunset_color: Color::new(1.0, 0.48, 0.20),
            planet_radius: planet_radius.max(1e-4),
            atmosphere_radius: atmosphere_radius.max(planet_radius * 1.0001),
            intensity: 1.5,
            falloff: 3.0,
            opacity: 1.0,
            airglow: 0.0,
            airglow_color: Color::new(0.35, 1.0, 0.55),
            ozone: 0.0,
            aurora: 0.0,
            // 558 nm atomic oxygen: the green everyone pictures.
            aurora_color: Color::new(0.25, 1.0, 0.45),
            aurora_colatitude: 23.0,
        }
    }

    /// Scattering tint.
    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Colour along the terminator, where the light has the longest path.
    pub fn sunset_color(mut self, color: Color) -> Self {
        self.sunset_color = color;
        self
    }

    /// Overall strength.
    pub fn intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.max(0.0);
        self
    }

    /// Density falloff with altitude — higher hugs the surface more tightly.
    pub fn falloff(mut self, falloff: f32) -> Self {
        self.falloff = falloff.clamp(0.1, 32.0);
        self
    }

    /// Ozone absorption strength. See [`ozone`](Self::ozone).
    pub fn ozone(mut self, ozone: f32) -> Self {
        self.ozone = ozone.max(0.0);
        self
    }

    /// Aurora brightness, colour, and the oval's angular radius in degrees.
    pub fn aurora(mut self, strength: f32, color: Color, colatitude: f32) -> Self {
        self.aurora = strength.max(0.0);
        self.aurora_color = color;
        self.aurora_colatitude = colatitude.clamp(1.0, 89.0);
        self
    }

    /// Opacity multiplier.
    pub fn opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Strength of the airglow band.
    pub fn airglow(mut self, strength: f32) -> Self {
        self.airglow = strength.max(0.0);
        self
    }

    /// Colour of the airglow band.
    pub fn airglow_color(mut self, color: Color) -> Self {
        self.airglow_color = color;
        self
    }

    /// Depth of air, in world units.
    pub fn thickness(&self) -> f32 {
        (self.atmosphere_radius - self.planet_radius).max(1e-5)
    }
}
