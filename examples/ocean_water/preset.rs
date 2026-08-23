//! Sea states and quality tiers.
//!
//! A preset is one complete look: wind, sky, water colour, foam, framing. They
//! are absolute rather than incremental — loading one replaces every field,
//! which is what makes switching between them at runtime predictable.

/// A complete sea state + sky + look. Presets are absolute, not patches: loading
/// one replaces every field, which is what makes switching between them at
/// runtime predictable.
#[derive(Clone, Copy, Debug)]
pub struct Preset {
    pub name: &'static str,
    pub blurb: &'static str,
    /// Wind speed in m/s. Drives significant wave height: 3–5 calm, 6–10
    /// moderate, 15–25 storm.
    pub wind_speed: f32,
    /// Wind direction in radians.
    pub wind_dir: f32,
    /// Dominant wavelength in metres. `0.0` derives it from `wind_speed` for a
    /// fully developed sea; setting it decouples wave *size* from wave *energy*.
    pub peak_wavelength: f32,
    /// Horizontal displacement. 0.5 smooth, 1.0 natural, 1.3+ breaking.
    pub choppiness: f32,
    /// Artistic wave-height multiplier. 1.0 is physically correct.
    pub amplitude: f32,
    /// Sun elevation and azimuth in degrees.
    pub sun_elevation: f32,
    pub sun_azimuth: f32,
    pub sun_color: [f32; 3],
    pub sun_intensity: f32,
    /// Preetham atmosphere: haze, sky-blue strength, aerosol density, forward
    /// scattering. Same four knobs as three.js `Sky`.
    pub turbidity: f32,
    pub rayleigh: f32,
    pub mie_coefficient: f32,
    pub mie_directional_g: f32,
    /// Per-metre absorption, one coefficient per channel. Red dies first, which
    /// is why deep water is blue and a metre of it is not.
    pub absorption: [f32; 3],
    pub absorption_scale: f32,
    /// Colour of light scattered back out of the water body.
    pub scatter_color: [f32; 3],
    /// Strength of the glow through a wave's crest with the sun behind it.
    pub sss: f32,
    pub foam_color: [f32; 3],
    /// How far the surface must fold before foam appears. Lower = more foam.
    pub foam_threshold: f32,
    pub foam_softness: f32,
    pub foam_amount: f32,
    /// Surface slope (as `1 - n.y`) at which crests start to spill. This is the
    /// wind-driven half of whitecapping; folding is the other half.
    pub foam_slope: f32,
    pub foam_slope_softness: f32,
    /// Metres of water within which shoreline foam surges.
    pub shore_depth: f32,
    /// Microfacet roughness of the surface — widens the sun glint.
    pub roughness: f32,
    /// Fresnel reflectance at normal incidence. Water is 0.02.
    pub fresnel_f0: f32,
    /// Bulk darkening and desaturation of the sky, 0 clear to 1 solid overcast.
    /// This is the *light* an overcast day has; [`Preset::clouds`] is the
    /// geometry of it.
    pub overcast: f32,
    /// Volumetric cloud coverage, 0 for a clear sky. Raymarched in the sky
    /// shader, which is also the function the water reflects — so clouds appear
    /// in the sea rather than only above it.
    pub clouds: f32,
    /// How far the refracted sample is allowed to travel from straight-through.
    /// 0 looks through the surface as if it were flat glass; 1 bends fully.
    pub refraction: f32,
    /// Screen-space reflection blended over the analytic sky.
    pub ssr: f32,
    /// Strength of the caustic pattern the surface projects onto the sea floor.
    pub caustics: f32,
    /// Glitter grain on the sun lobe. 0 leaves a smooth specular wedge.
    pub sparkle: f32,
    /// How strongly foam that broke earlier still shows.
    pub foam_persistence: f32,
    /// Per-second decay of that memory. 0.9 is roughly a one-second trail — long
    /// enough to read as a wake, short enough that the sea does not silt up.
    pub foam_decay: f32,
    /// How visible the wake behind a moving object is.
    pub wake_strength: f32,
    /// Rain intensity, 0 for none.
    pub rain: f32,
    /// Spray torn off breaking crests, 0 for none.
    pub spray: f32,
    /// Distance in metres over which the sea dissolves into the sky it
    /// reflects. Doubles as visibility: pull it in and the horizon closes.
    pub haze_near: f32,
    pub haze_far: f32,
    /// Linear-space colour of the sky at the horizon. The water reaches this by
    /// evaluating the atmosphere; the island and the buoy are ordinary lit
    /// materials and need it handed to them as scene fog, or they stay crisp in
    /// weather that has swallowed everything around them.
    pub horizon_tint: [f32; 3],
    /// Where the opening shot sits. Grazing angles turn water into a mirror, so
    /// a low camera sells wave shape and a high one sells depth and surf —
    /// which of those matters depends entirely on the sea state.
    pub camera_height: f32,
    pub camera_distance: f32,
    pub sky_reflect: f32,
    pub exposure: f32,
    pub sand_color: [f32; 3],
}

/// The preset the others are written as deltas from, so each entry below reads
/// as its *differences* rather than forty repeated fields.
const SUNSET: Preset = Preset {
    name: "sunset",
    blurb: "golden hour, warm colours, moderate swell",
    wind_speed: 8.0,
    wind_dir: 0.6,
    peak_wavelength: 0.0,
    choppiness: 1.0,
    amplitude: 1.0,
    sun_elevation: 2.6,
    sun_azimuth: 170.0,
    sun_color: [1.0, 0.66, 0.36],
    sun_intensity: 1.1,
    turbidity: 4.5,
    rayleigh: 3.0,
    mie_coefficient: 0.006,
    mie_directional_g: 0.82,
    absorption: [0.42, 0.09, 0.05],
    absorption_scale: 1.0,
    scatter_color: [0.035, 0.14, 0.16],
    sss: 1.3,
    foam_color: [0.94, 0.96, 0.97],
    foam_threshold: 0.16,
    foam_softness: 0.30,
    foam_amount: 1.0,
    foam_slope: 0.44,
    foam_slope_softness: 0.16,
    shore_depth: 2.4,
    roughness: 0.09,
    fresnel_f0: 0.02,
    overcast: 0.0,
    clouds: 0.5,
    refraction: 1.0,
    ssr: 0.85,
    caustics: 1.0,
    sparkle: 0.8,
    foam_persistence: 0.55,
    foam_decay: 1.2,
    wake_strength: 0.8,
    rain: 0.0,
    spray: 0.35,
    haze_near: 1500.0,
    haze_far: 5200.0,
    horizon_tint: [0.478, 0.378, 0.375],
    camera_height: 17.0,
    camera_distance: 330.0,
    sky_reflect: 1.0,
    exposure: 1.0,
    sand_color: [0.76, 0.68, 0.52],
};

/// Every preset, in the order `--list` prints them.
pub fn all() -> [Preset; 8] {
    [
        Preset {
            name: "calm",
            blurb: "light air, small chop, high sun",
            wind_speed: 4.5,
            sun_elevation: 46.0,
            sun_color: [1.0, 0.97, 0.92],
            sun_intensity: 1.25,
            turbidity: 2.6,
            rayleigh: 1.8,
            choppiness: 0.75,
            clouds: 0.45,
            camera_height: 34.0,
            horizon_tint: [0.846, 0.922, 0.943],
            foam_threshold: 0.24,
            scatter_color: [0.03, 0.15, 0.19],
            ..SUNSET
        },
        SUNSET,
        Preset {
            name: "dusk",
            blurb: "calm twilight swell beneath a low golden-pink sun",
            wind_speed: 5.5,
            peak_wavelength: 120.0,
            sun_elevation: 1.4,
            sun_azimuth: 205.0,
            sun_color: [1.0, 0.52, 0.34],
            sun_intensity: 0.85,
            turbidity: 8.0,
            rayleigh: 3.1,
            mie_coefficient: 0.008,
            choppiness: 0.8,
            clouds: 0.55,
            scatter_color: [0.025, 0.085, 0.115],
            horizon_tint: [0.357, 0.269, 0.269],
            foam_threshold: 0.24,
            exposure: 1.12,
            ..SUNSET
        },
        Preset {
            name: "storm",
            blurb: "violent waves under a dark, heavy sky",
            wind_speed: 19.0,
            wind_dir: 2.1,
            // A fully developed 19 m/s sea peaks near 270 m, which from any
            // human vantage reads as the horizon slowly tilting. Shortening the
            // peak keeps the energy and puts it at a size the eye can see —
            // the point of decoupling size from wind in the first place.
            peak_wavelength: 130.0,
            choppiness: 1.35,
            sun_elevation: 9.0,
            sun_azimuth: 120.0,
            sun_color: [0.66, 0.68, 0.74],
            sun_intensity: 0.6,
            turbidity: 10.0,
            rayleigh: 2.2,
            mie_coefficient: 0.03,
            mie_directional_g: 0.76,
            scatter_color: [0.03, 0.055, 0.065],
            absorption: [0.5, 0.16, 0.11],
            sss: 0.7,
            foam_threshold: 0.06,
            foam_softness: 0.35,
            foam_amount: 1.3,
            foam_slope: 0.17,
            foam_slope_softness: 0.2,
            rain: 1.0,
            spray: 1.0,
            shore_depth: 4.0,
            roughness: 0.17,
            camera_height: 13.0,
            camera_distance: 300.0,
            overcast: 0.72,
            clouds: 0.92,
            haze_near: 700.0,
            haze_far: 3000.0,
            horizon_tint: [0.442, 0.438, 0.442],
            exposure: 0.66,
            ..SUNSET
        },
        Preset {
            name: "arctic",
            blurb: "cold blue-grey water, stormy, low pale sun",
            wind_speed: 15.0,
            wind_dir: 4.0,
            peak_wavelength: 110.0,
            choppiness: 1.22,
            spray: 0.85,
            foam_slope: 0.22,
            foam_slope_softness: 0.2,
            sun_elevation: 8.0,
            sun_azimuth: 40.0,
            sun_color: [0.86, 0.9, 1.0],
            sun_intensity: 0.85,
            turbidity: 4.0,
            rayleigh: 1.6,
            mie_coefficient: 0.012,
            scatter_color: [0.055, 0.105, 0.125],
            absorption: [0.36, 0.14, 0.13],
            sss: 0.6,
            foam_threshold: 0.09,
            foam_amount: 1.15,
            foam_color: [0.9, 0.94, 1.0],
            overcast: 0.5,
            clouds: 0.7,
            exposure: 0.92,
            horizon_tint: [0.697, 0.673, 0.664],
            sand_color: [0.55, 0.58, 0.6],
            ..SUNSET
        },
        Preset {
            name: "moonlit",
            blurb: "night, a cold specular path across black water",
            wind_speed: 7.0,
            sun_elevation: 22.0,
            sun_azimuth: 245.0,
            sun_color: [0.62, 0.72, 1.0],
            // Preetham's sky brightness is set by the sun's *position*, not by
            // any intensity term — turning the sun down leaves broad daylight.
            // Night has to come out of exposure, with the light put back into
            // the terms exposure then divides.
            sun_intensity: 9.0,
            turbidity: 2.0,
            rayleigh: 0.9,
            mie_coefficient: 0.003,
            mie_directional_g: 0.9,
            scatter_color: [0.010, 0.026, 0.05],
            sss: 0.4,
            foam_color: [0.6, 0.68, 0.85],
            foam_threshold: 0.15,
            roughness: 0.05,
            clouds: 0.22,
            exposure: 0.085,
            horizon_tint: [0.125, 0.146, 0.148],
            sand_color: [0.20, 0.23, 0.32],
            ..SUNSET
        },
        Preset {
            name: "foggy",
            blurb: "flat light, short visibility, muted everything",
            wind_speed: 6.0,
            sun_elevation: 18.0,
            sun_azimuth: 300.0,
            sun_color: [0.85, 0.86, 0.88],
            sun_intensity: 0.8,
            turbidity: 22.0,
            rayleigh: 0.7,
            mie_coefficient: 0.045,
            mie_directional_g: 0.62,
            scatter_color: [0.06, 0.085, 0.095],
            camera_height: 22.0,
            sss: 0.5,
            foam_threshold: 0.2,
            roughness: 0.15,
            overcast: 0.88,
            clouds: 0.5,
            haze_near: 130.0,
            haze_far: 850.0,
            rain: 0.45,
            horizon_tint: [0.849, 0.848, 0.848],
            exposure: 0.95,
            ..SUNSET
        },
        Preset {
            name: "tropical",
            blurb: "shallow turquoise over pale sand, gentle swell",
            wind_speed: 7.5,
            peak_wavelength: 55.0,
            sun_elevation: 34.0,
            sun_azimuth: 95.0,
            sun_color: [1.0, 0.98, 0.94],
            sun_intensity: 1.3,
            turbidity: 3.2,
            rayleigh: 2.0,
            choppiness: 0.9,
            absorption: [0.48, 0.10, 0.065],
            scatter_color: [0.04, 0.24, 0.27],
            sss: 1.6,
            camera_height: 105.0,
            horizon_tint: [0.849, 0.915, 0.935],
            camera_distance: 430.0,
            foam_threshold: 0.18,
            foam_slope: 0.38,
            clouds: 0.36,
            shore_depth: 4.5,
            sand_color: [0.92, 0.86, 0.7],
            ..SUNSET
        },
    ]
}

/// Mesh density and spectral resolution.
///
/// Two knobs, because they cost in different places. `rings`/`sectors` set how
/// finely the surface under the camera is tessellated, which is vertex work.
/// `cascade_n` sets the lattice the spectrum is sampled on — 128² is 16 384
/// modes per cascade, 512² is 262 144 — which is FFT work, and is what decides
/// how much genuine detail there is to tessellate in the first place.
///
/// The cost is almost entirely the GPU's. Vertices are displaced by a compute
/// pass and never come back across the bus, so the CPU's per-frame work does not
/// grow with the grid at all. Measured on an M-series Mac, `--bench`, GPU with
/// the queue drained:
///
/// | level  | vertices | cpu/frame | cascades | displace | surface-state |
/// |--------|----------|-----------|----------|----------|---------------|
/// | high   |   18 433 |  0.23 ms  | 0.232 ms | 0.046 ms |    0.064 ms   |
/// | max    |   51 201 |  0.24 ms  | 0.246 ms | 0.064 ms |    0.064 ms   |
///
/// The cascade pass is the same cost at every tier — it is 18 dispatches over
/// three 256² fields regardless of how many vertices sample them. Displacing on
/// the CPU instead used to cost 6.7 ms a frame at `high` and 16.7 ms at `max`,
/// before any drawing.
///
/// The cpu column covers every compute pass in the frame: cascades, mesh
/// displacement, foam and three particle systems. They share one encoder and one
/// submission — six submits was 0.37 ms rather than 0.23, and a submission is not
/// free.
#[derive(Clone, Copy)]
pub struct Quality {
    pub name: &'static str,
    pub rings: usize,
    pub sectors: usize,
    /// Texels per side of each cascade. Power of two; see
    /// [`Cascades`](crate::ocean_fft::Cascades).
    pub cascade_n: usize,
}

pub const QUALITY_LEVELS: [Quality; 5] = [
    Quality {
        name: "low",
        rings: 48,
        sectors: 96,
        cascade_n: 128,
    },
    Quality {
        name: "medium",
        rings: 64,
        sectors: 128,
        cascade_n: 128,
    },
    Quality {
        name: "high",
        rings: 96,
        sectors: 192,
        cascade_n: 256,
    },
    Quality {
        name: "ultra",
        rings: 128,
        sectors: 256,
        cascade_n: 256,
    },
    Quality {
        name: "max",
        rings: 160,
        sectors: 320,
        cascade_n: 512,
    },
];
