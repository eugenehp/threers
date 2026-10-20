//! Gradients, and the colour maps worth having by name.
//!
//! d3's `scaleSequential` plus `d3-scale-chromatic`, which between them are the
//! most-copied thing in the library. The palettes are not decoration: viridis
//! and its family were built so that equal steps in the data look like equal
//! steps in the colour *and* survive being printed in grey or read by someone
//! colour-blind, which the rainbow they replaced does neither of.
//!
//! ```
//! use threers::math::ColorRamp;
//!
//! let heat = ColorRamp::viridis();
//! let cold = heat.at(0.0);
//! let hot = heat.at(1.0);
//! assert!(hot.luminance() > cold.luminance());
//!
//! // Or build one. Stops are positions in 0..1 and need not be evenly spaced.
//! use threers::Color;
//! let two = ColorRamp::new(vec![(0.0, Color::from_hex(0x001f3f)), (1.0, Color::from_hex(0xffdc00))]);
//! let mid = two.at(0.5);
//! ```

use super::color_schemes::{CATEGORICAL, CYCLIC, DIVERGING, EXTRA_SEQUENTIAL, SEQUENTIAL};
use crate::math::{Color, ColorSpace};

/// How a ramp blends between its stops: any [`ColorSpace`]. OkLab by
/// default — perceptually even, the midpoint of two colours the one a person
/// would name; linear light is right for a ramp standing in for light; sRGB
/// is what CSS gradients and d3's default do; OkLCh and LCh keep a ramp
/// between saturated colours saturated.
pub type RampSpace = ColorSpace;

/// What a named scheme is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemeKind {
    /// Low to high: one direction of more.
    Sequential,
    /// Two directions away from a meaningful middle.
    Diverging,
    /// Wraps round: its ends are the same colour, for angles and phases.
    Cyclic,
    /// Distinct colours for categories — a [`Palette`], not a ramp.
    Categorical,
}

/// The lookup form of a scheme's name: lower case, letters and digits only,
/// and without d3's `interpolate` or `scheme` prefix — so `"RdYlBu"`,
/// `"rd_yl_bu"` and `"interpolateRdYlBu"` are the same scheme.
fn scheme_key(name: &str) -> String {
    let k: String = name.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
    k.strip_prefix("interpolate").or_else(|| k.strip_prefix("scheme")).unwrap_or(&k).to_string()
}

/// A gradient: colours at positions, blended between.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorRamp {
    stops: Vec<(f32, Color)>,
    space: RampSpace,
}

impl ColorRamp {
    /// Build from `(position, colour)` pairs. Positions are sorted here, so
    /// they need not arrive in order, and a ramp with no stops reads as black.
    pub fn new(mut stops: Vec<(f32, Color)>) -> Self {
        stops.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Self {
            stops,
            space: RampSpace::Oklab,
        }
    }

    /// Evenly spaced stops, which is how the named palettes are defined.
    pub fn from_hex(colors: &[u32]) -> Self {
        let n = colors.len();
        Self::new(
            colors
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let t = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.0 };
                    (t, Color::from_hex(*c))
                })
                .collect(),
        )
    }

    /// Blend in linear light instead of OkLab. For a ramp standing in for
    /// *light* — a fire gradient driving an emissive, a sky — rather than for a
    /// colour scale being read.
    pub fn in_linear_rgb(mut self) -> Self {
        self.space = RampSpace::LinearRgb;
        self
    }

    /// Blend in `space` (see [`RampSpace`]).
    pub fn in_space(mut self, space: ColorSpace) -> Self {
        self.space = space;
        self
    }

    /// The space it blends in.
    pub fn space(&self) -> ColorSpace {
        self.space
    }

    /// The same ramp end to end: `at(t)` becomes `at(1 − t)`.
    pub fn reversed(mut self) -> Self {
        for s in &mut self.stops {
            s.0 = 1.0 - s.0;
        }
        self.stops.reverse();
        self
    }

    /// The part between `a` and `b`, stretched over `0..1` — a sequential
    /// scheme without its palest end on a white page, say.
    pub fn slice(&self, a: f32, b: f32) -> Self {
        let n = self.stops.len().max(2) * 2;
        Self {
            stops: (0..=n)
                .map(|i| {
                    let t = i as f32 / n as f32;
                    (t, self.at(a + (b - a) * t))
                })
                .collect(),
            space: self.space,
        }
    }

    /// A named scheme: one of the named constructors here, or any of
    /// d3-scale-chromatic's sequential, diverging and cyclic schemes —
    /// `"blues"`, `"yl_gn_bu"`, `"RdBu"`, `"interpolateSpectral"`,
    /// `"cividis"`, `"twilight"` (see [`scheme_names`](Self::scheme_names)).
    /// Case, separators and d3's `interpolate` prefix do not matter.
    pub fn named(name: &str) -> Option<Self> {
        let key = scheme_key(name);
        match key.as_str() {
            "viridis" => return Some(Self::viridis()),
            "magma" => return Some(Self::magma()),
            "inferno" => return Some(Self::inferno()),
            "plasma" => return Some(Self::plasma()),
            "turbo" => return Some(Self::turbo()),
            "coolwarm" => return Some(Self::cool_warm()),
            _ => {}
        }
        SEQUENTIAL
            .iter()
            .chain(EXTRA_SEQUENTIAL)
            .chain(DIVERGING)
            .chain(CYCLIC)
            .find(|(n, _)| scheme_key(n) == key)
            .map(|(_, colors)| Self::from_hex(colors))
    }

    /// Every scheme [`named`](Self::named) knows, with its kind.
    pub fn scheme_names() -> Vec<(&'static str, SchemeKind)> {
        let mut v: Vec<(&'static str, SchemeKind)> =
            ["viridis", "magma", "inferno", "plasma", "turbo"].iter().map(|n| (*n, SchemeKind::Sequential)).collect();
        v.push(("cool_warm", SchemeKind::Diverging));
        v.extend(SEQUENTIAL.iter().chain(EXTRA_SEQUENTIAL).map(|(n, _)| (*n, SchemeKind::Sequential)));
        v.extend(DIVERGING.iter().map(|(n, _)| (*n, SchemeKind::Diverging)));
        v.extend(CYCLIC.iter().map(|(n, _)| (*n, SchemeKind::Cyclic)));
        v.extend(CATEGORICAL.iter().map(|(n, _)| (*n, SchemeKind::Categorical)));
        v
    }

    /// Cividis: viridis reworked so it reads the same to the commonest
    /// colour-vision deficiency, blue to yellow.
    pub fn cividis() -> Self {
        Self::named("cividis").expect("built in")
    }

    /// Twilight: cyclic, for angles, phases and times of day — its ends meet.
    pub fn twilight() -> Self {
        Self::named("twilight").expect("built in")
    }

    /// The colour at `t`, clamped to the ends outside `0..1`.
    pub fn at(&self, t: f32) -> Color {
        match self.stops.len() {
            0 => Color::BLACK,
            1 => self.stops[0].1,
            _ => {
                let t = if t.is_nan() { self.stops[0].0 } else { t };
                let t = t.clamp(self.stops[0].0, self.stops[self.stops.len() - 1].0);
                // Linear scan: a palette has a handful of stops, and a binary
                // search over nine entries is slower than reading them.
                let mut i = 0;
                while i + 2 < self.stops.len() && self.stops[i + 1].0 < t {
                    i += 1;
                }
                let (p0, c0) = self.stops[i];
                let (p1, c1) = self.stops[i + 1];
                let span = p1 - p0;
                let local = if span.abs() < f32::EPSILON {
                    0.0
                } else {
                    (t - p0) / span
                };
                c0.mix(c1, local, self.space)
            }
        }
    }

    /// `n` colours evenly spaced across the ramp — a discrete palette.
    pub fn sample(&self, n: usize) -> Vec<Color> {
        (0..n)
            .map(|i| {
                self.at(if n > 1 {
                    i as f32 / (n - 1) as f32
                } else {
                    0.5
                })
            })
            .collect()
    }

    pub fn stops(&self) -> &[(f32, Color)] {
        &self.stops
    }

    // --- The named maps -----------------------------------------------------
    //
    // Nine stops each, sampled from the originals at eighths. The full
    // definitions are 256-entry tables; nine stops reproduce them closely
    // enough that the difference is below a perceptible step, and a table that
    // size does not belong in a renderer.

    /// Viridis. The default choice for a sequential scale: perceptually
    /// uniform, readable in greyscale, and safe for the common colour-vision
    /// deficiencies.
    pub fn viridis() -> Self {
        Self::from_hex(&[
            0x440154, 0x46327e, 0x3e4989, 0x31688e, 0x26828e, 0x1f9e89, 0x35b779, 0x6ece58,
            0xfde725,
        ])
    }

    /// Magma. Black through purple and red to near-white; the widest lightness
    /// range of the family, which makes it the best of them in print.
    pub fn magma() -> Self {
        Self::from_hex(&[
            0x000004, 0x180f3d, 0x440f76, 0x721f81, 0x9e2f7f, 0xcd4071, 0xf1605d, 0xfd9668,
            0xfcfdbf,
        ])
    }

    /// Inferno. Magma's hotter sibling — same structure, more saturated
    /// through the oranges.
    pub fn inferno() -> Self {
        Self::from_hex(&[
            0x000004, 0x1b0c41, 0x4a0c6b, 0x781c6d, 0xa52c60, 0xcf4446, 0xed6925, 0xfb9b06,
            0xfcffa4,
        ])
    }

    /// Plasma. Blue through magenta to yellow, with no dark end — useful when
    /// the background is already dark and a black low end would disappear.
    pub fn plasma() -> Self {
        Self::from_hex(&[
            0x0d0887, 0x4b03a1, 0x7d03a8, 0xa82296, 0xcb4679, 0xe56b5d, 0xf89441, 0xfdc328,
            0xf0f921,
        ])
    }

    /// Turbo. Google's replacement for jet: the rainbow look, without the false
    /// banding jet puts at its cyan and yellow. Still a rainbow, so it is for
    /// depth maps and flow fields rather than for anything being compared.
    pub fn turbo() -> Self {
        Self::from_hex(&[
            0x30123b, 0x4145ab, 0x4675ed, 0x39a2fc, 0x1bcfd4, 0x62fc6b, 0xd1e834, 0xfe9b2d,
            0xdb3a07,
        ])
    }

    /// Blue–white–red, for data with a meaningful zero in the middle. The only
    /// diverging map here, and the one case where a two-ended scale is right.
    pub fn cool_warm() -> Self {
        Self::from_hex(&[
            0x3b4cc0, 0x6889ee, 0x9abbff, 0xc9d7f0, 0xedd1c2, 0xf7a889, 0xe26952, 0xb40426,
        ])
    }
}

/// Distinct colours for categories: d3's `scaleOrdinal` with a scheme.
/// Not interpolated — category 3 is the fourth colour, and past the end the
/// colours repeat.
#[derive(Debug, Clone, PartialEq)]
pub struct Palette {
    colors: Vec<Color>,
}

impl Palette {
    pub fn new(colors: Vec<Color>) -> Self {
        Self { colors }
    }

    pub fn from_hex(colors: &[u32]) -> Self {
        Self::new(colors.iter().map(|&c| Color::from_hex(c)).collect())
    }

    /// A named qualitative scheme: `"category10"` (also `"tableau10"`),
    /// `"tableau20"`, `"set1"`, `"set2"`, `"set3"`, `"pastel1"`, `"pastel2"`,
    /// `"paired"`, `"dark2"`, `"accent"`. Or `n` evenly spaced colours of a
    /// named ramp, with [`from_ramp`](Self::from_ramp).
    pub fn named(name: &str) -> Option<Self> {
        let key = scheme_key(name);
        CATEGORICAL.iter().find(|(n, _)| scheme_key(n) == key).map(|(_, c)| Self::from_hex(c))
    }

    /// d3's `schemeCategory10` (Tableau 10): the default for categories.
    pub fn category10() -> Self {
        Self::named("category10").expect("built in")
    }

    /// `n` colours evenly spaced along a ramp — a discrete sequential or
    /// diverging scheme, as ColorBrewer's `n`-class ones are.
    pub fn from_ramp(ramp: &ColorRamp, n: usize) -> Self {
        Self::new(ramp.sample(n))
    }

    /// Category `i`'s colour, repeating past the end.
    pub fn at(&self, i: usize) -> Color {
        if self.colors.is_empty() {
            Color::BLACK
        } else {
            self.colors[i % self.colors.len()]
        }
    }

    pub fn len(&self) -> usize {
        self.colors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    pub fn colors(&self) -> &[Color] {
        &self.colors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ramp_hits_its_own_stops() {
        let a = Color::from_hex(0x112233);
        let b = Color::from_hex(0xccddee);
        let ramp = ColorRamp::new(vec![(0.0, a), (1.0, b)]);
        assert_eq!(ramp.at(0.0).to_hex(), a.to_hex());
        assert_eq!(ramp.at(1.0).to_hex(), b.to_hex());
        // And clamps rather than extrapolating off the end.
        assert_eq!(ramp.at(-5.0).to_hex(), a.to_hex());
        assert_eq!(ramp.at(5.0).to_hex(), b.to_hex());
    }

    #[test]
    fn stops_do_not_have_to_arrive_in_order() {
        let ramp = ColorRamp::new(vec![
            (1.0, Color::from_hex(0xffffff)),
            (0.0, Color::from_hex(0x000000)),
        ]);
        assert!(ramp.at(0.9).luminance() > ramp.at(0.1).luminance());
    }

    /// The whole argument for interpolating in OkLab. Blue to yellow through
    /// linear light passes through a desaturated grey, because the midpoint of
    /// the two spectra genuinely is grey — and that is never what a gradient is
    /// meant to show.
    #[test]
    fn oklab_keeps_a_midpoint_colourful_where_linear_light_does_not() {
        let blue = Color::from_hex(0x0000ff);
        let yellow = Color::from_hex(0xffff00);
        let chroma = |c: Color| {
            let lab = c.to_oklab();
            (lab[1] * lab[1] + lab[2] * lab[2]).sqrt()
        };
        let perceptual = chroma(blue.lerp_oklab(yellow, 0.5));
        let physical = chroma(blue.lerp(yellow, 0.5));
        assert!(
            perceptual > physical * 1.8,
            "OkLab midpoint chroma {perceptual:.4} against linear {physical:.4}"
        );
    }

    #[test]
    fn oklab_round_trips() {
        for hex in [0x000000u32, 0xffffff, 0xff0000, 0x3498db, 0x7f7f7f, 0x12ab34] {
            let c = Color::from_hex(hex);
            let back = Color::from_oklab(c.to_oklab());
            assert!(
                (back.r - c.r).abs() < 1e-4
                    && (back.g - c.g).abs() < 1e-4
                    && (back.b - c.b).abs() < 1e-4,
                "{hex:06x} came back as ({}, {}, {})",
                back.r,
                back.g,
                back.b
            );
        }
    }

    #[test]
    fn hsl_round_trips_and_matches_known_colours() {
        // Pure red is hue 0, full saturation, half lightness.
        let hsl = Color::from_hex(0xff0000).to_hsl();
        assert!(hsl[0].abs() < 1e-3 || (hsl[0] - 1.0).abs() < 1e-3, "{hsl:?}");
        assert!((hsl[1] - 1.0).abs() < 1e-3 && (hsl[2] - 0.5).abs() < 1e-3, "{hsl:?}");

        for hex in [0xff0000u32, 0x00ff00, 0x0000ff, 0x808080, 0x3498db, 0xffffff] {
            let c = Color::from_hex(hex);
            let [h, s, l] = c.to_hsl();
            assert_eq!(
                Color::from_hsl(h, s, l).to_hex(),
                hex,
                "{hex:06x} did not survive HSL"
            );
        }
    }

    #[test]
    fn hex_round_trips() {
        for hex in [0x000000u32, 0xffffff, 0x123456, 0xfedcba, 0x808080] {
            assert_eq!(Color::from_hex(hex).to_hex(), hex, "{hex:06x}");
        }
    }

    /// What makes these palettes worth naming: lightness climbs monotonically,
    /// which is why they survive being printed in grey. A rainbow does not.
    #[test]
    fn the_sequential_palettes_climb_in_lightness() {
        for (name, ramp) in [
            ("viridis", ColorRamp::viridis()),
            ("magma", ColorRamp::magma()),
            ("inferno", ColorRamp::inferno()),
            ("plasma", ColorRamp::plasma()),
        ] {
            let lightness: Vec<f32> = ramp.sample(32).iter().map(|c| c.to_oklab()[0]).collect();
            for w in lightness.windows(2) {
                assert!(
                    w[1] >= w[0] - 0.012,
                    "{name} dips in lightness: {:.3} then {:.3}",
                    w[0],
                    w[1]
                );
            }
            let climb = lightness[lightness.len() - 1] - lightness[0];
            assert!(climb > 0.55, "{name} only climbs {climb:.2} in lightness");
        }
    }

    /// Turbo is a rainbow and is *not* expected to climb — the test above would
    /// be meaningless if every palette passed it.
    #[test]
    fn turbo_does_not_climb_in_lightness_and_is_not_meant_to() {
        let l: Vec<f32> = ColorRamp::turbo()
            .sample(32)
            .iter()
            .map(|c| c.to_oklab()[0])
            .collect();
        assert!(
            l.windows(2).any(|w| w[1] < w[0] - 0.02),
            "turbo came back monotonic, so the palette data is wrong"
        );
    }

    #[test]
    fn sample_spans_the_whole_ramp() {
        let ramp = ColorRamp::viridis();
        let s = ramp.sample(5);
        assert_eq!(s.len(), 5);
        assert_eq!(s[0].to_hex(), ramp.at(0.0).to_hex());
        assert_eq!(s[4].to_hex(), ramp.at(1.0).to_hex());
        assert_eq!(ramp.sample(1).len(), 1);
        assert!(ramp.sample(0).is_empty());
    }

    #[test]
    fn an_empty_ramp_is_black_rather_than_a_panic() {
        assert_eq!(ColorRamp::new(Vec::new()).at(0.5), Color::BLACK);
        let one = ColorRamp::new(vec![(0.3, Color::RED)]);
        assert_eq!(one.at(0.0), Color::RED);
        assert_eq!(one.at(1.0), Color::RED);
    }

    #[test]
    fn named_schemes_are_found_however_they_are_spelled() {
        for name in ["blues", "Blues", "interpolateBlues", "yl_gn_bu", "YlGnBu", "RdBu", "spectral", "cividis", "twilight", "viridis"] {
            assert!(ColorRamp::named(name).is_some(), "{name}");
        }
        assert!(ColorRamp::named("nope").is_none());
        // ColorBrewer's own swatches at the ends.
        let blues = ColorRamp::named("blues").unwrap();
        assert_eq!(blues.at(0.0).to_hex(), 0xf7fbff);
        assert_eq!(blues.at(1.0).to_hex(), 0x08306b);
        let rdbu = ColorRamp::named("RdBu").unwrap();
        assert_eq!(rdbu.at(0.5).to_hex(), 0xf7f7f7, "a diverging scheme is neutral in the middle");
        // Twilight is cyclic: its ends nearly meet.
        let tw = ColorRamp::twilight();
        let (a, b) = (tw.at(0.0).to_oklab(), tw.at(1.0).to_oklab());
        assert!((a[0] - b[0]).abs() < 0.03, "{a:?} {b:?}");
        assert!(ColorRamp::scheme_names().len() > 40);
    }

    #[test]
    fn palettes_cycle_and_ramps_reverse_and_slice() {
        let p = Palette::category10();
        assert_eq!(p.len(), 10);
        assert_eq!(p.at(0).to_hex(), 0x1f77b4);
        assert_eq!(p.at(10), p.at(0));
        assert_eq!(Palette::named("Set1").unwrap().at(0).to_hex(), 0xe41a1c);
        let v = ColorRamp::viridis();
        assert_eq!(v.clone().reversed().at(0.0).to_hex(), v.at(1.0).to_hex());
        let s = v.slice(0.25, 0.75);
        assert_eq!(s.at(0.0).to_hex(), v.at(0.25).to_hex());
        assert_eq!(s.at(1.0).to_hex(), v.at(0.75).to_hex());
    }
}
