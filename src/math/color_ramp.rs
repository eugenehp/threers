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

use crate::math::Color;

/// How a ramp blends between its stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RampSpace {
    /// Through OkLab: perceptually even, and the midpoint of two colours is the
    /// one a person would name. The default, and what a gradient wants.
    #[default]
    Oklab,
    /// Through linear light: physically right for adding up light, and wrong
    /// for a gradient, because the midpoint of two saturated colours is grey.
    LinearRgb,
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

    /// The colour at `t`, clamped to the ends outside `0..1`.
    pub fn at(&self, t: f32) -> Color {
        match self.stops.len() {
            0 => Color::BLACK,
            1 => self.stops[0].1,
            _ => {
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
                match self.space {
                    RampSpace::Oklab => c0.lerp_oklab(c1, local),
                    RampSpace::LinearRgb => c0.lerp(c1, local),
                }
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
}
