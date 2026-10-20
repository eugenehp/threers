//! Data to colour: d3's `scaleSequential`, `scaleDiverging`, `scaleQuantize`,
//! `scaleQuantile` and `scaleThreshold`, over any [`ColorRamp`].
//!
//! ```
//! use threers::math::{ColorRamp, ColorScale};
//!
//! // Population density: log-scaled, so a city does not wash out the farms.
//! let density = ColorScale::sequential(ColorRamp::named("yl_or_rd").unwrap(), 1.0, 10_000.0).log(10.0);
//! let farm = density.map(12.0);
//! let city = density.map(8_000.0);
//!
//! // Temperature anomaly around zero.
//! let anomaly = ColorScale::diverging(ColorRamp::named("rd_bu").unwrap().reversed(), -3.0, 0.0, 3.0);
//!
//! // Five classes of equal count, for a choropleth.
//! let data = [3.0, 9.0, 1.0, 7.0, 4.0, 12.0, 6.0, 2.0];
//! let classes = ColorScale::quantile(ColorRamp::named("blues").unwrap(), &data, 5);
//! let rgba = classes.map_rgba8(&data);
//! ```

use super::{ticks, Color, ColorRamp};

/// How a [`ColorScale`] turns a value into a position on its ramp.
#[derive(Debug, Clone, PartialEq)]
pub enum ScaleKind {
    /// In proportion.
    Linear,
    /// By logarithm, for data spanning orders of magnitude. The domain must
    /// not cross zero (an all-negative one works, mirrored).
    Log { base: f32 },
    /// By a power: below 1 spreads the low end, above 1 the high end.
    Pow { exponent: f32 },
    /// Symmetric log: logarithmic far from zero, linear within `constant` of
    /// it, so data crossing zero still works.
    Symlog { constant: f32 },
    /// `classes` equal-width bins of the domain, a colour each.
    Quantize { classes: usize },
    /// Bins holding equal counts of the data, split at these values.
    Quantile { thresholds: Vec<f32> },
    /// Explicit class breaks: below the first, between each pair, above the
    /// last — one colour more than breaks.
    Threshold { breaks: Vec<f32> },
}

/// A mapping from numbers to colours.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorScale {
    ramp: ColorRamp,
    /// `[lo, hi]`, or `[lo, mid, hi]` for a diverging scale.
    domain: Vec<f32>,
    kind: ScaleKind,
    clamp: bool,
    unknown: Color,
}

impl ColorScale {
    /// Low to high over `lo..hi` — d3's `scaleSequential`.
    pub fn sequential(ramp: ColorRamp, lo: f32, hi: f32) -> Self {
        Self {
            ramp,
            domain: vec![lo, hi],
            kind: ScaleKind::Linear,
            clamp: true,
            unknown: Color::BLACK,
        }
    }

    /// Away from `mid` in both directions, `lo` at the ramp's start and `hi`
    /// at its end, `mid` at its middle — d3's `scaleDiverging`. The two
    /// halves stretch independently, so an asymmetric domain still puts the
    /// ramp's neutral colour on `mid`.
    pub fn diverging(ramp: ColorRamp, lo: f32, mid: f32, hi: f32) -> Self {
        Self {
            ramp,
            domain: vec![lo, mid, hi],
            kind: ScaleKind::Linear,
            clamp: true,
            unknown: Color::BLACK,
        }
    }

    /// `classes` equal-width bins over `lo..hi` — d3's `scaleQuantize`.
    pub fn quantize(ramp: ColorRamp, lo: f32, hi: f32, classes: usize) -> Self {
        Self {
            kind: ScaleKind::Quantize {
                classes: classes.max(1),
            },
            ..Self::sequential(ramp, lo, hi)
        }
    }

    /// `classes` bins holding equal counts of `data` — d3's `scaleQuantile`:
    /// a choropleth whose colours are each used about as often.
    pub fn quantile(ramp: ColorRamp, data: &[f32], classes: usize) -> Self {
        let mut sorted: Vec<f32> = data.iter().copied().filter(|v| v.is_finite()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let classes = classes.max(1);
        let thresholds = (1..classes)
            .map(|i| {
                if sorted.is_empty() {
                    return 0.0;
                }
                // d3's quantile: linear between order statistics.
                let p = i as f32 / classes as f32 * (sorted.len() - 1) as f32;
                let (k, f) = (p.floor() as usize, p.fract());
                let next = sorted[(k + 1).min(sorted.len() - 1)];
                sorted[k] + (next - sorted[k]) * f
            })
            .collect();
        let (lo, hi) = (
            sorted.first().copied().unwrap_or(0.0),
            sorted.last().copied().unwrap_or(1.0),
        );
        Self {
            kind: ScaleKind::Quantile { thresholds },
            ..Self::sequential(ramp, lo, hi)
        }
    }

    /// Explicit class breaks — d3's `scaleThreshold`: values below
    /// `breaks[0]` get the ramp's start, above the last its end, and the
    /// classes between evenly spaced colours.
    pub fn threshold(ramp: ColorRamp, mut breaks: Vec<f32>) -> Self {
        breaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let (lo, hi) = (
            breaks.first().copied().unwrap_or(0.0),
            breaks.last().copied().unwrap_or(1.0),
        );
        Self {
            kind: ScaleKind::Threshold { breaks },
            ..Self::sequential(ramp, lo, hi)
        }
    }

    /// Logarithmic, base `base`.
    pub fn log(mut self, base: f32) -> Self {
        self.kind = ScaleKind::Log {
            base: base.max(1.0001),
        };
        self
    }

    /// By a power of the value.
    pub fn pow(mut self, exponent: f32) -> Self {
        self.kind = ScaleKind::Pow { exponent };
        self
    }

    /// By the square root: area-like data, where a value's colour should
    /// grow as its radius would.
    pub fn sqrt(self) -> Self {
        self.pow(0.5)
    }

    /// Symmetric log, linear within `constant` of zero.
    pub fn symlog(mut self, constant: f32) -> Self {
        self.kind = ScaleKind::Symlog {
            constant: constant.max(1e-12),
        };
        self
    }

    /// Whether values outside the domain stop at its ends (the default) or
    /// carry on — which a ramp's own clamping ends at its last stop anyway.
    pub fn clamped(mut self, on: bool) -> Self {
        self.clamp = on;
        self
    }

    /// The colour for a value that is not a number (d3's `unknown`).
    pub fn with_unknown(mut self, color: Color) -> Self {
        self.unknown = color;
        self
    }

    pub fn ramp(&self) -> &ColorRamp {
        &self.ramp
    }

    pub fn domain(&self) -> &[f32] {
        &self.domain
    }

    pub fn kind(&self) -> &ScaleKind {
        &self.kind
    }

    fn transform(&self, v: f32) -> f32 {
        match self.kind {
            ScaleKind::Log { base } => {
                let sign = if self.domain[0] < 0.0 { -1.0 } else { 1.0 };
                sign * (sign * v).max(f32::MIN_POSITIVE).ln() / base.ln()
            }
            ScaleKind::Pow { exponent } => v.signum() * v.abs().powf(exponent),
            ScaleKind::Symlog { constant } => v.signum() * (v.abs() / constant).ln_1p(),
            _ => v,
        }
    }

    /// Where `v` falls along the ramp, `0..1` (outside it when not clamped).
    pub fn position(&self, v: f32) -> f32 {
        if !v.is_finite() {
            return f32::NAN;
        }
        let classed = |k: usize, classes: usize| {
            if classes <= 1 {
                0.5
            } else {
                k as f32 / (classes - 1) as f32
            }
        };
        match &self.kind {
            ScaleKind::Quantize { classes } => {
                let t = ((v - self.domain[0]) / (self.domain[1] - self.domain[0])).clamp(0.0, 1.0);
                classed(((t * *classes as f32) as usize).min(classes - 1), *classes)
            }
            ScaleKind::Quantile { thresholds } => classed(
                thresholds.iter().filter(|&&b| v >= b).count(),
                thresholds.len() + 1,
            ),
            ScaleKind::Threshold { breaks } => {
                classed(breaks.iter().filter(|&&b| v >= b).count(), breaks.len() + 1)
            }
            _ => {
                let f = |x: f32| self.transform(x);
                let t = if self.domain.len() == 3 {
                    let (lo, mid, hi) = (f(self.domain[0]), f(self.domain[1]), f(self.domain[2]));
                    let x = f(v);
                    if (x < mid) == (lo < mid) {
                        0.5 * (x - lo) / (mid - lo)
                    } else {
                        0.5 + 0.5 * (x - mid) / (hi - mid)
                    }
                } else {
                    let (lo, hi) = (f(self.domain[0]), f(self.domain[1]));
                    (f(v) - lo) / (hi - lo)
                };
                if self.clamp {
                    t.clamp(0.0, 1.0)
                } else {
                    t
                }
            }
        }
    }

    /// The colour for `v`.
    pub fn map(&self, v: f32) -> Color {
        let t = self.position(v);
        if t.is_nan() {
            self.unknown
        } else {
            self.ramp.at(t)
        }
    }

    /// Colours for `values` as sRGB bytes with full alpha — what marks layers
    /// and data textures take.
    pub fn map_rgba8(&self, values: &[f32]) -> Vec<[u8; 4]> {
        values
            .iter()
            .map(|&v| {
                let [r, g, b] = self.map(v).to_srgb();
                let q = |c: f32| (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                [q(r), q(g), q(b), 255]
            })
            .collect()
    }

    /// The scale's classes as `(from, to, colour)` — the rows of a legend.
    /// Continuous scales give `n` steps across the domain at round values.
    pub fn legend(&self, n: usize) -> Vec<(f32, f32, Color)> {
        let (lo, hi) = (self.domain[0], *self.domain.last().unwrap());
        let edges: Vec<f32> = match &self.kind {
            ScaleKind::Quantize { classes } => (0..=*classes)
                .map(|i| lo + (hi - lo) * i as f32 / *classes as f32)
                .collect(),
            ScaleKind::Quantile { thresholds } | ScaleKind::Threshold { breaks: thresholds } => {
                let mut e = vec![lo.min(thresholds.first().copied().unwrap_or(lo))];
                e.extend(thresholds.iter().copied());
                e.push(hi.max(thresholds.last().copied().unwrap_or(hi)));
                e
            }
            _ => {
                let mut t = ticks(lo, hi, n.max(2));
                if t.first() != Some(&lo) {
                    t.insert(0, lo);
                }
                if t.last() != Some(&hi) {
                    t.push(hi);
                }
                t
            }
        };
        edges
            .windows(2)
            .map(|w| (w[0], w[1], self.map((w[0] + w[1]) * 0.5)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn greys() -> ColorRamp {
        ColorRamp::new(vec![(0.0, Color::BLACK), (1.0, Color::WHITE)]).in_linear_rgb()
    }

    #[test]
    fn continuous_scales_place_values_as_d3_does() {
        let lin = ColorScale::sequential(greys(), 0.0, 10.0);
        assert!((lin.position(2.5) - 0.25).abs() < 1e-6);
        assert_eq!(lin.position(-5.0), 0.0, "clamped");
        assert_eq!(lin.clone().clamped(false).position(20.0), 2.0);
        let log = ColorScale::sequential(greys(), 1.0, 1000.0).log(10.0);
        assert!((log.position(10.0) - 1.0 / 3.0).abs() < 1e-5);
        assert!((log.position(100.0) - 2.0 / 3.0).abs() < 1e-5);
        let sqrt = ColorScale::sequential(greys(), 0.0, 100.0).sqrt();
        assert!((sqrt.position(25.0) - 0.5).abs() < 1e-5);
        let sym = ColorScale::sequential(greys(), -100.0, 100.0).symlog(1.0);
        assert!((sym.position(0.0) - 0.5).abs() < 1e-5);
        assert!(sym.position(-10.0) < 0.5 && sym.position(10.0) > 0.5);
        assert!(lin.map(f32::NAN) == Color::BLACK);
    }

    #[test]
    fn diverging_puts_the_middle_on_mid_whatever_the_ends() {
        let s = ColorScale::diverging(greys(), -1.0, 0.0, 10.0);
        assert!((s.position(0.0) - 0.5).abs() < 1e-6);
        assert!((s.position(-0.5) - 0.25).abs() < 1e-6);
        assert!((s.position(5.0) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn classed_scales_bin_their_values() {
        let q = ColorScale::quantize(greys(), 0.0, 10.0, 5);
        assert_eq!(q.position(0.0), 0.0);
        assert_eq!(q.position(1.9), 0.0);
        assert_eq!(q.position(2.1), 0.25);
        assert_eq!(q.position(10.0), 1.0);
        let data: Vec<f32> = (1..=100).map(|v| v as f32).collect();
        let quant = ColorScale::quantile(greys(), &data, 4);
        let ScaleKind::Quantile { thresholds } = quant.kind() else {
            unreachable!()
        };
        assert_eq!(thresholds.len(), 3);
        assert!((thresholds[1] - 50.5).abs() < 1e-4, "{thresholds:?}");
        // Each class holds a quarter of the data.
        for k in 0..4 {
            let n = data
                .iter()
                .filter(|&&v| (quant.position(v) * 3.0).round() as usize == k)
                .count();
            assert_eq!(n, 25, "class {k}");
        }
        let t = ColorScale::threshold(greys(), vec![0.0, 1.0]);
        assert_eq!(t.position(-5.0), 0.0);
        assert_eq!(t.position(0.5), 0.5);
        assert_eq!(t.position(7.0), 1.0);
        assert_eq!(t.legend(5).len(), 3);
    }

    #[test]
    fn bytes_and_legends() {
        let s = ColorScale::sequential(greys(), 0.0, 1.0);
        let px = s.map_rgba8(&[0.0, 1.0]);
        assert_eq!(px, vec![[0, 0, 0, 255], [255, 255, 255, 255]]);
        let legend = ColorScale::sequential(greys(), 0.0, 100.0).legend(5);
        assert!(
            legend.len() >= 4 && legend[0].0 == 0.0 && legend.last().unwrap().1 == 100.0,
            "{legend:?}"
        );
    }
}
