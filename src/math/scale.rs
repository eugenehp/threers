//! Mapping numbers from one range to another — d3's scales, and p5's `map`.
//!
//! The single most-reached-for function in either library, and this crate had
//! no spelling of it at all: not `map_range`, not `remap`, not `inverse_lerp`.
//! Every generator here that sizes a lattice by density or colours a planet by
//! elevation was writing `lo + (hi - lo) * ((v - a) / (b - a))` by hand, which
//! is four chances to get a sign wrong.
//!
//! ```
//! use threers::math::{map_range, ScaleLinear};
//!
//! // p5's `map`, with the same argument order.
//! assert_eq!(map_range(5.0, 0.0, 10.0, 0.0, 100.0), 50.0);
//!
//! // d3's scale, when the same mapping is used more than once.
//! let x = ScaleLinear::new([0.0, 100.0], [0.0, 640.0]).clamped();
//! assert_eq!(x.get(50.0), 320.0);
//! assert_eq!(x.get(200.0), 640.0);   // clamped
//! assert_eq!(x.invert(320.0), 50.0); // and it runs backwards
//! ```

/// Where `v` sits between `a` and `b`, as 0..1. The inverse of [`lerp`].
///
/// Returns 0 for a degenerate range rather than infinity, which is almost
/// always what a caller wants and never what the division gives.
#[inline]
pub fn inverse_lerp(v: f32, a: f32, b: f32) -> f32 {
    if (b - a).abs() < f32::EPSILON {
        0.0
    } else {
        (v - a) / (b - a)
    }
}

/// `a` to `b` by `t`, unclamped.
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Map `v` from `[in_lo, in_hi]` onto `[out_lo, out_hi]`, unclamped.
///
/// p5's `map`, argument for argument. Use [`map_range_clamped`] when values
/// outside the input range should stop at the ends rather than run past them.
#[inline]
pub fn map_range(v: f32, in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32) -> f32 {
    lerp(out_lo, out_hi, inverse_lerp(v, in_lo, in_hi))
}

/// [`map_range`], with the result held inside `[out_lo, out_hi]`.
///
/// Handles a reversed output range, which an ordinary `clamp` does not: mapping
/// onto `[1.0, 0.0]` to invert something is a normal thing to want.
#[inline]
pub fn map_range_clamped(v: f32, in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32) -> f32 {
    let out = map_range(v, in_lo, in_hi, out_lo, out_hi);
    if out_lo <= out_hi {
        out.clamp(out_lo, out_hi)
    } else {
        out.clamp(out_hi, out_lo)
    }
}

/// Hermite smoothstep: 0 below `e0`, 1 above `e1`, an S between.
#[inline]
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = inverse_lerp(x, e0, e1).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Ken Perlin's sixth-order smoothstep, whose *second* derivative is also zero
/// at the ends. Worth the extra multiply wherever the result drives a normal or
/// a displacement, because a discontinuity in curvature shows up as a visible
/// crease that plain smoothstep leaves behind.
#[inline]
pub fn smootherstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = inverse_lerp(x, e0, e1).clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// How a scale treats inputs outside its domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Clamping {
    #[default]
    Extrapolate,
    Clamp,
}

/// A linear mapping from a data domain onto an output range.
///
/// d3's `scaleLinear`. Worth having over [`map_range`] when the same mapping is
/// applied repeatedly, when it has to run backwards, or when the ticks matter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleLinear {
    domain: [f32; 2],
    range: [f32; 2],
    clamping: Clamping,
}

impl ScaleLinear {
    pub fn new(domain: [f32; 2], range: [f32; 2]) -> Self {
        Self {
            domain,
            range,
            clamping: Clamping::Extrapolate,
        }
    }

    /// Hold the output inside the range. Off by default, as in d3.
    pub fn clamped(mut self) -> Self {
        self.clamping = Clamping::Clamp;
        self
    }

    pub fn get(&self, v: f32) -> f32 {
        match self.clamping {
            Clamping::Extrapolate => {
                map_range(v, self.domain[0], self.domain[1], self.range[0], self.range[1])
            }
            Clamping::Clamp => map_range_clamped(
                v,
                self.domain[0],
                self.domain[1],
                self.range[0],
                self.range[1],
            ),
        }
    }

    /// The input that would produce `v`. Exact for a linear scale.
    pub fn invert(&self, v: f32) -> f32 {
        map_range(v, self.range[0], self.range[1], self.domain[0], self.domain[1])
    }

    pub fn domain(&self) -> [f32; 2] {
        self.domain
    }

    pub fn range(&self) -> [f32; 2] {
        self.range
    }

    /// Round the domain outwards to the nearest round numbers, so the ends of
    /// an axis land somewhere a human would have chosen.
    pub fn nice(mut self, count: usize) -> Self {
        let step = tick_step(self.domain[0], self.domain[1], count);
        if step > 0.0 && step.is_finite() {
            let (lo, hi) = (self.domain[0].min(self.domain[1]), self.domain[0].max(self.domain[1]));
            let (nlo, nhi) = ((lo / step).floor() * step, (hi / step).ceil() * step);
            self.domain = if self.domain[0] <= self.domain[1] {
                [nlo, nhi]
            } else {
                [nhi, nlo]
            };
        }
        self
    }

    /// About `count` round values spanning the domain. d3's `ticks`.
    pub fn ticks(&self, count: usize) -> Vec<f32> {
        ticks(self.domain[0], self.domain[1], count)
    }
}

/// A logarithmic mapping, for data that spans orders of magnitude.
///
/// The domain must not contain or touch zero — a log scale through zero has no
/// meaning, and an all-negative domain is handled by mapping its magnitudes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleLog {
    domain: [f32; 2],
    range: [f32; 2],
    clamping: Clamping,
}

impl ScaleLog {
    /// `None` when the domain contains zero or straddles it.
    pub fn new(domain: [f32; 2], range: [f32; 2]) -> Option<Self> {
        if domain[0] == 0.0 || domain[1] == 0.0 || domain[0].signum() != domain[1].signum() {
            return None;
        }
        Some(Self {
            domain,
            range,
            clamping: Clamping::Extrapolate,
        })
    }

    pub fn clamped(mut self) -> Self {
        self.clamping = Clamping::Clamp;
        self
    }

    pub fn get(&self, v: f32) -> f32 {
        let sign = self.domain[0].signum();
        let (a, b, x) = (
            (self.domain[0] * sign).ln(),
            (self.domain[1] * sign).ln(),
            (v * sign).max(f32::MIN_POSITIVE).ln(),
        );
        match self.clamping {
            Clamping::Extrapolate => map_range(x, a, b, self.range[0], self.range[1]),
            Clamping::Clamp => map_range_clamped(x, a, b, self.range[0], self.range[1]),
        }
    }

    pub fn invert(&self, v: f32) -> f32 {
        let sign = self.domain[0].signum();
        let (a, b) = ((self.domain[0] * sign).ln(), (self.domain[1] * sign).ln());
        map_range(v, self.range[0], self.range[1], a, b).exp() * sign
    }
}

/// A power mapping, `v^exponent`. `ScalePow::new(d, r, 0.5)` is d3's `scaleSqrt`
/// — the right scale for anything encoded as an *area*, because area grows as
/// the square of a radius and a linear radius scale over-reads by that much.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalePow {
    domain: [f32; 2],
    range: [f32; 2],
    exponent: f32,
    clamping: Clamping,
}

impl ScalePow {
    pub fn new(domain: [f32; 2], range: [f32; 2], exponent: f32) -> Self {
        Self {
            domain,
            range,
            exponent,
            clamping: Clamping::Extrapolate,
        }
    }

    pub fn clamped(mut self) -> Self {
        self.clamping = Clamping::Clamp;
        self
    }

    pub fn get(&self, v: f32) -> f32 {
        // Signed power, so a domain through zero behaves symmetrically rather
        // than returning NaN on the negative side.
        let p = |x: f32| x.abs().powf(self.exponent) * x.signum();
        let t = inverse_lerp(p(v), p(self.domain[0]), p(self.domain[1]));
        let out = lerp(self.range[0], self.range[1], t);
        match self.clamping {
            Clamping::Extrapolate => out,
            Clamping::Clamp => {
                if self.range[0] <= self.range[1] {
                    out.clamp(self.range[0], self.range[1])
                } else {
                    out.clamp(self.range[1], self.range[0])
                }
            }
        }
    }
}

/// The step between round ticks spanning `[lo, hi]` in about `count` of them.
///
/// d3's algorithm: take the raw step, then round it to 1, 2, 5 or 10 times a
/// power of ten, because those are the intervals people read without effort.
pub fn tick_step(lo: f32, hi: f32, count: usize) -> f32 {
    let count = count.max(1) as f32;
    let span = (hi - lo).abs();
    if span == 0.0 || !span.is_finite() {
        return 0.0;
    }
    let raw = span / count;
    let power = raw.log10().floor();
    let base = 10f32.powf(power);
    let err = raw / base;
    base * if err >= 7.071 {
        10.0
    } else if err >= 3.162 {
        5.0
    } else if err >= 1.414 {
        2.0
    } else {
        1.0
    }
}

/// Round values spanning `[lo, hi]`, about `count` of them. d3's `ticks`.
///
/// The count is approximate on purpose: forcing exactly N ticks means they land
/// on values like 3.33, which is worse than having six of them instead of five.
pub fn ticks(lo: f32, hi: f32, count: usize) -> Vec<f32> {
    let step = tick_step(lo, hi, count);
    if step <= 0.0 || !step.is_finite() {
        return Vec::new();
    }
    let (a, b) = (lo.min(hi), lo.max(hi));
    let first = (a / step).ceil() as i64;
    let last = (b / step).floor() as i64;
    if last < first {
        return Vec::new();
    }
    let mut out: Vec<f32> = (first..=last).map(|i| i as f32 * step).collect();
    if lo > hi {
        out.reverse();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_range_matches_the_arithmetic_it_replaces() {
        assert_eq!(map_range(5.0, 0.0, 10.0, 0.0, 100.0), 50.0);
        assert_eq!(map_range(0.0, 0.0, 10.0, 20.0, 30.0), 20.0);
        assert_eq!(map_range(10.0, 0.0, 10.0, 20.0, 30.0), 30.0);
        // Outside the domain it extrapolates, as p5's `map` does.
        assert_eq!(map_range(20.0, 0.0, 10.0, 0.0, 100.0), 200.0);
        // A reversed output range inverts.
        assert_eq!(map_range(0.25, 0.0, 1.0, 1.0, 0.0), 0.75);
    }

    /// A degenerate input range is a division by zero. Returning the low end
    /// beats returning infinity, and silently producing NaN is the worst of
    /// the three.
    #[test]
    fn a_zero_width_domain_does_not_produce_nan() {
        let v = map_range(5.0, 3.0, 3.0, 0.0, 100.0);
        assert!(v.is_finite(), "got {v}");
        assert_eq!(v, 0.0);
        assert_eq!(inverse_lerp(5.0, 3.0, 3.0), 0.0);
    }

    #[test]
    fn clamping_handles_a_reversed_output_range() {
        // `f32::clamp` panics when min > max, so this is not a free property.
        assert_eq!(map_range_clamped(2.0, 0.0, 1.0, 1.0, 0.0), 0.0);
        assert_eq!(map_range_clamped(-1.0, 0.0, 1.0, 1.0, 0.0), 1.0);
        assert_eq!(map_range_clamped(5.0, 0.0, 10.0, 0.0, 100.0), 50.0);
    }

    #[test]
    fn a_linear_scale_inverts_itself() {
        let s = ScaleLinear::new([0.0, 100.0], [0.0, 640.0]);
        for v in [0.0f32, 1.0, 33.3, 99.9, 100.0] {
            assert!((s.invert(s.get(v)) - v).abs() < 1e-3, "round trip at {v}");
        }
        assert_eq!(s.clamped().get(200.0), 640.0);
        assert_eq!(s.get(200.0), 1280.0, "unclamped by default, as in d3");
    }

    #[test]
    fn a_log_scale_spaces_decades_evenly() {
        let s = ScaleLog::new([1.0, 1000.0], [0.0, 3.0]).expect("valid domain");
        assert!((s.get(1.0) - 0.0).abs() < 1e-4);
        assert!((s.get(10.0) - 1.0).abs() < 1e-4);
        assert!((s.get(100.0) - 2.0).abs() < 1e-4);
        assert!((s.get(1000.0) - 3.0).abs() < 1e-4);
        assert!((s.invert(2.0) - 100.0).abs() < 1e-2);
        // A domain through zero has no logarithm and says so.
        assert!(ScaleLog::new([0.0, 10.0], [0.0, 1.0]).is_none());
        assert!(ScaleLog::new([-1.0, 10.0], [0.0, 1.0]).is_none());
    }

    /// The reason `scaleSqrt` exists: a value encoded as the *area* of a disc
    /// needs the square root, or a four-times-bigger number draws sixteen times
    /// the ink.
    #[test]
    fn a_sqrt_scale_encodes_area_rather_than_radius() {
        let s = ScalePow::new([0.0, 100.0], [0.0, 10.0], 0.5);
        let r_small = s.get(25.0);
        let r_big = s.get(100.0);
        assert!((r_big / r_small - 2.0).abs() < 1e-4, "radius should double");
        let area_ratio = (r_big * r_big) / (r_small * r_small);
        assert!((area_ratio - 4.0).abs() < 1e-3, "area should quadruple");
    }

    #[test]
    fn ticks_land_on_round_numbers() {
        assert_eq!(ticks(0.0, 10.0, 5), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        // Compared with a tolerance: the step is `i * 0.2` in f32, and whether
        // that lands exactly on 0.6 is a property of the rounding, not of the
        // algorithm being tested.
        let fine = ticks(0.0, 1.0, 5);
        assert_eq!(fine.len(), 6);
        for (got, want) in fine.iter().zip([0.0f32, 0.2, 0.4, 0.6, 0.8, 1.0]) {
            assert!((got - want).abs() < 1e-5, "{got} is not about {want}");
        }
        // Descending input gives descending ticks.
        let down = ticks(10.0, 0.0, 5);
        assert_eq!(down.first(), Some(&10.0));
        assert_eq!(down.last(), Some(&0.0));
        // Degenerate spans have no ticks rather than an infinite loop.
        assert!(ticks(3.0, 3.0, 5).is_empty());
    }

    #[test]
    fn nice_rounds_the_domain_outwards() {
        let s = ScaleLinear::new([0.3, 9.4], [0.0, 1.0]).nice(5);
        let [lo, hi] = s.domain();
        assert!(lo <= 0.3 && hi >= 9.4, "nice must not shrink the domain");
        assert_eq!([lo, hi], [0.0, 10.0]);
    }

    #[test]
    fn smootherstep_is_flatter_at_the_ends_than_smoothstep() {
        // The second derivative is what separates them, so compare slopes close
        // to the boundary rather than values in the middle.
        let d = 1e-3;
        let slope = |f: fn(f32, f32, f32) -> f32| (f(0.0, 1.0, d) - f(0.0, 1.0, 0.0)) / d;
        assert!(
            slope(smootherstep) < slope(smoothstep),
            "smootherstep should leave 0 more gently"
        );
        for f in [smoothstep as fn(f32, f32, f32) -> f32, smootherstep] {
            assert_eq!(f(0.0, 1.0, -1.0), 0.0);
            assert_eq!(f(0.0, 1.0, 2.0), 1.0);
            assert!((f(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
        }
    }
}
