//! The few places JavaScript's numbers differ from Rust's, reproduced so the
//! solver computes what its JavaScript original does, bit for bit where it
//! can (see `tests/softbody_parity.rs`).

/// `Math.hypot` as V8 computes it: scaled by the largest magnitude and summed
/// with Kahan compensation — not `f64::hypot`, which rounds differently.
pub fn hypot(v: &[f64]) -> f64 {
    let mut max = 0.0f64;
    for &x in v {
        let a = x.abs();
        if a.is_infinite() {
            return f64::INFINITY;
        }
        if a.is_nan() {
            return f64::NAN;
        }
        if a > max {
            max = a;
        }
    }
    if max == 0.0 {
        return 0.0;
    }
    let (mut sum, mut compensation) = (0.0f64, 0.0f64);
    for &x in v {
        let n = x.abs() / max;
        let summand = n * n - compensation;
        let preliminary = sum + summand;
        compensation = (preliminary - sum) - summand;
        sum = preliminary;
    }
    sum.sqrt() * max
}

#[inline]
pub fn hypot3(x: f64, y: f64, z: f64) -> f64 {
    hypot(&[x, y, z])
}

/// fdlibm, as V8's `Math.sin`, `Math.cos`, `Math.exp` and `Math.pow` are.
#[inline]
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}
#[inline]
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}
#[inline]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

/// JavaScript's `ToInt32`, as `^` and `|` apply it to a number.
#[inline]
pub fn to_int32(x: f64) -> i32 {
    if !x.is_finite() {
        return 0;
    }
    (x.trunc().rem_euclid(4294967296.0) as u64 as u32) as i32
}

/// A source of `Math.random()`-like numbers in [0, 1). The solver draws from
/// it where its original called `Math.random` (the goo a tear throws), so a
/// test can seed both sides alike.
pub trait Random {
    fn next(&mut self) -> f64;
}

/// A 32-bit linear congruential generator: small, and trivially the same in
/// JavaScript — `s = (Math.imul(s, 1664525) + 1013904223) >>> 0; s / 2**32`.
#[derive(Debug, Clone)]
pub struct Lcg(pub u32);

impl Random for Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 as f64 / 4294967296.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hypot_is_v8s() {
        // Values where the compensated sum and a plain sqrt part ways.
        assert_eq!(hypot(&[3.0, 4.0]), 5.0);
        assert_eq!(hypot(&[0.0, 0.0, 0.0]), 0.0);
        assert_eq!(hypot(&[-1e300, 1e300]), 1.4142135623730952e300);
        assert!(hypot(&[f64::NAN, 1.0]).is_nan());
    }

    #[test]
    fn to_int32_wraps() {
        assert_eq!(to_int32(4294967296.0 + 5.0), 5);
        assert_eq!(to_int32(2147483648.0), i32::MIN);
        assert_eq!(to_int32(-1.5), -1);
        assert_eq!(to_int32(-2147483649.0), i32::MAX);
    }
}
