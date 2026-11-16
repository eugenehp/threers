//! Angles in radians: normalising, differences, and sweeps.

use std::f64::consts::{PI, TAU};

/// `a` brought into `[0, 2π)`.
pub fn normalize(a: f64) -> f64 {
    let r = a.rem_euclid(TAU);

    // rem_euclid can round up to TAU itself
    if r >= TAU {
        0.0
    } else {
        r
    }
}

/// `a` brought into `(-π, π]`.
pub fn normalize_signed(a: f64) -> f64 {
    let r = normalize(a);

    if r > PI {
        r - TAU
    } else {
        r
    }
}

/// The counter-clockwise sweep from `from` to `to`, in `[0, 2π)`.
pub fn ccw_sweep(from: f64, to: f64) -> f64 {
    normalize(to - from)
}

/// Whether `a` lies on the sweep that starts at `start` and runs `sweep`
/// (negative runs clockwise), ends included.
pub fn in_sweep(a: f64, start: f64, sweep: f64) -> bool {
    if sweep.abs() >= TAU {
        return true;
    }

    if sweep >= 0.0 {
        ccw_sweep(start, a) <= sweep
    } else {
        ccw_sweep(a, start) <= -sweep
    }
}

/// Degrees to radians.
pub fn radians(degrees: f64) -> f64 {
    degrees.to_radians()
}

/// Radians to degrees.
pub fn degrees(radians: f64) -> f64 {
    radians.to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizing() {
        assert!((normalize(-PI / 2.0) - 1.5 * PI).abs() < 1e-12);
        assert_eq!(normalize(TAU), 0.0);
        assert!((normalize_signed(1.5 * PI) + PI / 2.0).abs() < 1e-12);
        assert!((normalize_signed(PI) - PI).abs() < 1e-12);
    }

    #[test]
    fn sweeps() {
        assert!(in_sweep(0.1, -0.5, 1.0));
        assert!(!in_sweep(0.6, -0.5, 1.0));
        assert!(in_sweep(-0.4, 0.0, -0.5));
        assert!(!in_sweep(0.1, 0.0, -0.5));
        assert!(in_sweep(3.0, 0.0, TAU));
    }
}
