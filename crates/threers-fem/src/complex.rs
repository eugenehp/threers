//! A complex number, as much of one as the Helmholtz solver needs.
//!
//! Time-harmonic fields here use the `e^{-iωt}` convention, so an outgoing
//! wave is `e^{+ikr}` — the physicists' sign, and the one most acoustics texts
//! outside electrical engineering use.

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// A complex number in double precision.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// `r·e^{iθ}`.
    pub fn from_polar(r: f64, theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        Self::new(r * c, r * s)
    }

    /// `e^{iθ}`.
    pub fn cis(theta: f64) -> Self {
        Self::from_polar(1.0, theta)
    }

    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// The modulus `|z|`.
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// `|z|²`, without the square root.
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// The argument, in `(-π, π]`.
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }

    pub fn exp(self) -> Self {
        Self::from_polar(self.re.exp(), self.im)
    }
}

impl From<f64> for Complex {
    fn from(re: f64) -> Self {
        Self::new(re, 0.0)
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

impl Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        self.scale(s)
    }
}

impl Div for Complex {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        // Smith's algorithm: no overflow from squaring a large denominator.
        if o.re.abs() >= o.im.abs() {
            let r = o.im / o.re;
            let d = o.re + o.im * r;
            Self::new((self.re + self.im * r) / d, (self.im - self.re * r) / d)
        } else {
            let r = o.re / o.im;
            let d = o.re * r + o.im;
            Self::new((self.re * r + self.im) / d, (self.im * r - self.re) / d)
        }
    }
}

impl Div<f64> for Complex {
    type Output = Self;
    fn div(self, s: f64) -> Self {
        Self::new(self.re / s, self.im / s)
    }
}

impl Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, o: Self) {
        self.re += o.re;
        self.im += o.im;
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, o: Self) {
        self.re -= o.re;
        self.im -= o.im;
    }
}

impl MulAssign for Complex {
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}

impl std::iter::Sum for Complex {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(-3.0, 0.5);
        let q = a / b;
        let back = q * b;
        assert!((back - a).abs() < 1e-15);
        assert!((Complex::cis(std::f64::consts::PI) + Complex::ONE).abs() < 1e-15);
    }
}
