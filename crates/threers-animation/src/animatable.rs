//! What can be animated.
//!
//! [`Animatable`] is the one trait everything here is generic over. Implement it
//! for your own types and they work with [`crate::Tween`], [`crate::Spring`] and
//! [`crate::Timeline`] unchanged.

use threers::math::{Color, Quaternion, Vector2, Vector3, Vector4};

/// A value that can be interpolated.
///
/// The contract is only that `lerp(a, b, 0) == a` and `lerp(a, b, 1) == b`. What
/// happens between is up to the type — rotations take the shortest arc rather
/// than interpolating four numbers independently, which is why this is a trait
/// and not a blanket implementation over arithmetic.
pub trait Animatable: Copy {
    /// Interpolate toward `other`. `t` may fall outside `0..1` when an
    /// overshooting easing curve is in play, and implementations must cope.
    fn lerp(self, other: Self, t: f32) -> Self;
}

impl Animatable for f32 {
    fn lerp(self, other: Self, t: f32) -> Self {
        self + (other - self) * t
    }
}

impl Animatable for f64 {
    fn lerp(self, other: Self, t: f32) -> Self {
        self + (other - self) * t as f64
    }
}

impl Animatable for Vector2 {
    fn lerp(self, other: Self, t: f32) -> Self {
        Vector2::new(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
        )
    }
}

impl Animatable for Vector3 {
    fn lerp(self, other: Self, t: f32) -> Self {
        Vector3::new(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
        )
    }
}

impl Animatable for Vector4 {
    fn lerp(self, other: Self, t: f32) -> Self {
        Vector4::new(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
            self.w + (other.w - self.w) * t,
        )
    }
}

impl Animatable for Quaternion {
    /// Spherical interpolation along the shortest arc.
    ///
    /// Component-wise interpolation would take the long way round half the time
    /// and change the rotation speed along the way.
    fn lerp(self, other: Self, t: f32) -> Self {
        // Overshooting curves can push t outside 0..1; slerp handles the ends
        // exactly, so clamping here only affects the overshoot, which has no
        // meaning for a shortest-arc rotation anyway.
        self.slerp(other, t.clamp(0.0, 1.0))
    }
}

impl Animatable for Color {
    fn lerp(self, other: Self, t: f32) -> Self {
        Color {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_are_exact_for_every_type() {
        assert_eq!(1.0f32.lerp(5.0, 0.0), 1.0);
        assert_eq!(1.0f32.lerp(5.0, 1.0), 5.0);

        let (a, b) = (Vector3::new(1.0, 2.0, 3.0), Vector3::new(-4.0, 0.0, 8.0));
        assert_eq!(a.lerp(b, 0.0), a);
        assert_eq!(a.lerp(b, 1.0), b);

        let c = Color::from_hex(0x112233);
        let d = Color::from_hex(0xffee00);
        assert_eq!(c.lerp(d, 0.0), c);
        assert_eq!(c.lerp(d, 1.0), d);
    }

    #[test]
    fn the_midpoint_really_is_halfway() {
        assert_eq!(0.0f32.lerp(10.0, 0.5), 5.0);
        let m = Vector3::new(0.0, 0.0, 0.0).lerp(Vector3::new(2.0, 4.0, 6.0), 0.5);
        assert_eq!(m, Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn scalars_extrapolate_but_rotations_do_not() {
        // An overshooting curve is meaningful for a position and meaningless for
        // a shortest-arc rotation, which has nowhere past the target to go.
        assert_eq!(0.0f32.lerp(10.0, 1.5), 15.0);

        let a = Quaternion::identity();
        let b = Quaternion::from_axis_angle(Vector3::UP, 1.0);
        assert_eq!(a.lerp(b, 1.5), a.lerp(b, 1.0));
    }

    #[test]
    fn quaternion_interpolation_takes_the_short_way_round() {
        use std::f32::consts::PI;
        let a = Quaternion::from_axis_angle(Vector3::UP, -0.1 * PI);
        // The same rotation, written with a negated quaternion.
        let far = Quaternion::from_axis_angle(Vector3::UP, 0.1 * PI);
        let negated = Quaternion::new(-far.x, -far.y, -far.z, -far.w);

        let midpoint = a.lerp(negated, 0.5);
        // Halfway between -18 and +18 degrees is 0 degrees, whichever
        // representation was handed in.
        assert!(
            midpoint.dot(Quaternion::identity()).abs() > 0.99,
            "slerp went the long way: {midpoint:?}"
        );
    }

    #[test]
    fn interpolated_rotations_stay_unit_length() {
        let a = Quaternion::from_euler_xyz(0.3, -1.2, 2.0).normalize();
        let b = Quaternion::from_euler_xyz(-2.0, 0.7, 1.1).normalize();
        for i in 0..=20 {
            let q = a.lerp(b, i as f32 / 20.0);
            let n = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).sqrt();
            assert!((n - 1.0).abs() < 1e-3, "drifted to {n} at {i}");
        }
    }
}
