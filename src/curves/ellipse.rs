use super::Curve2;
use crate::math::Vector2;
use std::f32::consts::PI;

/// Arc / ellipse / circle (with `x_radius == y_radius`). Matches three.js.
#[derive(Debug, Clone, Copy)]
pub struct EllipseCurve {
    pub center: Vector2,
    pub x_radius: f32,
    pub y_radius: f32,
    pub a_start: f32,
    pub a_end: f32,
    pub clockwise: bool,
    pub rotation: f32,
}

impl EllipseCurve {
    pub const fn new(
        center: Vector2,
        x_radius: f32,
        y_radius: f32,
        a_start: f32,
        a_end: f32,
        clockwise: bool,
        rotation: f32,
    ) -> Self {
        Self {
            center,
            x_radius,
            y_radius,
            a_start,
            a_end,
            clockwise,
            rotation,
        }
    }
}

impl Curve2 for EllipseCurve {
    fn get_point(&self, t: f32) -> Vector2 {
        let two_pi = PI * 2.0;
        let mut delta_angle = self.a_end - self.a_start;
        let same_points = delta_angle.abs() < f32::EPSILON;
        while delta_angle < 0.0 {
            delta_angle += two_pi;
        }
        while delta_angle > two_pi {
            delta_angle -= two_pi;
        }
        if delta_angle < f32::EPSILON {
            delta_angle = if same_points { 0.0 } else { two_pi };
        }
        if self.clockwise && !same_points {
            delta_angle = if (delta_angle - two_pi).abs() < f32::EPSILON {
                -two_pi
            } else {
                delta_angle - two_pi
            };
        }
        let angle = self.a_start + t * delta_angle;
        let x = self.x_radius * angle.cos();
        let y = self.y_radius * angle.sin();
        // `sin_cos` returns (sin, cos) — in that order. Binding it the other
        // way round rotates every ellipse a quarter turn, which is invisible on
        // a full circle and wrong on every arc.
        let (sr, cr) = self.rotation.sin_cos();
        Vector2::new(
            self.center.x + cr * x - sr * y,
            self.center.y + sr * x + cr * y,
        )
    }

    fn svg_segments(&self) -> Option<Vec<super::PathSegment>> {
        Some(super::svg_path::ellipse_segments(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vector2, b: Vector2) -> bool {
        (a - b).length() < 1e-4
    }

    #[test]
    fn unrotated_arc_starts_on_the_positive_x_axis() {
        let c = EllipseCurve::new(Vector2::ZERO, 5.0, 5.0, 0.0, PI, false, 0.0);
        assert!(
            close(c.get_point(0.0), Vector2::new(5.0, 0.0)),
            "start {:?}",
            c.get_point(0.0)
        );
        assert!(
            close(c.get_point(0.5), Vector2::new(0.0, 5.0)),
            "quarter {:?}",
            c.get_point(0.5)
        );
        assert!(
            close(c.get_point(1.0), Vector2::new(-5.0, 0.0)),
            "end {:?}",
            c.get_point(1.0)
        );
    }

    /// A quarter-turn `rotation` has to move the start a quarter turn, not
    /// leave it where an unrotated curve would be.
    #[test]
    fn rotation_turns_the_ellipse() {
        let c = EllipseCurve::new(Vector2::ZERO, 5.0, 5.0, 0.0, PI, false, PI / 2.0);
        assert!(
            close(c.get_point(0.0), Vector2::new(0.0, 5.0)),
            "start {:?}",
            c.get_point(0.0)
        );
    }

    /// An ellipse is not a circle: the radii must stay on their own axes.
    #[test]
    fn radii_apply_to_their_own_axis() {
        let c = EllipseCurve::new(Vector2::new(1.0, 2.0), 4.0, 1.0, 0.0, PI * 2.0, false, 0.0);
        assert!(close(c.get_point(0.0), Vector2::new(5.0, 2.0)));
        assert!(close(c.get_point(0.25), Vector2::new(1.0, 3.0)));
    }
}
