use crate::math::{Vector2, Vector3};

/// 2D parametric curve with `t` ∈ [0, 1].
pub trait Curve2: Send + Sync {
    fn get_point(&self, t: f32) -> Vector2;

    fn get_points(&self, divisions: usize) -> Vec<Vector2> {
        let n = divisions.max(1);
        (0..=n)
            .map(|i| self.get_point(i as f32 / n as f32))
            .collect()
    }

    fn get_tangent(&self, t: f32) -> Vector2 {
        let eps = 1e-4;
        let t0 = (t - eps).max(0.0);
        let t1 = (t + eps).min(1.0);
        (self.get_point(t1) - self.get_point(t0)).normalize()
    }

    fn get_length(&self, divisions: usize) -> f32 {
        let pts = self.get_points(divisions);
        pts.windows(2).map(|w| (w[1] - w[0]).length()).sum()
    }

    /// This curve written exactly as SVG path commands, if SVG can express it.
    ///
    /// `None` — the default — means it cannot, and the caller flattens the
    /// curve to line segments instead. So a `Curve2` implemented outside the
    /// crate still exports, just as a polyline, and only types that opt in pay
    /// for the conversion.
    ///
    /// The starting point is not part of the answer: the caller already knows
    /// where its pen is, and [`get_point(0.0)`](Self::get_point) is the same
    /// number for every implementor.
    fn svg_segments(&self) -> Option<Vec<super::PathSegment>> {
        None
    }
}

/// 3D parametric curve with `t` ∈ [0, 1].
pub trait Curve3: Send + Sync {
    fn get_point(&self, t: f32) -> Vector3;

    fn get_points(&self, divisions: usize) -> Vec<Vector3> {
        let n = divisions.max(1);
        (0..=n)
            .map(|i| self.get_point(i as f32 / n as f32))
            .collect()
    }

    fn get_tangent(&self, t: f32) -> Vector3 {
        let eps = 1e-4;
        let t0 = (t - eps).max(0.0);
        let t1 = (t + eps).min(1.0);
        (self.get_point(t1) - self.get_point(t0)).normalize()
    }

    fn get_length(&self, divisions: usize) -> f32 {
        let pts = self.get_points(divisions);
        pts.windows(2).map(|w| (w[1] - w[0]).length()).sum()
    }
}
