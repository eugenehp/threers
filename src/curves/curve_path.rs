use super::Curve2;
use crate::math::Vector2;

/// Sequence of 2D curves forming a single piecewise path. Matches three.js's
/// `CurvePath` for 2D paths. `t` is normalized across all sub-curves by
/// arc-length proportion.
pub struct CurvePath {
    pub curves: Vec<Box<dyn Curve2>>,
    pub auto_close: bool,
}

// `dyn Curve2` is not `Debug`, so this reports the shape of the path rather
// than its contents — enough to see in a test failure what you are holding.
impl std::fmt::Debug for CurvePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CurvePath")
            .field("curves", &self.curves.len())
            .field("auto_close", &self.auto_close)
            .finish()
    }
}

impl Default for CurvePath {
    fn default() -> Self {
        Self::new()
    }
}

impl CurvePath {
    pub fn new() -> Self {
        Self {
            curves: Vec::new(),
            auto_close: false,
        }
    }

    pub fn add(&mut self, c: Box<dyn Curve2>) -> &mut Self {
        self.curves.push(c);
        self
    }
}

impl Curve2 for CurvePath {
    fn get_point(&self, t: f32) -> Vector2 {
        if self.curves.is_empty() {
            return Vector2::ZERO;
        }
        let lengths: Vec<f32> = self.curves.iter().map(|c| c.get_length(32)).collect();
        let total: f32 = lengths.iter().sum();
        if total == 0.0 {
            return self.curves[0].get_point(0.0);
        }
        let target = t.clamp(0.0, 1.0) * total;
        let mut acc = 0.0;
        for (i, l) in lengths.iter().enumerate() {
            if acc + l >= target {
                let local = if *l > 0.0 { (target - acc) / l } else { 0.0 };
                return self.curves[i].get_point(local);
            }
            acc += l;
        }
        self.curves.last().unwrap().get_point(1.0)
    }
}
