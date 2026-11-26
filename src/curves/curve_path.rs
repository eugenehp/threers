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
        self.point_along(t, &self.lengths())
    }

    // The curves measured once for all the points, not for each of them.
    fn get_points(&self, divisions: usize) -> Vec<Vector2> {
        let (lengths, divisions) = (self.lengths(), divisions.max(1));
        let total: f32 = lengths.iter().sum();
        if self.curves.is_empty() || total == 0.0 {
            return vec![self.point_along(0.0, &lengths); divisions + 1];
        }
        // As `point_along` finds each, going on from where the last was.
        let (mut on, mut acc) = (0, 0.0);
        (0..=divisions)
            .map(|i| {
                let target = i as f32 / divisions as f32 * total;
                while on < lengths.len() && acc + lengths[on] < target {
                    acc += lengths[on];
                    on += 1;
                }
                match lengths.get(on) {
                    Some(&l) => self.curves[on].get_point(if l > 0.0 { (target - acc) / l } else { 0.0 }),
                    None => self.curves[self.curves.len() - 1].get_point(1.0),
                }
            })
            .collect()
    }
}

impl CurvePath {
    fn lengths(&self) -> Vec<f32> {
        self.curves.iter().map(|c| c.get_length(32)).collect()
    }

    fn point_along(&self, t: f32, lengths: &[f32]) -> Vector2 {
        if self.curves.is_empty() {
            return Vector2::ZERO;
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curves::LineCurve;

    /// All the points at once are each as found alone.
    #[test]
    fn many_points_are_as_each_alone() {
        let mut p = CurvePath::new();
        let at = |i: i32| Vector2::new(i as f32, (i * 37 % 11) as f32);
        for i in 0..40 {
            p.add(Box::new(LineCurve::new(at(i), at(i + 1))));
            if i % 9 == 0 {
                p.add(Box::new(LineCurve::new(at(i + 1), at(i + 1))));
            }
        }
        for n in [1, 7, 500] {
            for (i, got) in p.get_points(n).into_iter().enumerate() {
                let alone = p.get_point(i as f32 / n as f32);
                assert!((got - alone).length() < 1e-4, "{n} {i}: {got:?} {alone:?}");
            }
        }
        let mut dot = CurvePath::new();
        assert_eq!(dot.get_points(3), vec![Vector2::ZERO; 4]);
        dot.add(Box::new(LineCurve::new(at(3), at(3))));
        assert_eq!(dot.get_points(3), vec![at(3); 4]);
    }
}
