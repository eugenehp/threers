//! Maya / Blender-style F-curves: keyed scalars with Bezier tangents,
//! weighted handles, plateau auto-tangents, and cycle extrapolation.

/// How a key's tangents behave.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TangentMode {
    /// Smooth bezier using `in_tangent` / `out_tangent` slopes (value/time).
    #[default]
    Bezier,
    /// Flat hold until the next key.
    Step,
    /// Straight line to the next key (tangents ignored).
    Linear,
    /// Flat plateau — zero slope (held value with smooth departure via weights).
    Plateau,
}

/// Extrapolation outside the keyed range (graph-editor pre/post infinity).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Extrapolation {
    /// Hold the end key value.
    #[default]
    Constant,
    /// Continue with the end segment's slope.
    Linear,
    /// Repeat the curve in time.
    Cycle,
    /// Repeat, accumulating the value delta each cycle.
    CycleOffset,
    /// Ping-pong the curve.
    Oscillate,
}

/// One key on a channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveKey {
    pub time: f32,
    pub value: f32,
    /// Incoming slope dy/dt.
    pub in_tangent: f32,
    /// Outgoing slope dy/dt.
    pub out_tangent: f32,
    /// Incoming handle weight `0..1` (1 = full hermite influence).
    pub in_weight: f32,
    /// Outgoing handle weight `0..1`.
    pub out_weight: f32,
    pub mode: TangentMode,
}

impl CurveKey {
    pub fn linear(time: f32, value: f32) -> Self {
        Self {
            time,
            value,
            in_tangent: 0.0,
            out_tangent: 0.0,
            in_weight: 1.0,
            out_weight: 1.0,
            mode: TangentMode::Linear,
        }
    }

    pub fn bezier(time: f32, value: f32, in_tangent: f32, out_tangent: f32) -> Self {
        Self {
            time,
            value,
            in_tangent,
            out_tangent,
            in_weight: 1.0,
            out_weight: 1.0,
            mode: TangentMode::Bezier,
        }
    }

    pub fn weighted(
        time: f32,
        value: f32,
        in_tangent: f32,
        out_tangent: f32,
        in_weight: f32,
        out_weight: f32,
    ) -> Self {
        Self {
            time,
            value,
            in_tangent,
            out_tangent,
            in_weight: in_weight.clamp(0.0, 1.0),
            out_weight: out_weight.clamp(0.0, 1.0),
            mode: TangentMode::Bezier,
        }
    }

    pub fn step(time: f32, value: f32) -> Self {
        Self {
            time,
            value,
            in_tangent: 0.0,
            out_tangent: 0.0,
            in_weight: 1.0,
            out_weight: 1.0,
            mode: TangentMode::Step,
        }
    }

    pub fn plateau(time: f32, value: f32) -> Self {
        Self {
            time,
            value,
            in_tangent: 0.0,
            out_tangent: 0.0,
            in_weight: 1.0,
            out_weight: 1.0,
            mode: TangentMode::Plateau,
        }
    }
}

/// A scalar F-curve.
#[derive(Debug, Clone, PartialEq)]
pub struct FCurve {
    keys: Vec<CurveKey>,
    /// Behaviour before the first key.
    pub pre: Extrapolation,
    /// Behaviour after the last key.
    pub post: Extrapolation,
}

impl Default for FCurve {
    fn default() -> Self {
        Self {
            keys: Vec::new(),
            pre: Extrapolation::Constant,
            post: Extrapolation::Constant,
        }
    }
}

impl FCurve {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_keys(mut keys: Vec<CurveKey>) -> Self {
        keys.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
        Self {
            keys,
            pre: Extrapolation::Constant,
            post: Extrapolation::Constant,
        }
    }

    pub fn with_extrapolation(mut self, pre: Extrapolation, post: Extrapolation) -> Self {
        self.pre = pre;
        self.post = post;
        self
    }

    pub fn insert(&mut self, key: CurveKey) {
        match self
            .keys
            .binary_search_by(|k| k.time.partial_cmp(&key.time).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(i) => self.keys[i] = key,
            Err(i) => self.keys.insert(i, key),
        }
    }

    pub fn keys(&self) -> &[CurveKey] {
        &self.keys
    }

    pub fn keys_mut(&mut self) -> &mut [CurveKey] {
        &mut self.keys
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Zero in/out tangents on every key (classic plateau / flat handles).
    pub fn auto_plateau_tangents(&mut self) {
        for k in &mut self.keys {
            k.in_tangent = 0.0;
            k.out_tangent = 0.0;
            k.mode = TangentMode::Plateau;
        }
    }

    /// Smooth (Catmull-like) auto tangents from neighbours.
    pub fn auto_smooth_tangents(&mut self) {
        let n = self.keys.len();
        if n == 0 {
            return;
        }
        for i in 0..n {
            let prev = if i == 0 {
                self.keys[i]
            } else {
                self.keys[i - 1]
            };
            let next = if i + 1 >= n {
                self.keys[i]
            } else {
                self.keys[i + 1]
            };
            let dt = (next.time - prev.time).max(1e-8);
            let slope = (next.value - prev.value) / dt;
            self.keys[i].in_tangent = slope;
            self.keys[i].out_tangent = slope;
            self.keys[i].mode = TangentMode::Bezier;
        }
    }

    /// Evaluate at time `t`, honouring pre/post extrapolation.
    pub fn evaluate(&self, t: f32) -> f32 {
        if self.keys.is_empty() {
            return 0.0;
        }
        if self.keys.len() == 1 {
            return self.keys[0].value;
        }
        let first = self.keys[0];
        let last = self.keys[self.keys.len() - 1];

        if t < first.time {
            return self.extrapolate(t, true);
        }
        if t > last.time {
            return self.extrapolate(t, false);
        }
        self.evaluate_clamped(t)
    }

    fn extrapolate(&self, t: f32, before: bool) -> f32 {
        let first = self.keys[0];
        let last = self.keys[self.keys.len() - 1];
        let span = (last.time - first.time).max(1e-8);
        let mode = if before { self.pre } else { self.post };

        match mode {
            Extrapolation::Constant => {
                if before {
                    first.value
                } else {
                    last.value
                }
            }
            Extrapolation::Linear => {
                if before {
                    let slope = if self.keys.len() >= 2 {
                        let b = self.keys[1];
                        (b.value - first.value) / (b.time - first.time).max(1e-8)
                    } else {
                        first.out_tangent
                    };
                    first.value + slope * (t - first.time)
                } else {
                    let slope = if self.keys.len() >= 2 {
                        let a = self.keys[self.keys.len() - 2];
                        (last.value - a.value) / (last.time - a.time).max(1e-8)
                    } else {
                        last.in_tangent
                    };
                    last.value + slope * (t - last.time)
                }
            }
            Extrapolation::Cycle => {
                let wrapped = wrap_cycle(t, first.time, span);
                self.evaluate_clamped(wrapped)
            }
            Extrapolation::CycleOffset => {
                let (wrapped, cycles) = wrap_cycle_count(t, first.time, span);
                let delta = last.value - first.value;
                self.evaluate_clamped(wrapped) + cycles * delta
            }
            Extrapolation::Oscillate => {
                let wrapped = wrap_oscillate(t, first.time, span);
                self.evaluate_clamped(wrapped)
            }
        }
    }

    fn evaluate_clamped(&self, t: f32) -> f32 {
        let first = self.keys[0].time;
        let last = self.keys[self.keys.len() - 1].time;
        let t = t.clamp(first, last);
        // Inline segment eval without re-entering extrapolation.
        let mut i = 0;
        while i + 1 < self.keys.len() && self.keys[i + 1].time < t {
            i += 1;
        }
        if i + 1 >= self.keys.len() {
            return self.keys[i].value;
        }
        hermite_eval(self.keys[i], self.keys[i + 1], t)
    }
}

fn hermite_eval(a: CurveKey, b: CurveKey, t: f32) -> f32 {
    let dt = (b.time - a.time).max(1e-8);
    let u = ((t - a.time) / dt).clamp(0.0, 1.0);
    match a.mode {
        TangentMode::Step => {
            // Hold `a` until the next key; at/after `b.time` use `b`.
            if u >= 1.0 - 1e-7 {
                b.value
            } else {
                a.value
            }
        }
        TangentMode::Linear => a.value + (b.value - a.value) * u,
        TangentMode::Plateau | TangentMode::Bezier => {
            let mut m0 = a.out_tangent * dt * a.out_weight.clamp(0.0, 1.0);
            let mut m1 = b.in_tangent * dt * b.in_weight.clamp(0.0, 1.0);
            if matches!(a.mode, TangentMode::Plateau) {
                m0 = 0.0;
            }
            if matches!(b.mode, TangentMode::Plateau) {
                m1 = 0.0;
            }
            hermite(a.value, b.value, m0, m1, u)
        }
    }
}

fn hermite(p0: f32, p1: f32, m0: f32, m1: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * p0
        + (t3 - 2.0 * t2 + t) * m0
        + (-2.0 * t3 + 3.0 * t2) * p1
        + (t3 - t2) * m1
}

fn wrap_cycle(t: f32, start: f32, span: f32) -> f32 {
    let mut u = (t - start) % span;
    if u < 0.0 {
        u += span;
    }
    start + u
}

fn wrap_cycle_count(t: f32, start: f32, span: f32) -> (f32, f32) {
    let rel = t - start;
    let cycles = (rel / span).floor();
    let mut u = rel - cycles * span;
    if u < 0.0 {
        u += span;
    }
    (start + u, cycles)
}

fn wrap_oscillate(t: f32, start: f32, span: f32) -> f32 {
    let rel = t - start;
    let period = span * 2.0;
    let mut u = rel % period;
    if u < 0.0 {
        u += period;
    }
    if u <= span {
        start + u
    } else {
        start + (period - u)
    }
}

/// Bundled F-curves for every camera channel.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CameraCurves {
    pub target_x: FCurve,
    pub target_y: FCurve,
    pub target_z: FCurve,
    pub azimuth: FCurve,
    pub elevation: FCurve,
    pub radius: FCurve,
    pub roll: FCurve,
    pub fov: FCurve,
    pub focus_distance: FCurve,
    pub aperture: FCurve,
}

impl CameraCurves {
    pub fn evaluate(&self, t: f32, template: super::pose::CameraPose) -> super::pose::CameraPose {
        let mut p = template;
        if !self.target_x.is_empty() {
            p.target.x = self.target_x.evaluate(t);
        }
        if !self.target_y.is_empty() {
            p.target.y = self.target_y.evaluate(t);
        }
        if !self.target_z.is_empty() {
            p.target.z = self.target_z.evaluate(t);
        }
        if !self.azimuth.is_empty() {
            p.azimuth = self.azimuth.evaluate(t);
        }
        if !self.elevation.is_empty() {
            p.elevation = self.elevation.evaluate(t);
        }
        if !self.radius.is_empty() {
            p.radius = self.radius.evaluate(t).max(1e-4);
        }
        if !self.roll.is_empty() {
            p.roll = self.roll.evaluate(t);
        }
        if !self.fov.is_empty() {
            p.fov = self.fov.evaluate(t).max(1e-3);
        }
        if !self.focus_distance.is_empty() {
            p.focus_distance = self.focus_distance.evaluate(t).max(0.0);
        }
        if !self.aperture.is_empty() {
            p.aperture = self.aperture.evaluate(t).max(0.0);
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_keys_interpolate() {
        let c = FCurve::from_keys(vec![
            CurveKey::linear(0.0, 0.0),
            CurveKey::linear(1.0, 10.0),
        ]);
        assert!((c.evaluate(0.5) - 5.0).abs() < 1e-4);
    }

    #[test]
    fn step_holds() {
        let c = FCurve::from_keys(vec![
            CurveKey::step(0.0, 1.0),
            CurveKey::step(1.0, 5.0),
        ]);
        assert_eq!(c.evaluate(0.9), 1.0);
        assert_eq!(c.evaluate(1.0), 5.0);
    }

    #[test]
    fn bezier_hits_endpoints() {
        let c = FCurve::from_keys(vec![
            CurveKey::bezier(0.0, 0.0, 0.0, 2.0),
            CurveKey::bezier(1.0, 1.0, 2.0, 0.0),
        ]);
        assert!((c.evaluate(0.0)).abs() < 1e-5);
        assert!((c.evaluate(1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cycle_repeats() {
        let c = FCurve::from_keys(vec![
            CurveKey::linear(0.0, 0.0),
            CurveKey::linear(1.0, 2.0),
        ])
        .with_extrapolation(Extrapolation::Cycle, Extrapolation::Cycle);
        assert!((c.evaluate(1.5) - 1.0).abs() < 1e-3);
        assert!((c.evaluate(2.0) - 0.0).abs() < 1e-3 || (c.evaluate(2.0) - 2.0).abs() < 1e-3);
    }

    #[test]
    fn cycle_offset_accumulates() {
        let c = FCurve::from_keys(vec![
            CurveKey::linear(0.0, 0.0),
            CurveKey::linear(1.0, 3.0),
        ])
        .with_extrapolation(Extrapolation::Constant, Extrapolation::CycleOffset);
        let v = c.evaluate(2.5);
        // two full cycles (+6) + half of next (+1.5) = 7.5
        assert!((v - 7.5).abs() < 0.1, "{v}");
    }

    #[test]
    fn plateau_has_zero_departure_slope() {
        let mut c = FCurve::from_keys(vec![
            CurveKey::plateau(0.0, 0.0),
            CurveKey::plateau(1.0, 10.0),
        ]);
        c.auto_plateau_tangents();
        // Near the start, plateau should advance slower than linear.
        assert!(c.evaluate(0.1) < 1.0);
    }
}
