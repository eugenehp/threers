//! Where the tip goes, and what to pull to put it somewhere else.
//!
//! Two halves that only make sense together:
//!
//! - **Forward**, [`piecewise_constant_curvature`]: the closed-form model of a
//!   tendon-driven rod, in which each segment is a circular arc. Fast, exact
//!   for an ideal rod, and wrong for a real one — it knows nothing about
//!   gravity, cable friction or a payload.
//! - **Inverse**, [`TipController`]: a damped-least-squares step *on the
//!   analytic Jacobian*, closing the loop on the tip the simulation actually
//!   produced.
//!
//! The division of labour is the point. The analytic model supplies the
//! direction to move in, which it is good at and which is cheap; the
//! simulation supplies where the tip really is, which the analytic model
//! cannot know. A controller built on either alone is either fast and wrong or
//! correct and a hundred times slower — differentiating a settled simulation
//! costs `2 · 2S` rollouts per step.

use crate::clark::Clark;
use threers::math::{Quaternion, Vector3};

/// A pose along the rod.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    pub position: Vector3,
    pub rotation: Quaternion,
}

impl Frame {
    pub const IDENTITY: Self = Self {
        position: Vector3::ZERO,
        rotation: Quaternion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        },
    };
}

/// Where a chain of constant-curvature segments ends up.
///
/// Each entry of `segments` is `(arc length, bend angle in radians, direction
/// of the bending plane in radians)`. The rod leaves each segment along that
/// segment's own tip tangent, so the segments compose the way the real rod
/// does.
///
/// The rod runs along local **+Y**, matching [`crate::Continuum`].
///
/// A segment of arc length `L` bent through `Θ` is an arc of radius `L/Θ`, so
/// its tip is `r(1 − cos Θ)` off the original axis and `r sin Θ` along it. The
/// `Θ → 0` case is a straight segment, and is taken from a series rather than
/// from `L/Θ`, which is how a rod that is merely *nearly* straight would
/// otherwise come out at infinity.
pub fn piecewise_constant_curvature(segments: &[(f32, f32, f32)]) -> Frame {
    let mut frame = Frame::IDENTITY;
    for &(length, angle, direction) in segments {
        let (local_pos, local_rot) = arc(length, angle, direction);
        frame.position = frame.position + frame.rotation_apply(local_pos);
        frame.rotation = frame.rotation.multiply(local_rot);
    }
    frame
}

/// Every segment tip, root first — what a multi-point controller steers.
pub fn constant_curvature_frames(segments: &[(f32, f32, f32)]) -> Vec<Frame> {
    let mut frame = Frame::IDENTITY;
    let mut out = Vec::with_capacity(segments.len());
    for &(length, angle, direction) in segments {
        let (local_pos, local_rot) = arc(length, angle, direction);
        frame.position = frame.position + frame.rotation_apply(local_pos);
        frame.rotation = frame.rotation.multiply(local_rot);
        out.push(frame);
    }
    out
}

impl Frame {
    fn rotation_apply(&self, v: Vector3) -> Vector3 {
        v.apply_quaternion(self.rotation)
    }
}

/// One arc, in the frame of the segment's own root.
fn arc(length: f32, angle: f32, direction: f32) -> (Vector3, Quaternion) {
    let (c, s) = (direction.cos(), direction.sin());
    if angle.abs() < 1.0e-5 {
        // Straight, to first order — but keep the sideways term, or a
        // controller differentiating through here sees a zero column and
        // decides a nearly straight segment cannot steer.
        let sideways = length * angle * 0.5;
        return (
            Vector3::new(c * sideways, length, -s * sideways),
            Quaternion::from_axis_angle(Vector3::new(-s, 0.0, -c), angle),
        );
    }
    let radius = length / angle;
    let out = radius * (1.0 - angle.cos());
    let along = radius * angle.sin();
    (
        Vector3::new(c * out, along, -s * out),
        // Bending in the plane through `direction` turns the tangent about the
        // axis across it.
        Quaternion::from_axis_angle(Vector3::new(-s, 0.0, -c), angle),
    )
}

/// Closed-loop task-space control of a rod's tip.
///
/// Holds one [`Clark`] per segment and steps their coordinates toward whatever
/// puts the tip on target, using damped least squares so a rod near a
/// singularity — straight, or stretched out at full reach — slows down instead
/// of thrashing.
///
/// ```no_run
/// use threers_continuum::prelude::*;
/// use threers_physics::prelude::*;
///
/// # let mut world = World::new();
/// # let rod = Rod::new(0.3, 20).radius(0.006);
/// # let mut arm = Continuum::build(&mut world, rod.clone(), None, Vector3::ZERO, Vector3::UP);
/// # arm.add_tendon_ring(&mut world, 3, 0.004, 0.0, 0, 0.0);
/// let mut control = TipController::new(vec![Clark::new(3, 0.004, 0.0)], vec![0.3]);
/// let target = Vector3::new(0.05, 0.28, 0.0);
///
/// for _ in 0..600 {
///     let pulls = control.step(arm.tip(&world), target);
///     arm.set_pulls(&mut world, &pulls, 50.0);
///     world.step(1.0 / 60.0);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct TipController {
    /// One per segment, root first.
    pub segments: Vec<Clark>,
    /// Arc length of each segment.
    pub lengths: Vec<f32>,
    /// Current command, two numbers per segment.
    pub clark: Vec<[f32; 2]>,
    /// Damping for the least-squares solve. Larger is steadier and slower.
    pub damping: f32,
    /// Fraction of the computed step actually taken, per call.
    pub gain: f32,
    /// Largest bend one segment may be commanded to, in radians.
    pub max_bend: f32,
    /// Largest change in one Clark coordinate per call — the slew limit that
    /// keeps a large error from being answered with a lunge.
    pub max_step: f32,
}

impl TipController {
    /// A controller for the given segments and their arc lengths.
    pub fn new(segments: Vec<Clark>, lengths: Vec<f32>) -> Self {
        let n = segments.len();
        Self {
            segments,
            lengths,
            clark: vec![[0.0; 2]; n],
            damping: 0.02,
            gain: 0.35,
            max_bend: 2.0,
            max_step: 0.0005,
        }
    }

    /// Tune the solve: least-squares damping and step gain.
    pub fn tuned(mut self, damping: f32, gain: f32) -> Self {
        self.damping = damping.max(0.0);
        self.gain = gain.clamp(0.0, 1.0);
        self
    }

    /// Set the per-call slew limit on each Clark coordinate.
    pub fn slew(mut self, max_step: f32) -> Self {
        self.max_step = max_step.max(0.0);
        self
    }

    /// Where the analytic model says the tip is, for the current command.
    pub fn predicted_tip(&self) -> Vector3 {
        piecewise_constant_curvature(&self.arcs(&self.clark)).position
    }

    /// One control step: given where the tip **is** and where it should be,
    /// advance the command and return the pull for every cable, segment by
    /// segment in order.
    ///
    /// `tip` is the measured position — from the simulation, or from a tracker
    /// on real hardware. Feeding the analytic prediction back in instead turns
    /// this into open-loop control, which will confidently converge to the
    /// wrong place.
    pub fn step(&mut self, tip: Vector3, target: Vector3) -> Vec<f32> {
        let error = target - tip;
        let jacobian = self.jacobian();
        let delta = damped_least_squares(&jacobian, error, self.damping);

        for (i, c) in self.clark.iter_mut().enumerate() {
            let step = [
                (delta[2 * i] * self.gain).clamp(-self.max_step, self.max_step),
                (delta[2 * i + 1] * self.gain).clamp(-self.max_step, self.max_step),
            ];
            *c = [c[0] + step[0], c[1] + step[1]];
        }
        for (c, segment) in self.clark.iter_mut().zip(&self.segments) {
            *c = segment.clamp(*c, self.max_bend);
        }
        self.pulls()
    }

    /// The pulls for the command as it stands, without advancing it.
    pub fn pulls(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for (segment, c) in self.segments.iter().zip(&self.clark) {
            out.extend(segment.to_pulls(*c));
        }
        out
    }

    /// Put the command back to straight.
    pub fn home(&mut self) {
        for c in &mut self.clark {
            *c = [0.0; 2];
        }
    }

    fn arcs(&self, clark: &[[f32; 2]]) -> Vec<(f32, f32, f32)> {
        self.segments
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let (angle, direction) = s.bend(clark[i]);
                (*self.lengths.get(i).unwrap_or(&0.0), angle, direction)
            })
            .collect()
    }

    /// `∂tip / ∂clark`, three rows by two columns per segment.
    ///
    /// Differenced rather than derived. The closed form exists and is a page of
    /// chain rule through two trigonometric substitutions; the difference is
    /// six evaluations of a function that costs a handful of sines, and it
    /// cannot be quietly wrong about a sign.
    fn jacobian(&self) -> Vec<[f32; 3]> {
        const H: f32 = 1.0e-5;
        let mut columns = Vec::with_capacity(self.clark.len() * 2);
        for i in 0..self.clark.len() {
            for axis in 0..2 {
                let mut plus = self.clark.clone();
                let mut minus = self.clark.clone();
                plus[i][axis] += H;
                minus[i][axis] -= H;
                let a = piecewise_constant_curvature(&self.arcs(&plus)).position;
                let b = piecewise_constant_curvature(&self.arcs(&minus)).position;
                let d = (a - b) * (1.0 / (2.0 * H));
                columns.push([d.x, d.y, d.z]);
            }
        }
        columns
    }
}

/// Solve `J δ = e` for the smallest `δ` that gets close, damped by `λ`.
///
/// `(JᵀJ + λ²I) δ = Jᵀe`, built and solved in the *column* space because a rod
/// has two coordinates per segment and three error components: for one segment
/// that is a 2×2 system, and even a five-segment robot is 10×10. Gauss-Jordan
/// on it is cheaper than reasoning about which side is over-determined.
fn damped_least_squares(columns: &[[f32; 3]], error: Vector3, damping: f32) -> Vec<f32> {
    let n = columns.len();
    if n == 0 {
        return Vec::new();
    }
    let mut a = vec![0.0f32; n * n];
    let mut b = vec![0.0f32; n];
    let lambda = damping * damping;
    for i in 0..n {
        let ci = columns[i];
        b[i] = ci[0] * error.x + ci[1] * error.y + ci[2] * error.z;
        for j in 0..n {
            let cj = columns[j];
            a[i * n + j] = ci[0] * cj[0] + ci[1] * cj[1] + ci[2] * cj[2];
        }
        a[i * n + i] += lambda;
    }
    solve_in_place(&mut a, &mut b, n);
    b
}

/// Gauss-Jordan with partial pivoting. Leaves the solution in `b`.
fn solve_in_place(a: &mut [f32], b: &mut [f32], n: usize) {
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if a[row * n + col].abs() > a[pivot * n + col].abs() {
                pivot = row;
            }
        }
        if a[pivot * n + col].abs() < 1.0e-12 {
            // Singular even after damping, which means this direction cannot be
            // steered at all. Ask for nothing rather than for infinity.
            b[col] = 0.0;
            continue;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(pivot * n + k, col * n + k);
            }
            b.swap(pivot, col);
        }
        let inv = 1.0 / a[col * n + col];
        for k in col..n {
            a[col * n + k] *= inv;
        }
        b[col] *= inv;
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row * n + col];
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_straight_segment_ends_at_its_own_length() {
        let f = piecewise_constant_curvature(&[(0.3, 0.0, 0.0)]);
        assert!((f.position - Vector3::new(0.0, 0.3, 0.0)).length() < 1e-6);
    }

    #[test]
    fn a_quarter_turn_puts_the_tip_at_the_radius() {
        // Bent through 90°, an arc of length L has radius L/(π/2) and its tip
        // sits exactly one radius sideways and one radius along.
        let l = 1.0f32;
        let f = piecewise_constant_curvature(&[(l, std::f32::consts::FRAC_PI_2, 0.0)]);
        let r = l / std::f32::consts::FRAC_PI_2;
        assert!((f.position.x - r).abs() < 1e-5, "{:?}", f.position);
        assert!((f.position.y - r).abs() < 1e-5, "{:?}", f.position);
        assert!(f.position.z.abs() < 1e-5);
    }

    #[test]
    fn a_half_turn_comes_back_on_itself() {
        let l = 1.0f32;
        let f = piecewise_constant_curvature(&[(l, std::f32::consts::PI, 0.0)]);
        let r = l / std::f32::consts::PI;
        assert!((f.position.x - 2.0 * r).abs() < 1e-5, "{:?}", f.position);
        assert!(f.position.y.abs() < 1e-5, "{:?}", f.position);
    }

    #[test]
    fn the_bending_direction_sweeps_the_tip_around() {
        let l = 0.5f32;
        let angle = 1.0f32;
        let a = piecewise_constant_curvature(&[(l, angle, 0.0)]).position;
        let b =
            piecewise_constant_curvature(&[(l, angle, std::f32::consts::FRAC_PI_2)]).position;
        assert!(a.x > 0.01 && a.z.abs() < 1e-5, "{a:?}");
        assert!(b.z < -0.01 && b.x.abs() < 1e-5, "{b:?}");
        assert!(
            (a.y - b.y).abs() < 1e-5,
            "turning the bending plane must not change the height"
        );
    }

    #[test]
    fn two_segments_compose_along_the_first_ones_tangent() {
        // Two 90° quarter-arcs in the same plane make a half turn, and end up
        // where one 180° arc of the same total length does.
        let together = piecewise_constant_curvature(&[
            (0.5, std::f32::consts::FRAC_PI_2, 0.0),
            (0.5, std::f32::consts::FRAC_PI_2, 0.0),
        ]);
        let single = piecewise_constant_curvature(&[(1.0, std::f32::consts::PI, 0.0)]);
        assert!(
            (together.position - single.position).length() < 1e-4,
            "{:?} vs {:?}",
            together.position,
            single.position
        );
    }

    #[test]
    fn the_controller_walks_a_reachable_target_down_to_nothing() {
        // Closed loop on the analytic model itself: not a physics test, a test
        // that the Jacobian and the solve agree about which way is downhill.
        let mut control =
            TipController::new(vec![crate::clark::Clark::new(3, 0.004, 0.0)], vec![0.3])
                .slew(1.0);
        let target = piecewise_constant_curvature(&[(0.3, 0.6, 0.9)]).position;

        let start = (control.predicted_tip() - target).length();
        for _ in 0..500 {
            let tip = control.predicted_tip();
            control.step(tip, target);
        }
        let end = (control.predicted_tip() - target).length();
        assert!(
            end < 1e-3,
            "converged to {end} from {start}, target {target:?}"
        );
    }

    #[test]
    fn an_unreachable_target_does_not_blow_the_command_up() {
        let mut control =
            TipController::new(vec![crate::clark::Clark::new(3, 0.004, 0.0)], vec![0.3]);
        for _ in 0..2000 {
            let tip = control.predicted_tip();
            control.step(tip, Vector3::new(50.0, 50.0, 0.0));
        }
        let (angle, _) = control.segments[0].bend(control.clark[0]);
        assert!(
            angle <= control.max_bend + 1e-4,
            "the command ran past its own bend limit: {angle}"
        );
        assert!(control.clark[0][0].is_finite() && control.clark[0][1].is_finite());
    }

    #[test]
    fn damped_least_squares_solves_the_easy_case() {
        // One column straight up: the step is the error along it over its own
        // length squared, damped.
        let columns = [[0.0f32, 2.0, 0.0]];
        let d = damped_least_squares(&columns, Vector3::new(0.0, 1.0, 0.0), 0.0);
        assert!((d[0] - 0.5).abs() < 1e-6, "got {d:?}");
    }
}
