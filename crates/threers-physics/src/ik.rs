//! Inverse kinematics: pose a chain so its tip reaches a target.
//!
//! Two solvers, both iterative:
//!
//! - [`IkChain::solve`] runs **FABRIK** (Forward And Backward Reaching IK). It
//!   works on joint *positions* rather than angles, converges in a handful of
//!   passes, and produces natural-looking poses. This is the one to use.
//! - [`IkChain::solve_ccd`] runs **CCD** (Cyclic Coordinate Descent), which
//!   rotates one joint at a time from the tip down. It is slower to converge and
//!   tends to overwork the joint nearest the tip, but it respects a hinge axis
//!   exactly, so it is the better fit for machinery.
//!
//! Neither solver is part of the rigid-body simulation — IK is a posing tool,
//! and runs independently of [`crate::world::World`].
//!
//! ```
//! use threers_physics::prelude::*;
//!
//! // A three-bone arm lying along +X.
//! let mut arm = IkChain::from_points(&[
//!     Vector3::new(0.0, 0.0, 0.0),
//!     Vector3::new(1.0, 0.0, 0.0),
//!     Vector3::new(2.0, 0.0, 0.0),
//!     Vector3::new(3.0, 0.0, 0.0),
//! ]);
//!
//! let result = arm.solve(Vector3::new(0.0, 3.0, 0.0));
//! assert!(result.reached, "error {}", result.error);
//! // The root never moves, and the tip is on the target.
//! assert!(arm.root().length() < 1e-3);
//! assert!((arm.tip() - Vector3::new(0.0, 3.0, 0.0)).length() < 1e-2);
//! ```

use crate::math::{orthonormal_basis, try_normalize};
use threers::core::{ObjectArena, ObjectId};
use threers::math::{Quaternion, Vector3};

/// Limits on how a joint may bend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IkConstraint {
    /// The segment may deviate at most `half_angle` radians from the previous
    /// segment's direction. A shoulder or a hip.
    Cone { half_angle: f32 },
    /// The segment bends only in the plane across `axis`, optionally within
    /// `[min, max]` radians of the previous segment. An elbow or a knee.
    Hinge {
        /// Rotation axis, in world space.
        axis: Vector3,
        min: f32,
        max: f32,
    },
}

impl IkConstraint {
    /// A hinge with no angle limits.
    pub fn hinge(axis: Vector3) -> Self {
        Self::Hinge {
            axis: try_normalize(axis).unwrap_or(Vector3::UP),
            min: -std::f32::consts::PI,
            max: std::f32::consts::PI,
        }
    }

    /// A hinge that bends one way only, like a knee.
    pub fn hinge_limited(axis: Vector3, min: f32, max: f32) -> Self {
        Self::Hinge {
            axis: try_normalize(axis).unwrap_or(Vector3::UP),
            min: min.min(max),
            max: max.max(min),
        }
    }

    pub fn cone(half_angle: f32) -> Self {
        Self::Cone {
            half_angle: half_angle.clamp(0.0, std::f32::consts::PI),
        }
    }

    /// Clamp `direction` given the direction of the previous segment.
    fn apply(&self, direction: Vector3, previous: Vector3) -> Vector3 {
        let Some(direction) = try_normalize(direction) else {
            return previous;
        };
        match *self {
            Self::Cone { half_angle } => {
                let cos_limit = half_angle.cos();
                let cos_angle = direction.dot(previous).clamp(-1.0, 1.0);
                if cos_angle >= cos_limit {
                    return direction;
                }
                // Rotate back toward `previous` until exactly on the cone.
                let axis = match try_normalize(previous.cross(direction)) {
                    Some(a) => a,
                    // Exactly antiparallel: any perpendicular axis will do.
                    None => crate::math::orthonormal_basis(previous).0,
                };
                rotate_about(previous, axis, half_angle)
            }
            Self::Hinge { axis, min, max } => {
                // Flatten into the hinge plane.
                let flat = direction - axis * direction.dot(axis);
                let reference = previous - axis * previous.dot(axis);
                let (Some(flat), Some(reference)) = (try_normalize(flat), try_normalize(reference))
                else {
                    return previous;
                };
                // Signed angle from the previous segment, about the hinge axis.
                let angle = flat
                    .cross(reference)
                    .dot(axis)
                    .atan2(flat.dot(reference).clamp(-1.0, 1.0));
                let clamped = (-angle).clamp(min, max);
                rotate_about(reference, axis, clamped)
            }
        }
    }
}

fn rotate_about(v: Vector3, axis: Vector3, angle: f32) -> Vector3 {
    v.apply_quaternion(Quaternion::from_axis_angle(axis, angle))
}

/// One joint of a chain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IkJoint {
    /// World-space position. This is what the solver moves.
    pub position: Vector3,
    /// Distance to the next joint. Zero for the tip.
    pub length: f32,
    pub constraint: Option<IkConstraint>,
    /// Scene node this joint drives, for [`IkChain::apply_to_scene`].
    pub scene_object: Option<ObjectId>,
}

/// Outcome of a solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IkResult {
    /// Iterations actually run.
    pub iterations: usize,
    /// Final distance from tip to target.
    pub error: f32,
    /// Whether the tip landed within [`IkChain::tolerance`].
    pub reached: bool,
    /// The target is further away than the chain is long, so the chain is
    /// stretched straight toward it and `reached` cannot be true.
    pub out_of_reach: bool,
}

/// A chain of joints to be posed.
#[derive(Debug, Clone, PartialEq)]
pub struct IkChain {
    pub joints: Vec<IkJoint>,
    /// Iteration cap. FABRIK usually converges in under ten.
    pub max_iterations: usize,
    /// Tip-to-target distance considered "arrived".
    pub tolerance: f32,
    /// Direction a bone points in its own local frame. three.js bones run along
    /// `+Y`, which is the default.
    pub bone_axis: Vector3,
    /// Reference direction arriving at the root, if it has one.
    ///
    /// A joint's constraint bends the segment leaving it against the segment
    /// arriving, and nothing arrives at the root — so a constraint on joint 0 is
    /// ignored unless something says which way the chain is anchored. Set this
    /// to the direction the root is mounted along and the root gets a limit like
    /// every other joint: a shoulder in a socket, a leg on a hip.
    pub base_direction: Option<Vector3>,
}

impl IkChain {
    /// Build from world-space joint positions, root first. Segment lengths are
    /// taken from the initial spacing and held fixed thereafter.
    pub fn from_points(points: &[Vector3]) -> Self {
        let joints = points
            .iter()
            .enumerate()
            .map(|(i, &position)| IkJoint {
                position,
                length: points.get(i + 1).map_or(0.0, |next| (*next - position).length()),
                constraint: None,
                scene_object: None,
            })
            .collect();
        Self {
            joints,
            max_iterations: 16,
            tolerance: 1e-3,
            bone_axis: Vector3::UP,
            base_direction: None,
        }
    }

    /// Build from a list of scene nodes, root first, using their current world
    /// positions.
    ///
    /// Call [`ObjectArena::update_world_matrices`] first if the hierarchy has
    /// moved — this reads `matrix_world`.
    pub fn from_scene(arena: &ObjectArena, bones: &[ObjectId]) -> Option<Self> {
        if bones.len() < 2 {
            return None;
        }
        let points: Vec<Vector3> = bones
            .iter()
            .map(|&id| arena.get(id).map(|o| o.world_position()))
            .collect::<Option<Vec<_>>>()?;
        let mut chain = Self::from_points(&points);
        for (joint, &id) in chain.joints.iter_mut().zip(bones) {
            joint.scene_object = Some(id);
        }
        Some(chain)
    }

    /// Walk up the parent links from `tip` to `root` and build the chain
    /// between them, root first.
    pub fn from_scene_chain(arena: &ObjectArena, root: ObjectId, tip: ObjectId) -> Option<Self> {
        let mut bones = vec![tip];
        let mut current = tip;
        // Guard against a cycle or a tip that is not a descendant of the root.
        for _ in 0..1024 {
            if current == root {
                bones.reverse();
                return Self::from_scene(arena, &bones);
            }
            current = arena.get(current)?.parent?;
            bones.push(current);
        }
        None
    }

    pub fn set_constraint(&mut self, index: usize, constraint: IkConstraint) -> &mut Self {
        if let Some(joint) = self.joints.get_mut(index) {
            joint.constraint = Some(constraint);
        }
        self
    }

    pub fn root(&self) -> Vector3 {
        self.joints.first().map_or(Vector3::ZERO, |j| j.position)
    }

    pub fn tip(&self) -> Vector3 {
        self.joints.last().map_or(Vector3::ZERO, |j| j.position)
    }

    /// Sum of the segment lengths — the farthest the tip can possibly reach.
    pub fn total_length(&self) -> f32 {
        self.joints.iter().map(|j| j.length).sum()
    }

    /// Pose the chain so the tip reaches `target`, using FABRIK.
    ///
    /// The root stays put; everything else moves. Segment lengths are preserved
    /// exactly, so the chain never stretches.
    pub fn solve(&mut self, target: Vector3) -> IkResult {
        let n = self.joints.len();
        if n < 2 {
            return IkResult {
                iterations: 0,
                error: (self.tip() - target).length(),
                reached: false,
                out_of_reach: false,
            };
        }

        let root = self.joints[0].position;
        let reach = self.total_length();
        let distance = (target - root).length();

        // At or beyond full extension the pose is exact and unique — the chain
        // straight at the target — and no amount of iterating improves on it.
        //
        // The `>=` matters. FABRIK approaches full extension asymptotically, so
        // a target sitting *exactly* on the reach limit is the one case the
        // iteration handles worst: it is perfectly reachable, and the solver
        // would report it a centimetre short however long it ran.
        if distance >= reach {
            let direction = try_normalize(target - root).unwrap_or(Vector3::UP);
            for i in 1..n {
                self.joints[i].position =
                    self.joints[i - 1].position + direction * self.joints[i - 1].length;
            }
            let error = (self.tip() - target).length();
            return IkResult {
                iterations: 0,
                error,
                reached: error <= self.tolerance,
                out_of_reach: distance > reach,
            };
        }

        let mut iterations = 0;
        for _ in 0..self.max_iterations {
            iterations += 1;
            if (self.tip() - target).length() <= self.tolerance {
                break;
            }

            // Backward pass: pin the tip to the target and walk toward the root.
            self.joints[n - 1].position = target;
            for i in (0..n - 1).rev() {
                let direction = try_normalize(self.joints[i].position - self.joints[i + 1].position)
                    .unwrap_or(Vector3::UP);
                self.joints[i].position =
                    self.joints[i + 1].position + direction * self.joints[i].length;
            }

            // Forward pass: put the root back and walk out to the tip, applying
            // constraints as we go — this is the pass where the previous
            // segment's direction is already final.
            self.joints[0].position = root;
            for i in 0..n - 1 {
                let mut direction =
                    try_normalize(self.joints[i + 1].position - self.joints[i].position)
                        .unwrap_or(Vector3::UP);
                if let (Some(constraint), Some(previous)) = (self.joints[i].constraint, self.previous_direction(i)) {
                    direction = constraint.apply(direction, previous);
                }
                self.joints[i + 1].position =
                    self.joints[i].position + direction * self.joints[i].length;
            }
        }

        let error = (self.tip() - target).length();
        IkResult {
            iterations,
            error,
            reached: error <= self.tolerance,
            out_of_reach: false,
        }
    }

    /// Pose the chain with cyclic coordinate descent.
    ///
    /// Rotates each joint in turn, from the one before the tip back to the root,
    /// to swing the tip onto the target.
    pub fn solve_ccd(&mut self, target: Vector3) -> IkResult {
        /// How far to tip a chain out of an exactly-collinear pose. Small
        /// enough not to show in the result, large enough that the next pass
        /// has a real cross product to work with.
        /// How far to tip a chain out of an exactly-collinear pose. Small
        /// enough not to show in the result, large enough that the next pass
        /// has a real cross product to work with.
        const SYMMETRY_BREAK: f32 = 0.05;
        /// Below this much progress in a whole pass, the sweep has stopped
        /// getting anywhere and is not going to on its own.
        const STALLED: f32 = 1e-6;

        let n = self.joints.len();
        if n < 2 {
            return IkResult {
                iterations: 0,
                error: (self.tip() - target).length(),
                reached: false,
                out_of_reach: false,
            };
        }
        let reach = self.total_length();
        let out_of_reach = (target - self.joints[0].position).length() > reach;

        // CCD is greedy, so it has local minima, and a chain that starts
        // collinear with its target sits in a nasty one: every joint is already
        // pointing the right way, so every joint declines to move, and the pose
        // it declines to leave can be well short of the target. Escaping needs a
        // change no single greedy step would make, so the sweep is allowed to
        // stall and then get kicked. The best pose seen is kept and restored at
        // the end, which is what makes that safe: a kick can explore a worse
        // configuration without the caller ever being handed one.
        let mut best: Vec<Vector3> = self.joints.iter().map(|j| j.position).collect();
        let mut best_error = (self.tip() - target).length();
        let mut previous_error = best_error;
        let mut kicks = 0usize;

        let mut iterations = 0;
        for _ in 0..self.max_iterations {
            iterations += 1;
            if (self.tip() - target).length() <= self.tolerance {
                break;
            }
            for i in (0..n - 1).rev() {
                let pivot = self.joints[i].position;
                let (Some(to_tip), Some(to_target)) = (
                    try_normalize(self.tip() - pivot),
                    try_normalize(target - pivot),
                ) else {
                    continue;
                };
                // The swing that would put this joint's tip on the target.
                let (axis, angle) = match try_normalize(to_tip.cross(to_target)) {
                    Some(axis) => (axis, to_tip.dot(to_target).clamp(-1.0, 1.0).acos()),
                    // Collinear, so the cross product carries no axis. That is
                    // not the same as having nothing to do.
                    None => {
                        let perpendicular = orthonormal_basis(to_tip).0;
                        if to_tip.dot(to_target) < 0.0 {
                            // Exactly opposed: swing through half a turn, and
                            // any perpendicular axis will do it.
                            (perpendicular, std::f32::consts::PI)
                        } else if ((target - pivot).length()
                            - (self.tip() - pivot).length())
                        .abs()
                            > self.tolerance
                        {
                            // Aligned, but the wrong distance away — the target
                            // is on the ray the arm already points down, nearer
                            // than the tip can reach while straight. No rotation
                            // of any single joint improves the *direction*, so
                            // every joint sees "already aligned" and skips, and
                            // a straight arm stays straight forever. Reaching it
                            // means folding, which needs the symmetry broken
                            // first; one nudge is enough, and the ordinary
                            // passes converge from there.
                            (perpendicular, SYMMETRY_BREAK)
                        } else {
                            continue;
                        }
                    }
                };
                if angle.abs() < 1e-6 {
                    continue;
                }
                let rotation = Quaternion::from_axis_angle(axis, angle);
                for j in i + 1..n {
                    self.joints[j].position =
                        pivot + (self.joints[j].position - pivot).apply_quaternion(rotation);
                }
                // Re-apply this joint's limit after the swing.
                if let (Some(constraint), Some(previous)) =
                    (self.joints[i].constraint, self.previous_direction(i))
                {
                    let direction = try_normalize(self.joints[i + 1].position - pivot)
                        .unwrap_or(previous);
                    let clamped = constraint.apply(direction, previous);
                    self.rebuild_from(i, clamped);
                }
            }

            let error = (self.tip() - target).length();
            if error < best_error {
                best_error = error;
                for (slot, joint) in best.iter_mut().zip(self.joints.iter()) {
                    *slot = joint.position;
                }
            }
            if previous_error - error < STALLED && error > self.tolerance {
                kicks += 1;
                self.kick(kicks);
            }
            previous_error = error;
        }

        // Hand back the best pose found, not the last one walked through.
        if (self.tip() - target).length() > best_error {
            for (joint, position) in self.joints.iter_mut().zip(best.iter()) {
                joint.position = *position;
            }
        }

        let error = (self.tip() - target).length();
        IkResult {
            iterations,
            error,
            reached: error <= self.tolerance,
            out_of_reach,
        }
    }

    /// Fold the chain hard at one interior joint, to shake the sweep out of a
    /// local minimum. `seed` advances each time so successive kicks try a
    /// different joint and a different plane rather than repeating one that
    /// already failed to help.
    fn kick(&mut self, seed: usize) {
        let n = self.joints.len();
        if n < 3 {
            return;
        }
        let k = 1 + seed % (n - 2);
        let pivot = self.joints[k].position;
        let Some(direction) = try_normalize(self.joints[k + 1].position - pivot) else {
            return;
        };
        let (u, v) = orthonormal_basis(direction);
        let axis = if seed.is_multiple_of(2) { u } else { v };
        let rotation = Quaternion::from_axis_angle(axis, std::f32::consts::FRAC_PI_2);
        for j in k + 1..n {
            self.joints[j].position =
                pivot + (self.joints[j].position - pivot).apply_quaternion(rotation);
        }
    }

    /// Direction of the segment feeding into joint `i`, or `None` at the root
    /// (which has nothing to bend relative to).
    fn previous_direction(&self, i: usize) -> Option<Vector3> {
        if i == 0 {
            return self.base_direction.and_then(try_normalize);
        }
        try_normalize(self.joints[i].position - self.joints[i - 1].position)
    }

    /// Re-place joint `i + 1` along `direction` and drag the rest of the chain
    /// rigidly with it.
    fn rebuild_from(&mut self, i: usize, direction: Vector3) {
        let n = self.joints.len();
        let old = self.joints[i + 1].position;
        let new = self.joints[i].position + direction * self.joints[i].length;
        let delta = new - old;
        for j in i + 1..n {
            self.joints[j].position = self.joints[j].position + delta;
        }
    }

    /// Write the solved pose onto the bound scene nodes.
    ///
    /// Each bone is rotated so its [`Self::bone_axis`] points along the solved
    /// segment. Only rotations are written — the positions come from the
    /// hierarchy, which is what makes the result a valid skeleton pose rather
    /// than a pile of detached bones.
    ///
    /// Bones must be a parent-to-child chain, root first.
    pub fn apply_to_scene(&self, arena: &mut ObjectArena) {
        let n = self.joints.len();
        if n < 2 {
            return;
        }

        // Whatever the root hangs off already contributes its own rotation.
        let mut parent_rotation = self
            .joints
            .first()
            .and_then(|j| j.scene_object)
            .and_then(|id| arena.get(id))
            .and_then(|o| o.parent)
            .and_then(|p| arena.get(p))
            .map(|p| p.world_quaternion())
            .unwrap_or_else(Quaternion::identity);

        for i in 0..n - 1 {
            let Some(direction) =
                try_normalize(self.joints[i + 1].position - self.joints[i].position)
            else {
                continue;
            };
            let world_rotation = shortest_arc(self.bone_axis, direction);
            if let Some(id) = self.joints[i].scene_object {
                if let Some(object) = arena.get_mut(id) {
                    object.quaternion = parent_rotation.conjugate().multiply(world_rotation);
                    object.update_matrix();
                }
            }
            parent_rotation = world_rotation;
        }
    }
}

/// Shortest rotation taking `from` to `to`. Both are normalised first.
pub fn shortest_arc(from: Vector3, to: Vector3) -> Quaternion {
    let (Some(from), Some(to)) = (try_normalize(from), try_normalize(to)) else {
        return Quaternion::identity();
    };
    let dot = from.dot(to).clamp(-1.0, 1.0);
    if dot > 1.0 - 1e-6 {
        return Quaternion::identity();
    }
    if dot < -1.0 + 1e-6 {
        // Exactly opposed: no unique shortest arc, so pick any perpendicular axis.
        let axis = crate::math::orthonormal_basis(from).0;
        return Quaternion::from_axis_angle(axis, std::f32::consts::PI);
    }
    let axis = from.cross(to);
    Quaternion::new(axis.x, axis.y, axis.z, 1.0 + dot).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI};
    use threers::core::Object3D;

    fn straight_arm(segments: usize, length: f32) -> IkChain {
        let points: Vec<Vector3> = (0..=segments)
            .map(|i| Vector3::new(i as f32 * length, 0.0, 0.0))
            .collect();
        IkChain::from_points(&points)
    }

    fn segment_lengths(chain: &IkChain) -> Vec<f32> {
        chain
            .joints
            .windows(2)
            .map(|w| (w[1].position - w[0].position).length())
            .collect()
    }

    #[test]
    fn fabrik_reaches_a_target_within_range() {
        let mut arm = straight_arm(3, 1.0);
        let target = Vector3::new(1.0, 2.0, 0.0);
        let result = arm.solve(target);
        assert!(result.reached, "error {} after {} iters", result.error, result.iterations);
        assert!((arm.tip() - target).length() <= arm.tolerance);
    }

    #[test]
    fn the_root_never_moves_and_segments_never_stretch() {
        let mut arm = straight_arm(4, 0.75);
        let original = segment_lengths(&arm);
        for target in [
            Vector3::new(0.0, 2.0, 0.0),
            Vector3::new(-1.0, -1.0, 1.0),
            Vector3::new(0.1, 0.1, 0.1),
            Vector3::new(2.9, 0.0, 0.0),
        ] {
            arm.solve(target);
            assert!(arm.root().length() < 1e-4, "root drifted to {:?}", arm.root());
            for (a, b) in segment_lengths(&arm).iter().zip(&original) {
                assert!((a - b).abs() < 1e-3, "segment stretched: {a} vs {b}");
            }
        }
    }

    #[test]
    fn an_unreachable_target_stretches_the_chain_straight_at_it() {
        let mut arm = straight_arm(3, 1.0);
        let target = Vector3::new(0.0, 100.0, 0.0);
        let result = arm.solve(target);
        assert!(result.out_of_reach);
        assert!(!result.reached);
        // Straight up, and fully extended.
        assert!((arm.tip() - Vector3::new(0.0, 3.0, 0.0)).length() < 1e-3);
        for j in &arm.joints {
            assert!(j.position.x.abs() < 1e-3 && j.position.z.abs() < 1e-3);
        }
    }

    #[test]
    fn a_target_at_the_root_does_not_produce_nans() {
        let mut arm = straight_arm(3, 1.0);
        arm.solve(Vector3::ZERO);
        for j in &arm.joints {
            assert!(j.position.x.is_finite() && j.position.y.is_finite() && j.position.z.is_finite());
        }
    }

    #[test]
    fn ccd_also_reaches_and_preserves_lengths() {
        let mut arm = straight_arm(3, 1.0);
        let original = segment_lengths(&arm);
        let target = Vector3::new(1.0, 1.5, 0.5);
        let result = arm.solve_ccd(target);
        assert!(result.reached, "error {}", result.error);
        for (a, b) in segment_lengths(&arm).iter().zip(&original) {
            assert!((a - b).abs() < 1e-3, "segment stretched: {a} vs {b}");
        }
        assert!(arm.root().length() < 1e-4);
    }

    #[test]
    fn a_hinge_constraint_keeps_the_joint_in_its_plane() {
        let mut arm = straight_arm(3, 1.0);
        // Elbow bends only about z, so the chain must stay in the XY plane.
        arm.set_constraint(1, IkConstraint::hinge(Vector3::new(0.0, 0.0, 1.0)));
        arm.set_constraint(2, IkConstraint::hinge(Vector3::new(0.0, 0.0, 1.0)));
        arm.solve(Vector3::new(1.0, 1.0, 2.0));

        // The plane is the one the *hinges* establish, not the world XY plane.
        // A joint's constraint governs the segment leaving it, measured against
        // the segment arriving — so the root segment, which has nothing arriving
        // at it, is free by construction and can tilt the whole arm off z = 0.
        // What the hinges then guarantee is that nothing downstream of the first
        // one leaves the plane that one set, however far out of reach the target
        // is.
        let plane_z = arm.joints[1].position.z;
        for (i, j) in arm.joints.iter().enumerate().skip(2) {
            assert!(
                (j.position.z - plane_z).abs() < 0.05,
                "joint {i} left the hinge plane at z = {} (plane at {plane_z})",
                j.position.z
            );
        }
    }

    #[test]
    fn a_limited_hinge_never_bends_past_its_stop() {
        let mut arm = straight_arm(3, 1.0);
        // A knee: bends one way only, up to 90 degrees.
        for i in [1, 2] {
            arm.set_constraint(
                i,
                IkConstraint::hinge_limited(Vector3::new(0.0, 0.0, 1.0), 0.0, FRAC_PI_2),
            );
        }
        arm.solve(Vector3::new(-1.0, -2.0, 0.0));

        for i in 1..arm.joints.len() - 1 {
            let a = (arm.joints[i].position - arm.joints[i - 1].position).normalize();
            let b = (arm.joints[i + 1].position - arm.joints[i].position).normalize();
            let angle = a.dot(b).clamp(-1.0, 1.0).acos();
            assert!(
                angle <= FRAC_PI_2 + 0.05,
                "joint {i} bent {angle} rad, past its 90 degree stop"
            );
        }
    }

    #[test]
    fn a_cone_constraint_bounds_the_bend() {
        let mut arm = straight_arm(4, 1.0);
        let limit = 0.35;
        for i in 1..4 {
            arm.set_constraint(i, IkConstraint::cone(limit));
        }
        arm.solve(Vector3::new(0.0, 3.0, 0.0));
        for i in 1..arm.joints.len() - 1 {
            let a = (arm.joints[i].position - arm.joints[i - 1].position).normalize();
            let b = (arm.joints[i + 1].position - arm.joints[i].position).normalize();
            let angle = a.dot(b).clamp(-1.0, 1.0).acos();
            assert!(angle <= limit + 0.05, "joint {i} bent {angle}, past the {limit} cone");
        }
    }

    #[test]
    fn a_two_joint_chain_is_handled() {
        let mut arm = straight_arm(1, 2.0);
        let result = arm.solve(Vector3::new(0.0, 2.0, 0.0));
        assert!(result.reached);
        assert!((arm.tip() - Vector3::new(0.0, 2.0, 0.0)).length() < 1e-3);
    }

    #[test]
    fn a_degenerate_chain_reports_rather_than_panics() {
        let mut chain = IkChain::from_points(&[Vector3::ZERO]);
        let r = chain.solve(Vector3::new(1.0, 0.0, 0.0));
        assert!(!r.reached);
        let mut empty = IkChain::from_points(&[]);
        empty.solve(Vector3::ZERO);
        empty.solve_ccd(Vector3::ZERO);
    }

    #[test]
    fn shortest_arc_rotates_one_vector_onto_another() {
        for (from, to) in [
            (Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)),
            (Vector3::UP, Vector3::new(0.0, -1.0, 0.0)), // exactly opposed
            (Vector3::UP, Vector3::UP),                  // identical
            (Vector3::new(1.0, 2.0, 3.0), Vector3::new(-2.0, 0.5, 1.0)),
        ] {
            let q = shortest_arc(from, to);
            let rotated = from.normalize().apply_quaternion(q);
            assert!(
                (rotated - to.normalize()).length() < 1e-3,
                "{from:?} -> {to:?} gave {rotated:?}"
            );
        }
    }

    #[test]
    fn a_chain_can_be_read_from_and_written_back_to_a_scene() {
        let mut arena = ObjectArena::new();
        // Three bones running up +Y, each one unit long.
        let root = arena.insert(Object3D::group());
        let mid = arena.insert(Object3D::group());
        let tip = arena.insert(Object3D::group());
        arena.get_mut(mid).unwrap().position = Vector3::new(0.0, 1.0, 0.0);
        arena.get_mut(tip).unwrap().position = Vector3::new(0.0, 1.0, 0.0);
        arena.add_child(root, mid);
        arena.add_child(mid, tip);
        arena.update_world_matrices(root, threers::math::Matrix4::identity());

        let mut chain = IkChain::from_scene_chain(&arena, root, tip).unwrap();
        assert_eq!(chain.joints.len(), 3);
        assert!((chain.total_length() - 2.0).abs() < 1e-4);

        // Reach out along +X and write the pose back.
        let target = Vector3::new(2.0, 0.0, 0.0);
        let result = chain.solve(target);
        assert!(result.reached, "error {}", result.error);
        chain.apply_to_scene(&mut arena);
        arena.update_world_matrices(root, threers::math::Matrix4::identity());

        // The scene now agrees with the solved chain.
        let world_tip = arena.get(tip).unwrap().world_position();
        assert!(
            (world_tip - target).length() < 0.05,
            "scene tip at {world_tip:?}, target {target:?}"
        );
    }

    #[test]
    fn from_scene_chain_rejects_a_tip_that_is_not_a_descendant() {
        let mut arena = ObjectArena::new();
        let a = arena.insert(Object3D::group());
        let b = arena.insert(Object3D::group());
        // Never linked.
        assert!(IkChain::from_scene_chain(&arena, a, b).is_none());
    }

    #[test]
    fn constraints_do_not_stop_the_solver_converging_on_easy_targets() {
        let mut arm = straight_arm(3, 1.0);
        for i in 1..3 {
            arm.set_constraint(i, IkConstraint::cone(PI));
        }
        // A full-PI cone is no constraint at all, so this must still reach.
        let target = Vector3::new(1.0, 2.0, 0.0);
        assert!(arm.solve(target).reached);
    }
}
