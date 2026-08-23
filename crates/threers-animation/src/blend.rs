//! Handing a body between animation and physics.
//!
//! The hard part of a ragdoll is not the fall — it is the moment control
//! changes hands. Switch outright and the character visibly snaps from its
//! animated pose to whatever the solver thinks; switch back and it snaps again.
//! [`PhysicsBlend`] crossfades the two over a short window so neither transition
//! reads as a glitch.
//!
//! # Frame order
//!
//! ```text
//! 1. Advance animation (mixer / animator) → animated poses
//! 2. blend.update(dt, animated, &mut world) → pose to *draw*
//! 3. Write drawn poses onto scene nodes for blended bodies
//! 4. world.sync_from_scene — remaining kinematic bodies only
//! 5. world.step(dt)
//! 6. world.sync_to_scene_where(|b| !blending(b)) — skip bodies mid-handover
//! ```
//!
//! Calling [`World::sync_to_scene`](threers_physics::World::sync_to_scene) during
//! a handover overwrites the blended draw pose with the raw simulation pose and
//! undoes the crossfade.
//!
//! Requires the `physics` feature.

use crate::easing::Easing;
use threers::core::{ObjectArena, ObjectId};
use threers::math::{Quaternion, Vector3};
use threers_physics::prelude::{BodyId, BodyType, Isometry, RigidBody, World};

/// Which side currently owns a body's transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendState {
    /// Animation owns it outright.
    Animated,
    /// Crossfading toward the simulation.
    ToSimulated,
    /// The simulation owns it outright.
    Simulated,
    /// Crossfading back to animation.
    ToAnimated,
}

/// Crossfades one body between an animated pose and a simulated one.
///
/// ```no_run
/// use threers_animation::prelude::*;
/// use threers_physics::prelude::*;
///
/// let mut world = World::new();
/// let body = world.add_body(RigidBody::kinematic().shape(Shape::capsule(0.6, 0.3)));
/// let mut blend = PhysicsBlend::new(body, 0.25);
///
/// // The character is hit.
/// blend.go_limp(&mut world);
///
/// // Each frame: give it the animated pose, let it blend toward the simulation.
/// let animated = Isometry::from_translation(Vector3::new(0.0, 1.0, 0.0));
/// blend.update(1.0 / 60.0, animated, &mut world);
/// ```
#[derive(Debug, Clone)]
pub struct PhysicsBlend {
    body: BodyId,
    state: BlendState,
    /// How long a handover takes.
    pub duration: f32,
    pub easing: Easing,
    elapsed: f32,
    /// Pose frozen when the current transition began.
    anchor: Isometry,
    /// Last animated sample (for limp velocity inheritance + freeze).
    last_animated: Isometry,
    last_dt: f32,
}

impl PhysicsBlend {
    /// Start under animation control.
    pub fn new(body: BodyId, duration: f32) -> Self {
        Self {
            body,
            state: BlendState::Animated,
            duration: duration.max(0.0),
            easing: Easing::CubicInOut,
            elapsed: 0.0,
            anchor: Isometry::IDENTITY,
            last_animated: Isometry::IDENTITY,
            last_dt: 1.0 / 60.0,
        }
    }

    pub fn body(&self) -> BodyId {
        self.body
    }

    pub fn state(&self) -> BlendState {
        self.state
    }

    /// Fraction of the way through the current handover, `0..1`.
    pub fn progress(&self) -> f32 {
        if self.duration <= 0.0 {
            return 1.0;
        }
        (self.elapsed / self.duration).clamp(0.0, 1.0)
    }

    /// Hand the body to the simulation — the character is hit, or dies.
    ///
    /// The body becomes dynamic immediately, so it starts falling and colliding
    /// at once, but its *drawn* pose crossfades from the frozen animated pose
    /// over [`Self::duration`]. Momentum from the animation is carried into the
    /// simulation when the kinematic body had no velocity of its own yet.
    pub fn go_limp(&mut self, world: &mut World) {
        if matches!(self.state, BlendState::Simulated | BlendState::ToSimulated) {
            return;
        }
        let Some(body) = world.body_mut(self.body) else {
            return;
        };
        // Freeze the last animated sample so a continuing clip cannot yank the
        // blend start mid-handover (`anchor.lerp(animated, 1.0)` used to do that).
        self.anchor = self.last_animated;
        inherit_animation_velocity(body, self.last_animated, self.last_dt);
        body.set_body_type(BodyType::Dynamic);
        self.state = BlendState::ToSimulated;
        self.elapsed = 0.0;
    }

    /// Take the body back under animation control — standing up again.
    pub fn recover(&mut self, world: &mut World) {
        if matches!(self.state, BlendState::Animated | BlendState::ToAnimated) {
            return;
        }
        let Some(body) = world.body_mut(self.body) else {
            return;
        };
        self.anchor = body.position;
        // Stay dynamic through the blend: switching to kinematic now would stop
        // the body dead and lose the settling motion the blend is meant to hide.
        self.state = BlendState::ToAnimated;
        self.elapsed = 0.0;
    }

    /// Advance the blend and return the pose to draw.
    ///
    /// `animated` is where the animation says the body should be this frame.
    /// Call after stepping the world for simulated poses to be current, or
    /// before the step when still animated (so kinematic targets land next step).
    pub fn update(&mut self, dt: f32, animated: Isometry, world: &mut World) -> Isometry {
        if dt.is_finite() && dt > 0.0 {
            self.elapsed += dt;
            self.last_dt = dt;
        }
        self.last_animated = animated;

        let simulated = world
            .body(self.body)
            .map(|b| b.position)
            .unwrap_or(animated);

        match self.state {
            BlendState::Animated => {
                if let Some(body) = world.body_mut(self.body) {
                    if body.is_kinematic() {
                        body.set_kinematic_target(animated);
                    }
                }
                animated
            }
            BlendState::Simulated => simulated,

            BlendState::ToSimulated => {
                let t = self.easing.apply(self.progress());
                if self.progress() >= 1.0 {
                    self.state = BlendState::Simulated;
                }
                // Frozen animated pose → live simulation.
                blend_pose(self.anchor, simulated, t)
            }

            BlendState::ToAnimated => {
                let t = self.easing.apply(self.progress());
                if self.progress() >= 1.0 {
                    if let Some(body) = world.body_mut(self.body) {
                        body.set_body_type(BodyType::Kinematic);
                        body.set_position(animated);
                        body.set_linear_velocity(Vector3::ZERO);
                        body.set_angular_velocity(Vector3::ZERO);
                    }
                    self.state = BlendState::Animated;
                }
                // Frozen simulated pose at recover → live animation.
                blend_pose(self.anchor, animated, t)
            }
        }
    }

    /// Write the drawn pose onto the body's bound scene node (if any).
    pub fn apply_to_scene(&self, pose: Isometry, world: &World, arena: &mut ObjectArena) {
        let Some(body) = world.body(self.body) else {
            return;
        };
        let Some(id) = body.scene_object else {
            return;
        };
        write_pose(arena, id, pose);
    }

    /// Whether the body is currently mid-handover.
    pub fn is_blending(&self) -> bool {
        matches!(self.state, BlendState::ToSimulated | BlendState::ToAnimated)
    }

    /// True when [`World::sync_to_scene`] must not overwrite this body's mesh.
    pub fn owns_scene_write(&self) -> bool {
        !matches!(self.state, BlendState::Simulated)
    }
}

fn inherit_animation_velocity(body: &mut RigidBody, animated: Isometry, dt: f32) {
    let dt = dt.max(1e-4);
    // If the kinematic body already carries motion from `set_kinematic_target`
    // / `sync_from_scene`, keep it. Otherwise bake velocity from the last
    // animated sample so a mid-stride limp keeps travelling.
    if body.linear_velocity.length_sq() < 1e-8 {
        body.set_linear_velocity((animated.translation - body.position.translation) * (1.0 / dt));
    }
    if body.angular_velocity.length_sq() < 1e-8 {
        let mut delta = animated.rotation.multiply(body.position.rotation.conjugate());
        if delta.w < 0.0 {
            delta = Quaternion::new(-delta.x, -delta.y, -delta.z, -delta.w);
        }
        let sin_half = (delta.x * delta.x + delta.y * delta.y + delta.z * delta.z).sqrt();
        if sin_half > 1e-6 {
            let angle = 2.0 * sin_half.atan2(delta.w.clamp(-1.0, 1.0));
            body.set_angular_velocity(
                Vector3::new(delta.x, delta.y, delta.z) * (angle / (sin_half * dt)),
            );
        }
    }
}

fn write_pose(arena: &mut ObjectArena, id: ObjectId, pose: Isometry) {
    let Some(object) = arena.get_mut(id) else {
        return;
    };
    object.position = pose.translation;
    object.quaternion = pose.rotation;
    object.update_matrix();
}

fn blend_pose(from: Isometry, to: Isometry, t: f32) -> Isometry {
    let t = t.clamp(0.0, 1.0);
    Isometry::new(
        Vector3::new(
            from.translation.x + (to.translation.x - from.translation.x) * t,
            from.translation.y + (to.translation.y - from.translation.y) * t,
            from.translation.z + (to.translation.z - from.translation.z) * t,
        ),
        from.rotation.slerp(to.rotation, t),
    )
}

/// Blend a whole set of bodies at once — a ragdoll is many bones, not one.
#[derive(Debug, Clone, Default)]
pub struct RagdollBlend {
    parts: Vec<PhysicsBlend>,
}

impl RagdollBlend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, body: BodyId, duration: f32) -> &mut Self {
        self.parts.push(PhysicsBlend::new(body, duration));
        self
    }

    pub fn parts(&self) -> &[PhysicsBlend] {
        &self.parts
    }

    /// Hand every part to the simulation.
    pub fn go_limp(&mut self, world: &mut World) {
        for part in &mut self.parts {
            part.go_limp(world);
        }
    }

    pub fn recover(&mut self, world: &mut World) {
        for part in &mut self.parts {
            part.recover(world);
        }
    }

    /// Advance every part. `animated` supplies each part's animated pose by
    /// index; return `None` to hold the part where it is.
    pub fn update(
        &mut self,
        dt: f32,
        world: &mut World,
        mut animated: impl FnMut(usize, BodyId) -> Option<Isometry>,
    ) -> Vec<Isometry> {
        self.parts
            .iter_mut()
            .enumerate()
            .map(|(i, part)| {
                let pose = animated(i, part.body())
                    .or_else(|| world.body(part.body()).map(|b| b.position))
                    .unwrap_or(Isometry::IDENTITY);
                part.update(dt, pose, world)
            })
            .collect()
    }

    /// Write every drawn pose onto bound scene nodes.
    pub fn apply_to_scene(&self, poses: &[Isometry], world: &World, arena: &mut ObjectArena) {
        for (part, pose) in self.parts.iter().zip(poses.iter()) {
            part.apply_to_scene(*pose, world, arena);
        }
    }

    /// Bodies the blend currently draws (skip these in `sync_to_scene_where`).
    pub fn owns_scene_write(&self, body: BodyId) -> bool {
        self.parts
            .iter()
            .any(|p| p.body() == body && p.owns_scene_write())
    }

    pub fn len(&self) -> usize {
        self.parts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// True once every part has finished handing over.
    pub fn is_settled(&self) -> bool {
        self.parts.iter().all(|p| !p.is_blending())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers_physics::prelude::{RigidBody, Shape};

    fn world_with_body() -> (World, BodyId) {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()));
        let body = world.add_body(
            RigidBody::kinematic()
                .shape(Shape::capsule(0.6, 0.3))
                .translation(Vector3::new(0.0, 3.0, 0.0)),
        );
        (world, body)
    }

    #[test]
    fn an_animated_body_follows_the_animation_exactly() {
        let (mut world, body) = world_with_body();
        let mut blend = PhysicsBlend::new(body, 0.25);
        assert_eq!(blend.state(), BlendState::Animated);

        for i in 0..30 {
            let pose = Isometry::from_translation(Vector3::new(i as f32 * 0.1, 3.0, 0.0));
            let drawn = blend.update(1.0 / 60.0, pose, &mut world);
            world.step(1.0 / 60.0);
            assert_eq!(drawn.translation, pose.translation, "animation should win outright");
        }
    }

    #[test]
    fn going_limp_hands_over_without_a_visible_jump() {
        let (mut world, body) = world_with_body();
        let mut blend = PhysicsBlend::new(body, 0.25);
        let pose = Isometry::from_translation(Vector3::new(0.0, 3.0, 0.0));
        for _ in 0..10 {
            blend.update(1.0 / 60.0, pose, &mut world);
            world.step(1.0 / 60.0);
        }

        blend.go_limp(&mut world);
        assert_eq!(blend.state(), BlendState::ToSimulated);
        assert!(world.body(body).unwrap().is_dynamic(), "it must fall immediately");
        assert!(world.body(body).unwrap().mass() > 0.0);

        let first = blend.update(1.0 / 60.0, pose, &mut world);
        assert!(
            (first.translation - pose.translation).length() < 0.1,
            "the handover jumped to {:?}",
            first.translation
        );

        for _ in 0..120 {
            world.step(1.0 / 60.0);
            blend.update(1.0 / 60.0, pose, &mut world);
        }
        assert_eq!(blend.state(), BlendState::Simulated);
        let drawn = blend.update(1.0 / 60.0, pose, &mut world);
        let simulated = world.body(body).unwrap().position;
        assert_eq!(drawn.translation, simulated.translation);
        assert!(simulated.translation.y < 3.0, "it never fell");
    }

    #[test]
    fn limp_freezes_animated_pose_even_if_clip_keeps_moving() {
        let (mut world, body) = world_with_body();
        let mut blend = PhysicsBlend::new(body, 0.5);
        let start = Isometry::from_translation(Vector3::new(0.0, 3.0, 0.0));
        for _ in 0..5 {
            blend.update(1.0 / 60.0, start, &mut world);
            world.step(1.0 / 60.0);
        }
        blend.go_limp(&mut world);

        // Animation keeps walking away — the blend start must stay frozen.
        let far = Isometry::from_translation(Vector3::new(10.0, 3.0, 0.0));
        let first = blend.update(1.0 / 60.0, far, &mut world);
        assert!(
            (first.translation - start.translation).length() < 0.15,
            "blend followed the moving clip to {:?} instead of freezing at {:?}",
            first.translation,
            start.translation
        );
    }

    #[test]
    fn limp_inherits_animation_velocity_when_kinematic_was_still() {
        let (mut world, body) = world_with_body();
        let mut blend = PhysicsBlend::new(body, 0.25);
        // Drive animation sideways without stepping (no kinematic velocity baked).
        let a = Isometry::from_translation(Vector3::new(0.0, 3.0, 0.0));
        let b = Isometry::from_translation(Vector3::new(1.2, 3.0, 0.0));
        blend.update(1.0 / 60.0, a, &mut world);
        blend.update(1.0 / 60.0, b, &mut world);
        blend.go_limp(&mut world);
        let v = world.body(body).unwrap().linear_velocity;
        assert!(
            v.x > 1.0,
            "expected inherited +X velocity from the animated stride, got {v:?}"
        );
    }

    #[test]
    fn recovering_returns_control_to_the_animation() {
        let (mut world, body) = world_with_body();
        let mut blend = PhysicsBlend::new(body, 0.2);
        let pose = Isometry::from_translation(Vector3::new(0.0, 3.0, 0.0));

        blend.go_limp(&mut world);
        for _ in 0..120 {
            world.step(1.0 / 60.0);
            blend.update(1.0 / 60.0, pose, &mut world);
        }
        assert_eq!(blend.state(), BlendState::Simulated);

        blend.recover(&mut world);
        assert_eq!(blend.state(), BlendState::ToAnimated);
        for _ in 0..60 {
            world.step(1.0 / 60.0);
            blend.update(1.0 / 60.0, pose, &mut world);
        }
        assert_eq!(blend.state(), BlendState::Animated);
        assert!(world.body(body).unwrap().is_kinematic());
        let drawn = blend.update(1.0 / 60.0, pose, &mut world);
        assert!((drawn.translation - pose.translation).length() < 1e-4);
        assert!(world.body(body).unwrap().linear_velocity.length() < 1e-4);
    }

    #[test]
    fn repeated_calls_do_not_restart_a_handover() {
        let (mut world, body) = world_with_body();
        let mut blend = PhysicsBlend::new(body, 0.5);
        blend.update(1.0 / 60.0, Isometry::IDENTITY, &mut world);
        blend.go_limp(&mut world);
        blend.update(0.25, Isometry::IDENTITY, &mut world);
        let halfway = blend.progress();
        blend.go_limp(&mut world);
        assert!(
            (blend.progress() - halfway).abs() < 1e-5,
            "a second call restarted the blend"
        );
    }

    #[test]
    fn a_zero_duration_blend_switches_instantly() {
        let (mut world, body) = world_with_body();
        let mut blend = PhysicsBlend::new(body, 0.0);
        blend.update(1.0 / 60.0, Isometry::IDENTITY, &mut world);
        blend.go_limp(&mut world);
        blend.update(1.0 / 60.0, Isometry::IDENTITY, &mut world);
        assert_eq!(blend.state(), BlendState::Simulated);
    }

    #[test]
    fn a_removed_body_does_not_panic_the_blend() {
        let (mut world, body) = world_with_body();
        let mut blend = PhysicsBlend::new(body, 0.25);
        blend.update(1.0 / 60.0, Isometry::IDENTITY, &mut world);
        blend.go_limp(&mut world);
        world.remove_body(body);
        let pose = Isometry::from_translation(Vector3::new(1.0, 2.0, 3.0));
        let drawn = blend.update(1.0 / 60.0, pose, &mut world);
        assert!(drawn.translation.x.is_finite());
        blend.recover(&mut world);
    }

    #[test]
    fn a_ragdoll_hands_over_every_part() {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()));
        let mut ragdoll = RagdollBlend::new();
        let mut bodies = Vec::new();
        for i in 0..5 {
            let b = world.add_body(
                RigidBody::kinematic()
                    .shape(Shape::ball(0.2))
                    .translation(Vector3::new(0.0, 2.0 + i as f32 * 0.5, 0.0)),
            );
            bodies.push(b);
            ragdoll.add(b, 0.2);
        }
        assert_eq!(ragdoll.len(), 5);

        // Seed last_animated before limp.
        ragdoll.update(1.0 / 60.0, &mut world, |i, _| {
            Some(Isometry::from_translation(Vector3::new(
                0.0,
                2.0 + i as f32 * 0.5,
                0.0,
            )))
        });

        ragdoll.go_limp(&mut world);
        for b in &bodies {
            assert!(world.body(*b).unwrap().is_dynamic());
        }

        for _ in 0..120 {
            world.step(1.0 / 60.0);
            let poses = ragdoll.update(1.0 / 60.0, &mut world, |_, _| None);
            assert_eq!(poses.len(), 5);
            for p in poses {
                assert!(p.translation.y.is_finite());
            }
        }
        assert!(ragdoll.is_settled());
        assert!(world.body(bodies[4]).unwrap().translation().y < 4.0);
    }
}
