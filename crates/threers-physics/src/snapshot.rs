//! Capturing and restoring simulation state.
//!
//! A snapshot holds everything that *changes* — transforms, velocities, sleep
//! state, and the accumulated impulses the solver warm-starts from — but not the
//! bodies, shapes or joints themselves. Restoring rewinds a world to an earlier
//! moment; it does not rebuild one from nothing.
//!
//! That split is deliberate. Rollback networking, replays and
//! "simulate-ahead-then-undo" all rewind a world whose *structure* is unchanged,
//! and keeping shapes out of the snapshot makes it small enough to take every
//! frame.
//!
//! # Why the impulses are in here
//!
//! Warm starting means a step's result depends on the previous step's impulses.
//! A snapshot that restored only positions and velocities would produce a
//! *different* next step from the original, which defeats the point — the whole
//! value of a rollback is that replaying gives the same answer.

use crate::body::BodyType;
use crate::contact::ManifoldKey;
use crate::math::Isometry;
use crate::world::World;
use threers::math::Vector3;

/// One body's mutable state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodySnapshot {
    pub position: Isometry,
    pub previous_position: Isometry,
    pub linear_velocity: Vector3,
    pub angular_velocity: Vector3,
    pub body_type: BodyType,
    pub sleeping: bool,
    pub rest_time: f32,
    pub kinematic_target: Option<Isometry>,
    pub enabled: bool,
}

/// Warm-start impulses for one contact, keyed so they can be matched back.
///
/// The contact's position in `a`'s local frame travels with the impulses,
/// because that is what the next step matches on: an impulse restored onto a
/// manifold whose points have since moved further than
/// `MATCH_DISTANCE` is silently dropped, and the replay diverges on its first
/// step.
#[derive(Debug, Clone, PartialEq)]
pub struct ContactSnapshot {
    pub key: ManifoldKey,
    /// `(contact point in a's local frame, normal impulse, tangent impulses)`.
    pub points: Vec<(Vector3, f32, [f32; 2])>,
    /// Whether the pair was actually touching, not merely speculative. Restored
    /// so the next step reports the same begin/end collision events.
    pub touching: bool,
}

/// A point in a world's history.
#[derive(Debug, Clone, PartialEq)]
pub struct WorldSnapshot {
    /// Indexed by body slot; `None` for an empty slot.
    pub bodies: Vec<Option<BodySnapshot>>,
    pub contacts: Vec<ContactSnapshot>,
    /// Joint warm-start impulses, in joint-set order.
    pub joints: Vec<(Vector3, Vector3, f32)>,
    /// Leftover real time not yet consumed by a fixed step.
    pub accumulator: f32,
    pub gravity: Vector3,
}

impl WorldSnapshot {
    /// Bodies captured.
    pub fn body_count(&self) -> usize {
        self.bodies.iter().filter(|b| b.is_some()).count()
    }

    /// Rough size in bytes, for budgeting a rollback buffer.
    pub fn size_hint(&self) -> usize {
        self.bodies.len() * std::mem::size_of::<Option<BodySnapshot>>()
            + self
                .contacts
                .iter()
                .map(|c| {
                    std::mem::size_of::<ManifoldKey>()
                        + c.points.len() * std::mem::size_of::<(Vector3, f32, [f32; 2])>()
                })
                .sum::<usize>()
            + self.joints.len() * std::mem::size_of::<(Vector3, Vector3, f32)>()
    }
}

impl World {
    /// Capture the current state.
    ///
    /// ```
    /// use threers_physics::prelude::*;
    ///
    /// let mut world = World::new();
    /// world.add_body(RigidBody::fixed().shape(Shape::ground()));
    /// let ball = world.add_body(
    ///     RigidBody::dynamic()
    ///         .shape(Shape::ball(0.5))
    ///         .translation(Vector3::new(0.0, 5.0, 0.0)),
    /// );
    ///
    /// for _ in 0..60 { world.step_fixed(); }
    /// let saved = world.snapshot();
    /// let at_save = world.body(ball).unwrap().translation();
    ///
    /// for _ in 0..60 { world.step_fixed(); }
    /// world.restore(&saved);
    ///
    /// assert_eq!(world.body(ball).unwrap().translation(), at_save);
    /// ```
    pub fn snapshot(&self) -> WorldSnapshot {
        let mut bodies = Vec::with_capacity(self.bodies().slot_count());
        for slot in 0..self.bodies().slot_count() {
            bodies.push(self.bodies().by_index(slot).map(|b| BodySnapshot {
                position: b.position,
                previous_position: b.previous_position(),
                linear_velocity: b.linear_velocity,
                angular_velocity: b.angular_velocity,
                body_type: b.body_type,
                sleeping: b.is_sleeping(),
                rest_time: b.rest_time(),
                kinematic_target: b.kinematic_target(),
                enabled: b.enabled,
            }));
        }

        let contacts = self
            .contacts()
            .iter()
            .map(|m| ContactSnapshot {
                key: m.key,
                points: m
                    .points
                    .iter()
                    .map(|p| (p.local_a, p.normal_impulse, p.tangent_impulse))
                    .collect(),
                touching: m.touching,
            })
            .collect();

        let joints = self
            .joints()
            .iter()
            .map(|(_, j)| j.warm_start_state())
            .collect();

        WorldSnapshot {
            bodies,
            contacts,
            joints,
            accumulator: self.accumulator_remaining(),
            gravity: self.gravity,
        }
    }

    /// Rewind to a captured state.
    ///
    /// Returns `false` if the world's structure has changed since the snapshot
    /// was taken — bodies added or removed — because restoring into a different
    /// set of slots would put state on the wrong bodies. Rebuild the world and
    /// restore into that instead.
    pub fn restore(&mut self, snapshot: &WorldSnapshot) -> bool {
        // The world may have grown slots since the snapshot, as long as every
        // one of them is empty now: adding a body and removing it again leaves
        // a free slot behind, and that is not a structural change — there is
        // still nothing there to restore state onto.
        if snapshot.bodies.len() > self.bodies().slot_count() {
            return false;
        }
        for slot in snapshot.bodies.len()..self.bodies().slot_count() {
            if self.bodies().by_index(slot).is_some() {
                return false;
            }
        }

        for (slot, saved) in snapshot.bodies.iter().enumerate() {
            let occupied = self.bodies().by_index(slot).is_some();
            match (saved, occupied) {
                (Some(state), true) => {
                    let Some(body) = self.bodies_mut().by_index_mut(slot) else {
                        continue;
                    };
                    body.restore_state(
                        state.position,
                        state.previous_position,
                        state.linear_velocity,
                        state.angular_velocity,
                        state.body_type,
                        state.sleeping,
                        state.rest_time,
                        state.kinematic_target,
                        state.enabled,
                    );
                }
                (None, false) => {}
                // Slot occupancy differs: the structure changed underneath us.
                _ => return false,
            }
        }

        // Put the warm-start impulses back, or the next step diverges from the
        // one that originally followed this state.
        self.restore_contacts(&snapshot.contacts);

        let joints: Vec<_> = snapshot.joints.clone();
        for ((_, joint), state) in self.joints_mut().iter_mut().zip(joints) {
            joint.set_warm_start_state(state);
        }

        self.gravity = snapshot.gravity;
        self.set_accumulator(snapshot.accumulator);
        self.refresh_queries();
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    fn falling_world() -> (World, Vec<BodyId>) {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.8));
        let mut ids = Vec::new();
        for i in 0..12 {
            ids.push(world.add_body(
                RigidBody::dynamic()
                    .shape(Shape::cuboid(0.4, 0.4, 0.4))
                    .translation(Vector3::new(
                        (i % 3) as f32 * 0.9 - 0.9,
                        1.0 + (i / 3) as f32 * 0.95,
                        0.0,
                    ))
                    .friction(0.8),
            ));
        }
        (world, ids)
    }

    fn state(world: &World, ids: &[BodyId]) -> Vec<(Vector3, Vector3)> {
        ids.iter()
            .filter_map(|id| world.body(*id))
            .map(|b| (b.translation(), b.linear_velocity))
            .collect()
    }

    #[test]
    fn restoring_rewinds_the_world_exactly() {
        let (mut world, ids) = falling_world();
        for _ in 0..60 {
            world.step_fixed();
        }
        let saved = world.snapshot();
        let at_save = state(&world, &ids);

        for _ in 0..120 {
            world.step_fixed();
        }
        assert_ne!(state(&world, &ids), at_save, "the world should have moved on");

        assert!(world.restore(&saved));
        assert_eq!(state(&world, &ids), at_save, "restore did not rewind exactly");
    }

    #[test]
    fn a_replayed_run_matches_the_original_step_for_step() {
        // The property rollback networking actually needs: rewinding and
        // replaying must reproduce the same future, not merely the same past.
        // This is what forces the warm-start impulses into the snapshot.
        let (mut world, ids) = falling_world();
        for _ in 0..60 {
            world.step_fixed();
        }
        let saved = world.snapshot();

        let mut original = Vec::new();
        for _ in 0..90 {
            world.step_fixed();
            original.push(state(&world, &ids));
        }

        assert!(world.restore(&saved));
        for (step, expected) in original.iter().enumerate() {
            world.step_fixed();
            assert_eq!(
                &state(&world, &ids),
                expected,
                "replay diverged at step {step}"
            );
        }
    }

    #[test]
    fn sleep_state_survives_a_round_trip() {
        let (mut world, ids) = falling_world();
        for _ in 0..600 {
            world.step_fixed();
        }
        let sleeping: Vec<bool> = ids
            .iter()
            .map(|id| world.body(*id).unwrap().is_sleeping())
            .collect();
        assert!(sleeping.iter().any(|s| *s), "nothing settled");

        let saved = world.snapshot();
        // Wake everything, then rewind.
        world.wake_all();
        world.step_fixed();
        assert!(world.restore(&saved));

        let after: Vec<bool> = ids
            .iter()
            .map(|id| world.body(*id).unwrap().is_sleeping())
            .collect();
        assert_eq!(after, sleeping, "sleep state was not restored");
    }

    #[test]
    fn restoring_into_a_changed_world_is_refused_rather_than_corrupting_it() {
        let (mut world, _) = falling_world();
        for _ in 0..30 {
            world.step_fixed();
        }
        let saved = world.snapshot();

        let extra = world.add_body(RigidBody::dynamic().shape(Shape::ball(0.3)));
        assert!(
            !world.restore(&saved),
            "restoring into a world with an extra body should be refused"
        );

        // Removing it again makes the snapshot valid, since the slot is empty
        // in both.
        world.remove_body(extra);
        assert!(world.restore(&saved));
    }

    #[test]
    fn queries_are_correct_immediately_after_a_restore() {
        // The spatial index describes where bodies were, so a restore that
        // forgot to refresh it would answer with pre-rewind positions.
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()));
        let ball = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.5))
                .translation(Vector3::new(0.0, 20.0, 0.0))
                .can_sleep(false),
        );
        let saved = world.snapshot();

        for _ in 0..200 {
            world.step_fixed();
        }
        assert!(world.body(ball).unwrap().translation().y < 5.0);

        world.restore(&saved);
        let ray = Ray::new(Vector3::new(0.0, 30.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        let hit = world.raycast(&ray, 100.0, QueryFilter::default()).unwrap();
        assert_eq!(hit.body, ball, "the query saw the pre-restore position");
        assert!((hit.toi - 9.5).abs() < 0.1, "toi = {}", hit.toi);
    }

    #[test]
    fn a_snapshot_reports_its_own_size() {
        let (world, _) = falling_world();
        let saved = world.snapshot();
        assert_eq!(saved.body_count(), 13);
        assert!(saved.size_hint() > 0);
    }

    #[test]
    fn snapshots_can_be_taken_every_step_without_side_effects() {
        let (mut world, ids) = falling_world();
        let mut history = Vec::new();
        for _ in 0..120 {
            history.push(world.snapshot());
            world.step_fixed();
        }
        let ending = state(&world, &ids);

        // Rewinding to any point and replaying must reach the same place.
        world.restore(&history[40]);
        for _ in 40..120 {
            world.step_fixed();
        }
        assert_eq!(state(&world, &ids), ending, "replay from step 40 diverged");
    }
}
