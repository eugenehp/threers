//! Persistent contact manifolds.
//!
//! The narrow phase rebuilds contacts from scratch every step, but the solver
//! wants the *previous* step's impulses as a starting guess — warm starting is
//! the difference between a stack that settles in two steps and one that sinks
//! and jitters for twenty. This module matches freshly generated contacts to
//! last step's by their position in each body's local frame and carries the
//! impulses across.

use crate::body::BodyId;
use crate::math::Isometry;
use crate::narrowphase::RawManifold;
use std::collections::BTreeMap;
use threers::math::Vector3;

/// Identifies a contact pair across steps.
///
/// Sub-shape indices are part of the key so that each triangle of a mesh, and
/// each part of a compound, warm-starts independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ManifoldKey {
    pub body_a: BodyId,
    pub body_b: BodyId,
    pub collider_a: u32,
    pub collider_b: u32,
    pub sub_a: u32,
    pub sub_b: u32,
}

/// A single contact within a manifold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactPoint {
    /// Witness point on body `a`, world space.
    pub point_a: Vector3,
    /// Witness point on body `b`, world space.
    pub point_b: Vector3,
    /// `point_a` in `a`'s local frame — the identity used for warm-start matching.
    pub local_a: Vector3,
    pub local_b: Vector3,
    /// Penetration depth; negative for a speculative contact.
    pub depth: f32,
    /// Impulse applied along the normal last step. Non-negative.
    pub normal_impulse: f32,
    /// Impulses along the two friction tangents.
    pub tangent_impulse: [f32; 2],
}

impl ContactPoint {
    /// How hard this contact was pushed last step — useful for impact sounds
    /// and damage thresholds.
    pub fn impulse_magnitude(&self) -> f32 {
        let t = self.tangent_impulse;
        (self.normal_impulse * self.normal_impulse + t[0] * t[0] + t[1] * t[1]).sqrt()
    }
}

/// Contacts between one pair of colliders, all sharing a normal.
#[derive(Debug, Clone, PartialEq)]
pub struct ContactManifold {
    pub key: ManifoldKey,
    /// Unit vector from `b` toward `a`.
    pub normal: Vector3,
    pub points: Vec<ContactPoint>,
    /// Combined friction of the two surfaces.
    pub friction: f32,
    /// Combined restitution of the two surfaces.
    pub restitution: f32,
    /// Either collider is a sensor: report the overlap, apply no forces.
    pub is_sensor: bool,
    /// Whether any point is actually touching, as opposed to merely speculative.
    pub touching: bool,
    alive: bool,
}

impl ContactManifold {
    pub fn body_a(&self) -> BodyId {
        self.key.body_a
    }

    pub fn body_b(&self) -> BodyId {
        self.key.body_b
    }

    /// Total impulse across all points, along the normal.
    pub fn total_normal_impulse(&self) -> f32 {
        self.points.iter().map(|p| p.normal_impulse).sum()
    }

    /// Deepest penetration in this manifold.
    pub fn max_depth(&self) -> f32 {
        self.points
            .iter()
            .map(|p| p.depth)
            .fold(f32::NEG_INFINITY, f32::max)
    }
}

/// A pair starting or stopping contact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollisionEvent {
    pub body_a: BodyId,
    pub body_b: BodyId,
    pub collider_a: u32,
    pub collider_b: u32,
    /// `true` when the pair began touching this step, `false` when it stopped.
    pub started: bool,
    /// The pair involves a sensor, so no forces were applied.
    pub is_sensor: bool,
}

/// How far a contact point may move, in body-local space, and still be
/// recognised as the same contact for warm starting.
const MATCH_DISTANCE: f32 = 0.02;

/// All live manifolds, keyed for reuse across steps.
#[derive(Debug, Default, Clone)]
pub struct ContactSet {
    /// Ordered, not hashed. Sequential impulses are order-dependent, so
    /// solving in `HashMap` order would make the simulation differ from run to
    /// run — Rust randomises that order per process. A replay, a rollback and a
    /// lockstep peer all need this to be the same every time.
    manifolds: BTreeMap<ManifoldKey, ContactManifold>,
    events: Vec<CollisionEvent>,
}

impl ContactSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark every manifold stale. Anything not refreshed by [`Self::update`]
    /// before [`Self::end_step`] is treated as separated.
    pub fn begin_step(&mut self) {
        self.events.clear();
        for m in self.manifolds.values_mut() {
            m.alive = false;
        }
    }

    /// Fold a freshly generated manifold in, carrying impulses over from the
    /// matching contacts of the previous step.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        key: ManifoldKey,
        raw: &RawManifold,
        iso_a: &Isometry,
        iso_b: &Isometry,
        friction: f32,
        restitution: f32,
        is_sensor: bool,
    ) {
        let touching = raw.points.iter().any(|p| p.depth >= 0.0);
        let mut points: Vec<ContactPoint> = raw
            .points
            .iter()
            .map(|p| ContactPoint {
                point_a: p.point_a,
                point_b: p.point_b,
                local_a: iso_a.inverse_transform_point(p.point_a),
                local_b: iso_b.inverse_transform_point(p.point_b),
                depth: p.depth,
                normal_impulse: 0.0,
                tangent_impulse: [0.0; 2],
            })
            .collect();

        let was_touching = match self.manifolds.get(&key) {
            Some(old) => {
                // Carry impulses from the nearest previous contact, if any is
                // near enough to plausibly be the same feature.
                for p in &mut points {
                    let mut best: Option<(f32, &ContactPoint)> = None;
                    for q in &old.points {
                        let d = (p.local_a - q.local_a).length_sq();
                        if d < MATCH_DISTANCE * MATCH_DISTANCE
                            && best.is_none_or(|(bd, _)| d < bd)
                        {
                            best = Some((d, q));
                        }
                    }
                    if let Some((_, q)) = best {
                        p.normal_impulse = q.normal_impulse;
                        p.tangent_impulse = q.tangent_impulse;
                    }
                }
                old.touching
            }
            None => false,
        };

        if touching && !was_touching {
            self.events.push(CollisionEvent {
                body_a: key.body_a,
                body_b: key.body_b,
                collider_a: key.collider_a,
                collider_b: key.collider_b,
                started: true,
                is_sensor,
            });
        } else if !touching && was_touching {
            self.events.push(CollisionEvent {
                body_a: key.body_a,
                body_b: key.body_b,
                collider_a: key.collider_a,
                collider_b: key.collider_b,
                started: false,
                is_sensor,
            });
        }

        self.manifolds.insert(
            key,
            ContactManifold {
                key,
                normal: raw.normal,
                points,
                friction,
                restitution,
                is_sensor,
                touching,
                alive: true,
            },
        );
    }

    /// Drop manifolds that were not refreshed, emitting a stop event for each
    /// that had been touching.
    pub fn end_step(&mut self) {
        let events = &mut self.events;
        self.manifolds.retain(|key, m| {
            if m.alive {
                return true;
            }
            if m.touching {
                events.push(CollisionEvent {
                    body_a: key.body_a,
                    body_b: key.body_b,
                    collider_a: key.collider_a,
                    collider_b: key.collider_b,
                    started: false,
                    is_sensor: m.is_sensor,
                });
            }
            false
        });
    }

    /// Keep manifolds the narrow phase is not going to visit this step.
    ///
    /// A pair of sleeping bodies is skipped by the broad phase, so nothing
    /// refreshes its manifold and [`Self::end_step`] would drop it — reporting
    /// the contact as *ended*, and losing its warm-start impulses, for two
    /// bodies that have not moved a millimetre. A settled stack would announce
    /// that it had come apart and then, on the first nudge, solve itself cold.
    pub fn keep_dormant(&mut self, mut dormant: impl FnMut(&ManifoldKey) -> bool) {
        for (key, m) in self.manifolds.iter_mut() {
            if dormant(key) {
                m.alive = true;
            }
        }
    }

    /// Forget every manifold involving `body` — call when it is removed.
    pub fn remove_body(&mut self, body: BodyId) {
        self.manifolds
            .retain(|k, _| k.body_a != body && k.body_b != body);
    }

    /// Replace every manifold with a restored set — see [`crate::snapshot`].
    ///
    /// Only what the next step reads back matters here: each contact's position
    /// in `a`'s local frame, its impulses, and whether the pair was touching.
    /// The narrow phase regenerates normals, friction and depths before anything
    /// uses them, so those are left at defaults rather than invented.
    pub fn restore(
        &mut self,
        entries: impl IntoIterator<Item = (ManifoldKey, Vec<ContactPoint>, bool)>,
    ) {
        self.manifolds.clear();
        self.events.clear();
        for (key, points, touching) in entries {
            self.manifolds.insert(
                key,
                ContactManifold {
                    key,
                    normal: Vector3::UP,
                    points,
                    friction: 0.0,
                    restitution: 0.0,
                    is_sensor: false,
                    touching,
                    alive: false,
                },
            );
        }
    }

    pub fn clear(&mut self) {
        self.manifolds.clear();
        self.events.clear();
    }

    pub fn len(&self) -> usize {
        self.manifolds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.manifolds.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &ContactManifold> {
        self.manifolds.values()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut ContactManifold> {
        self.manifolds.values_mut()
    }

    /// Contacts started and stopped during the last step.
    pub fn events(&self) -> &[CollisionEvent] {
        &self.events
    }

    /// Every manifold touching `body`.
    pub fn contacts_with(&self, body: BodyId) -> impl Iterator<Item = &ContactManifold> {
        self.manifolds
            .values()
            .filter(move |m| m.key.body_a == body || m.key.body_b == body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::{BodySet, RigidBody};
    use crate::narrowphase::RawPoint;
    use crate::shape::Shape;

    fn ids() -> (BodyId, BodyId) {
        let mut set = BodySet::new();
        let a = set.insert(RigidBody::dynamic().shape(Shape::ball(1.0)));
        let b = set.insert(RigidBody::dynamic().shape(Shape::ball(1.0)));
        (a, b)
    }

    fn key(a: BodyId, b: BodyId) -> ManifoldKey {
        ManifoldKey {
            body_a: a,
            body_b: b,
            collider_a: 0,
            collider_b: 0,
            sub_a: 0,
            sub_b: 0,
        }
    }

    fn raw(point: Vector3, depth: f32) -> RawManifold {
        RawManifold {
            normal: Vector3::UP,
            points: vec![RawPoint {
                point_a: point,
                point_b: point,
                depth,
            }],
            sub_a: 0,
            sub_b: 0,
        }
    }

    #[test]
    fn impulses_carry_over_between_steps() {
        let (a, b) = ids();
        let iso = Isometry::IDENTITY;
        let mut set = ContactSet::new();

        set.begin_step();
        set.update(key(a, b), &raw(Vector3::ZERO, 0.1), &iso, &iso, 0.5, 0.0, false);
        set.end_step();
        // Pretend the solver ran.
        set.iter_mut().next().unwrap().points[0].normal_impulse = 7.5;

        // Next step, the contact barely moved: the impulse should survive.
        set.begin_step();
        set.update(
            key(a, b),
            &raw(Vector3::new(0.001, 0.0, 0.0), 0.1),
            &iso,
            &iso,
            0.5,
            0.0,
            false,
        );
        set.end_step();
        assert_eq!(set.iter().next().unwrap().points[0].normal_impulse, 7.5);
    }

    #[test]
    fn a_contact_that_jumped_across_the_body_does_not_inherit_impulse() {
        let (a, b) = ids();
        let iso = Isometry::IDENTITY;
        let mut set = ContactSet::new();
        set.begin_step();
        set.update(key(a, b), &raw(Vector3::ZERO, 0.1), &iso, &iso, 0.5, 0.0, false);
        set.end_step();
        set.iter_mut().next().unwrap().points[0].normal_impulse = 7.5;

        set.begin_step();
        set.update(
            key(a, b),
            &raw(Vector3::new(5.0, 0.0, 0.0), 0.1),
            &iso,
            &iso,
            0.5,
            0.0,
            false,
        );
        set.end_step();
        assert_eq!(set.iter().next().unwrap().points[0].normal_impulse, 0.0);
    }

    #[test]
    fn start_and_stop_events_fire_exactly_once() {
        let (a, b) = ids();
        let iso = Isometry::IDENTITY;
        let mut set = ContactSet::new();

        // Speculative only — not touching, so no event.
        set.begin_step();
        set.update(key(a, b), &raw(Vector3::ZERO, -0.01), &iso, &iso, 0.5, 0.0, false);
        set.end_step();
        assert!(set.events().is_empty());

        // Now touching.
        set.begin_step();
        set.update(key(a, b), &raw(Vector3::ZERO, 0.05), &iso, &iso, 0.5, 0.0, false);
        set.end_step();
        assert_eq!(set.events().len(), 1);
        assert!(set.events()[0].started);

        // Still touching — no repeat event.
        set.begin_step();
        set.update(key(a, b), &raw(Vector3::ZERO, 0.05), &iso, &iso, 0.5, 0.0, false);
        set.end_step();
        assert!(set.events().is_empty());

        // Separated entirely: the manifold disappears and a stop fires.
        set.begin_step();
        set.end_step();
        assert_eq!(set.events().len(), 1);
        assert!(!set.events()[0].started);
        assert!(set.is_empty());
    }

    #[test]
    fn removing_a_body_drops_its_manifolds() {
        let (a, b) = ids();
        let iso = Isometry::IDENTITY;
        let mut set = ContactSet::new();
        set.begin_step();
        set.update(key(a, b), &raw(Vector3::ZERO, 0.1), &iso, &iso, 0.5, 0.0, false);
        set.end_step();
        assert_eq!(set.len(), 1);
        set.remove_body(a);
        assert!(set.is_empty());
    }

    #[test]
    fn contacts_with_finds_both_sides_of_a_pair() {
        let (a, b) = ids();
        let iso = Isometry::IDENTITY;
        let mut set = ContactSet::new();
        set.begin_step();
        set.update(key(a, b), &raw(Vector3::ZERO, 0.1), &iso, &iso, 0.5, 0.0, false);
        set.end_step();
        assert_eq!(set.contacts_with(a).count(), 1);
        assert_eq!(set.contacts_with(b).count(), 1);
    }

    #[test]
    fn local_points_are_recorded_in_each_bodys_frame() {
        let (a, b) = ids();
        let iso_a = Isometry::from_translation(Vector3::new(10.0, 0.0, 0.0));
        let iso_b = Isometry::from_translation(Vector3::new(0.0, 20.0, 0.0));
        let mut set = ContactSet::new();
        set.begin_step();
        set.update(key(a, b), &raw(Vector3::ZERO, 0.1), &iso_a, &iso_b, 0.5, 0.0, false);
        set.end_step();
        let p = set.iter().next().unwrap().points[0];
        assert!((p.local_a - Vector3::new(-10.0, 0.0, 0.0)).length() < 1e-5);
        assert!((p.local_b - Vector3::new(0.0, -20.0, 0.0)).length() < 1e-5);
    }
}
