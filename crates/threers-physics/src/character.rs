//! A kinematic character controller: move-and-slide against the world.
//!
//! Characters are usually *not* simulated as dynamic bodies. A dynamic capsule
//! tips over, slides down ramps, gets shoved by every passing crate and
//! accelerates like a physical object rather than like a person. Instead this
//! controller sweeps a shape through the world, stops it where it hits
//! something, and slides the leftover motion along the surface — which is what
//! players expect.
//!
//! It is a pure query: it reads the world and returns where the character should
//! end up. Nothing is mutated, so it composes with whatever movement code you
//! already have.
//!
//! ```no_run
//! use threers_physics::prelude::*;
//!
//! let mut world = World::new();
//! let controller = CharacterController::new(Shape::capsule(0.6, 0.3));
//! let mut position = Isometry::from_translation(Vector3::new(0.0, 1.0, 0.0));
//! let mut vertical = 0.0f32;
//! let dt = 1.0 / 60.0;
//!
//! // Per frame: gravity, then walk forward.
//! vertical -= 9.81 * dt;
//! let desired = Vector3::new(0.0, vertical * dt, 3.0 * dt);
//! let result = controller.move_and_slide(&world, position, desired, QueryFilter::default());
//!
//! position.translation = position.translation + result.translation;
//! if result.grounded {
//!     vertical = 0.0;
//! }
//! ```

use crate::math::{try_normalize, Isometry};
use crate::query::QueryFilter;
use crate::shape::Shape;
use crate::world::World;
use threers::math::{Ray, Vector3};

/// What happened during a [`CharacterController::move_and_slide`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharacterMove {
    /// Movement actually achieved. Add this to the character's position.
    pub translation: Vector3,
    /// Standing on something walkable.
    pub grounded: bool,
    /// Normal of the surface underfoot, if grounded.
    pub ground_normal: Option<Vector3>,
    /// Blocked by something too steep to walk up.
    pub hit_wall: bool,
    /// Blocked from above.
    pub hit_ceiling: bool,
}

/// Sweeps a shape through the world and slides it along whatever it hits.
#[derive(Debug, Clone)]
pub struct CharacterController {
    /// The character's collision shape. A capsule is the usual choice: it has no
    /// corners to catch on step edges.
    pub shape: Shape,
    /// Which way is up.
    pub up: Vector3,
    /// Steepest slope that counts as ground rather than wall, in radians.
    pub max_slope: f32,
    /// Gap kept between the character and every surface. Without it, the
    /// character ends each move exactly touching, and the next sweep starts in
    /// contact — where the solve is numerically worst.
    pub skin: f32,
    /// How many times the leftover motion may be redirected in one move. Each
    /// slide costs a shape cast; three handles a corner, four handles a corner
    /// of a corner.
    pub max_slides: usize,
    /// How far to search downward for ground after moving. Lets the character
    /// stay glued walking down a ramp or off a small lip instead of launching
    /// into a fall.
    pub snap_to_ground: f32,
}

impl CharacterController {
    pub fn new(shape: Shape) -> Self {
        Self {
            shape,
            up: Vector3::UP,
            max_slope: std::f32::consts::FRAC_PI_4, // 45 degrees
            skin: 0.02,
            max_slides: 4,
            snap_to_ground: 0.2,
        }
    }

    pub fn max_slope_degrees(mut self, degrees: f32) -> Self {
        self.max_slope = degrees.to_radians();
        self
    }

    pub fn up(mut self, up: Vector3) -> Self {
        self.up = try_normalize(up).unwrap_or(Vector3::UP);
        self
    }

    pub fn skin(mut self, skin: f32) -> Self {
        self.skin = skin.max(0.0);
        self
    }

    pub fn snap_to_ground(mut self, distance: f32) -> Self {
        self.snap_to_ground = distance.max(0.0);
        self
    }

    /// Is a surface with this normal walkable?
    pub fn is_walkable(&self, normal: Vector3) -> bool {
        normal.dot(self.up) >= self.max_slope.cos()
    }

    /// Move by `desired`, stopping and sliding on contact.
    ///
    /// Returns the movement achieved — never more than `desired`.
    pub fn move_and_slide(
        &self,
        world: &World,
        start: Isometry,
        desired: Vector3,
        filter: QueryFilter,
    ) -> CharacterMove {
        let mut position = start;
        let mut remaining = desired;
        let mut result = CharacterMove {
            translation: Vector3::ZERO,
            grounded: false,
            ground_normal: None,
            hit_wall: false,
            hit_ceiling: false,
        };

        for _ in 0..self.max_slides {
            let Some(direction) = try_normalize(remaining) else {
                break;
            };
            let distance = remaining.length();
            if distance < 1e-6 {
                break;
            }

            match world.cast_shape(&self.shape, &position, direction, distance + self.skin, filter) {
                None => {
                    // Clear: take the whole remaining step. Nothing reads
                    // `remaining` past this point, so it is simply consumed.
                    position.translation = position.translation + remaining;
                    break;
                }
                Some(hit) => {
                    // Advance to just short of the surface.
                    let travel = (hit.toi - self.skin).max(0.0).min(distance);
                    position.translation = position.translation + direction * travel;

                    if self.is_walkable(hit.normal) {
                        result.grounded = true;
                        result.ground_normal = Some(hit.normal);
                    } else if hit.normal.dot(self.up) < -0.5 {
                        result.hit_ceiling = true;
                    } else {
                        result.hit_wall = true;
                    }

                    // Project what is left onto the surface plane, so the
                    // character keeps moving *along* the wall instead of
                    // stopping dead against it.
                    let leftover = remaining - direction * travel;
                    remaining = leftover - hit.normal * leftover.dot(hit.normal);
                }
            }
        }

        // Ground check, and a downward snap so walking off a small lip does not
        // start a fall.
        let feet = position.translation;
        if !result.grounded {
            let probe = self.snap_to_ground.max(self.skin * 2.0);
            if let Some(hit) =
                world.cast_shape(&self.shape, &position, -self.up, probe, filter)
            {
                if self.is_walkable(hit.normal) {
                    result.grounded = true;
                    result.ground_normal = Some(hit.normal);
                    // Only snap when already moving downward or level; snapping
                    // during a jump would pin the character to the floor.
                    if desired.dot(self.up) <= 0.0 {
                        position.translation =
                            position.translation - self.up * (hit.toi - self.skin).max(0.0);
                    }
                }
            }
        }
        let _ = feet;

        result.translation = position.translation - start.translation;
        result
    }

    /// Distance to the ground directly below, if any. Handy for footstep
    /// effects and landing animations.
    pub fn ground_distance(
        &self,
        world: &World,
        position: Isometry,
        max_distance: f32,
        filter: QueryFilter,
    ) -> Option<f32> {
        let ray = Ray::new(position.translation, -self.up);
        world.raycast(&ray, max_distance, filter).map(|h| h.toi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::RigidBody;

    fn flat_world() -> World {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()));
        world
    }

    fn capsule_at(y: f32) -> Isometry {
        Isometry::from_translation(Vector3::new(0.0, y, 0.0))
    }

    #[test]
    fn unobstructed_movement_is_taken_in_full() {
        let world = World::new();
        let c = CharacterController::new(Shape::capsule(0.5, 0.3));
        let desired = Vector3::new(1.0, 0.0, 2.0);
        let r = c.move_and_slide(&world, capsule_at(5.0), desired, QueryFilter::default());
        assert!((r.translation - desired).length() < 1e-4, "{:?}", r.translation);
        assert!(!r.grounded);
    }

    #[test]
    fn a_wall_stops_forward_motion_but_allows_sliding_along_it() {
        let mut world = flat_world();
        // A wall across the x axis at x = 2.
        world.add_body(
            RigidBody::fixed()
                .shape(Shape::cuboid(0.1, 5.0, 10.0))
                .translation(Vector3::new(2.0, 0.0, 0.0)),
        );
        let c = CharacterController::new(Shape::capsule(0.5, 0.3));
        let start = capsule_at(0.8);

        // The wall's near face is at x = 1.9 and the capsule's radius is 0.3, so
        // a character that stops when it touches stops at x = 1.6. Anything
        // beyond that is inside the wall.
        let stop = 1.9 - 0.3;

        // Straight into it: stopped by the wall well short of the 3.0 asked for.
        let r = c.move_and_slide(&world, start, Vector3::new(3.0, 0.0, 0.0), QueryFilter::default());
        assert!(r.hit_wall, "should have hit the wall");
        assert!(
            r.translation.x <= stop + 1e-3,
            "went through the wall to {:?}",
            r.translation
        );

        // Diagonally into it: blocked in x, but the z component survives.
        let r = c.move_and_slide(&world, start, Vector3::new(3.0, 0.0, 2.0), QueryFilter::default());
        assert!(r.translation.z > 1.5, "did not slide along the wall: {:?}", r.translation);
        assert!(
            r.translation.x <= stop + 1e-3,
            "went through the wall: {:?}",
            r.translation
        );
    }

    #[test]
    fn standing_on_the_ground_reports_grounded() {
        let world = flat_world();
        let c = CharacterController::new(Shape::capsule(0.5, 0.3));
        // Capsule half-height 0.5 + radius 0.3 = 0.8 tall from centre.
        let r = c.move_and_slide(
            &world,
            capsule_at(0.85),
            Vector3::new(0.0, -0.5, 0.0),
            QueryFilter::default(),
        );
        assert!(r.grounded, "should be standing on the plane");
        assert!(r.ground_normal.unwrap().y > 0.99);
        // And it did not sink through.
        assert!(r.translation.y > -0.2, "sank to {:?}", r.translation);
    }

    #[test]
    fn a_walkable_ramp_is_ground_and_a_steep_one_is_wall() {
        let gentle = CharacterController::new(Shape::capsule(0.5, 0.3)).max_slope_degrees(50.0);
        // A 45 degree surface.
        let ramp = Vector3::new(0.0, 1.0, 1.0).normalize();
        assert!(gentle.is_walkable(ramp));
        assert!(gentle.is_walkable(Vector3::UP));

        let strict = CharacterController::new(Shape::capsule(0.5, 0.3)).max_slope_degrees(30.0);
        assert!(!strict.is_walkable(ramp));
        // A vertical wall is never walkable.
        assert!(!strict.is_walkable(Vector3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn jumping_is_not_cancelled_by_the_ground_snap() {
        let world = flat_world();
        let c = CharacterController::new(Shape::capsule(0.5, 0.3));
        let r = c.move_and_slide(
            &world,
            capsule_at(0.81),
            Vector3::new(0.0, 0.3, 0.0),
            QueryFilter::default(),
        );
        assert!(r.translation.y > 0.25, "the jump was snapped back down: {:?}", r.translation);
    }

    #[test]
    fn a_filter_lets_the_character_pass_through_selected_bodies() {
        let mut world = flat_world();
        let ghost_wall = world.add_body(
            RigidBody::fixed()
                .shape(Shape::cuboid(0.1, 5.0, 10.0))
                .translation(Vector3::new(2.0, 0.0, 0.0)),
        );
        let c = CharacterController::new(Shape::capsule(0.5, 0.3));
        let desired = Vector3::new(3.0, 0.0, 0.0);

        let blocked = c.move_and_slide(&world, capsule_at(0.8), desired, QueryFilter::default());
        assert!(blocked.hit_wall);

        let through = c.move_and_slide(
            &world,
            capsule_at(0.8),
            desired,
            QueryFilter::default().exclude(ghost_wall),
        );
        assert!(through.translation.x > 2.9, "filter did not let it through");
    }

    #[test]
    fn ground_distance_measures_the_drop() {
        let world = flat_world();
        let c = CharacterController::new(Shape::capsule(0.5, 0.3));
        let d = c
            .ground_distance(&world, capsule_at(4.0), 100.0, QueryFilter::default())
            .unwrap();
        assert!((d - 4.0).abs() < 1e-3, "distance = {d}");
        // Nothing below when there is no ground in range.
        let empty = World::new();
        assert!(c
            .ground_distance(&empty, capsule_at(4.0), 100.0, QueryFilter::default())
            .is_none());
    }

    #[test]
    fn a_zero_move_is_a_no_op_that_still_reports_ground() {
        let world = flat_world();
        let c = CharacterController::new(Shape::capsule(0.5, 0.3));
        let r = c.move_and_slide(&world, capsule_at(0.81), Vector3::ZERO, QueryFilter::default());
        assert!(r.translation.length() < 0.05);
        assert!(r.grounded);
    }
}
