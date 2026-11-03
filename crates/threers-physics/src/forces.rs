//! Force fields: drag, buoyancy, wind and explosions.
//!
//! These are applied by you, once per frame, before stepping — they are not part
//! of the step, because what counts as "the air" or "the water" is a property of
//! your scene, not of the solver.
//!
//! All of them accumulate into the body's force buffer, so several can be
//! combined and the world's own gravity still applies on top.

use crate::body::BodyId;
use crate::math::try_normalize;
use crate::query::QueryFilter;
use crate::world::World;
use threers::math::Vector3;

impl World {
    /// Push everything away from a point.
    ///
    /// `impulse` is the strength at the centre, falling off to nothing at
    /// `radius`. The falloff is linear rather than inverse-square: an explosion
    /// is a pressure wave with an edge, not a point source, and `1/r²` mostly
    /// produces one body flung into orbit while its neighbour barely moves.
    ///
    /// Bodies shielded by level geometry are unaffected when `line_of_sight` is
    /// set — the ray is cast against fixed bodies only, so a crate does not
    /// shelter behind another crate.
    ///
    /// ```
    /// use threers_physics::prelude::*;
    ///
    /// let mut world = World::new();
    /// world.add_body(RigidBody::fixed().shape(Shape::ground()));
    /// let crate_body = world.add_body(
    ///     RigidBody::dynamic()
    ///         .shape(Shape::cuboid(0.5, 0.5, 0.5))
    ///         .translation(Vector3::new(2.0, 0.5, 0.0)),
    /// );
    ///
    /// world.apply_explosion(Vector3::new(0.0, 0.5, 0.0), 40.0, 6.0, false);
    /// world.step(1.0 / 60.0);
    /// assert!(world.body(crate_body).unwrap().linear_velocity.x > 0.0);
    /// ```
    pub fn apply_explosion(
        &mut self,
        centre: Vector3,
        impulse: f32,
        radius: f32,
        line_of_sight: bool,
    ) {
        if radius <= 0.0 || impulse == 0.0 {
            return;
        }

        // Gather first: the line-of-sight check needs an immutable borrow.
        let mut blast: Vec<(BodyId, Vector3)> = Vec::new();
        for (id, body) in self.bodies().iter() {
            if !body.is_dynamic() || !body.enabled {
                continue;
            }
            let target = body.world_center_of_mass();
            let offset = target - centre;
            let distance = offset.length();
            if distance > radius {
                continue;
            }
            let Some(direction) = try_normalize(offset) else {
                // Dead centre: no direction to push, so push up rather than
                // producing a NaN.
                blast.push((id, Vector3::UP * impulse));
                continue;
            };
            if line_of_sight {
                let ray = threers::math::Ray::new(centre, direction);
                let filter = QueryFilter::default().only_fixed();
                if let Some(hit) = self.raycast(&ray, distance, filter) {
                    if hit.toi < distance - 0.01 {
                        continue; // something solid is in the way
                    }
                }
            }
            let falloff = 1.0 - distance / radius;
            blast.push((id, direction * (impulse * falloff)));
        }

        for (id, push) in blast {
            if let Some(body) = self.body_mut(id) {
                body.apply_impulse(push);
            }
        }
    }

    /// Quadratic air resistance on every dynamic body.
    ///
    /// `drag` bundles the fluid density, the drag coefficient and a reference
    /// area into one number, because pulling them apart would imply a per-shape
    /// cross-section this engine does not compute. Around `0.05` reads as air;
    /// `2.0` reads as water.
    ///
    /// Force grows with the square of speed, which is what makes a falling body
    /// reach a terminal velocity rather than accelerating forever.
    pub fn apply_drag(&mut self, drag: f32, angular_drag: f32) {
        if drag <= 0.0 && angular_drag <= 0.0 {
            return;
        }
        let dt = self.timestep.max(1e-6);
        for (_, body) in self.bodies_mut().iter_mut() {
            if !body.is_dynamic() || !body.enabled || body.is_sleeping() {
                continue;
            }
            let velocity = body.linear_velocity;
            let speed = velocity.length();
            if drag > 0.0 && speed > 1e-4 {
                // Quadratic drag integrated explicitly can overshoot: at 10 m/s
                // through water a light body is asked for more than its own
                // momentum in one step, and it comes out going backwards. Real
                // drag only ever removes momentum, so the force is capped at
                // whatever brings the body exactly to rest this step.
                let ceiling = body.mass() * speed / dt;
                let magnitude = (drag * speed * speed).min(ceiling);
                body.add_force(velocity * (-magnitude / speed));
            }
            let spin = body.angular_velocity;
            let rate = spin.length();
            if angular_drag > 0.0 && rate > 1e-4 {
                // Same cap about the spin axis, using the inertia the body
                // actually turns with there.
                let axis = spin * (1.0 / rate);
                let inv_inertia = axis.dot(body.world_inv_inertia().mul_vec(axis)).max(1e-12);
                let ceiling = rate / (inv_inertia * dt);
                let magnitude = (angular_drag * rate * rate).min(ceiling);
                body.add_torque(spin * (-magnitude / rate));
            }
        }
    }

    /// A steady wind.
    ///
    /// Force is proportional to the square of the body's speed *relative to the
    /// air*, so a body already moving downwind feels less of it — which is why
    /// this is not the same as simply adding a constant force.
    pub fn apply_wind(&mut self, wind: Vector3, drag: f32) {
        if drag <= 0.0 {
            return;
        }
        for (_, body) in self.bodies_mut().iter_mut() {
            if !body.is_dynamic() || !body.enabled {
                continue;
            }
            let relative = wind - body.linear_velocity;
            let speed = relative.length();
            if speed > 1e-4 {
                body.add_force(relative * (drag * speed));
            }
        }
    }

    /// Float bodies on a level fluid surface.
    ///
    /// Buoyancy is proportional to the *submerged* volume, which is estimated
    /// from how much of each body's bounding box lies below `surface_height`. It
    /// is an approximation — a sphere and its bounding cube displace different
    /// amounts — but it is stable, costs nothing, and the alternative is
    /// clipping every collider against a plane every step.
    ///
    /// `fluid_density` compares against the body's own density: heavier than the
    /// fluid and it sinks, lighter and it floats. `linear_drag` is the water's
    /// resistance, and without some of it a floating body bobs forever.
    pub fn apply_buoyancy(
        &mut self,
        surface_height: f32,
        fluid_density: f32,
        linear_drag: f32,
        angular_drag: f32,
    ) {
        if fluid_density <= 0.0 {
            return;
        }
        let gravity = self.gravity;

        for (_, body) in self.bodies_mut().iter_mut() {
            if !body.is_dynamic() || !body.enabled {
                continue;
            }
            let aabb = body.compute_aabb();
            if aabb.is_empty() {
                continue;
            }
            let size = aabb.size();
            if size.y <= 0.0 {
                continue;
            }
            // Fraction of the body's height below the surface.
            let submerged = ((surface_height - aabb.min.y) / size.y).clamp(0.0, 1.0);
            if submerged <= 0.0 {
                continue;
            }

            // Displaced volume, from the body's own volume rather than its box,
            // so a sphere is not treated as a cube.
            let density = body.effective_density();
            let volume = if density > 0.0 {
                body.mass() / density
            } else {
                size.x * size.y * size.z
            };
            let displaced = volume * submerged;

            // Archimedes: the up-force is the weight of the displaced fluid.
            body.add_force(-gravity * (fluid_density * displaced));

            // Water resists motion far more than air, and scaled by how much of
            // the body is actually in it.
            if linear_drag > 0.0 {
                let v = body.linear_velocity;
                body.add_force(v * (-linear_drag * submerged));
            }
            if angular_drag > 0.0 {
                let w = body.angular_velocity;
                body.add_torque(w * (-angular_drag * submerged));
            }
        }
    }

    /// Pull or push everything toward a point with a constant acceleration.
    ///
    /// Unlike `apply_point_gravity` this does not fall off with
    /// distance, which makes it useful for gameplay effects — tractor beams,
    /// vortices, magnets — where an inverse-square law is unmanageable.
    pub fn apply_attractor(&mut self, centre: Vector3, acceleration: f32, radius: f32) {
        for (_, body) in self.bodies_mut().iter_mut() {
            if !body.is_dynamic() || !body.enabled {
                continue;
            }
            let offset = centre - body.world_center_of_mass();
            if radius > 0.0 && offset.length() > radius {
                continue;
            }
            let Some(direction) = try_normalize(offset) else {
                continue;
            };
            body.add_force(direction * (acceleration * body.mass()));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn an_explosion_pushes_outward_and_falls_off_with_distance() {
        let mut world = World::new().with_gravity(Vector3::ZERO);
        let near = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .translation(Vector3::new(1.0, 0.0, 0.0))
                .can_sleep(false),
        );
        let far = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .translation(Vector3::new(4.0, 0.0, 0.0))
                .can_sleep(false),
        );
        let outside = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .translation(Vector3::new(20.0, 0.0, 0.0))
                .can_sleep(false),
        );

        world.apply_explosion(Vector3::ZERO, 30.0, 5.0, false);
        world.step(1.0 / 60.0);

        let vn = world.body(near).unwrap().linear_velocity.x;
        let vf = world.body(far).unwrap().linear_velocity.x;
        assert!(vn > 0.0 && vf > 0.0, "both should be pushed outward");
        assert!(vn > vf, "the near body should be pushed harder: {vn} vs {vf}");
        assert_eq!(
            world.body(outside).unwrap().linear_velocity.length(),
            0.0,
            "a body beyond the radius should be untouched"
        );
    }

    #[test]
    fn an_explosion_at_a_bodys_exact_centre_does_not_produce_nan() {
        let mut world = World::new().with_gravity(Vector3::ZERO);
        let body = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .can_sleep(false),
        );
        world.apply_explosion(Vector3::ZERO, 10.0, 5.0, false);
        world.step(1.0 / 60.0);
        assert!(world.body(body).unwrap().linear_velocity.length().is_finite());
    }

    #[test]
    fn line_of_sight_shields_bodies_behind_walls() {
        let mut world = World::new().with_gravity(Vector3::ZERO);
        // A thick wall between the blast and the body.
        world.add_body(
            RigidBody::fixed()
                .shape(Shape::cuboid(0.3, 5.0, 5.0))
                .translation(Vector3::new(2.0, 0.0, 0.0)),
        );
        let sheltered = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .translation(Vector3::new(4.0, 0.0, 0.0))
                .can_sleep(false),
        );
        let exposed = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .translation(Vector3::new(0.0, 4.0, 0.0))
                .can_sleep(false),
        );

        world.apply_explosion(Vector3::ZERO, 40.0, 8.0, true);
        world.step(1.0 / 60.0);

        assert_eq!(
            world.body(sheltered).unwrap().linear_velocity.length(),
            0.0,
            "the wall should have blocked the blast"
        );
        assert!(
            world.body(exposed).unwrap().linear_velocity.length() > 0.0,
            "the body in the open should have been hit"
        );
    }

    #[test]
    fn drag_produces_a_terminal_velocity() {
        let mut world = World::new();
        let ball = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .translation(Vector3::new(0.0, 1000.0, 0.0))
                .linear_damping(0.0)
                .can_sleep(false),
        );

        let mut previous = 0.0f32;
        for _ in 0..600 {
            world.apply_drag(0.5, 0.0);
            world.step(1.0 / 60.0);
            previous = world.body(ball).unwrap().linear_velocity.y;
        }
        // Without drag, 10 seconds of falling reaches about -98 m/s.
        assert!(previous > -30.0, "drag did not limit the fall: {previous}");
        assert!(previous < -1.0, "it should still be falling: {previous}");

        // And it settles rather than continuing to accelerate.
        let before = previous;
        for _ in 0..120 {
            world.apply_drag(0.5, 0.0);
            world.step(1.0 / 60.0);
        }
        let after = world.body(ball).unwrap().linear_velocity.y;
        assert!(
            (after - before).abs() < 1.0,
            "terminal velocity not reached: {before} -> {after}"
        );
    }

    #[test]
    fn drag_never_reverses_a_body() {
        let mut world = World::new().with_gravity(Vector3::ZERO);
        let ball = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .linear_velocity(Vector3::new(10.0, 0.0, 0.0))
                .linear_damping(0.0)
                .can_sleep(false),
        );
        for _ in 0..600 {
            world.apply_drag(2.0, 0.0);
            world.step(1.0 / 60.0);
            assert!(
                world.body(ball).unwrap().linear_velocity.x >= -0.01,
                "drag pushed the body backwards"
            );
        }
    }

    #[test]
    fn a_light_body_floats_and_a_heavy_one_sinks() {
        let settle = |density: f32| {
            let mut world = World::new();
            let body = world.add_body(
                RigidBody::dynamic()
                    .shape(Shape::cuboid(0.5, 0.5, 0.5))
                    .translation(Vector3::new(0.0, 3.0, 0.0))
                    .density(density)
                    .linear_damping(0.0)
                    .can_sleep(false),
            );
            for _ in 0..900 {
                // Water surface at y = 0, density 1000.
                world.apply_buoyancy(0.0, 1000.0, 400.0, 50.0);
                world.step(1.0 / 60.0);
            }
            world.body(body).unwrap().translation().y
        };

        let cork = settle(200.0);
        let stone = settle(4000.0);
        assert!(
            cork > -1.0,
            "a body lighter than water should float near the surface, got {cork}"
        );
        assert!(stone < -3.0, "a body denser than water should sink, got {stone}");
    }

    #[test]
    fn buoyancy_leaves_bodies_above_the_surface_alone() {
        let mut world = World::new().with_gravity(Vector3::ZERO);
        let flyer = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .translation(Vector3::new(0.0, 50.0, 0.0))
                .can_sleep(false),
        );
        world.apply_buoyancy(0.0, 1000.0, 10.0, 1.0);
        world.step(1.0 / 60.0);
        assert_eq!(world.body(flyer).unwrap().linear_velocity.length(), 0.0);
    }

    #[test]
    fn wind_pushes_downwind_and_stops_pushing_once_matched() {
        let mut world = World::new().with_gravity(Vector3::ZERO);
        let leaf = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .mass(0.05)
                .linear_damping(0.0)
                .can_sleep(false),
        );
        let wind = Vector3::new(8.0, 0.0, 0.0);
        for _ in 0..900 {
            world.apply_wind(wind, 0.5);
            world.step(1.0 / 60.0);
        }
        let v = world.body(leaf).unwrap().linear_velocity;
        assert!(v.x > 6.0, "the leaf should approach wind speed, got {}", v.x);
        assert!(
            v.x <= 8.1,
            "it must not be blown faster than the wind, got {}",
            v.x
        );
    }

    #[test]
    fn an_attractor_draws_bodies_in() {
        let mut world = World::new().with_gravity(Vector3::ZERO);
        let body = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .translation(Vector3::new(10.0, 0.0, 0.0))
                .linear_damping(0.0)
                .can_sleep(false),
        );
        let outside = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .translation(Vector3::new(60.0, 0.0, 0.0))
                .can_sleep(false),
        );
        for _ in 0..120 {
            world.apply_attractor(Vector3::ZERO, 5.0, 20.0);
            world.step(1.0 / 60.0);
        }
        assert!(world.body(body).unwrap().translation().x < 10.0);
        assert_eq!(
            world.body(outside).unwrap().linear_velocity.length(),
            0.0,
            "a body beyond the radius should be untouched"
        );
    }

    #[test]
    fn fields_compose_and_ignore_nonsense_parameters() {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()));
        let body = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .translation(Vector3::new(0.0, 5.0, 0.0))
                .can_sleep(false),
        );
        for _ in 0..120 {
            world.apply_drag(0.1, 0.05);
            world.apply_wind(Vector3::new(1.0, 0.0, 0.0), 0.05);
            world.apply_buoyancy(-100.0, 1000.0, 1.0, 1.0);
            // Nonsense parameters must be no-ops, not panics.
            world.apply_explosion(Vector3::ZERO, 0.0, 0.0, false);
            world.apply_drag(-1.0, -1.0);
            world.apply_buoyancy(0.0, -5.0, 1.0, 1.0);
            world.step(1.0 / 60.0);
        }
        let p = world.body(body).unwrap().translation();
        assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
        assert!(p.x > 0.0, "the wind should have pushed it downwind");
    }
}
