//! A raycast vehicle: one rigid body, four rays, and tyre forces.
//!
//! # Why a car is not just a box with wheels bolted on
//!
//! Modelling wheels as rigid bodies on hinge joints is the obvious approach and
//! it does not work. A wheel is light and the chassis is heavy, so the mass
//! ratio at the joint is terrible; the contact patch is tiny, so it tunnels at
//! speed; and the whole assembly needs the solver to converge every step or the
//! car sinks into the road. Every engine that ships a working vehicle solves it
//! the same way instead: the car is *one* rigid body, and each wheel is a ray
//! cast downward that reports where the ground is. The suspension is a spring
//! along that ray and the tyre is a friction constraint at its far end.
//!
//! That buys stability and costs realism in exactly the places you can afford
//! it. Wheels have no mass of their own and cannot be knocked off; the car
//! cannot drive over something narrower than the gap between its rays. In
//! exchange it is rock solid at any speed and costs four raycasts a frame.
//!
//! ```no_run
//! use threers_physics::prelude::*;
//!
//! let mut world = World::new();
//! world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(1.0));
//!
//! let chassis = world.add_body(
//!     RigidBody::dynamic()
//!         .shape(Shape::cuboid(0.9, 0.4, 2.0))
//!         .mass(1200.0)
//!         .translation(Vector3::new(0.0, 1.0, 0.0)),
//! );
//! let mut car = Vehicle::new(chassis);
//! for (x, z, steers) in [(-0.8, 1.4, true), (0.8, 1.4, true), (-0.8, -1.4, false), (0.8, -1.4, false)] {
//!     car.add_wheel(
//!         WheelConfig::new(Vector3::new(x, -0.2, z), 0.35)
//!             .steering(steers)
//!             .powered(!steers),
//!     );
//! }
//!
//! car.set_steering(0.3);      // radians
//! car.set_drive(600.0);       // newton-metres at the axle
//! car.update(&mut world, 1.0 / 60.0);
//! ```
//!
//! Call [`Vehicle::update`] once per step, before [`World::step`]. It reads the
//! world, works out what each tyre is doing, and applies impulses to the
//! chassis; the solver then handles everything else — including collisions with
//! walls, which the vehicle code knows nothing about.

use crate::body::BodyId;
use crate::math::{try_normalize, Isometry};
use crate::query::QueryFilter;
use crate::world::World;
use threers::math::{Quaternion, Ray, Vector3};

/// How one wheel is mounted and how it behaves.
///
/// Distances are in world units and forces in newtons, so the numbers scale
/// with the chassis mass you set — a 1200 kg car wants roughly the defaults, a
/// go-kart wants a tenth of the stiffness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheelConfig {
    /// Where the suspension is bolted to the chassis, in chassis local space.
    pub attachment: Vector3,
    /// Which way the suspension pushes, in chassis local space. Down, normally.
    pub direction: Vector3,
    /// The axle, in chassis local space. Used to orient the wheel for drawing
    /// and to work out which way is "sideways" for the tyre.
    pub axle: Vector3,
    pub radius: f32,
    /// Suspension length with no load on it.
    pub rest_length: f32,
    /// Spring rate, newtons per unit of compression.
    pub stiffness: f32,
    /// Damping, newtons per unit of compression *speed*. Asymmetric on purpose:
    /// real dampers resist extension harder than compression, which is what
    /// stops a car pogoing after a bump.
    pub compression_damping: f32,
    pub rebound_damping: f32,
    /// Ceiling on the suspension force, so a wheel that bottoms out cannot
    /// launch the car.
    pub max_force: f32,
    /// How far the suspension may compress past `rest_length`.
    pub max_travel: f32,
    /// Tyre grip, combined with the ground's own friction.
    pub friction: f32,
    /// Sideways grip as a fraction of forward grip. Below 1 the car slides in
    /// corners; that is what makes it feel like a car rather than a train.
    pub lateral_grip: f32,
    /// Tyre and bearing losses, as a fraction of the load on the wheel. Around
    /// `0.015` for a road tyre on asphalt.
    ///
    /// Small, and not optional: a free-rolling wheel has nothing else opposing
    /// it along its own axis, so without this a car keeps whatever creep the
    /// suspension gave it while settling — forever, on flat ground, with the
    /// engine off.
    pub rolling_resistance: f32,
    /// Whether [`Vehicle::set_steering`] turns this wheel.
    pub steers: bool,
    /// Whether [`Vehicle::set_drive`] drives this wheel.
    pub powered: bool,
}

impl WheelConfig {
    /// A wheel at `attachment` of the given radius, with defaults sized for a
    /// road car of about a tonne.
    pub fn new(attachment: Vector3, radius: f32) -> Self {
        Self {
            attachment,
            direction: Vector3::new(0.0, -1.0, 0.0),
            axle: Vector3::new(1.0, 0.0, 0.0),
            radius: radius.max(1e-3),
            rest_length: 0.4,
            stiffness: 35_000.0,
            compression_damping: 2_500.0,
            rebound_damping: 4_500.0,
            max_force: 60_000.0,
            max_travel: 0.3,
            friction: 1.5,
            lateral_grip: 0.9,
            rolling_resistance: 0.015,
            steers: false,
            powered: false,
        }
    }

    pub fn steering(mut self, steers: bool) -> Self {
        self.steers = steers;
        self
    }

    pub fn powered(mut self, powered: bool) -> Self {
        self.powered = powered;
        self
    }

    /// Spring rate and damping in one call.
    pub fn suspension(mut self, rest_length: f32, stiffness: f32, damping: f32) -> Self {
        self.rest_length = rest_length.max(0.0);
        self.stiffness = stiffness.max(0.0);
        self.compression_damping = damping.max(0.0);
        self.rebound_damping = (damping * 1.8).max(0.0);
        self
    }

    pub fn grip(mut self, friction: f32, lateral: f32) -> Self {
        self.friction = friction.max(0.0);
        self.lateral_grip = lateral.clamp(0.0, 1.0);
        self
    }
}

/// What a wheel found under it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WheelContact {
    pub body: BodyId,
    /// Where the tyre meets the ground, in world space.
    pub point: Vector3,
    pub normal: Vector3,
    /// Distance from the attachment to the contact.
    pub distance: f32,
}

/// One wheel: its setup, its inputs, and what it did last update.
#[derive(Debug, Clone, PartialEq)]
pub struct Wheel {
    pub config: WheelConfig,
    /// Steering angle in radians, set by [`Vehicle::set_steering`].
    pub steering: f32,
    /// Torque at the axle, in newton-metres.
    pub drive_torque: f32,
    /// Braking torque, in newton-metres. Always opposes motion.
    pub brake_torque: f32,

    /// Current suspension length. Equal to `rest_length` in the air.
    pub suspension_length: f32,
    /// Force the spring pushed with last update, in newtons.
    pub suspension_force: f32,
    /// Ground under the wheel, or `None` if it is airborne.
    pub contact: Option<WheelContact>,
    /// Spin angle, accumulated for drawing.
    pub spin: f32,
    /// Spin rate in rad/s, derived from how fast the contact patch is moving.
    pub spin_rate: f32,
    /// How much the tyre is sliding rather than gripping, 0 to 1. Useful for
    /// skid marks, tyre squeal and traction control.
    pub slip: f32,
}

impl Wheel {
    fn new(config: WheelConfig) -> Self {
        Self {
            suspension_length: config.rest_length,
            config,
            steering: 0.0,
            drive_torque: 0.0,
            brake_torque: 0.0,
            suspension_force: 0.0,
            contact: None,
            spin: 0.0,
            spin_rate: 0.0,
            slip: 0.0,
        }
    }

    /// Whether the tyre is on the ground.
    pub fn is_grounded(&self) -> bool {
        self.contact.is_some()
    }

    /// How far the suspension is compressed from rest.
    pub fn compression(&self) -> f32 {
        (self.config.rest_length - self.suspension_length).max(0.0)
    }
}

/// A car, a truck, or anything else that rolls.
#[derive(Debug, Clone, PartialEq)]
pub struct Vehicle {
    /// The chassis. Everything is applied to this one body.
    pub body: BodyId,
    pub wheels: Vec<Wheel>,
    /// Downforce as a fraction of weight per (unit/s)². Zero for a road car.
    pub downforce: f32,
}

impl Vehicle {
    pub fn new(body: BodyId) -> Self {
        Self {
            body,
            wheels: Vec::new(),
            downforce: 0.0,
        }
    }

    /// Add a wheel, returning its index.
    pub fn add_wheel(&mut self, config: WheelConfig) -> usize {
        self.wheels.push(Wheel::new(config));
        self.wheels.len() - 1
    }

    /// Steer every wheel marked [`WheelConfig::steers`].
    pub fn set_steering(&mut self, radians: f32) {
        for wheel in &mut self.wheels {
            if wheel.config.steers {
                wheel.steering = radians;
            }
        }
    }

    /// Apply drive torque, split evenly between the powered wheels.
    ///
    /// Splitting rather than repeating matters: giving each of four wheels the
    /// full engine torque makes a four-wheel-drive car twice as quick as a
    /// rear-wheel-drive one for the same engine, which is not how cars work.
    pub fn set_drive(&mut self, total_torque: f32) {
        let powered = self.wheels.iter().filter(|w| w.config.powered).count();
        let each = if powered == 0 {
            0.0
        } else {
            total_torque / powered as f32
        };
        for wheel in &mut self.wheels {
            wheel.drive_torque = if wheel.config.powered { each } else { 0.0 };
        }
    }

    /// Brake every wheel.
    pub fn set_brake(&mut self, torque: f32) {
        let torque = torque.max(0.0);
        for wheel in &mut self.wheels {
            wheel.brake_torque = torque;
        }
    }

    /// Brake only the rear — the wheels that are not steering.
    pub fn set_handbrake(&mut self, torque: f32) {
        let torque = torque.max(0.0);
        for wheel in &mut self.wheels {
            if !wheel.config.steers {
                wheel.brake_torque = torque;
            }
        }
    }

    /// Forward speed in world units per second. Negative in reverse.
    pub fn speed(&self, world: &World) -> f32 {
        let Some(body) = world.body(self.body) else {
            return 0.0;
        };
        let forward = forward_axis(&body.position);
        body.linear_velocity.dot(forward)
    }

    /// Where to draw wheel `index`, including steering and spin.
    ///
    /// Returns `None` for an unknown index or a missing chassis.
    pub fn wheel_transform(&self, world: &World, index: usize) -> Option<Isometry> {
        let wheel = self.wheels.get(index)?;
        let body = world.body(self.body)?;
        let chassis = &body.position;
        let down = try_normalize(chassis.transform_vector(wheel.config.direction))?;
        let centre = chassis.transform_point(wheel.config.attachment) + down * wheel.suspension_length;

        let up = down * -1.0;
        let steer = Quaternion::from_axis_angle(up, wheel.steering);
        let axle = chassis.transform_vector(wheel.config.axle);
        let spin = Quaternion::from_axis_angle(try_normalize(axle)?, wheel.spin);
        Some(Isometry::new(
            centre,
            steer.multiply(chassis.rotation).multiply(spin),
        ))
    }

    /// How many wheels are touching the ground.
    pub fn wheels_on_ground(&self) -> usize {
        self.wheels.iter().filter(|w| w.is_grounded()).count()
    }

    /// Step the vehicle. Call once per frame, before [`World::step`].
    ///
    /// This casts a ray per wheel, works out the suspension and tyre forces, and
    /// applies them to the chassis as impulses. It does not move the chassis
    /// itself — the solver does that, which is why a car built this way still
    /// collides with walls and other cars for free.
    pub fn update(&mut self, world: &mut World, dt: f32) {
        if dt <= 0.0 || !dt.is_finite() {
            return;
        }
        let Some(body) = world.body(self.body) else {
            return;
        };
        let chassis = body.position;
        let mass = body.mass();
        let gravity = world.gravity;

        // Cast every wheel first, so the tyre forces all see the same chassis
        // pose. Interleaving casts with impulses would make wheel 0 push the car
        // before wheel 3 has looked at the ground, and the car would list toward
        // whichever wheel happened to be first in the list.
        let filter = QueryFilter::default().exclude(self.body);
        for wheel in &mut self.wheels {
            let origin = chassis.transform_point(wheel.config.attachment);
            let Some(down) = try_normalize(chassis.transform_vector(wheel.config.direction)) else {
                continue;
            };
            let reach = wheel.config.rest_length + wheel.config.radius;
            let ray = Ray::new(origin, down);
            match world.raycast(&ray, reach, filter) {
                Some(hit) => {
                    // The suspension holds the *axle*, so the length is to the
                    // hub, not to the ground: one radius shorter than the ray.
                    let length =
                        (hit.toi - wheel.config.radius).clamp(
                            wheel.config.rest_length - wheel.config.max_travel,
                            wheel.config.rest_length,
                        );
                    wheel.suspension_length = length;
                    wheel.contact = Some(WheelContact {
                        body: hit.body,
                        point: hit.point,
                        // A normal pointing the same way as the ray means the
                        // ray hit a back face; flip it so "up" is always up.
                        normal: if hit.normal.dot(down) > 0.0 {
                            hit.normal * -1.0
                        } else {
                            hit.normal
                        },
                        distance: hit.toi,
                    });
                }
                None => {
                    // Airborne: the suspension extends to rest and does nothing.
                    wheel.suspension_length = wheel.config.rest_length;
                    wheel.contact = None;
                    wheel.suspension_force = 0.0;
                    wheel.slip = 0.0;
                }
            }
        }

        // Aerodynamic downforce, which is what keeps a fast car on the road.
        if self.downforce > 0.0 {
            let speed = self.speed(world);
            let up = try_normalize(chassis.transform_vector(Vector3::new(0.0, 1.0, 0.0)));
            if let (Some(up), Some(b)) = (up, world.body_mut(self.body)) {
                let force = self.downforce * speed * speed * mass * gravity.length();
                b.apply_impulse(up * (-force * dt));
            }
        }

        // Every grounded wheel is cancelling the *same* rigid-body velocity, so
        // each may only ask for its share of it. Without this they each demand
        // the whole correction and the car is pulled several times too hard,
        // overshoots, and crabs sideways across flat ground for ever.
        let grounded = self
            .wheels
            .iter()
            .filter(|w| w.contact.is_some())
            .count()
            .max(1);
        // One snapshot for all four wheels, taken before any of them push.
        //
        // Reading the body live means each wheel measures its slip against a
        // chassis the wheels before it have already shoved: the suspension
        // impulses alone swing the roll rate by a tenth of a radian per second
        // between the first wheel and the second, which is larger than the slip
        // being measured. The tyres then spend their grip fighting each other's
        // transients instead of the car's motion, and a parked car crabs.
        // Real tyres act at the same instant; so should these.
        let snapshot = world.body(self.body).map(|b| {
            (
                b.linear_velocity,
                b.angular_velocity,
                b.world_center_of_mass(),
            )
        });
        for index in 0..self.wheels.len() {
            self.apply_wheel_forces(world, index, dt, mass, grounded, snapshot);
        }
    }

    fn apply_wheel_forces(
        &mut self,
        world: &mut World,
        index: usize,
        dt: f32,
        mass: f32,
        grounded: usize,
        snapshot: Option<(Vector3, Vector3, Vector3)>,
    ) {
        let wheel = &mut self.wheels[index];
        let Some(contact) = wheel.contact else {
            // In the air the wheel keeps spinning, slowed only by the brake, so
            // that landing on the brakes locks it.
            let decay = (wheel.brake_torque / (mass.max(1e-3) * wheel.config.radius)).max(0.0);
            let drop = (decay * dt).min(wheel.spin_rate.abs());
            wheel.spin_rate -= drop * wheel.spin_rate.signum();
            wheel.spin += wheel.spin_rate * dt;
            return;
        };
        let Some(body) = world.body(self.body) else {
            return;
        };
        let chassis = body.position;
        let Some(down) = try_normalize(chassis.transform_vector(wheel.config.direction)) else {
            return;
        };

        // ---- suspension ---------------------------------------------------
        let compression = wheel.config.rest_length - wheel.suspension_length;
        // From the pre-update snapshot, not the live body — see `update`.
        let velocity = match snapshot {
            Some((v, w, com)) => v + w.cross(contact.point - com),
            None => body.velocity_at_point(contact.point),
        };
        // Closing speed along the suspension. Positive means compressing.
        let closing = -velocity.dot(down * -1.0);
        let damping = if closing > 0.0 {
            wheel.config.compression_damping
        } else {
            wheel.config.rebound_damping
        };
        // The spring acts along the suspension, but only the part of it along
        // the contact normal actually pushes against the ground. On a slope
        // that projection is what stops the car being shoved sideways by its
        // own suspension.
        let normal_projection = (down * -1.0).dot(contact.normal).max(0.0);
        let spring = wheel.config.stiffness * compression + damping * closing;
        let force = (spring * normal_projection).clamp(0.0, wheel.config.max_force);
        wheel.suspension_force = force;

        // ---- tyre ---------------------------------------------------------
        // Build the contact frame: forward is the wheel's heading flattened
        // onto the ground, sideways is across it.
        let up = down * -1.0;
        let steer = Quaternion::from_axis_angle(up, wheel.steering);
        let heading = forward_axis(&chassis).apply_quaternion(steer);
        let normal = contact.normal;
        let Some(forward) = try_normalize(heading - normal * heading.dot(normal)) else {
            return;
        };
        let side = normal.cross(forward);

        let vf = velocity.dot(forward);
        let vs = velocity.dot(side);

        // How much force it actually takes to change the contact point's speed,
        // accounting for where on the chassis the wheel is bolted.
        //
        // Using the whole chassis mass here instead is the tempting shortcut and
        // it is wrong twice over: it ignores that a force at the corner spends
        // some of itself rotating the car, and — because every wheel does the
        // same sum — four wheels each cancel the full sideways velocity, so the
        // car is corrected four times over and creeps sideways forever.
        let arm = contact.point - body.world_center_of_mass();
        let inv_inertia = body.world_inv_inertia();
        let effective_mass = |direction: Vector3| {
            let cross = arm.cross(direction);
            let angular = inv_inertia.mul_vec(cross).cross(arm).dot(direction);
            let k = body.inv_mass() + angular;
            if k > 1e-9 {
                1.0 / k
            } else {
                0.0
            }
        };
        // Split between the wheels sharing the load — see `update`.
        let inv_dt = 1.0 / (dt.max(1e-6) * grounded as f32);

        // Coulomb: the tyre can only ever push as hard as it is pressed down.
        let grip = wheel.config.friction * force;
        let drive = wheel.drive_torque / wheel.config.radius;
        // Braking opposes whatever the wheel is doing, and cannot reverse it —
        // a brake that pushes the car backwards is an engine.
        let stopping = vf.abs() * effective_mass(forward) * inv_dt;
        let brake_capacity = wheel.brake_torque / wheel.config.radius;
        let brake = (-vf.signum() * brake_capacity).clamp(-stopping, stopping);
        // Rolling resistance opposes motion and, like the brake, may only ever
        // bring the wheel to rest rather than drag it backwards.
        let rolling = if vf.abs() > 1e-4 {
            (-vf.signum() * wheel.config.rolling_resistance * force).clamp(-stopping, stopping)
        } else {
            0.0
        };
        let wanted_forward = drive + brake + rolling;

        // Lateral force is a constraint, not a drive: whatever it takes to stop
        // the tyre sliding sideways, up to the grip available.
        let wanted_side = -vs * effective_mass(side) * inv_dt;
        let side_limit = grip * wheel.config.lateral_grip;

        // The friction circle: forward and sideways draw on the same budget, so
        // a tyre already at its limit cornering has nothing left for braking.
        // Skipping this is what makes a car feel like it is on rails.
        let clamped_forward = wanted_forward.clamp(-grip, grip);
        let clamped_side = wanted_side.clamp(-side_limit, side_limit);
        let magnitude = (clamped_forward * clamped_forward + clamped_side * clamped_side).sqrt();
        let (fx, fz) = if magnitude > grip && magnitude > 0.0 {
            let scale = grip / magnitude;
            (clamped_forward * scale, clamped_side * scale)
        } else {
            (clamped_forward, clamped_side)
        };

        wheel.slip = if grip > 0.0 {
            let demanded =
                (wanted_forward * wanted_forward + wanted_side * wanted_side).sqrt();
            ((demanded - grip) / grip).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // The spin rate follows the contact patch, plus whatever the tyre is
        // slipping by. It is for drawing and for tyre-squeal cues; nothing in
        // the simulation reads it back.
        wheel.spin_rate = vf / wheel.config.radius + wheel.slip * drive.signum() * 10.0;
        wheel.spin += wheel.spin_rate * dt;

        let impulse = (normal * force + forward * fx + side * fz) * dt;
        if let Some(b) = world.body_mut(self.body) {
            b.wake_up();
            b.apply_impulse_at_point(impulse, contact.point);
        }
        // Newton's third law: push back on whatever the wheel is standing on, so
        // a car parked on a raft sinks it and a car driving off a plank tips it.
        if let Some(other) = world.body_mut(contact.body) {
            if other.is_dynamic() {
                other.wake_up();
                other.apply_impulse_at_point(impulse * -1.0, contact.point);
            }
        }
    }
}

/// The chassis' local -Z, which is the direction it faces.
///
/// Matching three.js: an object with no rotation looks down -Z, so a car built
/// facing that way needs no extra transform to line up with its mesh.
fn forward_axis(iso: &Isometry) -> Vector3 {
    iso.transform_vector(Vector3::new(0.0, 0.0, -1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::RigidBody;
    use crate::shape::Shape;

    const DT: f32 = 1.0 / 60.0;

    fn car_world() -> (World, Vehicle) {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(1.0));
        let chassis = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.9, 0.4, 2.0))
                .mass(1200.0)
                .translation(Vector3::new(0.0, 1.0, 0.0))
                .can_sleep(false),
        );
        let mut car = Vehicle::new(chassis);
        for (x, z, front) in [
            (-0.8f32, 1.4f32, true),
            (0.8, 1.4, true),
            (-0.8, -1.4, false),
            (0.8, -1.4, false),
        ] {
            car.add_wheel(
                WheelConfig::new(Vector3::new(x, -0.2, z), 0.35)
                    .steering(front)
                    .powered(!front),
            );
        }
        (world, car)
    }

    fn drive(world: &mut World, car: &mut Vehicle, steps: usize) {
        for _ in 0..steps {
            car.update(world, DT);
            world.step(DT);
        }
    }

    #[test]
    fn the_suspension_holds_the_car_up() {
        let (mut world, mut car) = car_world();
        drive(&mut world, &mut car, 180);
        let y = world.body(car.body).unwrap().translation().y;
        // Resting height: wheel centre at radius above the ground, plus however
        // far the loaded suspension sits below the attachment.
        assert!(
            (0.4..0.9).contains(&y),
            "the car settled at y = {y}, which is either sunk or floating"
        );
        assert_eq!(car.wheels_on_ground(), 4, "not every wheel found the ground");
    }

    #[test]
    fn a_parked_car_stays_parked() {
        let (mut world, mut car) = car_world();
        drive(&mut world, &mut car, 240);
        let before = world.body(car.body).unwrap().translation();
        drive(&mut world, &mut car, 240);
        let after = world.body(car.body).unwrap().translation();
        assert!(
            (after - before).length() < 0.05,
            "a car with no throttle crept from {before:?} to {after:?}"
        );
    }

    #[test]
    fn throttle_moves_it_and_the_brake_stops_it() {
        let (mut world, mut car) = car_world();
        drive(&mut world, &mut car, 120);

        car.set_drive(4000.0);
        drive(&mut world, &mut car, 180);
        let cruising = car.speed(&world);
        assert!(cruising > 2.0, "three seconds of throttle reached {cruising} units/s");

        car.set_drive(0.0);
        car.set_brake(6000.0);
        drive(&mut world, &mut car, 180);
        let stopped = car.speed(&world);
        assert!(
            stopped.abs() < 0.5,
            "still doing {stopped} units/s after three seconds of braking"
        );
    }

    #[test]
    fn the_brake_never_drives_the_car_backwards() {
        // A brake implemented as a force opposing velocity will happily reverse
        // a slow car, which reads as the handbrake being a reverse gear.
        let (mut world, mut car) = car_world();
        drive(&mut world, &mut car, 120);
        car.set_brake(50_000.0);
        drive(&mut world, &mut car, 300);
        let speed = car.speed(&world);
        assert!(
            speed.abs() < 0.2,
            "a huge brake force pushed the car to {speed} units/s"
        );
    }

    #[test]
    fn steering_turns_it() {
        let (mut world, mut car) = car_world();
        drive(&mut world, &mut car, 120);
        let start = world.body(car.body).unwrap().translation();

        car.set_drive(4000.0);
        car.set_steering(0.35);
        drive(&mut world, &mut car, 300);

        let end = world.body(car.body).unwrap().translation();
        let travelled = end - start;
        assert!(travelled.length() > 3.0, "the car barely moved: {travelled:?}");
        // It set off down -Z; turning must give it some x as well.
        assert!(
            travelled.x.abs() > 0.5,
            "the car went straight despite full lock: {travelled:?}"
        );
    }

    #[test]
    fn steering_the_other_way_turns_the_other_way() {
        let mut ends = Vec::new();
        for lock in [-0.35f32, 0.35] {
            let (mut world, mut car) = car_world();
            drive(&mut world, &mut car, 120);
            car.set_drive(4000.0);
            car.set_steering(lock);
            drive(&mut world, &mut car, 300);
            ends.push(world.body(car.body).unwrap().translation().x);
        }
        assert!(
            ends[0].signum() != ends[1].signum(),
            "both lock directions went the same way: {ends:?}"
        );
    }

    #[test]
    fn the_tyres_run_out_of_grip() {
        // The friction circle is what stops a car cornering at any speed. With
        // grip turned right down it has to slide.
        let (mut world, mut car) = car_world();
        for wheel in &mut car.wheels {
            wheel.config = wheel.config.grip(0.15, 0.9);
        }
        drive(&mut world, &mut car, 120);
        car.set_drive(9000.0);
        car.set_steering(0.4);
        drive(&mut world, &mut car, 240);
        let slipping = car.wheels.iter().any(|w| w.slip > 0.05);
        assert!(slipping, "no wheel reported slip: {:?}", car.wheels.iter().map(|w| w.slip).collect::<Vec<_>>());
    }

    #[test]
    fn a_car_going_off_a_ledge_loses_its_wheels() {
        let mut world = World::new();
        // A short platform, then nothing.
        world.add_body(
            RigidBody::fixed()
                .shape(Shape::cuboid(4.0, 0.5, 4.0))
                .translation(Vector3::new(0.0, -0.5, 0.0))
                .friction(1.0),
        );
        let chassis = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.9, 0.4, 2.0))
                .mass(1200.0)
                .translation(Vector3::new(0.0, 1.0, 0.0))
                .can_sleep(false),
        );
        let mut car = Vehicle::new(chassis);
        for (x, z) in [(-0.8f32, 1.4f32), (0.8, 1.4), (-0.8, -1.4), (0.8, -1.4)] {
            car.add_wheel(WheelConfig::new(Vector3::new(x, -0.2, z), 0.35).powered(true));
        }
        drive(&mut world, &mut car, 120);
        assert_eq!(car.wheels_on_ground(), 4);

        car.set_drive(9000.0);
        drive(&mut world, &mut car, 240);
        assert_eq!(
            car.wheels_on_ground(),
            0,
            "the car drove off the platform but its wheels still found ground"
        );
        assert!(
            world.body(car.body).unwrap().translation().y < 0.0,
            "it drove off the edge and did not fall"
        );
    }

    /// What `plank` presses into `ground` with, summed over its contacts.
    fn ground_load(world: &World, plank: BodyId, ground: BodyId) -> f32 {
        world
            .contacts()
            .iter()
            .filter(|m| {
                let (a, b) = (m.body_a(), m.body_b());
                (a == plank && b == ground) || (a == ground && b == plank)
            })
            .map(|m| m.total_normal_impulse())
            .sum()
    }

    #[test]
    fn the_car_pushes_back_on_what_it_drives_over() {
        // Newton's third law. Without it a car can drive across a floating raft
        // without disturbing it.
        //
        // Measured as force, not displacement. A rigid resting contact stops the
        // approach *at* the surface rather than compressing, so a plank already
        // touching the ground does not visibly sink however much is stacked on
        // it — the load shows up in what it presses down with.
        let build = |with_car: bool| {
            let mut world = World::new();
            let ground = world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(1.0));
            let plank = world.add_body(
                RigidBody::dynamic()
                    .shape(Shape::cuboid(3.0, 0.1, 3.0))
                    .mass(40.0)
                    .translation(Vector3::new(0.0, 0.1, 0.0))
                    .can_sleep(false),
            );
            let mut car = None;
            if with_car {
                let chassis = world.add_body(
                    RigidBody::dynamic()
                        .shape(Shape::cuboid(0.9, 0.4, 2.0))
                        .mass(1200.0)
                        .translation(Vector3::new(0.0, 1.2, 0.0))
                        .can_sleep(false),
                );
                let mut v = Vehicle::new(chassis);
                for (x, z) in [(-0.8f32, 1.4f32), (0.8, 1.4), (-0.8, -1.4), (0.8, -1.4)] {
                    v.add_wheel(WheelConfig::new(Vector3::new(x, -0.2, z), 0.35));
                }
                car = Some(v);
            }
            for _ in 0..200 {
                if let Some(v) = car.as_mut() {
                    v.update(&mut world, DT);
                }
                world.step(DT);
            }
            ground_load(&world, plank, ground)
        };

        let bare = build(false);
        let loaded = build(true);
        // 1200 kg of car on a 40 kg plank is thirty times the load.
        assert!(
            loaded > bare * 5.0,
            "the car's weight never reached the plank: bare {bare}, loaded {loaded}"
        );
    }

    #[test]
    fn torque_is_split_between_the_driven_wheels() {
        let (_, mut car) = car_world();
        car.set_drive(4000.0);
        let driven: Vec<f32> = car
            .wheels
            .iter()
            .filter(|w| w.config.powered)
            .map(|w| w.drive_torque)
            .collect();
        assert_eq!(driven.len(), 2);
        assert!(driven.iter().all(|t| (t - 2000.0).abs() < 1e-3), "{driven:?}");
        assert!(
            car.wheels.iter().filter(|w| !w.config.powered).all(|w| w.drive_torque == 0.0),
            "an undriven wheel got torque"
        );
    }

    #[test]
    fn a_wheel_transform_follows_the_suspension() {
        let (mut world, mut car) = car_world();
        let airborne = car.wheel_transform(&world, 0).unwrap().translation;
        drive(&mut world, &mut car, 180);
        let loaded = car.wheel_transform(&world, 0).unwrap().translation;
        assert!(
            car.wheels[0].compression() > 0.0,
            "the suspension never compressed"
        );
        assert!(
            loaded.y != airborne.y,
            "the wheel did not move with the suspension"
        );
        assert!(car.wheel_transform(&world, 99).is_none());
    }

    #[test]
    fn a_vehicle_without_wheels_does_nothing_rather_than_panicking() {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()));
        let chassis = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.9, 0.4, 2.0))
                .translation(Vector3::new(0.0, 1.0, 0.0)),
        );
        let mut car = Vehicle::new(chassis);
        car.set_drive(1000.0);
        car.set_steering(0.5);
        drive(&mut world, &mut car, 60);
        assert_eq!(car.wheels_on_ground(), 0);
        assert_eq!(car.speed(&world), 0.0, "a wheelless chassis drove itself");
    }

    #[test]
    fn a_removed_chassis_is_survivable() {
        let (mut world, mut car) = car_world();
        drive(&mut world, &mut car, 60);
        world.remove_body(car.body);
        car.update(&mut world, DT); // must not panic
        assert_eq!(car.speed(&world), 0.0);
    }
}


