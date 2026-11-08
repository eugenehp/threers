//! Cars: a [`Vehicle`] as a component, driven by plain inputs.
//!
//! Spawn the body, then [`attach_car`](AttachCar::attach_car) or
//! [`Spawned::car`](crate::Spawned::car). Each frame, set [`Car::throttle`],
//! [`Car::brake`] and [`Car::steer`]; [`PhysicsPlugin`](crate::physics::PhysicsPlugin)
//! does the rest, and draws the wheels, which are children of the car and go when
//! it does.
//!
//! ```
//! # use threers::prelude::*;
//! # use threers_ecs::prelude::*;
//! # use threers_ecs::car::{car_step, Car, CarSpec};
//! # use threers_ecs::physics::*;
//! fn drive(w: &mut World) {
//!     let throttle = w.resource::<Input>().axis("accelerate");
//!     w.each::<&mut Car>(|car| car.throttle = throttle);
//! }
//!
//! let mut app = App::new();
//! app.plugin(PhysicsPlugin::default());
//! app.fixed_system(System::new(drive).before(car_step));
//! app.world.resource_mut::<Input>().bind_axis("accelerate", Key::KeyS, Key::KeyW);
//! let w = &mut app.world;
//! w.mesh(BoxGeometry::new(40.0, 1.0, 40.0), StandardMaterial::new(Color::WHITE))
//!     .at(Vector3::new(0.0, -0.5, 0.0))
//!     .body(RigidBody::fixed().shape(Shape::cuboid(20.0, 0.5, 20.0)));
//! w.mesh(BoxGeometry::new(1.8, 0.7, 4.0), StandardMaterial::new(Color::WHITE))
//!     .at(Vector3::new(0.0, 1.0, 0.0))
//!     .car(CarSpec::default());
//! app.frame(1.0 / 60.0);
//! ```

use crate::app::Time;
use crate::physics::{Body, Physics};
use crate::world::{Component, Entity, World};
use threers::core::{Mesh, Object3D};
use threers::geometries::CylinderGeometry;
use threers::materials::StandardMaterial;
use threers::math::{Color, Quaternion, Vector3};
use threers_physics::prelude::{BodyId, RigidBody, Shape, Vehicle, WheelConfig};

/// Which wheels the engine turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Driven {
    /// Front-wheel drive.
    Front,
    /// Rear-wheel drive: the default.
    #[default]
    Rear,
    /// Four-wheel drive, the pull split evenly.
    All,
}

/// What a car is like. Forces are in newtons at the road, so they mean the same
/// whatever the wheel size. The default is a light, quick road car of a tonne.
#[derive(Debug, Clone)]
pub struct CarSpec {
    /// In kilograms.
    pub mass: f32,
    /// Half the body's width, height and length.
    pub half_size: Vector3,
    /// In metres (units).
    pub wheel_radius: f32,
    /// How far each wheel sits from the centre line, and the axles from the middle.
    pub track: f32,
    /// Half the distance between the axles.
    pub wheelbase: f32,
    /// Which wheels the engine turns.
    pub driven: Driven,
    /// Pull at low speed.
    pub engine_force: f32,
    /// Power in watts: above `power / engine_force` the pull falls with speed,
    /// which is what sets the top speed.
    pub power: f32,
    /// Braking force at the road, in newtons, at full brake.
    pub brake_force: f32,
    /// How far the front wheels turn at rest, in radians.
    pub max_steer: f32,
    /// The speed at which the steering lock has halved, so that full lock at
    /// speed is a turn and not a spin.
    pub steer_falloff: f32,
    /// How quickly the wheels reach the lock asked for, in radians per second.
    pub steer_rate: f32,
    /// Forward grip: how much of the weight on a tyre it can push with before it spins.
    pub grip: f32,
    /// Sideways grip as a share of forward grip; below 1 the car slides.
    pub lateral_grip: f32,
    /// Downforce as a share of weight per (unit/s)².
    pub downforce: f32,
}

impl Default for CarSpec {
    fn default() -> Self {
        Self {
            mass: 1000.0,
            half_size: Vector3::new(0.9, 0.35, 2.0),
            wheel_radius: 0.35,
            track: 0.85,
            wheelbase: 1.35,
            driven: Driven::default(),
            engine_force: 9000.0,
            power: 110_000.0,
            brake_force: 14_000.0,
            max_steer: 0.55,
            steer_falloff: 22.0,
            steer_rate: 2.5,
            grip: 1.4,
            lateral_grip: 0.95,
            downforce: 0.0004,
        }
    }
}

/// A car on the road. Set the first five fields; read the rest.
pub struct Car {
    /// -1 to 1. Negative slows the car and then reverses it.
    pub throttle: f32,
    /// 0 to 1.
    pub brake: f32,
    /// -1 is full left, 1 full right.
    pub steer: f32,
    /// Locks the rear wheels: a slide on purpose.
    pub handbrake: bool,
    /// A brake that is always on, 0 to 1, for rough ground.
    pub drag: f32,
    /// Forward speed, negative in reverse.
    pub speed: f32,
    /// How fast it is moving sideways: a slide, in units per second.
    pub sliding: f32,
    /// The most any tyre is asked past its grip, 0 to 1: wheelspin or a skid.
    pub slip: f32,
    /// How many wheels touched the ground on the last step.
    pub wheels_on_ground: usize,
    /// The solver's vehicle, for what this component does not cover.
    pub vehicle: Vehicle,
    /// What it was made with. Changing it does not rebuild the car.
    pub spec: CarSpec,
    wheels: Vec<Entity>,
    steering: f32,
}
impl Component for Car {}

/// Give an entity a car's body and wheels.
pub trait AttachCar {
    /// Replaces any body the entity had. Returns `None` if the entity is dead.
    fn attach_car(&mut self, entity: Entity, spec: CarSpec) -> Option<BodyId>;
}

impl AttachCar for World {
    fn attach_car(&mut self, entity: Entity, spec: CarSpec) -> Option<BodyId> {
        use crate::physics::AttachBody;
        let half = spec.half_size;
        let body = RigidBody::dynamic()
            .shape(Shape::cuboid(half.x, half.y, half.z))
            .mass(spec.mass)
            // Low, or a tight corner rolls it over.
            .center_of_mass(Vector3::new(0.0, -half.y, 0.0))
            .friction(0.3);
        let id = self.attach_body(entity, body)?;

        let mut vehicle = Vehicle::new(id);
        vehicle.downforce = spec.downforce;
        let mut wheels = Vec::with_capacity(4);
        for (x, z) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            let front = z < 0.0;
            let powered = match spec.driven {
                Driven::Front => front,
                Driven::Rear => !front,
                Driven::All => true,
            };
            let at = Vector3::new(x * spec.track, -half.y * 0.5, z * spec.wheelbase);
            let mut wheel = WheelConfig::new(at, spec.wheel_radius)
                .steering(front)
                .powered(powered);
            wheel.friction = spec.grip;
            wheel.lateral_grip = spec.lateral_grip;
            vehicle.add_wheel(wheel);

            let geometry = CylinderGeometry::new(
                spec.wheel_radius,
                spec.wheel_radius,
                spec.wheel_radius * 0.7,
                20,
                1,
                false,
                0.0,
                std::f32::consts::TAU,
            );
            let material = StandardMaterial::new(Color::from_hex(0x1c1c1f));
            let mesh = Object3D::mesh(Mesh::new(geometry, material.into()));
            let wheel = self.spawn_under(entity, mesh);
            // A plain tyre looks the same at every angle: spokes on both faces
            // show it turning.
            let hub = Object3D::mesh(Mesh::new(
                spokes(spec.wheel_radius),
                StandardMaterial::new(Color::from_hex(0xb8bcc4)).into(),
            ));
            self.spawn_under(wheel, hub);
            wheels.push(wheel);
        }
        self.insert(
            entity,
            Car {
                throttle: 0.0,
                brake: 0.0,
                steer: 0.0,
                handbrake: false,
                drag: 0.0,
                speed: 0.0,
                sliding: 0.0,
                slip: 0.0,
                wheels_on_ground: 0,
                vehicle,
                spec,
                wheels,
                steering: 0.0,
            },
        );
        Some(id)
    }
}

/// A cross of two bars on each face of a wheel of this radius, which stands
/// along Y as the tyre's cylinder does.
fn spokes(radius: f32) -> threers::core::BufferGeometry {
    use threers::geometries::{BoxGeometry, MeshBuilder};
    use threers::math::Matrix4;
    let (span, bar, face) = (radius * 1.7, radius * 0.28, radius * 0.35 + 0.01);
    let mut mesh = MeshBuilder::new();
    for y in [face, -face] {
        let at = Matrix4::translation(Vector3::new(0.0, y, 0.0));
        mesh.append(&BoxGeometry::new(span, 0.02, bar), &at);
        mesh.append(&BoxGeometry::new(bar, 0.02, span), &at);
    }
    mesh.build()
}

/// Turn each [`Car`]'s inputs into tyre forces. Set the inputs before this runs.
pub fn car_step(w: &mut World) {
    let dt = w.resource::<Time>().delta;
    w.with_resource::<Physics, _>(|w, physics| {
        w.each::<&mut Car>(|car| {
            let spec = &car.spec;
            let speed = car.vehicle.speed(physics);
            let pull = spec.engine_force.min(spec.power / speed.abs().max(1.0));
            let throttle = car.throttle.clamp(-1.0, 1.0);
            // Backwards on the stick while rolling forwards means stop, then reverse.
            let (drive, braking) = if throttle < 0.0 && speed > 1.0 {
                (0.0, -throttle)
            } else {
                (throttle, 0.0)
            };
            let brake = car.brake.max(braking).max(car.drag).clamp(0.0, 1.0);

            let lock = spec.max_steer / (1.0 + (speed / spec.steer_falloff).powi(2));
            // Positive radians turn the wheels left.
            let wanted = -car.steer.clamp(-1.0, 1.0) * lock;
            let step = spec.steer_rate * dt;
            car.steering += (wanted - car.steering).clamp(-step, step);

            let radius = spec.wheel_radius;
            let (brake_torque, handbrake) = (
                brake * spec.brake_force * radius / 4.0,
                spec.brake_force * radius / 2.0,
            );
            let steering = car.steering;
            let hand = car.handbrake;
            let vehicle = &mut car.vehicle;
            vehicle.set_steering(steering);
            vehicle.set_drive(drive * pull * radius);
            vehicle.set_brake(brake_torque);
            if hand {
                vehicle.set_handbrake(handbrake);
            }
            vehicle.update(physics, dt);
            car.speed = speed;
            car.slip = car
                .vehicle
                .wheels
                .iter()
                .map(|wheel| wheel.slip)
                .fold(0.0, f32::max);
            car.sliding = physics.body(car.vehicle.body).map_or(0.0, |body| {
                let right = body.position.transform_vector(Vector3::new(1.0, 0.0, 0.0));
                body.linear_velocity.dot(right)
            });
            car.wheels_on_ground = car.vehicle.wheels_on_ground();
        });
    });
}

impl Car {
    /// Where each tyre touches the ground, for skid marks and dust; `None` for a
    /// wheel in the air.
    pub fn contacts(&self) -> impl Iterator<Item = Option<Vector3>> + '_ {
        self.vehicle
            .wheels
            .iter()
            .map(|wheel| wheel.contact.as_ref().map(|c| c.point))
    }
}

/// Pose each [`Car`]'s wheels under it for this frame.
pub fn car_wheels(w: &mut World) {
    let mut poses = Vec::new();
    {
        let physics = w.resource::<Physics>();
        w.each_ref::<(&Car, &Body)>(|(car, body)| {
            let Some(chassis) = physics.body(body.0) else {
                return;
            };
            let inverse = chassis.position.inverse();
            for (index, &wheel) in car.wheels.iter().enumerate() {
                if let Some(pose) = car.vehicle.wheel_transform(physics, index) {
                    poses.push((wheel, inverse.mul(&pose)));
                }
            }
        });
    }
    // A cylinder stands along Y, and a wheel lies along its axle.
    let on_side =
        Quaternion::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);
    for (wheel, pose) in poses {
        if let Some(node) = w.scene.get_mut(wheel) {
            node.position = pose.translation;
            node.quaternion = pose.rotation.multiply(on_side);
        }
    }
}
