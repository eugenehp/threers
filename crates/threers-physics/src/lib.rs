//! Rigid-body physics for [`threers`](https://docs.rs/threers).
//!
//! Gravity, collisions, joints, scene queries and inverse kinematics — pure
//! Rust, no dependencies beyond `threers` itself, and wasm-ready.
//!
//! ```
//! use threers_physics::prelude::*;
//!
//! let mut world = World::new();                     // gravity is -9.81 Y
//! world.add_body(RigidBody::fixed().shape(Shape::ground()));
//!
//! let ball = world.add_body(
//!     RigidBody::dynamic()
//!         .shape(Shape::ball(0.5))
//!         .translation(Vector3::new(0.0, 10.0, 0.0))
//!         .restitution(0.7),
//! );
//!
//! for _ in 0..600 {
//!     world.step(1.0 / 60.0);                       // once per frame
//! }
//!
//! // It fell, bounced itself out, and came to rest on the ground — then went
//! // to sleep, so it costs nothing from here on.
//! let body = world.body(ball).unwrap();
//! assert!((body.translation().y - 0.5).abs() < 0.05);
//! assert!(body.is_sleeping());
//! ```
//!
//! # Where things live
//!
//! | Module | Role |
//! |---|---|
//! | [`world`] | [`World`](world::World) — the object you actually talk to |
//! | [`body`] | [`RigidBody`](body::RigidBody), its builder, and the arena |
//! | [`shape`] | Collision shapes and analytic mass properties |
//! | [`collider`] | A shape plus where it sits on a body |
//! | [`joint`] | Fixed, spherical, revolute, prismatic, distance, spring |
//! | [`query`] | Raycasts, shape casts, overlap and point queries |
//! | [`ik`] | FABRIK and CCD chains with joint limits |
//! | [`character`] | Kinematic move-and-slide controller |
//! | [`material`] | Friction, restitution and collision filtering |
//!
//! The stages of a step — [`broadphase`], [`narrowphase`], [`contact`],
//! [`solver`] — are public too, so a custom pipeline can reuse the pieces.
//! [`gjk`] and [`hull`] are the geometry kernels underneath.
//!
//! # Conventions worth knowing up front
//!
//! - **Sizes are half-extents**, because that is what the solver works in. Every
//!   such constructor has a `*_from_size` sibling taking the full dimensions a
//!   `*Geometry` would: [`Shape::cuboid`](shape::Shape::cuboid) versus
//!   [`Shape::cuboid_from_size`](shape::Shape::cuboid_from_size).
//! - **Capsules, cylinders and cones are Y-aligned**, matching three.js.
//! - **Contact normals point from `b` toward `a`**: translating `a` along the
//!   normal separates the pair. This holds throughout [`narrowphase`],
//!   [`contact`] and [`gjk`].
//! - **Restitution combines with `Max`, friction with `Average`** — see
//!   [`PhysicsMaterial`](material::PhysicsMaterial) for why.
//! - **Jointed bodies do not collide with each other** unless you ask; see
//!   [`Joint::collide_connected`](joint::Joint::collide_connected).
//!
//! # Determinism
//!
//! The same scene stepped the same way produces the same result, run to run and
//! regardless of whether `parallel` is enabled. Constraint solve order and
//! reported collision events are sorted rather than left to hash-map iteration
//! order, which is what replays and lockstep networking need.
//!
//! # Feature flags
//!
//! | Flag | Effect |
//! |---|---|
//! | `parallel` | Multi-threaded narrow phase via rayon. Native only. |
//! | `gpu` | Broad phase on a wgpu compute shader. Native **and** wasm32/WebGPU. |
//! | `async` | Pulls in a tokio runtime for async callers. Native only. |
//!
//! rayon needs OS threads, so on `wasm32` `parallel` is deliberately a no-op
//! rather than a build error. [`gpu`] is the browser's route to parallelism —
//! read its module docs before reaching for it, since a dispatch and readback
//! costs more than the CPU sweep-and-prune below a few thousand bodies.

#[cfg(feature = "assembly")]
pub mod assembly;
pub mod body;
pub mod broadphase;
pub mod bvh;
pub mod character;
pub mod collider;
pub mod contact;
pub mod debug;
pub mod decompose;
pub mod diagnostics;
pub mod forces;
pub mod gjk;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod heightfield;
pub mod hull;
pub mod ik;
pub mod island;
pub mod joint;
pub mod material;
pub mod math;
/// A `.scad` mechanism, simulated — the physics-driven counterpart of
/// `ScadAnimation`.
#[cfg(all(feature = "assembly", feature = "openscad"))]
pub mod mechanism;
/// Checking a declared mechanism against the parts that have to perform it.
#[cfg(all(feature = "assembly", feature = "openscad"))]
pub mod verify;
pub mod narrowphase;
pub mod query;
/// Colliders straight from an OpenSCAD [`Solid`](threers::openscad::Solid).
#[cfg(feature = "openscad")]
pub mod scad;
pub mod shape;
pub mod snapshot;
pub mod solver;
pub mod tendon;
pub mod trimesh;
pub mod vehicle;
pub mod world;

/// Everything you normally need, in one import.
///
/// ```
/// use threers_physics::prelude::*;
/// ```
///
/// Re-exports `Vector3`, `Quaternion` and `Ray` from `threers` too, so simple
/// programs need only this one `use`.
pub mod prelude {
    pub use crate::body::{BodyId, BodySet, BodyType, LockedAxes, RigidBody, RigidBodyBuilder};
    pub use crate::character::{CharacterController, CharacterMove};
    pub use crate::collider::Collider;
    pub use crate::contact::{CollisionEvent, ContactManifold, ContactPoint, ContactSet};
    pub use crate::decompose::{decompose, decompose_to_shape, DecompositionConfig};
    pub use crate::gjk::{closest_points, Proximity, ShapeProxy, SupportMap};
    pub use crate::hull::{ConvexHull, HullFace};
    pub use crate::ik::{IkChain, IkConstraint, IkJoint, IkResult};
    pub use crate::joint::{
        Dof, DofMotion, Joint, JointId, JointKind, JointLimits, JointSet, JointSpring, Motor,
        Softness,
    };
    #[cfg(feature = "mechanism")]
    pub use crate::joint::{AxialKind, JointState, Servo};
    #[cfg(feature = "assembly")]
    pub use crate::assembly::{
        Assembly, AssemblyError, AssemblyReport, Axis, Easing, Feature, FeatureKind, Interference,
        Mate, MateId, MateKind, Part, PartId, PartPhysics, Sweep,
    };
    pub use crate::material::{CombineRule, InteractionGroups, PhysicsMaterial};
    pub use crate::math::{Aabb, Isometry, Mat3};
    pub use crate::query::{PointProjection, QueryFilter, RayHit, ShapeHit};
    pub use crate::shape::{ColliderFit, MassProperties, Shape};
    pub use crate::solver::{SimulationQuality, SolverConfig};
    pub use crate::tendon::{
        ResolvedPath, Tendon, TendonArc, TendonId, TendonKind, TendonNode, TendonObstacle,
        TendonPoint, TendonSet,
    };
    pub use crate::trimesh::TriMesh;
    pub use crate::vehicle::{Vehicle, Wheel, WheelConfig};
    pub use crate::world::{GravityModel, SleepConfig, World};
    pub use threers::math::{Quaternion, Ray, Vector3};
}
