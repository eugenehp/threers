//! Tendon-driven continuum robots for
//! [`threers-physics`](https://docs.rs/threers-physics).
//!
//! A continuum robot has no joints. It is a flexible rod, bent by cables run
//! down its length, and it goes where the balance between the two puts it —
//! which is why the interesting question is never *where are the joints* but
//! *how stiff is the rod and how hard are you pulling*.
//!
//! ```
//! use threers_continuum::prelude::*;
//! use threers_physics::prelude::*;
//!
//! let mut world = World::new();
//! // A rod is a long chain of constraints, and the defaults are tuned for
//! // loose bodies — see "Solver budget" below, which is not optional reading.
//! world.substeps = 16;
//! world.solver_config.velocity_iterations = 64;
//!
//! // A 300 mm nylon rod, 2 mm radius.
//! let rod = Rod::new(0.3, 10)
//!     .radius(0.002)
//!     .material(2.0e9, 0.35, 1200.0)
//!     .damped(0.05);
//!
//! let mut arm = Continuum::build(&mut world, rod, None, Vector3::ZERO, Vector3::UP);
//! arm.add_tendon_ring(&mut world, 3, 0.0015, 0.0, 0, 0.0);
//!
//! // Reel the first cable in by 1.5 mm and let it settle.
//! arm.set_pull(&mut world, 0, 0.0015, 20.0);
//! for _ in 0..900 {
//!     world.step_fixed();
//! }
//!
//! // It curled toward the cable that was pulled, and it did not stretch.
//! assert!(arm.tip(&world).x > 0.01);
//! assert!((arm.arc_length(&world) - 0.3).abs() < 0.005);
//! ```
//!
//! # What it is doing
//!
//! The rod is discretised: `n` links of `L/n` each, joined by elastic stations
//! whose stiffness comes from beam theory rather than from tuning.
//!
//! ```text
//! K_bend  = n·E·Iₓ / L      Iₓ = π/4 (rₒ⁴ − rᵢ⁴)
//! K_twist = n·G·I_z / L     I_z = 2Iₓ,  G = E / 2(1+ν)
//! ```
//!
//! The `n` in the numerator is the whole reason this works: a station standing
//! in for a shorter piece of the same beam is proportionally stiffer, so the
//! rod as a whole bends the same however finely it is chopped. `links` is a
//! fidelity knob, not a physics one — turn it up until the shape stops moving.
//!
//! The stiffness is solved as an implicit constraint rather than integrated as
//! a torque (see [`JointSpring`](threers_physics::joint::JointSpring)), which
//! is what lets a steel spine — `n·EI/L` of a hundred thousand and up — run at
//! 60 Hz without the timestep having to know about it.
//!
//! Cables are [`Tendon`](threers_physics::tendon::Tendon)s: one constraint on
//! the *total* routed length, through two guides per link, so pulling one end
//! loads every station it passes and does so along the direction the cable
//! actually leaves each guide.
//!
//! # Where things live
//!
//! | Module | Role |
//! |---|---|
//! | [`rod`] | [`Rod`](rod::Rod) — what the rod is, and the beam theory |
//! | [`build`] | [`Continuum`](build::Continuum) — the rod as bodies, stations and cables |
//! | [`clark`] | [`Clark`](clark::Clark) — three cables as one two-dimensional bend |
//! | [`kinematics`] | The constant-curvature model, and closed-loop tip control |
//! | [`scad`] | Building one out of a `.scad` model's `continuum()` declaration |
//!
//! # Solver budget
//!
//! **Raise it.** A rod is a chain, sequential impulses carry information about
//! roughly one constraint per iteration, and
//! [`World`](threers_physics::world::World)'s defaults — 4 substeps of 4
//! iterations — are tuned for scenes of loose bodies. On a chain they do not
//! merely let it sag: past a handful of links the chain *diverges*, stretching
//! visibly and thrashing. That is a property of the solver rather than of the
//! discretisation, and a plain row of
//! [`Joint::fixed`](threers_physics::joint::Joint::fixed) welds does the same
//! thing at the same length — which is how you can tell it is not the springs.
//!
//! Measured on a 300 mm nylon cantilever under its own weight, as the settled
//! tip droop over the `wL⁴/8EI` the beam it was built from predicts:
//!
//! | links | 4 sub / 4 it | 8 / 16 | 16 / 64 |
//! |---|---|---|---|
//! | 5 | diverges | **0.98** | **0.98** |
//! | 10 | diverges | diverges | **0.99** |
//!
//! **How much you need depends on the rod, not only on the link count.** The
//! [`sorosim`] comparison measures this properly against Cosserat-rod
//! solutions, and the two rods there could not be less alike:
//!
//! | links | soft, thick (TPU 5 mm) | stiff, thin (steel 0.8 mm) |
//! |---|---|---|
//! | 5 | 1.56% | **1.48%** |
//! | 10 | 0.63% | 2.31% |
//! | 20 | 0.32% | 6.95% |
//! | 30 | **0.23%** | 12.23% |
//!
//! at 32 substeps of 128 iterations, as a percentage of rod length. The soft
//! rod converges the way the discretisation promises and is indifferent to the
//! budget. The stiff one is **best at five links** and worse at thirty — more
//! links there is a longer chain of stiffer constraints on lighter bodies, and
//! the solver loses ground faster than the discretisation gains it.
//!
//! So: start at 16/64, raise it until the numbers stop moving, and for a stiff
//! light rod do not assume more links is better — measure. What limits the
//! stiff case is the solver's chain conditioning and nothing about the rod.
//!
//! # Units
//!
//! The model's own, consistently. SI throughout means `youngs` in pascals,
//! `length` in metres and `density` in kg/m³; a millimetre-based model needs
//! all three in millimetre units, and mixing them is the mistake that makes a
//! rod either infinitely stiff or a wet noodle. **Angles are radians** here,
//! unlike the `.scad` declarations they may have come from — [`scad`] is where
//! that conversion happens, once.
//!
//! # What it is not
//!
//! Not a Cosserat rod. The discretisation is a lumped-parameter approximation
//! and converges to one as `links` rises, with the error falling off roughly as
//! `1/n²`; it does not model shear or axial extension at all, both of which are
//! negligible for the slender rods this is for and neither of which is
//! negligible for a short thick one.

pub mod build;
pub mod clark;
pub mod kinematics;
pub mod rod;
pub mod sorosim;
#[cfg(feature = "openscad")]
pub mod scad;

pub use build::{Continuum, TendonRoute, ROD_GROUP};
pub use clark::Clark;
pub use kinematics::{
    constant_curvature_frames, piecewise_constant_curvature, Frame, TipController,
};
pub use rod::{LinkProperties, Rod};

/// Everything, in one `use`.
pub mod prelude {
    pub use crate::build::{Continuum, TendonRoute};
    pub use crate::clark::Clark;
    pub use crate::kinematics::{piecewise_constant_curvature, Frame, TipController};
    pub use crate::rod::{LinkProperties, Rod};
    #[cfg(feature = "openscad")]
    pub use crate::scad::{build_scad_continua, rod_from_spec, route_from_spec, ScadContinuum};
    pub use threers::math::{Quaternion, Vector3};
}
