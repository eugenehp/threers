//! Soft bodies that tear and grow back: XPBD on tetrahedral lattices.
//!
//! All particles live in one array cut into *blocks*: the body, and for each
//! part one attached slot plus two loose slots for torn-off copies. Glue holds
//! each part's root to the body; when a held part's glue stretches past its
//! limit the part is copied into a loose slot (which keeps simulating, and
//! twitches) and the attached slot restarts as a bud at 5 %, its rest lengths
//! and volumes scaled back to full size.
//!
//! The core of [`threers::softbody`](https://docs.rs/threers/latest/threers/softbody/)
//! on its own — the lattice, the solver, and the arithmetic they share with
//! their JavaScript original — with no GPU, windowing or shape dependencies.
//! threers builds creatures for it from signed distance fields and binds it
//! for the browser.
//!
//! A separate crate so it can be optimised for speed inside a wasm build that
//! is otherwise optimised for size (`[profile.release-wasm.package.threers-softbody]`).
//!
//! The solver is a port of Axolotl Jelly's (Dream Meridian, MIT), arithmetic
//! for arithmetic; threers' `tests/softbody_parity.rs` holds it to the
//! JavaScript bit for bit.

// The loops index as the JavaScript original does, value for value; iterator
// forms would hide the correspondence the parity test depends on.
#![allow(clippy::needless_range_loop)]

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod js_math;
mod lattice;
mod sim;
mod skin;

pub use lattice::{embed_points, Bounds, Embedding, Lattice, Part};
pub use skin::carry;

// The README's example, compiled and run with the doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

pub use sim::{Block, Chains, Event, Grab, Hit, Landmarks, Limb, Measure, Muscles, Params, Profile, Pulse, SoftBody, Topo, MAX_DROPS};
