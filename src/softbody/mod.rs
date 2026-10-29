//! Soft bodies that tear and grow back: XPBD on tetrahedral lattices, the
//! solver of the gummy creatures demo (`web/examples/jelly.html`).
//!
//! A creature is a soft *body* with *parts* glued on (tentacles, arms, a
//! tail), *chains* of particles rooted in it (gills, fringes, crests), and
//! anything else bound to its lattice to ride along (a skin, eyes, sugar).
//! Pull a part hard enough and it tears off, keeps twitching, and a bud grows
//! back from the stump.
//!
//! ```
//! use threers::math::Vector3;
//! use threers::sdf::Shape;
//! use threers::softbody::{ChainSpec, Creature, Marks, ShapeSpec};
//!
//! let v = Vector3::new;
//! // A blob with one arm.
//! let body = ShapeSpec { shape: Shape::ellipsoid(v(0.3, 0.15, 0.3)).translate(v(0.0, 0.15, 0.0)), cell: 0.03, h: 0.08 };
//! let arm = ShapeSpec { shape: Shape::capsule(v(0.1, 0.1, 0.0), v(0.6, 0.05, 0.0), 0.05), cell: 0.02, h: 0.05 };
//! let marks = Marks {
//!     head: [[0.0, 0.25, 0.0], [0.2, 0.2, 0.0], [0.0, 0.2, 0.2]],
//!     span: [[0.3, 0.15, 0.0], [-0.3, 0.15, 0.0]],
//! };
//! let c = Creature::from_shapes(&body, &[("arm".into(), arm)], &[], marks).unwrap();
//! let mut sim = c.into_soft_body(7);
//! for _ in 0..30 {
//!     sim.step(1.0 / 60.0);
//! }
//! assert!((sim.measure().volume - 1.0).abs() < 0.05);
//! ```
//!
//! The solver is a port of Axolotl Jelly's (Dream Meridian, MIT), arithmetic
//! for arithmetic, and `tests/softbody_parity.rs` holds it to the JavaScript:
//! the same lattice, node for node, and the same particles through a grab
//! and a tear.

mod creature;
#[cfg(all(target_arch = "wasm32", not(threers_no_js_bindings)))]
mod web;

pub use creature::{ChainSpec, Creature, Field, Marks, ShapeSpec};
// The lattice and the solver live in their own crate, `threers-softbody`.
pub use threers_softbody::*;
