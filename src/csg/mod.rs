//! Constructive solid geometry — Rust port of [three-bvh-csg@0.0.16](https://github.com/gkjohnson/three-bvh-csg).
//!
//! Enabled with the Cargo feature `bvh-csg` (implies `mesh-bvh`).
//!
//! # Dual stack
//!
//! | Path | Role |
//! |------|------|
//! | **Rust** (`src/csg/`) | Native / wasm boolean evaluation (`CsgEvaluator`, hierarchy ops) |
//! | **JS** (`web/csg/`) | Browser CSG when building with `BVH_CSG=1` (shim addon) |
//!
//! The hierarchy parity scene (shell − cut + sphere + window) matches JS with
//! **exact ordered TriKey topology** and exact vertex counts through all four
//! live steps. See `step_tests` and `scripts/ci-bvh-csg.sh`.
//!
//! # Quick start (native)
//!
//! ```ignore
//! use threers::{CsgBrush, CsgEvaluator, BoxGeometry, ADDITION, SUBTRACTION};
//!
//! let mut ev = CsgEvaluator::new();
//! let mut a = CsgBrush::new(BoxGeometry::new(2.0, 2.0, 2.0));
//! let mut b = CsgBrush::new(BoxGeometry::new(1.0, 1.0, 1.0));
//! let geom = ev.evaluate(&mut a, &mut b, SUBTRACTION);
//! ```
//!
//! Examples: `bvh_csg_hierarchy`, `bvh_csg_steps` (`--features bvh-csg`).

mod attribute_data;
mod brush;
/// BVH/bvhcast split-order parity helpers — used only by the parity tests.
#[cfg(test)]
mod bvhcast_parity;
mod constants;
mod evaluator;
mod geometry_prep;
mod half_edge_map;
mod hierarchy;
mod hierarchy_steps;
mod hit_side;
mod intersection_map;
/// f64 three.js math helpers used to match JS CSG float paths bit-for-bit where needed.
pub mod js_topology;
mod operations;
mod topology;
mod triangle_splitter;
mod triangle_utils;

#[cfg(test)]
mod step_tests;
#[cfg(test)]
mod tests;

pub use brush::CsgBrush;
pub use constants::*;
pub use evaluator::CsgEvaluator;
pub use geometry_prep::load_positions_geometry_bin;
pub use hierarchy::{
    build_bvh_csg_hierarchy_geometry, evaluate_hierarchy, CsgNode, CsgOperation, CsgOperationGroup,
};
pub use hierarchy_steps::{
    assert_step_verts, evaluate_live_through, evaluate_through, js_target_verts,
    load_reference_after_sphere, step1_shell_cut, step2_add_sphere, step3_win_cut, step4_win_frame,
    step_tolerance, JS_STEP1_VERTS, JS_STEP2_VERTS, JS_STEP3_VERTS, JS_STEP4_VERTS,
};
pub use topology::{assert_topology_overlap, topology_overlap, triangle_keys, TriKey};
