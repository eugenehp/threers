//! Constructive solid geometry — Rust port of [three-bvh-csg@0.0.16](https://github.com/gkjohnson/three-bvh-csg).
//!
//! Enabled with the Cargo feature `bvh-csg` (implies `mesh-bvh`).
//!
//! # In the browser
//!
//! The same evaluator runs there: `csgEvaluate` (`src/wasm_csg.rs`) does the
//! boolean, and `web/bvh-csg-impl.js` keeps three-bvh-csg's `Brush`,
//! `Operation`, `OperationGroup` and `Evaluator` bookkeeping around it.
//!
//! # Against three-bvh-csg
//!
//! `tests/csg_upstream.rs` replays every evaluation the bvh-csg parity scenes
//! make, recorded from three-bvh-csg@0.0.16 itself
//! (`tests/parity/export-bvh-csg-upstream.mjs`), and holds each result to the
//! same surface area and enclosed volume. Triangle counts differ by a few
//! percent: three-mesh-bvh and [`MeshBvh`](crate::mesh_bvh::MeshBvh) build
//! different trees, so a face meets its cutting triangles in a different order
//! and is cut into a different number of pieces covering the same region.
//! `tests/parity/run-bvh-csg.js` compares the scenes' pixels from nine views.
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
mod constants;
mod evaluator;
mod geometry_prep;
mod half_edge_map;
mod hierarchy;
mod hit_side;
mod intersection_map;
/// f64 three.js math helpers used to match JS CSG float paths bit-for-bit where needed.
pub mod js_topology;
mod operations;
mod triangle_splitter;
mod triangle_utils;

#[cfg(test)]
mod tests;

pub use brush::CsgBrush;
pub use constants::*;
pub use evaluator::{CsgEvaluator, GroupArrays};
pub use hierarchy::{evaluate_hierarchy, CsgNode, CsgOperation, CsgOperationGroup};
