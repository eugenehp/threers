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
pub mod js_topology;
mod operations;
mod triangle_splitter;
mod triangle_utils;

#[cfg(test)]
mod tests;

pub use brush::CsgBrush;

/// Triangle–triangle intersection on [`Triangle`](crate::math::Triangle), in
/// the f64 path the CSG port uses. A trait because `Triangle` lives in
/// `threers-math`, which knows nothing of CSG.
pub trait TriangleIntersection {
    /// Triangle–triangle intersection (including coplanar cases).
    fn intersects_triangle(&self, other: &crate::math::Triangle) -> bool {
        self.intersects_triangle_ext(other, true)
    }

    /// three-mesh-bvh's `ExtendedTriangle.intersectsTriangle`, in f64 as the
    /// CSG port runs it. The flag is three-mesh-bvh's third argument,
    /// `suppressLog`, which only silences its coplanar warning; it does not
    /// change the answer.
    fn intersects_triangle_ext(&self, other: &crate::math::Triangle, suppress_log: bool) -> bool;
}

impl TriangleIntersection for crate::math::Triangle {
    fn intersects_triangle_ext(&self, other: &crate::math::Triangle, _suppress_log: bool) -> bool {
        use js_topology::JsTriangle;
        JsTriangle::from_triangle(*self).intersects_triangle_js(JsTriangle::from_triangle(*other))
    }
}
pub use constants::*;
pub use evaluator::{CsgEvaluator, GroupArrays};
pub use hierarchy::{evaluate_hierarchy, CsgNode, CsgOperation, CsgOperationGroup};
