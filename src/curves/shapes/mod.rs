//! three.js r165's 2D shape stack — `Path`, `Shape`, `ShapePath`,
//! `ShapeUtils` and its `Earcut` — reproduced in `f64` and operation for
//! operation, so that a shape sampled, triangulated and extruded here yields
//! the buffers three.js yields in the browser.
//!
//! This sits beside [`crate::curves::Path`] / [`crate::curves::Shape`], the
//! `f32` paths the SVG, TTF and OpenSCAD code builds on. Those are general
//! drawing types; these are the exact three.js semantics that
//! [`ShapeGeometry`](crate::geometries::ShapeGeometry),
//! [`ExtrudeGeometry::from_shapes`](crate::geometries::ExtrudeGeometry::from_shapes)
//! and [`TextGeometry::from_font`](crate::geometries::TextGeometry::from_font)
//! — and the browser's `THREE.Shape` — need.

mod curve;
pub mod earcut;
mod path;
mod shape_path;
pub mod shape_utils;
#[cfg(test)]
mod tests;

pub use curve::{ShapeCurve, Vec2d, ARC_LENGTH_DIVISIONS};

/// three.js `Interpolations`: the scalar spline functions the curves are
/// built on, per coordinate, in `f64` as three.js computes them.
pub mod interpolations {
    pub use super::curve::{catmull_rom, cubic_bezier, quadratic_bezier};
}
pub(crate) use curve::{cubic_bezier, quadratic_bezier};
pub use path::{Path, Shape, ShapePoints};
pub use shape_path::{to_shapes, ShapePath};
