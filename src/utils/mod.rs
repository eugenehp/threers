//! Helper utilities mirroring three.js's `BufferGeometryUtils` and assorted
//! stand-alone math helpers (centroid, merge, normal/tangent computation).

mod buffer_geometry_utils;

/// Dependency-free PNG encode/decode (and the DEFLATE codec underneath).
pub mod png;

/// Data-parallel helpers (rayon behind the `parallel` feature, sequential otherwise).
pub mod parallel;

pub use buffer_geometry_utils::*;
