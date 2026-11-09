//! Helper utilities mirroring three.js's `BufferGeometryUtils` and assorted
//! stand-alone math helpers (centroid, merge, normal/tangent computation).

mod buffer_geometry_utils;

pub mod png;

pub mod parallel;

#[cfg(feature = "captions")]
pub mod contrast;
pub use threers_math::srgb;

mod svg_number;
pub use svg_number::format_svg_number;

pub use buffer_geometry_utils::*;

pub mod mesh_check;

pub mod simplify;
