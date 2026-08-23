//! BVH-accelerated raycasting for `BufferGeometry` (three-mesh-bvh compatible).

mod build;
mod bvhcast;
mod hit;
// The type lives in a file named after the module it defines; the parent is
// the facade that re-exports it.
#[allow(clippy::module_inception)]
mod mesh_bvh;
mod node;
mod serialize;

pub use build::{BuildOptions, AVERAGE, CENTER, SAH};
pub use hit::BvhHit;
pub use mesh_bvh::MeshBvh;
pub use serialize::{SerializedMeshBvh, SERIALIZE_VERSION};

/// Material side constants (mirror three.js).
pub const FRONT_SIDE: u32 = 0;
pub const BACK_SIDE: u32 = 1;
pub const DOUBLE_SIDE: u32 = 2;

/// Shapecast traversal result constants (mirror three-mesh-bvh).
pub const NOT_INTERSECTED: i32 = 0;
pub const INTERSECTED: i32 = 1;
pub const CONTAINED: i32 = 2;
