//! BVH-accelerated raycasting for `BufferGeometry` (three-mesh-bvh compatible).
//!
//! # Two kinds of caller, and why the defaults suit the other one
//!
//! Most callers *query* a tree — cast a ray, find a closest point — and want it
//! well shaped, because a subtree that failed to split is a linear scan. Build
//! those with [`BuildOptions::split_degenerate`] on.
//!
//! A second kind reads the tree's *shape*. [`bvhcast`](crate::mesh_bvh) walks
//! two trees together and reports the pairs of leaves that overlap, and the CSG
//! evaluator takes those pairs as its candidate set — so how finely the trees
//! were split decides how many candidates it sees, and with them what a boolean
//! returns. The defaults here reproduce three-mesh-bvh's tree so that those
//! callers, and the parity fixtures that pin them, keep their answer.

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
