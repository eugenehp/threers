//! Scene-graph and geometry primitives.

mod buffer_attribute;
mod buffer_geometry;
mod clock;
mod derived;
mod infer_quads;
mod instanced_mesh;
mod layers;
mod line_segments;
mod mesh;
mod object3d;
mod points;
mod raycaster;
mod skinning;
mod sprite;

pub use buffer_attribute::BufferAttribute;
pub use buffer_geometry::{BufferGeometry, DrawRange, GeometryGroup};
pub use clock::Clock;
pub use derived::DerivedCache;
pub use infer_quads::DEFAULT_INFER_QUADS_ANGLE;
pub use instanced_mesh::InstancedMesh;
pub use layers::Layers;
pub use line_segments::{LineSegments, LineStrip};
pub use mesh::{mesh_draws, Mesh, MeshDraw};
pub use object3d::{Object3D, ObjectArena, ObjectId, ObjectKind};
pub use points::Points;
pub use raycaster::{
    Face, Intersection, RaycastCamera, Raycaster, RaycasterParams, ThresholdParams,
};
pub use skinning::{
    morph_header, morph_uniforms, update_skeletons, BindMode, Bone, Morph, MorphAttributes, MorphTarget, MorphTexels, Skeleton, SkinnedMesh,
    MAX_MORPH_TARGETS,
};
pub(crate) use skinning::posed_positions;
#[allow(unused_imports)] // Read by the Metal backend only.
pub(crate) use skinning::MAX_BONES;
pub use sprite::Sprite;
