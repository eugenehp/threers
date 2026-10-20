//! Asset loaders. Hand-written parsers, zero binary-crate deps.

mod collada;
pub mod deflate;
#[cfg(feature = "usd")]
pub mod usd;
mod exr;
pub mod alembic;
pub mod draco;
pub mod fbx;
pub mod gltf_write;
pub mod ktx2;
pub mod meshopt;
mod gltf;
mod hdr;
pub(crate) mod json;
mod obj;
mod ply;
mod stl;
mod ttf;
pub mod typeface;
mod xml;

pub use collada::{ColladaError, ColladaLoader};
pub use exr::{ExrError, ExrLoader};
pub use fbx::{FbxError, FbxLoader, FbxScene};
pub use alembic::{AlembicError, Archive as AlembicArchive, PolyMesh};
pub use draco::{DracoError, Header as DracoHeader};
pub use ktx2::{Ktx2, Ktx2Error};
pub use meshopt::MeshoptError;
pub use gltf_write::{
    geometry_to_glb as gltf_geometry_to_glb, scene_to_glb, scene_to_glb_with, scene_to_gltf,
    scene_to_gltf_with, GltfExportOptions, GltfMaterialInfo, GltfMaterialKind,
};
pub use gltf::{add_to_scene as gltf_add_to_scene, GltfError, GltfImages, GltfLoader, GltfScene};
/// The three.js r165 `GLTFLoader` port the one-call API above is built on.
pub mod gltf_parser {
    pub use super::gltf::{
        decode_data_uri, sanitize_node_name, supply_images, texture_constants,
        three_attribute_name, world_matrix, Accessor, Assoc, CameraParams, Document, GltfClip,
        GltfGeometry, GltfMaterial, GltfObject, GltfOptions, GltfParser, GltfSkeleton,
        GltfTexture, GltfTrack, ImageData, ImageRequest, ImageSource, Instances, LightParams,
        MaterialType, ObjectType, Param, Record, TrackInterpolation, TrackKind,
    };
}
pub use hdr::{HdrError, HdrLoader};
pub use obj::ObjLoader;
pub use ply::PlyLoader;
pub use stl::StlLoader;
pub use ttf::{TtfError, TtfFont, TtfGlyph};
#[cfg(feature = "usd")]
pub use usd::{
    animated_scene_to_usda, animated_scene_to_usdc, animated_scene_to_usdz, geometry_to_usda,
    geometry_to_usdc, geometry_to_usdz, layer_to_usda, layer_to_usdc, layer_to_usdz,
    scene_to_usda, scene_to_usdc, scene_to_usdz, to_scene_at, UsdError, UsdExport, UsdFormat,
    UsdLayer,
    UsdLoader, UsdPrim, UsdScene, UsdValue, UsdzArchive, UsdzEntry, UsdzLayer,
};
