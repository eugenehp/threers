//! Asset loaders. Hand-written parsers, zero binary-crate deps.

mod collada;
pub mod deflate;
#[cfg(feature = "usd")]
pub mod usd;
mod exr;
mod fbx;
mod gltf;
mod hdr;
mod json;
mod obj;
mod ply;
mod stl;
mod ttf;
mod xml;

pub use collada::{ColladaError, ColladaLoader};
pub use exr::{ExrError, ExrLoader};
pub use fbx::{FbxError, FbxLoader};
pub use gltf::{add_to_scene as gltf_add_to_scene, GltfError, GltfImages, GltfLoader, GltfScene};
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
