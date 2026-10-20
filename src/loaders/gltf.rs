//! glTF 2.0, read as three.js r165's `GLTFLoader` reads it.
//!
//! [`GltfParser`] is a port of three.js's `GLTFParser`: the same objects
//! (`Group` per scene, a `Mesh` per primitive under a `Group` for a
//! multi-primitive mesh, `Bone`s, `SkinnedMesh`es bound in three.js's attached
//! mode, `InstancedMesh`es for `EXT_mesh_gpu_instancing`, cameras, punctual
//! lights with their targets), the same names (`createUniqueName`,
//! `_instance_<n>` clones), the same materials (`pbrMetallicRoughness`, alpha
//! modes, `KHR_materials_*`, `EXT_materials_bump`, `KHR_texture_transform`,
//! samplers, `assignFinalMaterial`'s derived materials), the same animation
//! tracks, `userData` from `extras`, and `parser.associations`. Every object is
//! built straight into an [`ObjectArena`] as a node of the kind it is; the
//! three.js-side description of each (its class, parameters, names) is kept
//! beside it for a JS layer to wrap the nodes with.
//!
//! Decoding is all here: `KHR_draco_mesh_compression` (`super::draco`),
//! `EXT_meshopt_compression` (`super::meshopt`), `KHR_mesh_quantization`,
//! sparse and normalized accessors, PNG and KTX2 images (other image formats
//! are decoded by the caller — a browser's image decoder — and handed in as
//! RGBA). External `.bin` files and image URIs are fetched by the caller.
//!
//! [`GltfLoader`] is the older one-call API over the same parser: the default
//! scene's root nodes in a fresh arena, with the animations as
//! [`AnimationClip`]s.

mod document;
mod material;
mod parser;
#[cfg(test)]
mod tests;

use std::collections::HashMap;

pub use document::{decode_data_uri, Accessor, Document, ImageSource};
pub use material::{
    constants as texture_constants, Assoc, GltfMaterial, GltfTexture, ImageData, MaterialType, Param,
};
pub use parser::{
    sanitize_node_name, three_attribute_name, world_matrix, CameraParams, GltfClip, GltfGeometry,
    GltfObject, GltfOptions, GltfParser, GltfSkeleton, GltfTrack, ImageRequest, Instances,
    LightParams, ObjectType, Record, TrackInterpolation, TrackKind,
};

use crate::animation::{AnimationClip, Interpolation, KeyframeTrack, TrackTarget};
use crate::core::{Bone, ObjectArena, ObjectId, Skeleton};
use crate::math::{Matrix4, Quaternion, Vector3};
use crate::scene::Scene;

/// Decoded image data the caller supplies for GLTFs that reference textures.
/// Keyed by GLTF `images[i]` index. Format must be 8-bit RGBA.
pub type GltfImages = HashMap<usize, (u32, u32, Vec<u8>)>;

#[derive(Debug)]
pub enum GltfError {
    Json(&'static str),
    BadMagic,
    UnsupportedVersion,
    UnsupportedAttribute(String),
    MissingField(&'static str),
    BadBase64,
    BadAccessor,
    /// The file lists an extension in `extensionsRequired` that this loader
    /// does not implement.
    ///
    /// `extensionsRequired` means what it says: the asset cannot be loaded
    /// correctly without it. Reading such a file anyway produces a scene that
    /// is *quietly* wrong — a mesh with no geometry, a material with no
    /// texture, instances that never appear — so the name of the missing
    /// extension is carried here rather than being guessed at from the damage.
    /// (three.js only warns; [`GltfOptions::refuse_unknown_required`] picks.)
    UnsupportedExtension(String),
    /// A `KHR_draco_mesh_compression` payload would not decode.
    Draco(super::draco::DracoError),
    /// Anything else, in three.js's words where three.js has some.
    Message(String),
}

impl GltfError {
    pub fn msg(m: impl Into<String>) -> Self {
        GltfError::Message(m.into())
    }
}

impl std::fmt::Display for GltfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GltfError::Json(e) => write!(f, "THREE.GLTFLoader: invalid JSON ({e})"),
            GltfError::BadMagic => write!(f, "THREE.GLTFLoader: Unsupported glTF-Binary header."),
            GltfError::UnsupportedVersion => {
                write!(f, "THREE.GLTFLoader: Unsupported asset. glTF versions >=2.0 are supported.")
            }
            GltfError::UnsupportedAttribute(a) => write!(f, "THREE.GLTFLoader: unsupported attribute {a}"),
            GltfError::MissingField(m) => write!(f, "THREE.GLTFLoader: missing {m}"),
            GltfError::BadBase64 => write!(f, "THREE.GLTFLoader: invalid base64 in a data URI"),
            GltfError::BadAccessor => write!(f, "THREE.GLTFLoader: invalid accessor"),
            GltfError::UnsupportedExtension(e) => {
                write!(f, "THREE.GLTFLoader: required extension \"{e}\" is not supported")
            }
            GltfError::Draco(e) => write!(f, "THREE.GLTFLoader: KHR_draco_mesh_compression: {e:?}"),
            GltfError::Message(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for GltfError {}

impl From<super::draco::DracoError> for GltfError {
    fn from(e: super::draco::DracoError) -> Self {
        GltfError::Draco(e)
    }
}

/// Resulting scene + the underlying arena. Add the root to your `Scene` if
/// you want it rendered.
pub struct GltfScene {
    pub roots: Vec<ObjectId>,
    pub arena: ObjectArena,
    pub animations: Vec<AnimationClip>,
    pub skeletons: Vec<Skeleton>,
}

pub struct GltfLoader;

impl GltfLoader {
    /// Parse a `.glb` byte slice with optional decoded images (caller-supplied).
    pub fn parse_glb_with_images(bytes: &[u8], images: &GltfImages) -> Result<GltfScene, GltfError> {
        let doc = Document::from_bytes(bytes)?;
        if !doc.is_glb {
            return Err(GltfError::BadMagic);
        }
        load(doc, images)
    }

    /// Parse a `.glb` byte slice. Returns a fresh `GltfScene` plus root nodes.
    pub fn parse_glb(bytes: &[u8]) -> Result<GltfScene, GltfError> {
        Self::parse_glb_with_images(bytes, &HashMap::new())
    }

    /// Parse a `.gltf` JSON string with optional named buffers (caller supplies
    /// any external `.bin` blob data).
    pub fn parse_gltf(
        json_str: &str,
        external_buffers: &HashMap<usize, Vec<u8>>,
    ) -> Result<GltfScene, GltfError> {
        Self::parse_gltf_with_images(json_str, external_buffers, &HashMap::new())
    }

    /// Parse a `.gltf` JSON string with external buffers AND decoded images.
    pub fn parse_gltf_with_images(
        json_str: &str,
        external_buffers: &HashMap<usize, Vec<u8>>,
        images: &GltfImages,
    ) -> Result<GltfScene, GltfError> {
        let mut doc = Document::from_parts(json_str.to_string(), None, false)?;
        for (i, b) in external_buffers {
            doc.set_buffer(*i, b.clone());
        }
        load(doc, images)
    }

    /// Build from JSON alone, for tests that have no binary chunk.
    #[cfg(test)]
    pub(crate) fn build_for_test(root: &crate::loaders::json::Value) -> Result<GltfScene, GltfError> {
        let doc = Document::from_parts(root.to_json(), None, false)?;
        load(doc, &HashMap::new())
    }
}

/// Supply every image a parser asks for: `images` first, then whatever the
/// file embeds (PNG and KTX2 decode here); anything else fails, leaving its
/// textures `null` as three.js would.
pub fn supply_images(parser: &mut GltfParser, images: &GltfImages) -> Result<(), GltfError> {
    loop {
        let pending = parser.pending_images()?;
        if pending.is_empty() {
            return Ok(());
        }
        for req in pending {
            if let Some((w, h, data)) = images.get(&req.index) {
                parser.set_image_rgba(req.index, *w, *h, data.clone());
            } else if let Some(bytes) = &req.source.bytes {
                if !parser.set_image_bytes(req.index, bytes) {
                    parser.set_image_failed(req.index, format!("no decoder here for {}", req.source.mime_type));
                }
            } else {
                parser.set_image_failed(req.index, "external image not supplied".into());
            }
        }
    }
}

fn load(doc: Document, images: &GltfImages) -> Result<GltfScene, GltfError> {
    let mut parser = GltfParser::new(doc, GltfOptions::default())?;
    supply_images(&mut parser, images)?;
    let mut arena = ObjectArena::new();
    parser.parse(&mut arena)?;
    let mut roots = Vec::new();
    if let Some(scene) = parser.default_scene() {
        let id = parser.objects[scene].id;
        roots = arena.get(id).map(|n| n.children.clone()).unwrap_or_default();
        // The roots stand on their own here, as they always have.
        arena.delete(id);
    } else if parser.scenes.is_empty() {
        // No scenes at all: every node is a root.
        let n = document::count(parser.root(), "nodes");
        let mut nodes = Vec::new();
        for i in 0..n {
            if let Some(o) = parser.load(&mut arena, "node", i)? {
                nodes.push(parser.objects[o].id);
            }
        }
        roots = nodes.into_iter().filter(|id| arena.get(*id).is_some_and(|n| n.parent.is_none())).collect();
    }
    let skeletons = parser
        .skeletons
        .iter()
        .map(|s| {
            Skeleton::new(
                s.bones
                    .iter()
                    .zip(&s.inverses)
                    .map(|(&b, inv)| Bone { node: parser.objects[b].id, inverse_bind: Matrix4 { elements: *inv } })
                    .collect(),
            )
        })
        .collect();
    let animations = parser
        .animations
        .iter()
        .flatten()
        .map(|&c| legacy_clip(&parser, &parser.clips[c]))
        .collect();
    Ok(GltfScene { roots, arena, animations, skeletons })
}

/// A clip in the crate's own animation types: one track per animated
/// property, and per morph target for `weights`.
fn legacy_clip(parser: &GltfParser, clip: &GltfClip) -> AnimationClip {
    let mut tracks = Vec::new();
    for t in &clip.tracks {
        let object = parser.objects[t.target].id;
        let interp = match t.interpolation {
            TrackInterpolation::Step => Interpolation::Step,
            TrackInterpolation::CubicSpline => Interpolation::Cubic,
            TrackInterpolation::Linear => Interpolation::Linear,
        };
        let cubic = t.interpolation == TrackInterpolation::CubicSpline;
        let keys = t.times.len();
        // The spline vertices of a cubic track (the middle of each triple).
        let vertices = |size: usize| -> Vec<f32> {
            if !cubic {
                return t.values.to_vec();
            }
            (0..keys)
                .flat_map(|k| t.values.get(k * size * 3 + size..k * size * 3 + 2 * size).unwrap_or(&[]).to_vec())
                .collect()
        };
        let track = match t.path.as_str() {
            "translation" | "scale" => {
                let vs = vertices(3).chunks_exact(3).map(|c| Vector3::new(c[0], c[1], c[2])).collect();
                let target = if t.path == "scale" { TrackTarget::Scale } else { TrackTarget::Position };
                Some(KeyframeTrack::vector(object, target, t.times.to_vec(), vs))
            }
            "rotation" => {
                let vs = vertices(4).chunks_exact(4).map(|c| Quaternion::new(c[0], c[1], c[2], c[3])).collect();
                Some(KeyframeTrack::quaternion(object, TrackTarget::Quaternion, t.times.to_vec(), vs))
            }
            "weights" => {
                // `[w0 … wN]` per key, or `[in(N), value(N), out(N)]` for
                // CUBICSPLINE: one scalar track per target (cubic keys keep
                // their in/value/out triplets).
                let per_key = if cubic { 3 } else { 1 };
                let morph_count = (t.values.len() / (keys.max(1) * per_key)).max(1);
                for mi in 0..morph_count {
                    let mut vs = Vec::with_capacity(keys * per_key);
                    for k in 0..keys {
                        for part in 0..per_key {
                            if let Some(&v) = t.values.get(k * morph_count * per_key + part * morph_count + mi) {
                                vs.push(v);
                            }
                        }
                    }
                    let mut s = KeyframeTrack::scalar(object, TrackTarget::MorphWeight { index: mi }, t.times.to_vec(), vs);
                    s.interpolation = interp;
                    tracks.push(s);
                }
                None
            }
            _ => None,
        };
        if let Some(mut track) = track {
            track.interpolation = interp;
            tracks.push(track);
        }
    }
    AnimationClip::new(clip.name.clone(), clip.duration(), tracks)
}

/// Append `loaded` into `scene` and return the inserted root ids. Re-parents
/// every loaded root under `scene.root`.
pub fn add_to_scene(scene: &mut Scene, loaded: GltfScene) -> Vec<ObjectId> {
    let mut remap: HashMap<ObjectId, ObjectId> = HashMap::new();
    let all: Vec<(ObjectId, crate::core::Object3D)> =
        loaded.arena.nodes.iter().map(|(k, v)| (k, v.clone())).collect();
    for (old, mut obj) in all {
        obj.parent = None;
        obj.children.clear();
        let new = scene.arena.insert(obj);
        remap.insert(old, new);
    }
    // Skinned meshes name their bones by id: point them at the new ones.
    for new in remap.values() {
        if let Some(crate::core::ObjectKind::SkinnedMesh(sm)) = scene.arena.get_mut(*new).map(|n| &mut n.kind) {
            let bones = sm
                .skeleton
                .bones
                .iter()
                .map(|b| Bone { node: remap.get(&b.node).copied().unwrap_or(b.node), inverse_bind: b.inverse_bind })
                .collect();
            sm.skeleton = std::sync::Arc::new(Skeleton::new(bones));
        }
    }
    let snapshot: Vec<(ObjectId, Vec<ObjectId>)> =
        loaded.arena.nodes.iter().map(|(k, v)| (k, v.children.clone())).collect();
    for (old_parent, children) in snapshot {
        let new_parent = remap[&old_parent];
        for c in children {
            if let Some(nc) = remap.get(&c) {
                scene.arena.add_child(new_parent, *nc);
            }
        }
    }
    let mut roots = Vec::new();
    for r in loaded.roots {
        if let Some(nr) = remap.get(&r) {
            scene.arena.add_child(scene.root, *nr);
            roots.push(*nr);
        }
    }
    roots
}
