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
<<<<<<< Updated upstream
                parser.set_image_failed(req.index, "external image not supplied".into());
=======
                joints
                    .iter()
                    .map(|&j_idx| Bone {
                        node: node_ids.get(j_idx).copied().unwrap_or_default(),
                        inverse_bind: Matrix4::identity(),
                    })
                    .collect()
            };
            skeletons.push(Skeleton::new(bones));
        }

        // Parse animations.
        let animations_json = obj
            .get("animations")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut animations: Vec<AnimationClip> = Vec::with_capacity(animations_json.len());
        for (clip_i, an) in animations_json.iter().enumerate() {
            let Some(o) = an.as_object() else {
                continue;
            };
            let name = o
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or(&format!("Animation_{clip_i}"))
                .to_string();
            let channels = o
                .get("channels")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            let samplers = o
                .get("samplers")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            let mut tracks: Vec<KeyframeTrack> = Vec::new();
            let mut max_time = 0.0_f32;
            for ch in &channels {
                let Some(co) = ch.as_object() else {
                    continue;
                };
                let sampler_idx = match co.get("sampler").and_then(|v| v.as_u64()) {
                    Some(i) => i as usize,
                    None => continue,
                };
                let target = co.get("target").and_then(|v| v.as_object());
                let (node_idx, path) = match target {
                    Some(t) => (
                        t.get("node").and_then(|v| v.as_u64()).map(|n| n as usize),
                        t.get("path")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    ),
                    None => continue,
                };
                let Some(node_idx) = node_idx else {
                    continue;
                };
                let Some(obj_id) = node_ids.get(node_idx).copied() else {
                    continue;
                };
                let Some(samp) = samplers.get(sampler_idx).and_then(|v| v.as_object()) else {
                    continue;
                };
                let input_idx = match samp.get("input").and_then(|v| v.as_u64()) {
                    Some(i) => i as usize,
                    None => continue,
                };
                let output_idx = match samp.get("output").and_then(|v| v.as_u64()) {
                    Some(i) => i as usize,
                    None => continue,
                };
                let interp = match samp
                    .get("interpolation")
                    .and_then(|v| v.as_str())
                    .unwrap_or("LINEAR")
                {
                    "STEP" => Interpolation::Step,
                    "CUBICSPLINE" => Interpolation::Cubic,
                    _ => Interpolation::Linear,
                };

                let (_, times) =
                    match read_accessor_floats(input_idx, &accessors, &buffer_views, &buffers) {
                        Ok(t) => t,
                        Err(_) => continue,
                    };
                if let Some(&last) = times.last() {
                    if last > max_time {
                        max_time = last;
                    }
                }
                let (item_size, values) =
                    match read_accessor_floats(output_idx, &accessors, &buffer_views, &buffers) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                let track = match path.as_str() {
                    "translation" => {
                        let vs: Vec<Vector3> = values
                            .chunks_exact(3)
                            .map(|c| Vector3::new(c[0], c[1], c[2]))
                            .collect();
                        let mut t =
                            KeyframeTrack::vector(obj_id, TrackTarget::Position, times.clone(), vs);
                        t.interpolation = interp;
                        Some(t)
                    }
                    "rotation" => {
                        let vs: Vec<Quaternion> = values
                            .chunks_exact(4)
                            .map(|c| Quaternion::new(c[0], c[1], c[2], c[3]))
                            .collect();
                        let mut t = KeyframeTrack::quaternion(
                            obj_id,
                            TrackTarget::Quaternion,
                            times.clone(),
                            vs,
                        );
                        t.interpolation = interp;
                        Some(t)
                    }
                    "scale" => {
                        let vs: Vec<Vector3> = values
                            .chunks_exact(3)
                            .map(|c| Vector3::new(c[0], c[1], c[2]))
                            .collect();
                        let mut t =
                            KeyframeTrack::vector(obj_id, TrackTarget::Scale, times.clone(), vs);
                        t.interpolation = interp;
                        Some(t)
                    }
                    "weights" => {
                        // glTF morph weights: one scalar track per morph index,
                        // interleaved as [w0,w1,…,wN] per keyframe.
                        let morph_count = item_size.max(1);
                        let mut out = Vec::new();
                        for mi in 0..morph_count {
                            let vs: Vec<f32> = values
                                .iter()
                                .skip(mi)
                                .step_by(morph_count)
                                .copied()
                                .collect();
                            if vs.len() != times.len() && !vs.is_empty() {
                                // tolerate length mismatch by truncating
                            }
                            let mut t = KeyframeTrack::scalar(
                                obj_id,
                                TrackTarget::MorphWeight { index: mi },
                                times.clone(),
                                vs,
                            );
                            t.interpolation = interp;
                            out.push(t);
                        }
                        for t in out {
                            tracks.push(t);
                        }
                        None
                    }
                    _ => {
                        let _ = item_size;
                        None
                    }
                };
                if let Some(t) = track {
                    tracks.push(t);
                }
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======

// ---------- helpers ----------

fn parse_material(
    mj: &Value,
    gltf_textures: &[Option<Arc<Texture>>],
    gltf_textures_linear: &[Option<Arc<Texture>>],
) -> Material {
    let mut std_mat = StandardMaterial::default();
    let mut emissive_strength = 1.0_f32;
    // KHR_materials_clearcoat / KHR_materials_ior / KHR_materials_transmission /
    // KHR_materials_sheen / KHR_materials_iridescence — collected into a Physical.
    let mut clearcoat = 0.0_f32;
    let mut clearcoat_roughness = 0.0_f32;
    let mut ior = 1.5_f32;
    let mut transmission = 0.0_f32;
    let mut sheen = 0.0_f32;
    let mut sheen_color = Color::BLACK;
    let mut sheen_roughness = 1.0_f32;
    let mut iridescence = 0.0_f32;
    let mut iridescence_ior = 1.3_f32;
    let mut use_physical = false;
    let mut is_unlit = false;

    let lookup_tex = |o: &HashMap<String, Value>, key: &str| -> Option<Arc<Texture>> {
        let info = o.get(key).and_then(|v| v.as_object())?;
        let idx = info.get("index").and_then(|v| v.as_u64())? as usize;
        gltf_textures.get(idx).and_then(|t| t.clone())
    };
    // Normal, metallic-roughness and occlusion are data, not colour.
    let lookup_data = |o: &HashMap<String, Value>, key: &str| -> Option<Arc<Texture>> {
        let info = o.get(key).and_then(|v| v.as_object())?;
        let idx = info.get("index").and_then(|v| v.as_u64())? as usize;
        gltf_textures_linear.get(idx).and_then(|t| t.clone())
    };
    if let Some(m) = mj.as_object() {
        if let Some(pbr) = m.get("pbrMetallicRoughness").and_then(|v| v.as_object()) {
            if let Some(bcf) = pbr.get("baseColorFactor").and_then(|v| v.as_array()) {
                std_mat.color = Color::new(
                    bcf.first().and_then(|v| v.as_f32()).unwrap_or(1.0),
                    bcf.get(1).and_then(|v| v.as_f32()).unwrap_or(1.0),
                    bcf.get(2).and_then(|v| v.as_f32()).unwrap_or(1.0),
                );
                std_mat.opacity = bcf.get(3).and_then(|v| v.as_f32()).unwrap_or(1.0);
            }
            if let Some(r) = pbr.get("roughnessFactor").and_then(|v| v.as_f32()) {
                std_mat.roughness = r;
            }
            if let Some(mv) = pbr.get("metallicFactor").and_then(|v| v.as_f32()) {
                std_mat.metalness = mv;
            }
            std_mat.map = lookup_tex(pbr, "baseColorTexture");
            let mr = lookup_data(pbr, "metallicRoughnessTexture");
            std_mat.roughness_map = mr.clone();
            std_mat.metalness_map = mr;
        }
        std_mat.normal_map = lookup_data(m, "normalTexture");
        std_mat.ao_map = lookup_data(m, "occlusionTexture");
        std_mat.emissive_map = lookup_tex(m, "emissiveTexture");
        if let Some(ef) = m.get("emissiveFactor").and_then(|v| v.as_array()) {
            std_mat.emissive = Color::new(
                ef.first().and_then(|v| v.as_f32()).unwrap_or(0.0),
                ef.get(1).and_then(|v| v.as_f32()).unwrap_or(0.0),
                ef.get(2).and_then(|v| v.as_f32()).unwrap_or(0.0),
            );
        }
        // KHR_ extensions.
        if let Some(ext) = m.get("extensions").and_then(|v| v.as_object()) {
            if ext.contains_key("KHR_materials_unlit") {
                is_unlit = true;
            }
            if let Some(e) = ext
                .get("KHR_materials_emissive_strength")
                .and_then(|v| v.as_object())
            {
                if let Some(s) = e.get("emissiveStrength").and_then(|v| v.as_f32()) {
                    emissive_strength = s;
                }
            }
            if let Some(e) = ext
                .get("KHR_materials_clearcoat")
                .and_then(|v| v.as_object())
            {
                use_physical = true;
                if let Some(s) = e.get("clearcoatFactor").and_then(|v| v.as_f32()) {
                    clearcoat = s;
                }
                if let Some(s) = e.get("clearcoatRoughnessFactor").and_then(|v| v.as_f32()) {
                    clearcoat_roughness = s;
                }
            }
            if let Some(e) = ext.get("KHR_materials_ior").and_then(|v| v.as_object()) {
                use_physical = true;
                if let Some(s) = e.get("ior").and_then(|v| v.as_f32()) {
                    ior = s;
                }
            }
            if let Some(e) = ext
                .get("KHR_materials_transmission")
                .and_then(|v| v.as_object())
            {
                use_physical = true;
                if let Some(s) = e.get("transmissionFactor").and_then(|v| v.as_f32()) {
                    transmission = s;
                }
            }
            if let Some(e) = ext.get("KHR_materials_sheen").and_then(|v| v.as_object()) {
                use_physical = true;
                if let Some(s) = e.get("sheenColorFactor").and_then(|v| v.as_array()) {
                    sheen_color = Color::new(
                        s.first().and_then(|v| v.as_f32()).unwrap_or(0.0),
                        s.get(1).and_then(|v| v.as_f32()).unwrap_or(0.0),
                        s.get(2).and_then(|v| v.as_f32()).unwrap_or(0.0),
                    );
                    let max = sheen_color.r.max(sheen_color.g).max(sheen_color.b);
                    sheen = max;
                }
                if let Some(s) = e.get("sheenRoughnessFactor").and_then(|v| v.as_f32()) {
                    sheen_roughness = s;
                }
            }
            if let Some(e) = ext
                .get("KHR_materials_iridescence")
                .and_then(|v| v.as_object())
            {
                use_physical = true;
                if let Some(s) = e.get("iridescenceFactor").and_then(|v| v.as_f32()) {
                    iridescence = s;
                }
                if let Some(s) = e.get("iridescenceIor").and_then(|v| v.as_f32()) {
                    iridescence_ior = s;
                }
            }
        }
    }
    std_mat.emissive_intensity = emissive_strength;
    if is_unlit {
        Material::Basic(BasicMaterial::new(std_mat.color))
    } else if use_physical {
        let p = PhysicalMaterial {
            color: std_mat.color,
            emissive: std_mat.emissive,
            emissive_intensity: std_mat.emissive_intensity,
            roughness: std_mat.roughness,
            metalness: std_mat.metalness,
            opacity: std_mat.opacity,
            map: std_mat.map,
            normal_map: std_mat.normal_map,
            roughness_map: std_mat.roughness_map,
            metalness_map: std_mat.metalness_map,
            ao_map: std_mat.ao_map,
            emissive_map: std_mat.emissive_map,
            clearcoat,
            clearcoat_roughness,
            ior,
            transmission,
            sheen,
            sheen_color,
            sheen_roughness,
            iridescence,
            iridescence_ior,
            ..PhysicalMaterial::default()
        };
        Material::Physical(p)
    } else {
        Material::Standard(std_mat)
    }
}

/// Parse KHR_lights_punctual: walks nodes for `extensions.KHR_lights_punctual.light`
/// and turns each into a `Light` attached to that node. The lights array itself
/// is stored at the root `extensions.KHR_lights_punctual.lights`.
#[allow(dead_code)]
pub fn parse_punctual_lights(root: &Value) -> Vec<Light> {
    let Some(obj) = root.as_object() else {
        return Vec::new();
    };
    let Some(exts) = obj.get("extensions").and_then(|v| v.as_object()) else {
        return Vec::new();
    };
    let Some(kp) = exts.get("KHR_lights_punctual").and_then(|v| v.as_object()) else {
        return Vec::new();
    };
    let Some(lights) = kp.get("lights").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(lights.len());
    for l in lights {
        let Some(lo) = l.as_object() else {
            continue;
        };
        let ty = lo
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("directional");
        let color = lo
            .get("color")
            .and_then(|v| v.as_array())
            .map(|c| {
                Color::new(
                    c.first().and_then(|v| v.as_f32()).unwrap_or(1.0),
                    c.get(1).and_then(|v| v.as_f32()).unwrap_or(1.0),
                    c.get(2).and_then(|v| v.as_f32()).unwrap_or(1.0),
                )
            })
            .unwrap_or(Color::WHITE);
        let intensity = lo.get("intensity").and_then(|v| v.as_f32()).unwrap_or(1.0);
        let range = lo.get("range").and_then(|v| v.as_f32()).unwrap_or(0.0);
        match ty {
            "directional" => {
                out.push(Light::Directional(DirectionalLight::new(color, intensity)));
            }
            "point" => {
                let mut p = PointLight::new(color, intensity);
                p.distance = range;
                out.push(Light::Point(p));
            }
            "spot" => {
                let mut s = SpotLight::new(color, intensity);
                s.distance = range;
                if let Some(spot) = lo.get("spot").and_then(|v| v.as_object()) {
                    if let Some(a) = spot.get("outerConeAngle").and_then(|v| v.as_f32()) {
                        s.angle = a;
                    }
                    if let Some(a) = spot.get("innerConeAngle").and_then(|v| v.as_f32()) {
                        s.penumbra = if s.angle > 0.0 {
                            (1.0 - a / s.angle).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                    }
                }
                out.push(Light::Spot(s));
            }
            "ambient" => out.push(Light::Ambient(AmbientLight::new(color, intensity))),
            _ => {}
        }
    }
    out
}

fn apply_trs(obj: &mut Object3D, n: &HashMap<String, Value>) {
    if let Some(m) = n.get("matrix").and_then(|v| v.as_array()) {
        if m.len() == 16 {
            let mut e = [0.0f32; 16];
            for (i, v) in m.iter().enumerate() {
                e[i] = v.as_f32().unwrap_or(0.0);
            }
            let mat = Matrix4 { elements: e };
            let (p, q, s) = mat.decompose();
            obj.position = p;
            obj.quaternion = q;
            obj.scale = s;
            return;
        }
    }
    if let Some(t) = n.get("translation").and_then(|v| v.as_array()) {
        obj.position = Vector3::new(
            t.first().and_then(|v| v.as_f32()).unwrap_or(0.0),
            t.get(1).and_then(|v| v.as_f32()).unwrap_or(0.0),
            t.get(2).and_then(|v| v.as_f32()).unwrap_or(0.0),
        );
    }
    if let Some(r) = n.get("rotation").and_then(|v| v.as_array()) {
        obj.quaternion = Quaternion::new(
            r.first().and_then(|v| v.as_f32()).unwrap_or(0.0),
            r.get(1).and_then(|v| v.as_f32()).unwrap_or(0.0),
            r.get(2).and_then(|v| v.as_f32()).unwrap_or(0.0),
            r.get(3).and_then(|v| v.as_f32()).unwrap_or(1.0),
        );
    }
    if let Some(s) = n.get("scale").and_then(|v| v.as_array()) {
        obj.scale = Vector3::new(
            s.first().and_then(|v| v.as_f32()).unwrap_or(1.0),
            s.get(1).and_then(|v| v.as_f32()).unwrap_or(1.0),
            s.get(2).and_then(|v| v.as_f32()).unwrap_or(1.0),
        );
    }
}

fn primitive_geometry(
    prim: &Value,
    accessors: &[Value],
    buffer_views: &[Value],
    buffers: &[Vec<u8>],
) -> Result<BufferGeometry, GltfError> {
    let prim_o = prim
        .as_object()
        .ok_or(GltfError::Json("primitive not object"))?;
    // A Draco primitive keeps its geometry in the extension, not in the
    // accessors. The accessors are still there and still describe the result,
    // but their buffer views hold the compressed stream — reading them as floats
    // is how a decoder that ignores the extension produces silent nonsense.
    if let Some(draco) = prim_o
        .get("extensions")
        .and_then(|v| v.as_object())
        .and_then(|e| e.get("KHR_draco_mesh_compression"))
    {
        return draco_geometry(draco, buffer_views, buffers);
    }
    let attrs = prim_o
        .get("attributes")
        .and_then(|v| v.as_object())
        .ok_or(GltfError::MissingField("attributes"))?;
    let mut g = BufferGeometry::new();
    for (name, acc_idx) in attrs.iter() {
        let idx = acc_idx.as_u64().ok_or(GltfError::BadAccessor)? as usize;
        let (item_size, data) = read_accessor_floats(idx, accessors, buffer_views, buffers)?;
        let gltf_name = name.as_str();
        let our_name = match gltf_name {
            "POSITION" => "position",
            "NORMAL" => "normal",
            "TEXCOORD_0" => "uv",
            "COLOR_0" => "color",
            "TANGENT" => "tangent",
            other => return Err(GltfError::UnsupportedAttribute(other.into())),
        };
        g.set_attribute(our_name, BufferAttribute::new(data, item_size));
    }
    if let Some(idx_acc) = prim_o.get("indices").and_then(|v| v.as_u64()) {
        let indices = read_accessor_u32(idx_acc as usize, accessors, buffer_views, buffers)?;
        g.set_index(indices);
    }
    Ok(g)
}

/// Geometry from a `KHR_draco_mesh_compression` primitive.
///
/// The extension's own `attributes` map is semantic → Draco *unique id*, not
/// semantic → accessor: the ids are how a decoded attribute is identified, and
/// they need not match either the accessor indices or the order the attributes
/// were written in. Matching by position instead happens to work on files whose
/// encoder wrote them in the obvious order, which is most of them, and fails
/// silently on the rest by swapping normals for texture coordinates.
fn draco_geometry(
    draco: &Value,
    buffer_views: &[Value],
    buffers: &[Vec<u8>],
) -> Result<BufferGeometry, GltfError> {
    let o = draco
        .as_object()
        .ok_or(GltfError::Json("draco extension not object"))?;
    let view_idx = o
        .get("bufferView")
        .and_then(|v| v.as_u64())
        .ok_or(GltfError::MissingField("bufferView"))? as usize;
    let view = buffer_views
        .get(view_idx)
        .and_then(|v| v.as_object())
        .ok_or(GltfError::BadAccessor)?;
    let buffer_idx = view.get("buffer").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let offset = view.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let length = view
        .get("byteLength")
        .and_then(|v| v.as_u64())
        .ok_or(GltfError::BadAccessor)? as usize;
    let bytes = buffers
        .get(buffer_idx)
        .and_then(|b| b.get(offset..offset + length))
        .ok_or(GltfError::BadAccessor)?;

    let mesh = super::draco::decode_mesh(bytes)?;
    let attrs = o
        .get("attributes")
        .and_then(|v| v.as_object())
        .ok_or(GltfError::MissingField("attributes"))?;

    let mut g = BufferGeometry::new();
    for (name, id) in attrs.iter() {
        let unique_id = id.as_u64().ok_or(GltfError::BadAccessor)? as u32;
        let our_name = match name.as_str() {
            "POSITION" => "position",
            "NORMAL" => "normal",
            "TEXCOORD_0" => "uv",
            "COLOR_0" => "color",
            "TANGENT" => "tangent",
            other => return Err(GltfError::UnsupportedAttribute(other.into())),
        };
        let attribute = mesh
            .attributes
            .iter()
            .find(|a| a.descriptor.unique_id == unique_id)
            .ok_or(GltfError::BadAccessor)?;
        g.set_attribute(
            our_name,
            BufferAttribute::new(attribute.as_f32(), attribute.num_components()),
        );
    }
    g.set_index(mesh.indices());
    Ok(g)
}

fn read_accessor_floats(
    idx: usize,
    accessors: &[Value],
    buffer_views: &[Value],
    buffers: &[Vec<u8>],
) -> Result<(usize, Vec<f32>), GltfError> {
    let a = accessors
        .get(idx)
        .and_then(|v| v.as_object())
        .ok_or(GltfError::BadAccessor)?;
    let view_idx = a
        .get("bufferView")
        .and_then(|v| v.as_u64())
        .ok_or(GltfError::BadAccessor)? as usize;
    let count = a
        .get("count")
        .and_then(|v| v.as_u64())
        .ok_or(GltfError::BadAccessor)? as usize;
    let ty = a
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or(GltfError::BadAccessor)?;
    let item_size = match ty {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        _ => return Err(GltfError::BadAccessor),
    };
    let comp_type = a
        .get("componentType")
        .and_then(|v| v.as_u64())
        .ok_or(GltfError::BadAccessor)? as u32;
    let view = buffer_views
        .get(view_idx)
        .and_then(|v| v.as_object())
        .ok_or(GltfError::BadAccessor)?;
    let buf_idx = view
        .get("buffer")
        .and_then(|v| v.as_u64())
        .ok_or(GltfError::BadAccessor)? as usize;
    let byte_offset_view = view.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let byte_offset_acc = a.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let buffer = buffers.get(buf_idx).ok_or(GltfError::BadAccessor)?;
    let start = byte_offset_view + byte_offset_acc;
    let bytes = buffer.get(start..).ok_or(GltfError::BadAccessor)?;

    let width = component_width(comp_type).ok_or(GltfError::BadAccessor)?;
    // A view may interleave several attributes, in which case `byteStride` is
    // the distance between elements and *not* the size of one. Reading it as
    // packed walks off into the neighbouring attribute, so a mesh comes back
    // with positions made partly of normals.
    let stride = view
        .get("byteStride")
        .and_then(|v| v.as_u64())
        .map(|s| s as usize)
        .filter(|s| *s > 0)
        .unwrap_or(item_size * width);
    // `normalized` says whether an integer is a fraction of its range or a
    // number. `KHR_mesh_quantization` writes positions as plain shorts, and
    // dividing those by 65535 collapses the mesh to a speck at the origin.
    let normalized = a
        .get("normalized")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let mut out = Vec::with_capacity(count * item_size);
    for element in 0..count {
        for component in 0..item_size {
            let at = element * stride + component * width;
            let slice = bytes.get(at..at + width).ok_or(GltfError::BadAccessor)?;
            out.push(read_component(comp_type, slice, normalized)?);
        }
    }
    Ok((item_size, out))
}

fn component_width(comp_type: u32) -> Option<usize> {
    Some(match comp_type {
        5120 | 5121 => 1,
        5122 | 5123 => 2,
        5125 | 5126 => 4,
        _ => return None,
    })
}

/// One component, as the float this crate's attributes hold.
///
/// A normalized integer is a fraction of its range — and a *signed* one maps to
/// [-1,1] with the divisor being `MAX`, not `MIN`, so the most negative value
/// clamps rather than reaching -1.00003.
fn read_component(comp_type: u32, bytes: &[u8], normalized: bool) -> Result<f32, GltfError> {
    Ok(match comp_type {
        5126 => f32::from_le_bytes(bytes.try_into().map_err(|_| GltfError::BadAccessor)?),
        5125 => u32::from_le_bytes(bytes.try_into().map_err(|_| GltfError::BadAccessor)?) as f32,
        5123 => {
            let v = u16::from_le_bytes(bytes.try_into().map_err(|_| GltfError::BadAccessor)?);
            if normalized {
                v as f32 / 65535.0
            } else {
                v as f32
            }
        }
        5122 => {
            let v = i16::from_le_bytes(bytes.try_into().map_err(|_| GltfError::BadAccessor)?);
            if normalized {
                (v as f32 / 32767.0).max(-1.0)
            } else {
                v as f32
            }
        }
        5121 => {
            let v = bytes[0];
            if normalized {
                v as f32 / 255.0
            } else {
                v as f32
            }
        }
        5120 => {
            let v = bytes[0] as i8;
            if normalized {
                (v as f32 / 127.0).max(-1.0)
            } else {
                v as f32
            }
        }
        _ => return Err(GltfError::BadAccessor),
    })
}

/// Replace every `EXT_meshopt_compression` buffer view with its decoded bytes.
///
/// The extension names a slice of a buffer, how many elements it holds, how
/// wide each is, which codec, and an optional filter. Each becomes a buffer of
/// its own so the view can point at offset zero and the rest of the loader does
/// not have to know.
fn decompress_meshopt(
    buffer_views: &mut [Value],
    buffers: &mut Vec<Vec<u8>>,
) -> Result<(), GltfError> {
    use crate::loaders::meshopt;
    for view in buffer_views.iter_mut() {
        let Some(ext) = view
            .get("extensions")
            .and_then(|e| e.get("EXT_meshopt_compression"))
            .and_then(|e| e.as_object())
        else {
            continue;
        };
        let number = |key: &str| ext.get(key).and_then(|v| v.as_u64()).map(|v| v as usize);
        let buffer = number("buffer").ok_or(GltfError::BadAccessor)?;
        let offset = number("byteOffset").unwrap_or(0);
        let length = number("byteLength").ok_or(GltfError::BadAccessor)?;
        let stride = number("byteStride").ok_or(GltfError::BadAccessor)?;
        let count = number("count").ok_or(GltfError::BadAccessor)?;
        let mode = ext
            .get("mode")
            .and_then(|v| v.as_str())
            .and_then(meshopt::Mode::parse)
            .ok_or(GltfError::BadAccessor)?;
        let filter = ext
            .get("filter")
            .and_then(|v| v.as_str())
            .map(meshopt::Filter::parse)
            .unwrap_or(meshopt::Filter::None);

        let source = buffers
            .get(buffer)
            .and_then(|b| b.get(offset..offset + length))
            .ok_or(GltfError::BadAccessor)?;
        let decoded = meshopt::decode(count, stride, source, mode, filter)
            .map_err(|_| GltfError::BadAccessor)?;

        buffers.push(decoded);
        let at = buffers.len() - 1;
        let Some(object) = view.as_object_mut() else {
            continue;
        };
        object.insert("buffer".into(), Value::Number(at as f64));
        object.insert("byteOffset".into(), Value::Number(0.0));
        object.insert("byteLength".into(), Value::Number((count * stride) as f64));
        object.insert("byteStride".into(), Value::Number(stride as f64));
    }
    Ok(())
}

fn read_accessor_u32(
    idx: usize,
    accessors: &[Value],
    buffer_views: &[Value],
    buffers: &[Vec<u8>],
) -> Result<Vec<u32>, GltfError> {
    let a = accessors
        .get(idx)
        .and_then(|v| v.as_object())
        .ok_or(GltfError::BadAccessor)?;
    let view_idx = a
        .get("bufferView")
        .and_then(|v| v.as_u64())
        .ok_or(GltfError::BadAccessor)? as usize;
    let count = a
        .get("count")
        .and_then(|v| v.as_u64())
        .ok_or(GltfError::BadAccessor)? as usize;
    let comp_type = a
        .get("componentType")
        .and_then(|v| v.as_u64())
        .ok_or(GltfError::BadAccessor)? as u32;
    let view = buffer_views
        .get(view_idx)
        .and_then(|v| v.as_object())
        .ok_or(GltfError::BadAccessor)?;
    let buf_idx = view
        .get("buffer")
        .and_then(|v| v.as_u64())
        .ok_or(GltfError::BadAccessor)? as usize;
    let byte_offset_view = view.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let byte_offset_acc = a.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let buffer = buffers.get(buf_idx).ok_or(GltfError::BadAccessor)?;
    let start = byte_offset_view + byte_offset_acc;
    let bytes = &buffer[start..];
    let mut out = Vec::with_capacity(count);
    match comp_type {
        5121 => {
            for &b in &bytes[..count] {
                out.push(b as u32);
            }
        } // UBYTE
        5123 => {
            // USHORT
            for i in 0..count {
                let off = i * 2;
                out.push(u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32);
            }
        }
        5125 => {
            // UINT
            for i in 0..count {
                let off = i * 4;
                out.push(u32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]));
            }
        }
        _ => return Err(GltfError::BadAccessor),
    }
    Ok(out)
}

fn decode_buffers(
    root: &Value,
    external: &HashMap<usize, Vec<u8>>,
) -> Result<Vec<Vec<u8>>, GltfError> {
    let Some(obj) = root.as_object() else {
        return Ok(Vec::new());
    };
    let buffers_json = obj
        .get("buffers")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::with_capacity(buffers_json.len());
    for (i, b) in buffers_json.iter().enumerate() {
        let Some(b_obj) = b.as_object() else {
            return Err(GltfError::BadAccessor);
        };
        if let Some(uri) = b_obj.get("uri").and_then(|v| v.as_str()) {
            if let Some(rest) = uri.strip_prefix("data:application/octet-stream;base64,") {
                out.push(decode_base64(rest)?);
            } else if let Some(rest) = uri.strip_prefix("data:application/gltf-buffer;base64,") {
                out.push(decode_base64(rest)?);
            } else if let Some(ext) = external.get(&i) {
                out.push(ext.clone());
            } else {
                return Err(GltfError::MissingField("external buffer not supplied"));
            }
        } else if let Some(ext) = external.get(&i) {
            out.push(ext.clone());
        } else {
            return Err(GltfError::MissingField("buffer missing uri"));
        }
    }
    Ok(out)
}

fn decode_base64(s: &str) -> Result<Vec<u8>, GltfError> {
    // Minimal RFC4648 base64 decoder, padding-tolerant.
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for &b in bytes {
        let v = match b {
            b'A'..=b'Z' => (b - b'A') as u32,
            b'a'..=b'z' => (b - b'a' + 26) as u32,
            b'0'..=b'9' => (b - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            b'=' | b'\n' | b'\r' | b' ' | b'\t' => continue,
            _ => return Err(GltfError::BadBase64),
        };
        buf = (buf << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod compression {
    use super::*;
    use crate::core::ObjectKind;

    const PLAIN: &[u8] = include_bytes!("testdata/sphere_plain.glb");
    const MESHOPT: &[u8] = include_bytes!("testdata/sphere_meshopt.glb");
    const DRACO: &[u8] = include_bytes!("testdata/sphere_draco.glb");

    /// The one mesh in a file, as positions, normals and indices.
    fn mesh(bytes: &[u8]) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<u32>>) {
        let scene = GltfLoader::parse_glb(bytes).expect("parses");
        let (_, object) = scene
            .arena
            .nodes
            .iter()
            .find(|(_, o)| matches!(o.kind, ObjectKind::Mesh(_)))
            .expect("a mesh");
        let ObjectKind::Mesh(m) = &object.kind else {
            unreachable!()
        };
        (
            m.geometry.get_attribute("position").unwrap().array.clone(),
            m.geometry.get_attribute("normal").map(|a| a.array.clone()),
            m.geometry.index.clone(),
        )
    }

    /// A meshopt-compressed file reads, and reads as the same mesh.
    ///
    /// These three files are one sphere written three ways, so the compressed
    /// ones have to come back as the uncompressed one — which is a far stronger
    /// statement than "it decoded without erroring".
    ///
    /// The comparison is by nearest neighbour, not by index: `gltf-transform`
    /// reorders vertices for cache locality, and sorting to compensate does not
    /// work either, because quantization perturbs each vertex just enough to
    /// make a lexicographic sort pair up unrelated ones.
    #[test]
    fn a_meshopt_file_reads_as_the_same_mesh() {
        let (plain, plain_normals, plain_index) = mesh(PLAIN);
        let (ours, normals, index) = mesh(MESHOPT);

        assert_eq!(ours.len(), plain.len(), "same vertex count");
        assert_eq!(
            index.as_ref().map(Vec::len),
            plain_index.as_ref().map(Vec::len),
            "same index count"
        );
        assert!(normals.is_some(), "normals survived the filter");

        // Every decoded vertex is a vertex of the original, within the error
        // 16-bit quantization allows over a 2-unit box.
        let tolerance = 2.0 / 32767.0 * 4.0;
        for v in ours.chunks_exact(3) {
            let nearest = plain
                .chunks_exact(3)
                .map(|p| {
                    ((p[0] - v[0]).powi(2) + (p[1] - v[1]).powi(2) + (p[2] - v[2]).powi(2)).sqrt()
                })
                .fold(f32::MAX, f32::min);
            assert!(
                nearest < tolerance,
                "decoded vertex {v:?} is {nearest} from anything in the original"
            );
        }
        // And nothing landed off the sphere, which a stride or plane slip does.
        for v in ours.chunks_exact(3) {
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((r - 1.0).abs() < 1e-3, "{v:?} is at radius {r}, not 1");
        }
        let _ = plain_normals;
    }

    /// The indices decode to triangles that index real vertices.
    #[test]
    fn meshopt_indices_are_a_valid_triangle_list() {
        let (positions, _, index) = mesh(MESHOPT);
        let index = index.expect("indices");
        let vertices = positions.len() / 3;
        assert_eq!(index.len() % 3, 0, "whole triangles");
        for &i in &index {
            assert!(
                (i as usize) < vertices,
                "index {i} past {vertices} vertices"
            );
        }
        // A closed sphere has no degenerate triangles.
        for t in index.chunks_exact(3) {
            assert!(
                t[0] != t[1] && t[1] != t[2] && t[0] != t[2],
                "degenerate {t:?}"
            );
        }
    }

    /// A file requiring something unimplemented is refused, by name.
    ///
    /// `extensionsRequired` means the asset cannot be read correctly without it.
    /// Without this check such a file fails only if it happens to become
    /// unreadable, which is luck: one requiring `KHR_texture_basisu` used to load
    /// with the texture silently missing. The name is carried in the error rather
    /// than guessed at from the damage.
    #[test]
    fn a_file_requiring_an_unimplemented_extension_is_refused() {
        let json = r#"{"asset":{"version":"2.0"},
            "extensionsRequired":["EXT_not_a_real_extension"],
            "scene":0,"scenes":[{"nodes":[]}],"nodes":[]}"#;
        let root = crate::loaders::json::parse(json).expect("parses");
        match GltfLoader::build_for_test(&root) {
            Err(GltfError::UnsupportedExtension(name)) => {
                assert_eq!(name, "EXT_not_a_real_extension");
            }
            Err(other) => panic!("expected a named refusal, got {other:?}"),
            Ok(_) => panic!("a file requiring an unknown extension must not load"),
        }
    }

    /// An extension merely *used* does not stop the load.
    ///
    /// The distinction is the whole point: `extensionsUsed` is a refinement a
    /// reader may ignore, `extensionsRequired` is not. Treating the first as
    /// fatal would refuse most real files.
    #[test]
    fn an_extension_only_used_is_not_fatal() {
        let json = r#"{"asset":{"version":"2.0"},
            "extensionsUsed":["KHR_materials_variants","SOMETHING_INVENTED"],
            "scene":0,"scenes":[{"nodes":[]}],"nodes":[]}"#;
        let root = crate::loaders::json::parse(json).expect("parses");
        assert!(
            GltfLoader::build_for_test(&root).is_ok(),
            "a used-but-not-required extension must not stop the load"
        );
    }

    /// And one that is required *and* implemented loads.
    #[test]
    fn a_required_extension_that_is_implemented_loads() {
        // The meshopt fixture requires both meshopt and quantization.
        assert!(GltfLoader::parse_glb(MESHOPT).is_ok());
    }

    /// A Draco primitive decodes to the same mesh as its uncompressed twin.
    ///
    /// Both fixtures are the same sphere, one compressed and one not, so this
    /// compares a decode against geometry that came from somewhere else entirely
    /// — not against the decoder's own output, and not against a number written
    /// down from a previous run.
    ///
    /// The comparison has to be order-independent: Draco emits points in the
    /// order its traversal reached them, which is not the order the plain file
    /// stores them in. So the two are compared as *sets* of positions and as sets
    /// of triangles, keyed on the quantization grid. That is a real check —
    /// swapping two attributes, or dropping a prediction, moves points off the
    /// set — while an index-by-index comparison would fail on a correct decode.
    #[test]
    fn a_draco_primitive_decodes_to_the_same_mesh_as_its_plain_twin() {
        let (dp, dn, di) = mesh(DRACO);
        let (pp, pn, pi) = mesh(PLAIN);
        assert_eq!(dp.len(), pp.len(), "the same number of position floats");
        assert_eq!(dp.len() / 3, 423);
        assert_eq!(di.as_ref().map(|i| i.len()), Some(2160));
        assert_eq!(pi.as_ref().map(|i| i.len()), Some(2160));
        assert!(dn.is_some() && pn.is_some(), "both carry normals");

        // Quantization moves every position by up to half a grid step, so the
        // two files are compared by proximity rather than by equality. Snapping
        // to a grid instead looks simpler and is wrong: a value sitting near a
        // cell boundary rounds one way in one file and the other way in the
        // other, and about a sixth of these do.
        let dedup = |v: &[f32]| -> Vec<[f32; 3]> {
            let mut out: Vec<[f32; 3]> = Vec::new();
            for p in v.chunks(3) {
                let p = [p[0], p[1], p[2]];
                if !out.contains(&p) {
                    out.push(p);
                }
            }
            out
        };
        let plain_unique = dedup(&pp);
        assert_eq!(
            plain_unique.len(),
            362,
            "distinct positions in the plain file"
        );
        // 423 points over 362 positions: the texture seam gives one position
        // several points, each with its own texture coordinates. Both files agree
        // on that, which is itself a check on the point assignment.
        assert_eq!(dedup(&dp).len(), 362, "distinct decoded positions");

        let nearest = |p: &[f32]| -> (usize, f32) {
            let mut best = (0usize, f32::MAX);
            for (i, q) in plain_unique.iter().enumerate() {
                let d = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2);
                if d < best.1 {
                    best = (i, d);
                }
            }
            (best.0, best.1.sqrt())
        };
        let mut worst = 0.0f32;
        let mut canonical = Vec::with_capacity(dp.len() / 3);
        for p in dp.chunks(3) {
            let (i, d) = nearest(p);
            worst = worst.max(d);
            canonical.push(i);
        }
        assert!(
            worst < 2e-3,
            "a decoded position is {worst} from the nearest in the plain file"
        );
        // Every plain position is accounted for, so the decode did not collapse
        // two vertices into one and duplicate a third.
        let hit: std::collections::BTreeSet<usize> = canonical.iter().copied().collect();
        assert_eq!(hit.len(), 362, "the decode does not cover every position");

        // And the faces, as sorted triples of those canonical positions, so
        // winding and starting corner do not matter.
        let plain_canonical: Vec<usize> = pp.chunks(3).map(|p| nearest(p).0).collect();
        let faces = |idx: &[u32], canon: &[usize]| -> std::collections::BTreeSet<[usize; 3]> {
            idx.chunks(3)
                .map(|t| {
                    let mut k = [
                        canon[t[0] as usize],
                        canon[t[1] as usize],
                        canon[t[2] as usize],
                    ];
                    k.sort();
                    k
                })
                .collect()
        };
        let df = faces(di.as_ref().unwrap(), &canonical);
        let pf = faces(pi.as_ref().unwrap(), &plain_canonical);
        assert_eq!(df.len(), 720, "720 distinct triangles decoded");
        assert_eq!(df, pf, "the decoded faces are not the plain file's faces");
    }

    /// The Draco extension is matched by unique id, not by position.
    ///
    /// `KHR_draco_mesh_compression` maps each semantic to a Draco *unique id*.
    /// Reordering the map must change nothing, because the ids still name the
    /// same attributes — a loader that paired them up by position would return
    /// normals where texture coordinates belong.
    #[test]
    fn draco_attributes_are_matched_by_id_not_order() {
        let (positions, normals, _) = mesh(DRACO);
        // Normals of a unit sphere point the same way as its positions, which is
        // what makes a swap detectable at all here.
        let normals = normals.expect("normals");
        assert_eq!(normals.len(), positions.len());
        let mut worst = 0.0f32;
        for (p, n) in positions.chunks(3).zip(normals.chunks(3)) {
            let dot = p[0] * n[0] + p[1] * n[1] + p[2] * n[2];
            worst = worst.max((dot - 1.0).abs());
        }
        assert!(
            worst < 0.05,
            "normals disagree with positions by {worst} — attributes are crossed"
        );
    }

    /// The uncompressed twin still reads, unchanged by the accessor rework.
    #[test]
    fn the_plain_file_still_reads() {
        let (positions, normals, index) = mesh(PLAIN);
        assert_eq!(positions.len() / 3, 423);
        assert!(normals.is_some());
        assert_eq!(index.map(|i| i.len()), Some(2160));
    }

    /// A truncated compressed stream is an error, not a panic.
    #[test]
    fn a_truncated_meshopt_stream_does_not_panic() {
        for cut in [16, 64, 256, 1024, 4096, 7000] {
            if cut < MESHOPT.len() {
                let _ = GltfLoader::parse_glb(&MESHOPT[..cut]);
            }
        }
    }
}
>>>>>>> Stashed changes
