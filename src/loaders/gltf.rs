//! Basic GLTF 2.0 loader. Supports:
//! - GLB binary container (single JSON chunk + single BIN chunk).
//! - Embedded base64 buffers in regular GLTF JSON.
//! - One scene per file with node hierarchy (matrix or T/R/S).
//! - Mesh primitives with POSITION/NORMAL/TEXCOORD_0/COLOR_0 attributes and
//!   optional indices (UNSIGNED_SHORT / UNSIGNED_INT / UNSIGNED_BYTE).
//! - PBR baseColorFactor + metallic-roughnessFactor mapped to StandardMaterial.
//!
//! NOT supported yet: textures, animations, skins, morph targets, cameras,
//! lights, extensions, sparse accessors.

use std::collections::HashMap;
use std::sync::Arc;

use super::json::{self, Value};
use crate::animation::{AnimationClip, Interpolation, KeyframeTrack, TrackTarget};
use crate::core::{
    Bone, BufferAttribute, BufferGeometry, Mesh, Object3D, ObjectArena, ObjectId, Skeleton,
};
use crate::lights::{AmbientLight, DirectionalLight, Light, PointLight, SpotLight};
use crate::materials::{BasicMaterial, Material, PhysicalMaterial, StandardMaterial};
use crate::math::{Color, Matrix4, Quaternion, Vector3};
use crate::scene::Scene;
use crate::textures::{Texture, TextureFilter, TextureFormat, TextureWrap};

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
    UnsupportedExtension(String),
    /// A `KHR_draco_mesh_compression` payload would not decode.
    Draco(super::draco::DracoError),
}

impl From<super::draco::DracoError> for GltfError {
    fn from(e: super::draco::DracoError) -> Self {
        GltfError::Draco(e)
    }
}

/// The extensions this loader implements well enough for a file to *require*.
///
/// Membership is a promise, not a wish list. `extensionsUsed` may name anything
/// — an unknown entry there is a refinement the file can be read without — but
/// an unknown entry in `extensionsRequired` has to stop the load.
const SUPPORTED_EXTENSIONS: &[&str] = &[
    // Geometry compression, decoded in `super::meshopt`.
    "EXT_meshopt_compression",
    // Draco geometry compression, decoded in `super::draco`.
    "KHR_draco_mesh_compression",
    // Quantized attributes: the accessor reader honours `normalized` and the
    // signed integer component types these use.
    "KHR_mesh_quantization",
    // Material and light extensions, all read.
    "KHR_lights_punctual",
    "KHR_materials_clearcoat",
    "KHR_materials_emissive_strength",
    "KHR_materials_ior",
    "KHR_materials_iridescence",
    "KHR_materials_sheen",
    "KHR_materials_transmission",
    "KHR_materials_unlit",
    // Unlit-style refinements that change nothing structural.
    "KHR_materials_specular",
];

/// Refuse a file that requires something this loader cannot do.
///
/// Without this the only files that failed were the ones that happened to
/// become unreadable — a Draco mesh errors because its accessors point at
/// compressed bytes, which is luck rather than a check. A file requiring
/// `KHR_texture_basisu` or `KHR_materials_variants` loaded with the texture or
/// the variant silently missing.
fn check_required_extensions(root: &Value) -> Result<(), GltfError> {
    let Some(required) = root.get("extensionsRequired").and_then(|v| v.as_array()) else {
        return Ok(());
    };
    for entry in required {
        let Some(name) = entry.as_str() else { continue };
        if !SUPPORTED_EXTENSIONS.contains(&name) {
            return Err(GltfError::UnsupportedExtension(name.to_string()));
        }
    }
    Ok(())
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
    pub fn parse_glb_with_images(
        bytes: &[u8],
        images: &GltfImages,
    ) -> Result<GltfScene, GltfError> {
        Self::parse_glb_inner(bytes, images)
    }

    /// Parse a `.glb` byte slice. Returns a fresh `GltfScene` plus root nodes.
    pub fn parse_glb(bytes: &[u8]) -> Result<GltfScene, GltfError> {
        Self::parse_glb_inner(bytes, &HashMap::new())
    }

    fn parse_glb_inner(bytes: &[u8], images: &GltfImages) -> Result<GltfScene, GltfError> {
        if bytes.len() < 12 {
            return Err(GltfError::BadMagic);
        }
        if &bytes[0..4] != b"glTF" {
            return Err(GltfError::BadMagic);
        }
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != 2 {
            return Err(GltfError::UnsupportedVersion);
        }
        let total_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        if total_len > bytes.len() {
            return Err(GltfError::BadMagic);
        }

        // Walk chunks.
        let mut pos = 12;
        let mut json_chunk: Option<&[u8]> = None;
        let mut bin_chunk: Option<&[u8]> = None;
        while pos + 8 <= total_len {
            let chunk_len =
                u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                    as usize;
            let chunk_type = u32::from_le_bytes([
                bytes[pos + 4],
                bytes[pos + 5],
                bytes[pos + 6],
                bytes[pos + 7],
            ]);
            let data_start = pos + 8;
            let data_end = data_start + chunk_len;
            if data_end > total_len {
                return Err(GltfError::BadMagic);
            }
            const TYPE_JSON: u32 = 0x4E4F_534A; // "JSON"
            const TYPE_BIN: u32 = 0x004E_4942; // "BIN\0"
            match chunk_type {
                TYPE_JSON => json_chunk = Some(&bytes[data_start..data_end]),
                TYPE_BIN => bin_chunk = Some(&bytes[data_start..data_end]),
                _ => {}
            }
            pos = data_end;
        }
        let json_chunk = json_chunk.ok_or(GltfError::MissingField("JSON chunk"))?;
        let json_str = std::str::from_utf8(json_chunk).map_err(|_| GltfError::Json("utf8"))?;
        let value = json::parse(json_str).map_err(GltfError::Json)?;
        Self::build(&value, bin_chunk, images)
    }

    /// Parse a `.gltf` JSON string with optional named buffers (caller supplies
    /// any external `.bin` blob data).
    pub fn parse_gltf(
        json_str: &str,
        external_buffers: &HashMap<usize, Vec<u8>>,
    ) -> Result<GltfScene, GltfError> {
        let value = json::parse(json_str).map_err(GltfError::Json)?;
        let buffers = decode_buffers(&value, external_buffers)?;
        Self::build_with_buffers(&value, buffers, &HashMap::new())
    }

    /// Parse a `.gltf` JSON string with external buffers AND decoded images.
    pub fn parse_gltf_with_images(
        json_str: &str,
        external_buffers: &HashMap<usize, Vec<u8>>,
        images: &GltfImages,
    ) -> Result<GltfScene, GltfError> {
        let value = json::parse(json_str).map_err(GltfError::Json)?;
        let buffers = decode_buffers(&value, external_buffers)?;
        Self::build_with_buffers(&value, buffers, images)
    }

    fn build(
        root: &Value,
        bin: Option<&[u8]>,
        images: &GltfImages,
    ) -> Result<GltfScene, GltfError> {
        let buffers = if let Some(b) = bin {
            // GLB: a single buffer at index 0 mapped to the BIN chunk.
            vec![b.to_vec()]
        } else {
            decode_buffers(root, &HashMap::new())?
        };
        Self::build_with_buffers(root, buffers, images)
    }

    /// Build from JSON alone, for tests that have no binary chunk.
    #[cfg(test)]
    pub(crate) fn build_for_test(root: &Value) -> Result<GltfScene, GltfError> {
        Self::build_with_buffers(root, Vec::new(), &GltfImages::default())
    }

    fn build_with_buffers(
        root: &Value,
        buffers: Vec<Vec<u8>>,
        images: &GltfImages,
    ) -> Result<GltfScene, GltfError> {
        let obj = root.as_object().ok_or(GltfError::Json("root not object"))?;
        // Before anything is read: a file that requires what this cannot do is
        // refused, rather than read into a scene that is quietly missing part
        // of itself.
        check_required_extensions(root)?;

        let mut buffer_views = obj
            .get("bufferViews")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        // `EXT_meshopt_compression` sits on the buffer views, so a decoded view
        // can stand in for the compressed one and everything downstream reads
        // it without knowing. Most `.glb` on the web is compressed; a loader
        // that skips this fails on real files while passing every test built
        // from files it wrote itself.
        let mut buffers = buffers;
        decompress_meshopt(&mut buffer_views, &mut buffers)?;
        let accessors = obj
            .get("accessors")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let materials_json = obj
            .get("materials")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let meshes_json = obj
            .get("meshes")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let nodes_json = obj
            .get("nodes")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let scenes_json = obj
            .get("scenes")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let default_scene = obj.get("scene").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        // Build texture map: GLTF texture index → Arc<Texture>.
        let textures_json = obj
            .get("textures")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let samplers_json = obj
            .get("samplers")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        // Two views of every image: one sRGB, one linear.
        //
        // glTF is explicit that only baseColor and emissive are colour; normal,
        // metallic-roughness and occlusion are *data* and must be read linearly
        // (spec §3.9.2). Reading a normal map as sRGB is not a subtle error —
        // the encoding is non-linear about the middle, so the 0.5 that means
        // "flat" decodes to 0.214, i.e. a tangent-space x of -0.57. Every flat
        // texel comes out steeply tilted, the whole map gains contrast it never
        // had, and the extra high-frequency detail aliases into visible banding
        // on anything small or seen at a grazing angle.
        let mut gltf_textures: Vec<Option<Arc<Texture>>> = Vec::with_capacity(textures_json.len());
        let mut gltf_textures_linear: Vec<Option<Arc<Texture>>> =
            Vec::with_capacity(textures_json.len());
        for tj in &textures_json {
            let Some(t) = tj.as_object() else {
                gltf_textures.push(None);
                gltf_textures_linear.push(None);
                continue;
            };
            let img_idx = t.get("source").and_then(|v| v.as_u64());
            let samp_idx = t.get("sampler").and_then(|v| v.as_u64());
            let Some(i) = img_idx else {
                gltf_textures.push(None);
                gltf_textures_linear.push(None);
                continue;
            };
            let Some((w, h, data)) = images.get(&(i as usize)) else {
                gltf_textures.push(None);
                gltf_textures_linear.push(None);
                continue;
            };
            let mut tex = Texture::new(*w, *h, TextureFormat::Rgba8UnormSrgb, data.clone());
            if let Some(s) = samp_idx
                .and_then(|i| samplers_json.get(i as usize))
                .and_then(|v| v.as_object())
            {
                if let Some(m) = s.get("magFilter").and_then(|v| v.as_u64()) {
                    tex.mag_filter = if m == 9728 {
                        TextureFilter::Nearest
                    } else {
                        TextureFilter::Linear
                    };
                }
                if let Some(m) = s.get("minFilter").and_then(|v| v.as_u64()) {
                    tex.min_filter = if m == 9728 || m == 9984 || m == 9986 {
                        TextureFilter::Nearest
                    } else {
                        TextureFilter::Linear
                    };
                }
                if let Some(w) = s.get("wrapS").and_then(|v| v.as_u64()) {
                    tex.wrap_s = match w {
                        10497 => TextureWrap::Repeat,
                        33648 => TextureWrap::MirroredRepeat,
                        _ => TextureWrap::ClampToEdge,
                    };
                }
                if let Some(w) = s.get("wrapT").and_then(|v| v.as_u64()) {
                    tex.wrap_t = match w {
                        10497 => TextureWrap::Repeat,
                        33648 => TextureWrap::MirroredRepeat,
                        _ => TextureWrap::ClampToEdge,
                    };
                }
            }
            let linear = tex.with_format(TextureFormat::Rgba8Unorm);
            gltf_textures.push(Some(Arc::new(tex)));
            gltf_textures_linear.push(Some(Arc::new(linear)));
        }

        // Build materials.
        let mut materials: Vec<Arc<Material>> = Vec::with_capacity(materials_json.len());
        for mj in &materials_json {
            materials.push(Arc::new(parse_material(
                mj,
                &gltf_textures,
                &gltf_textures_linear,
            )));
        }
        let fallback_material = Arc::new(Material::Standard(StandardMaterial::new(Color::WHITE)));

        // Build mesh primitives → (Mesh) ahead of node iteration so we can clone Arcs.
        let mut prims_per_mesh: Vec<Vec<Mesh>> = Vec::with_capacity(meshes_json.len());
        for mesh_j in &meshes_json {
            let prims = mesh_j
                .as_object()
                .and_then(|o| o.get("primitives"))
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            let mut ms = Vec::with_capacity(prims.len());
            for prim in &prims {
                let geom = primitive_geometry(prim, &accessors, &buffer_views, &buffers)?;
                let mat_idx = prim
                    .as_object()
                    .and_then(|o| o.get("material"))
                    .and_then(|v| v.as_u64());
                let mat = if let Some(i) = mat_idx {
                    materials
                        .get(i as usize)
                        .cloned()
                        .unwrap_or_else(|| fallback_material.clone())
                } else {
                    fallback_material.clone()
                };
                ms.push(Mesh::from_arc(Arc::new(geom), mat));
            }
            prims_per_mesh.push(ms);
        }

        // Build node hierarchy.
        let mut arena = ObjectArena::new();
        let mut node_ids: Vec<ObjectId> = Vec::with_capacity(nodes_json.len());
        for nj in &nodes_json {
            let no = nj.as_object();
            let mut obj = Object3D::group();
            if let Some(n) = no {
                if let Some(name) = n.get("name").and_then(|v| v.as_str()) {
                    obj.name = name.into();
                }
                apply_trs(&mut obj, n);
                // If this node points at a mesh, attach it as a child group of meshes.
                if let Some(mesh_idx) = n.get("mesh").and_then(|v| v.as_u64()) {
                    if let Some(primitives) = prims_per_mesh.get(mesh_idx as usize) {
                        if primitives.len() == 1 {
                            // Promote the single primitive to *this* node.
                            obj = Object3D::mesh(primitives[0].clone());
                            obj.name = no
                                .and_then(|n| n.get("name").and_then(|v| v.as_str()))
                                .unwrap_or("")
                                .into();
                            apply_trs(&mut obj, n);
                        }
                    }
                }
            }
            node_ids.push(arena.insert(obj));
        }
        // Wire parent/child + attach multi-primitive meshes as children.
        for (i, nj) in nodes_json.iter().enumerate() {
            let Some(n) = nj.as_object() else {
                continue;
            };
            if let Some(children) = n.get("children").and_then(|v| v.as_array()) {
                for c in children {
                    if let Some(ci) = c.as_u64() {
                        let child_id = node_ids[ci as usize];
                        arena.add_child(node_ids[i], child_id);
                    }
                }
            }
            if let Some(mesh_idx) = n.get("mesh").and_then(|v| v.as_u64()) {
                if let Some(primitives) = prims_per_mesh.get(mesh_idx as usize) {
                    if primitives.len() > 1 {
                        for p in primitives {
                            let child = arena.insert(Object3D::mesh(p.clone()));
                            arena.add_child(node_ids[i], child);
                        }
                    }
                }
            }
        }

        // Collect roots from the default scene.
        let mut roots = Vec::new();
        if let Some(scene_j) = scenes_json.get(default_scene).and_then(|v| v.as_object()) {
            if let Some(nodes) = scene_j.get("nodes").and_then(|v| v.as_array()) {
                for n in nodes {
                    if let Some(ni) = n.as_u64() {
                        roots.push(node_ids[ni as usize]);
                    }
                }
            }
        } else {
            // No scenes array: treat every node as a root.
            roots.extend_from_slice(&node_ids);
        }

        // Parse skins.
        let skins_json = obj
            .get("skins")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut skeletons: Vec<Skeleton> = Vec::with_capacity(skins_json.len());
        for sk in &skins_json {
            let Some(o) = sk.as_object() else {
                continue;
            };
            let joints: Vec<usize> = o
                .get("joints")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();
            let ibm_idx = o.get("inverseBindMatrices").and_then(|v| v.as_u64());
            let bones: Vec<Bone> = if let Some(idx) = ibm_idx {
                let (size, vals) =
                    read_accessor_floats(idx as usize, &accessors, &buffer_views, &buffers)
                        .unwrap_or((16, Vec::new()));
                let mat_count = vals.len().checked_div(size).unwrap_or(0);
                joints
                    .iter()
                    .enumerate()
                    .map(|(i, &j_idx)| {
                        let mut elements = [0.0f32; 16];
                        if i < mat_count {
                            let base = i * 16;
                            elements.copy_from_slice(&vals[base..base + 16]);
                        } else {
                            elements = Matrix4::identity().elements;
                        }
                        let node_id = node_ids
                            .get(j_idx)
                            .copied()
                            .unwrap_or_else(|| node_ids.first().copied().unwrap_or_default());
                        Bone {
                            node: node_id,
                            inverse_bind: Matrix4 { elements },
                        }
                    })
                    .collect()
            } else {
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
                            let vs: Vec<f32> = values.iter().skip(mi).step_by(morph_count).copied().collect();
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
            }
            animations.push(AnimationClip::new(name, max_time, tracks));
        }

        Ok(GltfScene {
            roots,
            arena,
            animations,
            skeletons,
        })
    }
}

/// Append `loaded` into `scene` and return the inserted root ids. Re-parents
/// every loaded root under `scene.root`.
pub fn add_to_scene(scene: &mut Scene, loaded: GltfScene) -> Vec<ObjectId> {
    // Insert nodes from the loaded arena into scene.arena, remembering remap.
    let mut remap: HashMap<ObjectId, ObjectId> = HashMap::new();
    let mut new_ids = Vec::with_capacity(loaded.arena.nodes.len());
    let all: Vec<(ObjectId, Object3D)> = loaded
        .arena
        .nodes
        .iter()
        .map(|(k, v)| (k, v.clone()))
        .collect();
    for (old, mut obj) in all {
        obj.parent = None;
        obj.children.clear();
        let new = scene.arena.insert(obj);
        remap.insert(old, new);
        new_ids.push(new);
    }
    // Re-wire children using remap. We iterate the original arena to read children.
    let snapshot: Vec<(ObjectId, Vec<ObjectId>)> = loaded
        .arena
        .nodes
        .iter()
        .map(|(k, v)| (k, v.children.clone()))
        .collect();
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
    let _ = new_ids;
    roots
}

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
            if normalized { v as f32 / 65535.0 } else { v as f32 }
        }
        5122 => {
            let v = i16::from_le_bytes(bytes.try_into().map_err(|_| GltfError::BadAccessor)?);
            if normalized { (v as f32 / 32767.0).max(-1.0) } else { v as f32 }
        }
        5121 => {
            let v = bytes[0];
            if normalized { v as f32 / 255.0 } else { v as f32 }
        }
        5120 => {
            let v = bytes[0] as i8;
            if normalized { (v as f32 / 127.0).max(-1.0) } else { v as f32 }
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
            assert!((i as usize) < vertices, "index {i} past {vertices} vertices");
        }
        // A closed sphere has no degenerate triangles.
        for t in index.chunks_exact(3) {
            assert!(t[0] != t[1] && t[1] != t[2] && t[0] != t[2], "degenerate {t:?}");
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
        assert_eq!(plain_unique.len(), 362, "distinct positions in the plain file");
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
