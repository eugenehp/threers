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

    fn build_with_buffers(
        root: &Value,
        buffers: Vec<Vec<u8>>,
        images: &GltfImages,
    ) -> Result<GltfScene, GltfError> {
        let obj = root.as_object().ok_or(GltfError::Json("root not object"))?;

        let buffer_views = obj
            .get("bufferViews")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
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
        let mut gltf_textures: Vec<Option<Arc<Texture>>> = Vec::with_capacity(textures_json.len());
        for tj in &textures_json {
            let Some(t) = tj.as_object() else {
                gltf_textures.push(None);
                continue;
            };
            let img_idx = t.get("source").and_then(|v| v.as_u64());
            let samp_idx = t.get("sampler").and_then(|v| v.as_u64());
            let Some(i) = img_idx else {
                gltf_textures.push(None);
                continue;
            };
            let Some((w, h, data)) = images.get(&(i as usize)) else {
                gltf_textures.push(None);
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
            gltf_textures.push(Some(Arc::new(tex)));
        }

        // Build materials.
        let mut materials: Vec<Arc<Material>> = Vec::with_capacity(materials_json.len());
        for mj in &materials_json {
            materials.push(Arc::new(parse_material(mj, &gltf_textures)));
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

fn parse_material(mj: &Value, gltf_textures: &[Option<Arc<Texture>>]) -> Material {
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
            let mr = lookup_tex(pbr, "metallicRoughnessTexture");
            std_mat.roughness_map = mr.clone();
            std_mat.metalness_map = mr;
        }
        std_mat.normal_map = lookup_tex(m, "normalTexture");
        std_mat.ao_map = lookup_tex(m, "occlusionTexture");
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
    let bytes = &buffer[start..];
    let mut out = Vec::with_capacity(count * item_size);
    match comp_type {
        5126 => {
            // FLOAT
            for i in 0..count * item_size {
                let off = i * 4;
                out.push(f32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]));
            }
        }
        5123 => {
            // UNSIGNED_SHORT
            for i in 0..count * item_size {
                let off = i * 2;
                let v = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as f32;
                out.push(v / 65535.0);
            }
        }
        5121 => {
            // UNSIGNED_BYTE
            for &b in &bytes[..count * item_size] {
                out.push(b as f32 / 255.0);
            }
        }
        _ => return Err(GltfError::BadAccessor),
    }
    Ok((item_size, out))
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
