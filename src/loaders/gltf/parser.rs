//! three.js r165 `GLTFParser`, without the promises.
//!
//! The JS loader resolves everything through `getDependency`, each kind of
//! thing cached by index, and builds three.js objects as the promises settle.
//! This does the same work in the same order — the order matters, because
//! names are made unique by a counter (`createUniqueName`), shared meshes,
//! cameras and lights are cloned with an `_instance_<n>` suffix in the order
//! their uses resolve, and a texture's `colorSpace` is whatever the last
//! material slot that touched it asked for — and builds each object as a node
//! of a Rust [`ObjectArena`] as it goes, so the scene a caller gets *is* the
//! one the loader made.
//!
//! What three.js does asynchronously is reproduced where it is visible: every
//! node, camera, light and scene name is taken before any mesh name (three.js
//! names meshes only once their materials and buffers have loaded), meshes
//! whose materials wait on images or whose geometry waits on Draco are named
//! after those that do not, and the rest follows request order.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::document::{self as doc, Accessor, Document, ImageSource};
use super::material::{
    build_rust_material, three_filter, three_wrap, Assoc, GltfMaterial, GltfTexture, ImageData,
    MaterialType, Param,
};
use super::GltfError;
use crate::core::{
    BindMode, Bone as SkinBone, BufferAttribute, BufferGeometry, InstancedMesh, LineSegments, Mesh,
    Object3D, ObjectArena, ObjectId, ObjectKind, Points, Skeleton, SkinnedMesh,
};
use crate::lights::{DirectionalLight, Light, PointLight, SpotLight};
use crate::loaders::json::Value;
use crate::materials::{Material, StandardMaterial};
use crate::math::{Box3, Color, Matrix4, Quaternion, Sphere, Vector3};

/// The three.js class an object is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectType {
    Object3D,
    Group,
    Bone,
    Mesh,
    SkinnedMesh,
    InstancedMesh,
    Line,
    LineSegments,
    LineLoop,
    Points,
    PerspectiveCamera,
    OrthographicCamera,
    DirectionalLight,
    PointLight,
    SpotLight,
}

impl ObjectType {
    pub fn name(self) -> &'static str {
        match self {
            ObjectType::Object3D => "Object3D",
            ObjectType::Group => "Group",
            ObjectType::Bone => "Bone",
            ObjectType::Mesh => "Mesh",
            ObjectType::SkinnedMesh => "SkinnedMesh",
            ObjectType::InstancedMesh => "InstancedMesh",
            ObjectType::Line => "Line",
            ObjectType::LineSegments => "LineSegments",
            ObjectType::LineLoop => "LineLoop",
            ObjectType::Points => "Points",
            ObjectType::PerspectiveCamera => "PerspectiveCamera",
            ObjectType::OrthographicCamera => "OrthographicCamera",
            ObjectType::DirectionalLight => "DirectionalLight",
            ObjectType::PointLight => "PointLight",
            ObjectType::SpotLight => "SpotLight",
        }
    }
    /// three.js `isMesh` (Mesh and its subclasses).
    pub fn is_mesh(self) -> bool {
        matches!(self, ObjectType::Mesh | ObjectType::SkinnedMesh | ObjectType::InstancedMesh)
    }
    pub fn is_line(self) -> bool {
        matches!(self, ObjectType::Line | ObjectType::LineSegments | ObjectType::LineLoop)
    }
    pub fn has_geometry(self) -> bool {
        self.is_mesh() || self.is_line() || self == ObjectType::Points
    }
    fn is_light(self) -> bool {
        matches!(self, ObjectType::DirectionalLight | ObjectType::PointLight | ObjectType::SpotLight)
    }
}

/// One `parser.associations` record: which glTF definition a thing came from.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Record {
    pub nodes: Option<usize>,
    pub meshes: Option<usize>,
    pub primitives: Option<usize>,
    pub materials: Option<usize>,
    pub textures: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraParams {
    /// `fov` in degrees, as three.js's constructor takes it.
    Perspective { fov: f64, aspect: f64, near: f64, far: f64 },
    Orthographic { left: f64, right: f64, top: f64, bottom: f64, near: f64, far: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightParams {
    pub color: [f32; 3],
    pub intensity: f32,
    pub distance: f32,
    pub decay: f32,
    pub angle: f32,
    pub penumbra: f32,
}

/// `EXT_mesh_gpu_instancing`'s per-instance data.
#[derive(Debug, Clone, PartialEq)]
pub struct Instances {
    pub count: usize,
    /// 16 per instance, column-major.
    pub matrices: Vec<f32>,
    /// `instanceColor`: `item_size` floats per instance.
    pub colors: Option<(Vec<f32>, usize)>,
}

#[derive(Debug, Clone)]
enum NameSpec {
    Fixed,
    /// A mesh primitive: named by `createUniqueName` once its mesh resolves.
    Mesh,
    /// A `clone()`: the source's name, plus `_instance_<n>` for the root of a
    /// `_getNodeRef` clone.
    Clone { source: usize, suffix: Option<u32> },
}

/// An object the loader made, and the node it is in the arena.
#[derive(Debug, Clone)]
pub struct GltfObject {
    pub id: ObjectId,
    pub ty: ObjectType,
    pub name: String,
    name_spec: NameSpec,
    name_override: Option<String>,
    pub user_data: Value,
    pub geometry: Option<usize>,
    pub material: Option<usize>,
    pub skeleton: Option<usize>,
    /// `morphTargetInfluences` — `None` when the geometry has no targets
    /// (three.js leaves it `undefined` then).
    pub morph_influences: Option<Vec<f32>>,
    pub morph_dictionary: Vec<(String, usize)>,
    pub instances: Option<Instances>,
    pub camera: Option<CameraParams>,
    pub light: Option<LightParams>,
    /// A light's `target`: a child, or (for a clone, as three.js's
    /// `DirectionalLight.copy` does it) a detached object of its own.
    pub light_target: Option<usize>,
    pub association: Assoc,
    finalized: bool,
}

/// A `BufferGeometry` the loader made.
#[derive(Debug, Clone)]
pub struct GltfGeometry {
    /// The geometry while it is being built; once built it lives in
    /// [`arc`](Self::arc) only (see [`get`](Self::get)).
    geometry: BufferGeometry,
    /// Whether `geometry` has moved into `arc`.
    moved: bool,
    /// three.js attribute names, in the order three.js has them.
    pub attribute_order: Vec<String>,
    /// For attributes read from normalized integer accessors: the component
    /// type three.js keeps them in (it writes `normalizeSkinWeights` back
    /// into that storage, rounding).
    pub normalized_types: HashMap<String, u32>,
    /// Morph attribute names (`position`, `normal`, `color`), in order.
    pub morph_order: Vec<String>,
    pub user_data: Value,
    /// The geometry the nodes share, once built.
    pub arc: Option<Arc<BufferGeometry>>,
}

impl GltfGeometry {
    fn new() -> Self {
        GltfGeometry {
            geometry: BufferGeometry::new(),
            moved: false,
            attribute_order: Vec::new(),
            normalized_types: HashMap::new(),
            morph_order: Vec::new(),
            user_data: empty_object(),
            arc: None,
        }
    }

    /// The geometry as three.js has it (the renderer's copy also carries the
    /// skinning attributes under the names the Rust skinning reads).
    pub fn get(&self) -> &BufferGeometry {
        match (&self.arc, self.moved) {
            (Some(a), true) => a,
            _ => &self.geometry,
        }
    }

    /// The geometry, to change (three.js mutates a shared geometry in place;
    /// the nodes built before keep the version they were given).
    fn edit(&mut self) -> &mut BufferGeometry {
        if self.moved {
            self.geometry = self.snapshot();
            self.moved = false;
        }
        self.arc = None;
        &mut self.geometry
    }

    /// A copy of the geometry as three.js has it.
    fn snapshot(&self) -> BufferGeometry {
        let mut g = self.get().clone();
        if self.moved {
            g.attributes.remove("joint");
            g.attributes.remove("weight");
        }
        g
    }

    /// Move the geometry into the `Arc` the nodes share.
    fn build(&mut self) {
        if self.arc.is_some() {
            return;
        }
        let mut geo = std::mem::take(&mut self.geometry);
        // The Rust skinning reads `joint` / `weight`.
        if let Some(a) = geo.attributes.get("skinIndex").cloned() {
            geo.attributes.insert("joint".into(), a);
        }
        if let Some(a) = geo.attributes.get("skinWeight").cloned() {
            geo.attributes.insert("weight".into(), a);
        }
        geo.bump_version();
        self.arc = Some(Arc::new(geo));
        self.moved = true;
    }
}

/// A `Skeleton`: bones (object indices) and their inverse bind matrices.
#[derive(Debug, Clone)]
pub struct GltfSkeleton {
    pub bones: Vec<usize>,
    pub inverses: Vec<[f32; 16]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackKind {
    Number,
    Vector,
    Quaternion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackInterpolation {
    Linear,
    Step,
    /// glTF CUBICSPLINE: values are `[in, value, out]` per key, and three.js
    /// samples them with its `GLTFCubicSplineInterpolant`.
    CubicSpline,
}

/// One `KeyframeTrack`: `<name of target>.<property>`.
#[derive(Debug, Clone)]
pub struct GltfTrack {
    pub target: usize,
    pub property: String,
    pub kind: TrackKind,
    pub times: Arc<Vec<f32>>,
    pub values: Arc<Vec<f32>>,
    pub interpolation: TrackInterpolation,
    /// The glTF path (`translation`, `rotation`, `scale`, `weights`).
    pub path: String,
}

#[derive(Debug, Clone)]
pub struct GltfClip {
    pub name: String,
    pub tracks: Vec<GltfTrack>,
}

impl GltfClip {
    /// three.js `resetDuration`: the latest key of any track.
    pub fn duration(&self) -> f32 {
        self.tracks
            .iter()
            .filter_map(|t| t.times.last().copied())
            .fold(0.0, f32::max)
    }
}

struct PendingChannel {
    node: usize,
    path: String,
    input: Accessor,
    output: Accessor,
    interpolation: TrackInterpolation,
}

/// What the image a texture needs is, as far as the decoder has got.
#[derive(Debug, Clone)]
enum ImageSlot {
    Unrequested,
    Requested,
    Decoded(ImageData),
    Failed(String),
}

/// An image the caller has to supply (see [`GltfParser::pending_images`]).
#[derive(Debug, Clone)]
pub struct ImageRequest {
    pub index: usize,
    pub source: ImageSource,
}

/// How a load behaves where three.js depends on its environment.
#[derive(Debug, Clone, Copy)]
pub struct GltfOptions {
    /// The image decoder takes WebP (`EXT_texture_webp`).
    pub webp: bool,
    /// The image decoder takes AVIF (`EXT_texture_avif`).
    pub avif: bool,
    /// Refuse a file whose `extensionsRequired` names something this loader
    /// does not implement. three.js only warns; the native API refuses, since
    /// such a file loads quietly wrong.
    pub refuse_unknown_required: bool,
}

impl Default for GltfOptions {
    fn default() -> Self {
        GltfOptions { webp: false, avif: false, refuse_unknown_required: true }
    }
}

/// The extensions three.js r165's loader registers as plugins (always known).
const PLUGIN_EXTENSIONS: &[&str] = &[
    "KHR_materials_clearcoat",
    "KHR_materials_dispersion",
    "KHR_texture_basisu",
    "EXT_texture_webp",
    "EXT_texture_avif",
    "KHR_materials_sheen",
    "KHR_materials_transmission",
    "KHR_materials_volume",
    "KHR_materials_ior",
    "KHR_materials_emissive_strength",
    "KHR_materials_specular",
    "KHR_materials_iridescence",
    "KHR_materials_anisotropy",
    "EXT_materials_bump",
    "KHR_lights_punctual",
    "EXT_meshopt_compression",
    "EXT_mesh_gpu_instancing",
];
/// Known when the file lists them in `extensionsUsed`.
const USED_EXTENSIONS: &[&str] = &[
    "KHR_materials_unlit",
    "KHR_draco_mesh_compression",
    "KHR_texture_transform",
    "KHR_mesh_quantization",
];

/// The material extensions whose `getMaterialType` makes a
/// `MeshPhysicalMaterial`, in plugin order.
const PHYSICAL_EXTENSIONS: &[&str] = &[
    "KHR_materials_clearcoat",
    "KHR_materials_dispersion",
    "KHR_materials_sheen",
    "KHR_materials_transmission",
    "KHR_materials_volume",
    "KHR_materials_ior",
    "KHR_materials_specular",
    "KHR_materials_iridescence",
    "KHR_materials_anisotropy",
    "EXT_materials_bump",
];

/// glTF attribute semantic → three.js attribute name (`ATTRIBUTES`), falling
/// back to the lower-cased semantic.
pub fn three_attribute_name(gltf: &str) -> String {
    match gltf {
        "POSITION" => "position".into(),
        "NORMAL" => "normal".into(),
        "TANGENT" => "tangent".into(),
        "TEXCOORD_0" => "uv".into(),
        "TEXCOORD_1" => "uv1".into(),
        "TEXCOORD_2" => "uv2".into(),
        "TEXCOORD_3" => "uv3".into(),
        "COLOR_0" => "color".into(),
        "WEIGHTS_0" => "skinWeight".into(),
        "JOINTS_0" => "skinIndex".into(),
        other => other.to_lowercase(),
    }
}

/// three.js `PropertyBinding.sanitizeNodeName`: whitespace to `_`, and
/// `[ ] . : /` (reserved in track names) removed.
pub fn sanitize_node_name(name: &str) -> String {
    let js_space = |c: char| {
        matches!(
            c,
            '\t' | '\n' | '\u{b}' | '\u{c}' | '\r' | ' ' | '\u{a0}' | '\u{1680}' | '\u{2028}'
                | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}' | '\u{feff}'
        ) || ('\u{2000}'..='\u{200a}').contains(&c)
    };
    name.chars()
        .filter(|c| !matches!(c, '[' | ']' | '.' | ':' | '/'))
        .map(|c| if js_space(c) { '_' } else { c })
        .collect()
}

/// A JS number as `String(n)` writes an integral index.
fn js_index(v: Option<&Value>) -> String {
    match v {
        None => "undefined".into(),
        Some(Value::Number(n)) if n.fract() == 0.0 && n.abs() < 1e21 => format!("{}", *n as i64),
        Some(Value::Number(n)) => format!("{n}"),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) => "null".into(),
        Some(Value::Bool(b)) => b.to_string(),
        Some(_) => "[object Object]".into(),
    }
}

/// `createAttributesKey`.
fn attributes_key(attrs: Option<&Value>) -> String {
    let Some(map) = attrs.and_then(|a| a.as_object()) else {
        return String::new();
    };
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    let mut out = String::new();
    for k in keys {
        out.push_str(k);
        out.push(':');
        out.push_str(&js_index(map.get(k)));
        out.push(';');
    }
    out
}

/// `createPrimitiveKey`.
fn primitive_key(prim: &Value) -> String {
    let mut key = if let Some(d) = doc::extension(prim, "KHR_draco_mesh_compression") {
        format!(
            "draco:{}:{}:{}",
            js_index(doc::get(d, "bufferView")),
            js_index(doc::get(d, "indices")),
            attributes_key(doc::get(d, "attributes"))
        )
    } else {
        format!(
            "{}:{}:{}",
            js_index(doc::get(prim, "indices")),
            attributes_key(doc::get(prim, "attributes")),
            js_index(doc::get(prim, "mode"))
        )
    };
    for t in doc::array(prim, "targets") {
        key.push(':');
        key.push_str(&attributes_key(Some(t)));
    }
    key
}

fn empty_object() -> Value {
    Value::Object(HashMap::new())
}

/// `assignExtrasToUserData`: an object's (or array's) members merged in, as
/// `Object.assign` does; anything else ignored with three.js's warning.
fn assign_extras(user_data: &mut Value, def: &Value, warnings: &mut Vec<String>) {
    let Some(extras) = doc::get(def, "extras") else { return };
    let Some(ud) = user_data.as_object_mut() else { return };
    match extras {
        Value::Object(map) => {
            for (k, v) in map {
                ud.insert(k.clone(), v.clone());
            }
        }
        Value::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                ud.insert(i.to_string(), v.clone());
            }
        }
        Value::Null => {}
        other => warnings.push(format!(
            "THREE.GLTFLoader: Ignoring primitive type .extras, {}",
            other.to_json()
        )),
    }
}

/// A local transform matrix from an arena node's TRS.
fn local_matrix(o: &Object3D) -> Matrix4 {
    Matrix4::compose(o.position, o.quaternion, o.scale)
}

/// The GLTFParser.
pub struct GltfParser {
    pub doc: Document,
    pub options: GltfOptions,
    pub warnings: Vec<String>,
    known_extensions: HashSet<String>,
    images: Vec<ImageSlot>,

    is_bone: Vec<bool>,
    mesh_skinned: Vec<bool>,
    mesh_refs: Vec<u32>,
    mesh_uses: Vec<u32>,
    camera_refs: Vec<u32>,
    camera_uses: Vec<u32>,
    light_refs: Vec<u32>,
    light_uses: Vec<u32>,
    names_used: HashMap<String, u32>,

    node_shallow: Vec<Option<usize>>,
    node_full: Vec<Option<usize>>,
    node_loading: Vec<bool>,
    mesh_cache: Vec<Option<usize>>,
    mesh_name_jobs: Vec<(usize, Vec<usize>, bool)>,
    camera_cache: Vec<Option<Option<usize>>>,
    light_cache: Vec<Option<usize>>,
    skin_cache: Vec<Option<usize>>,
    material_cache: Vec<Option<usize>>,
    material_textured: Vec<bool>,
    default_material: Option<usize>,
    derived_cache: HashMap<String, usize>,
    texture_index_cache: Vec<Option<Option<usize>>>,
    texture_key_cache: HashMap<String, Option<usize>>,
    source_cache: HashMap<usize, Option<usize>>,
    primitive_cache: HashMap<String, usize>,
    scene_cache: Vec<Option<usize>>,
    animation_cache: Vec<Option<usize>>,
    pending_channels: Vec<(usize, Vec<PendingChannel>)>,

    pub objects: Vec<GltfObject>,
    pub geometries: Vec<GltfGeometry>,
    pub materials: Vec<GltfMaterial>,
    pub textures: Vec<GltfTexture>,
    pub skeletons: Vec<GltfSkeleton>,
    pub clips: Vec<GltfClip>,
    pub records: Vec<Record>,
    /// Per glTF scene, the `Group` it became.
    pub scenes: Vec<Option<usize>>,
    /// Per glTF animation, its clip.
    pub animations: Vec<Option<usize>>,
    /// Per glTF camera, the camera object (`gltf.cameras`).
    pub cameras: Vec<Option<usize>>,
    /// The result's `userData` (root `extras`, unknown root extensions).
    pub user_data: Value,
    /// Objects whose association survives three.js's per-scene reduction.
    associated_objects: HashSet<usize>,
    parsed: bool,
    object_ids: HashMap<ObjectId, usize>,
}

impl GltfParser {
    pub fn new(doc: Document, options: GltfOptions) -> Result<GltfParser, GltfError> {
        let root = &doc.root;
        let mut known: HashSet<String> = PLUGIN_EXTENSIONS.iter().map(|s| s.to_string()).collect();
        if doc.is_glb {
            known.insert("KHR_binary_glTF".into());
        }
        let mut warnings = Vec::new();
        let required: Vec<&str> = doc::array(root, "extensionsRequired")
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        for used in doc::array(root, "extensionsUsed").iter().filter_map(|v| v.as_str()) {
            if USED_EXTENSIONS.contains(&used) {
                known.insert(used.to_string());
            } else if required.contains(&used) && !known.contains(used) {
                warnings.push(format!("THREE.GLTFLoader: Unknown extension \"{used}\"."));
            }
        }
        if options.refuse_unknown_required {
            for r in &required {
                if !PLUGIN_EXTENSIONS.contains(r) && !USED_EXTENSIONS.contains(r) {
                    return Err(GltfError::UnsupportedExtension(r.to_string()));
                }
            }
        }

        let n_nodes = doc::count(root, "nodes");
        let n_meshes = doc::count(root, "meshes");
        let n_cameras = doc::count(root, "cameras");
        let n_lights = doc::get(root, "extensions")
            .and_then(|e| doc::get(e, "KHR_lights_punctual"))
            .map(|k| doc::array(k, "lights").len())
            .unwrap_or(0);

        // `_markDefs`.
        let mut is_bone = vec![false; n_nodes];
        for skin in doc::array(root, "skins") {
            for j in doc::array(skin, "joints") {
                if let Some(j) = j.as_number().filter(|j| *j >= 0.0).map(|j| j as usize) {
                    if j < n_nodes {
                        is_bone[j] = true;
                    }
                }
            }
        }
        let mut mesh_skinned = vec![false; n_meshes];
        let mut mesh_refs = vec![0; n_meshes];
        let mut camera_refs = vec![0; n_cameras];
        let mut light_refs = vec![0; n_lights];
        for node in doc::array(root, "nodes") {
            if let Some(m) = doc::index(node, "mesh").filter(|m| *m < n_meshes) {
                mesh_refs[m] += 1;
                if doc::get(node, "skin").is_some() {
                    mesh_skinned[m] = true;
                }
            }
            if let Some(c) = doc::index(node, "camera").filter(|c| *c < n_cameras) {
                camera_refs[c] += 1;
            }
            if let Some(l) = doc::extension(node, "KHR_lights_punctual")
                .and_then(|e| doc::index(e, "light"))
                .filter(|l| *l < n_lights)
            {
                light_refs[l] += 1;
            }
        }

        let mut user_data = empty_object();
        add_unknown_extensions(&known, &mut user_data, root);
        assign_extras(&mut user_data, root, &mut warnings);

        Ok(GltfParser {
            images: vec![ImageSlot::Unrequested; doc::count(root, "images")],
            node_shallow: vec![None; n_nodes],
            node_full: vec![None; n_nodes],
            node_loading: vec![false; n_nodes],
            mesh_cache: vec![None; n_meshes],
            mesh_name_jobs: Vec::new(),
            camera_cache: vec![None; n_cameras],
            light_cache: vec![None; n_lights],
            skin_cache: vec![None; doc::count(root, "skins")],
            material_cache: vec![None; doc::count(root, "materials")],
            material_textured: vec![false; doc::count(root, "materials")],
            default_material: None,
            derived_cache: HashMap::new(),
            texture_index_cache: vec![None; doc::count(root, "textures")],
            texture_key_cache: HashMap::new(),
            source_cache: HashMap::new(),
            primitive_cache: HashMap::new(),
            scene_cache: vec![None; doc::count(root, "scenes")],
            animation_cache: vec![None; doc::count(root, "animations")],
            pending_channels: Vec::new(),
            is_bone,
            mesh_skinned,
            mesh_uses: vec![0; n_meshes],
            mesh_refs,
            camera_uses: vec![0; n_cameras],
            camera_refs,
            light_uses: vec![0; n_lights],
            light_refs,
            names_used: HashMap::new(),
            objects: Vec::new(),
            geometries: Vec::new(),
            materials: Vec::new(),
            textures: Vec::new(),
            skeletons: Vec::new(),
            clips: Vec::new(),
            records: Vec::new(),
            scenes: Vec::new(),
            animations: Vec::new(),
            cameras: Vec::new(),
            user_data,
            associated_objects: HashSet::new(),
            parsed: false,
            object_ids: HashMap::new(),
            known_extensions: known,
            doc,
            options,
            warnings,
        })
    }

    /// Which image formats the caller's decoder takes (three.js detects
    /// WebP and AVIF support before choosing a texture's source).
    pub fn set_image_support(&mut self, webp: bool, avif: bool) {
        self.options.webp = webp;
        self.options.avif = avif;
    }

    pub fn root(&self) -> &Value {
        &self.doc.root
    }

    /// `json.scene || 0`, when that scene exists.
    pub fn default_scene(&self) -> Option<usize> {
        let i = doc::index(&self.doc.root, "scene").unwrap_or(0);
        self.scenes.get(i).copied().flatten()
    }

    pub fn object_index(&self, id: ObjectId) -> Option<usize> {
        self.object_ids.get(&id).copied()
    }

    /// Whether an object is in `parser.associations`.
    pub fn object_associated(&self, i: usize) -> bool {
        match self.objects.get(i).map(|o| o.association) {
            Some(Assoc::Record(_)) => !self.parsed || self.associated_objects.contains(&i),
            _ => false,
        }
    }

    // ---------------------------------------------------------------- images

    /// Images that have to be supplied (fetched and decoded) before
    /// [`parse`](Self::parse) — those the materials in the scenes and the
    /// animated nodes use, by the source each texture would load. Each is
    /// returned once; call again after supplying them, since a failed KTX2
    /// decode makes the texture's fallback source needed.
    pub fn pending_images(&mut self) -> Result<Vec<ImageRequest>, GltfError> {
        let root = &self.doc.root;
        let mut start: Vec<usize> = Vec::new();
        for s in doc::array(root, "scenes") {
            start.extend(doc::array(s, "nodes").iter().filter_map(|n| n.as_number()).map(|n| n as usize));
        }
        for a in doc::array(root, "animations") {
            for c in doc::array(a, "channels") {
                if let Some(n) = doc::get(c, "target").and_then(|t| doc::index(t, "node")) {
                    start.push(n);
                }
            }
        }
        let materials = self.materials_under(start);
        self.pending_images_for_materials(&materials, &[])
    }

    /// Images a lazily loaded dependency (`kind` as in [`load`](Self::load))
    /// needs.
    pub fn pending_images_for(&mut self, kind: &str, index: usize) -> Result<Vec<ImageRequest>, GltfError> {
        let root = &self.doc.root;
        let (materials, textures) = match kind {
            "material" => (vec![index], vec![]),
            "texture" => (vec![], vec![index]),
            "mesh" => {
                let m = doc::def(root, "meshes", index)
                    .map(|m| doc::array(m, "primitives").iter().filter_map(|p| doc::index(p, "material")).collect())
                    .unwrap_or_default();
                (m, vec![])
            }
            "node" => (self.materials_under(vec![index]), vec![]),
            "scene" => {
                let nodes = doc::def(root, "scenes", index)
                    .map(|s| doc::array(s, "nodes").iter().filter_map(|n| n.as_number()).map(|n| n as usize).collect())
                    .unwrap_or_default();
                (self.materials_under(nodes), vec![])
            }
            "animation" => {
                let nodes = doc::def(root, "animations", index)
                    .map(|a| {
                        doc::array(a, "channels")
                            .iter()
                            .filter_map(|c| doc::get(c, "target").and_then(|t| doc::index(t, "node")))
                            .collect()
                    })
                    .unwrap_or_default();
                (self.materials_under(nodes), vec![])
            }
            _ => (vec![], vec![]),
        };
        self.pending_images_for_materials(&materials, &textures)
    }

    fn pending_images_for_materials(&mut self, materials: &[usize], extra: &[usize]) -> Result<Vec<ImageRequest>, GltfError> {
        let mut textures = extra.to_vec();
        for &m in materials {
            if let Some(def) = doc::def(&self.doc.root, "materials", m) {
                collect_material_textures(def, &mut textures);
            }
        }
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for t in textures {
            let Some(image) = self.texture_image_choice(t)? else { continue };
            if !seen.insert(image) {
                continue;
            }
            if matches!(self.images.get(image), Some(ImageSlot::Unrequested)) {
                match self.doc.image_source(image) {
                    Ok(source) => {
                        self.images[image] = ImageSlot::Requested;
                        out.push(ImageRequest { index: image, source });
                    }
                    Err(e) => self.images[image] = ImageSlot::Failed(e.to_string()),
                }
            }
        }
        Ok(out)
    }

    /// The materials three.js would load for these nodes: those of every
    /// mesh in their subtrees (and their skins' joints).
    fn materials_under(&self, start: Vec<usize>) -> Vec<usize> {
        let root = &self.doc.root;
        let mut stack = start;
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            let Some(node) = doc::def(root, "nodes", n) else { continue };
            for c in doc::array(node, "children") {
                if let Some(c) = c.as_number() {
                    stack.push(c as usize);
                }
            }
            if let Some(skin) = doc::index(node, "skin").and_then(|s| doc::def(root, "skins", s)) {
                for j in doc::array(skin, "joints") {
                    if let Some(j) = j.as_number() {
                        stack.push(j as usize);
                    }
                }
            }
            let Some(mesh) = doc::index(node, "mesh").and_then(|m| doc::def(root, "meshes", m)) else {
                continue;
            };
            for p in doc::array(mesh, "primitives") {
                if let Some(m) = doc::index(p, "material") {
                    if !out.contains(&m) {
                        out.push(m);
                    }
                }
            }
        }
        out
    }

    /// Which image a texture loads, following the plugins three.js asks in
    /// order — `KHR_texture_basisu`, `EXT_texture_webp`, `EXT_texture_avif` —
    /// then the texture's own `source`.
    fn texture_image_choice(&self, texture: usize) -> Result<Option<usize>, GltfError> {
        let root = &self.doc.root;
        let Some(def) = doc::def(root, "textures", texture) else {
            return Ok(None);
        };
        if let Some(src) = doc::extension(def, "KHR_texture_basisu").and_then(|e| doc::index(e, "source")) {
            match self.images.get(src) {
                Some(ImageSlot::Failed(reason)) => {
                    if self.required("KHR_texture_basisu") {
                        return Err(GltfError::msg(format!(
                            "THREE.GLTFLoader: KHR_texture_basisu texture {texture} cannot be decoded: {reason}"
                        )));
                    }
                }
                _ => return Ok(Some(src)),
            }
        }
        for (ext, supported, label) in [
            ("EXT_texture_webp", self.options.webp, "WebP"),
            ("EXT_texture_avif", self.options.avif, "AVIF"),
        ] {
            if let Some(src) = doc::extension(def, ext).and_then(|e| doc::index(e, "source")) {
                if supported {
                    return Ok(Some(src));
                }
                if self.required(ext) {
                    return Err(GltfError::msg(format!(
                        "THREE.GLTFLoader: {label} required by asset but unsupported."
                    )));
                }
                return Ok(doc::index(def, "source"));
            }
        }
        Ok(doc::index(def, "source"))
    }

    fn required(&self, ext: &str) -> bool {
        doc::array(&self.doc.root, "extensionsRequired")
            .iter()
            .any(|v| v.as_str() == Some(ext))
    }

    /// Supply an image's encoded bytes. PNG and KTX2 are decoded here;
    /// anything else (JPEG, WebP, AVIF, a PNG this decoder refuses) returns
    /// `false` for the caller to decode and pass to
    /// [`set_image_rgba`](Self::set_image_rgba).
    pub fn set_image_bytes(&mut self, i: usize, bytes: &[u8]) -> bool {
        if i >= self.images.len() {
            return true;
        }
        const KTX2: [u8; 12] = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];
        if bytes.len() >= 12 && bytes[..12] == KTX2 {
            self.images[i] = match super::super::ktx2::parse(bytes).and_then(|k| k.to_texture()) {
                Ok(t) => ImageSlot::Decoded(ImageData::Gpu(Arc::new(t))),
                Err(e) => ImageSlot::Failed(e.to_string()),
            };
            return true;
        }
        if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
            if let Ok(png) = crate::utils::png::decode_png(bytes) {
                self.images[i] = ImageSlot::Decoded(ImageData::Rgba {
                    width: png.width,
                    height: png.height,
                    data: Arc::new(png.rgba),
                });
                return true;
            }
        }
        false
    }

    /// Supply an image decoded elsewhere: 8-bit RGBA, rows top first.
    pub fn set_image_rgba(&mut self, i: usize, width: u32, height: u32, rgba: Vec<u8>) {
        if i < self.images.len() {
            self.images[i] = if rgba.len() == width as usize * height as usize * 4 {
                ImageSlot::Decoded(ImageData::Rgba { width, height, data: Arc::new(rgba) })
            } else {
                ImageSlot::Failed(format!("image {i}: {} bytes for {width}x{height}", rgba.len()))
            };
        }
    }

    /// The image could not be loaded; textures using it come out `null`, as
    /// three.js's do.
    pub fn set_image_failed(&mut self, i: usize, reason: String) {
        if i < self.images.len() {
            self.images[i] = ImageSlot::Failed(reason);
        }
    }

    // ------------------------------------------------------------------ parse

    /// `parser.parse`: every scene, every animation, every camera, into
    /// `arena`. Buffers and the images [`pending_images`](Self::pending_images)
    /// asked for have to be in by now.
    pub fn parse(&mut self, arena: &mut ObjectArena) -> Result<(), GltfError> {
        self.doc.decode_meshopt_views()?;
        let n_scenes = doc::count(&self.doc.root, "scenes");
        let n_anims = doc::count(&self.doc.root, "animations");
        let n_cams = doc::count(&self.doc.root, "cameras");
        self.scenes = vec![None; n_scenes];
        self.animations = vec![None; n_anims];
        self.cameras = vec![None; n_cams];
        for s in 0..n_scenes {
            self.scenes[s] = Some(self.get_scene(arena, s)?);
        }
        for a in 0..n_anims {
            self.animations[a] = Some(self.get_animation(arena, a)?);
        }
        for c in 0..n_cams {
            self.cameras[c] = self.get_camera(arena, c)?;
        }
        self.finish(arena);
        // three.js's `reduceAssociations`, scene by scene; across several
        // scenes the union (three.js's own result there depends on which
        // promise settles first).
        let mut keep = HashSet::new();
        for s in self.scenes.iter().flatten() {
            self.traverse(arena, *s, &mut |i| {
                keep.insert(i);
            });
        }
        if n_scenes > 0 {
            self.associated_objects = keep;
        } else {
            self.associated_objects = (0..self.objects.len()).collect();
        }
        self.parsed = true;
        Ok(())
    }

    /// After a load: name what is still unnamed, build the renderer's
    /// geometry, materials and textures, and fill the nodes in.
    fn finish(&mut self, arena: &mut ObjectArena) {
        self.expand_tracks(arena);
        self.resolve_names();
        self.finalize(arena);
    }

    /// Lazily load one more thing after `parse` (three.js's
    /// `parser.getDependency` for something no scene used). Returns its
    /// index in the matching list.
    pub fn load(&mut self, arena: &mut ObjectArena, kind: &str, index: usize) -> Result<Option<usize>, GltfError> {
        self.doc.decode_meshopt_views()?;
        let before = self.objects.len();
        let out = match kind {
            "scene" => Some(self.get_scene(arena, index)?),
            "node" => Some(self.get_node(arena, index)?),
            "mesh" => Some(self.get_mesh(arena, index)?),
            "material" => Some(self.get_material(index)?),
            "texture" => self.get_texture(index)?,
            "camera" => self.get_camera(arena, index)?,
            "skin" => Some(self.get_skin(arena, index)?),
            "animation" => Some(self.get_animation(arena, index)?),
            "light" => Some(self.load_light(arena, index)?),
            other => return Err(GltfError::msg(format!("Unknown type: {other}"))),
        };
        self.finish(arena);
        if self.parsed {
            // What a later `getDependency` makes goes into the map, as in
            // three.js (no scene reduction runs again).
            self.associated_objects.extend(before..self.objects.len());
        }
        Ok(out)
    }

    fn traverse(&self, arena: &ObjectArena, root: usize, f: &mut impl FnMut(usize)) {
        let mut seen = HashSet::new();
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            if !seen.insert(i) {
                continue;
            }
            f(i);
            if let Some(node) = arena.get(self.objects[i].id) {
                for c in node.children.iter().rev() {
                    if let Some(&ci) = self.object_ids.get(c) {
                        stack.push(ci);
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------- objects

    fn new_object(&mut self, arena: &mut ObjectArena, ty: ObjectType) -> usize {
        let mut o = Object3D::group();
        o.is_group = ty == ObjectType::Group;
        let id = arena.insert(o);
        let i = self.objects.len();
        self.objects.push(GltfObject {
            id,
            ty,
            name: String::new(),
            name_spec: NameSpec::Fixed,
            name_override: None,
            user_data: empty_object(),
            geometry: None,
            material: None,
            skeleton: None,
            morph_influences: None,
            morph_dictionary: Vec::new(),
            instances: None,
            camera: None,
            light: None,
            light_target: None,
            association: Assoc::Absent,
            finalized: false,
        });
        self.object_ids.insert(id, i);
        i
    }

    /// `parent.add(child)`, refusing a cycle (three.js would recurse forever
    /// on one).
    fn add(&mut self, arena: &mut ObjectArena, parent: usize, child: usize) {
        if parent == child {
            return;
        }
        let (p, c) = (self.objects[parent].id, self.objects[child].id);
        let mut cur = Some(p);
        while let Some(x) = cur {
            if x == c {
                self.warnings.push("THREE.GLTFLoader: node hierarchy has a cycle; link skipped".into());
                return;
            }
            cur = arena.get(x).and_then(|o| o.parent);
        }
        arena.add_child(p, c);
    }

    fn children(&self, arena: &ObjectArena, i: usize) -> Vec<usize> {
        arena
            .get(self.objects[i].id)
            .map(|o| o.children.iter().filter_map(|c| self.object_ids.get(c).copied()).collect())
            .unwrap_or_default()
    }

    /// `createUniqueName`.
    fn create_unique_name(&mut self, original: &str) -> String {
        let s = sanitize_node_name(original);
        if let Some(n) = self.names_used.get_mut(&s) {
            *n += 1;
            format!("{s}_{n}")
        } else {
            self.names_used.insert(s.clone(), 0);
            s
        }
    }

    fn new_record(&mut self, r: Record) -> Assoc {
        self.records.push(r);
        Assoc::Record(self.records.len() - 1)
    }

    /// three.js `object.clone()` (recursive): same geometry and material,
    /// `userData` copied, children cloned in order. A light's `target` is
    /// cloned again on its own, as `DirectionalLight.copy` does.
    fn clone_object(&mut self, arena: &mut ObjectArena, src: usize) -> usize {
        let ty = self.objects[src].ty;
        let c = self.new_object(arena, ty);
        let s = self.objects[src].clone();
        {
            let o = &mut self.objects[c];
            o.name = s.name.clone();
            o.name_spec = NameSpec::Clone { source: src, suffix: None };
            o.user_data = s.user_data.clone();
            o.geometry = s.geometry;
            o.material = s.material;
            o.skeleton = s.skeleton;
            o.morph_influences = s.morph_influences.clone();
            o.morph_dictionary = s.morph_dictionary.clone();
            o.instances = s.instances.clone();
            o.camera = s.camera;
            o.light = s.light;
        }
        if let (Some(sn), Some(cn)) = (arena.get(s.id).cloned(), arena.get_mut(self.objects[c].id)) {
            cn.position = sn.position;
            cn.quaternion = sn.quaternion;
            cn.scale = sn.scale;
            cn.visible = sn.visible;
        }
        for child in self.children(arena, src) {
            let cc = self.clone_object(arena, child);
            self.add(arena, c, cc);
        }
        if let Some(t) = s.light_target {
            let tc = self.clone_object(arena, t);
            self.objects[c].light_target = Some(tc);
        }
        c
    }

    /// `_getNodeRef`: the object itself for a single use, a clone named
    /// `<name>_instance_<n>` (with the associations copied) otherwise.
    fn get_node_ref(&mut self, arena: &mut ObjectArena, kind: RefKind, index: usize, object: usize) -> usize {
        let (refs, uses) = match kind {
            RefKind::Mesh => (&self.mesh_refs, &mut self.mesh_uses),
            RefKind::Camera => (&self.camera_refs, &mut self.camera_uses),
            RefKind::Light => (&self.light_refs, &mut self.light_uses),
        };
        if refs.get(index).copied().unwrap_or(0) <= 1 {
            return object;
        }
        let n = uses[index];
        uses[index] += 1;
        let c = self.clone_object(arena, object);
        self.objects[c].name_spec = NameSpec::Clone { source: object, suffix: Some(n) };
        self.copy_mappings(arena, object, c);
        c
    }

    fn copy_mappings(&mut self, arena: &ObjectArena, original: usize, clone: usize) {
        if let a @ Assoc::Record(_) = self.objects[original].association {
            self.objects[clone].association = a;
        }
        let (oc, cc) = (self.children(arena, original), self.children(arena, clone));
        for (o, c) in oc.into_iter().zip(cc) {
            self.copy_mappings(arena, o, c);
        }
    }

    // ------------------------------------------------------------------ nodes

    /// `getDependency('node', i)` → `loadNode`.
    fn get_node(&mut self, arena: &mut ObjectArena, i: usize) -> Result<usize, GltfError> {
        if i >= self.node_full.len() {
            return Err(GltfError::msg(format!("node {i} does not exist")));
        }
        if let Some(o) = self.node_full[i] {
            return Ok(o);
        }
        if self.node_loading[i] {
            return Err(GltfError::msg(format!("THREE.GLTFLoader: node {i} is its own ancestor")));
        }
        self.node_loading[i] = true;
        let result = self.load_node(arena, i);
        self.node_loading[i] = false;
        let node = result?;
        self.node_full[i] = Some(node);
        Ok(node)
    }

    fn load_node(&mut self, arena: &mut ObjectArena, i: usize) -> Result<usize, GltfError> {
        let def = doc::def(&self.doc.root, "nodes", i).cloned().unwrap_or(Value::Null);
        let node = self.load_node_shallow(arena, i)?;
        let mut children = Vec::new();
        for c in doc::array(&def, "children") {
            let c = c.as_number().filter(|c| *c >= 0.0).map(|c| c as usize).unwrap_or(usize::MAX);
            children.push(self.get_node(arena, c)?);
        }
        if let Some(skin) = doc::index(&def, "skin") {
            let skeleton = self.get_skin(arena, skin)?;
            let mut skinned = Vec::new();
            self.traverse(arena, node, &mut |o| skinned.push(o));
            for o in skinned {
                if self.objects[o].ty == ObjectType::SkinnedMesh {
                    // `mesh.bind(skeleton, identity)`.
                    self.objects[o].skeleton = Some(skeleton);
                }
            }
        }
        for c in children {
            self.add(arena, node, c);
        }
        Ok(node)
    }

    /// `_loadNodeShallow`.
    fn load_node_shallow(&mut self, arena: &mut ObjectArena, i: usize) -> Result<usize, GltfError> {
        if let Some(Some(o)) = self.node_shallow.get(i) {
            return Ok(*o);
        }
        let def = doc::def(&self.doc.root, "nodes", i)
            .cloned()
            .ok_or_else(|| GltfError::msg(format!("node {i} does not exist")))?;
        let def_name = doc::string(&def, "name").filter(|n| !n.is_empty()).map(str::to_string);
        let node_name = def_name.as_ref().map(|n| self.create_unique_name(n));
        let mut objects = Vec::new();
        if let Some(m) = self.create_node_mesh_plugins(arena, i, &def)? {
            objects.push(m);
        }
        if let Some(c) = doc::index(&def, "camera") {
            if let Some(cam) = self.get_camera(arena, c)? {
                objects.push(self.get_node_ref(arena, RefKind::Camera, c, cam));
            }
        }
        if let Some(l) = doc::extension(&def, "KHR_lights_punctual").and_then(|e| doc::index(e, "light")) {
            let light = self.load_light(arena, l)?;
            objects.push(self.get_node_ref(arena, RefKind::Light, l, light));
        }
        let node = if self.is_bone[i] {
            self.new_object(arena, ObjectType::Bone)
        } else if objects.len() > 1 {
            self.new_object(arena, ObjectType::Group)
        } else if objects.len() == 1 {
            objects[0]
        } else {
            self.new_object(arena, ObjectType::Object3D)
        };
        if objects.first() != Some(&node) {
            for o in objects {
                self.add(arena, node, o);
            }
        }
        if let Some(name) = def_name {
            if let Some(ud) = self.objects[node].user_data.as_object_mut() {
                ud.insert("name".into(), Value::String(name));
            }
            self.objects[node].name_override = node_name;
        }
        let mut ud = std::mem::replace(&mut self.objects[node].user_data, Value::Null);
        assign_extras(&mut ud, &def, &mut self.warnings);
        add_unknown_extensions(&self.known_extensions, &mut ud, &def);
        self.objects[node].user_data = ud;

        let id = self.objects[node].id;
        if let Some(o) = arena.get_mut(id) {
            if let Some(m) = doc::floats(&def, "matrix", 16, 0.0) {
                let mut e = [0.0f32; 16];
                e.copy_from_slice(&m);
                // `node.applyMatrix4(matrix)`.
                o.apply_matrix4(&Matrix4 { elements: e });
            } else {
                if let Some(t) = doc::floats(&def, "translation", 3, 0.0) {
                    o.position = Vector3::new(t[0], t[1], t[2]);
                }
                if let Some(r) = doc::floats(&def, "rotation", 4, 0.0) {
                    o.quaternion = Quaternion::new(r[0], r[1], r[2], r[3]);
                }
                if let Some(s) = doc::floats(&def, "scale", 3, 1.0) {
                    o.scale = Vector3::new(s[0], s[1], s[2]);
                }
            }
        }
        match self.objects[node].association {
            Assoc::Record(r) => self.records[r].nodes = Some(i),
            _ => {
                let a = self.new_record(Record { nodes: Some(i), ..Default::default() });
                self.objects[node].association = a;
            }
        }
        self.node_shallow[i] = Some(node);
        Ok(node)
    }

    /// `_invokeOne(ext => ext.createNodeMesh)`: `EXT_mesh_gpu_instancing`,
    /// then the parser's own.
    fn create_node_mesh_plugins(
        &mut self,
        arena: &mut ObjectArena,
        i: usize,
        def: &Value,
    ) -> Result<Option<usize>, GltfError> {
        if let (Some(ext), Some(mesh_index)) = (doc::extension(def, "EXT_mesh_gpu_instancing"), doc::index(def, "mesh")) {
            let modes_ok = doc::def(&self.doc.root, "meshes", mesh_index)
                .map(|m| {
                    doc::array(m, "primitives")
                        .iter()
                        .all(|p| matches!(doc::index(p, "mode"), None | Some(4) | Some(5) | Some(6)))
                })
                .unwrap_or(false);
            let attrs: Vec<(String, usize)> = doc::get(ext, "attributes")
                .and_then(|a| a.as_object())
                .map(|a| {
                    let mut v: Vec<(String, usize)> = a
                        .iter()
                        .filter_map(|(k, v)| v.as_number().map(|n| (k.clone(), n as usize)))
                        .collect();
                    v.sort();
                    v
                })
                .unwrap_or_default();
            if modes_ok && !attrs.is_empty() {
                return self.create_instanced_node_mesh(arena, i, def, &attrs).map(Some);
            }
        }
        self.create_node_mesh(arena, def)
    }

    /// `GLTFParser.createNodeMesh`.
    fn create_node_mesh(&mut self, arena: &mut ObjectArena, def: &Value) -> Result<Option<usize>, GltfError> {
        let Some(mesh_index) = doc::index(def, "mesh") else {
            return Ok(None);
        };
        let mesh = self.get_mesh(arena, mesh_index)?;
        let node = self.get_node_ref(arena, RefKind::Mesh, mesh_index, mesh);
        if let Some(weights) = doc::get(def, "weights").and_then(|w| w.as_array()).cloned() {
            let mut targets = Vec::new();
            self.traverse(arena, node, &mut |o| targets.push(o));
            for o in targets {
                if !self.objects[o].ty.is_mesh() {
                    continue;
                }
                if let Some(inf) = self.objects[o].morph_influences.as_mut() {
                    for (k, w) in weights.iter().enumerate() {
                        if let (Some(slot), Some(w)) = (inf.get_mut(k), w.as_f32()) {
                            *slot = w;
                        }
                    }
                }
            }
        }
        Ok(Some(node))
    }

    /// `GLTFMeshGpuInstancing.createNodeMesh`.
    fn create_instanced_node_mesh(
        &mut self,
        arena: &mut ObjectArena,
        _i: usize,
        def: &Value,
        attrs: &[(String, usize)],
    ) -> Result<usize, GltfError> {
        let mut accessors: Vec<(String, Accessor)> = Vec::new();
        for (name, acc) in attrs {
            accessors.push((name.clone(), self.doc.accessor(*acc)?));
        }
        let node_object = self
            .create_node_mesh(arena, def)?
            .ok_or_else(|| GltfError::msg("EXT_mesh_gpu_instancing on a node without a mesh"))?;
        let is_group = self.objects[node_object].ty == ObjectType::Group;
        let meshes = if is_group { self.children(arena, node_object) } else { vec![node_object] };
        let count = ["TRANSLATION", "ROTATION", "SCALE"]
            .iter()
            .find_map(|k| accessors.iter().find(|(n, _)| n == k))
            .or(accessors.first())
            .map(|(_, a)| a.count)
            .unwrap_or(0);
        let get = |name: &str| accessors.iter().find(|(n, _)| n == name).map(|(_, a)| a.values());
        let (t, r, s) = (get("TRANSLATION"), get("ROTATION"), get("SCALE"));
        let mut matrices = Vec::with_capacity(count * 16);
        let (mut p, mut q, mut sc) = (Vector3::ZERO, Quaternion::new(0.0, 0.0, 0.0, 1.0), Vector3::new(1.0, 1.0, 1.0));
        for k in 0..count {
            if let Some(t) = &t {
                if let Some(v) = t.get(k * 3..k * 3 + 3) {
                    p = Vector3::new(v[0], v[1], v[2]);
                }
            }
            if let Some(r) = &r {
                if let Some(v) = r.get(k * 4..k * 4 + 4) {
                    q = Quaternion::new(v[0], v[1], v[2], v[3]);
                }
            }
            if let Some(s) = &s {
                if let Some(v) = s.get(k * 3..k * 3 + 3) {
                    sc = Vector3::new(v[0], v[1], v[2]);
                }
            }
            matrices.extend_from_slice(&Matrix4::compose(p, q, sc).elements);
        }
        let mut instanced = Vec::new();
        for mesh in meshes {
            let im = self.new_object(arena, ObjectType::InstancedMesh);
            let src = self.objects[mesh].clone();
            let mut colors = None;
            for (name, acc) in &accessors {
                if name == "_COLOR_0" {
                    colors = Some((acc.values(), acc.item_size));
                } else if !matches!(name.as_str(), "TRANSLATION" | "ROTATION" | "SCALE") {
                    if let Some(g) = src.geometry {
                        self.set_geometry_attribute(g, name, BufferAttribute::new(acc.values(), acc.item_size));
                    }
                }
            }
            {
                let o = &mut self.objects[im];
                o.geometry = src.geometry;
                o.material = src.material;
                o.instances = Some(Instances { count, matrices: matrices.clone(), colors });
                // `Object3D.prototype.copy.call(instancedMesh, mesh)`: name,
                // transform, userData (and children).
                o.name_spec = NameSpec::Clone { source: mesh, suffix: None };
                o.user_data = src.user_data.clone();
            }
            // An InstancedMesh's own constructor sizes the influences.
            let morph = src.geometry.and_then(|g| self.morph_count(g));
            self.objects[im].morph_influences = morph.map(|n| vec![0.0; n]);
            self.objects[im].morph_dictionary = morph
                .map(|n| (0..n).map(|k| (k.to_string(), k)).collect())
                .unwrap_or_default();
            if let (Some(sn), Some(dn)) = (arena.get(src.id).cloned(), arena.get_mut(self.objects[im].id)) {
                dn.position = sn.position;
                dn.quaternion = sn.quaternion;
                dn.scale = sn.scale;
            }
            for child in self.children(arena, mesh) {
                let cc = self.clone_object(arena, child);
                self.add(arena, im, cc);
            }
            self.assign_final_material(im);
            instanced.push(im);
        }
        if is_group {
            // `nodeObject.clear()`, then the instanced meshes.
            for c in self.children(arena, node_object) {
                arena.remove_child(self.objects[c].id);
            }
            for im in instanced {
                self.add(arena, node_object, im);
            }
            return Ok(node_object);
        }
        Ok(instanced[0])
    }

    fn set_geometry_attribute(&mut self, g: usize, name: &str, attr: BufferAttribute) {
        let geo = &mut self.geometries[g];
        let e = geo.edit();
        let bounds = (e.bounding_box, e.bounding_sphere);
        e.set_attribute(name, attr);
        // three.js's `setAttribute` leaves computed bounds alone.
        e.bounding_box = bounds.0;
        e.bounding_sphere = bounds.1;
        if !geo.attribute_order.iter().any(|n| n == name) {
            geo.attribute_order.push(name.to_string());
        }
    }

    fn morph_count(&self, g: usize) -> Option<usize> {
        let geo = &self.geometries[g];
        let first = geo.morph_order.first()?;
        geo.get().morph_attributes.get(first).map(|l| l.len())
    }

    // ----------------------------------------------------------------- meshes

    /// `getDependency('mesh', i)` → `loadMesh`.
    fn get_mesh(&mut self, arena: &mut ObjectArena, i: usize) -> Result<usize, GltfError> {
        if let Some(Some(m)) = self.mesh_cache.get(i) {
            return Ok(*m);
        }
        let def = doc::def(&self.doc.root, "meshes", i)
            .cloned()
            .ok_or_else(|| GltfError::msg(format!("mesh {i} does not exist")))?;
        let primitives = doc::array(&def, "primitives").to_vec();
        let mut mats = Vec::with_capacity(primitives.len());
        let mut delayed = false;
        for p in &primitives {
            mats.push(match doc::index(p, "material") {
                None => self.default_material(),
                Some(m) => {
                    let mi = self.get_material(m)?;
                    delayed |= self.material_textured.get(m).copied().unwrap_or(false);
                    mi
                }
            });
            delayed |= doc::extension(p, "KHR_draco_mesh_compression").is_some();
        }
        let mut geometries = Vec::with_capacity(primitives.len());
        for p in &primitives {
            geometries.push(self.load_geometry(p)?);
        }
        let mut meshes = Vec::new();
        for (k, prim) in primitives.iter().enumerate() {
            let mode = doc::index(prim, "mode");
            let ty = match mode {
                None | Some(4) | Some(5) | Some(6) => {
                    if self.mesh_skinned.get(i).copied().unwrap_or(false) {
                        ObjectType::SkinnedMesh
                    } else {
                        ObjectType::Mesh
                    }
                }
                Some(1) => ObjectType::LineSegments,
                Some(3) => ObjectType::Line,
                Some(2) => ObjectType::LineLoop,
                Some(0) => ObjectType::Points,
                Some(m) => {
                    return Err(GltfError::msg(format!("THREE.GLTFLoader: Primitive mode unsupported: {m}")))
                }
            };
            let mut g = geometries[k];
            if ty == ObjectType::SkinnedMesh {
                self.normalize_skin_weights(g);
            }
            if matches!(mode, Some(5) | Some(6)) {
                g = self.triangulate_draw_mode(g, mode == Some(6));
            }
            let o = self.new_object(arena, ty);
            self.objects[o].geometry = Some(g);
            self.objects[o].material = Some(mats[k]);
            self.objects[o].name_spec = NameSpec::Mesh;
            if let Some(n) = self.morph_count(g) {
                // `updateMorphTargets(mesh, meshDef)`.
                let mut inf = vec![0.0f32; n];
                for (w, slot) in doc::array(&def, "weights").iter().zip(inf.iter_mut()) {
                    if let Some(w) = w.as_f32() {
                        *slot = w;
                    }
                }
                let mut dict: Vec<(String, usize)> = (0..n).map(|m| (m.to_string(), m)).collect();
                if let Some(names) = doc::get(&def, "extras")
                    .and_then(|e| doc::get(e, "targetNames"))
                    .and_then(|t| t.as_array())
                {
                    if names.len() == n {
                        dict = names
                            .iter()
                            .enumerate()
                            .map(|(m, v)| (js_string(v), m))
                            .collect();
                    } else {
                        self.warnings.push("THREE.GLTFLoader: Invalid extras.targetNames length. Ignoring names.".into());
                    }
                }
                self.objects[o].morph_influences = Some(inf);
                self.objects[o].morph_dictionary = dict;
            }
            let mut ud = empty_object();
            assign_extras(&mut ud, &def, &mut self.warnings);
            add_unknown_extensions(&self.known_extensions, &mut ud, prim);
            self.objects[o].user_data = ud;
            self.assign_final_material(o);
            meshes.push(o);
        }
        for (k, &m) in meshes.iter().enumerate() {
            let a = self.new_record(Record { meshes: Some(i), primitives: Some(k), ..Default::default() });
            self.objects[m].association = a;
        }
        self.mesh_name_jobs.push((i, meshes.clone(), delayed));
        let result = if meshes.len() == 1 {
            let mut ud = std::mem::replace(&mut self.objects[meshes[0]].user_data, Value::Null);
            add_unknown_extensions(&self.known_extensions, &mut ud, &def);
            self.objects[meshes[0]].user_data = ud;
            meshes[0]
        } else {
            let group = self.new_object(arena, ObjectType::Group);
            let mut ud = empty_object();
            add_unknown_extensions(&self.known_extensions, &mut ud, &def);
            self.objects[group].user_data = ud;
            let a = self.new_record(Record { meshes: Some(i), ..Default::default() });
            self.objects[group].association = a;
            for m in meshes {
                self.add(arena, group, m);
            }
            group
        };
        if let Some(slot) = self.mesh_cache.get_mut(i) {
            *slot = Some(result);
        }
        Ok(result)
    }

    /// `SkinnedMesh.normalizeSkinWeights` (on the shared geometry, as three.js),
    /// in double precision as JS computes it — and, for weights stored as
    /// normalized integers, rounded back into that storage as three.js's
    /// `setXYZW` does.
    fn normalize_skin_weights(&mut self, g: usize) {
        let quantize = self.geometries[g].normalized_types.get("skinWeight").copied();
        if !self.geometries[g].get().attributes.contains_key("skinWeight") {
            return;
        }
        let geo = self.geometries[g].edit();
        let Some(w) = geo.attributes.get_mut("skinWeight") else { return };
        let size = w.item_size.max(1);
        let max = match quantize {
            Some(5121) => Some((255.0f64, false)),
            Some(5120) => Some((127.0, true)),
            Some(5123) => Some((65535.0, false)),
            Some(5122) => Some((32767.0, true)),
            _ => None,
        };
        for v in w.array.chunks_mut(size) {
            // The values three.js reads: exact quotients of the stored integers.
            let mut x = [0.0f64; 4];
            for (k, slot) in x.iter_mut().enumerate().take(v.len().min(4)) {
                *slot = match max {
                    Some((m, signed)) => {
                        let raw = (v[k] as f64 * m).round();
                        if signed { (raw / m).max(-1.0) } else { raw / m }
                    }
                    None => v[k] as f64,
                };
            }
            let scale = 1.0 / (x[0].abs() + x[1].abs() + x[2].abs() + x[3].abs());
            if scale.is_finite() {
                for e in &mut x {
                    *e *= scale;
                }
            } else {
                x = [1.0, 0.0, 0.0, 0.0];
            }
            for (k, out) in v.iter_mut().enumerate().take(4) {
                *out = match max {
                    // `MathUtils.normalize`, then the attribute read back.
                    Some((m, signed)) => {
                        let q = (x[k] * m + 0.5).floor();
                        (if signed { (q / m).max(-1.0) } else { q / m }) as f32
                    }
                    None => x[k] as f32,
                };
            }
        }
        geo.bump_version();
    }

    /// `BufferGeometryUtils.toTrianglesDrawMode`: a new geometry with a
    /// triangle-list index (and the original indexed if it was not).
    fn triangulate_draw_mode(&mut self, g: usize, fan: bool) -> usize {
        if self.geometries[g].get().index.is_none() {
            let n = self.geometries[g].get().get_attribute("position").map(|p| p.count());
            match n {
                Some(n) => {
                    let e = self.geometries[g].edit();
                    let bounds = (e.bounding_box, e.bounding_sphere);
                    e.set_index((0..n as u32).collect());
                    e.bounding_box = bounds.0;
                    e.bounding_sphere = bounds.1;
                }
                None => {
                    self.warnings.push("THREE.BufferGeometryUtils.toTrianglesDrawMode(): Undefined position attribute. Processing not possible.".into());
                    return g;
                }
            }
        }
        let index = self.geometries[g].get().index.clone().unwrap_or_default();
        let tris = index.len().saturating_sub(2);
        let mut out = Vec::with_capacity(tris * 3);
        for k in 0..tris {
            if fan {
                out.extend_from_slice(&[index[0], index[k + 1], index[k + 2]]);
            } else if k % 2 == 0 {
                out.extend_from_slice(&[index[k], index[k + 1], index[k + 2]]);
            } else {
                out.extend_from_slice(&[index[k + 2], index[k + 1], index[k]]);
            }
        }
        let src = &self.geometries[g];
        let mut copy = GltfGeometry {
            geometry: src.snapshot(),
            moved: false,
            arc: None,
            ..src.clone()
        };
        let bounds = (copy.geometry.bounding_box, copy.geometry.bounding_sphere);
        copy.geometry.set_index(out);
        copy.geometry.clear_groups();
        copy.geometry.bounding_box = bounds.0;
        copy.geometry.bounding_sphere = bounds.1;
        self.geometries.push(copy);
        self.geometries.len() - 1
    }

    /// `loadGeometries` for one primitive: cached by `createPrimitiveKey`.
    fn load_geometry(&mut self, prim: &Value) -> Result<usize, GltfError> {
        let key = primitive_key(prim);
        if let Some(&g) = self.primitive_cache.get(&key) {
            return Ok(g);
        }
        let mut geo = GltfGeometry::new();
        if let Some(d) = doc::extension(prim, "KHR_draco_mesh_compression") {
            self.draco_geometry(&mut geo, prim, d)?;
        }
        // `addPrimitiveAttributes`.
        if let Some(attrs) = doc::get(prim, "attributes").and_then(|a| a.as_object()) {
            let mut names: Vec<(&String, &Value)> = attrs.iter().collect();
            names.sort_by_key(|(k, _)| attribute_rank(k));
            for (gltf_name, acc) in names {
                let three = three_attribute_name(gltf_name);
                if geo.geometry.attributes.contains_key(&three) {
                    continue;
                }
                let acc = acc.as_number().ok_or(GltfError::BadAccessor)? as usize;
                let a = self.doc.accessor(acc)?;
                geo.geometry.set_attribute(three.clone(), BufferAttribute::new(a.values(), a.item_size));
                if a.normalized && a.component_type != 5126 {
                    geo.normalized_types.insert(three.clone(), a.component_type);
                }
                geo.attribute_order.push(three);
            }
        }
        if let Some(ind) = doc::index(prim, "indices") {
            if geo.geometry.index.is_none() {
                let idx = self.doc.accessor_u32(ind)?;
                geo.geometry.set_index(idx);
            }
        }
        assign_extras(&mut geo.user_data, prim, &mut self.warnings);
        let targets = doc::array(prim, "targets").to_vec();
        if !targets.is_empty() {
            self.add_morph_targets(&mut geo, &targets)?;
        }
        self.compute_bounds(&mut geo, prim);
        self.geometries.push(geo);
        let g = self.geometries.len() - 1;
        self.primitive_cache.insert(key, g);
        Ok(g)
    }

    /// `GLTFDracoMeshCompressionExtension.decodePrimitive`, the decoder being
    /// `super::super::draco`.
    fn draco_geometry(&mut self, geo: &mut GltfGeometry, prim: &Value, ext: &Value) -> Result<(), GltfError> {
        let view = doc::index(ext, "bufferView").ok_or(GltfError::MissingField("bufferView"))?;
        let bytes = self.doc.buffer_view(view)?;
        let mesh = super::super::draco::decode_mesh(bytes)?;
        let attrs = doc::get(ext, "attributes")
            .and_then(|a| a.as_object())
            .ok_or(GltfError::MissingField("attributes"))?;
        let mut names: Vec<(&String, &Value)> = attrs.iter().collect();
        names.sort_by_key(|(k, _)| attribute_rank(k));
        for (gltf_name, id) in names {
            let unique_id = id.as_number().ok_or(GltfError::BadAccessor)? as u32;
            let attribute = mesh
                .attributes
                .iter()
                .find(|a| a.descriptor.unique_id == unique_id)
                .ok_or_else(|| GltfError::msg(format!("Draco attribute {unique_id} not in the stream")))?;
            // The accessor says what type three.js asks DRACOLoader for.
            let accessor = doc::get(prim, "attributes")
                .and_then(|a| doc::index(a, gltf_name))
                .and_then(|a| doc::def(&self.doc.root, "accessors", a));
            let component = accessor.and_then(|a| doc::index(a, "componentType")).unwrap_or(5126) as u32;
            let normalized = accessor
                .and_then(|a| doc::get(a, "normalized"))
                .and_then(|v| v.as_bool())
                == Some(true);
            let values = if component == 5126 {
                draco_float(attribute)
            } else {
                let raw = attribute.as_f32();
                if normalized {
                    let s = Accessor::normalized_scale(component);
                    let signed = matches!(component, 5120 | 5122);
                    raw.iter().map(|v| if signed { (v * s).max(-1.0) } else { v * s }).collect()
                } else {
                    raw
                }
            };
            let three = three_attribute_name(gltf_name);
            geo.geometry.set_attribute(three.clone(), BufferAttribute::new(values, attribute.num_components()));
            geo.attribute_order.push(three);
        }
        geo.geometry.set_index(mesh.indices());
        Ok(())
    }

    /// `addMorphTargets`: `position` / `normal` / `color` lists, one entry per
    /// target, relative. A target missing an attribute others have gets the
    /// base attribute, as three.js r165 does.
    fn add_morph_targets(&mut self, geo: &mut GltfGeometry, targets: &[Value]) -> Result<(), GltfError> {
        for (gltf, three) in [("POSITION", "position"), ("NORMAL", "normal"), ("COLOR_0", "color")] {
            if !targets.iter().any(|t| doc::get(t, gltf).is_some()) {
                continue;
            }
            let mut list = Vec::with_capacity(targets.len());
            for t in targets {
                let attr = match doc::index(t, gltf) {
                    Some(a) => {
                        let a = self.doc.accessor(a)?;
                        BufferAttribute::new(a.values(), a.item_size)
                    }
                    None => match geo.geometry.get_attribute(three) {
                        Some(base) => base.clone(),
                        None => {
                            let n = geo.geometry.get_attribute("position").map(|p| p.count()).unwrap_or(0);
                            let size = if three == "color" { 4 } else { 3 };
                            BufferAttribute::new(vec![0.0; n * size], size)
                        }
                    },
                };
                list.push(attr);
            }
            geo.geometry.morph_attributes.insert(three.into(), list);
            geo.morph_order.push(three.into());
        }
        if !geo.morph_order.is_empty() {
            geo.geometry.morph_targets_relative = true;
            geo.geometry.bump_version();
        }
        Ok(())
    }

    /// `computeBounds`: from the POSITION accessor's `min`/`max`, grown by the
    /// largest morph displacement.
    fn compute_bounds(&mut self, geo: &mut GltfGeometry, prim: &Value) {
        let root = &self.doc.root;
        let Some(pos) = doc::get(prim, "attributes").and_then(|a| doc::index(a, "POSITION")) else {
            return;
        };
        let Some(acc) = doc::def(root, "accessors", pos) else { return };
        let scale_of = |a: &Value| {
            if doc::get(a, "normalized").and_then(|v| v.as_bool()) == Some(true) {
                Accessor::normalized_scale(doc::index(a, "componentType").unwrap_or(5126) as u32)
            } else {
                1.0
            }
        };
        let (Some(min), Some(max)) = (doc::floats(acc, "min", 3, 0.0), doc::floats(acc, "max", 3, 0.0)) else {
            self.warnings.push("THREE.GLTFLoader: Missing min/max properties for accessor POSITION.".into());
            return;
        };
        let s = scale_of(acc);
        let mut bmin = Vector3::new(min[0] * s, min[1] * s, min[2] * s);
        let mut bmax = Vector3::new(max[0] * s, max[1] * s, max[2] * s);
        let mut disp = Vector3::ZERO;
        for t in doc::array(prim, "targets") {
            let Some(acc) = doc::index(t, "POSITION").and_then(|a| doc::def(root, "accessors", a)) else {
                continue;
            };
            match (doc::floats(acc, "min", 3, 0.0), doc::floats(acc, "max", 3, 0.0)) {
                (Some(mn), Some(mx)) => {
                    let s = scale_of(acc);
                    disp.x = disp.x.max(mn[0].abs().max(mx[0].abs()) * s);
                    disp.y = disp.y.max(mn[1].abs().max(mx[1].abs()) * s);
                    disp.z = disp.z.max(mn[2].abs().max(mx[2].abs()) * s);
                }
                _ => self.warnings.push("THREE.GLTFLoader: Missing min/max properties for accessor POSITION.".into()),
            }
        }
        bmin = Vector3::new(bmin.x - disp.x, bmin.y - disp.y, bmin.z - disp.z);
        bmax = Vector3::new(bmax.x + disp.x, bmax.y + disp.y, bmax.z + disp.z);
        let center = Vector3::new((bmin.x + bmax.x) * 0.5, (bmin.y + bmax.y) * 0.5, (bmin.z + bmax.z) * 0.5);
        let d = Vector3::new(bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z);
        let radius = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt() / 2.0;
        geo.geometry.bounding_box = Some(Box3 { min: bmin, max: bmax });
        geo.geometry.bounding_sphere = Some(Sphere { center, radius });
    }

    // --------------------------------------------------------------- materials

    /// `createDefaultMaterial`.
    fn default_material(&mut self) -> usize {
        if let Some(m) = self.default_material {
            return m;
        }
        let mut m = GltfMaterial::new(MaterialType::MeshStandardMaterial);
        m.params.insert("color", Param::Color([1.0, 1.0, 1.0]));
        m.params.insert("emissive", Param::Color([0.0, 0.0, 0.0]));
        m.params.insert("metalness", Param::Num(1.0));
        m.params.insert("roughness", Param::Num(1.0));
        m.params.insert("transparent", Param::Bool(false));
        m.params.insert("depthTest", Param::Bool(true));
        m.params.insert("side", Param::Num(0.0));
        self.materials.push(m);
        let i = self.materials.len() - 1;
        self.default_material = Some(i);
        i
    }

    /// `getDependency('material', i)` → `loadMaterial`.
    fn get_material(&mut self, i: usize) -> Result<usize, GltfError> {
        if let Some(Some(m)) = self.material_cache.get(i) {
            return Ok(*m);
        }
        let def = doc::def(&self.doc.root, "materials", i)
            .cloned()
            .ok_or_else(|| GltfError::msg(format!("material {i} does not exist")))?;
        let exts = doc::get(&def, "extensions");
        let has = |name: &str| exts.and_then(|e| doc::get(e, name)).is_some();
        let pbr = doc::get(&def, "pbrMetallicRoughness").cloned().unwrap_or(Value::Null);
        let mut m;
        let mut textured = false;
        if has("KHR_materials_unlit") {
            m = GltfMaterial::new(MaterialType::MeshBasicMaterial);
            self.base_color(&mut m, &pbr, &mut textured)?;
        } else {
            let ty = if PHYSICAL_EXTENSIONS.iter().any(|e| has(e)) {
                MaterialType::MeshPhysicalMaterial
            } else {
                MaterialType::MeshStandardMaterial
            };
            m = GltfMaterial::new(ty);
            self.base_color(&mut m, &pbr, &mut textured)?;
            m.params.insert("metalness", Param::Num(doc::num(&pbr, "metallicFactor").unwrap_or(1.0)));
            m.params.insert("roughness", Param::Num(doc::num(&pbr, "roughnessFactor").unwrap_or(1.0)));
            if let Some(t) = doc::get(&pbr, "metallicRoughnessTexture").cloned() {
                textured |= self.assign_texture(&mut m, "metalnessMap", &t, false)?;
                textured |= self.assign_texture(&mut m, "roughnessMap", &t, false)?;
            }
            if let Some(exts) = exts.cloned() {
                textured |= self.extend_material_params(&mut m, &exts)?;
            }
        }
        if doc::get(&def, "doubleSided").and_then(|v| v.as_bool()) == Some(true) {
            m.params.insert("side", Param::Num(2.0));
        }
        let alpha_mode = doc::string(&def, "alphaMode").unwrap_or("OPAQUE");
        if alpha_mode == "BLEND" {
            m.params.insert("transparent", Param::Bool(true));
            m.params.insert("depthWrite", Param::Bool(false));
        } else {
            m.params.insert("transparent", Param::Bool(false));
            if alpha_mode == "MASK" {
                m.params.insert("alphaTest", Param::Num(doc::num(&def, "alphaCutoff").unwrap_or(0.5)));
            }
        }
        let lit = m.ty != MaterialType::MeshBasicMaterial;
        if lit {
            if let Some(t) = doc::get(&def, "normalTexture").cloned() {
                textured |= self.assign_texture(&mut m, "normalMap", &t, false)?;
                let s = doc::num(&t, "scale").unwrap_or(1.0) as f32;
                m.params.insert("normalScale", Param::Vec2([s, s]));
                m.gltf_normal_scale = Some(s);
            }
            if let Some(t) = doc::get(&def, "occlusionTexture").cloned() {
                textured |= self.assign_texture(&mut m, "aoMap", &t, false)?;
                if let Some(s) = doc::num(&t, "strength") {
                    m.params.insert("aoMapIntensity", Param::Num(s));
                }
            }
            if let Some(e) = doc::floats(&def, "emissiveFactor", 3, 0.0) {
                m.params.insert("emissive", Param::Color([e[0], e[1], e[2]]));
            }
            if let Some(t) = doc::get(&def, "emissiveTexture").cloned() {
                textured |= self.assign_texture(&mut m, "emissiveMap", &t, true)?;
            }
        }
        if let Some(name) = doc::string(&def, "name").filter(|n| !n.is_empty()) {
            m.name = name.to_string();
        }
        assign_extras(&mut m.user_data, &def, &mut self.warnings);
        if doc::get(&def, "extensions").is_some() {
            add_unknown_extensions(&self.known_extensions, &mut m.user_data, &def);
        }
        m.association = self.new_record(Record { materials: Some(i), ..Default::default() });
        self.materials.push(m);
        let idx = self.materials.len() - 1;
        if let Some(slot) = self.material_cache.get_mut(i) {
            *slot = Some(idx);
        }
        if let Some(t) = self.material_textured.get_mut(i) {
            *t = textured;
        }
        Ok(idx)
    }

    fn base_color(&mut self, m: &mut GltfMaterial, pbr: &Value, textured: &mut bool) -> Result<(), GltfError> {
        m.params.insert("color", Param::Color([1.0, 1.0, 1.0]));
        m.params.insert("opacity", Param::Num(1.0));
        if let Some(f) = doc::get(pbr, "baseColorFactor").and_then(|f| f.as_array()) {
            let c = |k: usize| f.get(k).and_then(|v| v.as_f32()).unwrap_or(1.0);
            m.params.insert("color", Param::Color([c(0), c(1), c(2)]));
            if let Some(a) = f.get(3).and_then(|v| v.as_number()) {
                m.params.insert("opacity", Param::Num(a));
            }
        }
        if let Some(t) = doc::get(pbr, "baseColorTexture").cloned() {
            *textured |= self.assign_texture(m, "map", &t, true)?;
        }
        Ok(())
    }

    /// Every plugin's `extendMaterialParams`, in plugin order.
    fn extend_material_params(&mut self, m: &mut GltfMaterial, exts: &Value) -> Result<bool, GltfError> {
        let mut textured = false;
        let num = |e: &Value, k: &str| doc::num(e, k);
        if let Some(e) = doc::get(exts, "KHR_materials_clearcoat").cloned() {
            if let Some(v) = num(&e, "clearcoatFactor") {
                m.params.insert("clearcoat", Param::Num(v));
            }
            if let Some(t) = doc::get(&e, "clearcoatTexture").cloned() {
                textured |= self.assign_texture(m, "clearcoatMap", &t, false)?;
            }
            if let Some(v) = num(&e, "clearcoatRoughnessFactor") {
                m.params.insert("clearcoatRoughness", Param::Num(v));
            }
            if let Some(t) = doc::get(&e, "clearcoatRoughnessTexture").cloned() {
                textured |= self.assign_texture(m, "clearcoatRoughnessMap", &t, false)?;
            }
            if let Some(t) = doc::get(&e, "clearcoatNormalTexture").cloned() {
                textured |= self.assign_texture(m, "clearcoatNormalMap", &t, false)?;
                if let Some(s) = num(&t, "scale") {
                    m.params.insert("clearcoatNormalScale", Param::Vec2([s as f32, s as f32]));
                    m.gltf_clearcoat_normal_scale = Some(s as f32);
                }
            }
        }
        if let Some(e) = doc::get(exts, "KHR_materials_dispersion") {
            m.params.insert("dispersion", Param::Num(num(e, "dispersion").unwrap_or(0.0)));
        }
        if let Some(e) = doc::get(exts, "KHR_materials_sheen").cloned() {
            m.params.insert("sheenColor", Param::Color([0.0, 0.0, 0.0]));
            m.params.insert("sheenRoughness", Param::Num(0.0));
            m.params.insert("sheen", Param::Num(1.0));
            if let Some(c) = doc::floats(&e, "sheenColorFactor", 3, 0.0) {
                m.params.insert("sheenColor", Param::Color([c[0], c[1], c[2]]));
            }
            if let Some(v) = num(&e, "sheenRoughnessFactor") {
                m.params.insert("sheenRoughness", Param::Num(v));
            }
            if let Some(t) = doc::get(&e, "sheenColorTexture").cloned() {
                textured |= self.assign_texture(m, "sheenColorMap", &t, true)?;
            }
            if let Some(t) = doc::get(&e, "sheenRoughnessTexture").cloned() {
                textured |= self.assign_texture(m, "sheenRoughnessMap", &t, false)?;
            }
        }
        if let Some(e) = doc::get(exts, "KHR_materials_transmission").cloned() {
            if let Some(v) = num(&e, "transmissionFactor") {
                m.params.insert("transmission", Param::Num(v));
            }
            if let Some(t) = doc::get(&e, "transmissionTexture").cloned() {
                textured |= self.assign_texture(m, "transmissionMap", &t, false)?;
            }
        }
        if let Some(e) = doc::get(exts, "KHR_materials_volume").cloned() {
            m.params.insert("thickness", Param::Num(num(&e, "thicknessFactor").unwrap_or(0.0)));
            if let Some(t) = doc::get(&e, "thicknessTexture").cloned() {
                textured |= self.assign_texture(m, "thicknessMap", &t, false)?;
            }
            // `extension.attenuationDistance || Infinity`.
            let d = num(&e, "attenuationDistance").filter(|d| *d != 0.0 && !d.is_nan()).unwrap_or(f64::INFINITY);
            m.params.insert("attenuationDistance", Param::Num(d));
            let c = doc::floats(&e, "attenuationColor", 3, 0.0).unwrap_or(vec![1.0, 1.0, 1.0]);
            m.params.insert("attenuationColor", Param::Color([c[0], c[1], c[2]]));
        }
        if let Some(e) = doc::get(exts, "KHR_materials_ior") {
            m.params.insert("ior", Param::Num(num(e, "ior").unwrap_or(1.5)));
        }
        if let Some(e) = doc::get(exts, "KHR_materials_emissive_strength") {
            if let Some(v) = num(e, "emissiveStrength") {
                m.params.insert("emissiveIntensity", Param::Num(v));
            }
        }
        if let Some(e) = doc::get(exts, "KHR_materials_specular").cloned() {
            m.params.insert("specularIntensity", Param::Num(num(&e, "specularFactor").unwrap_or(1.0)));
            if let Some(t) = doc::get(&e, "specularTexture").cloned() {
                textured |= self.assign_texture(m, "specularIntensityMap", &t, false)?;
            }
            let c = doc::floats(&e, "specularColorFactor", 3, 0.0).unwrap_or(vec![1.0, 1.0, 1.0]);
            m.params.insert("specularColor", Param::Color([c[0], c[1], c[2]]));
            if let Some(t) = doc::get(&e, "specularColorTexture").cloned() {
                textured |= self.assign_texture(m, "specularColorMap", &t, true)?;
            }
        }
        if let Some(e) = doc::get(exts, "KHR_materials_iridescence").cloned() {
            if let Some(v) = num(&e, "iridescenceFactor") {
                m.params.insert("iridescence", Param::Num(v));
            }
            if let Some(t) = doc::get(&e, "iridescenceTexture").cloned() {
                textured |= self.assign_texture(m, "iridescenceMap", &t, false)?;
            }
            if let Some(v) = num(&e, "iridescenceIor") {
                m.params.insert("iridescenceIOR", Param::Num(v));
            }
            let mut range = [100.0f32, 400.0];
            if let Some(v) = num(&e, "iridescenceThicknessMinimum") {
                range[0] = v as f32;
            }
            if let Some(v) = num(&e, "iridescenceThicknessMaximum") {
                range[1] = v as f32;
            }
            m.params.insert("iridescenceThicknessRange", Param::Range(range));
            if let Some(t) = doc::get(&e, "iridescenceThicknessTexture").cloned() {
                textured |= self.assign_texture(m, "iridescenceThicknessMap", &t, false)?;
            }
        }
        if let Some(e) = doc::get(exts, "KHR_materials_anisotropy").cloned() {
            if let Some(v) = num(&e, "anisotropyStrength") {
                m.params.insert("anisotropy", Param::Num(v));
            }
            if let Some(v) = num(&e, "anisotropyRotation") {
                m.params.insert("anisotropyRotation", Param::Num(v));
            }
            if let Some(t) = doc::get(&e, "anisotropyTexture").cloned() {
                textured |= self.assign_texture(m, "anisotropyMap", &t, false)?;
            }
        }
        if let Some(e) = doc::get(exts, "EXT_materials_bump").cloned() {
            m.params.insert("bumpScale", Param::Num(num(&e, "bumpFactor").unwrap_or(1.0)));
            if let Some(t) = doc::get(&e, "bumpTexture").cloned() {
                textured |= self.assign_texture(m, "bumpMap", &t, false)?;
            }
        }
        Ok(textured)
    }

    /// `assignTexture`. Returns whether a texture was assigned.
    fn assign_texture(&mut self, m: &mut GltfMaterial, key: &'static str, map_def: &Value, srgb: bool) -> Result<bool, GltfError> {
        let Some(index) = doc::index(map_def, "index") else { return Ok(false) };
        let Some(mut t) = self.get_texture(index)? else { return Ok(false) };
        if let Some(tc) = doc::index(map_def, "texCoord").filter(|c| *c > 0) {
            let c = self.textures[t].clone_texture();
            self.textures.push(c);
            t = self.textures.len() - 1;
            self.textures[t].channel = tc as u32;
        }
        if self.known_extensions.contains("KHR_texture_transform") {
            if let Some(tr) = doc::extension(map_def, "KHR_texture_transform").cloned() {
                let reference = self.textures[t].association;
                t = self.extend_texture(t, &tr);
                self.textures[t].association = match reference {
                    Assoc::Absent => Assoc::Undefined,
                    a => a,
                };
            }
        }
        if srgb {
            self.textures[t].srgb = true;
        }
        m.params.insert(key, Param::Texture(t));
        Ok(true)
    }

    /// `GLTFTextureTransformExtension.extendTexture`.
    fn extend_texture(&mut self, t: usize, tr: &Value) -> usize {
        let tex_coord = doc::index(tr, "texCoord");
        let offset = doc::floats(tr, "offset", 2, 0.0);
        let rotation = doc::num(tr, "rotation");
        let scale = doc::floats(tr, "scale", 2, 1.0);
        if tex_coord.is_none_or(|c| c as u32 == self.textures[t].channel)
            && offset.is_none()
            && rotation.is_none()
            && scale.is_none()
        {
            return t;
        }
        let mut c = self.textures[t].clone_texture();
        if let Some(tc) = tex_coord {
            c.channel = tc as u32;
        }
        if let Some(o) = offset {
            c.offset = [o[0], o[1]];
        }
        if let Some(r) = rotation {
            c.rotation = r as f32;
        }
        if let Some(s) = scale {
            c.repeat = [s[0], s[1]];
        }
        self.textures.push(c);
        self.textures.len() - 1
    }

    /// `getDependency('texture', i)`: the plugins, then `loadTexture`.
    fn get_texture(&mut self, i: usize) -> Result<Option<usize>, GltfError> {
        if let Some(Some(t)) = self.texture_index_cache.get(i) {
            return Ok(*t);
        }
        if i >= self.texture_index_cache.len() {
            return Err(GltfError::msg(format!("texture {i} does not exist")));
        }
        let source = self.texture_image_choice(i)?;
        let t = match source {
            Some(s) => self.load_texture_image(i, s),
            None => {
                self.warnings.push(format!("THREE.GLTFLoader: texture {i} has no image source"));
                None
            }
        };
        self.texture_index_cache[i] = Some(t);
        Ok(t)
    }

    /// `loadTextureImage`.
    fn load_texture_image(&mut self, texture_index: usize, source_index: usize) -> Option<usize> {
        let root = &self.doc.root;
        let tdef = doc::def(root, "textures", texture_index).cloned().unwrap_or(Value::Null);
        let sdef = doc::def(root, "images", source_index).cloned().unwrap_or(Value::Null);
        let key = format!(
            "{}:{}",
            match doc::get(&sdef, "uri").filter(|u| u.as_str().is_some_and(|s| !s.is_empty())) {
                Some(u) => js_index(Some(u)),
                None => js_index(doc::get(&sdef, "bufferView")),
            },
            js_index(doc::get(&tdef, "sampler"))
        );
        if let Some(t) = self.texture_key_cache.get(&key) {
            return *t;
        }
        let t = self.load_image_source(source_index).inspect(|&t| {
            let tex = &mut self.textures[t];
            tex.name = doc::string(&tdef, "name")
                .filter(|n| !n.is_empty())
                .or_else(|| doc::string(&sdef, "name").filter(|n| !n.is_empty()))
                .unwrap_or("")
                .to_string();
            if tex.name.is_empty() {
                if let Some(uri) = doc::string(&sdef, "uri") {
                    if !uri.starts_with("data:image/") {
                        tex.name = uri.to_string();
                    }
                }
            }
            let sampler = doc::index(&tdef, "sampler").and_then(|s| doc::def(&self.doc.root, "samplers", s));
            let field = |k: &str| sampler.and_then(|s| doc::index(s, k));
            tex.mag_filter = three_filter(field("magFilter"), super::material::constants::LINEAR_FILTER);
            tex.min_filter = three_filter(field("minFilter"), super::material::constants::LINEAR_MIPMAP_LINEAR_FILTER);
            tex.wrap_s = three_wrap(field("wrapS"));
            tex.wrap_t = three_wrap(field("wrapT"));
        });
        if let Some(t) = t {
            let a = self.new_record(Record { textures: Some(texture_index), ..Default::default() });
            self.textures[t].association = a;
        }
        self.texture_key_cache.insert(key, t);
        t
    }

    /// `loadImageSource`: the first texture made from an image, or a clone
    /// of it for every later one.
    fn load_image_source(&mut self, source: usize) -> Option<usize> {
        if let Some(first) = self.source_cache.get(&source) {
            let first = (*first)?;
            let mut c = self.textures[first].clone_texture();
            // Cloned before any material slot set a colour space on the first.
            c.srgb = false;
            self.textures.push(c);
            return Some(self.textures.len() - 1);
        }
        let result = match self.images.get(source) {
            Some(ImageSlot::Decoded(data)) => {
                let mut t = GltfTexture::new(source, data.clone());
                let sdef = doc::def(&self.doc.root, "images", source).cloned().unwrap_or(Value::Null);
                assign_extras(&mut t.user_data, &sdef, &mut self.warnings);
                let mime = doc::string(&sdef, "mimeType")
                    .map(str::to_string)
                    .or_else(|| self.doc.image_source(source).ok().map(|s| s.mime_type))
                    .unwrap_or_else(|| "image/png".into());
                if let Some(ud) = t.user_data.as_object_mut() {
                    ud.insert("mimeType".into(), Value::String(mime));
                }
                self.textures.push(t);
                Some(self.textures.len() - 1)
            }
            Some(ImageSlot::Failed(reason)) => {
                self.warnings.push(format!("THREE.GLTFLoader: Couldn't load texture {source}: {reason}"));
                None
            }
            _ => {
                self.warnings.push(format!("THREE.GLTFLoader: Couldn't load texture {source}: image not supplied"));
                None
            }
        };
        self.source_cache.insert(source, result);
        result
    }

    /// `assignFinalMaterial`.
    fn assign_final_material(&mut self, o: usize) {
        let (Some(g), Some(mut mat)) = (self.objects[o].geometry, self.objects[o].material) else {
            return;
        };
        let geo = self.geometries[g].get();
        let derivative_tangents = !geo.attributes.contains_key("tangent");
        let vertex_colors = geo.attributes.contains_key("color");
        let flat_shading = !geo.attributes.contains_key("normal");
        let ty = self.objects[o].ty;
        if ty == ObjectType::Points || ty.is_line() {
            let (prefix, new_ty) = if ty == ObjectType::Points {
                ("PointsMaterial", MaterialType::PointsMaterial)
            } else {
                ("LineBasicMaterial", MaterialType::LineBasicMaterial)
            };
            let key = format!("{prefix}:{mat}");
            mat = match self.derived_cache.get(&key) {
                Some(&d) => d,
                None => {
                    let src = self.materials[mat].clone();
                    let mut d = GltfMaterial::new(new_ty);
                    d.name = src.name.clone();
                    d.user_data = src.user_data.clone();
                    // `Material.prototype.copy`: the base material's own fields.
                    for k in ["side", "opacity", "transparent", "depthTest", "depthWrite", "alphaTest", "vertexColors"] {
                        if let Some(v) = src.params.get(k) {
                            d.params.insert(k, v.clone());
                        }
                    }
                    d.params.insert("color", src.params.get("color").cloned().unwrap_or(Param::Color([1.0, 1.0, 1.0])));
                    if let Some(map) = src.params.get("map") {
                        d.params.insert("map", map.clone());
                    }
                    if new_ty == MaterialType::PointsMaterial {
                        // glTF points are one pixel.
                        d.params.insert("sizeAttenuation", Param::Bool(false));
                    }
                    self.materials.push(d);
                    let d = self.materials.len() - 1;
                    self.derived_cache.insert(key, d);
                    d
                }
            };
        }
        if derivative_tangents || vertex_colors || flat_shading {
            let mut key = format!("ClonedMaterial:{mat}:");
            if derivative_tangents {
                key.push_str("derivative-tangents:");
            }
            if vertex_colors {
                key.push_str("vertex-colors:");
            }
            if flat_shading {
                key.push_str("flat-shading:");
            }
            mat = match self.derived_cache.get(&key) {
                Some(&d) => d,
                None => {
                    let mut d = self.materials[mat].clone();
                    d.rust = None;
                    if vertex_colors {
                        d.params.insert("vertexColors", Param::Bool(true));
                    }
                    if flat_shading {
                        d.params.insert("flatShading", Param::Bool(true));
                    }
                    if derivative_tangents {
                        let pbr = matches!(d.ty, MaterialType::MeshStandardMaterial | MaterialType::MeshPhysicalMaterial);
                        if pbr {
                            let ns = match d.params.get("normalScale") {
                                Some(Param::Vec2(v)) => *v,
                                _ => [1.0, 1.0],
                            };
                            d.params.insert("normalScale", Param::Vec2([ns[0], -ns[1]]));
                        }
                        if d.ty == MaterialType::MeshPhysicalMaterial {
                            let ns = match d.params.get("clearcoatNormalScale") {
                                Some(Param::Vec2(v)) => *v,
                                _ => [1.0, 1.0],
                            };
                            d.params.insert("clearcoatNormalScale", Param::Vec2([ns[0], -ns[1]]));
                        }
                    }
                    d.association = match self.materials[mat].association {
                        Assoc::Absent => Assoc::Undefined,
                        a => a,
                    };
                    self.materials.push(d);
                    let d = self.materials.len() - 1;
                    self.derived_cache.insert(key, d);
                    d
                }
            };
        }
        self.objects[o].material = Some(mat);
    }

    // ------------------------------------------------- cameras, lights, skins

    /// `getDependency('camera', i)` → `loadCamera`.
    fn get_camera(&mut self, arena: &mut ObjectArena, i: usize) -> Result<Option<usize>, GltfError> {
        if let Some(Some(c)) = self.camera_cache.get(i) {
            return Ok(*c);
        }
        let def = doc::def(&self.doc.root, "cameras", i)
            .cloned()
            .ok_or_else(|| GltfError::msg(format!("camera {i} does not exist")))?;
        let ty = doc::string(&def, "type").unwrap_or("");
        let params = doc::get(&def, ty).cloned();
        let result = match (ty, params) {
            ("perspective", Some(p)) => {
                let o = self.new_object(arena, ObjectType::PerspectiveCamera);
                let or = |k: &str, d: f64| doc::num(&p, k).filter(|v| *v != 0.0 && !v.is_nan()).unwrap_or(d);
                self.objects[o].camera = Some(CameraParams::Perspective {
                    fov: doc::num(&p, "yfov").unwrap_or(f64::NAN).to_degrees(),
                    aspect: or("aspectRatio", 1.0),
                    near: or("znear", 1.0),
                    far: or("zfar", 2e6),
                });
                Some(o)
            }
            ("orthographic", Some(p)) => {
                let o = self.new_object(arena, ObjectType::OrthographicCamera);
                let xmag = doc::num(&p, "xmag").unwrap_or(f64::NAN);
                let ymag = doc::num(&p, "ymag").unwrap_or(f64::NAN);
                self.objects[o].camera = Some(CameraParams::Orthographic {
                    left: -xmag,
                    right: xmag,
                    top: ymag,
                    bottom: -ymag,
                    near: doc::num(&p, "znear").unwrap_or(0.1),
                    far: doc::num(&p, "zfar").unwrap_or(2000.0),
                });
                Some(o)
            }
            _ => {
                self.warnings.push("THREE.GLTFLoader: Missing camera parameters.".into());
                None
            }
        };
        if let Some(o) = result {
            if let Some(name) = doc::string(&def, "name").filter(|n| !n.is_empty()) {
                self.objects[o].name = self.create_unique_name(name);
            }
            let mut ud = empty_object();
            assign_extras(&mut ud, &def, &mut self.warnings);
            self.objects[o].user_data = ud;
        }
        if let Some(slot) = self.camera_cache.get_mut(i) {
            *slot = Some(result);
        }
        Ok(result)
    }

    /// `GLTFLightsExtension._loadLight`.
    fn load_light(&mut self, arena: &mut ObjectArena, i: usize) -> Result<usize, GltfError> {
        if let Some(Some(l)) = self.light_cache.get(i) {
            return Ok(*l);
        }
        let def = doc::get(&self.doc.root, "extensions")
            .and_then(|e| doc::get(e, "KHR_lights_punctual"))
            .and_then(|k| doc::get(k, "lights"))
            .and_then(|l| l.as_array())
            .and_then(|l| l.get(i))
            .cloned()
            .ok_or_else(|| GltfError::msg(format!("light {i} does not exist")))?;
        let color = doc::floats(&def, "color", 3, 0.0).unwrap_or(vec![1.0, 1.0, 1.0]);
        let range = doc::num(&def, "range").unwrap_or(0.0) as f32;
        let ty = doc::string(&def, "type").unwrap_or("");
        let mut params = LightParams {
            color: [color[0], color[1], color[2]],
            intensity: doc::num(&def, "intensity").unwrap_or(1.0) as f32,
            distance: 0.0,
            decay: 2.0,
            angle: std::f32::consts::FRAC_PI_3,
            penumbra: 0.0,
        };
        let (o, target) = match ty {
            "directional" => (self.new_object(arena, ObjectType::DirectionalLight), true),
            "point" => {
                params.distance = range;
                (self.new_object(arena, ObjectType::PointLight), false)
            }
            "spot" => {
                params.distance = range;
                let spot = doc::get(&def, "spot").cloned().unwrap_or(Value::Null);
                let inner = doc::num(&spot, "innerConeAngle").unwrap_or(0.0) as f32;
                let outer = doc::num(&spot, "outerConeAngle").unwrap_or(std::f64::consts::FRAC_PI_4) as f32;
                params.angle = outer;
                params.penumbra = 1.0 - inner / outer;
                (self.new_object(arena, ObjectType::SpotLight), true)
            }
            other => {
                return Err(GltfError::msg(format!("THREE.GLTFLoader: Unexpected light type: {other}")));
            }
        };
        self.objects[o].light = Some(params);
        if target {
            let t = self.new_object(arena, ObjectType::Object3D);
            if let Some(n) = arena.get_mut(self.objects[t].id) {
                n.position = Vector3::new(0.0, 0.0, -1.0);
            }
            self.add(arena, o, t);
            self.objects[o].light_target = Some(t);
        }
        let mut ud = empty_object();
        assign_extras(&mut ud, &def, &mut self.warnings);
        self.objects[o].user_data = ud;
        let name = doc::string(&def, "name").filter(|n| !n.is_empty()).map(str::to_string).unwrap_or(format!("light_{i}"));
        self.objects[o].name = self.create_unique_name(&name);
        if let Some(slot) = self.light_cache.get_mut(i) {
            *slot = Some(o);
        }
        Ok(o)
    }

    /// `getDependency('skin', i)` → `loadSkin`.
    fn get_skin(&mut self, arena: &mut ObjectArena, i: usize) -> Result<usize, GltfError> {
        if let Some(Some(s)) = self.skin_cache.get(i) {
            return Ok(*s);
        }
        let def = doc::def(&self.doc.root, "skins", i)
            .cloned()
            .ok_or_else(|| GltfError::msg(format!("skin {i} does not exist")))?;
        let joints: Vec<usize> = doc::array(&def, "joints")
            .iter()
            .map(|j| j.as_number().filter(|j| *j >= 0.0).map(|j| j as usize).unwrap_or(usize::MAX))
            .collect();
        let mut shallow = Vec::new();
        for &j in &joints {
            shallow.push(self.load_node_shallow(arena, j).ok());
        }
        let ibm = match doc::index(&def, "inverseBindMatrices") {
            Some(a) => Some(self.doc.accessor(a)?.values()),
            None => None,
        };
        let mut skeleton = GltfSkeleton { bones: Vec::new(), inverses: Vec::new() };
        for (k, node) in shallow.into_iter().enumerate() {
            match node {
                Some(n) => {
                    skeleton.bones.push(n);
                    let mut m = Matrix4::identity().elements;
                    if let Some(ibm) = &ibm {
                        if let Some(v) = ibm.get(k * 16..k * 16 + 16) {
                            m.copy_from_slice(v);
                        }
                    }
                    skeleton.inverses.push(m);
                }
                None => self.warnings.push(format!("THREE.GLTFLoader: Joint \"{}\" could not be found.", joints[k])),
            }
        }
        self.skeletons.push(skeleton);
        let s = self.skeletons.len() - 1;
        if let Some(slot) = self.skin_cache.get_mut(i) {
            *slot = Some(s);
        }
        Ok(s)
    }

    // ------------------------------------------------------- scenes, animation

    /// `getDependency('scene', i)` → `loadScene`.
    fn get_scene(&mut self, arena: &mut ObjectArena, i: usize) -> Result<usize, GltfError> {
        if let Some(Some(s)) = self.scene_cache.get(i) {
            return Ok(*s);
        }
        let def = doc::def(&self.doc.root, "scenes", i)
            .cloned()
            .ok_or_else(|| GltfError::msg(format!("scene {i} does not exist")))?;
        let group = self.new_object(arena, ObjectType::Group);
        if let Some(name) = doc::string(&def, "name").filter(|n| !n.is_empty()) {
            self.objects[group].name = self.create_unique_name(name);
        }
        let mut ud = empty_object();
        assign_extras(&mut ud, &def, &mut self.warnings);
        if doc::get(&def, "extensions").is_some() {
            add_unknown_extensions(&self.known_extensions, &mut ud, &def);
        }
        self.objects[group].user_data = ud;
        let mut nodes = Vec::new();
        for n in doc::array(&def, "nodes") {
            let n = n.as_number().filter(|n| *n >= 0.0).map(|n| n as usize).unwrap_or(usize::MAX);
            nodes.push(self.get_node(arena, n)?);
        }
        for n in nodes {
            self.add(arena, group, n);
        }
        if let Some(slot) = self.scene_cache.get_mut(i) {
            *slot = Some(group);
        }
        Ok(group)
    }

    /// `getDependency('animation', i)` → `loadAnimation`. The tracks are
    /// made at the end of the load, when every name is known.
    fn get_animation(&mut self, arena: &mut ObjectArena, i: usize) -> Result<usize, GltfError> {
        if let Some(Some(a)) = self.animation_cache.get(i) {
            return Ok(*a);
        }
        let def = doc::def(&self.doc.root, "animations", i)
            .cloned()
            .ok_or_else(|| GltfError::msg(format!("animation {i} does not exist")))?;
        let name = doc::string(&def, "name")
            .filter(|n| !n.is_empty())
            .map(str::to_string)
            .unwrap_or(format!("animation_{i}"));
        let samplers = doc::array(&def, "samplers").to_vec();
        let parameters = doc::get(&def, "parameters").cloned();
        let mut channels = Vec::new();
        for ch in doc::array(&def, "channels") {
            let Some(target) = doc::get(ch, "target") else { continue };
            let Some(node) = doc::index(target, "node") else { continue };
            let Some(sampler) = doc::index(ch, "sampler").and_then(|s| samplers.get(s)) else {
                self.warnings.push(format!("THREE.GLTFLoader: animation {i} has a channel without a sampler"));
                continue;
            };
            let mut input = doc::get(sampler, "input").cloned();
            let mut output = doc::get(sampler, "output").cloned();
            if let Some(p) = &parameters {
                input = input.as_ref().and_then(|k| k.as_str()).and_then(|k| doc::get(p, k)).cloned();
                output = output.as_ref().and_then(|k| k.as_str()).and_then(|k| doc::get(p, k)).cloned();
            }
            let node_obj = self.get_node(arena, node)?;
            let input = input.and_then(|v| v.as_number()).ok_or(GltfError::BadAccessor)? as usize;
            let output = output.and_then(|v| v.as_number()).ok_or(GltfError::BadAccessor)? as usize;
            let interpolation = match doc::string(sampler, "interpolation") {
                Some("STEP") => TrackInterpolation::Step,
                Some("CUBICSPLINE") => TrackInterpolation::CubicSpline,
                _ => TrackInterpolation::Linear,
            };
            channels.push(PendingChannel {
                node: node_obj,
                path: doc::string(target, "path").unwrap_or("").to_string(),
                input: self.doc.accessor(input)?,
                output: self.doc.accessor(output)?,
                interpolation,
            });
        }
        self.clips.push(GltfClip { name, tracks: Vec::new() });
        let c = self.clips.len() - 1;
        self.pending_channels.push((c, channels));
        if let Some(slot) = self.animation_cache.get_mut(i) {
            *slot = Some(c);
        }
        Ok(c)
    }

    /// `_createAnimationTracks` for every channel loaded so far.
    fn expand_tracks(&mut self, arena: &ObjectArena) {
        let pending = std::mem::take(&mut self.pending_channels);
        for (clip, channels) in pending {
            let mut tracks = Vec::new();
            for ch in channels {
                let (property, kind) = match ch.path.as_str() {
                    "translation" => ("position", TrackKind::Vector),
                    "rotation" => ("quaternion", TrackKind::Quaternion),
                    "scale" => ("scale", TrackKind::Vector),
                    "weights" => ("morphTargetInfluences", TrackKind::Number),
                    _ => (
                        "undefined",
                        if ch.output.item_size == 1 { TrackKind::Number } else { TrackKind::Vector },
                    ),
                };
                let targets: Vec<usize> = if ch.path == "weights" {
                    let mut t = Vec::new();
                    self.traverse(arena, ch.node, &mut |o| t.push(o));
                    t.into_iter().filter(|&o| self.objects[o].morph_influences.is_some()).collect()
                } else {
                    vec![ch.node]
                };
                let times = Arc::new(ch.input.raw.clone());
                let values = Arc::new(ch.output.scaled());
                for target in targets {
                    tracks.push(GltfTrack {
                        target,
                        property: property.to_string(),
                        kind,
                        times: times.clone(),
                        values: values.clone(),
                        interpolation: ch.interpolation,
                        path: ch.path.clone(),
                    });
                }
            }
            self.clips[clip].tracks = tracks;
        }
    }

    // ------------------------------------------------------------------ names

    /// Name every mesh (three.js's `createUniqueName` calls in `loadMesh`'s
    /// resolution), then the clones, then put the nodes' own names on.
    fn resolve_names(&mut self) {
        let jobs = std::mem::take(&mut self.mesh_name_jobs);
        for delayed in [false, true] {
            for (mesh, prims, d) in &jobs {
                if *d != delayed {
                    continue;
                }
                let base = doc::def(&self.doc.root, "meshes", *mesh)
                    .and_then(|m| doc::string(m, "name"))
                    .filter(|n| !n.is_empty())
                    .map(str::to_string)
                    .unwrap_or(format!("mesh_{mesh}"));
                for &p in prims {
                    self.objects[p].name = self.create_unique_name(&base);
                    self.objects[p].name_spec = NameSpec::Fixed;
                }
            }
        }
        for i in 0..self.objects.len() {
            if let NameSpec::Clone { source, suffix } = self.objects[i].name_spec.clone() {
                let mut name = self.base_name(source);
                if let Some(n) = suffix {
                    name.push_str(&format!("_instance_{n}"));
                }
                self.objects[i].name = name;
                self.objects[i].name_spec = NameSpec::Fixed;
            }
        }
        for o in &mut self.objects {
            if let Some(n) = o.name_override.take() {
                o.name = n;
            }
        }
    }

    fn base_name(&self, i: usize) -> String {
        match &self.objects[i].name_spec {
            NameSpec::Clone { source, suffix } => {
                let mut n = self.base_name(*source);
                if let Some(s) = suffix {
                    n.push_str(&format!("_instance_{s}"));
                }
                n
            }
            _ => self.objects[i].name.clone(),
        }
    }

    // --------------------------------------------------------------- finalize

    /// Build the renderer's geometry, textures and materials, and turn every
    /// new node into what its object is.
    fn finalize(&mut self, arena: &mut ObjectArena) {
        for g in &mut self.geometries {
            g.build();
        }
        for m in 0..self.materials.len() {
            if self.materials[m].rust.is_none() {
                let built = build_rust_material(&self.materials[m], &mut self.textures);
                self.materials[m].rust = Some(Arc::new(built));
            }
        }
        let fallback = Arc::new(Material::Standard(StandardMaterial::new(Color::WHITE)));
        for i in 0..self.objects.len() {
            let o = self.objects[i].clone();
            let Some(node) = arena.get_mut(o.id) else { continue };
            node.name = o.name.clone();
            if o.finalized {
                continue;
            }
            let geometry = o.geometry.and_then(|g| self.geometries[g].arc.clone());
            let material = o
                .material
                .and_then(|m| self.materials[m].rust.clone())
                .unwrap_or_else(|| fallback.clone());
            let dictionary: HashMap<String, usize> = o.morph_dictionary.iter().cloned().collect();
            let influences = o.morph_influences.clone().unwrap_or_default();
            match (o.ty, geometry) {
                (ObjectType::Mesh, Some(g)) => {
                    let mut m = Mesh::from_arc(g, material);
                    m.morph_influences = influences;
                    m.morph_target_dictionary = dictionary;
                    node.kind = ObjectKind::Mesh(m);
                }
                (ObjectType::SkinnedMesh, Some(g)) => {
                    let mut sm = SkinnedMesh::from_arc(g, material, Arc::new(Skeleton::default()));
                    sm.morph_influences = influences;
                    sm.morph_target_dictionary = dictionary;
                    sm.bind_matrix = Matrix4::identity();
                    sm.bind_mode = BindMode::Attached;
                    node.kind = ObjectKind::SkinnedMesh(sm);
                }
                (ObjectType::InstancedMesh, Some(g)) => {
                    let inst = o.instances.clone().unwrap_or(Instances { count: 0, matrices: Vec::new(), colors: None });
                    let mut im = InstancedMesh::from_arc(g, material, inst.count);
                    im.transforms = inst
                        .matrices
                        .chunks_exact(16)
                        .map(|c| {
                            let mut e = [0.0; 16];
                            e.copy_from_slice(c);
                            Matrix4 { elements: e }
                        })
                        .collect();
                    if let Some((c, size)) = &inst.colors {
                        im.colors = c
                            .chunks(*size.max(&1))
                            .map(|v| Color::new(v[0], *v.get(1).unwrap_or(&0.0), *v.get(2).unwrap_or(&0.0)))
                            .collect();
                    }
                    im.morph_influences = influences;
                    node.kind = ObjectKind::InstancedMesh(im);
                }
                (ObjectType::LineSegments, Some(g)) => {
                    node.kind = ObjectKind::LineSegments(LineSegments::from_arc(g, material));
                }
                (ObjectType::Line, Some(g)) => {
                    node.kind = ObjectKind::LineSegments(LineSegments::strip(g, material, false));
                }
                (ObjectType::LineLoop, Some(g)) => {
                    node.kind = ObjectKind::LineSegments(LineSegments::strip(g, material, true));
                }
                (ObjectType::Points, Some(g)) => {
                    node.kind = ObjectKind::Points(Points { geometry: g, material });
                }
                (ty, _) if ty.is_light() => {
                    let p = o.light.unwrap_or(LightParams {
                        color: [1.0; 3],
                        intensity: 1.0,
                        distance: 0.0,
                        decay: 2.0,
                        angle: 0.0,
                        penumbra: 0.0,
                    });
                    let color = Color::new(p.color[0], p.color[1], p.color[2]);
                    node.kind = ObjectKind::Light(match ty {
                        ObjectType::DirectionalLight => {
                            Light::Directional(DirectionalLight::new(color, p.intensity).with_direction(Vector3::new(0.0, 0.0, -1.0)))
                        }
                        ObjectType::PointLight => {
                            let mut l = PointLight::new(color, p.intensity);
                            l.distance = p.distance;
                            l.decay = p.decay;
                            Light::Point(l)
                        }
                        _ => {
                            let mut l = SpotLight::new(color, p.intensity);
                            l.direction = Vector3::new(0.0, 0.0, -1.0);
                            l.distance = p.distance;
                            l.decay = p.decay;
                            l.angle = p.angle;
                            l.penumbra = p.penumbra;
                            Light::Spot(l)
                        }
                    });
                }
                _ => {}
            }
            self.objects[i].finalized = true;
        }
        // Skeletons reference bones by node, so they go in once every node is.
        for i in 0..self.objects.len() {
            let Some(s) = self.objects[i].skeleton else { continue };
            let sk = &self.skeletons[s];
            let bones: Vec<SkinBone> = sk
                .bones
                .iter()
                .zip(&sk.inverses)
                .map(|(&b, inv)| SkinBone { node: self.objects[b].id, inverse_bind: Matrix4 { elements: *inv } })
                .collect();
            if let Some(ObjectKind::SkinnedMesh(sm)) = arena.get_mut(self.objects[i].id).map(|n| &mut n.kind) {
                sm.skeleton = Arc::new(Skeleton::new(bones));
            }
        }
        // A cloned light's detached target: aim at it from where the light is.
        for i in 0..self.objects.len() {
            let Some(t) = self.objects[i].light_target else { continue };
            let lid = self.objects[i].id;
            let tid = self.objects[t].id;
            if arena.get(tid).and_then(|n| n.parent) == Some(lid) {
                continue;
            }
            let lw = world_matrix(arena, lid);
            let tw = world_matrix(arena, tid);
            let d = [tw.elements[12] - lw.elements[12], tw.elements[13] - lw.elements[13], tw.elements[14] - lw.elements[14]];
            let inv = lw.invert().elements;
            let dir = Vector3::new(
                inv[0] * d[0] + inv[4] * d[1] + inv[8] * d[2],
                inv[1] * d[0] + inv[5] * d[1] + inv[9] * d[2],
                inv[2] * d[0] + inv[6] * d[1] + inv[10] * d[2],
            );
            if let Some(ObjectKind::Light(l)) = arena.get_mut(lid).map(|n| &mut n.kind) {
                match l {
                    Light::Directional(d) => d.direction = dir,
                    Light::Spot(s) => s.direction = dir,
                    _ => {}
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum RefKind {
    Mesh,
    Camera,
    Light,
}

/// A node's world matrix, from its ancestors' local transforms.
pub fn world_matrix(arena: &ObjectArena, id: ObjectId) -> Matrix4 {
    let mut chain = Vec::new();
    let mut cur = Some(id);
    let mut guard = 0;
    while let Some(c) = cur {
        let Some(n) = arena.get(c) else { break };
        chain.push(if n.matrix_auto_update { local_matrix(n) } else { n.matrix });
        cur = n.parent;
        guard += 1;
        if guard > 100_000 {
            break;
        }
    }
    let mut m = Matrix4::identity();
    for local in chain.iter().rev() {
        m = m.multiply(local);
    }
    m
}

/// `addUnknownExtensionsToUserData`.
fn add_unknown_extensions(known: &HashSet<String>, user_data: &mut Value, def: &Value) {
    let Some(exts) = doc::get(def, "extensions").and_then(|e| e.as_object()) else { return };
    for (name, value) in exts {
        if known.contains(name) {
            continue;
        }
        let Some(ud) = user_data.as_object_mut() else { return };
        let entry = ud
            .entry("gltfExtensions".to_string())
            .or_insert_with(|| Value::Object(HashMap::new()));
        if let Some(e) = entry.as_object_mut() {
            e.insert(name.clone(), value.clone());
        }
    }
}

/// JS `String(v)` for a `targetNames` entry.
fn js_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => js_index(Some(other)),
    }
}

/// Attribute order: glTF's JSON order is lost to the parser's map, so use
/// the conventional order (which is how exporters write them).
fn attribute_rank(name: &str) -> (u32, String) {
    let r = match name {
        "POSITION" => 0,
        "NORMAL" => 1,
        "TANGENT" => 2,
        "TEXCOORD_0" => 3,
        "TEXCOORD_1" => 4,
        "TEXCOORD_2" => 5,
        "TEXCOORD_3" => 6,
        "COLOR_0" => 7,
        "JOINTS_0" => 8,
        "WEIGHTS_0" => 9,
        _ => 10,
    };
    (r, name.to_string())
}

/// Draco's `ConvertValue<float>`: normalised integers divided by their type's
/// maximum, other integers as they are.
fn draco_float(a: &super::super::draco::attributes::DecodedAttribute) -> Vec<f32> {
    use super::super::draco::attributes::AttributeValues;
    let max = match a.descriptor.data_type {
        1 => Some(i8::MAX as f32),
        2 => Some(u8::MAX as f32),
        3 => Some(i16::MAX as f32),
        4 => Some(u16::MAX as f32),
        5 => Some(i32::MAX as f32),
        6 => Some(u32::MAX as f32),
        _ => None,
    };
    match (&a.values, a.descriptor.normalized, max) {
        (AttributeValues::Integer(v), true, Some(max)) => v.iter().map(|&x| x as f32 / max).collect(),
        _ => a.as_f32(),
    }
}

/// Every texture reference in a material definition three.js would read.
fn collect_material_textures(def: &Value, out: &mut Vec<usize>) {
    let mut push = |info: Option<&Value>| {
        if let Some(i) = info.and_then(|i| doc::index(i, "index")) {
            out.push(i);
        }
    };
    let pbr = doc::get(def, "pbrMetallicRoughness");
    let unlit = doc::extension(def, "KHR_materials_unlit").is_some();
    push(pbr.and_then(|p| doc::get(p, "baseColorTexture")));
    if unlit {
        return;
    }
    push(pbr.and_then(|p| doc::get(p, "metallicRoughnessTexture")));
    push(doc::get(def, "normalTexture"));
    push(doc::get(def, "occlusionTexture"));
    push(doc::get(def, "emissiveTexture"));
    let ext_textures: &[(&str, &[&str])] = &[
        ("KHR_materials_clearcoat", &["clearcoatTexture", "clearcoatRoughnessTexture", "clearcoatNormalTexture"]),
        ("KHR_materials_sheen", &["sheenColorTexture", "sheenRoughnessTexture"]),
        ("KHR_materials_transmission", &["transmissionTexture"]),
        ("KHR_materials_volume", &["thicknessTexture"]),
        ("KHR_materials_specular", &["specularTexture", "specularColorTexture"]),
        ("KHR_materials_iridescence", &["iridescenceTexture", "iridescenceThicknessTexture"]),
        ("KHR_materials_anisotropy", &["anisotropyTexture"]),
        ("EXT_materials_bump", &["bumpTexture"]),
    ];
    for (ext, keys) in ext_textures {
        if let Some(e) = doc::extension(def, ext) {
            for k in *keys {
                push(doc::get(e, k));
            }
        }
    }
}
