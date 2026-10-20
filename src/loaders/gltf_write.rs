//! Writing glTF 2.0 — `.gltf` with its buffer alongside, or a self-contained
//! `.glb`.
//!
//! The loader has read glTF since the beginning and there was no way to write
//! it, which makes the format a one-way door: a scene built or edited here
//! could not go back to the tools it came from.
//!
//! What goes out: the scene graph with its transforms, meshes with positions,
//! normals, texture coordinates and vertex colours, indices, and materials as
//! `pbrMetallicRoughness` with their textures embedded as PNG. Everything is
//! written through a JSON value type rather than by concatenating strings, because a
//! glTF that is subtly malformed loads in some readers and not others, which is
//! worse than one that fails everywhere.

use std::collections::HashMap;

use crate::core::{BufferGeometry, ObjectArena, ObjectId, ObjectKind};
use crate::loaders::json::Value;
use crate::materials::Material;

/// What to put in the file.
#[derive(Debug, Clone)]
pub struct GltfExportOptions {
    /// The `asset.generator` string.
    pub generator: String,
    /// Write normals when the geometry has them.
    pub normals: bool,
    /// Write texture coordinates when the geometry has them.
    pub uvs: bool,
    /// Write vertex colours when the geometry has them.
    pub vertex_colours: bool,
    /// Embed each material's maps as PNG. Off keeps the file small and the
    /// materials flat.
    pub textures: bool,
    /// Node transforms as `translation` / `rotation` / `scale` (the default),
    /// or as one column-major `matrix` when off — three.js `GLTFExporter`'s
    /// `trs: false`, its default.
    pub trs: bool,
    /// Follow three.js r165 `GLTFExporter`'s material and mesh conventions:
    /// `baseColorFactor` left out when white and opaque, `MeshBasicMaterial`
    /// as `KHR_materials_unlit`, `alphaMode: BLEND` from `transparent` rather
    /// than from the opacity, one material per distinct material (by `Arc`)
    /// and one mesh per distinct geometry *and* material.
    pub three_js: bool,
    /// `scenes[0].name`, when set.
    pub scene_name: Option<String>,
    /// three.js mode: what to write for a material, keyed by its `Arc`
    /// pointer (`Arc::as_ptr(&material) as usize`), for a caller whose
    /// materials carry more than [`Material`] does — a name, `transparent`,
    /// `alphaTest` on any kind. Missing entries are read off the material.
    pub material_info: HashMap<usize, GltfMaterialInfo>,
    /// `.gltf` only: carry the buffer inline as a base64 `data:` URI, as
    /// three.js does, instead of naming a `scene.bin` beside it.
    pub embed_buffer: bool,
}

impl Default for GltfExportOptions {
    fn default() -> Self {
        Self {
            generator: format!("threers {}", env!("CARGO_PKG_VERSION")),
            normals: true,
            uvs: true,
            vertex_colours: true,
            textures: true,
            trs: true,
            three_js: false,
            scene_name: None,
            material_info: HashMap::new(),
            embed_buffer: false,
        }
    }
}

/// Which three.js material family a material belongs to, for the three.js
/// mode's metal/rough defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GltfMaterialKind {
    /// `MeshStandardMaterial` / `MeshPhysicalMaterial`: its own factors.
    #[default]
    Standard,
    /// `MeshBasicMaterial`: `KHR_materials_unlit`, metal 0, rough 0.9.
    Basic,
    /// Anything else: metal 0.5, rough 0.5, as three.js writes it.
    Other,
}

/// The material values three.js r165's `GLTFExporter` reads.
#[derive(Debug, Clone, Default)]
pub struct GltfMaterialInfo {
    pub name: String,
    pub kind: GltfMaterialKind,
    /// Linear RGB, as `material.color` holds it.
    pub color: [f32; 3],
    pub opacity: f32,
    pub transparent: bool,
    pub metalness: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub alpha_test: f32,
    pub double_sided: bool,
}

impl GltfMaterialInfo {
    /// Read off a [`Material`].
    pub fn from_material(material: &Material) -> Self {
        let c = material.color();
        let e = material.emissive();
        let (kind, metalness, roughness) = match material {
            Material::Standard(m) => (GltfMaterialKind::Standard, m.metalness, m.roughness),
            Material::Physical(m) => (GltfMaterialKind::Standard, m.metalness, m.roughness),
            Material::Basic(_) => (GltfMaterialKind::Basic, 0.0, 0.9),
            _ => (GltfMaterialKind::Other, 0.5, 0.5),
        };
        Self {
            name: String::new(),
            kind,
            color: [c.r, c.g, c.b],
            opacity: material.opacity(),
            transparent: material.transparent(),
            metalness,
            roughness,
            emissive: [e.r, e.g, e.b],
            alpha_test: material.alpha_test(),
            double_sided: material.side() == 2,
        }
    }
}

/// A scene as a `.glb` — one file, nothing beside it.
pub fn scene_to_glb(arena: &ObjectArena, roots: &[ObjectId]) -> Vec<u8> {
    scene_to_glb_with(arena, roots, &GltfExportOptions::default())
}

pub fn scene_to_glb_with(
    arena: &ObjectArena,
    roots: &[ObjectId],
    options: &GltfExportOptions,
) -> Vec<u8> {
    let (json, bin) = build(arena, roots, options);
    package(&json.to_json(), &bin)
}

/// A scene as `.gltf` text, with the binary buffer returned beside it.
///
/// The JSON names the buffer `scene.bin`, so that is what the bytes have to be
/// written as for the pair to resolve.
pub fn scene_to_gltf(arena: &ObjectArena, roots: &[ObjectId]) -> (String, Vec<u8>) {
    scene_to_gltf_with(arena, roots, &GltfExportOptions::default())
}

pub fn scene_to_gltf_with(
    arena: &ObjectArena,
    roots: &[ObjectId],
    options: &GltfExportOptions,
) -> (String, Vec<u8>) {
    let (mut json, bin) = build(arena, roots, options);
    if let Some(object) = json.as_object_mut() {
        let uri = if options.embed_buffer {
            format!("data:application/octet-stream;base64,{}", base64(&bin))
        } else {
            "scene.bin".to_string()
        };
        if bin.is_empty() {
            object.remove("buffers");
        } else {
            object.insert(
                "buffers".into(),
                Value::Array(vec![Value::object([
                    ("byteLength", Value::number(bin.len() as f64)),
                    ("uri", Value::string(uri)),
                ])]),
            );
        }
    }
    (json.to_json(), bin)
}

/// One geometry as a `.glb`, with no scene around it.
pub fn geometry_to_glb(geometry: &BufferGeometry) -> Vec<u8> {
    let mut arena = ObjectArena::new();
    let id = arena.insert(crate::core::Object3D::mesh(crate::core::Mesh::new(
        geometry.clone(),
        crate::materials::StandardMaterial::new(crate::math::Color::new(0.8, 0.8, 0.8)).into(),
    )));
    scene_to_glb(&arena, &[id])
}

/// The buffer under construction, and where each piece of it landed.
#[derive(Default)]
struct Writer {
    bin: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
    meshes: Vec<Value>,
    materials: Vec<Value>,
    images: Vec<Value>,
    textures: Vec<Value>,
    samplers: Vec<Value>,
    nodes: Vec<Value>,
    /// Geometries already written, so a mesh used twice costs one copy.
    interned: HashMap<(usize, Option<usize>), usize>,
    /// The same for images: a texture shared by ten materials is one PNG.
    interned_textures: HashMap<usize, usize>,
    /// Materials already written, by `Arc` pointer (three.js mode only).
    interned_materials: HashMap<usize, usize>,
    extensions_used: std::collections::BTreeSet<String>,
}

impl Writer {
    /// Append a view over freshly written bytes, 4-byte aligned.
    ///
    /// glTF requires each accessor's offset to be a multiple of its component
    /// size; padding every view to 4 satisfies that for everything up to a
    /// float without having to reason about it per attribute.
    fn view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }
        let offset = self.bin.len();
        self.bin.extend_from_slice(bytes);
        let mut pairs = vec![
            ("buffer".to_string(), Value::number(0.0)),
            ("byteOffset".to_string(), Value::number(offset as f64)),
            ("byteLength".to_string(), Value::number(bytes.len() as f64)),
        ];
        if let Some(target) = target {
            pairs.push(("target".to_string(), Value::number(target as f64)));
        }
        self.views.push(Value::Object(pairs.into_iter().collect()));
        self.views.len() - 1
    }

    fn accessor(&mut self, value: Value) -> usize {
        self.accessors.push(value);
        self.accessors.len() - 1
    }

    /// A float attribute. `POSITION` additionally needs `min`/`max`, which
    /// validators enforce and which a viewer uses to frame the scene.
    fn floats(&mut self, data: &[f32], components: usize, bounds: bool) -> usize {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let view = self.view(&bytes, Some(34962));
        let count = data.len() / components;
        let kind = match components {
            1 => "SCALAR",
            2 => "VEC2",
            3 => "VEC3",
            _ => "VEC4",
        };
        let mut pairs = vec![
            ("bufferView".to_string(), Value::number(view as f64)),
            ("componentType".to_string(), Value::number(5126.0)),
            ("count".to_string(), Value::number(count as f64)),
            ("type".to_string(), Value::string(kind)),
        ];
        if bounds && count > 0 {
            let mut lo = vec![f32::MAX; components];
            let mut hi = vec![f32::MIN; components];
            for element in data.chunks_exact(components) {
                for k in 0..components {
                    lo[k] = lo[k].min(element[k]);
                    hi[k] = hi[k].max(element[k]);
                }
            }
            let to_array =
                |v: &[f32]| Value::Array(v.iter().map(|x| Value::number(*x as f64)).collect());
            pairs.push(("min".to_string(), to_array(&lo)));
            pairs.push(("max".to_string(), to_array(&hi)));
        }
        self.accessor(Value::Object(pairs.into_iter().collect()))
    }

    fn indices(&mut self, data: &[u32]) -> usize {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let view = self.view(&bytes, Some(34963));
        self.accessor(Value::object([
            ("bufferView", Value::number(view as f64)),
            ("componentType", Value::number(5125.0)),
            ("count", Value::number(data.len() as f64)),
            ("type", Value::string("SCALAR")),
        ]))
    }
}

fn build(arena: &ObjectArena, roots: &[ObjectId], options: &GltfExportOptions) -> (Value, Vec<u8>) {
    let mut w = Writer::default();
    let mut scene_roots = Vec::new();
    for root in roots {
        if let Some(node) = write_node(&mut w, arena, *root, options) {
            scene_roots.push(Value::number(node as f64));
        }
    }

    let mut scene = vec![("nodes".to_string(), Value::Array(scene_roots))];
    if let Some(name) = &options.scene_name {
        scene.push(("name".to_string(), Value::string(name.clone())));
    }
    let mut json = vec![
        (
            "asset".to_string(),
            Value::object([
                ("version", Value::string("2.0")),
                ("generator", Value::string(options.generator.clone())),
            ]),
        ),
        ("scene".to_string(), Value::number(0.0)),
        (
            "scenes".to_string(),
            Value::Array(vec![Value::Object(scene.into_iter().collect())]),
        ),
        ("nodes".to_string(), Value::Array(w.nodes)),
        (
            "buffers".to_string(),
            Value::Array(vec![Value::object([(
                "byteLength",
                Value::number(w.bin.len() as f64),
            )])]),
        ),
    ];
    if !w.extensions_used.is_empty() {
        json.push((
            "extensionsUsed".to_string(),
            Value::Array(
                w.extensions_used
                    .iter()
                    .map(|e| Value::string(e.clone()))
                    .collect(),
            ),
        ));
    }
    // Empty arrays are invalid in glTF, so each is written only if used.
    for (key, list) in [
        ("bufferViews", w.views),
        ("accessors", w.accessors),
        ("meshes", w.meshes),
        ("materials", w.materials),
        ("images", w.images),
        ("textures", w.textures),
        ("samplers", w.samplers),
    ] {
        if !list.is_empty() {
            json.push((key.to_string(), Value::Array(list)));
        }
    }
    (Value::Object(json.into_iter().collect()), w.bin)
}

fn write_node(
    w: &mut Writer,
    arena: &ObjectArena,
    id: ObjectId,
    options: &GltfExportOptions,
) -> Option<usize> {
    let object = arena.get(id)?;
    let mut pairs: Vec<(String, Value)> = Vec::new();
    if !object.name.is_empty() {
        pairs.push(("name".to_string(), Value::string(object.name.clone())));
    }
    // TRS rather than a matrix by default: it survives round trips through
    // tools that decompose, and a uniform scale stays uniform.
    let p = object.position;
    if !options.trs {
        let q = object.quaternion;
        let s = object.scale;
        let m = crate::exporters::compose(
            [p.x as f64, p.y as f64, p.z as f64],
            [q.x as f64, q.y as f64, q.z as f64, q.w as f64],
            [s.x as f64, s.y as f64, s.z as f64],
        );
        if m != crate::exporters::IDENTITY {
            pairs.push((
                "matrix".to_string(),
                Value::Array(m.iter().map(|v| Value::number(*v)).collect()),
            ));
        }
    } else if p.x != 0.0 || p.y != 0.0 || p.z != 0.0 {
        pairs.push((
            "translation".to_string(),
            Value::Array(vec![
                Value::number(p.x as f64),
                Value::number(p.y as f64),
                Value::number(p.z as f64),
            ]),
        ));
    }
    let q = object.quaternion;
    if options.trs && (q.x != 0.0 || q.y != 0.0 || q.z != 0.0 || q.w != 1.0) {
        pairs.push((
            "rotation".to_string(),
            Value::Array(vec![
                Value::number(q.x as f64),
                Value::number(q.y as f64),
                Value::number(q.z as f64),
                Value::number(q.w as f64),
            ]),
        ));
    }
    let s = object.scale;
    if options.trs && (s.x != 1.0 || s.y != 1.0 || s.z != 1.0) {
        pairs.push((
            "scale".to_string(),
            Value::Array(vec![
                Value::number(s.x as f64),
                Value::number(s.y as f64),
                Value::number(s.z as f64),
            ]),
        ));
    }

    if let ObjectKind::Mesh(mesh) = &object.kind {
        let material_of = |w: &mut Writer, m: &std::sync::Arc<Material>| {
            if options.three_js {
                let key = std::sync::Arc::as_ptr(m) as usize;
                match w.interned_materials.get(&key) {
                    Some(found) => Some(*found),
                    None => {
                        let built = write_material_three_js(w, m, options);
                        w.interned_materials.insert(key, built);
                        Some(built)
                    }
                }
            } else {
                write_material(w, m, options)
            }
        };
        // A material array: one primitive per group, sharing the attributes,
        // as three.js's `GLTFExporter.processMesh` writes it.
        if mesh.is_multi_material() {
            let mut prims = Vec::new();
            for g in &mesh.geometry.groups {
                let Some(m) = mesh.materials.get(g.material_index as usize) else {
                    continue;
                };
                let material = material_of(w, m);
                prims.push((material, Some((g.start, g.count))));
            }
            // No groups: no primitives, and three.js writes no mesh.
            if !prims.is_empty() {
                let index = write_mesh(w, &mesh.geometry, &prims, Some(mesh), options);
                pairs.push(("mesh".to_string(), Value::number(index as f64)));
            }
        }
        let material = if mesh.is_multi_material() {
            None
        } else {
            material_of(w, &mesh.material)
        };
        // three.js keys its mesh cache on geometry and material together; a
        // material per node (the native default) keys on geometry alone.
        let key = (
            std::sync::Arc::as_ptr(&mesh.geometry) as usize,
            if options.three_js { material } else { None },
        );
        // The same geometry on two nodes is written once, which is what a
        // forest of identical parts costs if you let it.
        if !mesh.is_multi_material() {
            let index = match w.interned.get(&key) {
                Some(found) => *found,
                None => {
                    let built =
                        write_mesh(w, &mesh.geometry, &[(material, None)], Some(mesh), options);
                    w.interned.insert(key, built);
                    built
                }
            };
            pairs.push(("mesh".to_string(), Value::number(index as f64)));
        }
    }

    let mut children = Vec::new();
    for child in &object.children {
        if let Some(node) = write_node(w, arena, *child, options) {
            children.push(Value::number(node as f64));
        }
    }
    if !children.is_empty() {
        pairs.push(("children".to_string(), Value::Array(children)));
    }

    w.nodes.push(Value::Object(pairs.into_iter().collect()));
    Some(w.nodes.len() - 1)
}

/// One glTF mesh: the geometry's attributes once, and a primitive per
/// `(material, group span)` — the span `(start, count)` of the index, `None`
/// for the whole geometry. Morph targets (POSITION and NORMAL, as three.js's
/// exporter writes; absolute targets are converted to the displacements glTF
/// stores) go on every primitive, with the mesh's influences as `weights` and
/// its dictionary as `extras.targetNames`.
/// One glTF primitive: its material, and the index range it draws (`None`
/// for the whole geometry).
type Primitive = (Option<usize>, Option<(u32, u32)>);

fn write_mesh(
    w: &mut Writer,
    geometry: &BufferGeometry,
    prims: &[Primitive],
    mesh: Option<&crate::core::Mesh>,
    options: &GltfExportOptions,
) -> usize {
    let mut attributes: Vec<(String, Value)> = Vec::new();
    if let Some(position) = geometry.get_attribute("position") {
        let a = w.floats(&position.array, 3, true);
        attributes.push(("POSITION".to_string(), Value::number(a as f64)));
    }
    if options.normals {
        if let Some(normal) = geometry.get_attribute("normal") {
            let a = w.floats(&normal.array, 3, options.three_js);
            attributes.push(("NORMAL".to_string(), Value::number(a as f64)));
        }
    }
    if options.uvs {
        if let Some(uv) = geometry.get_attribute("uv") {
            let a = w.floats(&uv.array, 2, options.three_js);
            attributes.push(("TEXCOORD_0".to_string(), Value::number(a as f64)));
        }
    }
    if options.vertex_colours {
        if let Some(colour) = geometry.get_attribute("color") {
            let a = w.floats(&colour.array, 3, options.three_js);
            attributes.push(("COLOR_0".to_string(), Value::number(a as f64)));
        }
    }

    // Morph targets, as displacements.
    let mut targets: Vec<Vec<(String, Value)>> = Vec::new();
    for (ours, theirs) in [("position", "POSITION"), ("normal", "NORMAL")] {
        if ours == "normal" && !options.normals {
            continue;
        }
        let (Some(list), Some(base)) = (
            geometry.morph_attributes.get(ours),
            geometry.get_attribute(ours),
        ) else {
            continue;
        };
        for (t, attr) in list.iter().enumerate() {
            let count = base.count();
            let mut delta = Vec::with_capacity(count * 3);
            for v in 0..count {
                for k in 0..3 {
                    let tv = attr
                        .array
                        .get(v * attr.item_size + k)
                        .copied()
                        .unwrap_or(0.0);
                    let bv = base
                        .array
                        .get(v * base.item_size + k)
                        .copied()
                        .unwrap_or(0.0);
                    delta.push(if geometry.morph_targets_relative {
                        tv
                    } else {
                        tv - bv
                    });
                }
            }
            let a = w.floats(&delta, 3, ours == "position");
            if targets.len() <= t {
                targets.resize_with(t + 1, Vec::new);
            }
            targets[t].push((theirs.to_string(), Value::number(a as f64)));
        }
    }

    // A material array over unindexed geometry gets an index, so each group
    // can be a primitive — three.js does the same.
    let index: Option<Vec<u32>> = match &geometry.index {
        Some(i) => Some(i.clone()),
        None if prims.iter().any(|p| p.1.is_some()) => {
            Some((0..geometry.draw_count() as u32).collect())
        }
        None => None,
    };
    let whole = index.as_ref().map(|i| w.indices(i));
    let attributes = Value::Object(attributes.into_iter().collect());
    let mut primitives = Vec::new();
    for &(material, span) in prims {
        let mut primitive: Vec<(String, Value)> = vec![
            ("attributes".to_string(), attributes.clone()),
            ("mode".to_string(), Value::number(4.0)),
        ];
        let indices = match (span, &index) {
            (Some((start, count)), Some(index)) => {
                let end = (start as usize + count as usize).min(index.len());
                let start = (start as usize).min(end);
                Some(w.indices(&index[start..end]))
            }
            _ => whole,
        };
        if let Some(a) = indices {
            primitive.push(("indices".to_string(), Value::number(a as f64)));
        }
        if let Some(material) = material {
            primitive.push(("material".to_string(), Value::number(material as f64)));
        }
        if !targets.is_empty() {
            primitive.push((
                "targets".to_string(),
                Value::Array(
                    targets
                        .iter()
                        .map(|t| Value::Object(t.iter().cloned().collect()))
                        .collect(),
                ),
            ));
        }
        primitives.push(Value::Object(primitive.into_iter().collect()));
    }
    let mut mesh_pairs: Vec<(String, Value)> =
        vec![("primitives".to_string(), Value::Array(primitives))];
    if let (Some(mesh), false) = (mesh, targets.is_empty()) {
        let n = targets.len();
        let weights = (0..n)
            .map(|i| Value::number(mesh.morph_influences.get(i).copied().unwrap_or(0.0) as f64))
            .collect();
        mesh_pairs.push(("weights".to_string(), Value::Array(weights)));
        // three.js writes `extras.targetNames` from the reversed dictionary.
        let mut names = vec![String::new(); n];
        let mut named = false;
        for (name, &i) in &mesh.morph_target_dictionary {
            if i < n {
                names[i] = name.clone();
                named = true;
            }
        }
        if named {
            mesh_pairs.push((
                "extras".to_string(),
                Value::object([(
                    "targetNames",
                    Value::Array(names.iter().map(Value::string).collect()),
                )]),
            ));
        }
    }
    w.meshes
        .push(Value::Object(mesh_pairs.into_iter().collect()));
    w.meshes.len() - 1
}

fn write_material(
    w: &mut Writer,
    material: &Material,
    options: &GltfExportOptions,
) -> Option<usize> {
    let colour = material.color();
    let opacity = material.opacity();
    let (roughness, metallic) = match material {
        Material::Standard(m) => (m.roughness, m.metalness),
        Material::Physical(m) => (m.roughness, m.metalness),
        // Unlit and flat materials have no meaningful metal-rough pair; a fully
        // rough dielectric is the closest thing glTF has.
        _ => (1.0, 0.0),
    };

    let mut pbr: Vec<(String, Value)> = vec![
        (
            "baseColorFactor".to_string(),
            Value::Array(vec![
                Value::number(colour.r as f64),
                Value::number(colour.g as f64),
                Value::number(colour.b as f64),
                Value::number(opacity as f64),
            ]),
        ),
        ("metallicFactor".to_string(), Value::number(metallic as f64)),
        (
            "roughnessFactor".to_string(),
            Value::number(roughness as f64),
        ),
    ];

    if options.textures {
        if let Some(texture) = base_colour_map(material) {
            if let Some(index) = write_texture(w, &texture) {
                pbr.push((
                    "baseColorTexture".to_string(),
                    Value::object([("index", Value::number(index as f64))]),
                ));
            }
        }
    }

    let mut pairs: Vec<(String, Value)> = vec![(
        "pbrMetallicRoughness".to_string(),
        Value::Object(pbr.into_iter().collect()),
    )];
    let emissive = material.emissive();
    if emissive.r > 0.0 || emissive.g > 0.0 || emissive.b > 0.0 {
        pairs.push((
            "emissiveFactor".to_string(),
            Value::Array(vec![
                Value::number(emissive.r as f64),
                Value::number(emissive.g as f64),
                Value::number(emissive.b as f64),
            ]),
        ));
    }
    // glTF's default is opaque, and a translucent material that does not say
    // so is drawn solid.
    if opacity < 1.0 {
        pairs.push(("alphaMode".to_string(), Value::string("BLEND")));
    } else if material.alpha_test() > 0.0 {
        pairs.push(("alphaMode".to_string(), Value::string("MASK")));
        pairs.push((
            "alphaCutoff".to_string(),
            Value::number(material.alpha_test() as f64),
        ));
    }
    if material.side() == 2 {
        pairs.push(("doubleSided".to_string(), Value::Bool(true)));
    }
    w.materials.push(Value::Object(pairs.into_iter().collect()));
    Some(w.materials.len() - 1)
}

/// A material the way three.js r165's `GLTFExporter.processMaterial` writes
/// it (with its `KHR_materials_unlit` extension for basic materials).
fn write_material_three_js(
    w: &mut Writer,
    material: &std::sync::Arc<Material>,
    options: &GltfExportOptions,
) -> usize {
    let info = options
        .material_info
        .get(&(std::sync::Arc::as_ptr(material) as usize))
        .cloned()
        .unwrap_or_else(|| GltfMaterialInfo::from_material(material));
    let mut pbr: Vec<(String, Value)> = Vec::new();
    let factor = [info.color[0], info.color[1], info.color[2], info.opacity];
    if factor != [1.0, 1.0, 1.0, 1.0] {
        pbr.push((
            "baseColorFactor".to_string(),
            Value::Array(factor.iter().map(|v| Value::number(*v as f64)).collect()),
        ));
    }
    let mut extensions: Vec<(String, Value)> = Vec::new();
    let (metallic, roughness) = match info.kind {
        GltfMaterialKind::Standard => (info.metalness, info.roughness),
        GltfMaterialKind::Basic => {
            extensions.push((
                "KHR_materials_unlit".to_string(),
                Value::Object(Default::default()),
            ));
            w.extensions_used.insert("KHR_materials_unlit".to_string());
            (0.0, 0.9)
        }
        GltfMaterialKind::Other => (0.5, 0.5),
    };
    pbr.push(("metallicFactor".to_string(), Value::number(metallic as f64)));
    pbr.push((
        "roughnessFactor".to_string(),
        Value::number(roughness as f64),
    ));
    if options.textures {
        if let Some(texture) = base_colour_map(material) {
            if let Some(index) = write_texture(w, &texture) {
                pbr.push((
                    "baseColorTexture".to_string(),
                    Value::object([("index", Value::number(index as f64))]),
                ));
            }
        }
    }
    let mut pairs: Vec<(String, Value)> = vec![(
        "pbrMetallicRoughness".to_string(),
        Value::Object(pbr.into_iter().collect()),
    )];
    let e = info.emissive;
    if e[0].max(e[1]).max(e[2]) > 0.0 {
        pairs.push((
            "emissiveFactor".to_string(),
            Value::Array(e.iter().map(|v| Value::number(*v as f64)).collect()),
        ));
    }
    if info.transparent {
        pairs.push(("alphaMode".to_string(), Value::string("BLEND")));
    } else if info.alpha_test > 0.0 {
        pairs.push(("alphaMode".to_string(), Value::string("MASK")));
        pairs.push((
            "alphaCutoff".to_string(),
            Value::number(info.alpha_test as f64),
        ));
    }
    if info.double_sided {
        pairs.push(("doubleSided".to_string(), Value::Bool(true)));
    }
    if !info.name.is_empty() {
        pairs.push(("name".to_string(), Value::string(info.name)));
    }
    if !extensions.is_empty() {
        pairs.push((
            "extensions".to_string(),
            Value::Object(extensions.into_iter().collect()),
        ));
    }
    w.materials.push(Value::Object(pairs.into_iter().collect()));
    w.materials.len() - 1
}

/// Standard base64 with padding, for `data:` URIs.
fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

fn base_colour_map(material: &Material) -> Option<std::sync::Arc<crate::textures::Texture>> {
    match material {
        Material::Standard(m) => m.map.clone(),
        Material::Physical(m) => m.map.clone(),
        Material::Basic(m) => m.map.clone(),
        Material::Sprite(m) => m.map.clone(),
        _ => None,
    }
}

/// An image, embedded as a PNG in the binary buffer.
fn write_texture(
    w: &mut Writer,
    texture: &std::sync::Arc<crate::textures::Texture>,
) -> Option<usize> {
    let key = std::sync::Arc::as_ptr(texture) as usize;
    if let Some(found) = w.interned_textures.get(&key) {
        return Some(*found);
    }
    let pixels = texture.to_rgba8()?;
    let png = crate::utils::png::encode_png(texture.width, texture.height, &pixels);
    let view = w.view(&png, None);
    w.images.push(Value::object([
        ("bufferView", Value::number(view as f64)),
        ("mimeType", Value::string("image/png")),
    ]));
    let image = w.images.len() - 1;
    // One sampler serves every texture: glTF's defaults are linear filtering
    // and repeat, which is what this crate's textures mostly are.
    if w.samplers.is_empty() {
        w.samplers.push(Value::object([
            ("magFilter", Value::number(9729.0)),
            ("minFilter", Value::number(9729.0)),
            ("wrapS", Value::number(10497.0)),
            ("wrapT", Value::number(10497.0)),
        ]));
    }
    w.textures.push(Value::object([
        ("source", Value::number(image as f64)),
        ("sampler", Value::number(0.0)),
    ]));
    let at = w.textures.len() - 1;
    w.interned_textures.insert(key, at);
    Some(at)
}

/// Wrap JSON and binary in the GLB container.
fn package(json: &str, bin: &[u8]) -> Vec<u8> {
    let mut json_bytes = json.as_bytes().to_vec();
    // The JSON chunk is padded with spaces and the binary chunk with zeroes —
    // the spec is specific about which, and a reader may checksum the tail.
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }
    let mut bin = bin.to_vec();
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let total = 12 + 8 + json_bytes.len() + if bin.is_empty() { 0 } else { 8 + bin.len() };
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // "JSON"
    out.extend_from_slice(&json_bytes);
    if !bin.is_empty() {
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x004E4942u32.to_le_bytes()); // "BIN\0"
        out.extend_from_slice(&bin);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BufferAttribute, Mesh, Object3D};
    use crate::loaders::GltfLoader;
    use crate::materials::StandardMaterial;
    use crate::math::{Color, Vector3};
    use std::sync::Arc;

    fn quad() -> BufferGeometry {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(
                vec![
                    -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0,
                ],
                3,
            ),
        );
        g.set_attribute("normal", BufferAttribute::new([0.0, 0.0, 1.0].repeat(4), 3));
        g.set_attribute(
            "uv",
            BufferAttribute::new(vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 2),
        );
        g.set_index(vec![0, 1, 2, 0, 2, 3]);
        g
    }

    fn scene() -> (ObjectArena, ObjectId) {
        let material = StandardMaterial {
            roughness: 0.4,
            metalness: 0.1,
            ..StandardMaterial::new(Color::new(0.9, 0.4, 0.2))
        };
        let mut arena = ObjectArena::new();
        let mut parent = Object3D::mesh(Mesh::new(quad(), material.clone().into()));
        parent.name = "Parent".into();
        let parent_id = arena.insert(parent);
        let mut child = Object3D::mesh(Mesh::new(quad(), material.into()));
        child.name = "Child".into();
        child.position = Vector3::new(2.5, 0.5, 0.0);
        child.scale = Vector3::new(0.5, 0.5, 0.5);
        let child_id = arena.insert(child);
        arena.add_child(parent_id, child_id);
        (arena, parent_id)
    }

    /// What is written reads back as what went in.
    #[test]
    fn a_scene_round_trips_through_glb() {
        let (arena, root) = scene();
        let bytes = scene_to_glb(&arena, &[root]);
        assert_eq!(&bytes[..4], b"glTF", "not a GLB");

        let back = GltfLoader::parse_glb(&bytes).expect("reads back");
        let names: Vec<String> = back
            .arena
            .nodes
            .iter()
            .map(|(_, o)| o.name.clone())
            .filter(|n| !n.is_empty())
            .collect();
        assert!(names.contains(&"Parent".to_string()), "{names:?}");
        assert!(names.contains(&"Child".to_string()), "{names:?}");

        let (_, child) = back
            .arena
            .nodes
            .iter()
            .find(|(_, o)| o.name == "Child")
            .expect("the child");
        assert!(
            (child.position.x - 2.5).abs() < 1e-5,
            "{:?}",
            child.position
        );
        assert!((child.scale.x - 0.5).abs() < 1e-5, "{:?}", child.scale);
    }

    /// The geometry arrives with every attribute it left with.
    #[test]
    fn attributes_and_indices_survive() {
        let (arena, root) = scene();
        let back = GltfLoader::parse_glb(&scene_to_glb(&arena, &[root])).expect("reads");
        let (_, object) = back
            .arena
            .nodes
            .iter()
            .find(|(_, o)| o.name == "Parent")
            .expect("the parent");
        let ObjectKind::Mesh(mesh) = &object.kind else {
            panic!("not a mesh");
        };
        let original = quad();
        for name in ["position", "normal", "uv"] {
            let a = original.get_attribute(name).unwrap();
            let b = mesh
                .geometry
                .get_attribute(name)
                .unwrap_or_else(|| panic!("{name} was lost"));
            assert_eq!(a.array.len(), b.array.len(), "{name} length");
            for (x, y) in a.array.iter().zip(&b.array) {
                assert!((x - y).abs() < 1e-6, "{name}: {x} vs {y}");
            }
        }
        assert_eq!(mesh.geometry.index, original.index, "indices");
    }

    /// A material's numbers come back, not just its presence.
    #[test]
    fn a_material_round_trips() {
        let (arena, root) = scene();
        let back = GltfLoader::parse_glb(&scene_to_glb(&arena, &[root])).expect("reads");
        let (_, object) = back
            .arena
            .nodes
            .iter()
            .find(|(_, o)| o.name == "Parent")
            .expect("the parent");
        let ObjectKind::Mesh(mesh) = &object.kind else {
            panic!("not a mesh")
        };
        let Material::Standard(m) = &*mesh.material else {
            panic!("{:?}", mesh.material)
        };
        assert!((m.color.r - 0.9).abs() < 1e-3, "{:?}", m.color);
        assert!((m.roughness - 0.4).abs() < 1e-3, "{}", m.roughness);
        assert!((m.metalness - 0.1).abs() < 1e-3, "{}", m.metalness);
    }

    /// A shared geometry is written once, and a shared texture likewise.
    ///
    /// A forest of a thousand identical trees should cost one mesh, not a
    /// thousand — and it is the `Arc` that says they are the same, so this
    /// checks sharing rather than equality.
    #[test]
    fn shared_geometry_and_textures_are_written_once() {
        let texture = Arc::new(crate::textures::Texture::new(
            2,
            2,
            crate::textures::TextureFormat::Rgba8UnormSrgb,
            vec![255u8; 16],
        ));
        let material: Material = StandardMaterial {
            map: Some(texture),
            ..StandardMaterial::new(Color::new(1.0, 1.0, 1.0))
        }
        .into();
        let geometry = Arc::new(quad());

        let mut arena = ObjectArena::new();
        let mut roots = Vec::new();
        for i in 0..4 {
            let mesh = Mesh::from_arc(geometry.clone(), Arc::new(material.clone()));
            let mut object = Object3D::mesh(mesh);
            object.name = format!("Copy{i}");
            roots.push(arena.insert(object));
        }
        let glb = scene_to_glb(&arena, &roots);
        // The JSON chunk, bounded by its own length — reading past it runs into
        // the binary chunk and stops being UTF-8.
        let json_len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        let json = std::str::from_utf8(&glb[20..20 + json_len]).expect("the JSON chunk");
        // Four nodes, but one mesh and one image between them.
        assert_eq!(json.matches("\"primitives\"").count(), 1, "one mesh");
        assert_eq!(json.matches("image/png").count(), 1, "one image");
    }

    /// A `.gltf` names its buffer, so the pair resolves.
    #[test]
    fn the_text_form_names_its_buffer() {
        let (arena, root) = scene();
        let (json, bin) = scene_to_gltf(&arena, &[root]);
        assert!(json.contains("\"uri\":\"scene.bin\""), "{json}");
        assert!(!bin.is_empty());
        // And the declared length is the real one.
        assert!(json.contains(&format!("\"byteLength\":{}", bin.len())));
    }

    /// Chunks are padded and the header length is the real one.
    ///
    /// A GLB whose total does not match its chunks loads in some readers and
    /// not others, which is the worst way for a writer to be wrong.
    #[test]
    fn the_glb_container_is_well_formed() {
        let (arena, root) = scene();
        let bytes = scene_to_glb(&arena, &[root]);
        let total = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        assert_eq!(total, bytes.len(), "declared length is the real one");
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            2,
            "version"
        );

        let mut at = 12;
        let mut kinds = Vec::new();
        while at < bytes.len() {
            let len = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
            let kind = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap());
            assert_eq!(len % 4, 0, "chunk {kind:#x} is not 4-aligned");
            kinds.push(kind);
            at += 8 + len;
        }
        assert_eq!(at, bytes.len(), "chunks exactly fill the file");
        assert_eq!(kinds, [0x4E4F534A, 0x004E4942], "JSON then BIN");
    }

    /// An integer field is written as one.
    ///
    /// glTF's schema types counts and indices as integers, and a validator
    /// rejects `"count": 4.0` — which is what a float serialiser produces
    /// unless it is asked not to.
    #[test]
    fn integers_are_not_written_as_floats() {
        let (arena, root) = scene();
        let (json, _) = scene_to_gltf(&arena, &[root]);
        assert!(
            !json.contains(".0,"),
            "a float slipped into an integer field"
        );
        assert!(json.contains("\"count\":4"), "{json}");
    }
}

#[cfg(test)]
mod groups_and_morphs {
    use super::*;
    use crate::core::{BufferAttribute, Mesh, Object3D};
    use crate::loaders::GltfLoader;
    use crate::materials::BasicMaterial;
    use crate::math::Color;
    use std::sync::Arc;

    fn quad() -> BufferGeometry {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.], 3),
        );
        g.set_index(vec![0, 1, 2, 0, 2, 3]);
        g
    }

    fn reload(arena: &ObjectArena, id: ObjectId) -> crate::loaders::GltfScene {
        GltfLoader::parse_glb(&scene_to_glb(arena, &[id])).expect("re-reads")
    }

    /// Absolute morph targets leave as displacements, with the influences as
    /// `weights` and the dictionary as `extras.targetNames`, and come back as
    /// three.js's loader would read them: relative, same shape.
    #[test]
    fn morph_targets_round_trip() {
        let mut g = quad();
        let raised: Vec<f32> = g
            .get_attribute("position")
            .unwrap()
            .array
            .iter()
            .enumerate()
            .map(|(i, v)| if i % 3 == 2 { v + 2.0 } else { *v })
            .collect();
        g.set_morph_attribute("position", vec![BufferAttribute::new(raised, 3)]);
        g.morph_target_names = vec!["lift".into()];
        let mut mesh = Mesh::new(g, BasicMaterial::new(Color::WHITE).into());
        mesh.morph_influences = vec![0.75];
        let mut arena = ObjectArena::new();
        let id = arena.insert(Object3D::mesh(mesh));
        let back = reload(&arena, id);
        let crate::core::ObjectKind::Mesh(m) = &back.arena.get(back.roots[0]).unwrap().kind else {
            panic!("a mesh");
        };
        assert!(m.geometry.morph_targets_relative);
        let t = &m.geometry.morph_attributes["position"][0].array;
        assert_eq!(t, &vec![0., 0., 2., 0., 0., 2., 0., 0., 2., 0., 0., 2.]);
        assert_eq!(m.morph_influences, vec![0.75]);
        assert_eq!(m.morph_target_dictionary.get("lift"), Some(&0));
    }

    /// A material array exports as one primitive per group over shared
    /// attributes — three.js `GLTFExporter` — and so re-reads (three.js's
    /// loader makes a mesh per primitive) as one child mesh per group, each
    /// with its group's triangles and material.
    #[test]
    fn groups_export_as_primitives() {
        let mut g = quad();
        g.add_group(0, 3, 1).add_group(3, 3, 0);
        let red = Arc::new(Material::from(BasicMaterial::new(Color::new(
            1.0, 0.0, 0.0,
        ))));
        let blue = Arc::new(Material::from(BasicMaterial::new(Color::new(
            0.0, 0.0, 1.0,
        ))));
        let mesh = Mesh::with_materials(Arc::new(g), vec![red, blue]);
        let mut arena = ObjectArena::new();
        let id = arena.insert(Object3D::mesh(mesh));
        let glb = scene_to_glb(&arena, &[id]);
        let back = GltfLoader::parse_glb(&glb).expect("re-reads");
        let node = back.arena.get(back.roots[0]).unwrap();
        assert_eq!(node.children.len(), 2, "one mesh per primitive");
        let mut seen = Vec::new();
        for c in &node.children {
            let crate::core::ObjectKind::Mesh(m) = &back.arena.get(*c).unwrap().kind else {
                panic!("a mesh");
            };
            seen.push((
                m.geometry.index.clone().unwrap(),
                m.material.color().b > 0.5,
            ));
        }
        assert_eq!(seen, vec![(vec![0, 1, 2], true), (vec![0, 2, 3], false)]);
    }
}

#[cfg(test)]
mod external {
    use super::*;
    use crate::core::{Mesh, Object3D};
    use crate::materials::StandardMaterial;
    use crate::math::{Color, Vector3};

    /// Write files for `gltf-transform validate` to judge.
    #[test]
    #[ignore = "writes files for external validation"]
    fn write_for_the_validator() {
        use crate::core::{BufferAttribute, BufferGeometry};
        use std::sync::Arc;

        let dir = std::env::var("GLTF_OUT").unwrap_or_else(|_| "/tmp".into());

        let mut geometry = BufferGeometry::new();
        geometry.set_attribute(
            "position",
            BufferAttribute::new(
                vec![
                    -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0,
                ],
                3,
            ),
        );
        geometry.set_attribute("normal", BufferAttribute::new([0.0, 0.0, 1.0].repeat(4), 3));
        geometry.set_attribute(
            "uv",
            BufferAttribute::new(vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 2),
        );
        geometry.set_index(vec![0, 1, 2, 0, 2, 3]);

        // A checker, so a viewer shows whether the UVs and the image agree.
        let mut pixels = Vec::new();
        for y in 0..16u32 {
            for x in 0..16u32 {
                let on = ((x / 4) + (y / 4)) % 2 == 0;
                pixels.extend_from_slice(if on {
                    &[240, 200, 40, 255]
                } else {
                    &[30, 40, 120, 255]
                });
            }
        }
        let texture = crate::textures::Texture::new(
            16,
            16,
            crate::textures::TextureFormat::Rgba8UnormSrgb,
            pixels,
        );
        let material = StandardMaterial {
            roughness: 0.4,
            metalness: 0.1,
            emissive: Color::new(0.05, 0.0, 0.1),
            map: Some(Arc::new(texture)),
            ..StandardMaterial::new(Color::new(0.9, 0.4, 0.2))
        };

        let mut arena = crate::core::ObjectArena::new();
        let mut parent = Object3D::mesh(Mesh::new(geometry.clone(), material.clone().into()));
        parent.name = "Parent".into();
        let parent_id = arena.insert(parent);

        let mut child = Object3D::mesh(Mesh::new(geometry, material.into()));
        child.name = "Child".into();
        child.position = Vector3::new(2.5, 0.5, 0.0);
        child.scale = Vector3::new(0.5, 0.5, 0.5);
        let child_id = arena.insert(child);
        arena.add_child(parent_id, child_id);

        std::fs::write(
            format!("{dir}/written.glb"),
            scene_to_glb(&arena, &[parent_id]),
        )
        .unwrap();
        let (json, bin) = scene_to_gltf(&arena, &[parent_id]);
        std::fs::write(format!("{dir}/written.gltf"), json).unwrap();
        std::fs::write(format!("{dir}/scene.bin"), bin).unwrap();
        println!("wrote written.glb / written.gltf + scene.bin to {dir}");
    }
}
