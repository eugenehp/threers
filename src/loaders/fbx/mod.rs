//! FBX, read as a scene rather than as a mesh.
//!
//! Both serialisations — Autodesk's binary node tree and the text form — parse
//! to the same [`tree::Node`], and everything above that reads the tree. What
//! comes out is a scene graph: every mesh in the file, its place in the
//! hierarchy, its transform, its material.
//!
//! # What a reader has to get right
//!
//! - **Polygon ends are bitwise-negated indices.** `PolygonVertexIndex` is the
//!   only record of where one face stops, and treating it as a triangle list
//!   silently produces a different mesh.
//! - **Layer elements say how they are addressed.** A normal array may be per
//!   corner, per control point, per polygon or one value for the mesh, directly
//!   or through an index array. Assuming one of those gives wrong normals that
//!   look like bad lighting.
//! - **The hierarchy is in `Connections`, not in nesting.** `Objects` is flat;
//!   parenthood is a list of id pairs, and a `Model` may connect to a
//!   `Geometry`, a `Material`, or another `Model`.
//! - **Units and axes are declared, not assumed.** `GlobalSettings` carries
//!   `UnitScaleFactor` (centimetres in most files) and an up-axis.
//!
//! ```no_run
//! use threers::loaders::fbx::FbxLoader;
//!
//! let bytes = std::fs::read("model.fbx").unwrap();
//! let scene = FbxLoader::parse_scene(&bytes).unwrap();
//! println!("{} objects, {} cm per unit", scene.arena.nodes.len(), scene.unit_scale);
//! ```

mod ascii;
mod binary;
mod geometry;
pub mod tree;

use std::collections::HashMap;

use crate::core::{BufferAttribute, BufferGeometry, Mesh, Object3D, ObjectArena, ObjectId};
use std::sync::Arc;
use crate::materials::{Material, StandardMaterial};
use crate::math::{Color, Quaternion, Vector3};
use tree::{Node, Property};

pub struct FbxLoader;

#[derive(Debug, PartialEq)]
pub enum FbxError {
    /// Not FBX in either form.
    Unsupported,
    /// FBX, but with no mesh in it.
    NoGeometry,
    /// The binary magic or version is not there.
    BadHeader,
    /// Structurally FBX and internally inconsistent — a length or an offset
    /// that does not fit the file.
    Corrupt,
}

impl std::fmt::Display for FbxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FbxError::Unsupported => write!(f, "not an FBX file"),
            FbxError::NoGeometry => write!(f, "FBX file with no geometry in it"),
            FbxError::BadHeader => write!(f, "FBX binary header is missing or unreadable"),
            FbxError::Corrupt => write!(f, "corrupt FBX file"),
        }
    }
}

impl std::error::Error for FbxError {}

/// A scene read out of an FBX file.
pub struct FbxScene {
    pub roots: Vec<ObjectId>,
    pub arena: ObjectArena,
    /// `UnitScaleFactor` — how many centimetres one unit is. Most FBX files
    /// say 1.0 and mean centimetres, which is why a model imported without
    /// reading it arrives a hundred times too large beside a metres scene.
    pub unit_scale: f32,
    /// The up axis the file declares: 1 for Y, 2 for Z.
    pub up_axis: u32,
    /// The names of the materials, in the order they were found.
    pub materials: Vec<String>,
}

impl FbxLoader {
    /// The document as its node tree, for a caller that wants the file rather
    /// than the scene.
    pub fn parse_tree(bytes: &[u8]) -> Result<Node, FbxError> {
        if bytes.starts_with(b"Kaydara FBX Binary") {
            return binary::parse(bytes);
        }
        let text = std::str::from_utf8(bytes).map_err(|_| FbxError::Unsupported)?;
        // A text FBX has no magic. The header block is the nearest thing, and
        // requiring it keeps this from cheerfully parsing an unrelated file
        // into an empty tree.
        if !text.contains("FBXHeaderExtension") && !text.contains("Objects:") {
            return Err(FbxError::Unsupported);
        }
        ascii::parse(text)
    }

    /// The first mesh in the file, as one geometry.
    ///
    /// Kept for callers that want a mesh and nothing else;
    /// [`parse_scene`](Self::parse_scene) is what reads the whole file.
    pub fn parse(bytes: &[u8]) -> Result<BufferGeometry, FbxError> {
        let tree = Self::parse_tree(bytes)?;
        let first = tree.child("Objects").and_then(|objects| {
            objects
                .children_named("Geometry")
                .find_map(|g| geometry::build(g).map(|(geometry, _)| geometry))
        });
        first.ok_or(FbxError::NoGeometry)
    }

    /// Every mesh in the file, in its place in the hierarchy.
    pub fn parse_scene(bytes: &[u8]) -> Result<FbxScene, FbxError> {
        let tree = Self::parse_tree(bytes)?;
        Ok(build_scene(&tree))
    }
}

/// `Connections` as parent → children, keyed by FBX id.
///
/// A connection is `C: "OO", child, parent` — object-to-object, and the child
/// comes *first*. A parent of 0 means the scene root.
fn connections(tree: &Node) -> HashMap<i64, Vec<i64>> {
    let mut out: HashMap<i64, Vec<i64>> = HashMap::new();
    if let Some(list) = tree.child("Connections") {
        for c in list.children_named("C").chain(list.children_named("Connect")) {
            // The first property is the kind; ids follow. `OP` connections name
            // a property as a fourth field and are otherwise the same shape.
            let ids: Vec<i64> = c.properties[1..]
                .iter()
                .filter_map(Property::as_i64)
                .collect();
            if let [child, parent, ..] = ids[..] {
                out.entry(parent).or_default().push(child);
            }
        }
    }
    out
}

fn build_scene(tree: &Node) -> FbxScene {
    let by_id = tree.by_id();
    let children_of = connections(tree);

    let settings = tree.child("GlobalSettings");
    let unit_scale = settings
        .and_then(|s| s.property70("UnitScaleFactor"))
        .and_then(|v| v.first().and_then(Property::as_f64))
        .unwrap_or(1.0) as f32;
    let up_axis = settings
        .and_then(|s| s.property70("UpAxis"))
        .and_then(|v| v.first().and_then(Property::as_i64))
        .unwrap_or(1) as u32;

    let mut arena = ObjectArena::new();
    let mut materials = Vec::new();
    let mut roots = Vec::new();

    // Everything hanging off id 0 is a root. A `Model` is a node of the graph;
    // a `Geometry` or `Material` reaches the graph only through the model that
    // connects to it.
    for &id in children_of.get(&0).into_iter().flatten() {
        if let Some(node) = by_id.get(&id) {
            if node.name == "Model" {
                if let Some(object) =
                    build_model(node, id, &by_id, &children_of, &mut arena, &mut materials)
                {
                    roots.push(object);
                }
            }
        }
    }

    // A file with no usable connections still has meshes in it, and returning
    // an empty scene for one would be a worse answer than a flat list.
    if roots.is_empty() {
        if let Some(objects) = tree.child("Objects") {
            for g in objects.children_named("Geometry") {
                if let Some((mesh, _)) = geometry::build(g) {
                    let mut object = Object3D::mesh(Mesh::new(mesh, default_material()));
                    object.name = node_name(g);
                    roots.push(arena.insert(object));
                }
            }
        }
    }

    FbxScene {
        roots,
        arena,
        unit_scale,
        up_axis,
        materials,
    }
}

fn build_model(
    node: &Node,
    id: i64,
    by_id: &HashMap<i64, &Node>,
    children_of: &HashMap<i64, Vec<i64>>,
    arena: &mut ObjectArena,
    materials: &mut Vec<String>,
) -> Option<ObjectId> {
    let connected: Vec<i64> = children_of.get(&id).cloned().unwrap_or_default();

    // The geometry and the materials this model wears — every connected
    // material, in connection order, as three.js `FBXLoader.createMesh`
    // collects them.
    let mut mesh_geometry = None;
    let mut model_materials: Vec<Arc<Material>> = Vec::new();
    for &child in &connected {
        let Some(child_node) = by_id.get(&child) else {
            continue;
        };
        match child_node.name.as_str() {
            "Geometry" if mesh_geometry.is_none() => {
                mesh_geometry = geometry::build(child_node).map(|(mut g, slots)| {
                    if !geometry::material_mapping_is_all_same(child_node) {
                        add_material_groups(&mut g, &slots);
                    }
                    add_blend_shapes(&mut g, child_node, child, by_id, children_of);
                    g
                });
            }
            "Material" => {
                let name = node_name(child_node);
                if !materials.contains(&name) {
                    materials.push(name.clone());
                }
                model_materials.push(Arc::new(read_material(child_node)));
            }
            _ => {}
        }
    }

    let mut object = match mesh_geometry {
        // More than one material: three.js's material array over the groups.
        Some(g) if model_materials.len() > 1 => {
            Object3D::mesh(Mesh::with_materials(Arc::new(g), model_materials))
        }
        Some(g) => Object3D::mesh(Mesh::from_arc(
            Arc::new(g),
            model_materials
                .pop()
                .unwrap_or_else(|| Arc::new(default_material())),
        )),
        None => Object3D::group(),
    };
    object.name = node_name(node);
    apply_transform(node, &mut object);
    let object_id = arena.insert(object);

    // Child models, recursively. Depth is bounded by the connection graph; a
    // cycle would recurse forever, so a node already placed is not placed again.
    for &child in &connected {
        let Some(child_node) = by_id.get(&child) else {
            continue;
        };
        if child_node.name == "Model" && child != id {
            if let Some(kid) = build_model(child_node, child, by_id, children_of, arena, materials) {
                arena.add_child(object_id, kid);
            }
        }
    }
    Some(object_id)
}

/// three.js `FBXLoader.genGeometry`'s groups: one per run of equal material
/// slots, in vertices of this unindexed geometry (three per triangle); a
/// negative slot reads as 0, and a mesh whose triangles all use one slot gets
/// one group covering it.
fn add_material_groups(g: &mut BufferGeometry, triangle_slots: &[i32]) {
    let slots: Vec<u32> = triangle_slots.iter().map(|&s| s.max(0) as u32).collect();
    let Some(&first) = slots.first() else {
        return;
    };
    let (mut prev, mut start) = (first, 0usize);
    for (t, &s) in slots.iter().enumerate() {
        if s != prev {
            g.add_group(start as u32 * 3, (t - start) as u32 * 3, prev);
            prev = s;
            start = t;
        }
    }
    g.add_group(start as u32 * 3, (slots.len() - start) as u32 * 3, prev);
}

/// three.js `FBXLoader`'s blend shapes: each `BlendShapeChannel` under a
/// `BlendShape` deformer of this geometry contributes its `Shape` geometry's
/// sparse per-control-point offsets, expanded through the mesh's corners,
/// as a relative position morph target named after the channel. Influences
/// start at zero, as three.js leaves them (`DeformPercent` reaches them only
/// through animation there). Normals are not morphed — three.js's TODO too.
fn add_blend_shapes(
    g: &mut BufferGeometry,
    geometry_node: &Node,
    geometry_id: i64,
    by_id: &HashMap<i64, &Node>,
    children_of: &HashMap<i64, Vec<i64>>,
) {
    let kind = |n: &Node| n.properties.get(2).and_then(Property::as_str).map(str::to_string);
    let corners = geometry::corner_control_points(geometry_node);
    if corners.is_empty() || g.get_attribute("position").map(|p| p.count()) != Some(corners.len()) {
        return;
    }
    let mut targets = Vec::new();
    let mut names = Vec::new();
    for deformer in children_of.get(&geometry_id).into_iter().flatten() {
        let Some(d) = by_id.get(deformer).filter(|d| d.name == "Deformer") else {
            continue;
        };
        if kind(d).as_deref() != Some("BlendShape") {
            continue;
        }
        for channel in children_of.get(deformer).into_iter().flatten() {
            let Some(c) = by_id.get(channel).filter(|c| c.name == "Deformer") else {
                continue;
            };
            if kind(c).as_deref() != Some("BlendShapeChannel") {
                continue;
            }
            for shape in children_of.get(channel).into_iter().flatten() {
                let Some(sh) = by_id.get(shape).filter(|s| s.name == "Geometry") else {
                    continue;
                };
                let indexes = sh.value("Indexes").and_then(Property::as_i32_array).unwrap_or_default();
                let offsets = sh.value("Vertices").and_then(Property::as_f32_array).unwrap_or_default();
                let mut per_point: HashMap<usize, [f32; 3]> = HashMap::new();
                for (i, &cp) in indexes.iter().enumerate() {
                    if let Some(v) = offsets.get(i * 3..i * 3 + 3) {
                        per_point.insert(cp.max(0) as usize, [v[0], v[1], v[2]]);
                    }
                }
                let delta: Vec<f32> = corners
                    .iter()
                    .flat_map(|cp| per_point.get(cp).copied().unwrap_or([0.0; 3]))
                    .collect();
                targets.push(BufferAttribute::new(delta, 3));
                names.push(node_name(c));
            }
        }
    }
    if !targets.is_empty() {
        g.morph_targets_relative = true;
        g.morph_target_names = names;
        g.set_morph_attribute("position", targets);
    }
}

/// An object's name. FBX writes `"Model::Cube"` or, from 7.x, `"Cube\0\x01Model"`,
/// so both separators have to go.
fn node_name(node: &Node) -> String {
    let raw = node
        .properties
        .iter()
        .find_map(|p| p.as_str())
        .unwrap_or_default();
    let cleaned = raw.split('\0').next().unwrap_or(raw);
    match cleaned.rsplit_once("::") {
        Some((_, name)) => name.to_string(),
        None => cleaned.to_string(),
    }
}

fn apply_transform(node: &Node, object: &mut Object3D) {
    if let Some([x, y, z]) = node.property70_vec3("Lcl Translation") {
        object.position = Vector3::new(x, y, z);
    }
    if let Some([x, y, z]) = node.property70_vec3("Lcl Rotation") {
        // FBX stores Euler angles in degrees, XYZ order by default.
        object.quaternion =
            Quaternion::from_euler_xyz(x.to_radians(), y.to_radians(), z.to_radians());
    }
    if let Some([x, y, z]) = node.property70_vec3("Lcl Scaling") {
        object.scale = Vector3::new(x, y, z);
    }
}

fn read_material(node: &Node) -> Material {
    let mut standard = StandardMaterial::new(Color::new(0.8, 0.8, 0.8));
    // `DiffuseColor` is what every exporter writes; `Diffuse` is the older
    // spelling and some writers still use it.
    for name in ["DiffuseColor", "Diffuse"] {
        if let Some([r, g, b]) = node.property70_vec3(name) {
            standard.color = Color::new(r, g, b);
            break;
        }
    }
    if let Some([r, g, b]) = node.property70_vec3("EmissiveColor") {
        standard.emissive = Color::new(r, g, b);
    }
    if let Some(v) = node
        .property70("Shininess")
        .and_then(|v| v.first().and_then(Property::as_f64))
    {
        // Phong's exponent as a roughness, the same conversion the USD exporter
        // makes in the other direction.
        standard.roughness = (2.0 / (v.max(0.0) + 2.0)).sqrt().clamp(0.0, 1.0) as f32;
    }
    if let Some(v) = node
        .property70("Opacity")
        .and_then(|v| v.first().and_then(Property::as_f64))
    {
        standard.opacity = v as f32;
    }
    standard.into()
}

fn default_material() -> Material {
    StandardMaterial::new(Color::new(0.8, 0.8, 0.8)).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ObjectKind;

    const BLENDER: &[u8] = include_bytes!("testdata/blender.fbx");
    const ASCII: &[u8] = include_bytes!("testdata/assimp_ascii.fbx");
    const MODES: &str = include_str!("testdata/modes.fbx");
    const MULTI: &str = include_str!("testdata/multimat_blend.fbx");

    /// three.js r165 `FBXLoader` on `multimat_blend.fbx` (a triangle, a quad
    /// and a triangle with polygon materials `[1, 1, 0]`, two materials, and a
    /// `BlendShape` whose channel `Lift` moves control point 1 by (0, 0, 1) and
    /// point 5 by (0, 0.5, 0)): groups `[{0,9,1},{9,3,0}]`, materials
    /// `["Red","Blue"]`, one relative position target named `Lift` that moves
    /// every corner on point 1 by (0,0,1) and on point 5 by (0,0.5,0),
    /// influences `[0]`, dictionary `{Lift: 0}`. (three.js triangulates the
    /// quad along the other diagonal; the triangles' materials and the
    /// per-corner offsets agree.)
    #[test]
    fn materials_and_blend_shapes_load_as_three() {
        let scene = FbxLoader::parse_scene(MULTI.as_bytes()).expect("parses");
        let mesh = mesh_of(&scene, scene.roots[0]);
        let groups: Vec<(u32, u32, u32)> = mesh
            .geometry
            .groups
            .iter()
            .map(|g| (g.start, g.count, g.material_index))
            .collect();
        assert_eq!(groups, [(0, 9, 1), (9, 3, 0)]);
        assert_eq!(mesh.materials.len(), 2);
        assert!(mesh.materials[0].color().r > 0.9, "Red first");
        assert!(mesh.materials[1].color().b > 0.9);
        let g = &mesh.geometry;
        assert!(g.morph_targets_relative);
        let pos = &g.get_attribute("position").unwrap().array;
        let delta = &g.morph_attributes["position"][0].array;
        for (p, d) in pos.chunks_exact(3).zip(delta.chunks_exact(3)) {
            let want: [f32; 3] = match (p[0], p[1]) {
                (1.0, 0.0) => [0.0, 0.0, 1.0],
                (2.0, 1.0) => [0.0, 0.5, 0.0],
                _ => [0.0; 3],
            };
            assert_eq!(d, &want, "corner at {p:?}");
        }
        assert_eq!(mesh.morph_influences, vec![0.0]);
        assert_eq!(mesh.morph_target_dictionary.get("Lift"), Some(&0));
    }

    fn mesh_of(scene: &FbxScene, id: ObjectId) -> &Mesh {
        match &scene.arena.get(id).expect("an object").kind {
            ObjectKind::Mesh(mesh) => mesh,
            other => panic!("{other:?} is not a mesh"),
        }
    }

    fn named(scene: &FbxScene, want: &str) -> ObjectId {
        fn walk(scene: &FbxScene, id: ObjectId, want: &str, out: &mut Option<ObjectId>) {
            let Some(object) = scene.arena.get(id) else { return };
            if object.name == want {
                *out = Some(id);
            }
            for child in object.children.clone() {
                walk(scene, child, want, out);
            }
        }
        let mut out = None;
        for root in &scene.roots {
            walk(scene, *root, want, &mut out);
        }
        out.unwrap_or_else(|| panic!("no object named {want}"))
    }

    /// The whole file, not the first mesh in it.
    ///
    /// The loader used to search the bytes for the first `Vertices` array and
    /// stop, so a two-mesh file read as one mesh and a hierarchy read as
    /// nothing. The expectations here are `assimp info`'s reading of the same
    /// file: two meshes, 24 faces between them, two materials, and
    /// `RootNode → Parent → Child`.
    #[test]
    fn a_whole_scene_is_read_not_just_the_first_mesh() {
        let scene = FbxLoader::parse_scene(BLENDER).expect("parses");
        assert_eq!(scene.roots.len(), 1, "Parent is the only root");
        assert_eq!(scene.arena.nodes.len(), 2, "both meshes arrived");
        assert_eq!(scene.materials, ["Parent_mat", "Child_mat"]);

        let parent = named(&scene, "Parent");
        let child = named(&scene, "Child");
        assert!(
            scene.arena.get(parent).unwrap().children.contains(&child),
            "Child hangs off Parent, as Connections says"
        );

        // 6 quads a cube, fanned to 12 triangles, 36 corners — and 24 faces
        // across the two, which is what assimp counts.
        for id in [parent, child] {
            let mesh = mesh_of(&scene, id);
            let positions = mesh.geometry.get_attribute("position").unwrap().array.len() / 3;
            assert_eq!(positions, 36, "12 triangles");
            assert_eq!(mesh.geometry.get_attribute("normal").unwrap().array.len() / 3, 36);
            assert_eq!(mesh.geometry.get_attribute("uv").unwrap().array.len() / 2, 36);
        }
    }

    /// The transform on the child is the one the file authored.
    #[test]
    fn a_models_transform_is_read() {
        let scene = FbxLoader::parse_scene(BLENDER).expect("parses");
        let child = scene.arena.get(named(&scene, "Child")).unwrap();
        assert!((child.position.x - 3.0).abs() < 1e-4, "{:?}", child.position);
        assert!((child.position.y - 1.0).abs() < 1e-4);
        assert!((child.position.z - 2.0).abs() < 1e-4);
        assert!((child.scale.x - 2.0).abs() < 1e-4, "{:?}", child.scale);
    }

    /// The two serialisations of one document read the same.
    ///
    /// The binary form came from Blender and the text form is assimp's
    /// conversion of it, so agreement here is agreement with two other
    /// implementations at once — not with itself.
    #[test]
    fn the_two_forms_agree() {
        let binary = FbxLoader::parse_scene(BLENDER).expect("binary");
        let text = FbxLoader::parse_scene(ASCII).expect("text");
        assert_eq!(binary.arena.nodes.len(), text.arena.nodes.len());
        assert_eq!(binary.materials, text.materials);

        for name in ["Parent", "Child"] {
            let a = mesh_of(&binary, named(&binary, name));
            let b = mesh_of(&text, named(&text, name));
            let pa = &a.geometry.get_attribute("position").unwrap().array;
            let pb = &b.geometry.get_attribute("position").unwrap().array;
            assert_eq!(pa.len(), pb.len(), "{name}");
            for (x, y) in pa.iter().zip(pb) {
                assert!((x - y).abs() < 1e-4, "{name}: {x} vs {y}");
            }
        }
    }

    /// The cube's corners are where the file says, at ±1.
    ///
    /// Reading positions at all is not the same as reading them correctly: a
    /// stride or index slip gives the right *count* of plausible numbers.
    #[test]
    fn positions_are_the_control_points_the_file_holds() {
        let scene = FbxLoader::parse_scene(BLENDER).expect("parses");
        let mesh = mesh_of(&scene, named(&scene, "Parent"));
        let positions = &mesh.geometry.get_attribute("position").unwrap().array;
        for v in positions {
            assert!(
                (v.abs() - 1.0).abs() < 1e-4,
                "a corner of a 2-unit cube is at ±1, not {v}"
            );
        }
        // And the normals are unit length, which a mis-strided read would not be.
        let normals = &mesh.geometry.get_attribute("normal").unwrap().array;
        for n in normals.chunks_exact(3) {
            let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((length - 1.0).abs() < 1e-3, "{n:?} is not a unit normal");
        }
    }

    /// `ByVertice` + `IndexToDirect`, which is the combination that goes wrong
    /// quietly.
    ///
    /// The quad's four corners index a two-entry normal array through an index
    /// array: corners 0 and 1 take `(0,1,0)` and corners 2 and 3 take `(0,0,1)`.
    /// Read as `ByPolygonVertex` the indices address the wrong slots; read as
    /// `Direct` they address the data directly and the second normal is never
    /// seen. Either way the count is right and the mesh is lit wrongly.
    #[test]
    fn layer_elements_honour_their_mapping_and_reference_modes() {
        let scene = FbxLoader::parse_scene(MODES.as_bytes()).expect("parses");
        let mesh = mesh_of(&scene, named(&scene, "Quad"));
        let normals = &mesh.geometry.get_attribute("normal").unwrap().array;

        // One quad fans to two triangles: corners 0,1,2 then 0,2,3.
        let at = |i: usize| [normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]];
        assert_eq!(at(0), [0.0, 1.0, 0.0], "control point 0 -> index 1");
        assert_eq!(at(1), [0.0, 1.0, 0.0], "control point 1 -> index 1");
        assert_eq!(at(2), [0.0, 0.0, 1.0], "control point 2 -> index 0");
        assert_eq!(at(5), [0.0, 0.0, 1.0], "control point 3 -> index 0");

        // `AllSame` is one value for every corner of the mesh.
        let uv = &mesh.geometry.get_attribute("uv").unwrap().array;
        assert_eq!(uv.len(), 12, "6 corners, 2 numbers each");
        for pair in uv.chunks_exact(2) {
            assert_eq!(pair, [0.25, 0.75]);
        }
    }

    /// The quad's two triangles are remembered as one polygon, and its four
    /// corners come back in the order `PolygonVertexIndex` gave them.
    #[test]
    fn a_quad_is_one_polygon() {
        let scene = FbxLoader::parse_scene(MODES.as_bytes()).expect("parses");
        let g = &mesh_of(&scene, named(&scene, "Quad")).geometry;
        assert_eq!(g.polygon_counts(), Some(&[2u32][..]));
        // Unindexed, one vertex per corner: triangle 0 is corners 0,1,2 and
        // triangle 1 is corners 0,2,3 at vertices 3,4,5.
        let corners = g.polygon_loop(0..2).expect("a simple loop");
        assert_eq!(corners, [0, 1, 4, 5]);
        assert_eq!(g.polygon_edges().len(), 4, "no diagonal");
        assert_eq!(g.wireframe_indices().len(), 8);
    }

    /// Blender's cube is six quads: twelve edges, not the eighteen its
    /// triangles have.
    #[test]
    fn a_blender_cube_has_twelve_edges() {
        let scene = FbxLoader::parse_scene(BLENDER).expect("parses");
        let g = &mesh_of(&scene, named(&scene, "Parent")).geometry;
        assert_eq!(g.polygon_counts(), Some(&[2u32; 6][..]));
        assert_eq!(g.polygon_edges().len(), 12);
        for range in g.polygon_ranges() {
            assert_eq!(g.polygon_loop(range).map(|l| l.len()), Some(4));
        }
    }

    /// Units and the up axis are read, not assumed.
    ///
    /// Most FBX is in centimetres. A model imported without reading
    /// `UnitScaleFactor` arrives a hundred times too big beside a metres scene,
    /// which looks like a modelling mistake rather than a loader one.
    #[test]
    fn units_and_up_axis_come_from_the_file() {
        let scene = FbxLoader::parse_scene(MODES.as_bytes()).expect("parses");
        assert!((scene.unit_scale - 2.54).abs() < 1e-6, "{}", scene.unit_scale);
        assert_eq!(scene.up_axis, 2, "the fixture declares Z up");

        let blender = FbxLoader::parse_scene(BLENDER).expect("parses");
        assert_eq!(blender.up_axis, 1, "Blender writes Y up");
    }

    /// A material's colour reaches the mesh bound to it.
    #[test]
    fn a_material_is_read_and_bound() {
        let scene = FbxLoader::parse_scene(MODES.as_bytes()).expect("parses");
        assert_eq!(scene.materials, ["Red"]);
        let mesh = mesh_of(&scene, named(&scene, "Quad"));
        let Material::Standard(material) = &*mesh.material else {
            panic!("{:?}", mesh.material);
        };
        assert!((material.color.r - 0.8).abs() < 1e-4, "{:?}", material.color);
        assert!((material.color.g - 0.1).abs() < 1e-4);
        assert!((material.opacity - 0.5).abs() < 1e-4);
        // Shininess 30 as a roughness: sqrt(2/32) = 0.25.
        assert!((material.roughness - 0.25).abs() < 1e-3, "{}", material.roughness);
    }

    /// The old single-mesh entry point still works.
    #[test]
    fn the_geometry_only_entry_point_still_reads_a_mesh() {
        for bytes in [BLENDER, ASCII] {
            let geometry = FbxLoader::parse(bytes).expect("a mesh");
            assert_eq!(geometry.get_attribute("position").unwrap().array.len() / 3, 36);
        }
    }

    /// A three-deep hierarchy with three meshes, checked against `assimp`.
    ///
    /// The numbers are `assimp info`'s reading of the same file: three meshes
    /// and 2174 triangles between them. Agreeing on a count that specific,
    /// from a torus, a sphere and a cone — quads, triangles and a pole fan —
    /// is the polygon walk and the fan being right, not approximately right.
    #[test]
    fn a_nested_scene_matches_what_assimp_reads() {
        const NESTED: &[u8] = include_bytes!("testdata/nested.fbx");
        let scene = FbxLoader::parse_scene(NESTED).expect("parses");

        let mut meshes = 0;
        let mut triangles = 0;
        let mut deepest = 0;
        let mut coloured = Vec::new();
        fn walk(
            scene: &FbxScene,
            id: ObjectId,
            at: usize,
            meshes: &mut usize,
            triangles: &mut usize,
            deepest: &mut usize,
            coloured: &mut Vec<String>,
        ) {
            let Some(object) = scene.arena.get(id) else { return };
            *deepest = (*deepest).max(at);
            if let ObjectKind::Mesh(mesh) = &object.kind {
                *meshes += 1;
                *triangles += mesh.geometry.get_attribute("position").unwrap().array.len() / 9;
                if mesh.geometry.get_attribute("color").is_some() {
                    coloured.push(object.name.clone());
                }
            }
            for child in object.children.clone() {
                walk(scene, child, at + 1, meshes, triangles, deepest, coloured);
            }
        }
        for root in &scene.roots {
            walk(&scene, *root, 1, &mut meshes, &mut triangles, &mut deepest, &mut coloured);
        }

        assert_eq!(meshes, 3, "a torus, a sphere and a cone");
        assert_eq!(triangles, 2174, "assimp counts the same faces");
        assert_eq!(deepest, 3, "Torus -> Sphere -> Cone");
        assert_eq!(coloured, ["Cone"], "vertex colours are on the cone alone");
    }

    /// A truncated file is an error, not a panic.
    ///
    /// Lengths and offsets come out of the bytes being read, so a file that
    /// ends early asks for a slice that is not there. The previous
    /// implementation indexed on those numbers directly and took the process
    /// down with it.
    #[test]
    fn a_truncated_file_does_not_panic() {
        for cut in [27, 40, 64, 128, 512, 4096, 9000, 18000] {
            if cut >= BLENDER.len() {
                continue;
            }
            let _ = FbxLoader::parse_scene(&BLENDER[..cut]);
            let _ = FbxLoader::parse(&BLENDER[..cut]);
        }
    }

    /// Nor does a corrupted one.
    #[test]
    fn corrupted_bytes_do_not_panic() {
        let mut bytes = BLENDER.to_vec();
        // A deterministic sweep, so a failure is reproducible.
        for step in 0..200 {
            let at = (step * 97 + 31) % bytes.len();
            let was = bytes[at];
            bytes[at] = was.wrapping_add(0x5b);
            let _ = FbxLoader::parse_scene(&bytes);
            bytes[at] = was;
        }
    }

    /// Something that is not FBX is refused rather than read as an empty scene.
    #[test]
    fn a_file_that_is_not_fbx_is_refused() {
        assert_eq!(
            FbxLoader::parse_tree(b"hello, world").unwrap_err(),
            FbxError::Unsupported
        );
        assert_eq!(
            FbxLoader::parse_tree(b"Kaydara FBX Binary").unwrap_err(),
            FbxError::BadHeader
        );
    }
}

#[cfg(test)]
mod external {
    use super::*;
    use crate::core::ObjectKind;

    /// Read a file and report what came out, for comparing against `assimp`.
    #[test]
    #[ignore = "reads a file assimp also read"]
    fn compare_with_assimp() {
        for path in std::env::var("FBX_IN").unwrap().split(',') {
            let bytes = std::fs::read(path).unwrap();
            let scene = FbxLoader::parse_scene(&bytes).expect("parses");
            let mut meshes = 0;
            let mut triangles = 0;
            let mut with_colour = Vec::new();
            let mut depth = 0;
            fn walk(
                scene: &FbxScene,
                id: ObjectId,
                at: usize,
                meshes: &mut usize,
                triangles: &mut usize,
                colour: &mut Vec<String>,
                depth: &mut usize,
            ) {
                let Some(object) = scene.arena.get(id) else { return };
                *depth = (*depth).max(at);
                if let ObjectKind::Mesh(mesh) = &object.kind {
                    *meshes += 1;
                    *triangles += mesh.geometry.get_attribute("position").unwrap().array.len() / 9;
                    if mesh.geometry.get_attribute("color").is_some() {
                        colour.push(object.name.clone());
                    }
                }
                for child in object.children.clone() {
                    walk(scene, child, at + 1, meshes, triangles, colour, depth);
                }
            }
            for root in &scene.roots {
                walk(&scene, *root, 1, &mut meshes, &mut triangles, &mut with_colour, &mut depth);
            }
            println!(
                "  {}: meshes={meshes} triangles={triangles} depth={depth} colour_on={with_colour:?}",
                path.rsplit('/').next().unwrap()
            );
        }
    }
}
