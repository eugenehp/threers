//! `UsdShade` — materials as the shader graphs they actually are.
//!
//! A USD material is not a struct of values. It is a graph: the material's
//! `outputs:surface` is *connected* to a shader, that shader's
//! `inputs:diffuseColor` may be connected to a texture reader, and that
//! reader's `inputs:st` may be connected to a primvar reader. Reading only the
//! values that happen to sit on the surface shader gets the constants and
//! silently misses every texture in the asset — which, for anything authored in
//! a DCC, is most of what the material is.
//!
//! # Binding is inherited
//!
//! `material:binding` applies to the prim it is on *and everything beneath it*.
//! A mesh with no binding of its own is not unbound; it wears whatever the
//! nearest ancestor binds. Missing that turns a bound asset into a grey one.
//!
//! # Textures are reported, not loaded
//!
//! Decoding an image needs an image decoder and access to the files beside the
//! layer, neither of which belongs in a scene-description reader. So a texture
//! comes back as a [`TextureRequest`] — which file, into which slot, with which
//! wrap modes and scale — and the caller loads it.

use crate::materials::{Material, PhysicalMaterial, StandardMaterial};
use crate::math::Color;

use super::parse::{UsdLayer, UsdPrim};
use super::value::UsdValue;

/// Which slot of a material a texture feeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureSlot {
    BaseColor,
    Normal,
    Roughness,
    Metalness,
    Emissive,
    Occlusion,
    Opacity,
    Displacement,
}

/// A texture the material wants, for the caller to load.
#[derive(Debug, Clone)]
pub struct TextureRequest {
    pub slot: TextureSlot,
    /// The asset path as authored, relative to the layer that authored it.
    pub file: String,
    /// The primvar the texture is sampled by, usually `st`.
    pub uv: String,
    /// `repeat`, `clamp`, `mirror` or `black`, as USD spells them.
    pub wrap_s: String,
    pub wrap_t: String,
    /// Applied as `sampled * scale + bias`. Not decoration: an 8-bit normal
    /// map is *required* to carry `(2,2,2,1)` and `(-1,-1,-1,0)` so that
    /// `[0,1]` becomes `[-1,1]`, and ignoring them leaves every normal pointing
    /// into the surface.
    pub scale: [f32; 4],
    pub bias: [f32; 4],
    /// `raw` for data, `sRGB` for colour. A normal map read as sRGB is wrong
    /// in a way that looks like bad lighting rather than a bug.
    pub color_space: String,
}

/// A material resolved from its graph.
#[derive(Debug, Clone)]
pub struct UsdMaterial {
    pub material: Material,
    pub textures: Vec<TextureRequest>,
}

/// The material bound to a prim, looking up the namespace if it has none.
///
/// The *closest* binding wins: a mesh's own beats its parent's, which beats its
/// grandparent's.
pub fn bound_material_path(layer: &UsdLayer, prim_path: &str) -> Option<String> {
    let mut path = prim_path.to_string();
    loop {
        if let Some(found) = layer
            .prim_at(&path)
            .and_then(|p| p.value("material:binding"))
            .and_then(|v| v.as_str())
        {
            return Some(found.to_string());
        }
        match path.rfind('/') {
            Some(0) | None => return None,
            Some(at) => path.truncate(at),
        }
    }
}

/// Split `/Mat/Shader.outputs:surface` into the prim and the output.
fn split_connection(path: &str) -> (&str, &str) {
    match path.rfind('.') {
        Some(at) => (&path[..at], &path[at + 1..]),
        None => (path, ""),
    }
}

/// Follow a connection to the shader that actually produces the value.
///
/// A `NodeGraph` is a pass-through: its own output is connected to something
/// inside it, so following one output may mean following several.
fn follow<'a>(layer: &'a UsdLayer, path: &str, depth: usize) -> Option<(&'a UsdPrim, String)> {
    if depth > 16 {
        return None;
    }
    let (prim_path, output) = split_connection(path);
    let prim = layer.prim_at(prim_path)?;

    // A node graph forwards; a shader answers.
    if prim.type_name == "NodeGraph" || prim.value("info:id").is_none() {
        if let Some(next) = prim.value(output).and_then(|v| v.as_str()) {
            if next != path {
                return follow(layer, next, depth + 1);
            }
        }
    }
    Some((prim, output.to_string()))
}

/// The surface shader of a material.
pub fn surface_shader<'a>(material: &'a UsdPrim, layer: &'a UsdLayer) -> Option<&'a UsdPrim> {
    // The connection is the answer when there is one.
    if let Some(path) = material.value("outputs:surface").and_then(|v| v.as_str()) {
        if let Some((shader, _)) = follow(layer, path, 0) {
            return Some(shader);
        }
    }
    // Failing that, the child that says it is one.
    fn search(prim: &UsdPrim) -> Option<&UsdPrim> {
        for child in &prim.children {
            if child.value("info:id").and_then(|v| v.as_str()) == Some("UsdPreviewSurface") {
                return Some(child);
            }
            if let Some(found) = search(child) {
                return Some(found);
            }
        }
        None
    }
    search(material).or_else(|| material.children.first())
}

/// What an input resolves to.
enum Input<'a> {
    Value(&'a UsdValue),
    Texture(TextureRequest),
}

/// Resolve one of a shader's inputs, following a connection where there is one.
fn input<'a>(
    shader: &'a UsdPrim,
    name: &str,
    slot: TextureSlot,
    layer: &'a UsdLayer,
) -> Option<Input<'a>> {
    let property = shader.property(&format!("inputs:{name}"))?;

    // A connected input names where its value comes from.
    if let UsdValue::Path(path) = &property.value {
        let (source, _) = follow(layer, path, 0)?;
        if source.value("info:id").and_then(|v| v.as_str()) == Some("UsdUVTexture") {
            return Some(Input::Texture(texture_request(source, slot, layer)));
        }
        // Connected to something that is not a texture — a constant reader, a
        // node this crate has no shading model for — so there is no value to
        // take rather than a wrong one to guess.
        return None;
    }
    Some(Input::Value(&property.value))
}

/// A `UsdUVTexture` as a request to load one.
fn texture_request(texture: &UsdPrim, slot: TextureSlot, layer: &UsdLayer) -> TextureRequest {
    let text = |name: &str, fallback: &str| -> String {
        texture
            .value(&format!("inputs:{name}"))
            .and_then(|v| v.as_str())
            .unwrap_or(fallback)
            .to_string()
    };
    let four = |name: &str, fallback: [f32; 4]| -> [f32; 4] {
        let Some(value) = texture.value(&format!("inputs:{name}")) else {
            return fallback;
        };
        let flat = value.flat_f32();
        if flat.len() >= 4 {
            [flat[0], flat[1], flat[2], flat[3]]
        } else {
            fallback
        }
    };

    // Which primvar drives it: `inputs:st` connects to a reader whose
    // `inputs:varname` is the name. With nothing connected it is `st`, which
    // is what every asset uses.
    let uv = texture
        .property("inputs:st")
        .and_then(|p| match &p.value {
            UsdValue::Path(path) => follow(layer, path, 0),
            _ => None,
        })
        .and_then(|(reader, _)| reader.value("inputs:varname").and_then(|v| v.as_str()))
        .unwrap_or("st")
        .to_string();

    TextureRequest {
        slot,
        file: text("file", ""),
        uv,
        wrap_s: text("wrapS", "repeat"),
        wrap_t: text("wrapT", "repeat"),
        scale: four("scale", [1.0; 4]),
        bias: four("bias", [0.0; 4]),
        color_space: text("sourceColorSpace", "auto"),
    }
}

/// Resolve a material prim into something a renderer can use.
pub fn resolve(material: &UsdPrim, layer: &UsdLayer) -> UsdMaterial {
    let mut standard = StandardMaterial::new(Color::new(0.8, 0.8, 0.8));
    let mut textures = Vec::new();

    let Some(shader) = surface_shader(material, layer) else {
        return UsdMaterial {
            material: standard.into(),
            textures,
        };
    };

    let mut colour = |name: &str, slot: TextureSlot, into: &mut Color| {
        match input(shader, name, slot, layer) {
            Some(Input::Value(value)) => {
                let n = value.flat_f32();
                if n.len() >= 3 {
                    *into = Color::new(n[0], n[1], n[2]);
                }
            }
            Some(Input::Texture(request)) => textures.push(request),
            None => {}
        }
    };
    colour("diffuseColor", TextureSlot::BaseColor, &mut standard.color);
    colour("emissiveColor", TextureSlot::Emissive, &mut standard.emissive);

    let mut scalar = |name: &str, slot: TextureSlot, into: &mut f32| {
        match input(shader, name, slot, layer) {
            Some(Input::Value(value)) => {
                if let Some(v) = value.as_f64() {
                    *into = v as f32;
                }
            }
            Some(Input::Texture(request)) => textures.push(request),
            None => {}
        }
    };
    scalar("roughness", TextureSlot::Roughness, &mut standard.roughness);
    scalar("metallic", TextureSlot::Metalness, &mut standard.metalness);
    scalar("opacity", TextureSlot::Opacity, &mut standard.opacity);
    scalar("occlusion", TextureSlot::Occlusion, &mut standard.ao_intensity);

    // Normal and displacement are textures or nothing: a constant normal is
    // not a thing anyone authors.
    for (name, slot) in [
        ("normal", TextureSlot::Normal),
        ("displacement", TextureSlot::Displacement),
    ] {
        if let Some(Input::Texture(request)) = input(shader, name, slot, layer) {
            textures.push(request);
        }
    }

    // Two of `UsdPreviewSurface`'s inputs have no home on a standard material,
    // and a file that authors either is describing a physical one. Reading
    // them into a standard material would drop them, which is how a physical
    // material exported by this crate came back as a duller standard one.
    let constant = |name: &str| match input(shader, name, TextureSlot::BaseColor, layer) {
        Some(Input::Value(value)) => value.as_f64().map(|v| v as f32),
        _ => None,
    };
    let ior = constant("ior");
    let clearcoat = constant("clearcoat");
    if ior.is_none() && clearcoat.is_none() {
        return UsdMaterial {
            material: standard.into(),
            textures,
        };
    }
    let physical = PhysicalMaterial {
        color: standard.color,
        emissive: standard.emissive,
        roughness: standard.roughness,
        metalness: standard.metalness,
        opacity: standard.opacity,
        ao_intensity: standard.ao_intensity,
        side: standard.side,
        ior: ior.unwrap_or(1.5),
        clearcoat: clearcoat.unwrap_or(0.0),
        clearcoat_roughness: constant("clearcoatRoughness").unwrap_or(0.0),
        ..Default::default()
    };
    UsdMaterial {
        material: physical.into(),
        textures,
    }
}

/// A face subset of a mesh, and the material bound to it.
///
/// A mesh with more than one material does not carry a list of them: it carries
/// `GeomSubset` children, each naming a set of faces and binding its own
/// material. Ignoring them draws the whole mesh in whichever material the mesh
/// itself binds — one colour where the asset has several, with nothing to say
/// anything went wrong.
#[derive(Debug, Clone)]
pub struct FaceSubset {
    pub name: String,
    /// Which faces, by index into `faceVertexCounts`.
    pub faces: Vec<u32>,
    /// The material bound to them, as an absolute path.
    pub material: Option<String>,
}

/// The material-binding subsets of a mesh, in the order authored.
///
/// Only the `materialBind` family counts: a mesh may carry subsets for all
/// sorts of purposes — a modeller's selection sets, a simulation's regions —
/// and binding a material off one of those would be inventing an assignment
/// nobody made.
pub fn material_subsets(mesh: &UsdPrim) -> Vec<FaceSubset> {
    mesh.children
        .iter()
        .filter(|child| child.type_name == "GeomSubset")
        .filter(|child| {
            child.value("familyName").and_then(|v| v.as_str()) == Some("materialBind")
        })
        // `elementType` defaults to `face`, which is the only kind that can
        // carry a material.
        .filter(|child| {
            !matches!(
                child.value("elementType").and_then(|v| v.as_str()),
                Some(other) if other != "face"
            )
        })
        .map(|child| FaceSubset {
            name: child.name.clone(),
            faces: child.value("indices").map(|v| v.flat_u32()).unwrap_or_default(),
            material: child
                .value("material:binding")
                .and_then(|v| v.as_str())
                .map(str::to_string),
        })
        .filter(|subset| !subset.faces.is_empty())
        .collect()
}

/// The faces of a mesh that no subset claims.
///
/// A `partition` family covers every face and leaves none; a `nonOverlapping`
/// one may leave some, and those keep the mesh's own material.
pub fn unclaimed_faces(total: usize, subsets: &[FaceSubset]) -> Vec<u32> {
    let mut claimed = vec![false; total];
    for subset in subsets {
        for face in &subset.faces {
            if let Some(slot) = claimed.get_mut(*face as usize) {
                *slot = true;
            }
        }
    }
    claimed
        .iter()
        .enumerate()
        .filter(|(_, taken)| !**taken)
        .map(|(i, _)| i as u32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parse::parse;
    use super::super::scene::to_scene;

    fn layer() -> UsdLayer {
        parse(include_str!("testdata/shade.usda")).expect("parses")
    }

    fn material_of(layer: &UsdLayer, prim: &str) -> UsdMaterial {
        let path = bound_material_path(layer, prim).expect("a binding");
        resolve(layer.prim_at(&path).expect("the material"), layer)
    }

    /// A material's values come off the shader its `outputs:surface` connects
    /// to, not off whichever child happens to be first.
    #[test]
    fn the_surface_shader_is_the_one_the_connection_names() {
        let layer = layer();
        let material = layer.prim_at("/World/Materials/Textured").unwrap();
        let shader = surface_shader(material, &layer).expect("a surface");
        assert_eq!(shader.name, "Surface");
        assert_eq!(
            shader.value("info:id").and_then(|v| v.as_str()),
            Some("UsdPreviewSurface")
        );
    }

    /// A `NodeGraph` forwards: its output is connected to something inside it,
    /// so following one connection means following several.
    #[test]
    fn a_node_graph_is_followed_through() {
        let layer = layer();
        let material = layer.prim_at("/World/Materials/ViaGraph").unwrap();
        let shader = surface_shader(material, &layer).expect("through the graph");
        assert_eq!(shader.name, "Inner");

        let resolved = resolve(material, &layer);
        let Material::Standard(standard) = &resolved.material else {
            panic!("standard");
        };
        assert!((standard.metalness - 0.8).abs() < 1e-5, "{}", standard.metalness);
        assert!((standard.color.r - 1.0).abs() < 1e-5);
    }

    /// Connected inputs are textures, and they come back as requests to load
    /// rather than as silently-defaulted constants.
    #[test]
    fn connected_inputs_become_texture_requests() {
        let layer = layer();
        let resolved = material_of(&layer, "/World/Painted");

        let mut slots: Vec<String> = resolved
            .textures
            .iter()
            .map(|t| format!("{:?}", t.slot))
            .collect();
        slots.sort();
        assert_eq!(slots, vec!["BaseColor", "Normal", "Roughness"]);

        let albedo = resolved
            .textures
            .iter()
            .find(|t| t.slot == TextureSlot::BaseColor)
            .unwrap();
        assert_eq!(albedo.file, "textures/albedo.png");
        assert_eq!(albedo.wrap_s, "repeat");
        assert_eq!(albedo.wrap_t, "clamp");
        // The primvar came through the reader the texture's `st` connects to.
        assert_eq!(albedo.uv, "st");

        // Values that were *not* connected are still read.
        let Material::Standard(standard) = &resolved.material else {
            panic!("standard");
        };
        assert!((standard.metalness - 0.25).abs() < 1e-5);
        assert!((standard.emissive.r - 0.1).abs() < 1e-5);
    }

    /// Scale and bias are not decoration: an 8-bit normal map carries
    /// `(2,2,2,1)` and `(-1,-1,-1,0)` so that `[0,1]` becomes `[-1,1]`, and
    /// `usdchecker` refuses a normal map without them.
    #[test]
    fn a_normal_map_keeps_its_scale_and_bias() {
        let layer = layer();
        let resolved = material_of(&layer, "/World/Painted");
        let normal = resolved
            .textures
            .iter()
            .find(|t| t.slot == TextureSlot::Normal)
            .expect("a normal map");
        assert_eq!(normal.scale, [2.0, 2.0, 2.0, 1.0]);
        assert_eq!(normal.bias, [-1.0, -1.0, -1.0, 0.0]);
        assert_eq!(normal.color_space, "raw", "a normal map is data, not colour");
    }

    /// A binding applies to everything beneath it. A mesh with none of its own
    /// is not unbound — it wears the nearest ancestor's, and missing that turns
    /// a bound asset grey.
    #[test]
    fn binding_is_inherited_down_the_namespace() {
        let layer = layer();
        assert_eq!(
            bound_material_path(&layer, "/World/Group/NoBindingOfItsOwn").as_deref(),
            Some("/World/Materials/Inherited"),
            "should have picked up World's binding"
        );
        // And a prim with its own binding keeps it.
        assert_eq!(
            bound_material_path(&layer, "/World/Painted").as_deref(),
            Some("/World/Materials/Textured")
        );

        let resolved = material_of(&layer, "/World/Group/NoBindingOfItsOwn");
        let Material::Standard(standard) = &resolved.material else {
            panic!("standard");
        };
        assert!((standard.color.b - 1.0).abs() < 1e-5, "the inherited blue");
        assert!((standard.roughness - 0.9).abs() < 1e-5);
    }

    /// The scene reports what each mesh's material wants loading.
    #[test]
    fn the_scene_reports_its_textures() {
        let scene = to_scene(&layer());
        let painted = scene
            .textures
            .iter()
            .find(|(name, _)| name == "Painted")
            .expect("Painted asked for textures");
        assert_eq!(painted.1.len(), 3);
        // A mesh with no textures is not listed at all.
        assert!(scene.textures.iter().all(|(name, _)| name != "Graphed"));
    }

    /// A mesh with two materials is two meshes here, because this crate's
    /// `Mesh` carries one. Ignoring the subsets would draw the whole thing in
    /// one colour with nothing to say anything was lost.
    #[test]
    fn a_mesh_with_subsets_splits_by_material() {
        use crate::core::ObjectKind;
        let layer = parse(include_str!("testdata/subset.usda")).unwrap();
        let scene = to_scene(&layer);

        fn find<'a>(
            scene: &'a super::super::scene::UsdScene,
            name: &str,
        ) -> Option<&'a crate::core::Object3D> {
            fn walk<'a>(
                scene: &'a super::super::scene::UsdScene,
                id: crate::core::ObjectId,
                name: &str,
            ) -> Option<&'a crate::core::Object3D> {
                let node = scene.arena.get(id)?;
                if node.name == name {
                    return Some(node);
                }
                node.children.iter().find_map(|c| walk(scene, *c, name))
            }
            scene.roots.iter().find_map(|r| walk(scene, *r, name))
        }

        let two_tone = find(&scene, "TwoTone").expect("the mesh");
        assert_eq!(two_tone.children.len(), 2, "one subset plus the remainder");

        let colours: Vec<(String, [f32; 3])> = two_tone
            .children
            .iter()
            .map(|c| {
                let node = scene.arena.get(*c).unwrap();
                let ObjectKind::Mesh(mesh) = &node.kind else {
                    panic!("{} should be a mesh", node.name);
                };
                let Material::Standard(standard) = &*mesh.material else {
                    panic!("standard");
                };
                (
                    node.name.clone(),
                    [standard.color.r, standard.color.g, standard.color.b],
                )
            })
            .collect();

        let blue = colours.iter().find(|(n, _)| n == "BluePart").expect("BluePart");
        assert_eq!(blue.1, [0.0, 0.0, 1.0], "the subset's own material");
        let rest = colours.iter().find(|(n, _)| n == "TwoTone").expect("the remainder");
        assert_eq!(rest.1, [1.0, 0.0, 0.0], "the mesh's own material");
    }

    /// Two of four faces belong to the subset, so each half has two.
    #[test]
    fn each_part_gets_only_its_own_faces() {
        use super::super::scene::mesh_geometry_of_faces;
        let layer = parse(include_str!("testdata/subset.usda")).unwrap();
        let mesh = layer.prim_at("/World/TwoTone").unwrap();

        let subsets = material_subsets(mesh);
        assert_eq!(subsets.len(), 1);
        assert_eq!(subsets[0].faces, vec![1, 3]);
        assert_eq!(unclaimed_faces(4, &subsets), vec![0, 2]);

        // Two quads is four triangles is twelve indices.
        let part = mesh_geometry_of_faces(mesh, 0, &subsets[0].faces).unwrap();
        assert_eq!(part.index.as_ref().map(|i| i.len()), Some(12));
        let rest = mesh_geometry_of_faces(mesh, 0, &unclaimed_faces(4, &subsets)).unwrap();
        assert_eq!(rest.index.as_ref().map(|i| i.len()), Some(12));
    }

    /// Only the `materialBind` family assigns materials. A mesh may carry
    /// subsets for a modeller's selection sets or a simulation's regions, and
    /// binding off one of those invents an assignment nobody made.
    #[test]
    fn only_the_material_family_counts() {
        let layer = parse(
            r#"#usda 1.0
def Mesh "M"
{
    int[] faceVertexCounts = [4, 4]
    int[] faceVertexIndices = [0, 1, 2, 3, 4, 5, 6, 7]
    point3f[] points = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (2, 0, 0), (3, 0, 0), (3, 1, 0), (2, 1, 0)
    ]

    def GeomSubset "Selection"
    {
        uniform token elementType = "face"
        uniform token familyName = "modellingSelection"
        int[] indices = [0]
    }

    def GeomSubset "Points"
    {
        uniform token elementType = "point"
        uniform token familyName = "materialBind"
        int[] indices = [0, 1]
    }
}
"#,
        )
        .unwrap();
        let subsets = material_subsets(layer.prim_at("/M").unwrap());
        assert!(
            subsets.is_empty(),
            "neither a different family nor a point subset assigns a material: {subsets:?}"
        );
    }

    /// And the same graph out of the crate form.
    #[test]
    fn the_binary_form_resolves_the_same_graph() {
        let binary =
            super::super::UsdLoader::parse_layer(include_bytes!("testdata/shade.usdc")).unwrap();
        let resolved = material_of(&binary, "/World/Painted");
        assert_eq!(resolved.textures.len(), 3);
        assert_eq!(
            resolved
                .textures
                .iter()
                .find(|t| t.slot == TextureSlot::BaseColor)
                .unwrap()
                .file,
            "textures/albedo.png"
        );
    }
}
