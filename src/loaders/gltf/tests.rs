use super::*;
use crate::core::ObjectKind;
use crate::materials::Material;

const PLAIN: &[u8] = include_bytes!("../testdata/sphere_plain.glb");
const MESHOPT: &[u8] = include_bytes!("../testdata/sphere_meshopt.glb");
const DRACO: &[u8] = include_bytes!("../testdata/sphere_draco.glb");
const FEATURES_GLTF: &str = include_str!("../testdata/gltf/features.gltf");
const FEATURES_GLB: &[u8] = include_bytes!("../testdata/gltf/features.glb");
const EXTERNAL_GLTF: &str = include_str!("../testdata/gltf/external.gltf");
const EXTERNAL_BIN: &[u8] = include_bytes!("../testdata/gltf/external.bin");
const CHECKER_PNG: &[u8] = include_bytes!("../testdata/gltf/checker.png");

/// Parse with the three.js-semantics parser, images decoded here.
fn parse(bytes: &[u8]) -> (GltfParser, ObjectArena) {
    let doc = Document::from_bytes(bytes).expect("container");
    let mut parser = GltfParser::new(doc, GltfOptions { refuse_unknown_required: false, ..Default::default() })
        .expect("parser");
    supply_images(&mut parser, &GltfImages::new()).expect("images");
    let mut arena = ObjectArena::new();
    parser.parse(&mut arena).expect("parse");
    (parser, arena)
}

/// The scene as three.js r165 prints it: `type "name" mat=Type:name inf=[..]`.
fn dump(p: &GltfParser, arena: &ObjectArena, o: usize, depth: usize, out: &mut String) {
    let obj = &p.objects[o];
    let ty = match obj.ty {
        // three.js r165's InstancedMesh keeps Mesh's `type`.
        ObjectType::InstancedMesh => "Mesh",
        t => t.name(),
    };
    out.push_str(&format!("{}{} \"{}\"", "  ".repeat(depth), ty, obj.name));
    if let Some(m) = obj.material {
        out.push_str(&format!(" mat={}:{}", p.materials[m].ty.name(), p.materials[m].name));
    }
    if let Some(inf) = &obj.morph_influences {
        let v: Vec<String> = inf.iter().map(|x| format!("{x}")).collect();
        out.push_str(&format!(" inf=[{}]", v.join(",")));
    }
    out.push('\n');
    for c in &arena.get(obj.id).unwrap().children {
        dump(p, arena, p.object_index(*c).unwrap(), depth + 1, out);
    }
}

fn find(p: &GltfParser, name: &str) -> usize {
    p.objects.iter().position(|o| o.name == name).unwrap_or_else(|| panic!("no object {name}"))
}

// Written by three.js r165's GLTFLoader for `testdata/gltf/features.gltf`
// (`tests/gltf/gltf-parity.mjs` checks every field; this is the outline).
const THREE_OUTLINE: &str = r#"Group "Main_Scene"
  Object3D "Parent"
    Mesh "Child" mat=MeshStandardMaterial:Textured
    Mesh "Dup" mat=MeshStandardMaterial:Blend
  Group "Dup_1"
    Mesh "Multi" mat=MeshBasicMaterial:Unlit
    Mesh "Multi_1" mat=MeshStandardMaterial:
    Mesh "Multi_2" mat=MeshStandardMaterial:
  Mesh "Shared_A" mat=MeshStandardMaterial:Plain
  Mesh "Shared_B" mat=MeshStandardMaterial:Plain
  Mesh "mesh_4" mat=MeshStandardMaterial:Blend
  PerspectiveCamera "CamPersp"
  OrthographicCamera "Cam_Ortho"
  PerspectiveCamera "CamShared"
  DirectionalLight "Lit"
    Object3D ""
  PointLight "light_1"
  SpotLight "SpotA"
    Object3D ""
  Group "SpotB"
    Group "_instance_1"
      Mesh "Multi" mat=MeshBasicMaterial:Unlit
      Mesh "Multi_1" mat=MeshStandardMaterial:
      Mesh "Multi_2" mat=MeshStandardMaterial:
    SpotLight "Spot_Light_instance_1"
      Object3D ""
  SkinnedMesh "Skinned" mat=MeshStandardMaterial:Plain
  Mesh "Morph" mat=MeshStandardMaterial:Plain inf=[0.1,0.2,0.3]
  Mesh "Instanced" mat=MeshStandardMaterial:Plain
  Group "Modes"
    Points "Modes_1" mat=PointsMaterial:Blend
    LineSegments "Modes_2" mat=LineBasicMaterial:Blend
    LineLoop "Modes_3" mat=LineBasicMaterial:
    Line "Modes_4" mat=LineBasicMaterial:Lines
    Mesh "Modes_5" mat=MeshStandardMaterial:Plain
    Mesh "Modes_6" mat=MeshStandardMaterial:Plain
  Object3D "Skeleton"
    Bone "Bone_Root"
      Bone "Bone_Tip"
  Mesh "Quantized" mat=MeshStandardMaterial:Textured
  Group "Multi_Morph"
    Mesh "MultiMorph" mat=MeshStandardMaterial: inf=[1]
    Mesh "MultiMorph_1" mat=MeshStandardMaterial: inf=[1]
  Group "Physical"
    Mesh "PhysicalMesh" mat=MeshPhysicalMaterial:Clearcoat & co
    Mesh "PhysicalMesh_1" mat=MeshPhysicalMaterial:Glass
    Mesh "PhysicalMesh_2" mat=MeshPhysicalMaterial:Brushed
"#;

#[test]
fn the_feature_file_has_three_js_structure_and_names() {
    for bytes in [FEATURES_GLTF.as_bytes(), FEATURES_GLB] {
        let (p, arena) = parse(bytes);
        let mut out = String::new();
        dump(&p, &arena, p.default_scene().unwrap(), 0, &mut out);
        assert_eq!(out, THREE_OUTLINE);
        let tracks: Vec<Vec<String>> = p
            .clips
            .iter()
            .map(|c| c.tracks.iter().map(|t| format!("{}.{}", p.objects[t.target].name, t.property)).collect())
            .collect();
        assert_eq!(
            tracks,
            vec![
                vec![
                    "Parent.position",
                    "Bone_Tip.quaternion",
                    "Shared_A.scale",
                    "Morph.morphTargetInfluences",
                    "Animated_Only.position"
                ],
                vec!["MultiMorph.morphTargetInfluences", "MultiMorph_1.morphTargetInfluences"],
            ]
        );
        assert_eq!(p.clips[0].name, "Move");
        assert_eq!(p.clips[1].name, "animation_1");
        let cams: Vec<String> = p.cameras.iter().map(|c| p.objects[c.unwrap()].name.clone()).collect();
        assert_eq!(cams, ["Lens", "Cam_Ortho", "Unused"]);
    }
}

#[test]
fn every_node_is_the_kind_of_rust_node_it_says() {
    let (p, arena) = parse(FEATURES_GLTF.as_bytes());
    for o in &p.objects {
        let node = arena.get(o.id).expect("node in the arena");
        assert_eq!(node.name, o.name);
        let ok = match (o.ty, &node.kind) {
            (ObjectType::Mesh, ObjectKind::Mesh(_)) => true,
            (ObjectType::SkinnedMesh, ObjectKind::SkinnedMesh(_)) => true,
            (ObjectType::InstancedMesh, ObjectKind::InstancedMesh(im)) => im.transforms.len() == 3,
            (ObjectType::Points, ObjectKind::Points(_)) => true,
            (ObjectType::LineSegments, ObjectKind::LineSegments(l)) => l.strip.is_none(),
            (ObjectType::Line, ObjectKind::LineSegments(l)) => l.strip.as_ref().is_some_and(|s| !s.closed),
            (ObjectType::LineLoop, ObjectKind::LineSegments(l)) => l.strip.as_ref().is_some_and(|s| s.closed),
            (ObjectType::DirectionalLight | ObjectType::PointLight | ObjectType::SpotLight, ObjectKind::Light(_)) => true,
            (ObjectType::Group, ObjectKind::Group) => node.is_group,
            (
                ObjectType::Object3D | ObjectType::Bone | ObjectType::PerspectiveCamera | ObjectType::OrthographicCamera,
                ObjectKind::Group,
            ) => !node.is_group,
            _ => false,
        };
        assert!(ok, "{:?} {:?} became {:?}", o.ty, o.name, std::mem::discriminant(&node.kind));
    }
}

#[test]
fn transforms_follow_trs_and_matrix() {
    let (p, arena) = parse(FEATURES_GLTF.as_bytes());
    let parent = arena.get(p.objects[find(&p, "Parent")].id).unwrap();
    assert_eq!(parent.position, Vector3::new(1.0, 2.0, 3.0));
    assert!((parent.quaternion.y - 0.382_683_4).abs() < 1e-6);
    assert_eq!(parent.scale, Vector3::new(2.0, 2.0, 2.0));
    // `matrix`: a rotation of -90° about X (y to -z) and a translation.
    let child = arena.get(p.objects[find(&p, "Child")].id).unwrap();
    assert_eq!(child.position, Vector3::new(5.0, 6.0, 7.0));
    assert!((child.quaternion.x + std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6, "{:?}", child.quaternion);
    assert!((child.scale.x - 1.0).abs() < 1e-6);
}

#[test]
fn attributes_are_read_as_three_reads_them() {
    let (p, _) = parse(FEATURES_GLTF.as_bytes());
    let geo = |name: &str| &p.geometries[p.objects[find(&p, name)].geometry.unwrap()];
    // Quantized: SHORT positions as they are, BYTE normals and UNSIGNED_SHORT
    // uvs and UNSIGNED_BYTE colours as fractions (signed ones clamped at -1).
    let q = geo("Quantized");
    assert_eq!(q.attribute_order, ["position", "normal", "uv", "color"]);
    let pos = &q.get().get_attribute("position").unwrap().array;
    assert_eq!(pos, &[0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 50.0, 0.0, 100.0, 0.0, 0.0, 0.0]);
    let n = &q.get().get_attribute("normal").unwrap().array;
    assert_eq!(n[2], 1.0);
    assert_eq!(n[8], -1.0);
    let uv = &q.get().get_attribute("uv").unwrap().array;
    assert_eq!(uv[7], 32768.0 / 65535.0);
    let c = q.get().get_attribute("color").unwrap();
    assert_eq!(c.item_size, 4);
    assert_eq!(c.array[7], 128.0 / 255.0);
    assert_eq!(q.get().index.as_ref().unwrap(), &vec![0, 1, 2, 2, 1, 3]);
    // Interleaved views read by byteStride.
    let multi = find(&p, "Multi_1");
    let g = p.geometries[p.objects[multi].geometry.unwrap()].get();
    assert_eq!(g.get_attribute("position").unwrap().array, vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0]);
    assert_eq!(g.get_attribute("normal").unwrap().array, vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
    // A sparse accessor over a buffer view: one value substituted in a copy.
    let sp = geo("mesh_4");
    assert_eq!(sp.get().get_attribute("position").unwrap().array[3..6], [0.5, 0.5, 0.5]);
    // Bounds from the accessor's min/max.
    let bb = geo("Child").get().bounding_box.unwrap();
    assert_eq!((bb.min, bb.max), (Vector3::ZERO, Vector3::new(1.0, 1.0, 0.0)));
}

#[test]
fn morph_targets_skins_and_instances() {
    let (p, arena) = parse(FEATURES_GLTF.as_bytes());
    let m = &p.objects[find(&p, "Morph")];
    let geo = &p.geometries[m.geometry.unwrap()];
    assert!(geo.get().morph_targets_relative);
    assert_eq!(geo.morph_order, ["position", "normal"]);
    let pos = &geo.get().morph_attributes["position"];
    // The sparse target, no buffer view: one vertex moved.
    assert_eq!(pos[1].array, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.0, 0.0, 0.0, 0.0]);
    // A target with NORMAL only gets the base positions (three.js r165).
    assert_eq!(pos[2].array, geo.get().get_attribute("position").unwrap().array);
    assert_eq!(m.morph_dictionary, [("bulge".into(), 0), ("sparse".into(), 1), ("tilt".into(), 2)]);

    let s = &p.objects[find(&p, "Skinned")];
    let sk = &p.skeletons[s.skeleton.unwrap()];
    assert_eq!(sk.bones.iter().map(|b| p.objects[*b].name.as_str()).collect::<Vec<_>>(), ["Bone_Root", "Bone_Tip"]);
    assert_eq!(sk.inverses[1][12..15], [-0.5, -1.0, 0.0]);
    let w = &p.geometries[s.geometry.unwrap()].get().get_attribute("skinWeight").unwrap().array;
    // normalizeSkinWeights, written back into the UNSIGNED_BYTE storage the
    // way three.js r165 does it: [100, 100] halves to 127.5 and rounds down
    // (the double-precision product is 127.49999…), as three.js reports.
    let bytes: Vec<f32> = w.iter().map(|v| (v * 255.0).round()).collect();
    assert_eq!(bytes, [200.0, 55.0, 0.0, 0.0, 127.0, 127.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0]);
    let ObjectKind::SkinnedMesh(sm) = &arena.get(s.id).unwrap().kind else { panic!() };
    assert_eq!(sm.skeleton.bones.len(), 2);
    assert!(sm.geometry.get_attribute("joint").is_some() && sm.geometry.get_attribute("weight").is_some());

    let inst = &p.objects[find(&p, "Instanced")];
    let i = inst.instances.as_ref().unwrap();
    assert_eq!(i.count, 3);
    assert_eq!(i.matrices[16 + 12..16 + 15], [2.0, 0.0, 0.0]);
    assert_eq!(i.colors.as_ref().unwrap().0[..3], [1.0, 0.0, 0.0]);
    // Custom instance attributes go on the geometry under their own name.
    assert!(p.geometries[inst.geometry.unwrap()].attribute_order.contains(&"_ID".to_string()));
}

#[test]
fn materials_carry_three_js_parameters() {
    let (p, _) = parse(FEATURES_GLTF.as_bytes());
    let mat = |name: &str| p.materials.iter().find(|m| m.name == name && m.params.contains_key("color")).unwrap();
    let textured = mat("Textured");
    assert_eq!(textured.params["color"], Param::Color([1.0, 0.5, 0.25]));
    assert_eq!(textured.num("opacity"), Some(0.75));
    assert_eq!(textured.num("alphaTest"), Some(0.3));
    assert_eq!(textured.num("side"), Some(2.0));
    assert_eq!(textured.num("aoMapIntensity"), Some(0.6));
    let map = &p.textures[textured.texture("map").unwrap()];
    assert_eq!((map.offset, map.repeat, map.rotation), ([0.5, 0.25], [2.0, 3.0], 0.3));
    assert!(map.srgb);
    let ao = &p.textures[textured.texture("aoMap").unwrap()];
    assert_eq!(ao.channel, 1);
    assert_eq!(ao.min_filter, 1008);
    let glass = mat("Glass");
    assert_eq!(glass.ty, MaterialType::MeshPhysicalMaterial);
    assert_eq!(glass.num("attenuationDistance"), Some(2.0));
    assert_eq!(glass.params["iridescenceThicknessRange"], Param::Range([100.0, 500.0]));
    assert_eq!(mat("Brushed").num("attenuationDistance"), Some(f64::INFINITY));
    // Unlit: a basic material; no normal map even though the file has one.
    let unlit = mat("Unlit");
    assert_eq!(unlit.ty, MaterialType::MeshBasicMaterial);
    assert!(unlit.texture("normalMap").is_none());
    // The renderer's material is built from the same parameters.
    let Some(Material::Physical(ph)) = mat("Clearcoat & co").rust.as_deref() else { panic!() };
    assert_eq!((ph.clearcoat, ph.ior, ph.emissive_intensity), (0.8, 1.8, 4.0));
    assert!(!ph.map.as_ref().unwrap().flip_y);
}

#[test]
fn external_resources_are_supplied_by_the_caller() {
    let doc = Document::from_bytes(EXTERNAL_GLTF.as_bytes()).unwrap();
    assert_eq!(doc.external_buffers(), vec![(0, "external.bin".to_string())]);
    let mut doc = doc;
    doc.set_buffer(0, EXTERNAL_BIN.to_vec());
    let mut parser = GltfParser::new(doc, GltfOptions::default()).unwrap();
    let pending = parser.pending_images().unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].source.uri.as_deref(), Some("checker.png"));
    assert!(parser.set_image_bytes(0, CHECKER_PNG));
    assert!(parser.pending_images().unwrap().is_empty());
    let mut arena = ObjectArena::new();
    parser.parse(&mut arena).unwrap();
    let m = &parser.materials[parser.objects[find(&parser, "Ext")].material.unwrap()];
    let t = &parser.textures[m.texture("map").unwrap()];
    assert_eq!(t.image_data.size(), (4, 4));
    assert_eq!(t.name, "checker.png");
}

// ---- the original fixtures ----

/// The one mesh in a file, as positions, normals and indices.
fn mesh(bytes: &[u8]) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<u32>>) {
    let scene = GltfLoader::parse_glb(bytes).expect("parses");
    let (_, object) = scene
        .arena
        .nodes
        .iter()
        .find(|(_, o)| matches!(o.kind, ObjectKind::Mesh(_)))
        .expect("a mesh");
    let ObjectKind::Mesh(m) = &object.kind else { unreachable!() };
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
    let (plain, _, plain_index) = mesh(PLAIN);
    let (ours, normals, index) = mesh(MESHOPT);
    assert_eq!(ours.len(), plain.len(), "same vertex count");
    assert_eq!(index.as_ref().map(Vec::len), plain_index.as_ref().map(Vec::len), "same index count");
    assert!(normals.is_some(), "normals survived the filter");
    let tolerance = 2.0 / 32767.0 * 4.0;
    for v in ours.chunks_exact(3) {
        let nearest = plain
            .chunks_exact(3)
            .map(|p| ((p[0] - v[0]).powi(2) + (p[1] - v[1]).powi(2) + (p[2] - v[2]).powi(2)).sqrt())
            .fold(f32::MAX, f32::min);
        assert!(nearest < tolerance, "decoded vertex {v:?} is {nearest} from anything in the original");
    }
    for v in ours.chunks_exact(3) {
        let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((r - 1.0).abs() < 1e-3, "{v:?} is at radius {r}, not 1");
    }
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
        Err(GltfError::UnsupportedExtension(name)) => assert_eq!(name, "EXT_not_a_real_extension"),
        Err(other) => panic!("expected a named refusal, got {other:?}"),
        Ok(_) => panic!("a file requiring an unknown extension must not load"),
    }
}

/// three.js only warns about it; so does the parser when asked to.
#[test]
fn an_unknown_required_extension_is_a_warning_in_three_mode() {
    let json = r#"{"asset":{"version":"2.0"},
        "extensionsUsed":["EXT_not_a_real_extension"],
        "extensionsRequired":["EXT_not_a_real_extension"],
        "scene":0,"scenes":[{"nodes":[]}],"nodes":[]}"#;
    let doc = Document::from_bytes(json.as_bytes()).unwrap();
    let p = GltfParser::new(doc, GltfOptions { refuse_unknown_required: false, ..Default::default() }).unwrap();
    assert_eq!(p.warnings, ["THREE.GLTFLoader: Unknown extension \"EXT_not_a_real_extension\"."]);
}

/// An extension merely *used* does not stop the load.
#[test]
fn an_extension_only_used_is_not_fatal() {
    let json = r#"{"asset":{"version":"2.0"},
        "extensionsUsed":["KHR_materials_variants","SOMETHING_INVENTED"],
        "scene":0,"scenes":[{"nodes":[]}],"nodes":[]}"#;
    let root = crate::loaders::json::parse(json).expect("parses");
    assert!(GltfLoader::build_for_test(&root).is_ok());
}

/// And one that is required *and* implemented loads.
#[test]
fn a_required_extension_that_is_implemented_loads() {
    // The meshopt fixture requires both meshopt and quantization.
    assert!(GltfLoader::parse_glb(MESHOPT).is_ok());
}

/// A glTF 1.0 asset is refused as three.js refuses it.
#[test]
fn a_version_one_asset_is_refused() {
    let doc = Document::from_bytes(br#"{"asset":{"version":"1.0"}}"#);
    assert!(matches!(doc, Err(GltfError::UnsupportedVersion)));
}

/// A Draco primitive decodes to the same mesh as its uncompressed twin.
///
/// The comparison has to be order-independent: Draco emits points in the
/// order its traversal reached them, which is not the order the plain file
/// stores them in. So the two are compared as *sets* of positions and as sets
/// of triangles.
#[test]
fn a_draco_primitive_decodes_to_the_same_mesh_as_its_plain_twin() {
    let (dp, dn, di) = mesh(DRACO);
    let (pp, pn, pi) = mesh(PLAIN);
    assert_eq!(dp.len(), pp.len(), "the same number of position floats");
    assert_eq!(dp.len() / 3, 423);
    assert_eq!(di.as_ref().map(|i| i.len()), Some(2160));
    assert_eq!(pi.as_ref().map(|i| i.len()), Some(2160));
    assert!(dn.is_some() && pn.is_some(), "both carry normals");
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
    assert!(worst < 2e-3, "a decoded position is {worst} from the nearest in the plain file");
    let hit: std::collections::BTreeSet<usize> = canonical.iter().copied().collect();
    assert_eq!(hit.len(), 362, "the decode does not cover every position");
    let plain_canonical: Vec<usize> = pp.chunks(3).map(|p| nearest(p).0).collect();
    let faces = |idx: &[u32], canon: &[usize]| -> std::collections::BTreeSet<[usize; 3]> {
        idx.chunks(3)
            .map(|t| {
                let mut k = [canon[t[0] as usize], canon[t[1] as usize], canon[t[2] as usize]];
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
#[test]
fn draco_attributes_are_matched_by_id_not_order() {
    let (positions, normals, _) = mesh(DRACO);
    let normals = normals.expect("normals");
    assert_eq!(normals.len(), positions.len());
    let mut worst = 0.0f32;
    for (p, n) in positions.chunks(3).zip(normals.chunks(3)) {
        let dot = p[0] * n[0] + p[1] * n[1] + p[2] * n[2];
        worst = worst.max((dot - 1.0).abs());
    }
    assert!(worst < 0.05, "normals disagree with positions by {worst} — attributes are crossed");
}

/// The uncompressed twin still reads.
#[test]
fn the_plain_file_still_reads() {
    let (positions, normals, index) = mesh(PLAIN);
    assert_eq!(positions.len() / 3, 423);
    assert!(normals.is_some());
    assert_eq!(index.map(|i| i.len()), Some(2160));
}

/// A truncated stream is an error, not a panic — of any of the fixtures.
#[test]
fn truncated_files_do_not_panic() {
    for file in [MESHOPT, DRACO, PLAIN, FEATURES_GLB, FEATURES_GLTF.as_bytes()] {
        for cut in [4, 11, 16, 64, 256, 1024, 4096, 7000, 12000] {
            if cut < file.len() {
                let _ = GltfLoader::parse_glb(&file[..cut]);
                let doc = Document::from_bytes(&file[..cut]);
                if let Ok(doc) = doc {
                    if let Ok(mut p) = GltfParser::new(doc, GltfOptions::default()) {
                        let _ = supply_images(&mut p, &GltfImages::new());
                        let _ = p.parse(&mut ObjectArena::new());
                    }
                }
            }
        }
    }
}

/// `testdata/morph_sparse.gltf`: one triangle, a sparse POSITION target
/// (accessor without a buffer view, one vertex moved) and a dense one,
/// `weights: [0.25, 0.5]`, `extras.targetNames: ["up", "wide"]`, and a
/// LINEAR `weights` animation over two keys.
///
/// three.js r165 `GLTFLoader` reads it as `morphTargetsRelative: true`,
/// position targets `[[0,0,0, 0,0,1, 0,0,0], [0.5,0,0, 0.5,0,0, 0,0.5,0]]`,
/// influences `[0.25, 0.5]`, dictionary `{up: 0, wide: 1}`, and one
/// `morphTargetInfluences` track, times `[0, 1]`, values `[0,1, 1,0]`.
#[test]
fn morph_targets_load_as_three() {
    let src = include_str!("../testdata/morph_sparse.gltf");
    let scene = GltfLoader::parse_gltf(src, &HashMap::new()).expect("parses");
    let id = scene.roots[0];
    let ObjectKind::Mesh(mesh) = &scene.arena.get(id).unwrap().kind else {
        panic!("a mesh node");
    };
    let g = &mesh.geometry;
    assert!(g.morph_targets_relative);
    let targets: Vec<Vec<f32>> = g.morph_attributes["position"].iter().map(|a| a.array.clone()).collect();
    assert_eq!(
        targets,
        vec![vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], vec![0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0]]
    );
    assert_eq!(mesh.morph_influences, vec![0.25, 0.5]);
    assert_eq!(mesh.morph_target_dictionary.get("up"), Some(&0));
    assert_eq!(mesh.morph_target_dictionary.get("wide"), Some(&1));
    let tracks = &scene.animations[0].tracks;
    assert_eq!(tracks.len(), 2);
    for (t, want) in tracks.iter().zip([[0.0, 1.0], [1.0, 0.0]]) {
        assert_eq!(t.object, id);
        assert_eq!(t.times, vec![0.0, 1.0]);
        assert_eq!(t.sample_scalar(0.0), Some(want[0]));
        assert_eq!(t.sample_scalar(1.0), Some(want[1]));
    }
}
