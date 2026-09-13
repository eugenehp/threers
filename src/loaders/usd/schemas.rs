//! The USD schemas a renderer needs beyond `UsdGeomMesh`.
//!
//! A `Mesh` is only one of the things a USD stage is made of. Lights carry the
//! lighting, cameras carry the shot, the quadric gprims carry most of the
//! blocking geometry in a layout file, and `Points` and `BasisCurves` carry
//! everything that is not a surface. A loader that handles only meshes opens a
//! production stage and shows a fraction of it — silently, since the prims are
//! all still there as empty groups.
//!
//! # What the numbers mean
//!
//! USD's light intensities are radiometric and its cameras are described in
//! millimetres of film. Neither maps onto a real-time renderer without a
//! convention, and the conventions here are stated where they are applied
//! rather than buried: an intensity is taken at face value, and a camera's
//! field of view comes from its aperture and focal length by the same formula
//! every DCC uses.

use crate::cameras::{OrthographicCamera, PerspectiveCamera};
use crate::core::{BufferAttribute, BufferGeometry, InstancedMesh, LineSegments, Points};
use crate::geometries::{
    BoxGeometry, CapsuleGeometry, ConeGeometry, CylinderGeometry, PlaneGeometry, SphereGeometry,
};
use crate::lights::{
    AmbientLight, DirectionalLight, Light, PointLight, RectAreaLight, SpotLight,
};
use crate::materials::{Material, PointsMaterial};
use crate::math::{Color, Matrix4, Quaternion, Vector3};

use super::parse::{UsdLayer, UsdPrim};

/// A camera found on the stage, with the node it belongs to.
///
/// Cameras are not part of the scene graph — nothing in `ObjectKind` is a
/// camera — so they come back beside it, each naming the object whose world
/// transform places it.
#[derive(Debug, Clone)]
pub struct UsdCamera {
    /// The scene-graph node carrying this camera's transform.
    pub object: crate::core::ObjectId,
    pub name: String,
    pub perspective: Option<PerspectiveCamera>,
    pub orthographic: Option<OrthographicCamera>,
}

/// Whether a prim's `purpose` means it should be drawn.
///
/// `render` and the default are drawn; `proxy` is a stand-in that a renderer
/// shows only when the real thing is unloaded, and `guide` is never shown.
pub fn is_drawable(prim: &UsdPrim, show_guides: bool, show_proxies: bool) -> bool {
    match prim.value("purpose").and_then(|v| v.as_str()) {
        Some("guide") => show_guides,
        Some("proxy") => show_proxies,
        _ => true,
    }
}

fn number(prim: &UsdPrim, name: &str, fallback: f32) -> f32 {
    prim.value(name)
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(fallback)
}

/// A light's colour, which USD writes under an `inputs:` prefix on the modern
/// schemas and bare on the older ones.
fn colour(prim: &UsdPrim, fallback: Color) -> Color {
    let value = prim
        .value("inputs:color")
        .or_else(|| prim.value("color"))
        .map(|v| v.flat_f32())
        .unwrap_or_default();
    if value.len() >= 3 {
        Color::new(value[0], value[1], value[2])
    } else {
        fallback
    }
}

fn light_number(prim: &UsdPrim, name: &str, fallback: f32) -> f32 {
    prim.value(&format!("inputs:{name}"))
        .or_else(|| prim.value(name))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(fallback)
}

/// A `UsdLux` light as this renderer's equivalent.
///
/// USD scales intensity by `exposure` as a power of two, which is how a
/// lighter states a stop rather than a multiplier.
pub fn light(prim: &UsdPrim) -> Option<Light> {
    let colour = colour(prim, Color::WHITE);
    let intensity = light_number(prim, "intensity", 1.0) * light_number(prim, "exposure", 0.0).exp2();

    Some(match prim.type_name.as_str() {
        // The sun: parallel rays, so only the direction matters.
        "DistantLight" => Light::Directional(DirectionalLight::new(colour, intensity)),
        // A sphere with a cone on it is a spotlight; without one it is a point.
        "SphereLight" | "DiskLight" | "CylinderLight" => {
            match prim.value("inputs:shaping:cone:angle").and_then(|v| v.as_f64()) {
                Some(angle) => {
                    let mut spot = SpotLight::new(colour, intensity);
                    // USD states the cone's half-angle in degrees.
                    spot.angle = (angle as f32).to_radians();
                    spot.penumbra = light_number(prim, "shaping:cone:softness", 0.0).clamp(0.0, 1.0);
                    Light::Spot(spot)
                }
                None => Light::Point(PointLight::new(colour, intensity)),
            }
        }
        "RectLight" => Light::RectArea(RectAreaLight::new(
            colour,
            intensity,
            light_number(prim, "width", 1.0),
            light_number(prim, "height", 1.0),
        )),
        // A dome is the sky: everywhere at once. With no texture to sample it
        // is an ambient term, which is what a real-time renderer can do with it.
        "DomeLight" => Light::Ambient(AmbientLight::new(colour, intensity)),
        "GeometryLight" | "PortalLight" => return None,
        _ => return None,
    })
}

/// The quadric and cube gprims, as the geometry each describes.
///
/// USD's defaults are not this renderer's: a `Cube` with nothing said is two
/// units across, a `Sphere` is one unit in *radius*, and a `Cylinder` is two
/// high. Taking a zero here instead would silently collapse a layout file.
pub fn gprim(prim: &UsdPrim) -> Option<BufferGeometry> {
    const SEGMENTS: usize = 32;
    let geometry = match prim.type_name.as_str() {
        "Cube" => {
            let size = number(prim, "size", 2.0);
            BoxGeometry::new(size, size, size)
        }
        "Sphere" => SphereGeometry::new(number(prim, "radius", 1.0), SEGMENTS, SEGMENTS / 2),
        "Cylinder" => {
            let radius = number(prim, "radius", 1.0);
            CylinderGeometry::new(
                radius,
                radius,
                number(prim, "height", 2.0),
                SEGMENTS,
                1,
                false,
                0.0,
                std::f32::consts::TAU,
            )
        }
        "Cone" => ConeGeometry::new(
            number(prim, "radius", 1.0),
            number(prim, "height", 2.0),
            SEGMENTS,
            1,
            false,
            0.0,
            std::f32::consts::TAU,
        ),
        // USD's `height` is the span between the cap centres, which is what
        // this generator calls `length`.
        "Capsule" => CapsuleGeometry::new(
            number(prim, "radius", 0.5),
            number(prim, "height", 1.0),
            SEGMENTS / 4,
            SEGMENTS,
        ),
        "Plane" => PlaneGeometry::new(number(prim, "width", 1.0), number(prim, "length", 1.0)),
        _ => return None,
    };
    Some(orient(geometry, prim))
}

/// Turn a gprim to face the axis it declares.
///
/// USD's quadrics stand along `axis`, which defaults to `Z`; every generator
/// here builds along `Y`. A cylinder that ignores this lies down.
fn orient(mut geometry: BufferGeometry, prim: &UsdPrim) -> BufferGeometry {
    let axis = prim
        .value("axis")
        .and_then(|v| v.as_str())
        .unwrap_or(match prim.type_name.as_str() {
            // A `Plane` defaults to lying in the XZ plane, which is the way
            // this generator does *not* build it.
            "Plane" => "Z",
            _ => "Z",
        })
        .to_string();
    let quarter = std::f32::consts::FRAC_PI_2;
    let (about, angle) = match (prim.type_name.as_str(), axis.as_str()) {
        ("Plane", "Y") => return geometry, // already flat in XZ
        ("Plane", "Z") => (Vector3::new(1.0, 0.0, 0.0), quarter),
        ("Plane", _) => (Vector3::new(0.0, 0.0, 1.0), quarter),
        (_, "Y") => return geometry,
        (_, "Z") => (Vector3::new(1.0, 0.0, 0.0), quarter),
        (_, _) => (Vector3::new(0.0, 0.0, 1.0), -quarter),
    };
    rotate(&mut geometry, about, angle);
    geometry
}

/// Rotate a geometry's positions and normals in place.
fn rotate(geometry: &mut BufferGeometry, axis: Vector3, angle: f32) {
    let (s, c) = angle.sin_cos();
    let turn = |v: [f32; 3]| -> [f32; 3] {
        let v = Vector3::new(v[0], v[1], v[2]);
        // Rodrigues' rotation, which needs no matrix type to be involved.
        let dot = axis.x * v.x + axis.y * v.y + axis.z * v.z;
        let cross = Vector3::new(
            axis.y * v.z - axis.z * v.y,
            axis.z * v.x - axis.x * v.z,
            axis.x * v.y - axis.y * v.x,
        );
        [
            v.x * c + cross.x * s + axis.x * dot * (1.0 - c),
            v.y * c + cross.y * s + axis.y * dot * (1.0 - c),
            v.z * c + cross.z * s + axis.z * dot * (1.0 - c),
        ]
    };
    for name in ["position", "normal"] {
        let Some(attribute) = geometry.get_attribute(name) else {
            continue;
        };
        let turned: Vec<f32> = attribute
            .array
            .chunks_exact(3)
            .flat_map(|c| turn([c[0], c[1], c[2]]))
            .collect();
        geometry.set_attribute(name, BufferAttribute::new(turned, 3));
    }
}

/// A `UsdGeomPoints` as a point cloud.
pub fn points(prim: &UsdPrim) -> Option<Points> {
    let positions = prim.value("points")?.flat_f32();
    if positions.is_empty() {
        return None;
    }
    let mut geometry = BufferGeometry::new();
    geometry.set_attribute("position", BufferAttribute::new(positions.clone(), 3));

    let colours = prim
        .value("primvars:displayColor")
        .map(|v| v.flat_f32())
        .unwrap_or_default();
    if colours.len() == positions.len() {
        geometry.set_attribute("color", BufferAttribute::new(colours, 3));
    }

    let mut material = PointsMaterial::default();
    // `widths` is a diameter per point; a single width applies to all of them.
    if let Some(widths) = prim.value("widths").map(|v| v.flat_f32()) {
        if let Some(first) = widths.first() {
            material.size = *first;
        }
    }
    Some(Points::new(geometry, Material::Points(material)))
}

/// A `UsdGeomBasisCurves` as line segments.
///
/// Cubic curves are drawn through their control points rather than evaluated:
/// the shape is right to within the hull, and a renderer that wants the true
/// curve has the control points to build it from.
pub fn curves(prim: &UsdPrim) -> Option<LineSegments> {
    let positions = prim.value("points")?.flat_f32();
    if positions.len() < 6 {
        return None;
    }
    let counts: Vec<u32> = prim
        .value("curveVertexCounts")
        .map(|v| v.flat_u32())
        .unwrap_or_else(|| vec![(positions.len() / 3) as u32]);

    // Each curve becomes a run of segments; `LineSegments` draws pairs, so
    // every interior point appears twice.
    let mut line = Vec::new();
    let mut at = 0usize;
    for count in counts {
        let count = count as usize;
        for i in 0..count.saturating_sub(1) {
            let a = (at + i) * 3;
            let b = (at + i + 1) * 3;
            if b + 2 >= positions.len() {
                break;
            }
            line.extend_from_slice(&positions[a..a + 3]);
            line.extend_from_slice(&positions[b..b + 3]);
        }
        at += count;
    }
    if line.is_empty() {
        return None;
    }
    let mut geometry = BufferGeometry::new();
    geometry.set_attribute("position", BufferAttribute::new(line, 3));
    Some(LineSegments::new(
        geometry,
        Material::Line(crate::materials::LineBasicMaterial::new(Color::WHITE)),
    ))
}

/// A `UsdGeomCamera`, as whichever projection it declares.
///
/// USD describes a camera the way a physical one is described: a focal length
/// and an aperture, both in millimetres of film. The field of view follows from
/// the two, which is the conversion every DCC applies and the reason a camera
/// authored in one looks the same in another.
pub fn camera(prim: &UsdPrim, object: crate::core::ObjectId) -> UsdCamera {
    let horizontal = number(prim, "horizontalAperture", 20.955);
    let vertical = number(prim, "verticalAperture", 15.2908);
    let focal = number(prim, "focalLength", 50.0).max(1e-6);
    let clip = prim
        .value("clippingRange")
        .map(|v| v.flat_f32())
        .unwrap_or_default();
    let near = clip.first().copied().unwrap_or(1.0);
    let far = clip.get(1).copied().unwrap_or(1_000_000.0);

    let orthographic = prim.value("projection").and_then(|v| v.as_str()) == Some("orthographic");
    let mut out = UsdCamera {
        object,
        name: prim.name.clone(),
        perspective: None,
        orthographic: None,
    };
    if orthographic {
        // An orthographic camera's apertures are in scene units rather than
        // millimetres, by USD's own definition.
        let (w, h) = (horizontal * 0.5, vertical * 0.5);
        out.orthographic = Some(OrthographicCamera::new(-w, w, h, -h, near, far));
    } else {
        let fov = 2.0 * (vertical * 0.5 / focal).atan();
        let aspect = if vertical > 0.0 { horizontal / vertical } else { 1.0 };
        let mut camera = PerspectiveCamera::new(fov.to_degrees(), aspect, near, far);
        // Depth of field, which USD states the same way a lens does: where it
        // is focused and how far the iris is open.
        camera.focus_distance = number(prim, "focusDistance", 0.0);
        camera.f_stop = number(prim, "fStop", 0.0);
        out.perspective = Some(camera);
    }
    out
}

/// A `UsdGeomPointInstancer` as one instanced mesh per prototype.
///
/// This is how a stage holds a forest: one tree, and a hundred thousand
/// positions to put it at. A loader without it either shows nothing or expands
/// every instance into its own mesh, and the second is how a scene that fits in
/// memory stops fitting.
///
/// Instances are grouped by the prototype they name, because an instanced draw
/// can only carry one geometry — so `[tree, rock, tree]` becomes two meshes of
/// two and one instances, not three meshes.
pub fn point_instancer(
    prim: &UsdPrim,
    layer: &UsdLayer,
    geometry_of: impl Fn(&UsdPrim) -> Option<(BufferGeometry, Material)>,
) -> Vec<(String, InstancedMesh)> {
    let Some(positions) = prim.value("positions").map(|v| v.flat_f32()) else {
        return Vec::new();
    };
    let count = positions.len() / 3;
    if count == 0 {
        return Vec::new();
    }

    // Which prototype each instance uses. With none said they all use the
    // first, which is what USD does.
    let indices: Vec<u32> = prim
        .value("protoIndices")
        .map(|v| v.flat_u32())
        .unwrap_or_else(|| vec![0; count]);

    // The prototypes themselves, by the paths the relationship names.
    let targets = prim
        .value("prototypes")
        .map(|v| {
            v.flat_tokens()
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if targets.is_empty() {
        return Vec::new();
    }

    let orientations = prim
        .value("orientations")
        .map(|v| v.flat_f32())
        .unwrap_or_default();
    let scales = prim.value("scales").map(|v| v.flat_f32()).unwrap_or_default();
    // `invisibleIds` is how a shot turns off individual instances without
    // rebuilding the set.
    let hidden: Vec<i64> = prim
        .value("invisibleIds")
        .map(|v| v.flat_f32().iter().map(|f| *f as i64).collect())
        .unwrap_or_default();

    let mut out: Vec<(String, InstancedMesh)> = Vec::new();
    for (slot, path) in targets.iter().enumerate() {
        let Some(prototype) = layer.prim_at(path) else {
            continue;
        };
        let Some((geometry, material)) = geometry_of(prototype) else {
            continue;
        };

        let transforms: Vec<Matrix4> = (0..count)
            .filter(|i| indices.get(*i).copied().unwrap_or(0) as usize == slot)
            .filter(|i| !hidden.contains(&(*i as i64)))
            .map(|i| {
                let position = Vector3::new(
                    positions[i * 3],
                    positions[i * 3 + 1],
                    positions[i * 3 + 2],
                );
                // USD writes a quaternion real part first; this one takes it
                // last.
                let rotation = match orientations.get(i * 4..i * 4 + 4) {
                    Some(q) => Quaternion::new(q[1], q[2], q[3], q[0]),
                    None => Quaternion::identity(),
                };
                let scale = match scales.get(i * 3..i * 3 + 3) {
                    Some(s) => Vector3::new(s[0], s[1], s[2]),
                    None => Vector3::new(1.0, 1.0, 1.0),
                };
                Matrix4::compose(position, rotation, scale)
            })
            .collect();
        if transforms.is_empty() {
            continue;
        }

        let mut mesh = InstancedMesh::new(geometry, material, transforms.len());
        mesh.transforms = transforms;
        out.push((prototype.name.clone(), mesh));
    }
    out
}

/// Whether a type name is one this module turns into something.
pub fn is_light(type_name: &str) -> bool {
    matches!(
        type_name,
        "DistantLight"
            | "SphereLight"
            | "DiskLight"
            | "CylinderLight"
            | "RectLight"
            | "DomeLight"
    )
}

/// Whether the name is a gprim with geometry of its own.
pub fn is_gprim(type_name: &str) -> bool {
    matches!(
        type_name,
        "Cube" | "Sphere" | "Cylinder" | "Cone" | "Capsule" | "Plane"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{parse::parse, scene::to_scene, scene::UsdScene};
    use crate::core::{Object3D, ObjectId, ObjectKind};

    fn stage() -> UsdScene {
        to_scene(&parse(include_str!("testdata/schemas.usda")).expect("parses"))
    }

    fn find<'a>(scene: &'a UsdScene, name: &str) -> &'a Object3D {
        fn walk<'a>(scene: &'a UsdScene, id: ObjectId, name: &str) -> Option<&'a Object3D> {
            let node = scene.arena.get(id)?;
            if node.name == name {
                return Some(node);
            }
            node.children.iter().find_map(|c| walk(scene, *c, name))
        }
        scene
            .roots
            .iter()
            .find_map(|r| walk(scene, *r, name))
            .unwrap_or_else(|| panic!("no prim called {name}"))
    }

    fn names(scene: &UsdScene) -> Vec<String> {
        fn walk(scene: &UsdScene, id: ObjectId, out: &mut Vec<String>) {
            let Some(node) = scene.arena.get(id) else { return };
            out.push(node.name.clone());
            for child in &node.children {
                walk(scene, *child, out);
            }
        }
        let mut out = Vec::new();
        for root in &scene.roots {
            walk(scene, *root, &mut out);
        }
        out
    }

    /// Every `UsdLux` light becomes the light it describes, rather than an
    /// empty group — which is what a mesh-only loader leaves behind, silently.
    #[test]
    fn lights_become_lights() {
        let scene = stage();

        let ObjectKind::Light(Light::Directional(sun)) = &find(&scene, "Sun").kind else {
            panic!("DistantLight should be directional, got {:?}", find(&scene, "Sun").kind);
        };
        // Intensity 3 at exposure 1 is three doubled.
        assert!((sun.intensity - 6.0).abs() < 1e-5, "{}", sun.intensity);
        assert!((sun.color.r - 1.0).abs() < 1e-5 && (sun.color.b - 0.8).abs() < 1e-5);

        assert!(matches!(
            find(&scene, "Bulb").kind,
            ObjectKind::Light(Light::Point(_))
        ));
        assert!(matches!(
            find(&scene, "Sky").kind,
            ObjectKind::Light(Light::Ambient(_))
        ));

        let ObjectKind::Light(Light::RectArea(panel)) = &find(&scene, "Panel").kind else {
            panic!("RectLight should be an area light");
        };
        assert_eq!((panel.width, panel.height), (4.0, 2.0));
    }

    /// A sphere light with a cone on it is a spotlight, which is how USD
    /// spells one — there is no `SpotLight` type.
    #[test]
    fn a_shaped_light_is_a_spot() {
        let scene = stage();
        let ObjectKind::Light(Light::Spot(torch)) = &find(&scene, "Torch").kind else {
            panic!("a ShapingAPI cone should make a spot, got {:?}", find(&scene, "Torch").kind);
        };
        assert!((torch.angle - 25f32.to_radians()).abs() < 1e-5, "{}", torch.angle);
        assert!((torch.penumbra - 0.4).abs() < 1e-5);
    }

    /// The quadrics carry their shape in a few numbers; a loader that ignores
    /// them drops most of a layout file.
    #[test]
    fn the_quadrics_become_geometry() {
        let scene = stage();
        for (name, expected_extent) in [
            ("Box", 3.0f32),
            ("Ball", 4.0),
            ("Tube", 4.0),
            ("Spike", 2.0),
            ("Pill", 4.0),
            ("Ground", 20.0),
        ] {
            let ObjectKind::Mesh(mesh) = &find(&scene, name).kind else {
                panic!("{name} should be a mesh, got {:?}", find(&scene, name).kind);
            };
            let position = mesh.geometry.get_attribute("position").expect("positions");
            assert!(position.count() > 3, "{name} has no vertices");

            // The largest span of the bounding box, which says the size was
            // read rather than defaulted.
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            for v in position.array.chunks_exact(3) {
                for i in 0..3 {
                    lo[i] = lo[i].min(v[i]);
                    hi[i] = hi[i].max(v[i]);
                }
            }
            let span = (0..3).map(|i| hi[i] - lo[i]).fold(0.0f32, f32::max);
            assert!(
                (span - expected_extent).abs() < 0.3,
                "{name}: span {span}, expected about {expected_extent}"
            );
        }
    }

    #[test]
    fn points_and_curves_become_their_own_node_kinds() {
        let scene = stage();

        let ObjectKind::Points(dust) = &find(&scene, "Dust").kind else {
            panic!("Points should be a point cloud");
        };
        assert_eq!(dust.geometry.get_attribute("position").unwrap().count(), 3);
        // Per-point colour came across as a vertex attribute.
        assert_eq!(dust.geometry.get_attribute("color").unwrap().count(), 3);

        let ObjectKind::LineSegments(hair) = &find(&scene, "Hair").kind else {
            panic!("BasisCurves should be line segments");
        };
        // Two curves of 3 and 2 points are 2 + 1 segments, two vertices each.
        assert_eq!(hair.geometry.get_attribute("position").unwrap().count(), 6);
    }

    /// A camera is described as a physical one; the field of view follows from
    /// the aperture and the focal length.
    #[test]
    fn cameras_come_back_with_the_stage() {
        let scene = stage();
        assert_eq!(scene.cameras.len(), 2, "both cameras");

        let eye = scene.cameras.iter().find(|c| c.name == "Eye").expect("Eye");
        let perspective = eye.perspective.as_ref().expect("perspective");
        // 24mm of film at 35mm focal length is about 37.8 degrees. The field
        // itself is radians, which is why it is converted to compare.
        assert!(
            (perspective.fov.to_degrees() - 37.85).abs() < 0.1,
            "fov was {} degrees",
            perspective.fov.to_degrees()
        );
        assert!((perspective.aspect - 1.5).abs() < 1e-5);
        assert_eq!((perspective.near, perspective.far), (0.1, 500.0));

        let ortho = scene.cameras.iter().find(|c| c.name == "Ortho").expect("Ortho");
        let orthographic = ortho.orthographic.as_ref().expect("orthographic");
        assert_eq!((orthographic.left, orthographic.right), (-50.0, 50.0));
        assert!(eye.perspective.is_some() && ortho.perspective.is_none());
    }

    /// `purpose = "guide"` is never drawn, and an invisible prim is present
    /// but not visible — two different things that are easy to conflate.
    #[test]
    fn purpose_and_visibility_are_honoured() {
        let scene = stage();
        assert!(!find(&scene, "Hidden").visible, "invisible should not be visible");

        let names = names(&scene);
        assert!(names.contains(&"Hidden".to_string()), "invisible still exists");
        assert!(
            !names.contains(&"GuideOnly".to_string()),
            "guide geometry should not be in the scene: {names:?}"
        );
    }

    /// A scatter becomes one instanced draw per prototype, not one mesh per
    /// instance — which is the difference between a forest that fits in memory
    /// and one that does not.
    #[test]
    fn a_point_instancer_becomes_instanced_draws() {
        let layer = parse(include_str!("testdata/instancer.usda")).expect("parses");
        let scene = to_scene(&layer);

        let scatter = find(&scene, "Scatter");
        assert_eq!(scatter.children.len(), 2, "one draw per prototype");

        let mut by_name: Vec<(String, usize)> = scatter
            .children
            .iter()
            .map(|c| {
                let node = scene.arena.get(*c).unwrap();
                let ObjectKind::InstancedMesh(mesh) = &node.kind else {
                    panic!("{} should be instanced, got {:?}", node.name, node.kind);
                };
                (node.name.clone(), mesh.transforms.len())
            })
            .collect();
        by_name.sort();
        // Four instances: two trees, two rocks, less the one turned off.
        assert_eq!(by_name, vec![("Rock".into(), 1), ("Tree".into(), 2)]);
    }

    /// The transforms are the ones the instancer authored — position,
    /// orientation and scale, composed.
    #[test]
    fn instance_transforms_are_composed_from_all_three_arrays() {
        let scene = to_scene(&parse(include_str!("testdata/instancer.usda")).unwrap());
        let scatter = find(&scene, "Scatter");
        let trees = scatter
            .children
            .iter()
            .map(|c| scene.arena.get(*c).unwrap())
            .find(|n| n.name == "Tree")
            .expect("the tree draw");
        let ObjectKind::InstancedMesh(mesh) = &trees.kind else {
            panic!("instanced");
        };

        // Instance 0 sits at the origin at unit scale; instance 2 is at x=10
        // and half size. The translation is the last column.
        let first = mesh.transforms[0].elements;
        assert_eq!((first[12], first[13], first[14]), (0.0, 0.0, 0.0));
        assert!((first[0] - 1.0).abs() < 1e-5, "unit scale");

        let third = mesh.transforms[1].elements;
        assert_eq!(third[12], 10.0);
        assert!((third[0] - 0.5).abs() < 1e-5, "half scale, got {}", third[0]);
    }

    /// `invisibleIds` turns instances off without rebuilding the set.
    #[test]
    fn hidden_instances_are_left_out() {
        let scene = to_scene(&parse(include_str!("testdata/instancer.usda")).unwrap());
        let total: usize = find(&scene, "Scatter")
            .children
            .iter()
            .map(|c| match &scene.arena.get(*c).unwrap().kind {
                ObjectKind::InstancedMesh(m) => m.transforms.len(),
                _ => 0,
            })
            .sum();
        assert_eq!(total, 3, "four instances less the one in invisibleIds");
    }

    /// An `instanceable` prim still composes to its referenced content: the
    /// sharing is an optimisation, not a change in what the stage contains.
    #[test]
    fn an_instanceable_reference_still_brings_its_content() {
        use super::super::compose::{compose, ComposeOptions, MemoryResolver};
        let layer = parse(include_str!("testdata/instancer.usda")).unwrap();
        let composed = compose(
            &layer,
            "instancer.usda",
            &MemoryResolver::new(),
            &ComposeOptions::default(),
        )
        .unwrap();
        assert!(
            composed.prim_at("/World/Copy/Shape").is_some(),
            "the instanced copy should carry the source's geometry"
        );
    }

    /// And all of it again from the crate form, which stores the same document
    /// through an entirely different code path.
    #[test]
    fn the_binary_form_gives_the_same_stage() {
        let text = stage();
        let binary = to_scene(
            &super::super::UsdLoader::parse_layer(include_bytes!("testdata/schemas.usdc"))
                .expect("the crate parses"),
        );
        assert_eq!(binary.cameras.len(), text.cameras.len());
        assert_eq!(names(&binary), names(&text));
        for name in ["Sun", "Bulb", "Panel", "Torch", "Box", "Ball", "Dust", "Hair"] {
            let a = format!("{:?}", std::mem::discriminant(&find(&text, name).kind));
            let b = format!("{:?}", std::mem::discriminant(&find(&binary, name).kind));
            assert_eq!(a, b, "{name} came out as a different kind of node");
        }
    }
}
