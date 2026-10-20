//! `InstancedMesh` with material arrays and morph targets, and morphed vertex
//! colours, held to three.js r165 on real pixels.
//!
//! The truth for an instanced draw is three.js's own definition of one: an
//! `InstancedMesh` draws, per instance, exactly the `Mesh` with the instance
//! matrix as its transform, the same material (array) and the instance's
//! morph influences (`morphinstance_vertex.glsl` reads them in place of the
//! uniform ones). So every scenario is built twice — instanced, and as one
//! plain `Mesh` per instance — and the instanced frame's interior pixels must
//! be the colour of the material the raycaster names on the plain scene
//! (three.js's `Mesh.raycast`: groups, draw range, `getVertexPosition` with
//! morphs). The raycaster on the instanced scene must name the same
//! instance and `materialIndex` where nothing is morphed (three.js's
//! `InstancedMesh.raycast` tests each instance through a `Mesh` that carries
//! no `morphTargetInfluences`).
//!
//! Shared by `tests/instanced_groups_morph.rs` (wgpu) and
//! `tests/metal_backend.rs` through `#[path]`.
#![allow(dead_code)]

use std::sync::Arc;
use threers::cameras::{Camera, OrthographicCamera};
use threers::core::{
    BufferAttribute, BufferGeometry, InstancedMesh, Mesh, Object3D, ObjectId, Raycaster,
};
use threers::geometries::PlaneGeometry;
use threers::materials::{BasicMaterial, Material};
use threers::math::{Color, Matrix4, Vector2, Vector3};
use threers::scene::Scene;

pub const N: u32 = 96;
pub const PALETTE: [u32; 3] = [0xff0000, 0x00ff00, 0x0000ff];

pub fn camera() -> OrthographicCamera {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 2.0);
    cam.target = Vector3::ZERO;
    cam
}

fn basic(hex: u32) -> Arc<Material> {
    Arc::new(BasicMaterial::new(Color::from_hex(hex)).into())
}

/// A strip of four quads, x ∈ [-0.5, 0.5], y ∈ [-0.15, 0.15], grouped
/// 2 / 0 / 1 / 1 (quad 3 cut by the draw range), with two absolute position
/// targets: 0 doubles the height, 1 shifts up by 0.12.
fn strip(groups: bool) -> BufferGeometry {
    let mut g = PlaneGeometry::with_segments(1.0, 0.3, 4, 1);
    if groups {
        g.add_group(0, 6, 2).add_group(6, 6, 0).add_group(12, 12, 1);
        g.set_draw_range(0, 18);
    }
    let base = g.get_attribute("position").unwrap().array.clone();
    let (mut t0, mut t1) = (Vec::new(), Vec::new());
    for p in base.chunks_exact(3) {
        t0.extend_from_slice(&[p[0], p[1] * 2.0, p[2]]);
        t1.extend_from_slice(&[p[0], p[1] + 0.12, p[2]]);
    }
    g.set_morph_attribute(
        "position",
        vec![BufferAttribute::new(t0, 3), BufferAttribute::new(t1, 3)],
    );
    g
}

/// One scenario: geometry, materials (empty = one red material), instance
/// matrices, and the influences each instance is drawn with — `Shared` the
/// mesh-level `morphTargetInfluences`, `PerInstance` rows set with
/// `set_morph_at`.
pub struct Case {
    pub name: &'static str,
    groups: bool,
    matrices: Vec<Matrix4>,
    influences: Influences,
}

enum Influences {
    Shared(Vec<f32>),
    PerInstance(Vec<Vec<f32>>),
}

pub fn cases() -> Vec<Case> {
    let rot = |a: f32| {
        Matrix4::from_quaternion(threers::math::Quaternion::from_axis_angle(
            Vector3::new(0.0, 0.0, 1.0),
            a,
        ))
    };
    let at = |x: f32, y: f32| Matrix4::translation(Vector3::new(x, y, 0.0));
    vec![
        Case {
            name: "groups",
            groups: true,
            matrices: vec![at(0.0, 0.55), at(0.0, -0.1).multiply(&rot(0.35)), at(0.0, -0.65)],
            influences: Influences::Shared(vec![0.0, 0.0]),
        },
        Case {
            name: "shared morph",
            groups: false,
            matrices: vec![at(0.0, 0.5), at(0.0, -0.4)],
            influences: Influences::Shared(vec![0.6, 0.5]),
        },
        Case {
            name: "per-instance morph",
            groups: false,
            matrices: vec![at(0.0, 0.6), at(0.0, 0.0), at(0.0, -0.6)],
            influences: Influences::PerInstance(vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.5, 1.0],
            ]),
        },
        Case {
            name: "per-instance morph + groups",
            groups: true,
            matrices: vec![at(0.0, 0.55), at(0.0, -0.05).multiply(&rot(-0.3)), at(0.0, -0.6)],
            influences: Influences::PerInstance(vec![
                vec![0.9, 0.0],
                vec![0.0, 1.0],
                vec![0.4, 0.4],
            ]),
        },
    ]
}

fn materials(case: &Case) -> Vec<Arc<Material>> {
    if case.groups {
        PALETTE.iter().map(|&c| basic(c)).collect()
    } else {
        Vec::new()
    }
}

/// The instanced scene.
pub fn instanced_scene(case: &Case) -> (Scene, ObjectId) {
    let g = Arc::new(strip(case.groups));
    let mats = materials(case);
    let mut im = if mats.is_empty() {
        InstancedMesh::from_arc(g, basic(PALETTE[0]), case.matrices.len())
    } else {
        InstancedMesh::with_materials(g, mats, case.matrices.len())
    };
    for (i, m) in case.matrices.iter().enumerate() {
        im.set_matrix_at(i, *m);
    }
    match &case.influences {
        Influences::Shared(w) => im.morph_influences = w.clone(),
        Influences::PerInstance(rows) => {
            for (i, w) in rows.iter().enumerate() {
                im.set_morph_at(i, w);
            }
        }
    }
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    let id = scene.add(Object3D::instanced_mesh(im));
    (scene, id)
}

/// The same draws as plain meshes, one per instance.
pub fn reference_scene(case: &Case) -> (Scene, Vec<ObjectId>) {
    let g = Arc::new(strip(case.groups));
    let mats = materials(case);
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    let mut ids = Vec::new();
    for (i, m) in case.matrices.iter().enumerate() {
        let mut mesh = if mats.is_empty() {
            Mesh::from_arc(g.clone(), basic(PALETTE[0]))
        } else {
            Mesh::with_materials(g.clone(), mats.clone())
        };
        mesh.morph_influences = match &case.influences {
            Influences::Shared(w) => w.clone(),
            Influences::PerInstance(rows) => rows[i].clone(),
        };
        let mut o = Object3D::mesh(mesh);
        o.set_matrix(*m);
        ids.push(scene.add(o));
    }
    (scene, ids)
}

fn pick(scene: &mut Scene, cam: &dyn Camera, x: u32, y: u32) -> Option<(ObjectId, Option<usize>, usize)> {
    let ndc = Vector2::new(
        (x as f32 + 0.5) / N as f32 * 2.0 - 1.0,
        1.0 - (y as f32 + 0.5) / N as f32 * 2.0,
    );
    let mut rc = Raycaster::default();
    rc.set_from_camera(ndc, cam);
    rc.intersect_object(&scene.arena, scene.root, true)
        .first()
        .map(|h| (h.object, h.instance_id, h.face.map(|f| f.material_index).unwrap_or(0)))
}

fn px(rgba: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * N + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2]]
}

/// Run every case through `render`; returns the failures.
pub fn check(
    backend: &str,
    render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>,
) -> Vec<String> {
    let cam = camera();
    let mut failures = Vec::new();
    for case in cases() {
        let (mut inst, inst_id) = instanced_scene(&case);
        let (mut reference, ids) = reference_scene(&case);
        reference
            .arena
            .update_world_matrices(reference.root, Matrix4::identity());
        inst.arena.update_world_matrices(inst.root, Matrix4::identity());
        let rgba = render(&mut inst, &cam);
        let morphed = !matches!(&case.influences, Influences::Shared(w) if w.iter().all(|&x| x == 0.0));
        let (mut checked, mut wrong, mut per_colour) = (0, Vec::new(), [0usize; 4]);
        for y in 1..N - 1 {
            for x in 1..N - 1 {
                let c = px(&rgba, x, y);
                if !(0..3).all(|dy| (0..3).all(|dx| px(&rgba, x + dx - 1, y + dy - 1) == c)) {
                    continue;
                }
                let want_hit = pick(&mut reference, &cam, x, y);
                let want = match want_hit {
                    Some((_, _, mi)) => {
                        let hex = if case.groups { PALETTE[mi] } else { PALETTE[0] };
                        [(hex >> 16) as u8, (hex >> 8) as u8, hex as u8]
                    }
                    None => [0, 0, 0],
                };
                checked += 1;
                per_colour[c.iter().position(|&v| v == 255).unwrap_or(3)] += 1;
                if c != want {
                    wrong.push((x, y, c, want));
                }
                // The raycaster on the instanced mesh: the instance and the
                // group's material, as on the plain meshes — where nothing is
                // morphed (three.js's InstancedMesh.raycast ignores morphs).
                if !morphed {
                    let got = pick(&mut inst, &cam, x, y);
                    let expect = want_hit.map(|(o, _, mi)| {
                        (inst_id, ids.iter().position(|&i| i == o), mi)
                    });
                    if got != expect {
                        failures.push(format!(
                            "{backend} {}: raycast at ({x}, {y}) {got:?}, plain meshes {expect:?}",
                            case.name
                        ));
                    }
                }
            }
        }
        eprintln!(
            "{backend} {:>28}: {checked} px, r/g/b/none {per_colour:?}, {} wrong",
            case.name,
            wrong.len()
        );
        if !wrong.is_empty() {
            failures.push(format!(
                "{backend} {}: {} of {checked} pixels differ from three.js, e.g. {:?}",
                case.name,
                wrong.len(),
                &wrong[..wrong.len().min(4)]
            ));
        }
        let drawn = per_colour[..3].iter().sum::<usize>();
        if drawn < 200 {
            failures.push(format!("{backend} {}: only {drawn} pixels drawn", case.name));
        }
        if case.groups && per_colour[..3].iter().any(|&n| n < 30) {
            failures.push(format!("{backend} {}: a material is missing: {per_colour:?}", case.name));
        }
    }
    failures
}

/// three.js's `morphcolor_vertex.glsl` with a 4-component colour
/// (`USE_COLOR_ALPHA`): the alpha morphs with the rgb. A quad coloured
/// opaque red with an absolute colour target of transparent blue:
///
/// - influence 0.5 on an opaque material: rgb (0.5, 0, 0.5) → bytes
///   (188, 0, 188);
/// - influence 1 on a transparent material: alpha 0, the background shows.
///
/// Returns the failures.
pub fn check_colour_morph(
    backend: &str,
    render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>,
) -> Vec<String> {
    let cam = camera();
    let mut failures = Vec::new();
    for (influence, transparent, want) in [(0.5f32, false, [188u8, 0, 188]), (1.0, true, [0, 77, 0])] {
        let mut g = PlaneGeometry::new(1.2, 1.2);
        let n = g.get_attribute("position").unwrap().count();
        g.set_attribute("color", BufferAttribute::new([1.0, 0.0, 0.0, 1.0].repeat(n), 4));
        g.set_morph_attribute(
            "color",
            vec![BufferAttribute::new([0.0, 0.0, 1.0, 0.0].repeat(n), 4)],
        );
        let mut m = BasicMaterial::new(Color::WHITE);
        m.transparent = transparent;
        let mut mesh = Mesh::new(g, m.into());
        mesh.morph_influences = vec![influence];
        let mut scene = Scene::new();
        // Dark green: distinct from both the red and the blue.
        scene.background = Color::from_hex(0x004d00);
        scene.add(Object3D::mesh(mesh));
        let rgba = render(&mut scene, &cam);
        let got = px(&rgba, N / 2, N / 2);
        let err = (0..3).map(|k| (got[k] as i32 - want[k] as i32).abs()).max().unwrap();
        eprintln!("{backend} colour morph {influence}: got {got:?} want {want:?}");
        if err > 2 {
            failures.push(format!(
                "{backend}: rgba colour morph at {influence}: got {got:?}, three.js {want:?}"
            ));
        }
    }
    failures
}

/// three.js `InstancedMesh.setColorAt`: `color_vertex.glsl` multiplies the
/// instance's colour into `vColor` (`vColor.xyz *= instanceColor.xyz`, after
/// any `color` attribute), and `color_fragment` multiplies `vColor` into the
/// diffuse colour — for every material. Instances past the end of `colors`
/// are white (three.js fills `instanceColor` with 1). Three quads in a row,
/// with and without a vertex-colour attribute; each quad's centre must be
/// `encode(material · vertex · instance)`. Returns the failures.
pub fn check_instance_colours(
    backend: &str,
    render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>,
) -> Vec<String> {
    let cam = camera();
    let encode = |x: f32| -> u8 {
        let x = x.clamp(0.0, 1.0);
        let s = if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
        (s * 255.0).round() as u8
    };
    let material = Color::new(0.9, 0.7, 0.5);
    let instance = [Color::new(0.5, 1.0, 0.25), Color::new(1.0, 0.1, 0.8)];
    let xs = [-0.6f32, 0.0, 0.6];
    let mut failures = Vec::new();
    for vertex in [None, Some([0.5f32, 0.8, 1.0])] {
        let mut g = PlaneGeometry::new(0.4, 0.4);
        if let Some(v) = vertex {
            let n = g.get_attribute("position").unwrap().count();
            g.set_attribute("color", BufferAttribute::new([v[0], v[1], v[2]].repeat(n), 3));
        }
        let mut im = InstancedMesh::from_arc(
            Arc::new(g),
            Arc::new(BasicMaterial::new(material).into()),
            xs.len(),
        );
        for (i, &x) in xs.iter().enumerate() {
            im.set_matrix_at(i, Matrix4::translation(Vector3::new(x, 0.0, 0.0)));
        }
        // Two colours for three instances: the third stays white.
        im.colors = instance.to_vec();
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        scene.add(Object3D::instanced_mesh(im));
        let rgba = render(&mut scene, &cam);
        for (i, &x) in xs.iter().enumerate() {
            let tint = instance.get(i).copied().unwrap_or(Color::WHITE);
            let v = vertex.unwrap_or([1.0; 3]);
            let want = [
                encode(material.r * v[0] * tint.r),
                encode(material.g * v[1] * tint.g),
                encode(material.b * v[2] * tint.b),
            ];
            let px_x = ((x + 1.0) / 2.0 * N as f32) as u32;
            let got = px(&rgba, px_x, N / 2);
            let err = (0..3).map(|k| (got[k] as i32 - want[k] as i32).abs()).max().unwrap();
            eprintln!("{backend} instance colour {i} (vertex colours {vertex:?}): got {got:?} want {want:?}");
            if err > 1 {
                failures.push(format!(
                    "{backend}: instance {i} (vertex colours {vertex:?}) is {got:?}, three.js {want:?}"
                ));
            }
        }
    }
    failures
}
