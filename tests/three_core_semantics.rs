//! three.js r165 semantics of the scene core — geometry groups with a
//! material array, `drawRange`, morph targets, `Object3D` matrices and the
//! raycaster's default layers — checked against values three.js itself
//! printed (`node` with three@0.165.0; the script is reproduced in each test's
//! comment where it matters).

use std::sync::Arc;
use threers::core::{mesh_draws, MeshDraw, ObjectKind};
use threers::{
    BasicMaterial, BufferAttribute, BufferGeometry, Euler, EulerOrder, Material, Matrix4, Mesh,
    Object3D, ObjectArena, Quaternion, Raycaster, Vector3,
};

const XS: [f32; 8] = [0.2, 0.7, 1.2, 1.7, 2.2, 2.7, 3.2, 3.7];
const YS: [f32; 2] = [0.15, 0.6];

/// Four unit quads along +x, two triangles each, facing +z — three.js:
/// `idx.push(a, c, b, b, c, d)` over the strip `(i, 0), (i, 1)`.
fn strip(indexed: bool) -> BufferGeometry {
    let mut pos = Vec::new();
    for i in 0..=4 {
        pos.extend_from_slice(&[i as f32, 0.0, 0.0, i as f32, 1.0, 0.0]);
    }
    let mut idx = Vec::new();
    for i in 0..4u32 {
        let (a, b, c, d) = (2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3);
        idx.extend_from_slice(&[a, c, b, b, c, d]);
    }
    let mut g = BufferGeometry::new();
    if indexed {
        g.set_attribute("position", BufferAttribute::new(pos, 3));
        g.set_index(idx);
    } else {
        let soup: Vec<f32> = idx
            .iter()
            .flat_map(|&v| pos[v as usize * 3..v as usize * 3 + 3].to_vec())
            .collect();
        g.set_attribute("position", BufferAttribute::new(soup, 3));
    }
    g
}

fn basic(side: u32) -> Arc<Material> {
    Arc::new(Material::Basic(BasicMaterial {
        side,
        ..Default::default()
    }))
}

/// `(faceIndex, face.materialIndex)` per hit, for every probe point.
fn probe(mesh: Mesh) -> Vec<Vec<(usize, usize)>> {
    let mut arena = ObjectArena::new();
    let id = arena.insert(Object3D::mesh(mesh));
    arena.update_world_matrices(id, Matrix4::identity());
    let mut out = Vec::new();
    for x in XS {
        for y in YS {
            let rc = Raycaster::new(Vector3::new(x, y, 5.0), Vector3::new(0.0, 0.0, -1.0), 0.0, f32::INFINITY);
            out.push(
                rc.intersect_object(&arena, id, false)
                    .iter()
                    .map(|h| (h.face_index, h.face.unwrap().material_index))
                    .collect(),
            );
        }
    }
    out
}

fn parse(table: &[&[(usize, usize)]]) -> Vec<Vec<(usize, usize)>> {
    table.iter().map(|r| r.to_vec()).collect()
}

/// three.js, for both the indexed strip and `toNonIndexed()`:
///
/// ```js
/// g.addGroup(0, 6, 1); g.addGroup(6, 12, 0); g.addGroup(3, 6, 2);
/// g.setDrawRange(3, 12);
/// const mesh = new Mesh(g, [basic, basic, basic({ side: DoubleSide })]);
/// ```
///
/// gives `[[],[],[],[[1,1],[1,2]],[[2,0],[2,2]],[[2,0],[2,2]],[[2,0],[2,2]],
/// [[3,0]],[[4,0]],[[4,0]],[[4,0]],[],[],[],[],[]]` — the overlapping group
/// reports its triangles a second time, and the draw range cuts both ends.
#[test]
fn groups_and_draw_range_raycast_as_three() {
    let want = parse(&[
        &[],
        &[],
        &[],
        &[(1, 1), (1, 2)],
        &[(2, 0), (2, 2)],
        &[(2, 0), (2, 2)],
        &[(2, 0), (2, 2)],
        &[(3, 0)],
        &[(4, 0)],
        &[(4, 0)],
        &[(4, 0)],
        &[],
        &[],
        &[],
        &[],
        &[],
    ]);
    for indexed in [true, false] {
        let mut g = strip(indexed);
        g.add_group(0, 6, 1).add_group(6, 12, 0).add_group(3, 6, 2);
        g.set_draw_range(3, 12);
        let mesh = Mesh::with_materials(Arc::new(g), vec![basic(0), basic(0), basic(2)]);
        assert_eq!(probe(mesh), want, "indexed {indexed}");
    }
}

/// The same geometry with one material: groups ignored, the draw range
/// alone. three.js: `[[],[],[],[[1,0]],[[2,0]],[[2,0]],[[2,0]],[[3,0]],
/// [[4,0]],[[4,0]],[[4,0]],[],[],[],[],[]]`.
#[test]
fn draw_range_alone_raycasts_as_three() {
    let want = parse(&[
        &[],
        &[],
        &[],
        &[(1, 0)],
        &[(2, 0)],
        &[(2, 0)],
        &[(2, 0)],
        &[(3, 0)],
        &[(4, 0)],
        &[(4, 0)],
        &[(4, 0)],
        &[],
        &[],
        &[],
        &[],
        &[],
    ]);
    for indexed in [true, false] {
        let mut g = strip(indexed);
        g.add_group(0, 6, 1).add_group(6, 12, 0).add_group(3, 6, 2);
        g.set_draw_range(3, 12);
        let mesh = Mesh::from_arc(Arc::new(g), basic(0));
        assert_eq!(probe(mesh), want, "indexed {indexed}");
    }
}

/// Draws are exactly the spans the raycaster walks: the draw range cut to
/// each group, empty ones gone, a group with no material skipped.
#[test]
fn mesh_draws_follow_render_buffer_direct() {
    let mut g = strip(true);
    g.add_group(0, 6, 1).add_group(6, 12, 0).add_group(3, 6, 2).add_group(18, 6, 7);
    g.set_draw_range(3, 12);
    let mats = vec![basic(0), basic(0), basic(2)];
    let single = basic(0);
    let spans = |d: Vec<MeshDraw>| -> Vec<(u32, u32, u32)> {
        d.iter().map(|d| (d.start, d.end, d.material_index)).collect()
    };
    assert_eq!(
        spans(mesh_draws(&g, &single, &mats)),
        vec![(3, 6, 1), (6, 15, 0), (3, 9, 2)]
    );
    assert_eq!(spans(mesh_draws(&g, &single, &[])), vec![(3, 15, 0)]);
    // No groups with a material array: nothing drawn, as in three.js.
    g.clear_groups();
    assert!(mesh_draws(&g, &single, &mats).is_empty());
    // The default range is everything.
    let full = strip(false);
    assert_eq!(spans(mesh_draws(&full, &single, &[])), vec![(0, 24, 0)]);
}

fn morph_plane(relative: bool) -> Mesh {
    // `new PlaneGeometry(2, 2, 1, 1)`: corners (-1, 1), (1, 1), (-1, -1), (1, -1).
    let base = [-1.0f32, 1.0, 0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, -1.0, 0.0];
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(base.to_vec(), 3));
    g.set_index(vec![0, 2, 1, 2, 3, 1]);
    let (mut t0, mut t1) = (Vec::new(), Vec::new());
    for p in base.chunks_exact(3) {
        if relative {
            t0.extend_from_slice(&[0.0, 0.0, 1.0]);
            t1.extend_from_slice(&[p[0] * 0.5, 0.0, 0.0]);
        } else {
            t0.extend_from_slice(&[p[0], p[1], p[2] + 1.0]);
            t1.extend_from_slice(&[p[0] * 1.5, p[1], p[2]]);
        }
    }
    g.morph_targets_relative = relative;
    g.set_morph_attribute(
        "position",
        vec![BufferAttribute::new(t0, 3), BufferAttribute::new(t1, 3)],
    );
    let mut mesh = Mesh::new(g, Material::Basic(BasicMaterial::default()));
    assert_eq!(mesh.morph_influences, vec![0.0, 0.0], "updateMorphTargets");
    assert_eq!(mesh.morph_target_dictionary.get("1"), Some(&1));
    mesh.morph_influences = vec![0.5, 0.25];
    mesh
}

/// three.js, both ways round (absolute targets and `morphTargetsRelative`
/// built to describe the same shapes): a ray down from (1.1, 0.3, 5) hits at
/// distance 4.5, point (1.1, 0.3, 0.5), face 1 — a point the rest pose does
/// not even cover (x = 1.1 is past its edge).
#[test]
fn morphed_meshes_raycast_where_three_puts_them() {
    for relative in [false, true] {
        let mut arena = ObjectArena::new();
        let id = arena.insert(Object3D::mesh(morph_plane(relative)));
        arena.update_world_matrices(id, Matrix4::identity());
        let rc = Raycaster::new(Vector3::new(1.1, 0.3, 5.0), Vector3::new(0.0, 0.0, -1.0), 0.0, f32::INFINITY);
        let hits = rc.intersect_object(&arena, id, false);
        assert_eq!(hits.len(), 1, "relative {relative}");
        let h = &hits[0];
        assert!((h.distance - 4.5).abs() < 1e-5, "{}", h.distance);
        assert!((h.point - Vector3::new(1.1, 0.3, 0.5)).length() < 1e-5);
        assert_eq!(h.face_index, 1);
    }
}

/// The CPU pose the path tracer and SVG renderer draw is the pose the
/// raycaster hits.
#[test]
fn posed_geometry_is_what_the_raycaster_hits() {
    let mesh = morph_plane(false);
    let posed = mesh.posed_geometry();
    assert!(posed.morph_attributes.is_empty(), "never blended twice");
    let p = &posed.get_attribute("position").unwrap().array;
    // Corner (1, 1): 0.25·(1,1,0) + 0.5·(1,1,1) + 0.25·(1.5,1,0).
    assert!((p[3] - 1.125).abs() < 1e-6 && (p[4] - 1.0).abs() < 1e-6 && (p[5] - 0.5).abs() < 1e-6);
    // Nothing to blend: the geometry itself.
    let mut still = morph_plane(false);
    still.morph_influences = vec![0.0, 0.0];
    assert!(Arc::ptr_eq(&still.posed_geometry(), &still.geometry));
}

/// three.js: a parent whose `matrix` is a shear (`matrixAutoUpdate = false`)
/// keeps it, and a child at (1, 2, 3) lands at
/// `[1,0,0,0, 0.5,1,0,0, 0,0,1,0, 4,2,3,1]`.
#[test]
fn a_non_trs_matrix_survives_the_world_update() {
    let mut arena = ObjectArena::new();
    let parent = arena.insert(Object3D::group());
    let mut shear = Matrix4::identity();
    shear.elements[4] = 0.5; // row 0, column 1
    shear.elements[12] = 2.0;
    arena.get_mut(parent).unwrap().set_matrix(shear);
    let mut c = Object3D::group();
    c.position = Vector3::new(1.0, 2.0, 3.0);
    let child = arena.insert(c);
    arena.add_child(parent, child);
    arena.update_world_matrices(parent, Matrix4::identity());
    let want = [1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0, 2.0, 3.0, 1.0];
    assert_eq!(arena.get(child).unwrap().matrix_world.elements, want);
    assert_eq!(arena.get(parent).unwrap().matrix.elements, shear.elements, "kept as written");
}

/// three.js:
///
/// ```js
/// o.position.set(1, 2, 3); o.rotation.set(0.1, 0.2, 0.3, 'ZYX'); o.scale.set(2, 1, 0.5);
/// o.applyMatrix4(new Matrix4().makeRotationY(0.7).setPosition(4, 0, 0));
/// ```
///
/// position (6.697495, 2, 1.650309), quaternion (0.081424, 0.436780,
/// 0.123116, 0.887376), scale (2, 1, 0.5).
#[test]
fn apply_matrix4_and_euler_orders_as_three() {
    let mut o = Object3D::group();
    o.position = Vector3::new(1.0, 2.0, 3.0);
    o.set_rotation(Euler::with_order(0.1, 0.2, 0.3, EulerOrder::ZYX));
    o.scale = Vector3::new(2.0, 1.0, 0.5);
    let mut m = Matrix4::from_quaternion(Quaternion::from_axis_angle(Vector3::UP, 0.7));
    m.elements[12] = 4.0;
    o.apply_matrix4(&m);
    let close = |a: f32, b: f64| (a as f64 - b).abs() < 2e-5;
    assert!(close(o.position.x, 6.697495248997562), "{:?}", o.position);
    assert!(close(o.position.y, 2.0) && close(o.position.z, 1.6503088746157744));
    let q = o.quaternion;
    assert!(close(q.x, 0.08142363703431536) && close(q.y, 0.4367804573531127));
    assert!(close(q.z, 0.12311640186217204) && close(q.w, 0.8873755546548608), "{q:?}");
    assert!(close(o.scale.x, 2.0) && close(o.scale.y, 1.0) && close(o.scale.z, 0.5));
    // The Euler view reads back in the object's order.
    assert_eq!(o.rotation_order, EulerOrder::ZYX);
}

/// three.js, a mesh and a directional light both at (0, 1, 2) under a parent
/// at (1, 0, 0) rotated (0.3, 0.5, -0.2), both `lookAt(3, -1, 4)`: the mesh
/// turns +Z to the target, quaternion (0.150602, -0.056512, 0.052465,
/// 0.985582); the light -Z, (-0.052465, 0.985582, 0.150602, 0.056512).
#[test]
fn look_at_turns_lights_and_meshes_as_three() {
    let mut arena = ObjectArena::new();
    let mut p = Object3D::group();
    p.position = Vector3::new(1.0, 0.0, 0.0);
    p.set_rotation(Euler::new(0.3, 0.5, -0.2));
    let parent = arena.insert(p);
    let mut m = Object3D::mesh(Mesh::new(BufferGeometry::new(), Material::Basic(BasicMaterial::default())));
    m.position = Vector3::new(0.0, 1.0, 2.0);
    let mesh = arena.insert(m);
    let mut l = Object3D::light(threers::lights::DirectionalLight::new(threers::Color::WHITE, 1.0));
    l.position = Vector3::new(0.0, 1.0, 2.0);
    let light = arena.insert(l);
    arena.add_child(parent, mesh);
    arena.add_child(parent, light);
    let target = Vector3::new(3.0, -1.0, 4.0);
    arena.look_at(mesh, target);
    arena.look_at(light, target);
    let close = |q: Quaternion, w: [f64; 4]| {
        // q and -q are one rotation.
        let d = |s: f64| {
            (q.x as f64 - s * w[0]).abs()
                + (q.y as f64 - s * w[1]).abs()
                + (q.z as f64 - s * w[2]).abs()
                + (q.w as f64 - s * w[3]).abs()
        };
        d(1.0).min(d(-1.0)) < 1e-5
    };
    let qm = arena.get(mesh).unwrap().quaternion;
    let ql = arena.get(light).unwrap().quaternion;
    assert!(
        close(qm, [0.1506021608553702, -0.05651240413073437, 0.05246456135049703, 0.9855824709923353]),
        "{qm:?}"
    );
    assert!(
        close(ql, [-0.052464561350497034, 0.9855824709923353, 0.15060216085537026, 0.05651240413073437]),
        "{ql:?}"
    );
}

/// three.js's renderer skips the world update when the *scene's*
/// `matrixWorldAutoUpdate` is off.
#[test]
fn scene_matrix_world_auto_update_off_freezes_the_frame() {
    let mut scene = threers::Scene::new();
    let mut o = Object3D::group();
    o.position = Vector3::new(5.0, 0.0, 0.0);
    let id = scene.add(o);
    scene.update_world();
    scene.get_mut(id).unwrap().position = Vector3::new(7.0, 0.0, 0.0);
    let root = scene.root;
    scene.arena.get_mut(root).unwrap().matrix_world_auto_update = false;
    scene.update_world();
    assert_eq!(scene.get(id).unwrap().matrix_world.elements[12], 5.0);
    scene.arena.get_mut(root).unwrap().matrix_world_auto_update = true;
    scene.update_world();
    assert_eq!(scene.get(id).unwrap().matrix_world.elements[12], 7.0);
}

/// three.js `new Raycaster().layers` is channel 0 only.
#[test]
fn raycaster_defaults_to_layer_zero() {
    let rc = Raycaster::default();
    assert_eq!(rc.layers.mask, 1);
    let mut arena = ObjectArena::new();
    let mut g = strip(true);
    g.set_draw_range(0, u32::MAX);
    let mut o = Object3D::mesh(Mesh::new(g, Material::Basic(BasicMaterial::default())));
    o.layers.set(3);
    let id = arena.insert(o);
    arena.update_world_matrices(id, Matrix4::identity());
    let mut rc = Raycaster::new(Vector3::new(0.2, 0.15, 5.0), Vector3::new(0.0, 0.0, -1.0), 0.0, f32::INFINITY);
    assert!(rc.intersect_object(&arena, id, false).is_empty(), "layer 3 is not seen");
    rc.layers.enable(3);
    assert_eq!(rc.intersect_object(&arena, id, false).len(), 1);
    if let ObjectKind::Mesh(m) = &arena.get(id).unwrap().kind {
        assert!(!m.is_multi_material());
    }
}
