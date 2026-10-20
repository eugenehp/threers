//! Geometry groups with a material array, `drawRange` and morph targets on the
//! wgpu backend, checked on real pixels against the CPU raycaster (which
//! follows three.js's `Mesh.raycast`, see `tests/three_core_semantics.rs`)
//! and against the CPU pose (`Mesh::posed_geometry`). Also: a skinned caster's
//! shadow follows its pose. Skipped when no GPU adapter is available.

use std::sync::Arc;
use threers::cameras::{Camera, OrthographicCamera};
use threers::core::{
    Bone, BufferAttribute, BufferGeometry, Mesh, Object3D, ObjectId, ObjectKind, Raycaster,
    Skeleton, SkinnedMesh,
};
use threers::geometries::PlaneGeometry;
use threers::lights::{AmbientLight, DirectionalLight};
use threers::materials::{BasicMaterial, LambertMaterial, Material};
use threers::math::{Color, Matrix4, Vector2, Vector3};
use threers::scene::Scene;
use threers::HeadlessRenderer;

const SIZE: u32 = 64;

fn renderer() -> Option<HeadlessRenderer> {
    let mut r = HeadlessRenderer::builder()
        .size(SIZE, SIZE)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()?;
    r.set_msaa(1);
    Some(r)
}

/// Looks down -Z at the unit square.
fn camera() -> OrthographicCamera {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 2.0);
    cam.target = Vector3::ZERO;
    cam
}

fn px(rgba: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
}

fn basic(hex: u32) -> Arc<Material> {
    Arc::new(BasicMaterial::new(Color::from_hex(hex)).into())
}

/// The first hit through pixel `(x, y)`'s centre: `(object, face.material_index)`.
fn pick(scene: &mut Scene, cam: &dyn Camera, x: u32, y: u32) -> Option<(ObjectId, usize)> {
    scene
        .arena
        .update_world_matrices(scene.root, Matrix4::identity());
    let ndc = Vector2::new(
        (x as f32 + 0.5) / SIZE as f32 * 2.0 - 1.0,
        1.0 - (y as f32 + 0.5) / SIZE as f32 * 2.0,
    );
    let mut rc = Raycaster::default();
    rc.set_from_camera(ndc, cam);
    rc.intersect_object(&scene.arena, scene.root, true)
        .first()
        .map(|h| (h.object, h.face.map(|f| f.material_index).unwrap_or(0)))
}

/// Pixels whose 3x3 neighbourhood is one colour — away from every edge, so
/// rasterisation and the ray agree without arguing about a boundary.
fn interior(rgba: &[u8]) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for y in 1..SIZE - 1 {
        for x in 1..SIZE - 1 {
            let c = px(rgba, x, y);
            if (0..3).all(|dy| (0..3).all(|dx| px(rgba, x + dx - 1, y + dy - 1) == c)) {
                out.push((x, y));
            }
        }
    }
    out
}

/// A strip of four quads across x ∈ [-0.8, 0.8], y ∈ [-0.5, 0.5]; quad `i`
/// is index entries `6i..6i+6`.
fn strip() -> BufferGeometry {
    PlaneGeometry::with_segments(1.6, 1.0, 4, 1)
}

/// Each group drawn with its own material, the draw range cutting the last
/// quad away, and every interior pixel the colour of the material the
/// raycaster names for it (background where it hits nothing).
#[test]
fn groups_draw_their_materials_where_the_raycaster_names_them() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut g = strip();
    // three.js's PlaneGeometry index runs row by row; with one row of four
    // quads, quad i is entries 6i..6i+6.
    g.add_group(0, 6, 2).add_group(6, 6, 0).add_group(12, 12, 1);
    g.set_draw_range(0, 18);
    let palette = [0xff0000u32, 0x00ff00, 0x0000ff];
    let mesh = Mesh::with_materials(Arc::new(g), palette.iter().map(|&c| basic(c)).collect());
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    let id = scene.add(Object3D::mesh(mesh));
    let cam = camera();
    let rgba = r.render_to_rgba(&mut scene, &cam);
    let (mut checked, mut colours) = (0, [0usize; 4]);
    for (x, y) in interior(&rgba) {
        let c = px(&rgba, x, y);
        let expect = match pick(&mut scene, &cam, x, y) {
            Some((o, mi)) if o == id => {
                let hex = palette[mi];
                [(hex >> 16) as u8, (hex >> 8) as u8, hex as u8]
            }
            _ => [0, 0, 0],
        };
        assert_eq!(&c[..3], &expect, "pixel ({x}, {y})");
        checked += 1;
        colours[c.iter().take(3).position(|&v| v == 255).map_or(3, |k| k)] += 1;
    }
    eprintln!("groups: {checked} pixels checked, red/green/blue/none {colours:?}");
    // Every material shows, and so does the cut quad's background.
    assert!(colours.iter().all(|&n| n > 20), "{colours:?}");
}

/// three.js sorts render items per group: a transparent group blends over
/// what is behind it while the opaque group of the same mesh does not.
#[test]
fn transparency_is_per_group() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut g = strip();
    g.add_group(0, 12, 0).add_group(12, 12, 1);
    let half = BasicMaterial {
        transparent: true,
        opacity: 0.5,
        ..BasicMaterial::new(Color::from_hex(0x0000ff))
    };
    let mesh = Mesh::with_materials(Arc::new(g), vec![basic(0x0000ff), Arc::new(half.into())]);
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::mesh(mesh));
    let rgba = r.render_to_rgba(&mut scene, &camera());
    // Left half (quads 0-1) opaque blue, right half blended at 0.5.
    let left = px(&rgba, 16, 32);
    let right = px(&rgba, 48, 32);
    assert_eq!(left[2], 255, "opaque group: {left:?}");
    assert!(right[2] > 100 && right[2] < 220, "transparent group blends: {right:?}");
}

fn morph_plane(relative: bool, influences: [f32; 2]) -> Mesh {
    let mut g = PlaneGeometry::with_segments(1.0, 1.0, 4, 4);
    let base = g.get_attribute("position").unwrap().array.clone();
    let count = base.len() / 3;
    let (mut t0, mut t1) = (Vec::new(), Vec::new());
    let mut n0 = Vec::new();
    let mut c0 = Vec::new();
    for v in 0..count {
        let (x, y, z) = (base[v * 3], base[v * 3 + 1], base[v * 3 + 2]);
        // Target 0 bulges the plane towards the camera and widens it; target
        // 1 shears it. Normal and colour targets ride along on target 0.
        let a = [x * 1.5, y + 0.2 * x, z + 0.3];
        let b = [x + 0.3 * y, y * 0.8, z];
        if relative {
            t0.extend_from_slice(&[a[0] - x, a[1] - y, a[2] - z]);
            t1.extend_from_slice(&[b[0] - x, b[1] - y, b[2] - z]);
            n0.extend_from_slice(&[0.6, 0.0, -0.2]);
            c0.extend_from_slice(&[0.0, -0.5, 0.5]);
        } else {
            t0.extend_from_slice(&a);
            t1.extend_from_slice(&b);
            n0.extend_from_slice(&[0.6, 0.0, 0.8]);
            c0.extend_from_slice(&[1.0, 0.5, 0.5]);
        }
    }
    g.set_attribute("color", BufferAttribute::new(vec![1.0; count * 3], 3));
    g.morph_targets_relative = relative;
    let n1 = if relative { vec![0.0; count * 3] } else { g.get_attribute("normal").unwrap().array.clone() };
    let c1 = if relative { vec![0.0; count * 3] } else { vec![1.0; count * 3] };
    g.morph_attributes.insert(
        "normal".into(),
        vec![BufferAttribute::new(n0, 3), BufferAttribute::new(n1, 3)],
    );
    g.morph_attributes.insert(
        "color".into(),
        vec![BufferAttribute::new(c0, 3), BufferAttribute::new(c1, 3)],
    );
    g.set_morph_attribute(
        "position",
        vec![BufferAttribute::new(t0, 3), BufferAttribute::new(t1, 3)],
    );
    // Vertex colours multiply in whenever the geometry has them.
    let mut mesh = Mesh::new(g, LambertMaterial::default().into());
    mesh.morph_influences = influences.to_vec();
    mesh
}

fn lit_scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    let mut l = Object3D::light(DirectionalLight::new(Color::WHITE, 2.0));
    l.position = Vector3::new(1.0, 0.5, 2.0);
    scene.add(l);
    scene.add(Object3D::light(AmbientLight::new(Color::WHITE, 0.2)));
    scene
}

/// The GPU morph — positions, normals and colours from the morph texture —
/// draws what the CPU pose (`Mesh::posed_geometry`, three.js's formulas)
/// draws as a plain mesh, for absolute and relative targets alike; and the
/// silhouette is where the raycaster hits.
#[test]
fn gpu_morph_matches_the_cpu_pose_and_the_raycaster() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let cam = camera();
    for relative in [false, true] {
        let mesh = morph_plane(relative, [0.6, 0.3]);
        let posed = Mesh::from_arc(mesh.posed_geometry(), mesh.material.clone());
        assert!(posed.morph_influences.is_empty());

        let mut gpu_scene = lit_scene();
        let id = gpu_scene.add(Object3D::mesh(mesh));
        let gpu = r.render_to_rgba(&mut gpu_scene, &cam);
        let mut cpu_scene = lit_scene();
        cpu_scene.add(Object3D::mesh(posed));
        let cpu = r.render_to_rgba(&mut cpu_scene, &cam);

        let (mut worst, mut lit) = (0i32, 0);
        for i in (0..gpu.len()).step_by(4) {
            for k in 0..3 {
                worst = worst.max((gpu[i + k] as i32 - cpu[i + k] as i32).abs());
            }
            lit += (gpu[i..i + 3].iter().any(|&v| v > 8)) as usize;
        }
        eprintln!("morph relative={relative}: {lit} lit, worst channel difference {worst}");
        // The rest plane covers 32x32 = 1024 pixels; the blend widens it.
        assert!(lit > 1150, "the morphed plane is larger than the rest one");
        assert!(worst <= 3, "GPU morph differs from the CPU pose by {worst}");

        let (mut agree, mut checked) = (0, 0);
        for (x, y) in interior(&gpu) {
            let here = px(&gpu, x, y)[..3].iter().any(|&v| v > 8);
            let hit = pick(&mut gpu_scene, &cam, x, y).is_some_and(|(o, _)| o == id);
            checked += 1;
            agree += (here == hit) as usize;
        }
        assert_eq!(agree, checked, "relative={relative}: picks disagree with pixels");
    }
}

/// A skinned mesh with morph targets: the targets apply first, then the
/// bones, as three.js chains `morphtarget_vertex` into `skinning_vertex`.
#[test]
fn skinned_morph_composes_morph_then_skin() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let cam = camera();
    let mesh = morph_plane(false, [0.8, 0.0]);
    let mut g = (*mesh.geometry).clone();
    let count = g.get_attribute("position").unwrap().count();
    let mut joints = vec![0.0f32; count * 4];
    let mut weights = vec![0.0f32; count * 4];
    let ys: Vec<f32> = g.get_attribute("position").unwrap().array.chunks(3).map(|p| p[1]).collect();
    for v in 0..count {
        joints[v * 4 + 1] = 1.0;
        let t = (ys[v] + 0.5).clamp(0.0, 1.0);
        weights[v * 4] = 1.0 - t;
        weights[v * 4 + 1] = t;
    }
    g.set_attribute("joint", BufferAttribute::new(joints, 4));
    g.set_attribute("weight", BufferAttribute::new(weights, 4));
    let mut skeleton = Skeleton::new(vec![
        Bone {
            node: ObjectId::default(),
            inverse_bind: Matrix4::identity(),
        },
        Bone {
            node: ObjectId::default(),
            inverse_bind: Matrix4::identity(),
        },
    ]);
    skeleton.bone_matrices = vec![
        Matrix4::translation(Vector3::new(-0.2, 0.0, 0.0)),
        Matrix4::from_quaternion(threers::math::Euler::new(0.0, 0.0, 0.5).to_quaternion()),
    ];
    let mut sm = SkinnedMesh::new(g, (*mesh.material).clone(), skeleton);
    assert_eq!(sm.morph_influences, vec![0.0, 0.0], "sized from the geometry");
    sm.morph_influences = vec![0.8, 0.0];

    let mut scene = lit_scene();
    let id = scene.add(Object3D::skinned_mesh(sm));
    scene.update_world();
    let world = scene.get(id).unwrap().matrix_world;
    let posed = match &scene.get(id).unwrap().kind {
        ObjectKind::SkinnedMesh(sm) => {
            let p = sm.posed_positions(&world).unwrap();
            let mut g = (*sm.geometry).clone();
            g.morph_attributes.clear();
            g.set_attribute("position", BufferAttribute::new(p, 3));
            g
        }
        _ => unreachable!(),
    };
    let gpu = r.render_to_rgba(&mut scene, &cam);
    let mut flat = Scene::new();
    flat.background = Color::from_hex(0x000000);
    flat.add(Object3D::mesh(Mesh::new(posed, BasicMaterial::new(Color::WHITE).into())));
    let cpu = r.render_to_rgba(&mut flat, &cam);
    let cover = |img: &[u8], x: u32, y: u32| px(img, x, y)[..3].iter().any(|&v| v > 8);
    let (mut differ, mut lit) = (0, 0);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let (a, b) = (cover(&gpu, x, y), cover(&cpu, x, y));
            lit += a as usize;
            differ += (a != b) as usize;
        }
    }
    eprintln!("skinned morph: {lit} lit, {differ} differ from the CPU pose");
    assert!(lit > 600);
    assert!(differ * 50 <= lit, "{differ} of {lit} pixels differ");
}

/// A skinned caster throws the shadow of its pose: bones move the mesh off
/// its bind position, and the floor darkens under the pose, not under the
/// bind position.
#[test]
fn skinned_casters_shadow_their_pose() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Looking straight down at a floor; the light straight above.
    let mut cam = OrthographicCamera::new(-2.0, 2.0, 2.0, -2.0, 0.1, 20.0);
    cam.position = Vector3::new(0.0, 5.0, 0.0);
    cam.target = Vector3::ZERO;
    cam.up = Vector3::new(0.0, 0.0, -1.0);
    let build = |pose_x: f32, skinned: bool| {
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        let mut floor_mat = threers::materials::StandardMaterial::new(Color::WHITE);
        floor_mat.metalness = 0.0;
        floor_mat.roughness = 1.0;
        let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(4.0, 4.0), floor_mat.into()));
        floor.quaternion = threers::math::Euler::new(-std::f32::consts::FRAC_PI_2, 0.0, 0.0).to_quaternion();
        floor.receive_shadow = true;
        scene.add(floor);
        let mut sun = DirectionalLight::new(Color::WHITE, 1.0);
        sun.cast_shadow = true;
        let mut l = Object3D::light(sun);
        l.position = Vector3::new(0.0, 10.0, 0.0);
        scene.add(l);
        // A small flat box at y = 1, bound to one bone that moves it `pose_x`
        // along x. A box rather than a plane: the shadow pass culls front
        // faces, as three.js's does by default, so a one-sided square facing
        // the light would cast nothing.
        let mut g = threers::geometries::BoxGeometry::new(0.8, 0.1, 0.8);
        let count = g.get_attribute("position").unwrap().count();
        let mut w = vec![0.0f32; count * 4];
        for v in 0..count {
            w[v * 4] = 1.0;
        }
        g.set_attribute("joint", BufferAttribute::new(vec![0.0; count * 4], 4));
        g.set_attribute("weight", BufferAttribute::new(w, 4));
        let mut skeleton = Skeleton::new(vec![Bone {
            node: ObjectId::default(),
            inverse_bind: Matrix4::identity(),
        }]);
        skeleton.bone_matrices = vec![Matrix4::translation(Vector3::new(pose_x, 1.0, 0.0))];
        let black: Material = BasicMaterial::new(Color::from_hex(0x000000)).into();
        let mut caster = if skinned {
            Object3D::skinned_mesh(SkinnedMesh::new(g, black, skeleton))
        } else {
            // The same square placed by its own transform: the control.
            let mut o = Object3D::mesh(Mesh::new(g, black));
            o.set_matrix(skeleton.bone_matrices[0]);
            o
        };
        caster.cast_shadow = true;
        scene.add(caster);
        scene
    };
    let rgba = r.render_to_rgba(&mut build(1.0, true), &cam);
    // World x maps to image x; the caster itself is black, so look at the
    // floor around it: under the pose (x = 1) is covered by the caster from
    // above, so probe the floor where the bind pose would shadow (x = 0)
    // against a far patch.
    let at = |wx: f32, wz: f32| {
        let x = ((wx + 2.0) / 4.0 * SIZE as f32) as u32;
        let y = ((wz + 2.0) / 4.0 * SIZE as f32) as u32;
        px(&rgba, x.min(SIZE - 1), y.min(SIZE - 1))[0]
    };
    let bind = at(0.0, 0.0);
    let far = at(-1.4, 1.4);
    eprintln!("floor under the bind pose {bind}, far floor {far}");
    assert!(far > 40, "the floor is lit");
    assert!(
        (bind as i32 - far as i32).abs() <= 4,
        "no shadow where the bind pose would be: {bind} vs {far}"
    );
    // With the light tilted the pose's shadow falls beside it, on open floor
    // — for the skinned caster as for the same square placed by its own
    // transform (the control, which says the probe can see a shadow).
    for skinned in [false, true] {
        let mut scene = build(1.0, skinned);
        for (_, o) in scene.arena.nodes.iter_mut() {
            if matches!(o.kind, ObjectKind::Light(_)) {
                o.position = Vector3::new(-10.0, 10.0, 0.0);
            }
        }
        let tilted = r.render_to_rgba(&mut scene, &cam);
        // Light from -x at 45°: the square at height 1 shadows x ∈ [1.6, 2.4]
        // roughly; x = 1.9 is open floor, beside the caster.
        let x = ((1.9f32 + 2.0) / 4.0 * SIZE as f32) as u32;
        let y = (2.0 / 4.0 * SIZE as f32) as u32;
        let shadowed = px(&tilted, x.min(SIZE - 1), y)[0];
        let open = px(&tilted, ((-1.4f32 + 2.0) / 4.0 * SIZE as f32) as u32, y)[0];
        eprintln!("tilted, skinned={skinned}: under the shadow {shadowed}, open floor {open}");
        assert!(
            (shadowed as i32) + 15 < open as i32,
            "skinned={skinned}: the pose casts its shadow: {shadowed} vs {open}"
        );
    }
}
