// Shared lighting, paper placement, and crease overlay for origami examples.
// Included with `include!("origami_common.inc");`
//
// This is a toolbox, textually included into each example's crate root, so any
// one example uses a handful of these and not the rest. That is the point of it
// being shared — the unused ones are not dead, they belong to a different bird.

use threers::{
    encode_png, AssembledBird, Assignment, Color, CreasePattern, DirectionalLight, EdgeKind,
    Euler, FoldedState, HeadlessRenderer, HemisphereLight, LineBasicMaterial, LineSegments, Mesh,
    Object3D, ObjectId, PerspectiveCamera, PlaneGeometry, Scene, ShadowSettings, StandardMaterial,
    Vector3,
};

#[allow(dead_code)]
fn origami_lights(scene: &mut Scene) {
    // Cool matte backdrop so red/green/cream paper separates cleanly.
    scene.background = Color::new(0.14, 0.15, 0.17);
    scene.add_light(threers::AmbientLight::new(Color::new(0.92, 0.94, 1.0), 0.12));
    scene.add_light(HemisphereLight::new(
        Color::new(0.85, 0.88, 0.95),
        Color::new(0.32, 0.28, 0.24),
        0.35,
    ));

    // Key light — high, warm, casts contact shadows onto the ground + paper.
    let mut key = DirectionalLight::new(Color::new(1.0, 0.97, 0.92), 3.2);
    key.cast_shadow = true;
    key.shadow = ShadowSettings {
        map_size: 4096,
        bias: 0.004,
        normal_bias: 0.06,
        camera_size: 28.0,
        camera_near: 0.5,
        camera_far: 80.0,
    };
    let mut key_obj = Object3D::light(key);
    key_obj.position = Vector3::new(6.5, 14.0, 7.0);
    scene.add(key_obj);

    // Cool rim / fill from the opposite side (no shadow — keeps folds readable).
    scene.add_light(
        DirectionalLight::new(Color::new(0.45, 0.55, 0.85), 0.55)
            .with_direction(Vector3::new(0.55, -0.2, -0.7).normalize()),
    );
    // Soft bounce from below-front so undersides aren't crushed.
    scene.add_light(
        DirectionalLight::new(Color::new(0.75, 0.7, 0.65), 0.28)
            .with_direction(Vector3::new(-0.15, 0.85, -0.35).normalize()),
    );
}

#[allow(dead_code)]
fn origami_ground(scene: &mut Scene, y: f32) {
    let mut mat = StandardMaterial::new(Color::new(0.22, 0.23, 0.25));
    mat.roughness = 0.95;
    mat.metalness = 0.0;
    let mut ground = Object3D::mesh(Mesh::new(PlaneGeometry::new(90.0, 90.0), mat.into()));
    ground.receive_shadow = true;
    ground.cast_shadow = false;
    ground.quaternion = Euler::new(-std::f32::consts::FRAC_PI_2, 0.0, 0.0).to_quaternion();
    ground.position = Vector3::new(0.0, y, 0.0);
    scene.add(ground);
}

#[allow(dead_code)]
fn paper_material(color: Color) -> StandardMaterial {
    let mut mat = StandardMaterial::new(color);
    mat.roughness = 0.78;
    mat.metalness = 0.0;
    // DoubleSide so mixed ring winding still fills panels.
    mat.side = 2;
    mat
}

#[allow(dead_code)]
fn bbox(folded: &FoldedState) -> ([f64; 3], [f64; 3]) {
    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];
    for v in &folded.verts {
        for k in 0..3 {
            min[k] = min[k].min(v[k]);
            max[k] = max[k].max(v[k]);
        }
    }
    (min, max)
}

#[allow(dead_code)]
fn prepare_folded(folded: &FoldedState) -> FoldedState {
    folded.clone()
}

#[allow(dead_code)]
fn add_mesh_shadow(scene: &mut Scene, parent: ObjectId, mesh: Mesh) -> ObjectId {
    let mut obj = Object3D::mesh(mesh);
    // Cast onto the ground, but do not receive — flat coplanar folds acne hard.
    obj.cast_shadow = true;
    obj.receive_shadow = false;
    scene.add_to(parent, obj)
}

#[allow(dead_code)]
fn add_crease_overlay(scene: &mut Scene, parent: ObjectId, folded: &FoldedState, segs: &[[usize; 2]], color: Color, width: f32) {
    if segs.is_empty() {
        return;
    }
    let mut mat = LineBasicMaterial::new(color);
    mat.line_width = width;
    scene.add_to(
        parent,
        Object3D::line_segments(LineSegments::new(folded.line_geometry(segs), mat.into())),
    );
}

#[allow(dead_code)]
fn hinge_creases(cp: &CreasePattern, tangents: &[f64]) -> (Vec<[usize; 2]>, Vec<[usize; 2]>) {
    let mut valley = Vec::new();
    let mut mountain = Vec::new();
    for (ei, e) in cp.edges.iter().enumerate() {
        if e.kind != EdgeKind::Hinge {
            continue;
        }
        let t = tangents.get(ei).copied().unwrap_or(0.0);
        if t > 1e-8 {
            valley.push([e.a, e.b]);
        } else if t < -1e-8 {
            mountain.push([e.a, e.b]);
        }
    }
    (valley, mountain)
}

#[allow(dead_code)]
fn add_folded(
    scene: &mut Scene,
    folded: &FoldedState,
    cp: &CreasePattern,
    tangents: &[f64],
    at: Vector3,
    scale: f32,
    euler: Euler,
) -> ObjectId {
    let folded = prepare_folded(folded);
    let (min, max) = bbox(&folded);
    let mut root = Object3D::group();
    root.position = at;
    root.scale = Vector3::new(scale, scale, scale);
    root.quaternion = euler.to_quaternion();
    let root_id = scene.add(root);

    let mut inner = Object3D::group();
    inner.position = Vector3::new(
        -0.5 * (min[0] + max[0]) as f32,
        -0.5 * (min[1] + max[1]) as f32,
        -0.5 * (min[2] + max[2]) as f32,
    );
    let inner_id = scene.add_to(root_id, inner);

    let (even, odd) = folded.to_geometry_checker();
    if even.get_attribute("position").is_some() {
        add_mesh_shadow(
            scene,
            inner_id,
            Mesh::new(even, paper_material(Color::new(0.94, 0.90, 0.82)).into()),
        );
    }
    if odd.get_attribute("position").is_some() {
        add_mesh_shadow(
            scene,
            inner_id,
            Mesh::new(odd, paper_material(Color::new(0.62, 0.74, 0.70)).into()),
        );
    }

    let (valley, mountain) = hinge_creases(cp, tangents);
    add_crease_overlay(
        scene,
        inner_id,
        &folded,
        &valley,
        Color::new(0.72, 0.18, 0.14),
        2.4,
    );
    add_crease_overlay(
        scene,
        inner_id,
        &folded,
        &mountain,
        Color::new(0.15, 0.32, 0.68),
        2.2,
    );
    root_id
}

/// Checkerboard panels tinted for a species (cardinal = red breast, etc.).
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn add_folded_bird(
    scene: &mut Scene,
    folded: &FoldedState,
    cp: &CreasePattern,
    tangents: &[f64],
    at: Vector3,
    scale: f32,
    euler: Euler,
    plumage: (Color, Color),
) -> ObjectId {
    let folded = prepare_folded(folded);
    let (min, max) = bbox(&folded);
    let mut root = Object3D::group();
    root.position = at;
    root.scale = Vector3::new(scale, scale, scale);
    root.quaternion = euler.to_quaternion();
    let root_id = scene.add(root);

    let mut inner = Object3D::group();
    inner.position = Vector3::new(
        -0.5 * (min[0] + max[0]) as f32,
        -0.5 * (min[1] + max[1]) as f32,
        -0.5 * (min[2] + max[2]) as f32,
    );
    let inner_id = scene.add_to(root_id, inner);
    attach_folded_part(scene, inner_id, &folded, plumage);

    let (valley, mountain) = hinge_creases(cp, tangents);
    add_crease_overlay(
        scene,
        inner_id,
        &folded,
        &valley,
        Color::new(0.55, 0.08, 0.06),
        2.2,
    );
    let mut mt_mat = LineBasicMaterial::new(Color::new(0.12, 0.28, 0.58));
    mt_mat.line_width = 2.0;
    mt_mat.dashed = true;
    if !mountain.is_empty() {
        scene.add_to(
            inner_id,
            Object3D::line_segments(LineSegments::new(
                folded.line_geometry(&mountain),
                mt_mat.into(),
            )),
        );
    }
    root_id
}

/// Folded mesh with crease edges — for hand-posed traditional models.
#[allow(dead_code)]
fn add_folded_shape(
    scene: &mut Scene,
    folded: &FoldedState,
    at: Vector3,
    scale: f32,
    euler: Euler,
    plumage: (Color, Color),
) -> ObjectId {
    let folded = prepare_folded(folded);
    let (min, max) = bbox(&folded);
    let mut root = Object3D::group();
    root.position = at;
    root.scale = Vector3::new(scale, scale, scale);
    root.quaternion = euler.to_quaternion();
    let root_id = scene.add(root);

    let mut inner = Object3D::group();
    // Centre the mesh, then raise so the lowest vertex clears the ground plane.
    let half_h = 0.5 * (max[1] - min[1]).max(0.15) as f32;
    inner.position = Vector3::new(
        -0.5 * (min[0] + max[0]) as f32,
        -0.5 * (min[1] + max[1]) as f32 + half_h + 0.04,
        -0.5 * (min[2] + max[2]) as f32,
    );
    let inner_id = scene.add_to(root_id, inner);
    attach_folded_part(scene, inner_id, &folded, plumage);
    root_id
}

#[allow(dead_code)]
fn attach_folded_part(
    scene: &mut Scene,
    parent_id: ObjectId,
    folded: &FoldedState,
    plumage: (Color, Color),
) {
    let (even, odd) = folded.to_geometry_checker();
    let even_ok = even.get_attribute("position").is_some();
    let odd_ok = odd.get_attribute("position").is_some();
    if even_ok {
        add_mesh_shadow(
            scene,
            parent_id,
            Mesh::new(even, paper_material(plumage.0).into()),
        );
    }
    if odd_ok {
        add_mesh_shadow(
            scene,
            parent_id,
            Mesh::new(odd, paper_material(plumage.1).into()),
        );
    }
    if !even_ok && !odd_ok {
        add_mesh_shadow(
            scene,
            parent_id,
            Mesh::new(folded.to_geometry(), paper_material(plumage.0).into()),
        );
    }
}

#[allow(dead_code)]
fn attach_folded_part_with_creases(
    scene: &mut Scene,
    parent_id: ObjectId,
    folded: &FoldedState,
    cp: &CreasePattern,
    tangents: &[f64],
    plumage: (Color, Color),
) {
    let folded = prepare_folded(folded);
    let (min, max) = bbox(&folded);
    let mut inner = Object3D::group();
    inner.position = Vector3::new(
        -0.5 * (min[0] + max[0]) as f32,
        -0.5 * (min[1] + max[1]) as f32,
        -0.5 * (min[2] + max[2]) as f32,
    );
    let inner_id = scene.add_to(parent_id, inner);
    attach_folded_part(scene, inner_id, &folded, plumage);

    let (valley, mountain) = hinge_creases(cp, tangents);
    add_crease_overlay(
        scene,
        inner_id,
        &folded,
        &valley,
        Color::new(0.55, 0.08, 0.06),
        2.2,
    );
    let mut mt = LineBasicMaterial::new(Color::new(0.12, 0.28, 0.58));
    mt.line_width = 2.0;
    mt.dashed = true;
    if !mountain.is_empty() {
        scene.add_to(
            inner_id,
            Object3D::line_segments(LineSegments::new(
                folded.line_geometry(&mountain),
                mt.into(),
            )),
        );
    }
}

/// Rigid-folded bird built from body, crest, wings, and tail parts.
#[allow(dead_code)]
fn add_assembled_bird(
    scene: &mut Scene,
    bird: &AssembledBird,
    at: Vector3,
    scale: f32,
    view: Euler,
    plumage: (Color, Color),
) -> ObjectId {
    let mut root = Object3D::group();
    root.position = at;
    root.quaternion = view.to_quaternion();
    root.scale = Vector3::new(scale, scale, scale);
    let root_id = scene.add(root);

    for (part, pose) in &bird.parts {
        let mut grp = Object3D::group();
        grp.position = Vector3::new(pose.position[0], pose.position[1], pose.position[2]);
        grp.quaternion =
            Euler::new(pose.euler[0], pose.euler[1], pose.euler[2]).to_quaternion();
        grp.scale = Vector3::new(pose.scale, pose.scale, pose.scale);
        let grp_id = scene.add_to(root_id, grp);
        attach_folded_part_with_creases(
            scene,
            grp_id,
            &part.folded,
            &part.cp,
            &part.tangents,
            plumage,
        );
    }
    root_id
}

#[allow(dead_code)]
fn bird_plumage(name: &str) -> (Color, Color) {
    match name {
        "cardinal" => (
            Color::new(0.86, 0.14, 0.12),
            Color::new(0.22, 0.12, 0.12),
        ),
        "crane" => (
            Color::new(0.96, 0.95, 0.92),
            Color::new(0.72, 0.70, 0.66),
        ),
        "owl" => (
            Color::new(0.68, 0.52, 0.34),
            Color::new(0.38, 0.28, 0.18),
        ),
        "sparrow" => (
            Color::new(0.58, 0.44, 0.28),
            Color::new(0.78, 0.70, 0.48),
        ),
        "flamingo" => (
            Color::new(0.96, 0.48, 0.58),
            Color::new(0.82, 0.28, 0.42),
        ),
        _ => (
            Color::new(0.94, 0.90, 0.82),
            Color::new(0.62, 0.74, 0.70),
        ),
    }
}

#[allow(dead_code)]
fn write_next_to(png: &str, name: &str, svg: &str) {
    let path = std::path::Path::new(png);
    let dir = path.parent().unwrap_or_else(|| std::path::Path::new("out"));
    let _ = std::fs::create_dir_all(dir);
    let p = dir.join(name);
    std::fs::write(&p, svg).expect("write svg");
    println!("wrote {}", p.display());
}

#[allow(dead_code)]
fn out_arg(default: &str) -> String {
    std::env::args()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .unwrap_or_else(|| default.into())
}

#[allow(dead_code)]
fn fold_or_die(cp: &CreasePattern, t: f64, what: &str) -> (Assignment, Vec<f64>, FoldedState) {
    cp.fold_any(t)
        .unwrap_or_else(|| panic!("{what}: no rigid folding from the flat state"))
}

#[allow(dead_code)]
fn render_scene(out: &str, w: u32, h: u32, cam: [f32; 3], look: Vector3, build: impl FnOnce(&mut Scene)) {
    let mut renderer = match HeadlessRenderer::builder()
        .size(w, h)
        .supersample(2)
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e})");
            std::process::exit(2);
        }
    };
    let (rw, rh) = renderer.render_size();
    let mut scene = Scene::new();
    origami_lights(&mut scene);
    // Ground under the lifted models (add_folded_shape raises by ~0.55·scale).
    origami_ground(&mut scene, 0.0);
    build(&mut scene);
    let mut camera = PerspectiveCamera::new(32.0, w as f32 / h as f32, 0.1, 120.0);
    camera.position = Vector3::new(cam[0], cam[1], cam[2]);
    camera.look_at(look);
    let rgba = renderer.render_to_rgba(&mut scene, &camera);
    if let Some(dir) = std::path::Path::new(out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(out, encode_png(rw, rh, &rgba)).expect("write png");
    println!("wrote {out} ({rw}x{rh})");
}
