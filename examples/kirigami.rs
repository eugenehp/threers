//! Kirigami Expanded Miura — gallery of folded states, plus crease-pattern nets.
//!
//! ```text
//! cargo run --release --example kirigami [-- out/kirigami.png]
//! cargo run --release --example kirigami_orbit   # interactive window
//! ```
//!
//! Also writes `out/kirigami-net.svg` and per-preset nets beside the PNG.

use threers::{
    encode_png, AmbientLight, Color, DirectionalLight, Euler, HeadlessRenderer, KirigamiMesh,
    KirigamiNet, KirigamiPreset, LineBasicMaterial, LineSegments, Mesh, Object3D,
    PerspectiveCamera, Scene, SphereGeometry, StandardMaterial, Vector3,
};

const COLUMNS: usize = 5;
const FOV: f32 = 24.0;
const THICK: f64 = 1.2;
const PITCH: f32 = 220.0;

fn main() {
    let out = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .unwrap_or_else(|| "out/kirigami.png".into());
    let out_dir = std::path::Path::new(&out)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("out"));
    let _ = std::fs::create_dir_all(out_dir);

    let mut meshes: Vec<(String, KirigamiMesh, KirigamiPreset)> = Vec::new();
    for p in KirigamiPreset::all() {
        let mesh = p.evaluate(p.default_nx(), p.default_ny(), THICK);
        println!(
            "{:<20} {}×{}  {:>3} plates  {:>2} cells",
            p.name(),
            mesh.nx,
            mesh.ny,
            mesh.faces.len(),
            mesh.discrete_cells().len()
        );
        let slug: String = p
            .name()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect();
        let _ = std::fs::write(
            out_dir.join(format!("kirigami-{slug}.svg")),
            mesh.develop().to_svg(),
        );
        meshes.push((p.name().to_string(), mesh, p));
    }

    let net = KirigamiPreset::Planar
        .evaluate(4, 6, 0.0)
        .develop_joined();
    std::fs::write(out_dir.join("kirigami-net.svg"), net.to_svg()).expect("svg");
    println!("wrote {}", out_dir.join("kirigami-net.svg").display());

    let count = meshes.len() + 1;
    let rows = count.div_ceil(COLUMNS);

    let (w, h) = (3000u32, 2400u32);
    let mut renderer = match HeadlessRenderer::builder()
        .size(w, h)
        .supersample(2)
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — SVG nets still written.");
            std::process::exit(2);
        }
    };
    let (rw, rh) = renderer.render_size();

    let mut scene = Scene::new();
    scene.background = Color::new(0.06, 0.07, 0.09);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.38));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 2.5)
            .with_direction(Vector3::new(-0.4, -0.72, -0.45).normalize()),
    );
    scene.add_light(
        DirectionalLight::new(Color::new(0.55, 0.65, 0.95), 0.85)
            .with_direction(Vector3::new(0.55, 0.15, 0.75).normalize()),
    );

    for (i, (_name, mesh, preset)) in meshes.into_iter().enumerate() {
        add_corrugation(&mut scene, &mesh, preset, tile_origin(i, rows));
    }
    add_net(&mut scene, &net, tile_origin(count - 1, rows));

    let aspect = w as f32 / h as f32;
    let half = (FOV * 0.5).to_radians().tan();
    let content = (COLUMNS as f32 * PITCH, rows as f32 * PITCH);
    let distance = (content.1 / (2.0 * half)).max(content.0 / (2.0 * half * aspect)) * 1.15;

    let mut camera = PerspectiveCamera::new(FOV, aspect, 1.0, distance * 4.0);
    camera.position = Vector3::new(0.0, 20.0, distance);
    camera.look_at(Vector3::ZERO);

    let rgba = renderer.render_to_rgba(&mut scene, &camera);
    std::fs::write(&out, encode_png(rw, rh, &rgba)).expect("write png");
    println!("wrote {out} ({rw}x{rh})");
}

fn tile_origin(i: usize, rows: usize) -> Vector3 {
    let col = i % COLUMNS;
    let row = i / COLUMNS;
    Vector3::new(
        (col as f32 - (COLUMNS - 1) as f32 * 0.5) * PITCH,
        ((rows - 1) as f32 * 0.5 - row as f32) * PITCH,
        0.0,
    )
}

fn add_corrugation(scene: &mut Scene, mesh: &KirigamiMesh, preset: KirigamiPreset, origin: Vector3) {
    let mut root = Object3D::group();
    root.position = origin;
    root.quaternion = Euler::new(-0.52, 0.62, 0.12).to_quaternion();
    let root_id = scene.add(root);
    let center = center_of(mesh);

    if preset.uses_cell_colours() {
        let mut mat = StandardMaterial::new(Color::WHITE);
        mat.metalness = 0.22;
        mat.roughness = 0.48;
        mat.side = 2;
        let mut plate = Object3D::mesh(Mesh::new(mesh.to_geometry_cells(), mat.into()));
        plate.position = center;
        scene.add_to(root_id, plate);

        let edge_mat = LineBasicMaterial::new(Color::new(0.12, 0.13, 0.16));
        let mut edges =
            Object3D::line_segments(LineSegments::new(mesh.crease_geometry(), edge_mat.into()));
        edges.position = center;
        scene.add_to(root_id, edges);

        let mut rivet_mat = StandardMaterial::new(Color::new(0.85, 0.78, 0.35));
        rivet_mat.metalness = 0.7;
        rivet_mat.roughness = 0.28;
        let ball = SphereGeometry::new(1.6, 10, 8);
        for p in mesh.rivet_points(1) {
            let mut s = Object3D::mesh(Mesh::new(ball.clone(), rivet_mat.clone().into()));
            s.position = Vector3::new(p[0] as f32, p[1] as f32, p[2] as f32) + center;
            scene.add_to(root_id, s);
        }
        return;
    }

    if preset.uses_lattice_core() {
        let core = preset.core_lattice().unwrap_or(threers::KirigamiCoreLattice::Gyroid);
        // Gallery packs every tile into one scene — keep cores light so the
        // combined vertex buffer stays under wgpu's 256 MiB limit.
        let geom = mesh.to_geometry_with_core(core, 0.12);
        let mut mat = StandardMaterial::new(Color::new(0.72, 0.78, 0.82));
        mat.metalness = 0.32;
        mat.roughness = 0.44;
        mat.side = 2;
        let mut plate = Object3D::mesh(Mesh::new(geom, mat.into()));
        plate.position = center;
        scene.add_to(root_id, plate);
        let edge_mat = LineBasicMaterial::new(Color::new(0.12, 0.13, 0.16));
        let mut edges =
            Object3D::line_segments(LineSegments::new(mesh.crease_geometry(), edge_mat.into()));
        edges.position = center;
        scene.add_to(root_id, edges);
        return;
    }

    let (bottom, top, inclined) = mesh.to_geometry_by_kind();
    let mut add_plates = |geom, color: Color, metal: f32, rough: f32| {
        let mut mat = StandardMaterial::new(color);
        mat.metalness = metal;
        mat.roughness = rough;
        mat.side = 2;
        let mut plate = Object3D::mesh(Mesh::new(geom, mat.into()));
        plate.position = center;
        scene.add_to(root_id, plate);
    };
    add_plates(bottom, Color::new(0.82, 0.78, 0.70), 0.35, 0.45);
    add_plates(top, Color::new(0.75, 0.82, 0.88), 0.40, 0.40);
    add_plates(inclined, Color::new(0.55, 0.72, 0.58), 0.20, 0.55);
}

fn add_net(scene: &mut Scene, net: &KirigamiNet, origin: Vector3) {
    let mut root = Object3D::group();
    root.position = origin;
    root.quaternion = Euler::new(-1.05, 0.15, 0.0).to_quaternion();
    let root_id = scene.add(root);

    let geom = net.to_geometry(0.8);
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    if let Some(pos) = geom.attributes.get("position") {
        for chunk in pos.array.chunks_exact(3) {
            for k in 0..3 {
                min[k] = min[k].min(chunk[k]);
                max[k] = max[k].max(chunk[k]);
            }
        }
    }
    let center = Vector3::new(
        -0.5 * (min[0] + max[0]),
        -0.5 * (min[1] + max[1]),
        -0.5 * (min[2] + max[2]),
    );

    let mut mat = StandardMaterial::new(Color::new(0.78, 0.82, 0.76));
    mat.metalness = 0.15;
    mat.roughness = 0.5;
    mat.side = 2;
    let mut plate = Object3D::mesh(Mesh::new(geom, mat.into()));
    plate.position = center;
    scene.add_to(root_id, plate);

    let edge_mat = LineBasicMaterial::new(Color::new(0.15, 0.16, 0.18));
    let mut edges =
        Object3D::line_segments(LineSegments::new(net.crease_geometry(), edge_mat.into()));
    edges.position = center;
    scene.add_to(root_id, edges);
}

fn center_of(mesh: &KirigamiMesh) -> Vector3 {
    let (min, max) = mesh.aabb();
    Vector3::new(
        -0.5 * (min[0] + max[0]) as f32,
        -0.5 * (min[1] + max[1]) as f32,
        -0.5 * (min[2] + max[2]) as f32,
    )
}
