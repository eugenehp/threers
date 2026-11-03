//! Face-connected cuboct voxels (Jenett et al., Sci. Adv. 2020).
//!
//! Four part types at the same mass, plus a chiral column programmed with
//! rule 1 so neighbouring faces do not cancel.
//!
//! Run: `cargo run --release --example cuboct [-- out/cuboct.png]`

use std::time::Instant;

use threers::{
    encode_png, AmbientLight, ChiralRule, Color, Cuboct, CuboctAssembly, DirectionalLight, Euler,
    HeadlessRenderer, Lattice, LatticeKind, Mesh, Object3D, PerspectiveCamera, Scene,
    StandardMaterial, Vector3,
};

const PART: f32 = 20.0;
const DENSITY: f32 = 0.18;
const RESOLUTION: usize = 22;
const WALL_SAMPLES: f32 = 2.5;
const BUDGET: usize = 12_000_000;
const COLUMNS: usize = 3;
const FOV: f32 = 32.0;

fn main() {
    let out = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .unwrap_or_else(|| "out/cuboct.png".into());

    println!(
        "{:<24} {:>9} {:>9} {:>6} {:>5} {:>9} {:>10}",
        "cell", "thickness", "density", "across", "grid", "triangles", "build"
    );

    let meshes = vec![
        tile("cuboct rigid", LatticeKind::Cuboct(Cuboct::Rigid), |l| l),
        tile(
            "cuboct compliant",
            LatticeKind::Cuboct(Cuboct::Compliant),
            |l| l.shape(0.15),
        ),
        tile(
            "cuboct auxetic",
            LatticeKind::Cuboct(Cuboct::Auxetic),
            |l| l.shape(0.20),
        ),
        tile(
            "cuboct chiral CCW",
            LatticeKind::Cuboct(Cuboct::ChiralCcw),
            |l| l.shape(0.12),
        ),
        tile(
            "cuboct chiral CW",
            LatticeKind::Cuboct(Cuboct::ChiralCw),
            |l| l.shape(0.12),
        ),
        tile(
            "chiral column R1",
            LatticeKind::Cuboct(Cuboct::ChiralCcw),
            |l| {
                let m = 3i32;
                l.shape(0.12)
                    .chiral_rule(ChiralRule::R1)
                    .program(move |_, _, k| Cuboct::column_half(k, m))
            },
        ),
        assembly_tile("exploded voxel", Cuboct::Rigid, [1, 1, 1], 0.35),
        assembly_tile("exploded 2×2×2", Cuboct::Compliant, [2, 2, 2], 0.28),
        assembly_tile("exploded auxetic", Cuboct::Auxetic, [1, 1, 1], 0.35),
        assembly_colored("colored + joints", Cuboct::Rigid, [1, 1, 1], 0.22),
        heterogeneous_tile("heterogeneous column"),
    ];

    let rows = meshes.len().div_ceil(COLUMNS);
    let (w, h) = (1400u32, 1400u32);
    let mut renderer = match HeadlessRenderer::builder()
        .size(w, h)
        .supersample(2)
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — needs a GPU adapter.");
            std::process::exit(2);
        }
    };
    let (rw, rh) = renderer.render_size();

    let mut scene = Scene::new();
    scene.background = Color::new(0.06, 0.07, 0.09);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.45));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 2.6)
            .with_direction(Vector3::new(-0.35, -0.75, -0.55).normalize()),
    );

    let pitch = PART * 1.75;
    for (i, mut mesh) in meshes.into_iter().enumerate() {
        let (col, row) = (i % COLUMNS, i / COLUMNS);
        mesh.position = Vector3::new(
            (col as f32 - (COLUMNS - 1) as f32 * 0.5) * pitch,
            ((rows - 1) as f32 * 0.5 - row as f32) * pitch,
            0.0,
        );
        mesh.quaternion = Euler::new(0.30, 0.55, 0.0).to_quaternion();
        scene.add(mesh);
    }

    let aspect = w as f32 / h as f32;
    let half_angle = (FOV * 0.5).to_radians().tan();
    let content = (COLUMNS as f32 * pitch, rows as f32 * pitch);
    let distance =
        (content.1 / (2.0 * half_angle)).max(content.0 / (2.0 * half_angle * aspect)) * 1.08;

    let mut camera = PerspectiveCamera::new(FOV, aspect, 1.0, distance * 3.0);
    camera.position = Vector3::new(0.0, 0.0, distance);
    camera.look_at(Vector3::ZERO);

    let rgba = renderer.render_to_rgba(&mut scene, &camera);
    if let Some(dir) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out, encode_png(rw, rh, &rgba)).expect("write png");
    println!("wrote {out} ({rw}x{rh})");

    let svg_path = std::path::Path::new(&out).with_file_name("cuboct-face.svg");
    let svg = CuboctAssembly::new(Cuboct::Rigid)
        .pitch(75.0)
        .beam(4.0)
        .pad(6.0)
        .svg();
    std::fs::write(&svg_path, svg).expect("write svg");
    println!("wrote {}", svg_path.display());

    let export_dir = std::path::Path::new(&out).parent().unwrap_or(std::path::Path::new("out"));
    let stl_dir = export_dir.join("cuboct_parts_stl");
    let obj_dir = export_dir.join("cuboct_parts_obj");
    let mold_asm = CuboctAssembly::new(Cuboct::Rigid)
        .pitch(75.0)
        .beam(4.0)
        .pad(6.0)
        .mold_ready(true)
        .draft(3.0)
        .rivet_diameter(2.38125)
        .resolution(20);
    let n_stl = mold_asm.export_stl_dir(&stl_dir).expect("stl export");
    let n_obj = mold_asm.export_obj_dir(&obj_dir).expect("obj export");
    println!("exported {n_stl} STL + {n_obj} OBJ parts");

    let scad_path = export_dir.join("cuboct-assembly.scad");
    std::fs::write(&scad_path, mold_asm.to_scad()).expect("write scad");
    let mold_path = export_dir.join("cuboct-mold.scad");
    std::fs::write(&mold_path, mold_asm.to_scad_mold()).expect("write mold scad");

    let plan = CuboctAssembly::new(Cuboct::Rigid)
        .pitch(20.0)
        .cells([2, 1, 1])
        .assembly_plan();
    let csv_path = export_dir.join("cuboct-assembly.csv");
    std::fs::write(&csv_path, plan.to_csv()).expect("write csv");
    println!(
        "wrote {}, {}, {}",
        scad_path.display(),
        mold_path.display(),
        csv_path.display()
    );
}

fn tile(
    name: &str,
    kind: LatticeKind,
    configure: impl for<'a> Fn(Lattice<'a>) -> Lattice<'a>,
) -> Object3D {
    let lattice = configure(
        Lattice::new(kind)
            .size(Vector3::new(PART, PART, PART))
            .cells([3, 3, 3])
            .resolution(RESOLUTION)
            .max_samples(BUDGET),
    )
    .fit_relative_density(DENSITY)
    .resolve_walls(WALL_SAMPLES);

    let thickness = lattice.current_thickness();
    let density = lattice.relative_density();
    let across = lattice.wall_samples();
    let grid = lattice.sample_grid();

    let started = Instant::now();
    let geometry = lattice.build();
    let elapsed = started.elapsed();

    let triangles = geometry.index.as_ref().map_or(0, |i| i.len() / 3);
    println!(
        "{name:<24} {thickness:>6.2} mm {:>8.1} % {across:>6.1} {:>4}³ {triangles:>9} {:>7.0} ms",
        density * 100.0,
        grid[0],
        elapsed.as_secs_f64() * 1000.0,
    );

    let mut material = StandardMaterial::new(Color::new(0.78, 0.62, 0.38));
    material.metalness = 0.12;
    material.roughness = 0.48;
    Object3D::mesh(Mesh::new(geometry, material.into()))
}

fn assembly_tile(name: &str, kind: Cuboct, cells: [usize; 3], explode: f32) -> Object3D {
    let asm = CuboctAssembly::new(kind)
        .pitch(PART / cells[0].max(1) as f32)
        .cells(cells)
        .shape(kind.default_shape())
        .explode(explode)
        .resolution(28);
    let n_parts = cells[0] * cells[1] * cells[2] * 6;
    let started = Instant::now();
    let geometry = asm.build();
    let elapsed = started.elapsed();
    let triangles = geometry.index.as_ref().map_or(0, |i| i.len() / 3);
    let n_joints = asm.joints().len();
    println!(
        "{name:<24} {n_parts:>6} parts {n_joints:>5} joints {triangles:>9} {:>7.0} ms",
        elapsed.as_secs_f64() * 1000.0,
    );
    let mut material = StandardMaterial::new(Color::new(0.82, 0.55, 0.32));
    material.metalness = 0.18;
    material.roughness = 0.40;
    Object3D::mesh(Mesh::new(geometry, material.into()))
}

fn assembly_colored(name: &str, kind: Cuboct, cells: [usize; 3], explode: f32) -> Object3D {
    let asm = CuboctAssembly::new(kind)
        .pitch(PART / cells[0].max(1) as f32)
        .cells(cells)
        .shape(kind.default_shape())
        .explode(explode)
        .resolution(24);
    let started = Instant::now();
    let geometry = asm.build_colored();
    let elapsed = started.elapsed();
    let triangles = geometry.index.as_ref().map_or(0, |i| i.len() / 3);
    println!(
        "{name:<24} colored {triangles:>9} {:>7.0} ms",
        elapsed.as_secs_f64() * 1000.0,
    );
    let mut material = StandardMaterial::new(Color::WHITE);
    material.metalness = 0.1;
    material.roughness = 0.45;
    Object3D::mesh(Mesh::new(geometry, material.into()))
}

fn heterogeneous_tile(name: &str) -> Object3D {
    let m = 3i32;
    let asm = CuboctAssembly::new(Cuboct::ChiralCcw)
        .pitch(PART / m as f32)
        .cells([1, 1, m as usize])
        .shape(0.12)
        .chiral_rule(ChiralRule::R1)
        .program(move |_, _, k| Cuboct::column_half(k, m))
        .explode(0.18)
        .resolution(22);
    let started = Instant::now();
    let geometry = asm.build();
    let elapsed = started.elapsed();
    let triangles = geometry.index.as_ref().map_or(0, |i| i.len() / 3);
    println!(
        "{name:<24} {} parts {triangles:>9} {:>7.0} ms",
        asm.parts().len(),
        elapsed.as_secs_f64() * 1000.0,
    );
    let mut material = StandardMaterial::new(Color::new(0.55, 0.72, 0.88));
    material.metalness = 0.14;
    material.roughness = 0.42;
    Object3D::mesh(Mesh::new(geometry, material.into()))
}
