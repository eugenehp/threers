//! Every lattice generator, rendered headless to a PNG.
//!
//! Each tile is a 20 mm cube of infill fitted to the same 25 % relative
//! density, so what varies between them is topology, not mass — which is the
//! comparison worth looking at when picking one. The last row is what can be
//! done *to* a generator rather than which one to pick: the same gyroid filled
//! solid instead of walled, and one graded across the part and poured into a
//! sphere.
//!
//! Run: `cargo run --release --example lattice [-- out/lattice.png]`

use std::time::Instant;

use threers::{
    encode_png, AmbientLight, Color, DirectionalLight, Euler, HeadlessRenderer, Lattice,
    LatticeKind, Mesh, Object3D, PerspectiveCamera, Scene, StandardMaterial, Vector3,
};

/// Cube side, in millimetres — the numbers printed below are real lengths.
const PART: f32 = 20.0;
/// Cells across the part.
const CELLS: [usize; 3] = [3, 3, 3];
/// What every tile is fitted to, so the comparison is like for like.
const DENSITY: f32 = 0.25;
/// Samples per cell to start from — enough for the thick-walled generators.
const RESOLUTION: usize = 20;
/// …and the floor every generator is then held to. At 25 % density the
/// high-area surfaces — Fischer–Koch S, Lidinoid, split P — reach it on walls a
/// third the thickness a gyroid needs, and a wall thinner than the sample step
/// comes out as gravel rather than as a wall. `resolve_walls` raises the
/// resolution per generator instead of paying the worst case on all of them.
const WALL_SAMPLES: f32 = 2.5;
/// The ceiling `resolve_walls` works under. 12 M samples is 48 MB.
const BUDGET: usize = 12_000_000;

const COLUMNS: usize = 6;
/// Vertical field of view, in degrees.
const FOV: f32 = 32.0;

fn main() {
    let out = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .unwrap_or_else(|| "out/lattice.png".into());

    println!(
        "{:<21} {:>9} {:>9} {:>6} {:>5} {:>9} {:>10}",
        "generator", "thickness", "density", "across", "grid", "triangles", "build"
    );

    let mut meshes: Vec<Object3D> = LatticeKind::all()
        .into_iter()
        .map(|kind| tile(kind.name(), kind, |l| l))
        .collect();

    // The last row is what can be done *to* a generator rather than which one
    // to pick. Both are the gyroid, so the comparison is with the first tile.

    // A minimal surface bounds two interlocking labyrinths. Every tile above
    // walls the surface itself; this one fills one labyrinth solid and leaves
    // the other as the void. Same mass, quite different stiffness — and a
    // single connected void rather than two.
    meshes.push(tile(
        "gyroid solid",
        LatticeKind::Tpms(threers::Tpms::Gyroid),
        |l| l.style(threers::LatticeStyle::Solid),
    ));

    // Thin walls at the left face, three times as thick at the right, and the
    // whole thing poured into a sphere instead of filling the box. The trim
    // closes over the cut ends, so this is still one watertight shell.
    meshes.push(tile(
        "gyroid graded+trimmed",
        LatticeKind::Tpms(threers::Tpms::Gyroid),
        |l| {
            l.grade(|p| 1.0 + (p.x / PART + 0.5) * 2.0)
                .trim(|p| PART * 0.5 - p.length())
        },
    ));

    // --- Scene ---
    let rows = meshes.len().div_ceil(COLUMNS);
    let (w, h) = (1700u32, 1600u32);
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
        // Every tile at the same three-quarter angle: face-on, several of
        // these read as a solid slab because you are looking straight down
        // their channels.
        mesh.quaternion = Euler::new(0.30, 0.55, 0.0).to_quaternion();
        scene.add(mesh);
    }

    // Pull back until the whole grid fits, rather than a distance tuned to one
    // tile count — adding a generator should not crop the picture.
    let aspect = w as f32 / h as f32;
    let half_angle = (FOV * 0.5).to_radians().tan();
    let content = (COLUMNS as f32 * pitch, rows as f32 * pitch);
    let distance =
        (content.1 / (2.0 * half_angle)).max(content.0 / (2.0 * half_angle * aspect)) * 1.06;

    let mut camera = PerspectiveCamera::new(FOV, aspect, 1.0, distance * 3.0);
    camera.position = Vector3::new(0.0, 0.0, distance);
    camera.look_at(Vector3::ZERO);

    let rgba = renderer.render_to_rgba(&mut scene, &camera);
    if let Some(dir) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out, encode_png(rw, rh, &rgba)).expect("write png");
    println!("wrote {out} ({rw}x{rh})");
}

/// Build one tile, report what it cost, and hand back a mesh ready to place.
fn tile(
    name: &str,
    kind: LatticeKind,
    configure: impl for<'a> Fn(Lattice<'a>) -> Lattice<'a>,
) -> Object3D {
    let lattice = configure(
        Lattice::new(kind)
            .size(Vector3::new(PART, PART, PART))
            .cells(CELLS)
            .resolution(RESOLUTION)
            .max_samples(BUDGET),
    )
    // Density first: the thickness it solves for is what decides how fine the
    // sampling has to be.
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
        "{name:<21} {thickness:>6.2} mm {:>8.1} % {across:>6.1} {:>4}³ {triangles:>9} {:>7.0} ms",
        density * 100.0,
        grid[0],
        elapsed.as_secs_f64() * 1000.0,
    );

    let mut material = StandardMaterial::new(Color::new(0.42, 0.66, 0.92));
    material.metalness = 0.15;
    material.roughness = 0.42;
    Object3D::mesh(Mesh::new(geometry, material.into()))
}
