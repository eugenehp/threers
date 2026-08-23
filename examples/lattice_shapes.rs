//! Lattices poured into shapes rather than boxes, rendered headless to a PNG.
//!
//! A [`Region`] is a field that is positive inside and negative out, plus the
//! box it occupies. `Lattice::fill` takes one, cuts the lattice to it, and
//! closes the mesh over the cut — so every tile here is still a single
//! watertight shell.
//!
//! Two of the tiles need features. `openscad` implies `mesh-bvh`, so the last
//! line builds all eight; `parallel` is worth adding for the two that query a
//! mesh per sample, which it speeds up by an order of magnitude.
//!
//! ```text
//! cargo run --release --example lattice_shapes                        # 6 primitives, curves, CSG
//! cargo run --release --example lattice_shapes --features mesh-bvh    # + a triangle mesh
//! cargo run --release --example lattice_shapes --features openscad,parallel   # + a .scad model
//! ```

use std::time::Instant;

use threers::{
    encode_png, AmbientLight, CatmullRomCurve3, Color, Cuboct, DirectionalLight, Euler,
    HeadlessRenderer, Lattice, LatticeKind, Mesh, Object3D, PerspectiveCamera, Region, Scene,
    StandardMaterial, Strut, Tpms, Vector3,
};

/// Roughly the size of every tile, in millimetres.
const PART: f32 = 20.0;
const DENSITY: f32 = 0.22;
const COLUMNS: usize = 4;
const FOV: f32 = 32.0;

fn main() {
    let out = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .unwrap_or_else(|| "out/lattice_shapes.png".into());

    println!(
        "{:<24} {:>9} {:>9} {:>5} {:>9} {:>10}",
        "region", "thickness", "density", "grid", "triangles", "build"
    );

    let half = PART * 0.5;
    let mut tiles: Vec<Object3D> = Vec::new();

    tiles.push(tile(
        "sphere",
        LatticeKind::Tpms(Tpms::Gyroid),
        DENSITY,
        |l| l.fill(Region::sphere(Vector3::ZERO, half)),
    ));

    tiles.push(tile(
        "cylinder",
        LatticeKind::Strut(Strut::Octet),
        DENSITY,
        |l| {
            l.fill(Region::cylinder(
                Vector3::new(0.0, -half, 0.0),
                Vector3::new(0.0, half, 0.0),
                half * 0.7,
            ))
        },
    ));

    tiles.push(tile(
        "cuboct sphere",
        LatticeKind::Cuboct(Cuboct::Auxetic),
        DENSITY,
        |l| {
            l.fill(Region::sphere(Vector3::ZERO, half))
                .shape(0.18)
        },
    ));

    tiles.push(tile(
        "torus",
        LatticeKind::Tpms(Tpms::Diamond),
        DENSITY,
        |l| {
            l.fill(
                Region::torus(Vector3::ZERO, half * 0.62, half * 0.32).rotate(
                    threers::Quaternion::from_axis_angle(
                        Vector3::new(1.0, 0.0, 0.0),
                        std::f32::consts::FRAC_PI_2,
                    ),
                ),
            )
        },
    ));

    // A swept curve: the tube follows it, and the lattice fills the tube.
    tiles.push(tile(
        "swept curve",
        LatticeKind::Tpms(Tpms::Gyroid),
        DENSITY,
        |l| {
            let curve = CatmullRomCurve3::new(vec![
                Vector3::new(-half, -half * 0.6, 0.0),
                Vector3::new(-half * 0.3, half * 0.7, half * 0.4),
                Vector3::new(half * 0.3, -half * 0.7, -half * 0.4),
                Vector3::new(half, half * 0.6, 0.0),
            ]);
            l.fill(Region::tube(&curve, half * 0.3, 64))
        },
    ));

    // Constructive solid geometry, on the regions themselves.
    tiles.push(tile(
        "cube − sphere − bar",
        LatticeKind::Tpms(Tpms::SchwarzP),
        DENSITY,
        |l| {
            let cube = Region::cuboid(threers::Box3::from_center_and_size(
                Vector3::ZERO,
                Vector3::new(PART, PART, PART),
            ));
            l.fill(
                cube.difference(Region::sphere(Vector3::new(half, half, half), half * 1.1))
                    .difference(Region::capsule(
                        Vector3::new(-PART, 0.0, 0.0),
                        Vector3::new(PART, 0.0, 0.0),
                        half * 0.3,
                    )),
            )
        },
    ));

    // A solid skin over the infill, cut open so you can see both. The cut is
    // just another region subtracted from the fill.
    // A higher target than the rest, because the skin is already a good part
    // of a shape this thin — ask for 22 % and there would be nothing left for
    // the lattice to be.
    tiles.push(tile(
        "skin, cut away",
        LatticeKind::Strut(Strut::Bcc),
        0.42,
        |l| {
            let quarter = || {
                Region::cuboid(threers::Box3::new(
                    Vector3::new(0.0, 0.0, -PART),
                    Vector3::new(PART, PART, PART),
                ))
            };
            // `fill` then `skin` makes the whole ball; `clip` sections it
            // afterwards, so the skin stops at the cut face instead of wrapping
            // round it and hiding the infill.
            l.fill(Region::sphere(Vector3::ZERO, half))
                .skin(0.7)
                .clip(quarter().invert())
        },
    ));

    #[cfg(feature = "mesh-bvh")]
    tiles.push(tile(
        "mesh",
        LatticeKind::Strut(Strut::Diamond),
        DENSITY,
        |l| {
            // Any closed mesh will do; a torus knot is one that is obviously not a
            // primitive.
            let knot = threers::TorusKnotGeometry::new(half * 0.6, half * 0.22, 128, 16, 2, 3);
            l.fill(Region::mesh(&knot).expect("a bvh over the knot"))
        },
    ));

    #[cfg(feature = "openscad")]
    tiles.push(tile(
        "openscad",
        LatticeKind::Tpms(Tpms::Gyroid),
        DENSITY,
        |l| {
            // Deliberately a boolean the exact-CSG kernel resolves quickly:
            // this example is about filling the result, not about stressing
            // the kernel. Three crossed cylinders through the same cube is a
            // far harder case, and it is the kernel that would be being timed.
            l.fill(
                Region::scad("difference(){ cube(20, center=true); sphere(11, $fn=32); }")
                    .expect("valid scad"),
            )
        },
    ));

    // --- Scene ---
    let rows = tiles.len().div_ceil(COLUMNS);
    let (w, h) = (1700u32, 900u32);
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

    let pitch = PART * 1.6;
    for (i, mut mesh) in tiles.into_iter().enumerate() {
        let (col, row) = (i % COLUMNS, i / COLUMNS);
        mesh.position = Vector3::new(
            (col as f32 - (COLUMNS - 1) as f32 * 0.5) * pitch,
            ((rows - 1) as f32 * 0.5 - row as f32) * pitch,
            0.0,
        );
        mesh.quaternion = Euler::new(0.28, 0.5, 0.0).to_quaternion();
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
}

/// Build one tile at the shared density, report what it cost, and place it.
fn tile(
    name: &str,
    kind: LatticeKind,
    density: f32,
    configure: impl for<'a> Fn(Lattice<'a>) -> Lattice<'a>,
) -> Object3D {
    let lattice = configure(
        Lattice::new(kind)
            .cell_size(Vector3::new(1.0, 1.0, 1.0) * (PART / 3.0))
            .resolution(20)
            .max_samples(12_000_000),
    )
    .fit_relative_density(density)
    .resolve_walls(2.5);

    let thickness = lattice.current_thickness();
    let density = lattice.relative_density();
    let grid = lattice.sample_grid();

    let started = Instant::now();
    let geometry = lattice.build();
    let elapsed = started.elapsed();

    let triangles = geometry.index.as_ref().map_or(0, |i| i.len() / 3);
    println!(
        "{name:<24} {thickness:>6.2} mm {:>8.1} % {:>4}³ {triangles:>9} {:>7.0} ms",
        density * 100.0,
        grid[0],
        elapsed.as_secs_f64() * 1000.0,
    );

    let mut material = StandardMaterial::new(Color::new(0.42, 0.66, 0.92));
    material.metalness = 0.15;
    material.roughness = 0.42;
    Object3D::mesh(Mesh::new(geometry, material.into()))
}
