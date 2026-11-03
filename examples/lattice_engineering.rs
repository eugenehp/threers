//! Lattices as engineering, not decoration: foams, conformal cells, a
//! field-driven grade, and the numbers that come out of all of it.
//!
//! Eight tiles, rendered headless to a PNG, and two tables on stdout — what
//! each tile measures as a porous medium, and what a handful of cells measure
//! as a material.
//!
//! The second table's foam rows dominate the run: a foam has no repeating cell,
//! so it has to be homogenised over a window several bubbles across, and that
//! window is 24³ voxels solved nine times over. They run on the GPU
//! ([`Solver::Gpu`]) for that reason, which is about ten times faster than the
//! CPU here and agrees with it to five figures.
//!
//! ```text
//! cargo run --release --example lattice_engineering
//! cargo run --release --example lattice_engineering --features parallel   # ~3x faster
//! ```

use std::time::Instant;

use threers::{
    encode_png, AmbientLight, Color, Conform, DirectionalLight, Euler, Field, HeadlessRenderer,
    Lattice, LatticeKind, LatticeMetrics, Mesh, Object3D, PerspectiveCamera, Region, Scene,
    SolidMaterial, Solver, StandardMaterial, Stochastic, Strut, Tpms, Vector3,
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
        .unwrap_or_else(|| "out/lattice_engineering.png".into());

    let half = PART * 0.5;
    let mut tiles: Vec<(String, Object3D, LatticeMetrics)> = Vec::new();

    // --- Foams: no direction is special, and no two cells are the same. ---
    for (name, cell) in [
        ("voronoi (open cell)", Stochastic::Voronoi),
        ("voronoi (closed cell)", Stochastic::VoronoiWall),
        ("spinodal", Stochastic::Spinodal),
        ("spinodal lamellar", Stochastic::SpinodalLamellar),
    ] {
        tiles.push(tile(name, LatticeKind::Stochastic(cell), |l| {
            l.seed(7).fill(Region::sphere(Vector3::ZERO, half))
        }));
    }

    // --- Conformal: cells that follow the part rather than being cut by it. ---

    // A nozzle. Twelve cells around it and three through the wall, and the
    // tiling closes on itself because the pitch divides the circumference.
    tiles.push(tile(
        "conformal nozzle",
        LatticeKind::Strut(Strut::Cubic),
        |l| {
            let axis = Vector3::new(0.0, 1.0, 0.0);
            let (top, bottom) = (
                Vector3::new(0.0, half, 0.0),
                Vector3::new(0.0, -half, 0.0),
            );
            let wall = 3.0;
            l.fill(
                Region::cone(bottom, top, half * 0.45, half * 0.85).difference(Region::cone(
                    bottom + Vector3::new(0.0, -1.0, 0.0),
                    top + Vector3::new(0.0, 1.0, 0.0),
                    half * 0.45 - wall,
                    half * 0.85 - wall,
                )),
            )
            .conform(Conform::cylindrical(Vector3::ZERO, axis, half * 0.65))
            .cell_size(Vector3::new(
                Conform::ring_pitch(half * 0.65, 12),
                4.0,
                wall / 2.0,
            ))
        },
    ));

    // A helmet liner: two arc lengths and a radius, so the cells tile the
    // sphere instead of being sliced by it.
    tiles.push(tile(
        "conformal shell",
        LatticeKind::Tpms(Tpms::Gyroid),
        |l| {
            l.fill(
                Region::sphere(Vector3::ZERO, half)
                    .difference(Region::sphere(Vector3::ZERO, half - 4.0)),
            )
            .conform(Conform::spherical(Vector3::ZERO, half - 2.0))
            .cell_size(Vector3::new(5.0, 5.0, 2.0))
        },
    ));

    // The cheap half of conforming: world x and y, but z is depth below the
    // surface — so a whole number of layers spans the wall wherever it curves.
    tiles.push(tile("depth-mapped dome", LatticeKind::Tpms(Tpms::Diamond), |l| {
        l.fill(
            Region::sphere(Vector3::ZERO, half)
                .difference(Region::sphere(Vector3::ZERO, half - 5.0)),
        )
        .conform(Conform::depth(Region::sphere(Vector3::ZERO, half)))
        .cell_size(Vector3::new(5.0, 5.0, 2.5))
    }));

    // --- Field-driven: thick where something said it had to be. ---
    tiles.push(tile("graded on a field", LatticeKind::Strut(Strut::Octet), |l| {
        // Stand-in for a solver: hot at one corner, cold at the far one.
        let bounds = threers::Box3::from_center_and_size(
            Vector3::ZERO,
            Vector3::new(PART, PART, PART),
        );
        let field = Field::scattered(
            bounds,
            [10, 10, 10],
            &[
                (Vector3::new(-half, -half, 0.0), 100.0),
                (Vector3::new(half, half, 0.0), 0.0),
            ],
        );
        // A grade of 0.4 makes the thinnest strut two fifths of the nominal
        // one, and `resolve_walls` sizes the grid off the nominal — so the
        // thin end is what decides the sampling here, and it has to be asked
        // for. Without this the lattice comes out as gravel where it is
        // thinnest, silently.
        l.bounds(bounds)
            .resolution(40)
            .grade(field.into_grade(1.8, 0.4))
    }));

    // --- What they measure as porous media. ---
    //
    // `open` is the share of the porosity that reaches the outside — what can
    // be drained of powder. `net` is the largest connected void as a share of
    // all of it: 1 for one space, a half for a sheet TPMS's two labyrinths,
    // near zero for a closed-cell foam. `flow` is the axes a fluid can cross.
    println!(
        "\n{:<22} {:>8} {:>8} {:>8} {:>8} {:>9} {:>8} {:>6} {:>5} {:>5}",
        "tile", "density", "pore", "ligament", "area/vol", "perm", "open", "net", "flow", ""
    );
    for (name, _, m) in &tiles {
        let axes = ["x", "y", "z"];
        let flow: String = (0..3)
            .filter(|&a| m.percolates[a])
            .map(|a| axes[a])
            .collect();
        println!(
            "{name:<22} {:>7.1} % {:>6.2} mm {:>6.2} mm {:>6.2} /mm {:>9.4} {:>7.0} % {:>6.2} {:>5}",
            m.relative_density * 100.0,
            m.pore_diameter,
            m.ligament_thickness,
            m.surface_area_to_volume,
            m.permeability,
            if m.porosity > 0.0 {
                100.0 * m.open_porosity / m.porosity
            } else {
                0.0
            },
            m.largest_void_fraction,
            if flow.is_empty() { "—".into() } else { flow },
        );
    }

    // --- And what a cell measures as a material. ---
    //
    // Six unit strains on a periodic window, as fractions of the base modulus
    // so the units the part is printed in do not matter. A repeating cell is
    // measured one cell at a time; a foam has no cell to repeat, so it gets a
    // window several bubbles across instead — one bubble wrapped in periodic
    // boundaries describes the wrap as much as the foam.
    //
    // Voxel homogenisation converges from above, so these are for ranking cells
    // against each other at one resolution rather than for quoting.
    println!(
        "\n{:<22} {:>7} {:>8} {:>9} {:>9} {:>9} {:>8} {:>7} {:>9} {:>5}",
        "cell", "voxels", "density", "Ex/Es", "Ez/Es", "anisotropy", "sy/500", "worked", "k/ks", "path"
    );
    let mut ran = None;
    // `window` is how many cells across the periodic window is, `per_cell` how
    // many voxels each of them gets. A repeating cell needs a window of one and
    // can afford the resolution; a foam needs the window and pays for it by
    // being coarser, which is a fair trade and not a free one — a coarse grid
    // over-connects, so these read stiffer than they are.
    for (name, kind, window, per_cell) in [
        ("octet truss", LatticeKind::Strut(Strut::Octet), 1, 16),
        ("bcc", LatticeKind::Strut(Strut::Bcc), 1, 16),
        ("gyroid sheet", LatticeKind::Tpms(Tpms::Gyroid), 1, 16),
        ("spinodal", LatticeKind::Stochastic(Stochastic::Spinodal), 3, 8),
        (
            "spinodal lamellar",
            LatticeKind::Stochastic(Stochastic::SpinodalLamellar),
            3,
            8,
        ),
        (
            "voronoi foam",
            LatticeKind::Stochastic(Stochastic::Voronoi),
            3,
            8,
        ),
    ] {
        let started = Instant::now();
        // Stiffness and strength come out of one set of solves.
        // On the GPU: ten times faster than the CPU on a window this size, and
        // the same answer to five figures. It falls back on a machine without
        // an adapter, and says so in `stiffness.solver`.
        let s = Lattice::new(kind)
            .size(Vector3::new(PART, PART, PART))
            .cells([1, 1, 1])
            .seed(7)
            .fit_relative_density(0.3)
            .solver(Solver::Gpu)
            .strength_window(window, per_cell, SolidMaterial::default());
        let c = s.stiffness;
        let e = c.youngs_moduli();
        // The same window, the scalar problem: how well it carries heat — or
        // current, or a diffusing species, which are the same equation. `on
        // path` is that conductivity over a straight bar of the same material
        // and the same density: the share of the material doing any work.
        let k = Lattice::new(kind)
            .size(Vector3::new(PART, PART, PART))
            .cells([1, 1, 1])
            .seed(7)
            .fit_relative_density(0.3)
            .solver(Solver::Gpu)
            .conductivity_window(window, per_cell, 1.0);
        // 316L stainless at 500 MPa, so `sy/500` is the lattice's own yield
        // stress in MPa. `worked` is the share of the material at yield when
        // it gives — 1 would be every ligament failing at once, and nothing
        // reaches it. A `*` means the grid did not resolve the ligaments and
        // the strength is overstated: see `Strength::resolved`.
        println!(
            "{name:<22} {:>6}³ {:>7.1} % {:>9.4} {:>9.4} {:>9.1} {:>7.1}{} {:>7.2} {:>9.4} {:>5.2}   {:.0} ms",
            window * per_cell,
            c.relative_density * 100.0,
            e[0],
            e[2],
            c.anisotropy(),
            s.uniaxial(500.0)[2],
            if s.resolved() { " " } else { "*" },
            s.efficiency()[2],
            k.principal()[0],
            k.tortuosity_factor(1.0),
            started.elapsed().as_secs_f64() * 1000.0,
        );
        ran = Some(c.solver);
    }
    if let Some(solver) = ran {
        println!("  solved on {solver:?}");
    }

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
            eprintln!("\nheadless renderer unavailable ({e}) — needs a GPU adapter.");
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
    for (i, (_, mut mesh, _)) in tiles.into_iter().enumerate() {
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
    println!("\nwrote {out} ({rw}x{rh})");
}

/// Build one tile at the shared density, measure it, and place it.
fn tile(
    name: &str,
    kind: LatticeKind,
    configure: impl for<'a> Fn(Lattice<'a>) -> Lattice<'a>,
) -> (String, Object3D, LatticeMetrics) {
    let lattice = configure(
        Lattice::new(kind)
            .size(Vector3::new(PART, PART, PART))
            .cell_size(Vector3::new(1.0, 1.0, 1.0) * (PART / 4.0))
            .resolution(20)
            .max_samples(6_000_000),
    )
    .fit_relative_density(DENSITY)
    .resolve_walls(2.5);

    let metrics = lattice.metrics();
    let geometry = lattice.build();

    let mut material = StandardMaterial::new(Color::new(0.42, 0.66, 0.92));
    material.metalness = 0.15;
    material.roughness = 0.42;
    (
        name.to_string(),
        Object3D::mesh(Mesh::new(geometry, material.into())),
        metrics,
    )
}
