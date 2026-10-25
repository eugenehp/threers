//! Voronoi foams on every kind of seed, open, closed and wet, rendered headless
//! to a PNG.
//!
//! Columns are the seed layouts — random, the Kelvin, Weaire–Phelan and C15
//! crystals, Lloyd-relaxed and two-size (power-diagram) — and rows the three
//! foams built on them: struts on the cell edges, walls on the cell faces, and
//! a wet foam whose rounded bubbles leave liquid in Plateau borders along the
//! edges. Every tile is fitted to a density, so the column you look at changes
//! the shape and not how much material there is.
//!
//! ```text
//! cargo run --release --example lattice_foams                      # out/lattice_foams.png
//! cargo run --release --example lattice_foams --features parallel  # the same, on every core
//! ```

use std::time::Instant;

use threers::{
    encode_png, AmbientLight, Color, DirectionalLight, Euler, HeadlessRenderer, Lattice,
    LatticeKind, Mesh, Object3D, PerspectiveCamera, Scene, SeedLayout, StandardMaterial,
    Stochastic, Vector3,
};

/// Every tile's side, in millimetres.
const PART: f32 = 20.0;
const FOV: f32 = 30.0;

fn main() {
    let out = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .unwrap_or_else(|| "out/lattice_foams.png".into());

    // Lattice cells per side for each layout, so that every column's foam
    // cells come out about the same size: the crystals put several seeds in
    // a cell, the random foam one.
    let layouts = [
        (SeedLayout::Random, 4),
        (SeedLayout::Kelvin, 3),
        (SeedLayout::WeairePhelan, 2),
        (SeedLayout::C15, 1),
        (SeedLayout::Relaxed, 2),
        (SeedLayout::Bimodal, 2),
    ];
    let foams = [
        (Stochastic::Voronoi, 0.10, Color::new(0.42, 0.66, 0.92)),
        (Stochastic::VoronoiWall, 0.10, Color::new(0.93, 0.62, 0.32)),
        (Stochastic::VoronoiWet, 0.25, Color::new(0.48, 0.82, 0.62)),
    ];

    println!(
        "{:<14} {:<13} {:>9} {:>9} {:>9} {:>8}",
        "layout", "foam", "thickness", "density", "triangles", "build"
    );
    let mut tiles = Vec::new();
    for &(kind, density, color) in &foams {
        for &(layout, cells) in &layouts {
            tiles.push(tile(kind, layout, cells, density, color));
        }
    }
    render(&out, tiles, layouts.len());
}

/// One foam, fitted to `density` and built, with what it cost.
fn tile(
    kind: Stochastic,
    layout: SeedLayout,
    cells: usize,
    density: f32,
    color: Color,
) -> Object3D {
    let lattice = Lattice::new(LatticeKind::Stochastic(kind))
        .size(Vector3::new(PART, PART, PART))
        .cells([cells; 3])
        .seed(11)
        .seed_layout(layout)
        // A wet foam's border sets how fat its Plateau borders are: about a
        // sixth of a foam cell here.
        .plateau_border(PART / cells as f32 / layout_cells(layout) * 0.16)
        .resolution(48 / cells)
        .fit_relative_density(density)
        .resolve_walls(2.5);
    let thickness = lattice.current_thickness();
    let density = lattice.relative_density();
    let started = Instant::now();
    let geometry = lattice.build();
    let elapsed = started.elapsed();
    println!(
        "{:<14} {:<13} {thickness:>6.2} mm {:>7.1} % {:>9} {:>5.0} ms",
        layout.name(),
        kind.name(),
        density * 100.0,
        geometry.index.as_ref().map_or(0, |i| i.len() / 3),
        elapsed.as_secs_f64() * 1000.0,
    );
    let mut material = StandardMaterial::new(color);
    material.metalness = 0.1;
    material.roughness = 0.45;
    Object3D::mesh(Mesh::new(geometry, material.into()))
}

/// Foam cells along a lattice cell's side: the cube root of its seeds.
fn layout_cells(layout: SeedLayout) -> f32 {
    match layout {
        SeedLayout::Random => 1.0,
        SeedLayout::Kelvin => 2f32.cbrt(),
        SeedLayout::C15 => 24f32.cbrt(),
        _ => 8f32.cbrt(),
    }
}

/// Lay the tiles out `columns` across and render them to `out`.
fn render(out: &str, tiles: Vec<Object3D>, columns: usize) {
    let rows = tiles.len().div_ceil(columns);
    let (w, h) = (1800u32, 900u32);
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
    let pitch = PART * 1.55;
    for (i, mut mesh) in tiles.into_iter().enumerate() {
        let (col, row) = (i % columns, i / columns);
        // Lattices are built from the origin out; centre each on its slot.
        mesh.position = Vector3::new(
            (col as f32 - (columns - 1) as f32 * 0.5) * pitch,
            ((rows - 1) as f32 * 0.5 - row as f32) * pitch,
            0.0,
        );
        mesh.quaternion = Euler::new(0.32, 0.55, 0.0).to_quaternion();
        scene.add(mesh);
    }
    let aspect = w as f32 / h as f32;
    let half_angle = (FOV * 0.5).to_radians().tan();
    let content = (columns as f32 * pitch, rows as f32 * pitch);
    let distance =
        (content.1 / (2.0 * half_angle)).max(content.0 / (2.0 * half_angle * aspect)) * 1.06;
    let mut camera = PerspectiveCamera::new(FOV, aspect, 1.0, distance * 3.0);
    camera.position = Vector3::new(0.0, 0.0, distance);
    camera.look_at(Vector3::ZERO);
    let rgba = renderer.render_to_rgba(&mut scene, &camera);
    if let Some(dir) = std::path::Path::new(out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(out, encode_png(rw, rh, &rgba)).expect("write png");
    println!("wrote {out} ({rw}x{rh})");
}
