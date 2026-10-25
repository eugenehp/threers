//! Textiles, screws and grown grains, rendered headless to a PNG.
//!
//! The top row is yarn rather than beams: a plain weave, a three-strand braid
//! and a six-fibre rope, whose strands pass one another instead of meeting at
//! nodes, then the helicoid — the screw-shaped minimal surface — as a sheet.
//! The bottom row is grain: a spinodal for reference, a Gaussian field with a
//! spread of pore sizes, hyperuniform fibres and a Gray–Scott reaction–
//! diffusion maze grown on a periodic tile.
//!
//! ```text
//! cargo run --release --example lattice_textiles                      # out/lattice_textiles.png
//! cargo run --release --example lattice_textiles --features parallel
//! ```

use std::time::Instant;

use threers::{
    encode_png, AmbientLight, Color, DirectionalLight, Euler, HeadlessRenderer, Lattice,
    LatticeKind, LatticeStyle, Mesh, Object3D, PerspectiveCamera, Scene, StandardMaterial,
    Stochastic, Strut, Tpms, Vector3,
};

/// Every tile's side, in millimetres.
const PART: f32 = 20.0;
const FOV: f32 = 30.0;

fn main() {
    let out = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .unwrap_or_else(|| "out/lattice_textiles.png".into());
    let yarn = Color::new(0.93, 0.62, 0.32);
    let grain = Color::new(0.42, 0.66, 0.92);
    // Kind, cells a side, style, what to fit, colour.
    let tiles: [(&str, LatticeKind, usize, LatticeStyle, Fit, Color); 8] = [
        (
            "weave",
            LatticeKind::Strut(Strut::Weave),
            2,
            LatticeStyle::Solid,
            Fit::Thickness(1.6),
            yarn,
        ),
        (
            "braid",
            LatticeKind::Strut(Strut::Braid),
            2,
            LatticeStyle::Solid,
            Fit::Thickness(1.2),
            yarn,
        ),
        (
            "helix-bundle",
            LatticeKind::Strut(Strut::HelixBundle),
            2,
            LatticeStyle::Solid,
            Fit::Thickness(1.1),
            yarn,
        ),
        (
            "helicoid",
            LatticeKind::Tpms(Tpms::Helicoid),
            2,
            LatticeStyle::Sheet,
            Fit::Density(0.2),
            yarn,
        ),
        (
            "spinodal",
            LatticeKind::Stochastic(Stochastic::Spinodal),
            3,
            LatticeStyle::Solid,
            Fit::Density(0.35),
            grain,
        ),
        (
            "gaussian-field",
            LatticeKind::Stochastic(Stochastic::GaussianField),
            3,
            LatticeStyle::Solid,
            Fit::Density(0.35),
            grain,
        ),
        // Fibres are sparse by nature (about 2 %); grow them modestly.
        (
            "hyperuniform",
            LatticeKind::Stochastic(Stochastic::Hyperuniform),
            4,
            LatticeStyle::Solid,
            Fit::Density(0.06),
            grain,
        ),
        (
            "reaction-diffusion",
            LatticeKind::Stochastic(Stochastic::ReactionDiffusion),
            1,
            LatticeStyle::Sheet,
            Fit::Density(0.3),
            grain,
        ),
    ];
    println!(
        "{:<20} {:>9} {:>9} {:>9} {:>8}",
        "lattice", "thickness", "density", "triangles", "build"
    );
    let tiles = tiles
        .into_iter()
        .map(|(name, kind, cells, style, fit, color)| tile(name, kind, cells, style, fit, color))
        .collect();
    render(&out, tiles, 4);
}

/// What a tile is sized by: a yarn's diameter, or a fitted density.
#[derive(Clone, Copy)]
enum Fit {
    Thickness(f32),
    Density(f32),
}

fn tile(
    name: &str,
    kind: LatticeKind,
    cells: usize,
    style: LatticeStyle,
    fit: Fit,
    color: Color,
) -> Object3D {
    let lattice = Lattice::new(kind)
        .size(Vector3::new(PART, PART, PART))
        .cells([cells; 3])
        .style(style)
        .seed(5)
        .resolution(64 / cells);
    let lattice = match fit {
        Fit::Thickness(t) => lattice.thickness(t),
        Fit::Density(d) => lattice.fit_relative_density(d),
    }
    .resolve_walls(2.5);
    let thickness = lattice.current_thickness();
    let density = lattice.relative_density();
    let started = Instant::now();
    let geometry = lattice.build();
    let elapsed = started.elapsed();
    println!(
        "{name:<20} {thickness:>6.2} mm {:>7.1} % {:>9} {:>5.0} ms",
        density * 100.0,
        geometry.index.as_ref().map_or(0, |i| i.len() / 3),
        elapsed.as_secs_f64() * 1000.0,
    );
    let mut material = StandardMaterial::new(color);
    material.metalness = 0.1;
    material.roughness = 0.45;
    Object3D::mesh(Mesh::new(geometry, material.into()))
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
