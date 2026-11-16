//! Scale-rich networks across their design space, and grown into shapes and
//! patterns, rendered headless to a PNG.
//!
//! The top three rows are the two-parameter design space: columns are the
//! decay exponent α — how fast each new ligament thins — and rows the relative
//! density, reached by choosing λ₀, the first ligament's width, from the
//! published design table. At α = 0 every ligament is the same width and the
//! network is nearly isotropic; by α = 0.9 a few thick early ligaments carry
//! the load and hundreds of thin ones fill between them.
//!
//! The bottom row is the same rule put to work: grown in a ring with its
//! ligaments following the ring round, graded by making seeds likelier on the
//! right, grown in an L with two round holes, and aligned to three directions.
//!
//! ```text
//! cargo run --release --example lattice_scale_rich   # out/lattice_scale_rich.png
//! ```

use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, TAU};
use std::time::Instant;

use threers::{
    center_geometry, encode_png, AmbientLight, Color, DirectionalLight, Euler, HeadlessRenderer,
    Mesh, Object3D, PerspectiveCamera, ScaleRich, ScaleRichAngles, ScaleRichNet, Scene,
    StandardMaterial, Vector3,
};

/// Every tile's side and thickness, in millimetres.
const PART: f32 = 20.0;
const DEPTH: f32 = 2.5;
const FOV: f32 = 30.0;

/// α for each column, and `(λ₀, relative density)` down it: the authors'
/// design table for 500 steps and a 0.03 gap.
const DESIGNS: [(f64, [(f64, f64); 3]); 4] = [
    (0.0, [(0.0044, 0.2), (0.01, 0.4), (0.015, 0.6)]),
    (0.3, [(0.015, 0.2), (0.03, 0.4), (0.05, 0.6)]),
    (0.6, [(0.04, 0.2), (0.08, 0.4), (0.12, 0.6)]),
    (0.9, [(0.07, 0.2), (0.15, 0.4), (0.25, 0.6)]),
];

fn main() {
    let out = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .unwrap_or_else(|| "out/lattice_scale_rich.png".into());

    println!(
        "{:<22} {:>8} {:>9} {:>6} {:>9} {:>7}",
        "network", "density", "ligaments", "eta_G", "triangles", "build"
    );
    let mut tiles = Vec::new();
    for row in 0..3 {
        for (column, &(alpha, designs)) in DESIGNS.iter().enumerate() {
            let (lambda0, target) = designs[row];
            let hue = column as f32 / (DESIGNS.len() - 1) as f32;
            let color = Color::new(0.95 - 0.45 * hue, 0.6 + 0.05 * hue, 0.35 + 0.55 * hue);
            let label = format!("α {alpha:.1} at {:.0} %", target * 100.0);
            tiles.push(tile(
                &label,
                ScaleRich::new(alpha, lambda0).steps(500).seed(5),
                1.0,
                color,
            ));
        }
    }

    // Fig. 6 of the paper: the same growth rule, shaped.
    let accent = Color::new(0.55, 0.85, 0.7);
    let circle = |cx: f64, cy: f64, r: f64| -> Vec<[f64; 2]> {
        (0..96)
            .map(|i| {
                let a = TAU * i as f64 / 96.0;
                [cx + r * a.cos(), cy + r * a.sin()]
            })
            .collect()
    };
    let ring = ScaleRich::new(0.3, 0.05)
        .steps(600)
        .seed(3)
        .domain(circle(0.0, 0.0, 1.0), vec![circle(0.0, 0.0, 0.42)])
        .angles(ScaleRichAngles::field(|p| p[1].atan2(p[0]) + FRAC_PI_2));
    tiles.push(tile("ring, along a field", ring, 0.5, accent));
    let graded = ScaleRich::new(0.3, 0.02)
        .steps(800)
        .seed(3)
        .nucleation(|p| 0.05 + 0.95 * p[0] * p[0]);
    tiles.push(tile("graded seeding", graded, 1.0, accent));
    let l_shape = vec![
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 0.9],
        [0.9, 0.9],
        [0.9, 2.0],
        [0.0, 2.0],
    ];
    let holed = ScaleRich::new(0.4, 0.07).steps(600).seed(3).domain(
        l_shape,
        vec![circle(0.45, 0.45, 0.22), circle(1.5, 0.45, 0.2)],
    );
    tiles.push(tile("L with holes", holed, 0.5, accent));
    let aligned = ScaleRich::new(0.3, 0.03)
        .seed(3)
        .angles(ScaleRichAngles::Set(vec![0.0, FRAC_PI_3, 2.0 * FRAC_PI_3]));
    tiles.push(tile("three directions", aligned, 1.0, accent));

    render(&out, tiles, DESIGNS.len());
}

/// Grows `network`, scales it by `size` × [`PART`] and builds the part.
fn tile(label: &str, network: ScaleRich, size: f32, color: Color) -> Object3D {
    let started = Instant::now();
    let net: ScaleRichNet = network.grow();
    let mut geometry = net.geometry(PART * size, DEPTH);
    let elapsed = started.elapsed();
    center_geometry(&mut geometry);
    println!(
        "{label:<22} {:>6.1} % {:>9} {:>6.2} {:>9} {:>4.0} ms",
        net.relative_density() * 100.0,
        net.ligaments().len(),
        net.geometric_anisotropy(),
        geometry.index.as_ref().map_or(0, |i| i.len() / 3),
        elapsed.as_secs_f64() * 1000.0,
    );
    let mut material = StandardMaterial::new(color);
    material.metalness = 0.05;
    material.roughness = 0.5;
    Object3D::mesh(Mesh::new(geometry, material.into()))
}

/// Lay the tiles out `columns` across and render them to `out`.
fn render(out: &str, tiles: Vec<Object3D>, columns: usize) {
    let rows = tiles.len().div_ceil(columns);
    let (w, h) = (1600u32, 1600u32);
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
    scene.add_light(AmbientLight::new(Color::WHITE, 0.5));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 2.4)
            .with_direction(Vector3::new(-0.3, -0.6, -0.75).normalize()),
    );
    let pitch = PART * 1.2;
    for (i, mut mesh) in tiles.into_iter().enumerate() {
        let (col, row) = (i % columns, i / columns);
        mesh.position = Vector3::new(
            (col as f32 - (columns - 1) as f32 * 0.5) * pitch,
            ((rows - 1) as f32 * 0.5 - row as f32) * pitch,
            0.0,
        );
        // A slight tilt, so the walls catch the light and read as solid.
        mesh.quaternion = Euler::new(-0.18, 0.12, 0.0).to_quaternion();
        scene.add(mesh);
    }
    let aspect = w as f32 / h as f32;
    let half_angle = (FOV * 0.5).to_radians().tan();
    let content = (columns as f32 * pitch, rows as f32 * pitch);
    let distance =
        (content.1 / (2.0 * half_angle)).max(content.0 / (2.0 * half_angle * aspect)) * 1.04;
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
