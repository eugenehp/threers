//! Terrain diffusion as an iterated graph (`--features rlx,rlx-geo`).
//!
//! ```text
//! cargo run --release --features "rlx,rlx-geo" --example rlx_geo_erosion
//! ```
//!
//! Writes `out/rlx_erosion_{before,after}.png` and prints how much relief each
//! pass removed.
//!
//! The height field is a regular grid, and `rlx-geo` triangulates it exactly —
//! worth noting, because a regular grid is the *worst* case for a
//! floating-point Delaunay: every cell's four corners are exactly cocircular,
//! so every quad is a coin-flip that a rounded in-circle test can get wrong in
//! both directions at once.
//!
//! The erosion itself is one 3×3 convolution compiled once and run a hundred
//! times, each pass feeding the last one's output back in. It is *thermal*
//! erosion — `h ← h + κ∇²h`, material creeping down-slope — not hydraulic:
//! there is no water, no sediment and no flow accumulation here, so it rounds
//! hills and fills hollows but will not carve a drainage network.

use threers::rlx::{geo, preferred_device, Diffusion};
use threers::{
    encode_png, AmbientLight, Color, DirectionalLight, HeadlessRenderer, Mesh, Object3D,
    PerspectiveCamera, Scene, StandardMaterial, Vector3,
};

const GRID: u32 = 220;
const EXTENT: f32 = 10.0;
const PASSES: usize = 120;

fn main() {
    std::fs::create_dir_all("out").ok();
    let device = preferred_device();
    println!("rlx device: {device:?}");

    // --- A rough terrain --------------------------------------------------
    let before: Vec<f32> = (0..GRID as usize * GRID as usize)
        .map(|i| {
            let (x, z) = cell_to_world(i);
            terrain(x, z)
        })
        .collect();

    // --- Erode it ---------------------------------------------------------
    let mut diffusion = Diffusion::new(GRID, GRID, 0.2, device);
    let after = diffusion.run(&before, PASSES).expect("diffuse");

    println!(
        "{PASSES} passes over {GRID}×{GRID}\n  relief   {:.3} → {:.3}\n  roughness {:.4} → {:.4}",
        relief(&before),
        relief(&after),
        roughness(&before),
        roughness(&after),
    );

    // --- Look at them -----------------------------------------------------
    let mut headless = match HeadlessRenderer::builder().size(900, 600).build() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — the numbers above still stand.");
            std::process::exit(2);
        }
    };
    shoot(&mut headless, &before, "before");
    shoot(&mut headless, &after, "after");
}

/// Ridged noise: sharp crests, the kind diffusion visibly rounds off.
fn terrain(x: f32, z: f32) -> f32 {
    let mut height = 0.0;
    let mut frequency = 0.35;
    let mut amplitude = 1.6;
    for _ in 0..4 {
        let v = (x * frequency).sin() * (z * frequency * 0.9 + 1.3).cos();
        height += amplitude * (1.0 - v.abs());
        frequency *= 2.1;
        amplitude *= 0.45;
    }
    height - 1.4
}

fn cell_to_world(i: usize) -> (f32, f32) {
    let (gx, gz) = (i % GRID as usize, i / GRID as usize);
    let step = 2.0 * EXTENT / (GRID - 1) as f32;
    (-EXTENT + gx as f32 * step, -EXTENT + gz as f32 * step)
}

/// Peak-to-trough: what diffusion should mostly *keep*.
fn relief(field: &[f32]) -> f32 {
    let (min, max) = field
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        });
    max - min
}

/// Mean absolute Laplacian: what diffusion should remove.
fn roughness(field: &[f32]) -> f32 {
    let w = GRID as usize;
    let mut total = 0.0;
    let mut count = 0;
    for y in 1..w - 1 {
        for x in 1..w - 1 {
            let c = field[y * w + x];
            let l = field[y * w + x - 1]
                + field[y * w + x + 1]
                + field[(y - 1) * w + x]
                + field[(y + 1) * w + x]
                - 4.0 * c;
            total += l.abs();
            count += 1;
        }
    }
    total / count as f32
}

fn shoot(headless: &mut HeadlessRenderer, field: &[f32], name: &str) {
    let points: Vec<Vector3> = field
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let (x, z) = cell_to_world(i);
            Vector3::new(x, *h, z)
        })
        .collect();
    let geometry = geo::heightfield_geometry(&points).expect("triangulate");

    let (width, height) = headless.render_size();
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0e131b);
    scene.add_light(AmbientLight::new(Color::from_hex(0x6d7f9a), 0.4));
    scene.add_light(
        DirectionalLight::new(Color::from_hex(0xfff2e0), 2.8)
            .with_direction(Vector3::new(-0.5, -0.72, -0.4).normalize()),
    );

    let mut material = StandardMaterial::new(Color::from_hex(0x7f8b6a));
    material.roughness = 0.9;
    scene.add(Object3D::mesh(Mesh::new(geometry, material.into())));

    let mut camera = PerspectiveCamera::new(40.0, width as f32 / height as f32, 0.1, 200.0);
    camera.position = Vector3::new(14.0, 11.0, 17.0);
    camera.look_at(Vector3::new(0.0, -0.5, 0.0));

    let rgba = headless.render_to_rgba(&mut scene, &camera);
    let path = format!("out/rlx_erosion_{name}.png");
    std::fs::write(&path, encode_png(width, height, &rgba)).expect("write png");
    println!("wrote {path}");
}
