//! Exact Delaunay and Voronoi through `rlx-geo` (`--features rlx-geo`).
//!
//! ```text
//! cargo run --release --features rlx-geo --example rlx_geo_terrain
//! ```
//!
//! * `out/rlx_geo_terrain.png` — 4 000 scattered samples triangulated into a
//!   mesh and rendered. Scattered, not gridded: a grid is trivially
//!   triangulable and would prove nothing. Jittered samples put four points on
//!   a common circle often enough that a floating-point predicate flips one,
//!   and a flipped triangle in a height field is a visible spike.
//! * `out/rlx_geo_voronoi.png` — the same sites as a Voronoi map, which is a
//!   texture rather than a mesh (cell noise, region masks, mosaics).

use threers::rlx::geo;
use threers::{
    encode_png, AmbientLight, Color, DirectionalLight, HeadlessRenderer, Mesh, Object3D,
    PerspectiveCamera, Scene, StandardMaterial, Vector2, Vector3,
};

const SITES: usize = 4_000;
const EXTENT: f32 = 10.0;
const SIZE: u32 = 900;

fn main() {
    std::fs::create_dir_all("out").ok();

    // --- Scattered samples over a hilly field ---------------------------
    let mut rng = Lcg::new(0x5eed);
    let samples: Vec<Vector3> = (0..SITES)
        .map(|_| {
            let x = rng.range(-EXTENT, EXTENT);
            let z = rng.range(-EXTENT, EXTENT);
            Vector3::new(x, height(x, z), z)
        })
        .collect();

    let geometry = geo::heightfield_geometry(&samples).expect("triangulate");
    let triangles = geometry.index.as_ref().map(|i| i.len() / 3).unwrap_or(0);
    println!("{SITES} samples → {triangles} triangles");

    // --- Voronoi over the same sites ------------------------------------
    //
    // Sites are in pixel coordinates, so scale the field onto the image.
    let scale = SIZE as f32 / (2.0 * EXTENT);
    let sites: Vec<Vector2> = samples
        .iter()
        .map(|p| Vector2::new((p.x + EXTENT) * scale, (p.z + EXTENT) * scale))
        .collect();
    // Sites are assigned palette entries by index, so neighbouring cells take
    // unrelated colours: keep the ramp narrow or the map reads as confetti.
    let palette: Vec<Color> = (0..8)
        .map(|i| {
            let t = i as f32 / 7.0;
            Color::new(0.06 + 0.20 * t, 0.16 + 0.34 * t, 0.30 + 0.28 * t)
        })
        .collect();
    let voronoi = geo::voronoi_texture(&sites, &palette, SIZE, SIZE);
    std::fs::write(
        "out/rlx_geo_voronoi.png",
        encode_png(voronoi.width, voronoi.height, &voronoi.data),
    )
    .expect("write png");
    println!("wrote out/rlx_geo_voronoi.png");

    // --- Render the terrain ---------------------------------------------
    let mut headless = match HeadlessRenderer::builder().size(SIZE, SIZE * 2 / 3).build() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — the Voronoi map was still written.");
            std::process::exit(2);
        }
    };
    let (width, height_px) = headless.render_size();

    let mut scene = Scene::new();
    scene.background = Color::new(0.05, 0.07, 0.10);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.35));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 2.2)
            .with_direction(Vector3::new(-0.5, -0.8, -0.3).normalize()),
    );
    let mut material = StandardMaterial::new(Color::new(0.35, 0.55, 0.32));
    material.roughness = 0.85;
    material.metalness = 0.0;
    scene.add(Object3D::mesh(Mesh::new(geometry, material.into())));

    let mut camera = PerspectiveCamera::new(45.0, width as f32 / height_px as f32, 0.1, 200.0);
    camera.position = Vector3::new(20.0, 16.0, 25.0);
    camera.look_at(Vector3::new(0.0, -2.5, 0.0));

    let rgba = headless.render_to_rgba(&mut scene, &camera);
    std::fs::write(
        "out/rlx_geo_terrain.png",
        encode_png(width, height_px, &rgba),
    )
    .expect("write png");
    println!("wrote out/rlx_geo_terrain.png ({width}x{height_px})");
}

/// Two ridges and a bowl — enough relief to see a bad triangle.
fn height(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    1.6 * (0.6 * x).sin() * (0.45 * z).cos() - 0.06 * r * r + 1.2 * (0.9 * r).cos()
}

/// A small deterministic generator, so the picture is the same every run.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (self.0 >> 32) as f32 / u32::MAX as f32;
        lo + (hi - lo) * u
    }
}
