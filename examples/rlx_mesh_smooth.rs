//! Mesh smoothing as a graph over the positions tensor (`--features rlx`).
//!
//! ```text
//! cargo run --release --features rlx --example rlx_mesh_smooth
//! ```
//!
//! Writes `out/rlx_mesh_{noisy,laplacian,taubin}.png` and prints what each
//! pass cost the model.
//!
//! This is the geometry half of the bridge. The topology is worked out once on
//! the host and becomes a `gather` index; every iteration after that is a
//! compiled graph running over an `[n, 3]` tensor, and the mesh only comes
//! back when the smoothing is done.
//!
//! The two passes differ in what they preserve. Laplacian smoothing moves each
//! vertex towards its neighbours' average, which removes the noise *and*
//! deflates the shape — the printed radius says by how much. Taubin alternates
//! a positive step with a slightly larger negative one and keeps the volume,
//! which is why it is the one to reach for.

use threers::core::{BufferAttribute, BufferGeometry};
use threers::rlx::{mesh, preferred_device};
use threers::{
    encode_png, AmbientLight, Color, DirectionalLight, HeadlessRenderer, Mesh, Object3D,
    PerspectiveCamera, Scene, SphereGeometry, StandardMaterial, Vector3,
};

const SIZE: u32 = 600;
const NOISE: f32 = 0.09;

fn main() {
    std::fs::create_dir_all("out").ok();
    let device = preferred_device();
    println!("rlx device: {device:?}");

    let mut headless = match HeadlessRenderer::builder().size(SIZE, SIZE).build() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — needs a GPU adapter.");
            std::process::exit(2);
        }
    };
    let (width, height) = headless.render_size();

    let noisy = crinkled_sphere();
    println!("{:<10} {:>8} {:>10}", "mesh", "radius", "roughness");
    report("noisy", &noisy);
    shoot(&mut headless, width, height, &noisy, "noisy");

    let mut laplacian = noisy.clone();
    mesh::laplacian_smooth(&mut laplacian, 16, 0.5, device).expect("smooth");
    report("laplacian", &laplacian);
    shoot(&mut headless, width, height, &laplacian, "laplacian");

    let mut taubin = noisy.clone();
    mesh::taubin_smooth(&mut taubin, 8, 0.5, -0.53, device).expect("smooth");
    report("taubin", &taubin);
    shoot(&mut headless, width, height, &taubin, "taubin");
}

/// A sphere with every vertex pushed in or out along its own normal — noise
/// with no preferred direction, which is what smoothing is meant to remove.
fn crinkled_sphere() -> BufferGeometry {
    let mut geometry = SphereGeometry::new(1.0, 96, 48);
    let mut array = geometry.get_attribute("position").unwrap().array.clone();
    let mut rng = Lcg(0xc0ffee);
    for v in array.chunks_exact_mut(3) {
        let scale = 1.0 + NOISE * (rng.unit() - 0.5) * 2.0;
        v[0] *= scale;
        v[1] *= scale;
        v[2] *= scale;
    }
    geometry.set_attribute("position", BufferAttribute::new(array, 3));
    threers::compute_vertex_normals(&mut geometry);
    geometry
}

/// Mean radius (how much the pass deflated the shape) and mean neighbour-to-
/// neighbour radius gap (how much noise is left).
fn report(name: &str, geometry: &BufferGeometry) {
    let radii: Vec<f32> = geometry
        .positions()
        .unwrap()
        .map(|p| (p.x * p.x + p.y * p.y + p.z * p.z).sqrt())
        .collect();
    let mean = radii.iter().sum::<f32>() / radii.len() as f32;

    // Roughness measured against the index buffer's neighbours, not against
    // the global mean: a pass that shrinks the ball evenly has not made it
    // rougher, and a global measure would say it had.
    let index = geometry.index.as_ref().expect("indexed");
    let mut sum = vec![0.0f32; radii.len()];
    let mut count = vec![0u32; radii.len()];
    for tri in index.chunks_exact(3) {
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let (u, v) = (tri[a] as usize, tri[b] as usize);
            sum[u] += radii[v];
            count[u] += 1;
            sum[v] += radii[u];
            count[v] += 1;
        }
    }
    let (mut total, mut counted) = (0.0f32, 0u32);
    for v in 0..radii.len() {
        if count[v] > 0 {
            total += (radii[v] - sum[v] / count[v] as f32).abs();
            counted += 1;
        }
    }
    println!(
        "{name:<10} {mean:>8.4} {:>10.5}",
        total / counted.max(1) as f32
    );
}

fn shoot(
    headless: &mut HeadlessRenderer,
    width: u32,
    height: u32,
    geometry: &BufferGeometry,
    name: &str,
) {
    let mut scene = Scene::new();
    scene.background = Color::new(0.04, 0.05, 0.07);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.2));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 3.2)
            .with_direction(Vector3::new(-0.4, -0.7, -0.6).normalize()),
    );
    let mut material = StandardMaterial::new(Color::new(0.75, 0.72, 0.68));
    material.roughness = 0.32;
    material.metalness = 0.05;
    scene.add(Object3D::mesh(Mesh::new(geometry.clone(), material.into())));

    let mut camera = PerspectiveCamera::new(40.0, width as f32 / height as f32, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.9, 3.6);
    camera.look_at(Vector3::ZERO);

    let rgba = headless.render_to_rgba(&mut scene, &camera);
    let path = format!("out/rlx_mesh_{name}.png");
    std::fs::write(&path, encode_png(width, height, &rgba)).expect("write png");
    println!("  wrote {path}");
}

/// Deterministic noise, so the picture is the same every run.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as f32 / u32::MAX as f32
    }
}
