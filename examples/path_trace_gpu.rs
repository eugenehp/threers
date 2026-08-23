//! The same path tracer on the GPU, timed against the CPU.
//!
//! Both backends trace the same [`RaytraceScene`](threers::raytrace::RaytraceScene)
//! with the same settings, so the two images are the same image with different
//! noise — which is what the comparison at the end checks. What differs is how
//! long it takes.
//!
//! ```text
//! cargo run --release --example path_trace_gpu --features raytrace
//! cargo run --release --example path_trace_gpu --features raytrace,parallel   # fairer CPU
//! ```
//!
//! Writes `out/path_trace_gpu.png` and `out/path_trace_cpu.png`.

use std::f32::consts::FRAC_PI_2;

use threers::raytrace::{gpu::GpuBackend, CpuBackend, RaytraceRenderer, RaytraceSettings};
use threers::{
    encode_png, BoxGeometry, Color, DirectionalLight, Material, Mesh, Object3D, PerspectiveCamera,
    PhysicalMaterial, PlaneGeometry, Scene, SphereGeometry, StandardMaterial, ToneMapping, Vector3,
};

const W: u32 = 800;
const H: u32 = 500;
const SAMPLES: u32 = 128;

fn main() {
    std::fs::create_dir_all("out").expect("create out/");
    let camera = {
        let mut c = PerspectiveCamera::new(38.0, W as f32 / H as f32, 0.1, 200.0);
        c.position = Vector3::new(4.2, 2.6, 6.4);
        c.target = Vector3::new(0.0, 0.7, 0.0);
        c
    };

    let settings = RaytraceSettings {
        samples_per_pixel: SAMPLES,
        max_bounces: 6,
        min_bounces: 3,
        clamp_indirect: 10.0,
        tone_mapping: ToneMapping::AcesFilmic,
        denoise: false, // compare the raw estimates, not two smoothed versions
        ..Default::default()
    };

    // ---- GPU
    let gpu_ms = match GpuBackend::headless() {
        Ok(backend) => {
            // Bigger batches amortise submission overhead; smaller ones keep
            // each dispatch short enough not to trip the driver's watchdog on a
            // heavy scene. 8 is comfortable for a scene this size.
            let backend = backend.with_samples_per_dispatch(8);
            let mut r = RaytraceRenderer::with_backend(W, H, Box::new(backend));
            r.set_settings(settings.clone());
            let mut scene = build_scene();
            let t = std::time::Instant::now();
            r.render(&mut scene, &camera).expect("gpu trace");
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            std::fs::write(
                "out/path_trace_gpu.png",
                encode_png(W, H, &r.resolve_rgba()),
            )
            .expect("write png");
            let (lo, hi) = r.sample_range();
            println!(
                "gpu: {SAMPLES} spp in {ms:.0} ms  ({lo}-{hi} samples/pixel, {:.0}% of budget)  \
                 -> out/path_trace_gpu.png",
                r.sample_efficiency() * 100.0
            );
            for note in r
                .traced_scene()
                .into_iter()
                .flat_map(|s| s.report.approximated.iter())
            {
                println!("  note: {note}");
            }
            Some((ms, r.resolve_hdr()))
        }
        Err(e) => {
            println!("gpu: unavailable ({e})");
            None
        }
    };

    // ---- CPU
    let mut r = RaytraceRenderer::with_backend(W, H, Box::new(CpuBackend::new()));
    r.set_settings(settings);
    let mut scene = build_scene();
    let t = std::time::Instant::now();
    r.render(&mut scene, &camera).expect("cpu trace");
    let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;
    std::fs::write(
        "out/path_trace_cpu.png",
        encode_png(W, H, &r.resolve_rgba()),
    )
    .expect("write");
    let (lo, hi) = r.sample_range();
    println!(
        "cpu: {SAMPLES} spp in {cpu_ms:.0} ms  ({lo}-{hi} samples/pixel, {:.0}% of budget)  \
         -> out/path_trace_cpu.png",
        r.sample_efficiency() * 100.0
    );
    #[cfg(not(feature = "parallel"))]
    println!("  (single-threaded — build with --features raytrace,parallel for a fair comparison)");

    // ---- do they agree?
    if let Some((gpu_ms, gpu_hdr)) = gpu_ms {
        println!("speedup: {:.1}x", cpu_ms / gpu_ms.max(1e-6));
        let cpu_hdr = r.resolve_hdr();
        let mean = |v: &[f32], c: usize| {
            v.chunks_exact(4).map(|p| p[c] as f64).sum::<f64>() / (v.len() / 4) as f64
        };
        println!("frame means (linear, should match to a percent or so):");
        for (c, name) in ["red", "green", "blue"].iter().enumerate() {
            println!(
                "  {name:<6} gpu {:.4}   cpu {:.4}",
                mean(&gpu_hdr, c),
                mean(&cpu_hdr, c)
            );
        }
    }
}

/// A scene with something for every part of the kernel to do: a rough metal, a
/// glass ball, a clearcoated dielectric, an emissive panel, a directional light
/// with an angular size, and a floor to catch it all.
fn build_scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::new(0.02, 0.025, 0.04);

    let mut floor_mat = StandardMaterial::new(Color::new(0.55, 0.55, 0.58));
    floor_mat.roughness = 0.55;
    floor_mat.metalness = 0.0;
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(60.0, 60.0),
        Material::Standard(floor_mat),
    ));
    floor.rotate_x(-FRAC_PI_2);
    scene.add(floor);

    // Rough gold — the one that would render visibly dark without the
    // energy-compensation term in the BSDF.
    let mut gold = StandardMaterial::new(Color::new(1.0, 0.77, 0.34));
    gold.roughness = 0.42;
    gold.metalness = 1.0;
    let mut ball = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.9, 64, 48),
        Material::Standard(gold),
    ));
    ball.position = Vector3::new(-1.9, 0.9, 0.0);
    scene.add(ball);

    // Glass, with a tint that only shows up over distance travelled inside it.
    let mut glass = PhysicalMaterial::new(Color::WHITE);
    glass.transmission = 1.0;
    glass.roughness = 0.02;
    glass.ior = 1.52;
    glass.attenuation_color = Color::new(0.72, 0.92, 0.86);
    glass.attenuation_distance = 1.5;
    let mut sphere = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.9, 64, 48),
        Material::Physical(glass),
    ));
    sphere.position = Vector3::new(0.2, 0.9, 0.6);
    scene.add(sphere);

    // Clearcoat over a deep red base: two specular lobes, one broad and tinted,
    // one sharp and colourless.
    let mut lacquer = PhysicalMaterial::new(Color::new(0.5, 0.05, 0.06));
    lacquer.roughness = 0.45;
    lacquer.clearcoat = 1.0;
    lacquer.clearcoat_roughness = 0.06;
    let mut cube = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.4, 1.4, 1.4),
        Material::Physical(lacquer),
    ));
    cube.position = Vector3::new(2.2, 0.7, -0.4);
    cube.rotate_y(0.55);
    scene.add(cube);

    // An emissive strip: geometry, so it is sampled as an area light and shows
    // up in the reflections.
    let mut strip_mat = StandardMaterial::new(Color::BLACK);
    strip_mat.emissive = Color::new(0.45, 0.75, 1.0);
    strip_mat.emissive_intensity = 14.0;
    let mut strip = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(4.0, 0.35),
        Material::Standard(strip_mat),
    ));
    strip.position = Vector3::new(0.0, 2.6, -2.2);
    strip.rotate_x(-0.6);
    scene.add(strip);

    // A sun with an angular size, so its shadows have a penumbra. The real one
    // is 0.00465 rad; this is wider, for a softer look.
    let mut sun = Object3D::light(DirectionalLight::new(Color::new(1.0, 0.96, 0.9), 2.2));
    sun.position = Vector3::new(5.0, 8.0, 4.0);
    scene.add(sun);

    scene.add_light(threers::HemisphereLight::new(
        Color::new(0.35, 0.45, 0.7),
        Color::new(0.15, 0.13, 0.12),
        0.5,
    ));

    scene
}
