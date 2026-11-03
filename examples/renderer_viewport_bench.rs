//! Headless benchmark for interactive viewport renderer paths.
//!
//! Compares warm tracing vs camera-only updates (cached BVH) vs forced
//! invalidation (old behavior that repacked scene geometry every orbit).
//!
//! ```text
//! cargo run --release --example renderer_viewport_bench --features raytrace
//! cargo run --release --example renderer_viewport_bench --features raytrace -- --gpu
//! cargo run --release --example renderer_viewport_bench --features raytrace -- --gpu --width 1280 --height 720 --iters 20
//! ```

use std::time::Instant;

use threers::cameras::PerspectiveCamera;
use threers::raytrace::{gpu::GpuBackend, CpuBackend, RaytraceRenderer, RaytraceSettings};
use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, Material, Mesh, Object3D, PhysicalMaterial,
    PlaneGeometry, Scene, SphereGeometry, StandardMaterial, Vector3,
};

struct BenchOpts {
    width: u32,
    height: u32,
    iters: u32,
    gpu: bool,
    heavy: bool,
    tile_samples: u32,
    max_tiles: u32,
    budget_ms: u32,
    pack_iters: u32,
}

impl BenchOpts {
    fn from_args() -> Self {
        let mut width = 960u32;
        let mut height = 540u32;
        let mut iters = 15u32;
        let mut gpu = false;
        let mut heavy = false;
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--gpu" => gpu = true,
                "--heavy" => heavy = true,
                "--width" => width = args.next().and_then(|s| s.parse().ok()).unwrap_or(width),
                "--height" => height = args.next().and_then(|s| s.parse().ok()).unwrap_or(height),
                "--iters" => iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(iters),
                "--help" | "-h" => {
                    eprintln!(
                        "usage: renderer_viewport_bench [--gpu] [--heavy] [--width N] [--height N] [--iters N]"
                    );
                    std::process::exit(0);
                }
                other => eprintln!("unknown arg: {other}"),
            }
        }
        Self {
            width: width.max(1),
            height: height.max(1),
            iters: iters.max(1),
            gpu,
            heavy,
            tile_samples: 1,
            max_tiles: 16,
            budget_ms: 14,
            pack_iters: 10,
        }
    }
}

fn time_one_tile(renderer: &mut RaytraceRenderer, scene: &mut Scene, cam: &PerspectiveCamera) -> Result<f64, String> {
    renderer.prepare_if_changed(scene, cam);
    let t = Instant::now();
    renderer
        .accumulate_unconverged_tiles_budgeted(0, 1, 1, 0)
        .map_err(|e| e.to_string())?;
    Ok(t.elapsed().as_secs_f64() * 1000.0)
}

fn bench_pack_overhead(
    renderer: &mut RaytraceRenderer,
    scene: &mut Scene,
    aspect: f32,
    iters: u32,
) -> Result<(f64, f64), String> {
    let mut cached = Vec::with_capacity(iters as usize);
    let mut invalidated = Vec::with_capacity(iters as usize);
    for i in 0..iters {
        let cam = camera(aspect, 200 + i);
        cached.push(time_one_tile(renderer, scene, &cam)?);
    }
    for i in 0..iters {
        let cam = camera(aspect, 400 + i);
        renderer.prepare_if_changed(scene, &cam);
        renderer.backend_invalidate();
        let t = Instant::now();
        renderer
            .accumulate_unconverged_tiles_budgeted(0, 1, 1, 0)
            .map_err(|e| e.to_string())?;
        invalidated.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    Ok((median_ms(&mut cached), median_ms(&mut invalidated)))
}

fn build_scene(heavy: bool) -> Scene {
    let mut scene = build_viewport_scene();
    if heavy {
        for x in -4..=4 {
            for z in -4..=4 {
                if x == 0 && z == 0 {
                    continue;
                }
                let mut obj = Object3D::mesh(Mesh::new(
                    BoxGeometry::new(0.35, 0.35, 0.35),
                    Material::Standard(
                        StandardMaterial::new(Color::new(0.5 + (x as f32 * 0.05), 0.4, 0.55))
                            .with_roughness(0.4),
                    ),
                ));
                obj.position = Vector3::new(x as f32 * 0.9, 0.18, z as f32 * 0.9);
                scene.add(obj);
            }
        }
    }
    scene
}

fn build_viewport_scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::new(0.04, 0.05, 0.08);

    let floor = StandardMaterial::new(Color::new(0.55, 0.55, 0.58)).with_roughness(0.85);
    let mut floor_obj = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(14.0, 14.0),
        Material::Standard(floor),
    ));
    floor_obj.rotate_x(-std::f32::consts::FRAC_PI_2);
    scene.add(floor_obj);

    let glass = PhysicalMaterial::new(Color::new(0.92, 0.96, 1.0))
        .with_roughness(0.04)
        .with_metalness(0.0)
        .with_transmission(1.0, 1.45, 0.5);
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.55, 32, 24),
        Material::Physical(glass),
    )));

    let mut hero = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.2, 1.2, 1.2),
        Material::Standard(StandardMaterial::new(Color::new(0.78, 0.35, 0.18)).with_roughness(0.35)),
    ));
    hero.position = Vector3::new(-1.2, 0.6, 0.0);
    scene.add(hero);

    scene.add_light(AmbientLight::new(Color::new(0.25, 0.28, 0.32), 0.35));
    let mut key = Object3D::light(DirectionalLight::new(Color::new(1.0, 0.96, 0.88), 2.2));
    key.position = Vector3::new(4.0, 6.0, 3.0);
    scene.add(key);
    scene
}

fn camera(aspect: f32, orbit: u32) -> PerspectiveCamera {
    let angle = orbit as f32 * 0.07;
    let mut cam = PerspectiveCamera::new(42.0, aspect, 0.1, 100.0);
    cam.position = Vector3::new(
        4.5 * angle.cos(),
        2.8,
        5.5 * angle.sin() + 5.5,
    );
    cam.look_at(Vector3::new(0.0, 0.6, 0.0));
    cam
}

fn median_ms(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        samples[n / 2]
    } else {
        (samples[n / 2 - 1] + samples[n / 2]) * 0.5
    }
}

fn bench_renderer(opts: &BenchOpts) -> Result<(), String> {
    let aspect = opts.width as f32 / opts.height as f32;
    let settings = RaytraceSettings::preview()
        .with_samples(256)
        .with_denoise(false)
        .with_adaptive(0.02)
        .with_sample_redistribution(true);

    let mut renderer = if opts.gpu {
        let backend = GpuBackend::headless()
            .map_err(|e| format!("GPU unavailable: {e}"))?
            .with_samples_per_dispatch(4);
        RaytraceRenderer::with_backend(opts.width, opts.height, Box::new(backend))
    } else {
        RaytraceRenderer::with_backend(opts.width, opts.height, Box::new(CpuBackend::new()))
    };
    renderer.set_settings(settings);

    let mut scene = build_scene(opts.heavy);
    let mut cam = camera(aspect, 0);

    // Cold: first prepare + interactive step (includes BVH build + first GPU upload on GPU).
    let t0 = Instant::now();
    renderer.prepare(&mut scene, &cam);
    renderer
        .accumulate_interactive(
            &mut scene,
            &cam,
            0,
            opts.tile_samples,
            opts.max_tiles,
            opts.budget_ms,
        )
        .map_err(|e| e.to_string())?;
    let cold_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let gpu_cached = renderer.gpu_scene_cached();

    // Warm: additional trace, same camera.
    let mut warm = Vec::with_capacity(opts.iters as usize);
    for _ in 0..opts.iters {
        let t = Instant::now();
        renderer
            .accumulate_unconverged_tiles_budgeted(
                0,
                opts.tile_samples,
                opts.max_tiles,
                opts.budget_ms,
            )
            .map_err(|e| e.to_string())?;
        warm.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    // Camera orbit with cached scene (current behavior).
    let mut orbit_cached = Vec::with_capacity(opts.iters as usize);
    for i in 0..opts.iters {
        cam = camera(aspect, i + 1);
        renderer.prepare_if_changed(&mut scene, &cam);
        let t = Instant::now();
        renderer
            .accumulate_unconverged_tiles_budgeted(
                0,
                opts.tile_samples,
                opts.max_tiles,
                opts.budget_ms,
            )
            .map_err(|e| e.to_string())?;
        orbit_cached.push(t.elapsed().as_secs_f64() * 1000.0);
        if opts.gpu && !renderer.gpu_scene_cached() {
            return Err("GPU scene cache lost after camera-only update".into());
        }
    }

    // Camera orbit with forced invalidate (simulates pre-fix behavior).
    let mut orbit_invalidate = Vec::with_capacity(opts.iters as usize);
    for i in 0..opts.iters {
        cam = camera(aspect, i + 100);
        renderer.prepare_if_changed(&mut scene, &cam);
        renderer.backend_invalidate();
        let t = Instant::now();
        renderer
            .accumulate_unconverged_tiles_budgeted(
                0,
                opts.tile_samples,
                opts.max_tiles,
                opts.budget_ms,
            )
            .map_err(|e| e.to_string())?;
        orbit_invalidate.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let warm_med = median_ms(&mut warm);
    let orbit_cached_med = median_ms(&mut orbit_cached);
    let orbit_invalidate_med = median_ms(&mut orbit_invalidate);
    let speedup = orbit_invalidate_med / orbit_cached_med.max(1e-6);
    let (pack_cached, pack_invalidate) =
        bench_pack_overhead(&mut renderer, &mut scene, aspect, opts.pack_iters)?;
    let pack_speedup = pack_invalidate / pack_cached.max(1e-6);

    println!("backend: {}", renderer.backend_name());
    if opts.heavy {
        println!("scene: viewport + 80 extra boxes");
    }
    println!(
        "film: {}×{} · {} iters · tile batch {} spp × {} tiles · {} ms budget",
        opts.width,
        opts.height,
        opts.iters,
        opts.tile_samples,
        opts.max_tiles,
        opts.budget_ms
    );
    if opts.gpu {
        println!(
            "gpu scene cached after cold: {gpu_cached} · max film side {:?}",
            renderer.max_film_side()
        );
    }
    println!();
    println!("{:<28} {:>8.2} ms", "cold (prepare + 1 step)", cold_ms);
    println!("{:<28} {:>8.2} ms  (median)", "warm trace", warm_med);
    println!(
        "{:<28} {:>8.2} ms  (median)",
        "camera orbit (cached BVH)",
        orbit_cached_med
    );
    println!(
        "{:<28} {:>8.2} ms  (median)",
        "camera orbit (+ invalidate)",
        orbit_invalidate_med
    );
    println!(
        "{:<28} {:>8.2} ms  (median, 1 tile)",
        "orbit pack test (cached)",
        pack_cached
    );
    println!(
        "{:<28} {:>8.2} ms  (median, 1 tile)",
        "orbit pack test (invalidate)",
        pack_invalidate
    );
    println!();
    println!(
        "cached vs invalidate speedup: {:.2}× ({:.1}% faster)",
        speedup,
        (1.0 - orbit_cached_med / orbit_invalidate_med.max(1e-6)) * 100.0
    );
    println!(
        "1-tile pack overhead speedup: {:.2}× ({:.1}% faster)",
        pack_speedup,
        (1.0 - pack_cached / pack_invalidate.max(1e-6)) * 100.0
    );

    Ok(())
}

fn main() {
    let opts = BenchOpts::from_args();
    if let Err(e) = bench_renderer(&opts) {
        eprintln!("bench failed: {e}");
        std::process::exit(1);
    }
}
