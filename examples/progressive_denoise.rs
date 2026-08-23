//! **Progressive path tracing with the denoiser in the loop** — write a frame
//! after each batch of samples, denoised, so the image is usable long before
//! the sample budget is spent.
//!
//! A final render traces every sample and denoises once. A viewport cannot: it
//! has to show something immediately and improve it. The denoiser is a
//! post-process over whatever the film holds, so it can run at any sample count
//! — this renders in batches and filters the accumulated buffer each time,
//! which is what Cycles does in its viewport
//! (`intern/cycles/integrator/render_scheduler.cpp`).
//!
//! Run:
//! ```sh
//! cargo run --release --example progressive_denoise \
//!     --features raytrace,learned-denoise -- --weights out/fixed.bin
//! ```
//!
//! | flag | meaning |
//! |---|---|
//! | `--weights FILE` | trained denoiser; without it the frames are unfiltered |
//! | `--samples N` | total sample budget                          [64] |
//! | `--batch N` | samples between frames                           [1] |
//! | `--start N` | do not show or denoise before this sample        [1] |
//! | `--step N` | minimum samples between frames                    [1] |
//! | `--size WxH` | frame size                                 [512x512] |
//! | `--out DIR` | where the PNGs go             [out/progressive] |

use std::sync::Arc;

use threers::raytrace::{ProgressiveOptions, RaytraceRenderer, RaytraceSettings};
use threers::{
    encode_png, BoxGeometry, Color, Material, Mesh, Object3D, PerspectiveCamera, PhysicalMaterial,
    PlaneGeometry, Scene, SphereGeometry, StandardMaterial, ToneMapping, Vector3,
};

fn arg(flag: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let total: u32 = arg("--samples").and_then(|s| s.parse().ok()).unwrap_or(64);
    let batch: u32 = arg("--batch").and_then(|s| s.parse().ok()).unwrap_or(1);
    let start: u32 = arg("--start").and_then(|s| s.parse().ok()).unwrap_or(1);
    let step: u32 = arg("--step").and_then(|s| s.parse().ok()).unwrap_or(1);
    let (w, h) = arg("--size")
        .and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((512u32, 512u32));
    let out = arg("--out").unwrap_or_else(|| "out/progressive".into());
    let _ = std::fs::create_dir_all(&out);

    let mut renderer = RaytraceRenderer::new(w, h);
    renderer.set_settings(RaytraceSettings {
        samples_per_pixel: total,
        max_bounces: 5,
        min_bounces: 3,
        clamp_indirect: 100.0,
        adaptive_threshold: 0.0,
        denoise: false,
        tone_mapping: ToneMapping::AcesFilmic,
        ..Default::default()
    });

    let mut denoising = false;
    #[cfg(feature = "learned-denoise")]
    if let Some(path) = arg("--weights") {
        match threers::raytrace::learned::LearnedDenoiser::open(&path) {
            Ok(d) => {
                println!("denoiser {path}");
                renderer.set_learned_denoiser(Some(d));
                denoising = true;
            }
            Err(e) => eprintln!("{path}: {e}"),
        }
    }
    if !denoising {
        println!("no denoiser: frames will be unfiltered");
    }

    let (mut scene, camera) = build_scene(w as f32 / h as f32);
    println!("{w}x{h}, {total} spp in batches of {batch}, from sample {start} every {step}");

    let mut wrote = 0;
    renderer
        .render_progressive(
            &mut scene,
            &camera,
            &ProgressiveOptions {
                batch,
                start_sample: start,
                min_step: step,
                denoise: denoising,
            },
            |frame| {
                let path = format!("{out}/spp_{:04}.png", frame.samples);
                let _ = std::fs::write(&path, encode_png(w, h, frame.image));
                wrote += 1;
                println!(
                    "  {:>4}/{} spp{}  {path}",
                    frame.samples,
                    frame.total,
                    if frame.denoised { "  denoised" } else { "" }
                );
                true
            },
        )
        .expect("render");
    println!("wrote {wrote} frames to {out}/");
}

/// A glass ball, a metal ball and a rough block under one panel — the same
/// arrangement the comparison example uses, so the frames here can be read
/// against those.
fn build_scene(aspect: f32) -> (Scene, PerspectiveCamera) {
    let mut scene = Scene::new();
    scene.background = Color::new(0.015, 0.02, 0.03);

    let mut floor_mat = PhysicalMaterial::new(Color::new(0.55, 0.55, 0.58));
    floor_mat.roughness = 0.4;
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(24.0, 24.0),
        Material::Physical(floor_mat),
    ));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    scene.add(floor);

    let mut metal = PhysicalMaterial::new(Color::new(0.95, 0.78, 0.42));
    metal.metalness = 1.0;
    metal.roughness = 0.15;
    let mut ball = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.55, 64, 32),
        Material::Physical(metal),
    ));
    ball.position = Vector3::new(-0.85, 0.55, 0.1);
    scene.add(ball);

    let mut glass = PhysicalMaterial::new(Color::new(0.98, 0.99, 1.0));
    glass.transmission = 1.0;
    glass.roughness = 0.02;
    glass.ior = 1.5;
    let mut sphere = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.5, 64, 32),
        Material::Physical(glass),
    ));
    sphere.position = Vector3::new(0.55, 0.5, -0.35);
    scene.add(sphere);

    let mut chalk = PhysicalMaterial::new(Color::new(0.75, 0.25, 0.22));
    chalk.roughness = 0.9;
    let mut block = Object3D::mesh(Mesh::new(
        BoxGeometry::new(0.7, 0.9, 0.7),
        Material::Physical(chalk),
    ));
    block.position = Vector3::new(0.35, 0.45, 0.85);
    scene.add(block);

    let mut lamp_mat = StandardMaterial::new(Color::WHITE);
    lamp_mat.emissive = Color::new(1.0, 0.94, 0.85);
    lamp_mat.emissive_intensity = 60.0;
    let mut lamp = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.28, 24, 12),
        Material::Standard(lamp_mat),
    ));
    lamp.position = Vector3::new(1.7, 3.1, 1.4);
    scene.add(lamp);

    let mut camera = PerspectiveCamera::new(38.0, aspect, 0.05, 100.0);
    camera.position = Vector3::new(2.6, 1.7, 2.4);
    camera.look_at(Vector3::new(0.0, 0.55, 0.0));
    let _ = Arc::new(());
    (scene, camera)
}
