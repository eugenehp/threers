//! Interactive viewport-style path tracing: rebuild the scene only when it
//! changes, reuse the BVH across camera moves, and resolve progressive frames.

use threers::cameras::PerspectiveCamera;
use threers::core::Object3D;
use threers::geometries::BoxGeometry;
use threers::lights::AmbientLight;
use threers::materials::{Material, StandardMaterial};
use threers::math::{Color, Vector3};
use threers::raytrace::{ProgressiveOptions, RaytraceRenderer, RaytraceSettings};
use threers::scene::Scene;

fn main() {
    let mut scene = Scene::new();
    scene.background = Color::new(0.04, 0.05, 0.08);
    scene.add(Object3D::mesh(threers::core::Mesh::new(
        BoxGeometry::new(2.0, 2.0, 2.0),
        Material::Standard(StandardMaterial::new(Color::new(0.75, 0.35, 0.2))),
    )));
    scene.add_light(AmbientLight::new(Color::WHITE, 0.8));

    let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(3.0, 2.5, 4.0);
    cam.target = Vector3::ZERO;

    let mut renderer = RaytraceRenderer::new(640, 480);
    renderer.set_settings(
        RaytraceSettings::preview()
            .with_samples(128)
            .with_denoise(true)
            .with_adaptive(0.02)
            .with_sample_redistribution(true),
    );

    let mut frame = 0u32;
    renderer
        .render_progressive(
            &mut scene,
            &cam,
            &ProgressiveOptions::default()
                .with_batch(4)
                .with_start_sample(4)
                .with_min_step(4)
                .with_tile_size(0)
                .with_max_tiles_per_batch(16)
                .with_max_ms(50)
                .with_autosave(32, "viewport_path_trace.rtfc")
                .with_stop_when_converged(0.95),
            |update| {
                frame += 1;
                eprintln!(
                    "frame {frame}: {} / {} spp, eff {:.0}%, converged {:.0}%, tiles {} — {}",
                    update.samples,
                    update.total,
                    update.efficiency * 100.0,
                    update.converged_fraction * 100.0,
                    update.unconverged_tile_count,
                    update.report_summary.as_deref().unwrap_or(""),
                );
                true
            },
        )
        .expect("render");

    // Orbit without rebuilding the BVH — only unconverged tiles are retraced:
    cam.position = Vector3::new(4.0, 3.0, 3.0);
    renderer.prepare_if_changed(&mut scene, &cam);
    while renderer.needs_more_samples() {
        renderer.accumulate_unconverged_tiles(32, 4).expect("accumulate");
    }

    let rgba = renderer.resolve_rgba();
    let out = "out/viewport_path_trace.png";
    if let Some(dir) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(out, threers::encode_png(640, 480, &rgba)).expect("write png");
}
