//! A graph applied to every frame of a clip (`--features rlx,video,native-codec`).
//!
//! ```text
//! cargo run --release --features "rlx,video,native-codec" --example rlx_video_grade
//! ```
//!
//! Writes `out/rlx_video_graded.gif` — an orbiting still life with a colour
//! grade applied by rlx between the renderer and the encoder.
//!
//! A single-frame example cannot show the thing that matters about
//! [`ColorGrade::filter`]: it compiles the graph once and runs it per frame.
//! Sixty frames is where a design that recompiled per call would be obvious,
//! and where the per-frame cost of the round trip through host memory is worth
//! reporting honestly — which the summary at the end does, against the cost of
//! rendering the frame in the first place.
//!
//! The grade here is fixed. `rlx_fit_grade` is the example that *learns* one.

use std::f32::consts::TAU;
use std::time::{Duration, Instant};

use threers::rlx::{preferred_device, ColorGrade};
use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, HeadlessRenderer, Mesh, Object3D,
    PerspectiveCamera, Scene, SphereGeometry, StandardMaterial, Vector3, VideoCodec,
    VideoExportEvent, VideoExporter,
};

const FRAMES: usize = 60;
const FPS: u32 = 20;

fn main() {
    std::fs::create_dir_all("out").ok();

    let mut headless = match HeadlessRenderer::builder().size(320, 240).build() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("headless renderer unavailable ({e}) — needs a GPU adapter.");
            std::process::exit(2);
        }
    };
    let (width, height) = headless.render_size();

    let mut scene = still_life();
    let mut camera = PerspectiveCamera::new(45.0, width as f32 / height as f32, 0.1, 100.0);

    // A cool, lifted look — the kind of thing `rlx_fit_grade` solves for.
    let grade = ColorGrade {
        matrix: [[0.88, 0.04, 0.10], [0.03, 0.86, 0.09], [0.02, 0.08, 0.95]],
        bias: [0.015, 0.02, 0.055],
    };

    let device = preferred_device();
    let compile_started = Instant::now();
    let mut filter = grade.filter(width, height, device);
    let compile = compile_started.elapsed();
    println!("compiled the grade for {device:?} in {compile:.1?}");

    let (mut render_time, mut filter_time) = (Duration::ZERO, Duration::ZERO);

    let exporter = VideoExporter::new("out/rlx_video_graded.gif")
        .size(width, height)
        .frames(FRAMES)
        .fps(FPS)
        .codec(VideoCodec::Gif)
        .gif_colors(128)
        .on_event(|event| {
            if let VideoExportEvent::Complete { output, .. } = event {
                eprintln!("\rwrote {output}");
            }
        });

    let result = exporter.export(|frame| {
        let angle = frame as f32 / FRAMES as f32 * TAU;
        camera.position = Vector3::new(4.2 * angle.sin(), 1.3, 4.2 * angle.cos());
        camera.look_at(Vector3::ZERO);

        let started = Instant::now();
        let rgba = headless.render_to_rgba(&mut scene, &camera);
        render_time += started.elapsed();

        let started = Instant::now();
        let (graded, _, _) = filter.apply(&rgba, width, height).expect("grade");
        filter_time += started.elapsed();
        graded
    });

    if let Err(e) = result {
        eprintln!("export failed: {e}");
        std::process::exit(1);
    }

    println!(
        "\n{FRAMES} frames at {width}×{height}\n  render {:>8.1?} total, {:>7.2?} each\n  grade  {:>8.1?} total, {:>7.2?} each",
        render_time,
        render_time / FRAMES as u32,
        filter_time,
        filter_time / FRAMES as u32,
    );
}

fn still_life() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::new(0.06, 0.07, 0.10);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.3));
    scene.add_light(
        DirectionalLight::new(Color::WHITE, 2.6)
            .with_direction(Vector3::new(-0.4, -0.8, -0.45).normalize()),
    );

    let mut warm = StandardMaterial::new(Color::new(0.85, 0.4, 0.15));
    warm.roughness = 0.35;
    let mut ball = Object3D::mesh(Mesh::new(SphereGeometry::new(0.85, 48, 24), warm.into()));
    ball.position = Vector3::new(-0.9, 0.0, 0.0);
    scene.add(ball);

    let mut cool = StandardMaterial::new(Color::new(0.3, 0.5, 0.62));
    cool.metalness = 0.4;
    cool.roughness = 0.4;
    let mut cube = Object3D::mesh(Mesh::new(BoxGeometry::new(1.2, 1.2, 1.2), cool.into()));
    cube.position = Vector3::new(1.0, -0.2, 0.0);
    scene.add(cube);

    scene
}
