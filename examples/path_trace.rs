//! A Cornell box, path traced.
//!
//! The Cornell box is the reference scene for global illumination for a reason:
//! everything interesting in it is *indirect*. The light is a small panel in the
//! ceiling, so the walls are lit almost entirely by bounce; the red and green
//! walls tint the white ones beside them (colour bleeding); the ceiling is lit
//! only by light that has come back up off the floor. A rasteriser with a
//! shadow map renders this scene as a grey room with a hard-edged pool of light,
//! and nothing you can do to it will produce the bleed — there is no light in
//! the pipeline that has bounced.
//!
//! ```text
//! cargo run --release --example path_trace --features raytrace
//! cargo run --release --example path_trace --features raytrace,parallel   # multi-core
//!
//! SPP=64 cargo run --release --example path_trace --features raytrace      # quick look
//! DENOISE=0 cargo run --release --example path_trace --features raytrace   # raw estimate
//! ```
//!
//! Writes `out/path_trace.png`, plus the albedo, normal and depth channels.

use std::f32::consts::FRAC_PI_2;

use threers::raytrace::{Aov, RaytraceRenderer, RaytraceSettings};
use threers::{
    encode_png, BoxGeometry, Color, Material, Mesh, Object3D, PerspectiveCamera, PhysicalMaterial,
    PlaneGeometry, Scene, SphereGeometry, StandardMaterial, ToneMapping, Vector3,
};

const W: u32 = 640;
const H: u32 = 640;

fn main() {
    let mut scene = build_scene();

    // The camera sits at the open front of the box looking straight in. A 40
    // degree field of view is roughly what the original measurements imply.
    let mut camera = PerspectiveCamera::new(40.0, W as f32 / H as f32, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 1.0, 3.9);
    camera.target = Vector3::new(0.0, 1.0, 0.0);

    let mut renderer = RaytraceRenderer::new(W, H);
    renderer.set_settings(RaytraceSettings {
        samples_per_pixel: env_u32("SPP", 512),
        // Enough for the light to reach the ceiling: down to the floor, back up
        // to the ceiling, and a couple more to fill the corners.
        max_bounces: 8,
        min_bounces: 4,
        // A small bright emitter found through a glossy chain produces the odd
        // enormous sample; clamping trades a fraction of a percent of energy
        // for not having permanent white dots.
        clamp_indirect: 8.0,
        tone_mapping: ToneMapping::AcesFilmic,
        exposure: 1.0,
        denoise: env_u32("DENOISE", 1) != 0,
        ..Default::default()
    });

    std::fs::create_dir_all("out").expect("create out/");

    // Render progressively so there is something to look at while it works, and
    // so the console says how far along it is.
    renderer.prepare(&mut scene, &camera);
    if let Some(report) = renderer.report() {
        println!("scene: {}", report.summary());
        for note in &report.approximated {
            println!("  note: {note}");
        }
    }

    let total = renderer.settings().samples_per_pixel;
    let batch = 32;
    let start = std::time::Instant::now();
    let mut done = 0;
    while done < total {
        let n = batch.min(total - done);
        renderer.accumulate(n).expect("trace");
        done += n;
        println!(
            "  {done:>4}/{total} samples   {:.1}s",
            start.elapsed().as_secs_f32()
        );
    }

    let rgba = renderer.resolve_rgba();
    std::fs::write("out/path_trace.png", encode_png(W, H, &rgba)).expect("write png");
    let (lo, hi) = renderer.sample_range();
    println!(
        "wrote out/path_trace.png in {:.1}s  (adaptive: {lo}-{hi} samples per pixel, \
         {:.0}% of the budget spent)",
        start.elapsed().as_secs_f32(),
        renderer.sample_efficiency() * 100.0
    );

    // The auxiliary channels come free — the tracer already knew all three at
    // the first hit — and they are what the denoiser is steering by.
    for (aov, name) in [
        (Aov::Albedo, "albedo"),
        (Aov::Normal, "normal"),
        (Aov::Depth, "depth"),
    ] {
        let mut settings = renderer.settings().clone();
        settings.aov = aov;
        let mut aux = RaytraceRenderer::new(W, H);
        aux.set_settings(settings);
        let rgba = aux.render_to_rgba(&mut scene, &camera);
        std::fs::write(
            format!("out/path_trace_{name}.png"),
            encode_png(W, H, &rgba),
        )
        .expect("write png");
    }
    println!("wrote out/path_trace_{{albedo,normal,depth}}.png");
}

/// Read an override from the environment, for trying the knobs without an edit.
fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The box: five walls, a ceiling panel that emits, and two objects inside.
fn build_scene() -> Scene {
    let mut scene = Scene::new();
    // Nothing outside the box contributes: no ambient, no environment, and a
    // black background so the open front reads as open.
    scene.background = Color::BLACK;

    let white = || Material::Standard(StandardMaterial::new(Color::new(0.73, 0.73, 0.73)));
    let red = || Material::Standard(StandardMaterial::new(Color::new(0.65, 0.05, 0.05)));
    let green = || Material::Standard(StandardMaterial::new(Color::new(0.12, 0.45, 0.15)));

    let s = 2.0; // the box is 2 x 2 x 2, centred on (0, 1, 0)

    // Floor.
    let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(s, s), white()));
    floor.rotate_x(-FRAC_PI_2);
    scene.add(floor);

    // Ceiling.
    let mut ceiling = Object3D::mesh(Mesh::new(PlaneGeometry::new(s, s), white()));
    ceiling.position = Vector3::new(0.0, s, 0.0);
    ceiling.rotate_x(FRAC_PI_2);
    scene.add(ceiling);

    // Back wall.
    let mut back = Object3D::mesh(Mesh::new(PlaneGeometry::new(s, s), white()));
    back.position = Vector3::new(0.0, s * 0.5, -s * 0.5);
    scene.add(back);

    // Left wall, red.
    let mut left = Object3D::mesh(Mesh::new(PlaneGeometry::new(s, s), red()));
    left.position = Vector3::new(-s * 0.5, s * 0.5, 0.0);
    left.rotate_y(FRAC_PI_2);
    scene.add(left);

    // Right wall, green.
    let mut right = Object3D::mesh(Mesh::new(PlaneGeometry::new(s, s), green()));
    right.position = Vector3::new(s * 0.5, s * 0.5, 0.0);
    right.rotate_y(-FRAC_PI_2);
    scene.add(right);

    // The light: a small emissive panel just below the ceiling. It is geometry,
    // not a `Light` — which is what makes its shadows soft and its reflection
    // visible in the metal ball. `emissive_intensity` is radiance, so a small
    // panel needs a large number to fill a room.
    let mut lamp_material = StandardMaterial::new(Color::BLACK);
    lamp_material.emissive = Color::new(1.0, 0.92, 0.78);
    lamp_material.emissive_intensity = 22.0;
    let mut lamp = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(0.6, 0.6),
        Material::Standard(lamp_material),
    ));
    lamp.position = Vector3::new(0.0, s - 0.01, 0.0);
    lamp.rotate_x(FRAC_PI_2);
    scene.add(lamp);

    // A tall box, rotated — the classic occluder, and what casts the soft
    // shadow that shows the light has area.
    let mut tall = StandardMaterial::new(Color::new(0.75, 0.75, 0.72));
    tall.roughness = 0.85;
    let mut block = Object3D::mesh(Mesh::new(
        BoxGeometry::new(0.6, 1.2, 0.6),
        Material::Standard(tall),
    ));
    block.position = Vector3::new(-0.35, 0.6, -0.3);
    block.rotate_y(0.3);
    scene.add(block);

    // A glass sphere, for refraction and the caustic under it.
    let mut glass = PhysicalMaterial::new(Color::WHITE);
    glass.transmission = 1.0;
    glass.roughness = 0.0;
    glass.ior = 1.52;
    glass.attenuation_color = Color::new(0.85, 0.95, 0.9);
    glass.attenuation_distance = 1.0;
    let mut ball = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.35, 64, 48),
        Material::Physical(glass),
    ));
    ball.position = Vector3::new(0.42, 0.35, 0.25);
    scene.add(ball);

    scene
}
