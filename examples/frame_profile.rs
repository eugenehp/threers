//! Where a planet frame's GPU time goes, by bisection.
//!
//! ```text
//! cargo run --release --features planet --example frame_profile
//! cargo run --release --features planet --example frame_profile -- --size 2560x1440
//! ```
//!
//! There are no timestamp queries here. Each configuration renders the same
//! scene with one thing removed and the difference is that thing's cost —
//! readback, buffer setup and the rest are identical across configurations, so
//! they subtract out. Crude, but it needs no device features and it answers the
//! question actually being asked, which is what to cut.
//!
//! The first frame is discarded: it uploads every texture and compiles every
//! pipeline, and is not what anyone means by a frame time.

use std::time::Instant;

use threers::planet::{Atmosphere, EarthTextures, Planet, Starfield};
use threers::{
    AmbientLight, Color, DirectionalLight, HeadlessRenderer, PerspectiveCamera, Scene, ToneMapping,
    Vector3,
};

fn arg(flag: &str) -> Option<String> {
    std::env::args()
        .position(|a| a == flag)
        .and_then(|i| std::env::args().nth(i + 1))
}

struct Config {
    name: &'static str,
    clouds: bool,
    air: bool,
    aurora: bool,
    sky: bool,
    moon: bool,
}

fn main() {
    let assets = arg("--assets").unwrap_or_else(|| "web/assets/earth".into());
    let (w, h) = arg("--size")
        .and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((1600u32, 900u32));
    let frames: usize = arg("--frames").and_then(|v| v.parse().ok()).unwrap_or(24);

    let src = EarthTextures::from_dir(&assets);
    let maps = src.load();
    let moon_maps = src.load_moon().unwrap_or_default();
    let sky = src.load_starfield();

    let mut r = HeadlessRenderer::builder()
        .size(w, h)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .expect("no GPU adapter");
    r.renderer().set_tone_mapping(ToneMapping::AcesFilmic, 1.0);

    let build = |c: &Config| -> Scene {
        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        scene.add_light(
            DirectionalLight::new(Color::WHITE, 3.1)
                .with_direction(Vector3::new(-1.0, -0.18, -0.55).normalize()),
        );
        scene.add_light(AmbientLight::new(Color::from_hex(0x0a1428), 0.32));
        let mut earth_maps = maps.clone();
        if !c.clouds {
            earth_maps.clouds = None;
        }
        Planet::earth()
            .maps(earth_maps)
            .segments(256, 128)
            .atmosphere(c.air.then(|| Atmosphere {
                aurora: if c.aurora { 2.2 } else { 0.0 },
                ..Atmosphere::earth()
            }))
            .add_to(&mut scene);
        if c.moon {
            Planet::moon()
                .maps(moon_maps.clone())
                .position(Vector3::new(6.0, 0.9, 3.0))
                .add_to(&mut scene);
        }
        if c.sky {
            Starfield::new(sky.clone())
                .faceted(0)
                .radius(320.0)
                .intensity(0.9)
                .add_to(&mut scene);
        }
        scene
    };

    let mut camera = PerspectiveCamera::new(40.0, w as f32 / h as f32, 0.01, 500.0);
    camera.position = Vector3::new(2.4, 0.5, 2.6);
    camera.look_at(Vector3::ZERO);

    let configs = [
        Config {
            name: "everything",
            clouds: true,
            air: true,
            aurora: true,
            sky: true,
            moon: true,
        },
        Config {
            name: "no aurora",
            clouds: true,
            air: true,
            aurora: false,
            sky: true,
            moon: true,
        },
        Config {
            name: "no atmosphere",
            clouds: true,
            air: false,
            aurora: false,
            sky: true,
            moon: true,
        },
        Config {
            name: "no clouds",
            clouds: false,
            air: true,
            aurora: false,
            sky: true,
            moon: true,
        },
        Config {
            name: "no sky",
            clouds: true,
            air: true,
            aurora: false,
            sky: false,
            moon: true,
        },
        Config {
            name: "no moon",
            clouds: true,
            air: true,
            aurora: false,
            sky: true,
            moon: false,
        },
        Config {
            name: "surface only",
            clouds: false,
            air: false,
            aurora: false,
            sky: false,
            moon: false,
        },
    ];

    println!("{w}x{h}, {frames} frames each\n");
    println!("{:<16} {:>9} {:>9}", "configuration", "ms/frame", "vs full");
    let mut full = 0.0f64;
    for c in &configs {
        let mut scene = build(c);
        // Warm: first frame uploads every texture and builds every pipeline.
        let _ = r.render_to_rgba(&mut scene, &camera);
        let t0 = Instant::now();
        for _ in 0..frames {
            let _ = r.render_to_rgba(&mut scene, &camera);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / frames as f64;
        if c.name == "everything" {
            full = ms;
            println!("{:<16} {ms:>9.2} {:>9}", c.name, "—");
        } else {
            println!("{:<16} {ms:>9.2} {:>+9.2}", c.name, ms - full);
        }
    }

    // Spikes: the same configuration frame by frame, so a periodic stall shows.
    let mut scene = build(&configs[0]);
    let _ = r.render_to_rgba(&mut scene, &camera);
    let mut times: Vec<f64> = Vec::with_capacity(frames * 2);
    for _ in 0..frames * 2 {
        let t = Instant::now();
        let _ = r.render_to_rgba(&mut scene, &camera);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "\nspikes over {} frames: p50 {:.2}ms  p95 {:.2}ms  max {:.2}ms  ({:.1}x p50)",
        times.len(),
        sorted[sorted.len() / 2],
        sorted[(sorted.len() as f64 * 0.95) as usize],
        sorted[sorted.len() - 1],
        sorted[sorted.len() - 1] / sorted[sorted.len() / 2],
    );
}
