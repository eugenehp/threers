//! Earth, as a video: the `realistic_earth` scene driven through the sequence
//! pipeline.
//!
//! ```text
//! cargo run --release --features planet,video --example earth_video
//! cargo run --release --features planet,video --example earth_video -- \
//!     --size 1920x1080 --frames 300 --seconds 10
//! ```
//!
//! | Flag | Effect |
//! |------|--------|
//! | `--frames N` | frames in the orbit (default 120) |
//! | `--seconds N` | shot length; sets the frame rate with `--frames` |
//! | `--size WxH` | resolution (default 1280x720) |
//! | `--assets DIR` | NASA imagery (default `web/assets/earth`) |
//! | `--out DIR` | where everything goes (default `out/earth_video`) |
//! | `--texture N` | procedural fallback map width (default 2048) |
//! | `--supersample N` | render at N times the output and downsample (default 2) |
//! | `--keep-frames` | keep the PNG stills after encoding |
//! | `--procedural` | ignore the assets directory |
//! | `--no-moon` | leave the Moon out |
//!
//! Run `scripts/fetch-earth-textures.sh --png --sky` first for the real
//! imagery; without it every map is generated and the shot still renders.
//!
//! # What makes a *video* of this look better than a still of it
//!
//! **Supersampling, not resolution.** A planet is mostly a long curved
//! edge against black, and an encoder spends its bitrate on edges. Rendering at
//! twice the output and downsampling costs four times the pixels and removes the
//! aliasing that would otherwise crawl along the limb for the whole shot — which
//! is both the ugliest artefact here and the most expensive one to encode.
//!
//! **A frame rate chosen from the motion.** `--seconds` sets the rate from the
//! frame count so the orbit takes the time it should, rather than leaving 120
//! frames to run at whatever the default is and turning a slow drift into a
//! spin.
//!
//! **Stills kept alongside.** The shot costs minutes; a re-encode costs
//! seconds. Keeping the frames means a change of codec, frame rate or grade
//! never re-renders anything.
//!
//! For the path-traced equivalent — where the interesting problem is noise
//! rather than aliasing — see `path_trace_video`.

use std::f32::consts::TAU;

use threers::planet::{
    earthshine, tidal_lock_yaw, Atmosphere, EarthTextures, Planet, PlanetMaps, Starfield,
};
use threers::{
    render_sequence, AmbientLight, Color, DirectionalLight, HeadlessRenderer, Material, Mesh,
    Object3D, PerspectiveCamera, Quaternion, Scene, SequenceOptions, SphereGeometry,
    StandardMaterial, ToneMapping, Vector3,
};

fn arg(flag: &str) -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == flag {
            return args.next();
        }
    }
    None
}
fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

const MOON_DISTANCE: f32 = 7.5;
const SUN_DISTANCE: f32 = 140.0;

fn moon_orbit(t: f32) -> Vector3 {
    let a = 3.2 + t * TAU * 0.15;
    Vector3::new(a.cos() * MOON_DISTANCE, 0.9, a.sin() * MOON_DISTANCE)
}

fn main() {
    let assets = arg("--assets").unwrap_or_else(|| "web/assets/earth".into());
    let out = arg("--out").unwrap_or_else(|| "out/earth_video".into());
    let frames: usize = arg("--frames")
        .and_then(|s| s.parse().ok())
        .unwrap_or(120)
        .max(1);
    let (w, h) = arg("--size")
        .and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((1280u32, 720u32));
    let tex: u32 = arg("--texture")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048)
        .clamp(256, 8192);
    let supersample: u32 = arg("--supersample")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2)
        .clamp(1, 4);
    // A frame count on its own says nothing about how fast the shot moves. Given
    // a duration, the rate follows from it.
    let fps = match arg("--seconds").and_then(|s| s.parse::<f32>().ok()) {
        Some(secs) if secs > 0.0 => ((frames as f32 / secs).round() as u32).max(1),
        _ => 30,
    };
    let want_moon = !flag("--no-moon");
    let keep_frames = flag("--keep-frames");

    let started = std::time::Instant::now();
    let textures = EarthTextures::from_dir(&assets).fallback_width(tex);
    let (earth_maps, sources) = if flag("--procedural") {
        println!("generating {tex}x{} maps…", tex / 2);
        (
            threers::planet::generate_earth_maps(tex),
            Default::default(),
        )
    } else {
        println!("loading maps from {assets}…");
        textures.load_reporting()
    };
    for name in &sources.loaded {
        println!("  loaded    {name}");
    }
    for (name, why) in &sources.failed {
        println!("  FAILED    {name}: {why}");
    }
    if !sources.generated.is_empty() {
        println!("  generated {}", sources.generated.join(", "));
    }

    // ---- scene ----
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x01020a);

    let earth = Planet::earth()
        .maps(earth_maps)
        .segments(256, 128)
        .atmosphere(Some(Atmosphere {
            aurora: 2.2,
            ..Atmosphere::earth()
        }))
        .add_to(&mut scene);

    let moon = want_moon.then(|| {
        let maps = if flag("--procedural") {
            None
        } else {
            textures.load_moon()
        };
        Planet::moon()
            .maps(maps.unwrap_or_else(PlanetMaps::new))
            .color(Color::from_hex(0x8f8b85))
            .displacement_centered(0.2727 * 0.03)
            .position(moon_orbit(0.0))
            .add_to(&mut scene)
    });

    let (sky, _) = if flag("--procedural") {
        (
            threers::planet::generate_starfield(2048, 12_000),
            Default::default(),
        )
    } else {
        textures.load_starfield_reporting()
    };
    Starfield::new(sky)
        .faceted(0)
        .radius(320.0)
        .intensity(0.9)
        .add_to(&mut scene);

    let sun_dir = Vector3::new(1.0, 0.16, 0.52).normalize();
    let mut sun_mat = StandardMaterial::new(Color::from_hex(0xfff6e0));
    sun_mat.emissive = Color::from_hex(0xfff4d6);
    sun_mat.emissive_intensity = 90.0;
    let mut sun = Object3D::mesh(Mesh::new(
        SphereGeometry::new(2.4, 48, 24),
        Material::Standard(sun_mat),
    ));
    sun.position = sun_dir * SUN_DISTANCE;
    scene.add(sun);

    scene.add_light(
        DirectionalLight::new(Color::from_hex(0xfff6ea), 3.1).with_direction(sun_dir * -1.0),
    );
    {
        let (dir, strength) = earthshine(sun_dir, Vector3::ZERO, moon_orbit(0.0), 1.0, 0.30);
        scene.add_light(
            DirectionalLight::new(Color::from_hex(0x7fa6d8), strength * 260.0).with_direction(dir),
        );
    }
    scene.add_light(AmbientLight::new(Color::from_hex(0x0a1428), 0.32));

    // ---- render ----
    let mut renderer = HeadlessRenderer::builder()
        .size(w, h)
        .supersample(supersample)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .high_resolution(true)
        .build()
        .expect("headless renderer (is a GPU adapter available?)");
    renderer
        .renderer()
        .set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
    // `render_to_rgba_resolved` downsamples, so its frames come back at the
    // output size — the supersampled size is internal and only shows up in the
    // render cost.
    let (rw, rh) = renderer.render_size();
    let mut camera = PerspectiveCamera::new(40.0, w as f32 / h as f32, 0.01, 500.0);

    println!(
        "{w}x{h} from {rw}x{rh}  {frames} frames at {fps} fps ({:.1}s)",
        frames as f32 / fps as f32
    );

    let mut opts = SequenceOptions::new(frames)
        .fps(fps)
        .video_to(format!("{out}/earth.mp4"))
        // A planet against black is mostly flat gradient — sky, terminator,
        // atmosphere — and that is what a mid-range CRF bands into visible
        // steps. 16 keeps them smooth.
        .crf(16);
    if keep_frames {
        opts = opts.frames_to(format!("{out}/frames")).frame_stem("earth");
    }

    let result = render_sequence(w, h, &opts, |i| {
        let t = i as f32 / frames as f32;
        let yaw = 0.6 + t * TAU;
        camera.position = Vector3::new(
            yaw.cos() * 3.4,
            0.5 + (t * TAU).sin() * 0.3,
            yaw.sin() * 3.4,
        );
        camera.look_at(Vector3::ZERO);

        let spin = |o: &mut Object3D, turns: f32| {
            o.quaternion = Quaternion::from_euler_xyz(23.44f32.to_radians(), t * TAU * turns, 0.0);
        };
        if let Some(o) = scene.get_mut(earth.surface) {
            spin(o, 0.35);
        }
        if let Some(o) = earth.clouds.and_then(|id| scene.get_mut(id)) {
            spin(o, 0.42);
        }
        if let Some(m) = moon {
            if let Some(o) = scene.get_mut(m.surface) {
                o.position = moon_orbit(t);
                o.quaternion = Quaternion::from_euler_xyz(0.0, tidal_lock_yaw(moon_orbit(t)), 0.0);
            }
        }
        renderer.render_to_rgba_resolved(&mut scene, &camera)
    });

    match result {
        Ok(report) => {
            println!(
                "wrote {out}/earth.mp4 — {} frames, {:.2}s/frame, {:.1}s total",
                report.frames,
                report.seconds_per_frame(),
                started.elapsed().as_secs_f64()
            );
            if keep_frames {
                println!("  stills in {out}/frames");
            }
        }
        Err(e) => eprintln!("sequence failed: {e}"),
    }
}
