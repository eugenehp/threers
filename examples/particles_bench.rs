//! What a frame of particles costs, on the CPU: the browser demo's two
//! scenes stepped by a `ParticleAnimator` without drawing, timed in thread
//! CPU time over a few hundred frames after a warm-up.
//!
//! With `draw`, what drawing the winter camp costs on the GPU instead, at
//! 1280 × 720 with 4× MSAA and the sun's shadow, and with each piece left
//! out in turn — the configurations interleaved, so a busy machine slows
//! them all alike.
//!
//! ```bash
//! cargo run --release --example particles_bench
//! cargo run --release --example particles_bench -- draw
//! ```

use std::time::Instant;
use threers::core::ObjectId;
use threers::core::{Mesh, Object3D};
use threers::geometries::{BoxGeometry, PlaneGeometry};
use threers::materials::{BasicMaterial, Material};
use threers::math::{Color, Vector3};
use threers::particles::sand::{Downwind, SandBed};
use threers::particles::snow::SnowCover;
use threers::particles::{presets, Heightfield, ParticleAnimator};
use threers::Scene;

/// This thread's CPU time, seconds: on a busy machine, wall-clock time
/// counts every moment another process had the core.
#[cfg(unix)]
fn cpu_seconds() -> f64 {
    #[repr(C)]
    struct Timespec {
        sec: i64,
        nsec: i64,
    }
    extern "C" {
        fn clock_gettime(clock: i32, t: *mut Timespec) -> i32;
    }
    const THREAD_CPU: i32 = if cfg!(target_os = "macos") { 16 } else { 3 };
    let mut t = Timespec { sec: 0, nsec: 0 };
    // SAFETY: `t` is a valid, writable timespec.
    unsafe { clock_gettime(THREAD_CPU, &mut t) };
    t.sec as f64 + t.nsec as f64 * 1e-9
}
#[cfg(not(unix))]
fn cpu_seconds() -> f64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// The 10th percentile: on a shared machine, the frames nothing else
/// interrupted.
fn quick(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 10]
}

fn bench(name: &str, scene: &mut Scene, anim: &mut ParticleAnimator, warm: f32) {
    let dt = 1.0 / 60.0;
    for _ in 0..(warm / dt) as usize {
        anim.update(scene, dt, Some(Vector3::new(4.0, 3.0, 7.0)));
    }
    let mut times = Vec::new();
    for _ in 0..240 {
        let t = cpu_seconds();
        anim.update(scene, dt, Some(Vector3::new(4.0, 3.0, 7.0)));
        times.push((cpu_seconds() - t) * 1000.0);
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
    println!(
        "{name}: {:.2} ms of CPU a frame (10th percentile), {mean:.2} mean; 50/75/90/100%: {:.2} {:.2} {:.2} {:.2}",
        quick(times),
        at(0.5),
        at(0.75),
        at(0.9),
        at(1.0)
    );
}

fn ground(scene: &mut Scene) {
    let mut g = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(80.0, 80.0),
        Material::Basic(BasicMaterial::new(Color::WHITE)),
    ));
    g.rotate_x(-std::f32::consts::FRAC_PI_2);
    scene.add(g);
}

/// The winter camp as the browser demo draws it, timed frame by frame.
fn draw() {
    use threers::lights::Light;
    use threers::materials::StandardMaterial;
    use threers::particles::Weather;
    use threers::prelude::{
        Camera, DirectionalLight, FogParams, HeadlessRenderer, HemisphereLight, PerspectiveCamera,
        ToneMapping,
    };
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x9aa7b8);
    scene.fog = FogParams {
        color: Color::from_hex(0x9aa7b8),
        density: 0.045,
        mode: 2,
        ..FogParams::default()
    };
    scene.add_light(HemisphereLight::new(
        Color::from_hex(0xdfe8f5),
        Color::from_hex(0x6a6f78),
        1.0,
    ));
    let mut sun = DirectionalLight::new(Color::from_hex(0xfff1e0), 1.3);
    sun.cast_shadow = true;
    sun.shadow.map_size = 2048;
    sun.shadow.camera_size = 9.0;
    let mut l = Object3D::light(Light::Directional(sun));
    l.position = Vector3::new(-5.0, 8.0, 4.0);
    let sun = scene.add(l);
    let solid = |scene: &mut Scene, g: threers::BufferGeometry, hex: u32, at: Vector3| {
        let mut o = Object3D::mesh(Mesh::new(
            g,
            Material::Standard(StandardMaterial::new(Color::from_hex(hex))),
        ));
        o.position = at;
        o.cast_shadow = true;
        o.receive_shadow = true;
        scene.add(o)
    };
    let g = solid(
        &mut scene,
        PlaneGeometry::new(80.0, 80.0),
        0x4a5040,
        Vector3::ZERO,
    );
    if let Some(o) = scene.get_mut(g) {
        o.rotate_x(-std::f32::consts::FRAC_PI_2);
    }
    solid(
        &mut scene,
        BoxGeometry::new(1.4, 0.9, 1.0),
        0x7a4a32,
        Vector3::new(-2.4, 0.45, -1.2),
    );
    solid(
        &mut scene,
        BoxGeometry::new(0.9, 0.6, 0.9),
        0x5a6b78,
        Vector3::new(-1.2, 0.3, -2.4),
    );
    let mut anim = ParticleAnimator::new();
    let cover = anim.add_snow(
        &mut scene,
        SnowCover::new(Heightfield::new([0.0, 0.0], [20.0, 20.0], 300, 0.0)),
        Some(20.0),
    );
    let mut w = Weather::snow(60_000);
    w.half_width = 14.0;
    w.half_depth = 14.0;
    w.height = 8.0;
    let weather = anim.add_weather(&mut scene, w);
    let mut snow = presets::snowfall(
        Vector3::ZERO,
        [22.0, 22.0],
        6.0,
        2000.0,
        0.0015,
        Vector3::new(1.0, 0.0, 0.3),
    );
    snow.max_particles = 40_000;
    let flakes = anim.add(&mut scene, snow);
    anim.settle(flakes, cover);
    let at = Vector3::new(0.0, 0.05, 1.0);
    let mut fire = vec![
        anim.add(&mut scene, presets::fire(at, 0.28)),
        anim.add(&mut scene, presets::embers(at, 0.22)),
        anim.add(
            &mut scene,
            presets::smoke(Vector3::new(0.0, 0.9, 1.0), 0.22),
        ),
        anim.add(&mut scene, presets::sparks(Vector3::new(0.0, 0.25, 1.0))),
    ];
    fire.push(anim.add(
        &mut scene,
        presets::heat_haze(Vector3::new(0.0, 0.5, 1.0), 0.2),
    ));
    let eye = Vector3::new(4.5, 3.2, 7.5);
    for _ in 0..(8.0 * 60.0) as usize {
        anim.update(&mut scene, 1.0 / 60.0, Some(eye));
    }
    let mut camera = PerspectiveCamera::new(45.0, 16.0 / 9.0, 0.1, 200.0);
    camera.position = eye;
    camera.look_at(Vector3::new(-0.5, 0.6, -0.5));
    let _ = camera.projection_matrix();
    let mut hr = HeadlessRenderer::builder()
        .size(1280, 720)
        .build()
        .expect("a GPU adapter");
    hr.set_msaa(4);
    hr.renderer().set_tone_mapping(ToneMapping::AcesFilmic, 1.0);

    // The smoke again, with pieces of its shading off: the same particles
    // (the same seed), drawn one variant at a time.
    let smoke = fire[2];
    let variant =
        |anim: &mut ParticleAnimator,
         scene: &mut Scene,
         f: &dyn Fn(&mut threers::particles::ParticleSystem, &mut Object3D)| {
            let mut s = presets::smoke(Vector3::new(0.0, 0.9, 1.0), 0.22);
            let mut o = threers::particles::particles_object(&s);
            f(&mut s, &mut o);
            o = {
                let mut p = threers::particles::particles_object(&s);
                p.receive_shadow = o.receive_shadow;
                p
            };
            let id = scene.add(o);
            anim.track(id, s);
            id
        };
    let unshadowed = variant(&mut anim, &mut scene, &|_, o| o.receive_shadow = false);
    let unlit = variant(&mut anim, &mut scene, &|s, _| s.look.lit = false);
    let plain = variant(&mut anim, &mut scene, &|s, _| s.look.detail = 0.0);
    let variants = [smoke, unshadowed, unlit, plain];
    for _ in 0..(8.0 * 60.0) as usize {
        anim.update(&mut scene, 1.0 / 60.0, Some(eye));
    }
    let only = |keep: ObjectId| {
        variants
            .iter()
            .copied()
            .filter(|v| *v != keep)
            .collect::<Vec<_>>()
    };
    let configs: Vec<(String, Vec<ObjectId>, bool)> = [
        ("smoke as is".to_string(), only(smoke), true),
        ("smoke unshadowed".into(), only(unshadowed), true),
        ("smoke unlit".into(), only(unlit), true),
        ("smoke without billows".into(), only(plain), true),
        ("no smoke".into(), variants.to_vec(), true),
    ]
    .into_iter()
    .chain(if std::env::var("PB").is_ok_and(|v| v.contains("smoke")) {
        vec![]
    } else {
        vec![
            ("everything".to_string(), only(smoke), true),
            ("no sun shadow".into(), only(smoke), false),
            (
                "no weather".into(),
                [only(smoke), vec![weather]].concat(),
                true,
            ),
            (
                "no settling flakes".into(),
                [only(smoke), vec![flakes]].concat(),
                true,
            ),
            (
                "no snow cover".into(),
                [only(smoke), vec![cover]].concat(),
                true,
            ),
        ]
    })
    .chain(
        ["fire", "embers", "smoke", "sparks", "heat haze"]
            .iter()
            .zip(&fire)
            .map(|(n, id)| (format!("no {n}"), [only(smoke), vec![*id]].concat(), true))
            .filter(|_| std::env::var("PB").map_or(true, |v| !v.contains("smoke"))),
    )
    .collect();
    let set = |scene: &mut Scene, hidden: &[ObjectId], shadow: bool| {
        for id in [weather, flakes, cover]
            .iter()
            .chain(&fire)
            .chain(&variants)
        {
            if let Some(o) = scene.get_mut(*id) {
                o.visible = !hidden.contains(id);
            }
        }
        if let Some(o) = scene.get_mut(sun) {
            if let threers::ObjectKind::Light(Light::Directional(d)) = &mut o.kind {
                d.cast_shadow = shadow;
            }
        }
    };
    let mut times = vec![Vec::new(); configs.len()];
    for round in 0..150 {
        for (k, (_, hidden, shadow)) in configs.iter().enumerate() {
            set(&mut scene, hidden, *shadow);
            if std::env::var("PB").is_ok_and(|v| v.contains("nocast")) {
                if let Some(o) = scene.get_mut(cover) {
                    o.cast_shadow = false;
                }
            }
            anim.update(&mut scene, 1.0 / 60.0, Some(eye));
            let t = Instant::now();
            hr.render(&mut scene, &camera);
            let _ = hr
                .device()
                .poll(threers::wgpu::PollType::wait_indefinitely());
            if round >= 5 {
                times[k].push(t.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }
    for ((name, _, _), t) in configs.iter().zip(times) {
        // The fastest: on a shared GPU, the one frame nothing else slowed.
        println!(
            "{name}: {:.2} ms",
            t.iter().copied().fold(f64::INFINITY, f64::min)
        );
    }
}

fn main() {
    if std::env::args().any(|a| a == "draw") {
        draw();
        return;
    }
    // The winter camp: snowfall settling and drifting, and a campfire.
    let mut scene = Scene::new();
    ground(&mut scene);
    let mut crate_ = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.4, 0.9, 1.0),
        Material::Basic(BasicMaterial::new(Color::WHITE)),
    ));
    crate_.position = Vector3::new(-2.4, 0.45, -1.2);
    scene.add(crate_);
    let mut anim = ParticleAnimator::new();
    // `PB=…` leaves pieces out, to see what each costs.
    let skip = std::env::var("PB").unwrap_or_default();
    let cover = anim.add_snow(
        &mut scene,
        SnowCover::new(Heightfield::new([0.0, 0.0], [20.0, 20.0], 300, 0.0)),
        Some(20.0),
    );
    if !skip.contains("drift") {
        anim.set_layer_wind(cover, Some((Downwind::PlusX, 30_000.0)));
    }
    let mut snow = presets::snowfall(
        Vector3::ZERO,
        [22.0, 22.0],
        6.0,
        2000.0,
        0.0015,
        Vector3::new(1.0, 0.0, 0.3),
    );
    snow.max_particles = 40_000;
    let flakes = anim.add(&mut scene, snow);
    // What is seen falling: on the GPU, so free here.
    anim.add_weather(&mut scene, threers::particles::Weather::snow(60_000));
    if !skip.contains("settle") {
        anim.settle(flakes, cover);
    }
    if skip.contains("fire") {
        bench("winter camp", &mut scene, &mut anim, 8.0);
        return;
    }
    anim.add(
        &mut scene,
        presets::fire(Vector3::new(0.0, 0.05, 1.0), 0.28),
    );
    anim.add(
        &mut scene,
        presets::embers(Vector3::new(0.0, 0.05, 1.0), 0.22),
    );
    anim.add(
        &mut scene,
        presets::smoke(Vector3::new(0.0, 0.9, 1.0), 0.22),
    );
    anim.add(&mut scene, presets::sparks(Vector3::new(0.0, 0.25, 1.0)));
    bench("winter camp", &mut scene, &mut anim, 8.0);
    println!("  {} flakes", anim.system(flakes).map_or(0, |s| s.len()));

    // The desert: a poured pile, and dunes under a steady wind.
    let mut scene = Scene::new();
    ground(&mut scene);
    let mut anim = ParticleAnimator::new();
    let pile = anim.add_sand(
        &mut scene,
        SandBed::new(Heightfield::new([0.0, 2.5], [4.0, 4.0], 220, 0.0)),
        Some(10.0),
    );
    let grains = anim.add(
        &mut scene,
        presets::sand_stream(Vector3::new(0.0, 1.8, 2.5), 5000.0, 0.002),
    );
    anim.settle(grains, pile);
    let mut bed = SandBed::new(Heightfield::new([0.0, -14.0], [30.0, 30.0], 200, 0.0));
    bed.spread(0.12, 0.12);
    let dunes = anim.add_sand(&mut scene, bed, Some(10.0));
    anim.set_layer_wind(dunes, Some((Downwind::PlusX, 200_000.0)));
    bench("desert", &mut scene, &mut anim, 3.0);
}
