//! Frame-time benchmarks, for profiling rather than for CI: `cargo test
//! --release --test frame_bench -- --ignored --nocapture` (with `--features
//! metal` to add the Metal renderer). `FRAME_BENCH_SECS` runs each scene for
//! that long instead, for a sampling profiler to attach to.
//!
//! Each scene is drawn under a moving camera, so per-frame work that a still
//! camera would let the renderer skip (shadow maps) is skipped as it would
//! be in an app: the cost reported is what a frame of that scene costs.

use threers::lights::{Light, ShadowType};
use threers::materials::TransparencyMode;
use threers::prelude::*;

const W: u32 = 960;
const H: u32 = 540;

fn floor(scene: &mut Scene) {
    let mut m = StandardMaterial::new(Color::from_hex(0xbbbbbb));
    m.roughness = 0.9;
    let mut f = Object3D::mesh(Mesh::new(PlaneGeometry::new(60.0, 60.0), m.into()));
    f.rotate_x(-std::f32::consts::FRAC_PI_2);
    f.receive_shadow = true;
    scene.add(f);
}

fn sun(scene: &mut Scene, kind: ShadowType) {
    let mut sun = DirectionalLight::new(Color::WHITE, 2.0);
    sun.direction = Vector3::new(-0.4, -1.0, -0.3).normalize();
    sun.cast_shadow = true;
    sun.shadow.camera_size = 20.0;
    sun.shadow.map_size = 2048;
    sun.shadow.filter = Some(kind);
    scene.add(Object3D::light(Light::Directional(sun)));
}

/// 2,000 lit, shadow-casting boxes in a grid.
fn many_meshes() -> Scene {
    let mut s = Scene::new();
    s.add_light(AmbientLight::new(Color::WHITE, 0.3));
    sun(&mut s, ShadowType::Pcf);
    floor(&mut s);
    let geo = std::sync::Arc::new(BoxGeometry::new(0.2, 0.2, 0.2));
    let mat = std::sync::Arc::new(Material::Standard(StandardMaterial::new(Color::from_hex(
        0x8899aa,
    ))));
    for i in 0..2000 {
        let mut o = Object3D::mesh(Mesh::from_arc(geo.clone(), mat.clone()));
        o.position = Vector3::new(
            (i % 50) as f32 * 0.3 - 7.5,
            0.1,
            (i / 50) as f32 * 0.3 - 6.0,
        );
        o.cast_shadow = true;
        o.receive_shadow = true;
        s.add(o);
    }
    s
}

/// Every kind of light casting, PCSS, a cascaded sun.
fn many_shadows() -> Scene {
    let mut s = Scene::new();
    s.add_light(AmbientLight::new(Color::WHITE, 0.2));
    let mut sun = DirectionalLight::new(Color::WHITE, 1.5);
    sun.direction = Vector3::new(-0.4, -1.0, -0.3).normalize();
    sun.cast_shadow = true;
    sun.shadow.cascades = 4;
    sun.shadow.cascade_distance = 40.0;
    sun.shadow.filter = Some(ShadowType::Pcss);
    s.add(Object3D::light(Light::Directional(sun)));
    for (i, x) in [-6.0f32, 0.0, 6.0].into_iter().enumerate() {
        let mut p = PointLight::new(Color::WHITE, 30.0);
        p.cast_shadow = true;
        p.distance = 20.0;
        let mut o = Object3D::light(Light::Point(p));
        o.position = Vector3::new(x, 3.0, 2.0);
        s.add(o);
        let mut sp = SpotLight::new(Color::WHITE, 40.0);
        sp.cast_shadow = true;
        sp.angle = 0.6;
        sp.direction = Vector3::new(0.0, -1.0, 0.0);
        let mut o = Object3D::light(Light::Spot(sp));
        o.position = Vector3::new(x, 6.0, -3.0 - i as f32);
        s.add(o);
    }
    floor(&mut s);
    for i in 0..200 {
        let mut o = Object3D::mesh(Mesh::new(
            SphereGeometry::new(0.3, 24, 12),
            Material::Standard(StandardMaterial::new(Color::from_hex(0xaa8866))),
        ));
        o.position = Vector3::new(
            (i % 20) as f32 * 0.8 - 8.0,
            0.3,
            (i / 20) as f32 * 0.8 - 4.0,
        );
        o.cast_shadow = true;
        o.receive_shadow = true;
        s.add(o);
    }
    s
}

/// Glass in every mode over a lit, shadowed floor.
fn glass(mode: TransparencyMode) -> Scene {
    let mut s = Scene::new();
    s.add_light(AmbientLight::new(Color::WHITE, 0.4));
    sun(&mut s, ShadowType::Pcf);
    floor(&mut s);
    for i in 0..60 {
        let glass = PhysicalMaterial::new(Color::from_hex(0xccffdd))
            .with_roughness(0.05)
            .with_transmission(1.0, 1.5, 0.5)
            .with_transparency(mode);
        let mut o = Object3D::mesh(Mesh::new(
            SphereGeometry::new(0.4, 32, 16),
            Material::Physical(glass),
        ));
        o.position = Vector3::new(
            (i % 10) as f32 * 0.7 - 3.5,
            0.5 + (i / 10) as f32 * 0.25,
            (i / 10) as f32 * 0.5 - 1.5,
        );
        o.cast_shadow = true;
        s.add(o);
    }
    s
}

fn camera(t: f32) -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(50.0, W as f32 / H as f32, 0.1, 200.0);
    c.position = Vector3::new(8.0 * t.cos(), 6.0, 8.0 * t.sin() + 4.0);
    c.look_at(Vector3::ZERO);
    c
}

fn scenes() -> Vec<(&'static str, Scene)> {
    vec![
        ("2000 meshes", many_meshes()),
        ("9 shadow lights, pcss, cascades", many_shadows()),
        ("glass refract", glass(TransparencyMode::Refract)),
        ("glass blended", glass(TransparencyMode::Glass)),
        ("glass oit", glass(TransparencyMode::Oit)),
    ]
}

/// Frames for `secs` seconds (or 60 frames), each to the screen-like path
/// and waited on: CPU encode plus GPU.
fn run(name: &str, backend: &str, mut frame: impl FnMut(f32)) {
    let secs: f32 = std::env::var("FRAME_BENCH_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    for i in 0..5 {
        frame(i as f32 * 0.01);
    }
    let start = std::time::Instant::now();
    let mut n = 0u32;
    while (secs > 0.0 && start.elapsed().as_secs_f32() < secs) || (secs <= 0.0 && n < 60) {
        frame(0.05 + n as f32 * 0.01);
        n += 1;
    }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
    eprintln!("{backend:6} {name:32} {ms:7.2} ms/frame over {n} frames");
}

#[test]
#[ignore]
fn frame_times() {
    if let Ok(mut hr) = HeadlessRenderer::builder().size(W, H).build() {
        hr.set_msaa(4);
        for (name, mut scene) in scenes() {
            let before = hr.renderer().shadow_stats();
            let mut frames = 0u64;
            run(name, "wgpu", |t| {
                let _ = hr.render_to_rgba(&mut scene, &camera(t));
                frames += 1;
            });
            let after = hr.renderer().shadow_stats();
            eprintln!(
                "       {:32} {:.1} shadow passes, {:.0} caster draws a frame",
                "",
                (after.0 - before.0) as f64 / frames as f64,
                (after.1 - before.1) as f64 / frames as f64
            );
        }
    }
    #[cfg(feature = "metal")]
    if let Ok(mut r) = threers::metal::MetalHeadlessRenderer::builder()
        .size(W, H)
        .msaa(4)
        .build()
    {
        for (name, mut scene) in scenes() {
            run(name, "metal", |t| {
                let _ = r.render_to_rgba(&mut scene, &camera(t));
            });
        }
    }
}
