//! The particle presets, each in its own panel: a campfire (fire, embers
//! and smoke), smoke, sparks,
//! rain, a fountain and a puff of dust, simulated for a few seconds and
//! rendered headless. Written to `out/particles.png`.
//!
//! ```bash
//! cargo run --release --example particles
//! ```

use threers::lights::Light;
use threers::particles::{presets, ParticleEffect, ParticleSystem};
use threers::prelude::*;

const W: u32 = 400;
const H: u32 = 300;

fn stage(night: bool) -> Scene {
    let mut s = Scene::new();
    s.background = Color::from_hex(if night { 0x07080c } else { 0x1a1f28 });
    s.add_light(HemisphereLight::new(
        Color::from_hex(0xb8ccff),
        Color::from_hex(0x2c241c),
        if night { 0.15 } else { 0.7 },
    ));
    let mut sun = DirectionalLight::new(Color::from_hex(0xfff0dc), if night { 0.4 } else { 2.2 });
    sun.cast_shadow = true;
    sun.shadow.camera_size = 6.0;
    sun.shadow.normal_bias = 0.02;
    let mut l = Object3D::light(Light::Directional(sun));
    l.position = Vector3::new(-3.0, 6.0, 4.0);
    s.add(l);
    let mut m = StandardMaterial::new(Color::from_hex(0x3a3f4a));
    m.roughness = 0.9;
    let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(30.0, 30.0), m.into()));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    floor.receive_shadow = true;
    s.add(floor);
    s
}

fn panel(
    hr: &mut HeadlessRenderer,
    systems: Vec<ParticleSystem>,
    seconds: f32,
    night: bool,
    camera: &PerspectiveCamera,
) -> Vec<u8> {
    let mut scene = stage(night);
    let mut effects: Vec<ParticleEffect> = systems
        .into_iter()
        .map(|s| ParticleEffect::add_to(&mut scene, s))
        .collect();
    let dt = 1.0 / 60.0;
    for _ in 0..(seconds / dt) as usize {
        for fx in &mut effects {
            fx.system.step(dt, None);
        }
    }
    for fx in &effects {
        fx.sync(&mut scene, Some(camera.position));
    }
    hr.render_to_rgba(&mut scene, camera)
}

fn main() {
    let mut hr = HeadlessRenderer::builder()
        .size(W, H)
        .build()
        .expect("a GPU adapter");
    hr.set_msaa(4);
    hr.renderer().set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
    let camera = |eye: Vector3, at: Vector3| {
        let mut c = PerspectiveCamera::new(45.0, W as f32 / H as f32, 0.05, 100.0);
        c.position = eye;
        c.look_at(at);
        c
    };
    let near = camera(Vector3::new(0.0, 1.2, 3.2), Vector3::new(0.0, 0.8, 0.0));
    let panels = [
        panel(
            &mut hr,
            vec![
                presets::smoke(Vector3::new(0.0, 0.9, 0.0), 0.2),
                presets::fire(Vector3::ZERO, 0.3),
                presets::embers(Vector3::ZERO, 0.25),
            ],
            4.0,
            true,
            &camera(Vector3::new(0.0, 1.4, 3.6), Vector3::new(0.0, 1.0, 0.0)),
        ),
        panel(
            &mut hr,
            vec![presets::smoke(Vector3::ZERO, 0.25)],
            6.0,
            false,
            &camera(Vector3::new(0.0, 1.8, 5.0), Vector3::new(0.6, 1.6, 0.0)),
        ),
        panel(
            &mut hr,
            vec![presets::sparks(Vector3::new(0.0, 1.0, 0.0))],
            2.0,
            true,
            &near,
        ),
        panel(
            &mut hr,
            vec![presets::rain(
                Vector3::ZERO,
                [8.0, 8.0],
                6.0,
                6000.0,
                Vector3::new(1.5, 0.0, 0.0),
            )],
            2.0,
            false,
            &camera(Vector3::new(0.0, 1.5, 4.0), Vector3::new(0.0, 1.2, 0.0)),
        ),
        panel(
            &mut hr,
            vec![presets::fountain(Vector3::ZERO, 1.6)],
            2.5,
            false,
            &camera(Vector3::new(0.0, 1.2, 3.6), Vector3::new(0.0, 1.0, 0.0)),
        ),
        panel(
            &mut hr,
            vec![presets::dust(Vector3::ZERO, 120)],
            1.2,
            false,
            &camera(Vector3::new(0.0, 1.0, 3.6), Vector3::new(0.0, 0.5, 0.0)),
        ),
    ];
    let cols = 3;
    let mut sheet = vec![0u8; (W * cols * H * 2 * 4) as usize];
    for (n, px) in panels.iter().enumerate() {
        let (cx, cy) = (n as u32 % cols, n as u32 / cols);
        for y in 0..H {
            let row = &px[(y * W * 4) as usize..((y + 1) * W * 4) as usize];
            let at = (((cy * H + y) * W * cols + cx * W) * 4) as usize;
            sheet[at..at + row.len()].copy_from_slice(row);
        }
    }
    std::fs::create_dir_all("out").unwrap();
    std::fs::write("out/particles.png", encode_png(W * cols, H * 2, &sheet)).unwrap();
    println!("wrote out/particles.png (campfire, smoke, sparks / rain, fountain, dust)");
}
