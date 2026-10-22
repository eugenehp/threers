//! A campfire on a snowy night: everything `threers::particles` does in one
//! frame. Snow has been falling for a while — lying on the ground, the logs
//! and the stones, drifted by the wind and glittering where the firelight
//! catches it — and is still falling. The fire is flame (black-body
//! coloured, broken into tongues), embers, sparks, smoke that rises, cools,
//! shades itself and casts a shadow, and heat haze bending the view above
//! it; its glow drives a point light that lights the snow and casts the
//! logs' shadows, flickering as the flames do. Rendered through bloom and
//! ACES. Written to `out/campfire.png`.
//!
//! ```bash
//! cargo run --release --example campfire
//! ```

use threers::lights::Light;
use threers::particles::sand::Downwind;
use threers::particles::snow::SnowCover;
use threers::particles::{presets, Heightfield, ParticleAnimator};
use threers::postprocessing::{EffectComposer, OutputPass, RenderPass, UnrealBloomPass};
use threers::prelude::*;
use threers::renderer::{readback, RenderTarget};

const W: u32 = 1280;
const H: u32 = 720;
const FORMAT: threers::wgpu::TextureFormat = threers::wgpu::TextureFormat::Rgba8Unorm;

fn standard(hex: u32, roughness: f32) -> Material {
    let mut m = StandardMaterial::new(Color::from_hex(hex));
    m.roughness = roughness;
    Material::Standard(m)
}

fn main() {
    let mut scene = Scene::new();
    let night = Color::from_hex(0x0a0f1c);
    scene.background = night;
    scene.fog = FogParams {
        color: Color::from_hex(0x0c1220),
        density: 0.07,
        mode: 2,
        ..FogParams::default()
    };
    scene.add_light(HemisphereLight::new(
        Color::from_hex(0x2a3a60),
        Color::from_hex(0x0a0806),
        0.35,
    ));
    // A cold moon, low and casting.
    let mut moon = DirectionalLight::new(Color::from_hex(0x9fb4ff), 0.35);
    moon.cast_shadow = true;
    moon.shadow.camera_size = 9.0;
    moon.shadow.map_size = 2048;
    moon.shadow.normal_bias = 0.02;
    moon.shadow.radius = 3.0;
    let mut l = Object3D::light(Light::Directional(moon));
    l.position = Vector3::new(-6.0, 7.0, -4.0);
    scene.add(l);

    // The camp: ground, logs to sit on, a ring of stones, firewood.
    let mut solids = Vec::new();
    let mut ground = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(60.0, 60.0),
        standard(0x2b2a26, 1.0),
    ));
    ground.rotate_x(-std::f32::consts::FRAC_PI_2);
    ground.receive_shadow = true;
    solids.push(scene.add(ground));
    let mut put =
        |scene: &mut Scene, g: BufferGeometry, m: Material, at: Vector3, turn: Vector3| {
            let mut o = Object3D::mesh(Mesh::new(g, m));
            o.position = at;
            o.set_rotation_from_axis_angle(Vector3::new(0.0, 1.0, 0.0), turn.y);
            if turn.z != 0.0 {
                o.rotate_z(turn.z);
            }
            o.cast_shadow = true;
            o.receive_shadow = true;
            solids.push(scene.add(o));
        };
    let log = || CylinderGeometry::new(0.2, 0.22, 2.0, 24, 1, false, 0.0, std::f32::consts::TAU);
    put(
        &mut scene,
        log(),
        standard(0x4a3424, 0.9),
        Vector3::new(-1.7, 0.21, 0.6),
        Vector3::new(0.0, 0.0, std::f32::consts::FRAC_PI_2),
    );
    put(
        &mut scene,
        log(),
        standard(0x3f2d20, 0.9),
        Vector3::new(1.5, 0.21, -1.2),
        Vector3::new(
            0.0,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::FRAC_PI_2,
        ),
    );
    for k in 0..11 {
        let a = k as f32 / 11.0 * std::f32::consts::TAU;
        put(
            &mut scene,
            SphereGeometry::new(0.13 + 0.03 * ((k * 7) % 3) as f32, 16, 10),
            standard(0x55524c, 0.8),
            Vector3::new(a.cos() * 0.55, 0.07, a.sin() * 0.55),
            Vector3::ZERO,
        );
    }
    let stick = || CylinderGeometry::new(0.04, 0.05, 0.8, 10, 1, false, 0.0, std::f32::consts::TAU);
    for k in 0..5 {
        let a = k as f32 / 5.0 * std::f32::consts::TAU;
        let mut o = Object3D::mesh(Mesh::new(stick(), standard(0x2a1b10, 0.9)));
        o.position = Vector3::new(a.cos() * 0.12, 0.25, a.sin() * 0.12);
        o.set_rotation_from_axis_angle(Vector3::new(-a.sin(), 0.0, a.cos()), 0.5);
        o.cast_shadow = true;
        scene.add(o);
    }
    scene.update_world();

    let mut particles = ParticleAnimator::new();
    // Snow that has been falling a while, drifted by the wind.
    let mut cover = SnowCover::new(Heightfield::new([0.0, 0.0], [18.0, 18.0], 300, 0.0));
    cover.field.base_from_scene(&scene.arena, &solids, 10.0);
    // None in the fire ring, which it melts.
    let mut flakes = presets::snowfall(
        Vector3::new(-2.0, 0.0, -0.6),
        [20.0, 20.0],
        6.0,
        6000.0,
        0.004,
        Vector3::new(0.4, 0.0, 0.12),
    );
    flakes.max_particles = 80_000;
    let snow = particles.add_snow(&mut scene, cover, None);
    let flakes = particles.add(&mut scene, flakes);
    particles.settle(flakes, snow);
    particles.set_layer_wind(snow, Some((Downwind::PlusX, 6_000.0)));
    let dt = 1.0 / 30.0;
    for _ in 0..(25.0 / dt) as usize {
        particles.update(&mut scene, dt, None);
    }
    // Melted round the fire.
    if let Some(threers::particles::Layer::Snow(c)) = particles.layer_mut(snow) {
        let f = &mut c.field;
        for j in 0..f.nz {
            for i in 0..f.nx {
                let [x, z] = f.cell_center(i, j);
                let r = (x * x + z * z).sqrt();
                let k = i + f.nx * j;
                f.depth[k] *= ((r - 0.75) / 0.6).clamp(0.0, 1.0);
            }
        }
        f.touch_all();
        f.version += 1;
    }
    // Lighter now.
    if let Some(s) = particles.system_mut(flakes) {
        s.emitters[0].rate = 2500.0;
    }

    // The fire, and the light it gives.
    let at = Vector3::new(0.0, 0.05, 0.0);
    let fire = particles.add(&mut scene, presets::fire(at, 0.26));
    // Embers burn by their own heat balance, and those that come down on
    // the snow are quenched there.
    let mut embers = presets::embers(at, 0.22);
    embers
        .colliders
        .push(threers::particles::Collider::Ground { height: 0.0 });
    embers.response = threers::particles::Response::Stick;
    if let Some(t) = &mut embers.thermal {
        t.ambient = 263.0;
        t.contact = 3000.0;
        t.surface_temperature = 268.0;
    }
    particles.add(&mut scene, embers);
    particles.add(&mut scene, presets::smoke(Vector3::new(0.0, 1.0, 0.0), 0.2));
    particles.add(
        &mut scene,
        presets::heat_haze(Vector3::new(0.0, 0.5, 0.0), 0.2),
    );
    let mut sparks = presets::sparks(Vector3::new(0.0, 0.3, 0.0));
    sparks.emitters[0].rate = 40.0;
    particles.add(&mut scene, sparks);
    let mut glow = PointLight::new(Color::from_hex(0xff9a4a), 9.0);
    glow.distance = 14.0;
    glow.decay = 2.0;
    glow.cast_shadow = true;
    glow.shadow.normal_bias = 0.02;
    let light = scene.add(Object3D::light(Light::Point(glow)));
    particles.light_with(fire, light, 9.0);

    let mut camera = PerspectiveCamera::new(38.0, W as f32 / H as f32, 0.05, 100.0);
    camera.position = Vector3::new(2.4, 1.25, 3.6);
    camera.look_at(Vector3::new(-0.2, 0.55, 0.0));
    for _ in 0..(4.0 / dt) as usize {
        particles.update(&mut scene, dt, Some(camera.position));
    }

    let mut hr = HeadlessRenderer::builder()
        .size(W, H)
        .color_format(FORMAT)
        .build()
        .expect("a GPU adapter");
    hr.set_msaa(4);
    hr.renderer().set_tone_mapping(ToneMapping::AcesFilmic, 1.1);
    let mut composer = EffectComposer::new(hr.device().clone(), hr.queue().clone(), W, H, FORMAT);
    composer.add_pass(RenderPass::new());
    composer.add_pass(UnrealBloomPass::new(W, H, 0.6, 0.45, 0.85));
    composer.add_pass(OutputPass::new());
    let out = RenderTarget::new(hr.device(), W, H, FORMAT);
    composer.render(hr.renderer(), &mut scene, &camera, &out.color_view);
    let px = readback::texture_bytes(hr.device(), hr.queue(), &out.color_texture);
    std::fs::create_dir_all("out").unwrap();
    std::fs::write("out/campfire.png", encode_png(W, H, &px)).unwrap();
    println!("wrote out/campfire.png");
}
