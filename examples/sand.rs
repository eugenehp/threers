//! Sand two ways, in `out/sand.png`:
//!
//! - poured from a nozzle, grains falling and piling into a cone that stands
//!   at sand's angle of repose (about 34°) whatever its size — after 5 s and
//!   after 30 s;
//! - a flat, patchy sheet blown by a steady wind into dunes by Werner's slab
//!   model: crescent barchans with their horns downwind, gentle windward
//!   slopes and steep slip faces.
//!
//! ```bash
//! cargo run --release --example sand
//! ```

use threers::lights::Light;
use threers::particles::sand::{Downwind, SandBed};
use threers::particles::{presets, Heightfield, ParticleEffect};
use threers::prelude::*;

const W: u32 = 480;
const H: u32 = 360;

fn lights(scene: &mut Scene, size: f32) {
    scene.add_light(HemisphereLight::new(
        Color::from_hex(0xbcd2ff),
        Color::from_hex(0x6b5a44),
        0.6,
    ));
    let mut sun = DirectionalLight::new(Color::from_hex(0xfff0d8), 2.6);
    sun.cast_shadow = true;
    sun.shadow.camera_size = size;
    sun.shadow.normal_bias = 0.01;
    sun.shadow.map_size = 2048;
    let mut l = Object3D::light(Light::Directional(sun));
    l.position = Vector3::new(-5.0, 4.0, 3.0);
    scene.add(l);
}

fn render(hr: &mut HeadlessRenderer, scene: &mut Scene, camera: &PerspectiveCamera) -> Vec<u8> {
    hr.render_to_rgba(scene, camera)
}

fn main() {
    let mut hr = HeadlessRenderer::builder()
        .size(W, H)
        .build()
        .expect("a GPU adapter");
    hr.set_msaa(4);
    hr.renderer().set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
    let mut panels = Vec::new();

    // ---- a poured pile -------------------------------------------------
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x2a3140);
    lights(&mut scene, 2.0);
    let mut floor_m = StandardMaterial::new(Color::from_hex(0x4a4640));
    floor_m.roughness = 0.9;
    let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(20.0, 20.0), floor_m.into()));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    floor.receive_shadow = true;
    scene.add(floor);
    let mut bed = SandBed::new(Heightfield::new([0.0, 0.0], [2.4, 2.4], 192, 0.0));
    // Just poured: no wind has rippled it.
    bed.ripples = 0.0;
    bed.add_to(&mut scene);
    // 2 litres a second, from 1.5 m up.
    let mut grains = presets::sand_stream(Vector3::new(0.0, 1.5, 0.0), 6000.0, 0.002);
    grains.look.casts_shadow = true;
    let mut fx = ParticleEffect::add_to(&mut scene, grains);
    let mut nozzle = Object3D::mesh(Mesh::new(
        CylinderGeometry::new(0.05, 0.03, 0.12, 24, 1, true, 0.0, std::f32::consts::TAU),
        {
            let mut m = StandardMaterial::new(Color::from_hex(0x8a8f99));
            m.metalness = 0.8;
            m.roughness = 0.35;
            m.side = 2;
            m.into()
        },
    ));
    nozzle.position = Vector3::new(0.0, 1.56, 0.0);
    scene.add(nozzle);
    let mut camera = PerspectiveCamera::new(38.0, W as f32 / H as f32, 0.05, 50.0);
    camera.position = Vector3::new(0.9, 0.55, 1.5);
    camera.look_at(Vector3::new(0.0, 0.35, 0.0));
    let dt = 1.0 / 60.0;
    let mut t = 0.0f32;
    for shot in [5.0f32, 30.0] {
        while t < shot {
            fx.system.step(dt, Some(&bed.field));
            bed.absorb(&mut fx.system);
            t += dt;
        }
        bed.sync(&mut scene);
        fx.sync(&mut scene, Some(camera.position));
        println!(
            "  {t:.0} s: {:.4} m³ piled, {} grains falling, flank {:.1}°",
            bed.field.volume(),
            fx.system.len(),
            bed.field.steepest(1e-3).to_degrees()
        );
        panels.push(render(&mut hr, &mut scene, &camera));
    }

    // ---- dunes ---------------------------------------------------------
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x9fb6d0);
    lights(&mut scene, 14.0);
    let mut bed = SandBed::new(Heightfield::new([0.0, 0.0], [24.0, 24.0], 160, 0.0));
    // Sand-starved: a patchy sheet averaging 12 cm, over a hard floor.
    bed.saltation.slab = 0.04;
    bed.spread(0.12, 0.12);
    let mut floor_m = StandardMaterial::new(Color::from_hex(0xa8865c));
    floor_m.roughness = 1.0;
    let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(80.0, 80.0), floor_m.into()));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    floor.position.y = -0.005;
    floor.receive_shadow = true;
    scene.add(floor);
    let start = std::time::Instant::now();
    for _ in 0..300 {
        bed.blow(Downwind::PlusX, 25_000);
    }
    println!(
        "  dunes: {:.1} s of blowing, tallest {:.2} m, slip faces {:.1}°",
        start.elapsed().as_secs_f32(),
        bed.field.depth.iter().cloned().fold(0.0, f32::max),
        bed.field.steepest(0.02).to_degrees()
    );
    bed.add_to(&mut scene);
    let mut camera = PerspectiveCamera::new(40.0, W as f32 / H as f32, 0.1, 200.0);
    camera.position = Vector3::new(-6.0, 9.0, 13.0);
    camera.look_at(Vector3::new(1.0, 0.0, 0.0));
    panels.push(render(&mut hr, &mut scene, &camera));

    let n = panels.len() as u32;
    let mut sheet = vec![0u8; (W * n * H * 4) as usize];
    for (slot, px) in panels.iter().enumerate() {
        for y in 0..H {
            let row = &px[(y * W * 4) as usize..((y + 1) * W * 4) as usize];
            let at = ((y * W * n + slot as u32 * W) * 4) as usize;
            sheet[at..at + row.len()].copy_from_slice(row);
        }
    }
    std::fs::create_dir_all("out").unwrap();
    std::fs::write("out/sand.png", encode_png(W * n, H, &sheet)).unwrap();
    println!("wrote out/sand.png (pile at 5 s, at 30 s, dunes)");
}
