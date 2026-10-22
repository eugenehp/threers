//! Snow falling on two sheds and a post, settling where it lands — on the
//! roofs, the ground and the post's top — found by casting rays down into
//! the scene. Three frames of a time-lapse side by side in `out/snow.png`:
//! the first flakes, a covering, and a deep fall slumping off the roofs'
//! edges at snow's angle of repose, and drifting in the lee of the sheds.
//!
//! ```bash
//! cargo run --release --example snow
//! ```

use threers::lights::Light;
use threers::particles::sand::Downwind;
use threers::particles::snow::SnowCover;
use threers::particles::{presets, Heightfield, ParticleEffect};
use threers::prelude::*;

const W: u32 = 480;
const H: u32 = 360;

fn standard(hex: u32, roughness: f32) -> Material {
    let mut m = StandardMaterial::new(Color::from_hex(hex));
    m.roughness = roughness;
    Material::Standard(m)
}

fn main() {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x9aa7b8);
    // Falling snow is what you see through, and not far.
    scene.fog = FogParams {
        color: Color::from_hex(0x9aa7b8),
        density: 0.06,
        mode: 2,
        ..FogParams::default()
    };
    scene.add_light(HemisphereLight::new(
        Color::from_hex(0xdfe8f5),
        Color::from_hex(0x6a6f78),
        1.1,
    ));
    let mut sun = DirectionalLight::new(Color::from_hex(0xf3f1ec), 1.4);
    sun.cast_shadow = true;
    sun.shadow.camera_size = 7.0;
    sun.shadow.normal_bias = 0.02;
    sun.shadow.map_size = 2048;
    let mut l = Object3D::light(Light::Directional(sun));
    l.position = Vector3::new(-4.0, 8.0, 3.0);
    scene.add(l);

    // The things snow lands on.
    let mut solids = Vec::new();
    let mut ground = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(40.0, 40.0),
        standard(0x4d5243, 0.95),
    ));
    ground.rotate_x(-std::f32::consts::FRAC_PI_2);
    ground.receive_shadow = true;
    solids.push(scene.add(ground));
    let mut add = |scene: &mut Scene, g: BufferGeometry, m: Material, at: Vector3| {
        let mut o = Object3D::mesh(Mesh::new(g, m));
        o.position = at;
        o.cast_shadow = true;
        o.receive_shadow = true;
        let id = scene.add(o);
        solids.push(id);
    };
    add(
        &mut scene,
        BoxGeometry::new(1.6, 1.0, 1.2),
        standard(0x7a4a32, 0.8),
        Vector3::new(-1.4, 0.5, -0.6),
    );
    add(
        &mut scene,
        BoxGeometry::new(1.0, 0.7, 1.0),
        standard(0x5a6b78, 0.7),
        Vector3::new(1.2, 0.35, 0.3),
    );
    add(
        &mut scene,
        CylinderGeometry::new(0.12, 0.12, 1.3, 24, 1, false, 0.0, std::f32::consts::TAU),
        standard(0x3b3026, 0.9),
        Vector3::new(0.1, 0.65, 1.4),
    );
    scene.update_world();

    // Snow over the yard, resting on whatever is highest under each cell.
    let mut field = Heightfield::new([0.0, 0.0], [16.0, 16.0], 320, 0.0);
    field.base_from_scene(&scene.arena, &solids, 10.0);
    let mut cover = SnowCover::new(field);
    cover.add_to(&mut scene);

    // 5 mm a second over the yard — a time-lapse — in light wind.
    // Falling 5 m at about 1 m/s, a flake drifts 5 s downwind: the cloud is
    // that far upwind of the yard, and wider than it, so the yard is
    // covered evenly to its edges.
    let wind = Vector3::new(0.6, 0.0, 0.2);
    let flakes = presets::snowfall(wind * -5.0, [18.0, 18.0], 5.0, 20000.0, 0.005, wind);
    let mut flakes = flakes;
    // 10 000 a second, each 4 to 7 s in the air.
    flakes.max_particles = 120_000;
    let mut fx = ParticleEffect::add_to(&mut scene, flakes);

    let mut hr = HeadlessRenderer::builder()
        .size(W, H)
        .build()
        .expect("a GPU adapter");
    hr.set_msaa(4);
    hr.renderer().set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
    let mut camera = PerspectiveCamera::new(42.0, W as f32 / H as f32, 0.05, 100.0);
    camera.position = Vector3::new(2.2, 2.6, 5.4);
    camera.look_at(Vector3::new(-0.2, 0.4, 0.0));

    let shots = [7.0f32, 16.0, 50.0];
    let dt = 1.0 / 30.0;
    let mut t = 0.0;
    let mut sheet = vec![0u8; (W * shots.len() as u32 * H * 4) as usize];
    for (slot, &shot) in shots.iter().enumerate() {
        while t < shot {
            fx.system.step(dt, Some(&cover.field));
            cover.absorb(&mut fx.system);
            // The wind drifting the lying snow: scoured from the open,
            // piled in the lee of the sheds.
            cover.drift(Downwind::PlusX, 1500);
            t += dt;
        }
        cover.sync(&mut scene);
        fx.sync(&mut scene, Some(camera.position));
        println!(
            "  {t:.0} s: {} flakes in the air, {:.3} m³ of snow, deepest {:.2} m",
            fx.system.len(),
            cover.field.volume(),
            cover.field.depth.iter().cloned().fold(0.0, f32::max)
        );
        let px = hr.render_to_rgba(&mut scene, &camera);
        let stride = W * shots.len() as u32;
        for y in 0..H {
            let row = &px[(y * W * 4) as usize..((y + 1) * W * 4) as usize];
            let at = ((y * stride + slot as u32 * W) * 4) as usize;
            sheet[at..at + row.len()].copy_from_slice(row);
        }
    }
    std::fs::create_dir_all("out").unwrap();
    std::fs::write(
        "out/snow.png",
        encode_png(W * shots.len() as u32, H, &sheet),
    )
    .unwrap();
    println!("wrote out/snow.png");
}
