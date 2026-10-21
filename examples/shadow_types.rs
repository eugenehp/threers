//! The four shadow filters on one scene: `Basic`, `Pcf` (the default),
//! `PcfSoft` and `Pcss`, left to right and top to bottom. A pole, a ball and
//! a knot stand on a floor under a low sun; PCSS is sharp where each meets
//! the ground and softens with the distance to the caster. Written to
//! `out/shadow_types.png`.
//!
//! ```bash
//! cargo run --release --example shadow_types
//! ```

use threers::lights::{Light, ShadowType};
use threers::prelude::*;

const W: u32 = 480;
const H: u32 = 320;

fn standard(hex: u32, roughness: f32) -> Material {
    let mut m = StandardMaterial::new(Color::from_hex(hex));
    m.roughness = roughness;
    m.metalness = 0.0;
    Material::Standard(m)
}

fn scene(kind: ShadowType) -> Scene {
    let mut s = Scene::new();
    s.background = Color::from_hex(0x20242c);
    s.add_light(HemisphereLight::new(
        Color::from_hex(0xbcd4ff),
        Color::from_hex(0x3a3028),
        0.5,
    ));
    let mut sun = DirectionalLight::new(Color::from_hex(0xfff1dc), 3.0);
    sun.cast_shadow = true;
    sun.shadow.map_size = 1024;
    sun.shadow.camera_size = 12.0;
    sun.shadow.camera_near = 0.5;
    sun.shadow.camera_far = 40.0;
    sun.shadow.normal_bias = 0.02;
    sun.shadow.radius = 2.0;
    sun.shadow.light_size = 0.06;
    sun.shadow.filter = Some(kind);
    let mut l = Object3D::light(Light::Directional(sun));
    l.position = Vector3::new(-7.0, 3.6, -2.5);
    s.add(l);

    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(40.0, 40.0),
        standard(0xd8d2c8, 0.9),
    ));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    floor.receive_shadow = true;
    s.add(floor);

    let mut add = |geometry: BufferGeometry, material: Material, at: Vector3| {
        let mut o = Object3D::mesh(Mesh::new(geometry, material));
        o.position = at;
        o.cast_shadow = true;
        o.receive_shadow = true;
        s.add(o);
    };
    add(
        CylinderGeometry::new(0.1, 0.1, 2.6, 24, 1, false, 0.0, std::f32::consts::TAU),
        standard(0x2f5d8a, 0.5),
        Vector3::new(-2.4, 1.3, -0.8),
    );
    add(
        SphereGeometry::new(0.8, 64, 32),
        standard(0xc8553d, 0.4),
        Vector3::new(-0.6, 0.8, 1.0),
    );
    add(
        TorusKnotGeometry::new(0.6, 0.18, 160, 24, 2, 3),
        standard(0xe0b040, 0.35),
        Vector3::new(0.9, 1.4, -0.6),
    );
    s
}

fn main() {
    let mut hr = HeadlessRenderer::builder()
        .size(W, H)
        .build()
        .expect("a GPU adapter");
    hr.set_msaa(4);
    let mut camera = PerspectiveCamera::new(40.0, W as f32 / H as f32, 0.1, 100.0);
    camera.position = Vector3::new(1.2, 4.2, 7.0);
    camera.look_at(Vector3::new(0.9, 0.3, 0.4));

    let kinds = [
        ShadowType::Basic,
        ShadowType::Pcf,
        ShadowType::PcfSoft,
        ShadowType::Pcss,
    ];
    let mut sheet = vec![0u8; (W * 2 * H * 2 * 4) as usize];
    for (i, kind) in kinds.into_iter().enumerate() {
        let px = hr.render_to_rgba(&mut scene(kind), &camera);
        let (cx, cy) = (i as u32 % 2, i as u32 / 2);
        for y in 0..H {
            let row = &px[(y * W * 4) as usize..((y + 1) * W * 4) as usize];
            let at = (((cy * H + y) * W * 2 + cx * W) * 4) as usize;
            sheet[at..at + row.len()].copy_from_slice(row);
        }
    }
    std::fs::create_dir_all("out").unwrap();
    std::fs::write("out/shadow_types.png", encode_png(W * 2, H * 2, &sheet)).unwrap();
    println!("wrote out/shadow_types.png (basic, pcf / pcf soft, pcss)");
}
