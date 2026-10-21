//! One scene through every renderer threers has: wgpu natively (Metal,
//! Vulkan or DX12 underneath) and, with `--features metal` on Apple
//! platforms, threers' own Metal renderer. The same scene, number for
//! number, is `web/examples/parity.html`, which draws it in a browser on
//! WebGPU and (with `?gl=1`) on WebGL2.
//!
//! Writes `out/backend_parity.png`, the renderers side by side, and prints
//! how far apart each pair is.
//!
//! ```bash
//! cargo run --release --example backend_parity
//! cargo run --release --features metal --example backend_parity
//! ```

use threers::lights::Light;
use threers::prelude::*;
use threers::ToneMapping;

pub const W: u32 = 640;
pub const H: u32 = 400;

fn standard(hex: u32, roughness: f32, metalness: f32) -> Material {
    let mut m = StandardMaterial::new(Color::from_hex(hex));
    m.roughness = roughness;
    m.metalness = metalness;
    Material::Standard(m)
}

/// Keep in step with `web/examples/parity.html`.
fn scene() -> (Scene, PerspectiveCamera) {
    let mut s = Scene::new();
    s.background = Color::from_hex(0x1b1f27);
    s.add_light(HemisphereLight::new(
        Color::from_hex(0xbfd4ff),
        Color::from_hex(0x3a2e22),
        0.8,
    ));
    let mut sun = DirectionalLight::new(Color::from_hex(0xfff0dc), 2.4);
    sun.cast_shadow = true;
    sun.shadow.map_size = 2048;
    sun.shadow.camera_size = 5.0;
    sun.shadow.camera_near = 0.5;
    sun.shadow.camera_far = 30.0;
    sun.shadow.normal_bias = 0.02;
    let mut l = Object3D::light(Light::Directional(sun));
    l.position = Vector3::new(-3.0, 6.0, 4.0);
    s.add(l);
    let mut point = PointLight::new(Color::from_hex(0xff9a50), 6.0);
    point.distance = 8.0;
    point.decay = 2.0;
    let mut lamp = Object3D::light(Light::Point(point));
    lamp.position = Vector3::new(1.6, 1.2, 1.4);
    s.add(lamp);

    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(12.0, 12.0),
        standard(0x9c958a, 0.85, 0.0),
    ));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    floor.receive_shadow = true;
    s.add(floor);

    let mut add = |geometry: BufferGeometry, material: Material, at: Vector3, turn: f32| {
        let mut o = Object3D::mesh(Mesh::new(geometry, material));
        o.position = at;
        o.set_rotation_from_axis_angle(Vector3::new(0.0, 1.0, 0.0), turn);
        o.cast_shadow = true;
        o.receive_shadow = true;
        s.add(o);
    };
    add(
        BoxGeometry::new(1.0, 1.0, 1.0),
        standard(0xc8553d, 0.55, 0.0),
        Vector3::new(-1.4, 0.5, 0.3),
        0.5,
    );
    add(
        SphereGeometry::new(0.6, 64, 32),
        standard(0xe8e4dc, 0.12, 0.0),
        Vector3::new(0.2, 0.6, -0.4),
        0.0,
    );
    add(
        TorusKnotGeometry::new(0.42, 0.14, 160, 24, 2, 3),
        standard(0x3d8fd0, 0.35, 0.0),
        Vector3::new(1.5, 0.75, 0.5),
        0.0,
    );
    add(
        CylinderGeometry::new(0.3, 0.3, 1.6, 48, 1, false, 0.0, std::f32::consts::TAU),
        standard(0xe0b040, 0.4, 0.0),
        Vector3::new(-0.3, 0.8, -1.8),
        0.0,
    );

    let mut camera = PerspectiveCamera::new(45.0, W as f32 / H as f32, 0.1, 50.0);
    camera.position = Vector3::new(3.6, 2.8, 5.4);
    camera.look_at(Vector3::new(0.0, 0.5, 0.0));
    (s, camera)
}

/// Mean absolute difference per channel, in 8-bit steps.
fn distance(a: &[u8], b: &[u8]) -> f64 {
    let sum: u64 = a
        .chunks(4)
        .zip(b.chunks(4))
        .map(|(p, q)| {
            (0..3)
                .map(|c| (p[c] as i32 - q[c] as i32).unsigned_abs() as u64)
                .sum::<u64>()
        })
        .sum();
    sum as f64 / (a.len() / 4 * 3) as f64
}

fn main() {
    let mut frames: Vec<(&str, Vec<u8>)> = Vec::new();

    let mut hr = HeadlessRenderer::builder()
        .size(W, H)
        .build()
        .expect("a GPU adapter");
    hr.set_msaa(4);
    hr.renderer().set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
    let (mut s, camera) = scene();
    frames.push(("wgpu", hr.render_to_rgba(&mut s, &camera)));

    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    {
        let mut metal = threers::metal::MetalHeadlessRenderer::builder()
            .size(W, H)
            .msaa(4)
            .build()
            .expect("a Metal device");
        metal
            .renderer()
            .set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
        let (mut s, camera) = scene();
        frames.push((
            "metal",
            metal.render_to_rgba(&mut s, &camera).expect("a frame"),
        ));
    }

    for i in 0..frames.len() {
        for j in i + 1..frames.len() {
            println!(
                "{} against {}: {:.2} levels mean difference",
                frames[i].0,
                frames[j].0,
                distance(&frames[i].1, &frames[j].1)
            );
        }
    }
    let n = frames.len() as u32;
    let mut sheet = vec![0u8; (W * n * H * 4) as usize];
    for (slot, (_, px)) in frames.iter().enumerate() {
        for y in 0..H {
            let row = &px[(y * W * 4) as usize..((y + 1) * W * 4) as usize];
            let at = ((y * W * n + slot as u32 * W) * 4) as usize;
            sheet[at..at + row.len()].copy_from_slice(row);
        }
    }
    std::fs::create_dir_all("out").unwrap();
    std::fs::write("out/backend_parity.png", encode_png(W * n, H, &sheet)).unwrap();
    let names: Vec<&str> = frames.iter().map(|(n, _)| *n).collect();
    println!("wrote out/backend_parity.png ({})", names.join(", "));
}
