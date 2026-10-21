//! Strand hair: a head of long hair hanging under gravity, and a curly,
//! clumped mane on a torus — rendered offscreen to `out/hair.png`.
//!
//! ```bash
//! cargo run --release --example hair
//! ```

use threers::fur::strands::{hair_object, StrandOptions};
use threers::fur::{FurAnimator, FurGroom, FurMaterial};
use threers::lights::Light;
use threers::prelude::*;

fn main() {
    let (w, h) = (960, 540);
    let mut hr = HeadlessRenderer::builder()
        .size(w, h)
        .build()
        .expect("a GPU adapter");
    hr.set_msaa(4);

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x1a1d24);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.6));
    let mut sun = DirectionalLight::new(Color::WHITE, 2.5);
    sun.direction = Vector3::new(-0.5, -0.8, -0.4).normalize();
    // The hair shadows the head, the ring and itself.
    sun.cast_shadow = true;
    sun.shadow.map_size = 2048;
    sun.shadow.camera_size = 6.0;
    sun.shadow.camera_near = 0.5;
    sun.shadow.camera_far = 30.0;
    let mut sun = Object3D::light(Light::Directional(sun));
    sun.position = Vector3::new(5.0, 8.0, 4.0);
    scene.add(sun);

    // A ball as the scalp, its lower half painted bald (vertex alpha 0).
    let mut scalp = SphereGeometry::new(0.8, 64, 32);
    let pos = scalp.get_attribute("position").unwrap().array.clone();
    let paint: Vec<f32> = pos
        .chunks(3)
        .flat_map(|p| [1.0, 1.0, 1.0, ((p[1] / 0.8 + 0.25) * 3.0).clamp(0.0, 1.0)])
        .collect();
    scalp.set_attribute("color", BufferAttribute::new(paint, 4));
    let skin = StandardMaterial {
        color: Color::from_hex(0x8a6450),
        roughness: 0.7,
        ..Default::default()
    };
    let mut head = Object3D::mesh(Mesh::new(scalp.clone(), Material::Standard(skin)));
    head.position = Vector3::new(-1.4, 0.2, 0.0);
    head.receive_shadow = true;
    let long = FurMaterial {
        groom: FurGroom {
            length: 1.1,
            gravity: Vector3::new(0.0, -3.0, 0.0),
            taper: 0.3,
            curl: 0.15,
            clumping: 0.35,
            clump_size: 3.0,
            length_variation: 0.2,
            ..Default::default()
        },
        root_color: Color::from_hex(0x24140a),
        tip_color: Color::from_hex(0x8a5a30),
        ..Default::default()
    };
    let options = StrandOptions {
        count: 30_000,
        segments: 12,
        width: 0.004,
        ..Default::default()
    };
    let mut hair = hair_object(&scalp, &long, &options);
    hair.position = head.position;
    scene.add(head);
    let hair_id = scene.add(hair);

    // A curly, clumped mane.
    let ring = TorusGeometry::new(0.7, 0.25, 32, 96, std::f32::consts::TAU);
    let mane = FurMaterial {
        groom: FurGroom {
            length: 0.45,
            curl: 1.5,
            clumping: 0.7,
            gravity: Vector3::new(0.0, -0.3, 0.0),
            ..Default::default()
        },
        root_color: Color::from_hex(0x1c1c2a),
        tip_color: Color::from_hex(0x9fb4e8),
        ..Default::default()
    };
    let mut torus = Object3D::mesh(Mesh::new(
        ring.clone(),
        Material::Standard(StandardMaterial {
            color: Color::from_hex(0x1c1c2a),
            ..Default::default()
        }),
    ));
    torus.receive_shadow = true;
    let mut strands = hair_object(
        &ring,
        &mane,
        &StrandOptions {
            count: 20_000,
            segments: 10,
            width: 0.005,
            ..Default::default()
        },
    );
    for o in [&mut torus, &mut strands] {
        o.position = Vector3::new(1.5, 0.0, 0.0);
        o.rotate_x(1.1);
    }
    scene.add(torus);
    let mane_id = scene.add(strands);
    // The animator bends each shadow caster under world gravity — the ring
    // is turned — and would carry wind, flutter and inertia frame to frame.
    let mut animator = FurAnimator::new();
    animator.track(hair_id, &long);
    animator.track(mane_id, &mane);
    animator.update(&mut scene, 0.0, 1.0 / 60.0);

    let mut camera = PerspectiveCamera::new(40.0, w as f32 / h as f32, 0.1, 50.0);
    camera.position = Vector3::new(0.0, 0.6, 6.0);
    camera.look_at(Vector3::ZERO);

    let rgba = hr.render_to_rgba(&mut scene, &camera);
    std::fs::create_dir_all("out").unwrap();
    std::fs::write("out/hair.png", encode_png(w, h, &rgba)).unwrap();
    println!("wrote out/hair.png");
}
