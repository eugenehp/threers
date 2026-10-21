//! Shell fur on a ball and a torus, rendered offscreen to `out/fur.png`.
//!
//! ```bash
//! cargo run --release --example fur
//! ```

use threers::fur::{fur_object, FurGroom, FurMaterial};
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
    sun.cast_shadow = true;
    scene.add(Object3D::light(Light::Directional(sun)));

    // A short, dense coat that droops a little.
    let coat = FurMaterial {
        groom: FurGroom {
            length: 0.22,
            density: 140.0,
            thickness: 0.3,
            gravity: Vector3::new(0.0, -0.6, 0.0),
            clumping: 0.4,
            ..Default::default()
        },
        root_color: Color::from_hex(0x2a1a0e),
        tip_color: Color::from_hex(0xd6a86a),
        shells: 40,
        ..Default::default()
    };
    let mut ball = fur_object(SphereGeometry::new(1.0, 96, 48), &coat);
    ball.position = Vector3::new(-1.4, 0.0, 0.0);
    scene.add(ball);

    // Long, curly, clumped fur in another colour.
    let shag = FurMaterial {
        groom: FurGroom {
            length: 0.35,
            density: 90.0,
            thickness: 0.35,
            taper: 0.05,
            curl: 1.5,
            clumping: 0.7,
            gravity: Vector3::new(0.0, -0.4, 0.0),
            ..Default::default()
        },
        root_color: Color::from_hex(0x1c1c2a),
        tip_color: Color::from_hex(0x9fb4e8),
        shells: 48,
        ..Default::default()
    };
    let mut torus = fur_object(
        TorusGeometry::new(0.75, 0.35, 64, 128, std::f32::consts::TAU),
        &shag,
    );
    torus.position = Vector3::new(1.5, 0.0, 0.0);
    torus.rotate_x(1.1);
    scene.add(torus);

    let mut camera = PerspectiveCamera::new(40.0, w as f32 / h as f32, 0.1, 50.0);
    camera.position = Vector3::new(0.0, 0.6, 6.0);
    camera.look_at(Vector3::ZERO);

    let rgba = hr.render_to_rgba(&mut scene, &camera);
    std::fs::create_dir_all("out").unwrap();
    std::fs::write("out/fur.png", encode_png(w, h, &rgba)).unwrap();
    println!("wrote out/fur.png");
}
