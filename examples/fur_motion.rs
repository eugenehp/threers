//! Fur in motion: a ball that accelerates, brakes and settles, its coat
//! trailing and swinging back (`FurAnimator`), with a light wind. Four frames
//! side by side in `out/fur_motion.png`.
//!
//! ```bash
//! cargo run --release --example fur_motion
//! ```

use threers::fur::{fur_object, FurAnimator, FurDynamics, FurGroom, FurMaterial};
use threers::lights::Light;
use threers::prelude::*;

fn main() {
    let (w, h) = (320u32, 320u32);
    let mut hr = HeadlessRenderer::builder()
        .size(w, h)
        .build()
        .expect("a GPU adapter");
    hr.set_msaa(4);

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x1a1d24);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.7));
    let mut sun = DirectionalLight::new(Color::WHITE, 2.2);
    sun.direction = Vector3::new(-0.4, -0.8, -0.5).normalize();
    scene.add(Object3D::light(Light::Directional(sun)));

    let coat = FurMaterial {
        groom: FurGroom {
            length: 0.45,
            density: 70.0,
            thickness: 0.32,
            gravity: Vector3::new(0.0, -0.25, 0.0),
            clumping: 0.5,
            ..Default::default()
        },
        root_color: Color::from_hex(0x3b2414),
        tip_color: Color::from_hex(0xe8c48c),
        shells: 40,
        wind: Vector3::new(2.5, 0.0, 1.0),
        flutter: 0.12,
        dynamics: FurDynamics {
            stiffness: 35.0,
            damping: 0.4,
            ..Default::default()
        },
        ..Default::default()
    };
    let ball = scene.add(fur_object(SphereGeometry::new(0.7, 96, 48), &coat));
    let mut animator = FurAnimator::new();
    animator.track(ball, &coat);

    let mut camera = PerspectiveCamera::new(40.0, 1.0, 0.1, 50.0);
    camera.position = Vector3::new(0.0, 0.3, 5.0);
    camera.look_at(Vector3::ZERO);

    // Rest, a hard acceleration right, a hard stop, and settled again.
    let dt = 1.0 / 60.0;
    let (mut x, mut v, mut t) = (-0.6f32, 0.0f32, 0.0f32);
    let mut sheet = vec![0u8; (w * 4 * h * 4) as usize];
    let mut grab =
        |hr: &mut HeadlessRenderer, scene: &mut Scene, slot: u32, camera: &PerspectiveCamera| {
            let px = hr.render_to_rgba(scene, camera);
            for y in 0..h {
                let row = &px[(y * w * 4) as usize..((y + 1) * w * 4) as usize];
                let at = ((y * w * 4 + slot * w) * 4) as usize;
                sheet[at..at + row.len()].copy_from_slice(row);
            }
        };
    let mut run = |frames: u32, accel: f32, scene: &mut Scene, animator: &mut FurAnimator| {
        for _ in 0..frames {
            v += accel * dt;
            if accel == 0.0 {
                v = 0.0;
            }
            x += v * dt;
            t += dt;
            scene.get_mut(ball).unwrap().position.x = x;
            animator.update(scene, t, dt);
        }
    };
    run(60, 0.0, &mut scene, &mut animator);
    grab(&mut hr, &mut scene, 0, &camera);
    run(14, 14.0, &mut scene, &mut animator);
    grab(&mut hr, &mut scene, 1, &camera);
    run(1, 0.0, &mut scene, &mut animator);
    run(5, 0.0, &mut scene, &mut animator);
    grab(&mut hr, &mut scene, 2, &camera);
    run(120, 0.0, &mut scene, &mut animator);
    grab(&mut hr, &mut scene, 3, &camera);

    std::fs::create_dir_all("out").unwrap();
    std::fs::write("out/fur_motion.png", encode_png(w * 4, h, &sheet)).unwrap();
    println!("wrote out/fur_motion.png (rest, accelerating, braking, settled)");
}
