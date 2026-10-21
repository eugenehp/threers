//! A hand in the coat (`FurAnimator::stroke`): the same ball before, while a
//! hand sweeps across its upper left (strands pushed along and parted round
//! it), and half a second after it lets go. Three frames side by side in
//! `out/fur_stroke.png`.
//!
//! ```bash
//! cargo run --release --example fur_stroke
//! ```

use threers::fur::{fur_object, FurAnimator, FurGroom, FurMaterial};
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
            length: 0.4,
            density: 70.0,
            thickness: 0.32,
            gravity: Vector3::new(0.0, -0.3, 0.0),
            clumping: 0.4,
            ..Default::default()
        },
        root_color: Color::from_hex(0x3b2414),
        tip_color: Color::from_hex(0xe8c48c),
        shells: 40,
        ..Default::default()
    };
    let ball = scene.add(fur_object(SphereGeometry::new(0.75, 96, 48), &coat));
    let mut animator = FurAnimator::new();
    animator.track(ball, &coat);
    let mut camera = PerspectiveCamera::new(40.0, 1.0, 0.1, 50.0);
    camera.position = Vector3::new(0.0, 0.2, 4.0);
    camera.look_at(Vector3::ZERO);

    let dt = 1.0 / 60.0;
    let mut t = 0.0f32;
    let mut sheet = vec![0u8; (w * 3 * h * 4) as usize];
    let mut grab = |hr: &mut HeadlessRenderer, scene: &mut Scene, slot: u32| {
        let px = hr.render_to_rgba(scene, &camera);
        for y in 0..h {
            let row = &px[(y * w * 4) as usize..((y + 1) * w * 4) as usize];
            let at = ((y * w * 3 + slot * w) * 4) as usize;
            sheet[at..at + row.len()].copy_from_slice(row);
        }
    };
    let mut step = |scene: &mut Scene, animator: &mut FurAnimator, frames: u32| {
        for _ in 0..frames {
            t += dt;
            animator.update(scene, t, dt);
        }
    };
    step(&mut scene, &mut animator, 10);
    grab(&mut hr, &mut scene, 0);
    // A hand sweeping right over the upper left of the ball, a frame at a
    // time, pushing the strands with it and parting them round it.
    for k in 0..12 {
        let x = -0.6 + 0.05 * k as f32;
        let at = Vector3::new(x, 0.45, 0.65);
        animator.stroke(ball, at, 0.45, Vector3::new(2.5, 0.0, 0.0), 1.6);
        step(&mut scene, &mut animator, 1);
    }
    grab(&mut hr, &mut scene, 1);
    step(&mut scene, &mut animator, 30);
    grab(&mut hr, &mut scene, 2);

    std::fs::create_dir_all("out").unwrap();
    std::fs::write("out/fur_stroke.png", encode_png(w * 3, h, &sheet)).unwrap();
    println!("wrote out/fur_stroke.png (before, stroked, let go)");
}
