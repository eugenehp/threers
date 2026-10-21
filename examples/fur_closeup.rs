//! Fur close up, one sample against 4× MSAA: each shell sweeps its strands
//! across its slab as the line of sight crosses it, so a strand is one
//! column rather than a stack of dots; under MSAA each sample is tested
//! against the strands, so their edges resolve as geometry does. Side by
//! side in `out/fur_closeup.png`.
//!
//! ```bash
//! cargo run --release --example fur_closeup
//! ```

use threers::fur::{fur_object, FurGroom, FurMaterial};
use threers::lights::Light;
use threers::prelude::*;

fn main() {
    let (w, h) = (480u32, 400u32);
    let mut sheet = vec![0u8; (w * 2 * h * 4) as usize];
    for (slot, samples) in [1u32, 4].into_iter().enumerate() {
        let mut hr = HeadlessRenderer::builder()
            .size(w, h)
            .build()
            .expect("a GPU adapter");
        hr.set_msaa(samples);
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x1a1d24);
        scene.add_light(AmbientLight::new(Color::WHITE, 0.6));
        let mut sun = DirectionalLight::new(Color::WHITE, 2.5);
        sun.direction = Vector3::new(-0.5, -0.8, -0.4).normalize();
        scene.add(Object3D::light(Light::Directional(sun)));
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
        scene.add(fur_object(SphereGeometry::new(1.0, 96, 48), &coat));
        let mut camera = PerspectiveCamera::new(40.0, w as f32 / h as f32, 0.05, 50.0);
        camera.position = Vector3::new(0.3, 0.6, 1.9);
        camera.look_at(Vector3::new(0.0, 0.75, 0.0));
        let px = hr.render_to_rgba(&mut scene, &camera);
        for y in 0..h {
            let row = &px[(y * w * 4) as usize..((y + 1) * w * 4) as usize];
            let at = ((y * w * 2 + slot as u32 * w) * 4) as usize;
            sheet[at..at + row.len()].copy_from_slice(row);
        }
    }
    std::fs::create_dir_all("out").unwrap();
    std::fs::write("out/fur_closeup.png", encode_png(w * 2, h, &sheet)).unwrap();
    println!("wrote out/fur_closeup.png (one sample, 4× MSAA)");
}
