//! A frame carries eight lights of each kind, not four.
//!
//! Six point lights hang in a row over a dark floor, each close enough to
//! light its own patch. Every patch has to come out lit: with the old limit
//! of four, the fifth and sixth were dropped without a word and their patches
//! stayed dark.
//!
//! Skipped when no GPU adapter is available.

use threers::lights::{Light, PointLight};
use threers::prelude::*;

const W: u32 = 384;
const H: u32 = 96;

#[test]
fn six_point_lights_all_light_the_floor() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(W, H).build() else {
        return;
    };
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(14.0, 4.0),
        Material::Standard(StandardMaterial::new(Color::WHITE)),
    ));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    scene.add(floor);
    let xs: Vec<f32> = (0..6).map(|i| -5.0 + i as f32 * 2.0).collect();
    for &x in &xs {
        let mut l = PointLight::new(Color::WHITE, 4.0);
        l.distance = 1.6;
        l.decay = 2.0;
        let mut o = Object3D::light(Light::Point(l));
        o.position = Vector3::new(x, 0.5, 0.0);
        scene.add(o);
    }
    // Straight down onto the floor: x across the image.
    let mut camera = OrthographicCamera::new(-7.0, 7.0, 1.75, -1.75, 0.1, 20.0);
    camera.position = Vector3::new(0.0, 5.0, 0.0);
    camera.up = Vector3::new(0.0, 0.0, -1.0);
    camera.look_at(Vector3::ZERO);
    let img = hr.render_to_rgba(&mut scene, &camera);
    let lum = |x: f32| {
        let px = ((x + 7.0) / 14.0 * W as f32) as u32;
        let o = (((H / 2) * W + px) * 4) as usize;
        img[o] as u32 + img[o + 1] as u32 + img[o + 2] as u32
    };
    for (i, &x) in xs.iter().enumerate() {
        let under = lum(x);
        let between = lum(x + 1.0);
        assert!(
            under > 60 && under > between + 30,
            "light {i}: its patch {under} against the dark between {between}"
        );
    }
}
