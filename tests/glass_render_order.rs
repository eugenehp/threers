//! `renderOrder` leads within the translucent draws, as in three.js's
//! transparent list, glass included.
//!
//! A translucent red quad sits behind a single-layer-glass quad
//! (`TransparencyMode::Glass`). At equal render orders the glass goes first —
//! its depth prepass, then its colour — and the red quad behind fails the depth
//! test: the glass hides it, as it always has. Given a higher render order, the
//! glass is drawn after the red quad and blends over it, so the red shows
//! through. Skipped without a GPU adapter.

use threers::cameras::OrthographicCamera;
use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::lights::AmbientLight;
use threers::materials::{BasicMaterial, Material, PhysicalMaterial, TransparencyMode};
use threers::math::{Color, Vector3};
use threers::scene::Scene;

const N: u32 = 32;

fn render(r: &mut threers::HeadlessRenderer, glass_order: i32) -> [u8; 3] {
    render_with(r, Some(glass_order))
}

fn render_with(r: &mut threers::HeadlessRenderer, glass_order: Option<i32>) -> [u8; 3] {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));

    let mut red = BasicMaterial::new(Color::from_hex(0xff0000));
    red.transparent = true;
    red.opacity = 0.8;
    let mut back = Object3D::mesh(Mesh::new(PlaneGeometry::new(1.6, 1.6), Material::from(red)));
    back.position = Vector3::new(0.0, 0.0, -0.5);
    scene.add(back);

    let mut glass = PhysicalMaterial::new(Color::WHITE);
    glass.opacity = 0.3;
    glass.transparency = TransparencyMode::Glass;
    let mut front = Object3D::mesh(Mesh::new(PlaneGeometry::new(1.6, 1.6), Material::Physical(glass)));
    front.position = Vector3::new(0.0, 0.0, 0.5);
    if let Some(o) = glass_order {
        front.render_order = o;
        scene.add(front);
    }

    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 5.0);
    cam.look_at(Vector3::ZERO);
    let px = r.render_to_rgba(&mut scene, &cam);
    let c = (((N / 2) * N + N / 2) * 4) as usize;
    [px[c], px[c + 1], px[c + 2]]
}

#[test]
fn render_order_puts_glass_after_what_is_inside_it() {
    // sRGB targets blend translucent draws in a separate encoded pass; plain
    // ones in the main pass. The order has to hold on both.
    for format in [wgpu::TextureFormat::Rgba8UnormSrgb, wgpu::TextureFormat::Rgba8Unorm] {
        let Ok(mut r) = threers::HeadlessRenderer::builder().size(N, N).color_format(format).build() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let hidden = render(&mut r, 0);
        let shown = render(&mut r, 1);
        // the glass is grey-white: red is only what the quad behind contributes
        let redness = |c: [u8; 3]| c[0] as i32 - c[1] as i32;
        assert!(redness(hidden) < 10, "{format:?}, equal orders: the glass prepass should hide the quad behind, got {hidden:?}");
        assert!(redness(shown) > 60, "{format:?}: glass ordered last should blend over the red quad, got {shown:?}");
    }
}
