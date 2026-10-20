//! Order-independent transparency is the same on both renderers.
//!
//! Weighted-blended OIT is an approximation: where two translucent layers
//! overlap it averages their colours by weight, which is not what blending
//! them back to front gives. So a frame through OIT is told from one through
//! sorted blending by the overlap's colour — and the Metal renderer, which
//! used to draw `Oit` as sorted blending, now has to land on wgpu's colour,
//! with MSAA and without.

use threers::materials::TransparencyMode;
use threers::metal::{MetalError, MetalHeadlessRenderer};
use threers::prelude::*;

const N: u32 = 64;

fn scene(mode: TransparencyMode) -> (Scene, PerspectiveCamera) {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(AmbientLight::new(Color::WHITE, std::f32::consts::PI));
    for (color, z) in [
        (Color::new(1.0, 0.0, 0.0), 0.0),
        (Color::new(0.0, 0.0, 1.0), -0.5),
    ] {
        let mut m = PhysicalMaterial::new(color).with_transparency(mode);
        m.opacity = 0.5;
        m.transparent = Some(true);
        m.metalness = 0.0;
        m.roughness = 1.0;
        let mut plane = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(2.0, 2.0),
            Material::Physical(m),
        ));
        plane.position = Vector3::new(0.0, 0.0, z);
        scene.add(plane);
    }
    let mut camera = PerspectiveCamera::new(30.0, 1.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 4.0);
    camera.look_at(Vector3::ZERO);
    (scene, camera)
}

fn centre(px: &[u8]) -> [i32; 3] {
    let i = ((N / 2 * N + N / 2) * 4) as usize;
    [px[i] as i32, px[i + 1] as i32, px[i + 2] as i32]
}

fn distance(a: [i32; 3], b: [i32; 3]) -> i32 {
    (0..3).map(|k| (a[k] - b[k]).abs()).max().unwrap()
}

#[test]
fn metal_oit_matches_wgpu_oit_and_not_sorted_blending() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let metal = |msaa: u32| match MetalHeadlessRenderer::builder()
        .size(N, N)
        .msaa(msaa)
        .build()
    {
        Ok(r) => Some(r),
        Err(MetalError::NoDevice) => None,
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    let Some(mut single) = metal(1) else { return };
    let (mut s, cam) = scene(TransparencyMode::Oit);
    let wgpu_oit = centre(&hr.render_to_rgba(&mut s, &cam));
    let (mut s, cam) = scene(TransparencyMode::Blend);
    let metal_sorted = centre(&single.render_to_rgba(&mut s, &cam).unwrap());
    for msaa in [1, 4] {
        let mut metal = metal(msaa).unwrap();
        let (mut s, cam) = scene(TransparencyMode::Oit);
        let metal_oit = centre(&metal.render_to_rgba(&mut s, &cam).unwrap());
        eprintln!("overlap: wgpu OIT {wgpu_oit:?}, metal OIT x{msaa} {metal_oit:?}, metal sorted {metal_sorted:?}");
        assert!(
            distance(metal_oit, wgpu_oit) <= 6,
            "metal OIT x{msaa} {metal_oit:?} against wgpu's {wgpu_oit:?}"
        );
        assert!(
            distance(metal_sorted, wgpu_oit) > 20,
            "the fixture cannot tell OIT from sorted blending ({metal_sorted:?} against {wgpu_oit:?})"
        );
    }
}
