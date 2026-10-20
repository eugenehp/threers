//! SSAO under MSAA on Metal. The pass samples depth as a `depth2d`, which a
//! multisampled depth is not, so it used to be skipped whenever MSAA was on.
//! A multisampled `MetalRenderTarget` now has the main pass resolve its depth
//! into a single-sampled one while depth is stored, and SSAO reads that: the
//! crease between a box and the floor darkens at 4x as it does at 1x.
//!
//! Skipped without a Metal device.

use threers::metal::{MetalHeadlessRenderer, SsaoSettings};
use threers::prelude::*;

const W: u32 = 128;
const H: u32 = 96;

fn scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::WHITE;
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(10.0, 10.0),
        Material::Standard(StandardMaterial::new(Color::from_hex(0xdddddd))),
    ));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    scene.add(floor);
    for x in [-1.2f32, 0.0, 1.2] {
        let mut b = Object3D::mesh(Mesh::new(
            BoxGeometry::new(0.8, 0.8, 0.8),
            Material::Standard(StandardMaterial::new(Color::from_hex(0xdddddd))),
        ));
        b.position = Vector3::new(x, 0.4, 0.0);
        scene.add(b);
    }
    scene
}

fn camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(45.0, W as f32 / H as f32, 0.5, 30.0);
    c.position = Vector3::new(0.0, 2.5, 4.0);
    c.look_at(Vector3::new(0.0, 0.3, 0.0));
    c
}

/// A frame's brightness per pixel, 0..255.
fn frame(msaa: u32, ssao: bool) -> Option<Vec<f32>> {
    let mut r = MetalHeadlessRenderer::builder()
        .size(W, H)
        .msaa(msaa)
        .build()
        .ok()?;
    if ssao {
        r.set_ssao(Some(SsaoSettings {
            strength: 2.0,
            radius: 0.5,
            near: 0.5,
            far: 30.0,
            tan_half_fov: (22.5f32).to_radians().tan(),
            aspect: W as f32 / H as f32,
        }));
    }
    let px = r.render_to_rgba(&mut scene(), &camera()).ok()?;
    Some(
        px.chunks(4)
            .map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0)
            .collect(),
    )
}

/// Pixels SSAO took more than 6 levels off.
fn occluded(plain: &[f32], ao: &[f32]) -> usize {
    plain.iter().zip(ao).filter(|(p, a)| *p - *a > 6.0).count()
}

#[test]
fn ssao_darkens_under_msaa_as_without() {
    let Some(plain1) = frame(1, false) else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let n1 = occluded(&plain1, &frame(1, true).unwrap());
    let n4 = occluded(&frame(4, false).unwrap(), &frame(4, true).unwrap());
    eprintln!("occluded pixels: {n1} at 1x, {n4} at 4x");
    assert!(n1 > 100, "SSAO at 1x darkened only {n1} pixels");
    assert!(
        n4 as f32 > 0.7 * n1 as f32 && (n4 as f32) < 1.3 * n1 as f32,
        "SSAO at 4x darkened {n4} pixels, at 1x {n1}"
    );
}

/// With OIT glass in the frame the multisampled depth is kept for the OIT
/// pass to test against, and resolved as well: SSAO still darkens, and the
/// glass still draws.
#[test]
fn ssao_under_msaa_with_oit_glass() {
    let Ok(mut r) = MetalHeadlessRenderer::builder().size(W, H).msaa(4).build() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let mut with_glass = scene();
    let glass = PhysicalMaterial::new(Color::new(0.1, 0.9, 0.2))
        .with_roughness(0.05)
        .with_transmission(1.0, 1.5, 0.3)
        .with_transparency(threers::materials::TransparencyMode::Oit);
    let mut ball = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.4, 24, 12),
        Material::Physical(glass),
    ));
    ball.position = Vector3::new(0.0, 1.3, 0.5);
    with_glass.add(ball);
    let plain = r.render_to_rgba(&mut with_glass, &camera()).unwrap();
    r.set_ssao(Some(SsaoSettings {
        strength: 2.0,
        radius: 0.5,
        near: 0.5,
        far: 30.0,
        tan_half_fov: (22.5f32).to_radians().tan(),
        aspect: W as f32 / H as f32,
    }));
    let ao = r.render_to_rgba(&mut with_glass, &camera()).unwrap();
    let lum = |px: &[u8]| {
        px.chunks(4)
            .map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0)
            .collect::<Vec<_>>()
    };
    let n = occluded(&lum(&plain), &lum(&ao));
    let green = ao
        .chunks(4)
        .filter(|p| p[1] as i32 > p[0] as i32 + 30)
        .count();
    eprintln!("with OIT glass: {n} occluded pixels, {green} green");
    assert!(
        n > 100,
        "SSAO with OIT glass at 4x darkened only {n} pixels"
    );
    assert!(
        green > 50,
        "the OIT glass did not draw ({green} green pixels)"
    );
}
