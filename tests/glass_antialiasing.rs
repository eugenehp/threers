//! Antialiasing asked for is antialiasing delivered, whatever glass is in view.
//!
//! A frame multisamples only when every draw can, and the glass modes once
//! could not: one `Glass`, `Oit` or `Refract` material anywhere turned MSAA
//! off for the whole frame, opaque edges and all, without a word. All three
//! multisample now — `Oit` into multisampled accumulation targets, `Refract`
//! over a multisampled opaque pass whose depth is resolved for its march.
//! Where a backend cannot (refraction on WebGL2, which cannot read a
//! multisampled depth), an offscreen frame gets FXAA instead.
//!
//! A white box tilted against a black background, beside a glass ball: with
//! `set_msaa(4)` its edge must come out with partial-coverage pixels in
//! every mode.
//!
//! Skipped when no GPU adapter is available.

use threers::materials::TransparencyMode;
use threers::prelude::*;

const N: u32 = 256;

/// Pixels strictly between the black background and the white box, inside
/// the box's half of the frame.
fn partial(rgba: &[u8]) -> usize {
    (0..N)
        .flat_map(|y| (0..N / 2).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            let v = rgba[((y * N + x) * 4) as usize];
            v > 12 && v < 243
        })
        .count()
}

fn frame(mode: Option<TransparencyMode>, msaa: u32) -> Option<Vec<u8>> {
    frame_and_sampling(mode, msaa).map(|(px, _)| px)
}

/// The frame, and whether the renderer multisampled it.
fn frame_and_sampling(mode: Option<TransparencyMode>, msaa: u32) -> Option<(Vec<u8>, bool)> {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let mut cube = Object3D::mesh(Mesh::new(
        BoxGeometry::new(0.9, 0.9, 0.9),
        Material::Basic(BasicMaterial::new(Color::WHITE)),
    ));
    cube.position = Vector3::new(-0.75, 0.0, 0.0);
    cube.rotate_z(0.3);
    scene.add(cube);
    if let Some(mode) = mode {
        let glass = PhysicalMaterial::new(Color::WHITE)
            .with_transmission(1.0, 1.5, 0.8)
            .with_transparency(mode);
        let mut ball = Object3D::mesh(Mesh::new(
            SphereGeometry::new(0.4, 32, 16),
            Material::Physical(glass),
        ));
        ball.position = Vector3::new(0.75, 0.0, 0.0);
        scene.add(ball);
    }
    let mut camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 3.0);
    camera.look_at(Vector3::ZERO);
    let mut hr = HeadlessRenderer::builder().size(N, N).build().ok()?;
    hr.set_msaa(msaa);
    let px = hr.render_to_rgba(&mut scene, &camera);
    Some((px, hr.renderer().last_frame_multisampled()))
}

#[test]
fn every_glass_mode_keeps_its_edges_antialiased() {
    let Some(hard) = frame(None, 1) else { return };
    let Some(smooth) = frame(None, 4) else { return };
    let (hard, smooth) = (partial(&hard), partial(&smooth));
    assert_eq!(hard, 0, "no antialiasing asked for, none given");
    assert!(
        smooth > 40,
        "4x MSAA smooths the box: {smooth} partial pixels"
    );
    for mode in [
        TransparencyMode::Blend,
        TransparencyMode::Glass,
        TransparencyMode::Oit,
        TransparencyMode::Refract,
    ] {
        let (px, multisampled) = frame_and_sampling(Some(mode), 4).unwrap();
        assert!(multisampled, "{mode:?} glass in view turned MSAA off");
        let edges = partial(&px);
        assert!(
            edges > smooth / 2,
            "{mode:?} glass in view: the box's edge has {edges} partial pixels (alone: {smooth})"
        );
    }
}
