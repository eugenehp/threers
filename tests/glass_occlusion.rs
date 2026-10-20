//! Glass behind an opaque object stays behind it — under MSAA too.
//!
//! The refraction pass tests the glass against a single-sampled copy of the
//! opaque depth. Under MSAA the opaque pass's depth is multisampled, so that
//! copy is made one of two ways: resolved from it where a shader can read a
//! multisampled texture, or the opaque scene's depth drawn again where it
//! cannot (WebGL2; forced here with `force_depth_replay`). Get either wrong
//! and the glass draws over whatever stands in front of it.
//!
//! Skipped when no GPU adapter is available.

use threers::materials::TransparencyMode;
use threers::prelude::*;

const N: u32 = 128;

/// The centre pixel: a red box standing in front of a glass ball.
fn centre(
    hr: &mut HeadlessRenderer,
    mode: TransparencyMode,
    msaa: u32,
    replay: bool,
) -> ([u8; 3], bool) {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x203040);
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
    let mut wall = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(8.0, 8.0),
        Material::Basic(BasicMaterial::new(Color::from_hex(0x2040ff))),
    ));
    wall.position = Vector3::new(0.0, 0.0, -3.0);
    scene.add(wall);
    // Strongly green: glass drawn over the red box turns it dark, where
    // refracting it unfiltered would still read red.
    let glass = PhysicalMaterial::new(Color::new(0.05, 0.9, 0.05))
        .with_roughness(0.05)
        .with_transmission(1.0, 1.5, 1.0)
        .with_transparency(mode);
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 32, 16),
        Material::Physical(glass),
    )));
    let mut front = Object3D::mesh(Mesh::new(
        BoxGeometry::new(0.6, 0.6, 0.6),
        Material::Basic(BasicMaterial::new(Color::from_hex(0xff0000))),
    ));
    front.position = Vector3::new(0.0, 0.0, 1.8);
    scene.add(front);
    let mut camera = PerspectiveCamera::new(40.0, 1.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 5.0);
    camera.look_at(Vector3::ZERO);
    hr.set_msaa(msaa);
    hr.renderer().force_depth_replay(replay);
    let px = hr.render_to_rgba(&mut scene, &camera);
    let multisampled = hr.renderer().last_frame_multisampled();
    hr.renderer().force_depth_replay(false);
    hr.set_msaa(1);
    let i = ((N / 2 * N + N / 2) * 4) as usize;
    ([px[i], px[i + 1], px[i + 2]], multisampled)
}

#[test]
fn glass_behind_an_opaque_box_stays_hidden() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    for mode in [
        TransparencyMode::Refract,
        TransparencyMode::Glass,
        TransparencyMode::Oit,
        TransparencyMode::Blend,
    ] {
        for (msaa, replay) in [(1, false), (4, false), (4, true)] {
            let (c, multisampled) = centre(&mut hr, mode, msaa, replay);
            eprintln!("{mode:?} x{msaa} replay {replay}: {c:?}");
            assert!(
                msaa == 1 || multisampled,
                "{mode:?} x{msaa}: not multisampled"
            );
            assert!(
                c[0] > 240 && c[1] < 20 && c[2] < 20,
                "{mode:?} x{msaa} (depth {}): the box in front reads {c:?} — the glass drew over it",
                if replay { "redrawn" } else { "resolved" }
            );
        }
    }
}
