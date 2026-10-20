//! Coloured glass tints what is behind it by channel, in every glass mode.
//!
//! The blended modes (`Glass`, `Oit`) used to blend glass with one alpha,
//! which can only darken what is behind evenly: green glass over a white wall
//! came out grey. They now multiply by the glass's transmittance per channel
//! (two-draw glass, coloured OIT revealage), as `Refract` — which reads the
//! scene behind and filters it — always did.
//!
//! The Metal renderer too: it has no OIT pass or screen-space refraction,
//! so `Oit` draws as `Glass` there and `Refract` as three.js's transmission,
//! and all three must still tint alike.
//!
//! Skipped when no GPU adapter is available.

use threers::materials::TransparencyMode;
use threers::prelude::*;

const N: u32 = 96;

/// The white wall behind a green glass plate.
fn scene(mode: TransparencyMode) -> (Scene, PerspectiveCamera) {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
    let mut wall = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(6.0, 6.0),
        Material::Basic(BasicMaterial::new(Color::WHITE)),
    ));
    wall.position = Vector3::new(0.0, 0.0, -1.0);
    scene.add(wall);
    let glass = PhysicalMaterial::new(Color::new(0.15, 0.9, 0.15))
        .with_roughness(0.0)
        .with_transmission(1.0, 1.5, 0.0)
        .with_transparency(mode);
    scene.add(Object3D::mesh(Mesh::new(
        PlaneGeometry::new(2.0, 2.0),
        Material::Physical(glass),
    )));
    let mut camera = PerspectiveCamera::new(40.0, 1.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 3.0);
    camera.look_at(Vector3::ZERO);
    (scene, camera)
}

fn centre(px: &[u8]) -> [u8; 3] {
    let i = ((N / 2 * N + N / 2) * 4) as usize;
    [px[i], px[i + 1], px[i + 2]]
}

/// The centre pixel: the wall seen through the glass.
fn through_glass(hr: &mut HeadlessRenderer, mode: TransparencyMode) -> [u8; 3] {
    let (mut scene, camera) = scene(mode);
    centre(&hr.render_to_rgba(&mut scene, &camera))
}

#[test]
fn coloured_glass_tints_by_channel_in_every_mode() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let reference = through_glass(&mut hr, TransparencyMode::Refract);
    eprintln!("Refract: {reference:?}");
    assert!(
        reference[1] > 2 * reference[0].max(1),
        "the reference is not green: {reference:?}"
    );
    // Single- and multisampled: the multisampled glass paths (OIT's resolved
    // accumulation, refraction over a resolved depth) must agree too.
    for (mode, msaa, replay) in [
        (TransparencyMode::Glass, 1, false),
        (TransparencyMode::Oit, 1, false),
        (TransparencyMode::Glass, 4, false),
        (TransparencyMode::Oit, 4, false),
        (TransparencyMode::Refract, 4, false),
        // WebGL2's path: the opaque depth redrawn rather than resolved.
        (TransparencyMode::Refract, 4, true),
    ] {
        hr.set_msaa(msaa);
        hr.renderer().force_depth_replay(replay);
        let c = through_glass(&mut hr, mode);
        assert!(msaa == 1 || hr.renderer().last_frame_multisampled(), "{mode:?} x{msaa} did not multisample");
        hr.renderer().force_depth_replay(false);
        hr.set_msaa(1);
        eprintln!("{mode:?} x{msaa}: {c:?}");
        assert!(
            c[1] as u32 > 2 * c[0].max(1) as u32 && c[1] as u32 > 2 * c[2].max(1) as u32,
            "{mode:?} x{msaa}: the wall through green glass is {c:?} — darkened, not tinted"
        );
        for k in 0..3 {
            assert!(
                (c[k] as i32 - reference[k] as i32).abs() <= 24,
                "{mode:?} x{msaa}: {c:?} against Refract's {reference:?}"
            );
        }
    }
}

#[cfg(feature = "metal")]
#[test]
fn coloured_glass_tints_by_channel_on_metal() {
    use threers::metal::{MetalError, MetalHeadlessRenderer};
    let mut r = match MetalHeadlessRenderer::builder().size(N, N).build() {
        Ok(r) => r,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    let mut through = |mode| {
        let (mut scene, camera) = scene(mode);
        centre(&r.render_to_rgba(&mut scene, &camera).expect("metal"))
    };
    let reference = through(TransparencyMode::Refract);
    eprintln!("metal Refract: {reference:?}");
    assert!(
        reference[1] > 2 * reference[0].max(1),
        "the reference is not green: {reference:?}"
    );
    for mode in [
        TransparencyMode::Blend,
        TransparencyMode::Glass,
        TransparencyMode::Oit,
    ] {
        let c = through(mode);
        eprintln!("metal {mode:?}: {c:?}");
        assert!(
            c[1] as u32 > 2 * c[0].max(1) as u32 && c[1] as u32 > 2 * c[2].max(1) as u32,
            "metal {mode:?}: the wall through green glass is {c:?}"
        );
        for k in 0..3 {
            assert!(
                (c[k] as i32 - reference[k] as i32).abs() <= 24,
                "metal {mode:?}: {c:?} against Refract's {reference:?}"
            );
        }
    }
}
