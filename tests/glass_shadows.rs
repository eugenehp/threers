//! A light's highlight on glass goes out where the light is blocked, in every
//! glass mode.
//!
//! `Refract` used to light its glass unshadowed: its pass swaps the shadow
//! maps for the scene capture to fit WebGL2's sixteen texture units, so a
//! ball in full shadow still caught a bright sun highlight. The shadows now
//! reach it packed in one texture (`fs_glass_shadow`).
//!
//! A glass ball under a directional light, with a plank above it that the
//! camera cannot see: the frames with and without the plank casting differ
//! only by its shadow, which must take the highlight away.
//!
//! On the Metal renderer too.
//!
//! Skipped when no GPU adapter is available.

use threers::materials::TransparencyMode;
use threers::prelude::*;

const N: u32 = 160;

fn frame(hr: &mut HeadlessRenderer, mode: TransparencyMode, blocked: bool) -> Vec<u8> {
    let (mut scene, camera) = scene(mode, blocked);
    hr.render_to_rgba(&mut scene, &camera)
}

fn scene(mode: TransparencyMode, blocked: bool) -> (Scene, PerspectiveCamera) {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let mut sun = DirectionalLight::new(Color::WHITE, 3.0);
    sun.cast_shadow = true;
    sun.shadow.camera_size = 4.0;
    sun.shadow.camera_near = 0.5;
    sun.shadow.camera_far = 20.0;
    sun.shadow.normal_bias = 0.02;
    let mut light = Object3D::light(threers::lights::Light::Directional(sun));
    light.position = Vector3::new(0.0, 6.0, 0.0);
    scene.add(light);

    let glass = PhysicalMaterial::new(Color::WHITE)
        .with_roughness(0.15)
        .with_transmission(1.0, 1.5, 0.8)
        .with_transparency(mode);
    let mut ball = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.5, 48, 24),
        Material::Physical(glass),
    ));
    ball.receive_shadow = true;
    scene.add(ball);

    // Above the frame, between the light and the ball.
    let mut plank = Object3D::mesh(Mesh::new(
        BoxGeometry::new(2.0, 0.2, 2.0),
        Material::Standard(StandardMaterial::new(Color::WHITE)),
    ));
    plank.position = Vector3::new(0.0, 2.5, 0.0);
    plank.cast_shadow = blocked;
    scene.add(plank);

    let mut camera = PerspectiveCamera::new(40.0, 1.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 3.0);
    camera.look_at(Vector3::ZERO);
    (scene, camera)
}

/// The brightest pixel: the highlight, when there is one.
fn peak(rgba: &[u8]) -> u32 {
    rgba.chunks(4)
        .map(|p| p[0] as u32 + p[1] as u32 + p[2] as u32)
        .max()
        .unwrap_or(0)
}

#[test]
fn glass_highlights_respect_shadows_in_every_mode() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    for mode in [
        TransparencyMode::Blend,
        TransparencyMode::Glass,
        TransparencyMode::Oit,
        TransparencyMode::Refract,
    ] {
        let lit = peak(&frame(&mut hr, mode, false));
        let shadowed = peak(&frame(&mut hr, mode, true));
        eprintln!("{mode:?}: peak {lit} lit, {shadowed} shadowed");
        assert!(
            lit > 450,
            "{mode:?}: no highlight to shadow ({lit}) — the fixture is off"
        );
        assert!(
            (shadowed as f64) < lit as f64 * 0.5,
            "{mode:?}: the highlight survives the shadow — peak {shadowed} against {lit} lit"
        );
    }
}

#[cfg(feature = "metal")]
#[test]
fn glass_highlights_respect_shadows_on_metal() {
    use threers::metal::{MetalError, MetalHeadlessRenderer};
    let mut r = match MetalHeadlessRenderer::builder().size(N, N).build() {
        Ok(r) => r,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    for mode in [
        TransparencyMode::Blend,
        TransparencyMode::Glass,
        TransparencyMode::Oit,
        TransparencyMode::Refract,
    ] {
        let mut peak_of = |blocked| {
            let (mut scene, camera) = scene(mode, blocked);
            peak(&r.render_to_rgba(&mut scene, &camera).expect("metal"))
        };
        let (lit, shadowed) = (peak_of(false), peak_of(true));
        eprintln!("metal {mode:?}: peak {lit} lit, {shadowed} shadowed");
        assert!(lit > 450, "metal {mode:?}: no highlight to shadow ({lit})");
        assert!(
            (shadowed as f64) < lit as f64 * 0.5,
            "metal {mode:?}: the highlight survives the shadow — {shadowed} against {lit}"
        );
    }
}
