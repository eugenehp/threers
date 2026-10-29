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

/// A glass ball held over a white floor, lit from the side, seen from above:
/// the floor's mean brightness over the half where its shadow falls.
fn floor_under_shadow(hr: &mut HeadlessRenderer, mode: TransparencyMode, casts: bool) -> f64 {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let mut sun = DirectionalLight::new(Color::WHITE, 3.0);
    sun.cast_shadow = true;
    sun.shadow.camera_size = 8.0;
    sun.shadow.camera_near = 0.5;
    sun.shadow.camera_far = 30.0;
    let mut light = Object3D::light(threers::lights::Light::Directional(sun));
    light.position = Vector3::new(6.0, 6.0, 0.0);
    scene.add(light);

    let glass = PhysicalMaterial::new(Color::WHITE)
        .with_roughness(0.15)
        .with_transmission(1.0, 1.5, 0.8)
        .with_transparency(mode);
    let mut ball = Object3D::mesh(Mesh::new(SphereGeometry::new(0.5, 48, 24), Material::Physical(glass)));
    ball.position = Vector3::new(0.0, 1.5, 0.0);
    ball.cast_shadow = casts;
    scene.add(ball);

    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(8.0, 8.0),
        Material::Standard(StandardMaterial::new(Color::WHITE)),
    ));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    floor.receive_shadow = true;
    scene.add(floor);

    let mut camera = PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 5.0, 0.0);
    camera.up = Vector3::new(0.0, 0.0, -1.0);
    camera.look_at(Vector3::ZERO);
    let rgba = hr.render_to_rgba(&mut scene, &camera);
    // The shadow lands about 1.5 units to -x; the ball itself sits in the
    // middle third of the frame, so read a square from the left third.
    let (x0, x1, y0, y1) = (N / 8, N / 3, 2 * N / 5, 3 * N / 5);
    let mut sum = 0.0;
    for y in y0..y1 {
        for x in x0..x1 {
            let p = &rgba[4 * (y * N + x) as usize..][..3];
            sum += (p[0] as f64 + p[1] as f64 + p[2] as f64) / 3.0;
        }
    }
    sum / ((x1 - x0) * (y1 - y0)) as f64
}

/// Glass is a shadow caster in every mode: solid glass fully, thin glass
/// (`Glass`) lightly. The modes with passes of their own (`Refract`, `Oit`,
/// `Glass`) used to be left out of the shadow maps altogether, so a refracting
/// object floated over the floor with no shadow under it.
#[test]
fn glass_casts_shadows_in_every_mode() {
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
        let open = floor_under_shadow(&mut hr, mode, false);
        let shaded = floor_under_shadow(&mut hr, mode, true);
        eprintln!("{mode:?}: floor {open:.1} without the ball's shadow, {shaded:.1} with it");
        assert!(open > 60.0, "{mode:?}: the floor is not lit ({open:.1}) — the fixture is off");
        // Thin clear glass lets nearly all the light through: its shadow is
        // the few percent it reflects (216 → 210 when this was written).
        let most = if mode == TransparencyMode::Glass { 0.99 } else { 0.7 };
        assert!(
            shaded < open * most,
            "{mode:?}: the ball casts no shadow — floor {shaded:.1} against {open:.1} unshadowed"
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
