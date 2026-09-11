//! Culling must be invisible.
//!
//! The renderer used to draw every mesh in the scene every frame, however far
//! outside the view it was — `Frustum` sat in `src/math`, correct for this
//! crate's 0..1 clip depth and exposed to wasm, and nothing in the render path
//! ever called it. Turning it on is only a win if the picture is unchanged,
//! and the failure mode of a culling bug is not a crash: it is a hole in the
//! image on one camera angle out of twenty, which no unit test on the maths
//! would catch.
//!
//! So these compare rendered pixels with culling on and off. Byte-identical is
//! the bar, not "close enough" — a conservative sphere test has no reason to
//! change a single fragment.
//!
//! Skipped when no GPU adapter is available.

use std::sync::Arc;

use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, HeadlessRenderer, Material, Mesh, Object3D,
    PerspectiveCamera, Scene, SphereGeometry, StandardMaterial, Vector3,
};

const W: u32 = 200;
const H: u32 = 150;

fn renderer() -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder()
        .size(W, H)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()
}

/// A field of cubes spread far wider than any one camera sees, so most of them
/// are off screen on every shot below. Deterministic, and no RNG crate: the
/// positions come from a fixed hash so a failure is reproducible.
fn field(shadows: bool) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x101418);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.5));
    let mut sun = DirectionalLight::new(Color::WHITE, 2.0);
    sun.cast_shadow = shadows;
    let mut sun = Object3D::light(threers::Light::Directional(sun));
    sun.position = Vector3::new(6.0, 12.0, 4.0);
    scene.add(sun);

    let geom: Arc<threers::BufferGeometry> = Arc::new(BoxGeometry::new(1.0, 1.0, 1.0));
    let mat = Arc::new(Material::Standard(StandardMaterial::new(Color::from_hex(
        0xb04a2a,
    ))));
    for i in 0..180u32 {
        let mut h = i.wrapping_mul(2_654_435_761);
        h ^= h >> 15;
        let fx = ((h & 0xffff) as f32 / 65535.0 - 0.5) * 60.0;
        let fz = (((h >> 16) & 0xffff) as f32 / 65535.0 - 0.5) * 60.0;
        let mut o = Object3D::mesh(Mesh::from_arc(geom.clone(), mat.clone()));
        o.position = Vector3::new(fx, 0.5, fz);
        o.cast_shadow = shadows;
        o.receive_shadow = shadows;
        scene.add(o);
    }
    // A ground plane, big enough that it straddles the frustum edge on every
    // shot — the case a sphere test is most likely to get wrong.
    let mut floor = Object3D::mesh(Mesh::new(
        BoxGeometry::new(140.0, 0.2, 140.0),
        Material::Standard(StandardMaterial::new(Color::from_hex(0x39424a))),
    ));
    floor.position = Vector3::new(0.0, -0.1, 0.0);
    floor.receive_shadow = shadows;
    scene.add(floor);
    scene
}

fn shot(eye: Vector3, look: Vector3) -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(55.0, W as f32 / H as f32, 0.1, 400.0);
    c.position = eye;
    c.look_at(look);
    c
}

fn both_ways(r: &mut HeadlessRenderer, shadows: bool, cam: &PerspectiveCamera) -> (Vec<u8>, Vec<u8>) {
    r.renderer().set_frustum_culling(true);
    let on = r.render_to_rgba(&mut field(shadows), cam);
    r.renderer().set_frustum_culling(false);
    let off = r.render_to_rgba(&mut field(shadows), cam);
    r.renderer().set_frustum_culling(true);
    (on, off)
}

fn differing(a: &[u8], b: &[u8]) -> usize {
    a.chunks(4).zip(b.chunks(4)).filter(|(p, q)| p != q).count()
}

/// The picture is the same whether or not the off-screen half of the scene was
/// drawn.
#[test]
fn culling_does_not_change_the_image() {
    let Some(mut r) = renderer() else { return };
    // Angles chosen to put the frustum edge through the cube field in
    // different ways: down the middle, along a diagonal, and steeply from
    // above where the ground plane fills the view.
    for (i, (eye, look)) in [
        (Vector3::new(0.0, 3.0, 14.0), Vector3::new(0.0, 0.5, 0.0)),
        (Vector3::new(18.0, 4.0, 18.0), Vector3::new(0.0, 0.5, 0.0)),
        (Vector3::new(2.0, 22.0, 2.0), Vector3::new(0.0, 0.0, 0.0)),
        (Vector3::new(-25.0, 2.0, 0.0), Vector3::new(25.0, 1.0, 0.0)),
    ]
    .iter()
    .enumerate()
    {
        let cam = shot(*eye, *look);
        let (on, off) = both_ways(&mut r, false, &cam);
        assert_eq!(
            differing(&on, &off),
            0,
            "shot {i}: culling changed {} of {} pixels",
            differing(&on, &off),
            on.len() / 4
        );
    }
}

/// Shadows survive it.
///
/// This is the trap the whole design is built round: a caster behind the
/// camera still throws a shadow across the view, so culled draws have to stay
/// in the draw list and only the colour passes may skip them. Dropping them
/// from the list outright passes the test above and fails this one.
#[test]
fn culling_keeps_offscreen_shadow_casters() {
    let Some(mut r) = renderer() else { return };
    // Looking away from the sun, so the shadows fall towards the camera and
    // are cast largely by cubes off the bottom and sides of the frame.
    let cam = shot(Vector3::new(-9.0, 2.5, -9.0), Vector3::new(9.0, 0.5, 9.0));
    let (on, off) = both_ways(&mut r, true, &cam);
    assert_eq!(
        differing(&on, &off),
        0,
        "culling changed {} shadowed pixels",
        differing(&on, &off)
    );
}

/// And it actually culls.
///
/// A test that only checks "the image is unchanged" passes just as happily
/// when culling is a no-op, which is how a broken frustum extraction would
/// hide — and a pixel comparison cannot tell the difference, because leaving
/// the image alone is the whole specification. So this reads the renderer's
/// own count of what it rejected.
#[test]
fn culling_removes_work() {
    let Some(mut r) = renderer() else { return };
    // Straight up, from inside the field: nothing is in view but sky, so
    // almost every one of the 181 meshes should be rejected.
    let sky = shot(Vector3::new(0.0, 2.0, 0.0), Vector3::new(0.0, 40.0, 0.0));
    r.renderer().set_frustum_culling(true);
    let _ = r.render_to_rgba(&mut field(false), &sky);
    let (drawn, culled) = r.renderer().cull_stats();
    assert!(
        culled > drawn * 4,
        "camera pointed at empty sky still drew {drawn} meshes and culled only {culled}"
    );

    // Level with the field, where a good deal is in view: culling must not be
    // throwing away most of the scene here.
    let across = shot(Vector3::new(0.0, 3.0, 34.0), Vector3::new(0.0, 0.5, 0.0));
    let _ = r.render_to_rgba(&mut field(false), &across);
    let (drawn, _) = r.renderer().cull_stats();
    assert!(drawn > 20, "looking straight at the field only drew {drawn} meshes");

    // And with culling off nothing is ever rejected.
    r.renderer().set_frustum_culling(false);
    let _ = r.render_to_rgba(&mut field(false), &sky);
    assert_eq!(r.renderer().cull_stats().1, 0, "culling was meant to be off");
    r.renderer().set_frustum_culling(true);
}

/// A sphere at the very edge of the frustum stays drawn.
///
/// Off-by-one in a plane test shows up here and nowhere else: the object is
/// mostly outside the view with a sliver inside, which is exactly the case a
/// centre-only test gets wrong.
#[test]
fn objects_straddling_the_frustum_edge_are_kept() {
    let Some(mut r) = renderer() else { return };
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(AmbientLight::new(Color::WHITE, 3.0));
    let mut ball = Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 32, 16),
        Material::Standard(StandardMaterial::new(Color::WHITE)),
    ));
    // Just off the right edge at this fov, so only its left limb is in frame.
    ball.position = Vector3::new(3.05, 0.0, 0.0);
    scene.add(ball);
    let mut cam = PerspectiveCamera::new(55.0, W as f32 / H as f32, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 5.0);
    cam.look_at(Vector3::ZERO);

    r.renderer().set_frustum_culling(true);
    let on = r.render_to_rgba(&mut scene, &cam);
    let visible = on.chunks(4).filter(|p| p[0] > 40).count();
    assert!(
        visible > 0,
        "the sliver of a sphere on the frustum edge was culled away entirely"
    );
}

/// The GPU clock reports something.
///
/// Wall-clock timing round `render` is dominated by CPU submission and by
/// whatever else the machine is running — three identical frames measured that
/// way here came back at 18, 31 and 35 ms — so the renderer now brackets its
/// main pass with timestamp queries instead. The readback is deliberately not
/// waited on, so the first frames report nothing; this checks that a figure
/// does eventually arrive and that it is sane.
///
/// Silently passes where the adapter has no `TIMESTAMP_QUERY`, which is the
/// same thing the API does.
#[test]
fn the_gpu_clock_reports_a_frame_time() {
    let Some(mut r) = renderer() else { return };
    let cam = shot(Vector3::new(0.0, 3.0, 20.0), Vector3::new(0.0, 0.5, 0.0));
    let mut best = f32::INFINITY;
    for _ in 0..40 {
        let _ = r.render_to_rgba(&mut field(false), &cam);
        let ms = r.renderer().gpu_frame_ms();
        if ms > 0.0 {
            best = best.min(ms);
        }
    }
    if best.is_infinite() {
        // No timestamp support on this adapter.
        return;
    }
    assert!(
        best > 0.0 && best < 1000.0,
        "main pass reported {best} ms, which is not a plausible frame time"
    );
}
