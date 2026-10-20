//! Glass seen through glass (`Renderer::set_refraction_layers`).
//!
//! Screen-space refraction reads a capture of what is behind the glass.
//! With one capture — three.js's way — glass behind glass is not in it, and
//! a red ball behind a clear one vanishes wherever the clear one covers it.
//! With layers, the farther glass is drawn first and captured, and the
//! nearer refracts it.
//!
//! Skipped when no GPU adapter is available.

use threers::materials::TransparencyMode;
use threers::prelude::*;

const N: u32 = 128;

fn glass(color: Color) -> Material {
    Material::Physical(
        PhysicalMaterial::new(color)
            .with_roughness(0.0)
            .with_transmission(1.0, 1.5, 0.0)
            .with_transparency(TransparencyMode::Refract),
    )
}

/// The frame: a white wall, a red glass plate on it, a clear plate in front
/// of the red one's left half.
fn frame(hr: &mut HeadlessRenderer, layers: usize, msaa: u32) -> Vec<u8> {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
    let mut wall = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(10.0, 10.0),
        Material::Basic(BasicMaterial::new(Color::WHITE)),
    ));
    wall.position = Vector3::new(0.0, 0.0, -2.0);
    scene.add(wall);
    let mut back = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(2.0, 2.0),
        glass(Color::new(0.95, 0.05, 0.05)),
    ));
    back.position = Vector3::new(0.0, 0.0, -1.0);
    scene.add(back);
    let mut front = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(1.4, 1.4),
        glass(Color::new(0.95, 0.95, 0.95)),
    ));
    front.position = Vector3::new(-0.6, 0.0, 0.0);
    scene.add(front);
    let mut camera = PerspectiveCamera::new(40.0, 1.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 5.0);
    camera.look_at(Vector3::ZERO);
    hr.renderer().set_refraction_layers(layers);
    hr.set_msaa(msaa);
    let px = hr.render_to_rgba(&mut scene, &camera);
    hr.set_msaa(1);
    px
}

fn at(px: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * N + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2]]
}

#[test]
fn glass_behind_glass_shows_through_it() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Where both plates cover (left of centre), and where only the red one
    // does (right of centre).
    let (both, red_only) = ((N / 2 - 12, N / 2), (N / 2 + 16, N / 2));
    let one = frame(&mut hr, 1, 1);
    let two = frame(&mut hr, 2, 1);
    let two_msaa = frame(&mut hr, 2, 4);
    let (o, t, tm) = (
        at(&one, both.0, both.1),
        at(&two, both.0, both.1),
        at(&two_msaa, both.0, both.1),
    );
    let red = at(&two, red_only.0, red_only.1);
    eprintln!("overlap: one layer {o:?}, two {t:?}, two with MSAA {tm:?}; red alone {red:?}");
    assert!(
        red[0] as u32 > 2 * red[1].max(1) as u32,
        "the red plate alone is not red: {red:?}"
    );
    // One capture: the red plate is missing behind the clear one.
    assert!(
        o[1] > 150 && o[2] > 150,
        "one layer should show the white wall through the clear plate: {o:?}"
    );
    // Two: the red plate shows through the clear one.
    for (name, c) in [("two layers", t), ("two layers with MSAA", tm)] {
        assert!(
            c[0] > 150 && c[1] < 80 && c[2] < 80,
            "{name}: the red plate behind the clear one reads {c:?}"
        );
    }
}

/// Order-independent transparency with refraction glass in the frame: an
/// OIT plate behind the glass is seen through it, one in front over it.
/// Both used to vanish — composited into the frame, then overwritten by the
/// refraction pass laying the capture back down.
#[test]
fn oit_surfaces_sit_behind_and_in_front_of_refraction_glass() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let oit = |c: Color| {
        let mut m = PhysicalMaterial::new(c).with_transparency(TransparencyMode::Oit);
        m.opacity = 0.9;
        m.transparent = Some(true);
        m.metalness = 0.0;
        m.roughness = 1.0;
        Material::Physical(m)
    };
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(AmbientLight::new(Color::WHITE, 3.0));
    let mut wall = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(10.0, 10.0),
        Material::Basic(BasicMaterial::new(Color::WHITE)),
    ));
    wall.position = Vector3::new(0.0, 0.0, -3.0);
    scene.add(wall);
    // Behind the glass: red, filling the left of the frame.
    let mut behind = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(1.2, 1.2),
        oit(Color::new(1.0, 0.0, 0.0)),
    ));
    behind.position = Vector3::new(-0.5, 0.0, -1.5);
    scene.add(behind);
    // The clear glass plate across the middle.
    scene.add(Object3D::mesh(Mesh::new(
        PlaneGeometry::new(2.4, 1.0),
        glass(Color::new(0.95, 0.95, 0.95)),
    )));
    // In front of the glass: blue, on the right.
    let mut front = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(0.6, 0.6),
        oit(Color::new(0.0, 0.0, 1.0)),
    ));
    front.position = Vector3::new(0.6, 0.0, 1.0);
    scene.add(front);
    let mut camera = PerspectiveCamera::new(40.0, 1.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 5.0);
    camera.look_at(Vector3::ZERO);
    let px = hr.render_to_rgba(&mut scene, &camera);
    let behind_px = at(&px, N / 2 - 14, N / 2);
    let front_px = at(&px, N / 2 + 20, N / 2);
    eprintln!("red behind the glass reads {behind_px:?}; blue in front {front_px:?}");
    assert!(
        behind_px[0] > 150 && behind_px[1] < 110 && behind_px[2] < 110,
        "the OIT plate behind the glass does not show through it: {behind_px:?}"
    );
    assert!(
        front_px[2] > 150 && front_px[0] < 110 && front_px[1] < 110,
        "the OIT plate in front of the glass is not drawn over it: {front_px:?}"
    );
}
