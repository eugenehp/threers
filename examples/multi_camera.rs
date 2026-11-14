//! Many cameras at once (`threers::views`): a grid of four, a picture in a
//! picture, a monitor in the room showing another camera, a camera looking
//! at its own screen (a corridor of screens), and three transitions caught
//! halfway (a morph, a crossfade, a wipe). Writes `out/multi_camera_*.png`.
//!
//! ```bash
//! cargo run --release --example multi_camera
//! ```

use std::f32::consts::PI;

use threers::prelude::*;
use threers::utils::png::encode_png;
use threers::views::{CameraSet, CameraTexture, Feedback, Layout, TransitionKind};

const W: u32 = 960;
const H: u32 = 540;

fn room() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x10141c);
    scene.add_light(AmbientLight::new(Color::from_hex(0x8899bb), 0.5));
    let mut sun = Object3D::light(
        DirectionalLight::new(Color::WHITE, 2.2).with_direction(Vector3::new(-0.5, -1.0, -0.3)),
    );
    sun.position = Vector3::new(3.0, 5.0, 2.0);
    scene.add(sun);
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(12.0, 12.0),
        StandardMaterial::new(Color::from_hex(0x3a3f4a))
            .with_roughness(0.8)
            .into(),
    ));
    floor.quaternion = Euler::new(-PI / 2.0, 0.0, 0.0).to_quaternion();
    scene.add(floor);
    for (i, (c, x, z)) in [
        (0xe4572e, -1.4, 0.0),
        (0x29a36a, 0.0, -0.8),
        (0x2f6fde, 1.4, 0.2),
        (0xf2c14e, 0.4, 1.3),
    ]
    .iter()
    .enumerate()
    {
        let geometry: BufferGeometry = match i {
            0 => BoxGeometry::new(0.9, 0.9, 0.9),
            1 => SphereGeometry::new(0.55, 40, 24),
            2 => TorusKnotGeometry::new(0.38, 0.13, 160, 20, 2, 3),
            _ => CylinderGeometry::new(0.35, 0.35, 0.9, 40, 1, false, 0.0, 2.0 * PI),
        };
        let mut o = Object3D::mesh(Mesh::new(
            geometry,
            StandardMaterial::new(Color::from_hex(*c))
                .with_roughness(0.4)
                .into(),
        ));
        o.position = Vector3::new(*x, 0.6, *z);
        scene.add(o);
    }
    scene
}

fn camera(fov: f32, at: Vector3, look: Vector3) -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(fov, W as f32 / H as f32, 0.05, 60.0);
    c.position = at;
    c.look_at(look);
    c
}

fn cameras() -> CameraSet {
    let mut set = CameraSet::new();
    let mid = Vector3::new(0.0, 0.5, 0.0);
    set.add("front", camera(45.0, Vector3::new(0.0, 1.6, 5.0), mid));
    set.add("side", camera(40.0, Vector3::new(5.0, 1.4, 0.5), mid));
    set.add("top", camera(50.0, Vector3::new(0.01, 7.0, 0.0), mid));
    set.add(
        "close",
        camera(
            30.0,
            Vector3::new(1.8, 1.2, 2.4),
            Vector3::new(1.2, 0.6, 0.0),
        ),
    );
    set
}

fn save(
    name: &str,
    hr: &mut HeadlessRenderer,
    scene: &mut Scene,
    set: &mut CameraSet,
    layout: &Layout,
) {
    set.render(hr, scene, layout);
    let path = format!("out/multi_camera_{name}.png");
    std::fs::write(&path, encode_png(W, H, &hr.read_rgba())).unwrap();
    println!("wrote {path}");
}

fn main() {
    std::fs::create_dir_all("out").unwrap();
    let mut hr = HeadlessRenderer::builder()
        .size(W, H)
        .build()
        .expect("a GPU adapter");
    let mut scene = room();
    let mut set = cameras();

    save(
        "grid",
        &mut hr,
        &mut scene,
        &mut set,
        &Layout::grid(&["front", "side", "top", "close"]),
    );
    save(
        "pip",
        &mut hr,
        &mut scene,
        &mut set,
        &Layout::picture_in_picture("front", "top"),
    );

    // A monitor in the room, showing the top camera.
    let mut screen = CameraTexture::new(hr.renderer(), 512, 288);
    let mut m = BasicMaterial::new(Color::WHITE);
    m.map = Some(screen.texture());
    let mut monitor = Object3D::mesh(Mesh::new(PlaneGeometry::new(1.6, 0.9), m.into()));
    monitor.position = Vector3::new(-0.4, 1.9, -2.2);
    let monitor = scene.add(monitor);
    let top = set.get("top").unwrap().clone();
    screen.render(hr.renderer(), &mut scene, &top);
    save(
        "monitor",
        &mut hr,
        &mut scene,
        &mut set,
        &Layout::One("front".into()),
    );

    // The front camera looking at a screen that shows what it sees.
    let mut feedback = Feedback::new(hr.renderer(), 512, 288, 6);
    if let Some(o) = scene.get_mut(monitor)
        && let ObjectKind::Mesh(mesh) = &mut o.kind
        && let Material::Basic(b) = std::sync::Arc::make_mut(&mut mesh.material)
    {
        b.map = Some(feedback.texture());
    }
    let mut eye = set.get("front").unwrap().clone();
    eye.aspect = 512.0 / 288.0;
    feedback.render(hr.renderer(), &mut scene, &eye);
    save(
        "feedback",
        &mut hr,
        &mut scene,
        &mut set,
        &Layout::One("front".into()),
    );

    // Transitions, halfway.
    for (name, kind) in [
        ("morph", TransitionKind::Morph),
        ("crossfade", TransitionKind::Crossfade),
        ("wipe", TransitionKind::Wipe(0.0)),
    ] {
        set.switch_to("front");
        set.transition_eased("side", kind, 2.0, threers::easing::Easing::Linear);
        set.update(1.0);
        save(name, &mut hr, &mut scene, &mut set, &Layout::Live);
    }
}
