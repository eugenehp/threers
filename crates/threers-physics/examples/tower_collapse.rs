//! Rigid bodies, rendered: a heavy ball thrown into a tower of planks laid
//! crosswise, as a Jenga tower is. Four frames side by side in
//! `out/tower_collapse.png` — the ball on its way, the impact, the tower
//! coming down, and the planks at rest. Every plank is a dynamic body bound
//! to a scene node; `sync_to_scene` moves the meshes each step and a
//! headless renderer draws them.
//!
//! ```text
//! cargo run --release -p threers-physics --example tower_collapse
//! ```

use threers::lights::Light;
use threers::prelude::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, HeadlessRenderer, Material, Mesh, Object3D,
    PerspectiveCamera, PlaneGeometry, Scene, SphereGeometry, StandardMaterial,
};
use threers_physics::prelude::*;

const W: u32 = 360;
const H: u32 = 420;

fn standard(hex: u32, roughness: f32) -> Material {
    let mut m = StandardMaterial::new(Color::from_hex(hex));
    m.roughness = roughness;
    Material::Standard(m)
}

fn main() {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x141821);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.55));
    let mut sun = DirectionalLight::new(Color::from_hex(0xfff2e0), 2.4);
    sun.cast_shadow = true;
    sun.shadow.map_size = 2048;
    sun.shadow.camera_size = 9.0;
    sun.shadow.normal_bias = 0.02;
    let mut light = Object3D::light(Light::Directional(sun));
    light.position = Vector3::new(-4.0, 9.0, 6.0);
    scene.add(light);

    let mut world = World::new();
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(40.0, 40.0),
        standard(0x3a4150, 0.9),
    ));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    floor.receive_shadow = true;
    scene.add(floor);
    world.add_body(RigidBody::fixed().shape(Shape::ground()));

    // Twelve layers of three planks, each layer turned a quarter from the
    // one below.
    let (len, tall, wide) = (1.5f32, 0.3f32, 0.5f32);
    let woods = [0xd9b380, 0xc89a62, 0xe3c08f];
    for layer in 0..12 {
        let crosswise = layer % 2 == 1;
        for k in 0..3 {
            let offset = (k as f32 - 1.0) * wide;
            let y = tall * (layer as f32 + 0.5);
            let (at, size) = if crosswise {
                (Vector3::new(offset, y, 0.0), (wide, tall, len))
            } else {
                (Vector3::new(0.0, y, offset), (len, tall, wide))
            };
            // A hair narrower than the gap, so the stack starts at rest
            // rather than pushing its neighbours apart.
            let (sx, sy, sz) = (size.0 * 0.98, size.1, size.2 * 0.98);
            let mut plank = Object3D::mesh(Mesh::new(
                BoxGeometry::new(sx, sy, sz),
                standard(woods[(layer * 3 + k) % 3], 0.7),
            ));
            plank.cast_shadow = true;
            plank.receive_shadow = true;
            let node = scene.add(plank);
            world.add_body(
                RigidBody::dynamic()
                    .shape(Shape::cuboid_from_size(sx, sy, sz))
                    .translation(at)
                    .density(0.6)
                    .friction(0.6)
                    .scene_object(node),
            );
        }
    }

    // The ball: dense, fast, and swept so it cannot pass through a plank
    // between two steps.
    let mut ball = Object3D::mesh(Mesh::new(SphereGeometry::new(0.42, 48, 24), {
        let mut m = StandardMaterial::new(Color::from_hex(0xc8453a));
        m.roughness = 0.35;
        Material::Standard(m)
    }));
    ball.cast_shadow = true;
    let node = scene.add(ball);
    world.add_body(
        RigidBody::dynamic()
            .shape(Shape::ball(0.42))
            .translation(Vector3::new(-7.0, 2.2, 0.4))
            .linear_velocity(Vector3::new(14.0, 1.6, 0.0))
            .density(9.0)
            .ccd(true)
            .scene_object(node),
    );

    let mut hr = HeadlessRenderer::builder()
        .size(W, H)
        .build()
        .expect("a GPU adapter");
    hr.set_msaa(4);
    let mut camera = PerspectiveCamera::new(42.0, W as f32 / H as f32, 0.1, 80.0);
    camera.position = Vector3::new(3.2, 3.6, 9.0);
    camera.look_at(Vector3::new(0.3, 1.6, 0.0));

    let shots = [0.15f32, 0.5, 0.9, 4.0];
    let dt = 1.0 / 120.0;
    let mut t = 0.0f32;
    let mut sheet = vec![0u8; (W * shots.len() as u32 * H * 4) as usize];
    for (slot, &shot) in shots.iter().enumerate() {
        while t < shot {
            world.step(dt);
            t += dt;
        }
        world.sync_to_scene(&mut scene.arena);
        let px = hr.render_to_rgba(&mut scene, &camera);
        let stride = W * shots.len() as u32;
        for y in 0..H {
            let row = &px[(y * W * 4) as usize..((y + 1) * W * 4) as usize];
            let at = ((y * stride + slot as u32 * W) * 4) as usize;
            sheet[at..at + row.len()].copy_from_slice(row);
        }
    }
    std::fs::create_dir_all("out").unwrap();
    std::fs::write(
        "out/tower_collapse.png",
        threers::encode_png(W * shots.len() as u32, H, &sheet),
    )
    .unwrap();
    println!(
        "wrote out/tower_collapse.png ({} bodies)",
        world.body_count()
    );
}
