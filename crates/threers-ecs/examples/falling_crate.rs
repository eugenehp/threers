//! A crate dropped on a slab, on the real renderer: two PNGs, before and after.
//!
//! ```text
//! cargo run -p threers-ecs --features physics --example falling_crate
//! ```

use threers_ecs::prelude::*;

fn main() {
    let mut app = App::new();
    app.plugin(PhysicsPlugin::default());
    let w = &mut app.world;

    let slab = w.scene.add_mesh(
        BoxGeometry::new(8.0, 0.5, 8.0),
        StandardMaterial::new(Color::from_hex(0x556677)).into(),
    );
    w.attach_body(
        slab,
        RigidBody::fixed().shape(Shape::cuboid(4.0, 0.25, 4.0)),
    );

    let block = w.scene.add_mesh(
        BoxGeometry::new(1.0, 1.0, 1.0),
        StandardMaterial::new(Color::from_hex(0xff8844)).into(),
    );
    w.scene.get_mut(block).unwrap().position.y = 4.0;
    w.attach_body(
        block,
        RigidBody::dynamic().shape(Shape::cuboid(0.5, 0.5, 0.5)),
    );

    let mut sun = DirectionalLight::new(Color::WHITE, 3.0);
    sun.direction = Vector3::new(-1.0, -2.0, -1.5).normalize();
    w.scene.add_light(sun);
    w.scene.add_light(AmbientLight::new(Color::WHITE, 0.4));

    let mut camera = PerspectiveCamera::new(50.0, 16.0 / 9.0, 0.1, 100.0);
    camera.position = Vector3::new(5.0, 4.0, 8.0);
    camera.look_at(Vector3::new(0.0, 1.5, 0.0));

    let mut renderer = HeadlessRenderer::builder().size(640, 360).build().unwrap();
    renderer.set_msaa(4);
    let mut shoot = |app: &mut App, path: &str| {
        let rgba = renderer.render_to_rgba(&mut app.world.scene, &camera);
        std::fs::write(path, threers::encode_png(640, 360, &rgba)).unwrap();
        println!(
            "{path}: crate at y = {:.3}",
            app.world.scene.get(block).unwrap().position.y
        );
    };

    shoot(&mut app, "falling_crate_start.png");
    for _ in 0..180 {
        app.frame(1.0 / 60.0);
    }
    shoot(&mut app, "falling_crate_end.png");
}
