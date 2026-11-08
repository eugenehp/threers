//! The shortest game: a cube you drive round a field. Copy this to start one.
//!
//! cargo run -p threers-ecs --features window --example minimal

use threers_ecs::prelude::*;

/// Numbers to tune while it runs: edit tuning/player.json and save.
#[derive(Default)]
struct Tuning {
    speed: f32,
    turn: f32,
}
reflect!(Tuning { speed, turn });

struct Player;
impl Component for Player {}

fn drive(w: &mut World) {
    let input = w.resource::<Input>();
    let (forward, turn) = (input.axis("forward"), input.axis("turn"));
    let dt = w.resource::<Time>().delta;
    let tuning = w.resource::<Tuning>();
    let (speed, rate) = (tuning.speed.max(4.0), tuning.turn.max(1.5));
    w.each::<(Node, With<Player>)>(|(node, _)| {
        node.quaternion = node
            .quaternion
            .multiply(Quaternion::from_axis_angle(Vector3::UP, -turn * rate * dt));
        let ahead = Vector3::new(0.0, 0.0, -1.0).apply_quaternion(node.quaternion);
        node.position = node.position + ahead * (forward * speed * dt);
    });
    // A line showing which way it faces, for this frame only.
    if let Some((node, _)) = w.single::<(NodeRef, With<Player>)>() {
        let (at, rotation) = (node.position, node.quaternion);
        let ahead = Vector3::new(0.0, 0.0, -2.0).apply_quaternion(rotation);
        w.resource_mut::<Gizmos>()
            .arrow(at, at + ahead, Color::from_hex(0xffdd00));
    }
}

pub fn build(app: &mut App) {
    app.tune::<Tuning>("tuning/player.json")
        .plugin(ChaseCamera::new().distance(8.0).height(4.0))
        .fixed_system(drive);
    let input = app.world.resource_mut::<Input>();
    input
        .bind_axis("forward", Key::KeyS, Key::KeyW)
        .bind_axis("turn", Key::KeyA, Key::KeyD);

    let w = &mut app.world;
    w.scene.background = Color::from_hex(0x9cc7e6);
    w.sun(Vector3::new(-1.0, -2.0, -1.0), 3.0, 60.0);
    w.scene.add_light(AmbientLight::new(Color::WHITE, 0.6));
    let mut field = MeshBuilder::new();
    field.grid(80.0, 80.0, 40, 40, |x, z| {
        (x * 0.2).sin() * (z * 0.2).cos() * 0.3
    });
    w.mesh(field, StandardMaterial::new(Color::from_hex(0x4c8a3f)))
        .shadows(false, true);
    w.mesh(
        BoxGeometry::new(1.0, 1.0, 1.0),
        StandardMaterial::new(Color::from_hex(0xd9342b)),
    )
    .at(Vector3::new(0.0, 0.8, 0.0))
    .shadows(true, false)
    .with((Player, CameraTarget));
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let mut app = App::new();
    build(&mut app);
    run(app, WindowOptions::new("minimal"));
}
