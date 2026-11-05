//! One import is a game's whole vocabulary, and the older three still work.
#![recursion_limit = "512"]

use threers_ecs::prelude::*;

struct Crate;
impl Component for Crate {}

#[test]
fn the_three_preludes_glob_together() {
    // `World` is the game's, and the solver is still there as `Solver`.
    let mut app = App::new();
    app.plugin(PhysicsPlugin::default());
    let w: &mut World = &mut app.world;
    let id = w
        .mesh(
            BoxGeometry::new(1.0, 1.0, 1.0),
            StandardMaterial::new(Color::WHITE),
        )
        .at(Vector3::new(0.0, 2.0, 0.0))
        .with(Crate)
        .body(RigidBody::dynamic().shape(Shape::cuboid(0.5, 0.5, 0.5)))
        .id();
    let solver: &Solver = &w.resource::<Physics>().0;
    assert!(solver.body(w.get::<Body>(id).unwrap().0).is_some());
    let _ = (
        Quaternion::IDENTITY,
        Easing::Linear,
        Isometry::IDENTITY,
        MeshBuilder::new(),
    );
}

mod the_old_three_globs {
    use threers::prelude::*;
    use threers_ecs::physics::*;
    use threers_ecs::prelude::*;

    #[test]
    fn still_resolve() {
        let mut app = App::new();
        app.plugin(PhysicsPlugin::default());
        let _: &mut World = &mut app.world;
        let _ = (
            Vector3::ZERO,
            Shape::ball(1.0),
            CarSpec::default(),
            MeshBuilder::new(),
        );
    }
}

#[test]
fn a_game_needs_nothing_else() {
    let mut game = TestGame::new(|app| {
        app.plugin(PhysicsPlugin::default())
            .plugin(ChaseCamera::new());
        let w = &mut app.world;
        w.sun(Vector3::new(-1.0, -2.0, -1.0), 3.0, 50.0);
        let mut ground = MeshBuilder::new();
        ground.grid(40.0, 40.0, 4, 4, |_, _| 0.0);
        w.mesh(ground, StandardMaterial::new(Color::WHITE)).solid();
        w.mesh(
            BoxGeometry::new(1.8, 0.7, 4.0),
            StandardMaterial::new(Color::WHITE),
        )
        .at(Vector3::new(0.0, 1.0, 0.0))
        .car(CarSpec::default())
        .with((Crate, CameraTarget));
    });
    game.seconds(1.0);
    let (car, _) = game.world().single::<(&Car, With<Crate>)>().unwrap();
    assert_eq!(car.wheels_on_ground, 4);
}
