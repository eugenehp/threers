//! Bodies on entities: who moves what, when, and what happens when one goes.

use threers::core::Object3D;
use threers::math::{Matrix4, Quaternion, Vector3};
use threers_ecs::physics::*;
use threers_ecs::prelude::*;

fn app() -> App {
    let mut app = App::new();
    app.plugin(PhysicsPlugin::default());
    app
}

fn at(w: &mut World, x: f32, y: f32) -> Entity {
    let id = w.spawn(Object3D::group());
    let node = w.scene.get_mut(id).unwrap();
    node.position = Vector3::new(x, y, 0.0);
    id
}

fn ball(w: &mut World, x: f32, y: f32) -> Entity {
    let id = at(w, x, y);
    w.attach_body(id, RigidBody::dynamic().shape(Shape::ball(0.5)));
    id
}

fn floor(w: &mut World) -> Entity {
    let id = w.spawn(Object3D::group());
    w.attach_body(id, RigidBody::fixed().shape(Shape::ground()));
    id
}

fn y(w: &World, id: Entity) -> f32 {
    w.scene.get(id).unwrap().position.y
}

fn body_y(w: &World, id: Entity) -> f32 {
    let body = w.get::<Body>(id).unwrap();
    w.resource::<Physics>()
        .body(body.0)
        .unwrap()
        .position
        .translation
        .y
}

fn body_count(w: &World) -> usize {
    w.resource::<Physics>().body_count()
}

#[test]
fn a_dynamic_node_falls_and_comes_to_rest_on_the_floor() {
    let mut app = app();
    floor(&mut app.world);
    let id = ball(&mut app.world, 0.0, 3.0);
    for _ in 0..300 {
        app.frame(1.0 / 60.0);
    }
    assert!(
        (y(&app.world, id) - 0.5).abs() < 0.02,
        "{}",
        y(&app.world, id)
    );
}

#[test]
fn the_frame_rate_does_not_change_where_a_body_lands() {
    let run = |fps: u32| {
        let mut app = app();
        floor(&mut app.world);
        let id = ball(&mut app.world, 0.0, 3.0);
        for _ in 0..fps {
            app.frame(1.0 / fps as f64);
        }
        (
            app.world.resource::<Time>().fixed_steps,
            body_y(&app.world, id),
        )
    };
    let (steps, height) = run(60);
    assert_eq!(steps, 60);
    // 120 and 240 divide the step exactly, so these are the same steps.
    assert_eq!(run(120), (steps, height));
    assert_eq!(run(240), (steps, height));
}

#[test]
fn the_solver_runs_at_the_rate_of_the_app() {
    let run = |hz: f64| {
        let mut app = app();
        app.fixed_hz(hz);
        let id = ball(&mut app.world, 0.0, 100.0);
        for _ in 0..hz as u32 {
            app.frame(1.0 / hz);
        }
        assert_eq!(app.world.resource::<Physics>().timestep, (1.0 / hz) as f32);
        100.0 - body_y(&app.world, id)
    };
    // A second of free fall is about 4.9 m at any rate; running at the solver's
    // own 60 Hz would have fallen a different distance at 240.
    for hz in [30.0, 60.0, 240.0] {
        let fallen = run(hz);
        assert!((fallen - 4.905).abs() < 0.2, "{hz} Hz fell {fallen}");
    }
}

#[test]
fn the_node_is_drawn_between_the_last_two_steps() {
    let mut app = app();
    let id = ball(&mut app.world, 0.0, 100.0);
    for _ in 0..30 {
        app.frame(1.0 / 60.0);
    }
    // Half a step: no new step runs, and the picture moves on anyway.
    app.frame(0.5 / 60.0);
    let body = *app.world.get::<Body>(id).unwrap();
    let physics = app.world.resource::<Physics>();
    let current = physics.body(body.0).unwrap().position.translation.y;
    let previous = physics
        .body(body.0)
        .unwrap()
        .previous_position()
        .translation
        .y;
    let drawn = y(&app.world, id);
    assert!(current < previous);
    assert!((drawn - (previous + current) / 2.0).abs() < 1e-4, "{drawn}");
}

#[test]
fn a_node_keeps_its_scale() {
    let mut app = app();
    let id = ball(&mut app.world, 0.0, 5.0);
    app.world.scene.get_mut(id).unwrap().scale = Vector3::new(2.0, 2.0, 2.0);
    app.frame(1.0 / 60.0);
    assert_eq!(app.world.scene.get(id).unwrap().scale.x, 2.0);
}

#[test]
fn a_body_at_the_origin_starts_at_its_node() {
    let mut app = app();
    let id = ball(&mut app.world, 3.0, 4.0);
    assert_eq!(body_y(&app.world, id), 4.0);
}

#[test]
fn a_body_given_a_position_moves_its_node() {
    let mut app = app();
    let id = at(&mut app.world, 1.0, 1.0);
    app.world.attach_body(
        id,
        RigidBody::dynamic()
            .shape(Shape::ball(0.5))
            .translation(Vector3::new(0.0, 7.0, 0.0)),
    );
    assert_eq!(y(&app.world, id), 7.0);
    assert_eq!(body_y(&app.world, id), 7.0);
}

#[test]
fn a_new_body_does_not_streak_from_the_origin() {
    let mut app = app();
    let id = ball(&mut app.world, 0.0, 50.0);
    app.frame(0.5 / 60.0);
    assert_eq!(y(&app.world, id), 50.0);
}

#[test]
fn a_body_under_a_moved_and_rotated_parent_lands_where_the_world_says() {
    let mut app = app();
    floor(&mut app.world);
    let parent = at(&mut app.world, 3.0, 2.0);
    app.world.scene.get_mut(parent).unwrap().quaternion =
        Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), 1.0);
    let child = app.world.spawn_under(parent, Object3D::group());
    app.world.scene.get_mut(child).unwrap().position = Vector3::new(1.0, 1.0, 0.0);
    app.world
        .attach_body(child, RigidBody::dynamic().shape(Shape::ball(0.5)));

    // Compare against the matrices the renderer would use.
    app.world.scene.update_world();
    let before = app
        .world
        .scene
        .get(child)
        .unwrap()
        .matrix_world
        .decompose()
        .0;
    let body = app.world.get::<Body>(child).unwrap().0;
    let start = app
        .world
        .resource::<Physics>()
        .body(body)
        .unwrap()
        .position
        .translation;
    assert!((before - start).length() < 1e-4, "{before:?} {start:?}");

    for _ in 0..300 {
        app.frame(1.0 / 60.0);
    }
    app.world.scene.update_world();
    let after = app
        .world
        .scene
        .get(child)
        .unwrap()
        .matrix_world
        .decompose()
        .0;
    assert!((after.y - 0.5).abs() < 0.02, "{after:?}");
    assert!(
        (after.x - start.x).abs() < 0.01 && (after.z - start.z).abs() < 0.01,
        "{after:?}"
    );
}

#[test]
fn a_kinematic_node_under_a_moved_parent_drives_its_body_in_world_space() {
    let mut app = app();
    let parent = at(&mut app.world, 10.0, 0.0);
    let child = app.world.spawn_under(parent, Object3D::group());
    app.world
        .attach_body(child, RigidBody::kinematic().shape(Shape::ball(0.5)));
    app.world.scene.get_mut(child).unwrap().position.x = 1.0;
    for _ in 0..3 {
        app.frame(1.0 / 60.0);
    }
    let body = app.world.get::<Body>(child).unwrap().0;
    let x = app
        .world
        .resource::<Physics>()
        .body(body)
        .unwrap()
        .position
        .translation
        .x;
    assert!((x - 11.0).abs() < 1e-3, "{x}");
}

#[test]
#[should_panic(expected = "scaled ancestor")]
fn a_body_under_a_scaled_parent_is_refused() {
    let mut app = app();
    let parent = at(&mut app.world, 1.0, 0.0);
    app.world.scene.get_mut(parent).unwrap().scale = Vector3::new(2.0, 2.0, 2.0);
    let child = app.world.spawn_under(parent, Object3D::group());
    app.world
        .attach_body(child, RigidBody::dynamic().shape(Shape::ball(0.5)));
}

#[test]
#[should_panic(expected = "scaled ancestor")]
fn a_body_under_a_transformed_scene_root_is_refused() {
    let mut app = app();
    app.world.scene.set_root_transform(Matrix4::identity());
    let id = app.world.spawn(Object3D::group());
    app.world
        .attach_body(id, RigidBody::dynamic().shape(Shape::ball(0.5)));
}

#[test]
fn a_body_under_a_group_at_rest_is_accepted() {
    let mut app = app();
    let parent = app.world.spawn(Object3D::group());
    let child = app.world.spawn_under(parent, Object3D::group());
    assert!(app
        .world
        .attach_body(child, RigidBody::dynamic().shape(Shape::ball(0.5)))
        .is_some());
}

#[test]
fn attaching_to_a_dead_entity_adds_nothing() {
    let mut app = app();
    let id = app.world.spawn(Object3D::group());
    app.world.despawn(id);
    assert!(app
        .world
        .attach_body(id, RigidBody::dynamic().shape(Shape::ball(0.5)))
        .is_none());
    assert_eq!(body_count(&app.world), 0);
}

#[test]
fn every_way_of_losing_the_component_removes_the_body() {
    let mut app = app();
    let w = &mut app.world;

    let removed = ball(w, 0.0, 1.0);
    w.remove::<Body>(removed);
    assert_eq!(body_count(w), 0);

    let despawned = ball(w, 0.0, 1.0);
    w.despawn(despawned);
    assert_eq!(body_count(w), 0);

    let parent = w.spawn(Object3D::group());
    let child = w.spawn_under(parent, Object3D::group());
    w.attach_body(child, RigidBody::dynamic().shape(Shape::ball(0.5)));
    w.despawn(parent);
    assert_eq!(body_count(w), 0);

    let replaced = ball(w, 0.0, 1.0);
    w.attach_body(replaced, RigidBody::dynamic().shape(Shape::ball(1.0)));
    assert_eq!(body_count(w), 1);
    w.despawn(replaced);

    let deleted = ball(w, 0.0, 1.0);
    w.scene.arena.delete(deleted);
    w.prune();
    assert_eq!(body_count(w), 0);
}

#[test]
fn despawning_while_the_solver_is_lent_out_still_removes_the_body() {
    let mut app = app();
    let id = ball(&mut app.world, 0.0, 1.0);
    app.world.with_resource::<Physics, _>(|w, _| {
        w.despawn(id);
    });
    assert_eq!(body_count(&app.world), 1);
    app.world.flush();
    assert_eq!(body_count(&app.world), 0);
}

#[test]
fn a_game_can_steer_a_kinematic_node_and_push_things_with_it() {
    fn advance(w: &mut World) {
        let dt = w.resource::<Time>().delta;
        w.each::<(Node, With<Pusher>)>(|(node, _)| node.position.x += 2.0 * dt);
    }

    struct Pusher;
    impl Component for Pusher {}

    let mut app = app();
    app.fixed_system(System::new(advance).before(physics_step));
    floor(&mut app.world);

    let pusher = at(&mut app.world, -3.0, 0.5);
    app.world.insert(pusher, Pusher);
    app.world.attach_body(
        pusher,
        RigidBody::kinematic().shape(Shape::cuboid(0.5, 0.5, 2.0)),
    );
    let target = ball(&mut app.world, 0.0, 0.5);

    for _ in 0..120 {
        app.frame(1.0 / 60.0);
    }

    // The sync must leave it alone, or it would snap back every frame.
    let x = app.world.scene.get(pusher).unwrap().position.x;
    assert!((x - 1.0).abs() < 1e-3, "pusher at {x}");
    let pushed = app.world.scene.get(target).unwrap().position.x;
    assert!(pushed > 1.0, "ball at {pushed}");
}

#[test]
fn a_fixed_node_is_left_alone() {
    let mut app = app();
    let id = at(&mut app.world, 1.0, 2.0);
    app.world
        .attach_body(id, RigidBody::fixed().shape(Shape::cuboid(1.0, 1.0, 1.0)));
    app.world.scene.get_mut(id).unwrap().position.x = 9.0;
    app.frame(1.0 / 60.0);
    assert_eq!(app.world.scene.get(id).unwrap().position.x, 9.0);
}

#[test]
fn a_ceiling_stops_a_jump_instead_of_pinning_the_character_to_it() {
    let mut app = app();
    floor(&mut app.world);
    let ceiling = at(&mut app.world, 0.0, 2.5);
    app.world.attach_body(
        ceiling,
        RigidBody::fixed().shape(Shape::cuboid(5.0, 0.2, 5.0)),
    );
    let id = at(&mut app.world, 0.0, 0.85);
    app.world.attach_character(
        id,
        CharacterController::new(Shape::capsule(0.4, 0.4)).snap_to_ground(0.1),
    );
    for _ in 0..10 {
        app.frame(1.0 / 60.0);
    }
    app.world.get_mut::<Character>(id).unwrap().jump(20.0);

    let mut after_contact = None;
    for frame in 0..60 {
        app.frame(1.0 / 60.0);
        let speed = app.world.get::<Character>(id).unwrap().vertical_speed;
        if speed <= 0.0 && after_contact.is_none() {
            after_contact = Some(frame);
        }
    }
    // Well under the 2 seconds gravity alone would take to stop it.
    assert!(after_contact.unwrap() < 20, "{after_contact:?}");
    assert!(app.world.get::<Character>(id).unwrap().grounded);
}

#[test]
fn a_character_is_drawn_between_steps_on_a_fast_display() {
    let mut app = app();
    floor(&mut app.world);
    let id = at(&mut app.world, 0.0, 0.85);
    app.world
        .attach_character(id, CharacterController::new(Shape::capsule(0.4, 0.4)));
    for _ in 0..30 {
        app.frame(1.0 / 60.0);
    }
    app.world.get_mut::<Character>(id).unwrap().walk = Vector3::new(6.0, 0.0, 0.0);

    // 240 Hz: four frames to a step. Every frame must move, by about the same.
    let mut xs = Vec::new();
    for _ in 0..24 {
        app.frame(1.0 / 240.0);
        xs.push(app.world.scene.get(id).unwrap().position.x);
    }
    let steps: Vec<f32> = xs.windows(2).skip(8).map(|p| p[1] - p[0]).collect();
    let (least, most) = steps
        .iter()
        .fold((f32::MAX, 0.0_f32), |(l, m), s| (l.min(*s), m.max(*s)));
    assert!(least > 0.0, "stood still between steps: {steps:?}");
    assert!(most < least * 1.5, "stuttered: {steps:?}");
}

#[test]
fn a_teleport_is_not_drawn_as_a_streak() {
    let mut app = app();
    let id = at(&mut app.world, 0.0, 0.85);
    app.world
        .attach_character(id, CharacterController::new(Shape::capsule(0.4, 0.4)));
    app.frame(1.0 / 60.0);
    app.world
        .get_mut::<Character>(id)
        .unwrap()
        .teleport(Vector3::new(50.0, 0.0, 0.0));
    app.frame(0.5 / 60.0);
    assert_eq!(app.world.scene.get(id).unwrap().position.x, 50.0);
}

#[derive(Default)]
struct Heard(Vec<Collision>);

fn listen(w: &mut World) {
    let events = w.events::<Collision>();
    w.resource_mut::<Heard>().0.extend(events);
}

#[test]
fn collisions_arrive_as_entities_once_each_way() {
    let mut app = app();
    app.resource(Heard::default());
    app.fixed_system(System::new(listen).after(physics_step));
    let ground = floor(&mut app.world);
    let id = ball(&mut app.world, 0.0, 1.0);

    for _ in 0..120 {
        app.frame(1.0 / 60.0);
    }
    let started: Vec<_> = app
        .world
        .resource::<Heard>()
        .0
        .iter()
        .filter(|c| c.started)
        .copied()
        .collect();
    assert_eq!(started.len(), 1, "{started:?}");
    assert_eq!(started[0].other(id), Some(Some(ground)));
    assert_eq!(started[0].other(ground), Some(Some(id)));
    assert!(!started[0].is_sensor);

    // Lifting it clear stops the contact.
    let body = app.world.get::<Body>(id).unwrap().0;
    app.world
        .resource_mut::<Physics>()
        .body_mut(body)
        .unwrap()
        .set_position(Isometry::from_translation(Vector3::new(0.0, 5.0, 0.0)));
    for _ in 0..2 {
        app.frame(1.0 / 60.0);
    }
    let heard = &app.world.resource::<Heard>().0;
    assert_eq!(heard.iter().filter(|c| !c.started).count(), 1, "{heard:?}");
}

#[test]
fn a_collision_with_someone_else_is_not_mine() {
    let collision = Collision {
        a: None,
        b: None,
        started: true,
        is_sensor: false,
    };
    let id = World::new().scene.root;
    assert_eq!(collision.other(id), None);
}

#[test]
fn a_collision_reaches_a_system_exactly_once_at_any_frame_rate() {
    for fps in [20.0, 60.0, 240.0] {
        let mut app = app();
        app.resource(Heard::default());
        app.system(listen);
        floor(&mut app.world);
        ball(&mut app.world, 0.0, 0.6);
        for _ in 0..(fps as u32) {
            app.frame(1.0 / fps);
        }
        let heard = &app.world.resource::<Heard>().0;
        assert_eq!(heard.iter().filter(|c| c.started).count(), 1, "{fps} fps");
    }
}

#[test]
fn a_prepared_solver_can_be_handed_in() {
    let mut solver = Solver::new();
    solver.set_gravity(Vector3::ZERO);
    let mut app = App::new();
    app.plugin(PhysicsPlugin::with_world(solver));
    let id = ball(&mut app.world, 0.0, 5.0);
    for _ in 0..60 {
        app.frame(1.0 / 60.0);
    }
    assert_eq!(y(&app.world, id), 5.0);
}

#[test]
fn f4_draws_colliders_and_takes_them_away() {
    use threers_ecs::dev::{DevState, DevTools};
    let mut app = app();
    app.plugin(DevTools);
    floor(&mut app.world);
    ball(&mut app.world, 0.0, 2.0);
    let nodes = app.world.scene.arena.nodes.len();

    app.world.resource_mut::<Input>().press(Key::F4);
    app.frame(1.0 / 60.0);
    let lines = app.world.scene.arena.nodes.len();
    assert_eq!(lines, nodes + 1);
    assert!(app.world.resource::<DevState>().colliders);

    let input = app.world.resource_mut::<Input>();
    input.release(Key::F4);
    input.press(Key::F4);
    app.frame(1.0 / 60.0);
    assert_eq!(app.world.scene.arena.nodes.len(), nodes);
}
