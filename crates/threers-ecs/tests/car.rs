//! Cars, the chase camera and the spawn builder, as a game uses them.
#![recursion_limit = "512"]

use threers_ecs::prelude::*;

struct Controls {
    throttle: f32,
    brake: f32,
    steer: f32,
}

fn apply(w: &mut World) {
    let c = w.resource::<Controls>();
    let (throttle, brake, steer) = (c.throttle, c.brake, c.steer);
    w.each::<&mut Car>(|car| {
        car.throttle = throttle;
        car.brake = brake;
        car.steer = steer;
    });
}

fn road() -> (App, Entity) {
    let mut app = App::new();
    app.plugin(PhysicsPlugin::default())
        .resource(Controls {
            throttle: 0.0,
            brake: 0.0,
            steer: 0.0,
        })
        .fixed_system(System::new(apply).before(threers_ecs::car::car_step));
    let w = &mut app.world;
    w.mesh(
        BoxGeometry::new(4000.0, 1.0, 4000.0),
        StandardMaterial::new(Color::WHITE),
    )
    .at(Vector3::new(0.0, -0.5, 0.0))
    .body(
        RigidBody::fixed()
            .shape(Shape::cuboid(2000.0, 0.5, 2000.0))
            .friction(1.0),
    );
    let car = w
        .mesh(
            BoxGeometry::new(1.8, 0.7, 4.0),
            StandardMaterial::new(Color::WHITE),
        )
        .at(Vector3::new(0.0, 0.9, 0.0))
        .car(CarSpec::default())
        .id();
    advance(&mut app, 1.0);
    (app, car)
}

fn advance(app: &mut App, seconds: f32) {
    for _ in 0..(seconds * 60.0) as u32 {
        app.frame(1.0 / 60.0);
    }
}

fn position(app: &App, car: Entity) -> Vector3 {
    app.world.scene.get(car).unwrap().position
}

#[test]
fn a_car_settles_on_its_wheels_and_stays_put() {
    let (mut app, car) = road();
    advance(&mut app, 2.0);
    let c = app.world.get::<Car>(car).unwrap();
    assert_eq!(c.wheels_on_ground, 4);
    assert!(c.speed.abs() < 0.1, "creeping at {}", c.speed);
    let y = position(&app, car).y;
    assert!((0.3..1.2).contains(&y), "resting at {y}");
}

#[test]
fn throttle_drives_it_forward_and_power_limits_it() {
    let (mut app, car) = road();
    app.world.resource_mut::<Controls>().throttle = 1.0;
    advance(&mut app, 4.0);
    let c = app.world.get::<Car>(car).unwrap();
    assert!(c.speed > 15.0, "{} after four seconds", c.speed);
    assert!(
        position(&app, car).z < -20.0,
        "forward is -Z: {:?}",
        position(&app, car)
    );
    advance(&mut app, 25.0);
    let top = app.world.get::<Car>(car).unwrap().speed;
    assert!((35.0..130.0).contains(&top), "top speed {top}");
    eprintln!("top speed {top}");
}

#[test]
fn the_brakes_stop_it() {
    let (mut app, car) = road();
    app.world.resource_mut::<Controls>().throttle = 1.0;
    advance(&mut app, 4.0);
    let start = app.world.get::<Car>(car).unwrap().speed;
    let controls = app.world.resource_mut::<Controls>();
    controls.throttle = 0.0;
    controls.brake = 1.0;
    advance(&mut app, 4.0);
    let end = app.world.get::<Car>(car).unwrap().speed;
    assert!(end.abs() < 0.5, "from {start} to {end}");
}

#[test]
fn back_on_the_stick_stops_and_then_reverses() {
    let (mut app, car) = road();
    app.world.resource_mut::<Controls>().throttle = -1.0;
    advance(&mut app, 3.0);
    let speed = app.world.get::<Car>(car).unwrap().speed;
    assert!(speed < -3.0, "{speed}");
}

#[test]
fn positive_steer_turns_right() {
    let (mut app, car) = road();
    let controls = app.world.resource_mut::<Controls>();
    controls.throttle = 0.5;
    controls.steer = 1.0;
    advance(&mut app, 3.0);
    let at = position(&app, car);
    assert!(at.x > 2.0, "facing -Z, right is +X: {at:?}");
}

#[test]
fn it_stays_upright_through_full_lock_at_speed() {
    let (mut app, car) = road();
    app.world.resource_mut::<Controls>().throttle = 1.0;
    advance(&mut app, 5.0);
    app.world.resource_mut::<Controls>().steer = 1.0;
    advance(&mut app, 4.0);
    let node = app.world.scene.get(car).unwrap();
    let up = Vector3::UP.apply_quaternion(node.quaternion);
    assert!(up.y > 0.7, "rolled to {up:?}");
}

#[test]
fn the_wheels_go_when_the_car_does() {
    let (mut app, car) = road();
    let nodes = app.world.scene.arena.nodes.len();
    app.world.despawn(car);
    // The body, four tyres and the spokes on each.
    assert_eq!(app.world.scene.arena.nodes.len(), nodes - 9);
    advance(&mut app, 0.1);
}

#[test]
fn a_teleported_car_arrives_at_rest() {
    let (mut app, car) = road();
    app.world.resource_mut::<Controls>().throttle = 1.0;
    advance(&mut app, 3.0);
    app.world.resource_mut::<Controls>().throttle = 0.0;
    let to = Vector3::new(50.0, 1.0, 50.0);
    assert!(app.world.teleport_body(car, to, Quaternion::IDENTITY));
    advance(&mut app, 0.5);
    let at = position(&app, car);
    assert!(
        at.distance_to(Vector3::new(50.0, at.y, 50.0)) < 0.5,
        "{at:?}"
    );
}

fn behind(app: &App) -> Vector3 {
    app.world.resource::<MainCamera>().0.position
}

fn chased(camera: ChaseCamera) -> (App, Entity) {
    let mut app = App::new();
    app.plugin(camera);
    let target = app.world.object(Object3D::group()).with(CameraTarget).id();
    app.frame(1.0 / 60.0);
    (app, target)
}

/// The angle swung through after a quarter turn, at some frame rate.
fn swing(fps: f32) -> Vector3 {
    let (mut app, target) = chased(ChaseCamera::new().distance(10.0).height(0.0).stiffness(3.0));
    let right = Quaternion::from_axis_angle(Vector3::UP, -std::f32::consts::FRAC_PI_2);
    app.world.node_mut(target).unwrap().quaternion = right;
    for _ in 0..(fps * 0.4) as u32 {
        app.frame(1.0 / fps as f64);
    }
    behind(&app)
}

#[test]
fn the_camera_sits_behind_what_it_follows() {
    let (mut app, target) = chased(ChaseCamera::new().distance(10.0).height(2.0));
    app.world.node_mut(target).unwrap().position = Vector3::new(0.0, 0.0, -100.0);
    app.frame(1.0 / 60.0);
    let at = behind(&app);
    assert!(at.x.abs() < 0.01 && (at.z + 90.0).abs() < 0.01, "{at:?}");
}

/// Smoothing the camera's position would leave it further back the faster it went.
#[test]
fn the_camera_keeps_its_distance_at_any_speed() {
    let (mut app, target) = chased(ChaseCamera::new().distance(10.0).height(0.0));
    for frame in 1..600 {
        app.world.node_mut(target).unwrap().position = Vector3::new(0.0, 0.0, -2.0 * frame as f32);
        app.frame(1.0 / 60.0);
        let gap = behind(&app).distance_to(app.world.node(target).unwrap().position);
        assert!((gap - 10.0).abs() < 0.01, "{gap} at frame {frame}");
    }
}

#[test]
fn the_camera_swings_the_same_at_any_frame_rate() {
    let (slow, fast) = (swing(30.0), swing(240.0));
    assert!(slow.distance_to(fast) < 0.05, "{slow:?} against {fast:?}");
    let settled = behind(&{
        let (mut app, target) = chased(ChaseCamera::new().distance(10.0).height(0.0));
        let right = Quaternion::from_axis_angle(Vector3::UP, -std::f32::consts::FRAC_PI_2);
        app.world.node_mut(target).unwrap().quaternion = right;
        for _ in 0..300 {
            app.frame(1.0 / 60.0);
        }
        app
    });
    assert!(
        settled.distance_to(Vector3::new(-10.0, 0.0, 0.0)) < 0.01,
        "{settled:?}"
    );
}

#[test]
fn the_camera_takes_the_short_way_round() {
    // Facing just left of south, then just right of it: a small turn, not a lap.
    let (mut app, target) = chased(ChaseCamera::new().distance(10.0).height(0.0).stiffness(1.0));
    let turned = |a: f32| Quaternion::from_axis_angle(Vector3::UP, a);
    app.world.node_mut(target).unwrap().quaternion = turned(3.1);
    app.world.resource_mut::<ChaseCamera>().snap();
    app.frame(1.0 / 60.0);
    app.world.node_mut(target).unwrap().quaternion = turned(-3.1);
    for _ in 0..10 {
        app.frame(1.0 / 60.0);
        assert!(behind(&app).z < -9.9, "{:?}", behind(&app));
    }
}

#[test]
fn the_camera_follows_a_child_through_its_parent() {
    let mut app = App::new();
    app.plugin(ChaseCamera::new().distance(10.0).height(0.0));
    let parent = app
        .world
        .object(Object3D::group())
        .at(Vector3::new(100.0, 0.0, 0.0))
        .id();
    let child = app.world.spawn_under(parent, Object3D::group());
    app.world.node_mut(child).unwrap().position = Vector3::new(0.0, 0.0, -5.0);
    app.world.insert(child, CameraTarget);
    app.frame(1.0 / 60.0);
    let at = behind(&app);
    assert!(
        at.distance_to(Vector3::new(100.0, 0.0, 5.0)) < 0.01,
        "{at:?}"
    );
}

#[test]
fn follow_beats_the_marker_and_a_dead_target_falls_back() {
    let mut app = App::new();
    let far = app
        .world
        .object(Object3D::group())
        .at(Vector3::new(500.0, 0.0, 0.0))
        .id();
    app.plugin(ChaseCamera::new().distance(10.0).height(0.0).follow(far));
    app.world.object(Object3D::group()).with(CameraTarget);
    app.frame(1.0 / 60.0);
    assert!(behind(&app).x > 400.0);
    app.world.despawn(far);
    app.world.resource_mut::<ChaseCamera>().snap();
    app.frame(1.0 / 60.0);
    assert!(behind(&app).x.abs() < 1.0, "{:?}", behind(&app));
}

#[test]
fn looking_back_turns_it_round() {
    let (mut app, _) = chased(ChaseCamera::new().distance(10.0).height(0.0));
    app.world.resource_mut::<ChaseCamera>().looking_back = true;
    for _ in 0..300 {
        app.frame(1.0 / 60.0);
    }
    assert!(behind(&app).z < -9.9, "{:?}", behind(&app));
}

#[test]
fn the_camera_comes_in_rather_than_through_a_wall() {
    let (mut app, _) = road();
    app.plugin(
        ChaseCamera::new()
            .distance(10.0)
            .height(0.0)
            .avoid_walls(0.5),
    );
    let w = &mut app.world;
    // Six units behind the target, which faces -Z from the origin.
    w.mesh(
        BoxGeometry::new(20.0, 20.0, 1.0),
        StandardMaterial::new(Color::WHITE),
    )
    .at(Vector3::new(100.0, 5.0, 6.0))
    .solid();
    let target = w
        .object(Object3D::group())
        .at(Vector3::new(100.0, 3.0, 0.0))
        .with(CameraTarget)
        .id();
    let _ = target;
    advance(&mut app, 0.1);
    let at = behind(&app);
    assert!(at.z < 5.5 && at.z > 3.0, "{at:?}");
}

#[test]
fn snap_ends_a_swing_at_once() {
    let (mut app, target) = chased(ChaseCamera::new().distance(10.0).height(0.0));
    let right = Quaternion::from_axis_angle(Vector3::UP, -std::f32::consts::FRAC_PI_2);
    app.world.node_mut(target).unwrap().quaternion = right;
    app.world.resource_mut::<ChaseCamera>().snap();
    app.frame(1.0 / 60.0);
    assert!(
        behind(&app).distance_to(Vector3::new(-10.0, 0.0, 0.0)) < 0.01,
        "{:?}",
        behind(&app)
    );
}

#[test]
fn an_optional_order_applies_when_the_system_is_there() {
    fn first(w: &mut World) {
        w.resource_mut::<Vec<u32>>().push(1);
    }
    fn second(w: &mut World) {
        w.resource_mut::<Vec<u32>>().push(2);
    }
    fn absent(_: &mut World) {}
    let mut app = App::new();
    app.world.insert_resource(Vec::<u32>::new());
    app.system(
        System::new(second)
            .after_if_added(first)
            .after_if_added(absent),
    );
    app.system(first);
    app.frame(0.016);
    assert_eq!(*app.world.resource::<Vec<u32>>(), [1, 2]);
}

#[test]
fn the_spawn_builder_leaves_a_body_where_the_node_is() {
    let (mut app, _) = road();
    let w = &mut app.world;
    let id = w
        .mesh(
            BoxGeometry::new(1.0, 1.0, 1.0),
            StandardMaterial::new(Color::WHITE),
        )
        .at(Vector3::new(30.0, 0.5, 0.0))
        .named("post")
        .body(RigidBody::fixed().shape(Shape::cuboid(0.5, 0.5, 0.5)))
        .id();
    let node = w.scene.get(id).unwrap();
    assert_eq!(node.name, "post");
    assert_eq!(node.position.x, 30.0);
    assert!(w.has::<Body>(id));
}

#[test]
fn not_in_state_runs_everywhere_else() {
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum Mode {
        Menu,
        Game,
    }
    impl GameState for Mode {}
    fn tick(w: &mut World) {
        *w.resource_mut::<u32>() += 1;
    }
    let mut app = App::new();
    app.state(Mode::Menu).resource(0u32);
    app.system(System::new(tick).not_in_state(Mode::Menu));
    app.frame(0.016);
    assert_eq!(*app.world.resource::<u32>(), 0);
    app.world.set_state(Mode::Game);
    app.frame(0.016);
    assert_eq!(*app.world.resource::<u32>(), 1);
}

/// Drawn wheels keep their axle across the car on any heading, and turn the
/// way the car rolls.
#[test]
fn drawn_wheels_roll_forward_on_any_heading() {
    let (mut app, car) = road();
    let turned = Quaternion::from_axis_angle(Vector3::UP, 2.0);
    app.world
        .teleport_body(car, Vector3::new(0.0, 0.9, 0.0), turned);
    advance(&mut app, 1.0);
    app.world.resource_mut::<Controls>().throttle = 0.6;
    advance(&mut app, 2.0);
    // The tyres are the children with spokes of their own; the third is a rear one.
    let rear = app
        .world
        .node(car)
        .unwrap()
        .children
        .iter()
        .copied()
        .filter(|&id| app.world.node(id).unwrap().children.len() == 1)
        .nth(2)
        .unwrap();
    let wheel = |app: &App| app.world.world_transform(rear).unwrap();
    let (_, chassis) = app.world.world_transform(car).unwrap();
    let axle = Vector3::new(1.0, 0.0, 0.0).apply_quaternion(chassis);
    let forward = Vector3::new(0.0, 0.0, -1.0).apply_quaternion(chassis);
    let (_, before) = wheel(&app);
    // The cylinder stands along Y, so its axis is the wheel's axle.
    let wheel_axle = Vector3::UP.apply_quaternion(before);
    assert!(
        wheel_axle.dot(axle).abs() > 0.99,
        "{wheel_axle:?} against {axle:?}"
    );
    // A point of the rim now at the top goes forward over the next frame.
    let rim = Vector3::UP.apply_quaternion(before.conjugate());
    let top_before = rim.apply_quaternion(before);
    app.frame(1.0 / 60.0);
    let (_, after) = wheel(&app);
    let moved = rim.apply_quaternion(after) - top_before;
    assert!(
        moved.dot(forward) > 0.0,
        "the top went backwards: {moved:?}, forward {forward:?}"
    );
}

/// A close look at a wheel, written for a person when asked.
#[test]
fn wheel_picture() {
    let Ok(path) = std::env::var("WHEEL_SHOT") else {
        return;
    };
    let mut game = TestGame::new(|app| {
        app.plugin(PhysicsPlugin::default());
        let w = &mut app.world;
        w.sun(Vector3::new(-1.0, -2.0, -1.0), 3.0, 30.0);
        w.scene.add_light(AmbientLight::new(Color::WHITE, 0.8));
        w.mesh(
            BoxGeometry::new(200.0, 1.0, 200.0),
            StandardMaterial::new(Color::from_hex(0x777777)),
        )
        .at(Vector3::new(0.0, -0.5, 0.0))
        .solid();
        w.mesh(
            BoxGeometry::new(1.8, 0.7, 4.0),
            StandardMaterial::new(Color::from_hex(0xd9342b)),
        )
        .at(Vector3::new(0.0, 0.9, 0.0))
        .car(CarSpec::default());
        let mut camera = PerspectiveCamera::new(40.0, 16.0 / 9.0, 0.1, 100.0);
        camera.position = Vector3::new(5.0, 1.0, 0.0);
        camera.look_at(Vector3::new(0.0, 0.4, 0.0));
        w.insert_resource(MainCamera(camera));
    });
    game.seconds(1.0);
    let rgba = game.render(960, 540);
    std::fs::write(path, threers::encode_png(960, 540, &rgba)).unwrap();
}
