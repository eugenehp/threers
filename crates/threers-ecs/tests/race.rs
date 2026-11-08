//! Plays the `race` example through the same input a window would feed it.
#![recursion_limit = "512"]

#[path = "../examples/race/main.rs"]
#[allow(dead_code)]
mod race;

use race::{Driver, LapDone, Phase, Player, Race, Racer, Track};
use threers::core::ObjectKind;
use threers_ecs::prelude::*;

fn game(laps: u32) -> App {
    let mut app = App::new();
    race::build(&mut app);
    app.world.resource_mut::<Race>().laps = laps;
    app
}

fn advance(app: &mut App, seconds: f32) {
    for _ in 0..(seconds * 60.0) as u32 {
        app.frame(1.0 / 60.0);
    }
}

fn phase(app: &App) -> Phase {
    app.world.state::<Phase>()
}

#[test]
fn nobody_moves_before_the_lights() {
    let mut app = game(1);
    app.world.resource_mut::<Input>().press(Key::ArrowUp);
    advance(&mut app, 2.5);
    assert_eq!(phase(&app), Phase::Countdown);
    let mut fastest = 0.0f32;
    app.world
        .each_ref::<&Car>(|car| fastest = fastest.max(car.speed.abs()));
    assert!(fastest < 0.5, "moving at {fastest} before the start");
}

#[test]
fn the_rivals_finish_a_race_on_their_own() {
    let mut app = game(1);
    advance(&mut app, 120.0);
    let mut report = Vec::new();
    app.world
        .each_ref::<(&Racer, &Car, With<Driver>)>(|(racer, car, _)| {
            report.push((racer.name, racer.distance, racer.finished, car.speed));
        });
    eprintln!("{report:?}");
    assert!(report.iter().all(|r| r.2.is_some()), "{report:?}");
}

/// Hold the keys a person would, looking only at what a person sees: where the
/// road goes, and how fast the car is.
fn drive_by_keys(app: &mut App) {
    let (node, car, racer, _) = app
        .world
        .single::<(NodeRef, &Car, &Racer, With<Player>)>()
        .unwrap();
    let (at, rotation, speed, index) = (node.position, node.quaternion, car.speed, racer.index);
    let (aim, bend) = {
        let track = app.world.resource::<Track>();
        let ahead = track.wrap(index as isize + (4.0 + speed * 0.2) as isize);
        let bend = (0..16)
            .map(|i| track.curvature[track.wrap(index as isize + i * 2)])
            .fold(0.0, f32::max);
        (track.points[ahead], bend)
    };
    let forward = Vector3::new(0.0, 0.0, -1.0).apply_quaternion(rotation);
    let right = Vector3::new(-forward.z, 0.0, forward.x);
    let to = aim - at;
    let bearing = to.dot(right).atan2(to.dot(forward));
    let safe = (9.0 / bend.max(1e-4)).sqrt();
    let input = app.world.resource_mut::<Input>();
    let mut set = |key: Key, on: bool| {
        if on {
            input.press(key);
        } else {
            input.release(key);
        }
    };
    set(Key::ArrowLeft, bearing < -0.03);
    set(Key::ArrowRight, bearing > 0.03);
    set(Key::ArrowUp, speed < safe);
    set(Key::ArrowDown, speed > safe + 3.0);
}

#[test]
fn a_race_can_be_played_to_the_finish_by_the_keys() {
    let mut app = game(1);
    advance(&mut app, race::COUNTDOWN + 0.2);
    assert_eq!(phase(&app), Phase::Racing);
    for _ in 0..60 * 150 {
        drive_by_keys(&mut app);
        app.frame(1.0 / 60.0);
        if phase(&app) == Phase::Finished {
            break;
        }
    }
    assert_eq!(
        phase(&app),
        Phase::Finished,
        "time {}",
        app.world.resource::<Race>().elapsed
    );
    let order = race::standings(&app.world);
    eprintln!(
        "finished in {} of {}",
        app.world.resource::<Race>().elapsed,
        order.len()
    );

    // The lap of honour is the car's own, and enter lines everyone up again.
    advance(&mut app, 2.0);
    app.world.resource_mut::<Input>().press(Key::Enter);
    advance(&mut app, 0.1);
    app.world.resource_mut::<Input>().release(Key::Enter);
    assert_eq!(phase(&app), Phase::Countdown);
    app.world
        .each_ref::<&Racer>(|racer| assert!(racer.distance < 0.0 && racer.finished.is_none()));
    assert_eq!(app.world.count::<(With<Player>, With<Driver>)>(), 0);
}

#[test]
fn turning_round_on_the_line_earns_nothing() {
    let mut app = game(1);
    advance(&mut app, race::COUNTDOWN + 0.2);
    let start = player_distance(&mut app);
    // Reverse from the grid, back along the road.
    let input = app.world.resource_mut::<Input>();
    input.press(Key::ArrowDown);
    advance(&mut app, 3.0);
    let after = player_distance(&mut app);
    assert!(after < start, "{start} to {after}");
}

/// However long the barriers, the scene stays a handful of draws.
#[test]
fn the_scenery_is_a_few_meshes() {
    let app = game(1);
    // Not the cars: each is a body, wheels and spokes, which is theirs to be.
    let in_a_car = |mut id: Option<Entity>| {
        while let Some(at) = id {
            if app.world.has::<threers_ecs::car::Car>(at) {
                return true;
            }
            id = app.world.node(at).and_then(|n| n.parent);
        }
        false
    };
    let count = app
        .world
        .iter::<(Entity, NodeRef)>()
        .filter(|(id, node)| matches!(node.kind, ObjectKind::Mesh(_)) && !in_a_car(Some(*id)))
        .count();
    assert!(count < 20, "{count} scenery meshes");
}

/// Merged into one compound body, the barriers still stop a car.
#[test]
fn a_car_cannot_leave_through_the_barrier() {
    let mut app = game(1);
    advance(&mut app, race::COUNTDOWN + 0.2);
    let input = app.world.resource_mut::<Input>();
    input.press(Key::ArrowUp);
    input.press(Key::ArrowLeft);
    let mut furthest = 0.0f32;
    for _ in 0..60 * 8 {
        app.frame(1.0 / 60.0);
        let (node, racer, _) = app
            .world
            .single::<(NodeRef, &Racer, With<Player>)>()
            .unwrap();
        let off = app
            .world
            .resource::<Track>()
            .offset(node.position, racer.index)
            .abs();
        furthest = furthest.max(off);
    }
    assert!(
        furthest < race::ROAD_HALF_WIDTH + 7.0,
        "got {furthest} from the centre"
    );
    assert!(
        furthest > race::ROAD_HALF_WIDTH + 2.0,
        "never reached the wall: {furthest}"
    );
}

fn player_distance(app: &mut App) -> f32 {
    app.world
        .single::<(&Racer, With<Player>)>()
        .unwrap()
        .0
        .distance
}

/// Writes pictures of the race for a person to look at, when asked.
#[test]
fn pictures() {
    let Ok(dir) = std::env::var("RACE_SHOTS") else {
        return;
    };
    let (width, height) = (960, 540);
    let mut game = threers_ecs::testing::TestGame::new(|app| {
        race::build(app);
        app.world
            .insert_resource(threers_ecs::hud::Hud::new(960, 540));
    });
    game.seconds(1.2);
    let shoot = |game: &mut threers_ecs::testing::TestGame, name: &str| {
        let rgba = game.render(width, height);
        std::fs::write(
            format!("{dir}/{name}.png"),
            threers::encode_png(width, height, &rgba),
        )
        .unwrap();
    };
    shoot(&mut game, "grid");
    game.seconds(2.0);
    for (name, seconds) in [("first-corner", 9.0), ("mid-lap", 26.0)] {
        for _ in 0..(seconds * 60.0) as u32 {
            drive_by_keys(&mut game.app);
            game.app.frame(1.0 / 60.0);
        }
        shoot(&mut game, name);
    }
}

/// How long a frame takes to draw, for comparing scenery layouts. Run by hand.
#[test]
fn frame_time() {
    if std::env::var_os("RACE_TIMING").is_none() {
        return;
    }
    let mut game = threers_ecs::testing::TestGame::new(race::build);
    game.seconds(4.0);
    let mut renderer = threers::HeadlessRenderer::builder()
        .size(1280, 720)
        .build()
        .unwrap();
    let mut times = Vec::new();
    for _ in 0..60 {
        let start = std::time::Instant::now();
        game.app.world.with_resource::<MainCamera, _>(|w, camera| {
            renderer.render_to_rgba(&mut w.scene, &camera.0)
        });
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    let meshes = game
        .app
        .world
        .scene
        .arena
        .nodes
        .values()
        .filter(|n| matches!(n.kind, ObjectKind::Mesh(_)))
        .count();
    eprintln!(
        "{meshes} meshes, frame ms p10 {:.2} p50 {:.2} p90 {:.2}",
        times[6], times[30], times[54]
    );
}

/// A race played once by the keys, recorded, and played again from the
/// recording: the same car in the same place, to the last bit.
#[test]
fn a_recorded_race_plays_back_exactly() {
    let mut app = game(1);
    app.start_recording();
    advance(&mut app, race::COUNTDOWN + 0.2);
    for _ in 0..60 * 12 {
        drive_by_keys(&mut app);
        app.frame(1.0 / 60.0);
    }
    let take = Recording::from_text(&app.stop_recording().unwrap().to_text()).unwrap();
    let place = |app: &App| {
        let (node, _) = app.world.single::<(NodeRef, With<Player>)>().unwrap();
        (node.position, node.quaternion)
    };
    let original = place(&app);

    let mut again = game(1);
    take.play(&mut again);
    assert_eq!(place(&again), original);
}

fn player(app: &App) -> Entity {
    app.world.single::<(Entity, With<Player>)>().unwrap().0
}

fn race_to_the_lights(laps: u32) -> App {
    let mut app = game(laps);
    advance(&mut app, race::COUNTDOWN + 0.2);
    assert_eq!(phase(&app), Phase::Racing);
    app
}

/// The road climbs and falls, and leans in the bends.
#[test]
fn the_circuit_has_hills_and_banked_corners() {
    let app = game(1);
    let track = app.world.resource::<Track>();
    let heights: Vec<f32> = track.points.iter().map(|p| p.y).collect();
    let (low, high) = heights
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &h| (a.min(h), b.max(h)));
    assert!(high - low > 10.0, "{low} to {high}");
    let steepest = track
        .points
        .windows(2)
        .map(|p| (p[1].y - p[0].y).abs() / track.spacing())
        .fold(0.0, f32::max);
    assert!(steepest < 0.12, "a {:.0}% climb", steepest * 100.0);
    let leaning = track.bank.iter().filter(|b| b.abs() > 0.03).count();
    assert!(
        leaning > track.points.len() / 10,
        "{leaning} banked samples"
    );
    // The ground stays under the road all the way round.
    for i in (0..track.points.len()).step_by(7) {
        for offset in [-7.0, 0.0, 7.0] {
            let road = track.surface(i, offset);
            assert!(
                track.ground(road.x, road.z) < road.y,
                "ground over the road at {i}, {offset}"
            );
        }
    }
}

/// A car stopped in a rival's lane is driven round, not into.
#[test]
fn rivals_go_round_a_stopped_car() {
    let mut app = race_to_the_lights(1);
    let me = player(&app);
    let (at, facing, index) = {
        let track = app.world.resource::<Track>();
        let index = track.wrap(30);
        (
            track.surface(index, 0.0) + Vector3::new(0.0, 0.8, 0.0),
            track.facing(index),
            index,
        )
    };
    app.world.teleport_body(me, at, facing);
    app.world.get_mut::<Racer>(me).unwrap().index = index;
    app.world.resource_mut::<Input>().press(Key::Space);
    let mut hits = 0;
    for _ in 0..60 * 20 {
        app.frame(1.0 / 60.0);
        hits += app
            .world
            .events::<threers_ecs::physics::Collision>()
            .iter()
            .filter(|c| c.started && (c.a == Some(me) || c.b == Some(me)))
            .count();
    }
    let mine = app.world.get::<Racer>(me).unwrap().distance;
    let passed = app
        .world
        .iter::<(&Racer, With<Driver>)>()
        .filter(|(r, _)| r.distance > mine + 20.0)
        .count();
    assert_eq!(passed, 3, "every rival got by");
    assert_eq!(hits, 0, "nobody touched the stopped car");
}

#[test]
fn a_finished_lap_is_timed_and_kept_as_the_best() {
    let mut app = race_to_the_lights(2);
    let me = player(&app);
    let mut laps = Vec::new();
    for _ in 0..60 * 200 {
        drive_by_keys(&mut app);
        app.frame(1.0 / 60.0);
        laps.extend(
            app.world
                .events::<LapDone>()
                .into_iter()
                .filter(|l| l.racer == me),
        );
        if phase(&app) == Phase::Finished {
            break;
        }
    }
    assert_eq!(laps.len(), 2, "{laps:?}");
    assert!(
        laps[0].best && laps.iter().all(|l| (40.0..120.0).contains(&l.time)),
        "{laps:?}"
    );
    let racer = app.world.get::<Racer>(me).unwrap();
    let quickest = laps.iter().map(|l| l.time).fold(f32::MAX, f32::min);
    assert_eq!(racer.best_lap, Some(quickest));
    assert_eq!(racer.last_lap, Some(laps[1].time));

    // The second lap was raced against the first, by a ghost.
    let ghost = app.world.resource::<race::ghost::Ghost>();
    assert!(ghost.has_lap());
    assert!(ghost.pose(10.0).is_some());
}

#[test]
fn the_ghost_shows_only_once_there_is_a_lap_to_race() {
    let mut app = race_to_the_lights(2);
    let node = app.world.resource::<race::ghost::Ghost>().node;
    let mut seen_first_lap = false;
    let mut seen_second_lap = false;
    for _ in 0..60 * 200 {
        drive_by_keys(&mut app);
        app.frame(1.0 / 60.0);
        let lap = app
            .world
            .single::<(&Racer, With<Player>)>()
            .unwrap()
            .0
            .laps_started;
        let visible = app.world.node(node).unwrap().visible;
        match lap {
            1 => seen_first_lap |= visible,
            2 if visible => {
                seen_second_lap = true;
                // It runs the same road: near the centre line.
                let at = app.world.node(node).unwrap().position;
                let track = app.world.resource::<Track>();
                let i = track.nearest_anywhere(at);
                assert!(track.offset(at, i).abs() < race::ROAD_HALF_WIDTH + 3.0);
            }
            _ => {}
        }
        if phase(&app) == Phase::Finished {
            break;
        }
    }
    assert!(!seen_first_lap, "a ghost with no lap to drive");
    assert!(seen_second_lap, "no ghost on the second lap");
}

#[test]
fn driving_backwards_round_the_track_warns() {
    let mut app = race_to_the_lights(1);
    let me = player(&app);
    let (at, backwards) = {
        let track = app.world.resource::<Track>();
        let i = track.wrap(20);
        let turned = track.facing(i).multiply(Quaternion::from_axis_angle(
            Vector3::UP,
            std::f32::consts::PI,
        ));
        (track.surface(i, 0.0) + Vector3::new(0.0, 0.8, 0.0), turned)
    };
    app.world.teleport_body(me, at, backwards);
    app.world.resource_mut::<Input>().press(Key::ArrowUp);
    // Into the oncoming field, so it may not last: was it raised at all?
    let mut warned = false;
    for _ in 0..60 * 3 {
        app.frame(1.0 / 60.0);
        warned |= app.world.get::<Racer>(me).unwrap().wrong_way();
    }
    assert!(warned);
    // And it is not raised for the right way round.
    let mut fine = race_to_the_lights(1);
    fine.world.resource_mut::<Input>().press(Key::ArrowUp);
    for _ in 0..60 * 3 {
        fine.frame(1.0 / 60.0);
        let me = player(&fine);
        assert!(!fine.world.get::<Racer>(me).unwrap().wrong_way());
    }
}

#[test]
fn a_handbrake_turn_leaves_marks_and_squeals() {
    let mut app = race_to_the_lights(1);
    let input = app.world.resource_mut::<Input>();
    input.press(Key::ArrowUp);
    advance(&mut app, 5.0);
    let input = app.world.resource_mut::<Input>();
    input.press(Key::Space);
    input.press(Key::ArrowLeft);
    let mut loudest = 0.0f32;
    for _ in 0..60 {
        app.frame(1.0 / 60.0);
        let me = player(&app);
        let tyres = &app.world.get::<Speakers>(me).unwrap().0[race::cars::TYRES];
        loudest = loudest.max(tyres.source.audio.volume);
    }
    assert!(app.world.resource::<race::effects::SkidMarks>().len() > 20);
    assert!(loudest > 0.1, "squeal at {loudest}");
}

#[test]
fn escape_pauses_and_the_menu_resumes_changes_and_restarts() {
    let mut app = race_to_the_lights(1);
    let tap = |app: &mut App, key: Key| {
        app.world.resource_mut::<Input>().press(key);
        app.frame(1.0 / 60.0);
        app.world.resource_mut::<Input>().release(key);
        app.frame(1.0 / 60.0);
    };
    tap(&mut app, Key::Escape);
    let clock = app.world.resource::<Race>().elapsed;
    advance(&mut app, 2.0);
    assert!(app.world.resource::<race::menu::Menu>().open);
    assert_eq!(
        app.world.resource::<Race>().elapsed,
        clock,
        "the race clock stopped"
    );

    tap(&mut app, Key::ArrowDown);
    tap(&mut app, Key::ArrowDown);
    tap(&mut app, Key::ArrowDown);
    assert_eq!(
        app.world.resource::<race::menu::Menu>().selected(),
        race::menu::Item::Effects
    );
    tap(&mut app, Key::ArrowLeft);
    assert!((app.world.resource::<race::menu::Menu>().effects - 0.9).abs() < 1e-4);

    tap(&mut app, Key::ArrowUp);
    tap(&mut app, Key::ArrowUp);
    tap(&mut app, Key::ArrowUp);
    tap(&mut app, Key::Enter);
    assert!(!app.world.resource::<race::menu::Menu>().open);
    advance(&mut app, 0.5);
    assert!(
        app.world.resource::<Race>().elapsed > clock,
        "running again"
    );

    tap(&mut app, Key::Escape);
    tap(&mut app, Key::ArrowDown);
    tap(&mut app, Key::Enter);
    assert_eq!(phase(&app), Phase::Countdown);
    assert!(!app.world.resource::<Time>().paused);
}

/// What the HUD costs a frame at a retina window's size, and how much of it the
/// host must upload. Run by hand.
#[test]
fn hud_cost() {
    if std::env::var_os("RACE_TIMING").is_none() {
        return;
    }
    let (width, height) = (2880, 1620);
    let mut game = TestGame::new(|app| {
        race::build(app);
        app.world
            .insert_resource(threers_ecs::hud::Hud::new(2880, 1620));
    });
    game.seconds(4.0);
    let mut times = Vec::new();
    let mut rows = 0;
    for _ in 0..60 {
        let start = std::time::Instant::now();
        game.app.frame(1.0 / 60.0);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        let hud = game.app.world.resource::<threers_ecs::hud::Hud>();
        rows = hud.0.dirty_bands().iter().map(|(a, b)| b - a).sum::<u32>();
    }
    times.sort_by(f64::total_cmp);
    eprintln!(
        "frame with HUD {width}x{height}: p50 {:.2} ms, dirty rows {rows} of {height}",
        times[30]
    );
    let mut game = TestGame::new(race::build);
    game.seconds(4.0);
    let mut times = Vec::new();
    for _ in 0..60 {
        let start = std::time::Instant::now();
        game.app.frame(1.0 / 60.0);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    eprintln!("frame without HUD: p50 {:.2} ms", times[30]);
}

/// F1 goes back five seconds, to exactly where the car was then, and the race
/// carries on from there.
#[test]
fn rewinding_puts_the_race_back_where_it_was() {
    use threers_ecs::replay::{Rewind, RewindState};
    let mut app = App::new();
    race::build(&mut app);
    app.world.resource_mut::<Race>().laps = 1;
    app.plugin(Rewind::new(race::build).seconds(5.0));
    let place = |app: &App| {
        app.world
            .single::<(NodeRef, With<Player>)>()
            .unwrap()
            .0
            .position
    };
    let mut history = Vec::new();
    advance(&mut app, race::COUNTDOWN + 0.2);
    for _ in 0..60 * 12 {
        drive_by_keys(&mut app);
        app.frame(1.0 / 60.0);
        history.push((app.world.resource::<Time>().elapsed, place(&app)));
    }
    let now = app.world.resource::<Time>().elapsed;
    app.world.rewind();
    app.frame(1.0 / 60.0);
    assert_eq!(app.world.resource::<RewindState>().rewinds, 1);
    let back = app.world.resource::<Time>().elapsed;
    assert!((now - back - 5.0).abs() < 0.1, "from {now} to {back}");
    let then = history
        .iter()
        .find(|(t, _)| (*t - back).abs() < 1e-6)
        .expect("a frame at that time")
        .1;
    assert_eq!(place(&app), then, "exactly where it was");
    // Time goes on from there, not from where it was.
    app.frame(1.0 / 60.0);
    let next = app.world.resource::<Time>().elapsed;
    assert!(
        (next - back - 1.0 / 60.0).abs() < 1e-6,
        "{back} then {next}"
    );

    // Still a race: the car goes on, and a second rewind works from here.
    for _ in 0..60 * 3 {
        drive_by_keys(&mut app);
        app.frame(1.0 / 60.0);
    }
    assert!(place(&app).distance_to(then) > 20.0);
    let before = app.world.resource::<Time>().elapsed;
    app.world.rewind();
    app.frame(1.0 / 60.0);
    let again = app.world.resource::<Time>().elapsed;
    assert!(
        (before - again - 5.0).abs() < 0.1,
        "from {before} to {again}"
    );
    assert_eq!(app.world.resource::<RewindState>().rewinds, 2);
}

/// What a rewind costs against the session it replays: printed, not checked.
/// `cargo test --release ... --test race rewind_cost -- --ignored --nocapture`
#[test]
#[ignore = "a measurement; run it in release"]
fn rewind_cost() {
    use threers_ecs::replay::Rewind;
    fn cpu() -> f64 {
        let mut t = Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // CLOCK_THREAD_CPUTIME_ID on macOS and Linux is 16 / 3.
        let id = if cfg!(target_os = "macos") { 16 } else { 3 };
        unsafe { clock_gettime(id, &mut t) };
        t.tv_sec as f64 + t.tv_nsec as f64 * 1e-9
    }
    #[repr(C)]
    struct Timespec {
        tv_sec: i64,
        tv_nsec: i64,
    }
    extern "C" {
        fn clock_gettime(id: i32, t: *mut Timespec) -> i32;
    }
    let mut app = App::new();
    race::build(&mut app);
    app.plugin(Rewind::new(race::build).seconds(5.0));
    advance(&mut app, race::COUNTDOWN + 0.2);
    let seconds = 60;
    let start = cpu();
    for _ in 0..60 * seconds {
        drive_by_keys(&mut app);
        app.frame(1.0 / 60.0);
    }
    let live = cpu() - start;
    app.world.rewind();
    let start = cpu();
    app.frame(1.0 / 60.0);
    let rewind = cpu() - start;
    eprintln!(
        "{seconds} s of racing: {live:.2} s CPU live, rewind {rewind:.2} s ({:.0}x real time)",
        seconds as f64 / rewind
    );
}

/// The race drawn through threers' own Metal renderer looks like the race
/// drawn through wgpu: the same picture region by region, and not a blank one.
/// What `--metal` in the example window draws with.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn the_race_on_metal_looks_like_the_race_on_wgpu() {
    use threers_ecs::testing::TestGame;
    let mut game = TestGame::new(race::build);
    game.seconds(race::COUNTDOWN as f64 + 1.0);
    let (width, height) = (320, 180);
    std::env::remove_var("THREERS_BACKEND");
    let wgpu = game.render(width, height);
    std::env::set_var("THREERS_BACKEND", "metal");
    let metal = game.render(width, height);
    std::env::remove_var("THREERS_BACKEND");
    if let Some(dir) = std::env::var_os("THREERS_SAVE_FRAMES") {
        let dir = std::path::Path::new(&dir);
        std::fs::write(
            dir.join("race-wgpu.png"),
            threers::encode_png(width, height, &wgpu),
        )
        .unwrap();
        std::fs::write(
            dir.join("race-metal.png"),
            threers::encode_png(width, height, &metal),
        )
        .unwrap();
    }
    // Mean colour of each cell of an 8 by 6 grid.
    let cells = |rgba: &[u8]| -> Vec<[f64; 3]> {
        let (cw, ch) = (width as usize / 8, height as usize / 6);
        let mut out = vec![[0.0; 3]; 48];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let cell = &mut out[(y / ch).min(5) * 8 + (x / cw).min(7)];
                for c in 0..3 {
                    cell[c] += rgba[(y * width as usize + x) * 4 + c] as f64 / (cw * ch) as f64;
                }
            }
        }
        out
    };
    let (a, b) = (cells(&wgpu), cells(&metal));
    let spread = |v: &[[f64; 3]]| {
        let lum: Vec<f64> = v.iter().map(|c| c[0] + c[1] + c[2]).collect();
        lum.iter().cloned().fold(f64::MIN, f64::max) - lum.iter().cloned().fold(f64::MAX, f64::min)
    };
    assert!(
        spread(&b) > 60.0,
        "the Metal frame is close to flat: {}",
        spread(&b)
    );
    let worst = a
        .iter()
        .zip(&b)
        .map(|(p, q)| (0..3).map(|c| (p[c] - q[c]).abs()).fold(0.0, f64::max))
        .fold(0.0, f64::max);
    eprintln!("largest cell difference between wgpu and Metal: {worst:.1} of 255");
    assert!(
        worst < 24.0,
        "Metal and wgpu differ by {worst:.1} in some region"
    );
}
