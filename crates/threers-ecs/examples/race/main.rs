//! A small circuit racer: three laps against three rivals, over hills and through
//! banked corners.
//!
//! W or up to accelerate, S or down to brake and reverse, A and D or left and
//! right to steer, space for the handbrake, C to look back, R to get back on the
//! road, enter to race again, escape for the menu. A gamepad works too.
//!
//! ```text
//! cargo run -p threers-ecs --release --features physics,window,gamepad,audio --example race
//! ```
//!
//! On a Mac, `--features metal` and `-- --metal` draw it through threers' own
//! Metal renderer instead of wgpu:
//!
//! ```text
//! cargo run -p threers-ecs --release --features physics,metal,gamepad,audio --example race -- --metal
//! ```
//!
//! | Module | Has |
//! |---|---|
//! | `track` | the circuit, its hills and banking, and the scenery |
//! | `rules` | laps, lap times, places, recovery, starting again |
//! | `ai` | rivals, who pass rather than ram |
//! | `cars` | bodies, engines and tyre squeal |
//! | `effects` | skid marks |
//! | `ghost` | the player's best lap, driven again |
//! | `hud` | times, start lights, the map, the wrong-way warning |
//! | `menu` | pause, sound levels, quit |

pub mod ai;
pub mod cars;
pub mod effects;
pub mod ghost;
pub mod hud;
pub mod menu;
pub mod rules;
pub mod track;

// What tests and the browser build reach for.
#[allow(unused_imports)]
pub use ai::Driver;
#[allow(unused_imports)]
pub use rules::{standings, LapDone, Phase, Player, Race, Racer, COUNTDOWN, LAPS};
#[allow(unused_imports)]
pub use track::{Track, ROAD_HALF_WIDTH};

use threers_ecs::prelude::*;

/// Everything but the window, so the game can also be played by a test.
pub fn build(app: &mut App) {
    use ai::drive_rivals;
    use rules::*;
    let track = Track::new();
    app.plugin(PhysicsPlugin::default())
        .plugin(
            ChaseCamera::new()
                .distance(7.5)
                .height(3.0)
                .look_at(7.0, 0.8)
                .fov(62.0)
                .full_speed(70.0)
                .fov_boost(16.0)
                .speed_pull(2.0)
                .avoid_walls(0.4),
        )
        .state(Phase::Countdown)
        .resource(Race {
            laps: LAPS,
            countdown: COUNTDOWN,
            elapsed: 0.0,
        })
        .resource(cars::Sounds::default())
        .resource(menu::Menu::default())
        .fixed_system(System::new(countdown).in_state(Phase::Countdown))
        .fixed_system(
            System::new(hold_at_the_line)
                .in_state(Phase::Countdown)
                .before(car_step),
        )
        .fixed_system(System::new(clock).in_state(Phase::Racing))
        .fixed_system(
            System::new(control)
                .in_state(Phase::Racing)
                .before(car_step),
        )
        .fixed_system(
            System::new(drive_rivals)
                .not_in_state(Phase::Countdown)
                .after(progress)
                .before(car_step),
        )
        .fixed_system(System::new(surface).before(car_step).after(progress))
        .fixed_system(System::new(progress).before(car_step))
        .fixed_system(
            System::new(rescue)
                .not_in_state(Phase::Countdown)
                .after(physics_step),
        )
        .fixed_system(System::new(cars::knocks).after(physics_step))
        .fixed_system(System::new(effects::lay_marks).after(physics_step))
        .fixed_system(
            System::new(ghost::record_ghost)
                .in_state(Phase::Racing)
                .after(progress),
        )
        .system(System::new(player_recover).in_state(Phase::Racing))
        .system(System::new(restart).in_state(Phase::Finished))
        .system(cars::engines)
        .system(look_back)
        .system(effects::draw_marks)
        .system(System::new(ghost::move_ghost).after(physics_sync))
        .system(menu::menu)
        .system(hud::hud)
        .system(System::new(menu::draw_menu).after(hud::hud));

    let input = app.world.resource_mut::<Input>();
    input.bind_axis("steer", Key::KeyA, Key::KeyD);
    input.bind_axis("steer", Key::ArrowLeft, Key::ArrowRight);
    input.bind_analog("steer", PadAxis::LeftX);
    input.bind_axis("drive", Key::KeyS, Key::KeyW);
    input.bind_axis("drive", Key::ArrowDown, Key::ArrowUp);
    input.bind_analog("drive", PadAxis::LeftY);
    input
        .bind("handbrake", Key::Space)
        .bind("handbrake", PadButton::South);
    input
        .bind("recover", Key::KeyR)
        .bind("recover", PadButton::North);
    input
        .bind("restart", Key::Enter)
        .bind("restart", PadButton::Start);
    input
        .bind("look_back", Key::KeyC)
        .bind("look_back", PadButton::West);
    input
        .bind("menu", Key::Escape)
        .bind("menu", PadButton::Select);

    app.plugin(
        TouchControls::new()
            .stick("steer", "drive")
            .button(TouchButton::new("handbrake", "B", 0.88, 0.75)),
    );

    let w = &mut app.world;
    w.scene.background = Color::from_hex(0x9cc7e6);
    w.sun(Vector3::new(-1.0, -2.0, -1.5), 3.0, 150.0);
    w.scene.add_light(AmbientLight::new(Color::WHITE, 0.7));

    track::build_circuit(w, &track);
    w.insert_resource(track);
    let marks = effects::SkidMarks::spawn(w);
    w.insert_resource(marks);
    let ghost = ghost::Ghost::spawn(w, CarSpec::default().half_size);
    w.insert_resource(ghost);

    for (name, colour, lane, nerve) in [
        ("Vesper", 0x2f6fd0, 1.5, 0.82),
        ("Quill", 0xe0a020, -1.5, 0.76),
        ("Bram", 0x7a3fc0, 0.0, 0.70),
    ] {
        let id = cars::spawn_car(w, name, colour, CarSpec::default());
        w.insert(id, Driver::new(lane, nerve));
    }
    let me = cars::spawn_car(w, "You", 0xd9342b, CarSpec::default());
    w.insert(me, (Player, CameraTarget));
    line_up(w);
}

fn look_back(w: &mut World) {
    let held = w.resource::<Input>().pressed("look_back");
    w.resource_mut::<ChaseCamera>().looking_back = held;
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn main() {
    let mut app = App::new();
    build(&mut app);
    app.plugin(AudioPlugin::Device);
    let mut options = WindowOptions::new("race");
    if std::env::args().any(|a| a == "--metal") {
        options = options.metal();
    }
    run(app, options);
}
