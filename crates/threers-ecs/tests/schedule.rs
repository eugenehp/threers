//! System order and the fixed-step loop.

use threers::core::Object3D;
use threers_ecs::prelude::*;

#[derive(Default)]
struct Log(Vec<&'static str>);

fn a(w: &mut World) {
    w.resource_mut::<Log>().0.push("a");
}
fn b(w: &mut World) {
    w.resource_mut::<Log>().0.push("b");
}
fn c(w: &mut World) {
    w.resource_mut::<Log>().0.push("c");
}

fn order(schedule: &mut Schedule) -> Vec<&'static str> {
    let mut w = World::new();
    w.insert_resource(Log::default());
    schedule.run(&mut w);
    w.remove_resource::<Log>().unwrap().0
}

#[test]
fn without_constraints_systems_run_as_added() {
    let mut s = Schedule::new();
    s.add(c).add(a).add(b);
    assert_eq!(order(&mut s), ["c", "a", "b"]);
}

#[test]
fn a_system_that_must_wait_moves_later_and_the_rest_keep_their_order() {
    let mut s = Schedule::new();
    s.add(System::new(a).after(c)).add(b).add(c);
    assert_eq!(order(&mut s), ["b", "c", "a"]);

    let mut s = Schedule::new();
    s.add(a).add(b).add(System::new(c).before(a));
    assert_eq!(order(&mut s), ["b", "c", "a"]);
}

#[test]
fn adding_a_system_later_is_sorted_in() {
    let mut s = Schedule::new();
    s.add(a);
    assert_eq!(order(&mut s), ["a"]);
    s.add(System::new(b).before(a));
    assert_eq!(order(&mut s), ["b", "a"]);
}

#[test]
#[should_panic(expected = "cycle")]
fn a_cycle_is_reported() {
    let mut s = Schedule::new();
    s.add(System::new(a).after(b)).add(System::new(b).after(a));
    order(&mut s);
}

#[test]
#[should_panic(expected = "not in this schedule")]
fn ordering_against_an_absent_system_is_reported() {
    let mut s = Schedule::new();
    s.add(System::new(a).after(b));
    order(&mut s);
}

#[test]
#[should_panic(expected = "added twice")]
fn adding_a_system_twice_is_refused() {
    Schedule::new().add(a).add(a);
}

#[test]
fn a_systems_commands_land_before_the_next_system() {
    struct Marker;
    impl Component for Marker {}

    fn write(w: &mut World) {
        w.commands().spawn(Object3D::group(), Marker);
    }
    fn read(w: &mut World) {
        assert_eq!(w.entities::<With<Marker>>().len(), 1);
        w.insert_resource(true);
    }

    let mut world = World::new();
    let mut s = Schedule::new();
    s.add(write).add(read);
    s.run(&mut world);
    assert!(*world.resource::<bool>());
}

#[derive(Default)]
struct Counts {
    fixed: u32,
    update: u32,
    fixed_deltas: Vec<f32>,
    update_deltas: Vec<f32>,
}

fn count_fixed(w: &mut World) {
    let delta = w.resource::<Time>().delta;
    let counts = w.resource_mut::<Counts>();
    counts.fixed += 1;
    counts.fixed_deltas.push(delta);
}

fn count_update(w: &mut World) {
    let delta = w.resource::<Time>().delta;
    let counts = w.resource_mut::<Counts>();
    counts.update += 1;
    counts.update_deltas.push(delta);
}

fn counting_app() -> App {
    let mut app = App::new();
    app.resource(Counts::default());
    app.fixed_system(count_fixed);
    app.system(count_update);
    app
}

fn counts(app: &App) -> (u32, u32) {
    let c = app.world.resource::<Counts>();
    (c.fixed, c.update)
}

#[test]
fn one_frame_per_step_at_the_fixed_rate() {
    let mut app = counting_app();
    for _ in 0..600 {
        app.frame(1.0 / 60.0);
    }
    assert_eq!(counts(&app), (600, 600));
}

#[test]
fn the_step_count_does_not_depend_on_the_frame_rate() {
    for (fps, frames) in [(30, 300), (120, 1200), (240, 2400)] {
        let mut app = counting_app();
        for _ in 0..frames {
            app.frame(1.0 / fps as f64);
        }
        let (fixed, update) = counts(&app);
        assert_eq!(update, frames);
        assert!((599..=600).contains(&fixed), "{fps} fps ran {fixed} steps");
    }
}

#[test]
fn delta_means_the_stage_it_is_read_in() {
    let mut app = counting_app();
    app.frame(1.0 / 30.0);
    let c = app.world.resource::<Counts>();
    assert_eq!(c.fixed_deltas, vec![1.0 / 60.0_f64 as f32; 2]);
    assert_eq!(c.update_deltas, vec![1.0 / 30.0_f64 as f32]);
}

#[test]
fn alpha_is_the_fraction_left_over() {
    let mut app = counting_app();
    app.fixed_hz(10.0);
    app.frame(0.25);
    let time = *app.world.resource::<Time>();
    assert_eq!(counts(&app).0, 2);
    assert!((time.alpha - 0.5).abs() < 1e-5, "{}", time.alpha);
    assert!((time.elapsed - 0.25).abs() < 1e-9);
    assert_eq!(time.frame, 1);
    assert_eq!(time.fixed_steps, 2);
}

#[test]
fn a_long_stall_is_capped_and_its_debt_forgotten() {
    let mut app = counting_app();
    app.max_fixed_steps(4);
    app.frame(100.0);
    assert_eq!(counts(&app).0, 4);
    assert_eq!(app.world.resource::<Time>().alpha, 0.0);
    app.frame(1.0 / 60.0);
    assert_eq!(counts(&app).0, 5);
}

#[test]
fn a_bad_timestamp_counts_as_no_time() {
    let mut app = counting_app();
    for delta in [f64::NAN, f64::INFINITY, -1.0] {
        app.frame(delta);
    }
    assert_eq!(counts(&app), (0, 3));
    app.frame(1.0 / 60.0);
    assert_eq!(counts(&app), (1, 4));
}

#[test]
fn commands_queued_between_frames_apply_before_the_first_system() {
    struct Marker;
    impl Component for Marker {}

    fn check(w: &mut World) {
        assert_eq!(w.entities::<With<Marker>>().len(), 1);
    }

    let mut app = App::new();
    app.fixed_system(check);
    app.world.commands().spawn(Object3D::group(), Marker);
    app.frame(1.0 / 60.0);
    assert_eq!(app.world.resource::<Time>().fixed_steps, 1);
}
