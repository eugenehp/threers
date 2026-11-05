//! Parallel systems: which run side by side, which don't, and that the world
//! comes out the same either way. Run under Miri as well: a batch hands its
//! systems' borrows to other threads.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use threers_ecs::prelude::*;

struct Velocity(f32);
impl Component for Velocity {}
struct Spin(f32);
impl Component for Spin {}

#[derive(Default)]
struct Score(u32);

/// Counts systems that have started; each waits a while for the other.
#[derive(Default, Clone)]
struct Gate {
    started: Arc<AtomicUsize>,
    /// How many systems saw both inside at once.
    met: Arc<AtomicUsize>,
    log: Arc<Mutex<Vec<&'static str>>>,
}

impl Gate {
    fn enter(&self, name: &'static str) {
        self.log.lock().unwrap().push(name);
        self.started.fetch_add(1, Ordering::SeqCst);
        // Long enough for a pool thread to pick up the other job, short enough
        // that the sequential cases stay quick. Miri runs slowly, so longer.
        let wait = if cfg!(miri) {
            Duration::from_secs(20)
        } else {
            Duration::from_millis(500)
        };
        let until = Instant::now() + wait;
        while Instant::now() < until {
            if self.started.load(Ordering::SeqCst) >= 2 {
                self.met.fetch_add(1, Ordering::SeqCst);
                return;
            }
            std::thread::yield_now();
        }
    }
}

fn moving(mut q: Query<&mut Velocity>, gate: Res<Gate>) {
    gate.enter("moving");
    q.each(|v| v.0 += 1.0);
}

fn spinning(mut q: Query<&mut Spin>, gate: Res<Gate>) {
    gate.enter("spinning");
    q.each(|s| s.0 += 2.0);
}

fn scoring(mut score: ResMut<Score>, gate: Res<Gate>) {
    gate.enter("scoring");
    score.0 += 1;
}

fn scoring_too(mut score: ResMut<Score>, gate: Res<Gate>) {
    gate.enter("scoring_too");
    score.0 += 10;
}

fn world() -> (App, Gate) {
    let mut app = App::new();
    let world = &mut app.world;
    let gate = Gate::default();
    world.insert_resource(gate.clone());
    world.insert_resource(Score::default());
    for i in 0..4 {
        let e = world.spawn_data();
        world.insert(e, Velocity(i as f32));
        let e = world.spawn_data();
        world.insert(e, Spin(i as f32));
    }
    (app, gate)
}

fn sums(world: &World) -> (f32, f32) {
    (
        world.iter::<&Velocity>().map(|v| v.0).sum(),
        world.iter::<&Spin>().map(|s| s.0).sum(),
    )
}

#[test]
fn disjoint_systems_run_at_once_and_both_apply() {
    let (mut app, gate) = world();
    app.system(System::parallel(moving))
        .system(System::parallel(spinning));
    app.frame(1.0 / 60.0);
    assert_eq!(sums(&app.world), (6.0 + 4.0, 6.0 + 8.0));
    let met = gate.met.load(Ordering::SeqCst);
    if cfg!(miri) {
        // The pool may not start a second thread before the first job is done
        // under the interpreter; what Miri checks here is the sharing.
        assert!(met >= 1);
    } else if cfg!(feature = "parallel") {
        assert_eq!(met, 2, "both were inside at once");
    } else {
        // One after the other: only the second sees the first's count.
        assert_eq!(met, 1);
    }
}

#[test]
fn systems_changing_the_same_thing_take_turns() {
    let (mut app, gate) = world();
    app.system(System::parallel(scoring))
        .system(System::parallel(scoring_too));
    app.frame(1.0 / 60.0);
    assert_eq!(app.world.resource::<Score>().0, 11);
    assert_eq!(gate.met.load(Ordering::SeqCst), 1, "never inside together");
    assert_eq!(*gate.log.lock().unwrap(), ["scoring", "scoring_too"]);
}

#[test]
fn an_order_between_them_keeps_them_apart() {
    let (mut app, gate) = world();
    app.system(System::parallel(spinning).after(moving))
        .system(System::parallel(moving));
    app.frame(1.0 / 60.0);
    assert_eq!(gate.met.load(Ordering::SeqCst), 1);
    assert_eq!(*gate.log.lock().unwrap(), ["moving", "spinning"]);
}

#[test]
fn a_plain_system_between_them_splits_the_batch() {
    let (mut app, gate) = world();
    let seen = Arc::new(AtomicUsize::new(0));
    let in_between = seen.clone();
    app.system(System::parallel(moving))
        .system(move |w: &mut World| {
            in_between.store(
                w.resource::<Gate>().started.load(Ordering::SeqCst),
                Ordering::SeqCst,
            );
        })
        .system(System::parallel(spinning));
    app.frame(1.0 / 60.0);
    assert_eq!(
        seen.load(Ordering::SeqCst),
        1,
        "the plain system ran after the first and before the second"
    );
    assert_eq!(gate.met.load(Ordering::SeqCst), 1);
}

#[test]
fn a_false_condition_leaves_the_rest_of_the_batch_running() {
    let (mut app, gate) = world();
    app.system(System::parallel(moving).run_if(|_: &World| false))
        .system(System::parallel(spinning));
    app.frame(1.0 / 60.0);
    assert_eq!(sums(&app.world), (6.0, 6.0 + 8.0));
    assert_eq!(*gate.log.lock().unwrap(), ["spinning"]);
}

fn exploding(_: Query<&Spin>) {
    panic!("bang");
}

#[test]
fn a_panic_in_a_batch_switches_off_that_system_only() {
    let (mut app, gate) = world();
    app.catch_panics(true);
    app.system(System::parallel(moving))
        .system(System::parallel(exploding));
    app.frame(1.0 / 60.0);
    app.frame(1.0 / 60.0);
    let failures = &app.world.resource::<threers_ecs::profile::Failures>().0;
    assert_eq!(failures.len(), 1, "switched off after the first");
    assert!(
        failures[0].system.contains("exploding"),
        "{}",
        failures[0].system
    );
    assert_eq!(failures[0].message, "bang");
    assert_eq!(gate.log.lock().unwrap().len(), 2, "moving ran both times");
}

#[test]
fn the_same_world_with_or_without_threads() {
    // A batch writes nothing until all of it is done, and commands and events
    // land after, so the outcome is that of running in order.
    let (mut together, _) = world();
    let (mut alone, _) = world();
    together
        .system(System::parallel(moving))
        .system(System::parallel(spinning));
    alone.system(moving).system(spinning);
    for _ in 0..3 {
        together.frame(1.0 / 60.0);
        alone.frame(1.0 / 60.0);
    }
    assert_eq!(sums(&together.world), sums(&alone.world));
}
