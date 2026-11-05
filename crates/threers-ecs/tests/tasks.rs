//! Background work: threads, rayon, tokio and parallel queries.

use std::collections::HashSet;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use threers::core::Object3D;
use threers_ecs::prelude::*;
use threers_ecs::tasks::{Jobs, Task, TaskFailed, Tasks};

fn wait<T>(task: &mut Task<T>) -> Result<T, TaskFailed> {
    let start = Instant::now();
    loop {
        if let Some(result) = task.poll() {
            return result;
        }
        assert!(start.elapsed() < Duration::from_secs(10), "timed out");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn a_result_comes_back_exactly_once() {
    let mut task = Tasks::new().spawn(|| 6 * 7);
    assert_eq!(wait(&mut task), Ok(42));
    assert!(task.is_finished());
    assert_eq!(task.poll(), None);
}

#[test]
fn poll_never_waits_for_a_slow_job() {
    let mut task = Tasks::new().spawn(|| std::thread::sleep(Duration::from_millis(300)));
    let start = Instant::now();
    assert!(task.poll().is_none());
    assert!(start.elapsed() < Duration::from_millis(100));
    wait(&mut task).unwrap();
}

#[test]
fn a_job_runs_off_this_thread() {
    let mut task = Tasks::new().spawn(|| std::thread::current().id());
    assert_ne!(wait(&mut task).unwrap(), std::thread::current().id());
}

#[test]
fn a_panic_comes_back_as_an_error_and_the_pool_survives() {
    let tasks = Tasks::new();
    let mut bad = tasks.spawn(|| -> u32 { panic!("no luck") });
    assert_eq!(wait(&mut bad), Err(TaskFailed("no luck".to_owned())));
    for _ in 0..64 {
        let mut bad = tasks.spawn(|| -> u32 { panic!("again") });
        assert!(wait(&mut bad).is_err());
    }
    let mut good = tasks.spawn(|| 1);
    assert_eq!(wait(&mut good), Ok(1));
}

#[test]
fn many_jobs_all_finish() {
    let tasks = Tasks::new();
    let mut all: Vec<Task<u32>> = (0..500).map(|i| tasks.spawn(move || i * 2)).collect();
    let mut total = 0;
    for task in &mut all {
        total += wait(task).unwrap();
    }
    assert_eq!(total, 499 * 500);
}

#[test]
fn jobs_run_by_name_here_or_on_a_thread() {
    fn double(input: &[u8]) -> Vec<u8> {
        input.iter().map(|b| b * 2).collect()
    }
    let jobs = Jobs::new().register("double", double);
    assert_eq!(jobs.run_here("double", &[1, 2]).unwrap(), [2, 4]);
    let tasks = Tasks::new();
    let mut task = jobs.run(&tasks, "double", vec![3]);
    assert_eq!(wait(&mut task).unwrap(), [6]);

    assert!(jobs.run_here("nope", &[]).is_err());
    let mut missing = jobs.run(&tasks, "nope", vec![]);
    assert!(wait(&mut missing).unwrap_err().0.contains("nope"));
}

struct Value(u64);
impl Component for Value {}

fn filled(count: u64) -> (World, Vec<Entity>) {
    let mut w = World::new();
    let ids = (0..count)
        .map(|i| {
            let id = w.spawn(Object3D::group());
            w.insert(id, Value(i));
            id
        })
        .collect();
    (w, ids)
}

#[test]
fn par_each_touches_every_row_once_with_its_own_entity() {
    for count in [0, 1, 50, 127, 128, 5000] {
        let (mut w, ids) = filled(count);
        let seen = Mutex::new(HashSet::new());
        w.par_each::<Value>(|id, value| {
            value.0 += 1;
            assert!(seen.lock().unwrap().insert(id), "visited twice");
        });
        assert_eq!(seen.into_inner().unwrap().len(), count as usize);
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(w.get::<Value>(*id).unwrap().0, i as u64 + 1);
        }
    }
}

#[test]
fn par_each_counts_as_a_change_and_skips_deleted_nodes() {
    let (mut w, ids) = filled(300);
    w.clear_trackers();
    w.scene.arena.delete(ids[10]);
    let visited = Mutex::new(0);
    w.par_each::<Value>(|_, _| *visited.lock().unwrap() += 1);
    assert_eq!(visited.into_inner().unwrap(), 299);
    assert_eq!(w.count::<Changed<Value>>(), 299);
}

#[cfg(feature = "parallel")]
#[test]
fn with_parallel_a_long_list_really_spreads_over_threads() {
    let (mut w, _) = filled(20_000);
    let threads = Mutex::new(HashSet::new());
    w.par_each::<Value>(|_, value| {
        for _ in 0..200 {
            value.0 = std::hint::black_box(value.0.wrapping_mul(31));
        }
        threads.lock().unwrap().insert(std::thread::current().id());
    });
    let used = threads.into_inner().unwrap().len();
    assert!(used > 1, "ran on {used} thread");
}

#[cfg(feature = "parallel")]
#[test]
fn with_parallel_a_short_list_stays_on_this_thread() {
    let (mut w, _) = filled(20);
    let here = std::thread::current().id();
    w.par_each::<Value>(move |_, _| assert_eq!(std::thread::current().id(), here));
}

#[cfg(feature = "tokio")]
mod with_tokio {
    use super::*;

    #[test]
    fn a_future_runs_on_the_runtime() {
        let tasks = Tasks::new();
        let mut task = tasks.spawn_async(async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            "done"
        });
        assert!(task.poll().is_none());
        assert_eq!(wait(&mut task), Ok("done"));
    }

    #[test]
    fn many_futures_share_it_without_a_thread_each() {
        let tasks = Tasks::new();
        let start = Instant::now();
        let mut all: Vec<_> = (0..2000)
            .map(|i| {
                tasks.spawn_async(async move {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    i
                })
            })
            .collect();
        let total: i32 = all.iter_mut().map(|task| wait(task).unwrap()).sum();
        assert_eq!(total, 1999 * 1000);
        assert!(start.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn a_panicking_future_is_an_error() {
        let tasks = Tasks::new();
        let mut task = tasks.spawn_async(async { panic!("boom") });
        let result: Result<(), TaskFailed> = wait(&mut task);
        assert!(result.is_err());
    }

    #[test]
    fn closing_does_not_wait_for_a_stuck_future() {
        let tasks = Tasks::new();
        let _task = tasks.spawn_async(async {
            tokio::time::sleep(Duration::from_secs(600)).await;
        });
        let start = Instant::now();
        drop(tasks);
        assert!(start.elapsed() < Duration::from_secs(1));
    }
}

#[test]
fn a_system_can_start_work_and_collect_it_on_a_later_frame() {
    struct Pending(Option<Task<u32>>);
    fn drive(w: &mut World) {
        if w.resource::<Pending>().0.is_none() && !w.has_resource_done() {
            let task = w.resource::<Tasks>().spawn(|| {
                std::thread::sleep(Duration::from_millis(30));
                7
            });
            w.resource_mut::<Pending>().0 = Some(task);
        }
        let result = w
            .resource_mut::<Pending>()
            .0
            .as_mut()
            .and_then(|task| task.poll());
        if let Some(result) = result {
            w.insert_resource(result.unwrap());
        }
    }
    trait Done {
        fn has_resource_done(&self) -> bool;
    }
    impl Done for World {
        fn has_resource_done(&self) -> bool {
            self.get_resource::<u32>().is_some()
        }
    }

    let mut app = App::new();
    app.resource(Tasks::new()).resource(Pending(None));
    app.system(drive);
    let start = Instant::now();
    let mut frames = 0;
    while app.world.get_resource::<u32>().is_none() {
        app.frame(1.0 / 60.0);
        frames += 1;
        assert!(start.elapsed() < Duration::from_secs(5));
    }
    assert_eq!(*app.world.resource::<u32>(), 7);
    assert!(frames > 100, "frames kept running meanwhile: {frames}");
}
