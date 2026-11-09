//! Work that takes longer than a frame.
//!
//! Start a job from a system, carry on, and pick the result up on a later frame.
//! A game loop has to keep its deadline, so nothing here ever blocks it.
//!
//! | Need | Use | Runs on |
//! |---|---|---|
//! | A closure, natively | [`Tasks::spawn`] | A thread pool, or rayon with `parallel` |
//! | A future | `Tasks::spawn_async` | tokio, with the `tokio` feature |
//! | The same job on every platform | [`Jobs`] | Threads natively, Web Workers in a browser |
//! | One step over many components | [`World::par_each`] | rayon, with `parallel` |
//!
//! ```
//! # #[cfg(not(target_arch = "wasm32"))] {
//! use threers_ecs::tasks::{Task, Tasks};
//!
//! let tasks = Tasks::new();
//! let mut sum: Task<u64> = tasks.spawn(|| (0..1000).sum());
//! loop {
//!     // In a game, this is one look per frame.
//!     if let Some(result) = sum.poll() {
//!         assert_eq!(result, Ok(499_500));
//!         break;
//!     }
//! }
//! # }
//! ```
//!
//! # Jobs and Web Workers
//!
//! A browser cannot run a closure on a worker: a worker is a separate wasm
//! instance sharing nothing, so all that crosses is a name and bytes. [`Jobs`] has
//! that shape on every platform, so a job written once runs on a thread natively
//! and in a worker on the web. The `web` module has the pool.

use crate::query::{Parallel, Term};
use crate::world::{Component, Entity, World};
use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};

/// A job that panicked or was lost, with what it said.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskFailed(pub String);

impl std::fmt::Display for TaskFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "task failed: {}", self.0)
    }
}

impl std::error::Error for TaskFailed {}

/// A result on its way.
pub struct Task<T> {
    receiver: Receiver<Result<T, TaskFailed>>,
    finished: bool,
}

/// The sending half, for whatever runs the job.
pub(crate) fn task<T>() -> (Sender<Result<T, TaskFailed>>, Task<T>) {
    let (sender, receiver) = channel();
    (
        sender,
        Task {
            receiver,
            finished: false,
        },
    )
}

impl<T> Task<T> {
    /// The result, once. `None` while it is still running, and after it has been
    /// taken. Never waits.
    pub fn poll(&mut self) -> Option<Result<T, TaskFailed>> {
        if self.finished {
            return None;
        }
        match self.receiver.try_recv() {
            Ok(result) => {
                self.finished = true;
                Some(result)
            }
            Err(TryRecvError::Empty) => None,
            // The sender went without a word, so nothing is ever coming.
            Err(TryRecvError::Disconnected) => {
                self.finished = true;
                Some(Err(TaskFailed("it was dropped".to_owned())))
            }
        }
    }

    /// Whether the result has been taken.
    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

#[cfg(any(
    not(target_arch = "wasm32"),
    all(target_feature = "atomics", feature = "parallel")
))]
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "it panicked".to_owned())
}

/// A panic in a job comes back as an error, rather than taking the game with it or
/// leaving someone waiting for ever.
#[cfg(any(
    not(target_arch = "wasm32"),
    all(target_feature = "atomics", feature = "parallel")
))]
fn run_caught<T>(job: impl FnOnce() -> T, sender: Sender<Result<T, TaskFailed>>) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job))
        .map_err(|payload| TaskFailed(panic_message(payload)));
    let _ = sender.send(result);
}

#[cfg(all(not(target_arch = "wasm32"), not(feature = "parallel")))]
mod pool {
    use std::sync::mpsc::{channel, Sender};
    use std::sync::{Arc, Mutex};

    type Work = Box<dyn FnOnce() + Send>;

    /// A fixed set of threads taking from one queue, so a burst of jobs queues up
    /// rather than starting a thread each.
    pub(super) struct Pool {
        work: Sender<Work>,
    }

    impl Pool {
        pub(super) fn new() -> Self {
            let (work, inbox) = channel::<Work>();
            let inbox = Arc::new(Mutex::new(inbox));
            let threads = std::thread::available_parallelism().map_or(2, |n| n.get());
            for index in 0..threads.saturating_sub(1).max(1) {
                let inbox = inbox.clone();
                let thread = std::thread::Builder::new().name(format!("threers-task-{index}"));
                let spawned = thread.spawn(move || loop {
                    // Held only to take a job, so the others can take theirs.
                    let next = inbox.lock().map(|inbox| inbox.recv());
                    match next {
                        Ok(Ok(job)) => job(),
                        _ => return,
                    }
                });
                if let Err(error) = spawned {
                    eprintln!("no task thread: {error}");
                }
            }
            Self { work }
        }

        pub(super) fn run(&self, job: Work) {
            let _ = self.work.send(job);
        }
    }
}

/// Runs jobs off the main thread, as a resource.
pub struct Tasks {
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "parallel")))]
    pool: pool::Pool,
    #[cfg(all(feature = "tokio", not(target_arch = "wasm32")))]
    runtime: Option<tokio::runtime::Runtime>,
}

impl Default for Tasks {
    fn default() -> Self {
        Self::new()
    }
}

impl Tasks {
    /// The task pools: rayon's with the `parallel` feature, a small one of its own
    /// otherwise, and a tokio runtime with `tokio`.
    pub fn new() -> Self {
        Self {
            #[cfg(all(not(target_arch = "wasm32"), not(feature = "parallel")))]
            pool: pool::Pool::new(),
            #[cfg(all(feature = "tokio", not(target_arch = "wasm32")))]
            runtime: tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|error| eprintln!("no tokio runtime: {error}"))
                .ok(),
        }
    }

    /// Run `job` in the background.
    #[cfg(any(
        not(target_arch = "wasm32"),
        all(target_feature = "atomics", feature = "parallel")
    ))]
    pub fn spawn<T: Send + 'static>(&self, job: impl FnOnce() -> T + Send + 'static) -> Task<T> {
        let (sender, task) = task();
        let work = move || run_caught(job, sender);
        #[cfg(feature = "parallel")]
        rayon::spawn(work);
        #[cfg(all(not(target_arch = "wasm32"), not(feature = "parallel")))]
        self.pool.run(Box::new(work));
        task
    }

    /// Run a future on the tokio runtime, for network calls and the like.
    #[cfg(all(feature = "tokio", not(target_arch = "wasm32")))]
    pub fn spawn_async<T: Send + 'static>(
        &self,
        future: impl std::future::Future<Output = T> + Send + 'static,
    ) -> Task<T> {
        let (sender, task) = task();
        let Some(runtime) = &self.runtime else {
            let _ = sender.send(Err(TaskFailed("there is no tokio runtime".to_owned())));
            return task;
        };
        let handle = runtime.spawn(future);
        // A second task watches the first, so that a panic in it is reported.
        runtime.spawn(async move {
            let result = handle.await.map_err(|error| TaskFailed(error.to_string()));
            let _ = sender.send(result);
        });
        task
    }

    /// The tokio runtime itself, for anything `spawn_async` does not cover.
    #[cfg(all(feature = "tokio", not(target_arch = "wasm32")))]
    pub fn runtime(&self) -> Option<&tokio::runtime::Runtime> {
        self.runtime.as_ref()
    }
}

#[cfg(all(feature = "tokio", not(target_arch = "wasm32")))]
impl Drop for Tasks {
    /// Dropping a runtime waits for its tasks, and a game closing should not.
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

/// A job that can run anywhere: bytes in, bytes out.
pub type JobFn = fn(&[u8]) -> Vec<u8>;

/// Named jobs, the same on every platform. See the [module docs](self#jobs-and-web-workers).
#[derive(Default, Clone)]
pub struct Jobs {
    table: HashMap<String, JobFn>,
}

impl Jobs {
    /// No jobs yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a job a worker can run by name.
    pub fn register(mut self, name: &str, job: JobFn) -> Self {
        self.table.insert(name.to_owned(), job);
        self
    }

    /// Run a job here and now. This is what a worker calls.
    ///
    /// # Errors
    ///
    /// If there is no job of that name.
    pub fn run_here(&self, name: &str, input: &[u8]) -> Result<Vec<u8>, TaskFailed> {
        let job = self
            .table
            .get(name)
            .ok_or_else(|| TaskFailed(format!("no job named {name:?}")))?;
        Ok(job(input))
    }

    /// Run a job on a background thread.
    #[cfg(any(
        not(target_arch = "wasm32"),
        all(target_feature = "atomics", feature = "parallel")
    ))]
    pub fn run(&self, tasks: &Tasks, name: &str, input: Vec<u8>) -> Task<Vec<u8>> {
        match self.table.get(name).copied() {
            Some(job) => tasks.spawn(move || job(&input)),
            None => {
                let (sender, task) = task();
                let _ = sender.send(Err(TaskFailed(format!("no job named {name:?}"))));
                task
            }
        }
    }
}

/// Whether blocking on other threads is allowed here. A browser's main thread may
/// not wait, so rayon's fork and join would trap there.
#[cfg(all(feature = "parallel", target_arch = "wasm32", feature = "web"))]
fn may_block() -> bool {
    use wasm_bindgen::JsCast;
    js_sys::global().dyn_into::<web_sys::Window>().is_err()
}

#[cfg(all(feature = "parallel", target_arch = "wasm32", not(feature = "web")))]
fn may_block() -> bool {
    false
}

#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
fn may_block() -> bool {
    true
}

/// How few rows make threads a waste: dispatch costs more than short work.
#[cfg(feature = "parallel")]
const MIN_PARALLEL: usize = threers::utils::parallel::MIN_PARALLEL;

impl World {
    /// Call `f` on every `C`, on many threads with the `parallel` feature.
    ///
    /// It goes over the packed rows directly, so `f` sees only the component, which
    /// is how it can be safe: no two threads share anything. The order is not
    /// defined, and short lists and a browser's main thread run on this thread.
    pub fn par_each<C: Component + Send>(&mut self, f: impl Fn(Entity, &mut C) + Send + Sync) {
        // Rows of nodes deleted behind the world's back would otherwise be visited,
        // and there is no arena to check against from inside the threads.
        self.prune();
        let tick = self.tick;
        let store = self.store_mut::<C>();
        store.changed.fill(tick);
        #[cfg(all(
            feature = "parallel",
            any(not(target_arch = "wasm32"), target_feature = "atomics")
        ))]
        if store.values.len() >= MIN_PARALLEL && may_block() {
            use rayon::prelude::*;
            store
                .values
                .par_iter_mut()
                .zip(store.ids.par_iter())
                .for_each(|(value, id)| f(*id, value));
            return;
        }
        for (value, id) in store.values.iter_mut().zip(&store.ids) {
            f(*id, value);
        }
    }

    /// [`each`](Self::each) over several components on many threads, with the
    /// `parallel` feature. Terms are [`Parallel`]: components and filters, not the
    /// node. The order is not defined, and short lists and a browser's main thread
    /// run on this thread.
    ///
    /// # Panics
    ///
    /// If `Q` borrows a component twice.
    pub fn par_query<Q: Term + Parallel>(&mut self, f: impl for<'a> Fn(Q::Item<'a>) + Send + Sync) {
        // As `par_each`: no arena to check against from inside the threads.
        self.prune();
        let mut claims = Vec::new();
        Q::claim(&mut claims);

        /// A query's raw views, shared between threads. `Parallel` is the promise
        /// that makes it sound.
        struct Shared<T>(T);
        unsafe impl<T> Send for Shared<T> {}
        unsafe impl<T> Sync for Shared<T> {}
        impl<T> Shared<T> {
            fn get(&self) -> &T {
                &self.0
            }
        }

        let world: *mut World = self;
        // SAFETY: as for `each`, and each id appears once, so no two threads
        // fetch one row. The world is borrowed exclusively throughout.
        unsafe {
            let state = Shared(Q::state(world));
            let every;
            let ids: &[Entity] = match Q::candidates(state.get()) {
                Some(ids) => &*ids,
                None => {
                    every = (*world).scene.arena.nodes.keys().collect::<Vec<_>>();
                    &every
                }
            };
            let visit = |id: Entity| {
                if Q::matches(state.get(), id)
                    && let Some(item) = Q::fetch(state.get(), id)
                {
                    f(item);
                }
            };
            #[cfg(all(
                feature = "parallel",
                any(not(target_arch = "wasm32"), target_feature = "atomics")
            ))]
            if ids.len() >= MIN_PARALLEL && may_block() {
                use rayon::prelude::*;
                ids.par_iter().for_each(|&id| visit(id));
                return;
            }
            ids.iter().copied().for_each(visit);
        }
    }
}
