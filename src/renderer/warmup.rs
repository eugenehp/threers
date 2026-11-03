//! Building a lazy renderer's pipelines ahead of the first draw.
//!
//! [`Renderer::new_lazy`](super::Renderer::new_lazy) returns in milliseconds
//! and leaves each pipeline to be built when a draw first needs it — which,
//! for a viewport, is a frame that stalls while the driver compiles. A
//! [`PipelineWarmup`] moves that work to wherever the application wants it:
//! a background thread while a loading state is up, several threads at once,
//! or spread over idle frames.
//!
//! ```ignore
//! let renderer = Renderer::new_lazy(device, queue, format, w, h);
//! let warmup = renderer.warm_up();
//! let workers: Vec<_> = (0..4).map(|_| {
//!     let w = warmup.clone();
//!     std::thread::spawn(move || w.run())
//! }).collect();
//! // …show a spinner; poll `warmup.progress()` for a bar…
//! warmup.wait();
//! assert!(warmup.errors().is_empty());
//! ```
//!
//! Building a pipeline here and drawing with it on another thread is safe:
//! each pipeline is built once, and a draw that gets there first makes the
//! warm-up wait for it rather than build it again.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use super::pipeline_state::Pipeline;

/// A pipeline that failed validation while warming up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineError {
    /// The descriptor's label.
    pub label: String,
    /// What wgpu's validation said.
    pub message: String,
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.label, self.message)
    }
}

/// Pipelines to build, and how far along building them is. Cheap to clone;
/// every clone is the same job. `Send + Sync`.
#[derive(Clone)]
pub struct PipelineWarmup {
    inner: Arc<Inner>,
}

struct Inner {
    pipelines: Vec<Pipeline>,
    /// The next index a `run` takes.
    next: AtomicUsize,
    /// How many are built (or failed).
    done: AtomicUsize,
    errors: Mutex<Vec<PipelineError>>,
    /// Signalled when `done` reaches the total.
    finished: (Mutex<()>, Condvar),
}

impl PipelineWarmup {
    pub(crate) fn new(pipelines: Vec<Pipeline>) -> Self {
        Self {
            inner: Arc::new(Inner {
                pipelines,
                next: AtomicUsize::new(0),
                done: AtomicUsize::new(0),
                errors: Mutex::new(Vec::new()),
                finished: (Mutex::new(()), Condvar::new()),
            }),
        }
    }

    /// How many pipelines this job builds.
    pub fn len(&self) -> usize {
        self.inner.pipelines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.pipelines.is_empty()
    }

    /// Build pipelines until none are left; returns how many this call built.
    /// Call it from as many threads as should share the work — each takes the
    /// next pipeline nobody has started.
    pub fn run(&self) -> usize {
        let mut built = 0;
        while self.step() {
            built += 1;
        }
        built
    }

    /// Build one pipeline; `false` when none were left to start. For
    /// spreading a warm-up over frames: a few steps per idle frame.
    pub fn step(&self) -> bool {
        let i = self.inner.next.fetch_add(1, Ordering::Relaxed);
        let Some(pipeline) = self.inner.pipelines.get(i) else {
            return false;
        };
        if let Err(message) = pipeline.compile() {
            self.inner.errors.lock().unwrap_or_else(|e| e.into_inner()).push(PipelineError {
                label: pipeline.label().unwrap_or("").to_owned(),
                message,
            });
        }
        let done = self.inner.done.fetch_add(1, Ordering::AcqRel) + 1;
        if done == self.len() {
            let (lock, cvar) = &self.inner.finished;
            let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
            cvar.notify_all();
        }
        true
    }

    /// `(built, total)`.
    pub fn progress(&self) -> (usize, usize) {
        (self.inner.done.load(Ordering::Acquire), self.len())
    }

    /// Every pipeline is built: the viewport can be shown without a first
    /// frame that stalls on compilation.
    pub fn is_ready(&self) -> bool {
        self.inner.done.load(Ordering::Acquire) == self.len()
    }

    /// Block until every pipeline is built — by whichever threads are
    /// running the job. Returns at once when it is empty or done. Something
    /// has to be calling [`run`](Self::run) or [`step`](Self::step), or this
    /// waits for ever.
    pub fn wait(&self) {
        let (lock, cvar) = &self.inner.finished;
        let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
        while !self.is_ready() {
            held = cvar.wait(held).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Pipelines that failed validation so far.
    pub fn errors(&self) -> Vec<PipelineError> {
        self.inner.errors.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

impl std::fmt::Debug for PipelineWarmup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (done, total) = self.progress();
        f.debug_struct("PipelineWarmup").field("built", &done).field("total", &total).finish()
    }
}

#[cfg(not(target_arch = "wasm32"))]
const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    send_sync::<PipelineWarmup>();
};
