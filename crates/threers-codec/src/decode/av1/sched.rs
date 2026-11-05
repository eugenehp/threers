//! Frame threading: frames decoded concurrently, in whatever order their
//! references allow.
//!
//! A frame is handed to the thread pool only once every frame it references
//! has finished, so a running frame never blocks on another. That is what
//! makes the pool safe to share with the tile and loop-filter work inside
//! each frame: a pool thread waiting in a parallel loop may pick up another
//! frame, and if frames could block, that frame could be waiting on the one
//! suspended beneath it. It also needs no thread of its own, which wasm
//! cannot spawn.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use super::decode::TileBufs;
use super::{caught, FrameJob, Reconstructed};

#[derive(Default)]
struct State {
    /// Frames whose references are done, by decode order: the earliest goes
    /// first, as it is the one output is waiting for.
    ready: BTreeMap<u64, FrameJob>,
    running: usize,
    /// Frames submitted and not yet finished.
    outstanding: usize,
    next_id: u64,
    /// The first failure in decode order, shown frame or not.
    error: Option<(u64, String)>,
}

pub(crate) struct Sched {
    limit: usize,
    state: Mutex<State>,
    idle: Condvar,
    pools: Mutex<Vec<Vec<TileBufs>>>,
    cancelled: AtomicBool,
}

impl Sched {
    /// A scheduler running up to `limit` frames at once, with `pool` as the
    /// first set of tile buffers.
    pub(crate) fn new(limit: usize, pool: Vec<TileBufs>) -> Arc<Self> {
        Arc::new(Sched {
            limit,
            state: Mutex::default(),
            idle: Condvar::new(),
            pools: Mutex::new(vec![pool]),
            cancelled: AtomicBool::new(false),
        })
    }

    /// Take a frame, to start once its references have finished.
    pub(crate) fn submit(self: &Arc<Self>, job: FrameJob) {
        let id = {
            let mut s = self.state.lock().unwrap();
            s.outstanding += 1;
            s.next_id += 1;
            s.next_id
        };
        let deps = job.dependencies();
        // One more than the references, so that none finishing during this
        // loop can start the frame before all are counted.
        let left = Arc::new(AtomicUsize::new(deps.len() + 1));
        let job = Arc::new(Mutex::new(Some(job)));
        let arrive = {
            let (me, left, job) = (self.clone(), left, job);
            Arc::new(move || {
                if left.fetch_sub(1, Ordering::AcqRel) == 1 {
                    let j = job.lock().unwrap().take().unwrap();
                    me.state.lock().unwrap().ready.insert(id, j);
                    me.pump();
                }
            })
        };
        for d in deps {
            let a = arrive.clone();
            d.when_done(Box::new(move || a()));
        }
        arrive();
    }

    /// Start ready frames, up to the limit.
    fn pump(self: &Arc<Self>) {
        loop {
            let (id, job) = {
                let mut s = self.state.lock().unwrap();
                if s.running >= self.limit {
                    return;
                }
                let Some(next) = s.ready.pop_first() else {
                    return;
                };
                s.running += 1;
                next
            };
            let me = self.clone();
            crate::spawn(move || me.run(id, job));
        }
    }

    fn run(self: &Arc<Self>, id: u64, job: FrameJob) {
        let r = if self.cancelled.load(Ordering::Relaxed) {
            Err("decoding stopped".into())
        } else {
            let mut pool = self.pools.lock().unwrap().pop().unwrap_or_default();
            let r = caught(|| job.reconstruct(&mut pool).map(Reconstructed::finish));
            self.pools.lock().unwrap().push(pool);
            r
        };
        if let Err(e) = &r {
            let mut s = self.state.lock().unwrap();
            if s.error.as_ref().is_none_or(|(first, _)| id < *first) {
                s.error = Some((id, e.clone()));
            }
        }
        job.done.set(r);
        {
            let mut s = self.state.lock().unwrap();
            s.running -= 1;
            s.outstanding -= 1;
        }
        self.idle.notify_all();
        self.pump();
    }

    /// Frames not yet started fail at once; used when output has stopped.
    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Wait for every frame, and return the tile buffers and the first error.
    pub(crate) fn finish(&self) -> (Vec<TileBufs>, Option<String>) {
        let mut s = self.state.lock().unwrap();
        while s.outstanding != 0 {
            s = self.idle.wait(s).unwrap();
        }
        let error = s.error.take().map(|(_, e)| e);
        (self.pools.lock().unwrap().pop().unwrap_or_default(), error)
    }
}
