//! Pictures decoded concurrently, for the codecs whose references are known
//! from their headers.
//!
//! The caller walks the stream in order, doing everything that needs only
//! headers (parameter sets, picture order, reference marking, output order),
//! and turns each picture into a job that depends on the pictures it
//! references. A job starts only once those have finished, so a running job
//! never waits on another: the thread pool stays safe to share, and it needs
//! no thread of its own, which wasm cannot spawn.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

type Waiter = Box<dyn FnOnce() + Send>;

struct SlotState<T> {
    value: Option<Arc<T>>,
    waiters: Vec<Waiter>,
}

/// A picture's result, once it has one. Every slot must be finished, whatever
/// went wrong in its job, or what depends on it never starts.
pub(crate) struct Slot<T> {
    state: Mutex<SlotState<T>>,
}

impl<T> Slot<T> {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Slot {
            state: Mutex::new(SlotState {
                value: None,
                waiters: Vec::new(),
            }),
        })
    }

    pub(crate) fn finish(&self, value: T) {
        let waiters = {
            let mut s = self.state.lock().unwrap();
            s.value = Some(Arc::new(value));
            std::mem::take(&mut s.waiters)
        };
        for w in waiters {
            w();
        }
    }

    pub(crate) fn get(&self) -> Option<Arc<T>> {
        self.state.lock().unwrap().value.clone()
    }

    fn on_finish(&self, f: Waiter) {
        let mut s = self.state.lock().unwrap();
        if s.value.is_some() {
            drop(s);
            f();
        } else {
            s.waiters.push(f);
        }
    }
}

/// Jobs in flight, bounded so that memory does not grow with the stream.
pub(crate) struct Jobs {
    limit: usize,
    outstanding: Mutex<usize>,
    idle: Condvar,
}

impl Jobs {
    /// Up to `limit` jobs at once, waiting ones included. Any limit is safe:
    /// the earliest job's references were submitted before it, so it can
    /// always run.
    pub(crate) fn new(limit: usize) -> Arc<Self> {
        Arc::new(Jobs {
            limit: limit.max(1),
            outstanding: Mutex::new(0),
            idle: Condvar::new(),
        })
    }

    /// Run `job` once every slot in `deps` has finished, blocking first while
    /// the limit is reached.
    pub(crate) fn submit<T: 'static>(
        self: &Arc<Self>,
        deps: &[Arc<Slot<T>>],
        job: impl FnOnce() + Send + 'static,
    ) {
        {
            let mut n = self.outstanding.lock().unwrap();
            while *n >= self.limit {
                n = self.idle.wait(n).unwrap();
            }
            *n += 1;
        }
        // One more than the references, so that none finishing during the
        // loop can start the job before all are counted.
        let left = Arc::new(AtomicUsize::new(deps.len() + 1));
        let jobs = self.clone();
        let job = Arc::new(Mutex::new(Some(job)));
        let arrive = move || {
            if left.fetch_sub(1, Ordering::AcqRel) == 1 {
                let job = job.lock().unwrap().take().expect("a job starts once");
                let jobs = jobs.clone();
                crate::spawn(move || {
                    job();
                    let mut n = jobs.outstanding.lock().unwrap();
                    *n -= 1;
                    jobs.idle.notify_all();
                });
            }
        };
        for dep in deps {
            let arrive = arrive.clone();
            dep.on_finish(Box::new(arrive));
        }
        arrive();
    }

    /// Block until every job has run.
    pub(crate) fn wait(&self) {
        let mut n = self.outstanding.lock().unwrap();
        while *n > 0 {
            n = self.idle.wait(n).unwrap();
        }
    }
}

/// `f`, with a panic turned into an error: bad input must not bring a pool
/// thread down, and a job has to get as far as finishing its slot.
pub(crate) fn caught<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|_| Err("decoder panicked on this picture".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn a_job_waits_for_every_dependency() {
        let jobs = Jobs::new(8);
        let (a, b) = (Slot::<u32>::new(), Slot::<u32>::new());
        let ran = Arc::new(AtomicBool::new(false));
        let (ran2, a2, b2) = (ran.clone(), a.clone(), b.clone());
        jobs.submit(&[a.clone(), b.clone()], move || {
            assert!(a2.get().is_some() && b2.get().is_some());
            ran2.store(true, Ordering::SeqCst);
        });
        assert!(!ran.load(Ordering::SeqCst));
        a.finish(1);
        assert!(!ran.load(Ordering::SeqCst));
        b.finish(2);
        jobs.wait();
        assert!(ran.load(Ordering::SeqCst));
    }

    /// A chain longer than the limit must drain: each job's reference was
    /// submitted before it, so the limit can only ever delay the caller.
    #[test]
    fn a_chain_longer_than_the_limit_finishes() {
        let jobs = Jobs::new(2);
        let count = Arc::new(AtomicUsize::new(0));
        let mut last: Option<Arc<Slot<u32>>> = None;
        for i in 0..100 {
            let slot = Slot::new();
            let deps: Vec<_> = last.iter().cloned().collect();
            let (mine, count) = (slot.clone(), count.clone());
            jobs.submit(&deps, move || {
                assert_eq!(count.fetch_add(1, Ordering::SeqCst), i);
                mine.finish(i as u32);
            });
            last = Some(slot);
        }
        jobs.wait();
        assert_eq!(count.load(Ordering::SeqCst), 100);
    }

    #[test]
    fn a_panic_becomes_an_error() {
        let out: Result<(), String> = caught(|| panic!("bad stream"));
        assert!(out.is_err());
    }
}
