//! Work done off the UI thread: [`Ui::spawn_job`] runs a [`Job`] (a plain
//! function from bytes to bytes) somewhere else and hands what it made to a
//! callback on the UI thread, at the next [`Input::tick`] after it is done.
//!
//! Where it runs is the app's choice, by the [`JobRunner`] it sets with
//! [`Ui::set_job_runner`]:
//!
//! - none: on the spot, inside `spawn_job` (the callback too), so an app
//!   that sets nothing behaves as if the work were a plain call;
//! - [`Threads`]: a pool of native threads;
//! - `Workers` (in a browser): Web Workers, each running its own
//!   instance of the app's WebAssembly module. A job is named by its
//!   function's place in the module's function table, which is the same in
//!   every instance of one module; its input and output cross as
//!   transferred buffers. No shared memory, so the page needs no
//!   cross-origin isolation.
//!
//! A job sees only its bytes (in a worker, statics set on the UI thread are
//! not there), so it encodes everything it needs into them.
//!
//! ```
//! use threers_ui::jobs::Threads;
//! use threers_ui::Ui;
//! use std::{cell::Cell, rc::Rc};
//!
//! fn total(input: &[u8]) -> Vec<u8> {
//!     vec![input.iter().fold(0u8, |a, b| a.wrapping_add(*b))]
//! }
//!
//! let mut ui = Ui::new();
//! ui.set_job_runner(Threads::new(2));
//! let got = Rc::new(Cell::new(None));
//! let g = got.clone();
//! ui.spawn_job(total, vec![1, 2, 3], move |_, out| g.set(Some(out[0])));
//! while ui.jobs_pending() {
//!     ui.poll_jobs();
//! }
//! assert_eq!(got.get(), Some(6));
//! ```
//!
//! [`Input::tick`]: crate::input::Input::tick

use std::collections::HashMap;

use crate::tree::Ui;

/// Work for another thread: from its input to its output. A plain
/// function, not a closure, so a worker with its own copy of the program
/// can find it.
pub type Job = fn(&[u8]) -> Vec<u8>;

/// Runs jobs somewhere and says which have finished.
pub trait JobRunner {
    /// Starts `job` on `input`; its output comes back from
    /// [`take_done`](Self::take_done) under `id`.
    fn submit(&mut self, id: u64, job: Job, input: Vec<u8>);
    /// The jobs finished since the last call.
    fn take_done(&mut self) -> Vec<(u64, Vec<u8>)>;
}

type Then = Box<dyn FnOnce(&mut Ui, Vec<u8>)>;

/// A job started by [`Ui::spawn_job`], to cancel it with
/// [`Ui::cancel_job`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobId(u64);

/// A callback waiting on a job, and what it belongs to.
struct Waiting {
    then: Then,
    /// Dropped, not run, if this node is gone when the job lands.
    owner: Option<crate::tree::NodeId>,
}

/// The jobs of a [`Ui`]: its runner and the callbacks waiting on it.
#[derive(Default)]
pub(crate) struct Jobs {
    runner: Option<Box<dyn JobRunner>>,
    next: u64,
    waiting: HashMap<u64, Waiting>,
    /// The newest job under each key (see [`Ui::spawn_latest`]).
    latest: HashMap<String, u64>,
}

impl Ui {
    /// Where jobs run from now on (see the [module docs](crate::jobs)).
    pub fn set_job_runner(&mut self, runner: impl JobRunner + 'static) {
        self.env.jobs.borrow_mut().runner = Some(Box::new(runner));
    }

    /// Runs `job` on `input` off this thread, then `then` with its output
    /// here. Without a runner both happen now.
    pub fn spawn_job(&mut self, job: Job, input: Vec<u8>, then: impl FnOnce(&mut Ui, Vec<u8>) + 'static) -> JobId {
        self.start_job(job, input, Box::new(then), None)
    }

    /// [`spawn_job`](Self::spawn_job) for work that belongs to `owner`: if
    /// it has been removed by the time the job lands, `then` does not run
    /// (a view closed while its picture loaded).
    pub fn spawn_job_for(&mut self, owner: crate::tree::NodeId, job: Job, input: Vec<u8>, then: impl FnOnce(&mut Ui, Vec<u8>) + 'static) -> JobId {
        self.start_job(job, input, Box::new(then), Some(owner))
    }

    /// [`spawn_job`](Self::spawn_job), cancelling the job last started
    /// under `key` if it has not landed: only the newest answer is wanted
    /// (a search run as each letter is typed).
    pub fn spawn_latest(&mut self, key: &str, job: Job, input: Vec<u8>, then: impl FnOnce(&mut Ui, Vec<u8>) + 'static) -> JobId {
        let before = self.env.jobs.borrow_mut().latest.remove(key);
        if let Some(old) = before {
            self.cancel_job(JobId(old));
        }
        let id = self.start_job(job, input, Box::new(then), None);
        let mut jobs = self.env.jobs.borrow_mut();
        if jobs.waiting.contains_key(&id.0) {
            jobs.latest.insert(key.to_owned(), id.0);
        }
        id
    }

    /// Forgets a job: its callback will not run. (The work itself may
    /// still be running; its output is let go when it lands.) Returns
    /// whether it was still waiting.
    pub fn cancel_job(&mut self, id: JobId) -> bool {
        self.env.jobs.borrow_mut().waiting.remove(&id.0).is_some()
    }

    fn start_job(&mut self, job: Job, input: Vec<u8>, then: Then, owner: Option<crate::tree::NodeId>) -> JobId {
        let mut jobs = self.env.jobs.borrow_mut();
        let id = jobs.next;
        jobs.next += 1;
        if jobs.runner.is_none() {
            drop(jobs);
            let out = job(&input);
            if owner.is_none_or(|o| self.contains(o)) {
                then(self, out);
            }
            return JobId(id);
        }
        jobs.waiting.insert(id, Waiting { then, owner });
        jobs.runner.as_mut().expect("checked").submit(id, job, input);
        JobId(id)
    }

    /// Hands finished jobs' output to their callbacks. Returns whether any
    /// had finished. [`Input::tick`](crate::input::Input::tick) calls it.
    pub fn poll_jobs(&mut self) -> bool {
        let done: Vec<(Waiting, Vec<u8>)> = {
            let mut jobs = self.env.jobs.borrow_mut();
            let Some(runner) = jobs.runner.as_mut() else { return false };
            let done = runner.take_done();
            let landed: Vec<(Waiting, Vec<u8>)> = done.into_iter().filter_map(|(id, out)| jobs.waiting.remove(&id).map(|w| (w, out))).collect();
            let Jobs { waiting, latest, .. } = &mut *jobs;
            latest.retain(|_, id| waiting.contains_key(id));
            landed
        };
        let mut any = false;
        for (w, out) in done {
            if w.owner.is_none_or(|o| self.contains(o)) {
                (w.then)(self, out);
                any = true;
            }
        }
        any
    }

    /// Whether a job is still running. [`Ui::needs_frame`] counts it, so
    /// frames go on until it lands.
    pub fn jobs_pending(&self) -> bool {
        !self.env.jobs.borrow().waiting.is_empty()
    }
}

/// Jobs on a pool of native threads.
#[cfg(not(target_arch = "wasm32"))]
pub struct Threads {
    work: Option<std::sync::mpsc::Sender<(u64, Job, Vec<u8>)>>,
    done: std::sync::mpsc::Receiver<(u64, Vec<u8>)>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Threads {
    /// `count` threads (at least one).
    pub fn new(count: usize) -> Self {
        use std::sync::{mpsc, Arc, Mutex};
        let (work, queue) = mpsc::channel::<(u64, Job, Vec<u8>)>();
        let (finished, done) = mpsc::channel();
        let queue = Arc::new(Mutex::new(queue));
        let threads = (0..count.max(1))
            .map(|i| {
                let (queue, finished) = (queue.clone(), finished.clone());
                std::thread::Builder::new()
                    .name(format!("threers-ui job {i}"))
                    .spawn(move || loop {
                        let next = queue.lock().map(|q| q.recv());
                        let Ok(Ok((id, job, input))) = next else { return };
                        if finished.send((id, job(&input))).is_err() {
                            return;
                        }
                    })
                    .expect("a thread")
            })
            .collect();
        Threads { work: Some(work), done, threads }
    }

    /// As many threads as the machine has cores, less the UI's own.
    pub fn for_this_machine() -> Self {
        let n = std::thread::available_parallelism().map_or(2, |n| n.get());
        Self::new(n.saturating_sub(1).clamp(1, 8))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl JobRunner for Threads {
    fn submit(&mut self, id: u64, job: Job, input: Vec<u8>) {
        if let Some(w) = &self.work {
            let _ = w.send((id, job, input));
        }
    }

    fn take_done(&mut self) -> Vec<(u64, Vec<u8>)> {
        self.done.try_iter().collect()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for Threads {
    fn drop(&mut self) {
        // Closing the queue ends each thread after its current job.
        self.work = None;
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::Workers;

#[cfg(target_arch = "wasm32")]
mod web {
    use std::cell::RefCell;
    use std::rc::Rc;

    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;
    use web_sys::{Blob, BlobPropertyBag, MessageEvent, Url, Worker, WorkerOptions, WorkerType};

    use super::{Job, JobRunner};

    /// Runs a job in a worker: called there by the worker's script (see
    /// [`Workers`]), not by the app.
    #[wasm_bindgen]
    pub fn threers_ui_run_job(job: u32, input: &[u8]) -> Vec<u8> {
        // SAFETY: `job` is a `Job`'s index in the function table, taken by
        // `Workers::submit` in another instance of this same module, whose
        // table holds the same functions at the same places. A wrong index
        // traps at the call (WebAssembly checks the signature) rather than
        // running anything else.
        let job: Job = unsafe { std::mem::transmute::<usize, Job>(job as usize) };
        job(input)
    }

    /// The worker's script: it instantiates the module it is sent, then
    /// runs each job it is given and sends the output back.
    const SCRIPT: &str = r#"import init, { threers_ui_run_job } from "GLUE";
let ready;
onmessage = async (e) => {
  if (e.data.module) {
    ready = init({ module_or_path: e.data.module });
    return;
  }
  await ready;
  const { id, job, input } = e.data;
  const output = threers_ui_run_job(job, input);
  postMessage({ id, output }, [output.buffer]);
};
"#;

    /// Outputs that have come back, by job.
    type Done = Rc<RefCell<Vec<(u64, Vec<u8>)>>>;

    /// Jobs on Web Workers in a browser: each worker runs the app's own
    /// WebAssembly module, compiled once and sent to it. Jobs go to the
    /// least busy worker.
    pub struct Workers {
        workers: Vec<(Worker, usize)>,
        done: Done,
        _listen: Closure<dyn FnMut(MessageEvent)>,
        sent: std::collections::HashMap<u64, usize>,
    }

    impl Workers {
        /// `count` workers (at least one), loading the app's
        /// wasm-bindgen JavaScript from `glue_url` (`--target web`; an
        /// absolute URL, e.g. `new URL("pkg/app.js", import.meta.url).href`).
        pub fn new(glue_url: &str, count: usize) -> Result<Workers, JsValue> {
            let script = SCRIPT.replace("GLUE", &glue_url.replace('\\', "\\\\").replace('"', "\\\""));
            let parts = js_sys::Array::of1(&JsValue::from_str(&script));
            let bag = BlobPropertyBag::new();
            bag.set_type("text/javascript");
            let blob = Blob::new_with_str_sequence_and_options(&parts, &bag)?;
            let url = Url::create_object_url_with_blob(&blob)?;
            let options = WorkerOptions::new();
            options.set_type(WorkerType::Module);
            let done = Rc::new(RefCell::new(Vec::new()));
            let d = done.clone();
            let listen = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
                let data = e.data();
                let get = |k: &str| js_sys::Reflect::get(&data, &JsValue::from_str(k)).unwrap_or(JsValue::UNDEFINED);
                let (Some(id), Ok(out)) = (get("id").as_f64(), get("output").dyn_into::<js_sys::Uint8Array>()) else {
                    return;
                };
                d.borrow_mut().push((id as u64, out.to_vec()));
            });
            let module = wasm_bindgen::module();
            let hello = js_sys::Object::new();
            js_sys::Reflect::set(&hello, &JsValue::from_str("module"), &module)?;
            let mut workers = Vec::new();
            for _ in 0..count.max(1) {
                let w = Worker::new_with_options(&url, &options)?;
                w.set_onmessage(Some(listen.as_ref().unchecked_ref()));
                w.post_message(&hello)?;
                workers.push((w, 0));
            }
            Url::revoke_object_url(&url)?;
            Ok(Workers { workers, done, _listen: listen, sent: Default::default() })
        }

        /// As many workers as the device has cores, less the page's own,
        /// up to four.
        pub fn for_this_device(glue_url: &str) -> Result<Workers, JsValue> {
            let n = web_sys::window().map_or(2.0, |w| w.navigator().hardware_concurrency()) as usize;
            Self::new(glue_url, n.saturating_sub(1).clamp(1, 4))
        }

        pub fn len(&self) -> usize {
            self.workers.len()
        }

        pub fn is_empty(&self) -> bool {
            self.workers.is_empty()
        }
    }

    impl JobRunner for Workers {
        fn submit(&mut self, id: u64, job: Job, input: Vec<u8>) {
            let Some(i) = (0..self.workers.len()).min_by_key(|&i| self.workers[i].1) else { return };
            let msg = js_sys::Object::new();
            let bytes = js_sys::Uint8Array::from(&input[..]);
            let set = |k: &str, v: &JsValue| js_sys::Reflect::set(&msg, &JsValue::from_str(k), v);
            let _ = set("id", &JsValue::from_f64(id as f64));
            let _ = set("job", &JsValue::from_f64(job as usize as f64));
            let _ = set("input", &bytes);
            let transfer = js_sys::Array::of1(&bytes.buffer());
            if self.workers[i].0.post_message_with_transfer(&msg, &transfer).is_ok() {
                self.workers[i].1 += 1;
                self.sent.insert(id, i);
            }
        }

        fn take_done(&mut self) -> Vec<(u64, Vec<u8>)> {
            let done = std::mem::take(&mut *self.done.borrow_mut());
            done.into_iter()
                .map(|(id, out)| {
                    if let Some(i) = self.sent.remove(&id) {
                        self.workers[i].1 -= 1;
                    }
                    (id, out)
                })
                .collect()
        }
    }

    impl Drop for Workers {
        fn drop(&mut self) {
            for (w, _) in &self.workers {
                w.terminate();
            }
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn doubled(input: &[u8]) -> Vec<u8> {
        input.iter().map(|b| b * 2).collect()
    }

    #[test]
    fn without_a_runner_a_job_runs_on_the_spot() {
        let mut ui = Ui::new();
        let got = Rc::new(RefCell::new(None));
        let g = got.clone();
        ui.spawn_job(doubled, vec![1, 2], move |_, out| *g.borrow_mut() = Some(out));
        assert_eq!(got.borrow().as_deref(), Some(&[2u8, 4][..]));
        assert!(!ui.jobs_pending());
    }

    #[test]
    fn threads_hand_each_output_to_its_own_callback() {
        let mut ui = Ui::new();
        ui.set_job_runner(Threads::new(3));
        let got = Rc::new(RefCell::new(Vec::new()));
        for i in 0..20u8 {
            let g = got.clone();
            ui.spawn_job(doubled, vec![i], move |_, out| g.borrow_mut().push((i, out[0])));
        }
        assert!(ui.jobs_pending());
        assert!(ui.needs_frame(), "frames go on while a job runs");
        let start = std::time::Instant::now();
        while ui.jobs_pending() {
            ui.poll_jobs();
            assert!(start.elapsed().as_secs() < 10, "jobs never came back");
        }
        let mut got = got.borrow().clone();
        got.sort();
        assert_eq!(got, (0..20u8).map(|i| (i, i * 2)).collect::<Vec<_>>());
    }

    fn settle(ui: &mut Ui) {
        let start = std::time::Instant::now();
        while ui.jobs_pending() {
            ui.poll_jobs();
            assert!(start.elapsed().as_secs() < 10, "jobs never came back");
        }
        // Outputs of cancelled jobs land too; give them time to.
        std::thread::sleep(std::time::Duration::from_millis(50));
        ui.poll_jobs();
    }

    #[test]
    fn a_job_for_a_removed_node_or_cancelled_or_superseded_does_not_call_back() {
        let mut ui = Ui::new();
        ui.set_job_runner(Threads::new(2));
        let got = Rc::new(RefCell::new(Vec::new()));
        let root = ui.root();
        let view = ui.add_box(root, crate::style! { width: 10px; });
        let g = got.clone();
        ui.spawn_job_for(view, doubled, vec![1], move |_, out| g.borrow_mut().push(("owned", out[0])));
        ui.remove(view);
        let g = got.clone();
        let id = ui.spawn_job(doubled, vec![2], move |_, out| g.borrow_mut().push(("cancelled", out[0])));
        assert!(ui.cancel_job(id));
        for n in 1..=5u8 {
            let g = got.clone();
            ui.spawn_latest("search", doubled, vec![n], move |_, out| g.borrow_mut().push(("search", out[0])));
        }
        settle(&mut ui);
        assert_eq!(*got.borrow(), [("search", 10)]);
        // Without a runner, the same rules.
        let mut here = Ui::new();
        let root = here.root();
        let gone = here.add_box(root, crate::style! { width: 10px; });
        here.remove(gone);
        let g = got.clone();
        here.spawn_job_for(gone, doubled, vec![3], move |_, out| g.borrow_mut().push(("gone", out[0])));
        assert_eq!(got.borrow().len(), 1);
    }
}
