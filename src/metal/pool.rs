//! A fixed set of render threads, reused job after job: [`MetalRenderPool`].

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::ffi::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use super::device::{MetalDevice, MetalError};
use super::objc::AutoreleasePool;
use super::renderer::MetalRenderer;

/// The scheduling class of the pool's threads — Darwin's QoS, which on Apple
/// silicon also decides performance or efficiency cores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetalWorkerQos {
    /// Frames someone is watching. The default.
    UserInteractive,
    /// Work the user asked for and is waiting on: an export, a thumbnail.
    UserInitiated,
    /// Long-running work with a progress bar.
    Utility,
    /// Work nobody is waiting for; runs on efficiency cores.
    Background,
}

impl MetalWorkerQos {
    fn raw(self) -> u32 {
        // `qos_class_t` from <sys/qos.h>.
        match self {
            Self::UserInteractive => 0x21,
            Self::UserInitiated => 0x19,
            Self::Utility => 0x11,
            Self::Background => 0x09,
        }
    }
}

/// How many workers, how deep a queue, and at what priority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalRenderPoolConfig {
    /// Worker threads, each with its own renderer. At least 1.
    pub threads: usize,
    /// Jobs that may wait for a free worker before `submit` blocks and
    /// `try_submit` refuses. 0 means a job is only accepted when a worker is
    /// ready to take it.
    pub queue_depth: usize,
    /// The workers' scheduling class.
    pub qos: MetalWorkerQos,
    /// Thread name prefix; workers are `{name}-0`, `{name}-1`, ….
    pub name: String,
}

impl MetalRenderPoolConfig {
    /// Sized for the machine this runs on.
    ///
    /// One worker per performance core, capped at 2 on iOS, tvOS and visionOS
    /// and at 4 on macOS. Every worker shares one command queue and one GPU,
    /// so past a few threads encoding faster only queues more for the GPU —
    /// and on a phone or a headset it spends thermal budget the GPU wanted.
    /// The queue holds one job per worker: enough to keep each fed while the
    /// last result is collected, and no further ahead than that.
    pub fn for_this_device() -> Self {
        let cap = if cfg!(target_os = "macos") { 4 } else { 2 };
        let threads = performance_cores().clamp(1, cap);
        Self {
            threads,
            queue_depth: threads,
            qos: MetalWorkerQos::UserInteractive,
            name: "threers-metal".into(),
        }
    }

    /// Exactly `threads` workers, with a queue as deep.
    pub fn with_threads(threads: usize) -> Self {
        let threads = threads.max(1);
        Self { threads, queue_depth: threads, ..Self::for_this_device() }
    }
}

impl Default for MetalRenderPoolConfig {
    fn default() -> Self {
        Self::for_this_device()
    }
}

/// Performance cores (`hw.perflevel0.physicalcpu`), else every core, else 2.
///
/// Apple silicon on every platform reports its performance cluster first; the
/// efficiency cores behind it are the wrong place for a frame.
fn performance_cores() -> usize {
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> i32;
    }
    let mut n: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    let ok = unsafe {
        sysctlbyname(
            c"hw.perflevel0.physicalcpu".as_ptr(),
            (&mut n as *mut i32).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    } == 0;
    if ok && n > 0 {
        return n as usize;
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2)
}

/// What a job is handed: the worker's renderer, device, and its own state.
pub struct MetalWorker {
    index: usize,
    device: MetalDevice,
    renderer: Option<MetalRenderer>,
    locals: HashMap<TypeId, Box<dyn Any + Send>>,
}

impl MetalWorker {
    /// Which worker, `0..threads`.
    pub fn index(&self) -> usize {
        self.index
    }

    /// The device every worker shares.
    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    /// This worker's renderer, built on its first job and kept — with its
    /// pipeline, geometry and texture caches — for every job after.
    pub fn renderer(&mut self) -> Result<&mut MetalRenderer, MetalError> {
        if self.renderer.is_none() {
            self.renderer = Some(MetalRenderer::with_device(self.device.clone())?);
        }
        Ok(self.renderer.as_mut().expect("just built"))
    }

    /// A value of type `T` that belongs to this worker and outlives the job:
    /// a render target, a post chain, a staging buffer. Starts as
    /// `T::default()` on the worker's first ask.
    pub fn local<T: Default + Send + 'static>(&mut self) -> &mut T {
        self.locals
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(T::default()))
            .downcast_mut()
            .expect("keyed by its own TypeId")
    }
}

/// Why a job did not produce its value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalJobError {
    /// The job panicked; carries the message. The worker survives — its
    /// renderer is dropped and rebuilt for the next job, in case the panic
    /// left it half-way through a frame.
    Panicked(String),
    /// The queue is full ([`MetalRenderPool::try_submit`] only).
    QueueFull,
    /// The pool is shutting down.
    Closed,
}

impl std::fmt::Display for MetalJobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Panicked(m) => write!(f, "render job panicked: {m}"),
            Self::QueueFull => write!(f, "render pool queue is full"),
            Self::Closed => write!(f, "render pool is closed"),
        }
    }
}

impl std::error::Error for MetalJobError {}

/// A submitted job's result, to collect with [`wait`](Self::wait).
pub struct MetalJob<T> {
    rx: Receiver<Result<T, MetalJobError>>,
}

impl<T> MetalJob<T> {
    /// Block until the job has run.
    pub fn wait(self) -> Result<T, MetalJobError> {
        self.rx.recv().unwrap_or(Err(MetalJobError::Closed))
    }

    /// The result if the job has finished, else the job back.
    pub fn try_wait(self) -> Result<Result<T, MetalJobError>, Self> {
        match self.rx.try_recv() {
            Ok(r) => Ok(r),
            Err(mpsc::TryRecvError::Empty) => Err(self),
            Err(mpsc::TryRecvError::Disconnected) => Ok(Err(MetalJobError::Closed)),
        }
    }
}

type Task = Box<dyn FnOnce(&mut MetalWorker) + Send>;

/// A bounded set of render threads sharing one device, reused job after job.
///
/// Spawning a thread per frame, or a renderer per job, throws away the two
/// things that are expensive to rebuild: the thread and the renderer's caches
/// (pipelines, uploaded geometry and textures). [`MetalRenderPool`] keeps both.
/// Its workers live as long as the pool, each with its own
/// [`MetalRenderer`] on the shared [`MetalDevice`], and a job is a closure
/// handed whichever worker is free.
///
/// On a phone, a headset or a TV the point is also to *bound* the work: a
/// couple of performance cores and a thermal budget do not want eight encoding
/// threads. [`MetalRenderPoolConfig::for_this_device`] sizes the pool from the
/// performance cores, capped lower on iOS, tvOS and visionOS than on a Mac,
/// and the queue in front of it is bounded too, so a producer outrunning the
/// GPU waits ([`MetalRenderPool::submit`]) or is told to drop the frame
/// ([`MetalRenderPool::try_submit`]) rather than queueing work without end.
///
/// ```no_run
/// # #[cfg(feature = "metal")]
/// # fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// use threers::metal::{enums::pixel_format, MetalDevice, MetalRenderPool, MetalRenderPoolConfig, MetalRenderTarget};
/// use threers::{PerspectiveCamera, Scene};
///
/// let device = MetalDevice::new()?;
/// let pool = MetalRenderPool::new(&device, MetalRenderPoolConfig::for_this_device());
///
/// let job = pool.submit(|worker| -> Result<Vec<u8>, threers::metal::MetalError> {
///     // Per-worker state survives between jobs: the target is made once per
///     // worker, not once per frame.
///     let device = worker.device().clone();
///     let target: &mut Option<MetalRenderTarget> = worker.local();
///     if target.is_none() {
///         *target = Some(MetalRenderTarget::new(&device, 256, 256, pixel_format::RGBA8_UNORM_SRGB, 4)?);
///     }
///     let attachments = target.as_ref().unwrap().attachments();
///     let mut scene = Scene::new();
///     let camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
///     worker.renderer()?.render(&mut scene, &camera, &attachments)?;
///     let target: &mut Option<MetalRenderTarget> = worker.local();
///     target.as_ref().unwrap().read_rgba(&device)
/// })?;
/// let rgba = job.wait()??;
/// # Ok(()) }
/// ```
///
/// Dropping the pool lets queued jobs finish, then joins the workers; their
/// renderers are released on their own threads.
pub struct MetalRenderPool {
    tx: Option<SyncSender<Task>>,
    workers: Vec<JoinHandle<()>>,
    config: MetalRenderPoolConfig,
}

impl MetalRenderPool {
    /// Start `config.threads` workers on `device`.
    pub fn new(device: &MetalDevice, config: MetalRenderPoolConfig) -> Self {
        let config = MetalRenderPoolConfig { threads: config.threads.max(1), ..config };
        let (tx, rx) = mpsc::sync_channel::<Task>(config.queue_depth);
        let rx = Arc::new(Mutex::new(rx));
        let workers = (0..config.threads)
            .map(|index| {
                let rx = rx.clone();
                let device = device.clone();
                let qos = config.qos;
                std::thread::Builder::new()
                    .name(format!("{}-{index}", config.name))
                    .spawn(move || worker_main(index, device, qos, rx))
                    .expect("spawning a render worker")
            })
            .collect();
        Self { tx: Some(tx), workers, config }
    }

    /// The configuration the pool was started with.
    pub fn config(&self) -> &MetalRenderPoolConfig {
        &self.config
    }

    /// Queue `job` for the next free worker, waiting for room in the queue if
    /// it is full.
    pub fn submit<T, F>(&self, job: F) -> Result<MetalJob<T>, MetalJobError>
    where
        T: Send + 'static,
        F: FnOnce(&mut MetalWorker) -> T + Send + 'static,
    {
        let (task, handle) = package(job);
        self.tx
            .as_ref()
            .ok_or(MetalJobError::Closed)?
            .send(task)
            .map_err(|_| MetalJobError::Closed)?;
        Ok(handle)
    }

    /// As [`submit`](Self::submit), but [`MetalJobError::QueueFull`] instead
    /// of waiting — for a frame loop that would rather drop a frame than fall
    /// further behind.
    pub fn try_submit<T, F>(&self, job: F) -> Result<MetalJob<T>, MetalJobError>
    where
        T: Send + 'static,
        F: FnOnce(&mut MetalWorker) -> T + Send + 'static,
    {
        let (task, handle) = package(job);
        match self.tx.as_ref().ok_or(MetalJobError::Closed)?.try_send(task) {
            Ok(()) => Ok(handle),
            Err(TrySendError::Full(_)) => Err(MetalJobError::QueueFull),
            Err(TrySendError::Disconnected(_)) => Err(MetalJobError::Closed),
        }
    }
}

impl Drop for MetalRenderPool {
    fn drop(&mut self) {
        drop(self.tx.take());
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

impl std::fmt::Debug for MetalRenderPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalRenderPool").field("config", &self.config).finish()
    }
}

fn package<T, F>(job: F) -> (Task, MetalJob<T>)
where
    T: Send + 'static,
    F: FnOnce(&mut MetalWorker) -> T + Send + 'static,
{
    let (tx, rx) = mpsc::sync_channel(1);
    let task: Task = Box::new(move |worker: &mut MetalWorker| {
        let result = catch_unwind(AssertUnwindSafe(|| job(worker))).map_err(|p| {
            // A panic may have left the renderer mid-frame; start it afresh.
            worker.renderer = None;
            MetalJobError::Panicked(panic_message(&*p))
        });
        // The submitter may have stopped waiting; that is not the worker's
        // problem.
        let _ = tx.send(result);
    });
    (task, MetalJob { rx })
}

fn panic_message(p: &(dyn Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".into())
}

fn worker_main(
    index: usize,
    device: MetalDevice,
    qos: MetalWorkerQos,
    rx: Arc<Mutex<Receiver<Task>>>,
) {
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    unsafe { pthread_set_qos_class_self_np(qos.raw(), 0) };
    let mut worker = MetalWorker { index, device, renderer: None, locals: HashMap::new() };
    loop {
        // Hold the lock only to take a job, not to run it.
        let task = match rx.lock() {
            Ok(rx) => rx.recv(),
            Err(_) => return,
        };
        let Ok(task) = task else { break };
        // A pool per job: whatever the job autoreleases is gone when it ends,
        // not when the thread does.
        let _pool = AutoreleasePool::new();
        task(&mut worker);
    }
    // The renderer and the locals release Metal objects; do it in a pool too.
    let _pool = AutoreleasePool::new();
    drop(worker);
}
