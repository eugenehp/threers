//! 3D views drawn on a thread of their own, at rates of their own, and shown
//! in the UI, which draws on another thread at its own rate.
//!
//! [`ViewProducer`] renders scenes into textures it owns, on any thread.
//! [`ViewFrames`], the other half of each view, stays with whoever composites
//! the window. It takes the newest finished frame whenever it draws, for
//! example as a UI surface node through
//! [`UiPass::set_surface`](super::UiPass::set_surface). Neither waits for the
//! other. A slow scene leaves the UI showing the last frame it got, and a slow
//! UI just skips the frames it never looks at.
//!
//! Most apps want [`RenderThread`](super::RenderThread), which runs a
//! producer's loop for you: frame rates, pausing, drawing on demand, changes
//! to the scene sent from the UI, and a panic reported rather than a frozen
//! view.
//!
//! # Several views
//!
//! A producer can have several views ([`ViewProducer::add_view`]), each with
//! its own size, render scale and [`ViewFrames`]. They share one renderer, so
//! a mesh is uploaded once and a pipeline built once for all of them, and each
//! view draws into multisampled targets of its own size without the others
//! reallocating them. Views of one scene (a CAD app's top, front, side and
//! perspective) belong on one producer. Separate producers on separate threads
//! pay off only for separate scenes, since every view shares one GPU anyway.
//!
//! # Handing frames over
//!
//! Each view has four textures: one being drawn, one finished and waiting,
//! and two on the UI's side (the one it shows and the one it showed before).
//! They change hands with a single atomic swap. Both sides submit to the same
//! queue, and a frame is published only after its commands are submitted, so
//! anything the UI submits afterwards runs on the GPU after the frame is
//! drawn. A texture goes back to the producer only two of the UI's frames
//! after it last showed, so even a UI that asks for the next frame before
//! submitting the last one never has a texture drawn over while the GPU
//! reads it.
//!
//! Each texture has its own size. When the box in the UI changes size, the UI
//! calls [`ViewFrames::set_size`], and the producer draws its next frame at the
//! new size. The UI keeps showing the last frame, stretched, until that one
//! arrives, so a resize never makes the UI wait.
//!
//! ```ignore
//! let gpu = window.shared_gpu()?;
//! let (mut producer, mut frames) = ViewProducer::on(&gpu, ViewOptions::default());
//! std::thread::spawn(move || {
//!     let mut pacer = FramePacer::new(30.0);
//!     // Draws only when asked, at most 30 times a second.
//!     while producer.wait_for_work() {
//!         pacer.wait();
//!         producer.render(&mut scene, &camera);
//!     }
//! });
//! // Each UI frame, after laying out:
//! frames.set_size(box_width_px, box_height_px);
//! if let Some(frame) = frames.latest() {
//!     if frame.new { ui_pass.set_surface(VIEW_ID, frame.view.clone()); }
//! }
//! ```
//!
//! A view can draw fewer pixels than its box has: a fixed
//! [`ViewMut::set_render_scale`], or [`AdaptiveScale`], which lowers the
//! scale while frames take longer than a budget and raises it again when they
//! don't. The UI stretches the frame to the box either way. Each view's frames
//! are timed on the GPU on their own, where the adapter can.
//!
//! The UI tells the producer what is worth drawing: a view whose box is
//! scrolled away or hidden is [not visible](ViewFrames::set_visible) and is
//! not drawn, and a view's [priority](ViewFrames::set_priority) decides which
//! gives up resolution first under a shared budget.
//!
//! Input goes the other way. The UI maps a pointer to the view
//! ([`SurfaceView::point`](crate::ui::surface_view::SurfaceView::point),
//! with the `ui` feature), sends it to the render thread, and asks for a
//! frame.
//!
//! [`SubmitGate`] lets a surface reconfigure while other threads submit to its
//! device. `wgpu::Surface::configure` needs the queue empty and fails if
//! another thread submits while it runs. Frames hold the gate shared and a
//! reconfigure holds it exclusively. [`SharedGpu`] carries the device, queue
//! and gate together: get it from the window
//! ([`WindowRenderer::shared_gpu`](super::WindowRenderer::shared_gpu)) or
//! [`SurfaceGpu::shared_gpu`](super::SurfaceGpu::shared_gpu). A reconfigure (a
//! window resize, a present-mode change) waits for the producer's frame in
//! progress, the only time the UI waits on a 3D view.
//!
//! [`FramePacer`] limits a loop to a frame rate, either by sleeping (on a
//! thread) or by saying whether a frame is due (in an event loop, and in the
//! browser, where `requestAnimationFrame` sets the beat).

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Duration;

use web_time::Instant;

use super::gpu_timer::GpuTimer;
use super::{RenderTarget, Renderer, RendererOptions};
use crate::cameras::Camera;
use crate::scene::Scene;

/// Lets a surface reconfigure while other threads submit to its device:
/// frames hold it shared, a reconfigure exclusively. Clones are the same gate.
#[derive(Clone, Default, Debug)]
pub struct SubmitGate(pub(crate) Arc<RwLock<()>>);

impl SubmitGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Hold while recording and submitting a frame. Other frames may run at
    /// the same time.
    pub fn frame(&self) -> RwLockReadGuard<'_, ()> {
        self.0.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Hold while reconfiguring a surface. Waits for frames in progress to
    /// finish and keeps new ones out.
    pub fn reconfigure(&self) -> RwLockWriteGuard<'_, ()> {
        self.0.write().unwrap_or_else(|e| e.into_inner())
    }
}

/// A device and queue to draw views on, with the [`SubmitGate`] of the
/// surfaces on that device. Clones share all three.
#[derive(Clone, Debug)]
pub struct SharedGpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub gate: SubmitGate,
}

impl SharedGpu {
    /// `device` and `queue` with a gate of their own: for a device no surface
    /// is reconfigured on (offscreen, tests), or to share the gate with
    /// surfaces you make on it later.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            device,
            queue,
            gate: SubmitGate::new(),
        }
    }
}

impl From<&crate::gpu::GpuContext> for SharedGpu {
    fn from(gpu: &crate::gpu::GpuContext) -> Self {
        Self::new(gpu.device.clone(), gpu.queue.clone())
    }
}

/// How a [`ViewProducer`] draws.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ViewOptions {
    /// The frames' format. sRGB (the default, `Rgba8UnormSrgb`) when a
    /// [`UiPass`](super::UiPass) composites them, so they are filtered and
    /// blended as light.
    pub format: wgpu::TextureFormat,
    /// 1 (off) or 4 (the default).
    pub msaa: u32,
    pub renderer: RendererOptions,
}

impl Default for ViewOptions {
    fn default() -> Self {
        Self {
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            msaa: 4,
            renderer: RendererOptions::default(),
        }
    }
}

impl ViewOptions {
    pub fn format(mut self, format: wgpu::TextureFormat) -> Self {
        self.format = format;
        self
    }

    pub fn msaa(mut self, samples: u32) -> Self {
        self.msaa = samples;
        self
    }

    pub fn renderer(mut self, options: RendererOptions) -> Self {
        self.renderer = options;
        self
    }
}

/// One of a view's four textures, as a render target (its own depth, and
/// multisampled when the producer is).
struct Slot {
    target: RenderTarget,
    size: (u32, u32),
    /// The render scale it was drawn at.
    scale: f32,
    /// Which frame it holds, from 1; 0 for none yet.
    frame: u64,
}

/// Bit set in [`Shared::ready`] while the ready slot holds a frame the
/// consumer has not taken.
const FRESH: u8 = 0b100;
const INDEX: u8 = 0b011;

/// What a producer and all its views share: the wake-up, and how the
/// producer ended.
struct Signal {
    /// Whether anything has asked for work since the producer last waited.
    dirty: Mutex<bool>,
    #[cfg(not(target_arch = "wasm32"))]
    wake: std::sync::Condvar,
    /// The producer has been dropped.
    gone: AtomicBool,
    /// Why the producer stopped, when it failed.
    failure: Mutex<Option<String>>,
    /// What publish times are measured from.
    epoch: Instant,
}

impl Signal {
    fn ask(&self) {
        *self.dirty.lock().unwrap_or_else(|e| e.into_inner()) = true;
        #[cfg(not(target_arch = "wasm32"))]
        self.wake.notify_all();
    }

    fn since_epoch(&self) -> u64 {
        Instant::now().duration_since(self.epoch).as_nanos() as u64
    }
}

/// One view's state, shared by the producer and its [`ViewFrames`].
struct Shared {
    signal: Arc<Signal>,
    /// Each slot is touched only by the side that owns it, as the swap in
    /// `ready` decides. The mutexes are never contended and are there so
    /// neither side needs `unsafe` code.
    slots: [Mutex<Option<Slot>>; 4],
    /// The index of the ready slot, plus [`FRESH`].
    ready: AtomicU8,
    /// The size the consumer wants, as `width << 32 | height`; 0 until set.
    wanted: AtomicU64,
    /// This view has asked for a frame since it last drew.
    asked: AtomicBool,
    visible: AtomicBool,
    priority: AtomicI32,
    /// The [`ViewFrames`] has been dropped.
    consumer_gone: AtomicBool,
    /// Frames published so far, and when the last was (ns since the epoch).
    published: AtomicU64,
    published_at: AtomicU64,
    /// The last frame's cost in µs and render scale, for [`ViewStats`].
    cost_us: AtomicU32,
    scale_bits: AtomicU32,
    /// Called after each frame is published.
    on_frame: Mutex<Option<OnFrame>>,
}

#[cfg(not(target_arch = "wasm32"))]
type OnFrame = Arc<dyn Fn() + Send + Sync>;
#[cfg(target_arch = "wasm32")]
type OnFrame = Arc<dyn Fn()>;

impl Shared {
    fn new(signal: Arc<Signal>) -> Self {
        Self {
            signal,
            slots: Default::default(),
            // Slot 0 is the producer's, 1 the ready one, 2 and 3 the consumer's.
            ready: AtomicU8::new(1),
            wanted: AtomicU64::new(0),
            asked: AtomicBool::new(false),
            visible: AtomicBool::new(true),
            priority: AtomicI32::new(0),
            consumer_gone: AtomicBool::new(false),
            published: AtomicU64::new(0),
            published_at: AtomicU64::new(0),
            cost_us: AtomicU32::new(0),
            scale_bits: AtomicU32::new(1.0f32.to_bits()),
            on_frame: Mutex::new(None),
        }
    }

    fn slot(&self, i: u8) -> std::sync::MutexGuard<'_, Option<Slot>> {
        self.slots[i as usize]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn ask(&self) {
        self.asked.store(true, Ordering::Release);
        self.signal.ask();
    }
}

/// Which of a producer's views: [`ViewId::PRIMARY`] for the one
/// [`ViewProducer::new`] makes, others from [`ViewProducer::add_view`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ViewId(pub(crate) usize);

impl ViewId {
    pub const PRIMARY: ViewId = ViewId(0);

    /// Its position among the producer's views, from 0.
    pub fn index(self) -> usize {
        self.0
    }
}

/// The producer's side of one view.
struct Output {
    shared: Arc<Shared>,
    /// The slot being drawn into.
    back: u8,
    frame: u64,
    scale: f32,
    adaptive: Option<AdaptiveScale>,
    /// This view's main-pass and shadow timers, swapped into the renderer
    /// while it draws.
    timers: (Option<GpuTimer>, Option<GpuTimer>),
    /// CPU milliseconds the last draw took, for when the GPU can't time it.
    cpu_ms: f32,
}

impl Output {
    fn new(shared: Arc<Shared>, timers: (Option<GpuTimer>, Option<GpuTimer>)) -> Self {
        Self {
            shared,
            back: 0,
            frame: 0,
            scale: 1.0,
            adaptive: None,
            timers,
            cpu_ms: 0.0,
        }
    }

    fn cost_ms(&self) -> f32 {
        let gpu = self.timers.0.as_ref().map_or(0.0, |t| t.last_ms())
            + self.timers.1.as_ref().map_or(0.0, |t| t.last_ms());
        if gpu > 0.0 { gpu } else { self.cpu_ms }
    }

    fn live(&self) -> bool {
        !self.shared.consumer_gone.load(Ordering::Acquire)
    }
}

/// Draws scenes into textures for [`ViewFrames`] to show, from any thread.
/// See the [module docs](self).
pub struct ViewProducer {
    renderer: Renderer,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    format: wgpu::TextureFormat,
    signal: Arc<Signal>,
    gate: Option<SubmitGate>,
    views: Vec<Output>,
    /// Submissions of the last frames, oldest first, while the GPU may still
    /// be on them.
    in_flight: std::collections::VecDeque<wgpu::SubmissionIndex>,
    max_in_flight: usize,
}

/// The consumer side of a view: the newest finished frame, and what the
/// producer should draw. See the [module docs](self).
pub struct ViewFrames {
    shared: Arc<Shared>,
    /// The slot on screen, and the one shown before it.
    front: u8,
    retiring: u8,
    /// Frames taken, and (when, published count) samples for the frame rate.
    taken: u64,
    rate: std::collections::VecDeque<(Instant, u64)>,
    /// Publish-to-take latency of taken frames, smoothed, in ms.
    latency_ms: f32,
}

/// A frame from [`ViewFrames::latest`].
#[derive(Clone, Debug)]
pub struct ViewFrame {
    pub view: wgpu::TextureView,
    pub texture: wgpu::Texture,
    /// Its size in texels: the size last passed to [`ViewFrames::set_size`]
    /// times [`scale`](Self::scale), or an earlier size while a resize is in
    /// progress.
    pub size: (u32, u32),
    /// The render scale it was drawn at (see [`ViewMut::set_render_scale`]);
    /// 1 for full resolution.
    pub scale: f32,
    /// The frame's number, from 1, counting every frame the view drew.
    pub frame: u64,
    /// Whether the frame is newer than the last call returned. Pass it to the
    /// compositor again when it is, e.g. with
    /// [`UiPass::set_surface`](super::UiPass::set_surface).
    pub new: bool,
}

/// How a view is doing, from [`ViewFrames::stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ViewStats {
    /// Frames published a second, over about the last second.
    pub fps: f32,
    /// What the last frame cost, in ms: GPU time where the adapter can time
    /// it, else the CPU time to record it.
    pub frame_ms: f32,
    /// The render scale of the last frame.
    pub scale: f32,
    pub published: u64,
    /// Frames the consumer took. `published - shown` were drawn and never
    /// shown: the producer drew faster than the UI looked.
    pub shown: u64,
    /// From a frame's publishing to the consumer taking it, smoothed, in ms.
    pub latency_ms: f32,
}

impl ViewProducer {
    /// A producer drawing in `format` on `device`, and the [`ViewFrames`] of
    /// its first view. `gate` is the gate of any surface on the same device.
    /// Pass `None` if no surface on it is ever reconfigured.
    /// [`on`](Self::on) takes the same from a [`SharedGpu`].
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        format: wgpu::TextureFormat,
        options: RendererOptions,
        gate: Option<SubmitGate>,
    ) -> (Self, ViewFrames) {
        let mut renderer =
            Renderer::with_options(device.clone(), queue.clone(), format, 1, 1, options);
        let signal = Arc::new(Signal {
            dirty: Mutex::new(false),
            #[cfg(not(target_arch = "wasm32"))]
            wake: std::sync::Condvar::new(),
            gone: AtomicBool::new(false),
            failure: Mutex::new(None),
            epoch: Instant::now(),
        });
        // The first view times with the renderer's own timers.
        let mut timers = (None, None);
        renderer.swap_gpu_timers(&mut timers);
        let shared = Arc::new(Shared::new(signal.clone()));
        let producer = Self {
            renderer,
            device,
            queue,
            format,
            signal,
            gate,
            views: vec![Output::new(shared.clone(), timers)],
            in_flight: Default::default(),
            max_in_flight: 1,
        };
        (producer, ViewFrames::new(shared))
    }

    /// A producer on `gpu`, drawing as `options` says, and the [`ViewFrames`]
    /// of its first view.
    pub fn on(gpu: &SharedGpu, options: ViewOptions) -> (Self, ViewFrames) {
        let (mut producer, frames) = Self::new(
            gpu.device.clone(),
            gpu.queue.clone(),
            options.format,
            options.renderer,
            Some(gpu.gate.clone()),
        );
        producer.renderer.set_msaa(options.msaa);
        (producer, frames)
    }

    /// Another view, drawn by this producer's renderer, and its
    /// [`ViewFrames`].
    pub fn add_view(&mut self) -> (ViewId, ViewFrames) {
        let shared = Arc::new(Shared::new(self.signal.clone()));
        let timers = (
            GpuTimer::new(&self.device, &self.queue),
            GpuTimer::new(&self.device, &self.queue),
        );
        self.views.push(Output::new(shared.clone(), timers));
        (ViewId(self.views.len() - 1), ViewFrames::new(shared))
    }

    /// The views whose [`ViewFrames`] is still alive, in the order they were
    /// made.
    pub fn view_ids(&self) -> impl Iterator<Item = ViewId> + '_ {
        self.views
            .iter()
            .enumerate()
            .filter(|(_, v)| v.live())
            .map(|(i, _)| ViewId(i))
    }

    /// One view, to draw it or change how it draws.
    ///
    /// # Panics
    /// When `id` is not one of this producer's.
    pub fn view(&mut self, id: ViewId) -> ViewMut<'_> {
        assert!(id.0 < self.views.len(), "{id:?} is not a view of this producer");
        ViewMut { producer: self, id }
    }

    /// The renderer, for its settings (MSAA, tone mapping, post-processing).
    /// One renderer draws every view.
    pub fn renderer(&mut self) -> &mut Renderer {
        &mut self.renderer
    }

    /// Builds the pipelines `scene` needs and uploads its meshes and
    /// textures, before the first frame, so that frame doesn't stall on them.
    /// Until a view has a frame, [`ViewFrames::has_frame`] is false and the UI
    /// can show something else. Native only: it waits for the GPU, which a
    /// browser can't.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn warm_up(&mut self, scene: &mut Scene, camera: &dyn Camera) {
        let _frame = self.gate.as_ref().map(SubmitGate::frame);
        self.renderer.warm_scene(scene, camera);
    }

    /// How many frames may be queued on the GPU, unfinished, when the
    /// producer starts another (at least 1). It is 1 by default.
    ///
    /// Without a limit, a producer that doesn't pace itself queues frames
    /// far faster than the GPU runs them, and since the UI shares the queue,
    /// its frames wait behind all of them: seconds, measured. With 1, a UI
    /// frame waits behind at most one 3D frame on the GPU. 2 lets the
    /// producer record a frame while the GPU runs the last one, for more 3D
    /// frames and up to two ahead of the UI. Native only; in the browser the
    /// page's frame loop paces the producer.
    pub fn set_max_frames_in_flight(&mut self, frames: usize) {
        self.max_in_flight = frames.max(1);
    }

    /// How many views the producer has made, live or not.
    pub fn view_count(&self) -> usize {
        self.views.len()
    }

    /// A function that wakes a producer waiting in
    /// [`wait_for_work`](Self::wait_for_work), from any thread, without asking
    /// any view for a frame: for a loop with work of its own to look at
    /// (commands from the UI).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn waker(&self) -> Arc<dyn Fn() + Send + Sync> {
        let signal = self.signal.clone();
        Arc::new(move || signal.ask())
    }

    /// Whether every view's [`ViewFrames`] has been dropped, meaning nobody
    /// shows the frames any more.
    pub fn is_closed(&self) -> bool {
        self.views.iter().all(|v| !v.live())
    }

    /// Marks a frame as wanted on every view. Call it from the thread that
    /// changes the scene. [`ViewFrames::request_redraw`] does the same for
    /// one view from the UI side.
    pub fn request_redraw(&self) {
        for v in &self.views {
            v.shared.ask();
        }
    }

    /// Whether `id` has asked for a frame since it last drew, and clears the
    /// request.
    pub fn take_request(&self, id: ViewId) -> bool {
        self.views[id.0].shared.asked.swap(false, Ordering::AcqRel)
    }

    /// Whether `id` has asked for a frame since it last drew.
    pub fn has_request(&self, id: ViewId) -> bool {
        self.views[id.0].shared.asked.load(Ordering::Acquire)
    }

    /// Blocks until a frame is wanted (a redraw was requested or a size
    /// changed, on any view). Returns `false` once every [`ViewFrames`] is
    /// dropped, so a render thread can loop `while producer.wait_for_work()`.
    /// Skip it to draw continuously.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_for_work(&self) -> bool {
        self.wait_for_work_timeout(None)
    }

    /// [`wait_for_work`](Self::wait_for_work), giving up after `timeout`. It
    /// returns `true` then as well, so the caller can check its own state
    /// (for example, whether an animation is running).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_for_work_timeout(&self, timeout: Option<Duration>) -> bool {
        let mut dirty = self.signal.dirty.lock().unwrap_or_else(|e| e.into_inner());
        let deadline = timeout.map(|t| Instant::now() + t);
        while !*dirty && !self.is_closed() {
            match deadline {
                None => {
                    dirty = self.signal.wake.wait(dirty).unwrap_or_else(|e| e.into_inner());
                }
                Some(d) => {
                    let left = d.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        break;
                    }
                    dirty = self
                        .signal
                        .wake
                        .wait_timeout(dirty, left)
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                }
            }
        }
        *dirty = false;
        !self.is_closed()
    }

    /// Records why the producer is stopping, for [`ViewFrames::failure`].
    /// [`RenderThread`](super::RenderThread) does this with a panic's message.
    pub fn fail(&self, why: impl Into<String>) {
        let mut failure = self.signal.failure.lock().unwrap_or_else(|e| e.into_inner());
        failure.get_or_insert_with(|| why.into());
    }

    // The primary view, as before there were several.

    /// [`ViewMut::render`] on the primary view.
    pub fn render(&mut self, scene: &mut Scene, camera: &dyn Camera) -> bool {
        self.view(ViewId::PRIMARY).render(scene, camera)
    }

    /// [`ViewMut::render_with`] on the primary view.
    pub fn render_with(
        &mut self,
        draw: impl FnOnce(&mut Renderer, &wgpu::TextureView, (u32, u32)),
    ) -> bool {
        self.view(ViewId::PRIMARY).render_with(draw)
    }

    /// [`ViewMut::size`] of the primary view.
    pub fn size(&self) -> Option<(u32, u32)> {
        unpack(self.views[0].shared.wanted.load(Ordering::Acquire))
    }

    /// [`ViewMut::aspect`] of the primary view.
    pub fn aspect(&self) -> f32 {
        aspect(self.size())
    }

    /// [`ViewMut::set_render_scale`] on the primary view.
    pub fn set_render_scale(&mut self, scale: f32) {
        self.view(ViewId::PRIMARY).set_render_scale(scale);
    }

    /// [`ViewMut::render_scale`] of the primary view.
    pub fn render_scale(&self) -> f32 {
        self.views[0].scale
    }

    /// [`ViewMut::set_adaptive_scale`] on the primary view.
    pub fn set_adaptive_scale(&mut self, adaptive: Option<AdaptiveScale>) {
        self.view(ViewId::PRIMARY).set_adaptive_scale(adaptive);
    }

    /// [`ViewMut::frame_ms`] of the primary view.
    pub fn frame_ms(&self) -> f32 {
        self.views[0].cost_ms()
    }

    /// Waits until fewer than `max_in_flight` frames are unfinished on the GPU.
    fn throttle(&mut self) {
        while self.in_flight.len() >= self.max_in_flight {
            let oldest = self.in_flight.pop_front().expect("non-empty");
            #[cfg(not(target_arch = "wasm32"))]
            if let Err(e) = self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(oldest),
                timeout: None,
            }) {
                log::warn!("view producer: waiting for the GPU: {e}");
            }
            #[cfg(target_arch = "wasm32")]
            drop(oldest);
        }
    }

    fn make_slot(&self, size: (u32, u32), scale: f32) -> Slot {
        let target = RenderTarget::new(&self.device, size.0, size.1, self.format);
        // It is the picture: tone-mapped as a canvas is.
        target.set_presents(true);
        target.set_samples(self.renderer.msaa());
        Slot {
            target,
            size,
            scale,
            frame: 0,
        }
    }
}

impl Drop for ViewProducer {
    fn drop(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        if std::thread::panicking() {
            self.fail("the render thread panicked");
        }
        self.signal.gone.store(true, Ordering::Release);
        self.signal.ask();
    }
}

/// One view of a [`ViewProducer`], from [`ViewProducer::view`].
pub struct ViewMut<'a> {
    producer: &'a mut ViewProducer,
    id: ViewId,
}

impl ViewMut<'_> {
    pub fn id(&self) -> ViewId {
        self.id
    }

    fn output(&self) -> &Output {
        &self.producer.views[self.id.0]
    }

    fn output_mut(&mut self) -> &mut Output {
        &mut self.producer.views[self.id.0]
    }

    /// The size the consumer last asked for, in texels, or `None` before it
    /// has asked. Frames are drawn at this times the
    /// [`render_scale`](Self::render_scale).
    pub fn size(&self) -> Option<(u32, u32)> {
        unpack(self.output().shared.wanted.load(Ordering::Acquire))
    }

    /// Width over height of [`size`](Self::size), for the camera; 1 before a
    /// size is known.
    pub fn aspect(&self) -> f32 {
        aspect(self.size())
    }

    /// Whether the consumer shows the view (see [`ViewFrames::set_visible`]).
    /// [`render`](Self::render) draws nothing while it doesn't.
    pub fn is_visible(&self) -> bool {
        self.output().shared.visible.load(Ordering::Acquire)
    }

    /// The consumer's [`ViewFrames::set_priority`].
    pub fn priority(&self) -> i32 {
        self.output().shared.priority.load(Ordering::Acquire)
    }

    /// Whether the [`ViewFrames`] is still alive.
    pub fn is_live(&self) -> bool {
        self.output().live()
    }

    /// Draws at `scale` times the size asked for (clamped to 1/16..=1), and
    /// the consumer stretches the frame to fill its box. Fewer pixels for a
    /// scene whose cost is in its pixels. Turns
    /// [`set_adaptive_scale`](Self::set_adaptive_scale) off.
    pub fn set_render_scale(&mut self, scale: f32) {
        let o = self.output_mut();
        o.adaptive = None;
        o.scale = clamp_scale(scale);
    }

    /// The render scale the next frame draws at.
    pub fn render_scale(&self) -> f32 {
        self.output().scale
    }

    /// Chooses the render scale each frame to keep this view's frames within
    /// a budget (see [`AdaptiveScale`]), or `None` to keep the scale where it
    /// is.
    pub fn set_adaptive_scale(&mut self, adaptive: Option<AdaptiveScale>) {
        self.output_mut().adaptive = adaptive;
    }

    /// What this view's last frames cost in milliseconds: GPU time in the
    /// main and shadow passes where the adapter can time them (a few frames
    /// old), and otherwise the CPU time of the last draw, which leaves out
    /// the GPU's work.
    pub fn frame_ms(&self) -> f32 {
        self.output().cost_ms()
    }

    /// Draws `scene` from `camera` and publishes the frame. Returns `false`,
    /// drawing nothing, before the consumer has set a size, while it is not
    /// [visible](Self::is_visible), or after it is dropped. Make the camera's
    /// aspect [`aspect`](Self::aspect).
    pub fn render(&mut self, scene: &mut Scene, camera: &dyn Camera) -> bool {
        self.draw(|renderer, target, _| renderer.render_to(scene, camera, target))
    }

    /// Like [`render`](Self::render), but the drawing is `draw(renderer,
    /// target, size)`, which may render several times or draw anything else
    /// into `target`, `size` texels. The renderer is sized to the target, so
    /// [`Renderer::render`] into it works. `draw` must submit its work before
    /// returning, as `render` does.
    pub fn render_with(
        &mut self,
        draw: impl FnOnce(&mut Renderer, &wgpu::TextureView, (u32, u32)),
    ) -> bool {
        self.draw(|renderer, target, size| {
            renderer.resize(size.0, size.1);
            draw(renderer, &target.color_view, size)
        })
    }

    fn draw(&mut self, draw: impl FnOnce(&mut Renderer, &RenderTarget, (u32, u32))) -> bool {
        let Some(wanted) = self.size() else {
            return false;
        };
        if !self.is_live() || !self.is_visible() {
            return false;
        }
        let p = &mut *self.producer;
        p.throttle();
        let o = &mut p.views[self.id.0];
        o.shared.asked.store(false, Ordering::Release);
        let cost = o.cost_ms();
        if let Some(adaptive) = &mut o.adaptive {
            o.scale = clamp_scale(adaptive.update(o.scale, cost));
        }
        let size = scaled(wanted, o.scale);
        let shared = o.shared.clone();
        let mut slot = shared.slot(o.back);
        let samples = p.renderer.msaa();
        if slot
            .as_ref()
            .is_none_or(|s| s.size != size || s.target.samples() != samples)
        {
            *slot = Some(p.make_slot(size, p.views[self.id.0].scale));
        }
        let slot_ref = slot.as_mut().expect("made above");
        let o = &mut p.views[self.id.0];
        p.renderer.swap_gpu_timers(&mut o.timers);
        let start = Instant::now();
        {
            let _frame = p.gate.as_ref().map(SubmitGate::frame);
            draw(&mut p.renderer, &slot_ref.target, size);
            // An empty submission orders after everything `draw` submitted:
            // when it is done, so is the frame.
            #[cfg(not(target_arch = "wasm32"))]
            p.in_flight.push_back(p.queue.submit([]));
        }
        let o = &mut p.views[self.id.0];
        o.cpu_ms = start.elapsed().as_secs_f32() * 1000.0;
        p.renderer.swap_gpu_timers(&mut o.timers);
        o.frame += 1;
        slot_ref.frame = o.frame;
        slot_ref.scale = o.scale;
        drop(slot);
        shared
            .cost_us
            .store((o.cost_ms() * 1000.0) as u32, Ordering::Relaxed);
        shared.scale_bits.store(o.scale.to_bits(), Ordering::Relaxed);
        // Release publishes the slot's contents. The submission above already
        // happened, so the consumer's later submissions are ordered after it.
        let old = shared.ready.swap(o.back | FRESH, Ordering::AcqRel);
        o.back = old & INDEX;
        shared
            .published_at
            .store(shared.signal.since_epoch(), Ordering::Release);
        shared.published.store(o.frame, Ordering::Release);
        let on_frame = shared
            .on_frame
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if let Some(f) = on_frame {
            f();
        }
        true
    }
}

impl ViewFrames {
    fn new(shared: Arc<Shared>) -> Self {
        Self {
            shared,
            front: 2,
            retiring: 3,
            taken: 0,
            rate: Default::default(),
            latency_ms: 0.0,
        }
    }

    /// The size, in texels, to draw at from now on, usually the box's laid-out
    /// size times the device pixel ratio. A change asks for a frame.
    pub fn set_size(&self, width: u32, height: u32) {
        let packed = pack(width.max(1), height.max(1));
        if self.shared.wanted.swap(packed, Ordering::AcqRel) != packed {
            self.shared.ask();
        }
    }

    /// Whether the view is shown at all: `false` for a box scrolled out of
    /// sight, clipped away or on a hidden page, and the producer draws
    /// nothing for it. Becoming visible again asks for a frame.
    pub fn set_visible(&self, visible: bool) {
        if self.shared.visible.swap(visible, Ordering::AcqRel) != visible && visible {
            self.shared.ask();
        }
    }

    pub fn is_visible(&self) -> bool {
        self.shared.visible.load(Ordering::Acquire)
    }

    /// How much the view matters next to the producer's others: under a
    /// shared GPU budget ([`RenderLoop::set_gpu_budget`](super::RenderLoop::set_gpu_budget)),
    /// lower priorities give up resolution first and draw last. 0 by default;
    /// give the view the user is working in more.
    pub fn set_priority(&self, priority: i32) {
        self.shared.priority.store(priority, Ordering::Release);
    }

    /// Calls `f` on the producer's thread each time a frame is published, so a
    /// UI that redraws only on demand can wake for it. With winit, pass a
    /// closure calling `window.request_redraw()`, which works from any thread.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn on_frame(&self, f: impl Fn() + Send + Sync + 'static) {
        *self.shared.on_frame.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(f));
    }

    /// Calls `f` each time a frame is published.
    #[cfg(target_arch = "wasm32")]
    pub fn on_frame(&self, f: impl Fn() + 'static) {
        *self.shared.on_frame.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(f));
    }

    /// Asks the producer for a frame, for example after a camera move.
    pub fn request_redraw(&self) {
        self.shared.ask();
    }

    /// Frames the producer has published so far.
    pub fn published(&self) -> u64 {
        self.shared.published.load(Ordering::Acquire)
    }

    /// Whether a frame has been published yet. Until then the UI may show a
    /// placeholder (a surface node with no frame draws nothing over its box's
    /// background).
    pub fn has_frame(&self) -> bool {
        self.published() > 0
    }

    /// Whether a frame newer than the last [`latest`](Self::latest) is
    /// waiting: a UI that draws only on change draws for it.
    pub fn has_new_frame(&self) -> bool {
        self.shared.ready.load(Ordering::Acquire) & FRESH != 0
    }

    /// Whether the producer has been dropped, so no new frames will come.
    pub fn is_closed(&self) -> bool {
        self.shared.signal.gone.load(Ordering::Acquire)
    }

    /// Why the producer stopped, when it failed: a panic's message from a
    /// [`RenderThread`](super::RenderThread), or "the render thread
    /// panicked" from any thread that dropped its producer while unwinding.
    pub fn failure(&self) -> Option<String> {
        self.shared
            .signal
            .failure
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// How the view is doing: frame rate, frame cost, render scale, frames
    /// never shown and latency. Rates are measured between calls to
    /// [`latest`](Self::latest).
    pub fn stats(&self) -> ViewStats {
        let published = self.published();
        let fps = match (self.rate.front(), self.rate.back()) {
            (Some(&(t0, n0)), Some(&(t1, n1))) if t1 > t0 => {
                (n1 - n0) as f32 / t1.duration_since(t0).as_secs_f32()
            }
            _ => 0.0,
        };
        ViewStats {
            fps,
            frame_ms: self.shared.cost_us.load(Ordering::Relaxed) as f32 / 1000.0,
            scale: f32::from_bits(self.shared.scale_bits.load(Ordering::Relaxed)),
            published,
            shown: self.taken,
            latency_ms: self.latency_ms,
        }
    }

    /// The newest finished frame, or `None` before the first. Call it once
    /// per UI frame, before recording that frame. The texture it returned two
    /// frames back goes back to the producer, so the UI must have submitted
    /// the frame that drew with it, which any UI that draws one frame after
    /// another has.
    pub fn latest(&mut self) -> Option<ViewFrame> {
        let now = Instant::now();
        let published = self.published();
        if self.rate.back().is_none_or(|&(_, n)| n != published) {
            self.rate.push_back((now, published));
        }
        while self
            .rate
            .front()
            .is_some_and(|&(t, _)| now.duration_since(t) > Duration::from_secs(1))
            && self.rate.len() > 2
        {
            self.rate.pop_front();
        }
        let mut new = false;
        if self.shared.ready.load(Ordering::Acquire) & FRESH != 0 {
            // Hand back the slot shown two frames ago; keep the last one
            // shown until the next swap.
            let old = self.shared.ready.swap(self.retiring, Ordering::AcqRel);
            self.retiring = self.front;
            self.front = old & INDEX;
            new = true;
            self.taken += 1;
            let at = self.shared.published_at.load(Ordering::Acquire);
            let age = self.shared.signal.since_epoch().saturating_sub(at) as f32 / 1e6;
            self.latency_ms = if self.taken == 1 { age } else { self.latency_ms + 0.2 * (age - self.latency_ms) };
        }
        let slot = self.shared.slot(self.front);
        let slot = slot.as_ref().filter(|s| s.frame > 0)?;
        Some(ViewFrame {
            view: slot.target.color_view.clone(),
            texture: slot.target.color_texture.clone(),
            size: slot.size,
            scale: slot.scale,
            frame: slot.frame,
            new,
        })
    }
}

impl Drop for ViewFrames {
    fn drop(&mut self) {
        self.shared.consumer_gone.store(true, Ordering::Release);
        self.shared.signal.ask();
    }
}

impl ViewFrame {
    /// The frame's pixels as tightly packed RGBA8, top row first. Blocks
    /// until the GPU has copied them: for tests and screenshots, not for
    /// every frame.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_rgba(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u8> {
        let (w, h) = (self.texture.width(), self.texture.height());
        let row = (w * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers view frame readback"),
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            self.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let index = queue.submit(Some(encoder.finish()));
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        });
        let data = buffer.slice(..).get_mapped_range().expect("mapped after the wait");
        data.chunks(row as usize)
            .flat_map(|r| r[..(w * 4) as usize].iter().copied())
            .collect()
    }
}

fn aspect(size: Option<(u32, u32)>) -> f32 {
    size.map_or(1.0, |(w, h)| w as f32 / h as f32)
}

fn clamp_scale(scale: f32) -> f32 {
    if scale.is_finite() { scale.clamp(1.0 / 16.0, 1.0) } else { 1.0 }
}

fn scaled((w, h): (u32, u32), scale: f32) -> (u32, u32) {
    let s = |v: u32| ((v as f32 * scale).round() as u32).max(1);
    (s(w), s(h))
}

/// Picks a view's render scale to keep its frames within a time budget: lower
/// while frames take longer, higher again once they are well under it.
///
/// A frame's cost is taken to grow with its pixels, the square of the scale,
/// so one step down goes most of the way. Going up is one step at a time,
/// and only when the bigger frame is predicted to stay under
/// [`margin`](Self::margin) of the budget. Scales are multiples of
/// [`step`](Self::step), so a view switches between a few sizes rather than
/// remaking its textures every frame. After a change it waits
/// [`settle`](Self::settle) frames before judging again, since the GPU's
/// timings arrive a few frames late.
#[derive(Clone, Debug)]
pub struct AdaptiveScale {
    /// Milliseconds a frame may take.
    pub budget_ms: f32,
    pub min: f32,
    pub max: f32,
    pub step: f32,
    pub settle: u32,
    /// The fraction of the budget a bigger frame must be predicted to fit in
    /// before the scale goes up.
    pub margin: f32,
    smoothed: Option<f32>,
    wait: u32,
}

impl AdaptiveScale {
    /// Keep frames within `budget_ms`: scales from 1/4 to 1 in steps of 1/8.
    pub fn new(budget_ms: f32) -> Self {
        Self {
            budget_ms,
            min: 0.25,
            max: 1.0,
            step: 0.125,
            settle: 8,
            margin: 0.9,
            smoothed: None,
            wait: 0,
        }
    }

    /// Takes what a frame cost at `scale`, in milliseconds, and returns the
    /// scale for the next. A cost of 0 (not measured yet) changes nothing.
    pub fn update(&mut self, scale: f32, cost_ms: f32) -> f32 {
        if cost_ms <= 0.0 || !cost_ms.is_finite() {
            return scale;
        }
        let cost = self.smoothed.map_or(cost_ms, |m| m + 0.3 * (cost_ms - m));
        self.smoothed = Some(cost);
        if self.wait > 0 {
            self.wait -= 1;
            return scale;
        }
        let step = self.step.max(1.0 / 64.0);
        let next = if cost > self.budget_ms {
            let ideal = scale * (self.budget_ms / cost).sqrt();
            ((ideal / step).floor() * step).min(scale - step)
        } else {
            let up = scale + step;
            let predicted = cost * (up / scale).powi(2);
            if predicted <= self.margin * self.budget_ms { up } else { scale }
        };
        let next = next.clamp(self.min, self.max);
        if (next - scale).abs() > 1e-4 {
            self.wait = self.settle;
            self.smoothed = None;
        }
        next
    }
}

fn pack(w: u32, h: u32) -> u64 {
    (w as u64) << 32 | h as u64
}

fn unpack(v: u64) -> Option<(u32, u32)> {
    (v != 0).then_some(((v >> 32) as u32, v as u32))
}

/// Holds a loop to a frame rate.
///
/// On a thread, [`wait`](Self::wait) sleeps until the next frame is due. In
/// an event loop, [`due`](Self::due) says whether to draw now and
/// [`next`](Self::next) gives the deadline to wake at, e.g. for winit's
/// `ControlFlow::WaitUntil`. Deadlines are spaced from one another rather than
/// from when each frame finished, so the rate holds on average. A loop that
/// falls more than a frame behind starts again from now instead of drawing a
/// burst of frames to catch up.
#[derive(Clone, Debug)]
pub struct FramePacer {
    interval: Option<Duration>,
    next: Instant,
}

impl FramePacer {
    /// At most `fps` frames a second. 0, a negative or a non-finite value means
    /// no limit.
    pub fn new(fps: f32) -> Self {
        let mut pacer = Self {
            interval: None,
            next: Instant::now(),
        };
        pacer.set_fps(fps);
        pacer
    }

    pub fn set_fps(&mut self, fps: f32) {
        self.interval =
            (fps.is_finite() && fps > 0.0).then(|| Duration::from_secs_f64(1.0 / fps as f64));
    }

    /// The limit in frames a second, `None` if unlimited.
    pub fn fps(&self) -> Option<f32> {
        self.interval.map(|i| (1.0 / i.as_secs_f64()) as f32)
    }

    /// When the next frame is due.
    pub fn next(&self) -> Instant {
        self.next
    }

    /// Whether a frame is due at `now`. If it is, the frame counts as drawn
    /// and the next deadline moves on by one interval.
    pub fn due(&mut self, now: Instant) -> bool {
        let Some(interval) = self.interval else {
            return true;
        };
        if now < self.next {
            return false;
        }
        self.next += interval;
        if self.next + interval < now {
            self.next = now + interval;
        }
        true
    }

    /// Sleeps until the next frame is due, then counts it as drawn.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(&mut self) {
        let now = Instant::now();
        if now < self.next {
            std::thread::sleep(self.next - now);
        }
        self.due(self.next.max(now));
    }
}

#[cfg(not(target_arch = "wasm32"))]
const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    const fn send<T: Send>() {}
    send_sync::<SubmitGate>();
    send_sync::<SharedGpu>();
    send::<ViewProducer>();
    send::<ViewFrames>();
    // The scene goes to the render thread with the producer.
    send::<Scene>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pacer_spaces_deadlines_and_resyncs_when_behind() {
        let mut p = FramePacer::new(10.0);
        let t0 = p.next();
        assert!(p.due(t0));
        assert!(!p.due(t0 + Duration::from_millis(50)));
        assert!(p.due(t0 + Duration::from_millis(100)));
        assert_eq!(p.next(), t0 + Duration::from_millis(200));
        // A second late: no burst of nine frames to catch up.
        let late = t0 + Duration::from_millis(1300);
        assert!(p.due(late));
        assert!(!p.due(late + Duration::from_millis(1)));
        assert_eq!(p.next(), late + Duration::from_millis(100));
    }

    #[test]
    fn unlimited_pacer_is_always_due() {
        let mut p = FramePacer::new(0.0);
        assert_eq!(p.fps(), None);
        let now = Instant::now();
        assert!(p.due(now) && p.due(now));
    }

    #[test]
    fn adaptive_scale_drops_when_over_budget_and_climbs_back() {
        let mut a = AdaptiveScale::new(16.0);
        a.settle = 0;
        // Four times over budget: half the pixels' side at once.
        let s = a.update(1.0, 64.0);
        assert_eq!(s, 0.5);
        // Under budget, but a step up would not be: stays.
        assert_eq!(a.update(s, 14.0), 0.5);
        // Well under: one step at a time.
        let mut a = AdaptiveScale::new(16.0);
        a.settle = 0;
        let mut s = 0.5;
        for want in [0.625, 0.75, 0.875, 1.0, 1.0] {
            s = a.update(s, 4.0);
            assert_eq!(s, want);
        }
        // Never below the minimum, and unmeasured frames change nothing.
        assert_eq!(a.update(0.25, 1000.0), 0.25);
        assert_eq!(a.update(0.75, 0.0), 0.75);
    }

    #[test]
    fn adaptive_scale_waits_for_late_timings_after_a_change() {
        let mut a = AdaptiveScale::new(16.0);
        assert_eq!(a.update(1.0, 40.0), 0.625);
        // The next few readings still describe the old scale.
        for _ in 0..a.settle {
            assert_eq!(a.update(0.625, 40.0), 0.625);
        }
        assert!(a.update(0.625, 40.0) < 0.625);
    }

    #[test]
    fn scaled_sizes_round_and_never_reach_zero() {
        assert_eq!(scaled((1001, 3), 0.5), (501, 2));
        assert_eq!(scaled((1, 1), 1.0 / 16.0), (1, 1));
        assert_eq!(clamp_scale(f32::NAN), 1.0);
        assert_eq!(clamp_scale(4.0), 1.0);
    }

    #[test]
    fn size_round_trips_through_packing() {
        assert_eq!(unpack(0), None);
        assert_eq!(unpack(pack(3840, 1)), Some((3840, 1)));
    }
}
