//! The loop that draws a [`ViewProducer`]'s views, so an app writes what a
//! frame draws and nothing else.
//!
//! [`RenderLoop`] holds the app's render state `S` (a scene, cameras,
//! anything) and the function that draws one view from it. Each
//! [`step`](RenderLoop::step) draws the views that are due, in priority
//! order, and says when the next one is due. Every view has its own frame rate
//! ([`set_fps`](RenderLoop::set_fps)), draws continuously or only when asked
//! ([`Redraw`]), can be paused, and is skipped while the UI says it isn't
//! visible. Under a shared GPU budget ([`set_gpu_budget`](RenderLoop::set_gpu_budget))
//! the views' resolutions are traded against each other, lowest priority
//! first.
//!
//! [`RenderThread`] runs a loop on a thread of its own (native). The UI keeps
//! the handle: [`update`](RenderThread::update) changes the state on the
//! render thread and asks for frames, [`query`](RenderThread::query) asks it a
//! question (a pick, a bounding box), and a panic in the drawing ends the loop
//! and is reported through [`failure`](RenderThread::failure) and every
//! view's [`ViewFrames::failure`] rather than leaving a frozen picture. In the
//! browser there is no thread: call [`RenderLoop::step`] from the page's
//! animation frame, with the same drawing code.
//!
//! ```ignore
//! struct World { scene: Scene, camera: PerspectiveCamera, knot: ObjectId }
//!
//! let gpu = window.shared_gpu()?;
//! let (producer, frames) = ViewProducer::on(&gpu, ViewOptions::default());
//! let render = RenderThread::spawn(RenderLoop::new(producer, world, |w: &mut World, view| {
//!     w.camera.set_aspect(view.aspect());
//!     w.scene.get_mut(w.knot).unwrap().rotation.y += view.dt;
//!     view.render(&mut w.scene, &w.camera);
//! }));
//! render.set_fps(ViewId::PRIMARY, 30.0);
//! // From the UI, on a drag:
//! render.update(move |w| w.camera.position.x += dx);
//! ```

#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

use web_time::Instant;

use super::view_thread::{FramePacer, ViewFrames, ViewId, ViewMut, ViewProducer};
use super::Renderer;
use crate::cameras::Camera;
use crate::scene::Scene;

/// When a view draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Redraw {
    /// Every frame its rate allows: for a scene that moves by itself.
    #[default]
    Continuous,
    /// Only when asked: a resize, [`ViewFrames::request_redraw`],
    /// [`RenderThread::update`]. A still scene costs nothing.
    OnDemand,
}

#[cfg(not(target_arch = "wasm32"))]
type DrawFn<S> = Box<dyn FnMut(&mut S, &mut ViewDraw<'_>) + Send>;
#[cfg(target_arch = "wasm32")]
type DrawFn<S> = Box<dyn FnMut(&mut S, &mut ViewDraw<'_>)>;

/// One view's schedule.
struct Schedule {
    pacer: FramePacer,
    redraw: Redraw,
    paused: bool,
    last: Option<Instant>,
}

impl Default for Schedule {
    fn default() -> Self {
        Self {
            pacer: FramePacer::new(60.0),
            redraw: Redraw::Continuous,
            paused: false,
            last: None,
        }
    }
}

/// The view being drawn, handed to the draw function.
pub struct ViewDraw<'a> {
    view: ViewMut<'a>,
    /// Seconds since the loop started.
    pub time: f32,
    /// Seconds since this view last drew (0 the first time).
    pub dt: f32,
}

impl<'a> ViewDraw<'a> {
    pub fn id(&self) -> ViewId {
        self.view.id()
    }

    /// Width over height of the view, for its camera.
    pub fn aspect(&self) -> f32 {
        self.view.aspect()
    }

    /// The size the view is shown at, in texels.
    pub fn size(&self) -> Option<(u32, u32)> {
        self.view.size()
    }

    /// Draws `scene` from `camera` into the view and publishes the frame.
    pub fn render(&mut self, scene: &mut Scene, camera: &dyn Camera) -> bool {
        self.view.render(scene, camera)
    }

    /// See [`ViewMut::render_with`].
    pub fn render_with(
        &mut self,
        draw: impl FnOnce(&mut Renderer, &wgpu::TextureView, (u32, u32)),
    ) -> bool {
        self.view.render_with(draw)
    }

    /// The view itself, for its render scale and costs.
    pub fn view(&mut self) -> &mut ViewMut<'a> {
        &mut self.view
    }
}

/// A GPU time budget shared by a loop's views: see
/// [`RenderLoop::set_gpu_budget`].
#[derive(Clone, Debug)]
struct Budget {
    ms: f32,
    /// Rounds to wait after a change, for the GPU's timings to catch up.
    settle: u32,
    wait: u32,
}

/// The loop: render state, the function that draws a view, and each view's
/// schedule. See the [module docs](self).
pub struct RenderLoop<S> {
    producer: ViewProducer,
    state: S,
    draw: DrawFn<S>,
    schedules: Vec<Schedule>,
    budget: Option<Budget>,
    start: Instant,
}

/// The render scale's step under a [`Budget`], and its floor.
const BUDGET_STEP: f32 = 0.125;
const BUDGET_MIN: f32 = 0.25;

impl<S> RenderLoop<S> {
    /// A loop drawing `producer`'s views from `state` with `draw`, called once
    /// for each view that is due. Every view starts continuous at 60 fps.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(
        producer: ViewProducer,
        state: S,
        draw: impl FnMut(&mut S, &mut ViewDraw<'_>) + Send + 'static,
    ) -> Self {
        Self::with_draw(producer, state, Box::new(draw))
    }

    /// A loop drawing `producer`'s views from `state` with `draw`, called once
    /// for each view that is due. Every view starts continuous at 60 fps.
    #[cfg(target_arch = "wasm32")]
    pub fn new(
        producer: ViewProducer,
        state: S,
        draw: impl FnMut(&mut S, &mut ViewDraw<'_>) + 'static,
    ) -> Self {
        Self::with_draw(producer, state, Box::new(draw))
    }

    fn with_draw(producer: ViewProducer, state: S, draw: DrawFn<S>) -> Self {
        let mut this = Self {
            producer,
            state,
            draw,
            schedules: Vec::new(),
            budget: None,
            start: Instant::now(),
        };
        this.fit_schedules();
        this
    }

    fn fit_schedules(&mut self) {
        let n = self.producer.view_count();
        if self.schedules.len() < n {
            self.schedules.resize_with(n, Schedule::default);
        }
    }

    fn schedule(&mut self, view: ViewId) -> &mut Schedule {
        if self.schedules.len() <= view.index() {
            self.schedules.resize_with(view.index() + 1, Schedule::default);
        }
        &mut self.schedules[view.index()]
    }

    /// The render state, to change it directly (on the loop's own thread).
    pub fn state(&mut self) -> &mut S {
        &mut self.state
    }

    /// Changes the state and asks every view for a frame.
    pub fn update(&mut self, f: impl FnOnce(&mut S)) {
        f(&mut self.state);
        self.producer.request_redraw();
    }

    /// The producer, for its renderer and views.
    pub fn producer(&mut self) -> &mut ViewProducer {
        &mut self.producer
    }

    /// Another view, drawn by the same function (tell views apart by
    /// [`ViewDraw::id`]), and its [`ViewFrames`].
    pub fn add_view(&mut self) -> (ViewId, ViewFrames) {
        let (id, frames) = self.producer.add_view();
        self.schedule(id);
        (id, frames)
    }

    /// At most `fps` frames a second for `view`; 0 for no limit.
    pub fn set_fps(&mut self, view: ViewId, fps: f32) {
        self.schedule(view).pacer.set_fps(fps);
        self.producer.request_redraw();
    }

    /// Whether `view` draws continuously or only when asked.
    pub fn set_redraw(&mut self, view: ViewId, redraw: Redraw) {
        self.schedule(view).redraw = redraw;
        self.producer.request_redraw();
    }

    /// Stops drawing `view` (its UI keeps the last frame) or starts again.
    pub fn set_paused(&mut self, view: ViewId, paused: bool) {
        self.schedule(view).paused = paused;
        self.producer.request_redraw();
    }

    /// Keeps the views' GPU time per round within `ms`, by lowering the render
    /// scale of the lowest-priority view drawing above the floor (1/4) while
    /// over, and raising the highest-priority view below full resolution
    /// while comfortably under. `None` hands each view's scale back to it
    /// (fixed, or its own [`AdaptiveScale`](super::AdaptiveScale)).
    pub fn set_gpu_budget(&mut self, ms: Option<f32>) {
        self.budget = ms.map(|ms| Budget {
            ms,
            settle: 6,
            wait: 0,
        });
        if ms.is_some() {
            let ids: Vec<ViewId> = self.producer.view_ids().collect();
            for id in ids {
                let mut v = self.producer.view(id);
                let s = v.render_scale();
                v.set_render_scale(s);
            }
        }
    }

    /// Draws every view that is due at `now`, highest priority first, and
    /// returns when the next is due: `None` when none will be until something
    /// asks (all paused, hidden or on demand with nothing asked).
    pub fn step(&mut self, now: Instant) -> Option<Instant> {
        self.fit_schedules();
        let mut ids: Vec<ViewId> = self.producer.view_ids().collect();
        ids.sort_by_key(|&id| std::cmp::Reverse(self.producer.view(id).priority()));
        let mut drew = Vec::new();
        let mut next: Option<Instant> = None;
        for id in ids {
            let visible = self.producer.view(id).is_visible();
            let has_size = self.producer.view(id).size().is_some();
            let asked = self.producer.has_request(id);
            let sched = &mut self.schedules[id.index()];
            if sched.paused || !visible || !has_size {
                continue;
            }
            let wants = sched.redraw == Redraw::Continuous || asked;
            if !wants {
                continue;
            }
            if !sched.pacer.due(now) {
                next = Some(next.map_or(sched.pacer.next(), |n| n.min(sched.pacer.next())));
                continue;
            }
            let dt = sched.last.map_or(0.0, |t| now.duration_since(t).as_secs_f32());
            sched.last = Some(now);
            // Cleared before drawing: a request made while it draws (by the
            // draw itself, or the UI) is a request for the next frame.
            self.producer.take_request(id);
            let time = now.duration_since(self.start).as_secs_f32();
            let mut view = ViewDraw {
                view: self.producer.view(id),
                time,
                dt,
            };
            (self.draw)(&mut self.state, &mut view);
            drew.push(id);
            let sched = &self.schedules[id.index()];
            if sched.redraw == Redraw::Continuous {
                next = Some(next.map_or(sched.pacer.next(), |n| n.min(sched.pacer.next())));
            }
        }
        // A view asked for while it drew (an animation driven by update)
        // goes again at its next slot.
        for (i, sched) in self.schedules.iter().enumerate() {
            let id = ViewId(i);
            if i < self.producer.view_count()
                && !sched.paused
                && self.producer.has_request(id)
                && drew.contains(&id)
            {
                next = Some(next.map_or(sched.pacer.next(), |n| n.min(sched.pacer.next())));
            }
        }
        if !drew.is_empty() {
            self.balance();
        }
        next
    }

    /// One budget adjustment, after a round that drew something.
    fn balance(&mut self) {
        let Some(budget) = &mut self.budget else {
            return;
        };
        if budget.wait > 0 {
            budget.wait -= 1;
            return;
        }
        let ms = budget.ms;
        let mut views: Vec<(ViewId, i32, f32, f32)> = Vec::new();
        let ids: Vec<ViewId> = self.producer.view_ids().collect();
        for id in ids {
            let paused = self.schedules.get(id.index()).is_some_and(|s| s.paused);
            let v = self.producer.view(id);
            if paused || !v.is_visible() || v.size().is_none() {
                continue;
            }
            views.push((id, v.priority(), v.render_scale(), v.frame_ms()));
        }
        let total: f32 = views.iter().map(|v| v.3).sum();
        if total <= 0.0 {
            return;
        }
        let pick = if total > ms {
            // Over: the lowest priority still above the floor, the largest
            // scale among equals.
            views
                .iter()
                .filter(|v| v.2 > BUDGET_MIN + 1e-4)
                .min_by(|a, b| a.1.cmp(&b.1).then(b.2.total_cmp(&a.2)))
                .map(|v| (v.0, (v.2 - BUDGET_STEP).max(BUDGET_MIN)))
        } else {
            // Under: the highest priority below full, if the bigger frame is
            // predicted to stay within 90% of the budget.
            views
                .iter()
                .filter(|v| v.2 < 1.0 - 1e-4)
                .max_by(|a, b| a.1.cmp(&b.1).then(b.2.total_cmp(&a.2)))
                .and_then(|v| {
                    let up = (v.2 + BUDGET_STEP).min(1.0);
                    let grown = v.3 * (up / v.2).powi(2);
                    (total - v.3 + grown <= 0.9 * ms).then_some((v.0, up))
                })
        };
        if let Some((id, scale)) = pick {
            self.producer.view(id).set_render_scale(scale);
            if let Some(b) = &mut self.budget {
                b.wait = b.settle;
            }
        }
    }
}

/// A [`RenderLoop`] on a thread of its own. See the [module docs](self).
///
/// Dropping it stops the loop and joins the thread.
#[cfg(not(target_arch = "wasm32"))]
pub struct RenderThread<S: Send + 'static> {
    commands: std::sync::mpsc::Sender<Command<S>>,
    wake: ViewFramesWaker,
    failure: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    join: Option<std::thread::JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
type Command<S> = Box<dyn FnOnce(&mut RenderLoop<S>) + Send>;

/// Wakes the render thread: the producer's shared signal, through a view the
/// loop keeps for the purpose.
#[cfg(not(target_arch = "wasm32"))]
struct ViewFramesWaker(std::sync::Arc<dyn Fn() + Send + Sync>);

#[cfg(not(target_arch = "wasm32"))]
impl<S: Send + 'static> RenderThread<S> {
    /// Runs `render_loop` on a new thread, named `threers render`.
    pub fn spawn(render_loop: RenderLoop<S>) -> Self {
        let (commands, inbox) = std::sync::mpsc::channel::<Command<S>>();
        let waker = render_loop.producer.waker();
        let failure = std::sync::Arc::new(std::sync::Mutex::new(None));
        let failed = failure.clone();
        let join = std::thread::Builder::new()
            .name("threers render".into())
            .spawn(move || {
                let mut render_loop = render_loop;
                let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    'run: loop {
                        loop {
                            match inbox.try_recv() {
                                Ok(command) => command(&mut render_loop),
                                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                                // The handle is gone: stop.
                                Err(std::sync::mpsc::TryRecvError::Disconnected) => break 'run,
                            }
                        }
                        if render_loop.producer.is_closed() {
                            break;
                        }
                        let next = render_loop.step(Instant::now());
                        let timeout = next.map(|n| n.saturating_duration_since(Instant::now()));
                        if timeout != Some(Duration::ZERO)
                            && !render_loop.producer.wait_for_work_timeout(timeout)
                        {
                            break;
                        }
                    }
                }));
                if let Err(payload) = run {
                    let why = panic_message(&payload);
                    render_loop.producer.fail(why.clone());
                    *failed.lock().unwrap_or_else(|e| e.into_inner()) = Some(why);
                }
            })
            .expect("spawning the render thread");
        Self {
            commands,
            wake: ViewFramesWaker(waker),
            failure,
            join: Some(join),
        }
    }

    fn send(&self, command: Command<S>) {
        if self.commands.send(command).is_ok() {
            (self.wake.0)();
        }
    }

    /// Runs `f` on the render thread with the state, then asks every view
    /// for a frame. In order with every other command.
    pub fn update(&self, f: impl FnOnce(&mut S) + Send + 'static) {
        self.send(Box::new(move |l: &mut RenderLoop<S>| l.update(f)));
    }

    /// Runs `f` on the render thread with the state, and hands back its
    /// result: `recv()` it (blocking until the thread gets to it) or
    /// `try_recv()` it on a later UI frame. A loop that has stopped drops
    /// the sender, so `recv` returns an error rather than hanging.
    pub fn query<R: Send + 'static>(
        &self,
        f: impl FnOnce(&mut S) -> R + Send + 'static,
    ) -> std::sync::mpsc::Receiver<R> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.send(Box::new(move |l: &mut RenderLoop<S>| {
            let _ = tx.send(f(&mut l.state));
        }));
        rx
    }

    /// Runs `f` on the render thread with the loop itself: its producer,
    /// schedules and state.
    pub fn with_loop(&self, f: impl FnOnce(&mut RenderLoop<S>) + Send + 'static) {
        self.send(Box::new(f));
    }

    /// [`RenderLoop::set_fps`] on the render thread.
    pub fn set_fps(&self, view: ViewId, fps: f32) {
        self.with_loop(move |l| l.set_fps(view, fps));
    }

    /// [`RenderLoop::set_redraw`] on the render thread.
    pub fn set_redraw(&self, view: ViewId, redraw: Redraw) {
        self.with_loop(move |l| l.set_redraw(view, redraw));
    }

    /// [`RenderLoop::set_paused`] on the render thread.
    pub fn set_paused(&self, view: ViewId, paused: bool) {
        self.with_loop(move |l| l.set_paused(view, paused));
    }

    /// [`RenderLoop::set_gpu_budget`] on the render thread.
    pub fn set_gpu_budget(&self, ms: Option<f32>) {
        self.with_loop(move |l| l.set_gpu_budget(ms));
    }

    /// Another view, made on the render thread; waits for it.
    pub fn add_view(&self) -> Option<(ViewId, ViewFrames)> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.with_loop(move |l| {
            let _ = tx.send(l.add_view());
        });
        rx.recv().ok()
    }

    /// The panic that ended the loop, if one did.
    pub fn failure(&self) -> Option<String> {
        self.failure
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Whether the loop is still running.
    pub fn is_running(&self) -> bool {
        self.join.as_ref().is_some_and(|j| !j.is_finished())
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<S: Send + 'static> Drop for RenderThread<S> {
    fn drop(&mut self) {
        // Closing the channel stops the loop at its next wake.
        let (closed, _) = std::sync::mpsc::channel();
        drop(std::mem::replace(&mut self.commands, closed));
        (self.wake.0)();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    let what = payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic with no message".into());
    format!("the render thread panicked: {what}")
}

#[cfg(not(target_arch = "wasm32"))]
const _: () = {
    const fn send<T: Send>() {}
    send::<RenderLoop<Scene>>();
    send::<RenderThread<Scene>>();
};
