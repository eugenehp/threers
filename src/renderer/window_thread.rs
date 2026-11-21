//! A window drawn on a thread of its own: the event loop stays on the main
//! thread, as winit requires, and a second thread owns the
//! [`WindowRenderer`], lays out and draws (a [`Ui`](crate::ui::Ui) and its
//! [`UiPass`](super::UiPass), a scene, both) and presents. A busy event loop
//! no longer holds up frames, and a slow frame no longer holds up the event
//! loop.
//!
//! The main thread creates the window and its renderer, hands both to
//! [`WindowThread::spawn`], and forwards its window events with
//! [`WindowThread::send`]. That never waits. The thread runs `init` first,
//! and whatever `init` builds stays on the thread. That is where a `Ui`
//! belongs: it holds `Rc`s and closures, so it is built and used on one
//! thread and never moved. `init` returns the handler, which gets each
//! [`ThreadEvent`] with a [`ThreadControl`]: the renderer, the window's
//! size and scale, and when to draw next.
//!
//! ```ignore
//! threers::sketch::run_window(attributes, |window| {
//!     let renderer = WindowRenderer::new(window.clone()).unwrap();
//!     let thread = WindowThread::spawn(window.clone(), renderer, |ctl| {
//!         // On the UI thread from here on.
//!         let device = ctl.renderer().renderer().unwrap().device_arc();
//!         let mut pass = UiPass::new(&device);
//!         let mut ui = Ui::new();
//!         // ... build the tree ...
//!         move |event, ctl: &mut ThreadControl| match event {
//!             ThreadEvent::Window(e) => { /* input: update `ui` */ ctl.request_redraw(); }
//!             ThreadEvent::Redraw => {
//!                 let scale = ctl.scale_factor() as f32;
//!                 let (w, h) = ctl.logical_size();
//!                 ui.layout(w, h, &mut pass.text_measure(scale));
//!                 let scene = ui.scene();
//!                 ctl.renderer().present_with(|device, queue, target, size, format| {
//!                     pass.draw_retained(device, queue, &scene, scale, target, size, format, [0.0, 0.0, 0.0, 1.0]);
//!                 });
//!             }
//!             _ => {}
//!         }
//!     }).unwrap();
//!     let mut thread = Some(thread);
//!     move |event, target| match event {
//!         Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
//!             thread.take().map(WindowThread::close);
//!             target.exit();
//!         }
//!         Event::WindowEvent { event, .. } => { thread.as_ref().map(|t| t.send(event)); }
//!         _ => {}
//!     }
//! });
//! ```
//!
//! The thread sleeps until something happens: an event, a
//! [`ThreadWaker`] (a [`ViewFrames::on_frame`](super::ViewFrames::on_frame)
//! callback, for a 3D view drawn on a third thread), or the time given to
//! [`ThreadControl::redraw_at`] (an animation, a frame cap). It handles every
//! event waiting before it draws, so a burst of pointer moves costs one
//! frame. The thread resizes the renderer itself before it passes on
//! `Resized`, and it turns `RedrawRequested` into a [`ThreadEvent::Redraw`].
//!
//! Don't call [`Window`] getters (`inner_size`, `scale_factor`, ...) on the
//! thread. On macOS each one waits for the main thread, which deadlocks if the
//! main thread is waiting for this one, as it does in
//! [`WindowThread::close`]. Use [`ThreadControl::size`] and
//! [`ThreadControl::scale_factor`], which follow the forwarded events.
//! [`Window::request_redraw`] only queues, so it is safe. Dropping the last
//! reference to the window waits too, so [`WindowThread`] keeps one until the
//! thread ends. Don't pass clones on to threads that may outlive it.
//!
//! The window's renderer must be on the wgpu backend. The native Metal
//! window cannot leave the main thread.

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use winit::event::WindowEvent;
use winit::window::Window;

use super::WindowRenderer;

enum Msg {
    Event(WindowEvent),
    Wake,
    Redraw,
    Close,
}

/// What a [`WindowThread`]'s handler is called with.
#[derive(Debug)]
pub enum ThreadEvent {
    /// A window event the event loop forwarded. `Resized` and
    /// `ScaleFactorChanged` arrive after [`ThreadControl`] has taken the new
    /// size and resized the renderer, and each asks for a redraw.
    /// `RedrawRequested` never arrives: it becomes [`Redraw`](Self::Redraw).
    Window(WindowEvent),
    /// [`ThreadWaker::wake`] was called.
    Wake,
    /// Time to draw a frame: one was asked for, or a
    /// [`redraw_at`](ThreadControl::redraw_at) time came. It follows every
    /// event that was waiting.
    Redraw,
    /// The window is closing. This is the last event, and the thread ends
    /// after it.
    Close,
}

/// The thread's side of the window: its renderer, its size, and when to
/// draw next.
pub struct ThreadControl {
    renderer: WindowRenderer,
    window: Arc<Window>,
    size: (u32, u32),
    scale_factor: f64,
    redraw: bool,
    redraw_at: Option<Instant>,
    exit: bool,
}

impl ThreadControl {
    /// The window's renderer, owned by this thread.
    pub fn renderer(&mut self) -> &mut WindowRenderer {
        &mut self.renderer
    }

    /// The window, for calls that only queue work for the main thread, such
    /// as [`Window::request_redraw`] and `set_cursor_icon`. Don't call its
    /// getters; see the [module docs](self).
    pub fn window(&self) -> &Arc<Window> {
        &self.window
    }

    /// The window's inner size in physical pixels, as the last forwarded
    /// `Resized` gave it.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// The window's scale factor, as the last forwarded
    /// `ScaleFactorChanged` gave it.
    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// [`size`](Self::size) in logical pixels, which a `Ui` lays out in.
    pub fn logical_size(&self) -> (f32, f32) {
        let s = self.scale_factor as f32;
        (self.size.0 as f32 / s, self.size.1 as f32 / s)
    }

    /// Draw once every waiting event has been handled.
    pub fn request_redraw(&mut self) {
        self.redraw = true;
    }

    /// Draw at `at`, or at an earlier time already asked for. Ask again each
    /// frame for an animation or a frame cap, e.g. with
    /// [`FramePacer::next`](super::FramePacer::next).
    pub fn redraw_at(&mut self, at: Instant) {
        self.redraw_at = Some(self.redraw_at.map_or(at, |t| t.min(at)));
    }

    /// End the thread after this event. The handler gets no
    /// [`ThreadEvent::Close`]. The event loop sees
    /// [`WindowThread::is_finished`].
    pub fn exit(&mut self) {
        self.exit = true;
    }
}

/// Wakes a [`WindowThread`] from any thread. Clones wake the same thread.
#[derive(Clone)]
pub struct ThreadWaker(Sender<Msg>);

impl ThreadWaker {
    /// Sends the handler a [`ThreadEvent::Wake`].
    pub fn wake(&self) {
        let _ = self.0.send(Msg::Wake);
    }

    /// Asks the thread to draw, as [`ThreadControl::request_redraw`] would.
    pub fn request_redraw(&self) {
        let _ = self.0.send(Msg::Redraw);
    }
}

/// A thread that owns a window's renderer and draws it. See the
/// [module docs](self).
pub struct WindowThread {
    tx: Sender<Msg>,
    /// Disconnects when the thread ends, however it ends.
    done: Receiver<()>,
    thread: Option<JoinHandle<()>>,
    /// Held so the thread never drops the last reference: on macOS a
    /// `Window`'s drop waits for the main thread.
    _window: Arc<Window>,
}

/// Moves a wgpu-backed [`WindowRenderer`] to the thread. Only the native
/// Metal backend makes it `!Send`, and [`WindowThread::spawn`] turns that
/// away first.
struct SendRenderer(WindowRenderer);
// SAFETY: built only for a renderer on the wgpu backend, whose parts are all
// `Send` (the type is `Send` outright when the Metal backend isn't built).
unsafe impl Send for SendRenderer {}

/// How long [`WindowThread`]'s drop waits for the thread to finish.
const CLOSE_WAIT: Duration = Duration::from_secs(2);

impl WindowThread {
    /// Starts the thread. Call this on the main thread: it reads `window`'s
    /// size and scale before it starts the thread. On the thread, `init`
    /// runs with the [`ThreadControl`] and returns the handler, and a first
    /// frame is asked for. Fails for a renderer on the native Metal
    /// backend, which must stay on the main thread.
    pub fn spawn<I, H>(window: Arc<Window>, renderer: WindowRenderer, init: I) -> Result<Self, String>
    where
        I: FnOnce(&mut ThreadControl) -> H + Send + 'static,
        H: FnMut(ThreadEvent, &mut ThreadControl) + 'static,
    {
        if renderer.is_native_metal() {
            return Err("a window on the native Metal backend can't be drawn on another thread; \
                 use the wgpu backend (unset THREERS_BACKEND)"
                .into());
        }
        let renderer = SendRenderer(renderer);
        let size = window.inner_size();
        let scale_factor = window.scale_factor();
        let (tx, rx) = mpsc::channel();
        let held = window.clone();
        let (done_tx, done) = mpsc::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("threers-window".into())
            .spawn(move || {
                // Dropped last, however the thread ends, so `done` sees it.
                let _done = done_tx;
                let SendRenderer(renderer) = { renderer };
                let mut ctl = ThreadControl {
                    renderer,
                    window,
                    size: (size.width.max(1), size.height.max(1)),
                    scale_factor,
                    redraw: true,
                    redraw_at: None,
                    exit: false,
                };
                let handler = init(&mut ctl);
                run(rx, &mut ctl, handler);
            })
            .map_err(|e| format!("a thread for the window: {e}"))?;
        Ok(Self {
            tx,
            done,
            thread: Some(thread),
            _window: held,
        })
    }

    /// Hands a window event to the thread. Never waits. Returns `false` once
    /// the thread has ended.
    pub fn send(&self, event: WindowEvent) -> bool {
        self.tx.send(Msg::Event(event)).is_ok()
    }

    /// Something that wakes the thread from anywhere, e.g. from a
    /// [`ViewFrames::on_frame`](super::ViewFrames::on_frame) callback.
    pub fn waker(&self) -> ThreadWaker {
        ThreadWaker(self.tx.clone())
    }

    /// Asks the thread to draw.
    pub fn request_redraw(&self) {
        let _ = self.tx.send(Msg::Redraw);
    }

    /// Whether the thread has ended: through [`ThreadControl::exit`], after
    /// [`ThreadEvent::Close`], or by panicking.
    pub fn is_finished(&self) -> bool {
        matches!(self.done.try_recv(), Err(TryRecvError::Disconnected))
    }

    /// Sends [`ThreadEvent::Close`] and waits for the thread to end, for up to
    /// two seconds. Returns `false` if the thread panicked or was still
    /// running when the wait ran out, in which case it is left to finish on
    /// its own.
    pub fn close(mut self) -> bool {
        self.shut(CLOSE_WAIT)
    }

    /// [`close`](Self::close), waiting up to `wait`.
    pub fn close_timeout(mut self, wait: Duration) -> bool {
        self.shut(wait)
    }

    fn shut(&mut self, wait: Duration) -> bool {
        let Some(thread) = self.thread.take() else {
            return true;
        };
        let _ = self.tx.send(Msg::Close);
        // Waits for the thread to end, but no longer than `wait`: a thread
        // waiting on the main thread (a window getter on macOS) would
        // never finish while the main thread waits here.
        match self.done.recv_timeout(wait) {
            Err(RecvTimeoutError::Disconnected) => thread.join().is_ok(),
            _ => false,
        }
    }
}

impl Drop for WindowThread {
    fn drop(&mut self) {
        self.shut(CLOSE_WAIT);
    }
}

fn run<H: FnMut(ThreadEvent, &mut ThreadControl)>(
    rx: Receiver<Msg>,
    ctl: &mut ThreadControl,
    mut handler: H,
) {
    loop {
        // Sleep until a message comes, or until a frame is due.
        let first = if ctl.redraw {
            match rx.try_recv() {
                Ok(m) => Some(m),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => Some(Msg::Close),
            }
        } else if let Some(at) = ctl.redraw_at {
            match rx.recv_timeout(at.saturating_duration_since(Instant::now())) {
                Ok(m) => Some(m),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => Some(Msg::Close),
            }
        } else {
            Some(rx.recv().unwrap_or(Msg::Close))
        };
        // Then handle all that is waiting before drawing.
        let mut next = first;
        while let Some(msg) = next {
            match msg {
                Msg::Event(WindowEvent::RedrawRequested) | Msg::Redraw => ctl.redraw = true,
                Msg::Event(event) => {
                    match &event {
                        WindowEvent::Resized(size) => {
                            ctl.size = (size.width.max(1), size.height.max(1));
                            ctl.renderer.resize(ctl.size.0, ctl.size.1);
                            ctl.redraw = true;
                        }
                        WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                            ctl.scale_factor = *scale_factor;
                            ctl.redraw = true;
                        }
                        _ => {}
                    }
                    handler(ThreadEvent::Window(event), ctl);
                }
                Msg::Wake => handler(ThreadEvent::Wake, ctl),
                Msg::Close => {
                    handler(ThreadEvent::Close, ctl);
                    return;
                }
            }
            if ctl.exit {
                return;
            }
            next = match rx.try_recv() {
                Ok(m) => Some(m),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => Some(Msg::Close),
            };
        }
        if ctl.redraw_at.is_some_and(|at| at <= Instant::now()) {
            ctl.redraw_at = None;
            ctl.redraw = true;
        }
        if ctl.redraw {
            ctl.redraw = false;
            handler(ThreadEvent::Redraw, ctl);
            if ctl.exit {
                return;
            }
        }
    }
}
