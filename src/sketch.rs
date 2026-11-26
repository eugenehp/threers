//! A window, a scene and a loop, in one call — p5's `setup`/`draw`.
//!
//! The shortest path from nothing to a moving picture used to be about eighty
//! lines: an event loop, an instance, an adapter, a device with two limits
//! raised by hand, a surface, a surface configuration, a renderer, and then a
//! `match` on window events with resize and redraw handled. Every interactive
//! example in this crate opens with the same eighty lines. None of it is
//! interesting and all of it has to be right.
//!
//! On macOS, built with `--features metal` and run with
//! `THREERS_BACKEND=metal`, the same sketch draws on the native Metal backend
//! — overlay and quad view included — through
//! [`WindowRenderer`].
//!
//! ```no_run
//! use threers::prelude::*;
//! use threers::sketch::{sketch, Frame, SketchOptions};
//!
//! sketch(SketchOptions::new("spinning cube"), |s| {
//!     let cube = s.scene.add_mesh(
//!         BoxGeometry::new(1.2, 1.2, 1.2),
//!         StandardMaterial::new(Color::from_hex(0xff6633)).into(),
//!     );
//!     s.scene.add_light(AmbientLight::new(Color::WHITE, 0.4));
//!     s.camera.position = Vector3::new(2.5, 2.0, 3.5);
//!     s.camera.look_at(Vector3::ZERO);
//!
//!     // Setup returns the draw closure, so whatever setup made is simply
//!     // captured — no shared state to thread through, and no `Option` that
//!     // is `None` until the first frame.
//!     move |f: &mut Frame| {
//!         if let Some(obj) = f.scene.get_mut(cube) {
//!             obj.quaternion = Euler::new(f.elapsed * 0.6, f.elapsed * 0.9, 0.0).to_quaternion();
//!         }
//!     }
//! });
//! ```
//!
//! # The overlay
//!
//! [`Frame::overlay`](crate::sketch::Frame::overlay) is a 2D canvas over the 3D
//! view — see [`crate::overlay`]. Written out in full because this module's docs
//! are concatenated with the `/// ` on `pub mod sketch` in `lib.rs` and then
//! resolved in *that* file's scope, where `Frame` is not in scope.
//! Cleared before `draw` and composited after it, and a frame that draws nothing
//! on it costs nothing:
//!
//! ```no_run
//! # use threers::prelude::*;
//! # use threers::sketch::{sketch, Frame, SketchOptions};
//! sketch(SketchOptions::new("hud"), |_s| {
//!     move |f: &mut Frame| {
//!         f.overlay.fill(Color::WHITE);
//!         f.overlay.text(12.0, 12.0, &format!("{:.0} fps", 1.0 / f.delta.max(1e-6)));
//!     }
//! });
//! ```
//!
//! # Four views
//!
//! [`SketchOptions::quad_view`](crate::sketch::SketchOptions::quad_view) draws
//! every frame through a [`QuadView`] — top, front
//! and side orthographic panes plus the sketch's perspective camera. `setup`
//! and `draw` get it as `quad_view`, to frame the scene or maximise a pane:
//!
//! ```no_run
//! # use threers::prelude::*;
//! # use threers::helpers::QuadView;
//! # use threers::sketch::{sketch, Frame, SketchOptions};
//! sketch(SketchOptions::new("four views").quad_view(QuadView::new()), |s| {
//!     s.scene.add_mesh(
//!         BoxGeometry::new(1.0, 1.0, 1.0),
//!         BasicMaterial::new(Color::WHITE).into(),
//!     );
//!     s.camera.position = Vector3::new(3.0, 2.0, 4.0);
//!     s.camera.look_at(Vector3::ZERO);
//!     if let Some(q) = s.quad_view.as_deref_mut() {
//!         q.frame_scene(s.scene);
//!     }
//!     move |_f: &mut Frame| {}
//! });
//! ```
//!
//! The mouse drives it through
//! [`QuadViewControls`]: left-drag in the
//! perspective pane orbits the camera and right/middle-drag pans it, any drag
//! in an orthographic pane pans all three together, the wheel zooms (one scale
//! for the three orthographic panes, about the cursor), and a double-click
//! maximises a pane or restores the four. Turn it off with
//! [`SketchOptions::quad_controls`](crate::sketch::SketchOptions::quad_controls()).
//!
//! Pane rectangles are logical pixels with a bottom-left origin, while
//! `Frame::pointer` is physical pixels from the top-left: divide by
//! `scale_factor` and flip y before calling `QuadView::pane_at`.
//!
//! This is a convenience, not a framework. It owns the window and the loop and
//! nothing else: `Frame` hands out `&mut Scene`, `&mut PerspectiveCamera`, the
//! overlay and the clock, and anything it does not cover is a reason to open the
//! event loop yourself rather than a reason to grow this.

use std::sync::Arc;
use std::time::Instant;

use crate::cameras::Camera;
use crate::helpers::{PointerButton, QuadView, QuadViewControls};
use crate::math::{Color, Vector2};
use crate::overlay::Overlay;
use crate::renderer::WindowRenderer;
use crate::{PerspectiveCamera, Scene};

/// One window and its event loop, the winit 0.30 way: the window is made in
/// `resumed`, through the running loop, and `init` turns it into the event
/// handler. That handler sees the same [`winit::event::Event`]s a 0.29
/// `EventLoop::run` closure did — `WindowEvent`, `AboutToWait`, `NewEvents`
/// and the rest — so a loop written in that shape moves over unchanged.
///
/// Blocks until the loop exits.
///
/// ```no_run
/// use threers::sketch::run_window;
/// use winit::event::{Event, WindowEvent};
///
/// let attributes = winit::window::Window::default_attributes().with_title("demo");
/// run_window(attributes, |window| {
///     move |event, target| match event {
///         Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => target.exit(),
///         Event::AboutToWait => window.request_redraw(),
///         _ => {}
///     }
/// });
/// ```
pub fn run_window<I, H>(attributes: winit::window::WindowAttributes, init: I)
where
    I: FnOnce(Arc<winit::window::Window>) -> H,
    H: FnMut(winit::event::Event<()>, &winit::event_loop::ActiveEventLoop),
{
    use winit::application::ApplicationHandler;
    use winit::event::{DeviceEvent, DeviceId, Event, StartCause, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::window::WindowId;

    struct App<I, H> {
        attributes: Option<winit::window::WindowAttributes>,
        init: Option<I>,
        handler: Option<H>,
    }

    impl<I, H> App<I, H>
    where
        H: FnMut(Event<()>, &ActiveEventLoop),
    {
        fn send(&mut self, event: Event<()>, target: &ActiveEventLoop) {
            if let Some(handler) = self.handler.as_mut() {
                handler(event, target);
            }
        }
    }

    impl<I, H> ApplicationHandler for App<I, H>
    where
        I: FnOnce(Arc<winit::window::Window>) -> H,
        H: FnMut(Event<()>, &ActiveEventLoop),
    {
        fn new_events(&mut self, target: &ActiveEventLoop, cause: StartCause) {
            self.send(Event::NewEvents(cause), target);
        }

        fn resumed(&mut self, target: &ActiveEventLoop) {
            if let (Some(attributes), Some(init)) = (self.attributes.take(), self.init.take()) {
                let window = Arc::new(target.create_window(attributes).expect("window"));
                self.handler = Some(init(window));
            }
            self.send(Event::Resumed, target);
        }

        fn window_event(
            &mut self,
            target: &ActiveEventLoop,
            window_id: WindowId,
            event: WindowEvent,
        ) {
            self.send(Event::WindowEvent { window_id, event }, target);
        }

        fn device_event(
            &mut self,
            target: &ActiveEventLoop,
            device_id: DeviceId,
            event: DeviceEvent,
        ) {
            self.send(Event::DeviceEvent { device_id, event }, target);
        }

        fn about_to_wait(&mut self, target: &ActiveEventLoop) {
            self.send(Event::AboutToWait, target);
        }

        fn suspended(&mut self, target: &ActiveEventLoop) {
            self.send(Event::Suspended, target);
        }

        fn exiting(&mut self, target: &ActiveEventLoop) {
            self.send(Event::LoopExiting, target);
        }

        fn memory_warning(&mut self, target: &ActiveEventLoop) {
            self.send(Event::MemoryWarning, target);
        }
    }

    let mut app = App {
        attributes: Some(attributes),
        init: Some(init),
        handler: None,
    };
    EventLoop::new()
        .expect("event loop")
        .run_app(&mut app)
        .expect("event loop");
}

/// Window and renderer settings. Everything has a default worth starting from.
#[derive(Debug, Clone)]
pub struct SketchOptions {
    pub title: String,
    pub width: u32,
    pub height: u32,
    /// Cleared to this before every frame.
    pub background: Color,
    /// Vertical field of view, in degrees.
    pub fov: f32,
    /// Multisampling. `1` is off; `4` is the usual choice when edges matter.
    /// Quad-view panes are multisampled too.
    pub msaa: u32,
    /// Draw through this quad view instead of one full-window view. See
    /// [`quad_view`](Self::quad_view).
    pub quad_view: Option<QuadView>,
    /// With a quad view: the mouse orbits, pans, zooms and maximises (see
    /// the [module docs](self#four-views)). On by default.
    pub quad_controls: bool,
    /// See [`retain_overlay`](Self::retain_overlay).
    pub retain_overlay: bool,
    /// See [`on_demand`](Self::on_demand).
    pub on_demand: bool,
    /// See [`ime`](Self::ime).
    pub ime: bool,
    /// See [`ask_to_close`](Self::ask_to_close).
    pub ask_to_close: bool,
}

impl SketchOptions {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Default::default()
        }
    }

    /// Window size in **logical** pixels — points of desk space, which is what
    /// asking for a window size means, and on a 2x display a 960x720 window is
    /// a 1920x1440 framebuffer.
    ///
    /// Everything handed to `draw` is physical instead: [`Frame::width`], the
    /// overlay and [`Frame::pointer`] are all real pixels and all agree with
    /// each other. [`Frame::scale_factor`] is the ratio between the two.
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    pub fn fov(mut self, degrees: f32) -> Self {
        self.fov = degrees;
        self
    }

    pub fn msaa(mut self, samples: u32) -> Self {
        self.msaa = samples;
        self
    }

    /// Split the window into `quad`'s four panes: top, front and side
    /// orthographic views, and the sketch camera in the perspective pane.
    /// `setup` and `draw` get it back as `quad_view` to frame or rearrange.
    pub fn quad_view(mut self, quad: QuadView) -> Self {
        self.quad_view = Some(quad);
        self
    }

    /// Whether the mouse drives the quad view (on by default).
    pub fn quad_controls(mut self, on: bool) -> Self {
        self.quad_controls = on;
        self
    }

    /// Keep the overlay between frames rather than clearing it before each:
    /// for an app that paints its window itself (a document, a form) and
    /// changes only what changed. Off by default.
    pub fn retain_overlay(mut self, on: bool) -> Self {
        self.retain_overlay = on;
        self
    }

    /// Draw only when there is reason to: after input, or when the frame
    /// asked for another ([`Frame::redraw_after`]); between, the loop sleeps.
    /// For an app (a document, a form) rather than an animation. Off by
    /// default: every frame drawn, as p5 does.
    pub fn on_demand(mut self, on: bool) -> Self {
        self.on_demand = on;
        self
    }

    /// Take text from the system's input methods: dead keys' accents, and
    /// Chinese, Japanese and Korean composed before it is typed
    /// ([`SketchEvent::Compose`], [`SketchEvent::Commit`]). Off by default.
    pub fn ime(mut self, on: bool) -> Self {
        self.ime = on;
        self
    }

    /// The window's close button asks ([`SketchEvent::Close`]) and the app
    /// closes it, when it will ([`Frame::exit`]): a document with changes
    /// unsaved. Off by default.
    pub fn ask_to_close(mut self, on: bool) -> Self {
        self.ask_to_close = on;
        self
    }
}

/// The keyboard's modifiers as an event found them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Modifiers {
    pub shift: bool,
    pub control: bool,
    pub alt: bool,
    /// ⌘ on a Mac, the Windows key elsewhere.
    pub meta: bool,
}

/// Input since the last frame, in order ([`Frame::events`]), for what the
/// per-frame pointer state does not carry: keys, typed text, the wheel, and
/// each press and release. Positions are physical pixels from the top-left,
/// as [`Frame::pointer`].
#[derive(Clone, Debug, PartialEq)]
pub enum SketchEvent {
    /// A key pressed (or repeating): its DOM name (`"ArrowLeft"`,
    /// `"Backspace"`, `"Enter"`) or, for a character key, the character as
    /// typed (`"a"`, `"A"`), with the text it types if it types any.
    Key { key: String, text: Option<String>, modifiers: Modifiers, repeat: bool },
    /// The left button (or a finger) down; `clicks` counts a double or
    /// triple click (each within 400 ms and a few pixels of the last).
    Press { x: f32, y: f32, clicks: u32, modifiers: Modifiers },
    /// The pointer moved.
    Move { x: f32, y: f32 },
    Release { x: f32, y: f32 },
    /// The wheel or a trackpad, in physical pixels; positive `dy` scrolls
    /// the content up (as DOM `deltaY`: down the page).
    Wheel { dx: f32, dy: f32, x: f32, y: f32 },
    /// Text an input method is composing, not yet typed (with
    /// [`SketchOptions::ime`]): shown in place, replaced by the next. Empty:
    /// none any more. `cursor` is the byte range selected in it.
    Compose { text: String, cursor: Option<(usize, usize)> },
    /// Text an input method has finished composing: typed.
    Commit(String),
    /// The window came to the front (`true`) or went behind another.
    Focus(bool),
    /// The system's appearance changed: dark (`true`) or light.
    Appearance(bool),
    /// A menu asked for at a point: the right button, or the left with ⌃
    /// (a Mac's).
    ContextMenu { x: f32, y: f32 },
    /// Two fingers pinching on a trackpad, at the pointer: `scale` the
    /// zoom this step asks (above 1 spreads, zooming in); `ended` once the
    /// fingers lift (its `scale` then 1).
    Pinch { scale: f32, x: f32, y: f32, ended: bool },
    /// The window asked to close (with [`SketchOptions::ask_to_close`]).
    Close,
}

/// The pointer's shape over the window ([`Frame::cursor`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Cursor {
    #[default]
    Default,
    /// Over text: the I-beam.
    Text,
    /// Over something to click: the hand.
    Pointer,
}

impl Default for SketchOptions {
    fn default() -> Self {
        Self {
            title: "threers".into(),
            width: 960,
            height: 720,
            background: Color::from_hex(0x101018),
            fov: 60.0,
            msaa: 1,
            quad_view: None,
            quad_controls: true,
            retain_overlay: false,
            on_demand: false,
            ime: false,
            ask_to_close: false,
        }
    }
}

/// What `setup` is given: an empty scene and a camera, both to be filled in.
pub struct Setup<'a> {
    pub scene: &'a mut Scene,
    pub camera: &'a mut PerspectiveCamera,
    /// Physical pixels per logical pixel. See [`Frame::scale_factor`].
    pub scale_factor: f32,
    /// The 2D overlay, for settings that only need saying once — the type size,
    /// say. Anything drawn here is erased before the first frame, because the
    /// overlay is cleared per frame.
    pub overlay: &'a mut Overlay,
    /// The quad view, when [`SketchOptions::quad_view`] asked for one.
    pub quad_view: Option<&'a mut QuadView>,
    pub width: u32,
    pub height: u32,
}

/// What `draw` is given, once per frame.
pub struct Frame<'a> {
    pub scene: &'a mut Scene,
    pub camera: &'a mut PerspectiveCamera,
    /// Immediate-mode 2D over the top of the render. Cleared before every
    /// `draw`, so draw what this frame should show and nothing persists.
    pub overlay: &'a mut Overlay,
    /// The quad view the frame is drawn through, when
    /// [`SketchOptions::quad_view`] asked for one. Its pane rectangles are in
    /// logical pixels, bottom-left origin.
    pub quad_view: Option<&'a mut QuadView>,
    /// Frames drawn so far. p5's `frameCount`.
    pub frame: u64,
    /// Seconds since the first frame.
    pub elapsed: f32,
    /// Seconds since the previous frame. Use this and not a fixed step, or the
    /// sketch runs at a different speed on a different monitor.
    pub delta: f32,
    pub width: u32,
    pub height: u32,
    /// Pointer position in pixels, origin top-left. p5's `mouseX`/`mouseY`.
    pub pointer: Vector2,
    /// Pointer position as `-1..1` with y up, which is the frame most 3D maths
    /// wants and saves every sketch the same two lines.
    pub pointer_ndc: Vector2,
    pub pointer_down: bool,
    /// Whether the pointer is over the window. When it leaves, `pointer` keeps
    /// the last position it had, which is rarely what hover effects want.
    pub pointer_inside: bool,
    /// Show the system cursor over the window. Set it to `false` to draw your
    /// own; it stays as last set until changed.
    pub cursor_visible: bool,
    /// Physical pixels per logical pixel — p5's `pixelDensity()`. `1.0` on an
    /// ordinary display and `2.0` on a Retina one.
    ///
    /// [`width`](Self::width), [`height`](Self::height),
    /// [`pointer`](Self::pointer) and the overlay are all in physical pixels,
    /// so they agree with each other and with
    /// [`Camera::world_to_screen`]
    /// without conversion. What they do *not* do is look the same size on two
    /// displays: a 15-pixel label is half as tall on a 2x screen as the window
    /// size asked for. Multiply anything meant to be read — type, line weights,
    /// hit radii — by this:
    ///
    /// ```no_run
    /// # use threers::sketch::Frame;
    /// # fn draw(f: &mut Frame) {
    /// f.overlay.text_size(15.0 * f.scale_factor);
    /// # }
    /// ```
    pub scale_factor: f32,
    /// Set this to stop the loop and close the window.
    pub exit: bool,
    /// Input since the last frame, in order.
    pub events: &'a [SketchEvent],
    /// With [`SketchOptions::on_demand`]: draw again this many seconds on
    /// (a caret's blink, an animation), else not until there is input.
    /// `None` when the frame is drawn.
    pub redraw_after: Option<f32>,
    /// Where text is being typed, in physical pixels (`x, y, width,
    /// height`): an input method's candidates are shown beside it.
    pub ime_area: Option<[f32; 4]>,
    /// Set to retitle the window (a document opened, its name changed).
    pub title: Option<String>,
    /// Whether the system shows windows dark (its appearance, as it is now:
    /// a change is a [`SketchEvent::Appearance`] too).
    pub dark_appearance: bool,
    /// The pointer's shape over the window; it stays as last set.
    pub cursor: Cursor,
    /// Whether the last frame reached the display (not when the window is
    /// hidden, minimised or covered: the system then shows nothing, and the
    /// loop waits a refresh rather than racing).
    pub presented: bool,
    /// How many times a second the window's display refreshes (120 on a
    /// ProMotion screen, 60 on most), where the system says.
    pub refresh_rate: Option<f32>,
}

/// Open a window and run `setup`, then its returned closure every frame.
///
/// Blocks until the window closes. Native only — the browser has its own loop
/// and `WebRenderer` is the way in there.
pub fn sketch<D, S>(options: SketchOptions, setup: S)
where
    S: FnOnce(&mut Setup) -> D,
    D: FnMut(&mut Frame) + 'static,
{
    use winit::event::{ElementState, Event, Ime, MouseButton, MouseScrollDelta, StartCause, WindowEvent};
    use winit::event_loop::ControlFlow;
    use winit::keyboard::{Key, NamedKey};

    let attributes = winit::window::Window::default_attributes()
        .with_title(options.title.clone())
        .with_inner_size(winit::dpi::LogicalSize::new(options.width, options.height));
    run_window(attributes, move |window| {
        // wgpu, or the native Metal backend with `--features metal` and
        // `THREERS_BACKEND=metal`.
        let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");
        view.set_msaa(options.msaa);
        // Viewport and scissor rectangles — the quad view's panes — are logical
        // pixels, like the window size asked for; this maps them to the surface.
        view.set_pixel_ratio(window.scale_factor() as f32);
        let mut quad_view = options.quad_view.clone();

        let mut scene = Scene::new();
        scene.background = options.background;
        let mut camera = PerspectiveCamera::new(options.fov, view.aspect(), 0.1, 1000.0);

        let mut overlay = Overlay::new(view.size().0, view.size().1);
        let mut draw = {
            let mut s = Setup {
                scene: &mut scene,
                camera: &mut camera,
                overlay: &mut overlay,
                quad_view: quad_view.as_mut(),
                scale_factor: window.scale_factor() as f32,
                width: view.size().0,
                height: view.size().1,
            };
            setup(&mut s)
        };

        let start = Instant::now();
        let mut last = start;
        let mut frame_index: u64 = 0;
        let mut pointer = Vector2::new(0.0, 0.0);
        let mut pointer_down = false;
        let mut pointer_inside = false;
        let mut cursor_visible = true;
        let win = window.clone();
        // Mouse → quad view, after `setup` has placed the camera.
        let mut quad_controls =
            (options.quad_controls && quad_view.is_some()).then(|| QuadViewControls::new(&camera));
        // The last left press, for telling a double-click (winit reports none).
        let mut last_click: Option<(Instant, Vector2)> = None;
        // Input for the next frame, and the clicks counted for it.
        let mut events: Vec<SketchEvent> = Vec::new();
        let mut modifiers = Modifiers::default();
        let mut press_run: Option<(Instant, Vector2, u32)> = None;
        let retain_overlay = options.retain_overlay;
        let on_demand = options.on_demand;
        if options.ime {
            window.set_ime_allowed(true);
        }
        let mut ime_area: Option<[f32; 4]> = None;
        let mut dark_appearance = window.theme() == Some(winit::window::Theme::Dark);
        let mut cursor = Cursor::Default;
        let mut presented = true;

        move |event, target| {
            // a frame asked for, its time come
            if let Event::NewEvents(StartCause::ResumeTimeReached { .. }) = event {
                win.request_redraw();
                return;
            }
            let Event::WindowEvent { event, window_id } = event else {
                return;
            };
            if window_id != win.id() {
                return;
            }
            // on demand: input is a reason to draw
            if on_demand && !matches!(event, WindowEvent::RedrawRequested) {
                win.request_redraw();
            }
            match event {
                WindowEvent::Ime(Ime::Preedit(text, cursor)) => events.push(SketchEvent::Compose { text, cursor }),
                WindowEvent::Ime(Ime::Commit(text)) => events.push(SketchEvent::Commit(text)),
                WindowEvent::Focused(on) => events.push(SketchEvent::Focus(on)),
                WindowEvent::ThemeChanged(t) => {
                    dark_appearance = t == winit::window::Theme::Dark;
                    events.push(SketchEvent::Appearance(dark_appearance));
                }
                WindowEvent::CloseRequested if options.ask_to_close => events.push(SketchEvent::Close),
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(s) => {
                    view.resize(s.width, s.height);
                    let (w, h) = view.size();
                    overlay.resize(w, h);
                    camera.set_aspect(view.aspect());
                    view.set_pixel_ratio(win.scale_factor() as f32);
                }
                WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    // The `Resized` that follows reconfigures the surface.
                    view.set_pixel_ratio(scale_factor as f32);
                }
                WindowEvent::CursorEntered { .. } => pointer_inside = true,
                WindowEvent::CursorLeft { .. } => pointer_inside = false,
                WindowEvent::ModifiersChanged(m) => {
                    let st = m.state();
                    modifiers = Modifiers {
                        shift: st.shift_key(),
                        control: st.control_key(),
                        alt: st.alt_key(),
                        meta: st.super_key(),
                    };
                }
                WindowEvent::KeyboardInput { event: key, .. } if key.state == ElementState::Pressed => {
                    let name = match &key.logical_key {
                        Key::Named(NamedKey::Space) => Some(" ".to_string()),
                        Key::Named(n) => Some(format!("{n:?}")),
                        Key::Character(c) => Some(c.to_string()),
                        _ => None,
                    };
                    if let Some(name) = name {
                        // text is what a key types: not with ⌘ or ⌃ held
                        let text = key
                            .text
                            .as_ref()
                            .map(|t| t.to_string())
                            .filter(|t| !t.chars().all(char::is_control) && !modifiers.meta && !modifiers.control);
                        events.push(SketchEvent::Key { key: name, text, modifiers, repeat: key.repeat });
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    // Some platforms send no `CursorEntered` for a pointer
                    // already over the window when it opens; a move proves it.
                    pointer_inside = true;
                    pointer = Vector2::new(position.x as f32, position.y as f32);
                    events.push(SketchEvent::Move { x: pointer.x, y: pointer.y });
                    if let (Some(ctl), Some(quad)) = (quad_controls.as_mut(), quad_view.as_mut()) {
                        let s = win.scale_factor() as f32;
                        ctl.pointer_move(quad, &mut camera, pointer.x / s, pointer.y / s);
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    // the right button, or ⌃ and the left: a menu
                    let menu = state == ElementState::Pressed
                        && (button == MouseButton::Right || (button == MouseButton::Left && modifiers.control));
                    if menu {
                        events.push(SketchEvent::ContextMenu { x: pointer.x, y: pointer.y });
                    } else if button == MouseButton::Left && !(modifiers.control && state == ElementState::Released && !pointer_down) {
                        pointer_down = state == ElementState::Pressed;
                        if pointer_down {
                            let now = Instant::now();
                            let s = win.scale_factor() as f32;
                            let clicks = match press_run {
                                Some((t, at, n)) if now.duration_since(t).as_millis() < 400 && (at - pointer).length() < 6.0 * s => n + 1,
                                _ => 1,
                            };
                            press_run = Some((now, pointer, clicks));
                            events.push(SketchEvent::Press { x: pointer.x, y: pointer.y, clicks, modifiers });
                        } else {
                            events.push(SketchEvent::Release { x: pointer.x, y: pointer.y });
                        }
                    }
                    if let (Some(ctl), Some(quad)) = (quad_controls.as_mut(), quad_view.as_mut()) {
                        let s = win.scale_factor() as f32;
                        ctl.set_size(view.size().0 as f32 / s, view.size().1 as f32 / s);
                        let (x, y) = (pointer.x / s, pointer.y / s);
                        let button = match button {
                            MouseButton::Left => Some(PointerButton::Primary),
                            MouseButton::Middle => Some(PointerButton::Middle),
                            MouseButton::Right => Some(PointerButton::Secondary),
                            _ => None,
                        };
                        match (state, button) {
                            (ElementState::Pressed, Some(PointerButton::Primary)) => {
                                let now = Instant::now();
                                let double = last_click.is_some_and(|(t, at)| {
                                    now.duration_since(t).as_millis() < 400
                                        && (at - pointer).length() < 6.0 * s
                                });
                                if double {
                                    last_click = None;
                                    ctl.dblclick(quad, x, y);
                                } else {
                                    last_click = Some((now, pointer));
                                    ctl.pointer_down(quad, &camera, x, y, PointerButton::Primary);
                                }
                            }
                            (ElementState::Pressed, Some(b)) => {
                                ctl.pointer_down(quad, &camera, x, y, b);
                            }
                            (ElementState::Released, Some(_)) => ctl.pointer_up(),
                            _ => {}
                        }
                    }
                }
                WindowEvent::PinchGesture { delta, phase, .. } => {
                    let ended = matches!(phase, winit::event::TouchPhase::Ended | winit::event::TouchPhase::Cancelled);
                    let scale = if ended { 1.0 } else { (1.0 + delta as f32).max(0.05) };
                    events.push(SketchEvent::Pinch { scale, x: pointer.x, y: pointer.y, ended });
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let s = win.scale_factor() as f32;
                    let (dx, dy) = match delta {
                        MouseScrollDelta::LineDelta(x, y) => (-x * 40.0 * s, -y * 40.0 * s),
                        MouseScrollDelta::PixelDelta(p) => (-(p.x as f32), -(p.y as f32)),
                    };
                    events.push(SketchEvent::Wheel { dx, dy, x: pointer.x, y: pointer.y });
                    if let (Some(ctl), Some(quad)) = (quad_controls.as_mut(), quad_view.as_mut()) {
                        let s = win.scale_factor() as f32;
                        ctl.set_size(view.size().0 as f32 / s, view.size().1 as f32 / s);
                        // As DOM `deltaY`: positive scrolls down (zooms out).
                        // winit's positive is up; a line is taken as 100 px,
                        // which is what a browser reports for a wheel notch.
                        let d = match delta {
                            MouseScrollDelta::LineDelta(_, y) => -y * 100.0,
                            MouseScrollDelta::PixelDelta(p) => -(p.y as f32) / s,
                        };
                        ctl.wheel(quad, &mut camera, pointer.x / s, pointer.y / s, d);
                    }
                }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let delta = now.duration_since(last).as_secs_f32();
                    last = now;

                    // Immediate mode: this frame draws what this frame shows
                    // (unless the app keeps its own picture there).
                    if !retain_overlay {
                        overlay.clear();
                    }
                    let frame_events = std::mem::take(&mut events);
                    let mut f = Frame {
                        scene: &mut scene,
                        camera: &mut camera,
                        overlay: &mut overlay,
                        quad_view: quad_view.as_mut(),
                        frame: frame_index,
                        elapsed: start.elapsed().as_secs_f32(),
                        delta,
                        width: view.size().0,
                        height: view.size().1,
                        pointer,
                        pointer_ndc: Vector2::new(
                            pointer.x / view.size().0.max(1) as f32 * 2.0 - 1.0,
                            1.0 - pointer.y / view.size().1.max(1) as f32 * 2.0,
                        ),
                        pointer_down,
                        pointer_inside,
                        cursor_visible,
                        // Read rather than cached, so dragging the window to a
                        // display with a different density needs no event: the
                        // `Resized` that follows already reconfigures the
                        // surface, and this follows it for free.
                        scale_factor: win.scale_factor() as f32,
                        exit: false,
                        events: &frame_events,
                        redraw_after: None,
                        ime_area,
                        title: None,
                        dark_appearance,
                        cursor,
                        presented,
                        refresh_rate: win
                            .current_monitor()
                            .and_then(|m| m.refresh_rate_millihertz())
                            .map(|mhz| mhz as f32 / 1000.0),
                    };
                    draw(&mut f);
                    let wants_exit = f.exit;
                    let redraw_after = f.redraw_after;
                    let f_refresh = f.refresh_rate;
                    if let Some(t) = f.title.take() {
                        win.set_title(&t);
                    }
                    if f.cursor != cursor {
                        cursor = f.cursor;
                        win.set_cursor(match cursor {
                            Cursor::Default => winit::window::CursorIcon::Default,
                            Cursor::Text => winit::window::CursorIcon::Text,
                            Cursor::Pointer => winit::window::CursorIcon::Pointer,
                        });
                    }
                    if f.ime_area != ime_area {
                        ime_area = f.ime_area;
                        if let Some([x, y, w, h]) = ime_area {
                            win.set_ime_cursor_area(winit::dpi::PhysicalPosition::new(x, y), winit::dpi::PhysicalSize::new(w, h));
                        }
                    }
                    if f.cursor_visible != cursor_visible {
                        cursor_visible = f.cursor_visible;
                        win.set_cursor_visible(cursor_visible);
                    }
                    frame_index += 1;

                    // Hidden or minimised: `render_frame` skips the frame.
                    let refresh = f_refresh.unwrap_or(60.0).max(1.0);
                    presented = view.render_frame(&mut scene, &camera, quad_view.as_mut(), Some(&mut overlay));
                    if wants_exit {
                        target.exit();
                    }
                    if !on_demand {
                        win.request_redraw();
                    } else {
                        match redraw_after {
                            // nothing shown (hidden): a refresh's wait, not a race
                            Some(s) if s <= 0.0 && !presented => target.set_control_flow(ControlFlow::WaitUntil(
                                Instant::now() + std::time::Duration::from_secs_f32(1.0 / refresh),
                            )),
                            Some(s) if s <= 0.0 => win.request_redraw(),
                            Some(s) => target.set_control_flow(ControlFlow::WaitUntil(
                                Instant::now() + std::time::Duration::from_secs_f32(s),
                            )),
                            None => target.set_control_flow(ControlFlow::Wait),
                        }
                    }
                }
                _ => {}
            }
        }
    });
}
