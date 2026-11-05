//! A native window around an [`App`].
//!
//! This is the only place the crate touches winit: it turns window events into
//! [`Input`], calls [`App::frame`] with the time that passed, and draws
//! `world.scene` through [`MainCamera`]. Everything else runs the same with no
//! window at all. With the `gamepad` feature it polls gamepads too.

use crate::app::{App, Exit, MainCamera};
use crate::hud::Hud;
use crate::input::{Input, Key, MouseButton};
use std::sync::Arc;
use std::time::Instant;
use threers::cameras::PerspectiveCamera;
use threers::math::Vector2;
use threers::renderer::{WindowBackend, WindowRenderer};
use winit::event::{DeviceEvent, ElementState, Event, MouseScrollDelta, TouchPhase, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, WindowBuilder};

/// Window settings. Everything has a default worth starting from.
#[derive(Debug, Clone)]
pub struct WindowOptions {
    pub title: String,
    pub width: u32,
    pub height: u32,
    /// Multisampling. `1` is off; `4` is the usual choice.
    pub msaa: u32,
    /// What draws it: wgpu, unless `THREERS_BACKEND=metal` asks for threers'
    /// native Metal renderer (with the `metal` feature) by default.
    pub backend: WindowBackend,
}

impl WindowOptions {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            width: 1280,
            height: 720,
            msaa: 4,
            backend: WindowBackend::Auto,
        }
    }

    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn msaa(mut self, samples: u32) -> Self {
        self.msaa = samples;
        self
    }

    /// Draw with threers' native Metal renderer rather than wgpu. Needs the
    /// `metal` feature and an Apple platform; [`run`] panics, saying so, if
    /// Metal cannot open.
    pub fn metal(mut self) -> Self {
        self.backend = WindowBackend::Metal;
        self
    }

    pub fn backend(mut self, backend: WindowBackend) -> Self {
        self.backend = backend;
        self
    }
}

/// Feed a key event to `input`. Keys with no [`Key`] are ignored.
pub fn apply_key(input: &mut Input, code: KeyCode, pressed: bool) {
    let Some(key) = Key::from_code(&format!("{code:?}")) else {
        return;
    };
    if pressed {
        input.press(key);
    } else {
        input.release(key);
    }
}

fn mouse_button(button: winit::event::MouseButton) -> Option<MouseButton> {
    match button {
        winit::event::MouseButton::Left => Some(MouseButton::Left),
        winit::event::MouseButton::Right => Some(MouseButton::Right),
        winit::event::MouseButton::Middle => Some(MouseButton::Middle),
        _ => None,
    }
}

/// Returns whether the pointer ended up captured. Platforms differ in which grab
/// they offer: macOS has only `Locked`, X11 only `Confined`.
fn set_pointer_lock(window: &winit::window::Window, lock: bool) -> bool {
    window.set_cursor_visible(!lock);
    if !lock {
        let _ = window.set_cursor_grab(CursorGrabMode::None);
        return false;
    }
    window
        .set_cursor_grab(CursorGrabMode::Locked)
        .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
        .is_ok()
}

/// Open a window and run `app` until it closes or a system inserts [`Exit`].
///
/// # Panics
///
/// If the window or a GPU for it cannot be had.
pub fn run(mut app: App, options: WindowOptions) {
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title(options.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(options.width, options.height))
            .build(&event_loop)
            .expect("window"),
    );

    let mut view = match WindowRenderer::with_backend(window.clone(), options.backend) {
        Ok(view) => view,
        Err(error) => panic!("no {:?} renderer for the window: {error}", options.backend),
    };
    view.set_msaa(options.msaa);
    view.set_pixel_ratio(window.scale_factor() as f32);

    if app.world.get_resource::<MainCamera>().is_none() {
        let camera = PerspectiveCamera::new(50.0, view.aspect(), 0.1, 1000.0);
        app.world.insert_resource(MainCamera(camera));
    }

    if app.world.get_resource::<Hud>().is_none() {
        let (width, height) = view.size();
        app.world.insert_resource(Hud::new(width, height));
    }

    #[cfg(feature = "gamepad")]
    let mut gamepads = match crate::gamepad::Gamepads::new() {
        Ok(gamepads) => Some(gamepads),
        Err(error) => {
            eprintln!("no gamepads: {error}");
            None
        }
    };
    #[cfg(feature = "gamepad")]
    let mut focused = window.has_focus();

    let mut last = Instant::now();
    let mut locked = false;
    let win = window.clone();
    event_loop
        .run(move |event, target| {
            let (event, window_id) = match event {
                Event::WindowEvent { event, window_id } => (event, window_id),
                // Motion with no position, which is the point of a locked pointer.
                Event::DeviceEvent {
                    event: DeviceEvent::MouseMotion { delta },
                    ..
                } if locked => {
                    let delta = Vector2::new(delta.0 as f32, delta.1 as f32);
                    app.world.resource_mut::<Input>().add_mouse_motion(delta);
                    return;
                }
                _ => return,
            };
            if window_id != win.id() {
                return;
            }
            let input = app.world.resource_mut::<Input>();
            match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(size) => {
                    view.resize(size.width, size.height);
                    let (width, height) = view.size();
                    app.world.resource_mut::<Hud>().0.resize(width, height);
                }
                WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    view.set_pixel_ratio(scale_factor as f32);
                }
                WindowEvent::Focused(has_focus) => {
                    #[cfg(feature = "gamepad")]
                    {
                        focused = has_focus;
                    }
                    if !has_focus {
                        input.release_all();
                    }
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        apply_key(input, code, event.state == ElementState::Pressed);
                    }
                    if event.state == ElementState::Pressed {
                        if let Some(text) = event
                            .text
                            .as_ref()
                            .filter(|t| !t.chars().any(char::is_control))
                        {
                            input.type_text(text);
                        }
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if let Some(button) = mouse_button(button) {
                        match state {
                            ElementState::Pressed => input.press(button),
                            ElementState::Released => input.release(button),
                        }
                    }
                }
                WindowEvent::Touch(touch) => {
                    let at = Vector2::new(touch.location.x as f32, touch.location.y as f32);
                    match touch.phase {
                        TouchPhase::Started => input.touch_start(touch.id, at),
                        TouchPhase::Moved => input.touch_move(touch.id, at),
                        TouchPhase::Ended | TouchPhase::Cancelled => input.touch_end(touch.id),
                    }
                }
                WindowEvent::CursorMoved { position, .. } if !locked => {
                    input.move_mouse(Vector2::new(position.x as f32, position.y as f32));
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    input.scroll += match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                    };
                }
                WindowEvent::RedrawRequested => {
                    #[cfg(feature = "gamepad")]
                    if let Some(gamepads) = &mut gamepads {
                        gamepads.poll(app.world.resource_mut::<Input>(), focused);
                    }
                    let now = Instant::now();
                    app.frame(now.duration_since(last).as_secs_f64());
                    last = now;

                    let wanted = app.world.resource::<Input>().lock_pointer;
                    if wanted != locked {
                        locked = set_pointer_lock(&win, wanted);
                        app.world.resource_mut::<Input>().pointer_locked = locked;
                    }

                    let aspect = view.aspect();
                    let world = &mut app.world;
                    world.resource_mut::<MainCamera>().0.aspect = aspect;
                    world.with_resource::<MainCamera, _>(|w, camera| {
                        w.with_resource::<Hud, _>(|w, hud| {
                            view.render_frame(&mut w.scene, &camera.0, None, Some(&mut hud.0));
                        });
                    });
                    if world.get_resource::<Exit>().is_some() {
                        target.exit();
                    }
                    win.request_redraw();
                }
                _ => {}
            }
        })
        .expect("event loop");
}
