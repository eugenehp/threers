//! A browser canvas around an [`App`].
//!
//! The browser's counterpart of the `window` module: a surface on the canvas,
//! `requestAnimationFrame` for the loop, and DOM events and the Gamepad API for
//! [`Input`]. With the `audio` feature, sound goes through the same
//! [`Mixer`](crate::audio::Mixer) as a native build.

use crate::app::{App, Exit, MainCamera};
use crate::hud::Hud;
use crate::input::{Input, Key, MouseButton, PadAxis, PadButton, PadEvent, PadTracker};
use crate::tasks::{Task, TaskFailed};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use threers::cameras::PerspectiveCamera;
use threers::math::Vector2;
use threers::Renderer;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{HtmlCanvasElement, KeyboardEvent, MouseEvent, TouchEvent, WheelEvent};

/// Which graphics API the browser gave, as a resource: `"BrowserWebGpu"` or `"Gl"`.
#[derive(Debug, Clone)]
pub struct GraphicsBackend(pub String);

/// What the browser's audio callback has produced, as a resource: how many
/// frames, and the loudest sample so far. Tells a page that sound really flowed.
#[cfg(feature = "audio")]
#[derive(Clone, Default)]
pub struct WebAudioMeter(pub Rc<std::cell::Cell<(u64, f32)>>);

struct Runner {
    app: App,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: Arc<wgpu::Device>,
    renderer: Renderer,
    canvas: HtmlCanvasElement,
    pads: PadTracker,
    /// Connected pad slots last frame, to notice one going.
    slots: Vec<bool>,
    last: f64,
}

fn now() -> f64 {
    web_sys::window()
        .and_then(|window| window.performance())
        .map_or(0.0, |performance| performance.now())
}

fn pixel_ratio() -> f64 {
    web_sys::window().map_or(1.0, |window| window.device_pixel_ratio())
}

/// Run `app` on `canvas` until the page goes away or a system inserts [`Exit`].
/// Resolves once the first frame is scheduled.
///
/// # Errors
///
/// If the browser offers neither WebGPU nor WebGL2.
pub async fn run(mut app: App, canvas: HtmlCanvasElement) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let (width, height) = (canvas.width().max(1), canvas.height().max(1));
    let (_instance, surface, adapter) = threers::wasm::open_browser_gpu(&canvas).await?;
    let (device, queue) = threers::wasm::acquire_browser_gpu(&adapter).await?;
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|format| format.is_srgb())
        .unwrap_or(caps.formats[0]);
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: threers::wasm::pick_surface_alpha_mode(&caps.alpha_modes),
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Srgb,
    };
    surface.configure(&device, &config);
    let renderer = Renderer::new(device.clone(), queue, format, width, height);
    app.world
        .insert_resource(GraphicsBackend(format!("{:?}", adapter.get_info().backend)));

    if app.world.get_resource::<MainCamera>().is_none() {
        let aspect = width as f32 / height as f32;
        let camera = PerspectiveCamera::new(50.0, aspect, 0.1, 1000.0);
        app.world.insert_resource(MainCamera(camera));
    }
    if app.world.get_resource::<Hud>().is_none() {
        app.world.insert_resource(Hud::new(width, height));
    }
    #[cfg(feature = "audio")]
    let unlock = start_audio(&mut app)?;
    #[cfg(not(feature = "audio"))]
    let unlock = Rc::new(|| ());

    let runner = Rc::new(RefCell::new(Runner {
        app,
        surface,
        config,
        device,
        renderer,
        canvas: canvas.clone(),
        pads: PadTracker::default(),
        slots: Vec::new(),
        last: now(),
    }));
    listen(&canvas, &runner, unlock)?;
    start_frames(runner);
    Ok(())
}

fn on<E: JsCast + 'static>(
    target: &web_sys::EventTarget,
    event: &str,
    mut handler: impl FnMut(E) + 'static,
) -> Result<(), JsValue> {
    let closure = Closure::<dyn FnMut(web_sys::Event)>::new(move |event: web_sys::Event| {
        if let Ok(event) = event.dyn_into::<E>() {
            handler(event);
        }
    });
    target.add_event_listener_with_callback(event, closure.as_ref().unchecked_ref())?;
    // Listeners last as long as the page, so the closure has to as well.
    closure.forget();
    Ok(())
}

fn mouse_button(button: i16) -> Option<MouseButton> {
    match button {
        0 => Some(MouseButton::Left),
        1 => Some(MouseButton::Middle),
        2 => Some(MouseButton::Right),
        _ => None,
    }
}

fn listen(
    canvas: &HtmlCanvasElement,
    runner: &Rc<RefCell<Runner>>,
    unlock: Rc<dyn Fn()>,
) -> Result<(), JsValue> {
    let window: web_sys::EventTarget = web_sys::window().ok_or("no window")?.into();
    let surface: &web_sys::EventTarget = canvas.as_ref();

    let (r, sound) = (runner.clone(), unlock.clone());
    on::<KeyboardEvent>(&window, "keydown", move |event| {
        sound();
        if let Some(key) = Key::from_code(&event.code()) {
            // A game's space bar should not also scroll the page.
            event.prevent_default();
            if !event.repeat() {
                r.borrow_mut().app.world.resource_mut::<Input>().press(key);
            }
        }
        // A printable key's character, for typing; not with a shortcut held.
        let key = event.key();
        if key.chars().count() == 1 && !event.ctrl_key() && !event.meta_key() {
            r.borrow_mut()
                .app
                .world
                .resource_mut::<Input>()
                .type_text(&key);
        }
    })?;
    let r = runner.clone();
    on::<KeyboardEvent>(&window, "keyup", move |event| {
        if let Some(key) = Key::from_code(&event.code()) {
            r.borrow_mut()
                .app
                .world
                .resource_mut::<Input>()
                .release(key);
        }
    })?;
    let r = runner.clone();
    on::<web_sys::Event>(&window, "blur", move |_| {
        r.borrow_mut()
            .app
            .world
            .resource_mut::<Input>()
            .release_all();
    })?;

    let (r, sound, surface_canvas) = (runner.clone(), unlock.clone(), canvas.clone());
    on::<MouseEvent>(surface, "mousedown", move |event| {
        sound();
        let mut runner = r.borrow_mut();
        let input = runner.app.world.resource_mut::<Input>();
        // A browser grants this only from a user gesture, which is why it is here
        // and not in the frame loop.
        if input.lock_pointer && !input.pointer_locked {
            surface_canvas.request_pointer_lock();
        }
        if let Some(button) = mouse_button(event.button()) {
            input.press(button);
        }
    })?;
    let sound = unlock;
    on::<TouchEvent>(surface, "touchend", move |_| sound())?;
    let r = runner.clone();
    on::<MouseEvent>(&window, "mouseup", move |event| {
        if let Some(button) = mouse_button(event.button()) {
            r.borrow_mut()
                .app
                .world
                .resource_mut::<Input>()
                .release(button);
        }
    })?;
    let r = runner.clone();
    on::<MouseEvent>(surface, "mousemove", move |event| {
        let ratio = pixel_ratio() as f32;
        let mut runner = r.borrow_mut();
        let input = runner.app.world.resource_mut::<Input>();
        if input.pointer_locked {
            let delta = Vector2::new(event.movement_x() as f32, event.movement_y() as f32);
            input.add_mouse_motion(delta);
        } else {
            let at = Vector2::new(
                event.offset_x() as f32 * ratio,
                event.offset_y() as f32 * ratio,
            );
            input.move_mouse(at);
        }
    })?;
    let r = runner.clone();
    let document: web_sys::EventTarget = web_sys::window()
        .and_then(|window| window.document())
        .ok_or("no document")?
        .into();
    on::<web_sys::Event>(&document, "pointerlockchange", move |_| {
        let mut runner = r.borrow_mut();
        let locked = web_sys::window()
            .and_then(|window| window.document())
            .and_then(|document| document.pointer_lock_element())
            .is_some();
        runner.app.world.resource_mut::<Input>().pointer_locked = locked;
    })?;
    for name in ["touchstart", "touchmove", "touchend", "touchcancel"] {
        let r = runner.clone();
        let canvas = canvas.clone();
        on::<TouchEvent>(surface, name, move |event| {
            // Keeps the page from scrolling or zooming under the player's thumbs.
            event.prevent_default();
            let ratio = pixel_ratio() as f32;
            let bounds = canvas.get_bounding_client_rect();
            let mut runner = r.borrow_mut();
            let input = runner.app.world.resource_mut::<Input>();
            let changed = event.changed_touches();
            for index in 0..changed.length() {
                let Some(touch) = changed.item(index) else {
                    continue;
                };
                let id = touch.identifier() as u64;
                let at = Vector2::new(
                    (touch.client_x() as f64 - bounds.left()) as f32 * ratio,
                    (touch.client_y() as f64 - bounds.top()) as f32 * ratio,
                );
                match name {
                    "touchstart" => input.touch_start(id, at),
                    "touchmove" => input.touch_move(id, at),
                    _ => input.touch_end(id),
                }
            }
        })?;
    }
    let r = runner.clone();
    on::<WheelEvent>(surface, "wheel", move |event| {
        r.borrow_mut().app.world.resource_mut::<Input>().scroll -= (event.delta_y() / 100.0) as f32;
    })?;
    Ok(())
}

fn start_frames(runner: Rc<RefCell<Runner>>) {
    let next: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let again = next.clone();
    *next.borrow_mut() = Some(Closure::new(move || {
        if runner.borrow_mut().frame() {
            request_frame(again.borrow().as_ref());
        }
    }));
    request_frame(next.borrow().as_ref());
}

fn request_frame(callback: Option<&Closure<dyn FnMut()>>) {
    if let (Some(window), Some(callback)) = (web_sys::window(), callback) {
        let _ = window.request_animation_frame(callback.as_ref().unchecked_ref());
    }
}

impl Runner {
    /// One frame. False when the game has asked to stop.
    fn frame(&mut self) -> bool {
        self.fit_canvas();
        self.poll_pads();

        let at = now();
        self.app.frame((at - self.last) / 1000.0);
        self.last = at;

        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        self.app.world.resource_mut::<MainCamera>().0.aspect = aspect;
        self.draw();
        self.app.world.get_resource::<Exit>().is_none()
    }

    /// Follow the canvas's size on the page, at the display's density.
    fn fit_canvas(&mut self) {
        let ratio = pixel_ratio();
        let width = ((self.canvas.client_width() as f64 * ratio) as u32).max(1);
        let height = ((self.canvas.client_height() as f64 * ratio) as u32).max(1);
        if (width, height) == (self.config.width, self.config.height) {
            return;
        }
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.renderer.resize(width, height);
        self.app.world.resource_mut::<Hud>().0.resize(width, height);
    }

    fn draw(&mut self) {
        use wgpu::CurrentSurfaceTexture as Acquired;
        let frame = match self.surface.get_current_texture() {
            Acquired::Success(frame) | Acquired::Suboptimal(frame) => frame,
            _ => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let format = self.config.format;
        let renderer = &mut self.renderer;
        self.app.world.with_resource::<MainCamera, _>(|w, camera| {
            renderer.render(&mut w.scene, &camera.0, &view, false);
            if let Some(hud) = w.get_resource_mut::<Hud>() {
                if hud.0.touched() {
                    renderer.draw_overlay(&mut hud.0, &view, format);
                }
            }
        });
        self.renderer.queue_arc().present(frame);
    }

    /// The Gamepad API has no events for buttons, so read every pad each frame.
    fn poll_pads(&mut self) {
        let Some(window) = web_sys::window() else {
            return;
        };
        let Ok(pads) = window.navigator().get_gamepads() else {
            return;
        };
        self.slots.resize(pads.length() as usize, false);
        for slot in 0..pads.length() {
            let pad = pads.get(slot).dyn_into::<web_sys::Gamepad>().ok();
            let index = slot as usize;
            match pad {
                Some(pad) if pad.connected() => {
                    self.slots[index] = true;
                    for event in standard_events(&pad) {
                        self.pads.update(index, event);
                    }
                }
                _ => {
                    if std::mem::take(&mut self.slots[index]) {
                        self.pads.update(index, PadEvent::Disconnected);
                    }
                }
            }
        }
        let focused = window
            .document()
            .and_then(|document| document.has_focus().ok())
            .unwrap_or(true);
        if focused {
            self.pads.sync(self.app.world.resource_mut::<Input>());
        }
        if let Some(rumble) = self.app.world.resource_mut::<Input>().take_rumble() {
            for slot in 0..pads.length() {
                if let Ok(pad) = pads.get(slot).dyn_into::<web_sys::Gamepad>() {
                    play_rumble(&pad, rumble);
                }
            }
        }
    }
}

/// `vibrationActuator` is not in every browser, and not in `web-sys`, so it is
/// reached by name. A pad without one is simply left alone.
fn play_rumble(pad: &web_sys::Gamepad, rumble: crate::input::Rumble) {
    use js_sys::{Function, Object, Reflect};
    let Ok(actuator) = Reflect::get(pad, &"vibrationActuator".into()) else {
        return;
    };
    let Ok(play) = Reflect::get(&actuator, &"playEffect".into()) else {
        return;
    };
    let Ok(play) = play.dyn_into::<Function>() else {
        return;
    };
    let options = Object::new();
    let duration = rumble.seconds as f64 * 1000.0;
    let _ = Reflect::set(&options, &"duration".into(), &duration.into());
    let strength = rumble.strength as f64;
    let _ = Reflect::set(&options, &"strongMagnitude".into(), &strength.into());
    let _ = play.call2(&actuator, &"dual-rumble".into(), &options);
}

/// `standard_mapping` of the Gamepad spec, by button number.
const BUTTONS: [PadButton; 16] = [
    PadButton::South,
    PadButton::East,
    PadButton::West,
    PadButton::North,
    PadButton::LeftShoulder,
    PadButton::RightShoulder,
    PadButton::LeftTrigger,
    PadButton::RightTrigger,
    PadButton::Select,
    PadButton::Start,
    PadButton::LeftThumb,
    PadButton::RightThumb,
    PadButton::DPadUp,
    PadButton::DPadDown,
    PadButton::DPadLeft,
    PadButton::DPadRight,
];

fn standard_events(pad: &web_sys::Gamepad) -> Vec<PadEvent> {
    let mut events = Vec::new();
    let buttons = pad.buttons();
    for (index, button) in BUTTONS.iter().enumerate() {
        let Ok(state) = buttons
            .get(index as u32)
            .dyn_into::<web_sys::GamepadButton>()
        else {
            continue;
        };
        events.push(if state.pressed() {
            PadEvent::Pressed(*button)
        } else {
            PadEvent::Released(*button)
        });
        match button {
            PadButton::LeftTrigger => {
                events.push(PadEvent::Axis(PadAxis::LeftTrigger, state.value() as f32));
            }
            PadButton::RightTrigger => {
                events.push(PadEvent::Axis(PadAxis::RightTrigger, state.value() as f32));
            }
            _ => {}
        }
    }
    let axes = pad.axes();
    let axis = |index: u32| axes.get(index).as_f64().unwrap_or(0.0) as f32;
    // The page's Y runs down; `PadAxis` has up positive.
    events.push(PadEvent::Axis(PadAxis::LeftX, axis(0)));
    events.push(PadEvent::Axis(PadAxis::LeftY, -axis(1)));
    events.push(PadEvent::Axis(PadAxis::RightX, axis(2)));
    events.push(PadEvent::Axis(PadAxis::RightY, -axis(3)));
    events
}

/// Read a URL into the asset channel.
pub(crate) fn fetch_bytes(
    url: String,
    path: String,
    parse: fn(&[u8]) -> Result<Arc<dyn std::any::Any + Send + Sync>, String>,
    sender: Sender<crate::assets::Arrival>,
) {
    wasm_bindgen_futures::spawn_local(async move {
        let bytes = fetch(&url).await.map_err(|e| format!("{url}: {e:?}"));
        let _ = sender.send((path, bytes.and_then(|bytes| parse(&bytes))));
    });
}

/// The worker's whole program: wait for a job, run it, send the bytes back.
const WORKER: &str = r#"
import init, { run_job } from "MODULE";
const ready = init();
onmessage = async ({ data: { id, name, input } }) => {
  await ready;
  try {
    const output = run_job(name, input);
    postMessage({ id, output }, [output.buffer]);
  } catch (error) {
    postMessage({ id, error: String(error) });
  }
};
"#;

type Pending = Rc<RefCell<std::collections::HashMap<u32, Sender<Result<Vec<u8>, TaskFailed>>>>>;

/// Web Workers running [`Jobs`](crate::tasks::Jobs). Each is its own instance of the
/// game's wasm, so this works in an ordinary build, with no atomics or special
/// headers. The game's package has to export the entry point, which is one line:
///
/// ```ignore
/// #[wasm_bindgen]
/// pub fn run_job(name: &str, input: &[u8]) -> Result<Vec<u8>, JsValue> {
///     my_jobs().run_here(name, input).map_err(|e| e.to_string().into())
/// }
/// ```
pub struct WorkerPool {
    workers: Vec<web_sys::Worker>,
    /// Jobs out on each worker, to send the next to the least busy.
    load: Rc<RefCell<Vec<u32>>>,
    pending: Pending,
    next: std::cell::Cell<u32>,
}

impl WorkerPool {
    /// Start `count` workers, each loading the game's JS glue from `module`, a URL
    /// relative to the page.
    ///
    /// # Errors
    ///
    /// If the browser refuses a worker.
    pub fn new(module: &str, count: u32) -> Result<Self, JsValue> {
        let window = web_sys::window().ok_or("no window")?;
        let base = window.location().href()?;
        let module = web_sys::Url::new_with_base(module, &base)?.href();
        let code = js_sys::Array::of1(&WORKER.replace("MODULE", &module).into());
        let options = web_sys::BlobPropertyBag::new();
        options.set_type("text/javascript");
        let blob = web_sys::Blob::new_with_str_sequence_and_options(&code, &options)?;
        let url = web_sys::Url::create_object_url_with_blob(&blob)?;

        let load = Rc::new(RefCell::new(vec![0; count.max(1) as usize]));
        let pending: Pending = Rc::default();
        let mut workers = Vec::new();
        for index in 0..count.max(1) as usize {
            let options = web_sys::WorkerOptions::new();
            options.set_type(web_sys::WorkerType::Module);
            let worker = web_sys::Worker::new_with_options(&url, &options)?;

            let (done_load, done_pending) = (load.clone(), pending.clone());
            let reply = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(
                move |event: web_sys::MessageEvent| {
                    done_load.borrow_mut()[index] -= 1;
                    finish(&done_pending, event.data());
                },
            );
            worker.set_onmessage(Some(reply.as_ref().unchecked_ref()));
            reply.forget();

            // A worker that fails to start would otherwise leave its jobs waiting.
            let lost = pending.clone();
            let failed = Closure::<dyn FnMut(web_sys::ErrorEvent)>::new(
                move |event: web_sys::ErrorEvent| {
                    for (_, sender) in lost.borrow_mut().drain() {
                        let _ = sender.send(Err(TaskFailed(event.message())));
                    }
                },
            );
            worker.set_onerror(Some(failed.as_ref().unchecked_ref()));
            failed.forget();
            workers.push(worker);
        }
        Ok(Self {
            workers,
            load,
            pending,
            next: std::cell::Cell::new(0),
        })
    }

    /// Run a job on the least busy worker.
    pub fn run(&self, name: &str, input: &[u8]) -> Task<Vec<u8>> {
        let (sender, task) = crate::tasks::task();
        let id = self.next.get();
        self.next.set(id.wrapping_add(1));

        let index = {
            let load = self.load.borrow();
            (0..load.len()).min_by_key(|&i| load[i]).unwrap_or(0)
        };
        let message = js_sys::Object::new();
        let set = |key: &str, value: &JsValue| js_sys::Reflect::set(&message, &key.into(), value);
        let sent = set("id", &id.into())
            .and_then(|_| set("name", &name.into()))
            .and_then(|_| set("input", &js_sys::Uint8Array::from(input).into()))
            .and_then(|_| self.workers[index].post_message(&message));
        match sent {
            Ok(()) => {
                self.load.borrow_mut()[index] += 1;
                self.pending.borrow_mut().insert(id, sender);
            }
            Err(error) => {
                let _ = sender.send(Err(TaskFailed(format!("{error:?}"))));
            }
        }
        task
    }
}

fn finish(pending: &Pending, data: JsValue) {
    let get = |key: &str| js_sys::Reflect::get(&data, &key.into()).unwrap_or(JsValue::UNDEFINED);
    let Some(id) = get("id").as_f64() else {
        return;
    };
    let Some(sender) = pending.borrow_mut().remove(&(id as u32)) else {
        return;
    };
    let error = get("error");
    let result = match error.as_string() {
        Some(message) => Err(TaskFailed(message)),
        None => Ok(js_sys::Uint8Array::new(&get("output")).to_vec()),
    };
    let _ = sender.send(result);
}

async fn fetch(url: &str) -> Result<Vec<u8>, JsValue> {
    let window = web_sys::window().ok_or("no window")?;
    let response: web_sys::Response = JsFuture::from(window.fetch_with_str(url))
        .await?
        .dyn_into()?;
    if !response.ok() {
        return Err(JsValue::from_str(&format!("HTTP {}", response.status())));
    }
    let buffer = JsFuture::from(response.array_buffer()?).await?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

/// The audio-thread half: a queue of stereo blocks, played in order.
#[cfg(feature = "audio")]
const WORKLET: &str = r#"
class ThreersOutput extends AudioWorkletProcessor {
  constructor() {
    super();
    this.blocks = [];
    this.at = 0;
    this.port.onmessage = e => this.blocks.push(e.data);
  }
  process(_, outputs) {
    const [left, right] = outputs[0];
    for (let i = 0; i < left.length; i++) {
      const block = this.blocks[0];
      if (!block) { left[i] = 0; if (right) right[i] = 0; continue; }
      left[i] = block[this.at * 2];
      if (right) right[i] = block[this.at * 2 + 1];
      if (++this.at * 2 >= block.length) {
        this.blocks.shift();
        this.at = 0;
        this.port.postMessage(0);
      }
    }
    return true;
  }
}
registerProcessor("threers-output", ThreersOutput);
"#;

/// Frames in a block, and how many blocks to keep queued on the audio thread: about
/// 70 ms at 44.1 kHz, which is what a stalled frame can take before sound breaks up.
#[cfg(feature = "audio")]
const BLOCK_FRAMES: usize = 1024;
#[cfg(feature = "audio")]
const BLOCKS_AHEAD: u32 = 3;

/// Route sound through an `AudioWorklet`. The mixer stays in wasm on this thread
/// and sends finished blocks ahead, so the audio thread never waits on a frame.
/// Returns what to call on a user gesture, since a page may not start sound until
/// it has had one.
#[cfg(feature = "audio")]
fn start_audio(app: &mut App) -> Result<Rc<dyn Fn()>, JsValue> {
    use crate::audio::{AudioOut, Mixer};

    if app.world.get_resource::<AudioOut>().is_none() {
        return Ok(Rc::new(|| ()));
    }
    let context = web_sys::AudioContext::new()?;
    let mixer = Rc::new(RefCell::new(Mixer::new(context.sample_rate())));
    app.world.insert_resource(AudioOut::web(mixer.clone()));
    let meter = WebAudioMeter::default();
    app.world.insert_resource(meter.clone());

    let code = js_sys::Array::of1(&WORKLET.into());
    let options = web_sys::BlobPropertyBag::new();
    options.set_type("text/javascript");
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&code, &options)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)?;

    let started = context.clone();
    wasm_bindgen_futures::spawn_local(async move {
        if let Err(error) = connect_worklet(started, url, mixer, meter).await {
            web_sys::console::error_1(&error);
        }
    });
    Ok(Rc::new(move || {
        let _ = context.resume();
    }))
}

#[cfg(feature = "audio")]
async fn connect_worklet(
    context: web_sys::AudioContext,
    url: String,
    mixer: Rc<RefCell<crate::audio::Mixer>>,
    meter: WebAudioMeter,
) -> Result<(), JsValue> {
    JsFuture::from(context.audio_worklet()?.add_module(&url)?).await?;
    let options = web_sys::AudioWorkletNodeOptions::new();
    let channels = js_sys::Array::of1(&2.into());
    options.set_output_channel_count(&channels);
    let node = web_sys::AudioWorkletNode::new_with_options(&context, "threers-output", &options)?;
    node.connect_with_audio_node(&context.destination())?;
    let port = node.port()?;

    let send = {
        let port = port.clone();
        move || {
            let mut block = vec![0.0_f32; BLOCK_FRAMES * 2];
            mixer.borrow_mut().render(&mut block, 2);
            let (count, peak) = meter.0.get();
            let loudest = block.iter().fold(peak, |peak, s| peak.max(s.abs()));
            meter.0.set((count + BLOCK_FRAMES as u64, loudest));
            let _ = port.post_message(&js_sys::Float32Array::from(block.as_slice()));
        }
    };
    for _ in 0..BLOCKS_AHEAD {
        send();
    }
    // Each block played asks for the next, so the queue stays the same length.
    let refill = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |_| send());
    port.set_onmessage(Some(refill.as_ref().unchecked_ref()));
    refill.forget();
    // Held by the page's audio graph from here on.
    std::mem::forget(node);
    Ok(())
}
