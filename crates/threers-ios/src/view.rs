//! Any [`App`] on a UIKit view's `CAMetalLayer` (or an AppKit one), drawn
//! by threers' Metal renderer, in one of two modes, switched while it runs
//! ([`View::set_mode`], or from the app with [`request`](crate::request)):
//!
//! - **window** — a flat screen: an iPhone or iPad, a window on Apple Vision
//!   Pro, a Mac. An orbit camera frames [`App::focus`]; one finger turns it,
//!   two pan it, a pinch zooms, and a tap is a select along its ray. There
//!   is no head, so [`Frame::head`] is `None`. The app moves the camera
//!   too ([`camera`](crate::camera): zoom, home, spin).
//! - **AR** — a phone's or iPad's camera, as ARKit tracks it: the session,
//!   and the camera feed behind the layer, are this crate's own
//!   (`ar`, iOS), so the shell does nothing for it. Elsewhere the camera
//!   can be handed over ([`View::set_ar_camera`]). The scene is drawn over
//!   the feed with no background. ARKit's world is the reference space:
//!   metres, +Y up, the origin where the session started. The head is the
//!   camera, and a finger on the screen is a transient pointer along the
//!   ray under it, as a pinch is on visionOS — so an app written for the
//!   headset runs here unchanged.
//!
//! Either way the app's UI ([`App::ui`]) is over the picture, and a finger
//! that comes down on it is the UI's ([`App::ui_event`]) until it lifts:
//! it scrolls the UI, and neither turns the camera nor points into the
//! scene.
//!
//! A frame ([`View::frame`]) may run on a thread of its own (the iOS shell
//! draws on a render thread, so the main one is free for UIKit and the
//! keyboard); the shell keeps every call into the view under one lock. A
//! change of mode touches UIKit (an AR session's camera view), so it is
//! [`View::apply_requests`]'s, on the main thread. [`App::update`] is the
//! app's per-frame work, and long work (downloads) belongs on threads of
//! its own.

use std::ffi::c_void;
use std::time::Instant;

use threers::cameras::{Camera, PerspectiveCamera};
use threers::controls::{OrbitControls, PointerEvent};
use threers::math::{Matrix4, Vector3};
use threers::metal::{MetalDevice, MetalRenderer, MetalSurface, RenderView};
use threers::scene::Scene;
use threers::ui::input::{Event, PointerButton, TouchPhase};
use threers::ui::native::{self, Control};
use threers::xr::{ray_pose, Frame, TargetRayMode, XrEvent, XrEventKind, XrInput};

use crate::gestures::{is_step, said, Inputs, KeyHeld, Pinch};
use crate::surface_ui::SurfaceUi;
use crate::{App, CameraRequest, Place, Request, Surface};

/// Which view: see the module docs.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Window = 0,
    Ar = 1,
}

thread_local!(static MARKS: std::cell::RefCell<(Option<Instant>, Vec<(&'static str, f64, u32)>)> = Default::default());

/// THREERS_PERF's stages: the time since the last mark, summed under `name`.
fn mark(name: &'static str) {
    MARKS.with(|m| {
        let mut m = m.borrow_mut();
        let now = Instant::now();
        let Some(last) = m.0 else { return };
        let ms = now.duration_since(last).as_secs_f64() * 1e3;
        m.0 = Some(now);
        match m.1.iter_mut().find(|e| e.0 == name) {
            Some(e) => {
                e.1 += ms;
                e.2 += 1;
            }
            None => m.1.push((name, ms, 1)),
        }
    });
}

fn mark_reset() {
    MARKS.with(|m| m.borrow_mut().0 = Some(Instant::now()));
}

/// Each stage's mean, ms, and forgotten.
fn mark_report() -> String {
    MARKS.with(|m| {
        let mut m = m.borrow_mut();
        let out = m.1.iter().map(|(n, t, k)| format!("{n} {:.1}", t / *k as f64)).collect::<Vec<_>>().join(", ");
        m.1.clear();
        out
    })
}

/// The share of the window's pixels (each way) a frame is drawn at while
/// the view moves (a gesture, a flight, a spin), and seconds still before
/// it is drawn whole again.
const MOVING_SCALE: f32 = 0.7;
const STILL: f32 = 0.15;

/// Seconds a reframe is flown in.
const FLIGHT: f32 = 0.6;

/// What [`View::set`] sets: a key and three floats, one calling
/// convention across the language boundary. An unknown key does nothing.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    /// The drawable resized: `a` × `b` pixels.
    Resize = 0,
    /// One finger: `a` its phase (0 began, 1 moved, 2 ended, 3 cancelled),
    /// `b`, `c` where it is, in pixels from the top left.
    Touch = 1,
    /// A pinch: `a` its scale since the last call (above 1 spreads); `b`,
    /// `c` where its fingers are, in pixels (0, 0: not said, the middle).
    /// `a` 0: the fingers lifted.
    Pinch = 2,
    /// Two fingers dragged: `a`, `b` pixels since the last call.
    Pan = 3,
    /// Pixels a point: `a` (the UI is laid out in points).
    Scale = 4,
    /// The safe area's insets, pixels: `a` top, `b` bottom, `c` either side.
    Insets = 5,
    /// A hardware keyboard's key: `a` its code point (the function keys as
    /// Apple's, Escape 27), `b` 1 with Shift held, `c` the others held (1
    /// Control, 2 Option, 4 Command). To [`App::key_with`].
    Keyboard = 6,
    /// The app at the front (`a` 1) or leaving it (0). To [`App::active`].
    Active = 7,
    /// The hardware keyboard's key last down, let go.
    KeyUp = 8,
}

/// A key's name as the web's `KeyboardEvent.key` gives it: `Escape`,
/// `ArrowUp`, …, else the character.
pub fn key_name(code: u32) -> Option<String> {
    Some(match code {
        27 => "Escape".into(),
        8 | 0x7F => "Backspace".into(),
        9 => "Tab".into(),
        0xF728 => "Delete".into(),
        13 | 3 => "Enter".into(),
        0xF700 => "ArrowUp".into(),
        0xF701 => "ArrowDown".into(),
        0xF702 => "ArrowLeft".into(),
        0xF703 => "ArrowRight".into(),
        0xF729 => "Home".into(),
        0xF72B => "End".into(),
        0xF72C => "PageUp".into(),
        0xF72D => "PageDown".into(),
        c => char::from_u32(c)?.to_string(),
    })
}

/// A finger moving less than this, in pixels, from where it went down is
/// a tap.
const TAP_SLOP: f32 = 12.0;
/// The window's MSAA.
const SAMPLES: u32 = 4;
/// A spin's pace, in drag pixels a second.
const SPIN: f32 = 60.0;

/// The finger on the screen.
struct Touch {
    down: (f32, f32),
    at: (f32, f32),
    /// It has gone further than a tap.
    dragged: bool,
    /// It came down on the UI, which has it until it lifts.
    ui: bool,
}

/// The AR camera as one eye.
struct ArEye {
    /// Camera to world (ARKit's camera transform, for the interface
    /// orientation).
    world: Matrix4,
    /// The projection, in this renderer's 0..1 depth.
    projection: [f32; 16],
}

/// An app on a layer. Made by [`View::new`]; driven by the C entry points
/// [`app!`](crate::app) exports.
pub struct View<A: App> {
    renderer: MetalRenderer,
    /// The scene drawn offscreen, on the window's device: the window's
    /// frame when the app asks for ambient occlusion (its SSAO), a
    /// snapshot, a recorded turn's frames. Made once, resized as needed.
    offscreen: Option<threers::metal::MetalHeadlessRenderer>,
    /// A recorded turn's own (its size the video's, not the window's).
    recorder: Option<threers::metal::MetalHeadlessRenderer>,
    /// The window's frame while a gesture or a flight moves it: drawn at
    /// `MOVING_SCALE` of its pixels (the GPU fills them, not its triangles:
    /// a frame at half the pixels comes in two thirds of the time), sharp
    /// again the moment it stops.
    moving: Option<threers::metal::MetalHeadlessRenderer>,
    /// seconds since a gesture or a flight last moved the view
    still_for: f32,
    surface: MetalSurface,
    scene: Scene,
    app: A,
    ui: SurfaceUi,
    mode: Mode,
    /// The window's camera and its controls.
    camera: PerspectiveCamera,
    controls: OrbitControls,
    framed: bool,
    /// Seconds a gesture still counts as held: the gap between a pinch's
    /// or a pan's steps is not a release.
    held: f32,
    /// Input since the last frame, for the controls.
    turn: (f32, f32),
    pan: (f32, f32),
    zoom: f32,
    /// the pinch's fingers, pixels; and the point it zooms toward (what was
    /// under them when it began), with how far off its last step was
    pinch_at: Option<(f32, f32)>,
    pinch_anchor: Option<(Vector3, f32)>,
    /// The pinch going on, offered to the app.
    app_pinch: Pinch,
    /// What came since the last frame.
    inputs: Inputs,
    /// The key held, and since when it is timed.
    key_held: KeyHeld,
    key_clock: Instant,
    /// a reframe flown to: from (position, target), to, seconds in
    flight: Option<(Vector3, Vector3, Vector3, Vector3, f32)>,
    touch: Option<Touch>,
    /// The finger as a transient pointer (AR), or a tap's ray (window).
    pointer: Option<[f32; 16]>,
    events: Vec<XrEvent>,
    /// The tap's ray stays in `pointer` for the frame that reports it.
    release: bool,
    ar: Option<ArEye>,
    /// ARKit's session and camera feed, while in AR.
    #[cfg(all(target_os = "ios", target_arch = "aarch64"))]
    session: Option<crate::ar::ArSession>,
    /// The app's `CAMetalLayer`.
    layer_ptr: *mut c_void,
    /// Pixels a UI point, and the safe area (top, right, bottom, left) in
    /// pixels.
    scale: f32,
    /// The app's UI zoom (`App::ui_zoom`): UI pixels to points
    ui_zoom: f32,
    insets: [f32; 4],
    spin: bool,
    start: Instant,
    last: f64,
    /// The UI's controls as last drawn, for the shell's accessibility
    /// elements over them (VoiceOver, Voice Control, Switch Control), and
    /// the version last sent as JSON.
    native: Vec<Control>,
    native_sent: Option<u64>,
    native_json: String,
    /// The text control last typed into through the shell's native field,
    /// still focused: the next text goes straight to it (no click to focus
    /// it again). A touch on the UI forgets it.
    typing: Option<u64>,
}

impl<A: App> View<A> {
    /// The UI's controls as the last frame drew them, as JSON
    /// (`threers_ui::native::to_json`; rectangles in UI points), when they
    /// changed since the last call: the shell puts an accessibility element
    /// over each, and an activation comes back by its key
    /// ([`crate::threers_ios_press`]). `None` while they are as they were.
    pub fn controls(&mut self) -> Option<&str> {
        // (a zoom changed: the same controls at other points)
        let v = native::version(&self.native) ^ u64::from(self.ui_zoom.to_bits());
        if self.native_sent == Some(v) {
            return None;
        }
        self.native_sent = Some(v);
        // in points: the UI's pixels, zoomed
        let z = self.ui_zoom;
        let points: Vec<Control> = self
            .native
            .iter()
            .cloned()
            .map(|mut c| {
                c.rect = threers::ui::Rect { x: c.rect.x * z, y: c.rect.y * z, width: c.rect.width * z, height: c.rect.height * z };
                c.radius *= z;
                c
            })
            .collect();
        self.native_json = native::to_json(&points);
        Some(&self.native_json)
    }

    /// The app, set up, drawing into `layer` at `width × height` pixels.
    ///
    /// # Safety
    /// `layer` must be a live `CAMetalLayer` that outlives the view.
    pub unsafe fn new(
        layer: *mut c_void,
        width: u32,
        height: u32,
        mode: Mode,
    ) -> Result<Self, String> {
        let device = MetalDevice::new().map_err(|e| e.to_string())?;
        let samples = if device.supports_sample_count(SAMPLES) {
            SAMPLES
        } else {
            1
        };
        let surface = unsafe {
            MetalSurface::from_layer(&device, layer, width.max(1), height.max(1), samples)
        }
        .map_err(|e| e.to_string())?;
        let renderer = MetalRenderer::with_device(device).map_err(|e| e.to_string())?;
        let mut scene = Scene::new();
        let app = A::setup(&mut scene);
        let lens = app.lens();
        let mut camera = PerspectiveCamera::new(
            lens.fov.to_degrees(),
            width.max(1) as f32 / height.max(1) as f32,
            lens.near,
            lens.far,
        );
        camera.position = Vector3::new(0.0, 0.0, 3.0);
        camera.target = Vector3::ZERO;
        let mut controls = OrbitControls::new(&camera);
        controls.enable_damping = true;
        controls.screen_space_panning = true;
        let mut view = View {
            renderer,
            offscreen: None,
            recorder: None,
            moving: None,
            still_for: 1.0,
            surface,
            scene,
            app,
            ui: SurfaceUi::on_gpu(),
            mode: Mode::Window,
            camera,
            controls,
            framed: false,
            held: 0.0,
            turn: (0.0, 0.0),
            pan: (0.0, 0.0),
            zoom: 1.0,
            pinch_at: None,
            app_pinch: Pinch::default(),
            inputs: Inputs::default(),
            key_held: KeyHeld::default(),
            key_clock: Instant::now(),
            pinch_anchor: None,
            flight: None,
            touch: None,
            pointer: None,
            events: Vec::new(),
            release: false,
            ar: None,
            #[cfg(all(target_os = "ios", target_arch = "aarch64"))]
            session: None,
            layer_ptr: layer,
            scale: 2.0,
            ui_zoom: 1.0,
            insets: [0.0; 4],
            spin: false,
            start: Instant::now(),
            last: 0.0,
            native: vec![],
            native_sent: None,
            native_json: String::new(),
            typing: None,
        };
        crate::set_current_mode(Mode::Window);
        if mode == Mode::Ar && !view.set_mode(Mode::Ar) {
            return Err("no AR here".into());
        }
        Ok(view)
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Switch between the window and AR. On iOS, AR starts ARKit and puts
    /// its camera feed behind the layer; elsewhere the camera has to be
    /// handed over ([`set_ar_camera`](Self::set_ar_camera)). Whether the
    /// view is in `mode` now.
    pub fn set_mode(&mut self, mode: Mode) -> bool {
        if mode == self.mode {
            return true;
        }
        if mode == Mode::Ar {
            #[cfg(all(target_os = "ios", target_arch = "aarch64"))]
            match unsafe { crate::ar::ArSession::start(self.layer_ptr) } {
                Some(s) => self.session = Some(s),
                None => return false,
            }
        } else {
            #[cfg(all(target_os = "ios", target_arch = "aarch64"))]
            {
                self.session = None;
            }
            self.ar = None;
            self.scene.background_alpha = 1.0;
            // frame the app's focus again
            self.framed = false;
        }
        // the layer composites see-through only when it is not opaque
        unsafe { crate::set_layer_opaque(self.layer_ptr, mode == Mode::Window) };
        self.mode = mode;
        self.touch = None;
        self.pointer = None;
        crate::set_current_mode(mode);
        true
    }

    /// Set one thing; see [`Key`].
    pub fn set(&mut self, key: u32, a: f32, b: f32, c: f32) {
        // fingers and keys wait for the frame, merged
        if Inputs::waits(key) {
            self.inputs.push((key, a, b, c));
        } else {
            self.apply(key, a, b, c);
        }
    }

    fn apply(&mut self, key: u32, a: f32, b: f32, c: f32) {
        // THREERS_TOUCH_LOG: each finger, pinch and pan as it comes, and
        // where it went (the UI or the view)
        static LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *LOG.get_or_init(|| std::env::var_os("THREERS_TOUCH_LOG").is_some()) && (1..=3).contains(&key) {
            eprintln!("touchlog key {key} a {a:.3} b {b:.1} c {c:.1} on_ui {} zoom {} scale {}", self.on_ui(), self.ui_zoom, self.scale);
        }
        match key {
            k if k == Key::Resize as u32 => {
                let (w, h) = ((a as u32).max(1), (b as u32).max(1));
                if let Err(e) = self.surface.resize(w, h) {
                    eprintln!("threers: resize: {e}");
                }
                self.camera.aspect = w as f32 / h as f32;
            }
            k if k == Key::Touch as u32 => self.touch(a as u32, (b, c)),
            // a pinch the app takes (zooming a page); over, with no scale
            k if k == Key::Pinch as u32 && a == 0.0 => self.pinch_over(),
            k if k == Key::Pinch as u32 && is_step(a) => {
                if !self.pinch_app(a, (b, c)) && !self.on_ui() {
                    self.zoom *= a;
                    self.pinch_at = said((b, c)).then_some((b, c));
                }
            }
            k if k == Key::Pan as u32 && !self.on_ui() => {
                self.pan.0 += a;
                self.pan.1 += b;
            }
            k if k == Key::Scale as u32 && a.is_finite() && a > 0.0 => self.scale = a,
            k if k == Key::Insets as u32 => {
                self.insets = [a, c, b, c].map(|v| if v.is_finite() { v.max(0.0) } else { 0.0 })
            }
            k if k == Key::Active as u32 => {
                self.key_held.up();
                self.app.active(a > 0.5);
            }
            k if k == Key::KeyUp as u32 => self.key_held.up(),
            k if k == Key::Keyboard as u32 && a.is_finite() && a >= 0.0 => {
                if let Some(name) = key_name(a as u32) {
                    let held = crate::KeyModifiers::from_shell(b, c);
                    self.key_held.down(&name, held, self.key_clock.elapsed().as_secs_f64());
                    self.app.key_with(&name, held);
                }
            }
            _ => {}
        }
    }

    /// The offscreen renderer, on the window's device, at `w` × `h`, set
    /// up as the app asks — its tone mapping, and its ambient occlusion
    /// for `camera` — for a frame from it.
    fn draw_offscreen<T>(
        &mut self,
        record: bool,
        w: u32,
        h: u32,
        camera: &threers::cameras::PerspectiveCamera,
        draw: impl FnOnce(&mut threers::metal::MetalHeadlessRenderer, &mut Scene) -> Result<T, threers::metal::MetalError>,
    ) -> Result<T, String> {
        let ao = self.app.ambient_occlusion();
        let (tone, exposure) = self.app.tone_mapping();
        let device = self.renderer.device().clone();
        let slot = if record { &mut self.recorder } else { &mut self.offscreen };
        let r = match slot {
            Some(r) => r,
            slot => {
                slot.insert(threers::metal::MetalHeadlessRenderer::builder().size(w, h).msaa(SAMPLES).build_on(&device).map_err(|e| format!("{e:?}"))?)
            }
        };
        r.resize(w, h).map_err(|e| format!("{e:?}"))?;
        r.renderer_mut().set_tone_mapping(tone, exposure);
        r.set_ssao(ao.map(|(radius, strength)| threers::metal::SsaoSettings {
            strength,
            radius,
            near: camera.near,
            far: camera.far,
            tan_half_fov: (camera.fov * 0.5).tan(),
            aspect: w as f32 / h.max(1) as f32,
        }));
        draw(r, &mut self.scene).map_err(|e| format!("{e:?}"))
    }

    /// The next frame of a turn being recorded ([`crate::record_turn`]):
    /// the camera turned about its target, `made / frames` of a whole turn,
    /// drawn offscreen at the window's size (at most 1920 wide, even), as
    /// BGRA for the shell's video writer.
    fn record_frame(&mut self, w: u32, h: u32) {
        let Ok(mut rec) = crate::RECORDING.lock() else { return };
        let Some(r) = rec.as_mut() else { return };
        if crate::FRAME.lock().map_or(true, |f| f.is_some()) || r.made >= r.frames {
            return;
        }
        let k = (1920.0 / w as f32).min(1.0);
        let (vw, vh) = (((w as f32 * k) as u32) & !1, ((h as f32 * k) as u32) & !1);
        let angle = std::f32::consts::TAU * r.made as f32 / r.frames as f32;
        // Rodrigues: the camera's offset from its target turned about its up
        let mut cam = self.camera.clone();
        cam.aspect = vw as f32 / vh as f32;
        cam.view_offset = None;
        let (t, axis) = (cam.target, cam.up.normalize());
        let v = cam.position - t;
        let (s, c) = angle.sin_cos();
        let turned = v * c + axis.cross(v) * s + axis * (axis.dot(v) * (1.0 - c));
        cam.position = t + turned;
        drop(rec);
        let shot = self.draw_offscreen(true, vw, vh, &cam, |rr, scene| rr.render_to_rgba(scene, &cam));
        let Ok(mut rec) = crate::RECORDING.lock() else { return };
        let Some(r) = rec.as_mut() else { return };
        match shot {
            Ok(mut px) => {
                for p in px.as_chunks_mut::<4>().0 {
                    p.swap(0, 2);
                }
                let last = r.made + 1 == r.frames;
                if let Ok(mut f) = crate::FRAME.lock() {
                    *f = Some((px, vw, vh, r.made, last));
                }
                r.made += 1;
            }
            Err(e) => {
                eprintln!("threers: recording: {e:?}");
                *rec = None;
                drop(rec);
                self.recorder = None;
            }
        }
    }

    /// A finger is on the UI.
    fn on_ui(&self) -> bool {
        self.touch.as_ref().is_some_and(|t| t.ui)
    }

    fn time(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// A pinch's step (`step` its scale since the last, about the fingers'
    /// midpoint `at`, pixels) to the app ([`App::pinch`]). Whether the pinch
    /// is the app's. (No finger is down for it: the platform takes the first
    /// one's touch away as the second lands.)
    fn pinch_app(&mut self, step: f32, at: (f32, f32)) -> bool {
        let k = self.scale * self.ui_zoom;
        let (w, h) = self.surface.size();
        let middle = (w as f32 / 2.0 / k, h as f32 / 2.0 / k);
        let app = &mut self.app;
        self.app_pinch.step(step, (at.0 / k, at.1 / k), middle, |s, x, y, phase| app.pinch(s, x, y, phase))
    }

    /// The fingers lifted.
    fn pinch_over(&mut self) {
        let app = &mut self.app;
        self.app_pinch.over(|s, x, y, phase| app.pinch(s, x, y, phase));
    }

    /// The finger to the UI, in its points. Whether the UI took it.
    fn touch_ui(&mut self, phase: TouchPhase, at: (f32, f32)) -> bool {
        // the finger may move the focus: the next text clicks its control again
        self.typing = None;
        let k = self.scale * self.ui_zoom;
        let (x, y) = (at.0 / k, at.1 / k);
        let time = self.time();
        let took = self.app.ui_event(
            Place::Screen,
            Event::Touch {
                id: 0,
                phase,
                x,
                y,
                time,
            },
            time,
        );
        if std::env::var_os("THREERS_TOUCH_LOG").is_some() {
            eprintln!("touchlog ui {phase:?} at ({x:.1}, {y:.1}) took {took}");
        }
        took
    }

    /// `kind`, of the one finger.
    fn event(&mut self, kind: XrEventKind) {
        self.events.push(XrEvent { kind, slot: 0 });
    }

    fn touch(&mut self, phase: u32, at: (f32, f32)) {
        match (phase, self.touch.as_mut()) {
            (0, _) => {
                if self.touch_ui(TouchPhase::Start, at) {
                    self.touch = Some(Touch {
                        down: at,
                        at,
                        dragged: false,
                        ui: true,
                    });
                    return;
                }
                // not the UI's: it forgets the finger
                self.touch_ui(TouchPhase::Cancel, at);
                self.touch = Some(Touch {
                    down: at,
                    at,
                    dragged: false,
                    ui: false,
                });
                if self.mode == Mode::Ar {
                    self.pointer = self.ray_at(at);
                    self.event(XrEventKind::Connected);
                    self.event(XrEventKind::SelectStart);
                }
            }
            (1, Some(t)) if t.ui => {
                t.at = at;
                self.touch_ui(TouchPhase::Move, at);
            }
            (2 | 3, Some(t)) if t.ui => {
                self.touch = None;
                self.touch_ui(
                    if phase == 2 {
                        TouchPhase::End
                    } else {
                        TouchPhase::Cancel
                    },
                    at,
                );
            }
            (1, Some(t)) => {
                if self.mode == Mode::Window {
                    self.turn.0 += at.0 - t.at.0;
                    self.turn.1 += at.1 - t.at.1;
                }
                t.at = at;
                t.dragged |= (at.0 - t.down.0).hypot(at.1 - t.down.1) > TAP_SLOP;
            }
            (2 | 3, Some(t)) => {
                let tap = phase == 2 && !t.dragged;
                self.touch = None;
                if self.mode == Mode::Window {
                    if !tap {
                        return;
                    }
                    // A tap is a whole select, in one frame.
                    self.pointer = self.ray_at(at);
                    self.event(XrEventKind::Connected);
                    self.event(XrEventKind::SelectStart);
                }
                self.event(XrEventKind::SelectEnd);
                if tap {
                    self.event(XrEventKind::Select);
                }
                self.event(XrEventKind::Disconnected);
                self.release = true;
            }
            _ => {}
        }
    }

    /// The AR camera this frame: `world` its camera-to-world transform and
    /// `projection` its projection, column-major, both for the interface
    /// orientation and the view's size (ARKit's
    /// `viewMatrix(for:).inverse` and `projectionMatrix(for:viewportSize:zNear:zFar:)`).
    /// Only the projection's focal lengths and centre are read: its depth
    /// is the app's [`lens`](App::lens), in this renderer's convention.
    pub fn set_ar_camera(&mut self, world: [f32; 16], projection: [f32; 16]) {
        let lens = self.app.lens();
        self.ar = Some(ArEye {
            world: Matrix4 { elements: world },
            projection: with_depth(projection, lens.near, lens.far),
        });
    }

    /// What a pinch zooms toward: the nearest surface under its fingers, else
    /// the point at the target's depth under them, else the target.
    fn zoom_point(&mut self) -> Vector3 {
        let t = self.camera.target;
        let Some(at) = self.pinch_at else { return t };
        let (w, h) = self.surface.size();
        let ndc = threers::math::Vector2::new(2.0 * at.0 / w as f32 - 1.0, 1.0 - 2.0 * at.1 / h as f32);
        self.scene.update_world();
        let mut ray = threers::Raycaster::default();
        ray.first_hit_only = true;
        // (the camera's picture is shifted off a docked sidebar: undone for
        // the ray, its offset left out)
        let mut cam = self.camera.clone();
        cam.view_offset = None;
        ray.set_from_camera(ndc, &cam);
        let hits = ray.intersect_object(&self.scene.arena, self.scene.root, true);
        if let Some(hit) = hits.first() {
            return hit.point;
        }
        // none: at the target's depth along the ray
        let d = ray.ray.direction;
        let along = (t - ray.ray.origin).dot(self.camera.target - self.camera.position) / (self.camera.target - self.camera.position).length().max(1e-9);
        let cos = d.dot((self.camera.target - self.camera.position).normalize()).max(0.2);
        ray.ray.origin + d * (along / cos)
    }

    /// The ray from the camera through the pixel `at`, in the reference
    /// space, as a pose looking down it.
    fn ray_at(&self, at: (f32, f32)) -> Option<[f32; 16]> {
        let (w, h) = self.surface.size();
        let ndc = (2.0 * at.0 / w as f32 - 1.0, 1.0 - 2.0 * at.1 / h as f32);
        // The AR camera is already in the reference space (ARKit's world);
        // the window's camera is in the scene's world.
        let (view, projection, to_reference) = match (self.mode, &self.ar) {
            (Mode::Ar, Some(eye)) => (
                eye.world.invert(),
                Matrix4 {
                    elements: eye.projection,
                },
                Matrix4::identity(),
            ),
            (Mode::Ar, None) => return None,
            (Mode::Window, _) => (
                self.camera.view_matrix(),
                self.camera.projection_matrix(),
                self.app.origin().invert(),
            ),
        };
        let near = Vector3::new(ndc.0, ndc.1, 0.0)
            .unproject(&view, &projection)
            .apply_matrix4(&to_reference);
        let far = Vector3::new(ndc.0, ndc.1, 1.0)
            .unproject(&view, &projection)
            .apply_matrix4(&to_reference);
        Some(ray_pose(near, far - near))
    }

    /// Update the app and draw one frame. `false` when nothing was drawn
    /// (no drawable, or no AR camera yet).
    /// The app's requests for the window or AR: on the main thread (an AR
    /// session puts its camera view in the window's).
    pub fn apply_requests(&mut self) {
        match crate::take_view_request() {
            Some(Request::Window) => drop(self.set_mode(Mode::Window)),
            Some(Request::Ar) => drop(self.set_mode(Mode::Ar)),
            _ => {}
        }
    }

    pub fn frame(&mut self) -> bool {
        for (key, a, b, c) in self.inputs.take() {
            self.apply(key, a, b, c);
        }
        if let Some((name, held)) = self.key_held.due(self.key_clock.elapsed().as_secs_f64()) {
            self.app.key_repeat(&name, held);
        }
        // THREERS_PERF: each 120 frames, how long apart they came and what
        // building one took here (the GPU's share is the rest); the view
        // spun so frames keep coming. THREERS_NO_AO: without the app's
        // ambient occlusion, to compare.
        static PERF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*PERF.get_or_init(|| std::env::var_os("THREERS_PERF").is_some()) {
            return self.frame_inner();
        }
        self.spin = true;
        let t0 = Instant::now();
        mark_reset();
        let drew = self.frame_inner();
        mark("draw");
        let built = t0.elapsed().as_secs_f64() * 1e3;
        thread_local!(static STATS: std::cell::RefCell<(Option<Instant>, Vec<f64>, Vec<f64>)> = Default::default());
        STATS.with(|s| {
            let mut s = s.borrow_mut();
            if let Some(last) = s.0 {
                let gap = t0.duration_since(last).as_secs_f64() * 1e3;
                s.1.push(gap);
                s.2.push(built);
            }
            s.0 = Some(t0);
            if s.1.len() >= 120 {
                let med = |v: &mut Vec<f64>| {
                    v.sort_by(f64::total_cmp);
                    (v[v.len() / 2], v[v.len() * 95 / 100])
                };
                let (g, g95) = med(&mut s.1);
                let (b, b95) = med(&mut s.2);
                eprintln!("threers perf stages: {}", mark_report());
                if let Some(r) = self.offscreen.as_ref() {
                    let st = r.stats();
                    eprintln!("threers perf scene: {} draws, {} triangles, {} culled, {} uploads, size {:?}", st.draw_calls, st.triangles, st.culled, st.geometry_uploads, self.surface.size());
                }
                eprintln!("threers perf: frames {g:.1} ms apart (95% {g95:.1}), built in {b:.1} ms (95% {b95:.1}), ao {}", self.app.ambient_occlusion().is_some() && std::env::var_os("THREERS_NO_AO").is_none());
                s.1.clear();
                s.2.clear();
            }
        });
        drew
    }

    fn frame_inner(&mut self) -> bool {
        #[cfg(all(target_os = "ios", target_arch = "aarch64"))]
        if let Some(session) = &self.session {
            let lens = self.app.lens();
            if let Some((world, projection)) =
                session.camera(self.surface.size(), lens.near, lens.far)
            {
                self.set_ar_camera(world, projection);
            }
        }
        let time = self.time();
        let dt = (time - self.last).clamp(0.0, 0.25);
        self.last = time;
        if self.mode == Mode::Ar
            && let Some(t) = self.touch.as_ref().filter(|t| !t.ui)
        {
            self.pointer = self.ray_at(t.at);
        }
        let inputs: Vec<XrInput> = self
            .pointer
            .map(|ray| XrInput {
                slot: 0,
                target_ray_mode: TargetRayMode::TransientPointer,
                ray: Some(ray),
                ..Default::default()
            })
            .into_iter()
            .collect();
        let origin = self.app.origin();
        let frame = Frame {
            time,
            dt,
            head: match self.mode {
                Mode::Ar => self.ar.as_ref().map(|e| e.world.elements),
                Mode::Window => None,
            },
            inputs,
            events: std::mem::take(&mut self.events),
            see_through: self.mode == Mode::Ar,
        };
        if self.release {
            self.release = false;
            self.pointer = None;
        }
        if self.mode == Mode::Window {
            self.app.window_camera(&self.camera);
        }
        mark("before update");
        self.app.update(&mut self.scene, &frame);
        mark("update");
        let view = match self.mode {
            Mode::Window => {
                self.orbit(dt as f32);
                RenderView::from_camera(&self.camera)
            }
            Mode::Ar => {
                let Some(eye) = &self.ar else { return false };
                self.scene.background_alpha = 0.0;
                let world = origin.multiply(&eye.world);
                let mut v = RenderView::from_matrices(
                    world.invert(),
                    Matrix4 {
                        elements: eye.projection,
                    },
                );
                v.position =
                    Vector3::new(world.elements[12], world.elements[13], world.elements[14]);
                v
            }
        };
        let (w, h) = self.surface.size();
        // the UI in its pixels, the app's zoom of a point each
        self.ui_zoom = self.app.ui_zoom(w as f32 / self.scale, h as f32 / self.scale).clamp(0.5, 3.0);
        let k = self.scale * self.ui_zoom;
        let surface = Surface {
            place: Place::Screen,
            width: w as f32 / k,
            height: h as f32 / k,
            scale: k,
            insets: self.insets.map(|v| v / k),
            see_through: self.mode == Mode::Ar,
        };
        // A control activated by assistive technology (its element's
        // `accessibilityActivate`): a click at its middle, on what shows.
        let pressed: Vec<(f32, f32)> =
            crate::take_presses_for(0, |k| native::find(&self.native, k).is_some())
                .into_iter()
                .filter_map(|k| native::find(&self.native, k))
                .map(|c| c.centre())
                .collect();
        for (x, y) in pressed {
            let button = PointerButton::Primary;
            self.app
                .ui_event(Place::Screen, Event::PointerMove { x, y }, time);
            self.app
                .ui_event(Place::Screen, Event::PointerDown { x, y, button }, time);
            self.app
                .ui_event(Place::Screen, Event::PointerUp { x, y, button }, time);
            // and away: no pointer is left over it (no hover look stays)
            self.app
                .ui_event(Place::Screen, Event::PointerMove { x: -1e6, y: -1e6 }, time);
        }
        // typed into a native text field over a text control (the shell's,
        // for the on-screen keyboard): the control focused (a click on it),
        // then given the text — as a volume's surfaces do
        // Keystrokes come faster than frames: only each control's last text
        // counts, and a control still focused from the last is not clicked
        // again (a click rebuilt the UI on every key).
        let mut typed: Vec<(u64, String)> = vec![];
        for (k, t) in crate::take_texts_for(0, |k| native::find(&self.native, k).is_some()) {
            match typed.iter_mut().find(|e| e.0 == k) {
                Some(e) => e.1 = t,
                None => typed.push((k, t)),
            }
        }
        for (k, text) in typed {
            let Some((x, y)) = native::find(&self.native, k).map(|c| c.centre()) else { continue };
            if self.typing != Some(k) {
                let button = PointerButton::Primary;
                self.app
                    .ui_event(Place::Screen, Event::PointerDown { x, y, button }, time);
                self.app
                    .ui_event(Place::Screen, Event::PointerUp { x, y, button }, time);
            }
            self.typing = self.app.ui_text(Place::Screen, text).then_some(k);
        }
        mark("orbit and the rest");
        let (shown, changed) = self.ui.update(&mut self.app, &surface);
        mark("ui");
        // the controls again only when the UI's pixels changed (a walk of
        // the whole tree, 2–3 ms a frame on an iPad, for the same answer)
        if !shown {
            self.native.clear();
        } else if changed || self.native.is_empty() {
            self.native = self.app.ui(&surface).map(|ui| native::controls(ui)).unwrap_or_default();
        }
        let (tone, exposure) = self.app.tone_mapping();
        self.renderer.set_tone_mapping(tone, exposure);
        // a turn being recorded: its next frame, once the shell has the last
        if self.mode == Mode::Window {
            self.record_frame(w, h);
        }
        // a picture asked for: the view once more, offscreen
        if crate::take_snapshot() && self.mode == Mode::Window {
            let camera = self.camera.clone();
            match self.draw_offscreen(false, w, h, &camera, |r, scene| r.render_to_rgba(scene, &camera)) {
                Ok(rgba) => self.app.snapshot(w, h, rgba),
                Err(e) => eprintln!("threers: snapshot: {e}"),
            }
        }
        mark("record, snapshot");
        let Some(next) = self.surface.next_frame() else {
            return false;
        };
        mark("next drawable");
        let pass = next.attachments();
        // the app's ambient occlusion: the scene drawn offscreen with its
        // SSAO, on this device, then shown, the UI over it
        if self.mode == Mode::Window && self.app.ambient_occlusion().is_some() && std::env::var_os("THREERS_NO_AO").is_none() {
            let camera = self.camera.clone();
            // moving: the smaller frame (its own renderer, kept, so a
            // gesture's start allocates nothing), filtered up
            let moving = self.still_for < STILL;
            if moving {
                std::mem::swap(&mut self.offscreen, &mut self.moving);
            }
            let (dw, dh) = if moving { (((w as f32) * MOVING_SCALE).round() as u32, ((h as f32) * MOVING_SCALE).round() as u32) } else { (w, h) };
            let texture = self
                .draw_offscreen(false, dw, dh, &camera, |r, scene| r.render_to_texture(scene, &camera));
            if moving {
                std::mem::swap(&mut self.offscreen, &mut self.moving);
            }
            let copy = |r: &mut threers::metal::MetalRenderer, t, p: &_| if moving { r.draw_texture_scaled(t, p) } else { r.draw_texture(t, p) };
            match texture {
                Ok(t) => {
                    let drawn = if shown {
                        copy(&mut self.renderer, t, &pass.without_drawable())
                            .and_then(|_| self.ui.pixels.draw(&mut self.renderer, &pass))
                    } else {
                        copy(&mut self.renderer, t, &pass)
                    };
                    if let Err(e) = drawn {
                        eprintln!("threers: frame: {e}");
                        return false;
                    }
                    return true;
                }
                Err(e) => eprintln!("threers: the frame offscreen: {e}"),
            }
        }
        let scene = &mut self.scene;
        let drawn = if shown {
            self.renderer
                .render_views(scene, &[view], &pass.without_drawable())
                .and_then(|_| self.ui.pixels.draw(&mut self.renderer, &pass))
        } else {
            self.renderer
                .render_views(scene, &[view], &pass)
                .map(drop)
        };
        if let Err(e) = drawn {
            eprintln!("threers: frame: {e}");
            return false;
        }
        true
    }

    /// Frame the app's focus the first time there is one, then apply the
    /// app's camera requests and the gestures since the last frame.
    fn orbit(&mut self, dt: f32) {
        for r in crate::take_camera_requests() {
            match r {
                CameraRequest::Home => self.framed = false,
                CameraRequest::Spin(on) => self.spin = on,
                CameraRequest::Zoom(k) if k.is_finite() && k > 0.0 => {
                    let t = self.camera.target;
                    let d = (self.camera.position - t) * k;
                    let d = if d.length() < self.controls.min_distance {
                        d.normalize() * self.controls.min_distance
                    } else {
                        d
                    };
                    self.camera.position = t + d;
                    self.controls.reseed_from_camera(&self.camera);
                }
                CameraRequest::Zoom(_) => {}
            }
        }
        // what the app's UI leaves of the view (a docked sidebar off its
        // left): the picture shifted to centre there, and framed to fit it
        let (sw, sh) = self.surface.size();
        let (sw, sh) = (sw as f32, sh.max(1) as f32);
        let left = (self.app.covered_left() * self.scale * self.ui_zoom).clamp(0.0, sw * 0.6);
        self.camera.view_offset = (left > 0.0).then(|| threers::cameras::ViewOffset { full_width: sw, full_height: sh, x: -left * 0.5, y: 0.0, width: sw, height: sh });
        let aspect = (sw - left) / sh;
        self.app.view_aspect(aspect);
        if !self.framed
            && let Some((centre, radius)) = self.app.focus()
        {
            let radius = radius.max(1e-6);
            // fit the narrower way: across, on a phone held upright
            let across = 2.0 * ((self.camera.fov * 0.5).tan() * aspect).atan();
            let distance = radius / (self.camera.fov.min(across) * 0.5).sin() * 1.1;
            // the first time, from the front; after, flown there, seen the
            // way it was (it jumped, turned back to the front)
            let first = self.camera.position == self.camera.target || self.camera.position.length_sq() == 0.0;
            let way = (self.camera.position - self.camera.target).normalize();
            let way = if first || !way.x.is_finite() { Vector3::new(0.0, 0.0, 1.0) } else { way };
            let (from, from_target) = (self.camera.position, self.camera.target);
            self.camera.target = centre;
            self.camera.position = centre + way * distance;
            if !first {
                self.flight = Some((from, from_target, self.camera.position, centre, 0.0));
            }
            self.pinch_anchor = None;
            self.camera.near = distance * 0.01;
            self.camera.far = distance * 100.0;
            // near enough to look into a thousandth of what is framed (a
            // cubic millimetre of cortex, framed in a whole brain); the near
            // plane follows the camera in (below)
            self.controls.min_distance = radius * 0.001;
            self.controls.max_distance = distance * 20.0;
            self.controls.reseed_from_camera(&self.camera);
            self.framed = true;
        }
        let (w, h) = self.surface.size();
        let size = (w as f32, h as f32);
        let none = PointerEvent {
            dx: 0.0,
            dy: 0.0,
            wheel: 0.0,
            rotating: false,
            panning: false,
            cursor: (0.0, 0.0),
        };
        // Damping by time, not by frames: under the fingers the brain keeps
        // up with them (half the way in a 60th of a second), let go it
        // glides a moment. At three.js's 0.05 a frame it trailed a drag by
        // half a second at 30 fps, more when frames came slower.
        if self.touch.is_some() || self.zoom != 1.0 || self.pan != (0.0, 0.0) {
            self.held = 0.1;
        } else {
            self.held = (self.held - dt).max(0.0);
        }
        let per_60th = if self.held > 0.0 { 0.5 } else { 0.12 };
        self.controls.damping = 1.0 - (1.0f32 - per_60th).powf((dt * 60.0).clamp(0.0, 8.0));
        if std::env::var_os("THREERS_TOUCH_LOG").is_some() && (self.turn != (0.0, 0.0) || self.zoom != 1.0 || self.pan != (0.0, 0.0)) {
            eprintln!("touchlog orbit dt {dt:.3} turn {:?} zoom {:.3} pan {:?} touch {} on_ui {} distance {:.4}", self.turn, self.zoom, self.pan, self.touch.is_some(), self.on_ui(), (self.camera.position - self.camera.target).length());
        }
        // what moves the view now (a finger, a pinch, two fingers, a flight)
        let moving = self.turn != (0.0, 0.0) || self.pan != (0.0, 0.0) || self.zoom != 1.0 || self.flight.is_some() || (self.spin && self.touch.is_none());
        self.still_for = if moving { 0.0 } else { self.still_for + dt };
        // flying to a reframe: eased there, any input ending it
        if let Some((p0, t0, p1, t1, t)) = self.flight.as_mut() {
            *t += dt;
            let k = (*t / FLIGHT).min(1.0);
            let e = k * k * (3.0 - 2.0 * k);
            let (p0, t0, p1, t1) = (*p0, *t0, *p1, *t1);
            let touched = self.turn != (0.0, 0.0) || self.pan != (0.0, 0.0) || self.zoom != 1.0;
            if k >= 1.0 || touched {
                self.camera.position = p1;
                self.camera.target = t1;
                self.flight = None;
            } else {
                // the target eased straight; the camera eased along, kept
                // as far from the target as the ends are (no dip through)
                let target = t0 + (t1 - t0) * e;
                let (d0, d1) = ((p0 - t0).length(), (p1 - t1).length());
                let way = ((p0 - t0) * (1.0 - e) + (p1 - t1) * e).normalize();
                self.camera.target = target;
                self.camera.position = target + way * (d0 * (d1 / d0.max(1e-9)).powf(e));
            }
            self.controls.reseed_from_camera(&self.camera);
        }
        // a pinch: toward what is under the fingers, at once (eased, it lagged
        // and drifted on), stopping short of it (it went through the brain)
        let zoom = std::mem::replace(&mut self.zoom, 1.0);
        if (zoom - 1.0).abs() > 1e-6 {
            let anchor = match self.pinch_anchor {
                Some(a) if a.1 < 0.35 => a.0,
                _ => self.zoom_point(),
            };
            self.pinch_anchor = Some((anchor, 0.0));
            let (p, t) = (self.camera.position, self.camera.target);
            let near = (p - anchor).length();
            let k = (1.0 / zoom).clamp(0.2, 5.0);
            // no nearer it than a hair, nor farther than the controls allow
            let k = k.max(self.controls.min_distance * 0.5 / near.max(1e-9)).min(self.controls.max_distance / (p - t).length().max(1e-9));
            self.camera.position = anchor + (p - anchor) * k;
            self.camera.target = anchor + (t - anchor) * k;
            self.controls.reseed_from_camera(&self.camera);
        }
        if let Some(a) = self.pinch_anchor.as_mut() {
            a.1 += dt;
        }
        let mut turn = std::mem::take(&mut self.turn);
        if self.spin && !self.on_ui() && self.touch.is_none() {
            turn.0 += SPIN * self.scale * dt;
        }
        if turn != (0.0, 0.0) {
            self.controls.update_dt(
                PointerEvent {
                    dx: turn.0,
                    dy: turn.1,
                    rotating: true,
                    ..none
                },
                &mut self.camera,
                size,
                dt,
            );
        }
        let pan = std::mem::take(&mut self.pan);
        if pan != (0.0, 0.0) {
            self.controls.update_dt(
                PointerEvent {
                    dx: pan.0,
                    dy: pan.1,
                    panning: true,
                    ..none
                },
                &mut self.camera,
                size,
                dt,
            );
        }
        // Every frame, with or without input: damping eases on after a fling.
        if self.flight.is_none() {
            self.controls.update_dt(none, &mut self.camera, size, dt);
        }
        // The far plane follows a zoom out; the near one a zoom in (fixed
        // when framed, it cut off what the camera came close to)
        let distance = (self.camera.position - self.camera.target).length();
        self.camera.far = self.camera.far.max(distance * 10.0);
        self.camera.near = (distance * 0.01).clamp(self.camera.far * 1e-7, self.camera.far * 1e-3);
    }

    pub fn scene(&mut self) -> &mut Scene {
        &mut self.scene
    }

    pub fn app(&mut self) -> &mut A {
        &mut self.app
    }

    /// The window's camera.
    pub fn camera(&self) -> &PerspectiveCamera {
        &self.camera
    }
}

/// `projection`'s focal lengths and centre with `near`..`far` mapped to
/// 0..1 depth (right-handed, looking down −Z).
fn with_depth(projection: [f32; 16], near: f32, far: f32) -> [f32; 16] {
    let p = projection;
    let mut m = [0.0; 16];
    m[0] = p[0];
    m[4] = p[4];
    m[5] = p[5];
    m[8] = p[8];
    m[9] = p[9];
    m[10] = far / (near - far);
    m[11] = -1.0;
    m[14] = near * far / (near - far);
    m
}

/// `app!`'s create.
///
/// # Safety
/// As [`View::new`].
#[doc(hidden)]
pub unsafe fn create<A: App>(
    layer: *mut c_void,
    width: u32,
    height: u32,
    mode: u32,
) -> *mut c_void {
    let mode = if mode == Mode::Ar as u32 {
        Mode::Ar
    } else {
        Mode::Window
    };
    match unsafe { View::<A>::new(layer, width, height, mode) } {
        Ok(v) => Box::into_raw(Box::new(v)).cast(),
        Err(e) => {
            eprintln!("threers: no view: {e}");
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `view` null, or from [`create::<A>`] and not destroyed.
#[doc(hidden)]
pub unsafe fn view<'a, A: App>(view: *mut c_void) -> Option<&'a mut View<A>> {
    unsafe { view.cast::<View<A>>().as_mut() }
}

/// # Safety
/// `view` null, or from [`create::<A>`]; not used again.
#[doc(hidden)]
pub unsafe fn destroy<A: App>(view: *mut c_void) {
    if !view.is_null() {
        drop(unsafe { Box::from_raw(view.cast::<View<A>>()) });
        // no view, no AR: the mode it left must not outlive it
        crate::set_current_mode(Mode::Window);
    }
}

/// The view's mode requests applied ([`View::apply_requests`]): from the
/// main thread.
///
/// # Safety
/// `v` null or from `create`.
pub unsafe fn requests<A: App>(v: *mut c_void) {
    if let Some(v) = unsafe { view::<A>(v) } {
        v.apply_requests();
    }
}

/// `m`, sixteen floats column-major, when `p` is not null.
///
/// # Safety
/// `p` null or sixteen readable floats.
#[doc(hidden)]
pub unsafe fn matrix(p: *const f32) -> Option<[f32; 16]> {
    (!p.is_null()).then(|| unsafe { std::ptr::read_unaligned(p.cast::<[f32; 16]>()) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ar_projection_keeps_its_lens_and_takes_the_apps_depth() {
        // A GL-style off-centre projection, as ARKit gives one.
        let gl = Matrix4::perspective(1.0, 0.5, 0.001, 1000.0);
        let mut p = gl.elements;
        p[8] = 0.02;
        p[9] = -0.03;
        let m = with_depth(p, 0.1, 50.0);
        assert_eq!((m[0], m[5], m[8], m[9]), (p[0], p[5], p[8], p[9]));
        let depth = |z: f32| {
            let (cz, cw) = (m[10] * z + m[14], m[11] * z + m[15]);
            cz / cw
        };
        assert!(depth(-0.1).abs() < 1e-5, "the near plane at 0");
        assert!((depth(-50.0) - 1.0).abs() < 1e-5, "the far plane at 1");
    }
}

#[doc(hidden)]
/// # Safety
/// `v` null or from `create`; `len` null or writable.
pub unsafe fn controls<A: App>(v: *mut c_void, len: *mut usize) -> *const u8 {
    let Some(json) = (unsafe { view::<A>(v) }).and_then(|v| v.controls()) else {
        return std::ptr::null();
    };
    if !len.is_null() {
        unsafe { *len = json.len() };
    }
    json.as_ptr()
}
