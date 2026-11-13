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
//! Everything here runs on the thread that calls it — the main thread, from
//! a display link, is right (UIKit is touched when the mode changes):
//! [`App::update`] is the app's per-frame work, and long work (downloads)
//! belongs on threads of its own.

use std::ffi::c_void;
use std::time::Instant;

use threers::cameras::{Camera, PerspectiveCamera};
use threers::controls::{OrbitControls, PointerEvent};
use threers::math::{Matrix4, Vector3};
use threers::metal::{MetalDevice, MetalRenderer, MetalSurface, RenderView};
use threers::scene::Scene;
use threers::ui::input::{Event, PointerButton, TouchPhase};
use threers::ui::native::{self, Control};
use threers::xr::{Frame, TargetRayMode, XrEvent, XrEventKind, XrInput, ray_pose};

use crate::surface_ui::SurfaceUi;
use crate::{App, CameraRequest, Place, Request, Surface};

/// Which view: see the module docs.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Window = 0,
    Ar = 1,
}

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
    /// A pinch: `a` its scale since the last call (above 1 spreads).
    Pinch = 2,
    /// Two fingers dragged: `a`, `b` pixels since the last call.
    Pan = 3,
    /// Pixels a point: `a` (the UI is laid out in points).
    Scale = 4,
    /// The safe area's insets, pixels: `a` top, `b` bottom, `c` either side.
    Insets = 5,
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
    surface: MetalSurface,
    scene: Scene,
    app: A,
    ui: SurfaceUi,
    mode: Mode,
    /// The window's camera and its controls.
    camera: PerspectiveCamera,
    controls: OrbitControls,
    framed: bool,
    /// Input since the last frame, for the controls.
    turn: (f32, f32),
    pan: (f32, f32),
    zoom: f32,
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
}

impl<A: App> View<A> {
    /// The UI's controls as the last frame drew them, as JSON
    /// (`threers_ui::native::to_json`; rectangles in UI points), when they
    /// changed since the last call: the shell puts an accessibility element
    /// over each, and an activation comes back by its key
    /// ([`crate::threers_ios_press`]). `None` while they are as they were.
    pub fn controls(&mut self) -> Option<&str> {
        let v = native::version(&self.native);
        if self.native_sent == Some(v) {
            return None;
        }
        self.native_sent = Some(v);
        self.native_json = native::to_json(&self.native);
        Some(&self.native_json)
    }

    /// The app, set up, drawing into `layer` at `width × height` pixels.
    ///
    /// # Safety
    /// `layer` must be a live `CAMetalLayer` that outlives the view.
    pub unsafe fn new(layer: *mut c_void, width: u32, height: u32, mode: Mode) -> Result<Self, String> {
        let device = MetalDevice::new().map_err(|e| e.to_string())?;
        let samples = if device.supports_sample_count(SAMPLES) { SAMPLES } else { 1 };
        let surface = unsafe { MetalSurface::from_layer(&device, layer, width.max(1), height.max(1), samples) }.map_err(|e| e.to_string())?;
        let renderer = MetalRenderer::with_device(device).map_err(|e| e.to_string())?;
        let mut scene = Scene::new();
        let app = A::setup(&mut scene);
        let lens = app.lens();
        let mut camera = PerspectiveCamera::new(lens.fov.to_degrees(), width.max(1) as f32 / height.max(1) as f32, lens.near, lens.far);
        camera.position = Vector3::new(0.0, 0.0, 3.0);
        camera.target = Vector3::ZERO;
        let mut controls = OrbitControls::new(&camera);
        controls.enable_damping = true;
        controls.screen_space_panning = true;
        let mut view = View {
            renderer,
            surface,
            scene,
            app,
            ui: SurfaceUi::default(),
            mode: Mode::Window,
            camera,
            controls,
            framed: false,
            turn: (0.0, 0.0),
            pan: (0.0, 0.0),
            zoom: 1.0,
            touch: None,
            pointer: None,
            events: Vec::new(),
            release: false,
            ar: None,
            #[cfg(all(target_os = "ios", target_arch = "aarch64"))]
            session: None,
            layer_ptr: layer,
            scale: 2.0,
            insets: [0.0; 4],
            spin: false,
            start: Instant::now(),
            last: 0.0,
            native: vec![],
            native_sent: None,
            native_json: String::new(),
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
        match key {
            k if k == Key::Resize as u32 => {
                let (w, h) = ((a as u32).max(1), (b as u32).max(1));
                if let Err(e) = self.surface.resize(w, h) {
                    eprintln!("threers: resize: {e}");
                }
                self.camera.aspect = w as f32 / h as f32;
            }
            k if k == Key::Touch as u32 => self.touch(a as u32, (b, c)),
            k if k == Key::Pinch as u32 && a.is_finite() && a > 0.0 && !self.on_ui() => self.zoom *= a,
            k if k == Key::Pan as u32 && !self.on_ui() => {
                self.pan.0 += a;
                self.pan.1 += b;
            }
            k if k == Key::Scale as u32 && a.is_finite() && a > 0.0 => self.scale = a,
            k if k == Key::Insets as u32 => self.insets = [a, c, b, c].map(|v| if v.is_finite() { v.max(0.0) } else { 0.0 }),
            _ => {}
        }
    }

    /// A finger is on the UI.
    fn on_ui(&self) -> bool {
        self.touch.as_ref().is_some_and(|t| t.ui)
    }

    fn time(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// The finger to the UI, in its points. Whether the UI took it.
    fn touch_ui(&mut self, phase: TouchPhase, at: (f32, f32)) -> bool {
        let (x, y) = (at.0 / self.scale, at.1 / self.scale);
        let time = self.time();
        self.app.ui_event(Place::Screen, Event::Touch { id: 0, phase, x, y, time }, time)
    }

    fn touch(&mut self, phase: u32, at: (f32, f32)) {
        match (phase, self.touch.as_mut()) {
            (0, _) => {
                if self.touch_ui(TouchPhase::Start, at) {
                    self.touch = Some(Touch { down: at, at, dragged: false, ui: true });
                    return;
                }
                // not the UI's: it forgets the finger
                self.touch_ui(TouchPhase::Cancel, at);
                self.touch = Some(Touch { down: at, at, dragged: false, ui: false });
                if self.mode == Mode::Ar {
                    self.pointer = self.ray_at(at);
                    self.events.push(XrEvent { kind: XrEventKind::Connected, slot: 0 });
                    self.events.push(XrEvent { kind: XrEventKind::SelectStart, slot: 0 });
                }
            }
            (1, Some(t)) if t.ui => {
                t.at = at;
                self.touch_ui(TouchPhase::Move, at);
            }
            (2 | 3, Some(t)) if t.ui => {
                self.touch = None;
                self.touch_ui(if phase == 2 { TouchPhase::End } else { TouchPhase::Cancel }, at);
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
                    self.events.push(XrEvent { kind: XrEventKind::Connected, slot: 0 });
                    self.events.push(XrEvent { kind: XrEventKind::SelectStart, slot: 0 });
                }
                self.events.push(XrEvent { kind: XrEventKind::SelectEnd, slot: 0 });
                if tap {
                    self.events.push(XrEvent { kind: XrEventKind::Select, slot: 0 });
                }
                self.events.push(XrEvent { kind: XrEventKind::Disconnected, slot: 0 });
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
        self.ar = Some(ArEye { world: Matrix4 { elements: world }, projection: with_depth(projection, lens.near, lens.far) });
    }

    /// The ray from the camera through the pixel `at`, in the reference
    /// space, as a pose looking down it.
    fn ray_at(&self, at: (f32, f32)) -> Option<[f32; 16]> {
        let (w, h) = self.surface.size();
        let ndc = (2.0 * at.0 / w as f32 - 1.0, 1.0 - 2.0 * at.1 / h as f32);
        // The AR camera is already in the reference space (ARKit's world);
        // the window's camera is in the scene's world.
        let (view, projection, to_reference) = match (self.mode, &self.ar) {
            (Mode::Ar, Some(eye)) => (eye.world.invert(), Matrix4 { elements: eye.projection }, Matrix4::identity()),
            (Mode::Ar, None) => return None,
            (Mode::Window, _) => (self.camera.view_matrix(), self.camera.projection_matrix(), self.app.origin().invert()),
        };
        let near = Vector3::new(ndc.0, ndc.1, 0.0).unproject(&view, &projection).apply_matrix4(&to_reference);
        let far = Vector3::new(ndc.0, ndc.1, 1.0).unproject(&view, &projection).apply_matrix4(&to_reference);
        Some(ray_pose(near, far - near))
    }

    /// Update the app and draw one frame. `false` when nothing was drawn
    /// (no drawable, or no AR camera yet).
    pub fn frame(&mut self) -> bool {
        match crate::take_view_request() {
            Some(Request::Window) => drop(self.set_mode(Mode::Window)),
            Some(Request::Ar) => drop(self.set_mode(Mode::Ar)),
            _ => {}
        }
        #[cfg(all(target_os = "ios", target_arch = "aarch64"))]
        if let Some(session) = &self.session {
            let lens = self.app.lens();
            if let Some((world, projection)) = session.camera(self.surface.size(), lens.near, lens.far) {
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
            .map(|ray| XrInput { slot: 0, target_ray_mode: TargetRayMode::TransientPointer, ray: Some(ray), ..Default::default() })
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
        self.app.update(&mut self.scene, &frame);
        let view = match self.mode {
            Mode::Window => {
                self.orbit(dt as f32);
                RenderView::from_camera(&self.camera)
            }
            Mode::Ar => {
                let Some(eye) = &self.ar else { return false };
                self.scene.background_alpha = 0.0;
                let world = origin.multiply(&eye.world);
                let mut v = RenderView::from_matrices(world.invert(), Matrix4 { elements: eye.projection });
                v.position = Vector3::new(world.elements[12], world.elements[13], world.elements[14]);
                v
            }
        };
        let (w, h) = self.surface.size();
        let k = self.scale;
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
            crate::take_presses_for(|k| native::find(&self.native, k).is_some()).into_iter().filter_map(|k| native::find(&self.native, k)).map(|c| c.centre()).collect();
        for (x, y) in pressed {
            let button = PointerButton::Primary;
            self.app.ui_event(Place::Screen, Event::PointerMove { x, y }, time);
            self.app.ui_event(Place::Screen, Event::PointerDown { x, y, button }, time);
            self.app.ui_event(Place::Screen, Event::PointerUp { x, y, button }, time);
        }
        let (shown, _) = self.ui.update(&mut self.app, &surface);
        self.native = match (shown, self.app.ui(&surface)) {
            (true, Some(ui)) => native::controls(ui),
            _ => vec![],
        };
        let (tone, exposure) = self.app.tone_mapping();
        self.renderer.set_tone_mapping(tone, exposure);
        let Some(next) = self.surface.next_frame() else { return false };
        let pass = next.attachments();
        let drawn = if shown {
            self.renderer
                .render_views(&mut self.scene, &[view], &pass.without_drawable())
                .and_then(|_| self.ui.pixels.draw(&mut self.renderer, &pass))
        } else {
            self.renderer.render_views(&mut self.scene, &[view], &pass).map(drop)
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
                    let d = if d.length() < self.controls.min_distance { d.normalize() * self.controls.min_distance } else { d };
                    self.camera.position = t + d;
                    self.controls.reseed_from_camera(&self.camera);
                }
                CameraRequest::Zoom(_) => {}
            }
        }
        if !self.framed
            && let Some((centre, radius)) = self.app.focus()
        {
            let radius = radius.max(1e-6);
            // fit the narrower way: across, on a phone held upright
            let across = 2.0 * ((self.camera.fov * 0.5).tan() * self.camera.aspect).atan();
            let distance = radius / (self.camera.fov.min(across) * 0.5).sin() * 1.1;
            self.camera.target = centre;
            self.camera.position = centre + Vector3::new(0.0, 0.0, distance);
            self.camera.near = distance * 0.01;
            self.camera.far = distance * 100.0;
            self.controls.min_distance = radius * 0.2;
            self.controls.max_distance = distance * 20.0;
            self.controls.reseed_from_camera(&self.camera);
            self.framed = true;
        }
        let (w, h) = self.surface.size();
        let size = (w as f32, h as f32);
        let none = PointerEvent { dx: 0.0, dy: 0.0, wheel: 0.0, rotating: false, panning: false, cursor: (0.0, 0.0) };
        let mut turn = std::mem::take(&mut self.turn);
        if self.spin && !self.on_ui() && self.touch.is_none() {
            turn.0 += SPIN * self.scale * dt;
        }
        if turn != (0.0, 0.0) {
            self.controls.update_dt(PointerEvent { dx: turn.0, dy: turn.1, rotating: true, ..none }, &mut self.camera, size, dt);
        }
        let pan = std::mem::take(&mut self.pan);
        if pan != (0.0, 0.0) {
            self.controls.update_dt(PointerEvent { dx: pan.0, dy: pan.1, panning: true, ..none }, &mut self.camera, size, dt);
        }
        // The controls scale the distance by 1 − wheel·0.001·zoom_speed; a
        // pinch's scale s divides it by s.
        let zoom = std::mem::replace(&mut self.zoom, 1.0);
        let wheel = if (zoom - 1.0).abs() > 1e-6 { (1.0 - 1.0 / zoom) / (0.001 * self.controls.zoom_speed.max(1e-6)) } else { 0.0 };
        // Every frame, with or without input: damping eases on after a fling.
        self.controls.update_dt(PointerEvent { wheel, ..none }, &mut self.camera, size, dt);
        // The far plane follows a zoom out.
        let distance = (self.camera.position - self.camera.target).length();
        self.camera.far = self.camera.far.max(distance * 10.0);
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
pub unsafe fn create<A: App>(layer: *mut c_void, width: u32, height: u32, mode: u32) -> *mut c_void {
    let mode = if mode == Mode::Ar as u32 { Mode::Ar } else { Mode::Window };
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
    let Some(json) = (unsafe { view::<A>(v) }).and_then(|v| v.controls()) else { return std::ptr::null() };
    if !len.is_null() {
        unsafe { *len = json.len() };
    }
    json.as_ptr()
}
