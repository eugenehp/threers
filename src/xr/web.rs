//! The browser's WebXR session, from Rust: [`XrSession`] opens one (or
//! takes over one the page opened), runs its frame loop, tracks the
//! controllers and hands, and puts each frame on the headset.
//!
//! Drawing stays with the app, on its own canvas and its own GPU (WebGPU or
//! WebGL2): inside the frame callback it sizes the canvas to the headset's
//! framebuffer ([`XrSession::fit_canvas`]), draws each of
//! [`XrSession::views`] into its viewport, and calls
//! [`XrSession::mark_drawn`]. A headset takes its frames from an
//! `XRWebGLLayer`, which belongs to a WebGL context, so the session keeps a
//! small WebGL2 context of its own and, when the callback returns, copies
//! the canvas into the layer in one draw (the eye rectangles already match).
//!
//! `wasm::WebXr` (the three.js shim's `renderer.xr`) is this, for
//! JavaScript; `threers-ui-demo` uses it directly.
//!
//! The device's own types are declared here, and only what is used of them:
//! the web-sys ones are behind `web_sys_unstable_apis`.

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use js_sys::{Array, Float32Array, Function, Object, Promise, Reflect};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext as Gl, WebGlFramebuffer, WebGlProgram, WebGlTexture};

use super::{HAND_JOINTS, Handedness, InputSlots, TargetRayMode, XrEvent, XrEventKind, XrEye, XrInput, XrSource};
use crate::math::Matrix4;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(extends = web_sys::EventTarget, js_name = XRSession)]
    #[derive(Clone)]
    type XrSessionJs;
    #[wasm_bindgen(method, js_class = "XRSession", js_name = requestReferenceSpace)]
    fn request_reference_space(this: &XrSessionJs, kind: &str) -> Promise;
    #[wasm_bindgen(method, js_class = "XRSession", js_name = updateRenderState)]
    fn update_render_state(this: &XrSessionJs, state: &Object);
    #[wasm_bindgen(method, js_class = "XRSession", js_name = requestAnimationFrame)]
    fn request_animation_frame(this: &XrSessionJs, callback: &Function) -> u32;
    #[wasm_bindgen(method, js_class = "XRSession", js_name = cancelAnimationFrame)]
    fn cancel_animation_frame(this: &XrSessionJs, handle: u32);
    #[wasm_bindgen(method, js_class = "XRSession")]
    fn end(this: &XrSessionJs) -> Promise;
    #[wasm_bindgen(method, getter, js_class = "XRSession", js_name = inputSources)]
    fn input_sources(this: &XrSessionJs) -> JsValue;
    #[wasm_bindgen(method, getter, js_class = "XRSession", js_name = environmentBlendMode)]
    fn environment_blend_mode(this: &XrSessionJs) -> Option<String>;

    #[wasm_bindgen(js_name = XRFrame)]
    type XrFrame;
    #[wasm_bindgen(method, js_class = "XRFrame", js_name = getViewerPose)]
    fn get_viewer_pose(this: &XrFrame, space: &JsValue) -> Option<XrPose>;
    #[wasm_bindgen(method, js_class = "XRFrame", js_name = getPose)]
    fn get_pose(this: &XrFrame, space: &JsValue, base: &JsValue) -> Option<XrPose>;
    #[wasm_bindgen(method, catch, js_class = "XRFrame", js_name = fillPoses)]
    fn fill_poses(this: &XrFrame, spaces: &Array, base: &JsValue, out: &Float32Array) -> Result<bool, JsValue>;
    #[wasm_bindgen(method, catch, js_class = "XRFrame", js_name = fillJointRadii)]
    fn fill_joint_radii(this: &XrFrame, joints: &Array, out: &Float32Array) -> Result<bool, JsValue>;

    /// An `XRPose` or `XRViewerPose` (which adds `views`).
    #[wasm_bindgen(js_name = XRPose)]
    type XrPose;
    #[wasm_bindgen(method, getter, js_class = "XRPose")]
    fn transform(this: &XrPose) -> XrRigidTransform;
    #[wasm_bindgen(method, getter, js_class = "XRPose")]
    fn views(this: &XrPose) -> Array;

    #[wasm_bindgen(js_name = XRView)]
    type XrViewJs;
    #[wasm_bindgen(method, getter, js_class = "XRView", js_name = projectionMatrix)]
    fn projection_matrix(this: &XrViewJs) -> Vec<f32>;
    #[wasm_bindgen(method, getter, js_class = "XRView")]
    fn transform(this: &XrViewJs) -> XrRigidTransform;

    #[wasm_bindgen(js_name = XRRigidTransform)]
    type XrRigidTransform;
    #[wasm_bindgen(method, getter, js_class = "XRRigidTransform")]
    fn matrix(this: &XrRigidTransform) -> Vec<f32>;

    #[wasm_bindgen(js_name = XRWebGLLayer)]
    type XrWebGlLayer;
    #[wasm_bindgen(constructor, catch, js_class = "XRWebGLLayer")]
    fn new(session: &XrSessionJs, gl: &Gl, init: &Object) -> Result<XrWebGlLayer, JsValue>;
    #[wasm_bindgen(method, getter, js_class = "XRWebGLLayer")]
    fn framebuffer(this: &XrWebGlLayer) -> Option<WebGlFramebuffer>;
    #[wasm_bindgen(method, getter, js_class = "XRWebGLLayer", js_name = framebufferWidth)]
    fn framebuffer_width(this: &XrWebGlLayer) -> u32;
    #[wasm_bindgen(method, getter, js_class = "XRWebGLLayer", js_name = framebufferHeight)]
    fn framebuffer_height(this: &XrWebGlLayer) -> u32;
    #[wasm_bindgen(method, js_class = "XRWebGLLayer", js_name = getViewport)]
    fn get_viewport(this: &XrWebGlLayer, view: &XrViewJs) -> Option<XrViewport>;

    #[wasm_bindgen(js_name = XRViewport)]
    type XrViewport;
    #[wasm_bindgen(method, getter, js_class = "XRViewport")]
    fn x(this: &XrViewport) -> i32;
    #[wasm_bindgen(method, getter, js_class = "XRViewport")]
    fn y(this: &XrViewport) -> i32;
    #[wasm_bindgen(method, getter, js_class = "XRViewport")]
    fn width(this: &XrViewport) -> i32;
    #[wasm_bindgen(method, getter, js_class = "XRViewport")]
    fn height(this: &XrViewport) -> i32;
}

fn get(o: &JsValue, key: &str) -> JsValue {
    Reflect::get(o, &key.into()).unwrap_or(JsValue::UNDEFINED)
}

fn set(o: &Object, key: &str, value: impl Into<JsValue>) {
    let _ = Reflect::set(o, &key.into(), &value.into());
}

/// `navigator.xr`, when the browser has one.
fn xr_system() -> Option<JsValue> {
    let nav = get(&web_sys::window()?.into(), "navigator");
    let xr = get(&nav, "xr");
    (!xr.is_undefined() && !xr.is_null()).then_some(xr)
}

fn call(o: &JsValue, name: &str, args: &[JsValue]) -> Result<JsValue, JsValue> {
    let f: Function = get(o, name).dyn_into().map_err(|_| JsValue::from_str(&format!("no {name}")))?;
    f.apply(o, &args.iter().collect())
}

/// The array-like `session.inputSources`, as a list.
fn sources_of(session: &XrSessionJs) -> Vec<JsValue> {
    let list = session.input_sources();
    let n = get(&list, "length").as_f64().unwrap_or(0.0) as u32;
    (0..n).map(|i| Reflect::get_u32(&list, i).unwrap_or(JsValue::UNDEFINED)).collect()
}

/// The canvas-to-layer copy: a texture the canvas is uploaded into, drawn
/// over the whole framebuffer by one triangle.
struct Blit {
    program: WebGlProgram,
    texture: WebGlTexture,
}

const BLIT_VS: &str = "#version 300 es
out vec2 uv;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}";

const BLIT_FS: &str = "#version 300 es
precision mediump float;
in vec2 uv;
uniform sampler2D source;
out vec4 color;
void main() { color = texture(source, uv); }";

impl Blit {
    fn new(gl: &Gl) -> Result<Blit, JsValue> {
        let shader = |kind, src: &str| -> Result<_, JsValue> {
            let s = gl.create_shader(kind).ok_or("no shader")?;
            gl.shader_source(&s, src);
            gl.compile_shader(&s);
            if !gl.get_shader_parameter(&s, Gl::COMPILE_STATUS).as_bool().unwrap_or(false) {
                return Err(gl.get_shader_info_log(&s).unwrap_or_default().into());
            }
            Ok(s)
        };
        let program = gl.create_program().ok_or("no program")?;
        gl.attach_shader(&program, &shader(Gl::VERTEX_SHADER, BLIT_VS)?);
        gl.attach_shader(&program, &shader(Gl::FRAGMENT_SHADER, BLIT_FS)?);
        gl.link_program(&program);
        if !gl.get_program_parameter(&program, Gl::LINK_STATUS).as_bool().unwrap_or(false) {
            return Err(gl.get_program_info_log(&program).unwrap_or_default().into());
        }
        let texture = gl.create_texture().ok_or("no texture")?;
        gl.bind_texture(Gl::TEXTURE_2D, Some(&texture));
        // Canvas and framebuffer are the same size: texel for pixel.
        for (p, v) in [
            (Gl::TEXTURE_MIN_FILTER, Gl::NEAREST),
            (Gl::TEXTURE_MAG_FILTER, Gl::NEAREST),
            (Gl::TEXTURE_WRAP_S, Gl::CLAMP_TO_EDGE),
            (Gl::TEXTURE_WRAP_T, Gl::CLAMP_TO_EDGE),
        ] {
            gl.tex_parameteri(Gl::TEXTURE_2D, p, v as i32);
        }
        Ok(Blit { program, texture })
    }

    /// `canvas` over the whole of `target`. The canvas's rows run top down
    /// and GL's bottom up, so the upload flips it; then the eye rectangles,
    /// both measured from the bottom, line up.
    fn draw(&self, gl: &Gl, canvas: &HtmlCanvasElement, target: Option<&WebGlFramebuffer>, w: u32, h: u32) {
        gl.bind_framebuffer(Gl::FRAMEBUFFER, target);
        gl.viewport(0, 0, w as i32, h as i32);
        gl.disable(Gl::SCISSOR_TEST);
        gl.disable(Gl::BLEND);
        gl.disable(Gl::DEPTH_TEST);
        gl.active_texture(Gl::TEXTURE0);
        gl.bind_texture(Gl::TEXTURE_2D, Some(&self.texture));
        gl.pixel_storei(Gl::UNPACK_FLIP_Y_WEBGL, 1);
        gl.pixel_storei(Gl::UNPACK_PREMULTIPLY_ALPHA_WEBGL, 1);
        let _ = gl.tex_image_2d_with_u32_and_u32_and_html_canvas_element(
            Gl::TEXTURE_2D,
            0,
            Gl::RGBA as i32,
            Gl::RGBA,
            Gl::UNSIGNED_BYTE,
            canvas,
        );
        gl.use_program(Some(&self.program));
        gl.uniform1i(gl.get_uniform_location(&self.program, "source").as_ref(), 0);
        gl.draw_arrays(Gl::TRIANGLES, 0, 3);
    }
}

/// One eye of the current frame, as the device gave it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XrView {
    /// The device's projection, in OpenGL depth (as three.js keeps it).
    pub projection: [f32; 16],
    /// The eye's pose in the reference space.
    pub pose: [f32; 16],
    /// `[x, y, width, height]` in framebuffer pixels, from the bottom left.
    pub viewport: [i32; 4],
}

type FrameLoop = Closure<dyn FnMut(f64, JsValue)>;
type Listener = Closure<dyn FnMut(web_sys::Event)>;
/// A frame callback: `(time, frame)`.
pub type FrameCallback = Box<dyn FnMut(f64, &JsValue)>;

struct Session {
    session: XrSessionJs,
    mode: String,
    space: JsValue,
    /// The app's canvas: sized to the layer, copied into it.
    canvas: HtmlCanvasElement,
    gl: Gl,
    layer: XrWebGlLayer,
    blit: Blit,
    /// Set inside a frame callback only (a frame is dead outside it).
    frame: Option<XrFrame>,
    views: Vec<XrView>,
    head: Option<[f32; 16]>,
    /// The app's per-frame callback, `(time, frame)`; taken out while it runs.
    callback: Option<FrameCallback>,
    frame_loop: Option<FrameLoop>,
    handle: Option<u32>,
    /// The canvas holds eyes drawn this frame.
    drawn: bool,
    depth: (f32, f32),
    slots: InputSlots<JsValue>,
    /// With the source each came from (gone from the slots once it left).
    events: Vec<(XrEvent, JsValue)>,
    listeners: Vec<(&'static str, Listener)>,
    ended: bool,
    on_end: Option<Box<dyn FnOnce()>>,
}

impl Session {
    fn request_frame(&mut self) {
        if let Some(f) = &self.frame_loop {
            self.handle = Some(self.session.request_animation_frame(f.as_ref().unchecked_ref()));
        }
    }

    /// The frame's eyes and head pose, read once before the app's callback.
    fn begin(&mut self, frame: XrFrame) {
        self.views.clear();
        self.head = None;
        if let Some(pose) = frame.get_viewer_pose(&self.space) {
            self.head = to16(&pose.transform().matrix());
            for v in pose.views().iter() {
                let v: XrViewJs = v.unchecked_into();
                let Some(vp) = self.layer.get_viewport(&v) else { continue };
                if let (Some(projection), Some(pose)) = (to16(&v.projection_matrix()), to16(&v.transform().matrix())) {
                    self.views.push(XrView { projection, pose, viewport: [vp.x(), vp.y(), vp.width(), vp.height()] });
                }
            }
        }
        self.frame = Some(frame);
        self.drawn = false;
    }

    /// After the app's callback: what it drew goes to the headset.
    fn finish(&mut self) {
        if self.drawn {
            let (w, h) = (self.layer.framebuffer_width(), self.layer.framebuffer_height());
            self.blit.draw(&self.gl, &self.canvas, self.layer.framebuffer().as_ref(), w, h);
        }
        self.frame = None;
        self.drawn = false;
    }

    fn event(&mut self, kind: XrEventKind, source: &JsValue) {
        let slot = match kind {
            XrEventKind::Connected => Some(self.slots.connect(source.clone())),
            XrEventKind::Disconnected => self.slots.disconnect(source),
            _ => self.slots.slot_of(source),
        };
        if let Some(slot) = slot {
            self.events.push((XrEvent { kind, slot }, source.clone()));
        }
    }

    fn shut(&mut self) {
        self.ended = true;
        if let Some(h) = self.handle.take() {
            self.session.cancel_animation_frame(h);
        }
        let held = (self.frame_loop.take(), std::mem::take(&mut self.listeners), self.callback.take());
        for (name, l) in &held.1 {
            let _ = self.session.remove_event_listener_with_callback(name, l.as_ref().unchecked_ref());
        }
        // Not dropped here: this may be running inside one of them.
        wasm_bindgen_futures::spawn_local(async move { drop(held) });
    }
}

fn to16(v: &[f32]) -> Option<[f32; 16]> {
    v.try_into().ok()
}

/// An immersive WebXR session. Cloning gives another handle to the same
/// session. See the module docs.
#[derive(Clone)]
pub struct XrSession {
    inner: Rc<RefCell<Session>>,
}

impl std::fmt::Debug for XrSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.inner.borrow();
        f.debug_struct("XrSession").field("mode", &s.mode).field("ended", &s.ended).finish()
    }
}

impl XrSession {
    /// Whether the browser can open a session of `mode` (`"immersive-vr"`,
    /// `"immersive-ar"`). False without WebXR, which includes a page that
    /// is not secure (http other than localhost).
    pub async fn is_supported(mode: &str) -> bool {
        let Some(xr) = xr_system() else { return false };
        let Ok(p) = call(&xr, "isSessionSupported", &[mode.into()]) else { return false };
        match JsFuture::from(Promise::from(p)).await {
            Ok(v) => v.as_bool().unwrap_or(false),
            Err(_) => false,
        }
    }

    /// Asks for a session of `mode` (from a click: browsers want a user
    /// gesture, and it lasts a few seconds) with `reference_space` (three.js's
    /// default is `"local-floor"`) and hand tracking where offered, and
    /// attaches it as [`attach`](Self::attach) does. `features` are further
    /// optional features to ask for.
    pub async fn request(
        mode: &str,
        canvas: HtmlCanvasElement,
        reference_space: &str,
        framebuffer_scale: f32,
        features: &[String],
    ) -> Result<XrSession, JsValue> {
        let xr = xr_system().ok_or("WebXR is not available here")?;
        let optional = Array::new();
        for f in features.iter().map(String::as_str).chain([reference_space, "hand-tracking"]) {
            if !optional.includes(&f.into(), 0) {
                optional.push(&f.into());
            }
        }
        let init = Object::new();
        set(&init, "optionalFeatures", optional);
        let session = JsFuture::from(Promise::from(call(&xr, "requestSession", &[mode.into(), init.into()])?)).await?;
        Self::attach(session, canvas, reference_space, framebuffer_scale).await
    }

    /// Takes over a session the page opened itself (three.js
    /// `xr.setSession`): a WebGL2 layer for it, a reference space (`"local"`
    /// if the one asked for is refused), and input tracking. Frames start
    /// with [`set_frame_callback`](Self::set_frame_callback).
    pub async fn attach(
        session: JsValue,
        canvas: HtmlCanvasElement,
        reference_space: &str,
        framebuffer_scale: f32,
    ) -> Result<XrSession, JsValue> {
        let session: XrSessionJs = session.unchecked_into();
        let mode = if session.environment_blend_mode().is_some_and(|m| m != "opaque") { "immersive-ar" } else { "immersive-vr" };
        let document = web_sys::window().and_then(|w| w.document()).ok_or("no document")?;
        let layer_canvas: HtmlCanvasElement = document.create_element("canvas")?.unchecked_into();
        let attrs = Object::new();
        set(&attrs, "xrCompatible", true);
        set(&attrs, "antialias", false);
        set(&attrs, "depth", false);
        set(&attrs, "alpha", mode == "immersive-ar");
        let gl: Gl = layer_canvas.get_context_with_context_options("webgl2", &attrs)?.ok_or("no WebGL2 for the layer")?.unchecked_into();
        let init = Object::new();
        set(&init, "framebufferScaleFactor", if framebuffer_scale > 0.0 { framebuffer_scale } else { 1.0 });
        set(&init, "antialias", false);
        set(&init, "depth", false);
        set(&init, "alpha", mode == "immersive-ar");
        let layer = XrWebGlLayer::new(&session, &gl, &init)?;
        let state = Object::new();
        set(&state, "baseLayer", &layer);
        session.update_render_state(&state);
        let space = match JsFuture::from(session.request_reference_space(reference_space)).await {
            Ok(s) => s,
            Err(_) => JsFuture::from(session.request_reference_space("local")).await?,
        };
        let inner = Rc::new(RefCell::new(Session {
            session: session.clone(),
            mode: mode.into(),
            space,
            canvas,
            blit: Blit::new(&gl)?,
            gl,
            layer,
            frame: None,
            views: vec![],
            head: None,
            callback: None,
            frame_loop: None,
            handle: None,
            drawn: false,
            depth: (0.1, 1000.0),
            slots: InputSlots::new(),
            events: vec![],
            listeners: vec![],
            ended: false,
            on_end: None,
        }));
        let weak = Rc::downgrade(&inner);
        let mut listeners = vec![];
        for name in ["select", "selectstart", "selectend", "squeeze", "squeezestart", "squeezeend"] {
            let weak = weak.clone();
            listeners.push((name, Listener::new(move |e: web_sys::Event| {
                if let (Some(s), Some(kind)) = (weak.upgrade(), XrEventKind::parse(name)) {
                    s.borrow_mut().event(kind, &get(&e, "inputSource"));
                }
            })));
        }
        let w = weak.clone();
        listeners.push(("inputsourceschange", Listener::new(move |e: web_sys::Event| {
            let Some(s) = w.upgrade() else { return };
            let mut s = s.borrow_mut();
            for src in Array::from(&get(&e, "removed")).iter() {
                s.event(XrEventKind::Disconnected, &src);
            }
            for src in Array::from(&get(&e, "added")).iter() {
                s.event(XrEventKind::Connected, &src);
            }
        })));
        let w = weak.clone();
        listeners.push(("end", Listener::new(move |_| {
            let Some(s) = w.upgrade() else { return };
            let on_end = {
                let mut s = s.borrow_mut();
                s.shut();
                s.on_end.take()
            };
            if let Some(f) = on_end {
                f();
            }
        })));
        for (name, l) in &listeners {
            session.add_event_listener_with_callback(name, l.as_ref().unchecked_ref())?;
        }
        {
            let mut s = inner.borrow_mut();
            s.listeners = listeners;
            // Sources there from the start get no change event.
            for src in sources_of(&session) {
                s.event(XrEventKind::Connected, &src);
            }
        }
        Ok(XrSession { inner })
    }

    /// Calls `callback(time, frame)` once per device frame (`None` stops
    /// calling it; the loop keeps the session alive). Inside it the views,
    /// head and inputs are this frame's; what is drawn and
    /// [marked](Self::mark_drawn) goes to the headset when it returns.
    pub fn set_frame_callback(&self, callback: Option<FrameCallback>) {
        let mut s = self.inner.borrow_mut();
        s.callback = callback;
        if s.frame_loop.is_some() || s.ended {
            return;
        }
        let weak: Weak<RefCell<Session>> = Rc::downgrade(&self.inner);
        s.frame_loop = Some(FrameLoop::new(move |time: f64, frame: JsValue| {
            let Some(rc) = weak.upgrade() else { return };
            let callback = {
                let mut s = rc.borrow_mut();
                if s.ended {
                    return;
                }
                s.request_frame();
                s.begin(frame.clone().unchecked_into());
                s.callback.take()
            };
            // Run unborrowed, so it can read this frame and draw.
            if let Some(mut f) = callback {
                f(time, &frame);
                let mut s = rc.borrow_mut();
                if s.callback.is_none() && !s.ended {
                    s.callback = Some(f);
                }
            }
            rc.borrow_mut().finish();
        }));
        s.request_frame();
    }

    /// Ends the session (the [`on_end`](Self::on_end) callback follows).
    pub fn end(&self) -> Promise {
        self.inner.borrow().session.end()
    }

    /// Called once when the session ends, by the page or the user.
    pub fn on_end(&self, callback: Option<Box<dyn FnOnce()>>) {
        self.inner.borrow_mut().on_end = callback;
    }

    /// The `XRSession`.
    pub fn session(&self) -> JsValue {
        self.inner.borrow().session.clone().into()
    }

    /// The `XRReferenceSpace` poses are in.
    pub fn reference_space(&self) -> JsValue {
        self.inner.borrow().space.clone()
    }

    /// `"immersive-vr"` or `"immersive-ar"` (see-through: draw no
    /// background where the room should show).
    pub fn mode(&self) -> String {
        self.inner.borrow().mode.clone()
    }

    pub fn ended(&self) -> bool {
        self.inner.borrow().ended
    }

    /// Whether this is inside a frame callback with a tracked head.
    pub fn in_frame(&self) -> bool {
        let s = self.inner.borrow();
        s.frame.is_some() && !s.views.is_empty()
    }

    /// The headset's framebuffer, `(width, height)` in pixels.
    pub fn framebuffer_size(&self) -> (u32, u32) {
        let s = self.inner.borrow();
        (s.layer.framebuffer_width(), s.layer.framebuffer_height())
    }

    /// The canvas sized to the headset's framebuffer (each eye then draws
    /// into its viewport of it). Returns the size.
    pub fn fit_canvas(&self) -> (u32, u32) {
        let s = self.inner.borrow();
        let (w, h) = (s.layer.framebuffer_width(), s.layer.framebuffer_height());
        if (s.canvas.width(), s.canvas.height()) != (w, h) {
            s.canvas.set_width(w);
            s.canvas.set_height(h);
        }
        (w, h)
    }

    /// Something was drawn on the canvas this frame: copy it to the headset.
    pub fn mark_drawn(&self) {
        self.inner.borrow_mut().drawn = true;
    }

    /// The near and far planes the device's projections use, from the next
    /// frame (match them to the camera's).
    pub fn set_depth_range(&self, near: f32, far: f32) {
        let mut s = self.inner.borrow_mut();
        if s.depth != (near, far) {
            s.depth = (near, far);
            let state = Object::new();
            set(&state, "depthNear", near);
            set(&state, "depthFar", far);
            s.session.update_render_state(&state);
        }
    }

    /// The head's pose in the reference space this frame (column-major),
    /// or `None` outside a frame or while tracking is lost.
    pub fn head(&self) -> Option<[f32; 16]> {
        self.inner.borrow().head
    }

    /// This frame's eyes, as the device gave them.
    pub fn views(&self) -> Vec<XrView> {
        self.inner.borrow().views.clone()
    }

    /// This frame's eyes as cameras see them, with the reference space at
    /// `origin` in the world (identity for a fixed rig).
    pub fn eyes(&self, origin: &Matrix4) -> Vec<XrEye> {
        let s = self.inner.borrow();
        s.views.iter().map(|v| XrEye::new(v.projection, origin, &Matrix4 { elements: v.pose }, v.viewport)).collect()
    }

    /// Controller events since the last call, each with its `XRInputSource`.
    pub fn take_events_with_sources(&self) -> Vec<(XrEvent, JsValue)> {
        std::mem::take(&mut self.inner.borrow_mut().events)
    }

    /// The `XRInputSource` in `slot`, while connected.
    pub fn source(&self, slot: usize) -> Option<JsValue> {
        self.inner.borrow().slots.iter().find(|(i, _)| *i == slot).map(|(_, s)| s.clone())
    }

    /// The connected inputs this frame (empty outside a frame).
    pub fn inputs(&self) -> Vec<XrInput> {
        let s = self.inner.borrow();
        let Some(frame) = &s.frame else { return vec![] };
        let pose_of = |space: &JsValue| -> Option<[f32; 16]> {
            if space.is_undefined() || space.is_null() {
                return None;
            }
            frame.get_pose(space, &s.space).and_then(|p| to16(&p.transform().matrix()))
        };
        let mut out = vec![];
        for (slot, src) in s.slots.iter() {
            let hand = get(src, "hand");
            let joints = (!hand.is_undefined() && !hand.is_null())
                .then(|| {
                    let joints: Array = HAND_JOINTS.iter().map(|n| call(&hand, "get", &[(*n).into()]).unwrap_or_default()).collect();
                    let (poses, radii) = (Float32Array::new_with_length(16 * 25), Float32Array::new_with_length(25));
                    frame.fill_poses(&joints, &s.space, &poses).unwrap_or(false).then(|| {
                        let _ = frame.fill_joint_radii(&joints, &radii);
                        (poses.to_vec(), radii.to_vec())
                    })
                })
                .flatten();
            let pad = get(src, "gamepad");
            let (mut axes, mut buttons) = (vec![], vec![]);
            if !pad.is_undefined() && !pad.is_null() {
                axes = Array::from(&get(&pad, "axes")).iter().map(|a| a.as_f64().unwrap_or(0.0) as f32).collect();
                buttons = Array::from(&get(&pad, "buttons"))
                    .iter()
                    .map(|b| (get(&b, "pressed").as_bool().unwrap_or(false), get(&b, "value").as_f64().unwrap_or(0.0) as f32))
                    .collect();
            }
            out.push(XrInput {
                slot,
                handedness: Handedness::parse(&get(src, "handedness").as_string().unwrap_or_default()),
                target_ray_mode: TargetRayMode::parse(&get(src, "targetRayMode").as_string().unwrap_or_default()),
                ray: pose_of(&get(src, "targetRaySpace")),
                grip: pose_of(&get(src, "gripSpace")),
                joints,
                axes,
                buttons,
            });
        }
        out
    }
}

impl XrSource for XrSession {
    fn head(&self) -> Option<[f32; 16]> {
        XrSession::head(self)
    }
    fn eyes(&self, origin: &Matrix4) -> Vec<XrEye> {
        XrSession::eyes(self, origin)
    }
    fn framebuffer_size(&self) -> (u32, u32) {
        XrSession::framebuffer_size(self)
    }
    fn inputs(&self) -> Vec<XrInput> {
        XrSession::inputs(self)
    }
    fn take_events(&mut self) -> Vec<XrEvent> {
        self.take_events_with_sources().into_iter().map(|(e, _)| e).collect()
    }
    fn see_through(&self) -> bool {
        self.mode() == "immersive-ar"
    }
}
