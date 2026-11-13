//! WebXR for the three.js shim (`renderer.xr`, see `WebXRManager` there):
//! [`crate::xr::web::XrSession`] with JavaScript callbacks, and a renderer
//! drawing both eyes into it.
//!
//! The renderer draws on its own canvas, whichever of WebGPU or WebGL2 it
//! runs on: each frame [`WebRenderer::render_xr`] sizes the canvas to the
//! headset's framebuffer and draws every eye into its rectangle of it, and
//! the session copies the canvas to the headset after the page's callback.
//! Poses come from the device in its reference space; the page's `origin`
//! (a rig's world matrix) places that space in the scene, and
//! [`crate::xr::XrEye`] does the rest.

use js_sys::{Array, Float32Array, Function, Object, Promise, Reflect};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use super::{CameraInner, WebCamera, WebRenderer, WebScene};
use crate::math::Matrix4;
use crate::xr::HAND_JOINTS;
use crate::xr::web::XrSession;

fn set(o: &Object, key: &str, value: impl Into<JsValue>) {
    let _ = Reflect::set(o, &key.into(), &value.into());
}

fn matrix_or_null(m: Option<[f32; 16]>) -> JsValue {
    m.map_or(JsValue::NULL, |m| Float32Array::from(&m[..]).into())
}

/// An immersive WebXR session and its frame loop, for JavaScript.
#[wasm_bindgen]
pub struct WebXr {
    session: XrSession,
}

#[wasm_bindgen]
impl WebXr {
    /// Whether the browser can open a session of `mode` (`"immersive-vr"`,
    /// `"immersive-ar"`).
    #[wasm_bindgen(js_name = isSessionSupported)]
    pub async fn is_session_supported(mode: String) -> bool {
        XrSession::is_supported(&mode).await
    }

    /// Asks for a session of `mode` (from a click) drawn from `canvas`; see
    /// [`XrSession::request`].
    pub async fn start(
        mode: String,
        canvas: HtmlCanvasElement,
        reference_space: String,
        framebuffer_scale: f32,
        features: Option<Vec<String>>,
    ) -> Result<WebXr, JsValue> {
        let features = features.unwrap_or_default();
        Ok(WebXr { session: XrSession::request(&mode, canvas, &reference_space, framebuffer_scale, &features).await? })
    }

    /// Takes over a session the page opened (three.js `xr.setSession`).
    pub async fn attach(
        session: JsValue,
        canvas: HtmlCanvasElement,
        reference_space: String,
        framebuffer_scale: f32,
    ) -> Result<WebXr, JsValue> {
        Ok(WebXr { session: XrSession::attach(session, canvas, &reference_space, framebuffer_scale).await? })
    }

    /// Calls `callback(time, frame)` once per device frame (`null` stops
    /// calling it). Inside it, [`WebRenderer::render_xr`] draws the eyes;
    /// the result goes to the headset when it returns.
    #[wasm_bindgen(js_name = setAnimationLoop)]
    pub fn set_animation_loop(&self, callback: Option<Function>) {
        self.session.set_frame_callback(callback.map(|f| {
            Box::new(move |time: f64, frame: &JsValue| {
                if let Err(e) = f.call2(&JsValue::NULL, &time.into(), frame) {
                    web_sys::console::error_1(&e);
                }
            }) as crate::xr::web::FrameCallback
        }));
    }

    /// Ends the session (the `end` callback follows).
    pub fn end(&self) -> Promise {
        self.session.end()
    }

    /// Called once when the session ends, by the page or the user.
    #[wasm_bindgen(js_name = onEnd)]
    pub fn on_end(&self, callback: Option<Function>) {
        self.session.on_end(callback.map(|f| {
            Box::new(move || {
                let _ = f.call0(&JsValue::NULL);
            }) as Box<dyn FnOnce()>
        }));
    }

    #[wasm_bindgen(getter)]
    pub fn session(&self) -> JsValue {
        self.session.session()
    }

    #[wasm_bindgen(getter, js_name = referenceSpace)]
    pub fn reference_space(&self) -> JsValue {
        self.session.reference_space()
    }

    /// `"immersive-vr"` or `"immersive-ar"` (see-through).
    #[wasm_bindgen(getter)]
    pub fn mode(&self) -> String {
        self.session.mode()
    }

    #[wasm_bindgen(getter)]
    pub fn ended(&self) -> bool {
        self.session.ended()
    }

    #[wasm_bindgen(getter, js_name = framebufferWidth)]
    pub fn framebuffer_width(&self) -> u32 {
        self.session.framebuffer_size().0
    }

    #[wasm_bindgen(getter, js_name = framebufferHeight)]
    pub fn framebuffer_height(&self) -> u32 {
        self.session.framebuffer_size().1
    }

    /// The head's pose in the reference space this frame (column-major),
    /// or `None` outside a frame or while tracking is lost.
    #[wasm_bindgen(js_name = headMatrix)]
    pub fn head_matrix(&self) -> Option<Vec<f32>> {
        self.session.head().map(|m| m.to_vec())
    }

    /// This frame's eyes, 36 floats each: the device's projection (OpenGL
    /// depth, as three.js keeps it), the pose in the reference space, and
    /// the viewport `[x, y, width, height]`.
    #[wasm_bindgen(js_name = eyeData)]
    pub fn eye_data(&self) -> Vec<f32> {
        let mut out = vec![];
        for v in self.session.views() {
            out.extend_from_slice(&v.projection);
            out.extend_from_slice(&v.pose);
            out.extend(v.viewport.iter().map(|&x| x as f32));
        }
        out
    }

    /// Controller events since the last call, `[type, slot, inputSource]`
    /// each: `connected`, `disconnected`, `select(start|end)`,
    /// `squeeze(start|end)`. Slot `i` is three.js's `getController(i)`.
    #[wasm_bindgen(js_name = takeEvents)]
    pub fn take_events(&self) -> Array {
        self.session.take_events_with_sources().into_iter().map(|(e, source)| Array::of3(&e.kind.name().into(), &(e.slot as u32).into(), &source)).collect()
    }

    /// The connected inputs this frame, one object each: `slot`,
    /// `inputSource`, `handedness`, the `ray` (target ray) and `grip`
    /// matrices in the reference space (null when untracked), and for a
    /// tracked hand `joints` (25 matrices, in [`HAND_JOINTS`] order) and
    /// their `radii`.
    pub fn inputs(&self) -> Array {
        self.session
            .inputs()
            .into_iter()
            .map(|i| {
                let o = Object::new();
                set(&o, "slot", i.slot as u32);
                set(&o, "inputSource", self.session.source(i.slot).unwrap_or_default());
                set(&o, "handedness", format!("{:?}", i.handedness).to_lowercase());
                set(&o, "ray", matrix_or_null(i.ray));
                set(&o, "grip", matrix_or_null(i.grip));
                if let Some((joints, radii)) = i.joints {
                    set(&o, "joints", Float32Array::from(&joints[..]));
                    set(&o, "radii", Float32Array::from(&radii[..]));
                }
                JsValue::from(o)
            })
            .collect()
    }

    /// [`HAND_JOINTS`], for naming a hand's joints.
    #[wasm_bindgen(js_name = jointNames)]
    pub fn joint_names() -> Array {
        HAND_JOINTS.iter().map(|n| JsValue::from(*n)).collect()
    }
}

#[wasm_bindgen]
impl WebRenderer {
    /// Draws `scene` once per eye of `xr`'s current frame, through
    /// `camera`'s lens (its depth of field, layers and near/far) at the
    /// eye's pose under `origin` (column-major; the rig's world matrix,
    /// identity if empty). The canvas is sized to the headset's framebuffer
    /// and each eye cleared and drawn in its own rectangle; viewport,
    /// scissor and pixel ratio are put back after. False outside a frame
    /// or while tracking is lost (nothing drawn).
    #[wasm_bindgen(js_name = renderXr)]
    pub fn render_xr(&mut self, scene: &mut WebScene, camera: &WebCamera, xr: &WebXr, origin: Vec<f32>) -> bool {
        let CameraInner::Perspective(lens) = &camera.inner else { return false };
        let session = &xr.session;
        if !session.in_frame() {
            return false;
        }
        // The device clips with the camera's range from the next frame.
        session.set_depth_range(lens.near, lens.far);
        session.mark_drawn();
        let (w, h) = session.fit_canvas();
        self.set_size(w, h);
        let origin = <[f32; 16]>::try_from(&origin[..]).map(|elements| Matrix4 { elements }).unwrap_or_else(|_| Matrix4::identity());
        let r = &self.renderer;
        let saved = (r.viewport(), r.scissor(), r.scissor_test(), r.auto_clear(), r.pixel_ratio());
        self.renderer.set_pixel_ratio(1.0);
        self.renderer.set_auto_clear(true);
        self.set_scissor_test(true);
        for eye in session.eyes(&origin) {
            let eye_camera = WebCamera { inner: CameraInner::Perspective(eye.camera(lens)) };
            let [x, y, vw, vh] = eye.viewport.map(|v| v as f32);
            self.set_viewport(x, y, vw, vh);
            self.renderer.set_scissor(x, y, vw, vh);
            self.render(scene, &eye_camera);
        }
        let (viewport, scissor, scissor_test, auto_clear, pixel_ratio) = saved;
        let r = &mut self.renderer;
        match viewport {
            Some(v) => r.set_viewport(v.x, v.y, v.width, v.height),
            None => r.reset_viewport(),
        }
        match scissor {
            Some(v) => r.set_scissor(v.x, v.y, v.width, v.height),
            None => r.reset_scissor(),
        }
        r.set_scissor_test(scissor_test);
        r.set_auto_clear(auto_clear);
        r.set_pixel_ratio(pixel_ratio);
        true
    }
}
