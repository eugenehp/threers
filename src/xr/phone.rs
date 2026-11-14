//! A phone as a window into the scene, where the browser has no immersive
//! WebXR to give (Safari on iPhone and iPad): an [`XrSource`] that turns
//! the view with the phone (its motion sensors: turning, not walking, three
//! degrees of freedom), shows the rear camera behind what it draws (or the
//! scene's own room, without one), and gives taps on the screen as fingers
//! (a ray from the eye through the point, a select while it is down). The
//! same apps, panels and rooms a headset shows draw into it unchanged.
//!
//! The page must be secure (https): browsers keep motion sensors and the
//! camera from others. Start it from a tap — iOS asks the user for the
//! motion sensors then, and only then.
//!
//! ```no_run
//! # async fn f(canvas: web_sys::HtmlCanvasElement) {
//! use threers::xr::phone::PhoneView;
//! let view = PhoneView::start(canvas, true).await.unwrap();
//! // Each page frame: Frame::take(&mut view, …), draw to the canvas at
//! // view.fit_canvas(), then view.mark_drawn().
//! # }
//! ```

use crate::math::{Quaternion, Vector3};

/// The eye's height above the floor, metres: a person standing.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub(crate) const EYE_HEIGHT: f32 = 1.6;
/// The view's height in degrees: about a phone camera's, held upright.
pub const FOV_Y: f32 = 60.0;

/// The phone's attitude as a rotation, from the browser's device
/// orientation (degrees: `alpha` about the vertical, `beta` front to back,
/// `gamma` side to side) and the screen's turn (`orient`, degrees), as a
/// camera looking out of the screen's back (−z) with y up the screen.
pub fn attitude(alpha: f32, beta: f32, gamma: f32, orient: f32) -> Quaternion {
    let axis = |x: f32, y: f32, z: f32, deg: f32| {
        Quaternion::from_axis_angle(Vector3::new(x, y, z), deg.to_radians())
    };
    // The device's frame in the world's (y up): turned by alpha about y,
    // then beta about x, then −gamma about z; then the camera, looking out
    // of the screen, a quarter turn back about x; then the screen's turn.
    axis(0.0, 1.0, 0.0, alpha)
        * axis(1.0, 0.0, 0.0, beta)
        * axis(0.0, 0.0, 1.0, -gamma)
        * axis(1.0, 0.0, 0.0, -90.0)
        * axis(0.0, 0.0, 1.0, -orient)
}

/// A perspective projection in OpenGL's depth (as a headset gives one).
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub(crate) fn gl_projection(fov_y: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fov_y.to_radians() / 2.0).tan();
    [
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        f,
        0.0,
        0.0,
        0.0,
        0.0,
        (far + near) / (near - far),
        -1.0,
        0.0,
        0.0,
        2.0 * far * near / (near - far),
        0.0,
    ]
}

// The browser's side: the sensors, the camera, the fingers.
#[cfg(target_arch = "wasm32")]
mod browser {
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    use js_sys::{Function, Object, Promise, Reflect};
    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::{JsCast, JsValue};
    use wasm_bindgen_futures::JsFuture;
    use web_sys::HtmlCanvasElement;

    use super::{attitude, gl_projection, EYE_HEIGHT, FOV_Y};
    use crate::math::{Matrix4, Quaternion, Vector3};
    use crate::xr::{
        ray_pose, InputSlots, TargetRayMode, XrEvent, XrEventKind, XrEye, XrInput, XrSource,
    };

    fn get(o: &JsValue, key: &str) -> JsValue {
        Reflect::get(o, &key.into()).unwrap_or(JsValue::UNDEFINED)
    }

    fn call(o: &JsValue, name: &str, args: &[JsValue]) -> Result<JsValue, JsValue> {
        let f: Function = get(o, name)
            .dyn_into()
            .map_err(|_| JsValue::from_str(&format!("no {name}")))?;
        f.apply(o, &args.iter().collect())
    }

    /// What the page's listeners hand over between frames.
    #[derive(Default)]
    struct Shared {
        /// alpha, beta, gamma, screen turn; `None` until the sensors report.
        orientation: Option<[f32; 4]>,
        /// Fingers down: id, where (0..1 of the canvas across and down), and
        /// whether it began or ended since the last frame.
        touches: Vec<(i32, f32, f32, TouchState)>,
        ended: bool,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum TouchState {
        Began,
        Down,
        Ended,
    }

    type Listener = Closure<dyn FnMut(web_sys::Event)>;
    /// What to call once it ends.
    type OnEnd = Rc<RefCell<Option<Box<dyn FnOnce()>>>>;

    /// A phone looking into the scene. See the module docs.
    pub struct PhoneView {
        canvas: HtmlCanvasElement,
        /// The camera feed behind the canvas, and its stream (stopped at the end).
        video: Option<web_sys::HtmlElement>,
        stream: JsValue,
        exit: Option<web_sys::HtmlElement>,
        shared: Rc<RefCell<Shared>>,
        listeners: Vec<(JsValue, &'static str, Listener)>,
        /// The canvas's style before, put back at the end.
        style_before: String,
        head: Option<[f32; 16]>,
        size: (u32, u32),
        near: f32,
        far: f32,
        slots: InputSlots<i32>,
        rays: Vec<(i32, [f32; 16])>,
        events: Vec<XrEvent>,
        leaving: Vec<i32>,
        on_end: OnEnd,
        ended: Rc<Cell<bool>>,
    }

    impl PhoneView {
        /// Whether this browser could show one: motion sensors and a touch
        /// screen (a desktop has neither), on a secure page.
        pub fn is_supported() -> bool {
            let Some(w) = web_sys::window() else {
                return false;
            };
            let secure = w.is_secure_context();
            let sensors = !get(&w.clone().into(), "DeviceOrientationEvent").is_undefined();
            let touch = !get(&w.clone().into(), "ontouchstart").is_undefined()
                || w.navigator().max_touch_points() > 0;
            secure && sensors && touch
        }

        /// Starts it over `canvas` (from a tap): asks for the motion sensors
        /// where the browser asks (iOS), and with `camera`, the rear camera
        /// behind the canvas (without it, or if refused, the scene's room
        /// shows). Fails if the motion sensors are refused.
        pub async fn start(canvas: HtmlCanvasElement, camera: bool) -> Result<PhoneView, JsValue> {
            let window = web_sys::window().ok_or("no window")?;
            let doc = window.document().ok_or("no document")?;
            // iOS: the user's leave for the motion sensors, asked once.
            let doe = get(&window.clone().into(), "DeviceOrientationEvent");
            if !get(&doe, "requestPermission").is_undefined() {
                let answer =
                    JsFuture::from(Promise::from(call(&doe, "requestPermission", &[])?)).await?;
                if answer.as_string().as_deref() != Some("granted") {
                    return Err("The motion sensors were not allowed: allow them in Settings → Safari → Motion & Orientation Access".into());
                }
            }
            // The rear camera, as a video behind the canvas.
            let (mut video, mut stream) = (None, JsValue::UNDEFINED);
            if camera {
                let devices = get(&window.navigator().into(), "mediaDevices");
                let want = Object::new();
                let video_want = Object::new();
                let _ = Reflect::set(&video_want, &"facingMode".into(), &"environment".into());
                let _ = Reflect::set(&want, &"video".into(), &video_want);
                let _ = Reflect::set(&want, &"audio".into(), &false.into());
                if let Ok(p) = call(&devices, "getUserMedia", &[want.into()])
                    && let Ok(s) = JsFuture::from(Promise::from(p)).await
                {
                    let v: web_sys::HtmlElement = doc.create_element("video")?.dyn_into()?;
                    for (k, val) in [("autoplay", ""), ("playsinline", ""), ("muted", "")] {
                        let _ = v.set_attribute(k, val);
                    }
                    let _ = v.set_attribute("style", "position:fixed;inset:0;width:100vw;height:100vh;object-fit:cover;z-index:0;background:#000");
                    let _ = Reflect::set(&v, &"muted".into(), &true.into());
                    let _ = Reflect::set(&v, &"srcObject".into(), &s);
                    doc.body().ok_or("no body")?.append_child(&v)?;
                    let _ = call(&v, "play", &[]);
                    (video, stream) = (Some(v), s);
                }
            }
            // The canvas over it, filling the screen, see-through.
            let style_before = canvas.get_attribute("style").unwrap_or_default();
            let _ = canvas.set_attribute("style", "position:fixed;inset:0;width:100vw;height:100vh;z-index:1;background:transparent;touch-action:none");
            // A way out, over it all.
            let ended = Rc::new(Cell::new(false));
            let on_end: OnEnd = Default::default();
            let exit: web_sys::HtmlElement = doc.create_element("button")?.dyn_into()?;
            exit.set_text_content(Some("Exit"));
            let _ = exit.set_attribute("aria-label", "Exit the look-around view");
            let _ = exit.set_attribute(
                "style",
                "position:fixed;top:calc(env(safe-area-inset-top) + 12px);left:12px;z-index:2;padding:10px 18px;border:0;border-radius:22px;background:rgba(20,22,28,0.72);color:#fff;font:600 16px system-ui",
            );
            doc.body().ok_or("no body")?.append_child(&exit)?;
            let shared: Rc<RefCell<Shared>> = Default::default();
            let mut listeners: Vec<(JsValue, &'static str, Listener)> = vec![];
            // The sensors.
            {
                let shared = shared.clone();
                let win = window.clone();
                let f = Closure::wrap(Box::new(move |e: web_sys::Event| {
                    let n = |k: &str| get(&e, k).as_f64().map(|v| v as f32);
                    let orient = get(
                        &get(&get(&win.clone().into(), "screen"), "orientation"),
                        "angle",
                    )
                    .as_f64()
                    .map(|v| v as f32)
                    .or_else(|| {
                        get(&win.clone().into(), "orientation")
                            .as_f64()
                            .map(|v| v as f32)
                    })
                    .unwrap_or(0.0);
                    if let (Some(a), Some(b), Some(g)) = (n("alpha"), n("beta"), n("gamma")) {
                        shared.borrow_mut().orientation = Some([a, b, g, orient]);
                    }
                }) as Box<dyn FnMut(web_sys::Event)>);
                listeners.push((window.clone().into(), "deviceorientation", f));
            }
            // Fingers on the canvas.
            for (name, state) in [
                ("touchstart", TouchState::Began),
                ("touchmove", TouchState::Down),
                ("touchend", TouchState::Ended),
                ("touchcancel", TouchState::Ended),
            ] {
                let (shared, area) = (shared.clone(), canvas.clone());
                let f = Closure::wrap(Box::new(move |e: web_sys::Event| {
                    e.prevent_default();
                    let r = area.get_bounding_client_rect();
                    let list = get(&e, "changedTouches");
                    let n = get(&list, "length").as_f64().unwrap_or(0.0) as u32;
                    let mut s = shared.borrow_mut();
                    for i in 0..n {
                        let t = call(&list, "item", &[i.into()]).unwrap_or(JsValue::UNDEFINED);
                        let id = get(&t, "identifier").as_f64().unwrap_or(0.0) as i32;
                        let u = ((get(&t, "clientX").as_f64().unwrap_or(0.0) - r.left())
                            / r.width().max(1.0)) as f32;
                        let v = ((get(&t, "clientY").as_f64().unwrap_or(0.0) - r.top())
                            / r.height().max(1.0)) as f32;
                        match s.touches.iter_mut().find(|t| t.0 == id) {
                            Some(t) => {
                                (t.1, t.2) = (u, v);
                                if state == TouchState::Ended {
                                    t.3 = TouchState::Ended;
                                }
                            }
                            None if state != TouchState::Ended => {
                                s.touches.push((id, u, v, TouchState::Began))
                            }
                            None => {}
                        }
                    }
                }) as Box<dyn FnMut(web_sys::Event)>);
                listeners.push((canvas.clone().into(), name, f));
            }
            // The way out.
            {
                let shared = shared.clone();
                let f = Closure::wrap(Box::new(move |_e: web_sys::Event| {
                    shared.borrow_mut().ended = true;
                }) as Box<dyn FnMut(web_sys::Event)>);
                listeners.push((exit.clone().into(), "click", f));
            }
            for (target, name, f) in &listeners {
                let opts = Object::new();
                let _ = Reflect::set(&opts, &"passive".into(), &false.into());
                let _ = call(
                    target,
                    "addEventListener",
                    &[(*name).into(), f.as_ref().clone(), opts.into()],
                );
            }
            Ok(PhoneView {
                canvas,
                video,
                stream,
                exit: Some(exit),
                shared,
                listeners,
                style_before,
                head: None,
                size: (1, 1),
                near: 0.1,
                far: 100.0,
                slots: InputSlots::new(),
                rays: vec![],
                events: vec![],
                leaving: vec![],
                on_end,
                ended,
            })
        }

        /// Reads the sensors and the fingers since the last frame: the head,
        /// and the taps as inputs. Call once a frame, before taking the frame.
        pub fn update(&mut self) {
            if self.ended.get() {
                return;
            }
            if self.shared.borrow().ended {
                self.end();
                return;
            }
            let o = self.shared.borrow().orientation;
            let turn = o.map_or(Quaternion::IDENTITY, |[a, b, g, s]| attitude(a, b, g, s));
            let pose = Matrix4::compose(
                Vector3::new(0.0, EYE_HEIGHT, 0.0),
                turn,
                Vector3::new(1.0, 1.0, 1.0),
            );
            self.head = Some(pose.elements);
            // Fingers that lifted last frame are let go.
            for id in std::mem::take(&mut self.leaving) {
                if let Some(slot) = self.slots.disconnect(&id) {
                    self.events.push(XrEvent {
                        kind: XrEventKind::Disconnected,
                        slot,
                    });
                }
                self.rays.retain(|r| r.0 != id);
            }
            let aspect = self.size.0.max(1) as f32 / self.size.1.max(1) as f32;
            let k = (FOV_Y.to_radians() / 2.0).tan();
            let touches = std::mem::take(&mut self.shared.borrow_mut().touches);
            let mut keep = vec![];
            for (id, u, v, state) in touches {
                // From the eye through the point on the screen.
                let dir_cam = Vector3::new((u * 2.0 - 1.0) * k * aspect, (1.0 - v * 2.0) * k, -1.0)
                    .normalize();
                let dir = dir_cam.apply_quaternion(turn);
                let ray = ray_pose(Vector3::new(0.0, EYE_HEIGHT, 0.0), dir);
                self.rays.retain(|r| r.0 != id);
                self.rays.push((id, ray));
                if state == TouchState::Began || self.slots.slot_of(&id).is_none() {
                    let slot = self.slots.connect(id);
                    self.events.push(XrEvent {
                        kind: XrEventKind::Connected,
                        slot,
                    });
                    self.events.push(XrEvent {
                        kind: XrEventKind::SelectStart,
                        slot,
                    });
                }
                if state == TouchState::Ended {
                    if let Some(slot) = self.slots.slot_of(&id) {
                        self.events.push(XrEvent {
                            kind: XrEventKind::SelectEnd,
                            slot,
                        });
                        self.events.push(XrEvent {
                            kind: XrEventKind::Select,
                            slot,
                        });
                    }
                    self.leaving.push(id);
                } else {
                    keep.push((id, u, v, TouchState::Down));
                }
            }
            // Still down: kept for the next frame, where they are.
            let mut s = self.shared.borrow_mut();
            for t in keep {
                if !s.touches.iter().any(|x| x.0 == t.0) {
                    s.touches.push(t);
                }
            }
        }

        /// Sizes the canvas's backing to the screen (CSS pixels × the device's
        /// ratio) and returns it: what to draw at.
        pub fn fit_canvas(&mut self) -> (u32, u32) {
            let dpr = web_sys::window().map_or(1.0, |w| w.device_pixel_ratio());
            let r = self.canvas.get_bounding_client_rect();
            let (w, h) = (
                ((r.width() * dpr).round() as u32).max(1),
                ((r.height() * dpr).round() as u32).max(1),
            );
            if (self.canvas.width(), self.canvas.height()) != (w, h) {
                self.canvas.set_width(w);
                self.canvas.set_height(h);
            }
            self.size = (w, h);
            (w, h)
        }

        /// The near and far planes the eye's projection uses.
        pub fn set_depth_range(&mut self, near: f32, far: f32) {
            (self.near, self.far) = (near.max(0.01), far.max(near + 0.01));
        }

        /// Nothing to hand on (the canvas is the page's own); for symmetry with
        /// a WebXR session.
        pub fn mark_drawn(&self) {}

        /// Whether it has been ended (its Exit button, or [`end`](Self::end)).
        pub fn ended(&self) -> bool {
            self.ended.get()
        }

        /// Calls `callback` once it ends.
        pub fn on_end(&self, callback: Option<Box<dyn FnOnce()>>) {
            *self.on_end.borrow_mut() = callback;
        }

        /// Ends it: the camera stopped, the video and the Exit button taken
        /// away, the canvas as it was, the listeners gone.
        pub fn end(&mut self) {
            if self.ended.replace(true) {
                return;
            }
            if !self.stream.is_undefined()
                && let Ok(tracks) = call(&self.stream, "getTracks", &[])
            {
                let n = get(&tracks, "length").as_f64().unwrap_or(0.0) as u32;
                for i in 0..n {
                    let _ = call(
                        &Reflect::get_u32(&tracks, i).unwrap_or(JsValue::UNDEFINED),
                        "stop",
                        &[],
                    );
                }
            }
            for e in [self.video.take(), self.exit.take()].into_iter().flatten() {
                e.remove();
            }
            let _ = self.canvas.set_attribute("style", &self.style_before);
            for (target, name, f) in self.listeners.drain(..) {
                let _ = call(
                    &target,
                    "removeEventListener",
                    &[name.into(), f.as_ref().clone()],
                );
            }
            if let Some(f) = self.on_end.borrow_mut().take() {
                f();
            }
        }
    }

    impl Drop for PhoneView {
        fn drop(&mut self) {
            self.end();
        }
    }

    impl XrSource for PhoneView {
        fn head(&self) -> Option<[f32; 16]> {
            self.head
        }
        fn eyes(&self, origin: &Matrix4) -> Vec<XrEye> {
            let Some(head) = self.head else { return vec![] };
            let (w, h) = self.size;
            let projection = gl_projection(
                FOV_Y,
                w.max(1) as f32 / h.max(1) as f32,
                self.near,
                self.far,
            );
            vec![XrEye::new(
                projection,
                origin,
                &Matrix4 { elements: head },
                [0, 0, w as i32, h as i32],
            )]
        }
        fn framebuffer_size(&self) -> (u32, u32) {
            self.size
        }
        fn inputs(&self) -> Vec<XrInput> {
            self.slots
                .iter()
                .map(|(slot, id)| XrInput {
                    slot,
                    target_ray_mode: TargetRayMode::Screen,
                    ray: self.rays.iter().find(|r| r.0 == *id).map(|r| r.1),
                    ..Default::default()
                })
                .collect()
        }
        fn take_events(&mut self) -> Vec<XrEvent> {
            std::mem::take(&mut self.events)
        }
        /// The camera shows through where nothing is drawn.
        fn see_through(&self) -> bool {
            self.video.is_some()
        }
    }
}
#[cfg(target_arch = "wasm32")]
pub use browser::PhoneView;

#[cfg(test)]
mod tests {
    use super::*;

    /// Held upright facing north (beta 90: the screen standing up), the
    /// camera looks out level along −z with y up; turned left by alpha, it
    /// looks left; tipped flat (beta 0, screen to the sky), it looks down.
    #[test]
    fn the_phones_attitude_turns_the_camera() {
        let look = |q: Quaternion| Vector3::new(0.0, 0.0, -1.0).apply_quaternion(q);
        let up = |q: Quaternion| Vector3::new(0.0, 1.0, 0.0).apply_quaternion(q);
        let near = |a: Vector3, b: Vector3| (a - b).length() < 1e-4;
        let upright = attitude(0.0, 90.0, 0.0, 0.0);
        assert!(
            near(look(upright), Vector3::new(0.0, 0.0, -1.0)),
            "{:?}",
            look(upright)
        );
        assert!(near(up(upright), Vector3::new(0.0, 1.0, 0.0)));
        // alpha grows anticlockwise seen from above: to the left (−x).
        let left = attitude(90.0, 90.0, 0.0, 0.0);
        assert!(
            near(look(left), Vector3::new(-1.0, 0.0, 0.0)),
            "{:?}",
            look(left)
        );
        let flat = attitude(0.0, 0.0, 0.0, 0.0);
        assert!(
            near(look(flat), Vector3::new(0.0, -1.0, 0.0)),
            "{:?}",
            look(flat)
        );
        // Held sideways (the screen turned 90°): the picture's up is the
        // phone's side, still the world's up.
        let landscape = attitude(0.0, 0.0, -90.0, 90.0);
        assert!(
            near(up(landscape), Vector3::new(0.0, 1.0, 0.0)),
            "{:?}",
            up(landscape)
        );
    }
}
