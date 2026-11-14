//! threers-ui in a browser: one page that uses the grid layout, state
//! styles, tooltips, drag and drop, virtual lists of five kinds (see
//! [`lists`]) with scrollbars, rich text, a text field selected with the pointer, an image,
//! a container query, a modal dialog, notifications, a command palette,
//! menus, prompts, popovers, tabs, a split pane, the keymap, the
//! frame-stats overlay and the inspector — painted on a 2D canvas, with the
//! accessibility tree mirrored into the page.
//!
//! Build and serve with `web/ui-demo/build.sh`.

pub mod app;
#[cfg(target_arch = "wasm32")]
mod gpu;
pub mod lists;
pub mod pages;
pub mod progress;
pub mod shaders;
pub mod showcase;
pub mod tailwind;
pub mod toolkit;
// The demo in a headset: the browser's (WebXR) and visionOS's.
#[cfg(any(target_arch = "wasm32", target_os = "visionos"))]
pub mod xr;

#[cfg(target_arch = "wasm32")]
mod web {
    use threers_ui::a11y_dom::{A11yAction, DomAccessibility};
    use threers_ui::canvas::CanvasPainter;
    use threers_ui::ime::Ime;
    use threers_ui::input::{Event, PointerButton, TouchPhase};
    use threers_ui::KeyModifiers;
    use wasm_bindgen::prelude::*;
    use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    use threers::xr::phone::PhoneView;
    use threers::xr::web::XrSession;

    use crate::app::App;
    use crate::gpu::{Headset, View};

    /// The page's `localStorage`, for settings kept between visits.
    struct LocalStorage(web_sys::Storage);

    impl threers_ui::storage::Storage for LocalStorage {
        fn get(&self, key: &str) -> Option<String> {
            self.0
                .get_item(&format!("threers-ui-demo.{key}"))
                .ok()
                .flatten()
        }
        fn set(&self, key: &str, value: &str) {
            let _ = self.0.set_item(&format!("threers-ui-demo.{key}"), value);
        }
        fn remove(&self, key: &str) {
            let _ = self.0.remove_item(&format!("threers-ui-demo.{key}"));
        }
    }

    /// The demo, driven by the page: it forwards events and calls `frame`.
    #[wasm_bindgen]
    pub struct Demo {
        app: App,
        painter: Paint,
        mirror: Option<DomAccessibility>,
        size: (f32, f32, f64),
        /// Per frame: app (events applied, layout), paint, mirror, in ms.
        timings: Vec<[f64; 3]>,
        text_cache: threers_ui::text_cache::TextCache,
        /// The page's frame function, `tick(time)`: a headset session calls
        /// it for each of its frames (the window's stop while it runs).
        ticker: Option<js_sys::Function>,
        /// A session being opened, the one showing, and what the browser
        /// said it can do (VR, AR).
        xr_opening: Rc<RefCell<Option<Result<View, String>>>>,
        headset: Option<(Headset, Rc<Cell<bool>>)>,
        xr_support: Rc<Cell<Option<(bool, bool)>>>,
    }

    /// What draws the demo: the 2D canvas, or the GPU (WebGPU or WebGL2).
    enum Paint {
        Canvas(Box<CanvasPainter>),
        Gpu(Box<crate::gpu::Gpu>),
    }

    impl Paint {
        /// What measures text for layout and hit tests, as it will be drawn.
        fn measure(&mut self, dpr: f32) -> Box<dyn threers_ui::TextMeasure + '_> {
            match self {
                Paint::Canvas(c) => Box::new(Borrowed(&mut **c)),
                Paint::Gpu(g) => Box::new(g.pass.text_measure(dpr)),
            }
        }
    }

    /// A measurer lent out for one layout or event.
    struct Borrowed<'a, T>(&'a mut T);

    impl<T: threers_ui::TextMeasure> threers_ui::TextMeasure for Borrowed<'_, T> {
        fn measure(&mut self, text: &str, style: &threers_ui::Style) -> (f32, f32) {
            self.0.measure(text, style)
        }
    }

    /// Characters before `chars` characters, in UTF-16 units.
    fn utf16_of(text: &str, chars: usize) -> u32 {
        text.chars().take(chars).map(|c| c.len_utf16() as u32).sum()
    }

    /// Characters before UTF-16 unit `units` (a split pair counts whole).
    fn chars_of(text: &str, units: u32) -> usize {
        let mut seen = 0u32;
        let mut n = 0;
        for c in text.chars() {
            if seen >= units {
                break;
            }
            seen += c.len_utf16() as u32;
            n += 1;
        }
        n
    }

    fn button(b: i16) -> PointerButton {
        match b {
            1 => PointerButton::Middle,
            2 => PointerButton::Secondary,
            _ => PointerButton::Primary,
        }
    }

    #[wasm_bindgen]
    impl Demo {
        #[wasm_bindgen(constructor)]
        pub fn new(canvas: HtmlCanvasElement) -> Result<Demo, JsValue> {
            let ctx: CanvasRenderingContext2d = canvas
                .get_context("2d")?
                .ok_or("no 2D context")?
                .dyn_into()?;
            let mut app = App::new();
            if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
                app.set_storage(std::rc::Rc::new(LocalStorage(s)));
            }
            Ok(Demo {
                app,
                painter: Paint::Canvas(Box::new(CanvasPainter::new(ctx, 1.0))),
                mirror: DomAccessibility::new(&canvas),
                size: (1.0, 1.0, 1.0),
                timings: vec![],
                text_cache: Default::default(),
                ticker: None,
                xr_opening: Default::default(),
                headset: None,
                xr_support: Default::default(),
            })
        }

        /// The demo drawn on the GPU: WebGPU if the browser has it, else
        /// WebGL2 (only WebGL2 without `webgpu`). Fails (and leaves the canvas unusable for 2D: use a
        /// fresh one then) when there is neither.
        pub async fn new_gpu(canvas: HtmlCanvasElement, webgpu: bool) -> Result<Demo, JsValue> {
            let gpu = crate::gpu::Gpu::open(canvas.clone(), webgpu)
                .await
                .map_err(|e| JsValue::from_str(&e))?;
            let mut app = App::new();
            if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
                app.set_storage(std::rc::Rc::new(LocalStorage(s)));
            }
            let xr_support: Rc<Cell<Option<(bool, bool)>>> = Default::default();
            let found = xr_support.clone();
            wasm_bindgen_futures::spawn_local(async move {
                found.set(Some((
                    XrSession::is_supported("immersive-vr").await,
                    XrSession::is_supported("immersive-ar").await,
                )));
            });
            Ok(Demo {
                app,
                painter: Paint::Gpu(Box::new(gpu)),
                mirror: DomAccessibility::new(&canvas),
                size: (1.0, 1.0, 1.0),
                timings: vec![],
                text_cache: Default::default(),
                ticker: None,
                xr_opening: Default::default(),
                headset: None,
                xr_support,
            })
        }

        /// The face the GPU renderer draws text in, from a TrueType file's
        /// bytes: the page fetches one, as a browser has no font files to
        /// read. Call before [`new_gpu`](Self::new_gpu).
        pub fn set_gpu_font(bytes: &[u8]) -> Result<(), JsValue> {
            threers::captions::CaptionFont::set_ui(bytes)
                .map_err(|e| JsValue::from_str(&format!("{e:?}")))
        }

        /// The GPU renderer's face's bold file (fetched beside the regular).
        pub fn set_gpu_font_bold(bytes: &[u8]) -> Result<(), JsValue> {
            threers::captions::CaptionFont::set_ui_style(true, false, bytes)
                .map_err(|e| JsValue::from_str(&format!("{e:?}")))
        }

        /// Runs the demo's heavy work (the 3D model's drawing) on Web
        /// Workers, each with its own instance of this module, loaded from
        /// `glue_url` (this build's JavaScript, as an absolute URL).
        /// Returns how many workers there are.
        pub fn use_workers(&mut self, glue_url: &str) -> Result<usize, JsValue> {
            let workers = threers_ui::jobs::Workers::for_this_device(glue_url)?;
            let n = workers.len();
            self.app.ui.set_job_runner(workers);
            Ok(n)
        }

        /// Custom shaders the GPU would not compile in the last frame (drawn
        /// as they are instead); 0 on the 2D canvas.
        pub fn failed_shaders(&self) -> usize {
            match &self.painter {
                Paint::Canvas(_) => 0,
                Paint::Gpu(g) => g.pass.stats().failed_effects,
            }
        }

        /// The page's frame function: while a headset session runs, its
        /// frames call `tick(time)` instead of the window's.
        pub fn set_ticker(&mut self, tick: js_sys::Function) {
            self.ticker = Some(tick);
        }

        /// The XR page's request, if it made one: open a session (from the
        /// tap that asked, as browsers want) or end the one showing.
        fn start_xr(&mut self) {
            let Some(mode) = self.app.state.xr_request.take() else {
                return;
            };
            if mode == "end" {
                if let Some((h, _)) = &mut self.headset {
                    h.session.end();
                }
                return;
            }
            let Paint::Gpu(g) = &self.painter else {
                self.app.state.toast = Some("XR needs the GPU renderer".into());
                return;
            };
            let (canvas, opening, tick) =
                (g.canvas(), self.xr_opening.clone(), self.ticker.clone());
            wasm_bindgen_futures::spawn_local(async move {
                // The phone's look-around view (its sensors and camera), or
                // a WebXR session.
                let r = match mode.as_str() {
                    "phone" => PhoneView::start(canvas, true).await.map(View::Phone),
                    _ => XrSession::request(&mode, canvas, "local-floor", 1.0, &[])
                        .await
                        .map(View::Xr),
                };
                *opening.borrow_mut() =
                    Some(r.map_err(|e| e.as_string().unwrap_or_else(|| format!("{e:?}"))));
                // A frame to take it up in, the page's being idle.
                if let (Some(w), Some(t)) = (web_sys::window(), tick) {
                    let _ = w.request_animation_frame(&t);
                }
            });
        }

        /// A session opened (or failed), taken up; one ended, put away; and
        /// what the XR page says.
        fn track_xr(&mut self) {
            if let Some(opened) = self.xr_opening.borrow_mut().take() {
                match opened {
                    Ok(session) => {
                        let ended = Rc::new(Cell::new(false));
                        let (flag, tick) = (ended.clone(), self.ticker.clone());
                        session.on_end(Some(Box::new(move || {
                            flag.set(true);
                            if let (Some(w), Some(t)) = (web_sys::window(), &tick) {
                                let _ = w.request_animation_frame(t);
                            }
                        })));
                        // A WebXR session calls the page's frame function
                        // itself; the phone's view runs on the page's own.
                        if let (Some(t), View::Xr(s)) = (self.ticker.clone(), &session) {
                            s.set_frame_callback(Some(Box::new(move |time, _| {
                                let _ = t.call1(&JsValue::NULL, &time.into());
                            })));
                        }
                        let phone = matches!(session, View::Phone(_));
                        let mut headset = Headset::new(session);
                        // A phone's view is narrow: the panel stands back
                        // until its width fits it (a headset's is wide).
                        if phone && let Paint::Gpu(g) = &self.painter {
                            let c = g.canvas();
                            let aspect =
                                c.client_width().max(1) as f32 / c.client_height().max(1) as f32;
                            let half_across =
                                ((threers::xr::phone::FOV_Y.to_radians() / 2.0).tan() * aspect)
                                    .atan();
                            let half_width =
                                headset.ui.panel.size.0 * headset.ui.panel.metres_per_pixel / 2.0;
                            headset.ui.panel.distance =
                                (half_width / half_across.tan() * 1.06).clamp(1.3, 4.5);
                        }
                        self.headset = Some((headset, ended));
                    }
                    Err(e) => self.app.state.toast = Some(format!("Couldn't start XR: {e}")),
                }
            }
            if self.headset.as_ref().is_some_and(|(_, ended)| ended.get()) {
                self.headset = None;
                if let Paint::Gpu(g) = &mut self.painter {
                    g.leave_xr();
                }
                // The page's own size back.
                let (w, h, dpr) = self.size;
                if let Paint::Gpu(g) = &self.painter {
                    let c = g.canvas();
                    c.set_width((w as f64 * dpr).round() as u32);
                    c.set_height((h as f64 * dpr).round() as u32);
                }
            }
            let s = &mut self.app.state;
            s.xr_presenting = self.headset.is_some();
            let secure = web_sys::window().is_some_and(|w| w.is_secure_context());
            // The phone's view where there is no immersive WebXR to open.
            let gpu = matches!(self.painter, Paint::Gpu(_));
            let (vr, ar) = self.xr_support.get().unwrap_or((false, false));
            let phone =
                gpu && !vr && !ar && self.xr_support.get().is_some() && PhoneView::is_supported();
            s.xr_can = if gpu { [vr, ar, phone] } else { [false; 3] };
            let phone_showing = matches!(
                self.headset.as_ref().map(|h| &h.0.session),
                Some(View::Phone(_))
            );
            s.xr_status = match (&self.painter, self.headset.is_some(), self.xr_support.get()) {
                _ if phone_showing => "Looking around: turn the phone to look, tap the panel to press. Exit at the top left.".into(),
                (_, true, _) => "Showing in the headset. Exit here, or from the headset's own menu.".into(),
                (Paint::Canvas(_), ..) => "XR needs the GPU renderer, and this page draws on a 2D canvas.".into(),
                _ if !secure => "XR needs a secure page: open this demo over https (or on localhost). Safari keeps WebXR, the motion sensors and the camera from other pages.".into(),
                (_, _, None) => "Looking for a headset…".into(),
                (_, _, Some((true, true))) => "Ready for VR and AR.".into(),
                (_, _, Some((true, false))) => "Ready for VR (this browser has no AR).".into(),
                (_, _, Some((false, true))) => "Ready for AR (this browser has no VR).".into(),
                _ if phone => "This browser has no immersive WebXR (Safari on iPhone and iPad has none): Look around shows the demo around you instead, turned by the phone, its camera behind.".into(),
                (_, _, Some((false, false))) => "This browser has no WebXR headset to show in.".into(),
            };
        }

        /// One headset frame: the pointers to the UI, the UI laid out at
        /// the panel's size, drawn on the panel in each eye.
        fn xr_frame(&mut self, time: f64) {
            let (w, h) = crate::xr::PANEL_PX;
            let scale = crate::xr::PANEL_SCALE;
            self.app.ui.set_pixel_ratio(scale);
            let Some((headset, _)) = &mut self.headset else {
                return;
            };
            let t = time / 1000.0;
            let frame = threers::xr::Frame::take(&mut headset.session, t, t - headset.last);
            headset.last = t;
            headset
                .room
                .animate(&mut headset.scene, t, frame.see_through);
            {
                let mut m = self.painter.measure(scale);
                headset.ui.input(&mut self.app, &frame, &mut *m);
            }
            let performance = web_sys::window().and_then(|w| w.performance());
            let clock = move || performance.as_ref().map_or(0.0, |p| p.now());
            {
                let mut m = self.painter.measure(scale);
                let mut measure = self.text_cache.measure_with(&mut *m);
                self.app
                    .frame_timed(time / 1000.0, w, h, &mut measure, &clock);
            }
            self.text_cache.finish_frame();
            if let Paint::Gpu(g) = &mut self.painter {
                let mut scene = threers_ui::scene::Scene::default();
                self.app.paint(&mut scene.recorder());
                let bg = self
                    .app
                    .ui
                    .style(self.app.ui.root())
                    .background
                    .unwrap_or([0.0, 0.0, 0.0, 1.0]);
                g.draw_xr(&scene, scale, bg, headset);
            }
            if let Some(m) = &mut self.mirror {
                m.update(&self.app.ui);
            }
        }

        /// Labels keys as a Mac's (⌘) or not (Ctrl): the page says which it
        /// runs on.
        /// Follows a link (`location.hash`): `#page`, `#page/section` or
        /// `#section`. See `App::follow_link`.
        pub fn follow_link(&mut self, link: &str) -> bool {
            self.app.follow_link(link)
        }

        /// Where the demo is, as a link (without the `#`).
        pub fn link(&self) -> String {
            self.app.link()
        }

        /// The browser's `prefers-contrast: more`: the theme's quiet
        /// colours stronger.
        pub fn set_increase_contrast(&mut self, on: bool) {
            self.app.ui.set_increase_contrast(on);
        }

        /// The browser's `prefers-reduced-motion`, as the Accessibility
        /// page's own setting (which can still be changed there).
        pub fn set_reduce_motion(&mut self, on: bool) {
            self.app.state.reduce_motion = on;
        }

        pub fn set_mac(&mut self, mac: bool) {
            self.app.state.mac = mac;
        }

        /// What draws: `WebGPU`, `WebGL2` or `Canvas 2D`.
        pub fn backend(&self) -> String {
            match &self.painter {
                Paint::Canvas(_) => "Canvas 2D".into(),
                Paint::Gpu(g) => g.backend.into(),
            }
        }

        /// One frame at `time` milliseconds, `width × height` CSS pixels at
        /// `dpr` device pixels each. Returns whether another is wanted
        /// soon (otherwise draw again on the next event).
        pub fn frame(&mut self, time: f64, width: f32, height: f32, dpr: f64) -> bool {
            self.size = (width, height, dpr);
            self.start_xr();
            self.track_xr();
            // In a headset its frames draw (the page's are idle then); the
            // phone's view draws on the page's, every one.
            if let Some((h, _)) = &mut self.headset {
                let phone = matches!(h.session, View::Phone(_));
                if let View::Phone(p) = &mut h.session {
                    p.update();
                }
                if h.session.in_frame() {
                    self.xr_frame(time);
                    self.start_xr();
                }
                return phone;
            }
            self.app.ui.set_pixel_ratio(dpr as f32);
            for a in self
                .mirror
                .as_mut()
                .map(|m| m.take_actions())
                .unwrap_or_default()
            {
                match a {
                    A11yAction::Focus(n) => {
                        self.app
                            .input
                            .set_focus(&mut self.app.ui, &mut self.app.state, Some(n));
                    }
                    A11yAction::Activate(n) => {
                        let r = self.app.ui.rect(n);
                        let (x, y) = (r.x + r.width / 2.0, r.y + r.height / 2.0);
                        let b = PointerButton::Primary;
                        self.app.event(Event::PointerDown { x, y, button: b });
                        self.app.event(Event::PointerUp { x, y, button: b });
                    }
                    A11yAction::Custom(n, name) => {
                        self.app.input.accessibility_action(
                            &mut self.app.ui,
                            &mut self.app.state,
                            n,
                            &name,
                        );
                    }
                    // A screen reader's step: the arrow key, on it.
                    A11yAction::Increment(n) | A11yAction::Decrement(n) => {
                        let key = if matches!(a, A11yAction::Increment(_)) {
                            "ArrowRight"
                        } else {
                            "ArrowLeft"
                        };
                        self.app
                            .input
                            .set_focus(&mut self.app.ui, &mut self.app.state, Some(n));
                        self.app.key(key, KeyModifiers::default());
                    }
                }
            }
            let performance = web_sys::window().and_then(|w| w.performance());
            let clock = move || performance.as_ref().map_or(0.0, |p| p.now());
            let t0 = clock();
            {
                // Strings measured once while they show, not every layout.
                let mut m = self.painter.measure(dpr as f32);
                let mut measure = self.text_cache.measure_with(&mut *m);
                self.app
                    .frame_timed(time / 1000.0, width, height, &mut measure, &clock);
            }
            self.text_cache.finish_frame();
            let t1 = clock();
            let (w, h) = (
                (width as f64 * dpr).round() as u32,
                (height as f64 * dpr).round() as u32,
            );
            match &mut self.painter {
                Paint::Canvas(c) => {
                    c.begin(w, h, dpr);
                    self.app.paint(&mut **c);
                    c.finish();
                }
                Paint::Gpu(g) => {
                    // Recorded, then drawn in one go.
                    let mut scene = threers_ui::scene::Scene::default();
                    self.app.paint(&mut scene.recorder());
                    let bg = self
                        .app
                        .ui
                        .style(self.app.ui.root())
                        .background
                        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
                    g.draw(&scene, dpr as f32, bg);
                }
            }
            let t2 = clock();
            if let Some(m) = &mut self.mirror {
                m.update(&self.app.ui);
                // A link went somewhere: the page's focus follows, so a
                // screen reader reads where it landed.
                if let Some(n) = self.app.take_followed() {
                    m.focus(n);
                }
            }
            let t3 = clock();
            self.timings.push([t1 - t0, t2 - t1, t3 - t2]);
            for line in self.app.take_log() {
                web_sys::console::log_1(&line.into());
            }
            self.app.needs_frame()
        }

        pub fn pointer_move(&mut self, x: f32, y: f32) {
            // Showing elsewhere: the page's input is not the UI's.
            if self.headset.is_some() {
                return;
            }
            self.app
                .pointer_move(x, y, &mut *self.painter.measure(self.size.2 as f32));
        }

        pub fn pointer_down(&mut self, x: f32, y: f32, button_: i16, shift: bool) {
            // Showing elsewhere: the page's input is not the UI's.
            if self.headset.is_some() {
                return;
            }
            self.app.pointer_down(
                x,
                y,
                button(button_),
                shift,
                &mut *self.painter.measure(self.size.2 as f32),
            );
        }

        pub fn pointer_up(&mut self, x: f32, y: f32, button_: i16) {
            if self.headset.is_none() {
                self.app.pointer_up(x, y, button(button_));
            }
            // Within the tap: browsers (iOS for the motion sensors) want one.
            self.start_xr();
        }

        pub fn wheel(&mut self, x: f32, y: f32, dx: f32, dy: f32) {
            // Showing elsewhere: the page's input is not the UI's.
            if self.headset.is_some() {
                return;
            }
            self.app.wheel(x, y, dx, dy);
        }

        /// A finger: `phase` is `start`, `move`, `end` or `cancel`; `force`
        /// how firmly it presses, 0 to 1, or 0 where the screen cannot tell.
        pub fn touch(&mut self, id: f64, phase: &str, x: f32, y: f32, time: f64, force: f32) {
            if self.headset.is_some() {
                return;
            }
            let phase = match phase {
                "start" => TouchPhase::Start,
                "move" => TouchPhase::Move,
                "end" => TouchPhase::End,
                _ => TouchPhase::Cancel,
            };
            self.app.touch(
                id as u64,
                phase,
                x,
                y,
                time / 1000.0,
                &mut *self.painter.measure(self.size.2 as f32),
            );
            if force > 0.0 {
                self.app.touch_force(id as u64, force);
            }
            self.start_xr();
        }

        /// A key (DOM `KeyboardEvent.key`); returns whether to prevent the
        /// page's default.
        pub fn key(&mut self, key: &str, shift: bool, ctrl: bool, alt: bool, meta: bool) -> bool {
            let handled = self.app.key(
                key,
                KeyModifiers {
                    shift,
                    control: ctrl,
                    alt,
                    meta,
                },
            );
            self.start_xr();
            handled
        }

        /// Composition from the page's input method: `update` with the text
        /// so far, `end` with the final text.
        pub fn composition(&mut self, phase: &str, text: String) {
            let ime = match phase {
                "end" => Ime::Commit(text),
                _ => {
                    let n = text.len();
                    Ime::Preedit {
                        text,
                        cursor: Some((n, n)),
                    }
                }
            };
            self.app.event(Event::Ime(ime));
        }

        /// The frames' timings so far, as JSON `[[app, paint, mirror], …]`
        /// in milliseconds, emptied; and the layouts per frame.
        pub fn take_timings(&mut self) -> String {
            let t = std::mem::take(&mut self.timings);
            format!(
                "{{\"frames\":{:?},\"layouts\":{}}}",
                t, self.app.layouts_last
            )
        }

        pub fn cursor(&self) -> String {
            self.app.cursor().to_owned()
        }

        /// Whether focus is in the text field (the page then routes typing
        /// through its hidden input, for input methods).
        pub fn wants_text(&self) -> bool {
            self.app.wants_text()
        }

        /// The focused text control's text (the field's when none has
        /// focus), for the page's text input to mirror.
        pub fn field_text(&self) -> String {
            self.app.text_now().0
        }

        /// Its selection `[anchor, caret]` in UTF-16 units, as a text input
        /// counts them.
        pub fn field_selection(&self) -> Vec<u32> {
            let (t, a, c) = self.app.text_now();
            vec![utf16_of(&t, a), utf16_of(&t, c)]
        }

        /// The page's text input changed (typing, deleting, autocorrect, an
        /// input method, a paste): the field takes its text and selection,
        /// given in UTF-16 units.
        pub fn set_field(&mut self, text: String, anchor: u32, caret: u32) {
            let (a, c) = (chars_of(&text, anchor), chars_of(&text, caret));
            self.app.set_field(text, a, c);
        }

        /// The field's box `[x, y, width, height]` in CSS pixels, empty
        /// while a tap there would not reach it.
        pub fn field_rect(&self) -> Vec<f32> {
            self.app
                .field_rect()
                .map_or(vec![], |r| vec![r.x, r.y, r.width, r.height])
        }

        /// The field's box and its text's box, `[x, y, width, height]` each,
        /// in CSS pixels; empty while a tap there would not reach it.
        pub fn field_boxes(&self) -> Vec<f32> {
            self.app
                .field_rect_and_text()
                .map_or(vec![], |(r, t, _, _)| {
                    vec![r.x, r.y, r.width, r.height, t.x, t.y, t.width, t.height]
                })
        }

        /// The field text's CSS `font` and `color`, for the page's input to
        /// draw it in.
        pub fn field_font(&self) -> Vec<String> {
            self.app
                .field_rect_and_text()
                .map_or(vec![], |(_, _, font, c)| {
                    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                    vec![
                        font,
                        format!("rgba({},{},{},{})", b(c[0]), b(c[1]), b(c[2]), c[3]),
                    ]
                })
        }

        /// See [`App::set_native_field`].
        pub fn set_native_field(&mut self, native: bool) {
            self.app.set_native_field(native);
        }

        /// Text cut or copied from the edit menu, for the page to put on the
        /// clipboard (while it is still handling the tap that chose it).
        pub fn take_copy(&mut self) -> Option<String> {
            self.app.state.copy_out.take()
        }

        /// Whether the edit menu asked to paste: the page reads the
        /// clipboard and calls [`paste`](Self::paste).
        pub fn take_paste(&mut self) -> bool {
            std::mem::take(&mut self.app.state.paste_wanted)
        }

        /// Shows `message` in the demo's toast.
        pub fn notify(&mut self, message: String) {
            self.app.state.toast = Some(message);
        }

        /// Text the edit menu's Share… chose, for the page's share sheet.
        pub fn take_share(&mut self) -> Option<String> {
            self.app.state.share_out.take()
        }

        /// A link followed, for the page to open.
        pub fn take_open(&mut self) -> Option<String> {
            self.app.state.open_out.take()
        }

        /// The system's appearance (the page's `prefers-color-scheme`).
        pub fn set_system_dark(&mut self, dark: bool) {
            self.app.set_system_dark(dark);
        }

        /// How the device is held (`deviceorientation`'s `beta` and
        /// `gamma`, in degrees).
        pub fn set_device_tilt(&mut self, beta: f32, gamma: f32) {
            self.app.set_device_tilt(beta, gamma);
        }

        /// The page's safe-area insets, in CSS pixels.
        pub fn set_safe_area(&mut self, top: f32, right: f32, bottom: f32, left: f32) {
            self.app.set_safe_area(top, right, bottom, left);
        }

        /// Pastes `text` into the field.
        pub fn paste(&mut self, text: &str) {
            self.app.paste(text);
        }

        /// See [`App::debug_state`].
        pub fn debug_state(&self) -> String {
            self.app.debug_state()
        }

        /// See [`App::focus_field`].
        pub fn focus_field(&mut self, focus: bool) {
            self.app.focus_field(focus);
        }

        /// The caret's box `[x, y, width, height]` in CSS pixels, empty when
        /// the field does not have focus.
        pub fn ime_area(&self) -> Vec<f32> {
            self.app
                .ime_area()
                .map_or(vec![], |r| vec![r.x, r.y, r.width, r.height])
        }
    }
}
