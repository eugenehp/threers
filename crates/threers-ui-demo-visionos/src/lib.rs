//! threers-ui's demo as a native Apple Vision Pro app: the whole demo on a
//! panel standing in a room of turning shapes, in a full immersive space.
//! Look at something and pinch to tap it; pinch and drag to scroll and
//! swipe. The system lights up the control looked at (its tracking areas:
//! the UI's controls, marked each frame); VoiceOver hears the UI's
//! announcements; and the system's text size, Reduce Motion, Bold Text and
//! Increase Contrast carry over.
//!
//! It is an ordinary [`xr::App`](threers::xr::App) made of the browser
//! build's headset pieces (`threers_ui_demo::xr`'s `Room` and `UiPanel`),
//! run by threers' visionOS runner and drawn by wgpu's Metal backend. The
//! shell around it is the one every threers Vision Pro app has
//! (`templates/visionos-app/app`); `app/build.sh` builds and runs it.

#[cfg(target_os = "visionos")]
mod script;

#[cfg(target_os = "visionos")]
mod app {
    use std::time::Instant;

    use threers::renderer::UiPass;
    use threers::xr::visionos::{announce, system_accessibility, SystemAccessibility};
    use threers::xr::{App as XrApp, Frame, Gpu};
    use threers::{PerspectiveCamera, Scene};
    use threers_ui_demo::app::App;
    use threers_ui_demo::pages::{DEPTH, PAGES};
    use threers_ui_demo::xr::{Room, UiPanel, PANEL_PX, PANEL_SCALE};

    use crate::script::Player;

    /// The face text is drawn in, as the browser build fetches it.
    const FONT: &[u8] = include_bytes!("../../../src/openscad/DejaVuSans.ttf");
    const BOLD: &[u8] = include_bytes!("../../../web/ui-demo/DejaVuSans-Bold.ttf");

    /// The demo as a Vision Pro app: its whole UI on a panel standing in a
    /// room, pinched or poked like a screen.
    pub struct VisionDemo {
        app: App,
        room: Room,
        ui: UiPanel,
        /// The UI's painter, made with the device on the first draw.
        pass: Option<UiPass>,
        /// `THREERS_XR_SCRIPT`'s pinches and pokes (see `script`).
        player: Option<Player>,
        /// `THREERS_DEMO_SHOW=candles`: scroll to the candles once laid out.
        to_candles: bool,
        /// The page last reported, for a script's run to be read back.
        shown_page: usize,
        start: Instant,
        /// The system's accessibility settings as last carried over.
        system: Option<SystemAccessibility>,
        /// The last UI announcement passed to VoiceOver.
        announced: u64,
    }

    impl XrApp for VisionDemo {
        fn setup(scene: &mut Scene) -> Self {
            if let Err(e) = threers::captions::CaptionFont::set_ui(FONT).and_then(|_| threers::captions::CaptionFont::set_ui_style(true, false, BOLD)) {
                eprintln!("threers: no UI font: {e:?}");
            }
            let mut app = App::new();
            app.state.xr_status = "Running natively on Apple Vision Pro: look at something and pinch, or touch it.".into();
            app.ui.set_pixel_ratio(PANEL_SCALE);
            // Eyes point less precisely than fingers: the audit holds
            // controls to visionOS's 60 points.
            app.ui.set_min_target(threers_ui::spatial::SPATIAL_MIN_TARGET);
            let to_candles = std::env::var("THREERS_DEMO_SHOW").is_ok_and(|v| v == "candles");
            if to_candles {
                app.state.page = DEPTH;
            }
            VisionDemo { app, room: Room::furnish(scene), ui: UiPanel::default(), pass: None, player: Player::from_env(), to_candles, shown_page: usize::MAX, start: Instant::now(), system: None, announced: 0 }
        }

        fn lens(&self) -> PerspectiveCamera {
            threers_ui_demo::xr::lens()
        }

        fn update(&mut self, scene: &mut Scene, frame: &Frame) {
            self.follow_system();
            self.room.animate(scene, frame.time, frame.see_through);
            if let (Some(p), Some(head), Some(plane)) = (&mut self.player, frame.head, self.ui.panel.plane()) {
                for done in p.step(frame.time, &head, &plane, PANEL_PX, &mut threers::xr::visionos::set_scripted_fingertip) {
                    eprintln!("threers script: {:?} at {:.1}s", done.gesture, frame.time);
                }
                if p.done() {
                    eprintln!("threers script: done");
                    self.player = None;
                }
            }
        }

        fn draw(&mut self, scene: &mut Scene, frame: &Frame, gpu: &mut Gpu<'_>) {
            let pass = self.pass.get_or_insert_with(|| UiPass::new(gpu.device));
            // As sharp as the eyes see the panel: nearer, finer.
            if self.ui.update_scale(gpu) {
                self.app.ui.set_pixel_ratio(self.ui.scale);
            }
            let scale = self.ui.scale;
            self.ui.input(&mut self.app, frame, &mut pass.text_measure(scale));
            // Already here: the XR page's Enter and Exit have nothing to do.
            if self.app.state.xr_request.take().is_some() {
                self.app.state.toast = Some("Close the space from the visionOS menu".into());
            }
            let start = self.start;
            let clock = move || start.elapsed().as_secs_f64() * 1000.0;
            self.app.frame_timed(frame.time, PANEL_PX.0, PANEL_PX.1, &mut pass.text_measure(scale), &clock);
            if self.to_candles && let Some(c) = self.app.showcase.candles.first() {
                self.app.ui.scroll_into_view(c.root);
                self.to_candles = false;
            }
            let mut ui_scene = threers::ui::scene::Scene::default();
            self.app.paint(&mut ui_scene.recorder());
            let clear = self.app.ui.style(self.app.ui.root()).background.unwrap_or([0.0, 0.0, 0.0, 1.0]);
            self.ui.draw(gpu, pass, &ui_scene, scale, clear, scene, &threers_ui_demo::xr::lens());
            // The controls, for the system to light up the one looked at.
            let regions = self.app.ui.hover_regions();
            let marked = self.ui.mark_hover(gpu, &regions);
            if std::env::var_os("THREERS_XR_DEBUG").is_some() && (frame.time * 2.0).fract() < frame.dt * 2.0 {
                eprintln!("threers demo: {marked} of {} controls marked, texture scale {}", regions.len(), self.ui.scale);
            }
            // What the UI announces ("Saved"), read out by VoiceOver.
            for (n, message, assertive) in self.app.ui.announcements_since(self.announced) {
                announce(&message, assertive);
                self.announced = n;
            }
            if self.app.state.page != self.shown_page {
                self.shown_page = self.app.state.page;
                eprintln!("threers: page {}", PAGES[self.shown_page.min(PAGES.len() - 1)].0);
            }
            for line in self.app.take_log() {
                eprintln!("{line}");
            }
        }
    }

    impl VisionDemo {
        /// The system's text size, Reduce Motion, Bold Text and Increase
        /// Contrast, when they change (the Accessibility page's own text
        /// steps and motion setting can still be changed there).
        fn follow_system(&mut self) {
            let Some(a) = system_accessibility().filter(|a| Some(*a) != self.system) else { return };
            self.system = Some(a);
            if std::env::var_os("THREERS_XR_DEBUG").is_some() {
                eprintln!("threers demo: system settings {a:?}");
            }
            self.app.state.reduce_motion = a.reduce_motion;
            // The system's size as it is, under the page's own steps.
            self.app.state.system_text_scale = a.text_scale;
            self.app.ui.set_bold_text(a.bold_text);
            self.app.ui.set_system_appearance(if a.dark { threers_ui::states::Appearance::Dark } else { threers_ui::states::Appearance::Light });
            self.app.ui.set_increase_contrast(a.increase_contrast);
        }
    }

    threers::visionos_app!(VisionDemo);
}
