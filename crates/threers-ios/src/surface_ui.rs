//! An app's UI on one surface: asked of the app ([`App::ui`]), laid out and
//! painted where it changed ([`MetalUi`]).

use threers::metal::MetalUi;

use crate::{App, Surface};

#[derive(Default)]
pub(crate) struct SurfaceUi {
    pub pixels: MetalUi,
    /// Whether the app showed a UI last time.
    pub shown: bool,
    /// Its images are the GPU's to draw ([`on_gpu`](Self::on_gpu)).
    gpu_images: bool,
    /// The system's settings it was last given.
    legible: Option<crate::Legibility>,
}

impl SurfaceUi {
    /// A UI shown with [`MetalUi::draw`], its images drawn by the GPU. Any
    /// other is shown by its pixels alone, so they are all in them.
    pub fn on_gpu() -> Self {
        SurfaceUi { gpu_images: true, ..Self::default() }
    }

    /// The app's UI for `surface`, painted: whether there is one, and
    /// whether its pixels changed.
    pub fn update<A: App>(&mut self, app: &mut A, surface: &Surface) -> (bool, bool) {
        let Some(ui) = app.ui(surface) else {
            self.shown = false;
            return (false, false);
        };
        // The system's light or dark, for a UI whose theme follows it.
        if let Some(a) = crate::system_appearance() {
            ui.set_system_appearance(a);
        }
        // and its settings, when they change
        if let Some(l) = crate::legibility().filter(|l| self.legible != Some(*l)) {
            ui.set_bold_text(l.bold_text);
            ui.set_increase_contrast(l.increase_contrast);
            ui.set_reduce_motion(l.reduce_motion);
            ui.set_text_scale(l.text_scale);
            self.legible = Some(l);
        }
        let (w, h) = (
            (surface.width * surface.scale).round() as u32,
            (surface.height * surface.scale).round() as u32,
        );
        // its opaque pictures drawn by the GPU, where it draws the UI: a page
        // of them scrolled or zoomed moves there, not painted again
        self.pixels.set_gpu_images(self.gpu_images);
        let changed = self.pixels.update(ui, w, h, surface.scale) || !self.shown;
        // THREERS_PERF: the UI's layout and paint, ms, each 120 updates
        if std::env::var_os("THREERS_PERF").is_some() {
            thread_local!(static SUM: std::cell::Cell<(f32, f32, u32, u32)> = const { std::cell::Cell::new((0.0, 0.0, 0, 0)) });
            let (laid, painted, all) = self.pixels.last_times();
            SUM.with(|s| {
                let (l, p, n, a) = s.get();
                let next = (l + laid, p + painted, n + 1, a + u32::from(all));
                if next.2 >= 120 {
                    eprintln!("threers perf ui: layout {:.1} ms, paint {:.1} ms, painted whole {} of {}", next.0 / next.2 as f32, next.1 / next.2 as f32, next.3, next.2);
                    s.set((0.0, 0.0, 0, 0));
                } else {
                    s.set(next);
                }
            });
        }
        self.shown = true;
        (true, changed)
    }
}
