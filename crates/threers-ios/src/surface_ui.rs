//! An app's UI on one surface: asked of the app ([`App::ui`]), laid out and
//! painted where it changed ([`MetalUi`]).

use threers::metal::MetalUi;

use crate::{App, Surface};

#[derive(Default)]
pub(crate) struct SurfaceUi {
    pub pixels: MetalUi,
    /// Whether the app showed a UI last time.
    pub shown: bool,
}

impl SurfaceUi {
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
        let (w, h) = ((surface.width * surface.scale).round() as u32, (surface.height * surface.scale).round() as u32);
        let changed = self.pixels.update(ui, w, h, surface.scale) || !self.shown;
        self.shown = true;
        (true, changed)
    }
}
