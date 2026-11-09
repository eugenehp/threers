//! Text and shapes over the 3D picture.
//!
//! [`Hud`] is a canvas that starts each frame clear, so a system draws what the
//! frame should show. With the `ui` feature, a retained [`UiLayer`] of styled boxes
//! is drawn onto it and reports clicks as [`UiClicked`] events.

use crate::world::World;
use threers::overlay::Overlay;

/// The canvas over the game, as a resource. Hosts composite it after the scene.
pub struct Hud(pub Overlay);

impl Hud {
    /// A clear canvas of this size, in pixels.
    pub fn new(width: u32, height: u32) -> Self {
        Self(Overlay::new(width, height))
    }
}

impl std::ops::Deref for Hud {
    type Target = Overlay;
    fn deref(&self) -> &Overlay {
        &self.0
    }
}

impl std::ops::DerefMut for Hud {
    fn deref_mut(&mut self) -> &mut Overlay {
        &mut self.0
    }
}

/// Draw on the HUD if the game has one. Lets a game run in a test with no HUD.
pub fn with_hud(w: &mut World, draw: impl FnOnce(&mut Hud)) {
    if let Some(hud) = w.get_resource_mut::<Hud>() {
        draw(hud);
    }
}

#[cfg(feature = "ui")]
mod retained {
    use super::*;
    use crate::app::{App, Plugin};
    use crate::input::{Input, MouseButton};
    use threers::ui::{NodeId, OverlayPainter, Ui};

    /// A retained UI drawn over the game.
    #[derive(Default)]
    pub struct UiLayer {
        /// The UI tree; build it in a system and it is drawn over the game.
        pub ui: Ui,
    }

    /// The left button was released over this node.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct UiClicked(pub NodeId);

    /// Lays out, draws and hit-tests a [`UiLayer`] each frame.
    #[derive(Default)]
    pub struct UiPlugin;

    impl Plugin for UiPlugin {
        fn build(self, app: &mut App) {
            app.resource(UiLayer::default()).system(ui_step);
        }
    }

    /// Run the retained UI for this frame.
    pub fn ui_step(w: &mut World) {
        let clicked = w.resource::<Input>().was_released(MouseButton::Left);
        let at = w.resource::<Input>().mouse_position;
        w.with_resource::<UiLayer, _>(|w, layer| {
            let Some(hud) = w.get_resource_mut::<Hud>() else {
                return;
            };
            let (width, height) = hud.0.size();
            let mut painter = OverlayPainter::new(&mut hud.0);
            layer.ui.layout(width as f32, height as f32, &mut painter);
            layer.ui.paint(&mut painter);
            if clicked && let Some(node) = layer.ui.hit_test(at.x, at.y) {
                w.send(UiClicked(node));
            }
        });
    }
}

#[cfg(feature = "ui")]
pub use retained::{ui_step, UiClicked, UiLayer, UiPlugin};
