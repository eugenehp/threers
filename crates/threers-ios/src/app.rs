//! What an app is to threers-ios: a scene it builds and updates, and a 2D
//! UI (threers-ui) that threers-ios shows wherever the app is — over the
//! picture on a screen, on a panel in the room, or as an image beside a
//! volume — and sends the touches on it back to.
//!
//! The scene is drawn by threers' Metal renderer
//! ([`MetalRenderer`](threers::metal::MetalRenderer)) in every mode; the UI
//! is painted on the CPU where it changed
//! ([`MetalUi`](threers::metal::MetalUi)) and blended over the frame, or
//! put on the panel as a texture. An app has no GPU code of its own.

use threers::math::{Matrix4, Vector3};
use threers::ui::input::Event;
use threers::ui::Ui;
use threers::xr::Frame;
use threers::{PerspectiveCamera, Scene, ToneMapping};

/// Where an app's UI is shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    /// Over the picture, filling the screen: a phone, an iPad, a window on
    /// Apple Vision Pro, a Mac (the 3D view and AR alike). Leave the
    /// middle clear: the scene is there.
    Screen,
    /// On a panel standing in the room (Vision Pro's immersive space): the
    /// whole surface is the UI's. Looked at and pinched.
    Panel,
    /// Alone, as an image the shell shows beside the volume (visionOS).
    Image,
    /// A bar of tools, alone, as a second image the shell shows under the
    /// volume (visionOS), on glass, as a window's toolbar is.
    Toolbar,
}

/// The surface the UI is laid out on, in UI pixels (points: a phone is
/// about 400 across).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Surface {
    pub place: Place,
    pub width: f32,
    pub height: f32,
    /// Device pixels a UI pixel.
    pub scale: f32,
    /// What the system covers at the edges, in UI pixels: the status bar
    /// and the home indicator, a notch (`[top, right, bottom, left]`).
    pub insets: [f32; 4],
    /// The app is over the camera (AR) or in the room: what it shows is
    /// the world, not a background of its own.
    pub see_through: bool,
}

impl Surface {
    /// Whether this is a narrow screen (a phone held upright): the web's
    /// 820-pixel breakpoint.
    pub fn narrow(&self) -> bool {
        self.width < 820.0
    }
}

/// A panel's size and place in the room.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanelSpec {
    /// The UI's size, in UI pixels.
    pub width: f32,
    pub height: f32,
    /// Metres a UI pixel.
    pub metres_per_pixel: f32,
    /// How far ahead of the head it stands, and how far below the eyes.
    pub distance: f32,
    pub drop: f32,
    /// How far to the right of straight ahead (negative: the left).
    pub side: f32,
}

impl Default for PanelSpec {
    fn default() -> Self {
        PanelSpec {
            width: 640.0,
            height: 400.0,
            metres_per_pixel: 0.0008,
            distance: 0.6,
            drop: 0.3,
            side: 0.0,
        }
    }
}

/// An app: see the module docs. Only [`setup`](App::setup) is required.
pub trait App: Sized + 'static {
    /// Build the scene: lights, meshes, and the app's own state.
    fn setup(scene: &mut Scene) -> Self;

    /// Once a frame, before it is drawn: the head (AR, the room) and the
    /// pointers (a finger, a pinch) are in `frame`. Touches the UI took are
    /// not.
    fn update(&mut self, _scene: &mut Scene, _frame: &Frame) {}

    /// The lens: its field of view is the window's, its near and far planes
    /// every mode's.
    fn lens(&self) -> PerspectiveCamera {
        PerspectiveCamera::new(60.0, 1.0, 0.02, 100.0)
    }

    /// The reference space (where the head and pointers are given) in the
    /// scene's world.
    fn origin(&self) -> Matrix4 {
        Matrix4::identity()
    }

    /// A drag in the volume, turning the model `radians` about the
    /// vertical: `true` when the app turned it itself (about its own
    /// upright, before any tip toward the viewer, as an orbit about it
    /// would), else the volume turns the whole scene about the room's
    /// vertical — which shows a tipped model rolled.
    fn turn(&mut self, _radians: f32) -> bool {
        false
    }

    /// Whether [`focus`](Self::focus) is a point the app chose to frame (a
    /// scene's, a double tap's) rather than its model's resting place: the
    /// volume's quad then looks there, not at the middle of what is shown.
    fn framing(&self) -> bool {
        false
    }

    /// How far in the user has zoomed (1: not at all): when it frames no
    /// point of its own, the volume's quad fits what is shown, this much
    /// closer.
    fn zoom(&self) -> f32 {
        1.0
    }

    /// What the window's camera frames, and how big it is, once known:
    /// `(centre, radius)` in the scene's world. A volume centres it too.
    fn focus(&self) -> Option<(Vector3, f32)> {
        None
    }

    /// The UI for `surface`, ready to lay out and paint: change it in
    /// place (threers-ios repaints only what changed), rebuild it when the
    /// place or size calls for another layout. `None`: no UI.
    fn ui(&mut self, _surface: &Surface) -> Option<&mut Ui> {
        None
    }

    /// A pointer on the UI shown at `place`, in its pixels: fingers as
    /// [`Event::Touch`] (so lists scroll and fling), a pinch on the panel
    /// too, a tap on the volume's image as a press and release. Whether
    /// the UI took it — it landed on something of the UI, not on the empty
    /// root — so a touch that starts on the UI does not turn the camera.
    fn ui_event(&mut self, _place: Place, _event: Event, _time: f64) -> bool {
        false
    }

    /// What the platform's keyboard holds for the text control the UI at
    /// `place` has focused (a search box): give it to the control, as
    /// threers-ui's `Input::set_focused_text`. Whether one had focus.
    fn ui_text(&mut self, _place: Place, _text: String) -> bool {
        false
    }

    /// The panel in the room.
    fn panel(&self) -> PanelSpec {
        PanelSpec::default()
    }

    /// How the renderer maps light to the screen, and the exposure: none by
    /// default (three.js's `NoToneMapping`). Read every frame.
    fn tone_mapping(&self) -> (ToneMapping, f32) {
        (ToneMapping::None, 1.0)
    }
}
