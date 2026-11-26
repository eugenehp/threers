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
    /// On a [`Piece`] by itself, by its [`id`](Piece::id): a part of the UI
    /// standing in the volume or in the room, in no window (visionOS).
    Piece(u32),
}

/// Where a [`Piece`] stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PieceIn {
    /// In the volume, among the app's model: its place from the volume's
    /// middle. What reaches past the volume's walls is cut off.
    #[default]
    Volume,
    /// In the room round you, in no window, while
    /// [`Request::EnterSpace`](crate::Request::EnterSpace) is in force: its
    /// place from the floor where you were as it opened (up is `y`, ahead
    /// `-z`).
    Room,
}

/// What a [`Piece`] is made of. Radii are its corners', in UI pixels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Glass {
    /// Nothing: its UI alone (a [`PieceSet`](crate::PieceSet) puts its
    /// theme's surface behind it, to be read).
    #[default]
    None,
    /// No pane at all: each thing in it stands in the room by itself, its
    /// words with a soft shade round them to be read on whatever is behind,
    /// its controls solid where it is [`raised`](Piece::raised) (blocks, a
    /// slider's ball: RealityKit's). Its UI takes
    /// [`Tokens::glass`](threers::ui::environment::Tokens::glass).
    Bare,
    /// The system's: a flat sheet, what is behind it blurred.
    System { radius: f32 },
    /// The system's, thicker at its edge, as a plate.
    Plate { radius: f32 },
    /// The system's, fading out over its last `soft` UI pixels: no edge.
    Feathered { radius: f32, soft: f32 },
    /// Dense: next to nothing of what is behind it but its colour.
    Frosted { radius: f32 },
    /// Clear, nothing blurred: a slab `thickness` metres deep you look
    /// through, darkened enough to read on.
    Clear { radius: f32, thickness: f32 },
    /// Liquid, its rim splitting the light into colours.
    Prism { radius: f32, thickness: f32 },
    /// Liquid: the system's glass in front of a clear, rounded slab
    /// `thickness` metres deep, which the room's light shines on and
    /// a bright rim runs round; looked at, it lights where you look.
    Liquid {
        radius: f32,
        thickness: f32,
        /// Its colour, and how strong (linear RGB, and 0 to 1).
        tint: [f32; 4],
    },
}

impl Glass {
    /// Liquid glass, clear, 12 mm deep.
    pub fn liquid(radius: f32) -> Self {
        Glass::Liquid { radius, thickness: 0.012, tint: [1.0, 1.0, 1.0, 0.04] }
    }

    /// Liquid glass of colour `r, g, b` (linear); a pale one takes dark text
    /// ([`is_pale`](Self::is_pale)).
    pub fn liquid_tinted(radius: f32, r: f32, g: f32, b: f32) -> Self {
        Glass::Liquid { radius, thickness: 0.012, tint: [r, g, b, 0.35] }
    }

    /// Whether it is pale: liquid glass of a light colour (a luminance of
    /// 0.5 or more) and not a trace of it. It is lightened rather than
    /// darkened, for dark text.
    pub fn is_pale(&self) -> bool {
        let Glass::Liquid { tint, .. } = self else {
            return false;
        };
        let [r, g, b, a] = tint.map(|t| if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 });
        a >= 0.2 && r * 0.2126 + g * 0.7152 + b * 0.0722 >= 0.5
    }
}

/// A part of the app's UI — a button, a card, a panel — as a RealityKit
/// entity of its own (visionOS): see [`App::pieces`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Piece {
    /// Which it is, from one frame to the next: its UI is [`App::ui`]'s for
    /// [`Place::Piece`] of it.
    pub id: u32,
    pub at: PieceIn,
    /// The UI's size, in UI pixels.
    pub width: f32,
    pub height: f32,
    /// Metres a UI pixel.
    pub metres_per_pixel: f32,
    /// Where its middle is, in metres, and how it is turned (facing `z`,
    /// unturned).
    pub position: Vector3,
    pub rotation: threers::math::Quaternion,
    /// It turns to face you, wherever you are.
    pub faces_you: bool,
    /// What it is made of, behind its UI (which leaves its background clear
    /// to show it). On glass a UI in the default theme takes the colours
    /// for it ([`Tokens::glass`](threers::ui::environment::Tokens::glass); on pale
    /// glass [`Tokens::pale_glass`](threers::ui::environment::Tokens::pale_glass)).
    pub glass: Glass,
    /// How much more of what is behind shows through its glass: 0 (the
    /// default), as its kind does, to 0.9, hardly there. Over 0 the system's
    /// own glass, which cannot be thinned, gives way to a thinned blur, and
    /// a soft halo in the glass's shade stands behind the words; how well
    /// they read is then up to what is behind.
    pub see_through: f32,
    /// How far its controls stand out of it, in metres: solid, pushed in as
    /// they are pressed, a slider's thumb a ball. 0 (the default): flat.
    pub raised: f32,
    /// Pressed by looking: in the room, the control the head points at is
    /// pressed after this many seconds on it (0, the default: never), a
    /// ring filling meanwhile. The head's direction, not the eyes' (which
    /// visionOS keeps to itself; a look and a pinch, and the system's Dwell
    /// Control, work as ever). A piece moved by its bar is pointed at where
    /// it is, and one in the volume while the room is open (the shell says
    /// where each is shown, `threers_ios_piece_pose`; the head is tracked
    /// only then).
    pub dwell: f32,
    /// It can be taken by the bar under it: moved, turned and drawn nearer
    /// with the hands, from where the app puts it, and stays there.
    pub movable: bool,
}

/// How pieces in a [`row`](Piece::row) or a [`column`](Piece::column) line
/// up across it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Align {
    /// By their tops, or their left edges.
    Start,
    /// By their middles.
    #[default]
    Center,
    /// By their feet, or their right edges.
    End,
}

impl Piece {
    /// Its width and height in metres.
    pub fn metres(&self) -> (f32, f32) {
        (self.width * self.metres_per_pixel, self.height * self.metres_per_pixel)
    }

    /// Lays `pieces` side by side, left to right, `gap` metres apart, the
    /// row's middle at `middle`, facing as it does (`facing`); `align` up
    /// and down.
    pub fn row(pieces: &mut [Piece], middle: Vector3, facing: threers::math::Quaternion, gap: f32, align: Align) {
        Self::lay(pieces, middle, facing, gap, align, true);
    }

    /// [`row`](Self::row), top to bottom; `align` sideways.
    pub fn column(pieces: &mut [Piece], middle: Vector3, facing: threers::math::Quaternion, gap: f32, align: Align) {
        Self::lay(pieces, middle, facing, gap, align, false);
    }

    fn lay(pieces: &mut [Piece], middle: Vector3, facing: threers::math::Quaternion, gap: f32, align: Align, across: bool) {
        let (along, over): (Vec<f32>, Vec<f32>) = pieces
            .iter()
            .map(|p| {
                let (w, h) = p.metres();
                if across { (w, h) } else { (h, w) }
            })
            .unzip();
        let most = over.iter().fold(0.0, |a: f32, b| a.max(*b));
        let total: f32 = along.iter().sum::<f32>() + gap * pieces.len().saturating_sub(1) as f32;
        let mut at = -total / 2.0;
        for ((p, along), over) in pieces.iter_mut().zip(along).zip(over) {
            let a = at + along / 2.0;
            at += along + gap;
            // from the middle toward the start
            let b = match align {
                Align::Start => (most - over) / 2.0,
                Align::Center => 0.0,
                Align::End => (over - most) / 2.0,
            };
            // (a column runs down; a row's start is up, a column's left)
            let off = if across { Vector3::new(a, b, 0.0) } else { Vector3::new(-b, -a, 0.0) };
            p.position = middle + off.apply_quaternion(facing);
            p.rotation = facing;
        }
    }

    /// Stands `pieces` round you, `radius` metres off at `height`, evenly
    /// over `sweep` radians from left to right, each facing where you are.
    pub fn around(pieces: &mut [Piece], height: f32, radius: f32, sweep: f32) {
        let n = pieces.len();
        for (i, p) in pieces.iter_mut().enumerate() {
            let t = if n > 1 { i as f32 / (n - 1) as f32 - 0.5 } else { 0.0 };
            let a = t * sweep;
            p.position = Vector3::new(a.sin() * radius, height, -a.cos() * radius);
            p.rotation = threers::math::Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), -a);
        }
    }

    /// `width × height` UI pixels at `position`, unturned, a UI pixel 0.8 mm.
    pub fn new(id: u32, at: PieceIn, width: f32, height: f32, position: Vector3) -> Self {
        Piece {
            id,
            at,
            width,
            height,
            metres_per_pixel: 0.0008,
            position,
            rotation: threers::math::Quaternion::IDENTITY,
            faces_you: false,
            glass: Glass::None,
            see_through: 0.0,
            raised: 0.0,
            movable: false,
            dwell: 0.0,
        }
    }
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

/// What is held with a hardware keyboard's key.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KeyModifiers {
    pub shift: bool,
    pub control: bool,
    /// Option.
    pub alt: bool,
    /// Command.
    pub meta: bool,
}

impl KeyModifiers {
    /// As the shell says them ([`crate::view::Key::Keyboard`]).
    pub(crate) fn from_shell(shift: f32, others: f32) -> Self {
        let bits = others as u32;
        Self { shift: shift > 0.5, control: bits & 1 != 0, alt: bits & 2 != 0, meta: bits & 4 != 0 }
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

    /// How much larger than its own pixels the app's UI is drawn in the
    /// window (`width × height` points): 1 a UI pixel to a point (the
    /// default); more, everything in it larger — text and controls alike,
    /// as a browser's zoom — for a UI made for a desktop's screen read on a
    /// tablet's at arm's length.
    fn ui_zoom(&self, _width: f32, _height: f32) -> f32 {
        1.0
    }

    /// The window's camera, each frame before [`update`](Self::update)
    /// (in a window; not in AR or a volume): where the view is and how
    /// wide, for an app that streams detail by what is on screen (finer
    /// tiles zoomed in) or places labels by it.
    fn window_camera(&mut self, _camera: &PerspectiveCamera) {}

    /// The window's shape (width over height) — of what its UI leaves
    /// uncovered ([`covered_left`](Self::covered_left)) — each frame before
    /// it is framed ([`focus`](Self::focus)): for an app that frames a
    /// portrait view closer than a landscape one.
    fn view_aspect(&mut self, _aspect: f32) {}

    /// How much of the window's left the app's own UI covers (a docked
    /// sidebar), in UI pixels: the view is centred, and framed, in the rest.
    fn covered_left(&self) -> f32 {
        0.0
    }

    /// Ambient occlusion on the volume's picture (threers' SSAO): its
    /// radius in the scene's world units and its strength (the web
    /// composite's soft `1 / (1 + strength · occlusion)`); `None`, none.
    fn ambient_occlusion(&self) -> Option<(f32, f32)> {
        None
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

    /// Two fingers pinching the window, their midpoint at `(x, y)` (the
    /// UI's points): `scale` how far they have spread since they came down.
    /// Whether it is the app's (asked as it begins; then told of it till it
    /// ends): else it zooms the camera.
    fn pinch(&mut self, _scale: f32, _x: f32, _y: f32, _phase: threers::ui::input::GesturePhase) -> bool {
        false
    }

    /// A hardware keyboard's key (`key` named as the web's
    /// `KeyboardEvent.key`: `Escape`, `ArrowUp`, `?`, `[`), `shift` held,
    /// while no text field has it. Whether the app used it.
    fn key(&mut self, _key: &str, _shift: bool) -> bool {
        false
    }

    /// [`App::key`] with every modifier held, for an app with shortcuts. One
    /// with Command or Control held is no plain key: not [`App::key`]'s.
    fn key_with(&mut self, key: &str, held: KeyModifiers) -> bool {
        !held.meta && !held.control && self.key(key, held.shift)
    }

    /// A key held down, again and again until it is let go (not a
    /// shortcut's, nor Escape): as [`App::key_with`], unless the app tells
    /// them apart.
    fn key_repeat(&mut self, key: &str, held: KeyModifiers) -> bool {
        self.key_with(key, held)
    }

    /// The app came to the front (`true`), or is leaving it: it may be
    /// ended with no other word, so what is to be kept is, now.
    fn active(&mut self, _on: bool) {}

    /// The window's view drawn for a picture ([`crate::snapshot`]):
    /// `width` × `height` RGBA, the scene alone (no UI over it).
    fn snapshot(&mut self, _width: u32, _height: u32, _rgba: Vec<u8>) {}

    /// How RealityKit lights the scene where RealityKit draws it (a volume,
    /// RealityKit on an iPhone's, an iPad's or a Mac's screen): by the
    /// room's light, the app's own lights, or both.
    fn reality_lighting(&self) -> crate::RealityLighting {
        crate::RealityLighting::Auto
    }

    /// What draws the scene in a visionOS volume: threers, as a picture
    /// facing you (the default), or RealityKit, as entities in depth.
    fn volume_drawing(&self) -> crate::VolumeDrawing {
        crate::VolumeDrawing::Picture
    }

    /// The parts of the UI standing by themselves as RealityKit entities,
    /// in the volume or in the room (visionOS; nothing elsewhere), read
    /// each frame: one left out goes, one moved moves. The UI of each is
    /// [`ui`](Self::ui)'s for [`Place::Piece`], and its events
    /// [`ui_event`](Self::ui_event)'s.
    fn pieces(&mut self) -> Vec<Piece> {
        Vec::new()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_keys_modifiers_are_as_the_shell_says_them() {
        assert_eq!(KeyModifiers::from_shell(0.0, 0.0), KeyModifiers::default());
        assert_eq!(KeyModifiers::from_shell(1.0, 4.0), KeyModifiers { shift: true, meta: true, ..Default::default() });
        assert_eq!(KeyModifiers::from_shell(0.0, 3.0), KeyModifiers { control: true, alt: true, ..Default::default() });
    }
}
