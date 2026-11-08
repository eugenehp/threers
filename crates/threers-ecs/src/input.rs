//! What the player is doing, as named actions.

use std::collections::{BTreeMap, HashMap, HashSet};
use threers::math::Vector2;

/// Sorted, for a message.
fn names<'a>(keys: impl Iterator<Item = &'a String>) -> String {
    let mut list: Vec<&str> = keys.map(String::as_str).collect();
    list.sort_unstable();
    if list.is_empty() {
        "none yet".to_owned()
    } else {
        list.join(", ")
    }
}

macro_rules! keys {
    ($($name:ident),+ $(,)?) => {
        /// A physical key, by position, named as in the W3C `KeyboardEvent.code`
        /// list. `KeyW` is the key above `KeyS` on any layout, which is what
        /// movement wants.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum Key {
            $(#[doc = concat!("`", stringify!($name), "`")] $name,)+
        }

        impl Key {
            /// From a `KeyboardEvent.code` string. winit's `KeyCode` variants have
            /// the same names, so its `Debug` output works too.
            pub fn from_code(code: &str) -> Option<Key> {
                match code {
                    $(stringify!($name) => Some(Key::$name),)+
                    _ => None,
                }
            }
        }
    };
}

keys!(
    KeyA,
    KeyB,
    KeyC,
    KeyD,
    KeyE,
    KeyF,
    KeyG,
    KeyH,
    KeyI,
    KeyJ,
    KeyK,
    KeyL,
    KeyM,
    KeyN,
    KeyO,
    KeyP,
    KeyQ,
    KeyR,
    KeyS,
    KeyT,
    KeyU,
    KeyV,
    KeyW,
    KeyX,
    KeyY,
    KeyZ,
    Digit0,
    Digit1,
    Digit2,
    Digit3,
    Digit4,
    Digit5,
    Digit6,
    Digit7,
    Digit8,
    Digit9,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    Space,
    Enter,
    Escape,
    Tab,
    Backspace,
    Delete,
    ShiftLeft,
    ShiftRight,
    ControlLeft,
    ControlRight,
    AltLeft,
    AltRight,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    Backquote,
    Minus,
    Equal,
    BracketLeft,
    BracketRight,
    Backslash,
    Semicolon,
    Quote,
    Comma,
    Period,
    Slash,
);

/// Mouse buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    /// The primary button.
    Left,
    /// The secondary button.
    Right,
    /// The wheel pressed.
    Middle,
}

/// Gamepad buttons by position, so `South` is A on one pad and Cross on another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadButton {
    /// The bottom face button: A, or Cross.
    South,
    /// The right face button: B, or Circle.
    East,
    /// The left face button: X, or Square.
    West,
    /// The top face button: Y, or Triangle.
    North,
    /// The left bumper.
    LeftShoulder,
    /// The right bumper.
    RightShoulder,
    /// The trigger pulled past its click point. Its travel is
    /// [`PadAxis::LeftTrigger`].
    LeftTrigger,
    /// As `LeftTrigger`, on the right.
    RightTrigger,
    /// The stick pressed in.
    LeftThumb,
    /// As `LeftThumb`, on the right.
    RightThumb,
    /// Start, Menu or Options.
    Start,
    /// Select, View or Share.
    Select,
    /// The pad's up direction.
    DPadUp,
    /// The pad's down direction.
    DPadDown,
    /// The pad's left direction.
    DPadLeft,
    /// The pad's right direction.
    DPadRight,
}

impl PadButton {
    /// Every button, for binding menus and polling.
    pub const ALL: [PadButton; 16] = [
        PadButton::South,
        PadButton::East,
        PadButton::West,
        PadButton::North,
        PadButton::LeftShoulder,
        PadButton::RightShoulder,
        PadButton::LeftTrigger,
        PadButton::RightTrigger,
        PadButton::LeftThumb,
        PadButton::RightThumb,
        PadButton::Start,
        PadButton::Select,
        PadButton::DPadUp,
        PadButton::DPadDown,
        PadButton::DPadLeft,
        PadButton::DPadRight,
    ];
}

/// Sticks run from -1 to 1 with up and right positive, so `LeftY` and the W key
/// agree. Triggers run from 0 to 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadAxis {
    /// Left stick, right positive.
    LeftX,
    /// Left stick, up positive.
    LeftY,
    /// Right stick, right positive.
    RightX,
    /// Right stick, up positive.
    RightY,
    /// Left trigger, 0 to 1.
    LeftTrigger,
    /// Right trigger, 0 to 1.
    RightTrigger,
}

impl PadAxis {
    /// Every axis.
    pub const ALL: [PadAxis; 6] = [
        PadAxis::LeftX,
        PadAxis::LeftY,
        PadAxis::RightX,
        PadAxis::RightY,
        PadAxis::LeftTrigger,
        PadAxis::RightTrigger,
    ];
}

/// Something one gamepad did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PadEvent {
    /// A button went down.
    Pressed(PadButton),
    /// A button came up.
    Released(PadButton),
    /// An axis moved to this value.
    Axis(PadAxis, f32),
    /// The pad went away; what it held is released.
    Disconnected,
}

#[derive(Debug, Clone, Default)]
struct Pad {
    down: HashSet<PadButton>,
    axes: HashMap<PadAxis, f32>,
}

/// Turns events from any number of pads into one [`Input`].
///
/// It keeps each pad's state and works out the combined result, rather than
/// forwarding events: one pad letting go must not release a button another still
/// holds, and an unplugged pad sends no release at all. This holds no device, so
/// any source of pad events can drive it, a browser's or a test's.
#[derive(Debug, Clone, Default)]
pub struct PadTracker {
    pads: BTreeMap<usize, Pad>,
}

impl PadTracker {
    /// Record an event from `pad`, any number that identifies it.
    pub fn update(&mut self, pad: usize, event: PadEvent) {
        if event == PadEvent::Disconnected {
            self.pads.remove(&pad);
            return;
        }
        let state = self.pads.entry(pad).or_default();
        match event {
            PadEvent::Pressed(button) => {
                state.down.insert(button);
            }
            PadEvent::Released(button) => {
                state.down.remove(&button);
            }
            PadEvent::Axis(axis, value) => {
                state.axes.insert(axis, value);
            }
            PadEvent::Disconnected => {}
        }
    }

    /// Make `input` match. A button is down if any pad holds it, and an axis
    /// reads whichever pad is pushing it furthest. Call this after each event, or
    /// a press and release between two calls would never be seen.
    pub fn sync(&self, input: &mut Input) {
        for button in PadButton::ALL {
            if self.pads.values().any(|pad| pad.down.contains(&button)) {
                input.press(button);
            } else {
                input.release(button);
            }
        }
        for axis in PadAxis::ALL {
            let value = self
                .pads
                .values()
                .filter_map(|pad| pad.axes.get(&axis).copied())
                .fold(
                    0.0_f32,
                    |best, v| if v.abs() > best.abs() { v } else { best },
                );
            input.set_analog(axis, value);
        }
    }
}

/// Anything that is either down or up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Button {
    /// A key.
    Key(Key),
    /// A mouse button.
    Mouse(MouseButton),
    /// A gamepad button, on any pad.
    Pad(PadButton),
    /// An on-screen button, by its place in [`TouchControls`](crate::touch::TouchControls).
    Touch(u8),
}

impl From<Key> for Button {
    fn from(key: Key) -> Self {
        Button::Key(key)
    }
}

impl From<MouseButton> for Button {
    fn from(button: MouseButton) -> Self {
        Button::Mouse(button)
    }
}

impl From<PadButton> for Button {
    fn from(button: PadButton) -> Self {
        Button::Pad(button)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum AxisSource {
    Buttons {
        negative: Button,
        positive: Button,
    },
    Analog(PadAxis),
    /// One axis of the on-screen stick: 0 for x, 1 for y. Its own source, since a
    /// gamepad host rewrites the `PadAxis` values every frame.
    TouchStick(usize),
}

/// One finger on the screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Touch {
    /// The same for the life of the touch.
    pub id: u64,
    /// In window pixels.
    pub position: Vector2,
    /// Where it came down.
    pub start: Vector2,
}

/// A request to shake the gamepads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rumble {
    /// From 0 to 1.
    pub strength: f32,
    /// How long.
    pub seconds: f32,
}

#[derive(Debug, Clone, Default)]
struct Edges {
    pressed: HashSet<Button>,
    released: HashSet<Button>,
}

impl Edges {
    fn clear(&mut self) {
        self.pressed.clear();
        self.released.clear();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
enum Stage {
    #[default]
    Variable,
    Fixed,
}

/// What the player is doing, as a resource.
///
/// Game code asks about `"jump"`, not the space bar, so rebinding a key or adding
/// a gamepad changes one line of setup. The host feeds this from whatever it has,
/// and a test can feed it too, which is how a game gets played without a window.
///
/// # Presses and fixed steps
///
/// A frame runs any number of fixed steps, so "pressed this frame" means nothing
/// to a fixed system: a tap on a frame with no step would be lost, and one on a
/// frame with two would jump twice. So each kind of system gets its own edges.
/// Variable systems see a press on the frame it arrives. Fixed systems see it on
/// the next step that runs, and only on that step.
#[derive(Debug, Clone)]
pub struct Input {
    /// Stick travel to ignore, from 0 to 1, since a pad at rest rarely reads zero.
    pub deadzone: f32,
    /// Pointer position in window pixels.
    pub mouse_position: Vector2,
    /// Pointer movement since the last frame, in pixels.
    pub mouse_delta: Vector2,
    /// Wheel movement since the last frame.
    pub scroll: f32,
    /// Fingers on the screen now.
    pub touches: Vec<Touch>,
    /// Ask the host to capture the pointer, so `mouse_delta` keeps going past the
    /// edge of the window, as a first-person camera wants.
    pub lock_pointer: bool,
    /// Whether the host has it captured. In a browser that takes a click first.
    pub pointer_locked: bool,
    /// Something is taking the keys for itself, such as a console being typed
    /// into: while set, actions and axes read as idle, so the game does not
    /// drive while you type. Raw keys (`is_down`, `was_pressed`) still report.
    pub captured: bool,
    /// Text typed this frame, as the keyboard layout produced it.
    typed: String,
    touch_stick: [f32; 2],
    rumble: Option<Rumble>,
    down: HashSet<Button>,
    analog: HashMap<PadAxis, f32>,
    variable: Edges,
    fixed: Edges,
    stage: Stage,
    actions: HashMap<String, Vec<Button>>,
    /// Events since the last frame, while [`App`](crate::App) records.
    pub(crate) log: Option<Vec<crate::replay::InputEvent>>,
    axes: HashMap<String, Vec<AxisSource>>,
}

impl Default for Input {
    fn default() -> Self {
        Self {
            deadzone: 0.15,
            mouse_position: Vector2::new(0.0, 0.0),
            mouse_delta: Vector2::new(0.0, 0.0),
            scroll: 0.0,
            touches: Vec::new(),
            lock_pointer: false,
            pointer_locked: false,
            touch_stick: [0.0; 2],
            rumble: None,
            down: HashSet::new(),
            analog: HashMap::new(),
            variable: Edges::default(),
            fixed: Edges::default(),
            stage: Stage::Variable,
            actions: HashMap::new(),
            log: None,
            captured: false,
            typed: String::new(),
            axes: HashMap::new(),
        }
    }
}

impl Input {
    // ---- bindings ---------------------------------------------------------

    /// Make `button` trigger `action`. An action can have any number of buttons.
    pub fn bind(&mut self, action: &str, button: impl Into<Button>) -> &mut Self {
        self.actions
            .entry(action.to_owned())
            .or_default()
            .push(button.into());
        self
    }

    /// Make two buttons drive `axis`, to -1 and +1.
    pub fn bind_axis(
        &mut self,
        axis: &str,
        negative: impl Into<Button>,
        positive: impl Into<Button>,
    ) -> &mut Self {
        let source = AxisSource::Buttons {
            negative: negative.into(),
            positive: positive.into(),
        };
        self.axes.entry(axis.to_owned()).or_default().push(source);
        self
    }

    /// Make a stick or trigger drive `axis`.
    pub fn bind_analog(&mut self, axis: &str, analog: PadAxis) -> &mut Self {
        self.axes
            .entry(axis.to_owned())
            .or_default()
            .push(AxisSource::Analog(analog));
        self
    }

    /// Make the on-screen stick drive `axis`: 0 for its x, 1 for its y.
    pub fn bind_touch_stick(&mut self, axis: &str, stick_axis: usize) -> &mut Self {
        self.axes
            .entry(axis.to_owned())
            .or_default()
            .push(AxisSource::TouchStick(stick_axis));
        self
    }

    // ---- reading actions --------------------------------------------------

    /// Held right now.
    ///
    /// # Panics
    ///
    /// This and the other readers panic on a name that was never bound. A typo
    /// would otherwise be a control that silently does nothing.
    #[track_caller]
    pub fn pressed(&self, action: &str) -> bool {
        if self.captured {
            // Still checked, so a misspelt name says so even while typing.
            let _ = self.buttons(action);
            return false;
        }
        self.buttons(action).iter().any(|b| self.down.contains(b))
    }

    /// Went down since this kind of system last looked. See
    /// [presses and fixed steps](Self#presses-and-fixed-steps).
    #[track_caller]
    pub fn just_pressed(&self, action: &str) -> bool {
        if self.captured {
            // Still checked, so a misspelt name says so even while typing.
            let _ = self.buttons(action);
            return false;
        }
        let pressed = &self.edges().pressed;
        self.buttons(action).iter().any(|b| pressed.contains(b))
    }

    /// Came up since this kind of system last looked.
    #[track_caller]
    pub fn just_released(&self, action: &str) -> bool {
        if self.captured {
            // Still checked, so a misspelt name says so even while typing.
            let _ = self.buttons(action);
            return false;
        }
        let released = &self.edges().released;
        self.buttons(action).iter().any(|b| released.contains(b))
    }

    /// From -1 to 1. Of several sources, the one pushed furthest wins, so a
    /// resting stick cannot cancel a held key.
    #[track_caller]
    pub fn axis(&self, axis: &str) -> f32 {
        let Some(sources) = self.axes.get(axis) else {
            panic!("no axis named {axis:?} is bound: bind it with bind_axis or bind_analog. Bound axes: {}", names(self.axes.keys()))
        };
        if self.captured {
            return 0.0;
        }
        let mut value = 0.0_f32;
        for source in sources {
            let v = self.read(source);
            if v.abs() > value.abs() {
                value = v;
            }
        }
        value.clamp(-1.0, 1.0)
    }

    /// Two axes as a direction no longer than 1. Separately they would make a
    /// diagonal about 41% faster than straight ahead.
    #[track_caller]
    pub fn axis2(&self, x: &str, y: &str) -> Vector2 {
        let (x, y) = (self.axis(x), self.axis(y));
        let length = (x * x + y * y).sqrt();
        if length > 1.0 {
            Vector2::new(x / length, y / length)
        } else {
            Vector2::new(x, y)
        }
    }

    fn read(&self, source: &AxisSource) -> f32 {
        match *source {
            AxisSource::Buttons { negative, positive } => {
                let held = |b| self.down.contains(&b) as i32 as f32;
                held(positive) - held(negative)
            }
            AxisSource::Analog(axis) => {
                let v = self.analog.get(&axis).copied().unwrap_or(0.0);
                let travel = (v.abs() - self.deadzone) / (1.0 - self.deadzone);
                travel.clamp(0.0, 1.0) * v.signum()
            }
            AxisSource::TouchStick(axis) => self.touch_stick[axis.min(1)],
        }
    }

    #[track_caller]
    fn buttons(&self, action: &str) -> &[Button] {
        match self.actions.get(action) {
            Some(buttons) => buttons,
            None => panic!(
                "no action named {action:?} is bound: bind it with bind. Bound actions: {}",
                names(self.actions.keys())
            ),
        }
    }

    fn edges(&self) -> &Edges {
        match self.stage {
            Stage::Variable => &self.variable,
            Stage::Fixed => &self.fixed,
        }
    }

    // ---- reading raw devices ----------------------------------------------

    /// Whether it is held now. Raw, so it reports even while [`captured`](Self::captured).
    pub fn is_down(&self, button: impl Into<Button>) -> bool {
        self.down.contains(&button.into())
    }

    /// Like [`just_pressed`](Self::just_pressed), for one button.
    pub fn was_pressed(&self, button: impl Into<Button>) -> bool {
        self.edges().pressed.contains(&button.into())
    }

    /// Like [`just_released`](Self::just_released), for one button.
    pub fn was_released(&self, button: impl Into<Button>) -> bool {
        self.edges().released.contains(&button.into())
    }

    /// Take a press for yourself: nothing that runs after this, in this frame or
    /// a fixed step, sees it as just pressed. For an overlay that closes on a
    /// key the game also uses, as the console does with Escape. The key still
    /// reads as held.
    pub fn consume(&mut self, button: impl Into<Button>) {
        let button = button.into();
        self.variable.pressed.remove(&button);
        self.fixed.pressed.remove(&button);
    }

    // ---- what the host feeds in -------------------------------------------

    /// A button went down. Held-key repeats are ignored, so a host can pass them
    /// straight through.
    pub fn press(&mut self, button: impl Into<Button>) {
        let button: Button = button.into();
        self.note(|| crate::replay::InputEvent::Press(button));
        if self.down.insert(button) {
            self.variable.pressed.insert(button);
            self.fixed.pressed.insert(button);
        }
    }

    /// A button came up.
    pub fn release(&mut self, button: impl Into<Button>) {
        let button: Button = button.into();
        self.note(|| crate::replay::InputEvent::Release(button));
        if self.down.remove(&button) {
            self.variable.released.insert(button);
            self.fixed.released.insert(button);
        }
    }

    /// Everything came up. Hosts call this when the window loses focus: the
    /// release would go to another app, and the key would stay held here for good.
    pub fn release_all(&mut self) {
        self.note(|| crate::replay::InputEvent::ReleaseAll);
        for button in std::mem::take(&mut self.down) {
            self.variable.released.insert(button);
            self.fixed.released.insert(button);
        }
        self.analog.clear();
        self.touches.clear();
        self.touch_stick = [0.0; 2];
    }

    /// A stick or trigger moved, to a value from -1 to 1.
    pub fn set_analog(&mut self, axis: PadAxis, value: f32) {
        self.note(|| crate::replay::InputEvent::Analog(axis, value));
        self.analog.insert(axis, value.clamp(-1.0, 1.0));
    }

    /// Set the on-screen stick, each axis from -1 to 1.
    pub fn set_touch_stick(&mut self, x: f32, y: f32) {
        self.note(|| crate::replay::InputEvent::TouchStick(x, y));
        self.touch_stick = [x.clamp(-1.0, 1.0), y.clamp(-1.0, 1.0)];
    }

    /// A finger came down.
    pub fn touch_start(&mut self, id: u64, position: Vector2) {
        self.note(|| crate::replay::InputEvent::TouchStart(id, position));
        self.touches.retain(|touch| touch.id != id);
        self.touches.push(Touch {
            id,
            position,
            start: position,
        });
    }

    /// A finger moved.
    pub fn touch_move(&mut self, id: u64, position: Vector2) {
        self.note(|| crate::replay::InputEvent::TouchMove(id, position));
        if let Some(touch) = self.touches.iter_mut().find(|touch| touch.id == id) {
            touch.position = position;
        }
    }

    /// A finger came up, or the system took the touch away.
    pub fn touch_end(&mut self, id: u64) {
        self.note(|| crate::replay::InputEvent::TouchEnd(id));
        self.touches.retain(|touch| touch.id != id);
    }

    /// Pointer movement with no position, as a locked pointer reports.
    pub fn add_mouse_motion(&mut self, delta: Vector2) {
        self.note(|| crate::replay::InputEvent::MouseMotion(delta));
        self.mouse_delta = self.mouse_delta + delta;
    }

    /// Shake the gamepads, for any that can. A stronger request replaces a weaker
    /// one made in the same frame.
    pub fn rumble(&mut self, strength: f32, seconds: f32) {
        let strength = strength.clamp(0.0, 1.0);
        if self.rumble.is_none_or(|old| strength >= old.strength) {
            self.rumble = Some(Rumble { strength, seconds });
        }
    }

    /// The rumble waiting to be played. Hosts call this.
    pub fn take_rumble(&mut self) -> Option<Rumble> {
        self.rumble.take()
    }

    /// Text the keyboard produced, as typed: hosts call this with what the
    /// platform reports, so it follows the player's layout.
    pub fn type_text(&mut self, text: &str) {
        self.note(|| crate::replay::InputEvent::Text(text.to_owned()));
        self.typed.push_str(text);
    }

    /// What was typed this frame.
    pub fn text(&self) -> &str {
        &self.typed
    }

    /// The pointer moved to `position`, in window pixels.
    pub fn move_mouse(&mut self, position: Vector2) {
        self.note(|| crate::replay::InputEvent::MouseMove(position));
        self.mouse_delta = self.mouse_delta + (position - self.mouse_position);
        self.mouse_position = position;
    }

    fn note(&mut self, event: impl FnOnce() -> crate::replay::InputEvent) {
        if let Some(log) = &mut self.log {
            log.push(event());
        }
    }

    // ---- what `App` calls -------------------------------------------------

    pub(crate) fn enter_fixed(&mut self) {
        self.stage = Stage::Fixed;
    }

    pub(crate) fn consume_fixed(&mut self) {
        self.fixed.clear();
    }

    pub(crate) fn leave_fixed(&mut self) {
        self.stage = Stage::Variable;
    }

    pub(crate) fn end_frame(&mut self) {
        self.variable.clear();
        self.typed.clear();
        self.mouse_delta = Vector2::new(0.0, 0.0);
        self.scroll = 0.0;
    }
}
