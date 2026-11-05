//! Recording what the player did, and playing it back.
//!
//! A [`Recording`] holds each frame's length and the input that arrived before
//! it. Played back into a fresh copy of the same game, it gives the same game:
//! the same fixed steps, the same presses on the same steps. That turns a bug
//! someone found by playing into a test that finds it every time.
//!
//! ```
//! # use threers_ecs::prelude::*;
//! fn game(app: &mut App) {
//!     app.world.resource_mut::<Input>().bind("jump", Key::Space);
//! }
//! let mut app = App::new();
//! game(&mut app);
//! app.start_recording();
//! app.world.resource_mut::<Input>().press(Key::Space);
//! app.frame(1.0 / 60.0);
//! let take = app.stop_recording().unwrap();
//!
//! let text = take.to_text();                    // save it beside the tests
//! let again = Recording::from_text(&text).unwrap();
//! let mut copy = App::new();
//! game(&mut copy);
//! again.play(&mut copy);
//! ```
//!
//! With [`DevTools`](crate::dev::DevTools), F12 starts and stops a recording and
//! writes it to `recordings/`.
//!
//! [`Rewind`] uses the same exactness to go back in time: F1 rebuilds the game
//! and plays it forward to a few seconds ago.

use crate::app::App;
use crate::input::{Button, Input, Key, MouseButton, PadAxis, PadButton};
use crate::world::World;
use threers::math::Vector2;

/// One thing the host told [`Input`].
#[derive(Debug, Clone, PartialEq)]
pub enum InputEvent {
    Press(Button),
    Release(Button),
    ReleaseAll,
    Analog(PadAxis, f32),
    TouchStick(f32, f32),
    TouchStart(u64, Vector2),
    TouchMove(u64, Vector2),
    TouchEnd(u64),
    MouseMove(Vector2),
    MouseMotion(Vector2),
    Scroll(f32),
    Text(String),
}

/// Frames of input, each with its length in seconds.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Recording {
    pub frames: Vec<(f64, Vec<InputEvent>)>,
}

/// The recording going on, as a resource.
#[derive(Default)]
pub struct Recorder(pub(crate) Option<Recording>);

impl Recorder {
    /// Frames recorded so far.
    pub fn frames(&self) -> usize {
        self.0.as_ref().map_or(0, |r| r.frames.len())
    }
}

impl World {
    /// Start keeping every input event and frame length.
    pub fn start_recording(&mut self) {
        self.resource_mut::<Input>().log = Some(Vec::new());
        self.resource_mut::<Recorder>().0 = Some(Recording::default());
    }

    /// Whether a recording is going.
    pub fn is_recording(&self) -> bool {
        self.resource::<Recorder>().0.is_some()
    }

    /// Stop, and hand over what was recorded.
    pub fn stop_recording(&mut self) -> Option<Recording> {
        self.resource_mut::<Input>().log = None;
        self.resource_mut::<Recorder>().0.take()
    }
}

impl App {
    /// [`World::start_recording`].
    pub fn start_recording(&mut self) {
        self.world.start_recording();
    }

    /// [`World::stop_recording`].
    pub fn stop_recording(&mut self) -> Option<Recording> {
        self.world.stop_recording()
    }

    /// Called at the start of each frame.
    pub(crate) fn record_frame(&mut self, delta: f64) {
        if !self.world.is_recording() {
            return;
        }
        let input = self.world.resource_mut::<Input>();
        let mut events = input.log.as_mut().map(std::mem::take).unwrap_or_default();
        if input.scroll != 0.0 {
            events.push(InputEvent::Scroll(input.scroll));
        }
        if let Some(recording) = &mut self.world.resource_mut::<Recorder>().0 {
            recording.frames.push((delta, events));
        }
    }
}

impl Recording {
    /// Feed it to `app` frame by frame, as the host did.
    pub fn play(&self, app: &mut App) {
        for frame in 0..self.frames.len() {
            self.play_frame(app, frame);
        }
    }

    /// One frame: its input, then [`App::frame`].
    pub fn play_frame(&self, app: &mut App, frame: usize) {
        let Some((delta, events)) = self.frames.get(frame) else {
            return;
        };
        let input = app.world.resource_mut::<Input>();
        for event in events {
            apply(input, event);
        }
        app.frame(*delta);
    }

    /// Total length in seconds.
    pub fn seconds(&self) -> f64 {
        self.frames.iter().map(|(delta, _)| delta).sum()
    }

    /// As text: a frame to a line, its length and then its events.
    pub fn to_text(&self) -> String {
        let mut out = String::from("threers-recording 1\n");
        for (delta, events) in &self.frames {
            out.push_str(&format!("{delta:?}"));
            for event in events {
                out.push(' ');
                out.push_str(&event_text(event));
            }
            out.push('\n');
        }
        out
    }

    /// # Errors
    ///
    /// Naming the line and what on it could not be read.
    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut lines = text.lines().enumerate();
        match lines.next() {
            Some((_, "threers-recording 1")) => {}
            _ => {
                return Err(
                    "not a recording: the first line should be `threers-recording 1`".into(),
                )
            }
        }
        let mut frames = Vec::new();
        for (number, line) in lines {
            if line.trim().is_empty() {
                continue;
            }
            let mut words = line.split_whitespace();
            let fail = |what: &str| format!("line {}: {what}", number + 1);
            let delta = words
                .next()
                .and_then(|w| w.parse::<f64>().ok())
                .ok_or_else(|| fail("no frame length"))?;
            let events = words
                .map(|word| parse_event(word).ok_or_else(|| fail(&format!("cannot read `{word}`"))))
                .collect::<Result<Vec<_>, _>>()?;
            frames.push((delta, events));
        }
        Ok(Self { frames })
    }
}

fn apply(input: &mut Input, event: &InputEvent) {
    match *event {
        InputEvent::Press(button) => input.press(button),
        InputEvent::Release(button) => input.release(button),
        InputEvent::ReleaseAll => input.release_all(),
        InputEvent::Analog(axis, value) => input.set_analog(axis, value),
        InputEvent::TouchStick(x, y) => input.set_touch_stick(x, y),
        InputEvent::TouchStart(id, at) => input.touch_start(id, at),
        InputEvent::TouchMove(id, at) => input.touch_move(id, at),
        InputEvent::TouchEnd(id) => input.touch_end(id),
        InputEvent::MouseMove(at) => input.move_mouse(at),
        InputEvent::MouseMotion(by) => input.add_mouse_motion(by),
        InputEvent::Scroll(by) => input.scroll += by,
        InputEvent::Text(ref text) => input.type_text(text),
    }
}

fn button_text(button: Button) -> String {
    match button {
        Button::Key(key) => format!("key:{key:?}"),
        Button::Mouse(mouse) => format!("mouse:{mouse:?}"),
        Button::Pad(pad) => format!("pad:{pad:?}"),
        Button::Touch(index) => format!("touch:{index}"),
    }
}

fn parse_button(text: &str) -> Option<Button> {
    let (kind, name) = text.split_once(':')?;
    match kind {
        "key" => Key::from_code(name).map(Button::Key),
        "mouse" => [MouseButton::Left, MouseButton::Right, MouseButton::Middle]
            .into_iter()
            .find(|m| format!("{m:?}") == name)
            .map(Button::Mouse),
        "pad" => PadButton::ALL
            .into_iter()
            .find(|p| format!("{p:?}") == name)
            .map(Button::Pad),
        "touch" => name.parse().ok().map(Button::Touch),
        _ => None,
    }
}

fn event_text(event: &InputEvent) -> String {
    match event {
        InputEvent::Press(b) => format!("+{}", button_text(*b)),
        InputEvent::Release(b) => format!("-{}", button_text(*b)),
        InputEvent::ReleaseAll => "release-all".into(),
        InputEvent::Analog(axis, v) => format!("axis:{axis:?}={v:?}"),
        InputEvent::TouchStick(x, y) => format!("stick={x:?},{y:?}"),
        InputEvent::TouchStart(id, at) => format!("touch-start:{id}={:?},{:?}", at.x, at.y),
        InputEvent::TouchMove(id, at) => format!("touch-move:{id}={:?},{:?}", at.x, at.y),
        InputEvent::TouchEnd(id) => format!("touch-end={id}"),
        InputEvent::MouseMove(at) => format!("mouse-at={:?},{:?}", at.x, at.y),
        InputEvent::MouseMotion(by) => format!("mouse-by={:?},{:?}", by.x, by.y),
        InputEvent::Scroll(by) => format!("scroll={by:?}"),
        InputEvent::Text(text) => format!("text={}", escape(text)),
    }
}

/// Text with spaces and anything unusual as `%XX` bytes, so it is one word.
fn escape(text: &str) -> String {
    text.bytes()
        .map(|b| {
            if b.is_ascii_graphic() && b != b'%' {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}

fn unescape(word: &str) -> Option<String> {
    let mut bytes = Vec::new();
    let mut rest = word.as_bytes();
    while let Some((&b, tail)) = rest.split_first() {
        if b == b'%' {
            let hex = std::str::from_utf8(tail.get(..2)?).ok()?;
            bytes.push(u8::from_str_radix(hex, 16).ok()?);
            rest = &tail[2..];
        } else {
            bytes.push(b);
            rest = tail;
        }
    }
    String::from_utf8(bytes).ok()
}

fn pair(text: &str) -> Option<(f32, f32)> {
    let (a, b) = text.split_once(',')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

fn parse_event(word: &str) -> Option<InputEvent> {
    if let Some(rest) = word.strip_prefix('+') {
        return parse_button(rest).map(InputEvent::Press);
    }
    if let Some(rest) = word.strip_prefix('-') {
        return parse_button(rest).map(InputEvent::Release);
    }
    if word == "release-all" {
        return Some(InputEvent::ReleaseAll);
    }
    let (head, value) = word.split_once('=')?;
    let (kind, arg) = head.split_once(':').unwrap_or((head, ""));
    Some(match kind {
        "axis" => {
            let axis = PadAxis::ALL.into_iter().find(|a| format!("{a:?}") == arg)?;
            InputEvent::Analog(axis, value.parse().ok()?)
        }
        "stick" => {
            let (x, y) = pair(value)?;
            InputEvent::TouchStick(x, y)
        }
        "touch-start" => {
            let (x, y) = pair(value)?;
            InputEvent::TouchStart(arg.parse().ok()?, Vector2::new(x, y))
        }
        "touch-move" => {
            let (x, y) = pair(value)?;
            InputEvent::TouchMove(arg.parse().ok()?, Vector2::new(x, y))
        }
        "mouse-at" => {
            let (x, y) = pair(value)?;
            InputEvent::MouseMove(Vector2::new(x, y))
        }
        "mouse-by" => {
            let (x, y) = pair(value)?;
            InputEvent::MouseMotion(Vector2::new(x, y))
        }
        "scroll" => InputEvent::Scroll(value.parse().ok()?),
        "text" => InputEvent::Text(unescape(value)?),
        "touch-end" => InputEvent::TouchEnd(value.parse().ok()?),
        _ => return None,
    })
}

/// Go back a few seconds with F1, to look again at what just happened.
///
/// It records from the moment it is added, which should be right after the
/// game is built, and rewinding builds the game again with `build` and plays the
/// recording into it up to the point asked for. That is exact for the same
/// reason [`Recording::play`] is, and its cost grows with how long the session
/// has run: a development tool. The race replays at about 24 times real time
/// in a release build (a minute of racing in 2.5 s of CPU), so a rewind ten
/// minutes in takes some 25 seconds. What the host added after `build`, such as the
/// window's camera, the HUD and the sound output, is kept; sounds playing are
/// stopped and start again from the rewound world.
///
/// ```no_run
/// # use threers_ecs::prelude::*;
/// # use threers_ecs::replay::Rewind;
/// fn build(app: &mut App) { /* the game */ }
/// let mut app = App::new();
/// build(&mut app);
/// app.plugin(Rewind::new(build).seconds(5.0));
/// ```
pub struct Rewind {
    build: fn(&mut App),
    seconds: f64,
}

/// Rewind's settings, and a request waiting for the end of the frame.
pub struct RewindState {
    build: fn(&mut App),
    pub seconds: f64,
    pub(crate) requested: bool,
    /// How many times it has rewound.
    pub rewinds: u32,
}

impl RewindState {
    /// Whether a rewind is waiting for the end of the frame.
    pub fn requested(&self) -> bool {
        self.requested
    }
}

impl Rewind {
    pub fn new(build: fn(&mut App)) -> Self {
        Self {
            build,
            seconds: 5.0,
        }
    }

    /// How far back F1 goes; 5 seconds by default.
    pub fn seconds(mut self, seconds: f64) -> Self {
        self.seconds = seconds.max(0.0);
        self
    }
}

impl crate::app::Plugin for Rewind {
    fn build(self, app: &mut App) {
        app.world.insert_resource(RewindState {
            build: self.build,
            seconds: self.seconds,
            requested: false,
            rewinds: 0,
        });
        app.start_recording();
        app.system(|w: &mut World| {
            if w.resource::<Input>().was_pressed(Key::F1) {
                w.resource_mut::<RewindState>().requested = true;
            }
        });
    }
}

/// Stop every sound: a rewound world starts its own again.
#[cfg(feature = "audio")]
fn silence(w: &mut World) {
    if let Some(out) = w.get_resource_mut::<crate::audio::AudioOut>() {
        out.stop_all();
    }
}

#[cfg(not(feature = "audio"))]
fn silence(_: &mut World) {}

impl World {
    /// Ask [`Rewind`] to go back, at the end of this frame.
    pub fn rewind(&mut self) {
        if let Some(state) = self.get_resource_mut::<RewindState>() {
            state.requested = true;
        }
    }
}

impl App {
    /// Called at the end of each frame: rewind if asked.
    pub(crate) fn rewind_if_asked(&mut self) {
        let Some(state) = self.world.get_resource_mut::<RewindState>() else {
            return;
        };
        if !std::mem::take(&mut state.requested) {
            return;
        }
        let (build, seconds) = (state.build, state.seconds);
        let Some(recording) = self
            .world
            .get_resource::<Recorder>()
            .and_then(|r| r.0.clone())
        else {
            return;
        };
        // Keep the frames up to `seconds` before the end.
        let total = recording.seconds();
        let target = (total - seconds).max(0.0);
        let mut kept = Recording::default();
        let mut elapsed = 0.0;
        for frame in &recording.frames {
            if elapsed + frame.0 > target + 1e-9 {
                break;
            }
            elapsed += frame.0;
            kept.frames.push(frame.clone());
        }

        let mut past = App::new();
        build(&mut past);
        kept.play(&mut past);

        silence(&mut self.world);
        let rewinds = self
            .world
            .get_resource::<RewindState>()
            .map_or(0, |s| s.rewinds)
            + 1;
        let mut old = std::mem::replace(&mut self.world, std::mem::take(&mut past.world));
        // The App's own clock goes back with the world, or the next frame
        // would put the old time back.
        self.take_clock_from(&past);
        old.remove_resource::<RewindState>();
        self.world.adopt_missing_resources(&mut old);
        // The host's view of the screen, not the game's.
        self.world.take_resource_from::<crate::hud::Hud>(&mut old);
        self.world.insert_resource(RewindState {
            build,
            seconds,
            requested: false,
            rewinds,
        });
        // Recording goes on from here, as what happened before it is now this.
        self.world.resource_mut::<Input>().log = Some(Vec::new());
        self.world.insert_resource(Recorder(Some(kept)));
    }
}
