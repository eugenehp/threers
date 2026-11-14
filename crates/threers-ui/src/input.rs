//! Input: pointer and keyboard events delivered to the nodes they concern,
//! focus, and keys turned into actions.
//!
//! The model is the DOM's. A pointer event goes to the
//! node under the pointer (the *target*) and to its ancestors: first the
//! **capture** phase, from the root inward, then the **bubble** phase, from
//! the target outward. Any listener can stop it going further
//! ([`EventCx::stop_propagation`]) or cancel what would happen next by
//! default ([`EventCx::prevent_default`]). Key events go the same way to the
//! focused node. Hit-testing uses the layout last computed, so dispatch
//! between [`Ui::layout`] calls, as a frame does.
//!
//! What follows by default, unless a listener prevents it:
//!
//! - a press focuses the nearest focusable node under the pointer (one whose
//!   [`Semantics`](super::Semantics) are focusable), or takes focus away;
//! - a press and a release make a [`Event::Click`] on the nearest node that
//!   contains both — and Enter or Space on a focused button, checkbox or
//!   other control clicks it too, as a page does, so keyboard users can
//!   reach everything a pointer can;
//! - Tab and Shift+Tab move focus through [`Ui::focus_order`];
//! - the wheel scrolls the innermost scrolling box under the pointer that
//!   can still scroll that way, then the next one out.
//!
//! Keys are first offered to the [`Keymap`]: a binding that matches in the
//! focused node's key contexts becomes an [`Action`], handled by the
//! nearest [`Input::on_action`] handler from the focused node outward. Keys
//! no binding takes arrive as [`Event::Key`].
//!
//! Listeners get the app's own state as `&mut S` along with the tree, so
//! handlers change state directly rather than through shared cells:
//!
//! ```
//! use threers_ui::input::{Event, Input};
//! use threers_ui::{style, FixedWidthText, Role, Semantics, Ui};
//!
//! struct App { clicks: u32 }
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let button = ui.add_box(root, style! { width: 80px; height: 24px; });
//! ui.set_semantics(button, Semantics::new(Role::Button).label("Add"));
//! ui.layout(200.0, 100.0, &mut FixedWidthText(7.0));
//!
//! let mut input = Input::new();
//! input.on_click(button, |app: &mut App, _ui, _cx| app.clicks += 1);
//!
//! let mut app = App { clicks: 0 };
//! input.click(&mut ui, &mut app, 10.0, 10.0);
//! assert_eq!(app.clicks, 1);
//! assert_eq!(input.focused(), Some(button));
//! // The keyboard reaches it too.
//! input.press(&mut ui, &mut app, "enter");
//! assert_eq!(app.clicks, 2);
//! ```

use std::any::TypeId;
use std::collections::HashMap;

use super::a11y::Role;
use super::keymap::{key_name, Action, Keymap, Keystroke, Match};
use super::text_field::KeyModifiers;
use super::tree::{NodeId, Ui};

/// Which pointer button.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PointerButton {
    #[default]
    Primary,
    Secondary,
    Middle,
}

/// Something the user did, in UI pixels from the top-left.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    PointerMove {
        x: f32,
        y: f32,
    },
    PointerDown {
        x: f32,
        y: f32,
        button: PointerButton,
    },
    PointerUp {
        x: f32,
        y: f32,
        button: PointerButton,
    },
    /// The press is not one after all (a finger on a slider that went on to
    /// scroll the page): whoever captured the pointer puts back what the
    /// press did. Sent only to the node that captured it.
    PointerCancel,
    /// A press and release on the same node, or a control activated from
    /// the keyboard. Made by dispatch; send the press and release instead.
    Click {
        x: f32,
        y: f32,
        button: PointerButton,
    },
    /// Scroll amounts in pixels; positive `dy` scrolls content up (moves
    /// the view down), as a wheel turned towards you does.
    Wheel {
        x: f32,
        y: f32,
        dx: f32,
        dy: f32,
    },
    /// A secondary click, or a finger held still: a context menu is wanted
    /// at `(x, y)`. Bubbles from the node under it; a listener that prevents
    /// its default stops a held finger being a long press too.
    ContextMenu {
        x: f32,
        y: f32,
    },
    /// A key, as the DOM's `KeyboardEvent.key` names it (`"a"`, `"A"`,
    /// `"Enter"`, `"ArrowLeft"`), with the modifiers held.
    Key {
        key: String,
        modifiers: KeyModifiers,
    },
    /// Text from an input method or a paste, to insert as it is.
    Text(String),
    /// Focus arrived at the target (it bubbles, as `focusin` does).
    Focus,
    /// Focus left the target (it bubbles, as `focusout` does).
    Blur,
    /// The pointer moved onto the target or into it. Only the target hears
    /// it, as with `pointerenter`.
    PointerEnter,
    /// The pointer left the target and everything in it.
    PointerLeave,
    /// A finger on a touch screen, `id` telling fingers apart, at `time`
    /// seconds on any steady clock. See [`Gesture`] for what fingers make.
    Touch {
        id: u64,
        phase: TouchPhase,
        x: f32,
        y: f32,
        time: f64,
    },
    /// What fingers did, recognised: delivered to the node the gesture
    /// began on. Made by dispatch from [`Event::Touch`].
    Gesture(Gesture),
    /// From an input method, to the focused node (see [`crate::ime`]).
    Ime(crate::ime::Ime),
}

/// Where a finger is in its touch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TouchPhase {
    Start,
    Move,
    End,
    /// The system took the touch away (a call came in).
    Cancel,
}

/// Where a continuous gesture is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GesturePhase {
    Began,
    Changed,
    Ended,
}

/// Where a pull to refresh is (see [`Input::on_refresh`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullPhase {
    /// Being pulled, not yet far enough to refresh on release.
    Pulling,
    /// Pulled far enough: letting go refreshes.
    Armed,
    /// Refreshing, held open at [`PULL_THRESHOLD`] until
    /// [`Input::finish_refresh`].
    Refreshing,
    /// Springing back closed.
    Settling,
}

/// A pull to refresh in progress: how far the content is pulled down, in
/// pixels, and what it is doing. The app draws the indicator in that space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pull {
    pub container: NodeId,
    pub distance: f32,
    pub phase: PullPhase,
}

/// How far a pull to refresh goes before letting go refreshes, in pixels;
/// also the space held open while refreshing.
pub const PULL_THRESHOLD: f32 = 64.0;

type RefreshHandler<S> = Box<dyn FnMut(&mut S, &mut Ui)>;
type FocusHandler<S> = Box<dyn FnMut(&mut S, &mut Ui)>;
type DragMoveHandler<S> = Box<dyn FnMut(&mut S, &mut Ui, &dyn std::any::Any, f32, f32)>;
type FilesHandler<S> = Box<dyn FnMut(&mut S, &mut Ui, &[DroppedFile], f32, f32)>;
type TimerFn<S> = Box<dyn FnMut(&mut S, &mut Ui)>;
type AfterHook<S> = Box<dyn FnMut(&mut Input<S>, &mut Ui, &mut S) -> bool>;

/// A file dropped onto the UI from outside it (the desktop, a file
/// manager): what the platform says of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DroppedFile {
    pub name: String,
    pub size: u64,
    /// Its media type, as far as the platform knows (`image/png`), or empty.
    pub kind: String,
}

/// How a node takes the pointer from what is under it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HitBehavior {
    /// It takes the pointer, as any node does.
    #[default]
    Block,
    /// It takes presses and hovering, but the wheel scrolls what is under
    /// it: an overlay over a list.
    BlockExceptScroll,
}

/// A timer (see [`Input::after`]), to cancel it by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimerId(u64);

struct Timer<S> {
    id: TimerId,
    due: f64,
    every: Option<f64>,
    f: TimerFn<S>,
    /// Stopped when this node is gone (see [`Input::after_for`]).
    owner: Option<NodeId>,
}
type DeleteHandler<S> = Box<dyn FnMut(&mut S, &mut Ui, NodeId)>;
type AccessActionFn<S> = Box<dyn FnMut(&mut S, &mut Ui, &str)>;
type TextGet<S> = Box<dyn Fn(&S) -> (String, usize, usize)>;
type TextSet<S> = Box<dyn FnMut(&mut S, &mut Ui, String, usize, usize)>;

/// What a swiped row is doing (see [`Input::on_swipe_delete`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwipePhase {
    /// Under the finger.
    Dragging,
    /// Let go part way: held open on its action, [`SWIPE_ACTION_WIDTH`]
    /// to the left, until a tap.
    Open,
    /// Springing back closed.
    Closing,
}

/// A row slid sideways in a swipeable list: which row, how far (negative,
/// to the left, in pixels), and what it is doing. The app moves the row's
/// content by `offset` and shows the action in the space it leaves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Swipe {
    pub container: NodeId,
    pub row: NodeId,
    pub offset: f32,
    pub phase: SwipePhase,
}

/// How wide a swiped row's action is: let go past half of it and the row
/// stays open on it.
pub const SWIPE_ACTION_WIDTH: f32 = 84.0;
/// Let go past this share of the row's width, and the row is deleted.
pub const SWIPE_DELETE_SHARE: f32 = 0.5;

/// A gesture recognised from touches. A finger also drives the pointer, so a tap clicks and a finger can
/// drag a slider that captures the pointer; a finger that travels instead
/// pans, two pinch, and one held still long-presses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gesture {
    /// A finger down and up without travelling, quickly.
    Tap { x: f32, y: f32 },
    /// A finger held still (see [`Input::tick`]).
    LongPress { x: f32, y: f32 },
    /// One finger travelling: by `(dx, dy)` since the last report. Unless
    /// a listener prevents it, the content under it scrolls with it.
    Pan {
        x: f32,
        y: f32,
        dx: f32,
        dy: f32,
        phase: GesturePhase,
    },
    /// Two fingers: `scale` is their distance over what it was when the
    /// second came down, about their midpoint.
    Pinch {
        x: f32,
        y: f32,
        scale: f32,
        phase: GesturePhase,
    },
}

/// How far a finger travels before it is a pan rather than a tap.
pub const TOUCH_SLOP: f32 = 8.0;
/// How far left of a scrollbar's thumb a finger still takes it: a thumb is
/// 6 to 10 pixels wide, much less than a fingertip.
const FINGER_REACH: f32 = 16.0;
/// How long a finger rests before it is a long press, in seconds.
pub const LONG_PRESS: f64 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq)]
enum TouchMode {
    /// One finger down, not yet anything else.
    Pending,
    Pan,
    Pinch {
        start: f32,
    },
    /// Recognised and finished (a long press, a pinch with one finger
    /// left): the rest of the touch is ignored.
    Done,
    /// The pointer it drives was captured: the touch is a pointer drag.
    Drag,
    /// Captured by a sideways control (a horizontal slider): a drag of it
    /// if the finger travels sideways, a pan of the page if it travels up
    /// or down (the control's press undone, [`Event::PointerCancel`]).
    Held,
    /// Sliding a row of a swipeable list sideways (see
    /// [`Input::on_swipe_delete`]); `base` is where the row was when the
    /// finger came down.
    Swipe {
        base: f32,
    },
}

/// A finger: its id, where it started, where it is.
type Finger = (u64, (f32, f32), (f32, f32));

#[derive(Debug, Clone, Default)]
struct Touches {
    down: Vec<Finger>,
    start_time: f64,
    mode: Option<TouchMode>,
    target: Option<NodeId>,
}

/// The phase an event is in when a listener hears it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// From the root inward to the target.
    Capture,
    /// From the target outward to the root. Where most listeners belong.
    Bubble,
}

/// What a listener can learn about the event in hand and ask of it.
#[derive(Debug)]
pub struct EventCx {
    /// The node the event is for.
    pub target: NodeId,
    /// The node whose listener is running.
    pub current: NodeId,
    pub phase: Phase,
    stopped: bool,
    prevented: bool,
    propagate: bool,
    focus: Option<Option<NodeId>>,
    capture: Option<Option<NodeId>>,
}

impl EventCx {
    fn new(target: NodeId) -> Self {
        EventCx {
            target,
            current: target,
            phase: Phase::Capture,
            stopped: false,
            prevented: false,
            propagate: false,
            focus: None,
            capture: None,
        }
    }

    /// No further listener hears this event.
    pub fn stop_propagation(&mut self) {
        self.stopped = true;
    }

    /// What would follow by default does not: no focus change on a press,
    /// no click from Enter, no focus move on Tab, no scrolling on a wheel.
    pub fn prevent_default(&mut self) {
        self.prevented = true;
    }

    /// For an action handler: let the action go on to handlers further out
    /// instead of stopping here.
    pub fn propagate(&mut self) {
        self.propagate = true;
    }

    /// Moves focus to `node` once this event is dispatched.
    pub fn focus(&mut self, node: NodeId) {
        self.focus = Some(Some(node));
    }

    /// Takes focus away once this event is dispatched.
    pub fn blur(&mut self) {
        self.focus = Some(None);
    }

    /// Sends pointer moves and the release to the listening node, wherever
    /// the pointer goes, until the button is released: what a slider being
    /// dragged wants.
    pub fn capture_pointer(&mut self) {
        self.capture = Some(Some(self.current));
    }

    pub fn release_pointer(&mut self) {
        self.capture = Some(None);
    }

    /// Whether the listening node is the event's target.
    pub fn at_target(&self) -> bool {
        self.current == self.target
    }
}

/// What dispatching an event came to.
#[derive(Debug, Default)]
pub struct Dispatched {
    /// The node it was delivered to.
    pub target: Option<NodeId>,
    /// A listener stopped it or prevented its default, or an action handler
    /// took it.
    pub handled: bool,
    /// The node a click landed on, if this event made one.
    pub clicked: Option<NodeId>,
    /// The drop target a drag was released on, if this event ended one
    /// there.
    pub dropped: Option<NodeId>,
    /// The action the keymap matched, whether or not a handler took it.
    pub action: Option<Box<dyn Action>>,
    pub focus_changed: bool,
    /// A scrolling box whose `scroll_top` this event changed (lay out again
    /// to see it).
    pub scrolled: Option<NodeId>,
}

type Listener<S> = Box<dyn FnMut(&mut S, &mut Ui, &Event, &mut EventCx)>;
type TooltipBuilder<S> = Box<dyn FnMut(&mut S, &mut Ui, NodeId) -> NodeId>;

/// The tooltip shown or waiting to be.
#[derive(Debug, Clone, Copy)]
struct Tip {
    /// The node with the tooltip the pointer is over.
    owner: NodeId,
    /// When the pointer was first seen over it (by `tick`); infinite once
    /// something dismissed it.
    since: f64,
    shown: Option<NodeId>,
}

type PreviewBuilder<S> =
    Box<dyn FnMut(&mut S, &mut Ui, &dyn std::any::Any, NodeId, NodeId) -> NodeId>;

type DragSource<S> = Box<dyn FnMut(&mut S, &Ui) -> Option<Box<dyn std::any::Any>>>;
type DropHandler<S> = Box<dyn FnMut(&mut S, &mut Ui, &dyn std::any::Any, &mut EventCx)>;

/// How far the pointer travels from a press before it drags.
pub const DRAG_THRESHOLD: f32 = 4.0;

/// A drag in progress.
struct Drag {
    source: NodeId,
    data: Box<dyn std::any::Any>,
    over: Option<NodeId>,
}
type ActionHandler<S> = Box<dyn FnMut(&mut S, &mut Ui, &dyn Action, &mut EventCx)>;

/// Listeners, focus, hover and the keymap for one [`Ui`]. `S` is the app's
/// state, handed to every listener.
pub struct Input<S = ()> {
    pub keymap: Keymap,
    listeners: HashMap<NodeId, Vec<(Phase, Listener<S>)>>,
    actions: HashMap<NodeId, Vec<(TypeId, ActionHandler<S>)>>,
    contexts: HashMap<NodeId, String>,
    focus: Option<NodeId>,
    hovered: Vec<NodeId>,
    pressed: Option<(NodeId, PointerButton)>,
    captured: Option<NodeId>,
    pointer: (f32, f32),
    touches: Touches,
    /// The press being dispatched is a finger's.
    finger: bool,
    ime_area: Option<super::Rect>,
    drag_sources: HashMap<NodeId, DragSource<S>>,
    drop_targets: HashMap<NodeId, Vec<(TypeId, DropHandler<S>)>>,
    /// A press on a drag source, not yet far enough to drag.
    armed: Option<(NodeId, (f32, f32))>,
    drag: Option<Drag>,
    /// A node shown under the pointer while dragging, and the `display`
    /// it shows with.
    preview: Option<(NodeId, super::Display)>,
    /// Makes a preview for each drag (see [`Input::drag_preview_with`]).
    preview_builder: Option<PreviewBuilder<S>>,
    /// The preview made for this drag, and where in it the pointer holds
    /// it.
    built_preview: Option<(NodeId, (f32, f32))>,
    tooltips: HashMap<NodeId, TooltipBuilder<S>>,
    tip: Option<Tip>,
    /// Seconds the pointer rests on a node before its tooltip shows.
    pub tooltip_delay: f64,
    /// Focus came from the keyboard.
    focus_visible: bool,
    click_outside: HashMap<NodeId, Listener<S>>,
    /// Everything outside this is inert to the pointer and out of the Tab
    /// order.
    modal: Option<NodeId>,
    /// The pointer is held near a scrolling box's edge, scrolling it.
    autoscrolling: bool,
    /// The clock, as `tick` last gave it.
    now: Option<f64>,
    /// The last press: when, where, which button, and how many in a row.
    last_press: Option<(f64, (f32, f32), PointerButton, u32)>,
    clicks: u32,
    /// A scrollbar thumb being dragged: its box, and where in the thumb
    /// it was grabbed.
    thumb_drag: Option<(NodeId, f32)>,
    /// Where a wheel notch is taking each box it is animating.
    wheel_targets: HashMap<NodeId, f32>,
    /// Content still gliding after a finger let go: what it was over, and
    /// how fast it moves (pixels a second, positive down the content).
    fling: Option<(NodeId, f32)>,
    /// The same sideways: positive along the content to the right.
    fling_x: Option<(NodeId, f32)>,
    /// A panning finger's recent positions, `(time, y)`, for its speed when
    /// it lets go.
    pan_samples: Vec<(f64, f32)>,
    /// Its recent sideways positions, `(time, x)`.
    pan_samples_x: Vec<(f64, f32)>,
    /// Boxes that take a finger's gestures first, or alongside scrolling
    /// (see [`Input::set_gesture_priority`]).
    gesture_priority: HashMap<NodeId, GesturePriority>,
    /// Custom accessibility actions' handlers (see
    /// [`Input::on_accessibility_action`]).
    a11y_actions: HashMap<NodeId, AccessActionFn<S>>,
    /// Text controls a host mirrors (see [`Input::register_text`]).
    text_targets: HashMap<NodeId, (TextGet<S>, TextSet<S>)>,
    /// Which way the pan scrolls, settled as it starts.
    pan_axis: PanAxis,
    /// What a pan scrolls sideways: the node under the finger, kept as the
    /// pan's vertical target follows the box it scrolls.
    pan_target_x: Option<NodeId>,
    /// The box a pan last scrolled sideways: it settles on a snap point
    /// when the finger lifts.
    pan_scrolled_x: Option<NodeId>,
    /// A snapping box the wheel last moved sideways, and when: it settles
    /// once the wheel stops.
    wheel_x: Option<(NodeId, f64)>,
    /// The speed content was gliding at when a finger caught it: a flick
    /// the same way adds to it.
    carry: f32,
    /// Handlers for focus coming into a node's subtree, and going out of it.
    focus_in: HashMap<NodeId, Vec<FocusHandler<S>>>,
    focus_out: HashMap<NodeId, Vec<FocusHandler<S>>>,
    /// Nodes that let the wheel through (see [`HitBehavior`]).
    hit_behavior: HashMap<NodeId, HitBehavior>,
    /// Handlers for a drag moving over a node.
    drag_move_handlers: HashMap<NodeId, Vec<(TypeId, DragMoveHandler<S>)>>,
    /// Handlers for files dropped onto a node.
    files: HashMap<NodeId, FilesHandler<S>>,
    timers: Vec<Timer<S>>,
    next_timer: u64,
    /// Run after each dispatch and tick, with the input itself (see
    /// [`Input::on_after_dispatch`]).
    after_hooks: Vec<AfterHook<S>>,
    /// Boxes that refresh when pulled down from their top.
    refresh: std::collections::HashMap<NodeId, RefreshHandler<S>>,
    pull: Option<Pull>,
    /// Lists whose rows a finger swipes left to delete.
    swipe_delete: std::collections::HashMap<NodeId, DeleteHandler<S>>,
    swipe: Option<Swipe>,
    /// Flicks in a row, each while the last one's glide was still fast.
    streak: u32,
    /// Wheel deltas of a notch or more (a mouse wheel) move there over a
    /// tenth of a second or so, as browsers scroll; smaller ones (a
    /// trackpad, already smooth) at once.
    pub smooth_wheel: bool,
}

/// A wheel delta at least this large is a notch, and is animated.
const NOTCH: f32 = 40.0;
/// How long a notch's scroll takes.
const NOTCH_SECONDS: f32 = 0.16;
/// Seconds without a wheel event before a snapping box settles.
const WHEEL_SETTLE: f64 = 0.15;
/// How fast a fling loses speed: its velocity falls by e every 1/this
/// seconds. 2.0 is iOS's normal deceleration (0.998 a millisecond), so a
/// flick at `v` pixels a second travels `v / 2` pixels.
const FLING_FRICTION: f32 = 2.0;
/// A finger's speed at release is measured over this much of its last
/// movement, as phones measure it: earlier, slower movement does not count.
const FLING_WINDOW: f64 = 0.08;
/// A finger that stayed still this long before lifting lets go without a
/// fling.
const FLING_REST: f64 = 0.1;
/// No single fling is faster than this, in pixels a second.
const FLING_MAX: f32 = 8000.0;
/// A flick counts as one more in a row if the content was still gliding
/// at least this fast, in pixels a second, when the finger came down.
const FLING_STREAK_SPEED: f32 = 300.0;
/// Each flick in a row multiplies the glide's speed by this once more.
const FLING_BOOST: f32 = 1.5;
/// The multiplier stops growing after this many flicks in a row.
const FLING_STREAK_MAX: u32 = 12;
/// How fast a released pull to refresh springs to rest: the gap closes by
/// e every 1/this seconds.
const PULL_SPRING: f32 = 14.0;

/// How far a finger moving `dy` pulls a pull to refresh already `distance`
/// open: half as far, and less the further it is open, so it stretches
/// rather than following the finger.
fn pull_resisted(distance: f32, dy: f32) -> f32 {
    dy * 0.5 * (1.0 - distance / (PULL_THRESHOLD * 3.0)).max(0.15)
}

impl<S> Default for Input<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> Input<S> {
    pub fn new() -> Self {
        Input {
            keymap: Keymap::new(),
            listeners: HashMap::new(),
            actions: HashMap::new(),
            contexts: HashMap::new(),
            focus: None,
            hovered: vec![],
            pressed: None,
            captured: None,
            pointer: (f32::NAN, f32::NAN),
            touches: Touches::default(),
            finger: false,
            ime_area: None,
            drag_sources: HashMap::new(),
            drop_targets: HashMap::new(),
            armed: None,
            drag: None,
            preview: None,
            preview_builder: None,
            built_preview: None,
            tooltips: HashMap::new(),
            tip: None,
            tooltip_delay: 0.5,
            focus_visible: false,
            click_outside: HashMap::new(),
            modal: None,
            autoscrolling: false,
            now: None,
            last_press: None,
            clicks: 0,
            thumb_drag: None,
            wheel_targets: HashMap::new(),
            fling: None,
            fling_x: None,
            pan_samples: vec![],
            pan_samples_x: vec![],
            pan_axis: PanAxis::Free,
            gesture_priority: HashMap::new(),
            a11y_actions: HashMap::new(),
            text_targets: HashMap::new(),
            pan_target_x: None,
            pan_scrolled_x: None,
            wheel_x: None,
            carry: 0.0,
            focus_in: HashMap::new(),
            focus_out: HashMap::new(),
            hit_behavior: HashMap::new(),
            drag_move_handlers: HashMap::new(),
            files: HashMap::new(),
            timers: vec![],
            next_timer: 0,
            after_hooks: vec![],
            refresh: Default::default(),
            pull: None,
            swipe_delete: Default::default(),
            swipe: None,
            streak: 0,
            smooth_wheel: true,
        }
    }

    /// Calls `f` for every event reaching `node` in `phase`. Listeners stay
    /// until [`Input::remove_listeners`], or until the node is removed from
    /// the tree.
    pub fn on(
        &mut self,
        node: NodeId,
        phase: Phase,
        f: impl FnMut(&mut S, &mut Ui, &Event, &mut EventCx) + 'static,
    ) {
        self.listeners
            .entry(node)
            .or_default()
            .push((phase, Box::new(f)));
    }

    /// Calls `f` when `node`, or something inside it, is clicked — by the
    /// pointer or from the keyboard. Stops the click there.
    pub fn on_click(
        &mut self,
        node: NodeId,
        mut f: impl FnMut(&mut S, &mut Ui, &mut EventCx) + 'static,
    ) {
        self.on(node, Phase::Bubble, move |s, ui, e, cx| {
            if let Event::Click { .. } = e {
                f(s, ui, cx);
                cx.stop_propagation();
            }
        });
    }

    /// Calls `f` for key events the keymap did not take, while focus is in
    /// `node`. `f` returns whether it used the key; a used key stops there
    /// and does nothing by default.
    pub fn on_key(
        &mut self,
        node: NodeId,
        mut f: impl FnMut(&mut S, &mut Ui, &str, KeyModifiers, &mut EventCx) -> bool + 'static,
    ) {
        self.on(node, Phase::Bubble, move |s, ui, e, cx| {
            if let Event::Key { key, modifiers } = e
                && f(s, ui, key, *modifiers, cx)
            {
                cx.stop_propagation();
                cx.prevent_default();
            }
        });
    }

    /// Handles actions of type `A` while focus is in `node`. The nearest
    /// handler to the focused node runs, and the action stops there unless
    /// it calls [`EventCx::propagate`].
    pub fn on_action<A: Action>(
        &mut self,
        node: NodeId,
        mut f: impl FnMut(&mut S, &mut Ui, &A, &mut EventCx) + 'static,
    ) {
        let handler: ActionHandler<S> = Box::new(move |s, ui, a, cx| {
            if let Some(a) = a.as_any().downcast_ref::<A>() {
                f(s, ui, a, cx)
            }
        });
        self.actions
            .entry(node)
            .or_default()
            .push((TypeId::of::<A>(), handler));
    }

    /// Forgets `node`'s listeners, action handlers and key context.
    pub fn remove_listeners(&mut self, node: NodeId) {
        self.listeners.remove(&node);
        self.actions.remove(&node);
        self.contexts.remove(&node);
    }

    /// Names a key context that bindings can be limited to, in force while
    /// focus is inside `node`.
    pub fn set_key_context(&mut self, node: NodeId, context: impl Into<String>) {
        self.contexts.insert(node, context.into());
    }

    /// The shortcuts that work where focus is: the keymap's bindings in
    /// force there (see [`Keymap::available`]) whose actions a node on the
    /// focus path, or the root, handles (see [`on_action`](Self::on_action)).
    /// While a chord is started, only what can finish it. Each with its
    /// keys' label (`⌘S` with `mac`, `Ctrl+S` without) and its action's
    /// name.
    pub fn shortcuts_here(&self, ui: &Ui, mac: bool) -> Vec<(String, &'static str)> {
        let contexts = self.key_contexts(ui);
        let path: Vec<NodeId> = match self.focus.filter(|&f| ui.contains(f)) {
            Some(f) => ui.ancestors(f).collect(),
            None => vec![ui.root()],
        };
        let handled = |b: &super::keymap::Binding| {
            let id = b.action.as_any().type_id();
            path.iter().chain(std::iter::once(&ui.root())).any(|n| {
                self.actions
                    .get(n)
                    .is_some_and(|hs| hs.iter().any(|(t, _)| *t == id))
            })
        };
        let bindings = if self.keymap.is_pending() {
            self.keymap.continuations(&contexts)
        } else {
            self.keymap.available(&contexts)
        };
        bindings
            .into_iter()
            .filter(|b| handled(b))
            .map(|b| {
                (
                    b.keys
                        .iter()
                        .map(|k| k.label(mac))
                        .collect::<Vec<_>>()
                        .join(" "),
                    b.action.name().rsplit("::").next().unwrap_or(""),
                )
            })
            .collect()
    }

    /// The key contexts on the focus path, innermost first.
    pub fn key_contexts(&self, ui: &Ui) -> Vec<&str> {
        let Some(start) = self.focus.filter(|&f| ui.contains(f)) else {
            return self
                .contexts
                .get(&ui.root())
                .map(|c| vec![c.as_str()])
                .unwrap_or_default();
        };
        ui.ancestors(start)
            .filter_map(|n| self.contexts.get(&n).map(String::as_str))
            .collect()
    }

    pub fn focused(&self) -> Option<NodeId> {
        self.focus
    }

    /// Whether focus is on a text box or text area, where an input method
    /// should be on (winit's `set_ime_allowed`).
    pub fn wants_ime(&self, ui: &Ui) -> bool {
        self.focus.filter(|&f| ui.contains(f)).is_some_and(|f| {
            matches!(
                ui.semantics(f).map(|s| s.role),
                Some(Role::TextBox | Role::TextArea | Role::SearchBox)
            )
        })
    }

    /// Makes `node` (a text box, text area or search box) a text control a
    /// host can mirror into the platform's own text input, for its keyboard
    /// and input methods: `get` reads its text and selection (anchor and
    /// caret, in characters); `set` replaces them.
    pub fn register_text(
        &mut self,
        node: NodeId,
        get: impl Fn(&S) -> (String, usize, usize) + 'static,
        set: impl FnMut(&mut S, &mut Ui, String, usize, usize) + 'static,
    ) {
        self.text_targets.retain(|n, _| *n != node);
        self.text_targets
            .insert(node, (Box::new(get), Box::new(set)));
    }

    /// The focused text control (see [`register_text`](Self::register_text)):
    /// its node, text, anchor and caret.
    pub fn focused_text(&self, ui: &Ui, state: &S) -> Option<(NodeId, String, usize, usize)> {
        let f = self.focus.filter(|&f| ui.contains(f))?;
        let n = ui
            .ancestors(f)
            .find(|n| self.text_targets.contains_key(n))?;
        let (text, a, c) = (self.text_targets[&n].0)(state);
        Some((n, text, a, c))
    }

    /// Gives the focused text control new text and selection (what the
    /// platform's input holds after the user typed). Returns whether one
    /// has focus.
    pub fn set_focused_text(
        &mut self,
        ui: &mut Ui,
        state: &mut S,
        text: String,
        anchor: usize,
        caret: usize,
    ) -> bool {
        let Some(f) = self.focus.filter(|&f| ui.contains(f)) else {
            return false;
        };
        let Some(n) = ui.ancestors(f).find(|n| self.text_targets.contains_key(n)) else {
            return false;
        };
        if let Some((_, set)) = self.text_targets.get_mut(&n) {
            set(state, ui, text, anchor, caret);
        }
        self.run_after_hooks(ui, state);
        true
    }

    /// Tells the platform where the caret is, so the input method's
    /// candidate window opens beside it (winit's `set_ime_cursor_area`):
    /// set by whatever draws the focused field, read by the host.
    pub fn set_ime_area(&mut self, area: Option<super::Rect>) {
        self.ime_area = area;
    }

    pub fn ime_area(&self) -> Option<super::Rect> {
        self.ime_area
    }

    /// Where the pointer is, as of the last pointer event.
    pub fn pointer_position(&self) -> (f32, f32) {
        self.pointer
    }

    /// The node the pointer is over, as of the last pointer event.
    pub fn hovered(&self) -> Option<NodeId> {
        self.hovered.first().copied()
    }

    /// Whether the pointer is over `node` or something inside it (CSS
    /// `:hover`).
    pub fn is_hovered(&self, node: NodeId) -> bool {
        self.hovered.contains(&node)
    }

    /// The pointer's shape now: the cursor of the node a captured pointer
    /// is held by, or of the node under it. Set the window's (or canvas's)
    /// cursor from it after each pointer event.
    pub fn cursor(&self, ui: &Ui) -> super::Cursor {
        match self
            .captured
            .filter(|&c| ui.contains(c))
            .or_else(|| self.hovered())
        {
            Some(n) if ui.contains(n) => ui.cursor_of(n),
            _ => super::Cursor::Default,
        }
    }

    /// Whether `node` is being pressed (CSS `:active`).
    pub fn is_pressed(&self, ui: &Ui, node: NodeId) -> bool {
        self.pressed
            .is_some_and(|(p, _)| ui.contains(p) && ui.ancestors(p).any(|a| a == node))
    }

    /// Whether focus is on `node` or inside it (CSS `:focus-within`).
    pub fn contains_focus(&self, ui: &Ui, node: NodeId) -> bool {
        self.focus
            .is_some_and(|f| ui.contains(f) && ui.ancestors(f).any(|a| a == node))
    }

    /// Moves focus, sending [`Event::Blur`] and [`Event::Focus`]. Returns
    /// whether it moved.
    pub fn set_focus(&mut self, ui: &mut Ui, state: &mut S, node: Option<NodeId>) -> bool {
        let node = node.filter(|&n| ui.contains(n));
        if node == self.focus {
            return false;
        }
        let old = std::mem::replace(&mut self.focus, node);
        // Bindings typed halfway belong to where focus was.
        self.keymap.cancel_pending();
        if let Some(old) = old.filter(|&o| ui.contains(o)) {
            self.deliver(ui, state, old, &Event::Blur);
        }
        if let Some(new) = node {
            self.deliver(ui, state, new, &Event::Focus);
        }
        // Into and out of whole subtrees, as the DOM's focusin and focusout:
        // a panel hears focus leave it, whichever of its fields had it.
        let path = |n: Option<NodeId>| -> Vec<NodeId> {
            n.filter(|&n| ui.contains(n))
                .map(|n| ui.ancestors(n).collect())
                .unwrap_or_default()
        };
        let (was, is) = (path(old), path(node));
        for n in was.iter().filter(|n| !is.contains(n)) {
            self.run_focus(ui, state, *n, false);
        }
        for n in is.iter().filter(|n| !was.contains(n)) {
            self.run_focus(ui, state, *n, true);
        }
        true
    }

    fn run_focus(&mut self, ui: &mut Ui, state: &mut S, node: NodeId, into: bool) {
        let map = if into {
            &mut self.focus_in
        } else {
            &mut self.focus_out
        };
        let Some(mut hs) = map.remove(&node) else {
            return;
        };
        for h in &mut hs {
            h(state, ui);
        }
        let map = if into {
            &mut self.focus_in
        } else {
            &mut self.focus_out
        };
        map.entry(node).or_default().splice(0..0, hs);
    }

    /// Calls `f` when focus comes into `node` or anything inside it from
    /// outside (the DOM's `focusin`).
    pub fn on_focus_in(&mut self, node: NodeId, f: impl FnMut(&mut S, &mut Ui) + 'static) {
        self.focus_in.entry(node).or_default().push(Box::new(f));
    }

    /// Calls `f` when focus leaves `node` and everything inside it: a popup
    /// closing, a form saving, when the user moves on.
    pub fn on_focus_out(&mut self, node: NodeId, f: impl FnMut(&mut S, &mut Ui) + 'static) {
        self.focus_out.entry(node).or_default().push(Box::new(f));
    }

    /// How `node` takes the pointer from what is under it (see
    /// [`HitBehavior`]).
    pub fn set_hit_behavior(&mut self, node: NodeId, behavior: HitBehavior) {
        self.hit_behavior.insert(node, behavior);
    }

    /// Calls `f` as a drag carrying a `T` moves over `node` (or what is
    /// inside it), with where the pointer is: for showing where it would
    /// land, or scrolling near an edge.
    pub fn on_drag_move<T: std::any::Any>(
        &mut self,
        node: NodeId,
        mut f: impl FnMut(&mut S, &mut Ui, &T, f32, f32) + 'static,
    ) {
        let h: DragMoveHandler<S> = Box::new(move |s, ui, data, x, y| {
            if let Some(d) = data.downcast_ref::<T>() {
                f(s, ui, d, x, y);
            }
        });
        self.drag_move_handlers
            .entry(node)
            .or_default()
            .push((TypeId::of::<T>(), h));
    }

    /// Calls `f` with files dropped onto `node` (or what is inside it) from
    /// outside the UI, and where.
    pub fn on_drop_files(
        &mut self,
        node: NodeId,
        f: impl FnMut(&mut S, &mut Ui, &[DroppedFile], f32, f32) + 'static,
    ) {
        self.files.insert(node, Box::new(f));
    }

    /// The node that would take files dropped at `(x, y)`, to show it while
    /// they are dragged over.
    pub fn files_target(&self, ui: &Ui, x: f32, y: f32) -> Option<NodeId> {
        let hit = ui.hit_test(x, y).filter(|&t| self.inside_modal(ui, t))?;
        ui.ancestors(hit).find(|n| self.files.contains_key(n))
    }

    /// Files dropped at `(x, y)` from outside (the host hears the drop):
    /// the node taking them is handed them. Returns it.
    pub fn drop_files(
        &mut self,
        ui: &mut Ui,
        state: &mut S,
        x: f32,
        y: f32,
        files: &[DroppedFile],
    ) -> Option<NodeId> {
        let target = self.files_target(ui, x, y)?;
        let mut f = self.files.remove(&target)?;
        f(state, ui, files, x, y);
        self.files.entry(target).or_insert(f);
        Some(target)
    }

    /// Calls `f` when a context menu is wanted on `node` (a secondary
    /// click, or a finger held still), with where; a held finger is then no
    /// long press.
    pub fn on_context_menu(
        &mut self,
        node: NodeId,
        mut f: impl FnMut(&mut S, &mut Ui, f32, f32) + 'static,
    ) {
        self.on(node, Phase::Bubble, move |s, ui, e, cx| {
            if let Event::ContextMenu { x, y } = *e {
                f(s, ui, x, y);
                cx.prevent_default();
                cx.stop_propagation();
            }
        });
    }

    /// Calls `f` once, `seconds` from the last [`tick`](Self::tick) (browsers'
    /// `setTimeout`), from a later tick.
    pub fn after(&mut self, seconds: f64, f: impl FnMut(&mut S, &mut Ui) + 'static) -> TimerId {
        self.add_timer(seconds, None, Box::new(f))
    }

    /// Calls `f` every `seconds`, from ticks, until cancelled.
    pub fn every(&mut self, seconds: f64, f: impl FnMut(&mut S, &mut Ui) + 'static) -> TimerId {
        self.add_timer(seconds, Some(seconds.max(1e-3)), Box::new(f))
    }

    fn add_timer(&mut self, seconds: f64, every: Option<f64>, f: TimerFn<S>) -> TimerId {
        self.next_timer += 1;
        let id = TimerId(self.next_timer);
        let due = self.now.unwrap_or(0.0) + seconds.max(0.0);
        self.timers.push(Timer {
            id,
            due,
            every,
            f,
            owner: None,
        });
        id
    }

    /// [`after`](Self::after), for as long as `node` is in the tree: the
    /// timer goes with it.
    pub fn after_for(
        &mut self,
        node: NodeId,
        seconds: f64,
        f: impl FnMut(&mut S, &mut Ui) + 'static,
    ) -> TimerId {
        let id = self.after(seconds, f);
        self.own_timer(id, node);
        id
    }

    /// [`every`](Self::every), for as long as `node` is in the tree.
    pub fn every_for(
        &mut self,
        node: NodeId,
        seconds: f64,
        f: impl FnMut(&mut S, &mut Ui) + 'static,
    ) -> TimerId {
        let id = self.every(seconds, f);
        self.own_timer(id, node);
        id
    }

    fn own_timer(&mut self, id: TimerId, node: NodeId) {
        if let Some(t) = self.timers.iter_mut().find(|t| t.id == id) {
            t.owner = Some(node);
        }
    }

    /// Calls `f` once, on the next dispatch or tick: after `node` is
    /// built (a load to start, a field to focus).
    pub fn on_appear(
        &mut self,
        node: NodeId,
        mut f: impl FnMut(&mut Input<S>, &mut Ui, &mut S) + 'static,
    ) {
        self.on_after_dispatch(move |input, ui, state| {
            if ui.contains(node) {
                f(input, ui, state);
            }
            false
        });
    }

    /// Calls `f` once, on the first dispatch or tick after `node` has gone
    /// from the tree.
    pub fn on_disappear(&mut self, node: NodeId, mut f: impl FnMut(&mut S, &mut Ui) + 'static) {
        self.on_after_dispatch(move |_, ui, state| {
            if ui.contains(node) {
                return true;
            }
            f(state, ui);
            false
        });
    }

    /// Shows what `open` builds (a prompt, a popover, a palette: it returns
    /// its root) while `presented` says so: the state decides, not a call. Checked after each
    /// dispatch and tick. When the state stops asking, the root is removed
    /// (the component tidies up: modality, focus); when it closes itself
    /// (answered, dismissed), `dismissed` tells the state.
    pub fn present(
        &mut self,
        presented: impl Fn(&S) -> bool + 'static,
        mut dismissed: impl FnMut(&mut S) + 'static,
        mut open: impl FnMut(&mut Ui, &mut Input<S>, &mut S) -> NodeId + 'static,
    ) {
        let mut shown: Option<NodeId> = None;
        self.on_after_dispatch(move |input, ui, state| {
            if let Some(n) = shown {
                if !ui.contains(n) {
                    shown = None;
                    if presented(state) {
                        dismissed(state);
                    }
                } else if !presented(state) {
                    if input.modal() == Some(n) {
                        input.set_modal(ui, state, None);
                    }
                    ui.remove(n);
                    shown = None;
                }
            } else if presented(state) {
                shown = Some(open(ui, input, state));
            }
            true
        });
    }

    /// Stops a timer. Returns whether it was waiting.
    pub fn cancel_timer(&mut self, id: TimerId) -> bool {
        let before = self.timers.len();
        self.timers.retain(|t| t.id != id);
        self.timers.len() != before
    }

    /// Seconds until the next timer is due, from the last tick: a host that
    /// sleeps between frames wakes then.
    pub fn next_timer(&self) -> Option<f64> {
        let now = self.now.unwrap_or(0.0);
        self.timers
            .iter()
            .map(|t| (t.due - now).max(0.0))
            .min_by(|a, b| a.total_cmp(b))
    }

    fn run_timers(&mut self, ui: &mut Ui, state: &mut S, time: f64) {
        self.timers
            .retain(|t| t.owner.is_none_or(|o| ui.contains(o)));
        let mut due: Vec<Timer<S>> = vec![];
        let mut i = 0;
        while i < self.timers.len() {
            if self.timers[i].due <= time {
                due.push(self.timers.remove(i));
            } else {
                i += 1;
            }
        }
        due.sort_by(|a, b| a.due.total_cmp(&b.due));
        for mut t in due {
            (t.f)(state, ui);
            if let Some(every) = t.every {
                // Due again an interval on (or on the next tick, if late).
                t.due = (t.due + every).max(time);
                self.timers.push(t);
            }
        }
    }

    /// Types `keys` (`"cmd-a backspace h i enter"`: keystrokes as keymaps
    /// spell them, by spaces), for tests.
    pub fn simulate_keystrokes(
        &mut self,
        ui: &mut Ui,
        state: &mut S,
        keys: &str,
    ) -> Result<(), String> {
        for k in keys.split_whitespace() {
            let stroke = super::keymap::Keystroke::parse(k)?;
            let key = match stroke.key.as_str() {
                "enter" => "Enter".to_owned(),
                "escape" => "Escape".to_owned(),
                "tab" => "Tab".to_owned(),
                "space" => " ".to_owned(),
                "backspace" => "Backspace".to_owned(),
                "delete" => "Delete".to_owned(),
                "up" | "down" | "left" | "right" => {
                    let mut d = stroke.key.clone();
                    d[..1].make_ascii_uppercase();
                    format!("Arrow{d}")
                }
                "home" => "Home".to_owned(),
                "end" => "End".to_owned(),
                "pageup" => "PageUp".to_owned(),
                "pagedown" => "PageDown".to_owned(),
                "minus" => "-".to_owned(),
                k if stroke.modifiers.shift && k.chars().count() == 1 => k.to_uppercase(),
                k => k.to_owned(),
            };
            self.dispatch(
                ui,
                state,
                Event::Key {
                    key,
                    modifiers: stroke.modifiers,
                },
            );
        }
        Ok(())
    }

    /// Delivers `event` and does what follows from it. See the
    /// [module docs](self).
    pub fn dispatch(&mut self, ui: &mut Ui, state: &mut S, event: Event) -> Dispatched {
        let secondary = match event {
            Event::PointerDown {
                x,
                y,
                button: PointerButton::Secondary,
            } => Some((x, y)),
            _ => None,
        };
        let out = self.dispatch_inner(ui, state, event);
        // A secondary click asks for a context menu where it was.
        if let Some((x, y)) = secondary
            && let Some(t) = ui.hit_test(x, y).filter(|&t| self.inside_modal(ui, t))
        {
            self.deliver(ui, state, t, &Event::ContextMenu { x, y });
        }
        self.run_after_hooks(ui, state);
        self.sync_states(ui);
        out
    }

    /// Calls `f` after each dispatch and tick, with the input itself, for
    /// as long as it returns `true`: for what listeners cannot do, since
    /// they are not given the input (a menu opening its submenu, closing
    /// itself and giving focus back).
    pub fn on_after_dispatch(
        &mut self,
        f: impl FnMut(&mut Input<S>, &mut Ui, &mut S) -> bool + 'static,
    ) {
        self.after_hooks.push(Box::new(f));
    }

    fn run_after_hooks(&mut self, ui: &mut Ui, state: &mut S) {
        if self.after_hooks.is_empty() {
            return;
        }
        let mut hooks = std::mem::take(&mut self.after_hooks);
        hooks.retain_mut(|h| h(self, ui, state));
        hooks.append(&mut self.after_hooks);
        self.after_hooks = hooks;
    }

    /// Whether focus last came from the keyboard (CSS `:focus-visible`).
    pub fn focus_visible(&self) -> bool {
        self.focus_visible && self.focus.is_some()
    }

    /// Tells the tree what the pointer and keyboard are doing, for state
    /// styles.
    pub fn sync_states(&self, ui: &mut Ui) {
        let path = |n: Option<NodeId>| -> Vec<NodeId> {
            n.filter(|&n| ui.contains(n))
                .map(|n| ui.ancestors(n).collect())
                .unwrap_or_default()
        };
        let interaction = super::states::Interaction {
            hovered: self.hovered.clone(),
            pressed: path(self.pressed.map(|p| p.0)),
            focused: self.focus.filter(|&f| ui.contains(f)),
            focus_visible: self.focus_visible,
        };
        ui.set_interaction(interaction);
    }

    fn dispatch_inner(&mut self, ui: &mut Ui, state: &mut S, event: Event) -> Dispatched {
        self.forget_removed(ui);
        let mut out = Dispatched::default();
        match &event {
            Event::PointerMove { x, y } => {
                if let Some((c, grab)) = self.thumb_drag.filter(|(c, _)| ui.contains(*c)) {
                    ui.reveal_scrollbar(c);
                    if let Some(to) = ui.scroll_for_thumb(c, *y - grab) {
                        ui.scroll_to(c, to);
                        out.scrolled = Some(c);
                    }
                    self.pointer = (*x, *y);
                    return out;
                }
                if self.drag_move(ui, state, *x, *y) {
                    return out;
                }
                let target = self.pointer_target(ui, *x, *y);
                self.hover(ui, state, *x, *y);
                if let Some(t) = target {
                    let cx = self.deliver(ui, state, t, &event);
                    self.apply(ui, state, &cx, &mut out);
                    out.target = Some(t);
                }
            }
            Event::PointerDown { x, y, button } => {
                self.dismiss_tooltip(ui);
                self.count_click((*x, *y), *button);
                // A scrollbar is the box's own: it scrolls, and nothing
                // else hears the press.
                if *button == PointerButton::Primary
                    && let Some((c, part)) = ui
                        .scrollbar_reach(*x, *y, if self.finger { FINGER_REACH } else { 0.0 })
                        .filter(|(c, _)| self.inside_modal(ui, *c))
                {
                    let g = ui.scrollbar(c).expect("found");
                    ui.reveal_scrollbar(c);
                    match part {
                        super::scrolling::ScrollbarPart::Thumb => {
                            self.thumb_drag = Some((c, *y - g.thumb.y));
                        }
                        super::scrolling::ScrollbarPart::Track => {
                            let (_, view, now) = ui.scroll_extent(c);
                            let page = if *y < g.thumb.y { -view } else { view };
                            ui.scroll_to(c, now + page);
                        }
                    }
                    out.scrolled = Some(c);
                    out.handled = true;
                    return out;
                }
                let hit = self
                    .captured
                    .filter(|&c| ui.contains(c))
                    .or_else(|| ui.hit_test(*x, *y));
                self.press_outside(ui, state, hit, &event);
                self.hover(ui, state, *x, *y);
                let Some(t) = self.pointer_target(ui, *x, *y) else {
                    return out;
                };
                out.target = Some(t);
                self.pressed = Some((t, *button));
                if *button == PointerButton::Primary {
                    self.armed = ui
                        .ancestors(t)
                        .find(|n| self.drag_sources.contains_key(n))
                        .map(|n| (n, (*x, *y)));
                }
                let cx = self.deliver(ui, state, t, &event);
                let prevented = cx.prevented;
                self.apply(ui, state, &cx, &mut out);
                if !prevented && cx.focus.is_none() {
                    let to = ui.ancestors(t).find(|&n| Self::focusable(ui, n));
                    self.focus_visible = false;
                    out.focus_changed |= self.set_focus(ui, state, to);
                }
            }
            Event::PointerCancel => {
                self.armed = None;
                self.pressed = None;
                let Some(t) = self.captured.take().filter(|&c| ui.contains(c)) else {
                    return out;
                };
                out.target = Some(t);
                let cx = self.deliver(ui, state, t, &event);
                self.apply(ui, state, &cx, &mut out);
                return out;
            }
            Event::PointerUp { x, y, button } => {
                self.armed = None;
                if self.thumb_drag.take().is_some() {
                    return out;
                }
                if self.drag.is_some() {
                    self.pressed = None;
                    out.dropped = self.drop(ui, state, *x, *y);
                    return out;
                }
                let target = self.pointer_target(ui, *x, *y);
                self.captured = None;
                let pressed = self.pressed.take();
                self.hover(ui, state, *x, *y);
                let Some(t) = target else { return out };
                out.target = Some(t);
                let cx = self.deliver(ui, state, t, &event);
                self.apply(ui, state, &cx, &mut out);
                if let Some((p, b)) = pressed
                    && b == *button
                    && ui.contains(p)
                    && ui.contains(t)
                {
                    let on: Vec<NodeId> = ui.ancestors(p).collect();
                    let common = ui.ancestors(t).find(|n| on.contains(n));
                    if let Some(common) = common {
                        let click = Event::Click {
                            x: *x,
                            y: *y,
                            button: *button,
                        };
                        let cx = self.deliver(ui, state, common, &click);
                        self.apply(ui, state, &cx, &mut out);
                        out.clicked = Some(common);
                    }
                }
            }
            Event::Wheel { x, y, dx, dy } => {
                self.dismiss_tooltip(ui);
                self.hover(ui, state, *x, *y);
                let Some(mut t) = ui.hit_test(*x, *y).filter(|&t| self.inside_modal(ui, t)) else {
                    return out;
                };
                // Through an overlay that lets the wheel by, to what is under.
                while let Some(o) = ui
                    .ancestors(t)
                    .find(|n| self.hit_behavior.get(n) == Some(&HitBehavior::BlockExceptScroll))
                {
                    match ui.hit_test_except(*x, *y, o) {
                        Some(under) if under != t => t = under,
                        _ => break,
                    }
                }
                out.target = Some(t);
                let cx = self.deliver(ui, state, t, &event);
                let prevented = cx.prevented;
                self.apply(ui, state, &cx, &mut out);
                if !prevented {
                    self.fling = None;
                    self.fling_x = None;
                    out.scrolled = if self.smooth_wheel && dy.abs() >= NOTCH {
                        Self::scroll_chain_x(ui, t, *dx);
                        self.scroll_smoothly_by(ui, t, *dy)
                    } else {
                        self.wheel_targets.clear();
                        Self::scroll_chain(ui, t, *dx, *dy)
                    };
                    if *dx != 0.0
                        && let Some(n) = ui
                            .ancestors(t)
                            .find(|&n| ui.style(n).scroll_x && ui.scroll_snap(n).is_some())
                    {
                        self.wheel_x = Some((n, self.now.unwrap_or(0.0)));
                    }
                }
            }
            Event::Key { key, .. } if self.drag.is_some() && key == "Escape" => {
                self.cancel_drag(ui);
                out.handled = true;
            }
            Event::Key { key, modifiers } => {
                self.dismiss_tooltip(ui);
                let t = self.key_target(ui);
                out.target = Some(t);
                let stroke = Keystroke::from_event(key, *modifiers);
                let contexts: Vec<String> = self
                    .key_contexts(ui)
                    .into_iter()
                    .map(str::to_owned)
                    .collect();
                let contexts: Vec<&str> = contexts.iter().map(String::as_str).collect();
                match self.keymap.match_stroke(&stroke, &contexts) {
                    Match::Pending => {
                        out.handled = true;
                        return out;
                    }
                    Match::Action(action) => {
                        let handled = self.run_action(ui, state, t, &*action, &mut out);
                        out.action = Some(action);
                        if handled {
                            out.handled = true;
                            return out;
                        }
                    }
                    Match::None => {}
                }
                let cx = self.deliver(ui, state, t, &event);
                let prevented = cx.prevented;
                self.apply(ui, state, &cx, &mut out);
                if !prevented {
                    self.key_default(ui, state, t, &stroke, &mut out);
                }
            }
            Event::Text(_) | Event::Ime(_) => {
                let t = self.key_target(ui);
                out.target = Some(t);
                let cx = self.deliver(ui, state, t, &event);
                self.apply(ui, state, &cx, &mut out);
            }
            Event::Touch {
                id,
                phase,
                x,
                y,
                time,
            } => return self.touch(ui, state, *id, *phase, *x, *y, *time),
            Event::ContextMenu { x, y } => {
                // Sent from outside: to what is under it.
                if let Some(t) = ui.hit_test(*x, *y).filter(|&t| self.inside_modal(ui, t)) {
                    out.target = Some(t);
                    let cx = self.deliver(ui, state, t, &event);
                    self.apply(ui, state, &cx, &mut out);
                }
            }
            Event::Gesture(g) => {
                // Sent from outside: to what is under it.
                let (x, y) = match *g {
                    Gesture::Tap { x, y } | Gesture::LongPress { x, y } => (x, y),
                    Gesture::Pan { x, y, .. } | Gesture::Pinch { x, y, .. } => (x, y),
                };
                if let Some(t) = ui.hit_test(x, y) {
                    out.target = Some(t);
                    let cx = self.deliver(ui, state, t, &event);
                    self.apply(ui, state, &cx, &mut out);
                }
            }
            Event::Click { .. }
            | Event::Focus
            | Event::Blur
            | Event::PointerEnter
            | Event::PointerLeave => {
                // Made by dispatch itself; sent from outside, they go to
                // where focus is, as a synthetic event would.
                let t = self.key_target(ui);
                out.target = Some(t);
                let cx = self.deliver(ui, state, t, &event);
                self.apply(ui, state, &cx, &mut out);
            }
        }
        out
    }

    /// Runs `action` as if a key bound to it were pressed: from the focused
    /// node outward. Returns whether a handler took it.
    pub fn dispatch_action(&mut self, ui: &mut Ui, state: &mut S, action: &dyn Action) -> bool {
        self.forget_removed(ui);
        let t = self.key_target(ui);
        let mut out = Dispatched::default();
        self.run_action(ui, state, t, action, &mut out)
    }

    /// Makes `node` (and what is inside it) a drag source: a press there
    /// that travels [`DRAG_THRESHOLD`] starts a drag carrying what `data`
    /// returns (`None` declines). A drag never also clicks.
    pub fn draggable<T: std::any::Any>(
        &mut self,
        node: NodeId,
        mut data: impl FnMut(&mut S, &Ui) -> Option<T> + 'static,
    ) {
        self.drag_sources.insert(
            node,
            Box::new(move |s, ui| data(s, ui).map(|t| Box::new(t) as Box<dyn std::any::Any>)),
        );
    }

    /// Calls `f` when a drag carrying a `T` is released over `node` (or
    /// something inside it that does not take it first).
    pub fn on_drop<T: std::any::Any>(
        &mut self,
        node: NodeId,
        mut f: impl FnMut(&mut S, &mut Ui, &T, &mut EventCx) + 'static,
    ) {
        let handler: DropHandler<S> = Box::new(move |s, ui, data, cx| {
            if let Some(t) = data.downcast_ref::<T>() {
                f(s, ui, t, cx)
            }
        });
        self.drop_targets
            .entry(node)
            .or_default()
            .push((TypeId::of::<T>(), handler));
    }

    /// A node to show under the pointer while dragging: make it hidden
    /// (`display: none`), and while a drag lasts it is shown, painted over
    /// everything, and kept by the pointer. `None` for no preview.
    pub fn set_drag_preview(&mut self, ui: &mut Ui, node: Option<NodeId>) {
        self.preview = node.map(|n| {
            let shown = match ui.style(n).display {
                super::Display::None => super::Display::Block,
                d => d,
            };
            (n, shown)
        });
    }

    /// Makes the preview of each drag with `build`, given what is dragged,
    /// the node it was dragged from and the root to add it under: a copy of
    /// the dragged item, say. It is held where it
    /// was grabbed, painted over everything, and removed when the drag
    /// ends. Takes the place of [`set_drag_preview`](Self::set_drag_preview).
    pub fn drag_preview_with(
        &mut self,
        build: impl FnMut(&mut S, &mut Ui, &dyn std::any::Any, NodeId, NodeId) -> NodeId + 'static,
    ) {
        self.preview_builder = Some(Box::new(build));
    }

    /// The node showing the drag, if one does.
    pub fn drag_preview(&self) -> Option<NodeId> {
        self.built_preview
            .map(|(n, _)| n)
            .or(self.drag.as_ref().and(self.preview.map(|(n, _)| n)))
    }

    /// What is being dragged, if anything.
    pub fn dragging(&self) -> Option<&dyn std::any::Any> {
        self.drag.as_ref().map(|d| d.data.as_ref())
    }

    /// The node a drag started from.
    pub fn drag_source(&self) -> Option<NodeId> {
        self.drag.as_ref().map(|d| d.source)
    }

    /// The drop target under the pointer that would take what is being
    /// dragged: highlight it.
    pub fn drag_over(&self) -> Option<NodeId> {
        self.drag.as_ref().and_then(|d| d.over)
    }

    /// Ends a drag without dropping (Escape does this).
    pub fn cancel_drag(&mut self, ui: &mut Ui) {
        self.drag = None;
        self.armed = None;
        self.hide_preview(ui);
    }

    fn hide_preview(&mut self, ui: &mut Ui) {
        if let Some((n, _)) = self.built_preview.take() {
            ui.remove(n);
        }
        if let Some((n, _)) = self.preview.filter(|(n, _)| ui.contains(*n)) {
            ui.undefer(n);
            ui.unanchor(n);
            ui.style_mut(n).display = super::Display::None;
        }
    }

    /// The nearest node under `(x, y)` taking drops of `ty`.
    fn drop_target(&self, ui: &Ui, x: f32, y: f32, ty: TypeId) -> Option<NodeId> {
        let shown = self
            .built_preview
            .map(|(n, _)| n)
            .or(self.preview.map(|(n, _)| n));
        let hit = match shown {
            Some(p) if ui.contains(p) => ui.hit_test_except(x, y, p),
            _ => ui.hit_test(x, y),
        }?;
        ui.ancestors(hit).find(|n| {
            self.drop_targets
                .get(n)
                .is_some_and(|hs| hs.iter().any(|(t, _)| *t == ty))
        })
    }

    /// Starts or carries on a drag. Returns whether the move belonged to
    /// one (and so goes no further).
    fn drag_move(&mut self, ui: &mut Ui, state: &mut S, x: f32, y: f32) -> bool {
        if self.drag.is_none() {
            let Some((source, (sx, sy))) = self.armed else {
                return false;
            };
            if ((x - sx).powi(2) + (y - sy).powi(2)).sqrt() < DRAG_THRESHOLD
                || self.captured.is_some()
            {
                return false;
            }
            self.armed = None;
            let data = match self.drag_sources.get_mut(&source) {
                Some(f) => f(state, ui),
                None => None,
            };
            let Some(data) = data else { return false };
            self.pressed = None;
            if let Some(build) = &mut self.preview_builder {
                let r = ui.rect(source);
                let root = ui.root();
                let n = build(state, ui, data.as_ref(), source, root);
                ui.defer(n, i32::MAX);
                self.built_preview = Some((n, (sx - r.x, sy - r.y)));
            } else if let Some((n, shown)) = self.preview.filter(|(n, _)| ui.contains(*n)) {
                ui.style_mut(n).display = shown;
                ui.defer(n, i32::MAX);
            }
            self.drag = Some(Drag {
                source,
                data,
                over: None,
            });
        }
        let ty = self
            .drag
            .as_ref()
            .expect("dragging")
            .data
            .as_ref()
            .type_id();
        let over = self.drop_target(ui, x, y, ty);
        self.drag.as_mut().expect("dragging").over = over;
        // Whatever is under it listening for the drag's moves hears them.
        let shown = self
            .built_preview
            .map(|(n, _)| n)
            .or(self.preview.map(|(n, _)| n));
        let hit = match shown {
            Some(p) if ui.contains(p) => ui.hit_test_except(x, y, p),
            _ => ui.hit_test(x, y),
        };
        if let Some(hit) = hit {
            let listening: Vec<NodeId> = ui
                .ancestors(hit)
                .filter(|n| self.drag_move_handlers.contains_key(n))
                .collect();
            for n in listening {
                let Some(mut hs) = self.drag_move_handlers.remove(&n) else {
                    continue;
                };
                let data = self.drag.as_ref().expect("dragging").data.as_ref();
                for (t, h) in &mut hs {
                    if *t == ty {
                        h(state, ui, data, x, y);
                    }
                }
                self.drag_move_handlers.insert(n, hs);
            }
        }
        self.pointer = (x, y);
        use super::layers::{Anchor, AnchorTo, Corner, Fit};
        if let Some((n, (gx, gy))) = self.built_preview.filter(|(n, _)| ui.contains(*n)) {
            // Held where it was grabbed.
            ui.anchor(
                n,
                Anchor::new(AnchorTo::Point(x - gx, y - gy), Corner::TopLeft).fit(Fit::None),
            );
        } else if let Some((n, _)) = self.preview.filter(|(n, _)| ui.contains(*n)) {
            ui.anchor(
                n,
                Anchor::new(AnchorTo::Point(x, y), Corner::TopLeft)
                    .offset(10.0, 10.0)
                    .fit(Fit::Snap),
            );
        }
        true
    }

    /// Releases a drag at `(x, y)`: the target there takes it, if one
    /// does. Returns the target.
    fn drop(&mut self, ui: &mut Ui, state: &mut S, x: f32, y: f32) -> Option<NodeId> {
        let drag = self.drag.take()?;
        self.hide_preview(ui);
        let ty = drag.data.as_ref().type_id();
        let target = self.drop_target(ui, x, y, ty)?;
        let mut handlers = self.drop_targets.remove(&target)?;
        let mut cx = EventCx::new(target);
        cx.phase = Phase::Bubble;
        for (t, f) in handlers.iter_mut() {
            if *t == ty {
                f(state, ui, drag.data.as_ref(), &mut cx);
            }
        }
        if ui.contains(target) {
            let added = self.drop_targets.remove(&target).unwrap_or_default();
            handlers.extend(added);
            self.drop_targets.insert(target, handlers);
        }
        let mut out = Dispatched::default();
        self.apply(ui, state, &cx, &mut out);
        Some(target)
    }

    /// A press at `(x0, y0)`, a move to `(x1, y1)` in a few steps, and a
    /// release there: a drag, if it starts on a drag source.
    pub fn drag(
        &mut self,
        ui: &mut Ui,
        state: &mut S,
        from: (f32, f32),
        to: (f32, f32),
    ) -> Dispatched {
        let b = PointerButton::Primary;
        self.dispatch(
            ui,
            state,
            Event::PointerMove {
                x: from.0,
                y: from.1,
            },
        );
        self.dispatch(
            ui,
            state,
            Event::PointerDown {
                x: from.0,
                y: from.1,
                button: b,
            },
        );
        for k in 1..=4 {
            let t = k as f32 / 4.0;
            let (x, y) = (from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t);
            self.dispatch(ui, state, Event::PointerMove { x, y });
        }
        self.dispatch(
            ui,
            state,
            Event::PointerUp {
                x: to.0,
                y: to.1,
                button: b,
            },
        )
    }

    // Simulated input: what a test (or an automation) does instead of a
    // person.

    /// A primary-button press and release at `(x, y)`.
    pub fn click(&mut self, ui: &mut Ui, state: &mut S, x: f32, y: f32) -> Dispatched {
        let button = PointerButton::Primary;
        self.dispatch(ui, state, Event::PointerMove { x, y });
        self.dispatch(ui, state, Event::PointerDown { x, y, button });
        self.dispatch(ui, state, Event::PointerUp { x, y, button })
    }

    /// Presses the keystrokes in `keys`, a binding's spelling separated by
    /// spaces (`"ctrl-s"`, `"tab"`, `"ctrl-k ctrl-d"`). Returns what the last
    /// came to. Panics on a keystroke that does not parse.
    pub fn press(&mut self, ui: &mut Ui, state: &mut S, keys: &str) -> Dispatched {
        let mut last = Dispatched::default();
        for k in keys.split_whitespace() {
            let stroke = Keystroke::parse(k).unwrap_or_else(|e| panic!("{e}"));
            let key = dom_key(&stroke);
            last = self.dispatch(
                ui,
                state,
                Event::Key {
                    key,
                    modifiers: stroke.modifiers,
                },
            );
        }
        last
    }

    /// Types `text`, a key event per character, as a keyboard would.
    pub fn type_text(&mut self, ui: &mut Ui, state: &mut S, text: &str) {
        for c in text.chars() {
            let modifiers = KeyModifiers {
                shift: c.is_uppercase(),
                ..Default::default()
            };
            self.dispatch(
                ui,
                state,
                Event::Key {
                    key: c.to_string(),
                    modifiers,
                },
            );
        }
    }

    /// Lets time pass for gestures that are made by waiting: a finger held
    /// still for [`LONG_PRESS`] seconds becomes a [`Gesture::LongPress`].
    /// Call it each frame while a finger is down, with the touch clock.
    pub fn tick(&mut self, ui: &mut Ui, state: &mut S, time: f64) -> Dispatched {
        let mut out = Dispatched::default();
        let dt = self.now.map_or(0.0, |n| (time - n).max(0.0)) as f32;
        self.now = Some(time);
        ui.poll_jobs();
        self.autoscroll(ui, dt);
        self.glide(ui, dt);
        self.settle_pull(ui, dt);
        self.run_timers(ui, state, time);
        self.run_after_hooks(ui, state);
        self.settle_swipe(ui, dt);
        // The wheel has stopped moving a snapping box: it settles.
        if let Some((n, at)) = self.wheel_x
            && time - at > WHEEL_SETTLE
        {
            self.wheel_x = None;
            if ui.contains(n) {
                let aim = ui.style(n).scroll_left;
                ui.settle_x(n, aim);
            }
        }
        self.wheel_targets
            .retain(|n, _| ui.contains(*n) && ui.is_animating(*n));
        self.tick_tooltip(ui, state, time);
        self.sync_states(ui);
        let t = &self.touches;
        if t.mode == Some(TouchMode::Pending)
            && t.down.len() == 1
            && time - t.start_time >= LONG_PRESS
        {
            let (x, y) = t.down[0].2;
            self.touches.mode = Some(TouchMode::Done);
            // A long press is not a click when the finger lifts.
            self.pressed = None;
            // A context menu, if something under the finger has one; else
            // the long press.
            let menu = ui
                .hit_test(x, y)
                .filter(|&t| self.inside_modal(ui, t))
                .is_some_and(|t| {
                    self.deliver(ui, state, t, &Event::ContextMenu { x, y })
                        .prevented
                });
            if !menu {
                self.gesture(ui, state, Gesture::LongPress { x, y }, &mut out);
            }
        }
        out
    }

    /// Shows a tooltip for `node` when the pointer rests on it for
    /// [`tooltip_delay`](Self::tooltip_delay) seconds (counted by
    /// [`tick`](Self::tick)): `build` adds it under the root it is given and
    /// returns it, and it is painted over everything, hung below `node`
    /// (above, if there is no room), and removed when the pointer leaves, or
    /// a press, key or wheel dismisses it.
    pub fn tooltip(
        &mut self,
        node: NodeId,
        build: impl FnMut(&mut S, &mut Ui, NodeId) -> NodeId + 'static,
    ) {
        self.tooltips.insert(node, Box::new(build));
    }

    /// The tooltip showing, if one is.
    pub fn tooltip_shown(&self) -> Option<NodeId> {
        self.tip.and_then(|t| t.shown)
    }

    /// Removes a tooltip showing, and keeps it away until the pointer moves
    /// to another node.
    pub fn dismiss_tooltip(&mut self, ui: &mut Ui) {
        if let Some(tip) = &mut self.tip {
            if let Some(n) = tip.shown.take() {
                ui.remove(n);
            }
            tip.since = f64::INFINITY;
        }
    }

    fn tick_tooltip(&mut self, ui: &mut Ui, state: &mut S, time: f64) {
        let owner = self
            .hovered()
            .filter(|&h| ui.contains(h))
            .and_then(|h| ui.ancestors(h).find(|n| self.tooltips.contains_key(n)));
        if self.tip.map(|t| Some(t.owner)) != Some(owner) {
            if let Some(n) = self.tip.and_then(|t| t.shown) {
                ui.remove(n);
            }
            self.tip = owner.map(|owner| Tip {
                owner,
                since: time,
                shown: None,
            });
        }
        let Some(tip) = self.tip else { return };
        let idle = self.pressed.is_none() && self.drag.is_none();
        if tip.shown.is_none() && idle && time - tip.since >= self.tooltip_delay {
            let root = ui.root();
            let Some(build) = self.tooltips.get_mut(&tip.owner) else {
                return;
            };
            let n = build(state, ui, root);
            use super::layers::{Anchor, AnchorTo, Corner, Fit};
            ui.defer(n, i32::MAX - 1);
            ui.anchor(
                n,
                Anchor::new(
                    AnchorTo::Node(tip.owner, Corner::BottomLeft),
                    Corner::TopLeft,
                )
                .offset(0.0, 6.0)
                .fit(Fit::Flip),
            );
            self.tip = Some(Tip {
                shown: Some(n),
                ..tip
            });
        }
    }

    /// How many presses in a row the last one was: 2 for a double click, 3
    /// for a triple. Presses count together when they come within half a
    /// second and a few pixels of each other, by the clock
    /// [`tick`](Self::tick) gives (without one, every press is the first).
    pub fn click_count(&self) -> u32 {
        self.clicks
    }

    fn count_click(&mut self, at: (f32, f32), button: PointerButton) {
        let now = self.now;
        let again = match (now, self.last_press) {
            (Some(t), Some((t0, p0, b0, n)))
                if b0 == button
                    && t - t0 <= 0.5
                    && (at.0 - p0.0).abs() <= 4.0
                    && (at.1 - p0.1).abs() <= 4.0 =>
            {
                n + 1
            }
            _ => 1,
        };
        self.clicks = again;
        self.last_press = now.map(|t| (t, at, button, again));
    }

    /// While a captured pointer is held past the edge of the scrolling box
    /// around what holds it, scrolls that box toward it — faster the
    /// further out it is — so a drag can select past what shows.
    fn autoscroll(&mut self, ui: &mut Ui, dt: f32) {
        self.autoscrolling = false;
        let Some(held) = self.captured.filter(|&c| ui.contains(c)) else {
            return;
        };
        let Some(scroller) = ui.scroll_container(held) else {
            return;
        };
        let r = ui.rect(scroller);
        let y = self.pointer.1;
        let past = if y < r.y {
            y - r.y
        } else if y > r.y + r.height {
            y - (r.y + r.height)
        } else {
            return;
        };
        self.autoscrolling = true;
        let speed = (past.abs() * 10.0).max(60.0) * past.signum();
        let (content, view, _) = ui.scroll_extent(scroller);
        let now = ui.style(scroller).scroll_top;
        let to = (now + speed * dt).clamp(0.0, (content - view).max(0.0));
        if to != now {
            ui.style_mut(scroller).scroll_top = to;
        }
    }

    /// Whether input wants another frame: a tooltip waiting to show, a
    /// finger down (a long press may come), or a drag or a pointer captured
    /// near a scrolling box's edge (autoscroll). With
    /// [`Ui::needs_frame`](super::Ui::needs_frame), all an idle loop asks.
    pub fn needs_frame(&self) -> bool {
        let tip_pending = self
            .tip
            .is_some_and(|t| t.shown.is_none() && t.since.is_finite());
        let pull_moving = self.pull.is_some_and(|p| match p.phase {
            PullPhase::Settling => true,
            PullPhase::Refreshing => (p.distance - PULL_THRESHOLD).abs() > 0.5,
            _ => false,
        });
        let swipe_moving = self.swipe.is_some_and(|s| match s.phase {
            SwipePhase::Closing => true,
            SwipePhase::Open => (s.offset + SWIPE_ACTION_WIDTH).abs() > 0.5,
            SwipePhase::Dragging => false,
        });
        tip_pending
            || self.touching()
            || self.drag.is_some()
            || self.autoscrolling
            || self.fling.is_some()
            || self.fling_x.is_some()
            || self.wheel_x.is_some()
            || pull_moving
            || swipe_moving
    }

    /// Makes `container`, a scrolling box, refresh when pulled down from its
    /// top, as lists on a phone do: a finger dragging down with the box at
    /// its top pulls it open (against growing resistance) instead of
    /// scrolling, and letting go past [`PULL_THRESHOLD`] calls `f`. The pull
    /// is then held open until [`finish_refresh`](Self::finish_refresh).
    /// [`pull`](Self::pull) says how far it is open, for the app to draw an
    /// indicator in.
    pub fn on_refresh(&mut self, container: NodeId, f: impl FnMut(&mut S, &mut Ui) + 'static) {
        self.refresh.insert(container, Box::new(f));
    }

    /// Makes the rows of `container` (its children: a virtual list's rows)
    /// swipe left to delete, as on a phone. A finger moving sideways on a
    /// row slides it; let go past [`SWIPE_DELETE_SHARE`] of its width, `f`
    /// is called with the row to delete; past half [`SWIPE_ACTION_WIDTH`],
    /// it stays open on its action (a tap there is the app's to handle,
    /// like any click); short of that, it closes. A tap anywhere else closes an open row and does nothing
    /// else. [`swipe`](Self::swipe) says where the row is, for the app to
    /// draw it.
    pub fn on_swipe_delete(
        &mut self,
        container: NodeId,
        f: impl FnMut(&mut S, &mut Ui, NodeId) + 'static,
    ) {
        self.swipe_delete.insert(container, Box::new(f));
    }

    /// The row slid sideways in `container`, while one is.
    pub fn swipe(&self, container: NodeId) -> Option<Swipe> {
        self.swipe.filter(|s| s.container == container)
    }

    /// Closes an open row.
    pub fn close_swipe(&mut self) {
        if let Some(s) = &mut self.swipe {
            s.phase = SwipePhase::Closing;
        }
    }

    /// The swipeable list and row under the touch, if any.
    fn swipe_row(&self, ui: &Ui) -> Option<(NodeId, NodeId)> {
        let t = self.touches.target.filter(|&t| ui.contains(t))?;
        let mut child = t;
        for n in ui.ancestors(t) {
            if self.swipe_delete.contains_key(&n) {
                return (n != t).then_some((n, child));
            }
            child = n;
        }
        None
    }

    /// A finger came down at `(x, y)` with a row open: closes it unless the
    /// finger is on its action. Returns whether it closed one.
    fn close_swipe_unless_on_action(&mut self, ui: &Ui, x: f32, y: f32) -> bool {
        let Some(s) = self.swipe.filter(|s| s.phase == SwipePhase::Open) else {
            return false;
        };
        if ui.contains(s.row) {
            let r = ui.rect(s.row);
            let on_action = y >= r.y
                && y < r.y + r.height
                && x >= r.x + r.width + s.offset
                && x < r.x + r.width;
            if on_action {
                return false;
            }
        }
        self.close_swipe();
        true
    }

    /// The finger let go of a swiped row: deleted, held open, or closed.
    fn release_swipe(&mut self, ui: &mut Ui, state: &mut S, cancel: bool) {
        let Some(s) = &mut self.swipe else { return };
        let width = ui.rect(s.row).width.max(1.0);
        if !cancel && -s.offset >= width * SWIPE_DELETE_SHARE {
            let (c, row) = (s.container, s.row);
            self.swipe = None;
            if let Some(mut f) = self.swipe_delete.remove(&c) {
                f(state, ui, row);
                self.swipe_delete.entry(c).or_insert(f);
            }
        } else if !cancel && -s.offset >= SWIPE_ACTION_WIDTH / 2.0 {
            s.phase = SwipePhase::Open;
        } else {
            s.phase = SwipePhase::Closing;
        }
    }

    /// Moves a released row toward where it rests: open on its action, or
    /// closed.
    fn settle_swipe(&mut self, ui: &Ui, dt: f32) {
        let Some(s) = &mut self.swipe else { return };
        if !ui.contains(s.row) {
            self.swipe = None;
            return;
        }
        let to = match s.phase {
            SwipePhase::Open => -SWIPE_ACTION_WIDTH,
            SwipePhase::Closing => 0.0,
            SwipePhase::Dragging => return,
        };
        s.offset += (to - s.offset) * (1.0 - (-PULL_SPRING * dt).exp());
        if (s.offset - to).abs() <= 0.5 {
            s.offset = to;
            if s.phase == SwipePhase::Closing {
                self.swipe = None;
            }
        }
    }

    /// The pull to refresh on `container`, while one is open.
    pub fn pull(&self, container: NodeId) -> Option<Pull> {
        self.pull.filter(|p| p.container == container)
    }

    /// The refresh of `container` is done: its pull springs back closed.
    pub fn finish_refresh(&mut self, container: NodeId) {
        if let Some(p) = &mut self.pull
            && p.container == container
            && p.phase == PullPhase::Refreshing
        {
            p.phase = PullPhase::Settling;
        }
    }

    /// A finger panned `finger_dy` (down positive): a pull to refresh takes
    /// it if one is open or can open here, else it scrolls.
    fn pan_by(&mut self, ui: &mut Ui, target: NodeId, finger_dy: f32) -> Option<NodeId> {
        if !ui.contains(target) {
            return None;
        }
        if let Some(p) = &mut self.pull
            && matches!(p.phase, PullPhase::Pulling | PullPhase::Armed)
        {
            let c = p.container;
            let mut rest = 0.0;
            if finger_dy > 0.0 {
                p.distance += pull_resisted(p.distance, finger_dy);
            } else {
                let back = (-finger_dy).min(p.distance);
                p.distance -= back;
                rest = -finger_dy - back;
            }
            p.phase = if p.distance >= PULL_THRESHOLD {
                PullPhase::Armed
            } else {
                PullPhase::Pulling
            };
            if p.distance <= 0.0 {
                self.pull = None;
            }
            // Pushed back past closed: the rest scrolls.
            return if rest > 0.0 {
                Self::scroll_chain(ui, target, 0.0, rest)
            } else {
                Some(c)
            };
        }
        if finger_dy > 0.0 && self.pull.is_none() {
            // The nearest refreshing box, with it and every scrolling box
            // inside it at the top: there is nothing left to scroll up.
            let mut at_top = true;
            for n in ui.ancestors(target) {
                if ui.style(n).scroll && ui.style(n).scroll_top > 0.0 {
                    at_top = false;
                }
                if self.refresh.contains_key(&n) {
                    if at_top {
                        let distance = pull_resisted(0.0, finger_dy);
                        let phase = if distance >= PULL_THRESHOLD {
                            PullPhase::Armed
                        } else {
                            PullPhase::Pulling
                        };
                        self.pull = Some(Pull {
                            container: n,
                            distance,
                            phase,
                        });
                        self.fling = None;
                        return Some(n);
                    }
                    break;
                }
            }
        }
        Self::scroll_chain(ui, target, 0.0, -finger_dy)
    }

    /// The finger let go: an armed pull refreshes, any other springs back.
    /// Returns whether a pull was open (so the release does not fling).
    fn release_pull(&mut self, ui: &mut Ui, state: &mut S, cancel: bool) -> bool {
        let Some(p) = &mut self.pull else {
            return false;
        };
        match p.phase {
            PullPhase::Armed if !cancel => {
                p.phase = PullPhase::Refreshing;
                let c = p.container;
                if let Some(mut f) = self.refresh.remove(&c) {
                    f(state, ui);
                    self.refresh.entry(c).or_insert(f);
                }
                true
            }
            PullPhase::Pulling | PullPhase::Armed => {
                p.phase = PullPhase::Settling;
                true
            }
            _ => false,
        }
    }

    /// Moves a pull that is settling or refreshing toward where it rests:
    /// closed, or held at the threshold.
    fn settle_pull(&mut self, ui: &Ui, dt: f32) {
        let Some(p) = &mut self.pull else { return };
        if !ui.contains(p.container) {
            self.pull = None;
            return;
        }
        let to = match p.phase {
            PullPhase::Settling => 0.0,
            PullPhase::Refreshing => PULL_THRESHOLD,
            _ => return,
        };
        p.distance += (to - p.distance) * (1.0 - (-PULL_SPRING * dt).exp());
        if p.phase == PullPhase::Settling && p.distance < 0.5 {
            self.pull = None;
        } else if p.phase == PullPhase::Refreshing && (p.distance - to).abs() <= 0.5 {
            p.distance = to;
        }
    }

    /// Whether a finger is down (so [`tick`](Self::tick) wants calling).
    pub fn touching(&self) -> bool {
        !self.touches.down.is_empty()
    }

    fn gesture(&mut self, ui: &mut Ui, state: &mut S, g: Gesture, out: &mut Dispatched) -> bool {
        let Some(t) = self.touches.target.filter(|&t| ui.contains(t)) else {
            return false;
        };
        out.target = Some(t);
        let cx = self.deliver(ui, state, t, &Event::Gesture(g));
        let prevented = cx.prevented;
        self.apply(ui, state, &cx, out);
        match self.priority_at(ui, t) {
            // Its gestures are its own: nothing scrolls under them.
            Some(GesturePriority::High) => true,
            // Heard, and scrolling goes on as well.
            Some(GesturePriority::Simultaneous) => false,
            None => prevented,
        }
    }

    /// Gives `node` (and what is inside it) a say over a finger's
    /// gestures: [`High`](GesturePriority::High) takes every pan that
    /// starts on it from scrolling, swiping rows and dragging (a drawing
    /// canvas, a map); [`Simultaneous`](GesturePriority::Simultaneous)
    /// hears its pans while what is under still scrolls, even if its
    /// listener prevents them (a parallax header). `None` clears it.
    pub fn set_gesture_priority(&mut self, node: NodeId, priority: Option<GesturePriority>) {
        match priority {
            Some(p) => self.gesture_priority.insert(node, p),
            None => self.gesture_priority.remove(&node),
        };
    }

    /// Offers `actions` on `node` to assistive technology (a row's
    /// "Delete" for someone who cannot swipe it), and calls `f` with the
    /// one asked for.
    pub fn on_accessibility_action(
        &mut self,
        ui: &mut Ui,
        node: NodeId,
        actions: &[&str],
        f: impl FnMut(&mut S, &mut Ui, &str) + 'static,
    ) {
        ui.semantics_mut(node).actions = actions.iter().map(|a| a.to_string()).collect();
        self.a11y_actions.retain(|n, _| ui.contains(*n));
        self.a11y_actions.insert(node, Box::new(f));
    }

    /// Performs custom action `name` on `node`, as assistive technology
    /// asks (the DOM mirror's `A11yAction::Custom`).
    /// Returns whether `node` has it.
    pub fn accessibility_action(
        &mut self,
        ui: &mut Ui,
        state: &mut S,
        node: NodeId,
        name: &str,
    ) -> bool {
        let offered = ui
            .semantics(node)
            .is_some_and(|s| s.actions.iter().any(|a| a == name));
        let Some(f) = self.a11y_actions.get_mut(&node).filter(|_| offered) else {
            return false;
        };
        f(state, ui, name);
        self.run_after_hooks(ui, state);
        true
    }

    /// The priority in effect at `node`: its own or its nearest ancestor's.
    fn priority_at(&self, ui: &Ui, node: NodeId) -> Option<GesturePriority> {
        if self.gesture_priority.is_empty() {
            return None;
        }
        ui.ancestors(node)
            .find_map(|a| self.gesture_priority.get(&a).copied())
    }

    #[allow(clippy::too_many_arguments)]
    fn touch(
        &mut self,
        ui: &mut Ui,
        state: &mut S,
        id: u64,
        phase: TouchPhase,
        x: f32,
        y: f32,
        time: f64,
    ) -> Dispatched {
        let mut out = Dispatched::default();
        let b = PointerButton::Primary;
        let primary = self.touches.down.first().map(|d| d.0);
        match phase {
            TouchPhase::Start => {
                // A finger still down under this id lost its end on the way
                // (a browser can drop it): it is gone, or every touch from
                // now on would count as a second finger.
                if self.touches.down.iter().any(|d| d.0 == id) {
                    self.touch(ui, state, id, TouchPhase::Cancel, x, y, time);
                }
                self.touches.down.push((id, (x, y), (x, y)));
                match self.touches.down.len() {
                    1 if self.close_swipe_unless_on_action(ui, x, y) => {
                        // A row was open and this was not its action: the
                        // touch only closes it, as on a phone.
                        self.touches.mode = Some(TouchMode::Done);
                    }
                    1 => {
                        // A finger down catches gliding content.
                        self.carry = self.fling.take().map_or(0.0, |(_, v)| v);
                        self.fling_x = None;
                        self.pan_scrolled_x = None;
                        self.pan_samples.clear();
                        self.pan_samples.push((time, y));
                        self.pan_samples_x.clear();
                        self.pan_samples_x.push((time, x));
                        self.touches.start_time = time;
                        self.touches.mode = Some(TouchMode::Pending);
                        self.touches.target = ui.hit_test(x, y);
                        self.finger = true;
                        out = self.dispatch(ui, state, Event::PointerDown { x, y, button: b });
                        self.finger = false;
                        // On a scrollbar's thumb, or captured: the finger
                        // drags, as the mouse would, and does not pan.
                        if self.thumb_drag.is_some() {
                            self.touches.mode = Some(TouchMode::Drag);
                        } else if let Some(c) = self.captured {
                            self.touches.mode = Some(if Self::sideways(ui, c) {
                                TouchMode::Held
                            } else {
                                TouchMode::Drag
                            });
                        }
                    }
                    2 if !matches!(self.touches.mode, Some(TouchMode::Drag | TouchMode::Held)) => {
                        // A second finger: a pinch, and no click.
                        self.pressed = None;
                        if self.touches.mode == Some(TouchMode::Pan) {
                            let (cx, cy) = self.touches.down[0].2;
                            let end = Gesture::Pan {
                                x: cx,
                                y: cy,
                                dx: 0.0,
                                dy: 0.0,
                                phase: GesturePhase::Ended,
                            };
                            self.gesture(ui, state, end, &mut out);
                        }
                        let (a, c) = (self.touches.down[0].2, self.touches.down[1].2);
                        let start = ((a.0 - c.0).powi(2) + (a.1 - c.1).powi(2)).sqrt().max(1.0);
                        self.touches.mode = Some(TouchMode::Pinch { start });
                        let (mx, my) = ((a.0 + c.0) / 2.0, (a.1 + c.1) / 2.0);
                        let g = Gesture::Pinch {
                            x: mx,
                            y: my,
                            scale: 1.0,
                            phase: GesturePhase::Began,
                        };
                        self.gesture(ui, state, g, &mut out);
                    }
                    _ => {}
                }
            }
            TouchPhase::Move => {
                let Some(finger) = self.touches.down.iter_mut().find(|d| d.0 == id) else {
                    return out;
                };
                let last = finger.2;
                finger.2 = (x, y);
                let start = finger.1;
                match self.touches.mode {
                    Some(TouchMode::Drag) if Some(id) == primary => {
                        out = self.dispatch(ui, state, Event::PointerMove { x, y });
                    }
                    Some(TouchMode::Held) if Some(id) == primary => {
                        let (dx, dy) = (x - start.0, y - start.1);
                        if (dx * dx + dy * dy).sqrt() > TOUCH_SLOP {
                            if dx.abs() >= dy.abs() {
                                // sideways: it was a drag of the control
                                self.touches.mode = Some(TouchMode::Drag);
                                out = self.dispatch(ui, state, Event::PointerMove { x, y });
                            } else {
                                // up or down: the page scrolls, as a phone's
                                // does; the control's press is undone
                                self.dispatch(ui, state, Event::PointerCancel);
                                self.touches.mode = Some(TouchMode::Pending);
                                self.touches.target = ui
                                    .hit_test(start.0, start.1)
                                    .and_then(|t| ui.ancestors(t).find(|&n| ui.style(n).scroll));
                                // as if it had moved here from where it began
                                if let Some(f) = self.touches.down.iter_mut().find(|d| d.0 == id) {
                                    f.2 = last;
                                }
                                return self.touch(ui, state, id, TouchPhase::Move, x, y, time);
                            }
                        }
                    }
                    Some(TouchMode::Pending) => {
                        let moved = ((x - start.0).powi(2) + (y - start.1).powi(2)).sqrt();
                        let high = self.touches.target.is_some_and(|t| {
                            self.priority_at(ui, t) == Some(GesturePriority::High)
                        });
                        if moved > TOUCH_SLOP && self.armed.is_some() && !high {
                            // Pressed on something draggable: the finger
                            // drags it, rather than scrolling what is under.
                            self.touches.mode = Some(TouchMode::Drag);
                            out = self.dispatch(ui, state, Event::PointerMove { x, y });
                        } else if moved > TOUCH_SLOP
                            && !high
                            && (x - start.0).abs() > (y - start.1).abs()
                            && let Some((container, row)) = self.swipe_row(ui)
                        {
                            // Sideways on a row of a swipeable list: the row
                            // slides, from where it was if it was open.
                            let base = self
                                .swipe
                                .filter(|s| s.row == row)
                                .map_or(0.0, |s| s.offset);
                            self.touches.mode = Some(TouchMode::Swipe { base });
                            self.pressed = None;
                            self.fling = None;
                            let offset = (base + x - start.0).min(0.0);
                            self.swipe = Some(Swipe {
                                container,
                                row,
                                offset,
                                phase: SwipePhase::Dragging,
                            });
                        } else if moved > TOUCH_SLOP {
                            self.touches.mode = Some(TouchMode::Pan);
                            self.pressed = None;
                            self.pan_target_x = self.touches.target;
                            self.pan_axis = self.touches.target.map_or(PanAxis::Free, |t| {
                                pan_axis(ui, t, x - start.0, y - start.1)
                            });
                            let began = Gesture::Pan {
                                x,
                                y,
                                dx: x - start.0,
                                dy: y - start.1,
                                phase: GesturePhase::Began,
                            };
                            if !self.gesture(ui, state, began, &mut out)
                                && let Some(t) = self.touches.target
                            {
                                out.scrolled = self.pan_both(ui, t, x - start.0, y - start.1);
                            }
                        }
                    }
                    Some(TouchMode::Pan) => {
                        let (dx, dy) = (x - last.0, y - last.1);
                        self.pan_samples.push((time, y));
                        self.pan_samples_x.push((time, x));
                        self.pan_samples_x
                            .retain(|&(t, _)| t >= time - 4.0 * FLING_WINDOW);
                        // Only the recent ones matter.
                        let keep = time - 4.0 * FLING_WINDOW;
                        self.pan_samples.retain(|&(t, _)| t >= keep);
                        let g = Gesture::Pan {
                            x,
                            y,
                            dx,
                            dy,
                            phase: GesturePhase::Changed,
                        };
                        if !self.gesture(ui, state, g, &mut out)
                            && let Some(t) = self.touches.target
                        {
                            out.scrolled = self.pan_both(ui, t, dx, dy);
                        }
                    }
                    Some(TouchMode::Swipe { base }) if Some(id) == primary => {
                        if let Some(s) = &mut self.swipe {
                            let width = ui.rect(s.row).width.max(1.0);
                            s.offset = (base + x - start.0).clamp(-width, 0.0);
                        }
                    }
                    Some(TouchMode::Pinch { start }) if self.touches.down.len() >= 2 => {
                        let (a, c) = (self.touches.down[0].2, self.touches.down[1].2);
                        let d = ((a.0 - c.0).powi(2) + (a.1 - c.1).powi(2)).sqrt();
                        let (mx, my) = ((a.0 + c.0) / 2.0, (a.1 + c.1) / 2.0);
                        let g = Gesture::Pinch {
                            x: mx,
                            y: my,
                            scale: d / start,
                            phase: GesturePhase::Changed,
                        };
                        self.gesture(ui, state, g, &mut out);
                    }
                    _ => {}
                }
            }
            TouchPhase::End | TouchPhase::Cancel => {
                let Some(at) = self.touches.down.iter().position(|d| d.0 == id) else {
                    return out;
                };
                let (_, _, (lx, ly)) = self.touches.down.remove(at);
                let cancel = phase == TouchPhase::Cancel;
                match self.touches.mode {
                    Some(TouchMode::Pending) if !cancel => {
                        out = self.dispatch(ui, state, Event::PointerUp { x, y, button: b });
                        let mut tap = Dispatched::default();
                        self.gesture(ui, state, Gesture::Tap { x, y }, &mut tap);
                    }
                    Some(TouchMode::Drag | TouchMode::Held) if Some(id) == primary => {
                        if cancel {
                            self.pressed = None;
                            self.captured = None;
                            self.thumb_drag = None;
                        } else {
                            out = self.dispatch(ui, state, Event::PointerUp { x, y, button: b });
                        }
                    }
                    Some(TouchMode::Pan) => {
                        let g = Gesture::Pan {
                            x: lx,
                            y: ly,
                            dx: 0.0,
                            dy: 0.0,
                            phase: GesturePhase::Ended,
                        };
                        let prevented = self.gesture(ui, state, g, &mut out);
                        // Let go of a pull: it refreshes or springs back, and
                        // nothing glides.
                        let prevented = self.release_pull(ui, state, cancel) || prevented;
                        // A finger that lets go moving keeps the content
                        // gliding, at the speed of its last movement; one
                        // that stopped first does not.
                        // The speed is the one up to its last move: a
                        // browser can report the lift itself well after it
                        // (Safari does), so timing back from the lift would
                        // find too few moves and no speed.
                        let moved = self.pan_samples.last().map_or(time, |&(t, _)| t);
                        let rested = time - moved > FLING_REST;
                        let recent: Vec<(f64, f32)> = self
                            .pan_samples
                            .iter()
                            .copied()
                            .filter(|&(t, _)| moved - t <= FLING_WINDOW)
                            .collect();
                        let v = match (recent.first(), recent.last()) {
                            _ if rested => 0.0,
                            (Some(&(t0, y0)), Some(&(t1, y1))) if t1 - t0 > 0.005 => {
                                -(y1 - y0) / (t1 - t0) as f32
                            }
                            _ => 0.0,
                        };
                        // Flicking again the way it was already going
                        // adds to the glide, as on a phone.
                        // Flicking again the way it was still going fast adds
                        // to the glide, and each flick in a row adds more: a
                        // list of any length can be crossed in a few flicks.
                        // One flick alone goes as far as ever.
                        let again = v * self.carry > 0.0 && self.carry.abs() > FLING_STREAK_SPEED;
                        self.streak = if again { self.streak + 1 } else { 0 };
                        let (v, cap) = if again {
                            let boost = FLING_BOOST.powi(self.streak.min(FLING_STREAK_MAX) as i32);
                            // As fast as one glide over everything there is
                            // to scroll, and no faster.
                            let travel = self
                                .touches
                                .target
                                .filter(|&t| ui.contains(t))
                                .and_then(|t| ui.ancestors(t).find(|&n| ui.style(n).scroll))
                                .map_or(0.0, |n| {
                                    let (content, view, _) = ui.scroll_extent(n);
                                    (content - view).max(0.0)
                                });
                            (
                                (v + self.carry) * boost,
                                FLING_MAX.max(travel * FLING_FRICTION),
                            )
                        } else {
                            (v, FLING_MAX)
                        };
                        let v = v.clamp(-cap, cap);
                        if !prevented
                            && v.abs() > 100.0
                            && let Some(t) = self.touches.target
                        {
                            self.fling = Some((t, v));
                            // The glide is timed from the release.
                            self.now = Some(time);
                        }
                        // Sideways, likewise (without the streak).
                        let recent: Vec<(f64, f32)> = self
                            .pan_samples_x
                            .iter()
                            .copied()
                            .filter(|&(t, _)| moved - t <= FLING_WINDOW)
                            .collect();
                        let vx = match (recent.first(), recent.last()) {
                            _ if rested || self.pan_axis == PanAxis::Y => 0.0,
                            (Some(&(t0, x0)), Some(&(t1, x1))) if t1 - t0 > 0.005 => {
                                -(x1 - x0) / (t1 - t0) as f32
                            }
                            _ => 0.0,
                        };
                        // A box that snaps eases onto the point nearest where
                        // the glide would have stopped, instead of gliding.
                        let snapping = self
                            .pan_scrolled_x
                            .filter(|&c| ui.contains(c) && ui.scroll_snap(c).is_some());
                        if let Some(c) = snapping.filter(|_| !prevented) {
                            let glide = if vx.abs() > 100.0 {
                                vx.clamp(-FLING_MAX, FLING_MAX) / FLING_FRICTION
                            } else {
                                0.0
                            };
                            let aim = ui.style(c).scroll_left + glide;
                            ui.settle_x(c, aim);
                        } else if !prevented
                            && vx.abs() > 100.0
                            && let Some(t) = self.pan_target_x
                        {
                            self.fling_x = Some((t, vx.clamp(-FLING_MAX, FLING_MAX)));
                            self.now = Some(time);
                        }
                        self.carry = 0.0;
                    }
                    Some(TouchMode::Swipe { .. }) if Some(id) == primary => {
                        self.release_swipe(ui, state, cancel);
                    }
                    Some(TouchMode::Pinch { start }) => {
                        let rest: Vec<(f32, f32)> = self
                            .touches
                            .down
                            .iter()
                            .map(|d| d.2)
                            .chain([(lx, ly)])
                            .collect();
                        let (a, c) = (rest[0], rest[rest.len() - 1]);
                        let d = ((a.0 - c.0).powi(2) + (a.1 - c.1).powi(2)).sqrt();
                        let g = Gesture::Pinch {
                            x: (a.0 + c.0) / 2.0,
                            y: (a.1 + c.1) / 2.0,
                            scale: d / start,
                            phase: GesturePhase::Ended,
                        };
                        self.gesture(ui, state, g, &mut out);
                        self.touches.mode = Some(TouchMode::Done);
                    }
                    _ => {}
                }
                if !matches!(self.touches.mode, Some(TouchMode::Pan)) {
                    // A finger that caught the content and let go without a
                    // flick: no glide carries over to the next.
                    self.carry = 0.0;
                    self.streak = 0;
                }
                if cancel {
                    self.pressed = None;
                }
                if self.touches.down.is_empty() {
                    self.touches = Touches::default();
                    // No finger hovers once lifted: hover styles stay on
                    // what was tapped otherwise, as a phone's browser leaves
                    // them, over the state the tap set.
                    if self.drag.is_none() {
                        self.hover(ui, state, f32::NEG_INFINITY, f32::NEG_INFINITY);
                    }
                }
            }
        }
        out
    }

    /// Turns the wheel by `dy` pixels with the pointer at `(x, y)`.
    pub fn scroll(&mut self, ui: &mut Ui, state: &mut S, x: f32, y: f32, dy: f32) -> Dispatched {
        self.dispatch(ui, state, Event::Wheel { x, y, dx: 0.0, dy })
    }

    // Dispatch itself.

    fn forget_removed(&mut self, ui: &Ui) {
        self.listeners.retain(|n, _| ui.contains(*n));
        self.actions.retain(|n, _| ui.contains(*n));
        self.contexts.retain(|n, _| ui.contains(*n));
        self.hovered.retain(|&n| ui.contains(n));
        self.drag_sources.retain(|n, _| ui.contains(*n));
        self.tooltips.retain(|n, _| ui.contains(*n));
        self.click_outside.retain(|n, _| ui.contains(*n));
        if self.modal.is_some_and(|m| !ui.contains(m)) {
            self.modal = None;
        }
        self.drop_targets.retain(|n, _| ui.contains(*n));
        self.focus_in.retain(|n, _| ui.contains(*n));
        self.focus_out.retain(|n, _| ui.contains(*n));
        self.hit_behavior.retain(|n, _| ui.contains(*n));
        self.drag_move_handlers.retain(|n, _| ui.contains(*n));
        self.files.retain(|n, _| ui.contains(*n));
        if self.preview.is_some_and(|(n, _)| !ui.contains(n)) {
            self.preview = None;
        }
        if self.focus.is_some_and(|f| !ui.contains(f)) {
            self.focus = None;
        }
        if self.captured.is_some_and(|c| !ui.contains(c)) {
            self.captured = None;
        }
    }

    fn pointer_target(&self, ui: &Ui, x: f32, y: f32) -> Option<NodeId> {
        self.captured
            .filter(|&c| ui.contains(c))
            .or_else(|| ui.hit_test(x, y))
            .filter(|&t| self.inside_modal(ui, t))
    }

    /// Whether `node` is inside the modal (or there is none).
    fn inside_modal(&self, ui: &Ui, node: NodeId) -> bool {
        self.modal
            .filter(|&m| ui.contains(m))
            .is_none_or(|m| ui.contains(node) && ui.ancestors(node).any(|a| a == m))
    }

    /// Calls `f` when the pointer is pressed anywhere outside `node` and
    /// everything inside it: a menu closing
    /// when the user clicks elsewhere.
    pub fn on_click_outside(
        &mut self,
        node: NodeId,
        f: impl FnMut(&mut S, &mut Ui, &Event, &mut EventCx) + 'static,
    ) {
        self.click_outside.insert(node, Box::new(f));
    }

    /// Makes everything outside `node` inert, as a dialog does: the pointer
    /// reaches only what is inside it (presses outside still reach
    /// [`on_click_outside`](Self::on_click_outside)), and Tab cycles only
    /// through what is inside it. Focus moves in if it was outside. `None`
    /// lifts it.
    pub fn set_modal(&mut self, ui: &mut Ui, state: &mut S, node: Option<NodeId>) {
        self.modal = node.filter(|&n| ui.contains(n));
        // Assistive technology is kept in it too.
        ui.set_accessibility_modal(self.modal);
        if let Some(m) = self.modal {
            let inside = self
                .focus
                .is_some_and(|f| ui.contains(f) && ui.ancestors(f).any(|a| a == m));
            if !inside {
                let first = ui.next_focus_within(Some(m), None, false);
                self.focus_visible = true;
                self.set_focus(ui, state, first);
            }
        }
        self.sync_states(ui);
    }

    pub fn modal(&self) -> Option<NodeId> {
        self.modal
    }

    fn press_outside(&mut self, ui: &mut Ui, state: &mut S, target: Option<NodeId>, event: &Event) {
        let outside: Vec<NodeId> = self
            .click_outside
            .keys()
            .copied()
            .filter(|&n| target.is_none_or(|t| !ui.ancestors(t).any(|a| a == n)))
            .collect();
        for n in outside {
            if let Some(mut f) = self.click_outside.remove(&n) {
                let mut cx = EventCx::new(n);
                cx.phase = Phase::Bubble;
                f(state, ui, event, &mut cx);
                if ui.contains(n) && !self.click_outside.contains_key(&n) {
                    self.click_outside.insert(n, f);
                }
            }
        }
    }

    fn key_target(&self, ui: &Ui) -> NodeId {
        self.focus.filter(|&f| ui.contains(f)).unwrap_or(ui.root())
    }

    fn focusable(ui: &Ui, node: NodeId) -> bool {
        ui.semantics(node).is_some_and(|s| s.is_focusable())
    }

    /// Updates what the pointer is over, telling the nodes it left and the
    /// ones it entered.
    fn hover(&mut self, ui: &mut Ui, state: &mut S, x: f32, y: f32) {
        self.pointer = (x, y);
        // Over a scrollbar, even one faded out: it shows, to be grabbed.
        if let Some((c, _)) = ui.scrollbar_at(x, y) {
            ui.reveal_scrollbar(c);
        }
        let path: Vec<NodeId> = match ui.hit_test(x, y).filter(|&t| self.inside_modal(ui, t)) {
            Some(t) => ui.ancestors(t).collect(),
            None => vec![],
        };
        if path == self.hovered {
            return;
        }
        let old = std::mem::replace(&mut self.hovered, path.clone());
        for &n in old.iter().filter(|n| !path.contains(n)) {
            if ui.contains(n) {
                self.deliver_to_target(ui, state, n, &Event::PointerLeave);
            }
        }
        for &n in path.iter().rev().filter(|n| !old.contains(n)) {
            if ui.contains(n) {
                self.deliver_to_target(ui, state, n, &Event::PointerEnter);
            }
        }
    }

    fn deliver_to_target(&mut self, ui: &mut Ui, state: &mut S, target: NodeId, event: &Event) {
        let mut cx = EventCx::new(target);
        for phase in [Phase::Capture, Phase::Bubble] {
            cx.phase = phase;
            self.call(ui, state, target, event, &mut cx);
        }
    }

    /// Capture from the root in to `target`, then bubble back out.
    fn deliver(&mut self, ui: &mut Ui, state: &mut S, target: NodeId, event: &Event) -> EventCx {
        let path: Vec<NodeId> = ui.ancestors(target).collect();
        let mut cx = EventCx::new(target);
        cx.phase = Phase::Capture;
        for &n in path.iter().rev() {
            if cx.stopped {
                return cx;
            }
            self.call(ui, state, n, event, &mut cx);
        }
        cx.phase = Phase::Bubble;
        for &n in &path {
            if cx.stopped {
                return cx;
            }
            self.call(ui, state, n, event, &mut cx);
        }
        cx
    }

    /// `node`'s listeners for the current phase. They are taken out of the
    /// map while they run, so they can be handed the tree and the state.
    fn call(&mut self, ui: &mut Ui, state: &mut S, node: NodeId, event: &Event, cx: &mut EventCx) {
        if !ui.contains(node) {
            return;
        }
        let Some(mut list) = self.listeners.remove(&node) else {
            return;
        };
        cx.current = node;
        for (phase, f) in list.iter_mut() {
            if *phase == cx.phase && !cx.stopped {
                f(state, ui, event, cx);
            }
        }
        // Listeners added for this node while it ran go after its own.
        if let Some(added) = self.listeners.remove(&node) {
            list.extend(added);
        }
        if ui.contains(node) {
            self.listeners.insert(node, list);
        }
    }

    /// Carries out what listeners asked for: focus and pointer capture.
    fn apply(&mut self, ui: &mut Ui, state: &mut S, cx: &EventCx, out: &mut Dispatched) {
        out.handled |= cx.stopped || cx.prevented;
        if let Some(capture) = cx.capture {
            self.captured = capture;
        }
        if let Some(focus) = cx.focus {
            out.focus_changed |= self.set_focus(ui, state, focus);
        }
    }

    fn run_action(
        &mut self,
        ui: &mut Ui,
        state: &mut S,
        target: NodeId,
        action: &dyn Action,
        out: &mut Dispatched,
    ) -> bool {
        let ty = action.as_any().type_id();
        let path: Vec<NodeId> = ui.ancestors(target).collect();
        let mut cx = EventCx::new(target);
        cx.phase = Phase::Bubble;
        let mut handled = false;
        for n in path {
            if !ui.contains(n) {
                continue;
            }
            let Some(mut list) = self.actions.remove(&n) else {
                continue;
            };
            cx.current = n;
            let mut stop = false;
            for (t, f) in list.iter_mut() {
                if *t == ty && !stop {
                    cx.propagate = false;
                    f(state, ui, action, &mut cx);
                    handled = true;
                    stop = !cx.propagate || cx.stopped;
                }
            }
            if let Some(added) = self.actions.remove(&n) {
                list.extend(added);
            }
            if ui.contains(n) {
                self.actions.insert(n, list);
            }
            if stop {
                break;
            }
        }
        self.apply(ui, state, &cx, out);
        handled
    }

    fn key_default(
        &mut self,
        ui: &mut Ui,
        state: &mut S,
        target: NodeId,
        stroke: &Keystroke,
        out: &mut Dispatched,
    ) {
        let m = stroke.modifiers;
        let plain = !m.control && !m.alt && !m.meta;
        match stroke.key.as_str() {
            "tab" if plain => {
                self.focus_visible = true;
                let scope = self.modal.filter(|&m| ui.contains(m));
                let next =
                    ui.next_focus_within(scope, self.focus.filter(|&f| ui.contains(f)), m.shift);
                out.focus_changed |= self.set_focus(ui, state, next);
            }
            key @ ("enter" | "space") if plain && !m.shift && Some(target) == self.focus => {
                let role = ui.semantics(target).map(|s| s.role);
                let activates = match key {
                    "enter" => matches!(
                        role,
                        Some(Role::Button | Role::Link | Role::MenuItem | Role::Tab | Role::Switch)
                    ),
                    _ => matches!(
                        role,
                        Some(
                            Role::Button
                                | Role::Checkbox
                                | Role::Switch
                                | Role::Radio
                                | Role::MenuItem
                                | Role::Tab
                        )
                    ),
                };
                if activates {
                    let r = ui.rect(target);
                    let click = Event::Click {
                        x: r.x + r.width / 2.0,
                        y: r.y + r.height / 2.0,
                        button: PointerButton::Primary,
                    };
                    let cx = self.deliver(ui, state, target, &click);
                    self.apply(ui, state, &cx, out);
                    out.clicked = Some(target);
                }
            }
            _ => {}
        }
    }

    /// Scrolls the innermost box under `target` that can still move by
    /// Whether `n` (or the control it is in) slides sideways: a slider
    /// wider than tall. A finger on one waits to see which way it goes.
    fn sideways(ui: &Ui, n: NodeId) -> bool {
        ui.ancestors(n).take(3).any(|a| {
            let r = ui.rect(a);
            ui.semantics(a).is_some_and(|s| s.role == Role::Slider) && r.width > r.height
        })
    }

    /// `dy`, chaining outward from one at its end, as a page does.
    /// A wheel notch: the innermost box that can move animates there, from
    /// wherever it is, to where the notches so far are taking it.
    fn scroll_smoothly_by(&mut self, ui: &mut Ui, target: NodeId, dy: f32) -> Option<NodeId> {
        use super::animation::{Animation, Property, Value};
        let chain: Vec<NodeId> = ui
            .ancestors(target)
            .filter(|&n| ui.style(n).scroll)
            .collect();
        for n in chain {
            let (content, visible, _) = ui.scroll_extent(n);
            let max = (content - visible).max(0.0);
            let now = ui.style(n).scroll_top.clamp(0.0, max);
            let from = self
                .wheel_targets
                .get(&n)
                .copied()
                .filter(|_| ui.is_animating(n))
                .unwrap_or(now);
            let to = (from + dy).clamp(0.0, max);
            if to != from {
                self.wheel_targets.insert(n, to);
                ui.animate(
                    n,
                    Animation::new(
                        Property::ScrollTop,
                        Value::Number(now),
                        Value::Number(to),
                        NOTCH_SECONDS,
                    )
                    .easing(threers_math::easing::Easing::QuadOut),
                );
                return Some(n);
            }
        }
        None
    }

    /// Scrolls the innermost box around `target` that scrolls sideways and
    /// can move by `dx`.
    fn scroll_chain_x(ui: &mut Ui, target: NodeId, dx: f32) -> Option<NodeId> {
        if dx == 0.0 {
            return None;
        }
        let chain: Vec<NodeId> = ui
            .ancestors(target)
            .filter(|&n| ui.style(n).scroll_x)
            .collect();
        for n in chain {
            let (content, visible, _) = ui.scroll_extent_x(n);
            let max = (content - visible).max(0.0);
            let scrolled = ui.style(n).scroll_left.clamp(0.0, max);
            let to = (scrolled + dx).clamp(0.0, max);
            if to != scrolled {
                // Scrolled directly: settling onto a snap point gives way.
                ui.stop_animations(n, Some(super::animation::Property::ScrollLeft));
                ui.style_mut(n).scroll_left = to;
                return Some(n);
            }
        }
        None
    }

    /// A finger panned by `(dx, dy)`: scrolls what the pan's axis allows,
    /// sideways from the node it started on (as the vertical target follows
    /// the box it scrolls, a sideways strip inside a page keeps moving).
    fn pan_both(&mut self, ui: &mut Ui, t: NodeId, dx: f32, dy: f32) -> Option<NodeId> {
        let down = if self.pan_axis == PanAxis::X {
            None
        } else {
            self.pan_by(ui, t, dy)
        };
        let side = match (self.pan_axis, self.pan_target_x.filter(|&s| ui.contains(s))) {
            (PanAxis::Y, _) | (_, None) => None,
            (_, Some(s)) => Self::scroll_chain_x(ui, s, -dx),
        };
        if side.is_some() {
            self.pan_scrolled_x = side;
        }
        self.follow(down);
        down.or(side)
    }

    /// A pan scrolled `scrolled`: from now on it, and the fling after it,
    /// move that box (and on to the boxes around it), not the node the
    /// finger first touched. That node may be a row of a virtual list, which
    /// is dropped as it scrolls out of sight; then the pan and the glide
    /// would stop dead.
    fn follow(&mut self, scrolled: Option<NodeId>) {
        if scrolled.is_some() {
            self.touches.target = scrolled;
        }
    }

    /// Carries a fling on by `dt` seconds.
    fn glide(&mut self, ui: &mut Ui, dt: f32) {
        if let Some((target, v)) = self.fling_x {
            if !ui.contains(target) {
                self.fling_x = None;
            } else if dt > 0.0 {
                let decay = (-FLING_FRICTION * dt).exp();
                let moved = Self::scroll_chain_x(ui, target, v * (1.0 - decay) / FLING_FRICTION);
                let v = v * decay;
                self.fling_x = (moved.is_some() && v.abs() > 20.0).then_some((target, v));
            }
        }
        let Some((target, v)) = self.fling else {
            return;
        };
        if !ui.contains(target) {
            // Gone: nothing to glide, and no more frames wanted for it.
            self.fling = None;
            return;
        }
        if dt <= 0.0 {
            return;
        }
        // Exactly what the decay covers in `dt`, so the glide goes as far
        // at 30 frames a second as at 120.
        let decay = (-FLING_FRICTION * dt).exp();
        let distance = v * (1.0 - decay) / FLING_FRICTION;
        let v = v * decay;
        let moved = Self::scroll_chain(ui, target, 0.0, distance);
        self.fling = (moved.is_some() && v.abs() > 20.0).then_some((target, v));
    }

    fn scroll_chain(ui: &mut Ui, target: NodeId, dx: f32, dy: f32) -> Option<NodeId> {
        let sideways = Self::scroll_chain_x(ui, target, dx);
        if dy == 0.0 {
            return sideways;
        }
        let chain: Vec<NodeId> = ui
            .ancestors(target)
            .filter(|&n| ui.style(n).scroll)
            .collect();
        for n in chain {
            let (content, visible, _) = ui.scroll_extent(n);
            let max = (content - visible).max(0.0);
            // From where it was last asked to be, which several events
            // between two layouts move on from.
            let scrolled = ui.style(n).scroll_top.clamp(0.0, max);
            let to = (scrolled + dy).clamp(0.0, max);
            if to != scrolled {
                // Scrolled directly: any notch animating it gives way.
                ui.stop_animations(n, Some(super::animation::Property::ScrollTop));
                ui.style_mut(n).scroll_top = to;
                return Some(n);
            }
        }
        None
    }
}

/// The DOM `key` a keystroke would come with.
fn dom_key(stroke: &Keystroke) -> String {
    let k = stroke.key.as_str();
    let named = match k {
        "left" | "right" | "up" | "down" => {
            let mut c = k.chars();
            let first = c.next().unwrap().to_ascii_uppercase();
            return format!("Arrow{first}{}", c.as_str());
        }
        "space" => return " ".into(),
        "enter" => "Enter",
        "escape" => "Escape",
        "tab" => "Tab",
        "backspace" => "Backspace",
        "delete" => "Delete",
        "home" => "Home",
        "end" => "End",
        "pageup" => "PageUp",
        "pagedown" => "PageDown",
        _ => {
            if k.chars().count() == 1 {
                return if stroke.modifiers.shift {
                    k.to_uppercase()
                } else {
                    k.to_owned()
                };
            }
            // F-keys and the rest: capitalised, as the DOM writes them.
            let mut c = k.chars();
            let first = c.next().map(|f| f.to_ascii_uppercase()).unwrap_or_default();
            return format!("{first}{}", c.as_str());
        }
    };
    debug_assert_eq!(key_name(named), k);
    named.into()
}

/// Which way a pan scrolls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PanAxis {
    /// Both ways (a box that scrolls both ways, or only one kind of box
    /// under the finger).
    Free,
    /// Sideways only: it started sideways over a box that scrolls so.
    X,
    /// Up and down only: it started that way.
    Y,
}

/// The axis a pan starting by `(dx, dy)` over `target` keeps to, as phones
/// lock a pan between a strip that scrolls sideways and the page around
/// it: the way it starts, where boxes scroll both ways on that line.
fn pan_axis(ui: &Ui, target: NodeId, dx: f32, dy: f32) -> PanAxis {
    let sideways = ui.ancestors(target).find(|&n| ui.style(n).scroll_x);
    let down = ui.ancestors(target).find(|&n| ui.style(n).scroll);
    match (sideways, down) {
        (Some(a), Some(b)) if a != b => {
            if dx.abs() > dy.abs() {
                PanAxis::X
            } else {
                PanAxis::Y
            }
        }
        _ => PanAxis::Free,
    }
}

/// A box's say over a finger's gestures (see [`Input::set_gesture_priority`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GesturePriority {
    /// Its gestures come first: nothing under it scrolls, swipes or drags.
    High,
    /// It hears gestures while what is under it scrolls too.
    Simultaneous,
}
