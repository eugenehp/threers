//! Time-based game logic: countdowns, cooldowns, "do this in two seconds", and
//! things that glide from one place to another.
//!
//! All of it runs on game time, so pausing the game pauses it and slow motion
//! slows it.

use crate::app::Time;
use crate::query::Node;
use crate::world::{Component, Entity, World};
use threers::easing::Easing;
use threers::math::{Quaternion, Vector3};

/// A countdown. Tick it with the frame's or the step's `delta`.
///
/// ```
/// # use threers_ecs::prelude::*;
/// let mut cooldown = Timer::once(0.5);
/// assert!(!cooldown.tick(0.3));
/// assert!(cooldown.tick(0.3));       // finished on this tick
/// assert!(!cooldown.tick(0.3));      // and only reported once
/// assert!(cooldown.finished());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timer {
    duration: f32,
    elapsed: f32,
    repeating: bool,
    /// Times it went off on the last tick: more than one for a short repeating
    /// timer on a long tick.
    fired: u32,
}

impl Timer {
    /// Goes off once, after `seconds`.
    pub fn once(seconds: f32) -> Self {
        Self {
            duration: seconds.max(0.0),
            elapsed: 0.0,
            repeating: false,
            fired: 0,
        }
    }

    /// Goes off every `seconds`, keeping the remainder so it does not drift.
    pub fn repeating(seconds: f32) -> Self {
        Self {
            duration: seconds.max(1e-6),
            elapsed: 0.0,
            repeating: true,
            fired: 0,
        }
    }

    /// Move on by `delta`, returning whether it went off.
    pub fn tick(&mut self, delta: f32) -> bool {
        self.fired = 0;
        if !self.repeating && self.finished() {
            return false;
        }
        self.elapsed += delta.max(0.0);
        if self.repeating {
            while self.elapsed >= self.duration {
                self.elapsed -= self.duration;
                self.fired += 1;
            }
        } else if self.elapsed >= self.duration {
            self.elapsed = self.duration;
            self.fired = 1;
        }
        self.fired > 0
    }

    /// How many times it went off on the last tick.
    pub fn times_fired(&self) -> u32 {
        self.fired
    }

    /// A one-shot that has gone off. A repeating timer never finishes.
    pub fn finished(&self) -> bool {
        !self.repeating && self.elapsed >= self.duration
    }

    /// 0 at the start, 1 when it goes off.
    pub fn fraction(&self) -> f32 {
        if self.duration <= 0.0 {
            1.0
        } else {
            self.elapsed / self.duration
        }
    }

    /// Seconds left; 0 once done.
    pub fn remaining(&self) -> f32 {
        (self.duration - self.elapsed).max(0.0)
    }

    /// Back to the start, as if just made.
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
        self.fired = 0;
    }
}

type Delayed = Box<dyn FnOnce(&mut World)>;

/// Work waiting for its time, as a resource. Use [`World::after`].
#[derive(Default)]
pub struct Delays {
    waiting: Vec<(f64, Delayed)>,
}

impl World {
    /// Run `f` after `seconds` of game time.
    ///
    /// ```
    /// # use threers_ecs::prelude::*;
    /// let mut app = App::new();
    /// app.world.after(0.5, |w| { w.insert_resource(Exit); });
    /// app.frame(0.4);
    /// assert!(app.world.get_resource::<Exit>().is_none());
    /// app.frame(0.2);
    /// assert!(app.world.get_resource::<Exit>().is_some());
    /// ```
    pub fn after(&mut self, seconds: f32, f: impl FnOnce(&mut World) + 'static) {
        let at = self.resource::<Time>().elapsed + seconds.max(0.0) as f64;
        self.resource_mut::<Delays>()
            .waiting
            .push((at, Box::new(f)));
    }
}

pub(crate) fn run_delays(w: &mut World) {
    let now = w.resource::<Time>().elapsed;
    let delays = w.resource_mut::<Delays>();
    if !delays.waiting.iter().any(|(at, _)| *at <= now) {
        return;
    }
    let (due, waiting): (Vec<_>, Vec<_>) = std::mem::take(&mut delays.waiting)
        .into_iter()
        .partition(|(at, _)| *at <= now);
    delays.waiting = waiting;
    let mut due = due;
    due.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (_, f) in due {
        f(w);
    }
}

/// Glides a node's position, rotation or scale to a new value. Add it to an
/// entity; it removes itself when done and sends [`TweenFinished`].
///
/// ```
/// # use threers_ecs::prelude::*;
/// let mut app = App::new();
/// let door = app.world.spawn(Object3D::group());
/// app.world.insert(door, Tween::to_position(Vector3::new(0.0, 3.0, 0.0), 1.0).easing(Easing::QuadOut));
/// for _ in 0..70 { app.frame(1.0 / 60.0); }
/// assert_eq!(app.world.node(door).unwrap().position.y, 3.0);
/// ```
#[derive(Debug, Clone)]
pub struct Tween {
    target: Target,
    from: Option<Target>,
    timer: Timer,
    easing: Easing,
}
impl Component for Tween {}

#[derive(Debug, Clone, Copy)]
enum Target {
    Position(Vector3),
    Scale(Vector3),
    Rotation(Quaternion),
}

/// A [`Tween`] reached its end.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TweenFinished(pub Entity);

impl Tween {
    /// Move to `position` over `seconds`.
    pub fn to_position(position: Vector3, seconds: f32) -> Self {
        Self::new(Target::Position(position), seconds)
    }

    /// Scale to `scale` over `seconds`.
    pub fn to_scale(scale: Vector3, seconds: f32) -> Self {
        Self::new(Target::Scale(scale), seconds)
    }

    /// Turn to `rotation` over `seconds`.
    pub fn to_rotation(rotation: Quaternion, seconds: f32) -> Self {
        Self::new(Target::Rotation(rotation), seconds)
    }

    fn new(target: Target, seconds: f32) -> Self {
        Self {
            target,
            from: None,
            timer: Timer::once(seconds),
            easing: Easing::QuadInOut,
        }
    }

    /// The curve it follows; `QuadInOut` by default.
    pub fn easing(mut self, easing: Easing) -> Self {
        self.easing = easing;
        self
    }
}

pub(crate) fn run_tweens(w: &mut World) {
    let dt = w.resource::<Time>().delta;
    let mut done = Vec::new();
    w.each::<(Entity, Node, &mut Tween)>(|(id, node, tween)| {
        let from = *tween.from.get_or_insert(match tween.target {
            Target::Position(_) => Target::Position(node.position),
            Target::Scale(_) => Target::Scale(node.scale),
            Target::Rotation(_) => Target::Rotation(node.quaternion),
        });
        tween.timer.tick(dt);
        let t = tween.easing.apply(tween.timer.fraction().clamp(0.0, 1.0));
        match (from, tween.target) {
            (Target::Position(a), Target::Position(b)) => node.position = a.lerp(b, t),
            (Target::Scale(a), Target::Scale(b)) => node.scale = a.lerp(b, t),
            (Target::Rotation(a), Target::Rotation(b)) => node.quaternion = a.slerp(b, t),
            _ => {}
        }
        if tween.timer.finished() {
            done.push(id);
        }
    });
    for id in done {
        w.remove::<Tween>(id);
        w.send(TweenFinished(id));
    }
}
