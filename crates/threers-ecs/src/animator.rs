//! Which animation an entity is playing, and when it changes.
//!
//! An [`Animator`] is a small state machine over clips. The game sets parameters
//! such as a speed and fires triggers; transitions decide the rest, and crossfade
//! between states.
//!
//! ```no_run
//! # use threers_ecs::prelude::*;
//! # use threers_ecs::animator::*;
//! # use threers::animation::AnimationClip;
//! # fn clips() -> (AnimationClip, AnimationClip, AnimationClip) { unimplemented!() }
//! let (idle, run, jump) = clips();
//! let mut animator = Animator::new();
//! animator
//!     .state("idle", idle)
//!     .state("run", run)
//!     .once("jump", jump)
//!     .transition("idle", "run", Condition::above("speed", 0.1), 0.2)
//!     .transition("run", "idle", Condition::below("speed", 0.1), 0.2)
//!     .from_any("jump", Condition::trigger("jump"), 0.1)
//!     .transition("jump", "idle", Condition::Finished, 0.2);
//! ```

use crate::app::Time;
use crate::query::With;
use crate::world::{Component, Entity, World};
use std::collections::{HashMap, HashSet};
use threers::animation::{AnimationClip, AnimationMixer, LoopMode};

/// When a transition fires.
#[derive(Debug, Clone, PartialEq)]
pub enum Condition {
    /// A float parameter, set with [`Animator::set`], went above this value.
    Above(String, f32),
    /// A float parameter went below this value.
    Below(String, f32),
    /// Fired once with [`Animator::trigger`].
    Trigger(String),
    /// The state's clip has played through. For states added with
    /// [`once`](Animator::once).
    Finished,
}

impl Condition {
    /// [`Condition::Above`], without the `to_owned`.
    pub fn above(parameter: &str, value: f32) -> Self {
        Condition::Above(parameter.to_owned(), value)
    }

    /// [`Condition::Below`], without the `to_owned`.
    pub fn below(parameter: &str, value: f32) -> Self {
        Condition::Below(parameter.to_owned(), value)
    }

    /// [`Condition::Trigger`], without the `to_owned`.
    pub fn trigger(name: &str) -> Self {
        Condition::Trigger(name.to_owned())
    }
}

/// A one-shot state's clip reached its end.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimationFinished {
    /// The entity whose [`Animator`] it is.
    pub entity: Entity,
    /// The state whose clip finished.
    pub state: String,
}

struct AnimState {
    name: String,
    action: usize,
    duration: f32,
    looping: bool,
}

struct Transition {
    from: Option<usize>,
    to: usize,
    condition: Condition,
    fade: f32,
}

/// An entity's animation state machine.
pub struct Animator {
    mixer: Option<AnimationMixer>,
    states: Vec<AnimState>,
    transitions: Vec<Transition>,
    parameters: HashMap<String, f32>,
    triggers: HashSet<String>,
    current: Option<usize>,
    /// Seconds in the current state.
    time: f32,
    finished: bool,
}

impl Default for Animator {
    fn default() -> Self {
        Self::new()
    }
}

impl Component for Animator {}

impl Animator {
    /// An animator with no states. It starts in the first one added.
    pub fn new() -> Self {
        Self {
            mixer: Some(AnimationMixer::new()),
            states: Vec::new(),
            transitions: Vec::new(),
            parameters: HashMap::new(),
            triggers: HashSet::new(),
            current: None,
            time: 0.0,
            finished: false,
        }
    }

    fn add(&mut self, name: &str, clip: AnimationClip, looping: bool) -> &mut Self {
        let duration = clip.duration;
        let mixer = self
            .mixer
            .as_mut()
            .expect("the mixer is only out during a step");
        let action = mixer.clip_action(clip);
        // Actions start enabled, and enabled ones all play and blend together.
        mixer.actions[action].stop();
        self.states.push(AnimState {
            name: name.to_owned(),
            action,
            duration,
            looping,
        });
        self
    }

    /// Add a state that loops. The first state added is where it starts.
    pub fn state(&mut self, name: &str, clip: AnimationClip) -> &mut Self {
        self.add(name, clip, true)
    }

    /// Add a state that plays through once and holds its last pose.
    pub fn once(&mut self, name: &str, clip: AnimationClip) -> &mut Self {
        self.add(name, clip, false)
    }

    fn index(&self, name: &str) -> usize {
        self.states
            .iter()
            .position(|state| state.name == name)
            .unwrap_or_else(|| panic!("no animation state named {name:?}"))
    }

    /// Move from one state to another when `condition` holds, crossfading over
    /// `fade` seconds. Transitions are tried in the order added.
    ///
    /// # Panics
    ///
    /// If either state has not been added.
    pub fn transition(
        &mut self,
        from: &str,
        to: &str,
        condition: Condition,
        fade: f32,
    ) -> &mut Self {
        let (from, to) = (self.index(from), self.index(to));
        self.transitions.push(Transition {
            from: Some(from),
            to,
            condition,
            fade,
        });
        self
    }

    /// A transition from every other state.
    ///
    /// # Panics
    ///
    /// If the state has not been added.
    pub fn from_any(&mut self, to: &str, condition: Condition, fade: f32) -> &mut Self {
        let to = self.index(to);
        self.transitions.push(Transition {
            from: None,
            to,
            condition,
            fade,
        });
        self
    }

    /// Set a parameter that conditions can compare against.
    pub fn set(&mut self, parameter: &str, value: f32) {
        self.parameters.insert(parameter.to_owned(), value);
    }

    /// Fire a trigger. It is consumed by the first transition it causes, and
    /// dropped at the end of the step if none does.
    pub fn trigger(&mut self, name: &str) {
        self.triggers.insert(name.to_owned());
    }

    /// The state playing now.
    pub fn current(&self) -> Option<&str> {
        self.current.map(|index| self.states[index].name.as_str())
    }

    fn holds(&self, condition: &Condition) -> bool {
        let parameter = |name: &String| self.parameters.get(name).copied().unwrap_or(0.0);
        match condition {
            Condition::Above(name, value) => parameter(name) > *value,
            Condition::Below(name, value) => parameter(name) < *value,
            Condition::Trigger(name) => self.triggers.contains(name),
            Condition::Finished => self.finished,
        }
    }

    fn next_transition(&self) -> Option<usize> {
        let current = self.current?;
        self.transitions.iter().position(|transition| {
            transition.to != current
                && transition.from.is_none_or(|from| from == current)
                && self.holds(&transition.condition)
        })
    }

    fn enter(&mut self, state: usize, from: Option<usize>, fade: f32) {
        let mixer = self
            .mixer
            .as_mut()
            .expect("the mixer is only out during a step");
        let (to, looping) = (self.states[state].action, self.states[state].looping);
        let action = &mut mixer.actions[to];
        action.stop();
        action.loop_mode = if looping {
            LoopMode::Repeat
        } else {
            LoopMode::Once
        };
        match from {
            Some(from) => mixer.cross_fade(self.states[from].action, to, fade),
            None => action.play(),
        }
        self.current = Some(state);
        self.time = 0.0;
        self.finished = false;
    }
}

/// Advance every [`Animator`]. Runs once a frame, at game speed, so a pause
/// freezes animation too.
pub fn animator_step(w: &mut World) {
    let delta = w.resource::<Time>().delta;
    let mut finished = Vec::new();
    for id in w.entities::<With<Animator>>() {
        let Some(animator) = w.get_mut::<Animator>(id) else {
            continue;
        };
        if animator.current.is_none() && !animator.states.is_empty() {
            animator.enter(0, None, 0.0);
        }
        if let Some(index) = animator.next_transition() {
            let (to, fade, condition) = {
                let t = &animator.transitions[index];
                (t.to, t.fade, t.condition.clone())
            };
            let from = animator.current;
            animator.enter(to, from, fade);
            if let Condition::Trigger(name) = condition {
                animator.triggers.remove(&name);
            }
        }
        animator.triggers.clear();

        animator.time += delta;
        if let Some(state) = animator.current.map(|index| &animator.states[index])
            && !state.looping
            && !animator.finished
            && animator.time >= state.duration
        {
            animator.finished = true;
            finished.push(AnimationFinished {
                entity: id,
                state: state.name.clone(),
            });
        }

        // Out while the mixer has the scene: the component lives in the world too.
        let Some(mut mixer) = animator.mixer.take() else {
            continue;
        };
        mixer.update(&mut w.scene, delta);
        if let Some(animator) = w.get_mut::<Animator>(id) {
            animator.mixer = Some(mixer);
        }
    }
    for event in finished {
        w.send(event);
    }
}
