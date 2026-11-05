//! Menu, playing, game over.
//!
//! A change requested during a frame takes effect at the start of the next one,
//! so every system in a frame agrees on what state the game is in.

use crate::schedule::Schedule;
use crate::world::World;
use std::any::{type_name, Any};
use std::collections::HashMap;
use std::hash::Hash;

/// An enum of the states a game can be in.
///
/// ```
/// # use threers_ecs::prelude::*;
/// #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
/// enum Screen { Menu, Playing }
/// impl GameState for Screen {}
///
/// fn start_game(w: &mut World) { w.set_state(Screen::Playing); }
/// fn gameplay(w: &mut World) {}
///
/// let mut app = App::new();
/// app.state(Screen::Menu);
/// app.fixed_system(System::new(gameplay).in_state(Screen::Playing));
/// ```
pub trait GameState: Copy + Eq + Hash + 'static {}

pub(crate) struct State<S> {
    current: S,
    next: Option<S>,
}

impl<S: GameState> State<S> {
    pub(crate) fn new(initial: S) -> Self {
        Self {
            current: initial,
            next: None,
        }
    }
}

impl World {
    /// The current state.
    ///
    /// # Panics
    ///
    /// If [`App::state`](crate::App::state) has not added this type.
    #[track_caller]
    pub fn state<S: GameState>(&self) -> S {
        match self.get_resource::<State<S>>() {
            Some(state) => state.current,
            None => panic!(
                "no state {}: call App::state(initial) while building",
                type_name::<S>()
            ),
        }
    }

    /// Ask to change state at the start of the next frame. Asking twice in a
    /// frame keeps the later request.
    ///
    /// # Panics
    ///
    /// If [`App::state`](crate::App::state) has not added this type.
    pub fn set_state<S: GameState>(&mut self, state: S) {
        self.get_resource_mut::<State<S>>()
            .unwrap_or_else(|| panic!("no state {}: call App::state", type_name::<S>()))
            .next = Some(state);
    }
}

pub(crate) trait AnyDriver: Any {
    fn apply(&mut self, world: &mut World);
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// The systems tied to one state type.
pub(crate) struct Driver<S> {
    enter: HashMap<S, Schedule>,
    exit: HashMap<S, Schedule>,
    started: bool,
}

impl<S> Default for Driver<S> {
    fn default() -> Self {
        Self {
            enter: HashMap::new(),
            exit: HashMap::new(),
            started: false,
        }
    }
}

impl<S: GameState> Driver<S> {
    pub(crate) fn enter(&mut self, state: S) -> &mut Schedule {
        self.enter.entry(state).or_default()
    }

    pub(crate) fn exit(&mut self, state: S) -> &mut Schedule {
        self.exit.entry(state).or_default()
    }
}

impl<S: GameState> AnyDriver for Driver<S> {
    fn apply(&mut self, world: &mut World) {
        if !self.started {
            self.started = true;
            let current = world.state::<S>();
            if let Some(schedule) = self.enter.get_mut(&current) {
                schedule.run(world);
            }
        }
        // An enter system may ask for another change, which is taken here too.
        while let Some(next) = world.resource_mut::<State<S>>().next.take() {
            let current = world.state::<S>();
            if next == current {
                continue;
            }
            if let Some(schedule) = self.exit.get_mut(&current) {
                schedule.run(world);
            }
            world.resource_mut::<State<S>>().current = next;
            if let Some(schedule) = self.enter.get_mut(&next) {
                schedule.run(world);
            }
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
