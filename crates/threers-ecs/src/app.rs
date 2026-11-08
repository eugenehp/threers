//! The frame: fixed steps as many times as are due, then one variable update.
//!
//! Gameplay and physics want a constant step, or a jump's height depends on the
//! frame rate; drawing wants to run as often as the display does. So time arrives
//! here in whatever sizes the host has, and is paid out to fixed systems in equal
//! parts. What is left over becomes [`Time::alpha`], for interpolating the picture
//! between the last two steps.
//!
//! `App` opens no window, so the same game runs under a test, a browser or a
//! native event loop: each calls [`App::frame`] with the time that has passed.

use crate::hud::Hud;
use crate::input::Input;
use crate::schedule::Schedule;
use crate::states::{AnyDriver, Driver, GameState, State};
use crate::world::World;
use std::any::TypeId;
use std::collections::HashMap;
use threers::cameras::PerspectiveCamera;
use threers::Scene;

/// The clock, as the resource every system can read.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Time {
    /// Seconds this step covers: the fixed step inside a fixed system, the
    /// frame's duration inside a variable one. Movement code is the same in both.
    pub delta: f32,
    /// Total simulated seconds. Fixed systems see the time of the current step.
    pub elapsed: f64,
    /// How far through the next fixed step this frame is, from 0 up to 1.
    pub alpha: f32,
    /// Frames completed.
    pub frame: u64,
    /// Fixed steps completed.
    pub fixed_steps: u64,
    /// Seconds that have really passed, whatever `paused` and `scale` say.
    pub real_elapsed: f64,
    /// Seconds that really passed this frame, whatever `paused` and `scale` say.
    /// For things that keep moving in a pause, such as a menu.
    pub real_delta: f32,
    /// Stop game time: no fixed steps run, and `delta` is zero. Variable systems
    /// still run, so a pause menu works.
    pub paused: bool,
    /// Game time per real second. `0.5` is slow motion.
    pub scale: f64,
    /// While paused, fixed steps to run anyway, one per frame. For stepping
    /// through a bug a step at a time.
    pub step_once: u32,
}

impl Default for Time {
    fn default() -> Self {
        Self {
            delta: 0.0,
            elapsed: 0.0,
            alpha: 0.0,
            frame: 0,
            fixed_steps: 0,
            real_elapsed: 0.0,
            real_delta: 0.0,
            paused: false,
            scale: 1.0,
            step_once: 0,
        }
    }
}

/// The camera the window draws through, as a resource. A window opens with a
/// default one unless the game inserted its own; the aspect ratio is kept up to date.
pub struct MainCamera(pub PerspectiveCamera);

/// Insert this as a resource to ask the host to stop.
#[derive(Debug, Clone, Copy, Default)]
pub struct Exit;

/// A feature that sets itself up on an [`App`]: resources, systems and hooks.
pub trait Plugin {
    /// Add this feature's resources, systems and hooks to `app`.
    fn build(self, app: &mut App);
}

/// A world and the two schedules that run over it.
pub struct App {
    /// Everything the game holds. Systems get it; so can code outside them.
    pub world: World,
    fixed: Schedule,
    update: Schedule,
    fixed_delta: f64,
    max_fixed_steps: u32,
    accumulator: f64,
    frame_elapsed: f64,
    states: HashMap<TypeId, Box<dyn AnyDriver>>,
}

impl App {
    /// An app with an empty scene, [`Time`](crate::app::Time), [`Input`](crate::input::Input) and the other resources every game uses.
    pub fn new() -> Self {
        Self::with_scene(Scene::new())
    }

    /// As [`new`](Self::new), over a scene that already has content.
    pub fn with_scene(scene: Scene) -> Self {
        let mut world = World::with_scene(scene);
        world.insert_resource(Time::default());
        world.insert_resource(Input::default());
        world.insert_resource(crate::gizmos::Gizmos::default());
        world.insert_resource(crate::timer::Delays::default());
        world.insert_resource(crate::tuning::TuneFiles::default());
        world.insert_resource(crate::replay::Recorder::default());
        world.insert_resource(crate::profile::Profile::default());
        world.insert_resource(crate::profile::Failures::default());
        Self {
            world,
            fixed: Schedule::new(),
            update: Schedule::new(),
            fixed_delta: 1.0 / 60.0,
            max_fixed_steps: 8,
            accumulator: 0.0,
            frame_elapsed: 0.0,
            states: HashMap::new(),
        }
    }

    /// Catch a system's panic, switch that system off and carry on, reporting it
    /// in [`Failures`](crate::profile::Failures) and a
    /// [`SystemFailed`](crate::profile::SystemFailed) event. For development:
    /// a system half done may have left the world half changed. Needs panics
    /// that unwind, so does nothing where they abort, as in a wasm release.
    pub fn catch_panics(&mut self, on: bool) -> &mut Self {
        self.world.catch_panics = on;
        self
    }

    /// Take another App's clocks: its game time and the fixed step it is part
    /// way through.
    pub(crate) fn take_clock_from(&mut self, other: &App) {
        self.frame_elapsed = other.frame_elapsed;
        self.accumulator = other.accumulator;
    }

    /// Fixed steps per second; 60 by default.
    ///
    /// # Panics
    ///
    /// If `hz` is not a positive number.
    pub fn fixed_hz(&mut self, hz: f64) -> &mut Self {
        assert!(hz.is_finite() && hz > 0.0, "fixed rate must be positive");
        self.fixed_delta = 1.0 / hz;
        self
    }

    /// The most fixed steps one frame may run; 8 by default. After a long stall,
    /// such as a hidden browser tab, catching up in full would take longer than
    /// the time it recovers, and each frame would fall further behind. Time past
    /// this limit is dropped, so the game slows down instead.
    pub fn max_fixed_steps(&mut self, steps: u32) -> &mut Self {
        self.max_fixed_steps = steps;
        self
    }

    /// Add a system that runs at the fixed rate, for gameplay and physics.
    pub fn fixed_system<M>(&mut self, system: impl crate::params::IntoSystem<M>) -> &mut Self {
        self.fixed.add(system);
        self
    }

    /// Add a system that runs once a frame, for cameras and anything visual.
    pub fn system<M>(&mut self, system: impl crate::params::IntoSystem<M>) -> &mut Self {
        self.update.add(system);
        self
    }

    /// Add a game state, starting at `initial`. Change it with
    /// [`World::set_state`], and gate systems with
    /// [`System::in_state`](crate::System::in_state).
    pub fn state<S: GameState>(&mut self, initial: S) -> &mut Self {
        self.world.insert_resource(State::new(initial));
        self.states
            .insert(TypeId::of::<S>(), Box::new(Driver::<S>::default()));
        self
    }

    /// Run `system` each time `state` is entered, including at the start.
    ///
    /// # Panics
    ///
    /// If [`state`](Self::state) has not added this type.
    pub fn on_enter<S: GameState, M>(
        &mut self,
        state: S,
        system: impl crate::params::IntoSystem<M>,
    ) -> &mut Self {
        self.driver::<S>().enter(state).add(system);
        self
    }

    /// Run `system` each time `state` is left.
    ///
    /// # Panics
    ///
    /// If [`state`](Self::state) has not added this type.
    pub fn on_exit<S: GameState, M>(
        &mut self,
        state: S,
        system: impl crate::params::IntoSystem<M>,
    ) -> &mut Self {
        self.driver::<S>().exit(state).add(system);
        self
    }

    fn driver<S: GameState>(&mut self) -> &mut Driver<S> {
        self.states
            .get_mut(&TypeId::of::<S>())
            .and_then(|driver| driver.as_any_mut().downcast_mut())
            .unwrap_or_else(|| panic!("no state {}: call App::state", std::any::type_name::<S>()))
    }

    /// Set up a [`Plugin`].
    pub fn plugin(&mut self, plugin: impl Plugin) -> &mut Self {
        plugin.build(self);
        self
    }

    /// Insert a resource, replacing any of its type: [`World::insert_resource`] that chains.
    pub fn resource<R: 'static>(&mut self, resource: R) -> &mut Self {
        self.world.insert_resource(resource);
        self
    }

    /// Advance by `delta` seconds, which should be how much time has really
    /// passed. Nothing is rendered: the caller draws `world.scene` afterwards.
    /// A negative or non-finite `delta` counts as zero, since one bad timestamp
    /// should not poison the accumulator for the rest of the session.
    pub fn frame(&mut self, delta: f64) {
        self.record_frame(delta);
        crate::tuning::watch_tuning(&mut self.world);
        if let Some(hud) = self.world.get_resource_mut::<Hud>() {
            hud.0.clear();
        }
        let real_delta = if delta.is_finite() {
            delta.max(0.0)
        } else {
            0.0
        };
        self.world.flush();
        for driver in self.states.values_mut() {
            driver.apply(&mut self.world);
        }

        let clock = *self.world.resource::<Time>();
        let delta = if clock.paused {
            0.0
        } else {
            real_delta * clock.scale.max(0.0)
        };
        self.accumulator += delta;
        if clock.paused && clock.step_once > 0 {
            self.world.resource_mut::<Time>().step_once -= 1;
            self.accumulator = self.accumulator.max(self.fixed_delta);
        }

        let mut steps = 0;
        self.world.resource_mut::<Input>().enter_fixed();
        while self.accumulator >= self.fixed_delta && steps < self.max_fixed_steps {
            self.accumulator -= self.fixed_delta;
            steps += 1;
            let fixed_delta = self.fixed_delta;
            let time = self.world.resource_mut::<Time>();
            time.delta = fixed_delta as f32;
            time.elapsed += fixed_delta;
            time.fixed_steps += 1;
            self.world
                .resource_mut::<crate::gizmos::Gizmos>()
                .enter_fixed_step();
            self.fixed.run(&mut self.world);
            self.world
                .resource_mut::<crate::gizmos::Gizmos>()
                .leave_fixed();
            // Only the first step of a frame sees a press.
            self.world.resource_mut::<Input>().consume_fixed();
        }
        // A key tapped during a pause is not waiting for the game to resume.
        if clock.paused {
            self.world.resource_mut::<Input>().consume_fixed();
        }
        self.world.resource_mut::<Input>().leave_fixed();
        if self.accumulator >= self.fixed_delta {
            self.accumulator = 0.0;
        }

        self.frame_elapsed += delta;
        let (elapsed, alpha) = (self.frame_elapsed, self.accumulator / self.fixed_delta);
        let time = self.world.resource_mut::<Time>();
        time.delta = delta as f32;
        time.real_delta = real_delta as f32;
        time.real_elapsed += real_delta;
        time.elapsed = elapsed;
        time.alpha = alpha as f32;
        crate::timer::run_delays(&mut self.world);
        crate::timer::run_tweens(&mut self.world);
        #[cfg(feature = "video")]
        crate::video::run_videos(&mut self.world);
        self.world.flush();
        self.update.run(&mut self.world);
        crate::gizmos::draw_gizmos(&mut self.world);
        self.world.resource_mut::<Time>().frame += 1;
        self.world.resource_mut::<Input>().end_frame();
        self.world.end_frame();
        self.rewind_if_asked();
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
