//! Entities, components and systems over the [`threers`](https://docs.rs/threers)
//! scene graph.
//!
//! Most ECS libraries own the world and treat a renderer as something to feed.
//! This one goes the other way. A scene node's [`ObjectId`](threers::core::ObjectId)
//! is already a generational key, so it *is* the entity: components live in side
//! tables keyed by it, while position, hierarchy, meshes and lights stay on the
//! [`Object3D`](threers::core::Object3D) where the renderer reads them. Nothing is
//! mirrored, so nothing can fall out of sync, and a node from any loader can take
//! components the moment it exists.
//!
//! ```
//! use threers::prelude::*;
//! use threers_ecs::prelude::*;
//!
//! struct Velocity(Vector3);
//! impl Component for Velocity {}
//!
//! fn movement(w: &mut World) {
//!     let dt = w.resource::<Time>().delta;
//!     w.each::<(Node, &Velocity)>(|(node, velocity)| {
//!         node.position = node.position + velocity.0 * dt;
//!     });
//! }
//!
//! let mut app = App::new();
//! app.fixed_system(movement);
//!
//! let ball = app.world.spawn(Object3D::group());
//! app.world.insert(ball, Velocity(Vector3::new(1.0, 0.0, 0.0)));
//!
//! app.frame(1.0 / 60.0);
//! assert!(app.world.scene.get(ball).unwrap().position.x > 0.0);
//! ```
//!
//! # Which tool for which job
//!
//! | Need | Use |
//! |---|---|
//! | Run code over everything with some components | [`World::each`] |
//! | Spawn or despawn from inside a query | [`Commands`] |
//! | One value for the whole game | [`World::insert_resource`] |
//! | Use a resource and a query together | [`World::with_resource`] |
//! | Say which system runs first | [`System::after`], [`System::before`] |
//! | Step gameplay at a steady rate | [`App::fixed_system`] |
//! | Clean up when a component leaves | [`World::on_remove`] |
//! | Tell other systems something happened | [`World::send`], [`World::events`] |
//! | Menu, playing, game over | [`App::state`], [`System::in_state`] |
//! | Pause, slow motion, single-step | [`Time`] |
//! | React to components appearing or changing | [`Added`], [`Changed`] |
//! | Load files, levels and hot reload | [`assets`] |
//! | Play sounds | `audio` module (`audio` feature) |
//! | Save the game, stamp out copies, inspect fields | [`registry`] |
//! | Choose which animation plays | [`animator`] |
//! | Text and UI over the game | [`hud`] |
//! | Play a game from a test | [`testing`] |
//! | Look inside a running game | [`dev`] |
//! | Draw lines to see what code thinks | [`gizmos`] |
//! | Countdowns, delays, things that glide | [`timer`] |
//! | Tune numbers without rebuilding | [`tuning`] |
//! | Turn a play session into a test | [`replay`] |
//! | Keep playing across a rebuild | `hot` module, with `scripts/watch-game.py` |
//! | Read the player by action | [`Input`] |
//! | Play on a phone | [`touch`] |
//! | Keep saves between sessions | [`saves`] |
//! | Work that takes longer than a frame | [`tasks`] |
//! | Rigid bodies, characters, collisions | `physics` module (`physics` feature) |
//! | Drive a car | `car` module (`physics` feature) |
//! | Follow something with the camera | [`camera`] |
//! | Spawn and set up a node in one go | [`World::mesh`] |
//! | Open a window | `window` module (`window` feature) |
//! | Run in a browser | `web` module (`web` feature) |
//! | Read real gamepads | `gamepad` module (`gamepad` feature) |
//!
//! # Removing things
//!
//! Remove entities with [`World::despawn`], which takes the subtree and its
//! components with it. A side table cannot see
//! [`ObjectArena::delete`](threers::core::ObjectArena::delete): queries skip what
//! that leaves behind, but the rows stay until [`World::prune`].
//!
//! # World matrices
//!
//! Systems see this frame's `position` but the `matrix_world` the renderer last
//! computed. Call [`Scene::update_world`](threers::Scene::update_world) first if a
//! system needs a world position that is exact.

// `Send` checks on a parsed scene walk deep through the renderer's types.
#![recursion_limit = "512"]
#![warn(missing_docs)]

pub mod animator;
mod app;
pub mod assets;
#[cfg(feature = "audio")]
pub mod audio;
pub mod batch;
pub mod camera;
#[cfg(feature = "physics")]
pub mod car;
mod clock;
mod commands;
pub mod console;
pub mod dev;
mod events;
#[cfg(feature = "gamepad")]
pub mod gamepad;
pub mod gizmos;
#[cfg(not(target_arch = "wasm32"))]
pub mod hot;
pub mod hud;
mod input;
pub mod params;
#[cfg(feature = "physics")]
// Its own `World` and `Easing` hide the physics prelude's, on purpose.
#[allow(hidden_glob_reexports)]
pub mod physics;
pub mod profile;
mod query;
pub mod registry;
pub mod replay;
pub mod saves;
mod schedule;
mod spawn;
mod states;
mod store;
pub mod tasks;
pub mod testing;
pub mod timer;
pub mod touch;
pub mod tuning;
#[cfg(feature = "video")]
pub mod video;
#[cfg(all(feature = "web", target_arch = "wasm32"))]
pub mod web;
#[cfg(all(feature = "window", not(target_arch = "wasm32")))]
pub mod window;
mod world;

pub use app::{App, Exit, MainCamera, Plugin, Time};
pub use commands::Commands;
pub use events::EVENT_FRAMES;
pub use input::{
    Button, Input, Key, MouseButton, PadAxis, PadButton, PadEvent, PadTracker, Rumble, Touch,
};
pub use query::{
    Added, Changed, Node, NodeRef, Parallel, QueryIter, ReadOnly, Term, With, Without,
};
pub use registry::{Field, FieldValue, Fields, Prefab, Reflect};
pub use schedule::{Schedule, System};
pub use spawn::Spawned;
pub use states::GameState;
pub use world::{Bundle, Component, Entity, World};

/// Everything a game names often, in one import: `use threers_ecs::prelude::*;`.
///
/// That includes `threers`'s prelude (maths, geometry, materials, lights,
/// `MeshBuilder`) and, with their features, physics, cars, sound and the window.
/// The physics solver's own `World` is `Solver` here, so `World` is always the
/// game's. Rarer things stay in their modules: `assets`, `saves`, `tasks`, `dev`.
pub mod prelude {
    pub use threers::prelude::*;

    pub use crate::batch::Static;
    pub use crate::camera::{CameraTarget, ChaseCamera};
    pub use crate::gizmos::Gizmos;
    pub use crate::hud::{with_hud, Hud};
    pub use crate::params::{
        EventReader, EventWriter, IntoParallel, IntoSystem, Query, Res, ResMut,
    };
    pub use crate::reflect;
    pub use crate::replay::Recording;
    pub use crate::testing::TestGame;
    pub use crate::timer::{Timer, Tween, TweenFinished};
    pub use crate::touch::{TouchButton, TouchControls};
    pub use crate::tuning::Retuned;
    pub use crate::{
        Added, App, Bundle, Button, Changed, Commands, Component, Entity, Exit, GameState, Input,
        Key, MainCamera, MouseButton, Node, NodeRef, PadAxis, PadButton, PadEvent, PadTracker,
        Plugin, QueryIter, Schedule, Spawned, System, Time, With, Without, World, EVENT_FRAMES,
    };

    #[cfg(feature = "audio")]
    pub use crate::audio::{
        tone, waveform, AudioOut, AudioPlugin, Bus, SoundAsset, SoundFinished, Speaker, Speakers,
    };
    #[cfg(feature = "physics")]
    pub use crate::car::{car_step, AttachCar, Car, CarSpec, Driven};
    #[cfg(feature = "physics")]
    pub use crate::physics::*;
    #[cfg(feature = "video")]
    pub use crate::video::{Video, VideoPlayer, VideoTexture};
    #[cfg(all(feature = "window", not(target_arch = "wasm32")))]
    pub use crate::window::{run, WindowOptions};
}
