//! Keep the game where it was across a rebuild.
//!
//! [`HotReload`] writes the world's saved state (see [`registry`](crate::registry))
//! to a file twice a second, and when the game starts under the watcher it reads
//! it back. `scripts/watch-game.py` is the watcher: it rebuilds when a source
//! file changes, and only once the build has worked does it stop the old game
//! and start the new one, which picks up where the old one was.
//!
//! ```text
//! python3 scripts/watch-game.py -p threers-ecs --example race --features physics,window,audio
//! ```
//!
//! Only what is registered for saves carries over, so register what a session
//! should keep: the player's progress, a level's state. Anything else starts as
//! the game builds it.
//!
//! ```no_run
//! # use threers_ecs::prelude::*;
//! # use threers_ecs::hot::HotReload;
//! let mut app = App::new();
//! // build the game, then:
//! app.plugin(HotReload::new("target/hot/race.json"));
//! ```

use crate::app::{App, Plugin, Time};
use crate::world::World;
use std::path::PathBuf;

/// The environment variable the watcher sets on the games it starts.
pub const RESUME_VAR: &str = "THREERS_HOT_RESUME";

/// Saves for the next build, and resumes from the last.
pub struct HotReload {
    path: PathBuf,
    resume: Option<bool>,
    every: f64,
}

/// Where the state goes, and when it last went.
pub struct HotState {
    /// The file the state is saved to and resumed from.
    pub path: PathBuf,
    every: f64,
    last: f64,
    last_text: String,
    /// Whether this session resumed from a file, and any problem doing so.
    pub resumed: Result<bool, String>,
}

impl HotReload {
    /// Save the registered state to `path` twice a second; a game started by the
    /// watcher resumes from it.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            resume: None,
            every: 0.5,
        }
    }

    /// Whether to read the file at start; by default, when the watcher
    /// started this game (it sets [`RESUME_VAR`]).
    pub fn resume(mut self, resume: bool) -> Self {
        self.resume = Some(resume);
        self
    }

    /// How often to save, in seconds of real time.
    pub fn every(mut self, seconds: f64) -> Self {
        self.every = seconds.max(0.05);
        self
    }
}

impl Plugin for HotReload {
    fn build(self, app: &mut App) {
        let resume = self
            .resume
            .unwrap_or_else(|| std::env::var_os(RESUME_VAR).is_some());
        let resumed = if resume {
            load(&mut app.world, &self.path)
        } else {
            Ok(false)
        };
        if let Err(problem) = &resumed {
            eprintln!(
                "hot reload: could not resume from {}: {problem}",
                self.path.display()
            );
        }
        app.world.insert_resource(HotState {
            path: self.path,
            every: self.every,
            last: f64::MIN,
            last_text: String::new(),
            resumed,
        });
        app.system(save_now_and_then);
    }
}

fn load(w: &mut World, path: &std::path::Path) -> Result<bool, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => w.load(&text).map(|()| true),
        Err(_) => Ok(false),
    }
}

/// Write the state if it changed and the time has come.
pub fn save_now_and_then(w: &mut World) {
    let now = w.resource::<Time>().real_elapsed;
    let state = w.resource::<HotState>();
    if now - state.last < state.every {
        return;
    }
    let text = w.save();
    let state = w.resource_mut::<HotState>();
    state.last = now;
    if text == state.last_text {
        return;
    }
    if let Some(dir) = state.path.parent().filter(|d| !d.as_os_str().is_empty()) {
        let _ = std::fs::create_dir_all(dir);
    }
    // Written beside and renamed, so a game stopped mid-write leaves the last
    // whole file rather than half of a new one.
    let partial = state.path.with_extension("partial");
    if std::fs::write(&partial, &text)
        .and_then(|()| std::fs::rename(&partial, &state.path))
        .is_ok()
    {
        state.last_text = text;
    }
}
