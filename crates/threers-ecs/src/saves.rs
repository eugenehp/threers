//! Keeping saves between sessions: files natively, `localStorage` in a browser.
//!
//! ```no_run
//! use threers_ecs::prelude::*;
//! use threers_ecs::saves::SaveSlots;
//!
//! fn checkpoint(w: &mut World) {
//!     SaveSlots::new("my-game").write(w, "slot1").expect("saved");
//! }
//! ```

use crate::world::World;

/// A named set of saves for one game.
#[derive(Debug, Clone)]
pub struct SaveSlots {
    game: String,
    #[cfg(not(target_arch = "wasm32"))]
    dir: std::path::PathBuf,
}

/// A slot name becomes a file name, so it may not reach outside the directory.
fn check(slot: &str) -> Result<(), String> {
    let fine = !slot.is_empty()
        && slot
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_');
    fine.then_some(())
        .ok_or_else(|| format!("{slot:?} is not a slot name: use letters, digits, - and _"))
}

impl SaveSlots {
    /// Saves for the game called `game`, in the user's data directory natively.
    pub fn new(game: &str) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let base = std::env::var_os("XDG_DATA_HOME")
                .map(Into::into)
                .or_else(|| {
                    std::env::var_os("HOME").map(|home| {
                        let home = std::path::PathBuf::from(home);
                        if cfg!(target_os = "macos") {
                            home.join("Library/Application Support")
                        } else {
                            home.join(".local/share")
                        }
                    })
                })
                .or_else(|| std::env::var_os("APPDATA").map(Into::into))
                .unwrap_or_else(std::env::temp_dir);
            Self::in_dir(game, base.join(game).join("saves"))
        }
        #[cfg(target_arch = "wasm32")]
        Self {
            game: game.to_owned(),
        }
    }

    /// Saves in a directory of your choosing.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn in_dir(game: &str, dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            game: game.to_owned(),
            dir: dir.into(),
        }
    }

    /// The game this is for.
    pub fn game(&self) -> &str {
        &self.game
    }

    /// Save the world to `slot`, replacing what is there.
    ///
    /// # Errors
    ///
    /// If the name is not usable or storage is unavailable or full.
    pub fn write(&self, world: &World, slot: &str) -> Result<(), String> {
        check(slot)?;
        self.put(slot, &world.save())
    }

    /// Load `slot` into the world.
    ///
    /// # Errors
    ///
    /// If there is no such save, or it cannot be read. See [`World::load`].
    pub fn read(&self, world: &mut World, slot: &str) -> Result<(), String> {
        check(slot)?;
        let text = self
            .get(slot)?
            .ok_or_else(|| format!("no save called {slot}"))?;
        world.load(&text)
    }

    /// Whether a save called `slot` exists.
    pub fn exists(&self, slot: &str) -> bool {
        check(slot).is_ok() && matches!(self.get(slot), Ok(Some(_)))
    }

    /// Every slot, in name order.
    pub fn list(&self) -> Vec<String> {
        let mut slots = self.names();
        slots.sort();
        slots
    }

    /// Remove a slot. A slot that is not there is no error.
    ///
    /// # Errors
    ///
    /// If the name is not usable or storage refuses.
    pub fn delete(&self, slot: &str) -> Result<(), String> {
        check(slot)?;
        self.remove(slot)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl SaveSlots {
    fn path(&self, slot: &str) -> std::path::PathBuf {
        self.dir.join(format!("{slot}.save"))
    }

    /// Through a temporary file and a rename, so a crash half way through a write
    /// leaves the old save rather than half of a new one.
    fn put(&self, slot: &str, text: &str) -> Result<(), String> {
        let describe = |e: std::io::Error| format!("{}: {e}", self.dir.display());
        std::fs::create_dir_all(&self.dir).map_err(describe)?;
        let temporary = self.dir.join(format!("{slot}.tmp"));
        std::fs::write(&temporary, text).map_err(describe)?;
        std::fs::rename(&temporary, self.path(slot)).map_err(describe)
    }

    fn get(&self, slot: &str) -> Result<Option<String>, String> {
        match std::fs::read_to_string(self.path(slot)) {
            Ok(text) => Ok(Some(text)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.to_string()),
        }
    }

    fn names(&self) -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        entries
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().into_string().ok()?;
                Some(name.strip_suffix(".save")?.to_owned())
            })
            .collect()
    }

    fn remove(&self, slot: &str) -> Result<(), String> {
        match std::fs::remove_file(self.path(slot)) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.to_string()),
            _ => Ok(()),
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl SaveSlots {
    fn storage(&self) -> Result<web_sys::Storage, String> {
        web_sys::window()
            .and_then(|window| window.local_storage().ok().flatten())
            .ok_or_else(|| "this browser has no localStorage here".to_owned())
    }

    fn key(&self, slot: &str) -> String {
        format!("{}/save/{slot}", self.game)
    }

    fn put(&self, slot: &str, text: &str) -> Result<(), String> {
        self.storage()?
            .set_item(&self.key(slot), text)
            .map_err(|e| format!("storage refused the save: {e:?}"))
    }

    fn get(&self, slot: &str) -> Result<Option<String>, String> {
        self.storage()?
            .get_item(&self.key(slot))
            .map_err(|e| format!("{e:?}"))
    }

    fn names(&self) -> Vec<String> {
        let Ok(storage) = self.storage() else {
            return Vec::new();
        };
        let prefix = self.key("");
        (0..storage.length().unwrap_or(0))
            .filter_map(|index| storage.key(index).ok().flatten())
            .filter_map(|key| key.strip_prefix(&prefix).map(str::to_owned))
            .collect()
    }

    fn remove(&self, slot: &str) -> Result<(), String> {
        self.storage()?
            .remove_item(&self.key(slot))
            .map_err(|e| format!("{e:?}"))
    }
}
