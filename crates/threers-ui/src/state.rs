//! State that outlives the nodes showing it.
//!
//! A UI rebuilt from data each frame gets new nodes each frame, so nothing
//! stored on a node survives: not a panel's scroll position, not whether a
//! section is open, not a text field's caret. [`ElementState`] keeps such
//! values under keys the app chooses (a path of names and indices, say),
//! frame to frame, and lets go of a value when a frame passes without
//! anything asking for it — the element is gone.
//!
//! ```
//! use threers_ui::state::ElementState;
//!
//! #[derive(Default)]
//! struct Section { open: bool }
//!
//! let mut state = ElementState::new();
//! state.get_or_default::<Section>(("sidebar", 2)).open = true;
//! state.finish_frame();
//! assert!(state.get_or_default::<Section>(("sidebar", 2)).open);
//! state.finish_frame();
//! state.finish_frame(); // a frame without it
//! assert!(!state.get_or_default::<Section>(("sidebar", 2)).open);
//! ```

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};

/// Values of any type, by key and type. See the [module docs](self).
#[derive(Default)]
pub struct ElementState {
    values: HashMap<(TypeId, u64), (Box<dyn Any>, u64)>,
    frame: u64,
    hasher: std::collections::hash_map::RandomState,
}

impl std::fmt::Debug for ElementState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ElementState({} values, frame {})", self.values.len(), self.frame)
    }
}

impl ElementState {
    pub fn new() -> Self {
        Self::default()
    }

    fn slot<T: 'static>(&self, key: impl Hash) -> (TypeId, u64) {
        (TypeId::of::<T>(), self.hasher.hash_one(key))
    }

    /// The `T` under `key`, made with `make` the first time (or the first
    /// time after it was let go). Marks it used this frame.
    pub fn get_or_insert_with<T: 'static>(&mut self, key: impl Hash, make: impl FnOnce() -> T) -> &mut T {
        let slot = self.slot::<T>(key);
        let frame = self.frame;
        let entry = self
            .values
            .entry(slot)
            .or_insert_with(|| (Box::new(make()), frame));
        entry.1 = frame;
        entry.0.downcast_mut::<T>().expect("keyed by its type")
    }

    pub fn get_or_default<T: Default + 'static>(&mut self, key: impl Hash) -> &mut T {
        self.get_or_insert_with(key, T::default)
    }

    /// The `T` under `key`, if there is one; does not mark it used.
    pub fn get<T: 'static>(&self, key: impl Hash) -> Option<&T> {
        self.values.get(&self.slot::<T>(key))?.0.downcast_ref()
    }

    pub fn remove<T: 'static>(&mut self, key: impl Hash) -> Option<T> {
        let slot = self.slot::<T>(key);
        self.values.remove(&slot)?.0.downcast().ok().map(|b| *b)
    }

    /// Ends a frame: values not asked for during it are let go.
    pub fn finish_frame(&mut self) {
        let frame = self.frame;
        self.values.retain(|_, (_, used)| *used == frame);
        self.frame += 1;
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}
