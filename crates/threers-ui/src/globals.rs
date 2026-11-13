//! State the whole app shares, one value per type: settings, the
//! signed-in user, a theme. Each is kept as a [`Model`], so anything can
//! observe it.
//!
//! ```
//! use threers_ui::Ui;
//!
//! #[derive(Debug, PartialEq)]
//! struct Settings { font_size: f32 }
//!
//! let mut ui = Ui::new();
//! ui.set_global(Settings { font_size: 13.0 });
//! let settings = ui.global::<Settings>().unwrap();
//! let watching = settings.observe(|s| println!("now {}", s.font_size));
//! ui.update_global(|s: &mut Settings| s.font_size = 15.0);
//! assert_eq!(settings.read().font_size, 15.0);
//! drop(watching);
//! ```

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::rc::Rc;

use crate::model::Model;
use crate::tree::Ui;

/// The globals of a [`Ui`]: a model for each type set.
#[derive(Clone, Default)]
pub(crate) struct Globals(HashMap<TypeId, Rc<dyn Any>>);

impl std::fmt::Debug for Globals {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Globals({})", self.0.len())
    }
}

impl Ui {
    /// Sets the app-wide value of type `T` (telling its observers, if it
    /// had one).
    pub fn set_global<T: 'static>(&mut self, value: T) {
        match self.global::<T>() {
            Some(m) => m.set(value),
            None => {
                self.globals.0.insert(TypeId::of::<T>(), Rc::new(Model::new(value)));
            }
        }
    }

    /// The app-wide value of type `T`, as a model to read, update and
    /// observe.
    pub fn global<T: 'static>(&self) -> Option<Model<T>> {
        self.globals.0.get(&TypeId::of::<T>())?.downcast_ref::<Model<T>>().cloned()
    }

    /// Changes the app-wide `T` with `f`, telling its observers. `None`
    /// when none is set.
    pub fn update_global<T: 'static, R>(&mut self, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        Some(self.global::<T>()?.update(f))
    }

    /// Whether an app-wide `T` is set.
    pub fn has_global<T: 'static>(&self) -> bool {
        self.globals.0.contains_key(&TypeId::of::<T>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn globals_are_one_per_type_shared_and_observed() {
        let mut ui = Ui::new();
        assert!(ui.global::<u32>().is_none() && ui.update_global(|n: &mut u32| *n += 1).is_none());
        ui.set_global(1u32);
        ui.set_global("name");
        let seen = Rc::new(Cell::new(0));
        let s = seen.clone();
        ui.global::<u32>().unwrap().observe(move |n| s.set(*n)).detach();
        ui.update_global(|n: &mut u32| *n += 1);
        ui.set_global(10u32);
        assert_eq!(seen.get(), 10, "setting it again tells its observers");
        assert_eq!(*ui.global::<&str>().unwrap().read(), "name");
        // A clone of the UI shares them.
        let other = ui.clone();
        ui.update_global(|n: &mut u32| *n = 3);
        assert_eq!(*other.global::<u32>().unwrap().read(), 3);
    }
}
