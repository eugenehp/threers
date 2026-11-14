//! Values kept between runs: a [`Storage`] holds strings by key (the
//! browser's `localStorage`, a file, memory in tests), and
//! [`stored`] makes a [`Binding`] backed by one, read once and written
//! on every change, so a setting bound to a control survives a reload.
//!
//! ```
//! use std::rc::Rc;
//! use threers_ui::storage::{stored, MemoryStorage, Storage};
//!
//! let store: Rc<dyn Storage> = Rc::new(MemoryStorage::default());
//! let tab = stored::<(), usize>(store.clone(), "tab", 0);
//! tab.set(&mut (), 2);
//! // Next run: the same key reads it back.
//! let again = stored::<(), usize>(store, "tab", 0);
//! assert_eq!(again.get(&()), 2);
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::str::FromStr;

use crate::binding::Binding;

/// Strings by key, kept between runs.
pub trait Storage {
    fn get(&self, key: &str) -> Option<String>;
    fn set(&self, key: &str, value: &str);
    fn remove(&self, key: &str);
}

/// A storage that lasts as long as it does: for tests, and where nothing
/// is kept.
#[derive(Debug, Default)]
pub struct MemoryStorage(RefCell<HashMap<String, String>>);

impl Storage for MemoryStorage {
    fn get(&self, key: &str) -> Option<String> {
        self.0.borrow().get(key).cloned()
    }
    fn set(&self, key: &str, value: &str) {
        self.0.borrow_mut().insert(key.to_owned(), value.to_owned());
    }
    fn remove(&self, key: &str) {
        self.0.borrow_mut().remove(key);
    }
}

/// A binding to the value under `key` in `storage` (`default` when there is
/// none, or it does not parse), written back as text on each change.
pub fn stored<S: 'static, T: Clone + FromStr + ToString + 'static>(
    storage: Rc<dyn Storage>,
    key: &str,
    default: T,
) -> Binding<S, T> {
    let value = storage
        .get(key)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default);
    let cell = Rc::new(RefCell::new(value));
    let (c, key) = (cell.clone(), key.to_owned());
    Binding::new(
        move |_| cell.borrow().clone(),
        move |_, v: T| {
            storage.set(&key, &v.to_string());
            *c.borrow_mut() = v;
        },
    )
}
