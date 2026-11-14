//! A control's value kept in one place: a
//! [`Binding`] reads a value out of the app's state and writes it back, so
//! a switch, tabs or a disclosure show the state as it is — changed by
//! them, by another control bound to the same value, or by the app — and
//! keep no copy of their own to fall out of step.
//!
//! ```
//! use threers_ui::binding;
//!
//! struct Settings { dark: bool }
//! let dark = binding!(|s: Settings| s.dark);
//! let mut s = Settings { dark: false };
//! dark.set(&mut s, true);
//! assert!(dark.get(&s));
//! ```

use std::cell::RefCell;
use std::rc::Rc;

type Setter<S, T> = Rc<dyn Fn(&mut S, T)>;

/// Reads a `T` out of state `S` and writes one back.
pub struct Binding<S, T> {
    get: Rc<dyn Fn(&S) -> T>,
    set: Setter<S, T>,
}

impl<S, T> Clone for Binding<S, T> {
    fn clone(&self) -> Self {
        Self {
            get: self.get.clone(),
            set: self.set.clone(),
        }
    }
}

impl<S: 'static, T: Clone + 'static> Binding<S, T> {
    pub fn new(get: impl Fn(&S) -> T + 'static, set: impl Fn(&mut S, T) + 'static) -> Self {
        Self {
            get: Rc::new(get),
            set: Rc::new(set),
        }
    }

    /// A value the control keeps itself, for one nothing else needs to
    /// see.
    pub fn local(value: T) -> Self {
        let cell = Rc::new(RefCell::new(value));
        let c = cell.clone();
        Self::new(
            move |_| cell.borrow().clone(),
            move |_, v| *c.borrow_mut() = v,
        )
    }

    /// A value that never changes: a control shown, not used.
    pub fn constant(value: T) -> Self {
        Self::new(move |_| value.clone(), |_, _| {})
    }

    pub fn get(&self, state: &S) -> T {
        (self.get)(state)
    }

    pub fn set(&self, state: &mut S, value: T) {
        (self.set)(state, value)
    }

    /// The same binding, calling `f` with each value written (after it is).
    pub fn on_set(self, f: impl Fn(&mut S, &T) + 'static) -> Self {
        let set = self.set.clone();
        Self {
            get: self.get,
            set: Rc::new(move |s, v: T| {
                set(s, v.clone());
                f(s, &v);
            }),
        }
    }
}

/// A [`Binding`] to a place in the state: `binding!(|s: State| s.a.b)`.
#[macro_export]
macro_rules! binding {
    (|$s:ident : $S:ty| $($place:tt)+) => {
        $crate::binding::Binding::new(
            |$s: &$S| ($($place)+).clone(),
            |$s: &mut $S, v| $($place)+ = v,
        )
    };
}
