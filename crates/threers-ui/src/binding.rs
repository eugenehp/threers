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

    /// A binding to a part of this one's value: `inner` reads it out of the
    /// value and writes it back, and the value, changed, is written back
    /// here.
    ///
    /// ```
    /// use threers_ui::binding;
    /// struct Point { x: f32, y: f32 }
    /// struct S { at: Point }
    /// let x = binding!(|s: S| s.at).then(binding!(|p: Point| p.x));
    /// let mut s = S { at: Point { x: 1.0, y: 2.0 } };
    /// x.set(&mut s, 5.0);
    /// assert_eq!((x.get(&s), s.at.y), (5.0, 2.0));
    /// # impl Clone for Point { fn clone(&self) -> Self { Point { x: self.x, y: self.y } } }
    /// ```
    pub fn then<U: Clone + 'static>(self, inner: Binding<T, U>) -> Binding<S, U> {
        let (get, again, set) = (self.get.clone(), self.get, self.set);
        let (read, write) = (inner.get, inner.set);
        Binding::new(
            move |s| read(&get(s)),
            move |s, u| {
                let mut t = again(s);
                write(&mut t, u);
                set(s, t);
            },
        )
    }

    /// The value as another type: `to` reads it, `from` turns one back (a
    /// slider's `f32` for a count kept as `u32`).
    ///
    /// ```
    /// use threers_ui::binding;
    /// struct S { count: u32 }
    /// let level = binding!(|s: S| s.count).map(|n| n as f32, |x: f32| x.round().max(0.0) as u32);
    /// let mut s = S { count: 3 };
    /// level.set(&mut s, 7.6);
    /// assert_eq!((s.count, level.get(&s)), (8, 8.0));
    /// ```
    pub fn map<U: Clone + 'static>(self, to: impl Fn(T) -> U + 'static, from: impl Fn(U) -> T + 'static) -> Binding<S, U> {
        let (get, set) = (self.get, self.set);
        Binding::new(move |s| to(get(s)), move |s, u| set(s, from(u)))
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
/// Any place that can be read and assigned: a field, a field of a field,
/// an element (`s.rows[i].done`, `i` copied in: a local can be used).
///
/// ```
/// use threers_ui::binding;
/// struct Row { done: bool }
/// struct S { rows: Vec<Row> }
/// let bindings: Vec<_> = (0..2).map(|i| binding!(|s: S| s.rows[i].done)).collect();
/// let mut s = S { rows: vec![Row { done: false }, Row { done: false }] };
/// bindings[1].set(&mut s, true);
/// assert!(!s.rows[0].done && s.rows[1].done);
/// ```
///
/// Anything else is an error that says what it takes:
///
/// ```compile_fail
/// let b = threers_ui::binding!(s.dark);
/// ```
#[macro_export]
macro_rules! binding {
    (|$s:ident : $S:ty| $($place:tt)+) => {
        $crate::binding::Binding::new(
            move |$s: &$S| ($($place)+).clone(),
            move |$s: &mut $S, v| $($place)+ = v,
        )
    };
    ($($other:tt)*) => {
        ::core::compile_error!(::core::concat!(
            "binding! takes a closure to a place in the state, `|s: State| s.field` (or `s.a.b`, `s.rows[i]`), not `",
            ::core::stringify!($($other)*),
            "`"
        ))
    };
}
