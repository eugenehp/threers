//! State shared by parts of a UI, watched for changes: a [`Model`] holds a value; [`update`](Model::update) changes it and
//! tells everything [observing](Model::observe) it; an [`Emitter`] sends
//! events to its [subscribers](Emitter::subscribe). Observing returns a
//! [`Subscription`], and dropping that stops it.
//!
//! A part of the UI built from a model rebuilds when told, or checks
//! [`version`](Model::version) each frame and rebuilds only when it moved:
//! nothing is copied into the tree while nothing changes.
//!
//! ```
//! use std::cell::Cell;
//! use std::rc::Rc;
//! use threers_ui::model::Model;
//!
//! let count = Model::new(0);
//! let seen = Rc::new(Cell::new(0));
//! let s = seen.clone();
//! let watching = count.observe(move |n| s.set(*n));
//! count.update(|n| *n += 2);
//! assert_eq!((*count.read(), seen.get()), (2, 2));
//! drop(watching);
//! count.update(|n| *n += 1);
//! assert_eq!(seen.get(), 2, "no longer watching");
//! ```

use std::cell::{Cell, Ref, RefCell};
use std::rc::Rc;

type Observer<T> = Box<dyn FnMut(&T)>;

/// Who is listening, by id, so a [`Subscription`] can stop its own.
struct Listeners<T: ?Sized> {
    next: Cell<u64>,
    list: RefCell<Vec<(u64, Observer<T>)>>,
}

impl<T: ?Sized> Default for Listeners<T> {
    fn default() -> Self {
        Self {
            next: Cell::new(0),
            list: RefCell::new(vec![]),
        }
    }
}

impl<T: ?Sized + 'static> Listeners<T> {
    fn add(self: &Rc<Self>, f: Observer<T>) -> Subscription {
        let id = self.next.get();
        self.next.set(id + 1);
        self.list.borrow_mut().push((id, f));
        let weak = Rc::downgrade(self);
        Subscription {
            cancel: Some(Box::new(move || {
                if let Some(l) = weak.upgrade() {
                    l.list.borrow_mut().retain(|(i, _)| *i != id);
                }
            })),
        }
    }

    /// Calls every listener with `value`. Ones added meanwhile hear the next.
    fn call(&self, value: &T) {
        let mut now = std::mem::take(&mut *self.list.borrow_mut());
        for (_, f) in &mut now {
            f(value);
        }
        let mut list = self.list.borrow_mut();
        let added = std::mem::take(&mut *list);
        *list = now;
        list.extend(added);
    }
}

/// Keeps an observer or subscriber listening; dropping it stops it.
#[must_use = "dropping a Subscription stops it at once; `detach` keeps it"]
pub struct Subscription {
    cancel: Option<Box<dyn FnOnce()>>,
}

impl Subscription {
    /// Keeps listening for as long as the model or emitter lives.
    pub fn detach(mut self) {
        self.cancel = None;
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        if let Some(c) = self.cancel.take() {
            c();
        }
    }
}

/// A model read while [`track`] ran: its version counter, and its version
/// then.
#[derive(Clone)]
pub struct Read(Rc<Cell<u64>>, u64);

/// The models read by something, to tell whether it would now come out
/// differently.
#[derive(Clone, Default)]
pub struct Reads(Vec<Read>);

impl Reads {
    /// Whether any model read has changed since.
    pub fn changed(&self) -> bool {
        self.0.iter().any(|Read(v, seen)| v.get() != *seen)
    }

    /// How many reads were recorded.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

thread_local! {
    static READS: std::cell::RefCell<Option<Vec<Read>>> = const { std::cell::RefCell::new(None) };
}

/// Runs `f`, recording every [`Model::read`] in it, as automatic
/// dependency tracking does: the result is to be computed again only once
/// [`Reads::changed`] says so.
pub fn track<R>(f: impl FnOnce() -> R) -> (R, Reads) {
    let outer = READS.with(|r| r.borrow_mut().replace(vec![]));
    let out = f();
    let reads = READS
        .with(|r| std::mem::replace(&mut *r.borrow_mut(), outer))
        .unwrap_or_default();
    // Reads inside count for the tracking around it too.
    READS.with(|r| {
        if let Some(o) = r.borrow_mut().as_mut() {
            o.extend(reads.iter().cloned());
        }
    });
    (out, Reads(reads))
}

/// A value shared by parts of a UI (clones share it), which tells its
/// observers when it changes. See the [module docs](self).
pub struct Model<T> {
    value: Rc<RefCell<T>>,
    version: Rc<Cell<u64>>,
    observers: Rc<Listeners<T>>,
}

impl<T> Clone for Model<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            version: self.version.clone(),
            observers: self.observers.clone(),
        }
    }
}

impl<T: 'static> Model<T> {
    pub fn new(value: T) -> Self {
        Self {
            value: Rc::new(RefCell::new(value)),
            version: Rc::new(Cell::new(0)),
            observers: Rc::default(),
        }
    }

    /// The value. Not while it is being updated. Inside [`track`], the
    /// read is recorded.
    pub fn read(&self) -> Ref<'_, T> {
        READS.with(|r| {
            if let Some(reads) = r.borrow_mut().as_mut() {
                reads.push(Read(self.version.clone(), self.version.get()));
            }
        });
        self.value.borrow()
    }

    /// Changes the value with `f`, then tells the observers. An observer may not update this model while
    /// it is told: it runs while the value is read.
    pub fn update<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        let r = f(&mut self.value.borrow_mut());
        self.version.set(self.version.get() + 1);
        self.observers.call(&self.value.borrow());
        r
    }

    /// Replaces the value.
    pub fn set(&self, value: T) {
        self.update(|v| *v = value);
    }

    /// Calls `f` with the value after each change, until the returned
    /// [`Subscription`] is dropped.
    pub fn observe(&self, f: impl FnMut(&T) + 'static) -> Subscription {
        self.observers.add(Box::new(f))
    }

    /// How many times it has changed: a frame that saw version `v` has
    /// nothing to rebuild while it is still `v`.
    pub fn version(&self) -> u64 {
        self.version.get()
    }

    /// Whether `a` and `b` are the same model (not equal values).
    pub fn same(a: &Self, b: &Self) -> bool {
        Rc::ptr_eq(&a.value, &b.value)
    }
}

/// Events of type `E` sent to subscribers. Clones share the subscribers.
pub struct Emitter<E> {
    subscribers: Rc<Listeners<E>>,
}

impl<E> Clone for Emitter<E> {
    fn clone(&self) -> Self {
        Self {
            subscribers: self.subscribers.clone(),
        }
    }
}

impl<E: 'static> Default for Emitter<E> {
    fn default() -> Self {
        Self {
            subscribers: Rc::default(),
        }
    }
}

impl<E: 'static> Emitter<E> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Calls `f` with every event emitted, until the returned
    /// [`Subscription`] is dropped.
    pub fn subscribe(&self, f: impl FnMut(&E) + 'static) -> Subscription {
        self.subscribers.add(Box::new(f))
    }

    pub fn emit(&self, event: E) {
        self.subscribers.call(&event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observers_hear_changes_until_dropped_and_versions_count_them() {
        let m = Model::new(vec![1]);
        let other = m.clone();
        let log = Rc::new(RefCell::new(vec![]));
        let l = log.clone();
        let a = m.observe(move |v| l.borrow_mut().push(v.len()));
        let l = log.clone();
        other
            .observe(move |v| l.borrow_mut().push(100 + v.len()))
            .detach();
        let v0 = m.version();
        other.update(|v| v.push(2));
        assert_eq!(
            *log.borrow(),
            [2, 102],
            "clones share the value and the observers"
        );
        assert_eq!(m.version(), v0 + 1);
        drop(a);
        m.set(vec![]);
        assert_eq!(*log.borrow(), [2, 102, 100]);
        assert!(Model::same(&m, &other) && !Model::same(&m, &Model::new(vec![])));
    }

    #[test]
    fn an_observer_added_while_others_are_told_hears_the_next_change() {
        let m = Model::new(0);
        let count = Rc::new(Cell::new(0));
        let (mc, c) = (m.clone(), count.clone());
        let late: Rc<RefCell<Option<Subscription>>> = Rc::default();
        let l = late.clone();
        m.observe(move |_| {
            if l.borrow().is_none() {
                let c = c.clone();
                *l.borrow_mut() = Some(mc.observe(move |_| c.set(c.get() + 1)));
            }
        })
        .detach();
        m.set(1);
        assert_eq!(count.get(), 0);
        m.set(2);
        assert_eq!(count.get(), 1);
    }

    #[test]
    fn emitters_send_events_to_subscribers() {
        let e: Emitter<&'static str> = Emitter::new();
        let log = Rc::new(RefCell::new(vec![]));
        let l = log.clone();
        let s = e.subscribe(move |ev| l.borrow_mut().push(*ev));
        e.emit("saved");
        drop(s);
        e.emit("closed");
        assert_eq!(*log.borrow(), ["saved"]);
    }
}
