//! Systems that say what they use.
//!
//! A system is any function of [`World`]; it can also be a function of
//! parameters, each one thing it reads or writes:
//!
//! ```
//! # use threers_ecs::prelude::*;
//! struct Velocity(Vector3);
//! impl Component for Velocity {}
//!
//! fn movement(mut moving: Query<(Node, &Velocity)>, time: Res<Time>) {
//!     moving.each(|(node, velocity)| node.position = node.position + velocity.0 * time.delta);
//! }
//!
//! let mut app = App::new();
//! app.fixed_system(movement);
//! ```
//!
//! | Parameter | Gives |
//! |---|---|
//! | [`Query<Q>`] | every entity matching `Q`: `each`, `iter`, `single`, `get` |
//! | [`Res<R>`] / [`ResMut<R>`] | a resource, to read or to change |
//! | [`EventReader<T>`] / [`EventWriter<T>`] | events this system has not seen, and sending |
//! | [`Commands`](crate::Commands) | spawn and despawn, applied after the system |
//!
//! Up to eight of them. Two that would alias, such as `Query<&mut Health>` and
//! `Query<&Health>`, or `ResMut<Score>` twice, are refused when the system is
//! added, naming what both want. A system that needs the whole world still
//! takes `&mut World`, and the two kinds mix freely in one schedule.

use crate::commands::Commands;
use crate::events::EventQueue;
use crate::query::{Access, QueryIter, ReadOnly, Term};
use crate::schedule::System;
use crate::world::{Entity, World};
use std::any::type_name;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

/// Something a system function can take.
///
/// # Safety
///
/// `access` must declare everything `fetch` reads or writes, and `fetch` may
/// reach the world only through what it declares.
pub unsafe trait SystemParam {
    type Item<'w>;
    #[doc(hidden)]
    fn access(access: &mut Access);
    #[doc(hidden)]
    unsafe fn fetch<'w>(world: *mut World) -> Self::Item<'w>;
}

/// Every entity matching `Q`, as a system parameter.
pub struct Query<'w, Q: Term> {
    world: *mut World,
    /// The views, taken when the system starts: on the main thread, before a
    /// parallel system's body goes to another.
    state: Q::State,
    _borrow: PhantomData<(&'w mut World, Q)>,
}

impl<'w, Q: Term> Query<'w, Q> {
    /// Run `f` on every match, as [`World::each`] does.
    pub fn each(&mut self, f: impl for<'a> FnMut(Q::Item<'a>)) {
        // SAFETY: the system's access check kept every other parameter off
        // what `Q` touches, and `&mut self` keeps this query's items from
        // overlapping with each other.
        unsafe { crate::world::each_with_state::<Q>(self.world, &self.state, f, false) }
    }

    /// `Q` for one entity, changed by `f`, as [`World::update`] does.
    pub fn get_mut<R>(
        &mut self,
        id: Entity,
        f: impl for<'a> FnOnce(Q::Item<'a>) -> R,
    ) -> Option<R> {
        // SAFETY: as for `each`.
        unsafe {
            let alive = (*self.world).is_alive(id);
            (alive && Q::matches(&self.state, id)).then(|| Q::fetch(&self.state, id).map(f))?
        }
    }
}

impl<'w, Q: Term + ReadOnly> Query<'w, Q> {
    /// Every match, as an iterator.
    pub fn iter(&self) -> QueryIter<'_, Q> {
        // SAFETY: read-only terms; other parameters do not write what `Q` reads.
        unsafe { QueryIter::new(&*self.world) }
    }

    /// The first match, for what there is one of.
    pub fn single(&self) -> Option<Q::Item<'_>> {
        // SAFETY: as for `iter`.
        unsafe { (*self.world).single::<Q>() }
    }

    /// `Q` for one entity.
    pub fn get(&self, id: Entity) -> Option<Q::Item<'_>> {
        // SAFETY: as for `iter`.
        unsafe { (*self.world).read::<Q>(id) }
    }

    pub fn count(&self) -> usize {
        self.iter().count()
    }
}

unsafe impl<Q: Term> SystemParam for Query<'_, Q> {
    type Item<'w> = Query<'w, Q>;
    fn access(access: &mut Access) {
        Q::claim(&mut Vec::new());
        Q::access(access);
    }
    unsafe fn fetch<'w>(world: *mut World) -> Query<'w, Q> {
        Query {
            world,
            state: Q::state(world),
            _borrow: PhantomData,
        }
    }
}

/// A resource to read.
pub struct Res<'w, R: 'static>(&'w R);

impl<R: 'static> Deref for Res<'_, R> {
    type Target = R;
    fn deref(&self) -> &R {
        self.0
    }
}

unsafe impl<R: 'static> SystemParam for Res<'_, R> {
    type Item<'w> = Res<'w, R>;
    fn access(access: &mut Access) {
        access.read::<Res<'static, R>>();
    }
    unsafe fn fetch<'w>(world: *mut World) -> Res<'w, R> {
        Res(&*crate::world::resource_ptr::<R>(world))
    }
}

/// A resource to change.
pub struct ResMut<'w, R: 'static>(&'w mut R);

impl<R: 'static> Deref for ResMut<'_, R> {
    type Target = R;
    fn deref(&self) -> &R {
        self.0
    }
}

impl<R: 'static> DerefMut for ResMut<'_, R> {
    fn deref_mut(&mut self) -> &mut R {
        self.0
    }
}

unsafe impl<R: 'static> SystemParam for ResMut<'_, R> {
    type Item<'w> = ResMut<'w, R>;
    fn access(access: &mut Access) {
        access.write::<Res<'static, R>>();
    }
    unsafe fn fetch<'w>(world: *mut World) -> ResMut<'w, R> {
        ResMut(&mut *crate::world::resource_ptr::<R>(world))
    }
}

/// Events of a type this system has not read yet.
pub struct EventReader<'w, T: 'static> {
    queue: Option<&'w mut EventQueue<T>>,
    reader: Option<crate::schedule::SystemKey>,
}

impl<T: 'static> EventReader<'_, T> {
    /// Lend `f` each event, once per system, as [`World::read_events`] does.
    pub fn read(&mut self, f: impl FnMut(&T)) {
        if let Some(queue) = &mut self.queue {
            queue.read_with(self.reader, f);
        }
    }
}

impl<T: Clone + 'static> EventReader<'_, T> {
    /// The events, copied.
    pub fn collect(&mut self) -> Vec<T> {
        let mut out = Vec::new();
        self.read(|e| out.push(e.clone()));
        out
    }
}

unsafe impl<T: 'static> SystemParam for EventReader<'_, T> {
    type Item<'w> = EventReader<'w, T>;
    fn access(access: &mut Access) {
        access.write::<EventQueue<T>>();
    }
    unsafe fn fetch<'w>(world: *mut World) -> EventReader<'w, T> {
        let reader = *std::ptr::addr_of!((*world).reader);
        EventReader {
            queue: crate::world::event_queue_ptr::<T>(world, false).as_mut(),
            reader,
        }
    }
}

/// Sending events of a type.
pub struct EventWriter<'w, T: 'static>(&'w mut EventQueue<T>);

impl<T: 'static> EventWriter<'_, T> {
    pub fn send(&mut self, event: T) {
        self.0.push(event);
    }
}

unsafe impl<T: 'static> SystemParam for EventWriter<'_, T> {
    type Item<'w> = EventWriter<'w, T>;
    fn access(access: &mut Access) {
        access.write::<EventQueue<T>>();
    }
    unsafe fn fetch<'w>(world: *mut World) -> EventWriter<'w, T> {
        EventWriter(&mut *crate::world::event_queue_ptr::<T>(world, true))
    }
}

unsafe impl SystemParam for Commands {
    type Item<'w> = Commands;
    fn access(_: &mut Access) {}
    unsafe fn fetch<'w>(world: *mut World) -> Self::Item<'w> {
        (*std::ptr::addr_of!((*world).commands)).clone()
    }
}

/// A parameter a [parallel](System::parallel) system may take: what it gives may
/// cross to another thread. `Commands` may not, being single-threaded.
///
/// # Safety
///
/// The parameter's item may be sent to another thread and used there while
/// other systems, with access disjoint from it, run on others.
pub unsafe trait ParallelParam: SystemParam {}
unsafe impl<Q: Term + crate::query::CrossThread> ParallelParam for Query<'_, Q> {}
unsafe impl<R: Sync + 'static> ParallelParam for Res<'_, R> {}
unsafe impl<R: Send + 'static> ParallelParam for ResMut<'_, R> {}
unsafe impl<T: Send + 'static> ParallelParam for EventReader<'_, T> {}
unsafe impl<T: Send + 'static> ParallelParam for EventWriter<'_, T> {}

/// One parallel system's body with its parameters, ready to go to a thread.
pub(crate) type Job = Box<dyn FnOnce() + Send>;

/// Fetch a parallel system's parameters on this thread, and hand back its body.
pub(crate) type Prepare = Box<dyn FnMut(*mut World) -> Job>;

/// Moves what `ParallelParam` and `Send` bounds have already vouched for.
struct Vouched<T>(T);
unsafe impl<T> Send for Vouched<T> {}
impl<T> Vouched<T> {
    fn take(self) -> T {
        self.0
    }
}

/// Something that can be a parallel system: a function of parallel
/// parameters that can itself go to another thread.
pub trait IntoParallel<M> {
    fn into_parallel(self) -> System;
}

/// Something that can be a system: a function of the world, a function of
/// parameters, or a [`System`] already set up. `M` tells the kinds apart.
pub trait IntoSystem<M> {
    fn into_system(self) -> System;
}

impl IntoSystem<System> for System {
    fn into_system(self) -> System {
        self
    }
}

#[doc(hidden)]
pub struct WholeWorld;

/// Kept for code written before parameter systems: `system.into()`.
impl<F: FnMut(&mut World) + 'static> From<F> for System {
    fn from(system: F) -> System {
        system.into_system()
    }
}

impl<F: FnMut(&mut World) + 'static> IntoSystem<WholeWorld> for F {
    fn into_system(self) -> System {
        System::of_world(self)
    }
}

/// Check a system's parameters against each other, once, as it is made.
fn check(name: &str, accesses: &[Access]) {
    for (i, a) in accesses.iter().enumerate() {
        for b in &accesses[i + 1..] {
            if let Some(what) = a.conflict(b) {
                let what = what
                    .replace("threers_ecs::params::Res<", "resource <")
                    .replace("threers_ecs::events::EventQueue<", "events <");
                panic!(
                    "{name}: two parameters both use {what}, and one of them changes it. \
                     Put them in one query, or split the system"
                );
            }
        }
    }
}

macro_rules! param_system {
    ($($p:ident),+) => {
        impl<F, $($p: SystemParam + 'static),+> IntoSystem<fn($($p,)+)> for F
        where
            F: FnMut($($p),+) + for<'w> FnMut($($p::Item<'w>),+) + 'static,
        {
            #[allow(non_snake_case)]
            fn into_system(mut self) -> System {
                let accesses = [$({
                    let mut a = Access::default();
                    $p::access(&mut a);
                    a
                }),+];
                check(type_name::<F>(), &accesses);
                let run = move |world: &mut World| {
                    let world: *mut World = world;
                    // SAFETY: the access check above found no two parameters
                    // that alias, and each reaches the world only through
                    // what it declared.
                    unsafe {
                        $(let $p = $p::fetch(world);)+
                        self($($p),+);
                    }
                };
                System::labelled::<F>(Box::new(run))
            }
        }
    };
}

macro_rules! parallel_system {
    ($($p:ident),+) => {
        impl<F, $($p: ParallelParam + 'static),+> IntoParallel<fn($($p,)+)> for F
        where
            F: FnMut($($p),+) + for<'w> FnMut($($p::Item<'w>),+) + Send + 'static,
        {
            #[allow(non_snake_case)]
            fn into_parallel(mut self) -> System {
                let accesses = [$({
                    let mut a = Access::default();
                    $p::access(&mut a);
                    a
                }),+];
                check(type_name::<F>(), &accesses);
                let mut access = Access::default();
                for a in &accesses {
                    access.reads.extend(a.reads.iter().copied());
                    access.writes.extend(a.writes.iter().copied());
                }
                let prepare = move |world: *mut World| -> Job {
                    // SAFETY: fetched here, on the thread that owns the world,
                    // while nothing else runs; the schedule sends the body to
                    // another thread only beside systems whose access is
                    // disjoint from this one's, and joins before the world is
                    // touched again. `ParallelParam` and `F: Send` vouch for
                    // what crosses; the system is in one batch at most once,
                    // so the pointer to it is not shared.
                    unsafe {
                        let items = Vouched(($($p::fetch(world),)+));
                        let system = Vouched(&mut self as *mut F);
                        Box::new(move || {
                            let ($($p,)+) = items.take();
                            (*system.take())($($p),+);
                        })
                    }
                };
                System::parallel_from::<F>(Box::new(prepare), access)
            }
        }
    };
}

parallel_system!(A);
parallel_system!(A, B);
parallel_system!(A, B, C);
parallel_system!(A, B, C, D);
parallel_system!(A, B, C, D, E);
parallel_system!(A, B, C, D, E, G);
parallel_system!(A, B, C, D, E, G, H);
parallel_system!(A, B, C, D, E, G, H, I);

param_system!(A);
param_system!(A, B);
param_system!(A, B, C);
param_system!(A, B, C, D);
param_system!(A, B, C, D, E);
param_system!(A, B, C, D, E, G);
param_system!(A, B, C, D, E, G, H);
param_system!(A, B, C, D, E, G, H, I);
