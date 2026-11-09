//! What a query can ask for.

use crate::store::Store;
use crate::world::{Component, Entity, World};
use slotmap::SecondaryMap;
use std::any::{type_name, Any, TypeId};
use std::marker::PhantomData;
use std::ptr::{addr_of, addr_of_mut};
use threers::core::{Object3D, ObjectArena};

/// The entity's scene node: transform, visibility, mesh and the rest.
pub struct Node;

/// The entity's scene node, read-only, for [`World::each_ref`].
pub struct NodeRef;

/// Require a component without reading it.
pub struct With<C>(PhantomData<C>);

/// Exclude entities that have a component.
pub struct Without<C>(PhantomData<C>);

/// Entities given a `C` since this system last ran.
pub struct Added<C>(PhantomData<C>);

/// Entities whose `C` was inserted or mutably borrowed since this system last
/// ran. A `&mut C` term counts as a change even if nothing was written, so ask
/// for `&C` where you can.
pub struct Changed<C>(PhantomData<C>);

/// Terms that only read, usable with [`World::each_ref`].
#[diagnostic::on_unimplemented(
    message = "`{Self}` writes, and this only reads",
    note = "use `&C` for `&mut C` and `NodeRef` for `Node`, or call `each` / `update`, which may write"
)]
pub trait ReadOnly {}

/// Terms [`World::par_query`] can hand to several threads at once. Not
/// [`Node`]: the scene's arena is one value, not a row each.
///
/// # Safety
///
/// Fetching different ids from one state on different threads must be sound.
pub unsafe trait Parallel {}

/// Something [`World::each`] can ask for. Sealed, because `each` trusts what a
/// term claims to touch.
///
/// | Term | Yields | Matches |
/// |---|---|---|
/// | `Entity` | the id | everything |
/// | [`Node`] | `&mut Object3D` | everything |
/// | [`NodeRef`] | `&Object3D` | everything |
/// | `&C` / `&mut C` | the component | entities with `C` |
/// | `Option<&C>` / `Option<&mut C>` | the component, if any | everything |
/// | [`With<C>`] | `()` | entities with `C` |
/// | [`Without<C>`] | `()` | entities without `C` |
/// | [`Added<C>`] | `()` | entities given `C` since this system last ran |
/// | [`Changed<C>`] | `()` | entities whose `C` changed since this system last ran |
///
/// Tuples of terms match what all of them match. The filters read only a row's
/// bookkeeping, so one can name a component the query also borrows, as in
/// `(&C, Changed<C>)`.
///
/// # Safety
///
/// `claim` must name every value `fetch` hands out. `fetch` runs only on an id
/// `matches` accepted, and only `fetch` may write.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not something a query can ask for",
    note = "a query takes `&C`, `&mut C`, `Option<&C>`, `With<C>`, `Without<C>`, `Added<C>`, `Changed<C>`, `Entity`, `Node` or `NodeRef`, or a tuple of them",
    note = "a component type `C` needs `impl Component for C {{}}`"
)]
pub unsafe trait Term: sealed::Sealed {
    /// What the query hands out for each entity.
    type Item<'w>;
    #[doc(hidden)]
    type State;
    #[doc(hidden)]
    fn claim(claims: &mut Claims);
    /// What fetching this reads and writes, as a system parameter declares it.
    #[doc(hidden)]
    fn access(access: &mut Access);
    #[doc(hidden)]
    unsafe fn state(world: *mut World) -> Self::State;
    /// The ids this term limits a query to, if it limits it at all.
    #[doc(hidden)]
    fn candidates(state: &Self::State) -> Option<*const [Entity]>;
    /// Whether `id` matches. A query asks every term before fetching from any,
    /// because fetching `&mut C` marks the row changed: a later term turning the
    /// row down would leave it marked for a write that never happened.
    #[doc(hidden)]
    unsafe fn matches(state: &Self::State, id: Entity) -> bool;
    #[doc(hidden)]
    unsafe fn fetch<'w>(state: &Self::State, id: Entity) -> Option<Self::Item<'w>>;
}

mod sealed {
    pub trait Sealed {}
}

#[doc(hidden)]
pub type Claims = Vec<TypeId>;

/// What a system reads and writes, for telling which of its parameters would
/// alias. The node arena counts as one thing.
#[doc(hidden)]
#[derive(Debug, Default, Clone)]
pub struct Access {
    pub(crate) reads: Vec<(TypeId, &'static str)>,
    pub(crate) writes: Vec<(TypeId, &'static str)>,
}

impl Access {
    pub(crate) fn read<T: 'static>(&mut self) {
        self.reads.push((TypeId::of::<T>(), type_name::<T>()));
    }
    pub(crate) fn write<T: 'static>(&mut self) {
        self.writes.push((TypeId::of::<T>(), type_name::<T>()));
    }
    /// The first thing written by one and touched by another, by its name.
    pub(crate) fn conflict(&self, other: &Access) -> Option<&'static str> {
        let touches =
            |a: &Access, id: TypeId| a.reads.iter().chain(&a.writes).any(|(t, _)| *t == id);
        self.writes
            .iter()
            .find(|(id, _)| touches(other, *id))
            .or_else(|| other.writes.iter().find(|(id, _)| touches(self, *id)))
            .map(|(_, name)| *name)
    }
}

fn claim<T: 'static>(claims: &mut Claims) {
    let id = TypeId::of::<T>();
    assert!(
        !claims.contains(&id),
        "a query names {} twice",
        type_name::<T>()
    );
    claims.push(id);
}

/// Raw views of one store, taken once per query. `values` is the buffer's base
/// pointer because `Vec::get_mut` would reborrow the whole slice for each row.
#[doc(hidden)]
pub struct View<C> {
    rows: *const SecondaryMap<Entity, u32>,
    ids: *const [Entity],
    values: *mut C,
    added: *const u64,
    changed: *mut u64,
    tick: u64,
    since: u64,
}

impl<C> View<C> {
    unsafe fn row(&self, id: Entity) -> Option<usize> {
        unsafe { (*self.rows).get(id) }.map(|&row| row as usize)
    }
}

/// Goes through a raw world pointer and touches only `stores`, so that a
/// `Node` term's pointer into `scene`, taken earlier, stays valid.
unsafe fn view<C: Component>(world: *mut World) -> Option<View<C>> {
    let (tick, since) = unsafe { ((*world).tick, (*world).since) };
    let stores = unsafe { &mut *addr_of_mut!((*world).stores) };
    let store: &mut dyn Any = stores.get_mut(&TypeId::of::<C>())?.as_mut();
    let store: &mut Store<C> = store.downcast_mut()?;
    Some(View {
        rows: &*store.rows,
        ids: store.ids.as_slice(),
        values: store.values.as_mut_ptr(),
        added: store.added.as_ptr(),
        changed: store.changed.as_mut_ptr(),
        tick,
        since,
    })
}

/// The same, for terms that only read: `each_ref` has just a shared world, so
/// nothing here may take `&mut`, and nothing may later write through the result.
unsafe fn read_view<C: Component>(world: *mut World) -> Option<View<C>> {
    let (tick, since) = unsafe { ((*world).tick, (*world).since) };
    let stores = unsafe { &*addr_of!((*world).stores) };
    let store: &dyn Any = stores.get(&TypeId::of::<C>())?.as_ref();
    let store: &Store<C> = store.downcast_ref()?;
    Some(View {
        rows: &*store.rows,
        ids: store.ids.as_slice(),
        values: store.values.as_ptr() as *mut C,
        added: store.added.as_ptr(),
        changed: store.changed.as_ptr() as *mut u64,
        tick,
        since,
    })
}

/// What a required component limits a query to. A component with no store has
/// no rows, so nothing can match.
fn rows_of<C>(view: &Option<View<C>>) -> Option<*const [Entity]> {
    const NONE: &[Entity] = &[];
    Some(view.as_ref().map_or(NONE as *const _, |v| v.ids))
}

impl sealed::Sealed for Entity {}
unsafe impl Term for Entity {
    type Item<'w> = Entity;
    type State = ();
    fn claim(_: &mut Claims) {}
    fn access(_: &mut Access) {}
    unsafe fn state(_: *mut World) {}
    fn candidates(_: &()) -> Option<*const [Entity]> {
        None
    }
    unsafe fn matches(_: &Self::State, _: Entity) -> bool {
        true
    }
    unsafe fn fetch<'w>(_: &(), id: Entity) -> Option<Self::Item<'w>> {
        Some(id)
    }
}

impl sealed::Sealed for Node {}
unsafe impl Term for Node {
    type Item<'w> = &'w mut Object3D;
    type State = *mut ObjectArena;
    fn claim(claims: &mut Claims) {
        claim::<Node>(claims);
    }
    fn access(access: &mut Access) {
        access.write::<Node>();
    }
    unsafe fn state(world: *mut World) -> *mut ObjectArena {
        unsafe { addr_of_mut!((*world).scene.arena) }
    }
    fn candidates(_: &*mut ObjectArena) -> Option<*const [Entity]> {
        None
    }
    unsafe fn matches(_: &Self::State, _: Entity) -> bool {
        true
    }
    unsafe fn fetch<'w>(arena: &*mut ObjectArena, id: Entity) -> Option<&'w mut Object3D> {
        unsafe { (**arena).get_mut(id) }
    }
}

impl<C: Component> sealed::Sealed for &C {}
unsafe impl<C: Component> Term for &C {
    type Item<'w> = &'w C;
    type State = Option<View<C>>;
    fn claim(claims: &mut Claims) {
        claim::<C>(claims);
    }
    fn access(access: &mut Access) {
        access.read::<C>();
    }
    unsafe fn state(world: *mut World) -> Self::State {
        unsafe { read_view::<C>(world) }
    }
    fn candidates(state: &Self::State) -> Option<*const [Entity]> {
        rows_of(state)
    }
    unsafe fn matches(state: &Self::State, id: Entity) -> bool {
        state
            .as_ref()
            .is_some_and(|view| unsafe { view.row(id) }.is_some())
    }
    unsafe fn fetch<'w>(state: &Self::State, id: Entity) -> Option<&'w C> {
        let view = state.as_ref()?;
        Some(unsafe { &*view.values.add(view.row(id)?) })
    }
}

impl<C: Component> sealed::Sealed for &mut C {}
unsafe impl<C: Component> Term for &mut C {
    type Item<'w> = &'w mut C;
    type State = Option<View<C>>;
    fn claim(claims: &mut Claims) {
        claim::<C>(claims);
    }
    fn access(access: &mut Access) {
        access.write::<C>();
    }
    unsafe fn state(world: *mut World) -> Self::State {
        unsafe { view::<C>(world) }
    }
    fn candidates(state: &Self::State) -> Option<*const [Entity]> {
        rows_of(state)
    }
    unsafe fn matches(state: &Self::State, id: Entity) -> bool {
        state
            .as_ref()
            .is_some_and(|view| unsafe { view.row(id) }.is_some())
    }
    unsafe fn fetch<'w>(state: &Self::State, id: Entity) -> Option<&'w mut C> {
        let view = state.as_ref()?;
        let row = unsafe { view.row(id) }?;
        unsafe { *view.changed.add(row) = view.tick };
        Some(unsafe { &mut *view.values.add(row) })
    }
}

impl<C: Component> sealed::Sealed for Option<&C> {}
unsafe impl<C: Component> Term for Option<&C> {
    type Item<'w> = Option<&'w C>;
    type State = Option<View<C>>;
    fn claim(claims: &mut Claims) {
        claim::<C>(claims);
    }
    fn access(access: &mut Access) {
        access.read::<C>();
    }
    unsafe fn state(world: *mut World) -> Self::State {
        unsafe { read_view::<C>(world) }
    }
    fn candidates(_: &Self::State) -> Option<*const [Entity]> {
        None
    }
    unsafe fn matches(_: &Self::State, _: Entity) -> bool {
        true
    }
    unsafe fn fetch<'w>(state: &Self::State, id: Entity) -> Option<Option<&'w C>> {
        Some(unsafe { <&C as Term>::fetch(state, id) })
    }
}

impl<C: Component> sealed::Sealed for Option<&mut C> {}
unsafe impl<C: Component> Term for Option<&mut C> {
    type Item<'w> = Option<&'w mut C>;
    type State = Option<View<C>>;
    fn claim(claims: &mut Claims) {
        claim::<C>(claims);
    }
    fn access(access: &mut Access) {
        access.write::<C>();
    }
    unsafe fn state(world: *mut World) -> Self::State {
        unsafe { view::<C>(world) }
    }
    fn candidates(_: &Self::State) -> Option<*const [Entity]> {
        None
    }
    unsafe fn matches(_: &Self::State, _: Entity) -> bool {
        true
    }
    unsafe fn fetch<'w>(state: &Self::State, id: Entity) -> Option<Option<&'w mut C>> {
        Some(unsafe { <&mut C as Term>::fetch(state, id) })
    }
}

impl<C: Component> sealed::Sealed for With<C> {}
unsafe impl<C: Component> Term for With<C> {
    type Item<'w> = ();
    type State = Option<View<C>>;
    fn claim(_: &mut Claims) {}
    fn access(access: &mut Access) {
        access.read::<C>();
    }
    unsafe fn state(world: *mut World) -> Self::State {
        unsafe { read_view::<C>(world) }
    }
    fn candidates(state: &Self::State) -> Option<*const [Entity]> {
        rows_of(state)
    }
    unsafe fn matches(state: &Self::State, id: Entity) -> bool {
        state
            .as_ref()
            .is_some_and(|view| unsafe { view.row(id) }.is_some())
    }
    unsafe fn fetch<'w>(_: &Self::State, _: Entity) -> Option<Self::Item<'w>> {
        Some(())
    }
}

impl<C: Component> sealed::Sealed for Without<C> {}
unsafe impl<C: Component> Term for Without<C> {
    type Item<'w> = ();
    type State = Option<View<C>>;
    fn claim(_: &mut Claims) {}
    fn access(access: &mut Access) {
        access.read::<C>();
    }
    unsafe fn state(world: *mut World) -> Self::State {
        unsafe { read_view::<C>(world) }
    }
    fn candidates(_: &Self::State) -> Option<*const [Entity]> {
        None
    }
    unsafe fn matches(state: &Self::State, id: Entity) -> bool {
        !state
            .as_ref()
            .is_some_and(|view| unsafe { view.row(id) }.is_some())
    }
    unsafe fn fetch<'w>(_: &Self::State, _: Entity) -> Option<Self::Item<'w>> {
        Some(())
    }
}

impl sealed::Sealed for NodeRef {}
unsafe impl Term for NodeRef {
    type Item<'w> = &'w Object3D;
    type State = *const ObjectArena;
    /// As `Node`: the two would be a shared and a mutable view of one value.
    fn claim(claims: &mut Claims) {
        claim::<Node>(claims);
    }
    fn access(access: &mut Access) {
        access.read::<Node>();
    }
    unsafe fn state(world: *mut World) -> *const ObjectArena {
        unsafe { addr_of!((*world).scene.arena) }
    }
    fn candidates(_: &*const ObjectArena) -> Option<*const [Entity]> {
        None
    }
    unsafe fn matches(_: &Self::State, _: Entity) -> bool {
        true
    }
    unsafe fn fetch<'w>(arena: &*const ObjectArena, id: Entity) -> Option<Self::Item<'w>> {
        unsafe { (**arena).get(id) }
    }
}

impl<C: Component> sealed::Sealed for Added<C> {}
unsafe impl<C: Component> Term for Added<C> {
    type Item<'w> = ();
    type State = Option<View<C>>;
    fn claim(_: &mut Claims) {}
    fn access(access: &mut Access) {
        access.read::<C>();
    }
    unsafe fn state(world: *mut World) -> Self::State {
        unsafe { read_view::<C>(world) }
    }
    fn candidates(state: &Self::State) -> Option<*const [Entity]> {
        rows_of(state)
    }
    unsafe fn matches(state: &Self::State, id: Entity) -> bool {
        state
            .as_ref()
            .is_some_and(|v| unsafe { v.row(id).is_some_and(|row| *v.added.add(row) > v.since) })
    }
    unsafe fn fetch<'w>(_: &Self::State, _: Entity) -> Option<Self::Item<'w>> {
        Some(())
    }
}

impl<C: Component> sealed::Sealed for Changed<C> {}
unsafe impl<C: Component> Term for Changed<C> {
    type Item<'w> = ();
    type State = Option<View<C>>;
    fn claim(_: &mut Claims) {}
    fn access(access: &mut Access) {
        access.read::<C>();
    }
    unsafe fn state(world: *mut World) -> Self::State {
        unsafe { read_view::<C>(world) }
    }
    fn candidates(state: &Self::State) -> Option<*const [Entity]> {
        rows_of(state)
    }
    unsafe fn matches(state: &Self::State, id: Entity) -> bool {
        state
            .as_ref()
            .is_some_and(|v| unsafe { v.row(id).is_some_and(|row| *v.changed.add(row) > v.since) })
    }
    unsafe fn fetch<'w>(_: &Self::State, _: Entity) -> Option<Self::Item<'w>> {
        Some(())
    }
}

/// What [`World::iter`] returns.
pub struct QueryIter<'w, Q: Term> {
    world: &'w World,
    state: Q::State,
    ids: *const [Entity],
    /// Holds `ids` where the query has no list of its own to walk.
    _every: Vec<Entity>,
    at: usize,
}

impl<'w, Q: Term + ReadOnly> QueryIter<'w, Q> {
    pub(crate) fn new(world: &'w World) -> Self {
        Q::claim(&mut Vec::new());
        // SAFETY: `ReadOnly` terms only read, and `'w` keeps the stores as they are.
        let state = unsafe { Q::state(world as *const World as *mut World) };
        let (ids, every) = match Q::candidates(&state) {
            Some(ids) => (ids, Vec::new()),
            None => {
                let every: Vec<Entity> = world.scene.arena.nodes.keys().collect();
                (every.as_slice() as *const [Entity], every)
            }
        };
        Self {
            world,
            state,
            ids,
            _every: every,
            at: 0,
        }
    }
}

impl<'w, Q: Term + ReadOnly> Iterator for QueryIter<'w, Q> {
    type Item = Q::Item<'w>;

    fn next(&mut self) -> Option<Self::Item> {
        // SAFETY: `ids` lives in a store or in `_every`, and neither changes.
        unsafe {
            let ids = &*self.ids;
            while let Some(&id) = ids.get(self.at) {
                self.at += 1;
                if self.world.is_alive(id)
                    && Q::matches(&self.state, id)
                    && let Some(item) = Q::fetch(&self.state, id)
                {
                    return Some(item);
                }
            }
        }
        None
    }
}

impl ReadOnly for Entity {}
impl ReadOnly for NodeRef {}
impl<C: Component> ReadOnly for &C {}
impl<C: Component> ReadOnly for Option<&C> {}
impl<C> ReadOnly for With<C> {}
impl<C> ReadOnly for Without<C> {}
impl<C> ReadOnly for Added<C> {}
impl<C> ReadOnly for Changed<C> {}

/// Terms a whole query can be run with on another thread, as a parallel system
/// does: what they hand out may cross threads. Unlike [`Parallel`] this
/// includes the node, as one system's query runs on one thread.
///
/// # Safety
///
/// What `fetch` gives may be sent to, or shared with, another thread.
pub unsafe trait CrossThread {}
unsafe impl CrossThread for Entity {}
// A node is handed to another thread by reference. In a browser it holds
// WebGPU handles that must stay on their thread, so with wasm threads a
// parallel system cannot take nodes; without them nothing crosses threads.
#[cfg(any(not(target_arch = "wasm32"), not(target_feature = "atomics")))]
unsafe impl CrossThread for Node {}
#[cfg(any(not(target_arch = "wasm32"), not(target_feature = "atomics")))]
unsafe impl CrossThread for NodeRef {}
#[cfg(not(target_arch = "wasm32"))]
const _: fn() = || {
    fn shared_across_threads<T: Send + Sync>() {}
    shared_across_threads::<threers::core::Object3D>();
};
unsafe impl<C: Component + Sync> CrossThread for &C {}
unsafe impl<C: Component + Send> CrossThread for &mut C {}
unsafe impl<C: Component + Sync> CrossThread for Option<&C> {}
unsafe impl<C: Component + Send> CrossThread for Option<&mut C> {}
unsafe impl<C> CrossThread for With<C> {}
unsafe impl<C> CrossThread for Without<C> {}
unsafe impl<C> CrossThread for Added<C> {}
unsafe impl<C> CrossThread for Changed<C> {}

unsafe impl Parallel for Entity {}
unsafe impl<C: Component + Sync> Parallel for &C {}
unsafe impl<C: Component + Send> Parallel for &mut C {}
unsafe impl<C: Component + Sync> Parallel for Option<&C> {}
unsafe impl<C: Component + Send> Parallel for Option<&mut C> {}
unsafe impl<C> Parallel for With<C> {}
unsafe impl<C> Parallel for Without<C> {}
unsafe impl<C> Parallel for Added<C> {}
unsafe impl<C> Parallel for Changed<C> {}

macro_rules! term_tuple {
    ($($name:ident),+) => {
        impl<$($name: Term),+> sealed::Sealed for ($($name,)+) {}
        impl<$($name: ReadOnly),+> ReadOnly for ($($name,)+) {}
        unsafe impl<$($name: Parallel),+> Parallel for ($($name,)+) {}
        unsafe impl<$($name: CrossThread),+> CrossThread for ($($name,)+) {}
        unsafe impl<$($name: Term),+> Term for ($($name,)+) {
            type Item<'w> = ($($name::Item<'w>,)+);
            type State = ($($name::State,)+);
            fn claim(claims: &mut Claims) {
                $($name::claim(claims);)+
            }
            fn access(access: &mut Access) {
                $($name::access(access);)+
            }
            unsafe fn state(world: *mut World) -> Self::State {
                ($(unsafe { $name::state(world) },)+)
            }
            /// The shortest list wins: the query visits only those ids.
            #[allow(non_snake_case)]
            fn candidates(state: &Self::State) -> Option<*const [Entity]> {
                let ($($name,)+) = state;
                let mut best: Option<*const [Entity]> = None;
                $(
                    if let Some(ids) = $name::candidates($name) {
                        if best.is_none_or(|b| ids.len() < b.len()) {
                            best = Some(ids);
                        }
                    }
                )+
                best
            }
            #[allow(non_snake_case)]
            unsafe fn matches(state: &Self::State, id: Entity) -> bool {
                let ($($name,)+) = state;
                // SAFETY: each `matches` asks what this one does of its caller.
                unsafe { $($name::matches($name, id))&&+ }
            }
            #[allow(non_snake_case)]
            unsafe fn fetch<'w>(state: &Self::State, id: Entity) -> Option<Self::Item<'w>> {
                let ($($name,)+) = state;
                Some(($(unsafe { $name::fetch($name, id) }?,)+))
            }
        }
    };
}

term_tuple!(A);
term_tuple!(A, B);
term_tuple!(A, B, C);
term_tuple!(A, B, C, D);
term_tuple!(A, B, C, D, E);
term_tuple!(A, B, C, D, E, F);
term_tuple!(A, B, C, D, E, F, G);
term_tuple!(A, B, C, D, E, F, G, H);
