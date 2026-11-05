//! The scene, the component tables beside it, and the game's resources.

use crate::commands::{Commands, Queued};
use crate::events::{AnyEvents, EventQueue};
use crate::query::{QueryIter, ReadOnly, Term};
use crate::registry::Registry;
use crate::schedule::SystemKey;
use crate::store::{AnyStore, Store};
use std::any::{type_name, Any, TypeId};
use std::collections::{HashMap, HashSet};
use std::ptr::{addr_of, addr_of_mut};
use std::rc::Rc;
use threers::core::{Object3D, ObjectId};
use threers::math::{Quaternion, Vector3};
use threers::Scene;

/// A scene node, seen as something that can hold components.
pub type Entity = ObjectId;

/// Data that can be attached to an entity.
///
/// ```
/// use threers_ecs::Component;
///
/// struct Health(f32);
/// impl Component for Health {}
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a component",
    label = "used as a component here",
    note = "add `impl Component for {Self} {{}}` beside it"
)]
pub trait Component: 'static {}

/// One component, or a tuple of them, for [`World::insert`].
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be inserted as components",
    note = "insert a component (`impl Component for {Self} {{}}`) or a tuple of up to eight"
)]
pub trait Bundle: 'static {
    #[doc(hidden)]
    fn write(self, world: &mut World, id: Entity);
}

impl<C: Component> Bundle for C {
    fn write(self, world: &mut World, id: Entity) {
        let tick = world.tick;
        if let Some(old) = world.store_mut::<C>().insert(id, self, tick) {
            world.run_hooks(id, &old);
        }
    }
}

macro_rules! bundle_tuple {
    ($($name:ident),+) => {
        impl<$($name: Bundle),+> Bundle for ($($name,)+) {
            #[allow(non_snake_case)]
            fn write(self, world: &mut World, id: Entity) {
                let ($($name,)+) = self;
                $($name.write(world, id);)+
            }
        }
    };
}

bundle_tuple!(A);
bundle_tuple!(A, B);
bundle_tuple!(A, B, C);
bundle_tuple!(A, B, C, D);
bundle_tuple!(A, B, C, D, E);
bundle_tuple!(A, B, C, D, E, F);
bundle_tuple!(A, B, C, D, E, F, G);
bundle_tuple!(A, B, C, D, E, F, G, H);

/// The scene plus everything gameplay keeps beside it.
///
/// `scene` stays public and is the same [`Scene`] the renderer takes, so
/// `renderer.render_to_rgba(&mut world.scene, &camera)` needs no adapter.
pub struct World {
    pub scene: Scene,
    pub(crate) stores: HashMap<TypeId, Box<dyn AnyStore>>,
    resources: HashMap<TypeId, Box<dyn Any>>,
    /// Names of what is in `resources`, and of what `with_resource` has lent out,
    /// for messages.
    resource_names: HashMap<TypeId, &'static str>,
    lent: Vec<&'static str>,
    /// Whether a system's panic is caught and the system switched off.
    pub(crate) catch_panics: bool,
    pub(crate) commands: Commands,
    events: HashMap<TypeId, Box<dyn AnyEvents>>,
    /// Advances as systems run; rows and readers compare against it.
    pub(crate) tick: u64,
    /// The system running now, or `None` outside any system.
    pub(crate) reader: Option<SystemKey>,
    /// When `reader` last ran. `Added` and `Changed` mean "since then".
    pub(crate) since: u64,
    last_run: HashMap<SystemKey, u64>,
    /// Reused by queries that walk the whole arena.
    scratch: Vec<Entity>,
    outside_since: u64,
    pub(crate) registry: Registry,
}

/// What to do about a missing resource, by its type's name.
fn resource_hint(name: &str) -> &'static str {
    // Only this crate's own types: a game's `Physics` is its own business.
    let short = match name.strip_prefix("threers_ecs::") {
        Some(rest) => rest.rsplit("::").next().unwrap_or(rest),
        None => "",
    };
    match short {
        "Time" | "Input" | "Gizmos" | "Delays" | "TuneFiles" | "Recorder" | "Profile" | "Failures" => {
            "App::new() adds it. A bare World::new() has none: use App::new() and app.world"
        }
        "Physics" => "add the physics plugin: app.plugin(PhysicsPlugin::default())",
        "AudioOut" | "Listener" => "add the audio plugin: app.plugin(AudioPlugin::default()), or AudioPlugin::Offline in a test",
        "ChaseCamera" => "add it as a plugin: app.plugin(ChaseCamera::new())",
        "MainCamera" => "a window adds one, and so does the ChaseCamera plugin; or app.resource(MainCamera(PerspectiveCamera::new(..)))",
        "Hud" => "a window adds one. Draw through with_hud(w, |hud| ..) and a game without one still runs",
        "Assets" => "add the assets plugin: app.plugin(AssetsPlugin::new(\"assets\"))",
        "DevState" => "add the dev tools: app.plugin(DevTools)",
        "UiLayer" => "add the UI plugin: app.plugin(UiPlugin) (feature `ui`)",
        _ => "insert it first: app.resource(..) while building, or w.insert_resource(..) in a system",
    }
}

#[cold]
#[track_caller]
fn missing_resource(name: &str, w: &World) -> ! {
    if w.lent.contains(&name) {
        panic!(
            "no resource {name}: it is lent out, as this runs inside with_resource::<{}>. \
             Use the `&mut` that with_resource passed in",
            name.rsplit("::").next().unwrap_or(name)
        );
    }
    let mut have: Vec<&str> = w
        .resource_names
        .values()
        .map(|n| n.rsplit("::").next().unwrap_or(n))
        .collect();
    have.sort_unstable();
    panic!(
        "no resource {name}: {}.\nThe world has: {}",
        resource_hint(name),
        if have.is_empty() {
            "nothing".to_owned()
        } else {
            have.join(", ")
        }
    )
}

impl World {
    pub fn new() -> Self {
        Self::with_scene(Scene::new())
    }

    /// Take over a scene that already has content; each of its nodes can take
    /// components straight away.
    pub fn with_scene(scene: Scene) -> Self {
        Self {
            scene,
            stores: HashMap::new(),
            resources: HashMap::new(),
            resource_names: HashMap::new(),
            lent: Vec::new(),
            catch_panics: false,
            commands: Commands::default(),
            events: HashMap::new(),
            tick: 1,
            reader: None,
            since: 0,
            last_run: HashMap::new(),
            scratch: Vec::new(),
            outside_since: 0,
            registry: Registry::default(),
        }
    }

    /// Add `object` under the scene root.
    pub fn spawn(&mut self, object: Object3D) -> Entity {
        self.scene.add(object)
    }

    /// Add `object` under `parent`, so it moves with it and despawns with it.
    pub fn spawn_under(&mut self, parent: Entity, object: Object3D) -> Entity {
        self.scene.add_to(parent, object)
    }

    /// An entity that is only data, such as a quest or a timer. Its node is in
    /// the arena but under no root, so nothing walks it and nothing draws it.
    pub fn spawn_data(&mut self) -> Entity {
        self.scene.arena.insert(Object3D::group())
    }

    pub fn is_alive(&self, id: Entity) -> bool {
        self.scene.arena.nodes.contains_key(id)
    }

    /// Attach components, replacing any of the same type. Returns false, and
    /// stores nothing, if the entity is dead: a command queued for something
    /// that has since been despawned is ordinary, not an error.
    pub fn insert(&mut self, id: Entity, bundle: impl Bundle) -> bool {
        if !self.is_alive(id) {
            return false;
        }
        bundle.write(self, id);
        true
    }

    pub fn remove<C: Component>(&mut self, id: Entity) -> Option<C> {
        let value = self.store_mut_if_any::<C>()?.remove(id)?;
        self.run_hooks(id, &value);
        Some(value)
    }

    /// Run `hook` whenever a `C` stops being attached: removed, replaced by
    /// [`insert`](Self::insert), or gone with its entity. This is how a
    /// component that owns something elsewhere, such as a rigid body, lets go of
    /// it however the component left.
    ///
    /// For a despawn or a [`prune`](Self::prune) the entity is already dead when
    /// the hook runs, so it gets the value and the id but no node.
    pub fn on_remove<C: Component>(&mut self, hook: impl Fn(&mut World, Entity, &C) + 'static) {
        self.store_mut::<C>().hooks.push(Rc::new(hook));
    }

    pub fn get<C: Component>(&self, id: Entity) -> Option<&C> {
        self.store::<C>()?.get(id)
    }

    pub fn get_mut<C: Component>(&mut self, id: Entity) -> Option<&mut C> {
        let tick = self.tick;
        self.store_mut_if_any::<C>()?.get_mut(id, tick)
    }

    pub fn has<C: Component>(&self, id: Entity) -> bool {
        self.get::<C>(id).is_some()
    }

    /// Remove an entity, its descendants and all their components. Returns false
    /// if it was already dead, or is the scene root, which the scene needs.
    pub fn despawn(&mut self, id: Entity) -> bool {
        self.despawn_many(&[id]) == 1
    }

    /// [`despawn`](Self::despawn) for many entities at once, returning how many
    /// were alive. Far cheaper than a loop when they share a parent: each parent
    /// has its children walked once, not once per entity.
    pub fn despawn_many(&mut self, ids: &[Entity]) -> usize {
        let root = self.scene.root;
        let live: Vec<Entity> = ids
            .iter()
            .copied()
            .filter(|&id| id != root && self.is_alive(id))
            .collect();
        let count = live.iter().collect::<HashSet<_>>().len();
        let mut farewells = Vec::new();
        for id in self.scene.arena.delete_subtrees(&live) {
            for store in self.stores.values_mut() {
                farewells.extend(store.discard(id));
            }
        }
        for farewell in farewells {
            farewell(self);
        }
        count
    }

    /// Drop component rows for nodes removed behind the world's back, through
    /// [`ObjectArena::delete`](threers::core::ObjectArena::delete). Costs one
    /// check per row, so it is here to call when needed, not run every frame.
    pub fn prune(&mut self) {
        let mut farewells = Vec::new();
        for store in self.stores.values_mut() {
            store.prune(&self.scene.arena, &mut farewells);
        }
        for farewell in farewells {
            farewell(self);
        }
    }

    /// Call `f` for each entity matching `Q`.
    ///
    /// A closure rather than an iterator because only one match may be borrowed
    /// at a time: `SlotMap::get_mut` reborrows every slot, which would invalidate
    /// a `&mut Object3D` an iterator had already handed out.
    ///
    /// ```
    /// # use threers::prelude::*;
    /// # use threers_ecs::prelude::*;
    /// # struct Health(f32);
    /// # impl Component for Health {}
    /// # struct Enemy;
    /// # impl Component for Enemy {}
    /// # let mut w = World::new();
    /// w.each::<(Entity, &mut Health, With<Enemy>)>(|(id, health, _)| {
    ///     health.0 -= 1.0;
    /// });
    /// ```
    ///
    /// # Panics
    ///
    /// If `Q` borrows a component twice, or the node twice, since that would be
    /// two views of one value. Filters do not count.
    pub fn each<Q: Term>(&mut self, f: impl for<'a> FnMut(Q::Item<'a>)) {
        let mut claims = Vec::new();
        Q::claim(&mut claims);
        // SAFETY: `self` is borrowed exclusively for the whole call, and `f`
        // cannot reach the world.
        unsafe { each_raw::<Q>(self, f) }
    }

    /// [`each`](Self::each) for queries that only read, so it needs only `&self`.
    /// Use [`NodeRef`](crate::NodeRef) in place of `Node`.
    pub fn each_ref<Q: Term + ReadOnly>(&self, mut f: impl for<'a> FnMut(Q::Item<'a>)) {
        let mut claims = Vec::new();
        Q::claim(&mut claims);
        let world = self as *const World as *mut World;
        // SAFETY: as for `each`. `ReadOnly` terms only ever read through these
        // pointers, so sharing the world is enough.
        unsafe {
            let state = Q::state(world);
            let mut visit = |id: Entity| {
                if self.is_alive(id) && Q::matches(&state, id) {
                    if let Some(item) = Q::fetch(&state, id) {
                        f(item);
                    }
                }
            };
            match Q::candidates(&state) {
                Some(ids) => (&*ids).iter().copied().for_each(&mut visit),
                None => self.scene.arena.nodes.keys().for_each(&mut visit),
            }
        }
    }

    /// The first entity matching `Q`, for what there is one of: the player, the
    /// boss. `None` if nothing matches.
    ///
    /// ```
    /// # use threers::prelude::*;
    /// # use threers_ecs::prelude::*;
    /// struct Player;
    /// impl Component for Player {}
    /// struct Health(f32);
    /// impl Component for Health {}
    ///
    /// let mut w = World::new();
    /// let id = w.object(Object3D::group()).with((Player, Health(3.0))).id();
    /// let (found, health, _) = w.single::<(Entity, &Health, With<Player>)>().unwrap();
    /// assert_eq!((found, health.0), (id, 3.0));
    /// ```
    pub fn single<Q: Term + ReadOnly>(&self) -> Option<Q::Item<'_>> {
        let mut claims = Vec::new();
        Q::claim(&mut claims);
        let world = self as *const World as *mut World;
        // SAFETY: as for `each_ref`. The item borrows from stores that `&self`
        // keeps as they are for as long as it lives.
        unsafe {
            let state = Q::state(world);
            let accept = |id: Entity| self.is_alive(id) && Q::matches(&state, id);
            let id = match Q::candidates(&state) {
                Some(ids) => (&*ids).iter().copied().find(|&id| accept(id)),
                None => self.scene.arena.nodes.keys().find(|&id| accept(id)),
            };
            Q::fetch(&state, id?)
        }
    }

    /// Everything matching `Q`, as an iterator, for `filter`, `map` and the rest.
    /// [`each`](Self::each) is the way to change things.
    ///
    /// ```
    /// # use threers::prelude::*;
    /// # use threers_ecs::prelude::*;
    /// struct Health(f32);
    /// impl Component for Health {}
    ///
    /// let mut w = World::new();
    /// w.object(Object3D::group()).with(Health(3.0));
    /// w.object(Object3D::group()).with(Health(0.0));
    /// let alive = w.iter::<&Health>().filter(|health| health.0 > 0.0).count();
    /// assert_eq!(alive, 1);
    /// ```
    pub fn iter<Q: Term + ReadOnly>(&self) -> QueryIter<'_, Q> {
        QueryIter::new(self)
    }

    /// `Q` for one entity, if it matches.
    ///
    /// ```
    /// # use threers::prelude::*;
    /// # use threers_ecs::prelude::*;
    /// struct Health(f32);
    /// impl Component for Health {}
    /// struct Name(&'static str);
    /// impl Component for Name {}
    ///
    /// let mut w = World::new();
    /// let id = w.object(Object3D::group()).with((Health(3.0), Name("Bo"))).id();
    /// let (health, name) = w.read::<(&Health, &Name)>(id).unwrap();
    /// assert_eq!((health.0, name.0), (3.0, "Bo"));
    /// ```
    pub fn read<Q: Term + ReadOnly>(&self, id: Entity) -> Option<Q::Item<'_>> {
        Q::claim(&mut Vec::new());
        let world = self as *const World as *mut World;
        // SAFETY: as for `each_ref`, and as for `single` on how long it lasts.
        unsafe {
            let state = Q::state(world);
            (self.is_alive(id) && Q::matches(&state, id)).then(|| Q::fetch(&state, id))?
        }
    }

    /// Run `f` on `Q` for one entity, if it matches, and return what it gives.
    pub fn update<Q: Term, R>(
        &mut self,
        id: Entity,
        f: impl for<'a> FnOnce(Q::Item<'a>) -> R,
    ) -> Option<R> {
        Q::claim(&mut Vec::new());
        let alive = self.is_alive(id);
        let world: *mut World = self;
        // SAFETY: as for `each`, for one row.
        unsafe {
            let state = Q::state(world);
            (alive && Q::matches(&state, id)).then(|| Q::fetch(&state, id).map(f))?
        }
    }

    /// An entity's node.
    pub fn node(&self, id: Entity) -> Option<&Object3D> {
        self.scene.get(id)
    }

    /// An entity's node, to change.
    pub fn node_mut(&mut self, id: Entity) -> Option<&mut Object3D> {
        self.scene.get_mut(id)
    }

    /// Where a node is in the world and how it is turned, through its parents.
    /// Worked out now, so it does not lag behind as `matrix_world` can.
    pub fn world_transform(&self, id: Entity) -> Option<(Vector3, Quaternion)> {
        let node = self.scene.get(id)?;
        let (mut position, mut rotation) = (node.position, node.quaternion);
        let mut next = node.parent;
        while let Some(parent) = next.and_then(|id| self.scene.get(id)) {
            let scaled = Vector3::new(
                position.x * parent.scale.x,
                position.y * parent.scale.y,
                position.z * parent.scale.z,
            );
            position = parent.position + scaled.apply_quaternion(parent.quaternion);
            rotation = parent.quaternion.multiply(rotation);
            next = parent.parent;
        }
        Some((position, rotation))
    }

    /// How many entities match `Q`.
    pub fn count<Q: Term + ReadOnly>(&self) -> usize {
        let mut count = 0;
        self.each_ref::<Q>(|_| count += 1);
        count
    }

    /// The entities matching `Q`, for when the work needs the whole `World`.
    pub fn entities<Q: Term + ReadOnly>(&self) -> Vec<Entity> {
        let mut found = Vec::new();
        self.each_ref::<(Entity, Q)>(|(id, _)| found.push(id));
        found
    }

    /// A handle for changes that cannot happen inside a query. It owns its
    /// queue, so it can be captured by the closure passed to [`World::each`].
    pub fn commands(&self) -> Commands {
        self.commands.clone()
    }

    /// Apply queued commands, oldest first, including any they queue themselves.
    /// [`Schedule`](crate::Schedule) does this after every system.
    pub fn flush(&mut self) {
        while let Some(command) = self.commands.pop() {
            match command {
                Queued::Run(command) => command(self),
                Queued::Despawn(id) => {
                    let mut ids = vec![id];
                    self.commands.pop_despawns(&mut ids);
                    self.despawn_many(&ids);
                }
            }
        }
    }

    /// Publish an event. Every system that reads this type sees it once, whenever
    /// it next runs. It lasts [`EVENT_FRAMES`](crate::EVENT_FRAMES) frames, so a
    /// system that does not run for longer misses it.
    pub fn send<T: 'static>(&mut self, event: T) {
        self.events
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(EventQueue::<T>::default()))
            .as_any_mut()
            .downcast_mut::<EventQueue<T>>()
            .expect("queue kept under its own type")
            .push(event);
    }

    /// Events of this type the running system has not seen yet. Each system keeps
    /// its own place, in fixed and variable schedules alike, so the frame rate
    /// cannot lose or repeat one. Code outside any system shares a place of its own.
    pub fn events<T: Clone + 'static>(&mut self) -> Vec<T> {
        let reader = self.reader;
        self.events
            .get_mut(&TypeId::of::<T>())
            .and_then(|queue| queue.as_any_mut().downcast_mut::<EventQueue<T>>())
            .map_or_else(Vec::new, |queue| queue.read(reader))
    }

    /// [`events`](Self::events) without copying: `f` is lent each event this
    /// system has not seen. For events that are large, or many, like contacts.
    /// `f` cannot reach the world; collect what it needs and act after.
    ///
    /// ```
    /// # use threers_ecs::prelude::*;
    /// #[derive(Clone)]
    /// struct Hit(f32);
    /// let mut w = World::new();
    /// w.send(Hit(2.0));
    /// w.send(Hit(3.0));
    /// let mut total = 0.0;
    /// w.read_events::<Hit>(|hit| total += hit.0);
    /// assert_eq!(total, 5.0);
    /// ```
    pub fn read_events<T: 'static>(&mut self, f: impl FnMut(&T)) {
        let reader = self.reader;
        if let Some(queue) = self
            .events
            .get_mut(&TypeId::of::<T>())
            .and_then(|queue| queue.as_any_mut().downcast_mut::<EventQueue<T>>())
        {
            queue.read_with(reader, f);
        }
    }

    /// Age events by a frame. [`App`](crate::App) calls this.
    pub fn end_frame(&mut self) {
        for queue in self.events.values_mut() {
            queue.end_frame();
        }
    }

    /// Start a new "since" for `Added` and `Changed` in code outside any system.
    pub fn clear_trackers(&mut self) {
        self.outside_since = self.tick;
        self.tick += 1;
        if self.reader.is_none() {
            self.since = self.outside_since;
        }
    }

    pub(crate) fn enter_system(&mut self, system: SystemKey) {
        self.tick += 1;
        self.reader = Some(system);
        self.since = self.last_run.get(&system).copied().unwrap_or(0);
    }

    pub(crate) fn leave_system(&mut self, system: SystemKey) {
        self.last_run.insert(system, self.tick);
        // Past the system's own writes, so what happens next counts as later.
        self.tick += 1;
        self.reader = None;
        self.since = self.outside_since;
    }

    pub fn insert_resource<R: 'static>(&mut self, resource: R) -> Option<R> {
        self.resource_names
            .insert(TypeId::of::<R>(), type_name::<R>());
        self.resources
            .insert(TypeId::of::<R>(), Box::new(resource))
            .map(|old| *old.downcast().expect("resource stored under its own type"))
    }

    pub fn remove_resource<R: 'static>(&mut self) -> Option<R> {
        self.resource_names.remove(&TypeId::of::<R>());
        self.resources
            .remove(&TypeId::of::<R>())
            .map(|old| *old.downcast().expect("resource stored under its own type"))
    }

    pub fn get_resource<R: 'static>(&self) -> Option<&R> {
        self.resources.get(&TypeId::of::<R>())?.downcast_ref()
    }

    pub fn get_resource_mut<R: 'static>(&mut self) -> Option<&mut R> {
        self.resources.get_mut(&TypeId::of::<R>())?.downcast_mut()
    }

    /// # Panics
    ///
    /// If the resource was never inserted.
    #[track_caller]
    pub fn resource<R: 'static>(&self) -> &R {
        match self.get_resource() {
            Some(resource) => resource,
            None => missing_resource(type_name::<R>(), self),
        }
    }

    /// # Panics
    ///
    /// If the resource was never inserted.
    #[track_caller]
    pub fn resource_mut<R: 'static>(&mut self) -> &mut R {
        if !self.resources.contains_key(&TypeId::of::<R>()) {
            missing_resource(type_name::<R>(), self);
        }
        self.get_resource_mut().expect("checked above")
    }

    /// Move every resource `other` has and this world does not into this one:
    /// what a host added beside the game, when the game's world is swapped.
    pub(crate) fn adopt_missing_resources(&mut self, other: &mut World) {
        let missing: Vec<TypeId> = other
            .resources
            .keys()
            .filter(|k| !self.resources.contains_key(k))
            .copied()
            .collect();
        for key in missing {
            if let Some(value) = other.resources.remove(&key) {
                if let Some(name) = other.resource_names.remove(&key) {
                    self.resource_names.insert(key, name);
                }
                self.resources.insert(key, value);
            }
        }
    }

    /// Move one resource across, replacing this world's.
    pub(crate) fn take_resource_from<R: 'static>(&mut self, other: &mut World) {
        if let Some(value) = other.remove_resource::<R>() {
            self.insert_resource(value);
        }
    }

    /// Lend `f` a resource and the world together, which a query needs when it
    /// writes to something like a score. The resource is out of the world during
    /// the call, and goes back afterwards.
    ///
    /// # Panics
    ///
    /// If the resource was never inserted, or `f` inserts one of the same type.
    #[track_caller]
    pub fn with_resource<R: 'static, T>(&mut self, f: impl FnOnce(&mut World, &mut R) -> T) -> T {
        let Some(resource) = self.remove_resource::<R>() else {
            missing_resource(type_name::<R>(), self)
        };
        self.lent.push(type_name::<R>());
        /// Puts the resource back if `f` panics, so a caught panic leaves the
        /// world whole rather than missing whatever was lent.
        struct Lent<R: 'static> {
            world: *mut World,
            resource: Option<R>,
        }
        impl<R: 'static> Drop for Lent<R> {
            fn drop(&mut self) {
                if let Some(resource) = self.resource.take() {
                    // SAFETY: only reached while unwinding out of `f`, when no
                    // borrow of the world taken inside it is alive any more.
                    let world = unsafe { &mut *self.world };
                    world.lent.pop();
                    if world.get_resource::<R>().is_none() {
                        world.insert_resource(resource);
                    }
                }
            }
        }
        let mut lent = Lent {
            world: self as *mut World,
            resource: Some(resource),
        };
        // SAFETY: `lent.world` is `self`, which `f` gets as `&mut` for exactly
        // this call; the guard touches it only after `f` is done.
        let result = f(
            unsafe { &mut *lent.world },
            lent.resource.as_mut().expect("just put there"),
        );
        let resource = lent.resource.take().expect("not yet given back");
        std::mem::forget(lent);
        self.lent.pop();
        assert!(
            self.insert_resource(resource).is_none(),
            "{} was inserted while lent out",
            type_name::<R>()
        );
        result
    }

    fn run_hooks<C: Component>(&mut self, id: Entity, value: &C) {
        for hook in self.store_mut::<C>().hooks.clone() {
            hook(self, id, value);
        }
    }

    pub(crate) fn entities_ref<C: Component>(&self) -> Vec<Entity> {
        self.store::<C>()
            .map_or_else(Vec::new, |store| store.ids.clone())
            .into_iter()
            .filter(|&id| self.is_alive(id))
            .collect()
    }

    fn store<C: Component>(&self) -> Option<&Store<C>> {
        let store: &dyn Any = self.stores.get(&TypeId::of::<C>())?.as_ref();
        store.downcast_ref()
    }

    fn store_mut_if_any<C: Component>(&mut self) -> Option<&mut Store<C>> {
        let store: &mut dyn Any = self.stores.get_mut(&TypeId::of::<C>())?.as_mut();
        store.downcast_mut()
    }

    pub(crate) fn store_mut<C: Component>(&mut self) -> &mut Store<C> {
        let store: &mut dyn Any = self
            .stores
            .entry(TypeId::of::<C>())
            .or_insert_with(|| Box::new(Store::<C>::new()))
            .as_mut();
        store.downcast_mut().expect("store kept under its own type")
    }
}

/// The loop behind [`World::each`], through a raw world: it touches only the
/// stores, the arena and the scratch list, so a system parameter can run it
/// while another holds a resource.
///
/// # Safety
///
/// `world` is valid, nothing else is using the stores or the arena `Q` claims,
/// and `Q` has passed its claim check. Each item is dropped before the next
/// fetch, because the bound on `f` stops it escaping, so no two are ever live.
pub(crate) unsafe fn each_raw<Q: Term>(world: *mut World, f: impl for<'a> FnMut(Q::Item<'a>)) {
    let state = Q::state(world);
    each_with_state::<Q>(world, &state, f, true)
}

/// As [`each_raw`], with the views already taken. Without `scratch` it lists a
/// whole-arena query's ids in a vector of its own, which is what lets several
/// queries run at once on different threads.
///
/// # Safety
///
/// As for [`each_raw`], and `state` was taken from `world` for this query.
pub(crate) unsafe fn each_with_state<Q: Term>(
    world: *mut World,
    state: &Q::State,
    mut f: impl for<'a> FnMut(Q::Item<'a>),
    scratch: bool,
) {
    let arena = addr_of!((*world).scene.arena);
    let mut visit = |id: Entity| {
        if (*arena).nodes.contains_key(id) && Q::matches(state, id) {
            if let Some(item) = Q::fetch(state, id) {
                f(item);
            }
        }
    };
    match Q::candidates(state) {
        Some(ids) => {
            for index in 0..ids.len() {
                visit((&*ids)[index]);
            }
        }
        None if scratch => {
            let scratch = addr_of_mut!((*world).scratch);
            let mut ids = std::mem::take(&mut *scratch);
            ids.clear();
            ids.extend((*arena).nodes.keys());
            ids.iter().copied().for_each(&mut visit);
            *scratch = ids;
        }
        None => {
            let ids: Vec<Entity> = (*arena).nodes.keys().collect();
            ids.into_iter().for_each(&mut visit);
        }
    }
}

/// A resource's address, through a raw world: only the resource table is
/// touched, so other parameters' borrows elsewhere in the world stay valid.
///
/// # Panics
///
/// If there is no such resource, with the same advice as [`World::resource`].
///
/// # Safety
///
/// `world` is valid and nothing else borrows this resource.
#[track_caller]
pub(crate) unsafe fn resource_ptr<R: 'static>(world: *mut World) -> *mut R {
    let table = &mut *std::ptr::addr_of_mut!((*world).resources);
    match table
        .get_mut(&TypeId::of::<R>())
        .and_then(|b| b.downcast_mut::<R>())
    {
        Some(r) => r,
        None => missing_resource(type_name::<R>(), &*world),
    }
}

/// An event type's queue, made if `create` and missing. Null when missing and
/// not to be made.
///
/// # Safety
///
/// As for [`resource_ptr`], for the event table.
pub(crate) unsafe fn event_queue_ptr<T: 'static>(
    world: *mut World,
    create: bool,
) -> *mut EventQueue<T> {
    let table = &mut *std::ptr::addr_of_mut!((*world).events);
    if create {
        table
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(EventQueue::<T>::default()));
    }
    match table
        .get_mut(&TypeId::of::<T>())
        .and_then(|q| q.as_any_mut().downcast_mut::<EventQueue<T>>())
    {
        Some(q) => q,
        None => std::ptr::null_mut(),
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}
