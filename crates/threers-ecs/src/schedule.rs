//! Systems, and the order they run in.
//!
//! Systems run one at a time. They share one arena, and on small game logic a
//! thread pool's dispatch costs more than the work, so a parallel scheduler would
//! add machinery without adding speed. What matters here is that the order is
//! stated and stable.

use crate::states::GameState;
use crate::world::World;
use std::any::{type_name, TypeId};

/// Which system this is, for what it has seen: its type, and which of that type.
/// One function in the fixed and the frame schedule shares a place, since the two
/// are the same system; capturing closures of one type do not.
pub(crate) type SystemKey = (TypeId, u32);

#[derive(Clone, Copy, PartialEq)]
struct Label {
    id: TypeId,
    name: &'static str,
}

impl Label {
    fn of<F: 'static>() -> Self {
        Self {
            id: TypeId::of::<F>(),
            name: type_name::<F>(),
        }
    }
}

/// A function over the world, and what it must run before or after.
///
/// Every function has its own type, so ordering refers to the function itself.
/// Rename one and this stops compiling, where a string label would just stop
/// applying.
///
/// ```
/// # use threers_ecs::prelude::*;
/// fn read_input(w: &mut World) {}
/// fn movement(w: &mut World) {}
///
/// let mut app = App::new();
/// app.system(System::new(movement).after(read_input));
/// app.system(read_input);
/// ```
pub struct System {
    label: Label,
    /// How many of this type came before it in its schedule.
    nth: u32,
    /// Whether two of this type can differ: a capturing closure or a function
    /// pointer can, a plain function cannot.
    can_repeat: bool,
    run: Box<dyn FnMut(&mut World)>,
    /// For a parallel system: fetch its parameters and hand back its body.
    prepare: Option<crate::params::Prepare>,
    /// What a parameter system reads and writes.
    access: Option<crate::query::Access>,
    /// Switched off after a caught panic.
    failed: bool,
    before: Vec<Label>,
    after: Vec<Label>,
    after_if_added: Vec<Label>,
    conditions: Vec<Condition>,
}

type Condition = Box<dyn FnMut(&World) -> bool>;

impl System {
    /// A system from a function of the world, of parameters (see
    /// [`params`](crate::params)), or an existing one.
    pub fn new<M, F: crate::params::IntoSystem<M>>(system: F) -> Self {
        system.into_system()
    }

    /// A function of the world, as it is.
    pub(crate) fn of_world<F: FnMut(&mut World) + 'static>(system: F) -> Self {
        Self {
            label: Label::of::<F>(),
            nth: 0,
            can_repeat: std::mem::size_of::<F>() > 0,
            run: Box::new(system),
            prepare: None,
            access: None,
            failed: false,
            before: Vec::new(),
            after: Vec::new(),
            after_if_added: Vec::new(),
            conditions: Vec::new(),
        }
    }

    /// Known by `F`'s type, whatever wraps it, so ordering against the plain
    /// function names it.
    pub(crate) fn labelled<F: 'static>(run: Box<dyn FnMut(&mut World)>) -> Self {
        Self {
            label: Label::of::<F>(),
            nth: 0,
            can_repeat: std::mem::size_of::<F>() > 0,
            run,
            prepare: None,
            access: None,
            failed: false,
            before: Vec::new(),
            after: Vec::new(),
            after_if_added: Vec::new(),
            conditions: Vec::new(),
        }
    }

    /// A system that may run on another thread, at the same time as others
    /// whose parameters touch none of what its parameters change. Every
    /// parameter must be able to cross threads (not `Commands`), and so must
    /// the function: the compiler says if not.
    ///
    /// ```
    /// # use threers_ecs::prelude::*;
    /// struct Velocity(Vector3);
    /// impl Component for Velocity {}
    /// struct Spin(f32);
    /// impl Component for Spin {}
    ///
    /// fn movement(mut q: Query<(Node, &Velocity)>, time: Res<Time>) {
    ///     q.each(|(node, v)| node.position = node.position + v.0 * time.delta);
    /// }
    /// fn spinning(mut q: Query<&mut Spin>, time: Res<Time>) {
    ///     q.each(|spin| spin.0 += time.delta);
    /// }
    /// let mut app = App::new();
    /// // Neither changes what the other uses, so they run side by side.
    /// app.fixed_system(System::parallel(movement)).fixed_system(System::parallel(spinning));
    /// ```
    ///
    /// Side by side means: next to each other in the schedule's order, neither
    /// ordered after the other, both their run conditions true when the first
    /// starts. Without the `parallel` feature, or in a browser without threads,
    /// they run one after another as any system does.
    pub fn parallel<M, F: crate::params::IntoParallel<M>>(system: F) -> Self {
        system.into_parallel()
    }

    pub(crate) fn parallel_from<F: 'static>(
        prepare: crate::params::Prepare,
        access: crate::query::Access,
    ) -> Self {
        let mut system = Self::labelled::<F>(Box::new(|_| {
            unreachable!("a parallel system runs through prepare")
        }));
        system.prepare = Some(prepare);
        system.access = Some(access);
        system
    }

    /// Whether this system can share a batch with others.
    pub fn is_parallel(&self) -> bool {
        self.prepare.is_some()
    }

    pub fn before<M, F: crate::params::IntoSystem<M> + 'static>(mut self, _other: F) -> Self {
        self.before.push(Label::of::<F>());
        self
    }

    pub fn after<M, F: crate::params::IntoSystem<M> + 'static>(mut self, _other: F) -> Self {
        self.after.push(Label::of::<F>());
        self
    }

    /// As [`after`](Self::after), but fine if `other` is not in the schedule: for
    /// a plugin that works with another plugin and without it.
    pub fn after_if_added<M, F: crate::params::IntoSystem<M> + 'static>(
        mut self,
        _other: F,
    ) -> Self {
        self.after_if_added.push(Label::of::<F>());
        self
    }
}

impl System {
    /// Run only while `condition` holds. A system skipped this way keeps its
    /// place in `Added`, `Changed` and events, for as long as they last.
    pub fn run_if(mut self, condition: impl FnMut(&World) -> bool + 'static) -> Self {
        self.conditions.push(Box::new(condition));
        self
    }

    /// Run only in this state.
    pub fn in_state<S: GameState>(self, state: S) -> Self {
        self.run_if(move |w| w.state::<S>() == state)
    }

    /// Run in every state but this one.
    pub fn not_in_state<S: GameState>(self, state: S) -> Self {
        self.run_if(move |w| w.state::<S>() != state)
    }
}

/// Systems in the order they were added, except that one which must wait for
/// another moves later: each turn goes to the first system whose constraints are met.
#[derive(Default)]
pub struct Schedule {
    systems: Vec<System>,
    order: Option<Vec<usize>>,
    /// `reach[a][b]`: `a` must run before `b`, directly or through others.
    reach: Vec<Vec<bool>>,
}

impl Schedule {
    pub fn new() -> Self {
        Self::default()
    }

    /// # Panics
    ///
    /// If this function is already here, which would run the same code twice.
    /// Capturing closures and function pointers can repeat, as two of one type
    /// need not do the same thing; ordering against that type means all of them.
    pub fn add<M>(&mut self, system: impl crate::params::IntoSystem<M>) -> &mut Self {
        let mut system = system.into_system();
        let earlier = self
            .systems
            .iter()
            .filter(|s| s.label == system.label)
            .count();
        assert!(
            system.can_repeat || earlier == 0,
            "{} was added twice",
            system.label.name
        );
        system.nth = earlier as u32;
        self.systems.push(system);
        self.order = None;
        self
    }

    /// Run every system, applying its queued commands before the next starts.
    ///
    /// # Panics
    ///
    /// If the constraints form a cycle or name a system that was never added. It
    /// reports this on the first run, once everything has had a chance to be added.
    pub fn run(&mut self, world: &mut World) {
        let order = match self.order.take() {
            Some(order) => order,
            None => {
                let (order, reach) = self.sort();
                self.reach = reach;
                order
            }
        };
        let mut at = 0;
        while at < order.len() {
            let index = order[at];
            at += 1;
            if !self.ready(index, world) {
                continue;
            }
            if self.systems[index].is_parallel() && crate::clock::threads_available() {
                let mut batch = vec![index];
                while let Some(&next) = order.get(at) {
                    if !self.systems[next].is_parallel() || !self.fits(&batch, next) {
                        break;
                    }
                    at += 1;
                    if self.ready(next, world) {
                        batch.push(next);
                    }
                }
                if batch.len() > 1 {
                    self.run_batch(&batch, world);
                    continue;
                }
            }
            self.run_one(index, world);
        }
        self.order = Some(order);
    }

    /// Not switched off, and its conditions hold.
    fn ready(&mut self, index: usize, world: &World) -> bool {
        let system = &mut self.systems[index];
        !system.failed
            && system
                .conditions
                .iter_mut()
                .all(|condition| condition(world))
    }

    /// Whether `next` can run beside every system in `batch`: no order between
    /// them, and nothing one changes that another uses.
    fn fits(&self, batch: &[usize], next: usize) -> bool {
        let access = |i: usize| self.systems[i].access.as_ref();
        batch.iter().all(|&b| {
            !self.reach[b][next]
                && !self.reach[next][b]
                && match (access(b), access(next)) {
                    (Some(x), Some(y)) => x.conflict(y).is_none(),
                    _ => false,
                }
        })
    }

    fn run_one(&mut self, index: usize, world: &mut World) {
        let system = &mut self.systems[index];
        let key = (system.label.id, system.nth);
        world.enter_system(key);
        let start = crate::clock::now_ms();
        let outcome = if let Some(prepare) = &mut system.prepare {
            let job = prepare(world);
            run_caught(world.catch_panics, job)
        } else if world.catch_panics {
            let run = &mut system.run;
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(world)))
                .map_err(|p| crate::profile::panic_message(&*p))
        } else {
            (system.run)(world);
            Ok(())
        };
        let spent = crate::clock::now_ms() - start;
        self.finish(index, world, spent, outcome);
        world.leave_system(key);
        world.flush();
    }

    /// Fetch each system's parameters here, run the bodies on the pool, and
    /// record them once all have finished.
    fn run_batch(&mut self, batch: &[usize], world: &mut World) {
        let mut jobs = Vec::with_capacity(batch.len());
        for &index in batch {
            let system = &mut self.systems[index];
            let key = (system.label.id, system.nth);
            world.enter_system(key);
            let prepare = system
                .prepare
                .as_mut()
                .expect("only parallel systems batch");
            jobs.push((index, prepare(world)));
            world.leave_system(key);
        }
        let catch = world.catch_panics;
        let results: Vec<(usize, f64, Result<(), String>)> = crate::clock::run_together(
            jobs.into_iter()
                .map(|(index, job)| {
                    move || {
                        let start = crate::clock::now_ms();
                        let outcome = run_caught(catch, job);
                        (index, crate::clock::now_ms() - start, outcome)
                    }
                })
                .collect(),
        );
        for (index, spent, outcome) in results {
            self.finish(index, world, spent, outcome);
        }
        world.flush();
    }

    /// Record a run: its time, and switch it off if it failed.
    fn finish(&mut self, index: usize, world: &mut World, spent: f64, outcome: Result<(), String>) {
        let system = &mut self.systems[index];
        if let Err(message) = outcome {
            if !world.catch_panics {
                panic!("{message}");
            }
            system.failed = true;
            let failure = crate::profile::SystemFailed {
                system: system.label.name,
                message,
            };
            if let Some(failures) = world.get_resource_mut::<crate::profile::Failures>() {
                failures.0.push(failure.clone());
            }
            world.send(failure);
        }
        if let Some(profile) = world.get_resource_mut::<crate::profile::Profile>() {
            profile.record(system.label.name, spent);
        }
    }

    /// Every system of this type, other than `asking`.
    fn indices_of(&self, label: Label, asking: usize) -> Vec<usize> {
        let found: Vec<usize> = (0..self.systems.len())
            .filter(|&i| self.systems[i].label == label)
            .collect();
        assert!(
            !found.is_empty(),
            "{} is ordered against {}, which is not in this schedule",
            self.systems[asking].label.name,
            label.name
        );
        found.into_iter().filter(|&i| i != asking).collect()
    }

    fn sort(&self) -> (Vec<usize>, Vec<Vec<bool>>) {
        let count = self.systems.len();
        // `waits[a]` lists what must run before `a`.
        let mut waits = vec![Vec::new(); count];
        for (index, system) in self.systems.iter().enumerate() {
            for &label in &system.after {
                waits[index].extend(self.indices_of(label, index));
            }
            for &label in &system.after_if_added {
                let found = (0..count).filter(|&i| i != index && self.systems[i].label == label);
                waits[index].extend(found);
            }
            for &label in &system.before {
                for later in self.indices_of(label, index) {
                    waits[later].push(index);
                }
            }
        }

        let mut done = vec![false; count];
        let mut order = Vec::with_capacity(count);
        while order.len() < count {
            let ready = (0..count).find(|&i| !done[i] && waits[i].iter().all(|&w| done[w]));
            let Some(next) = ready else {
                let stuck: Vec<_> = (0..count)
                    .filter(|&i| !done[i])
                    .map(|i| self.systems[i].label.name)
                    .collect();
                panic!("systems are ordered in a cycle: {}", stuck.join(", "));
            };
            done[next] = true;
            order.push(next);
        }
        // What must come before what, through any chain of constraints.
        let mut reach = vec![vec![false; count]; count];
        for &b in &order {
            for &a in &waits[b] {
                reach[a][b] = true;
                for row in reach.iter_mut() {
                    if row[a] {
                        row[b] = true;
                    }
                }
            }
        }
        (order, reach)
    }
}

/// Run a body, catching a panic as a message when asked to.
fn run_caught(catch: bool, job: crate::params::Job) -> Result<(), String> {
    if catch {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(job))
            .map_err(|p| crate::profile::panic_message(&*p))
    } else {
        job();
        Ok(())
    }
}
