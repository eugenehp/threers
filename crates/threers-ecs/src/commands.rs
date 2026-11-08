//! Changes queued for when no query is running.

use crate::world::{Bundle, Component, Entity, World};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use threers::core::Object3D;

pub(crate) type Command = Box<dyn FnOnce(&mut World)>;

pub(crate) enum Queued {
    Run(Command),
    Despawn(Entity),
}

/// A queue of changes to the world, shared by every clone.
///
/// A query holds the world exclusively, so it cannot despawn what it is looking
/// at. It queues the change here instead:
///
/// ```
/// # use threers_ecs::prelude::*;
/// # struct Health(f32);
/// # impl Component for Health {}
/// fn remove_dead(w: &mut World) {
///     let commands = w.commands();
///     w.each::<(Entity, &Health)>(|(id, health)| {
///         if health.0 <= 0.0 {
///             commands.despawn(id);
///         }
///     });
/// }
/// ```
#[derive(Clone, Default)]
pub struct Commands {
    queue: Rc<RefCell<VecDeque<Queued>>>,
}

impl Commands {
    /// Queue any change to the world.
    pub fn add(&self, command: impl FnOnce(&mut World) + 'static) {
        self.queue
            .borrow_mut()
            .push_back(Queued::Run(Box::new(command)));
    }

    /// Queue a spawn of `object` under the scene root, with `bundle` attached.
    pub fn spawn(&self, object: Object3D, bundle: impl Bundle) {
        self.add(move |w| {
            let id = w.spawn(object);
            w.insert(id, bundle);
        });
    }

    /// Despawns queued one after another are applied as one batch, which is why
    /// this is not just a closure: removing many siblings one at a time costs
    /// a walk of their parent's children each.
    pub fn despawn(&self, id: Entity) {
        self.queue.borrow_mut().push_back(Queued::Despawn(id));
    }

    /// Queue attaching components. Does nothing if the entity is gone by then.
    pub fn insert(&self, id: Entity, bundle: impl Bundle) {
        self.add(move |w| {
            w.insert(id, bundle);
        });
    }

    /// Queue taking a component off. Does nothing if it is not there.
    pub fn remove<C: Component>(&self, id: Entity) {
        self.add(move |w| {
            w.remove::<C>(id);
        });
    }

    /// Own statement so the queue is not still borrowed while a command runs.
    pub(crate) fn pop(&self) -> Option<Queued> {
        self.queue.borrow_mut().pop_front()
    }

    /// The despawns at the front of the queue, in order.
    pub(crate) fn pop_despawns(&self, into: &mut Vec<Entity>) {
        let mut queue = self.queue.borrow_mut();
        while let Some(Queued::Despawn(id)) = queue.front() {
            into.push(*id);
            queue.pop_front();
        }
    }
}
