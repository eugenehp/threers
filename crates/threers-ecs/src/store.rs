//! One component type's rows: a sparse set keyed by `ObjectId`.
//!
//! A `SecondaryMap` alone would be the short answer, but iterating one walks
//! every slot in the arena, so a query for the single `Player` would cost as much
//! as the scene has nodes. Here the values are packed, and a query costs what its
//! rarest component has rows.

use crate::commands::Command;
use crate::world::World;
use slotmap::SecondaryMap;
use std::any::Any;
use std::rc::Rc;
use threers::core::{ObjectArena, ObjectId};

pub(crate) type Hook<T> = Rc<dyn Fn(&mut World, ObjectId, &T)>;

pub(crate) struct Store<T> {
    /// Boxed so a query's pointer to it does not point into the `Store`: two
    /// terms of one query can view a store, and the second borrow of the `Store`
    /// would end the first's claim on anything inside it.
    pub(crate) rows: Box<SecondaryMap<ObjectId, u32>>,
    pub(crate) ids: Vec<ObjectId>,
    pub(crate) values: Vec<T>,
    /// World tick each row was added at, and last borrowed mutably at.
    pub(crate) added: Vec<u64>,
    pub(crate) changed: Vec<u64>,
    pub(crate) hooks: Vec<Hook<T>>,
}

impl<T> Store<T> {
    pub(crate) fn new() -> Self {
        Self {
            rows: Box::default(),
            ids: Vec::new(),
            values: Vec::new(),
            added: Vec::new(),
            changed: Vec::new(),
            hooks: Vec::new(),
        }
    }

    pub(crate) fn insert(&mut self, id: ObjectId, value: T, tick: u64) -> Option<T> {
        if let Some(&row) = self.rows.get(id) {
            self.changed[row as usize] = tick;
            return Some(std::mem::replace(&mut self.values[row as usize], value));
        }
        self.rows.insert(id, self.ids.len() as u32);
        self.ids.push(id);
        self.values.push(value);
        self.added.push(tick);
        self.changed.push(tick);
        None
    }

    pub(crate) fn get(&self, id: ObjectId) -> Option<&T> {
        self.rows.get(id).map(|&row| &self.values[row as usize])
    }

    pub(crate) fn get_mut(&mut self, id: ObjectId, tick: u64) -> Option<&mut T> {
        let row = *self.rows.get(id)? as usize;
        self.changed[row] = tick;
        Some(&mut self.values[row])
    }

    pub(crate) fn remove(&mut self, id: ObjectId) -> Option<T> {
        let row = self.rows.remove(id)?;
        Some(self.take_row(row as usize))
    }

    /// Remove by position, moving the last row into the gap. A dead key may be
    /// the row that moves; `SecondaryMap::insert` ignores a key older than the
    /// slot's, so it cannot overwrite the live entity that reused the slot.
    fn take_row(&mut self, row: usize) -> T {
        self.ids.swap_remove(row);
        self.added.swap_remove(row);
        self.changed.swap_remove(row);
        let value = self.values.swap_remove(row);
        if let Some(&moved) = self.ids.get(row) {
            self.rows.insert(moved, row as u32);
        }
        value
    }

    /// The hooks to run for a value leaving the store, as one call for later:
    /// they take the world, so they cannot run while the store is borrowed.
    fn farewell(&self, id: ObjectId, value: T) -> Option<Command>
    where
        T: 'static,
    {
        if self.hooks.is_empty() {
            return None;
        }
        let hooks = self.hooks.clone();
        Some(Box::new(move |world| {
            for hook in &hooks {
                hook(world, id, &value);
            }
        }))
    }
}

/// What the world needs from a store without knowing its type.
pub(crate) trait AnyStore: Any {
    /// Remove a row, returning its hooks for the caller to run.
    fn discard(&mut self, id: ObjectId) -> Option<Command>;
    /// Drop every row whose node has left the arena, collecting their hooks.
    fn prune(&mut self, arena: &ObjectArena, farewells: &mut Vec<Command>);
}

impl<T: 'static> AnyStore for Store<T> {
    fn discard(&mut self, id: ObjectId) -> Option<Command> {
        let value = self.remove(id)?;
        self.farewell(id, value)
    }

    fn prune(&mut self, arena: &ObjectArena, farewells: &mut Vec<Command>) {
        let mut row = 0;
        while row < self.ids.len() {
            let id = self.ids[row];
            if arena.nodes.contains_key(id) {
                row += 1;
            } else {
                self.rows.remove(id);
                let value = self.take_row(row);
                farewells.extend(self.farewell(id, value));
            }
        }
    }
}
