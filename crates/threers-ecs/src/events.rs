//! Messages between systems.
//!
//! A fixed system and a variable one run a different number of times per frame,
//! so "this frame's events" would lose some and repeat others. Instead every
//! system keeps its own place in the queue, and reads what it has not read yet.

use crate::schedule::SystemKey;
use std::any::Any;
use std::collections::{HashMap, VecDeque};

/// How many frames an event stays readable. Long enough for a fixed system on a
/// display several times faster than the fixed rate; a system that stays gated
/// off for longer misses what it slept through.
pub const EVENT_FRAMES: u32 = 30;

pub(crate) trait AnyEvents: Any {
    fn end_frame(&mut self);
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub(crate) struct EventQueue<T> {
    /// Sequence number of `items[0]`.
    first: u64,
    /// Each event, and how many frames it has left.
    items: VecDeque<(T, u32)>,
    places: HashMap<Option<SystemKey>, u64>,
}

impl<T> Default for EventQueue<T> {
    fn default() -> Self {
        Self {
            first: 0,
            items: VecDeque::new(),
            places: HashMap::new(),
        }
    }
}

impl<T> EventQueue<T> {
    pub(crate) fn push(&mut self, event: T) {
        self.items.push_back((event, EVENT_FRAMES));
    }
}

impl<T: Clone> EventQueue<T> {
    pub(crate) fn read(&mut self, reader: Option<SystemKey>) -> Vec<T> {
        let end = self.first + self.items.len() as u64;
        let place = self.places.entry(reader).or_insert(self.first);
        let start = (*place).max(self.first);
        *place = end;
        self.items
            .iter()
            .skip((start - self.first) as usize)
            .map(|(event, _)| event.clone())
            .collect()
    }
}

impl<T> EventQueue<T> {
    /// As `read`, lending each event rather than copying it.
    pub(crate) fn read_with(&mut self, reader: Option<SystemKey>, mut f: impl FnMut(&T)) {
        let end = self.first + self.items.len() as u64;
        let place = self.places.entry(reader).or_insert(self.first);
        let start = (*place).max(self.first);
        *place = end;
        for (event, _) in self.items.iter().skip((start - self.first) as usize) {
            f(event);
        }
    }
}

impl<T: 'static> AnyEvents for EventQueue<T> {
    fn end_frame(&mut self) {
        for (_, frames) in &mut self.items {
            *frames -= 1;
        }
        while self.items.front().is_some_and(|(_, frames)| *frames == 0) {
            self.items.pop_front();
            self.first += 1;
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
