//! What each system costs, and which have failed.
//!
//! Every [`App`](crate::App) times its systems into [`Profile`]; the dev panel
//! shows the slowest. With [`App::catch_panics`](crate::App::catch_panics) on, as
//! [`DevTools`](crate::dev::DevTools) turns it, a system that panics is switched
//! off and reported instead of ending the game.

use std::collections::HashMap;

/// A system's cost, smoothed over recent runs.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SystemTime {
    /// Milliseconds a run, averaged over roughly the last thirty.
    pub ms: f64,
    /// The slowest single run lately, decaying back towards `ms`.
    pub peak_ms: f64,
    pub runs: u64,
}

/// Time per system, as a resource.
#[derive(Debug, Default)]
pub struct Profile {
    pub(crate) systems: HashMap<&'static str, SystemTime>,
}

impl Profile {
    pub(crate) fn record(&mut self, name: &'static str, ms: f64) {
        let entry = self.systems.entry(name).or_default();
        let blend = if entry.runs == 0 { 1.0 } else { 1.0 / 30.0 };
        entry.ms += (ms - entry.ms) * blend;
        entry.peak_ms = (entry.peak_ms * 0.98).max(ms).max(entry.ms);
        entry.runs += 1;
    }

    /// Systems, slowest first, by their function's name.
    pub fn slowest(&self) -> Vec<(&'static str, SystemTime)> {
        let mut all: Vec<_> = self.systems.iter().map(|(n, t)| (*n, *t)).collect();
        all.sort_by(|a, b| b.1.ms.total_cmp(&a.1.ms));
        all
    }

    pub fn get(&self, name: &str) -> Option<SystemTime> {
        self.systems
            .iter()
            .find(|(n, _)| short(n) == short(name))
            .map(|(_, t)| *t)
    }

    /// Everything, per run of every system, in milliseconds.
    pub fn total_ms(&self) -> f64 {
        self.systems.values().map(|t| t.ms).sum()
    }
}

/// A system panicked and was switched off.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemFailed {
    pub system: &'static str,
    pub message: String,
}

/// The systems switched off so far, as a resource.
#[derive(Debug, Default)]
pub struct Failures(pub Vec<SystemFailed>);

/// A function's path without its crate and modules.
pub fn short(name: &str) -> &str {
    let name = name.split('<').next().unwrap_or(name);
    name.rsplit("::")
        .find(|part| !part.starts_with('{'))
        .unwrap_or(name)
}

pub(crate) fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "a panic with no message".to_owned())
}
