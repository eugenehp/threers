//! Taps with several fingers, counted: a two-finger tap, a two-finger
//! double tap, a three-finger tap… Fed every touch (any number of fingers
//! at once, each by its id), it reports a [`MultiTap`] each time all the
//! fingers of a tap have lifted: how many were down together, and how many
//! such taps came in a row.
//!
//! A tap is fingers that come down close together in time, stay put (none
//! travels past [`TapConfig::slop`]) and all lift within
//! [`TapConfig::max_duration`]. The next tap continues the count when it has
//! as many fingers, starts within [`TapConfig::max_gap`] of the last lifting
//! and lands near it.
//!
//! ```
//! use threers_ui::multitouch::{MultiTap, TapTracker, TouchPhase};
//! let mut t = TapTracker::default();
//! let mut got = None;
//! for (k, at) in [(0, 0.0), (1, 0.25)] {
//!     t.touch(1, TouchPhase::Down, 100.0, 100.0, at);
//!     t.touch(2, TouchPhase::Down, 140.0, 100.0, at + 0.01);
//!     t.touch(1, TouchPhase::Up, 100.0, 100.0, at + 0.08);
//!     got = t.touch(2, TouchPhase::Up, 140.0, 100.0, at + 0.09);
//!     assert_eq!(got.map(|g| g.count), Some(k + 1));
//! }
//! assert_eq!(got, Some(MultiTap { fingers: 2, count: 2, x: 120.0, y: 100.0 }));
//! ```

use std::collections::HashMap;

/// Where a touch is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TouchPhase {
    Down,
    Moved,
    Up,
    /// The system took the touch (a gesture of its own, a call): no tap.
    Cancelled,
}

/// A tap of `fingers` fingers, the `count`th in a row, about `(x, y)` (where
/// the fingers came down, on average).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiTap {
    pub fingers: u32,
    pub count: u32,
    pub x: f32,
    pub y: f32,
}

/// What still counts as a tap, and as the next of a series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TapConfig {
    /// How far a finger may travel (points).
    pub slop: f32,
    /// From the first finger down to the last up (seconds).
    pub max_duration: f64,
    /// From one tap's last finger up to the next's first down (seconds).
    pub max_gap: f64,
    /// How far apart two taps of a series may land (points).
    pub max_distance: f32,
}

impl Default for TapConfig {
    fn default() -> Self {
        TapConfig { slop: 10.0, max_duration: 0.35, max_gap: 0.35, max_distance: 60.0 }
    }
}

#[derive(Debug, Clone, Copy)]
struct Finger {
    start: (f32, f32),
}

/// The fingers down now (one tap in the making).
#[derive(Debug, Clone)]
struct Group {
    started: f64,
    /// The most fingers down at once.
    most: u32,
    /// Where each finger came down, summed.
    sum: (f32, f32),
    downs: u32,
    /// Still a tap: nothing travelled, nothing was cancelled.
    tap: bool,
}

/// Follows touches and finds multi-finger taps. See the [module](self).
#[derive(Debug, Clone, Default)]
pub struct TapTracker {
    pub config: TapConfig,
    fingers: HashMap<u64, Finger>,
    group: Option<Group>,
    /// The last tap: what it was and when its fingers lifted.
    last: Option<(MultiTap, f64)>,
}

impl TapTracker {
    pub fn new(config: TapConfig) -> Self {
        TapTracker { config, ..Default::default() }
    }

    /// How many fingers are down now.
    pub fn fingers_down(&self) -> usize {
        self.fingers.len()
    }

    /// The most fingers down together since the first of these came down
    /// (0 with none down): above 1, single-finger gestures should stand
    /// aside.
    pub fn most_fingers(&self) -> u32 {
        self.group.as_ref().map_or(0, |g| g.most)
    }

    /// A touch `id` at `(x, y)`, at `time` seconds. A tap when its last
    /// finger lifts.
    pub fn touch(&mut self, id: u64, phase: TouchPhase, x: f32, y: f32, time: f64) -> Option<MultiTap> {
        match phase {
            TouchPhase::Down => {
                let g = self.group.get_or_insert(Group { started: time, most: 0, sum: (0.0, 0.0), downs: 0, tap: true });
                self.fingers.insert(id, Finger { start: (x, y) });
                g.most = g.most.max(self.fingers.len() as u32);
                g.sum.0 += x;
                g.sum.1 += y;
                g.downs += 1;
                None
            }
            TouchPhase::Moved => {
                if let (Some(f), Some(g)) = (self.fingers.get(&id), self.group.as_mut())
                    && (x - f.start.0).hypot(y - f.start.1) > self.config.slop
                {
                    g.tap = false;
                }
                None
            }
            TouchPhase::Up | TouchPhase::Cancelled => {
                let f = self.fingers.remove(&id)?;
                let g = self.group.as_mut()?;
                if phase == TouchPhase::Cancelled || (x - f.start.0).hypot(y - f.start.1) > self.config.slop {
                    g.tap = false;
                }
                if !self.fingers.is_empty() {
                    return None;
                }
                let g = self.group.take()?;
                if !g.tap || time - g.started > self.config.max_duration {
                    self.last = None;
                    return None;
                }
                let (x, y) = (g.sum.0 / g.downs as f32, g.sum.1 / g.downs as f32);
                let count = match self.last {
                    Some((prev, at))
                        if prev.fingers == g.most
                            && g.started - at <= self.config.max_gap
                            && (x - prev.x).hypot(y - prev.y) <= self.config.max_distance =>
                    {
                        prev.count + 1
                    }
                    _ => 1,
                };
                let tap = MultiTap { fingers: g.most, count, x, y };
                self.last = Some((tap, time));
                Some(tap)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use TouchPhase::*;

    fn two(t: &mut TapTracker, at: f64, dx: f32) -> Option<MultiTap> {
        t.touch(1, Down, 100.0 + dx, 100.0, at);
        t.touch(2, Down, 140.0 + dx, 100.0, at + 0.02);
        t.touch(2, Up, 140.0 + dx, 100.0, at + 0.1);
        t.touch(1, Up, 100.0 + dx, 100.0, at + 0.11)
    }

    #[test]
    fn two_finger_taps_count_up() {
        let mut t = TapTracker::default();
        assert_eq!(two(&mut t, 0.0, 0.0).map(|g| (g.fingers, g.count)), Some((2, 1)));
        assert_eq!(two(&mut t, 0.3, 0.0).map(|g| (g.fingers, g.count)), Some((2, 2)));
        assert_eq!(two(&mut t, 0.6, 0.0).map(|g| g.count), Some(3));
    }

    #[test]
    fn too_slow_too_far_or_moved_starts_over() {
        let mut t = TapTracker::default();
        two(&mut t, 0.0, 0.0);
        assert_eq!(two(&mut t, 1.0, 0.0).map(|g| g.count), Some(1), "a long gap");
        assert_eq!(two(&mut t, 1.3, 200.0).map(|g| g.count), Some(1), "somewhere else");
        // fingers that travel: a scroll, not a tap
        t.touch(1, Down, 0.0, 0.0, 2.0);
        t.touch(2, Down, 40.0, 0.0, 2.0);
        t.touch(1, Moved, 0.0, 50.0, 2.05);
        t.touch(1, Up, 0.0, 50.0, 2.1);
        assert_eq!(t.touch(2, Up, 40.0, 0.0, 2.1), None);
        // held: not a tap
        t.touch(1, Down, 0.0, 0.0, 3.0);
        assert_eq!(t.touch(1, Up, 0.0, 0.0, 4.0), None);
    }

    #[test]
    fn one_finger_and_two_finger_taps_are_told_apart() {
        let mut t = TapTracker::default();
        t.touch(7, Down, 10.0, 10.0, 0.0);
        assert_eq!(t.most_fingers(), 1);
        let one = t.touch(7, Up, 10.0, 10.0, 0.05).unwrap();
        assert_eq!((one.fingers, one.count), (1, 1));
        // a two-finger tap right after doesn't continue a one-finger series
        assert_eq!(two(&mut t, 0.2, -90.0).map(|g| (g.fingers, g.count)), Some((2, 1)));
        t.touch(1, Down, 0.0, 0.0, 1.0);
        t.touch(2, Down, 30.0, 0.0, 1.0);
        assert_eq!(t.most_fingers(), 2, "single-finger gestures stand aside");
        assert_eq!(t.touch(1, Cancelled, 0.0, 0.0, 1.05), None);
        assert_eq!(t.touch(2, Up, 30.0, 0.0, 1.06), None, "cancelled: no tap");
    }
}
