//! Timelines: many tracks sharing one clock.
//!
//! A timeline holds *when* things happen; it does not hold the things. Each
//! track is a start time, a duration and a curve, and you ask it for a track's
//! progress and apply that yourself. Keeping the data and the effect separate is
//! what lets a timeline be inspected, seeked, scrubbed backwards and tested —
//! none of which is possible once callbacks own the state.
//!
//! ```
//! use threers_animation::prelude::*;
//!
//! let mut intro = Timeline::new();
//! let slide = intro.add(0.0, 1.0, Easing::CubicOut);
//! let fade  = intro.add(0.5, 1.0, Easing::Linear);   // overlaps the slide
//!
//! intro.update(0.75);
//! assert!(intro.progress_of(slide) > 0.0);
//! assert!(intro.progress_of(fade) > 0.0);
//!
//! // Read a value straight off a track.
//! let x: f32 = intro.value_of(slide, 0.0, 100.0);
//! assert!(x > 0.0 && x < 100.0);
//! ```

use crate::animatable::Animatable;
use crate::easing::Easing;
use crate::tween::Repeat;

/// Handle to a track within a [`Timeline`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TrackId(usize);

#[derive(Debug, Clone, Copy, PartialEq)]
struct Track {
    start: f32,
    duration: f32,
    easing: Easing,
}

/// A clock with tracks hung off it.
#[derive(Debug, Clone, Default)]
pub struct Timeline {
    tracks: Vec<Track>,
    markers: Vec<(f32, String)>,
    time_remap: Option<fn(f32) -> f32>,
    time: f32,
    duration: f32,
    /// Playback rate. Negative runs the timeline backwards.
    pub speed: f32,
    pub repeat: Repeat,
    playing: bool,
    cycles_done: u32,
    /// Unwrapped position, so a loop can count its laps.
    ///
    /// `time` is wrapped into `0..duration`, so advancing from it can never
    /// cross more than one boundary and the lap counter would never get past 1.
    elapsed: f32,
}

impl Timeline {
    pub fn new() -> Self {
        Self {
            tracks: Vec::new(),
            markers: Vec::new(),
            time_remap: None,
            time: 0.0,
            duration: 0.0,
            speed: 1.0,
            repeat: Repeat::Once,
            playing: true,
            cycles_done: 0,
            elapsed: 0.0,
        }
    }

    /// Add a track starting at `start`, lasting `duration`.
    ///
    /// The timeline's own duration grows to cover it.
    pub fn add(&mut self, start: f32, duration: f32, easing: Easing) -> TrackId {
        let track = Track {
            start: start.max(0.0),
            duration: duration.max(0.0),
            easing,
        };
        self.duration = self.duration.max(track.start + track.duration);
        self.tracks.push(track);
        TrackId(self.tracks.len() - 1)
    }

    /// Add a track that begins as the previous one ends. Chaining, without
    /// having to add up durations by hand.
    pub fn then(&mut self, duration: f32, easing: Easing) -> TrackId {
        self.add(self.duration, duration, easing)
    }

    /// Add a track starting at the same time as `other`.
    pub fn alongside(&mut self, other: TrackId, duration: f32, easing: Easing) -> TrackId {
        let start = self.tracks.get(other.0).map_or(0.0, |t| t.start);
        self.add(start, duration, easing)
    }

    /// Named marker on the timeline clock (editorial cue / camera bind point).
    pub fn add_marker(&mut self, time: f32, name: impl Into<String>) {
        self.markers.push((time.max(0.0), name.into()));
        self.markers
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Optional remapping applied after seek wrapping (`t → t'`).
    pub fn set_time_remap(&mut self, remap: Option<fn(f32) -> f32>) {
        self.time_remap = remap;
    }

    /// Latest marker at or before the current time.
    pub fn marker_at(&self, time: f32) -> Option<&str> {
        let mut hit = None;
        for (t, name) in &self.markers {
            if *t <= time {
                hit = Some(name.as_str());
            } else {
                break;
            }
        }
        hit
    }

    pub fn update(&mut self, dt: f32) {
        if !self.playing || !dt.is_finite() {
            return;
        }
        self.seek(self.elapsed + dt * self.speed);
    }

    /// Jump to an absolute time, applying the repeat mode.
    pub fn seek(&mut self, time: f32) {
        if !time.is_finite() {
            return;
        }
        if self.duration <= 0.0 {
            self.time = 0.0;
            self.elapsed = 0.0;
            return;
        }

        let limit = match self.repeat {
            Repeat::Once => Some(1),
            Repeat::Times(n) => Some(n + 1),
            Repeat::Forever => None,
        };

        let mut t = time;
        if t < 0.0 {
            // Running backwards past the start.
            match limit {
                Some(_) => {
                    t = 0.0;
                    self.elapsed = 0.0;
                    self.cycles_done = 0;
                }
                None => {
                    let cycles = (-t / self.duration).ceil();
                    t += cycles * self.duration;
                }
            }
        }

        let cycle = (t / self.duration) as u32;
        match limit {
            Some(max) if cycle >= max => {
                self.time = self.duration;
                self.elapsed = max as f32 * self.duration;
                self.cycles_done = max - 1;
            }
            _ => {
                self.elapsed = t;
                self.cycles_done = cycle;
                self.time = t % self.duration;
            }
        }
        if let Some(remap) = self.time_remap {
            self.time = remap(self.time);
        }
    }

    /// Eased progress of a track, `0..1`.
    ///
    /// `0` before it starts and `1` after it ends, so applying it unconditionally
    /// is safe — a track that has not begun simply holds its start value.
    pub fn progress_of(&self, track: TrackId) -> f32 {
        let Some(t) = self.tracks.get(track.0) else {
            return 0.0;
        };
        if t.duration <= 0.0 {
            return if self.time >= t.start { 1.0 } else { 0.0 };
        }
        let local = (self.time - t.start) / t.duration;
        t.easing.apply(local.clamp(0.0, 1.0))
    }

    /// Interpolate between two values using a track's progress.
    pub fn value_of<T: Animatable>(&self, track: TrackId, from: T, to: T) -> T {
        from.lerp(to, self.progress_of(track))
    }

    /// Whether a track is between its start and end right now. Useful for
    /// firing an effect exactly while a segment runs.
    pub fn is_active(&self, track: TrackId) -> bool {
        let Some(t) = self.tracks.get(track.0) else {
            return false;
        };
        self.time >= t.start && self.time <= t.start + t.duration
    }

    pub fn time(&self) -> f32 {
        self.time
    }

    /// Total length: the end of the last track.
    pub fn duration(&self) -> f32 {
        self.duration
    }

    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    pub fn cycles(&self) -> u32 {
        self.cycles_done
    }

    pub fn is_finished(&self) -> bool {
        match self.repeat {
            Repeat::Forever => false,
            _ => self.duration > 0.0 && self.time >= self.duration,
        }
    }

    pub fn play(&mut self) {
        self.playing = true;
    }

    pub fn pause(&mut self) {
        self.playing = false;
    }

    pub fn is_playing(&self) -> bool {
        self.playing
    }

    pub fn restart(&mut self) {
        self.time = 0.0;
        self.cycles_done = 0;
        self.playing = true;
    }

    /// Overall progress through the timeline, `0..1`.
    pub fn progress(&self) -> f32 {
        if self.duration <= 0.0 {
            1.0
        } else {
            (self.time / self.duration).clamp(0.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_start_and_end_when_they_say() {
        let mut tl = Timeline::new();
        let a = tl.add(0.0, 1.0, Easing::Linear);
        let b = tl.add(2.0, 1.0, Easing::Linear);
        assert_eq!(tl.duration(), 3.0);

        // Before b starts, it reads as 0 — safe to apply unconditionally.
        tl.seek(0.5);
        assert!((tl.progress_of(a) - 0.5).abs() < 1e-4);
        assert_eq!(tl.progress_of(b), 0.0);
        assert!(tl.is_active(a) && !tl.is_active(b));

        // After a ends, it holds at 1.
        tl.seek(2.5);
        assert_eq!(tl.progress_of(a), 1.0);
        assert!((tl.progress_of(b) - 0.5).abs() < 1e-4);
    }

    #[test]
    fn chaining_puts_each_track_after_the_last() {
        let mut tl = Timeline::new();
        let a = tl.then(1.0, Easing::Linear);
        let b = tl.then(2.0, Easing::Linear);
        let c = tl.then(0.5, Easing::Linear);
        assert_eq!(tl.duration(), 3.5);

        tl.seek(1.5);
        assert_eq!(tl.progress_of(a), 1.0);
        assert!(tl.progress_of(b) > 0.0 && tl.progress_of(b) < 1.0);
        assert_eq!(tl.progress_of(c), 0.0);
    }

    #[test]
    fn alongside_shares_a_start_time() {
        let mut tl = Timeline::new();
        let a = tl.add(1.0, 2.0, Easing::Linear);
        let b = tl.alongside(a, 1.0, Easing::Linear);
        tl.seek(1.5);
        assert!(tl.progress_of(a) > 0.0);
        assert!(tl.progress_of(b) > 0.0);
    }

    #[test]
    fn it_can_be_scrubbed_in_both_directions() {
        let mut tl = Timeline::new();
        let a = tl.add(0.0, 2.0, Easing::Linear);
        tl.seek(1.0);
        let forward = tl.progress_of(a);
        tl.seek(0.25);
        assert!(tl.progress_of(a) < forward, "seeking backwards did not rewind");

        // Negative playback.
        tl.seek(1.0);
        tl.speed = -1.0;
        tl.update(0.5);
        assert!((tl.time() - 0.5).abs() < 1e-4, "reverse playback: {}", tl.time());
    }

    #[test]
    fn repeat_modes_behave() {
        let mut once = Timeline::new();
        once.add(0.0, 1.0, Easing::Linear);
        once.update(5.0);
        assert!(once.is_finished());
        assert_eq!(once.time(), 1.0);

        let mut forever = Timeline::new();
        let t = forever.add(0.0, 1.0, Easing::Linear);
        forever.repeat = Repeat::Forever;
        for _ in 0..100 {
            forever.update(0.1);
            assert!(!forever.is_finished());
            let p = forever.progress_of(t);
            assert!((0.0..=1.0).contains(&p));
        }
        assert!(forever.cycles() > 5);

        let mut twice = Timeline::new();
        twice.add(0.0, 1.0, Easing::Linear);
        twice.repeat = Repeat::Times(1);
        twice.update(1.5);
        assert!(!twice.is_finished());
        twice.update(1.0);
        assert!(twice.is_finished());
    }

    #[test]
    fn pausing_stops_the_clock() {
        let mut tl = Timeline::new();
        tl.add(0.0, 2.0, Easing::Linear);
        tl.update(0.5);
        tl.pause();
        let held = tl.time();
        tl.update(1.0);
        assert_eq!(tl.time(), held);
        tl.play();
        tl.update(0.5);
        assert!(tl.time() > held);
    }

    #[test]
    fn an_empty_timeline_is_harmless() {
        let mut tl = Timeline::new();
        tl.update(1.0);
        assert_eq!(tl.duration(), 0.0);
        assert_eq!(tl.progress(), 1.0);
        // A handle from another timeline must not panic here.
        let stray = TrackId(99);
        assert_eq!(tl.progress_of(stray), 0.0);
        assert!(!tl.is_active(stray));
        assert_eq!(tl.value_of(stray, 1.0f32, 2.0), 1.0);
    }

    #[test]
    fn nonsense_time_is_ignored() {
        let mut tl = Timeline::new();
        tl.add(0.0, 1.0, Easing::Linear);
        tl.update(0.5);
        let held = tl.time();
        tl.update(f32::NAN);
        tl.seek(f32::INFINITY);
        assert_eq!(tl.time(), held);
    }

    #[test]
    fn zero_length_tracks_act_as_instants() {
        let mut tl = Timeline::new();
        let cue = tl.add(1.0, 0.0, Easing::Linear);
        tl.seek(0.5);
        assert_eq!(tl.progress_of(cue), 0.0);
        tl.seek(1.0);
        assert_eq!(tl.progress_of(cue), 1.0);
    }

    #[test]
    fn values_interpolate_along_a_track() {
        use threers::math::Vector3;
        let mut tl = Timeline::new();
        let t = tl.add(0.0, 1.0, Easing::Linear);
        tl.seek(0.5);
        let v: Vector3 = tl.value_of(t, Vector3::ZERO, Vector3::new(10.0, 0.0, 0.0));
        assert!((v.x - 5.0).abs() < 1e-3);
    }
}
