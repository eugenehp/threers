//! Keyframe animation: several properties of one box, each on its own
//! track of values at times, eased from one to the next, run once, some
//! times or for ever ([`Keyframes`], played with [`Ui::play_keyframes`]).
//! And phases: a box stepping through a few looks in turn, each held for a
//! while ([`Keyframes::phases`]), as a pulse, a breath or an attention
//! wiggle.
//!
//! With reduced motion, tracks of motion (place, size, turn) go straight to
//! their last values; the others (opacity, colour) still play.
//!
//! ```
//! use threers_ui::animation::{Property, Repeat};
//! use threers_ui::keyframes::Keyframes;
//! use threers_ui::{style, Ui};
//! use threers_math::easing::Easing;
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let bell = ui.add_box(root, style! { width: 40px; height: 40px; });
//! // A wiggle: turned one way, the other, and back, while it swells.
//! let ring = Keyframes::new()
//!     .track(Property::RotateZ, &[(0.0, 0.0), (0.1, 0.3), (0.3, -0.3), (0.45, 0.0)])
//!     .track_eased(Property::Scale, &[(0.0, 1.0, Easing::Linear), (0.2, 1.15, Easing::QuadOut), (0.45, 1.0, Easing::BackOut)])
//!     .repeat(Repeat::Times(2));
//! ui.play_keyframes(bell, ring);
//! ui.step(0.1);
//! assert!((ui.transform3d(bell).unwrap().rotate_z - 0.3).abs() < 1e-4);
//! ```

use std::rc::Rc;

use threers_math::easing::Easing;

use crate::animation::{Property, Repeat, Value};
use crate::tree::{NodeId, Ui};

/// One property's values over time: `(seconds, value, easing into it)`.
#[derive(Debug, Clone)]
struct Track {
    property: Property,
    keys: Vec<(f32, Value, Easing)>,
}

/// Tracks of keyframes for one box. See the [module docs](self).
#[derive(Debug, Clone, Default)]
pub struct Keyframes {
    tracks: Vec<Track>,
    repeat: Repeat,
}

impl Keyframes {
    pub fn new() -> Self {
        Self::default()
    }

    /// A track of numbers at times (seconds from the start), eased
    /// smoothly (in and out) from each to the next.
    pub fn track(self, property: Property, keys: &[(f32, f32)]) -> Self {
        let keys: Vec<(f32, f32, Easing)> = keys.iter().map(|&(t, v)| (t, v, Easing::SineInOut)).collect();
        self.track_eased(property, &keys)
    }

    /// A track of numbers at times, each with the easing that leads into
    /// it.
    pub fn track_eased(mut self, property: Property, keys: &[(f32, f32, Easing)]) -> Self {
        let mut keys: Vec<(f32, Value, Easing)> = keys.iter().map(|&(t, v, e)| (t.max(0.0), Value::Number(v), e)).collect();
        keys.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.tracks.push(Track { property, keys });
        self
    }

    /// A track of colours (straight RGBA) at times.
    pub fn track_colors(mut self, property: Property, keys: &[(f32, [f32; 4])]) -> Self {
        let mut keys: Vec<(f32, Value, Easing)> = keys.iter().map(|&(t, c)| (t.max(0.0), Value::Color(c), Easing::SineInOut)).collect();
        keys.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.tracks.push(Track { property, keys });
        self
    }

    /// Phases: the box takes each look in turn, `seconds` each (eased by
    /// `easing` into it), and back to the first; a look is property
    /// values. Properties missing from a phase hold where they were.
    pub fn phases(phases: &[&[(Property, f32)]], seconds: f32, easing: Easing) -> Self {
        let mut out = Keyframes::new();
        let mut props: Vec<Property> = vec![];
        for p in phases.iter().flat_map(|p| p.iter()) {
            if !props.contains(&p.0) {
                props.push(p.0);
            }
        }
        for prop in props {
            let mut keys = vec![];
            let mut last = None;
            for (i, phase) in phases.iter().chain(phases.first()).enumerate() {
                let v = phase.iter().find(|p| p.0 == prop).map(|p| p.1).or(last);
                if let Some(v) = v {
                    keys.push((i as f32 * seconds, v, easing));
                    last = Some(v);
                }
            }
            out = out.track_eased(prop, &keys);
        }
        out
    }

    /// How many times it runs (once by default).
    pub fn repeat(mut self, repeat: Repeat) -> Self {
        self.repeat = repeat;
        self
    }

    /// Seconds one run takes: to its latest key.
    pub fn duration(&self) -> f32 {
        self.tracks.iter().filter_map(|t| t.keys.last().map(|k| k.0)).fold(0.0, f32::max)
    }

    /// Each track's value `t` seconds into a run.
    fn sample(&self, t: f32) -> Vec<(Property, Value)> {
        self.tracks
            .iter()
            .filter_map(|track| {
                let keys = &track.keys;
                let first = keys.first()?;
                if t <= first.0 {
                    return Some((track.property, first.1));
                }
                let i = keys.windows(2).position(|w| t < w[1].0);
                let v = match i {
                    None => keys.last()?.1,
                    Some(i) => {
                        let ((t0, a, _), (t1, b, e)) = (keys[i], keys[i + 1]);
                        let f = if t1 > t0 { e.apply((t - t0) / (t1 - t0)) } else { 1.0 };
                        lerp(a, b, f)
                    }
                };
                Some((track.property, v))
            })
            .collect()
    }
}

fn lerp(a: Value, b: Value, t: f32) -> Value {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => Value::Number(a + (b - a) * t),
        (Value::Color(a), Value::Color(b)) => Value::Color(std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)),
        (_, b) => b,
    }
}

/// A box's keyframes playing.
#[derive(Debug, Clone)]
pub(crate) struct Playing {
    node: NodeId,
    keyframes: Rc<Keyframes>,
    elapsed: f32,
}

impl Ui {
    /// Plays `keyframes` on `node`, from the start (in place of any it was
    /// playing).
    pub fn play_keyframes(&mut self, node: NodeId, keyframes: Keyframes) {
        self.env.keyframes.retain(|p| p.node != node);
        // Reduced motion: motion goes straight to where it ends.
        let skipped: Vec<Track> = keyframes.tracks.iter().filter(|t| self.motion_skipped(t.property)).cloned().collect();
        for t in &skipped {
            if let Some(k) = t.keys.last() {
                self.write(node, t.property, k.1);
            }
        }
        let rest = Keyframes { tracks: keyframes.tracks.into_iter().filter(|t| !self.motion_skipped(t.property)).collect(), repeat: keyframes.repeat };
        if rest.tracks.is_empty() {
            return;
        }
        for (p, v) in rest.sample(0.0) {
            self.write(node, p, v);
        }
        self.env.keyframes.push(Playing { node, keyframes: Rc::new(rest), elapsed: 0.0 });
    }

    /// Stops `node`'s keyframes where they are.
    pub fn stop_keyframes(&mut self, node: NodeId) {
        self.env.keyframes.retain(|p| p.node != node);
    }

    /// Whether `node` is playing keyframes.
    pub fn playing_keyframes(&self, node: NodeId) -> bool {
        self.env.keyframes.iter().any(|p| p.node == node)
    }

    /// Moves keyframes on by `dt` seconds (from [`Ui::step`]).
    pub(crate) fn step_keyframes(&mut self, dt: f32) {
        let mut playing = std::mem::take(&mut self.env.keyframes);
        playing.retain(|p| self.contains(p.node));
        let mut keep = Vec::with_capacity(playing.len());
        for mut p in playing {
            p.elapsed += dt;
            let d = p.keyframes.duration().max(1e-4);
            let runs = match p.keyframes.repeat {
                Repeat::Once => Some(1),
                Repeat::Times(n) => Some(n.max(1)),
                Repeat::Forever => None,
            };
            let done = runs.is_some_and(|n| p.elapsed >= d * n as f32);
            let t = if done { d } else { p.elapsed % d };
            for (prop, v) in p.keyframes.sample(t) {
                self.write(p.node, prop, v);
            }
            if !done {
                keep.push(p);
            }
        }
        keep.append(&mut self.env.keyframes);
        self.env.keyframes = keep;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style;

    #[test]
    fn tracks_run_side_by_side_ease_between_keys_and_end_on_the_last() {
        let mut ui = Ui::new();
        let root = ui.root();
        let b = ui.add_box(root, style! { width: 20px; height: 20px; });
        let k = Keyframes::new()
            .track_eased(Property::Opacity, &[(0.0, 1.0, Easing::Linear), (1.0, 0.0, Easing::Linear)])
            .track_eased(Property::Scale, &[(0.0, 1.0, Easing::Linear), (0.5, 2.0, Easing::Linear), (1.0, 1.0, Easing::Linear)]);
        assert_eq!(k.duration(), 1.0);
        ui.play_keyframes(b, k);
        ui.step(0.25);
        assert!((ui.style(b).opacity - 0.75).abs() < 1e-4);
        assert!((ui.transform3d(b).unwrap().scale - 1.5).abs() < 1e-4);
        assert!(ui.needs_frame());
        ui.step(0.5);
        assert!((ui.transform3d(b).unwrap().scale - 1.5).abs() < 1e-4, "on its way back");
        ui.step(0.5);
        assert_eq!(ui.style(b).opacity, 0.0);
        assert_eq!(ui.transform3d(b).unwrap().scale, 1.0);
        assert!(!ui.playing_keyframes(b));
    }

    #[test]
    fn phases_loop_and_reduced_motion_skips_to_the_end_of_motion() {
        let mut ui = Ui::new();
        let root = ui.root();
        let b = ui.add_box(root, style! { width: 20px; height: 20px; });
        let pulse = Keyframes::phases(&[&[(Property::Scale, 1.0), (Property::Opacity, 1.0)], &[(Property::Scale, 1.3), (Property::Opacity, 0.5)]], 0.5, Easing::Linear).repeat(Repeat::Forever);
        assert_eq!(pulse.duration(), 1.0, "there and back");
        ui.play_keyframes(b, pulse.clone());
        ui.step(0.5);
        assert!((ui.transform3d(b).unwrap().scale - 1.3).abs() < 1e-4);
        ui.step(0.75);
        assert!((ui.transform3d(b).unwrap().scale - 1.15).abs() < 1e-3, "round again");
        assert!(ui.playing_keyframes(b));
        ui.stop_keyframes(b);
        // Reduced motion: no pulse in size; the fade still plays.
        let mut calm = Ui::new();
        calm.set_reduce_motion(true);
        let root = calm.root();
        let c = calm.add_box(root, style! { width: 20px; height: 20px; });
        calm.play_keyframes(c, pulse);
        calm.step(0.5);
        assert_eq!(calm.transform3d(c).map_or(1.0, |t| t.scale), 1.0);
        assert!((calm.style(c).opacity - 0.5).abs() < 1e-4);
    }
}
