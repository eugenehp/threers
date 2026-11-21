//! A swipe as the model reads it: [`FRAMES`] points evenly spaced along the
//! path, [`FEATURES`] numbers each.
//!
//! Evenly spaced along the path, not in time, so a slow swipe and a quick
//! one of the same word look alike; how fast the finger went is kept as a
//! feature instead, since it slows at the letters it means.

use serde::{Deserialize, Serialize};

/// Points the model reads per swipe.
pub const FRAMES: usize = 64;

/// Numbers per point: position (2), direction (2), turn, speed, progress.
pub const FEATURES: usize = 7;

/// A touch: where, in key units ([`KeyboardGeometry::to_units`]), and
/// when, in seconds (any origin).
///
/// [`KeyboardGeometry::to_units`]: crate::KeyboardGeometry::to_units
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TouchPoint {
    pub x: f32,
    pub y: f32,
    pub t: f32,
}

impl TouchPoint {
    pub fn new(x: f32, y: f32, t: f32) -> Self {
        Self { x, y, t }
    }
}

/// The path's length in key units.
pub fn length(points: &[TouchPoint]) -> f32 {
    points
        .windows(2)
        .map(|w| (w[1].x - w[0].x).hypot(w[1].y - w[0].y))
        .sum()
}

/// `n` points evenly spaced along the path, times interpolated. A tap
/// (no length) gives `n` copies of where it was.
pub fn resample(points: &[TouchPoint], n: usize) -> Vec<TouchPoint> {
    assert!(n >= 2, "resample to at least 2 points");
    let Some(&first) = points.first() else {
        return vec![TouchPoint::new(0.0, 0.0, 0.0); n];
    };
    let total = length(points);
    let last_t = points.last().map_or(first.t, |p| p.t);
    if total < 1e-6 {
        return (0..n)
            .map(|i| {
                let f = i as f32 / (n - 1) as f32;
                TouchPoint::new(first.x, first.y, first.t + f * (last_t - first.t))
            })
            .collect();
    }
    let mut out = Vec::with_capacity(n);
    let (mut j, mut walked) = (0usize, 0.0f32);
    for i in 0..n {
        let target = total * i as f32 / (n - 1) as f32;
        loop {
            if j + 1 >= points.len() {
                out.push(*points.last().unwrap());
                break;
            }
            let (a, b) = (points[j], points[j + 1]);
            let seg = (b.x - a.x).hypot(b.y - a.y);
            if walked + seg >= target || j + 2 == points.len() {
                let f = if seg > 0.0 {
                    ((target - walked) / seg).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                out.push(TouchPoint::new(
                    a.x + f * (b.x - a.x),
                    a.y + f * (b.y - a.y),
                    a.t + f * (b.t - a.t),
                ));
                break;
            }
            walked += seg;
            j += 1;
        }
    }
    out
}

/// The model's input for a swipe: [`FRAMES`] × [`FEATURES`], row-major.
/// Points are in key units, centred on the keyboard as its
/// [`KeyboardGeometry`](crate::KeyboardGeometry) is.
pub fn features(points: &[TouchPoint]) -> Vec<f32> {
    let p = resample(points, FRAMES);
    let n = p.len();
    let mut out = vec![0.0f32; n * FEATURES];
    let dir = |a: TouchPoint, b: TouchPoint| {
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        let len = dx.hypot(dy);
        if len > 1e-6 {
            (dx / len, dy / len)
        } else {
            (0.0, 0.0)
        }
    };
    for i in 0..n {
        let prev = p[i.saturating_sub(1)];
        let next = p[(i + 1).min(n - 1)];
        let (cx, cy) = dir(prev, next);
        let (inx, iny) = dir(prev, p[i]);
        let (outx, outy) = dir(p[i], next);
        let turn = (inx * outy - iny * outx).atan2(inx * outx + iny * outy) / std::f32::consts::PI;
        let dist = (next.x - prev.x).hypot(next.y - prev.y);
        let dt = next.t - prev.t;
        let speed = if dt > 1e-4 {
            (1.0 + dist / dt).ln()
        } else {
            0.0
        };
        let f = &mut out[i * FEATURES..(i + 1) * FEATURES];
        f[0] = p[i].x;
        f[1] = p[i].y;
        f[2] = cx;
        f[3] = cy;
        f[4] = turn;
        f[5] = speed / 4.0;
        f[6] = i as f32 / (n - 1) as f32 * 2.0 - 1.0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resamples_evenly_by_length() {
        let pts = [
            TouchPoint::new(0.0, 0.0, 0.0),
            TouchPoint::new(1.0, 0.0, 1.0),
            TouchPoint::new(1.0, 3.0, 1.5),
        ];
        let r = resample(&pts, 5);
        assert_eq!(r.len(), 5);
        assert!((r[1].x - 1.0).abs() < 1e-5 && r[1].y.abs() < 1e-5);
        assert!((r[2].y - 1.0).abs() < 1e-5);
        assert_eq!(r[4], pts[2]);
        let tap = resample(&pts[..1], 4);
        assert!(tap.iter().all(|p| p.x == 0.0 && p.y == 0.0));
    }

    #[test]
    fn features_turn_at_a_corner() {
        let pts: Vec<_> = (0..=20)
            .map(|i| {
                let s = i as f32 / 10.0;
                if s <= 1.0 {
                    TouchPoint::new(s, 0.0, s)
                } else {
                    TouchPoint::new(1.0, s - 1.0, s)
                }
            })
            .collect();
        let f = features(&pts);
        assert_eq!(f.len(), FRAMES * FEATURES);
        let turn = |i: usize| f[i * FEATURES + 4];
        let peak = (0..FRAMES).map(turn).fold(0.0f32, f32::max);
        assert!(peak > 0.2, "a left turn shows: {peak}");
        assert!(turn(5).abs() < 1e-4, "straight before it");
    }
}
