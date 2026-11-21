//! The keyboard and the swipe moved together, so the model learns the
//! shape of a swipe against the keys it crossed, not one layout's map.
//!
//! Each training swipe gets its own stretch, shear, turn and mirror,
//! applied alike to the touch points and the key centres, so the swipe
//! still crosses the same keys. Sometimes the swipe also runs backwards
//! (the word reversed), and the finger is a little off the keys as a whole.

use crate::synth::Rng;
use crate::trace::TouchPoint;

/// How much each swipe may be moved.
#[derive(Debug, Clone)]
pub struct AugmentConfig {
    pub scale_x: (f32, f32),
    pub scale_y: (f32, f32),
    /// Horizontal shear, at most.
    pub shear: f32,
    /// Turn, at most, in degrees.
    pub rotate: f32,
    /// Chances of mirroring left-right and top-bottom.
    pub flip_x: f32,
    pub flip_y: f32,
    /// Chance of running the swipe backwards.
    pub reverse: f32,
    /// The swipe off the keys, standard deviation in key units.
    pub shift: f32,
    /// Chance of the times lost (all zero), as from an input with no
    /// clock, so the model does not lean on speed alone.
    pub no_time: f32,
    /// Chance of the times rounded to 60 Hz frames, as a UI that stamps
    /// each event with its frame's time gives them.
    pub frame_time: f32,
}

impl Default for AugmentConfig {
    fn default() -> Self {
        Self {
            scale_x: (0.85, 1.15),
            scale_y: (0.75, 1.3),
            shear: 0.12,
            rotate: 8.0,
            flip_x: 0.5,
            flip_y: 0.2,
            reverse: 0.1,
            shift: 0.06,
            no_time: 0.1,
            frame_time: 0.3,
        }
    }
}

impl AugmentConfig {
    /// None of it: what evaluation uses.
    pub fn none() -> Self {
        Self {
            scale_x: (1.0, 1.0),
            scale_y: (1.0, 1.0),
            shear: 0.0,
            rotate: 0.0,
            flip_x: 0.0,
            flip_y: 0.0,
            reverse: 0.0,
            shift: 0.0,
            no_time: 0.0,
            frame_time: 0.0,
        }
    }
}

/// `keys` and `points` moved alike, and `target` reversed if the swipe
/// is.
pub fn augment(
    keys: &mut [(f32, f32)],
    points: &mut [TouchPoint],
    target: &mut [usize],
    cfg: &AugmentConfig,
    rng: &mut Rng,
) {
    let sx =
        rng.range(cfg.scale_x.0, cfg.scale_x.1) * if rng.chance(cfg.flip_x) { -1.0 } else { 1.0 };
    let sy =
        rng.range(cfg.scale_y.0, cfg.scale_y.1) * if rng.chance(cfg.flip_y) { -1.0 } else { 1.0 };
    let sh = rng.range(-cfg.shear, cfg.shear);
    let r = rng.range(-cfg.rotate, cfg.rotate).to_radians();
    let (c, s) = (r.cos(), r.sin());
    // Rotation · shear · scale.
    let m = |x: f32, y: f32| {
        let (x, y) = (x * sx, y * sy);
        let x = x + sh * y;
        (c * x - s * y, s * x + c * y)
    };
    for k in keys.iter_mut() {
        *k = m(k.0, k.1);
    }
    let (dx, dy) = (rng.normal() * cfg.shift, rng.normal() * cfg.shift);
    for p in points.iter_mut() {
        let (x, y) = m(p.x, p.y);
        (p.x, p.y) = (x + dx, y + dy);
    }
    if rng.chance(cfg.no_time) {
        for p in points.iter_mut() {
            p.t = 0.0;
        }
    } else if rng.chance(cfg.frame_time) {
        for p in points.iter_mut() {
            p.t = (p.t * 60.0).floor() / 60.0;
        }
    }
    if rng.chance(cfg.reverse) {
        let end = points.last().map_or(0.0, |p| p.t);
        points.reverse();
        for p in points.iter_mut() {
            p.t = end - p.t;
        }
        target.reverse();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_and_points_move_together() {
        let mut rng = Rng::new(9);
        let cfg = AugmentConfig {
            reverse: 1.0,
            ..AugmentConfig::default()
        };
        let mut keys = vec![(1.0, 0.5), (-2.0, 1.0)];
        let mut pts = vec![
            TouchPoint::new(1.0, 0.5, 0.0),
            TouchPoint::new(-2.0, 1.0, 0.3),
        ];
        let mut target = vec![0, 1];
        augment(
            &mut keys,
            &mut pts,
            &mut target,
            &AugmentConfig {
                shift: 0.0,
                no_time: 0.0,
                frame_time: 0.0,
                ..cfg
            },
            &mut rng,
        );
        assert_eq!(target, vec![1, 0]);
        assert!((pts[0].x - keys[1].0).abs() < 1e-5 && (pts[1].y - keys[0].1).abs() < 1e-5);
        assert_eq!((pts[0].t, pts[1].t), (0.0, 0.3));
    }
}
