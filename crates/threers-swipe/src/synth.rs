//! Swipes made up from words, for keyboards no one has swiped on yet.
//!
//! A hand drawing a word aims near each key, not at its centre, curves
//! between them and slows at each one. So: an aim point scattered around
//! each key (and the whole word shifted a little, as a thumb is off one
//! way), a bend between keys, a smooth path through all of it, and along
//! each stretch between two letters the minimum-jerk profile — the speed
//! a reaching arm has, slow, fast, slow — taking as long as Fitts' law
//! gives for the distance. Then sampled at the touch screen's rate, with a
//! little jitter.
//!
//! Recorded swipes beat these; these beat none, and teach the model the
//! letters of a layout before any recording exists.

use crate::geometry::KeyboardGeometry;
use crate::trace::TouchPoint;

/// A small, fast, seedable generator (SplitMix64); runs are reproducible.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }

    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f32 {
        let u = self.uniform().max(1e-7);
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }

    pub fn chance(&mut self, p: f32) -> bool {
        self.uniform() < p
    }
}

/// How sloppy and how quick the made-up hands are.
#[derive(Debug, Clone)]
pub struct SynthConfig {
    /// How far from a key's centre the finger aims, standard deviation in
    /// key units.
    pub aim: f32,
    /// The whole word shifted, standard deviation in key units.
    pub offset: f32,
    /// The bend between two keys, standard deviation as a fraction of the
    /// distance.
    pub bend: f32,
    /// Noise on every sample, key units.
    pub jitter: f32,
    /// Touch samples per second.
    pub rate: f32,
    /// Fitts' law, `a + b·log2(1 + distance)` seconds a stretch: ranges
    /// each swipe draws its own from, so some hands are quicker.
    pub fitts_a: (f32, f32),
    pub fitts_b: (f32, f32),
    /// Extra seconds at a letter typed twice running.
    pub dwell: f32,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            aim: 0.22,
            offset: 0.08,
            bend: 0.12,
            jitter: 0.012,
            rate: 60.0,
            fitts_a: (0.02, 0.06),
            fitts_b: (0.06, 0.13),
            dwell: 0.05,
        }
    }
}

/// A made-up swipe of `word` on `keyboard`, in its key units; `None` when
/// a letter has no key there.
pub fn swipe(
    keyboard: &KeyboardGeometry,
    word: &str,
    cfg: &SynthConfig,
    rng: &mut Rng,
) -> Option<Vec<TouchPoint>> {
    let keys = keyboard.word_keys(word)?;
    if keys.is_empty() {
        return None;
    }
    let (ox, oy) = (rng.normal() * cfg.offset, rng.normal() * cfg.offset);
    // Aim points, one per letter run, and the dwell each gets.
    let mut aims: Vec<(f32, f32)> = vec![];
    let mut dwells: Vec<f32> = vec![];
    for (i, &k) in keys.iter().enumerate() {
        if i > 0 && keys[i - 1] == k {
            *dwells.last_mut().unwrap() += cfg.dwell * rng.range(0.0, 2.0);
            continue;
        }
        let key = &keyboard.keys[k];
        aims.push((
            key.x + ox + rng.normal() * cfg.aim,
            key.y + oy + rng.normal() * cfg.aim,
        ));
        dwells.push(0.0);
    }
    let a = rng.range(cfg.fitts_a.0, cfg.fitts_a.1);
    let b = rng.range(cfg.fitts_b.0, cfg.fitts_b.1);
    let dt = 1.0 / cfg.rate;
    let mut t = 0.0f32;
    let mut out = vec![];
    let mut push = |x: f32, y: f32, t: f32, rng: &mut Rng| {
        out.push(TouchPoint::new(
            x + rng.normal() * cfg.jitter,
            y + rng.normal() * cfg.jitter,
            t,
        ));
    };

    if aims.len() == 1 {
        // A one-letter word: a short press with a little drift.
        let (x, y) = aims[0];
        let n = 4 + rng.below(6);
        let (dx, dy) = (rng.normal() * 0.05, rng.normal() * 0.05);
        for i in 0..n {
            let f = i as f32 / (n - 1) as f32;
            push(x + dx * f, y + dy * f, t, rng);
            t += dt;
        }
        return Some(out);
    }

    // Control points: the aims with a bent midpoint between each two, so
    // a spline through them curves as a hand does.
    let mut ctrl = vec![aims[0]];
    for w in aims.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        let (dx, dy) = (x1 - x0, y1 - y0);
        let along = rng.range(0.35, 0.65);
        let side = rng.normal() * cfg.bend;
        ctrl.push((x0 + dx * along - dy * side, y0 + dy * along + dx * side));
        ctrl.push(w[1]);
    }

    for i in 0..aims.len() - 1 {
        // The stretch from aim i to aim i+1: two spline pieces.
        let mut poly = vec![];
        for piece in [2 * i, 2 * i + 1] {
            let p = |j: isize| ctrl[j.clamp(0, ctrl.len() as isize - 1) as usize];
            let j = piece as isize;
            let (p0, p1, p2, p3) = (p(j - 1), p(j), p(j + 1), p(j + 2));
            let steps = 12;
            for s in 0..steps {
                let u = s as f32 / steps as f32;
                poly.push(catmull_rom(p0, p1, p2, p3, u));
            }
        }
        poly.push(aims[i + 1]);
        let cum: Vec<f32> = std::iter::once(0.0)
            .chain(poly.windows(2).scan(0.0, |acc, w| {
                *acc += (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1);
                Some(*acc)
            }))
            .collect();
        let len = *cum.last().unwrap();
        let duration = a + b * (1.0 + len).log2();
        let n = ((duration * cfg.rate).ceil() as usize).max(2);
        // Dwell at the letter this stretch leaves.
        let mut d = dwells[i];
        while d > 0.0 {
            push(poly[0].0, poly[0].1, t, rng);
            t += dt;
            d -= dt;
        }
        let last = if i + 2 == aims.len() { n + 1 } else { n };
        for k in 0..last {
            let tau = (k as f32 / n as f32).min(1.0);
            let s = min_jerk(tau) * len;
            let (x, y) = at_length(&poly, &cum, s);
            push(x, y, t, rng);
            t += dt;
        }
    }
    Some(out)
}

/// A random word of `words` that `keyboard` can type, swiped.
pub fn random_swipe<'a>(
    keyboard: &KeyboardGeometry,
    words: &'a [String],
    cfg: &SynthConfig,
    rng: &mut Rng,
) -> Option<(&'a str, Vec<TouchPoint>)> {
    for _ in 0..16 {
        let w = &words[rng.below(words.len())];
        if let Some(s) = swipe(keyboard, w, cfg, rng) {
            return Some((w, s));
        }
    }
    None
}

/// Distance covered by time `tau` of a minimum-jerk reach, both in [0, 1].
pub fn min_jerk(tau: f32) -> f32 {
    let t = tau.clamp(0.0, 1.0);
    t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
}

fn catmull_rom(
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    p3: (f32, f32),
    u: f32,
) -> (f32, f32) {
    let (u2, u3) = (u * u, u * u * u);
    let f = |a: f32, b: f32, c: f32, d: f32| {
        0.5 * (2.0 * b
            + (-a + c) * u
            + (2.0 * a - 5.0 * b + 4.0 * c - d) * u2
            + (-a + 3.0 * b - 3.0 * c + d) * u3)
    };
    (f(p0.0, p1.0, p2.0, p3.0), f(p0.1, p1.1, p2.1, p3.1))
}

fn at_length(poly: &[(f32, f32)], cum: &[f32], s: f32) -> (f32, f32) {
    let i = cum.partition_point(|&c| c < s).clamp(1, poly.len() - 1);
    let (c0, c1) = (cum[i - 1], cum[i]);
    let f = if c1 > c0 {
        ((s - c0) / (c1 - c0)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (a, b) = (poly[i - 1], poly[i]);
    (a.0 + f * (b.0 - a.0), a.1 + f * (b.1 - a.1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers_ui::virtual_keyboard::{KeyboardConfig, KeyboardLayout};

    #[test]
    fn swipes_start_and_end_near_the_letters() {
        let g = KeyboardGeometry::from_config(
            &KeyboardConfig::iphone(),
            &KeyboardLayout::english(),
            390.0,
        );
        let mut rng = Rng::new(3);
        let cfg = SynthConfig::default();
        let (mut start, mut end) = (0.0, 0.0);
        let n = 200;
        for _ in 0..n {
            let s = swipe(&g, "quiz", &cfg, &mut rng).unwrap();
            let (q, z) = (
                &g.keys[g.key_for('q').unwrap()],
                &g.keys[g.key_for('z').unwrap()],
            );
            start += (s[0].x - q.x).hypot(s[0].y - q.y);
            let l = s.last().unwrap();
            end += (l.x - z.x).hypot(l.y - z.y);
            assert!(s.windows(2).all(|w| w[1].t > w[0].t), "time goes forward");
        }
        assert!(start / (n as f32) < 0.45 && end / (n as f32) < 0.45);
        assert!(swipe(&g, "日本", &cfg, &mut rng).is_none());
    }

    #[test]
    fn min_jerk_is_slow_fast_slow() {
        assert_eq!(min_jerk(0.0), 0.0);
        assert!((min_jerk(1.0) - 1.0).abs() < 1e-6);
        assert!((min_jerk(0.5) - 0.5).abs() < 1e-6);
        assert!(min_jerk(0.1) < 0.1 && min_jerk(0.9) > 0.9);
    }
}
