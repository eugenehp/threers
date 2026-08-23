//! Broad phase: cheap rejection of pairs that cannot possibly touch.
//!
//! Sweep-and-prune along one axis. The entry list is kept between steps, so the
//! sort sees an almost-sorted array every time — which is the case pattern
//! `sort_unstable_by` handles best, and why an incremental structure buys little
//! at the scales a browser scene reaches.

use crate::math::Aabb;
use threers::math::Vector3;

/// One body's bounds for this step.
#[derive(Debug, Clone, Copy)]
pub struct BroadEntry {
    pub min: Vector3,
    pub max: Vector3,
    /// Dense body-slot index.
    pub index: u32,
    /// Whether this body can move under its own steam this step. A pair with no
    /// active body is skipped: two fixed bodies, or two sleeping ones, cannot
    /// begin to touch.
    pub active: bool,
}

/// Sweep-and-prune broad phase.
#[derive(Debug, Default, Clone)]
pub struct BroadPhase {
    entries: Vec<BroadEntry>,
    axis: usize,
}

impl BroadPhase {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn add(&mut self, index: u32, aabb: &Aabb, active: bool) {
        self.entries.push(BroadEntry {
            min: aabb.min,
            max: aabb.max,
            index,
            active,
        });
    }

    /// The bounds currently held, one per body the broad phase was given.
    ///
    /// This is what a debug view wants: a pair the narrow phase never saw is a
    /// pair whose boxes here did not overlap.
    pub fn entries(&self) -> &[BroadEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Collect every overlapping pair into `out`, as `(index_a, index_b)` with
    /// `index_a < index_b`.
    pub fn find_pairs(&mut self, out: &mut Vec<(u32, u32)>) {
        out.clear();
        if self.entries.len() < 2 {
            return;
        }

        self.axis = self.best_axis();
        let axis = self.axis;
        let coord = |v: Vector3| match axis {
            0 => v.x,
            1 => v.y,
            _ => v.z,
        };

        self.entries.sort_unstable_by(|a, b| {
            coord(a.min)
                .partial_cmp(&coord(b.min))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for i in 0..self.entries.len() {
            let a = self.entries[i];
            let a_max = coord(a.max);
            for j in i + 1..self.entries.len() {
                let b = self.entries[j];
                // Sorted by min: once a box starts past this one's end, so does
                // every box after it.
                if coord(b.min) > a_max {
                    break;
                }
                if !a.active && !b.active {
                    continue;
                }
                if overlaps(&a, &b) {
                    let (lo, hi) = if a.index < b.index {
                        (a.index, b.index)
                    } else {
                        (b.index, a.index)
                    };
                    out.push((lo, hi));
                }
            }
        }
    }

    /// Sweep along whichever axis spreads the bodies most — the one that prunes
    /// the most pairs. Sweeping a fixed axis degenerates badly for scenes laid
    /// out along a different one (a tall tower swept along x compares
    /// everything with everything).
    fn best_axis(&self) -> usize {
        let n = self.entries.len() as f32;
        if n < 2.0 {
            return 0;
        }
        let mut sum = Vector3::ZERO;
        let mut sum_sq = Vector3::ZERO;
        for e in &self.entries {
            let c = (e.min + e.max) * 0.5;
            sum = sum + c;
            sum_sq = sum_sq + Vector3::new(c.x * c.x, c.y * c.y, c.z * c.z);
        }
        let mean = sum * (1.0 / n);
        let var = Vector3::new(
            sum_sq.x / n - mean.x * mean.x,
            sum_sq.y / n - mean.y * mean.y,
            sum_sq.z / n - mean.z * mean.z,
        );
        if var.x >= var.y && var.x >= var.z {
            0
        } else if var.y >= var.z {
            1
        } else {
            2
        }
    }
}

#[inline]
fn overlaps(a: &BroadEntry, b: &BroadEntry) -> bool {
    a.min.x <= b.max.x
        && a.max.x >= b.min.x
        && a.min.y <= b.max.y
        && a.max.y >= b.min.y
        && a.min.z <= b.max.z
        && a.max.z >= b.min.z
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aabb(cx: f32, cy: f32, cz: f32, h: f32) -> Aabb {
        Aabb::new(
            Vector3::new(cx - h, cy - h, cz - h),
            Vector3::new(cx + h, cy + h, cz + h),
        )
    }

    fn pairs(items: &[(f32, f32, f32, f32, bool)]) -> Vec<(u32, u32)> {
        let mut bp = BroadPhase::new();
        for (i, &(x, y, z, h, active)) in items.iter().enumerate() {
            bp.add(i as u32, &aabb(x, y, z, h), active);
        }
        let mut out = Vec::new();
        bp.find_pairs(&mut out);
        out.sort_unstable();
        out
    }

    #[test]
    fn overlapping_boxes_pair_and_distant_ones_do_not() {
        let p = pairs(&[
            (0.0, 0.0, 0.0, 1.0, true),
            (1.5, 0.0, 0.0, 1.0, true), // overlaps 0
            (50.0, 0.0, 0.0, 1.0, true), // far away
        ]);
        assert_eq!(p, vec![(0, 1)]);
    }

    #[test]
    fn two_inactive_bodies_never_pair() {
        // Both fixed and overlapping: still no pair, since neither can move.
        let p = pairs(&[(0.0, 0.0, 0.0, 1.0, false), (0.5, 0.0, 0.0, 1.0, false)]);
        assert!(p.is_empty());

        // One active is enough.
        let p = pairs(&[(0.0, 0.0, 0.0, 1.0, false), (0.5, 0.0, 0.0, 1.0, true)]);
        assert_eq!(p, vec![(0, 1)]);
    }

    #[test]
    fn pairs_are_ordered_and_unique() {
        let p = pairs(&[
            (0.0, 0.0, 0.0, 1.0, true),
            (0.5, 0.0, 0.0, 1.0, true),
            (1.0, 0.0, 0.0, 1.0, true),
        ]);
        assert_eq!(p, vec![(0, 1), (0, 2), (1, 2)]);
        for (a, b) in p {
            assert!(a < b);
        }
    }

    #[test]
    fn sweeping_picks_the_most_spread_axis() {
        // A tall stack along y. Sweeping x would compare every pair; sweeping y
        // prunes. Either way the answer must be the same — this checks the
        // optimisation did not change the result.
        let items: Vec<_> = (0..40)
            .map(|i| (0.0, i as f32 * 3.0, 0.0, 1.0, true))
            .collect();
        let p = pairs(&items);
        assert!(p.is_empty(), "boxes 3 apart with half-extent 1 do not touch");

        let items: Vec<_> = (0..10)
            .map(|i| (0.0, i as f32 * 1.5, 0.0, 1.0, true))
            .collect();
        let p = pairs(&items);
        // Each box overlaps its immediate neighbour (gap 1.5 < 2).
        assert!(p.contains(&(0, 1)));
        assert!(p.contains(&(8, 9)));
        assert!(!p.contains(&(0, 5)));
    }

    #[test]
    fn an_unbounded_box_pairs_with_everything_active() {
        let mut bp = BroadPhase::new();
        bp.add(
            0,
            &Aabb::new(
                Vector3::new(-1e9, -1e9, -1e9),
                Vector3::new(1e9, 1e9, 1e9),
            ),
            false, // a fixed ground plane
        );
        for i in 1..6 {
            bp.add(i, &aabb(i as f32 * 20.0, 0.0, 0.0, 1.0), true);
        }
        let mut out = Vec::new();
        bp.find_pairs(&mut out);
        assert_eq!(out.len(), 5, "ground should pair with all five bodies");
    }

    #[test]
    fn an_empty_or_single_entry_phase_yields_nothing() {
        assert!(pairs(&[]).is_empty());
        assert!(pairs(&[(0.0, 0.0, 0.0, 1.0, true)]).is_empty());
    }
}
