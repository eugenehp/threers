//! Checks that need more than one pose.
//!
//! A single pose can be interrogated all day and will not tell you whether the
//! mechanism works. Interference at rest says nothing about the sweep; a hinge
//! that has come apart sits exactly where a hinge that has not sits. Both
//! questions here are about what *persists* across poses, which is why they take
//! all of them at once.

use super::{correspond, nearest, Tri};

/// One body, in one pose.
pub type Body = Vec<Tri>;

/// Which body in each pose is which body in the first one.
///
/// `align(poses)[p][b]` is the index, within pose `p`, of the body that is body
/// `b` of pose 0 — or `None` where it could not be found.
///
/// Matched **stepwise**, pose to adjacent pose, and composed. That matters:
/// [`crate::assembly::pose::correspond`] separates identical parts by which one is nearest, which is
/// sound over a small step and a guess over a large one. Matching pose 20 to
/// pose 0 directly is the large one; matching it through the nineteen poses in
/// between is not.
///
/// # When a body goes missing
///
/// `None` is not a failure to try harder. A body that has no counterpart has
/// genuinely changed shape — a boolean that took a different branch, two solids
/// that merged into one shell, a part that was not there before — and reporting
/// that is more useful than pairing it with whatever was closest.
pub fn align(poses: &[Vec<Body>]) -> Vec<Vec<Option<usize>>> {
    if poses.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<Vec<Option<usize>>> = vec![(0..poses[0].len()).map(Some).collect()];
    for p in 1..poses.len() {
        let mut step = vec![None; poses[p - 1].len()];
        for (i, j) in correspond(&poses[p - 1], &poses[p]) {
            step[i] = Some(j);
        }
        let mapped = out[p - 1]
            .iter()
            .map(|slot| slot.and_then(|i| step.get(i).copied().flatten()))
            .collect();
        out.push(mapped);
    }
    out
}

/// Two bodies that touch, and what became of that over the poses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Engagement {
    /// Body indices, in pose 0's numbering.
    pub a: usize,
    pub b: usize,
    /// Poses in which the two were within tolerance.
    pub touching: usize,
    /// Poses in which both were found at all.
    pub present: usize,
    /// The widest they ever got, in model units.
    ///
    /// For a pair that stays engaged this sits under the tolerance. For one that
    /// comes apart it is how far apart it came, which is the number that says
    /// whether a joint has failed or merely opened a rounding error.
    pub widest: f64,
    /// Whether they were touching in the first pose both appeared in.
    pub at_first: bool,
    /// Whether they were touching in the last one.
    pub at_last: bool,
}

impl Engagement {
    /// Whether the pair touched in every pose where both were present.
    pub fn persistent(&self) -> bool {
        self.touching == self.present && self.present > 0
    }

    /// Whether they started together and ended apart.
    ///
    /// The question a joint is asked, and not the same as "did not touch in
    /// every pose". A lid swinging into a shelf touches it halfway through and
    /// is not persistent, and nothing has come apart — it has come *together*.
    /// A hinge that separates starts touching and stops.
    pub fn came_apart(&self) -> bool {
        self.at_first && !self.at_last
    }

    /// Whether they started apart and ended together — a collision, not a
    /// failure.
    pub fn came_together(&self) -> bool {
        !self.at_first && self.at_last
    }
}

/// Which bodies touch, and whether they go on touching.
///
/// Reports every pair that touches in **at least one** pose, so a pair that
/// separates is in the list with [`Engagement::persistent`] false — that is the
/// finding. Pairs that never touch are not engagement and are not reported;
/// [`crate::assembly::pose::floating`] is where a body with no contacts at all shows up.
///
/// `tolerance` is how close counts as touching. Model it on the clearance the
/// parts were drawn with: a hinge pin in a bore with 0.2 mm of slop is touching
/// at 0.2, separated at 2, and a tolerance between those two tells them apart.
///
/// # Cost
///
/// A closest-approach measurement per pair per pose, pruned by bounds. On an
/// assembly of a dozen parts over a hundred poses that is thousands of mesh
/// distance queries — seconds, not milliseconds. It is a check, not a frame
/// loop.
pub fn engagement(poses: &[Vec<Body>], tolerance: f64) -> Vec<Engagement> {
    engagement_with(poses, &align(poses), tolerance)
}

/// The same, when the correspondence is already known.
///
/// [`crate::assembly::pose::engagement`] works out which body is which by shape and proximity, which is
/// the right thing when the poses came out of a rebuilt model and all you have
/// is triangles. It is the wrong thing when you already know — a simulation
/// moves bodies it can name, and asking it to guess again risks getting a
/// different answer for two identical parts.
///
/// `alignment` has the shape [`crate::assembly::pose::align`] returns: `alignment[p][b]` is the index,
/// within pose `p`, of body `b`.
pub fn engagement_with(
    poses: &[Vec<Body>],
    alignment: &[Vec<Option<usize>>],
    tolerance: f64,
) -> Vec<Engagement> {
    let bodies = alignment.first().map_or(0, |p| p.len());
    let mut out = Vec::new();

    for a in 0..bodies {
        for b in (a + 1)..bodies {
            let mut touching = 0;
            let mut present = 0;
            let mut widest: f64 = 0.0;
            let mut ever = false;
            let mut at_first = false;
            let mut at_last = false;
            let mut seen = false;

            for (p, pose) in poses.iter().enumerate() {
                let (Some(ia), Some(ib)) = (alignment[p][a], alignment[p][b]) else {
                    continue;
                };
                let (Some(ba), Some(bb)) = (pose.get(ia), pose.get(ib)) else {
                    continue;
                };
                present += 1;
                // Bounds first: two parts at opposite ends of the model are the
                // common case, and they can be rejected without touching a
                // triangle.
                let gap = if bounds_gap(ba, bb) > tolerance {
                    bounds_gap(ba, bb)
                } else {
                    nearest(ba, bb)
                };
                widest = widest.max(gap);
                let close = gap <= tolerance;
                if close {
                    touching += 1;
                    ever = true;
                }
                // First and last pose in which BOTH were found, which is not
                // necessarily the first and last pose.
                if !seen {
                    at_first = close;
                    seen = true;
                }
                at_last = close;
            }

            if ever {
                out.push(Engagement {
                    a,
                    b,
                    touching,
                    present,
                    widest,
                    at_first,
                    at_last,
                });
            }
        }
    }
    out
}

/// Bodies that touch nothing, in any pose.
///
/// An unjoined body is almost always a mistake, and a quiet one: it renders,
/// it has mass, it interferes with nothing, and every check that looks for
/// things overlapping passes. The only sign is that it is not attached to the
/// mechanism at all.
///
/// Returns indices in pose 0's numbering. A body that could not be matched in
/// some pose is judged on the poses where it was found; one that is never
/// matched anywhere is reported as floating, because nothing was ever shown to
/// touch it.
pub fn floating(poses: &[Vec<Body>], tolerance: f64) -> Vec<usize> {
    floating_with(poses, &align(poses), tolerance)
}

/// The same, when the correspondence is already known — see [`engagement_with`].
pub fn floating_with(
    poses: &[Vec<Body>],
    alignment: &[Vec<Option<usize>>],
    tolerance: f64,
) -> Vec<usize> {
    let bodies = alignment.first().map_or(0, |p| p.len());
    let attached: std::collections::HashSet<usize> = engagement_with(poses, alignment, tolerance)
        .iter()
        .flat_map(|e| [e.a, e.b])
        .collect();
    (0..bodies).filter(|i| !attached.contains(i)).collect()
}

/// Gap between two bodies' bounds — zero when they overlap. A lower bound on
/// the real distance, and enormously cheaper.
fn bounds_gap(a: &[Tri], b: &[Tri]) -> f64 {
    let (alo, ahi) = super::aabb(a);
    let (blo, bhi) = super::aabb(b);
    let mut d = 0.0;
    for k in 0..3 {
        let gap = (alo[k] - bhi[k]).max(blo[k] - ahi[k]).max(0.0);
        d += gap * gap;
    }
    d.sqrt()
}
