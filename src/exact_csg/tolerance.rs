//! The tolerance ladder for the arrangement kernel.
//!
//! # Why these are in one file
//!
//! The kernel carries roughly sixty absolute tolerance literals, spanning `1e-2`
//! to `1e-18`, of which three are scale-relative. They are not independent: they
//! form a ladder, and each rung has to sit below the one above or the stage that
//! reads it starts disagreeing with its neighbour about whether two points are
//! the same point.
//!
//! Nothing enforced that ordering, and nothing wrote it down. The consequence is
//! reproducible — moving the plane gate one rung, from `1e-7` to `1e-6`, which
//! looks like a harmless loosening, collapses the reference model from 40,173
//! triangles to 2,937; moving it to `3e-6` explodes it to 199,390. Neither is a
//! bug in the gate. Both are the gate crossing a neighbour.
//!
//! ```text
//! rung              value    what it decides
//! ---------------------------------------------------------------------------
//! weld              1e-6     two computed points are the same point
//! on-segment        1e-6     a point lies on a segment (perpendicular distance)
//! plane gate        1e-7     a point lies on a face's plane
//! plane grouping    1e-4     two triangles belong to the same flat face
//! ```
//!
//! The plane gate sits *below* the on-segment tolerance, which is the inversion
//! worth understanding before touching either: a seam point can be accepted as
//! lying on a segment and rejected as lying on the plane that segment runs
//! through. That is a real defect — it is one of the two mechanisms behind the
//! kernel's non-watertight results — but widening the gate to match does not fix
//! it, it detonates the model, because the gate is also holding the plane
//! grouping apart from the weld.
//!
//! # What this module is not
//!
//! It is not a fix. The values are exactly what they were; this only gives them
//! one home, states the relationships, and adds a test that fails when a change
//! breaks the ordering instead of letting someone discover it via a collapsed
//! model. The real repair is architectural — quantise once at ingest and compute
//! exactly on that grid, so "the same point" is bit-identical by construction and
//! the ladder is not needed at all. The `manifold` feature takes that route by
//! delegating to a kernel already built that way.

/// Two computed points closer than this are the same point.
///
/// `find_or_add` compares squared distance against `1e-12`, which is this
/// distance squared.
pub const WELD: f64 = 1e-6;

/// Perpendicular distance within which a point counts as lying on a segment.
///
/// `on_segment_interior` tests `sqlen(cross(ab, v-a)) > 1e-12 * ablen`, which
/// reduces to exactly this: `|cross| = |ab| · d`, so `sqlen(cross) = ablen · d²`
/// and the comparison is `d² > 1e-12`.
pub const ON_SEGMENT: f64 = 1e-6;

/// Distance within which a global cut point counts as lying on a face's plane.
///
/// Read by the cross-face T-junction heal and by the `touched` test that decides
/// whether an otherwise-uncut face has to be triangulated anyway. Both compare
/// against a *normalised* normal, so this is a true distance.
///
/// Note this is an order of magnitude tighter than [`ON_SEGMENT`] — see the
/// module docs.
pub const PLANE_GATE: f64 = 1e-7;

/// Quantisation of the plane key that groups triangles into flat faces.
///
/// Coarse on purpose: it has to absorb the CDT's few-ULP drift while still
/// separating adjacent facets of a tessellated curve.
pub const PLANE_GROUP: f64 = 1e-4;

#[cfg(test)]
mod tests {
    use super::*;

    /// The ladder, asserted. A change that inverts a rung fails here rather than
    /// in a model that mysteriously collapses.
    ///
    /// Checked in `const` blocks, so an inverted rung is a compile error rather
    /// than a failing test: these are relations between constants, and there is
    /// no reason to wait until something runs to find out.
    #[test]
    fn the_ladder_is_ordered() {
        const {
            assert!(
                WELD <= ON_SEGMENT,
                "two points that weld together must also count as lying on a segment \
                 through them, or a welded point stops splitting the seam it is on"
            );
        }
        const {
            assert!(
                PLANE_GROUP > ON_SEGMENT,
                "the face grouping has to be coarser than the on-segment test, or a \
                 face gets split into groups whose shared edge no longer registers"
            );
        }
        const {
            assert!(
                PLANE_GROUP > PLANE_GATE,
                "a point on a face's plane must still be inside that face's group"
            );
        }
    }

    /// Documents the inversion rather than asserting it away. If someone fixes
    /// the underlying defect and raises `PLANE_GATE` to meet `ON_SEGMENT`, this
    /// test fails and points at the measurements that say what to expect.
    #[test]
    fn the_plane_gate_is_tighter_than_the_segment_test() {
        const {
            assert!(
                PLANE_GATE < ON_SEGMENT,
                "PLANE_GATE has been raised to meet ON_SEGMENT. That is the right \
                 shape of fix and it has been tried: on its own it collapses the \
                 reference model to 2937 triangles at 1e-6 and explodes it to 199390 \
                 at 3e-6. Re-measure with tests/watertight_gate.rs and csg_check \
                 before trusting it."
            );
        }
    }
}
