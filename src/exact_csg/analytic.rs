//! The analytic accelerator for `corefine` — Stage 2 of the B-rep roadmap.
//!
//! When both meshes carry surface provenance ([`crate::brep`]), a candidate
//! triangle pair can be resolved by asking what their *surfaces* do instead of
//! intersecting the triangles numerically. Three things follow, in increasing
//! order of value:
//!
//! * **Disjoint** — the surfaces provably never meet, so the pair is skipped.
//!   Pure speedup; a proven negative cannot be wrong.
//! * **Coincident** — the two triangles lie on the *same* surface, so they
//!   cannot transversally cross. Any segment `tri_tri_segment` reports for such
//!   a pair is numerical noise off a near-coincidence, and injecting it produces
//!   the slivers that make the kernel give up. This is the case behind the two
//!   degeneracies named at `super::mod`'s doc comment.
//! * **Curves** — the surfaces meet along an exact curve. Reported, but *not*
//!   acted on: see [Two things that did not work](#two-things-that-did-not-work).
//!
//! It also supplies `surfaces_coincide`, which the kernel's *classifier* uses
//! to decide whether a refined sub-triangle lies on a face the other solid also
//! has. That turns out to be where the accelerator actually pays: see
//! `coincident_face` in the parent module.
//!
//! # Two things that did not work
//!
//! Both were implemented, measured, and removed. Recorded because the reasons
//! generalise.
//!
//! **Projecting seam points onto the exact curve.** A `tri_tri_segment`
//! endpoint lies on a triangle edge of one of the two meshes, and the
//! refinement downstream depends on that incidence; the exact curve is where
//! the *surfaces* meet, not where the triangles do. Moving the point onto it
//! took two overlapping spheres from `Exact` to `NeedsArrangement`.
//!
//! **Synthesising cuts for boundary contacts.** A capped cylinder's rim lies on
//! the cylinder, so a cap stacked flush against a side touches without crossing
//! and `tri_tri_segment` finds nothing. Cutting the side facet with the cap's
//! plane looked like the fix. It was not needed — the coplanar-overlap path
//! already produces those cuts — and it broke `cyl ∪ cyl parallel`, whose caps
//! are coplanar, by adding cuts that path had deliberately not made.
//!
//! # The contract
//!
//! Everything here is gated on both meshes carrying provenance for the pair in
//! question. Without it — or for any surface pair with no closed form — the
//! lookup returns `Decision::Numeric` and the caller runs exactly the code it
//! ran before this module existed.
//!
//! The testable form of that is stronger and is what the acceptance suite
//! asserts: **a tagged boolean is never worse than the same boolean with its
//! provenance stripped.** An accelerator that never fires satisfies that
//! trivially, so [`analytic_report`] exists to show separately that it does.

use std::collections::HashMap;

use crate::brep::{ssi, SsiResult, SurfaceTable};
use crate::core::BufferGeometry;
use crate::nurbs::v3;

use super::V3;

/// What the analytic layer says about a candidate triangle pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decision {
    /// Skip the pair — the surfaces provably do not meet.
    Skip,
    /// The triangles are on the same surface, so they do not cross. Take the
    /// coplanar-overlap path if they are coplanar, and nothing otherwise.
    SameSurface,
    /// Run the numeric path, then snap its endpoints onto the exact curve with
    /// [`Analytic::snap`]. The curves are not carried in the decision because
    /// that would hold a borrow across the caller's mutation of its own buffers.
    Refine,
    /// Run the numeric path unchanged.
    Numeric,
}

/// Per-boolean cache of surface-pair intersections.
///
/// Keyed by *surface* index pair, not triangle pair: a cylinder contributes one
/// surface and hundreds of triangles, so without this the same `ssi` call would
/// run once per candidate pair instead of once per boolean.
pub(crate) struct Analytic<'a> {
    a: Option<&'a SurfaceTable>,
    b: Option<&'a SurfaceTable>,
    cache: HashMap<(u32, u32), SsiResult>,
    /// How many pairs each outcome resolved — reported so a caller can see
    /// whether the accelerator did anything, rather than inferring it.
    pub skipped: usize,
    pub same_surface: usize,
    /// Pairs where the numeric tri-tri produced a usable segment.
    pub numeric_hits: usize,
    /// Seam endpoints moved by the edge root find, and the largest move.
    pub sharpened: usize,
    pub max_sharpen: f64,
}

impl<'a> Analytic<'a> {
    pub fn new(a: Option<&'a SurfaceTable>, b: Option<&'a SurfaceTable>) -> Self {
        Self {
            a,
            b,
            cache: HashMap::new(),
            skipped: 0,
            same_surface: 0,
            numeric_hits: 0,
            sharpened: 0,
            max_sharpen: 0.0,
        }
    }

    /// Is there any provenance at all? When there is not, every lookup is
    /// `Numeric` and the caller can skip the machinery entirely.
    pub fn active(&self) -> bool {
        self.a.is_some() && self.b.is_some()
    }

    /// The cached `ssi` for the surfaces of triangles `i` (mesh A) and `j`
    /// (mesh B), or `None` when either triangle is untagged.
    fn lookup(&mut self, i: usize, j: usize) -> Option<&SsiResult> {
        let (ta, tb) = (self.a?, self.b?);
        let si = ta.surface_index_of(i)? as u32;
        let sj = tb.surface_index_of(j)? as u32;
        Some(
            self.cache
                .entry((si, sj))
                .or_insert_with(|| ssi(&ta.surfaces()[si as usize], &tb.surfaces()[sj as usize])),
        )
    }

    /// The surfaces of triangles `i` and `j`, when both are tagged.
    pub fn surfaces_of(
        &self,
        i: usize,
        j: usize,
    ) -> Option<(&crate::brep::Surface, &crate::brep::Surface)> {
        let (ta, tb) = (self.a?, self.b?);
        Some((ta.surface_of(i)?, tb.surface_of(j)?))
    }

    /// Classify a candidate pair.
    pub fn decide(&mut self, i: usize, j: usize) -> Decision {
        match self.lookup(i, j) {
            Some(SsiResult::Disjoint) => {
                self.skipped += 1;
                Decision::Skip
            }
            Some(SsiResult::Coincident { .. }) => {
                self.same_surface += 1;
                Decision::SameSurface
            }
            Some(SsiResult::Curves(_)) => Decision::Refine,
            _ => Decision::Numeric,
        }
    }
}

/// What the analytic layer would do for a given pair of meshes, without running
/// the boolean.
///
/// Exists so a test can tell "the accelerator resolved nothing" from "the
/// accelerator resolved everything and it made no difference" — a
/// never-regresses property is trivially satisfied by an accelerator that never
/// fires, so the fact that it fires has to be measurable on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AnalyticReport {
    /// Candidate triangle pairs considered.
    pub candidates: usize,
    /// Pairs skipped because the surfaces provably do not meet.
    pub skipped: usize,
    /// Pairs on the same surface, so incapable of crossing.
    pub same_surface: usize,
    /// Pairs whose surfaces meet along a closed-form curve.
    pub curves: usize,
    /// Pairs with no closed form — these run the numeric path unchanged.
    pub numeric: usize,
    /// Was any provenance available at all?
    pub active: bool,
}

/// Run the decision pass over every candidate pair of two meshes and report it.
pub fn analytic_report(a: &BufferGeometry, b: &BufferGeometry) -> AnalyticReport {
    let ta = super::triangles(a);
    let tb = super::triangles(b);
    let bvh_b = super::bvh::Bvh::build(&tb);
    let mut an = Analytic::new(a.surface_table(), b.surface_table());

    let mut out = AnalyticReport {
        active: an.active(),
        ..Default::default()
    };
    let mut cand = Vec::new();
    for (i, t) in ta.iter().enumerate() {
        bvh_b.overlaps(super::aabb(t), &mut cand);
        for &j in &cand {
            out.candidates += 1;
            match an.decide(i, j) {
                Decision::Skip => out.skipped += 1,
                Decision::SameSurface => out.same_surface += 1,
                Decision::Refine => out.curves += 1,
                Decision::Numeric => out.numeric += 1,
            }
        }
    }
    out
}

/// Are two surfaces the same surface?
///
/// The predicate the kernel's `coincident_face` needs. It gates on `coplanar()`
/// — `orient3d(...) == 0.0` **exactly** — which is right for original mesh
/// triangles, whose vertices are shared verbatim, and wrong for *refined* ones,
/// whose vertices come out of the CDT recomputed and land a few ULPs off the
/// plane they belong to. The test then fails, the sub-triangle falls through to
/// a ray-parity classification whose ray starts *on* the other solid's boundary,
/// and the answer is a coin flip.
///
/// Measured on `cylinder(4, 2) ∪ the same translated 2 along its axis`: the coin
/// flip landed both-keep on some facets and neither-keep on others, so the
/// result carried duplicate triangles in one place and holes in another — 34
/// malformed edges, every one on the two rim circles, and a deferral to the
/// float kernel.
///
/// Surface identity has no such fragility: two triangles are on the same surface
/// when their surfaces are the same surface, decided by comparing axes and radii
/// rather than by asking whether a recomputed vertex is exactly where it was.
pub(crate) fn surfaces_coincide(a: &crate::brep::Surface, b: &crate::brep::Surface) -> bool {
    matches!(ssi(a, b), SsiResult::Coincident { .. })
}

/// Make a seam endpoint exact **without moving it off its edge**.
///
/// `tri_tri_segment` endpoints carry ~1e-6 of error, which is what the kernel's
/// snapping and T-junction healing exist to absorb. The endpoint lies on an edge
/// of one of the two triangles; the correct sharpening is therefore to slide it
/// *along that edge* until it sits on the other surface, which is a bracketed
/// root find on a signed distance.
///
/// Contrast the obvious alternative — projecting onto the surfaces' intersection
/// curve — which moves the point off its edge and breaks the refinement. That
/// was tried; see the module docs.
///
/// Returns `p` unchanged unless an edge is found that both contains it and
/// crosses the other surface, and the correction is smaller than `max_move`.
pub(crate) fn exact_on_edge(
    p: V3,
    a: &[V3; 3],
    b: &[V3; 3],
    sa: &crate::brep::Surface,
    sb: &crate::brep::Surface,
    max_move: f64,
) -> V3 {
    // Which edge does `p` lie on, and therefore which surface must it be moved
    // onto? An endpoint on an edge of `a` has to end up on `b`'s surface, and
    // vice versa.
    let mut best: Option<(f64, V3)> = None;
    for (tri, other) in [(a, sb), (b, sa)] {
        for k in 0..3 {
            let (e0, e1) = (tri[k], tri[(k + 1) % 3]);
            if point_to_segment(p, e0, e1) > max_move {
                continue;
            }
            let Some(q) = other.edge_crossing(e0, e1, p) else {
                continue;
            };
            let d = v3::dist(p, q);
            if d <= max_move && best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, q));
            }
        }
    }
    best.map_or(p, |(_, q)| q)
}

/// Distance from a point to a segment.
fn point_to_segment(p: V3, a: V3, b: V3) -> f64 {
    let ab = v3::sub(b, a);
    let len2 = v3::dot(ab, ab);
    if len2 < 1e-300 {
        return v3::dist(p, a);
    }
    let t = (v3::dot(v3::sub(p, a), ab) / len2).clamp(0.0, 1.0);
    v3::dist(p, v3::add(a, v3::scale(ab, t)))
}
