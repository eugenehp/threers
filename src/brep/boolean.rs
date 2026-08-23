//! Boolean operations on bodies, computed from **exact** surface intersections.
//!
//! # What this is, and what it refuses to be
//!
//! Every intersection curve here comes from [`crate::brep::ssi`] in closed form.
//! Nothing is marched, nothing is fitted, and no tolerance decides where two
//! surfaces meet. Where a face pair has no closed form — any NURBS patch, a
//! torus against a cylinder — this returns `None` rather than an approximation,
//! and the caller falls back to the mesh kernel.
//!
//! That is the same contract Stage 2 established for the accelerator, applied to
//! a whole operation: **it may be exact or it may decline, never approximate.**
//! A boolean that quietly produces a nearly-right solid is worse than one that
//! says it cannot, because the error surfaces later and somewhere else.
//!
//! # How a face is split
//!
//! An intersection curve lands on a face as a curve in that face's parameters.
//! Two shapes of split are handled, and they cover the solids these bodies are
//! made of:
//!
//! * **A closed curve lying inside a planar face.** A cylinder meeting a plate's
//!   top is a circle strictly inside the rectangle. The face becomes the outside
//!   region with the circle as a *hole*, and the inside region as its own face —
//!   the trim-loop machinery already handles both.
//! * **A curve at constant parameter on a swept face.** A plane cutting a
//!   cylinder is a circle at constant `v`, which trims the face's range.
//!
//! Anything else — a curve crossing a face's boundary, so that the split is a
//! genuine planar subdivision — needs a 2D arrangement in parameter space, and
//! returns `None`. That is the honest boundary of this implementation and the
//! next thing to build.
//!
//! # Classification
//!
//! Whether a surviving region is inside the other body is decided by ray parity
//! against that body's tessellation. The *geometry* stays exact — only the
//! in/out question is answered numerically, and it is a question about a point
//! well away from any surface, which is where ray parity is reliable.

use crate::nurbs::v3;
use crate::nurbs::V3;

use super::body::{Body, Edge, Face, Footprint, TrimLoop};
use super::{ssi, Curve3d, SsiResult, Surface};
use std::collections::{HashMap, HashSet};

use super::planar;

/// Which boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    Union,
    Difference,
    Intersection,
}

/// Why a boolean declined.
///
/// Reported rather than collapsed into `None`, because "these two surfaces have
/// no closed form" and "this split needs an arrangement" call for different
/// responses — the first is inherent, the second is the next feature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Declined {
    /// A face pair whose surfaces have no closed-form intersection.
    NoClosedForm { face_a: usize, face_b: usize },
    /// An intersection curve crosses a face's boundary, so splitting it needs a
    /// planar subdivision in parameter space.
    NeedsArrangement { face: usize },
    /// A piece of a split face had no point that could be classified as inside
    /// or outside, so whether it belongs in the result is unknown.
    ///
    /// Dropping it instead would return a solid with a face missing — which
    /// looks like success and is not. This is the difference between a boolean
    /// that declines and one that is quietly wrong.
    UnclassifiablePiece { face: usize },
    /// The pieces were classified, but what they assemble into is not closed.
    ///
    /// The last gate before returning: whatever the cause, a result with a hole
    /// in it is worse than no result, because a caller cannot tell it is wrong.
    NotWatertight { open_edges: usize },
    /// The two surfaces *touch* along their intersection rather than crossing
    /// it, so the curve pinches to a point and two of its branches meet there.
    ///
    /// A torus with an off-axis cylinder through it is the standard case: the
    /// cylinder grazes the torus's inner equator at exactly one point, the two
    /// intersection loops meet, and the curve is no longer a manifold. Resolving
    /// it needs that point as a *node* of the curve network, with four branches
    /// leaving it, which the tracer does not produce — `settle` has no direction
    /// to step in when the normals are parallel.
    ///
    /// Reported separately because it is not a fault. Left as `NotWatertight` it
    /// reads as "the kernel built something broken" when it means "this pair is
    /// not supported yet", and those want different responses.
    TangentialContact { face_a: usize, face_b: usize },
    /// A body that is not a closed solid has no inside, so no boolean.
    NotASolid,
}

/// An edge carried over from an input body: which body, its index there, its
/// points, and whether it closes.
type Carried = (u8, usize, Vec<V3>, bool, Vec<usize>);

/// Which stretch of a carried rim: the body, the edge, and its two ends welded
/// to the grid. Two pieces of one face can run along different parts of it.
type RimPart = (u8, usize, (i64, i64, i64), (i64, i64, i64));

/// One piece of an edge more than two faces claim: the span of its vertex list,
/// and the two faces that run along exactly that span.
type RimCut = ((usize, usize), Vec<usize>);

/// Half of an edge cut where the boundary turns: its vertices, and the two
/// faces whose loops name both of its ends.
type EdgeHalf = (Vec<usize>, Vec<usize>);

/// A curve crossing a face: which shared curve, its points in that face's
/// parameters, and where each of those sits along the curve.
///
/// The last part is what keeps a seam *shared*: a chord re-cut at a periodic
/// seam is ordered by parameter, so a point's position in the chord is no
/// longer its position along the curve.
type Chord = (usize, Vec<[f64; 2]>, Vec<usize>);

/// A face of the result, before it is turned back into a [`Face`].
struct Piece {
    surface: Surface,
    loops: Vec<TrimLoop>,
    u_range: (f64, f64),
    v_range: (f64, f64),
    u_wraps: bool,
    v_wraps: bool,
    flipped: bool,
    /// A point on the piece, away from its boundary, for classification.
    sample: V3,
    /// Set when this piece lies on a wall the two solids share: the other
    /// face's index, and whether the two normals point the same way.
    ///
    /// Such a piece is not inside or outside anything — it is *on* the boundary
    /// of both — so ray parity cannot classify it and the rule below decides.
    /// Nothing sets it yet: cutting the shared wall out needs the grazing
    /// contacts such a model also produces, which `clip_to_face` cannot resolve.
    shared_wall: Option<(usize, bool)>,
    /// Shared intersection curves bounding this piece, by index into the
    /// boolean's curve list. Both sides of a seam name the same one, which is
    /// what makes the result watertight rather than merely coincident.
    bounding: Vec<usize>,
    /// Edges the face already had, carried over from the input body: which body
    /// it came from, its index there, and its vertex positions.
    ///
    /// A face no curve touches keeps its boundary rather than growing a new
    /// one. Its neighbours keep the same one — the key is what makes the two
    /// resolve to a single result edge — so a rim stays a rim instead of
    /// becoming two independently-sampled circles that merely coincide.
    carried: Vec<Carried>,
}

/// An intersection curve, sampled **once** and referenced by both faces it
/// separates.
///
/// Sampling per face is the mistake this exists to prevent: two faces deriving
/// the same curve from the same `ssi` and then evaluating it independently land
/// on nearby-but-unequal points, and the result has a crack along every seam.
struct SharedCurve {
    /// Indices into the result's vertex list, in order. A closed curve repeats
    /// its first index last; an open one does not.
    vertices: Vec<usize>,
    /// Whether it closes on the faces it bounds. A closed curve is a *hole* and
    /// the region around it is the rest — a containment test settles it. An open
    /// one enters and leaves through the boundary and **cuts** the face, which
    /// no containment test can describe; that needs [`crate::brep::planar`].
    closed: bool,
}

impl Body {
    /// Boolean this body against another, exactly or not at all.
    ///
    /// `tolerance` is used for the tessellation that answers inside/outside
    /// questions, not for the geometry — the surfaces of the result are the
    /// surfaces of the inputs, unchanged.
    pub fn boolean(&self, other: &Body, op: BooleanOp, tolerance: f64) -> Result<Body, Declined> {
        if !self.is_valid_solid(tolerance) || !other.is_valid_solid(tolerance) {
            return Err(Declined::NotASolid);
        }

        // Every interacting pair must be resolvable before anything is built:
        // discovering half way through that a pair has no closed form would mean
        // returning a partially-constructed body, and there is no such thing.
        let mut curves: Vec<(usize, usize, Vec<Curve3d>)> = Vec::new();
        // Face pairs on one surface, with whether their normals oppose.
        let mut coincident: Vec<(usize, usize, bool)> = Vec::new();
        // The region a traced curve is allowed to wander in: everything either
        // solid occupies, with room to spare.
        //
        // Drawing it where *both* solids are instead — the intersection of their
        // boxes, sampled over each face's surface rather than its vertices — is
        // measurably better in two places and worse in one, isolated:
        //
        //     box alone:  ball bored then cross-bored -> NotWatertight { 77 }
        //     baseline:   the same -> Ok, 4 faces, valid
        //
        // With it, a bored ball cut twice becomes valid and a rod crossed by a
        // rod resolves and round-trips; the whole suite passes. Without it those
        // two decline.
        //
        // And the one it costs loses **nothing structural**. Opened either way,
        // the cross-bored ball is the same body:
        //
        //     baseline 4 faces 7 edges 0 defects shells [true] free [0,0,0,0]
        //              326114 tris,  0 open
        //     box      4 faces 7 edges 0 defects shells [true] free [0,0,0,0]
        //              373283 tris, 77 open
        //
        // Identical topology; only the sampling differs, and the finer sampling
        // tessellates open. So this is a fill that does not survive its input
        // getting finer, not a piece lost — which is a different repair from any
        // tried against it so far. The settle and the seam pairing that accompany it are
        // clean on their own — this one case is the box's alone, and it is the
        // last thing between that combination and landing.
        //
        // Which is a per-*model* scale answering a per-feature question, and it
        // loses intersections. `seeds` keeps a candidate only if it is further
        // than `span(bounds) / 12` from every candidate already found. A drill
        // sixty long through a ball of radius three makes this box span 66, so
        // that radius is 5.5 — and the drill's two rings on the sphere, entry and
        // exit, are 5.2 apart. The second is discarded as a duplicate of the
        // first, and the boolean gets one ring where there are two.
        //
        //     `march` on that pair, called with a box around the overlap:
        //         2 curves, 41 points each, z -2.780..-2.381 and 2.380..2.780
        //     the same pair inside the boolean, with this box:
        //         1 curve, re-sampled to 6 points
        //
        // The tracing is right and the seeding is right; the *bounds* are wrong.
        // This is the same fault as the note on `march`'s step being
        // `span(bounds) * 0.01`, which is also a model scale used for a feature.
        //
        // Narrowing it to where *both* solids are — the intersection of the two
        // boxes, since a curve on both surfaces cannot leave either — takes that
        // pair from one traced curve to **two**, which is the fix. The operation
        // then declines `NeedsArrangement { face: 0 }`, and what the arrangement
        // is handed says why:
        //
        //     subdivide face 0 (sphere): outer 272 pts area 15.4687,
        //                                paths [22, 22], already 0
        //
        // The drill's two rings arrive as 22-point *chords*, not as rings.
        //
        // Not because that outline is degenerate: its area of 15.4687 is exactly
        // 2π × 2.462, the face's `u` period by its `v` range, so it is a clean
        // rectangle and a crossing test on it is sound. And the rings sit well
        // inside it — the drill meets the sphere at z ±2.38..2.78, which is
        // v ±0.92..1.18 against a range of ±1.231, at u ≈ 0.494 with the seam at
        // ±π. Nothing about them is near an edge.
        //
        // They are chords because they are not rings by the time anything looks:
        //
        //     sample_closed on sphere: 21 pts, closed false, invert fails on 0
        //
        // Traced with a box around the overlap by hand, that pair gives two
        // *closed* rings of 41 points. Traced with the intersection box computed
        // here, it gives open curves of 21 — because `bounds` is taken over a
        // body's **vertices**, and a bored ball's vertices are only its bore
        // rims, radius 1 and z ±2.83. Intersecting that with the drill's
        // x ∈ [0.95, 1.65] leaves a sliver, and the sliver clips the rings open.
        //
        // So the intersection is the right idea and a vertex hull is the wrong
        // way to get it: a body's vertices do not span the body. Taking each
        // face's surface across its own parameter range as well, and
        // intersecting the two, brings that case back a **closed solid**:
        //
        //     3 faces 5 edges, shells closed [true], free ends [0, 0, 0]
        //     defects [EdgeOffSurface { edge: 1, deviation_scaled: 1 }]
        //
        // One shell, no free ends, every edge between two faces. What is left is
        // not the tracing — `settle` converges to `tolerance * 1e-3` — because of
        // two rings traced identically,
        //
        //     e0 sphere/cylinder 41 verts  worst off-surface 0.000000
        //     e1 sphere/cylinder 41 verts  worst off-surface 0.001162
        //
        // one is exact and the other just over a tolerance of 0.001.
        //
        // Not the weld: the deviation is 0.001162 with its quantum at
        // `tolerance * 0.5` and 0.001162 with it at `tolerance * 0.1`, to the
        // digit. Not the tracing either — marched with a box built by hand, both
        // rings come back exact to 5e-8 over 37 points.
        //
        // And not the box either, except in how coarsely it makes the marcher
        // step. Marched with each:
        //
        //     union box (shipped)  span 66.0  step 0.660  1 curve,  6 pts, worst 7e-7
        //     surface-extent isect span  6.6  step 0.066  2 curves, 37 pts, worst 5e-8
        //     tight overlap        span  6.0  step 0.060  2 curves, 41 pts, worst 4e-7
        //
        // Exact in every case. What the box decides is the step and the seed
        // spacing, not the accuracy — so the point a tolerance out is one the
        // arrangement *added*, and its size says which: a step of 0.066 across a
        // ring of radius ~0.35 has a sagitta near 0.0016, and a crossing found by
        // intersecting two polylines lands exactly there. Which is the effect
        // `near_curve` already exists to undo, recorded there as "1.3e-3 at a
        // tolerance of 1e-3".
        //
        // Applying that snap to a *hole's* crossings as well as the outer ring's
        // is correct and harmless — the whole suite stays green — and does not
        // fix this. Nor does letting `near_curve` search *every* shared curve
        // rather than only the chords', which restores the identity a ring loses
        // by arriving as an `already` path. Both green, neither moves it.
        //
        // Because the offending vertex is not a crossing at all:
        //
        //     e1 sphere/cylinder, 41 verts, 3 off
        //       idx 0/40 vertex 39: 2e-6            (the closing repeat)
        //       idx 37   vertex 79: sphere 1.35e-4, cylinder 1.162e-3
        //
        // One vertex, on the sphere and *off the cylinder*. It was minted from a
        // parameter on one surface and never settled onto both — which is a
        // different fault from a polyline crossing, and the sagitta arithmetic
        // that matched its size was a coincidence of scale.
        let box_both = {
            let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
            for b in [self, other] {
                extent_of(b, &mut lo, &mut hi);
            }
            let pad = (0..3).map(|k| hi[k] - lo[k]).fold(0.0f64, f64::max) * 0.05 + tolerance;
            (
                [lo[0] - pad, lo[1] - pad, lo[2] - pad],
                [hi[0] + pad, hi[1] + pad, hi[2] + pad],
            )
        };
        for (ia, fa) in self.faces().iter().enumerate() {
            for (ib, fb) in other.faces().iter().enumerate() {
                let (sa, sb) = (&self.surfaces()[fa.surface], &other.surfaces()[fb.surface]);
                match ssi(sa, sb) {
                    SsiResult::Disjoint => {}
                    SsiResult::Coincident { opposite } => {
                        // Two faces on one surface. Where their regions overlap
                        // the solids share a wall: it is on the boundary of
                        // *both*, so ray parity has nothing to say about it and
                        // a rule decides which copy survives.
                        if self.faces_overlap(ia, other, ib, tolerance) {
                            coincident.push((ia, ib, opposite));
                        }
                    }
                    SsiResult::Unknown => {
                        // No closed form. Trace it instead — declining a
                        // cross-drilled hole because its seam is a quartic
                        // would be declining something entirely ordinary.
                        let traced = super::intersect::march(sa, sb, box_both, tolerance);
                        let reaching: Vec<Curve3d> = traced
                            .into_iter()
                            .filter(|c| {
                                self.curve_reaches_face(ia, c, tolerance)
                                    && other.curve_reaches_face(ib, c, tolerance)
                            })
                            .collect();
                        if reaching.is_empty() {
                            continue;
                        }
                        curves.push((ia, ib, reaching));
                    }
                    SsiResult::Curves(cs) => {
                        // Surfaces meeting is not faces meeting. Two planes
                        // *always* intersect in a line — a plate's side wall and
                        // a drill's end cap included, though their trimmed
                        // regions are nowhere near each other. Keeping only the
                        // curves that actually reach both faces is what stops
                        // every plane pair in a model from demanding an
                        // arrangement it does not need.
                        let reaching: Vec<Curve3d> = cs
                            .into_iter()
                            .filter(|c| {
                                self.curve_reaches_face(ia, c, tolerance)
                                    && other.curve_reaches_face(ib, c, tolerance)
                            })
                            .collect();
                        // An intersection that is a *point* is two surfaces
                        // touching, not crossing: two spheres a diameter apart,
                        // a sphere resting on a plane. There is no curve to
                        // split a face along, and the pair used to come back
                        // `NeedsArrangement` — which is not what is wrong with
                        // it. Nothing is arranged; the two graze.
                        if reaching.iter().any(|c| matches!(c, Curve3d::Point(_))) {
                            return Err(Declined::TangentialContact {
                                face_a: ia,
                                face_b: ib,
                            });
                        }
                        if !reaching.is_empty() {
                            curves.push((ia, ib, reaching));
                        }
                    }
                }
            }
        }

        // Sample every surviving curve once, into one vertex list.
        let mut vertices: Vec<V3> = Vec::new();
        let mut shared: Vec<SharedCurve> = Vec::new();
        let mut by_pair: Vec<(usize, usize, Vec<usize>)> = Vec::new();
        for (ia, ib, cs) in &curves {
            let mut ids = Vec::with_capacity(cs.len());
            for c in cs {
                let surface = &self.surfaces()[self.faces()[*ia].surface];
                // A closed curve is only a *ring* where the whole of it lies on
                // both faces.
                //
                // A plane cutting a sphere gives a full circle, but if that
                // plane is a wall two units wide, only the arc across the wall
                // is a seam — the rest of the circle is nowhere. Taken whole it
                // becomes an island on the sphere, and the sphere comes back in
                // as many disconnected pieces as there were walls, none of them
                // meeting anything. Where it is not wholly on both, it is
                // clipped like any open curve.
                //
                // Only asked of a face with a boundary that encloses something.
                // A swept face's loops are its rims, which are straight lines in
                // parameter space and enclose nothing, so there is no region to
                // be inside of — such a face is bounded by its parameter range
                // and a curve on it is inside by construction.
                let full = match c {
                    Curve3d::Sampled { points, closed } => {
                        (if *closed {
                            points.len()
                        } else {
                            points.len() - 1
                        }) as f64
                    }
                    _ => std::f64::consts::TAU,
                };
                let all_of_it = |body: &Body, fi: usize| -> bool {
                    // The same rule as everywhere else — see `Footprint`.
                    let f = &body.faces()[fi];
                    let rings = body.face_loops(fi);
                    if matches!(
                        Footprint::of(&rings, f.u_range, f.v_range),
                        Footprint::Rectangle { .. }
                    ) {
                        return true;
                    }
                    body.clip_to_face(fi, c, tolerance).is_some_and(|spans| {
                        spans.iter().map(|(t0, t1)| t1 - t0).sum::<f64>() >= full * 0.999
                    })
                };
                let whole = sample_closed_curve(c, surface, tolerance)
                    .filter(|_| all_of_it(self, *ia) && all_of_it(other, *ib));
                if let Some(uv) = whole {
                    let start = vertices.len();
                    for p in &uv {
                        vertices.push(surface.point(p[0], p[1]));
                    }
                    let mut vs: Vec<usize> = (start..vertices.len()).collect();
                    vs.push(start);
                    shared.push(SharedCurve {
                        vertices: vs,
                        closed: true,
                    });
                    ids.push(shared.len() - 1);
                    continue;
                }

                // An *open* curve. It exists only where both faces do, so it is
                // clipped to each and the overlap is what the two share — which
                // is also what makes the two ends land on a boundary, since one
                // face or the other is what stopped it.
                let Some(mine) = self.clip_to_face(*ia, c, tolerance) else {
                    return Err(Declined::NeedsArrangement { face: *ia });
                };
                let Some(theirs) = other.clip_to_face(*ib, c, tolerance) else {
                    return Err(Declined::NeedsArrangement { face: *ib });
                };
                // Every stretch both faces carry. A curve can be on a face more
                // than once, so this is an intersection of two *sets* of
                // intervals, and each surviving one is its own seam.
                //
                // Which makes this the place a clip that flickers turns into
                // wreckage. Measured on a bore cutting the second sphere of a
                // difference, `clip_to_face` came back with 65 intervals against
                // 27, in a pattern with period exactly 1.0 —
                //
                //     (0.000, 0.359) (0.640, 1.359) (1.640, 2.359) (2.640, 3.359)
                //
                // — one cycle per *sample* of the curve, which is a containment
                // test changing its mind sample by sample rather than a curve
                // that leaves and re-enters. Crossed with the other face's 27,
                // it became 64 three-point curves and an arrangement that could
                // not be built.
                for (a0, a1) in &mine {
                    for (b0, b1) in &theirs {
                        let (t0, t1) = (a0.max(*b0), a1.min(*b1));
                        if t1 - t0 <= tolerance {
                            continue; // they only touch, or do not overlap
                        }
                        let pts = sample_open_curve(c, t0, t1, tolerance);
                        if pts.len() < 2 {
                            continue;
                        }
                        let start = vertices.len();
                        vertices.extend(pts);
                        shared.push(SharedCurve {
                            vertices: (start..vertices.len()).collect(),
                            closed: false,
                        });
                        ids.push(shared.len() - 1);
                    }
                }
            }
            // Do the two surfaces cross here, or only touch?
            //
            // Where they touch, the normals are parallel and the intersection
            // pinches instead of passing through. Measured over the sampled
            // curve, `|n1 × n2|` is the sine of the angle between them: it comes
            // to 1.8e-3 on a torus grazed by an off-axis cylinder, and to 0.87
            // or more on every pair that genuinely crosses — three orders of
            // separation, so the line between them is not a delicate one.
            {
                let sa = &self.surfaces()[self.faces()[*ia].surface];
                let sb = &other.surfaces()[other.faces()[*ib].surface];
                // Asking that of each *sample* makes the answer luck. The window
                // where the test trips does scale with the model — for two equal
                // cylinders crossing, `sin` is about `sqrt(2)·x/r` near the
                // graze, so `1e-2` is `|x| < r·7.1e-3` — but whether a sample
                // lands inside it does not scale with anything. Measured on that
                // very pair: at `r = 1` one does and the graze is reported; at
                // `r = 2` none does, and the same shape twice the size came back
                // `NotWatertight` with 82 open edges instead. Same geometry, two
                // different reasons, decided by where the samples happened to
                // fall.
                //
                // So bracket it instead of sampling it. Both normals are
                // perpendicular to the curve they meet along, so `n1 × n2` is
                // parallel to that curve's tangent: taken *along* the curve it
                // is a signed quantity, and a graze is where it passes through
                // zero. A sign change between two samples finds one lying
                // between them, which no threshold on either sample can.
                let touching = ids.iter().any(|&id| {
                    let vs = &shared[id].vertices;
                    let lean_at = |w: usize| -> Option<(f64, f64)> {
                        let p = vertices[vs[w]];
                        let q = vertices[vs[(w + 1) % vs.len()]];
                        let ((ua, va), (ub, vb)) = (sa.invert(p)?, sb.invert(p)?);
                        let (na, nb) = (sa.normal(ua, va)?, sb.normal(ub, vb)?);
                        let x = v3::cross(na, nb);
                        let step = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
                        Some((v3::dot(x, x).sqrt(), v3::dot(x, step)))
                    };
                    let last = if shared[id].closed {
                        vs.len()
                    } else {
                        vs.len().saturating_sub(1)
                    };
                    let mut before: Option<(f64, f64)> = None;
                    for w in 0..last {
                        let Some((sine, lean)) = lean_at(w) else {
                            before = None;
                            continue;
                        };
                        if sine < 1e-2 {
                            return true;
                        }
                        if let Some((was, dir)) = before {
                            if dir * lean < 0.0 && was.min(sine) < 0.2 {
                                return true;
                            }
                        }
                        before = Some((sine, lean));
                    }
                    false
                });
                if touching {
                    return Err(Declined::TangentialContact {
                        face_a: *ia,
                        face_b: *ib,
                    });
                }
            }
            by_pair.push((*ia, *ib, ids));
        }

        // Tessellations, for the in/out tests only.
        // Coarser than the model, deliberately.
        //
        // These answer one question — is this sample point inside that solid —
        // and how coarse they may be is set by how far the sample sits from its
        // own piece's boundary, because a mesh deviating further than that puts
        // the point on the wrong side. That distance is now chosen rather than
        // taken (see `sample_between`), which is what makes this safe: at
        // `span × 1e-4` a sample is typically a tenth of its piece from the
        // edge and the mesh is out by a thousandth of the model, a couple of
        // hundred times less. Before the sample was chosen, this same figure
        // put a piece on the wrong side and a plate came back with an edge
        // carrying three faces.
        //
        // Relative to the model, not absolute, because a millimetre is coarse
        // on a watch and fine on a bridge.
        //
        // These meshes answer one question — is this sample point inside that
        // solid — and a sample is chosen away from the boundary on purpose, so
        // what they need is to place it, not to be accurate. Built at the
        // modelling tolerance they were 58% of the whole boolean: 184ms of
        // 317ms at `1e-4`, and worse the finer it gets, for an answer that does
        // not change. The floor is relative to the model, because a millimetre
        // is coarse on a watch and fine on a bridge.
        let classify = {
            let span = (0..3)
                .map(|k| box_both.1[k] - box_both.0[k])
                .fold(0.0f64, f64::max);
            tolerance.max(span * 1e-4)
        };
        // A body that has already been cut carries trim loops sampled at
        // whatever tolerance cut them, and asking for a *coarser* tessellation
        // than that can come back open — a plate on its second bore did. So the
        // coarse mesh is an attempt, not a decision: if it is not a solid, the
        // modelling tolerance still is.
        let solid_at = |b: &Body| {
            let coarse = b.solid_triangles(classify);
            if coarse.is_empty() {
                b.solid_triangles(tolerance)
            } else {
                coarse
            }
        };
        let mesh_a = solid_at(self);
        let mesh_b = solid_at(other);
        if mesh_a.is_empty() || mesh_b.is_empty() {
            return Err(Declined::NotASolid);
        }

        // Every edge of either body, materialised once.
        //
        // A swept face cut across the sweep needs its parameter rectangle cut
        // open somewhere, and its rims have to reach that seam — but a rim is
        // shared with the face beyond it, which knows nothing about this face's
        // seam. So the point goes into the edge *here*, before anything is
        // split, and both faces read the same list.
        let mut carried_points: HashMap<(u8, usize), Vec<V3>> = HashMap::new();
        for (tag, body) in [(0u8, self), (1u8, other)] {
            for (i, e) in body.edges().iter().enumerate() {
                carried_points.insert(
                    (tag, i),
                    e.vertices
                        .iter()
                        .map(|&v| *body.vertices().get(v).unwrap_or(&[0.0; 3]))
                        .collect(),
                );
            }
        }

        // Cut the shared wall out of both faces.
        //
        // The cut is the other face's outline trimmed to the part lying on this
        // one — `planar::clip_to_region`, which is what makes this a polygon
        // *boolean* and not merely a subdivision. Both faces compute the same
        // crossings from the same two boundaries, so the piece each cuts out has
        // the same corners and the two weld.
        for &(ia, ib, _) in &coincident {
            for (for_a, into, from) in [
                (true, (self, ia), (other, ib)),
                (false, (other, ib), (self, ia)),
            ] {
                for pts in Body::wall_outline(into.0, into.1, from.0, from.1) {
                    if pts.len() < 2 {
                        continue;
                    }
                    let closed = v3::dist(pts[0], *pts.last().unwrap()) <= tolerance;
                    let start = vertices.len();
                    let n = if closed { pts.len() - 1 } else { pts.len() };
                    vertices.extend(pts.into_iter().take(n));
                    let mut vs: Vec<usize> = (start..vertices.len()).collect();
                    if closed {
                        vs.push(start);
                    }
                    shared.push(SharedCurve {
                        vertices: vs,
                        closed,
                    });
                    let id = shared.len() - 1;
                    // `usize::MAX` on the side that must not see it: a face is
                    // cut by the *other's* outline, never by its own.
                    if for_a {
                        by_pair.push((ia, usize::MAX, vec![id]));
                    } else {
                        by_pair.push((usize::MAX, ib, vec![id]));
                    }
                }
            }
        }

        // Weld the curves to one another before anything is split.
        //
        // Several curves can meet at one place — three surfaces through a
        // corner — and each arrives with its own endpoint vertex there, a few
        // nanometres from its neighbours'. Left apart, any attempt to name such
        // a point by the curve it came from picks a different vertex on each
        // face, and the seam opens; welded, they are one point and naming it is
        // safe. Position welding downstream hides the problem for the faces that
        // rely on it and cannot help the ones that do not.
        {
            // The radius is the tolerance, not half of it. Two points closer
            // than the tolerance are the same point — that is what a tolerance
            // means — and welding at half of it leaves pairs the kernel's own
            // definition says are one, which is the crack every later stage
            // has to work around.
            let quantum = tolerance.max(1e-12);
            let key = |p: V3| {
                let q = |x: f64| (x / quantum).round() as i64;
                (q(p[0]), q(p[1]), q(p[2]))
            };
            // Neighbouring cells too: two points a nanometre apart can still
            // fall either side of a cell boundary, and a grid that only looks in
            // its own cell leaves them separate — which is the whole problem
            // this is here to solve.
            //
            // The *radius* is half the tolerance, and that is the ceiling on how
            // far apart two arrivals at one point may be. Nanometres is what it
            // was written for and what the shipped tree produces — scanned every
            // result of a 45-case chaining corpus and there is not one pair in
            // (0.5·tol, tol], nor a single repeated ring point. But it is the
            // first thing to give when the sampling changes. Under a trace box
            // drawn where both solids are, one rim vertex arrives down the
            // sphere at (-2.8916, -0.2217, -0.7678) and down the bore wall at
            // (-2.8914, -0.2220, -0.7686): 8.9e-4 apart, inside the tolerance so
            // by definition one point, outside this radius so welded into two —
            // and the two fills then disagree by exactly the two segments around
            // it. Widening the radius to the full tolerance fixes that and is
            // otherwise inert (whole suite green, all 45 chaining cases
            // identical), so it is not shipped on its own; it is what any of the
            // four sampling changes will need first.
            //
            // Why the two arrivals are that far apart is worth knowing, because
            // it is live here and not only under that box. A crossing vertex
            // spliced into a rim is not settled onto the *other* surface the rim
            // lies on, so it sits off the shared curve — and each face then
            // mints its own from a different direction. Scanned every ring of a
            // 45-case corpus for points near another surface but not on it:
            //
            //     ball − cross      v226/v227   2.64e-3 off the cylinder
            //     rod  − cross      v80/v81     2.64e-3 off, and the *same*
            //                                   vertex reprojects 2.8e-3 apart
            //                                   between the two faces that own it
            //
            // Those results are valid solids today only because the fill takes a
            // ring vertex's position from the body vertex, not from its face's
            // own `uv` — so the two faces disagree about where the point is and
            // the disagreement never reaches the mesh. Settling such a vertex
            // onto both its surfaces, and recomputing each ring's `uv` from the
            // result, is the repair this points at; it has not been tried.
            let mut first: HashMap<(i64, i64, i64), usize> = HashMap::new();
            let mut remap: Vec<usize> = Vec::with_capacity(vertices.len());
            for (i, p) in vertices.iter().enumerate() {
                let (x, y, z) = key(*p);
                let mut found = None;
                'search: for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            if let Some(&j) = first.get(&(x + dx, y + dy, z + dz)) {
                                if v3::dist(*p, vertices[j]) <= quantum {
                                    found = Some(j);
                                    break 'search;
                                }
                            }
                        }
                    }
                }
                match found {
                    Some(j) => remap.push(j),
                    None => {
                        first.insert((x, y, z), i);
                        remap.push(i);
                    }
                }
            }
            for curve in &mut shared {
                for v in &mut curve.vertices {
                    *v = remap[*v];
                }
                curve.vertices.dedup();
                if curve.closed && curve.vertices.len() > 1 {
                    let last = curve.vertices.len() - 1;
                    if curve.vertices[0] != curve.vertices[last] {
                        curve.vertices.push(curve.vertices[0]);
                    }
                }
            }
        }

        // Seam vertices go in *after* the weld, not before.
        //
        // The weld deduplicates coincident points, and a seam vertex sits on a
        // curve between two samples — close enough to one of them, on a curve
        // traced finely, to be welded away again. The cylinder of an off-axis
        // torus lost exactly one that way: the vertex went in at the seam, the
        // weld took it out, and the chord then started a sample step short of
        // the boundary, so the subdivision saw a cut stopping in open space.
        //
        // It also has to be after the shared-wall cut, which adds curves of its
        // own that need the same treatment.
        // Cut a carried rim where an intersection curve meets it.
        //
        // A rim carried over from an input body is a polyline of that body's
        // own points, and a curve crossing the result runs onto it at a place
        // that polyline has no vertex for. The two sides then describe the same
        // seam differently: the rim passes straight through while the curve
        // stops, and three segments meet at a point that ought to be a corner.
        // No amount of care about *labelling* those points fixes it — four
        // attempts, and each left the same T-junction — because the points
        // themselves do not line up until one is put there.
        //
        // Only a curve's *ends* matter. Its interior lies across the face, not
        // along the rim; where it stops is where the boundary turns.
        for pts in carried_points.values_mut() {
            if pts.len() < 2 {
                continue;
            }
            for curve in &shared {
                for &vi in [curve.vertices.first(), curve.vertices.last()]
                    .into_iter()
                    .flatten()
                {
                    let Some(&at) = vertices.get(vi) else {
                        continue;
                    };
                    if pts.iter().any(|p| v3::dist(*p, at) <= tolerance) {
                        continue;
                    }
                    // Between the neighbours it falls between, and only if it
                    // really is on the rim rather than merely near it.
                    let mut best = (f64::MAX, 0usize);
                    for i in 0..pts.len() - 1 {
                        let detour = v3::dist(pts[i], at) + v3::dist(at, pts[i + 1])
                            - v3::dist(pts[i], pts[i + 1]);
                        if detour < best.0 {
                            best = (detour, i + 1);
                        }
                    }
                    if best.0 <= tolerance {
                        pts.insert(best.1, at);
                    }
                }
            }
        }

        let mut seam_origins: HashMap<(u8, usize), f64> = HashMap::new();
        // Where each face's seam was actually cut. The outline has to be broken
        // at the same places, and only this pass knows them.
        let mut seam_cuts: HashMap<(u8, usize), Vec<(f64, usize)>> = HashMap::new();
        for (tag, body) in [(0u8, self), (1u8, other)] {
            for (fi, face) in body.faces().iter().enumerate() {
                if !face.u_wraps {
                    continue;
                }
                let mine: Vec<usize> = by_pair
                    .iter()
                    .filter(|(a, b, _)| if tag == 0 { *a == fi } else { *b == fi })
                    .flat_map(|(_, _, ids)| ids.iter().copied())
                    .collect();
                if mine.is_empty() {
                    continue;
                }
                let surface = &body.surfaces()[face.surface];
                let Some(origin) = seam_origin(face, surface, &mine, &shared, &vertices) else {
                    continue;
                };
                for &e in &face.edges {
                    if let Some(pts) = carried_points.get_mut(&(tag, e)) {
                        insert_at_u(pts, surface, origin);
                    }
                }
                // And into the curves themselves. A curve that wraps this face
                // is cut open at the seam, and the point it is cut at has to be
                // a *vertex* — the face on the other side of that curve joins
                // the same points with an edge straight across, and one face
                // splitting that edge while the other does not is exactly the
                // gap the two then leave.
                // Remember where the seam went. Working it out a second time
                // gives a *different* answer: the seam goes in the widest
                // stretch of `u` no curve occupies, and the vertex inserted
                // below fills that stretch, so the next call picks somewhere
                // else. The sphere's seam vertex went in at 3.069 and the face
                // was then cut at 1.498, which left six edges open on a bored
                // sphere — the two faces of a seam described it with points
                // neither had.
                seam_origins.insert((tag, fi), origin);
                for &id in &mine {
                    // The surface across this curve: the seam vertex is on it
                    // as well as on this one.
                    let across = by_pair.iter().find_map(|(a, b, ids)| {
                        if !ids.contains(&id) {
                            return None;
                        }
                        let (ours, theirs) = if tag == 0 { (*a, *b) } else { (*b, *a) };
                        if ours != fi {
                            return None;
                        }
                        let them = if tag == 0 { other } else { self };
                        them.faces()
                            .get(theirs)
                            .map(|f| them.surfaces()[f.surface].clone())
                    });
                    if let Some(v) = insert_seam_vertex(
                        &mut shared[id],
                        &mut vertices,
                        surface,
                        across.as_ref(),
                        origin,
                        face.u_range,
                    ) {
                        seam_cuts.entry((tag, fi)).or_default().push(v);
                    }
                }
            }
        }

        // Give a two-point curve a midpoint.
        //
        // The arrangement records which curve a piece's boundary came from on
        // the curve's *interior* points, so a curve with none — two samples and
        // nothing between — leaves no trace, and `from_pieces` makes no `Edge`
        // for it. That is not a rare shape: a plane meets a cylinder in a
        // straight line, and two samples describe a line exactly. Measured on a
        // bored ball cut by a column, all eight of the wall's chords had two
        // points and every one of its nine pieces came back claiming its whole
        // outline was the face's own boundary — so the wall was bounded by rim
        // arcs alone, with four free ends apiece, and the solid never closed.
        //
        // Splitting it is exact rather than a refinement: a curve sampled at two
        // points is straight to within tolerance by construction, since that is
        // what the sampler decided when it stopped there.
        for c in shared.iter_mut() {
            if c.closed || c.vertices.len() != 2 {
                continue;
            }
            let (a, b) = (vertices[c.vertices[0]], vertices[c.vertices[1]]);
            vertices.push(v3::scale(v3::add(a, b), 0.5));
            c.vertices.insert(1, vertices.len() - 1);
        }

        // A name for every point of every carried rim, made once.
        //
        // `carried_points` is already shared — one list per edge, which every
        // face along it reads — so the geometry of a rim never disagreed. Its
        // *identity* did: the outline threw the edge away and kept the points,
        // and the result recovered which vertex was which by comparing
        // positions, in thirty-seven places. A point that two faces round to
        // differently is then two vertices at one spot, which is what a seam
        // coming apart looks like.
        //
        // Named here instead, in the numbering the result uses, and reusing a
        // vertex a shared curve already put at that place rather than adding a
        // second one beside it.
        let mut carried_vertices: HashMap<(u8, usize), Vec<usize>> = HashMap::new();
        {
            // On a grid, not by scanning. Every rim point against every vertex
            // is quadratic and showed it — the chained-CSG suite went from 7
            // seconds to 51 — and the question is only "is one already here",
            // which a cell answers.
            let cell = tolerance.max(1e-12);
            let key_of = |p: V3| -> (i64, i64, i64) {
                (
                    (p[0] / cell).round() as i64,
                    (p[1] / cell).round() as i64,
                    (p[2] / cell).round() as i64,
                )
            };
            let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
            for (i, q) in vertices.iter().enumerate() {
                grid.entry(key_of(*q)).or_default().push(i);
            }
            for (&key, pts) in carried_points.iter() {
                let mut ids = Vec::with_capacity(pts.len());
                for p in pts {
                    let (x, y, z) = key_of(*p);
                    let mut found = None;
                    'search: for dx in -1..=1 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                for &i in grid.get(&(x + dx, y + dy, z + dz)).into_iter().flatten()
                                {
                                    if v3::dist(*p, vertices[i]) <= tolerance * 0.5 {
                                        found = Some(i);
                                        break 'search;
                                    }
                                }
                            }
                        }
                    }
                    ids.push(match found {
                        Some(v) => v,
                        None => {
                            vertices.push(*p);
                            grid.entry((x, y, z)).or_default().push(vertices.len() - 1);
                            vertices.len() - 1
                        }
                    });
                }
                carried_vertices.insert(key, ids);
            }
        }

        let mut pieces: Vec<Piece> = Vec::new();
        let wall_of_a: HashMap<usize, (usize, bool)> = coincident
            .iter()
            .map(|&(ia, ib, opposite)| (ia, (ib, !opposite)))
            .collect();
        let wall_of_b: HashMap<usize, (usize, bool)> = coincident
            .iter()
            .map(|&(ia, ib, opposite)| (ib, (ia, !opposite)))
            .collect();

        // A's faces, split by the curves B puts on them.
        for (ia, fa) in self.faces().iter().enumerate() {
            let mine: Vec<usize> = by_pair
                .iter()
                .filter(|(a, _, _)| *a == ia)
                .flat_map(|(_, _, ids)| ids.iter().copied())
                .collect();
            let mut made = self.split_face(
                0,
                ia,
                fa,
                &mine,
                &shared,
                &vertices,
                tolerance,
                &carried_points,
                &carried_vertices,
                &seam_origins,
                &seam_cuts,
            )?;
            if let Some(&(fj, same_way)) = wall_of_a.get(&ia) {
                for piece in &mut made {
                    if other.point_on_face(fj, piece.sample, tolerance) {
                        piece.shared_wall = Some((fj, same_way));
                    }
                }
            }
            pieces.extend(made);
        }
        let from_a = pieces.len();
        for (ib, fb) in other.faces().iter().enumerate() {
            let mine: Vec<usize> = by_pair
                .iter()
                .filter(|(_, b, _)| *b == ib)
                .flat_map(|(_, _, ids)| ids.iter().copied())
                .collect();
            let mut made = other.split_face(
                1,
                ib,
                fb,
                &mine,
                &shared,
                &vertices,
                tolerance,
                &carried_points,
                &carried_vertices,
                &seam_origins,
                &seam_cuts,
            )?;
            if let Some(&(fi, same_way)) = wall_of_b.get(&ib) {
                for piece in &mut made {
                    if self.point_on_face(fi, piece.sample, tolerance) {
                        piece.shared_wall = Some((fi, same_way));
                    }
                }
            }
            pieces.extend(made);
        }

        // Keep what the operation asks for. A difference turns B's surviving
        // walls inward — they become the cavity, and a cavity's normals face the
        // material, not the void.
        let mut kept: Vec<Piece> = Vec::new();
        for (i, mut piece) in pieces.into_iter().enumerate() {
            let from_this = i < from_a;
            let inside_other = if from_this {
                point_in_mesh(&mesh_b, piece.sample)
            } else {
                point_in_mesh(&mesh_a, piece.sample)
            };
            let (keep, flip) = match piece.shared_wall {
                // A wall the two solids share. It is on the boundary of both, so
                // it is neither inside nor outside and the rule below decides.
                //
                // Facing the *same* way, both solids lie on the same side: the
                // wall survives a union and an intersection, once, and a
                // difference removes the material behind it. Facing each other,
                // the wall is between them: it is interior to a union, has no
                // thickness in an intersection, and a difference leaves the near
                // one untouched because nothing was taken from that side.
                Some((_, same_way)) => match (op, from_this, same_way) {
                    (BooleanOp::Union, true, true) => (true, false),
                    (BooleanOp::Intersection, true, true) => (true, false),
                    (BooleanOp::Difference, _, true) => (false, false),
                    (BooleanOp::Difference, true, false) => (true, false),
                    _ => (false, false),
                },
                None => match (op, from_this, inside_other) {
                    (BooleanOp::Union, _, inside) => (!inside, false),
                    (BooleanOp::Intersection, _, inside) => (inside, false),
                    (BooleanOp::Difference, true, inside) => (!inside, false),
                    (BooleanOp::Difference, false, inside) => (inside, true),
                },
            };
            if keep {
                piece.flipped ^= flip;
                kept.push(piece);
            }
        }

        let result = Body::from_pieces(kept, vertices, shared, tolerance);

        // The last gate. Everything above is checked, and it is still possible
        // for the pieces to assemble into something with a hole in it — a face
        // whose parameter range does not quite meet its neighbour's, a rim that
        // two faces describe differently. A result like that is *worse* than no
        // result, because it looks like success and a caller cannot tell.
        //
        // So the promise this layer makes is enforced here rather than argued
        // for: what comes back is closed, or nothing does. It costs one
        // tessellation, which is the right price for the difference between a
        // boolean that declines and one that is quietly wrong.
        // Topology first: it is the cheaper half of the promise and the half
        // closure cannot see. An edge with three faces on it is not an open
        // edge, so a result carrying one passed this gate and looked like
        // success.
        let mut check = result.clone();
        check.refine_edges(tolerance);
        let (_, report) = check.tessellate(tolerance);
        if !report.is_closed() || report.carried_through > 0 {
            return Err(Declined::NotWatertight {
                open_edges: report.boundary_edges,
            });
        }
        Ok(result)
    }
    /// Is `p` within this face's trimmed region?
    fn point_on_face(&self, fi: usize, p: V3, tolerance: f64) -> bool {
        let surface = &self.surfaces()[self.faces()[fi].surface];
        if surface.distance(p) > tolerance.max(1e-9) {
            return false;
        }
        let Some((u, v)) = surface.invert(p) else {
            return false;
        };
        let loops = self.face_loops(fi);
        // Asked of the derived rings, by the same rule the tessellator uses on
        // the stored ones — see `Footprint`.
        //
        // A whole cylinder wall is bounded by its two rims, and a rim is a line
        // at constant `v` with no area — so the first ring that is not a hole is
        // a *rim*, and `point_in_ring` against a line is false for every point,
        // including every point of the face's own middle.
        let face = &self.faces()[fi];
        let inside = match Footprint::of(&loops, face.u_range, face.v_range) {
            Footprint::Rings(rings) => rings
                .iter()
                .find(|l| l.is_outer())
                .is_some_and(|outer| outer.contains([u, v], surface.period())),
            Footprint::Rectangle { u: ur, v: vr } => {
                let within = |x: f64, r: (f64, f64), period: Option<f64>| {
                    let (lo, hi) = (r.0.min(r.1), r.0.max(r.1));
                    if x >= lo - tolerance && x <= hi + tolerance {
                        return true;
                    }
                    match period {
                        Some(t) => {
                            let folded = x - t * ((x - (lo + hi) * 0.5) / t).round();
                            folded >= lo - tolerance && folded <= hi + tolerance
                        }
                        None => false,
                    }
                };
                let (pu, pv) = surface.period();
                within(u, ur, pu) && within(v, vr, pv)
            }
        };
        inside
            && !loops
                .iter()
                .filter(|l| l.is_hole())
                .any(|h| h.contains([u, v], surface.period()))
    }

    /// The other face's outline, trimmed to the part lying on this one.
    ///
    /// Two coincident faces need not contain one another — that is what makes a
    /// shared wall need a boolean and not just a cut — so the outline is clipped
    /// to this face's region first. What comes back is a chord where the two
    /// boundaries cross, and a closed ring where the other sits wholly inside.
    /// Faces that are the *same* region need no cut at all.
    fn wall_outline(body: &Body, fi: usize, other: &Body, fj: usize) -> Vec<Vec<V3>> {
        let surface = &body.surfaces()[body.faces()[fi].surface];
        let mine = body.face_loops(fi);
        let Some(outer) = mine.iter().find(|l| l.is_outer()) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for ring in other.face_loops(fj) {
            if ring.is_hole() {
                continue;
            }
            let mut uv: Vec<[f64; 2]> = ring
                .vertices
                .iter()
                .filter_map(|&v| other.vertices().get(v))
                .filter_map(|p| surface.invert(*p))
                .map(|(u, v)| [u, v])
                .collect();
            if uv.len() < 3 {
                continue;
            }
            uv.push(uv[0]);
            if same_ring(&outer.uv, &uv) {
                continue; // the wall is the whole of both
            }
            for run in planar::clip_to_region(&outer.uv, &uv) {
                if run.len() < 2 {
                    continue;
                }
                out.push(run.iter().map(|p| surface.point(p[0], p[1])).collect());
            }
        }
        out
    }

    /// Do two faces on one surface actually share any *area* of it?
    ///
    /// Two questions have to be told apart, and vertex containment answers
    /// neither. Faces that are the same region share all of it, and every one of
    /// their vertices sits on the other's boundary. Faces that merely *adjoin* —
    /// which every pair of walls of two solids side by side does, on the same
    /// plane — share none of it, and their vertices sit on each other's boundary
    /// too.
    ///
    /// What separates them is whether an *interior* point of one falls in the
    /// other, or whether their boundaries properly cross.
    fn faces_overlap(&self, fi: usize, other: &Body, fj: usize, tolerance: f64) -> bool {
        let surface = &self.surfaces()[self.faces()[fi].surface];
        let mine = self.face_loops(fi);
        let theirs = other.face_loops(fj);
        // Neither face need state a region to share one. Two whole cylinder
        // walls are each bounded by two rims and nothing else, and comparing
        // rims finds no overlap however completely the walls coincide — which is
        // how a solid cut twice by one tool kept both copies of the wall between
        // them and came back an eighth smaller.
        // The same rule as everywhere else — see `Footprint`.
        let bounded = |ls: &[TrimLoop], f: &Face| {
            matches!(Footprint::of(ls, f.u_range, f.v_range), Footprint::Rings(_))
        };
        if !bounded(&mine, &self.faces()[fi]) || !bounded(&theirs, &other.faces()[fj]) {
            let face = &self.faces()[fi];
            let (ur, vr) = (face.u_range, face.v_range);
            const N: usize = 6;
            return (1..N).any(|i| {
                (1..N).any(|j| {
                    let u = ur.0 + (ur.1 - ur.0) * i as f64 / N as f64;
                    let v = vr.0 + (vr.1 - vr.0) * j as f64 / N as f64;
                    let p = surface.point(u, v);
                    self.point_on_face(fi, p, tolerance) && other.point_on_face(fj, p, tolerance)
                })
            });
        }
        let (Some(a), Some(b)) = (
            mine.iter().find(|l| l.is_outer()),
            theirs.iter().find(|l| l.is_outer()),
        ) else {
            return false;
        };
        // Theirs in *my* parameters, since the surface is the same one.
        let theirs_here: Vec<[f64; 2]> = b
            .vertices
            .iter()
            .filter_map(|&v| other.vertices().get(v))
            .filter_map(|p| surface.invert(*p))
            .map(|(u, v)| [u, v])
            .collect();
        if theirs_here.len() < 3 || a.uv.len() < 3 {
            return false;
        }

        // An interior point of either inside the other.
        for (ring, against) in [(&a.uv, &theirs_here), (&theirs_here, &a.uv)] {
            if let Some(p) = interior_point(ring) {
                if point_in_ring(against, p) {
                    return true;
                }
            }
        }

        // Or their boundaries crossing, which is how two regions overlap
        // partly without either's interior point falling in the other.
        let n = a.uv.len();
        let m = theirs_here.len();
        for i in 0..n {
            for j in 0..m {
                if segments_properly_cross(
                    a.uv[i],
                    a.uv[(i + 1) % n],
                    theirs_here[j],
                    theirs_here[(j + 1) % m],
                ) {
                    return true;
                }
            }
        }
        let _ = tolerance;
        false
    }

    /// Does any point of `curve` lie within this face's trim region?
    ///
    /// Sampled rather than solved: the question is only ever used to *discard*
    /// pairs, and a sample dense enough to cross the face is enough for that. A
    /// missed grazing contact would show up as a face that is not split, which
    /// the closure check then catches — where a wrongly *kept* pair would send
    /// the whole boolean down the arrangement path for nothing.
    fn curve_reaches_face(&self, face_index: usize, curve: &Curve3d, tolerance: f64) -> bool {
        use std::f64::consts::TAU;
        let face = &self.faces()[face_index];
        let surface = &self.surfaces()[face.surface];
        let loops = self.face_loops(face_index);

        // An unbounded line is sampled over the face's own reach.
        let extent = {
            let (u0, u1) = face.u_range;
            let (v0, v1) = face.v_range;
            (u1 - u0).abs().max((v1 - v0).abs()).max(1.0) * 2.0
        };
        const N: usize = 64;
        let sample = |i: usize| -> V3 {
            match curve {
                Curve3d::Line { .. } => curve.point(-extent + 2.0 * extent * i as f64 / N as f64),
                Curve3d::Point(p) => *p,
                // A traced curve is parameterised by sample index.
                Curve3d::Sampled { points, closed } => {
                    let last = if *closed {
                        points.len()
                    } else {
                        points.len() - 1
                    };
                    curve.point(last as f64 * i as f64 / N as f64)
                }
                _ => curve.point(TAU * i as f64 / N as f64),
            }
        };

        for i in 0..=N {
            let p = sample(i);
            // The same question `clip_to_face` was asking wrongly: a sampled
            // curve is walked along its chords, and a chord sags off a curved
            // surface, so a point between two samples can fail this while the
            // curve runs right through the face. It matters less here — a
            // sagging sample is *skipped* rather than the curve rejected, and
            // one surviving sample is enough — but the sampling is 64 evenly
            // spaced positions along a curve of however many points, so most of
            // them do fall between two of its own.
            //
            // Exempting sampled curves, as `clip_to_face` now does, changes
            // nothing measurable: same suite, same chaining, the same three
            // failures with the same open-edge counts. So it is not here. The
            // hazard is real and latent, and this is where it lives.
            if surface.distance(p) > tolerance.max(1e-9) {
                continue;
            }
            let Some((mut u, mut v)) = surface.invert(p) else {
                continue;
            };
            // Into *this face's* range, not the surface's natural one.
            //
            // A face cut open at a seam runs its `u` from wherever that seam
            // fell — a bored ball's sphere goes from 3.069 to 9.352 — while
            // `invert` answers in the surface's own terms. A curve at `u = 0.49`
            // is at 6.77 as far as this face is concerned, and testing the raw
            // value against the face's loops rejects it out of hand. A drill
            // through a bored ball found its curve on the sphere, was told the
            // curve did not reach the face, and the second cut declined.
            let (pu, pv) = surface.periodic();
            if pu {
                u = face.u_range.0 + (u - face.u_range.0).rem_euclid(TAU);
            }
            if pv {
                v = face.v_range.0 + (v - face.v_range.0).rem_euclid(TAU);
            }
            let in_footprint = u >= face.u_range.0 - 1e-9
                && u <= face.u_range.1 + 1e-9
                && v >= face.v_range.0 - 1e-9
                && v <= face.v_range.1 + 1e-9;
            // A swept face's "loops" are its rims, which map to *straight lines*
            // at constant `v` and enclose no area — containment in them is
            // meaningless and rejects everything. Such a face is bounded by its
            // footprint instead, which is what made the whole drill disappear
            // from a difference.
            let degenerate_loops = loops.iter().all(|l| l.area.abs() < 1e-12);
            if loops.is_empty() || degenerate_loops {
                if in_footprint {
                    return true;
                }
                continue;
            }
            let Some(outer) = loops.iter().find(|l| l.is_outer()) else {
                continue;
            };
            // Strictly inside. A curve running *along* the boundary does not
            // cut the face — the boundary already describes it — and counting
            // it as reaching sends the whole boolean looking for a split that
            // is not there.
            if deep_in_ring(&outer.uv, [u, v], ring_margin(&outer.uv))
                && !loops
                    .iter()
                    .filter(|l| l.is_hole())
                    .any(|h| h.contains([u, v], surface.period()))
            {
                return true;
            }
        }
        false
    }

    /// The parameter span of `curve` lying inside face `fi`'s trimmed region.
    ///
    /// A surface intersection is unbounded — two planes meet in a whole line —
    /// but the *faces* are not, so the seam is only the stretch both of them
    /// carry. Found by sampling and then bisecting the two ends to tolerance,
    /// which is the same accuracy the vertices are stored at.
    ///
    /// `None` if the curve enters and leaves more than once: that is several
    /// chords, and pretending it is one would join pieces that are apart.
    fn clip_to_face(&self, fi: usize, curve: &Curve3d, tolerance: f64) -> Option<Vec<(f64, f64)>> {
        use std::f64::consts::TAU;
        let face = &self.faces()[fi];
        let surface = &self.surfaces()[face.surface];
        let loops = self.face_loops(fi);
        // A face may have no boundary that encloses anything — a whole sphere
        // has no edges at all, and a swept face's loops are rims that are
        // straight lines in parameter space. Such a face is bounded by its
        // *parameter range* instead, and every point of the surface within that
        // range is on it. `curve_reaches_face` already reads them that way;
        // refusing here instead left a sphere with no clippable region and the
        // whole operation declined.
        let outer = loops.iter().find(|l| l.is_outer());
        let margin = outer.map(|o| ring_margin(&o.uv)).unwrap_or(0.0);

        let (lo, hi) = match curve {
            Curve3d::Line { .. } => {
                // Far enough to leave the face at both ends.
                let reach = self
                    .vertices()
                    .iter()
                    .map(|p| v3::norm(v3::sub(*p, curve.point(0.0))))
                    .fold(0.0f64, f64::max)
                    .max(1.0)
                    * 2.0;
                (-reach, reach)
            }
            Curve3d::Point(_) => return None,
            Curve3d::Sampled { points, closed } => {
                let last = if *closed {
                    points.len()
                } else {
                    points.len() - 1
                };
                (0.0, last as f64)
            }
            _ => (0.0, TAU),
        };

        // A sampled curve *is* its polyline, and `point` walks the chords. On a
        // curved surface a chord sags away from it, so asking whether an
        // interpolated point lies on the surface asks the wrong question and
        // gets a different answer at each end of every sample step: the clip
        // came back "on the face" for about seven tenths of each step and off it
        // for the rest, sixty-five times along a sixty-five point curve. The
        // curve was traced onto both surfaces; it does not leave them between
        // samples, only the straight line drawn through it does.
        let sagging = matches!(curve, Curve3d::Sampled { .. });
        let inside = |t: f64| -> bool {
            let p = curve.point(t);
            if !sagging && surface.distance(p) > tolerance.max(1e-9) {
                return false;
            }
            let Some((mut u, mut v)) = surface.invert(p) else {
                return false;
            };
            // Into *this face's* range. A face cut open at a seam runs its `u`
            // from wherever that seam fell, while `invert` answers in the
            // surface's own terms — the same mismatch that had
            // `curve_reaches_face` throwing away curves it had just found.
            let (pu, pv) = surface.periodic();
            if pu {
                u = face.u_range.0 + (u - face.u_range.0).rem_euclid(std::f64::consts::TAU);
            }
            if pv {
                v = face.v_range.0 + (v - face.v_range.0).rem_euclid(std::f64::consts::TAU);
            }
            let within = match outer {
                // Strictly, so a curve running along the boundary reads as
                // outside throughout rather than as a scatter of in and out.
                Some(o) => deep_in_ring(&o.uv, [u, v], margin),
                None => {
                    let face = &self.faces()[fi];
                    let pad = 1e-9;
                    u >= face.u_range.0 - pad
                        && u <= face.u_range.1 + pad
                        && v >= face.v_range.0 - pad
                        && v <= face.v_range.1 + pad
                }
            };
            within
                && !loops
                    .iter()
                    .filter(|l| l.is_hole())
                    .any(|h| h.contains([u, v], surface.period()))
        };

        // Where a curve *runs along* the boundary rather than crossing it, its
        // answer is a coin toss: asked "is this inside" of a point on the edge,
        // the predicate gives whatever rounding decides.
        //
        // Two ways to pick those samples out have been tried and neither is it.
        // Distance to the boundary catches every chord's *endpoint* as well —
        // that is what makes a chord a chord — and treating those as ambiguous
        // stops a seam reaching the edge it lands on; five tests, first-order
        // ones among them, say so. Direction is sound in principle and fires on
        // nothing here: the drill grazing a bored ball still clips into six
        // pieces and seven, unchanged, so whatever fragments that circle is not
        // the curve lying along an edge.
        const N: usize = 512;
        let at = |i: usize| lo + (hi - lo) * i as f64 / N as f64;
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut start: Option<usize> = None;
        for i in 0..=N {
            if inside(at(i)) {
                start.get_or_insert(i);
            } else if let Some(s) = start.take() {
                runs.push((s, i - 1));
            }
        }
        if let Some(s) = start {
            runs.push((s, N));
        }
        if runs.is_empty() {
            return None;
        }

        // Bisect outward from the last sample known to be inside.
        //
        // Run to precision rather than to the modelling tolerance. These
        // endpoints are not merely *near* the face's boundary — they have to
        // land *on* it, and on each other: two chords meeting at a corner of the
        // other solid are the same point, and the subdivision has to see them as
        // one. Stopping at `tolerance` leaves them a micron apart, which reads
        // as a cut stopping in open space.
        let edge = |inside_at: f64, outside_at: f64| -> f64 {
            let (mut i, mut o) = (inside_at, outside_at);
            let span = (hi - lo).abs().max(1.0);
            for _ in 0..80 {
                let m = 0.5 * (i + o);
                if inside(m) {
                    i = m;
                } else {
                    o = m;
                }
                if (i - o).abs() <= span * 1e-14 {
                    break;
                }
            }
            0.5 * (i + o)
        };
        // Every stretch, not just the first.
        //
        // A circle crossing a strip is inside it *twice*, and a curve entering
        // and leaving a face more than once is ordinary — a sphere cut by a
        // narrow wall does it. Returning only one of them, or refusing because
        // there is more than one, loses the rest of the seam.
        let spans: Vec<(f64, f64)> = runs
            .into_iter()
            .filter_map(|(a, b)| {
                let t0 = if a == 0 {
                    at(0)
                } else {
                    edge(at(a), at(a - 1))
                };
                let t1 = if b == N {
                    at(N)
                } else {
                    edge(at(b), at(b + 1))
                };
                (t1 > t0).then_some((t0, t1))
            })
            .collect();
        (!spans.is_empty()).then_some(spans)
    }

    /// Split one face by the shared curves that land on it.
    #[allow(clippy::too_many_arguments)]
    fn split_face(
        &self,
        tag: u8,
        face_index: usize,
        face: &Face,
        curve_ids: &[usize],
        shared: &[SharedCurve],
        vertices: &[V3],
        tolerance: f64,
        carried_points: &HashMap<(u8, usize), Vec<V3>>,
        carried_vertices: &HashMap<(u8, usize), Vec<usize>>,
        seam_origins: &HashMap<(u8, usize), f64>,
        seam_cuts: &HashMap<(u8, usize), Vec<(f64, usize)>>,
    ) -> Result<Vec<Piece>, Declined> {
        let surface = self.surfaces()[face.surface].clone();
        // A carried-through loop keeps its parameters but *not* its vertex
        // indices: those number the input body, and the result numbers its own
        // shared-curve vertices. Left alone they silently resolve to whatever
        // happens to sit at that index — a plate's corners came back at the
        // bore's radius, collapsing four walls onto the drill. Only the seam
        // rings below carry indices, which is the point: they are the vertices
        // both sides of a seam must agree on.
        let mut loops = self.face_loops(face_index);
        for l in &mut loops {
            l.vertices = vec![usize::MAX; l.uv.len()];
        }

        // Taken from the map rather than from this body, because a seam point
        // one face needs on a rim must be on the *same* rim the face beyond it
        // uses. Building each copy here independently is what put the two a
        // point apart.
        let carried: Vec<Carried> = face
            .edges
            .iter()
            .filter_map(|&e| {
                let edge = self.edges().get(e)?;
                let pts = carried_points.get(&(tag, e))?.clone();
                let ids = carried_vertices.get(&(tag, e)).cloned().unwrap_or_default();
                Some((tag, e, pts, edge.closed, ids))
            })
            .collect();

        if curve_ids.is_empty() {
            let Some(sample) = self.face_sample(face_index, &surface, &loops) else {
                return Err(Declined::UnclassifiablePiece { face: face_index });
            };
            // Reproduce the face as it was: its own edges, and no trim loops to
            // contradict them.
            //
            // Carrying both is how a rim comes to be described twice — a fixed
            // polyline beside an edge that refines, diverging the moment the
            // boundary is curved.
            //
            // It has a cost on a *second* cut. A cross-hole's wall keeps the
            // whole drill's parameter range and is trimmed to the rod by its
            // loop alone; dropping that loop fills the wall from end to end of
            // the drill, five units outside the solid, and a rod with a
            // cross-hole re-cut opens 490 edges. Keeping it instead costs 1.3%
            // of a blind hole's volume, because the loop is coarser than the
            // edges beside it — which is the trim-loop refinement problem, still
            // open, and this is a second thing waiting on it.
            //
            // A third, now measured. This is where `ball - cross` loses the
            // three rings its sphere is described by:
            //
            //     after `ball - cross`   sphere  loops Some([130, 63, 63])
            //     after `- bore`         sphere  loops None
            //
            // A sphere with holes is not a parameter rectangle, so with nothing
            // stored the grid fill takes that face, fails, and it is dropped
            // whole — `carried_through: 1`, and the four faces around it left
            // holding 407 open edges against a face that is not there. It is the
            // largest single failure in the chaining corpus and it is this line.
            //
            // And it is not refinement that makes them diverge — the ring is
            // *stale*. Opened `a_bore_across_a_rod`'s union with the rings kept:
            // six open edges, all at the bore's end caps, each one the cap's
            // ring cutting a corner off the wall's. The face list says why —
            // caps holding rings of 16 points, walls holding 78. A face the
            // boolean does not split arrives with the sampling it already had,
            // while the boundary beside it was re-sampled, and no amount of
            // refining either one afterwards makes 16 agree with 78.
            //
            // Which points at a repair the three runs before this one missed:
            // restate every ring in the vertices of the edges it runs along,
            // matching by *endpoint* rather than by adjacency, so the cap's step
            // from one rim vertex to another becomes the rim's own five points.
            // Built (`body.rs`, after the edges move) it closes that union
            // outright and is green on its own, and keeping the rings then costs
            // six boolean tests instead of nine. Not enough to ship either
            // half — but the first thing in four attempts that moved the number.
            //
            // Three ways of keeping them were tried, and every one costs more
            // than the face is worth:
            //
            //   * keep them always — nine boolean tests and two STEP tests, and
            //     the corpus gains four declines it did not have;
            //   * keep them where the *split* face has a hole — two tests, and
            //     it does not reach this face at all, whose rings are not holes
            //     yet at that point;
            //   * keep them here where the face has a hole — four tests and one
            //     STEP test, though it does let four `rod:` chains resolve.
            //
            // So the fix is the refinement, not a rule about which loops to
            // trust. `refine_edges` records that the walk which rebuilds a ring
            // from its edges was built, measured exact, and left out for want of
            // anything that would gain by it. This face is that thing.
            return Ok(vec![Piece {
                surface,
                // Only rings that bound something. A ring of zero area in
                // parameter space is not an outline — a cylinder's rim is a line
                // at constant `v`, and a face whose "loops" are its two rims has
                // been handed two lines and no region. The grid fill draws that
                // wall correctly; the loop fill cannot draw it at all, and a
                // tube came back holding 11.08 of the 125.66 it should.
                loops: if loops.iter().all(|l| l.area.abs() > 1e-12) {
                    loops
                } else {
                    Vec::new()
                },
                u_range: face.u_range,
                v_range: face.v_range,
                u_wraps: face.u_wraps,
                v_wraps: face.v_wraps,
                flipped: face.flipped,
                sample,
                shared_wall: None,
                bounding: Vec::new(),
                carried,
            }]);
        }

        // Each shared curve, in *this* face's parameters. Both faces of a seam
        // invert the same points, so their loops are made of the same vertices.
        let mut rings: Vec<(usize, TrimLoop)> = Vec::new();
        // (curve, points in parameter space, index of each into that curve)
        let mut chords: Vec<Chord> = Vec::new();
        for &id in curve_ids {
            let closed = shared[id].closed;
            let vs = &shared[id].vertices;
            let last = if closed { vs.len() - 1 } else { vs.len() };
            let mut uv = Vec::with_capacity(last);
            // Anchored to the middle of *this face's* range, not to nothing.
            //
            // Each point is unwrapped to sit near the one before it, which keeps
            // a curve continuous but says nothing about where the curve as a
            // whole should be: the first point is taken as `invert` gives it, in
            // the surface's own terms. A bored ball's sphere runs `u` from 3.069
            // to 9.352 and its chords came back at 0.27 to 0.73 — a whole period
            // away, outside the outline they were meant to cut, and the
            // subdivision had nothing to do.
            let mut anchor: Option<[f64; 2]> = Some([
                0.5 * (face.u_range.0 + face.u_range.1),
                0.5 * (face.v_range.0 + face.v_range.1),
            ]);
            let (pu, pv) = surface.periodic();
            let mut ok = true;
            for &vi in &vs[..last] {
                match surface.invert(vertices[vi]) {
                    Some((mut u, mut v)) => {
                        if let Some(a) = anchor {
                            if pu {
                                u = unwrap_near(u, a[0]);
                            }
                            if pv {
                                v = unwrap_near(v, a[1]);
                            }
                        }
                        anchor = Some([u, v]);
                        uv.push([u, v]);
                    }
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok || uv.len() < 2 || (closed && uv.len() < 3) {
                return Err(Declined::NeedsArrangement { face: face_index });
            }
            if closed {
                let area = ring_area(&uv);
                rings.push((
                    id,
                    TrimLoop {
                        vertices: vs[..last].to_vec(),
                        uv,
                        area,
                    },
                ));
            } else {
                let order = (0..uv.len()).collect();
                chords.push((id, uv, order));
            }
        }

        if matches!(surface, Surface::Plane { .. }) {
            return self.split_planar(
                face_index, face, &surface, loops, rings, chords, carried, shared, true, vertices,
                tolerance,
            );
        }
        // A swept face whose cuts are all at constant `v` is a stack of bands,
        // and that is both cheaper and better: the bands keep their parameter
        // rectangles, so they re-tessellate at any tolerance. Anything else —
        // a cut across the sweep, or a ring that closes inside the face — needs
        // the same subdivision the planar case uses, in `(u, v)`.
        if chords.is_empty() {
            if let Ok(bands) =
                self.split_swept(face_index, face, &surface, rings.clone(), carried.clone())
            {
                return Ok(bands);
            }
        }
        // Everything else — a cut across the sweep, or a ring closing inside
        // the face — is the same subdivision the planar case uses, run in
        // `(u, v)` over the face's own parameter rectangle.
        //
        // Where to put the seam matters. A wrapping parameter has to be cut
        // somewhere to become a rectangle, and a curve straddling that cut is
        // not inside it at all: two cylinders crossing meet in two loops, and
        // on each cylinder one of them sits right where `u` wraps. So the seam
        // goes in a gap — the widest stretch of `u` no curve occupies.
        // Where the seam went, as decided *once* before anything was split.
        // Re-deriving it here would move it — see the note at the call site.
        let chosen_origin = || {
            seam_origins
                .get(&(tag, face_index))
                .copied()
                .or_else(|| seam_origin(face, &surface, curve_ids, shared, vertices))
        };
        let mut face = face.clone();
        // A periodic `v` is cut open the same way `u` is.
        //
        // A torus wraps in both, and a curve on one can straddle the `v` seam
        // just as readily — folded relative to a point the curve happens to
        // start at, its parameters then run outside the rectangle, and the
        // containment test that decides whether a ring is a hole rejects it out
        // of hand. The seam goes in the widest stretch of `v` no curve occupies.
        // Where `v` was cut open, for the chords that cross it — see below.
        let mut v_seam: Option<(f64, f64)> = None;
        if face.v_wraps {
            let period = face.v_range.1 - face.v_range.0;
            let used: Vec<f64> = rings
                .iter()
                .flat_map(|(_, r)| r.uv.iter())
                .chain(chords.iter().flat_map(|(_, c, _)| c.iter()))
                .map(|p| p[1])
                .collect();
            if !used.is_empty() {
                let origin = widest_gap(&used, face.v_range.0, period);
                let fold = |v: f64| origin + (v - origin).rem_euclid(period);
                for (_, r) in rings.iter_mut() {
                    for p in r.uv.iter_mut() {
                        p[1] = fold(p[1]);
                    }
                    r.area = ring_area(&r.uv);
                }
                for (_, c, _) in chords.iter_mut() {
                    for p in c.iter_mut() {
                        p[1] = fold(p[1]);
                    }
                }
                face.v_range = (origin, origin + period);
                v_seam = Some((origin, period));
            }
        }

        // Only when the face says it wraps, and a boolean's output face does not
        // say so even when it does: a sphere piece covering the whole turn comes
        // out with `u_wraps = false`. Gating on the *surface* being periodic
        // instead runs the branch for those faces — and changes nothing
        // measurable, because their rings do not wrap either. Measured on a
        // ball-minus-ball then bored:
        //
        //     face 0 (sphere)   period 6.2832  rings [(99 pts, span 1.445)]
        //     face 0 (cylinder) period 6.2832  rings [(64, 6.183), (64, 6.183), (65, 2.738)]
        //
        // The wrapping rings are the bore's circles on the *cylinder*, where the
        // branch already runs. Note their margin: 6.183 against a threshold of
        // 0.98 × 6.2832 = 6.158, four parts in a thousand. The note below about
        // this measure being the wrong one is not theoretical.
        if face.u_wraps {
            let period = face.u_range.1 - face.u_range.0;
            // A ring that goes *all the way round* is not a hole. A bore right
            // through a rod meets its wall in a curve that wraps the wall
            // completely: on the parameter rectangle that is a cut from one
            // seam edge to the other, splitting the sweep in two, and treating
            // it as a hole asks the rectangle to contain something as wide as
            // itself.
            let mut wrapping: Vec<(usize, Vec<[f64; 2]>)> = Vec::new();
            rings.retain(|(id, r)| {
                // Does it go all the way round? Measured as the width its
                // parameters span, and that measure is *wrong* in a way worth
                // knowing about.
                //
                // A ring with points either side of the seam spans the rectangle
                // less the gap between its last point and its first, so one
                // sampled at 43 points came to 0.976 against this 0.98 and was
                // missed. Two better tests have been tried. Detecting the jump
                // catches those and also catches a small ring that merely
                // straddles the seam, which is not wrapping at all. Summing how
                // far `u` turns before the ring closes tells those two apart
                // exactly — a period against nothing — and both take re-cutting
                // from six of fourteen to four.
                //
                // Retried, and the measure is right: summing the turn — each
                // step taken the short way round — costs **one** test now
                // rather than the two chainings it cost when first tried,
                // `a_result_can_be_cut_again` with `NotWatertight { open_edges:
                // 7 }`. Seven, for the seven-point ring named below.
                //
                // But *why* is not what is written below. With the turning
                // measure in, `seam_chord` gets that ring and cuts it exactly:
                //
                //     seam_chord: 7 pts, origin 0.4677, first folded u 0.467688
                //                 (gap from seam 0.000000)
                //
                // The seam vertex is there and the chord starts on it. So the
                // threshold is not protecting `seam_chord` from a coarse ring —
                // that cut is exact.
                //
                // The seven open edges are one *missing face*: the result has
                // two faces and `EdgeFaceCount { edge: 0, uses: 1 }`, all seven
                // lie on the sphere in a closed loop round the drill's entry,
                // and the drill's wall is not in the result at all.
                //
                // Both measures on both rings of that case:
                //
                //     sphere   7 pts: span 0.4572  turn  0.0000  span hole, turn hole
                //     cylinder 7 pts: span 5.2788  turn -6.2832  span hole, turn WRAPS
                //
                // The wall's ring turns exactly one period — it does go right
                // round — and the span measure misses it because seven samples
                // of a full turn span less than the turn they make. So that ring
                // is treated as a hole today, which is wrong, and the operation
                // succeeds anyway. Classified correctly it takes the wrapping
                // path, and the face is then lost — but not by that path, and
                // not by the classification either. Both do their jobs:
                //
                //     subdivide (cylinder): outer 36 pts, chords [8] -> 2 pieces
                //     piece from B on cylinder: sample [1.6495, 0.6804, -20.8609]  outside
                //     piece from B on cylinder: sample [1.1501, 1.0163,  19.1391]  outside
                //
                // The drill is sixty long and the ball has radius three, so those
                // two are the wall *beyond* the ball at each end and discarding
                // them is right. The piece inside the ball was never formed:
                // a drill passing right through meets the sphere in **two**
                // rings and wants three parts from two chords. It got one.
                //
                // And the drill's second ring never exists. Counted where the
                // curves are made:
                //
                //     first  boolean, pair (0,0): 2 curves [Circle, Circle]
                //     second boolean, pair (0,0): 1 curve  [sampled 6, closed]
                //
                // The drill is parallel to the sphere's axis but offset, so that
                // pair has no closed form and is traced — and the trace returns
                // **one** ring of six points where a drill passing right through
                // a ball meets its surface in two, entry and exit. Six points
                // is also the seven-point ring seen downstream, once closed.
                //
                // So everything from `seam_chord` to the keep decision is right
                // on what it is given, and what it is given is half the
                // intersection. The span measure's misclassification happens to
                // survive that; the correct measure does not.
                //
                // And `march` is not the one giving it. Called directly on that
                // pair it returns both rings, in full:
                //
                //     2 curves: 41 pts closed, z -2.780..-2.381
                //               41 pts closed, z  2.380.. 2.780
                //
                // Two rings of forty-one points each become one ring of six by
                // the time the boolean has them. The loss and the coarsening
                // both happen between `march` returning and `shared` being
                // built — in the re-sampling and the reaching/clipping, not in
                // the tracing.
                //
                // Which is the thing to know before touching either: the measure
                // is not merely imprecise, it is wrong in a way the code relies
                // on. The wrapping path has to handle a face whose only ring
                // wraps *first*; the measure second.
                //
                // Because a ring of seven points can go right round while its
                // samples span 0.84 of the period, and this threshold has been
                // *misclassifying* it, and the wrapping path it would otherwise
                // take leaves seven edges open. The threshold is not protecting
                // the measurement, it is protecting `seam_chord` from a coarse
                // ring, and that is the thing to fix before this can be right.
                let n = r.uv.len();
                let mut turn = 0.0;
                for i in 0..n {
                    let step = r.uv[(i + 1) % n][0] - r.uv[i][0];
                    turn += step - period * (step / period).round();
                }
                if turn.abs() < period * 0.5 {
                    return true;
                }
                wrapping.push((*id, r.uv.clone()));
                false
            });
            if !wrapping.is_empty() {
                // The seam has to fall *on* these curves, since they leave no
                // gap. It was chosen — and put onto this face's rims — before
                // anything was split, so both this face and the one beyond each
                // rim carry the point.
                let Some(origin) = chosen_origin() else {
                    return Err(Declined::NeedsArrangement { face: face_index });
                };
                let cut_open = chords.len();
                for (id, uv) in wrapping {
                    let Some((chord, order)) = seam_chord(&uv, origin, period) else {
                        return Err(Declined::NeedsArrangement { face: face_index });
                    };
                    chords.push((id, chord, order));
                }
                for (_, r) in rings.iter_mut() {
                    for p in r.uv.iter_mut() {
                        p[0] = origin + (p[0] - origin).rem_euclid(period);
                    }
                    r.area = ring_area(&r.uv);
                }
                for (_, c, _) in chords.iter_mut() {
                    for p in c.iter_mut() {
                        let folded = origin + (p[0] - origin).rem_euclid(period);
                        // Keep an endpoint that landed exactly on the far edge
                        // there rather than folding it back to the near one.
                        p[0] = if (p[0] - (origin + period)).abs() <= period * 1e-9 {
                            origin + period
                        } else {
                            folded
                        };
                    }
                }
                // A chord cut from a wrapping ring crosses the rectangle, so one
                // end belongs on each edge — and *which* end is on which is a
                // question about the way the ring runs, not about where its
                // parameters happened to land. Deciding it by proximity puts
                // both ends of a ring wound the other way on the same edge:
                //
                //     path 0: starts [2.1247 27.2199] ends [8.4079 27.2199]
                //     path 1: starts [8.4079 32.7801] ends [8.4079 32.7801]
                //
                // The second is a cut that begins and ends at one point, which
                // divides nothing, and `subdivide` rightly refuses the face. The
                // two rings are a drill's entry and exit and they run opposite
                // ways round it, as entry and exit must.
                //
                // So take the direction from the ring itself: sum its steps the
                // short way round, and a chord that turns forward starts at the
                // near edge and ends at the far one.
                for (_, c, order) in chords[cut_open..].iter_mut() {
                    let turn: f64 = c
                        .windows(2)
                        .map(|w| {
                            let d = w[1][0] - w[0][0];
                            d - period * (d / period).round()
                        })
                        .sum();
                    // Turned back to front, not merely labelled back to front:
                    // pinning the ends of a path whose middle still runs the
                    // other way makes it leave the near edge, cross to the far
                    // one, come back, and finish where it started.
                    if turn < 0.0 {
                        c.reverse();
                        order.reverse();
                    }
                    let last = c.len() - 1;
                    c[0][0] = origin;
                    c[last][0] = origin + period;
                }
                face.u_range = (origin, origin + period);
                // A chord that runs across the `v` seam has to be cut there too.
                //
                // Folding `v` point by point leaves the same whole-period jump the `u`
                // fold used to: on a torus that jump draws a line right round the tube,
                // and the face on the other side of it has no such edge. An off-axis
                // cylinder through a torus came back with a hundred and thirty-eight
                // open edges lying along one circle of constant tube angle, which is
                // exactly where `v` had been cut.
                //
                // This runs last because the wrapping-ring branch above turns rings
                // into chords, and those need the treatment as much as the rest.
                if let Some((origin, period)) = v_seam {
                    let mut cut: Vec<Chord> = Vec::new();
                    for (id, c, order) in chords.drain(..) {
                        cut.extend(
                            split_at_seam(&c, &order, origin, period, 1)
                                .into_iter()
                                .map(|(uv, order)| (id, uv, order)),
                        );
                    }
                    chords = cut;
                }
                // Where the chords meet the seam edges, so the outline can
                // name those places. Both edges are the same line on the
                // surface, so a chord crossing there gives the same `v` to each.
                let known: &[(f64, usize)] = seam_cuts
                    .get(&(tag, face_index))
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let plain = parameter_outline(&face, &surface, tolerance, &carried, known);
                let split = self.split_planar(
                    face_index,
                    &face,
                    &surface,
                    vec![plain],
                    rings.clone(),
                    chords.clone(),
                    carried.clone(),
                    shared,
                    false,
                    vertices,
                    tolerance,
                );
                if split.is_ok() {
                    return split;
                }
                // Only where the face was refused anyway.
                //
                // An outline broken at the places its chords meet the seam is
                // what the walk needs to attach them — but those points have no
                // edge behind them, and a seam is held by edges. Given to every
                // face, it opens what it fixes: 48 edges on a bore across a rod
                // and eleven of twenty-two ratios down to three, all of them
                // declines rather than wrong answers, but lost all the same.
                //
                // So it is a second attempt and not a first. A face that splits
                // without the extra points keeps exactly the split it had, and
                // one that does not was going to be declined regardless — this
                // can add a result and cannot take one away. The proper repair
                // is a vertex at each of those points, which is what would let
                // the first attempt carry them.
                let seam_cuts: Vec<(f64, usize)> = chords[cut_open..]
                    .iter()
                    .flat_map(|(_, c, _)| [c[0][1], c[c.len() - 1][1]])
                    .map(|v| (v, usize::MAX))
                    .collect();
                if seam_cuts.is_empty() {
                    return split;
                }
                let broken = parameter_outline(&face, &surface, tolerance, &carried, &seam_cuts);
                return self
                    .split_planar(
                        face_index,
                        &face,
                        &surface,
                        vec![broken],
                        rings,
                        chords,
                        carried,
                        shared,
                        false,
                        vertices,
                        tolerance,
                    )
                    .or(split);
            }
            // Nothing wanting a seam is an answer, not a refusal.
            //
            // `seam_origin` declines when every curve on the face lies at
            // constant `v` — a stack of bands, which needs no seam. That is
            // usually `split_swept`'s case, and it was: rings at constant `v`
            // leave no chords and go there. But an *arc* at constant `v` is a
            // chord, so the face comes here instead, and a missing origin was
            // read as a face that could not be cut open rather than one that did
            // not need to be. Two cubes stacked flush and re-drilled through the
            // step declined for that, on the drill's own wall, cut by four arcs
            // and every one of them level.
            //
            // Where it is already cut is where it stays cut. What must not be
            // skipped is the *fold*: `invert` answers in the surface's own
            // terms, and a face whose range runs -π to π was being handed chords
            // between 0 and 2π, outside the outline they were meant to divide.
            let origin = chosen_origin().unwrap_or(face.u_range.0);
            let fold = |u: f64| origin + (u - origin).rem_euclid(period);
            for (_, r) in rings.iter_mut() {
                for p in r.uv.iter_mut() {
                    p[0] = fold(p[0]);
                }
                r.area = ring_area(&r.uv);
            }
            let mut cut: Vec<Chord> = Vec::new();
            for (id, c, order) in chords.drain(..) {
                let folded: Vec<[f64; 2]> = c.iter().map(|p| [fold(p[0]), p[1]]).collect();
                cut.extend(
                    split_at_seam(&folded, &order, origin, period, 0)
                        .into_iter()
                        .map(|(uv, order)| (id, uv, order)),
                );
            }
            chords = cut;
            face.u_range = (origin, origin + period);
        }
        // A chord that runs across the `v` seam has to be cut there too.
        //
        // Folding `v` point by point leaves the same whole-period jump the `u`
        // fold used to: on a torus that jump draws a line right round the tube,
        // and the face on the other side of it has no such edge. An off-axis
        // cylinder through a torus came back with a hundred and thirty-eight
        // open edges lying along one circle of constant tube angle, which is
        // exactly where `v` had been cut.
        //
        // This runs last because the wrapping-ring branch above turns rings
        // into chords, and those need the treatment as much as the rest.
        if let Some((origin, period)) = v_seam {
            let mut cut: Vec<Chord> = Vec::new();
            for (id, c, order) in chords.drain(..) {
                cut.extend(
                    split_at_seam(&c, &order, origin, period, 1)
                        .into_iter()
                        .map(|(uv, order)| (id, uv, order)),
                );
            }
            chords = cut;
        }
        let outline = parameter_outline(&face, &surface, tolerance, &carried, &[]);
        self.split_planar(
            face_index,
            &face,
            &surface,
            vec![outline],
            rings,
            chords,
            carried,
            shared,
            false,
            vertices,
            tolerance,
        )
    }

    /// A planar face cut by curves that *cross* it rather than close on it.
    ///
    /// The pieces come from [`crate::brep::planar`], and the work here is
    /// getting them back to vertex indices. A point that came from a chord takes
    /// the shared curve's index, so both faces of the seam name the same vertex
    /// and the result closes; a point from the boundary or from a crossing has
    /// no index yet and is welded by position later.
    #[allow(clippy::too_many_arguments)]
    fn subdivide_face(
        &self,
        face_index: usize,
        face: &Face,
        surface: &Surface,
        outer: &TrimLoop,
        chords: &[Chord],
        // Boundaries the face already had, with no curve behind them.
        already: &[Vec<[f64; 2]>],
        shared: &[SharedCurve],
        carried: &[Carried],
        vertices: &[V3],
        tolerance: f64,
    ) -> Result<Vec<Piece>, Declined> {
        // A few multiples of the tolerance: the error is the sagitta of one
        // sample step, a little over what the curve was sampled to hold.
        let reach = tolerance * 4.0;
        let near_curve = |uv: [f64; 2]| -> Option<usize> {
            let p = surface.point(uv[0], uv[1]);
            let mut best: Option<(f64, usize)> = None;
            for (id, _, _) in chords {
                for &vi in &shared[*id].vertices {
                    let Some(q) = vertices.get(vi) else { continue };
                    let d = v3::dist(p, *q);
                    if d <= reach && best.is_none_or(|(bd, _)| d < bd) {
                        best = Some((d, vi));
                    }
                }
            }
            best.map(|(_, vi)| vi)
        };
        let mut paths: Vec<Vec<[f64; 2]>> = chords.iter().map(|(_, uv, _)| uv.clone()).collect();
        paths.extend(already.iter().cloned());
        // Measured where a bore refuses to cut a sphere piece: the paths handed
        // over are 65 fragments of one curve, two points each, with gaps —
        //
        //     [3.544, 0.000]..[3.544, 0.024]   [3.545, 0.042]..[3.545, 0.090]
        //     [3.546, 0.108]..[3.547, 0.155]   [3.547, 0.174]..[3.549, 0.221]
        //
        // — so `subdivide` returning `None` is the honest answer to what it was
        // given. The fragments are *not* the intersection failing: that pair is
        // traced, and it traces cleanly to one curve of 65 points. Counted at
        // the source,
        //
        //     closed-form pair (0,0): 2 curves      traced pair (1,0): 1 of 65
        //
        // and yet 64 shared curves of three points exist by the time the face is
        // split. Whatever breaks the curve up happens after the surfaces are
        // intersected and before the arrangement is called.
        // A face whose only cuts are chords across the seam still refuses here,
        // and the reason is in the outline rather than the chords. Measured on a
        // drill's wall cut by a ball it passes through:
        //
        //     outer 35 pts u [2.1247 8.4079] v [0 60]
        //     outer points on the near edge: v = ["60.0000"]
        //     path 0: [2.1247 27.2199] -> [8.4079 27.2199]
        //     path 1: [2.1247 32.7801] -> [8.4079 32.7801]
        //
        // Both chords are sound — they cross the rectangle, stay in their bands
        // and take no step over 0.15 — and they meet the near edge at 27.22 and
        // 32.78, where the outline has no vertex to meet. The walk has nothing
        // to attach them to. What wants fixing is `parameter_outline`, which
        // must carry the points where a chord lands on a seam edge.
        let regions = planar::subdivide(&outer.uv, &paths)
            .ok_or(Declined::NeedsArrangement { face: face_index })?;
        // One region *with a hole* is a cut: a closed curve inside a face
        // leaves it connected and takes a piece out of the middle. One region
        // with no holes means nothing happened, and leaving the face whole would
        // drop the seam the other side is expecting.
        if regions.len() < 2 && regions.iter().all(|r| r.holes.is_empty()) {
            return Err(Declined::NeedsArrangement { face: face_index });
        }

        let mut out = Vec::with_capacity(regions.len());
        for region in regions {
            let vertices: Vec<usize> = region
                .outer
                .sources
                .iter()
                .map(|s| match s {
                    // Through the chord's own ordering: a chord re-cut at the
                    // seam is sorted by parameter, so its position is no longer
                    // its position along the curve, and taking the vertex at
                    // that index would name a different point entirely.
                    planar::Source::Chord { chord, point } => chords
                        .get(*chord)
                        .and_then(|c| c.2.get(*point).map(|k| (c.0, *k)))
                        .and_then(|(id, k)| shared[id].vertices.get(k))
                        .copied()
                        .unwrap_or(usize::MAX),
                    // A point of the face's own boundary, which now says which
                    // vertex it is. It used to be welded by position in
                    // `from_pieces` — and a rim shared by two faces was then two
                    // sets of points that had to round the same way to come back
                    // together. They are the same vertex; this says so.
                    planar::Source::Boundary(i) => {
                        outer.vertices.get(*i).copied().unwrap_or(usize::MAX)
                    }
                    // A crossing belongs to no input, so it has no name yet.
                    planar::Source::Crossing => usize::MAX,
                })
                .collect();
            // A crossing the subdivision worked out for itself, put back onto
            // the curve it belongs to.
            //
            // Where a chord meets the boundary, the subdivision intersects the
            // chord's *polyline* with the boundary's, and that crossing lands a
            // sagitta from the curve — 1.3e-3 at a tolerance of 1e-3, which the
            // weld will not close. The curve already has a vertex there.
            //
            // Only a *crossing*: a boundary point is the face's own and belongs
            // to the rim the face beyond it walks.
            let vertices: Vec<usize> = vertices
                .into_iter()
                .zip(region.outer.sources.iter().zip(&region.outer.uv))
                .map(|(v, (src, uv))| {
                    if v == usize::MAX && matches!(src, planar::Source::Crossing) {
                        near_curve(*uv).unwrap_or(usize::MAX)
                    } else {
                        v
                    }
                })
                .collect();
            // Which curves this piece's boundary runs along, from the sources
            // the subdivision recorded for every point of it.
            //
            // Measured, a cylinder gets none of them. Cutting a bored ball with
            // a column splits the bore's wall into nine regions, and every one
            // reports its whole outline as `Boundary` — chord=0 of 8 chords
            // passed — while the column's own plane faces record theirs
            // normally. So `along` comes out empty, no `Edge` is made for the
            // plane-against-cylinder curves, and the wall pieces end up bounded
            // by rim arcs alone with four free ends each. That is the whole
            // reason a bore cannot be cut a second time: not the arrangement,
            // which finds the pieces, but the provenance it hands back for them.
            let mut along: Vec<usize> = region
                .outer
                .sources
                .iter()
                .chain(region.holes.iter().flat_map(|h| h.sources.iter()))
                .filter_map(|src| match src {
                    planar::Source::Chord { chord, .. } => chords.get(*chord).map(|c| c.0),
                    _ => None,
                })
                .collect();
            along.sort_unstable();
            along.dedup();
            let ring = TrimLoop {
                area: region.area.abs(),
                uv: region.outer.uv,
                vertices,
            };
            let mut loops = vec![ring];
            for hole in region.holes {
                let vertices = hole
                    .sources
                    .iter()
                    .map(|s| match s {
                        planar::Source::Chord { chord, point } => chords
                            .get(*chord)
                            .and_then(|c| c.2.get(*point).map(|k| (c.0, *k)))
                            .and_then(|(id, k)| shared[id].vertices.get(k))
                            .copied()
                            .unwrap_or(usize::MAX),
                        planar::Source::Boundary(i) => {
                            outer.vertices.get(*i).copied().unwrap_or(usize::MAX)
                        }
                        planar::Source::Crossing => usize::MAX,
                    })
                    .collect();
                let area = -ring_area(&hole.uv).abs();
                loops.push(TrimLoop {
                    area,
                    uv: hole.uv,
                    vertices,
                });
            }
            // A piece whose outer ring goes once around the surface is not a
            // piece. Two rims cut a bore's wall into three bands, and each band
            // is bounded by going out along one rim and back along the other —
            // net travel zero. A ring that instead totals a whole period closed
            // itself at the seam after one turn, so what it bounds is the rim's
            // own wiggle rather than a band, and the band it should have made is
            // simply absent.
            //
            // That is what the erratic ratios are. A rod bored across by a tool
            // 0.70 of its radius gives, in place of three bands:
            //
            //     v [0.00 7.00]  v [6.71 12.00]  v [5.00 5.29] wraps [true, _]
            //
            // where the third is exactly the rim's own `v` extent and the middle
            // band is gone. Measured at 0.70 and 0.85 it fires on precisely the
            // sliver and on none of the ten sound pieces beside it.
            //
            // It ends today as `NotWatertight` with 146 to 163 open edges, which
            // is a true statement about the result and says nothing about the
            // cause. Named here instead: the face needs an arrangement the walk
            // did not give it.
            // Only where the surface has no point at which it closes. A
            // spherical cap is bounded by *one* circle that goes around once —
            // the other end of it is the pole, which is a point and travels
            // nowhere — and a cone's apex does the same. Applied to those, this
            // rejects four sound results. A cylinder closes nowhere, so every
            // band on one is bounded at both ends and a single turn is always a
            // ring that stopped early.
            if surface.kind() == "cylinder" && loops[0].wraps(surface.period()).iter().any(|&w| w) {
                return Err(Declined::NeedsArrangement { face: face_index });
            }
            let Some(sample) = self.sample_between(surface, &loops) else {
                return Err(Declined::UnclassifiablePiece { face: face_index });
            };
            // Only the rims this piece actually runs along.
            //
            // A face keeps every edge it had, and its *pieces* were keeping them
            // too — so the drill's end rims, five units outside the rod it was
            // cutting, were carried by the sliver of wall that survived inside.
            // Each became an edge one face claimed and none other could, because
            // there is nothing on the far side of a rim the result does not
            // reach, and `is_valid_solid` reads exactly that.
            let touching: Vec<Carried> = carried
                .iter()
                .filter(|(_, _, pts, _, _)| {
                    // Any point, not every point: a rim the piece runs along
                    // only part of is still a rim it runs along, and dropping it
                    // takes the connection between two faces with it — a shell
                    // then reads as open. Only a rim the piece does not reach at
                    // all is dropped.
                    //
                    // Trimming to the part actually run along is better in one
                    // way and worse in another: it takes a column bored through
                    // a ball from four defects to none, by giving the upper and
                    // lower piece of a side face a rim each instead of both
                    // claiming the whole one — and it costs a corner overlap's
                    // union its shell closure. 2 defects and 4 valid against 6
                    // and 5. Neither is better, so this stays until the corner
                    // case is understood.
                    pts.iter().any(|p| {
                        loops.iter().any(|l| {
                            l.uv.iter()
                                .any(|q| v3::dist(surface.point(q[0], q[1]), *p) <= tolerance * 4.0)
                        })
                    })
                })
                .cloned()
                .collect();
            out.push(Piece {
                surface: surface.clone(),
                loops,
                u_range: face.u_range,
                v_range: face.v_range,
                u_wraps: false,
                v_wraps: false,
                flipped: face.flipped,
                sample,
                shared_wall: None,
                // The curves this piece's boundary runs along, from the
                // sources the subdivision recorded for each of its points.
                //
                // These used to be left empty, on the reasoning that the seam is
                // held by the loops' shared vertex indices rather than by an
                // `Edge` — which is true of the *tessellation* and false of the
                // topology. A piece that names no edge leaves the curve it runs
                // along claimed by one face instead of two, `is_valid_solid`
                // reads that, and the result cannot be used as the input to
                // another boolean: a rod with a cross-hole could not be touched
                // again.
                //
                // Still not the whole answer. One edge per curve is too coarse
                // when a chord is cut at a crossing and each piece runs along
                // only part of it; the edges want splitting per *stretch*, which
                // the same sources describe and which is the next work.
                bounding: along,
                carried: touching,
            });
        }
        Ok(out)
    }

    /// A planar face cut by rings that close inside it: the outside keeps them
    /// as holes, and each inside becomes a face of its own.
    #[allow(clippy::too_many_arguments)]
    fn split_planar(
        &self,
        face_index: usize,
        face: &Face,
        surface: &Surface,
        loops: Vec<TrimLoop>,
        rings: Vec<(usize, TrimLoop)>,
        chords: Vec<Chord>,
        carried: Vec<Carried>,
        shared: &[SharedCurve],
        outline_from_edges: bool,
        vertices: &[V3],
        tolerance: f64,
    ) -> Result<Vec<Piece>, Declined> {
        let Some(outer) = loops.iter().find(|l| l.is_outer()) else {
            return Err(Declined::NeedsArrangement { face: face_index });
        };

        if !chords.is_empty() {
            // A hole *and* a cut on one face: the rings go in as closed paths
            // alongside the chords, and the subdivision hands back whichever
            // came out enclosed rather than separating.
            let mut paths = chords;
            for (id, r) in rings {
                let mut uv = r.uv.clone();
                uv.push(uv[0]);
                let mut order: Vec<usize> = (0..r.uv.len()).collect();
                order.push(0);
                paths.push((id, uv, order));
            }
            // And the holes the face already had.
            //
            // A face cut once and cut again has holes of its own, and they were
            // not going into the arrangement at all — so a curve ending on one
            // dangled, and the subdivision refused. Two cubes stacked flush,
            // re-drilled through the step, declined for exactly that: both
            // chords ran from the square hole's edge and neither reached
            // anything the arrangement knew about.
            //
            // They carry no curve, so they go in as bare paths: the points come
            // back with no index and are welded by position, like any other
            // point of a face's own boundary.
            let already: Vec<Vec<[f64; 2]>> = loops
                .iter()
                .filter(|l| l.is_hole())
                .map(|l| {
                    let mut uv = l.uv.clone();
                    uv.push(uv[0]);
                    uv
                })
                .collect();
            return self.subdivide_face(
                face_index, face, surface, outer, &paths, &already, shared, &carried, vertices,
                tolerance,
            );
        }
        // A ring that does not fit inside the face is not a hole in it.
        //
        // Measured where a bore fails to cut the second sphere of a difference:
        // that face's split is handed the ring — one ring of 65 points, no
        // chords — and the face comes out with its 128-point outline and no hole
        // at all, so the curve is lost between here and the pieces. Its `outer`
        // is that outline: the parameter rectangle, which runs down the seam and
        // back and collapses a row of points at each pole. Asking
        // `point_in_ring` about a polygon shaped like that is the first thing to
        // check.
        //
        // A sphere meeting a box's wall gives a circle larger than the wall, and
        // only the arc within it is a seam — the rest is off the face entirely.
        // Clipped, each surviving arc enters and leaves through the boundary,
        // which is a chord, which the subdivision already handles.
        let mut rings = rings;
        let mut chords = chords;
        // Folded into *this face's* range before any containment question. A
        // face cut open at a seam runs its `u` from wherever that seam fell, and
        // a ring straddling that seam carries parameters either side of it: the
        // bore's ring on the second sphere of a difference encircles that
        // sphere's own seam, and 33 of its 65 points read as outside an outline
        // covering the whole rectangle. Half a ring outside is not a ring
        // leaving the face; it is a ring measured against the wrong turn.
        let (pu_fold, pv_fold) = surface.periodic();
        let fold = |p: [f64; 2]| -> [f64; 2] {
            let mut q = p;
            if pu_fold {
                q[0] = face.u_range.0 + (q[0] - face.u_range.0).rem_euclid(std::f64::consts::TAU);
            }
            if pv_fold {
                q[1] = face.v_range.0 + (q[1] - face.v_range.0).rem_euclid(std::f64::consts::TAU);
            }
            q
        };
        {
            let mut clipped: Vec<Chord> = Vec::new();
            // Measured: a bore's ring on the second sphere of a difference has
            // **33 of its 65 points** judged inside this outline, and the
            // outline covers the whole parameter rectangle. The ring encircles
            // that sphere's own seam, so half its parameters sit on the far side
            // of the face's `u_range` and are read as outside. Half a ring
            // outside is not a ring leaving the face; it is a ring measured
            // against the wrong turn.
            //
            // Folding into the face's range before asking — the same correction
            // `curve_reaches_face` and `clip_to_face` already carry — does fix
            // the count, and puts the operation *back* to `NeedsArrangement`
            // from the `NotWatertight` it had reached.
            //
            // And it does not get as far as the arrangement to say so: with the
            // fold on, this face never reaches `subdivide_face` at all. It is
            // turned away by one of the earlier guards, which means keeping the
            // ring whole is not two halves of one change but the first of at
            // least three. `a_curve_is_clipped_by_where_it_runs_not_by_where_
            // its_chords_sag` catches taking only the first.
            rings.retain(|(id, r)| {
                if r.uv.iter().all(|p| point_in_ring(&outer.uv, fold(*p))) {
                    return true;
                }
                let mut closed = r.uv.clone();
                closed.push(closed[0]);
                for run in planar::clip_to_region(&outer.uv, &closed) {
                    if run.len() < 2 {
                        continue;
                    }
                    // The points that came from the ring keep their place along
                    // it; the two ends are crossings this face worked out, and
                    // weld by position — the face on the other side of the curve
                    // computes the same ones from the same two boundaries.
                    let order: Vec<usize> = run
                        .iter()
                        .map(|p| {
                            r.uv.iter()
                                .position(|q| (q[0] - p[0]).hypot(q[1] - p[1]) <= 1e-12)
                                .unwrap_or(usize::MAX)
                        })
                        .collect();
                    clipped.push((*id, run, order));
                }
                false
            });
            chords.extend(clipped);
        }

        // Strictly inside, or the split crosses a boundary and needs a genuine
        // planar subdivision. The same question the retain above just asked, so
        // it has to be asked the same way — folded. Answering it differently
        // here kept the ring and then threw the face out for having it.
        for (_, r) in &rings {
            if !r.uv.iter().all(|p| point_in_ring(&outer.uv, fold(*p))) {
                return Err(Declined::NeedsArrangement { face: face_index });
            }
        }

        // Now that the containment questions are settled, put the rings on the
        // same turn as the face itself. What follows splices them into the outer
        // ring and fills the result, and a hole with half its parameters a
        // period away from the other half is not a hole any fill can use.
        for (_, r) in rings.iter_mut() {
            for p in r.uv.iter_mut() {
                *p = fold(*p);
            }
            r.area = ring_area(&r.uv);
        }

        let mut out = Vec::with_capacity(rings.len() + 1);

        // The outside: the original loops plus each ring as a hole.
        let mut outside = loops.clone();
        for (_, r) in &rings {
            outside.push(oriented(r.clone(), false));
        }
        let Some(sample) = self.sample_between(surface, &outside) else {
            return Err(Declined::UnclassifiablePiece { face: face_index });
        };
        {
            // Keep the loops only when nothing else describes the boundary.
            //
            // A loop is a fixed polyline; an edge gets *refined* to tolerance.
            // Carry both and they diverge the moment the boundary is curved —
            // a cylinder's cap held a 16-point rim while the wall beside it
            // refined the same circle to 145, and the two came apart along
            // every one of the 144 segments in between. Straight boundaries hid
            // it, because refining a straight edge changes nothing.
            // Only when the outer loop *came from* those edges. A swept face's
            // outline is synthesised over its parameter rectangle, and its edges
            // are the rims alone — they do not describe it, and dropping the
            // loops in favour of them leaves the piece with no boundary at all.
            let backed = outline_from_edges && !carried.is_empty();
            out.push(Piece {
                surface: surface.clone(),
                loops: if backed { Vec::new() } else { outside },
                u_range: face.u_range,
                v_range: face.v_range,
                u_wraps: false,
                v_wraps: false,
                flipped: face.flipped,
                sample,
                shared_wall: None,
                bounding: rings.iter().map(|(id, _)| *id).collect(),
                carried: carried.clone(),
            });
        }

        // Each island, as its own face.
        for (id, r) in rings {
            let ring = oriented(r, true);
            let Some(sample) = self.sample_between(surface, std::slice::from_ref(&ring)) else {
                return Err(Declined::UnclassifiablePiece { face: face_index });
            };
            out.push(Piece {
                surface: surface.clone(),
                // A planar island is bounded by the curve it names, which
                // refines with everything else where the ring itself would not.
                //
                // On a *curved* surface the ring has to be kept: an island there
                // is filled from its trim loops, and the parameter-rectangle fill
                // that would otherwise take it wants rims at constant `v` — which
                // a traced seam is not. Without the ring such a face simply does
                // not fill, and the solid comes back with two of them missing.
                loops: if matches!(surface, Surface::Plane { .. }) {
                    Vec::new()
                } else {
                    vec![ring.clone()]
                },
                u_range: face.u_range,
                v_range: face.v_range,
                u_wraps: false,
                v_wraps: false,
                flipped: face.flipped,
                sample,
                shared_wall: None,
                bounding: vec![id],
                carried: Vec::new(),
            });
        }
        Ok(out)
    }

    /// A swept face cut at constant parameter: the rings slice its range.
    fn split_swept(
        &self,
        face_index: usize,
        face: &Face,
        surface: &Surface,
        rings: Vec<(usize, TrimLoop)>,
        carried: Vec<Carried>,
    ) -> Result<Vec<Piece>, Declined> {
        let range = (face.v_range.1 - face.v_range.0).abs().max(1e-12);
        let mut cuts: Vec<(f64, usize)> = Vec::new();
        for (id, r) in &rings {
            let vs: Vec<f64> = r.uv.iter().map(|p| p[1]).collect();
            let spread = vs.iter().cloned().fold(f64::MIN, f64::max)
                - vs.iter().cloned().fold(f64::MAX, f64::min);
            if spread > range * 1e-3 {
                return Err(Declined::NeedsArrangement { face: face_index });
            }
            let v = vs.iter().sum::<f64>() / vs.len() as f64;
            if v > face.v_range.0 + range * 1e-6 && v < face.v_range.1 - range * 1e-6 {
                cuts.push((v, *id));
            }
        }
        cuts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // A rim the face already carries can cross the bands instead of ending
        // one, and then the bands are not what this face is — the sphere-sphere
        // circle on a sphere that is then bored spans `v` and lies wholly inside
        // the range, the bands claim the whole rectangle anyway, and 124 of that
        // result's 128 open edges are along it.
        //
        // Declining here sends the face to the arrangement instead, which can
        // state such a region. Tried, and it gets there: the arrangement runs and
        // the operation fails later at `UnclassifiablePiece { face: 0 }`, unable
        // to find a point that says which side a piece is on. So the region is
        // no longer wrong — it is unclassifiable, which is a different and
        // earlier failure, and nothing measurable improved. Not taken.

        // A wrapping `v` has no first or last band: the cuts divide a *circle*,
        // so the stretch from the highest back round to the lowest is one band,
        // not two with a seam down the middle. Treating it as an interval splits
        // that band in half at the parameter origin and leaves the two halves
        // with nothing shared between them — a torus intersected with a coaxial
        // cylinder came back with 282 open edges along a seam that is not there.
        let bounds: Vec<(f64, Option<usize>)> = if face.v_wraps && cuts.len() >= 2 {
            let period = face.v_range.1 - face.v_range.0;
            let mut round: Vec<(f64, Option<usize>)> =
                cuts.iter().map(|&(v, id)| (v, Some(id))).collect();
            let (first_v, first_id) = (round[0].0, round[0].1);
            round.push((first_v + period, first_id));
            round
        } else {
            let mut linear: Vec<(f64, Option<usize>)> = vec![(face.v_range.0, None)];
            linear.extend(cuts.iter().map(|&(v, id)| (v, Some(id))));
            linear.push((face.v_range.1, None));
            linear
        };

        let mut out = Vec::with_capacity(bounds.len() - 1);
        for w in bounds.windows(2) {
            let mid = 0.5 * (w[0].0 + w[1].0);
            let u_mid = 0.5 * (face.u_range.0 + face.u_range.1);
            // A band's outer rim is not a seam — it is where the face already
            // ended, so it keeps the edge it already had. Without this the
            // cylinder's end rim and its cap each sample their own circle and
            // the two only coincide: a union came back with 232 open edges
            // along a rim both sides agreed the position of.
            let at_end =
                |v: f64| (v - w[0].0).abs() < range * 1e-6 || (v - w[1].0).abs() < range * 1e-6;
            // Only rims that lie *at* a band's end, which is a rim at constant
            // `v`. Measured, that is not every rim a face carries: a sphere
            // already cut by another sphere carries a circle at constant `x`,
            // which crosses the bands rather than ending one, and every band
            // drops it. The result is a body of four edges where the
            // sphere-sphere circle is claimed by one face and not the other —
            // three faces, no free ends anywhere, and 195 open edges in the
            // tessellation because a whole rim has only one side.
            let mine: Vec<_> = carried
                .iter()
                .filter(|(_, _, pts, _, _)| {
                    let vs: Vec<f64> = pts
                        .iter()
                        .filter_map(|p| surface.invert(*p))
                        .map(|(_, v)| v)
                        .collect();
                    if vs.is_empty() {
                        return false;
                    }
                    // Ends *or* crosses. A rim at constant `v` ends a band and
                    // its mean lands on the boundary; a rim that came from an
                    // earlier cut need not be at constant `v` at all, and the
                    // mean of one crossing this band can fall outside it
                    // entirely. Asking whether any of it is in range keeps both,
                    // and keeps the at-end rims for exactly the band they end,
                    // since a face's own rims sit at the outer edges of its
                    // range and no other band reaches them.
                    let pad = range * 1e-6;
                    at_end(vs.iter().sum::<f64>() / vs.len() as f64)
                        || vs.iter().any(|v| *v >= w[0].0 - pad && *v <= w[1].0 + pad)
                })
                .cloned()
                .collect();
            // No loops: a band is the rectangle between two cuts, and its
            // parameter range says so exactly.
            //
            // Which is only true when nothing *else* bounds it. Measured on a
            // sphere already cut by another sphere and then bored: the surviving
            // band carries the sphere-sphere rim across it, the band still says
            // it is the whole rectangle, and the fill covers ground the face does
            // not have. 124 of the 128 open edges in that result are on this
            // face, along the rim the band never mentions.
            out.push(Piece {
                surface: surface.clone(),
                loops: Vec::new(),
                u_range: face.u_range,
                v_range: (w[0].0, w[1].0),
                u_wraps: face.u_wraps,
                v_wraps: false,
                flipped: face.flipped,
                sample: surface.point(u_mid, mid),
                shared_wall: None,
                bounding: [w[0].1, w[1].1].into_iter().flatten().collect(),
                carried: mine,
            });
        }
        Ok(out)
    }

    /// A point on a face, well inside its trim region.
    fn face_sample(&self, face_index: usize, surface: &Surface, loops: &[TrimLoop]) -> Option<V3> {
        let face = &self.faces()[face_index];
        // A swept face's boundary loops are its rims, and a rim is a straight
        // line in parameter space — it encloses nothing, so ring containment
        // rejects every candidate and the face reads as unclassifiable. Such a
        // face is bounded by its parameter range instead, which describes it
        // completely.
        let span = ((face.u_range.1 - face.u_range.0) * (face.v_range.1 - face.v_range.0)).abs();
        let enclosing = loops
            .iter()
            .any(|l| l.area.abs() > span * 1e-9 && l.uv.len() >= 3);
        if !enclosing {
            return Some(surface.point(
                0.5 * (face.u_range.0 + face.u_range.1),
                0.5 * (face.v_range.0 + face.v_range.1),
            ));
        }
        self.sample_between(surface, loops)
    }

    /// A point inside the outer loop and outside every hole.
    fn sample_between(&self, surface: &Surface, loops: &[TrimLoop]) -> Option<V3> {
        let outer = loops.iter().find(|l| l.is_outer())?;
        // Asked of the rings, so the candidate is brought to the branch each
        // ring lives on first. A cylinder wall's rings can sit an unwrapped turn
        // from where a raw comparison puts a point, and then a candidate outside
        // the piece reads as inside it — which decides, by one point, whether
        // the piece is kept at all.
        let period = surface.period();
        let usable = |c: [f64; 2]| -> bool {
            outer.contains(c, period)
                && !loops
                    .iter()
                    .filter(|l| l.is_hole())
                    .any(|h| h.contains(c, period))
        };

        // The *furthest* candidate from the boundary, in space, not the first
        // one that is inside.
        //
        // This point is what decides which side of the other solid the piece
        // falls on, by a ray cast against a tessellation of it — so how far it
        // is from its own boundary is how much that tessellation is allowed to
        // deviate. Taking the first candidate that landed inside gave points a
        // few thousandths from the edge, and a classification mesh coarser than
        // that put the piece on the wrong side: a plate's first bore came back
        // with an edge carrying three faces.
        //
        // Depth in *parameters* is not distance. Near a pole, near a seam, on a
        // cone, the parameterisation bunches and a point deep by that measure is
        // a hair from the boundary in space. So the margin is measured where it
        // matters.
        let edge: Vec<V3> = {
            let total: usize = loops.iter().map(|l| l.uv.len()).sum();
            let stride = total.div_ceil(256).max(1);
            loops
                .iter()
                .flat_map(|l| l.uv.iter())
                .step_by(stride)
                .map(|q| surface.point(q[0], q[1]))
                .collect()
        };
        let margin = |c: [f64; 2]| -> f64 {
            let p = surface.point(c[0], c[1]);
            edge.iter()
                .map(|b| v3::dist(p, *b))
                .fold(f64::MAX, f64::min)
        };
        // Far enough that looking harder is not worth it: a tenth of the piece.
        let target = {
            let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
            for p in &edge {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            (0..3).map(|k| hi[k] - lo[k]).fold(0.0f64, f64::max) * 0.1
        };
        let mut best: Option<(f64, [f64; 2])> = None;
        let consider = |c: [f64; 2], best: &mut Option<(f64, [f64; 2])>| -> bool {
            if !usable(c) {
                return false;
            }
            let m = margin(c);
            if best.is_none_or(|(b, _)| m > b) {
                *best = Some((m, c));
            }
            m >= target
        };

        // Triangle centroids of the outer ring, which land inside a convex
        // region and usually inside a concave one.
        for i in 1..outer.uv.len().saturating_sub(1) {
            let c = [
                (outer.uv[0][0] + outer.uv[i][0] + outer.uv[i + 1][0]) / 3.0,
                (outer.uv[0][1] + outer.uv[i][1] + outer.uv[i + 1][1]) / 3.0,
            ];
            if consider(c, &mut best) {
                return best.map(|(_, c)| surface.point(c[0], c[1]));
            }
        }

        // A fan from one corner of a quad gives two candidates, both near the
        // middle — so a hole in the middle swallows every one of them and the
        // face reads as unclassifiable. It is not: the material is the *ring*
        // around the hole. Look there too, by walking in from each vertex, and
        // then over a grid, before giving up.
        let centroid = {
            let n = outer.uv.len().max(1) as f64;
            [
                outer.uv.iter().map(|p| p[0]).sum::<f64>() / n,
                outer.uv.iter().map(|p| p[1]).sum::<f64>() / n,
            ]
        };
        for t in [0.05, 0.15, 0.3, 0.5] {
            for v in &outer.uv {
                let c = [
                    v[0] + (centroid[0] - v[0]) * t,
                    v[1] + (centroid[1] - v[1]) * t,
                ];
                if consider(c, &mut best) {
                    return best.map(|(_, c)| surface.point(c[0], c[1]));
                }
            }
        }

        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for p in &outer.uv {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        const N: usize = 24;
        for i in 0..N {
            for j in 0..N {
                let c = [
                    lo[0] + (hi[0] - lo[0]) * (i as f64 + 0.5) / N as f64,
                    lo[1] + (hi[1] - lo[1]) * (j as f64 + 0.5) / N as f64,
                ];
                if consider(c, &mut best) {
                    return best.map(|(_, c)| surface.point(c[0], c[1]));
                }
            }
        }
        best.map(|(_, c)| surface.point(c[0], c[1]))
    }

    /// This body as a triangle list, for in/out tests only.
    fn solid_triangles(&self, tolerance: f64) -> Vec<[V3; 3]> {
        // `refine_edges` is not optional here. Without it the edges stay at the
        // resolution the body was built at while the fills go to tolerance, the
        // two disagree, and the mesh is not closed — nine tests' worth.
        let mut body = self.clone();
        body.refine_edges(tolerance);
        let (mesh, report) = body.tessellate(tolerance);
        if !report.is_closed() {
            return Vec::new();
        }
        let Some(pos) = mesh.get_attribute("position") else {
            return Vec::new();
        };
        let Some(idx) = &mesh.index else {
            return Vec::new();
        };
        let v = |i: u32| -> V3 {
            let o = i as usize * 3;
            [
                pos.array[o] as f64,
                pos.array[o + 1] as f64,
                pos.array[o + 2] as f64,
            ]
        };
        idx.chunks_exact(3)
            .map(|t| [v(t[0]), v(t[1]), v(t[2])])
            .collect()
    }

    /// Assemble surviving pieces into a body.
    ///
    /// The pieces already carry their surfaces and trim loops, so this is
    /// bookkeeping — but note what it does *not* do: it does not rebuild the
    /// shared edges between them. The result tessellates from its faces' loops,
    /// which are exact, and its faces meet along curves both sides derived from
    /// the same `ssi` — but they are not yet the same `Edge`, so the result is
    /// not guaranteed watertight the way an authored body is.
    fn from_pieces(
        mut pieces: Vec<Piece>,
        mut vertices: Vec<V3>,
        shared: Vec<SharedCurve>,
        tolerance: f64,
    ) -> Body {
        // Weld the carried-through loop corners.
        //
        // A seam is shared because both faces name the same curve; a shared
        // *corner* has no curve to name it, so each face would mint its own and
        // the two would merely coincide. That is not enough — a box came back
        // with all twelve of its edges cracked open. Weld by position into the
        // same vertex list the seams use, so adjacent faces meet at one index.
        let quantum = tolerance.max(1e-12);
        let key = |p: V3| -> (i64, i64, i64) {
            let q = |x: f64| (x / quantum).round() as i64;
            (q(p[0]), q(p[1]), q(p[2]))
        };
        let mut welded: HashMap<(i64, i64, i64), usize> = HashMap::new();
        for (i, v) in vertices.iter().enumerate() {
            welded.entry(key(*v)).or_insert(i);
        }
        // Keyed by *which part* of the rim, not just which rim: two pieces of
        // one face can each run along a different stretch of the same edge.
        let mut carried_edge: HashMap<RimPart, usize> = HashMap::new();
        let mut carried_vertices: Vec<(Vec<usize>, bool)> = Vec::new();
        for piece in &pieces {
            for (tag, id, pts, closed, _) in &piece.carried {
                let (Some(first), Some(last)) = (pts.first(), pts.last()) else {
                    continue;
                };
                let slot = (*tag, *id, key(*first), key(*last));
                if carried_edge.contains_key(&slot) {
                    continue;
                }
                let vs = pts
                    .iter()
                    .map(|p| {
                        *welded.entry(key(*p)).or_insert_with(|| {
                            vertices.push(*p);
                            vertices.len() - 1
                        })
                    })
                    .collect();
                carried_edge.insert(slot, carried_vertices.len());
                carried_vertices.push((vs, *closed));
            }
        }

        for piece in &mut pieces {
            for l in &mut piece.loops {
                for (slot, uv) in l.vertices.iter_mut().zip(&l.uv) {
                    if *slot < vertices.len() {
                        continue;
                    }
                    // Minted from *this* surface's parameters, so it lands on
                    // this surface and nowhere else in particular. Where the
                    // boundary is shared that is not enough: measured, a sphere
                    // piece minted a point 1.35e-4 off the sphere and 1.162e-3
                    // off the cylinder it was meant to meet, which is the one
                    // `EdgeOffSurface` left on an otherwise closed solid.
                    //
                    // Looking further afield before minting — the neighbouring
                    // cells, nearest vertex within a tolerance — does not help:
                    // there is no vertex there to find. The other side never made
                    // one, so this is not a duplicate to be welded but a point
                    // that needs *settling* onto both surfaces. Which cannot be
                    // done here, where only this piece's surface is known — but
                    // can be where the edges are built, since an edge names both.
                    //
                    // Doing it there works. Newton on the pair, applied to every
                    // shared vertex more than a thousandth of a tolerance off,
                    // with moves up to four tolerances allowed — the point is
                    // 1.16e-3 out, so a limit of one tolerance rejects exactly
                    // the move that is needed. Together with a box drawn where
                    // both solids are, `a_result_can_be_cut_again`'s bored ball
                    // comes back **valid, no defects**, and all 34 boolean tests
                    // pass.
                    //
                    // What stops it is not a regression. With that box a rod
                    // crossed by a rod *resolves* — 7 faces, 6 edges, valid —
                    // where today it declines `NeedsArrangement`. What fails is
                    // the STEP round trip of that new result, and face by face:
                    //
                    //     OUT  F0 cylinder edges=4 loops=Some([34, 146, 147])
                    //     BACK F0 cylinder edges=4 loops=None, wraps (true, false)
                    //
                    // The wall leaves as an outline with two holes where the
                    // crossing rod passes and comes back a whole wrapping
                    // cylinder with none, so the fill covers them. That is the
                    // importer's standing limitation — `loops: None` always,
                    // extent from the edges' vertices — already recorded against
                    // the STEP complement. The settle alone is green and moves
                    // nothing measurable; the box waits on that gap.
                    let p = piece.surface.point(uv[0], uv[1]);
                    *slot = *welded.entry(key(p)).or_insert_with(|| {
                        vertices.push(p);
                        vertices.len() - 1
                    });
                }
            }
        }

        let mut surfaces = Vec::with_capacity(pieces.len());
        let mut faces = Vec::with_capacity(pieces.len());
        let mut edges: Vec<Edge> = Vec::new();
        let mut edge_of_curve: Vec<Option<usize>> = vec![None; shared.len()];
        let mut edge_of_carried: Vec<Option<usize>> = vec![None; carried_vertices.len()];

        for p in pieces {
            surfaces.push(p.surface);
            // Each bounding curve becomes one `Edge`, created on first use and
            // then *shared* — which is what closes the seam.
            let mut face_edges = Vec::with_capacity(p.bounding.len());
            for id in p.bounding {
                let e = *edge_of_curve[id].get_or_insert_with(|| {
                    edges.push(Edge {
                        // Filled in once both faces are known; the pair is only
                        // used for the geometric self-check.
                        surfaces: (surfaces.len() - 1, surfaces.len() - 1),
                        vertices: shared[id].vertices.clone(),
                        // As the curve is. A chord is not a ring, and saying it
                        // was went unnoticed while only closed curves became
                        // edges.
                        closed: shared[id].closed,
                    });
                    edges.len() - 1
                });
                if edges[e].surfaces.0 == edges[e].surfaces.1 {
                    edges[e].surfaces.1 = surfaces.len() - 1;
                }
                face_edges.push(e);
            }
            for (tag, id, pts, _, _) in &p.carried {
                let (Some(first), Some(last)) = (pts.first(), pts.last()) else {
                    continue;
                };
                let Some(&slot) = carried_edge.get(&(*tag, *id, key(*first), key(*last))) else {
                    continue;
                };
                let e = *edge_of_carried[slot].get_or_insert_with(|| {
                    let (vs, closed) = carried_vertices[slot].clone();
                    edges.push(Edge {
                        surfaces: (surfaces.len() - 1, surfaces.len() - 1),
                        vertices: vs,
                        closed,
                    });
                    edges.len() - 1
                });
                if edges[e].surfaces.0 == edges[e].surfaces.1 {
                    edges[e].surfaces.1 = surfaces.len() - 1;
                }
                face_edges.push(e);
            }
            faces.push(Face {
                surface: surfaces.len() - 1,
                edges: face_edges,
                flipped: p.flipped,
                u_range: p.u_range,
                v_range: p.v_range,
                u_wraps: p.u_wraps,
                v_wraps: p.v_wraps,
                loops: (!p.loops.is_empty()).then_some(p.loops),
            });
        }
        // Cut a rim only when more than two faces claim it.
        //
        // A piece keeps the rims it runs along, whole, because trimming each one
        // to the part it actually runs along costs a corner overlap its shell
        // closure — a piece needs the whole rim to stay attached to its
        // neighbour. But a sphere cutting a column's side into an upper piece
        // and a lower one leaves both claiming the column's corner rim, and two
        // sides times two pieces is one edge with four faces.
        //
        // A piece cannot tell those apart from its own loops; here, with every
        // piece in hand, it is simply whether anyone else claims the same rim.
        // So keep them whole and cut the ones that turn out to need it.
        {
            let mut claims: Vec<Vec<usize>> = vec![Vec::new(); edges.len()];
            for (fi, f) in faces.iter().enumerate() {
                for &e in &f.edges {
                    claims[e].push(fi);
                }
            }
            // The stretch of an edge a face's loops run along, as a contiguous
            // run: picking out every vertex that passes leaves a polyline with
            // holes in it the moment one in the middle does not.
            let hits_of = |vs: &[usize], fi: usize| -> Option<Vec<bool>> {
                let ls = faces[fi].loops.as_ref()?;
                let s = &surfaces[faces[fi].surface];
                let on = |p: V3| {
                    ls.iter().any(|l| {
                        let n = l.uv.len();
                        (0..n).any(|i| {
                            let (x, y) = (l.uv[i], l.uv[(i + 1) % n]);
                            let (a, b) = (s.point(x[0], x[1]), s.point(y[0], y[1]));
                            v3::dist(a, p) + v3::dist(p, b) - v3::dist(a, b) <= tolerance * 4.0
                        })
                    })
                };
                let hits: Vec<bool> = vs.iter().map(|&v| on(vertices[v])).collect();
                hits.iter().any(|h| *h).then_some(hits)
            };
            // Worked out first, applied after: the runs are read from the
            // faces and the application writes to them.
            let mut plan: Vec<(usize, Vec<RimCut>)> = Vec::new();
            for e in 0..claims.len() {
                if claims[e].len() <= 2 || edges[e].vertices.len() < 3 {
                    continue;
                }
                let vs = edges[e].vertices.clone();
                // A closed rim repeats its first vertex last, and that repeat is
                // not a position of its own: a stretch spanning it comes out as
                // an edge from a vertex to itself, which leaves the stretches
                // either side with nothing to meet.
                let n = if vs.len() > 1 && vs.first() == vs.last() {
                    vs.len() - 1
                } else {
                    vs.len()
                };

                // Which faces reach each vertex of the rim, one row per face.
                let mut rows: Vec<(usize, Vec<bool>)> = Vec::new();
                let mut usable = true;
                for &fi in &claims[e] {
                    match hits_of(&vs, fi) {
                        Some(h) => rows.push((fi, h)),
                        None => {
                            usable = false;
                            break;
                        }
                    }
                }
                if !usable {
                    continue;
                }

                // A vertex reached by exactly two faces belongs to the stretch
                // they share. The rest are the corners where the second solid
                // took the rim away, and a vertex either side of one that only
                // one face reaches — the containment test is a tolerance wide,
                // so its ends are a vertex fuzzy. Neither names a stretch of its
                // own; both carry the label of the stretch before them, so every
                // vertex is used and the pieces still meet.
                let pair_at = |k: usize| -> Option<(usize, usize)> {
                    let mut it = rows.iter().filter(|(_, h)| h[k]).map(|(fi, _)| *fi);
                    match (it.next(), it.next(), it.next()) {
                        (Some(a), Some(b), None) => Some((a.min(b), a.max(b))),
                        _ => None,
                    }
                };
                let first = (0..n).find_map(|k| pair_at(k).map(|p| (k, p)));
                let Some((start, first_pair)) = first else {
                    continue;
                };
                // A vertex that names no pair is one of two things, and they
                // look identical here. Where the containment test blurs a
                // boundary by a vertex — it is a tolerance wide — the pair
                // before and after are the same, and carrying the one before is
                // right. Where three faces meet, the stretch genuinely changes
                // hands, and carrying is wrong: it puts the junction inside the
                // stretch before it, so that stretch runs on past where it
                // stopped bounding those two faces. A rim came out covering the
                // boundary of two different wall pieces with only one of them
                // credited.
                //
                // What separates them is what comes next: at a junction the pair
                // after is not the pair before.
                let next_pair = |from: usize| -> Option<(usize, usize)> {
                    (1..n).find_map(|d| pair_at((from + d) % n))
                };
                let mut label: Vec<(usize, usize)> = vec![first_pair; n];
                let mut carry = first_pair;
                for step in 0..n {
                    let k = (start + step) % n;
                    match pair_at(k) {
                        Some(p) => carry = p,
                        None => {
                            if let Some(next) = next_pair(k) {
                                if next != carry {
                                    carry = next;
                                }
                            }
                        }
                    }
                    label[k] = carry;
                }

                // Maximal stretches of one label, in order.
                let mut spans: Vec<((usize, usize), usize, usize)> = Vec::new();
                let mut at = start;
                let mut run_start = start;
                for step in 1..=n {
                    let k = (start + step) % n;
                    if step == n || label[k] != label[at] {
                        spans.push((label[at], run_start, (at + 1) % n));
                        run_start = k;
                    }
                    at = k;
                }
                // A label that holds for a single vertex makes a stretch from
                // that vertex to itself: an edge of no length, which leaves the
                // stretch either side of it with nothing to meet. Merge those
                // into the stretch before them, so the rim stays a chain.
                let mut merged: Vec<((usize, usize), usize, usize)> = Vec::new();
                for span in spans {
                    let len = (span.2 + n - span.1) % n;
                    match merged.last_mut() {
                        Some(prev) if len < 2 => prev.2 = span.2,
                        _ => merged.push(span),
                    }
                }
                // The rim closes, so a short *last* stretch folds into the
                // first rather than having nothing before it.
                if merged.len() > 1 {
                    let last = merged[merged.len() - 1];
                    if (last.2 + n - last.1) % n < 2 {
                        merged.remove(merged.len() - 1);
                        merged[0].1 = last.1;
                    }
                }
                let spans = merged;
                if spans.len() < 2 {
                    continue;
                }
                let cuts: Vec<RimCut> = spans
                    .iter()
                    .map(|&((a, b), from, to)| ((from, to), vec![a, b]))
                    .collect();
                plan.push((e, cuts));
            }
            for (e, groups) in plan {
                let vs = edges[e].vertices.clone();
                // As above: the repeated last vertex of a closed rim is not a
                // position of its own.
                let n = if vs.len() > 1 && vs.first() == vs.last() {
                    vs.len() - 1
                } else {
                    vs.len()
                };
                for (i, ((a, b), fs)) in groups.into_iter().enumerate() {
                    // `b` is where the next stretch begins, and taking it as the
                    // last vertex of this one is what makes the two meet. The
                    // walk is modular because a rim is closed and one stretch
                    // runs through its ends.
                    let part: Vec<usize> = {
                        let mut out = Vec::new();
                        let mut k = a;
                        loop {
                            out.push(vs[k]);
                            if k == b || out.len() > vs.len() {
                                break;
                            }
                            k = (k + 1) % n;
                        }
                        out
                    };
                    if part.len() < 2 {
                        continue;
                    }
                    let pair = (faces[fs[0]].surface, faces[fs[1]].surface);
                    let target = if i == 0 {
                        edges[e].vertices = part;
                        edges[e].closed = false;
                        edges[e].surfaces = pair;
                        e
                    } else {
                        edges.push(Edge {
                            surfaces: pair,
                            vertices: part,
                            closed: false,
                        });
                        edges.len() - 1
                    };
                    for &fi in &fs {
                        // A face on more than one stretch names the rim once and
                        // is about to be given several pieces of it, so after the
                        // first has taken its slot the rest are appended. The
                        // sphere a bore is cut into borders every piece.
                        let mut took = false;
                        for slot in faces[fi].edges.iter_mut() {
                            if *slot == e {
                                *slot = target;
                                took = true;
                            }
                        }
                        if !took {
                            faces[fi].edges.push(target);
                        }
                    }
                }
            }
        }

        // Give a face the edge for the stretch it shares with the other solid.
        //
        // A notch wall, a fillet, any face of the second solid: its loops are
        // right and its tessellation closes on them, but it carries only the
        // edges it had, and the part of its boundary meeting the first solid has
        // none. Without this a corner overlap's union comes back as *two closed
        // six-face shells, each of Euler 2* — the kernel reporting two separate
        // boxes where there is one solid, and `is_valid_solid` agreeing. That is
        // worse than an honest failure: such a result is accepted as the input
        // to another boolean.
        //
        // Nine earlier attempts handed the face a whole curve where it wants a
        // stretch, and failed alike because the two sides worked out what they
        // run along by different routes — a ring on one side, the subdivision's
        // sources on the other. Here, with every face in hand, both are asked
        // the same question: which of this curve's vertices do your own loops
        // name? Two faces that answer alike share that stretch and get one edge
        // for it.
        {
            let mut want: HashMap<(usize, Vec<usize>), Vec<usize>> = HashMap::new();
            for (fi, f) in faces.iter().enumerate() {
                let Some(ls) = f.loops.as_ref() else { continue };
                let named: std::collections::HashSet<usize> =
                    ls.iter().flat_map(|l| l.vertices.iter().copied()).collect();
                for (id, c) in shared.iter().enumerate() {
                    let run: Vec<usize> = c
                        .vertices
                        .iter()
                        .copied()
                        .filter(|v| named.contains(v))
                        .collect();
                    if run.len() >= 2 {
                        want.entry((id, run)).or_default().push(fi);
                    }
                }
            }
            for ((_, run), fs) in want {
                if fs.len() != 2 {
                    continue;
                }
                // Not if something already describes it. An edge on the same
                // ground is a second description, and the vertex between the two
                // then has three edges at it rather than two.
                let covered = fs.iter().any(|&fi| {
                    faces[fi].edges.iter().any(|&e| {
                        run.iter().all(|&v| {
                            edges[e].vertices.windows(2).any(|w| {
                                let (a, b) = (vertices[w[0]], vertices[w[1]]);
                                let p = vertices[v];
                                v3::dist(a, p) + v3::dist(p, b) - v3::dist(a, b) <= tolerance
                            })
                        })
                    })
                });
                if covered {
                    continue;
                }
                edges.push(Edge {
                    surfaces: (faces[fs[0]].surface, faces[fs[1]].surface),
                    vertices: run,
                    closed: false,
                });
                let e = edges.len() - 1;
                for &fi in &fs {
                    faces[fi].edges.push(e);
                }
            }
        }

        // Cut an edge that runs past a corner, when the halves come apart
        // cleanly.
        //
        // A cube's rim runs the whole side of the face it bounded; the operation
        // keeps half of that side, and the seam turns off at the middle. The rim
        // then passes *through* the vertex where the seam's edge stops — three
        // segments at a point the boundary merely turns at — and the face's
        // boundary does not close.
        //
        // Cutting is not enough by itself: both halves stay on the face and the
        // vertex gets two of their ends plus the seam's. Each half has to go to
        // the faces whose own loops name both of its ends. Done unconditionally
        // that closes the shells and wrecks the edge counts — some halves find
        // one face, some three. So it is done only where every half finds
        // exactly two, and otherwise the edge is left exactly as it was: this
        // can make a shell closed or leave it alone, never break what is already
        // shared.
        {
            let named: Vec<std::collections::HashSet<usize>> = faces
                .iter()
                .map(|f| {
                    f.loops
                        .as_ref()
                        .map(|ls| ls.iter().flat_map(|l| l.vertices.iter().copied()).collect())
                        .unwrap_or_default()
                })
                .collect();
            let mut claims: Vec<Vec<usize>> = vec![Vec::new(); edges.len()];
            for (fi, f) in faces.iter().enumerate() {
                for &e in &f.edges {
                    claims[e].push(fi);
                }
            }
            let mut stops: Vec<std::collections::HashSet<usize>> =
                vec![std::collections::HashSet::new(); faces.len()];
            for (e, cs) in claims.iter().enumerate() {
                let vs = &edges[e].vertices;
                if edges[e].closed || vs.len() < 2 {
                    continue;
                }
                for &fi in cs {
                    stops[fi].insert(vs[0]);
                    stops[fi].insert(vs[vs.len() - 1]);
                }
            }
            let mut plan: Vec<(usize, Vec<EdgeHalf>)> = Vec::new();
            for e in 0..claims.len() {
                let vs = edges[e].vertices.clone();
                if edges[e].closed || vs.len() < 3 {
                    continue;
                }
                let at: Vec<usize> = (1..vs.len() - 1)
                    .filter(|&k| claims[e].iter().any(|&fi| stops[fi].contains(&vs[k])))
                    .collect();
                if at.is_empty() {
                    continue;
                }
                let mut bounds = vec![0usize];
                bounds.extend(at);
                bounds.push(vs.len() - 1);
                let mut halves: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
                let mut clean = true;
                for w in bounds.windows(2) {
                    let part = vs[w[0]..=w[1]].to_vec();
                    if part.len() < 2 {
                        continue;
                    }
                    // Naming the ends is not enough. A rim through a bored
                    // column has both of its cut points on the *sphere's* loop
                    // too — they are on the curve where the two meet — while the
                    // rim between them runs through the sphere's inside. So the
                    // middle has to be on the face's boundary as well.
                    let takers: Vec<usize> = (0..faces.len())
                        .filter(|&fi| {
                            if !part.iter().all(|v| named[fi].contains(v)) {
                                return false;
                            }
                            let Some(ls) = faces[fi].loops.as_ref() else {
                                return true;
                            };
                            let sf = &surfaces[faces[fi].surface];
                            part.windows(2).all(|w| {
                                let mid = v3::scale(v3::add(vertices[w[0]], vertices[w[1]]), 0.5);
                                ls.iter().any(|l| {
                                    let n = l.uv.len();
                                    (0..n).any(|i| {
                                        let (x, y) = (l.uv[i], l.uv[(i + 1) % n]);
                                        let (a, b) = (sf.point(x[0], x[1]), sf.point(y[0], y[1]));
                                        v3::dist(a, mid) + v3::dist(mid, b) - v3::dist(a, b)
                                            <= tolerance * 4.0
                                    })
                                })
                            })
                        })
                        .collect();
                    // Two faces, or none at all. A half no face's loops name is
                    // the part of the rim the operation took away — interior to
                    // the result, on nobody's boundary — and it goes. Anything
                    // else means the cut does not come apart cleanly and the
                    // edge is left as it was.
                    match takers.len() {
                        2 => halves.push((part, takers)),
                        0 => {}
                        _ => {
                            clean = false;
                            break;
                        }
                    }
                }
                if clean && !halves.is_empty() {
                    plan.push((e, halves));
                }
            }
            for (e, halves) in plan {
                for f in faces.iter_mut() {
                    f.edges.retain(|&x| x != e);
                }
                for (i, (part, takers)) in halves.into_iter().enumerate() {
                    let pair = (faces[takers[0]].surface, faces[takers[1]].surface);
                    let slot = if i == 0 {
                        edges[e].vertices = part;
                        edges[e].closed = false;
                        edges[e].surfaces = pair;
                        e
                    } else {
                        edges.push(Edge {
                            surfaces: pair,
                            vertices: part,
                            closed: false,
                        });
                        edges.len() - 1
                    };
                    for &fi in &takers {
                        faces[fi].edges.push(slot);
                    }
                }
            }
        }

        // One boundary described as a ring on one side and as sides on the
        // other.
        //
        // Two cubes sharing a wall: the lower one's top face gets a square hole,
        // one closed edge of four corners, and the upper one's four walls each
        // bring the side they stand on. Same square, five edges, and every one
        // of them claimed by a single face — so neither shell closes and the
        // union is not a solid.
        //
        // Cut the ring at its own corners and let the pieces meet the sides they
        // coincide with. Only edges no second face claims are touched, and only
        // where each piece finds exactly one partner, so this can join what was
        // separate and can do nothing else.
        {
            let mut claims: Vec<Vec<usize>> = vec![Vec::new(); edges.len()];
            for (fi, f) in faces.iter().enumerate() {
                for &e in &f.edges {
                    claims[e].push(fi);
                }
            }
            let lone: Vec<usize> = (0..edges.len()).filter(|&e| claims[e].len() == 1).collect();
            let same_span = |x: &[usize], y: &[usize]| -> bool {
                if x.len() < 2 || y.len() < 2 {
                    return false;
                }
                let ends = |v: &[usize]| (vertices[v[0]], vertices[v[v.len() - 1]]);
                let ((a0, a1), (b0, b1)) = (ends(x), ends(y));
                (v3::dist(a0, b0) <= tolerance && v3::dist(a1, b1) <= tolerance)
                    || (v3::dist(a0, b1) <= tolerance && v3::dist(a1, b0) <= tolerance)
            };
            for &ring in &lone {
                let vs = edges[ring].vertices.clone();
                if !edges[ring].closed || vs.len() < 4 {
                    continue;
                }
                // Its corners, as consecutive pieces.
                let parts: Vec<Vec<usize>> = vs.windows(2).map(|w| w.to_vec()).collect();
                let matched: Vec<Option<usize>> = parts
                    .iter()
                    .map(|part| {
                        let mut found = None;
                        for &other in &lone {
                            if other == ring || !same_span(part, &edges[other].vertices) {
                                continue;
                            }
                            if found.is_some() {
                                return None;
                            }
                            found = Some(other);
                        }
                        found
                    })
                    .collect();
                if matched.iter().any(|m| m.is_none()) {
                    continue;
                }
                // Every side has its partner: give the ring's face those edges
                // instead of the ring.
                let holder = claims[ring][0];
                faces[holder].edges.retain(|&e| e != ring);
                for m in matched.into_iter().flatten() {
                    faces[holder].edges.push(m);
                    edges[m].surfaces = (faces[holder].surface, faces[claims[m][0]].surface);
                }
            }
        }

        // An edge no face names is not an edge. The pairing above leaves the
        // ring it replaced behind, and a body carrying one reads as defective
        // for a reason that is only bookkeeping.
        {
            let mut used = vec![false; edges.len()];
            for f in &faces {
                for &e in &f.edges {
                    used[e] = true;
                }
            }
            if used.iter().any(|u| !u) {
                let mut slot = vec![usize::MAX; edges.len()];
                let mut kept: Vec<Edge> = Vec::with_capacity(edges.len());
                for (e, edge) in edges.into_iter().enumerate() {
                    if used[e] {
                        slot[e] = kept.len();
                        kept.push(edge);
                    }
                }
                edges = kept;
                for f in faces.iter_mut() {
                    for e in f.edges.iter_mut() {
                        *e = slot[*e];
                    }
                }
            }
        }

        settle_shared_vertices(&surfaces, &mut vertices, &mut faces, tolerance);
        prune_orphan_vertices(&mut vertices, &mut edges, &mut faces);
        materialise_seams(&mut edges, &mut faces);
        materialise_seam_steps(&mut edges, &mut faces);
        materialise_shared(&mut edges, &mut faces);
        split_at_seam_ends(&mut edges, &mut faces);
        let mut body = Body::from_parts(surfaces, vertices, edges, faces);
        body.restate_rings();
        body
    }
}

/// Where a swept face's parameter rectangle should be cut open, decided from the
/// curves that land on it — before any face is split, so its rims can be given
/// that point while both faces still read the same list.
///
/// The choice is the same in both cases the split makes: where a curve reaches
/// all the way round the seam must fall *on* it, and otherwise it goes in the
/// widest stretch no curve occupies. Either way the rims need a point there,
/// which is why this is decided once and used by both.
fn seam_origin(
    face: &Face,
    surface: &Surface,
    curve_ids: &[usize],
    shared: &[SharedCurve],
    vertices: &[V3],
) -> Option<f64> {
    let period = face.u_range.1 - face.u_range.0;
    if !period.is_finite() || period <= 0.0 {
        return None;
    }
    let v_span = (face.v_range.1 - face.v_range.0).abs().max(1e-12);
    // Whether this face will be subdivided at all. A face whose curves are every
    // one at constant `v` is a stack of bands and keeps its rectangle; only the
    // others are cut open, and only they need their rims reaching the cut.
    let mut wants_seam = false;
    for &id in curve_ids {
        let uv: Vec<(f64, f64)> = shared[id]
            .vertices
            .iter()
            .filter_map(|&vi| vertices.get(vi))
            .filter_map(|p| surface.invert(*p))
            .map(|(u, v)| (face.u_range.0 + (u - face.u_range.0).rem_euclid(period), v))
            .collect();
        // Two points is a curve. A straight seam — a plane through a
        // cylinder's axis cuts it in a line — arrives with exactly two, and
        // skipping those meant a face cut only by straight seams looked like a
        // face nothing wanted a seam on, so it declined outright. Rounding an
        // edge is that case: four curves, two of them lines along the axis.
        if uv.len() < 2 {
            continue;
        }
        // A curve at constant `v` cuts the sweep into bands and needs no seam:
        // that is `split_swept`'s case, and it keeps the parameter rectangles.
        // Forcing a seam onto the rims for it costs the bands a triangle each —
        // three open edges on a bored cone, which is how this was noticed.
        let (v_lo, v_hi) = uv.iter().fold((f64::MAX, f64::MIN), |(l, h), (_, v)| {
            (l.min(*v), h.max(*v))
        });
        if v_hi - v_lo <= v_span * 1e-3 {
            continue;
        }
        wants_seam = true;
        let us: Vec<f64> = uv.iter().map(|(u, _)| *u).collect();
        // Does it reach all the way round? Only then is a seam forced onto it.
        let mut sorted = us.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let gap = sorted
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(sorted[0] + period - sorted[sorted.len() - 1], f64::max);
        if gap < period * 0.02 {
            return Some(us[0]);
        }
    }
    // Nothing wraps, so nothing forces the seam onto a curve — but the face is
    // still being cut open, and its rims still have to reach where it is cut.
    // The seam goes in the widest stretch no curve occupies.
    if !wants_seam || !face.u_wraps {
        return None;
    }
    let used: Vec<f64> = curve_ids
        .iter()
        .flat_map(|&id| shared[id].vertices.iter())
        .filter_map(|&vi| vertices.get(vi))
        .filter_map(|p| surface.invert(*p))
        .map(|(u, _)| u)
        .collect();
    (!used.is_empty()).then(|| widest_gap(&used, face.u_range.0, period))
}

/// Cut a chord where it crosses the seam.
///
/// Folding puts every point back inside the rectangle, but a chord that runs
/// *across* the seam then jumps a whole period between two consecutive points,
/// and that jump draws, in parameter space, a line clear across the face. A
/// sphere with a square column bored through it came back as fifteen slivers
/// because of two such lines: the column's openings leave no gap in `u`, so the
/// seam has to fall on the curves, and it fell in the middle of an arc.
///
/// The crossing already carries a vertex — [`insert_seam_vertex`] puts one there
/// before either face looks at the curve — so the cut is exact rather than
/// interpolated: one run ends on one seam edge, the next begins at the same
/// point on the other, and both name the same vertex.
/// A ring that crosses the seam must cross it at the same place on both sides.
///
/// Both sides are the same line, so the parameter that is not the periodic one
/// has to agree across the crossing. `ball - bore - cross` at 2e-4 stores a
/// sphere ring that spans exactly one period and does not:
///
/// ```text
/// #158  v370  u=7.853982  v=-1.230959     right of the seam
/// #159  v688  u=7.853982  v=-1.211726
///  …
/// #572  v688  u=1.570796  v=-1.211726     left of it, 2pi away exactly
/// #0    v371  u=1.610563  v=-1.230959     0.0398 *past* the seam
/// ```
///
/// At `v = -1.230959` — the latitude where the bore's rim meets the seam — the
/// right side carries the rim vertex and the left side has nothing, so the ring
/// steps to the next rim vertex along instead. The two legs of that step bound a
/// wedge no triangle covers, and they are two of the four edges the fill leaves
/// open at that tolerance. It bites only when a seam crossing lands on a rim
/// vertex, which is why four other tolerances are clean.
///
/// Worth being exact about what the ring is *not*: its `uv` points are all
/// distinct — zero duplicated positions in 573 — and the repeated vertex indices
/// are the seam's two sides naming one point, not a path retraced. An earlier
/// reading of this as a spur run out and back was wrong.
///
/// Repaired for a long time by a pass that reconciled the two runs afterwards,
/// and now not needed: both sides of a seam are cut from one list, so they carry
/// the same values and the same names, and the corner each end is a place with
/// two parameters rather than two places
/// correctly. "Both sides span the same `v`" is false of a ring that merely
/// *spans* a period without lying along the seam — its extreme points are single
/// and their `v` has no reason to agree — and asserting it there cost five
/// tests, `rod - cross` among them. What is true is narrower: where the ring
/// runs *along* the seam on both sides, two or more points at one `u`, those two
/// runs are one line and must end together.
fn split_at_seam(
    uv: &[[f64; 2]],
    order: &[usize],
    origin: f64,
    period: f64,
    axis: usize,
) -> Vec<(Vec<[f64; 2]>, Vec<usize>)> {
    let other = 1 - axis;
    let mut out = Vec::new();
    let mut run: (Vec<[f64; 2]>, Vec<usize>) = (Vec::new(), Vec::new());
    for i in 0..uv.len() {
        run.0.push(uv[i]);
        run.1.push(order[i]);
        let Some(&next) = uv.get(i + 1) else { continue };
        if (next[axis] - uv[i][axis]).abs() <= period * 0.5 {
            continue;
        }
        // End the run *on* the edge rather than at the neighbour brought back
        // alongside it: the neighbour is a whole step past the seam, and a run
        // that overshoots the rectangle is no longer a chord of it.
        let shift = if next[axis] > uv[i][axis] {
            -period
        } else {
            period
        };
        let (a, b) = (uv[i], {
            let mut q = next;
            q[axis] = next[axis] + shift;
            q
        });
        let edge = if b[axis] > a[axis] {
            origin + period
        } else {
            origin
        };
        let far = if edge > origin {
            origin
        } else {
            origin + period
        };
        let eps = period * 1e-6;

        // Take the across-seam parameter from the crossing vertex, if the curve
        // carries one.
        //
        // It nearly always does: `insert_seam_vertex` puts one there, and puts
        // it there by bisecting onto the surface rather than by interpolating
        // between two samples. That distinction is the whole point. A `v` got
        // by interpolation lands on this surface but not on the one across the
        // seam — by the sagitta of a step, which on a bored sphere was 3e-3,
        // enough that the two faces described the seam with different points
        // and six edges stayed open.
        let at_seam = if (a[axis] - edge).abs() <= eps {
            a[other]
        } else if (b[axis] - edge).abs() <= eps {
            b[other]
        } else {
            let t = if (b[axis] - a[axis]).abs() > f64::EPSILON {
                ((edge - a[axis]) / (b[axis] - a[axis])).clamp(0.0, 1.0)
            } else {
                0.0
            };
            a[other] + (b[other] - a[other]) * t
        };

        // End the run *on* the edge rather than at the neighbour brought back
        // alongside it: the neighbour is a whole step past the seam, and a run
        // that overshoots the rectangle is no longer a chord of it. If the
        // crossing vertex is already the last point, it is that terminator.
        if (a[axis] - edge).abs() > eps {
            let mut q = a;
            q[axis] = edge;
            q[other] = at_seam;
            run.0.push(q);
            run.1.push(order[i + 1]);
        }
        out.push(std::mem::take(&mut run));
        // The far edge is the same place on the surface, so the next run starts
        // there — the two sides of the cut meet at one point, not at a gap. When
        // the crossing vertex is the point coming up, it starts the run itself.
        if (next[axis] - far).abs() > eps {
            let mut q = a;
            q[axis] = far;
            q[other] = at_seam;
            run.0.push(q);
            run.1.push(order[i + 1]);
        }
    }
    if run.0.len() > 1 {
        out.push(run);
    }
    out.retain(|(p, _)| p.len() > 1);
    out
}

/// Put a vertex where a curve crosses a face's seam, in the curve's own order.
///
/// The curve is shared: the face on the *other* side of it joins the same points
/// with an edge running straight across the crossing. If only the wrapping face
/// splits that edge, the two describe the seam with different edges and it comes
/// apart — so the split goes into the shared curve, once, before either face
/// looks at it.
/// Returns the `v` of the vertex it put in *and* which vertex it is, so the
/// face's outline can be broken at the same place and name the same point: a
/// chord cut from a wrapping ring meets the seam edge here, and an edge that
/// knows only where has to find out which by rounding, which is how one place
/// becomes two vertices.
fn insert_seam_vertex(
    curve: &mut SharedCurve,
    vertices: &mut Vec<V3>,
    surface: &Surface,
    across: Option<&Surface>,
    origin: f64,
    range: (f64, f64),
) -> Option<(f64, usize)> {
    let period = range.1 - range.0;
    if !period.is_finite() || period <= 0.0 || curve.vertices.len() < 3 {
        return None;
    }
    let fold = |u: f64| origin + (u - origin).rem_euclid(period);
    let u_of = |vi: usize| -> Option<f64> {
        vertices
            .get(vi)
            .and_then(|p| surface.invert(*p))
            .map(|(u, _)| fold(u))
    };

    let n = curve.vertices.len();
    let last = if curve.closed { n - 1 } else { n };
    for i in 0..last {
        let j = (i + 1) % last;
        let (Some(a), Some(b)) = (u_of(curve.vertices[i]), u_of(curve.vertices[j])) else {
            return None;
        };
        // The one step that jumps the whole way is the crossing.
        if (b - a).abs() < period * 0.5 {
            continue;
        }
        // Measured from the seam, not from zero: `fold` puts these in
        // `[origin, origin + period)`, so a value "at the seam" is `origin`.
        let (from_a, from_b) = (a - origin, b - origin);
        if from_a.min(from_b) <= period * 1e-9 || (period - from_a.max(from_b)) <= period * 1e-9 {
            return None; // already split there
        }
        let (pa, pb) = (vertices[curve.vertices[i]], vertices[curve.vertices[j]]);

        // Bisect for the crossing rather than interpolating to it.
        //
        // The straight line between two points of a traced curve does not lie
        // on the surface, so a fraction worked out from their parameters lands
        // somewhere else once it is inverted back — by most of a step, which is
        // to say not on the seam at all. What is wanted is the point whose own
        // parameter *is* the seam, and the signed offset to it changes sign
        // exactly there.
        let offset = |t: f64| -> Option<f64> {
            let p = v3::add(pa, v3::scale(v3::sub(pb, pa), t));
            let (u, _) = surface.invert(p)?;
            let mut d = (u - origin).rem_euclid(period);
            if d > period * 0.5 {
                d -= period;
            }
            Some(d)
        };
        let (Some(da), Some(_db)) = (offset(0.0), offset(1.0)) else {
            return None;
        };
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..60 {
            let m = 0.5 * (lo + hi);
            let dm = offset(m)?;
            if (dm > 0.0) == (da > 0.0) {
                lo = m;
            } else {
                hi = m;
            }
            if hi - lo <= 1e-14 {
                break;
            }
        }
        let t = 0.5 * (lo + hi);
        let mid = v3::add(pa, v3::scale(v3::sub(pb, pa), t));
        // Onto the seam, and onto the surface across the curve, at once.
        //
        // The bisection above finds *where along the step* the seam is; the
        // point it returns is on the straight line between two samples of a
        // traced curve, off both surfaces by that step's sagitta. Evaluating its
        // parameters on this surface moves the error to the other one instead,
        // and alternating between the two converges to whichever the last step
        // enforced — a rod off by a sagitta or a column off by twice the
        // tolerance, depending which way round it ended.
        //
        // But this is not three unknowns. The vertex is on this surface's seam
        // by definition, and that seam is a curve `surface.point(origin, v)` with
        // one parameter. So it is a root along `v`: the place where that curve
        // meets the surface across this one.
        let mid = across
            .and_then(|other| {
                let on_seam = |v: f64| surface.point(origin, v);
                // Signed, so a root can be bracketed: the offset from the other
                // surface, along that surface's own normal.
                let gap = |v: f64| -> Option<f64> {
                    let p = on_seam(v);
                    let (u2, v2) = other.invert(p)?;
                    let n = other.normal(u2, v2)?;
                    Some(v3::dot(v3::sub(p, other.point(u2, v2)), n))
                };
                let (_, va) = surface.invert(pa)?;
                let (_, vb) = surface.invert(pb)?;
                let (mut lo, mut hi) = (va, vb);
                let (ga, gb) = (gap(lo)?, gap(hi)?);
                if ga == 0.0 {
                    return Some(on_seam(lo));
                }
                if (ga > 0.0) == (gb > 0.0) {
                    return None; // not bracketed; the chord point stands
                }
                for _ in 0..60 {
                    let m = 0.5 * (lo + hi);
                    let gm = gap(m)?;
                    if (gm > 0.0) == (ga > 0.0) {
                        lo = m;
                    } else {
                        hi = m;
                    }
                    if (hi - lo).abs() <= 1e-15 {
                        break;
                    }
                }
                Some(on_seam(0.5 * (lo + hi)))
            })
            .unwrap_or_else(|| {
                // The bracket can fail, and then the chord point stands — but a
                // seam vertex is on *this* surface's seam by definition, and the
                // chord point is not: measured 2.637e-3 off it on a rod bored
                // across, against a tolerance of 1e-3. Nothing else could then
                // land on it, because everything else that names the seam is
                // built from `surface.point(origin, v)`.
                match surface.invert(mid) {
                    Some((_, v)) => surface.point(origin, v),
                    None => mid,
                }
            });
        let cut = surface.invert(mid).map(|(_, v)| (v, vertices.len()));
        vertices.push(mid);
        curve.vertices.insert(i + 1, vertices.len() - 1);
        if curve.closed {
            // The repeated first index moved with the insert; keep it last.
            let first = curve.vertices[0];
            let end = curve.vertices.len() - 1;
            curve.vertices[end] = first;
        }
        return cut;
    }
    None
}

/// Put a point at parameter `u` into a rim, where it belongs along it.
///
/// The rim lies at a constant `v`, so the point is on the surface exactly; what
/// this decides is only *where in the list* it goes.
fn insert_at_u(points: &mut Vec<V3>, surface: &Surface, u: f64) {
    let Some((_, v)) = points.first().and_then(|p| surface.invert(*p)) else {
        return;
    };
    // Only a rim: every point at the same `v`.
    if points
        .iter()
        .filter_map(|p| surface.invert(*p))
        .any(|(_, w)| (w - v).abs() > 1e-9)
    {
        return;
    }
    let at = surface.point(u, v);
    if points.iter().any(|p| v3::dist(*p, at) <= 1e-9) {
        return;
    }
    // Between the two neighbours it falls between along the rim.
    let mut best = (f64::MAX, 0usize);
    for i in 0..points.len().saturating_sub(1) {
        let d = v3::dist(points[i], at) + v3::dist(at, points[i + 1])
            - v3::dist(points[i], points[i + 1]);
        if d < best.0 {
            best = (d, i + 1);
        }
    }
    points.insert(best.1, at);
}

/// Re-cut a ring that wraps a periodic parameter into a chord running from one
/// seam edge to the other.
///
/// `None` when the ring doubles back — more than one point at the same `u` —
/// because then it is not a single cut across the sweep and pretending it is
/// would join pieces that are apart.
fn seam_chord(uv: &[[f64; 2]], origin: f64, period: f64) -> Option<(Vec<[f64; 2]>, Vec<usize>)> {
    // *Rotated* to start at the seam, never sorted.
    //
    // The same curve bounds a face on the other side too, where it is a plain
    // ring in the order it was traced. Sorting these points by parameter gives
    // the same *set* in a different order, so the two faces end up joining them
    // with different edges — and a seam is held by edges, not by points. A
    // rotation leaves every edge intact and only chooses where to cut.
    let n = uv.len();
    if n < 3 {
        return None;
    }
    let fold = |u: f64| origin + (u - origin).rem_euclid(period);

    // The one place the folded parameter jumps is where the curve crosses the
    // seam. Which way it jumps says which way the curve runs.
    let mut cut = 0usize;
    let mut worst = 0.0;
    let mut rising = true;
    for i in 0..n {
        let step = fold(uv[(i + 1) % n][0]) - fold(uv[i][0]);
        if step.abs() > worst {
            worst = step.abs();
            cut = (i + 1) % n;
            // A jump *up* means the curve was running down through the seam.
            rising = step < 0.0;
        }
    }
    if worst < period * 0.5 {
        return None; // it does not wrap after all
    }

    // Taken in the direction of increasing `u`, so the chord starts at the near
    // seam edge and ends at the far one. A curve traced the other way rotates
    // onto the *high* end instead, and its near end is then a whole sample step
    // short of the boundary — which reads as a cut stopping in open space.
    let step: isize = if rising { 1 } else { -1 };
    let mut cut = if rising { cut } else { (cut + n - 1) % n };

    // If the point *before* the cut is already at the far edge, start there.
    //
    // A curve traced finely enough can happen to have a sample sitting on the
    // seam. `insert_seam_vertex` sees that and leaves the curve alone — rightly,
    // there is nothing to add — but the rotation then begins at the sample
    // *after* the jump, which is a step past the near edge, and the chord starts
    // in open space. That sample is the seam: `fold` puts it a hair under
    // `origin + period`, and the same place a hair over `origin`.
    let before = ((cut as isize - step).rem_euclid(n as isize)) as usize;
    let at_edge = |u: f64| {
        let f = fold(u);
        (f - origin).min(origin + period - f) <= period * 1e-6
    };
    if at_edge(uv[before][0]) {
        cut = before;
    }

    let mut out = Vec::with_capacity(n + 1);
    let mut idx = Vec::with_capacity(n + 1);
    for k in 0..n {
        let i = ((cut as isize + step * k as isize).rem_euclid(n as isize)) as usize;
        out.push([fold(uv[i][0]), uv[i][1]]);
        idx.push(i);
    }
    // The ring opened at the seam: its first point again, at the far edge.
    //
    // Nothing is fabricated and nothing is dropped. The closing edge — the one
    // running from the last point back to the first — becomes the edge reaching
    // the far seam, so the chord carries *every* edge the ring had, and the face
    // on the other side of the curve joins the same points the same way.
    out.push([origin + period, out[0][1]]);
    idx.push(idx[0]);
    Some((out, idx))
}
/// The start of the widest stretch of a periodic parameter that `used` leaves
/// empty — where a wrapping domain can be cut open without cutting a curve.
fn widest_gap(used: &[f64], lo: f64, period: f64) -> f64 {
    if used.is_empty() {
        return lo;
    }
    let mut sorted: Vec<f64> = used.iter().map(|u| (u - lo).rem_euclid(period)).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut best = (
        sorted[0] + period - sorted[sorted.len() - 1],
        sorted[sorted.len() - 1],
    );
    for w in sorted.windows(2) {
        if w[1] - w[0] > best.0 {
            best = (w[1] - w[0], w[0]);
        }
    }
    // Halfway into the gap, so neither end is near a curve.
    lo + best.1 + best.0 * 0.5
}

/// Give a boundary two faces walk, but no edge covers, an edge.
///
/// Pairs runs *across* faces. A ring can also need this within one face: a wall
/// between two rims stores an outline that runs rim, seam, rim, seam, and
/// measured on a rod crossed by a rod that outline is 34 points with **2 steps
/// no edge covers** — one crossing at each end, the same line traversed both
/// ways. Every vertex is on an edge, so `materialise_seams` (which looks for
/// vertices no edge names) cannot see it, and the two runs are one segment each,
/// so the reversed-run pairing does not fire either.
///
/// That is what stops the file carrying such a face: an `EDGE_LOOP` needs an
/// edge for every step, and two are missing.
///
/// [`materialise_seams`] looks for *vertices* no edge names, which finds a seam
/// a face walks twice. It cannot find this: where two pieces meet along a line
/// the arrangement drew, both ends of that line are vertices of other edges, and
/// only the step between them is uncovered. So look at segments instead — each
/// consecutive pair of a ring — and ask whether any edge of the face walks it.
///
/// Measured on a bored ball cut by a column: the two halves of one wall arc each
/// walk the line at (0, −1) up the bore, each has both its ends from the rim
/// stretches either side, and neither has an edge for the line. Both faces were
/// open at exactly those two vertices.
///
/// Either direction: which way two faces walk a boundary they share is a matter
/// of their orientation, and both are the same curve.
fn materialise_shared(edges: &mut Vec<Edge>, faces: &mut [Face]) {
    let mut covered: HashSet<(usize, usize)> = HashSet::new();
    for e in edges.iter() {
        for w in e.vertices.windows(2) {
            covered.insert((w[0].min(w[1]), w[0].max(w[1])));
        }
    }
    // Each face's ring steps that no edge anywhere walks, as runs.
    let mut runs: Vec<(usize, Vec<usize>)> = Vec::new();
    for (fi, face) in faces.iter().enumerate() {
        let Some(loops) = face.loops.as_ref() else {
            continue;
        };
        for ring in loops {
            let n = ring.vertices.len();
            if n < 2 {
                continue;
            }
            let gap = |i: usize| {
                let (a, b) = (ring.vertices[i], ring.vertices[(i + 1) % n]);
                a != b && !covered.contains(&(a.min(b), a.max(b)))
            };
            let Some(start) = (0..n).find(|&i| gap(i) && !gap((i + n - 1) % n)) else {
                continue;
            };
            let mut cur: Vec<usize> = Vec::new();
            for step in 0..n {
                let i = (start + step) % n;
                if gap(i) {
                    if cur.is_empty() {
                        cur.push(ring.vertices[i]);
                    }
                    cur.push(ring.vertices[(i + 1) % n]);
                } else if !cur.is_empty() {
                    runs.push((fi, std::mem::take(&mut cur)));
                }
            }
            if !cur.is_empty() {
                runs.push((fi, cur));
            }
        }
    }

    let mut taken = vec![false; runs.len()];
    for a in 0..runs.len() {
        if taken[a] || runs[a].1.len() < 2 {
            continue;
        }
        for b in (a + 1)..runs.len() {
            if taken[b] || runs[b].0 == runs[a].0 {
                continue;
            }
            let same = runs[b].1 == runs[a].1;
            let mirrored = runs[b].1.iter().rev().eq(runs[a].1.iter());
            if !same && !mirrored {
                continue;
            }
            taken[a] = true;
            taken[b] = true;
            let (fa, fb) = (runs[a].0, runs[b].0);
            let e = edges.len();
            edges.push(Edge {
                surfaces: (faces[fa].surface, faces[fb].surface),
                vertices: runs[a].1.clone(),
                closed: false,
            });
            faces[fa].edges.push(e);
            faces[fb].edges.push(e);
            break;
        }
    }
}

/// Cut an edge where it crosses a seam.
///
/// A hole edge can run across the line the parameterisation closes on. In three
/// dimensions it is one curve; in the parameters it is two runs at opposite
/// ends of the domain, and the face's boundary walks it as two. So the edge is
/// one thing and the ring uses it as two, and nothing that reads the ring off
/// the edges — this crate's STEP writer among them — can state such a face.
///
/// Every CAD system cuts these, and the cut costs nothing here: the vertex is
/// already in the edge and already in the ring, so this is subdivision of the
/// topology with no new geometry and nothing to converge.
///
/// The places to cut are the seam edge's own ends, which is where a boundary
/// meets a seam by definition.
fn split_at_seam_ends(edges: &mut Vec<Edge>, faces: &mut [Face]) {
    let mut cuts: HashSet<usize> = HashSet::new();
    for face in faces.iter() {
        for &e in &face.edges {
            if face.edges.iter().filter(|&&x| x == e).count() != 2 {
                continue;
            }
            if let Some(edge) = edges.get(e) {
                if let (Some(&a), Some(&b)) = (edge.vertices.first(), edge.vertices.last()) {
                    cuts.insert(a);
                    cuts.insert(b);
                }
            }
        }
    }
    if cuts.is_empty() {
        return;
    }

    let mut replace: HashMap<usize, Vec<usize>> = HashMap::new();
    for ei in 0..edges.len() {
        let v = edges[ei].vertices.clone();
        if v.len() < 3 || edges[ei].closed {
            continue;
        }
        let inner: Vec<usize> = (1..v.len() - 1).filter(|&i| cuts.contains(&v[i])).collect();
        if inner.is_empty() {
            continue;
        }
        let surfaces = edges[ei].surfaces;
        let mut parts: Vec<Vec<usize>> = Vec::new();
        let mut start = 0usize;
        for &i in &inner {
            // The cut vertex ends one part and begins the next, so the two
            // still meet where they did.
            parts.push(v[start..=i].to_vec());
            start = i;
        }
        parts.push(v[start..].to_vec());

        let mut ids = Vec::with_capacity(parts.len());
        for (k, part) in parts.into_iter().enumerate() {
            if k == 0 {
                edges[ei].vertices = part;
                ids.push(ei);
            } else {
                ids.push(edges.len());
                edges.push(Edge {
                    surfaces,
                    vertices: part,
                    closed: false,
                });
            }
        }
        replace.insert(ei, ids);
    }
    if replace.is_empty() {
        return;
    }

    for face in faces.iter_mut() {
        let mut out = Vec::with_capacity(face.edges.len());
        for &e in &face.edges {
            match replace.get(&e) {
                Some(ids) => out.extend(ids.iter().copied()),
                None => out.push(e),
            }
        }
        face.edges = out;
    }
}

/// Give a face's seam runs an edge.
///
/// Where a face's boundary runs along the seam of a closed surface there is no
/// edge, because an edge is shared by two faces and here the face meets itself.
/// The ring names those vertices anyway — once going up and once coming back —
/// and measured on a ball with a square column through it, the two runs are the
/// *same* vertices in opposite order: 51 of 51 shared, one exactly the reverse
/// of the other. So the curve is already there and already shared; only the
/// edge is missing.
///
/// That pairing is the test used here, rather than anything about parameters: a
/// run that appears twice reversed is a seam, whatever surface it lies on. A run
/// that does not pair is left alone and whatever declines downstream still
/// declines.
///
/// An edge used twice by one face is what a seam *is*, and `Defect::EdgeFaceCount`
/// counts uses rather than distinct faces so that it can be said.
/// Drop vertices nothing refers to.
///
/// Assembly mints a vertex whenever a piece names a point, and some of those
/// points end up on no edge and in no ring — a piece that was discarded, a
/// crossing that was superseded. They are invisible in the solid and they are
/// not free: they read as *duplicates* of the points that survived, twelve of
/// them among a bored ball's hundred and fifty-seven, one pair 7.1e-16 apart.
/// Anything asking "are two vertices of this body the same point" gets a yes
/// about geometry that is not there.
fn prune_orphan_vertices(vertices: &mut Vec<V3>, edges: &mut [Edge], faces: &mut [Face]) {
    let mut used = vec![false; vertices.len()];
    for e in edges.iter() {
        for v in &e.vertices {
            if let Some(slot) = used.get_mut(*v) {
                *slot = true;
            }
        }
    }
    for f in faces.iter() {
        let Some(ls) = f.loops.as_ref() else {
            continue;
        };
        for l in ls {
            for v in &l.vertices {
                if let Some(slot) = used.get_mut(*v) {
                    *slot = true;
                }
            }
        }
    }
    if used.iter().all(|u| *u) {
        return;
    }
    let mut slot = vec![usize::MAX; vertices.len()];
    let mut kept: Vec<V3> = Vec::with_capacity(vertices.len());
    for (i, keep) in used.iter().enumerate() {
        if *keep {
            slot[i] = kept.len();
            kept.push(vertices[i]);
        }
    }
    *vertices = kept;
    for e in edges.iter_mut() {
        for v in e.vertices.iter_mut() {
            *v = slot.get(*v).copied().unwrap_or(usize::MAX);
        }
    }
    for f in faces.iter_mut() {
        let Some(ls) = f.loops.as_mut() else {
            continue;
        };
        for l in ls {
            for v in l.vertices.iter_mut() {
                *v = slot.get(*v).copied().unwrap_or(usize::MAX);
            }
        }
    }
}

/// Make an edge of a seam a ring crosses in a single step.
///
/// [`materialise_seams`] looks for a *run* of ring vertices that no edge names,
/// which is what a seam looks like when it is sampled. A swept face's seam is
/// not sampled: it is one straight step from one rim to the other, and both its
/// ends are rim vertices that edges do name — so the run never starts and the
/// step stays backed by nothing.
///
/// Measured on a rod with a bore across it: of a 36-step outer ring, exactly two
/// steps are unbacked, `v90 -> v107` and `v107 -> v90`. The same line, walked up
/// one side of the parameter domain and down the other, which is what a seam is.
/// That is the test used here — a step no edge carries, walked twice by the one
/// face — and it is why this cannot fire on a boundary shared with a neighbour,
/// which is walked once.
fn materialise_seam_steps(edges: &mut Vec<Edge>, faces: &mut [Face]) {
    let mut backed: HashSet<(usize, usize)> = HashSet::new();
    for e in edges.iter() {
        for w in e.vertices.windows(2) {
            backed.insert((w[0], w[1]));
            backed.insert((w[1], w[0]));
        }
    }
    for face in faces.iter_mut() {
        let Some(loops) = face.loops.as_ref() else {
            continue;
        };
        // Not where the surface runs to a point. A ring that steps from a
        // vertex to itself is crossing a pole, and a seam that ends at one is
        // pole-to-pole: AP203 can write it, but this crate's reader cannot place
        // the two passes of it, so backing it here costs a round trip that
        // works. The gain measured is on swept faces, which have no pole.
        let poled = loops.iter().any(|r| {
            let n = r.vertices.len();
            n >= 3 && (0..n).any(|i| r.vertices[i] == r.vertices[(i + 1) % n])
        });
        if poled {
            continue;
        }
        let mut want: HashMap<(usize, usize), usize> = HashMap::new();
        for ring in loops {
            let n = ring.vertices.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                let (a, b) = (ring.vertices[i], ring.vertices[(i + 1) % n]);
                if a == b || backed.contains(&(a, b)) {
                    continue;
                }
                *want.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        for ((a, b), walks) in want {
            if walks != 2 {
                continue;
            }
            let id = edges.len();
            edges.push(Edge {
                surfaces: (face.surface, face.surface),
                vertices: vec![a, b],
                closed: false,
            });
            // Twice, because the face walks it twice: once up each side. An
            // edge named once by one face is an edge with a free side.
            face.edges.push(id);
            face.edges.push(id);
            backed.insert((a, b));
            backed.insert((b, a));
        }
    }
}

fn materialise_seams(edges: &mut Vec<Edge>, faces: &mut [Face]) {
    for face in faces.iter_mut() {
        let Some(loops) = face.loops.as_ref() else {
            continue;
        };
        let named: HashSet<usize> = face
            .edges
            .iter()
            .filter_map(|&e| edges.get(e))
            .flat_map(|e| e.vertices.iter().copied())
            .collect();

        let mut found: Vec<Vec<usize>> = Vec::new();
        for ring in loops {
            let n = ring.vertices.len();
            if n < 3 {
                continue;
            }
            // Naming a vertex *twice* is what a seam really is — the ring passes
            // it once at each end of the parameter domain — and switching this
            // test to that generalises the rule to a complement face, whose
            // outline is off the edges entirely and so has no on/off boundary to
            // start a run from.
            //
            // It works, and it is measurably better here: two balls cut apart
            // come back with three edges instead of one, both faces seam-backed,
            // nothing off-edge, and still a valid solid. What it does not do is
            // anything for the case that wanted it — cutting that result with a
            // bore still declines `NeedsArrangement`, so that failure is not
            // about a boundary with no edge behind it. And it still costs the
            // STEP round trip two tests, because the reader cannot place the two
            // passes of a pole-to-pole seam. Structure alone is not enough to
            // pay for a regression.
            let off: Vec<bool> = ring.vertices.iter().map(|v| !named.contains(v)).collect();
            // A ring wholly on the edges has no seam in it; one wholly off them
            // is a parameter outline, which is not a curve on the solid at all.
            let Some(start) = (0..n).find(|&i| off[i] && !off[(i + n - 1) % n]) else {
                continue;
            };
            // A run has to end where the boundary meets something else, and
            // that is either an edge already there or a place the surface
            // degenerates. Where the neighbour is on an edge the run already
            // ends on it — the junction is named twice, once at each end of the
            // domain, so it is inside the run. Where the neighbour is on no
            // edge it is a pole, named once because every parameter there is
            // the same point, and the run has to reach out and take it or the
            // ring keeps a step no edge covers.
            let mut runs: Vec<Vec<usize>> = Vec::new();
            let mut cur: Vec<usize> = Vec::new();
            let mut i = start;
            for _ in 0..n {
                if off[i] {
                    if cur.is_empty() {
                        cur.push(ring.vertices[(i + n - 1) % n]);
                    }
                    cur.push(ring.vertices[i]);
                } else if !cur.is_empty() {
                    cur.push(ring.vertices[i]);
                    runs.push(std::mem::take(&mut cur));
                }
                i = (i + 1) % n;
            }
            if !cur.is_empty() {
                cur.push(ring.vertices[i]);
                runs.push(cur);
            }

            let mut used = vec![false; runs.len()];
            for a in 0..runs.len() {
                if used[a] || runs[a].len() < 2 {
                    continue;
                }
                for b in (a + 1)..runs.len() {
                    if used[b] {
                        continue;
                    }
                    if runs[b].iter().rev().eq(runs[a].iter()) {
                        used[a] = true;
                        used[b] = true;
                        found.push(runs[a].clone());
                        break;
                    }
                }
            }
        }

        for run in found {
            let e = edges.len();
            edges.push(Edge {
                surfaces: (face.surface, face.surface),
                vertices: run,
                closed: false,
            });
            // Twice: the boundary walks it once in each direction.
            face.edges.push(e);
            face.edges.push(e);
        }
    }
}

/// A swept face's parameter rectangle as a ring, sampled to hold `tolerance`.
///
/// The corners alone will not do. A side at constant `u` on a torus is a minor
/// circle, and two points do not describe it; more importantly the refinement
/// that follows may not split *boundary* edges, so whatever this produces is
/// what the face's outline stays. Both sides of a wrapping seam are sampled the
/// same way and land on the same points, which is what welds them.
fn parameter_outline(
    face: &Face,
    surface: &Surface,
    tolerance: f64,
    carried: &[Carried],
    seam_cuts: &[(f64, usize)],
) -> TrimLoop {
    let (u0, u1) = face.u_range;
    let (v0, v1) = face.v_range;
    let period = u1 - u0;

    // A rim the face already has *is* that side of the rectangle, and it is
    // shared with whatever lies beyond it. Sampling our own copy instead would
    // put the two a fraction apart, which is exactly how a seam opens.
    let rim_at = |v: f64, forward: bool| -> Option<(Vec<[f64; 2]>, Vec<usize>)> {
        for (_, _, pts, _, names) in carried {
            let mut uv: Vec<[f64; 2]> = Vec::with_capacity(pts.len());
            // The rim's own vertices, in the edge's order, one per point. The
            // outline used to drop these and let the result work out afterwards
            // which vertex each point had been — by position, and a point two
            // faces round differently becomes two vertices in one place.
            let mut ids: Vec<usize> = Vec::with_capacity(pts.len());
            let mut ok = true;
            for (k, p) in pts.iter().enumerate() {
                match surface.invert(*p) {
                    Some((u, w)) if (w - v).abs() <= (v1 - v0).abs() * 1e-6 => {
                        uv.push([u0 + (u - u0).rem_euclid(period), w]);
                        ids.push(names.get(k).copied().unwrap_or(usize::MAX));
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok || uv.len() < 3 {
                continue;
            }
            // *Rotated* into the rectangle, never sorted.
            //
            // The rim is shared with the face beyond it, which walks it in the
            // order the edge stores. Sorting by parameter gives the same points
            // joined by different edges, and a seam is held by edges — the same
            // mistake `seam_chord` made, in the same place, for the same reason.
            if uv.len() > 1 && dist2(uv[0], uv[uv.len() - 1]) <= (period * 1e-9).powi(2) {
                uv.pop(); // a closed rim repeats its first point
                ids.pop();
            }
            let n = uv.len();
            if n < 3 {
                continue;
            }
            let (mut cut, mut worst, mut rising) = (0usize, 0.0, true);
            for i in 0..n {
                let step = uv[(i + 1) % n][0] - uv[i][0];
                if step.abs() > worst {
                    worst = step.abs();
                    cut = (i + 1) % n;
                    rising = step < 0.0;
                }
            }
            let dir: isize = if rising { 1 } else { -1 };
            let cut = if rising { cut } else { (cut + n - 1) % n };
            let mut rotated = Vec::with_capacity(n);
            let mut rotated_ids = Vec::with_capacity(n);
            for k in 0..n {
                let i = ((cut as isize + dir * k as isize).rem_euclid(n as isize)) as usize;
                rotated.push(uv[i]);
                rotated_ids.push(ids[i]);
            }
            uv = rotated;
            ids = rotated_ids;
            // Folding puts the rim in `[u₀, u₁)`, so it stops one point short of
            // the far edge and the rectangle's corner becomes a diagonal — with
            // nothing for a chord landing on that edge to land on. The two edges
            // are the same place on the surface, so the point at `u₀` closes the
            // rim onto `u₁` as well.
            if (uv[0][0] - u0).abs() <= period * 1e-9 {
                uv.push([u1, uv[0][1]]);
                // The far corner is the near corner: same place, same vertex.
                // Naming it makes that true by construction, where before the
                // two were welded afterwards and one could be lost.
                ids.push(ids[0]);
            }
            if !forward {
                uv.reverse();
                ids.reverse();
            }
            return Some((uv, ids));
        }
        None
    };
    let along = |a: [f64; 2], b: [f64; 2]| -> Vec<[f64; 2]> {
        let n = super::body::adaptive_steps(
            tolerance,
            |s, t| {
                let at = |x: f64| surface.point(a[0] + (b[0] - a[0]) * x, a[1] + (b[1] - a[1]) * x);
                let m = 0.5 * (s + t);
                v3::dist(at(m), v3::scale(v3::add(at(s), at(t)), 0.5))
            },
            0.0,
            1.0,
        );
        (0..n)
            .map(|i| {
                let x = i as f64 / n as f64;
                [a[0] + (b[0] - a[0]) * x, a[1] + (b[1] - a[1]) * x]
            })
            .collect()
    };
    // A seam edge broken where something lands on it.
    //
    // `along` samples for *curvature*, and an edge at constant `u` on a cylinder
    // is straight, so it comes back as a single point — one corner, and nothing
    // between. A chord cut from a wrapping ring then meets that edge partway
    // along, at a place the outline does not name, and the walk has nothing to
    // attach it to. Measured on a drill's wall cut by a ball it passes through:
    // the near edge held `v = 60` alone while two chords arrived at 27.22 and
    // 32.78, and `subdivide` refused the face.
    //
    // The points are the cut, not a refinement of it: the edge is straight and
    // stays straight, it just now says where it is met.
    // The seam, drawn once.
    //
    // Both sides of a cut-open face are the same curve, and drawing them
    // separately made them disagree by construction: `along` emits `n` points
    // and leaves out its endpoint, so the run from `v0` to `v1` holds `v0` and
    // the run back holds `v1`, and the two are offset by a whole sample. Every
    // sphere outline measured had it — 64 points against 64, never mirror-equal,
    // and one pair 128 against 129.
    //
    // That offset used to be repaired afterwards, one point at a time, by a pass
    // called `close_seam_runs` — 97 lines whose whole job it was. Built from one
    // list there is nothing to repair, and the pass is gone.
    let seam_run = |a: [f64; 2], b: [f64; 2]| -> Vec<([f64; 2], usize)> {
        let mut stops: Vec<(f64, usize)> = seam_cuts
            .iter()
            .copied()
            .filter(|(v, _)| (v - a[1]) * (v - b[1]) < 0.0)
            .collect();
        stops.sort_by(|x, y| {
            (x.0 - a[1])
                .abs()
                .partial_cmp(&(y.0 - a[1]).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut out: Vec<([f64; 2], usize)> = Vec::new();
        let mut from = a;
        for (v, name) in stops {
            let to = [a[0], v];
            // The run up to the stop is drawn here and has no names; the stop
            // itself is a vertex `insert_seam_vertex` already made, and saying
            // so is the difference between the chord landing on it and the two
            // being welded together afterwards, or not.
            out.extend(along(from, to).into_iter().map(|c| (c, usize::MAX)));
            if let Some(last) = out.last_mut() {
                if (last.0[1] - v).abs() <= f64::EPSILON {
                    last.1 = name;
                }
            }
            from = to;
        }
        out.extend(along(from, b).into_iter().map(|c| (c, usize::MAX)));
        // Inclusive: the far corner belongs to the list, and which side keeps it
        // is decided where the list is used.
        out.push((b, usize::MAX));
        out
    };
    // Assembled as points *with names*. A side that comes from a rim knows the
    // vertex of every point on it; a side this function draws itself has none
    // yet, and says so.
    let mut pts: Vec<([f64; 2], usize)> = Vec::new();
    let unnamed = |v: Vec<[f64; 2]>| -> Vec<([f64; 2], usize)> {
        v.into_iter().map(|c| (c, usize::MAX)).collect()
    };
    match rim_at(v0, true) {
        Some((r, ids)) => pts.extend(r.into_iter().zip(ids)),
        None => pts.extend(unnamed(along([u0, v0], [u1, v0]))),
    }
    // One side keeps every point but its far corner; the other is the same list
    // reversed, keeping every point but *its* far corner. Their interiors are
    // then identical values rather than two samplings that have to agree.
    // Whether a `v` edge of the rectangle is a point rather than a line: a
    // sphere's pole, a cone's apex. The whole edge maps to one place, so it is
    // not a side the ring travels along — it is the single corner where the two
    // seam runs meet, and it belongs to both of them.
    let degenerate = |v: f64| -> bool {
        let at = surface.point(u0, v);
        v3::dist(surface.point(u0 + period * 0.5, v), at) <= tolerance * 0.25
            && v3::dist(surface.point(u0 + period * 0.25, v), at) <= tolerance * 0.25
    };
    let seam = seam_run([u1, v0], [u1, v1]);
    {
        let forward = seam.clone();
        // Both corners stay on both sides. A corner of the rectangle is one
        // place with two parameters — `(u0, v)` and `(u1, v)` are the same
        // point — so neither side owns it and neither may drop it. Dropping one
        // left each run ending a sample short of where the other ended, which
        // is the asymmetry the old repair pass spent 97 lines undoing: 46 of its
        // 47 firings were a sphere's pole, the 47th a bore's rim on a sphere.
        let _ = degenerate(v1);
        pts.extend(forward);
    }
    match rim_at(v1, false) {
        Some((r, ids)) => pts.extend(r.into_iter().zip(ids)),
        None => pts.extend(unnamed(along([u1, v1], [u0, v1]))),
    }
    {
        let back: Vec<([f64; 2], usize)> =
            seam.iter().rev().map(|(c, n)| ([u0, c[1]], *n)).collect();
        let _ = degenerate(v0);
        pts.extend(back);
    }

    // Deduplicated by where the points *land*, not by their parameters.
    //
    // A seam point can fall on a rim vertex already there. Their parameters then
    // differ by a hair while the surface puts them in the same place, so they
    // weld to one vertex — and the loop names that vertex twice in a row. The
    // ear clip is entitled to make anything of a zero-length edge, and what it
    // made of this one was a triangulation covering barely half the rim.
    //
    // Where one of a welded pair is named and the other is not, the name is what
    // survives: that is the whole point of carrying them.
    let same = |a: [f64; 2], b: [f64; 2]| {
        // Never the two sides of the seam. They are one place with two
        // parameters — that is what cutting a closed surface open means — and a
        // test by position folds them together and drops one. At a pole this is
        // the whole difficulty: both seam runs end there, and the ring needs to
        // say so twice, once in each side's parameters.
        if (a[0] - b[0]).abs() > period * 0.5 {
            return false;
        }
        v3::dist(surface.point(a[0], a[1]), surface.point(b[0], b[1])) <= tolerance * 0.25
    };
    let mut out: Vec<([f64; 2], usize)> = Vec::with_capacity(pts.len());
    for (c, id) in pts {
        match out.last_mut() {
            Some((prev, kept)) if same(*prev, c) => {
                if *kept == usize::MAX {
                    *kept = id;
                }
            }
            _ => out.push((c, id)),
        }
    }
    if out.len() > 2 && same(out[0].0, out[out.len() - 1].0) {
        let (_, id) = out.pop().expect("checked non-empty");
        if out[0].1 == usize::MAX {
            out[0].1 = id;
        }
    }

    let uv: Vec<[f64; 2]> = out.iter().map(|(c, _)| *c).collect();
    let vertices: Vec<usize> = out.iter().map(|(_, v)| *v).collect();
    let area = ring_area(&uv);
    TrimLoop { vertices, uv, area }
}

/// Sample an open stretch of a curve, finely enough to meet `tolerance`.
fn sample_open_curve(curve: &Curve3d, t0: f64, t1: f64, tolerance: f64) -> Vec<V3> {
    if let Curve3d::Sampled { points, .. } = curve {
        // Keep the samples that fall inside, with the exact ends added.
        let mut out = vec![curve.point(t0)];
        for (i, p) in points.iter().enumerate() {
            if (i as f64) > t0 + 1e-9 && (i as f64) < t1 - 1e-9 {
                out.push(*p);
            }
        }
        out.push(curve.point(t1));
        return out;
    }
    let radius = match curve {
        Curve3d::Circle { radius, .. } => *radius,
        Curve3d::Ellipse { a, .. } => *a,
        // A line is straight: its ends describe it exactly.
        _ => return vec![curve.point(t0), curve.point(t1)],
    };
    let per = 2.0
        * (1.0 - (tolerance / radius.max(1e-12)).min(1.0))
            .acos()
            .max(1e-6);
    let n = (((t1 - t0).abs() / per).ceil() as usize).clamp(2, 512);
    (0..=n)
        .map(|i| curve.point(t0 + (t1 - t0) * i as f64 / n as f64))
        .collect()
}

/// Sample a closed intersection curve into this surface's parameters.
///
/// `None` for a line — an unbounded curve does not close on a face, so the split
/// it makes is a subdivision rather than a hole.
fn sample_closed_curve(
    curve: &Curve3d,
    surface: &Surface,
    tolerance: f64,
) -> Option<Vec<[f64; 2]>> {
    use std::f64::consts::TAU;
    // A traced curve already *is* its samples, spaced to hold the tolerance it
    // was traced at. Re-sampling it against a formula it does not have would
    // only lose accuracy.
    if let Curve3d::Sampled { points, closed } = curve {
        if !*closed || points.len() < 3 {
            return None;
        }
        let (periodic_u, periodic_v) = surface.periodic();
        let mut uv = Vec::with_capacity(points.len());
        let mut anchor: Option<[f64; 2]> = None;
        for p in points {
            let (mut u, mut v) = surface.invert(*p)?;
            if let Some(a) = anchor {
                if periodic_u {
                    u = unwrap_near(u, a[0]);
                }
                if periodic_v {
                    v = unwrap_near(v, a[1]);
                }
            }
            anchor = Some([u, v]);
            uv.push([u, v]);
        }
        return Some(uv);
    }

    let radius = match curve {
        Curve3d::Circle { radius, .. } => *radius,
        Curve3d::Ellipse { a, .. } => *a,
        _ => return None,
    };
    if !radius.is_finite() || radius <= 0.0 {
        return None;
    }
    // Enough segments that the chord stays within tolerance.
    let n = (((TAU) / (2.0 * (1.0 - (tolerance / radius).min(1.0)).acos().max(1e-6))).ceil()
        as usize)
        .clamp(12, 512);

    let mut uv = Vec::with_capacity(n);
    let mut anchor: Option<[f64; 2]> = None;
    let (periodic_u, periodic_v) = surface.periodic();
    for i in 0..n {
        let p = curve.point(TAU * i as f64 / n as f64);
        let (mut u, mut v) = surface.invert(p)?;
        if let Some(a) = anchor {
            if periodic_u {
                u = unwrap_near(u, a[0]);
            }
            if periodic_v {
                v = unwrap_near(v, a[1]);
            }
        }
        anchor = Some([u, v]);
        uv.push([u, v]);
    }
    Some(uv)
}

/// Pull a vertex two faces share onto both of their surfaces.
///
/// A crossing spliced into a rim is computed on one surface and left there, so
/// it sits off the curve the rim actually is — measured at 2.6e-3 on a 1e-3
/// model, with the two faces that own it disagreeing by 2.8e-3 about where it
/// is. Nothing downstream sees that today, because a fill takes a ring vertex's
/// position from the body vertex rather than from its own `uv`; but every change
/// to how a curve is sampled has failed by prying that gap open, each face
/// minting its own copy of a point that is on neither curve.
///
/// So put the point where it belongs: on both surfaces, by the same Newton step
/// the marcher uses, and recompute each ring's `uv` from the result. Only a
/// vertex claimed by exactly two surfaces is touched, only a move within a few
/// tolerances is taken, and only when it lands nearer both than it started.
fn settle_shared_vertices(
    surfaces: &[Surface],
    vertices: &mut [V3],
    faces: &mut [Face],
    tolerance: f64,
) {
    let mut claims: HashMap<usize, Vec<usize>> = HashMap::new();
    for f in faces.iter() {
        let Some(ls) = f.loops.as_ref() else {
            continue;
        };
        for l in ls {
            for &v in &l.vertices {
                let e = claims.entry(v).or_default();
                if !e.contains(&f.surface) {
                    e.push(f.surface);
                }
            }
        }
    }
    let mut moved: HashMap<usize, V3> = HashMap::new();
    for (v, ss) in &claims {
        if ss.len() != 2 {
            continue;
        }
        let p = vertices[*v];
        let (a, b) = (&surfaces[ss[0]], &surfaces[ss[1]]);
        let was = a.distance(p).abs().max(b.distance(p).abs());
        if was <= 1e-9 {
            continue;
        }
        let Some(q) = crate::brep::intersect::settle(a, b, p, tolerance) else {
            continue;
        };
        if v3::dist(p, q) > tolerance * 4.0 {
            continue;
        }
        if a.distance(q).abs().max(b.distance(q).abs()) >= was {
            continue;
        }
        moved.insert(*v, q);
    }
    if moved.is_empty() {
        return;
    }
    for (v, q) in &moved {
        vertices[*v] = *q;
    }
    for f in faces.iter_mut() {
        let s = &surfaces[f.surface];
        let (pu, pv) = s.periodic();
        let Some(ls) = f.loops.as_mut() else {
            continue;
        };
        for l in ls {
            for k in 0..l.vertices.len() {
                let Some(q) = moved.get(&l.vertices[k]) else {
                    continue;
                };
                let Some((u, w)) = s.invert(*q) else {
                    continue;
                };
                let old = l.uv[k];
                l.uv[k] = [
                    if pu { unwrap_near(u, old[0]) } else { u },
                    if pv { unwrap_near(w, old[1]) } else { w },
                ];
            }
        }
    }
}

/// Grow `lo`/`hi` to cover a body.
///
/// Vertices first, because they are exact and they are what the corners of a
/// solid are. But a body need not have any: a whole torus is one face with no
/// edges and no vertices, and so is a whole sphere. Reading the extent off the
/// vertices alone left such a body contributing *nothing*, so two of them gave
/// the marcher an inverted box, it found no intersection curve, and the boolean
/// concluded the two do not meet — returning an untouched torus, valid and
/// plausible and wrong, from a difference that should have cut a bite out of it.
/// That is the one outcome this crate is not allowed to produce.
///
/// So when a body has no vertices, take its extent from the faces themselves,
/// sampled across their parameter ranges. A bounding box wants to be generous,
/// not exact, and the caller pads it further.
fn extent_of(b: &Body, lo: &mut V3, hi: &mut V3) {
    let mut add = |p: V3| {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    };
    if !b.vertices().is_empty() {
        for p in b.vertices() {
            add(*p);
        }
        return;
    }
    const N: usize = 8;
    for f in b.faces() {
        let s = &b.surfaces()[f.surface];
        for i in 0..=N {
            let u = f.u_range.0 + (f.u_range.1 - f.u_range.0) * i as f64 / N as f64;
            for j in 0..=N {
                let v = f.v_range.0 + (f.v_range.1 - f.v_range.0) * j as f64 / N as f64;
                add(s.point(u, v));
            }
        }
    }
}

fn unwrap_near(x: f64, anchor: f64) -> f64 {
    use std::f64::consts::TAU;
    x - TAU * ((x - anchor) / TAU).round()
}

/// Wind a loop so it reads as outer (`want_outer`) or as a hole.
fn oriented(mut l: TrimLoop, want_outer: bool) -> TrimLoop {
    let is_outer = l.area > 0.0;
    if is_outer != want_outer {
        l.area = -l.area;
        l.uv.reverse();
        l.vertices.reverse();
    }
    l
}

/// A point strictly inside a ring, from the fan of triangles about its first
/// vertex — the same construction the classification sampler uses.
fn interior_point(ring: &[[f64; 2]]) -> Option<[f64; 2]> {
    let margin = ring_margin(ring);
    for i in 1..ring.len().saturating_sub(1) {
        let c = [
            (ring[0][0] + ring[i][0] + ring[i + 1][0]) / 3.0,
            (ring[0][1] + ring[i][1] + ring[i + 1][1]) / 3.0,
        ];
        if deep_in_ring(ring, c, margin) {
            return Some(c);
        }
    }
    None
}

/// Do two segments cross at a point interior to both?
fn segments_properly_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let side = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| {
        (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    };
    let (d1, d2) = (side(c, d, a), side(c, d, b));
    let (d3, d4) = (side(a, b, c), side(a, b, d));
    ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
}

fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

/// Are these the same ring, to within a hair of its own size?
fn same_ring(a: &[[f64; 2]], b: &[[f64; 2]]) -> bool {
    let near = ring_margin(a);
    let on = |ring: &[[f64; 2]], p: [f64; 2]| {
        let n = ring.len();
        (0..n).any(|i| {
            let (x, y) = (ring[i], ring[(i + 1) % n]);
            let (dx, dy) = (y[0] - x[0], y[1] - x[1]);
            let len2 = dx * dx + dy * dy;
            let t = if len2 <= f64::MIN_POSITIVE {
                0.0
            } else {
                (((p[0] - x[0]) * dx + (p[1] - x[1]) * dy) / len2).clamp(0.0, 1.0)
            };
            (p[0] - (x[0] + dx * t)).hypot(p[1] - (x[1] + dy * t)) <= near
        })
    };
    b.iter().all(|p| on(a, *p)) && a.iter().all(|p| on(b, *p))
}

/// Is `p` inside `ring` by more than `margin`?
///
/// Merely "inside" is not a usable answer for a point *on* the boundary, and a
/// curve lying along one is not rare — two solids sharing a wall also meet
/// edge-on along every adjoining face, and their intersection there runs exactly
/// down the boundary of both. Winding number decides such a point by whichever
/// way the arithmetic falls, so a grazing curve reads as a scatter of in and out
/// and every consumer of that answer gets confused.
///
/// A grazing curve does not *cut* the face — the boundary already describes it —
/// so the useful question is whether anything is strictly inside.
fn deep_in_ring(ring: &[[f64; 2]], p: [f64; 2], margin: f64) -> bool {
    if !point_in_ring(ring, p) {
        return false;
    }
    let n = ring.len();
    (0..n).all(|i| {
        let (a, b) = (ring[i], ring[(i + 1) % n]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len2 = dx * dx + dy * dy;
        let t = if len2 <= f64::MIN_POSITIVE {
            0.0
        } else {
            (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
        };
        (p[0] - (a[0] + dx * t)).hypot(p[1] - (a[1] + dy * t)) > margin
    })
}

/// How far from a boundary a point must be to count as inside it.
fn ring_margin(ring: &[[f64; 2]]) -> f64 {
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for p in ring {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-12) * 1e-9
}

fn ring_area(uv: &[[f64; 2]]) -> f64 {
    let n = uv.len();
    let mut a = 0.0;
    for i in 0..n {
        let (p, q) = (uv[i], uv[(i + 1) % n]);
        a += p[0] * q[1] - q[0] * p[1];
    }
    a * 0.5
}

/// Even-odd containment in a closed ring.
fn point_in_ring(ring: &[[f64; 2]], p: [f64; 2]) -> bool {
    let n = ring.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (ring[i], ring[j]);
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let t = (p[1] - a[1]) / (b[1] - a[1]);
            if p[0] < a[0] + t * (b[0] - a[0]) {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Ray parity against a closed triangle mesh.
///
/// The ray direction is irrational on purpose: an axis-aligned one hits shared
/// edges and vertices of a tessellated quadric constantly, and each such hit is
/// a coin flip on the parity.
fn point_in_mesh(tris: &[[V3; 3]], p: V3) -> bool {
    let dir = [0.577_215_66, 0.413_255_11, 0.700_113_47];
    let mut hits = 0usize;
    for t in tris {
        if let Some(d) = ray_triangle(p, dir, t) {
            if d > 1e-9 {
                hits += 1;
            }
        }
    }
    hits % 2 == 1
}

/// Möller–Trumbore.
fn ray_triangle(origin: V3, dir: V3, t: &[V3; 3]) -> Option<f64> {
    let e1 = v3::sub(t[1], t[0]);
    let e2 = v3::sub(t[2], t[0]);
    let h = v3::cross(dir, e2);
    let a = v3::dot(e1, h);
    if a.abs() < 1e-14 {
        return None;
    }
    let f = 1.0 / a;
    let s = v3::sub(origin, t[0]);
    let u = f * v3::dot(s, h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = v3::cross(s, e1);
    let v = f * v3::dot(dir, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    Some(f * v3::dot(e2, q))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Signed volume of a closed mesh, by the divergence theorem.
    fn volume(body: &Body, tolerance: f64) -> f64 {
        let mut b = body.clone();
        b.refine_edges(tolerance);
        let (mesh, report) = b.tessellate(tolerance);
        assert!(report.is_closed(), "{} open edges", report.boundary_edges);
        let pos = &mesh.get_attribute("position").unwrap().array;
        let idx = mesh.index.as_ref().unwrap();
        let v = |i: u32| -> V3 {
            let o = i as usize * 3;
            [pos[o] as f64, pos[o + 1] as f64, pos[o + 2] as f64]
        };
        idx.chunks_exact(3)
            .map(|t| {
                let (a, b, c) = (v(t[0]), v(t[1]), v(t[2]));
                v3::dot(a, v3::cross(b, c)) / 6.0
            })
            .sum::<f64>()
            .abs()
    }

    fn plate_and_drill() -> (Body, Body) {
        (
            Body::cuboid([10.0, 8.0, 2.0]),
            Body::cylinder([0.0, 0.0, -3.0], [0.0, 0.0, 1.0], 2.0, 6.0),
        )
    }

    #[test]
    fn a_difference_produces_the_solid_it_should() {
        // A plate with a bore, *computed* — the same shape `Body::plate_with_hole`
        // authors, arrived at by intersecting surfaces instead of by construction.
        let (plate, drill) = plate_and_drill();
        let result = plate
            .boolean(&drill, BooleanOp::Difference, 1e-3)
            .expect("every pair has a closed form");

        assert_eq!(result.faces().len(), 7, "six walls and a bore");
        // The two faces the bore passes through are the planar ones bounded by
        // a closed edge — the seam circle. Asserted on the *edges* rather than
        // on trim loops, because a boundary an edge already describes does not
        // get a loop as well: the loop would be a fixed polyline beside an edge
        // that refines, and the two would part company.
        let holed = result
            .faces()
            .iter()
            .filter(|f| result.surfaces()[f.surface].kind() == "plane")
            .filter(|f| f.edges.iter().any(|&e| result.edges()[e].closed))
            .count();
        assert_eq!(holed, 2, "the two faces the bore passes through");

        // 10 × 8 × 2, less a bore of radius 2 through a depth of 2.
        let expected = 160.0 - std::f64::consts::PI * 4.0 * 2.0;
        let got = volume(&result, 1e-3);
        assert!(
            (got - expected).abs() / expected < 0.01,
            "volume {got}, expected {expected}"
        );
    }

    #[test]
    fn a_difference_is_watertight() {
        // The property the shared-curve sampling exists for. Each face samples
        // the seam from the *same* vertex list, so the result closes by
        // construction — sampling per face leaves a crack along every seam.
        let (plate, drill) = plate_and_drill();
        let mut result = plate.boolean(&drill, BooleanOp::Difference, 1e-3).unwrap();
        // The plate's own twelve edges, carried through, plus one shared edge
        // per seam. Every one is used by exactly two faces — that is what makes
        // the result close rather than merely line up.
        assert_eq!(result.edges().len(), 14);
        for i in 0..result.edges().len() {
            let users = result
                .faces()
                .iter()
                .filter(|f| f.edges.contains(&i))
                .count();
            assert_eq!(users, 2, "edge {i} used by {users} faces");
        }
        result.refine_edges(1e-3);
        let (_, report) = result.tessellate(1e-3);
        assert_eq!(report.carried_through, 0);
        assert!(report.is_closed(), "{} open edges", report.boundary_edges);
    }

    #[test]
    fn union_and_intersection_give_the_volumes_they_should() {
        let (plate, drill) = plate_and_drill();
        let bore = std::f64::consts::PI * 4.0 * 2.0; // the overlap
        let rod = std::f64::consts::PI * 4.0 * 6.0;

        let inter = plate
            .boolean(&drill, BooleanOp::Intersection, 1e-3)
            .expect("closed form");
        let got = volume(&inter, 1e-3);
        assert!(
            (got - bore).abs() / bore < 0.01,
            "intersection {got}, expected {bore}"
        );

        let union = plate
            .boolean(&drill, BooleanOp::Union, 1e-3)
            .expect("closed form");
        let expected = 160.0 + rod - bore;
        let got = volume(&union, 1e-3);
        assert!(
            (got - expected).abs() / expected < 0.01,
            "union {got}, expected {expected}"
        );
    }

    #[test]
    fn the_three_operations_are_consistent_with_each_other() {
        // |A| + |B| = |A ∪ B| + |A ∩ B|, which holds whatever the shapes are and
        // does not depend on any hand-computed expectation.
        let (plate, drill) = plate_and_drill();
        let a = volume(&plate, 1e-3);
        let b = volume(&drill, 1e-3);
        let u = volume(
            &plate.boolean(&drill, BooleanOp::Union, 1e-3).unwrap(),
            1e-3,
        );
        let i = volume(
            &plate
                .boolean(&drill, BooleanOp::Intersection, 1e-3)
                .unwrap(),
            1e-3,
        );
        let d = volume(
            &plate.boolean(&drill, BooleanOp::Difference, 1e-3).unwrap(),
            1e-3,
        );
        assert!(
            ((a + b) - (u + i)).abs() / (a + b) < 0.01,
            "|A|+|B| = {} but |A∪B|+|A∩B| = {}",
            a + b,
            u + i
        );
        assert!(
            (d - (a - i)).abs() / a < 0.01,
            "|A−B| = {d} but |A|−|A∩B| = {}",
            a - i
        );
    }

    #[test]
    fn a_sphere_drilled_by_a_coaxial_cylinder() {
        // A curved face split by a curved one, so the seam is not a plane cut.
        let ball = Body::sphere([0.0; 3], 3.0);
        let drill = Body::cylinder([0.0, 0.0, -5.0], [0.0, 0.0, 1.0], 1.0, 10.0);
        let result = ball
            .boolean(&drill, BooleanOp::Difference, 1e-3)
            .expect("a coaxial sphere and cylinder have a closed form");
        let mut r = result.clone();
        r.refine_edges(1e-3);
        let (_, report) = r.tessellate(1e-3);
        assert_eq!(report.carried_through, 0);

        // A sphere less a coaxial bore: the exact volume of the remainder is
        // 4/3·π·(r² − a²)^{3/2}, the "napkin ring" identity.
        let h = (9.0f64 - 1.0).sqrt();
        let expected = 4.0 / 3.0 * std::f64::consts::PI * h.powi(3);
        let got = volume(&result, 1e-3);
        assert!(
            (got - expected).abs() / expected < 0.02,
            "volume {got}, expected {expected}"
        );
    }

    #[test]
    fn a_body_that_is_not_a_solid_has_no_boolean() {
        // An open tube has no inside, so "inside the other body" is not a
        // question with an answer.
        let tube = Body::cylinder([0.0; 3], [0.0, 0.0, 1.0], 1.0, 2.0);
        let mut open = tube.clone();
        open.faces_mut().truncate(1); // drop the caps
        let cube = Body::cuboid([4.0, 4.0, 4.0]);
        assert!(matches!(
            open.boolean(&cube, BooleanOp::Union, 1e-3),
            Err(Declined::NotASolid)
        ));
    }

    #[test]
    fn a_pair_without_a_closed_form_is_traced_not_fitted() {
        // A torus against a cylinder off its axis is a quartic: there is no
        // conic to name. It is *traced* — walked along `n₁ × n₂` with every
        // point driven back onto both surfaces — rather than fitted to
        // something plausible, and what matters is that the result lies on both
        // surfaces to the tolerance it claims.
        let torus = Surface::torus([0.0; 3], [0.0, 0.0, 1.0], 4.0, 1.0);
        let cyl = Surface::cylinder([4.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.5);
        let curves = super::super::intersect::march(&torus, &cyl, ([-6.0; 3], [6.0; 3]), 1e-3);
        assert!(!curves.is_empty(), "the cylinder passes through the tube");
        for c in &curves {
            let Curve3d::Sampled { points, closed } = c else {
                panic!("a quartic has no closed form to return");
            };
            assert!(closed, "the tube is entered and left");
            for p in points {
                assert!(
                    torus.distance(*p) <= 1e-6,
                    "{} off the torus",
                    torus.distance(*p)
                );
                assert!(
                    cyl.distance(*p) <= 1e-6,
                    "{} off the cylinder",
                    cyl.distance(*p)
                );
            }
        }
    }

    #[test]
    fn a_solid_with_itself() {
        // Every face coincident, every pair facing the same way. The rule says
        // one copy survives a union and an intersection, and a difference takes
        // the whole thing away — which is what these operations mean.
        let a = Body::cuboid([2.0, 2.0, 2.0]);
        let b = Body::cuboid([2.0, 2.0, 2.0]);
        assert_eq!(
            a.boolean(&b, BooleanOp::Union, 1e-3).unwrap().faces().len(),
            6
        );
        assert_eq!(
            a.boolean(&b, BooleanOp::Intersection, 1e-3)
                .unwrap()
                .faces()
                .len(),
            6
        );
        assert!(a
            .boolean(&b, BooleanOp::Difference, 1e-3)
            .unwrap()
            .faces()
            .is_empty());
    }

    #[test]
    fn ray_parity_agrees_with_the_shape_it_is_asked_about() {
        let mut body = Body::sphere([0.0; 3], 2.0);
        body.refine_edges(1e-2);
        let tris = body.solid_triangles(1e-2);
        assert!(!tris.is_empty());
        assert!(
            point_in_mesh(&tris, [0.0, 0.0, 0.0]),
            "the centre is inside"
        );
        assert!(point_in_mesh(&tris, [1.0, 0.5, -0.5]));
        assert!(!point_in_mesh(&tris, [5.0, 0.0, 0.0]), "well outside");
        assert!(!point_in_mesh(&tris, [0.0, 0.0, 3.0]));
    }

    #[test]
    fn containment_in_a_ring() {
        let square = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        assert!(point_in_ring(&square, [0.0, 0.0]));
        assert!(!point_in_ring(&square, [2.0, 0.0]));
        assert!(!point_in_ring(&square, [0.0, -3.0]));
    }
}
